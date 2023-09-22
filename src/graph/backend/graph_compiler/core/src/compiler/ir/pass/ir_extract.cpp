/*******************************************************************************
 * Copyright 2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#include <utility>
#include "../viewer.hpp"
#include "ir_extract.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

void ir_var_extractor_t::view(var_c v) {
    ir_viewer_t::view(v);
    if (dedup_.find(v) == dedup_.end()) {
        dedup_.insert(v);
        vars_.emplace_back(v);
    }
}

void ir_var_extractor_t::operator()(const stmt_c &s) {
    dedup_.clear();
    ir_viewer_t::dispatch(s);
}

void ir_var_extractor_t::operator()(const expr_c &s) {
    dedup_.clear();
    ir_viewer_t::dispatch(s);
}

void ir_var_extractor_t::operator()(func_c s) {
    dedup_.clear();
    ir_viewer_t::dispatch(std::move(s));
}

void ir_var_capturer_t::view(define_c v) {
    if (v->var_.isa<var>()) { vars_.emplace_back(v->var_); }
    ir_viewer_t::view(v);
}

void ir_var_capturer_t::view(stmts_c v) {
    size_t prev_sz = vars_.size();
    for (auto &s : v->seq_) {
        if (terminated_) { return; }
        if (s.ptr_same(terminator_.remove_const())) {
            terminated_ = true;
            return;
        }
        ir_viewer_t::dispatch(s);
    }
    if (!terminated_) { vars_.resize(prev_sz); }
}

void ir_var_capturer_t::view(for_loop_c v) {
    if (terminated_) { return; }
    size_t prev_sz = vars_.size();
    vars_.emplace_back(v->var_);
    if (!v->body_.ptr_same(terminator_.remove_const())) {
        ir_viewer_t::dispatch(v->body_);
    } else {
        terminated_ = true;
    }
    if (!terminated_) { vars_.resize(prev_sz); }
}

void ir_var_capturer_t::view(if_else_c v) {
    if (terminated_) { return; }
    size_t prev_sz = vars_.size();
    if (v->then_case_.ptr_same(terminator_.remove_const())) {
        terminated_ = true;
        return;
    } else {
        ir_viewer_t::dispatch(v->then_case_);
    }
    if (!terminated_) {
        vars_.resize(prev_sz);
    } else
        return;

    if (v->else_case_.defined()) {
        if (v->else_case_.ptr_same(terminator_.remove_const())) {
            terminated_ = true;
            return;
        } else {
            ir_viewer_t::dispatch(v->else_case_);
            if (!terminated_) { vars_.resize(prev_sz); }
        }
    }
}

void ir_var_capturer_t::operator()(const stmt_c &v) {
    terminated_ = false;
    ir_viewer_t::dispatch(v);
}

void ir_var_capturer_t::operator()(const func_c &v) {
    terminated_ = false;

    size_t prev_sz = vars_.size();
    for (auto &p : v->params_) {
        if (p.isa<var>()) { vars_.emplace_back(p); }
    }
    if (v->body_.ptr_same(terminator_.remove_const())) {
        terminated_ = true;
        return;
    }
    ir_viewer_t::dispatch(v->body_);
    if (!terminated_) { vars_.resize(prev_sz); }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
