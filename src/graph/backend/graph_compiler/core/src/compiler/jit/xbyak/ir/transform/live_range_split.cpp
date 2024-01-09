/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <compiler/ir/builder.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/viewer.hpp>
#include <compiler/ir/visitor.hpp>
#include <compiler/jit/xbyak/ir/util/utils.hpp>
#include <util/any_map.hpp>

#include "live_range_split.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

struct split_analysis_data_t {
    //
    bool cross_call_ = false;
    //
    std::unordered_set<expr_c> used_vars_;
    std::unordered_set<expr_c> modified_vars_;

    static split_analysis_data_t &get(const expr_c &v) {
        return v->temp_data().get<split_analysis_data_t>();
    }

    static bool is_set(const expr_c &v) {
        return v->get_temp_data().isa<split_analysis_data_t>();
    }
};

class split_analysis_viewer_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;

    std::vector<expr_c> cur_loop_iter_;
    std::vector<std::vector<expr_c>> cur_loop_scopes_;
    std::unordered_set<expr_c> initialized_var_;
    std::unordered_map<expr_c, bool> defined_var_map_;

    bool var_cross_call(const expr_c &v) {
        auto iter = defined_var_map_.find(v);
        return (iter != defined_var_map_.end()) && iter->second;
    }

    void filter_curr_scope(std::unordered_set<expr_c> &vars) {
        for (auto it = vars.begin(); it != vars.end();) {
            bool a = defined_var_map_.find(*it) == defined_var_map_.end();
            bool b = initialized_var_.find(*it) == initialized_var_.end();
            if (a || b) {
                it = vars.erase(it);
            } else {
                ++it;
            }
        }
    }

    void filter_inner_scope(std::unordered_set<expr_c> &inner,
            const std::unordered_set<expr_c> &curr) {
        for (auto it = inner.begin(); it != inner.end();) {
            if (curr.find(*it) != curr.end()) {
                it = inner.erase(it);
            } else {
                ++it;
            }
        }
    }

    void merge_outer_scope(std::unordered_set<expr_c> &outer,
            const std::unordered_set<expr_c> &curr) {
        outer.insert(curr.begin(), curr.end());
    }

    void view(call_c v) override {
        ir_viewer_t::view(v);
        for (auto &kv : defined_var_map_) {
            kv.second = true;
        }
        if (!cur_loop_iter_.empty()) {
            split_analysis_data_t::get(cur_loop_iter_.back()).cross_call_
                    = true;
        }
    }

    void view(var_c v) override {
        ir_viewer_t::view(v);
        if (is_x86_simd(v->dtype_)) {
            if (!cur_loop_iter_.empty() && var_cross_call(v)) {
                auto &data = split_analysis_data_t::get(cur_loop_iter_.back());
                data.used_vars_.insert(v);
            }
        }
    }

    void view(define_c v) override {
        ir_viewer_t::view(v);
        if (is_x86_simd(v->var_->dtype_)) {
            defined_var_map_[v->var_] = false;
            if (cur_loop_iter_.empty() && v->init_.defined()) {
                initialized_var_.insert(v->var_);
            }
        }
    }

    void view(assign_c v) override {
        ir_viewer_t::view(v);
        if (is_x86_simd(v->var_->dtype_)) {
            if (!cur_loop_iter_.empty() && var_cross_call(v->var_)) {
                auto &data = split_analysis_data_t::get(cur_loop_iter_.back());
                data.modified_vars_.insert(v->var_);
            }
            if (cur_loop_iter_.empty()) { initialized_var_.insert(v->var_); }
        }
    }

    void view(for_loop_c v) override {
        // recored this as inner loop of outer
        if (!cur_loop_scopes_.empty()) {
            cur_loop_scopes_.back().emplace_back(v->var_);
        }
        cur_loop_scopes_.emplace_back(std::vector<expr_c>());
        cur_loop_iter_.emplace_back(v->var_);
        // build analysis data
        v->var_->temp_data() = split_analysis_data_t();
        auto &data = split_analysis_data_t::get(v->var_);

        // dispatch into loop
        ir_viewer_t::view(v);

        // if loop contains call, no spilt for inner loop vars
        if (split_analysis_data_t::get(v->var_).cross_call_) {
            data.used_vars_.clear();
            data.modified_vars_.clear();
            for (const auto &loop_var : cur_loop_iter_) {
                split_analysis_data_t::get(loop_var).cross_call_ = true;
            }
        }
        cur_loop_iter_.pop_back();

        // filter vars to spilt when defined outside loop
        // propagate spilt vars to most outer loop when possible
        filter_curr_scope(data.used_vars_);
        filter_curr_scope(data.modified_vars_);
        if (!cur_loop_iter_.empty()) {
            auto &outer = split_analysis_data_t::get(cur_loop_iter_.back());
            merge_outer_scope(outer.used_vars_, data.used_vars_);
            merge_outer_scope(outer.modified_vars_, data.modified_vars_);
        }
        for (auto &inner_loop : cur_loop_scopes_.back()) {
            auto &inner = split_analysis_data_t::get(inner_loop);
            filter_inner_scope(inner.used_vars_, data.used_vars_);
            filter_inner_scope(inner.modified_vars_, data.modified_vars_);
        }
        cur_loop_scopes_.pop_back();
    }
};

// var a
// a = ...
// call(...)
// for(...) {
//    a = ...
// }
//
// spilt live range of a:
//
// var a
// a = ...
// call(...)
// var a_1 = a
// for(...) {
//    a_1 = ...
// }

class live_range_splitter_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    size_t var_counter_ = 0;
    std::unordered_map<expr_c, expr_c> var_replace_map_;

    expr_c visit(var_c v) override {
        auto iter = var_replace_map_.find(v);
        return (iter == var_replace_map_.end()) ? v : iter->second;
    }

    stmt_c visit(for_loop_c v) override {
        auto &data = split_analysis_data_t::get(v->var_);
        if (data.used_vars_.empty()) {
            return ir_visitor_t::visit(std::move(v));
        } else {
            std::vector<stmt> seq;
            // insert define before loop
            for (const auto &x : data.used_vars_) {
                auto new_v = x->remake().static_as<var>();
                new_v->name_ += std::string("_spilt_")
                        + std::to_string(var_counter_++);
                var_replace_map_[x] = new_v;
                seq.emplace_back(builder::make_var_tensor_def_unattached(
                        new_v, linkage::local, x));
            }
            // dispatch loop
            seq.emplace_back(ir_visitor_t::visit(std::move(v)).remove_const());
            // insert assign after loop
            for (const auto &x : data.modified_vars_) {
                seq.emplace_back(builder::make_assign_unattached(
                        x, var_replace_map_[x]));
            }
            // remove replacment vars
            for (const auto &x : data.used_vars_) {
                var_replace_map_.erase(x);
            }

            return make_stmt<stmts_node_t>(std::move(seq));
        }
    }
};

func_c live_range_splitter_t::operator()(func_c v) {
    split_analysis_viewer_t analyzer;
    analyzer.dispatch(v);

    live_range_splitter_impl_t live_range_splitter;
    return live_range_splitter.dispatch(std::move(v));
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
