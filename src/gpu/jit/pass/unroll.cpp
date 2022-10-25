/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gpu/jit/pass/unroll.hpp"

#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class unrolling_updater_t : public ir_mutator_t {
public:
    object_t _mutate(const let_t &obj) override {
        if (level_ == 0) {
            // Skip top-level let statements.
            return ir_mutator_t::_mutate(obj);
        }
        lets_.push_back(&obj);
        auto new_body = mutate(obj.body);
        if (!lets_.back()) {
            // Let was moved to the innermost loop.
            lets_.pop_back();
            return new_body;
        }
        lets_.pop_back();
        if (new_body.is_same(obj.body)) return obj;
        return let_t::make(obj.var, obj.value, new_body);
    }

    object_t _mutate(const for_t &obj) override {
        if (in_compute_loop_) level_++;
        found_loop_ = false;
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (in_compute_loop_) level_--;
        if (!found_loop_) {
            // Innermost loop, inject let statements.
            auto body = get_stmt_body(new_obj);
            for (auto it = lets_.rbegin(); it != lets_.rend(); ++it) {
                body = let_t::make((*it)->var, (*it)->value, body);
                *it = nullptr;
            }
            new_obj = replace_stmt_body(new_obj, body);
        }
        found_loop_ = true;
        return new_obj;
    }

    object_t _mutate(const stmt_group_t &obj) override {
        if (obj.label == stmt_label_t::compute_loop()) {
            in_compute_loop_ = true;
        }
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (obj.label == stmt_label_t::compute_loop()) {
            in_compute_loop_ = false;
        }
        return new_obj;
    }

private:
    bool found_loop_ = false;
    bool in_compute_loop_ = false;
    int level_ = 0;
    std::vector<const let_t *> lets_;
};

stmt_t update_loops_for_unrolling(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = unrolling_updater_t().mutate(s);
    trace_pass("update_loops_for_unrolling", ret, ir_ctx);
    return ret;
}

class loop_unroller_t : public ir_mutator_t {
public:
    loop_unroller_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    object_t _mutate(const for_t &obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto &_for = new_obj.as<for_t>();
        // No unrolling.
        if (_for.unroll == 1) return new_obj;

        ir_assert(is_const(_for.init))
                << "Can't unroll loop with non-const bound: " << _for.init;
        ir_assert(is_const(_for.bound))
                << "Can't unroll loop with non-const bound: " << _for.bound;

        auto init = to_cpp<int>(_for.init);
        auto bound = to_cpp<int>(_for.bound);

        ir_assert(_for.unroll == (bound - init))
                << "Only full loop unroll is supported.";

        stmt_t ret;
        for (int i = init; i < bound; i++) {
            auto iter_stmt = substitute(
                    _for.body, _for.var, to_expr(i, _for.var.type()));
            iter_stmt = rename_let_alloc(iter_stmt, i - init);
            ret = ret.append(iter_stmt);
        }
        return std::move(ret);
    }

private:
    stmt_t rename_let_alloc(const stmt_t &s, int idx) {
        auto lets = find_objects<let_t>(s);
        auto ret = s;
        for (auto &_let : lets) {
            auto &let = _let.as<let_t>();
            auto &var = let.var.as<var_t>();
            auto new_var = ir_ctx_.create_tmp_var(var.type, var.name);
            ret = substitute(ret, let.var, new_var);
        }
        auto allocs = find_objects<alloc_t>(s);
        for (auto &_alloc : allocs) {
            auto &alloc = _alloc.as<alloc_t>();
            auto &buf = alloc.buf.as<var_t>();
            auto new_buf = ir_ctx_.create_tmp_var(buf.type, buf.name);
            ret = substitute(ret, alloc.buf, new_buf);
        }
        return ret;
    }

    ir_context_t &ir_ctx_;
};

stmt_t unroll_loops(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = loop_unroller_t(ir_ctx).mutate(s);
    trace_pass("unroll_loops", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
