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

#include "gpu/jit/pass/pass.hpp"

#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class external_var_visitor_t : public scope_visitor_t {
public:
    void _visit(const var_t &obj) {
        if (!is_expr_defined(obj)) external_vars.insert(obj);
    }

    object_eq_set_t<expr_t> external_vars;
};

stmt_t inject_external_var_let(const stmt_t &_stmt, ir_context_t &ir_ctx) {
    trace_start();
    auto stmt = _stmt;
    external_var_visitor_t v;
    v.visit(stmt);

    for (auto &var : v.external_vars)
        stmt = let_t::make(var, {}, stmt);

    trace_pass("inject_external_var_let", stmt, ir_ctx);
    return stmt;
}

class spurious_send_mask_cast_remover_t : public ir_mutator_t {
public:
    object_t _mutate(const cast_t &obj) override {
        if (in_send_ && obj.is_bool_vec_u16() && obj.expr.type().is_bool())
            return mutate(obj.expr);
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const func_call_t &obj) override {
        if (!is_func_call<send_t>(obj)) return obj;

        in_send_ = true;
        auto new_obj = ir_mutator_t::_mutate(obj);
        in_send_ = false;
        return new_obj;
    }

private:
    bool in_send_ = false;
};

stmt_t remove_spurious_send_mask_cast(const stmt_t &s, ir_context_t &ir_ctx) {
    spurious_send_mask_cast_remover_t mutator;
    trace_start();
    auto ret = mutator.mutate(s);
    trace_pass("remove_spurious_send_mask_cast", ret, ir_ctx);
    return ret;
}

class store_splitter_t : public ir_mutator_t {
public:
    store_splitter_t(ngen::HW hw) : hw_(hw) {}

    object_t _mutate(const store_t &obj) override {
        int elems = obj.value.type().elems();
        int elem_size = obj.value.type().scalar().size();
        int stride = (obj.has_default_stride() ? 1 : obj.stride / elem_size);
        int store_size = elem_size * stride * elems;
        const auto grf_size = ngen::GRF::bytes(hw_);
        if (store_size <= 2 * grf_size) return ir_mutator_t::_mutate(obj);

        int step = 2 * grf_size / (stride * elem_size);
        stmt_t new_stmt;
        for (int i = 0; i < elems; i += step) {
            int cur_elems = std::min(step, elems - i);
            ir_assert(math::is_pow2(cur_elems));
            int off = i * stride * elem_size;
            auto store = store_t::make(obj.buf, obj.off + off,
                    split_expr(obj.value, i, i + cur_elems), obj.stride);
            new_stmt = new_stmt.append(store);
        }
        return std::move(new_stmt);
    }

private:
    static expr_t split_expr(const expr_t &e, int beg, int end) {
        auto *shuffle = e.as_ptr<shuffle_t>();
        if (shuffle) return shuffle_t::make(shuffle, beg, end);

        auto *binary = e.as_ptr<binary_op_t>();
        if (binary) {
            auto a = split_expr(binary->a, beg, end);
            auto b = split_expr(binary->b, beg, end);
            return binary_op_t::make(binary->op_kind, a, b);
        }
        ir_error_not_expected();
        return expr_t();
    }

    ngen::HW hw_;
};

stmt_t split_wide_stores(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = store_splitter_t(ir_ctx.hw_cfg().hw()).mutate(s);
    trace_pass("split_wide_stores", ret, ir_ctx);
    return ret;
}

class if_condition_fixer_t : public ir_mutator_t {
public:
    if_condition_fixer_t(int simd_size) : simd_size_(simd_size) {}

    object_t _mutate(const if_t &obj) override {
        auto _new_obj = ir_mutator_t::_mutate(obj);
        auto &new_obj = _new_obj.as<if_t>();
        auto cond = shuffle_t::make_broadcast(new_obj.cond, simd_size_);
        return if_t::make(cond, new_obj.body, new_obj.else_body);
    }

private:
    int simd_size_;
};

stmt_t fixup_if_conditions(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = if_condition_fixer_t(ir_ctx.hw_cfg().simd_size()).mutate(s);
    trace_pass("fixup_if_conditions", ret, ir_ctx);
    return ret;
}

stmt_t maybe_strip_prefetches(
        const stmt_t &s, ir_context_t &ir_ctx, int reserved_regs) {
    trace_start();
    int ir_usage = get_peak_grf_usage(s, ir_ctx.hw_cfg().grf_size());
    int grf_usage = ir_usage + reserved_regs;
    auto ret = s;
    //strip prefetches when they exceed available registers
    if (grf_usage > ir_ctx.hw_cfg().regs()) {
        ret = remove_stmt_group(s, stmt_label_t::prefetch());
        ir_warning() << "Dropping prefetches due to too lack of available "
                        "registers.\n";
    }
    trace_pass("maybe_strip_prefetches", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
