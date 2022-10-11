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

#include "gpu/jit/pass/hoist.hpp"

#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class hoist_exprs_mutator_t : public ir_mutator_t {
public:
    hoist_exprs_mutator_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    ~hoist_exprs_mutator_t() override { ir_assert(let_vars_.empty()); }

    object_t _mutate(const func_call_t &obj) override {
        if (!obj.func.is<send_t>()) return ir_mutator_t::_mutate(obj);

        std::vector<expr_t> new_args;
        for (auto &e : obj.args) {
            new_args.push_back(hoist_expr(e));
        }

        if (ir_utils::is_equal(new_args, obj.args)) return obj;

        return func_call_t::make(obj.func, new_args, obj.attr);
    }

    object_t _mutate(const stmt_group_t &obj) override {
        if (obj.body.is<for_t>()) {
            loops_.emplace_back(obj.body.as<for_t>().var);
            const for_t *for_obj = obj.body.as_ptr<for_t>();
            auto body = for_obj ? ir_mutator_t::_mutate(*for_obj) : for_obj;
            if (body.is_same(obj.body)) return obj;
            auto new_obj = stmt_group_t::make(obj.label, body);
            return injects_lets_and_pop_loop(new_obj);
        }
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const store_t &obj) override {
        auto value = hoist_expr(obj.value);
        if (value.is_equal(obj.value)) return obj;
        return store_t::make(obj.buf, obj.off, value, obj.stride);
    }

    object_t _mutate(const for_t &obj) override {
        loops_.emplace_back(obj.var);
        auto new_obj = ir_mutator_t::_mutate(obj);
        return injects_lets_and_pop_loop(new_obj);
    }

    object_t _mutate(const let_t &obj) override {
        bool fully_hoisted = false;
        expr_t new_value;
        bool is_const_let = is_const(obj.value) || is_shuffle_const(obj.value);
        if (is_const_let && loops_.size() > 0) {
            fully_hoisted = true;
            register_let(obj.var, obj.value);
            loops_[0].lets.push_back(let_t::make(obj.var, obj.value));
        } else {
            new_value = hoist_expr(obj.value, obj.var, &fully_hoisted);
        }
        if (fully_hoisted) return mutate(obj.body);
        register_let(obj.var, new_value);
        auto new_obj = let_t::make(
                obj.var, new_value, ir_mutator_t::mutate(obj.body));
        unregister_let(obj.var);
        return std::move(new_obj);
    }

private:
    struct loop_info_t {
        loop_info_t(const expr_t &var) : var(var) {}

        expr_t var;
        int var_count = 0;
        std::vector<stmt_t> lets;
    };

    expr_t hoist_expr(const expr_t &expr, const expr_t &expr_var = {},
            bool *fully_hoisted = nullptr) {
        if (expr.is_empty()) return expr;
        if (expr.type().is_ptr()) return expr;
        if (expr.type().is_bool()) return expr;
        if (is_const(expr) || is_shuffle_const(expr) || is_var(expr))
            return expr;

        auto hoisted_expr = hoist_expr_with_add(expr, expr_var, fully_hoisted);
        if (!hoisted_expr.is_equal(expr)) return hoisted_expr;

        // hoist_expr_with_add() doesn't handle cast so try to hoist it manually.
        auto *cast = expr.as_ptr<cast_t>();
        if (!cast) return hoisted_expr;

        auto hoisted_cast_expr = hoist_expr(cast->expr);
        if (!hoisted_cast_expr.is_equal(cast->expr)) {
            hoisted_expr = cast_t::make(
                    cast->type, hoisted_cast_expr, cast->saturate);
        }
        return hoisted_expr;
    }

    expr_t hoist_expr_with_add(const expr_t &expr, const expr_t &expr_var = {},
            bool *fully_hoisted = nullptr) {
        auto cur_expr = nary_op_canonicalize(expr);

        auto is_nary_add = [](const expr_t &e) {
            auto *nary = e.as_ptr<nary_op_t>();
            return nary && (nary->op_kind == op_kind_t::_add);
        };

        for (size_t i = 0; i < loops_.size(); i++) {
            std::vector<expr_t> invariant_args;
            std::vector<expr_t> other_args;
            std::vector<expr_t> nary_args;
            if (is_nary_add(cur_expr)) {
                nary_args = cvt_expr_to_nary_op_args(cur_expr);
            } else {
                nary_args.push_back(cur_expr);
            }
            for (auto &_a : nary_args) {
                auto a = nary_op_back_transform(_a);
                bool is_inv_arg = true;
                for (size_t j = i; j < loops_.size(); j++) {
                    if (!is_invariant(a, loops_[j].var)) is_inv_arg = false;
                }
                if (is_inv_arg) {
                    invariant_args.push_back(_a);
                } else {
                    other_args.push_back(_a);
                }
            }
            // Nothing to hoist for this loop, continue.
            if (invariant_args.empty()) continue;
            if (invariant_args.size() == 1 && is_var(invariant_args[0])
                    && !other_args.empty())
                continue;

            // Introduce new variable for the invariant sub-expression.
            auto inv_expr = nary_op_back_transform(
                    make_nary_op(op_kind_t::_add, invariant_args));
            expr_t inv_var;
            if (!expr_var.is_empty() && other_args.empty()) {
                // If nothing to hoist further, reuse the old variable and
                // return.
                inv_var = expr_var;
            } else {
                inv_var = ir_ctx_.create_tmp_var(inv_expr.type());
            }
            auto let = let_t::make(inv_var, inv_expr);
            register_let(inv_var, inv_expr);
            loops_[i].lets.push_back(let);

            if (other_args.empty()) {
                if (fully_hoisted) *fully_hoisted = true;
                return inv_var;
            }

            other_args.push_back(inv_var);
            cur_expr = make_nary_op(op_kind_t::_add, other_args);
        }
        return nary_op_back_transform(cur_expr);
    }

    stmt_t injects_lets_and_pop_loop(const stmt_t &_s) {
        stmt_t s = _s;
        // Inject let statements if any.
        auto &lets = loops_.back().lets;
        for (auto it = lets.rbegin(); it != lets.rend(); ++it) {
            auto &let = it->as<let_t>();
            s = let_t::make(let.var, let.value, s);
            unregister_let(let.var);
        }
        loops_.pop_back();
        return s;
    }

    void register_let(const expr_t &var, const expr_t &value) {
        let_vars_.insert({var, value});
    }

    void unregister_let(const expr_t &var) { let_vars_.erase(var); }

    bool is_invariant(const expr_t &e, const expr_t &var) const {
        if (contains_object(e, var)) return false;
        if (!find_objects<load_t>(e).empty()) return false;

        // Check value if this is a let variable.
        auto it = let_vars_.find(e);
        if (it != let_vars_.end()) return is_invariant(it->second, var);

        if (is_var(e)) return true;

        // Check transitive dependencies.
        auto vars = find_unique_objects<var_t>(e);
        for (auto &v : vars) {
            if (!is_invariant(v, var)) return false;
        }
        return true;
    }

    ir_context_t &ir_ctx_;
    std::vector<loop_info_t> loops_;

    object_map_t<expr_t, expr_t> let_vars_;
};

stmt_t hoist_exprs(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = hoist_exprs_mutator_t(ir_ctx).mutate(s);
    trace_pass("hoist_exprs", ret, ir_ctx);
    return ret;
}

class hoist_send_masks_mutator_t : public ir_mutator_t {
public:
    hoist_send_masks_mutator_t(
            ir_context_t &ir_ctx, const stmt_label_t &label, bool split_by_and)
        : ir_ctx_(ir_ctx), label_(label), split_by_and_(split_by_and) {}

    object_t _mutate(const for_t &obj) override {
        loop_deps_.insert(obj.var);
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const func_call_t &obj) override {
        if (!in_stmt_group || !is_func_call<send_t>(obj))
            return ir_mutator_t::_mutate(obj);

        auto &mask = send_t::arg_mask(obj);
        if (mask.is_empty()) return ir_mutator_t::_mutate(obj);

        auto new_args = obj.args;
        auto hoisted_mask = hoist_mask(mask);
        if (hoisted_mask.is_same(mask)) return ir_mutator_t::_mutate(obj);

        ir_assert(hoisted_mask.type().is_u16()) << hoisted_mask;

        send_t::arg_mask(new_args) = cast(hoisted_mask, mask.type());
        return func_call_t::make(obj.func, new_args, obj.attr);
    }

    object_t _mutate(const let_t &obj) override {
        auto value_vars = find_objects<var_t>(obj.value);
        for (auto &v : value_vars) {
            if (is_loop_dependency(v)) {
                loop_deps_.insert(obj.var);
                break;
            }
        }

        if (in_stmt_group) {
            ir_assert(!obj.value.is_empty());
            let_values_.emplace(obj.var, expand(obj.value, value_vars));
        }

        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const stmt_group_t &obj) override {
        bool is_stmt_group = (obj.label == label_);
        if (is_stmt_group) in_stmt_group = true;
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (is_stmt_group) {
            in_stmt_group = false;
            return create_mask_stmt(new_obj);
        }
        return new_obj;
    }

private:
    bool is_loop_dependency(const expr_t &v) const {
        ir_assert(is_var(v)) << v;
        return loop_deps_.count(v) != 0;
    }

    expr_t hoist_mask(const expr_t &e) {
        ir_assert(e.type().is_bool()) << e;

        if (e.type().elems() > 16) return e;
        if (is_const(e) || is_shuffle_const(e)) return e;

        // Can't hoist a mask containing loop vars.
        auto vars = find_objects<var_t>(e);
        for (auto &v : vars) {
            if (is_loop_dependency(v)) return e;
        }

        auto e_expanded = expand(e, vars);

        // Can't hoist a mask containing loads.
        if (!find_objects<load_t>(e_expanded).empty()) return e;

        auto it = hoisted_masks_.find(e_expanded);
        if (it != hoisted_masks_.end()) return it->second;

        auto var = ir_ctx_.create_tmp_var(type_t::u16());
        hoisted_masks_.emplace(e_expanded, var);

        return var;
    }

    expr_t expand(const expr_t &_e, const std::vector<object_t> &e_vars) const {
        auto e = _e;
        for (auto &v : e_vars) {
            auto it = let_values_.find(v);
            if (it == let_values_.end()) continue;
            e = substitute(e, v, it->second);
        }
        return e;
    }

    stmt_t create_mask_stmt(const stmt_t &body) {
        stmt_t s = body;

        object_eq_map_t<expr_t, expr_t> and_ops;
        object_eq_map_t<expr_t, expr_t> mask_exprs;
        for (auto &kv : hoisted_masks_) {
            if (split_by_and_) {
                auto e = split_by_and_ops(kv.first, and_ops);
                mask_exprs.emplace(e, kv.second);
            }
        }
        if (and_ops.size() < mask_exprs.size()) {
            for (auto &kv : mask_exprs) {
                s = let_t::make(kv.second, cast(kv.first, kv.second.type()), s);
            }
            for (auto &kv : and_ops) {
                s = let_t::make(kv.second, cast(kv.first, kv.second.type()), s);
            }
        } else {
            for (auto &kv : hoisted_masks_)
                s = let_t::make(kv.second, cast(kv.first, kv.second.type()), s);
        }

        return s;
    }

    expr_t split_by_and_ops(
            const expr_t &e, object_eq_map_t<expr_t, expr_t> &ops) {
        auto *binary_op = e.as_ptr<binary_op_t>();
        if (!binary_op || binary_op->op_kind != op_kind_t::_and) {
            auto it = ops.find(e);
            if (it != ops.end()) return it->second;

            auto var = ir_ctx_.create_tmp_var(type_t::u16());
            ops.emplace(e, var);
            return var;
        }

        auto a = split_by_and_ops(binary_op->a, ops);
        auto b = split_by_and_ops(binary_op->b, ops);
        return binary_op_t::make(op_kind_t::_and, a, b);
    }

    bool in_stmt_group = false;
    object_set_t<expr_t> loop_deps_;
    object_eq_map_t<expr_t, expr_t> hoisted_masks_;
    object_map_t<expr_t, expr_t> let_values_;

    ir_context_t &ir_ctx_;
    stmt_label_t label_;
    bool split_by_and_;
};

stmt_t hoist_send_masks(const stmt_t &s, ir_context_t &ir_ctx,
        const stmt_label_t &label, bool split_by_and) {
    trace_start();
    hoist_send_masks_mutator_t mutator(ir_ctx, label, split_by_and);

    auto ret = mutator.mutate(s);
    trace_pass("hoist_send_masks", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
