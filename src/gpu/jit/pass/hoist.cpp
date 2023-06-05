/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "gpu/jit/codegen/register_allocator.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class sum_expr_t {
public:
    sum_expr_t(const expr_t &e)
        : type_(e.type()), args_(split_by_add(e, e.type().elems())) {}

    std::vector<expr_t> args() const { return args_; }

    bool is_trivial() const { return args_.size() <= 1; }

    expr_t expr() const { return make_add(args_, type_); }

    static expr_t make_add(
            const std::vector<expr_t> &args, const type_t &type) {
        auto maybe_bcast = [&](const expr_t &e) {
            if (e.type().elems() == type.elems()) return e;
            ir_assert(e.type().is_scalar());
            return shuffle_t::make_broadcast(e, type.elems());
        };
        if (args.empty()) return cast(0, type);
        auto ret = maybe_bcast(args[0]);
        for (int i = 1; i < (int)args.size(); i++)
            ret += maybe_bcast(args[i]);
        return ret;
    }

private:
    static std::vector<expr_t> split_by_add(const expr_t &e, int elems) {
        auto *shuffle = e.as_ptr<shuffle_t>();
        if (shuffle && shuffle->is_broadcast() && shuffle->elems() == elems) {
            return split_by_add(shuffle->vec[0], elems);
        }
        auto *op = e.as_ptr<binary_op_t>();
        if (!op || op->op_kind != op_kind_t::_add) return {e};
        auto a_args = split_by_add(op->a, elems);
        auto b_args = split_by_add(op->b, elems);
        std::vector<expr_t> args;
        args.insert(args.end(), a_args.begin(), a_args.end());
        args.insert(args.end(), b_args.begin(), b_args.end());
        return args;
    }

    type_t type_;
    std::vector<expr_t> args_;
};

class hoist_exprs_mutator_t : public ir_mutator_t {
public:
    hoist_exprs_mutator_t(ir_context_t &ir_ctx,
            int max_hoist_size = std::numeric_limits<int>::max())
        : ir_ctx_(ir_ctx), max_hoist_size_(max_hoist_size) {}

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
        if (is_const_let && loops_.size() > 0 && can_hoist(obj.var)) {
            fully_hoisted = true;
            register_let(obj.var, obj.value);
            add_hoist_let(loops_[0], obj.var, obj.value);
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

    bool can_hoist(const expr_t &expr) const {
        return expr.type().size() <= max_hoist_size_ - current_hoist_size_;
    }

    void add_hoist_let(
            loop_info_t &loop, const expr_t &var, const expr_t &value) {
        loop.lets.emplace_back(let_t::make(var, value));
        current_hoist_size_ += utils::rnd_up(
                var.type().size(), reg_allocator_t::granularity);
    }

    expr_t hoist_expr(const expr_t &expr, const expr_t &expr_var = {},
            bool *fully_hoisted = nullptr) {
        if (expr.is_empty()) return expr;
        if (expr.type().is_ptr()) return expr;
        if (expr.type().is_bool()) return expr;
        if (is_const(expr) || is_shuffle_const(expr) || is_var(expr))
            return expr;
        if (!can_hoist(expr)) return expr;

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
        const type_t &type = expr.type();
        sum_expr_t cur_expr(expr);

        for (size_t i = 0; i < loops_.size(); i++) {
            std::vector<expr_t> invariant_args;
            std::vector<expr_t> other_args;
            std::vector<expr_t> nary_args;
            if (!cur_expr.is_trivial()) {
                nary_args = cur_expr.args();
            } else {
                nary_args.push_back(cur_expr.expr());
            }
            for (auto &a : nary_args) {
                bool is_inv_arg = true;
                for (size_t j = i; j < loops_.size(); j++) {
                    if (!is_invariant(a, loops_[j].var)) is_inv_arg = false;
                }
                if (is_inv_arg) {
                    invariant_args.push_back(a);
                } else {
                    other_args.push_back(a);
                }
            }
            // Nothing to hoist for this loop, continue.
            if (invariant_args.empty()) continue;
            if (invariant_args.size() == 1 && is_var(invariant_args[0])
                    && !other_args.empty())
                continue;
            if (invariant_args.size() == 1
                    && (is_const(invariant_args[0])
                            || is_const_broadcast(invariant_args[0])))
                continue;

            // Introduce new variable for the invariant sub-expression.
            auto inv_expr = sum_expr_t::make_add(invariant_args, type);
            expr_t inv_var;
            if (!expr_var.is_empty() && other_args.empty()) {
                // If nothing to hoist further, reuse the old variable and
                // return.
                inv_var = expr_var;
            } else {
                inv_var = ir_ctx_.create_tmp_var(inv_expr.type());
            }
            register_let(inv_var, inv_expr);
            add_hoist_let(loops_[i], inv_var, inv_expr);

            if (other_args.empty()) {
                if (fully_hoisted) *fully_hoisted = true;
                return inv_var;
            }

            other_args.push_back(inv_var);
            cur_expr = sum_expr_t::make_add(other_args, type);
        }
        return cur_expr.expr();
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
        if (e.is_empty()) return true;
        if (!can_hoist(e)) return false;

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
    int max_hoist_size_;
    int current_hoist_size_ = 0;

    object_map_t<expr_t, expr_t> let_vars_;
};
stmt_t hoist_exprs_impl(
        const stmt_t &s, ir_context_t &ir_ctx, int reserved_regs) {

    int grf_size = ir_ctx.hw_cfg().grf_size();
    int available_regs = ir_ctx.exec_cfg().regs() - reserved_regs;
    int memory_usage_limit = available_regs * grf_size;

    auto stmt = hoist_exprs_mutator_t(ir_ctx).mutate(s);

    int memory_usage = get_peak_regs(stmt, grf_size) * grf_size;
    if (memory_usage >= memory_usage_limit) {
        // Pessimistically hoist expressions. Does not identify and account for
        // hoists which do not change memory usage.
        int memory_usage_original = get_peak_regs(s, grf_size) * grf_size;
        stmt = hoist_exprs_mutator_t(
                ir_ctx, memory_usage_limit - memory_usage_original)
                       .mutate(s);
    }
    return stmt;
}

stmt_t hoist_exprs(const stmt_t &s, ir_context_t &ir_ctx, int reserved_regs) {
    trace_start();
    auto ret = hoist_exprs_impl(s, ir_ctx, reserved_regs);
    trace_pass("hoist_exprs", ret, ir_ctx);
    return ret;
}

class hoist_send_masks_mutator_t : public ir_mutator_t {
public:
    hoist_send_masks_mutator_t(ir_context_t &ir_ctx, const stmt_label_t &label,
            bool split_by_and,
            int max_hoist_size = std::numeric_limits<int>::max())
        : ir_ctx_(ir_ctx)
        , label_(label)
        , split_by_and_(split_by_and)
        , max_hoist_size_(max_hoist_size) {}

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

        ir_assert(hoisted_mask.type().is_u16() || hoisted_mask.type().is_u32())
                << hoisted_mask;

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

    bool can_hoist(const expr_t &expr) {
        return expr.type().size() <= max_hoist_size_ - current_hoist_size_;
    }

    expr_t hoist_mask(const expr_t &e) {
        ir_assert(e.type().is_bool()) << e;

        if (is_const(e) || is_shuffle_const(e)) return e;

        // Can't hoist a mask containing loop vars.
        auto vars = find_objects<var_t>(e);
        for (auto &v : vars) {
            if (is_loop_dependency(v)) return e;
        }

        auto e_expanded = simplify(expand(e, vars));

        // Can't hoist a mask containing loads.
        if (!find_objects<load_t>(e_expanded).empty()) return e;

        auto it = hoisted_masks_.find(e_expanded);
        if (it != hoisted_masks_.end()) return it->second;

        auto var = ir_ctx_.create_tmp_var(
                bool_imm_t::get_packed_type(e.type().elems()));
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
            for (auto &kv : sort_var_map_by_value(mask_exprs)) {
                s = let_t::make(kv.second, cast(kv.first, kv.second.type()), s);
            }
            for (auto &kv : sort_var_map_by_value(and_ops)) {
                s = let_t::make(kv.second, cast(kv.first, kv.second.type()), s);
            }
        } else {
            for (auto &kv : sort_var_map_by_value(hoisted_masks_))
                s = let_t::make(kv.second, cast(kv.first, kv.second.type()), s);
        }

        return s;
    }

    expr_t split_by_and_ops(
            const expr_t &e, object_eq_map_t<expr_t, expr_t> &ops) {
        auto *binary_op = e.as_ptr<binary_op_t>();
        if (!binary_op || binary_op->op_kind != op_kind_t::_and) {
            auto _e = simplify(e);
            auto it = ops.find(_e);
            if (it != ops.end()) return it->second;

            if (can_hoist(_e)) {
                auto var = ir_ctx_.create_tmp_var(
                        bool_imm_t::get_packed_type(e.type().elems()));
                ops.emplace(_e, var);
                current_hoist_size_ += utils::rnd_up(
                        var.type().size(), reg_allocator_t::granularity);
                return var;
            } else {
                return _e;
            }
        }
        auto a = split_by_and_ops(binary_op->a, ops);
        auto b = split_by_and_ops(binary_op->b, ops);
        if (a.type() != b.type()) a = cast(a, b.type());
        return binary_op_t::make(binary_op->op_kind, a, b);
    }

    bool in_stmt_group = false;
    object_set_t<expr_t> loop_deps_;
    object_eq_map_t<expr_t, expr_t> hoisted_masks_;
    object_map_t<expr_t, expr_t> let_values_;

    ir_context_t &ir_ctx_;
    stmt_label_t label_;
    bool split_by_and_;
    int max_hoist_size_;
    int current_hoist_size_ = 0;
};

stmt_t hoist_send_masks(const stmt_t &s, ir_context_t &ir_ctx,
        const stmt_label_t &label, bool split_by_and, int reserved_regs) {
    trace_start();
    int grf_size = ir_ctx.hw_cfg().grf_size();
    int available_regs = ir_ctx.exec_cfg().regs() - reserved_regs;
    int memory_usage_limit = available_regs * grf_size;

    auto ret
            = hoist_send_masks_mutator_t(ir_ctx, label, split_by_and).mutate(s);

    int memory_usage = get_peak_regs(ret, grf_size) * grf_size;
    if (memory_usage >= memory_usage_limit) {
        // Pessimistically hoist expressions. Does not identify and account for
        // hoists which do not change memory usage.
        int memory_usage_original = get_peak_regs(s, grf_size) * grf_size;
        ret = hoist_send_masks_mutator_t(ir_ctx, label, split_by_and,
                memory_usage_limit - memory_usage_original)
                      .mutate(s);
    }

    trace_pass("hoist_send_masks", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
