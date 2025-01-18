/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "gpu/intel/jit/pass/cse.hpp"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include <type_traits>
#include <unordered_map>

#include "gpu/intel/jit/codegen/register_allocator.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/utils/trace.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

// Common subexpression elimination support.

// Represents an expression-candidate to eliminate.
class cse_expr_t {
public:
    cse_expr_t(const expr_t &expr, const expr_t &orig_expr,
            const ir_path_t &path, int refs = 1, const expr_t &cse_var = {})
        : expr(expr)
        , orig_expr(orig_expr)
        , path(path)
        , refs(refs)
        , cse_var(cse_var) {
        ir_trace() << "cse_pass: add expression: " << expr;
    }

    void add_usage(const ir_path_t &other_path, bool do_increment = true) {
        if (do_increment) refs++;
        path.merge(other_path);
        ir_trace() << "cse_pass: add usage: " << expr
                   << ", total refs: " << refs;
    }

    // Expression to eliminate via let.
    expr_t expr;
    // Original expression to eliminate (doesn't contain any CSEed vars).
    expr_t orig_expr;
    // Path to the innermost IR node where the expression can be defined.
    ir_path_t path;
    // Number of references to the expression.
    int refs;
    // Variable assigned to the expression (if decided to eliminate).
    expr_t cse_var;
};

// Helper class for CSE variables to query computational cost
// while tracking dependencies to other potential CSE
// variables.
class cse_var_entry_t {
public:
    cse_var_entry_t(const cse_expr_t *cse_expr) : cse_expr_(cse_expr) {}

    const cse_expr_t *cse_expr() const { return cse_expr_; }

    bool allocated() const { return allocated_; }

    void set_unallocated() { allocated_ = false; }
    void set_allocated() { allocated_ = true; }

    int size() const {
        return utils::rnd_up(
                cse_expr_->cse_var.type().size(), reg_allocator_t::granularity);
    }

    int cost() const { return cost_; }

    void set_var2entry(
            const object_map_t<expr_t, cse_var_entry_t *> &var2entry) {
        var2entry_ = &var2entry;
    }

    void recompute_cost() {
        cost_ = expr_cost(cse_expr_->expr, var2entry_) * cse_expr_->refs;
    }

    static int expr_cost(const expr_t &e,
            const object_map_t<expr_t, cse_var_entry_t *> *var2entry) {
        if (is_var(e)) {
            if (var2entry == nullptr) return 0;
            auto it = var2entry->find(e);
            if (it == var2entry->end()) return 0;
            if (it->second->allocated()) return 0;
            // If variable is not allocated, its value
            // has to be recomputed every time.
            return it->second->cost();
        }
        if (is_const(e)) return 0;
        if (e.is<cast_t>()) return e.type().is_bool();
        if (auto *op = e.as_ptr<binary_op_t>()) {
            return expr_cost(op->a, var2entry) + expr_cost(op->b, var2entry)
                    + 1;
        }
        if (auto *op = e.as_ptr<unary_op_t>()) {
            return expr_cost(op->a, var2entry) + 1;
        }
        if (auto *s = e.as_ptr<shuffle_t>()) {
            if (s->is_broadcast()) return 0;
            return s->elems();
        }
        ir_error_not_expected() << "Unhandled expression: " << e;
        return 0;
    }

private:
    const cse_expr_t *cse_expr_ = nullptr;
    int cost_ = 0;
    bool allocated_ = true;
    const object_map_t<expr_t, cse_var_entry_t *> *var2entry_ = nullptr;
};

// Greedily marks the least beneficial cse entries as unallocated, so that those
// expressions can be skipped in the final CSE output.
class cse_skipper_t : public ir_visitor_t {
public:
    cse_skipper_t(const object_eq_map_t<expr_t, cse_expr_t> &cse_exprs,
            int grf_limit, int grf_size)
        : grf_limit_(grf_limit), grf_size_(grf_size) {

        for (auto &kv : cse_exprs) {
            auto &cse_expr = kv.second;
            if (cse_expr.cse_var.is_empty()) continue;
            entries_.emplace_back(&cse_expr);
        }

        for (auto &e : entries_) {
            var2entry_.emplace(e.cse_expr()->cse_var, &e);
            e.set_var2entry(var2entry_);
        }
    }

    void _visit(const alloc_t &obj) override {
        auto size = obj.register_alloc_size(grf_size_);
        grf_usage_ += size;
        handle_grf_overflow();

        ir_visitor_t::_visit(obj);

        grf_usage_ -= size;
    }

    void _visit(const let_t &obj) override {
        auto it = var2entry_.find(obj.var);
        auto *e = it != var2entry_.end() ? var2entry_.find(obj.var)->second
                                         : nullptr;

        int size = obj.register_alloc_size();
        if (e) {
            var_stack_.emplace_back(e);
            if (e->allocated()) { grf_usage_ += size; }
        } else {
            grf_usage_ += size;
        }
        handle_grf_overflow();

        ir_visitor_t::_visit(obj);

        if (e) {
            if (e->allocated()) grf_usage_ -= size;
            var_stack_.pop_back();
        } else {
            grf_usage_ -= size;
        }
    }

    void handle_grf_overflow() {
        if (grf_usage_ <= grf_limit_) return;

        std::vector<cse_var_entry_t *> sorted_var_entries = [&]() {
            std::vector<cse_var_entry_t *> ret;
            for (auto v : var_stack_) {
                if (v->allocated()) ret.emplace_back(v);
            }
            return ret;
        }();

        auto it = sorted_var_entries.begin();
        while (grf_usage_ > grf_limit_ && it != sorted_var_entries.end()) {
            // var_stack_ is guaranteed to be in topological order due to
            // traversing the IR tree.
            for (auto &e : var_stack_) {
                e->recompute_cost();
            }
            std::sort(it, sorted_var_entries.end(),
                    [&](const cse_var_entry_t *a, const cse_var_entry_t *b) {
                        // Sort by cost per byte
                        return a->cost() * b->size() < b->cost() * a->size();
                    });
            auto &e = **it;

            ir_trace() << "cse_pass: skipping " << e.cse_expr()->expr
                       << " with cost " << e.cost() << ", size " << e.size()
                       << ", and cost per byte " << (double)e.cost() / e.size();

            e.set_unallocated();
            grf_usage_ -= e.size();
            ++it;
        }
    }

    const std::vector<cse_var_entry_t> &entries() const { return entries_; };

private:
    std::vector<cse_var_entry_t> entries_;
    object_map_t<expr_t, cse_var_entry_t *> var2entry_;
    std::vector<cse_var_entry_t *> var_stack_;
    int grf_usage_ = 0;
    int grf_limit_ = 0;
    int grf_size_ = 0;
};

// Stores information about all expressions subject to CSEing.
class cse_context_t {
public:
    cse_context_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    ir_context_t &ir_ctx() { return ir_ctx_; }

    bool has(const expr_t &e) const { return cse_exprs_.count(e) != 0; }

    cse_expr_t &find_cse_expr(const expr_t &e) {
        ir_assert(has(e)) << e;
        return cse_exprs_.at(e);
    }

    const cse_expr_t &find_cse_expr(const expr_t &e) const {
        ir_assert(has(e)) << e;
        return cse_exprs_.at(e);
    }

    bool has_var(const expr_t &e) const {
        return !find_cse_expr(e).cse_var.is_empty();
    }

    int get_refs(const expr_t &e) const {
        if (!has(e)) return 0;
        return find_cse_expr(e).refs;
    }

    void register_expr(const expr_t &e, const ir_path_t &path) {
        auto ret = cse_exprs_.insert({e, cse_expr_t(e, e, path)});
        ir_assert(ret.second) << e;
        MAYBE_UNUSED(ret);
    }

    void register_expr(const cse_expr_t &cse_expr) {
        auto ret = cse_exprs_.insert({cse_expr.expr, cse_expr});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
    }

    expr_t get_or_assign_var(const expr_t &e) {
        auto &cse_expr = find_cse_expr(e);
        if (cse_expr.cse_var.is_empty()) {
            cse_expr.cse_var = ir_ctx_.create_tmp_var(e.type().is_bool()
                            ? bool_imm_t::get_packed_type(e.type().elems())
                            : e.type());
            ir_trace() << "cse_pass: assigning var: " << e << " -> "
                       << cse_expr.cse_var;
        }
        return cse_expr.cse_var;
    }

    const expr_t &get_var(const expr_t &e) const {
        return find_cse_expr(e).cse_var;
    }

    const ir_path_t &get_path(const expr_t &e) const {
        return find_cse_expr(e).path;
    }

    void add_usage(
            const expr_t &e, const ir_path_t &path, bool do_increment = true) {
        find_cse_expr(e).add_usage(path, do_increment);
    }

    void update_expr(const expr_t &old_expr, const expr_t &new_expr) {
        auto it = cse_exprs_.find(old_expr);
        ir_assert(it != cse_exprs_.end()) << old_expr;
        auto &old_cse_expr = it->second;
        auto new_cse_expr = cse_expr_t(new_expr, old_cse_expr.orig_expr,
                old_cse_expr.path, old_cse_expr.refs, old_cse_expr.cse_var);
        cse_exprs_.erase(it);
        auto ret = cse_exprs_.insert({new_expr, new_cse_expr});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
    }

    template <typename F>
    void for_each(const F &f) const {
        auto sorted_exprs = sort_var_map(cse_exprs_,
                [](const std::pair<expr_t, cse_expr_t> &a,
                        const std::pair<expr_t, cse_expr_t> &b) {
                    auto &a_var = a.second.cse_var.as<var_t>();
                    auto &b_var = b.second.cse_var.as<var_t>();
                    return a_var.name < b_var.name;
                });
        for (auto &kv : sorted_exprs)
            f(kv.first);
    }

    bool should_assign_var(const expr_t &e) const {
        if (!has(e) || e.is<var_t>() || e.is<ptr_t>() || e.is<cast_t>()
                || is_const(e))
            return false;
        auto &cse_expr = find_cse_expr(e);

        if (cse_expr.refs <= 1) return false;
        if (e.type().is_bool()) {
            // Account for possible cost to move bool variable to and from flag
            // register
            auto cost = cse_var_entry_t::expr_cost(cse_expr.expr, nullptr);
            if (cost + cse_expr.refs + 1 >= cost * cse_expr.refs) return false;
        }
        if (skip_exprs_.count(cse_expr.orig_expr) != 0) return false;
        return true;
    }

    bool set_skip_exprs(const stmt_t &root, int limit, int grf_size) {
        cse_skipper_t skipper(cse_exprs_, limit, grf_size);
        skipper.visit(root);

        // TODO: Rather than rerun CSE, just delete `let_t` and substitute
        // variables with their value.
        for (auto &e : skipper.entries()) {
            if (e.allocated()) continue;
            skip_exprs_.insert(e.cse_expr()->orig_expr);
        }
        return !skip_exprs_.empty();
    }

    void reset_cse_exprs() { cse_exprs_.clear(); }

private:
    ir_context_t &ir_ctx_;
    object_eq_map_t<expr_t, cse_expr_t> cse_exprs_;
    object_eq_set_t<expr_t> skip_exprs_;
};

// Collects statistics about expressions for common subexpression elimination.
class cse_visitor_t : public ir_visitor_t {
public:
    cse_visitor_t(cse_context_t &ctx) : ctx_(ctx) {}

    void _visit(const binary_op_t &obj) override { visit_expr(obj); }
    void _visit(const shuffle_t &obj) override {
        if (is_const_broadcast(obj)) return;
        visit_expr(obj);
    }
    void _visit(const unary_op_t &obj) override { visit_expr(obj); }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type &obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

private:
    template <typename T>
    void visit_expr(const T &obj) {
        // Exclude loads as they may have side effects.
        if (count_objects<load_t>(obj) > 0) {
            ir_visitor_t::_visit(obj);
            return;
        }

        if (std::is_same<T, shuffle_t>::value) {
            auto &shuffle = reinterpret_cast<const shuffle_t &>(obj);
            if (shuffle.is_broadcast()) {
                ir_visitor_t::_visit(obj);
                return;
            }
        }

        if (propagate_path_) {
            if (ctx_.has(obj))
                ctx_.add_usage(obj, root_path_, /*do_increment=*/false);
            ir_visitor_t::_visit(obj);
            return;
        }
        if (ctx_.has(obj)) {
            ctx_.add_usage(obj, root_path_);
            propagate_path_ = true;
            ir_visitor_t::_visit(obj);
            propagate_path_ = false;
            return;
        }
        ir_visitor_t::_visit(obj);
        ctx_.register_expr(obj, root_path_);
    }

    template <typename T>
    void visit_stmt(const T &obj) {
        if (std::is_same<T, for_t>::value) {
            visit_for((const object_impl_t &)obj);
            return;
        }
        if (std::is_same<T, let_t>::value) {
            visit_let((const object_impl_t &)obj);
            return;
        }
        root_path_.push(&obj);
        ir_visitor_t::_visit(obj);
        root_path_.pop();
    }

    void visit_for(const object_impl_t &_obj) {
        auto &obj = (const for_t &)_obj;

        visit(obj.var);
        visit(obj.init);
        visit(obj.bound);
        root_path_.push(&obj);
        visit(obj.body);
        root_path_.pop();
    }

    void visit_let(const object_impl_t &_obj) {
        auto &obj = (const let_t &)_obj;

        visit(obj.var);
        visit(obj.value);
        root_path_.push(&obj);
        visit(obj.body);
        root_path_.pop();
    }

    cse_context_t &ctx_;
    ir_path_t root_path_;

    bool propagate_path_ = false;
};

// Verifies all IR paths are correct (for debugging purposes).
class cse_verifier_t : public scope_visitor_t {
public:
    cse_verifier_t(cse_context_t &ctx) : ctx_(ctx) {}

    ~cse_verifier_t() override { ir_assert(to_check_.empty()); }

    void _visit(const binary_op_t &obj) override { visit_expr(obj); }
    void _visit(const shuffle_t &obj) override { visit_expr(obj); }
    void _visit(const unary_op_t &obj) override { visit_expr(obj); }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type &obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

    void verify(const stmt_t &s) {
        // Phase 0: collect IR paths for expressions.
        phase_ = 0;
        visit(s);

        // Phase 1: verify all expressions are defined at their path.
        phase_ = 1;
        visit(s);
    }

private:
    template <typename T>
    void visit_expr(const T &obj) {
        // Expressions are not used during phase 1.
        if (phase_ == 1) return;
        if (ctx_.has(obj)) {
            auto &path = ctx_.get_path(obj);
            to_check_[path.back()].push_back(obj);
        }
        scope_visitor_t::_visit(obj);
    }

    template <typename T>
    void visit_stmt(const T &obj) {
        scope_visitor_t::_visit(obj);

        // Statements are not used during phase 0.
        if (phase_ == 0) return;

        // Phase 1: check that all attached expressions are defined at this
        // statement.
        auto it = to_check_.find(obj);
        if (it != to_check_.end()) {
            for (auto &e : it->second) {
                ir_assert(is_expr_defined(e))
                        << "Expression contains undefined variables: " << e;
                MAYBE_UNUSED(e);
            }
            to_check_.erase(it);
        }
    }

    cse_context_t &ctx_;

    int phase_ = 0;
    object_map_t<stmt_t, std::vector<expr_t>> to_check_;
};

// Generates let statements for expressions being eliminated.
class cse_let_generator_t : public ir_visitor_t {
public:
    cse_let_generator_t(const cse_context_t &ctx, const stmt_t &stmt)
        : ctx_(ctx), stmt_(stmt) {}

    void _visit(const binary_op_t &obj) override { visit_expr(obj); }
    void _visit(const shuffle_t &obj) override { visit_expr(obj); }
    void _visit(const unary_op_t &obj) override { visit_expr(obj); }
    void _visit(const var_t &obj) override {
        auto it = all_vars_.find(obj);
        if (it == all_vars_.end()) return;
        if (seen_vars_.count(obj) == 0) generate_for_expr(it->second);
    }

    stmt_t generate() {
        ctx_.for_each([&](const expr_t &e) {
            auto &cse_var = ctx_.get_var(e);
            auto ret = all_vars_.insert({cse_var, e});
            ir_assert(ret.second);
            MAYBE_UNUSED(ret);
        });
        ctx_.for_each([&](const expr_t &e) { generate_for_expr(e); });
        for (auto it = lets_.rbegin(); it != lets_.rend(); ++it) {
            auto &let = it->as<let_t>();
            stmt_ = let_t::make(let.var, let.value, stmt_);
        }
        return stmt_;
    }

private:
    void generate_for_expr(const expr_t &e) {
        auto &cse_var = ctx_.get_var(e);
        if (seen_vars_.count(cse_var) == 1) return;
        visit(e);
    }

    template <typename T>
    void visit_expr(const T &obj) {
        ir_visitor_t::_visit(obj);
        if (ctx_.has(obj) && ctx_.has_var(obj)) {
            auto &var = ctx_.get_var(obj);
            auto ret = seen_vars_.insert(var);
            if (ret.second)
                lets_.push_back(let_t::make(var,
                        obj.type.is_bool() ? cast(obj,
                                bool_imm_t::get_packed_type(obj.type.elems()))
                                           : obj));
        }
    }

    const cse_context_t &ctx_;
    stmt_t stmt_;

    object_map_t<expr_t, expr_t> all_vars_; // Var -> expression.
    object_set_t<expr_t> seen_vars_;

    std::vector<stmt_t> lets_;
};

// Eliminates expressions from the statement.
class cse_mutator_t : public ir_mutator_t {
public:
    cse_mutator_t(cse_context_t &ctx) : ctx_(ctx) {}

    object_t _mutate(const binary_op_t &obj) override {
        return mutate_expr(obj);
    }
    object_t _mutate(const shuffle_t &obj) override {
        return cast(mutate_expr(obj), obj.type);
    }
    object_t _mutate(const unary_op_t &obj) override {
        return mutate_expr(obj);
    }

#define HANDLE_IR_OBJECT(type) \
    object_t _mutate(const type &obj) override { return mutate_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

private:
    template <typename T>
    object_t mutate_expr(const T &obj) {
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (ctx_.has(obj) && !new_obj.is_equal(obj)) {
            ctx_.update_expr(obj, new_obj);
        }
        if (ctx_.should_assign_var(new_obj)) {
            bool has_var = ctx_.has_var(new_obj);
            auto var = ctx_.get_or_assign_var(new_obj);
            auto &path = ctx_.get_path(new_obj);
            if (!has_var) to_update_[path.back()].push_back(new_obj);
            if (obj.type.is_bool()) var = cast(var, obj.type);
            return std::move(var);
        }
        return new_obj;
    }

    template <typename T>
    object_t mutate_stmt(const T &obj) {
        // skip if it contains dp4a tenrary op, as there are issues mutating it
        if (std::is_same<T, store_t>::value
                && count_objects<ternary_op_t>(obj) > 0)
            return obj;
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto it = to_update_.find(obj);
        if (it == to_update_.end()) return new_obj;

        cse_context_t local_ctx(ctx_.ir_ctx());
        for (auto &e : it->second) {
            local_ctx.register_expr(ctx_.find_cse_expr(e));
        }
        to_update_.erase(it);

        auto body = get_stmt_body(new_obj);
        cse_let_generator_t g(local_ctx, body);
        body = g.generate();
        new_obj = replace_stmt_body(new_obj, body);
        return new_obj;
    }

    cse_context_t &ctx_;
    object_map_t<stmt_t, std::vector<expr_t>> to_update_;
};

stmt_t eliminate_common_subexprs_impl(const stmt_t &_stmt, cse_context_t &ctx,
        int grf_size, int memory_usage_limit, int run_idx) {
    auto stmt = _stmt;

    // Collect statistics.
    cse_visitor_t visitor(ctx);
    visitor.visit(stmt);

#if !defined(NDEBUG) || defined(DNNL_DEV_MODE)
    // Verify that collected IR paths are correct (cse_expr_t objects are
    // defined at those paths).
    cse_verifier_t verifier(ctx);
    verifier.verify(stmt);
#endif

    // Eliminate subexpressions.
    cse_mutator_t mutator(ctx);
    stmt = mutator.mutate(stmt);

    // The second run is the last run.
    if (run_idx != 0) {
        gpu_assert(
                get_peak_regs(stmt, grf_size) * grf_size <= memory_usage_limit
                || get_peak_regs(_stmt, grf_size) * grf_size
                        >= memory_usage_limit);
        return stmt;
    }

    // If memory usage exceeds the limit, exclude some expressions from CSE and
    // retry the whole process from scratch.
    bool has_skip = ctx.set_skip_exprs(stmt, memory_usage_limit, grf_size);
    if (!has_skip) return stmt;

    int memory_usage = get_peak_regs(stmt, grf_size) * grf_size;
    ir_trace() << "CSE exceeded GRF usage limit. Usage: " << memory_usage
               << ", limit: " << memory_usage_limit
               << ". Retry CSE and skip some expressions...";
    ctx.reset_cse_exprs();
    return stmt_t();
}

stmt_t eliminate_common_subexprs(
        const stmt_t &_stmt, ir_context_t &ir_ctx, int memory_usage_limit) {
    trace_start();
    stmt_t stmt;
    cse_context_t cse_ctx(ir_ctx);

    int grf_size = ir_ctx.hw().grf_size();
    stmt = eliminate_common_subexprs_impl(
            _stmt, cse_ctx, grf_size, memory_usage_limit, 0);
    // Retry if statement is empty, rely on the updated
    // skip_exprs from the CSE context.
    if (stmt.is_empty()) {
        stmt = eliminate_common_subexprs_impl(
                _stmt, cse_ctx, grf_size, memory_usage_limit, 1);
    }
    trace_pass("eliminate_common_subexprs", stmt, ir_ctx);
    return stmt;
}

class g2s_buf_visitor_t : public ir_visitor_t {
public:
    int g2s_buf_size() const {
        int ret = 0;
        for (auto &kv : g2s_bufs_) {
            ir_assert(kv.second != 0);
            ret += kv.second;
        }
        return ret;
    }

    void _visit(const alloc_t &obj) override {
        ir_visitor_t::_visit(obj);
        auto it = g2s_bufs_.find(obj.buf);
        if (it != g2s_bufs_.end()) it->second = obj.size;
    }

    void _visit(const func_call_t &obj) override {
        if (!in_g2s_) {
            ir_visitor_t::_visit(obj);
            return;
        }
        if (auto *func = obj.func.as_ptr<send_t>()) {
            ir_assert(func->is_load()) << func;
            auto &buf = send_t::arg_reg_buf(obj);
            g2s_bufs_.emplace(get_base(buf), 0);
        }
        ir_visitor_t::_visit(obj);
    }

    void _visit(const stmt_group_t &obj) override {
        bool is_g2s = obj.label == stmt_label_t::g2s_load();
        if (is_g2s) in_g2s_ = true;
        ir_visitor_t::_visit(obj);
        if (is_g2s) in_g2s_ = false;
    }

private:
    object_map_t<expr_t, int> g2s_bufs_;
    bool in_g2s_ = false;
};

stmt_t eliminate_common_subexprs(const stmt_t &stmt, ir_context_t &ir_ctx,
        int reserved_regs, int gmem_bufs) {
    int grf_size = ir_ctx.grf_size();
    int available_regs = ir_ctx.exec_cfg().regs() - reserved_regs;
    int memory_usage_limit = available_regs * grf_size;
    if (gmem_bufs > 1) {
        g2s_buf_visitor_t v;
        v.visit(stmt);
        memory_usage_limit -= (gmem_bufs - 1) * v.g2s_buf_size();
    }
    return eliminate_common_subexprs(stmt, ir_ctx, memory_usage_limit);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
