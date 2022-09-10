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

#include "gpu/jit/pass/overflow.hpp"

#include "gpu/jit/pass/expr_scalarizer.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class overflow_bound_finder_t : public bound_finder_base_t {
public:
    bool has_var(const expr_t &e) const {
        ir_assert(is_var(e)) << "Expected variable, found: " << e;
        auto it = var_bounds_.find(e);
        return it != var_bounds_.end();
    }

    std::pair<int64_t, int64_t> find_bounds(const expr_t &e) const {
        int64_t lo = find_low_bound(e);
        int64_t hi = find_high_bound(e);
        return std::make_pair(lo, hi);
    }

    int64_t get_var_bound(const expr_t &e, bool is_low) const override {
        ir_assert(has_var(e)) << "Variable not found: " << e;
        auto &lo_hi = var_bounds_.at(e);
        return is_low ? lo_hi.first : lo_hi.second;
    }

    void set_var_bounds(
            const expr_t &e, const std::pair<int64_t, int64_t> &lo_hi) {
        ir_assert(is_good_bound(lo_hi.first))
                << "Can't compute low bound for " << e;
        ir_assert(is_good_bound(lo_hi.second))
                << "Can't compute high bound for " << e;
        var_bounds_.emplace(e, lo_hi);
    }

protected:
    int64_t find_bound_impl(const expr_t &e, bool is_low) const override {
        auto *cast = e.as_ptr<cast_t>();
        if (cast) {
            if (e.type().is_u64() && cast->expr.type().is_ptr()) {
                return is_low ? 0 : std::numeric_limits<uint32_t>::max();
            } else if (e.type().is_u32() && cast->expr.type().is_ptr()) {
                return is_low ? 0 : std::numeric_limits<uint16_t>::max();
            }
        }
        return bound_finder_base_t::find_bound_impl(e, is_low);
    }

private:
    object_map_t<expr_t, std::pair<int64_t, int64_t>> var_bounds_;
};

class overflow_fixer_t : public ir_mutator_t {
public:
    overflow_fixer_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {
        for (auto &kv : ir_ctx.cset().relations()) {
            int64_t lo = bound_finder_base_t::unlimited_bound(true);
            int64_t hi = bound_finder_base_t::unlimited_bound(false);
            for (auto &rel : kv.second) {
                bool is_ge = (rel.op_kind() == op_kind_t::_ge);
                bool is_le = (rel.op_kind() == op_kind_t::_le);
                ir_assert(is_ge || is_le);
                if (rel.op_kind() == op_kind_t::_ge) {
                    lo = std::max(to_cpp<int64_t>(rel.rhs()), lo);
                } else if (rel.op_kind() == op_kind_t::_le) {
                    hi = std::min(to_cpp<int64_t>(rel.rhs()), hi);
                } else {
                    ir_error_not_expected()
                            << "Only >= or <= is expected, found: "
                            << to_string(rel.op_kind());
                }
            }
            bound_finder_.set_var_bounds(kv.first, {lo, hi});
        }
    }

    object_t _mutate(const alloc_t &obj) override {
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const binary_op_t &obj) override {
        return mutate_expr(obj);
    }

    object_t _mutate(const for_t &obj) override {
        auto lo = to_cpp<int64_t>(obj.init);
        auto hi = to_cpp<int64_t>(obj.bound) - 1;
        bound_finder_.set_var_bounds(obj.var, {lo, hi});
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const let_t &obj) override {
        bool ok = true;
        if (!obj.var.type().is_int()) ok = false;
        if (ok && obj.value.is_empty()) ok = false;
        if (ok && bound_finder_.has_var(obj.var)) ok = false;

        if (ok) {
            if (contains_load(obj.value)) {
                vars_with_load_.insert(obj.var);
                ok = false;
            }
        }

        if (ok) {
            int elems = obj.var.type().elems();
            vec_vars_[obj.var].reserve(elems);
            for (int i = 0; i < elems; i++) {
                auto var_i = make_vec_var(obj.var, elems, i);
                expr_scalarizer_t scalarizer(elems, i, vec_vars_);
                auto value_i = scalarizer.mutate(obj.value);
                auto lo_hi = bound_finder_.find_bounds(value_i);
                bound_finder_.set_var_bounds(var_i, lo_hi);
                vec_vars_[obj.var].push_back(var_i);
            }
        }
        expr_t var = obj.var;
        expr_t value = mutate(obj.value);
        stmt_t body = mutate(obj.body);
        if (value.is_same(obj.value) && body.is_same(obj.body)) return obj;
        if (!value.is_empty() && value.type() != obj.value.type()) {
            auto old_var = var;
            var = ir_ctx_.create_tmp_var(
                    value.type(), old_var.as<var_t>().name);
            body = substitute(body, old_var, var);
        }
        return let_t::make(var, value, body);
    }

    object_t _mutate(const unary_op_t &obj) override {
        return mutate_expr(obj);
    }

private:
    template <typename T>
    object_t mutate_expr(const T &obj) {
        expr_t new_obj = ir_mutator_t::_mutate(obj);
        if (!new_obj.type().is_x32()) return std::move(new_obj);
        if (contains_load(new_obj)) return std::move(new_obj);

        bool found_overflow = false;
        int elems = new_obj.type().elems();
        for (int i = 0; i < elems; i++) {
            expr_scalarizer_t scalarizer(elems, i, vec_vars_);
            expr_t value = scalarizer.mutate(new_obj);
            int64_t lo = bound_finder_.find_low_bound(value);
            int64_t hi = bound_finder_.find_high_bound(value);
            bool ok = bound_finder_base_t::is_good_bound(lo)
                    && bound_finder_base_t::is_good_bound(hi);
            if (ok) {
                int64_t type_lo = value.type().is_s32()
                        ? (int64_t)std::numeric_limits<int32_t>::min()
                        : (int64_t)std::numeric_limits<uint32_t>::min();
                int64_t type_hi = value.type().is_s32()
                        ? (int64_t)std::numeric_limits<int32_t>::max()
                        : (int64_t)std::numeric_limits<uint32_t>::max();

                bool is_overflow = (lo < type_lo || hi > type_hi);
                if (is_overflow) {
                    found_overflow = true;
                    ir_warning() << "Found overflow: " << value
                                 << " low bound: " << lo
                                 << " high bound: " << hi << std::endl;
                    break;
                }
            }
        }
        if (found_overflow) return fix_overflow(new_obj);
        return std::move(new_obj);
    }

    bool contains_load(const expr_t &e) const {
        if (!find_objects<load_t>(e).empty()) return true;
        for (auto &v : find_objects<var_t>(e)) {
            if (vars_with_load_.count(v) != 0) return true;
        }
        return false;
    }

    static expr_t make_vec_var(const expr_t &_var, int elems, int idx) {
        if (elems == 1) return _var;
        auto &var = _var.as<var_t>();
        auto vec_name = var.name + "_" + std::to_string(idx) + "_";
        return var_t::make(var.type.scalar(), vec_name);
    }

    static expr_t fix_overflow(const expr_t &e) {
        auto *binary = e.as_ptr<binary_op_t>();
        if (binary) {
            return binary_op_t::make(binary->op_kind,
                    cast(binary->a, type_t::u64(e.type().elems())), binary->b);
        }

        ir_error_not_expected() << "Can't fix overflow: " << e;
        return e;
    }

    ir_context_t &ir_ctx_;
    overflow_bound_finder_t bound_finder_;
    object_map_t<expr_t, std::vector<expr_t>> vec_vars_;
    object_set_t<expr_t> vars_with_load_;
};

stmt_t fix_int32_overflow(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = overflow_fixer_t(ir_ctx).mutate(s);
    trace_pass("fix_int32_overflow", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
