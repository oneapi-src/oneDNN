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

#ifndef GPU_JIT_IR_LINEAR_EXPR_HPP
#define GPU_JIT_IR_LINEAR_EXPR_HPP

#include <iostream>
#include <string>
#include <unordered_map>

#include "common/math_utils.hpp"
#include "gpu/jit/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

bool should_use_linear_op(const expr_t &a);
bool should_use_linear_op(const expr_t &a, const expr_t &b);

class linear_constraint_t {
public:
    void set_value(int value) {
        has_value_ = true;
        value_ = value;
    }
    bool has_value() const { return has_value_; }
    int value() const {
        ir_assert(has_value_);
        return value_;
    }
    void set_divisor(int divisor) {
        ir_assert(math::is_pow2(divisor));
        divisor_ = divisor;
    }
    int divisor() const { return divisor_; }

    std::string str() const {
        std::ostringstream oss;
        if (has_value_) oss << "value: " << value_ << " ";
        oss << "divisor: " << divisor_;
        return oss.str();
    }

    IR_DEFINE_DUMP()

    bool operator==(const linear_constraint_t &other) const {
        return (has_value_ == other.has_value_) && (value_ == other.value_)
                && (divisor_ == other.divisor_);
    }

    bool operator!=(const linear_constraint_t &other) const {
        return !operator==(other);
    }

    size_t get_hash() const {
        return ir_utils::get_hash(has_value_, value_, divisor_);
    }

    void serialize(std::ostream &out) const {
        ir_utils::serialize(has_value_, out);
        ir_utils::serialize(value_, out);
        ir_utils::serialize(divisor_, out);
    }

    void deserialize(std::istream &in) {
        ir_utils::deserialize(has_value_, in);
        ir_utils::deserialize(value_, in);
        ir_utils::deserialize(divisor_, in);
    }

private:
    bool has_value_ = false;
    int value_ = 0;
    int divisor_ = 1;
};

int max_pow2_divisor(const expr_t &e);

class linear_expr_ctx_t {
public:
    static bool is_set() { return ctx_; }
    static void set(linear_expr_ctx_t *ptr) { ctx_ = ptr; }

    static linear_expr_ctx_t &get() {
        ir_assert(ctx_);
        return *ctx_;
    }

    static expr_t expand(const expr_t &e) { return get().expand_impl(e); }

    const object_map_t<expr_t, expr_t> &const_vars() const {
        return const_vars_;
    }

    expr_t add_const(const expr_t &_expr) {
        auto expr = normalize(_expr);
        auto try_find = find(expr);
        if (!try_find.is_empty()) return try_find;
        auto name = "tmp_" + std::to_string(id_++);
        auto var = const_var_t::make(type_t::s32(), name);
        const_vars_.emplace(var, expr);
        return var;
    }

    void add_constraint(
            const expr_t &expr, const linear_constraint_t &constraint) {
        ir_assert(expr.is<const_var_t>());
        constraints_.emplace(expr, constraint);
    }

    linear_constraint_t constraint(const expr_t &expr) const {
        auto it = constraints_.find(expr);
        if (it == constraints_.end()) return linear_constraint_t();
        return it->second;
    }

    const expr_t &const_value(const expr_t &expr) const {
        auto it = const_vars_.find(expr);
        ir_assert(it != const_vars_.end());
        return it->second;
    }

    int max_pow2_divisor(const expr_t &a) const {
        auto it = const_vars_.find(a);
        if (it == const_vars_.end()) return constraint(a).divisor();
        auto &op = it->second.as<binary_op_t>();
        switch (op.op_kind) {
            case op_kind_t::_mul:
                return jit::max_pow2_divisor(op.a)
                        * jit::max_pow2_divisor(op.b);
            case op_kind_t::_add:
                return math::gcd(jit::max_pow2_divisor(op.a),
                        jit::max_pow2_divisor(op.b));
            case op_kind_t::_div_up: return 1;
            case op_kind_t::_minus: return jit::max_pow2_divisor(op.a);
            default: ir_error_not_expected();
        }
        return 1;
    }

    expr_t try_div(const expr_t &a, int b) const {
        if (b == 1) return a;
        auto div_result = try_div_impl(a, b);
        if (div_result.divisor == 1) return div_result.value;
        return expr_t();
    }

    expr_t expand_impl(const expr_t &e) const {
        if (auto *linear = e.as_ptr<linear_t>()) {
            auto c = expand_impl(linear->c);
            auto u_vec = linear->u_vec;
            for (auto &u : u_vec)
                u = expand_impl(u);
            return linear_t::make(c, u_vec, linear->v_vec);
        }
        auto cit = constraints_.find(e);
        if (cit != constraints_.end()) {
            auto &c = cit->second;
            if (c.has_value()) return expr_t(c.value());
        }

        auto it = const_vars_.find(e);
        if (it == const_vars_.end()) return e;

        auto &op = it->second.as<binary_op_t>();
        auto a = expand_impl(op.a);
        auto b = expand_impl(op.b);
        return binary_op_t::make(op.op_kind, a, b);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "expr_ctx:" << std::endl;
        if (const_vars_.empty()) {
            oss << "  (empty)";
        } else {
            auto usage = collect_usage();
            object_set_t<expr_t> seen;
            for (auto it = const_vars_.begin(); it != const_vars_.end(); it++)
                print(it, seen, usage, oss);
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    struct div_result_t {
        div_result_t() = default;
        div_result_t(const expr_t &value) : divisor(1), value(value) {}
        div_result_t(int divisor, const expr_t &value = expr_t(1))
            : divisor(divisor), value(value) {}

        int divisor = 1;
        expr_t value;
    };

    expr_t find(const expr_t &expr) const {
        for (auto &kv : const_vars_) {
            if (kv.second.is_equal(expr)) return kv.first;
        }
        return expr_t();
    }

    div_result_t try_div_impl(
            const expr_t &a, const div_result_t &div_result) const {
        if (div_result.divisor == 1) return a * div_result.value;
        if (a.is<int_imm_t>()) {
            int factor = math::gcd(to_int(a), div_result.divisor);
            return div_result_t(
                    factor, (to_int(a) / factor) * div_result.value);
        }
        auto it = const_vars_.find(a);
        if (it == const_vars_.end()) return div_result;

        auto &op = it->second.as<binary_op_t>();
        switch (op.op_kind) {
            case op_kind_t::_add: {
                auto a = try_div_impl(op.a, div_result);
                auto b = try_div_impl(op.b, div_result);
                int divisor = math::lcm(a.divisor, b.divisor);
                a.value *= (divisor / a.divisor);
                b.value *= (divisor / b.divisor);
                return div_result_t(divisor, a.value + b.value);
            }
            case op_kind_t::_mul: {
                div_result_t r = div_result;
                r = try_div_impl(op.a, r);
                r = try_div_impl(op.b, r);
                return r;
            }
            default: break;
        }
        return div_result;
    }

    static expr_t normalize(const expr_t &e) {
        if (!e.is<binary_op_t>()) return e;

        auto &op = e.as<binary_op_t>();
        bool ac = op.a.is<const_var_t>();
        bool ai = op.a.is<int_imm_t>();
        bool bc = op.b.is<const_var_t>();
        bool bi = op.b.is<int_imm_t>();
        switch (op.op_kind) {
            case op_kind_t::_minus: ir_assert(ac); return e;
            case op_kind_t::_div_up: ir_assert(ac && bi); return e;
            case op_kind_t::_add:
            case op_kind_t::_mul: {
                // Order operands according to deterministic rules.
                bool swap = false;
                if (ac && bi) {
                    swap = true;
                } else if (ac && bc) {
                    auto &a_name = op.a.as<const_var_t>().name;
                    auto &b_name = op.b.as<const_var_t>().name;
                    if (a_name > b_name) swap = true;
                } else {
                    ir_assert(ai && bc);
                }
                if (swap) return binary_op_t::make(op.op_kind, op.b, op.a);
                return e;
            }
            default: ir_error_not_expected();
        }
        return e;
    }

    void print(object_map_t<expr_t, expr_t>::const_iterator it,
            object_set_t<expr_t> &seen, const object_map_t<expr_t, int> &usage,
            std::ostringstream &oss) const {
        auto &key = it->first;
        auto &value = it->second;
        if (seen.count(key) > 0) return;
        seen.insert(key);
        auto &op = value.as<binary_op_t>();
        auto a_it = const_vars_.find(op.a);
        auto b_it = const_vars_.find(op.b);
        if (a_it != const_vars_.end()) print(a_it, seen, usage, oss);
        if (b_it != const_vars_.end()) print(b_it, seen, usage, oss);
        auto a_expanded = expand_impl(op.a);
        auto b_expanded = expand_impl(op.b);
        auto value_expanded
                = binary_op_t::make(op.op_kind, a_expanded, b_expanded);
        oss << "  " << key << " = " << value << std::endl;
        oss << "    [" << value_expanded << "]";
        if (seen.size() != const_vars_.size()) oss << std::endl;
    }

    object_map_t<expr_t, int> collect_usage() const {
        object_map_t<expr_t, int> ret;
        for (auto it = const_vars_.begin(); it != const_vars_.end(); it++) {
            if (ret.count(it->first) == 0) ret[it->first] = 0;
            auto &op = it->second.as<binary_op_t>();
            auto a_it = const_vars_.find(op.a);
            auto b_it = const_vars_.find(op.b);
            if (a_it != const_vars_.end()) ret[a_it->first]++;
            if (b_it != const_vars_.end()) ret[b_it->first]++;
        }
        return ret;
    }

    static thread_local linear_expr_ctx_t *ctx_;

    int id_ = 0;
    object_map_t<expr_t, expr_t> const_vars_;
    object_map_t<expr_t, linear_constraint_t> constraints_;
};

inline int max_pow2_divisor(const expr_t &e) {
    const int large_pow2 = (1 << 20);
    if (is_zero(e)) return large_pow2;
    if (e.is<const_var_t>())
        return linear_expr_ctx_t::get().max_pow2_divisor(e);
    if (e.is<var_t>()) return 1;
    if (auto *imm = e.as_ptr<int_imm_t>())
        return ir_utils::max_pow2_divisor(imm->value);
    if (auto *linear = e.as_ptr<linear_t>()) {
        int ret = max_pow2_divisor(linear->c);
        for (auto &u : linear->u_vec)
            ret = math::gcd(ret, max_pow2_divisor(u));
        return ret;
    }
    ir_error_not_expected();
    return 1;
}

inline expr_t to_linear(const expr_t &e) {
    if (e.is<linear_t>()) return e;
    if (e.is<int_imm_t>() || e.is<const_var_t>()) return linear_t::make(e);
    if (e.is<var_t>()) return linear_t::make(0, {e});
    ir_error_not_expected();
    return expr_t();
}

inline expr_t minus_linear(const expr_t &_a) {
    auto &a = _a.as<linear_t>();
    auto c = -a.c;
    auto u_vec = a.u_vec;
    for (auto &u : u_vec)
        u *= -1;
    return linear_t::make(c, u_vec, a.v_vec);
}

inline expr_t add_linear(const expr_t &_a, const expr_t &_b) {
    auto &a = _a.as<linear_t>();
    auto &b = _b.as<linear_t>();
    auto c = a.c + b.c;
    auto u_vec = a.u_vec;
    auto v_vec = a.v_vec;
    for (int i = 0; i < b.nargs(); i++) {
        bool found = false;
        for (int j = 0; j < a.nargs(); j++) {
            if (v_vec[j].impl() == b.v_vec[i].impl()) {
                u_vec[j] += b.u_vec[i];
                found = true;
                break;
            }
        }
        if (!found) {
            u_vec.push_back(b.u_vec[i]);
            v_vec.push_back(b.v_vec[i]);
        }
    }
    return linear_t::make(c, u_vec, v_vec);
}

inline expr_t mul_linear(const expr_t &_a, const expr_t &b) {
    auto &a = _a.as<linear_t>();
    auto c = a.c * b;
    auto u_vec = a.u_vec;
    for (auto &u : u_vec)
        u *= b;
    return linear_t::make(c, u_vec, a.v_vec);
}

inline expr_t div_linear(const expr_t &_a, int b) {
    auto &a = _a.as<linear_t>();
    auto c = a.c / b;
    auto u_vec = a.u_vec;
    for (auto &u : u_vec)
        u /= b;
    return linear_t::make(c, u_vec, a.v_vec);
}

inline expr_t linear_op(op_kind_t op_kind, const expr_t &a) {
    ir_assert(op_kind == op_kind_t::_minus);
    if (a.is<var_t>() || a.is<linear_t>()) return minus_linear(to_linear(a));
    auto *a_imm = a.as_ptr<int_imm_t>();
    if (a_imm) return expr_t(-a_imm->value);
    ir_assert(a.is<const_var_t>());
    return -1 * a;
}

inline expr_t linear_op(op_kind_t op_kind, const expr_t &a, const expr_t &b) {
    switch (op_kind) {
        case op_kind_t::_add: {
            if (is_zero(a)) return b;
            if (is_zero(b)) return a;
            if (a.is<var_t>() || a.is<linear_t>() || b.is<var_t>()
                    || b.is<linear_t>()) {
                return add_linear(to_linear(a), to_linear(b));
            }
            ir_assert(!a.is<int_imm_t>() || !b.is<int_imm_t>());
            ir_assert(a.is<const_var_t>() || b.is<const_var_t>());
            auto op = binary_op_t::make(op_kind_t::_add, a, b);
            return linear_expr_ctx_t::get().add_const(op);
        }
        case op_kind_t::_sub: return a + (-1 * b);
        case op_kind_t::_mul: {
            if (is_zero(a) || is_zero(b)) return 0;
            if (is_one(a)) return b;
            if (is_one(b)) return a;
            if (a.is<var_t>() || a.is<linear_t>())
                return mul_linear(to_linear(a), b);
            if (b.is<var_t>() || b.is<linear_t>())
                return mul_linear(to_linear(b), a);
            auto *a_imm = a.as_ptr<int_imm_t>();
            auto *b_imm = b.as_ptr<int_imm_t>();
            ir_assert(!a_imm || !b_imm);
            ir_assert(a.is<const_var_t>() || b.is<const_var_t>());
            auto op = binary_op_t::make(op_kind_t::_mul, a, b);
            return linear_expr_ctx_t::get().add_const(op);
        }
        case op_kind_t::_div: {
            ir_assert(b.is<int_imm_t>());
            int b_value = to_int(b);
            if (auto *a_imm = a.as_ptr<int_imm_t>()) {
                int a_value = a_imm->value;
                ir_assert(b_value > 0 && a_value % b_value == 0);
                return expr_t(a_value / b_value);
            }
            if (a.is<linear_t>()) return div_linear(a, b_value);
            ir_assert(a.is<const_var_t>());
            auto ret = linear_expr_ctx_t::get().try_div(a, b_value);
            ir_assert(!ret.is_empty());
            return ret;
        }
        case op_kind_t::_div_up: {
            if (is_one(b)) return a;
            auto op = binary_op_t::make(op_kind_t::_div_up, a, b);
            return linear_expr_ctx_t::get().add_const(op);
        }
        default: ir_error_not_expected();
    }
    return expr_t();
}

// Returns the base and the increments of linear expression `expr` when
// incrementing `idxs[i]` by 1:
//     init = expr(idxs[i] = 0)
//     incs[i] = expr(idxs[i] + 1) - expr(idx[i]).
void split_to_linear(const expr_t &expr, const std::vector<expr_t> &idxs,
        expr_t &init, std::vector<expr_t> &incs);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
