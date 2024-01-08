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

#include "gpu/jit/ir/linear_expr.hpp"

#include "common/math_utils.hpp"
#include "gpu/jit/pass/simplify.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

std::vector<expr_t> op_split(op_kind_t kind, const expr_t &e) {
    auto *op = e.as_ptr<binary_op_t>();
    if (!op || op->op_kind != kind) return {e};
    auto a_args = op_split(kind, op->a);
    auto b_args = op_split(kind, op->b);
    std::vector<expr_t> args;
    args.insert(args.end(), a_args.begin(), a_args.end());
    args.insert(args.end(), b_args.begin(), b_args.end());
    return args;
}

expr_t op_combine(op_kind_t kind, const std::vector<expr_t> &args) {
    bool is_add = (kind == op_kind_t::_add);
    bool is_mul = (kind == op_kind_t::_mul);
    ir_assert(is_add || is_mul);
    expr_t ret = (is_add ? 0 : 1);
    for (auto &a : args) {
        if (a.is_empty()) continue;
        ret = binary_op_t::make(kind, ret, a);
    }
    return simplify_rewrite(ret);
}

bool is_const_expr(const expr_t &e) {
    if (e.is<const_var_t>()) return true;
    if (e.is<var_t>()) return false;
    if (e.is<int_imm_t>()) return true;
    if (auto *op = e.as_ptr<unary_op_t>()) return is_const_expr(op->a);
    if (auto *op = e.as_ptr<binary_op_t>()) {
        return is_const_expr(op->a) && is_const_expr(op->b);
    }
    ir_error_not_expected() << e;
    return false;
}

// Expands multiplications to put them inside additions. Does not expand
// multiplications resulting in multiple terms referencing the same
// non-constant expression.
// Example 1:
//     2 * (a * v1 + b * v2) -> (2 * a) * v1 + (2 * b) * v2
// Example 2. Skip expansion to avoid multiple terms with v1:
//    (a + b) * v1 -> (a + b) * v1
class linear_normalize_expander_t : public ir_mutator_t {
public:
    object_t _mutate(const binary_op_t &_obj) override {
        auto obj = ir_mutator_t::_mutate(_obj);
        auto &op = obj.as<binary_op_t>();
        if (op.op_kind != op_kind_t::_mul) return obj;

        auto a = op.a;
        auto b = op.b;
        if (!is_const_expr(b)) std::swap(a, b);
        ir_assert(is_const_expr(b));
        auto a_args = op_split(op_kind_t::_add, a);
        auto b_args = op_split(op_kind_t::_add, b);
        expr_t ret = 0;
        for (auto &a : a_args) {
            if (!is_const_expr(a)) {
                // Do not expand for non-const a.
                ret += a * op_combine(op_kind_t::_add, b_args);
                continue;
            }
            for (auto &b : b_args) {
                ret += a * b;
            }
        }
        return simplify_rewrite(ret);
    }
};

expr_t linear_normalize_reduce(const expr_t &e,
        object_eq_map_t<expr_t, int> factors, int const_factor) {
    auto mul_args = op_split(op_kind_t::_mul, e);
    for (auto &ma : mul_args) {
        if (is_const(ma)) {
            int ma_const = to_cpp<int>(ma);
            int div = math::gcd(const_factor, ma_const);
            const_factor /= div;
            ma = ma_const / div;
            continue;
        }
        auto it = factors.find(ma);
        if (it == factors.end() || it->second == 0) continue;
        factors[ma]--;
        ma = expr_t();
    }
    ir_assert(const_factor == 1);
    for (auto &kv : factors) {
        ir_assert(kv.second == 0);
    }
    return op_combine(op_kind_t::_mul, mul_args);
}

object_eq_map_t<expr_t, int> find_common_factors(
        const std::vector<expr_t> &add_args, int &const_factor) {
    const_factor = 1;
    object_eq_map_t<expr_t, int> common;
    for (int i = 0; i < (int)add_args.size(); i++) {
        auto mul_args = op_split(op_kind_t::_mul, add_args[i]);
        if (i == 0) {
            for (auto &ma : mul_args) {
                if (is_const(ma)) {
                    const_factor *= to_cpp<int>(ma);
                    continue;
                }
                common[ma]++;
            }
        } else {
            auto i_common = common;
            int i_const_factor = 1;
            common.clear();
            for (auto &ma : mul_args) {
                if (is_const(ma)) {
                    i_const_factor *= to_cpp<int>(ma);
                    continue;
                }
                auto it = i_common.find(ma);
                if (it == i_common.end() || it->second == 0) continue;
                it->second--;
                common[ma]++;
            }
            const_factor = math::gcd(const_factor, i_const_factor);
        }
    }
    return common;
}

// Factors out common factors for constant expressions. This simplifies
// division implementation for linear expression coefficients.
// Example: (c * a + 2 * c * b) -> c * (a + 2 * b)
expr_t linear_normalize_const_factor_out(const expr_t &_e) {
    auto e = simplify_rewrite(_e);
    ir_assert(is_const_expr(e));
    auto add_args = op_split(op_kind_t::_add, e);
    if (add_args.size() <= 1) return e;

    // Find common factors of all summands.
    int const_factor;
    auto common = find_common_factors(add_args, const_factor);
    if (common.empty() && const_factor == 1) return e;

    // Reduce summands by the found factors.
    for (auto &a : add_args) {
        a = linear_normalize_reduce(a, common, const_factor);
    }

    std::vector<expr_t> v_common;
    v_common.push_back(const_factor);
    for (auto &kv : common) {
        for (int i = 0; i < kv.second; i++)
            v_common.push_back(kv.first);
    }
    auto a = op_combine(op_kind_t::_mul, v_common);
    auto b = op_combine(op_kind_t::_add, add_args);
    return simplify_rewrite(a * b);
}

std::pair<expr_t, expr_t> split_to_coef_and_index(const expr_t &e) {
    auto args = op_split(op_kind_t::_mul, e);
    expr_t coef = 1;
    expr_t idx;
    for (auto &a : args) {
        if (a.is<var_t>()) {
            ir_assert(idx.is_empty());
            idx = a;
        } else if (is_const_expr(a)) {
            coef *= a;
        } else {
            ir_error_not_expected() << a;
        }
    }
    return std::make_pair(coef, idx);
}

expr_t to_linear(const expr_t &_e) {
    auto e = linear_normalize_expander_t().mutate(_e);
    auto add_args = op_split(op_kind_t::_add, e);
    expr_t c = 0;
    std::vector<expr_t> u;
    std::vector<expr_t> v;
    for (auto &a : add_args) {
        auto p = split_to_coef_and_index(a);
        if (p.second.is_empty()) {
            c += p.first;
            continue;
        }
        u.push_back(linear_normalize_const_factor_out(p.first));
        v.push_back(p.second);
    }
    c = linear_normalize_const_factor_out(c);
    return linear_t::make(c, u, v);
}

// Linear coefficient of a linear expression, implements logic to work with
// factors for division/modulus.
class linear_coef_t {
public:
    explicit linear_coef_t(const expr_t &value = expr_t(0)) : imm_(1) {
        auto mul_args = op_split(op_kind_t::_mul, value);
        for (auto &a : mul_args)
            mul_impl(a);
    }

    bool is_zero() const { return factors_.empty() && imm_ == 0; }
    int imm() const { return imm_; }
    void set_imm(int imm) { imm_ = imm; }

    linear_coef_t &operator/=(int factor) {
        ir_assert(imm_ % factor == 0);
        imm_ /= factor;
        return *this;
    }

    linear_coef_t &intersect(const linear_coef_t &other) {
        if (other.is_zero()) return *this;
        if (is_zero()) {
            *this = other;
            return *this;
        }
        imm_ = math::gcd(imm_, other.imm_);
        auto lhs = op_combine(op_kind_t::_mul, factors_);
        auto rhs = op_combine(op_kind_t::_mul, other.factors_);
        int const_factor = 1;
        auto common = find_common_factors({lhs, rhs}, const_factor);
        ir_assert(const_factor == 1);
        factors_.clear();
        for (auto &kv : common) {
            for (int i = 0; i < kv.second; i++)
                factors_.push_back(kv.first);
        }
        return *this;
    }

    expr_t to_expr() const {
        expr_t ret = imm_;
        for (auto &f : factors_)
            ret *= f;
        return simplify_rewrite(ret);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "imm: " << imm_;
        if (factors_.empty()) return oss.str();
        oss << std::endl << "factors:";
        for (auto &f : factors_) {
            oss << std::endl;
            oss << " " << f;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

    static expr_t div(const expr_t &e, int factor) {
        linear_coef_t coef(e);
        coef /= factor;
        return coef.to_expr();
    }

    static std::vector<expr_t> div(const std::vector<expr_t> &v, int factor) {
        std::vector<expr_t> ret;
        for (auto &e : v)
            ret.push_back(div(e, factor));
        return ret;
    }

private:
    void mul_impl(const expr_t &e) {
        ir_assert(is_const_expr(e)) << e;
        if (is_const(e)) {
            imm_ *= to_cpp<int>(e);
            if (imm_ == 0) factors_.clear();
            return;
        }
        factors_.push_back(e);
    }

    int imm_ = 0;
    std::vector<expr_t> factors_;
};

int linear_max_pow2_divisor_impl(const expr_t &e) {
    const int large_pow2 = (1 << 20);
    if (is_zero(e)) return large_pow2;
    if (e.is<const_var_t>()) return 1;
    if (e.is<var_t>()) return 1;
    if (auto *imm = e.as_ptr<int_imm_t>())
        return ir_utils::max_pow2_divisor(imm->value);
    if (auto *op = e.as_ptr<unary_op_t>()) {
        return linear_max_pow2_divisor_impl(op->a);
    }
    if (auto *op = e.as_ptr<binary_op_t>()) {
        switch (op->op_kind) {
            case op_kind_t::_add:
            case op_kind_t::_sub: {
                auto a = linear_max_pow2_divisor_impl(op->a);
                auto b = linear_max_pow2_divisor_impl(op->b);
                return math::gcd(a, b);
            }
            case op_kind_t::_mul: {
                auto a = linear_max_pow2_divisor_impl(op->a);
                auto b = linear_max_pow2_divisor_impl(op->b);
                return a * b;
            }
            case op_kind_t::_div:
            case op_kind_t::_div_up:
            case op_kind_t::_mod: return 1;
            default: ir_error_not_expected() << e;
        }
    }
    ir_error_not_expected() << e;
    return 1;
}

int linear_max_pow2_divisor(const expr_t &e) {
    auto _linear = to_linear(e);
    auto &linear = _linear.as<linear_t>();
    int ret = linear_max_pow2_divisor_impl(linear.c);
    for (auto &u : linear.u_vec)
        ret = math::gcd(ret, linear_max_pow2_divisor_impl(u));
    return ret;
}

expr_t linear_div(const expr_t &e, int factor) {
    auto _linear = to_linear(e);
    auto &linear = _linear.as<linear_t>();
    auto c = linear_coef_t::div(linear.c, factor);
    auto u_vec = linear_coef_t::div(linear.u_vec, factor);
    auto v_vec = linear.v_vec;
    return linear_t::to_expr(c, u_vec, v_vec);
}

expr_t linear_mod_impl(const expr_t &e, int factor) {
    if (factor == 1) return 0;
    if (is_const(e)) return to_cpp<int>(e) % factor;
    if (e.is<const_var_t>()) return e;
    if (auto *op = e.as_ptr<binary_op_t>()) {
        auto a = linear_mod_impl(op->a, factor);
        auto b = linear_mod_impl(op->b, factor);
        switch (op->op_kind) {
            case op_kind_t::_add:
                if (is_zero(a)) return b;
                if (is_zero(b)) return a;
                return simplify_rewrite(a + b);
            case op_kind_t::_mul:
                if (is_zero(a)) return 0;
                if (is_zero(b)) return 0;
                return simplify_rewrite(a * b);
            default: break;
        }
    }
    ir_error_not_expected() << e;
    return 1;
}

expr_t linear_mod(const expr_t &e, int factor) {
    ir_assert(factor > 0);
    if (factor == 1) return 0;
    auto _linear = to_linear(e);
    auto &linear = _linear.as<linear_t>();
    std::vector<linear_coef_t> coefs;
    coefs.emplace_back(linear.c);
    for (auto &u : linear.u_vec)
        coefs.emplace_back(u);
    linear_coef_t common;
    for (auto &c : coefs) {
        if (c.imm() % factor == 0) continue;
        common.intersect(c);
    }
    if (common.imm() == 0) return 0;
    int div = math::gcd(common.imm(), factor);
    int new_factor = factor / div;
    common.set_imm(1);
    return linear_mod_impl(common.to_expr(), new_factor) % new_factor;
}

// Updates the base and the increment of linear expression `expr` when
// incrementing `idx` by 1:
//     inc_idx = expr(idx + 1) - expr(idx).
//     inc += inc_idx
//     return expr(idx = 0)
expr_t split_to_linear_impl(
        const expr_t &expr, const expr_t &idx, expr_t &inc) {
    if (auto *linear = expr.as_ptr<linear_t>()) {
        for (int i = 0; i < linear->nargs(); i++) {
            if (linear->v_vec[i].impl() == idx.impl()) {
                auto u_vec = linear->u_vec;
                auto v_vec = linear->v_vec;
                u_vec.erase(u_vec.begin() + i);
                v_vec.erase(v_vec.begin() + i);
                inc = linear->u_vec[i];
                return linear_t::make(linear->c, u_vec, v_vec);
            }
        }
        inc = expr_t(0);
        return expr;
    }

    ir_error_not_expected() << expr;
    return expr;
}

void split_to_linear(const expr_t &expr, const std::vector<expr_t> &idxs,
        expr_t &init, std::vector<expr_t> &incs) {
    incs = std::vector<expr_t>(idxs.size());
    init = to_linear(expr);
    for (size_t i = 0; i < idxs.size(); i++) {
        init = split_to_linear_impl(init, idxs[i], incs[i]);
    }
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
