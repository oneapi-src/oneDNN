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

#include <utility>
#include <vector>

#include <compiler/ir/builder.hpp>
#include <compiler/ir/visitor.hpp>
#include <compiler/jit/xbyak/ir/util/utils.hpp>
#include <util/any_map.hpp>
#include <util/utils.hpp>

#include "constant_optimizer.hpp"

#define HAS_KEY_ENCODE(EXPR) \
    ((EXPR)->attr_ && (EXPR)->attr_->has_key(attr_keys::force_simd_encode))

#define HAS_KEY_LOAD(EXPR) \
    ((EXPR)->attr_ && (EXPR)->attr_->has_key(attr_keys::load_simd_value))

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class constant_optimizer_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    stmt_c visit(define_c v) override {
        if (v->init_.defined() && v->init_.isa<constant>()
                && is_x86_simd(v->init_->dtype_)) {
            v->init_->attr().set(attr_keys::load_simd_value, true);
        }
        return ir_visitor_t::visit(std::move(v));
    }

    stmt_c visit(assign_c v) override {
        if (v->value_.isa<constant>() && is_x86_simd(v->value_->dtype_)) {
            v->value_->attr().set(attr_keys::load_simd_value, true);
        }
        return ir_visitor_t::visit(std::move(v));
    }

    expr_c visit(tensor_c v) override {
        // avoid constant_optimizer for loop index dependent tensor
        return v;
    }

    expr_c visit(constant_c v) override {
        const auto &dtype = v->dtype_;
        const auto &value = v->value_;
        // Transform multi-lane constant to broadcasted single-lane constant
        if (!HAS_KEY_ENCODE(v) && dtype.lanes_ > 1 && value.size() == 1) {
            auto new_const = single_lane_constant(v);
            new_const->attr().set(attr_keys::force_simd_encode, true);
            return builder::make_broadcast(new_const, dtype.lanes_);
        } else if (HAS_KEY_LOAD(v)) {
            return make_expr<intrin_call_node>(intrin_type::load_const_mem,
                    std::vector<expr> {v.remove_const()}, any_map_t());
        } else {
            return v;
        }
    }

    expr_c visit(mul_c v) override {
        // If constant is 2^N, transform mul(lhs, 2^n) to shl(rhs, n)
        auto transform_const_mul = [this](const expr &lhs, const expr &rhs) {
            auto const_rhs = rhs.static_as<constant>();
            if (is_power_of_2(const_rhs)) {
                auto imm = power_of_2_exponent(const_rhs);
                return builder::make_shl(lhs, imm);
            } else {
                return builder::make_mul(lhs, rhs);
            }
        };
        // Make constant in multiply always positioned in rhs
        auto vv = ir_visitor_t::visit(std::move(v)).static_as<mul_c>();
        if (vv->r_.isa<constant>()) {
            return transform_const_mul(vv->l_, vv->r_);
        } else if (vv->l_.isa<constant>()) {
            return transform_const_mul(vv->r_, vv->l_);
        } else {
            return vv;
        }
    }

    expr_c visit(div_c v) override {
        // Transform unsigned div if divisor is a const of power of 2
        if (v->r_.isa<constant>()
                && v->r_->dtype_.is_etype(sc_data_etype::S32)) {
            v->r_->attr().set(attr_keys::force_simd_encode, false);
        }
        auto vv = ir_visitor_t::visit(std::move(v)).static_as<div_c>();
        if (vv->r_.isa<constant>()) {
            auto const_rhs = vv->r_.static_as<constant>();
            bool transform_pow_2
                    = utils::is_one_of(vv->l_->dtype_, datatypes::u8,
                              datatypes::u16, datatypes::u32, datatypes::index)
                    && is_power_of_2(const_rhs);
            // If constant is 2^N, transform div(x, 2^n) to shr(x, n)
            if (transform_pow_2) {
                auto imm = power_of_2_exponent(const_rhs);
                return builder::make_shr(vv->l_, imm);
            }
        }
        return vv;
    }

    expr_c visit(mod_c v) override {
        // Transform unsigned mod if divisor is a const of power of 2
        auto vv = ir_visitor_t::visit(std::move(v)).static_as<mod_c>();
        if (vv->r_.isa<constant>()) {
            auto const_rhs = vv->r_.static_as<constant>();
            bool transform_pow_2
                    = utils::is_one_of(vv->l_->dtype_, datatypes::u8,
                              datatypes::u16, datatypes::u32, datatypes::index)
                    && is_power_of_2(const_rhs);
            // If constant is 2^N, transform mod(x, 2^n) to int_and(x, 2^n - 1)
            if (transform_pow_2) {
                auto imm = power_of_2_minus_1(const_rhs);
                return builder::make_int_and(vv->l_, imm);
            }
        }
        return vv;
    }

    expr_c visit(select_c v) override {
        // Keep zero blend for potential simd optimization
        if (v->r_.isa<constant>()) {
            auto const_rhs = v->r_.static_as<constant>();
            if (is_zero(const_rhs)) {
                const_rhs->attr().set(attr_keys::force_simd_encode, false);
            }
        }
        return ir_visitor_t::visit(std::move(v));
    }

    expr_c visit(intrin_call_c v) override {
        // convert same value constant simd shift to scalar value
        // so xbyak can use imm in shift function
        auto transform_shift = [this](intrin_call_c v) -> expr_c {
            assert(v->args_.size() == 2);
            auto &l = v->args_[0];
            auto &r = v->args_[1];
            if (r.isa<constant>()) {
                auto const_rhs = r.static_as<constant>();
                if (const_rhs->value_.size() == 1
                        && const_rhs->dtype_.lanes_ > 1) {
                    auto lhs = ir_visitor_t::dispatch(l).remove_const();
                    auto rhs = single_lane_constant(const_rhs);
                    return make_expr<intrin_call_node>(v->type_,
                            std::vector<expr> {lhs, rhs}, any_map_t());
                }
            }
            return ir_visitor_t::visit(std::move(v));
        };
        // mark constant broadcast value as simd encode
        auto transform_broadcast = [this](intrin_call_c v) -> expr_c {
            assert(v->args_.size() == 1);
            auto &r = v->args_[0];
            if (r.isa<constant>()) {
                r->attr().set(attr_keys::force_simd_encode, true);
            }
            return ir_visitor_t::visit(std::move(v));
        };
        // Transform intrin_call node with const value
        switch (v->type_) {
            case intrin_type::shl:
            case intrin_type::shr: {
                return transform_shift(std::move(v));
            } break;
            case intrin_type::broadcast: {
                return transform_broadcast(std::move(v));
            } break;
            default: break;
        }
        return ir_visitor_t::visit(std::move(v));
    }

    // TODO(XXX): Maybe create a dedicated x86 legalize pass in the future
    // make sure x86 imm always rhs, just like expr_c visit(mul_c v)
    expr_c visit(cmp_c v) override {
        auto make_swap_cmp = [&](sc_expr_type t, const expr_c &lhs,
                                     const expr_c &rhs) {
            using sc_etype = sc_expr_type;
            switch (t) {
                case sc_etype::cmp_eq: return builder::make_cmp_eq(rhs, lhs);
                case sc_etype::cmp_lt: return builder::make_cmp_gt(rhs, lhs);
                case sc_etype::cmp_le: return builder::make_cmp_ge(rhs, lhs);
                case sc_etype::cmp_ne: return builder::make_cmp_ne(rhs, lhs);
                case sc_etype::cmp_ge: return builder::make_cmp_le(rhs, lhs);
                case sc_etype::cmp_gt: return builder::make_cmp_lt(rhs, lhs);
                default: COMPILE_ASSERT(false, "Invalid compare type: " << t);
            }
            return expr();
        };
        // swap cmp(l, r) to equivalent cmp(r, l)
        if (v->l_.isa<constant>()) {
            return make_swap_cmp(
                    v->node_type_, v->l_, ir_visitor_t::dispatch(v->r_));
        } else {
            return ir_visitor_t::visit(std::move(v));
        }
    }

    bool is_zero(const constant &v) {
        uint64_t val = v->value_[0].u64;
        return (v->value_.size() == 1) && (val == 0);
    }

    bool is_power_of_2(const constant &v) {
        const auto dtype = v->dtype_;
        if (utils::is_one_of(dtype, datatypes::u8, datatypes::u16,
                    datatypes::u32, datatypes::index)) {
            uint64_t val = v->value_[0].u64;
            return utils::is_power_of_2(val);
        }
        return false;
    }

    expr power_of_2_exponent(const constant_c &v) {
        // Calculate integer log2(2^n)
        uint64_t val = v->value_[0].u64;
        return builder::make_constant({uint64_t(utils::ctz(val))}, v->dtype_);
    }

    expr power_of_2_minus_1(const constant_c &v) {
        // Calculate integer 2^n - 1
        uint64_t val = v->value_[0].u64;
        return builder::make_constant({val - 1}, v->dtype_);
    }

    expr single_lane_constant(const constant_c &v) {
        // Transform simd constant to scalar constant
        uint64_t val = v->value_[0].u64;
        return builder::make_constant(
                {val}, sc_data_type_t(v->dtype_.type_code_, 1));
    }
};

func_c constant_optimizer_t::operator()(func_c v) {
    constant_optimizer_impl_t constant_optimizer;

    return constant_optimizer.dispatch(std::move(v));
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
