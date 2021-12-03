/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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
#include "bf16_legalize.hpp"
#include <vector>
#include "../builder.hpp"
#include <util/utils.hpp>

namespace sc {

std::tuple<expr_c, expr_c> bf16_promote_impl_t::docast(
        const expr &orig_a, const expr &orig_b, bool *is_bfloat16) {
    auto a = dispatch(orig_a);
    auto b = dispatch(orig_b);
    *is_bfloat16 = false;
    if (a->dtype_.is_etype(sc_data_etype::BF16)) {
        COMPILE_ASSERT(utils::is_one_of(b->dtype_.type_code_,
                               sc_data_etype::BF16, sc_data_etype::F32),
                "bfloat16 should be calculated with bfloat16/f32");
        *is_bfloat16 = true;
    } else if (b->dtype_.is_etype(sc_data_etype::BF16)) {
        COMPILE_ASSERT(utils::is_one_of(a->dtype_.type_code_,
                               sc_data_etype::BF16, sc_data_etype::F32),
                "bfloat16 should be calculated with bfloat16");
        *is_bfloat16 = true;
    }
    if (*is_bfloat16) {
        sc_data_type_t fp32ty = sc_data_type_t::f32(a->dtype_.lanes_);
        a = a->dtype_.is_etype(sc_data_etype::BF16)
                ? builder::make_cast(fp32ty, a)
                : a;
        b = b->dtype_.is_etype(sc_data_etype::BF16)
                ? builder::make_cast(fp32ty, b)
                : b;
    }
    return std::make_tuple(a, b);
}

expr_c bf16_promote_impl_t::visit(binary_c v) {
    expr_c a, b;
    bool is_bfloat16 = false;
    std::tie(a, b) = docast(v->l_, v->r_, &is_bfloat16);
    bool changed = !a.ptr_same(v->l_) || !b.ptr_same(v->r_) || is_bfloat16;
    if (changed) {
        if (is_bfloat16) {
            return copy_attr(*v,
                    builder::make_cast(sc_data_type_t::bf16(v->dtype_.lanes_),
                            builder::remake_binary(a, b, v)));
        }
        return copy_attr(*v, builder::remake_binary(a, b, v));
    } else {
        return v;
    }
}

expr_c bf16_promote_impl_t::visit(cmp_c v) {
    expr_c a, b;
    bool is_bfloat16 = false;
    std::tie(a, b) = docast(v->l_, v->r_, &is_bfloat16);
    bool changed = !a.ptr_same(v->l_) || !b.ptr_same(v->r_) || is_bfloat16;
    if (changed) {
        return copy_attr(*v, builder::remake_binary(a, b, v));
    } else {
        return v;
    }
}

expr_c bf16_promote_impl_t::visit(intrin_call_c v) {
    std::vector<expr> args;
    bool is_bfloat16 = false;
    bool changed = false;
    switch (v->type_) {
        case intrin_type::min:
        case intrin_type::max:
        case intrin_type::abs:
        case intrin_type::round:
        case intrin_type::floor:
        case intrin_type::ceil:
        case intrin_type::exp:
        case intrin_type::sqrt:
        case intrin_type::rsqrt:
        case intrin_type::reduce_add:
        case intrin_type::reduce_mul:
        case intrin_type::fmadd:
        case intrin_type::unpack_low:
        case intrin_type::unpack_high:
        case intrin_type::isnan:
            for (size_t i = 0; i < v->args_.size(); i++) {
                auto in = dispatch(v->args_[i]);
                changed = changed || !in.ptr_same(v->args_[i]);
                if (in->dtype_.is_etype(sc_data_etype::BF16)) {
                    in = builder::make_cast(
                            sc_data_type_t::f32(in->dtype_.lanes_), in);
                    is_bfloat16 = true;
                }
                args.emplace_back(in.remove_const());
            }
            if (is_bfloat16) {
                for (size_t i = 0; i < args.size(); i++) {
                    COMPILE_ASSERT(args[i]->dtype_.is_etype(sc_data_etype::F32),
                            "All input args should be f32 from bf16.");
                }
            }
            break;
        case intrin_type::shuffle:
        case intrin_type::permute:
        case intrin_type::broadcast:
            for (size_t i = 0; i < v->args_.size(); i++) {
                auto in = dispatch(v->args_[i]);
                changed = changed || !in.ptr_same(v->args_[i]);
                args.emplace_back(in.remove_const());
            }
            break;
        case intrin_type::int_and:
        case intrin_type::int_or:
        case intrin_type::int_xor:
        case intrin_type::round_and_cast:
        case intrin_type::saturated_cast:
        case intrin_type::shl:
        case intrin_type::shr:
        case intrin_type::brgemm:
        case intrin_type::list_brgemm:
            for (size_t i = 0; i < v->args_.size(); i++) {
                auto in = dispatch(v->args_[i]);
                changed = changed || !in.ptr_same(v->args_[i]);
                COMPILE_ASSERT(!in->dtype_.is_etype(sc_data_etype::BF16),
                        "Intrin type " << v->type_
                                       << " does not support bf16 args");
                args.emplace_back(in.remove_const());
            }
            break;
        case intrin_type::reinterpret: break;
        default:
            COMPILE_ASSERT(false, "Unsupport BF16 intrin type: " << v->type_);
    }

    changed = changed || is_bfloat16;
    if (changed) {
        if (is_bfloat16) {
            if (v->type_ == intrin_type::reduce_add) {
                return copy_attr(*v,
                        builder::make_cast(sc_data_type_t::bf16(),
                                builder::remake_intrin_call(v, args)));
            } else {
                return copy_attr(*v,
                        builder::make_cast(
                                sc_data_type_t::bf16(args[0]->dtype_.lanes_),
                                builder::remake_intrin_call(v, args)));
            }
        } else {
            return copy_attr(*v, builder::remake_intrin_call(v, args));
        }
    } else {
        return v;
    }
}

expr_c bf16_cast_elimination_impl_t::visit(cast_c v) {
    auto in = dispatch(v->in_);
    if (v->dtype_.is_etype(sc_data_etype::F32)) {
        if (in.isa<cast_c>()) {
            auto inin = in.static_as<cast_c>();
            if (inin->dtype_.is_etype(sc_data_etype::BF16)
                    && inin->in_->dtype_.is_etype(sc_data_etype::F32)) {
                return inin->in_;
            }
        } else if (in.isa<var>() && cvt_map_.find(in) != cvt_map_.end()) {
            return cvt_map_[in];
        }
    }
    bool changed = !in.ptr_same(v->in_);
    if (changed) {
        return copy_attr(*v, builder::make_cast(v->dtype_, in));
    } else {
        return v;
    }
}

stmt_c bf16_cast_elimination_impl_t::visit(define_c v) {
    if (v->var_.isa<var>() && v->var_->dtype_.is_etype(sc_data_etype::BF16)) {
        expr_c newv = copy_attr(*v->var_,
                builder::make_var(sc_data_type_t::f32(v->var_->dtype_.lanes_),
                        v->var_.static_as<var>()->name_));
        cvt_map_.insert({v->var_, newv});
        return copy_attr(*v, builder::make_var_tensor_def_unattached(newv));
    }
    return ir_visitor_t::visit(std::move(v));
}

stmt_c bf16_cast_elimination_impl_t::visit(assign_c v) {
    auto var = dispatch(v->var_);
    auto val = dispatch(v->value_);
    bool changed = !var.ptr_same(v->var_) || !val.ptr_same(v->value_);
    if (cvt_map_.find(var) != cvt_map_.end()) {
        assert(val->dtype_.is_etype(sc_data_etype::BF16)
                || val->dtype_.is_etype(sc_data_etype::U16));
        var = cvt_map_[var];
        if (val.isa<cast_c>()) {
            val = val.static_as<cast_c>()->in_;
        } else {
            val = builder::make_cast(
                    sc_data_type_t::f32(val->dtype_.lanes_), val);
        }
        changed = true;
    }
    if (cvt_map_.find(val) != cvt_map_.end()) {
        val = builder::make_cast(
                sc_data_type_t::bf16(val->dtype_.lanes_), cvt_map_[val]);
        changed = true;
    }
    if (changed) {
        return copy_attr(*v, builder::make_assign_unattached(var, val));
    }
    return v;
}

stmt_c bf16_cast_elimination_impl_t::visit(returns_c v) {
    if (v->value_.isa<var>()) {
        COMPILE_ASSERT(cvt_map_.find(v->value_) == cvt_map_.end(),
                "Not support return a bf16 local buffer now");
    }
    return ir_visitor_t::visit(v);
}

func_c bf16_legalize_t::operator()(func_c f) {
    bf16_promote_impl_t promote_pass;
    bf16_cast_elimination_impl_t elimination_pass;
    f = promote_pass.dispatch(f);
    f = elimination_pass.dispatch(f);
    return f;
}

stmt_c bf16_legalize_t::operator()(stmt_c f) {
    bf16_promote_impl_t promote_pass;
    bf16_cast_elimination_impl_t elimination_pass;
    f = promote_pass.dispatch(f);
    f = elimination_pass.dispatch(f);
    return f;
}

expr_c bf16_legalize_t::operator()(expr_c f) {
    bf16_promote_impl_t promote_pass;
    bf16_cast_elimination_impl_t elimination_pass;
    f = promote_pass.dispatch(f);
    f = elimination_pass.dispatch(f);
    return f;
}
} // namespace sc
