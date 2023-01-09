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

#include <compiler/ir/builder.hpp>
#include <compiler/ir/visitor.hpp>

#include "low_level_legalizer.hpp"

namespace sc {
namespace sc_xbyak {

class low_level_legalizer_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    expr_c visit(cast_c v) override {
        auto vv = ir_visitor_t::visit(std::move(v)).static_as<cast_c>();
        const auto src_dtype = vv->in_->dtype_;
        const auto dst_dtype = vv->dtype_;
        const auto is_f32_int8_cast = [](const sc_data_type_t &type_dst,
                                              const sc_data_type_t &type_src) {
            return (type_dst.type_code_ == sc_data_etype::U8
                           || type_dst.type_code_ == sc_data_etype::S8)
                    && type_src.type_code_ == sc_data_etype::F32;
        };
        if (is_f32_int8_cast(dst_dtype, src_dtype)
                || is_f32_int8_cast(src_dtype, dst_dtype)) {
            // int8 to f32 must cast to s32 first
            return builder::make_cast(dst_dtype,
                    builder::make_cast(
                            sc_data_type_t::s32(src_dtype.lanes_), vv->in_));
        }
        return vv;
    }

    stmt_c visit(assign_c v) override {
        auto vv = ir_visitor_t::visit(std::move(v)).static_as<assign_c>();
        auto &var = vv->var_;
        auto &value = vv->value_;
        if (value.isa<tensor>()) {
            // Must cast tensor to dst dtype before assign
            return make_stmt<assign_node_t>(
                    var.remove_const(), builder::make_cast(var->dtype_, value));
        }
        return vv;
    }

    expr_c visit(intrin_call_c v) override {
        auto vv = ir_visitor_t::visit(std::move(v)).static_as<intrin_call_c>();
        auto dst_dtype = vv->dtype_;
        switch (vv->type_) {
            case intrin_type::abs: {
                // float and bf16
                if (dst_dtype.is_etype(sc_data_etype::F32)) {
                    return transform_fabs(vv->args_[0], UINT64_C(0x7FFFFFFF));
                } else if (dst_dtype.is_etype(sc_data_etype::BF16)) {
                    return transform_fabs(vv->args_[0], UINT64_C(0x7FFF));
                }
            } break;
            case intrin_type::rsqrt: {
                if (dst_dtype == datatypes::f32) {
                    return builder::make_constant(1.f)
                            / builder::make_sqrt(vv->args_[0]);
                }
            } break;
            case intrin_type::reduce_add: {
                if (!dst_dtype.is_etype(sc_data_etype::F32)) {
                    return transform_reduce(
                            vv->args_[0], dst_dtype, &builder::make_reduce_add);
                }
            } break;
            case intrin_type::reduce_mul: {
                if (!dst_dtype.is_etype(sc_data_etype::F32)) {
                    return transform_reduce(
                            vv->args_[0], dst_dtype, &builder::make_reduce_mul);
                }
            } break;
            default: break;
        }
        return vv;
    }

    using make_unary_f = expr (*)(const expr_c &);
    using make_binary_f = expr (*)(const expr_c &, const expr_c &);

    expr_c transform_fabs(const expr &src, const union_val &val) {
        return builder::make_int_and(
                src, builder::make_constant({val}, src->dtype_));
    }

    expr_c transform_reduce(const expr &src, sc_data_type_t dst_dtype,
            make_unary_f make_reduce) {
        auto type_to_f32 = sc_data_type_t::f32(src->dtype_.lanes_);
        return builder::make_cast(dst_dtype, //
                make_reduce(builder::make_cast(type_to_f32, src)));
    }
};

func_c low_level_legalizer_t::operator()(func_c v) {
    low_level_legalizer_impl_t low_level_legalizer;

    return low_level_legalizer.dispatch(std::move(v));
}

} // namespace sc_xbyak
} // namespace sc
