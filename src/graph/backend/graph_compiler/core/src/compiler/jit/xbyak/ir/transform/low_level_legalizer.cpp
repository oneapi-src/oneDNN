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

#include <atomic>
#include <string>
#include <utility>
#include <vector>

#include <compiler/ir/builder.hpp>
#include <compiler/ir/visitor.hpp>

#include "low_level_legalizer.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class low_level_legalizer_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    low_level_legalizer_impl_t(const runtime::target_machine_t &target_machine)
        : max_f32_lanes_(target_machine.get_device_flags().get_max_vector_lanes(
                sc_data_etype::F32)) {}

    expr_c visit(tensor_c v) override {
        // avoid dispatch into for loop index dependent tensor
        return v;
    }

    expr_c visit(cast_c v) override {
        auto ret = ir_visitor_t::visit(std::move(v));
        assert(ret.isa<cast>());
        auto vv = ret.static_as<cast_c>();
        return transform_f32_cast(vv->dtype_, vv->in_);
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
        // TODO(longsheng): reduce no need to cast to f32
        auto can_cast_to_f32 = !dst_dtype.is_etype(sc_data_etype::F32)
                && (dst_dtype.lanes_ <= max_f32_lanes_);
        switch (vv->type_) {
            case intrin_type::abs: {
                // float and bf16
                if (dst_dtype.is_etype(sc_data_etype::F32)) {
                    return transform_fabs(vv->args_[0], UINT64_C(0x7FFFFFFF));
                } else if (dst_dtype.is_etype(sc_data_etype::BF16)
                        || dst_dtype.is_etype(sc_data_etype::F16)) {
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
                if (can_cast_to_f32) {
                    return transform_reduce(
                            vv->args_[0], dst_dtype, &builder::make_reduce_add);
                }
            } break;
            case intrin_type::reduce_mul: {
                if (can_cast_to_f32) {
                    return transform_reduce(
                            vv->args_[0], dst_dtype, &builder::make_reduce_mul);
                }
            } break;
            case intrin_type::reduce_min: {
                if (can_cast_to_f32) {
                    return transform_reduce(
                            vv->args_[0], dst_dtype, &builder::make_reduce_min);
                }
            } break;
            case intrin_type::reduce_max: {
                if (can_cast_to_f32) {
                    return transform_reduce(
                            vv->args_[0], dst_dtype, &builder::make_reduce_max);
                }
            } break;
            case intrin_type::gather: {
                if (vv->dtype_.lanes_ == 1) {
                    return builder::make_indexing(vv->args_[0], {vv->args_[1]});
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
        return transform_f32_cast(dst_dtype, //
                make_reduce(transform_f32_cast(type_to_f32, src)));
    }

    expr_c transform_f32_cast(sc_data_type_t dst_dtype, const expr &src) {
        const auto src_dtype = src->dtype_;
        const auto is_f32_int8_cast = [](const sc_data_type_t &type_dst,
                                              const sc_data_type_t &type_src) {
            return (type_dst.type_code_ == sc_data_etype::U8
                           || type_dst.type_code_ == sc_data_etype::S8)
                    && type_src.type_code_ == sc_data_etype::F32;
        };
        const auto is_f16_scalar_cast
                = [](const sc_data_type_t &type_dst,
                          const sc_data_type_t &type_src) {
                      return (type_dst.type_code_ == sc_data_etype::F16)
                              && (utils::is_one_of(type_src.type_code_,
                                      sc_data_etype::U16, sc_data_etype::S8,
                                      sc_data_etype::U8))
                              && type_dst.lanes_ == 1;
                  };
        const auto convert_s32_transform
                = [](const sc_data_type_t &dst_dtype,
                          const sc_data_type_t &src_dtype, const expr &src) {
                      return builder::make_cast(dst_dtype,
                              builder::make_cast(
                                      sc_data_type_t::s32(src_dtype.lanes_),
                                      src));
                  };
        if (is_f32_int8_cast(dst_dtype, src_dtype)
                || is_f32_int8_cast(src_dtype, dst_dtype)) {
            // int8 to f32 must cast to s32 first
            return convert_s32_transform(dst_dtype, src_dtype, src);
        } else if (is_f16_scalar_cast(dst_dtype, src_dtype)
                || is_f16_scalar_cast(src_dtype, dst_dtype)) {
            // other dtype to f16 must cast to s32 first
            return convert_s32_transform(dst_dtype, src_dtype, src);
        } else {
            return builder::make_cast(dst_dtype, src);
        }
    }

private:
    const uint16_t max_f32_lanes_;
};

func_c low_level_legalizer_t::operator()(func_c v) {
    low_level_legalizer_impl_t low_level_legalizer(target_machine_);
    return low_level_legalizer.dispatch(std::move(v));
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
