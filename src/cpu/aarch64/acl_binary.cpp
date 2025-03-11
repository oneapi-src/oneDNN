/*******************************************************************************
* Copyright 2022, 2024 Arm Ltd. and affiliates
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

#include "acl_binary.hpp"

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/experimental/operators/CpuAdd.h"
#include "arm_compute/runtime/experimental/operators/CpuElementwise.h"
#include "arm_compute/runtime/experimental/operators/CpuMul.h"
#include "arm_compute/runtime/experimental/operators/CpuSub.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_binary_t::pd_t::init(engine_t *engine) {
    using namespace acl_utils;

    // Only support f16/f32/s32 for now
    data_type_t ddt = dst_md(0)->data_type;
    if (!utils::one_of(ddt, data_type::f16, data_type::f32, data_type::s32))
        return status::unimplemented;

    // Only support src and dst all matching for now
    if (ddt != src_md(0)->data_type || src_md(1)->data_type != ddt)
        return status::unimplemented;

    // Sets the memory format of dst from any to src_md(0) blocking desc
    CHECK(set_default_params());

    if (!attr()->has_default_values()) return status::unimplemented;

    asp_.alg = desc()->alg_kind;

    // All the algorithms we support
    if (!utils::one_of(asp_.alg, alg_kind::binary_add, alg_kind::binary_sub,
                alg_kind::binary_mul, alg_kind::binary_div,
                alg_kind::binary_max, alg_kind::binary_min))
        return status::unimplemented;

    // s32 div in ACL does not round as oneDNN expects
    if (ddt == data_type::s32 && asp_.alg == alg_kind::binary_div)
        return status::unimplemented;

    // ACL pointwise arithmetic operators assume that the innermost
    // dimensions are dense for src0, src1 and dst. Reordering the
    // logical dimensions by stride does this (if reordered_dims >= 1 )
    // and also makes memory accesses contiguous in ACL (without any
    // data reordering).
    memory_desc_t src_d0_permed, src_d1_permed, dst_d_permed;
    int reordered_dims = reorder_dimensions_by_stride(
            {&src_d0_permed, &src_d1_permed, &dst_d_permed},
            {src_md(0), src_md(1), dst_md()});
    if (reordered_dims < 1) return status::unimplemented;

    // Create ACL tensor infos with permuted descs
    CHECK(tensor_info(asp_.src0_info, src_d0_permed));
    CHECK(tensor_info(asp_.src1_info, src_d1_permed));
    CHECK(tensor_info(asp_.dst_info, dst_d_permed));

    // In this case ACL tries to treat src0 and src1 as a 1D array, but
    // fails because the strides aren't equal. TODO: remove when fixed
    // in ACL
    if (asp_.alg == alg_kind::binary_add
            && asp_.src0_info.tensor_shape() == asp_.src1_info.tensor_shape()
            && asp_.src0_info.strides_in_bytes()
                    != asp_.src1_info.strides_in_bytes()) {
        return status::unimplemented;
    }

    // This forces ACL not to parallelise with small workloads, this is
    // a temporary fix and should be removed in future versions (TODO)
    memory_desc_wrapper dst_d(dst_md());
    if (dst_d.nelems() < 40000) {
        size_t acl_y_axis_i = 1;
        CHECK(insert_singleton_dimension(asp_.src0_info, acl_y_axis_i));
        CHECK(insert_singleton_dimension(asp_.src1_info, acl_y_axis_i));
        CHECK(insert_singleton_dimension(asp_.dst_info, acl_y_axis_i));
    }

    // Call operator specific validate function to check support
    ACL_CHECK_VALID(validate(asp_));

    return status::success;
}

arm_compute::Status acl_binary_t::pd_t::validate(const acl_binary_conf_t &asp) {
    switch (asp.alg) {
        case alg_kind::binary_add:
            return arm_compute::experimental::op::CpuAdd::validate(
                    &asp.src0_info, &asp.src1_info, &asp.dst_info,
                    arm_compute::ConvertPolicy::SATURATE);
        case alg_kind::binary_sub:
            return arm_compute::experimental::op::CpuSub::validate(
                    &asp.src0_info, &asp.src1_info, &asp.dst_info,
                    arm_compute::ConvertPolicy::SATURATE);
        case alg_kind::binary_div:
            return arm_compute::experimental::op::CpuElementwiseDivision::
                    validate(&asp.src0_info, &asp.src1_info, &asp.dst_info);
        case alg_kind::binary_mul:
            return arm_compute::experimental::op::CpuMul::validate(
                    &asp.src0_info, &asp.src1_info, &asp.dst_info, 1.0f,
                    arm_compute::ConvertPolicy::SATURATE,
                    arm_compute::RoundingPolicy::TO_ZERO);
        case alg_kind::binary_min:
            return arm_compute::experimental::op::CpuElementwiseMin::validate(
                    &asp.src0_info, &asp.src1_info, &asp.dst_info);
        case alg_kind::binary_max:
            return arm_compute::experimental::op::CpuElementwiseMax::validate(
                    &asp.src0_info, &asp.src1_info, &asp.dst_info);
        default:
            return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR,
                    "unsupported alg_kind");
    }
}

status_t acl_binary_t::init(engine_t *engine) {
    auto asp = pd()->asp_;

    switch (asp.alg) {
        case alg_kind::binary_add: {
            auto add_op
                    = std::make_unique<arm_compute::experimental::op::CpuAdd>();
            add_op->configure(&asp.src0_info, &asp.src1_info, &asp.dst_info,
                    arm_compute::ConvertPolicy::SATURATE);
            binary_op_ = std::move(add_op);
            break;
        }
        case alg_kind::binary_sub: {
            auto sub_op
                    = std::make_unique<arm_compute::experimental::op::CpuSub>();
            sub_op->configure(&asp.src0_info, &asp.src1_info, &asp.dst_info,
                    arm_compute::ConvertPolicy::SATURATE);
            binary_op_ = std::move(sub_op);
            break;
        }
        case alg_kind::binary_div: {
            auto div_op = std::make_unique<
                    arm_compute::experimental::op::CpuElementwiseDivision>();
            div_op->configure(&asp.src0_info, &asp.src1_info, &asp.dst_info);
            binary_op_ = std::move(div_op);
            break;
        }
        case alg_kind::binary_mul: {
            auto mul_op
                    = std::make_unique<arm_compute::experimental::op::CpuMul>();
            mul_op->configure(&asp.src0_info, &asp.src1_info, &asp.dst_info,
                    1.0f, arm_compute::ConvertPolicy::SATURATE,
                    arm_compute::RoundingPolicy::TO_ZERO);
            binary_op_ = std::move(mul_op);
            break;
        }
        case alg_kind::binary_min: {
            auto min_op = std::make_unique<
                    arm_compute::experimental::op::CpuElementwiseMin>();
            min_op->configure(&asp.src0_info, &asp.src1_info, &asp.dst_info);
            binary_op_ = std::move(min_op);
            break;
        }
        case alg_kind::binary_max: {
            auto max_op = std::make_unique<
                    arm_compute::experimental::op::CpuElementwiseMax>();
            max_op->configure(&asp.src0_info, &asp.src1_info, &asp.dst_info);
            binary_op_ = std::move(max_op);
            break;
        }
        default: return status::runtime_error;
    }

    return status::success;
}

status_t acl_binary_t::execute_forward(const exec_ctx_t &ctx, const void *src0,
        const void *src1, void *dst) const {

    auto asp = pd()->asp_;

    arm_compute::Tensor src0_tensor;
    arm_compute::Tensor src1_tensor;
    arm_compute::Tensor dst_tensor;

    src0_tensor.allocator()->init(asp.src0_info);
    src0_tensor.allocator()->import_memory(const_cast<void *>(src0));
    src1_tensor.allocator()->init(asp.src1_info);
    src1_tensor.allocator()->import_memory(const_cast<void *>(src1));
    dst_tensor.allocator()->init(asp.dst_info);
    dst_tensor.allocator()->import_memory(dst);

    arm_compute::ITensorPack run_pack {
            {arm_compute::TensorType::ACL_SRC_0, &src0_tensor},
            {arm_compute::TensorType::ACL_SRC_1, &src1_tensor},
            {arm_compute::TensorType::ACL_DST, &dst_tensor}};

    binary_op_->run(run_pack);

    return status::success;
}

status_t acl_binary_t::execute_forward(const exec_ctx_t &ctx) const {

    auto src0 = CTX_IN_MEM(const void *, DNNL_ARG_SRC_0);
    auto src1 = CTX_IN_MEM(const void *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    return execute_forward(ctx, src0, src1, dst);
}

status_t acl_binary_t::execute(exec_ctx_t &ctx) const {
    return execute_forward(ctx);
}

const acl_binary_t::pd_t *acl_binary_t::pd() const {
    return static_cast<const pd_t *>(primitive_t::pd().get());
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
