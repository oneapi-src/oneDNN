/*******************************************************************************
* Copyright 2021-2022, 2024 Arm Ltd. and affiliates
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

#include "acl_indirect_gemm_convolution.hpp"
#include "acl_convolution_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {
using data_t = typename prec_traits_t<data_type::f32>::type;

// Keys are anonymous. So deduce the type automagically.
using conv_key_t = decltype(memory_tracking::names::key_gemm_tmp_buffer);

// Map: [slot , key]
const std::map<int, conv_key_t> indirect_conv_keys
        = {{0, conv_key_t::key_gemm_tmp_buffer},
                {2, conv_key_t::key_gemm_pretranspose},
                {3, conv_key_t::key_conv_permuted_weights}};
} // namespace

status_t acl_indirect_gemm_convolution_fwd_t::init(engine_t *engine) {
    auto acp_ = pd()->acp_;
    acl_obj_->conv.configure(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info,
            arm_compute::Conv2dInfo(acp_.padstride_info, acp_.dilation_info,
                    acp_.act_info, acp_.fast_math, 1, acp_.weights_info));
    acl_obj_->aux_mem_req = acl_obj_->conv.workspace();
    return status::success;
}

status_t acl_indirect_gemm_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    return execute_forward_conv_acl<acl_obj_t<Op>, pd_t, data_t>(
            ctx, acl_obj_.get(), pd(), indirect_conv_keys);
}

status_t acl_indirect_gemm_convolution_fwd_t::pd_t::init_conf() {
    if (weights_md_.ndims != 4) return status::unimplemented;

    // Indirect is slower for small convolution kernels, except when src, weight and dst are BF16
    if (weights_md_.dims[2] == 1 && weights_md_.dims[3] == 1
            && !dnnl::impl::utils::everyone_is(data_type::bf16,
                    src_md_.data_type, weights_md_.data_type,
                    dst_md_.data_type))
        return status::unimplemented;

    CHECK(acl_convolution_utils::acl_init_conf(
            acp_, src_md_, weights_md_, dst_md_, bias_md_, *desc(), *attr()));

    // If we do not need to pad input channels for fast math mode then it would
    // be faster to run convolution with im2row instead of using indirect kernel
    int block_by = arm_compute::block_by(acp_.weights_info.weight_format());
    int ic = src_md_.dims[1];
    if (acp_.fast_math && ic % block_by == 0) return status::unimplemented;

    // clang-format off
    // NOTE: indirect convolution method supports only nhwc layout.
    ACL_CHECK_VALID(Op::validate(
        &acp_.src_tensor_info,
        &acp_.wei_tensor_info,
        acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
        &acp_.dst_tensor_info,
        arm_compute::Conv2dInfo(acp_.padstride_info,
                                acp_.dilation_info,
                                acp_.act_info,
                                acp_.fast_math,
                                1, acp_.weights_info)));
    // clang-format on

    return status::success;
}

status_t acl_indirect_gemm_convolution_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using smask_t = primitive_attr_t::skip_mask_t;

    const bool is_fp16_ok = expect_data_types(f16, f16, f16, f16, undef)
            && attr()->has_default_values(smask_t::post_ops, f16);
    const bool is_bf16_ok = expect_data_types(bf16, bf16, bf16, bf16, undef)
            && attr_.post_ops_.len() == 0;
    const bool is_fp32_ok = expect_data_types(f32, f32, f32, f32, undef)
            && attr()->has_default_values(
                    smask_t::post_ops | smask_t::fpmath_mode, f32);
    bool ok = is_fwd() && set_default_alg_kind(alg_kind::convolution_direct)
            && utils::one_of(true, is_fp16_ok, is_bf16_ok, is_fp32_ok)
            && !has_zero_dim_memory();
    if (!ok) return status::unimplemented;

    CHECK(init_conf());

    // Book memory.
    Op conv;
    conv.configure(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info,
            arm_compute::Conv2dInfo(acp_.padstride_info, acp_.dilation_info,
                    acp_.act_info, acp_.fast_math, 1, acp_.weights_info));

    auto scratchpad = scratchpad_registry().registrar();
    return init_scratchpad(conv, scratchpad, indirect_conv_keys, engine,
            post_ops, attr_.post_ops_, acp_.act_info, acp_.use_dst_acc_for_sum,
            dst_md_);
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
