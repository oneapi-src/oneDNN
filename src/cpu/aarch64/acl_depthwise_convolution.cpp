/*******************************************************************************
* Copyright 2023-2024 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_depthwise_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {
using data_t = prec_traits_t<data_type::f32>::type;

// Keys are anonymous. So deduce the type automagically.
using conv_key_t = decltype(memory_tracking::names::key_gemm_tmp_buffer);

// Map: [slot , key]
const std::map<int, conv_key_t> depthwise_conv_keys
        = {{0, conv_key_t::key_gemm_tmp_buffer},
                {1, conv_key_t::key_conv_permuted_weights}};
} // namespace

status_t acl_depthwise_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    return execute_forward_conv_acl<acl_obj_t<Op>, pd_t, data_t>(
            ctx, acl_obj_.get(), pd(), depthwise_conv_keys);
}

status_t acl_depthwise_convolution_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;

    const bool is_fp16_ok = expect_data_types(f16, f16, f16, f16, undef)
            && attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::post_ops, f16);
    const bool is_fp32_ok = expect_data_types(f32, f32, f32, f32, undef)
            && attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::post_ops, f32);
    bool ok = is_fwd() && set_default_alg_kind(alg_kind::convolution_direct)
            && utils::one_of(true, is_fp16_ok, is_fp32_ok)
            && !has_zero_dim_memory();
    if (!ok) return status::unimplemented;

    if (weights_md_.ndims != 5) return status::unimplemented;

    CHECK(acl_convolution_utils::acl_init_conf(
            acp_, src_md_, weights_md_, dst_md_, bias_md_, *desc(), *attr()));

    ACL_CHECK_VALID(Op::validate(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info, acp_.padstride_info,
            1, // depth multiplier default value
            acp_.act_info, acp_.dilation_info));

    Op conv;
    conv.configure(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info, acp_.padstride_info,
            1, // depth multiplier default value
            acp_.act_info, acp_.dilation_info);

    auto scratchpad = scratchpad_registry().registrar();
    return init_scratchpad(conv, scratchpad, depthwise_conv_keys, engine,
            post_ops, attr_.post_ops_, acp_.act_info, acp_.use_dst_acc_for_sum,
            dst_md_);
}

status_t acl_depthwise_convolution_fwd_t::init(engine_t *engine) {
    auto acp_ = pd()->acp_;
    acl_obj_->conv.configure(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info, acp_.padstride_info,
            1, // depth multiplier default value
            acp_.act_info, acp_.dilation_info);
    acl_obj_->aux_mem_req = acl_obj_->conv.workspace();
    return status::success;
}
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
