/*******************************************************************************
* Copyright 2020-2025 Arm Ltd. and affiliates
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

#include "acl_gemm_convolution.hpp"
#include "acl_convolution_utils.hpp"
#include "common/memory_tracking.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {
// Keys are anonymous. So deduce the type automagically.
using conv_key_t = decltype(memory_tracking::names::key_gemm_tmp_buffer);

// Map: [slot , key]
// These correspond to the information provided by Op::workspace, which
// specifies a unique numbered slot (not necessarily in continous ascending
// order) for each key.
const std::map<int, conv_key_t> gemm_conv_keys
        = {{0, conv_key_t::key_gemm_asm_tmp_buffer},
                {1, conv_key_t::key_gemm_pretranspose_b},
                {2, conv_key_t::key_gemm_pretranspose},
                {3, conv_key_t::key_gemm_interleaved_lhs},
                {4, conv_key_t::key_gemm_pretransposed_rhs},
                {5, conv_key_t::key_gemm_transposed_1xwrhs},
                {6, conv_key_t::key_gemm_tmp_buffer},
                {7, conv_key_t::key_gemm_mm_result_s32},
                {8, conv_key_t::key_gemm_mm_signed_a},
                {9, conv_key_t::key_gemm_mm_signed_output},
                {10, conv_key_t::key_conv_gemm_col},
                {11, conv_key_t::key_conv_permuted_weights},
                {12, conv_key_t::key_gemm_output}};
} // namespace

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
status_t acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t, bia_t>::pd_t::init(
        engine_t *engine) {
    using namespace data_type;

    bool ok = is_fwd() && set_default_alg_kind(alg_kind::convolution_direct)
            && expect_data_types(src_t, wei_t, bia_t, dst_t, undef)
            && !has_zero_dim_memory() && output_scales_mask_ok()
            && zero_points_ok();

    if (!ok) return status::unimplemented;

    if (weights_md_.ndims != 4) return status::unimplemented;

    // currently, only CpuGemmConv2d has the static quantization update interface.
    acp_.is_quantized
            = utils::one_of(dst_md_.data_type, data_type::s8, data_type::u8);

    // General Compute Library checks, memory tags are also set there
    CHECK(acl_convolution_utils::acl_init_conf(
            acp_, src_md_, weights_md_, dst_md_, bias_md_, *desc(), *attr()));

    // Validate convolution manually to check for return status
    ACL_CHECK_VALID(Op::validate(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info, acp_.padstride_info, acp_.weights_info,
            acp_.dilation_info, acp_.act_info, acp_.fast_math));

    Op conv;
    conv.configure(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info, acp_.padstride_info, acp_.weights_info,
            acp_.dilation_info, acp_.act_info, acp_.fast_math);

    auto scratchpad = scratchpad_registry().registrar();
    const auto mem_req = conv.workspace();
    return init_scratchpad(conv, scratchpad, gemm_conv_keys, engine, post_ops,
            attr_.post_ops_, acp_.act_info, acp_.use_dst_acc_for_sum, dst_md_,
            bias_md_, acp_.is_quantized);
}

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
bool acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t,
        bia_t>::pd_t::output_scales_mask_ok() const {
    int mask_src = attr()->scales_.get(DNNL_ARG_SRC).mask_;
    int mask_wei = attr()->scales_.get(DNNL_ARG_WEIGHTS).mask_;
    int mask_dst = attr()->scales_.get(DNNL_ARG_DST).mask_;
    return mask_src == 0 && mask_wei == 0 && mask_dst == 0;
}

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
bool acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t,
        bia_t>::pd_t::zero_points_ok() const {
    return attr()->zero_points_.common();
}

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
status_t acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t, bia_t>::init(
        engine_t *engine) {
    auto acp_ = pd()->acp_;
    acl_obj_->conv.configure(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info, acp_.padstride_info, acp_.weights_info,
            acp_.dilation_info, acp_.act_info, acp_.fast_math);
    acl_obj_->aux_mem_req = acl_obj_->conv.workspace();
    return status::success;
}

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
status_t
acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t, bia_t>::execute_forward(
        const exec_ctx_t &ctx) const {
    return execute_forward_conv_acl<acl_obj_t<Op>, pd_t, src_data_t, wei_data_t,
            dst_data_t, bia_data_t>(ctx, acl_obj_.get(), pd(), gemm_conv_keys);
}

using namespace data_type;
template struct acl_gemm_convolution_fwd_t<f32>;
template struct acl_gemm_convolution_fwd_t<f16>;
template struct acl_gemm_convolution_fwd_t<s8, s8, s8, s32>;
template struct acl_gemm_convolution_fwd_t<u8, s8, u8, s32>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
