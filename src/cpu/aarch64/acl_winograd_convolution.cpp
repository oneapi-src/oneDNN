/*******************************************************************************
* Copyright 2020-2024 Arm Ltd. and affiliates
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

#include "acl_winograd_convolution.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {
using data_t = prec_traits<data_type::f32>::type;

// Keys are anonymous. So deduce the type automagically.
using conv_key_t = decltype(memory_tracking::names::key_gemm_tmp_buffer);

// Map: [slot , key]
const std::map<int, conv_key_t> wino_conv_keys
        = {{0, conv_key_t::key_gemm_asm_tmp_buffer},
                {1, conv_key_t::key_gemm_pretranspose_b},
                {2, conv_key_t::key_gemm_pretranspose},
                {3, conv_key_t::key_gemm_interleaved_lhs},
                {4, conv_key_t::key_gemm_pretransposed_rhs},
                {5, conv_key_t::key_gemm_transposed_1xwrhs},
                {6, conv_key_t::key_gemm_tmp_buffer},
                {7, conv_key_t::key_conv_permuted_outputs},
                {8, conv_key_t::key_conv_permuted_inputs},
                {9, conv_key_t::key_wino_workspace},
                {10, conv_key_t::key_wino_transformed_weights},
                {11, conv_key_t::key_conv_permuted_weights}};
} // namespace

status_t acl_wino_convolution_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    const bool is_fp16_ok = expect_data_types(f16, f16, f16, f16, undef)
            && attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::post_ops, f16);
    const bool is_fp32_ok = expect_data_types(f32, f32, f32, f32, undef)
            && attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::post_ops, f32);
    bool ok = is_fwd()
            && utils::one_of(desc()->alg_kind, alg_kind::convolution_auto,
                    alg_kind::convolution_winograd)
            && utils::one_of(true, is_fp16_ok, is_fp32_ok)
            && !has_zero_dim_memory();

    ok = ok && DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL;
    if (!ok) return status::unimplemented;

    CHECK(init_conf());

    set_default_alg_kind(alg_kind::convolution_winograd);

    Op conv;
    conv.configure(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info, acp_.padstride_info, acp_.act_info,
            true); // to support 5x5, 7x7 filter shapes in addition to 3x3

    auto scratchpad = scratchpad_registry().registrar();
    const auto aux_mem = conv.workspace();
    return init_scratchpad(conv, scratchpad, wino_conv_keys, engine, post_ops,
            attr_.post_ops_, acp_.act_info, acp_.use_dst_acc_for_sum, dst_md_,
            bias_md_, acp_.is_quantized);
}

status_t acl_wino_convolution_fwd_t::init(engine_t *engine) {
    // commented due to hot fix solution for stateless API which should be replaced soon.
    //     auto acp = pd()->acp_;
    //     acl_obj_->conv.configure(&acp.src_tensor_info, &acp.wei_tensor_info,
    //             acp.with_bias ? &acp.bia_tensor_info : nullptr,
    //             &acp.dst_tensor_info, acp.padstride_info, acp.act_info,
    //             true); // to support 5x5, 7x7 filter shapes in addition to 3x3

    //     acl_obj_->aux_mem_req = acl_obj_->conv.workspace();
    return status::success;
}

status_t acl_wino_convolution_fwd_t::pd_t::init_conf() {

    // Under these conditions, fallback to faster GEMM-based convolution
    // unless the user explicitly specifies Winograd algorithm
    if (utils::one_of(true, src_md_.dims[2] > 112, // ih
                src_md_.dims[3] > 112, // iw
                src_md_.dims[1] < 64, // ic
                dst_md_.dims[1]<64, // oc
                        dnnl_get_max_threads()> 28)
            && desc()->alg_kind == alg_kind::convolution_auto) {
        return status::unimplemented;
    }

    // General Compute Library checks, memory tags are also set there
    acp_.alg_winograd = true;
    CHECK(acl_convolution_utils::acl_init_conf(
            acp_, src_md_, weights_md_, dst_md_, bias_md_, *desc(), *attr()));

    const bool shape_ok
            // only unit strides allowed
            = (acp_.padstride_info.stride()
                      == std::pair<unsigned int, unsigned int> {1, 1})
            // Note: Compute Library supports arbitrary padding for wino kernels
            // but we only allow small padding to be consistent with oneDNN
            && (acp_.padstride_info.pad().first <= 1) // padding left/right
            && (acp_.padstride_info.pad().second <= 1) // padding top/bottom
            // only non-dilated convolutions allowed
            && (acp_.dilation_info == arm_compute::Size2D(1, 1));

    ACL_CHECK_SUPPORT(!shape_ok, "shape not supported by winograd kernels");

    // Validate convolution manually to check for return status
    ACL_CHECK_VALID(Op::validate(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info, acp_.padstride_info, acp_.act_info,
            true)); // enable_fast_math flag in ACL Winograd

    return status::success;
}

std::unique_ptr<acl_obj_t<acl_wino_convolution_fwd_t::Op>>
acl_wino_convolution_fwd_t::reinitialize_acl_obj() const {
    auto acp = pd()->acp_;
    std::unique_ptr<acl_obj_t<Op>> acl_obj = std::make_unique<acl_obj_t<Op>>();
    acl_obj->conv.configure(&acp.src_tensor_info, &acp.wei_tensor_info,
            acp.with_bias ? &acp.bia_tensor_info : nullptr,
            &acp.dst_tensor_info, acp.padstride_info, acp.act_info,
            true); // to support 5x5, 7x7 filter shapes in addition to 3x3

    acl_obj->aux_mem_req = acl_obj->conv.workspace();
    return acl_obj;
}

status_t acl_wino_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    // Temporary hotfix: We're using a local acl_obj instance in this method
    // instead of the class member acl_obj_. This hotfix is to bypass persistent aux mem requirements but is not the ideal solution.
    // It should be refactored or removed in the future when a more permanent fix is implemented.
    const auto acl_obj = reinitialize_acl_obj();
    return execute_forward_conv_acl<acl_obj_t<Op>, pd_t, data_t>(
            ctx, acl_obj.get(), pd(), wino_conv_keys);
}
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
