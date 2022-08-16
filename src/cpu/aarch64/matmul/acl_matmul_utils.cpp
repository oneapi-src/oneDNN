/*******************************************************************************
* Copyright 2021-2022 Arm Ltd. and affiliates
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

#include "cpu/matmul/matmul_utils.hpp"

#include "cpu/aarch64/matmul/acl_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace acl_matmul_utils {

status_t init_conf_matmul(acl_matmul_conf_t &amp, memory_desc_t &src_md,
        memory_desc_t &wei_md, memory_desc_t &dst_md, const matmul_desc_t &md,
        const primitive_attr_t &attr) {

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper wei_d(&wei_md);
    const memory_desc_wrapper dst_d(&dst_md);

    cpu::matmul::matmul_helper_t helper(src_d, wei_d, dst_d);
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t dst_batch = helper.batch();
    const dim_t src_batch = helper.src_batch();
    const dim_t wei_batch = helper.wei_batch();

    // ACL supports broadcast for 3D shapes, and 4D shapes
    // for e.g when ab in abcd is 1x1
    bool batch_ok = IMPLICATION(src_batch > 1, wei_batch == 1)
            && IMPLICATION(wei_batch > 1, src_batch == 1);
    ACL_CHECK_SUPPORT(src_d.ndims() == 4 && src_batch != wei_batch && !batch_ok,
            "matmul broadcast supported only for 3D shapes and 4D shapes when "
            "ab is 1x1");

    // ACL does not support bias
    bool with_bias = md.bias_desc.format_kind != format_kind::undef;
    ACL_CHECK_SUPPORT(with_bias, "ACL does not support bias for matmul");

    using namespace format_tag;
    auto src_tag = memory_desc_matches_one_of_tag(
            src_md, abcd, abdc, abc, acb, ab, ba);
    auto wei_tag = memory_desc_matches_one_of_tag(
            wei_md, abcd, abdc, abc, acb, ab, ba);
    auto dst_tag
            = memory_desc_matches_one_of_tag(dst_md, abcd, abc, acb, ab, ba);
    ACL_CHECK_SUPPORT(
            utils::one_of(format_tag::undef, src_tag, wei_tag, dst_tag),
            "Format tag is undefined");

    // Transpose A (src) or B (wei)
    amp.is_transA = helper.transA() == 'T';
    amp.is_transB = helper.transB() == 'T';
    if (amp.is_transA)
        amp.src_acc_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(M, K, 1, src_batch), 1,
                arm_compute::DataType::F32);
    if (amp.is_transB)
        amp.wei_acc_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(K, N, wei_batch), 1,
                arm_compute::DataType::F32);

    amp.src_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(K, M, 1, src_batch), 1,
            arm_compute::DataType::F32);
    amp.wei_info
            = arm_compute::TensorInfo(arm_compute::TensorShape(N, K, wei_batch),
                    1, arm_compute::DataType::F32);
    amp.dst_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(N, M, 1, dst_batch), 1,
            arm_compute::DataType::F32);

    // Fast-math mode
    auto math_mode = get_fpmath_mode();
    bool is_fastmath_enabled
            = utils::one_of(math_mode, fpmath_mode::bf16, fpmath_mode::any);
    amp.gemm_info.set_fast_math(is_fastmath_enabled);

    // Fused ReLU activation
    amp.gemm_info.set_activation_info(get_acl_act(attr));

    // Set alpha (output scaling)
    amp.alpha = attr.output_scales_.scales_[0];

    // Validate ACL transpose
    if (amp.is_transA)
        ACL_CHECK_VALID(arm_compute::NETranspose::validate(
                &amp.src_acc_info, &amp.src_info));
    if (amp.is_transB)
        ACL_CHECK_VALID(arm_compute::NETranspose::validate(
                &amp.wei_acc_info, &amp.wei_info));
    // Validate ACL GEMM
    ACL_CHECK_VALID(arm_compute::NEGEMM::validate(&amp.src_info, &amp.wei_info,
            nullptr, &amp.dst_info, amp.alpha, 0.0f, amp.gemm_info));

    return status::success;
}

arm_compute::ActivationLayerInfo get_acl_act(const primitive_attr_t &attr) {
    const auto &post_ops = attr.post_ops_;
    const int entry_idx = post_ops.find(primitive_kind::eltwise);
    if (entry_idx == -1) { return arm_compute::ActivationLayerInfo(); }

    const auto eltwise_alg = post_ops.entry_[entry_idx].eltwise.alg;
    float alpha = post_ops.entry_[entry_idx].eltwise.alpha;
    float beta = post_ops.entry_[entry_idx].eltwise.beta;

    using acl_act_t = arm_compute::ActivationLayerInfo::ActivationFunction;
    acl_act_t acl_act_alg;
    switch (eltwise_alg) {
        case eltwise_relu:
            // oneDNN defines RELU: f(x) = (x > 0) ? x : a*x
            // Compute Library defines LEAKY_RELU: f(x) = (x > 0) ? x : a*x
            // whilst Compute Library RELU is defined as: f(x) = max(0,x)
            if (alpha == 0) {
                acl_act_alg = acl_act_t::RELU;
            } else {
                acl_act_alg = acl_act_t::LEAKY_RELU;
            }
            break;
        case eltwise_tanh:
            // oneDNN defines TANH activation as:          f(x) = tanh(x)
            // Compute Library defines TANH activation as: f(x) = a*tanh(b*x)
            // Setting a=b=1 makes the two equivalent
            alpha = 1.f;
            beta = 1.f;
            acl_act_alg = acl_act_t::TANH;
            break;
        case eltwise_elu: acl_act_alg = acl_act_t::ELU; break;
        case eltwise_square: acl_act_alg = acl_act_t::SQUARE; break;
        case eltwise_abs: acl_act_alg = acl_act_t::ABS; break;
        case eltwise_sqrt: acl_act_alg = acl_act_t::SQRT; break;
        case eltwise_linear: acl_act_alg = acl_act_t::LINEAR; break;
        case eltwise_bounded_relu: acl_act_alg = acl_act_t::BOUNDED_RELU; break;
        case eltwise_soft_relu: acl_act_alg = acl_act_t::SOFT_RELU; break;
        case eltwise_logistic: acl_act_alg = acl_act_t::LOGISTIC; break;
        default: return arm_compute::ActivationLayerInfo();
    }

    return arm_compute::ActivationLayerInfo(acl_act_alg, alpha, beta);
}

bool acl_act_ok(alg_kind_t eltwise_activation) {
    return utils::one_of(eltwise_activation, eltwise_relu, eltwise_tanh,
            eltwise_elu, eltwise_square, eltwise_abs, eltwise_sqrt,
            eltwise_linear, eltwise_bounded_relu, eltwise_soft_relu,
            eltwise_logistic);
}

} // namespace acl_matmul_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
