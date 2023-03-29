/*******************************************************************************
* Copyright 2021-2023 Arm Ltd. and affiliates
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

    // We can only broadcast on one of src or wei at once
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

    // The two innermost dimensions can be transposed, but the batch dimensions
    // must be the outermost
    using namespace format_tag;
    auto src_tag = memory_desc_matches_one_of_tag(
            src_md, abcd, abdc, abc, acb, ab, ba);
    auto dst_tag = memory_desc_matches_one_of_tag(dst_md, abcd, abc, ab, ba);
    ACL_CHECK_SUPPORT(utils::one_of(format_tag::undef, src_tag, dst_tag),
            "Format tag is undefined");

    // Transpose A (src)
    amp.is_transA = helper.transA() == 'T';

    auto acl_src_data_t = acl_utils::get_acl_data_t(src_md.data_type);
    auto acl_wei_data_t = acl_utils::get_acl_data_t(wei_md.data_type);
    auto acl_dst_data_t = acl_utils::get_acl_data_t(dst_md.data_type);

    if (amp.is_transA)
        amp.src_acc_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(M, K, 1, src_batch), 1,
                acl_src_data_t);

    amp.src_tensor_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(K, M, 1, src_batch), 1, acl_src_data_t);
    amp.wei_tensor_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(N, K, wei_batch), 1, acl_wei_data_t);
    amp.dst_tensor_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(N, M, 1, dst_batch), 1, acl_dst_data_t);

    // Set alpha (output scaling)
    // TODO: Add runtime scales support. Creation time scales will be remove
    // in 3.0.
    amp.alpha = 1.0f; // default value
    if (!attr.output_scales_.has_default_values()) return status::unimplemented;

    // Validate ACL transpose
    if (amp.is_transA)
        ACL_CHECK_VALID(arm_compute::NETranspose::validate(
                &amp.src_acc_info, &amp.src_tensor_info));

    bool is_fastmath_enabled = utils::one_of(
            attr.fpmath_mode_, fpmath_mode::bf16, fpmath_mode::any);
    amp.gemm_info.set_fast_math(is_fastmath_enabled);

    amp.gemm_info.set_fixed_format(true);

    // WeightFormat::ANY tells ACL we can handle any format
    amp.gemm_info.set_weight_format(arm_compute::WeightFormat::ANY);

    // Get the format that the ACL kernel will expect the weights to be
    // in (if a kernel exists). Note that these are referred to as fixed format
    // kernels, because they require one specific weights format
    arm_compute::WeightFormat expected_weight_format;
    ACL_CHECK_VALID(arm_compute::NEGEMM::has_opt_impl(expected_weight_format,
            &amp.src_tensor_info, &amp.wei_tensor_info, nullptr,
            &amp.dst_tensor_info, amp.alpha, 0.0f, amp.gemm_info));

    // Set gemm weights info to the one returned by has_opt_impl
    amp.gemm_info.set_weight_format(expected_weight_format);

    // has_opt_impl may return a non fast math kernel, even if we requested one
    amp.gemm_info.set_fast_math(
            arm_compute::is_fixed_format_fast_math(expected_weight_format));

    // Logical dimension indices
    dim_t innermost_dim = wei_md.ndims - 1;
    dim_t N_dim = innermost_dim;
    dim_t K_dim = innermost_dim - 1;

    // The logical indices of dimensions related to the batch, ordered from
    // innermost to outermost
    std::vector<dim_t> batch_dims = {};
    for (dim_t i = K_dim - 1; i >= 0; --i)
        batch_dims.push_back(i);

    acl_utils::reorder_to_weight_format(amp.wei_tensor_info, wei_md,
            expected_weight_format, K_dim, N_dim, {}, batch_dims);

    return status::success;
}

} // namespace acl_matmul_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
