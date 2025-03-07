/*******************************************************************************
* Copyright 2021-2025 Arm Ltd. and affiliates
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

#include "cpu/aarch64/matmul/acl_matmul_utils.hpp"
#include "cpu/aarch64/acl_utils.hpp"
#include "cpu/matmul/matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace acl_matmul_utils {

template <bool IsFixedFormat>
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

    if (src_d.ndims() == 4 && src_batch == wei_batch
            && src_d.dims()[0] != wei_d.dims()[0]) { // 4D broadcast occurred
        if (src_d.dims()[0] == 1 && wei_d.dims()[0] != 1) { // Broadcast src
            ACL_CHECK_SUPPORT(
                    IMPLICATION(src_d.dims()[1] != 1, wei_d.dims()[1] == 1),
                    "acl only broadcasts one of src or wei at once");
        }

        if (wei_d.dims()[0] == 1 && src_d.dims()[0] != 1) { // Broadcast wei
            ACL_CHECK_SUPPORT(
                    IMPLICATION(src_d.dims()[1] == 1, wei_d.dims()[1] != 1),
                    "acl only broadcasts one of src or wei at once");
        }
    }

    // ACL does not support bias
    bool with_bias = md.bias_desc.format_kind != format_kind::undef;
    ACL_CHECK_SUPPORT(with_bias, "ACL does not support bias for matmul");

    // The two innermost dimensions can be transposed, but the batch dimensions
    // must be the outermost
    using namespace format_tag;
    if (IsFixedFormat) {
        auto src_tag = memory_desc_matches_one_of_tag(
                src_md, abcd, abdc, abc, acb, ab, ba);
        auto dst_tag
                = memory_desc_matches_one_of_tag(dst_md, abcd, abc, ab, ba);
        ACL_CHECK_SUPPORT(utils::one_of(format_tag::undef, src_tag, dst_tag),
                "Format tag is undefined");
    } else {
        auto src_tag = memory_desc_matches_one_of_tag(
                src_md, acdb, abcd, abdc, abc, acb, ab, ba);
        auto dst_tag = memory_desc_matches_one_of_tag(dst_md, abcd, abc, ab);
        ACL_CHECK_SUPPORT(utils::one_of(format_tag::undef, src_tag, dst_tag),
                "Format tag is undefined");
    }

    // Transpose A (src) and/or B (wei). Transpose B is not needed for fixed format.
    amp.is_transA = helper.transA() == 'T';
    amp.is_transB = IsFixedFormat ? false : helper.transB() == 'T';

    // Do (BA)^T instead of (A^T)(B^T), if the cost of transposing (BA)
    // which is ~M*N, is less than the cost of tranposing A and B which
    // is ~(M*K + K*N).
    amp.do_transC = amp.is_transA && amp.is_transB && M * N <= K * (M + N);

    auto acl_src_data_t = acl_utils::get_acl_data_t(src_md.data_type);
    auto acl_wei_data_t = acl_utils::get_acl_data_t(wei_md.data_type);
    auto acl_dst_data_t = acl_utils::get_acl_data_t(dst_md.data_type);

    if (amp.is_transA && !amp.do_transC) {
        amp.src_acc_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(M, K, 1, src_batch), 1,
                acl_src_data_t);
    }
    if (amp.is_transB && !amp.do_transC) {
        amp.wei_acc_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(K, N, wei_batch), 1, acl_wei_data_t);
    }
    if (amp.do_transC) {
        amp.dst_acc_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(M, N, 1, dst_batch), 1,
                acl_dst_data_t);
        amp.src_tensor_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(M, K, src_batch), 1, acl_src_data_t);
        amp.src_tensor_info.set_are_values_constant(false);
        amp.wei_tensor_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(K, N, 1, wei_batch), 1,
                acl_wei_data_t);
    } else {
        amp.src_tensor_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(K, M, 1, src_batch), 1,
                acl_src_data_t);
        amp.wei_tensor_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(N, K, wei_batch), 1, acl_wei_data_t);
        amp.wei_tensor_info.set_are_values_constant(false);
    }

    amp.dst_tensor_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(N, M, 1, dst_batch), 1, acl_dst_data_t);

    // Validate ACL transpose
    if (amp.is_transA && !amp.do_transC)
        ACL_CHECK_VALID(arm_compute::experimental::op::CpuTranspose::validate(
                &amp.src_acc_info, &amp.src_tensor_info));
    if (amp.is_transB && !amp.do_transC)
        ACL_CHECK_VALID(arm_compute::experimental::op::CpuTranspose::validate(
                &amp.wei_acc_info, &amp.wei_tensor_info));
    if (amp.do_transC)
        ACL_CHECK_VALID(arm_compute::experimental::op::CpuTranspose::validate(
                &amp.dst_acc_info, &amp.dst_tensor_info));

    bool is_fastmath_enabled = utils::one_of(
            attr.fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any);
    amp.gemm_info.set_fast_math(is_fastmath_enabled);

    if (IsFixedFormat) {
        amp.gemm_info.set_fixed_format(true);

        // WeightFormat::ANY tells ACL we can handle any format
        amp.gemm_info.set_weight_format(arm_compute::WeightFormat::ANY);

        // Get the format that the ACL kernel will expect the weights to be
        // in (if a kernel exists). Note that these are referred to as fixed format
        // kernels, because they require one specific weights format
        arm_compute::WeightFormat expected_weight_format;
        ACL_CHECK_VALID(
                arm_compute::experimental::op::ll::CpuGemmAssemblyDispatch::
                        has_opt_impl(expected_weight_format,
                                &amp.src_tensor_info, &amp.wei_tensor_info,
                                nullptr, &amp.dst_tensor_info, amp.gemm_info));

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
    }

    return status::success;
}

status_t init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const acl_matmul_conf_t &amp, const memory_desc_t &src_md,
        const memory_desc_t &weights_md, const memory_desc_t &dst_md,
        const arm_compute::experimental::MemoryRequirements &aux_mem_req) {
    if (amp.use_dst_acc_for_sum) {
        const memory_desc_wrapper dst_d(&dst_md);
        scratchpad.book(memory_tracking::names::key_matmul_dst_in_acc_dt,
                dst_d.nelems(), dst_d.data_type_size());
    }
    if (aux_mem_req.size() != 0) {
        for (const auto &key : matmul_keys) {
            const auto id = key.first;
            if (aux_mem_req[id].size > 0) {
                scratchpad.book(key.second, aux_mem_req[id].size, 1,
                        aux_mem_req[id].alignment, aux_mem_req[id].alignment);
            }
        }
    }
    if (amp.is_transA) {
        const memory_desc_wrapper src_d(&src_md);
        scratchpad.book(memory_tracking::names::key_matmul_src_trans,
                src_d.nelems(), src_d.data_type_size());
    }
    if (amp.is_transB) {
        const memory_desc_wrapper wei_d(&weights_md);
        scratchpad.book(memory_tracking::names::key_matmul_wei_trans,
                wei_d.nelems(), wei_d.data_type_size());
    }
    if (amp.do_transC) {
        const memory_desc_wrapper dst_d(&dst_md);
        scratchpad.book(memory_tracking::names::key_matmul_dst_trans,
                dst_d.nelems(), dst_d.data_type_size());
    }
    return status::success;
}

template status_t init_conf_matmul<true>(acl_matmul_conf_t &amp,
        memory_desc_t &src_md, memory_desc_t &wei_md, memory_desc_t &dst_md,
        const matmul_desc_t &md, const primitive_attr_t &attr);
template status_t init_conf_matmul<false>(acl_matmul_conf_t &amp,
        memory_desc_t &src_md, memory_desc_t &wei_md, memory_desc_t &dst_md,
        const matmul_desc_t &md, const primitive_attr_t &attr);

} // namespace acl_matmul_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
