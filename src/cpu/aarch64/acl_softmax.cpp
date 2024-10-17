/*******************************************************************************
* Copyright 2021-2024 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_softmax.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

const acl_softmax_fwd_t::pd_t *acl_softmax_fwd_t::pd() const {
    return static_cast<const pd_t *>(primitive_t::pd().get());
}

status_t acl_softmax_fwd_t::pd_t::init(engine_t *engine) {

    bool ok = is_fwd()
            && set_default_formats() == status::success
            // ACL only supports matching src/dst (this must come after
            // set_default_formats() to handle format_kind::any)
            && *src_md() == *dst_md()
            && utils::one_of(
                    src_md()->data_type, data_type::f32, data_type::f16)
            && attr()->has_default_values();
    if (!ok) return status::unimplemented;

    // Get memory desc to find sizes and dims
    const memory_desc_wrapper src_d(src_md());
    const data_type_t data_type = src_d.data_type();

    // ACL only supports plain tensors, can be permuted but not blocked
    if (!src_d.is_plain()) return status::unimplemented;

    // Guards against a 0-sized dimension
    if (src_d.has_zero_dim()) return status::unimplemented;

    // No scaling
    asp_.beta = 1;

    asp_.is_logsoftmax = is_logsoftmax();

    // The strides give us the in memory inner size
    dim_t inner_size_ = src_d.blocking_desc().strides[axis()];

    dim_t axis_size_ = axis_size();

    // The outer size is any left-over dimensions not inner or on the axis
    dim_t outer_size_ = src_d.nelems() / (inner_size_ * axis_size_);

    // In this context, NHWC tells ACL that the logical and physical
    // dimensions are the same
    arm_compute::DataLayout acl_layout = arm_compute::DataLayout::NHWC;

    const arm_compute::DataType acl_data_t
            = acl_utils::get_acl_data_t(data_type);

    const int threads = dnnl_get_max_threads();

    // A rough empirical heuristic created by fitting a polynomial
    // of the tensor sizes and thread count to the run time of the
    // ref and ACL softmax. This variable is greater than zero when
    // ref is faster, and less than zero when ACL is faster. We can
    // interpret the constant term as the constant overhead
    // associated with calling the external library and the negative
    // coefficient on total_size as ACL being faster at processing
    // each element
    auto calculate_performance_diff = [](dnnl::impl::dim_t outer_size,
                                              dnnl::impl::dim_t axis_size,
                                              const int threads,
                                              double sec_coff) {
        double acl_ref_performance_diff = 1 + 0.005 * outer_size
                + sec_coff * axis_size
                        * std::ceil(double(outer_size) / threads);

        if (threads > 1 || outer_size > 1) {
            acl_ref_performance_diff
                    += 17; // Adds constant overhead for using threads within ACL
        }
        return acl_ref_performance_diff;
    };

    if (inner_size_ == 1) {
        double acl_ref_performance_diff = calculate_performance_diff(
                outer_size_, axis_size_, threads, -0.0027);
        if (acl_ref_performance_diff > 0) return status::unimplemented;

        // If the inner size is 1, we can get rid of the dimension.
        // This stops ACL doing a unnecessary permute
        arm_compute::TensorShape acl_tensor_shape
                = arm_compute::TensorShape(axis_size_, outer_size_);
        asp_.axis = 0;

        asp_.src_info = arm_compute::TensorInfo(
                acl_tensor_shape, 1, acl_data_t, acl_layout);
        asp_.dst_info = arm_compute::TensorInfo(
                acl_tensor_shape, 1, acl_data_t, acl_layout);
    } else {
        // A rough empirical heuristic, see comment above
        // The only difference here is that ACL does a reorder, and so
        // is considerably better
        double acl_ref_performance_diff = calculate_performance_diff(
                outer_size_, axis_size_, threads, -0.01);
        if (acl_ref_performance_diff > 0) return status::unimplemented;

        // Irrespective of the input dimensions, we construct a tensor
        // with dimensions such that softmax can be applied over the
        // middle axis (1), with the correct stride and vector length.
        arm_compute::TensorShape acl_tensor_shape = arm_compute::TensorShape(
                inner_size_, axis_size_, outer_size_);
        asp_.axis = 1;

        asp_.src_info = arm_compute::TensorInfo(
                acl_tensor_shape, 1, acl_data_t, acl_layout);
        asp_.dst_info = arm_compute::TensorInfo(
                acl_tensor_shape, 1, acl_data_t, acl_layout);
    }

    // Validate manually to check for return status
    ACL_CHECK_VALID(arm_compute::experimental::op::CpuSoftmax::validate(
            &asp_.src_info, &asp_.dst_info, asp_.beta, asp_.axis));

    return status::success;
}

status_t acl_softmax_fwd_t::init(engine_t *engine) {
    auto asp = pd()->asp_;

    auto op = std::make_unique<arm_compute::experimental::op::CpuSoftmax>();

    softmax_op_ = std::move(op);
    // Configure softmax operation, mem allocation happens.
    softmax_op_->configure(&asp.src_info, &asp.dst_info, asp.beta, asp.axis,
            asp.is_logsoftmax);

    return status::success;
}

status_t acl_softmax_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    auto asp = pd()->asp_;

    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;

    src_tensor.allocator()->init(asp.src_info);
    src_tensor.allocator()->import_memory(const_cast<void *>(src));
    dst_tensor.allocator()->init(asp.dst_info);
    dst_tensor.allocator()->import_memory(dst);

    arm_compute::ITensorPack run_pack {
            {arm_compute::TensorType::ACL_SRC_0, &src_tensor},
            {arm_compute::TensorType::ACL_DST, &dst_tensor}};

    softmax_op_->run(run_pack);

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
