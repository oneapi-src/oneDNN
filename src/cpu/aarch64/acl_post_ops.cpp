/*******************************************************************************
* Copyright 2022-2023 Arm Ltd. and affiliates
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

#include "common/float16.hpp"
#include "cpu/aarch64/acl_gemm_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_post_ops_t::execute(const exec_ctx_t &ctx, void *src_orig) const {

    int post_op_index = 0;

    // As these are post ops, this src will also be our dst. If we have a sum
    // post op, the src/dst will start off in a temporary, then change to
    // DNNL_ARG_DST after the sum.
    void *src = src_orig;

    // Post ops must operate in place on dst, unless when we have a sum op
    if (!has_sum() && src != CTX_OUT_MEM(void *, DNNL_ARG_DST)) {
        return status::runtime_error;
    }

    for (auto &post_op : post_op_primitives) {
        if (post_op->kind() == primitive_kind::binary) {
            auto binary_post_op = dynamic_cast<acl_binary_t *>(post_op.get());
            if (binary_post_op == nullptr) return status::runtime_error;

            // Sum post op accumulates to dst and changes future dst
            if (post_op_index == sum_index) {
                // Change src to final dst, then add orig source to it
                src = CTX_OUT_MEM(void *, DNNL_ARG_DST);
                CHECK(binary_post_op->execute_forward(ctx, src_orig, src, src));
            } else {
                const void *src1 = CTX_IN_MEM(const void *,
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                                | DNNL_ARG_SRC_1));
                CHECK(binary_post_op->execute_forward(ctx, src, src1, src));
            }
        } else if (post_op->kind() == primitive_kind::eltwise) {
            // The post op at the sum index must be binary
            if (post_op_index == sum_index) return status::runtime_error;

            auto eltwise_post_op
                    = dynamic_cast<acl_eltwise_fwd_t *>(post_op.get());
            if (eltwise_post_op == nullptr) return status::runtime_error;

            if (dst_data_type == data_type::f16) {
                // in this case we want to cast the src tensor up to fp32
                arm_compute::TensorInfo src_info
                        = eltwise_post_op->pd()->aep.data_info;
                // new src tensor with fp32 datatype
                arm_compute::Tensor src_tensor;
                src_tensor.allocator()->init(src_info);
                src_tensor.allocator()->allocate();
                float *src_f32 = (float *)src_tensor.buffer();
                // total_size gives the size in bytes, we divide by 4 because the src_tensor is fp32
                size_t num_elements = src_tensor.info()->total_size() / 4;
                // cast src up to fp32 and store the result in src_f32
                cvt_float16_to_float(
                        src_f32, (dnnl::impl::float16_t *)src, num_elements);
                // perform the operation in fp32
                status_t eltwise_status = eltwise_post_op->execute_forward(
                        ctx, src_f32, src_f32);
                if (eltwise_status == status::success) {
                    // cast src_f32 down and store final result in src
                    cvt_float_to_float16((dnnl::impl::float16_t *)src, src_f32,
                            num_elements);
                }
                src_tensor.allocator()->free();
                CHECK(eltwise_status);

            } else {
                CHECK(eltwise_post_op->execute_forward(ctx, src, src));
            }

        } else {
            return status::runtime_error;
        }

        ++post_op_index;
    }

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
