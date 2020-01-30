/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "ocl/ref_convolution.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

status_t ref_convolution_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    auto eltwise_alpha = pd()->eltwise_alpha();
    auto eltwise_beta = pd()->eltwise_beta();
    auto sum_scale = pd()->sum_scale();

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);
    arg_list.set(4, eltwise_alpha);
    arg_list.set(5, eltwise_beta);
    arg_list.set(6, sum_scale);
    if (utils::one_of(
                pd()->src_md()->data_type, data_type::u8, data_type::s8)) {
        if (pd()->with_common_scales()) {
            float scales = pd()->attr()->output_scales_.scales_[0];
            arg_list.set(7, scales);
        }
        if (pd()->with_per_oc_scales()) {
            arg_list.set(7, *scales_mem_->memory_storage());
        }
    }

    const auto *jit_kernel = this->pd()->kernel();
    auto nd_range = jit_kernel->dispatch().nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);
    return status;
}

status_t ref_convolution_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {

    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, weights);
    arg_list.set(2, diff_dst);
    arg_list.set(3, bias);

    const auto *jit_kernel = this->pd()->kernel();
    auto nd_range = jit_kernel->dispatch().nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

status_t ref_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {

    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, diff_weights);
    arg_list.set(2, diff_bias);
    arg_list.set(3, diff_dst);

    const auto *jit_kernel = this->pd()->kernel();
    auto nd_range = jit_kernel->dispatch().nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl
