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

#include "ocl/ref_layer_normalization.hpp"

#include "common/primitive_exec_types.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

status_t ref_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = pd()->stats_are_src() ? CTX_IN_STORAGE(DNNL_ARG_MEAN)
                                       : CTX_OUT_STORAGE(DNNL_ARG_MEAN);

    auto &variance = pd()->stats_are_src() ? CTX_IN_STORAGE(DNNL_ARG_VARIANCE)
                                           : CTX_OUT_STORAGE(DNNL_ARG_VARIANCE);

    auto &scaleshift = CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &jln = pd()->jln_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, dst);
    arg_list.set(4, scaleshift);
    arg_list.set(5, jln.eps);

    auto nd_range_kernel = jln.dispatch.nd_range();
    status_t status
            = compute_stream->parallel_for(nd_range_kernel, kernel_, arg_list);

    return status;
}

status_t ref_layer_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &scaleshift = CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT);

    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &diff_scaleshift = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SCALE_SHIFT);

    const auto &jln = pd()->jln_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, diff_dst);
    arg_list.set(4, scaleshift);
    arg_list.set(5, diff_src);
    arg_list.set(6, diff_scaleshift);
    arg_list.set(7, jln.eps);

    auto nd_range_kernel = jln.dispatch.nd_range();
    status_t status
            = compute_stream->parallel_for(nd_range_kernel, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl
