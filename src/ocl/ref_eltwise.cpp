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

#include "ocl/ref_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

/* execution:
 * 1. Get memory
 * 2. Set arguments to the kernel
 * 3. Run kernel
 * */

status_t ref_eltwise_fwd_t::execute_forward_dense(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(MKLDNN_ARG_DST);

    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    const auto &jel = pd()->jel_;

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, dst);
    kernel_.set_arg(2, alpha);
    kernel_.set_arg(3, beta);

    auto nd_range = cl_nd_range_t(jel.gws_d);
    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

status_t ref_eltwise_bwd_t::execute_backward_dense(
        const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST);
    auto &diff_src = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SRC);

    const float alpha = pd()->desc()->alpha;

    const auto &jel = pd()->jel_;

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, diff_src);
    kernel_.set_arg(2, diff_dst);
    kernel_.set_arg(3, alpha);

    auto nd_range = cl_nd_range_t(jel.gws_d);
    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

} // namespace ocl
} // namespace impl
} // namespace mkldnn
