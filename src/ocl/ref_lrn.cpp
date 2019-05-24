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

#include "ocl/ref_lrn.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <impl::data_type_t data_type>
status_t ref_lrn_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(MKLDNN_ARG_DST);
    auto &ws = CTX_OUT_STORAGE(MKLDNN_ARG_WORKSPACE);

    kernel_.set_arg(0, src);
    if (pd()->desc()->prop_kind == prop_kind::forward_training) {
        kernel_.set_arg(1, ws);
        kernel_.set_arg(2, dst);
    } else {
        kernel_.set_arg(1, dst);
    }

    auto nd_range = cl_nd_range_t(3, pd()->gws, pd()->lws);
    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

template <impl::data_type_t data_type>
status_t ref_lrn_bwd_t<data_type>::execute_backward(
    const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST);
    auto &ws = CTX_IN_STORAGE(MKLDNN_ARG_WORKSPACE);
    auto &diff_src = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SRC);

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, diff_dst);
    kernel_.set_arg(2, ws);
    kernel_.set_arg(3, diff_src);

    auto nd_range = cl_nd_range_t(3, pd()->gws, pd()->lws);
    auto &executor
        = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

template struct ref_lrn_fwd_t<data_type::f16>;
template struct ref_lrn_fwd_t<data_type::f32>;
template struct ref_lrn_bwd_t<data_type::f32>;
} // namespace ocl
} // namespace impl
} // namespace mkldnn
