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

#include "ocl/ref_softmax.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <impl::data_type_t data_type>
status_t ref_softmax_fwd_t<data_type>::execute_generic(
        const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->desc()->data_desc).has_zero_dim())
        return status::success;

    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(MKLDNN_ARG_DST);

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, dst);

    auto nd_range = cl_nd_range_t(pd()->gws.size(), pd()->gws.data());
    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

template <impl::data_type_t data_type>
status_t ref_softmax_bwd_t<data_type>::execute_generic(
        const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->desc()->diff_desc).has_zero_dim())
        return status::success;

    auto &dst = CTX_IN_STORAGE(MKLDNN_ARG_DST);
    auto &diff_dst = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST);
    auto &diff_src = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SRC);

    kernel_.set_arg(0, dst);
    kernel_.set_arg(1, diff_src);
    kernel_.set_arg(2, diff_dst);

    auto nd_range = cl_nd_range_t(pd()->gws.size(), pd()->gws.data());
    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

template struct ref_softmax_fwd_t<data_type::f16>;
template struct ref_softmax_fwd_t<data_type::f32>;
template struct ref_softmax_bwd_t<data_type::f32>;

} // namespace ocl
} // namespace impl
} // namespace mkldnn

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
