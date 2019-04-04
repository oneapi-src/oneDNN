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

#include <assert.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "common/mkldnn_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "ocl/cl_stream.hpp"

#include "ocl/ref_pooling.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <data_type_t data_type, data_type_t acc_type>
status_t ref_pooling_fwd_t<data_type, acc_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(MKLDNN_ARG_DST);
    auto &ws = CTX_OUT_STORAGE(MKLDNN_ARG_WORKSPACE);

    const auto &jpp = ker_->jpp;

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, ws);
    kernel_.set_arg(2, dst);

    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

    auto nd_range = cl_nd_range_t(jpp.gws_d, jpp.lws_d);
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

template <data_type_t data_type, data_type_t acc_type>
status_t ref_pooling_bwd_t<data_type, acc_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    auto &diff_src = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SRC);
    auto &diff_dst = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST);
    auto &ws = CTX_IN_STORAGE(MKLDNN_ARG_WORKSPACE);

    const auto &jpp = ker_->jpp;

    kernel_.set_arg(0, diff_src);
    kernel_.set_arg(1, ws);
    kernel_.set_arg(2, diff_dst);

    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

    auto nd_range = cl_nd_range_t(jpp.gws_d, jpp.lws_d);
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

template struct ref_pooling_fwd_t<data_type::f32>;
template struct ref_pooling_bwd_t<data_type::f32>;
template struct ref_pooling_fwd_t<data_type::f16>;
template struct ref_pooling_fwd_t<data_type::s8, data_type::s32>;
template struct ref_pooling_fwd_t<data_type::u8, data_type::s32>;
} // namespace ocl
} // namespace impl
} // namespace mkldnn
