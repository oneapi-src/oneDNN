/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "ocl/jit_gen9_common_convolution.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace ocl {

using math::saturate;

status_t jit_gen9_common_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &jcp = ker_->jcp;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);
    arg_list.set(4, jcp.eltwise.alpha);
    arg_list.set(5, jcp.eltwise.beta);
    arg_list.set(6, jcp.sum_scale);

    auto nd_range = compute::nd_range_t(jcp.gws_d, jcp.lws_d);
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

status_t jit_gen9_common_convolution_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    const auto &jcp = ker_->jcp;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, weights);
    arg_list.set(2, diff_dst);
    arg_list.set(3, bias);

    auto nd_range = compute::nd_range_t(jcp.gws_d, jcp.lws_d);
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

status_t jit_gen9_common_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    const auto &jcp = ker_->jcp;

    std::unique_ptr<memory_storage_t> wht_work;
    std::unique_ptr<memory_storage_t> bias_work;
    std::unique_ptr<memory_storage_t> tails;

    if (jcp.ver == ver_8ow16c) {
        tails = ctx.get_scratchpad_grantor().get_memory_storage(key_conv_tails);

        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src);
        arg_list.set(1, *tails);
        status_t status = compute_stream->parallel_for(
                compute::nd_range_t({1}), load_tails_, arg_list);
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    if (jcp.ver == ver_16mb16c || jcp.ver == ver_8ow16c
            || jcp.ver == ver_1stconv) {
        wht_work = ctx.get_scratchpad_grantor().get_memory_storage(
                key_conv_wei_reduction);
        bias_work = ctx.get_scratchpad_grantor().get_memory_storage(
                key_conv_bia_reduction);

        arg_list.set(1, *wht_work);
        arg_list.set(2, *bias_work);
    } else {
        arg_list.set(1, diff_weights);
        arg_list.set(2, diff_bias);
    }
    arg_list.set(3, diff_dst);
    if (jcp.ver == ver_8ow16c) { arg_list.set(4, *tails); }

    status_t status = compute_stream->parallel_for(
            compute::nd_range_t(jcp.gws_d, jcp.lws_d), kernel_, arg_list);
    if (status != status::success) return status;

    if (jcp.ver == ver_16mb16c || jcp.ver == ver_8ow16c
            || jcp.ver == ver_1stconv) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, diff_weights);
        arg_list.set(1, *wht_work);
        arg_list.set(2, diff_bias);
        arg_list.set(3, *bias_work);
        status_t status = compute_stream->parallel_for(
                compute::nd_range_t(2, jcp.gws_d, jcp.lws_d), reduce_kernel_,
                arg_list);
        if (status != status::success) return status;
    }

    return status::success;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
