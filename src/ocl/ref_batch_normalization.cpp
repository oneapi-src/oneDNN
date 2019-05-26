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

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "common/mkldnn_thread.hpp"
#include "common/mkldnn_traits.hpp"
#include "common/type_helpers.hpp"

#include "ocl/ref_batch_normalization.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <impl::data_type_t data_type>
status_t ref_batch_normalization_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);

    auto &mean_ = pd()->stats_is_src() ? CTX_IN_STORAGE(MKLDNN_ARG_MEAN)
                                       : CTX_OUT_STORAGE(MKLDNN_ARG_MEAN);

    auto &variance_ = pd()->stats_is_src()
            ? CTX_IN_STORAGE(MKLDNN_ARG_VARIANCE)
            : CTX_OUT_STORAGE(MKLDNN_ARG_VARIANCE);

    //auto idx_scaleshift = 1 + 2*pd()->stats_is_src();
    auto &scaleshift = CTX_IN_STORAGE(MKLDNN_ARG_SCALE_SHIFT);

    auto &dst = CTX_OUT_STORAGE(MKLDNN_ARG_DST);
    auto &ws = CTX_OUT_STORAGE(MKLDNN_ARG_WORKSPACE);

    const auto &jbn = ker_->jbn;

    auto *mean_ptr = &mean_;
    auto *variance_ptr = &variance_;
    if (jbn.use_16mb_unroll && jbn.calculate_stats && !jbn.save_stats) {
        mean_ptr = temp_reduce.get();
        variance_ptr = temp_reduce.get();
    }

    auto &mean = *mean_ptr;
    auto &variance = *variance_ptr;

    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

    if (jbn.use_16mb_unroll && jbn.calculate_stats) {
        status_t status;

        calculate_mean_kernel_.set_arg(0, src);
        calculate_mean_kernel_.set_arg(1, *temp_reduce);

        auto nd_range_mean = cl_nd_range_t(
                { jbn.sp_chunk, jbn.mb_chunk, jbn.ic }, { 1, 1, 16 });
        status = executor.parallel_for(nd_range_mean, calculate_mean_kernel_);
        if (status != status::success)
            return status;

        reduce_mean_kernel_.set_arg(0, *temp_reduce);
        reduce_mean_kernel_.set_arg(1, mean);

        status = executor.parallel_for(
                cl_nd_range_t({ jbn.ic }, { 1 }), reduce_mean_kernel_);
        if (status != status::success)
            return status;

        calculate_variance_kernel_.set_arg(0, src);
        calculate_variance_kernel_.set_arg(1, mean);
        calculate_variance_kernel_.set_arg(2, *temp_reduce);

        auto nd_range_calculate_variance = cl_nd_range_t(
                { jbn.sp_chunk, jbn.mb_chunk, jbn.ic }, { 1, 1, 16 });
        status = executor.parallel_for(
                nd_range_calculate_variance, calculate_variance_kernel_);
        if (status != status::success)
            return status;

        reduce_variance_kernel_.set_arg(0, *temp_reduce);
        reduce_variance_kernel_.set_arg(1, variance);

        auto nd_range_reduce_variance = cl_nd_range_t({ jbn.ic }, { 1 });
        status = executor.parallel_for(
                nd_range_reduce_variance, reduce_variance_kernel_);
        if (status != status::success)
            return status;
    }

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, mean);
    kernel_.set_arg(2, variance);
    kernel_.set_arg(3, dst);
    kernel_.set_arg(4, scaleshift);
    kernel_.set_arg(5, ws);
    kernel_.set_arg(6, jbn.eps);

    auto nd_range_kernel = cl_nd_range_t(jbn.gws_d, jbn.lws_d);
    status_t status = executor.parallel_for(nd_range_kernel, kernel_);

    return status;
}

template <impl::data_type_t data_type>
status_t ref_batch_normalization_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(MKLDNN_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(MKLDNN_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST);
    auto &scaleshift = CTX_IN_STORAGE(MKLDNN_ARG_SCALE_SHIFT);
    auto &ws = CTX_IN_STORAGE(MKLDNN_ARG_WORKSPACE);

    auto &diff_src = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SRC);
    auto &diff_scaleshift_ = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SCALE_SHIFT);

    const auto &jbn = ker_->jbn;

    auto &diff_scaleshift = (jbn.use_16mb_unroll && !jbn.diff_scaleshift)
            ? *temp_reduce
            : diff_scaleshift_;

    if (jbn.use_16mb_unroll) {
        status_t status;

        calculate_stats_kernel_.set_arg(0, src);
        calculate_stats_kernel_.set_arg(1, mean);
        calculate_stats_kernel_.set_arg(2, diff_dst);
        calculate_stats_kernel_.set_arg(3, ws);
        calculate_stats_kernel_.set_arg(4, *temp_reduce);

        auto &executor = *(
                utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());
        auto nd_range = cl_nd_range_t(
                { jbn.sp_chunk, jbn.mb_chunk, jbn.ic }, { 1, 1, 16 });
        status = executor.parallel_for(nd_range, calculate_stats_kernel_);
        if (status != status::success)
            return status;

        reduce_stats_kernel_.set_arg(0, *temp_reduce);
        reduce_stats_kernel_.set_arg(1, diff_scaleshift);
        reduce_stats_kernel_.set_arg(2, variance);
        reduce_stats_kernel_.set_arg(3, jbn.eps);

        status = executor.parallel_for(
                cl_nd_range_t({ jbn.ic }, { 1 }), reduce_stats_kernel_);
        if (status != status::success)
            return status;
    }

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, mean);
    kernel_.set_arg(2, variance);
    kernel_.set_arg(3, diff_dst);
    kernel_.set_arg(4, scaleshift);
    kernel_.set_arg(5, ws);
    kernel_.set_arg(6, diff_src);
    kernel_.set_arg(7, diff_scaleshift);
    kernel_.set_arg(8, jbn.eps);

    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

    auto kernel_nd_range = cl_nd_range_t(jbn.gws_d, jbn.lws_d);
    status_t status = executor.parallel_for(kernel_nd_range, kernel_);

    return status;
}

template struct ref_batch_normalization_fwd_t<data_type::f16>;
template struct ref_batch_normalization_fwd_t<data_type::f32>;
template struct ref_batch_normalization_bwd_t<data_type::f32>;

} // namespace ocl
} // namespace impl
} // namespace mkldnn

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
