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

#include "ocl/ref_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type,
        data_type_t acc_type>
status_t ref_inner_product_fwd_t<src_type, wei_type, dst_type,
        acc_type>::execute_forward(const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(MKLDNN_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(MKLDNN_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(MKLDNN_ARG_DST);

    const auto &jip = ker_->jip;

    auto eltwise_alpha = pd()->eltwise_alpha();
    auto eltwise_beta = pd()->eltwise_beta();
    auto sum_scale = pd()->sum_scale();

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, weights);
    kernel_.set_arg(2, bias);
    kernel_.set_arg(3, dst);
    kernel_.set_arg(4, eltwise_alpha);
    kernel_.set_arg(5, eltwise_beta);
    kernel_.set_arg(6, sum_scale);

    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

    auto nd_range = cl_nd_range_t({ jip.mb * jip.oc });
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

template <data_type_t diff_src_type, data_type_t wei_type,
        data_type_t diff_dst_type, data_type_t acc_type>
status_t ref_inner_product_bwd_data_t<diff_src_type, wei_type, diff_dst_type,
        acc_type>::execute_backward_data(const exec_ctx_t &ctx) const {

    auto &diff_dst = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(MKLDNN_ARG_WEIGHTS);
    auto &diff_src = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SRC);

    const auto &jip = ker_->jip;

    kernel_.set_arg(0, diff_src);
    kernel_.set_arg(1, weights);
    kernel_.set_arg(2, diff_dst);

    auto nd_range
            = cl_nd_range_t({ jip.mb * jip.ic * jip.id * jip.ih * jip.iw });
    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

template <impl::data_type_t data_type>
status_t ref_inner_product_bwd_weights_t<data_type>::execute_backward_weights(
        const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_BIAS);

    const auto &jip = ker_->jip;

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, diff_weights);
    kernel_.set_arg(2, diff_bias);
    kernel_.set_arg(3, diff_dst);

    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

    auto nd_range
            = cl_nd_range_t({ jip.oc * jip.ic * jip.ih * jip.iw * jip.id });
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

using namespace data_type;
template struct ref_inner_product_fwd_t<f32>;
template struct ref_inner_product_fwd_t<f16>;
template struct ref_inner_product_bwd_data_t<f32, f32, f32, f32>;
template struct ref_inner_product_bwd_weights_t<f32>;

} // namespace ocl
} // namespace impl
} // namespace mkldnn
