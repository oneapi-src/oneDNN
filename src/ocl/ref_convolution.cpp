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

namespace mkldnn {
namespace impl {
namespace ocl {

status_t ref_convolution_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(MKLDNN_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(MKLDNN_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(MKLDNN_ARG_DST);

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
    if (utils::one_of(pd()->src_md()->data_type, data_type::u8,
        data_type::s8)) {
        float scales = pd()->attr()->output_scales_.scales_[0];
        kernel_.set_arg(7, scales);
    }

    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

    const auto *jit_kernel = this->pd()->kernel();
    auto nd_range = cl_nd_range_t(jit_kernel->gws());
    status_t status = executor.parallel_for(nd_range, kernel_);
    return status;
}

status_t ref_convolution_bwd_data_t::execute_backward_data
    (const exec_ctx_t &ctx) const {

    auto &diff_dst = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(MKLDNN_ARG_WEIGHTS);
    auto &diff_src = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SRC);

    kernel_.set_arg(0, diff_src);
    kernel_.set_arg(1, weights);
    kernel_.set_arg(2, diff_dst);

    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

    const auto *jit_kernel = this->pd()->kernel();
    auto nd_range = cl_nd_range_t(jit_kernel->gws());
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

status_t ref_convolution_bwd_weights_t::execute_backward_weights
    (const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_BIAS);

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, diff_weights);
    kernel_.set_arg(2, diff_bias);
    kernel_.set_arg(3, diff_dst);

    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

    const auto *jit_kernel = this->pd()->kernel();
    auto nd_range = cl_nd_range_t(jit_kernel->gws());
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

}
}
}
