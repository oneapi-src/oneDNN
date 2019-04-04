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

#include "ocl/ocl_stream.hpp"

#include "ocl/gemm_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type,
        data_type_t acc_type>
status_t gemm_inner_product_fwd_t<src_type, wei_type, dst_type,
        acc_type>::execute_forward(const exec_ctx_t &ctx) const {
    exec_args_t gemm_args;
    gemm_args[MKLDNN_ARG_SRC_0] = ctx.args().at(MKLDNN_ARG_WEIGHTS);
    gemm_args[MKLDNN_ARG_SRC_1] = ctx.args().at(MKLDNN_ARG_SRC);
    gemm_args[MKLDNN_ARG_DST] = ctx.args().at(MKLDNN_ARG_DST);

    exec_ctx_t gemm_ctx(ctx.stream(), std::move(gemm_args));
    status_t gemm_exec_status = gemm_->execute(gemm_ctx);
    if (gemm_exec_status != status::success)
        return gemm_exec_status;

    if (pd()->with_bias()) {
        auto &bias = CTX_IN_STORAGE(MKLDNN_ARG_BIAS);
        auto &dst = CTX_OUT_STORAGE(MKLDNN_ARG_DST);

        bias_kernel_.set_arg(0, bias);
        bias_kernel_.set_arg(1, dst);

        auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

        auto nd_range = cl_nd_range_t({ pd()->MB() * pd()->OC() });
        status_t bias_status = executor.parallel_for(nd_range, bias_kernel_);
        if (bias_status != status::success)
            return bias_status;
    }

    return status::success;
}

using namespace data_type;
template struct gemm_inner_product_fwd_t<f16>;
template struct gemm_inner_product_fwd_t<f32>;

template <data_type_t diff_src_type, data_type_t wei_type,
        data_type_t diff_dst_type, data_type_t acc_type>
status_t gemm_inner_product_bwd_data_t<diff_src_type, wei_type, diff_dst_type,
        acc_type>::execute_backward_data(const exec_ctx_t &ctx) const {
    exec_args_t gemm_args;
    gemm_args[MKLDNN_ARG_SRC_0] = ctx.args().at(MKLDNN_ARG_WEIGHTS);
    gemm_args[MKLDNN_ARG_SRC_1] = ctx.args().at(MKLDNN_ARG_DIFF_DST);
    gemm_args[MKLDNN_ARG_DST] = ctx.args().at(MKLDNN_ARG_DIFF_SRC);

    exec_ctx_t gemm_ctx(ctx.stream(), std::move(gemm_args));
    status_t gemm_exec_status = gemm_->execute(gemm_ctx);
    if (gemm_exec_status != status::success)
        return gemm_exec_status;

    return status::success;
}

template struct gemm_inner_product_bwd_data_t<f32>;

template <data_type_t data_type>
status_t gemm_inner_product_bwd_weights_t<data_type>::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    exec_args_t gemm_args;
    if (pd()->wei_tr()) {
        gemm_args[MKLDNN_ARG_SRC_0] = ctx.args().at(MKLDNN_ARG_DIFF_DST);
        gemm_args[MKLDNN_ARG_SRC_1] = ctx.args().at(MKLDNN_ARG_SRC);
    } else {
        gemm_args[MKLDNN_ARG_SRC_0] = ctx.args().at(MKLDNN_ARG_SRC);
        gemm_args[MKLDNN_ARG_SRC_1] = ctx.args().at(MKLDNN_ARG_DIFF_DST);
    }
    gemm_args[MKLDNN_ARG_DST] = ctx.args().at(MKLDNN_ARG_DIFF_WEIGHTS);

    exec_ctx_t gemm_ctx(ctx.stream(), std::move(gemm_args));
    status_t gemm_exec_status = gemm_->execute(gemm_ctx);
    if (gemm_exec_status != status::success)
        return gemm_exec_status;

    if (pd()->with_bias()) {
        auto &diff_dst = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST);
        auto &diff_bias = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_BIAS);

        bias_kernel_.set_arg(0, diff_dst);
        bias_kernel_.set_arg(1, diff_bias);

        auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

        auto nd_range = cl_nd_range_t({ pd()->OC() });
        status_t bias_status = executor.parallel_for(nd_range, bias_kernel_);
        if (bias_status != status::success)
            return bias_status;
    }

    return status::success;
}

template struct gemm_inner_product_bwd_weights_t<f32>;
}
}
}
