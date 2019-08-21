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

#include "ocl/gemm_x8s8s32x_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

status_t gemm_x8s8s32x_inner_product_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    exec_args_t gemm_args;
    gemm_args[DNNL_ARG_SRC_0] = ctx.args().at(DNNL_ARG_WEIGHTS);
    gemm_args[DNNL_ARG_SRC_1] = ctx.args().at(DNNL_ARG_SRC);
    gemm_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);

    exec_ctx_t gemm_ctx(ctx.stream(), std::move(gemm_args));
    status_t gemm_exec_status = gemm_->execute(gemm_ctx);
    if (gemm_exec_status != status::success) return gemm_exec_status;

    // TODO: post proccess kernel should be here.
    return status::success;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl
