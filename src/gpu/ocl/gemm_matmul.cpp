/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "gpu/ocl/gemm_matmul.hpp"

#include "gpu/gemm/gpu_gemm.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gemm_matmul_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC);
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS);
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST);
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS);

    memory_storage_t *a0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);

    memory_storage_t *b0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);

    memory_storage_t *c0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    gemm_exec_args_t gemm_args;
    gemm_args.a = &CTX_IN_STORAGE(DNNL_ARG_SRC);
    gemm_args.b = &CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    gemm_args.c = &CTX_OUT_STORAGE(DNNL_ARG_DST);
    gemm_args.bias = &CTX_IN_STORAGE(DNNL_ARG_BIAS);

    // Note: we have to swap `a` and `b` zero-point arguments because,
    // - gemm primitive is created with row major desc,
    // - parameters to gemm are passed as row major
    // - but gemm implementation assumes column major
    gemm_args.a_zero_point = b0;
    gemm_args.b_zero_point = a0;
    gemm_args.c_zero_point = c0;
    gemm_args.a_scales
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    gemm_args.b_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    gemm_args.c_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    gemm_args.exec_args = ctx.args();
    auto gemm_desc = create_gemm_desc(src_d.md_, weights_d.md_, dst_d.md_,
            bia_d.md_, pd()->desc()->accum_data_type, ctx.stream()->engine());

    gemm_exec_ctx_t gemm_ctx(ctx, gemm_args, &gemm_desc);

    nested_scratchpad_t ns(ctx, key_nested, gemm_);
    gemm_ctx.set_scratchpad_grantor(ns.grantor());

    status_t gemm_exec_status = gpu_gemm(gemm_)->execute(gemm_ctx);
    if (gemm_exec_status != status::success) return gemm_exec_status;

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
