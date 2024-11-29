/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include "gpu/nvidia/cudnn_gemm.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_gemm_t::execute(const intel::gemm_exec_ctx_t &ctx) const {
    exec_args_t mm_args;
    std::unique_ptr<memory_t, memory_deleter_t> a, b, c, bias;

    safe_ptr_assign(a,
            new memory_t(ctx.stream()->engine(), pd()->mm_pd_->src_md(0),
                    ctx.args().a->clone()));

    safe_ptr_assign(b,
            new memory_t(ctx.stream()->engine(), pd()->mm_pd_->weights_md(0),
                    ctx.args().b->clone()));
    safe_ptr_assign(c,
            new memory_t(ctx.stream()->engine(), pd()->mm_pd_->dst_md(0),
                    ctx.args().c->clone()));

    mm_args[DNNL_ARG_SRC] = {a.get(), true};
    mm_args[DNNL_ARG_WEIGHTS] = {b.get(), true};
    mm_args[DNNL_ARG_DST] = {c.get(), false};

    if (ctx.args().bias) {
                safe_ptr_assign(bias,
                new memory_t(ctx.stream()->engine(),
                        pd()->mm_pd_->weights_md(1), ctx.args().bias->clone()));
        mm_args[DNNL_ARG_BIAS] = {bias.get(), true};
    }

    auto mm_exec_ctx = ctx.into_exec_ctx_t(std::move(mm_args));
    auto status = matmul_->execute(mm_exec_ctx);

    return status;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
