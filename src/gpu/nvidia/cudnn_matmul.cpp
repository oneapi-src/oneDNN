/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "gpu/nvidia/cudnn_matmul.hpp"
#include "gpu/nvidia/cudnn_matmul_lt.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "gpu/nvidia/cudnn_matmul_executor.hpp"
#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/stream.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_matmul_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bias_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    nvidia::stream_t *cuda_stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());

    status_t status = executor_->execute(ctx, ctx.stream()->engine(),
            matmul_impl_, pd()->params_, src_d, weights_d, dst_d, bias_d);

    if (pd()->params_->has_runtime_params_) {
        auto &evts = cuda_stream->sycl_ctx().get_sycl_deps().events;
        for (auto e : evts) {
            e.wait();
        }
    }
    return status;
}

status_t cudnn_matmul_lt_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    nvidia::stream_t *cuda_stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());

    const bool has_dst_scales
            = ctx.args().find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)
            != ctx.args().end();

    CHECK(executor_->execute(ctx, ctx.stream()->engine(), matmul_impl_,
            pd()->params_, src_d, weights_d, dst_d));

    if (pd()->params_->with_bias_) {
        // bias sycl binary
        exec_args_t binary_args;
        std::unique_ptr<memory_t, memory_deleter_t> scratch_mem;
        if (dst_d.data_type() == dnnl_s8) {
            auto scratchpad_storage
                    = ctx.get_scratchpad_grantor().get_memory_storage(
                            memory_tracking::names::key_matmul_dst_in_acc_dt);

            safe_ptr_assign(scratch_mem,
                    new memory_t(ctx.stream()->engine(), &pd()->s32_dst_md_,
                            std::move(scratchpad_storage)));
            binary_args[DNNL_ARG_SRC_0]
                    = memory_arg_t {scratch_mem.get(), true};
        } else {
            binary_args[DNNL_ARG_SRC_0]
                    = memory_arg_t {ctx.args().at(DNNL_ARG_DST).mem, true};
        }
        binary_args[DNNL_ARG_SRC_1] = ctx.args().at(DNNL_ARG_BIAS);
        binary_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);

        exec_ctx_t binary_ctx(ctx, std::move(binary_args));

        CHECK(binary_->execute(binary_ctx));
    }

    if (has_dst_scales
            && (pd()->params_->multi_dst_scale_
                    || pd()->params_->acc_type_ == CUDA_R_32I)) {
        // dst scale sycl binary
        exec_args_t dst_scale_binary_args;
        dst_scale_binary_args[DNNL_ARG_SRC_0]
                = memory_arg_t {ctx.args().at(DNNL_ARG_DST).mem, true};
        dst_scale_binary_args[DNNL_ARG_SRC_1] = memory_arg_t {
                ctx.args().at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST).mem, true};
        dst_scale_binary_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);

        exec_ctx_t binary_ctx(ctx, std::move(dst_scale_binary_args));

        CHECK(dst_scale_binary_->execute(binary_ctx));
    }

    if (pd()->params_->has_runtime_params_) {
        auto &evts = cuda_stream->sycl_ctx().get_sycl_deps().events;
        for (auto e : evts) {
            e.wait();
        }
    }

    return status::success;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
