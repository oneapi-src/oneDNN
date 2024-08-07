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

    status_t status;
    size_t bias_scratchpad_size
            = 0; // To avoid extra allocation in an executor.

    bool has_runtime_args = matmul_impl_->has_runtime_params();
    if (has_runtime_args) {
        // Initialise all runtime parameters
        status = matmul_impl_->init_parameters(src_d, weights_d, dst_d, bias_d);
        if (status != status::success) return status;

        bias_scratchpad_size = matmul_impl_->bias_scratch_size();
    }

    nvidia::stream_t *cuda_stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());

    status = executor_->execute(
            ctx, ctx.stream()->engine(), matmul_impl_, bias_scratchpad_size);

    if (has_runtime_args) {
        auto &evts = cuda_stream->sycl_ctx().get_sycl_deps().events;
        for (auto e : evts) {
            e.wait();
        }
        matmul_impl_->cleanup();
    }
    return status;
}

status_t cudnn_matmul_lt_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bias_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    status_t status = status_t::dnnl_success;
    // To avoid extra allocation in an executor.
    size_t algo_scratchpad_size = 0;
    size_t bias_scratchpad_size = 0;
    size_t block_a_scratchpad_size = 0;
    size_t block_b_scratchpad_size = 0;
    size_t block_c_scratchpad_size = 0;
    size_t src_scale_scratchpad_size = 0;
    size_t wei_scale_scratchpad_size = 0;
    
    bool has_runtime_args = matmul_impl_->has_runtime_params();
    if (has_runtime_args) {
        // Initialise all runtime parameters
        auto engine = ctx.stream()->engine();
        status = matmul_impl_->init_parameters(
                src_d, weights_d, dst_d, bias_d, engine);
        if (status != status::success) return status;

        matmul_impl_->handle_imma_case(src_d, weights_d, dst_d);

        algo_scratchpad_size = matmul_impl_->algo_scratch_size();
        bias_scratchpad_size = matmul_impl_->bias_scratch_size();
        block_a_scratchpad_size = matmul_impl_->block_a_scratch_size();
        block_b_scratchpad_size = matmul_impl_->block_b_scratch_size();
        block_c_scratchpad_size = matmul_impl_->block_c_scratch_size();
        src_scale_scratchpad_size = matmul_impl_->src_scale_size();
        wei_scale_scratchpad_size = matmul_impl_->wei_scale_size();
    }

    nvidia::stream_t *cuda_stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());

    if (matmul_impl_->multi_src_scale()) {
        // src scale sycl binary
        exec_args_t src_scale_binary_args;
        src_scale_binary_args[DNNL_ARG_SRC_0]
                = memory_arg_t {ctx.args().at(DNNL_ARG_SRC).mem, true};
        src_scale_binary_args[DNNL_ARG_SRC_1] = memory_arg_t {
                ctx.args().at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC).mem, true};

        std::unique_ptr<memory_t> scratch_mem;
        auto scratchpad_storage
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        memory_tracking::names::key_matmul_lt_src_scale);
        safe_ptr_assign(scratch_mem,
                new memory_t(ctx.stream()->engine(), pd()->src_md(),
                        std::move(scratchpad_storage)));
        src_scale_binary_args[DNNL_ARG_DST]
                = memory_arg_t {scratch_mem.get(), false};

        exec_ctx_t binary_ctx(ctx, std::move(src_scale_binary_args));

        status = src_scale_binary_->execute(binary_ctx);
    }
    if (matmul_impl_->multi_wei_scale() && status == status_t::dnnl_success) {
        // wei scale sycl binary
        exec_args_t wei_scale_binary_args;
        wei_scale_binary_args[DNNL_ARG_SRC_0]
                = memory_arg_t {ctx.args().at(DNNL_ARG_WEIGHTS).mem, true};
        wei_scale_binary_args[DNNL_ARG_SRC_1] = memory_arg_t {
                ctx.args().at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS).mem,
                true};

        std::unique_ptr<memory_t> scratch_mem;
        auto scratchpad_storage
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        memory_tracking::names::key_matmul_lt_wei_scale);
        safe_ptr_assign(scratch_mem,
                new memory_t(ctx.stream()->engine(), pd()->weights_md(0),
                        std::move(scratchpad_storage)));
        wei_scale_binary_args[DNNL_ARG_DST]
                = memory_arg_t {scratch_mem.get(), false};

        exec_ctx_t binary_ctx(ctx, std::move(wei_scale_binary_args));

        status = wei_scale_binary_->execute(binary_ctx);
    }

    if (status == status_t::dnnl_success) {
        status = executor_->execute(ctx, ctx.stream()->engine(), matmul_impl_,
                algo_scratchpad_size, bias_scratchpad_size,
                block_a_scratchpad_size, block_b_scratchpad_size,
                block_c_scratchpad_size, src_scale_scratchpad_size,
                wei_scale_scratchpad_size);
    } else {
        return status_t::dnnl_runtime_error;
    }

    if (matmul_impl_->with_bias() && status == dnnl_success) {
        // bias sycl binary
        exec_args_t binary_args;
        binary_args[DNNL_ARG_SRC_0]
                = memory_arg_t {ctx.args().at(DNNL_ARG_DST).mem, true};
        binary_args[DNNL_ARG_SRC_1] = ctx.args().at(DNNL_ARG_BIAS);
        binary_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);

        exec_ctx_t binary_ctx(ctx, std::move(binary_args));

        status = binary_->execute(binary_ctx);
    }

    if (matmul_impl_->multi_dst_scale() && status == status_t::dnnl_success) {
        // dst scale sycl binary
        exec_args_t dst_scale_binary_args;
        dst_scale_binary_args[DNNL_ARG_SRC_0]
                = memory_arg_t {ctx.args().at(DNNL_ARG_DST).mem, true};
        dst_scale_binary_args[DNNL_ARG_SRC_1] = memory_arg_t {
                ctx.args().at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST).mem, true};
        dst_scale_binary_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);

        exec_ctx_t binary_ctx(ctx, std::move(dst_scale_binary_args));

        status = dst_scale_binary_->execute(binary_ctx);
    }

    if (has_runtime_args) {
        auto &evts = cuda_stream->sycl_ctx().get_sycl_deps().events;
        for (auto e : evts) {
            e.wait();
        }

        matmul_impl_->cleanup();
    }

    return status;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
