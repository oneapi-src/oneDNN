/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "gpu/nvidia/cudnn_matmul_executor.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_stream_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_matmul_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    const bool has_runtime_args = matmul_impl_->has_runtime_params();

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bias_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    status_t status;
    size_t scratchpad_size = 0; // To avoid extra allocation in an executor.
    if (has_runtime_args) {
        // Initialise all runtime parameters
        status = matmul_impl_->init_parameters(src_d, weights_d, dst_d, bias_d);
        if (status != status::success) return status;

        scratchpad_size = pd()->scratchpad_size(dst_d.md_);
    }

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    status = executor_->execute(
            ctx, ctx.stream()->engine(), matmul_impl_, scratchpad_size);

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
