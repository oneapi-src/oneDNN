/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020-2022 Codeplay Software Limited
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

#include "gpu/nvidia/cudnn_binary.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_stream_utils.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_binary_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->src_md(0)).has_zero_dim())
        return status::success;

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    if (!pd()->attr()->scales_.get(DNNL_ARG_SRC_0).defined())
        CHECK(stream_utils::copy_input_arg_to_host(ctx, cuda_stream,
                &host_scales_[0], DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0,
                sizeof(float)));
    if (!pd()->attr()->scales_.get(DNNL_ARG_SRC_1).defined())
        CHECK(stream_utils::copy_input_arg_to_host(ctx, cuda_stream,
                &host_scales_[1], DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1,
                sizeof(float)));

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_src_0 = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC_0);
        auto arg_src_1 = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC_1);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            void *a = arg_src_0.get_native_pointer(ih);
            void *b = arg_src_1.get_native_pointer(ih);
            void *c = arg_dst.get_native_pointer(ih);

            pd()->binary_impl_->execute(
                    handle, a, b, c, &host_scales_[0], &host_scales_[1]);
        });
    });
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
