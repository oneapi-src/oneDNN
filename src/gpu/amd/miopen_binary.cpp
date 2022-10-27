/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "gpu/amd/miopen_binary.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

status_t miopen_binary_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->src_md(0)).has_zero_dim())
        return status::success;
    amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_src_0 = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC_0);
        auto arg_src_1 = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC_1);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
        auto arg_scale0
                = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0);
        auto arg_scale1
                = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                    hip_stream->engine());
            auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();

            void *a = arg_src_0.get_native_pointer(ih);
            void *b = arg_src_1.get_native_pointer(ih);
            void *c = arg_dst.get_native_pointer(ih);
            void *s0 = arg_scale0.get_native_pointer(ih);
            void *s1 = arg_scale1.get_native_pointer(ih);

            pd()->binary_impl_->execute(handle, a, b, c, s0, s1);
        });
    });
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
