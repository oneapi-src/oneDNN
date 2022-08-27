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

#include "gpu/amd/miopen_pooling.hpp"
#include "common/nstl.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

status_t miopen_pooling_fwd_t::execute(const exec_ctx_t &ctx) const {
    // If dst is empty, do nothing
    memory_desc_wrapper dst_wrap(pd()->dst_md());
    if (dst_wrap.size() == 0) return status::success;

    amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

    memory_desc_wrapper src_wrap(pd()->src_md());

    if (src_wrap.size() == 0 && dst_wrap.size() != 0) {
        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);

                void *dst = arg_dst.get_native_pointer(ih);
                if (dst_wrap.data_type() == data_type_t::dnnl_f32) {
                    auto val = nstl::numeric_limits<float>::lowest();
                    HIP_EXECUTE_FUNC(hipMemsetD16Async,
                            reinterpret_cast<hipDeviceptr_t>(dst),
                            reinterpret_cast<int &>(val), dst_wrap.nelems(),
                            hip_stream->get_underlying_stream());
                } else if (dst_wrap.data_type() == data_type_t::dnnl_f16) {
                    float16_t val = nstl::numeric_limits<float16_t>::lowest();
                    HIP_EXECUTE_FUNC(hipMemsetD16Async,
                            reinterpret_cast<hipDeviceptr_t>(dst),
                            reinterpret_cast<unsigned short &>(val),
                            dst_wrap.nelems(),
                            hip_stream->get_underlying_stream());
                } else if (dst_wrap.data_type() == data_type_t::dnnl_s8) {
                    auto val = nstl::numeric_limits<int8_t>::lowest();
                    HIP_EXECUTE_FUNC(hipMemsetD16Async,
                            reinterpret_cast<hipDeviceptr_t>(dst),
                            reinterpret_cast<unsigned char &>(val),
                            dst_wrap.nelems(),
                            hip_stream->get_underlying_stream());
                }
            });
        });
    }

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
        auto arg_wkspace = CTX_OUT_SYCL_MEMORY(DNNL_ARG_WORKSPACE);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                    hip_stream->engine());
            auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();
            void *x = arg_src.get_native_pointer(ih);
            void *y = arg_dst.get_native_pointer(ih);
            void *ws = arg_wkspace.get_native_pointer(ih);
            pd()->pooling_impl_->execute(handle, x, y, ws);
        });
    });
}

status_t miopen_pooling_bwd_t::execute(const exec_ctx_t &ctx) const {
    if (has_zero_dims(pd()->diff_src_md()->dims, pd()->diff_src_md()->ndims)
            || has_zero_dims(
                    pd()->diff_dst_md()->dims, pd()->diff_dst_md()->ndims)) {
        return status::success;
    }

    memory_desc_wrapper wrap(pd()->diff_src_md());
    if (wrap.size() == 0) { return status::success; }
    amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
        auto arg_wkspace = CTX_IN_SYCL_MEMORY(DNNL_ARG_WORKSPACE);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                    hip_stream->engine());
            auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = hip_stream->get_miopen_handle();
            void *dx = arg_diff_src.get_native_pointer(ih);
            void *dy = arg_diff_dst.get_native_pointer(ih);
            void *ws = arg_wkspace.get_native_pointer(ih);

            pd()->pooling_impl_->execute(handle, dx, dy, ws);
        });
    });
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
