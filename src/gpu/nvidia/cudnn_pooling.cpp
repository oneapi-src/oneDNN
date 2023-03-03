/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "gpu/nvidia/cudnn_pooling.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"

#include "common/nstl.hpp"

#include "sycl/sycl_memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_pooling_fwd_t::execute(const exec_ctx_t &ctx) const {
    // If dst is empty, do nothing
    memory_desc_wrapper dst_wrap(pd()->dst_md());
    if (dst_wrap.size() == 0) return status::success;

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    memory_desc_wrapper src_wrap(pd()->src_md());
    auto dst_offset_bytes = src_wrap.nelems() * src_wrap.data_type_size();

    // If src is empty and dst is not, fill dst with
    // numeric_limits<dt>::lowest() to match the other backends' behaviour
    if (src_wrap.size() == 0 && dst_wrap.size() != 0) {
        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                        cuda_stream->engine());
                auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);

                void *dst = arg_dst.get_native_pointer(ih);

                if (dst_wrap.data_type() == data_type_t::dnnl_f32) {
                    auto val = nstl::numeric_limits<float>::lowest();
                    cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(dst),
                            reinterpret_cast<int &>(val), dst_wrap.nelems(),
                            cuda_stream->get_underlying_stream());
                } else if (dst_wrap.data_type() == data_type_t::dnnl_f16) {
                    float16_t val = nstl::numeric_limits<float16_t>::lowest();
                    cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(dst),
                            reinterpret_cast<unsigned short &>(val),
                            dst_wrap.nelems(),
                            cuda_stream->get_underlying_stream());
                } else if (dst_wrap.data_type() == data_type_t::dnnl_s8) {
                    auto val = nstl::numeric_limits<int8_t>::lowest();
                    cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(dst),
                            reinterpret_cast<unsigned char &>(val),
                            dst_wrap.nelems(),
                            cuda_stream->get_underlying_stream());
                } else if (dst_wrap.data_type() == data_type_t::dnnl_bf16) {
                    bfloat16_t val = nstl::numeric_limits<bfloat16_t>::lowest();
                    cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(dst),
                            reinterpret_cast<unsigned short &>(val),
                            dst_wrap.nelems(),
                            cuda_stream->get_underlying_stream());
                }
            });
        });
    }

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
        auto arg_wkspace = CTX_OUT_SYCL_MEMORY(DNNL_ARG_WORKSPACE);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            void *x = arg_src.get_native_pointer(ih);
            void *y = arg_dst.get_native_pointer(ih);

            uint8_t *ws_x = nullptr, *ws_y = nullptr;
            if (!arg_wkspace.empty()) {
                ws_x = static_cast<uint8_t *>(
                        arg_wkspace.get_native_pointer(ih));
                ws_y = ws_x + dst_offset_bytes;
            }

            pd()->pooling_impl_->execute(handle, x, y, ws_x, ws_y);
        });
    });
}

status_t cudnn_pooling_bwd_t::execute(const exec_ctx_t &ctx) const {
    if (has_zero_dims(pd()->diff_src_md()->dims, pd()->diff_src_md()->ndims)
            || has_zero_dims(
                    pd()->diff_dst_md()->dims, pd()->diff_dst_md()->ndims)) {
        return status::success;
    }

    memory_desc_wrapper wrap(pd()->diff_src_md());
    if (wrap.size() == 0) { return status::success; }
    const auto dst_offset_bytes = wrap.size();

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
        auto arg_wkspace = CTX_IN_SYCL_MEMORY(DNNL_ARG_WORKSPACE);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            void *dx = arg_diff_src.get_native_pointer(ih);
            void *dy = arg_diff_dst.get_native_pointer(ih);
            void *ws_x = arg_wkspace.get_native_pointer(ih);

            auto ws_y = (uint8_t *)ws_x + dst_offset_bytes;

            pd()->pooling_impl_->execute(handle, dx, dy, ws_x, ws_y);
        });
    });
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
