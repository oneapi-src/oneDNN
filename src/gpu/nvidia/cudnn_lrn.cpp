/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "gpu/nvidia/cudnn_lrn.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"

#include "sycl_cuda_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_lrn_fwd_t::execute(const exec_ctx_t &ctx) const {

    if (memory_desc_wrapper(pd()->desc()->data_desc).has_zero_dim())
        return status::success;

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        auto *mem_src = static_cast<sycl::sycl_memory_storage_base_t *>(
                &CTX_IN_STORAGE(DNNL_ARG_SRC));
        auto src_acc
                = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(DNNL_ARG_SRC))>(
                        mem_src, cgh);

        auto *mem_dst = static_cast<sycl::sycl_memory_storage_base_t *>(
                &CTX_OUT_STORAGE(DNNL_ARG_DST));
        auto dst_acc
                = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(DNNL_ARG_DST))>(
                        mem_dst, cgh);

        auto *mem_wrksp = pd()->is_training()
                ? static_cast<sycl::sycl_memory_storage_base_t *>(
                        &CTX_OUT_STORAGE(DNNL_ARG_DST))
                : mem_dst;
        auto wrksp_acc = pd()->is_training()
                ? get_cudnn_accessor<decltype(
                        CTX_OUT_ACCESSOR(DNNL_ARG_WORKSPACE))>(mem_wrksp, cgh)
                : dst_acc;

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            void *dst_ = get_cudnn_ptr(sc, ih, dst_acc, mem_dst);
            void *src_ = get_cudnn_ptr(sc, ih, src_acc, mem_src);
            void *ws_ = get_cudnn_ptr(sc, ih, wrksp_acc, mem_wrksp);

            std::vector<void *> args {src_, dst_, ws_};
            pd()->lrn_impl_->execute(handle, args);
        });
    });
}

status_t cudnn_lrn_bwd_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->desc()->data_desc).has_zero_dim())
        return status::success;

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        auto *mem_src = static_cast<sycl::sycl_memory_storage_base_t *>(
                &CTX_IN_STORAGE(DNNL_ARG_SRC));
        auto src_acc
                = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(DNNL_ARG_SRC))>(
                        mem_src, cgh);

        auto *diff_mem_dst = static_cast<sycl::sycl_memory_storage_base_t *>(
                &CTX_IN_STORAGE(DNNL_ARG_DIFF_DST));
        auto diff_dst_acc = get_cudnn_accessor<decltype(
                CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST))>(diff_mem_dst, cgh);

        auto *diff_mem_src = static_cast<sycl::sycl_memory_storage_base_t *>(
                &CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC));
        auto diff_src_acc = get_cudnn_accessor<decltype(
                CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC))>(diff_mem_src, cgh);

        auto *mem_ws = static_cast<sycl::sycl_memory_storage_base_t *>(
                &CTX_IN_STORAGE(DNNL_ARG_WORKSPACE));
        auto ws_acc = get_cudnn_accessor<decltype(
                CTX_IN_ACCESSOR(DNNL_ARG_WORKSPACE))>(mem_ws, cgh);

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            std::vector<void *> args;
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            args.push_back(get_cudnn_ptr(sc, ih, src_acc, mem_src));
            args.push_back(get_cudnn_ptr(sc, ih, ws_acc, mem_ws));
            args.push_back(get_cudnn_ptr(sc, ih, diff_src_acc, diff_mem_src));
            args.push_back(get_cudnn_ptr(sc, ih, diff_dst_acc, diff_mem_dst));

            pd()->lrn_impl_->execute(handle, args);
        });
    });
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
