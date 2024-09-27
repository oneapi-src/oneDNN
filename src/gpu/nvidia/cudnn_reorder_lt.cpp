/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2024 Codeplay Software Limited
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

#include "gpu/nvidia/cudnn_reorder_lt.hpp"
#include "gpu/nvidia/stream.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream_utils.hpp"

#include "xpu/sycl/memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_reorder_lt_t::execute_internal_reorder(const exec_ctx_t &ctx,
        const memory_arg_t &src, const memory_arg_t &dst,
        const memory_arg_t *src_scales, const memory_arg_t *dst_scales) const {

    exec_args_t r_args;
    r_args[DNNL_ARG_SRC] = src;
    r_args[DNNL_ARG_DST] = dst;
    if (src_scales) r_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = *src_scales;
    if (dst_scales) r_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = *dst_scales;

    exec_ctx_t r_ctx(ctx, std::move(r_args));
    nested_scratchpad_t ns(
            ctx, memory_tracking::names::key_nested, generic_reorder_);

    r_ctx.set_scratchpad_grantor(ns.grantor());

    return generic_reorder_->execute(r_ctx);
}

status_t cudnn_reorder_lt_t::execute(const exec_ctx_t &ctx) const {
    memory_desc_wrapper wrap(pd()->src_md());
    if (wrap.size() == 0) { return status::success; }

    nvidia::stream_t *cuda_stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());

    auto arg_src_md = ctx.args().at(DNNL_ARG_SRC);
    auto arg_dst_md = ctx.args().at(DNNL_ARG_DST);
    auto arg_src_scale_md
            = ctx.args().find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto arg_dst_scale_md
            = ctx.args().find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    const memory_arg_t *src_scales = nullptr;
    if (arg_src_scale_md != ctx.args().end()) {
        src_scales = &arg_src_scale_md->second;
    }
    const memory_arg_t *dst_scales = nullptr;
    if (arg_dst_scale_md != ctx.args().end()) {
        dst_scales = &arg_dst_scale_md->second;
    }
    if (pd()->src_float_) {
        std::unique_ptr<memory_t, memory_deleter_t> scratch_mem;
        auto scratchpad_storage
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        memory_tracking::names::key_reorder_cublaslt_src_float);
        safe_ptr_assign(scratch_mem,
                new memory_t(ctx.stream()->engine(), &pd()->dst_scratch_md_,
                        std::move(scratchpad_storage)));
        auto arg_src_scratch_md = memory_arg_t {scratch_mem.get(), false};

        execute_internal_reorder(
                ctx, arg_src_md, arg_src_scratch_md, src_scales, dst_scales);
    }

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_src_scratch = CTX_SCRATCH_SYCL_MEMORY(
                memory_tracking::names::key_reorder_cublaslt_src_float);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
        auto arg_dst_scratch = CTX_SCRATCH_SYCL_MEMORY(
                memory_tracking::names::key_reorder_cublaslt_dst_float);

        auto arg_src_scale
                = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);

        auto arg_dst_scale
                = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

        compat::host_task(cgh, [=, this](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<nvidia::engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cublas_handle();

            void *src_ = pd()->src_float_
                    ? arg_src_scratch.get_native_pointer(ih)
                    : arg_src.get_native_pointer(ih);
            void *dst_ = pd()->dst_float_
                    ? arg_dst_scratch.get_native_pointer(ih)
                    : arg_dst.get_native_pointer(ih);

            auto a = static_cast<uint8_t *>(src_);
            auto b = static_cast<uint8_t *>(dst_);

            void *src_sc = arg_src_scale.get_native_pointer(ih);
            void *dst_sc = arg_dst_scale.get_native_pointer(ih);

            cublaslt_reorder_->execute(handle, a, b, src_sc, dst_sc);
        });

        if (pd()->dst_float_) {
            std::unique_ptr<memory_t, memory_deleter_t> scratch_mem;
            auto scratchpad_storage
                    = ctx.get_scratchpad_grantor().get_memory_storage(
                            memory_tracking::names::
                                    key_reorder_cublaslt_dst_float);
            safe_ptr_assign(scratch_mem,
                    new memory_t(ctx.stream()->engine(), &pd()->src_scratch_md_,
                            std::move(scratchpad_storage)));
            auto arg_dst_scratch_md = memory_arg_t {scratch_mem.get(), false};

            execute_internal_reorder(ctx, arg_dst_scratch_md, arg_dst_md,
                    src_scales, dst_scales);
        }
    });
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
