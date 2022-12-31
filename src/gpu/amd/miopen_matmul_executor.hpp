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

#ifndef GPU_AMD_MIOPEN_MATMUL_EXECUTOR_HPP
#define GPU_AMD_MIOPEN_MATMUL_EXECUTOR_HPP

#include "gpu/amd/miopen_matmul.hpp"
#include "gpu/amd/miopen_matmul_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"

#include <memory>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_matmul_exec_base_t {
    virtual status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<miopen_matmul_impl_t> matmul_impl_,
            std::size_t scratchpad_size)
            = 0;
    virtual ~miopen_matmul_exec_base_t() = default;

protected:
    template <::sycl::access::mode bias_m, ::sycl::access::mode scratch_m>
    void interop_task(std::shared_ptr<miopen_matmul_impl_t> matmul_impl_,
            engine_t *engine, ::sycl::handler &cgh,
            amd::sycl_hip_stream_t *hip_stream,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>
                    arg_weights,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read> arg_src,
            impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write> arg_dst,
            impl::sycl::sycl_memory_arg_t<bias_m> arg_bias,
            impl::sycl::sycl_memory_arg_t<scratch_m> arg_scratch) {

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                    hip_stream->engine());
            auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
            auto native_stream = hip_stream->get_underlying_stream();
            auto rocblas_handle = hip_stream->get_rocblas_handle(native_stream);
            auto miopen_handle = hip_stream->get_miopen_handle(native_stream);

            void *scratch = arg_scratch.get_native_pointer(ih);
            void *bias = arg_bias.get_native_pointer(ih);
            void *weights = arg_weights.get_native_pointer(ih);
            void *src = arg_src.get_native_pointer(ih);
            void *dst = arg_dst.get_native_pointer(ih);

            matmul_impl_->execute(rocblas_handle, miopen_handle, weights, src,
                    dst, bias, scratch);
        });
    }
};

struct miopen_matmul_scratch_runtime_args_base_exec_t
    : public miopen_matmul_exec_base_t {
    virtual status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<miopen_matmul_impl_t> matmul_impl_,
            std::size_t scratchpad_size)
            = 0;
    virtual ~miopen_matmul_scratch_runtime_args_base_exec_t() = default;

protected:
    void init_scratch_buffer(std::size_t scratch_size) {
        if (scratch_size > 0) {
            scratch_buff_.reset(new ::sycl::buffer<uint8_t, 1>(scratch_size));
        }
    }

    std::shared_ptr<::sycl::buffer<uint8_t, 1>> scratch_buff_ {nullptr};
};

struct miopen_matmul_scratch_runtime_args_bias_exec_t
    : public miopen_matmul_scratch_runtime_args_base_exec_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<miopen_matmul_impl_t> matmul_impl_,
            std::size_t scratchpad_size) override {

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        init_scratch_buffer(scratchpad_size);

        return hip_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);
            auto arg_scratch = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>(*scratch_buff_, cgh);

            interop_task(matmul_impl_, engine, cgh, hip_stream, arg_wt, arg_src,
                    arg_dst, arg_bias, arg_scratch);
        });
    }
};

struct miopen_matmul_runtime_args_scratch_exec_t
    : public miopen_matmul_scratch_runtime_args_base_exec_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<miopen_matmul_impl_t> matmul_impl_,
            std::size_t scratchpad_size) override {

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        init_scratch_buffer(scratchpad_size);

        return hip_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_scratch = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>(*scratch_buff_, cgh);
            auto arg_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read>();

            interop_task(matmul_impl_, engine, cgh, hip_stream, arg_wt, arg_src,
                    arg_dst, /*nullptr*/ arg_bias, arg_scratch);
        });
    }
};

struct miopen_matmul_runtime_args_bias_exec_t
    : public miopen_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<miopen_matmul_impl_t> matmul_impl_,
            std::size_t scratchpad_size) override {

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);

            auto arg_scratch = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>();

            interop_task(matmul_impl_, engine, cgh, hip_stream, arg_wt, arg_src,
                    arg_dst, arg_bias, /*nullptr*/ arg_scratch);
        });
    }
};

struct miopen_matmul_runtime_args_exec_t : public miopen_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<miopen_matmul_impl_t> matmul_impl_,
            std::size_t scratchpad_size) override {

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

            auto arg_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read>();
            auto arg_scratch = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>();

            interop_task(matmul_impl_, engine, cgh, hip_stream, arg_wt, arg_src,
                    arg_dst, /*nullptr*/ arg_bias,
                    /*nullptr*/ arg_scratch);
        });
    }
};

struct miopen_matmul_bias_scratch_exec_t : public miopen_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<miopen_matmul_impl_t> matmul_impl_,
            std::size_t scratchpad_size) override {

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);
            auto arg_scratch = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_dst_in_acc_dt);

            interop_task(matmul_impl_, engine, cgh, hip_stream, arg_wt, arg_src,
                    arg_dst, arg_bias, arg_scratch);
        });
    }
};

struct miopen_matmul_scratch_exec_t : public miopen_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<miopen_matmul_impl_t> matmul_impl_,
            std::size_t scratchpad_size) override {

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_scratch = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_dst_in_acc_dt);

            auto arg_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read>();

            interop_task(matmul_impl_, engine, cgh, hip_stream, arg_wt, arg_src,
                    arg_dst, /*nullptr*/ arg_bias, arg_scratch);
        });
    }
};

struct miopen_matmul_bias_exec_t : public miopen_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<miopen_matmul_impl_t> matmul_impl_,
            std::size_t scratchpad_size) override {

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);

            auto arg_scratch = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>();

            interop_task(matmul_impl_, engine, cgh, hip_stream, arg_wt, arg_src,
                    arg_dst, arg_bias, /*nullptr*/ arg_scratch);
        });
    }
};

struct miopen_matmul_exec_t : public miopen_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<miopen_matmul_impl_t> matmul_impl_,
            std::size_t scratchpad_size) override {

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

            auto arg_bias = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read>();
            auto arg_scratch = impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>();

            interop_task(matmul_impl_, engine, cgh, hip_stream, arg_wt, arg_src,
                    arg_dst, /*nullptr*/ arg_bias,
                    /*nullptr*/ arg_scratch);
        });
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
