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

#ifndef GPU_NVIDIA_CUDNN_MATMUL_EXECUTOR_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_EXECUTOR_HPP

#include "common/primitive_exec_types.hpp"
#include "gpu/nvidia/cudnn_matmul.hpp"
#include "gpu/nvidia/cudnn_matmul_impl.hpp"
#include "gpu/nvidia/cudnn_matmul_lt_impl.hpp"
#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/stream.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "xpu/sycl/memory_storage_helper.hpp"

#include <memory>

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_matmul_base_exec_t {

    virtual status_t execute(const exec_ctx_t &ctx, impl::engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            std::size_t bias_scratch_size)
            = 0;

protected:
    template <::sycl::access::mode bias_m, ::sycl::access::mode scratch_m>
    void interop_task(std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            impl::engine_t *engine, ::sycl::handler &cgh,
            nvidia::stream_t *cuda_stream,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read>
                    arg_weights,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read> arg_src,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::write>
                    arg_dst,
            xpu::sycl::interop_memory_arg_t<bias_m> arg_bias,
            xpu::sycl::interop_memory_arg_t<scratch_m> arg_bias_scratch,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read>
                    arg_src_scale,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read>
                    arg_wei_scale,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read>
                    arg_dst_scale) {

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<nvidia::engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            // SYCL out-of-order queue encapsulates multiple CUstream objects.
            // Every query of the CUstream object can return any of those
            // therefore we need to make sure that we activate both cuDNN and
            // cuBLAS handles for the same CUstream object.
            auto native_stream = cuda_stream->get_underlying_stream();
            auto cublas_handle = cuda_stream->get_cublas_handle(native_stream);
            auto cudnn_handle = cuda_stream->get_cudnn_handle(native_stream);

            void *reorder_scratch = arg_bias_scratch.get_native_pointer(ih);
            void *bias = arg_bias.get_native_pointer(ih);
            void *weights = arg_weights.get_native_pointer(ih);
            void *src = arg_src.get_native_pointer(ih);
            void *dst = arg_dst.get_native_pointer(ih);

            void *src_scale = arg_src_scale.get_native_pointer(ih);
            void *wei_scale = arg_wei_scale.get_native_pointer(ih);
            void *dst_scale = arg_dst_scale.get_native_pointer(ih);

            matmul_impl_->execute(cublas_handle, cudnn_handle, weights, src,
                    dst, bias, reorder_scratch, src_scale, wei_scale,
                    dst_scale);
        });
    }

    xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read_write>
    init_scratch_from_buffer(std::size_t size_of_buffer,
            std::shared_ptr<::sycl::buffer<uint8_t, 1>> buffer,
            ::sycl::handler &cgh) {
        auto scratch = xpu::sycl::interop_memory_arg_t<
                ::sycl::access::mode::read_write>();
        if (size_of_buffer > 0) {
            scratch = xpu::sycl::interop_memory_arg_t<
                    ::sycl::access::mode::read_write>(*buffer, cgh);
        }
        return scratch;
    }
};

struct cudnn_matmul_exec_t final : cudnn_matmul_base_exec_t {

    status_t execute(const exec_ctx_t &ctx, impl::engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            std::size_t bias_scratch_size) override {

        nvidia::stream_t *cuda_stream
                = utils::downcast<nvidia::stream_t *>(ctx.stream());

        return cuda_stream->interop_task([=, this](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);

            auto arg_bias_scratch = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_dst_in_acc_dt);

            auto arg_src_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto arg_wei_scale = CTX_IN_SYCL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

            interop_task(matmul_impl_, engine, cgh, cuda_stream, arg_wt,
                    arg_src, arg_dst, arg_bias, arg_bias_scratch, arg_src_scale,
                    arg_wei_scale, arg_dst_scale);
        });
    }

    ~cudnn_matmul_exec_t() = default;
};

struct cudnn_matmul_runtime_args_exec_t final
    : public cudnn_matmul_base_exec_t {
    status_t execute(const exec_ctx_t &ctx, impl::engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            std::size_t bias_scratch_size) override {

        nvidia::stream_t *cuda_stream
                = utils::downcast<nvidia::stream_t *>(ctx.stream());

        init_scratch_buffers(bias_scratch_size);

        return cuda_stream->interop_task([=, this](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);
            auto arg_bias_scratch = init_scratch_from_buffer(
                    bias_scratch_size, bias_scratch_buff_, cgh);

            auto arg_src_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto arg_wei_scale = CTX_IN_SYCL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

            interop_task(matmul_impl_, engine, cgh, cuda_stream, arg_wt,
                    arg_src, arg_dst, arg_bias, arg_bias_scratch, arg_src_scale,
                    arg_wei_scale, arg_dst_scale);
        });
    }
    ~cudnn_matmul_runtime_args_exec_t() = default;

protected:
    void init_scratch_buffers(std::size_t reorder_scratch_size) {

        if (reorder_scratch_size > 0) {
            bias_scratch_buff_.reset(
                    new ::sycl::buffer<uint8_t, 1>(reorder_scratch_size));
            cudaDeviceSynchronize();
        }
    }

    std::shared_ptr<::sycl::buffer<uint8_t, 1>> bias_scratch_buff_ {nullptr};
};

struct cudnn_matmul_lt_base_exec_t {

    virtual status_t execute(const exec_ctx_t &ctx, impl::engine_t *engine,
            const std::shared_ptr<cudnn_matmul_lt_impl_t> matmul_impl_,
            std::size_t algo_scratch_size, std::size_t bias_scratch_size,
            std::size_t block_a_scratch_size, std::size_t block_b_scratch_size,
            std::size_t block_c_scratch_size,
            std::size_t src_scale_scratchpad_size,
            std::size_t wei_scale_scratchpad_size)
            = 0;

protected:
    template <::sycl::access::mode bias_m, ::sycl::access::mode scratch_m>
    void interop_task(std::shared_ptr<cudnn_matmul_lt_impl_t> matmul_impl_,
            impl::engine_t *engine, ::sycl::handler &cgh,
            nvidia::stream_t *cuda_stream,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read>
                    arg_weights,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read> arg_src,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::write>
                    arg_dst,
            xpu::sycl::interop_memory_arg_t<bias_m> arg_bias,
            xpu::sycl::interop_memory_arg_t<scratch_m> arg_algo_scratch,
            xpu::sycl::interop_memory_arg_t<scratch_m> arg_bias_scratch,
            xpu::sycl::interop_memory_arg_t<scratch_m> arg_block_a_scratch,
            xpu::sycl::interop_memory_arg_t<scratch_m> arg_block_b_scratch,
            xpu::sycl::interop_memory_arg_t<scratch_m> arg_block_c_scratch,
            xpu::sycl::interop_memory_arg_t<scratch_m> scaled_arg_src,
            xpu::sycl::interop_memory_arg_t<scratch_m> scaled_arg_wt,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read>
                    arg_src_scale,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read>
                    arg_wei_scale,
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read>
                    arg_dst_scale) {

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<nvidia::engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            // SYCL out-of-order queue encapsulates multiple CUstream objects.
            // Every query of the CUstream object can return any of those
            // therefore we need to make sure that we activate both cuDNN and
            // cuBLAS handles for the same CUstream object.
            auto native_stream = cuda_stream->get_underlying_stream();
            auto cublas_handle = cuda_stream->get_cublas_handle(native_stream);
            auto cudnn_handle = cuda_stream->get_cudnn_handle(native_stream);

            void *reorder_scratch = arg_bias_scratch.get_native_pointer(ih);
            void *algo_scratch = arg_algo_scratch.get_native_pointer(ih);
            void *block_a_scratch = arg_block_a_scratch.get_native_pointer(ih);
            void *block_b_scratch = arg_block_b_scratch.get_native_pointer(ih);
            void *block_c_scratch = arg_block_c_scratch.get_native_pointer(ih);

            void *scaled_src = scaled_arg_src.get_native_pointer(ih);
            void *scaled_wt = scaled_arg_wt.get_native_pointer(ih);

            void *bias = arg_bias.get_native_pointer(ih);
            void *weights = arg_weights.get_native_pointer(ih);
            void *src = arg_src.get_native_pointer(ih);
            void *dst = arg_dst.get_native_pointer(ih);

            void *src_scale = arg_src_scale.get_native_pointer(ih);
            void *wei_scale = arg_wei_scale.get_native_pointer(ih);
            void *dst_scale = arg_dst_scale.get_native_pointer(ih);

            matmul_impl_->execute(cublas_handle, cudnn_handle, weights, src,
                    dst, bias, algo_scratch, reorder_scratch, block_a_scratch,
                    block_b_scratch, block_c_scratch, scaled_src, scaled_wt,
                    src_scale, wei_scale, dst_scale);
        });
    }

protected:
    xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read_write>
    init_scratch_from_buffer(std::size_t size_of_buffer,
            std::shared_ptr<::sycl::buffer<uint8_t, 1>> buffer,
            ::sycl::handler &cgh) {
        auto scratch = xpu::sycl::interop_memory_arg_t<
                ::sycl::access::mode::read_write>();
        if (size_of_buffer > 0) {
            scratch = xpu::sycl::interop_memory_arg_t<
                    ::sycl::access::mode::read_write>(*buffer, cgh);
        }
        return scratch;
    }
};

struct cudnn_matmul_lt_exec_t final : public cudnn_matmul_lt_base_exec_t {

    status_t execute(const exec_ctx_t &ctx, impl::engine_t *engine,
            const std::shared_ptr<cudnn_matmul_lt_impl_t> matmul_impl_,
            std::size_t algo_scratch_size, std::size_t bias_scratch_size,
            std::size_t block_a_scratch_size, std::size_t block_b_scratch_size,
            std::size_t block_c_scratch_size,
            std::size_t src_scale_scratchpad_size,
            std::size_t wei_scale_scratchpad_size) override {

        nvidia::stream_t *cuda_stream
                = utils::downcast<nvidia::stream_t *>(ctx.stream());

        return cuda_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

            auto arg_src_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);

            auto arg_wei_scale = CTX_IN_SYCL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
            auto arg_algo_scratch = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_lt_algo_scratch);
            auto arg_bias_scratch = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_dst_in_acc_dt);
            auto arg_block_a_scratch = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_lt_block_a);
            auto arg_block_b_scratch = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_lt_block_b);
            auto arg_block_c_scratch = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_lt_block_c);
            auto scaled_arg_src = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_lt_src_scale);
            auto scaled_arg_wt = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_lt_wei_scale);

            interop_task(matmul_impl_, engine, cgh, cuda_stream, arg_wt,
                    arg_src, arg_dst, arg_bias, arg_algo_scratch,
                    arg_bias_scratch, arg_block_a_scratch, arg_block_b_scratch,
                    arg_block_c_scratch, scaled_arg_src, scaled_arg_wt,
                    arg_src_scale, arg_wei_scale, arg_dst_scale);
        });
    }

    ~cudnn_matmul_lt_exec_t() = default;
};

struct cudnn_matmul_lt_runtime_args_exec_t final
    : public cudnn_matmul_lt_base_exec_t {
    status_t execute(const exec_ctx_t &ctx, impl::engine_t *engine,
            const std::shared_ptr<cudnn_matmul_lt_impl_t> matmul_impl_,
            std::size_t algo_scratch_size, std::size_t bias_scratch_size,
            std::size_t block_a_scratch_size, std::size_t block_b_scratch_size,
            std::size_t block_c_scratch_size,
            std::size_t src_scale_scratchpad_size,
            std::size_t wei_scale_scratchpad_size) {

        nvidia::stream_t *cuda_stream
                = utils::downcast<nvidia::stream_t *>(ctx.stream());

        init_scratch_buffers(bias_scratch_size, algo_scratch_size,
                block_a_scratch_size, block_b_scratch_size,
                block_c_scratch_size, src_scale_scratchpad_size,
                wei_scale_scratchpad_size);

        return cuda_stream->interop_task([=](::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);

            auto arg_algo_scratch = init_scratch_from_buffer(
                    algo_scratch_size, algo_scratch_buff_, cgh);

            auto arg_bias_scratch = init_scratch_from_buffer(
                    bias_scratch_size, bias_scratch_buff_, cgh);

            auto arg_block_a_scratch = init_scratch_from_buffer(
                    block_a_scratch_size, block_a_scratch_buff_, cgh);
            auto arg_block_b_scratch = init_scratch_from_buffer(
                    block_b_scratch_size, block_b_scratch_buff_, cgh);
            auto arg_block_c_scratch = init_scratch_from_buffer(
                    block_c_scratch_size, block_c_scratch_buff_, cgh);

            auto scaled_arg_src = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_lt_src_scale);
            auto scaled_arg_wt = CTX_SCRATCH_SYCL_MEMORY(
                    memory_tracking::names::key_matmul_lt_wei_scale);

            auto arg_src_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto arg_wei_scale = CTX_IN_SYCL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

            interop_task(matmul_impl_, engine, cgh, cuda_stream, arg_wt,
                    arg_src, arg_dst, arg_bias, arg_algo_scratch,
                    arg_bias_scratch, arg_block_a_scratch, arg_block_b_scratch,
                    arg_block_c_scratch, scaled_arg_src, scaled_arg_wt,
                    arg_src_scale, arg_wei_scale, arg_dst_scale);
        });
    }

    ~cudnn_matmul_lt_runtime_args_exec_t() = default;

protected:
    void init_scratch_buffers(std::size_t reorder_scratch_size,
            std::size_t algo_scratch_size, std::size_t block_a_scratch_size,
            std::size_t block_b_scratch_size, std::size_t block_c_scratch_size,
            std::size_t src_scale_scratchpad_size,
            std::size_t wei_scale_scratchpad_size) {
        if (reorder_scratch_size > 0) {
            bias_scratch_buff_.reset(
                    new ::sycl::buffer<uint8_t, 1>(reorder_scratch_size));
        }
        if (algo_scratch_size > 0) {
            algo_scratch_buff_.reset(
                    new ::sycl::buffer<uint8_t, 1>(algo_scratch_size));
        }
        if (block_a_scratch_size > 0) {
            block_a_scratch_buff_.reset(
                    new ::sycl::buffer<uint8_t, 1>(block_a_scratch_size));
        }
        if (block_b_scratch_size > 0) {
            block_b_scratch_buff_.reset(
                    new ::sycl::buffer<uint8_t, 1>(block_b_scratch_size));
        }
        if (block_c_scratch_size > 0) {
            block_c_scratch_buff_.reset(
                    new ::sycl::buffer<uint8_t, 1>(block_c_scratch_size));
        }
        if (src_scale_scratchpad_size > 0) {
            src_scale_scratch_buff_.reset(
                    new ::sycl::buffer<uint8_t, 1>(src_scale_scratchpad_size));
        }
        if (wei_scale_scratchpad_size > 0) {
            wei_scale_scratch_buff_.reset(
                    new ::sycl::buffer<uint8_t, 1>(wei_scale_scratchpad_size));
        }

        if (reorder_scratch_size > 0 || algo_scratch_size > 0
                || block_a_scratch_size > 0 || block_b_scratch_size > 0
                || block_c_scratch_size > 0 || src_scale_scratchpad_size > 0
                || wei_scale_scratchpad_size > 0) {
            cudaDeviceSynchronize();
        }
    }

    std::shared_ptr<::sycl::buffer<uint8_t, 1>> bias_scratch_buff_ {nullptr};
    std::shared_ptr<::sycl::buffer<uint8_t, 1>> algo_scratch_buff_ {nullptr};
    std::shared_ptr<::sycl::buffer<uint8_t, 1>> block_a_scratch_buff_ {nullptr};
    std::shared_ptr<::sycl::buffer<uint8_t, 1>> block_b_scratch_buff_ {nullptr};
    std::shared_ptr<::sycl::buffer<uint8_t, 1>> block_c_scratch_buff_ {nullptr};
    std::shared_ptr<::sycl::buffer<uint8_t, 1>> src_scale_scratch_buff_ {
            nullptr};
    std::shared_ptr<::sycl::buffer<uint8_t, 1>> wei_scale_scratch_buff_ {
            nullptr};
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
