/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2020-2024 Codeplay Software Limited
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

#include "common/compiler_workarounds.hpp"
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
            const std::shared_ptr<cublas_params> params,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d)
            = 0;

protected:
    template <::sycl::access::mode bias_m, ::sycl::access::mode scratch_m>
    void interop_task(std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            const std::shared_ptr<cublas_params> params, impl::engine_t *engine,
            ::sycl::handler &cgh, nvidia::stream_t *cuda_stream,
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
                    arg_dst_scale,
            uint8_t *bias_scratch_ptr) {

        compat::host_task(cgh,
                [= WA_THIS_COPY_CAPTURE](const compat::interop_handle &ih) {
                    auto &sycl_engine = *utils::downcast<nvidia::engine_t *>(
                            cuda_stream->engine());
                    auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
                    // SYCL out-of-order queue encapsulates multiple CUstream objects.
                    // Every query of the CUstream object can return any of those
                    // therefore we need to make sure that we activate both cuDNN and
                    // cuBLAS handles for the same CUstream object.
                    auto native_stream = cuda_stream->get_underlying_stream();
                    auto cublas_handle
                            = cuda_stream->get_cublas_handle(native_stream);
                    auto cudnn_handle
                            = cuda_stream->get_cudnn_handle(native_stream);

                    void *reorder_scratch
                            = arg_bias_scratch.get_native_pointer(ih);
                    void *bias = arg_bias.get_native_pointer(ih);
                    void *weights = arg_weights.get_native_pointer(ih);
                    void *src = arg_src.get_native_pointer(ih);
                    void *dst = arg_dst.get_native_pointer(ih);

                    void *src_scale = arg_src_scale.get_native_pointer(ih);
                    void *wei_scale = arg_wei_scale.get_native_pointer(ih);
                    void *dst_scale = arg_dst_scale.get_native_pointer(ih);

                    matmul_impl_->execute(cublas_handle, cudnn_handle, params,
                            weights, src, dst, bias, reorder_scratch, src_scale,
                            wei_scale, dst_scale);

                    if (params->has_runtime_params_) {
                        sync_device();
                        free_runtime_scratch(
                                cublas_handle, cuda_stream, bias_scratch_ptr);
                        params->cleanup();
                    }
                });
    }

    void free_runtime_scratch(cublasHandle_t cublas_handle,
            nvidia::stream_t *cuda_stream, uint8_t *bias_scratch_ptr) {
        if (bias_scratch_ptr) {
            ::sycl::free(bias_scratch_ptr, cuda_stream->queue());
        }
    }

    xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read_write>
    init_scratch_from_ptr(std::size_t size_of_ptr, uint8_t *ptr) {
        auto scratch = xpu::sycl::interop_memory_arg_t<
                ::sycl::access::mode::read_write>();
        if (size_of_ptr > 0) {
            scratch = xpu::sycl::interop_memory_arg_t<
                    ::sycl::access::mode::read_write>(ptr);
        }
        return scratch;
    }
};

struct cudnn_matmul_exec_t final : cudnn_matmul_base_exec_t {

    status_t execute(const exec_ctx_t &ctx, impl::engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            const std::shared_ptr<cublas_params> params,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &bias_d) override {

        nvidia::stream_t *cuda_stream
                = utils::downcast<nvidia::stream_t *>(ctx.stream());

        return cuda_stream->interop_task([= WA_THIS_COPY_CAPTURE](
                                                 ::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);

            auto arg_bias_scratch = params->reorder_scratch_size_ != 0
                    ? CTX_SCRATCH_SYCL_MEMORY(
                            memory_tracking::names::key_matmul_dst_in_acc_dt)
                    : xpu::sycl::interop_memory_arg_t<
                            ::sycl::access::mode::read_write>();

            auto arg_src_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto arg_wei_scale = CTX_IN_SYCL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

            interop_task(matmul_impl_, params, engine, cgh, cuda_stream, arg_wt,
                    arg_src, arg_dst, arg_bias, arg_bias_scratch, arg_src_scale,
                    arg_wei_scale, arg_dst_scale, nullptr);
        });
    }

    ~cudnn_matmul_exec_t() = default;
};

struct cudnn_matmul_runtime_args_exec_t final
    : public cudnn_matmul_base_exec_t {
    status_t execute(const exec_ctx_t &ctx, impl::engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            const std::shared_ptr<cublas_params> params,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &bias_d) override {

        nvidia::stream_t *cuda_stream
                = utils::downcast<nvidia::stream_t *>(ctx.stream());

        std::shared_ptr<cublas_params> matmul_params
                = std::make_shared<cublas_params>();
        matmul_params->init_from_params(params);
        if (matmul_params->has_runtime_params_) {
            matmul_params->cleanup();
            matmul_params->set_params(src_d, weights_d, dst_d, bias_d);
        }

        uint8_t *bias_scratch_ptr = nullptr;
        if (matmul_params->reorder_scratch_size_ > 0) {
            bias_scratch_ptr = ::sycl::malloc_device<uint8_t>(
                    matmul_params->reorder_scratch_size_, cuda_stream->queue());
        }

        return cuda_stream->interop_task([= WA_THIS_COPY_CAPTURE](
                                                 ::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);
            auto arg_bias_scratch = init_scratch_from_ptr(
                    matmul_params->reorder_scratch_size_, bias_scratch_ptr);

            auto arg_src_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            auto arg_wei_scale = CTX_IN_SYCL_MEMORY(
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

            interop_task(matmul_impl_, matmul_params, engine, cgh, cuda_stream,
                    arg_wt, arg_src, arg_dst, arg_bias, arg_bias_scratch,
                    arg_src_scale, arg_wei_scale, arg_dst_scale,
                    bias_scratch_ptr);
        });
    }
    ~cudnn_matmul_runtime_args_exec_t() = default;

protected:
};

struct cudnn_matmul_lt_base_exec_t {

    virtual status_t execute(const exec_ctx_t &ctx, impl::engine_t *engine,
            const std::shared_ptr<cudnn_matmul_lt_impl_t> matmul_impl_,
            const std::shared_ptr<cublas_lt_params> params,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d)
            = 0;

protected:
    template <::sycl::access::mode bias_m, ::sycl::access::mode scratch_m>
    void interop_task(std::shared_ptr<cudnn_matmul_lt_impl_t> matmul_impl_,
            const std::shared_ptr<cublas_lt_params> params,
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
            xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read>
                    arg_dst_scale,
            uint8_t *algo_scratch_ptr, uint8_t *bias_scratch_ptr,
            uint8_t *block_a_scratch_ptr, uint8_t *block_b_scratch_ptr,
            uint8_t *block_c_scratch_ptr) {

        compat::host_task(cgh,
                [= WA_THIS_COPY_CAPTURE](const compat::interop_handle &ih) {
                    auto &sycl_engine = *utils::downcast<nvidia::engine_t *>(
                            cuda_stream->engine());
                    auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
                    // SYCL out-of-order queue encapsulates multiple CUstream objects.
                    // Every query of the CUstream object can return any of those
                    // therefore we need to make sure that we activate both cuDNN and
                    // cuBLAS handles for the same CUstream object.
                    auto native_stream = cuda_stream->get_underlying_stream();
                    auto cublas_handle
                            = cuda_stream->get_cublas_handle(native_stream);

                    void *reorder_scratch
                            = arg_bias_scratch.get_native_pointer(ih);
                    void *algo_scratch
                            = arg_algo_scratch.get_native_pointer(ih);
                    void *block_a_scratch
                            = arg_block_a_scratch.get_native_pointer(ih);
                    void *block_b_scratch
                            = arg_block_b_scratch.get_native_pointer(ih);
                    void *block_c_scratch
                            = arg_block_c_scratch.get_native_pointer(ih);

                    void *bias = arg_bias.get_native_pointer(ih);
                    void *weights = arg_weights.get_native_pointer(ih);
                    void *src = arg_src.get_native_pointer(ih);
                    void *dst = arg_dst.get_native_pointer(ih);

                    void *dst_scale = arg_dst_scale.get_native_pointer(ih);

                    matmul_impl_->execute(cublas_handle, params, weights, src,
                            dst, bias, algo_scratch, reorder_scratch,
                            block_a_scratch, block_b_scratch, block_c_scratch,
                            nullptr, nullptr, dst_scale);

                    free_runtime_scratch(params->has_runtime_params_,
                            cublas_handle, cuda_stream, algo_scratch_ptr,
                            bias_scratch_ptr, block_a_scratch_ptr,
                            block_b_scratch_ptr, block_c_scratch_ptr);
                    if (params->has_runtime_params_) { params->rt_cleanup(); }
                });
    }

protected:
    void free_runtime_scratch(bool has_runtime_params,
            cublasHandle_t cublas_handle, nvidia::stream_t *cuda_stream,
            uint8_t *algo_scratch_ptr, uint8_t *bias_scratch_ptr,
            uint8_t *block_a_scratch_ptr, uint8_t *block_b_scratch_ptr,
            uint8_t *block_c_scratch_ptr) {
        if (has_runtime_params || bias_scratch_ptr) {
            cudaStream_t streamId;
            cublasGetStream(cublas_handle, &streamId);
            cudaStreamSynchronize(streamId);
            if (algo_scratch_ptr) {
                ::sycl::free(algo_scratch_ptr, cuda_stream->queue());
            }
            if (bias_scratch_ptr) {
                ::sycl::free(bias_scratch_ptr, cuda_stream->queue());
            }
            if (block_a_scratch_ptr) {
                ::sycl::free(block_a_scratch_ptr, cuda_stream->queue());
            }
            if (block_b_scratch_ptr) {
                ::sycl::free(block_b_scratch_ptr, cuda_stream->queue());
            }
            if (block_c_scratch_ptr) {
                ::sycl::free(block_c_scratch_ptr, cuda_stream->queue());
            }
        }
    }

    xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read_write>
    init_scratch_from_ptr(std::size_t size_of_ptr, uint8_t *ptr) {
        auto scratch = xpu::sycl::interop_memory_arg_t<
                ::sycl::access::mode::read_write>();
        if (size_of_ptr > 0) {
            scratch = xpu::sycl::interop_memory_arg_t<
                    ::sycl::access::mode::read_write>(ptr);
        }
        return scratch;
    }
};

struct cudnn_matmul_lt_exec_t final : public cudnn_matmul_lt_base_exec_t {

    status_t execute(const exec_ctx_t &ctx, impl::engine_t *engine,
            const std::shared_ptr<cudnn_matmul_lt_impl_t> matmul_impl_,
            const std::shared_ptr<cublas_lt_params> params,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d) override {

        nvidia::stream_t *cuda_stream
                = utils::downcast<nvidia::stream_t *>(ctx.stream());

        return cuda_stream->interop_task([= WA_THIS_COPY_CAPTURE, &params](
                                                 ::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
            auto arg_algo_scratch = params->algo_scratch_size_ != 0
                    ? CTX_SCRATCH_SYCL_MEMORY(
                            memory_tracking::names::key_matmul_lt_algo_scratch)
                    : xpu::sycl::interop_memory_arg_t<
                            ::sycl::access::mode::read_write>();
            auto arg_bias_scratch = params->reorder_scratch_size_ != 0
                    ? CTX_SCRATCH_SYCL_MEMORY(
                            memory_tracking::names::key_matmul_dst_in_acc_dt)
                    : xpu::sycl::interop_memory_arg_t<
                            ::sycl::access::mode::read_write>();
            auto arg_block_a_scratch = params->weight_size_ != 0
                    ? CTX_SCRATCH_SYCL_MEMORY(
                            memory_tracking::names::key_gemm_blocked_a)
                    : xpu::sycl::interop_memory_arg_t<
                            ::sycl::access::mode::read_write>();
            auto arg_block_b_scratch = params->source_size_ != 0
                    ? CTX_SCRATCH_SYCL_MEMORY(
                            memory_tracking::names::key_gemm_blocked_b)
                    : xpu::sycl::interop_memory_arg_t<
                            ::sycl::access::mode::read_write>();
            auto arg_block_c_scratch = params->dest_size_ != 0
                    ? CTX_SCRATCH_SYCL_MEMORY(
                            memory_tracking::names::key_matmul_lt_block_c)
                    : xpu::sycl::interop_memory_arg_t<
                            ::sycl::access::mode::read_write>();

            interop_task(matmul_impl_, params, engine, cgh, cuda_stream, arg_wt,
                    arg_src, arg_dst, arg_bias, arg_algo_scratch,
                    arg_bias_scratch, arg_block_a_scratch, arg_block_b_scratch,
                    arg_block_c_scratch, arg_dst_scale, nullptr, nullptr,
                    nullptr, nullptr, nullptr);
        });
    }

    ~cudnn_matmul_lt_exec_t() = default;
};

struct cudnn_matmul_lt_runtime_args_exec_t final
    : public cudnn_matmul_lt_base_exec_t {
    status_t execute(const exec_ctx_t &ctx, impl::engine_t *engine,
            const std::shared_ptr<cudnn_matmul_lt_impl_t> matmul_impl_,
            const std::shared_ptr<cublas_lt_params> params,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d) {

        std::shared_ptr<cublas_lt_params> matmul_params
                = std::make_shared<cublas_lt_params>();
        matmul_params->init_from_params(params);
        if (matmul_params->has_runtime_params_) {
            matmul_params->rt_cleanup();
            matmul_params->set_params(src_d, weights_d, dst_d, engine);
        }

        nvidia::stream_t *cuda_stream
                = utils::downcast<nvidia::stream_t *>(ctx.stream());

        uint8_t *bias_scratch_ptr = alloc_ptr(
                matmul_params->algo_scratch_size_, cuda_stream->queue());

        uint8_t *algo_scratch_ptr = alloc_ptr(
                matmul_params->reorder_scratch_size_, cuda_stream->queue());

        uint8_t *block_a_scratch_ptr
                = alloc_ptr(matmul_params->weight_size_, cuda_stream->queue());

        uint8_t *block_b_scratch_ptr
                = alloc_ptr(matmul_params->source_size_, cuda_stream->queue());

        uint8_t *block_c_scratch_ptr
                = alloc_ptr(matmul_params->dest_size_, cuda_stream->queue());

        return cuda_stream->interop_task([= WA_THIS_COPY_CAPTURE](
                                                 ::sycl::handler &cgh) {
            auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
            auto arg_wt = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);
            auto arg_bias = CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS);

            auto arg_algo_scratch = init_scratch_from_ptr(
                    matmul_params->algo_scratch_size_, algo_scratch_ptr);
            auto arg_bias_scratch = init_scratch_from_ptr(
                    matmul_params->reorder_scratch_size_, bias_scratch_ptr);
            auto arg_block_a_scratch = init_scratch_from_ptr(
                    matmul_params->source_size_, block_a_scratch_ptr);
            auto arg_block_b_scratch = init_scratch_from_ptr(
                    matmul_params->weight_size_, block_b_scratch_ptr);
            auto arg_block_c_scratch = init_scratch_from_ptr(
                    matmul_params->dest_size_, block_c_scratch_ptr);

            auto arg_dst_scale
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

            interop_task(matmul_impl_, matmul_params, engine, cgh, cuda_stream,
                    arg_wt, arg_src, arg_dst, arg_bias, arg_algo_scratch,
                    arg_bias_scratch, arg_block_a_scratch, arg_block_b_scratch,
                    arg_block_c_scratch, arg_dst_scale, algo_scratch_ptr,
                    bias_scratch_ptr, block_a_scratch_ptr, block_b_scratch_ptr,
                    block_c_scratch_ptr);
        });
    }

    ~cudnn_matmul_lt_runtime_args_exec_t() = default;

protected:
    uint8_t *alloc_ptr(size_t size, ::sycl::queue q) {
        uint8_t *ptr = nullptr;
        if (size > 0) { ptr = ::sycl::malloc_device<uint8_t>(size, q); }
        return ptr;
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
