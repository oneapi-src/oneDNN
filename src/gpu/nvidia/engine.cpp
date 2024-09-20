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

#include "xpu/sycl/utils.hpp"

#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/stream.hpp"
#include "gpu/nvidia/sycl_cuda_compat.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index) {
    CHECK(nvidia::check_device(engine_kind));
    std::unique_ptr<nvidia::engine_t, engine_deleter_t> e(
            (new nvidia::engine_t(dev, ctx, index)));
    if (!e) return status::out_of_memory;

    CHECK(e->init());
    *engine = e.release();

    return status::success;
}

engine_t::engine_t(
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : impl::gpu::engine_t(
            new xpu::sycl::engine_impl_t(engine_kind::gpu, dev, ctx, index)) {
    assert(xpu::sycl::is_nvidia_gpu(dev));
    set_cudnn_handle();
    set_cublas_handle();
}

status_t engine_t::set_cublas_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the cublas handle.
    cuda_sycl_scoped_context_handler_t sc(*this);
    cublasHandle_t handle;
    CHECK(CUBLAS_EXECUTE_FUNC_S(cublasCreate, &handle));
    cublas_handle_.set(
            std::unique_ptr<cublasHandle_t, void (*)(cublasHandle_t *)>(
                    new cublasHandle_t(handle), [](cublasHandle_t *h) {
                        if (h != nullptr)
                            CUBLAS_EXECUTE_FUNC_V(cublasDestroy, *h);
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}

status_t engine_t::set_cudnn_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the cudnn handle.
    cuda_sycl_scoped_context_handler_t sc(*this);
    cudnnHandle_t handle;
    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreate, &handle));
    cudnn_handle_.set(std::unique_ptr<cudnnHandle_t, void (*)(cudnnHandle_t *)>(
            new cudnnHandle_t(handle), [](cudnnHandle_t *h) {
                if (h != nullptr) CUDNN_EXECUTE_FUNC_V(cudnnDestroy, *h);
                delete h;
            }));
    handle = nullptr;
    return status::success;
}

CUcontext engine_t::get_underlying_context() const {
    return compat::get_native<CUcontext>(device());
}

CUdevice engine_t::get_underlying_device() const {
    return compat::get_native<CUdevice>(device());
}

status_t engine_t::create_stream(
        impl::stream_t **stream, impl::stream_impl_t *stream_impl) {
    return nvidia::stream_t::create_stream(stream, this, stream_impl);
}

cudnnHandle_t *engine_t::get_cudnn_handle() {
    if (!cudnn_handle_.is_set()) set_cudnn_handle();
    return cudnn_handle_.get().get();
}

cublasHandle_t *engine_t::get_cublas_handle() {
    if (!cublas_handle_.is_set()) set_cublas_handle();
    return cublas_handle_.get().get();
}

void engine_t::activate_stream_cublas(CUstream cuda_stream) {
    cuda_sycl_scoped_context_handler_t sc(*this);
    cudaStream_t current_stream_id = nullptr;
    auto cublas_handle = get_cublas_handle();
    CUBLAS_EXECUTE_FUNC(cublasGetStream, *cublas_handle, &current_stream_id);
    if (current_stream_id != cuda_stream) {
        CUBLAS_EXECUTE_FUNC(cublasSetStream, *cublas_handle, cuda_stream);
    }
}

void engine_t::activate_stream_cudnn(CUstream cuda_stream) {
    cuda_sycl_scoped_context_handler_t sc(*this);
    cudaStream_t current_stream_id = nullptr;
    auto cudnn_handle = get_cudnn_handle();
    CUDNN_EXECUTE_FUNC(cudnnGetStream, *cudnn_handle, &current_stream_id);
    if (current_stream_id != cuda_stream) {
        CUDNN_EXECUTE_FUNC(cudnnSetStream, *cudnn_handle, cuda_stream);
    }
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
