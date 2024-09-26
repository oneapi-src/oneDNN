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

#ifndef GPU_NVIDIA_ENGINE_HPP
#define GPU_NVIDIA_ENGINE_HPP

#include <cudnn.h>
#include <cublas_v2.h>

#include "common/stream.hpp"
#include "common/thread_local_storage.hpp"
#include "gpu/gpu_engine.hpp"
#include "gpu/gpu_impl_list.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"
#include "xpu/sycl/engine_impl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index);

class engine_t : public gpu::engine_t {
public:
    engine_t(const ::sycl::device &dev, const ::sycl::context &ctx,
            size_t index);

    status_t init() { return init_impl(); }

    status_t create_stream(
            impl::stream_t **stream, impl::stream_impl_t *stream_impl) override;

    void activate_stream_cudnn(CUstream cuda_stream);
    void activate_stream_cublas(CUstream cuda_stream);

    CUcontext get_underlying_context() const;
    CUdevice get_underlying_device() const;
    cudnnHandle_t *get_cudnn_handle();
    cublasHandle_t *get_cublas_handle();

    bool mayiuse_system_memory_allocators() const override {
        return impl()->mayiuse_system_memory_allocators();
    }

    DECLARE_COMMON_SYCL_ENGINE_FUNCTIONS();

protected:
    const xpu::sycl::engine_impl_t *impl() const {
        return (const xpu::sycl::engine_impl_t *)impl::engine_t::impl();
    }

private:
    status_t set_cudnn_handle();
    status_t set_cublas_handle();
    // To avoid performance penalty cudnn/cublas required to have one handle per
    // thread per context therefor the handles will be the properties of the
    // engine. an engine can be assigned to multiple streams: lets say engine
    // eng(kind, 0); stream str1(eng,...); stream str2(eng,...); stream
    // str3(eng,...); In multi-threading environment both engin and stream
    // should be created in a different thread in order to allow safe
    // multi-threading programming If all the streams belongs to one thread, the
    // same handle will be used for all. Creation of handle is expensive and
    // must be avoided when it is not necessary.
    utils::thread_local_storage_t<
            std::unique_ptr<cudnnHandle_t, void (*)(cudnnHandle_t *)>>
            cudnn_handle_;
    utils::thread_local_storage_t<
            std::unique_ptr<cublasHandle_t, void (*)(cublasHandle_t *)>>
            cublas_handle_;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
