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

#ifndef GPU_AMD_ENGINE_HPP
#define GPU_AMD_ENGINE_HPP

#include <stdexcept>

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>

#include "common/stream.hpp"
#include "common/thread_local_storage.hpp"

#include "xpu/sycl/engine_impl.hpp"

#include "gpu/gpu_engine.hpp"
#include "gpu/gpu_impl_list.hpp"

#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index);

class engine_t : public gpu::engine_t {
public:
    engine_t(const ::sycl::device &dev, const ::sycl::context &ctx,
            size_t index);

    status_t init() { return init_impl(); }

    status_t create_stream(
            impl::stream_t **stream, impl::stream_impl_t *stream_impl) override;

    void activate_stream_miopen(HIPstream hip_stream);
    void activate_stream_rocblas(HIPstream hip_stream);

    hipCtx_t get_underlying_context() const;
    hipDevice_t get_underlying_device() const;
    miopenHandle_t *get_miopen_handle();
    rocblas_handle *get_rocblas_handle();
    const bool has_primary_context() const { return primary_context_; }

    bool mayiuse_system_memory_allocators() const override {
        return impl()->mayiuse_system_memory_allocators();
    }

    DECLARE_COMMON_SYCL_ENGINE_FUNCTIONS();

protected:
    const xpu::sycl::engine_impl_t *impl() const {
        return (const xpu::sycl::engine_impl_t *)impl::engine_t::impl();
    }

    ~engine_t() override = default;

private:
    status_t set_miopen_handle();
    status_t set_rocblas_handle();
    utils::thread_local_storage_t<
            std::unique_ptr<miopenHandle_t, void (*)(miopenHandle_t *)>>
            miopen_handle_;
    utils::thread_local_storage_t<
            std::unique_ptr<rocblas_handle, void (*)(rocblas_handle *)>>
            rocblas_handle_;
    bool primary_context_;
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
