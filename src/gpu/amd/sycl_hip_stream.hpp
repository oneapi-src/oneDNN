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

#ifndef GPU_AMD_SYCL_HIP_STREAM_HPP
#define GPU_AMD_SYCL_HIP_STREAM_HPP
#include "common/engine.hpp"
#include "sycl/sycl_stream.hpp"
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

class sycl_hip_stream_t : public dnnl::impl::sycl::sycl_stream_t {
public:
    using base_t = dnnl::impl::sycl::sycl_stream_t;

    miopenHandle_t &get_miopen_handle(HIPstream hip_stream = nullptr);
    rocblas_handle &get_rocblas_handle(HIPstream hip_stream = nullptr);

    static status_t create_stream(impl::stream_t **stream,
            impl::engine_t *engine, impl::stream_impl_t *stream_impl) {
        std::unique_ptr<sycl_hip_stream_t> s(
                new sycl_hip_stream_t(engine, stream_impl));
        if (!s) return status::out_of_memory;

        status_t status = s->init();
        if (status != status::success) {
            // Stream owns stream_impl only if it's created successfully (including initialization).
            s->impl_.release();
            return status;
        }

        *stream = s.release();
        return status::success;
    }

    status_t interop_task(std::function<void(::sycl::handler &)>);
    hipStream_t get_underlying_stream();
    hipCtx_t get_underlying_context();
    hipDevice_t get_underlying_device();

private:
    status_t init();
    sycl_hip_stream_t(impl::engine_t *engine, impl::stream_impl_t *stream_impl)
        : base_t(engine, stream_impl) {}
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
