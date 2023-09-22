/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

    static status_t create_stream(
            stream_t **stream, engine_t *engine, unsigned flags) {
        std::unique_ptr<sycl_hip_stream_t> sycl_stream(
                new sycl_hip_stream_t(engine, flags));
        if (!sycl_stream) return status::out_of_memory;

        CHECK(sycl_stream->init());
        *stream = sycl_stream.release();
        return status::success;
    }

    static status_t create_stream(
            stream_t **stream, engine_t *engine, ::sycl::queue &queue) {
        unsigned flags;
        CHECK(base_t::init_flags(&flags, queue));

        std::unique_ptr<sycl_hip_stream_t> sycl_stream(
                new sycl_hip_stream_t(engine, flags, queue));

        CHECK(sycl_stream->init());

        *stream = sycl_stream.release();
        return status::success;
    }

    status_t interop_task(std::function<void(::sycl::handler &)>);
    hipStream_t get_underlying_stream();
    hipCtx_t get_underlying_context();
    hipDevice_t get_underlying_device();

private:
    status_t init();
    sycl_hip_stream_t(engine_t *engine, unsigned flags, ::sycl::queue &queue)
        : base_t(engine, flags, queue) {}
    sycl_hip_stream_t(engine_t *engine, unsigned flags)
        : base_t(engine, flags) {}
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
