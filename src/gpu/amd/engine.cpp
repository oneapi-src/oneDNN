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

#include "gpu/amd/engine.hpp"
#include "gpu/amd/stream.hpp"
#include "gpu/amd/sycl_hip_compat.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index) {
    CHECK(amd::check_device(engine_kind));
    std::unique_ptr<amd::engine_t, engine_deleter_t> e(
            (new amd::engine_t(dev, ctx, index)));
    if (!e) return status::out_of_memory;

    CHECK(e->init());
    *engine = e.release();

    return status::success;
}

engine_t::engine_t(
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : impl::gpu::engine_t(
            new xpu::sycl::engine_impl_t(engine_kind::gpu, dev, ctx, index)) {
    assert(xpu::sycl::is_amd_gpu(dev));
    set_miopen_handle();
    set_rocblas_handle();
}

status_t engine_t::set_rocblas_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the rocblas handle.
    hip_sycl_scoped_context_handler_t sc(*this);
    rocblas_handle handle;
    CHECK(ROCBLAS_EXECUTE_FUNC_S(rocblas_create_handle, &handle));
    rocblas_handle_.set(
            std::unique_ptr<rocblas_handle, void (*)(rocblas_handle *)>(
                    new rocblas_handle(handle), [](rocblas_handle *h) {
                        if (h != nullptr)
                            ROCBLAS_EXECUTE_FUNC_V(rocblas_destroy_handle, *h);
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}

status_t engine_t::set_miopen_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the miopen handle.
    hip_sycl_scoped_context_handler_t sc(*this);
    miopenHandle_t handle;
    CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreate, &handle));
    miopen_handle_.set(
            std::unique_ptr<miopenHandle_t, void (*)(miopenHandle_t *)>(
                    new miopenHandle_t(handle), [](miopenHandle_t *h) {
                        if (h != nullptr)
                            MIOPEN_EXECUTE_FUNC_V(miopenDestroy, *h);
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}
hipCtx_t engine_t::get_underlying_context() const {
    return compat::get_native<hipCtx_t>(device());
}

hipDevice_t engine_t::get_underlying_device() const {
    return compat::get_native<hipDevice_t>(device());
}

status_t engine_t::create_stream(
        impl::stream_t **stream, impl::stream_impl_t *stream_impl) {
    return amd::stream_t::create_stream(stream, this, stream_impl);
}

miopenHandle_t *engine_t::get_miopen_handle() {
    if (!miopen_handle_.is_set()) set_miopen_handle();
    return miopen_handle_.get().get();
}

rocblas_handle *engine_t::get_rocblas_handle() {
    if (!rocblas_handle_.is_set()) set_rocblas_handle();
    return rocblas_handle_.get().get();
}

void engine_t::activate_stream_rocblas(HIPstream hip_stream) {
    hip_sycl_scoped_context_handler_t sc(*this);
    hipStream_t current_stream_id = nullptr;
    auto rocblas_handle = get_rocblas_handle();
    ROCBLAS_EXECUTE_FUNC(
            rocblas_get_stream, *rocblas_handle, &current_stream_id);
    if (current_stream_id != hip_stream) {
        ROCBLAS_EXECUTE_FUNC(rocblas_set_stream, *rocblas_handle, hip_stream);
    }
}

void engine_t::activate_stream_miopen(HIPstream hip_stream) {
    hip_sycl_scoped_context_handler_t sc(*this);
    hipStream_t current_stream_id = nullptr;
    auto miopen_handle = get_miopen_handle();
    MIOPEN_EXECUTE_FUNC_S(miopenGetStream, *miopen_handle, &current_stream_id);
    if (current_stream_id != hip_stream) {
        MIOPEN_EXECUTE_FUNC_S(miopenSetStream, *miopen_handle, hip_stream);
    }
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
