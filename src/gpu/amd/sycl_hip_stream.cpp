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

#include "gpu/amd/sycl_hip_stream.hpp"
#include "gpu/amd/sycl_hip_compat.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

rocblas_handle &sycl_hip_stream_t::get_rocblas_handle(HIPstream hip_stream) {
    if (!hip_stream) hip_stream = get_underlying_stream();
    auto e = utils::downcast<sycl_hip_engine_t *>(engine());
    assert(e->context() == queue().get_context());
    e->activate_stream_rocblas(hip_stream);
    return *(e->get_rocblas_handle());
}
miopenHandle_t &sycl_hip_stream_t::get_miopen_handle(HIPstream hip_stream) {
    auto e = utils::downcast<sycl_hip_engine_t *>(engine());
    assert(e->context() == queue().get_context());
    if (!hip_stream) hip_stream = get_underlying_stream();
    e->activate_stream_miopen(hip_stream);
    return *(e->get_miopen_handle());
}
// the sycl_hip_stream_t will not own this. it is an observer pointer
HIPstream sycl_hip_stream_t::get_underlying_stream() {
    return compat::get_native<HIPstream>(*queue_);
}

// the sycl_hip_stream_t will not own this. it is an observer pointer
HIPcontext sycl_hip_stream_t::get_underlying_context() {
    return compat::get_native<HIPcontext>(queue_->get_context());
}

// the sycl_hip_stream_t will not own this. it is an observer pointer
HIPdevice sycl_hip_stream_t::get_underlying_device() {
    return compat::get_native<HIPdevice>(queue_->get_device());
}

status_t sycl_hip_stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    // If queue_ is not set then construct it
    auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine());
    auto status = status::success;

    if (!queue_) {
        auto &sycl_ctx = sycl_engine.context();
        auto &sycl_dev = sycl_engine.device();
        queue_.reset(new ::sycl::queue(sycl_ctx, sycl_dev));
    } else {
        // We need to check that the given queue is associated with
        // the device and context of the engine.
        auto sycl_dev = queue().get_device();
        bool args_ok
                = engine()->kind() == engine_kind::gpu && sycl_dev.is_gpu();
        if (!args_ok) return status::invalid_arguments;

        auto queue_context = get_underlying_context();
        auto queue_device = get_underlying_device();

        auto engine_context = sycl_engine.get_underlying_context();
        auto engine_device = sycl_engine.get_underlying_device();

        status = ((engine_device != queue_device)
                         || (engine_context != queue_context))
                ? status::invalid_arguments
                : status::success;
    }

    return status;
}

status_t sycl_hip_stream_t::interop_task(
        std::function<void(::sycl::handler &)> sycl_hip_interop_) {
    try {
        this->set_deps({queue().submit([&](::sycl::handler &cgh) {
            cgh.depends_on(get_deps());
            sycl_hip_interop_(cgh);
        })});
        return status::success;
    } catch (std::runtime_error &e) {
        error::wrap_c_api(status::runtime_error, e.what());
        return status::runtime_error;
    }
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
