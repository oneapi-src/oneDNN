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

#include "common/verbose.hpp"

#include "gpu/nvidia/sycl_cuda_compat.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

cublasHandle_t &sycl_cuda_stream_t::get_cublas_handle(CUstream cuda_stream) {
    if (!cuda_stream) cuda_stream = get_underlying_stream();
    auto e = utils::downcast<sycl_cuda_engine_t *>(engine());
    assert(e->context() == queue().get_context());
    e->activate_stream_cublas(cuda_stream);
    return *(e->get_cublas_handle());
}

cudnnHandle_t &sycl_cuda_stream_t::get_cudnn_handle(CUstream cuda_stream) {
    if (!cuda_stream) cuda_stream = get_underlying_stream();
    auto e = utils::downcast<sycl_cuda_engine_t *>(engine());
    assert(e->context() == queue().get_context());
    e->activate_stream_cudnn(cuda_stream);
    return *(e->get_cudnn_handle());
}
// the sycl_cuda_stream_t will not own this. it is an observer pointer
CUstream sycl_cuda_stream_t::get_underlying_stream() {
    return compat::get_native<CUstream>(*queue_);
}

// the sycl_cuda_stream_t will not own this. it is an observer pointer
CUcontext sycl_cuda_stream_t::get_underlying_context() {
    return compat::get_native<CUcontext>(queue_->get_context());
}

// the sycl_cuda_stream_t will not own this. it is an observer pointer
CUdevice sycl_cuda_stream_t::get_underlying_device() {
    return compat::get_native<CUdevice>(queue_->get_device());
}

status_t sycl_cuda_stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    VCONDCHECK(primitive, create, check, stream,
            is_profiling_enabled() == false, status::unimplemented,
            VERBOSE_PROFILING_UNSUPPORTED);

    // If queue_ is not set then construct it
    auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine());
    auto status = status::success;

    if (!queue_) {
        auto &sycl_ctx = sycl_engine.context();
        auto &sycl_dev = sycl_engine.device();
        ::sycl::property_list prop_list;
        if (flags() & stream_flags::in_order)
            prop_list = {::sycl::property::queue::in_order {}};
        queue_.reset(new ::sycl::queue(sycl_ctx, sycl_dev, prop_list));
    } else {
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

status_t sycl_cuda_stream_t::interop_task(
        std::function<void(::sycl::handler &)> sycl_cuda_interop_) {
    try {
        auto event = queue().submit([&](::sycl::handler &cgh) {
            cgh.depends_on(sycl_ctx().get_sycl_deps().events);
            sycl_cuda_interop_(cgh);
        });
        this->sycl_ctx().get_sycl_deps().events = {event};
        return status::success;
    } catch (std::runtime_error &e) {
        error::wrap_c_api(status::runtime_error, e.what());
        return status::runtime_error;
    }
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
