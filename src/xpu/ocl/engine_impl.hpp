/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef XPU_OCL_ENGINE_IMPL_HPP
#define XPU_OCL_ENGINE_IMPL_HPP

#include <tuple>
#include <CL/cl.h>

#include "common/engine_impl.hpp"
#include "common/utils.hpp"

#include "xpu/ocl/engine_id.hpp"
#include "xpu/ocl/stream_impl.hpp"
#include "xpu/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

class engine_impl_t : public impl::engine_impl_t {
public:
    engine_impl_t() = delete;
    engine_impl_t(cl_device_id device, cl_context context, size_t index)
        : impl::engine_impl_t(engine_kind::gpu, runtime_kind::ocl, index)
        , device_(device)
        , context_(context)
        , is_user_context_(context) {}

    ~engine_impl_t() override = default;

    status_t init() override {
        cl_int err = CL_SUCCESS;
        err = clGetDeviceInfo(device(), CL_DEVICE_PLATFORM, sizeof(platform_),
                &platform_, nullptr);
        if (err != CL_SUCCESS) {
            device_ = nullptr;
            context_ = nullptr;
        }

        OCL_CHECK(err);

        err = clRetainDevice(device());
        if (err != CL_SUCCESS) {
            device_ = nullptr;
            context_ = nullptr;
        }

        OCL_CHECK(err);

        if (is_user_context_) {
            err = clRetainContext(context());
            if (err != CL_SUCCESS) context_ = nullptr;
        } else {
            context_ = clCreateContext(
                    nullptr, 1, &device_.unwrap(), nullptr, nullptr, &err);
        }

        OCL_CHECK(err);

        CHECK(check_device(engine_kind::gpu, device(), context()));

        // TODO: Remove it as soon as device info is generalized.
        size_t param_size = 0;
        err = clGetDeviceInfo(device_, CL_DEVICE_NAME, 0, nullptr, &param_size);
        OCL_CHECK(err);

        name_ = std::string(param_size, '\0');
        err = clGetDeviceInfo(
                device_, CL_DEVICE_NAME, param_size, &name_[0], &param_size);
        OCL_CHECK(err);

        err = clGetDeviceInfo(
                device_, CL_DRIVER_VERSION, 0, nullptr, &param_size);
        OCL_CHECK(err);

        std::string driver_version(param_size, '\0');
        err = clGetDeviceInfo(device_, CL_DRIVER_VERSION, param_size,
                &driver_version[0], nullptr);
        OCL_CHECK(err);

        if (runtime_version_.set_from_string(&driver_version[0])
                != status::success) {
            runtime_version_.major = 0;
            runtime_version_.minor = 0;
            runtime_version_.build = 0;
        }

        return status::success;
    }

    status_t create_memory_storage(memory_storage_t **storage, engine_t *engine,
            unsigned flags, size_t size, void *handle) const override;

    cl_device_id device() const { return device_; }
    cl_context context() const { return context_; }
    cl_platform_id platform() const { return platform_; }

    engine_id_t engine_id() const override {
        return engine_id_t(new xpu::ocl::engine_id_impl_t(
                device(), context(), kind(), runtime_kind(), index()));
    }

    status_t create_stream_impl(
            impl::stream_impl_t **stream_impl, unsigned flags) const override {
        auto *si = new xpu::ocl::stream_impl_t(flags);
        if (!si) return status::out_of_memory;
        *stream_impl = si;
        return status::success;
    }

    // TODO: The device info class should be generalized to support multiple
    // vendors. For now, put common device info parts in engine_impl_t
    // directly.
    const std::string &name() const { return name_; }
    const runtime_version_t &runtime_version() const {
        return runtime_version_;
    }

    int get_buffer_alignment() const override { return 128; }

private:
    std::string name_;
    runtime_version_t runtime_version_;

    xpu::ocl::wrapper_t<cl_device_id> device_;
    xpu::ocl::wrapper_t<cl_context> context_;
    cl_platform_id platform_ = nullptr;
    bool is_user_context_;
};

#define DECLARE_COMMON_OCL_ENGINE_FUNCTIONS() \
    cl_device_id device() const { return impl()->device(); } \
    cl_context context() const { return impl()->context(); } \
    cl_platform_id platform() const { return impl()->platform(); }

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
