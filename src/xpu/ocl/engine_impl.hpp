/*******************************************************************************
* Copyright 2024 Intel Corporation
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

        return status::success;
    }

    cl_device_id device() const { return device_; }
    cl_context context() const { return context_; }
    cl_platform_id platform() const { return platform_; }

    device_id_t device_id() const {
        return std::make_tuple(0, reinterpret_cast<uint64_t>(device()), 0);
    }

private:
    xpu::ocl::wrapper_t<cl_device_id> device_;
    xpu::ocl::wrapper_t<cl_context> context_;
    cl_platform_id platform_ = nullptr;
    bool is_user_context_;
};

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
