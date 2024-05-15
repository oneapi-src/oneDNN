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

#ifndef XPU_SYCL_ENGINE_IMPL_HPP
#define XPU_SYCL_ENGINE_IMPL_HPP

#include "common/engine_impl.hpp"
#include "common/utils.hpp"

#include "xpu/sycl/compat.hpp"
#include "xpu/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

class engine_impl_t : public impl::engine_impl_t {
public:
    engine_impl_t() = delete;
    engine_impl_t(engine_kind_t kind, const ::sycl::device &device,
            const ::sycl::context &context, size_t index)
        : impl::engine_impl_t(kind, runtime_kind::sycl, index)
        , device_(device)
        , context_(context)
        , backend_(backend_t::unknown) {}

    ~engine_impl_t() override = default;

    status_t init() override {
        backend_ = xpu::sycl::get_backend(device_);
        VERROR_ENGINE_IMPL(
                utils::one_of(backend_, backend_t::host, backend_t::opencl,
                        backend_t::level0, backend_t::nvidia, backend_t::amd),
                status::invalid_arguments, VERBOSE_UNSUPPORTED_BACKEND, "sycl");

        CHECK(check_device(kind(), device_, context_));
        return status::success;
    }

    const ::sycl::device &device() const { return device_; }
    const ::sycl::context &context() const { return context_; }

    backend_t backend() const { return backend_; }

    cl_device_id ocl_device() const {
        if (backend() != backend_t::opencl) {
            assert(!"not expected");
            return nullptr;
        }
        assert(device_.is_cpu() || device_.is_gpu());
        return xpu::ocl::make_wrapper(
                compat::get_native<cl_device_id>(device()));
    }

    cl_context ocl_context() const {
        if (backend() != backend_t::opencl) {
            assert(!"not expected");
            return nullptr;
        }
        assert(device_.is_cpu() || device_.is_gpu());
        return xpu::ocl::make_wrapper(
                compat::get_native<cl_context>(context()));
    }

    device_id_t device_id() const { return xpu::sycl::device_id(device_); }

private:
    ::sycl::device device_;
    ::sycl::context context_;

    backend_t backend_;
};

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
