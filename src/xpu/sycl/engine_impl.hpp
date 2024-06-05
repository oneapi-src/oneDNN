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

#include "xpu/ocl/utils.hpp"
#include "xpu/sycl/compat.hpp"
#include "xpu/sycl/engine_id.hpp"
#include "xpu/sycl/stream_impl.hpp"
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

    status_t create_memory_storage(memory_storage_t **storage, engine_t *engine,
            unsigned flags, size_t size, void *handle) const;

    const ::sycl::device &device() const { return device_; }
    const ::sycl::context &context() const { return context_; }

    backend_t backend() const { return backend_; }

    engine_id_t engine_id() const override {
        return engine_id_t(new xpu::sycl::engine_id_impl_t(
                device(), context(), kind(), runtime_kind(), index()));
    }

    status_t create_stream_impl(
            impl::stream_impl_t **stream_impl, unsigned flags) const override {
        auto *si = new xpu::sycl::stream_impl_t(flags);
        if (!si) return status::out_of_memory;
        *stream_impl = si;
        return status::success;
    }

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
