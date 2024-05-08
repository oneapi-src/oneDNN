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

// TODO: Move backend specific code to gpu/intel/sycl
#if __has_include(<sycl/backend/opencl.hpp>)
#include <sycl/backend/opencl.hpp>
#elif __has_include(<CL/sycl/backend/opencl.hpp>)
#include <CL/sycl/backend/opencl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "xpu/sycl/compat.hpp"
#include "xpu/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

namespace {
template <typename sycl_object_t>
void *get_native_impl(backend_t backend, const sycl_object_t &sycl_object) {
    if (backend == backend_t::opencl) {
        return ::sycl::get_native<::sycl::backend::opencl>(sycl_object);
    } else if (backend == backend_t::level0) {
        return ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(
                sycl_object);
    } else {
        assert(!"unexpected");
        return nullptr;
    }
    return nullptr;
}

} // namespace

namespace compat {

void *get_native(const ::sycl::device &dev) {
    auto backend = get_backend(dev);
    return get_native_impl(backend, dev);
}

void *get_native(const ::sycl::context &ctx) {
    auto devices = ctx.get_devices();
    assert(!devices.empty());
    if (devices.empty()) return nullptr;
    // backend is expected to be the same for all devices in a context.
    auto backend = get_backend(devices[0]);
    return get_native_impl(backend, ctx);
}

} // namespace compat
} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl
