/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include <level_zero/ze_api.h>
#include <type_traits>

#include "oneapi/dnnl/dnnl_config.h"
#include "sycl/sycl_utils.hpp"

#if __has_include(<sycl/backend/opencl.hpp>)
#include <sycl/backend/opencl.hpp>
#elif __has_include(<CL/sycl/backend/opencl.hpp>)
#include <CL/sycl/backend/opencl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "common/utils.hpp"
#include "gpu/compute/device_info.hpp"
#include "sycl/level_zero_utils.hpp"
#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_engine_base.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

status_t func_zeKernelCreate(
        ze_module_handle_t, const ze_kernel_desc_t *, ze_kernel_handle_t *);

namespace compat {

using namespace gpu::compute;

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

void *get_native(const ::sycl::device &dev) {
    auto backend = get_sycl_backend(dev);
    return get_native_impl(backend, dev);
}

void *get_native(const ::sycl::context &ctx) {
    auto devices = ctx.get_devices();
    assert(!devices.empty());
    if (devices.empty()) return nullptr;
    // backend is expected to be the same for all devices in a context.
    auto backend = get_sycl_backend(devices[0]);
    return get_native_impl(backend, ctx);
}

status_t make_kernel(std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const sycl_engine_base_t *sycl_engine,
        const gpu::compute::binary_t &binary, const char *kernel_name) {
    auto backend = get_sycl_backend(sycl_engine->device());
    if (backend == backend_t::opencl) {
        gpu::ocl::ocl_wrapper_t<cl_program> ocl_program;
        CHECK(create_ocl_program(ocl_program, sycl_engine->ocl_device(),
                sycl_engine->ocl_context(), binary));
        cl_int err;
        cl_kernel ocl_kernel = clCreateKernel(ocl_program, kernel_name, &err);
        OCL_CHECK(err);
        sycl_kernel = utils::make_unique<::sycl::kernel>(
                ::sycl::make_kernel<::sycl::backend::opencl>(
                        ocl_kernel, sycl_engine->context()));
    } else if (backend == backend_t::level0) {
        CHECK(sycl_create_kernel_with_level_zero(
                sycl_kernel, kernel_name, sycl_engine, binary));
    } else {
        assert(!"unexpected");
        return status::invalid_arguments;
    }
    return status::success;
}

uint64_t init_extensions(const ::sycl::device &dev) {
    uint64_t extensions = 0;

#if DNNL_USE_SYCL121_API
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    constexpr auto base_atomics_aspect = ::sycl::aspect::int64_base_atomics;
    constexpr auto extended_atomics_aspect
            = ::sycl::aspect::int64_extended_atomics;
#else
    constexpr auto base_atomics_aspect = ::sycl::aspect::atomic64;
    constexpr auto extended_atomics_aspect = ::sycl::aspect::atomic64;
#endif

    for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
            i_ext <<= 1) {
        bool is_ext_supported = false;
        switch (static_cast<device_ext_t>(i_ext)) {
            case device_ext_t::khr_fp16:
                is_ext_supported = dev.has(::sycl::aspect::fp16);
                break;
            case device_ext_t::khr_fp64:
                is_ext_supported = dev.has(::sycl::aspect::fp64);
                break;
            case device_ext_t::khr_global_int32_base_atomics:
            case device_ext_t::khr_local_int32_base_atomics:
            case device_ext_t::khr_int64_base_atomics:
            case device_ext_t::ext_float_atomics:
                is_ext_supported = dev.has(base_atomics_aspect);
                break;
            case device_ext_t::khr_global_int32_extended_atomics:
            case device_ext_t::khr_local_int32_extended_atomics:
            case device_ext_t::khr_int64_extended_atomics:
                is_ext_supported = dev.has(extended_atomics_aspect);
                break;
            // SYCL 2020 assumes that subroups are always supported.
            case device_ext_t::intel_subgroups:
            case device_ext_t::intel_required_subgroup_size:
            case device_ext_t::intel_subgroups_char:
            case device_ext_t::intel_subgroups_short:
            case device_ext_t::intel_subgroups_long:
                is_ext_supported = true;
                break;
                // The workaround for future extensions should be used to cover
                // the remaining extensions.
            default: is_ext_supported = false;
        }
        if (is_ext_supported) extensions |= i_ext;
    }
#if DNNL_USE_SYCL121_API
#pragma clang diagnostic pop
#endif
    return extensions;
}

} // namespace compat
} // namespace sycl
} // namespace impl
} // namespace dnnl
