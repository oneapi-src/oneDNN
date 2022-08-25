/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include <CL/sycl.hpp>
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

#if DNNL_USE_SYCL121_API
#include <CL/sycl/backend/level_zero.hpp>
#else
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#endif

#include "common/utils.hpp"
#include "gpu/compute/device_info.hpp"
#include "sycl/level_zero_utils.hpp"
#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_engine_base.hpp"

#if DNNL_USE_SYCL121_API
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace dnnl {
namespace impl {
namespace sycl {

status_t func_zeKernelCreate(
        ze_module_handle_t, const ze_kernel_desc_t *, ze_kernel_handle_t *);

namespace compat {

using namespace gpu::compute;

namespace {
#if !DNNL_USE_SYCL121_API
status_t get_kernel_from_bundle(std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const ::sycl::kernel_bundle<::sycl::bundle_state::executable>
                &kernel_bundle,
        const std::string &kernel_name, const sycl_engine_base_t *sycl_engine) {

    auto backend = get_sycl_backend(sycl_engine->device());

    if (backend == backend_t::opencl) {
        auto ocl_programs
                = ::sycl::get_native<::sycl::backend::opencl>(kernel_bundle);
        // This function expects kernel bundle that was created from a single
        // native object.
        assert(ocl_programs.size() == 1);
        if (ocl_programs.size() != 1) return status::runtime_error;

        cl_int err;
        cl_kernel ocl_kernel
                = clCreateKernel(ocl_programs[0], kernel_name.c_str(), &err);
        OCL_CHECK(err);
        sycl_kernel = utils::make_unique<::sycl::kernel>(
                ::sycl::make_kernel<::sycl::backend::opencl>(
                        ocl_kernel, sycl_engine->context()));
    } else if (backend == backend_t::level0) {
        auto ze_modules
                = ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(
                        kernel_bundle);
        // This function expects kernel bundle that was created from a single native
        // object.
        assert(ze_modules.size() == 1);
        if (ze_modules.size() != 1) return status::runtime_error;

        ze_kernel_handle_t ze_kernel;
        ze_kernel_desc_t ze_kernel_desc {
                ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, kernel_name.c_str()};

        CHECK(func_zeKernelCreate(ze_modules[0], &ze_kernel_desc, &ze_kernel));

        auto k = ::sycl::make_kernel<::sycl::backend::ext_oneapi_level_zero>(
                {kernel_bundle, ze_kernel}, sycl_engine->context());
        sycl_kernel = utils::make_unique<::sycl::kernel>(k);
    } else {
        assert(!"unexpected");
        return status::invalid_arguments;
    }
    return status::success;
}
#endif

template <typename T>
status_t cache_program(
        const binary_t *binary, const T &program, program_list_t *programs) {
    if (!programs) return status::success;

#if DNNL_USE_SYCL121_API
    static_assert(std::is_same<T, ::sycl::program>::value,
            "This function expects sycl::program when SYCL 2017 API is used");
    programs->add(binary, new T(program));
#else
    static_assert(
            std::is_same<T,
                    ::sycl::kernel_bundle<::sycl::bundle_state::executable>>::
                    value,
            "This function expects sycl::kernel_bundle when SYCL 2020 API is "
            "used");
    programs->add(binary, new T(program));
#endif
    return status::success;
}

template <typename sycl_object_t>
void *get_native_impl(backend_t backend, const sycl_object_t &sycl_object) {
    if (backend == backend_t::opencl) {
#if DNNL_USE_SYCL121_API
        return sycl_object.template get_native<::sycl::backend::opencl>();
#else
        return ::sycl::get_native<::sycl::backend::opencl>(sycl_object);
#endif
    } else if (backend == backend_t::level0) {
#if DNNL_USE_SYCL121_API
        return sycl_object.template get_native<::sycl::backend::level_zero>();
#else
        return ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(
                sycl_object);
#endif
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
        const std::string &kernel_name, const sycl_engine_base_t *sycl_engine,
        void *native_program_handle, const binary_t *binary,
        program_list_t *programs) {
    auto backend = get_sycl_backend(sycl_engine->device());

    if (backend == backend_t::opencl) {
        gpu::ocl::ocl_wrapper_t<cl_program> ocl_program(
                reinterpret_cast<cl_program>(native_program_handle));
#if DNNL_USE_SYCL121_API
        auto sycl_program = ::sycl::opencl::make<::sycl::program>(
                sycl_engine->context(), ocl_program.release());

        sycl_kernel = utils::make_unique<::sycl::kernel>(
                sycl_program.get_kernel(kernel_name));
        CHECK(cache_program(binary, sycl_program, programs));
#else
        cl_int err;
        cl_kernel ocl_kernel
                = clCreateKernel(ocl_program, kernel_name.c_str(), &err);
        OCL_CHECK(err);
        sycl_kernel = utils::make_unique<::sycl::kernel>(
                ::sycl::make_kernel<::sycl::backend::opencl>(
                        ocl_kernel, sycl_engine->context()));

        {
            // Create a kernel bundle for caching.
            ::sycl::kernel_bundle<::sycl::bundle_state::executable>
                    kernel_bundle
                    = ::sycl::make_kernel_bundle<::sycl::backend::opencl,
                            ::sycl::bundle_state::executable>(
                            {ocl_program}, sycl_engine->context());
            CHECK(cache_program(binary, kernel_bundle, programs));
        }

#endif
    } else if (backend == backend_t::level0) {
        ze_module_handle_t ze_module
                = reinterpret_cast<ze_module_handle_t>(native_program_handle);
#if DNNL_USE_SYCL121_API
        auto sycl_program = ::sycl::level_zero::make<::sycl::program>(
                sycl_engine->context(), ze_module);
        sycl_kernel = utils::make_unique<::sycl::kernel>(
                sycl_program.get_kernel(kernel_name));
        CHECK(cache_program(binary, sycl_program, programs));
#else
        ::sycl::kernel_bundle<::sycl::bundle_state::executable> kernel_bundle
                = ::sycl::make_kernel_bundle<
                        ::sycl::backend::ext_oneapi_level_zero,
                        ::sycl::bundle_state::executable>(
                        {ze_module}, sycl_engine->context());
        CHECK(cache_program(binary, kernel_bundle, programs));

        ze_kernel_handle_t ze_kernel;
        ze_kernel_desc_t ze_kernel_desc {
                ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, kernel_name.c_str()};

        CHECK(func_zeKernelCreate(ze_module, &ze_kernel_desc, &ze_kernel));

        auto k = ::sycl::make_kernel<::sycl::backend::ext_oneapi_level_zero>(
                {kernel_bundle, ze_kernel}, sycl_engine->context());
        sycl_kernel = utils::make_unique<::sycl::kernel>(k);
#endif
    } else {
        assert(!"unexpected");
        return status::invalid_arguments;
    }
    return status::success;
}

status_t make_kernel(std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const std::string &kernel_name, const sycl_engine_base_t *sycl_engine,
        const binary_t *binary, const program_list_t *programs) {
    if (!programs) return status::success;

#if DNNL_USE_SYCL121_API
    auto *p = programs->get<::sycl::program *>(binary);
    if (p) {
        sycl_kernel = utils::make_unique<::sycl::kernel>(
                p->get_kernel(kernel_name));
        if (!sycl_kernel) return status::out_of_memory;
    }
#else
    auto *kb = programs->get<
            ::sycl::kernel_bundle<::sycl::bundle_state::executable> *>(binary);
    if (kb) {
        CHECK(get_kernel_from_bundle(
                sycl_kernel, *kb, kernel_name, sycl_engine));
    }
#endif
    return status::success;
}

std::function<void(void *)> get_program_list_deleter() {
#if DNNL_USE_SYCL121_API
    return [](void *p) { delete reinterpret_cast<::sycl::program *>(p); };
#else
    return [](void *p) {
        delete reinterpret_cast<
                ::sycl::kernel_bundle<::sycl::bundle_state::executable> *>(p);
    };
#endif
}

uint64_t init_extensions(const ::sycl::device &dev) {
    uint64_t extensions = 0;
#if DNNL_USE_SYCL121_API
    for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
            i_ext <<= 1) {
        const char *s_ext = ext2cl_str((device_ext_t)i_ext);
        if (s_ext && dev.has_extension(s_ext)) { extensions |= i_ext; }
    }
#else

// The compiler marks `int64_base_atomics` and `int64_extended_atomics`
// as deprecated but hasn't implemented the `aspect::atomic64` replacement yet.
// TODO: Replace the deprecated aspects once the replacement is implemented.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
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
                is_ext_supported = dev.has(::sycl::aspect::int64_base_atomics);
                break;
            case device_ext_t::khr_global_int32_extended_atomics:
            case device_ext_t::khr_local_int32_extended_atomics:
            case device_ext_t::khr_int64_extended_atomics:
                is_ext_supported
                        = dev.has(::sycl::aspect::int64_extended_atomics);
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
#pragma clang diagnostic pop

#endif
    return extensions;
}

#if DNNL_USE_SYCL121_API
#pragma clang diagnostic pop
#endif

} // namespace compat
} // namespace sycl
} // namespace impl
} // namespace dnnl
