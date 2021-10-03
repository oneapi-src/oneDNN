/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <type_traits>

#include "oneapi/dnnl/dnnl_config.h"

#include <CL/sycl.hpp>

#ifdef DNNL_WITH_LEVEL_ZERO
#include <level_zero/ze_api.h>

#include <CL/sycl/backend/level_zero.hpp>
#endif

#include "common/utils.hpp"
#include "gpu/compute/device_info.hpp"
#include "sycl/level_zero_utils.hpp"
#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_gpu_engine.hpp"
#include "sycl/sycl_utils.hpp"

#if DNNL_USE_SYCL121_API
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace dnnl {
namespace impl {
namespace sycl {

#ifdef DNNL_WITH_LEVEL_ZERO
status_t func_zeKernelCreate(
        ze_module_handle_t, const ze_kernel_desc_t *, ze_kernel_handle_t *);
#endif

namespace compat {

using namespace gpu::compute;

namespace {
#if !DNNL_USE_SYCL121_API
status_t get_kernel_from_bundle(std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const ::sycl::kernel_bundle<::sycl::bundle_state::executable>
                &kernel_bundle,
        const std::string &kernel_name, const sycl_gpu_engine_t *sycl_engine) {

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
#ifdef DNNL_WITH_LEVEL_ZERO
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
#else // DNNL_WITH_LEVEL_ZERO
        assert(!"unexpected");
        return status::invalid_arguments;
#endif
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

} // namespace

status_t make_kernel(std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const std::string &kernel_name, const sycl_gpu_engine_t *sycl_engine,
        void *native_program_handle, const binary_t *binary,
        program_list_t *programs) {
    auto backend = get_sycl_backend(sycl_engine->device());

    if (backend == backend_t::opencl) {
        cl_program ocl_program
                = reinterpret_cast<cl_program>(native_program_handle);
#if DNNL_USE_SYCL121_API
        auto sycl_program = ::sycl::opencl::make<::sycl::program>(
                sycl_engine->context(), ocl_program);

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
#ifdef DNNL_WITH_LEVEL_ZERO
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
#else // DNNL_WITH_LEVEL_ZERO
        assert(!"unexpected");
        return status::invalid_arguments;
#endif
    } else {
        assert(!"unexpected");
        return status::invalid_arguments;
    }
    return status::success;
}

status_t make_kernel(std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const std::string &kernel_name, const sycl_gpu_engine_t *sycl_engine,
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

#if DNNL_USE_SYCL121_API
#pragma clang diagnostic pop
#endif

} // namespace compat
} // namespace sycl
} // namespace impl
} // namespace dnnl
