/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include "gpu/intel/sycl/utils.hpp"
#include "gpu/intel/utils.hpp"
#include "oneapi/dnnl/dnnl_config.h"

#include "gpu/intel/sycl/l0/level_zero/ze_api.h"

#if __has_include(<sycl/backend/opencl.hpp>)
#include <sycl/backend/opencl.hpp>
#elif __has_include(<CL/sycl/backend/opencl.hpp>)
#include <CL/sycl/backend/opencl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "common/utils.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/sycl/compat.hpp"
#include "gpu/intel/sycl/engine.hpp"
#include "gpu/intel/sycl/l0/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

namespace compat {

using namespace gpu::intel::compute;

status_t make_kernels(
        std::vector<std::unique_ptr<::sycl::kernel>> &sycl_kernels,
        const std::vector<const char *> &kernel_names,
        const gpu::intel::sycl::engine_t *sycl_engine,
        const xpu::binary_t &binary) {
    auto backend = xpu::sycl::get_backend(sycl_engine->device());
    if (backend == xpu::sycl::backend_t::opencl) {
        xpu::ocl::wrapper_t<cl_program> ocl_program;
        CHECK(xpu::ocl::create_program(ocl_program, sycl_engine->ocl_device(),
                sycl_engine->ocl_context(), binary));

        sycl_kernels.resize(kernel_names.size());
        for (size_t i = 0; i < kernel_names.size(); i++) {
            if (kernel_names[i] == nullptr) continue;
            cl_int err;
            xpu::ocl::wrapper_t<cl_kernel> ocl_kernel
                    = clCreateKernel(ocl_program, kernel_names[i], &err);
            OCL_CHECK(err);
            sycl_kernels[i] = utils::make_unique<::sycl::kernel>(
                    ::sycl::make_kernel<::sycl::backend::opencl>(
                            ocl_kernel, sycl_engine->context()));
        }
    } else if (backend == xpu::sycl::backend_t::level0) {
        CHECK(sycl_create_kernels_with_level_zero(
                sycl_kernels, kernel_names, sycl_engine, binary));
    } else {
        gpu_error_not_expected();
        return status::invalid_arguments;
    }
    return status::success;
}

status_t make_kernel(std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const char *kernel_name, const gpu::intel::sycl::engine_t *sycl_engine,
        const xpu::binary_t &binary) {
    std::vector<std::unique_ptr<::sycl::kernel>> sycl_kernels;
    std::vector<const char *> kernel_names = {kernel_name};
    CHECK(make_kernels(sycl_kernels, kernel_names, sycl_engine, binary));

    if (sycl_kernels.empty()) return status::runtime_error;

    sycl_kernel = std::move(sycl_kernels[0]);
    return status::success;
}

uint64_t init_extensions(const ::sycl::device &dev) {
    uint64_t extensions = 0;

    constexpr auto base_atomics_aspect = ::sycl::aspect::atomic64;
    constexpr auto extended_atomics_aspect = ::sycl::aspect::atomic64;

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

    return extensions;
}

} // namespace compat
} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
