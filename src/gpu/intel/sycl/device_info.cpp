/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gpu/intel/sycl/device_info.hpp"
#include "gpu/intel/sycl/compat.hpp"
#include "gpu/intel/sycl/engine.hpp"

#include "gpu/intel/ocl/hw_info.hpp"

#include "gpu/intel/ze/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

status_t device_info_t::init_arch(impl::engine_t *engine) {
    auto *sycl_engine
            = utils::downcast<const gpu::intel::sycl::engine_t *>(engine);
    auto &device = sycl_engine->device();
    auto &ctx = sycl_engine->context();

    // skip cpu engines
    if (!device.is_gpu()) return status::success;

    // skip other vendors
    if (!xpu::sycl::is_intel_device(device)) return status::success;

    auto status = status::success;
    auto be = xpu::sycl::get_backend(device);
    if (be == xpu::sycl::backend_t::opencl) {
        auto ocl_dev = xpu::sycl::compat::get_native<cl_device_id>(device);
        auto ocl_ctx = xpu::sycl::compat::get_native<cl_context>(ctx);

        status = gpu::intel::ocl::init_gpu_hw_info(engine, ocl_dev, ocl_ctx,
                ip_version_, gpu_arch_, product_, native_extensions_,
                mayiuse_systolic_, mayiuse_ngen_kernels_, is_efficient_64bit_);
    } else if (be == xpu::sycl::backend_t::ze) {
        auto ze_dev = xpu::sycl::compat::get_native<ze_device_handle_t>(device);
        auto ze_ctx = xpu::sycl::compat::get_native<ze_context_handle_t>(ctx);

        status = gpu::intel::ze::init_gpu_hw_info(engine, ze_dev, ze_ctx,
                ip_version_, gpu_arch_, product_, native_extensions_,
                mayiuse_systolic_, mayiuse_ngen_kernels_, is_efficient_64bit_);
    } else {
        assert(!"not_expected");
        status = status::unimplemented;
    }

    return status;
}

status_t device_info_t::init_device_name(impl::engine_t *engine) {
    auto &device
            = utils::downcast<const xpu::sycl::engine_impl_t *>(engine->impl())
                      ->device();
    name_ = device.get_info<::sycl::info::device::name>();
    return status::success;
}

status_t device_info_t::init_runtime_version(impl::engine_t *engine) {
    auto &device
            = utils::downcast<const xpu::sycl::engine_impl_t *>(engine->impl())
                      ->device();
    auto driver_version
            = device.get_info<::sycl::info::device::driver_version>();

    if (runtime_version_.set_from_string(driver_version.c_str())
            != status::success) {
        runtime_version_.major = 0;
        runtime_version_.minor = 0;
        runtime_version_.build = 0;
    }

    return status::success;
}

status_t device_info_t::init_extensions(impl::engine_t *engine) {
    using namespace gpu::intel::compute;

    auto &device
            = utils::downcast<const xpu::sycl::engine_impl_t *>(engine->impl())
                      ->device();
#ifdef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
    for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
            i_ext <<= 1) {
        const char *s_ext = ext2cl_str((device_ext_t)i_ext);
        if (s_ext && device.ext_oneapi_supports_cl_extension(s_ext)) {
            extensions_ |= i_ext;
        }
    }
#else
    extensions_ = gpu::intel::sycl::compat::init_extensions(device);
#endif // DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER

    // Handle future extensions, not yet supported by the DPC++ API
    extensions_
            |= (uint64_t)get_future_extensions(gpu_arch(), mayiuse_systolic());

    return status::success;
}

status_t device_info_t::init_attributes(impl::engine_t *engine) {
    auto &device
            = utils::downcast<const xpu::sycl::engine_impl_t *>(engine->impl())
                      ->device();
    if (device.has(::sycl::aspect::ext_intel_gpu_eu_count)) {
        eu_count_ = static_cast<int32_t>(
                device.get_info<
                        ::sycl::ext::intel::info::device::gpu_eu_count>());
    } else {
        eu_count_ = device.get_info<::sycl::info::device::max_compute_units>();
    }
    max_clock_mhz_ = static_cast<int32_t>(
            device.get_info<::sycl::info::device::max_clock_frequency>());
    max_wg_size_ = device.get_info<::sycl::info::device::max_work_group_size>();
    memory_size_ = device.get_info<::sycl::info::device::global_mem_size>();
    l3_cache_size_
            = device.get_info<::sycl::info::device::global_mem_cache_size>();
    mayiuse_system_memory_allocators_
            = device.has(::sycl::aspect::usm_system_allocations);
    device_address_bits_
            = device.get_info<::sycl::info::device::address_bits>();
    max_allocation_size_
            = device.get_info<::sycl::info::device::max_mem_alloc_size>();
    return status::success;
}

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
