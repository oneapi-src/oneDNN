/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "gpu/intel/ocl/device_info.hpp"
#include "gpu/intel/ocl/engine.hpp"
#include "gpu/intel/ocl/hw_info.hpp"

#include <CL/cl_ext.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t device_info_t::init_arch(impl::engine_t *engine) {
    cl_int err = CL_SUCCESS;
    auto device = utils::downcast<const engine_t *>(engine)->device();

    // skip other vendors
    const cl_uint intel_vendor_id = 0x8086;
    cl_uint vendor_id;
    err = xpu::ocl::clGetDeviceInfo(
            device, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, nullptr);
    OCL_CHECK(err);
    if (vendor_id != intel_vendor_id) return status::success;

    cl_context context = xpu::ocl::clCreateContext(
            nullptr, 1, &device, nullptr, nullptr, &err);
    OCL_CHECK(err);

    CHECK(init_gpu_hw_info(engine, device, context, ip_version_, gpu_arch_,
            product_, native_extensions_, mayiuse_systolic_,
            mayiuse_ngen_kernels_, is_efficient_64bit_));

    err = xpu::ocl::clReleaseContext(context);
    OCL_CHECK(err);

    // XXX: temporary WA for different Xe_HP devices
    if (gpu_arch_ == compute::gpu_arch_t::xe_hp) {
        // query extensions
        size_t param_size = 0;
        err = xpu::ocl::clGetDeviceInfo(
                device, CL_DEVICE_EXTENSIONS, 0, nullptr, &param_size);
        OCL_CHECK(err);

        std::string extension_string(param_size, '\0');
        err = xpu::ocl::clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS,
                param_size, &extension_string[0], &param_size);
        OCL_CHECK(err);
        if (extension_string.find(ext2cl_str(compute::device_ext_t::khr_fp64))
                == std::string::npos)
            gpu_arch_ = compute::gpu_arch_t::xe_hpg;
    }
    return status::success;
}

status_t device_info_t::init_device_name(impl::engine_t *engine) {
    cl_int err = CL_SUCCESS;
    auto device = utils::downcast<const engine_t *>(engine)->device();

    size_t param_size = 0;
    err = xpu::ocl::clGetDeviceInfo(
            device, CL_DEVICE_NAME, 0, nullptr, &param_size);
    OCL_CHECK(err);

    name_ = std::string(param_size, '\0');
    err = xpu::ocl::clGetDeviceInfo(
            device, CL_DEVICE_NAME, param_size, &name_[0], &param_size);
    OCL_CHECK(err);

    return status::success;
}

status_t device_info_t::init_runtime_version(impl::engine_t *engine) {
    auto device = utils::downcast<const engine_t *>(engine)->device();
    runtime_version_ = get_driver_version(device);
    return status::success;
}

status_t device_info_t::init_extensions(impl::engine_t *engine) {
    auto device = utils::downcast<const engine_t *>(engine)->device();

    // query device for extensions
    std::string extension_string;
    CHECK(xpu::ocl::get_extensions(device, extension_string));

    // convert to ours
    using namespace compute;
    for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
            i_ext <<= 1) {
        const char *s_ext = ext2cl_str((device_ext_t)i_ext);
        if (s_ext && extension_string.find(s_ext) != std::string::npos) {
            extensions_ |= i_ext;
        }
    }

    // Handle future extensions, not yet supported by the OpenCL API
    extensions_
            |= (uint64_t)get_future_extensions(gpu_arch(), mayiuse_systolic());

    return status::success;
}

status_t device_info_t::init_attributes(impl::engine_t *engine) {
    cl_int err = CL_SUCCESS;
    auto device = utils::downcast<const engine_t *>(engine)->device();

    CHECK(get_ocl_device_eu_count(device, &eu_count_));

    cl_uint max_clock_mhz = 0;
    if (xpu::ocl::clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                sizeof(max_clock_mhz), &max_clock_mhz, nullptr)
            == CL_SUCCESS)
        max_clock_mhz_ = static_cast<int32_t>(max_clock_mhz);

    size_t max_wg_size = 0;
    err = xpu::ocl::clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(max_wg_size), &max_wg_size, nullptr);
    OCL_CHECK(err);
    max_wg_size_ = max_wg_size;

    cl_ulong mem_size;
    err = xpu::ocl::clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(mem_size), &mem_size, nullptr);
    OCL_CHECK(err);
    memory_size_ = mem_size;

    cl_ulong mem_cache_size;
    err = xpu::ocl::clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
            sizeof(mem_cache_size), &mem_cache_size, nullptr);
    OCL_CHECK(err);
    l3_cache_size_ = mem_cache_size;

    size_t max_kernel_param_size;
    err = xpu::ocl::clGetDeviceInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE,
            sizeof(max_kernel_param_size), &max_kernel_param_size, nullptr);
    OCL_CHECK(err);
    max_kernel_param_size_ = max_kernel_param_size;

    cl_uint device_address_bits;
    err = xpu::ocl::clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS,
            sizeof(device_address_bits), &device_address_bits, nullptr);
    OCL_CHECK(err);
    device_address_bits_ = device_address_bits;

    cl_ulong max_allocation_size;
    err = xpu::ocl::clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
            sizeof(max_allocation_size), &max_allocation_size, nullptr);
    OCL_CHECK(err);
    max_allocation_size_ = max_allocation_size;

#ifdef cl_intel_unified_shared_memory
    cl_device_unified_shared_memory_capabilities_intel
            system_memory_capabilities_intel
            = 0;
    err = xpu::ocl::clGetDeviceInfo(device,
            CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL,
            sizeof(cl_device_unified_shared_memory_capabilities_intel),
            &system_memory_capabilities_intel, nullptr);
    OCL_CHECK(err);
    mayiuse_system_memory_allocators_ = system_memory_capabilities_intel
            & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL;
#endif

    return status::success;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
