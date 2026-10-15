/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/intel/ze/device_info.hpp"
#include "gpu/intel/ze/engine.hpp"
#include "gpu/intel/ze/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ze {

status_t device_info_t::init_device_name(impl::engine_t *engine) {
    auto *ze_engine = utils::downcast<const gpu::intel::ze::engine_t *>(engine);
    auto device = ze_engine->device();

    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    ZE_CHECK(xpu::ze::zeDeviceGetProperties(device, &device_properties));
    name_ = std::string(device_properties.name);

    return status::success;
}

status_t device_info_t::init_arch(impl::engine_t *engine) {
    auto *ze_engine = utils::downcast<const gpu::intel::ze::engine_t *>(engine);
    auto context = ze_engine->context();
    auto device = ze_engine->device();

    return init_gpu_hw_info(engine, device, context, ip_version_, gpu_arch_,
            product_, native_extensions_, mayiuse_systolic_,
            mayiuse_ngen_kernels_, is_efficient_64bit_);
}

status_t device_info_t::init_runtime_version(impl::engine_t *engine) {
    auto *ze_engine = utils::downcast<const gpu::intel::ze::engine_t *>(engine);
    auto driver = ze_engine->driver();

    ze_driver_properties_t driver_properties = {};
    driver_properties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;

    ZE_CHECK(xpu::ze::zeDriverGetProperties(driver, &driver_properties));

    // Note: for some reason this returns a different value for a minor version
    // when compared to internal API to get a version string. For a version
    // checked the correct one returns 14 and this version returned 3.
    runtime_version_.major
            = (driver_properties.driverVersion & 0xFF000000) >> 24;
    runtime_version_.minor
            = (driver_properties.driverVersion & 0x00FF0000) >> 16;
    runtime_version_.build = driver_properties.driverVersion & 0x0000FFFF;

    return status::success;
}

status_t device_info_t::init_extensions(impl::engine_t *engine) {
    std::string extension_string;
    // TODO: using OpenCL runtime because Level Zero runtime does not provide
    // this information.
    auto *ze_engine = utils::downcast<const gpu::intel::ze::engine_t *>(engine);
    CHECK(xpu::ocl::get_extensions(ze_engine->ocl_device(), extension_string));

    for (uint64_t i_ext = 1; i_ext < (uint64_t)compute::device_ext_t::last;
            i_ext <<= 1) {
        const char *s_ext = ext2cl_str((compute::device_ext_t)i_ext);

        if (s_ext && extension_string.find(s_ext) != std::string::npos) {
            extensions_ |= i_ext;
        }
    }

    extensions_
            |= (uint64_t)get_future_extensions(gpu_arch(), mayiuse_systolic());

    return status::success;
}

status_t device_info_t::init_attributes(impl::engine_t *engine) {
    auto *ze_engine = utils::downcast<const gpu::intel::ze::engine_t *>(engine);
    auto device = ze_engine->device();

    ze_eu_count_ext_t eu_count_ext = {};
    eu_count_ext.stype = ZE_STRUCTURE_TYPE_EU_COUNT_EXT;

    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    device_properties.pNext = &eu_count_ext;

    ZE_CHECK(xpu::ze::zeDeviceGetProperties(device, &device_properties));

    eu_count_ = eu_count_ext.numTotalEUs;
    max_clock_mhz_ = static_cast<int32_t>(device_properties.coreClockRate);

    ze_device_compute_properties_t device_compute_properties = {};
    device_compute_properties.stype
            = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;

    ZE_CHECK(xpu::ze::zeDeviceGetComputeProperties(
            device, &device_compute_properties));

    max_wg_size_ = device_compute_properties.maxTotalGroupSize;

    uint32_t device_cache_properties_count = 0;
    ZE_CHECK(xpu::ze::zeDeviceGetCacheProperties(
            device, &device_cache_properties_count, nullptr));

    std::vector<ze_device_cache_properties_t> device_cache_properties(
            device_cache_properties_count);
    for (ze_device_cache_properties_t &p : device_cache_properties) {
        p.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
    }

    ZE_CHECK(xpu::ze::zeDeviceGetCacheProperties(device,
            &device_cache_properties_count, device_cache_properties.data()));
    for (uint32_t i = 0; i < device_cache_properties_count; i++) {
        if (device_cache_properties[i].flags == 0) {
            l3_cache_size_ = device_cache_properties[i].cacheSize;
            break;
        }
    }

    ze_device_module_properties_t device_module_props = {};
    device_module_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    ZE_CHECK(
            xpu::ze::zeDeviceGetModuleProperties(device, &device_module_props));
    max_kernel_param_size_ = device_module_props.maxArgumentsSize;

    ze_device_memory_access_properties_t device_memory_access_properties = {};
    device_memory_access_properties.stype
            = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;

    ZE_CHECK(xpu::ze::zeDeviceGetMemoryAccessProperties(
            device, &device_memory_access_properties));
    mayiuse_system_memory_allocators_
            = device_memory_access_properties.sharedSystemAllocCapabilities;

    return status::success;
}

} // namespace ze
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
