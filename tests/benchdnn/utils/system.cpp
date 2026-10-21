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

#include <vector>

#include <assert.h>

#include "oneapi/dnnl/dnnl.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_ocl.hpp"
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
#include "oneapi/dnnl/dnnl_ze.hpp"
#endif

#include "cpu/platform.hpp"

#include "common.hpp"
#include "utils/engine.hpp"
#include "utils/system.hpp"

#if defined(_WIN32)
#include "windows.h"

size_t get_cpu_ram_size() {
    MEMORYSTATUSEX s {};
    s.dwLength = sizeof(s);
    GlobalMemoryStatusEx(&s);
    return s.ullTotalPhys;
}
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__QNXNTO__)
#include <unistd.h>
#include <sys/sysctl.h>

size_t get_cpu_ram_size() {
#ifdef __APPLE__
    int query_ram[] = {CTL_HW, HW_MEMSIZE};
#else
    int query_ram[] = {CTL_HW, HW_PHYSMEM};
#endif
    int query_ram_len = sizeof(query_ram) / sizeof(*query_ram);
    size_t totalram = 0;
    size_t length = sizeof(totalram);

    sysctl(query_ram, query_ram_len, &totalram, &length, NULL, 0);
    return totalram;
}
#else
#include <sys/sysinfo.h>

size_t get_cpu_ram_size() {
    struct sysinfo s {};
    sysinfo(&s);
    return s.totalram;
}
#endif

int get_gpu_ram_sizes(size_t &ram_size, size_t &max_alloc_size) {
    if (!is_gpu()) return OK;
    if (ram_size > 0 && max_alloc_size > 0) return OK;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    auto eng = dnnl::engine(get_test_engine(), true);
    cl_device_id ocl_dev = dnnl::ocl_interop::get_device(eng);

    cl_ulong ram_sz = 0;
    cl_int status = clGetDeviceInfo(ocl_dev, CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(cl_ulong), &ram_sz, nullptr);
    if (status != CL_SUCCESS) return FAIL;

    ram_size = (size_t)ram_sz;
    // For OCL runtime we allow allocation of buffers up to VRAM size,
    // with the usage of CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL flag.
    max_alloc_size = (size_t)ram_sz;
    return OK;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    auto eng = dnnl::engine(get_test_engine(), true);
    auto sycl_dev = dnnl::sycl_interop::get_device(eng);
    ram_size = (size_t)sycl_dev
                       .get_info<::sycl::info::device::global_mem_size>();
    max_alloc_size
            = (size_t)sycl_dev
                      .get_info<::sycl::info::device::max_mem_alloc_size>();
    return OK;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    auto eng = dnnl::engine(get_test_engine(), true);
    auto ze_dev = dnnl::ze_interop::get_device(eng);

    ze_result_t status = ZE_RESULT_SUCCESS;

    uint32_t count = 0;
    status = zeDeviceGetMemoryProperties(ze_dev, &count, nullptr);
    if (status != ZE_RESULT_SUCCESS) return FAIL;
    if (count > 1) {
        assert(!"Found more than a single entry for memory.");
        return FAIL;
    }

    ze_device_memory_properties_t ze_dev_mem_props {};
    ze_dev_mem_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    status = zeDeviceGetMemoryProperties(ze_dev, &count, &ze_dev_mem_props);
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    ram_size = (size_t)ze_dev_mem_props.totalSize;

    ze_device_properties_t ze_dev_props {};
    ze_dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ze_dev_props.pNext = nullptr;
    status = zeDeviceGetProperties(ze_dev, &ze_dev_props);
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    max_alloc_size = (size_t)ze_dev_props.maxMemAllocSize;
    return OK;
#else
    assert(!"unsupported GPU runtime");
#endif
    ram_size = 0;
    max_alloc_size = 0;
    return OK;
}

int get_cpu_cache_size(cpu_cache_args_t &cache_args) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    using namespace dnnl::impl::cpu::platform;
    cache_args.L2_size = get_per_core_cache_size(2);
    cache_args.L3_size = get_per_core_cache_size(3);
    cache_args.num_cores = get_num_cores();
    cache_args.total_socket_size
            = (cache_args.L2_size + cache_args.L3_size) * cache_args.num_cores;
#else
    // If functions are not available, just use 150 MiB.
    cache_args.total_socket_size = 150 * 1024 * 1024;
#endif
    return OK;
}

int get_gpu_cache_size(size_t &cache_size) {
    if (!is_gpu()) return OK;

    static size_t _cache_size = 0;
    if (_cache_size > 0) {
        cache_size = _cache_size;
        return OK;
    }

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    auto eng = dnnl::engine(get_test_engine(), true);
    cl_int status = CL_SUCCESS;
    cl_device_id ocl_device = dnnl::ocl_interop::get_device(eng);

    cl_ulong cache_sz = 0;
    status = clGetDeviceInfo(ocl_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
            sizeof(cl_ulong), &cache_sz, nullptr);
    if (status != CL_SUCCESS) return FAIL;

    _cache_size = (size_t)cache_sz;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    auto eng = dnnl::engine(get_test_engine(), true);
    auto sycl_dev = dnnl::sycl_interop::get_device(eng);
    _cache_size
            = (size_t)sycl_dev
                      .get_info<::sycl::info::device::global_mem_cache_size>();
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    auto eng = dnnl::engine(get_test_engine(), true);
    auto ze_dev = dnnl::ze_interop::get_device(eng);

    ze_result_t status = ZE_RESULT_SUCCESS;

    uint32_t count = 0;
    status = zeDeviceGetCacheProperties(ze_dev, &count, nullptr);
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    std::vector<ze_device_cache_properties_t> ze_cache_props(count);
    for (auto &e : ze_cache_props) {
        e.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
    }

    status = zeDeviceGetCacheProperties(ze_dev, &count, ze_cache_props.data());
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    for (uint32_t i = 0; i < count; i++) {
        if (ze_cache_props[i].flags == 0) {
            _cache_size = ze_cache_props[i].cacheSize;
            break;
        }
    }
#else
    assert(!"unsupported GPU runtime");
#endif
    cache_size = _cache_size;
    return OK;
}
