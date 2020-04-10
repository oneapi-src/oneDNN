/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "dnnl.hpp"

#include "dnnl_memory.hpp"
#include "dnnl_reorder.hpp"

#if DNNL_WITH_SYCL
#include <CL/sycl.hpp>
#endif

int dnn_mem_t::reorder(const dnn_mem_t &rhs, const attr_bundle_t *attr_bundle) {
    if (this == &rhs) return OK;
    return execute_reorder(rhs, *this, attr_bundle);
}

dnn_mem_t dnn_mem_t::create_from_host_ptr(
        const dnnl_memory_desc_t &md, dnnl_engine_t engine, void *host_ptr) {
    dnnl_engine_kind_t eng_kind;
    DNN_SAFE_V(dnnl_engine_get_kind(engine, &eng_kind));

    // XXX: allows to construct CPU memory only.
    assert(eng_kind == dnnl_cpu);
    (void)eng_kind;

    std::shared_ptr<void> handle;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    using buf_type = cl::sycl::buffer<uint8_t, 1>;
    size_t sz = dnnl_memory_desc_get_size(&md);
    handle.reset(new buf_type((uint8_t *)host_ptr, cl::sycl::range<1>(sz)),
            [](void *ptr) { delete (buf_type *)ptr; });

#else
    handle.reset(host_ptr, [](void *) {});
#endif
    return dnn_mem_t(md, engine, handle.get());
}

#if defined(_WIN32) && !defined(__GNUC__)
#include "windows.h"

static size_t get_cpu_ram_size() {
    MEMORYSTATUSEX s {};
    s.dwLength = sizeof(s);
    GlobalMemoryStatusEx(&s);
    return s.ullTotalPhys;
}
#elif defined(__APPLE__)
#include <unistd.h>
#include <sys/sysctl.h>

static size_t get_cpu_ram_size() {
    int query_ram[] = {CTL_HW, HW_MEMSIZE};
    int query_ram_len = sizeof(query_ram) / sizeof(*query_ram);
    size_t totalram = 0;
    size_t length = sizeof(totalram);

    sysctl(query_ram, query_ram_len, &totalram, &length, NULL, 0);
    return totalram;
}
#else
#include <sys/sysinfo.h>

static size_t get_cpu_ram_size() {
    struct sysinfo s {};
    sysinfo(&s);
    return s.totalram;
}
#endif

static size_t get_gpu_ram_size() {
    dnnl::engine eng(engine_tgt, true);
    if (eng.get_kind() != dnnl::engine::kind::gpu) return 0;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_int status = CL_SUCCESS;
    // Get single device attached to the engine.
    cl_device_id ocl_device = eng.get_ocl_device();

    cl_ulong ram_size = 0;
    status = clGetDeviceInfo(ocl_device, CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(cl_ulong), &ram_size, NULL);
    if (status == CL_SUCCESS) return (size_t)ram_size;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    auto sycl_dev = eng.get_sycl_device();
    return (size_t)sycl_dev.get_info<cl::sycl::info::device::global_mem_size>();
#endif
    return 0;
}

int dnn_mem_t::check_mem_size(const_dnnl_primitive_desc_t const_pd) {
    if (!mem_check) return OK;

    static uint64_t cpu_device_capacity = get_cpu_ram_size();
    static uint64_t gpu_device_capacity = get_gpu_ram_size();

    const uint64_t devices_max_capacity = engine_tgt_kind == dnnl_cpu
            ? cpu_device_capacity
            : MIN2(cpu_device_capacity, gpu_device_capacity);
    // 0.75f is taken randomly. A subject to change in future.
    const double benchdnn_limit = 0.75f * devices_max_capacity;
    assert(benchdnn_limit > 0);

    // get all amount of memories to collect mem_size over all of them
    const int n_memories = dnnl_primitive_desc_query_s32(
                                   const_pd, dnnl_query_num_of_inputs_s32, 0)
            + dnnl_primitive_desc_query_s32(
                    const_pd, dnnl_query_num_of_outputs_s32, 0);

    const auto get_mem_size = [const_pd](dnnl_query_t query, int index = 0) {
        const auto md = dnnl_primitive_desc_query_md(const_pd, query, index);
        auto mem_size = dnnl_memory_desc_get_size(md);
        // runtime mem size is not defined
        if (mem_size == DNNL_RUNTIME_SIZE_VAL) mem_size = 0;
        return 2 * mem_size; // 2 for library and benchdnn ref memories
    };

    double total_mem_size = 0;

#define MD(name) dnnl_query_##name##_md
    for (auto query : {MD(src), MD(diff_src), MD(weights), MD(diff_weights),
                 MD(dst), MD(diff_dst)}) {
        for (int idx = 0; idx < n_memories; ++idx)
            total_mem_size += get_mem_size(query, idx);
    }

    for (auto query : {MD(workspace), MD(scratchpad)})
        total_mem_size += get_mem_size(query);
#undef MD

    int64_t library_internal_mem_size = 0;
    dnnl_primitive_desc_query(const_pd, dnnl_query_memory_consumption_s64, 0,
            &library_internal_mem_size);
    total_mem_size += library_internal_mem_size;

    const bool fits_device_ram = total_mem_size <= benchdnn_limit;
    if (!fits_device_ram) {
        auto GB = [](double bytes) { return bytes / powf(2, 30); };

        BENCHDNN_PRINT(2,
                "benchdnn: not enough RAM for a problem.\nRequested: %g GB, "
                "benchdnn limit: %g GB, CPU RAM capacity: %g GB, GPU RAM "
                "capacity: %g GB\n",
                GB(total_mem_size), GB(benchdnn_limit), GB(cpu_device_capacity),
                GB(gpu_device_capacity));
    }

    return fits_device_ram ? OK : FAIL;
}
