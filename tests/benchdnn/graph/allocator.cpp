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

#include <functional>
#include "allocator.hpp"
#include "graph_memory.hpp"

// Throw an exception for not enough memory, which will be caught in
// DNN_GRAPH_SAFE.
#define CHECK_GRAPH_MEM_SIZE(flag, size) \
    if ((flag) && !check_memory_size(size)) \
        throw dnnl::error(dnnl_out_of_memory, "not enough memory");

namespace graph {

namespace {

bool check_memory_size(size_t size) {
    auto &graph_mem_req = graph_memory_req_args_t::get_instance();
    if (is_gpu()) {
        size_t benchdnn_device_limit = get_benchdnn_device_limit();
        size_t total_gpu_req = graph_mem_req.get_mem_req(GPU_REQ) + size;
        const bool fits_device_ram = total_gpu_req <= benchdnn_device_limit;

        if (!fits_device_ram) {
            BENCHDNN_PRINT(2,
                    "[CHECK_MEM]: Not enough device RAM for a problem. "
                    "Allocation of size %g GB doesn't fit allocation limit of "
                    "%g GB. \n",
                    GB(total_gpu_req), GB(benchdnn_device_limit));
            return false;
        }
        graph_mem_req.increase_mem_req(GPU_REQ, GRAPH_INTERNAL, size);
    } else {
        size_t benchdnn_cpu_limit = get_benchdnn_cpu_limit();
        size_t total_cpu_req = graph_mem_req.get_mem_req(CPU_REQ) + size;

        bool fits_cpu_ram = total_cpu_req <= benchdnn_cpu_limit;
        if (!fits_cpu_ram) {
            BENCHDNN_PRINT(2,
                    "[CHECK_MEM]: Not enough CPU RAM for a problem. Allocation "
                    "of "
                    "size %g GB doesn't fit allocation limit of %g GB. \n",
                    GB(total_cpu_req), GB(benchdnn_cpu_limit));
            return false;
        }
        graph_mem_req.increase_mem_req(CPU_REQ, GRAPH_INTERNAL, size);
    }

    return true;
}

} // namespace

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
void graph_mem_manager_t::clear_memory_pool() {
    mem_pool_.clear();
}
#endif

void *graph_mem_manager_t::host_malloc_wrapper(
        size_t size, size_t alignment) const {
    CHECK_GRAPH_MEM_SIZE(need_mem_check_, size);

    void *ptr = nullptr;
    const size_t align = alignment == 0 ? default_alignment : alignment;
#ifdef _WIN32
    ptr = _aligned_malloc(size, align);
    int rc = ((ptr) ? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, align, size);
#endif /* _WIN32 */
    return (rc == 0) ? ptr : nullptr;
}

void graph_mem_manager_t::host_free_wrapper(void *ptr) {
#ifdef _WIN32
    _aligned_free((void *)ptr);
#else
    ::free((void *)ptr);
#endif /* _WIN32 */
}

void *host_allocator(size_t size, size_t alignment) {
    auto &graph_mem_mgr = graph_mem_manager_t::get_instance();
    return graph_mem_mgr.host_malloc_wrapper(size, alignment);
}

void host_deallocator(void *ptr) {
    auto &graph_mem_mgr = graph_mem_manager_t::get_instance();
    graph_mem_mgr.host_free_wrapper(ptr);
}

#ifdef DNNL_WITH_SYCL
void *graph_mem_manager_t::sycl_malloc_wrapper(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    void *ptr {nullptr};
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (!has_bench_mode_bit(mode_bit_t::corr) && is_gpu()) {
        bool need_alloc_new_mm = mem_pool_.check_allocated_mem(ptr, size);

        if (need_alloc_new_mm) {
            CHECK_GRAPH_MEM_SIZE(need_mem_check_, size);
            ptr = mem_pool_.allocate(size, alignment, dev, ctx);
        }
        return ptr;
    }
#endif
    CHECK_GRAPH_MEM_SIZE(need_mem_check_, size);
    ptr = default_sycl_malloc(size, alignment, dev, ctx);
    return ptr;
}

void graph_mem_manager_t::sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (!has_bench_mode_bit(mode_bit_t::corr) && is_gpu())
        mem_pool_.deallocate(ptr, device, context, event);
    else
#endif
        default_sycl_free(ptr, device, context, event);
}

void *sycl_allocator(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    auto &graph_mem_mgr = graph_mem_manager_t::get_instance();
    return graph_mem_mgr.sycl_malloc_wrapper(size, alignment, dev, ctx);
}

void sycl_deallocator(
        void *ptr, const void *device, const void *context, void *event) {
    auto &graph_mem_mgr = graph_mem_manager_t::get_instance();
    graph_mem_mgr.sycl_free_wrapper(ptr, device, context, event);
}

#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
void *graph_mem_manager_t::ocl_malloc_wrapper(size_t size, size_t alignment,
        cl_device_id device, cl_context context) {
    void *ptr {nullptr};
    bool need_alloc_new_mm = mem_pool_.check_allocated_mem(ptr, size);

    if (need_alloc_new_mm) {
        CHECK_GRAPH_MEM_SIZE(need_mem_check_, size);
        ptr = mem_pool_.allocate(size, alignment, device, context);
    }
    return ptr;
}

void graph_mem_manager_t::ocl_free_wrapper(
        void *buf, cl_device_id device, cl_context context, cl_event event) {
    mem_pool_.deallocate(buf, device, context, event);
}

void *ocl_allocator(size_t size, size_t alignment, cl_device_id device,
        cl_context context) {
    auto &graph_mem_mgr = graph_mem_manager_t::get_instance();
    return graph_mem_mgr.ocl_malloc_wrapper(size, alignment, device, context);
}

void ocl_deallocator(
        void *buf, cl_device_id device, cl_context context, cl_event event) {
    auto &graph_mem_mgr = graph_mem_manager_t::get_instance();
    graph_mem_mgr.ocl_free_wrapper(buf, device, context, event);
}
#endif

dnnl::graph::allocator &get_graph_allocator() {
    if (is_cpu()) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        static auto alloc = dnnl::graph::sycl_interop::make_allocator(
                sycl_allocator, sycl_deallocator);
#else
        static dnnl::graph::allocator alloc(host_allocator, host_deallocator);
#endif
        return alloc;
    } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        static auto alloc = dnnl::graph::sycl_interop::make_allocator(
                sycl_allocator, sycl_deallocator);
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        static auto alloc = dnnl::graph::ocl_interop::make_allocator(
                ocl_allocator, ocl_deallocator);
#else
        static dnnl::graph::allocator alloc(host_allocator, host_deallocator);
#endif
        return alloc;
    }
}

} // namespace graph
