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

#ifndef BENCHDNN_GRAPH_ALLOCATOR_HPP
#define BENCHDNN_GRAPH_ALLOCATOR_HPP

#include <unordered_set>

#include "dnnl_common.hpp"
#include "memory_pool.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

namespace graph {

constexpr static size_t default_alignment = 64;
struct graph_mem_manager_t {

public:
    static graph_mem_manager_t &get_instance() {
        static graph_mem_manager_t _instance;
        return _instance;
    }

    void *host_malloc_wrapper(size_t size, size_t alignment) const;
    void host_free_wrapper(void *ptr);

#ifdef DNNL_WITH_SYCL
    void *sycl_malloc_wrapper(
            size_t size, size_t alignment, const void *dev, const void *ctx);
    // Note: For performance mode, the memory pool will cache the allocated
    // memories in the free function, but not really release them. The cached
    // memories will be finally released with the deconstruction of the cache
    // manager.
    void sycl_free_wrapper(
            void *ptr, const void *device, const void *context, void *event);
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void *ocl_malloc_wrapper(size_t size, size_t alignment, cl_device_id device,
            cl_context context);
    void ocl_free_wrapper(
            void *buf, cl_device_id device, cl_context context, cl_event event);
#endif

    void start_graph_mem_check() { need_mem_check_ = true; }
    void stop_graph_mem_check() { need_mem_check_ = false; }

#if defined(DNNL_WITH_SYCL) || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void clear_memory_pool();
#endif

private:
    graph_mem_manager_t() : need_mem_check_(false) {}
    ~graph_mem_manager_t() {};

#ifdef DNNL_WITH_SYCL
    void *default_sycl_malloc(
            size_t n, size_t alignment, const void *dev, const void *ctx) {
        return malloc_device(n, *static_cast<const sycl::device *>(dev),
                *static_cast<const sycl::context *>(ctx));
    }

    void default_sycl_free(
            void *ptr, const void *dev, const void *context, void *event) {
        (void)(dev);
        if (event) {
            static_cast<sycl::event *>(const_cast<void *>(event))->wait();
        }
        free(ptr, *static_cast<const sycl::context *>(context));
    }
#endif

    bool need_mem_check_;
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    simple_memory_pool_t mem_pool_;
#endif
};

dnnl::graph::allocator &get_graph_allocator();

} // namespace graph

#endif
