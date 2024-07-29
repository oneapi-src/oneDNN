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

#ifndef MEMORY_POOL_HPP
#define MEMORY_POOL_HPP

#include <mutex>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_graph_ocl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

struct cpu_deletor {
    cpu_deletor() = default;
    void operator()(void *ptr) {
        if (ptr) free(ptr);
    }
};

#ifdef DNNL_WITH_SYCL
struct sycl_deletor {
    sycl_deletor() = delete;
    ::sycl::context ctx_;
    sycl_deletor(const ::sycl::context &ctx) : ctx_(ctx) {}
    void operator()(void *ptr) {
        if (ptr) ::sycl::free(ptr, ctx_);
    }
};

void *sycl_malloc_wrapper(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    return malloc_shared(size, *static_cast<const ::sycl::device *>(dev),
            *static_cast<const ::sycl::context *>(ctx));
}

void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
    // Device is not used in this example, but it may be useful for some users
    // application.
    UNUSED(device);
    // immediate synchronization here is for test purpose. For performance,
    // users may need to store the ptr and event and handle them separately
    if (event) {
        auto sycl_deps_ptr = static_cast<::sycl::event *>(event);
        sycl_deps_ptr->wait();
    }
    free(ptr, *static_cast<const ::sycl::context *>(context));
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#define OCL_CHECK(x) \
    do { \
        cl_int s = (x); \
        if (s != CL_SUCCESS) { \
            std::cout << "[" << __FILE__ << ":" << __LINE__ << "] '" << #x \
                      << "' failed (status code: " << s << ")." << std::endl; \
            exit(1); \
        } \
    } while (0)

static void *ocl_malloc_device(
        size_t size, size_t alignment, cl_device_id dev, cl_context ctx) {
    using F = void *(*)(cl_context, cl_device_id, cl_ulong *, size_t, cl_uint,
            cl_int *);
    if (size == 0) return nullptr;

    cl_platform_id platform;
    OCL_CHECK(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    const char *f_name = "clDeviceMemAllocINTEL";
    auto f = reinterpret_cast<F>(
            clGetExtensionFunctionAddressForPlatform(platform, f_name));
    cl_int err;
    void *p = f(ctx, dev, nullptr, size, static_cast<cl_uint>(alignment), &err);
    OCL_CHECK(err);
    return p;
}

static void ocl_free(
        void *ptr, cl_device_id dev, const cl_context ctx, cl_event event) {
    if (nullptr == ptr) return;
    using F = cl_int (*)(cl_context, void *);
    if (event) { OCL_CHECK(clWaitForEvents(1, &event)); }
    cl_platform_id platform;
    OCL_CHECK(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    const char *f_name = "clMemBlockingFreeINTEL";
    auto f = reinterpret_cast<F>(
            clGetExtensionFunctionAddressForPlatform(platform, f_name));
    OCL_CHECK(f(ctx, ptr));
}
#endif

// This memory pool is for benchdnn graph performance test. The clear and
// set_capacity functions aren't thread safe. The multi-threaded scenario is
// mainly used in Graph Compiler backend.
// The graph example memory pool is same as this. It's just for simplification.
// TODO: add some comments to clarify the design and interfaces
class simple_memory_pool_t {
public:
#if defined(DNNL_WITH_SYCL) || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#ifdef DNNL_WITH_SYCL
    void *allocate(
            size_t size, size_t alignment, const void *dev, const void *ctx)
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
            void *allocate(size_t size, size_t alignment, cl_device_id dev,
                    cl_context ctx)
#endif
    {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        // fake malloc for 0 size
        if (size == 0) return nullptr;

        void *ptr {nullptr};
        bool need_alloc_new_mm = true;
        // find alloc mm with same size
        const auto cnt = map_size_ptr_.count(size);
        if (cnt > 0) {
            const auto Iter = map_size_ptr_.equal_range(size);
            for (auto it = Iter.first; it != Iter.second; ++it) {
                // check if same size mm is free
                if (is_free_ptr_[it->second.get()]) {
                    ptr = it->second.get();
                    is_free_ptr_[ptr] = false;
                    need_alloc_new_mm = false;
                }
            }
        }
        if (need_alloc_new_mm) {
#ifdef DNNL_WITH_SYCL
            auto sh_ptr = std::shared_ptr<void> {
                    sycl_malloc_wrapper(size, alignment, dev, ctx),
                    sycl_deletor {*static_cast<const sycl::context *>(ctx)}};
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
            auto sh_ptr = std::shared_ptr<void> {
                    ocl_malloc_device(size, alignment, dev, ctx),
                    [ctx, dev](void *p) { ocl_free(p, dev, ctx, {}); }};
#endif
            ptr = sh_ptr.get();
            // record the map of mm size and its ptr for reuse
            map_size_ptr_.emplace(std::make_pair(size, sh_ptr));
            is_free_ptr_[ptr] = false;
        }
        return ptr;
    }
#endif

    void *allocate_host(size_t size, size_t alignment) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        if (size == 0) return nullptr;

        void *ptr {nullptr};
        bool need_alloc_new_mm = true;
        // find alloc mm with same size
        const auto cnt = map_size_ptr_.count(size);
        if (cnt > 0) {
            const auto Iter = map_size_ptr_.equal_range(size);
            for (auto it = Iter.first; it != Iter.second; ++it) {
                // check if same size mm is free
                if (is_free_ptr_[it->second.get()]) {
                    ptr = it->second.get();
                    is_free_ptr_[ptr] = false;
                    need_alloc_new_mm = false;
                }
            }
        }
        if (need_alloc_new_mm) {
            auto sh_ptr = std::shared_ptr<void> {malloc(size), cpu_deletor {}};
            ptr = sh_ptr.get();
            // record the map of mm size and its ptr for reuse
            map_size_ptr_.emplace(std::make_pair(size, sh_ptr));
            is_free_ptr_[ptr] = false;
        }
        return ptr;
    }

#ifdef DNNL_WITH_SYCL
    void deallocate(
            void *ptr, const void *device, const void *context, void *event) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        if (event) {
            auto sycl_deps_ptr = static_cast<::sycl::event *>(event);
            sycl_deps_ptr->wait();
        }
        is_free_ptr_[ptr] = true;
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void deallocate(
            void *ptr, cl_device_id dev, const cl_context ctx, cl_event event) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        if (event) { OCL_CHECK(clWaitForEvents(1, &event)); }
        is_free_ptr_[ptr] = true;
        return;
    }
#endif
    void deallocate_host(void *ptr) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        is_free_ptr_[ptr] = true;
        return;
    }
    void clear() {
        dnnl::graph::set_compiled_partition_cache_capacity(0);
        map_size_ptr_.clear();
        is_free_ptr_.clear();
    }

private:
    std::mutex pool_lock;
    std::unordered_multimap<size_t, std::shared_ptr<void>> map_size_ptr_;
    std::unordered_map<void *, bool> is_free_ptr_;
};

#endif
