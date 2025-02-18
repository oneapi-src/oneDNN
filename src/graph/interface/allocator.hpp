/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef GRAPH_INTERFACE_ALLOCATOR_HPP
#define GRAPH_INTERFACE_ALLOCATOR_HPP

#include "oneapi/dnnl/dnnl_graph.h"

#include "graph/interface/c_types_map.hpp"

#include "graph/utils/alloc.hpp"
#include "graph/utils/verbose.hpp"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "graph/utils/ocl_check.hpp"
#include "oneapi/dnnl/dnnl_graph_ocl.h"
#endif

struct dnnl_graph_allocator {
public:
    dnnl_graph_allocator() = default;

    dnnl_graph_allocator(dnnl_graph_host_allocate_f host_malloc,
            dnnl_graph_host_deallocate_f host_free)
        : host_malloc_(host_malloc), host_free_(host_free) {}

#ifdef DNNL_WITH_SYCL
    dnnl_graph_allocator(dnnl_graph_sycl_allocate_f sycl_malloc,
            dnnl_graph_sycl_deallocate_f sycl_free)
        : sycl_malloc_(sycl_malloc), sycl_free_(sycl_free) {}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    dnnl_graph_allocator(dnnl_graph_ocl_allocate_f ocl_malloc,
            dnnl_graph_ocl_deallocate_f ocl_free)
        : ocl_malloc_(ocl_malloc), ocl_free_(ocl_free) {}
#endif

    enum class mem_type_t {
        persistent = 0,
        output = 1,
        temp = 2,
    };

    /// Allocator attributes
    struct mem_attr_t {
        mem_type_t type_;
        size_t alignment_;

        /// Default constructor for an uninitialized attribute
        mem_attr_t() : type_(mem_type_t::persistent), alignment_(0) {}

        mem_attr_t(mem_type_t type, size_t alignment)
            : type_(type), alignment_(alignment) {}
    };

    void *allocate(size_t size, mem_attr_t attr = {}) const {
        void *buffer = host_malloc_(size, attr.alignment_);
        return buffer;
    }

#ifdef DNNL_WITH_SYCL
    void *allocate(size_t size, const ::sycl::device &dev,
            const ::sycl::context &ctx, mem_attr_t attr = {}) const {
        void *buffer = sycl_malloc_(size, attr.alignment_,
                static_cast<const void *>(&dev),
                static_cast<const void *>(&ctx));
        return buffer;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void *allocate(size_t size, cl_device_id dev, cl_context ctx,
            mem_attr_t attr = {}) const {
        void *buffer = ocl_malloc_(size, attr.alignment_, dev, ctx);
        return buffer;
    }
#endif

    template <typename T>
    T *allocate(size_t nelem, mem_attr_t attr = {}) {
        const size_t size = nelem * sizeof(T);
        void *buffer = allocate(size, attr);
        return reinterpret_cast<T *>(buffer);
    }

#ifdef DNNL_WITH_SYCL
    template <typename T>
    T *allocate(size_t nelem, const ::sycl::device &dev,
            const ::sycl::context &ctx, mem_attr_t attr = {}) {
        const size_t size = nelem * sizeof(T);
        void *buffer = allocate(size, dev, ctx, attr);
        return reinterpret_cast<T *>(buffer);
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    template <typename T>
    T *allocate(size_t nelem, cl_device_id dev, cl_context ctx,
            mem_attr_t attr = {}) {
        const size_t size = nelem * sizeof(T);
        void *buffer = allocate(size, dev, ctx, attr);
        return reinterpret_cast<T *>(buffer);
    }
#endif

    void deallocate(void *buffer) const {
        if (buffer) { host_free_(buffer); }
    }

#ifdef DNNL_WITH_SYCL
    void deallocate(void *buffer, const ::sycl::device &dev,
            const ::sycl::context &ctx, ::sycl::event deps) const {
        if (buffer) {
            sycl_free_(buffer, static_cast<const void *>(&dev),
                    static_cast<const void *>(&ctx),
                    static_cast<void *>(&deps));
        }
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void deallocate(void *buffer, cl_device_id dev, cl_context ctx,
            cl_event deps) const {
        if (buffer) {
            ocl_free_(buffer, dev, ctx, deps);
            buffer = nullptr;
        }
    }
#endif

private:
    dnnl_graph_host_allocate_f host_malloc_ {
            dnnl::impl::graph::utils::cpu_allocator_t::malloc};
    dnnl_graph_host_deallocate_f host_free_ {
            dnnl::impl::graph::utils::cpu_allocator_t::free};

#ifdef DNNL_WITH_SYCL
    dnnl_graph_sycl_allocate_f sycl_malloc_ {
            dnnl::impl::graph::utils::sycl_allocator_t::malloc};
    dnnl_graph_sycl_deallocate_f sycl_free_ {
            dnnl::impl::graph::utils::sycl_allocator_t::free};
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    // By default, use the malloc and free functions provided by the library.
    dnnl_graph_ocl_allocate_f ocl_malloc_ {
            dnnl::impl::graph::utils::ocl_allocator_t::malloc};
    dnnl_graph_ocl_deallocate_f ocl_free_ {
            dnnl::impl::graph::utils::ocl_allocator_t::free};
#endif
};

#endif
