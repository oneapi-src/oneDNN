/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef INTERFACE_ALLOCATOR_HPP
#define INTERFACE_ALLOCATOR_HPP

#include <cstdlib>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.h"

#include "interface/c_types_map.hpp"

#include "utils/id.hpp"
#include "utils/rw_mutex.hpp"
#include "utils/utils.hpp"
#include "utils/verbose.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

struct dnnl_graph_allocator final : public dnnl::graph::impl::utils::id_t {
public:
    /// Allocator attributes
    struct attribute_t {
        friend struct dnnl_graph_allocator;

        dnnl::graph::impl::allocator_attr_t data;

        /// Default constructor for an uninitialized attribute
        attribute_t() {
            data.type = dnnl::graph::impl::allocator_lifetime::persistent;
            data.alignment = 0;
        }

        attribute_t(dnnl::graph::impl::allocator_lifetime_t type,
                size_t alignment) {
            data.type = type;
            data.alignment = alignment;
        }

        /// Copy constructor
        attribute_t(const attribute_t &other) = default;

        /// Assign operator
        attribute_t &operator=(const attribute_t &other) = default;
    };

    struct mem_info_t {
        mem_info_t(size_t size, dnnl_graph_allocator_lifetime_t type)
            : size_(size), type_(type) {}
        size_t size_;
        dnnl_graph_allocator_lifetime_t type_;
    };

    struct monitor_t {
    private:
        static std::unordered_map<const dnnl_graph_allocator *, size_t>
                persist_mem_;
        static std::unordered_map<const dnnl_graph_allocator *,
                std::unordered_map<const void *, mem_info_t>>
                persist_mem_infos_;

        thread_local static std::unordered_map<const dnnl_graph_allocator *,
                size_t>
                temp_mem_;
        thread_local static std::unordered_map<const dnnl_graph_allocator *,
                size_t>
                peak_temp_mem_;
        thread_local static std::unordered_map<const dnnl_graph_allocator *,
                std::unordered_map<const void *, mem_info_t>>
                temp_mem_infos_;

        // Since the memory operation will be performaed from multiple threads,
        // so we use the rw lock to guarantee the thread safety of the global
        // persistent memory monitoring.
        static dnnl::graph::impl::utils::rw_mutex_t rw_mutex_;

    public:
        static void record_allocate(const dnnl_graph_allocator *alloc,
                const void *buf, size_t size,
                const dnnl_graph_allocator::attribute_t &attr);

        static void record_deallocate(
                const dnnl_graph_allocator *alloc, const void *buf);

        static void reset_peak_temp_memory(const dnnl_graph_allocator *alloc);

        static size_t get_peak_temp_memory(const dnnl_graph_allocator *alloc);

        static size_t get_total_persist_memory(
                const dnnl_graph_allocator *alloc);
    };

    dnnl_graph_allocator() = default;

    dnnl_graph_allocator(dnnl_graph_cpu_allocate_f cpu_malloc,
            dnnl_graph_cpu_deallocate_f cpu_free)
        : cpu_malloc_ {cpu_malloc}, cpu_free_ {cpu_free} {}

#if DNNL_GRAPH_WITH_SYCL
    dnnl_graph_allocator(dnnl_graph_sycl_allocate_f sycl_malloc,
            dnnl_graph_sycl_deallocate_f sycl_free)
        : sycl_malloc_(sycl_malloc), sycl_free_(sycl_free) {}
#endif

    void *allocate(size_t n, attribute_t attr = {}) const {
        void *buffer = cpu_malloc_(n, attr.data);
#ifndef NDEBUG
        monitor_t::record_allocate(this, buffer, n, attr);
#endif
        return buffer;
    }

#if DNNL_GRAPH_WITH_SYCL
    void *allocate(size_t n, const cl::sycl::device &dev,
            const cl::sycl::context &ctx, attribute_t attr) const {
        void *buffer = sycl_malloc_(n, static_cast<const void *>(&dev),
                static_cast<const void *>(&ctx), attr.data);
#ifndef NDEBUG
        monitor_t::record_allocate(this, buffer, n, attr);
#endif
        return buffer;
    }
#endif

    template <typename T>
    T *allocate(size_t num_elem, attribute_t attr = {}) {
        T *buffer = static_cast<T *>(
                cpu_malloc_(num_elem * sizeof(T), attr.data));
#ifndef NDEBUG
        monitor_t::record_allocate(this, buffer, num_elem * sizeof(T), attr);
#endif
        return buffer;
    }

#if DNNL_GRAPH_WITH_SYCL
    template <typename T>
    T *allocate(size_t num_elem, const cl::sycl::device &dev,
            const cl::sycl::context &ctx, attribute_t attr = {}) {
        T *buffer = static_cast<T *>(sycl_malloc_(num_elem * sizeof(T),
                static_cast<const void *>(&dev),
                static_cast<const void *>(&ctx), attr.data));
#ifndef NDEBUG
        monitor_t::record_allocate(
                this, (void *)buffer, num_elem * sizeof(T), attr);
#endif
        return buffer;
    }
#endif

    void deallocate(void *buffer) const {
        if (buffer) {
            cpu_free_(buffer);
#ifndef NDEBUG
            monitor_t::record_deallocate(this, buffer);
#endif
        }
    }

#if DNNL_GRAPH_WITH_SYCL
    void deallocate(void *buffer, const cl::sycl::context &ctx) const {
        if (buffer) {
            sycl_free_(buffer, static_cast<const void *>(&ctx));
#ifndef NDEBUG
            monitor_t::record_deallocate(this, buffer);
#endif
        }
    }
#endif

private:
    dnnl_graph_cpu_allocate_f cpu_malloc_ {
            dnnl::graph::impl::utils::cpu_allocator_t::malloc};
    dnnl_graph_cpu_deallocate_f cpu_free_ {
            dnnl::graph::impl::utils::cpu_allocator_t::free};

#if DNNL_GRAPH_WITH_SYCL
    dnnl_graph_sycl_allocate_f sycl_malloc_ {
            dnnl::graph::impl::utils::sycl_allocator_t::malloc};
    dnnl_graph_sycl_deallocate_f sycl_free_ {
            dnnl::graph::impl::utils::sycl_allocator_t::free};
#endif
};

#endif
