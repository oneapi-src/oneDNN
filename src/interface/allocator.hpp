/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <atomic>
#include <cstdlib>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.h"

#include "interface/c_types_map.hpp"

#include "utils/allocator.hpp"
#include "utils/id.hpp"
#include "utils/rw_mutex.hpp"
#include "utils/utils.hpp"
#include "utils/verbose.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

struct dnnl_graph_allocator final : public dnnl::graph::impl::utils::id_t {
private:
    // Make constructor and destructor private, so that users can only create
    // and destroy allocator through the public static creator and release
    // method.
    dnnl_graph_allocator() = default;

    dnnl_graph_allocator(dnnl_graph_cpu_allocate_f cpu_malloc,
            dnnl_graph_cpu_deallocate_f cpu_free)
        : cpu_malloc_ {cpu_malloc}, cpu_free_ {cpu_free} {}

#ifdef DNNL_GRAPH_WITH_SYCL
    dnnl_graph_allocator(dnnl_graph_sycl_allocate_f sycl_malloc,
            dnnl_graph_sycl_deallocate_f sycl_free)
        : sycl_malloc_(sycl_malloc), sycl_free_(sycl_free) {}
#endif

    ~dnnl_graph_allocator() = default;

public:
    // CAVEAT: The invocation number of release() should be exactly equal to the
    // added invocation number of create() and retain(). Otherwise, error will
    // occur!

    // This function increments the reference count
    void retain() {
        counter_.fetch_add(1, std::memory_order::memory_order_relaxed);
    }

    // This function decrements the reference count. If the reference count is
    // decremented to zero, the object will be destroyed.
    void release() {
        if (counter_.fetch_sub(1, std::memory_order_relaxed) == 1) {
            delete this;
        }
    }

    // The following three static functions are used to create an allocator
    // object. The initial reference count of created object is 1.
    static dnnl_graph_allocator *create() {
        return new dnnl_graph_allocator {};
    }

    static dnnl_graph_allocator *create(dnnl_graph_cpu_allocate_f cpu_malloc,
            dnnl_graph_cpu_deallocate_f cpu_free) {
        return new dnnl_graph_allocator {cpu_malloc, cpu_free};
    }

    static dnnl_graph_allocator *create(const dnnl_graph_allocator *alloc) {
#ifdef DNNL_GRAPH_WITH_SYCL
        return new dnnl_graph_allocator {
                alloc->sycl_malloc_, alloc->sycl_free_};
#else
        return new dnnl_graph_allocator {alloc->cpu_malloc_, alloc->cpu_free_};
#endif
    }

#ifdef DNNL_GRAPH_WITH_SYCL
    static dnnl_graph_allocator *create(dnnl_graph_sycl_allocate_f sycl_malloc,
            dnnl_graph_sycl_deallocate_f sycl_free) {
        return new dnnl_graph_allocator {sycl_malloc, sycl_free};
    }
#endif

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

        static std::unordered_map<std::thread::id,
                std::unordered_map<const dnnl_graph_allocator *, size_t>>
                temp_mem_;
        static std::unordered_map<std::thread::id,
                std::unordered_map<const dnnl_graph_allocator *, size_t>>
                peak_temp_mem_;
        static std::unordered_map<std::thread::id,
                std::unordered_map<const dnnl_graph_allocator *,
                        std::unordered_map<const void *, mem_info_t>>>
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

        static void lock_write();
        static void unlock_write();
    };

    void *allocate(size_t n, attribute_t attr = {}) const {
#ifndef NDEBUG
        monitor_t::lock_write();
        void *buffer = cpu_malloc_(n, attr.data);
        monitor_t::record_allocate(this, buffer, n, attr);
        monitor_t::unlock_write();
#else
        void *buffer = cpu_malloc_(n, attr.data);
#endif
        return buffer;
    }

#ifdef DNNL_GRAPH_WITH_SYCL
    void *allocate(size_t n, const ::sycl::device &dev,
            const ::sycl::context &ctx, attribute_t attr) const {
#ifndef NDEBUG
        monitor_t::lock_write();
        void *buffer = sycl_malloc_(n, static_cast<const void *>(&dev),
                static_cast<const void *>(&ctx), attr.data);
        monitor_t::record_allocate(this, buffer, n, attr);
        monitor_t::unlock_write();
#else
        void *buffer = sycl_malloc_(n, static_cast<const void *>(&dev),
                static_cast<const void *>(&ctx), attr.data);
#endif
        return buffer;
    }
#endif

    template <typename T>
    T *allocate(size_t num_elem, attribute_t attr = {}) {
#ifndef NDEBUG
        monitor_t::lock_write();
        T *buffer = static_cast<T *>(
                cpu_malloc_(num_elem * sizeof(T), attr.data));
        monitor_t::record_allocate(this, buffer, num_elem * sizeof(T), attr);
        monitor_t::unlock_write();
#else
        T *buffer = static_cast<T *>(
                cpu_malloc_(num_elem * sizeof(T), attr.data));
#endif
        return buffer;
    }

#ifdef DNNL_GRAPH_WITH_SYCL
    template <typename T>
    T *allocate(size_t num_elem, const ::sycl::device &dev,
            const ::sycl::context &ctx, attribute_t attr = {}) {
#ifndef NDEBUG
        monitor_t::lock_write();
        T *buffer = static_cast<T *>(sycl_malloc_(num_elem * sizeof(T),
                static_cast<const void *>(&dev),
                static_cast<const void *>(&ctx), attr.data));
        monitor_t::record_allocate(
                this, (void *)buffer, num_elem * sizeof(T), attr);
        monitor_t::unlock_write();
#else
        T *buffer = static_cast<T *>(sycl_malloc_(num_elem * sizeof(T),
                static_cast<const void *>(&dev),
                static_cast<const void *>(&ctx), attr.data));
#endif
        return buffer;
    }
#endif

    void deallocate(void *buffer) const {
        if (buffer) {
#ifndef NDEBUG
            monitor_t::lock_write();
            monitor_t::record_deallocate(this, buffer);
            cpu_free_(buffer);
            monitor_t::unlock_write();
#else
            cpu_free_(buffer);
#endif
        }
    }

#ifdef DNNL_GRAPH_WITH_SYCL
    void deallocate(void *buffer, const ::sycl::device &dev,
            const ::sycl::context &ctx, ::sycl::event deps) const {
        if (buffer) {
#ifndef NDEBUG
            monitor_t::lock_write();
            monitor_t::record_deallocate(this, buffer);
            sycl_free_(buffer, static_cast<const void *>(&dev),
                    static_cast<const void *>(&ctx),
                    static_cast<void *>(&deps));
            monitor_t::unlock_write();
#else
            sycl_free_(buffer, static_cast<const void *>(&dev),
                    static_cast<const void *>(&ctx),
                    static_cast<void *>(&deps));
#endif
        }
    }
#endif

private:
    dnnl_graph_cpu_allocate_f cpu_malloc_ {
            dnnl::graph::impl::utils::cpu_allocator_t::malloc};
    dnnl_graph_cpu_deallocate_f cpu_free_ {
            dnnl::graph::impl::utils::cpu_allocator_t::free};

#ifdef DNNL_GRAPH_WITH_SYCL
    dnnl_graph_sycl_allocate_f sycl_malloc_ {
            dnnl::graph::impl::utils::sycl_allocator_t::malloc};
    dnnl_graph_sycl_deallocate_f sycl_free_ {
            dnnl::graph::impl::utils::sycl_allocator_t::free};
#endif

    std::atomic<int32_t> counter_ {1}; // align to oneDNN to use int32_t type
};

#endif
