/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_DNNL_CONSTANT_CACHE_HPP
#define GRAPH_BACKEND_DNNL_CONSTANT_CACHE_HPP

#include <atomic>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "common/engine.hpp"
#include "common/rw_mutex.hpp"
#include "common/utils.hpp"

#include "graph/backend/dnnl/common.hpp"

#include "oneapi/dnnl/dnnl.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct constant_buffer_t {
    constant_buffer_t(
            size_t size, const dnnl::engine &p_engine, const allocator_t *alc)
        : size_(size), p_engine_(p_engine), alc_(alc) {
        data_ = dnnl_allocator_t::malloc(
                size, p_engine, alc, allocator_t::mem_type_t::persistent);
        const_cast<engine_t *>(p_engine_.get())->retain();
    }

    ~constant_buffer_t() {
#ifdef DNNL_WITH_SYCL
        dnnl_allocator_t::free(data_, p_engine_, alc_, {});
#else
        dnnl_allocator_t::free(data_, p_engine_, alc_);
#endif
        const_cast<engine_t *>(p_engine_.get())->release();
    }

    template <typename T>
    T *data() {
        return static_cast<T *>(data_);
    }

    size_t size() const { return size_; }

private:
    void *data_;
    size_t size_;
    const dnnl::engine p_engine_;
    const allocator_t *alc_;
};

struct constant_cache_t {
    using key_t = size_t;
    using cached_t = std::shared_ptr<constant_buffer_t>;
    using value_t = std::shared_future<cached_t>;

    // This function increments the reference count
    void retain() { counter_.fetch_add(1, std::memory_order_relaxed); }

    void release() {
        if (counter_.fetch_sub(1, std::memory_order_relaxed) == 1) {
            delete this;
        }
    }

    ~constant_cache_t() {
        if (constant_map().empty()) return;

#if defined(_WIN32) && defined(DNNL_WITH_SYCL)
        // The library unloading issue affects only DPCPP runtimes on Windows when
        // DNNL_GRAPH_ENABLE_COMPILED_PARTITION_CACHE is ON. The ntdll.dll library
        // is located in system32 therefore setting additional environment is not
        // required.
        HMODULE handle = LoadLibraryExA(
                "ntdll.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!handle) {
            constant_map_.release();
            return;
        }

        // RtlDllShutdownInProgress returns TRUE if the whole process terminates and
        // FALSE if DLL is being unloaded dynamically or if it’s called from an
        // executable.
        auto f = reinterpret_cast<BOOLEAN (*)(void)>(
                GetProcAddress(handle, "RtlDllShutdownInProgress"));
        if (!f) {
            auto ret = FreeLibrary(handle);
            assert(ret);
            MAYBE_UNUSED(ret);
            constant_map_.release();
            return;
        }

        bool is_process_termination_in_progress = f();

        auto ret = FreeLibrary(handle);
        assert(ret);
        MAYBE_UNUSED(ret);

        if (is_process_termination_in_progress) {
            // The whole process is being terminated hence destroying content of the
            // primitive cache cannot be done safely. Theoretically, we can check
            // all entries and remove those that are not affected e.g. native CPU.
            // We can do this after switching to use dnnl engine.
            for (auto it = constant_map().begin();
                    it != constant_map().end();) {
#ifdef DNNL_WITH_SYCL
                ++it;
#else
                it = constant_map().erase(it);
#endif
            }
            constant_map_.release();
        } else {
            // Three scenarios possible:
            // 1. oneDNN Graph is being dynamically unloaded
            // 2. Another dynamic library that contains statically linked oneDNN
            //    Graph is dynamically unloaded
            // 3. oneDNN Graph is statically linked in an executable which is done
            //    and now the process terminates In all these scenarios content of
            //    the primitive cache can be safely destroyed.
            constant_map_.reset();
        }
#else
        // Always destroy the content of the primitive cache for non-Windows OSes,
        // and non-sycl and non-ocl runtimes because there is no a problem with
        // library unloading order in such cases.
        constant_map_.reset();
#endif
    }

    static constant_cache_t *get_global_constant_cache() {
        static auto global_cache
                = std::shared_ptr<constant_cache_t>(new constant_cache_t {},
                        [](constant_cache_t *ptr) { return ptr->release(); });
        return global_cache.get();
    }

    status_t set_capacity(size_t capacity);
    size_t get_capacity();
    value_t get_or_add(const key_t &key, const value_t &value);
    void remove_if_exist(const key_t &key);

    size_t get_size() const;

private:
    constant_cache_t() : counter_(1) {
        constant_map_ = impl::utils::make_unique<
                std::unordered_map<key_t, timed_entry_t>>();
    }

    void evict(size_t n);
    value_t get(const key_t &key);
    void add(const key_t &key, const value_t &constant);

    void lock_read() { rw_mutex_.lock_read(); }
    void lock_write() { rw_mutex_.lock_write(); }
    void unlock_read() { rw_mutex_.unlock_read(); }
    void unlock_write() { rw_mutex_.unlock_write(); }

    constant_cache_t &operator=(const constant_cache_t &other) = delete;

    struct timed_entry_t {
        value_t value_;
        std::atomic<size_t> timestamp_;
        timed_entry_t(const value_t &value, size_t timestamp)
            : value_(value), timestamp_(timestamp) {}
    };

    std::unordered_map<key_t, timed_entry_t> &constant_map() {
        return *constant_map_;
    }

    const std::unordered_map<key_t, timed_entry_t> &constant_map() const {
        return *constant_map_;
    }

    // Each entry in the cache has a corresponding key and timestamp.
    // NOTE: pairs that contain atomics cannot be stored in an unordered_map *as
    // an element*, since it invokes the copy constructor of std::atomic, which
    // is deleted.
    std::unique_ptr<std::unordered_map<key_t, timed_entry_t>> constant_map_;
    impl::utils::rw_mutex_t rw_mutex_;
    size_t capacity_ = std::numeric_limits<size_t>::max();

    std::atomic<int32_t> counter_;
};

constant_cache_t &get_global_constant_cache();

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
