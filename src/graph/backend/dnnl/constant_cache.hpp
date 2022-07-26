/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#include "graph/utils/rw_mutex.hpp"

#include "graph/backend/dnnl/common.hpp"

#include "oneapi/dnnl/dnnl.hpp"

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
        const_cast<allocator_t *>(alc)->retain();
    }

    ~constant_buffer_t() {
#ifdef DNNL_WITH_SYCL
        dnnl_allocator_t::free(data_, p_engine_, alc_, {});
#else
        dnnl_allocator_t::free(data_, p_engine_, alc_);
#endif
        const_cast<allocator_t *>(alc_)->release();
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

    constant_cache_t() = default;

    status_t set_capacity(size_t capacity);
    size_t get_capacity() const;
    value_t get_or_add(const key_t &key, const value_t &value);
    void remove_if_exist(const key_t &key);

private:
    void evict(size_t n) const;
    value_t get(const key_t &key);
    void add(const key_t &key, const value_t &constant);
    size_t get_size() const;

    void lock_read() { rw_mutex_.lock_read(); }
    void lock_write() { rw_mutex_.lock_write(); }
    void unlock_read() { rw_mutex_.unlock_read(); }
    void unlock_write() { rw_mutex_.unlock_write(); }

    constant_cache_t(const constant_cache_t &other) = delete;
    constant_cache_t &operator=(const constant_cache_t &other) = delete;

    struct timed_entry_t {
        value_t value_;
        std::atomic<size_t> timestamp_;
        timed_entry_t(const value_t &value, size_t timestamp)
            : value_(value), timestamp_(timestamp) {}
    };

    // Each entry in the cache has a corresponding key and timestamp.
    // NOTE: pairs that contain atomics cannot be stored in an unordered_map *as
    // an element*, since it invokes the copy constructor of std::atomic, which
    // is deleted.
    static std::unordered_map<key_t, timed_entry_t> constant_map_;
    static graph::utils::rw_mutex_t rw_mutex_;
    size_t capacity_ = std::numeric_limits<size_t>::max();
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
