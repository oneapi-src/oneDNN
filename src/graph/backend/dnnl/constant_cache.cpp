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

#include <algorithm>
#include <unordered_map>

#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/constant_cache.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

using key_t = constant_cache_t::key_t;
using value_t = constant_cache_t::value_t;

static size_t get_timestamp() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}

status_t constant_cache_t::set_capacity(size_t capacity) {
    lock_write();
    capacity_ = static_cast<size_t>(capacity);
    if (get_size() > capacity_) {
        // Evict excess buffers
        size_t excess_size = get_size() - capacity_;
        evict(excess_size);
    }
    unlock_write();
    return status::success;
}

size_t constant_cache_t::get_capacity() {
    impl::utils::lock_read_t lock_r(rw_mutex_);
    return capacity_;
}

value_t constant_cache_t::get_or_add(const key_t &key, const value_t &value) {
    // 1. Section with shared access (read lock)
    lock_read();
    // Check if the cache is enabled.
    if (capacity_ == 0) {
        unlock_read();
        return value_t();
    }
    // Check if the requested entry is present in the cache (likely cache_hit)
    auto e = get(key);
    if (e.valid()) {
        unlock_read();
        return e;
    }

    unlock_read();

    // 2. Section with exclusive access (write lock).
    // In a multithreaded scenario, in the context of one thread the cache
    // may have changed by another thread between releasing the read lock and
    // acquiring the write lock (a.k.a. ABA problem), therefore additional
    // checks have to be performed for correctness.
    // Double check the capacity due to possible race condition
    lock_write();
    if (capacity_ == 0) {
        unlock_write();
        return value_t();
    }

    // Double check if the requested entry is present in the cache (unlikely
    // cache_hit).
    e = get(key);
    if (!e.valid()) {
        // If the entry is missing in the cache then add it (cache_miss)
        add(key, value);
    }
    unlock_write();
    return e;
}

void constant_cache_t::remove_if_exist(const key_t &key) {
    lock_write();
    if (constant_map().count(key) == 0) {
        unlock_write();
    } else {
        constant_map().erase(key);
        unlock_write();
    }
}

// Get the total size of all cached buffers
size_t constant_cache_t::get_size() const {
    size_t total_size = 0;
    for (const auto &pair : constant_map()) {
        total_size += pair.second.value_.get()->size();
    }
    return total_size;
}

void constant_cache_t::add(const key_t &key, const value_t &constant) {
    size_t current_size = get_size();
    if (current_size >= capacity_) {
        // FIXME(qun) because we can't know the concrete size of the constant,
        // so we could only guarantee that the total size before adding  not
        // beyond capacity here.
        evict(current_size - capacity_);
    }

    size_t timestamp = get_timestamp();

    auto res = constant_map().emplace(std::piecewise_construct,
            std::forward_as_tuple(key),
            std::forward_as_tuple(constant, timestamp));
    UNUSED(res);
    assert(res.second);
}

value_t constant_cache_t::get(const key_t &key) {
    auto it = constant_map().find(key);
    if (it == constant_map().end()) return value_t();

    size_t timestamp = get_timestamp();
    it->second.timestamp_.store(timestamp);
    // Return the entry
    return it->second.value_;
}

// Evict n size of cached buffers
void constant_cache_t::evict(size_t n) {
    using v_t = std::unordered_map<key_t, timed_entry_t>::value_type;
    if (n == get_size()) {
        constant_map().clear();
        return;
    }

    size_t evicted_size = 0;
    while (evicted_size < n) {
        // Find the smallest timestamp
        auto it = std::min_element(constant_map().begin(), constant_map().end(),
                [&](const v_t &left, const v_t &right) {
                    // By default, load() and operator T use sequentially
                    // consistent memory ordering, which enforces writing the
                    // timestamps into registers in the same exact order they
                    // are read from the CPU cache line. Since eviction is
                    // performed under a write lock, this order is not
                    // important, therefore we can safely use the weakest memory
                    // ordering (relaxed).
                    return left.second.timestamp_.load(
                                   std::memory_order_relaxed)
                            < right.second.timestamp_.load(
                                    std::memory_order_relaxed);
                });
        evicted_size += it->second.value_.get()->size();
        auto res = constant_map().erase(it->first);
        UNUSED(res);
        assert(res);
    }
}

constant_cache_t &get_global_constant_cache() {
    return *constant_cache_t::get_global_constant_cache();
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
