/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include <chrono>
#include <vector>
#include <unordered_map>

#include "interface/c_types_map.hpp"
#include "interface/partition.hpp"
#include "interface/partition_cache.hpp"

#include "utils/rw_mutex.hpp"

namespace dnnl {
namespace graph {
namespace impl {

namespace {

size_t get_timestamp() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}

std::vector<const logical_tensor_t *> get_raw_ptrs(
        const std::vector<logical_tensor_t> &ports) {
    std::vector<const logical_tensor_t *> ret(ports.size(), nullptr);
    std::transform(ports.begin(), ports.end(), ret.begin(),
            [](const logical_tensor_t &lt) { return &lt; });
    return ret;
}

} // namespace

compiled_partition_cache_t &compiled_partition_cache() {
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    static const int capacity = utils::getenv_int(
            "DNNL_GRAPH_COMPILED_PARTITION_CACHE_CAPACITY", 1024);
#else
    static const int capacity = 0;
#endif
    static lru_compiled_partition_cache_t cache(capacity);
    return cache;
}

status_t get_compiled_partition_cache_size(int *size) {
    if (size == nullptr) return status::invalid_argument;
    *size = 0;
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    *size = compiled_partition_cache().get_size();
#endif
    return status::success;
}

bool is_partition_in_cache(const partition_t *partition,
        const std::vector<const logical_tensor_t *> &ins,
        const std::vector<const logical_tensor_t *> &outs) {
    partition_hashing::key_t key(partition, ins, outs);
    return bool(compiled_partition_cache().get_partition(key));
}

bool is_compiled_partition_in_cache(const compiled_partition_t *cp) {
    return is_partition_in_cache(&(cp->src_partition()),
            get_raw_ptrs(cp->get_inputs()), get_raw_ptrs(cp->get_outputs()));
}

status_t lru_compiled_partition_cache_t::set_capacity(int capacity) {
    utils::lock_write_t lock_w(rw_mutex());
    capacity_ = static_cast<size_t>(capacity);
    if (cache_mapper_.size() > capacity_) {
        // Evict excess entries
        size_t n_excess_entries = cache_mapper_.size() - capacity_;
        evict(n_excess_entries);
    }
    return status::success;
}

int lru_compiled_partition_cache_t::get_capacity() const {
    utils::lock_read_t lock_r(rw_mutex());
    return static_cast<int>(capacity_);
}

int lru_compiled_partition_cache_t::get_size() const {
    utils::lock_read_t lock_r(rw_mutex());
    return static_cast<int>(cache_mapper_.size());
}

lru_compiled_partition_cache_t::value_t
lru_compiled_partition_cache_t::get_or_add(
        const key_t &key, const value_t &value) {
    // 1. Section with shared access (read lock)
    lock_read();
    // Check if the cache is enabled.
    if (capacity_ == 0) {
        unlock_read();
        return value_t();
    }
    // Check if the requested entry is present in the cache (likely cache hit)
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
    // Double check the capacity due to possible race condition.
    lock_write();
    if (capacity_ == 0) {
        unlock_write();
        return value_t();
    }

    // Double check if the requested entry is present in the cache (unlikely
    // cache hit).
    e = get(key);
    if (!e.valid()) {
        // If the entry is missing in the cache then add it (cache_miss)
        add(key, value);
    }
    unlock_write();
    return e;
}

void lru_compiled_partition_cache_t::add(
        const key_t &key, const value_t &value) {
    // std::list::size() method has linear complexity. Check the primitive cache
    // size using std::unordered_map::size();
    if (cache_mapper_.size() == capacity_) {
        // Evict the least recently used entry
        evict(1);
    }

    size_t timestamp = get_timestamp();

    auto res = cache_mapper_.emplace(std::piecewise_construct,
            std::forward_as_tuple(key),
            std::forward_as_tuple(value, timestamp));
    UNUSED(res);
    assert(res.second);
}

lru_compiled_partition_cache_t::value_t lru_compiled_partition_cache_t::get(
        const key_t &key) {
    auto it = cache_mapper_.find(key);
    if (it == cache_mapper_.end()) return value_t();

    size_t timestamp = get_timestamp();
    it->second.timestamp_.store(timestamp);
    // Return the entry
    return it->second.value_;
}

const partition_t *lru_compiled_partition_cache_t::get_partition(
        const key_t &key) {
    lock_read();
    auto e = get(key);
    unlock_read();

    if (e.valid()) return &(e.get().compiled_partition->src_partition());
    return nullptr;
}

void lru_compiled_partition_cache_t::remove_if_invalidated(const key_t &key) {
    lock_write();
    auto it = cache_mapper_.find(key);
    if (it == cache_mapper_.end()) {
        // The entry has been already evicted at this point
        unlock_write();
        return;
    }

    const auto &value = it->second.value_;
    if (value.get().compiled_partition) {
        // If the entry is not invalidated
        unlock_write();
        return;
    }

    // Remove the invalidated entry
    cache_mapper_.erase(it);
    unlock_write();
}

void lru_compiled_partition_cache_t::update_entry(const key_t &key,
        const partition_t *partition,
        const std::vector<const logical_tensor_t *> &ins,
        const std::vector<const logical_tensor_t *> &outs) {
    utils::lock_write_t lock_w(rw_mutex());
    auto it = cache_mapper_.find(key);

    // There is nothing to do in two cases:
    // 1. The requested entry is not in the cache because it has been evicted
    //    by another thread
    // 2. After the requested entry had been evicted it was inserted again
    //    by another thread
    if (it == cache_mapper_.end() || it->first.thread_id() != key.thread_id())
        return;

    // Update key in cache_mapper_
    it->first.partition_id_ = static_cast<size_t>(partition->id());
    it->first.ops_ = partition_hashing::get_raw_ptrs(partition->get_ops());
    it->first.ins_.clear();
    it->first.outs_.clear();
    it->first.ins_.reserve(ins.size());
    it->first.outs_.reserve(outs.size());
    for (auto &in : ins) {
        it->first.ins_.emplace(*in);
    }
    for (auto &out : outs) {
        it->first.outs_.emplace(*out);
    }
}

void lru_compiled_partition_cache_t::evict(size_t n) {
    if (n == capacity_) {
        cache_mapper_.clear();
        return;
    }

    for (size_t e = 0; e < n; e++) {
        // Find the smallest timestamp
        // TODO(zixuanwe): revisit the eviction algorithm due to O(n)
        // complexity, E.g. maybe evict multiple entries at once.
        auto it = std::min_element(cache_mapper_.begin(), cache_mapper_.end(),
                [&](const decltype(cache_mapper_)::value_type &left,
                        const decltype(cache_mapper_)::value_type &right) {
                    // By default, load() and operator T use sequentially
                    // consistent memory ordering, which enforces writing the
                    // timestamps into registers in the same exact order they
                    // are read from the CPU cache line. Since eviction is
                    // performed under a write lock, this order is not
                    // important, therefore we can safely use the weakest memory
                    // ordering (relaxed).
                    return left.second.timestamp_.load(
                                   std::memory_order::memory_order_relaxed)
                            < right.second.timestamp_.load(
                                    std::memory_order::memory_order_relaxed);
                });
        auto res = cache_mapper_.erase(it->first);
        UNUSED(res);
        assert(res);
    }
}

} // namespace impl
} // namespace graph
} // namespace dnnl

// API
dnnl::graph::impl::status_t dnnl_graph_get_compiled_partition_cache_capacity(
        int *capacity) {
    if (capacity == nullptr) return dnnl::graph::impl::status::invalid_argument;
    *capacity = 0;
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    *capacity = dnnl::graph::impl::compiled_partition_cache().get_capacity();
#endif
    return dnnl::graph::impl::status::success;
}

dnnl::graph::impl::status_t dnnl_graph_set_compiled_partition_cache_capacity(
        int capacity) {
    if (capacity < 0) return dnnl::graph::impl::status::invalid_argument;
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    return dnnl::graph::impl::compiled_partition_cache().set_capacity(capacity);
#endif
    return dnnl::graph::impl::status::success;
}
