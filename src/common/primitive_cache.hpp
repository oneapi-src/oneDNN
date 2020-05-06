/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_CACHE_HPP
#define COMMON_PRIMITIVE_CACHE_HPP

#include <list>
#include <memory>
#include <unordered_map>

#include "c_types_map.hpp"
#include "dnnl.h"
#include "primitive_hashing.hpp"
#include "rw_mutex.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {

struct primitive_t;
struct primitive_cache_t : public c_compatible {
    using key_t = primitive_hashing::key_t;
    using value_t = std::shared_ptr<primitive_t>;

    virtual int get_capacity() const = 0;
    virtual status_t set_capacity(int capacity) = 0;

    // for undocumented API
    virtual int get_size() const = 0;

    virtual void add(const key_t &key, const value_t &impl) = 0;
    virtual value_t get(const key_t &key) = 0;

    virtual ~primitive_cache_t() = default;

    static utils::rw_mutex_t &rw_mutex() {
        static utils::rw_mutex_t mutex;
        return mutex;
    }
};

// The cache uses LRU replacement policy
struct lru_primitive_cache_t : public primitive_cache_t {
    lru_primitive_cache_t(int capacity) : capacity_(capacity) {}

    int get_capacity() const override { return (int)capacity_; }

    status_t set_capacity(int capacity) override {
        capacity_ = (size_t)capacity;
        // Check if number of entries exceeds the new capacity
        if (cache_list_.size() > capacity_) {
            // Evict excess entries
            size_t n_excess_entries = cache_list_.size() - capacity_;
            evict(n_excess_entries);
        }
        return status::success;
    }

    // for undocumented API
    int get_size() const override { return (int)cache_list_.size(); }

    void add(const key_t &key, const value_t &impl) override {
        // cache is disabled
        if (capacity_ == 0) return;

        if (cache_list_.size() >= capacity_) {
            // evict the least recently used entry
            evict(1);
        }
        // place a new entry to cache_list_ and update cache_mapper_
        cache_list_.emplace_front(key, impl);
        cache_mapper_.insert(std::make_pair(key, cache_list_.begin()));
    }

    value_t get(const key_t &key) override {
        // cache is disabled
        if (capacity_ == 0) return nullptr;

        auto it = cache_mapper_.find(key);
        if (it == cache_mapper_.end()) { return nullptr; }
        // move 1 cache_list_ node to the front of the cache_list_
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        return cache_list_.front().second;
    }
    DNNL_DISALLOW_COPY_AND_ASSIGN(lru_primitive_cache_t);

private:
    // an aux member function for evicting n the least recently used entries
    void evict(size_t n) {
        for (size_t e = 0; e < n; e++) {
            cache_mapper_.erase(cache_list_.back().first);
            cache_list_.pop_back();
        }
    }

    size_t capacity_;
    using cache_list_t = std::list<std::pair<key_t, value_t>>;
    cache_list_t cache_list_;
    std::unordered_map<key_t, cache_list_t::iterator> cache_mapper_;
};

lru_primitive_cache_t &primitive_cache();

// undocumented API, for testing only
status_t DNNL_API get_primitive_cache_size(int *size);

} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
