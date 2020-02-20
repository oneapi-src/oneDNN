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

#ifndef PRIMITIVE_CACHE_HPP
#define PRIMITIVE_CACHE_HPP

#include <list>
#include <memory>
#include <unordered_map>

#include "c_types_map.hpp"
#include "dnnl.h"
#include "primitive_hashing.hpp"
#include "primitive_impl.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {

struct primitive_cache_t : public c_compatible {
    using key_type = primitive_hashing::key_t;
    using value_type = std::shared_ptr<primitive_impl_t>;

    virtual void add(const key_type &key, const value_type &impl) = 0;
    virtual value_type get(const key_type &key) = 0;

    virtual ~primitive_cache_t() = default;
};

// The cache uses LRU replacement policy
struct lru_primitive_cache_t : public primitive_cache_t {
    lru_primitive_cache_t(size_t capacity) : capacity_(capacity) {}

    virtual void add(const key_type &key, const value_type &impl) override {
        // cache is disabled
        if (capacity_ == 0) return;

        if (cache_list_.size() >= capacity_) {
            // invalidate the least recently used entry
            cache_mapper_.erase(cache_list_.back().first);
            cache_list_.pop_back();
        }
        // place a new entry to cache_list_ and update cache_mapper_
        cache_list_.emplace_front(key, impl);
        cache_mapper_.insert(std::make_pair(key, cache_list_.begin()));
    }

    virtual value_type get(const key_type &key) override {
        // cache is disabled
        if (capacity_ == 0) return nullptr;

        auto it = cache_mapper_.find(key);
        if (it == cache_mapper_.end()) { return nullptr; }
        // move 1 cache_list_ node to the front of the cache_list_
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        return cache_list_.front().second;
    }

private:
    size_t capacity_;
    using cache_list_type = std::list<std::pair<key_type, value_type>>;
    cache_list_type cache_list_;
    std::unordered_map<key_type, cache_list_type::iterator> cache_mapper_;
};

} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
