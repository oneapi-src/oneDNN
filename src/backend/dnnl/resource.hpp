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
#ifndef BACKEND_DNNL_RESOURCE_HPP
#define BACKEND_DNNL_RESOURCE_HPP

#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include "utils/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

// An interface class. For each kernel, we can inherit the interface and devired
// subclass to represent different resource
class resource_t {
public:
    virtual ~resource_t() {};
};

// In multithreads scenarios, there are two requests to the kernel's execute()
// method:
// 1. To support multithreads execution, the kernel instance must be immutable
//    after its creating, so its execute() method must be const and stateless.
// 2. To reduce execution overhead, we want to create some mutable resource once
//    in execute(), cache them and only change few field of them in every
//    iteration. Those resources are dnnl::memory objects (only the wrapper, not
//    contains the underlying buffer) at this moment
// Base on above two requests, those mutable resources can't be a part of the
// kernel instance. So we design this resource_cache_t class to hold those
// mutable resources.
// Kernel should search the resource it needs in the cache, if found, it should
// use the found one, otherwise it should create a new one and cache it to the
// cache. At this moment, we observed that those resources will not be shared
// between threads, so we made the cache be thread local to reduce the search
// and sync overhead.
class resource_cache_t {
public:
    using key_t = size_t;
    using cached_t = std::unique_ptr<resource_t>;
    using creator_t = std::function<cached_t()>;

    resource_cache_t() = default;

    bool has_resource(const key_t &key) const;
    size_t size() const;
    void clear();

    // Add a resource to the cache and return the added resource's raw pointer
    // note: transfer the ownership of resource from users to the cache
    template <typename T>
    T *add(const key_t &key, cached_t &&resource) {
        assertm(resource_map_.count(key) == 0, "key has existed");
        auto ret = resource_map_.emplace(key, std::move(resource));
        return static_cast<T *>(ret.first->second.get());
    }

    // Get out the raw pointer of cached resource
    template <typename T>
    T *get(const key_t &key) const {
        assertm(resource_map_.count(key), "no such key");
        return static_cast<T *>(resource_map_.at(key).get());
    }

    // A wrapper for usability. Get out the raw pointer of cached resource. If
    // the resource is not found, the function will call the creator func to
    // creat a new one and do the adding process
    template <typename T>
    T *get(const key_t &key, const creator_t &func) {
        if (has_resource(key)) {
            return static_cast<T *>(resource_map_.at(key).get());
        } else {
            auto ret = resource_map_.emplace(key, func());
            return static_cast<T *>(ret.first->second.get());
        }
    }

private:
    resource_cache_t(const resource_cache_t &other) = delete;
    resource_cache_t &operator=(const resource_cache_t &other) = delete;

    thread_local static std::unordered_map<key_t, cached_t> resource_map_;
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
