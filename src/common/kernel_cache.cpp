/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "common/kernel_cache.hpp"
#include "common/cache_utils.hpp"

// inject a specialization of std::hash for kernel_cache::key_t into std
// namespace
namespace std {
template <>
struct hash<dnnl::impl::kernel_cache::key_t> {
    using argument_type = dnnl::impl::kernel_cache::key_t;
    using result_type = std::size_t;
    result_type operator()(const argument_type &key) const {
        return key.hash();
    }
};
} // namespace std

namespace dnnl {
namespace impl {
namespace kernel_cache {

struct iface_t::cache_t {
    using result_t = iface_t::result_t;
    using create_func_t = iface_t::create_func_t;

    cache_t(int capacity) : cache_(capacity) {};

    ~cache_t() = default;

    status_t set_capacity(int capacity) {
        return cache_.set_capacity(capacity);
    }
    int get_capacity() const { return cache_.get_capacity(); }
    int get_size() const { return cache_.get_size(); }

    result_t get_or_create(
            const key_t &key, create_func_t create, void *create_context) {
        return cache_.get_or_create(key, create, create_context);
    }

private:
    utils::lru_cache_t<key_t, value_t, result_t> cache_;
};

iface_t get() {
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    static const int capacity
            = getenv_int_user("PRIMITIVE_CACHE_CAPACITY", 1024);
#else
    static const int capacity = 0;
#endif
    static iface_t::cache_t cache(capacity);
    return cache;
}

status_t iface_t::set_capacity(int capacity) {
    return cache_.set_capacity(capacity);
}

int iface_t::get_capacity() const {
    return cache_.get_capacity();
}

int iface_t::get_size() const {
    return cache_.get_size();
}

iface_t::result_t iface_t::get_or_create(
        const key_t &key, create_func_t create, void *create_context) {
    auto r = cache_.get_or_create(key, create, create_context);
    return {r.value, r.status};
}

} // namespace kernel_cache
} // namespace impl
} // namespace dnnl
