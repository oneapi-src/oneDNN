/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "primitive_cache.hpp"
#include "c_types_map.hpp"
#include "cache_utils.hpp"
#include "kernel_cache.hpp"
#include "primitive.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_iface.hpp"
#include "z_magic.hpp"

namespace dnnl {
namespace impl {

// The cache uses LRU replacement policy
struct primitive_cache_t {
    using key_t = primitive_hashing::key_t;
    using result_t = primitive_cache_iface_t::result_t;
    using create_func_t = result_t (&)(void *);

    primitive_cache_t(int capacity) : cache_(capacity) {};

    ~primitive_cache_t() = default;

    status_t set_capacity(int capacity) {
        return cache_.set_capacity(capacity);
    }
    int get_capacity() const { return cache_.get_capacity(); }
    int get_size() const { return cache_.get_size(); }

    std::shared_ptr<primitive_desc_t> get_pd(const key_t &key) {
        result_t result = cache_.get(key);
        return result.value != nullptr ? result.value->pd() : nullptr;
    }

    result_t get_or_create(
            const key_t &key, create_func_t create, void *create_context) {
        return cache_.get_or_create(key, create, create_context);
    }

private:
    static void update_key(const key_t &key, const primitive_t &p) {
        const primitive_desc_t *pd = p.pd().get();
        key.op_desc_ = pd->op_desc();
        key.attr_ = pd->attr();
    }
    // Used for testing.
    friend size_t DNNL_API set_primitive_cache_capacity_without_clearing(
            size_t capacity);
    void set_capacity_without_clearing(int capacity) {
        cache_.set_capacity_without_clearing(capacity);
    }

    utils::lru_cache_t<key_t, primitive_t, result_t, update_key> cache_;
};

primitive_cache_t &global_primitive_cache() {
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    static const int capacity
            = getenv_int_user("PRIMITIVE_CACHE_CAPACITY", 1024);
#else
    static const int capacity = 0;
#endif
    static primitive_cache_t cache(capacity);
    return cache;
}

primitive_cache_iface_t primitive_cache() {
    return global_primitive_cache();
}

// Undocumented API, for testing only
status_t get_primitive_cache_size(int *size) {
    if (size == nullptr) return dnnl::impl::status::invalid_arguments;
    *size = 0;
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    *size = global_primitive_cache().get_size();
#endif
    return dnnl::impl::status::success;
}

bool is_pd_in_cache(const primitive_desc_iface_t *pd_iface) {
    const auto *pd = pd_iface->impl().get();
    const auto *engine = pd_iface->engine();
    primitive_hashing::key_t key(pd, engine);
    return bool(global_primitive_cache().get_pd(key));
}

bool is_primitive_in_cache(const primitive_iface_t *p_iface) {
    return is_pd_in_cache(p_iface->pd());
}

size_t set_primitive_cache_capacity_without_clearing(size_t capacity) {
    size_t old_capacity = global_primitive_cache().get_capacity();
    global_primitive_cache().set_capacity_without_clearing((int)capacity);
    return old_capacity;
}

status_t primitive_cache_iface_t::set_capacity(int capacity) {
    return cache_.set_capacity(capacity);
}

int primitive_cache_iface_t::get_capacity() const {
    return cache_.get_capacity();
}

int primitive_cache_iface_t::get_size() const {
    return cache_.get_size();
}

std::shared_ptr<primitive_desc_t> primitive_cache_iface_t::get_pd(
        const key_t &key) {
    return cache_.get_pd(key);
}

primitive_cache_iface_t::result_t primitive_cache_iface_t::get_or_create(
        const key_t &key, create_func_t create, void *create_context) {
    auto r = cache_.get_or_create(key, create, create_context);
    return {std::move(r.value), r.status};
}

} // namespace impl
} // namespace dnnl

// API
dnnl::impl::status_t dnnl_get_primitive_cache_capacity(int *capacity) {
    if (capacity == nullptr) return dnnl::impl::status::invalid_arguments;
    *capacity = 0;
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    *capacity = dnnl::impl::global_primitive_cache().get_capacity();
    assert(*capacity == dnnl::impl::kernel_cache::get().get_capacity());
#endif
    return dnnl::impl::status::success;
}

dnnl::impl::status_t dnnl_set_primitive_cache_capacity(int capacity) {
    if (capacity < 0) return dnnl::impl::status::invalid_arguments;
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    auto status = dnnl::impl::global_primitive_cache().set_capacity(capacity);
    if (status != dnnl::impl::status::success) return status;
    return dnnl::impl::kernel_cache::get().set_capacity(capacity);
#endif
    return dnnl::impl::status::success;
}
