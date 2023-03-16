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
#include "primitive.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_iface.hpp"
#include "z_magic.hpp"

namespace dnnl {
namespace impl {

primitive_cache_t &primitive_cache() {
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    static const int capacity
            = getenv_int_user("PRIMITIVE_CACHE_CAPACITY", 1024);
#else
    static const int capacity = 0;
#endif
    static lru_primitive_cache_t cache(capacity);
    return cache;
}

// Undocumented API, for testing only
status_t get_primitive_cache_size(int *size) {
    if (size == nullptr) return dnnl::impl::status::invalid_arguments;
    *size = 0;
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    *size = primitive_cache().get_size();
#endif
    return dnnl::impl::status::success;
}

bool is_pd_in_cache(const primitive_desc_iface_t *pd_iface) {
    const auto *pd = pd_iface->impl().get();
    const auto *engine = pd_iface->engine();
    primitive_hashing::key_t key(pd, engine);
    return bool(primitive_cache().get_pd(key));
}

bool is_primitive_in_cache(const primitive_iface_t *p_iface) {
    return is_pd_in_cache(p_iface->pd());
}

size_t set_primitive_cache_capacity_without_clearing(size_t capacity) {
    size_t old_capacity = primitive_cache().get_capacity();
    lru_primitive_cache_t &lru_primitive_cache
            = static_cast<lru_primitive_cache_t &>(primitive_cache());
    lru_primitive_cache.set_capacity_without_clearing((int)capacity);
    return old_capacity;
}

std::shared_ptr<primitive_desc_t> lru_primitive_cache_t::get_pd(
        const key_t &key) {
    primitive_cache_t::result_t result = cache_.get(key);
    return result.value != nullptr ? result.value->pd() : nullptr;
};

void lru_primitive_cache_t::update_key(
        const primitive_hashing::key_t &k, const primitive_t &p) {
    const primitive_desc_t *pd = p.pd().get();
    k.op_desc_ = pd->op_desc();
    k.attr_ = pd->attr();
}

} // namespace impl
} // namespace dnnl

// API
dnnl::impl::status_t dnnl_get_primitive_cache_capacity(int *capacity) {
    if (capacity == nullptr) return dnnl::impl::status::invalid_arguments;
    *capacity = 0;
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    *capacity = dnnl::impl::primitive_cache().get_capacity();
#endif
    return dnnl::impl::status::success;
}

dnnl::impl::status_t dnnl_set_primitive_cache_capacity(int capacity) {
    if (capacity < 0) return dnnl::impl::status::invalid_arguments;
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    return dnnl::impl::primitive_cache().set_capacity(capacity);
#endif
    return dnnl::impl::status::success;
}
