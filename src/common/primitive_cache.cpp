/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#include "primitive.hpp"
#include "rw_mutex.hpp"

namespace dnnl {
namespace impl {

lru_primitive_cache_t &primitive_cache() {
#ifdef DNNL_ENABLE_PRIMITIVE_CACHE
    static const int capacity
            = getenv_int("DNNL_PRIMITIVE_CACHE_CAPACITY", 1024);
#else
    static const int capacity = 0;
#endif
    static lru_primitive_cache_t cache(capacity);
    return cache;
}

// undocumented API, for testing only
status_t get_primitive_cache_size(int *size) {
    if (size == nullptr) return dnnl::impl::status::invalid_arguments;
    *size = 0;
#ifdef DNNL_ENABLE_PRIMITIVE_CACHE
    utils::lock_read_t lock_r(primitive_cache_t::rw_mutex());
    *size = primitive_cache().get_size();
#endif
    return dnnl::impl::status::success;
}

} // namespace impl
} // namespace dnnl

// API
dnnl::impl::status_t dnnl_get_primitive_cache_capacity(int *capacity) {
    if (capacity == nullptr) return dnnl::impl::status::invalid_arguments;
    *capacity = 0;
#ifdef DNNL_ENABLE_PRIMITIVE_CACHE
    dnnl::impl::utils::lock_read_t lock_r(
            dnnl::impl::primitive_cache_t::rw_mutex());
    *capacity = dnnl::impl::primitive_cache().get_capacity();
#endif
    return dnnl::impl::status::success;
}

dnnl::impl::status_t dnnl_set_primitive_cache_capacity(int capacity) {
    if (capacity < 0) return dnnl::impl::status::invalid_arguments;
#ifdef DNNL_ENABLE_PRIMITIVE_CACHE
    dnnl::impl::utils::lock_write_t lock_w(
            dnnl::impl::primitive_cache_t::rw_mutex());
    return dnnl::impl::primitive_cache().set_capacity(capacity);
#endif
    return dnnl::impl::status::success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
