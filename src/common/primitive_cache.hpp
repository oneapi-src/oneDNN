/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "c_types_map.hpp"
#include "cache_utils.hpp"
#include "oneapi/dnnl/dnnl.h"
#include "primitive_hashing.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {

struct primitive_t;
struct primitive_cache_t : public c_compatible {
    using key_t = primitive_hashing::key_t;
    using result_t = utils::cache_object_t<primitive_t>;
    using create_func_t = result_t (&)(void *);

    virtual ~primitive_cache_t() = default;

    virtual status_t set_capacity(int capacity) = 0;
    virtual int get_capacity() const = 0;
    virtual int get_size() const = 0;

    virtual std::shared_ptr<primitive_desc_t> get_pd(const key_t &key) = 0;
    virtual result_t get_or_create(
            const key_t &key, create_func_t create, void *create_context)
            = 0;
};

// The cache uses LRU replacement policy
struct lru_primitive_cache_t final : public primitive_cache_t {
    lru_primitive_cache_t(int capacity) : cache_(capacity) {};

    ~lru_primitive_cache_t() override = default;

    status_t set_capacity(int capacity) override {
        return cache_.set_capacity(capacity);
    }
    int get_capacity() const override { return cache_.get_capacity(); }
    int get_size() const override { return cache_.get_size(); }

    std::shared_ptr<primitive_desc_t> get_pd(const key_t &key) override;
    result_t get_or_create(const key_t &key, create_func_t create,
            void *create_context) override {
        return cache_.get_or_create(key, create, create_context);
    }

private:
    static void update_key(const key_t &key, const primitive_t &p);
    // Used for testing.
    friend size_t DNNL_API set_primitive_cache_capacity_without_clearing(
            size_t capacity);
    void set_capacity_without_clearing(int capacity) {
        cache_.set_capacity_without_clearing(capacity);
    }

    utils::lru_cache_t<key_t, primitive_t, utils::cache_object_t<primitive_t>,
            update_key>
            cache_;
};

primitive_cache_t &primitive_cache();

// Undocumented API for testing.
status_t DNNL_API get_primitive_cache_size(int *size);
bool DNNL_API is_primitive_in_cache(const primitive_iface_t *p_iface);
bool DNNL_API is_pd_in_cache(const primitive_desc_iface_t *pd_iface);
size_t DNNL_API set_primitive_cache_capacity_without_clearing(size_t capacity);

} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
