/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef PRIMITIVE_HPP
#define PRIMITIVE_HPP

#include <assert.h>

#include "dnnl.h"

#include "c_types_map.hpp"
#include "memory_storage.hpp"
#include "memory_tracking.hpp"
#include "primitive_desc.hpp"
#include "primitive_exec_types.hpp"
#include "rw_mutex.hpp"
#include "scratchpad.hpp"

#include <type_traits>

namespace dnnl {
namespace impl {

struct resource_mapper_t;
// Primitive implementation
struct primitive_t : public c_compatible {
    using primitive_list_t = std::vector<const primitive_t *>;

    primitive_t(const primitive_desc_t *pd) : pd_(pd->clone()) {}
    virtual ~primitive_t() = default;

    virtual status_t init(engine_t *engine) { return status::success; }

    status_t init(engine_t *engine, bool use_global_scratchpad) {
        CHECK(init(engine));
        use_global_scratchpad_ = use_global_scratchpad;
        return status::success;
    }

    const std::shared_ptr<primitive_desc_t> &pd() const { return pd_; }
    primitive_kind_t kind() const { return pd_->kind(); }
    virtual status_t execute(const exec_ctx_t &ctx) const = 0;

    virtual status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const {
        return status::success;
    }

    bool use_global_scratchpad() const { return use_global_scratchpad_; }

protected:
    template <typename impl_type, typename pd_t>
    static status_t create_primitive_common(
            std::shared_ptr<primitive_t> &primitive, const pd_t *pd,
            engine_t *engine, bool use_global_scratchpad,
            bool is_primitive_nested) {
        const auto print_verbose = [&](int level, bool cache_hit,
                                           const char *pd_info, double time) {
            if (level >= 2) {
                const char *str = cache_hit ? "dnnl_verbose,create:cache_hit"
                                            : "dnnl_verbose,create:cache_miss";
                printf("%s,%s,%g\n", str, pd_info, time);
                fflush(0);
            }
        };
        auto &global_primitive_cache = primitive_cache();
        double ms = get_msec();
        int nthreads = dnnl_get_max_threads();
        primitive_hashing::key_t key_to_lookup(pd, engine, nthreads);

        if (!is_primitive_nested) primitive_cache_t::rw_mutex().lock_read();

        auto p = global_primitive_cache.get(key_to_lookup);
        auto status = status::success;
        bool cache_hit = false;
        if (p) {
            if (!is_primitive_nested)
                primitive_cache_t::rw_mutex().unlock_read();
            cache_hit = true;
        } else {
            if (!is_primitive_nested) {
                primitive_cache_t::rw_mutex().unlock_read();
                primitive_cache_t::rw_mutex().lock_write();
                // double check to workaround the ABA problem
                p = global_primitive_cache.get(key_to_lookup);
            }

            if (!p) {
                // the requested primitive hasn't been added to the cache yet
                p = std::make_shared<impl_type>(pd);
                status = p->init(engine, use_global_scratchpad);
                if (status != status::success) {
                    if (!is_primitive_nested)
                        primitive_cache_t::rw_mutex().unlock_write();
                    return status;
                }
                primitive_hashing::key_t key_to_cache(
                        p->pd().get(), engine, nthreads);
                global_primitive_cache.add(key_to_cache, p);
                if (!is_primitive_nested)
                    primitive_cache_t::rw_mutex().unlock_write();
                cache_hit = false;
            } else {
                // another thread added the requested primitive
                if (!is_primitive_nested)
                    primitive_cache_t::rw_mutex().unlock_write();
                cache_hit = true;
            }
        }
        primitive = p;
        ms = get_msec() - ms;
        print_verbose(get_verbose(), cache_hit, p->pd()->info(engine), ms);
        return status;
    }

    std::shared_ptr<primitive_desc_t> pd_;
    bool use_global_scratchpad_;

private:
    primitive_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(primitive_t);
};

// This is a helper class which is used for forwarding a scratchpad
// from master primitive to the nested ones.
struct nested_scratchpad_t {
    nested_scratchpad_t(const exec_ctx_t &master_ctx, int key,
            const std::shared_ptr<primitive_t> &nested_p);
    const memory_tracking::grantor_t *grantor() const { return grantor_.get(); }

    DNNL_DISALLOW_COPY_AND_ASSIGN(nested_scratchpad_t);

private:
    std::unique_ptr<memory_storage_t> scratchpad_mem_storage_;
    std::unique_ptr<memory_tracking::grantor_t> grantor_;
};

// The resource_t abstraction is a base class for all resource classes.
// Those are responsible for holding a part of a primitive implementation that
// cannot be stored in the primitive cache as part of the implementation.
// Currently, there are two such things:
// 1. Any memory (memory_t, memory_storage_t, etc...), because it contains
// an engine.
// 2. (for GPU only) compiled kernels, because they are context dependent.
//
// The idea is that each primitive implementation should be able to create
// a resource and put there everything it needs to run, which cannot be stored
// in the cache as part of the primitive implementation. To create the resource
// each primitive implementation can override a function `create_resource`.
//
// This abstraction takes ownership of all content it holds hence it should be
// responsible for destroying it as well.
struct resource_t : public c_compatible {
    virtual ~resource_t() = default;
};

// The resource_mapper_t is an abstraction for holding resources for
// a particular primitive implementation and providing corresponding mapping.
//
// Interacting with the mapper happens in two steps:
// 1. Initialization. Each derived from impl::primitive_t class may define
// `create_resource` member function that is responsible for creating a
// certain derived from resource_t object and filling it with some content,
// e.g. memory for scales, OpenCL kernels etc...
// 2. Passing it to the execution function which extracts needed resources and
// uses them at execution time. The mapper is passed to the execution function
// with the execution context.
//
// The resource_mapper_t takes ownership of all resources hence it should be
// responsible for destroying them as well.
struct resource_mapper_t {
    using key_t = const primitive_t;
    using mapped_t = std::unique_ptr<resource_t>;

    resource_mapper_t() = default;

    bool has_resource(const primitive_t *p) const {
        return primitive_to_resource_.count(p);
    }

    void add(key_t *p, mapped_t &&r) {
        assert(primitive_to_resource_.count(p) == 0);
        primitive_to_resource_.emplace(p, std::move(r));
    }

    template <typename T>
    const T *get(key_t *p) const {
        assert(primitive_to_resource_.count(p));
        return utils::downcast<T *>(primitive_to_resource_.at(p).get());
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(resource_mapper_t);

private:
    std::unordered_map<key_t *, mapped_t> primitive_to_resource_;
};

} // namespace impl
} // namespace dnnl

#define ARG_TYPE(t) \
    typename std::remove_cv<typename std::remove_pointer<t>::type>::type

#define CTX_IN_MEM(type, arg) \
    static_cast<const ARG_TYPE(type) *>(CTX_IN_STORAGE(arg).data_handle())

#define CTX_OUT_MEM(type, arg) \
    static_cast<ARG_TYPE(type) *>(CTX_OUT_STORAGE(arg).data_handle())

// dnnl_primitive is a user facing entity that has an alias primitive_iface_t
// for internal use.
// The primitive_iface_t is responsible for holding:
// 1. impl::primitive_t - a primitive implementation that can be
// stored in the primitive cache. Other data members are NOT stored in
// the cache
// 2. scratchpad_t - a memory for scratchpad
// 3. primitive_desc_iface_t - an alias for dnnl_primitive_desc and is
// a user facing primitive descriptor (the one a user should create prior
// creating a primitive)
// 4. resource_mapper_t - a resource mapper that provides a mapping between
// impl::primitive_t and its resource
//
// Note: primitive_desc_iface_t and impl::primitive_t share the same
// impl::primitive_desc_t
struct dnnl_primitive : public dnnl::impl::c_compatible {
    dnnl_primitive(const std::shared_ptr<dnnl::impl::primitive_t> &primitive,
            dnnl::impl::engine_t *engine);

    dnnl::impl::status_t init();
    dnnl::impl::engine_t *engine() const;
    const primitive_desc_iface_t *pd() const;
    const std::shared_ptr<dnnl::impl::primitive_t> &get_primitive() const;
    dnnl::impl::status_t execute(dnnl::impl::exec_ctx_t &ctx) const;

private:
    std::shared_ptr<dnnl::impl::primitive_t> primitive_;
    std::unique_ptr<dnnl::impl::scratchpad_t> scratchpad_;
    std::unique_ptr<primitive_desc_iface_t> pd_;
    dnnl::impl::resource_mapper_t resource_mapper_;

    dnnl_primitive() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(dnnl_primitive);
};

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
