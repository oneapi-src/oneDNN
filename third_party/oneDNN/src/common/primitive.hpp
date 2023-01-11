/*******************************************************************************
* Copyright 2016-2022 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_HPP
#define COMMON_PRIMITIVE_HPP

#include <assert.h>
#include <atomic>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "cache_blob.hpp"
#include "memory_storage.hpp"
#include "memory_tracking.hpp"
#include "primitive_desc.hpp"
#include "primitive_exec_types.hpp"
#include "rw_mutex.hpp"
#include "scratchpad.hpp"

#include <future>
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

    status_t init(engine_t *engine, bool use_global_scratchpad,
            const cache_blob_t &cache_blob) {
        cache_blob_ = cache_blob;
        CHECK(init(engine));
        CHECK(init_cached_resource(engine));
        use_global_scratchpad_ = use_global_scratchpad;
        // The `cache_blob_` is no longer needed after primitive creation.
        cache_blob_ = cache_blob_t();
        return status::success;
    }

    const std::shared_ptr<primitive_desc_t> &pd() const { return pd_; }
    primitive_kind_t kind() const { return pd_->kind(); }
    virtual status_t execute(const exec_ctx_t &ctx) const = 0;

    virtual status_t get_cache_blob(
            engine_t *engine, cache_blob_t &cache_blob) const {
        assert(!"unexpected");
        return status::runtime_error;
    }

    virtual status_t get_cache_blob_size(size_t *size) const {
        assert(!"unexpected");
        return status::runtime_error;
    }

    virtual status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const {
        return status::success;
    }

    // Although this function is marked as `const` it changes primitive_t state.
    // The only place where this function should be used is in:
    // `init(engine_t *engine, bool use_global_scratchpad)` during primitive_t
    // creation in `create_primitive_common`.
    // The rationale behind marking it as `const` is to simplify enabling the
    // primitive cache mode for storing compiled GPU kernels instead of
    // binaries - DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE=ON and to preserve the
    // current primitive cache implementation.
    //
    // The main idea is to create a resource inside the primitive_t only once
    // and cache it as part of primitive_t.
    // TODO: The ultimate goal is to switch completely to caching compiled
    // GPU kernels therefore the code will be thrown out once it's done.
    virtual status_t init_cached_resource(engine_t *engine) const {
        return status::success;
    }

    bool use_global_scratchpad() const { return use_global_scratchpad_; }
    cache_blob_t cache_blob() const { return cache_blob_; }

protected:
    template <typename impl_type, typename pd_t>
    static status_t create_primitive_common(
            std::pair<std::shared_ptr<primitive_t>, bool> &primitive,
            const pd_t *pd, engine_t *engine, bool use_global_scratchpad,
            const cache_blob_t &cache_blob) {

        auto &global_primitive_cache = primitive_cache();
        primitive_hashing::key_t key(pd, engine);

        std::promise<primitive_cache_t::cache_value_t> p_promise;
        // Try to get the shared future from the cache, if it's missing then
        // a shared future with no shared state is returned and the passed
        // shared future is added, otherwise a valid shared future is returned
        // and no insertion is performed.
        auto p_future = global_primitive_cache.get_or_add(
                key, p_promise.get_future());

        bool is_from_cache = p_future.valid();

        auto status = status::success;
        std::shared_ptr<primitive_t> p;

        if (is_from_cache) {
            // The requested primitive is present in the cache or is being
            // created by another thread.
            p = p_future.get().primitive;
            if (!p) return p_future.get().status;
        } else {
            // The requested primitive is NOT present in the cache therefore
            // we have to create it and notify the waiting threads
            // once the creation is done.
            p = std::make_shared<impl_type>(pd);
            status = p->init(engine, use_global_scratchpad, cache_blob);
            if (status != status::success) {
                // Communicate an error.
                p_promise.set_value({nullptr, status});
                // Remove the shared future from the cache because it's
                // invalidated. An invalidated shared future is the one that
                // stores a nullptr.
                global_primitive_cache.remove_if_invalidated(key);
                return status;
            } else {
                // Store the created primitive in the shared future and notify
                // the waiting threads.
                p_promise.set_value({p, status});

                // The key_t contains pointers to op_desc and attr objects that
                // reside in pd. When primitive_t is created it copies the pd
                // and hence contains a copy.
                // Since the created primitive_t is stored in the cache with
                // the corresponding key, the key must contain pointers to
                // op_desc and attr that reside in the coppied pd
                // in the primitive_t.
                // Therefore the pointers in the key, which has already been put
                // into the cache, must be updated.
                global_primitive_cache.update_entry(key, p->pd().get());
            }
        }
        primitive = std::make_pair(p, is_from_cache);
        return status;
    }

    std::shared_ptr<primitive_desc_t> pd_;
    bool use_global_scratchpad_;
    cache_blob_t cache_blob_;

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

    ~nested_scratchpad_t();

    DNNL_DISALLOW_COPY_AND_ASSIGN(nested_scratchpad_t);

private:
    std::unique_ptr<memory_storage_t> scratchpad_mem_storage_;
    std::unique_ptr<memory_tracking::grantor_t> grantor_;
};

} // namespace impl
} // namespace dnnl

#define ARG_TYPE(t) \
    typename std::remove_cv<typename std::remove_pointer<t>::type>::type

#define CTX_IN_MEM(type, arg) \
    static_cast<const ARG_TYPE(type) *>(ctx.host_ptr(arg))

// Returns destination memory which may not have been zero pad initialized.
#define CTX_OUT_MEM(type, arg) static_cast<ARG_TYPE(type) *>(ctx.host_ptr(arg))

// Returns destination memory which has been zero pad initialized. This macro
// may result in a failure returned via the `status` input since zero pad
// may fail.
#define CTX_OUT_CLEAN_MEM(type, arg, status) \
    static_cast<ARG_TYPE(type) *>(ctx.host_ptr(arg, true, &status))

#endif
