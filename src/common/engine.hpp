/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <mutex>

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive.hpp"
#include "primitive_cache.hpp"
#include "utils.hpp"

/** \brief An abstraction of an execution unit with shared resources
 *
 * Responsibilities:
 *   - Provide engine specific memory allocation
 *   - Provide engine specific primitive_desc_t creators
 *   - Provide engine specific primitive cache
 */
struct mkldnn_engine : public mkldnn::impl::c_compatible {
    mkldnn_engine(mkldnn::impl::engine_kind_t kind,
            mkldnn::impl::backend_kind_t backend_kind)
        : kind_(kind), backend_kind_(backend_kind) {
        size_t cache_capacity
                = mkldnn::impl::getenv_int("MKLDNN_CACHE_CAPACITY", 0);
        primitive_cache_ = mkldnn::impl::utils::make_unique<
                mkldnn::impl::lru_primitive_cache_t>(cache_capacity);

        static_assert(std::has_virtual_destructor<
                              mkldnn::impl::primitive_cache_t>::value,
                "primitive_cache_t should have a virtual destructor");
    }

    virtual ~mkldnn_engine() {}

    /** get kind of the current engine */
    mkldnn::impl::engine_kind_t kind() const { return kind_; }

    /** get the backend kind of the current engine */
    mkldnn::impl::backend_kind_t backend_kind() const { return backend_kind_; }

    /** create memory storage */
    virtual mkldnn::impl::status_t create_memory_storage(
            mkldnn::impl::memory_storage_t **storage, unsigned flags,
            size_t size, void *handle)
            = 0;
    mkldnn::impl::status_t create_memory_storage(
            mkldnn::impl::memory_storage_t **storage, size_t size) {
        return create_memory_storage(
                storage, mkldnn::impl::memory_flags_t::alloc, size, nullptr);
    }

    /** create stream */
    virtual mkldnn::impl::status_t create_stream(
            mkldnn::impl::stream_t **stream, unsigned flags)
            = 0;

    /** implementation section (typedefs) */

    // TODO: remove engine?
    typedef mkldnn::impl::status_t (*reorder_primitive_desc_create_f)(
            mkldnn::impl::reorder_pd_t **reorder_pd,
            mkldnn::impl::engine_t *engine,
            const mkldnn::impl::primitive_attr_t *attr,
            mkldnn::impl::engine_t *src_engine,
            const mkldnn::impl::memory_desc_t *src_md,
            mkldnn::impl::engine_t *dst_engine,
            const mkldnn::impl::memory_desc_t *dst_md);

    typedef mkldnn::impl::status_t (*concat_primitive_desc_create_f)(
            mkldnn::impl::concat_pd_t **concat_pd,
            mkldnn::impl::engine_t *engine,
            const mkldnn::impl::primitive_attr_t *attr,
            const mkldnn::impl::memory_desc_t *dst_md, int n, int concat_dim,
            const mkldnn::impl::memory_desc_t *src_mds);

    typedef mkldnn::impl::status_t (*sum_primitive_desc_create_f)(
            mkldnn::impl::sum_pd_t **sum_pd, mkldnn::impl::engine_t *engine,
            const mkldnn::impl::primitive_attr_t *attr,
            const mkldnn::impl::memory_desc_t *dst_md, int n,
            const float *scales, const mkldnn::impl::memory_desc_t *src_mds);

    typedef mkldnn::impl::status_t (*primitive_desc_create_f)(
            mkldnn::impl::primitive_desc_t **, const mkldnn::impl::op_desc_t *,
            const mkldnn::impl::primitive_attr_t *attr,
            mkldnn::impl::engine_t *, const mkldnn::impl::primitive_desc_t *);

    /* implementation section */

    /** return the list of reorder implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const reorder_primitive_desc_create_f *
    get_reorder_implementation_list() const = 0;

    /** return the list of concat implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const concat_primitive_desc_create_f *
    get_concat_implementation_list() const = 0;

    /** return the list of sum implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const sum_primitive_desc_create_f *
    get_sum_implementation_list() const = 0;

    /** return the list of implementations. engine guarantees to return a
     * NULL-terminated list */
    virtual const primitive_desc_create_f *get_implementation_list() const = 0;

    template <typename F>
    mkldnn::impl::status_t get_primitive(mkldnn::impl::primitive_t **primitive,
            const mkldnn::impl::primitive_desc_t *pd,
            const F &create_primitive_impl, bool use_global_scratchpad) {
        double ms = mkldnn::impl::get_msec();

        // create a key for the requested primitive
        int dummy_impl_id = 0;
        mkldnn::impl::primitive_hashing::key_t key(pd->kind(), pd->op_desc(),
                pd->attr(), dummy_impl_id, this->mkldnn_get_max_threads());

        // lock cache
        recursive_mutex_.lock();
        auto primitive_impl = primitive_cache_->get(key);
        if (primitive_impl) { // cache hit
            // unlock cache because it's safe to create a wrapper in parallel
            recursive_mutex_.unlock();
            // create a wrapper for primitive_impl
            auto status
                    = mkldnn::impl::safe_ptr_assign<mkldnn::impl::primitive_t>(
                            *primitive,
                            new mkldnn::impl::primitive_t(
                                    primitive_impl, use_global_scratchpad));
            if (status != mkldnn::impl::status::success) return status;

            ms = mkldnn::impl::get_msec() - ms;
            if (mkldnn::impl::mkldnn_verbose()->level >= 2) {
                printf("mkldnn_verbose,create:cache hit,%s,%g\n",
                        (*primitive)->pd()->info(), ms);
                fflush(0);
            }
            return status;
        }

        // cache miss - create a requested primitive_impl and a wrapper
        auto status = mkldnn::impl::safe_ptr_assign<mkldnn::impl::primitive_t>(
                *primitive,
                new mkldnn::impl::primitive_t(
                        create_primitive_impl(), use_global_scratchpad));

        if (status != mkldnn::impl::status::success) {
            recursive_mutex_.unlock();
            return status;
        }

        status = (*primitive)->init();
        if (status != mkldnn::impl::status::success) {
            recursive_mutex_.unlock();
            delete *primitive;
            return status;
        }

        // update op_desc and attr pointers in the key
        key.op_desc_ = (*primitive)->pd()->op_desc();
        key.attr_ = (*primitive)->pd()->attr();

        primitive_cache_->add(key, (*primitive)->get_primitive_impl());
        recursive_mutex_.unlock();

        ms = mkldnn::impl::get_msec() - ms;
        if (mkldnn::impl::mkldnn_verbose()->level >= 2) {
            printf("mkldnn_verbose,create:cache miss,%s,%g\n",
                    (*primitive)->pd()->info(), ms);
            fflush(0);
        }
        return status;
    }
    int mkldnn_get_max_threads();

protected:
    mkldnn::impl::engine_kind_t kind_;
    mkldnn::impl::backend_kind_t backend_kind_;
    std::unique_ptr<mkldnn::impl::primitive_cache_t> primitive_cache_;
    // As a primitive can be created inside another one a recursive_mutex is
    // required
    std::recursive_mutex recursive_mutex_;
};

namespace mkldnn {
namespace impl {

struct engine_factory_t : public c_compatible {
    virtual size_t count() const = 0;
    virtual status_t engine_create(engine_t **engine, size_t index) const = 0;
    virtual ~engine_factory_t() = default;
};

} // namespace impl
} // namespace mkldnn

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
