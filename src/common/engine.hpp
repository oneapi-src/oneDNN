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

#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <mutex>

#include "dnnl.h"

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
struct dnnl_engine : public dnnl::impl::c_compatible {
    dnnl_engine(dnnl::impl::engine_kind_t kind,
            dnnl::impl::runtime_kind_t runtime_kind)
        : kind_(kind), runtime_kind_(runtime_kind) {
        primitive_cache_ = dnnl::impl::utils::make_unique<
                dnnl::impl::lru_primitive_cache_t>(
                get_primitive_cache_capacity());

        static_assert(std::has_virtual_destructor<
                              dnnl::impl::primitive_cache_t>::value,
                "primitive_cache_t should have a virtual destructor");
    }

    virtual ~dnnl_engine() {}

    /** get kind of the current engine */
    dnnl::impl::engine_kind_t kind() const { return kind_; }

    /** get the runtime kind of the current engine */
    dnnl::impl::runtime_kind_t runtime_kind() const { return runtime_kind_; }

    /** create memory storage */
    virtual dnnl::impl::status_t create_memory_storage(
            dnnl::impl::memory_storage_t **storage, unsigned flags, size_t size,
            void *handle)
            = 0;
    dnnl::impl::status_t create_memory_storage(
            dnnl::impl::memory_storage_t **storage, size_t size) {
        return create_memory_storage(
                storage, dnnl::impl::memory_flags_t::alloc, size, nullptr);
    }

    /** create stream */
    virtual dnnl::impl::status_t create_stream(dnnl::impl::stream_t **stream,
            unsigned flags, const dnnl::impl::stream_attr_t *attr)
            = 0;

    /** implementation section (typedefs) */

    // TODO: remove engine?
    typedef dnnl::impl::status_t (*reorder_primitive_desc_create_f)(
            dnnl::impl::reorder_pd_t **reorder_pd, dnnl::impl::engine_t *engine,
            const dnnl::impl::primitive_attr_t *attr,
            dnnl::impl::engine_t *src_engine,
            const dnnl::impl::memory_desc_t *src_md,
            dnnl::impl::engine_t *dst_engine,
            const dnnl::impl::memory_desc_t *dst_md);

    typedef dnnl::impl::status_t (*concat_primitive_desc_create_f)(
            dnnl::impl::concat_pd_t **concat_pd, dnnl::impl::engine_t *engine,
            const dnnl::impl::primitive_attr_t *attr,
            const dnnl::impl::memory_desc_t *dst_md, int n, int concat_dim,
            const dnnl::impl::memory_desc_t *src_mds);

    typedef dnnl::impl::status_t (*sum_primitive_desc_create_f)(
            dnnl::impl::sum_pd_t **sum_pd, dnnl::impl::engine_t *engine,
            const dnnl::impl::primitive_attr_t *attr,
            const dnnl::impl::memory_desc_t *dst_md, int n, const float *scales,
            const dnnl::impl::memory_desc_t *src_mds);

    typedef dnnl::impl::status_t (*primitive_desc_create_f)(
            dnnl::impl::primitive_desc_t **, const dnnl::impl::op_desc_t *,
            const dnnl::impl::primitive_attr_t *attr, dnnl::impl::engine_t *,
            const dnnl::impl::primitive_desc_t *);

    /* implementation section */

    /** return the list of reorder implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const reorder_primitive_desc_create_f *
    get_reorder_implementation_list(const dnnl::impl::memory_desc_t *src_md,
            const dnnl::impl::memory_desc_t *dst_md) const = 0;

    /** return the list of concat implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const concat_primitive_desc_create_f *
    get_concat_implementation_list() const = 0;

    /** return the list of sum implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const sum_primitive_desc_create_f *
    get_sum_implementation_list() const = 0;

    /** return the list of implementations for a given descriptor.
     * engine guarantees to return a NULL-terminated list */
    virtual const primitive_desc_create_f *get_implementation_list(
            const dnnl::impl::op_desc_t *desc) const = 0;

    template <typename F>
    dnnl::impl::status_t get_primitive_iface(
            dnnl::impl::primitive_iface_t **primitive_iface,
            const dnnl::impl::primitive_desc_t *pd, const F &create_primitive,
            bool use_global_scratchpad) {

        auto print_verbose =
                [](int level, bool is_cache_hit,
                        const dnnl::impl::primitive_iface_t *p_iface,
                        double time) {
                    if (level >= 2) {
#ifdef DNNL_ENABLE_PRIMITIVE_CACHE
                        const char *str = is_cache_hit
                                ? "dnnl_verbose,create:cache_hit"
                                : "dnnl_verbose,create:cache_miss";
#else
                        const char *str = "dnnl_verbose,create";
#endif
                        printf("%s,%s,%g\n", str, p_iface->pd()->info(), time);
                        fflush(0);
                    }
                };

        double ms = dnnl::impl::get_msec();

        // create a key for the requested primitive
        dnnl::impl::primitive_hashing::key_t key(
                pd, this->dnnl_get_max_threads());

        // lock cache
        recursive_mutex_.lock();
        dnnl::impl::primitive_iface_t *p_iface = nullptr;
        auto primitive = primitive_cache_->get(key);
        if (primitive) {
            // cache hit
            // unlock cache because it's safe to create a wrapper in parallel
            recursive_mutex_.unlock();
            // create a wrapper for primitive
            auto status = dnnl::impl::safe_ptr_assign<
                    dnnl::impl::primitive_iface_t>(p_iface,
                    new dnnl::impl::primitive_iface_t(
                            primitive, use_global_scratchpad));
            if (status != dnnl::impl::status::success) return status;

            ms = dnnl::impl::get_msec() - ms;
            print_verbose(dnnl::impl::get_verbose(), true, p_iface, ms);
            (*primitive_iface) = p_iface;
            return status;
        }

        // cache miss
        // create a requested primitive
        primitive = create_primitive();
        // create a wrapper over created primitive
        auto status
                = dnnl::impl::safe_ptr_assign<dnnl::impl::primitive_iface_t>(
                        p_iface,
                        new dnnl::impl::primitive_iface_t(
                                primitive, use_global_scratchpad));

        if (status != dnnl::impl::status::success) {
            recursive_mutex_.unlock();
            return status;
        }

        status = p_iface->init();
        if (status != dnnl::impl::status::success) {
            recursive_mutex_.unlock();
            delete p_iface;
            return status;
        }

        // update op_desc and attr pointers in the key
        key.op_desc_ = p_iface->pd()->op_desc();
        key.attr_ = p_iface->pd()->attr();

        primitive_cache_->add(key, p_iface->get_primitive());
        recursive_mutex_.unlock();

        ms = dnnl::impl::get_msec() - ms;
        print_verbose(dnnl::impl::get_verbose(), false, p_iface, ms);
        (*primitive_iface) = p_iface;
        return status;
    }

    size_t get_primitive_cache_capacity() const {
        // Default capacity is 0 - primitive cache is disabled by default
        // Use call_once to avoid performance impact due to multiple getenv
        // calls
        static size_t primitive_cache_capacity = 0;
#ifdef DNNL_ENABLE_PRIMITIVE_CACHE
        static std::once_flag initialized;
        std::call_once(initialized, [&] {
            primitive_cache_capacity = dnnl::impl::getenv_int(
                    "DNNL_PRIMITIVE_CACHE_CAPACITY", 200);
        });
#endif
        return primitive_cache_capacity;
    }

    int dnnl_get_max_threads();

protected:
    dnnl::impl::engine_kind_t kind_;
    dnnl::impl::runtime_kind_t runtime_kind_;
    std::unique_ptr<dnnl::impl::primitive_cache_t> primitive_cache_;
    // As a primitive can be created inside another one a recursive_mutex is
    // required
    std::recursive_mutex recursive_mutex_;
};

namespace dnnl {
namespace impl {

inline runtime_kind_t get_default_runtime(engine_kind_t kind) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (kind == engine_kind::gpu) return runtime_kind::ocl;
#endif
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SEQ
    return runtime_kind::seq;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    return runtime_kind::omp;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_TBB
    return runtime_kind::tbb;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    return runtime_kind::threadpool;
#else
    return runtime_kind::none;
#endif
}

inline bool is_native_runtime(runtime_kind_t kind) {
    return utils::one_of(kind, runtime_kind::seq, runtime_kind::omp,
            runtime_kind::tbb, runtime_kind::threadpool);
}

struct engine_factory_t : public c_compatible {
    virtual size_t count() const = 0;
    virtual status_t engine_create(engine_t **engine, size_t index) const = 0;
    virtual ~engine_factory_t() = default;
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
