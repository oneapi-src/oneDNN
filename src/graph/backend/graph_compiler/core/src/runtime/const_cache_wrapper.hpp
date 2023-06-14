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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_CONST_CACHE_WRAPPER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_CONST_CACHE_WRAPPER_HPP
#include <atomic>
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <runtime/context.hpp>
#include <util/def.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

/**
 * The helper class to manage ref count manually for an object allocated with
 * shared ptr. It holds an additional shared ptr reference to the object and
 * contains an additional self-managed refcount. The refcount will be set to 1
 * when the object is initialized (see init()). When the refcount counts down to
 * 0, the additional shared ptr is reset.
 */
struct ref_count_managed {
    ref_count_managed() = default;
    ref_count_managed(const std::shared_ptr<void> &keep_alive) {
        init(keep_alive);
    }
    void init(const std::shared_ptr<void> &keep_alive) {
        keep_alive_ = keep_alive;
        ref_count_.store(1);
    }

    void ref() { ++ref_count_; }
    void deref() {
        auto newv = --ref_count_;
        if (newv == 0) { keep_alive_ = nullptr; }
    }

    // atomically check if ref_count_ > 0. if so, ref() the object and return
    // true. Otherwise (if ref_count_==0), return false
    bool check_alive_and_ref() {
        auto oldv = ref_count_.load();
        for (;;) {
            if (oldv <= 0) { return false; }
            if (ref_count_.compare_exchange_strong(oldv, oldv + 1)) {
                return true;
            }
            // CAS failed, oldv has now the newest known value of ref_count_
        }
    }

    bool is_alive() const { return ref_count_ > 0; }

private:
    std::shared_ptr<void> keep_alive_;
    std::atomic<int> ref_count_ {0};
};

/**
 * The proxy for the constant cache of Graph API. It holds a shared ptr pointing
 * to the cache item in the cache manager (keep_alive) to extend the lifetime by
 * refcount, @see ref_count_managed. To access the memory buffer of the const
 * cache, use sc_acquire_const_cache and sc_release_const_cache functions. They
 * will ref/deref the const_cache_proxy to make sure the cache is alive after
 * calling sc_acquire_const_cache and before sc_release_const_cache. The cache
 * manager of Graph API may evict the cache item by dereferenceing this
 * ref_count_managed object. sc_{acquire,release}_const_cache functions will
 * find out that the cache has been invalidated and they will then use the
 * memory allocator in the runtime::stream_t to re-allocate the buffer. Usually
 * we expect JIT modules to hold shared ptr to const_cache_proxy via
 * cached_const_graph_tensor.
 * If is_lazy_ == true, the cache item's lifetime will be managed by the cache
 * manager of Graph API and it is filled with data after the first execution of
 * the computation. Otherwise, the cache item is always alive as long as the
 * jit_module of the kernel is alive.
 */
struct const_cache_proxy : ref_count_managed {
    const_cache_proxy(const std::shared_ptr<void> &keep_alive, void *buffer,
            size_t size, bool is_lazy)
        : ref_count_managed(keep_alive)
        , size_(size)
        , is_lazy_(is_lazy)
        , buffer_(buffer) {}
    ~const_cache_proxy();

    // get the buffer and increment the refcount. If the buffer is evicted,
    // returns null
    void *acquire(int32_t *inited);
    // decrement the refcount
    bool release();

    void *get_buffer_if_not_lazy() const {
        if (is_lazy_) {
            throw std::runtime_error(
                    "get_buffer_if_not_lazy: The buffer must be lazy");
        }
        return buffer_;
    }

    size_t size_;
    // if the buffer is lazy-initialized. If false, it should be filled before
    // computation
    bool is_lazy_;

private:
    // raw pointer to the buffer
    void *buffer_;
    // if the buffer has been initialized. calling release() will set this to 1
    int32_t initialized_ = 0;
};

// allocate the const cache buffer and register it to Graph API cache manager
std::shared_ptr<const_cache_proxy> create_and_register_const_cache(
        dnnl::impl::graph::gc::runtime::engine_t *engine, size_t size);

// unregister it in Graph API cache manager
void unregister_const_cache(const_cache_proxy *cache);

} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

// acquire the cached constant buffer pointer. If the cache has been evicted,
// this function will allocate a new buffer and set *inited to 0. If the cache
// is still alive, this function will increment the refcount of the buffer to
// keep it alive, and set `*inited = cacheptr->initialized_ & *inited`
extern "C" SC_API void *sc_acquire_const_cache(
        dnnl::impl::graph::gc::runtime::stream_t *stream,
        dnnl::impl::graph::gc::runtime::const_cache_proxy *cacheptr,
        size_t size, int32_t *inited);
// release the cached constant buffer pointer. If the cache has been evicted,
// this function will free the newly allocated buffer. If the cache
// is still alive, this function will decrement the refcount of the buffer
extern "C" SC_API void sc_release_const_cache(
        dnnl::impl::graph::gc::runtime::stream_t *stream,
        dnnl::impl::graph::gc::runtime::const_cache_proxy *cacheptr, void *ptr);

#endif
