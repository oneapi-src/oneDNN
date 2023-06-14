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

#include <atomic>
#include <memory>
#include <stdint.h>
#include <runtime/const_cache_wrapper.hpp>
#include <runtime/memorypool.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

const_cache_proxy::~const_cache_proxy() {
    if (is_lazy_) { unregister_const_cache(this); }
}

void *const_cache_proxy::acquire(int32_t *inited) {
    if (check_alive_and_ref()) {
        *inited = *inited && initialized_;
        return buffer_;
    }
    return nullptr;
}

bool const_cache_proxy::release() {
    if (is_alive()) {
        deref();
        initialized_ = 1;
        return true;
    }
    return false;
}

std::shared_ptr<const_cache_proxy> create_and_register_const_cache(
        dnnl::impl::graph::gc::runtime::engine_t *engine, size_t size) {
    // simply allocate buffer and return
    std::shared_ptr<void> base = std::shared_ptr<void> {
            engine->vtable_->persistent_alloc(engine, size), [engine](void *p) {
                engine->vtable_->persistent_dealloc(engine, p);
            }};
    return std::make_shared<const_cache_proxy>(base, base.get(), size, true);
}

void unregister_const_cache(const_cache_proxy *cache) {
    // currently do nothing
}

} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

extern "C" SC_API void *sc_acquire_const_cache(
        dnnl::impl::graph::gc::runtime::stream_t *stream,
        dnnl::impl::graph::gc::runtime::const_cache_proxy *cacheptr,
        size_t size, int32_t *inited) {
    if (auto buf = cacheptr->acquire(inited)) { return buf; }
    *inited = 0;
    return sc_aligned_malloc(stream, size);
}
extern "C" SC_API void sc_release_const_cache(
        dnnl::impl::graph::gc::runtime::stream_t *stream,
        dnnl::impl::graph::gc::runtime::const_cache_proxy *cacheptr,
        void *ptr) {
    if (cacheptr->release()) { return; }
    return sc_aligned_free(stream, ptr);
}
