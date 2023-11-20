/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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
#include <memory>

#include <atomic>
#include <future>
#include <memory>
#include "compiler_allocator.hpp"
#include "compiler_backend.hpp"
#include "runtime/const_cache_wrapper.hpp"
#include "runtime/runtime.hpp"

#define ALLOCATOR_ALIGNMENT 64
namespace dnnl {
namespace impl {
namespace graph {

namespace gc {
namespace runtime {

static std::atomic<uint64_t> const_count {0};

struct gc_constant_buffer_t : public constant_buffer_t {
    std::weak_ptr<const_cache_proxy> proxy_;
    // we need to remember the backend id, because we don't know when
    // compiler_impl::compiler_backend_t::get_singleton() is destryed
    constant_tensor_cache_t::key_t backend_id_;
    constant_tensor_cache_t::key_t id_;
    compiler_impl::compiler_graph_engine_t *gcengine_;

    static void *malloc_func(size_t sz, impl::engine_t *, allocator_t *alloc) {
        return alloc->allocate(
                sz, {allocator_t::mem_type_t::persistent, ALLOCATOR_ALIGNMENT});
    }
    static void free_func(void *p, impl::engine_t *, allocator_t *alloc) {
        alloc->deallocate(p);
    }

    gc_constant_buffer_t(
            compiler_impl::compiler_graph_engine_t *engine, size_t size)
        : constant_buffer_t {size, engine->engine_,
                static_cast<allocator_t *>(engine->engine_->get_allocator()),
                malloc_func, free_func}
        , backend_id_ {compiler_impl::compiler_backend_t::get_singleton()
                               .get_id()}
        , id_ {++const_count}
        , gcengine_ {engine} {}

    void notify_evict() override {
        // when the cache is evicted from the constant_tensor_cache, notify
        // the GC runtime that it should not be used
        auto p = proxy_.lock();
        if (p) { p->deref(); }
    }
};

// this function will be used in GC runtime
static void destroy_const_cache(const_cache_proxy *cache) {
    // when the const_cache_proxy expires in GC side, all compiled JIT function
    // referencing this cache is destroyed. This function will be
    // called to remove the cache from constant_tensor_cache.
    if (cache->check_alive_and_ref()) {
        // if the cache has not been evicted
        auto ret = reinterpret_cast<gc_constant_buffer_t *>(
                cache->unsafe_get_ptr());
        ret->gcengine_->cache_->remove_if_exist(ret->backend_id_, ret->id_);
    }
    delete cache;
}

// this function will be used in GC
static std::shared_ptr<const_cache_proxy> do_create_and_register_const_cache(
        dnnl::impl::graph::gc::runtime::engine_t *engine, size_t size) {
    auto eng = static_cast<compiler_impl::compiler_graph_engine_t *>(engine);
    // make a constant_buffer_t and alloc memory
    auto buffer_obj = std::make_shared<gc_constant_buffer_t>(eng, size);

    // create the proxy object
    auto ret = std::shared_ptr<const_cache_proxy>(
            new const_cache_proxy {
                    buffer_obj, buffer_obj->data<void>(), size, true},
            destroy_const_cache);
    // link the proxy object with constant_buffer_t to allow eviction
    buffer_obj->proxy_ = ret;

    // register into constant_tensor_cache_t
    std::promise<constant_tensor_cache_t::cached_t> fut {};
    fut.set_value(buffer_obj);
    eng->cache_->get_or_add(
            buffer_obj->backend_id_, buffer_obj->id_, size, fut.get_future());

    return ret;
}

} // namespace runtime
} // namespace gc

namespace compiler_impl {

using namespace gc::runtime;

using sc_engine_t = gc::runtime::engine_t;

compiler_graph_engine_t::compiler_graph_engine_t(
        gc::runtime::engine_vtable_t *vtable, graph::engine_t *engine,
        const std::shared_ptr<engine_ref_data> &engine_ref_data_ptr)
    : gc::runtime::engine_t {vtable}
    , engine_ {engine}
    , cache_ {get_constant_tensor_cache(engine_->kind(), engine_->index())}
    , engine_ref_data_ptr_ {engine_ref_data_ptr} {
    engine_->retain();
    cache_->retain();
}

compiler_graph_engine_t::~compiler_graph_engine_t() {
    std::lock_guard<std::mutex> lock(engine_ref_data_ptr_->global_mutex_);
    gc::release_runtime_memory(this);
    for (auto iter = engine_ref_data_ptr_->engine_map_.begin();
            iter != engine_ref_data_ptr_->engine_map_.end();) {
        if (iter->second.lock() == nullptr) {
            iter = engine_ref_data_ptr_->engine_map_.erase(iter);
        } else {
            ++iter;
        }
    }
    cache_->release();
    engine_->release();
}

static void *compiler_graph_global_alloc(sc_engine_t *eng, size_t sz) {
    allocator_t *alloc = static_cast<allocator_t *>(
            static_cast<compiler_graph_engine_t *>(eng)
                    ->engine_->get_allocator());
    return alloc->allocate(
            sz, {allocator_t::mem_type_t::persistent, ALLOCATOR_ALIGNMENT});
}

static void compiler_graph_global_free(sc_engine_t *eng, void *p) {
    allocator_t *alloc = static_cast<allocator_t *>(
            static_cast<compiler_graph_engine_t *>(eng)
                    ->engine_->get_allocator());
    alloc->deallocate(p);
}

#if 0
static void *compiler_graph_temp_alloc(sc_engine_t *eng, size_t sz) {
    return static_cast<compiler_graph_engine_t *>(eng)->allocator_->allocate(sz,
            {allocator_t::mem_type_t::temp, ALLOCATOR_ALIGNMENT});
}

static void compiler_graph_temp_free(sc_engine_t *eng, void *p) {
    static_cast<compiler_graph_engine_t *>(eng)->allocator_->deallocate(p);
}
#endif

static size_t get_engine_tensor_cache_cap(sc_engine_t *engine) {
    auto eng = static_cast<compiler_impl::compiler_graph_engine_t *>(engine);
    return eng->cache_->get_capacity();
}

engine_vtable_t graph_engine_vtable {compiler_graph_global_alloc,
        compiler_graph_global_free, compiler_graph_global_alloc,
        compiler_graph_global_free,
        gc::runtime::do_create_and_register_const_cache,
        get_engine_tensor_cache_cap};

compiler_graph_stream_t::compiler_graph_stream_t(
        compiler_graph_engine_t *eng, const dnnl_stream *stream)
    : gc::runtime::stream_t {
            {sc_parallel_call_cpu_with_env_impl, stream}, eng} {}

} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
