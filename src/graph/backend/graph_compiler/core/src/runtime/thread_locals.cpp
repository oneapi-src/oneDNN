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

#include "thread_locals.hpp"
#include <list>
#include <mutex>
#include "config.hpp"
#include <runtime/context.hpp>
#ifdef SC_KERNEL_PROFILE
#include <sstream>
#include <stdio.h>
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace runtime {

// if registry is destoryed, it will be set to true
static bool registry_destroyed = false;

// the registry of all TLS resources.
struct thread_local_registry_t {
    std::mutex lock_;
    std::list<thread_local_buffer_t *> tls_buffers_;
    // release all registered TLS resources. The registry still keeps track of
    // the TLS objects
    void release(engine_t *engine) {
        std::lock_guard<std::mutex> guard(lock_);
        write_traces(tls_buffers_);
        for (auto node : tls_buffers_) {
            if (engine == nullptr || node->engine_ == engine) {
                node->main_memory_pool_.release();
                node->thread_memory_pool_.release();
                node->amx_buffer_.release(node->engine_);
                node->engine_ = nullptr;
            }
        }
    }
    ~thread_local_registry_t() { release(nullptr); }
};

struct registry_guard_t {
    std::shared_ptr<thread_local_registry_t> ptr_
            = std::make_shared<thread_local_registry_t>();
    ~registry_guard_t() { registry_destroyed = true; }
};

static const std::shared_ptr<thread_local_registry_t> &get_registry() {
    static registry_guard_t registry;
    return registry.ptr_;
}

thread_local_buffer_t::additional_t::additional_t() {
    assert(!registry_destroyed);
    registry_ = get_registry();
}

// register itself into registry
thread_local_buffer_t::thread_local_buffer_t()
    : additional_(std::unique_ptr<additional_t>(new additional_t {})) {
    auto &registry = *additional_->registry_;
    std::lock_guard<std::mutex> guard(registry.lock_);
    registry.tls_buffers_.emplace_back(this);
    cur_pos_ = registry.tls_buffers_.end();
    // cur_pos should point to the current buffer iterator in tls_buffers_
    --cur_pos_;
}

// the destructor of TLS. It will unregister `this` pointer in registry. Note
// that C++ standard makes sure that thread local objects are destroyed
// before "static lifetime" objects. However, OpenMP seems not have clearly
// specified whether/when C++11 thread_local is destructed. Experiences on
// g++ 8.4.0 shows that the destructor of thread_local objects in OpenMP threads
// are NEVER called! So we still need to check if registry has already been
// destructed
thread_local_buffer_t::~thread_local_buffer_t() {
    // C++ compiler will call ~thread_local_buffer_t() first and then call dtor
    // of its fields. Note that after ~thread_local_buffer_t() returns, the
    // lock will be released and dtors of member fields will not be protected by
    // the lock. This is OK because we have removed the reference to `this`
    // pointer from the registry and the registry has no chance to call
    // release() on `this` any more. So there will be only one thread calling
    // dtor/release() on the members at the same time
    auto &registry = *additional_->registry_;
    std::lock_guard<std::mutex> guard(registry.lock_);
    assert(*cur_pos_ = this);
    registry.tls_buffers_.erase(cur_pos_);
}

} // namespace runtime

SC_API void release_runtime_memory(runtime::engine_t *engine) {
    // in case that someone calls release_runtime_memory() after registry is
    // destroyed
    if (runtime::registry_destroyed) { return; }
    auto &registry = runtime::get_registry();
    registry->release(engine);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

extern "C" SC_API void *sc_get_tls_amx_buffer(
        dnnl::impl::graph::gc::runtime::stream_t *stream) noexcept {
    auto &tls = dnnl::impl::graph::gc::runtime::get_tls(stream);
    return &tls.amx_buffer_;
}
