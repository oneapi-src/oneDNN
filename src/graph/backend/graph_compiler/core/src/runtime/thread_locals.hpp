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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_THREAD_LOCALS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_THREAD_LOCALS_HPP
#include <assert.h>
#include <list>
#include <memory>
#include <vector>
#include "context.hpp"
#include "memorypool.hpp"
#include "trace.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

// the thread-local AMX-related data
struct amx_buffer_t {
    // AMX scratch pad
    void *ptr_ = nullptr;
    // the pointer to the current configured palette
    const char *cur_palette = nullptr;
    // if there is an alive tile palette
    bool need_release_tile_ = false;
    // alloc memory for AMX scratch pad at ptr_ using the allocator in the
    // stream(engine)
    void reset(stream_t *stream);
    // release memory for AMX scratch pad at ptr_. Destructor of this class will
    // not call this function. It must be manually called.
    void release(engine_t *engine);
};

struct thread_local_registry_t;
// a container for thread local resources. Users can call
// release_runtime_memory to manually release all thread local memory
// managed by this struct
struct thread_local_buffer_t {
    // the additional thread local data. Referenced via a pointer in
    // thread_local_buffer_t to reduce TLS size and improve performance
    struct additional_t {
        int linear_thread_id_ = 0;
        int instance_id_ = 0;
        bool is_main_thread_ = false;
        memory_pool::filo_memory_pool_t dyn_threadpool_mem_pool_ {4096 * 16};
        trace_manager_t trace_;
        // the pointer to keep registry alive
        std::shared_ptr<thread_local_registry_t> registry_;
        additional_t();
    };
    bool in_managed_thread_pool_ = false;
    engine_t *engine_ = nullptr;
    amx_buffer_t amx_buffer_;

    // if the current thread is the "main" thread, use this pool
    memory_pool::filo_memory_pool_t main_memory_pool_ {
            memory_pool::main_chunk_size};
    // if the current thread is a worker thread, use this pool
    memory_pool::filo_memory_pool_t thread_memory_pool_ {
            memory_pool::threadlocal_chunk_size};

    std::unique_ptr<additional_t> additional_;

    ~thread_local_buffer_t();
    using list_type = std::list<thread_local_buffer_t *>;

    static inline thread_local_buffer_t &tls_buffer() {
        static thread_local thread_local_buffer_t tls_buffer_;
        return tls_buffer_;
    }

    // disable move and copy
    thread_local_buffer_t(const thread_local_buffer_t &) = delete;
    thread_local_buffer_t(thread_local_buffer_t &&) = delete;

    thread_local_buffer_t &operator=(const thread_local_buffer_t &) = delete;
    thread_local_buffer_t &operator=(thread_local_buffer_t &&) = delete;

private:
    friend struct thread_local_registry_t;
    // private ctor makes sure this struct can only be used in TLS
    thread_local_buffer_t();
    // the current position in thread_local_registry
    list_type::iterator cur_pos_;
};

// gets the Thread Local Storage associated with the stream. Note that we assume
// that a thread will be attached to one stream when the thread runs a kernel at
// the first time and it will not switch between streams at the run time. We
// also have the same assumption on the "main" thread which invokes the main
// entry of the kernel
inline thread_local_buffer_t &get_tls(runtime::stream_t *stream) {
    auto &ret = thread_local_buffer_t::tls_buffer();
    assert(ret.engine_ == nullptr || ret.engine_ == stream->engine_);
    ret.engine_ = stream->engine_;
    return ret;
}

const std::shared_ptr<thread_local_registry_t> &get_thread_locals_registry();
} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

extern "C" SC_API void *sc_get_tls_amx_buffer(
        dnnl::impl::graph::gc::runtime::stream_t *stream) noexcept;

#endif
