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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_THREADPOOL_C_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_THREADPOOL_C_HPP

#include <stdint.h>
#include "context.hpp"
#include <runtime/generic_val.hpp>
#include <util/def.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
namespace dynamic_threadpool {

using closure_t = void (*)(void *stream, void *mod_data, uint64_t *itr,
        void **shared_buffers, generic_val *);
struct threadpool_scheduler;
struct work_item_shared_data;

namespace work_item_flags {
enum flags : uint64_t {
    is_root = (1ULL << 32),
    bind_last_level = (1ULL << 33),
    thread_id_step_mask = (1ULL << 32) - 1
};
}

using main_func_t = void (*)(stream_t *, void *, generic_val *);
void thread_main(
        main_func_t f, stream_t *stream, void *mod_data, generic_val *args);

} // namespace dynamic_threadpool
} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

/**
 * initialize an threadpool_scheduler
 * @param sched the memory buffer for the scheduler
 * @param stream the runtime stream object
 * @param arena_size the memory allocator initial size (bytes)
 * @param queue_size the number of workitems for each per-thread queues
 * @param num_threads the expected number of threads
 */
extern "C" SC_API void sc_dyn_threadpool_sched_init(
        dnnl::impl::graph::gc::runtime::stream_t *stream, void *module_data,
        dnnl::impl::graph::gc::generic_val *args, uint64_t num_roots,
        uint64_t queue_size, uint64_t num_threads);

/**
 * Destory the threadpool_scheduler.
 */
extern "C" SC_API void sc_dyn_threadpool_sched_destroy();

extern "C" SC_API void *sc_dyn_threadpool_shared_buffer(uint64_t size);

extern "C" SC_API void sc_dyn_threadpool_create_work_items(
        dnnl::impl::graph::gc::runtime::dynamic_threadpool::closure_t pfunc,
        uint64_t *iter, uint64_t num_iter, uint64_t loop_len,
        uint64_t num_blocks, uint64_t outer_loop_hash, uint64_t num_buffers,
        void **buffers, uint64_t flags);

extern "C" SC_API dnnl::impl::graph::gc::runtime::dynamic_threadpool::
        work_item_shared_data *
        sc_dyn_threadpool_loop_end(
                dnnl::impl::graph::gc::runtime::dynamic_threadpool::
                        work_item_shared_data *current,
                uint64_t up_levels);

extern "C" SC_API void sc_dyn_threadpool_run();

#endif
