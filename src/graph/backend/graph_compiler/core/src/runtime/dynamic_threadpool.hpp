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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_THREADPOOL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_THREADPOOL_HPP

#include <atomic>
#include <cassert>
#include <mutex>
#include <queue>
#include <stdint.h>
#include <vector>
#include "context.hpp"
#include <util/compiler_macros.hpp>
#include <util/def.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace memory_pool {
struct filo_memory_pool_t;
}
namespace runtime {
struct thread_local_buffer_t;

namespace dynamic_threadpool {

using closure_t = void (*)(void *stream, void *mod_data, uint64_t *itr,
        void **shared_buffers, generic_val *);

enum class schedule_policy : uint64_t {
    current_layer_first,
    next_layer_first,
};

struct threadpool_arena {
    static void *alloc(memory_pool::filo_memory_pool_t *, stream_t *s,
            uint64_t size, uint64_t alignment);
    // allocate variable sized struct
    template <typename T, typename TArray>
    static T *alloc_vsize(memory_pool::filo_memory_pool_t *pool, stream_t *s,
            uint64_t len_arr) {
        uint64_t size = sizeof(T) + sizeof(TArray) * len_arr;
        return (T *)alloc(pool, s, size, alignof(T));
    }
};

template <typename T>
struct arena_vector {
    T *ptr_;
    uint64_t size_;
    T &operator[](uint64_t idx) {
        assert(idx < size_);
        return ptr_[idx];
    }

    void init_empty() {
        size_ = 0;
        ptr_ = nullptr;
    }
    void init(
            memory_pool::filo_memory_pool_t *pool, stream_t *s, uint64_t size) {
        if (size > 0) {
            ptr_ = reinterpret_cast<T *>(threadpool_arena::alloc(
                    pool, s, sizeof(T) * size, alignof(T)));
        }
        size_ = size;
    }
};

/**
 * A layer of the threadpool represents a pure nested parallel for. "Pure" here
 * means the parallel fors are tightly nested without any statements between
 * each level. A "task" represents an instance of parallel-for. If a
 * parallel-for is executed N times by an outer parallel-loop, there will be N
 * "tasks" for this level of parallel-for
 */
struct threadpool_layer {
    closure_t pfunc_;
};

struct trigger;

struct shared_buffer {
    std::atomic<uint64_t> ref_count_;
    uint64_t size_;
    alignas(64) char buffer_[0];
};

struct trigger {
    // the count of pending subtasks within this task
    alignas(64) std::atomic<int64_t> pending_;
};

struct work_item_shared_data {
    work_item_shared_data *parent_;
    threadpool_layer layer_;
    arena_vector<void *> buffers_;
    trigger trigger_;
    uint64_t num_shared_iter_;
    uint64_t shared_iter_[0];
};

struct work_item {
    work_item_shared_data *data_;
    // the last-level iterator idx
    uint64_t base_idx_;
    uint64_t size_;
};

struct queue {
    // the lock to protect head_and_tail_ and base_
    alignas(64) std::atomic<uint64_t> lock_;
    // lock-free fast slot for single work item
    std::atomic<work_item *> fast_slot_;
    // the newest version of broadcast work item that this queue(thread) has
    // seen
    std::atomic<uint64_t> broadcast_ob_ver_;
    work_item **base_;
    uint32_t size_;
    enum class thread_state : uint32_t {
        // The expected thread-id is never executed. It may occur in the case of
        // customized thread-pool, when a specific thread is busy for other jobs
        // unrelated to our kernel
        UNATTENDED,
        // The thread is scheduling/waiting on the task
        SCHEDULING,
        // The thread is working on a payload
        RUNNING,
    } thr_state_;
    // head: the next position to insert. It should point to an empty position
    // tail: the next position to dequeue. If head!=tail, it should point to a
    // valid item
    std::atomic<uint64_t> head_and_tail_;
    bool enqueue(work_item *item) noexcept;
    bool jump_queue(work_item *item) noexcept;
    void lock() noexcept;
    work_item *dequeue() noexcept;
    uint64_t volatile_length() noexcept;
    queue() = default;
    queue(const queue &) = delete;
    queue(queue &&) = delete;
    void init(uint32_t size);
    ~queue();
};

struct fallback_queue {
    std::mutex lock_;
    uint64_t size_ = 0;
    std::queue<work_item *> queue_;
};

struct threadpool_section {
    void *module_;
    generic_val *args_;
    // the pending sink jobs, if it counts down to 0, all jobs are done
    std::atomic<uint64_t> pending_sink_;
};

struct broadcast_work_item {
    threadpool_layer layer_;
    trigger trigger_;
    uint64_t loop_len_;
    uint64_t broadcast_events_ver_;
};

struct threadpool_scheduler {
    stream_t *stream_;
    // per-thread queue
    queue *queues_;
    uint64_t num_queues_;
    // shared queue when per-thread queue is full
    fallback_queue fallback_queue_;
    // only used by the main thread
    uint64_t num_broadcast_events_;
    std::atomic<broadcast_work_item *> broadcast_work_;
    std::atomic<threadpool_section *> cur_section_;
    threadpool_scheduler(
            stream_t *stream, uint64_t queue_size, uint64_t num_threads);
    ~threadpool_scheduler();
    void select_and_run_jobs(uint64_t tid);
};

using main_func_t = void (*)(stream_t *, void *, generic_val *);
struct threadpool_adapter_t {
    static constexpr bool can_optimize_single_thread = false;
    using TyState = std::atomic<int64_t>;

    static threadpool_scheduler *all_thread_prepare(
            threadpool_scheduler *ths, runtime::stream_t *stream, int threads);
    static void main_thread(threadpool_scheduler *sched, main_func_t f,
            runtime::stream_t *stream, void *mod_data, generic_val *args);

    static void worker_thread(threadpool_scheduler *sched, int tid);

    static void after_parallel(threadpool_scheduler *ths) {}

    static void single_thread(threadpool_scheduler *ths, main_func_t f,
            runtime::stream_t *stream, void *mod_data, generic_val *args);
    static int64_t before_parallel(threadpool_scheduler *ths);
    static int64_t parse_tid(std::atomic<int64_t> &v, threadpool_scheduler *ths,
            thread_local_buffer_t &tls, int64_t i);
};

} // namespace dynamic_threadpool
} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
