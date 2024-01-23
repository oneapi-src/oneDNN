/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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

#include <algorithm>
#include <memory>
#include <mmintrin.h>
#include <string.h>
#include "dynamic_threadpool.hpp"
#include "dynamic_threadpool_c.hpp"
#include "memorypool.hpp"
#include "parallel.hpp"
#include "thread_locals.hpp"
#include <runtime/low_level_threadpool_wrapper.hpp>
#include <runtime/managed_thread_pool_exports.hpp>
#include <util/os.hpp>
#include <util/simple_math.hpp>

#ifdef _MSC_VER
#define __builtin_expect(EXP_, C) (EXP_)
#endif

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define DYN_THREAD_POOL_PROFILE 0

#if DYN_THREAD_POOL_PROFILE
#include <vector>
#include <util/scoped_timer.hpp>
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
extern void cleanup_worker_thread_state();
namespace dynamic_threadpool {

static thread_local std::unique_ptr<threadpool_scheduler> main_sched;

static thread_local struct threadlocals_t {
    threadpool_scheduler *current_sched;
    work_item_shared_data *current_work_data;
    uint64_t *cur_itr;
    // the tid of the current work should belong to, not necessarily the real
    // current thread id
    uint64_t work_tid;
    memory_pool::filo_memory_pool_t *mem_pool = nullptr;
} threadlocals;

// 64KB memory pool
static memory_pool::filo_memory_pool_t *get_mem_pool(threadlocals_t &tls) {
    if (unlikely(!tls.mem_pool)) {
        tls.mem_pool = &thread_local_buffer_t::tls_buffer()
                                .additional_->dyn_threadpool_mem_pool_;
    }
    return tls.mem_pool;
}
static threadpool_scheduler *get_current_sched() {
    return threadlocals.current_sched;
}

#if DYN_THREAD_POOL_PROFILE
static thread_local uint64_t payload_us = 0;
static thread_local uint64_t postprocess_us = 0;
static thread_local decltype(
        std::chrono::high_resolution_clock::now()) main_start_us;
static thread_local uint64_t launch_us = 0;
#endif

void *threadpool_arena::alloc(memory_pool::filo_memory_pool_t *pool,
        stream_t *stream, uint64_t size, uint64_t alignment) {
    return pool->alloc(stream, size);
}

void queue::init(uint32_t size) {
    base_ = new work_item *[size];
    size_ = size;
    thr_state_ = thread_state::UNATTENDED;
    broadcast_ob_ver_.store(0, std::memory_order_relaxed);
    fast_slot_.store(nullptr, std::memory_order_relaxed);
    lock_ = 0;
    head_and_tail_ = 0;
}

queue::~queue() {
    delete[] base_;
}

void queue::lock() noexcept {
    uint64_t expected = 0;
    while (!lock_.compare_exchange_strong(
            expected, 1, std::memory_order_acq_rel)) {
        expected = 0;
    }
}

static bool try_enqueue_fast_slot(
        std::atomic<work_item *> &fast_slot, work_item *item) {
    auto old_slot = fast_slot.load(std::memory_order_relaxed);
    if (!old_slot) {
        // if fast slot is empty
        if (fast_slot.compare_exchange_strong(
                    old_slot, item, std::memory_order_acq_rel)) {
            return true;
        }
    }
    return false;
}

bool queue::enqueue(work_item *item) noexcept {
    if (try_enqueue_fast_slot(fast_slot_, item)) { return true; }
    if (volatile_length() >= size_ - 1) { return false; }
    lock();
    // double check locking
    auto head_and_tail = head_and_tail_.load(std::memory_order_relaxed);
    auto head = head_and_tail >> 32;
    auto tail = head_and_tail & 0xFFFFFFFF;
    // head_new = (head + 1) % size_
    auto head_new = head + 1;
    if (head_new >= size_) { head_new -= size_; }
    if (unlikely(head_new == tail)) {
        lock_.store(0, std::memory_order_release);
        return false;
    }
    base_[head] = item;
    head_and_tail = (head_new << 32) | tail;
    head_and_tail_.store(head_and_tail, std::memory_order_relaxed);
    lock_.store(0, std::memory_order_release);
    return true;
}

bool queue::jump_queue(work_item *item) noexcept {
    if (try_enqueue_fast_slot(fast_slot_, item)) { return true; }
    if (unlikely(volatile_length() >= size_ - 1)) { return false; }
    lock();
    // double check locking
    auto head_and_tail = head_and_tail_.load(std::memory_order_relaxed);
    auto head = head_and_tail >> 32;
    auto tail = head_and_tail & 0xFFFFFFFF;
    auto tail_new = (tail == 0) ? (size_ - 1) : tail - 1;
    if (unlikely(tail_new == head)) {
        lock_.store(0, std::memory_order_release);
        return false;
    }
    base_[tail_new] = item;
    head_and_tail = (head << 32) | tail_new;
    head_and_tail_.store(head_and_tail, std::memory_order_relaxed);
    lock_.store(0, std::memory_order_release);
    return true;
}

work_item *queue::dequeue() noexcept {
    if (auto old_slot = fast_slot_.load(std::memory_order_relaxed)) {
        // if fast slot has a work item
        auto ret = old_slot;
        if (fast_slot_.compare_exchange_strong(
                    old_slot, nullptr, std::memory_order_acq_rel)) {
            return ret;
        }
    }
    if (volatile_length() == 0) { return nullptr; }
    lock();
    // double check locking
    auto head_and_tail = head_and_tail_.load(std::memory_order_relaxed);
    auto head = head_and_tail >> 32;
    auto tail = head_and_tail & 0xFFFFFFFF;
    if (head == tail) {
        lock_.store(0, std::memory_order_release);
        return nullptr;
    }
    auto ret = base_[tail];
    auto tail_new = tail + 1;
    if (tail_new >= size_) { tail_new -= size_; }

    head_and_tail = (head << 32) | tail_new;
    head_and_tail_.store(head_and_tail, std::memory_order_relaxed);
    lock_.store(0, std::memory_order_release);
    return ret;
}

uint64_t queue::volatile_length() noexcept {
    auto head_and_tail = head_and_tail_.load(std::memory_order_relaxed);
    auto head = head_and_tail >> 32;
    auto tail = head_and_tail & 0xFFFFFFFF;
    if (head < tail) { head += size_; }
    return head - tail;
}

threadpool_scheduler::threadpool_scheduler(
        stream_t *stream, uint64_t queue_size, uint64_t num_threads)
    : stream_(stream), num_queues_ {num_threads} {
    threadlocals.current_sched = this;
    threadlocals.work_tid = 0;
    num_broadcast_events_ = 0;
    static_assert(sizeof(queue) == 64, "expecting sizeof(queue) == 64");
    queues_ = (queue *)aligned_alloc(64, sizeof(queue) * num_threads);
    for (uint64_t i = 0; i < num_threads; i++) {
        queues_[i].init(queue_size);
    }
}

threadpool_scheduler::~threadpool_scheduler() {
    aligned_free(queues_);
}

static bool find_queue_and_insert_work(threadpool_scheduler *scheduler,
        work_item *item, uint64_t &outqid, queue *&theq) {
    for (uint64_t i = outqid; i < scheduler->num_queues_; i++) {
        if (scheduler->queues_[i].enqueue(item)) {
            theq = &scheduler->queues_[i];
            outqid = i;
            return true;
        }
    }
    for (uint64_t i = 0; i < outqid; i++) {
        if (scheduler->queues_[i].enqueue(item)) {
            theq = &scheduler->queues_[i];
            outqid = i;
            return true;
        }
    }
    return false;
}

static void deref_buffer(void *ptr) {
    auto buffer = (shared_buffer *)((uint8_t *)ptr
            - offsetof(shared_buffer, buffer_));
    auto cnt = buffer->ref_count_.fetch_sub(1, std::memory_order_acq_rel);
    assert(cnt >= 1);
    if (cnt == 1) { aligned_free(buffer); }
}

static void ref_buffer(void *ptr, uint64_t count) {
    auto buffer = (shared_buffer *)((uint8_t *)ptr
            - offsetof(shared_buffer, buffer_));
    buffer->ref_count_.fetch_add(count, std::memory_order_acq_rel);
}

static void on_work_item_done(threadpool_scheduler *scheduler,
        work_item_shared_data *shared_data, uint64_t shared_buffer_count) {
    auto &the_signal = scheduler->cur_section_.load(std::memory_order_relaxed)
                               ->pending_sink_;
    the_signal.fetch_sub(1, std::memory_order_acq_rel);
    for (uint64_t i = 0; i < shared_buffer_count; i++) {
        auto buf = shared_data->buffers_[i];
        deref_buffer(buf);
    }
}

static bool submit_single_work_item(threadpool_scheduler *scheduler,
        work_item *item, uint64_t &qid, queue *&theq,
        bool (queue::*insert_method)(work_item *)) {
    if (unlikely(item->size_ == 0)) {
        on_work_item_done(scheduler, item->data_, item->data_->buffers_.size_);
        return false;
    }
    // insert in the prefered queue
    if (likely((theq->*insert_method)(item))) { return false; }
    // insert in an non-empty queue
    if (find_queue_and_insert_work(scheduler, item, qid, theq)) { return true; }
    // fallback to slow queue
    {
        auto &fallbackq = scheduler->fallback_queue_;
        std::lock_guard<std::mutex> guard {fallbackq.lock_};
        fallbackq.size_++;
        fallbackq.queue_.push(item);
    }
    return false;
}

static void do_work(threadpool_scheduler *scheduler, threadpool_section *sect,
        void *stream, void *module_data, threadlocals_t *tls, work_item *item,
        queue *local_q) {
#if DYN_THREAD_POOL_PROFILE
    auto timer = utils::create_scoped_timer(true, [](utils::time_duration v) {
        postprocess_us
                += std::chrono::duration_cast<std::chrono::nanoseconds>(v)
                           .count();
    });
#endif
    auto shared_data = item->data_;
    tls->current_work_data = shared_data;
    auto &cur_layer = shared_data->layer_;
    auto num_levels = shared_data->num_shared_iter_ + 1;
    // prepare the iterators
    uint64_t iter[6];
    if (unlikely(num_levels > 6)) { std::abort(); }
    tls->cur_itr = iter;
    for (uint64_t i = 0; i < num_levels - 1; i++) {
        iter[i] = shared_data->shared_iter_[i];
    }
    // call the workload function
    {
        local_q->thr_state_ = queue::thread_state::RUNNING;
#if DYN_THREAD_POOL_PROFILE
        auto timer
                = utils::create_scoped_timer(true, [](utils::time_duration v) {
                      payload_us += std::chrono::duration_cast<
                              std::chrono::nanoseconds>(v)
                                            .count();
                  });
#endif
        auto base_idx = item->base_idx_;
        for (uint64_t i = 0; i < item->size_; i++) {
            iter[num_levels - 1] = base_idx + i;
            cur_layer.pfunc_(stream, module_data, iter,
                    shared_data->buffers_.ptr_, sect->args_);
        }
        local_q->thr_state_ = queue::thread_state::SCHEDULING;
    }
    on_work_item_done(scheduler, shared_data, shared_data->buffers_.size_);
    tls->current_work_data = nullptr;
}

static void try_execute_broadcast_work(threadpool_scheduler *sched,
        threadpool_section *cur_sect, uint64_t cur_tid, queue *q) {
    auto thework = sched->broadcast_work_.load(std::memory_order_acquire);
    auto exec = [](threadpool_scheduler *sched, threadpool_section *cur_sect,
                        queue *q, broadcast_work_item *thework,
                        uint64_t work_ver, uint64_t tid) {
        auto cur_bcast_ver
                = q[tid].broadcast_ob_ver_.load(std::memory_order_acquire);
        // wait until cur_bcast_ver reaches the expected value
        for (;;) {
            if (likely(cur_bcast_ver >= work_ver - 1)) { break; }
            cur_bcast_ver
                    = q[tid].broadcast_ob_ver_.load(std::memory_order_acquire);
        }
        if (cur_bcast_ver >= work_ver) { return; }
        // take the broadcast work
        if (!q[tid].broadcast_ob_ver_.compare_exchange_strong(
                    cur_bcast_ver, work_ver, std::memory_order_acq_rel)) {
            // fail to take the work due to contention, skip
            return;
        }
        if (tid < thework->loop_len_) {
            threadlocals.cur_itr = &tid;
            thework->layer_.pfunc_(sched->stream_, cur_sect->module_, &tid,
                    nullptr, cur_sect->args_);
        }
        --thework->trigger_.pending_;
    };
    if (thework) {
        auto work_ver = thework->broadcast_events_ver_;
        exec(sched, cur_sect, q, thework, work_ver, cur_tid);
        // q[cur_tid].broadcast_ob_ver_ = thework->broadcast_events_ver_;
        if (cur_tid == 0) {
            uint64_t min_unattended = 0;
            while (thework->trigger_.pending_.load(std::memory_order_acquire)) {
                for (; min_unattended < sched->num_queues_; min_unattended++) {
                    if (q[min_unattended].thr_state_
                            == queue::thread_state::UNATTENDED) {
                        exec(sched, cur_sect, q, thework, work_ver,
                                min_unattended);
                    }
                }
                _mm_pause();
            }
            // printf("MAIN release %lu\n", thework->loop_len_);
            sched->broadcast_work_.store(nullptr, std::memory_order_release);
            return;
        }
        while (sched->broadcast_work_.load(std::memory_order_acquire)
                == thework) {}
    }
}

#if DYN_THREAD_POOL_PROFILE
struct tp_profile_t {
    uint32_t count;
    uint32_t steal_count;
    uint32_t overflow;
    uint32_t total_us;
    uint32_t payload_us;
    uint32_t postprocess_us;
    uint32_t main_us;
};

struct tp_profile_holder {
    std::vector<std::vector<tp_profile_t>> vec;
    tp_profile_holder(size_t num_thr, size_t init_sz) {
        printf("WARNING: threadpool tracing is ON: threads=%zu, init_sz=%zu\n",
                num_thr, init_sz);
        vec.resize(num_thr);
        for (auto &v : vec) {
            v.reserve(init_sz);
        }
    }
    void clear() {
        for (size_t i = 0; i < vec.size(); i++) {
            auto &v = vec[i];
            uint64_t total = 0, payload = 0, postprocess = 0;
            for (size_t j = 1; j < v.size(); j++) {
                auto &trace = v[j];
                printf("tid=%zu,itr=%zu,queued=%u,steal=%u,overflow=%u,"
                       "total_us=%u,payload_us=%u,postprocess_ns=%u,main_us=%"
                       "u\n",
                        i, j, trace.count, trace.steal_count, trace.overflow,
                        trace.total_us, trace.payload_us / 1000,
                        trace.postprocess_us, trace.main_us);
                total += trace.total_us;
                payload += trace.payload_us;
                postprocess += trace.postprocess_us;
            }
            if (v.size()) {
                printf("************ payload/total=%f %lu/%lu "
                       "post=%f schedule=%f***********\n",
                        double(payload) / 1000 / total, payload / 1000, total,
                        postprocess / 1000.f, total - postprocess / 1000.f);
            }
            v.clear();
        }
    }
    void trace(int tid, uint32_t count, uint32_t steal_count, uint32_t overflow,
            uint32_t total_us, uint32_t payload_us, uint32_t post_us) {
        vec.at(tid).emplace_back(tp_profile_t {count, steal_count, overflow,
                total_us, payload_us, post_us, 0});
    }
    ~tp_profile_holder() { clear(); }
};
static tp_profile_holder trace_holder {64, 128};
#endif

static threadpool_section dummy_section {nullptr, nullptr, {0}};

void threadpool_scheduler::select_and_run_jobs(uint64_t tid) {
    auto num_threads = num_queues_;
    const auto qid = tid % num_threads;
    const auto local_q = &queues_[qid];
    local_q->thr_state_ = queue::thread_state::SCHEDULING;
    auto *tls = &threadlocals;
    auto last_steal_queue = qid;
    size_t count = 0;
    bool has_unattended_queue = true;
#if DYN_THREAD_POOL_PROFILE
    size_t overflow = 0;
    size_t steal_count = 0;
    payload_us = 0;
    postprocess_us = 0;
    auto timer = utils::create_scoped_timer(true, [&](utils::time_duration v) {
        auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(v)
                                .count();
        trace_holder.trace(tid, count, steal_count, overflow, total_us,
                payload_us, postprocess_us);
    });
#endif
    for (;;) {
        auto cur_sect = cur_section_.load(std::memory_order_acquire);
        if (!cur_sect) { continue; }
        if (cur_sect == &dummy_section) { break; }
        try_execute_broadcast_work(this, cur_sect, tid, this->queues_);
        // pick a work item from this thread's queue
        work_item *item = local_q->dequeue();
        if (likely(item)) {
            count++;
            tls->work_tid = qid;
            do_work(this, cur_sect, stream_, cur_sect->module_, tls, item,
                    local_q);
            continue;
        }
// steal jobs from other's queue
#if 1
        // only enable job stealing when the current thread has done at least 1
        // work item and there is any thread that is unattended
        if (unlikely(count > 0 && has_unattended_queue)) {
            has_unattended_queue = false;
            for (uint64_t q = 0; q < num_threads; q++) {
                auto the_q = (last_steal_queue + q) % num_threads;
                auto &victim = queues_[the_q];
#if 1
                if (victim.thr_state_ != queue::thread_state::UNATTENDED) {
                    // if the queue's main thread is present in our thread pool,
                    // and it is not busy, we expect it will take the job soon
                    continue;
                }
                has_unattended_queue = true;
#endif
                item = victim.dequeue();
                if (item) {
#if DYN_THREAD_POOL_PROFILE
                    steal_count++;
#endif
                    tls->work_tid = the_q;
                    do_work(this, cur_sect, stream_, cur_sect->module_, tls,
                            item, local_q);
                    last_steal_queue = q;
                    break;
                }
            }
        }
#endif
        if (unlikely(fallback_queue_.size_)) {
            item = nullptr;
            {
                std::lock_guard<std::mutex> guard {fallback_queue_.lock_};
                if (!fallback_queue_.queue_.empty()) {
                    fallback_queue_.size_--;
                    item = fallback_queue_.queue_.front();
                    fallback_queue_.queue_.pop();
                }
            }
            if (item) {
#if DYN_THREAD_POOL_PROFILE
                overflow++;
#endif
                tls->work_tid = 0;
                do_work(this, cur_sect, stream_, cur_sect->module_, tls, item,
                        local_q);
                continue;
            }
        }
        if (unlikely(!item && tid == 0)) {
            // no job available, check if all tasks are done
            if (cur_sect->pending_sink_.load(std::memory_order_acquire) == 0) {
                cur_section_.store(nullptr, std::memory_order_release);
                break;
            }
        }
    }
}

int64_t threadpool_adapter_t::before_parallel(threadpool_scheduler *ths) {
    return 0;
}

int64_t threadpool_adapter_t::parse_tid(std::atomic<int64_t> &v,
        threadpool_scheduler *ths, thread_local_buffer_t &tls, int64_t i) {
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
    return v++;
#else
    return i;
#endif
}

threadpool_scheduler *threadpool_adapter_t::all_thread_prepare(
        threadpool_scheduler *ths, runtime::stream_t *stream, int threads) {
    if (!main_sched || main_sched->num_queues_ != (size_t)threads) {
        main_sched = std::unique_ptr<threadpool_scheduler>(
                new threadpool_scheduler(stream, 16, threads));
    }
    get_tls(stream); // set the TLS to the current engine
    main_sched->num_broadcast_events_ = 0;
    main_sched->stream_ = stream;
    for (uint64_t i = 0; i < main_sched->num_queues_; i++) {
        main_sched->queues_[i].thr_state_ = queue::thread_state::UNATTENDED;
        main_sched->queues_[i].broadcast_ob_ver_.store(
                0, std::memory_order_relaxed);
    }
    main_sched->broadcast_work_.store(nullptr, std::memory_order_release);
    main_sched->cur_section_.store(nullptr, std::memory_order_release);
    return main_sched.get();
}
void threadpool_adapter_t::main_thread(threadpool_scheduler *sched,
        main_func_t f, runtime::stream_t *stream, void *mod_data,
        generic_val *args) {
    get_tls(stream); // set the TLS to the current engine
    threadlocals.current_sched = sched;
    threadlocals.work_tid = 0;
    sched->queues_[0].thr_state_ = queue::thread_state::SCHEDULING;
    f(stream, mod_data, args);
    sched->cur_section_.store(&dummy_section, std::memory_order_release);
    get_mem_pool(threadlocals)->clear();
    threadlocals.current_sched = nullptr;
    cleanup_worker_thread_state();
}

void threadpool_adapter_t::worker_thread(threadpool_scheduler *sched, int tid) {
    get_tls(sched->stream_); // set the TLS to the current engine
    threadlocals.current_sched = sched;
    sched->select_and_run_jobs(tid);
    get_mem_pool(threadlocals)->clear();
    threadlocals.current_sched = nullptr;
    cleanup_worker_thread_state();
}

void threadpool_adapter_t::single_thread(threadpool_scheduler *ths,
        main_func_t f, runtime::stream_t *stream, void *mod_data,
        generic_val *args) {
    std::abort();
}

void thread_main(main_func_t f, runtime::stream_t *stream, void *mod_data,
        generic_val *args) {
    call_threadpool<threadpool_adapter_t, threadpool_scheduler>(
            nullptr, f, stream, mod_data, args);
}

} // namespace dynamic_threadpool
} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

using namespace dnnl::impl::graph::gc::runtime::dynamic_threadpool;
using namespace dnnl::impl::graph::gc;

#if DYN_THREAD_POOL_PROFILE
extern "C" SC_API void sc_dyn_threadpool_print_trace() {
    trace_holder.clear();
}
#endif

extern "C" SC_API void sc_dyn_threadpool_run() {
    auto sched = get_current_sched();
    sched->select_and_run_jobs(0);
}

static thread_local threadpool_section tls_section;

extern "C" SC_API void sc_dyn_threadpool_sched_init(runtime::stream_t *stream,
        void *module, generic_val *args, uint64_t num_roots,
        uint64_t queue_size, uint64_t num_threads) {
#if DYN_THREAD_POOL_PROFILE
    main_start_us = std::chrono::high_resolution_clock::now();
#endif
    auto ret = get_current_sched();
    tls_section.args_ = args;
    tls_section.module_ = module;
    // make sure worker threads are alive before the first job submision
    tls_section.pending_sink_.store(num_roots, std::memory_order_release);
    ret->cur_section_.store(&tls_section, std::memory_order_release);
}

extern "C" SC_API void sc_dyn_threadpool_sched_destroy() {
#if DYN_THREAD_POOL_PROFILE
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - main_start_us)
                            .count();
    trace_holder.vec[0].back().main_us = total_us;
#endif
}

extern "C" SC_API work_item_shared_data *sc_dyn_threadpool_loop_end(
        work_item_shared_data *current, uint64_t up_levels) {
    if (!current) { current = threadlocals.current_work_data; }
    for (uint64_t i = 0; i < up_levels; i++) {
        current = current->parent_;
    }
    auto updated_count = current->trigger_.pending_.fetch_sub(
            1, std::memory_order_acq_rel);
    assert(updated_count >= 1);
    if (unlikely(updated_count == 1)) { return current; }
    return nullptr;
}

static bool submit_bcast_work(threadpool_scheduler *sched,
        memory_pool::filo_memory_pool_t *pool, closure_t pfunc,
        uint64_t loop_len, uint64_t flags) {
    auto old_bcast = sched->broadcast_work_.load(std::memory_order_acquire);
    if (old_bcast) { return false; }
    auto item = threadpool_arena::alloc_vsize<broadcast_work_item, uint64_t>(
            pool, sched->stream_, 0);
    ++sched->num_broadcast_events_;
    item->broadcast_events_ver_ = sched->num_broadcast_events_;
    item->layer_.pfunc_ = pfunc;
    item->loop_len_ = loop_len;
    item->trigger_.pending_.store(
            sched->num_queues_, std::memory_order_relaxed);

    sched->broadcast_work_.store(item, std::memory_order_release);

    bool is_root = flags & work_item_flags::is_root;
    if (unlikely(is_root)) {
        // notify other threads that the first submission is done
        sched->cur_section_.load(std::memory_order_relaxed)
                ->pending_sink_.fetch_sub(1, std::memory_order_acq_rel);
    }
    // printf("MAIN submit %lu\n", item->broadcast_events_ver_);
    return true;
}

extern "C" SC_API void sc_dyn_threadpool_create_work_items(closure_t pfunc,
        uint64_t *iter, uint64_t num_iter, uint64_t loop_len,
        uint64_t num_blocks, uint64_t outer_loop_hash, uint64_t num_buffers,
        void **buffers, uint64_t flags) {
    auto &tls = threadlocals;
    auto sched = tls.current_sched;
    auto work_tid = tls.work_tid;
    auto parent = tls.current_work_data;
    auto pool = get_mem_pool(tls);
    if (num_iter == 0 && num_blocks == sched->num_queues_
            && loop_len <= num_blocks && num_buffers == 0 && loop_len >= 4) {
        // it is a broadcast job
        if (likely(submit_bcast_work(sched, pool, pfunc, loop_len, flags))) {
            return;
        }
    }
    auto shared
            = threadpool_arena::alloc_vsize<work_item_shared_data, uint64_t>(
                    pool, sched->stream_, num_iter);
    shared->parent_ = parent;
    shared->layer_.pfunc_ = pfunc;
    shared->buffers_.init(pool, sched->stream_, num_buffers);
    if (unlikely(!buffers && num_buffers)) { buffers = parent->buffers_.ptr_; }
    for (uint64_t i = 0; i < num_buffers; i++) {
        shared->buffers_[i] = buffers[i];
        ref_buffer(buffers[i], num_blocks);
    }
    shared->trigger_.pending_.store(loop_len, std::memory_order_release);
    shared->num_shared_iter_ = num_iter;
    if (likely(iter == nullptr)) { iter = tls.cur_itr; }
    for (uint64_t i = 0; i < num_iter; i++) {
        shared->shared_iter_[i] = iter[i];
    }
    sched->cur_section_.load(std::memory_order_relaxed)
            ->pending_sink_.fetch_add(
                    std::min(loop_len, num_blocks), std::memory_order_acq_rel);
    bool bind_last_level = flags & work_item_flags::bind_last_level;
    uint64_t thread_id_step = flags & work_item_flags::thread_id_step_mask;
    uint64_t thread_block_size = num_blocks * thread_id_step;
    uint64_t base_tid = work_tid / thread_block_size * thread_block_size;
    auto tp_num_threads = sched->num_queues_;
    auto items = (work_item *)threadpool_arena::alloc(
            pool, sched->stream_, sizeof(work_item) * num_blocks, 64);
    if (likely(loop_len <= num_blocks)) {
        for (uint64_t tid = 0; tid < loop_len; tid++) {
            auto item = &items[tid];
            item->base_idx_ = tid;
            item->size_ = 1;
            auto suggested_queue = base_tid + tid * thread_id_step;
            item->data_ = shared;
            auto qid = suggested_queue % tp_num_threads;
            auto q = &sched->queues_[qid];
            submit_single_work_item(sched, item, qid, q, &queue::jump_queue);
        }
    } else {
        uint64_t end = loop_len;
        uint64_t begin = 0;
        uint64_t step = 1;
        uint64_t len = end - begin;
        uint64_t num_jobs = len;
        uint64_t my_jobs = utils::divide_and_ceil(num_jobs, num_blocks);
        assert(my_jobs > 0);
        uint64_t my_jobs_2 = my_jobs - 1;
        uint64_t the_tid = num_jobs - my_jobs_2 * num_blocks;
        for (uint64_t tid = 0; tid < num_blocks; tid++) {
            uint64_t cur_jobs = tid < the_tid ? my_jobs : my_jobs_2;
            uint64_t my_begin = tid <= the_tid
                    ? tid * my_jobs
                    : the_tid * my_jobs + (tid - the_tid) * my_jobs_2;
            my_begin = my_begin * step + begin;
            auto item = &items[tid];
            item->base_idx_ = my_begin;
            item->size_ = cur_jobs;
            auto suggested_queue = base_tid + tid * thread_id_step;
            item->data_ = shared;

            auto qid = suggested_queue % tp_num_threads;
            auto q = &sched->queues_[qid];
            submit_single_work_item(sched, item, qid, q, &queue::jump_queue);
        }
    }
    bool is_root = flags & work_item_flags::is_root;
    if (unlikely(is_root)) {
        // notify other threads that the first submission is done
        sched->cur_section_.load(std::memory_order_relaxed)
                ->pending_sink_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

extern "C" SC_API void *sc_dyn_threadpool_shared_buffer(uint64_t size) {
    auto ret = (shared_buffer *)aligned_alloc(64, sizeof(shared_buffer) + size);
    ret->size_ = size;
    ret->ref_count_.store(0, std::memory_order_release);
    return &ret->buffer_[0];
}
