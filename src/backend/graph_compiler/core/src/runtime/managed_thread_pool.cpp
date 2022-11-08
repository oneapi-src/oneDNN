/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#include <atomic>
#ifdef SC_KERNEL_PROFILE
#include <chrono>
#endif
#include <immintrin.h>
#include "config.hpp"
#include "managed_thread_pool.hpp"
#include "memorypool.hpp"
#include "runtime.hpp"
#include "thread_locals.hpp"
#include "thread_pool_flags.hpp"
#include <cpu/x64/amx_tile_configure.hpp>
#include <runtime/microkernel/cpu/kernel_timer.hpp>
#include <util/simple_math.hpp>

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
// clang-format off
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/parallel_for_each.h>
// clang-format on
#endif

using namespace sc;
using sc::runtime::thread_manager;
static void do_dispatch(thread_manager *s, int tid);
namespace sc {
namespace runtime {
#ifdef SC_KERNEL_PROFILE
static void make_trace(int in_or_out, int count) {
    if (sc_is_trace_enabled()) { sc_make_trace_kernel(2, in_or_out, count); }
}

#else
#define make_trace(v, count) SC_UNUSED(count)
#endif

constexpr uint64_t max_wait_count = 1;

void thread_manager::thread_pool_state::wait_all() {
    make_trace(0, 0);
    auto idl_f = idle_func;
    bool has_idle_func = idl_f
            && (execution_flags & thread_pool_flags::THREAD_POOL_RUN_IDLE_FUNC);
    int count = 0;
    if (has_idle_func && remaining.load(std::memory_order_acquire) != 0) {
        count = idl_f(&remaining, 0, 0, idle_args);
    }
    for (;;) {
        if (remaining.load(std::memory_order_acquire) == 0) {
            make_trace(1, count);
            break;
        }
        _mm_pause();
    }
}

void thread_manager::thread_pool_state::reset_scoreboard() {
    remaining.store(num_threads - 1, std::memory_order_release);
}

#ifdef SC_KERNEL_PROFILE
static std::atomic<int> instances {0};
#endif
thread_manager::thread_manager() {
    state.trigger = 1;
#ifdef SC_KERNEL_PROFILE
    instance_id_ = instances++;
#endif
}

static void do_cleanup() {
    auto &tls = thread_local_buffer_t::tls_buffer_;
    tls.in_managed_thread_pool_ = false;
    auto &need_release_amx = tls.amx_buffer_.need_release_tile_;
    if (need_release_amx) {
        dnnl::impl::cpu::x64::amx_tile_release();
        need_release_amx = false;
        // force to re-configure next time
        tls.amx_buffer_.cur_palette = nullptr;
    }
}

static void worker_func(thread_manager *ths, int tid) {
    int st;
    auto &task = ths->state.task;
    int current_job_id = 2;

    while (true) {
        auto idl_f = ths->state.idle_func;
        bool has_idle_func = idl_f
                && (ths->state.execution_flags
                        & thread_pool_flags::THREAD_POOL_RUN_IDLE_FUNC);
        int count = 0;
        if (has_idle_func && (st = ths->state.trigger) != current_job_id) {
            count = idl_f(&ths->state.trigger, current_job_id, tid,
                    ths->state.idle_args);
        }
        while ((st = ths->state.trigger.load(std::memory_order_relaxed))
                != current_job_id) {
            if (st == -1) {
                do_cleanup();
                make_trace(1, count);
                return;
            }
            _mm_pause();
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        make_trace(1, count);
        do_dispatch(ths, tid);
        make_trace(0, 0);
        --ths->state.remaining;
        current_job_id++;
    }
}

void thread_manager::run_main_function(main_func_t f, runtime::stream_t *stream,
        void *mod_data, generic_val *args) {
    int threads = runtime_config_t::get().get_num_threads();
    state.num_threads = threads;
    if (threads > 1) {
        state.trigger = 1;
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
#pragma omp parallel for
        for (int i = 0; i < threads; i++) {
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
        tbb::parallel_for(0, threads, 1, [&](int64_t i) {
#endif
            auto &tls = thread_local_buffer_t::tls_buffer_;
            tls.in_managed_thread_pool_ = true;
            tls.additional_->linear_thread_id_ = i;
#ifdef SC_KERNEL_PROFILE
            tls.additional_->instance_id_ = instance_id_;
#endif
            if (i == 0) {
                f(stream, mod_data, args);
                state.trigger = -1;
                do_cleanup();
            } else {
                worker_func(this, i);
            }
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
        }
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
        });
#endif

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_SEQ
        throw std::runtime_error("Running SEQ in thread pool");
#endif
    } else {
        auto &tls = thread_local_buffer_t::tls_buffer_;
        tls.in_managed_thread_pool_ = true;
        tls.additional_->linear_thread_id_ = 0;
#ifdef SC_KERNEL_PROFILE
        tls.additional_->instance_id_ = instance_id_;
#endif
        f(stream, mod_data, args);
        do_cleanup();
    }
}

alignas(64) thread_local thread_manager thread_manager::cur_mgr;
} // namespace runtime
} // namespace sc

// using balance211 to dispatch the workloads
static void do_dispatch(thread_manager *s, int tid) {
    size_t end = s->state.task.end;
    size_t begin = s->state.task.begin;
    size_t step = s->state.task.step;
    size_t len = end - begin;
    size_t num_jobs = utils::divide_and_ceil(len, s->state.task.step);
    size_t my_jobs = utils::divide_and_ceil(num_jobs, s->state.num_threads);
    assert(my_jobs > 0);
    size_t my_jobs_2 = my_jobs - 1;
    size_t the_tid = num_jobs - my_jobs_2 * s->state.num_threads;
    size_t cur_jobs = (size_t)tid < the_tid ? my_jobs : my_jobs_2;
    size_t my_begin = (size_t)tid <= the_tid
            ? tid * my_jobs
            : the_tid * my_jobs + (tid - the_tid) * my_jobs_2;
    my_begin = my_begin * step + begin;
    bool disable_rolling = s->state.execution_flags
            & sc::runtime::thread_pool_flags::THREAD_POOL_DISABLE_ROLLING;
    for (size_t jid = 0; jid < cur_jobs; jid++) {
        // Rolling i with tid
        size_t real_jid = disable_rolling ? jid : ((jid + tid) % cur_jobs);
        size_t rolling_i = real_jid * step + my_begin;
        s->state.task.pfunc(s->state.task.stream, s->state.task.module_env,
                rolling_i, s->state.task.args);
    }
}

void sc_parallel_call_managed(
        void (*pfunc)(void *, void *, int64_t, sc::generic_val *),
        uint64_t execution_flags, void *rtl_ctx, void *module_env,
        int64_t begin, int64_t end, int64_t step, sc::generic_val *args) {
    sc::runtime::thread_local_buffer_t::tls_buffer_.additional_->is_main_thread_
            = true;
    thread_manager *stream = &thread_manager::cur_mgr;
    stream->state.execution_flags = execution_flags;
    stream->state.reset_scoreboard();
    stream->state.task = thread_manager::thread_pool_state::task_type {
            pfunc, rtl_ctx, module_env, begin, end, step, args};
    stream->state.trigger
            = stream->state.trigger.load(std::memory_order_relaxed) + 1;
    do_dispatch(stream, 0);
    stream->state.wait_all();

    if (execution_flags
            & sc::runtime::thread_pool_flags::THREAD_POOL_RUN_IDLE_FUNC) {
        stream->state.idle_func = nullptr;
    }
    stream->state.execution_flags
            = sc::runtime::thread_pool_flags::THREAD_POOL_DEFAULT;
}

void sc_set_idle_func_managed(thread_manager::idle_func_t func, void *args) {
    thread_manager *mgr = &thread_manager::cur_mgr;
    mgr->state.idle_func = func;
    mgr->state.idle_args = args;
}
