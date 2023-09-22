/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#include "managed_thread_pool_exports.hpp"
#include "memorypool.hpp"
#include "runtime.hpp"
#include "thread_locals.hpp"
#include "thread_pool_flags.hpp"
#include <cpu/x64/amx_tile_configure.hpp>
#include <runtime/microkernel/cpu/kernel_timer.hpp>
#include <util/compiler_macros.hpp>
#include <util/simple_math.hpp>

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
#include <common/dnnl_thread.hpp>
#endif

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
// clang-format off
#include <tbb/parallel_for_each.h>
// clang-format on
#endif

using namespace dnnl::impl::graph::gc;
using runtime::thread_manager;
static void do_dispatch(thread_manager *s, int tid);
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
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
    auto &tls = thread_local_buffer_t::tls_buffer();
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
    thread_manager::idle_func_t idl_f = nullptr;
    uint64_t exec_flags = 0;
    while (true) {
        bool has_idle_func = idl_f
                && (exec_flags & thread_pool_flags::THREAD_POOL_RUN_IDLE_FUNC);
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
        idl_f = ths->state.idle_func;
        exec_flags = ths->state.execution_flags;
        make_trace(1, count);
        do_dispatch(ths, tid);
        make_trace(0, 0);
        // check for last parallel-for fast exit
        if (exec_flags & thread_pool_flags::THREAD_POOL_EXIT) {
            do_cleanup();
            make_trace(1, count);
            return;
        }
        --ths->state.remaining;
        current_job_id++;
    }
}

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
static thread_local thread_manager *current_active_thr_mgr = nullptr;
#endif

static thread_local_buffer_t &get_tls_helper() {
    return thread_local_buffer_t::tls_buffer();
}

void thread_manager::run_main_function(main_func_t f, runtime::stream_t *stream,
        void *mod_data, generic_val *args) {
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
    int threads = dnnl::impl::threadpool_utils::get_active_threadpool()
                          ->get_num_threads();
    int runtime_threads = runtime_config_t::get().get_num_threads();
    threads = (threads < runtime_threads) ? threads : runtime_threads;
#else
    int threads = runtime_config_t::get().get_num_threads();
#endif
    state.num_threads = threads;
    if (threads > 1) {
        state.trigger = 1;
        state.execution_flags
                = gc::runtime::thread_pool_flags::THREAD_POOL_DEFAULT;

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
        SC_NO_OP();
#pragma omp parallel for
        for (int i = 0; i < threads; i++) {
            SC_NO_OP();
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
        oneapi::tbb::task_arena arena(threads);
        arena.execute([&] {
            tbb::parallel_for(0, threads, 1, [&](int64_t i) {
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
        dnnl::impl::parallel(threads, [&](int64_t i, int64_t dummy) {
            current_active_thr_mgr = this;
#endif
            // use helper func to workaround a icx compiler bug
            auto &tls = get_tls_helper();
            tls.in_managed_thread_pool_ = true;
            tls.additional_->linear_thread_id_ = i;
#ifdef SC_KERNEL_PROFILE
            tls.additional_->instance_id_ = instance_id_;
#endif
            if (i == 0) {
                f(stream, mod_data, args);
                if (!(state.execution_flags
                            & thread_pool_flags::THREAD_POOL_EXIT)) {
                    // send the exit signal. If it has EXIT flag, don't send,
                    // and let the thread exit by itself. If we send the signal
                    // in this case, it may result in some threads skipping
                    // the last job
                    state.trigger = -1;
                }
                do_cleanup();
            } else {
                worker_func(this, i);
            }
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
        }
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
            });
        });
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
            current_active_thr_mgr = nullptr;
        });
#endif
        if (state.execution_flags & thread_pool_flags::THREAD_POOL_EXIT) {
            state.trigger = -1;
            state.execution_flags = 0;
        }
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_SEQ
        throw std::runtime_error("Running SEQ in thread pool");
#endif
    } else {
        auto &tls = thread_local_buffer_t::tls_buffer();
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
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

static thread_manager *get_current_active_thr_mgr() {
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
    return dnnl::impl::graph::gc::runtime::current_active_thr_mgr;
#else
    return &thread_manager::cur_mgr;
#endif
}

// using balance211 to dispatch the workloads
static void do_dispatch(thread_manager *s, int tid) {
    size_t end = s->state.task.end;
    size_t begin = s->state.task.begin;
    size_t step = s->state.task.step;
    size_t len = end - begin;
    size_t num_jobs = utils::divide_and_ceil(len, s->state.task.step);
    if (num_jobs == (unsigned)s->state.num_threads) {
        s->state.task.pfunc(s->state.task.stream, s->state.task.module_env,
                begin + step * tid, s->state.task.args);
        return;
    }
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
            & runtime::thread_pool_flags::THREAD_POOL_DISABLE_ROLLING;
    for (size_t jid = 0; jid < cur_jobs; jid++) {
        // Rolling i with tid
        size_t real_jid = disable_rolling ? jid : ((jid + tid) % cur_jobs);
        size_t rolling_i = real_jid * step + my_begin;
        s->state.task.pfunc(s->state.task.stream, s->state.task.module_env,
                rolling_i, s->state.task.args);
    }
}

void sc_parallel_call_managed(
        void (*pfunc)(void *, void *, int64_t, generic_val *),
        uint64_t execution_flags, void *rtl_ctx, void *module_env,
        int64_t begin, int64_t end, int64_t step, generic_val *args) {
    runtime::thread_local_buffer_t::tls_buffer().additional_->is_main_thread_
            = true;
    thread_manager *stream = get_current_active_thr_mgr();
    stream->state.execution_flags = execution_flags;
    stream->state.reset_scoreboard();
    stream->state.task = thread_manager::thread_pool_state::task_type {
            pfunc, rtl_ctx, module_env, begin, end, step, args};
    stream->state.trigger
            = stream->state.trigger.load(std::memory_order_relaxed) + 1;
    do_dispatch(stream, 0);
    if (execution_flags
            & dnnl::impl::graph::gc::runtime::thread_pool_flags::
                    THREAD_POOL_EXIT) {
        return;
    }
    stream->state.wait_all();

    if (execution_flags
            & runtime::thread_pool_flags::THREAD_POOL_RUN_IDLE_FUNC) {
        stream->state.idle_func = nullptr;
    }
    stream->state.execution_flags
            = runtime::thread_pool_flags::THREAD_POOL_DEFAULT;
}

void sc_set_idle_func_managed(thread_manager::idle_func_t func, void *args) {
    thread_manager *mgr = get_current_active_thr_mgr();
    mgr->state.idle_func = func;
    mgr->state.idle_args = args;
}
