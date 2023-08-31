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

#include <algorithm>
#include <memory>
#include "config.hpp"
#include "context.hpp"
#include <runtime/generic_val.hpp>
#include <runtime/parallel.hpp>
#include <util/compiler_macros.hpp>
#include <util/simple_math.hpp>
#include <util/utils.hpp>
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
#include <common/dnnl_thread.hpp>
#include <oneapi/dnnl/dnnl_threadpool.h>
#endif

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
#include <tbb/global_control.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_arena.h>
#endif

#include <runtime/thread_locals.hpp>
#ifdef SC_KERNEL_PROFILE
#include <atomic>
#endif

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
#include <omp.h>
#endif

using namespace dnnl::impl::graph::gc;

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM

static int get_num_threads() {
    int ret = 0;
    dnnl_threadpool_interop_get_max_concurrency(&ret);
    return ret;
}

extern "C" void sc_parallel_call_cpu_with_env_impl(
        void (*pfunc)(void *, void *, int64_t, generic_val *), uint64_t flags,
        void *rtl_ctx, void *module_env, int64_t begin, int64_t end,
        int64_t step, generic_val *args) {
    runtime::thread_local_buffer_t::tls_buffer().additional_->is_main_thread_
            = true;
    using namespace dnnl::impl;
    auto num_jobs
            = dnnl::impl::graph::gc::utils::divide_and_ceil(end - begin, step);
    int nthr = adjust_num_threads(
            std::min(get_num_threads(), dnnl_get_current_num_threads()),
            num_jobs);
    if (nthr) {
        dnnl::impl::parallel(nthr, [&](int ithr, int nthr) {
            runtime::thread_local_buffer_t::tls_buffer()
                    .additional_->linear_thread_id_
                    = ithr;
            auto f = [&](int64_t i) {
                pfunc(rtl_ctx, module_env, i * step + begin, args);
            };
            for_nd(ithr, nthr, num_jobs, f);
        });
    }
    runtime::thread_local_buffer_t::tls_buffer().additional_->linear_thread_id_
            = 0;
}

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
int get_max_threadpool_concurrency() {
    static int v = get_num_threads();
    return v;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

static void set_num_threads(int num) {
    (void)get_max_threadpool_concurrency();
    dnnl_threadpool_interop_set_max_concurrency(num);
}

static int get_thread_num() {
    return runtime::thread_local_buffer_t::tls_buffer()
            .additional_->linear_thread_id_;
}

static int get_in_parallel() {
    return dnnl::impl::threadpool_utils::get_active_threadpool()
            ->get_in_parallel();
}

#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB

static int &get_default_threads() {
    static int num_threads = oneapi::tbb::info::default_concurrency();
    return num_threads;
}
extern "C" void sc_parallel_call_cpu_with_env_impl(
        void (*pfunc)(void *, void *, int64_t, generic_val *), uint64_t flags,
        void *rtl_ctx, void *module_env, int64_t begin, int64_t end,
        int64_t step, generic_val *args) {
    runtime::thread_local_buffer_t::tls_buffer().additional_->is_main_thread_
            = true;
    oneapi::tbb::task_arena arena(get_default_threads());
    arena.execute([&] {
        tbb::parallel_for(
                begin, end, step,
                [&](int64_t i) { pfunc(rtl_ctx, module_env, i, args); },
                tbb::simple_partitioner());
    });
}

static int get_num_threads() {
    return get_default_threads();
}

static void set_num_threads(int num) {
    static std::unique_ptr<oneapi::tbb::global_control> ctrl;
    ctrl = utils::make_unique<oneapi::tbb::global_control>(
            oneapi::tbb::global_control::max_allowed_parallelism, num);
    get_default_threads() = num;
}

static int get_thread_num() {
    return tbb::this_task_arena::current_thread_index();
}
static int get_in_parallel() {
    return 0;
}

#else

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
#define get_num_threads omp_get_max_threads
#define set_num_threads omp_set_num_threads
#define get_thread_num omp_get_thread_num
#define get_in_parallel omp_in_parallel
#else
static int get_num_threads() {
    return 1;
}

static void set_num_threads(int num) {}
static int get_thread_num() {
    return 0;
}
static int get_in_parallel() {
    return 0;
}
#endif

#ifdef SC_KERNEL_PROFILE
static std::atomic<int> instance_cnt = {0};
static thread_local int instance_id = instance_cnt++;
#endif

// omp or sequential
extern "C" void sc_parallel_call_cpu_with_env_impl(
        void (*pfunc)(void *, void *, int64_t, generic_val *), uint64_t flags,
        void *rtl_ctx, void *module_env, int64_t begin, int64_t end,
        int64_t step, generic_val *args) {
#ifdef SC_KERNEL_PROFILE
    int parent_instance_id = instance_id;
#endif
    runtime::thread_local_buffer_t::tls_buffer().additional_->is_main_thread_
            = true;

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
    SC_NO_OP();
#pragma omp parallel for
#endif
    for (int64_t i = begin; i < end; i += step) {
        SC_NO_OP();
#ifdef SC_KERNEL_PROFILE
        auto &tls = runtime::thread_local_buffer_t::tls_buffer();
        tls.additional_->instance_id_ = parent_instance_id;
        tls.additional_->linear_thread_id_ = get_thread_num();
#endif
        pfunc(rtl_ctx, module_env, i, args);
    }
}

#endif
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
thread_pool_table sc_pool_table {&sc_parallel_call_cpu_with_env_impl, nullptr,
        &get_num_threads, &set_num_threads, &get_thread_num, &get_in_parallel};
}
} // namespace graph
} // namespace impl
} // namespace dnnl
