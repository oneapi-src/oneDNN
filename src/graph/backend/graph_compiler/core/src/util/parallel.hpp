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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_PARALLEL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_PARALLEL_HPP

#include <utility>
#include <runtime/config.hpp>
#include <util/compiler_macros.hpp>
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
#include <tbb/parallel_for.h>
#endif
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
#include <common/dnnl_thread.hpp>
#include <util/simple_math.hpp>
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
template <typename F>
void parallel(F &&f, int64_t begin, int64_t end, int64_t step = 1,
        int num_threads = dnnl_get_current_num_threads()) {
    auto num_jobs = utils::divide_and_ceil(end - begin, step);
    int nthr = adjust_num_threads(num_threads, num_jobs);
    auto execf = [&](int64_t i) { f(i * step + begin, end); };
    if (nthr)
        dnnl::impl::parallel(nthr, [&](int ithr, int nthr) {
            for_nd(ithr, nthr, num_jobs, execf);
        });
}

template <typename F>
void parallel_for(int64_t begin, int64_t end, int64_t step, F &&f) {
    auto num_jobs = utils::divide_and_ceil(end - begin, step);
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), num_jobs);
    auto execf = [&](int64_t i) { f(i * step + begin); };
    if (nthr)
        dnnl::impl::parallel(nthr, [&](int ithr, int nthr) {
            for_nd(ithr, nthr, num_jobs, execf);
        });
}
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
template <typename F>
void parallel(F &&f, int64_t begin, int64_t end, int64_t step = 1,
        int num_threads = runtime_config_t::get().get_num_threads()) {
    tbb::parallel_for(begin, end, step, [&](int64_t i) { f(i, end); });
}

template <typename F>
void parallel_for(int64_t begin, int64_t end, int64_t step, F &&f) {
    tbb::parallel_for(begin, end, step, std::forward<F>(f));
}

#else
template <typename F>
void parallel(F f, int64_t begin, int64_t end, int64_t step = 1,
        int num_threads = runtime_config_t::get().get_num_threads()) {
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
    SC_NO_OP();
#pragma omp parallel for num_threads(num_threads)
#endif
    for (int64_t i = begin; i < end; i += step) {
        SC_NO_OP();
        f(i, end);
    }
}

template <typename F>
void parallel_for(int64_t begin, int64_t end, int64_t step, F &&f) {
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
    SC_NO_OP();
#pragma omp parallel for
#endif
    for (int64_t i = begin; i < end; i += step) {
        SC_NO_OP();
        f(i);
    }
}
#endif

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
