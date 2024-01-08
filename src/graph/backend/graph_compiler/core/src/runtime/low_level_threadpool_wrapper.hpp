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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_LOW_LEVEL_THREADPOOL_WRAPPER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_LOW_LEVEL_THREADPOOL_WRAPPER_HPP

#include <exception>
#include "thread_locals.hpp"
#include <runtime/config.hpp>
#include <util/compiler_macros.hpp>
#include <util/def.hpp>

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
#include <common/dnnl_thread.hpp>
#endif

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
// clang-format off
#include <tbb/parallel_for_each.h>
// clang-format on
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

inline thread_local_buffer_t &get_tls_helper() {
    return thread_local_buffer_t::tls_buffer();
}

using main_func_t = void (*)(runtime::stream_t *, void *, generic_val *);
template <typename T, typename TSched>
void call_threadpool(TSched *ths, main_func_t f, runtime::stream_t *stream,
        void *mod_data, generic_val *args) {
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
    int threads = dnnl::impl::threadpool_utils::get_active_threadpool()
                          ->get_num_threads();
    int runtime_threads = runtime_config_t::get().get_num_threads();
    threads = (threads < runtime_threads) ? threads : runtime_threads;
#else
    int threads = runtime_config_t::get().get_num_threads();
#endif
    ths = T::all_thread_prepare(ths, stream, threads);
    if (threads > 1 || !T::can_optimize_single_thread) {
        typename T::TyState rtl_state {T::before_parallel(ths)};

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
        SC_NO_OP();
#pragma omp parallel for
        for (int it = 0; it < threads; it++) {
            int64_t i = it;
            SC_NO_OP();
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
        oneapi::tbb::task_arena arena(threads);
        arena.execute([&] {
            tbb::parallel_for(0, threads, 1, [&](int64_t i) {
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
        dnnl::impl::parallel(threads, [&](int64_t i, int64_t dummy) {
#endif
            // use helper func to workaround a icx compiler bug
            auto &tls = get_tls_helper();
            i = T::parse_tid(rtl_state, ths, tls, i);
            tls.in_managed_thread_pool_ = true;
            tls.additional_->linear_thread_id_ = i;
            if (i == 0) {
                T::main_thread(ths, f, stream, mod_data, args);
            } else {
                T::worker_thread(ths, i);
            }
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
        }
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
            });
        });
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
        });
#endif
        T::after_parallel(ths);
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_SEQ
        throw std::runtime_error("Running SEQ in thread pool");
#endif
    } else {
        T::single_thread(ths, f, stream, mod_data, args);
    }
}

} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
