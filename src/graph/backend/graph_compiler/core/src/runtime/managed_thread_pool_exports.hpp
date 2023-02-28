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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MANAGED_THREAD_POOL_EXPORTS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MANAGED_THREAD_POOL_EXPORTS_HPP
#include <atomic>
#include <runtime/context.hpp>

extern "C" SC_API void sc_parallel_call_managed(
        void (*pfunc)(
                void *, void *, int64_t, dnnl::impl::graph::gc::generic_val *),
        uint64_t execution_flags, void *rtl_ctx, void *module_env,
        int64_t begin, int64_t end, int64_t step,
        dnnl::impl::graph::gc::generic_val *args);

extern "C" SC_API void sc_set_idle_func_managed(
        uint64_t (*func)(std::atomic<int> *remaining, int expected_remain,
                int tid, void *args),
        void *args);

#endif
