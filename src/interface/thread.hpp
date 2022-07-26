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

#ifndef INTERFACE_THREAD_HPP
#define INTERFACE_THREAD_HPP

#if DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_OMP
#include "omp.h"
inline int dnnl_graph_get_max_threads() {
    return omp_get_max_threads();
}
#elif DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_TBB \
        || DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_SYCL
#include "tbb/task_arena.h"
inline int dnnl_graph_get_max_threads() {
    return tbb::this_task_arena::max_concurrency();
}
#else
// For SEQ, it's expected to use only one thread to compile and execute. For
// THREADPOOL, usually we don't know the threadpool information at compilation
// stage. TODO(xxx): It's not accurate but do we need to read core number to
// decide the max thread number?
inline int dnnl_graph_get_max_threads() {
    return 1;
}
#endif
#endif
