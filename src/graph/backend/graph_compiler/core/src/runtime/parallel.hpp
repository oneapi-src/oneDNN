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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_PARALLEL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_PARALLEL_HPP
#include <runtime/config.hpp>
#include <runtime/generic_val.hpp>
#include <util/def.hpp>

// the default implementation of SC's thread pool
extern "C" SC_API void sc_parallel_call_cpu_with_env_impl(
        void (*pfunc)(
                void *, void *, int64_t, dnnl::impl::graph::gc::generic_val *),
        uint64_t flags, void *rtl_ctx, void *module_env, int64_t begin,
        int64_t end, int64_t step, dnnl::impl::graph::gc::generic_val *args);

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
extern thread_pool_table sc_pool_table;
}
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
