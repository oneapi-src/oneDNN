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
#include <stdint.h>
#include <util/def.hpp>

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OPS_UTIL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OPS_UTIL_HPP

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
struct dispatch_table_t;
inline void *run_query_and_wait(
        void *(*f)(dispatch_table_t *ths, uint64_t *keys, uint64_t num_keys),
        dispatch_table_t *table, uint64_t *keys, uint64_t num_keys) {
    return f(table, keys, num_keys);
}

} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
