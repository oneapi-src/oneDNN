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
#include <stdint.h>

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_DISPATCH_TABLE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_DISPATCH_TABLE_HPP

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

struct dispatch_table_t {
    virtual ~dispatch_table_t() = default;
    using dispatch_func_t = void *(*)(dispatch_table_t *ths, uint64_t *keys,
            uint64_t num_keys);
    virtual dispatch_func_t get_dispatch_func() = 0;
    virtual void *get(uint64_t *keys, uint64_t num_keys) = 0;
    virtual void set(uint64_t *keys, uint64_t num_keys, void *value) = 0;
};

} // namespace runtime

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
