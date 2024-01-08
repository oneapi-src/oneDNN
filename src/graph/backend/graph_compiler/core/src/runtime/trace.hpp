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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_TRACE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_TRACE_HPP
#include <stdint.h>
#include <string>
#include <vector>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

struct thread_local_buffer_t;
struct thread_local_registry_t;
struct trace_manager_t {
    struct trace_log_t {
        uint16_t func_id_;
        char in_or_out_;
        int32_t arg_;
        int64_t tick_;
    };
    std::vector<trace_log_t> trace_logs_;
};

void write_traces(thread_local_registry_t *r);

} // namespace runtime
int register_traced_func(const std::string &name);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
