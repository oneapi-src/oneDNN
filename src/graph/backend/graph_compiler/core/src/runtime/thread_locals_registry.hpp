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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_THREAD_LOCALS_REGISTRY_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_THREAD_LOCALS_REGISTRY_HPP
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "thread_locals.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

struct trace_env_t {
    std::mutex name_lock_;
    std::vector<std::string> names_;
    trace_env_t();
};

// the registry of all TLS resources.
struct thread_local_registry_t {
    std::mutex lock_;
    std::list<thread_local_buffer_t *> tls_buffers_;
    std::vector<std::unique_ptr<thread_local_buffer_t::additional_t>>
            dead_threads_;
    trace_env_t trace_env_;
    void release(engine_t *engine);
    void for_each_tls_additional(
            const std::function<void(thread_local_buffer_t::additional_t *)>
                    &f);

    thread_local_registry_t();
    ~thread_local_registry_t();
};

} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
