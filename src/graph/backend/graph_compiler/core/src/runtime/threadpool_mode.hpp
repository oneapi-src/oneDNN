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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_THREADPOOL_MODE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_THREADPOOL_MODE_HPP
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

enum class thread_pool_mode_t {
    DIRECT, // directly using underlying thread pool (OMP, Eigen, etc.)
    MANAGED, // managed thread pool
    DYNAMIC, // queue-based dynamic thread pool
};

}
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
