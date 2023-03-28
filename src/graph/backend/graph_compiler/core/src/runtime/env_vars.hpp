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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_ENV_VARS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_ENV_VARS_HPP
#include <util/def.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace env_key {
enum key {
    SC_CPU_JIT,
    SC_OPT_LEVEL,
    SC_TRACE,
    SC_VERBOSE,
    SC_PRINT_PASS_RESULT,
    SC_DUMP_GENCODE,
    SC_C_INCLUDE,
    SC_TRACE_INIT_CAP,
    SC_MANAGED_THREAD_POOL,
    NUM_KEYS
};
} // namespace env_key

extern const char *env_names[env_key::NUM_KEYS];
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
