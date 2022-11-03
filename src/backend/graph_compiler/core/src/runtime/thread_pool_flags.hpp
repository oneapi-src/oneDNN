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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_THREAD_POOL_FLAGS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_THREAD_POOL_FLAGS_HPP
#include <stdint.h>

namespace sc {

namespace runtime {

namespace thread_pool_flags {
constexpr int THREAD_POOL_DEFAULT = 0;
constexpr int THREAD_POOL_RUN_IDLE_FUNC = 1;
constexpr int THREAD_POOL_DISABLE_ROLLING = 1 << 1;
} // namespace thread_pool_flags

} // namespace runtime
} // namespace sc

#endif
