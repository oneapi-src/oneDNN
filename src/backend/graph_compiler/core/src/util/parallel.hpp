/*******************************************************************************
 * Copyright 2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_PARALLEL_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_PARALLEL_HPP

#include <runtime/config.hpp>

namespace sc {
namespace utils {
template <typename F>
void parallel(F f, uint64_t begin, uint64_t end, uint64_t step = 1,
        int numthreads = runtime_config_t::get().threads_per_instance_) {
#ifdef SC_OMP_ENABLED
#pragma omp parallel for num_threads(numthreads)
#endif
    for (uint64_t i = begin; i < end; i += step) {
        f(i, end);
    }
}
} // namespace utils
} // namespace sc

#endif
