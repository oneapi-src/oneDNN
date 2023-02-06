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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_BARRIER_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_BARRIER_HPP
#include <atomic>
#include <stdint.h>
#include <util/def.hpp>

namespace sc {
namespace runtime {

struct barrier_t {
    alignas(64) std::atomic<int64_t> pending_;
    std::atomic<uint64_t> rounds_;
    uint64_t total_;
    // pad barrier to size of cacheline to avoid false sharing
    char padding_[64 - 4 * sizeof(uint64_t)];
};

typedef uint64_t (*barrier_idle_func)(
        std::atomic<uint64_t> *remaining, uint64_t expected_remain, void *args);

} // namespace runtime
} // namespace sc

extern "C" SC_API void sc_arrive_at_barrier(sc::runtime::barrier_t *b,
        sc::runtime::barrier_idle_func idle_func, void *idle_args);
extern "C" SC_API void sc_init_barrier(
        sc::runtime::barrier_t *b, int num_barriers, uint64_t thread_count);
#endif
