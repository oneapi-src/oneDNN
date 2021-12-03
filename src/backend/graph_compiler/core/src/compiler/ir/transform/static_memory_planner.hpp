/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_STATIC_MEMORY_PLANNER_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_STATIC_MEMORY_PLANNER_HPP

#include <stdint.h>
#include <vector>
#include <unordered_map>

namespace sc {
namespace memory_optim {
struct memory_alloc_trace_t {
    // unique id of a buffer
    uintptr_t buffer_id_;
    // if > 0, size of the buffer allocation, if = 0, it is a deallocation trace
    std::size_t size_;

    memory_alloc_trace_t(uintptr_t buffer_id = 0, std::size_t size = 0) {
        buffer_id_ = buffer_id;
        size_ = size;
    }
};

/**
 * Given a list of memory buffer alloc and free traces, try to use a large
 * buffer to hold all allocated memory, and statically allocate each memory
 * buffer from the large buffer for better memory reuse.
 * @param traces the list of memory alloc and free traces, sorted by event time.
 * @param out_schedule the output schedule for each buffer: the location that
 * the buffer should be in the large buffer (as an offset in number of elements)
 * @param alignment the alignment in number of elements
 * @param hot_first use the hot buffer first, instead of using best fit in size
 * @return the size of the large buffer, in number of elements
 * */
std::size_t schedule_memory_allocations(
        const std::vector<memory_alloc_trace_t> &traces, std::size_t alignment,
        bool hot_first,
        std::unordered_map<uintptr_t, std::size_t> &out_schedule);
} // namespace memory_optim
} // namespace sc

#endif
