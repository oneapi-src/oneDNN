/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_STATIC_MEMORY_PLANNER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_STATIC_MEMORY_PLANNER_HPP

#include <stdint.h>
#include <utility>
#include <vector>
#include "tensor_inplace_info.hpp"
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
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

using inplace_info = std::pair<uintptr_t, inplace_kind>;

using inplace_info_map
        = std::unordered_map<uintptr_t, std::vector<inplace_info>>;

/**
 * Given a list of memory buffer alloc and free traces, try to use a large
 * buffer to hold all allocated memory, and statically allocate each memory
 * buffer from the large buffer for better memory reuse.
 * @param traces the list of memory alloc and free traces, sorted by event time.
 * @param alignment the alignment in number of elements
 * @param hot_first use the hot buffer first, instead of using best fit in size
 * @param inplace_map the map from the tensor to alloc into the candidate
 * tensors that can be inplace reused for it.
 * @param out_schedule the output schedule for each buffer: the location that
 * the buffer should be in the large buffer (as an offset in number of elements)
 * @param out_inplace_selection the output buffer id -> inplace buffer it reuses
 * @return the size of the large buffer, in number of elements
 * */
std::size_t schedule_memory_allocations(
        const std::vector<memory_alloc_trace_t> &traces, std::size_t alignment,
        bool hot_first, const inplace_info_map &inplace_map,
        std::unordered_map<uintptr_t, std::size_t> &out_schedule,
        std::unordered_map<uintptr_t, std::vector<uintptr_t>>
                &out_inplace_selection);
} // namespace memory_optim
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
