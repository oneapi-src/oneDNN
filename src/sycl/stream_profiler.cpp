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

#include <atomic>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>
#include <unordered_map>

#include "sycl/stream_profiler.hpp"

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_stream.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

status_t sycl_stream_profiler_t::get_info(profiling_data_kind_t data_kind,
        int *num_entries, uint64_t *data) const {
    using namespace ::sycl::info;
    if (!num_entries) return status::invalid_arguments;
    if (!data) {
        std::unordered_set<uint64_t> seen;
        for (auto &ev : events_)
            seen.insert(ev.stamp);
        *num_entries = (int)seen.size();
        return status::success;
    }

    std::unordered_map<uint64_t, stream_profiler_t::entry_t> stamp2entry;
    for (auto &ev : events_) {
        const sycl_event_t &sycl_event
                = *utils::downcast<sycl_event_t *>(ev.event.get());
        assert(sycl_event.size() == 1);
        auto beg
                = sycl_event[0]
                          .get_profiling_info<event_profiling::command_start>();
        auto end = sycl_event[0]
                           .get_profiling_info<event_profiling::command_end>();
        auto &entry = stamp2entry[ev.stamp];
        entry.min_nsec = std::min(entry.min_nsec, beg);
        entry.max_nsec = std::max(entry.max_nsec, end);
        entry.kernel_count++;
    }
    return stream_profiler_t::get_info_impl(stamp2entry, data_kind, data);
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
