/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include <CL/cl.h>

#include <map>
#include <unordered_set>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "xpu/ocl/context.hpp"
#include "xpu/ocl/stream_profiler.hpp"

#include "gpu/gpu_stream.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

status_t stream_profiler_t::get_info(profiling_data_kind_t data_kind,
        int *num_entries, uint64_t *data) const {
    if (!num_entries) return status::invalid_arguments;
    if (!data) {
        std::unordered_set<uint64_t> seen;
        for (auto &ev : events_)
            seen.insert(ev.stamp);
        *num_entries = (int)seen.size();
        return status::success;
    }

    std::map<uint64_t, xpu::stream_profiler_t::entry_t> stamp2entry;
    for (auto &ev : events_) {
        auto &entry = stamp2entry[ev.stamp];
        const xpu::ocl::event_t &ocl_event
                = *utils::downcast<xpu::ocl::event_t *>(ev.event.get());
        cl_ulong beg, end;
        assert(ocl_event.size() == 1);
        OCL_CHECK(clGetEventProfilingInfo(ocl_event[0].get(),
                CL_PROFILING_COMMAND_START, sizeof(beg), &beg, nullptr));
        OCL_CHECK(clGetEventProfilingInfo(ocl_event[0].get(),
                CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
        entry.min_nsec = std::min(entry.min_nsec, beg);
        entry.max_nsec = std::max(entry.max_nsec, end);
        const auto *gpu_stream
                = utils::downcast<const gpu::stream_t *>(stream_);
        entry.freq += gpu_stream->get_freq(*ev.event);
        entry.kernel_count++;
    }
    return xpu::stream_profiler_t::get_info_impl(stamp2entry, data_kind, data);
}

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl
