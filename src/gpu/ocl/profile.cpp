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

#include <atomic>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>
#include <CL/cl.h>
#include <unordered_map>

#include "gpu/ocl/profile.hpp"
#include "gpu/profile.hpp"

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/ocl/mdapi_utils.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::gpu::ocl;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct profile_event_t {
    profile_event_t(cl_event event, const ocl_stream_t *stream, uint64_t stamp)
        : event(event), stream(stream), stamp(stamp) {}

    cl_event event;
    const ocl_stream_t *stream;
    uint64_t stamp;
};

static std::vector<profile_event_t> events;
static std::atomic<uint64_t> stamp(0);

void notify_before_exec() {
    stamp++;
}

void register_profile_event(cl_event event, const ocl_stream_t *ocl_stream) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    events.emplace_back(event, ocl_stream, stamp);
}

status_t get_profile_info(uint64_t &nsec, double &freq, int mode) {
    nsec = 0;
    freq = 0;
    std::unordered_map<uint64_t, profile_entry_t> stamp2entry;
    for (auto &ev : events) {
        cl_ulong beg, end;
        OCL_CHECK(clGetEventProfilingInfo(ev.event, CL_PROFILING_COMMAND_START,
                sizeof(beg), &beg, nullptr));
        OCL_CHECK(clGetEventProfilingInfo(ev.event, CL_PROFILING_COMMAND_END,
                sizeof(end), &end, nullptr));
        auto &entry = stamp2entry[ev.stamp];
        entry.nsec += (end - beg);
        entry.freq += ev.stream->mdapi_helper().get_freq(ev.event);
        entry.kernel_count++;
    }
    return get_profile_info_impl(nsec, freq, mode, stamp2entry);
}

status_t reset_profiling() {
    for (auto &ev : events)
        clReleaseEvent(ev.event);
    events.clear();
    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
