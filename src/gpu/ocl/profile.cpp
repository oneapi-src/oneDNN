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

#include <mutex>
#include <utility>
#include <vector>
#include <CL/cl.h>

#include "gpu/ocl/profile.hpp"

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

static std::vector<std::pair<cl_event, const ocl_stream_t *>> events;

void register_profiling_event(cl_event event, const ocl_stream_t *ocl_stream) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    events.emplace_back(event, ocl_stream);
}

status_t get_profiling_info(uint64_t &nsec, double &freq) {
    nsec = 0;
    freq = 0;
    for (auto &p : events) {
        auto &ev = p.first;
        auto *stream = p.second;
        cl_ulong beg, end;
        OCL_CHECK(clGetEventProfilingInfo(
                ev, CL_PROFILING_COMMAND_START, sizeof(beg), &beg, nullptr));
        OCL_CHECK(clGetEventProfilingInfo(
                ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
        nsec += (end - beg);
        freq += stream->mdapi_helper().get_freq(ev);
    }
    freq /= events.size();
    return status::success;
}

status_t reset_profiling() {
    for (auto &p : events) {
        clReleaseEvent(p.first);
    }
    events.clear();
    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
