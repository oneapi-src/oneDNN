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
#include <mutex>
#include <vector>

#include "sycl/profile.hpp"

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/profile.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::sycl;

namespace dnnl {
namespace impl {
namespace sycl {

struct profile_event_t {
    profile_event_t(const ::sycl::event &event, uint64_t stamp)
        : event(event), stamp(stamp) {}

    ::sycl::event event;
    uint64_t stamp;
};

static std::vector<profile_event_t> events;
static std::atomic<uint64_t> stamp(0);

void notify_before_exec() {
    stamp++;
}

void register_profile_event(const ::sycl::event &event) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    events.emplace_back(event, stamp);
}

status_t get_profile_info(uint64_t &nsec, double &freq, int mode) {
    using namespace ::sycl::info;
    std::unordered_map<uint64_t, gpu::profile_entry_t> stamp2entry;
    for (auto &ev : events) {
        auto beg
                = ev.event.get_profiling_info<event_profiling::command_start>();
        auto end = ev.event.get_profiling_info<event_profiling::command_end>();
        auto &entry = stamp2entry[ev.stamp];
        entry.nsec += (end - beg);
        entry.kernel_count++;
    }
    return gpu::get_profile_info_impl(nsec, freq, mode, stamp2entry);
}

status_t reset_profiling() {
    events.clear();
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
