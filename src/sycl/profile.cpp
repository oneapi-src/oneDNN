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
#include <vector>

#include "sycl/profile.hpp"

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

using namespace dnnl::impl::sycl;

namespace dnnl {
namespace impl {
namespace sycl {

static std::vector<::sycl::event> events;

void register_profiling_event(const ::sycl::event &event) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    events.push_back(event);
}

status_t get_profiling_time(uint64_t *nsec) {
    using namespace ::sycl::info;
    *nsec = 0;
    for (auto ev : events) {
        auto beg = ev.get_profiling_info<event_profiling::command_start>();
        auto end = ev.get_profiling_info<event_profiling::command_end>();
        *nsec += (end - beg);
    }
    return status::success;
}

status_t reset_profiling() {
    events.clear();
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
