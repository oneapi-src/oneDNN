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

#ifndef GPU_PROFILE_HPP
#define GPU_PROFILE_HPP

#include <unordered_map>

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

enum class profile_mode_t : int {
    sum = 0,
    min = 1,
};

struct profile_entry_t {
    uint64_t nsec = 0;
    double freq = 0;
    int kernel_count = 0;
};

bool is_profiling_enabled();

status_t get_profile_info_impl(
        const std::unordered_map<uint64_t, profile_entry_t> &entries,
        uint64_t *nsecs, uint64_t *cycles);

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
