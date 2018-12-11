/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include "common/c_types_map.hpp"

#include <CL/sycl.hpp>
#include <vector>

namespace mkldnn {
namespace impl {
namespace sycl {

static inline std::vector<cl::sycl::device> get_sycl_devices(
        engine_kind_t engine_kind) {
    std::vector<cl::sycl::device> devices;
    auto all_platforms = cl::sycl::platform::get_platforms();
    for (auto &plat : all_platforms) {
        auto dev_type = (engine_kind == engine_kind::cpu)
                ? cl::sycl::info::device_type::cpu
                : cl::sycl::info::device_type::gpu;
        auto devs = plat.get_devices(dev_type);
        if (devs.empty())
            continue;
        devices.insert(devices.end(), devs.begin(), devs.end());
    }
    return devices;
}
} // namespace sycl
} // namespace impl
} // namespace mkldnn

#endif
