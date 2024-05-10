/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef COMMON_XPU_UTILS_HPP
#define COMMON_XPU_UTILS_HPP

#include <tuple>
#include <vector>

#include "common/utils.hpp"

// This file contains utility functionality for heterogeneous runtimes such
// as OpenCL and SYCL.

namespace dnnl {
namespace impl {
namespace xpu {

using binary_t = std::vector<uint8_t>;
using device_uuid_t = std::tuple<uint64_t, uint64_t>;

struct device_uuid_hasher_t {
    size_t operator()(const device_uuid_t &uuid) const;
};

} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
