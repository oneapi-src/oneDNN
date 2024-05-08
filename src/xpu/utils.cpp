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

#include <tuple>
#include <vector>

#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {

size_t device_uuid_hasher_t::operator()(const device_uuid_t &uuid) const {
    const size_t seed = hash_combine(0, std::get<0>(uuid));
    return hash_combine(seed, std::get<1>(uuid));
}

} // namespace xpu
} // namespace impl
} // namespace dnnl
