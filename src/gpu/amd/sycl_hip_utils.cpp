/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

bool compare_hip_devices(const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_hip_handle = compat::get_native<HIPdevice>(lhs);
    auto rhs_hip_handle = compat::get_native<HIPdevice>(rhs);

    return lhs_hip_handle == rhs_hip_handle;
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
