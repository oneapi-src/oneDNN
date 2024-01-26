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

#include "sycl_hip_compat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {
namespace compat {

template <>
HIPcontext get_native(const ::sycl::device &device) {
    HIPdevice nativeDevice
            = ::sycl::get_native<::sycl::backend::ext_oneapi_hip>(device);
    HIPcontext nativeContext;
    if (hipDevicePrimaryCtxRetain(&nativeContext, nativeDevice) != hipSuccess) {
        throw std::runtime_error("Could not create a native context");
    }
    return nativeContext;
}
} // namespace compat
} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
