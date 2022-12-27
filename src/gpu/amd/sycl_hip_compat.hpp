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

#ifndef GPU_AMD_SYCL_HIP_COMPAT_HPP
#define GPU_AMD_SYCL_HIP_COMPAT_HPP

#include <hip/hip_runtime.h>

#include "sycl/sycl_compat.hpp"

#include "gpu/amd/sycl_hip_utils.hpp"

#if __has_include(<sycl/ext/oneapi/backend/hip.hpp>)
#include <sycl/ext/oneapi/backend/hip.hpp>
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {
namespace compat {

using interop_handle = ::sycl::interop_handle;
template <typename T, typename U>
T get_native_mem(const interop_handle &ih, U acc) {
    return reinterpret_cast<T>(
            ih.get_native_mem<::sycl::backend::ext_oneapi_hip>(acc));
}

template <typename T>
void host_task(::sycl::handler &cgh, const T &task) {
    cgh.host_task(task);
}

template <typename native_object_t, typename sycl_object_t>
native_object_t get_native(const sycl_object_t &sycl_object) {
    auto handle
            = ::sycl::get_native<::sycl::backend::ext_oneapi_hip>(sycl_object);
    return reinterpret_cast<native_object_t>(handle);
}

} // namespace compat
} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
