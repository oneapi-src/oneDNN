/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef GPU_NVIDIA_SYCL_CUDA_COMPAT_HPP
#define GPU_NVIDIA_SYCL_CUDA_COMPAT_HPP

#include <cuda.h>

#include "sycl/sycl_compat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {
namespace compat {

using interop_handle = ::sycl::interop_handle;
template <typename T, typename U>
T get_native_mem(const interop_handle &ih, U acc) {
    return reinterpret_cast<T>(
            ih.get_native_mem<::sycl::backend::ext_oneapi_cuda>(acc));
}

template <typename T>
void host_task(::sycl::handler &cgh, const T &task) {
    cgh.host_task(task);
}

template <typename native_object_t, typename sycl_object_t>
native_object_t get_native(const sycl_object_t &sycl_object) {
    static_assert(!std::is_same_v<native_object_t, CUcontext>,
            "Use compat::get_native_context() to obtain the native cuda "
            "context");
    auto handle
            = ::sycl::get_native<::sycl::backend::ext_oneapi_cuda>(sycl_object);
    return reinterpret_cast<native_object_t>(handle);
}

inline CUcontext get_native_context(const ::sycl::device &sycl_device) {
    CUdevice cu_device = get_native<CUdevice>(sycl_device);
    CUcontext cu_ctx;
    const CUresult cu_result = cuDevicePrimaryCtxRetain(&cu_ctx, cu_device);
    assert(CUDA_SUCCESS == cu_result);
    MAYBE_UNUSED(cu_result);
    return cu_ctx;
}

} // namespace compat
} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
