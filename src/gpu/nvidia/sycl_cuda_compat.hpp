/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <CL/sycl/backend/cuda.hpp>
#pragma clang diagnostic pop

#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {
namespace compat {

#if DNNL_USE_SYCL121_API
using interop_handle = ::sycl::interop_handler;
template <typename T, typename U>
T get_native_mem(const interop_handle &ih, U acc) {
    return reinterpret_cast<T>(ih.get_mem<::sycl::backend::cuda>(acc));
}

template <typename T>
void host_task(::sycl::handler &cgh, const T &task) {
    cgh.interop_task(task);
}

#else
using interop_handle = ::sycl::interop_handle;
template <typename T, typename U>
T get_native_mem(const interop_handle &ih, U acc) {
    return reinterpret_cast<T>(ih.get_native_mem<::sycl::backend::cuda>(acc));
}

template <typename T>
void host_task(::sycl::handler &cgh, const T &task) {
    cgh.host_task(task);
}

#endif

} // namespace compat
} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
