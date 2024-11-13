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

#ifndef COMMON_XPU_SYCL_COMPAT_HPP
#define COMMON_XPU_SYCL_COMPAT_HPP

// This file contains a common SYCL compatibility layer. All vendor specific
// SYCL code that requires compatbility must reside in the vendor directories.

#include "xpu/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {
namespace compat {

void *get_native(const ::sycl::device &dev);
void *get_native(const ::sycl::context &ctx);

template <typename native_object_t, typename sycl_object_t>
native_object_t get_native(const sycl_object_t &sycl_object) {
    return reinterpret_cast<native_object_t>(get_native(sycl_object));
}

// Automatically use host_task if it is supported by compiler,
// otherwise fall back to codeplay_host_task.
template <typename H, typename F>
inline auto host_task_impl(H &cgh, F &&f, int) -> decltype(cgh.host_task(f)) {
    cgh.host_task(f);
}

template <typename H, typename F>
inline auto host_task_impl(H &cgh, F &&f, long)
        -> decltype(cgh.codeplay_host_task(f)) {
    cgh.codeplay_host_task(f);
}

template <typename H, typename F>
inline void host_task(H &cgh, F &&f) {
    // Third argument is 0 (int) which prefers the
    // host_task option if both are available.
    host_task_impl(cgh, f, 0);
}

constexpr auto target_device = ::sycl::target::device;

template <typename T, int dims>
using local_accessor = ::sycl::accessor<T, dims,
        ::sycl::access::mode::read_write, ::sycl::access::target::local>;

using ext_intel_gpu_slices = ::sycl::ext::intel::info::device::gpu_slices;
using ext_intel_gpu_subslices_per_slice
        = ::sycl::ext::intel::info::device::gpu_subslices_per_slice;

inline const auto &cpu_selector_v = ::sycl::cpu_selector_v;
inline const auto &gpu_selector_v = ::sycl::gpu_selector_v;

} // namespace compat
} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
