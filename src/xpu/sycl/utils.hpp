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

#ifndef XPU_SYCL_UTILS_HPP
#define XPU_SYCL_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "xpu/utils.hpp"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

using buffer_u8_t = ::sycl::buffer<uint8_t, 1>;

enum class backend_t { unknown, host, level0, opencl, nvidia, amd };

std::string to_string(backend_t backend);
std::string to_string(::sycl::info::device_type dev_type);
backend_t get_gpu_backend();

bool is_host(const ::sycl::device &dev);
bool is_host(const ::sycl::platform &plat);
backend_t get_backend(const ::sycl::device &dev);
bool are_equal(const ::sycl::device &lhs, const ::sycl::device &rhs);

status_t check_device(engine_kind_t eng_kind, const ::sycl::device &dev,
        const ::sycl::context &ctx);

bool dev_ctx_consistency_check(
        const ::sycl::device &dev, const ::sycl::context &ctx);

bool is_intel_device(const ::sycl::device &dev);
bool is_intel_platform(const ::sycl::platform &plat);

bool is_nvidia_gpu(const ::sycl::device &dev);
bool is_amd_gpu(const ::sycl::device &dev);

bool is_subdevice(const ::sycl::device &dev);

::sycl::device get_root_device(const ::sycl::device &dev);
::sycl::device get_parent_device(const ::sycl::device &dev);

std::vector<::sycl::device> get_devices(::sycl::info::device_type dev_type,
        backend_t backend = backend_t::unknown);

status_t get_device_index(size_t *index, const ::sycl::device &dev);

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
