/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef SYCL_LEVEL_ZERO_UTILS_HPP
#define SYCL_LEVEL_ZERO_UTILS_HPP

#include <memory>
#include <string>
#include <vector>

#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/sycl/compat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

class engine_t;

xpu::device_uuid_t get_device_uuid(const ::sycl::device &dev);

status_t sycl_create_kernels_with_level_zero(
        std::vector<std::unique_ptr<::sycl::kernel>> &sycl_kernels,
        const std::vector<const char *> &kernel_names,
        const gpu::intel::sycl::engine_t *sycl_engine,
        const xpu::binary_t &binary);

bool compare_ze_devices(const ::sycl::device &lhs, const ::sycl::device &rhs);

status_t func_zeModuleGetNativeBinary(ze_module_handle_t hModule, size_t *pSize,
        uint8_t *pModuleNativeBinary);

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // SYCL_LEVEL_ZERO_UTILS_HPP
