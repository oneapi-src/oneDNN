/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "gpu/compute/kernel.hpp"
#include "sycl/sycl_compat.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

using device_uuid_t = std::tuple<uint64_t, uint64_t>;
device_uuid_t get_device_uuid(const ::sycl::device &dev);

// including sycl_engine_base.hpp leads to circular dependencies, w/a for now.
class sycl_engine_base_t;

status_t sycl_create_kernel_with_level_zero(
        std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const std::string &kernel_name, const sycl_engine_base_t *sycl_engine,
        const gpu::compute::binary_t &binary);

bool compare_ze_devices(const ::sycl::device &lhs, const ::sycl::device &rhs);

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // SYCL_LEVEL_ZERO_UTILS_HPP
