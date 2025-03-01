/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GPU_INTEL_SYCL_UTILS_HPP
#define GPU_INTEL_SYCL_UTILS_HPP

#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/ocl/engine.hpp"
#include "xpu/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

class engine_t;

::sycl::nd_range<3> to_sycl_nd_range(
        const gpu::intel::compute::nd_range_t &range);

status_t create_ocl_engine(
        std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t>
                *ocl_engine,
        const gpu::intel::sycl::engine_t *engine);

status_t get_kernel_binary(const ::sycl::kernel &kernel, xpu::binary_t &binary);

status_t create_ocl_engine(
        std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t>
                *ocl_engine,
        const gpu::intel::sycl::engine_t *engine);

gpu_utils::device_id_t device_id(const ::sycl::device &dev);

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
