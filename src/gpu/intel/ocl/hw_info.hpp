/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_HW_INFO_HPP
#define GPU_INTEL_OCL_HW_INFO_HPP

#include <CL/cl.h>

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/device_info.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

xpu::runtime_version_t get_driver_version(cl_device_id device);

status_t init_gpu_hw_info(impl::engine_t *engine, cl_device_id device,
        cl_context context, uint32_t &ip_version, compute::gpu_arch_t &gpu_arch,
        int &gpu_product_family, int &stepping_id, uint64_t &native_extensions,
        bool &mayiuse_systolic, bool &mayiuse_ngen_kernels);

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_OCL_HW_INFO_HPP
