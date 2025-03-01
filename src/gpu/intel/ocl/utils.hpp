/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_UTILS_HPP
#define GPU_INTEL_OCL_UTILS_HPP

#include <string.h>
#include <string>
#include <utility>
#include <vector>
#include <CL/cl.h>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "common/cpp_compat.hpp"
#include "common/verbose.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/kernel_arg_list.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "xpu/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

enum { OCL_BUFFER_ALIGNMENT = 128 };

bool mayiuse_microkernels(const impl::engine_t *engine);

status_t get_ocl_kernel_arg_type(compute::scalar_type_t *type,
        cl_kernel ocl_kernel, int idx, bool allow_undef = false);

status_t get_ocl_program_binary(
        cl_program program, cl_device_id device, xpu::binary_t &binary);

status_t get_ocl_program_binary(
        cl_kernel kernel, cl_device_id device, xpu::binary_t &binary);

status_t get_ocl_kernel_binary(cl_kernel ocl_kernel, xpu::binary_t &binary);

status_t get_ocl_program_binary_size(
        cl_kernel kernel, cl_device_id device, size_t *size);

void debugdump_processed_source(const std::string &source,
        const std::string &options, const std::string &ocl_options);

status_t get_kernel_arg_types(cl_kernel ocl_kernel,
        std::vector<gpu::intel::compute::scalar_type_t> *arg_types);

status_t get_ocl_device_eu_count(cl_device_id device,
        gpu::intel::compute::gpu_arch_t arch, int32_t *eu_count);

status_t get_ocl_device_enabled_systolic_intel(
        cl_device_id device, bool &systolic_enabled);

status_t get_ocl_device_enabled_native_float_atomics(
        cl_device_id device, uint64_t &native_extensions, bool is_xelpg);

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
