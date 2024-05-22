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

#include "gpu/intel/ocl/ocl_gpu_hw_info.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"

#include "gpu/intel/jit/binary_format.hpp"
#include "gpu/intel/jit/jit_generator.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"

#ifndef CL_DEVICE_IP_VERSION_INTEL
#define CL_DEVICE_IP_VERSION_INTEL 0x4250
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

void init_gpu_hw_info(engine_t *engine, cl_device_id device, cl_context context,
        uint32_t &ip_version, compute::gpu_arch_t &gpu_arch, int &stepping_id,
        uint64_t &native_extensions, bool &mayiuse_systolic,
        bool &mayiuse_ngen_kernels) {
    using namespace ngen;
    HW hw = HW::Unknown;
    Product product = {ProductFamily::Unknown, 0};
    jit::jit_generator<HW::Unknown>::detectHWInfo(context, device, hw, product);
    bool is_xelpg = (product.family == ngen::ProductFamily::ARL
            || product.family == ngen::ProductFamily::MTL);

    gpu_arch = jit::convert_ngen_arch_to_dnnl(hw);
    stepping_id = product.stepping;

    mayiuse_systolic = false;
    status_t ret
            = get_ocl_device_enabled_systolic_intel(device, mayiuse_systolic);
    assert(ret == CL_SUCCESS);
    ret = get_ocl_device_enabled_native_float_atomics(
            device, native_extensions, is_xelpg);
    assert(ret == CL_SUCCESS);
    MAYBE_UNUSED(ret);

    auto status
            = jit::gpu_supports_binary_format(&mayiuse_ngen_kernels, engine);
    if (status != status::success) mayiuse_ngen_kernels = false;

    ip_version = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_IP_VERSION_INTEL, sizeof(ip_version),
                &ip_version, nullptr)
            != CL_SUCCESS)
        ip_version = 0;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
