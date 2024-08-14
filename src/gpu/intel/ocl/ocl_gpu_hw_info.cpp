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

xpu::runtime_version_t get_driver_version(cl_device_id device) {
    cl_int err;
    xpu::runtime_version_t runtime_version(-1, -1, -1);

    size_t param_size = 0;
    err = clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, nullptr, &param_size);
    std::string driver_version(param_size, '\0');

    if (err == CL_SUCCESS) {
        err = clGetDeviceInfo(device, CL_DRIVER_VERSION, param_size,
                &driver_version[0], nullptr);
    }

    if (err != CL_SUCCESS
            || runtime_version.set_from_string(&driver_version[0])
                    != status::success) {
        runtime_version.major = 0;
        runtime_version.minor = 0;
        runtime_version.build = 0;
    }

    return runtime_version;
}

/// Tries to build a kernel with assembly instructions to check to see if the
/// OpenCL compiler supports microkernels.
bool try_building_with_microkernels(cl_context context, cl_device_id device) {
    const char *kernel_code = R""""(
        kernel void igc_check() {
            __asm__ volatile(
                    ".decl AA0 v_type=G type=ud num_elts=1\n"
                    ".decl AA1 v_type=G type=ud num_elts=1\n"
                    ".implicit_PSEUDO_INPUT AA0 offset=256 size=4\n"
                    ".implicit_PSEUDO_INPUT AA1 offset=256 size=4\n"
                    "mov (M1_NM,1) AA0(0,0)<1> AA1(0,0)<0;1,0>\n"
            );
        }
        )"""";
    cl_int err;
    /// Not using existing build infrastructure to avoid error messages in the CI logs
    xpu::ocl::wrapper_t<cl_program> program = xpu::ocl::make_wrapper(
            clCreateProgramWithSource(context, 1, &kernel_code, nullptr, &err));
    if (err != CL_SUCCESS) return false;
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    return err == CL_SUCCESS;
}

void init_gpu_hw_info(impl::engine_t *engine, cl_device_id device,
        cl_context context, uint32_t &ip_version, compute::gpu_arch_t &gpu_arch,
        int &gpu_product_family, int &stepping_id, uint64_t &native_extensions,
        bool &mayiuse_systolic, bool &mayiuse_ngen_kernels,
        bool &mayiuse_microkernels) {
    using namespace ngen;
    HW hw = HW::Unknown;
    Product product = {ProductFamily::Unknown, 0};
    jit::jit_generator<HW::Unknown>::detectHWInfo(context, device, hw, product);
    bool is_xelpg = (product.family == ngen::ProductFamily::ARL
            || product.family == ngen::ProductFamily::MTL);

    gpu_arch = jit::convert_ngen_arch_to_dnnl(hw);
    gpu_product_family = static_cast<int>(product.family);
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

    mayiuse_microkernels = get_driver_version(device)
            >= xpu::runtime_version_t(24, 22, 29735);
    if (!mayiuse_microkernels) {
        mayiuse_microkernels = try_building_with_microkernels(context, device);
    }

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
