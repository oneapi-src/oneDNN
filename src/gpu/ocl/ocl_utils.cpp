/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <CL/cl_ext.h>

#include "gpu/ocl/ocl_gpu_kernel.hpp"
#include "gpu/ocl/ocl_utils.hpp"

#ifndef DNNL_ENABLE_JIT_DUMP
#define DNNL_ENABLE_JIT_DUMP 1
#endif

#ifndef CL_KERNEL_BINARY_PROGRAM_INTEL
#define CL_KERNEL_BINARY_PROGRAM_INTEL 0x407D
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t get_ocl_devices(
        std::vector<cl_device_id> *devices, cl_device_type device_type) {
    cl_uint num_platforms = 0;

    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    // No platforms - a valid scenario
    if (err == CL_PLATFORM_NOT_FOUND_KHR) return status::success;

    OCL_CHECK(err);

    std::vector<cl_platform_id> platforms(num_platforms);
    OCL_CHECK(clGetPlatformIDs(num_platforms, &platforms[0], nullptr));

    for (size_t i = 0; i < platforms.size(); ++i) {
        cl_uint num_devices = 0;
        cl_int err = clGetDeviceIDs(
                platforms[i], device_type, 0, nullptr, &num_devices);

        if (!utils::one_of(err, CL_SUCCESS, CL_DEVICE_NOT_FOUND)) {
            return status::runtime_error;
        }

        if (num_devices != 0) {
            std::vector<cl_device_id> plat_devices;
            plat_devices.resize(num_devices);
            OCL_CHECK(clGetDeviceIDs(platforms[i], device_type, num_devices,
                    &plat_devices[0], nullptr));

            // Use Intel devices only
            for (size_t j = 0; j < plat_devices.size(); ++j) {
                cl_uint vendor_id;
                clGetDeviceInfo(plat_devices[j], CL_DEVICE_VENDOR_ID,
                        sizeof(cl_uint), &vendor_id, nullptr);
                if (vendor_id == 0x8086) {
                    devices->push_back(plat_devices[j]);
                }
            }
        }
    }
    // No devices found but still return success
    return status::success;
}

status_t get_ocl_device_index(size_t *index, cl_device_id device) {
    std::vector<cl_device_id> ocl_devices;
    CHECK(get_ocl_devices(&ocl_devices, CL_DEVICE_TYPE_GPU));

    auto it = std::find(ocl_devices.begin(), ocl_devices.end(), device);
    if (it == ocl_devices.end()) return status::invalid_arguments;
    *index = it - ocl_devices.begin();
    return status::success;
}

status_t get_ocl_kernel_arg_type(compute::scalar_type_t *type,
        cl_kernel ocl_kernel, cl_uint idx, bool allow_undef) {
    char s_type[16];
    OCL_CHECK(clGetKernelArgInfo(ocl_kernel, idx, CL_KERNEL_ARG_TYPE_NAME,
            sizeof(s_type), s_type, nullptr));
#define CASE(x) \
    if (!strcmp(STRINGIFY(x), s_type)) { \
        *type = compute::scalar_type_t::_##x; \
        return status::success; \
    }
    CASE(char)
    CASE(float)
    CASE(half)
    CASE(int)
    CASE(long)
    CASE(short)
    CASE(uchar)
    CASE(uint)
    CASE(ulong)
    CASE(ushort)
    CASE(zero_pad_mask_t)
#undef CASE

    if (allow_undef) {
        *type = compute::scalar_type_t::undef;
        return status::success;
    }

    assert(!"Not expected");
    return status::runtime_error;
}

cl_mem clCreateBuffer_wrapper(cl_context context, cl_mem_flags flags,
        size_t size, void *host_ptr, cl_int *errcode_ret) {
    return clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}

#if DNNL_ENABLE_JIT_DUMP
void dump_kernel_binary(
        const engine_t *engine, const compute::kernel_t &binary_kernel) {
    if (!get_jit_dump()) return;

    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    static int counter = 0;
    compute::kernel_t realized_kernel;
    auto status = binary_kernel.realize(&realized_kernel, engine, nullptr);

    // Ignore error.
    if (status != status::success) return;

    auto *kernel
            = utils::downcast<const ocl_gpu_kernel_t *>(realized_kernel.impl());

    cl_int err;

    size_t binary_size;
    err = clGetKernelInfo(kernel->ocl_kernel(), CL_KERNEL_BINARY_PROGRAM_INTEL,
            0, nullptr, &binary_size);

    // Ignore error.
    if (err != CL_SUCCESS) return;

    std::vector<uint8_t> binary(binary_size);
    err = clGetKernelInfo(kernel->ocl_kernel(), CL_KERNEL_BINARY_PROGRAM_INTEL,
            binary.size(), binary.data(), nullptr);

    // Ignore error.
    if (err != CL_SUCCESS) return;

    std::ostringstream fname;
    auto *kernel_name
            = utils::downcast<const ocl_gpu_kernel_t *>(binary_kernel.impl())
                      ->name();
    fname << "dnnl_dump_gpu_" << kernel_name << "." << counter << ".bin";

    FILE *fp = fopen(fname.str().c_str(), "wb+");

    // Ignore error.
    if (!fp) return;

    fwrite(binary.data(), binary.size(), 1, fp);
    fclose(fp);

    counter++;
}
#else
void dump_kernel_binary(const engine_t *, const compute::kernel_t &) {}
#endif

status_t get_kernel_arg_types(cl_kernel ocl_kernel,
        std::vector<gpu::compute::scalar_type_t> *arg_types) {
    cl_uint nargs;
    OCL_CHECK(clGetKernelInfo(
            ocl_kernel, CL_KERNEL_NUM_ARGS, sizeof(nargs), &nargs, nullptr));

    *arg_types = std::vector<gpu::compute::scalar_type_t>(nargs);

    for (cl_uint i = 0; i < nargs; i++) {
        gpu::compute::scalar_type_t type {};
        CHECK(gpu::ocl::get_ocl_kernel_arg_type(
                &type, ocl_kernel, i, /*allow_undef=*/true));
        (*arg_types)[i] = type;
    }

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
