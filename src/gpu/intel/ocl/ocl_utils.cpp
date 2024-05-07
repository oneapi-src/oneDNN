/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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
#include <cstring>
#include <iostream>
#include <CL/cl_ext.h>

#include "gpu/intel/ocl/ocl_gpu_engine.hpp"
#include "gpu/intel/ocl/ocl_gpu_kernel.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"
#include "xpu/ocl/utils.hpp"

#ifndef CL_KERNEL_BINARY_PROGRAM_INTEL
#define CL_KERNEL_BINARY_PROGRAM_INTEL 0x407D
#endif

#ifndef CL_DEVICE_NUM_SLICES_INTEL
#define CL_DEVICE_NUM_SLICES_INTEL 0x4252
#endif

#ifndef CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL
#define CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL 0x4253
#endif

#ifndef CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL
#define CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL 0x4254
#endif

#ifndef CL_DEVICE_FEATURE_CAPABILITIES_INTEL
#define CL_DEVICE_FEATURE_CAPABILITIES_INTEL 0x4256
#endif

#ifndef CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT
#define CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT 0x4231
#endif

#ifndef CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT
#define CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT 0x4232
#endif

#ifndef CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT
#define CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT 0x4233
#endif

#ifndef CL_DEVICE_ATOMIC_FLAGS
#define CL_DEVICE_ATOMIC_FLAGS
#define CL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT (1 << 0)
#define CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT (1 << 1)
#define CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT (1 << 2)
#define CL_DEVICE_LOCAL_FP_ATOMIC_LOAD_STORE_EXT (1 << 16)
#define CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT (1 << 17)
#define CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT (1 << 18)
#endif

#ifndef CL_DEVICE_FEATURE_FLAG_DPAS_INTEL
#define CL_DEVICE_FEATURE_FLAG_DPAS_INTEL (1 << 1)
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t get_ocl_kernel_arg_type(compute::scalar_type_t *type,
        cl_kernel ocl_kernel, cl_uint idx, bool allow_undef) {
    char s_type[16];
    auto cl_status = clGetKernelArgInfo(ocl_kernel, idx,
            CL_KERNEL_ARG_TYPE_NAME, sizeof(s_type), s_type, nullptr);
    if (cl_status == CL_SUCCESS) {
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
    }

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

static status_t get_number_devices(cl_program program, size_t *n_devices) {
    cl_int err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES,
            sizeof(size_t), n_devices, nullptr);
    OCL_CHECK(err);
    return status::success;
}

status_t get_ocl_program_binary_size(
        cl_kernel kernel, cl_device_id device, size_t *size) {
    cl_program program;
    cl_int err = clGetKernelInfo(
            kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, nullptr);
    OCL_CHECK(err);

    size_t n_devices = 0;
    CHECK(get_number_devices(program, &n_devices));

    std::vector<size_t> binary_sizes(n_devices);
    err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
            sizeof(size_t) * n_devices, binary_sizes.data(), nullptr);
    OCL_CHECK(err);

    // Identify local device index in the list of devices the program was
    // compiled for. Using global indexing through `get_device_index` may
    // fail due to presence of two or more physical devices in the system.
    std::vector<cl_device_id> devices(n_devices);
    err = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
            sizeof(cl_device_id) * n_devices, devices.data(), nullptr);
    OCL_CHECK(err);

    auto device_it = std::find(devices.begin(), devices.end(), device);
    if (device_it == devices.end()) return status::invalid_arguments;

    size_t device_idx = std::distance(devices.begin(), device_it);
    (*size) = binary_sizes[device_idx];
    return status::success;
}

status_t get_ocl_program_binary(
        cl_program program, cl_device_id device, xpu::binary_t &binary) {
    size_t n_devices = 0;
    CHECK(get_number_devices(program, &n_devices));

    std::vector<size_t> binarySize(n_devices);
    cl_int err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
            sizeof(size_t) * n_devices, binarySize.data(), nullptr);
    OCL_CHECK(err);

    std::vector<cl_device_id> devices(n_devices);
    err = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
            sizeof(cl_device_id) * n_devices, devices.data(), nullptr);
    OCL_CHECK(err);

    size_t device_idx = std::distance(
            devices.begin(), std::find(devices.begin(), devices.end(), device));
    std::vector<uint8_t *> binary_pointers(n_devices);
    std::vector<xpu::binary_t> binaries(n_devices);
    for (size_t i = 0; i < n_devices; ++i) {
        binaries[i] = xpu::binary_t(binarySize[i]);
        binary_pointers[i] = binaries[i].data();
    }

    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
            sizeof(uint8_t *) * n_devices, binary_pointers.data(), nullptr);
    OCL_CHECK(err);
    binary = binaries[device_idx];
    return status::success;
}

status_t get_ocl_program_binary(
        cl_kernel kernel, cl_device_id device, xpu::binary_t &binary) {
    cl_int err;

    cl_program program;
    err = clGetKernelInfo(
            kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, nullptr);
    OCL_CHECK(err);

    return get_ocl_program_binary(program, device, binary);
}

status_t get_ocl_kernel_binary(cl_kernel ocl_kernel, xpu::binary_t &binary) {
    binary.clear();
    size_t binary_size;
    OCL_CHECK(clGetKernelInfo(ocl_kernel, CL_KERNEL_BINARY_PROGRAM_INTEL, 0,
            nullptr, &binary_size));
    binary.resize(binary_size);
    OCL_CHECK(clGetKernelInfo(ocl_kernel, CL_KERNEL_BINARY_PROGRAM_INTEL,
            binary.size(), binary.data(), nullptr));
    return status::success;
}

void debugdump_processed_source(const std::string &source,
        const std::string &options, const std::string &cl_options) {
#if defined(__linux__) && defined(DNNL_DEV_MODE)
    if (get_verbose(verbose_t::debuginfo) >= 10) {
        auto get_defines = [](const std::string &from) {
            std::string ret;
            size_t pos = 0;
            while (pos < from.length()) {
                // Find next define argument
                pos = from.find("-D", pos);

                // Generate argument, quotes are interpreted literally, but
                // other special shell characters need escaped. Does not
                // currently handle quotes with the ' character or nested quotes
                char quote_parity = true;
                while (pos < from.length()) {
                    if (quote_parity
                            && utils::one_of(from[pos], '~', '#', '$', '&', '*',
                                    '(', ')', '\\', '|', '[', ']', '{', '}',
                                    ';', '\'', '<', '>', '/', '?', '!')) {
                        ret += '\\';
                    }
                    ret += from[pos];
                    if (from[pos] == '"') quote_parity ^= true;
                    if (from[pos] == ' ' && quote_parity) break;

                    pos++;
                }
            }
            return ret;
        };
        auto execute_command = [](const std::string &cmd,
                                       const std::string &stdin) {
            std::string result;
            std::array<char, 256> buffer;
            FILE *pipe = popen(cmd.c_str(), "w");
            fputs(stdin.c_str(), pipe);
            if (pipe) {
                while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                    result += buffer.data();
                }
            }
            pclose(pipe);
            return result;
        };

        // Run utilities to evaluate preprocessor defines and format the file
        // Theoretically, we can accomplish this task with libclang, but it
        // seems more work than it is worth. Instead, wrapping this in OCL_DEBUG
        // so that calls to the system are not included in the default build.

        // Due to the use of a different C preprocessor, warnings should not be
        // ignored, as they may correspond to a different behavior in the OpenCL
        // C preprocessor
        auto o = get_defines(options) + get_defines(cl_options);
        std::string preprocess_cmd
                = std::string() + "cpp -P " + o + " | clang-format";
        execute_command(preprocess_cmd, source);
        std::cout << "OCL_ARCH_OPTIONS: " << cl_options << std::endl;
    }
#endif
}

status_t get_kernel_arg_types(cl_kernel ocl_kernel,
        std::vector<gpu::intel::compute::scalar_type_t> *arg_types) {
    cl_uint nargs;
    OCL_CHECK(clGetKernelInfo(
            ocl_kernel, CL_KERNEL_NUM_ARGS, sizeof(nargs), &nargs, nullptr));

    *arg_types = std::vector<gpu::intel::compute::scalar_type_t>(nargs);

    for (cl_uint i = 0; i < nargs; i++) {
        gpu::intel::compute::scalar_type_t type {};
        CHECK(gpu::intel::ocl::get_ocl_kernel_arg_type(
                &type, ocl_kernel, i, /*allow_undef=*/true));
        (*arg_types)[i] = type;
    }

    return status::success;
}

static status_t get_ocl_device_eu_count_intel(cl_device_id device,
        gpu::intel::compute::gpu_arch_t arch, int32_t *eu_count) {
    cl_uint num_slices = 0;
    cl_uint num_sub_slices_per_slice = 0;
    cl_uint num_eus_per_sub_slice = 0;

    OCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NUM_SLICES_INTEL,
            sizeof(num_slices), &num_slices, nullptr));
    OCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL,
            sizeof(num_sub_slices_per_slice), &num_sub_slices_per_slice,
            nullptr));
    OCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL,
            sizeof(num_eus_per_sub_slice), &num_eus_per_sub_slice, nullptr));

    if (arch == gpu::intel::compute::gpu_arch_t::xe2)
        num_eus_per_sub_slice = 8; /* runtime reports incorrect value */

    *eu_count = (int32_t)(
            num_slices * num_sub_slices_per_slice * num_eus_per_sub_slice);
    return status::success;
}

status_t get_ocl_device_enabled_systolic_intel(
        cl_device_id device, bool &enabled_systolic) {
    cl_bitfield res;
    OCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_FEATURE_CAPABILITIES_INTEL,
            sizeof(cl_bitfield), &res, nullptr));
    enabled_systolic = res & CL_DEVICE_FEATURE_FLAG_DPAS_INTEL;
    return status::success;
}

status_t get_ocl_device_enabled_native_float_atomics(
        cl_device_id device, uint64_t &native_extensions, bool is_xelpg) {
    cl_bitfield res;

    cl_int err
            = clGetDeviceInfo(device, CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT,
                    sizeof(cl_bitfield), &res, nullptr);
    if (err == status::success) {
        if (res & CL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT
                && res & CL_DEVICE_LOCAL_FP_ATOMIC_LOAD_STORE_EXT)
            native_extensions |= (uint64_t)
                    gpu::intel::compute::native_ext_t::fp16_atomic_load_store;
        if (res & CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT
                && res & CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)
            native_extensions |= (uint64_t)
                    gpu::intel::compute::native_ext_t::fp16_atomic_add;
        if (res & CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT
                && res & CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)
            native_extensions |= (uint64_t)
                    gpu::intel::compute::native_ext_t::fp16_atomic_min_max;
    }

    err = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT,
            sizeof(cl_bitfield), &res, nullptr);
    if (err == status::success) {
        if (res & CL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT
                && res & CL_DEVICE_LOCAL_FP_ATOMIC_LOAD_STORE_EXT)
            native_extensions |= (uint64_t)
                    gpu::intel::compute::native_ext_t::fp32_atomic_load_store;
        if (res & CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT
                && res & CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)
            native_extensions |= (uint64_t)
                    gpu::intel::compute::native_ext_t::fp32_atomic_add;
        if (res & CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT
                && res & CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)
            native_extensions |= (uint64_t)
                    gpu::intel::compute::native_ext_t::fp32_atomic_min_max;
    }

    // XeLPG lacks native support for f64 atomics.
    if (!is_xelpg) {
        err = clGetDeviceInfo(device,
                CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT,
                sizeof(cl_bitfield), &res, nullptr);
        if (err == status::success) {
            if (res & CL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT
                    && res & CL_DEVICE_LOCAL_FP_ATOMIC_LOAD_STORE_EXT)
                native_extensions |= (uint64_t)gpu::intel::compute::
                        native_ext_t::fp64_atomic_load_store;
            if (res & CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT
                    && res & CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)
                native_extensions |= (uint64_t)
                        gpu::intel::compute::native_ext_t::fp64_atomic_add;
            if (res & CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT
                    && res & CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)
                native_extensions |= (uint64_t)
                        gpu::intel::compute::native_ext_t::fp64_atomic_min_max;
        }
    }

    return status::success;
}

status_t get_ocl_device_eu_count(cl_device_id device,
        gpu::intel::compute::gpu_arch_t arch, int32_t *eu_count) {
    // Try to use Intel-specific slices/sub-slices to deduce EU count.
    auto status = get_ocl_device_eu_count_intel(device, arch, eu_count);
    if (status == status::success) return status;

    // If failed, fall back to common OpenCL query.
    cl_uint max_compute_units = 0;
    OCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
            sizeof(max_compute_units), &max_compute_units, nullptr));
    *eu_count = (int32_t)max_compute_units;

    return status::success;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
