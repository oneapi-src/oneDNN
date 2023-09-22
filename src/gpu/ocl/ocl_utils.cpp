/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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
#include <iostream>
#include <mutex>
#include <CL/cl_ext.h>

#include "gpu/ocl/ocl_gpu_engine.hpp"
#include "gpu/ocl/ocl_gpu_kernel.hpp"
#include "gpu/ocl/ocl_utils.hpp"

#ifndef DNNL_ENABLE_JIT_DUMP
#define DNNL_ENABLE_JIT_DUMP 1
#endif

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

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

template <typename T, typename F>
static std::string get_ocl_name(T obj, F get_func, cl_uint name_query) {
    size_t name_size;
    cl_int err = get_func(obj, name_query, 0, nullptr, &name_size);
    // Ignore error.
    if (err != CL_SUCCESS) return {};

    // Include null terminator explicitly - to safely overwrite it in
    // clGetKernelInfo
    std::string name(name_size, 0);
    err = get_func(obj, name_query, name_size, &name[0], nullptr);
    // Ignore error.
    if (err != CL_SUCCESS) return {};

    // Remove the null terminator as std::string already includes it
    name.resize(name_size - 1);
    return name;
}

static std::string get_kernel_name(cl_kernel kernel) {
    return get_ocl_name(kernel, clGetKernelInfo, CL_KERNEL_FUNCTION_NAME);
}

static std::string get_platform_name(cl_platform_id platform) {
    return get_ocl_name(platform, clGetPlatformInfo, CL_PLATFORM_NAME);
}

static bool is_intel_platform(cl_platform_id platform) {
    auto name = get_platform_name(platform);
    return name.find("Intel") != std::string::npos;
}

status_t check_device(
        engine_kind_t eng_kind, cl_device_id dev, cl_context ctx) {
    assert(dev && ctx);

    // Check device and context consistency.
    size_t dev_bytes;
    OCL_CHECK(
            clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, nullptr, &dev_bytes));

    std::vector<cl_device_id> ctx_devices(dev_bytes / sizeof(cl_device_id));
    OCL_CHECK(clGetContextInfo(
            ctx, CL_CONTEXT_DEVICES, dev_bytes, &ctx_devices[0], nullptr));

    bool found = false;
    for (size_t i = 0; i < ctx_devices.size(); ++i) {
        if (ctx_devices[i] == dev) {
            found = true;
            break;
        }
    }
    if (!found) return status::invalid_arguments;

    // Check engine kind and device consistency.
    cl_device_type dev_type;
    OCL_CHECK(clGetDeviceInfo(
            dev, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, nullptr));
    if ((eng_kind == engine_kind::cpu)
            && (dev_type & CL_DEVICE_TYPE_CPU) == 0) {
        return status::invalid_arguments;
    }
    if ((eng_kind == engine_kind::gpu)
            && (dev_type & CL_DEVICE_TYPE_GPU) == 0) {
        return status::invalid_arguments;
    }

    // Check that the platform is an Intel platform.
    cl_platform_id platform;
    OCL_CHECK(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    if (!is_intel_platform(platform)) return status::invalid_arguments;

    return status::success;
}

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
        if (!is_intel_platform(platforms[i])) continue;

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
                OCL_CHECK(clGetDeviceInfo(plat_devices[j], CL_DEVICE_VENDOR_ID,
                        sizeof(cl_uint), &vendor_id, nullptr));
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

    // Search the top level device unconditionally
    auto parent_device = device;
    auto top_level_device = device;
    while (parent_device) {
        top_level_device = parent_device;
        OCL_CHECK(clGetDeviceInfo(top_level_device, CL_DEVICE_PARENT_DEVICE,
                sizeof(cl_device_id), &parent_device, nullptr));
    }

    // Find the top level device in the list
    auto it = std::find(
            ocl_devices.begin(), ocl_devices.end(), top_level_device);
    if (it != ocl_devices.end()) {
        *index = it - ocl_devices.begin();
        return status::success;
    } else {
        *index = SIZE_MAX;
        return status::invalid_arguments;
    }
}

cl_platform_id get_ocl_platform(cl_device_id device) {
    cl_platform_id platform;
    cl_int err = clGetDeviceInfo(
            device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
    if (err != CL_SUCCESS) return nullptr;
    return platform;
}

cl_platform_id get_ocl_platform(engine_t *engine) {
    return utils::downcast<ocl_gpu_engine_t *>(engine)->platform();
}

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
    // compiled for. Using global indexing through `get_ocl_device_index` may
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
        cl_program program, cl_device_id device, compute::binary_t &binary) {
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
    std::vector<compute::binary_t> binaries(n_devices);
    for (size_t i = 0; i < n_devices; ++i) {
        binaries[i] = compute::binary_t(binarySize[i]);
        binary_pointers[i] = binaries[i].data();
    }

    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
            sizeof(uint8_t *) * n_devices, binary_pointers.data(), nullptr);
    OCL_CHECK(err);
    binary = binaries[device_idx];
    return status::success;
}

status_t get_ocl_program_binary(
        cl_kernel kernel, cl_device_id device, compute::binary_t &binary) {
    cl_int err;

    cl_program program;
    err = clGetKernelInfo(
            kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, nullptr);
    OCL_CHECK(err);

    return get_ocl_program_binary(program, device, binary);
}

#if DNNL_ENABLE_JIT_DUMP
void dump_kernel_binary(
        const compute::binary_t &binary, const std::string &name) {
    if (!get_jit_dump()) return;

    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    static int counter = 0;
    std::ostringstream fname;
    fname << "dnnl_dump_gpu_" << name << "." << counter << ".bin";

    FILE *fp = fopen(fname.str().c_str(), "wb+");

    // Ignore error.
    if (!fp) return;

    fwrite(binary.data(), binary.size(), 1, fp);
    fclose(fp);

    counter++;
}

void dump_kernel_binary(cl_kernel ocl_kernel) {
    if (!get_jit_dump()) return;

    cl_int err;

    size_t binary_size;
    err = clGetKernelInfo(ocl_kernel, CL_KERNEL_BINARY_PROGRAM_INTEL, 0,
            nullptr, &binary_size);
    // Ignore error.
    if (err != CL_SUCCESS) return;

    std::vector<uint8_t> binary(binary_size);
    err = clGetKernelInfo(ocl_kernel, CL_KERNEL_BINARY_PROGRAM_INTEL,
            binary.size(), binary.data(), nullptr);
    // Ignore error.
    if (err != CL_SUCCESS) return;

    auto name = get_kernel_name(ocl_kernel);
    // Ignore error.
    if (name.empty()) return;
    dump_kernel_binary(binary, name);
}

void dump_kernel_binary(
        const engine_t *engine, const compute::kernel_t &kernel) {
    if (!get_jit_dump()) return;
    auto *kernel_impl
            = utils::downcast<const ocl_gpu_kernel_t *>(kernel.impl());
    dump_kernel_binary(kernel_impl->ocl_kernel());
}
#else
void dump_kernel_binary(const engine_t *, const compute::kernel_t &) {}
void dump_kernel_binary(compute::binary_t binary, const std::string &name) {}
void dump_kernel_binary(cl_kernel) {}
#endif

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
        auto o = get_defines(options);
        std::string preprocess_cmd
                = std::string() + "cpp -P " + o.c_str() + " | clang-format";
        execute_command(preprocess_cmd, source);
        std::cout << "OCL_ARCH_OPTIONS: " << cl_options << std::endl;
    }
#endif
}

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

static status_t get_ocl_device_eu_count_intel(
        cl_device_id device, int32_t *eu_count) {
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

    *eu_count = (int32_t)(
            num_slices * num_sub_slices_per_slice * num_eus_per_sub_slice);
    return status::success;
}

status_t get_ocl_device_eu_count(cl_device_id device, int32_t *eu_count) {
    // Try to use Intel-specific slices/sub-slices to deduce EU count.
    auto status = get_ocl_device_eu_count_intel(device, eu_count);
    if (status == status::success) return status;

    // If failed, fall back to common OpenCL query.
    cl_uint max_compute_units = 0;
    OCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
            sizeof(max_compute_units), &max_compute_units, nullptr));
    *eu_count = (int32_t)max_compute_units;

    return status::success;
}

status_t clone_kernel(cl_kernel kernel, cl_kernel *cloned_kernel) {
    cl_int err;
#if !defined(DNNL_SYCL_HIP) && !defined(DNNL_SYCL_CUDA) \
        && defined(CL_VERSION_2_1)
    *cloned_kernel = clCloneKernel(kernel, &err);
    OCL_CHECK(err);
#else
    // clCloneKernel is not available - recreate from the program.
    auto name = get_kernel_name(kernel);

    cl_program program;
    err = clGetKernelInfo(
            kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, nullptr);
    OCL_CHECK(err);

    *cloned_kernel = clCreateKernel(program, name.c_str(), &err);
    OCL_CHECK(err);
#endif

    return status::success;
}

status_t create_ocl_program(gpu::ocl::ocl_wrapper_t<cl_program> &ocl_program,
        cl_device_id dev, cl_context ctx,
        const gpu::compute::binary_t &binary) {
    cl_int err;
    const unsigned char *binary_buffer = binary.data();
    size_t binary_size = binary.size();
    assert(binary_size > 0);

    ocl_program = clCreateProgramWithBinary(
            ctx, 1, &dev, &binary_size, &binary_buffer, nullptr, &err);
    OCL_CHECK(err);
    err = clBuildProgram(ocl_program, 1, &dev, nullptr, nullptr, nullptr);
    OCL_CHECK(err);

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
