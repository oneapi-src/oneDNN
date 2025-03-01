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

#include <algorithm>
#include <cstring>
#include <iostream>
#include <mutex>
#include <CL/cl_ext.h>

#include "gpu/intel/ocl/engine.hpp"
#include "gpu/intel/ocl/hw_info.hpp"
#include "gpu/intel/ocl/kernel.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "xpu/ocl/utils.hpp"

#ifdef DNNL_WITH_SYCL
#include "gpu/intel/sycl/engine.hpp"
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
    xpu::ocl::wrapper_t<cl_program> program(
            clCreateProgramWithSource(context, 1, &kernel_code, nullptr, &err));
    if (err != CL_SUCCESS) return false;
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    return err == CL_SUCCESS;
}

int get_sycl_ocl_device_and_context(
        xpu::ocl::wrapper_t<cl_context> &ocl_context,
        xpu::ocl::wrapper_t<cl_device_id> &ocl_device,
        const impl::engine_t *engine) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    auto *sycl_engine = utils::downcast<const sycl::engine_t *>(engine);
    auto &device = sycl_engine->device();

    auto be = xpu::sycl::get_backend(device);
    if (be == xpu::sycl::backend_t::opencl) {
        cl_int err = CL_SUCCESS;
        auto ocl_dev = xpu::sycl::compat::get_native<cl_device_id>(device);
        ocl_device = xpu::ocl::make_wrapper(ocl_dev, true);

        ocl_context = xpu::ocl::make_wrapper(
                clCreateContext(nullptr, 1, &ocl_dev, nullptr, nullptr, &err),
                true);
        if (err) return -1;
    } else if (be == xpu::sycl::backend_t::level0) {
        std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t> ocl_engine;
        auto err
                = gpu::intel::sycl::create_ocl_engine(&ocl_engine, sycl_engine);
        if (err != status::success) return -1;
        ocl_device = xpu::ocl::make_wrapper(ocl_engine->device(), true);
        ocl_context = xpu::ocl::make_wrapper(ocl_engine->context(), true);
    }
#endif
    return 0;
}

bool mayiuse_microkernels(const impl::engine_t *engine) {
    auto mayiuse_mk = [](const impl::engine_t *engine) {
        xpu::ocl::wrapper_t<cl_device_id> ocl_device;
        xpu::ocl::wrapper_t<cl_context> ocl_context;

        switch (engine->runtime_kind()) {
            case runtime_kind::sycl: {
                auto err = get_sycl_ocl_device_and_context(
                        ocl_context, ocl_device, engine);
                if (err) return false;
            } break;
            case runtime_kind::ocl: {
                const engine_t *eng = utils::downcast<const engine_t *>(engine);
                ocl_device = xpu::ocl::make_wrapper(eng->device(), true);
                ocl_context = xpu::ocl::make_wrapper(eng->context(), true);
            } break;
            default: return false;
        }

        bool mayiuse_microkernels = get_driver_version(ocl_device)
                >= xpu::runtime_version_t(24, 22, 29735);
        if (!mayiuse_microkernels) {
            mayiuse_microkernels
                    = try_building_with_microkernels(ocl_context, ocl_device);
        }
        return mayiuse_microkernels;
    };

    static std::map<engine_id_t, bool> engine_microkernel_map {
            {engine->engine_id(), mayiuse_mk(engine)}};

    static std::mutex map_mutex;
    std::lock_guard<std::mutex> map_lock(map_mutex);
    auto it = engine_microkernel_map.find(engine->engine_id());
    if (it != std::end(engine_microkernel_map)) { return it->second; }
    return engine_microkernel_map[engine->engine_id()] = mayiuse_mk(engine);
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
    // Start with standard OpenCL query.
    cl_uint max_compute_units = 0;
    OCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
            sizeof(max_compute_units), &max_compute_units, nullptr));

    // Try to use Intel-specific slice/sub-slice queries to correct EU count
    //   for certain buggy drivers.
    bool ok = true;

#ifdef _WIN32
    // But don't try this on Windows Xe2 to avoid undercounting EUs.
    ok &= (arch != gpu::intel::compute::gpu_arch_t::xe2);
#endif

    auto do_query = [&](cl_uint query) -> cl_uint {
        cl_uint val = 0;
        ok = ok
                && (clGetDeviceInfo(device, query, sizeof(val), &val, nullptr)
                        == CL_SUCCESS);
        return val;
    };

    cl_uint num_slices = do_query(CL_DEVICE_NUM_SLICES_INTEL);
    cl_uint num_sub_slices_per_slice
            = do_query(CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL);
    cl_uint num_eus_per_sub_slice
            = do_query(CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL);

    if (ok) {
        /* Some drivers report incorrect values on Xe2 */
        if (arch == gpu::intel::compute::gpu_arch_t::xe2)
            num_eus_per_sub_slice = 8;
        max_compute_units = std::min(max_compute_units,
                num_slices * num_sub_slices_per_slice * num_eus_per_sub_slice);
    }

    *eu_count = (int32_t)max_compute_units;

    return status::success;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
