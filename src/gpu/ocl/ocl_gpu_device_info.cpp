/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "gpu/ocl/ocl_gpu_device_info.hpp"
#include "gpu/ocl/ocl_gpu_detect.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ocl_gpu_device_info_t::init_arch(engine_t *engine) {
    cl_int err = CL_SUCCESS;
    auto device = utils::downcast<const ocl_gpu_engine_t *>(engine)->device();

    // skip other vendors
    const cl_uint intel_vendor_id = 0x8086;
    cl_uint vendor_id;
    err = clGetDeviceInfo(
            device, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, nullptr);
    OCL_CHECK(err);
    if (vendor_id != intel_vendor_id) return status::success;

    // try to detect gpu by device name first
    gpu_arch_ = detect_gpu_arch_by_device_name(name());
    if (gpu_arch_ != compute::gpu_arch_t::unknown) return status::success;

    // if failed, use slower method
    cl_context context
            = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    OCL_CHECK(err);

    gpu_arch_ = detect_gpu_arch(device, context);
    err = clReleaseContext(context);
    OCL_CHECK(err);

    return status::success;
}

status_t ocl_gpu_device_info_t::init_device_name(engine_t *engine) {
    cl_int err = CL_SUCCESS;
    auto device = utils::downcast<const ocl_gpu_engine_t *>(engine)->device();

    size_t param_size = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &param_size);
    OCL_CHECK(err);

    name_ = std::string(param_size, '\0');
    err = clGetDeviceInfo(
            device, CL_DEVICE_NAME, param_size, &name_[0], &param_size);
    OCL_CHECK(err);

    return status::success;
}

status_t ocl_gpu_device_info_t::init_runtime_version(engine_t *engine) {
    cl_int err = CL_SUCCESS;
    auto device = utils::downcast<const ocl_gpu_engine_t *>(engine)->device();

    size_t param_size = 0;
    err = clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, nullptr, &param_size);
    OCL_CHECK(err);

    std::string driver_version(param_size, '\0');
    err = clGetDeviceInfo(
            device, CL_DRIVER_VERSION, param_size, &driver_version[0], nullptr);
    OCL_CHECK(err);

    if (runtime_version_.set_from_string(&driver_version[0])
            != status::success) {
        runtime_version_.major = 0;
        runtime_version_.minor = 0;
        runtime_version_.build = 0;
    }

    return status::success;
}

status_t ocl_gpu_device_info_t::init_extensions(engine_t *engine) {
    cl_int err = CL_SUCCESS;
    auto device = utils::downcast<const ocl_gpu_engine_t *>(engine)->device();

    // query device for extensions
    size_t param_size = 0;
    err = clGetDeviceInfo(
            device, CL_DEVICE_EXTENSIONS, 0, nullptr, &param_size);
    OCL_CHECK(err);

    std::string extension_string(param_size, '\0');
    err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, param_size,
            &extension_string[0], &param_size);
    OCL_CHECK(err);

    // convert to ours
    using namespace compute;
    for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
            i_ext <<= 1) {
        const char *s_ext = ext2cl_str((device_ext_t)i_ext);
        if (s_ext && extension_string.find(s_ext) != std::string::npos) {
            extensions_ |= i_ext;
        }
    }

    // Handle future extensions, not yet supported by the OpenCL API
    extensions_ |= (uint64_t)get_future_extensions(gpu_arch());

    return status::success;
}

status_t ocl_gpu_device_info_t::init_attributes(engine_t *engine) {
    cl_int err = CL_SUCCESS;
    auto device = utils::downcast<const ocl_gpu_engine_t *>(engine)->device();

    cl_uint eu_count = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
            &eu_count, nullptr);
    OCL_CHECK(err);
    eu_count_ = (int32_t)eu_count;

    return status::success;
}

std::string ocl_gpu_device_info_t::get_cl_ext_options() const {
    using namespace compute;

    std::string opts;
    for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
            i_ext <<= 1) {
        auto ext = (device_ext_t)i_ext;

        // Use real GPU extensions
        if (!has(ext)) continue;

        // These extensions are not handled properly by the OpenCL runtime.
        // Pass macros for them manually.
        if (utils::one_of(ext, device_ext_t::intel_dot_accumulate))

            opts += std::string("-D") + ext2cl_str(ext) + " ";
    }
    if (!opts.empty()) { opts[opts.size() - 1] = '\0'; }
    return opts;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
