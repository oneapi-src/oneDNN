/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <CL/cl_ext.h>

#include "ocl/cl_engine.hpp"

#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

namespace ocl_utils {

status_t get_ocl_devices(
        std::vector<cl_device_id> *devices, cl_device_type device_type) {
    cl_uint num_platforms = 0;

    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    // No platforms - a valid scenario
    if (err == CL_PLATFORM_NOT_FOUND_KHR)
        return status::success;

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
            devices->swap(plat_devices);
            return status::success;
        }
    }
    // No devices found but still return success
    return status::success;
}

} // namespace ocl_utils

status_t ocl_jit_t::build(const engine_t *engine) {
    auto *cl_engine = utils::downcast<const cl_engine_t *>(engine);
    cl_context ctx = cl_engine->ocl_context();
    cl_device_id dev = cl_engine->ocl_device();

    cl_int err = CL_SUCCESS;

    program_ = clCreateProgramWithSource(ctx, 1, &code_, &code_size_, &err);
    status_t status = ocl_utils::convert_to_mkldnn(err);
    if (status != status::success)
        return status;

    const char *opt_str = options_.data();
    err = clBuildProgram(program_, 1, &dev, opt_str, nullptr, nullptr);
#ifndef NDEBUG
    if (err != CL_SUCCESS) {
        size_t log_length = 0;
        err = clGetProgramBuildInfo(
                program_, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_length);
        assert(err == CL_SUCCESS);

        std::vector<char> log_buf(log_length);
        err = clGetProgramBuildInfo(program_, dev, CL_PROGRAM_BUILD_LOG,
                log_length, log_buf.data(), 0);
        assert(err == CL_SUCCESS);
        printf("Error during the build of OpenCL program.\nBuild log:\n%s\n",
                log_buf.data());
    }
#endif
    return ocl_utils::convert_to_mkldnn(err);
}

} // namespace ocl
} // namespace impl
} // namespace mkldnn
