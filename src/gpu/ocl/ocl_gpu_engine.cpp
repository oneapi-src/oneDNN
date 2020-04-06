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

#include <CL/cl.h>

#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"
#include "gpu/ocl/ocl_kernel_list.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ocl_gpu_engine_t::init() {
    CHECK(compute_engine_t::init());

    cl_int err = CL_SUCCESS;
    if (is_user_context_) {
        err = clRetainContext(context_);
        if (err != CL_SUCCESS) context_ = nullptr;
    } else {
        context_
                = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    }

    OCL_CHECK(err);

    status_t status = check_device(engine_kind::gpu, device_, context_);
    if (status != status::success) return status;

    stream_t *service_stream_ptr;
    status = create_stream(
            &service_stream_ptr, stream_flags::default_flags, nullptr);
    if (status != status::success) return status;
    service_stream_.reset(service_stream_ptr);
    return status::success;
}

status_t ocl_gpu_engine_t::create_memory_storage(
        memory_storage_t **storage, unsigned flags, size_t size, void *handle) {
    auto _storage = new ocl_memory_storage_t(this);
    if (_storage == nullptr) return status::out_of_memory;
    status_t status = _storage->init(flags, size, handle);
    if (status != status::success) {
        delete _storage;
        return status;
    }
    *storage = _storage;
    return status::success;
}

status_t ocl_gpu_engine_t::create_stream(
        stream_t **stream, unsigned flags, const stream_attr_t *attr) {
    MAYBE_UNUSED(attr);
    return ocl_stream_t::create_stream(stream, this, flags);
}

status_t ocl_gpu_engine_t::create_stream(
        stream_t **stream, cl_command_queue queue) {
    return ocl_stream_t::create_stream(stream, this, queue);
}

cl_uint count_lines(const char *code[]) {
    cl_uint i = 0;
    const char *code_line = code[i];
    while (strcmp("END_OF_KERNEL", code_line)) {
        ++i;
        code_line = code[i];
    }
    return i;
}

status_t ocl_gpu_engine_t::create_kernels(
        std::vector<compute::kernel_t> *kernels,
        const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx) const {
    const std::string &options = kernel_ctx.options();

    std::vector<const char **> code_strings;
    code_strings.reserve(kernel_names.size());
    for (auto *kernel_name : kernel_names) {
        const char **code = get_ocl_kernel_source(kernel_name);
        code_strings.push_back(code);
    }

    *kernels = std::vector<compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); ++i) {
        if (!kernel_names[i] || (*kernels)[i]) continue;

        const char **code = code_strings[i];

        cl_uint count = count_lines(code);
        cl_int err;
        cl_program program = clCreateProgramWithSource(
                context(), count, code, nullptr, &err);
        OCL_CHECK(err);

        cl_device_id dev = device();
        err = clBuildProgram(
                program, 1, &dev, options.c_str(), nullptr, nullptr);
#ifndef NDEBUG
        if (err != CL_SUCCESS) {
            size_t log_length = 0;
            err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0,
                    nullptr, &log_length);
            assert(err == CL_SUCCESS);

            std::vector<char> log_buf(log_length);
            err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                    log_length, log_buf.data(), 0);
            assert(err == CL_SUCCESS);
            printf("Error during the build of OpenCL program.\nBuild "
                   "log:\n%s\n",
                    log_buf.data());
            OCL_CHECK(err);
        }
#endif
        for (size_t j = i; j < kernel_names.size(); ++j) {
            if (code_strings[j] == code_strings[i]) {
                cl_kernel ocl_kernel
                        = clCreateKernel(program, kernel_names[j], &err);
                OCL_CHECK(err);
                (*kernels)[j]
                        = compute::kernel_t(new ocl_gpu_kernel_t(ocl_kernel));
            }
        }

        OCL_CHECK(clReleaseProgram(program));
    }
    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
