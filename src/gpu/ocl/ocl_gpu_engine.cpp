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
#include <sstream>
#include <CL/cl.h>

#include "gpu/ocl/ocl_gpu_engine.hpp"

#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/compute/kernel_list.hpp"
#include "gpu/ocl/kernel_utils.hpp"
#include "gpu/ocl/ocl_gpu_device_info.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ocl_gpu_engine_t::init() {
    return init({});
}

status_t ocl_gpu_engine_t::init(const std::vector<uint8_t> &cache_blob) {
    cl_int err = CL_SUCCESS;
    err = clGetDeviceInfo(device_, CL_DEVICE_PLATFORM, sizeof(platform_),
            &platform_, nullptr);
    if (err != CL_SUCCESS) {
        device_ = nullptr;
        context_ = nullptr;
    }

    OCL_CHECK(err);

    err = clRetainDevice(device_);
    if (err != CL_SUCCESS) {
        device_ = nullptr;
        context_ = nullptr;
    }

    OCL_CHECK(err);

    if (is_user_context_) {
        err = clRetainContext(context_);
        if (err != CL_SUCCESS) context_ = nullptr;
    } else {
        context_
                = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    }

    OCL_CHECK(err);

    CHECK(check_device(engine_kind::gpu, device_, context_));
    compute::compute_engine_t::init(cache_blob);

    return status::success;
}

status_t ocl_gpu_engine_t::create_memory_storage(
        memory_storage_t **storage, unsigned flags, size_t size, void *handle) {
    auto _storage = new ocl_buffer_memory_storage_t(this);
    if (_storage == nullptr) return status::out_of_memory;
    status_t status = _storage->init(flags, size, handle);
    if (status != status::success) {
        delete _storage;
        return status;
    }
    *storage = _storage;
    return status::success;
}

status_t ocl_gpu_engine_t::create_stream(stream_t **stream, unsigned flags) {
    return ocl_stream_t::create_stream(stream, this, flags);
}

status_t ocl_gpu_engine_t::create_stream(
        stream_t **stream, cl_command_queue queue) {
    return ocl_stream_t::create_stream(stream, this, queue);
}

namespace {

status_t create_ocl_kernel_from_cache_blob(const ocl_gpu_engine_t *ocl_engine,
        cache_blob_t cache_blob, const std::vector<const char *> &kernel_names,
        std::vector<compute::kernel_t> *kernels) {
    auto dev = ocl_engine->device();
    auto ctx = ocl_engine->context();
    cl_int err = CL_SUCCESS;
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (!kernel_names[i]) continue;
        std::string kernel_name(kernel_names[i]);

        const uint8_t *binary = nullptr;
        size_t binary_size = 0;

        CHECK(cache_blob.get_binary(&binary, &binary_size));

        auto program = make_ocl_wrapper(clCreateProgramWithBinary(
                ctx, 1, &dev, &binary_size, &binary, nullptr, &err));
        OCL_CHECK(err);
        err = clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);
        OCL_CHECK(err);

        if (kernel_name.empty()) {
            // Handle the ngen cases when kernel name is not available.
            // Query the kernel name from the program. It's expected that
            // an ngen based program contains only 1 kernel.
            if (kernel_names.size() != 1 || kernels->size() != 1)
                return status::invalid_arguments;
            size_t kernel_name_size = 0;
            err = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, nullptr,
                    &kernel_name_size);
            OCL_CHECK(err);

            kernel_name.resize(kernel_name_size);
            err = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES,
                    kernel_name_size, &kernel_name[0], nullptr);
            OCL_CHECK(err);
            assert(!kernel_name.empty());
            if (kernel_name.empty()) return status::runtime_error;
            // Remove the null terminator as std::string already includes it.
            kernel_name.pop_back();
        }
        auto ocl_kernel = make_ocl_wrapper(
                clCreateKernel(program, kernel_name.c_str(), &err));
        OCL_CHECK(err);

        std::vector<gpu::compute::scalar_type_t> arg_types;
        CHECK(get_kernel_arg_types(ocl_kernel, &arg_types));
        OCL_CHECK(err);

        (*kernels)[i] = compute::kernel_t(
                new ocl_gpu_kernel_t(ocl_kernel, arg_types));
        dump_kernel_binary(ocl_engine, (*kernels)[i]);
    }

    return status::success;
}

cl_int maybe_print_debug_info(
        cl_int err, cl_program program, cl_device_id dev) {
    // Return error code if verbose is not enabled.
    if (err == CL_SUCCESS || get_verbose() == 0) return err;

    size_t log_length = 0;
    err = clGetProgramBuildInfo(
            program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_length);
    assert(err == CL_SUCCESS);

    std::vector<char> log_buf(log_length);
    err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_length,
            log_buf.data(), nullptr);
    assert(err == CL_SUCCESS);
    printf("Error during the build of OpenCL program.\nBuild "
           "log:\n%s\n",
            log_buf.data());
    return err;
};

inline status_t preprocess_headers(
        std::stringstream &pp_code, const char *code) {
    std::stringstream code_stream(code);

    for (std::string line; std::getline(code_stream, line);) {
        const size_t include_pos = line.find("#include");
        if (include_pos != std::string::npos) {
            static constexpr size_t include_len = 8;
            const size_t first_quote_pos
                    = line.find("\"", include_pos + include_len);
            const size_t second_quote_pos
                    = line.find("\"", first_quote_pos + 1);
            const size_t kernel_name_len
                    = second_quote_pos - first_quote_pos - 1;
            const auto header_name
                    = line.substr(first_quote_pos + 1, kernel_name_len);
            CHECK(preprocess_headers(pp_code, get_kernel_header(header_name)));
        } else {
            pp_code << line << std::endl;
        }
    }
    return status::success;
}

} // namespace

status_t ocl_gpu_engine_t::create_kernel(compute::kernel_t *kernel,
        jit::jit_generator_base *jitter, cache_blob_t cache_blob) const {
    if (!jitter && !cache_blob) return status::invalid_arguments;

    const char *kernel_name = jitter ? jitter->kernel_name() : "";

    if (cache_blob) {
        std::vector<compute::kernel_t> kernels(1);
        auto status = create_ocl_kernel_from_cache_blob(
                this, cache_blob, {kernel_name}, &kernels);
        CHECK(status);
        (*kernel) = kernels[0];
        return status::success;
    }

    ocl_wrapper_t<cl_kernel> ocl_kernel
            = jitter->get_kernel(context(), device());
    std::vector<gpu::compute::scalar_type_t> arg_types;
    CHECK(get_kernel_arg_types(ocl_kernel, &arg_types));

    *kernel = compute::kernel_t(new ocl_gpu_kernel_t(ocl_kernel, arg_types));
    dump_kernel_binary(this, *kernel);

    return status::success;
}

status_t ocl_gpu_engine_t::create_kernels(
        std::vector<compute::kernel_t> *kernels,
        const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx,
        cache_blob_t cache_blob) const {

    *kernels = std::vector<compute::kernel_t>(kernel_names.size());

    if (cache_blob) {
        return create_ocl_kernel_from_cache_blob(
                this, cache_blob, kernel_names, kernels);
    }

    compute::kernel_list_t kernel_list;
    for (size_t i = 0; i < kernels->size(); ++i) {
        if (kernel_names[i]) kernel_list.add(kernel_names[i], &(*kernels)[i]);
    }

    return ocl::create_kernels(this, kernel_list, kernel_ctx);
}

status_t ocl_gpu_engine_t::create_kernels_from_ocl_source(
        std::vector<compute::kernel_t> *kernels,
        const std::vector<const char *> &kernel_names, const char *code_string,
        const compute::kernel_ctx_t &kernel_ctx) const {
    std::string options = kernel_ctx.options();

    // XXX: Update options by adding macros for OpenCL extensions that are not
    // handled properly by the OpenCL runtime
    auto *dev_info
            = utils::downcast<const ocl_gpu_device_info_t *>(device_info());
    options += " " + dev_info->get_cl_ext_options();

    cl_int err;
    std::stringstream pp_code;
    // The `cl_cache` requires using `clBuildProgram`. Unfortunately, unlike
    // `clCompileProgram` `clBuildProgram` doesn't take headers. Because of
    // that, a manual preprocessing of `include` header directives in the
    // OpenCL kernels is required.
    CHECK(preprocess_headers(pp_code, code_string));
    std::string pp_code_str = pp_code.str();
    const char *pp_code_str_ptr = pp_code_str.c_str();

    auto program = make_ocl_wrapper(clCreateProgramWithSource(
            context(), 1, &pp_code_str_ptr, nullptr, &err));
    OCL_CHECK(err);

    cl_device_id dev = device();
    err = clBuildProgram(program, 1, &dev, options.c_str(), nullptr, nullptr);
    OCL_CHECK(maybe_print_debug_info(err, program, dev));

    *kernels = std::vector<compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); ++i) {
        cl_int err;
        ocl_wrapper_t<cl_kernel> ocl_kernel
                = clCreateKernel(program, kernel_names[i], &err);
        OCL_CHECK(err);
        std::vector<gpu::compute::scalar_type_t> arg_types;
        CHECK(get_kernel_arg_types(ocl_kernel, &arg_types));

        (*kernels)[i] = compute::kernel_t(
                new ocl_gpu_kernel_t(ocl_kernel, arg_types));
        dump_kernel_binary(this, (*kernels)[i]);
    }

    return status::success;
}

status_t ocl_gpu_engine_t::init_device_info() {
    return init_device_info({});
}

status_t ocl_gpu_engine_t::init_device_info(
        const std::vector<uint8_t> &cache_blob) {
    device_info_ = std::make_shared<ocl_gpu_device_info_t>();
    CHECK(device_info_->init(this, cache_blob));
    return status::success;
}

status_t ocl_gpu_engine_t::serialize_device(
        serialization_stream_t &sstream) const {
    size_t platform_name_len;
    cl_int err = clGetPlatformInfo(
            platform_, CL_PLATFORM_NAME, 0, nullptr, &platform_name_len);
    OCL_CHECK(err);

    std::vector<char> platform_name(platform_name_len);
    err = clGetPlatformInfo(platform_, CL_PLATFORM_NAME, platform_name.size(),
            platform_name.data(), nullptr);
    OCL_CHECK(err);

    sstream.write(platform_name.data(), platform_name.size());
    sstream.write(device_info()->name().data(), device_info()->name().size());
    sstream.write(&device_info()->runtime_version().major);
    sstream.write(&device_info()->runtime_version().minor);
    sstream.write(&device_info()->runtime_version().build);

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
