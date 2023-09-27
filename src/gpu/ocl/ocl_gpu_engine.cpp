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
    std::unique_ptr<memory_storage_t> _storage;

    if (flags & memory_flags_t::prefer_device_usm) {
        _storage.reset(new ocl_usm_memory_storage_t(
                this, usm::ocl_usm_kind_t::device));
    } else
        _storage.reset(new ocl_buffer_memory_storage_t(this));

    if (!_storage) return status::out_of_memory;

    status_t status = _storage->init(flags, size, handle);
    if (status != status::success) return status;

    *storage = _storage.release();
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
        const cache_blob_t &cache_blob,
        const std::vector<const char *> &kernel_names,
        std::vector<compute::kernel_t> *kernels) {
    auto dev = ocl_engine->device();
    auto ctx = ocl_engine->context();
    cl_int err = CL_SUCCESS;
    *kernels = std::vector<compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (!kernel_names[i] && kernel_names.size() > 1) continue;
        std::string kernel_name(kernel_names[i] ? kernel_names[i] : "");

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

        std::shared_ptr<compute::kernel_impl_t> kernel_impl
                = std::make_shared<ocl_gpu_kernel_t>(ocl_kernel, arg_types);
        (*kernels)[i] = std::move(kernel_impl);
        dump_kernel_binary(ocl_engine, (*kernels)[i]);
    }

    return status::success;
}

cl_int maybe_print_debug_info(
        cl_int err_, cl_program program, cl_device_id dev) {
    // Return error code if verbose is not enabled.
    if (err_ == CL_SUCCESS || !get_verbose(verbose_t::error)) return err_;

    size_t log_length = 0;
    auto err = clGetProgramBuildInfo(
            program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_length);
    assert(err == CL_SUCCESS);

    std::vector<char> log_buf(log_length);
    err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_length,
            log_buf.data(), nullptr);
    assert(err == CL_SUCCESS);
    VERROR(common, ocl,
            "Error during the build of OpenCL program. Build log:\n%s",
            log_buf.data());
    MAYBE_UNUSED(err);
    return err_;
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

status_t ocl_gpu_engine_t::build_program_from_source(
        ocl_wrapper_t<cl_program> &program, const char *code_string,
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

    debugdump_processed_source(
            pp_code_str, options, dev_info->get_cl_ext_options());

    program = make_ocl_wrapper(clCreateProgramWithSource(
            context(), 1, &pp_code_str_ptr, nullptr, &err));
    OCL_CHECK(err);

    auto dev = device();
    err = clBuildProgram(program, 1, &dev, options.c_str(), nullptr, nullptr);
    OCL_CHECK(maybe_print_debug_info(err, program, dev));
    return status::success;
}

status_t ocl_gpu_engine_t::create_binary_from_ocl_source(
        compute::binary_t &binary, const char *code_string,
        const compute::kernel_ctx_t &kernel_ctx) const {
    ocl_wrapper_t<cl_program> program;
    CHECK(build_program_from_source(program, code_string, kernel_ctx));

    CHECK(get_ocl_program_binary(program, device(), binary));
    return status::success;
}

status_t ocl_gpu_engine_t::create_compiled_bundle(
        compute::compiled_bundle_t &generator,
        const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx) const {

    const char *source = ocl::get_kernel_source(kernel_names[0]);
    for (const auto &kernel_name : kernel_names) {
        assert(ocl::get_kernel_source(kernel_name) == source);
        MAYBE_UNUSED(kernel_name);
    }

    compute::binary_t kernel_binary {};
    CHECK(create_binary_from_ocl_source(kernel_binary, source, kernel_ctx));
    generator = compute::compiled_bundle_t(kernel_binary);
    return status::success;
};

status_t ocl_gpu_engine_t::create_compiled_kernel(
        compute::compiled_kernel_t &generator,
        jit::jit_generator_base &jitter) const {
    auto &ocl_engine = *utils::downcast<const ocl_gpu_engine_t *>(this);
    generator = compute::compiled_kernel_t(
            jitter.get_binary(ocl_engine.context(), ocl_engine.device()),
            jitter.kernel_name());
    return status::success;
}

status_t ocl_gpu_engine_t::create_kernels_from_bundle(
        std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names,
        const compute::compiled_bundle_t &generator) const {

    auto dev = this->device();
    auto ctx = this->context();
    cl_int err = CL_SUCCESS;

    auto &binary = generator.binary();
    const uint8_t *binary_data = binary.data();
    size_t binary_size = binary.size();
    auto program = make_ocl_wrapper(clCreateProgramWithBinary(
            ctx, 1, &dev, &binary_size, &binary_data, nullptr, &err));
    OCL_CHECK(err);

    err = clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);
    OCL_CHECK(err);

    kernels = std::vector<compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); ++i) {
        ocl_wrapper_t<cl_kernel> ocl_kernel
                = clCreateKernel(program, kernel_names[i], &err);
        OCL_CHECK(err);
        std::vector<gpu::compute::scalar_type_t> arg_types;
        CHECK(get_kernel_arg_types(ocl_kernel, &arg_types));

        std::shared_ptr<compute::kernel_impl_t> kernel_impl
                = std::make_shared<ocl_gpu_kernel_t>(ocl_kernel, arg_types);
        kernels[i] = std::move(kernel_impl);
        dump_kernel_binary(this, kernels[i]);
    }

    return status::success;
}

status_t ocl_gpu_engine_t::create_kernel_from_binary(compute::kernel_t &kernel,
        const compute::binary_t &binary, const char *kernel_name) const {
    ocl_wrapper_t<cl_program> program;
    CHECK(ocl::create_ocl_program(
            program, this->device(), this->context(), binary));

    cl_int err;
    auto ocl_kernel
            = make_ocl_wrapper(clCreateKernel(program, kernel_name, &err));
    OCL_CHECK(err);

    std::vector<gpu::compute::scalar_type_t> arg_types;
    CHECK(get_kernel_arg_types(ocl_kernel, &arg_types));

    std::shared_ptr<compute::kernel_impl_t> kernel_impl
            = std::make_shared<ocl_gpu_kernel_t>(ocl_kernel, arg_types);
    kernel = std::move(kernel_impl);
    dump_kernel_binary(this, kernel);

    return status::success;
}

status_t ocl_gpu_engine_t::create_kernels_from_cache_blob(
        const cache_blob_t &cache_blob, std::vector<compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names) const {
    return create_ocl_kernel_from_cache_blob(
            this, cache_blob, kernel_names, &kernels);
}

status_t ocl_gpu_engine_t::create_kernel(compute::kernel_t *kernel,
        jit::jit_generator_base *jitter, const cache_blob_t &cache_blob) const {
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

    compute::binary_t binary = jitter->get_binary(context(), device());
    return create_kernel_from_binary(*kernel, binary, kernel_name);
}

status_t ocl_gpu_engine_t::create_kernels(
        std::vector<compute::kernel_t> *kernels,
        const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx,
        const cache_blob_t &cache_blob) const {

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
    ocl_wrapper_t<cl_program> program;
    CHECK(build_program_from_source(program, code_string, kernel_ctx));

    *kernels = std::vector<compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); ++i) {
        cl_int err;
        ocl_wrapper_t<cl_kernel> ocl_kernel
                = clCreateKernel(program, kernel_names[i], &err);
        OCL_CHECK(err);
        std::vector<gpu::compute::scalar_type_t> arg_types;
        CHECK(get_kernel_arg_types(ocl_kernel, &arg_types));

        std::shared_ptr<compute::kernel_impl_t> kernel_impl
                = std::make_shared<ocl_gpu_kernel_t>(ocl_kernel, arg_types);
        (*kernels)[i] = std::move(kernel_impl);
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
