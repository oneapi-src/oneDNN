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
#include <sstream>
#include <CL/cl.h>

#include "gpu/intel/ocl/ocl_gpu_engine.hpp"

#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "xpu/ocl/memory_storage.hpp"

#include "gpu/intel/compute/kernel_list.hpp"
#include "gpu/intel/microkernels/fuser.hpp"
#include "gpu/intel/ocl/kernel_utils.hpp"
#include "gpu/intel/ocl/ocl_gpu_device_info.hpp"
#include "gpu/intel/ocl/ocl_gpu_engine.hpp"
#include "gpu/intel/ocl/ocl_gpu_kernel.hpp"
#include "gpu/intel/ocl/ocl_stream.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        cl_device_id dev, cl_context ctx, size_t index,
        const std::vector<uint8_t> &cache_blob) {
    gpu_assert(engine_kind == engine_kind::gpu);
    std::unique_ptr<intel::ocl::ocl_gpu_engine_t, engine_deleter_t> e(
            (new intel::ocl::ocl_gpu_engine_t(dev, ctx, index)));
    if (!e) return status::out_of_memory;

    CHECK(e->init(cache_blob));
    *engine = e.release();

    return status::success;
}

void maybe_print_build_info(const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx) {
#ifndef DISABLE_VERBOSE
    // Print out kernel options if the correct verbosity is set
    if (get_verbose(verbose_t::debuginfo) >= 5) {
        std::ostringstream oss;
        for (const char *name : kernel_names)
            oss << name << " ";

        VFORMAT(get_msec(), primitive, exec, VERBOSE_debug,
                "kernel options,%s,%s", oss.str().c_str(),
                kernel_ctx.options().c_str());
    }
#endif
}

status_t ocl_gpu_engine_t::init() {
    return init({});
}

status_t ocl_gpu_engine_t::init(const std::vector<uint8_t> &cache_blob) {
    CHECK(init_impl());
    CHECK(compute::compute_engine_t::init(cache_blob));
    return status::success;
}

status_t ocl_gpu_engine_t::create_memory_storage(
        memory_storage_t **storage, unsigned flags, size_t size, void *handle) {
    std::unique_ptr<memory_storage_t> _storage;

    if (flags & memory_flags_t::prefer_device_usm) {
        _storage.reset(new xpu::ocl::usm_memory_storage_t(
                this, xpu::ocl::usm::kind_t::device));
    } else
        _storage.reset(new xpu::ocl::buffer_memory_storage_t(this));

    if (!_storage) return status::out_of_memory;

    status_t status = _storage->init(flags, size, handle);
    if (status != status::success) return status;

    *storage = _storage.release();
    return status::success;
}

status_t ocl_gpu_engine_t::create_stream(
        impl::stream_t **stream, impl::stream_impl_t *stream_impl) {
    return ocl_stream_t::create_stream(stream, this, stream_impl);
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

        auto program = xpu::ocl::make_wrapper(clCreateProgramWithBinary(
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
        auto ocl_kernel = xpu::ocl::make_wrapper(
                clCreateKernel(program, kernel_name.c_str(), &err));
        OCL_CHECK(err);

        std::vector<gpu::intel::compute::scalar_type_t> arg_types;
        CHECK(get_kernel_arg_types(ocl_kernel, &arg_types));
        OCL_CHECK(err);

        std::shared_ptr<compute::kernel_impl_t> kernel_impl
                = std::make_shared<ocl_gpu_kernel_t>(
                        std::move(ocl_kernel), arg_types);
        (*kernels)[i] = std::move(kernel_impl);
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

inline status_t preprocess_headers(std::stringstream &pp_code, const char *code,
        const compute::kernel_ctx_t &kernel_ctx) {
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
            const char *header_source
                    = kernel_ctx.get_custom_header(header_name);
            if (!header_source) header_source = get_kernel_header(header_name);
            CHECK(preprocess_headers(pp_code, header_source, kernel_ctx));
        } else {
            pp_code << line << std::endl;
        }
    }
    return status::success;
}

inline status_t fuse_microkernels(cl_context context, cl_device_id device,
        xpu::ocl::wrapper_t<cl_program> &program, const char *code) {
    if (micro::hasMicrokernels(code)) {
        cl_int status = CL_SUCCESS;
        size_t binary_size = 0;
        OCL_CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                sizeof(binary_size), &binary_size, nullptr));

        std::vector<uint8_t> binary(binary_size);
        auto binary_data = binary.data();
        OCL_CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                sizeof(binary_data), &binary_data, nullptr));

        try {
            micro::fuseMicrokernels(binary, code);
        } catch (...) { return status::runtime_error; }

        auto nbinary_size = binary.size();
        auto nbinary_data = const_cast<const uint8_t *>(binary.data());

        program = xpu::ocl::make_wrapper(clCreateProgramWithBinary(context, 1,
                &device, &nbinary_size, &nbinary_data, nullptr, &status));
        OCL_CHECK(status);
        OCL_CHECK(clBuildProgram(program, 1, &device, "", nullptr, nullptr));
    }

    return status::success;
}

} // namespace

status_t ocl_gpu_engine_t::build_program_from_source(
        xpu::ocl::wrapper_t<cl_program> &program, const char *code_string,
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
    CHECK(preprocess_headers(pp_code, code_string, kernel_ctx));
    std::string pp_code_str = pp_code.str();
    const char *pp_code_str_ptr = pp_code_str.c_str();

    debugdump_processed_source(
            pp_code_str, options, dev_info->get_cl_ext_options());

    auto ctx = context();
    program = xpu::ocl::make_wrapper(
            clCreateProgramWithSource(ctx, 1, &pp_code_str_ptr, nullptr, &err));
    OCL_CHECK(err);

    auto dev = device();
    err = clBuildProgram(program, 1, &dev, options.c_str(), nullptr, nullptr);
    OCL_CHECK(maybe_print_debug_info(err, program, dev));

    if (kernel_ctx.has_custom_headers())
        CHECK(fuse_microkernels(ctx, dev, program, pp_code_str_ptr));

    return status::success;
}

status_t ocl_gpu_engine_t::create_binary_from_ocl_source(xpu::binary_t &binary,
        const char *code_string,
        const compute::kernel_ctx_t &kernel_ctx) const {
    xpu::ocl::wrapper_t<cl_program> program;
    CHECK(build_program_from_source(program, code_string, kernel_ctx));

    CHECK(get_ocl_program_binary(program, device(), binary));
    return status::success;
}

status_t ocl_gpu_engine_t::create_kernel_from_binary(compute::kernel_t &kernel,
        const xpu::binary_t &binary, const char *kernel_name) const {
    xpu::ocl::wrapper_t<cl_program> program;
    CHECK(xpu::ocl::create_program(
            program, this->device(), this->context(), binary));

    cl_int err;
    auto ocl_kernel = xpu::ocl::make_wrapper(
            clCreateKernel(program, kernel_name, &err));
    OCL_CHECK(err);

    std::vector<gpu::intel::compute::scalar_type_t> arg_types;
    CHECK(get_kernel_arg_types(ocl_kernel, &arg_types));

    std::shared_ptr<compute::kernel_impl_t> kernel_impl
            = std::make_shared<ocl_gpu_kernel_t>(
                    std::move(ocl_kernel), arg_types);
    kernel = std::move(kernel_impl);

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

    xpu::binary_t binary = jitter->get_binary(context(), device());
    if (binary.empty()) return status::runtime_error;
    return create_kernel_from_binary(*kernel, binary, kernel_name);
}

status_t ocl_gpu_engine_t::create_kernels(
        std::vector<compute::kernel_t> *kernels,
        const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx,
        const cache_blob_t &cache_blob) const {
    maybe_print_build_info(kernel_names, kernel_ctx);

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
    xpu::ocl::wrapper_t<cl_program> program;
    CHECK(build_program_from_source(program, code_string, kernel_ctx));

    *kernels = std::vector<compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); ++i) {
        if (!kernel_names[i]) continue;
        cl_int err;
        xpu::ocl::wrapper_t<cl_kernel> ocl_kernel
                = clCreateKernel(program, kernel_names[i], &err);
        OCL_CHECK(err);
        std::vector<gpu::intel::compute::scalar_type_t> arg_types;
        CHECK(get_kernel_arg_types(ocl_kernel, &arg_types));

        std::shared_ptr<compute::kernel_impl_t> kernel_impl
                = std::make_shared<ocl_gpu_kernel_t>(
                        std::move(ocl_kernel), arg_types);
        (*kernels)[i] = std::move(kernel_impl);
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
    cl_int err = clGetPlatformInfo(impl()->platform(), CL_PLATFORM_NAME, 0,
            nullptr, &platform_name_len);
    OCL_CHECK(err);

    std::vector<char> platform_name(platform_name_len);
    err = clGetPlatformInfo(impl()->platform(), CL_PLATFORM_NAME,
            platform_name.size(), platform_name.data(), nullptr);
    OCL_CHECK(err);

    sstream.write(platform_name.data(), platform_name.size());
    sstream.write(device_info()->name().data(), device_info()->name().size());
    sstream.write(&device_info()->runtime_version().major);
    sstream.write(&device_info()->runtime_version().minor);
    sstream.write(&device_info()->runtime_version().build);

    return status::success;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
