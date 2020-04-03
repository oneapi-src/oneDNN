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

#include <CL/sycl.hpp>

#include "common/utils.hpp"
#include "sycl/level_zero_utils.hpp"
#include "sycl/sycl_ocl_gpu_kernel.hpp"
#include "sycl/sycl_stream.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

static void set_scalar_arg(
        cl::sycl::handler &cgh, int index, size_t size, const void *value) {
    switch (size) {
        case sizeof(uint8_t):
            cgh.set_arg(index, *static_cast<const uint8_t *>(value));
            break;
        case sizeof(uint16_t):
            cgh.set_arg(index, *static_cast<const uint16_t *>(value));
            break;
        case sizeof(uint32_t):
            cgh.set_arg(index, *static_cast<const uint32_t *>(value));
            break;
        case sizeof(uint64_t):
            cgh.set_arg(index, *static_cast<const uint64_t *>(value));
            break;
        default:
            assert(!"Please add another case");
            throw std::runtime_error("Internal error");
    }
}

sycl_ocl_gpu_kernel_t::~sycl_ocl_gpu_kernel_t() {
    if (ocl_kernel_) OCL_CHECK_V(clReleaseKernel(ocl_kernel_));
}

status_t sycl_create_kernel(std::unique_ptr<cl::sycl::kernel> &sycl_kernel,
        const sycl_gpu_engine_t *sycl_engine, cl_kernel ocl_kernel,
        void **handle_to_destroy) {
    cl_program ocl_program;
    OCL_CHECK(clGetKernelInfo(ocl_kernel, CL_KERNEL_PROGRAM,
            sizeof(ocl_program), &ocl_program, nullptr));

    std::string kernel_name(128, '\0');
    OCL_CHECK(clGetKernelInfo(ocl_kernel, CL_KERNEL_FUNCTION_NAME,
            kernel_name.size(), &kernel_name[0], nullptr));

    if (sycl_engine->backend() == backend_t::opencl) {
        cl::sycl::program sycl_program(sycl_engine->context(), ocl_program);
        sycl_kernel.reset(
                new cl::sycl::kernel(sycl_program.get_kernel(kernel_name)));
        return status::success;
    }

#if defined(DNNL_SYCL_DPCPP) && defined(DNNL_WITH_LEVEL_ZERO)
    if (sycl_engine->backend() != backend_t::level0)
        return status::invalid_arguments;

    size_t binary_size = 0;
    OCL_CHECK(clGetProgramInfo(ocl_program, CL_PROGRAM_BINARY_SIZES,
            sizeof(size_t), &binary_size, nullptr));

    std::vector<unsigned char> binary(binary_size);
    auto *binary_ptr = binary.data();
    OCL_CHECK(clGetProgramInfo(ocl_program, CL_PROGRAM_BINARIES, binary_size,
            &binary_ptr, nullptr));

    ze_module_desc_t desc {ZE_MODULE_DESC_VERSION_CURRENT};
    desc.format = ZE_MODULE_FORMAT_NATIVE;
    desc.inputSize = binary_size;
    desc.pInputModule = binary_ptr;
    desc.pBuildFlags = "";
    desc.pConstants = nullptr;

    auto ze_device = (ze_device_handle_t)sycl_engine->device().get();

    ze_module_handle_t ze_module;
    CHECK(func_zeModuleCreate(ze_device, &desc, &ze_module, nullptr));
    *handle_to_destroy = ze_module;

    cl::sycl::program sycl_program(
            sycl_engine->context(), reinterpret_cast<cl_program>(ze_module));
    sycl_kernel.reset(
            new cl::sycl::kernel(sycl_program.get_kernel(kernel_name)));

    return status::success;
#else
    return status::invalid_arguments;
#endif
}

status_t sycl_ocl_gpu_kernel_t::parallel_for(stream_t &stream,
        const gpu::compute::nd_range_t &range,
        const gpu::compute::kernel_arg_list_t &arg_list) const {
    if (range.is_zero()) return status::success;

    auto *sycl_stream = utils::downcast<sycl::sycl_stream_t *>(&stream);
    auto *sycl_engine
            = utils::downcast<sycl::sycl_gpu_engine_t *>(sycl_stream->engine());
    auto &queue = sycl_stream->queue();

    std::unique_ptr<cl::sycl::kernel> sycl_kernel;
    void *handle_to_destroy = nullptr;
    CHECK(sycl_create_kernel(
            sycl_kernel, sycl_engine, ocl_kernel_, &handle_to_destroy));

    auto event = queue.submit([&](cl::sycl::handler &cgh) {
#ifdef DNNL_SYCL_DPCPP
        cgh.depends_on(sycl_stream->get_deps());
#endif
        for (int i = 0; i < arg_list.nargs(); ++i) {
            auto &arg = arg_list.get(i);
            if (arg.is_global()) {
                auto *mem_storage
                        = static_cast<const memory_storage_t *>(arg.value());
                if (*mem_storage) {
                    auto *sycl_mem_storage = utils::downcast<
                            const sycl_memory_storage_base_t *>(mem_storage);
                    switch (sycl_mem_storage->memory_api_kind()) {
                        case memory_api_kind_t::buffer: {
                            auto *m = utils::downcast<
                                    const sycl_buffer_memory_storage_t *>(
                                    mem_storage);
                            auto &sycl_buf = m->buffer();
                            cgh.set_arg((int)i,
                                    sycl_buf.get_access<
                                            cl::sycl::access::mode::read_write>(
                                            cgh));
                            break;
                        }
#ifdef DNNL_SYCL_DPCPP
                        case memory_api_kind_t::usm: {
                            auto *m = utils::downcast<
                                    const sycl_usm_memory_storage_t *>(
                                    mem_storage);
                            cgh.set_arg((int)i, m->usm_ptr());
                            break;
                        }
#endif
                        default: assert(!"not expected");
                    }
                } else {
                    cgh.set_arg((int)i, nullptr);
                }
            } else if (arg.is_local()) {
                auto acc = cl::sycl::accessor<uint8_t, 1,
                        cl::sycl::access::mode::read_write,
                        cl::sycl::access::target::local>(
                        cl::sycl::range<1>(arg.size()), cgh);
                cgh.set_arg((int)i, acc);
            } else {
                gpu::compute::scalar_type_t real_arg_type;
                gpu::ocl::get_ocl_kernel_arg_type(
                        &real_arg_type, ocl_kernel_, i);
                auto cvt_arg
                        = gpu::compute::kernel_arg_t::cast(real_arg_type, arg);
                set_scalar_arg(cgh, (int)i, cvt_arg.size(), cvt_arg.value());
            }
        }
        if (range.local_range()) {
            auto sycl_nd_range = to_sycl_nd_range(range);
            cgh.parallel_for(sycl_nd_range, *sycl_kernel);
        } else {
            auto sycl_range = to_sycl_range(range);
            cgh.parallel_for(sycl_range, *sycl_kernel);
        }
    });

#if defined(DNNL_SYCL_DPCPP) && defined(DNNL_WITH_LEVEL_ZERO)
    if (sycl_engine->backend() == backend_t::level0) {
        // L0 module should be destroyed manually.
        sycl_kernel.reset();
        auto ze_module
                = reinterpret_cast<ze_module_handle_t>(handle_to_destroy);
        CHECK(func_zeModuleDestroy(ze_module));
    }
#endif

    sycl_stream->set_deps({event});
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
