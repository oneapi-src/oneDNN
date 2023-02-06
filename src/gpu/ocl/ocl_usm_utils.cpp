/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include <type_traits>

#include <CL/cl.h>

#include "common/cpp_compat.hpp"

#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_usm_utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
namespace usm {

namespace {

cl_device_id get_ocl_device(engine_t *engine) {
    return utils::downcast<ocl_gpu_engine_t *>(engine)->device();
}

cl_context get_ocl_context(engine_t *engine) {
    return utils::downcast<ocl_gpu_engine_t *>(engine)->context();
}

cl_command_queue get_ocl_queue(stream_t *stream) {
    return utils::downcast<ocl_stream_t *>(stream)->queue();
}

} // namespace

bool is_usm_supported(engine_t *engine) {
    using clSharedMemAllocINTEL_func_t = void *(*)(cl_context, cl_device_id,
            cl_ulong *, size_t, cl_uint, cl_int *);
    static ext_func_t<clSharedMemAllocINTEL_func_t> ext_func(
            "clSharedMemAllocINTEL");
    return (bool)ext_func.get_func(engine);
}

void *malloc_host(engine_t *engine, size_t size) {
    using clHostMemAllocINTEL_func_t = void *(*)(cl_context, const cl_ulong *,
            size_t, cl_uint, cl_int *);

    if (size == 0) return nullptr;

    static ext_func_t<clHostMemAllocINTEL_func_t> ext_func(
            "clHostMemAllocINTEL");
    cl_int err;
    void *p = ext_func(engine, get_ocl_context(engine), nullptr, size, 0, &err);
    assert(utils::one_of(err, CL_SUCCESS, CL_OUT_OF_RESOURCES,
            CL_OUT_OF_HOST_MEMORY, CL_INVALID_BUFFER_SIZE));
    return p;
}

void *malloc_device(engine_t *engine, size_t size) {
    using clDeviceMemAllocINTEL_func_t = void *(*)(cl_context, cl_device_id,
            cl_ulong *, size_t, cl_uint, cl_int *);

    if (size == 0) return nullptr;

    static ext_func_t<clDeviceMemAllocINTEL_func_t> ext_func(
            "clDeviceMemAllocINTEL");
    cl_int err;
    void *p = ext_func(engine, get_ocl_context(engine), get_ocl_device(engine),
            nullptr, size, 0, &err);
    assert(utils::one_of(err, CL_SUCCESS, CL_OUT_OF_RESOURCES,
            CL_OUT_OF_HOST_MEMORY, CL_INVALID_BUFFER_SIZE));
    return p;
}

void *malloc_shared(engine_t *engine, size_t size) {
    using clSharedMemAllocINTEL_func_t = void *(*)(cl_context, cl_device_id,
            cl_ulong *, size_t, cl_uint, cl_int *);

    if (size == 0) return nullptr;

    static ext_func_t<clSharedMemAllocINTEL_func_t> ext_func(
            "clSharedMemAllocINTEL");
    cl_int err;
    void *p = ext_func(engine, get_ocl_context(engine), get_ocl_device(engine),
            nullptr, size, 0, &err);
    assert(utils::one_of(err, CL_SUCCESS, CL_OUT_OF_RESOURCES,
            CL_OUT_OF_HOST_MEMORY, CL_INVALID_BUFFER_SIZE));
    return p;
}

void free(engine_t *engine, void *ptr) {
    using clMemFreeINTEL_func_t = cl_int (*)(cl_context, void *);

    if (!ptr) return;
    static ext_func_t<clMemFreeINTEL_func_t> ext_func("clMemFreeINTEL");
    cl_int err = ext_func(engine, get_ocl_context(engine), ptr);
    assert(err == CL_SUCCESS);
    MAYBE_UNUSED(err);
}

status_t set_kernel_arg_usm(engine_t *engine, cl_kernel kernel, int arg_index,
        const void *arg_value) {
    using clSetKernelArgMemPointerINTEL_func_t
            = cl_int (*)(cl_kernel, cl_uint, const void *);
    static ext_func_t<clSetKernelArgMemPointerINTEL_func_t> ext_func(
            "clSetKernelArgMemPointerINTEL");
    return convert_to_dnnl(ext_func(engine, kernel, arg_index, arg_value));
}

status_t memcpy(stream_t *stream, void *dst, const void *src, size_t size,
        cl_uint num_events, const cl_event *events, cl_event *out_event) {
    using clEnqueueMemcpyINTEL_func_t
            = cl_int (*)(cl_command_queue, cl_bool, void *, const void *,
                    size_t, cl_uint, const cl_event *, cl_event *);
    static ext_func_t<clEnqueueMemcpyINTEL_func_t> ext_func(
            "clEnqueueMemcpyINTEL");
    return convert_to_dnnl(ext_func(stream->engine(), get_ocl_queue(stream),
            /* blocking */ CL_FALSE, dst, src, size, num_events, events,
            out_event));
}

status_t memcpy(stream_t *stream, void *dst, const void *src, size_t size) {
    return memcpy(stream, dst, src, size, 0, nullptr, nullptr);
}

status_t fill(stream_t *stream, void *ptr, const void *pattern,
        size_t pattern_size, size_t size, cl_uint num_events,
        const cl_event *events, cl_event *out_event) {
    using clEnqueueMemFillINTEL_func_t
            = cl_int (*)(cl_command_queue, void *, const void *, size_t, size_t,
                    cl_uint, const cl_event *, cl_event *);
    static ext_func_t<clEnqueueMemFillINTEL_func_t> ext_func(
            "clEnqueueMemFillINTEL");
    return convert_to_dnnl(ext_func(stream->engine(), get_ocl_queue(stream),
            ptr, pattern, pattern_size, size, num_events, events, out_event));
}

status_t memset(stream_t *stream, void *ptr, int value, size_t size) {
    uint8_t pattern = (uint8_t)value;
    return fill(
            stream, ptr, &pattern, sizeof(uint8_t), size, 0, nullptr, nullptr);
}

ocl_usm_kind_t get_pointer_type(engine_t *engine, const void *ptr) {
    using clGetMemAllocInfoINTEL_func_t = cl_int (*)(
            cl_context, const void *, cl_uint, size_t, void *, size_t *);

    // The values are taken from cl_ext.h to avoid dependency on the header.
    static constexpr cl_uint cl_mem_type_unknown_intel = 0x4196;
    static constexpr cl_uint cl_mem_type_host_intel = 0x4197;
    static constexpr cl_uint cl_mem_type_device_intel = 0x4198;
    static constexpr cl_uint cl_mem_type_shared_intel = 0x4199;

    static constexpr cl_uint cl_mem_alloc_type_intel = 0x419A;

    static ext_func_t<clGetMemAllocInfoINTEL_func_t> ext_func(
            "clGetMemAllocInfoINTEL");

    if (!ptr) return ocl_usm_kind_t::unknown;

    cl_uint alloc_type;
    cl_int err = ext_func(engine, get_ocl_context(engine), ptr,
            cl_mem_alloc_type_intel, sizeof(alloc_type), &alloc_type, nullptr);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) return ocl_usm_kind_t::unknown;

    switch (alloc_type) {
        case cl_mem_type_unknown_intel: return ocl_usm_kind_t::unknown;
        case cl_mem_type_host_intel: return ocl_usm_kind_t::host;
        case cl_mem_type_device_intel: return ocl_usm_kind_t::device;
        case cl_mem_type_shared_intel: return ocl_usm_kind_t::shared;
        default: assert(!"unknown alloc type");
    }
    return ocl_usm_kind_t::unknown;
}

} // namespace usm
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
