/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include "xpu/ocl/engine_impl.hpp"
#include "xpu/ocl/stream_impl.hpp"

#include "gpu/intel/ocl/usm_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {
namespace usm {

namespace {

cl_device_id get_ocl_device(impl::engine_t *engine) {
    return utils::downcast<const xpu::ocl::engine_impl_t *>(engine->impl())
            ->device();
}

cl_context get_ocl_context(impl::engine_t *engine) {
    return utils::downcast<const xpu::ocl::engine_impl_t *>(engine->impl())
            ->context();
}

cl_command_queue get_ocl_queue(impl::stream_t *stream) {
    return utils::downcast<xpu::ocl::stream_impl_t *>(stream->impl())->queue();
}

} // namespace

bool is_usm_supported(impl::engine_t *engine) {
    using clSharedMemAllocINTEL_func_t = void *(*)(cl_context, cl_device_id,
            cl_ulong *, size_t, cl_uint, cl_int *);
    static xpu::ocl::ext_func_t<clSharedMemAllocINTEL_func_t> ext_func(
            "clSharedMemAllocINTEL");
    return (bool)ext_func.get_func(engine);
}

static xpu::memory_registry_t usm_shared_mem;
static xpu::memory_registry_t usm_device_mem;
static xpu::memory_registry_t usm_host_mem;

void *malloc_host(impl::engine_t *engine, size_t size) {
    using clHostMemAllocINTEL_func_t = void *(*)(cl_context, const cl_ulong *,
            size_t, cl_uint, cl_int *);

    if (size == 0) return nullptr;

    printf("USM malloc host with size %lu, total allocation size: %f GiB\n",
            size, 1.0 * (usm_host_mem.size() + size) / (1024 * 1024 * 1024));

    static xpu::ocl::ext_func_t<clHostMemAllocINTEL_func_t> ext_func(
            "clHostMemAllocINTEL");
    cl_int err;
    void *p = ext_func(engine, get_ocl_context(engine), nullptr, size, 0, &err);
    assert(utils::one_of(err, CL_SUCCESS, CL_OUT_OF_RESOURCES,
            CL_OUT_OF_HOST_MEMORY, CL_INVALID_BUFFER_SIZE));
    printf("USM malloc host ptr=%p\n", p);
    return p;
}

void *malloc_device(impl::engine_t *engine, size_t size) {
    using clDeviceMemAllocINTEL_func_t = void *(*)(cl_context, cl_device_id,
            cl_ulong *, size_t, cl_uint, cl_int *);

    if (size == 0) return nullptr;

    printf("USM malloc device with size %lu, total allocation size: %f GiB\n",
            size, 1.0 * (usm_device_mem.size() + size) / (1024 * 1024 * 1024));

    static xpu::ocl::ext_func_t<clDeviceMemAllocINTEL_func_t> ext_func(
            "clDeviceMemAllocINTEL");
    cl_int err;
    void *p = ext_func(engine, get_ocl_context(engine), get_ocl_device(engine),
            nullptr, size, 0, &err);
    assert(utils::one_of(err, CL_SUCCESS, CL_OUT_OF_RESOURCES,
            CL_OUT_OF_HOST_MEMORY, CL_INVALID_BUFFER_SIZE));
    usm_device_mem.add(p, size);
    printf("USM malloc device ptr=%p\n", p);
    return p;
}

void *malloc_shared(impl::engine_t *engine, size_t size) {
    using clSharedMemAllocINTEL_func_t = void *(*)(cl_context, cl_device_id,
            cl_ulong *, size_t, cl_uint, cl_int *);

    if (size == 0) return nullptr;

    printf("USM malloc shared with size %lu, total allocation size: %f GiB\n",
            size, 1.0 * (usm_shared_mem.size() + size) / (1024 * 1024 * 1024));

    static xpu::ocl::ext_func_t<clSharedMemAllocINTEL_func_t> ext_func(
            "clSharedMemAllocINTEL");
    cl_int err;
    void *p = ext_func(engine, get_ocl_context(engine), get_ocl_device(engine),
            nullptr, size, 0, &err);
    assert(utils::one_of(err, CL_SUCCESS, CL_OUT_OF_RESOURCES,
            CL_OUT_OF_HOST_MEMORY, CL_INVALID_BUFFER_SIZE));
    usm_shared_mem.add(p, size);
    printf("USM malloc shared ptr=%p\n", p);
    return p;
}

void free(impl::engine_t *engine, void *ptr) {
    using clMemFreeINTEL_func_t = cl_int (*)(cl_context, void *);

    if (!ptr) return;
    auto mem_entry = [&]() {
        if (usm_shared_mem.allocations.find(ptr)
                != usm_shared_mem.allocations.end()) {
            auto ret = usm_shared_mem.allocations[ptr];
            usm_shared_mem.remove(ptr);
            return std::tuple<const char *, size_t, size_t> {
                    "shared", ret, usm_shared_mem.size()};
        }
        if (usm_device_mem.allocations.find(ptr)
                != usm_device_mem.allocations.end()) {
            auto ret = usm_device_mem.allocations[ptr];
            usm_device_mem.remove(ptr);
            return std::tuple<const char *, size_t, size_t> {
                    "device", ret, usm_device_mem.size()};
        }
        if (usm_host_mem.allocations.find(ptr)
                != usm_host_mem.allocations.end()) {
            auto ret = usm_host_mem.allocations[ptr];
            usm_host_mem.remove(ptr);
            return std::tuple<const char *, size_t, size_t> {
                    "host", ret, usm_host_mem.size()};
        }
        return std::tuple<const char *, size_t, size_t> {"unknown", 0ul, 0ul};
    }();

    static xpu::ocl::ext_func_t<clMemFreeINTEL_func_t> ext_func(
            "clMemFreeINTEL");
    cl_int err = ext_func(engine, get_ocl_context(engine), ptr);
    assert(err == CL_SUCCESS);
    MAYBE_UNUSED(err);

    printf("USM free %s ptr=%p with size %lu, total allocation=%f GiB\n",
            std::get<0>(mem_entry), ptr, std::get<1>(mem_entry),
            1.0 * std::get<2>(mem_entry) / (1024 * 1024 * 1024));
}

status_t set_kernel_arg(impl::engine_t *engine, cl_kernel kernel, int arg_index,
        const void *arg_value) {
    using clSetKernelArgMemPointerINTEL_func_t
            = cl_int (*)(cl_kernel, cl_uint, const void *);
    static xpu::ocl::ext_func_t<clSetKernelArgMemPointerINTEL_func_t> ext_func(
            "clSetKernelArgMemPointerINTEL");
    return xpu::ocl::convert_to_dnnl(
            ext_func(engine, kernel, arg_index, arg_value));
}

status_t memcpy(impl::stream_t *stream, void *dst, const void *src, size_t size,
        cl_uint num_events, const cl_event *events, cl_event *out_event) {
    using clEnqueueMemcpyINTEL_func_t
            = cl_int (*)(cl_command_queue, cl_bool, void *, const void *,
                    size_t, cl_uint, const cl_event *, cl_event *);
    static xpu::ocl::ext_func_t<clEnqueueMemcpyINTEL_func_t> ext_func(
            "clEnqueueMemcpyINTEL");
    return xpu::ocl::convert_to_dnnl(
            ext_func(stream->engine(), get_ocl_queue(stream),
                    /* blocking */ CL_FALSE, dst, src, size, num_events, events,
                    out_event));
}

status_t memcpy(
        impl::stream_t *stream, void *dst, const void *src, size_t size) {
    return memcpy(stream, dst, src, size, 0, nullptr, nullptr);
}

status_t fill(impl::stream_t *stream, void *ptr, const void *pattern,
        size_t pattern_size, size_t size, cl_uint num_events,
        const cl_event *events, cl_event *out_event) {
    using clEnqueueMemFillINTEL_func_t
            = cl_int (*)(cl_command_queue, void *, const void *, size_t, size_t,
                    cl_uint, const cl_event *, cl_event *);
    static xpu::ocl::ext_func_t<clEnqueueMemFillINTEL_func_t> ext_func(
            "clEnqueueMemFillINTEL");
    return xpu::ocl::convert_to_dnnl(
            ext_func(stream->engine(), get_ocl_queue(stream), ptr, pattern,
                    pattern_size, size, num_events, events, out_event));
}

status_t memset(impl::stream_t *stream, void *ptr, int value, size_t size) {
    uint8_t pattern = (uint8_t)value;
    return fill(
            stream, ptr, &pattern, sizeof(uint8_t), size, 0, nullptr, nullptr);
}

xpu::ocl::usm::kind_t get_pointer_type(
        impl::engine_t *engine, const void *ptr) {
    using clGetMemAllocInfoINTEL_func_t = cl_int (*)(
            cl_context, const void *, cl_uint, size_t, void *, size_t *);

    // The values are taken from cl_ext.h to avoid dependency on the header.
    static constexpr cl_uint cl_mem_type_unknown_intel = 0x4196;
    static constexpr cl_uint cl_mem_type_host_intel = 0x4197;
    static constexpr cl_uint cl_mem_type_device_intel = 0x4198;
    static constexpr cl_uint cl_mem_type_shared_intel = 0x4199;

    static constexpr cl_uint cl_mem_alloc_type_intel = 0x419A;

    static xpu::ocl::ext_func_t<clGetMemAllocInfoINTEL_func_t> ext_func(
            "clGetMemAllocInfoINTEL");

    if (!ptr) return xpu::ocl::usm::kind_t::unknown;

    cl_uint alloc_type;
    cl_int err = ext_func(engine, get_ocl_context(engine), ptr,
            cl_mem_alloc_type_intel, sizeof(alloc_type), &alloc_type, nullptr);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) return xpu::ocl::usm::kind_t::unknown;

    switch (alloc_type) {
        case cl_mem_type_unknown_intel: return xpu::ocl::usm::kind_t::unknown;
        case cl_mem_type_host_intel: return xpu::ocl::usm::kind_t::host;
        case cl_mem_type_device_intel: return xpu::ocl::usm::kind_t::device;
        case cl_mem_type_shared_intel: return xpu::ocl::usm::kind_t::shared;
        default: assert(!"unknown alloc type");
    }
    return xpu::ocl::usm::kind_t::unknown;
}

} // namespace usm
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
