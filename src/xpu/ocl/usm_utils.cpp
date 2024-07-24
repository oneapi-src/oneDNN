/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "xpu/ocl/usm_utils.hpp"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/ocl/usm_utils.hpp"

#define HANDLE_USM_CALL_V(e, ...) \
    assert(e->kind() == engine_kind::gpu); \
    gpu::intel::ocl::usm::__VA_ARGS__

#define HANDLE_USM_CALL(e, ...) \
    assert(e->kind() == engine_kind::gpu); \
    return gpu::intel::ocl::usm::__VA_ARGS__

#else
#error "Unsupported vendor"
#endif

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {
namespace usm {

bool is_usm_supported(impl::engine_t *engine) {
    HANDLE_USM_CALL(engine, is_usm_supported(engine));
}

void *malloc_host(impl::engine_t *engine, size_t size) {
    HANDLE_USM_CALL(engine, malloc_host(engine, size));
}

void *malloc_device(impl::engine_t *engine, size_t size) {
    HANDLE_USM_CALL(engine, malloc_device(engine, size));
}

void *malloc_shared(impl::engine_t *engine, size_t size) {
    HANDLE_USM_CALL(engine, malloc_shared(engine, size));
}

void free(impl::engine_t *engine, void *ptr) {
    HANDLE_USM_CALL_V(engine, free(engine, ptr));
}

status_t set_kernel_arg(impl::engine_t *engine, cl_kernel kernel, int arg_index,
        const void *arg_value) {
    HANDLE_USM_CALL(
            engine, set_kernel_arg(engine, kernel, arg_index, arg_value));
}

status_t memcpy(impl::stream_t *stream, void *dst, const void *src, size_t size,
        cl_uint num_events, const cl_event *events, cl_event *out_event) {
    HANDLE_USM_CALL(stream->engine(),
            memcpy(stream, dst, src, size, num_events, events, out_event));
}

status_t memcpy(
        impl::stream_t *stream, void *dst, const void *src, size_t size) {
    return memcpy(stream, dst, src, size, 0, nullptr, nullptr);
}

status_t fill(impl::stream_t *stream, void *ptr, const void *pattern,
        size_t pattern_size, size_t size, cl_uint num_events,
        const cl_event *events, cl_event *out_event) {
    HANDLE_USM_CALL(stream->engine(),
            fill(stream, ptr, pattern, pattern_size, size, num_events, events,
                    out_event));
}

status_t memset(impl::stream_t *stream, void *ptr, int value, size_t size) {
    uint8_t pattern = (uint8_t)value;
    return fill(
            stream, ptr, &pattern, sizeof(uint8_t), size, 0, nullptr, nullptr);
}

kind_t get_pointer_type(impl::engine_t *engine, const void *ptr) {
    HANDLE_USM_CALL(engine, get_pointer_type(engine, ptr));
}

} // namespace usm
} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl
