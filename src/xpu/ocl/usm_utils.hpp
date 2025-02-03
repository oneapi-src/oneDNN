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

#ifndef XPU_OCL_USM_UTILS_HPP
#define XPU_OCL_USM_UTILS_HPP

#include <CL/cl.h>

#include "common/engine.hpp"
#include "common/stream.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {
namespace usm {

enum class kind_t { unknown, host, device, shared };

bool is_usm_supported(impl::engine_t *engine);
void *malloc_host(impl::engine_t *engine, size_t size);
void DNNL_API *malloc_device(impl::engine_t *engine, size_t size);
void DNNL_API *malloc_shared(impl::engine_t *engine, size_t size);

void DNNL_API free(impl::engine_t *engine, void *ptr);
status_t DNNL_API set_kernel_arg(impl::engine_t *engine, cl_kernel kernel, int arg_index,
        const void *arg_value);
status_t memcpy(impl::stream_t *stream, void *dst, const void *src, size_t size,
        cl_uint num_events, const cl_event *events, cl_event *out_event);
status_t DNNL_API memcpy(
        impl::stream_t *stream, void *dst, const void *src, size_t size);
status_t fill(impl::stream_t *stream, void *ptr, const void *pattern,
        size_t pattern_size, size_t size, cl_uint num_events,
        const cl_event *events, cl_event *out_event);
status_t DNNL_API memset(
        impl::stream_t *stream, void *ptr, int value, size_t size);
kind_t DNNL_API get_pointer_type(impl::engine_t *engine, const void *ptr);

} // namespace usm
} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
