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

#ifndef GPU_INTEL_OCL_USM_UTILS_HPP
#define GPU_INTEL_OCL_USM_UTILS_HPP

#include "xpu/ocl/usm_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {
namespace usm {

bool is_usm_supported(impl::engine_t *engine);
void *malloc_host(impl::engine_t *engine, size_t size);
void *malloc_device(impl::engine_t *engine, size_t size);
void *malloc_shared(impl::engine_t *engine, size_t size);

void free(impl::engine_t *engine, void *ptr);
status_t set_kernel_arg(impl::engine_t *engine, cl_kernel kernel, int arg_index,
        const void *arg_value);
status_t memcpy(impl::stream_t *stream, void *dst, const void *src, size_t size,
        cl_uint num_events, const cl_event *events, cl_event *out_event);
status_t memcpy(
        impl::stream_t *stream, void *dst, const void *src, size_t size);
status_t fill(impl::stream_t *stream, void *ptr, const void *pattern,
        size_t pattern_size, size_t size, cl_uint num_events,
        const cl_event *events, cl_event *out_event);
status_t memset(impl::stream_t *stream, void *ptr, int value, size_t size);
xpu::ocl::usm::kind_t get_pointer_type(impl::engine_t *engine, const void *ptr);

} // namespace usm
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
