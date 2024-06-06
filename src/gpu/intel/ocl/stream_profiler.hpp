/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_STREAM_PROFILER_HPP
#define GPU_INTEL_OCL_STREAM_PROFILER_HPP

#include "common/c_types_map.hpp"

#include "xpu/stream_profiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ocl_stream_profiler_t : public xpu::stream_profiler_t {
    ocl_stream_profiler_t(const impl::stream_t *stream)
        : stream_profiler_t(stream) {}

    status_t get_info(profiling_data_kind_t data_kind, int *num_entries,
            uint64_t *data) const override;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
