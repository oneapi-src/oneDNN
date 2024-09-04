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

#ifndef GPU_INTEL_COMPUTE_COMPUTE_STREAM_HPP
#define GPU_INTEL_COMPUTE_COMPUTE_STREAM_HPP

#include "gpu/gpu_stream.hpp"
#include "xpu/context.hpp"
#include "xpu/stream_profiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

class nd_range_t;
class kernel_arg_list_t;

class compute_stream_t : public gpu::stream_t {
public:
    using stream_t::stream_t;

    status_t notify_profiling_complete() const override;

    virtual status_t barrier() = 0;
    virtual status_t enter_immediate_mode() { return status::success; }
    virtual status_t exit_immediate_mode() { return status::success; }

protected:
    bool has_zero_pad_primitive() const {
        return engine()->kind() == dnnl_gpu;
    };

    status_t zero_pad(const memory_t *memory, const exec_ctx_t &ctx) override;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
