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

#ifndef GPU_COMPUTE_COMPUTE_STREAM_HPP
#define GPU_COMPUTE_COMPUTE_STREAM_HPP

#include "common/stream.hpp"
#include "gpu/compute/context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

class nd_range_t;
class kernel_arg_list_t;
struct stream_profiler_t;

class compute_stream_t : public stream_t {
public:
    using stream_t::stream_t;

    virtual status_t copy(const memory_storage_t &src,
            const memory_storage_t &dst, size_t size, const event_t &dep,
            event_t &out_dep)
            = 0;
    virtual status_t fill(const memory_storage_t &dst, uint8_t pattern,
            size_t size, const event_t &deps, event_t &out_dep)
            = 0;

    virtual context_t &ctx() = 0;
    virtual const context_t &ctx() const = 0;
    virtual const compute::stream_profiler_t &profiler() const = 0;
    virtual compute::stream_profiler_t &profiler() = 0;
    status_t notify_profiling_complete() const override;

protected:
    bool has_zero_pad_primitive() const {
        return engine()->kind() == dnnl_gpu;
    };

    status_t zero_pad(const memory_t *memory, const exec_ctx_t &ctx) override;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
