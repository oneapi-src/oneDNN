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

#ifndef GPU_GPU_STREAM_HPP
#define GPU_GPU_STREAM_HPP

#include "common/memory_storage.hpp"
#include "common/stream.hpp"

#include "xpu/context.hpp"
#include "xpu/stream_profiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

class stream_t : public impl::stream_t {
public:
    using dnnl::impl::stream_t::stream_t;

    virtual status_t copy(const memory_storage_t &src,
            const memory_storage_t &dst, size_t size, const xpu::event_t &dep,
            xpu::event_t &out_dep)
            = 0;
    virtual status_t fill(const memory_storage_t &dst, uint8_t pattern,
            size_t size, const xpu::event_t &deps, xpu::event_t &out_dep)
            = 0;

    virtual xpu::context_t &ctx() = 0;
    virtual const xpu::context_t &ctx() const = 0;
    virtual const xpu::stream_profiler_t &profiler() const {
        return *profiler_;
    }
    xpu::stream_profiler_t &profiler() { return *profiler_; }

    virtual double get_freq(const xpu::event_t &event) const { return 0.0; }

protected:
    std::unique_ptr<xpu::stream_profiler_t> profiler_;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
