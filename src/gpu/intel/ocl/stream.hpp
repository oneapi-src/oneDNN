/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_STREAM_HPP
#define GPU_INTEL_OCL_STREAM_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/thread_local_storage.hpp"

#include "xpu/stream_profiler.hpp"

#include "xpu/ocl/context.hpp"
#include "xpu/ocl/stream_impl.hpp"

#include "gpu/intel/compute/compute_stream.hpp"
#include "gpu/intel/ocl/mdapi_utils.hpp"
#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct stream_t : public compute::compute_stream_t {
    static status_t create_stream(impl::stream_t **stream,
            impl::engine_t *engine, impl::stream_impl_t *stream_impl) {

        std::unique_ptr<stream_t> s(new stream_t(engine, stream_impl));
        if (!s) return status::out_of_memory;

        status_t status = s->init();
        if (status != status::success) {
            // Stream owns stream_impl only if it's created successfully (including initialization).
            s->impl_.release();
            return status;
        }

        *stream = s.release();
        return status::success;
    }

    status_t wait() override { return impl()->wait(); }

    void before_exec_hook() override;
    void after_exec_hook() override;

    double get_freq(const xpu::event_t &event) const override {
        const auto &ocl_event = xpu::ocl::event_t::from(event).events;
        gpu_assert(ocl_event.size() == 1);
        return mdapi_helper().get_freq(ocl_event[0]);
    }

    status_t reset_profiling() override {
        if (!is_profiling_enabled()) return status::invalid_arguments;
        profiler_->reset();
        return status::success;
    }

    status_t get_profiling_data(profiling_data_kind_t data_kind,
            int *num_entries, uint64_t *data) const override {
        if (!is_profiling_enabled()) return status::invalid_arguments;
        return profiler_->get_info(data_kind, num_entries, data);
    }

    cl_command_queue queue() const { return impl()->queue(); }

    const mdapi_helper_t &mdapi_helper() const { return *mdapi_helper_; }

    status_t copy(const memory_storage_t &src, const memory_storage_t &dst,
            size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep) override;

    status_t fill(const memory_storage_t &dst, uint8_t pattern, size_t size,
            const xpu::event_t &deps, xpu::event_t &out_dep) override;

    status_t barrier() override;

    ~stream_t() override = default;

    const xpu::ocl::context_t &ocl_ctx() const { return impl()->ocl_ctx(); }
    xpu::ocl::context_t &ocl_ctx() { return impl()->ocl_ctx(); }
    xpu::context_t &ctx() override { return impl()->ocl_ctx(); }
    const xpu::context_t &ctx() const override { return impl()->ocl_ctx(); }

    const xpu::ocl::wrapper_t<cl_event> &get_output_event() const {
        return impl()->get_output_event();
    }

private:
    xpu::ocl::stream_impl_t *impl() const {
        return (xpu::ocl::stream_impl_t *)impl::stream_t::impl_.get();
    }

    stream_t(impl::engine_t *engine, impl::stream_impl_t *stream_impl)
        : compute_stream_t(engine, stream_impl) {}

    status_t init();

    cl_command_queue create_queue(
            cl_context ctx, cl_device_id dev, cl_int *err) const;

    std::unique_ptr<mdapi_helper_t> mdapi_helper_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
