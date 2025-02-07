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

#ifndef GPU_INTEL_SYCL_STREAM_HPP
#define GPU_INTEL_SYCL_STREAM_HPP

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <utility>
#include <CL/cl.h>

#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/primitive_iface.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "xpu/sycl/context.hpp"
#include "xpu/sycl/memory_storage.hpp"
#include "xpu/sycl/stream_impl.hpp"
#include "xpu/sycl/stream_profiler.hpp"

#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/compute_stream.hpp"

#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

struct stream_t : public gpu::intel::compute::compute_stream_t {
    static status_t create_stream(impl::stream_t **stream,
            impl::engine_t *engine, impl::stream_impl_t *stream_impl) {
        std::unique_ptr<intel::sycl::stream_t> s(
                new intel::sycl::stream_t(engine, stream_impl));
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

    ::sycl::queue &queue() const { return *impl()->queue(); }

    status_t enqueue_primitive(const primitive_iface_t *prim_iface,
            exec_ctx_t &exec_ctx) override {
        return prim_iface->execute(exec_ctx);
    }

    status_t copy(const memory_storage_t &src, const memory_storage_t &dst,
            size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep) override {
        return impl()->copy(
                this, src, dst, size, deps, out_dep, profiler_.get());
    }

    status_t fill(const memory_storage_t &dst, uint8_t pattern, size_t size,
            const xpu::event_t &deps, xpu::event_t &out_dep) override {
        return impl()->fill(dst, pattern, size, deps, out_dep, profiler_.get());
    }

    status_t barrier() override { return impl()->barrier(); }

    const xpu::sycl::context_t &sycl_ctx() const { return impl()->sycl_ctx(); }
    xpu::sycl::context_t &sycl_ctx() { return impl()->sycl_ctx(); }

    xpu::context_t &ctx() override { return impl()->sycl_ctx(); }
    const xpu::context_t &ctx() const override { return impl()->sycl_ctx(); }

    ::sycl::event get_output_event() const {
        return impl()->get_output_event();
    }

    void register_deps(::sycl::handler &cgh) const {
        return impl()->register_deps(cgh);
    }

    bool recording() const;
    using weak_graph_t = ::sycl::ext::oneapi::weak_object<
            ::sycl::ext::oneapi::experimental::command_graph<::sycl::ext::
                            oneapi::experimental::graph_state::modifiable>>;
    weak_graph_t get_current_graph_weak() const;

    status_t enter_immediate_mode() override;
    status_t exit_immediate_mode() override;

protected:
    xpu::sycl::stream_impl_t *impl() const {
        return (xpu::sycl::stream_impl_t *)impl::stream_t::impl_.get();
    }

    stream_t(impl::engine_t *engine, impl::stream_impl_t *stream_impl)
        : gpu::intel::compute::compute_stream_t(engine, stream_impl) {}

private:
    status_t init();

    status_t pause_recording();
    status_t resume_recording();

    std::mutex immediate_mode_mutex_;
    int immediate_mode_level_ = 0;
    std::unique_ptr<::sycl::ext::oneapi::experimental::command_graph<
            ::sycl::ext::oneapi::experimental::graph_state::modifiable>>
            paused_graph_;
    xpu::sycl::event_t paused_dep_;
};

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
