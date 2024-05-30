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

#ifndef SYCL_STREAM_HPP
#define SYCL_STREAM_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/primitive_iface.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/compute_stream.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"
#include "gpu/sycl/sycl_gpu_engine.hpp"
#include "sycl/stream_profiler.hpp"
#include "sycl/sycl_context.hpp"
#include "xpu/sycl/memory_storage.hpp"
#include "xpu/sycl/stream_impl.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "sycl/sycl_stream_cpu_thunk.hpp"
#include "sycl/sycl_stream_submit_cpu_primitive.hpp"
#endif

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <utility>
#include <CL/cl.h>

namespace dnnl {
namespace impl {
namespace sycl {

struct sycl_stream_t : public gpu::intel::compute::compute_stream_t {
    static status_t create_stream(
            stream_t **stream, engine_t *engine, unsigned flags) {
        std::unique_ptr<sycl_stream_t> sycl_stream(
                new sycl_stream_t(engine, flags));
        if (!sycl_stream) return status::out_of_memory;

        status_t status = sycl_stream->init();
        if (status != status::success) return status;
        *stream = sycl_stream.release();
        return status::success;
    }

    static status_t create_stream(
            stream_t **stream, engine_t *engine, ::sycl::queue &queue) {
        unsigned flags;
        status_t status = sycl_stream_t::init_flags(&flags, queue);
        if (status != status::success) return status;

        std::unique_ptr<sycl_stream_t> sycl_stream(
                new sycl_stream_t(engine, flags, queue));

        status = sycl_stream->init();
        if (status != status::success) return status;

        *stream = sycl_stream.release();
        return status::success;
    }

    status_t wait() override {
        queue().wait_and_throw();
        return status::success;
    }

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

    const xpu::stream_profiler_t &profiler() const override {
        return *profiler_;
    }

    xpu::stream_profiler_t &profiler() override { return *profiler_; }

    ::sycl::queue &queue() const { return *impl()->queue(); }

    status_t enqueue_primitive(const primitive_iface_t *prim_iface,
            exec_ctx_t &exec_ctx) override {
        auto execute_func = [&]() {
            status_t status = status::success;
            if (engine()->kind() == engine_kind::cpu) {

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
                auto event = queue().submit([&](::sycl::handler &cgh) {
                    register_deps(cgh);
                    submit_cpu_primitive(this, prim_iface, exec_ctx, cgh);
                });
                sycl_ctx().set_deps({event});
#else
                assert(!"not expected");
                return status::runtime_error;
#endif
            } else if (engine()->kind() == engine_kind::gpu) {
                status = prim_iface->execute(exec_ctx);
            } else {
                assert(!"not expected");
            }
            return status;
        };
        status_t status = execute_func();
        return status;
    }

    status_t copy(const memory_storage_t &src, const memory_storage_t &dst,
            size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep) override {
        CHECK(impl()->copy(this, src, dst, size, deps, out_dep));

        if (is_profiling_enabled()) {
            assert(impl::sycl::sycl_event_t::from(out_dep).size() == 1);
            auto sycl_event = utils::make_unique<impl::sycl::sycl_event_t>(
                    std::vector<::sycl::event> {
                            impl::sycl::sycl_event_t::from(out_dep)[0]});
            profiler_->register_event(std::move(sycl_event));
        }
        return status::success;
    }

    status_t fill(const memory_storage_t &dst, uint8_t pattern, size_t size,
            const xpu::event_t &deps, xpu::event_t &out_dep) override {
        CHECK(impl()->fill(dst, pattern, size, deps, out_dep));

        if (is_profiling_enabled()) {
            assert(impl::sycl::sycl_event_t::from(out_dep).size() == 1);
            auto sycl_event = utils::make_unique<impl::sycl::sycl_event_t>(
                    std::vector<::sycl::event> {
                            impl::sycl::sycl_event_t::from(out_dep)[0]});
            profiler_->register_event(std::move(sycl_event));
        }
        return status::success;
    }

    const sycl_context_t &sycl_ctx() const { return impl()->sycl_ctx(); }
    sycl_context_t &sycl_ctx() { return impl()->sycl_ctx(); }

    xpu::context_t &ctx() override { return impl()->sycl_ctx(); }
    const xpu::context_t &ctx() const override { return impl()->sycl_ctx(); }

    ::sycl::event get_output_event() const {
        return impl()->get_output_event();
    }

    void register_deps(::sycl::handler &cgh) const {
        return impl()->register_deps(cgh);
    }

    template <::sycl::access_mode mode>
    ::sycl::accessor<uint8_t, 1, mode> get_dummy_accessor(
            ::sycl::handler &cgh) {
        return dummy_buffer_.get_access<mode>(cgh);
    }

protected:
    xpu::sycl::stream_impl_t *impl() const {
        return (xpu::sycl::stream_impl_t *)stream_t::impl_.get();
    }

    sycl_stream_t(engine_t *engine, unsigned flags)
        : gpu::intel::compute::compute_stream_t(
                engine, new xpu::sycl::stream_impl_t(flags)) {}
    sycl_stream_t(engine_t *engine, unsigned flags, ::sycl::queue &queue)
        : gpu::intel::compute::compute_stream_t(
                engine, new xpu::sycl::stream_impl_t(queue, flags)) {}

    static status_t init_flags(unsigned *flags, ::sycl::queue &queue) {
        *flags = queue.is_in_order() ? stream_flags::in_order
                                     : stream_flags::out_of_order;

#ifdef DNNL_EXPERIMENTAL_PROFILING
        if (queue.has_property<::sycl::property::queue::enable_profiling>())
            *flags |= stream_flags::profiling;
#endif
        return status::success;
    }

    std::unique_ptr<xpu::stream_profiler_t> profiler_;

    // XXX: this is a temporary solution to make sycl_memory_arg_t
    // default constructible.
    xpu::sycl::buffer_u8_t dummy_buffer_ = xpu::sycl::buffer_u8_t(1);

private:
    status_t init();
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
