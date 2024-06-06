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

#include "common/engine.hpp"
#include "common/stream.hpp"

#include "xpu/stream_profiler.hpp"

#include "xpu/ocl/stream_impl.hpp"

// TODO: move memory storage to xpu
#include "gpu/intel/ocl/ocl_memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

status_t stream_impl_t::copy(impl::stream_t *stream,
        const memory_storage_t &src, const memory_storage_t &dst, size_t size,
        const xpu::event_t &deps, xpu::event_t &out_dep,
        xpu::stream_profiler_t *stream_profiler) {

    if (size == 0) return status::success;

    std::vector<cl_event> events = [&] {
        if (flags() & stream_flags::out_of_order) {
            const auto &event_wrappers
                    = gpu::intel::ocl::ocl_event_t::from(deps).events;
            return std::vector<cl_event>(
                    event_wrappers.begin(), event_wrappers.end());
        }
        return std::vector<cl_event> {};
    }();
    cl_uint num_events = (cl_uint)events.size();
    const cl_event *events_ptr = events.data();

    xpu::ocl::wrapper_t<cl_event> out_event;
    bool need_out_event
            = is_profiling_enabled() || flags() & stream_flags::out_of_order;
    cl_event *out_event_ptr = need_out_event ? &out_event.unwrap() : nullptr;

    if (dst.engine()->kind() == engine_kind::gpu
            && src.engine() == dst.engine()) {
        auto *ocl_src = utils::downcast<
                const gpu::intel::ocl::ocl_memory_storage_base_t *>(&src);
        auto *ocl_dst = utils::downcast<
                const gpu::intel::ocl::ocl_memory_storage_base_t *>(&dst);

        if (ocl_src->memory_kind() == ocl_dst->memory_kind()) {
            if (ocl_src->memory_kind() == gpu::intel::ocl::memory_kind::usm
                    && ocl_dst->memory_kind()
                            == gpu::intel::ocl::memory_kind::usm) {
                const auto *ocl_usm_src = utils::downcast<
                        const gpu::intel::ocl::ocl_usm_memory_storage_t *>(
                        ocl_src);
                const auto *ocl_usm_dst = utils::downcast<
                        const gpu::intel::ocl::ocl_usm_memory_storage_t *>(
                        ocl_dst);
                CHECK(gpu::intel::ocl::usm::memcpy(stream,
                        ocl_usm_dst->usm_ptr(), ocl_usm_src->usm_ptr(), size,
                        num_events, events_ptr, out_event_ptr));
            }
            if (ocl_src->memory_kind() == gpu::intel::ocl::memory_kind::buffer
                    && ocl_dst->memory_kind()
                            == gpu::intel::ocl::memory_kind::buffer) {
                const auto *ocl_buffer_src = utils::downcast<
                        const gpu::intel::ocl::ocl_buffer_memory_storage_t *>(
                        ocl_src);
                const auto *ocl_buffer_dst = utils::downcast<
                        const gpu::intel::ocl::ocl_buffer_memory_storage_t *>(
                        ocl_dst);
                OCL_CHECK(clEnqueueCopyBuffer(queue(),
                        ocl_buffer_src->mem_object(),
                        ocl_buffer_dst->mem_object(), src.offset(),
                        dst.offset(), size, num_events, events_ptr,
                        out_event_ptr));
            }
        }
    } else if (src.engine()->kind() == engine_kind::cpu
            && is_native_runtime(src.engine()->runtime_kind())) {
        assert(dst.engine()->kind() == engine_kind::gpu);

        void *src_ptr = nullptr;
        src.get_data_handle(&src_ptr);

        const auto *ocl_dst = utils::downcast<
                const gpu::intel::ocl::ocl_memory_storage_base_t *>(&dst);
        bool usm_dst
                = ocl_dst->memory_kind() == gpu::intel::ocl::memory_kind::usm;

        if (usm_dst) {
            const auto *ocl_usm_dst = utils::downcast<
                    const gpu::intel::ocl::ocl_usm_memory_storage_t *>(ocl_dst);
            CHECK(gpu::intel::ocl::usm::memcpy(stream, ocl_usm_dst->usm_ptr(),
                    src_ptr, size, num_events, events_ptr, out_event_ptr));
        } else {
            const auto *ocl_buffer_dst = utils::downcast<
                    const gpu::intel::ocl::ocl_buffer_memory_storage_t *>(
                    ocl_dst);

            cl_mem ocl_mem = ocl_buffer_dst->mem_object();
            cl_int err = clEnqueueWriteBuffer(queue(), ocl_mem, CL_TRUE, 0,
                    size, src_ptr, num_events, events_ptr, out_event_ptr);
            OCL_CHECK(err);
        }
    } else if (dst.engine()->kind() == engine_kind::cpu
            && is_native_runtime(dst.engine()->runtime_kind())) {
        assert(src.engine()->kind() == engine_kind::gpu);

        void *dst_ptr = nullptr;
        dst.get_data_handle(&dst_ptr);

        const auto *ocl_src = utils::downcast<
                const gpu::intel::ocl::ocl_memory_storage_base_t *>(&src);
        bool usm_src
                = ocl_src->memory_kind() == gpu::intel::ocl::memory_kind::usm;

        if (usm_src) {
            const auto *ocl_usm_src = utils::downcast<
                    const gpu::intel::ocl::ocl_usm_memory_storage_t *>(ocl_src);
            CHECK(gpu::intel::ocl::usm::memcpy(stream, dst_ptr,
                    ocl_usm_src->usm_ptr(), size, num_events, events_ptr,
                    out_event_ptr));
        } else {
            const auto *ocl_buffer_src = utils::downcast<
                    const gpu::intel::ocl::ocl_buffer_memory_storage_t *>(
                    ocl_src);

            cl_mem ocl_mem = ocl_buffer_src->mem_object();
            cl_int err = clEnqueueReadBuffer(queue(), ocl_mem, CL_TRUE, 0, size,
                    dst_ptr, num_events, events_ptr, out_event_ptr);
            OCL_CHECK(err);
        }
    } else {
        CHECK(wait());

        // Use map/unmap
        void *src_mapped_ptr;
        void *dst_mapped_ptr;

        CHECK(src.map_data(&src_mapped_ptr, stream, size));
        CHECK(dst.map_data(&dst_mapped_ptr, stream, size));

        std::memcpy(static_cast<void *>(dst_mapped_ptr),
                static_cast<const void *>(src_mapped_ptr), size);

        CHECK(src.unmap_data(src_mapped_ptr, stream));
        CHECK(dst.unmap_data(dst_mapped_ptr, stream));

        // Short-circuit event management due to calls to wait
        return status::success;
    }

    if (flags() & stream_flags::out_of_order)
        gpu::intel::ocl::ocl_event_t::from(out_dep).events
                = {std::move(out_event)};

    return status::success;
}

status_t stream_impl_t::fill(impl::stream_t *stream,
        const memory_storage_t &dst, uint8_t pattern, size_t size,
        const xpu::event_t &deps, xpu::event_t &out_dep,
        xpu::stream_profiler_t *stream_profiler) {
    using namespace dnnl::impl::utils;

    const auto *ocl_dst
            = downcast<const gpu::intel::ocl::ocl_memory_storage_base_t *>(
                    &dst);

    std::vector<cl_event> events = [&] {
        if (flags() & stream_flags::out_of_order) {
            const auto &event_wrappers
                    = gpu::intel::ocl::ocl_event_t::from(deps).events;
            return std::vector<cl_event>(
                    event_wrappers.begin(), event_wrappers.end());
        }
        return std::vector<cl_event> {};
    }();
    cl_uint num_events = (cl_uint)events.size();
    const cl_event *events_ptr = events.data();

    xpu::ocl::wrapper_t<cl_event> out_event;
    bool need_out_event
            = is_profiling_enabled() || flags() & stream_flags::out_of_order;
    cl_event *out_event_ptr = need_out_event ? &out_event.unwrap() : nullptr;

    if (ocl_dst->memory_kind() == gpu::intel::ocl::memory_kind::usm) {
        const auto *ocl_usm_dst
                = downcast<const gpu::intel::ocl::ocl_usm_memory_storage_t *>(
                        ocl_dst);
        CHECK(gpu::intel::ocl::usm::fill(stream, ocl_usm_dst->usm_ptr(),
                &pattern, sizeof(pattern), size, num_events, events_ptr,
                out_event_ptr));
    } else {
        const auto *ocl_buffer_dst = downcast<
                const gpu::intel::ocl::ocl_buffer_memory_storage_t *>(ocl_dst);
        cl_int err = clEnqueueFillBuffer(queue(), ocl_buffer_dst->mem_object(),
                &pattern, sizeof(uint8_t), dst.offset(), size, num_events,
                events_ptr, out_event_ptr);
        OCL_CHECK(err);
    }

    if (is_profiling_enabled()) {
        auto ocl_event = utils::make_unique<gpu::intel::ocl::ocl_event_t>(
                std::vector<xpu::ocl::wrapper_t<cl_event>> {out_event});
        stream_profiler->register_event(std::move(ocl_event));
    }

    if (flags() & stream_flags::out_of_order)
        gpu::intel::ocl::ocl_event_t::from(out_dep).events
                = {std::move(out_event)};

    return status::success;
}

const gpu::intel::ocl::ocl_context_t &stream_impl_t::ocl_ctx() const {
    static gpu::intel::ocl::ocl_context_t empty_ctx {};
    return ctx_.get(empty_ctx);
}

gpu::intel::ocl::ocl_context_t &stream_impl_t::ocl_ctx() {
    const gpu::intel::ocl::ocl_context_t &ctx
            = const_cast<const stream_impl_t *>(this)->ocl_ctx();
    return *const_cast<gpu::intel::ocl::ocl_context_t *>(&ctx);
}

xpu::context_t &stream_impl_t::ctx() {
    return ocl_ctx();
}

const xpu::context_t &stream_impl_t::ctx() const {
    return ocl_ctx();
}

const xpu::ocl::wrapper_t<cl_event> &stream_impl_t::get_output_event() const {
    auto &deps = gpu::intel::ocl::ocl_event_t::from(ctx().get_deps());
    assert(deps.size() == 1);
    return deps[0];
}

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl
