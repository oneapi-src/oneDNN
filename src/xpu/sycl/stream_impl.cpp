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

#include "xpu/sycl/memory_storage.hpp"
#include "xpu/sycl/stream_impl.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

status_t stream_impl_t::copy(impl::stream_t *stream,
        const memory_storage_t &src, const memory_storage_t &dst, size_t size,
        const xpu::event_t &deps, xpu::event_t &out_dep,
        xpu::stream_profiler_t *stream_profiler) {

    if (size == 0) return status::success;
    // TODO: add src and dst sizes check

    const bool host_mem_src = src.engine()->kind() == engine_kind::cpu
            && is_native_runtime(src.engine()->runtime_kind());
    const bool host_mem_dst = dst.engine()->kind() == engine_kind::cpu
            && is_native_runtime(dst.engine()->runtime_kind());

    // Handle cases when GPU runtime is SYCL and CPU runtime is not.
    if (host_mem_src || host_mem_dst) {
        void *src_mapped_ptr;
        void *dst_mapped_ptr;

        // It is allowed for src or dst memory to be created for an engine that
        // is not associated with the stream passed to this function. It is done
        // to enabled cross engine reordering.
        //
        // For example, there are two memory objects created using different
        // engines. One of the engines was then used to create the reorder
        // primitive and a stream. In this case only one memory object contains
        // an engine that matches the engine contained by the stream.
        //
        // The SYCL copy routines require both pointers (src and dst) to be
        // associated with the same context as the queue the copy routine runs
        // on.
        auto *src_map_stream
                = src.engine() == stream->engine() ? stream : nullptr;
        auto *dst_map_stream
                = dst.engine() == stream->engine() ? stream : nullptr;

        CHECK(src.map_data(&src_mapped_ptr, src_map_stream, size));
        CHECK(dst.map_data(&dst_mapped_ptr, dst_map_stream, size));

        std::memcpy(static_cast<void *>(dst_mapped_ptr),
                static_cast<const void *>(src_mapped_ptr), size);

        CHECK(src.unmap_data(src_mapped_ptr, src_map_stream));
        CHECK(dst.unmap_data(dst_mapped_ptr, dst_map_stream));

        return status::success;
    }

    // Handle all other cases.
    auto *sycl_src
            = utils::downcast<const xpu::sycl::memory_storage_base_t *>(&src);
    auto *sycl_dst
            = utils::downcast<const xpu::sycl::memory_storage_base_t *>(&dst);
    bool usm_src = sycl_src->memory_kind() == xpu::sycl::memory_kind::usm;
    bool usm_dst = sycl_dst->memory_kind() == xpu::sycl::memory_kind::usm;
    ::sycl::event e;

    if (usm_src && usm_dst) {
        auto *usm_src
                = utils::downcast<const xpu::sycl::usm_memory_storage_t *>(
                        &src);
        auto *usm_dst
                = utils::downcast<const xpu::sycl::usm_memory_storage_t *>(
                        &dst);
        e = queue()->submit([&](::sycl::handler &cgh) {
            cgh.depends_on(xpu::sycl::event_t::from(deps).events);
            cgh.memcpy(usm_dst->usm_ptr(), usm_src->usm_ptr(), size);
        });
    } else if (usm_src && !usm_dst) {
        auto *usm_src
                = utils::downcast<const xpu::sycl::usm_memory_storage_t *>(
                        &src);
        auto *buffer_dst
                = utils::downcast<const xpu::sycl::buffer_memory_storage_t *>(
                        &dst);
        auto &b_dst = buffer_dst->buffer();
        e = queue()->submit([&](::sycl::handler &cgh) {
            cgh.depends_on(xpu::sycl::event_t::from(deps).events);
            auto acc_dst = b_dst.get_access<::sycl::access::mode::write>(cgh);
            cgh.copy(usm_src->usm_ptr(), acc_dst);
        });
    } else if (!usm_src && usm_dst) {
        auto *buffer_src
                = utils::downcast<const xpu::sycl::buffer_memory_storage_t *>(
                        &src);
        auto &b_src = buffer_src->buffer();
        auto *usm_dst
                = utils::downcast<const xpu::sycl::usm_memory_storage_t *>(
                        &dst);
        e = queue()->submit([&](::sycl::handler &cgh) {
            cgh.depends_on(xpu::sycl::event_t::from(deps).events);
            auto acc_src = b_src.get_access<::sycl::access::mode::read>(cgh);
            cgh.copy(acc_src, usm_dst->usm_ptr());
        });
    } else { // if (!usm_src && !usm_dst)
        assert(!usm_src && !usm_dst && "USM is not supported yet");
        auto *buffer_src
                = utils::downcast<const xpu::sycl::buffer_memory_storage_t *>(
                        &src);
        auto *buffer_dst
                = utils::downcast<const xpu::sycl::buffer_memory_storage_t *>(
                        &dst);
        auto &b_src = buffer_src->buffer();
        auto &b_dst = buffer_dst->buffer();
        e = queue()->submit([&](::sycl::handler &cgh) {
            auto acc_src = b_src.get_access<::sycl::access::mode::read>(cgh);
            auto acc_dst = b_dst.get_access<::sycl::access::mode::write>(cgh);
            cgh.depends_on(xpu::sycl::event_t::from(deps).events);
            cgh.copy(acc_src, acc_dst);
        });
    }

    if (is_profiling_enabled()) {
        auto sycl_event = utils::make_unique<xpu::sycl::event_t>(
                std::vector<::sycl::event> {e});
        stream_profiler->register_event(std::move(sycl_event));
    }

    xpu::sycl::event_t::from(out_dep).events = {e};

    return status::success;
}

status_t stream_impl_t::fill(const memory_storage_t &dst, uint8_t pattern,
        size_t size, const xpu::event_t &deps, xpu::event_t &out_dep,
        xpu::stream_profiler_t *stream_profiler) {
    auto *sycl_dst
            = utils::downcast<const xpu::sycl::memory_storage_base_t *>(&dst);
    bool usm = sycl_dst->memory_kind() == xpu::sycl::memory_kind::usm;

    ::sycl::event out_event;

    if (usm) {
        auto *usm_dst
                = utils::downcast<const xpu::sycl::usm_memory_storage_t *>(
                        &dst);
        auto dst_ptr = static_cast<uint8_t *>(usm_dst->usm_ptr());
        // Note: we cannot use queue_.fill since it cannot handle
        // events as input
        out_event = queue()->submit([&](::sycl::handler &cgh) {
            cgh.depends_on(xpu::sycl::event_t::from(deps).events);
            cgh.memset(dst_ptr, pattern, size);
        });
    } else {
        auto *buffer_dst
                = utils::downcast<const xpu::sycl::buffer_memory_storage_t *>(
                        &dst);
        out_event = queue()->submit([&](::sycl::handler &cgh) {
            // need a u8 accessor to get the proper range
            ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write,
                    xpu::sycl::compat::target_device>
                    acc_dst(buffer_dst->buffer(), cgh, ::sycl::range<1>(size),
                            ::sycl::id<1>(0));
            cgh.depends_on(xpu::sycl::event_t::from(deps).events);
            cgh.fill(acc_dst, pattern);
        });
    }

    if (is_profiling_enabled()) {
        auto sycl_event = utils::make_unique<xpu::sycl::event_t>(
                std::vector<::sycl::event> {out_event});
        stream_profiler->register_event(std::move(sycl_event));
    }

    xpu::sycl::event_t::from(out_dep).events = {out_event};
    return status::success;
}

status_t stream_impl_t::barrier() {
    queue()->ext_oneapi_submit_barrier();
    return status::success;
}

const xpu::sycl::context_t &stream_impl_t::sycl_ctx() const {
    static xpu::sycl::context_t empty_ctx {};
    return ctx_.get(empty_ctx);
}

xpu::sycl::context_t &stream_impl_t::sycl_ctx() {
    const xpu::sycl::context_t &ctx
            = const_cast<const stream_impl_t *>(this)->sycl_ctx();
    return *const_cast<xpu::sycl::context_t *>(&ctx);
}

xpu::context_t &stream_impl_t::ctx() {
    return sycl_ctx();
}
const xpu::context_t &stream_impl_t::ctx() const {
    return sycl_ctx();
}

::sycl::event stream_impl_t::get_output_event() {
    // Fast path: if only one event, return it.
    auto &deps = sycl_ctx().get_sycl_deps();
    if (deps.size() == 1) return deps[0];

    // Otherwise, we run a trivial kernel to gather all deps. The
    // dummy task is needed to not get an error related to empty
    // kernel.
    auto e = queue()->submit([&](::sycl::handler &cgh) {
        register_deps(cgh);
        cgh.single_task<class dnnl_dummy_kernel>([]() {});
    });
    return e;
}

void stream_impl_t::register_deps(::sycl::handler &cgh) const {
    cgh.depends_on(sycl_ctx().get_sycl_deps().events);
}

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl
