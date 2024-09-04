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

#ifndef XPU_SYCL_STREAM_IMPL_HPP
#define XPU_SYCL_STREAM_IMPL_HPP

#include "common/stream_impl.hpp"
#include "common/thread_local_storage.hpp"
#include "common/utils.hpp"

#include "xpu/context.hpp"
#include "xpu/stream_profiler.hpp"

#include "xpu/sycl/compat.hpp"
#include "xpu/sycl/context.hpp"
#include "xpu/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

class stream_impl_t : public impl::stream_impl_t {
public:
    stream_impl_t() = delete;
    stream_impl_t(unsigned flags) : impl::stream_impl_t(flags) {}
    stream_impl_t(const ::sycl::queue &queue, unsigned flags)
        : impl::stream_impl_t(flags), queue_(new ::sycl::queue(queue)) {}

    ~stream_impl_t() override = default;

    status_t set_queue(::sycl::queue queue) {
        queue_.reset(new ::sycl::queue(queue));
        return status::success;
    }

    ::sycl::queue *queue() { return queue_.get(); }

    status_t wait() {
        queue()->wait_and_throw();
        return status::success;
    }

    status_t copy(impl::stream_t *stream, const memory_storage_t &src,
            const memory_storage_t &dst, size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep,
            xpu::stream_profiler_t *stream_profiler = nullptr);

    status_t fill(const memory_storage_t &dst, uint8_t pattern, size_t size,
            const xpu::event_t &deps, xpu::event_t &out_dep,
            xpu::stream_profiler_t *stream_profiler = nullptr);

    status_t barrier();

    const xpu::sycl::context_t &sycl_ctx() const;
    xpu::sycl::context_t &sycl_ctx();

    xpu::context_t &ctx();
    const xpu::context_t &ctx() const;

    ::sycl::event get_output_event();

    void register_deps(::sycl::handler &cgh) const;

    static status_t init_flags(unsigned *flags, ::sycl::queue &queue) {
        *flags = queue.is_in_order() ? stream_flags::in_order
                                     : stream_flags::out_of_order;

#ifdef DNNL_EXPERIMENTAL_PROFILING
        if (queue.has_property<::sycl::property::queue::enable_profiling>())
            *flags |= stream_flags::profiling;
#endif
        return status::success;
    }

    template <::sycl::access_mode mode>
    ::sycl::accessor<uint8_t, 1, mode> get_dummy_accessor(
            ::sycl::handler &cgh) {
        return dummy_buffer_.get_access<mode>(cgh);
    }

private:
    std::unique_ptr<::sycl::queue> queue_;

    mutable utils::thread_local_storage_t<xpu::sycl::context_t> ctx_;

    // XXX: this is a temporary solution to make sycl_memory_arg_t
    // default constructible.
    xpu::sycl::buffer_u8_t dummy_buffer_ = xpu::sycl::buffer_u8_t(1);
};

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
