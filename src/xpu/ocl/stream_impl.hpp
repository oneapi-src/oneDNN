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

#ifndef XPU_OCL_STREAM_IMPL_HPP
#define XPU_OCL_STREAM_IMPL_HPP

#include "common/stream_impl.hpp"
#include "common/thread_local_storage.hpp"
#include "common/utils.hpp"

#include "xpu/context.hpp"
#include "xpu/stream_profiler.hpp"

#include "xpu/ocl/context.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

class stream_impl_t : public impl::stream_impl_t {
public:
    stream_impl_t() = delete;
    stream_impl_t(unsigned flags)
        : impl::stream_impl_t(flags), queue_(nullptr) {}
    stream_impl_t(cl_command_queue queue, unsigned flags)
        : impl::stream_impl_t(flags), queue_(queue) {
        assert(queue_);
    }

    ~stream_impl_t() override {
        if (queue_) { clReleaseCommandQueue(queue_); }
    }

    status_t set_queue(cl_command_queue queue) {
        queue_ = queue;
        return status::success;
    }

    cl_command_queue queue() { return queue_; }

    status_t wait() {
        OCL_CHECK(clFinish(queue()));
        return status::success;
    }

    status_t copy(impl::stream_t *stream, const memory_storage_t &src,
            const memory_storage_t &dst, size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep,
            xpu::stream_profiler_t *stream_profiler = nullptr);

    status_t fill(impl::stream_t *stream, const memory_storage_t &dst,
            uint8_t pattern, size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep,
            xpu::stream_profiler_t *stream_profiler = nullptr);

    status_t barrier();

    const xpu::ocl::context_t &ocl_ctx() const;
    xpu::ocl::context_t &ocl_ctx();
    xpu::context_t &ctx();
    const xpu::context_t &ctx() const;
    const xpu::ocl::wrapper_t<cl_event> &get_output_event() const;

    static status_t init_flags(unsigned *flags, cl_command_queue queue) {
        *flags = 0;
        // Determine if the passed queue is in-order/out-of-order
        cl_command_queue_properties props;
        OCL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES,
                sizeof(cl_command_queue_properties), &props, nullptr));

        *flags |= (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
                ? stream_flags::out_of_order
                : stream_flags::in_order;
#ifdef DNNL_EXPERIMENTAL_PROFILING
        if (props & CL_QUEUE_PROFILING_ENABLE)
            *flags |= stream_flags::profiling;
#endif
        return status::success;
    }

private:
    cl_command_queue queue_;

    mutable utils::thread_local_storage_t<xpu::ocl::context_t> ctx_;
};

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
