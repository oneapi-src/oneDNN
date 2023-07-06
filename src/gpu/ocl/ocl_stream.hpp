/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef GPU_OCL_OCL_STREAM_HPP
#define GPU_OCL_OCL_STREAM_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/thread_local_storage.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/mdapi_utils.hpp"
#include "gpu/ocl/ocl_context.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/stream_profiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ocl_stream_t : public compute::compute_stream_t {
    static status_t create_stream(
            stream_t **stream, engine_t *engine, unsigned flags) {

        std::unique_ptr<ocl_stream_t> ocl_stream(
                new ocl_stream_t(engine, flags));
        if (!ocl_stream) return status::out_of_memory;

        status_t status = ocl_stream->init();
        if (status != status::success) return status;

        *stream = ocl_stream.release();
        return status::success;
    }

    static status_t create_stream(
            stream_t **stream, engine_t *engine, cl_command_queue queue) {
        unsigned flags;
        status_t status = ocl_stream_t::init_flags(&flags, queue);
        if (status != status::success) return status;

        std::unique_ptr<ocl_stream_t> ocl_stream(
                new ocl_stream_t(engine, flags, queue));
        if (!ocl_stream) return status::out_of_memory;

        status = ocl_stream->init();
        if (status != status::success) return status;

        *stream = ocl_stream.release();
        return status::success;
    }

    status_t wait() override {
        OCL_CHECK(clFinish(queue_));
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

    cl_command_queue queue() const { return queue_; }

    const mdapi_helper_t &mdapi_helper() const { return *mdapi_helper_; }

    status_t copy(const memory_storage_t &src, const memory_storage_t &dst,
            size_t size, const compute::event_t &deps,
            compute::event_t &out_dep) override;

    status_t fill(const memory_storage_t &dst, uint8_t pattern, size_t size,
            const compute::event_t &deps, compute::event_t &out_dep) override;

    ~ocl_stream_t() override {
        if (queue_) { clReleaseCommandQueue(queue_); }
    }

    const ocl_context_t &ocl_ctx() const {
        static ocl_context_t empty_ctx {};
        return ctx_.get(empty_ctx);
    }
    ocl_context_t &ocl_ctx() {
        const ocl_context_t &ctx
                = const_cast<const ocl_stream_t *>(this)->ocl_ctx();
        return *const_cast<ocl_context_t *>(&ctx);
    }
    gpu::compute::context_t &ctx() override { return ocl_ctx(); }
    const gpu::compute::context_t &ctx() const override { return ocl_ctx(); }

    const ocl_wrapper_t<cl_event> &get_output_event() const {
        auto &deps = ocl_event_t::from(ctx().get_deps());
        assert(deps.size() == 1);
        return deps[0];
    }

    const compute::stream_profiler_t &profiler() const override {
        return *profiler_;
    }
    compute::stream_profiler_t &profiler() override { return *profiler_; }

private:
    ocl_stream_t(engine_t *engine, unsigned flags)
        : compute_stream_t(engine, flags), queue_(nullptr) {}
    ocl_stream_t(engine_t *engine, unsigned flags, cl_command_queue queue)
        : compute_stream_t(engine, flags), queue_(queue) {}
    status_t init();

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

    cl_command_queue create_queue(
            cl_context ctx, cl_device_id dev, cl_int *err) const;

    cl_command_queue queue_;
    std::unique_ptr<mdapi_helper_t> mdapi_helper_;
    std::unique_ptr<compute::stream_profiler_t> profiler_;
    mutable utils::thread_local_storage_t<ocl_context_t> ctx_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
