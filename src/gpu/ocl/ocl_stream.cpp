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

#include <cstring>

#include <CL/cl.h>

#include "gpu/ocl/ocl_stream.hpp"

#include "common/verbose.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/stream_profiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ocl_stream_t::init() {
    if (is_profiling_enabled()) {
        profiler_ = utils::make_unique<ocl_stream_profiler_t>(this);
        mdapi_helper_ = utils::make_unique<mdapi_helper_t>();
    }
    // Restore queue on successful exit, otherwise queue may be released
    // without retain
    cl_command_queue queue = queue_;
    queue_ = nullptr;

    assert(engine()->kind() == engine_kind::gpu);

    ocl_gpu_engine_t *ocl_engine
            = utils::downcast<ocl_gpu_engine_t *>(engine());

    // Create queue if it is not set
    if (!queue) {
        cl_int err;
        queue = create_queue(ocl_engine->context(), ocl_engine->device(), &err);
        OCL_CHECK(err);
    } else {
        // Check that queue is compatible with the engine
        cl_context ocl_ctx;
        OCL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
                sizeof(cl_context), &ocl_ctx, nullptr));

        cl_device_id ocl_dev;
        OCL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
                sizeof(cl_device_id), &ocl_dev, nullptr));

        if (ocl_engine->device() != ocl_dev || ocl_engine->context() != ocl_ctx)
            return status::invalid_arguments;

        OCL_CHECK(clRetainCommandQueue(queue));
    }
    queue_ = queue;

    if (is_profiling_enabled()) {
        cl_command_queue_properties props;
        OCL_CHECK(clGetCommandQueueInfo(
                queue_, CL_QUEUE_PROPERTIES, sizeof(props), &props, nullptr));
        bool is_out_of_order
                = (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;
        if (is_out_of_order) {
            VERROR(common, ocl,
                    "OpenCL kernel profiling is not "
                    "supported with out-of-order queues");
            return status::invalid_arguments;
        }
    }

    return status::success;
}

cl_command_queue ocl_stream_t::create_queue(
        cl_context ctx, cl_device_id dev, cl_int *err) const {
    if (is_profiling_enabled() && mdapi_helper_) {
        auto ret = mdapi_helper_->create_queue(ctx, dev, err);
        if (ret) return ret;
    }

    const bool is_out_of_order = (flags() & stream_flags::out_of_order);

    cl_command_queue_properties queue_props {};
    if (is_profiling_enabled()) queue_props |= CL_QUEUE_PROFILING_ENABLE;
    if (is_out_of_order) queue_props |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, queue_props, 0};
    return clCreateCommandQueueWithProperties(ctx, dev, props, err);
#else
    return clCreateCommandQueue(ctx, dev, queue_props, err);
#endif
}

void ocl_stream_t::before_exec_hook() {
    if (is_profiling_enabled()) profiler_->start_profiling();
}

void ocl_stream_t::after_exec_hook() {
    ocl_ctx().set_deps(ocl_event_t());
    if (is_profiling_enabled()) profiler_->stop_profiling();
}

status_t ocl_stream_t::copy(const memory_storage_t &src,
        const memory_storage_t &dst, size_t size, const compute::event_t &deps,
        compute::event_t &out_dep) {

    if (size == 0) return status::success;

    std::vector<cl_event> events = [&] {
        if (flags() & stream_flags::out_of_order) {
            const auto &event_wrappers = ocl_event_t::from(deps).events;
            return std::vector<cl_event>(
                    event_wrappers.begin(), event_wrappers.end());
        }
        return std::vector<cl_event> {};
    }();
    cl_uint num_events = (cl_uint)events.size();
    const cl_event *events_ptr = events.data();

    ocl_wrapper_t<cl_event> out_event;
    bool need_out_event
            = is_profiling_enabled() || flags() & stream_flags::out_of_order;
    cl_event *out_event_ptr = need_out_event ? &out_event.unwrap() : nullptr;

    if (dst.engine()->kind() == engine_kind::gpu
            && src.engine() == dst.engine()) {
        auto *ocl_src
                = utils::downcast<const ocl_memory_storage_base_t *>(&src);
        auto *ocl_dst
                = utils::downcast<const ocl_memory_storage_base_t *>(&dst);

        if (ocl_src->memory_kind() == ocl_dst->memory_kind()) {
            if (ocl_src->memory_kind() == memory_kind::usm
                    && ocl_dst->memory_kind() == memory_kind::usm) {
                const auto *ocl_usm_src
                        = utils::downcast<const ocl_usm_memory_storage_t *>(
                                ocl_src);
                const auto *ocl_usm_dst
                        = utils::downcast<const ocl_usm_memory_storage_t *>(
                                ocl_dst);
                CHECK(usm::memcpy(this, ocl_usm_dst->usm_ptr(),
                        ocl_usm_src->usm_ptr(), size, num_events, events_ptr,
                        out_event_ptr));
            }
            if (ocl_src->memory_kind() == memory_kind::buffer
                    && ocl_dst->memory_kind() == memory_kind::buffer) {
                const auto *ocl_buffer_src
                        = utils::downcast<const ocl_buffer_memory_storage_t *>(
                                ocl_src);
                const auto *ocl_buffer_dst
                        = utils::downcast<const ocl_buffer_memory_storage_t *>(
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

        const auto *ocl_dst
                = utils::downcast<const ocl_memory_storage_base_t *>(&dst);
        bool usm_dst = ocl_dst->memory_kind() == memory_kind::usm;

        if (usm_dst) {
            const auto *ocl_usm_dst
                    = utils::downcast<const ocl_usm_memory_storage_t *>(
                            ocl_dst);
            CHECK(usm::memcpy(this, ocl_usm_dst->usm_ptr(), src_ptr, size,
                    num_events, events_ptr, out_event_ptr));
        } else {
            const auto *ocl_buffer_dst
                    = utils::downcast<const ocl_buffer_memory_storage_t *>(
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

        const auto *ocl_src
                = utils::downcast<const ocl_memory_storage_base_t *>(&src);
        bool usm_src = ocl_src->memory_kind() == memory_kind::usm;

        if (usm_src) {
            const auto *ocl_usm_src
                    = utils::downcast<const ocl_usm_memory_storage_t *>(
                            ocl_src);
            CHECK(usm::memcpy(this, dst_ptr, ocl_usm_src->usm_ptr(), size,
                    num_events, events_ptr, out_event_ptr));
        } else {
            const auto *ocl_buffer_src
                    = utils::downcast<const ocl_buffer_memory_storage_t *>(
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

        CHECK(src.map_data(&src_mapped_ptr, this, size));
        CHECK(dst.map_data(&dst_mapped_ptr, this, size));

        std::memcpy(static_cast<void *>(dst_mapped_ptr),
                static_cast<const void *>(src_mapped_ptr), size);

        CHECK(src.unmap_data(src_mapped_ptr, this));
        CHECK(dst.unmap_data(dst_mapped_ptr, this));

        // Short-circuit event management due to calls to wait
        return status::success;
    }

    if (is_profiling_enabled()) {
        auto ocl_event = utils::make_unique<ocl_event_t>(
                std::vector<ocl_wrapper_t<cl_event>> {out_event});
        profiler_->register_event(std::move(ocl_event));
    }

    if (flags() & stream_flags::out_of_order)
        ocl_event_t::from(out_dep).events = {std::move(out_event)};

    return status::success;
}

status_t ocl_stream_t::fill(const memory_storage_t &dst, uint8_t pattern,
        size_t size, const compute::event_t &deps, compute::event_t &out_dep) {
    using namespace dnnl::impl::utils;

    const auto *ocl_dst = downcast<const ocl_memory_storage_base_t *>(&dst);

    std::vector<cl_event> events = [&] {
        if (flags() & stream_flags::out_of_order) {
            const auto &event_wrappers = ocl_event_t::from(deps).events;
            return std::vector<cl_event>(
                    event_wrappers.begin(), event_wrappers.end());
        }
        return std::vector<cl_event> {};
    }();
    cl_uint num_events = (cl_uint)events.size();
    const cl_event *events_ptr = events.data();

    ocl_wrapper_t<cl_event> out_event;
    bool need_out_event
            = is_profiling_enabled() || flags() & stream_flags::out_of_order;
    cl_event *out_event_ptr = need_out_event ? &out_event.unwrap() : nullptr;

    if (ocl_dst->memory_kind() == memory_kind::usm) {
        const auto *ocl_usm_dst
                = downcast<const ocl_usm_memory_storage_t *>(ocl_dst);
        CHECK(usm::fill(this, ocl_usm_dst->usm_ptr(), &pattern, sizeof(pattern),
                size, num_events, events_ptr, out_event_ptr));
    } else {
        const auto *ocl_buffer_dst
                = downcast<const ocl_buffer_memory_storage_t *>(ocl_dst);
        cl_int err = clEnqueueFillBuffer(queue(), ocl_buffer_dst->mem_object(),
                &pattern, sizeof(uint8_t), dst.offset(), size, num_events,
                events_ptr, out_event_ptr);
        OCL_CHECK(err);
    }

    if (is_profiling_enabled()) {
        auto ocl_event = utils::make_unique<ocl_event_t>(
                std::vector<ocl_wrapper_t<cl_event>> {out_event});
        profiler_->register_event(std::move(ocl_event));
    }

    if (flags() & stream_flags::out_of_order)
        ocl_event_t::from(out_dep).events = {std::move(out_event)};

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
