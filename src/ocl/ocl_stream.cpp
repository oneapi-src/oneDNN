/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <CL/cl.h>

#include "ocl/ocl_memory_storage.hpp"
#include "ocl/ocl_utils.hpp"

#include "ocl/ocl_stream.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

status_t ocl_stream_t::init() {
    // Restore queue on successful exit, otherwise queue may be released
    // without retain
    cl_command_queue queue = queue_;
    queue_ = nullptr;

    assert(engine()->kind() == engine_kind::gpu);

    // Out-of-order is not supported
    bool args_ok = (flags() & stream_flags::out_of_order) == 0;
    if (!args_ok) return status::unimplemented;

    ocl_gpu_engine_t *ocl_engine
            = utils::downcast<ocl_gpu_engine_t *>(engine());

    // Create queue if it is not set
    if (!queue) {
        cl_int err;
#ifdef CL_VERSION_2_0
        queue = clCreateCommandQueueWithProperties(
                ocl_engine->context(), ocl_engine->device(), nullptr, &err);
#else
        queue = clCreateCommandQueue(
                ocl_engine->context(), ocl_engine->device(), 0, &err);
#endif
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
    return status::success;
}

status_t ocl_stream_t::copy(
        const memory_storage_t &src, const memory_storage_t &dst, size_t size) {

    if (size == 0) return status::success;

    if (src.engine()->kind() == engine_kind::cpu
            && is_native_runtime(src.engine()->runtime_kind())) {
        assert(dst.engine()->kind() == engine_kind::gpu);

        void *src_ptr = nullptr;
        src.get_data_handle(&src_ptr);

        auto &ocl_dst = *utils::downcast<const ocl_memory_storage_t *>(&dst);
        cl_mem ocl_mem = ocl_dst.mem_object();
        cl_int err = clEnqueueWriteBuffer(queue(), ocl_mem, CL_TRUE, 0, size,
                src_ptr, 0, nullptr, nullptr);
        OCL_CHECK(err);
    } else if (dst.engine()->kind() == engine_kind::cpu
            && is_native_runtime(dst.engine()->runtime_kind())) {
        assert(src.engine()->kind() == engine_kind::gpu);

        void *dst_ptr = nullptr;
        dst.get_data_handle(&dst_ptr);

        auto &ocl_src = *utils::downcast<const ocl_memory_storage_t *>(&src);
        cl_mem ocl_mem = ocl_src.mem_object();
        cl_int err = clEnqueueReadBuffer(queue(), ocl_mem, CL_TRUE, 0, size,
                dst_ptr, 0, nullptr, nullptr);
        OCL_CHECK(err);
    } else {
        wait();

        // Use map/unmap
        void *src_mapped_ptr;
        void *dst_mapped_ptr;

        CHECK(src.map_data(&src_mapped_ptr));
        CHECK(dst.map_data(&dst_mapped_ptr));

        utils::array_copy(static_cast<uint8_t *>(dst_mapped_ptr),
                static_cast<const uint8_t *>(src_mapped_ptr), size);

        CHECK(src.unmap_data(src_mapped_ptr));
        CHECK(dst.unmap_data(dst_mapped_ptr));
    }
    return status::success;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl
