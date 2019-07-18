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

#include "ocl/ocl_executor.hpp"
#include "ocl/ocl_utils.hpp"

#include "ocl/ocl_stream.hpp"

namespace mkldnn {
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
    if (!args_ok)
        return status::unimplemented;

    ocl_engine_t *ocl_engine = utils::downcast<ocl_engine_t *>(engine());

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
    cl_stream_t::set_cl_executor(new ocl_executor_t(this));

    return status::success;
}

} // namespace ocl
} // namespace impl
} // namespace mkldnn
