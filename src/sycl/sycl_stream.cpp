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

#include "sycl/sycl_stream.hpp"

#include "ocl/ocl_utils.hpp"
#include "sycl/sycl_engine.hpp"

#include <map>
#include <memory>
#include <CL/cl.h>

namespace mkldnn {
namespace impl {
namespace sycl {

status_t sycl_stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    // If queue_ is not set then construct it
    if (!queue_) {
        auto &sycl_engine = *utils::downcast<sycl_engine_base_t *>(engine());
        auto &sycl_ctx = sycl_engine.context();
        auto &sycl_dev = sycl_engine.device();

        auto ocl_dev = ocl::ocl_utils::make_ocl_wrapper(sycl_dev.get());

        // TODO: enable this code and remove all the workarounds below after
        // the related SYCL bugs are fixed
#if 0
        // cl::sycl::queue does not have ctor taking ctx and dev so switch to
        // OpenCL interop API
        auto ocl_dev = ocl::ocl_utils::make_ocl_wrapper(sycl_dev.get());
        auto ocl_ctx = ocl::ocl_utils::make_ocl_wrapper(sycl_ctx.get());
        cl_command_queue ocl_queue = clCreateCommandQueueWithProperties(
                ocl_ctx, ocl_dev, nullptr, nullptr);
        queue_.reset(new cl::sycl::queue(ocl_queue, sycl_ctx));
        clReleaseCommandQueue(ocl_queue);
#else
        // FIXME: workaround for Intel SYCL
        // Intel SYCL does not work with multiple queues so try to reuse
        // the service stream from the engine.
        // That way all MKL-DNN streams constructed without interop API are
        // mapped to the same SYCL queue.
        // If service stream is NULL then the current stream will be service
        // so construct it from scratch.
        auto *service_stream = utils::downcast<sycl_stream_t *>(
                sycl_engine.service_stream());
        if (!service_stream) {
#ifdef MKLDNN_SYCL_COMPUTECPP
            // FIXME: workaround for ComputeCpp SYCL
            // ComputeCpp SYCL works incorrectly with OpenCL interop API for
            // queues so let's create a queue based on the SYCL context and
            // the engine kind only and check that the device is the same.
            if (sycl_engine.kind() == engine_kind::cpu) {
                queue_.reset(new cl::sycl::queue(
                        sycl_ctx, cl::sycl::cpu_selector {}));
            } else {
                assert(sycl_engine.kind() == engine_kind::gpu);
                queue_.reset(new cl::sycl::queue(
                        sycl_ctx, cl::sycl::gpu_selector {}));
            }
            auto queue_ocl_dev = ocl::ocl_utils::make_ocl_wrapper(
                    queue_->get_device().get());
            if (queue_ocl_dev.get() != ocl_dev.get()) {
                assert(!"Not expected");
                return status::runtime_error;
            }
#else
            auto ocl_ctx = ocl::ocl_utils::make_ocl_wrapper(sycl_ctx.get());
            cl_int err;
            cl_command_queue ocl_queue = clCreateCommandQueueWithProperties(
                    ocl_ctx, ocl_dev, nullptr, &err);
            assert(err == CL_SUCCESS);
            queue_.reset(new cl::sycl::queue(ocl_queue, sycl_ctx));
            err = clReleaseCommandQueue(ocl_queue);
            assert(err == CL_SUCCESS);
#endif
        } else {
            queue_.reset(new cl::sycl::queue(service_stream->queue()));
        }
#endif
    } else {
        cl_int err;
        status_t status;

        // Validate that the queue is compatible with the engine
        auto sycl_dev = queue_->get_device();
        bool args_ok = true
                && IMPLICATION(
                        engine()->kind() == engine_kind::gpu, sycl_dev.is_gpu())
                && IMPLICATION(engine()->kind() == engine_kind::cpu,
                        (sycl_dev.is_cpu() || sycl_dev.is_host()));
        if (!args_ok) return status::invalid_arguments;

        auto ocl_queue = ocl::ocl_utils::make_ocl_wrapper(queue_->get());
        cl_context ocl_ctx;
        err = clGetCommandQueueInfo(ocl_queue, CL_QUEUE_CONTEXT,
                sizeof(cl_context), &ocl_ctx, nullptr);
        status = ocl::ocl_utils::convert_to_mkldnn(err);
        if (status != status::success) return status;

        cl_device_id ocl_dev;
        err = clGetCommandQueueInfo(ocl_queue, CL_QUEUE_DEVICE,
                sizeof(cl_device_id), &ocl_dev, nullptr);
        status = ocl::ocl_utils::convert_to_mkldnn(err);
        if (status != status::success) return status;

        auto *sycl_engine = utils::downcast<sycl_engine_base_t *>(engine());

        auto sycl_ocl_dev
                = ocl::ocl_utils::make_ocl_wrapper(sycl_engine->device().get());
        auto sycl_ocl_ctx = ocl::ocl_utils::make_ocl_wrapper(
                sycl_engine->context().get());
        if (sycl_ocl_dev != ocl_dev || sycl_ocl_ctx != ocl_ctx)
            return status::invalid_arguments;
    }

    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace mkldnn
