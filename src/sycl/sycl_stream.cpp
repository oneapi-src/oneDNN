/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

namespace dnnl {
namespace impl {
namespace sycl {

class fixed_device_selector_t : public cl::sycl::device_selector {
public:
    fixed_device_selector_t(const cl::sycl::device &device)
        : fixed_device_(device) {
        if (fixed_device_.is_cpu() || fixed_device_.is_gpu())
            ocl_fixed_device_
                    = ocl::ocl_utils::make_ocl_wrapper(fixed_device_.get());
    }

    virtual int operator()(const cl::sycl::device &device) const override {
        // Never choose devices other than fixed_device_
        // XXX: there is no reliable way to compare SYCL devices so try heuristics:
        // 1) For CPU and GPU SYCL devices compare their OpenCL devices
        // 2) For Host device assume it's always unique
        if (ocl_fixed_device_) {
            if (!device.is_cpu() && !device.is_gpu()) return -1;

            auto ocl_dev = ocl::ocl_utils::make_ocl_wrapper(device.get());
            return (ocl_dev == ocl_fixed_device_ ? 1 : -1);
        }
        assert(fixed_device_.is_host());
        return device.is_host() ? 1 : -1;
    }

private:
    cl::sycl::device fixed_device_;
    cl_device_id ocl_fixed_device_ = nullptr;
};

status_t sycl_stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    // If queue_ is not set then construct it
    if (!queue_) {
        auto &sycl_engine = *utils::downcast<sycl_engine_base_t *>(engine());
        auto &sycl_ctx = sycl_engine.context();
        auto &sycl_dev = sycl_engine.device();

        // FIXME: workaround for Intel(R) oneAPI DPC++ Compiler
        // Intel(R) oneAPI DPC++ Compiler does not work with multiple queues so
        // try to reuse the service stream from the engine.
        // That way all MKL-DNN streams constructed without interop API are
        // mapped to the same SYCL queue.
        // If service stream is NULL then the current stream will be service
        // so construct it from scratch.
        auto *service_stream = utils::downcast<sycl_stream_t *>(
                sycl_engine.service_stream());
        if (!service_stream) {
            queue_.reset(new cl::sycl::queue(
                    sycl_ctx, fixed_device_selector_t(sycl_dev)));

        } else {
            // XXX: multiple queues support has some issues, so always re-use
            // the same queue from the service stream.
            queue_.reset(new cl::sycl::queue(service_stream->queue()));
        }
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

        if (sycl_dev.is_cpu() || sycl_dev.is_gpu()) {
            auto ocl_queue = ocl::ocl_utils::make_ocl_wrapper(queue_->get());
            cl_context ocl_ctx;
            err = clGetCommandQueueInfo(ocl_queue, CL_QUEUE_CONTEXT,
                    sizeof(cl_context), &ocl_ctx, nullptr);
            status = ocl::ocl_utils::convert_to_dnnl(err);
            if (status != status::success) return status;

            cl_device_id ocl_dev;
            err = clGetCommandQueueInfo(ocl_queue, CL_QUEUE_DEVICE,
                    sizeof(cl_device_id), &ocl_dev, nullptr);
            status = ocl::ocl_utils::convert_to_dnnl(err);
            if (status != status::success) return status;

            auto *sycl_engine = utils::downcast<sycl_engine_base_t *>(engine());

            auto sycl_ocl_dev = ocl::ocl_utils::make_ocl_wrapper(
                    sycl_engine->device().get());
            auto sycl_ocl_ctx = ocl::ocl_utils::make_ocl_wrapper(
                    sycl_engine->context().get());
            if (sycl_ocl_dev != ocl_dev || sycl_ocl_ctx != ocl_ctx)
                return status::invalid_arguments;
        }
    }

    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
