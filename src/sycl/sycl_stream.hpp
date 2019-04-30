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

#ifndef SYCL_STREAM_HPP
#define SYCL_STREAM_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "ocl/cl_stream.hpp"
#include "ocl/ocl_utils.hpp"
#include "sycl/sycl_gpu_engine.hpp"
#include "sycl/sycl_memory_storage.hpp"
#include "sycl/sycl_stream_cpu_thunk.hpp"

#if MKLDNN_CPU_BACKEND == MKLDNN_BACKEND_SYCL
#include "sycl/sycl_stream_submit_cpu_primitive.hpp"
#endif

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <algorithm>
#include <map>
#include <memory>
#include <utility>

namespace mkldnn {
namespace impl {
namespace sycl {

struct sycl_stream_t : public ocl::cl_stream_t {
    static status_t create_stream(
            stream_t **stream, engine_t *engine, unsigned generic_flags) {
        unsigned flags;
        status_t status = sycl_stream_t::init_flags(&flags, generic_flags);
        if (status != status::success)
            return status;

        std::unique_ptr<sycl_stream_t> sycl_stream(
                new sycl_stream_t(engine, flags));
        if (!sycl_stream)
            return status::out_of_memory;

        status = sycl_stream->init();
        if (status != status::success)
            return status;
        *stream = sycl_stream.release();
        return status::success;
    }

    static status_t create_stream(
            stream_t **stream, engine_t *engine, cl::sycl::queue &queue) {
        unsigned flags;
        status_t status = sycl_stream_t::init_flags(&flags, queue);
        if (status != status::success)
            return status;

        std::unique_ptr<sycl_stream_t> sycl_stream(
                new sycl_stream_t(engine, flags, queue));

        status = sycl_stream->init();
        if (status != status::success)
            return status;

        *stream = sycl_stream.release();
        return status::success;
    }

    virtual status_t wait() override {
        queue_->wait();
        return status::success;
    }

    cl::sycl::queue &queue() { return *queue_; }

    virtual status_t enqueue_primitive(
            const primitive_t *prim, const exec_ctx_t &exec_ctx) override {
        auto execute_func = [&]() {
            status_t status = status::success;
            if (engine()->kind() == engine_kind::cpu) {
#if MKLDNN_CPU_BACKEND == MKLDNN_BACKEND_SYCL
                queue_->submit([&](cl::sycl::handler &cgh) {
                    submit_cpu_primitive(this, prim, exec_ctx, cgh);
                });
#else
                assert(!"not expected");
                return status::runtime_error;
#endif
            } else if (engine()->kind() == engine_kind::gpu) {
                status = prim->execute(exec_ctx);
            } else {
                assert(!"not expected");
            }
            return status;
        };
        status_t status = execute_func();
        // Emulate in-order behavior
        if (flags() & stream_flags::in_order)
            wait();
        return status;
    }

private:
    sycl_stream_t(engine_t *engine, unsigned flags)
        : cl_stream_t(engine, flags) {}
    sycl_stream_t(engine_t *engine, unsigned flags, cl::sycl::queue &queue)
        : cl_stream_t(engine, flags), queue_(new cl::sycl::queue(queue)) {}

    status_t init();

    static status_t init_flags(unsigned *flags, unsigned generic_flags) {
        *flags = 0;
        if (generic_flags & stream_flags::default_order)
            *flags |= stream_flags::out_of_order;
        else
            *flags |= generic_flags;
        return status::success;
    }

    static status_t init_flags(unsigned *flags, cl::sycl::queue &queue) {
        // SYCL queue is always out-of-order
        *flags = stream_flags::out_of_order;
        return status::success;
    }

private:
    std::unique_ptr<cl::sycl::queue> queue_;
};

} // namespace sycl
} // namespace impl
} // namespace mkldnn

#endif
