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

#ifndef SYCL_STREAM_HPP
#define SYCL_STREAM_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "sycl/sycl_gpu_engine.hpp"
#include "sycl/sycl_memory_storage.hpp"
#include "sycl/sycl_stream_cpu_thunk.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "sycl/sycl_stream_submit_cpu_primitive.hpp"
#endif

#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <CL/cl.h>
#include <CL/sycl.hpp>

namespace dnnl {
namespace impl {
namespace sycl {

struct sycl_stream_t : public gpu::compute::compute_stream_t {
    static status_t create_stream(
            stream_t **stream, engine_t *engine, unsigned generic_flags) {
        unsigned flags;
        status_t status = sycl_stream_t::init_flags(&flags, generic_flags);
        if (status != status::success) return status;

        std::unique_ptr<sycl_stream_t> sycl_stream(
                new sycl_stream_t(engine, flags));
        if (!sycl_stream) return status::out_of_memory;

        status = sycl_stream->init();
        if (status != status::success) return status;
        *stream = sycl_stream.release();
        return status::success;
    }

    static status_t create_stream(
            stream_t **stream, engine_t *engine, cl::sycl::queue &queue) {
        unsigned flags;
        status_t status = sycl_stream_t::init_flags(&flags, queue);
        if (status != status::success) return status;

        std::unique_ptr<sycl_stream_t> sycl_stream(
                new sycl_stream_t(engine, flags, queue));

        status = sycl_stream->init();
        if (status != status::success) return status;

        *stream = sycl_stream.release();
        return status::success;
    }

    virtual status_t wait() override {
        queue_->wait_and_throw();
        return status::success;
    }

    cl::sycl::queue &queue() { return *queue_; }

    virtual status_t enqueue_primitive(const primitive_iface_t *prim_iface,
            exec_ctx_t &exec_ctx) override {
        auto execute_func = [&]() {
            status_t status = status::success;
            if (engine()->kind() == engine_kind::cpu) {

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
                auto event = queue_->submit([&](cl::sycl::handler &cgh) {
#ifdef DNNL_SYCL_DPCPP
                    cgh.depends_on(deps_);
#endif
                    submit_cpu_primitive(this, prim_iface, exec_ctx, cgh);
                });
                deps_ = {event};
                // XXX: wait() to workaround a hang happening in DPC++ RT.
                this->wait();
#else
                assert(!"not expected");
                return status::runtime_error;
#endif
            } else if (engine()->kind() == engine_kind::gpu) {
                status = prim_iface->execute(exec_ctx);
            } else {
                assert(!"not expected");
            }
            return status;
        };
        status_t status = execute_func();
        // Emulate in-order behavior
        if (flags() & stream_flags::in_order) wait();
        return status;
    }

    virtual status_t copy(const memory_storage_t &src,
            const memory_storage_t &dst, size_t size) override {
        if (size == 0) return status::success;

        // TODO: add src and dst sizes check

        void *src_mapped_ptr;
        void *dst_mapped_ptr;

        // When handling the copy by mapping/unmapping, we do not
        // consume/produce events, so we have to synchronize the stream
        // TODO: enqueue the copy
        this->wait();
        CHECK(src.map_data(&src_mapped_ptr, nullptr));
        CHECK(dst.map_data(&dst_mapped_ptr, nullptr));

        utils::array_copy(static_cast<uint8_t *>(dst_mapped_ptr),
                static_cast<const uint8_t *>(src_mapped_ptr), size);

        CHECK(src.unmap_data(src_mapped_ptr, nullptr));
        CHECK(dst.unmap_data(dst_mapped_ptr, nullptr));
        this->wait();

        return status::success;
    }

    virtual status_t fill(const memory_storage_t &dst, const void *pattern,
            size_t pattern_size, size_t size) override {
        void *mapped_ptr;

        // When handling the filling by mapping/unmapping, we do not
        // consume/produce events, so we have to synchronize the stream
        // TODO: enqueue the fill (memset)
        this->wait();
        CHECK(dst.map_data(&mapped_ptr, this));

        assert(size % pattern_size == 0);
        for (size_t i = 0; i < size / pattern_size; ++i) {
            memcpy(static_cast<uint8_t *>(mapped_ptr) + i * pattern_size,
                    pattern, pattern_size);
        }

        CHECK(dst.unmap_data(mapped_ptr, this));
        this->wait();
        return status::success;
    }

    std::vector<cl::sycl::event> &get_deps() { return deps_; }
    void set_deps(const std::vector<cl::sycl::event> &deps) { deps_ = deps; }

private:
    sycl_stream_t(engine_t *engine, unsigned flags)
        : gpu::compute::compute_stream_t(engine, flags, nullptr) {}
    sycl_stream_t(engine_t *engine, unsigned flags, cl::sycl::queue &queue)
        : gpu::compute::compute_stream_t(engine, flags, nullptr)
        , queue_(new cl::sycl::queue(queue)) {}

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
    // XXX: This is a temporary solution, ideally events should be a part of
    // execution context.
    std::vector<cl::sycl::event> deps_;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
