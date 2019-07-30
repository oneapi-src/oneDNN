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
#include "compute/compute_stream.hpp"
#include "ocl/ocl_utils.hpp"
#include "sycl/sycl_gpu_engine.hpp"
#include "sycl/sycl_memory_storage.hpp"
#include "sycl/sycl_stream_cpu_thunk.hpp"

#if MKLDNN_CPU_RUNTIME == MKLDNN_RUNTIME_SYCL
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

struct sycl_stream_t : public compute::compute_stream_t {
    ~sycl_stream_t() override { wait(); }

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
        queue_->wait_and_throw();
        return status::success;
    }

    cl::sycl::queue &queue() { return *queue_; }

    virtual status_t enqueue_primitive(
            const primitive_t *prim, const exec_ctx_t &exec_ctx) override {
        auto execute_func = [&]() {
            status_t status = status::success;
            if (engine()->kind() == engine_kind::cpu) {
#if MKLDNN_CPU_RUNTIME == MKLDNN_RUNTIME_SYCL
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

    virtual status_t copy(const memory_storage_t &src,
            const memory_storage_t &dst, size_t size) const override {
        if (size == 0)
            return status::success;

        assert(utils::one_of(src.engine()->backend_kind(), backend_kind::sycl,
                backend_kind::native));
        assert(utils::one_of(dst.engine()->backend_kind(), backend_kind::sycl,
                backend_kind::native));

        if (src.engine()->backend_kind() == backend_kind::sycl
                && dst.engine()->backend_kind() == backend_kind::sycl) {
            auto *src_sycl_storage
                    = utils::downcast<const sycl::sycl_memory_storage_t *>(
                            src.impl());
            auto *dst_sycl_storage
                    = utils::downcast<const sycl::sycl_memory_storage_t *>(
                            dst.impl());
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
            auto sycl_buf_src
                    = mkldnn::get_sycl_buffer(src_sycl_storage->vptr());
            auto sycl_buf_dst
                    = mkldnn::get_sycl_buffer(dst_sycl_storage->vptr());
#else
            auto &sycl_buf_src = src_sycl_storage->buffer();
            auto &sycl_buf_dst = dst_sycl_storage->buffer();
#endif
            size_t src_size = sycl_buf_src.get_size();
            size_t dst_size = sycl_buf_dst.get_size();

            assert(src_size == dst_size);
            MAYBE_UNUSED(src_size);
            MAYBE_UNUSED(dst_size);

            // FIXME: Intel SYCL fails to compile the SYCL kernel for GPU due to
            // unresolved references to mkldnn_impl_sycl_cpu_thunk so switch to
            // blocking map/unmap.
#if 0
        auto *sycl_stream = utils::downcast<sycl::sycl_stream_t *>(stream());
        auto &sycl_queue = sycl_stream->queue();

        sycl_queue.submit([&](cl::sycl::handler &cgh) {
            auto dst_acc
                    = sycl_buf_dst.get_access<cl::sycl::access::mode::write>(
                            cgh);
            auto src_acc
                    = sycl_buf_src.get_access<cl::sycl::access::mode::read>(
                            cgh);
            cgh.parallel_for<mkldnn_copy_tag>(cl::sycl::range<1>(src_size),
                    [=](cl::sycl::id<1> i) { dst_acc[i] = src_acc[i]; });
        });
#else
            void *src_mapped_ptr;
            void *dst_mapped_ptr;

            src.map_data(&src_mapped_ptr);
            dst.map_data(&dst_mapped_ptr);

            utils::array_copy(static_cast<uint8_t *>(dst_mapped_ptr),
                    static_cast<const uint8_t *>(src_mapped_ptr), size);

            src.unmap_data(src_mapped_ptr);
            dst.unmap_data(dst_mapped_ptr);
#endif
        } else if (src.engine()->kind() == engine_kind::cpu
                && src.engine()->backend_kind() == backend_kind::native) {
            assert(dst.engine()->backend_kind() == backend_kind::sycl);

            void *src_ptr;
            src.get_data_handle(&src_ptr);
            auto *src_ptr_u8 = static_cast<uint8_t *>(src_ptr);

            auto &sycl_dst
                    = *utils::downcast<const sycl::sycl_memory_storage_t *>(
                            dst.impl());
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
            auto sycl_buf = mkldnn::get_sycl_buffer(sycl_dst.vptr());
            copy_to_buffer(src_ptr_u8, sycl_buf, size);
#else
            auto &sycl_buf = sycl_dst.buffer();
            copy_to_buffer(src_ptr_u8, sycl_buf, size);
#endif
        } else if (dst.engine()->kind() == engine_kind::cpu
                && dst.engine()->backend_kind() == backend_kind::native) {
            assert(src.engine()->backend_kind() == backend_kind::sycl);

            void *dst_ptr;
            dst.get_data_handle(&dst_ptr);

            auto &sycl_src
                    = *utils::downcast<const sycl::sycl_memory_storage_t *>(
                            src.impl());
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
            auto sycl_buf = mkldnn::get_sycl_buffer(sycl_src.vptr());
            copy_from_buffer(sycl_buf, dst_ptr, size);
#else
            auto &sycl_buf = sycl_src.buffer();
            copy_from_buffer(sycl_buf, dst_ptr, size);
#endif
        } else {
            assert(!"Not expected");
            return status::runtime_error;
        }
        return status::success;
    }

private:
    sycl_stream_t(engine_t *engine, unsigned flags)
        : compute::compute_stream_t(engine, flags) {}
    sycl_stream_t(engine_t *engine, unsigned flags, cl::sycl::queue &queue)
        : compute::compute_stream_t(engine, flags)
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
};

} // namespace sycl
} // namespace impl
} // namespace mkldnn

#endif
