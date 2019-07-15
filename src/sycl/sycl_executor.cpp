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

#include "sycl/sycl_executor.hpp"

#include "common/utils.hpp"
#include "ocl/ocl_utils.hpp"
#include "sycl/sycl_memory_storage.hpp"
#include "sycl/sycl_stream.hpp"
#include "sycl/sycl_utils.hpp"

class mkldnn_copy_tag;

namespace mkldnn {
namespace impl {
namespace sycl {

inline cl::sycl::nd_range<3> to_sycl_nd_range(const ocl::cl_nd_range_t &range) {
    auto *global_range = range.global_range();
    auto *local_range = range.local_range();

    auto sycl_global_range = cl::sycl::range<3>(
            global_range[0], global_range[1], global_range[2]);
    if (!local_range) {
        assert(!"not expected");
        return cl::sycl::nd_range<3>(
                sycl_global_range, cl::sycl::range<3>(1, 1, 1));
    }

    auto sycl_local_range = cl::sycl::range<3>(
            local_range[0], local_range[1], local_range[2]);
    return cl::sycl::nd_range<3>(sycl_global_range, sycl_local_range);
}

inline cl::sycl::range<3> to_sycl_range(const ocl::cl_nd_range_t &range) {
    auto *global_range = range.global_range();
    return cl::sycl::range<3>(
            global_range[0], global_range[1], global_range[2]);
}

sycl_executor_t::sycl_executor_t(sycl_stream_t *stream)
    : cl_executor_t(stream) {}

static void set_scalar_arg(
        cl::sycl::handler &cgh, int index, size_t size, void *value) {
    switch (size) {
    case sizeof(uint8_t):
        cgh.set_arg(index, *static_cast<uint8_t *>(value));
        break;
    case sizeof(uint16_t):
        cgh.set_arg(index, *static_cast<uint16_t *>(value));
        break;
    case sizeof(uint32_t):
        cgh.set_arg(index, *static_cast<uint32_t *>(value));
        break;
    case sizeof(uint64_t):
        cgh.set_arg(index, *static_cast<uint64_t *>(value));
        break;
    default:
        assert(!"Please add another case");
        throw std::runtime_error("Internal error");
    }
}

status_t sycl_executor_t::parallel_for(const ocl::cl_nd_range_t &range,
        const ocl::ocl_kernel_t &kernel) {
    auto *sycl_stream = utils::downcast<sycl::sycl_stream_t *>(stream());
    auto *sycl_engine
            = utils::downcast<sycl::sycl_gpu_engine_t *>(sycl_stream->engine());
    auto &queue = sycl_stream->queue();
    cl::sycl::kernel sycl_kernel(kernel.kernel(), sycl_engine->context());
    queue.submit([&](cl::sycl::handler &cgh) {
        auto *kernel_args = kernel.args();
        for (size_t i = 0; i < kernel.nargs(); ++i) {
            auto &arg = kernel_args[i];
            if (arg.is_global()) {
                auto *mem_storage
                        = static_cast<const memory_storage_t *>(arg.value());
                if (*mem_storage) {
                    auto *sycl_mem_storage = utils::downcast<
                            const sycl::sycl_memory_storage_t *>(
                            mem_storage->impl());
#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR
                    auto buf = mkldnn::get_sycl_buffer(
                            sycl_mem_storage->vptr());
                    auto acc = buf.get_access<
                            cl::sycl::access::mode::read_write>(cgh);
                    cgh.set_arg(i, acc);
#else
                    auto &sycl_buf = sycl_mem_storage->buffer();
                    cgh.set_arg(i,
                            sycl_buf.get_access<
                                    cl::sycl::access::mode::read_write>(cgh));
#endif
                } else {
                    cgh.set_arg(i, nullptr);
                }
            } else {
                // XXX: workaround for bug in the SYCL library:
                // set_arg() does not work with constant scalars
                set_scalar_arg(
                        cgh, i, arg.size(), const_cast<void *>(arg.value()));
            }
        }
        if (range.local_range()) {
            auto sycl_nd_range = to_sycl_nd_range(range);
            cgh.parallel_for(sycl_nd_range, sycl_kernel);
        } else {
            auto sycl_range = to_sycl_range(range);
            cgh.parallel_for(sycl_range, sycl_kernel);
        }
    });
    return status::success;
}

status_t sycl_executor_t::copy(
        const memory_storage_t &src, const memory_storage_t &dst, size_t size) {
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

        auto &sycl_dst = *utils::downcast<const sycl::sycl_memory_storage_t *>(
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

        auto &sycl_src = *utils::downcast<const sycl::sycl_memory_storage_t *>(
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

} // namespace sycl
} // namespace impl
} // namespace mkldnn
