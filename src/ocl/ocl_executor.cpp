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

#include "common/utils.hpp"
#include "ocl/ocl_memory_storage.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

#include "ocl/ocl_executor.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

ocl_executor_t::ocl_executor_t(ocl_stream_t *stream) : cl_executor_t(stream) {}

status_t ocl_executor_t::parallel_for(
        const cl_nd_range_t &range, const ocl_kernel_t &kernel) {
    auto *ocl_stream = utils::downcast<ocl_stream_t *>(stream());
    cl_command_queue queue = ocl_stream->queue();
    cl_kernel ocl_kernel = kernel.kernel();

    assert(ocl_kernel && "kernel is NULL");

    auto *kernel_args = kernel.args();
    for (cl_uint i = 0; i < kernel.nargs(); ++i) {
        auto &arg = kernel_args[i];
        cl_int set_arg_err;
        if (arg.is_global()) {
            auto *mem_storage
                    = static_cast<const memory_storage_t *>(arg.value());
            cl_mem ocl_mem = nullptr;
            if (!mem_storage->is_null()) {
                auto *ocl_mem_storage
                        = utils::downcast<const ocl_memory_storage_t *>(
                                mem_storage);
                ocl_mem = ocl_mem_storage->mem_object();
            }
            set_arg_err
                    = clSetKernelArg(ocl_kernel, i, sizeof(cl_mem), &ocl_mem);
        } else {
            set_arg_err
                    = clSetKernelArg(ocl_kernel, i, arg.size(), arg.value());
        }
        status_t status = ocl_utils::convert_to_mkldnn(set_arg_err);
        if (status != status::success)
            return status;
    }

    cl_uint ndims = static_cast<cl_uint>(range.ndims());
    if (range.is_zero()) {
        return status::success;
    }
    cl_int err = clEnqueueNDRangeKernel(queue, ocl_kernel, ndims, nullptr,
            range.global_range(), range.local_range(), 0, nullptr, nullptr);
    status_t status = ocl_utils::convert_to_mkldnn(err);
    return status;
}

status_t ocl_executor_t::copy(
        const memory_storage_t &src, const memory_storage_t &dst, size_t size) {
    auto *ocl_stream = utils::downcast<ocl_stream_t *>(stream());
    cl_command_queue queue = ocl_stream->queue();

    if (src.engine()->kind() == engine_kind::cpu
            && src.engine()->backend_kind() == backend_kind::native) {
        assert(dst.engine()->kind() == engine_kind::gpu);

        void *src_ptr;
        src.get_data_handle(&src_ptr);

        auto &ocl_dst = *utils::downcast<const ocl_memory_storage_t *>(&dst);
        cl_mem ocl_mem = ocl_dst.mem_object();
        cl_int err = clEnqueueWriteBuffer(
                queue, ocl_mem, CL_TRUE, 0, size, src_ptr, 0, nullptr, nullptr);
        assert(err == CL_SUCCESS);
        UNUSED(err);

    } else {
        assert(src.engine()->kind() == engine_kind::gpu);

        void *dst_ptr;
        dst.get_data_handle(&dst_ptr);

        auto &ocl_src = *utils::downcast<const ocl_memory_storage_t *>(&src);
        cl_mem ocl_mem = ocl_src.mem_object();
        cl_int err = clEnqueueReadBuffer(
                queue, ocl_mem, CL_TRUE, 0, size, dst_ptr, 0, nullptr, nullptr);
        assert(err == CL_SUCCESS);
        UNUSED(err);
    }
    return status::success;
}

} // namespace ocl
} // namespace impl
} // namespace mkldnn
