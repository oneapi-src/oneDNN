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

#include <assert.h>
#include <string>
#include <CL/cl.h>

#include "gpu/ocl/ocl_gpu_kernel.hpp"

#include "common/utils.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

ocl_gpu_kernel_t::~ocl_gpu_kernel_t() {
    if (ocl_kernel_) OCL_CHECK_V(clReleaseKernel(ocl_kernel_));
}

status_t ocl_gpu_kernel_t::parallel_for(stream_t &stream,
        const compute::nd_range_t &range,
        const compute::kernel_arg_list_t &arg_list) const {

    auto *ocl_stream = utils::downcast<ocl_stream_t *>(&stream);
    cl_command_queue queue = ocl_stream->queue();

    assert(ocl_kernel_ && "kernel is NULL");

    for (int i = 0; i < arg_list.nargs(); ++i) {
        auto &arg = arg_list.get(i);
        cl_int set_err;
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
            set_err = clSetKernelArg(ocl_kernel_, i, sizeof(cl_mem), &ocl_mem);
        } else {
            char type[100];
            OCL_CHECK(clGetKernelArgInfo(ocl_kernel_, i,
                    CL_KERNEL_ARG_TYPE_NAME, sizeof(type), type, nullptr));
            auto str_type = std::string(type);
            size_t type_size = arg.size();
            auto value = arg.value();
            compute::kernel_arg_t temp_arg;

            //downcast if necessary
            if (str_type.compare("half") == 0 && sizeof(float) == arg.size()) {
                float v = *((float *)arg.value());
                temp_arg.set_value((float16_t)v);
                type_size = temp_arg.size();
                value = temp_arg.value();

            } else if (str_type.compare("uchar") == 0
                    && sizeof(uint8_t) != arg.size()) {
                if (sizeof(int64_t) == arg.size()) {
                    int64_t v = *((int64_t *)arg.value());
                    temp_arg.set_value((uint8_t)v);
                } else if (sizeof(int32_t) == arg.size()) {
                    int32_t v = *((int32_t *)arg.value());
                    temp_arg.set_value((uint8_t)v);
                } else {
                    assert(!"not expected");
                }
                type_size = temp_arg.size();
                value = temp_arg.value();
            } else if (str_type.compare("char") == 0
                    && sizeof(char) != arg.size()) {
                if (sizeof(int64_t) == arg.size()) {
                    int64_t v = *((int64_t *)arg.value());
                    temp_arg.set_value((char)v);
                } else if (sizeof(int32_t) == arg.size()) {
                    int32_t v = *((int32_t *)arg.value());
                    temp_arg.set_value((char)v);
                } else {
                    assert(!"not expected");
                }
                type_size = temp_arg.size();
                value = temp_arg.value();
            }

            set_err = clSetKernelArg(ocl_kernel_, i, type_size, value);
        }
        status_t status = convert_to_dnnl(set_err);
        if (status != status::success) return status;
    }

    cl_uint ndims = static_cast<cl_uint>(range.ndims());
    if (range.is_zero()) { return status::success; }
    cl_int err = clEnqueueNDRangeKernel(queue, ocl_kernel_, ndims, nullptr,
            range.global_range(), range.local_range(), 0, nullptr, nullptr);
    status_t status = convert_to_dnnl(err);
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
