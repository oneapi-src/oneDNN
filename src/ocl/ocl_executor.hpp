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

#ifndef OCL_EXECUTOR_HPP
#define OCL_EXECUTOR_HPP

#include "ocl/cl_executor.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

struct ocl_stream_t;

// Implementation of cl_executor_t for OpenCL
struct ocl_executor_t : public cl_executor_t {
    ocl_executor_t(ocl_stream_t *stream);
    virtual status_t parallel_for(
            const cl_nd_range_t &range, const ocl_kernel_t &kernel) override;
    virtual status_t copy(const memory_storage_t &src,
            const memory_storage_t &dst, size_t size) override;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
