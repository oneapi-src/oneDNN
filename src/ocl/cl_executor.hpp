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

#ifndef CL_EXECUTOR_HPP
#define CL_EXECUTOR_HPP

#include "common/c_types_map.hpp"
#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

// Executor provides OpenCL-like functionality whose implementation
// is specific for a given stream.
struct cl_executor_t {
    cl_executor_t(stream_t *stream) : stream_(stream) {}
    virtual ~cl_executor_t() = default;

    stream_t *stream() { return stream_; }

    virtual status_t parallel_for(
            const cl_nd_range_t &range, const ocl_kernel_t &kernel)
            = 0;

    virtual status_t copy(const memory_storage_t &src,
            const memory_storage_t &dst, size_t size)
            = 0;

private:
    stream_t *stream_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
