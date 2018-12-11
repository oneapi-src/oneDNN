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

#ifndef SYCL_EXECUTOR_HPP
#define SYCL_EXECUTOR_HPP

#include "ocl/cl_executor.hpp"

namespace mkldnn {
namespace impl {
namespace sycl {

struct sycl_stream_t;

struct sycl_executor_t : public ocl::cl_executor_t {
    sycl_executor_t(sycl_stream_t *stream);
    virtual status_t parallel_for(const ocl::cl_nd_range_t &range,
            const ocl::ocl_kernel_t &kernel) override;
    virtual status_t copy(const memory_storage_t &src,
            const memory_storage_t &dst, size_t size) override;
};

} // namespace sycl
} // namespace impl
} // namespace mkldnn

#endif
