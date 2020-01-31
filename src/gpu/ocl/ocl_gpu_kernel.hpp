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

#ifndef GPU_OCL_OCL_GPU_KERNEL_HPP
#define GPU_OCL_OCL_GPU_KERNEL_HPP

#include <assert.h>
#include <string>
#include <CL/cl.h>

#include "compute/compute.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class ocl_gpu_kernel_t : public compute::kernel_impl_t {
public:
    ocl_gpu_kernel_t(cl_kernel ocl_kernel) : ocl_kernel_(ocl_kernel) {}
    virtual ~ocl_gpu_kernel_t() override;

    cl_kernel ocl_kernel() const { return ocl_kernel_; }

    status_t parallel_for(stream_t &stream, const compute::nd_range_t &range,
            const compute::kernel_arg_list_t &arg_list) const override;

private:
    cl_kernel ocl_kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_OCL_OCL_GPU_KERNEL_HPP
