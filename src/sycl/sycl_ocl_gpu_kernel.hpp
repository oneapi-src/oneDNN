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

#ifndef SYCL_OCL_GPU_KERNEL_HPP
#define SYCL_OCL_GPU_KERNEL_HPP

#include <CL/cl.h>

#include "gpu/compute/compute.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_ocl_gpu_kernel_t : public gpu::compute::kernel_impl_t {
public:
    sycl_ocl_gpu_kernel_t(cl_kernel ocl_kernel) : ocl_kernel_(ocl_kernel) {}
    virtual ~sycl_ocl_gpu_kernel_t() override;

    status_t parallel_for(stream_t &stream,
            const gpu::compute::nd_range_t &range,
            const gpu::compute::kernel_arg_list_t &arg_list) const override;

private:
    cl_kernel ocl_kernel_;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // SYCL_OCL_GPU_KERNEL_HPP
