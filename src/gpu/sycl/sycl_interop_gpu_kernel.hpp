/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef GPU_SYCL_SYCL_OCL_GPU_KERNEL_HPP
#define GPU_SYCL_SYCL_OCL_GPU_KERNEL_HPP

#include <assert.h>
#include <string>

#include "gpu/compute/compute.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

class sycl_interop_gpu_kernel_t : public gpu::compute::kernel_impl_t {
public:
    sycl_interop_gpu_kernel_t(const ::sycl::kernel &sycl_kernel,
            const std::vector<gpu::compute::scalar_type_t> &arg_types)
        : sycl_kernel_(new ::sycl::kernel(sycl_kernel))
        , arg_types_(arg_types) {}

    ::sycl::kernel sycl_kernel() const { return *sycl_kernel_; }

    status_t parallel_for(stream_t &stream,
            const gpu::compute::nd_range_t &range,
            const gpu::compute::kernel_arg_list_t &arg_list,
            const gpu::compute::event_t &deps,
            gpu::compute::event_t &out_dep) override;

    const std::vector<gpu::compute::scalar_type_t> &arg_types() const override {
        return arg_types_;
    }

private:
    std::unique_ptr<::sycl::kernel> sycl_kernel_;
    std::vector<gpu::compute::scalar_type_t> arg_types_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // SYCL_SYCL_INTEROP_GPU_KERNEL_HPP
