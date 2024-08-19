/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_SYCL_SYCL_INTEROP_GPU_KERNEL_HPP
#define GPU_INTEL_SYCL_SYCL_INTEROP_GPU_KERNEL_HPP

#include <string>

#include "gpu/intel/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

class sycl_interop_gpu_kernel_t : public gpu::intel::compute::kernel_impl_t {
public:
    sycl_interop_gpu_kernel_t(std::unique_ptr<::sycl::kernel> &&sycl_kernel)
        : sycl_kernel_(std::move(sycl_kernel)) {}

    ::sycl::kernel sycl_kernel() const { return *sycl_kernel_; }

    status_t parallel_for(impl::stream_t &stream,
            const gpu::intel::compute::nd_range_t &range,
            const gpu::intel::compute::kernel_arg_list_t &arg_list,
            const xpu::event_t &deps, xpu::event_t &out_dep) override;

    const std::vector<gpu::intel::compute::scalar_type_t> &
    arg_types() const override {
        return arg_types_;
    }

    status_t dump() const override;
    std::string name() const override {
        return sycl_kernel_->get_info<::sycl::info::kernel::function_name>();
    }

private:
    std::unique_ptr<::sycl::kernel> sycl_kernel_;
    std::vector<gpu::intel::compute::scalar_type_t> arg_types_;
};

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // SYCL_SYCL_INTEROP_GPU_KERNEL_HPP
