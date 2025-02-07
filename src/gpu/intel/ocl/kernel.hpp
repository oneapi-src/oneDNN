/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_KERNEL_HPP
#define GPU_INTEL_OCL_KERNEL_HPP

#include <string>
#include <CL/cl.h>

#include "gpu/intel/compute/kernel.hpp"
#include "xpu/ocl/utils.hpp"
#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

class kernel_cache_t;

class kernel_t : public compute::kernel_impl_t {
public:
    kernel_t(xpu::ocl::wrapper_t<cl_kernel> &&ocl_kernel,
            const std::vector<gpu::intel::compute::scalar_type_t> &arg_types,
            compute::program_src_t src);
    ~kernel_t() override = default;

    cl_kernel ocl_kernel() const { return ocl_kernel_; }

    status_t get_binary(
            const impl::engine_t *engine, xpu::binary_t &binary) const override;
    status_t get_binary_size(
            const impl::engine_t *engine, size_t *binary_size) const override;

    status_t parallel_for(impl::stream_t &stream,
            const compute::nd_range_t &range,
            const compute::kernel_arg_list_t &arg_list,
            const xpu::event_t &deps, xpu::event_t &out_dep) override;

    const std::vector<gpu::intel::compute::scalar_type_t> &
    arg_types() const override {
        return arg_types_;
    }

    void save_output_events() override { save_events_ = true; }

    status_t dump() const override;
    std::string name() const override;
    const compute::program_src_t &src() const { return src_; }

private:
    xpu::ocl::wrapper_t<cl_kernel> ocl_kernel_;
    std::vector<gpu::intel::compute::scalar_type_t> arg_types_;
    std::shared_ptr<kernel_cache_t> cache_;
    compute::program_src_t src_;
    bool save_events_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_OCL_KERNEL_HPP
