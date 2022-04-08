/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_SYCL_SYCL_GPU_KERNEL_HPP
#define GPU_SYCL_SYCL_GPU_KERNEL_HPP

#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct sycl_gpu_kernel_t : public compute::kernel_impl_t {
    using kernel_bundle_e_t
            = ::sycl::kernel_bundle<::sycl::bundle_state::executable>;

    sycl_gpu_kernel_t(const kernel_bundle_e_t &kernel_bundle)
        : kernel_bundle_(utils::make_unique<kernel_bundle_e_t>(kernel_bundle)) {
    }

    status_t parallel_for(stream_t &stream,
            const std::function<void(void *)> &cgf) const override;

private:
    std::unique_ptr<kernel_bundle_e_t> kernel_bundle_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
