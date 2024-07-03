/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_SYCL_GPU_KERNEL_HPP
#define GPU_GENERIC_SYCL_SYCL_GPU_KERNEL_HPP

#include "xpu/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct kernel_t {
    using kernel_bundle_e_t
            = ::sycl::kernel_bundle<::sycl::bundle_state::executable>;

    kernel_t() = default;
    kernel_t(const kernel_bundle_e_t &kernel_bundle)
        : kernel_bundle_(utils::make_unique<kernel_bundle_e_t>(kernel_bundle)) {
    }

    kernel_t(const kernel_t &other) = delete;
    kernel_t &operator=(const kernel_t &other) = delete;

    kernel_t(kernel_t &&other) = default;
    kernel_t &operator=(kernel_t &&other) = default;

    ~kernel_t() = default;

    status_t parallel_for(impl::stream_t &stream,
            const std::function<void(::sycl::handler &)> &cgf) const;

private:
    std::unique_ptr<kernel_bundle_e_t> kernel_bundle_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
