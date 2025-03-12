/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2024 Codeplay Software

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

#ifndef GPU_GENERIC_SYCL_SPECIALIZATION_CONSTANTS_HPP
#define GPU_GENERIC_SYCL_SPECIALIZATION_CONSTANTS_HPP

#include <sycl/sycl.hpp>

#include "xpu/sycl/types.hpp"

namespace dnnl::impl::gpu::generic::sycl {
namespace detail {
namespace matmul {
static constexpr ::sycl::specialization_id<xpu::sycl::md_t_spec_const_pod>
        md_t_spec_const_id;
}
} // namespace detail
} // namespace dnnl::impl::gpu::generic::sycl

#endif
