/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_SYCL_COMPAT_HPP
#define GPU_INTEL_SYCL_COMPAT_HPP

#include "xpu/sycl/compat.hpp"

#include "gpu/intel/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_engine_base_t;

}
} // namespace impl
} // namespace dnnl

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

namespace compat {

status_t make_kernel(std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const impl::sycl::sycl_engine_base_t *sycl_engine,
        const xpu::binary_t &binary, const char *kernel_name);

uint64_t init_extensions(const ::sycl::device &dev);

} // namespace compat
} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
