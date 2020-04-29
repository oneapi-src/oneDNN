/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "sycl/sycl_utils.hpp"

#if defined(DNNL_SYCL_DPCPP) && defined(__INTEL_CLANG_COMPILER) \
        && (__SYCL_COMPILER_VERSION >= 20200402)
// Only Intel DPC++ supports `pi::getPreferredBE()` so far
// XXX: remove once OSS compiler start supporting L0
// XXX: find a better way than using `sycl::detail` namespace
#define USE_PI_GET_PREFERRED_BE
#include <CL/sycl/detail/pi.hpp>
#endif

namespace dnnl {
namespace impl {
namespace sycl {

backend_t get_sycl_gpu_backend() {
#if defined(USE_PI_GET_PREFERRED_BE)
    switch (cl::sycl::detail::pi::getPreferredBE()) {
        case cl::sycl::detail::pi::SYCL_BE_PI_OPENCL: return backend_t::opencl;
#ifdef DNNL_WITH_LEVEL_ZERO
        case cl::sycl::detail::pi::SYCL_BE_PI_LEVEL0: return backend_t::level0;
#endif
        // Ignore preferred backend and use OpenCL in this case.
        default: return backend_t::opencl;
    }
#else
    return backend_t::opencl;
#endif
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
