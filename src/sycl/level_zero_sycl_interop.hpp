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

#ifndef SYCL_LEVEL_ZERO_SYCL_INTEROP_HPP
#define SYCL_LEVEL_ZERO_SYCL_INTEROP_HPP

#if defined(DNNL_WITH_LEVEL_ZERO)

#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>

#if defined(DNNL_SYCL_DPCPP) && defined(__SYCL_COMPILER_VERSION) \
        && (__SYCL_COMPILER_VERSION > 20200511)
#include <CL/sycl/backend/Intel_level0.hpp>
#define USE_DIRECT_LEVEL_ZERO_SYCL_INTEROP
#endif

namespace dnnl {
namespace impl {
namespace sycl {

inline cl::sycl::program make_program(
        const cl::sycl::context &ctx, ze_module_handle_t ze_module_handle) {
#ifdef USE_DIRECT_LEVEL_ZERO_SYCL_INTEROP
    return cl::sycl::level0::make<cl::sycl::program>(ctx, ze_module_handle);
#else
    return cl::sycl::program(
            ctx, reinterpret_cast<cl_program>(ze_module_handle));
#endif
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#ifdef USE_DIRECT_LEVEL_ZERO_SYCL_INTEROP
#undef USE_DIRECT_LEVEL_ZERO_SYCL_INTEROP
#endif

#endif
#endif
