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

#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"

#include <vector>
#include <CL/sycl.hpp>

// Intel(R) oneAPI DPC++ Compiler uses reversed global work-item IDs starting
// from 10-24-2019.
// ComputeCpp version >= 1.1.6 uses reversed global work-item IDs.
#if defined(DNNL_SYCL_DPCPP) && (__SYCL_COMPILER_VERSION >= 20191024)
#define DNNL_SYCL_REVERSE_RANGE 1
#elif defined(DNNL_SYCL_COMPUTECPP) \
        && (COMPUTECPP_VERSION_MAJOR > 1 \
                || (COMPUTECPP_VERSION_MAJOR == 1 \
                        && (COMPUTECPP_VERSION_MINOR > 1 \
                                || (COMPUTECPP_VERSION_MINOR == 1 \
                                        && COMPUTECPP_VERSION_PATCH >= 6))))
#define DNNL_SYCL_REVERSE_RANGE 1
#else
#define DNNL_SYCL_REVERSE_RANGE 0
#endif

namespace dnnl {
namespace impl {
namespace sycl {

using buffer_u8_t = cl::sycl::buffer<uint8_t, 1>;

inline cl::sycl::range<3> to_sycl_range(const gpu::compute::nd_range_t &range) {
    auto *global_range = range.global_range();
#if DNNL_SYCL_REVERSE_RANGE
    auto sycl_global_range = cl::sycl::range<3>(
            global_range[2], global_range[1], global_range[0]);
#else
    auto sycl_global_range = cl::sycl::range<3>(
            global_range[0], global_range[1], global_range[2]);
#endif
    return sycl_global_range;
}

inline cl::sycl::nd_range<3> to_sycl_nd_range(
        const gpu::compute::nd_range_t &range) {
    auto *global_range = range.global_range();
    auto *local_range = range.local_range();

    auto sycl_global_range = to_sycl_range(range);

    if (!local_range) {
        assert(!"not expected");
        return cl::sycl::nd_range<3>(
                sycl_global_range, cl::sycl::range<3>(1, 1, 1));
    }

#if DNNL_SYCL_REVERSE_RANGE
    auto sycl_local_range = cl::sycl::range<3>(
            local_range[2], local_range[1], local_range[0]);
#else
    auto sycl_local_range = cl::sycl::range<3>(
            local_range[0], local_range[1], local_range[2]);
#endif
    return cl::sycl::nd_range<3>(sycl_global_range, sycl_local_range);
}

// Automatically use run_on_host_intel if it is supported by compiler,
// otherwise fall back to single_task.
template <typename K, typename H, typename F>
inline auto host_task_impl(H &cgh, F f, int)
        -> decltype(cgh.run_on_host_intel(f)) {
    cgh.template run_on_host_intel(f);
}

template <typename K, typename H, typename F>
inline void host_task_impl(H &cgh, F f, long) {
    cgh.template single_task<K>(f);
}

template <typename K, typename H, typename F>
inline void host_task(H &cgh, F f) {
    // Third argument is 0 (int) which prefers the
    // run_on_host_intel option if both are available.
    host_task_impl<K>(cgh, f, 0);
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#undef DNNL_SYCL_REVERSE_RANGE

#endif
