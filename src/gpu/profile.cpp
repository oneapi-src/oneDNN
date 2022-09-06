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

#include <vector>

#include "gpu/profile.hpp"

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/ocl/profile.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "sycl/profile.hpp"
#endif

using namespace dnnl::impl;

namespace dnnl {
namespace impl {
namespace gpu {

static setting_t<bool> profile {false};

bool is_profiling_enabled() {
    return profile.get();
}

status_t get_profile_info_impl(uint64_t &nsec, double &freq, int _mode,
        const std::unordered_map<uint64_t, profile_entry_t> &stamp2entry) {
    auto mode = static_cast<profile_mode_t>(_mode);
    switch (mode) {
        case profile_mode_t::sum:
            nsec = 0;
            freq = 0;
            for (auto &kv : stamp2entry) {
                auto &e = kv.second;
                nsec += e.nsec;
                freq += e.freq / e.kernel_count;
            }
            freq /= stamp2entry.size();
            break;
        case profile_mode_t::min:
            nsec = std::numeric_limits<uint64_t>::max();
            freq = 0;
            for (auto &kv : stamp2entry) {
                auto &e = kv.second;
                if (e.nsec < nsec) {
                    nsec = e.nsec;
                    freq = e.freq / e.kernel_count;
                }
            }
            break;
        default: assert(!"Unexpected mode");
    }
    return status::success;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl

extern "C" status_t DNNL_API dnnl_impl_gpu_set_profiling(int flag) {
    dnnl::impl::gpu::profile.set((bool)flag);
    return status::success;
}

extern "C" status_t DNNL_API dnnl_impl_gpu_reset_profiling() {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    return dnnl::impl::gpu::ocl::reset_profiling();
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    return dnnl::impl::sycl::reset_profiling();
#endif
    return status::unimplemented;
}

extern "C" status_t DNNL_API dnnl_impl_gpu_get_profile_info(
        uint64_t &nsec, double &freq, int mode) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    return dnnl::impl::gpu::ocl::get_profile_info(nsec, freq, mode);
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    return dnnl::impl::sycl::get_profile_info(nsec, freq, mode);
#endif
    return status::unimplemented;
}
