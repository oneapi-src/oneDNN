/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#ifndef GPU_OVERRIDE_UTILS_HPP
#define GPU_OVERRIDE_UTILS_HPP

#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

inline int dev_getenv(const char *name, int default_value) {
#ifdef DNNL_DEV_MODE
    return getenv_int(name, default_value);
#else
    return default_value;
#endif
}

inline bool dev_getenv(const char *s, bool def) {
    return dev_getenv(s, def ? 1 : 0) == 1;
}

inline std::string dev_getenv(const char *s, const std::string &def) {
#ifdef DNNL_DEV_MODE
    char buf[1024];
    int ret = getenv(s, buf, sizeof(buf));
    if (ret > 0) return buf;
    return def;
#else
    return def;
#endif
}

// Input is a comma separate list containing gpu_arch and optionally eu_count.
inline compute::gpu_arch_t dev_getenv(const char *s, compute::gpu_arch_t arch,
        int *eu_count = nullptr, int *max_wg_size = nullptr) {
#ifdef DNNL_DEV_MODE
    char buf[1024];
    int ret = getenv(s, buf, sizeof(buf));
    if (ret > 0) {
        char *arch_str = buf, *eu_str = nullptr;
        for (int i = 0; i < ret; i++) {
            if (buf[i] == ',') {
                buf[i] = 0;
                if (i < ret - 1) { eu_str = &buf[i + 1]; }
                break;
            }
        }
        arch = compute::str2gpu_arch(arch_str);
        if (eu_count && eu_str) { *eu_count = atoi(eu_str); }
        if (max_wg_size) {
            // Assume maximum wg size is basically the number of threads
            // available in a subslice with simd_size 16
            const int max_eus_per_wg
                    = compute::device_info_t::max_eus_per_wg(arch);
            const int simd_size = 16;
            const int thr_per_eu = utils::rnd_down_pow2(
                    compute::device_info_t::threads_per_eu(arch));
            *max_wg_size = simd_size * max_eus_per_wg * thr_per_eu;
        }
    }
    return arch;
#else
    return arch;
#endif
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
