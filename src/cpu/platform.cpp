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

#include "cpu/cpu_isa_traits.hpp"

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace platform {

bool has_data_type_support(data_type_t data_type) {
    switch (data_type) {
        case data_type::bf16: return mayiuse(avx512_core);
        case data_type::f16: return false;
        default: return true;
    }
}

unsigned get_per_core_cache_size(int level) {
    if (cpu.getDataCacheLevels() == 0) {
        // If Xbyak is not able to fetch the cache topology, default to 32KB
        // of L1, 512KB of L2 and 1MB of L3 per core.
        switch (level) {
            case (1): return 32U * 1024;
            case (2): return 512U * 1024;
            case (3): return 1024U * 1024;
            default: return 0;
        }
    }

    if (level > 0 && (unsigned)level <= cpu.getDataCacheLevels()) {
        unsigned l = level - 1;
        return cpu.getDataCacheSize(l) / cpu.getCoresSharingDataCache(l);
    } else
        return 0;
}

unsigned get_num_cores() {
    return cpu.getNumCores(Xbyak::util::CoreLevel);
}

int get_vector_register_size() {
    if (mayiuse(avx512_common)) return cpu_isa_traits<avx512_common>::vlen;
    if (mayiuse(avx)) return cpu_isa_traits<avx>::vlen;
    if (mayiuse(sse41)) return cpu_isa_traits<sse41>::vlen;
    return 0;
}

} // namespace platform
} // namespace cpu
} // namespace impl
} // namespace dnnl
