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

#include "graph/backend/dnnl/platform.hpp"

#if DNNL_X64
#include "cpu/x64/cpu_isa_traits.hpp"
#elif DNNL_AARCH64
#if DNNL_AARCH64_USE_ACL
// For checking if fp16 isa is supported on the platform
#include "arm_compute/core/CPP/CPPTypes.h"
// For setting the number of threads for ACL
#include "src/common/cpuinfo/CpuInfo.h"
#endif
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace platform {

bool get_dtype_support_status(engine_kind_t eng, data_type_t dtype) {

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    const bool has_bf16_support = is_gpu(eng)
            || (is_cpu(eng) && has_cpu_data_type_support(dnnl_bf16));
    const bool has_f16_support = is_gpu(eng)
            || (is_cpu(eng) && has_cpu_data_type_support(dnnl_f16));
#else
    const bool has_bf16_support = is_gpu(eng);
    const bool has_f16_support = is_gpu(eng);
#endif

    bool is_unsupported = false;
    switch (dtype) {
        case dnnl_bf16: is_unsupported = !has_bf16_support; break;
        case dnnl_f16: is_unsupported = !has_f16_support; break;
        default: break;
    }
    if (is_unsupported) return false;
    return true;
}

bool has_cpu_data_type_support(data_type_t data_type) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    switch (data_type) {
        case data_type::bf16:
#if DNNL_X64
            using namespace dnnl::impl::cpu::x64;
            return mayiuse(avx512_core, /*soft=*/true)
                    || mayiuse(avx2_vnni_2, /*soft=*/true);
#elif DNNL_PPC64
#if defined(USE_CBLAS) && defined(BLAS_HAS_SBGEMM) && defined(__MMA__)
            return true;
#endif
#elif DNNL_AARCH64_USE_ACL
            return arm_compute::CPUInfo::get().has_bf16();
#else
            return false;
#endif
        case data_type::f16:
#if DNNL_X64
            return mayiuse(avx512_core_fp16, /*soft=*/true)
                    || mayiuse(avx2_vnni_2, /*soft=*/true);
#elif DNNL_AARCH64_USE_ACL
            return arm_compute::CPUInfo::get().has_fp16();
#else
            return false;
#endif
        default: return true;
    }
#else
    return false;
#endif
}

} // namespace platform
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
