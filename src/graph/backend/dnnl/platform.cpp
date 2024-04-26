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

bool get_dtype_support_status(engine_kind_t eng, data_type_t dtype, dir_t dir) {

    bool is_supported = false;
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    switch (dtype) {
        case dnnl_bf16: {
            // bf16 is supported on AVX512-CORE+
            is_supported = is_gpu(eng)
                    || (is_cpu(eng) && has_cpu_data_type_support(dnnl_bf16)
                            && IMPLICATION(!(dir & dir_t::FLAG_INF),
                                    has_cpu_training_support(dnnl_bf16)));
            break;
        }
        case dnnl_f16: {
            is_supported = is_gpu(eng)
                    || (is_cpu(eng) && has_cpu_data_type_support(dnnl_f16)
                            && IMPLICATION(!(dir & dir_t::FLAG_INF),
                                    has_cpu_training_support(dnnl_f16)));
            break;
        }
        case dnnl_f8_e5m2: {
            is_supported = is_gpu(eng)
                    || (is_cpu(eng) && has_cpu_data_type_support(dnnl_f8_e5m2)
                            && IMPLICATION(!(dir & dir_t::FLAG_INF),
                                    has_cpu_training_support(dnnl_f8_e5m2)));
            break;
        }
        case dnnl_f8_e4m3: {
            is_supported = is_gpu(eng)
                    || (is_cpu(eng) && has_cpu_data_type_support(dnnl_f8_e4m3)
                            && IMPLICATION(!(dir & dir_t::FLAG_FWD),
                                    has_cpu_training_support(dnnl_f8_e4m3)));
            break;
        }
        default: break;
    }
#else
    switch (dtype) {
        case dnnl_bf16: {
            is_supported = is_gpu(eng);
            break;
        }
        case dnnl_f16: {
            // TODO(zhitao): f16 for backward on gpu?
            is_supported = is_gpu(eng);
            break;
        }
        case dnnl_f8_e5m2: {
            is_supported = is_gpu(eng);
            break;
        }
        case dnnl_f8_e4m3: {
            is_supported = is_gpu(eng);
            break;
        }
        default: break;
    }
#endif

    return is_supported;
}

bool has_cpu_data_type_support(data_type_t data_type) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    switch (data_type) {
        case data_type::bf16:
#if DNNL_X64
            using namespace dnnl::impl::cpu::x64;
            return mayiuse(avx512_core) || mayiuse(avx2_vnni_2);
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
            return mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2);
#elif DNNL_AARCH64_USE_ACL
            return arm_compute::CPUInfo::get().has_fp16();
#else
            return false;
#endif
        case data_type::f8_e5m2:
        case data_type::f8_e4m3:
#if DNNL_X64
            return mayiuse(avx512_core_fp16);
#else
            return false;
#endif
        default: return true;
    }
#else
    return false;
#endif
}

bool has_cpu_training_support(data_type_t data_type) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    switch (data_type) {
        case data_type::bf16:
#if DNNL_X64
            using namespace dnnl::impl::cpu::x64;
            return mayiuse(avx512_core);
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
            return mayiuse(avx512_core_fp16);
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
