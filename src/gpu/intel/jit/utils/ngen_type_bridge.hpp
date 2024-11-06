/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_UTILS_NGEN_TYPE_BRIDGE_HPP
#define GPU_INTEL_JIT_UTILS_NGEN_TYPE_BRIDGE_HPP

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/jit/ngen/ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

inline ngen::DataType convert_dnnl_type_to_ngen(data_type_t dt) {
    using namespace ngen;

    DataType dt_out = DataType::invalid;

    switch (dt) {
        case data_type::f16: dt_out = DataType::hf; break;
        case data_type::bf16: dt_out = DataType::bf; break;
        case data_type::f32: dt_out = DataType::f; break;
        case data_type::s32: dt_out = DataType::d; break;
        case data_type::s8: dt_out = DataType::b; break;
        case data_type::u8: dt_out = DataType::ub; break;
        case data_type::f8_e5m2: dt_out = DataType::bf8; break;
        case data_type::f8_e4m3: dt_out = DataType::hf8; break;
        default: assert(!"Unknown datatype");
    }

    return dt_out;
}

inline data_type_t convert_ngen_type_to_dnnl(ngen::DataType dt) {
    using namespace ngen;

    data_type_t dt_out = data_type::undef;

    switch (dt) {
        case DataType::hf: dt_out = data_type::f16; break;
        case DataType::bf: dt_out = data_type::bf16; break;
        case DataType::f: dt_out = data_type::f32; break;
        case DataType::d: dt_out = data_type::s32; break;
        case DataType::b: dt_out = data_type::s8; break;
        case DataType::ub: dt_out = data_type::u8; break;
        case DataType::bf8: dt_out = data_type::f8_e5m2; break;
        case DataType::hf8: dt_out = data_type::f8_e4m3; break;
        default: assert(!"Unknown datatype");
    }

    return dt_out;
}

inline ngen::HW convert_dnnl_arch_to_ngen(compute::gpu_arch_t gpu_arch) {
    switch (gpu_arch) {
        case compute::gpu_arch_t::gen9: return ngen::HW::Gen9;
        case compute::gpu_arch_t::gen11: return ngen::HW::Gen11;
        case compute::gpu_arch_t::xe_lp: return ngen::HW::XeLP;
        case compute::gpu_arch_t::xe_hp: return ngen::HW::XeHP;
        case compute::gpu_arch_t::xe_hpg: return ngen::HW::XeHPG;
        case compute::gpu_arch_t::xe_hpc: return ngen::HW::XeHPC;
        case compute::gpu_arch_t::xe2: return ngen::HW::Xe2;
        case compute::gpu_arch_t::xe3: return ngen::HW::Xe3;
        case compute::gpu_arch_t::unknown: return ngen::HW::Unknown;
    }
    return ngen::HW::Unknown;
}

inline compute::gpu_arch_t convert_ngen_arch_to_dnnl(ngen::HW gpu_arch) {
    switch (gpu_arch) {
        case ngen::HW::Gen9: return compute::gpu_arch_t::gen9;
        case ngen::HW::Gen11: return compute::gpu_arch_t::gen11;
        case ngen::HW::XeLP: return compute::gpu_arch_t::xe_lp;
        case ngen::HW::XeHP: return compute::gpu_arch_t::xe_hp;
        case ngen::HW::XeHPG: return compute::gpu_arch_t::xe_hpg;
        case ngen::HW::XeHPC: return compute::gpu_arch_t::xe_hpc;
        case ngen::HW::Xe2: return compute::gpu_arch_t::xe2;
        case ngen::HW::Xe3: return compute::gpu_arch_t::xe3;
        case ngen::HW::Gen10:
            // Gen10 is not supported. Included here instead of default so
            // warnings are emitted when new architectures are added.
        case ngen::HW::Unknown: return compute::gpu_arch_t::unknown;
    }
    return compute::gpu_arch_t::unknown;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
