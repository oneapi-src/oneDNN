/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GPU_INTEL_JIT_UTILS_TYPE_BRIDGE_HPP
#define GPU_INTEL_JIT_UTILS_TYPE_BRIDGE_HPP

#include "common/c_types_map.hpp"
#include "gemmstone/dsl/type.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "ngen.hpp"

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
        case compute::gpu_arch_t::xe_lp: return ngen::HW::XeLP;
        case compute::gpu_arch_t::xe_hp: return ngen::HW::XeHP;
        case compute::gpu_arch_t::xe_hpg: return ngen::HW::XeHPG;
        case compute::gpu_arch_t::xe_hpc: return ngen::HW::XeHPC;
        case compute::gpu_arch_t::xe2: return ngen::HW::Xe2;
        case compute::gpu_arch_t::xe3: return ngen::HW::Xe3;
        case compute::gpu_arch_t::xe3p: return ngen::HW::Xe3p;
        case compute::gpu_arch_t::unknown: return ngen::HW::Unknown;
    }
    return ngen::HW::Unknown;
}

inline compute::gpu_arch_t convert_ngen_arch_to_dnnl(ngen::HW gpu_arch) {
    switch (gpu_arch) {
        case ngen::HW::XeLP: return compute::gpu_arch_t::xe_lp;
        case ngen::HW::XeHP: return compute::gpu_arch_t::xe_hp;
        case ngen::HW::XeHPG: return compute::gpu_arch_t::xe_hpg;
        case ngen::HW::XeHPC: return compute::gpu_arch_t::xe_hpc;
        case ngen::HW::Xe2: return compute::gpu_arch_t::xe2;
        case ngen::HW::Xe3: return compute::gpu_arch_t::xe3;
        case ngen::HW::Xe3p: return compute::gpu_arch_t::xe3p;
        case ngen::HW::Gen9:
        case ngen::HW::Gen10:
        case ngen::HW::Gen11:
            // Gen9, Gen10, Gen11 are not supported anymore. Included
            // here instead of default to emit warnings at this spot
            // when new architectures are added.
        case ngen::HW::Unknown: return compute::gpu_arch_t::unknown;
    }
    return compute::gpu_arch_t::unknown;
}

// gemmstone::dsl::type_t and dnnl_data_type_t convertors.
inline data_type_t to_dnnl(const gemmstone::dsl::type_t &type) {
    gpu_assert(type.elems() == 1) << type;
    gpu_assert(!type.is_ptr() == 1) << type;
    if (type.is_f4_e2m1()) return data_type::f4_e2m1;
    if (type.is_bf8()) return data_type::f8_e5m2;
    if (type.is_hf8()) return data_type::f8_e4m3;
    if (type.is_bf16()) return data_type::bf16;
    if (type.is_f16()) return data_type::f16;
    if (type.is_tf32()) return data_type::tf32;
    if (type.is_f32()) return data_type::f32;
    if (type.is_f64()) return data_type::f64;
    if (type.is_s32()) return data_type::s32;
    if (type.is_s8()) return data_type::s8;
    if (type.is_u8()) return data_type::u8;
    gpu_error_not_expected();
    return data_type::undef;
}

inline gemmstone::dsl::type_t to_ir(const data_type_t &dt) {
    if (dt == data_type::undef) return gemmstone::dsl::type_t();
    switch ((int)dt) {
#define CASE(x) \
    case data_type::x: return gemmstone::dsl::type_t::x();
        CASE(f4_e2m1);
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(bf16);
        CASE(f16);
        CASE(tf32);
        CASE(f32);
        CASE(f64);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(s4);
        CASE(u4);
        CASE(u2);
#undef CASE
        default: gpu_error_not_expected();
    }
    return {};
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
