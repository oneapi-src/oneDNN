/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl_types.h"

// Must be included before emulation.hpp
#include "ngen/ngen.hpp"

#include "common/impl_registration.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/jit/emulation.hpp"
#include "gpu/intel/jit/reduction_injector.hpp"
#include "ngen/ngen_core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

using namespace ngen;

template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::sum_fwd(
        int simd, const ngen::GRF &acc, const ngen::GRF &val) {
    eadd(h, simd, acc, acc, val);
}
template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::max_fwd(
        int simd, const ngen::GRF &acc, const ngen::GRF &val) {
    h.max_(simd, acc, acc, val);
}
template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::min_fwd(
        int simd, const ngen::GRF &acc, const ngen::GRF &val) {
    h.min_(simd, acc, acc, val);
}
template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::mul_fwd(
        int simd, const ngen::GRF &acc, const ngen::GRF &val) {
    emul(h, simd, acc, acc, val);
}

template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::initialize(int simd, const ngen::GRF &reg) {
    switch (alg_) {
        case dnnl_reduction_sum:
        case dnnl_reduction_mean: emov(h, simd, reg, 0.0f); break;
        case dnnl_reduction_max:
            emov(h, simd, reg, nstl::numeric_limits<float>::lowest());
            break;
        case dnnl_reduction_min:
            emov(h, simd, reg, nstl::numeric_limits<float>::max());
            break;
        case dnnl_reduction_mul: emov(h, simd, reg, 1.0f); break;
        default:
            gpu_assert(false) << "unsupported reduction algorithm, " << alg_;
    }
}

template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::eload(
        const ngen::GRFRange &dst, const ngen::GRF &base_src_addr) {
    const int grf_bytes = ngen::GRF::bytes(hw);
    int nregs = dst.getLen();
    bool force_legacy
            = gpu_utils::dev_getenv("jit_reduction_force_legacy_send", false);
    bool use_legacy = force_legacy || hw < ngen::HW::XeHPG;
    const int max_load_size = use_legacy ? 128 : 512;
    gpu_assert(max_load_size % grf_bytes == 0) << "Unexpected load size";
    const int max_load_regs = max_load_size / grf_bytes;

    // Load in chunks
    int reg_start = 0;
    while (reg_start < nregs) {
        int load_regs = nstl::min(max_load_regs, nregs - reg_start);
        // Compute the src address
        ngen::GRF addr = ra.alloc().uq();
        eadd(h, 1, addr, base_src_addr, reg_start * grf_bytes);
        if (use_legacy) {
            // Reduce load_regs according to valid load sizes
            const int oword_per_grf = grf_bytes / 16;
            for (auto load_owords : {8, 4, 2, 1}) {
                if (load_owords / oword_per_grf > load_regs) continue;
                load_regs = load_owords / oword_per_grf;
                break;
            }

            // Do the load
            auto dt = ngen::aligned_block_oword(load_regs * oword_per_grf);
            h.load(1, dst[reg_start], dt, h.A64, addr);
        } else {
            // Reduce load_regs according to valid load sizes
            const int d64_per_grf = grf_bytes / 8;
            for (auto load_d64s : {64, 32, 16, 8, 4, 3, 2, 1}) {
                if (load_d64s / d64_per_grf > load_regs) continue;
                load_regs = load_d64s / d64_per_grf;
                break;
            }

            // Do the load
            ngen::DataSpecLSC lscspec = ngen::CacheSettingsLSC::L1UC_L3WB;
            lscspec |= ngen::block(
                    ngen::DataSizeLSC::D64, load_regs * d64_per_grf);
            h.load.ugm(1, dst[reg_start], lscspec, h.A64, addr);
        }
        reg_start += load_regs;
        ra.release(addr);
    }
}

template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::compute(const ngen::GRF &src_ptr,
        const ngen::GRFRange &acc, dim_t stride, dim_t iters) {
    using namespace alg_kind;
#ifdef DNNL_DEV_MODE
    int pre_regs = ra.get_alloced_regs();
#endif
    assert(src_ptr.getType() == ngen::DataType::uq);

    int dt_size = sizeof(float);
    int reg_size = ngen::GRF::bytes(hw);
    int elems_per_reg = reg_size / dt_size;
    int nregs = acc.getLen();

    int regs_per_inst = std::min(nregs, []() {
        int reg_size = ngen::GRF::bytes(hw);
        compute::gpu_arch_t gpu_arch = convert_ngen_arch_to_dnnl(hw);
        int max_exec_size = compute::device_info_t::max_exec_size(gpu_arch);
        return max_exec_size / reg_size;
    }());

    ngen::GRF load_addr = ra.alloc().uq();
    emov(h, 1, load_addr, src_ptr);

    // Set up GRFs used for loop indices
    ngen::Subregister loop_index = ra.alloc_sub(ngen::DataType::d);
    ngen::GRFRange val = ra.alloc_range(nregs);
    ngen::FlagRegister loop_flag = ra.alloc_flag(true);

    for (int i = 0; i < nregs; i += regs_per_inst) {
        int inst_nregs = std::min(regs_per_inst, nregs - i);
        int simd = inst_nregs * elems_per_reg;
        initialize(simd, acc[i].f());
    }

    // Initialize loop
    ngen::Label loop_start;
    emov(h, 1, loop_index, 0);
    h.mark(loop_start);

    // Load data - coalesce calls when possible
    eload(val, load_addr);

    // Accumulate
    for (int i = 0; i < nregs; i += regs_per_inst) {
        int inst_nregs = std::min(regs_per_inst, nregs - i);
        int simd = inst_nregs * elems_per_reg;
        switch (alg_) {
            case dnnl_reduction_sum:
            case dnnl_reduction_mean:
                sum_fwd(simd, acc[i].f(), val[i].f());
                break;
            case dnnl_reduction_max:
                max_fwd(simd, acc[i].f(), val[i].f());
                break;
            case dnnl_reduction_min:
                min_fwd(simd, acc[i].f(), val[i].f());
                break;
            case dnnl_reduction_mul:
                mul_fwd(simd, acc[i].f(), val[i].f());
                break;
            default: gpu_assert(false) << "unsupported reduction algorithm";
        }
    }

    // Iterate
    eadd(h, 1, loop_index, loop_index, 1);
    h.cmp(1 | h.lt | loop_flag, loop_index, iters);
    eadd(h, 1, load_addr, load_addr, stride * dt_size);
    h.jmpi(1 | loop_flag, loop_start);

    // Release used registers
    ra.release(load_addr);
    ra.release(loop_index);
    ra.release(val);
    ra.release(loop_flag);

#ifdef DNNL_DEV_MODE
    int remaining_regs = ra.get_alloced_regs() - pre_regs;
    gpu_assert(remaining_regs == 0)
            << remaining_regs
            << " registers are allocated that need to be released.";
#endif
}

template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::emov(generator_t<hw> &host,
        const ngen::InstructionModifier &mod, const ngen::RegData &dst,
        const ngen::Immediate &src0) {
    EmulationImplementation::emov(host, mod, dst, src0, emu_strategy);
}

template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::emov(generator_t<hw> &host,
        const ngen::InstructionModifier &mod, const ngen::RegData &dst,
        const ngen::RegData &src0) {
    EmulationImplementation::emov(host, mod, dst, src0, emu_strategy);
}

template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::eadd(generator_t<hw> &host,
        const ngen::InstructionModifier &mod, const ngen::RegData &dst,
        const ngen::RegData &src0, const ngen::Immediate &src1) {
    EmulationState state;
    state.temp[0] = ra.alloc();
    state.temp[1] = ra.alloc();
    EmulationImplementation::eadd(
            host, mod, dst, src0, src1, emu_strategy, state);
    ra.release(state.temp[0]);
    ra.release(state.temp[1]);
}

template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::eadd(generator_t<hw> &host,
        const ngen::InstructionModifier &mod, const ngen::RegData &dst,
        const ngen::RegData &src0, const ngen::RegData &src1) {
    EmulationState state;
    state.temp[0] = ra.alloc();
    state.temp[1] = ra.alloc();
    EmulationImplementation::eadd(
            host, mod, dst, src0, src1, emu_strategy, state);
    ra.release(state.temp[0]);
    ra.release(state.temp[1]);
}

template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::emul(generator_t<hw> &host,
        const ngen::InstructionModifier &mod, const ngen::RegData &dst,
        const ngen::RegData &src0, const ngen::Immediate &src1) {
    EmulationState state;
    state.temp[0] = ra.alloc();
    state.temp[1] = ra.alloc();
    EmulationImplementation::emul(
            host, mod, dst, src0, src1, emu_strategy, state);
    ra.release(state.temp[0]);
    ra.release(state.temp[1]);
}

template <gpu_gen_t hw>
void reduction_injector_f32_t<hw>::emul(generator_t<hw> &host,
        const ngen::InstructionModifier &mod, const ngen::RegData &dst,
        const ngen::RegData &src0, const ngen::RegData &src1) {
    EmulationState state;
    state.temp[0] = ra.alloc();
    state.temp[1] = ra.alloc();
    EmulationImplementation::emul(
            host, mod, dst, src0, src1, emu_strategy, state);
    ra.release(state.temp[0]);
    ra.release(state.temp[1]);
}

REG_GEN9_ISA(template struct reduction_injector_f32_t<gpu_gen9>);
REG_GEN11_ISA(template struct reduction_injector_f32_t<gpu_gen11>);
REG_XELP_ISA(template struct reduction_injector_f32_t<gpu_xe_lp>);
REG_XEHP_ISA(template struct reduction_injector_f32_t<gpu_xe_hp>);
REG_XEHPG_ISA(template struct reduction_injector_f32_t<gpu_xe_hpg>);
REG_XEHPC_ISA(template struct reduction_injector_f32_t<gpu_xe_hpc>);
REG_XE2_ISA(template struct reduction_injector_f32_t<gpu_xe2>);
REG_XE3_ISA(template struct reduction_injector_f32_t<gpu_xe3>);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
