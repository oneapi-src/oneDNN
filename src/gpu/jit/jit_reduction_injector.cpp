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

#include "oneapi/dnnl/dnnl_types.h"

// Must be included before emulation.hpp
#include "gpu/jit/ngen/ngen.hpp"

#include "common/impl_registration.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/jit/emulation.hpp"
#include "gpu/jit/jit_reduction_injector.hpp"
#include "gpu/jit/ngen/ngen_core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace ngen;

template <gpu_gen_t hw>
void jit_reduction_injector_f32<hw>::sum_fwd(
        int simd, const ngen::GRF &acc, const ngen::GRF &val) {
    eadd(h, simd, acc, acc, val);
}
template <gpu_gen_t hw>
void jit_reduction_injector_f32<hw>::max_fwd(
        int simd, const ngen::GRF &acc, const ngen::GRF &val) {
    h.max_(simd, acc, acc, val);
}
template <gpu_gen_t hw>
void jit_reduction_injector_f32<hw>::min_fwd(
        int simd, const ngen::GRF &acc, const ngen::GRF &val) {
    h.min_(simd, acc, acc, val);
}
template <gpu_gen_t hw>
void jit_reduction_injector_f32<hw>::mul_fwd(
        int simd, const ngen::GRF &acc, const ngen::GRF &val) {
    emul(h, simd, acc, acc, val);
}

template <gpu_gen_t hw>
void jit_reduction_injector_f32<hw>::initialize(
        int simd, const ngen::GRF &reg) {
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
void jit_reduction_injector_f32<hw>::eload(
        int simd, int dt_size, const ngen::GRF &dst, const ngen::GRF &addr) {
    bool force_legacy
            = gpu_utils::dev_getenv("jit_reduction_force_legacy_load", false);
    if (!force_legacy && hw >= ngen::HW::XeHPG) {
        // LSC load
        ngen::DataSpecLSC lscspec = ngen::block(ngen::DataSizeLSC::D32, simd);
        lscspec |= ngen::CacheSettingsLSC::L1C_L3C;
        h.load.ugm(1, dst, lscspec, h.A64, addr);
    } else {
        // Legacy load
        int load_size = simd * dt_size;
        int load_owords = load_size / 16;
        h.load(1, dst, ngen::aligned_block_oword(load_owords), h.A64, addr);
    }
}
template <gpu_gen_t hw>
void jit_reduction_injector_f32<hw>::compute(const ngen::GRF &src_ptr,
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

    int nloads = utils::div_up(nregs, regs_per_inst);
    ngen::GRFRange load_addr = ra.alloc_range(nloads);
    emov(h, 1, load_addr[0].uq(), src_ptr.uq());
    for (int i = 1; i < nloads; i++) {
        eadd(h, 1, load_addr[i].uq(), src_ptr.uq(),
                i * reg_size * regs_per_inst);
    }

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
    for (int i = 0; i < nregs; i += regs_per_inst) {
        int inst_nregs = std::min(regs_per_inst, nregs - i);
        int simd = inst_nregs * elems_per_reg;
        eload(simd, dt_size, val[i].f(), load_addr[i / regs_per_inst]);
    }

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
    for (int i = 0; i < nloads; i++) {
        eadd(h, 1, load_addr[i].uq(), load_addr[i].uq(), stride * dt_size);
    }
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
void jit_reduction_injector_f32<hw>::emov(jit_generator<hw> &host,
        const ngen::InstructionModifier &mod, const ngen::RegData &dst,
        const ngen::Immediate &src0) {
    EmulationImplementation::emov(host, mod, dst, src0, emu_strategy);
}

template <gpu_gen_t hw>
void jit_reduction_injector_f32<hw>::emov(jit_generator<hw> &host,
        const ngen::InstructionModifier &mod, const ngen::RegData &dst,
        const ngen::RegData &src0) {
    EmulationImplementation::emov(host, mod, dst, src0, emu_strategy);
}

template <gpu_gen_t hw>
void jit_reduction_injector_f32<hw>::eadd(jit_generator<hw> &host,
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
void jit_reduction_injector_f32<hw>::eadd(jit_generator<hw> &host,
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
void jit_reduction_injector_f32<hw>::emul(jit_generator<hw> &host,
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
void jit_reduction_injector_f32<hw>::emul(jit_generator<hw> &host,
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

REG_GEN9_ISA(template struct jit_reduction_injector_f32<gpu_gen9>);
REG_GEN11_ISA(template struct jit_reduction_injector_f32<gpu_gen11>);
REG_XELP_ISA(template struct jit_reduction_injector_f32<gpu_xe_lp>);
REG_XEHP_ISA(template struct jit_reduction_injector_f32<gpu_xe_hp>);
REG_XEHPG_ISA(template struct jit_reduction_injector_f32<gpu_xe_hpg>);
REG_XEHPC_ISA(template struct jit_reduction_injector_f32<gpu_xe_hpc>);
REG_XE2_ISA(template struct jit_reduction_injector_f32<gpu_xe2>);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
