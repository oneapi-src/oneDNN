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
#include "common/utils.hpp"
#include "gpu/jit/ngen/ngen.hpp"

#include "common/impl_registration.hpp"
#include "common/nstl.hpp"
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
void jit_reduction_injector_f32<hw>::initialize(const ngen::GRF &reg) {
    int simd = ngen::GRF::bytes(hw) / reg.getBytes();
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
    if (hw >= ngen::HW::XeHPG) {
        // LSC load
        ngen::DataSpecLSC lscspec = ngen::block(ngen::DataSizeLSC::D32, simd);
        lscspec |= ngen::CacheSettingsLSC::L1C_L3C;
        h.load.ugm(1, dst.ud(), lscspec, h.A64, addr);
    } else {
        // Legacy load
        h.load(1, dst.ud(), ngen::aligned_block_oword(4), h.A64, addr);
    }
}
template <gpu_gen_t hw>
void jit_reduction_injector_f32<hw>::compute(const ngen::GRF &src_ptr,
        const ngen::GRF &acc, dim_t stride, dim_t iters) {
    using namespace alg_kind;

    assert(src_ptr.getType() == ngen::DataType::uq);

    int dt_size = acc.getBytes();
    int simd = GRF::bytes(hw) / dt_size;

    ngen::GRF load_addr = ra.alloc().uq();
    emov(h, simd, load_addr.ud(), 0); // fill with zeros
    emov(h, 1, load_addr, src_ptr);

    // Set up GRFs used for loop indices
    ngen::Subregister loop_index = ra.alloc_sub(ngen::DataType::d);
    ngen::GRF val = ra.alloc().f();
    ngen::FlagRegister loop_flag = ra.alloc_flag(true);

    initialize(acc);

    // Initialize loop
    ngen::Label loop_start;
    emov(h, 1, loop_index, 0);
    h.mark(loop_start);

    // Load data
    eload(simd, dt_size, val, load_addr);

    // Accumulate
    switch (alg_) {
        case dnnl_reduction_sum:
        case dnnl_reduction_mean: sum_fwd(simd, acc, val); break;
        case dnnl_reduction_max: max_fwd(simd, acc, val); break;
        case dnnl_reduction_min: min_fwd(simd, acc, val); break;
        case dnnl_reduction_mul: mul_fwd(simd, acc, val); break;
        default: gpu_assert(false) << "unsupported reduction algorithm";
    }

    // Iterate
    eadd(h, 1, load_addr, load_addr, stride * dt_size);
    eadd(h, 1, loop_index, loop_index, 1);
    h.cmp(1 | h.lt | loop_flag, loop_index, iters);
    h.jmpi(1 | loop_flag, loop_start);

    // Release used registers
    ra.release(load_addr);
    ra.release(loop_index);
    ra.release(val);
    ra.release(loop_flag);
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
