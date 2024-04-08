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

#ifndef GPU_JIT_REDUCTION_GENERATOR_HPP
#define GPU_JIT_REDUCTION_GENERATOR_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/jit/codegen/register_allocator.hpp"
#include "gpu/jit/emulation.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/jit_reduction_injector.hpp"
#include "gpu/jit/ngen/ngen_core.hpp"
#include "gpu/jit/ngen/ngen_interface.hpp"
#include "gpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace ngen;

template <gpu_gen_t hw>
class jit_reduction_generator_t : public jit_generator<hw> {
    friend struct EmulationImplementation;
    NGEN_FORWARD_OPENCL(hw);

public:
    jit_reduction_generator_t(const compute::device_info_t &device_info,
            alg_kind_t alg, dim_t stride, dim_t iters, int nregs)
        : ra(hw, "ngen_jit_reduction")
        , emu_strategy(hw, device_info.stepping_id()) {
        constexpr auto GlobalPtr = ExternalArgumentType::GlobalPtr;

        // Number of dst elements computed per thread
        const int grf_bytes = ngen::GRF::bytes(hw);
        const int dt_size = sizeof(float);
        int simd = grf_bytes / dt_size;

        newArgument("src_ptr", GlobalPtr);
        newArgument("dst_ptr", GlobalPtr);
        setDefaultAutoSWSB();
        requireSIMD(simd);

        requireLocalSize();
        externalName("ngen_jit_reduction");
        finalizeInterface();

        prologue();
        setDefaultNoMask();

        ra.claim(r0);
        ngen::Subregister tg_idx0 = r0.ud(1);
        ngen::Subregister tg_idx1 = r0.ud(6);
        ngen::Subregister tid = r0.ud(2).b(0);
        ngen::Subregister tg_size0 = getLocalSize(0).uw();
        ngen::Subregister src_ptr = getArgument("src_ptr").uq();
        ngen::Subregister dst_ptr = getArgument("dst_ptr").uq();
        ra.claim(src_ptr);
        ra.claim(dst_ptr);
        ra.claim(tg_size0);

        // SRC offset: tid*grf_bytes*nregs + tg_idx0*tg_size0*dt_size*nregs + tg_idx1*stride*iters*dt_size
        // DST offset: tid*grf_bytes*nregs + tg_idx0*tg_size0*dt_size*nregs + tg_idx1*stride*dt_size
        const int max_write_size = device_info.max_exec_size();
        int nwrites = utils::div_up(regs_per_thr * grf_bytes, max_write_size);
        int regs_per_write = max_write_size / grf_bytes;

        ngen::Subregister inner_off = ra.alloc_sub(ngen::DataType::ud);
        ngen::Subregister outer_off = ra.alloc_sub(ngen::DataType::ud);
        emul(1, inner_off, tg_idx0, tg_size0);
        emul(1, inner_off, inner_off, dt_size * nregs);
        emad(1, inner_off, inner_off, tid, nregs * grf_bytes);

        emul(1, outer_off, tg_idx1, stride * dt_size);

        ngen::GRF src_addr = ra.alloc().uq();
        ngen::GRFRange dst_addr = ra.alloc_range(nwrites);
        emad(1, src_addr, inner_off, outer_off, iters);
        eadd(1, src_addr, src_addr, src_ptr);
        eadd3(1, dst_addr[0].uq(), dst_ptr, inner_off, outer_off);
        for (int i = 1; i < nwrites; i++) {
            eadd(1, dst_addr[i].uq(), dst_addr[0].uq(),
                    i * grf_bytes * regs_per_write);
        }
        ra.release(inner_off);
        ra.release(outer_off);

        ngen::GRFRange acc = ra.alloc_range(nregs);
        jit_reduction_injector_f32<hw> reduce(
                *this, alg, ra, device_info.stepping_id());
        reduce.compute(src_addr, acc, stride, iters);

        finalize(simd, alg, acc, iters);

        constexpr int oword_bytes = 16;
        const int max_write_owords = max_write_size / oword_bytes;
        int total_write_owords = regs_per_thr * grf_bytes / oword_bytes;
        for (int i = 0; i < total_write_owords; i += max_write_owords) {
            int reg_idx = i / (grf_bytes / oword_bytes);
            int write_size = std::min(max_write_owords, total_write_owords - i);
            int write_regs = write_size * oword_bytes / grf_bytes;
            bool force_legacy = gpu_utils::dev_getenv(
                    "jit_reduction_force_legacy_load", false);
            if (!force_legacy && hw >= ngen::HW::XeHPG) {
                // LSC store
                ngen::DataSpecLSC lscspec = ngen::block(
                        ngen::DataSizeLSC::D32, simd * write_regs);
                lscspec |= ngen::CacheSettingsLSC::L1UC_L3WB;
                store.ugm(1, lscspec, A64, dst_addr[i / max_write_owords].uq(),
                        acc[reg_idx].f());
            } else {
                // legacy store
                store(1, ngen::aligned_block_oword(write_size), A64,
                        dst_addr[i / max_write_owords].uq(), acc[reg_idx].f());
            }
        }

        epilogue();
    }

protected:
    void emov(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0) {
        EmulationImplementation::emov(*this, mod, dst, src0, emu_strategy);
    }
    void emov(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::Immediate &src0) {
        EmulationImplementation::emov(*this, mod, dst, src0, emu_strategy);
    }
    // TODO: Change EmulationState register allocation so it can be handled
    // by the EmulationImplementation directly, instead of maintaining allocation
    // for the entire lifetime of the EmulationState. This would eliminate overeager
    // register allocation when using several injectors which each have emulation.
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        EmulationState state;
        state.temp[0] = ra.alloc();
        state.temp[1] = ra.alloc();
        EmulationImplementation::emul(
                *this, mod, dst, src0, src1, emu_strategy, state);
        ra.release(state.temp[0]);
        ra.release(state.temp[1]);
    }
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::Immediate &src1) {
        EmulationState state;
        state.temp[0] = ra.alloc();
        state.temp[1] = ra.alloc();
        EmulationImplementation::emul(
                *this, mod, dst, src0, src1, emu_strategy, state);
        ra.release(state.temp[0]);
        ra.release(state.temp[1]);
    }
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        // src1 does not support bytes
        if (src1.getType() == ngen::DataType::b) {
            ngen::Subregister src1_w = ra.alloc_sub(ngen::DataType::w);
            emov(mod, src1_w, src1);
            eadd(mod, dst, src0, src1_w);
            ra.release(src1_w);
            return;
        }
        EmulationState state;
        state.temp[0] = ra.alloc();
        state.temp[1] = ra.alloc();
        EmulationImplementation::eadd(
                *this, mod, dst, src0, src1, emu_strategy, state);
        ra.release(state.temp[0]);
        ra.release(state.temp[1]);
    }
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::Immediate &src1) {
        EmulationState state;
        state.temp[0] = ra.alloc();
        state.temp[1] = ra.alloc();
        EmulationImplementation::eadd(
                *this, mod, dst, src0, src1, emu_strategy, state);
        ra.release(state.temp[0]);
        ra.release(state.temp[1]);
    }
    void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const ngen::RegData &src2) {
        // mad is only supported for dw/w types
        auto supported = [](const ngen::RegData &data) -> bool {
            return EmulationImplementation::isDW(data)
                    || EmulationImplementation::isW(data);
        };
        bool src2_supported = EmulationImplementation::isW(src2);
        if (supported(dst) && supported(src0) && supported(src1)
                && src2_supported) {
            mad(mod, dst, src0, src1, src2);
        } else {
            // emulate with separate mul/add
            if (src0 == dst) {
                ngen::Subregister tmp = ra.alloc_sub(dst.getType());
                emul(mod, tmp, src1, src2);
                eadd(mod, dst, tmp, src0);
                ra.release(tmp);
            } else {
                emul(mod, dst, src1, src2);
                eadd(mod, dst, dst, src0);
            }
        }
    }
    void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const ngen::Immediate &src2) {
        auto supported = [](const ngen::RegData &data) -> bool {
            return EmulationImplementation::isDW(data)
                    || EmulationImplementation::isW(data);
        };
        bool src2_supported = EmulationImplementation::isW(src2);
        bool imm_supported
                = ngen::getBytes(src2.getType()) <= 2 && src2_supported;
        bool mad_supported = supported(dst) && supported(src0)
                && supported(src1) && imm_supported;
        if (mad_supported) {
            mad(mod, dst, src0, src1, src2);
        } else {
            // emulate with separate mul/add
            if (src0 == dst) {
                ngen::Subregister tmp = ra.alloc_sub(dst.getType());
                emul(mod, tmp, src1, src2);
                eadd(mod, dst, tmp, src0);
                ra.release(tmp);
            } else {
                emul(mod, dst, src1, src2);
                eadd(mod, dst, dst, src0);
            }
        }
    }
    void eadd3(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const ngen::Immediate &src2) {
        // add3 only supports dw/w types - emulate other options with 2 adds
        auto supported = [](const ngen::RegData &data) -> bool {
            return EmulationImplementation::isDW(data)
                    || EmulationImplementation::isW(data);
        };
        bool src2_supported = utils::one_of(src2.getType(), ngen::DataType::uw,
                ngen::DataType::w, ngen::DataType::ud, ngen::DataType::d);
        if (supported(dst) && supported(src0) && supported(src1)
                && src2_supported) {
            add3(mod, dst, src0, src1, src2);
        } else {
            eadd(mod, dst, src0, src1);
            eadd(mod, dst, dst, src2);
        }
    }
    void eadd3(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const ngen::RegData &src2) {
        // add3 only supports dw/w types - emulate other options with 2 adds
        auto supported = [](const ngen::RegData &data) -> bool {
            return EmulationImplementation::isDW(data)
                    || EmulationImplementation::isW(data);
        };
        if (supported(dst) && supported(src0) && supported(src1)
                && supported(src2)) {
            add3(mod, dst, src0, src1, src2);
        } else {
            eadd(mod, dst, src0, src1);
            eadd(mod, dst, dst, src2);
        }
    }

    void finalize(
            int simd, alg_kind_t alg, const ngen::GRFRange &acc, dim_t iters) {
        int nregs = acc.getLen();
        for (int i = 0; i < nregs; i++) {
            switch (alg) {
                case alg_kind::reduction_mean:
                    mul(simd, acc[i].f(), acc[i].f(), 1.0f / iters);
                    break;
                default: break;
            }
        }
    }

    reg_allocator_t ra;
    EmulationStrategy emu_strategy;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_JIT_JIT_REDUCTION_GENERATOR_HPP
