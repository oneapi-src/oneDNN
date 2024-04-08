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
#include "gpu/compute/device_info.hpp"
#include "gpu/jit/codegen/register_allocator.hpp"
#include "gpu/jit/codegen/register_scope.hpp"
#include "gpu/jit/emulation.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/jit_reduction_injector.hpp"
#include "gpu/jit/ngen/ngen_core.hpp"
#include "gpu/jit/ngen/ngen_interface.hpp"

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
            alg_kind_t alg, dim_t stride, dim_t iters)
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
        requireLocalID(1);
        requireLocalSize();
        externalName("ngen_jit_reduction");
        finalizeInterface();

        prologue();
        setDefaultNoMask();

        ra.claim(r0);
        ngen::Subregister tg_idx0 = r0.ud(1);
        ngen::Subregister tg_idx1 = r0.ud(6);
        ngen::Subregister tg_size0 = getLocalSize(0).uw();
        ngen::Subregister tg_size1 = getLocalSize(1).uw();
        ngen::GRF lid = getLocalID(0);
        ngen::Subregister src_ptr = getArgument("src_ptr").uq();
        ngen::Subregister dst_ptr = getArgument("dst_ptr").uq();
        ra.claim(src_ptr);
        ra.claim(dst_ptr);
        ra.claim(tg_size0);
        ra.claim(tg_size1);
        ra.claim(lid);

        // SRC offset: (tg_idx0 * simd + tg_idx1 * stride * iters + lid) * sizeof(float)
        // DST offset: (tg_idx0 * simd + tg_idx1 * stride + lid) * sizeof(float)
        // - Use the tg_idx0 * simd + lid terms in common for both
        ngen::GRF off_generic = ra.alloc().ud();
        emov(simd, off_generic, lid); // uw -> ud
        emad(simd, off_generic, off_generic, tg_idx0, simd);

        ngen::GRF src_off = ra.alloc().ud();
        ngen::GRF src_addr = ra.alloc_range(2)[0].uq();
        emad(simd, src_off, off_generic, tg_idx1, stride * iters);
        emul(simd, src_off, src_off, dt_size);
        // Change src_off stride before adding to src_ptr
        emov(simd, src_addr, 0);
        emov(simd, src_addr.ud(0)(2), src_off);
        eadd(simd, src_addr, src_addr, src_ptr);

        ngen::GRF acc = ra.alloc().f();
        jit_reduction_injector_f32<hw> reduce(
                *this, alg, ra, device_info.stepping_id());
        reduce.compute(src_addr, acc, stride, iters);

        finalize(alg, acc, iters);

        ngen::GRF dst_off = ra.alloc().ud();
        ngen::GRF dst_addr = ra.alloc_range(2)[0].uq();
        emad(simd, dst_off, off_generic, tg_idx1, stride);
        emul(simd, dst_off, dst_off, dt_size);
        // Change dst_off stride before adding dst_ptr
        emov(simd, dst_addr, 0);
        emov(simd, dst_addr.ud(0)(2), dst_off);
        eadd(simd, dst_addr, dst_addr, dst_ptr);
        store(simd, ngen::scattered_byte(4), A64, dst_addr, acc);

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

    void finalize(alg_kind_t alg, const ngen::GRF &acc, dim_t iters) {
        int simd = ngen::GRF::bytes(hw) / acc.getBytes();
        switch (alg) {
            case alg_kind::reduction_mean:
                mul(simd, acc, acc, 1.0f / iters);
                break;
            default: break;
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
