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
#include "gpu/jit/emulated_generator.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/jit_reduction_injector.hpp"
#include "gpu/jit/ngen/ngen_core.hpp"
#include "gpu/jit/ngen/ngen_interface.hpp"
#include "gpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

template <gpu_gen_t hw>
class jit_reduction_generator_t : public emulated_generator_t<hw> {
protected:
    NGEN_FORWARD_OPENCL(hw);
    FORWARD_EMULATION(hw);

public:
    jit_reduction_generator_t(const compute::device_info_t &device_info,
            alg_kind_t alg, dim_t stride, dim_t iters, int nregs)
        : emulated_generator_t<hw>(device_info, "ngen_jit_reduction") {
        constexpr auto GlobalPtr = ngen::ExternalArgumentType::GlobalPtr;

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

        ra().claim(r0);
        ngen::Subregister tg_idx0 = r0.ud(1);
        ngen::Subregister tg_idx1 = r0.ud(6);
        ngen::Subregister tid = r0.ud(2).b(0);
        ngen::Subregister tg_size0 = getLocalSize(0).uw();
        ngen::Subregister src_ptr = getArgument("src_ptr").uq();
        ngen::Subregister dst_ptr = getArgument("dst_ptr").uq();
        ra().claim(src_ptr);
        ra().claim(dst_ptr);
        ra().claim(tg_size0);

        // SRC offset: tid*grf_bytes*nregs + tg_idx0*tg_size0*dt_size*nregs + tg_idx1*stride*iters*dt_size
        // DST offset: tid*grf_bytes*nregs + tg_idx0*tg_size0*dt_size*nregs + tg_idx1*stride*dt_size
        const int max_write_size = device_info.max_exec_size();
        int nwrites = utils::div_up(regs_per_thr * grf_bytes, max_write_size);
        int regs_per_write = max_write_size / grf_bytes;

        ngen::Subregister inner_off = ra().alloc_sub(ngen::DataType::ud);
        ngen::Subregister outer_off = ra().alloc_sub(ngen::DataType::ud);
        emul(1, inner_off, tg_idx0, tg_size0);
        emul(1, inner_off, inner_off, dt_size * nregs);
        emad(1, inner_off, inner_off, tid, nregs * grf_bytes);

        emul(1, outer_off, tg_idx1, stride * dt_size);

        ngen::GRF src_addr = ra().alloc().uq();
        ngen::GRFRange dst_addr = ra().alloc_range(nwrites);
        emad(1, src_addr, inner_off, outer_off, iters);
        eadd(1, src_addr, src_addr, src_ptr);
        eadd3(1, dst_addr[0].uq(), dst_ptr, inner_off, outer_off);
        for (int i = 1; i < nwrites; i++) {
            eadd(1, dst_addr[i].uq(), dst_addr[0].uq(),
                    i * grf_bytes * regs_per_write);
        }
        ra().release(inner_off);
        ra().release(outer_off);

        ngen::GRFRange acc = ra().alloc_range(nregs);
        jit_reduction_injector_f32<hw> reduce(
                *this, alg, ra(), device_info.stepping_id());
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
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_JIT_JIT_REDUCTION_GENERATOR_HPP
