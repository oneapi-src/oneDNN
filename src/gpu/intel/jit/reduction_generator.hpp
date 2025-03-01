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

#ifndef GPU_INTEL_JIT_REDUCTION_GENERATOR_HPP
#define GPU_INTEL_JIT_REDUCTION_GENERATOR_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/jit/emulated_generator.hpp"
#include "gpu/intel/jit/generator.hpp"
#include "gpu/intel/jit/reduction_injector.hpp"
#include "gpu/intel/utils.hpp"
#include "ngen/ngen_core.hpp"
#include "ngen/ngen_interface.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

template <gpu_gen_t hw>
class reduction_generator_t : public emulated_generator_t<hw> {
protected:
    NGEN_FORWARD_OPENCL(hw);
    FORWARD_EMULATION(hw);

public:
    reduction_generator_t(const compute::device_info_t &device_info,
            alg_kind_t alg, dim_t stride, dim_t iters, int nregs)
        : emulated_generator_t<hw>(device_info, "ngen_jit_reduction",
                {GENERATOR_NAME, GENERATOR_LINE}) {
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
        ngen::Subregister inner_off = ra().alloc_sub(ngen::DataType::ud);
        ngen::Subregister outer_off = ra().alloc_sub(ngen::DataType::ud);
        emul(1, inner_off, tg_idx0, tg_size0);
        emul(1, inner_off, inner_off, dt_size * nregs);
        emad(1, inner_off, inner_off, tid, nregs * grf_bytes);

        emul(1, outer_off, tg_idx1, stride * dt_size);

        ngen::GRF src_addr = ra().alloc().uq();
        ngen::GRF dst_addr = ra().alloc().uq();
        emad(1, src_addr, inner_off, outer_off, iters);
        eadd(1, src_addr, src_addr, src_ptr);
        eadd3(1, dst_addr, dst_ptr, inner_off, outer_off);
        ra().release(inner_off);
        ra().release(outer_off);

        ngen::GRFRange acc = ra().alloc_range(nregs);
        reduction_injector_f32_t<hw> reduce(
                *this, alg, ra(), device_info.stepping_id());
        reduce.compute(src_addr, acc, stride, iters);
        ra().release(src_addr);

        finalize(simd, alg, acc, iters);

        estore(dst_addr, acc);

        ra().release(acc);
        ra().release(dst_addr);

        ra().release(r0);
        ra().release(src_ptr);
        ra().release(dst_ptr);
        ra().release(tg_size0);
#ifdef DNNL_DEV_MODE
        gpu_assert(ra().get_alloced_regs() == 0)
                << ra().get_alloced_regs()
                << " registers are allocated that need to be released.";
#endif

        epilogue();
    }

protected:
    // Store data from a contiguous range of registers into a contiguous
    // range in global memory (block store)
    void estore(const ngen::GRF &base_dst_addr, const ngen::GRFRange &src) {
        const int grf_bytes = ngen::GRF::bytes(hw);
        int nregs = src.getLen();
        bool force_legacy = gpu_utils::dev_getenv(
                "jit_reduction_force_legacy_send", false);
        bool use_legacy = force_legacy || hw < ngen::HW::XeHPG;
        const int max_store_size = use_legacy ? 128 : 512;
        gpu_assert(max_store_size % grf_bytes == 0) << "Unexpected store size";
        const int max_store_regs = max_store_size / grf_bytes;

        // Load in chunks
        int reg_start = 0;
        while (reg_start < nregs) {
            int store_regs = nstl::min(max_store_regs, nregs - reg_start);
            // Compute the src address
            ngen::GRF addr = ra().alloc().uq();
            eadd(1, addr, base_dst_addr, reg_start * grf_bytes);
            if (use_legacy) {
                // Reduce store_regs according to valid store sizes
                const int oword_per_grf = grf_bytes / 16;
                for (auto store_owords : {8, 4, 2, 1}) {
                    if (store_owords / oword_per_grf > store_regs) continue;
                    store_regs = store_owords / oword_per_grf;
                    break;
                }

                // Do the store
                auto dt = ngen::aligned_block_oword(store_regs * oword_per_grf);
                store(1, dt, A64, addr, src[reg_start]);
            } else {
                // Reduce store_regs according to valid store sizes
                const int d64_per_grf = grf_bytes / 8;
                for (auto store_d64s : {64, 32, 16, 8, 4, 3, 2, 1}) {
                    if (store_d64s / d64_per_grf > store_regs) continue;
                    store_regs = store_d64s / d64_per_grf;
                    break;
                }

                // Do the store
                ngen::DataSpecLSC lscspec = ngen::CacheSettingsLSC::L1UC_L3WB;
                lscspec |= ngen::block(
                        ngen::DataSizeLSC::D64, store_regs * d64_per_grf);
                store.ugm(1, lscspec, A64, addr, src[reg_start]);
            }
            reg_start += store_regs;
            ra().release(addr);
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
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_JIT_REDUCTION_GENERATOR_HPP
