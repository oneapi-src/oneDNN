/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "cpu/x64/lrn/jit_avx512_common_lrn_bwd_blocked.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

using acc_data_t = float;

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_bwd_blocked_t<d_type>::
        jit_avx512_common_lrn_kernel_bwd_blocked_t(
                const struct nChw16c_across_t &J, float A, float B,
                int use_h_parallel, void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size)
    , nalphabeta(-2 * A * B)
    , use_h_parallelism(use_h_parallel)
    , bf16_emu_(nullptr) {

    vlen = d_type == bf16 ? 32 : 64;
    reg_block = 3;
    src_prev_offset = vlen - 4 * sizeof(data_t);

    xmm_size = 4 * sizeof(acc_data_t);
    zmm_size = 64;
    buffer_block = xmm_size + zmm_size + xmm_size;
    buffer_nest_offset = xmm_size + zmm_size;

    if (d_type == bf16 && !mayiuse(avx512_core_bf16)) {
        bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_scratch,
                bf16_emu_reserv_4);
        bf16_emu_->init_vcvtneps2bf16();
    }

    this->preamble();

#define GET_OFF(field) offsetof(jit_args_bwd_t, field)
    mov(src, ptr[param + GET_OFF(src)]);
    mov(diffdst, ptr[param + GET_OFF(diff_dst)]);
    mov(workspace0, ptr[param + GET_OFF(ws0)]);
    mov(workspace1, ptr[param + GET_OFF(ws1)]);
    mov(diffsrc, ptr[param + GET_OFF(diff_src)]);
#undef GET_OFF

    W = J.W;
    HW = J.H * J.W;
    int LSB = this->use_h_parallelism ? W : HW;

    sub(t, reg_block * buffer_block);
    mov(imm_addr64, float2int(this->nalphabeta));
    movq(xnalphabeta, imm_addr64);
    vbroadcastss(znalphabeta, xnalphabeta);

    version = J.version;

    if (version == across_version::First || version == across_version::Single) {
        vxorps(xmm1, xmm1, xmm1);
        for (int irb = 0; irb < reg_block; irb++) {
            vmovups(ptr[t + irb * buffer_block], xmm1);
        }
    }
    if (version == across_version::Last || version == across_version::Single) {
        vxorps(xmm1, xmm1, xmm1);
        for (int irb = 0; irb < reg_block; irb++) {
            vmovups(ptr[t + irb * buffer_block + buffer_nest_offset], xmm1);
        }
    }

    int LSREST = LSB % reg_block;
    int LS = LSB - LSREST;

    Label lrn_loop;

    if (LS > 0) {
        mov(hw, LS);

        L(lrn_loop);
        {
            compute_loop(reg_block, 1, 1);

            add(src, reg_block * vlen);
            add(diffsrc, reg_block * vlen);
            add(diffdst, reg_block * vlen);
            add(workspace0, reg_block * vlen);
            add(workspace1, reg_block * vlen);

            for (int irb = 0; irb < reg_block; irb++)
                dec(hw);
            cmp(hw, 0);
            jne(lrn_loop, T_NEAR);
        }
    }

    compute_loop(LSREST, 1, this->use_h_parallelism ? 0 : 1);

    add(t, reg_block * buffer_block);
    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_blocked_t<d_type>::compute_loop(
        int loop_size_param, int prefetchL1, int prefetchL2) {
    // loop_size - param for IRB_LOOP macro
    int loop_size = loop_size_param;
    const int prf0_offt = 1 * reg_block;
    const int prf2_offt = 8 * reg_block;

    auto xreg = [=](int irb, int i) { return Xmm(irb * 6 + i); };

    auto zreg = [=](int irb, int i) { return Zmm(irb * 6 + i); };
    auto load_data = [=](Xmm reg, const Address p) {
        if (d_type == bf16) {
            vpmovzxwd(reg, p);
            vpslld(reg, reg, 0x10);
        } else
            vmovups(reg, p);
    };

    auto store_data = [=](bool nt, const Address addr, Zmm zr) {
        if (d_type == bf16) {
            Ymm yr = Ymm(zr.getIdx());
            if (mayiuse(avx512_core_bf16))
                vcvtneps2bf16(yr, zr);
            else
                bf16_emu_->vcvtneps2bf16(yr, zr);
            vmovdqu16(addr, yr);
        } else if (nt)
            uni_vmovntps(addr, zr);
        else
            uni_vmovups(addr, zr);
    };

    // ---- prefetching -------------------------------------------
    if (version != across_version::First && version != across_version::Single) {
        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(
                    ptr[workspace1 + (irb + prf0_offt - 2 * HW) * vlen]));
        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(
                    ptr[diffdst + (irb + prf0_offt - HW) * vlen]));
    }

    if (prefetchL1)
        IRB_LOOP(mic_prefetcht0(ptr[src + (irb + prf0_offt) * vlen]));
    if (prefetchL2)
        IRB_LOOP(mic_prefetcht2(ptr[src + (irb + prf2_offt) * vlen]));

    if (prefetchL1)
        IRB_LOOP(mic_prefetcht0(ptr[workspace1 + (irb + prf0_offt) * vlen]));

    if (prefetchL1)
        IRB_LOOP(mic_prefetcht0(ptr[diffdst + (irb + prf0_offt) * vlen]));

    if (version != across_version::Last && version != across_version::Single) {
        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(
                    ptr[workspace1 + (irb + prf0_offt + 2 * HW) * vlen]));
        if (prefetchL2)
            IRB_LOOP(mic_prefetcht2(
                    ptr[workspace1 + (irb + prf2_offt + 2 * HW) * vlen]));

        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(
                    ptr[diffdst + (irb + prf0_offt + HW) * vlen]));
        if (prefetchL2)
            IRB_LOOP(mic_prefetcht2(
                    ptr[diffdst + (irb + prf2_offt + HW) * vlen]));
    }
    if (prefetchL1)
        IRB_LOOP(mic_prefetcht0(ptr[workspace0 + (irb + prf0_offt) * vlen]));
    if (prefetchL2)
        IRB_LOOP(mic_prefetcht2(ptr[workspace0 + (irb + prf2_offt) * vlen]));
    // -----------------------------------------------------------

    if (loop_size_param == 0) return;

    if (version != across_version::First && version != across_version::Single) {
        IRB_LOOP(load_data(xreg(irb, xws1_prev),
                ptr[workspace1 + (irb - 2 * HW) * vlen + src_prev_offset]));
        IRB_LOOP(load_data(xreg(irb, xdiffdst_prev),
                ptr[diffdst + (irb - HW) * vlen + src_prev_offset]));
        IRB_LOOP(vmulps(xreg(irb, xdiffdst_prev), xreg(irb, xdiffdst_prev),
                xreg(irb, xws1_prev)));
    }

    IRB_LOOP(load_data(
            zreg(irb, zws1), EVEX_compress_addr(workspace1, irb * vlen)));
    IRB_LOOP(load_data(
            zreg(irb, zdiffdst), EVEX_compress_addr(diffdst, irb * vlen)));
    IRB_LOOP(vmulps(zreg(irb, zdiffsrc), zreg(irb, zdiffdst), zreg(irb, zws1)));

    if (version != across_version::Last && version != across_version::Single) {
        IRB_LOOP(load_data(
                xreg(irb, xws1_next), ptr[workspace1 + (irb + 2 * HW) * vlen]));
        IRB_LOOP(load_data(
                xreg(irb, xdiffdst_next), ptr[diffdst + (irb + HW) * vlen]));
        IRB_LOOP(vmulps(xreg(irb, xdiffdst_next), xreg(irb, xdiffdst_next),
                xreg(irb, xws1_next)));
    }

    if (version != across_version::First && version != across_version::Single) {
        IRB_LOOP(
                vmovups(ptr[t + irb * buffer_block], xreg(irb, xdiffdst_prev)));
    }
    IRB_LOOP(vmovups(EVEX_compress_addr(t, irb * buffer_block + xmm_size),
            zreg(irb, zdiffsrc)));
    if (version != across_version::Last && version != across_version::Single) {
        IRB_LOOP(vmovups(ptr[t + irb * buffer_block + buffer_nest_offset],
                xreg(irb, xdiffdst_next)));
    }
    size_t acc_size = sizeof(acc_data_t);
    IRB_LOOP(vmovups(zreg(irb, za),
            EVEX_compress_addr(
                    t, irb * buffer_block + xmm_size - 2 * acc_size)));
    IRB_LOOP(vmovups(zreg(irb, zb),
            EVEX_compress_addr(
                    t, irb * buffer_block + xmm_size - 1 * acc_size)));
    IRB_LOOP(vmovups(zreg(irb, zd),
            EVEX_compress_addr(
                    t, irb * buffer_block + xmm_size + 1 * acc_size)));
    IRB_LOOP(vmovups(zreg(irb, ze),
            EVEX_compress_addr(
                    t, irb * buffer_block + xmm_size + 2 * acc_size)));
    IRB_LOOP(vaddps(zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, za)));
    assert(zsrc == za);
    IRB_LOOP(load_data(zreg(irb, zsrc), EVEX_compress_addr(src, irb * vlen)));
    IRB_LOOP(vaddps(zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, zb)));
    IRB_LOOP(vaddps(zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, zd)));
    IRB_LOOP(vaddps(zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, ze)));
    IRB_LOOP(vmulps(zreg(irb, zsrc), zreg(irb, zsrc), znalphabeta));

    IRB_LOOP(load_data(
            zreg(irb, zws0), EVEX_compress_addr(workspace0, irb * vlen)));
    IRB_LOOP(vdivps(zreg(irb, zdiffdst), zreg(irb, zdiffdst), zreg(irb, zws0)));
    IRB_LOOP(vfmadd213ps(
            zreg(irb, zdiffsrc), zreg(irb, zsrc), zreg(irb, zdiffdst)));

    Label unaligned_store, end_store;
    test(diffsrc, vlen - 1);
    jnz(unaligned_store, T_NEAR);
    IRB_LOOP(store_data(true, EVEX_compress_addr(diffsrc, irb * vlen),
            zreg(irb, zdiffsrc)));
    jmp(end_store, T_NEAR);
    L(unaligned_store);
    {
        IRB_LOOP(store_data(false, EVEX_compress_addr(diffsrc, irb * vlen),
                zreg(irb, zdiffsrc)));
    }
    L(end_store);
}

template class jit_avx512_common_lrn_kernel_bwd_blocked_t<f32>;
template class jit_avx512_common_lrn_kernel_bwd_blocked_t<bf16>;

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
