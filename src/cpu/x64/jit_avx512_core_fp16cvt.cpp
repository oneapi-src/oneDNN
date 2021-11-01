/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <assert.h>

#include "common/float16.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_avx512_core_fp16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(f16_support::jit_call_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

void jit_avx512_core_fp16_cvt_ps_to_f16_t::generate() {
    preamble();

    auto cvt = [&](size_t idx, Xbyak::Opmask ktail_mask) {
        vmovups(fp32_inp | ktail_mask | T_z,
                ptr[reg_inp + sizeof(float) * (idx)]);
        vcvtps2ph(f16_out, fp32_inp, _op_mxcsr);
        vmovdqu16(yword[reg_out + sizeof(float16_t) * (idx)] | ktail_mask,
                f16_out);
    };

    mov(reg_inp, ptr[abi_param1 + GET_OFF(inp)]);
    mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
    if (is_dynamic_size_) mov(reg_nelems, ptr[abi_param1 + GET_OFF(nelems)]);

    mov(reg32_tail, 0xffff);
    kmovw(ktail_mask, reg32_tail);

    if (is_dynamic_size_) { // determine nelems after JIT is called
        constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
        Xbyak::Label l_simd_loop[n_unroll + 2], l_simd_notail;
        for (int i = n_unroll; i >= 0; i--) {
            const int unroll = 1 << i; // 4, 2, 1
            L(l_simd_loop[i + 1]);
            {
                cmp(reg_nelems, simd_w_ * unroll);
                jl(l_simd_loop[i], T_NEAR);
                for (int j = 0; j < simd_w_ * unroll; j += simd_w_) {
                    cvt(j, ktail_mask);
                }
                add(reg_inp, simd_w_ * unroll * sizeof(float));
                add(reg_out, simd_w_ * unroll * sizeof(float16_t));
                sub(reg_nelems, simd_w_ * unroll);
                jmp(l_simd_loop[i + 1], T_NEAR);
            }
        }
        L(l_simd_loop[0]);
        test(reg_nelems, reg_nelems);
        jz(l_simd_notail);
        // JIT of `tail_mask_ = (1 << (nelems_ % simd_w_)) - 1;`
        mov(reg32_mask, 1);
        mov(reg64_tail, reg_nelems);
        shl(reg32_mask, reg8_mask_shift);
        sub(reg32_mask, 1);
        kmovd(ktail_mask, reg32_mask);
        cvt(0, ktail_mask);
        L(l_simd_notail);
    } else {
        size_t blocked_size = (nelems_ / simd_w_) * simd_w_;
        const size_t loop_length = 1024;
        const size_t number_of_loops = blocked_size / loop_length;
        const size_t tail_of_loops = blocked_size % loop_length;

        if (number_of_loops > 0) {
            Xbyak::Label l_number_of_loops;
            mov(reg_nelems, number_of_loops);
            L(l_number_of_loops);
            for (size_t i = 0; i < loop_length; i += simd_w_)
                cvt(i, ktail_mask);
            add(reg_inp, sizeof(float) * loop_length);
            add(reg_out, sizeof(float16_t) * loop_length);

            dec(reg_nelems);
            cmp(reg_nelems, 0);
            jg(l_number_of_loops, T_NEAR);
        }
        if (tail_of_loops > 0) {
            for (size_t i = 0; i < tail_of_loops; i += simd_w_)
                cvt(i, ktail_mask);
            add(reg_inp, sizeof(float) * tail_of_loops);
            add(reg_out, sizeof(float16_t) * tail_of_loops);
        }
        if (tail_mask_ != 0) {
            mov(reg32_tail, tail_mask_);
            kmovw(ktail_mask, reg32_tail);
            cvt(0, ktail_mask);
        }
    }
    postamble();
}

void jit_avx512_core_fp16_cvt_f16_to_ps_t::generate() {
    const int simd = 16;
    const bool long_row_stride = (row_stride_ * sizeof(float16_t) >> 32) != 0;

#ifdef _WIN32
    Reg64 reg_out = rax; // instead of abi_param1 (rcx)
    Reg64 reg_nrows = abi_param4;
    Reg64 reg_long_row_stride = rdi; // non-volatile
    const bool preserve_reg_long_row_stride = long_row_stride;
#else
    Reg64 reg_out = abi_param1;
    Reg64 reg_nrows = rax; // instead of abi_param4 (rcx)
    Reg64 reg_long_row_stride = r8;
    const bool preserve_reg_long_row_stride = false;
#endif
    mov(rax, rcx); // rcx is used for mask computations

    if (preserve_reg_long_row_stride) push(reg_long_row_stride);

    Reg64 reg_inp = abi_param2;
    Reg64 reg_nelems = abi_param3;

    Reg64 reg_nelems_save = r10;
    Reg64 reg_rollback = r11; // - round_down(len, simd)
    Label l_exit;

    auto vreg = [&](int offset) {
        const int id = offset / simd;
        assert(0 <= id && id < 6); // volatile registers on both win & lnx
        return Zmm(id);
    };

    auto safe_ret = [&]() {
        if (preserve_reg_long_row_stride) pop(reg_long_row_stride);
        ret();
    };

    if (row_stride_) {
        test(reg_nrows, reg_nrows);
        jz(l_exit, T_NEAR); // fast exit: nrows == 0
        mov(reg_nelems_save, reg_nelems);
        mov(reg_rollback, reg_nelems);
        and_(reg_rollback, ~(simd - 1));
        neg(reg_rollback);
        if (long_row_stride) {
            mov(reg_long_row_stride, row_stride_ * sizeof(float16_t));
            lea(reg_long_row_stride,
                    ptr[reg_long_row_stride
                            + reg_rollback * sizeof(float16_t)]);
        }
    }

    Label l_row_start, l_row_end;
    L(l_row_start);

    constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
    Label l_simd_loop[n_unroll + 2];
    for (int i = n_unroll; i >= 0; i--) {
        const int unroll = 1 << i; // 4, 2, 1
        L(l_simd_loop[i + 1]);
        {
            cmp(reg_nelems, simd * unroll);
            jl(l_simd_loop[i], T_NEAR);
            for (int j = 0; j < simd * unroll; j += simd) {
                auto out_addr = zword[reg_out + sizeof(float) * j];
                vcvtph2psx(vreg(j), ptr[reg_inp + sizeof(float16_t) * j]);
                if (with_add_) vaddps(vreg(j), vreg(j), out_addr);
                vmovdqu32(out_addr, vreg(j));
            }
            add(reg_inp, simd * unroll * sizeof(float16_t));
            add(reg_out, simd * unroll * sizeof(float));
            sub(reg_nelems, simd * unroll);
            if (i == n_unroll && n_unroll != 0) jmp(l_simd_loop[i + 1], T_NEAR);
        }
    }
    L(l_simd_loop[0]);

    test(reg_nelems, reg_nelems);
    jz(l_row_end, T_NEAR);

    // tail processing
    Reg8 reg8_mask_shift = cl;
    Opmask ktail_mask = k1;

    // ktail_mask <-- (1 << (nelems % simd)) - 1
    mov(reg8_mask_shift.cvt64(), reg_nelems);
    Reg32 reg32_mask = reg_nelems.cvt32(); // no need for reg_nelems anymore
    mov(reg32_mask, 1);
    shl(reg32_mask, reg8_mask_shift);
    sub(reg32_mask, 1);
    kmovd(ktail_mask, reg32_mask);

    vcvtph2psx(vreg(0) | ktail_mask | T_z, zword[reg_inp]);
    if (with_add_) vaddps(vreg(0) | ktail_mask | T_z, vreg(0), zword[reg_out]);
    vmovdqu32(zword[reg_out] | ktail_mask, vreg(0));

    L(l_row_end);
    if (!row_stride_) return safe_ret();

    dec(reg_nrows);
    jz(l_exit, T_NEAR);

    // wraparound
    lea(reg_out, ptr[reg_out + reg_rollback * sizeof(float)]);
    if (long_row_stride)
        add(reg_inp, reg_long_row_stride);
    else
        lea(reg_inp,
                ptr[reg_inp + reg_rollback * sizeof(float16_t)
                        + row_stride_ * sizeof(float16_t)]);
    mov(reg_nelems, reg_nelems_save);
    jmp(l_row_start);

    L(l_exit);
    return safe_ret();
}

void jit_avx512_core_fp16_add_cvt_ps_to_f16_t::generate() {
    preamble();

    auto add_cvt = [&](size_t idx, Xbyak::Opmask ktail_mask) {
        vmovups(fp32_inp | ktail_mask | T_z,
                ptr[reg_inp + sizeof(float) * (idx)]);
        vaddps(fp32_inp | ktail_mask | T_z, fp32_inp,
                ptr[reg_add + sizeof(float) * (idx)]);

        vcvtps2ph(f16_out, fp32_inp, _op_mxcsr);

        vmovdqu16(yword[reg_out + sizeof(float16_t) * (idx)] | ktail_mask,
                f16_out);
    };

    mov(reg_inp, ptr[abi_param1 + GET_OFF(inp)]);
    mov(reg_add, ptr[abi_param1 + GET_OFF(add)]);
    mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
    mov(reg_nelems, ptr[abi_param1 + GET_OFF(nelems)]);

    mov(reg32_tail, 0xffff);
    kmovw(ktail_mask, reg32_tail);

    constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
    Xbyak::Label l_simd_loop[n_unroll + 2], l_simd_notail;
    for (int i = n_unroll; i >= 0; i--) {
        const int unroll = 1 << i; // 4, 2, 1
        L(l_simd_loop[i + 1]);
        {
            cmp(reg_nelems, simd_w_ * unroll);
            jl(l_simd_loop[i], T_NEAR);
            for (int j = 0; j < simd_w_ * unroll; j += simd_w_) {
                add_cvt(j, ktail_mask);
            }
            add(reg_inp, simd_w_ * unroll * sizeof(float));
            add(reg_add, simd_w_ * unroll * sizeof(float));
            add(reg_out, simd_w_ * unroll * sizeof(float16_t));

            sub(reg_nelems, simd_w_ * unroll);
            jmp(l_simd_loop[i + 1], T_NEAR);
        }
    }
    L(l_simd_loop[0]);
    test(reg_nelems, reg_nelems);
    jz(l_simd_notail);
    // JIT of `tail_mask_ = (1 << (nelems_ % simd_w_)) - 1;`
    mov(reg32_mask, 1);
    mov(reg64_tail, reg_nelems);
    shl(reg32_mask, reg8_mask_shift);
    sub(reg32_mask, 1);
    kmovd(ktail_mask, reg32_mask);
    add_cvt(0, ktail_mask);
    L(l_simd_notail);

    postamble();
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
