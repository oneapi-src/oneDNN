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
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_convert_xf16.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

#define GET_OFF(field) offsetof(cvt_xf16_support::jit_call_t, field)

template <cpu_isa_t isa>
void jit_uni_cvt_ps_to_xf16_t<isa>::generate() {

    preamble();

    mov(reg_input, ptr[abi_param1 + GET_OFF(inp)]);
    mov(reg_output, ptr[abi_param1 + GET_OFF(out)]);
    if (is_dynamic_size_) mov(reg_nelems, ptr[abi_param1 + GET_OFF(nelems)]);

    init_bf16();

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
                    cvt_ps_to_xf16(j, false);
                }
                add(reg_input, simd_w_ * unroll * sizeof(float));
                add(reg_output, simd_w_ * unroll * sizeof(float16_t));
                sub(reg_nelems, simd_w_ * unroll);
                jmp(l_simd_loop[i + 1], T_NEAR);
            }
        }
        L(l_simd_loop[0]);

        test(reg_nelems, reg_nelems);
        jz(l_simd_notail, T_NEAR);

        mov(reg_tail, reg_nelems);
        setup_mask();

        cvt_ps_to_xf16(0, true);

        L(l_simd_notail);
    } else {
        const size_t blocked_size = (nelems_ / simd_w_) * simd_w_;
        constexpr size_t unroll_length = 1024;
        const size_t number_of_loops = blocked_size / unroll_length;
        const size_t loop_tail = blocked_size % unroll_length;

        if (number_of_loops > 0) {
            Xbyak::Label l_number_of_loops;
            mov(reg_nelems, number_of_loops);
            L(l_number_of_loops);
            for (size_t i = 0; i < unroll_length; i += simd_w_)
                cvt_ps_to_xf16(i, false);
            add(reg_input, sizeof(float) * unroll_length);
            add(reg_output, sizeof(float16_t) * unroll_length);

            dec(reg_nelems);
            cmp(reg_nelems, 0);
            jg(l_number_of_loops, T_NEAR);
        }
        if (loop_tail > 0) {
            for (size_t i = 0; i < loop_tail; i += simd_w_)
                cvt_ps_to_xf16(i, false);
            add(reg_input, sizeof(float) * loop_tail);
            add(reg_output, sizeof(float16_t) * loop_tail);
        }
        if (tail_size_ != 0) {
            setup_mask();
            cvt_ps_to_xf16(0, true);
        }
    }
    postamble();
}

template <cpu_isa_t isa>
void jit_uni_cvt_ps_to_xf16_t<isa>::setup_mask() {
    const Xbyak::Reg32 reg_mask = reg_tmp.cvt32();
    if (is_dynamic_size_) {
        mov(reg_mask, 1);
        shl(reg_mask, reg_tail.cvt8());
        sub(reg_mask, 1);
    } else {
        mov(reg_mask, (1 << tail_size_) - 1);
    }
    kmovd(ktail_xf16_mask, reg_mask);
    kmovw(ktail_f32_mask, reg_mask);
}

// NOTE: putting the function's definition in the header results in
// a compilation error for VS.
template <cpu_isa_t isa>
void jit_uni_cvt_ps_to_xf16_t<isa>::cvt_ps_to_xf16(
        const int idx, const bool is_tail) {
    assert(!"unimplemented template");
}

template <>
void jit_uni_cvt_ps_to_xf16_t<avx512_core_fp16>::cvt_ps_to_xf16(
        const int idx, const bool is_tail) {
    const Vmm vmm_m_in = is_tail ? vmm_input | ktail_f32_mask | T_z : vmm_input;
    const size_t out_offset = sizeof(float16_t) * idx;
    const auto addr_m_out = is_tail
            ? vword_out[reg_output + out_offset] | ktail_xf16_mask
            : vword_out[reg_output + out_offset];
    vmovups(vmm_m_in, ptr[reg_input + sizeof(float) * idx]);
    vcvtps2ph(addr_m_out, vmm_input, _op_mxcsr);
}

void jit_avx512_core_cvt_ps_to_bf16_t::cvt_ps_to_xf16(
        const int idx, const bool is_tail) {
    const size_t out_offset = sizeof(float16_t) * idx;
    const auto addr_m_out = is_tail
            ? vword_out[reg_output + out_offset] | ktail_xf16_mask
            : vword_out[reg_output + out_offset];

    if (use_bf16_emu_) {
        const Vmm vmm_m_in
                = is_tail ? vmm_input | ktail_f32_mask | T_z : vmm_input;
        vmovups(vmm_m_in, ptr[reg_input + sizeof(float) * idx]);
        bf16_emu_->vcvtneps2bf16(vmm_output, vmm_input);
    } else {
        const auto vmm_m_out
                = is_tail ? vmm_output | ktail_xf16_mask | T_z : vmm_output;
        vcvtneps2bf16(vmm_m_out, ptr[reg_input + sizeof(float) * idx]);
    }
    vmovdqu16(addr_m_out, vmm_output);
}

#undef GET_OFF

template struct jit_uni_cvt_ps_to_xf16_t<avx512_core>;
template struct jit_uni_cvt_ps_to_xf16_t<avx512_core_fp16>;

#define GET_OFF(field) \
    offsetof(cvt_xf16_support::jit_cvt_xf16_to_ps_params_t, field)

template <cpu_isa_t isa>
void jit_uni_cvt_xf16_to_ps_t<isa>::generate() {
    preamble();
    const bool long_row_stride = (row_stride_ * sizeof(float16_t) >> 32) != 0;
    MAYBE_UNUSED(long_row_stride);

    mov(reg_input, ptr[abi_param1 + GET_OFF(inp)]);
    mov(reg_output, ptr[abi_param1 + GET_OFF(out)]);
    mov(reg_nelems, ptr[abi_param1 + GET_OFF(nelems)]);
    mov(reg_nrows, ptr[abi_param1 + GET_OFF(rows)]);

    Label l_row_start, l_row_end;
    Label l_exit; // used for row_stride_

    if (row_stride_) {
        test(reg_nrows, reg_nrows);
        jz(l_exit, T_NEAR); // fast exit: nrows == 0
        mov(reg_nelems_save, reg_nelems);
        mov(reg_rollback, reg_nelems);
        and_(reg_rollback, ~(simd_w_ - 1));
        neg(reg_rollback);
        if (long_row_stride) {
            mov(reg_long_row_stride, row_stride_ * sizeof(float16_t));
            lea(reg_long_row_stride,
                    ptr[reg_long_row_stride
                            + reg_rollback * sizeof(float16_t)]);
        }
    }

    L(l_row_start);

    constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
    Label l_simd_loop[n_unroll + 2];
    for (int i = n_unroll; i >= 0; i--) {
        const int unroll = 1 << i; // 4, 2, 1
        L(l_simd_loop[i + 1]);
        {
            cmp(reg_nelems, simd_w_ * unroll);
            jl(l_simd_loop[i], T_NEAR);
            for (int j = 0; j < unroll; ++j)
                convert_xf16(j);
            add(reg_input, simd_w_ * unroll * sizeof(float16_t));
            add(reg_output, simd_w_ * unroll * sizeof(float));
            sub(reg_nelems, simd_w_ * unroll);
            if (i == n_unroll && n_unroll != 0) jmp(l_simd_loop[i + 1], T_NEAR);
        }
    }
    L(l_simd_loop[0]);

    test(reg_nelems, reg_nelems);
    jz(l_row_end, T_NEAR);

    mov(reg_tail, reg_nelems);
    cvt_tail();

    L(l_row_end);

    if (row_stride_) {
        dec(reg_nrows);
        jz(l_exit, T_NEAR);

        // wraparound
        lea(reg_output, ptr[reg_output + reg_rollback * sizeof(float)]);
        if (long_row_stride)
            add(reg_input, reg_long_row_stride);
        else
            lea(reg_input,
                    ptr[reg_input + reg_rollback * sizeof(float16_t)
                            + row_stride_ * sizeof(float16_t)]);
        mov(reg_nelems, reg_nelems_save);
        jmp(l_row_start);

        L(l_exit);
    }

    postamble();
}

template <cpu_isa_t isa>
void jit_uni_cvt_xf16_to_ps_t<isa>::convert_xf16(const int idx) {
    const size_t offset = idx * simd_w_;
    const auto out_addr = ptr[reg_output + sizeof(float) * offset];
    const auto in_addr = ptr[reg_input + sizeof(bfloat16_t) * offset];
    switch (input_dt_) {
        case data_type::bf16:
            vpmovzxwd(vmm_cvt(idx), in_addr);
            vpslld(vmm_cvt(idx), vmm_cvt(idx), 0x10);
            break;
        case data_type::f16:
            assert(utils::one_of(isa, avx512_core_fp16));
            vcvtph2psx(vmm_cvt(idx), in_addr);
            break;
        default: assert(!"Invalid datatype");
    }
    if (with_add_) vaddps(vmm_cvt(idx), vmm_cvt(idx), out_addr);
    uni_vmovdqu(out_addr, vmm_cvt(idx));
}

template <cpu_isa_t isa>
void jit_uni_cvt_xf16_to_ps_t<isa>::cvt_tail() {
    const Reg32 reg32_mask
            = reg_nelems.cvt32(); // no need for reg_nelems anymore

    // ktail_mask <-- (1 << (nelems % simd_w_)) - 1
    mov(reg32_mask, 1);
    shl(reg32_mask, reg_tail.cvt8());
    sub(reg32_mask, 1);
    kmovd(ktail_mask, reg32_mask);

    auto vmm_masked = vmm_cvt(0) | ktail_mask | T_z;
    switch (input_dt_) {
        case data_type::bf16:
            vpmovzxwd(vmm_masked, ptr[reg_input]);
            vpslld(vmm_masked, vmm_cvt(0), 0x10);
            break;
        case data_type::f16: vcvtph2psx(vmm_masked, ptr[reg_input]); break;
        default: assert(!"Invalid datatype");
    }
    if (with_add_) vaddps(vmm_masked, vmm_cvt(0), ptr[reg_output]);
    vmovdqu32(ptr[reg_output] | ktail_mask, vmm_cvt(0));
}

#undef GET_OFF

template struct jit_uni_cvt_xf16_to_ps_t<avx512_core>;
template struct jit_uni_cvt_xf16_to_ps_t<avx512_core_fp16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
