/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

template <>
void jit_uni_cvt_ps_to_xf16_t<avx2_vnni_2>::setup_mask() {
    static const uint32_t mask_in[16]
            = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                    0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0, 0};
    const Xbyak::Reg64 reg64_mask = reg_tmp;

    if (!is_dynamic_size_) {
        constexpr int max_words_in_ymm = 8;
        auto mask_in_offset = max_words_in_ymm - tail_size_;
        mov(reg64_mask, reinterpret_cast<size_t>(&mask_in[mask_in_offset]));
    } else {
        mov(reg64_mask, reinterpret_cast<size_t>(&mask_in[8]));
        mov(reg_scratch, reg_tail);
        shl(reg_scratch, 2);
        sub(reg64_mask, reg_scratch);
    }
    vmovups(vmm_in_mask, ptr[reg64_mask]);
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
            ? ptr[reg_output + out_offset] | ktail_xf16_mask
            : ptr[reg_output + out_offset];
    vmovups(vmm_m_in, ptr[reg_input + sizeof(float) * idx]);
    vcvtps2ph(addr_m_out, vmm_input, _op_mxcsr);
}

void jit_avx512_core_cvt_ps_to_bf16_t::cvt_ps_to_xf16(
        const int idx, const bool is_tail) {
    const size_t out_offset = sizeof(float16_t) * idx;
    const auto addr_m_out = is_tail
            ? ptr[reg_output + out_offset] | ktail_xf16_mask
            : ptr[reg_output + out_offset];

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

template <>
void jit_uni_cvt_ps_to_xf16_t<avx2_vnni_2>::cvt_ps_to_xf16(
        const int idx, const bool is_tail) {
    if (is_tail) {
        uni_vxorps(vmm_input, vmm_input, vmm_input);
        vmaskmovps(
                vmm_input, vmm_in_mask, ptr[reg_input + sizeof(float) * idx]);
    } else if (output_dt_ == data_type::f16) {
        vmovups(vmm_input, ptr[reg_input + sizeof(float) * idx]);
    }

    switch (output_dt_) {
        case data_type::bf16:
            if (is_tail)
                vcvtneps2bf16(vmm_output, vmm_input, Xbyak::VexEncoding);
            else
                vcvtneps2bf16(vmm_output,
                        yword[reg_input + sizeof(float) * idx],
                        Xbyak::VexEncoding);
            break;
        case data_type::f16:
            if (is_tail)
                vcvtps2ph(vmm_output, vmm_input, _op_mxcsr);
            else
                vcvtps2ph(ptr[reg_output + sizeof(float16_t) * idx], vmm_input,
                        _op_mxcsr);
            break;
        default: assert(!"Invalid datatype");
    }

    if (is_tail) {
        auto tail_store = [&](int load_size) {
            store_bytes(vmm_output, reg_output, sizeof(float16_t) * idx,
                    sizeof(float16_t) * load_size);
        };
        if (is_dynamic_size_)
            runtime_tail_process<Xbyak::Xmm>(
                    reg_tail, reg_tmp, tail_store, data_type::f16);
        else
            tail_store(tail_size_);

    } else if (output_dt_ == data_type::bf16)
        vmovups(ptr[reg_output + sizeof(bfloat16_t) * idx], vmm_output);
}

#undef GET_OFF

template struct jit_uni_cvt_ps_to_xf16_t<avx2_vnni_2>;
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
        assert(IMPLICATION(unroll > 1, unroll % 2 == 0));
        L(l_simd_loop[i + 1]);
        {
            cmp(reg_nelems, simd_w_ * unroll);
            jl(l_simd_loop[i], T_NEAR);
            for (int j = 0; j < utils::div_up(unroll, elem_granularity); ++j)
                convert_xf16(j, unroll > 1);
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
void jit_uni_cvt_xf16_to_ps_t<isa>::convert_xf16(
        const int idx, const bool handle_x2) {
    const size_t offset = idx * simd_w_;
    const auto out_addr = ptr[reg_output + sizeof(float) * offset];
    const auto in_addr = ptr[reg_input + sizeof(bfloat16_t) * offset];
    switch (input_dt_) {
        case data_type::bf16:
            vpmovzxwd(get_vmm_src(idx), in_addr);
            vpslld(get_vmm_src(idx), get_vmm_src(idx), 0x10);
            break;
        case data_type::f16: vcvtph2psx(get_vmm_src(idx), in_addr); break;
        default: assert(!"Invalid datatype");
    }
    if (with_add_) vaddps(get_vmm_src(idx), get_vmm_src(idx), out_addr);
    uni_vmovdqu(out_addr, get_vmm_src(idx));
}

template <typename Wmm>
struct helper_avx2_cvt_xf16_t {
    static void convert_xf16(jit_generator *host,
            const impl::data_type_t input_dt, const Xbyak::Address in_addr,
            const int even_src, const int odd_src, const int tmp_1,
            const int tmp_2) {
        const Wmm vmm_even_src = Wmm(even_src);
        const Wmm vmm_odd_src = Wmm(odd_src);
        const Wmm vmm_tmp_1 = Wmm(tmp_1);
        const Wmm vmm_tmp_2 = Wmm(tmp_2);

        switch (input_dt) {
            case data_type::bf16:
                host->vcvtneebf162ps(vmm_even_src, in_addr);
                host->vcvtneobf162ps(vmm_odd_src, in_addr);
                break;
            case data_type::f16:
                host->vcvtneeph2ps(vmm_even_src, in_addr);
                host->vcvtneoph2ps(vmm_odd_src, in_addr);
                break;
            default: assert(!"Invalid datatype");
        }
        host->vpunpckldq(vmm_tmp_1, vmm_even_src, vmm_odd_src);
        host->vpunpckhdq(vmm_tmp_2, vmm_even_src, vmm_odd_src);
    }
};

template <>
void jit_uni_cvt_xf16_to_ps_t<avx2_vnni_2>::convert_xf16(
        const int idx, const bool handle_x2) {
    const Vmm vmm_tmp_1 = vmm_tmp;
    const Vmm vmm_tmp_2 = Vmm(get_even_src_idx(idx));
    const size_t offset = idx * simd_w_ * elem_granularity;
    const auto in_addr = ptr[reg_input + sizeof(bfloat16_t) * offset];
    auto get_out_addr = [&](const size_t offset_xmmword = 0) {
        return ptr[reg_output + sizeof(float) * (offset + offset_xmmword)];
    };

    if (handle_x2)
        helper_avx2_cvt_xf16_t<Xbyak::Ymm>::convert_xf16(this, input_dt_,
                in_addr, get_even_src_idx(idx), get_odd_src_idx(idx),
                vmm_tmp_1.getIdx(), vmm_tmp_2.getIdx());
    else
        helper_avx2_cvt_xf16_t<Xbyak::Xmm>::convert_xf16(this, input_dt_,
                in_addr, get_even_src_idx(idx), get_odd_src_idx(idx),
                vmm_tmp_1.getIdx(), vmm_tmp_2.getIdx());

    vperm2f128(vmm_dst, vmm_tmp_1, vmm_tmp_2, 0x20);
    if (handle_x2) vperm2f128(vmm_dst_2, vmm_tmp_1, vmm_tmp_2, 0x31);

    if (with_add_) {
        vaddps(vmm_dst, vmm_dst, get_out_addr());
        if (handle_x2) vaddps(vmm_dst_2, vmm_dst_2, get_out_addr(simd_w_));
    }
    uni_vmovdqu(get_out_addr(), vmm_dst);
    if (handle_x2) uni_vmovdqu(get_out_addr(simd_w_), vmm_dst_2);
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

    auto vmm_masked = get_vmm_src(0) | ktail_mask | T_z;
    switch (input_dt_) {
        case data_type::bf16:
            vpmovzxwd(vmm_masked, ptr[reg_input]);
            vpslld(vmm_masked, get_vmm_src(0), 0x10);
            break;
        case data_type::f16: vcvtph2psx(vmm_masked, ptr[reg_input]); break;
        default: assert(!"Invalid datatype");
    }
    if (with_add_) vaddps(vmm_masked, get_vmm_src(0), ptr[reg_output]);
    vmovdqu32(ptr[reg_output] | ktail_mask, get_vmm_src(0));
}

template <>
void jit_uni_cvt_xf16_to_ps_t<avx2_vnni_2>::cvt_tail() {
    const Vmm vmm_output = get_vmm_src(0);
    const Vmm_down_t vmm_input = Vmm_down_t(vmm_output.getIdx());
    auto runtime_tail_load = [&](int load_size) {
        load_bytes(vmm_input, reg_input, 0, sizeof(bfloat16_t) * load_size);
    };
    auto runtime_tail_store = [&](int load_size) {
        store_data(data_type::f32, vmm_output, reg_output, 0, load_size);
    };

    runtime_tail_process<Xbyak::Xmm>(
            reg_tail, reg_tmp, runtime_tail_load, data_type::f16);
    switch (input_dt_) {
        case data_type::bf16:
            vpmovzxwd(vmm_output, vmm_input);
            vpslld(vmm_output, vmm_input, 0x10);
            break;
        case data_type::f16: vcvtph2ps(vmm_output, vmm_input); break;
        default: assert(!"Invalid datatype");
    }
    runtime_tail_process<Xbyak::Ymm>(
            reg_tail, reg_tmp, runtime_tail_store, data_type::f32);
}

#undef GET_OFF

template struct jit_uni_cvt_xf16_to_ps_t<avx2_vnni_2>;
template struct jit_uni_cvt_xf16_to_ps_t<avx512_core>;
template struct jit_uni_cvt_xf16_to_ps_t<avx512_core_fp16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
