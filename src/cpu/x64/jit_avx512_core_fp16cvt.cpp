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

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

#define GET_OFF(field) offsetof(f16_support::jit_call_t, field)

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
#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
