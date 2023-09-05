/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "cpu/x64/jit_avx512_sparse_decompress_kernel.hpp"

#define GET_OFF(field) offsetof(call_params_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

Xbyak::Opmask jit_avx512_sparse_decompress_kernel_t::get_opmask(int idx) {
    switch (idx) {
        case 0: return k1; break;
        case 1: return k2; break;
        case 2: return k3; break;
        case 3: return k4; break;
        default: assert(!"incorrect index"); return k0;
    }
}

Xbyak::Opmask jit_avx512_sparse_decompress_kernel_t::get_load_mask(int idx) {
    // This function prepares a load mask for loading packed values. The minimum
    // and maximum number of bits that can be set is 0 and 64 respectively.
    // Since `shl` instruction doesn't work when the `count` operand is 64
    // (the instruction actually uses `count % 64` instead of just `count`)
    // we have to split 1 shift into 2 shifts (3 if there is a tail).
    //
    //
    // The following pseudo-code is implemented in JIT below:
    // shift = reg_popcnt / 2;
    // res = 1;
    // res = res << shift;
    // res = res << shift;
    // shift_tail = reg_popcnt % 2;
    // res = res << shift_tail;

    // Save original number of bits set in 1.
    mov(reg_popcnt_tmp, reg_popcnt);

    mov(reg_tmp, 1);

    // shift = reg_popcnt / 2
    shr(reg_popcnt, 1);

    // Apply shift two times.
    shl(reg_tmp, reg_popcnt.cvt8());
    shl(reg_tmp, reg_popcnt.cvt8());

    // Calculate shift_tail = reg_popcnt % 2.
    mov(reg_popcnt, reg_popcnt_tmp);
    and_(reg_popcnt, 1);

    // Apply shift_tail.
    shl(reg_tmp, reg_popcnt.cvt8());

    sub(reg_tmp, 1);

    // Restore the value (used to advance the pointer to packed values).
    mov(reg_popcnt, reg_popcnt_tmp);

    auto opmask = get_opmask(idx);
    kmovq(opmask, reg_tmp);

    return opmask;
}

Xbyak::Opmask jit_avx512_sparse_decompress_kernel_t::get_expand_mask(int idx) {
    return get_opmask(idx);
}

Xbyak::Reg64 jit_avx512_sparse_decompress_kernel_t::get_reg_mask_tmp(int idx) {
    switch (idx) {
        case 0: return r13;
        case 1: return r14;
        case 2: return r15;
        case 3: return rax;
        default: assert(!"incorrect index"); return Xbyak::Reg64(0);
    }
};

Xbyak::Zmm jit_avx512_sparse_decompress_kernel_t::get_zmm(int idx) {
    switch (idx) {
        case 0: return Xbyak::Zmm(25);
        case 1: return Xbyak::Zmm(26);
        case 2: return Xbyak::Zmm(27);
        case 3: return Xbyak::Zmm(28);
        default: assert(!"incorrect index"); return Xbyak::Zmm(0);
    }
}

void jit_avx512_sparse_decompress_kernel_t::generate() {
    preamble();
    mov(reg_bitmask_ptr, ptr[param1 + GET_OFF(bitmask_ptr)]);
    mov(reg_dst_ptr, ptr[param1 + GET_OFF(dst_ptr)]);
    mov(reg_src_ptr, ptr[param1 + GET_OFF(src_ptr)]);

    assert(unroll_factor() == 4);

    for (int b = 0; b < nblks_to_decompress_; b++) {
        const int blk_offset = b * blk_sz_;
        const int bitmask_off = blk_offset / CHAR_BIT;
        const int nbytes_per_load = 64;

        for (int i = 0; i < b_blk_sz_; i += unroll_factor()) {
            for (int uf = 0; uf < unroll_factor(); uf++) {
                auto reg_mask_tmp = get_reg_mask_tmp(uf);
                mov(reg_mask_tmp,
                        ptr[reg_bitmask_ptr + (i + uf) * sizeof(uint64_t)
                                + bitmask_off]);
                popcnt(reg_popcnt, reg_mask_tmp);

                auto load_mask = get_load_mask(uf);
                auto zmm_reg = get_zmm(uf);
                vmovdqu8(zmm_reg | load_mask | T_z, ptr[reg_src_ptr]);
                add(reg_src_ptr, reg_popcnt);

                auto expand_mask = get_expand_mask(uf);
                kmovq(expand_mask, reg_mask_tmp);
                vpexpandb(zmm_reg | expand_mask | T_z, zmm_reg);
                vmovdqu8(ptr[reg_dst_ptr + blk_offset
                                 + (i + uf) * nbytes_per_load],
                        zmm_reg);
            }
        }
    }
    postamble();
}

#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
