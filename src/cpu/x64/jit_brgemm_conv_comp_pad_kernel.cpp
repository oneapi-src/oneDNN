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

#include "cpu/x64/jit_brgemm_conv_comp_pad_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace nstl;
using namespace data_type;

namespace jit_avx512_core_brgemm_conv_comp_pad_kernel {

#define GET_OFF(field) offsetof(jit_brgemm_conv_comp_pad_call_s, field)

jit_avx512_core_brgemm_conv_comp_pad_kernel_t::
        jit_avx512_core_brgemm_conv_comp_pad_kernel_t(
                const jit_brgemm_conv_conf_t &ajcp)
    : jcp_(ajcp)
    , inp_dsz_(jcp_.wei_dsz)
    , out_dsz_(jcp_.acc_dsz)
    , nb_ic_(utils::div_up(jcp_.ic, 4))
    , inp_ic_sz_(static_cast<size_t>(inp_dsz_) * jcp_.oc_block * 4)
    , inp_kh_sz_(static_cast<size_t>(inp_dsz_) * jcp_.kw * jcp_.icp
              * jcp_.oc_block)
    , inp_kd_sz_(static_cast<size_t>(inp_dsz_) * jcp_.kh * jcp_.kw * jcp_.icp
              * jcp_.oc_block)
    , inp_ocb_sz_(static_cast<size_t>(inp_dsz_) * jcp_.kd * jcp_.kh * jcp_.kw
              * jcp_.icp * jcp_.oc_block)
    , out_ow_sz_(out_dsz_ * jcp_.oc_block) {}

size_t jit_avx512_core_brgemm_conv_comp_pad_kernel_t::out_oc_offset(
        const int m, const int n) const {
    return static_cast<size_t>(out_dsz_)
            * (n * m_block2_
                    + m * jcp_.ker_ranges_size * jcp_.ow * jcp_.oc_block);
}
size_t jit_avx512_core_brgemm_conv_comp_pad_kernel_t::inp_ic_offset(
        const int icb, const int m, const int n) const {
    return static_cast<size_t>(inp_dsz_) * n * m_block2_ * last_ic_block_
            + m * inp_ocb_sz_ + icb * inp_ic_sz_;
}
Xbyak::Zmm jit_avx512_core_brgemm_conv_comp_pad_kernel_t::accum(
        const int n_block, const int m, const int n) const {
    return Xbyak::Zmm(m * n_block + n);
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::store(
        const int m_block, const int n_block) {
    if (jcp_.src_zero_point) {
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto zmm = accum(n_block, m, n);
            auto zmm_tmp = zmm_tmp_1();
            const auto offset = out_oc_offset(m, n);
            auto zp_addr = ptr[reg_aux_zp_comp_out + offset];

            vpmulld(zmm_tmp, zmm, zmm_zp_shift);
            vpaddd(zmm_tmp, zmm_tmp, zp_addr);
            vmovups(zp_addr, zmm_tmp);
        }
    }

    if (jcp_.s8s8_avx512) {
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto zmm = accum(n_block, m, n);
            auto zmm_tmp = zmm_tmp_1();
            const auto offset = out_oc_offset(m, n);
            auto cp_addr = ptr[reg_aux_comp_out + offset];

            vpmulld(zmm_tmp, zmm, zmm_cp_shift);
            vpaddd(zmm_tmp, zmm_tmp, cp_addr);
            vmovups(cp_addr, zmm_tmp);
        }
    }
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::store_accumulators(
        const int m_block, const int n_block) {
    Xbyak::Label ow_loop, ow_loop_end;

    mov(reg_ow_l, ptr[param1 + GET_OFF(ow_l)]);
    mov(reg_aux_comp_out, reg_comp_out);
    mov(reg_aux_zp_comp_out, reg_zp_comp_out);

    L_aligned(ow_loop);
    {
        cmp(reg_ow_l, 0);
        je(ow_loop_end, T_NEAR);

        store(m_block, n_block);

        add(reg_aux_comp_out, out_ow_sz_);
        add(reg_aux_zp_comp_out, out_ow_sz_);
        dec(reg_ow_l);
        jmp(ow_loop, T_NEAR);
    }
    L_aligned(ow_loop_end);
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::zero_accumulators(
        const int m_block, const int n_block) {
    for (int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto zmm = accum(n_block, m, n);
            vpxord(zmm, zmm, zmm);
        }
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::compute(
        const int m_block, const int n_block) {
    for_(size_t icb = 0; icb < nb_ic_; icb++)
    for_(int m = 0; m < m_block; m++)
    for (int n = 0; n < n_block; n++) {
        auto zmm = accum(n_block, m, n);
        const auto oc_offset = inp_ic_offset(icb, m, n);
        auto addr = EVEX_compress_addr(reg_aux_in, oc_offset);
        vpdpbusd(zmm, zmm_one_bytes, addr);
    }
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::kdh_loop(
        const int m_block, const int n_block) {

    Xbyak::Label kd_loop, kh_loop, kd_loop_end, kh_loop_end;
    zero_accumulators(m_block, n_block);
    mov(reg_kd_l, ptr[param1 + GET_OFF(kd_l)]);
    mov(reg_aux_kd_in, reg_in);

    L_aligned(kd_loop);
    {
        cmp(reg_kd_l, 0);
        je(kd_loop_end, T_NEAR);
        mov(reg_kh_l, ptr[param1 + GET_OFF(kh_l)]);
        mov(reg_aux_in, reg_aux_kd_in);
        L_aligned(kh_loop);
        {
            cmp(reg_kh_l, 0);
            je(kh_loop_end, T_NEAR);
            compute(m_block, n_block);
            add(reg_aux_in, inp_kh_sz_);
            dec(reg_kh_l);
            jmp(kh_loop, T_NEAR);
        }
        L_aligned(kh_loop_end);

        add(reg_aux_kd_in, inp_kd_sz_);
        dec(reg_kd_l);
        jmp(kd_loop, T_NEAR);
    }
    L_aligned(kd_loop_end);
    mov(reg_aux_kd_in, reg_in);

    store_accumulators(m_block, n_block);
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::load_params() {
    mov(reg_in, ptr[param1 + GET_OFF(ptr_in)]);
    mov(reg_zp_comp_out, ptr[param1 + GET_OFF(ptr_zp_out)]);
    mov(reg_comp_out, ptr[param1 + GET_OFF(ptr_cp_out)]);

    mov(reg_ow_l, ptr[param1 + GET_OFF(ow_l)]);
    mov(reg_kd_l, ptr[param1 + GET_OFF(kd_l)]);
    mov(reg_kh_l, ptr[param1 + GET_OFF(kh_l)]);
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::generate() {
    preamble();

    load_params();

    // fill registers with byte ones
    const auto reg32_scratch = reg_tmp.cvt32();
    mov(reg32_scratch, 0x1010101);
    vpbroadcastd(zmm_one_bytes, reg32_scratch);

    // fill register with -128 && -1
    mov(reg32_scratch, -128);
    vpbroadcastd(zmm_cp_shift, reg32_scratch);

    mov(reg32_scratch, -1);
    vpbroadcastd(zmm_zp_shift, reg32_scratch);

    const int nb = jcp_.oc_block / m_block2_;
    const int nb2 = nb / n_max_regs_;
    const int nb2_tail = nb % n_block2_;
    const int n_block = (nb2 == 0) ? nstl::max(1, nb2_tail) : 4;

    const auto M = jcp_.nb_oc;
    const auto m_max_regs = max_regs_ / n_block;
    const auto m_block = nstl::min(m_max_regs, M);

    const int mb_count = M / m_block;
    const int mb_tail = M % m_block;

    Xbyak::Label label_mb_loop;
    mov(reg_mb_count, mb_count);

    L(label_mb_loop);
    {
        kdh_loop(m_block, n_block);
        add(reg_in, m_block * inp_ocb_sz_);
        if (jcp_.s8s8_avx512) add(reg_comp_out, out_oc_offset(m_block, 0));
        if (jcp_.src_zero_point)
            add(reg_zp_comp_out, out_oc_offset(m_block, 0));
        dec(reg_mb_count);
        jnz(label_mb_loop);
    }
    if (mb_tail > 0) kdh_loop(mb_tail, n_block);

    postamble();
}

} // namespace jit_avx512_core_brgemm_conv_comp_pad_kernel

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
