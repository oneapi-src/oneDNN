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
    : jit_generator(jit_name())
    , jcp_(ajcp)
    , inp_dsz_(jcp_.wei_dsz)
    , out_dsz_(jcp_.acc_dsz)
    , nb_ic_(utils::div_up(jcp_.ic, 4))
    , inp_ic_sz_(static_cast<size_t>(inp_dsz_) * jcp_.oc_block * 4)
    , inp_kw_sz_(static_cast<size_t>(inp_dsz_) * jcp_.icp * jcp_.oc_block)
    , inp_kh_sz_(static_cast<size_t>(jcp_.kw) * inp_kw_sz_)
    , inp_kd_sz_(static_cast<size_t>(jcp_.kh) * inp_kh_sz_) {}

size_t jit_avx512_core_brgemm_conv_comp_pad_kernel_t::out_oc_offset(
        const int n) const {
    return static_cast<size_t>(out_dsz_) * n * m_block2_;
}
size_t jit_avx512_core_brgemm_conv_comp_pad_kernel_t::inp_ic_offset(
        const int m_block, const int icb, const int m, const int n) const {
    return static_cast<size_t>(inp_dsz_) * n * m_block2_ * last_ic_block_
            + ((icb * m_block) + m) * inp_ic_sz_;
}
Xbyak::Zmm jit_avx512_core_brgemm_conv_comp_pad_kernel_t::accum(
        const int n_block, const int m, const int n) const {
    return Xbyak::Zmm(m * n_block + n);
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::store_accumulators(
        const int m_block, const int n_block) {
    if (jcp_.src_zero_point) {
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto zmm = accum(n_block, m, n);
            auto zmm_tmp = zmm_tmp_1();
            const auto offset = out_oc_offset(n);
            auto zp_addr = ptr[reg_zp_comp_out + offset];

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
            const auto offset = out_oc_offset(n);
            auto cp_addr = ptr[reg_comp_out + offset];

            vpmulld(zmm_tmp, zmm, zmm_cp_shift);
            vpaddd(zmm_tmp, zmm_tmp, cp_addr);
            vmovups(cp_addr, zmm_tmp);
        }
    }
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::zero_accumulators(
        const int m_block, const int n_block) {
    for_(int m = 0; m < m_block; m++)
    for (int n = 0; n < n_block; n++) {
        auto zmm = accum(n_block, m, n);
        vpxord(zmm, zmm, zmm);
    }
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::compute(const int ic_step,
        const int m_block, const int n_block, const int m_tail,
        const bool is_mb_tail) {

    for_(int ic = 0; ic < ic_step; ++ic)
    for (int m = 0; m < m_block; ++m) {
        if (is_mb_tail && (ic * m_block + m) >= m_tail) break;
        for (int n = 0; n < n_block; ++n) {
            auto zmm = accum(n_block, m, n);
            const auto oc_offset = inp_ic_offset(m_block, ic, m, n);
            auto addr = EVEX_compress_addr(reg_aux_in, oc_offset);
            vpdpbusd(zmm, zmm_one_bytes, addr);
        }
    }
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::icb_loop(const int icb,
        const int icb_tail, const int ic_step, const int m_block,
        const int mb_tail, const int n_block) {
    Xbyak::Label label_icb_loop, label_loop_end;

    mov(reg_aux_in, reg_aux_kw_in);
    mov(reg_icb, icb);

    L(label_icb_loop);
    {
        cmp(reg_icb, 0);
        je(label_loop_end, T_NEAR);
        compute(ic_step, m_block, n_block, 0, false);
        add(reg_aux_in, ic_step * m_block * inp_ic_sz_);
        dec(reg_icb);
        jmp(label_icb_loop, T_NEAR);
    }
    L_aligned(label_loop_end);

    if (icb_tail) compute(ic_step, mb_tail, n_block, icb_tail, true);
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::khw_loop(const int icb,
        const int icb_tail, const int ic_step, const int m_block,
        const int mb_tail, const int n_block) {
    Xbyak::Label label_kw_loop, label_kw_end, label_kh_loop, label_kh_end;
    mov(reg_kh_l, ptr[param1 + GET_OFF(kh_l)]);
    mov(reg_aux_kh_in, reg_in);

    L_aligned(label_kh_loop);
    {
        cmp(reg_kh_l, 0);
        je(label_kh_end, T_NEAR);
        mov(reg_kw_l, ptr[param1 + GET_OFF(kw_l)]);
        mov(reg_aux_kw_in, reg_aux_kh_in);
        L_aligned(label_kw_loop);
        {
            cmp(reg_kw_l, 0);
            je(label_kw_end, T_NEAR);
            icb_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block);
            add(reg_aux_kw_in, inp_kw_sz_);
            dec(reg_kw_l);
            jmp(label_kw_loop, T_NEAR);
        }
        L_aligned(label_kw_end);

        add(reg_aux_kh_in, inp_kh_sz_);
        dec(reg_kh_l);
        jmp(label_kh_loop, T_NEAR);
    }
    L_aligned(label_kh_end);
}

void jit_avx512_core_brgemm_conv_comp_pad_kernel_t::load_params() {
    mov(reg_in, ptr[param1 + GET_OFF(ptr_in)]);
    mov(reg_zp_comp_out, ptr[param1 + GET_OFF(ptr_zp_out)]);
    mov(reg_comp_out, ptr[param1 + GET_OFF(ptr_cp_out)]);
}

int jit_avx512_core_brgemm_conv_comp_pad_kernel_t::compute_ic_step(
        const int m_max_regs, const int m_block, const int n_block) const {
    int best_ic_step = 1;
    float best_block_eff = 0.f;

    int max_ic_step
            = nstl::min(static_cast<size_t>(m_block), div_up(nb_ic_, m_block));

    // Introduce ic_step to increase kernel efficiency
    // Compute the ic_step based on the optimal kernel efficiency
    for (int ic_s = max_ic_step; ic_s >= 1; --ic_s) {
        const auto blocks = ic_s * m_block;
        const float block_disb
                = static_cast<float>(nb_ic_) / rnd_up(nb_ic_, blocks);
        const float eff = (static_cast<float>(n_block) * blocks)
                / ((n_block + blocks) * max_ic_step);
        const float block_eff = block_disb * eff;
        float block_footprint = static_cast<float>(inp_dsz_) * blocks
                * jcp_.oc_block * last_ic_block_;
        if (block_footprint <= static_cast<float>(
                    platform::get_per_core_cache_size(1))
                && (block_eff > best_block_eff)) {
            best_ic_step = ic_s;
            best_block_eff = block_eff;
        }
    }

    return best_ic_step;
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

    const int max_regs = jcp_.s8s8_avx512 ? 28 : 29;
    const int nb = div_up(nstl::min(jcp_.oc, jcp_.oc_block), m_block2_);
    const int nb2 = nb / n_max_regs_;
    const int nb2_tail = nb % n_block2_;
    const int n_block = (nb2 == 0) ? nstl::max(1, nb2_tail) : 4;

    const size_t m_max_regs = max_regs / n_block;
    const int m_block = nstl::min(m_max_regs, nb_ic_);
    const int ic_step = compute_ic_step(m_max_regs, m_block, n_block);

    const auto blocks = m_block * ic_step;
    const auto icb = nb_ic_ / blocks;
    const auto icb_tail = nb_ic_ % blocks;
    const auto mb_tail = div_up(icb_tail, ic_step);

    Xbyak::Label label_kd_loop, label_loop_end;
    mov(reg_kd_l, ptr[param1 + GET_OFF(kd_l)]);

    zero_accumulators(m_block, n_block);

    L_aligned(label_kd_loop);
    {
        cmp(reg_kd_l, 0);
        je(label_loop_end, T_NEAR);
        khw_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block);
        add(reg_in, inp_kd_sz_);
        dec(reg_kd_l);
        jmp(label_kd_loop, T_NEAR);
    }
    L_aligned(label_loop_end);

    store_accumulators(m_block, n_block);

    postamble();
}

} // namespace jit_avx512_core_brgemm_conv_comp_pad_kernel

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
