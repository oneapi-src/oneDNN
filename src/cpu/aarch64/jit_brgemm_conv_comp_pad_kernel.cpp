/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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

#include "cpu/aarch64/jit_brgemm_conv_comp_pad_kernel.hpp"
#include "cpu/aarch64/jit_brgemm_conv_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;
using namespace nstl;
using namespace data_type;
using namespace Xbyak_aarch64;

namespace jit_uni_brgemm_conv_comp_pad_kernel {

#define GET_OFF(field) offsetof(jit_brgemm_conv_comp_pad_call_s, field)

jit_uni_brgemm_conv_comp_pad_kernel_t::jit_uni_brgemm_conv_comp_pad_kernel_t(
        const jit_brgemm_conv_conf_t &ajcp)
    : jcp_(ajcp)
    , inp_dsz_(jcp_.wei_dsz)
    , out_dsz_(jcp_.acc_dsz)
    , nb_ic_(utils::div_up(jcp_.ic, 4))
    , inp_ic_sz_(static_cast<size_t>(inp_dsz_) * jcp_.oc_block * 4)
    , inp_kw_sz_(static_cast<size_t>(inp_dsz_) * jcp_.icp * jcp_.oc_block)
    , inp_kh_sz_(static_cast<size_t>(jcp_.kw) * inp_kw_sz_)
    , inp_kd_sz_(static_cast<size_t>(jcp_.kh) * inp_kh_sz_)
    , isa_max_regs(isa_num_vregs(jcp_.isa)) {}

size_t jit_uni_brgemm_conv_comp_pad_kernel_t::out_oc_offset(const int n) const {
    return static_cast<size_t>(out_dsz_) * n * m_block2_;
}
size_t jit_uni_brgemm_conv_comp_pad_kernel_t::inp_ic_offset(
        const int m_block, const int icb, const int m, const int n) const {
    return static_cast<size_t>(inp_dsz_) * n * m_block2_ * last_ic_block_
            + ((icb * m_block) + m) * inp_ic_sz_;
}
Xbyak_aarch64::ZReg jit_uni_brgemm_conv_comp_pad_kernel_t::accum(
        const int n_block, const int m, const int n) const {
    return Xbyak_aarch64::ZReg(m * n_block + n);
}

void jit_uni_brgemm_conv_comp_pad_kernel_t::store_accumulators(
        const int m_block, const int n_block) {
    if (jcp_.src_zero_point) {
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            str(vmm_zp_shift, ptr(X_TRANSLATOR_STACK, -1, MUL_VL));

            auto vmm = accum(n_block, m, n);
            auto vmm_tmp = vmm_tmp_1();
            auto vmm_zp = vmm_zp_shift;
            ldr(vmm_tmp, ptr(X_TRANSLATOR_STACK, -1, MUL_VL));
            add_imm(X_DEFAULT_ADDR, reg_zp_comp_out, out_oc_offset(n), X_TMP_0);
            ldr(vmm_zp, ptr(X_DEFAULT_ADDR));

            mul(vmm_tmp.s, P_ALL_ONE / T_m, vmm.s);
            add(vmm_tmp.s, vmm_tmp.s, vmm_zp.s);
            st1w(vmm_tmp.s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));

            ldr(vmm_zp_shift, ptr(X_TRANSLATOR_STACK, -1, MUL_VL));
        }
    }

    if (jcp_.s8s8_compensation_required) {
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            str(vmm_cp_shift, ptr(X_TRANSLATOR_STACK, -1, MUL_VL));

            auto vmm = accum(n_block, m, n);
            auto vmm_tmp = vmm_tmp_1();
            auto vmm_zp = vmm_cp_shift;
            ldr(vmm_tmp, ptr(X_TRANSLATOR_STACK, -1, MUL_VL));
            add_imm(X_DEFAULT_ADDR, reg_comp_out, out_oc_offset(n), X_TMP_0);
            ldr(vmm_zp, ptr(X_DEFAULT_ADDR));

            mul(vmm_tmp.s, P_ALL_ONE / T_m, vmm.s);
            add(vmm_tmp.s, vmm_tmp.s, vmm_zp.s);
            st1w(vmm_tmp.s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));

            ldr(vmm_cp_shift, ptr(X_TRANSLATOR_STACK, -1, MUL_VL));
        }
    }
}

void jit_uni_brgemm_conv_comp_pad_kernel_t::zero_accumulators(
        const int m_block, const int n_block) {
    for_(int m = 0; m < m_block; m++)
    for (int n = 0; n < n_block; n++) {
        auto vmm = accum(n_block, m, n);
        eor(vmm.d, vmm.d, vmm.d);
    }
}

void jit_uni_brgemm_conv_comp_pad_kernel_t::compute(const int ic_step,
        const int m_block, const int n_block, const int m_tail,
        const bool is_mb_tail) {

    for_(int ic = 0; ic < ic_step; ++ic)
    for (int m = 0; m < m_block; ++m) {
        if (is_mb_tail && (ic * m_block + m) >= m_tail) break;
        for (int n = 0; n < n_block; ++n) {
            auto vmm = accum(n_block, m, n);
            const auto oc_offset = inp_ic_offset(m_block, ic, m, n);
            add_imm(X_DEFAULT_ADDR, reg_aux_in, oc_offset, X_TMP_0);
            ldr(vmm_tmp, ptr(X_DEFAULT_ADDR));
            sdot(vmm.s, vmm_one_bytes.b, vmm_tmp.b);
        }
    }
}

void jit_uni_brgemm_conv_comp_pad_kernel_t::icb_loop(const int icb,
        const int icb_tail, const int ic_step, const int m_block,
        const int mb_tail, const int n_block) {
    Xbyak_aarch64::Label label_icb_loop, label_loop_end;

    mov(reg_aux_in, reg_aux_kw_in);
    mov_imm(reg_icb, icb);

    L(label_icb_loop);
    {
        cmp(reg_icb, 0);
        b(EQ, label_loop_end);
        compute(ic_step, m_block, n_block, 0, false);
        add_imm(reg_aux_in, reg_aux_in, ic_step * m_block * inp_ic_sz_,
                X_TMP_0);
        sub(reg_icb, reg_icb, 1);
        b(label_icb_loop);
    }
    L_aligned(label_loop_end);

    if (icb_tail) compute(ic_step, mb_tail, n_block, icb_tail, true);
}

void jit_uni_brgemm_conv_comp_pad_kernel_t::khw_loop(const int icb,
        const int icb_tail, const int ic_step, const int m_block,
        const int mb_tail, const int n_block) {
    Xbyak_aarch64::Label label_kw_loop, label_kw_end, label_kh_loop,
            label_kh_end;
    add_imm(reg_kh_l, param1, GET_OFF(kh_l), X_TMP_0);
    mov(reg_aux_kh_in, reg_in);

    L_aligned(label_kh_loop);
    {
        cmp(reg_kh_l, 0);
        b(EQ, label_kh_end);
        add_imm(reg_kw_l, param1, GET_OFF(kw_l), X_TMP_0);
        mov(reg_aux_kw_in, reg_aux_kh_in);
        L_aligned(label_kw_loop);
        {
            cmp(reg_kw_l, 0);
            b(EQ, label_kw_end);
            icb_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block);
            add_imm(reg_aux_kw_in, reg_aux_kw_in, inp_kw_sz_, X_TMP_0);
            sub(reg_kw_l, reg_kw_l, 1);
            b(label_kw_loop);
        }
        L_aligned(label_kw_end);

        add_imm(reg_aux_kh_in, reg_aux_kh_in, inp_kh_sz_, X_TMP_0);
        sub(reg_kh_l, reg_kh_l, 1);
        b(label_kh_loop);
    }
    L_aligned(label_kh_end);
}

void jit_uni_brgemm_conv_comp_pad_kernel_t::load_params() {
    add_imm(reg_in, param1, GET_OFF(ptr_in), X_TMP_0);
    add_imm(reg_zp_comp_out, param1, GET_OFF(ptr_zp_out), X_TMP_1);
    add_imm(reg_comp_out, param1, GET_OFF(ptr_cp_out), X_TMP_2);
}

int jit_uni_brgemm_conv_comp_pad_kernel_t::compute_ic_step(
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

void jit_uni_brgemm_conv_comp_pad_kernel_t::generate() {
    preamble();

    load_params();

    // fill registers with byte ones
    const auto reg32_scratch = WReg(reg_tmp.getIdx());
    mov_imm(reg32_scratch, 0x1010101);
    dup(vmm_one_bytes.s, reg32_scratch);

    // fill register with -128 && -1
    mov(reg32_scratch, -128);
    dup(vmm_cp_shift.s, reg32_scratch);

    mov(reg32_scratch, -1);
    dup(vmm_zp_shift.s, reg32_scratch);

    const bool is_int8_sve = (jcp_.src_dt == data_type::s8 || jcp_.src_dt == u8)
            && jcp_.wei_dt == data_type::s8 && !jcp_.has_int8_vnni;
    if (is_int8_sve) {
        mov(reg32_scratch, 0x1);
        dup(zmm_one_words.h, reg32_scratch);
    }

    const int max_regs = isa_max_regs
            - (is_int8_sve ? 6 : (jcp_.s8s8_compensation_required ? 4 : 3));
    const int nb = div_up(nstl::min(jcp_.oc, jcp_.oc_block), m_block2_);
    const int nb2 = nb / n_max_regs_;
    const int nb2_tail = nb % n_block2_;
    const int n_block = (nb2 == 0) ? nstl::max(1, nb2_tail) : 4;

    const size_t m_max_regs = max_regs / n_block;
    const int m_block = nstl::min(m_max_regs, nb_ic_);
    const int ic_step = compute_ic_step(m_max_regs, m_block, n_block);

    assert(m_block * n_block <= max_regs);

    const auto blocks = m_block * ic_step;
    const auto icb = nb_ic_ / blocks;
    const auto icb_tail = nb_ic_ % blocks;
    const auto mb_tail = div_up(icb_tail, ic_step);

    Label label_kd_loop, label_loop_end;
    add_imm(reg_kd_l, param1, GET_OFF(kd_l), X_TMP_0);

    zero_accumulators(m_block, n_block);

    L_aligned(label_kd_loop);
    {
        cmp(reg_kd_l, 0);
        b(EQ, label_loop_end);
        khw_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block);
        add_imm(reg_in, reg_in, inp_kd_sz_, X_TMP_0);
        sub(reg_kd_l, reg_kd_l, 1);
        b(label_kd_loop);
    }
    L_aligned(label_loop_end);

    store_accumulators(m_block, n_block);

    postamble();
}

} // namespace jit_uni_brgemm_conv_comp_pad_kernel

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
