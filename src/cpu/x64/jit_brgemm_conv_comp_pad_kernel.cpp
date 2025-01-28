/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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
using namespace prop_kind;

namespace jit_uni_brgemm_conv_comp_pad_kernel {

#define GET_OFF(field) offsetof(jit_brgemm_conv_comp_pad_call_s, field)

template <typename Vmm>
jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::
        jit_uni_brgemm_conv_comp_pad_kernel_t(
                const jit_brgemm_conv_conf_t &ajcp)
    : jit_generator(jit_name())
    , jcp_(ajcp)
    , inp_dsz_(jcp_.wei_dsz)
    , out_dsz_(jcp_.acc_dsz)
    , nb_ic_(utils::div_up(
              jcp_.prop_kind == backward_data ? jcp_.oc : jcp_.ic, 4))
    , inp_ic_sz_(static_cast<size_t>(inp_dsz_)
              * (jcp_.prop_kind == backward_data ? jcp_.ic_block
                                                 : jcp_.oc_block)
              * 4)
    , inp_kw_sz_(static_cast<size_t>(inp_dsz_)
              * (jcp_.prop_kind == backward_data ? jcp_.ocp * jcp_.ic_block
                                                 : jcp_.icp * jcp_.oc_block))
    , inp_kh_sz_(static_cast<size_t>(jcp_.kw) * inp_kw_sz_)
    , inp_kd_sz_(static_cast<size_t>(jcp_.kh) * inp_kh_sz_)
    , out_ow_sz_(static_cast<size_t>(out_dsz_)
              * (jcp_.prop_kind == backward_data ? jcp_.ic_block
                                                 : jcp_.oc_block))
    , out_ker_sz_(static_cast<size_t>(out_ow_sz_)
              * (jcp_.exec_type == exec_trans ? jcp_.prop_kind == backward_data
                                      ? jcp_.iw
                                      : jcp_.ow
                                              : 1))
    , isa_max_regs(isa_num_vregs(jcp_.isa)) {}

template <typename Vmm>
size_t jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::out_oc_offset(
        const int n, const int w) const {
    return static_cast<size_t>(out_dsz_) * n * m_block2_ + w * out_ow_sz_;
}
template <typename Vmm>
size_t jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::inp_ic_offset(
        const int m_block, const int icb, const int m, const int n) const {
    return static_cast<size_t>(inp_dsz_) * n * m_block2_ * last_ic_block_
            + ((icb * m_block) + m) * inp_ic_sz_;
}
template <typename Vmm>
Vmm jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::accum(
        const int n_block, const int m, const int n) const {
    return Vmm(m * n_block + n);
}

template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::store_accumulators(
        const int m_block, const int n_block, const int ow_b, const int ow_e) {
    if (jcp_.src_zero_point) {
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto vmm = accum(n_block, m, n);
            auto vmm_tmp = vmm_tmp_1();
            auto vmm_tmp2 = vmm_one_bytes;
            uni_vpmulld(vmm_tmp, vmm, vmm_zp_shift);

            for (int w = ow_b; w < ow_e; w++) {
                const auto offset = out_oc_offset(n, w);
                auto zp_addr = is_superset(jcp_.isa, avx512_core)
                        ? EVEX_compress_addr(reg_zp_comp_out, offset)
                        : ptr[reg_zp_comp_out + offset];

                uni_vpaddd(vmm_tmp2, vmm_tmp, zp_addr);
                uni_vmovups(zp_addr, vmm_tmp2);
            }
        }
    }

    if (jcp_.s8s8_compensation_required) {
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto vmm = accum(n_block, m, n);
            auto vmm_tmp = vmm_tmp_1();
            auto vmm_tmp2 = vmm_one_bytes;
            uni_vpmulld(vmm_tmp, vmm, vmm_cp_shift);

            for (int w = ow_b; w < ow_e; w++) {
                const auto offset = out_oc_offset(n, w);
                auto cp_addr = is_superset(jcp_.isa, avx512_core)
                        ? EVEX_compress_addr(reg_comp_out, offset)
                        : ptr[reg_comp_out + offset];

                uni_vpaddd(vmm_tmp2, vmm_tmp, cp_addr);
                uni_vmovups(cp_addr, vmm_tmp2);
            }
        }
    }

    // refill register with -1
    const auto reg32_scratch = reg_tmp.cvt32();
    mov(reg32_scratch, 0x1010101);
    uni_vpbroadcastd(vmm_one_bytes, reg32_scratch);
}

template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::copy_ow_body(
        const int n_block, const int ow_b, const int ow_e) {

    if (jcp_.src_zero_point) {
        for_(int w = ow_b; w < ow_e; w++)
        for (int n = 0; n < n_block; n++) {
            auto vmm_tmp = vmm_tmp_1();
            const auto offset = out_oc_offset(n, w);
            auto copy_zp_addr
                    = maybe_EVEX_compress_addr(reg_zp_comp_out, offset);
            auto zp_addr
                    = maybe_EVEX_compress_addr(reg_aux_zp_comp_out, offset);
            vmovups(vmm_tmp, copy_zp_addr);
            vmovups(zp_addr, vmm_tmp);
        }
    }

    if (jcp_.s8s8_compensation_required) {
        for_(int w = ow_b; w < ow_e; w++)
        for (int n = 0; n < n_block; n++) {
            auto vmm_tmp = vmm_tmp_1();
            const auto offset = out_oc_offset(n, w);
            auto copy_cp_addr = maybe_EVEX_compress_addr(reg_comp_out, offset);
            auto cp_addr = maybe_EVEX_compress_addr(reg_aux_comp_out, offset);
            vmovups(vmm_tmp, copy_cp_addr);
            vmovups(cp_addr, vmm_tmp);
        }
    }
}

template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::copy_ow(
        const int m_block, const int n_block, const int ow_b, const int ow_e) {
    mov(reg_ker_l, ptr[param1 + GET_OFF(ker_l)]);
    mov(reg_aux_zp_comp_out, reg_zp_comp_out);
    mov(reg_aux_comp_out, reg_comp_out);

    Xbyak::Label label_ker_loop, label_ker_end;
    L_aligned(label_ker_loop);
    {
        cmp(reg_ker_l, 1);
        je(label_ker_end, T_NEAR);
        if (jcp_.src_zero_point) add(reg_aux_zp_comp_out, out_ker_sz_);
        if (jcp_.s8s8_compensation_required) add(reg_aux_comp_out, out_ker_sz_);
        copy_ow_body(n_block, ow_b, ow_e);
        dec(reg_ker_l);
        jmp(label_ker_loop, T_NEAR);
    }
    L_aligned(label_ker_end);
}

template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::zero_accumulators(
        const int m_block, const int n_block) {
    for_(int m = 0; m < m_block; m++)
    for (int n = 0; n < n_block; n++) {
        auto vmm = accum(n_block, m, n);
        uni_vpxor(vmm, vmm, vmm);
    }
}

template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::compute(const int ic_step,
        const int m_block, const int n_block, const int m_tail,
        const bool is_mb_tail) {

    for_(int ic = 0; ic < ic_step; ++ic)
    for (int m = 0; m < m_block; ++m) {
        if (is_mb_tail && (ic * m_block + m) >= m_tail) break;
        for (int n = 0; n < n_block; ++n) {
            auto vmm = accum(n_block, m, n);
            const auto oc_offset = inp_ic_offset(m_block, ic, m, n);
            auto addr = is_superset(jcp_.isa, avx512_core)
                    ? EVEX_compress_addr(reg_aux_in, oc_offset)
                    : ptr[reg_aux_in + oc_offset];
            if (jcp_.has_int8_vnni) {
                vpdpbusd(vmm, vmm_one_bytes, addr, get_encoding());
            } else {
                vpmaddubsw(zmm_int8_temp, vmm_one_bytes, addr);
                vpmaddwd(zmm_int8_temp, zmm_int8_temp, zmm_one_words);
                vpaddd(vmm, vmm, zmm_int8_temp);
            }
        }
    }
}

template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::icb_loop(const int icb,
        const int icb_tail, const int ic_step, const int m_block,
        const int mb_tail, const int n_block) {
    Xbyak::Label label_icb_loop, label_loop_end;

    mov(reg_aux_in, reg_aux_kh_in);
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
template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::kdh_loop(const int icb,
        const int icb_tail, const int ic_step, const int m_block,
        const int mb_tail, const int n_block) {
    Xbyak::Label label_kd_loop, label_kd_end, label_kh_loop, label_kh_end;
    mov(reg_kd_l, ptr[param1 + GET_OFF(kd_l)]);
    mov(reg_aux_kd_in, reg_in);

    L_aligned(label_kd_loop);
    {
        cmp(reg_kd_l, 0);
        je(label_kd_end, T_NEAR);
        mov(reg_kh_l, ptr[param1 + GET_OFF(kh_l)]);
        mov(reg_aux_kh_in, reg_aux_kd_in);
        L_aligned(label_kh_loop);
        {
            cmp(reg_kh_l, 0);
            je(label_kh_end, T_NEAR);
            icb_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block);
            add(reg_aux_kh_in,
                    jcp_.prop_kind == backward_data ? inp_kh_sz_ * jcp_.stride_h
                                                    : inp_kh_sz_);
            dec(reg_kh_l);
            jmp(label_kh_loop, T_NEAR);
        }
        L_aligned(label_kh_end);

        add(reg_aux_kd_in,
                jcp_.prop_kind == backward_data ? inp_kd_sz_ * jcp_.stride_d
                                                : inp_kd_sz_);
        dec(reg_kd_l);
        jmp(label_kd_loop, T_NEAR);
    }
    L_aligned(label_kd_end);
}

template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::bwd_kw_iw_loop(const int icb,
        const int icb_tail, const int ic_step, const int m_block,
        const int mb_tail, const int n_block) {
    const auto DW = jcp_.dilate_w + 1;
    const auto SW = jcp_.stride_w;
    const auto KW = jcp_.kw;
    const auto LP = jcp_.l_pad;
    const auto nb_iw = div_up(jcp_.iw, SW);

    vector<int> ker_kw_ow_b(SW * KW, -1);
    vector<int> ker_kw_ow_e(SW * KW, -1);

    for_(int sw = 0; sw < SW; sw++)
    for (int iwb = 0; iwb < nb_iw; iwb++) {
        const auto iw = iwb * SW + sw;
        const auto ker_iw = sw * nb_iw + iwb;

        int s {0}, o_test {0};
        while (true) {
            o_test = iw + LP - s * DW;
            if (o_test % SW == 0) break;
            s++;
        }
        const int k_f
                = nstl::min(jcp_.kw, static_cast<int>(div_up(iw + LP + 1, DW)));
        int k_s = nstl::max(
                0, static_cast<int>(div_up(iw + LP - jcp_.ow * SW + 1, DW)));
        while (k_s % SW != s)
            k_s++;

        for (int kw = k_s; kw < k_f; kw += SW) {
            const auto adj_kw = sw * KW + kw;
            ker_kw_ow_b[adj_kw]
                    = ker_kw_ow_b[adj_kw] == -1 ? ker_iw : ker_kw_ow_b[adj_kw];
            ker_kw_ow_e[adj_kw] = ker_iw + 1;
        }
    }

    for (int kw = 0; kw < jcp_.kw; kw++) {
        bool has_kw_computed = false;
        for (int sw = 0; sw < SW; sw++) {
            const auto ker_iw_b = ker_kw_ow_b[sw * KW + kw];
            const auto ker_iw_e = ker_kw_ow_e[sw * KW + kw];

            if (ker_iw_b < ker_iw_e && ker_iw_b >= 0) {
                if (!has_kw_computed) {
                    zero_accumulators(m_block, n_block);
                    kdh_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block);
                }
                store_accumulators(m_block, n_block, ker_iw_b, ker_iw_e);
                has_kw_computed = true;
            }
        }
        add(reg_in, inp_kw_sz_);
    }
}

template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::fwd_kw_ow_loop(const int icb,
        const int icb_tail, const int ic_step, const int m_block,
        const int mb_tail, const int n_block, const bool use_inversion) {
    vector<int> kw_ow_b(jcp_.kw, -1);
    vector<int> kw_ow_e(jcp_.kw, -1);
    const auto DW = jcp_.dilate_w + 1;

    for (int ow = 0; ow < jcp_.ow; ow++) {
        const auto iiw = ow * jcp_.stride_w - jcp_.l_pad;
        const auto kw_s = div_up(nstl::max(0, -iiw), DW);
        const auto kw_f = jcp_.kw
                - div_up(nstl::max(0, iiw - jcp_.iw + (jcp_.kw - 1) * DW + 1),
                        DW);
        for (int kw = 0; kw < jcp_.kw; kw++) {
            if (kw >= kw_s && kw < kw_f) {
                const auto inv_kw = use_inversion ? jcp_.kw - 1 - kw : kw;
                kw_ow_b[inv_kw] = kw_ow_b[inv_kw] == -1 ? ow : kw_ow_b[inv_kw];
                kw_ow_e[inv_kw] = ow + 1;
            }
        }
    }

    for (int kw = 0; kw < jcp_.kw; kw++) {
        const auto ow_b = kw_ow_b[kw];
        const auto ow_e = kw_ow_e[kw];

        if (ow_b < ow_e) {
            zero_accumulators(m_block, n_block);
            kdh_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block);
            store_accumulators(m_block, n_block, ow_b, ow_e);
        }
        add(reg_in,
                jcp_.prop_kind == backward_data ? inp_kw_sz_ * jcp_.stride_w
                                                : inp_kw_sz_);
    }

    copy_ow(m_block, n_block, 0, jcp_.ow);
}
template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::kw_loop_base(const int icb,
        const int icb_tail, const int ic_step, const int m_block,
        const int mb_tail, const int n_block) {
    Xbyak::Label label_kw_loop, label_loop_end;
    mov(reg_kw_l, ptr[param1 + GET_OFF(kw_l)]);

    zero_accumulators(m_block, n_block);

    L_aligned(label_kw_loop);
    {
        cmp(reg_kw_l, 0);
        je(label_loop_end, T_NEAR);
        kdh_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block);
        add(reg_in,
                jcp_.prop_kind == backward_data ? inp_kw_sz_ * jcp_.stride_w
                                                : inp_kw_sz_);
        dec(reg_kw_l);
        jmp(label_kw_loop, T_NEAR);
    }
    L_aligned(label_loop_end);

    store_accumulators(m_block, n_block, 0, 1);
}

template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::kw_loop(const int icb,
        const int icb_tail, const int ic_step, const int m_block,
        const int mb_tail, const int n_block, const bool use_inversion) {

    if (jcp_.prop_kind == backward_data) {
        if (jcp_.exec_type == exec_trans)
            bwd_kw_iw_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block);
        else
            kw_loop_base(icb, icb_tail, ic_step, m_block, mb_tail, n_block);
    } else {
        if (jcp_.exec_type == exec_vpad)
            kw_loop_base(icb, icb_tail, ic_step, m_block, mb_tail, n_block);
        else
            fwd_kw_ow_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block,
                    use_inversion);
    }
}

template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::load_params() {
    mov(reg_in, ptr[param1 + GET_OFF(ptr_in)]);
    mov(reg_zp_comp_out, ptr[param1 + GET_OFF(ptr_zp_out)]);
    mov(reg_comp_out, ptr[param1 + GET_OFF(ptr_cp_out)]);
}

template <typename Vmm>
int jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::compute_ic_step(
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
                * (jcp_.prop_kind == backward_data ? jcp_.ic_block
                                                   : jcp_.oc_block)
                * last_ic_block_;
        if (block_footprint <= static_cast<float>(
                    platform::get_per_core_cache_size(1))
                && (block_eff > best_block_eff)) {
            best_ic_step = ic_s;
            best_block_eff = block_eff;
        }
    }

    return best_ic_step;
}

template <typename Vmm>
void jit_uni_brgemm_conv_comp_pad_kernel_t<Vmm>::generate() {
    preamble();

    load_params();

    // fill registers with byte ones
    const auto reg32_scratch = reg_tmp.cvt32();
    mov(reg32_scratch, 0x1010101);
    uni_vpbroadcastd(vmm_one_bytes, reg32_scratch);

    // fill register with -128 && -1
    mov(reg32_scratch, -128);
    uni_vpbroadcastd(vmm_cp_shift, reg32_scratch);

    mov(reg32_scratch, -1);
    uni_vpbroadcastd(vmm_zp_shift, reg32_scratch);

    const bool is_int8_avx512_core = utils::one_of(jcp_.src_dt, s8, u8)
            && jcp_.wei_dt == s8 && !jcp_.has_int8_vnni;
    if (is_int8_avx512_core) {
        mov(reg_tmp.cvt16(), 0x1);
        vpbroadcastw(zmm_one_words, reg_tmp.cvt16());
    }

    const int max_regs = isa_max_regs
            - (is_int8_avx512_core ? 6
                                   : (jcp_.s8s8_compensation_required ? 4 : 3));
    const int nb = div_up(
            nstl::min(jcp_.prop_kind == backward_data ? jcp_.ic : jcp_.oc,
                    jcp_.prop_kind == backward_data ? jcp_.ic_block
                                                    : jcp_.oc_block),
            m_block2_);
    const int nb2 = nb / n_max_regs_;
    const int nb2_tail = nb % n_max_regs_;
    const int n_block = (nb2 == 0) ? nstl::max(1, nb2_tail) : n_max_regs_;

    const size_t m_max_regs = max_regs / n_block;
    const int m_block = nstl::min(m_max_regs, nb_ic_);
    const int ic_step = compute_ic_step(m_max_regs, m_block, n_block);

    assert(m_block * n_block <= max_regs);

    const auto blocks = m_block * ic_step;
    const auto icb = nb_ic_ / blocks;
    const auto icb_tail = nb_ic_ % blocks;
    const auto mb_tail = div_up(icb_tail, ic_step);

    Xbyak::Label label_kw_without_inversion, label_done;

    mov(reg_use_inversion, ptr[param1 + GET_OFF(use_inversion)]);
    cmp(reg_use_inversion, 0);
    jz(label_kw_without_inversion, T_NEAR);
    kw_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block, true);
    jmp(label_done, T_NEAR);

    L_aligned(label_kw_without_inversion);
    kw_loop(icb, icb_tail, ic_step, m_block, mb_tail, n_block, false);

    L_aligned(label_done);

    postamble();
}
template <typename Vmm>
size_t jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Vmm>::out_oc_offset(
        const int n, const int w) const {
    return static_cast<size_t>(out_dsz_) * n * inp_oc_block_ + w * out_ow_sz_;
}
template <typename Vmm>
size_t jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Vmm>::inp_ic_offset(
        const int kw, const int ic, const int n) const {
    return static_cast<size_t>(kw) * inp_kw_sz_ + n * inp_oc_sz_
            + ic * inp_ic_sz_;
}

template <typename Vmm>
Vmm jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Vmm>::accum(
        const int n, const bool has_s8s8_shift) const {
    return has_s8s8_shift ? Vmm(n_max_regs_ + n) : Vmm(n);
}
template <typename Vmm>
void jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Vmm>::zero_accumulators(
        const int n_block) {
    for (int n = 0; n < n_block; ++n) {
        auto vmm = accum(n);
        uni_vpxor(vmm, vmm, vmm);
    }
}
template <typename Vmm>
void jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Vmm>::store_accumulators(
        const int n_block, const int ow_b, const int ow_e) {
    if (jcp_.src_zero_point) {
        for_(int n = 0; n < n_block; n++)
        for (int w = ow_b; w < ow_e; w++) {
            auto vmm = accum(n);
            const auto offset = out_oc_offset(n, w);
            auto zp_addr = ptr[reg_aux_zp_comp_out + offset];
            vmovups(zp_addr, vmm);
        }
    }

    if (jcp_.s8s8_compensation_required) {
        for_(int n = 0; n < n_block; n++)
        for (int w = ow_b; w < ow_e; w++) {
            auto vmm = accum(n, true);
            const auto offset = out_oc_offset(n, w);
            auto cp_addr = ptr[reg_aux_comp_out + offset];
            vmovups(cp_addr, vmm);
        }
    }
}

template <typename Vmm>
void jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Vmm>::store(
        const int n_block, const int ow_b, const int ow_e) {
    mov(reg_aux_zp_comp_out, reg_zp_comp_out);
    mov(reg_aux_comp_out, reg_comp_out);
    mov(reg_ker_l, ptr[param1 + GET_OFF(ker_l)]);

    Xbyak::Label label_ker_loop, label_done;
    L_aligned(label_ker_loop);
    {
        cmp(reg_ker_l, 0);
        je(label_done, T_NEAR);
        store_accumulators(n_block, ow_b, ow_e);
        if (jcp_.src_zero_point) add(reg_aux_zp_comp_out, out_ker_sz_);
        if (jcp_.s8s8_compensation_required) add(reg_aux_comp_out, out_ker_sz_);
        dec(reg_ker_l);
        jmp(label_ker_loop, T_NEAR);
    }
    L_aligned(label_done);
}

template <typename Vmm>
void jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Vmm>::kw_loop(
        const int n_block) {
    vector<int> ow_kw_b(jcp_.ow, -1);
    vector<int> ow_kw_e(jcp_.ow, -1);
    const auto DW = jcp_.dilate_w + 1;
    for (int ow = 0; ow < jcp_.ow; ow++) {
        const auto iiw = ow * jcp_.stride_w - jcp_.l_pad;
        const auto kw_s = div_up(nstl::max(0, -iiw), DW);
        const auto kw_f = jcp_.kw
                - div_up(nstl::max(0, iiw - jcp_.iw + (jcp_.kw - 1) * DW + 1),
                        DW);
        ow_kw_b[ow] = kw_s;
        ow_kw_e[ow] = kw_f;
    }

    for (int ow = 0; ow < jcp_.ow;) {
        const auto kw_b = ow_kw_b[ow];
        const auto kw_e = ow_kw_e[ow];
        int ow_e = ow + 1;
        while (ow_e < jcp_.ow) {
            if (ow_kw_b[ow_e] != kw_b || ow_kw_e[ow_e] != kw_e) break;
            ow_e++;
        }
        if (kw_b < kw_e) {
            zero_accumulators(n_block);
            compute(n_block, kw_b, kw_e);
            store(n_block, ow, ow_e);
        }
        ow = ow_e;
    }
}

template <typename Vmm>
void jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Vmm>::compute(
        const int n_block, const int kw_b, const int kw_e) {
    Xbyak::Label label_kh_loop, label_end;
    mov(reg_kh_l, ptr[param1 + GET_OFF(kh_l)]);
    mov(reg_aux_in, reg_in);

    L_aligned(label_kh_loop);
    {
        cmp(reg_kh_l, 0);
        je(label_end, T_NEAR);
        for_(int kw = kw_b; kw < kw_e; kw++)
        for_(int n = 0; n < n_block; n++)
        for (int ic = 0; ic < jcp_.ic; ic++) {
            auto vmm = accum(n);
            const auto offs = inp_ic_offset(kw, ic, n);
            auto addr = EVEX_compress_addr(reg_aux_in, offs);
            vpmovsxbd(vmm_tmp, addr);
            vpsubd(vmm, vmm, vmm_tmp);
        }
        add(reg_aux_in, inp_kh_sz_);
        dec(reg_kh_l);
        jmp(label_kh_loop, T_NEAR);
    }
    L_aligned(label_end);

    // Apply s8s8 shift to accumulators
    if (jcp_.s8s8_compensation_required) {
        for (int n = 0; n < n_block; n++) {
            vpmulld(accum(n, true), accum(n), vmm_cp_shift);
        }
    }
}

template <typename Vmm>
void jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Vmm>::load_params() {
    mov(reg_in, ptr[param1 + GET_OFF(ptr_in)]);
    mov(reg_zp_comp_out, ptr[param1 + GET_OFF(ptr_zp_out)]);
    mov(reg_comp_out, ptr[param1 + GET_OFF(ptr_cp_out)]);
    mov(reg_last_ocb, ptr[param1 + GET_OFF(last_ocb)]);
}

template <typename Vmm>
void jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Vmm>::generate() {
    preamble();

    load_params();

    // fill registers with byte 128
    const auto reg32_scratch = reg_tmp.cvt32();
    mov(reg32_scratch, 128);
    uni_vpbroadcastd(vmm_cp_shift, reg32_scratch);

    const int last_oc_block = nstl::min(
            jcp_.oc_block, jcp_.oc - (jcp_.nb_oc - 1) * jcp_.oc_block);
    const int max_n_block = div_up(jcp_.oc_block, inp_oc_block_);
    const int last_n_block = div_up(last_oc_block, inp_oc_block_);

    Xbyak::Label label_last_ocb, label_done;

    cmp(reg_last_ocb, 0);
    jnz(label_last_ocb, T_NEAR);
    kw_loop(max_n_block);
    jmp(label_done, T_NEAR);

    L_aligned(label_last_ocb);
    kw_loop(last_n_block);

    L_aligned(label_done);

    postamble();
}

template <typename Vmm>
jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Vmm>::
        jit_uni_brgemm_conv_relo_comp_pad_kernel_t(
                const jit_brgemm_conv_conf_t &ajcp)
    : jit_generator(jit_name())
    , jcp_(ajcp)
    , inp_dsz_(jcp_.wei_dsz)
    , out_dsz_(jcp_.acc_dsz)
    , inp_oc_block_(static_cast<size_t>(16))
    , inp_ic_sz_(static_cast<size_t>(inp_dsz_) * inp_oc_block_)
    , inp_kw_sz_(static_cast<size_t>(inp_ic_sz_) * jcp_.ic
              * (jcp_.is_relo_whi() ? jcp_.kh : 1))
    , inp_kh_sz_(static_cast<size_t>(inp_ic_sz_) * jcp_.ic
              * (jcp_.is_relo_whi() ? 1 : jcp_.kw))
    , inp_oc_sz_(static_cast<size_t>(inp_ic_sz_) * jcp_.ic * jcp_.kh * jcp_.kw)
    , out_ow_sz_(static_cast<size_t>(out_dsz_) * jcp_.oc_block)
    , out_ker_sz_(static_cast<size_t>(out_ow_sz_) * jcp_.ow)
    , isa_max_regs_(isa_num_vregs(jcp_.isa)) {}

template struct jit_uni_brgemm_conv_comp_pad_kernel_t<Xbyak::Zmm>;
template struct jit_uni_brgemm_conv_comp_pad_kernel_t<Xbyak::Ymm>;
template struct jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Xbyak::Zmm>;
template struct jit_uni_brgemm_conv_relo_comp_pad_kernel_t<Xbyak::Ymm>;

} // namespace jit_uni_brgemm_conv_comp_pad_kernel

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
