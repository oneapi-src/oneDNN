/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
* Copyright 2021-2024 FUJITSU LIMITED
* Copyright 2022 Arm Ltd. and affiliates
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"

#define IDX(a) static_cast<uint32_t>(a.getIdx())

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace eltwise_injector {

bool is_isa_supported(cpu_isa_t isa) {
    return is_superset(isa, sve_128);
}

bool is_alg_supported(alg_kind_t alg) {
    using namespace alg_kind;
    return utils::one_of(alg, eltwise_relu, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
            eltwise_soft_relu, eltwise_logistic, eltwise_mish, eltwise_exp,
            eltwise_gelu_tanh, eltwise_hardsigmoid, eltwise_hardswish,
            eltwise_swish, eltwise_log, eltwise_clip, eltwise_clip_v2,
            /*eltwise_pow, */ eltwise_gelu_erf, eltwise_round,
            eltwise_relu_use_dst_for_bwd, eltwise_tanh_use_dst_for_bwd,
            eltwise_elu_use_dst_for_bwd, eltwise_sqrt_use_dst_for_bwd,
            eltwise_logistic_use_dst_for_bwd, eltwise_exp_use_dst_for_bwd,
            eltwise_clip_v2_use_dst_for_bwd);
}

bool is_supported(cpu_isa_t isa, alg_kind_t alg) {
    return is_isa_supported(isa) && is_alg_supported(alg);
}

} // namespace eltwise_injector

using namespace Xbyak_aarch64;

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    using namespace alg_kind;
    using namespace Xbyak_aarch64::util;
    p_all = h->P_ALL_ONE;
    preserved_vecs_count = 0;
    vecs_to_preserve = aux_vecs_count();
    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin()) + 1;
    start_idx_tail = vmm_idxs.begin();

    for (size_t idx = preserved_vecs_count; idx < vecs_count; idx++) {
        if (preserved_vecs_count >= vecs_to_preserve) break;
        if (start_idx <= idx && idx < end_idx) continue;

        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    size_t preserved_vecs_count_tail = vecs_to_preserve - preserved_vecs_count;
    for (size_t i = 0; i < preserved_vecs_count_tail; i++) {
        preserved_vec_idxs[preserved_vecs_count++] = *start_idx_tail;
        ++start_idx_tail;
    }

    assert(preserved_vecs_count == vecs_to_preserve);

    // Same logic but to allocate gprs
    size_t preserved_gprs_count = 0;
    for (size_t gpr_idx = 0; gpr_idx <= 30; ++gpr_idx) {
        int _idx = 30 - gpr_idx; // we allocate from the end
        if (preserved_gprs_count < aux_gprs_count()
                && (((unsigned)_idx) != x_table.getIdx()))
            preserved_gpr_idxs[preserved_gprs_count++] = _idx;
    }
    assert(preserved_gprs_count == aux_gprs_count());

    if (save_state_) {
        const int reg_size = h->x0.getBit() / 8;
        if (preserve_p_table_) h->str(x_table, pre_ptr(h->X_SP, -reg_size));
        for (size_t i = 0; i < preserved_gprs_count; ++i)
            h->str(XReg(preserved_gpr_idxs[i]), pre_ptr(h->X_SP, -reg_size));

        if (preserve_vmm_) {
            if (preserved_vecs_count)
                h->sub_imm(h->X_SP, h->X_SP, preserved_vecs_count * vlen,
                        h->X_TMP_0);
            for (size_t i = 0; i < preserved_vecs_count; ++i)
                h->str(ZReg(preserved_vec_idxs[i]), ptr(h->X_SP, i, MUL_VL));
        }
        load_table_addr();
    }

    assign_regs();
    set_coef_to_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble_tail(
        const injector_utils::vmm_index_set_iterator_t start_idx_it) {
    size_t tail_vecs_to_preserve = std::distance(start_idx_it, start_idx_tail);
    if (tail_vecs_to_preserve == 0) return;

    const int idx_off = vecs_to_preserve - tail_vecs_to_preserve;

    if (save_state_) {
        if (idx_off) h->add_imm(h->X_SP, h->X_SP, idx_off * vlen, h->X_TMP_0);

        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->ldr(ZReg(preserved_vec_idxs[idx_off + i]),
                    ptr(h->X_SP, i, MUL_VL));
    }

    for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
        preserved_vec_idxs[idx_off + i] += tail_vecs_to_preserve;

    if (save_state_ && preserve_vmm_) {
        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->str(ZReg(preserved_vec_idxs[idx_off + i]),
                    ptr(h->X_SP, i, MUL_VL));

        if (idx_off) h->sub_imm(h->X_SP, h->X_SP, idx_off * vlen, h->X_TMP_0);
    }

    assign_regs();
    set_coef_to_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_postamble() {
    using namespace Xbyak_aarch64::util;
    const int reg_size = h->x0.getBit() / 8;
    if (!save_state_) return;

    if (preserve_vmm_) {
        for (size_t i = 0; i < preserved_vecs_count; ++i)
            h->ldr(ZReg(preserved_vec_idxs[i]), ptr(h->X_SP, i, MUL_VL));

        if (preserved_vecs_count)
            h->add_imm(
                    h->X_SP, h->X_SP, preserved_vecs_count * vlen, h->X_TMP_0);
    }

    for (int i = aux_gprs_count() - 1; i >= 0; --i)
        h->ldr(XReg(preserved_gpr_idxs[i]), post_ptr(h->X_SP, reg_size));
    if (preserve_p_table_) h->ldr(x_table, post_ptr(h->X_SP, reg_size));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::assign_regs() {
    /* For translation of x64's memory operand instructions */
    z_tmp = TRegS(static_cast<uint32_t>(preserved_vec_idxs[0]));

    vmm_mask = TRegS(preserved_vec_idxs[1]);
    vmm_aux0 = TRegS(preserved_vec_idxs[1]);
    vmm_aux1 = TRegS(preserved_vec_idxs[2]);
    vmm_aux2 = TRegS(preserved_vec_idxs[3]);
    vmm_aux3 = TRegS(preserved_vec_idxs[4]);
    vmm_aux4 = TRegS(preserved_vec_idxs[5]);
    vmm_aux5 = TRegS(preserved_vec_idxs[6]);
    vmm_aux6 = TRegS(preserved_vec_idxs[7]);
    vmm_aux7 = TRegS(preserved_vec_idxs[8]);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::set_coef_to_regs() {
    using namespace alg_kind;

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu:
                if (alpha_ != 0.f) table_val(alpha, z_tmp);
                break;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: table_val(alpha, vmm_aux4); break;
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh:
            case eltwise_square:
            case eltwise_abs:
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: break;
            case eltwise_linear:
            case eltwise_clip:
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2:
            case eltwise_hardswish:
            case eltwise_hardsigmoid:
                table_val(alpha, z_tmp);
                table_val(beta, vmm_aux0);
                break;
            case eltwise_soft_relu:
            case eltwise_mish:
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic:
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp:
            case eltwise_gelu_tanh:
            case eltwise_swish:
            case eltwise_log:
            case eltwise_gelu_erf:
            case eltwise_round: break;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu:
            case eltwise_soft_relu: table_val(alpha, z_tmp); break;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu:
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh:
            case eltwise_square:
            case eltwise_abs:
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt:
            case eltwise_linear:
            case eltwise_mish:
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic:
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp:
            case eltwise_gelu_tanh:
            case eltwise_swish:
            case eltwise_log:
            case eltwise_gelu_erf: break;
            case eltwise_hardsigmoid:
            case eltwise_hardswish: h->fmov(vmm_aux1, 1.);
            case eltwise_clip:
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2:
                table_val(alpha, z_tmp);
                table_val(beta, vmm_aux0);
                break;
            default: assert(!"unsupported eltwise algorithm");
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_cmp_mask(
        const TRegS &vmm_src, const TRegS &compare_operand, int cmp_predicate) {
    enum {
        EQ_OQ = 0,
        LT_OS = 1,
        LE_OS = 2,
        UNORD_Q = 3,
        NEQ_UQ = 4,
        NLT_US = 5,
        NLE_US = 6,
        ORD_Q = 7,
        EQ_UQ = 8,
        NGE_US = 9,
        NGT_US = 10,
        FALSE_OQ = 11,
        NEQ_OQ = 12,
        GE_OS = 13,
        GT_OS = 14,
        TRUE_UQ = 15,
        EQ_OS = 16,
        LT_OQ = 17,
        LE_OQ = 18,
        UNORD_S = 19,
        NEQ_US = 20,
        NLT_UQ = 21,
        NLE_UQ = 22,
        ORD_S = 23,
        EQ_US = 24,
        NGE_UQ = 25,
        NGT_UQ = 26,
        FALSE_OS = 27,
        NEQ_OS = 28,
        GE_OQ = 29,
        GT_OQ = 30,
        TRUE_US = 31,
    };

    h->mov(PRegB(IDX(p_tmp0)), p_all / T_z, p_all.b);
    switch (cmp_predicate) {
        case EQ_OQ:
            h->fcmeq(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case LT_OS:
            h->fcmlt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case LE_OS:
            h->fcmle(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NEQ_UQ:
            h->fcmne(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NLT_US:
            h->fcmge(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NLE_US:
            h->fcmgt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case EQ_UQ:
            h->fcmeq(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NGE_US:
            h->fcmlt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NGT_US:
            h->fcmle(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NEQ_OQ:
            h->fcmne(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case GE_OS:
            h->fcmge(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case GT_OS:
            h->fcmgt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case EQ_OS:
            h->fcmeq(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case LT_OQ:
            h->fcmlt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case LE_OQ:
            h->fcmle(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NEQ_US:
            h->fcmne(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NLT_UQ:
            h->fcmge(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NLE_UQ:
            h->fcmgt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case EQ_US:
            h->fcmeq(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NGE_UQ:
            h->fcmlt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NGT_UQ:
            h->fcmle(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NEQ_OS:
            h->fcmne(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case GE_OQ:
            h->fcmge(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case GT_OQ:
            h->fcmgt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;

        case UNORD_Q:
        case ORD_Q:
        case FALSE_OQ:
        case TRUE_UQ:
        case UNORD_S:
        case ORD_S:
        case FALSE_OS:
        case TRUE_US:
        default: assert(!"Unsupported compare mode"); break;
    }
}

// Uses injector masks objects: p_mask
// Blends a result of second input into a first input w/ a stored mask.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::blend_with_mask(
        const TRegS &vmm_dst, const TRegS &src) {
    h->sel(vmm_dst, p_mask / T_m, src, vmm_dst);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_fwd(
        const TRegS &vmm_src) {

    const auto &t0 = ZRegS(IDX(vmm_src));
    const auto &t1 = ZRegS(IDX(vmm_aux1));
    const auto &t2 = ZRegS(IDX(vmm_aux2));
    h->fmin(t0, p_all, ZRegS(IDX(table_val(exp_ln_flt_max_f, z_tmp))));
    h->fmax(t0, p_all, ZRegS(IDX(table_val(exp_ln_flt_min_f, z_tmp))));
    h->fmul(t0, t0, ZRegS(IDX(table_val(exp_log2ef, z_tmp))));
    h->movprfx(t1, p_all, t0);
    h->frintm(t1, p_all, t0);
    h->fcvtzs(t2, p_all, t1);
    h->fsub(t1, t0, t1);
    h->fadd(t0, t1, ZRegS(IDX(table_val(one, z_tmp))));
    h->lsr(t1, t0, 17);
    h->fexpa(t1, t1);
    h->fscale(t1, p_all, t2);
    h->and_(ZRegD(t2.getIdx()), ZRegD(t0.getIdx()),
            ZRegD(IDX(table_val(exp_not_mask17, z_tmp))));
    h->fsub(t2, t0, t2);
    h->movprfx(t0, p_all, ZRegS(IDX(table_val(exp_coeff2, z_tmp))));
    h->fmad(t0, p_all, t2, ZRegS(IDX(table_val(exp_coeff1, z_tmp))));
    h->fmad(t0, p_all, t2, ZRegS(IDX(table_val(one, z_tmp))));
    h->fmul(t0, t1, t0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->mov(ZRegD(vmm_aux0.getIdx()), ZRegD(vmm_src.getIdx()));
    h->fcmgt(p_mask.s, p_all, vmm_src, 0.0);
    h->fmul(vmm_src, vmm_src, z_tmp);
    h->sel(vmm_src, p_mask, vmm_aux0, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_zero_ns_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->fmaxnm(vmm_src, p_all, 0.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_fwd(
        const TRegS &vmm_src) {
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->mov(ZRegD(vmm_aux3.getIdx()), ZRegD(vmm_src.getIdx()));

    // compute exponent
    exp_compute_vector_fwd(vmm_src);

    // alpha * (exp(x) - 1)
    h->fsub(vmm_src, p_all / T_m, 1.);
    h->fmul(vmm_src, vmm_src, vmm_aux4);

    // combine with mask
    h->fcmgt(p_mask.s, p_all / T_z, vmm_aux3, 0.);
    h->mov(vmm_src, p_mask / T_m, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<
        isa>::tanh_polynomial_approx_compute_vector_fwd(const TRegS &vmm_src) {

    if (!utils::one_of(isa, sve_512)) return;

    using namespace Xbyak_aarch64::util;

    const int tanh_n_polynomials = 32;

    // Register mapping
    TRegS vmm_dst = vmm_aux1, vmm_src_shift = vmm_aux1, vmm_coeff = vmm_aux1,
          vmm_pol = vmm_aux2, vmm_indices = vmm_aux3, vmm_tmp = vmm_aux3,
          vmm_src_pos = vmm_aux4, vmm_sign = vmm_aux4;

    const auto &mask = PReg(6); // avoid pred regs used in *conv_kernel*

    // Helper function to gather polynomial coefficients
    auto gather_coefficient = [&](TRegS vmm_coeff, int coeff_idx,
                                      TRegS vmm_pol_idx) {
        h->add_imm(h->X_TMP_1, x_table,
                table_off(tanh_pol_table, coeff_idx * tanh_n_polynomials),
                h->X_TMP_0);
        h->ld1w(ZRegS(IDX(vmm_coeff)), p_all,
                ptr(h->X_TMP_1, ZRegS(IDX(vmm_pol_idx)), SXTW));
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x postive
    // and reapply sign at the end
    h->fabs(vmm_src_pos, p_all / T_z, vmm_src);

    // Compute indices for the table lookup
    h->sub(ZRegS(IDX(vmm_indices)), ZRegS(IDX(vmm_src_pos)),
            ZRegS(IDX(table_val(tanh_idx_bias, z_tmp))));
    h->and_(ZRegD(IDX(vmm_indices)), ZRegD(IDX(vmm_indices)),
            ZRegD(IDX(table_val(tanh_idx_mask, z_tmp))));
    h->lsr(ZRegD(IDX(vmm_indices)), ZRegD(IDX(vmm_indices)), 20);

    // Argument reduction
    h->and_(ZRegD(IDX(vmm_src_shift)), ZRegD(IDX(vmm_src_pos)),
            ZRegD(IDX(table_val(tanh_idx_mask, z_tmp))));
    h->fsub(vmm_src_pos, vmm_src_pos, ZRegS(IDX(vmm_src_shift)));

    gather_coefficient(vmm_pol, 6, vmm_indices);
    for (int deg = 5; deg >= 0; --deg) {
        gather_coefficient(vmm_coeff, deg, vmm_indices);
        h->fmad(vmm_pol, p_all / T_m, vmm_src_pos, vmm_coeff);
    }

    // Restore src_pos
    h->fabs(vmm_src_pos, p_all / T_z, vmm_src);

    // Now Blend the results
    // [saturation_ubound; +inf] : return +/- 1
    table_val(one, vmm_dst);

    // [linear_ubound; saturation_lbound] :  return +/- P(x)
    table_val(tanh_saturation_lbound, vmm_tmp);
    h->fcmgt(PRegS(IDX(mask)), p_all / T_z, vmm_tmp, vmm_src_pos);
    h->sel(vmm_dst, mask / T_m, vmm_pol, vmm_dst);

    // [0; linear_ubound]  :  return x
    table_val(tanh_linear_ubound, vmm_tmp);
    h->fcmgt(PRegS(IDX(mask)), p_all / T_z, vmm_tmp, vmm_src_pos);
    h->sel(vmm_dst, mask / T_m, vmm_src_pos, vmm_dst);

    // Reapply sign and return
    h->and_(ZRegD(IDX(vmm_sign)), ZRegD(IDX(vmm_src)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_dst)), ZRegD(IDX(vmm_sign)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_fwd(
        const TRegS &vmm_src) {

    if (utils::one_of(isa, sve_512)) {
        tanh_polynomial_approx_compute_vector_fwd(vmm_src);
        return;
    }

    // tanh(x) = x(1 + (-1/3)x^2) for |x| < tanh_range
    // tanh(x) = 1 - 2/(1 + exp(2 x)) for otherwise

    const auto &t0 = ZRegS(IDX(vmm_src));
    const auto &t1 = ZRegS(IDX(vmm_aux1));
    const auto &t2 = ZRegS(IDX(vmm_aux2));
    const auto &t3 = ZRegS(IDX(vmm_aux3));
    const auto &oneS = ZRegS(IDX(vmm_aux4));
    const auto &mask = PReg(6); // avoid pred regs used in *conv_kernel*

    h->fcpy(oneS, p_all, 1);
    // make mask for small x
    h->mov(t3, p_all, t0);
    h->fabs(t1, p_all, t0);
    h->cmplt(mask.s, p_all, t1, ZRegS(IDX(table_val(tanh_range, z_tmp))));

    // 2x
    h->fadd(t0, t0, t0);
    // exp(2x)
    exp_compute_vector_fwd(t0);
    // 1+exp(2x)
    h->fadd(t0, t0, oneS);
    // 1/(1+exp(2x))
    // 1st aprox ; a = 1/x + e
    h->frecpe(t1, t0);
    // 2nd aprox ; a' = (2 - ax)a = 1/x - e^2 x
    h->frecps(t2, t0, t1);
    h->fmul(t2, t2, t1);
    // 3rd aprox ; a'' = (2 - a'x)a'
    h->frecps(t0, t0, t2);
    h->fmul(t0, t0, t2);

    // 2/(1+exp(2x))
    h->fadd(t0, t0, t0);
    // 1-2/(1+exp(2x))
    h->fsub(t0, oneS, t0);

    // tanh(x) = x(1 - x^2/3) for |x| < tanh_range
    h->fmul(t1, t3, t3);
    h->fmad(t1, p_all, ZRegS(IDX(table_val(tanh_m1d3, z_tmp))), oneS);
    h->fmul(t1, p_all, t3);
    // select the correct value according to mask
    h->mov(t0, mask, t1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->mov(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(vmm_src)));

    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    h->fmul(vmm_src, vmm_src, vmm_src);
    h->mov(ZRegD(IDX(vmm_aux1)),
            ZRegD(IDX(table_val(gelu_tanh_fitting_const, z_tmp))));
    /* Do not use 1.f, which is a float constant,
       but 1., which is a double constant. */
    h->fmov(z_tmp, 1.);
    h->fmad(vmm_src, p_all / T_m, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));
    h->fmul(vmm_src, vmm_src, vmm_aux0);
    h->fmul(vmm_src, vmm_src,
            ZRegS(IDX(table_val(gelu_tanh_sqrt_two_over_pi, z_tmp))));

    // save x on stack as tanh uses vmm_aux0
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->str(ZReg(IDX(vmm_aux0)), ptr(h->X_TMP_0));

    // compute tanh(G(x))
    tanh_compute_vector_fwd(vmm_src);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux0)), ptr(h->X_TMP_0));
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    // compute 0.5 * x * (1 + tanh(G(x)))
    h->fadd(vmm_src, p_all / T_m, 1.);
    h->fmul(vmm_src, p_all / T_m, 0.5f);
    h->fmul(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->fmul(vmm_src, vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->fabs(vmm_src, p_all / T_m, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->fsqrt(vmm_src, p_all / T_m, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_fwd(
        const TRegS &vmm_src) {
    // compute x = alpha * x + beta;
    h->fmad(vmm_src, p_all / T_m, z_tmp, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->fmaxnm(vmm_src, p_all, z_tmp);
    h->fminnm(vmm_src, p_all, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::mish_compute_vector_fwd(
        const TRegS &vmm_src) {
    // An equation other than mish(x) = x*tanh(srelu(x)) was used
    // to calculate mish, but it should be remembered that it is equivalent
    // equation, it uses the following rule:
    // tanh(x) = (e^x - e^-x) / (e^x + e^-x),
    // hence the equation for mish can take the form:
    // mish(x) = x * ((e^x + 1)^2 - 1)/((e^x + 1)^2 + 1).
    // This option was chosen because computing tanh requires more registers
    // than exp, and also requires more constants to be stored in memory,
    // making the algorithm slower.

    // IMPORTANT: we use vmm_aux3 to save src as exp does not use it.
    h->mov(ZRegD(vmm_aux3.getIdx()), ZRegD(vmm_src.getIdx())); // vmm_aux3 = x
    h->fminnm(vmm_src, p_all / T_m,
            table_val(fwd_mish_max_x_for_equation_f, z_tmp));
    exp_compute_vector_fwd(vmm_src);

    // (e^x+1)^2
    h->fadd(vmm_src, p_all / T_m, 1.);
    h->fmul(vmm_src, vmm_src, vmm_src);

    // save (e^x+1)^2 as it appears in both the denominator and the numerator
    h->mov(ZRegD(vmm_aux1.getIdx()), ZRegD(vmm_src.getIdx()));

    // x * ((e^x + 1)^2 - 1) / ((e^x + 1)^2 + 1)
    h->fsub(vmm_src, p_all / T_m, 1.);
    h->fadd(vmm_aux1, p_all / T_m, 1.);
    h->fdiv(vmm_src, p_all / T_m, vmm_aux1);
    h->fmul(vmm_src, vmm_src, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::hardswish_compute_vector_fwd(
        const TRegS &vmm_src) {
    // result = x * hardsigmoid(x)
    h->mov(ZRegD(vmm_aux1.getIdx()), ZRegD(vmm_src.getIdx()));
    hardsigmoid_compute_vector_fwd(vmm_src);
    h->fmul(vmm_src, vmm_src, vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::hardsigmoid_compute_vector_fwd(
        const TRegS &vmm_src) {
    // alpha:z_tmp, beta:vmm_aux0
    // result = max(0, min(1, alpha * x + beta))
    h->fmul(vmm_src, vmm_src, z_tmp);
    h->fadd(vmm_src, vmm_src, vmm_aux0);
    h->fminnm(vmm_src, p_all / T_m, 1.);
    h->fmaxnm(vmm_src, p_all / T_m, 0.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_fwd(
        const TRegS &vmm_src) {
    // alpha scaling
    if (alpha_ == 1.f) {
        // Nothing to do
    }
    if (alpha_ == 0.5 || alpha_ == 2.0)
        h->fmul(vmm_src, p_all / T_m, alpha_);
    else
        h->fmul(vmm_src, vmm_src, table_val(alpha, z_tmp));

    // ln(1 + exp(x)) =
    // = ln(1 + exp(n * ln(2) + r)) // divide x by ln(2) and get quot and rem
    // = ln(1 + 2^n * exp(r)) // simplify the exp(n*ln(2)) expression
    // = ln(2 ^ 0 + 2^n * exp(r)) // note 1 = 2^0
    // = ln(2 ^ (n - n) + 2^n * exp(r)) // 2^0 = 2^(n-n)
    // = ln(2 ^ n * (2^-n + exp(r))) // factorize with 2^n
    // = n * ln(2) + ln(2^-n + exp(r)) // take the 2^n factor out of the ln

    // keep src for further computations
    h->mov(ZRegD(IDX(vmm_aux2)), ZRegD(IDX(vmm_src)));

    h->fminnm(ZRegS(IDX(table_val(exp_ln_flt_max_f, z_tmp))), p_all, vmm_src);
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(z_tmp)));
    h->fmaxnm(ZRegS(IDX(table_val(exp_ln_flt_min_f, z_tmp))), p_all, vmm_src);

    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(z_tmp)));
    h->mov(ZRegD(IDX(vmm_aux1)), ZRegD(IDX(vmm_src)));

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(exp_log2ef, z_tmp))));
    h->fadd(vmm_src, p_all / T_m, 0.5f);

    // tmp = floorf(fx)
    h->frintm(vmm_aux0, p_all / T_m, vmm_src);

    // keep vmm_src = fx for further computations
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux0)));

    // x = x - fx * ln2
    h->fmul(vmm_aux0, vmm_aux0, ZRegS(IDX(table_val(ln2f, z_tmp))));
    h->fsub(vmm_aux1, vmm_aux1, vmm_aux0);
    // compute exponent polynomial
    h->mov(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(table_val(exp_pol, z_tmp, 4))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1,
            ZRegS(IDX(table_val(exp_pol, z_tmp, 3))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1,
            ZRegS(IDX(table_val(exp_pol, z_tmp, 2))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1,
            ZRegS(IDX(table_val(exp_pol, z_tmp, 1))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1,
            ZRegS(IDX(table_val(exp_pol, z_tmp, 0))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));

    // We do not count 2^-n here, because n can reach 128 and 2^(-128) is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^-n + exp(r) will be counted (2^-(n-1) + 2*exp(r))/2, because 2^(-127)
    // and 2 are numbers representable in fp32.

    // compute 2^-(n-1)
    // vmm_src now represents n-1
    h->fsub(vmm_src, p_all / T_m, 1.);
    h->fneg(vmm_aux1, p_all / T_m, vmm_src);

    h->frinti(vmm_aux1, p_all / T_m, vmm_aux1);
    h->fcvtzs(vmm_aux1, p_all / T_m, vmm_aux1);
    // restore vmm_src to n
    h->fadd(vmm_src, p_all / T_m, 1.);

    h->add(vmm_aux1, vmm_aux1, ZRegS(IDX(table_val(exponent_bias, z_tmp))));
    h->lsl(vmm_aux1, vmm_aux1, n_mantissa_bits);
    // calculate ln(1 + y)
    h->fmul(vmm_aux3, p_all / T_m, 2.); // 2*exp(r)
    h->fadd(vmm_aux3, vmm_aux3,
            vmm_aux1); // 2^-(n-1) + 2*exp(r)
    h->fmul(vmm_aux3, p_all / T_m, 0.5); // (2^-(n-1) + 2*exp(r))/2

    // frexp()
    h->lsr(vmm_src, vmm_aux3, n_mantissa_bits);
    h->scvtf(vmm_src, p_all / T_m, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->fsub(vmm_src, vmm_src,
            ZRegS(IDX(table_val(soft_relu_one_twenty_six, z_tmp))));

    // and with mask (to get 0.5 * mantissa)
    h->and_(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_aux3)),
            ZRegD(IDX(table_val(soft_relu_mantissa_sign_mask, z_tmp))));
    // got y. (mantisa)  0.5 < y < 1 (or with (to get 0.5 * mantissa))
    h->orr(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_aux3)),
            ZRegD(IDX(table_val(half, z_tmp))));
    // y  = y - 1
    h->fsub(vmm_aux3, p_all / T_m, 1.);

    // compute log1p polynomial
    h->mov(ZRegD(IDX(vmm_aux1)),
            ZRegD(IDX(table_val(soft_relu_pol, z_tmp, 8))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 7))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 6))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 5))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 4))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 3))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 2))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 1))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 0))));
    //calculate ln(2) * n
    h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(ln2f, z_tmp))));
    h->fadd(vmm_src, vmm_src, vmm_aux1);
    h->fadd(vmm_src, vmm_src, vmm_aux0);

    // get vmm_mask = src > max logf
    // y = (x < max log f) ? soft_relu(x) : x
    compute_cmp_mask(vmm_aux2, table_val(exp_ln_flt_max_f, z_tmp), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux2);
    if (alpha_ == 1.f) { // standard soft_relu case
        // Skip an instruction.
    } else if (alpha_ == -1) { // logsigmoid case
        /* Do not use -1.f, which is a float constant,
       but -1., which is a double constant. */
        h->fmov(z_tmp, -1.);
        h->fmul(vmm_src, vmm_src, z_tmp);
    } else { // General case.
        h->fdiv(vmm_src, p_all / T_m, table_val(alpha, z_tmp));
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_fwd(
        const TRegS &vmm_src) {
    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->mov(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_src)));
    // we store the original sign and make x negative
    h->and_(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_aux3)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));
    h->orr(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));

    exp_compute_vector_fwd(vmm_src);

    // dup exp(x)
    h->mov(ZRegD(IDX(vmm_aux1)), ZRegD(IDX(vmm_src)));
    // (exp(x) + 1)
    h->fadd(vmm_aux1, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));
    // y = exp(x) / (exp(x) + 1)
    h->fdiv(vmm_src, p_all, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    h->mov(ZRegD(IDX(vmm_aux2)), ZRegD(IDX(table_val(one, z_tmp))));
    h->fsub(vmm_aux2, vmm_aux2, vmm_src);

    h->and_(ZRegD(IDX(z_tmp)), ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_aux3)));
    h->cmpne(PRegS(IDX(p_mask)), p_all / T_z, z_tmp, 0);

    blend_with_mask(vmm_aux2, vmm_src);

    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux2)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_fwd(
        const TRegS &vmm_src) {
    // Save src data on stack for later usage
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->str(ZReg(IDX(vmm_src)), ptr(h->X_TMP_0));
    // x*alpha
    h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(alpha, z_tmp))));
    // sigmoid(x*alpha)
    logistic_compute_vector_fwd(vmm_src);
    // x*sigmoid(alpha*x)
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux0)), ptr(h->X_TMP_0));
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    h->fmul(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_fwd(
        const TRegS &vmm_src) {

    const auto &t0 = ZRegS(IDX(vmm_src));
    const auto &t1 = ZRegS(IDX(vmm_aux1));
    const auto &t2 = ZRegS(IDX(vmm_aux2));
    const auto &t3 = ZRegS(IDX(vmm_aux3));
    const auto &t4 = ZRegS(IDX(vmm_aux4));
    const auto &mask = p_tmp0.s;
    const auto &wt0 = h->W_TMP_0;
    const auto &xt0 = h->X_TMP_0;
    auto set_imm = [&](const ZRegS &dst, uint32_t imm) {
        h->mov_imm(wt0, imm);
        h->cpy(dst, p_all, wt0);
        return dst;
    };
    Label tbl1L, tbl2L, exitL;
    const size_t tblL = 5;
    const size_t tblN = 1 << tblL;
    union fi {
        float f;
        uint32_t i;
    };
    //h->brk(0);
    h->mov(t4, p_all, t0);
    h->fmul(t0, t0, set_imm(z_tmp, float2int(std::sqrt(2))));
    set_imm(t3, 127 << 23);
    h->sub(t1, t0, t3);
    h->asr(t1, t1, 23); // n
    h->scvtf(t1, p_all, t1); // int -> float
    h->and_(t0, p_all, set_imm(z_tmp, 0x7fffff));
    h->asr(t2, t0, 23 - tblL); // d
    h->lsl(t2, t2, 2); // d *= 4
    h->orr(t0, p_all, t3); // y
    h->fmul(t0, t0, set_imm(z_tmp, float2int(1 / std::sqrt(2))));
    h->adr(xt0, tbl1L);
    h->ld1w(t3, p_all, ptr(xt0, t2, SXTW)); // f
    h->fcpy(z_tmp, p_all, 1.0f);
    h->fnmsb(t0, p_all, t3, z_tmp); // y = y * f - 1
    h->adr(xt0, tbl2L);
    h->ld1w(t2, p_all, ptr(xt0, t2, SXTW)); // h
    h->fsub(t3, t4, z_tmp); // x-1
    set_imm(z_tmp, float2int(1.0 / 32));
    h->facge(mask, p_all, z_tmp, t3); // 1/32 >= abs(x-1)
    h->mov(t0, mask, t3);
    h->eor(t2, mask, t2);
    h->fnmsb(t1, p_all, set_imm(z_tmp, float2int(std::log(2))),
            t2); // x = n * log2 - h
    set_imm(z_tmp, float2int(0.333332205));
    h->movprfx(t2, p_all, z_tmp);
    set_imm(z_tmp, float2int(-0.499999851));
    h->fmad(t2, p_all, t0, z_tmp); // f
    h->fcpy(z_tmp, p_all, 1.0f);
    h->fmad(t2, p_all, t0, z_tmp); // f * y + 1
    h->fmad(t0, p_all, t2, t1); // y * f + x
    // check nan/inf
    h->fcmlt(mask, p_all, t4, 0.0f); // neg
    h->mov(wt0, 0x7fc00000); // qnan
    h->cpy(t0, mask, wt0);
    h->fcmeq(mask, p_all, t4, 0.0f); // = 0
    h->mov(wt0, 0xff800000); // -Inf
    h->cpy(t0, mask, wt0);
    h->mov(wt0, 0x7f800000); // Inf
    h->dup(t1, wt0);
    h->fcmeq(mask, p_all, t4, t1);
    h->sel(t0, mask, t1, t0);

    h->b(exitL);
    h->L(tbl1L);
    const float *tbl1Addr = (const float *)h->getCurr();
    for (size_t i = 0; i < tblN; i++) {
        fi fi;
        fi.i = (127 << 23) | (i << (23 - tblL));
        fi.f = std::sqrt(2) / fi.f;
        h->dd(fi.i);
    }
    h->L(tbl2L);
    for (size_t i = 0; i < tblN; i++) {
        fi fi;
        fi.f = std::log(tbl1Addr[i]);
        h->dd(fi.i);
    }
    h->L(exitL);
}
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<
        isa>::gelu_erf_minimax_approx_compute_vector_fwd(const TRegS &vmm_src) {
    if (isa != sve_512) { // TODO: change this condition based on cpu id.
        return;
    }

    // register mapping
    TRegS vmm_pol = vmm_aux0;
    TRegS vmm_src_pos = vmm_aux1;
    TRegS vmm_indices = vmm_aux2;
    TRegS vmm_tmp = vmm_aux3; // this is for immediate read after write

    auto gather_coefficient
            = [&](TRegS vmm_coeff, int coeff_idx, TRegS vmm_pol_idx) {
                  // we actually have 25 polynomials but pad to avoid unaligned accesses/
                  int gelu_erf_n_polynomials = 32;
                  h->add_imm(h->X_TMP_1, x_table,
                          table_off(gelu_erf_minimax_pol,
                                  coeff_idx * gelu_erf_n_polynomials),
                          h->X_TMP_0);
                  h->ld1w(ZRegS(IDX(vmm_coeff)), p_all / T_z,
                          ptr(h->X_TMP_1, ZRegS(IDX(vmm_pol_idx)), SXTW));
              };

    // we use the erf function symmetry erf(-x) = -erf(x)
    // So we make x positive, we will reapply the sign after erf evaluation
    h->fabs(vmm_src_pos, p_all / T_z, vmm_src);

    // Compute indices for table lookup
    h->add(vmm_indices, vmm_src_pos,
            ZRegS(IDX(table_val(gelu_erf_idx_bias, z_tmp, 0))));

    // An arithmetic shift is needed to properly map denormals to
    // their polynomial. we shift by 21 as we use 2 bits of mantissa
    // for indexing.
    h->asr(ZRegS(IDX(vmm_indices)), ZRegS(IDX(vmm_indices)), 21);

    // Apply special rules
    h->smax(vmm_indices, p_all / T_z,
            ZRegS(IDX(table_val(gelu_erf_one, z_tmp))));
    h->smin(vmm_indices, p_all / T_z,
            ZRegS(IDX(table_val(gelu_erf_twenty_four, z_tmp))));

    // We have to check
    //     index = x_pos > rbound ? 23 : index;
    // for erf to return -1/1 when we should.
    h->fcmlt(p_mask.s, p_all / T_z, vmm_src_pos,
            ZRegS(IDX(table_val(gelu_erf_rbound, z_tmp))));
    h->sel(vmm_indices, p_mask, vmm_indices,
            ZRegS(IDX(table_val(gelu_erf_twenty_three, z_tmp))));

    // Adjusting indices
    h->mul(ZRegS(IDX(vmm_indices)), sizeof(float));

    // Evaluate the polynomial
    gather_coefficient(vmm_pol, 5, vmm_indices);
    for (int deg = 4; deg >= 0; --deg) {
        gather_coefficient(vmm_tmp, deg, vmm_indices);
        h->fmad(vmm_pol, p_all / T_z, vmm_src_pos, vmm_tmp);
    }

    // Set the sign of vmm_pol properly
    h->mov(ZRegD(IDX(vmm_tmp)), ZRegD(IDX(vmm_src)));
    h->and_(ZRegD(IDX(vmm_tmp)), ZRegD(IDX(vmm_tmp)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));
    h->eor(ZRegD(IDX(vmm_pol)), p_all / T_z, ZRegD(IDX(vmm_tmp)));

    // Compute the final output
    h->fadd(vmm_pol, vmm_pol, ZRegS(IDX(table_val(one, z_tmp))));
    h->fmul(vmm_src, p_all / T_z, vmm_pol);
    h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(half, z_tmp))));
}
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_fwd(
        const TRegS &vmm_src) {

    if (isa == sve_512) { // TODO: consider performance improvement for lower ISA
        gelu_erf_minimax_approx_compute_vector_fwd(vmm_src);
        return;
    }
    // Here we approximate erf(x) using the expression by
    // Abramowitz and Stegun from ``Handbook of Mathematical
    // Functions''
    // NOTE: The performance of this kernel can be further improved
    // with a minimax polynomialial expansion, thereby avoiding division
    // and exp. However, so far, this has costed larger accuracy
    // differences with respect to glibc erf based GELU, in particular
    // ~1.0e-5 -- 1.0e-3 absolute error at s = -5.

    constexpr unsigned sign_mask = 0x80000000;

    // use vmm_aux3 to store original src.
    h->mov(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_src)));

    // x = s / sqrt(2)
    h->fmul(vmm_src, vmm_src,
            ZRegS(IDX(table_val(gelu_erf_one_over_sqrt_two, z_tmp))));

    // abs(x)
    h->fabs(vmm_aux1, p_all / T_m, vmm_src);

    // t = 1 / (p*x + 1)
    table_val(gelu_erf_approx_const, vmm_aux2);
    h->fdup(vmm_aux4, 1.0f);
    h->fmad(vmm_aux2, p_all / T_m, vmm_aux1, vmm_aux4);
    h->fdiv(vmm_aux4, p_all, vmm_aux2);

    // -exp(-x*x)
    h->fmul(vmm_src, vmm_src, vmm_src);
    h->eor(vmm_src, sign_mask);
    exp_compute_vector_fwd(vmm_src);
    h->eor(vmm_src, sign_mask);

    // get sign
    h->mov(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(vmm_aux3)));
    h->and_(vmm_aux0, sign_mask);

    // -exp(-x*x)*t
    h->fmul(vmm_src, vmm_src, vmm_aux4);

    // compute polynomialial r
    table_val(gelu_erf_pol, vmm_aux1, 4);
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 3))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 2))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 1))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 0))));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->fmad(vmm_src, p_all / T_m, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux0)));

    // S = 0.5 * s
    h->fmul(vmm_aux3, p_all / T_m, 0.5f);
    // GELU = 0.5 * s * (1 + erf) = S + S * erf
    h->fmad(vmm_src, p_all / T_m, vmm_aux3, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_bwd(
        const TRegS &vmm_src) {
    h->fcmgt(p_mask.s, p_all / T_z, vmm_src, 0.);
    h->mov(ZRegD(vmm_src.getIdx()), ZRegD(z_tmp.getIdx()));
    /* Do not use 1.f, which is a float constant,
       but 1., which is a double constant. */
    h->fmov(vmm_src, p_mask / T_m, 1.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_bwd(
        const TRegS &vmm_src) {
    if (!use_dst_) {
        // R = exp(s)
        exp_compute_vector_fwd(vmm_src);
        // after exponentiation, get mask by comparing with exp(0)=1.f, not 0.f
        compute_cmp_mask(vmm_src, table_val(one, z_tmp), _cmp_gt_os);
        // R * alpha, then blend with 1.f
        h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(alpha, z_tmp))));
    } else {
        // get mask of `d` > 0
        compute_cmp_mask(vmm_src, table_val(zero, z_tmp), _cmp_gt_os);
        // R = `d` + alpha, then blend with 1.f
        h->fadd(vmm_src, vmm_src, ZRegS(IDX(table_val(alpha, z_tmp))));
    }
    blend_with_mask(vmm_src, table_val(one, z_tmp));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 1 - d^2 = 1 - tanh^2(s)
    if (!use_dst_) tanh_compute_vector_fwd(vmm_src);
    h->mov(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(table_val(one, z_tmp))));

    h->fmls(vmm_aux0, p_all / T_m, vmm_src, vmm_src);

    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux0)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_bwd(
        const TRegS &vmm_src) {
    h->mov(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(vmm_src)));

    // compute G1(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x^2)
    // compute G2(x) = sqrt_root_two_over_pi * x * (1 + 3 * fitting_const * x^2)
    h->fmul(vmm_src, vmm_src, vmm_src);

    // keep G2 in a separate register
    h->mov(ZRegD(IDX(vmm_aux2)),
            ZRegD(IDX(table_val(gelu_tanh_fitting_const_times_three, z_tmp))));
    h->fmad(vmm_aux2, p_all / T_m, vmm_src, ZRegS(IDX(table_val(one, z_tmp))));

    h->mov(ZRegD(IDX(vmm_aux1)),
            ZRegD(IDX(table_val(gelu_tanh_fitting_const, z_tmp))));
    h->fmad(vmm_src, p_all / T_m, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));
    h->fmul(vmm_aux0, vmm_aux0,
            ZRegS(IDX(table_val(gelu_tanh_sqrt_two_over_pi, z_tmp))));
    h->fmul(vmm_src, vmm_src, vmm_aux0);
    h->fmul(vmm_aux2, vmm_aux2, vmm_aux0);

    // save G2 on stack as tanh uses all available registers
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->str(ZReg(IDX(vmm_aux2)), ptr(h->X_TMP_0));

    // T = tanh(G1(x))
    tanh_compute_vector_fwd(vmm_src);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux2)), ptr(h->X_TMP_0));
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    // compute 0.5 * (1 + T) * (1 + G2 * (1 - T))
    // 1) R = G2 * (1 - T) = G2 - G2 * T
    h->fmls(vmm_aux2, p_all / T_m, vmm_aux2, vmm_src);
    // 2) Q = 1 + T
    h->fadd(vmm_src, vmm_src, ZRegS(IDX(table_val(one, z_tmp))));
    // 3) res = Q * (1 + R) = Q + Q * R
    h->fmla(vmm_src, p_all / T_m, vmm_src, vmm_aux2);

    h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(half, z_tmp))));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 2 * s
    h->fmul(vmm_src, p_all / T_m, 2.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_bwd(
        const TRegS &vmm_src) {
    // replace positive values with 1.f
    compute_cmp_mask(vmm_src, table_val(zero, z_tmp), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one, z_tmp));
    // replace negative values with -1.f
    compute_cmp_mask(vmm_src, table_val(zero, z_tmp), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(minus_one, z_tmp));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 0.5 / d = 0.5 / sqrt(s)
    if (!use_dst_) sqrt_compute_vector_fwd(vmm_src);
    h->mov(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(table_val(half, z_tmp))));
    h->fdiv(vmm_aux0, p_all, vmm_src);
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux0)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_bwd(
        const TRegS &vmm_src) {
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(table_val(alpha, z_tmp))));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_bwd(
        const TRegS &vmm_src) {
    if (alpha_ == 1.f) {
        // Skip an instruction.
    } else if (alpha_ == 0.5f || alpha_ == 2.f)
        h->fmul(vmm_src, p_all / T_m, alpha_);
    else
        h->fmul(vmm_src, vmm_src, z_tmp);
    logistic_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::mish_compute_vector_bwd(
        const TRegS &vmm_src) {
    // IMPORTANT: we use vmm_aux3 to save src as exp does not use it.
    h->mov(ZRegD(vmm_aux3.getIdx()), ZRegD(vmm_src.getIdx())); // vmm_aux3 = x

    h->fminnm(vmm_src, p_all / T_m,
            table_val(bwd_mish_max_x_for_equation_f, z_tmp));
    exp_compute_vector_fwd(vmm_src);
    h->mov(ZRegD(vmm_aux2.getIdx()), ZRegD(vmm_src.getIdx())); // vmm_aux2 = e^x

    // e^3x + 4*e^2x
    h->fmul(vmm_src, vmm_src, vmm_src); // e^2x
    h->mov(ZRegD(vmm_aux1.getIdx()), ZRegD(vmm_src.getIdx()));
    h->fmul(vmm_aux1, p_all / T_m, 2.);
    h->fmul(vmm_aux1, p_all / T_m, 2.); // 4*e^2x
    h->fmad(vmm_src, p_all / T_m, vmm_aux2, vmm_aux1);

    // e^3x + 4*e^2x + 4*e^x*(x+1.5)
    h->fadd(vmm_aux3, p_all / T_m, 1.); // vmm_aux3 = x + 1
    h->mov(ZRegD(vmm_aux1.getIdx()), ZRegD(vmm_aux3.getIdx()));
    h->fadd(vmm_aux1, p_all / T_m, 0.5f);
    h->fmul(vmm_aux1, p_all / T_m, 2.);
    h->fmul(vmm_aux1, p_all / T_m, 2.);
    h->fmla(vmm_src, p_all / T_m, vmm_aux1, vmm_aux2);

    // omega = e^3x + 4*e^2x + 4*e^x*(x+1.5) + 4*(x+1)
    h->fmul(vmm_aux3, p_all / T_m, 2.);
    /* Do not use 2.f, which is a float constant,
       but 2., which is a double constant. */
    h->fmov(z_tmp, 2.);
    h->fmla(vmm_src, p_all / T_m, vmm_aux3, z_tmp);

    // delta = (e^x+1)^2 + 1
    h->mov(ZRegD(vmm_aux1.getIdx()), ZRegD(vmm_aux2.getIdx()));
    h->fadd(vmm_aux1, p_all / T_m, 1.);
    h->fmul(vmm_aux1, vmm_aux1, vmm_aux1);
    h->fadd(vmm_aux1, p_all / T_m, 1.);
    h->fmul(vmm_aux1, vmm_aux1, vmm_aux1);

    // e^x * omega / delta^2
    h->fmul(vmm_src, vmm_src, vmm_aux2);
    h->fdiv(vmm_src, p_all / T_m, vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = d * (1 - d) = d - d * d; d = logistic(s)
    if (!use_dst_) logistic_compute_vector_fwd(vmm_src);
    /* Do not use 1.f, which is a float constant,
       but 1., which is a double constant. */
    h->fmov(vmm_aux0, 1.);
    h->fsub(vmm_aux0, vmm_aux0, vmm_src);
    h->fmul(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_bwd(
        const TRegS &vmm_src) {
    if (!use_dst_) exp_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_bwd(
        const TRegS &vmm_src) {
    // R = alpha * s
    h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(alpha, z_tmp))));

    // Save R on stack for later usage
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->str(ZReg(IDX(vmm_src)), ptr(h->X_TMP_0));

    // Q = sigmoid(alpha * s)
    logistic_compute_vector_fwd(vmm_src);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux0)), ptr(h->X_TMP_0));

    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    // compute Q * (1 + R * (1 - Q))
    // T = R * (1 - Q) = R - R * Q
    h->fmls(vmm_aux0, p_all / T_m, vmm_aux0, vmm_src);

    // Q * (1 + T) = Q + Q * T
    h->fmla(vmm_src, p_all / T_m, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 1 / s
    /* Do not use 1.f, which is a float constant,
       but 1., which is a double constant. */
    h->fmov(z_tmp, 1.);
    h->fdiv(z_tmp, p_all, vmm_src);
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(z_tmp)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_bwd(
        const TRegS &vmm_src) {
    using namespace alg_kind;
    // z_tmp:alpha, vmm_aux0:beta

    // get mask of values > beta and blend with 0.f
    if (alg_ == eltwise_clip)
        h->fcmle(p_mask.s, p_all / T_z, vmm_src, vmm_aux0);
    else
        h->fcmlt(p_mask.s, p_all / T_z, vmm_src, vmm_aux0);
    // get mask of values <= alpha and blend with 0.f
    h->fcmgt(p_mask.s, p_mask / T_z, vmm_src, z_tmp);

    h->mov(vmm_src, 0);
    h->fmov(vmm_src, p_mask / T_m, 1.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_bwd(
        const TRegS &vmm_src) {
    // R = s / sqrt(2)
    h->fmul(vmm_src, vmm_src,
            ZRegS(IDX(table_val(gelu_erf_one_over_sqrt_two, z_tmp))));

    // Save R on stack for later usage
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->str(ZReg(IDX(vmm_src)), ptr(h->X_TMP_0));

    // Q = exp(-R*R)
    h->fmul(vmm_src, vmm_src, vmm_src);
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));
    exp_compute_vector_fwd(vmm_src);

    // T = R / sqrt(pi) * Q
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux2)), ptr(h->X_TMP_0));
    h->fmul(vmm_aux2, vmm_aux2,
            ZRegS(IDX(table_val(gelu_erf_one_over_sqrt_pi, z_tmp))));
    h->fmul(vmm_aux2, vmm_aux2, vmm_src);

    // -Q
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));

    // get sign
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux0)), ptr(h->X_TMP_0));
    h->and_(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(vmm_aux0)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));

    // abs(x)
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux1)), ptr(h->X_TMP_0));
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    abs_compute_vector_fwd(vmm_aux1);

    // W = 1 / (p * s + 1)
    h->mov(ZRegD(IDX(vmm_aux3)),
            ZRegD(IDX(table_val(gelu_erf_approx_const, z_tmp))));
    h->mov(ZRegD(IDX(vmm_aux4)), ZRegD(IDX(table_val(one, z_tmp))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1, vmm_aux4);
    h->fdiv(vmm_aux4, p_all, vmm_aux3);

    // Q * W
    h->fmul(vmm_src, vmm_src, vmm_aux4);

    // compute polynomial r
    h->mov(ZRegD(IDX(vmm_aux1)), ZRegD(IDX(table_val(gelu_erf_pol, z_tmp, 4))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 3))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 2))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 1))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 0))));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->fmad(vmm_src, p_all / T_m, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux0)));

    // P = T + 0.5
    h->fadd(vmm_aux2, vmm_aux2, ZRegS(IDX(table_val(half, z_tmp))));
    // res = P + 0.5 * erf
    h->fmla(vmm_aux2, p_all / T_m, vmm_src, ZRegS(IDX(table_val(half, z_tmp))));
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux2)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::hardswish_compute_vector_bwd(
        const TRegS &vmm_src) {
    // alpha:z_tmp, beta:vmm_aux0, 1.f:vmm_aux1
    // Get mask for 0 < alpha * x + beta < 1
    h->mov(ZRegD(vmm_aux2.getIdx()), ZRegD(vmm_src.getIdx()));
    h->fmul(vmm_aux2, vmm_aux2, z_tmp);
    h->fadd(vmm_aux2, vmm_aux2, vmm_aux0);
    // Form a derivative value
    h->fmul(vmm_src, vmm_src, z_tmp);
    h->fadd(vmm_src, vmm_src, vmm_aux2);

    h->fcmle(p_mask.s, p_all / T_z, vmm_aux2, 0.);
    h->mov(vmm_src, p_mask / T_m, 0);
    h->fcmge(p_mask.s, p_all / T_z, vmm_aux2, vmm_aux1);
    h->fmov(vmm_src, p_mask / T_m, 1.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::hardsigmoid_compute_vector_bwd(
        const TRegS &vmm_src) {
    // alpha:z_tmp, beta:vmm_aux0, 1:vmm_aux1
    // Get mask for 0 < alpha * x + beta < 1
    // Zero rest values.
    h->fmul(vmm_src, vmm_src, z_tmp);
    h->fadd(vmm_src, vmm_src, vmm_aux0);

    h->fcmgt(p_mask.s, p_all / T_z, vmm_src, 0.);
    h->fcmlt(p_mask.s, p_mask / T_z, vmm_src, vmm_aux1);

    h->mov(vmm_src, 0);
    h->fmov(vmm_src, p_mask / T_m, 1.);
    h->fmul(vmm_src, vmm_src, z_tmp);
}

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_gprs_count() {
    using namespace alg_kind;
    switch (alg_) {
        case eltwise_tanh_use_dst_for_bwd:
        case eltwise_tanh:
        case eltwise_gelu_tanh: return 0;
        default: return 0;
    }
    return 0;
};

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::round_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->frintn(vmm_src, p_all / T_m, vmm_src);
}

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_vecs_count() {
    using namespace alg_kind;

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return (alpha_ == 0.f) ? 1 : 3;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return 6; /* = exp + 2 */
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh: return 9;
            case eltwise_square: return 0;
            case eltwise_abs: return 1;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 0;
            case eltwise_linear: return 2;
            case eltwise_soft_relu: return 5;
            case eltwise_mish: return 5; /* = exp + 1 */
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic: return 5; /* = exp + 1 */
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp: return 4;
            case eltwise_gelu_tanh: return 9; /* = tanh */
            case eltwise_swish: return 6; /* = logistic */
            case eltwise_log: return 6;
            case eltwise_clip:
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2: return 2;
            case eltwise_gelu_erf: return 6;
            case eltwise_round: return 0;
            case eltwise_hardswish: return 3; /* = hardsigmoid + 1 */
            case eltwise_hardsigmoid: return 2;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return 1;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return 4; /* = exp */
            case eltwise_tanh_use_dst_for_bwd: return 2;
            case eltwise_tanh: return 9;
            case eltwise_square: return 1;
            case eltwise_abs: return 1;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 2;
            case eltwise_linear: return 1;
            case eltwise_soft_relu: return 5; /* = max(1, logistic) */
            case eltwise_mish: return 5; /* = exp + 1 */
            case eltwise_logistic_use_dst_for_bwd: return 2;
            case eltwise_logistic: return 5; /* = logistic */
            case eltwise_exp_use_dst_for_bwd: return 0;
            case eltwise_exp: return 4; /* = exp */
            case eltwise_gelu_tanh: return 9; /* = tanh */
            case eltwise_swish: return 6; /* = logistic */
            case eltwise_log: return 1;
            case eltwise_clip:
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2: return 2;
            case eltwise_gelu_erf: return 6;
            case eltwise_hardswish: return 4;
            case eltwise_hardsigmoid: return 3;
            default: assert(!"unsupported eltwise algorithm");
        }
    }

    return 0;
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_body(
        const injector_utils::vmm_index_set_iterator_t &start_idx_it,
        const injector_utils::vmm_index_set_iterator_t &end_idx_it) {
    using namespace alg_kind;
    std::for_each(start_idx_it, end_idx_it, [&](size_t idx) {
        if (is_fwd_) {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu:
                    if (alpha_ == 0.f)
                        relu_zero_ns_compute_vector_fwd(TRegS(idx));
                    else
                        relu_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_square:
                    square_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_abs: abs_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_swish: swish_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_linear:
                    linear_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_mish: mish_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_log: log_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_clip:
                case eltwise_clip_v2_use_dst_for_bwd:
                case eltwise_clip_v2:
                    clip_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_round: round_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_hardswish:
                    hardswish_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_hardsigmoid:
                    hardsigmoid_compute_vector_fwd(TRegS(idx));
                    break;
                default: assert(!"unsupported eltwise algorithm");
            }
        } else {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu: relu_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_square:
                    square_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_abs: abs_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_linear:
                    linear_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_mish: mish_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_swish: swish_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_log: log_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_clip:
                case eltwise_clip_v2_use_dst_for_bwd:
                case eltwise_clip_v2:
                    clip_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_hardswish:
                    hardswish_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_hardsigmoid:
                    hardsigmoid_compute_vector_bwd(TRegS(idx));
                    break;
                default: assert(!"unsupported eltwise algorithm");
            }
        }
        if (scale_ != 1.f) {
            h->fmul(ZRegS(IDX(TRegS(idx))), ZRegS(IDX(TRegS(idx))),
                    ZRegS(IDX(table_val(scale, vmm_mask))));
        }
    });
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    const auto &start_idx_it = vmm_idxs.begin();
    const auto &end_idx_it = vmm_idxs.end();
    assert(*start_idx_it < *vmm_idxs.rbegin() + 1
            && *vmm_idxs.rbegin() <= vecs_count);

    injector_preamble(vmm_idxs);
    compute_body(start_idx_tail, end_idx_it);
    injector_preamble_tail(start_idx_it);
    compute_body(start_idx_it, start_idx_tail);
    injector_postamble();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::prepare_table(bool gen_table) {
    if (!gen_table) return;

    h->align(64);
    h->L(l_table);

    // Assumption: entries can be inserted with dd, so they should be 4 bytes.
    assert(sizeof(table_entry_val_t) == 4);

    // Assumption: iterating on entry_map_ here has the same order as
    // when we set the offsets. We verify that in asserts.
    // table_entry_val_t is assumed to be 32 bits
#ifndef NDEBUG
    size_t off = 0;
    key_t curr_key = undef_key;
    int key_occurences = 0;
#endif

    // Run through the map and insert values stored there
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        const auto &te = (*it).second; // get map entry for a given key
        const auto len = te.bcast ? vlen : sizeof(table_entry_val_t);
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
            h->dd(te.val);

#ifndef NDEBUG
        // we check that the precomputed offsets match the registered ones
        const auto &key = (*it).first; // get map entry key
        if (key != curr_key) {
            curr_key = key;
            key_occurences = 0;
        }
        key_occurences++;
        auto expected_off = table_off(key, key_occurences - 1);
        assert(off == expected_off);
        MAYBE_UNUSED(expected_off);
        off += len;
#endif
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::register_table_entries() {
    // This function is responsible to pick all necessary constants
    // for a given algorithm, compute right offset for them to be used
    // in table_val() and save the hexadecimal value of them, which
    // will be finally used in prepare_table(). We rely on fact that
    // the map iterator order is deterministic for a fixed map.

    // common values used in several algorithms
    static const table_t common_values {{zero, {0x00000000, true}},
            {half, {0x3f000000, true}}, {one, {0x3f800000, true}},
            {two, {0x40000000, true}}, {minus_one, {0xbf800000, true}},
            {minus_two, {0xc0000000, true}}, {ln2f, {0x3f317218, true}},
            {positive_mask, {0x7fffffff, true}},
            {sign_mask, {0x80000000, true}},
            {exponent_bias, {0x0000007f, true}}};

    // exp(x) constants
    static const table_t exp_consts {{exp_log2ef, {0x3fb8aa3b, true}},
            {exp_ln_flt_max_f, {0x42b17218, true}},
            {exp_ln_flt_min_f, {0xc2aeac50, true}}};

    // exp(x) polynomial approximation
    static const table_t exp_polynomial {
            {exp_pol, {0x3f7ffffb, true}}, // p1 = 0.999999701f
            {exp_pol, {0x3efffee3, true}}, // p2 = 0.499991506f
            {exp_pol, {0x3e2aad40, true}}, // p3 = 0.166676521f
            {exp_pol, {0x3d2b9d0d, true}}, // p4 = 0.0418978221f
            {exp_pol, {0x3c07cfce, true}} // p5 = 0.00828929059f
    };
    // exp(x) constants2
    static const table_t exp_consts2 {
            {exp_coeff1, {0x3f31721c, true}},
            {exp_coeff2, {0x3e772df2, true}},
            {exp_not_mask17, {~((1u << 17) - 1), true}},
    };

    // mish(x) constants
    static const table_t mish_consts {
            {fwd_mish_max_x_for_equation_f, {0x42317217, true}},
            {bwd_mish_max_x_for_equation_f, {0x41b17217, true}}};

    // tanh(x) constants for four interval approximation
    // and for polynomial approximation
    static const table_t tanh_consts {{tanh_range, {0x3d4ccccd, true}},
            {tanh_m1d3, {0xbeaaaaab, true}},
            {tanh_idx_bias, {0x39800000, true}},
            {tanh_idx_mask, {0xffc00000, true}},
            {tanh_linear_ubound, {0x39ddb3d7, true}},
            {tanh_saturation_lbound, {0x41102cb3, true}}};

    // tanh(x) polynomial approximation
    // For each coefficient, there is 32 entries
    static const table_t tanh_polynomial_table {
            // coefficients of degree 0
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x39bfffff, false}},
            {tanh_pol_table, {0x39ffffff, false}},
            {tanh_pol_table, {0x3a3ffffe, false}},
            {tanh_pol_table, {0x3a7ffffb, false}},
            {tanh_pol_table, {0x3abffff7, false}},
            {tanh_pol_table, {0x3affffeb, false}},
            {tanh_pol_table, {0x3b3fffdc, false}},
            {tanh_pol_table, {0x3b7fffab, false}},
            {tanh_pol_table, {0x3bbfff70, false}},
            {tanh_pol_table, {0x3bfffeab, false}},
            {tanh_pol_table, {0x3c3ffdc0, false}},
            {tanh_pol_table, {0x3c7ffaab, false}},
            {tanh_pol_table, {0x3cbff701, false}},
            {tanh_pol_table, {0x3cffeaad, false}},
            {tanh_pol_table, {0x3d3fdc08, false}},
            {tanh_pol_table, {0x3d7faacd, false}},
            {tanh_pol_table, {0x3dbf7081, false}},
            {tanh_pol_table, {0x3dfeacc9, false}},
            {tanh_pol_table, {0x3e3dc7fd, false}},
            {tanh_pol_table, {0x3e7acbf5, false}},
            {tanh_pol_table, {0x3eb77a9f, false}},
            {tanh_pol_table, {0x3eec9a9f, false}},
            {tanh_pol_table, {0x3f22991f, false}},
            {tanh_pol_table, {0x3f42f7d6, false}},
            {tanh_pol_table, {0x3f67b7cc, false}},
            {tanh_pol_table, {0x3f76ca83, false}},
            {tanh_pol_table, {0x3f7ebbe9, false}},
            {tanh_pol_table, {0x3f7fd40c, false}},
            {tanh_pol_table, {0x3f7fff32, false}},
            {tanh_pol_table, {0x3f7ffffc, false}},
            {tanh_pol_table, {0x3f800000, false}},
            // coefficients of degree 1
            {tanh_pol_table, {0x3f800000, false}},
            {tanh_pol_table, {0x3f800018, false}},
            {tanh_pol_table, {0x3f7fffe8, false}},
            {tanh_pol_table, {0x3f7fffda, false}},
            {tanh_pol_table, {0x3f7fffdc, false}},
            {tanh_pol_table, {0x3f7fffdc, false}},
            {tanh_pol_table, {0x3f7fffac, false}},
            {tanh_pol_table, {0x3f7fff70, false}},
            {tanh_pol_table, {0x3f7ffeec, false}},
            {tanh_pol_table, {0x3f7ffdc0, false}},
            {tanh_pol_table, {0x3f7ffbed, false}},
            {tanh_pol_table, {0x3f7ff704, false}},
            {tanh_pol_table, {0x3f7feff5, false}},
            {tanh_pol_table, {0x3f7fdbca, false}},
            {tanh_pol_table, {0x3f7fbfff, false}},
            {tanh_pol_table, {0x3f7f7041, false}},
            {tanh_pol_table, {0x3f7f009b, false}},
            {tanh_pol_table, {0x3f7dc36c, false}},
            {tanh_pol_table, {0x3f7c0aa8, false}},
            {tanh_pol_table, {0x3f7734b8, false}},
            {tanh_pol_table, {0x3f70a4de, false}},
            {tanh_pol_table, {0x3f5f1fd8, false}},
            {tanh_pol_table, {0x3f495493, false}},
            {tanh_pol_table, {0x3f18b9ec, false}},
            {tanh_pol_table, {0x3ed706cb, false}},
            {tanh_pol_table, {0x3e390b06, false}},
            {tanh_pol_table, {0x3d90b11f, false}},
            {tanh_pol_table, {0x3c21a053, false}},
            {tanh_pol_table, {0x3aaf7fdb, false}},
            {tanh_pol_table, {0x37ccc1a3, false}},
            {tanh_pol_table, {0x355c6733, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 2
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xbe4e0ff1, false}},
            {tanh_pol_table, {0x3d25b1b1, false}},
            {tanh_pol_table, {0x3d6b6dab, false}},
            {tanh_pol_table, {0x3c9fb1d5, false}},
            {tanh_pol_table, {0xbabff06f, false}},
            {tanh_pol_table, {0x3c07b3f6, false}},
            {tanh_pol_table, {0xbb3fc1bc, false}},
            {tanh_pol_table, {0x3a9f5921, false}},
            {tanh_pol_table, {0xbbbf06f2, false}},
            {tanh_pol_table, {0xbbb0f402, false}},
            {tanh_pol_table, {0xbc47db9e, false}},
            {tanh_pol_table, {0xbc73d5e7, false}},
            {tanh_pol_table, {0xbca25bda, false}},
            {tanh_pol_table, {0xbcfca780, false}},
            {tanh_pol_table, {0xbd40e07c, false}},
            {tanh_pol_table, {0xbd7dab03, false}},
            {tanh_pol_table, {0xbdbe4a0f, false}},
            {tanh_pol_table, {0xbdfb14a5, false}},
            {tanh_pol_table, {0xbe36cc8d, false}},
            {tanh_pol_table, {0xbe6bd102, false}},
            {tanh_pol_table, {0xbe9fe7c5, false}},
            {tanh_pol_table, {0xbeba0f10, false}},
            {tanh_pol_table, {0xbec206a8, false}},
            {tanh_pol_table, {0xbea3c388, false}},
            {tanh_pol_table, {0xbe277d62, false}},
            {tanh_pol_table, {0xbd8b7960, false}},
            {tanh_pol_table, {0xbc209f49, false}},
            {tanh_pol_table, {0xbaad44ca, false}},
            {tanh_pol_table, {0xb7c6eeac, false}},
            {tanh_pol_table, {0xb663aa41, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 3
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x45b3ae96, false}},
            {tanh_pol_table, {0xc414eb20, false}},
            {tanh_pol_table, {0xc450e02e, false}},
            {tanh_pol_table, {0xc3152b4e, false}},
            {tanh_pol_table, {0xbead2f56, false}},
            {tanh_pol_table, {0xc2162e02, false}},
            {tanh_pol_table, {0xbeb4bd5a, false}},
            {tanh_pol_table, {0xc11a59a4, false}},
            {tanh_pol_table, {0xbed2f507, false}},
            {tanh_pol_table, {0xc020d32c, false}},
            {tanh_pol_table, {0x3dd0f506, false}},
            {tanh_pol_table, {0xbf2a75e2, false}},
            {tanh_pol_table, {0xbff950e3, false}},
            {tanh_pol_table, {0xbed47334, false}},
            {tanh_pol_table, {0xbe809b8c, false}},
            {tanh_pol_table, {0xbeb64532, false}},
            {tanh_pol_table, {0xbe961a5b, false}},
            {tanh_pol_table, {0xbe9b63ac, false}},
            {tanh_pol_table, {0xbea0d4b2, false}},
            {tanh_pol_table, {0xbe828a77, false}},
            {tanh_pol_table, {0xbe378612, false}},
            {tanh_pol_table, {0xbdc20908, false}},
            {tanh_pol_table, {0x3d2d3957, false}},
            {tanh_pol_table, {0x3dd46e89, false}},
            {tanh_pol_table, {0x3db3f629, false}},
            {tanh_pol_table, {0x3d2c5e7b, false}},
            {tanh_pol_table, {0x3bd20403, false}},
            {tanh_pol_table, {0x3a59dfae, false}},
            {tanh_pol_table, {0x3770af45, false}},
            {tanh_pol_table, {0x372cc014, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 4
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xcc981a1b, false}},
            {tanh_pol_table, {0x4a7edd3d, false}},
            {tanh_pol_table, {0x4ab1007c, false}},
            {tanh_pol_table, {0x48fedd9c, false}},
            {tanh_pol_table, {0x41a557b5, false}},
            {tanh_pol_table, {0x477ee32a, false}},
            {tanh_pol_table, {0x422557f5, false}},
            {tanh_pol_table, {0x45ff3ce4, false}},
            {tanh_pol_table, {0x42a55641, false}},
            {tanh_pol_table, {0x446e0867, false}},
            {tanh_pol_table, {0xc33dc19a, false}},
            {tanh_pol_table, {0x42915214, false}},
            {tanh_pol_table, {0x43af4fad, false}},
            {tanh_pol_table, {0x4110fe88, false}},
            {tanh_pol_table, {0xc1099b75, false}},
            {tanh_pol_table, {0x3fc8a8dc, false}},
            {tanh_pol_table, {0xbfbeaef5, false}},
            {tanh_pol_table, {0xbe365aad, false}},
            {tanh_pol_table, {0x3f4d9652, false}},
            {tanh_pol_table, {0x3ddfa08f, false}},
            {tanh_pol_table, {0x3e34e9b8, false}},
            {tanh_pol_table, {0x3e2d07a6, false}},
            {tanh_pol_table, {0x3dc63567, false}},
            {tanh_pol_table, {0x3cdaeb78, false}},
            {tanh_pol_table, {0xbcd17537, false}},
            {tanh_pol_table, {0xbc92829c, false}},
            {tanh_pol_table, {0xbb43ab99, false}},
            {tanh_pol_table, {0xb9b471dd, false}},
            {tanh_pol_table, {0xb6baad5a, false}},
            {tanh_pol_table, {0xb78bafc7, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 5
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x52f688d5, false}},
            {tanh_pol_table, {0xd0505c72, false}},
            {tanh_pol_table, {0xd08f98e3, false}},
            {tanh_pol_table, {0xce505cc9, false}},
            {tanh_pol_table, {0xc7162b8a, false}},
            {tanh_pol_table, {0xcc5061d6, false}},
            {tanh_pol_table, {0xc7162bdf, false}},
            {tanh_pol_table, {0xca50b37f, false}},
            {tanh_pol_table, {0xc7162a3a, false}},
            {tanh_pol_table, {0xc8422086, false}},
            {tanh_pol_table, {0x471a714e, false}},
            {tanh_pol_table, {0xc5ece1f1, false}},
            {tanh_pol_table, {0xc70e3d90, false}},
            {tanh_pol_table, {0xc3eba94a, false}},
            {tanh_pol_table, {0x43e0c424, false}},
            {tanh_pol_table, {0xc21f4552, false}},
            {tanh_pol_table, {0x42217cc8, false}},
            {tanh_pol_table, {0x405e7dc4, false}},
            {tanh_pol_table, {0xc10dd401, false}},
            {tanh_pol_table, {0x3e96b602, false}},
            {tanh_pol_table, {0xbd1a6d2f, false}},
            {tanh_pol_table, {0xbd393883, false}},
            {tanh_pol_table, {0xbd674682, false}},
            {tanh_pol_table, {0xbd310016, false}},
            {tanh_pol_table, {0xb961e269, false}},
            {tanh_pol_table, {0x3ba32495, false}},
            {tanh_pol_table, {0x3a7680d5, false}},
            {tanh_pol_table, {0x38b3173c, false}},
            {tanh_pol_table, {0x35a9deea, false}},
            {tanh_pol_table, {0x375c3f2a, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 6
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xd8995ed1, false}},
            {tanh_pol_table, {0x558285ea, false}},
            {tanh_pol_table, {0x55b2cd69, false}},
            {tanh_pol_table, {0x53028625, false}},
            {tanh_pol_table, {0x4bc9991f, false}},
            {tanh_pol_table, {0x5082898a, false}},
            {tanh_pol_table, {0x4b4999b3, false}},
            {tanh_pol_table, {0x4e02c07c, false}},
            {tanh_pol_table, {0x4ac99764, false}},
            {tanh_pol_table, {0x4b72c822, false}},
            {tanh_pol_table, {0xca40c0e1, false}},
            {tanh_pol_table, {0x489413e4, false}},
            {tanh_pol_table, {0x49b12224, false}},
            {tanh_pol_table, {0x46134c4e, false}},
            {tanh_pol_table, {0xc60c2d57, false}},
            {tanh_pol_table, {0x43c83910, false}},
            {tanh_pol_table, {0xc3c872d1, false}},
            {tanh_pol_table, {0xc186bc9e, false}},
            {tanh_pol_table, {0x42325bc3, false}},
            {tanh_pol_table, {0xbf2ffa4a, false}},
            {tanh_pol_table, {0x3d9a203c, false}},
            {tanh_pol_table, {0xbc545a43, false}},
            {tanh_pol_table, {0xbae08fee, false}},
            {tanh_pol_table, {0x3c80225d, false}},
            {tanh_pol_table, {0x3b1fd1df, false}},
            {tanh_pol_table, {0xba36b9d1, false}},
            {tanh_pol_table, {0xb91de544, false}},
            {tanh_pol_table, {0xb71f100f, false}},
            {tanh_pol_table, {0xb408e2ed, false}},
            {tanh_pol_table, {0xb685fec8, false}},
            {tanh_pol_table, {0x00000000, false}},
    };

    // soft_relu(x) constants
    static const table_t soft_relu_consts {
            {soft_relu_one_twenty_six, {0x42fc0000, true}},
            {soft_relu_mantissa_sign_mask, {0x807fffff, true}},
    };

    // soft_relu ln(1 + x) polynomial approximation
    static const table_t soft_relu_polynomial {
            {soft_relu_pol, {0xb2b4637d, true}}, // p0 = 0.0000000244f
            {soft_relu_pol, {0x3f7fff8e, true}}, // p1 = 0.9999976971f
            {soft_relu_pol, {0xbf001759, true}}, // p2 = -0.5002478215f
            {soft_relu_pol, {0x3ea70608, true}}, // p3 = 0.3272714505f
            {soft_relu_pol, {0xbea3d7bf, true}}, // p4 = -0.3153830071f
            {soft_relu_pol, {0xbe361d04, true}}, // p5 = -0.1701777461f
            {soft_relu_pol, {0xbfa8f1e6, true}}, // p6 = -1.3254635147f
            {soft_relu_pol, {0xbfe1e812, true}}, // p7 = -1.7971917960f
            {soft_relu_pol, {0xbfc4d30e, true}}, // p8 = -1.5652673123f
    };

    // gelu_tanh(x) constants (formula defined)
    static const table_t gelu_tanh_consts {
            {gelu_tanh_fitting_const, {0x3d372713, true}},
            {gelu_tanh_fitting_const_times_three, {0x3e095d4f, true}},
            {gelu_tanh_sqrt_two_over_pi, {0x3f4c422a, true}},
    };

    // gelu_erf(x) constants (formula defined)
    static const table_t gelu_erf_consts {
            {gelu_erf_approx_const, {0x3ea7ba05, true}},
            {gelu_erf_one_over_sqrt_two, {0x3f3504f3, true}},
            {gelu_erf_one_over_sqrt_pi, {0x3f106eba, true}},
    };

    // gelu_erf(x) polynomial approximation
    static const table_t gelu_erf_polynomial {
            {gelu_erf_pol, {0x3e827906, true}}, // p1 = 0.254829592f
            {gelu_erf_pol, {0xbe91a98e, true}}, // p2 = -0.284496736f
            {gelu_erf_pol, {0x3fb5f0e3, true}}, // p3 = 1.421413741f
            {gelu_erf_pol, {0xbfba00e3, true}}, // p4 = -1.453152027f
            {gelu_erf_pol, {0x3f87dc22, true}}, // p5 = 1.061405429f
    };
    // gelu_erf(x) constants for direct erf approximation (formula defined)
    static const table_t gelu_erf_minimax_consts {
            {gelu_erf_idx_bias, {0xc21fffff, true}},
            {gelu_erf_rbound, {0x40b15cee, true}},
            {gelu_erf_one, {0x00000001, true}},
            {gelu_erf_twenty_three, {0x00000017, true}},
            {gelu_erf_twenty_four, {0x00000018, true}},
    };
    // gelu_erf(x) minimax polynomials for piecewise approximaxtion
    static const table_t gelu_erf_minimax_polynomial {
            // coefficients of degree  0
            {gelu_erf_minimax_pol, {0xa6f2cb94, false}}, // -0x1.e59728p-50
            {gelu_erf_minimax_pol, {0x32827792, false}}, // 0x1.04ef24p-26
            {gelu_erf_minimax_pol, {0x3381cc0c, false}}, // 0x1.039818p-24
            {gelu_erf_minimax_pol, {0x34523d4a, false}}, // 0x1.a47a94p-23
            {gelu_erf_minimax_pol, {0x351ac44d, false}}, // 0x1.35889ap-21
            {gelu_erf_minimax_pol, {0x35f36d88, false}}, // 0x1.e6db1p-20
            {gelu_erf_minimax_pol, {0x36ee8229, false}}, // 0x1.dd0452p-18
            {gelu_erf_minimax_pol, {0x37b8a3bb, false}}, // 0x1.714776p-16
            {gelu_erf_minimax_pol, {0x3867a213, false}}, // 0x1.cf4426p-15
            {gelu_erf_minimax_pol, {0x3940033b, false}}, // 0x1.800676p-13
            {gelu_erf_minimax_pol, {0x3a2a5a1d, false}}, // 0x1.54b43ap-11
            {gelu_erf_minimax_pol, {0x3ae35863, false}}, // 0x1.c6b0c6p-10
            {gelu_erf_minimax_pol, {0x3b7828f2, false}}, // 0x1.f051e4p-9
            {gelu_erf_minimax_pol, {0x3c08b14b, false}}, // 0x1.116296p-7
            {gelu_erf_minimax_pol, {0x3c515ed3, false}}, // 0x1.a2bda6p-7
            {gelu_erf_minimax_pol, {0xbb503236, false}}, // -0x1.a0646cp-9
            {gelu_erf_minimax_pol, {0xbd8d8e5e, false}}, // -0x1.1b1cbcp-4
            {gelu_erf_minimax_pol, {0xbe8abcd9, false}}, // -0x1.1579b2p-2
            {gelu_erf_minimax_pol, {0xbf0c19a2, false}}, // -0x1.183344p-1
            {gelu_erf_minimax_pol, {0xbeccb328, false}}, // -0x1.99665p-2
            {gelu_erf_minimax_pol, {0x3e176ced, false}}, // 0x1.2ed9dap-3
            {gelu_erf_minimax_pol, {0x3f470d99, false}}, // 0x1.8e1b32p-1
            {gelu_erf_minimax_pol, {0x3f7abb28, false}}, // 0x1.f5765p-1
            {gelu_erf_minimax_pol, {0x3f800000, false}}, // 0x1p0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            // coefficients of degree 1
            {gelu_erf_minimax_pol, {0x3f4c422a, false}}, // 0x1.988454p-1
            {gelu_erf_minimax_pol, {0x3f4c421f, false}}, // 0x1.98843ep-1
            {gelu_erf_minimax_pol, {0x3f4c4207, false}}, // 0x1.98840ep-1
            {gelu_erf_minimax_pol, {0x3f4c41cb, false}}, // 0x1.988396p-1
            {gelu_erf_minimax_pol, {0x3f4c413b, false}}, // 0x1.988276p-1
            {gelu_erf_minimax_pol, {0x3f4c3fad, false}}, // 0x1.987f5ap-1
            {gelu_erf_minimax_pol, {0x3f4c3a2f, false}}, // 0x1.98745ep-1
            {gelu_erf_minimax_pol, {0x3f4c2d40, false}}, // 0x1.985a8p-1
            {gelu_erf_minimax_pol, {0x3f4c146a, false}}, // 0x1.9828d4p-1
            {gelu_erf_minimax_pol, {0x3f4bc341, false}}, // 0x1.978682p-1
            {gelu_erf_minimax_pol, {0x3f4ad08c, false}}, // 0x1.95a118p-1
            {gelu_erf_minimax_pol, {0x3f48f8cf, false}}, // 0x1.91f19ep-1
            {gelu_erf_minimax_pol, {0x3f45fac7, false}}, // 0x1.8bf58ep-1
            {gelu_erf_minimax_pol, {0x3f404e07, false}}, // 0x1.809c0ep-1
            {gelu_erf_minimax_pol, {0x3f3b980f, false}}, // 0x1.77301ep-1
            {gelu_erf_minimax_pol, {0x3f48dff3, false}}, // 0x1.91bfe6p-1
            {gelu_erf_minimax_pol, {0x3f78b21b, false}}, // 0x1.f16436p-1
            {gelu_erf_minimax_pol, {0x3fbb0704, false}}, // 0x1.760e08p0
            {gelu_erf_minimax_pol, {0x40019c32, false}}, // 0x1.033864p1
            {gelu_erf_minimax_pol, {0x3fe536d6, false}}, // 0x1.ca6dacp0
            {gelu_erf_minimax_pol, {0x3f81331e, false}}, // 0x1.02663cp0
            {gelu_erf_minimax_pol, {0x3e6c8684, false}}, // 0x1.d90d08p-3
            {gelu_erf_minimax_pol, {0x3c98f936, false}}, // 0x1.31f26cp-6
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x3f800000, false}}, // 0x1p0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            // coefficients of degree 2
            {gelu_erf_minimax_pol, {0xb62173f4, false}}, // -0x1.42e7e8p-19
            {gelu_erf_minimax_pol, {0x3735e4cf, false}}, // 0x1.6bc99ep-17
            {gelu_erf_minimax_pol, {0x37f2ff89, false}}, // 0x1.e5ff12p-16
            {gelu_erf_minimax_pol, {0x388c23be, false}}, // 0x1.18477cp-14
            {gelu_erf_minimax_pol, {0x3917535c, false}}, // 0x1.2ea6b8p-13
            {gelu_erf_minimax_pol, {0x39ab2ab0, false}}, // 0x1.56556p-12
            {gelu_erf_minimax_pol, {0x3a60fadb, false}}, // 0x1.c1f5b6p-11
            {gelu_erf_minimax_pol, {0x3af9b960, false}}, // 0x1.f372cp-10
            {gelu_erf_minimax_pol, {0x3b6e5491, false}}, // 0x1.dca922p-9
            {gelu_erf_minimax_pol, {0x3c0a4ec5, false}}, // 0x1.149d8ap-7
            {gelu_erf_minimax_pol, {0x3ca5aa8c, false}}, // 0x1.4b5518p-6
            {gelu_erf_minimax_pol, {0x3d2138d9, false}}, // 0x1.4271b2p-5
            {gelu_erf_minimax_pol, {0x3d8737d4, false}}, // 0x1.0e6fa8p-4
            {gelu_erf_minimax_pol, {0x3ddfb660, false}}, // 0x1.bf6ccp-4
            {gelu_erf_minimax_pol, {0x3e0f27ab, false}}, // 0x1.1e4f56p-3
            {gelu_erf_minimax_pol, {0x3d94004b, false}}, // 0x1.280096p-4
            {gelu_erf_minimax_pol, {0xbe0efdeb, false}}, // -0x1.1dfbd6p-3
            {gelu_erf_minimax_pol, {0xbf1d96c3, false}}, // -0x1.3b2d86p-1
            {gelu_erf_minimax_pol, {0xbf89db58, false}}, // -0x1.13b6bp0
            {gelu_erf_minimax_pol, {0xbf6d9897, false}}, // -0x1.db312ep-1
            {gelu_erf_minimax_pol, {0xbef69fb8, false}}, // -0x1.ed3f7p-2
            {gelu_erf_minimax_pol, {0xbdc4f8a8, false}}, // -0x1.89f15p-4
            {gelu_erf_minimax_pol, {0xbbde6422, false}}, // -0x1.bcc844p-8
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            // coefficients of degree 3
            {gelu_erf_minimax_pol, {0xbe081a19, false}}, // -0x1.103432p-3
            {gelu_erf_minimax_pol, {0xbe084570, false}}, // -0x1.108aep-3
            {gelu_erf_minimax_pol, {0xbe08639b, false}}, // -0x1.10c736p-3
            {gelu_erf_minimax_pol, {0xbe089837, false}}, // -0x1.11306ep-3
            {gelu_erf_minimax_pol, {0xbe08f409, false}}, // -0x1.11e812p-3
            {gelu_erf_minimax_pol, {0xbe09ab95, false}}, // -0x1.13572ap-3
            {gelu_erf_minimax_pol, {0xbe0b66d0, false}}, // -0x1.16cdap-3
            {gelu_erf_minimax_pol, {0xbe0e400a, false}}, // -0x1.1c8014p-3
            {gelu_erf_minimax_pol, {0xbe124df8, false}}, // -0x1.249bfp-3
            {gelu_erf_minimax_pol, {0xbe1bde02, false}}, // -0x1.37bc04p-3
            {gelu_erf_minimax_pol, {0xbe2f19c9, false}}, // -0x1.5e3392p-3
            {gelu_erf_minimax_pol, {0xbe4931bf, false}}, // -0x1.92637ep-3
            {gelu_erf_minimax_pol, {0xbe685fbc, false}}, // -0x1.d0bf78p-3
            {gelu_erf_minimax_pol, {0xbe89c95f, false}}, // -0x1.1392bep-2
            {gelu_erf_minimax_pol, {0xbe96cbca, false}}, // -0x1.2d9794p-2
            {gelu_erf_minimax_pol, {0xbe8044aa, false}}, // -0x1.008954p-2
            {gelu_erf_minimax_pol, {0xbe0550f2, false}}, // -0x1.0aa1e4p-3
            {gelu_erf_minimax_pol, {0x3dcfd6a1, false}}, // 0x1.9fad42p-4
            {gelu_erf_minimax_pol, {0x3e94c826, false}}, // 0x1.29904cp-2
            {gelu_erf_minimax_pol, {0x3e79345f, false}}, // 0x1.f268bep-3
            {gelu_erf_minimax_pol, {0x3decec91, false}}, // 0x1.d9d922p-4
            {gelu_erf_minimax_pol, {0x3ca46568, false}}, // 0x1.48cadp-6
            {gelu_erf_minimax_pol, {0x3aa1e00a, false}}, // 0x1.43c014p-10
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            // coefficients of degree 4
            {gelu_erf_minimax_pol, {0xba3d61db, false}}, // -0x1.7ac3b6p-11
            {gelu_erf_minimax_pol, {0x39f097a3, false}}, // 0x1.e12f46p-12
            {gelu_erf_minimax_pol, {0x3a5845dc, false}}, // 0x1.b08bb8p-11
            {gelu_erf_minimax_pol, {0x3ab1fa35, false}}, // 0x1.63f46ap-10
            {gelu_erf_minimax_pol, {0x3b0cefb8, false}}, // 0x1.19df7p-9
            {gelu_erf_minimax_pol, {0x3b653ab6, false}}, // 0x1.ca756cp-9
            {gelu_erf_minimax_pol, {0x3bcae527, false}}, // 0x1.95ca4ep-8
            {gelu_erf_minimax_pol, {0x3c221712, false}}, // 0x1.442e24p-7
            {gelu_erf_minimax_pol, {0x3c6c5840, false}}, // 0x1.d8b08p-7
            {gelu_erf_minimax_pol, {0x3cc0a703, false}}, // 0x1.814e06p-6
            {gelu_erf_minimax_pol, {0x3d1dcc19, false}}, // 0x1.3b9832p-5
            {gelu_erf_minimax_pol, {0x3d63656d, false}}, // 0x1.c6cadap-5
            {gelu_erf_minimax_pol, {0x3d955907, false}}, // 0x1.2ab20ep-4
            {gelu_erf_minimax_pol, {0x3dbf9910, false}}, // 0x1.7f322p-4
            {gelu_erf_minimax_pol, {0x3dd53f69, false}}, // 0x1.aa7ed2p-4
            {gelu_erf_minimax_pol, {0x3db7dcef, false}}, // 0x1.6fb9dep-4
            {gelu_erf_minimax_pol, {0x3d639ebe, false}}, // 0x1.c73d7cp-5
            {gelu_erf_minimax_pol, {0xba6ede48, false}}, // -0x1.ddbc9p-11
            {gelu_erf_minimax_pol, {0xbd22be69, false}}, // -0x1.457cd2p-5
            {gelu_erf_minimax_pol, {0xbd041cf1, false}}, // -0x1.0839e2p-5
            {gelu_erf_minimax_pol, {0xbc64f5ab, false}}, // -0x1.c9eb56p-7
            {gelu_erf_minimax_pol, {0xbb097a32, false}}, // -0x1.12f464p-9
            {gelu_erf_minimax_pol, {0xb8ebf380, false}}, // -0x1.d7e7p-14
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            // coefficients of degree 5
            {gelu_erf_minimax_pol, {0x3cb7d80c, false}}, // 0x1.6fb018p-6
            {gelu_erf_minimax_pol, {0x3c9b6050, false}}, // 0x1.36c0ap-6
            {gelu_erf_minimax_pol, {0x3c978d11, false}}, // 0x1.2f1a22p-6
            {gelu_erf_minimax_pol, {0x3c92e850, false}}, // 0x1.25d0ap-6
            {gelu_erf_minimax_pol, {0x3c8d058b, false}}, // 0x1.1a0b16p-6
            {gelu_erf_minimax_pol, {0x3c848454, false}}, // 0x1.0908a8p-6
            {gelu_erf_minimax_pol, {0x3c6cd623, false}}, // 0x1.d9ac46p-7
            {gelu_erf_minimax_pol, {0x3c4c824b, false}}, // 0x1.990496p-7
            {gelu_erf_minimax_pol, {0x3c2a7935, false}}, // 0x1.54f26ap-7
            {gelu_erf_minimax_pol, {0x3be0b390, false}}, // 0x1.c1672p-8
            {gelu_erf_minimax_pol, {0x3b0651ac, false}}, // 0x1.0ca358p-9
            {gelu_erf_minimax_pol, {0xbb232f53, false}}, // -0x1.465ea6p-9
            {gelu_erf_minimax_pol, {0xbbd42fa0, false}}, // -0x1.a85f4p-8
            {gelu_erf_minimax_pol, {0xbc2c5366, false}}, // -0x1.58a6ccp-7
            {gelu_erf_minimax_pol, {0xbc492c9e, false}}, // -0x1.92593cp-7
            {gelu_erf_minimax_pol, {0xbc2a7aa6, false}}, // -0x1.54f54cp-7
            {gelu_erf_minimax_pol, {0xbbd55d04, false}}, // -0x1.aaba08p-8
            {gelu_erf_minimax_pol, {0xba823a76, false}}, // -0x1.0474ecp-10
            {gelu_erf_minimax_pol, {0x3b102aa8, false}}, // 0x1.20555p-9
            {gelu_erf_minimax_pol, {0x3ae25a7e, false}}, // 0x1.c4b4fcp-10
            {gelu_erf_minimax_pol, {0x3a31f792, false}}, // 0x1.63ef24p-11
            {gelu_erf_minimax_pol, {0x38b84375, false}}, // 0x1.7086eap-14
            {gelu_erf_minimax_pol, {0x3689bb5a, false}}, // 0x1.1376b4p-18
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
    };

    // This object takes care about which constants and polynomials to include.
    struct need_t {
        need_t(alg_kind_t alg) {
            using namespace alg_kind;
            switch (alg) {
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu:
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp:
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                case eltwise_swish: exp_ = true; break;
                case eltwise_gelu_erf: gelu_erf_ = true; break;
                case eltwise_gelu_tanh:
                    exp_ = true;
                    gelu_tanh_ = true;
                    break;
                case eltwise_log: log_ = true; break;
                case eltwise_soft_relu: soft_relu_ = true; break;
                case eltwise_mish: mish_ = true; break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh:
                    exp_ = true;
                    tanh_ = true;
                    break;
                default: break;
            }
        }

        bool exp_ = false;
        bool mish_ = false;
        bool tanh_ = false;
        bool soft_relu_ = false;
        bool gelu_tanh_ = false;
        bool gelu_erf_ = false;
        bool log_ = false;

        bool exp() const { return exp_ || soft_relu_ || gelu_erf_ || mish_; }
        bool mish() const { return mish_; }
        bool tanh() const { return tanh_ || gelu_tanh_; }
        bool soft_relu() const { return soft_relu_; }
        bool gelu_tanh() const { return gelu_tanh_; }
        bool gelu_erf() const { return gelu_erf_; }
        bool log() const { return log_; }
    };

    need_t need(alg_);

    auto push_arg_entry_of = [&](const key_t key, const table_entry_val_t val,
                                     const bool broadcast) {
        mapped_table_entry_t te {0, val, broadcast};
        entry_map_.insert(std::make_pair(key, te));
    };

    auto push_entries_of = [&](const table_t &t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto key = (*it).first;
            auto te = (*it).second; // copy values from table
            push_arg_entry_of(key, te.val, te.bcast);
        }
    };

    push_arg_entry_of(scale, float2int(scale_), true);
    push_arg_entry_of(alpha, float2int(alpha_), true);
    push_arg_entry_of(beta, float2int(beta_), true);
    push_entries_of(common_values);
    if (need.exp()) push_entries_of(exp_consts);
    if (need.exp()) push_entries_of(exp_polynomial);
    if (need.exp()) push_entries_of(exp_consts2);
    if (need.mish()) push_entries_of(mish_consts);
    if (need.tanh()) push_entries_of(tanh_consts);
    if (need.tanh()) push_entries_of(tanh_polynomial_table);
    if (need.soft_relu()) push_entries_of(soft_relu_consts);
    if (need.soft_relu()) push_entries_of(soft_relu_polynomial);
    if (need.gelu_tanh()) push_entries_of(gelu_tanh_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_polynomial);
    if (need.gelu_erf()) push_entries_of(gelu_erf_minimax_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_minimax_polynomial);
    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        auto &te = (*it).second;
        te.off = off;
        off += te.bcast ? vlen : sizeof(table_entry_val_t);
    }
}

template struct jit_uni_eltwise_injector_f32<sve_512>;
template struct jit_uni_eltwise_injector_f32<sve_256>;
template struct jit_uni_eltwise_injector_f32<sve_128>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
