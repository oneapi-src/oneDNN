/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace eltwise_injector {

#define VCHECK_ELT_INJ_BOOL(cond, msg) \
    VCONDCHECK(primitive, create, check, binary_injector, cond, false, msg);

bool is_isa_supported(cpu_isa_t isa) {
    return is_superset(isa, sse41);
}

bool is_alg_supported(alg_kind_t alg) {
    using namespace alg_kind;
    return utils::one_of(alg, eltwise_relu, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
            eltwise_soft_relu, eltwise_logistic, eltwise_mish, eltwise_exp,
            eltwise_gelu_tanh, eltwise_hardsigmoid, eltwise_hardswish,
            eltwise_swish, eltwise_log, eltwise_clip, eltwise_clip_v2,
            eltwise_pow, eltwise_gelu_erf, eltwise_round,
            eltwise_relu_use_dst_for_bwd, eltwise_tanh_use_dst_for_bwd,
            eltwise_elu_use_dst_for_bwd, eltwise_sqrt_use_dst_for_bwd,
            eltwise_logistic_use_dst_for_bwd, eltwise_exp_use_dst_for_bwd,
            eltwise_clip_v2_use_dst_for_bwd);
}

bool is_supported(cpu_isa_t isa, alg_kind_t alg, data_type_t dt) {
    VCHECK_ELT_INJ_BOOL(dt == data_type::f32, VERBOSE_UNSUPPORTED_DT);
    VCHECK_ELT_INJ_BOOL(is_isa_supported(isa), VERBOSE_UNSUPPORTED_ISA);
    VCHECK_ELT_INJ_BOOL(is_alg_supported(alg), "Unsupported algorithm");
    return true;
}

#undef VCHECK_ELT_INJ_BOOL

} // namespace eltwise_injector

using namespace Xbyak;

template <cpu_isa_t isa, typename Wmm>
size_t jit_uni_eltwise_injector<isa, Wmm>::get_stack_vmm_space() {
    return (save_state_ * preserve_vmm_ * n_vregs_to_preserve_
                   + op_vecs_count(alg_, is_fwd_))
            * vlen_;
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::injector_preamble(
        const injector_utils::vmm_index_set_t &vmm_compute_idxs,
        injector_utils::vmm_index_set_iterator_t &start_idx_tail_it,
        const injector_utils::vmm_index_set_t &vmm_aux_indices) {
    using namespace Xbyak::util;

    // Mask register is a Vmm register on AVX2 and below. It qualifies as an
    // additional register to preserve.
    need_vmm_mask_register_ = need_mask_register(alg_, is_fwd_, alpha_);

    n_vregs_preserved_ = 0;
    assert(IMPLICATION(!vmm_aux_indices.empty(),
            vmm_aux_indices.size() == n_vregs_to_preserve_));

    const auto start_idx = *(vmm_compute_idxs.begin());
    const auto end_idx = *(vmm_compute_idxs.rbegin()) + 1;
    // For sse41 mask register must be Xmm(0), reserve it unconditionally.
    if (isa == sse41 && need_vmm_mask_register_) {
        static constexpr int mask_idx = 0;
        assert(start_idx > mask_idx);
        // When external indices come, they must secure first index is 0.
        assert(IMPLICATION(!vmm_aux_indices.empty(),
                *(vmm_aux_indices.begin()) == mask_idx));
        preserved_vmm_tail_indices_[n_vregs_preserved_++] = mask_idx;
    }

    // `n_vregs_preserved_` is 0 for isa higher than `sse41`.
    // Note that by happy coincidence, idx=0 is skipped for `sse41`.
    for (size_t idx = n_vregs_preserved_; idx < n_vregs_; idx++) {
        // Once reserved enough vmm registers, break the loop.
        if (n_vregs_preserved_ >= n_vregs_to_preserve_) break;

        // Thanks to `std::set` LegacyBidirectionalIterator `iterator` member
        // that doesn't have operator+ defined.
        size_t external_idx = 0;
        if (!vmm_aux_indices.empty()) {
            auto it = vmm_aux_indices.begin();
            for (size_t i = 0; i < idx; i++)
                it++;
            external_idx = *it;
            assert(it != vmm_aux_indices.end());
        }
        size_t preserve_idx = vmm_aux_indices.empty() ? idx : external_idx;

        // `start_idx` and `end_idx` is the range of indices passed to the
        // injector to apply `alg_` on top of them. Thus, they don't need to
        // be preserved.
        if (vmm_aux_indices.empty()) {
            // When injector decides on indices to preserve, it skips those to
            // compute alg on...
            if (start_idx <= preserve_idx && preserve_idx < end_idx) continue;
        } else {
            // ... but when indices are passed, it secures that external indices
            // don't overlap with those to compute alg on.
            assert(!(start_idx <= preserve_idx && preserve_idx < end_idx));
        }

        // Preserve vmm mask register first. If preserved last, there may not be
        // enough free registers to fit all vmm registers, and an algorithm
        // requiring a mask will fail.
        // Note: sse41 has it preserved already.
        if (need_vmm_mask_register_ && n_vregs_preserved_ == 0)
            preserved_vmm_tail_indices_[n_vregs_preserved_++] = preserve_idx;
        else {
            preserved_vmm_indices_[n_vregs_preserved_ - need_vmm_mask_register_]
                    = preserve_idx;
            n_vregs_preserved_++;
        }
    }

    // If it happened that there was not enough spare registers to preserve,
    // injector will take first `n_vregs_not_preserved` from `vmm_compute_idxs`
    // to have legit generated code. This fact is saved through
    // `start_idx_tail_it` iterator and a second round of compute will happen.
    size_t n_vregs_not_preserved = n_vregs_to_preserve_ - n_vregs_preserved_;
    for (size_t i = 0; i < n_vregs_not_preserved; i++) {
        preserved_vmm_indices_[n_vregs_preserved_ - need_vmm_mask_register_]
                = *start_idx_tail_it;
        n_vregs_preserved_++;
        start_idx_tail_it++;
    }
    assert(n_vregs_preserved_ == n_vregs_to_preserve_);

    // Preserve GPRs.
    size_t preserved_gprs_count = 0;
    size_t n_gprs_to_preserve = aux_gprs_count(alg_, is_fwd_, alpha_);
    if (n_gprs_to_preserve > 0) {
        // Allocate GPRs from the end not to mess with ABI compatibility.
        for (int gpr_idx = Operand::R15; gpr_idx >= 0; gpr_idx--) {
            // Restrict using stack GPR and table address GPR as temporary.
            if (preserved_gprs_count < n_gprs_to_preserve
                    && !utils::one_of(gpr_idx, p_table_.getIdx(), Operand::RSP))
                preserved_gpr_indices_[preserved_gprs_count++] = gpr_idx;
        }
        assert(preserved_gprs_count == n_gprs_to_preserve);
    }

    if (need_vmm_stack_ptr(alg_, is_fwd_, alpha_)) {
        reg_vmm_stack_ptr_ = Reg64(preserved_gpr_indices_[0]);
    }

    if (save_state_) {
        if (preserve_p_table_) h->push(p_table_);

        for (size_t i = 0; i < preserved_gprs_count; ++i)
            h->push(Reg64(preserved_gpr_indices_[i]));
    }

    const auto stack_vmm_space = get_stack_vmm_space();
    if (stack_vmm_space) {
        // - Let's align stack space used for vmm spilling at runtime. To do
        // this we pad the rsp, and allocate the pre-estimated space required.
        // - To keep regular gpr spilling as-is we use another register to track
        // the vmm_stack_space.
        // - Finally, the original stack pointer (rsp) is stored just above the
        // vmm stack space, to revert back to address before padding.
        h->mov(reg_vmm_stack_ptr_, h->rsp);
        h->sub(h->rsp, 8);
        const uint32_t mask = ~(static_cast<uint32_t>(vlen_) - 1);
        h->and_(h->rsp, mask);
        h->mov(ptr[h->rsp], reg_vmm_stack_ptr_);
        h->sub(h->rsp, stack_vmm_space);
        h->mov(reg_vmm_stack_ptr_, h->rsp);
    }

    if (save_state_) {
        if (preserve_vmm_) {
            // External indices imply no vmm preservation happens.
            assert(vmm_aux_indices.empty());

            if (need_vmm_mask_register_) {
                size_t i = 0;
                h->uni_vmovups(h->ptr[reg_vmm_stack_ptr_ + i * vlen_],
                        Vmm(preserved_vmm_tail_indices_[i]));
            }

            for (size_t i = need_vmm_mask_register_; i < n_vregs_preserved_;
                    ++i)
                h->uni_vmovups(h->ptr[reg_vmm_stack_ptr_ + i * vlen_],
                        Vmm(preserved_vmm_indices_[i
                                - need_vmm_mask_register_]));
            if (stack_vmm_space)
                h->add(reg_vmm_stack_ptr_, n_vregs_preserved_ * vlen_);
        }

        load_table_addr();
    }

    assign_regs();
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::injector_preamble_tail(
        size_t n_vregs_not_preserved) {
    // There was enough vmm registers to compute everything in one round.
    if (n_vregs_not_preserved == 0) return;

    const size_t idx_off = n_vregs_to_preserve_ - n_vregs_not_preserved;
    assert(idx_off < n_vregs_to_preserve_); // Overflow is undesired.

    if (save_state_) {
        // TODO: either this requires `preserve_vmm_` to be checked, or it's
        // an issue and `!preserve_vmm_` case never reaches this piece.
        assert(preserve_vmm_);

        for (size_t i = 0; i < n_vregs_not_preserved; ++i)
            h->uni_vmovups(Vmm(preserved_vmm_indices_[idx_off + i
                                   - need_vmm_mask_register_]),
                    h->ptr[reg_vmm_stack_ptr_
                            + (i - n_vregs_not_preserved) * vlen_]);
    }

    // Update the rightmost indices. The injector uses vmms with indices coming
    // after compute vmm indices.
    // TODO: is it always a valid index?
    for (size_t i = 0; i < n_vregs_not_preserved; ++i)
        preserved_vmm_indices_[idx_off + i - need_vmm_mask_register_]
                += n_vregs_not_preserved;

    if (save_state_ && preserve_vmm_) {
        for (size_t i = 0; i < n_vregs_not_preserved; ++i)
            h->uni_vmovups(h->ptr[reg_vmm_stack_ptr_
                                   + (i - n_vregs_not_preserved) * vlen_],
                    Vmm(preserved_vmm_indices_[idx_off + i
                            - need_vmm_mask_register_]));
    }

    assign_regs();
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::injector_postamble() {
    using namespace Xbyak::util;
    const int stack_vmm_space = get_stack_vmm_space();

    if (save_state_ && preserve_vmm_) {
        for (size_t i = need_vmm_mask_register_; i < n_vregs_preserved_; ++i)
            h->uni_vmovups(
                    Vmm(preserved_vmm_indices_[i - need_vmm_mask_register_]),
                    h->ptr[reg_vmm_stack_ptr_
                            + (i - n_vregs_preserved_) * vlen_]);

        if (need_vmm_mask_register_) {
            size_t i = 0;
            h->uni_vmovups(Vmm(preserved_vmm_tail_indices_[i]),
                    h->ptr[reg_vmm_stack_ptr_
                            + (i - n_vregs_preserved_) * vlen_]);
        }
        if (n_vregs_preserved_)
            h->mov(h->rsp,
                    ptr[reg_vmm_stack_ptr_
                            + op_vecs_count(alg_, is_fwd_) * vlen_]);
    } else if (stack_vmm_space) {
        h->mov(h->rsp, ptr[reg_vmm_stack_ptr_ + stack_vmm_space]);
    }

    if (!save_state_) return;
    for (int i = static_cast<int>(aux_gprs_count(alg_, is_fwd_, alpha_)) - 1;
            i >= 0; --i)
        h->pop(Reg64(preserved_gpr_indices_[i]));

    if (preserve_p_table_) h->pop(p_table_);
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::assign_regs() {
    vmm_mask_ = Vmm(preserved_vmm_tail_indices_[0]);

    // For avx we need a register to save the upper part of Ymm
    // Note: the number in `aux_vecs_count` is accounted for here.
    using namespace alg_kind;
    const bool preserve_vec_for_avx = isa == avx
            && utils::one_of(alg_, eltwise_tanh, eltwise_elu, eltwise_abs,
                    eltwise_soft_relu, eltwise_mish, eltwise_logistic,
                    eltwise_exp, eltwise_gelu_tanh, eltwise_swish,
                    eltwise_gelu_erf, eltwise_tanh_use_dst_for_bwd,
                    eltwise_elu_use_dst_for_bwd,
                    eltwise_logistic_use_dst_for_bwd,
                    eltwise_exp_use_dst_for_bwd);

    if (preserve_vec_for_avx) {
        vmm_tmp_ = Vmm(preserved_vmm_indices_[n_vregs_to_preserve_ - 1]);
        ymm_tmp_ = Ymm(preserved_vmm_indices_[n_vregs_to_preserve_ - 1]);
        xmm_tmp_ = Xmm(preserved_vmm_indices_[n_vregs_to_preserve_ - 1]);
    }
}

// This function provides vmm aux registers based on indices from
// `preserved_vmm_indices_`. This is an internal container which can be
// initialized with stock values from the injector, or with external values
// provided by the user.
template <cpu_isa_t isa, typename Wmm>
Wmm jit_uni_eltwise_injector<isa, Wmm>::vmm_aux(size_t idx) {
    assert(idx < (n_vregs_preserved_ - need_vmm_mask_register_));
    return Vmm(preserved_vmm_indices_[idx]);
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::vec_shift(const Vmm &vmm_dst,
        const Vmm &vmm_src, bool shift_left, const int imm) {
    if (isa != avx) {
        if (shift_left)
            h->uni_vpslld(vmm_dst, vmm_src, imm);
        else
            h->uni_vpsrld(vmm_dst, vmm_src, imm);
    } else {
        // Declare appropriate vectors to use non-uni instructions
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());
        Ymm ymm_src = Ymm(vmm_src.getIdx());
        if (vmm_dst.getIdx() != vmm_src.getIdx()) h->vmovups(ymm_dst, ymm_src);
        h->vextractf128(xmm_tmp_, ymm_dst, 1);
        if (shift_left) {
            h->vpslld(xmm_dst, xmm_dst, imm);
            h->vpslld(xmm_tmp_, xmm_tmp_, imm);
        } else {
            h->vpsrld(xmm_dst, xmm_dst, imm);
            h->vpsrld(xmm_tmp_, xmm_tmp_, imm);
        }
        h->vinsertf128(ymm_dst, ymm_dst, xmm_tmp_, 1);
    }
}

// Uses injector masks objects: k_mask_ (>= avx512_core) or vmm_mask_ (<= avx2).
// Stores a mask by applying cmpps on two inputs w/ a given predicate.
template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::compute_cmp_mask(const Vmm &vmm_src,
        const Xbyak::Operand &compare_operand, int cmp_predicate) {
    if (is_avx512_) {
        h->vcmpps(k_mask_, vmm_src, compare_operand, cmp_predicate);
    } else {
        h->uni_vcmpps(vmm_mask_, vmm_src, compare_operand, cmp_predicate);
    }
}

// Uses injector masks objects: k_mask_ (>= avx512_core) or vmm_mask_ (<= avx2).
// Blends a result of second input into a first input w/ a stored mask.
template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::blend_with_mask(
        const Vmm &vmm_dst, const Xbyak::Operand &src) {
    if (is_avx512_) {
        h->vblendmps(vmm_dst | k_mask_, vmm_dst, src);
    } else {
        h->uni_vblendvps(vmm_dst, vmm_dst, src, vmm_mask_);
    }
}

// Uses injector masks objects: k_mask_ (>= avx512_core) or vmm_mask_ (<= avx2).
// Tests a mask for all zeros. If all zeroes occur, set ZF = 1.
// Nicely combines with jump_if_zero (jz).
template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::test_mask() {
    if (is_avx512_) {
        h->kortestw(k_mask_, k_mask_);
    } else {
        h->uni_vtestps(vmm_mask_, vmm_mask_);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::exp_compute_vector_fwd(
        const Vmm &vmm_src) {
    // exp(x) =
    // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
    // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    h->uni_vmovups(vmm_aux(0), table_val(exp_ln_flt_min_f));

    compute_cmp_mask(vmm_src, vmm_aux(0), _cmp_lt_os);

    h->uni_vminps(vmm_src, vmm_src, table_val(exp_ln_flt_max_f));
    h->uni_vmaxps(vmm_src, vmm_src, vmm_aux(0));
    h->uni_vmovups(vmm_aux(0), vmm_src);

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(exp_log2ef));
    h->uni_vaddps(vmm_src, vmm_src, table_val(half));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux(1), vmm_src, _op_floor);

    // keep vmm_src = fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux(1));

    // x = x - fx * ln2
    h->uni_vfnmadd231ps(vmm_aux(0), vmm_aux(1), table_val(ln2f));

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    h->uni_vsubps(vmm_src, vmm_src, table_val(one));
    h->uni_vcvtps2dq(vmm_aux(1), vmm_src);
    if (isa != avx)
        h->uni_vpaddd(vmm_aux(1), vmm_aux(1), table_val(exponent_bias));
    else {
        Ymm ymm_aux2 = Ymm(vmm_aux(1).getIdx());
        Xmm xmm_aux2 = Xmm(vmm_aux(1).getIdx());
        h->vextractf128(xmm_tmp_, ymm_aux2, 1);
        h->vpaddd(xmm_tmp_, xmm_tmp_, table_val(exponent_bias));
        h->vpaddd(xmm_aux2, xmm_aux2, table_val(exponent_bias));
        h->vinsertf128(ymm_aux2, ymm_aux2, xmm_tmp_, 1);
    }
    vec_shift(vmm_aux(1), vmm_aux(1), true /*shift_left*/, n_mantissa_bits_);
    // use vmm_src as tmp vmm_zero when applying mask
    h->uni_vxorps(vmm_src, vmm_src, vmm_src);
    // set zeroes at those points which were < log(FLT_MIN)
    blend_with_mask(vmm_aux(1), vmm_src);

    // compute polynomial
    h->uni_vmovups(vmm_src, table_val(exp_pol, 4));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(0), table_val(exp_pol, 3));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(0), table_val(exp_pol, 2));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(0), table_val(exp_pol, 1));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(0), table_val(exp_pol, 0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(0), table_val(one));
    // y = y * 2^n
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(1));
    h->uni_vmulps(vmm_src, vmm_src, table_val(two));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux(0), vmm_src);
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    blend_with_mask(vmm_src, vmm_aux(0));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::relu_zero_ns_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::elu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // IMPORTANT: we use vmm_aux(2) for the mask as exp_compute does not use it.
    h->uni_vmovups(vmm_aux(2), vmm_src);
    // compute exponent
    exp_compute_vector_fwd(vmm_src);

    // alpha * (exp(x) - 1)
    h->uni_vsubps(vmm_src, vmm_src, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));

    // combine with mask
    compute_cmp_mask(vmm_aux(2), table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux(2));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    // we add a check as the avx2 code cannot be used for avx
    assert(IMPLICATION(isa == avx2, mayiuse(avx2)));

    using namespace Xbyak::util;
    const int XMM_float_lanes_count = 4;
    const int tanh_n_polynomials = 32;

    // register mapping
    // TODO: put sign on stack and alias zmm_table2 with vmm_sign to save a reg ?
    Vmm vmm_dst = vmm_aux(0);
    Vmm vmm_src_shift = vmm_aux(0);
    Vmm vmm_coeff = vmm_aux(0);
    Vmm vmm_pol = vmm_aux(1);
    Vmm vmm_indices = vmm_aux(2);
    Vmm vmm_tmp = is_avx512_ ? vmm_aux(2) // index `2` works for AVX512_CORE
                             : vmm_mask_; // TODO: why `vmm_mask_` directly?
    Vmm vmm_src_original = vmm_aux(3);
    Vmm vmm_sign = vmm_aux(3);
    Reg64 gpr_idx[XMM_float_lanes_count];

    if (isa == sse41 || isa == avx) {
        assert(aux_gprs_count(alg_, is_fwd_, alpha_) >= XMM_float_lanes_count);
        for (int i = 0; i < XMM_float_lanes_count; i++)
            gpr_idx[i] = Reg64(preserved_gpr_indices_[i
                    + need_vmm_stack_ptr(alg_, is_fwd_, alpha_)]);
    }

    // We split the positive domain in 33 intervals:
    // a) [0; linear_ubound]: in this interval tanh(x) = x
    // b) [linear_ubound; 0x1.8p-12]: This interval spans part of a
    //    half binade
    // c) [0x1.8p-12; 0x1.0p-11], ..., [0x1.8p2; 0x1.0p3]:
    //    one interval for each half binade, there are 29 of those
    // d) [0x1.0p3; saturation_ubound]:
    //    This interval spans part of a half binade
    // e) [0x1.205966p3; saturation_ubound]: in this interval, tanh(x) = 1
    // For b-d, we need 31 polynomials and will do a table lookup for those.
    // To simplify the logic, we will also put a) in the table.

    // The polynomials are of degree 6, so we need to gather 7 coefficients.
    // - sse4.1: we do it the naive way using vextract/vinsert.
    //           Here we will extract the indices in gpr only once and
    //           reuse them as there are only 4 of them.
    // - avx: we do the same as for sse4.1 but use half of the 64-bits
    //           registers to store the idx of second half of YMM and half for
    //           responding XMM. Halfway through the copy we exchange Xmm and
    //           higher half of Ymm and we get the expected result.
    // - avx2: we use vpermps and blend for each coefficient.
    //         This needs an extra vmm to store the mask
    // - avx512: because the table fits in 2 registers, we can use vpermi2d.
    auto coeffs_off = [&](int coeff_off, int off = 0) {
        return table_off(tanh_pol_table, coeff_off * tanh_n_polynomials + off);
    };
    auto coeffs_address = [&](int coeff_off, int off = 0) {
        return table_val(tanh_pol_table, coeff_off * tanh_n_polynomials + off);
    };
    auto gather_coefficient_init = [&](Vmm vmm_pol_idx, int nelems) {
        switch (isa) {
            case sse41:
                for (int i = 0; i < XMM_float_lanes_count; ++i)
                    h->pextrd(gpr_idx[i].cvt32(), vmm_pol_idx, i);
                break;
            case avx: {
                Xmm xmm_pol_idx = Xmm(vmm_pol_idx.getIdx());
                for (int i = 0; i < XMM_float_lanes_count; ++i)
                    h->vpextrd(gpr_idx[i].cvt32(), xmm_pol_idx, i);
            } break;
            case avx2_vnni_2:
            case avx2:
                // needed for gather instruction
                h->uni_vxorps(vmm_mask_, vmm_mask_, vmm_mask_);
                break;
            case avx512_core_fp16:
            case avx512_core_bf16:
            case avx512_core: break;
            default: assert(!"unimplemented");
        }
    };
    auto gather_coefficient = [&](Vmm vmm_coeff, int coeff_idx,
                                      Vmm vmm_pol_idx) {
        switch (isa) {
            case sse41:
                for (int idx = 0; idx < 4; ++idx) {
                    Xbyak::Address coeff_addr
                            = ptr[p_table_ + coeffs_off(coeff_idx)
                                    + gpr_idx[idx] * sizeof(float)];
                    h->pinsrd(vmm_coeff, coeff_addr, idx);
                }
                break;
            case avx: {
                Xmm xmm_coeff = Xmm(vmm_coeff.getIdx());
                for (int idx = 0; idx < 4; ++idx) {
                    Xbyak::Address coeff_addr
                            = ptr[p_table_ + coeffs_off(coeff_idx)
                                    + gpr_idx[idx] * sizeof(float)];
                    h->vpinsrd(xmm_coeff, xmm_coeff, coeff_addr, idx);
                }
            } break;
            case avx2_vnni_2:
            case avx2: {
                Xbyak::Address idx_addr = ptr[p_table_ + coeffs_off(coeff_idx)
                        + vmm_pol_idx * sizeof(float)];
                // we set the mask to all ones to gather full
                // register.  needs to be done after each gather since
                // since the gather instructions zeros the mask if
                // successful
                h->uni_vcmpps(vmm_mask_, vmm_mask_, vmm_mask_, _cmp_eq_oq);
                h->vgatherdps(vmm_coeff, idx_addr, vmm_mask_);
                break;
            }
                // use gather instruction
            case avx512_core_fp16:
            case avx512_core_bf16:
            case avx512_core:
                // we use vpermt2ps to not override the indices
                // this also enables to save a register for table loading
                {
                    Zmm zmm_coeff(vmm_coeff.getIdx());
                    Zmm zmm_pol_idx(vmm_pol_idx.getIdx());
                    h->uni_vmovups(zmm_coeff, coeffs_address(coeff_idx, 0));
                    h->vpermt2ps(zmm_coeff, zmm_pol_idx,
                            coeffs_address(coeff_idx, 16));
                    break;
                }
            default: assert(!"unimplemented");
        }
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x postive
    // and reapply sign at the end
    h->uni_vmovups(vmm_src_original, vmm_src);
    h->uni_vandps(vmm_src, vmm_src, table_val(positive_mask));

    // We compute the indices for the table lookup
    h->uni_vmovups(vmm_indices, vmm_src);
    if (isa != avx)
        h->uni_vpsubd(vmm_indices, vmm_indices, table_val(tanh_idx_bias));
    else {
        Ymm ymm_indices = Ymm(vmm_indices.getIdx());
        Xmm xmm_indices = Xmm(vmm_indices.getIdx());
        h->vextractf128(xmm_tmp_, ymm_indices, 1);
        h->vpsubd(xmm_tmp_, xmm_tmp_, table_val(tanh_idx_bias));
        h->vpsubd(xmm_indices, xmm_indices, table_val(tanh_idx_bias));
        h->vinsertf128(ymm_indices, ymm_indices, xmm_tmp_, 1);
    }
    h->uni_vandps(vmm_indices, vmm_indices, table_val(tanh_idx_mask));
    vec_shift(vmm_indices, vmm_indices, false, 22);

    // we do the argument reduction
    h->uni_vmovups(vmm_src_shift, vmm_src);
    h->uni_vandps(vmm_src_shift, vmm_src_shift, table_val(tanh_idx_mask));
    h->uni_vsubps(vmm_src, vmm_src, vmm_src_shift);

    // we gather and evaluate the polynonials
    gather_coefficient_init(vmm_indices, vlen_ / sizeof(float));
    gather_coefficient(vmm_pol, 6, vmm_indices);
    for (int deg = 5; deg >= 0; --deg) {
        gather_coefficient(vmm_coeff, deg, vmm_indices);
        h->uni_vfmadd213ps(vmm_pol, vmm_src, vmm_coeff);
    }

    if (isa == avx) {
        Ymm ymm_indices = Ymm(vmm_indices.getIdx());
        Ymm ymm_pol = Ymm(vmm_pol.getIdx());
        Ymm ymm_src = Ymm(vmm_src.getIdx());
        Xmm xmm_src = Xmm(vmm_src.getIdx());
        Xmm xmm_coeff = Xmm(vmm_coeff.getIdx());

        h->vperm2f128(ymm_src, ymm_src, ymm_src, 1);
        h->vperm2f128(ymm_indices, ymm_indices, ymm_indices, 1);
        gather_coefficient_init(vmm_indices, vlen_ / sizeof(float));
        gather_coefficient(vmm_tmp_, 6, vmm_indices);
        for (int deg = 5; deg >= 0; --deg) {
            gather_coefficient(vmm_coeff, deg, vmm_indices);
            h->vmulps(xmm_tmp_, xmm_tmp_, xmm_src);
            h->vaddps(xmm_tmp_, xmm_tmp_, xmm_coeff);
        }
        h->vinsertf128(ymm_pol, ymm_pol, xmm_tmp_, 1);
    }

    // we restore src with cleared sign, and keep sign
    assert(vmm_sign.getIdx() == vmm_src_original.getIdx());
    h->uni_vmovups(vmm_src, vmm_src_original);
    h->uni_vandps(vmm_sign, vmm_sign, table_val(sign_mask));
    h->uni_vandps(vmm_src, vmm_src, table_val(positive_mask));

    // Now we blend the results
    // [saturation_ubound; +inf[ : we return +/- 1
    h->uni_vmovups(vmm_dst, table_val(one));
    // [linear_ubound; saturation_lbound] : we return +/- P(x)
    h->uni_vmovups(vmm_tmp, table_val(tanh_saturation_lbound));
    compute_cmp_mask(vmm_tmp, vmm_src, _cmp_gt_os);
    blend_with_mask(vmm_dst, vmm_pol);
    // [0; linear_ubound]  : we return x
    h->uni_vmovups(vmm_tmp, table_val(tanh_linear_ubound));
    compute_cmp_mask(vmm_tmp, vmm_src, _cmp_gt_os);
    blend_with_mask(vmm_dst, vmm_src);

    // We reapply the sign and return
    h->uni_vxorps(vmm_dst, vmm_dst, vmm_sign);
    h->uni_vmovups(vmm_src, vmm_dst);
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::gelu_tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux(0), vmm_src);

    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vmovups(vmm_aux(1), table_val(gelu_tanh_fitting_const));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(1), table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(0));
    h->uni_vmulps(vmm_src, vmm_src, table_val(gelu_tanh_sqrt_two_over_pi));

    // save x on stack as tanh uses vmm_aux0
    h->uni_vmovups(h->ptr[reg_vmm_stack_ptr_], vmm_aux(0));

    // compute tanh(G(x))
    tanh_compute_vector_fwd(vmm_src);

    h->uni_vmovups(vmm_aux(0), h->ptr[reg_vmm_stack_ptr_]);

    // compute 0.5 * x * (1 + tanh(G(x)))
    h->uni_vaddps(vmm_src, vmm_src, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, table_val(half));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(0));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::square_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::abs_compute_vector_fwd(
        const Vmm &vmm_src) {
    // compute abs(x) = _mm_and_ps(x, 01111..111));
    h->uni_vandps(vmm_src, vmm_src, table_val(positive_mask));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::sqrt_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vsqrtps(vmm_src, vmm_src);
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::linear_compute_vector_fwd(
        const Vmm &vmm_src) {
    // compute x = alpha * x + beta;
    h->uni_vmovups(vmm_aux(0), table_val(alpha));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(0), table_val(beta));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::clip_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmaxps(vmm_src, vmm_src, table_val(alpha));
    h->uni_vminps(vmm_src, vmm_src, table_val(beta));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::mish_compute_vector_fwd(
        const Vmm &vmm_src) {
    // An equation other than mish(x) = x*tanh(srelu(x)) was used
    // to calculate mish, but it should be remembered that it is equivalent
    // equation, it uses the following rule:
    // tanh(x) = (e^x - e^-x) / (e^x + e^-x),
    // hence the equation for mish can take the form:
    // mish(x) = x * ((e^x + 1)^2 - 1)/((e^x + 1)^2 + 1).
    // This option was chosen because computing tanh requires more registers
    // than exp, and also requires more constants to be stored in memory,
    // making the algorithm slower.

    // IMPORTANT: we use vmm_aux(2) to save src as exp does not use it.
    h->uni_vmovups(vmm_aux(2), vmm_src); // vmm_aux(2) = x

    h->uni_vminps(vmm_src, vmm_src, table_val(fwd_mish_max_x_for_equation_f));
    exp_compute_vector_fwd(vmm_src);

    // (e^x+1)^2
    h->uni_vaddps(vmm_src, vmm_src, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);

    // save (e^x+1)^2 as it appears in both the denominator and the numerator
    h->uni_vmovups(vmm_aux(0), vmm_src);

    // x * ((e^x + 1)^2 - 1) / ((e^x + 1)^2 + 1)
    h->uni_vsubps(vmm_src, vmm_src, table_val(one));
    h->uni_vaddps(vmm_aux(0), vmm_aux(0), table_val(one));
    h->uni_vdivps(vmm_src, vmm_src, vmm_aux(0));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(2));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::hardswish_compute_vector_fwd(
        const Vmm &vmm_src) {
    // result = x * hardsigmoid(x)
    h->uni_vmovups(vmm_aux(0), vmm_src);
    hardsigmoid_compute_vector_fwd(vmm_src);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(0));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::hardsigmoid_compute_vector_fwd(
        const Vmm &vmm_src) {
    // result = max(0, min(1, alpha * x + beta))
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    h->uni_vaddps(vmm_src, vmm_src, table_val(beta));
    h->uni_vminps(vmm_src, vmm_src, table_val(one));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::soft_relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // alpha scaling
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));

    // ln(1 + exp(x)) =
    // = ln(1 + exp(n * ln(2) + r)) // divide x by ln(2) and get quot and rem
    // = ln(1 + 2^n * exp(r)) // simplify the exp(n*ln(2)) expression
    // = ln(2 ^ 0 + 2^n * exp(r)) // note 1 = 2^0
    // = ln(2 ^ (n - n) + 2^n * exp(r)) // 2^0 = 2^(n-n)
    // = ln(2 ^ n * (2^-n + exp(r))) // factorize with 2^n
    // = n * ln(2) + ln(2^-n + exp(r)) // take the 2^n factor out of the ln

    // keep src for further computations
    h->uni_vmovups(vmm_aux(2), vmm_src);

    h->uni_vminps(vmm_src, vmm_src, table_val(exp_ln_flt_max_f));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(exp_ln_flt_min_f));
    h->uni_vmovups(vmm_aux(1), vmm_src);

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(exp_log2ef));
    h->uni_vaddps(vmm_src, vmm_src, table_val(half));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux(0), vmm_src, _op_floor);

    // keep vmm_src = fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux(0));

    // x = x - fx * ln2
    h->uni_vmulps(vmm_aux(0), vmm_aux(0), table_val(ln2f));
    h->uni_vsubps(vmm_aux(1), vmm_aux(1), vmm_aux(0));
    // compute exponent polynomial
    h->uni_vmovups(vmm_aux(3), table_val(exp_pol, 4));
    h->uni_vfmadd213ps(vmm_aux(3), vmm_aux(1), table_val(exp_pol, 3));
    h->uni_vfmadd213ps(vmm_aux(3), vmm_aux(1), table_val(exp_pol, 2));
    h->uni_vfmadd213ps(vmm_aux(3), vmm_aux(1), table_val(exp_pol, 1));
    h->uni_vfmadd213ps(vmm_aux(3), vmm_aux(1), table_val(exp_pol, 0));
    h->uni_vfmadd213ps(vmm_aux(3), vmm_aux(1), table_val(one));

    // We do not count 2^-n here, because n can reach 128 and 2^(-128) is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^-n + exp(r) will be counted (2^-(n-1) + 2*exp(r))/2, because 2^(-127)
    // and 2 are numbers representable in fp32.

    // compute 2^-(n-1)
    // vmm_src now represents n-1
    h->uni_vsubps(vmm_src, vmm_src, table_val(one));
    if (is_avx512_) {
        h->vmulps(vmm_aux(1), vmm_src, table_val(minus_one));
        h->vcvtps2dq(vmm_aux(1), vmm_aux(1));
    } else if (isa == avx) {
        h->uni_vxorps(vmm_aux(1), vmm_src, table_val(sign_mask));
        h->uni_vcvtps2dq(vmm_aux(1), vmm_aux(1));
    } else {
        h->uni_vcvtps2dq(vmm_aux(1), vmm_src);
        h->uni_vpsignd(vmm_aux(1), vmm_aux(1), table_val(minus_one));
    }
    // restore vmm_src to n
    h->uni_vaddps(vmm_src, vmm_src, table_val(one));

    if (isa != avx)
        h->uni_vpaddd(vmm_aux(1), vmm_aux(1), table_val(exponent_bias));
    else {
        Ymm ymm_aux1 = Ymm(vmm_aux(1).getIdx());
        Xmm xmm_aux1 = Xmm(vmm_aux(1).getIdx());
        h->vextractf128(xmm_tmp_, ymm_aux1, 1);
        h->vpaddd(xmm_tmp_, xmm_tmp_, table_val(exponent_bias));
        h->vpaddd(xmm_aux1, xmm_aux1, table_val(exponent_bias));
        h->vinsertf128(ymm_aux1, ymm_aux1, xmm_tmp_, 1);
    }
    vec_shift(vmm_aux(1), vmm_aux(1), true /*shift_left*/, n_mantissa_bits_);
    // calculate ln(1 + y)
    h->uni_vmulps(vmm_aux(3), vmm_aux(3), table_val(two)); // 2*exp(r)
    h->uni_vaddps(vmm_aux(3), vmm_aux(3), vmm_aux(1)); // 2^-(n-1) + 2*exp(r)
    h->uni_vdivps(
            vmm_aux(3), vmm_aux(3), table_val(two)); // (2^-(n-1) + 2*exp(r))/2
    // frexp()
    vec_shift(vmm_src, vmm_aux(3), false /*shift_left*/, n_mantissa_bits_);
    h->uni_vcvtdq2ps(vmm_src, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->uni_vsubps(vmm_src, vmm_src, table_val(soft_relu_one_twenty_six));

    // and with mask (to get 0.5 * mantissa)
    h->uni_vandps(
            vmm_aux(3), vmm_aux(3), table_val(soft_relu_mantissa_sign_mask));
    // got y. (mantisa)  0.5 < y < 1 (or with (to get 0.5 * mantissa))
    h->uni_vorps(vmm_aux(3), vmm_aux(3), table_val(half));
    // y  = y - 1
    h->uni_vsubps(vmm_aux(3), vmm_aux(3), table_val(one));

    // compute log1p polynomial
    h->uni_vmovups(vmm_aux(1), table_val(soft_relu_pol, 8));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(3), table_val(soft_relu_pol, 7));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(3), table_val(soft_relu_pol, 6));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(3), table_val(soft_relu_pol, 5));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(3), table_val(soft_relu_pol, 4));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(3), table_val(soft_relu_pol, 3));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(3), table_val(soft_relu_pol, 2));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(3), table_val(soft_relu_pol, 1));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(3), table_val(soft_relu_pol, 0));
    //calculate ln(2) * n
    h->uni_vmulps(vmm_src, vmm_src, table_val(ln2f));
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux(1));
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux(0));

    // get vmm_mask_ = src > max logf
    // y = (x < max log f) ? soft_relu(x) : x
    compute_cmp_mask(vmm_aux(2), table_val(exp_ln_flt_max_f), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux(2));
    if (alpha_ == 1.f) { // standard soft_relu case
        // Skip an instruction.
    } else if (alpha_ == -1) { // logsigmoid case
        h->uni_vmulps(vmm_src, vmm_src, table_val(minus_one));
    } else { // General case.
        h->uni_vdivps(vmm_src, vmm_src, table_val(alpha));
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::logistic_compute_vector_fwd(
        const Vmm &vmm_src) {
    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.

    // IMPORTANT: we use vmm_aux(2) for the mask as exp_compute does not use it.
    h->uni_vmovups(vmm_aux(2), vmm_src);
    // we store the original sign and make x negative
    h->uni_vandps(vmm_aux(2), vmm_aux(2), table_val(sign_mask));
    h->uni_vorps(vmm_src, vmm_src, table_val(sign_mask));

    exp_compute_vector_fwd(vmm_src);
    // dup exp(x)
    h->uni_vmovups(vmm_aux(0), vmm_src);
    // (exp(x) + 1)
    h->uni_vaddps(vmm_aux(0), vmm_aux(0), table_val(one));
    // y = exp(x) / (exp(x) + 1)
    h->uni_vdivps(vmm_src, vmm_src, vmm_aux(0));

    // Now we have to apply the "symmetry" based on original sign
    h->uni_vmovups(vmm_aux(1), table_val(one));
    h->uni_vsubps(vmm_aux(1), vmm_aux(1), vmm_src);
    if (is_avx512_) {
        h->vptestmd(k_mask_, vmm_aux(2), vmm_aux(2));
    } else {
        h->uni_vmovups(vmm_mask_, vmm_aux(2));
    }
    blend_with_mask(vmm_aux(1), vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux(1));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::swish_compute_vector_fwd(
        const Vmm &vmm_src) {
    // Save src data for later usage
    h->uni_vmovups(vmm_aux(3), vmm_src);

    // x*alpha
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    // sigmoid(x*alpha)
    logistic_compute_vector_fwd(vmm_src);
    // x*sigmoid(alpha*x)
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(3));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::log_compute_vector_fwd(
        const Vmm &vmm_src) {
    // From J.-M. Muller and others, Handbook of Floating-Point Arithmetic, 2010
    // Here is a brief mathematics to approximate log(x):
    // log(x) = E * log(2) + log(y), where -log(2)/2 <= log(y) <= log(2)/2;
    // log(y) = log(1 + z) - log(r_i), where z = y * r_i - 1, r_i approximates
    //   1 / y, i is index of one of precomputed values;
    // log(1 + z) ~~ polynomial(z), =>
    // if (x is normal)
    //     log(x) ~~ E * log(2) + polynomial(z) - log(r_i),
    // where log(r_i) is table value.
    //
    // If (x == 0) result = -inf;
    // If (x < 0) result = qnan; (qnan value taken from table_val)
    // If (x == inf) result = inf;
    // If (x == qnan) result = qnan; (qnan value taken from src)
    // If (x == 1) result = 0;

    // set unused register as tmp for avx
    if (isa == avx) {
        ymm_tmp_ = Ymm(vmm_aux(0).getIdx());
        xmm_tmp_ = Xmm(vmm_aux(0).getIdx());
    }

    // save source on stack to check neg and zero values at the end
    h->uni_vmovups(vmm_aux(4), vmm_src);

    // compute i
    const int approx_order = 5;
    vec_shift(vmm_aux(1), vmm_src, false, n_mantissa_bits_ - approx_order);
    h->uni_vandps(vmm_aux(1), vmm_aux(1), table_val(log_five_bit_offset));
    vec_shift(vmm_aux(1), vmm_aux(1), true, 1); // multiply i by 2

    // compute anticancellation i
    vec_shift(vmm_aux(2), vmm_aux(1), false, approx_order);

    // get E, don't care about sign as only positive numbers are considered
    vec_shift(vmm_aux(3), vmm_src, false, n_mantissa_bits_);
    if (isa != avx)
        h->uni_vpaddd(vmm_aux(3), vmm_aux(3), vmm_aux(2));
    else {
        Ymm ymm_aux2 = Ymm(vmm_aux(2).getIdx());
        Ymm ymm_aux3 = Ymm(vmm_aux(3).getIdx());
        Xmm xmm_aux2 = Xmm(vmm_aux(2).getIdx());
        Xmm xmm_aux3 = Xmm(vmm_aux(3).getIdx());
        h->vextractf128(xmm_tmp_, ymm_aux3, 1);
        h->vpaddd(xmm_aux3, xmm_aux3, xmm_aux2);
        h->vperm2f128(ymm_aux2, ymm_aux2, ymm_aux2, 1);
        h->vpaddd(xmm_tmp_, xmm_tmp_, xmm_aux2);
        h->vperm2f128(ymm_aux2, ymm_aux2, ymm_aux2, 1);
        h->vinsertf128(ymm_aux3, ymm_aux3, xmm_tmp_, 1);
    }
    h->uni_vcvtdq2ps(vmm_aux(3), vmm_aux(3));

    // get m (mantissa)
    h->uni_vxorps(vmm_aux(2), vmm_aux(2), table_val(exponent_bias));
    vec_shift(vmm_aux(2), vmm_aux(2), true, n_mantissa_bits_);
    h->uni_vandps(vmm_src, vmm_src, table_val(log_mantissa_mask));
    h->uni_vorps(vmm_src, vmm_src, vmm_aux(2));

    // At first, adjust indices for table structure which broadcasts elements
    // by multiplying by simd_w
    const int simd_w = math::ilog2q(
            vlen_ / sizeof(float)); // equal to 2/3/4 for xmm/ymm/zmm
    vec_shift(vmm_aux(1), vmm_aux(1), true, simd_w);

    const auto it = entry_map_.find(log_predefined_vals);
    if (it == entry_map_.end()) {
        assert(it != entry_map_.end());
        return;
    }
    const auto table_start_idx = (*it).second.off;

    auto gather_table_values = [&](const Vmm &vmm_dst, const Vmm &vmm_idxs,
                                       size_t offt = 0) {
        Xbyak::Address table_idx = h->ptr[p_table_ + table_start_idx + offt
                + vmm_idxs * sizeof(float)];
        if (is_avx512_) {
            h->kmovw(k_mask_, table_val(log_full_k_reg_mask));
            h->vgatherdps(vmm_dst | k_mask_, table_idx);
        } else if (utils::one_of(isa, avx2, avx2_vnni_2)) {
            h->uni_vmovups(vmm_mask_, table_val(sign_mask));
            h->vgatherdps(vmm_dst, table_idx, vmm_mask_);
        } else if (isa == avx || isa == sse41) {
            Xbyak::Reg64 reg_tmp
                    = p_table_.getIdx() != h->r9.getIdx() ? h->r9 : h->r10;

            const int gpr_size = 8;
            // save reg_tmp state as we are not allowed to spoil it.
            h->sub(h->rsp, gpr_size);
            h->mov(h->ptr[h->rsp], reg_tmp);

            // rest of code puts indices on stack, fetching a table number based
            // on an index, replaces index with the value, and, finally, moves
            // fetched values into vector register.
            h->uni_vmovups(h->ptr[reg_vmm_stack_ptr_ + vlen_], vmm_idxs);

            for (size_t i = 0; i < vlen_ / sizeof(float); ++i) {
                h->mov(reg_tmp.cvt32(),
                        h->ptr[reg_vmm_stack_ptr_ + vlen_ + i * sizeof(float)]);
                h->shl(reg_tmp.cvt32(), 2); // multiply by simd_w
                table_idx = h->ptr[p_table_ + table_start_idx + offt + reg_tmp];
                h->mov(reg_tmp.cvt32(), table_idx);
                h->mov(h->ptr[reg_vmm_stack_ptr_ + vlen_ + i * sizeof(float)],
                        reg_tmp.cvt32());
            }

            h->uni_vmovups(vmm_dst, h->ptr[reg_vmm_stack_ptr_ + vlen_]);
            // restore GPR state
            h->mov(reg_tmp, h->ptr[h->rsp]);
            h->add(h->rsp, gpr_size);
        }
    };

    // get r_i, same as table(i)
    gather_table_values(vmm_aux(2), vmm_aux(1), 0);

    // compute relative error (rel_err = m * r_i - 1)
    h->uni_vfmsub213ps(vmm_aux(2), vmm_src, table_val(one));

    // compute polynomial(rel_err)
    h->uni_vmovups(vmm_src, table_val(log_pol, 3));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(2), table_val(log_pol, 2));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(2), table_val(log_pol, 1));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(2), table_val(log_pol, 0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(2), table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(2));

    // get log(r_i) = table(i+1)
    gather_table_values(vmm_aux(2), vmm_aux(1), vlen_);

    // compute partial result (pres = E * ln(2) - log(r_i))
    h->uni_vfmadd231ps(vmm_aux(2), vmm_aux(3), table_val(ln2f));

    // compute (result = polynomial + pres) w/ TwoSum algorithm
    // TODO: restore this instead of version below when asserts are gone
    // h->uni_vaddps(vmm_aux(1), vmm_src, vmm_aux(2)); // res_hi = pol + pres
    // h->uni_vsubps(vmm_aux(3), vmm_aux(1), vmm_aux(2)); // res_lo = res_hi - pres
    // h->uni_vsubps(vmm_aux(3), vmm_aux(3), vmm_src); // res_lo = res_lo - pol
    // h->uni_vaddps(vmm_src, vmm_aux(1), vmm_aux(3)); // res_hi = pol + pres

    h->uni_vmovups(vmm_aux(1), vmm_src);
    h->uni_vaddps(vmm_aux(1), vmm_aux(1), vmm_aux(2)); // res_hi = pol + pres
    h->uni_vmovups(vmm_aux(3), vmm_aux(1));
    h->uni_vsubps(vmm_aux(3), vmm_aux(3), vmm_aux(2)); // res_lo = res_hi - pres
    h->uni_vsubps(vmm_aux(3), vmm_aux(3), vmm_src); // res_lo = res_lo - pol
    h->uni_vmovups(vmm_src, vmm_aux(1));
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux(3)); // res_hi = pol + pres

    // Check original source for zero and neg values. skip blend w/ extreme
    // values if all src values were positive.
    h->uni_vmovups(vmm_aux(1), vmm_aux(4));

    Xbyak::Label end_log_zero_label;
    compute_cmp_mask(vmm_aux(1), table_val(zero), _cmp_le_os);
    test_mask();
    h->jz(end_log_zero_label);

    // Blend extreme values into src if reach here.
    // First zero for -inf values...
    compute_cmp_mask(vmm_aux(1), table_val(zero), _cmp_eq_oq);
    blend_with_mask(vmm_src, table_val(log_minus_inf));

    // ...then negative for qnan values.
    compute_cmp_mask(vmm_aux(1), table_val(zero), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(log_qnan));

    h->L(end_log_zero_label);

    // Leave inf values same as in src.
    compute_cmp_mask(vmm_aux(1), table_val(log_inf), _cmp_eq_oq);
    Xbyak::Label end_log_inf_label;
    test_mask();
    h->jz(end_log_inf_label);
    blend_with_mask(vmm_src, table_val(log_inf));
    h->L(end_log_inf_label);

    // Detect qnans if src != src and blend with qnans.
    compute_cmp_mask(vmm_aux(1), vmm_aux(1), _cmp_neq_uq);
    Xbyak::Label end_log_nan_label;
    test_mask();
    h->jz(end_log_nan_label);
    blend_with_mask(vmm_src, vmm_aux(1));
    h->L(end_log_nan_label);

    // Detect ones and blend with zeros.
    compute_cmp_mask(vmm_aux(1), table_val(one), _cmp_eq_oq);
    Xbyak::Label end_log_one_label;
    test_mask();
    h->jz(end_log_one_label);
    blend_with_mask(vmm_src, table_val(zero));
    h->L(end_log_one_label);
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::pow_compute_vector_fwd(
        const Vmm &vmm_src) {
    // dispatch between special cases.
    if (beta_ == -1) { // alpha / x
        h->uni_vmovups(vmm_aux(0), table_val(alpha));
        h->uni_vdivps(vmm_src, vmm_aux(0), vmm_src, vmm_aux(0));
    } else if (beta_ == 0) { // alpha
        h->uni_vmovups(vmm_src, table_val(alpha));
    } else if (beta_ == 0.5) { // alpha * sqrt(x)
        sqrt_compute_vector_fwd(vmm_src);
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else if (beta_ == 1) { // alpha * x
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else if (beta_ == 2) { // alpha * x^2
        square_compute_vector_fwd(vmm_src);
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else { // general path
        // caller obligation to save gprs as callee may use them
        size_t gpr_size = 8;
        Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->rax,
                h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

        h->sub(h->rsp, n_gprs_to_save * gpr_size);
        for (size_t i = 0; i < n_gprs_to_save; ++i)
            h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

        // caller obligation to save k-regs as callee may use them
        static constexpr int k_mask_size = 8;
        size_t n_k_regs_to_save = 8;
        if (is_avx512_) {
            h->sub(h->rsp, n_k_regs_to_save * k_mask_size);
            for (size_t i = 0; i < n_k_regs_to_save; ++i) {
                if (mayiuse(avx512_core))
                    h->kmovq(h->ptr[h->rsp + i * k_mask_size], Opmask(i));
                else
                    h->kmovw(h->ptr[h->rsp + i * k_mask_size], Opmask(i));
            }
        }

        // 1. Caller obligation to save vector registers as callee may use them.
        // 2. Additionally save space for vmm_src, to put the answer in-place on
        // this space and space for beta.
        // 3. There is an implicit assumption that the host code uses the same
        // `isa` as the injector. Once the assumption is wrong, `n_vregs_` and
        // `vlen_` should be replaced with `host_isa::vlen_` and
        // `host_isa::n_vregs_`.
        for (size_t i = 2; i < n_vregs_ + 2; ++i)
            h->uni_vmovups(h->ptr[reg_vmm_stack_ptr_ + i * vlen_], Vmm(i - 2));
        h->uni_vmovups(h->ptr[reg_vmm_stack_ptr_ + 0 * vlen_], vmm_src); // src
        h->uni_vmovups(vmm_src, table_val(beta));
        h->uni_vmovups(h->ptr[reg_vmm_stack_ptr_ + 1 * vlen_], vmm_src); // beta

        // save function address in gpr to pass in in call instruction
        h->mov(h->rbp, reinterpret_cast<uintptr_t>(powf));

        // The 64-bit Windows ABI requires the caller to allocate 32 bytes of
        // a so called "shadow space" for the callee.  It also requires that
        // the stack be 16 byte aligned before the call instruction is issued.
        // In order to allocate the shadow space and ensure the 16-byte alignment
        // of the stack we may actually need to allocate 40 bytes (32 bytes for
        // the "shadow space" + 8 bytes to align the stack) if the stack
        // pointer is not currently 16 byte aligned.

        // align stack on 16-byte as ABI requires
        h->mov(h->rbx, h->rsp);
        // Get alignment offset.
        h->and_(h->rbx, 0xf);
        h->add(h->rbx, 0x20);
        h->sub(h->rsp, h->rbx);

        // Take src, apply powf on it and replace value on a stack with dst.
        Xmm xmm0 = Xmm(0), xmm1 = Xmm(1);
        for (size_t i = 0; i < vlen_ / sizeof(float); ++i) {
            const Address &source
                    = h->ptr[reg_vmm_stack_ptr_ + i * sizeof(float)];
            h->uni_vmovss(xmm0, source);
            h->uni_vmovss(xmm1, h->ptr[reg_vmm_stack_ptr_ + vlen_]); // beta
            h->uni_vzeroupper(); // eliminate performance penalties on avx
            h->call(h->rbp);
            // eliminate performance penalties on sse isa
            if (isa == sse41) h->uni_vzeroupper();
            h->uni_vmovss(source, xmm0);
        }

        h->add(h->rsp, h->rbx);

        // restore vector registers
        for (size_t i = n_vregs_ + 1; i >= 2; --i)
            h->uni_vmovups(Vmm(i - 2), h->ptr[reg_vmm_stack_ptr_ + i * vlen_]);
        h->uni_vmovups(vmm_src, h->ptr[reg_vmm_stack_ptr_ + 0 * vlen_]);

        // restore k registers
        if (is_avx512_) {
            for (int i = n_k_regs_to_save - 1; i >= 0; --i) {
                if (mayiuse(avx512_core))
                    h->kmovq(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
                else
                    h->kmovw(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
            }
            h->add(h->rsp, n_k_regs_to_save * k_mask_size);
        }

        // restore gpr registers
        for (int i = n_gprs_to_save - 1; i >= 0; --i)
            h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
        h->add(h->rsp, n_gprs_to_save * gpr_size);

        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa,
        Wmm>::gelu_erf_minimax_approx_compute_vector_fwd(const Vmm &vmm_src) {
    using namespace Xbyak::util;

    // TODO: consider enabling for lower ISA
    if (!is_avx512_) return;

    // register mapping
    Vmm vmm_pol = vmm_aux(0);
    Vmm vmm_src_pos = vmm_aux(1);
    Vmm vmm_indices = vmm_aux(2);
    Vmm vmm_tmp = vmm_aux(3); // this is for immediate read after write

    auto coeffs_address = [&](int coeff_off, int off = 0) {
        // we actually have 25 polynomials but pad to avoid unaligned accesses/
        int gelu_erf_n_polynomials = 32;
        return table_val(
                gelu_erf_minimax_pol, coeff_off * gelu_erf_n_polynomials + off);
    };
    auto gather_coefficient = [&](Vmm vmm_coeff, int coeff_idx,
                                      Vmm vmm_pol_idx) {
        Zmm zmm_coeff(vmm_coeff.getIdx());
        Zmm zmm_pol_idx(vmm_pol_idx.getIdx());
        h->uni_vmovups(zmm_coeff, coeffs_address(coeff_idx, 0));
        h->vpermt2ps(zmm_coeff, zmm_pol_idx, coeffs_address(coeff_idx, 16));
    };

    // we use the erf function symmetry erf(-x) = -erf(x)
    // So we make x positive, we will reapply the sign after erf evaluation
    h->uni_vmovups(vmm_src_pos, vmm_src);
    h->uni_vandps(vmm_src_pos, vmm_src_pos, table_val(positive_mask));

    // we compute indices for table lookup.
    h->uni_vmovups(vmm_indices, vmm_src_pos);
    h->uni_vpaddd(vmm_indices, vmm_indices, table_val(gelu_erf_idx_bias));
    // An arithmetic shift is needed to properly map denormals to
    // their polynomial. we shift by 21 as we use 2 bits of mantissa
    // for indexing.
    h->vpsrad(vmm_indices, vmm_indices, 21);

    // we need to apply special rules
    h->uni_vpmaxsd(vmm_indices, vmm_indices, table_val(gelu_erf_one));
    h->uni_vpminsd(vmm_indices, vmm_indices, table_val(gelu_erf_twenty_four));
    // We have to check
    //     index = x_pos > rbound ? 23 : index;
    // for erf to return -1/1 when we should.
    h->uni_vmovups(vmm_tmp, table_val(gelu_erf_rbound));
    compute_cmp_mask(vmm_tmp, vmm_src_pos, _cmp_lt_os);
    blend_with_mask(vmm_indices, table_val(gelu_erf_twenty_three));

    // we can now evaluate the polynomial
    gather_coefficient(vmm_pol, 5, vmm_indices);
    for (int deg = 4; deg >= 0; --deg) {
        gather_coefficient(vmm_tmp, deg, vmm_indices);
        h->uni_vfmadd213ps(vmm_pol, vmm_src_pos, vmm_tmp);
    }

    // we set the sign of vmm_pol properly
    h->uni_vandps(vmm_tmp, vmm_src, table_val(sign_mask));
    h->uni_vxorps(vmm_pol, vmm_pol, vmm_tmp);

    // we compute the final output
    h->uni_vaddps(vmm_pol, vmm_pol, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, vmm_pol);
    h->uni_vmulps(vmm_src, vmm_src, table_val(half));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::gelu_erf_compute_vector_fwd(
        const Vmm &vmm_src) {
    if (is_avx512_) {
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

    // use vmm_aux(3) to store original src.
    h->uni_vmovups(vmm_aux(3), vmm_src);

    // x = s / sqrt(2)
    h->uni_vmulps(vmm_src, vmm_src,
            table_val(gelu_erf_Abramowitz_Stegun_one_over_sqrt_two));

    // abs(x)
    h->uni_vmovups(vmm_aux(4), vmm_src);
    abs_compute_vector_fwd(vmm_aux(4));

    // t = 1 / (p*x + 1)
    h->uni_vmovups(
            vmm_aux(2), table_val(gelu_erf_Abramowitz_Stegun_approx_const));
    h->uni_vfmadd213ps(vmm_aux(2), vmm_aux(4), table_val(one));
    h->uni_vmovups(vmm_aux(4), table_val(one));
    h->uni_vdivps(vmm_aux(4), vmm_aux(4), vmm_aux(2));

    // -exp(-x*x)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));
    exp_compute_vector_fwd(vmm_src); // pollutes vmm_aux(0), vmm_aux(1)
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));

    // get sign
    h->uni_vmovups(vmm_aux(0), vmm_aux(3));
    h->uni_vandps(vmm_aux(0), vmm_aux(0), table_val(sign_mask));

    // -exp(-x*x)*t
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(4));

    // compute polynomialial r
    h->uni_vmovups(vmm_aux(1), table_val(gelu_erf_Abramowitz_Stegun_pol, 4));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(4),
            table_val(gelu_erf_Abramowitz_Stegun_pol, 3));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(4),
            table_val(gelu_erf_Abramowitz_Stegun_pol, 2));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(4),
            table_val(gelu_erf_Abramowitz_Stegun_pol, 1));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(4),
            table_val(gelu_erf_Abramowitz_Stegun_pol, 0));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->uni_vfmadd213ps(vmm_src, vmm_aux(1), table_val(one));
    h->uni_vxorps(vmm_src, vmm_src, vmm_aux(0));

    // S = 0.5 * s
    h->uni_vmulps(vmm_aux(3), vmm_aux(3), table_val(half));
    // GELU = 0.5 * s * (1 + erf) = S + S * erf
    h->uni_vfmadd213ps(vmm_src, vmm_aux(3), vmm_aux(3));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    // invariant to whether `s` or `d` is passed.
    // get mask of `s` > 0
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    // fill with alpha, then blend with 1.f
    h->uni_vmovups(vmm_src, table_val(alpha));
    blend_with_mask(vmm_src, table_val(one));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::elu_compute_vector_bwd(
        const Vmm &vmm_src) {
    if (use_dst_) {
        // get mask of `d` > 0
        compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
        // R = `d` + alpha, then blend with 1.f
        h->uni_vaddps(vmm_src, vmm_src, table_val(alpha));
        blend_with_mask(vmm_src, table_val(one));
    } else {
        // Note: use vmm_aux(2) for copy as exp_compute doesn't use it.
        h->uni_vmovups(vmm_aux(2), vmm_src);
        // R = exp(s)
        exp_compute_vector_fwd(vmm_src);
        // R *= alpha
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
        // Get mask of src copy, not of exponent, because comparing with
        // exp(0)=1.f after exponentiation may lead to incorrect results due to
        // small values like `9.61717e-09f` are converted into 1.f by exp, which
        // get multiplied by `alpha` but they should not.
        compute_cmp_mask(vmm_aux(2), table_val(zero), _cmp_gt_os);
        blend_with_mask(vmm_src, table_val(one));
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 1 - d^2 = 1 - tanh^2(s)
    if (!use_dst_) tanh_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux(0), table_val(one));
    h->uni_vfnmadd231ps(vmm_aux(0), vmm_src, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux(0));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::gelu_tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux(0), vmm_src);

    // compute G1(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x^2)
    // compute G2(x) = sqrt_root_two_over_pi * x * (1 + 3 * fitting_const * x^2)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);

    // keep G2 in a separate register
    h->uni_vmovups(vmm_aux(2), table_val(gelu_tanh_fitting_const_times_three));
    h->uni_vfmadd213ps(vmm_aux(2), vmm_src, table_val(one));

    h->uni_vmovups(vmm_aux(1), table_val(gelu_tanh_fitting_const));
    h->uni_vfmadd213ps(vmm_src, vmm_aux(1), table_val(one));
    h->uni_vmulps(
            vmm_aux(0), vmm_aux(0), table_val(gelu_tanh_sqrt_two_over_pi));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(0));
    h->uni_vmulps(vmm_aux(2), vmm_aux(2), vmm_aux(0));

    // save G2 on stack as tanh uses all available registers
    h->uni_vmovups(vmm_aux(4), vmm_aux(2));

    // T = tanh(G1(x))
    tanh_compute_vector_fwd(vmm_src);

    h->uni_vmovups(vmm_aux(2), vmm_aux(4));

    // compute 0.5 * (1 + T) * (1 + G2 * (1 - T))
    if (isa == sse41 || isa == avx) {
        h->uni_vmovups(vmm_aux(3), table_val(one));
        h->uni_vsubps(vmm_aux(3), vmm_aux(3), vmm_src);
        h->uni_vmulps(vmm_aux(2), vmm_aux(2), vmm_aux(3));
        h->uni_vaddps(vmm_src, vmm_src, table_val(one));
        h->uni_vmulps(vmm_aux(2), vmm_aux(2), vmm_src);
        h->uni_vaddps(vmm_src, vmm_src, vmm_aux(2));
    } else {
        // 1) R = G2 * (1 - T) = G2 - G2 * T
        h->uni_vfnmadd231ps(vmm_aux(2), vmm_aux(2), vmm_src);
        // 2) Q = 1 + T
        h->uni_vaddps(vmm_src, vmm_src, table_val(one));
        // 3) res = Q * (1 + R) = Q + Q * R
        h->uni_vfmadd231ps(vmm_src, vmm_src, vmm_aux(2));
    }
    h->uni_vmulps(vmm_src, vmm_src, table_val(half));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::square_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 2 * s
    h->uni_vmulps(vmm_src, vmm_src, table_val(two));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::abs_compute_vector_bwd(
        const Vmm &vmm_src) {
    // replace positive values with 1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one));
    // replace negative values with -1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(minus_one));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::sqrt_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 0.5 / d = 0.5 / sqrt(s)
    if (!use_dst_) sqrt_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux(0), table_val(half));
    // h->uni_vdivps(vmm_src, vmm_aux(0), vmm_src); // bless sse41
    h->uni_vdivps(vmm_aux(0), vmm_aux(0), vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux(0));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::linear_compute_vector_bwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_src, table_val(alpha));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::soft_relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    logistic_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::mish_compute_vector_bwd(
        const Vmm &vmm_src) {
    // IMPORTANT: we use vmm_aux(2) to save src as exp does not use it.
    h->uni_vmovups(vmm_aux(2), vmm_src); // vmm_aux(2) = x

    h->uni_vminps(vmm_src, vmm_src, table_val(bwd_mish_max_x_for_equation_f));
    exp_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux(1), vmm_src); // vmm_aux(1) = e^x

    // e^3x + 4*e^2x
    h->uni_vmulps(vmm_src, vmm_src, vmm_src); // e^2x
    h->uni_vmovups(vmm_aux(0), vmm_src);
    h->uni_vmulps(vmm_aux(0), vmm_aux(0), table_val(two));
    h->uni_vmulps(vmm_aux(0), vmm_aux(0), table_val(two)); // 4*e^2x
    h->uni_vfmadd213ps(vmm_src, vmm_aux(1), vmm_aux(0));

    // e^3x + 4*e^2x + 4*e^x*(x+1.5)
    h->uni_vaddps(vmm_aux(2), vmm_aux(2), table_val(one)); // vmm_aux(2) = x + 1
    h->uni_vmovups(vmm_aux(0), vmm_aux(2));
    h->uni_vaddps(vmm_aux(0), vmm_aux(0), table_val(half));
    h->uni_vmulps(vmm_aux(0), vmm_aux(0), table_val(two));
    h->uni_vmulps(vmm_aux(0), vmm_aux(0), table_val(two));
    h->uni_vfmadd231ps(vmm_src, vmm_aux(0), vmm_aux(1));

    // omega = e^3x + 4*e^2x + 4*e^x*(x+1.5) + 4*(x+1)
    h->uni_vmulps(vmm_aux(2), vmm_aux(2), table_val(two));
    h->uni_vfmadd231ps(vmm_src, vmm_aux(2), table_val(two));

    // delta = (e^x+1)^2 + 1
    h->uni_vmovups(vmm_aux(0), vmm_aux(1));
    h->uni_vaddps(vmm_aux(0), vmm_aux(0), table_val(one));
    h->uni_vmulps(vmm_aux(0), vmm_aux(0), vmm_aux(0));
    h->uni_vaddps(vmm_aux(0), vmm_aux(0), table_val(one));
    h->uni_vmulps(vmm_aux(0), vmm_aux(0), vmm_aux(0));

    // e^x * omega / delta^2
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(1));
    h->uni_vdivps(vmm_src, vmm_src, vmm_aux(0));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::logistic_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = d * (1 - d) = d - d * d; d = logistic(s)
    if (!use_dst_) logistic_compute_vector_fwd(vmm_src);
    // h->uni_vfnmadd231ps(vmm_src, vmm_src, vmm_src); // bless sse41
    h->uni_vmovups(vmm_aux(0), table_val(one));
    h->uni_vsubps(vmm_aux(0), vmm_aux(0), vmm_src);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(0));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::exp_compute_vector_bwd(
        const Vmm &vmm_src) {
    if (!use_dst_) exp_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::swish_compute_vector_bwd(
        const Vmm &vmm_src) {
    // R = alpha * s
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    // Save R on stack for later usage
    h->uni_vmovups(vmm_aux(3), vmm_src);
    // Q = sigmoid(alpha * s)
    logistic_compute_vector_fwd(vmm_src);
    // compute Q * (1 + R * (1 - Q))
    if (utils::one_of(isa, sse41, avx)) {
        h->uni_vmovups(vmm_aux(1), table_val(one));
        h->uni_vsubps(vmm_aux(1), vmm_aux(1), vmm_src);
        h->uni_vmulps(vmm_aux(1), vmm_aux(1), vmm_aux(3));
        h->uni_vaddps(vmm_aux(1), vmm_aux(1), table_val(one));
        h->uni_vmulps(vmm_src, vmm_src, vmm_aux(1));
    } else {
        // T = R * (1 - Q) = R - R * Q
        h->uni_vfnmadd231ps(vmm_aux(3), vmm_aux(3), vmm_src);
        // Q * (1 + T) = Q + Q * T
        h->uni_vfmadd231ps(vmm_src, vmm_src, vmm_aux(3));
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::log_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 1 / s
    h->uni_vmovups(vmm_aux(0), table_val(one));
    // h->uni_vdivps(vmm_src, vmm_aux(0), vmm_src); // bless sse41
    h->uni_vdivps(vmm_aux(0), vmm_aux(0), vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux(0));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::clip_compute_vector_bwd(
        const Vmm &vmm_src) {
    // set result with 1.f
    h->uni_vmovups(vmm_aux(0), table_val(one));
    const auto cmp_flag
            = alg_ == alg_kind::eltwise_clip ? _cmp_gt_os : _cmp_ge_os;
    // get mask of values > beta (or >= beta) and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(beta), cmp_flag);
    blend_with_mask(vmm_aux(0), table_val(zero));
    // get mask of values <= alpha and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(alpha), _cmp_le_os);
    blend_with_mask(vmm_aux(0), table_val(zero));
    h->uni_vmovups(vmm_src, vmm_aux(0));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::pow_compute_vector_bwd(
        const Vmm &vmm_src) {
    // dispatch some special cases.
    if (beta_ == 0) { // zero
        h->uni_vmovups(vmm_src, table_val(zero));
    } else if (beta_ == 0.5) { // 0.5 * alpha / sqrt(s)
        sqrt_compute_vector_bwd(vmm_src);
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else if (beta_ == 1) { // alpha
        h->uni_vmovups(vmm_src, table_val(alpha));
    } else {
        // Save `s` for later usage
        h->uni_vmovups(vmm_aux(1), vmm_src);
        // R = alpha * pow(s, beta)
        pow_compute_vector_fwd(vmm_src);
        // Restore `s`
        h->uni_vmovups(vmm_aux(0), vmm_aux(1));
        // Save mask of zero elements to convert them into zeros at the end
        if (beta_ >= 1)
            compute_cmp_mask(vmm_aux(0), table_val(zero), _cmp_eq_oq);
        // res = alpha * beta * pow(s, beta - 1) = beta * R / s;
        h->uni_vdivps(vmm_src, vmm_src, vmm_aux(0));
        h->uni_vmulps(vmm_src, vmm_src, table_val(beta));

        // beta < 1 leads to NaN as `s` appears in denominator, but beta >= 1
        // should lead to zero, when `s` is zero.
        if (beta_ >= 1) blend_with_mask(vmm_src, table_val(zero));
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::gelu_erf_compute_vector_bwd(
        const Vmm &vmm_src) {
    // R = s / sqrt(2)
    h->uni_vmulps(vmm_src, vmm_src,
            table_val(gelu_erf_Abramowitz_Stegun_one_over_sqrt_two));

    // Save R on stack for later usage
    h->uni_vmovups(vmm_aux(5), vmm_src);

    // Q = exp(-R*R)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));
    exp_compute_vector_fwd(vmm_src);

    // T = R / sqrt(pi) * Q
    h->uni_vmovups(vmm_aux(2), vmm_aux(5));
    h->uni_vmulps(vmm_aux(2), vmm_aux(2),
            table_val(gelu_erf_Abramowitz_Stegun_one_over_sqrt_pi));
    h->uni_vmulps(vmm_aux(2), vmm_aux(2), vmm_src);

    // -Q
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));

    // get sign
    h->uni_vmovups(vmm_aux(0), vmm_aux(5));
    h->uni_vandps(vmm_aux(0), vmm_aux(0), table_val(sign_mask));

    // abs(x)
    h->uni_vmovups(vmm_aux(1), vmm_aux(5));
    abs_compute_vector_fwd(vmm_aux(1));

    // W = 1 / (p * s + 1)
    h->uni_vmovups(
            vmm_aux(3), table_val(gelu_erf_Abramowitz_Stegun_approx_const));
    h->uni_vmovups(vmm_aux(4), table_val(one));
    h->uni_vfmadd213ps(vmm_aux(3), vmm_aux(1), vmm_aux(4));
    h->uni_vdivps(vmm_aux(4), vmm_aux(4), vmm_aux(3));

    // Q * W
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux(4));

    // compute polynomial r
    h->uni_vmovups(vmm_aux(1), table_val(gelu_erf_Abramowitz_Stegun_pol, 4));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(4),
            table_val(gelu_erf_Abramowitz_Stegun_pol, 3));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(4),
            table_val(gelu_erf_Abramowitz_Stegun_pol, 2));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(4),
            table_val(gelu_erf_Abramowitz_Stegun_pol, 1));
    h->uni_vfmadd213ps(vmm_aux(1), vmm_aux(4),
            table_val(gelu_erf_Abramowitz_Stegun_pol, 0));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->uni_vfmadd213ps(vmm_src, vmm_aux(1), table_val(one));
    h->uni_vxorps(vmm_src, vmm_src, vmm_aux(0));

    // P = T + 0.5
    h->uni_vaddps(vmm_aux(2), vmm_aux(2), table_val(half));
    // res = P + 0.5 * erf
    h->uni_vfmadd231ps(vmm_aux(2), vmm_src, table_val(half));
    h->uni_vmovups(vmm_src, vmm_aux(2));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::hardswish_compute_vector_bwd(
        const Vmm &vmm_src) {
    // Get mask for 0 < alpha * x + beta < 1
    h->uni_vmovups(vmm_aux(0), vmm_src);
    h->uni_vmulps(vmm_aux(0), vmm_aux(0), table_val(alpha));
    h->uni_vaddps(vmm_aux(0), vmm_aux(0), table_val(beta));
    // Form a derivative value
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux(0));

    compute_cmp_mask(vmm_aux(0), table_val(zero), _cmp_le_os);
    blend_with_mask(vmm_src, table_val(zero));
    compute_cmp_mask(vmm_aux(0), table_val(one), _cmp_ge_os);
    blend_with_mask(vmm_src, table_val(one));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::hardsigmoid_compute_vector_bwd(
        const Vmm &vmm_src) {
    // Get mask for 0 < alpha * x + beta < 1
    // Zero rest values.
    h->uni_vmovups(vmm_aux(0), vmm_src);
    h->uni_vmulps(vmm_aux(0), vmm_aux(0), table_val(alpha));
    h->uni_vaddps(vmm_aux(0), vmm_aux(0), table_val(beta));

    h->uni_vmovups(vmm_src, table_val(one));
    compute_cmp_mask(vmm_aux(0), table_val(zero), _cmp_le_os);
    blend_with_mask(vmm_src, table_val(zero));
    compute_cmp_mask(vmm_aux(0), table_val(one), _cmp_ge_os);
    blend_with_mask(vmm_src, table_val(zero));
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::round_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vroundps(vmm_src, vmm_src, _op_mxcsr);
}

template <cpu_isa_t isa, typename Wmm>
size_t jit_uni_eltwise_injector<isa, Wmm>::aux_gprs_count(
        alg_kind_t alg, bool is_fwd, float alpha) {
    using namespace alg_kind;
    int ret = 0;
    switch (alg) {
        case eltwise_tanh_use_dst_for_bwd:
        case eltwise_tanh:
        case eltwise_gelu_tanh: ret = isa == sse41 || isa == avx ? 4 : 0; break;
        default: ret = 0;
    }
    return ret + need_vmm_stack_ptr(alg, is_fwd, alpha);
};

template <cpu_isa_t isa, typename Wmm>
bool jit_uni_eltwise_injector<isa, Wmm>::need_vmm_stack_ptr(
        alg_kind_t alg, bool is_fwd, float alpha) {
    return op_vecs_count(alg, is_fwd) + aux_vecs_count(alg, is_fwd, alpha);
}

template <cpu_isa_t isa, typename Wmm>
size_t jit_uni_eltwise_injector<isa, Wmm>::op_vecs_count(
        alg_kind_t alg, bool is_fwd) {
    using namespace alg_kind;
    int ret = 0;
    if (is_fwd) {
        switch (alg) {
            case eltwise_gelu_tanh: ret = 1; break;
            case eltwise_log: ret = 1 + utils::one_of(isa, sse41, avx); break;
            case eltwise_pow: ret = n_vregs_ + 2; break;
            default: ret = 0;
        }
    } else {
        switch (alg) {
            case eltwise_pow: ret = 1 + (n_vregs_ + 2 /*calls fwd*/); break;
            default: ret = 0;
        }
    }

    return ret;
}

template <cpu_isa_t isa, typename Wmm>
size_t jit_uni_eltwise_injector<isa, Wmm>::aux_vecs_count(
        alg_kind_t alg, bool is_fwd, float alpha) {
    // For avx we need a register to save the upper part of Ymm
    const bool extra_avx_vmm = isa == avx;
    size_t n_vmms = 0;

    using namespace alg_kind;
    if (is_fwd) {
        switch (alg) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: n_vmms = (alpha != 0.f); break;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: n_vmms = 3 + extra_avx_vmm; break;
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh: n_vmms = 4 + extra_avx_vmm; break;
            case eltwise_square: n_vmms = 0; break;
            case eltwise_abs: n_vmms = 0 + extra_avx_vmm; break;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: n_vmms = 0; break;
            case eltwise_linear: n_vmms = 1; break;
            case eltwise_soft_relu: n_vmms = 4 + extra_avx_vmm; break;
            case eltwise_mish: n_vmms = 3 + extra_avx_vmm; break;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic: n_vmms = 3 + extra_avx_vmm; break;
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp: n_vmms = 2 + extra_avx_vmm; break;
            case eltwise_gelu_tanh: n_vmms = 4 + extra_avx_vmm; break;
            case eltwise_swish: n_vmms = 4 + extra_avx_vmm; break;
            case eltwise_log: n_vmms = 5; break;
            case eltwise_clip:
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2: n_vmms = 0; break;
            case eltwise_pow: n_vmms = 1; break;
            case eltwise_gelu_erf:
                n_vmms = 5 + extra_avx_vmm;
                break; // 4 on avx512+
            case eltwise_round: n_vmms = 0; break;
            case eltwise_hardswish: n_vmms = 1; break;
            case eltwise_hardsigmoid: n_vmms = 0; break;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: n_vmms = 0; break;
            case eltwise_elu_use_dst_for_bwd: n_vmms = 0 + extra_avx_vmm; break;
            case eltwise_elu: n_vmms = 3 + extra_avx_vmm; break;
            case eltwise_tanh_use_dst_for_bwd:
                n_vmms = 1 + extra_avx_vmm;
                break;
            case eltwise_tanh: n_vmms = 4 + extra_avx_vmm; break;
            case eltwise_square: n_vmms = 0; break;
            case eltwise_abs: n_vmms = 0 + extra_avx_vmm; break;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: n_vmms = 1; break;
            case eltwise_linear: n_vmms = 0; break;
            case eltwise_soft_relu: n_vmms = 3 + extra_avx_vmm; break;
            case eltwise_mish: n_vmms = 3 + extra_avx_vmm; break;
            case eltwise_logistic_use_dst_for_bwd:
                n_vmms = 1 + extra_avx_vmm;
                break;
            case eltwise_logistic: n_vmms = 3 + extra_avx_vmm; break;
            case eltwise_exp_use_dst_for_bwd: n_vmms = 0 + extra_avx_vmm; break;
            case eltwise_exp: n_vmms = 2 + extra_avx_vmm; break;
            case eltwise_gelu_tanh: n_vmms = 5 + extra_avx_vmm; break;
            case eltwise_swish: n_vmms = 4 + extra_avx_vmm; break;
            case eltwise_log: n_vmms = 1; break;
            case eltwise_clip:
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2: n_vmms = 1; break;
            case eltwise_pow: n_vmms = 2; break;
            case eltwise_gelu_erf: n_vmms = 6 + extra_avx_vmm; break;
            case eltwise_hardswish: n_vmms = 1; break;
            case eltwise_hardsigmoid: n_vmms = 1; break;
            default: assert(!"unsupported eltwise algorithm");
        }
    }

    return n_vmms + need_mask_register(alg, is_fwd, alpha);
}

template <cpu_isa_t isa, typename Wmm>
bool jit_uni_eltwise_injector<isa, Wmm>::need_mask_register(
        alg_kind_t alg, bool is_fwd, float alpha) {
    if (is_superset(isa, avx512_core)) return false;

    using namespace alg_kind;
    if (is_fwd) {
        switch (alg) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return alpha != 0.f;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return true;
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh: return true;
            case eltwise_square: return false;
            case eltwise_abs: return false;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return false;
            case eltwise_linear: return false;
            case eltwise_soft_relu: return true;
            case eltwise_mish: return true;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic: return true;
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp: return true;
            case eltwise_gelu_tanh: return true;
            case eltwise_swish: return true;
            case eltwise_log: return true;
            case eltwise_clip:
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2: return false;
            case eltwise_pow: return false;
            case eltwise_gelu_erf: return true;
            case eltwise_round: return false;
            case eltwise_hardswish: return false;
            case eltwise_hardsigmoid: return false;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return true;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return true;
            case eltwise_tanh_use_dst_for_bwd: return false;
            case eltwise_tanh: return true;
            case eltwise_square: return false;
            case eltwise_abs: return true;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return false;
            case eltwise_linear: return false;
            case eltwise_soft_relu: return true;
            case eltwise_mish: return true;
            case eltwise_logistic_use_dst_for_bwd: return false;
            case eltwise_logistic: return true;
            case eltwise_exp_use_dst_for_bwd: return false;
            case eltwise_exp: return true;
            case eltwise_gelu_tanh: return true;
            case eltwise_swish: return true;
            case eltwise_log: return false;
            case eltwise_clip:
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2: return true;
            case eltwise_pow: return true;
            case eltwise_gelu_erf: return true;
            case eltwise_hardswish: return true;
            case eltwise_hardsigmoid: return true;
            default: assert(!"unsupported eltwise algorithm");
        }
    }

    return false;
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::compute_body(
        const injector_utils::vmm_index_set_iterator_t &start_idx_it,
        const injector_utils::vmm_index_set_iterator_t &end_idx_it) {
    using namespace alg_kind;
    std::for_each(start_idx_it, end_idx_it, [&](size_t idx) {
        if (is_fwd_) {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu:
                    if (alpha_ == 0.f)
                        relu_zero_ns_compute_vector_fwd(Vmm(idx));
                    else
                        relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_square: square_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_abs: abs_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_swish: swish_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_linear: linear_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_mish: mish_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_log: log_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_clip:
                case eltwise_clip_v2_use_dst_for_bwd:
                case eltwise_clip_v2: clip_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_pow: pow_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_round: round_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_hardswish:
                    hardswish_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_hardsigmoid:
                    hardsigmoid_compute_vector_fwd(Vmm(idx));
                    break;
                default: assert(!"unsupported eltwise algorithm");
            }
        } else {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu: relu_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_square: square_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_abs: abs_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_linear: linear_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_mish: mish_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_swish: swish_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_log: log_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_clip:
                case eltwise_clip_v2_use_dst_for_bwd:
                case eltwise_clip_v2: clip_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_pow: pow_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_hardswish:
                    hardswish_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_hardsigmoid:
                    hardsigmoid_compute_vector_bwd(Vmm(idx));
                    break;
                default: assert(!"unsupported eltwise algorithm");
            }
        }
        if (scale_ != 1.f) {
            h->uni_vmulps(Vmm(idx), Vmm(idx), table_val(scale));
        }
    });
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::compute_vector_range(
        size_t start_compute_idx, size_t end_compute_idx,
        const injector_utils::vmm_index_set_t &vmm_aux_indices) {
    injector_utils::vmm_index_set_t vmm_compute_idxs;
    for (size_t i = start_compute_idx; i < end_compute_idx; i++)
        vmm_compute_idxs.emplace(i);
    compute_vector_range(vmm_compute_idxs, vmm_aux_indices);
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_compute_idxs,
        const injector_utils::vmm_index_set_t &vmm_aux_indices) {
    if (vmm_compute_idxs.empty()) return;

    const auto &start_idx_it = vmm_compute_idxs.begin();
    const auto &end_idx_it = vmm_compute_idxs.end();
    assert(*start_idx_it < *vmm_compute_idxs.rbegin() + 1
            && *vmm_compute_idxs.rbegin() <= n_vregs_);

    // This is something that can be moved in preamble.
    auto start_idx_tail_it = vmm_compute_idxs.begin();
    injector_preamble(vmm_compute_idxs, start_idx_tail_it, vmm_aux_indices);
    compute_body(start_idx_tail_it, end_idx_it);

    size_t n_vregs_not_preserved
            = std::distance(start_idx_it, start_idx_tail_it);
    injector_preamble_tail(n_vregs_not_preserved);
    compute_body(start_idx_it, start_idx_tail_it);
    injector_postamble();
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::prepare_table(bool gen_table) {
    if (!gen_table) return;

    h->align(64);
    h->L(l_table_);

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
        const auto len = te.bcast ? vlen_ : sizeof(table_entry_val_t);
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

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector<isa, Wmm>::register_table_entries() {
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
            // p0 = 1.0f
            {exp_pol, {0x3f7ffffb, true}}, // p1 = 0.999999701f
            {exp_pol, {0x3efffee3, true}}, // p2 = 0.499991506f
            {exp_pol, {0x3e2aad40, true}}, // p3 = 0.166676521f
            {exp_pol, {0x3d2b9d0d, true}}, // p4 = 0.0418978221f
            {exp_pol, {0x3c07cfce, true}} // p5 = 0.00828929059f
    };

    // mish(x) constants
    static const table_t mish_consts {
            {fwd_mish_max_x_for_equation_f, {0x42317217, true}},
            {bwd_mish_max_x_for_equation_f, {0x41b17217, true}}};

    // tanh(x) constants for four interval approximation
    static const table_t tanh_consts {{tanh_idx_bias, {0x39800000, true}},
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

    // gelu_erf(x) constants for approximation based on Abramowitz and Stegun
    // algorithm (formula defined)
    static const table_t gelu_erf_Abramowitz_Stegun_consts {
            {gelu_erf_Abramowitz_Stegun_approx_const, {0x3ea7ba05, true}},
            {gelu_erf_Abramowitz_Stegun_one_over_sqrt_two, {0x3f3504f3, true}},
            {gelu_erf_Abramowitz_Stegun_one_over_sqrt_pi, {0x3f106eba, true}},
    };

    // gelu_erf(x) polynomial approximation based on Abramowitz and Stegun
    // algorithm
    static const table_t gelu_erf_Abramowitz_Stegun_polynomial {
            // p1 = 0.254829592f
            {gelu_erf_Abramowitz_Stegun_pol, {0x3e827906, true}},
            // p2 = -0.284496736f
            {gelu_erf_Abramowitz_Stegun_pol, {0xbe91a98e, true}},
            // p3 = 1.421413741f
            {gelu_erf_Abramowitz_Stegun_pol, {0x3fb5f0e3, true}},
            // p4 = -1.453152027f
            {gelu_erf_Abramowitz_Stegun_pol, {0xbfba00e3, true}},
            // p5 = 1.061405429f
            {gelu_erf_Abramowitz_Stegun_pol, {0x3f87dc22, true}},
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

    // log(x) constants
    static const table_t log_consts {
            {log_inf, {0x7f800000, true}},
            {log_minus_inf, {0xff800000, true}},
            {log_qnan, {0x7fc00000, true}},
            {log_mantissa_mask, {0x007fffff, true}},
            {log_full_k_reg_mask, {0x0000ffff, true}},
            {log_five_bit_offset, {0x0000001f, true}},
    };

    // log(x) polynomial approximation
    static const table_t log_polynomial {
            {log_pol, {0xbf000000, true}}, // p1 = -0.5f
            {log_pol, {0x3eaaaaab, true}}, // p2 =  0.333333343f
            {log_pol, {0xbe8004ab, true}}, // p3 = -0.250035613f
            {log_pol, {0x3e4cc8a3, true}}, // p4 =  0.199984118f
    };

    // log(x) pre-defined values. First goes index}, then val[index].
    static const table_t log_predefined_values {
            {log_predefined_vals, {0x3f800000, true}}, //  0: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, //  1: -88.029693603515625
            {log_predefined_vals, {0x3f780000, true}}, //  2: 0.96875
            {log_predefined_vals,
                    {0xc2affef2, true}}, //  3: -87.9979400634765625
            {log_predefined_vals, {0x3f700000, true}}, //  4: 0.9375
            {log_predefined_vals,
                    {0xc2afee29, true}}, //  5: -87.9651565551757812
            {log_predefined_vals, {0x3f680000, true}}, //  6: 0.90625
            {log_predefined_vals,
                    {0xc2afdccd, true}}, //  7: -87.9312515258789062
            {log_predefined_vals, {0x3f600000, true}}, //  8: 0.875
            {log_predefined_vals,
                    {0xc2afcad6, true}}, //  9: -87.8961639404296875
            {log_predefined_vals, {0x3f580000, true}}, // 10: 0.84375
            {log_predefined_vals,
                    {0xc2afb837, true}}, // 11: -87.859794616699218
            {log_predefined_vals, {0x3f580000, true}}, // 12: 0.84375
            {log_predefined_vals,
                    {0xc2afb837, true}}, // 13: -87.859794616699218
            {log_predefined_vals, {0x3f500000, true}}, // 14: 0.8125
            {log_predefined_vals,
                    {0xc2afa4e4, true}}, // 15: -87.822052001953125
            {log_predefined_vals, {0x3f480000, true}}, // 16: 0.78125
            {log_predefined_vals,
                    {0xc2af90cf, true}}, // 17: -87.782829284667968
            {log_predefined_vals, {0x3f480000, true}}, // 18: 0.78125
            {log_predefined_vals,
                    {0xc2af90cf, true}}, // 19: -87.782829284667968
            {log_predefined_vals, {0x3f400000, true}}, // 20: 0.75
            {log_predefined_vals,
                    {0xc2af7be9, true}}, // 21: -87.742012023925781
            {log_predefined_vals, {0x3f400000, true}}, // 22: 0.75
            {log_predefined_vals,
                    {0xc2af7be9, true}}, // 23: -87.742012023925781
            {log_predefined_vals, {0x3f380000, true}}, // 24: 0.71875
            {log_predefined_vals,
                    {0xc2af661e, true}}, // 25: -87.699447631835937
            {log_predefined_vals, {0x3f380000, true}}, // 26: 0.71875
            {log_predefined_vals,
                    {0xc2af661e, true}}, // 27: -87.699447631835937
            {log_predefined_vals, {0x3f300000, true}}, // 28: 0.6875
            {log_predefined_vals,
                    {0xc2af4f5c, true}}, // 29: -87.654998779296875
            {log_predefined_vals, {0x3f300000, true}}, // 30: 0.6875
            {log_predefined_vals,
                    {0xc2af4f5c, true}}, // 31: -87.654998779296875
            {log_predefined_vals, {0x3fa80000, true}}, // 32: 1.3125
            {log_predefined_vals,
                    {0xc2b09a6f, true}}, // 33: -88.301628112792968
            {log_predefined_vals, {0x3fa80000, true}}, // 34: 1.3125
            {log_predefined_vals,
                    {0xc2b09a6f, true}}, // 35: -88.301628112792968
            {log_predefined_vals, {0x3fa00000, true}}, // 36: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 37: -88.252838134765625
            {log_predefined_vals, {0x3fa00000, true}}, // 38: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 39: -88.252838134765625
            {log_predefined_vals, {0x3fa00000, true}}, // 40: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 41: -88.252838134765625
            {log_predefined_vals, {0x3f980000, true}}, // 42: 1.1875
            {log_predefined_vals,
                    {0xc2b06731, true}}, // 43: -88.201545715332031
            {log_predefined_vals, {0x3f980000, true}}, // 44: 1.1875
            {log_predefined_vals,
                    {0xc2b06731, true}}, // 45: -88.201545715332031
            {log_predefined_vals, {0x3f900000, true}}, // 46: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 47: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 48: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 49: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 50: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 51: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 52: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 53: -88.147476196289062
            {log_predefined_vals, {0x3f880000, true}}, // 54: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 55: -88.090316772460937
            {log_predefined_vals, {0x3f880000, true}}, // 56: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 57: -88.090316772460937
            {log_predefined_vals, {0x3f880000, true}}, // 58: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 59: -88.090316772460937
            {log_predefined_vals, {0x3f800000, true}}, // 60: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, // 61: -88.029693603515625
            {log_predefined_vals, {0x3f800000, true}}, // 62: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, // 63: -88.029693603515625
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
                case eltwise_gelu_tanh: gelu_tanh_ = true; break;
                case eltwise_log: log_ = true; break;
                case eltwise_soft_relu: soft_relu_ = true; break;
                case eltwise_mish: mish_ = true; break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_ = true; break;
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
    if (need.mish()) push_entries_of(mish_consts);
    if (need.tanh()) push_entries_of(tanh_consts);
    if (need.tanh()) push_entries_of(tanh_polynomial_table);
    if (need.soft_relu()) push_entries_of(soft_relu_consts);
    if (need.soft_relu()) push_entries_of(soft_relu_polynomial);
    if (need.gelu_tanh()) push_entries_of(gelu_tanh_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_Abramowitz_Stegun_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_Abramowitz_Stegun_polynomial);
    if (need.gelu_erf() && is_avx512_) push_entries_of(gelu_erf_minimax_consts);
    if (need.gelu_erf() && is_avx512_)
        push_entries_of(gelu_erf_minimax_polynomial);

    if (need.log()) push_entries_of(log_consts);
    if (need.log()) push_entries_of(log_polynomial);
    if (need.log()) push_entries_of(log_predefined_values);

    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        auto &te = (*it).second;
        te.off = off;
        off += te.bcast ? vlen_ : sizeof(table_entry_val_t);
    }
}

template struct jit_uni_eltwise_injector<avx512_core_fp16>;
template struct jit_uni_eltwise_injector<avx512_core_fp16, Xbyak::Ymm>;
template struct jit_uni_eltwise_injector<avx512_core_fp16, Xbyak::Xmm>;
template struct jit_uni_eltwise_injector<avx512_core_bf16>;
template struct jit_uni_eltwise_injector<avx512_core>;
template struct jit_uni_eltwise_injector<avx512_core, Ymm>;
template struct jit_uni_eltwise_injector<avx512_core, Xmm>;
template struct jit_uni_eltwise_injector<avx2_vnni_2>;
template struct jit_uni_eltwise_injector<avx2_vnni_2, Xmm>;
template struct jit_uni_eltwise_injector<avx2>;
template struct jit_uni_eltwise_injector<avx2, Xmm>;
template struct jit_uni_eltwise_injector<avx>;
template struct jit_uni_eltwise_injector<avx, Xmm>;
template struct jit_uni_eltwise_injector<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
