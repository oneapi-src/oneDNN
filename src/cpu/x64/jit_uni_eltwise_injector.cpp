/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "cpu/x64/jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble(
        size_t start_idx, size_t end_idx) {
    preserved_vecs_count = 0;
    vecs_to_preserve = aux_vecs_count();
    start_idx_tail = start_idx;

    // For sse41 mask register has to be Xmm(0)
    if (isa == sse41 && vecs_to_preserve > 0) {
        size_t idx = 0;
        assert(idx < start_idx);
        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    for (size_t idx = preserved_vecs_count; idx < vecs_count; idx++) {
        if (preserved_vecs_count >= vecs_to_preserve) break;
        if (start_idx <= idx && idx < end_idx) continue;

        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    size_t preserved_vecs_count_tail = vecs_to_preserve - preserved_vecs_count;
    for (size_t i = 0; i < preserved_vecs_count_tail; i++) {
        preserved_vec_idxs[preserved_vecs_count++] = start_idx_tail++;
    }

    assert(preserved_vecs_count == vecs_to_preserve);

    if (save_state_) {
        h->push(p_table);

        if (preserved_vecs_count) h->sub(h->rsp, preserved_vecs_count * vlen);

        for (size_t i = 0; i < preserved_vecs_count; ++i)
            h->uni_vmovups(
                    h->ptr[h->rsp + i * vlen], Vmm(preserved_vec_idxs[i]));

        load_table_addr();
    }

    assign_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble_tail(
        size_t start_idx) {
    size_t tail_vecs_to_preserve = start_idx_tail - start_idx;
    if (tail_vecs_to_preserve == 0) return;

    const int idx_off = vecs_to_preserve - tail_vecs_to_preserve;

    if (save_state_) {
        if (idx_off) h->add(h->rsp, idx_off * vlen);

        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(Vmm(preserved_vec_idxs[idx_off + i]),
                    h->ptr[h->rsp + i * vlen]);
    }

    for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
        preserved_vec_idxs[idx_off + i] += tail_vecs_to_preserve;

    if (save_state_) {
        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen],
                    Vmm(preserved_vec_idxs[idx_off + i]));

        if (idx_off) h->sub(h->rsp, idx_off * vlen);
    }

    assign_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_postamble() {
    if (!save_state_) return;

    for (size_t i = 0; i < preserved_vecs_count; ++i)
        h->uni_vmovups(Vmm(preserved_vec_idxs[i]), h->ptr[h->rsp + i * vlen]);

    if (preserved_vecs_count) h->add(h->rsp, preserved_vecs_count * vlen);

    h->pop(p_table);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::assign_regs() {
    vmm_mask = Vmm(preserved_vec_idxs[0]);
    vmm_aux0 = Vmm(preserved_vec_idxs[0]);
    vmm_aux1 = Vmm(preserved_vec_idxs[1]);
    vmm_aux2 = Vmm(preserved_vec_idxs[2]);
    vmm_aux3 = Vmm(preserved_vec_idxs[3]);
    vmm_aux4 = Vmm(preserved_vec_idxs[4]);
}

// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Stores a mask by applying cmpps on two inputs w/ a given predicate.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_cmp_mask(const Vmm &vmm_src,
        const Xbyak::Operand &compare_operand, int cmp_predicate) {
    if (has_avx512()) {
        h->vcmpps(k_mask, vmm_src, compare_operand, cmp_predicate);
    } else {
        h->uni_vcmpps(vmm_mask, vmm_src, compare_operand, cmp_predicate);
    }
}

// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Blends a result of second input into a first input w/ a stored mask.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::blend_with_mask(
        const Vmm &vmm_dst, const Xbyak::Operand &src) {
    if (has_avx512()) {
        h->vblendmps(vmm_dst | k_mask, vmm_dst, src);
    } else {
        h->uni_vblendvps(vmm_dst, vmm_dst, src, vmm_mask);
    }
}

// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Tests a mask for all zeros. If all zeroes occur, set ZF = 1.
// Nicely combines with jump_if_zero (jz).
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::test_mask() {
    if (has_avx512()) {
        h->kortestw(k_mask, k_mask);
    } else {
        h->uni_vtestps(vmm_mask, vmm_mask);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_fwd(
        const Vmm &vmm_src) {
    // get mask of values lower than log(FLT_MIN) to zero them in the output
    compute_cmp_mask(vmm_src, table_val(exp_ln_flt_min_f), _cmp_lt_os);

    h->uni_vminps(vmm_src, vmm_src, table_val(exp_ln_flt_max_f));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(exp_ln_flt_min_f));
    h->uni_vmovups(vmm_aux1, vmm_src);
    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(exp_log2ef));
    h->uni_vaddps(vmm_src, vmm_src, table_val(half));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux2, vmm_src, _op_floor);

    // keep vmm_src = fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux2);

    // x = x - fx * ln2
    h->uni_vfnmadd231ps(vmm_aux1, vmm_aux2, table_val(ln2f));

    // compute 2^n
    h->uni_vcvtps2dq(vmm_aux2, vmm_src);
    h->uni_vpaddd(vmm_aux2, vmm_aux2, table_val(exponent_bias));
    h->uni_vpslld(vmm_aux2, vmm_aux2, n_mantissa_bits); //Vmm(6) = 2^-fx

    // use vmm_src as tmp vmm_zero when applying mask
    h->uni_vpxor(vmm_src, vmm_src, vmm_src);
    // set zeroes at those points which were < log(FLT_MIN)
    blend_with_mask(vmm_aux2, vmm_src);

    // compute polynomial
    h->uni_vmovups(vmm_src, table_val(exp_pol, 4));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 3));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 2));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 1));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    // y = y * 2^n
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux1, vmm_src);
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    blend_with_mask(vmm_src, vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_zero_ns_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);
    // compute exponent
    exp_compute_vector_fwd(vmm_src);

    // alpha * (exp(x) - 1)
    h->uni_vsubps(vmm_src, vmm_src, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));

    // combine with mask
    compute_cmp_mask(vmm_aux3, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    // register mapping
    // vmm_src contains input
    // vmm_mask contains mask of currently valid results.
    //     1 is need computation, 0 is already computed
    // vmm_aux1 contains current output
    // vmm_aux2, vmm_aux3 contains auxiliary values
    // vmm_aux4 contains the original sign of inputs

    Label end_tanh_label;

    auto test_exit = [&](const Xbyak::Address &threshold) {
        compute_cmp_mask(vmm_src, threshold, _cmp_ge_os);
        test_mask();
        h->jz(end_tanh_label, Xbyak::CodeGenerator::T_NEAR);
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x postive
    // and reapply sign at the end
    // mov is not necessary for >AVX, but should not matter for performance
    h->uni_vmovups(vmm_aux4, vmm_src);
    h->uni_vandps(vmm_aux4, vmm_aux4, table_val(sign_mask));
    h->uni_vandps(vmm_src, vmm_src, table_val(positive_mask));

    // if x < linear_sat_point for all inputs, we just return the input
    h->uni_vmovups(vmm_aux1, vmm_src);
    test_exit(table_val(tanh_bound_x));

    // if one of the mask is one, we have to compute an better approx
    h->uni_vmovups(vmm_aux2, vmm_src);
    h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_aux2);
    h->uni_vmovups(vmm_aux3, table_val(tanh_pol, 4));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(tanh_pol, 3));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(tanh_pol, 2));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(tanh_pol, 1));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(tanh_pol, 0));
    h->uni_vmulps(vmm_aux3, vmm_aux3, vmm_src);

    // we blend only the result that need update
    blend_with_mask(vmm_aux1, vmm_aux3);

    // if x < exp_bound_point, we go to return point
    test_exit(table_val(tanh_bound_pol));

    // if not we use a better approx 1 - 2 / (1 + exp(2x))
    // compute 2x
    h->uni_vmovups(vmm_aux3, vmm_src);
    h->uni_vaddps(vmm_aux3, vmm_aux3, vmm_aux3);

    // Compute exp(2x)
    // We need to save kmask, vmm_mask, vmm_aux1, vmm_aux2 and vmm_src as exp
    // uses them.
    // vmm_src is not more read afterwards, so we do not have to save it
    auto stack_size = 4 * vlen + has_avx512() * k_mask_size;
    h->sub(h->rsp, stack_size);
    h->uni_vmovups(h->ptr[h->rsp + 0 * vlen], vmm_mask);
    h->uni_vmovups(h->ptr[h->rsp + 1 * vlen], vmm_aux1);
    h->uni_vmovups(h->ptr[h->rsp + 2 * vlen], vmm_aux2);
    h->uni_vmovups(h->ptr[h->rsp + 3 * vlen], vmm_src);
    if (has_avx512()) h->kmovw(h->ptr[h->rsp + 4 * vlen], k_mask);

    exp_compute_vector_fwd(vmm_aux3);

    h->uni_vmovups(vmm_mask, h->ptr[h->rsp + 0 * vlen]);
    h->uni_vmovups(vmm_aux1, h->ptr[h->rsp + 1 * vlen]);
    h->uni_vmovups(vmm_aux2, h->ptr[h->rsp + 2 * vlen]);
    h->uni_vmovups(vmm_src, h->ptr[h->rsp + 3 * vlen]);
    if (has_avx512()) h->kmovw(k_mask, h->ptr[h->rsp + 4 * vlen]);
    h->add(h->rsp, stack_size);

    // 1 + exp(2x)
    h->uni_vaddps(vmm_aux3, vmm_aux3, table_val(one));

    // 1 - 2 / (1 + exp(2x))
    h->uni_vmovups(vmm_aux2, table_val(minus_two));
    h->uni_vdivps(vmm_aux2, vmm_aux2, vmm_aux3);
    h->uni_vaddps(vmm_aux2, vmm_aux2, table_val(one));

    // we blend only the result that need update
    blend_with_mask(vmm_aux1, vmm_aux2);

    // finally, we saturate to 1 if needed
    // TODO: maybe move that up if most inputs saturate in practice
    compute_cmp_mask(vmm_src, table_val(tanh_bound_one), _cmp_ge_os);
    h->uni_vmovups(vmm_aux2, table_val(one));
    blend_with_mask(vmm_aux1, vmm_aux2);

    h->L(end_tanh_label);
    {
        // we apply the sign of x to the result and we are done
        h->uni_vmovups(vmm_src, vmm_aux1);
        h->uni_vpxor(vmm_src, vmm_src, vmm_aux4);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux0, vmm_src);

    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vmovups(vmm_aux1, table_val(gelu_tanh_fitting_const));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
    h->uni_vmulps(vmm_src, vmm_src, table_val(gelu_tanh_sqrt_two_over_pi));

    // save x on stack as tanh uses vmm_aux0
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_aux0);

    // compute tanh(G(x))
    tanh_compute_vector_fwd(vmm_src);

    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    // compute 0.5 * x * (1 + tanh(G(x)))
    h->uni_vaddps(vmm_src, vmm_src, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, table_val(half));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_fwd(
        const Vmm &vmm_src) {
    // compute abs(x) = _mm_and_ps(x, 01111..111));
    h->uni_vandps(vmm_src, vmm_src, table_val(positive_mask));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vsqrtps(vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_fwd(
        const Vmm &vmm_src) {
    // compute x = alpha * x + beta;
    h->uni_vmovups(vmm_aux0, table_val(alpha));
    h->uni_vfmadd213ps(vmm_src, vmm_aux0, table_val(beta));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero));
    h->uni_vminps(vmm_src, vmm_src, table_val(alpha));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmaxps(vmm_src, vmm_src, table_val(alpha));
    h->uni_vminps(vmm_src, vmm_src, table_val(beta));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // keep src for further computations
    h->uni_vmovups(vmm_aux2, vmm_src);

    h->uni_vminps(vmm_src, vmm_src, table_val(exp_ln_flt_max_f));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(exp_ln_flt_min_f));
    h->uni_vmovups(vmm_aux1, vmm_src);
    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(exp_log2ef));
    h->uni_vaddps(vmm_src, vmm_src, table_val(half));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux0, vmm_src, _op_floor);

    // keep vmm_src = fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux0);

    // x = x - fx * ln2
    h->uni_vmulps(vmm_aux0, vmm_aux0, table_val(ln2f));
    h->uni_vsubps(vmm_aux1, vmm_aux1, vmm_aux0);
    // compute exponent polynomial
    h->uni_vmovups(vmm_aux3, table_val(exp_pol, 4));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 3));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 2));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 1));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 0));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(one));

    // compute 2^(-n)
    if (has_avx512()) {
        h->vmulps(vmm_aux1, vmm_src, table_val(minus_one));
        h->vcvtps2dq(vmm_aux1, vmm_aux1);
    } else {
        h->uni_vcvtps2dq(vmm_aux1, vmm_src);
        h->uni_vpsignd(vmm_aux1, vmm_aux1, table_val(minus_one));
    }

    h->uni_vpaddd(vmm_aux1, vmm_aux1, table_val(exponent_bias));
    h->uni_vpslld(vmm_aux1, vmm_aux1, n_mantissa_bits); //vmm_aux1 = 2^-fx
    // calculate ln(1 + y)
    h->uni_vaddps(vmm_aux3, vmm_aux3, vmm_aux1);
    // frexp()
    h->uni_vpsrld(vmm_src, vmm_aux3, n_mantissa_bits);
    h->uni_vcvtdq2ps(vmm_src, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->uni_vsubps(vmm_src, vmm_src, table_val(soft_relu_one_twenty_six));

    // and with mask (to get 0.5 * mantissa)
    h->uni_vandps(vmm_aux3, vmm_aux3, table_val(soft_relu_mantissa_sign_mask));
    // got y. (mantisa)  0.5 < y < 1 (or with (to get 0.5 * mantissa))
    h->uni_vorps(vmm_aux3, vmm_aux3, table_val(half));
    // y  = y - 1
    h->uni_vsubps(vmm_aux3, vmm_aux3, table_val(one));

    // compute log1p polynomial
    h->uni_vmovups(vmm_aux1, table_val(soft_relu_pol, 8));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 7));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 6));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 5));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 4));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 3));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 2));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 1));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 0));
    //calculate ln(2) * n
    h->uni_vmulps(vmm_src, vmm_src, table_val(ln2f));
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux1);
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux0);

    // get vmm_mask = src > max logf
    // y = (x < max log f) ? soft_relu(x) : x
    compute_cmp_mask(vmm_aux2, table_val(exp_ln_flt_max_f), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_fwd(
        const Vmm &vmm_src) {
    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.

    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);
    // we store the original sign and make x negative
    h->uni_vandps(vmm_aux3, vmm_aux3, table_val(sign_mask));
    h->uni_vorps(vmm_src, vmm_src, table_val(sign_mask));

    exp_compute_vector_fwd(vmm_src);
    // dup exp(x)
    h->uni_vmovups(vmm_aux1, vmm_src);
    // (exp(x) + 1)
    h->uni_vaddps(vmm_aux1, vmm_aux1, table_val(one));
    // y = exp(x) / (exp(x) + 1)
    h->uni_vdivps(vmm_src, vmm_src, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    h->uni_vmovups(vmm_aux2, table_val(one));
    h->uni_vsubps(vmm_aux2, vmm_aux2, vmm_src);
    if (has_avx512()) {
        h->vptestmd(k_mask, vmm_aux3, vmm_aux3);
    } else {
        h->uni_vmovups(vmm_mask, vmm_aux3);
    }
    blend_with_mask(vmm_aux2, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_fwd(
        const Vmm &vmm_src) {
    // Save src data on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);
    // x*alpha
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    // sigmoid(x*alpha)
    logistic_compute_vector_fwd(vmm_src);
    // x*sigmoid(alpha*x)
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_fwd(
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
    // If (x < 0) result = qnan;

    // save source on stack to check neg and zero values at the end
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);

    // compute i
    const int approx_order = 5;
    h->uni_vpsrld(vmm_aux1, vmm_src, n_mantissa_bits - approx_order);
    h->uni_vandps(vmm_aux1, vmm_aux1, table_val(log_five_bit_offset));
    h->uni_vpslld(vmm_aux1, vmm_aux1, 1); // multiply i by 2

    // compute anticancellation i
    h->uni_vpsrld(vmm_aux2, vmm_aux1, approx_order);

    // get E, don't care about sign as only positive numbers are considered
    h->uni_vpsrld(vmm_aux3, vmm_src, n_mantissa_bits);
    h->uni_vpaddd(vmm_aux3, vmm_aux3, vmm_aux2);
    h->uni_vcvtdq2ps(vmm_aux3, vmm_aux3);

    // get m (mantissa)
    h->uni_vxorps(vmm_aux2, vmm_aux2, table_val(exponent_bias));
    h->uni_vpslld(vmm_aux2, vmm_aux2, n_mantissa_bits);
    h->uni_vandps(vmm_src, vmm_src, table_val(log_mantissa_mask));
    h->uni_vorps(vmm_src, vmm_src, vmm_aux2);

    // At first, adjust indices for table structure which broadcasts elements
    if (has_avx512()) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 4); // multiply by simd_w = 16
    } else if (isa == avx2) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 3); // multiply by simd_w = 8
    } else if (isa == sse41) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 2); // multiply by simd_w = 4
    }

    const auto it = entry_map_.find(log_predefined_vals);
    assert(it != entry_map_.end());
    const auto table_start_idx = (*it).second.first;

    auto gather_table_values = [&](const Vmm &vmm_dst, const Vmm &vmm_idxs,
                                       size_t offt = 0) {
        Xbyak::Address table_idx = h->ptr[p_table + table_start_idx * vlen
                + offt + vmm_idxs * sizeof(float)];
        if (has_avx512()) {
            h->kmovw(k_mask, table_val(log_full_k_reg_mask));
            h->vgatherdps(vmm_dst | k_mask, table_idx);
        } else if (isa == avx2) {
            h->uni_vmovups(vmm_mask, table_val(log_full_vector_reg_mask));
            h->vgatherdps(vmm_dst, table_idx, vmm_mask);
        } else if (isa == sse41) {
            Xbyak::Reg64 reg_tmp
                    = p_table.getIdx() != h->r9.getIdx() ? h->r9 : h->r10;

            const int gpr_size = 8;
            // save reg_tmp state as we are not allowed to spoil it.
            h->sub(h->rsp, gpr_size);
            h->mov(h->ptr[h->rsp], reg_tmp);

            // rest of code puts indices on stack, fetching a table number based
            // on an index, replaces index with the value, and, finally, moves
            // fetched values into vector register.
            h->sub(h->rsp, vlen);
            h->uni_vmovups(h->ptr[h->rsp], vmm_idxs);

            for (size_t i = 0; i < vlen / sizeof(float); ++i) {
                h->mov(reg_tmp.cvt32(), h->ptr[h->rsp + i * sizeof(float)]);
                h->shl(reg_tmp.cvt32(), 2); // multiply by simd_w
                table_idx = h->ptr[p_table + table_start_idx * vlen + offt
                        + reg_tmp];
                h->mov(reg_tmp.cvt32(), table_idx);
                h->mov(h->ptr[h->rsp + i * sizeof(float)], reg_tmp.cvt32());
            }

            h->uni_vmovups(vmm_dst, h->ptr[h->rsp]);
            h->add(h->rsp, vlen);
            // restore GPR state
            h->mov(reg_tmp, h->ptr[h->rsp]);
            h->add(h->rsp, gpr_size);
        }
    };

    // get r_i, same as table(i)
    gather_table_values(vmm_aux2, vmm_aux1, 0);

    // compute relative error (rel_err = m * r_i - 1)
    h->uni_vfmsub213ps(vmm_aux2, vmm_src, table_val(one));

    // compute polynomial(rel_err)
    h->uni_vmovups(vmm_src, table_val(log_pol, 3));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 2));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 1));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);

    // get log(r_i) = table(i+1)
    gather_table_values(vmm_aux2, vmm_aux1, vlen);

    // compute partial result (pres = E * ln(2) - log(r_i))
    h->uni_vfmadd231ps(vmm_aux2, vmm_aux3, table_val(ln2f));

    // compute (result = polynomial + pres) w/ TwoSum algorithm
    // TODO: restore this instead of version below when asserts are gone
    // h->uni_vaddps(vmm_aux1, vmm_src, vmm_aux2); // res_hi = pol + pres
    // h->uni_vsubps(vmm_aux3, vmm_aux1, vmm_aux2); // res_lo = res_hi - pres
    // h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src); // res_lo = res_lo - pol
    // h->uni_vaddps(vmm_src, vmm_aux1, vmm_aux3); // res_hi = pol + pres

    h->uni_vmovups(vmm_aux1, vmm_src);
    h->uni_vaddps(vmm_aux1, vmm_aux1, vmm_aux2); // res_hi = pol + pres
    h->uni_vmovups(vmm_aux3, vmm_aux1);
    h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_aux2); // res_lo = res_hi - pres
    h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src); // res_lo = res_lo - pol
    h->uni_vmovups(vmm_src, vmm_aux1);
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux3); // res_hi = pol + pres

    // Check original source for zero and neg values. skip blend w/ extreme
    // values if all src values were positive.
    h->uni_vmovups(vmm_aux1, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    Xbyak::Label end_log_label;
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_le_os);
    test_mask();
    h->jz(end_log_label);

    // Blend extreme values into src if reach here.
    // First zero for -inf values...
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_eq_oq);
    blend_with_mask(vmm_src, table_val(log_minus_inf));

    // ...then negative for qnan values.
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(log_qnan));

    h->L(end_log_label);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::pow_compute_vector_fwd(
        const Vmm &vmm_src) {
    // dispatch between special cases.
    if (beta_ == -1) { // alpha / x
        h->uni_vmovups(vmm_aux0, table_val(alpha));
        h->uni_vdivps(vmm_src, vmm_aux0, vmm_src, vmm_aux0);
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
        size_t n_k_regs_to_save = 8;
        if (has_avx512()) {
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
        // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
        // `vlen` should be replaced with `host_isa::vlen` and
        // `host_isa::vecs_count`.
        h->sub(h->rsp, (vecs_count + 2) * vlen);
        for (size_t i = 2; i < vecs_count + 2; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen], Vmm(i - 2));
        h->uni_vmovups(h->ptr[h->rsp + 0 * vlen], vmm_src); // src
        h->uni_vmovups(vmm_src, table_val(beta));
        h->uni_vmovups(h->ptr[h->rsp + 1 * vlen], vmm_src); // beta

        // save function address in gpr to pass in in call instruction
        h->mov(h->rbp, reinterpret_cast<uintptr_t>(powf));

        // align stack on 16-byte as ABI requires
        h->mov(h->rbx, h->rsp);
        h->and_(h->rbx, 0xf);
        h->sub(h->rsp, h->rbx);

        // Take src, apply powf on it and replace value on a stack with dst.
        Xmm xmm0 = Xmm(0), xmm1 = Xmm(1);
        for (size_t i = 0; i < vlen / sizeof(float); ++i) {
            const Address &source = h->ptr[h->rsp + h->rbx + i * sizeof(float)];
            h->uni_vmovss(xmm0, source);
            h->uni_vmovss(xmm1, h->ptr[h->rsp + h->rbx + vlen]); // beta
            h->call(h->rbp);
            h->uni_vmovss(source, xmm0);
        }

        h->add(h->rsp, h->rbx);

        // restore vector registers
        for (size_t i = vecs_count + 1; i >= 2; --i)
            h->uni_vmovups(Vmm(i - 2), h->ptr[h->rsp + i * vlen]);
        h->uni_vmovups(vmm_src, h->ptr[h->rsp + 0 * vlen]);
        h->add(h->rsp, (vecs_count + 2) * vlen);

        // restore k registers
        if (has_avx512()) {
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

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_fwd(
        const Vmm &vmm_src) {
    // Here we approximate erf(x) using the expression by
    // Abramowitz and Stegun from ``Handbook of Mathematical
    // Functions''
    // NOTE: The performance of this kernel can be further improved
    // with a minimax polynomialial expansion, thereby avoiding division
    // and exp. However, so far, this has costed larger accuracy
    // differences with respect to glibc erf based GELU, in particular
    // ~1.0e-5 -- 1.0e-3 absolute error at s = -5.

    // x = s / sqrt(2)
    h->uni_vmulps(vmm_src, vmm_src, table_val(gelu_erf_one_over_sqrt_two));

    // IMPORTANT: we use vmm_aux3 to save `x` as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);

    // -exp(-x*x)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));
    exp_compute_vector_fwd(vmm_src);
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));

    // get sign
    h->uni_vmovups(vmm_aux0, vmm_aux3);
    h->uni_vandps(vmm_aux0, vmm_aux0, table_val(sign_mask));

    // abs(x)
    h->uni_vmovups(vmm_aux1, vmm_aux3);
    abs_compute_vector_fwd(vmm_aux1);

    // t = 1 / (p*x + 1)
    h->uni_vmovups(vmm_aux2, table_val(gelu_erf_approx_const));
    h->uni_vfmadd213ps(vmm_aux2, vmm_aux1, table_val(one));
    h->uni_vmovups(vmm_aux4, table_val(one));
    h->uni_vdivps(vmm_aux4, vmm_aux4, vmm_aux2);

    // -exp(-x*x)*t
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux4);

    // compute polynomialial r
    h->uni_vmovups(vmm_aux1, table_val(gelu_erf_pol, 4));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 3));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 2));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 1));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 0));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vxorps(vmm_src, vmm_src, vmm_aux0);

    // S = 0.5 * s = x / sqrt^2(2)
    h->uni_vmulps(vmm_aux3, vmm_aux3, table_val(gelu_erf_one_over_sqrt_two));
    // GELU = 0.5 * s * (1 + erf) = S + S * erf
    h->uni_vfmadd213ps(vmm_src, vmm_aux3, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    // invariant to whether `s` or `d` is passed.
    // get mask of `s` > 0
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    // fill with alpha, then blend with 1.f
    h->uni_vmovups(vmm_src, table_val(alpha));
    blend_with_mask(vmm_src, table_val(one));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_bwd(
        const Vmm &vmm_src) {
    if (!use_dst_) {
        // R = exp(s)
        exp_compute_vector_fwd(vmm_src);
        // after exponentiation, get mask by comparing with exp(0)=1.f, not 0.f
        compute_cmp_mask(vmm_src, table_val(one), _cmp_gt_os);
        // R * alpha, then blend with 1.f
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else {
        // get mask of `d` > 0
        compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
        // R = `d` + alpha, then blend with 1.f
        h->uni_vaddps(vmm_src, vmm_src, table_val(alpha));
    }
    blend_with_mask(vmm_src, table_val(one));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 1 - d^2 = 1 - tanh^2(s)
    if (!use_dst_) tanh_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux0, table_val(one));
    h->uni_vfnmadd231ps(vmm_aux0, vmm_src, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux0, vmm_src);

    // compute G1(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x^2)
    // compute G2(x) = sqrt_root_two_over_pi * x * (1 + 3 * fitting_const * x^2)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);

    // keep G2 in a separate register
    h->uni_vmovups(vmm_aux2, table_val(gelu_tanh_fitting_const_times_three));
    h->uni_vfmadd213ps(vmm_aux2, vmm_src, table_val(one));

    h->uni_vmovups(vmm_aux1, table_val(gelu_tanh_fitting_const));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vmulps(vmm_aux0, vmm_aux0, table_val(gelu_tanh_sqrt_two_over_pi));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
    h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_aux0);

    // save G2 on stack as tanh uses all available registers
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_aux2);

    // T = tanh(G1(x))
    tanh_compute_vector_fwd(vmm_src);

    h->uni_vmovups(vmm_aux2, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    // compute 0.5 * (1 + T) * (1 + G2 * (1 - T))
    if (isa == sse41) {
        h->uni_vmovups(vmm_aux3, table_val(one));
        h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src);
        h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_aux3);
        h->uni_vaddps(vmm_src, vmm_src, table_val(one));
        h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_src);
        h->uni_vaddps(vmm_src, vmm_src, vmm_aux2);
    } else {
        // 1) R = G2 * (1 - T) = G2 - G2 * T
        h->uni_vfnmadd231ps(vmm_aux2, vmm_aux2, vmm_src);
        // 2) Q = 1 + T
        h->uni_vaddps(vmm_src, vmm_src, table_val(one));
        // 3) res = Q * (1 + R) = Q + Q * R
        h->uni_vfmadd231ps(vmm_src, vmm_src, vmm_aux2);
    }
    h->uni_vmulps(vmm_src, vmm_src, table_val(half));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 2 * s
    h->uni_vmulps(vmm_src, vmm_src, table_val(two));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_bwd(
        const Vmm &vmm_src) {
    // replace positive values with 1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one));
    // replace negative values with -1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(minus_one));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 0.5 / d = 0.5 / sqrt(s)
    if (!use_dst_) sqrt_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux0, table_val(half));
    // h->uni_vdivps(vmm_src, vmm_aux0, vmm_src); // bless sse41
    h->uni_vdivps(vmm_aux0, vmm_aux0, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_bwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_src, table_val(alpha));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    // get mask of values > alpha and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(alpha), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(zero));
    // make all negative values zeros
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero));
    // everything bigger than 0.f should be 1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    logistic_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = d * (1 - d) = d - d * d; d = logistic(s)
    if (!use_dst_) logistic_compute_vector_fwd(vmm_src);
    // h->uni_vfnmadd231ps(vmm_src, vmm_src, vmm_src); // bless sse41
    h->uni_vmovups(vmm_aux0, table_val(one));
    h->uni_vsubps(vmm_aux0, vmm_aux0, vmm_src);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_bwd(
        const Vmm &vmm_src) {
    if (!use_dst_) exp_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_bwd(
        const Vmm &vmm_src) {
    // R = alpha * s
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    // Save R on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);
    // Q = sigmoid(alpha * s)
    logistic_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    // compute Q * (1 + R * (1 - Q))
    if (isa == sse41) {
        h->uni_vmovups(vmm_aux1, table_val(one));
        h->uni_vsubps(vmm_aux1, vmm_aux1, vmm_src);
        h->uni_vmulps(vmm_aux1, vmm_aux1, vmm_aux0);
        h->uni_vaddps(vmm_aux1, vmm_aux1, table_val(one));
        h->uni_vmulps(vmm_src, vmm_src, vmm_aux1);
    } else {
        // T = R * (1 - Q) = R - R * Q
        h->uni_vfnmadd231ps(vmm_aux0, vmm_aux0, vmm_src);
        // Q * (1 + T) = Q + Q * T
        h->uni_vfmadd231ps(vmm_src, vmm_src, vmm_aux0);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 1 / s
    h->uni_vmovups(vmm_aux0, table_val(one));
    // h->uni_vdivps(vmm_src, vmm_aux0, vmm_src); // bless sse41
    h->uni_vdivps(vmm_aux0, vmm_aux0, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_bwd(
        const Vmm &vmm_src) {
    // set result with 1.f
    h->uni_vmovups(vmm_aux1, table_val(one));
    // get mask of values > beta and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(beta), _cmp_gt_os);
    blend_with_mask(vmm_aux1, table_val(zero));
    // get mask of values <= alpha and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(alpha), _cmp_le_os);
    blend_with_mask(vmm_aux1, table_val(zero));
    h->uni_vmovups(vmm_src, vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::pow_compute_vector_bwd(
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
        // Save `s` on stack for later usage
        h->sub(h->rsp, vlen);
        h->uni_vmovups(h->ptr[h->rsp], vmm_src);
        // R = alpha * pow(s, beta)
        pow_compute_vector_fwd(vmm_src);
        // Restore `s` from stack
        h->uni_vmovups(vmm_aux1, h->ptr[h->rsp]);
        h->add(h->rsp, vlen);
        // Save mask of zero elements to convert them into zeros at the end
        if (beta_ >= 1) compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_eq_oq);
        // res = alpha * beta * pow(s, beta - 1) = beta * R / s;
        h->uni_vdivps(vmm_src, vmm_src, vmm_aux1);
        h->uni_vmulps(vmm_src, vmm_src, table_val(beta));

        // beta < 1 leads to NaN as `s` appears in denominator, but beta >= 1
        // should lead to zero, when `s` is zero.
        if (beta_ >= 1) blend_with_mask(vmm_src, table_val(zero));
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_bwd(
        const Vmm &vmm_src) {
    // R = s / sqrt(2)
    h->uni_vmulps(vmm_src, vmm_src, table_val(gelu_erf_one_over_sqrt_two));

    // Save R on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);

    // Q = exp(-R*R)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));
    exp_compute_vector_fwd(vmm_src);

    // T = R / sqrt(pi) * Q
    h->uni_vmovups(vmm_aux2, h->ptr[h->rsp]);
    h->uni_vmulps(vmm_aux2, vmm_aux2, table_val(gelu_erf_one_over_sqrt_pi));
    h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_src);

    // -Q
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));

    // get sign
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->uni_vandps(vmm_aux0, vmm_aux0, table_val(sign_mask));

    // abs(x)
    h->uni_vmovups(vmm_aux1, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    abs_compute_vector_fwd(vmm_aux1);

    // W = 1 / (p * s + 1)
    h->uni_vmovups(vmm_aux3, table_val(gelu_erf_approx_const));
    h->uni_vmovups(vmm_aux4, table_val(one));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, vmm_aux4);
    h->uni_vdivps(vmm_aux4, vmm_aux4, vmm_aux3);

    // Q * W
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux4);

    // compute polynomial r
    h->uni_vmovups(vmm_aux1, table_val(gelu_erf_pol, 4));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 3));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 2));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 1));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 0));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vxorps(vmm_src, vmm_src, vmm_aux0);

    // P = T + 0.5
    h->uni_vaddps(vmm_aux2, vmm_aux2, table_val(half));
    // res = P + 0.5 * erf
    h->uni_vfmadd231ps(vmm_aux2, vmm_src, table_val(half));
    h->uni_vmovups(vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_vecs_count() {
    using namespace alg_kind;
    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return (alpha_ == 0.f) ? 0 : 2;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return 4;
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh: return 5;
            case eltwise_square: return 0;
            case eltwise_abs: return 0;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 0;
            case eltwise_linear: return 1;
            case eltwise_bounded_relu: return 0;
            case eltwise_soft_relu: return 4;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic: return 4;
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp: return 3;
            case eltwise_gelu_tanh: return 5;
            case eltwise_swish: return 4;
            case eltwise_log: return 5;
            case eltwise_clip: return 0;
            case eltwise_pow: return 2;
            case eltwise_gelu_erf: return 5;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return 1;
            case eltwise_elu_use_dst_for_bwd: return 1;
            case eltwise_elu: return 3;
            case eltwise_tanh_use_dst_for_bwd: return 1;
            case eltwise_tanh: return 5;
            case eltwise_square: return 0;
            case eltwise_abs: return 0;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 1;
            case eltwise_linear: return 0;
            case eltwise_bounded_relu: return 1;
            case eltwise_soft_relu: return 4;
            case eltwise_logistic_use_dst_for_bwd: return 1;
            case eltwise_logistic: return 4;
            case eltwise_exp_use_dst_for_bwd: return 0;
            case eltwise_exp: return 3;
            case eltwise_gelu_tanh: return 5;
            case eltwise_swish: return 4;
            case eltwise_log: return 1;
            case eltwise_clip: return 2;
            case eltwise_pow: return 2;
            case eltwise_gelu_erf: return 5;
            default: assert(!"unsupported eltwise algorithm");
        }
    }

    return 0;
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_body(
        size_t start_idx, size_t end_idx) {
    using namespace alg_kind;
    for (size_t idx = start_idx; idx < end_idx; idx++) {
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
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_fwd(Vmm(idx));
                    break;
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
                case eltwise_clip: clip_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_pow: pow_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_fwd(Vmm(idx));
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
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_bwd(Vmm(idx));
                    break;
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
                case eltwise_clip: clip_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_pow: pow_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_bwd(Vmm(idx));
                    break;
                default: assert(!"unsupported eltwise algorithm");
            }
        }
        if (scale_ != 1.f) {
            h->uni_vmulps(Vmm(idx), Vmm(idx), table_val(scale));
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    assert(start_idx < end_idx && end_idx <= vecs_count);

    injector_preamble(start_idx, end_idx);
    compute_body(start_idx_tail, end_idx);
    injector_preamble_tail(start_idx);
    compute_body(start_idx, start_idx_tail);
    injector_postamble();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::prepare_table(bool gen_table) {
    if (!gen_table) return;

    h->align(64);
    h->L(l_table);

    // Run through the map an insert values stored there
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        const auto &t_e = (*it).second; // get map entry for a given key
        const auto &val = t_e.second; // get hex value
        for (size_t d = 0; d < vlen / sizeof(float); ++d)
            h->dd(val);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::register_table_entries() {
    // This function is responsible to pick all necessary constants for a given
    // algorithm, compute right offset for them to be used in table_val() and
    // save the hexadecimal value of them, which will be finally used in
    // prepare_table(). We rely on fact that map iterator will walk through the
    // map in the same order as we saved entries there.

    // IMPORTANT NOTICE: a single key should have a single offset value match.
    // Thus, polynomials have a single offset. This is due multimap.find() may
    // return any value for a given key if there are multiple pairs
    // {key, val[i]}. Proper offset calculation is handled by alg compute code.

    // common values used in several algorithms
    static const table_t common_values {
            {zero, {0, 0x00000000}},
            {half, {1, 0x3f000000}},
            {one, {2, 0x3f800000}},
            {two, {3, 0x40000000}},
            {minus_one, {4, 0xbf800000}},
            {minus_two, {5, 0xc0000000}},
            {ln2f, {6, 0x3f317218}},
            {positive_mask, {7, 0x7fffffff}},
            {sign_mask, {8, 0x80000000}},
            {exponent_bias, {9, 0x0000007f}},
    };

    // exp(x) constants
    static const table_t exp_consts {
            {exp_log2ef, {0, 0x3fb8aa3b}},
            {exp_ln_flt_max_f, {1, 0x42b17218}},
            {exp_ln_flt_min_f, {2, 0xc2aeac50}},
    };

    // exp(x) polynomial approximation
    static const table_t exp_polynomial {
            {exp_pol, {0, 0x3f7ffffb}}, // p1 = 0.999999701f
            {exp_pol, {0, 0x3efffee3}}, // p2 = 0.499991506f
            {exp_pol, {0, 0x3e2aad40}}, // p3 = 0.166676521f
            {exp_pol, {0, 0x3d2b9d0d}}, // p4 = 0.0418978221f
            {exp_pol, {0, 0x3c07cfce}}, // p5 = 0.00828929059f
    };

    // tanh(x) constants for four interval approximation
    static const table_t tanh_consts {
            {tanh_bound_x, {0, 0x39ddb3d7}},
            {tanh_bound_pol, {1, 0x3f0c9f54}},
            {tanh_bound_one, {2, 0x41102cb4}},
    };

    // tanh(x) polynomial approximation
    // Sollya generation script:
    // # comes from Taylor expansion error bound
    //  > linear_sat_point = single(sqrt(3) * 1b-12);
    // # comes from the exp formula cancellation
    //  > exp_bound_point = (single(log(3)/2));
    // # comes from rounding accuracy in float
    //  > one_sat_point = round(atanh(1 - 1b-25), single, RU);
    //  > P = fpminimax(f, [|1, 3, 5, 7, 9|], [|24... |],
    //            [linear_sat_point, exp_bound_point], relative, floating);
    //  > err_bound = D(sup(supnorm(P, tanh(x),
    //          [linear_sat_point, exp_bound_point], relative, theta)));
    //    0x1.fffd6f00b9539p-25
    //  > P;
    //    x * (0x1.fffffep-1 + x^0x1p1 * (-0x1.55539ep-2 + x^0x1p1 *
    //        (0x1.10be3ep-3 + x^0x1p1 * (-0x1.ae57b4p-5
    //        + x^0x1p1 * 0x1.09fa1p-6))))
    static const table_t tanh_polynomial {
            {tanh_pol, {0, 0x3f7fffff}}, // p0 = 0x1.fffffep-1
            {tanh_pol, {0, 0xbeaaa9cf}}, // p1 = 0x1.55539ep-2
            {tanh_pol, {0, 0x3e085f1f}}, // p2 = 0x1.10be3ep-3
            {tanh_pol, {0, 0xbd572bda}}, // p3 = 0x1.ae57b4p-5
            {tanh_pol, {0, 0x3c84fd08}}, // p4 = 0x1.09fa1p-6
    };

    // soft_relu(x) constants
    static const table_t soft_relu_consts {
            {soft_relu_one_twenty_six, {0, 0x42fc0000}},
            {soft_relu_mantissa_sign_mask, {1, 0x807fffff}},
    };

    // soft_relu ln(1 + x) polynomial approximation
    static const table_t soft_relu_polynomial {
            {soft_relu_pol, {0, 0xb2b4637d}}, // p0 = 0.0000000244f
            {soft_relu_pol, {0, 0x3f7fff8e}}, // p1 = 0.9999976971f
            {soft_relu_pol, {0, 0xbf001759}}, // p2 = -0.5002478215f
            {soft_relu_pol, {0, 0x3ea70608}}, // p3 = 0.3272714505f
            {soft_relu_pol, {0, 0xbea3d7bf}}, // p4 = -0.3153830071f
            {soft_relu_pol, {0, 0xbe361d04}}, // p5 = -0.1701777461f
            {soft_relu_pol, {0, 0xbfa8f1e6}}, // p6 = -1.3254635147f
            {soft_relu_pol, {0, 0xbfe1e812}}, // p7 = -1.7971917960f
            {soft_relu_pol, {0, 0xbfc4d30e}}, // p8 = -1.5652673123f
    };

    // gelu_tanh(x) constants (formula defined)
    static const table_t gelu_tanh_consts {
            {gelu_tanh_fitting_const, {0, 0x3d372713}},
            {gelu_tanh_fitting_const_times_three, {1, 0x3e095d4f}},
            {gelu_tanh_sqrt_two_over_pi, {2, 0x3f4c4229}},
    };

    // gelu_erf(x) constants (formula defined)
    static const table_t gelu_erf_consts {
            {gelu_erf_approx_const, {0, 0x3ea7ba05}},
            {gelu_erf_one_over_sqrt_two, {1, 0x3f3504f3}},
            {gelu_erf_one_over_sqrt_pi, {2, 0x3f106eba}},
    };

    // gelu_erf(x) polynomial approximation
    static const table_t gelu_erf_polynomial {
            {gelu_erf_pol, {0, 0x3e827906}}, // p1 = 0.254829592f
            {gelu_erf_pol, {0, 0xbe91a98e}}, // p2 = -0.284496736f
            {gelu_erf_pol, {0, 0x3fb5f0e3}}, // p3 = 1.421413741f
            {gelu_erf_pol, {0, 0xbfba00e3}}, // p4 = -1.453152027f
            {gelu_erf_pol, {0, 0x3f87dc22}}, // p5 = 1.061405429f
    };

    // log(x) constants
    static const table_t log_consts {
            {log_minus_inf, {0, 0xff800000}},
            {log_qnan, {1, 0x7fc00000}},
            {log_mantissa_mask, {2, 0x007fffff}},
            {log_full_k_reg_mask, {3, 0x0000ffff}},
            {log_full_vector_reg_mask, {4, 0xffffffff}},
            {log_five_bit_offset, {5, 0x0000001f}},
    };

    // log(x) polynomial approximation
    static const table_t log_polynomial {
            {log_pol, {0, 0xbf000000}}, // p1 = -0.5f
            {log_pol, {0, 0x3eaaaaab}}, // p2 =  0.333333343f
            {log_pol, {0, 0xbe8004ab}}, // p3 = -0.250035613f
            {log_pol, {0, 0x3e4cc8a3}}, // p4 =  0.199984118f
    };

    // log(x) pre-defined values. First goes index}, then val[index].
    static const table_t log_predefined_values {
            {log_predefined_vals, {0, 0x3f800000}}, //  0: 1
            {log_predefined_vals, {0, 0xc2b00f34}}, //  1: -88.029693603515625
            {log_predefined_vals, {0, 0x3f780000}}, //  2: 0.96875
            {log_predefined_vals, {0, 0xc2affef2}}, //  3: -87.9979400634765625
            {log_predefined_vals, {0, 0x3f700000}}, //  4: 0.9375
            {log_predefined_vals, {0, 0xc2afee29}}, //  5: -87.9651565551757812
            {log_predefined_vals, {0, 0x3f680000}}, //  6: 0.90625
            {log_predefined_vals, {0, 0xc2afdccd}}, //  7: -87.9312515258789062
            {log_predefined_vals, {0, 0x3f600000}}, //  8: 0.875
            {log_predefined_vals, {0, 0xc2afcad6}}, //  9: -87.8961639404296875
            {log_predefined_vals, {0, 0x3f580000}}, // 10: 0.84375
            {log_predefined_vals, {0, 0xc2afb837}}, // 11: -87.859794616699218
            {log_predefined_vals, {0, 0x3f580000}}, // 12: 0.84375
            {log_predefined_vals, {0, 0xc2afb837}}, // 13: -87.859794616699218
            {log_predefined_vals, {0, 0x3f500000}}, // 14: 0.8125
            {log_predefined_vals, {0, 0xc2afa4e4}}, // 15: -87.822052001953125
            {log_predefined_vals, {0, 0x3f480000}}, // 16: 0.78125
            {log_predefined_vals, {0, 0xc2af90cf}}, // 17: -87.782829284667968
            {log_predefined_vals, {0, 0x3f480000}}, // 18: 0.78125
            {log_predefined_vals, {0, 0xc2af90cf}}, // 19: -87.782829284667968
            {log_predefined_vals, {0, 0x3f400000}}, // 20: 0.75
            {log_predefined_vals, {0, 0xc2af7be9}}, // 21: -87.742012023925781
            {log_predefined_vals, {0, 0x3f400000}}, // 22: 0.75
            {log_predefined_vals, {0, 0xc2af7be9}}, // 23: -87.742012023925781
            {log_predefined_vals, {0, 0x3f380000}}, // 24: 0.71875
            {log_predefined_vals, {0, 0xc2af661e}}, // 25: -87.699447631835937
            {log_predefined_vals, {0, 0x3f380000}}, // 26: 0.71875
            {log_predefined_vals, {0, 0xc2af661e}}, // 27: -87.699447631835937
            {log_predefined_vals, {0, 0x3f300000}}, // 28: 0.6875
            {log_predefined_vals, {0, 0xc2af4f5c}}, // 29: -87.654998779296875
            {log_predefined_vals, {0, 0x3f300000}}, // 30: 0.6875
            {log_predefined_vals, {0, 0xc2af4f5c}}, // 31: -87.654998779296875
            {log_predefined_vals, {0, 0x3fa80000}}, // 32: 1.3125
            {log_predefined_vals, {0, 0xc2b09a6f}}, // 33: -88.301628112792968
            {log_predefined_vals, {0, 0x3fa80000}}, // 34: 1.3125
            {log_predefined_vals, {0, 0xc2b09a6f}}, // 35: -88.301628112792968
            {log_predefined_vals, {0, 0x3fa00000}}, // 36: 1.25
            {log_predefined_vals, {0, 0xc2b08174}}, // 37: -88.252838134765625
            {log_predefined_vals, {0, 0x3fa00000}}, // 38: 1.25
            {log_predefined_vals, {0, 0xc2b08174}}, // 39: -88.252838134765625
            {log_predefined_vals, {0, 0x3fa00000}}, // 40: 1.25
            {log_predefined_vals, {0, 0xc2b08174}}, // 41: -88.252838134765625
            {log_predefined_vals, {0, 0x3f980000}}, // 42: 1.1875
            {log_predefined_vals, {0, 0xc2b06731}}, // 43: -88.201545715332031
            {log_predefined_vals, {0, 0x3f980000}}, // 44: 1.1875
            {log_predefined_vals, {0, 0xc2b06731}}, // 45: -88.201545715332031
            {log_predefined_vals, {0, 0x3f900000}}, // 46: 1.125
            {log_predefined_vals, {0, 0xc2b04b82}}, // 47: -88.147476196289062
            {log_predefined_vals, {0, 0x3f900000}}, // 48: 1.125
            {log_predefined_vals, {0, 0xc2b04b82}}, // 49: -88.147476196289062
            {log_predefined_vals, {0, 0x3f900000}}, // 50: 1.125
            {log_predefined_vals, {0, 0xc2b04b82}}, // 51: -88.147476196289062
            {log_predefined_vals, {0, 0x3f900000}}, // 52: 1.125
            {log_predefined_vals, {0, 0xc2b04b82}}, // 53: -88.147476196289062
            {log_predefined_vals, {0, 0x3f880000}}, // 54: 1.0625
            {log_predefined_vals, {0, 0xc2b02e3e}}, // 55: -88.090316772460937
            {log_predefined_vals, {0, 0x3f880000}}, // 56: 1.0625
            {log_predefined_vals, {0, 0xc2b02e3e}}, // 57: -88.090316772460937
            {log_predefined_vals, {0, 0x3f880000}}, // 58: 1.0625
            {log_predefined_vals, {0, 0xc2b02e3e}}, // 59: -88.090316772460937
            {log_predefined_vals, {0, 0x3f800000}}, // 60: 1
            {log_predefined_vals, {0, 0xc2b00f34}}, // 61: -88.029693603515625
            {log_predefined_vals, {0, 0x3f800000}}, // 62: 1
            {log_predefined_vals, {0, 0xc2b00f34}}, // 63: -88.029693603515625
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
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_ = true; break;
                default: break;
            }
        }

        bool exp_ = false;
        bool tanh_ = false;
        bool soft_relu_ = false;
        bool gelu_tanh_ = false;
        bool gelu_erf_ = false;
        bool log_ = false;

        bool exp() const {
            return exp_ || tanh_ || soft_relu_ || gelu_tanh_ || gelu_erf_;
        }
        bool tanh() const { return tanh_ || gelu_tanh_; }
        bool soft_relu() const { return soft_relu_; }
        bool gelu_tanh() const { return gelu_tanh_; }
        bool gelu_erf() const { return gelu_erf_; }
        bool log() const { return log_; }
    };

    need_t need(alg_);
    size_t off = 0;

    auto push_arg_entry_of = [&](const key_t key, const float val) {
        table_entry_t te = std::make_pair(off++, float2int(val));
        entry_map_.insert(std::make_pair(key, te));
    };

    auto push_entries_of = [&](const table_t &t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto te = (*it).second; // copy values from table
            te.first += off; // shift offset with current off value
            entry_map_.insert(std::make_pair((*it).first, te)); // put in map
        }
        off += t.size();
    };

    push_arg_entry_of(scale, scale_);
    push_arg_entry_of(alpha, alpha_);
    push_arg_entry_of(beta, beta_);
    push_entries_of(common_values);
    if (need.exp()) push_entries_of(exp_consts);
    if (need.exp()) push_entries_of(exp_polynomial);
    if (need.tanh()) push_entries_of(tanh_consts);
    if (need.tanh()) push_entries_of(tanh_polynomial);
    if (need.soft_relu()) push_entries_of(soft_relu_consts);
    if (need.soft_relu()) push_entries_of(soft_relu_polynomial);
    if (need.gelu_tanh()) push_entries_of(gelu_tanh_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_polynomial);
    if (need.log()) push_entries_of(log_consts);
    if (need.log()) push_entries_of(log_polynomial);
    if (need.log()) push_entries_of(log_predefined_values);
}

template struct jit_uni_eltwise_injector_f32<avx512_core>;
template struct jit_uni_eltwise_injector_f32<avx512_common>;
template struct jit_uni_eltwise_injector_f32<avx2>;
template struct jit_uni_eltwise_injector_f32<sse41>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
