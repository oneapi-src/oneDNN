/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace Xbyak;
static constexpr int n_mantissa_bits = 23;
static constexpr int k_mask_size = 4;

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble(
        size_t start_idx, size_t end_idx) {
    preserved_vecs_count = 0;
    vecs_to_preserve = aux_vecs_count(alg_);
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
    if (utils::one_of(isa, avx512_common, avx512_core)) {
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
    if (utils::one_of(isa, avx512_common, avx512_core)) {
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
    if (utils::one_of(isa, avx512_common, avx512_core)) {
        h->kortestw(k_mask, k_mask);
    } else {
        h->uni_vtestps(vmm_mask, vmm_mask);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector(const Vmm &vmm_src) {
    // get mask of values lower than log(FLT_MIN) to zero them in the output
    compute_cmp_mask(vmm_src, table_val(11), _cmp_lt_os);

    h->uni_vminps(vmm_src, vmm_src, table_val(10));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(11));
    h->uni_vmovups(vmm_aux1, vmm_src);
    //calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(2));
    h->uni_vaddps(vmm_src, vmm_src, table_val(1));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux2, vmm_src, _op_floor);

    //keep fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux2); //vmm_src = fx

    //x = x - fx * ln2
    h->uni_vfnmadd231ps(vmm_aux1, vmm_aux2, table_val(3));

    // compute 2^n
    h->uni_vcvtps2dq(vmm_aux2, vmm_src);
    h->uni_vpaddd(vmm_aux2, vmm_aux2, table_val(4));
    h->uni_vpslld(vmm_aux2, vmm_aux2, n_mantissa_bits); //Vmm(6) = 2^-fx

    // use vmm_src as tmp vmm_zero when applying mask
    h->uni_vpxor(vmm_src, vmm_src, vmm_src);
    // set zeroes at those points which were < log(FLT_MIN)
    blend_with_mask(vmm_aux2, vmm_src);

    // y = p5
    h->uni_vmovups(vmm_src, table_val(9));
    // y = y * x + p4
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(8));
    // y = y * x + p3
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(7));
    // y = y * x + p2
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(6));
    // y = y * x + p1
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(5));
    // y = y * x + 1.f
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(0)); //exp(q)
    // y = y * 2^n
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector(
        const Vmm &vmm_src) {
    const int alpha_off = 0, zero_off = 1;

    h->uni_vmovups(vmm_aux1, vmm_src);
    compute_cmp_mask(vmm_src, table_val(zero_off), _cmp_gt_os);
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha_off));
    blend_with_mask(vmm_src, vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_zero_ns_compute_vector(
        const Vmm &vmm_src) {
    const int zero_off = 1;
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero_off));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector(const Vmm &vmm_src) {
    const int alpha_off = 25, zero_off = 26;

    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    // compute exponent
    h->uni_vmovups(vmm_aux3, vmm_src);
    exp_compute_vector(vmm_src);

    // alpha * (exp(x) - 1)
    h->uni_vsubps(vmm_src, vmm_src, table_val(0));
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha_off));

    // combine with mask
    compute_cmp_mask(vmm_aux3, table_val(zero_off), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector(
        const Vmm &vmm_src) {
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
    h->uni_vandps(vmm_aux4, vmm_aux4, table_val(12));
    h->uni_vandps(vmm_src, vmm_src, table_val(17));

    // if x < linear_sat_point for all inputs, we just return the input
    h->uni_vmovups(vmm_aux1, vmm_src);
    test_exit(table_val(13));

    // if one of the mask is one, we have to compute an better approx
    h->uni_vmovups(vmm_aux2, vmm_src);
    h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_aux2);
    h->uni_vmovups(vmm_aux3, table_val(22));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(21));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(20));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(19));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(18));
    h->uni_vmulps(vmm_aux3, vmm_aux3, vmm_src);

    // we blend only the result that need update
    blend_with_mask(vmm_aux1, vmm_aux3);

    // if x < exp_bound_point, we go to return point
    test_exit(table_val(14));

    // if not we use a better approx 1 - 2 / (1 + exp(2x))
    // compute 2x
    h->uni_vmovups(vmm_aux3, vmm_src);
    h->uni_vaddps(vmm_aux3, vmm_aux3, vmm_aux3);

    // Compute exp(2x)
    // We need to save kmask, vmm_mask, vmm_aux1, vmm_aux2 and vmm_src as exp
    // uses them.
    // vmm_src is not more read afterwards, so we do not have to save it
    auto stack_size = 4 * vlen
            + utils::one_of(isa, avx512_common, avx512_core) * k_mask_size;
    h->sub(h->rsp, stack_size);
    h->uni_vmovups(h->ptr[h->rsp + 0 * vlen], vmm_mask);
    h->uni_vmovups(h->ptr[h->rsp + 1 * vlen], vmm_aux1);
    h->uni_vmovups(h->ptr[h->rsp + 2 * vlen], vmm_aux2);
    h->uni_vmovups(h->ptr[h->rsp + 3 * vlen], vmm_src);
    if (utils::one_of(isa, avx512_common, avx512_core))
        h->kmovw(h->ptr[h->rsp + 4 * vlen], k_mask);

    exp_compute_vector(vmm_aux3);

    h->uni_vmovups(vmm_mask, h->ptr[h->rsp + 0 * vlen]);
    h->uni_vmovups(vmm_aux1, h->ptr[h->rsp + 1 * vlen]);
    h->uni_vmovups(vmm_aux2, h->ptr[h->rsp + 2 * vlen]);
    h->uni_vmovups(vmm_src, h->ptr[h->rsp + 3 * vlen]);
    if (utils::one_of(isa, avx512_common, avx512_core))
        h->kmovw(k_mask, h->ptr[h->rsp + 4 * vlen]);
    h->add(h->rsp, stack_size);

    // 1 + exp(2x)
    h->uni_vaddps(vmm_aux3, vmm_aux3, table_val(0));

    // 1 - 2 / (1 + exp(2x))
    h->uni_vmovups(vmm_aux2, table_val(16));
    h->uni_vdivps(vmm_aux2, vmm_aux2, vmm_aux3);
    h->uni_vaddps(vmm_aux2, vmm_aux2, table_val(0));

    // we blend only the result that need update
    blend_with_mask(vmm_aux1, vmm_aux2);

    // finally, we saturate to 1 if needed
    // TODO: maybe move that up if most inputs saturate in practice
    compute_cmp_mask(vmm_src, table_val(15), _cmp_ge_os);
    h->uni_vmovups(vmm_aux2, table_val(0));
    blend_with_mask(vmm_aux1, vmm_aux2);

    h->L(end_tanh_label);
    {
        // we apply the sign of x to the result and we are done
        h->uni_vmovups(vmm_src, vmm_aux1);
        h->uni_vpxor(vmm_src, vmm_src, vmm_aux4);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_compute_vector(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux0, vmm_src);

    // compute G(x) = a * x * (1 + b * x * x)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vmovups(vmm_aux1, table_val(23));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(0));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
    h->uni_vmulps(vmm_src, vmm_src, table_val(24));

    // save x on stack as tanh uses vmm_aux0
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_aux0);

    // compute tanh G(x)
    tanh_compute_vector(vmm_src);

    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    // compute 0.5 * x * (1 + tanh)
    h->uni_vaddps(vmm_src, vmm_src, table_val(0));
    h->uni_vmulps(vmm_src, vmm_src, table_val(1));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector(
        const Vmm &vmm_src) {
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector(const Vmm &vmm_src) {
    // compute abs(x) = _mm_and_ps(x, 01111..111));
    h->uni_vandps(vmm_src, vmm_src, table_val(0));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector(
        const Vmm &vmm_src) {
    h->uni_vsqrtps(vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector(
        const Vmm &vmm_src) {
    // compute x = alpha * x + beta;
    h->uni_vmovups(vmm_aux0, table_val(0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux0, table_val(1));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector(
        const Vmm &vmm_src) {
    // compute bounded relu */
    int upper_bound_off = 0, lower_bound_off = 1;
    h->uni_vmaxps(vmm_src, vmm_src, table_val(lower_bound_off));
    h->uni_vminps(vmm_src, vmm_src, table_val(upper_bound_off));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector(
        const Vmm &vmm_src) {
    // compute clip
    int upper_bound_off = 1, lower_bound_off = 0;
    h->uni_vmaxps(vmm_src, vmm_src, table_val(lower_bound_off));
    h->uni_vminps(vmm_src, vmm_src, table_val(upper_bound_off));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector(
        const Vmm &vmm_src) {
    // duplicate src
    h->uni_vmovups(vmm_aux2, vmm_src);

    h->uni_vminps(vmm_src, vmm_src, table_val(24));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(25));
    h->uni_vmovups(vmm_aux1, vmm_src);
    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(2));
    h->uni_vaddps(vmm_src, vmm_src, table_val(1));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux0, vmm_src, _op_floor);

    // keep fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux0); //vmm_src = fx
    // calculation fx * ln2
    h->uni_vmulps(vmm_aux0, vmm_aux0, table_val(3));
    // x = x - fx * ln2
    h->uni_vsubps(vmm_aux1, vmm_aux1, vmm_aux0);
    // y = p5
    h->uni_vmovups(vmm_aux3, table_val(22));
    // y = y * x + p4
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(21));
    // y = y * x + p3
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(20));
    // y = y * x + p2
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(19));
    // y = y * x + p1
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(0));
    // y = y * x + p0
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(17));

    // compute 2^(-n)
    if (utils::one_of(isa, avx512_common, avx512_core)) {
        h->vmulps(vmm_aux1, vmm_src, table_val(23));
        h->vcvtps2dq(vmm_aux1, vmm_aux1);
    } else {
        h->uni_vcvtps2dq(vmm_aux1, vmm_src);
        h->uni_vpsignd(vmm_aux1, vmm_aux1, table_val(23));
    }

    h->uni_vpaddd(vmm_aux1, vmm_aux1, table_val(4));
    h->uni_vpslld(vmm_aux1, vmm_aux1, n_mantissa_bits); //vmm_aux1 = 2^-fx
    // calculate ln(1 + y)
    h->uni_vaddps(vmm_aux3, vmm_aux3, vmm_aux1);
    // frexp()
    h->uni_vpsrld(vmm_src, vmm_aux3, n_mantissa_bits);
    h->uni_vcvtdq2ps(vmm_src, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->uni_vsubps(vmm_src, vmm_src, table_val(5));

    h->uni_vandps(vmm_aux3, vmm_aux3, table_val(6));
    // got y. (mantisa)  0.5 < y < 1
    h->uni_vorps(vmm_aux3, vmm_aux3, table_val(7));
    // y  = y - 1
    h->uni_vsubps(vmm_aux3, vmm_aux3, table_val(0));
    // y = p8
    h->uni_vmovups(vmm_aux1, table_val(16));
    // y = y * x + p7
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(15));
    // y = y * x + p6
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(14));
    // y = y * x + p5
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(13));
    // y = y * x + p4
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(12));
    // y = y * x + p3
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(11));
    // y = y * x + p2
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(10));
    // y = y * x + p1
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(9));
    // y = y * x + p0 ; p0 = 0
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(8));
    //calculate ln(2) * n
    h->uni_vmulps(vmm_src, vmm_src, table_val(3));
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux1);
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux0);

    // get vmm_mask = src > max logf
    // y = (x < max log f) ? soft_relu(x) : x
    compute_cmp_mask(vmm_aux2, table_val(24), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector(
        const Vmm &vmm_src) {
    // we store the original sign and make x negative
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);
    h->uni_vandps(vmm_aux3, vmm_aux3, table_val(12));
    h->uni_vorps(vmm_src, vmm_src, table_val(12));

    exp_compute_vector(vmm_src);
    // dup exp(x)
    h->uni_vmovups(vmm_aux1, vmm_src);
    // (exp(x) + 1)
    h->uni_vaddps(vmm_aux1, vmm_aux1, table_val(0));
    // y = exp(x) / (exp(x) + 1)
    h->uni_vdivps(vmm_src, vmm_src, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    h->uni_vmovups(vmm_aux2, table_val(0));
    h->uni_vsubps(vmm_aux2, vmm_aux2, vmm_src);
    if (utils::one_of(isa, avx512_common, avx512_core)) {
        h->vptestmd(k_mask, vmm_aux3, vmm_aux3);
    } else {
        h->uni_vmovups(vmm_mask, vmm_aux3);
    }
    blend_with_mask(vmm_aux2, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector(
        const Vmm &vmm_src) {
    const int alpha_off = 25;
    // Save src data on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);
    // x*alpha
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha_off));
    // sigmoid(x*alpha)
    logistic_compute_vector(vmm_src);
    // x*sigmoid(alpha*x)
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}

// Source: J.-M. Muller and others, Handbook of Floating-Point Arithmetic, 2010.
// Here is a brief mathematics to approximate log(x):
// log(x) = E * log(2) + log(y), where -log(2)/2 <= log(y) <= log(2)/2;
// log(y) = log(1 + z) - log(r_i), where z = y * r_i - 1, r_i approximates 1/y,
//   i is index of one of precomputed values;
// log(1 + z) ~~ polynom(z), =>
// if (x is normal)
//     log(x) ~~ E * log(2) + polynom(z) - log(r_i),
// where log(r_i) is table value.
//
// If (x == 0) result = -inf;
// If (x < 0) result = qnan;
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector(const Vmm &vmm_src) {
    // save source on stack to check neg and zero values at the end
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);

    // compute i
    const int index_5bits_off = 1, approx_order = 5;
    h->uni_vpsrld(vmm_aux1, vmm_src, n_mantissa_bits - approx_order);
    h->uni_vandps(vmm_aux1, vmm_aux1, table_val(index_5bits_off));
    h->uni_vpslld(vmm_aux1, vmm_aux1, 1); // multiply i by 2

    // compute anticancellation i
    h->uni_vpsrld(vmm_aux2, vmm_aux1, approx_order);

    // get E, don't care about sign as only positive numbers are considered
    h->uni_vpsrld(vmm_aux3, vmm_src, n_mantissa_bits);
    h->uni_vpaddd(vmm_aux3, vmm_aux3, vmm_aux2);
    h->uni_vcvtdq2ps(vmm_aux3, vmm_aux3);

    // get m (mantissa)
    const int mant_mask_off = 2, one_off = 3, exp_bias_off = 4;
    h->uni_vxorps(vmm_aux2, vmm_aux2, table_val(exp_bias_off));
    h->uni_vpslld(vmm_aux2, vmm_aux2, n_mantissa_bits);
    h->uni_vandps(vmm_src, vmm_src, table_val(mant_mask_off));
    h->uni_vorps(vmm_src, vmm_src, vmm_aux2);

    // At first, adjust indices for table structure which broadcasts elements
    if (utils::one_of(isa, avx512_common, avx512_core)) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 4); // multiply by simd_w = 16
    } else if (isa == avx2) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 3); // multiply by simd_w = 8
    } else if (isa == sse41) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 2); // multiply by simd_w = 4
    }

    auto gather_table_values = [&](const Vmm &vmm_dst, const Vmm &vmm_idxs,
                                       size_t offt = 0) {
        const int table_start_idx = 14;
        // IMPORTANT: this is a slippery place as gatherps relies on the fact
        //         that there is integer scale offset: p_table + **vlen**
        Xbyak::Address table_idx = h->ptr[p_table + vlen
                + table_start_idx * vlen + offt + vmm_idxs * sizeof(float)];
        if (utils::one_of(isa, avx512_common, avx512_core)) {
            h->kmovw(k_mask, table_val(5));
            h->vgatherdps(vmm_dst | k_mask, table_idx);
        } else if (isa == avx2) {
            h->uni_vmovups(vmm_mask, table_val(6));
            h->vgatherdps(vmm_dst, table_idx, vmm_mask);
        } else if (isa == sse41) {
            Xbyak::Reg64 reg_tmp = p_table.getIdx() != Xbyak::util::r9.getIdx()
                    ? Xbyak::util::r9
                    : Xbyak::util::r10;

            int gpr_size = 8;
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
                // IMPORTANT: same notice as above
                table_idx = h->ptr[p_table + vlen + table_start_idx * vlen
                        + offt + reg_tmp];
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
    h->uni_vfmsub213ps(vmm_aux2, vmm_src, table_val(one_off));

    // compute polynom(rel_err)
    const int p0_off = 7, p1_off = 8, p2_off = 9, p3_off = 10, p4_off = one_off;
    h->uni_vmovups(vmm_src, table_val(p0_off));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(p1_off));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(p2_off));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(p3_off));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(p4_off));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);

    // get log(r_i) = table(i+1)
    gather_table_values(vmm_aux2, vmm_aux1, vlen);

    // compute partial result (pres = E * log(2) - log(r_i))
    const int log2_const_off = 11;
    h->uni_vfmadd231ps(vmm_aux2, vmm_aux3, table_val(log2_const_off));

    // compute (result = polynom + pres) w/ TwoSum algorithm
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
    const int zero_off = 0;
    compute_cmp_mask(vmm_aux1, table_val(zero_off), _cmp_le_os);
    test_mask();
    h->jz(end_log_label);

    // Blend extreme values into src if reach here, first zero then negative
    const int minus_inf_off = 12;
    compute_cmp_mask(vmm_aux1, table_val(zero_off), _cmp_eq_oq);
    blend_with_mask(vmm_src, table_val(minus_inf_off));

    const int qnan_off = 13;
    compute_cmp_mask(vmm_aux1, table_val(zero_off), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(qnan_off));

    h->L(end_log_label);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d)
        h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d)
        h->dd(0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_prepare_table() {
    const unsigned int cvals[] = {
            0x3f800000, // [0] 1.0f
            0x3f000000, // [1] 0.5f
            0x3fb8aa3b, // [2] log2ef = 1.44269502f
            0x3f317218, // [3] ln2f =   0.69314718f
            0x0000007f, // [4] 0x7f
            // exp(x) polynom
            0x3f7ffffb, // [5] p1 = 0.999999701f
            0x3efffee3, // [6] p2 = 0.499991506f
            0x3e2aad40, // [7] p3 = 0.166676521f
            0x3d2b9d0d, // [8] p4 = 0.0418978221f
            0x3c07cfce, // [9] p5 = 0.00828929059f
            0x42b17218, //[10] logf(FLT_MAX)
            0xc2aeac50, //[11] logf(FLT_MIN)
            // tanh(x) constants,
            0x80000000, //[12] mask to extract sign
            0x39ddb3d7, //[13] arg below which tanh(x) = x
            0x3f0c9f54, //[14] arg below which pol approx is valid
            0x41102cb4, //[15] arg after which tanh(x) = 1
            0xc0000000, //[16] -2.0f
            0x7fffffff, //[17] mask to make positive
            // tanh pol approx
            0x3f7fffff, //[18] p0
            0xbeaaa9cf, //[19] p1
            0x3e085f1f, //[20] p2
            0xbd572bda, //[21] p3
            0x3c84fd08, //[22] p4
            // gelu approx constants
            0x3d372713, //[23] 0.044715
            0x3f4c4229, //[24] sqrt(2/pi)
    };

    for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(float); ++d)
            h->dd(cvals[i]);
    }

    for (size_t d = 0; d < vlen / sizeof(float); ++d)
        h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d)
        h->dd(0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_prepare_table() {
    const unsigned int cvals[] = {
            0x3f800000, // [0] 1.0f
            0x3f000000, // [1] 0.5f
            0x3fb8aa3b, // [2] log2ef = 1.44269502f
            0x3f317218, // [3] ln2f =   0.69314718f
            0x0000007f, // [4] 0x7f
            0x42fc0000, // [5] 126
            0x807fffff, // [6] and with (to get 0.5 * mantissa)
            0x3f000000, // [7] or with (to get 0.5 * mantissa)
            // ln(1 + x) polynomial
            0xb2b4637d, // [8]  p0 = 0.0000000244f
            0x3f7fff8e, // [9]  p1 = 0.9999976971f
            0xbf001759, //[10]  p2 = -0.5002478215f
            0x3ea70608, //[11]  p3 = 0.3272714505f
            0xbea3d7bf, //[12]  p4 = -0.3153830071f
            0xbe361d04, //[13]  p5 = -0.1701777461f
            0xbfa8f1e6, //[14]  p6 = -1.3254635147f
            0xbfe1e812, //[15]  p7 = -1.7971917960f
            0xbfc4d30e, //[16]  p8 = -1.5652673123f
            // exp(x) polynomial
            0x3f800001, //[17]  p0 = 1.0000001f
            0x3f800000, //[18]  p1 = 1.0f
            0x3efffe85, //[19]  p2 = 0.4999887f
            0x3e2aaa3e, //[20]  p3 = 0.16666505f
            0x3d2bb1b1, //[21]  p4 = 0.041917507f
            0x3c091ec1, //[22]  p5 = 0.008369149f
            0xbf800000, //[23] is required for sign changing
            // TODO: update values [24] and [25] from comments as they are more precise
            0x42b0c0a5, //[24] max logf = 88.3762589f //0x42b17218, //[24] logf(FLT_MAX)
            0xc1766666 //[25] min logf = -14.5f      //0xc2aeac50, //[25] logf(FLT_MIN)
    };

    for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(float); ++d) {
            h->dd(cvals[i]);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d)
        h->dd(0); //  [0] zero

    const unsigned int cvals[] = {
            0x0000001f, //  [1] 31 = 2^approx_order - 1
            0x007fffff, //  [2] mask for mantissa bits
            0x3f800000, //  [3] 1.0f, also a mask for exponent bits
            0x0000007f, //  [4] 127: exponent bias
            0x0000ffff, //  [5] k_mask set of one
            0xffffffff, //  [6] set vmm_mask full of ones
            0x3e4cc8a3, //  [7] p0 =  0.199984118f
            0xbe8004ab, //  [8] p1 = -0.250035613f
            0x3eaaaaab, //  [9] p2 =  0.333333343f
            0xbf000000, // [10] p3 = -0.5f
            0x3f317218, // [11] log(2)
            0xff800000, // [12] -inf for zero src values
            0x7fc00000, // [13] qnan for negative src values
            // table values for log. notation: "i: value(i)"
            0x3f800000, // [14]  0: 1
            0xc2b00f34, // [15]  1: -88.029693603515625
            0x3f780000, // [16]  2: 0.96875
            0xc2affef2, // [17]  3: -87.9979400634765625
            0x3f700000, // [18]  4: 0.9375
            0xc2afee29, // [19]  5: -87.96515655517578125
            0x3f680000, // [20]  6: 0.90625
            0xc2afdccd, // [21]  7: -87.93125152587890625
            0x3f600000, // [22]  8: 0.875
            0xc2afcad6, // [23]  9: -87.8961639404296875
            0x3f580000, // [24] 10: 0.84375
            0xc2afb837, // [25] 11: -87.85979461669921875
            0x3f580000, // [26] 12: 0.84375
            0xc2afb837, // [27] 13: -87.85979461669921875
            0x3f500000, // [28] 14: 0.8125
            0xc2afa4e4, // [29] 15: -87.822052001953125
            0x3f480000, // [30] 16: 0.78125
            0xc2af90cf, // [31] 17: -87.78282928466796875
            0x3f480000, // [32] 18: 0.78125
            0xc2af90cf, // [33] 19: -87.78282928466796875
            0x3f400000, // [34] 20: 0.75
            0xc2af7be9, // [35] 21: -87.74201202392578125
            0x3f400000, // [36] 22: 0.75
            0xc2af7be9, // [37] 23: -87.74201202392578125
            0x3f380000, // [38] 24: 0.71875
            0xc2af661e, // [39] 25: -87.6994476318359375
            0x3f380000, // [40] 26: 0.71875
            0xc2af661e, // [41] 27: -87.6994476318359375
            0x3f300000, // [42] 28: 0.6875
            0xc2af4f5c, // [43] 29: -87.654998779296875
            0x3f300000, // [44] 30: 0.6875
            0xc2af4f5c, // [45] 31: -87.654998779296875
            0x3fa80000, // [46] 32: 1.3125
            0xc2b09a6f, // [47] 33: -88.30162811279296875
            0x3fa80000, // [48] 34: 1.3125
            0xc2b09a6f, // [49] 35: -88.30162811279296875
            0x3fa00000, // [50] 36: 1.25
            0xc2b08174, // [51] 37: -88.252838134765625
            0x3fa00000, // [52] 38: 1.25
            0xc2b08174, // [53] 39: -88.252838134765625
            0x3fa00000, // [54] 40: 1.25
            0xc2b08174, // [55] 41: -88.252838134765625
            0x3f980000, // [56] 42: 1.1875
            0xc2b06731, // [57] 43: -88.20154571533203125
            0x3f980000, // [58] 44: 1.1875
            0xc2b06731, // [59] 45: -88.20154571533203125
            0x3f900000, // [60] 46: 1.125
            0xc2b04b82, // [61] 47: -88.1474761962890625
            0x3f900000, // [62] 48: 1.125
            0xc2b04b82, // [63] 49: -88.1474761962890625
            0x3f900000, // [64] 50: 1.125
            0xc2b04b82, // [65] 51: -88.1474761962890625
            0x3f900000, // [66] 52: 1.125
            0xc2b04b82, // [67] 53: -88.1474761962890625
            0x3f880000, // [68] 54: 1.0625
            0xc2b02e3e, // [69] 55: -88.0903167724609375
            0x3f880000, // [70] 56: 1.0625
            0xc2b02e3e, // [71] 57: -88.0903167724609375
            0x3f880000, // [72] 58: 1.0625
            0xc2b02e3e, // [73] 59: -88.0903167724609375
            0x3f800000, // [74] 60: 1
            0xc2b00f34, // [75] 61: -88.029693603515625
            0x3f800000, // [76] 62: 1
            0xc2b00f34, // [77] 63: -88.029693603515625
    };

    for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(float); ++d)
            h->dd(cvals[i]);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d)
        h->dd(0x7fffffff);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d)
        h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d)
        h->dd(float2int(beta_));
}

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_vecs_count(alg_kind_t alg_) {
    switch (alg_) {
        case alg_kind::eltwise_relu: return (alpha_ == 0.f) ? 0 : 2;
        case alg_kind::eltwise_elu: return 4;
        case alg_kind::eltwise_tanh: return 5;
        case alg_kind::eltwise_square: return 0;
        case alg_kind::eltwise_abs: return 0;
        case alg_kind::eltwise_sqrt: return 2;
        case alg_kind::eltwise_swish: return 4;
        case alg_kind::eltwise_linear: return 1;
        case alg_kind::eltwise_bounded_relu: return 0;
        case alg_kind::eltwise_soft_relu: return 4;
        case alg_kind::eltwise_logistic: return 4;
        case alg_kind::eltwise_exp: return 3;
        case alg_kind::eltwise_gelu: return 5;
        case alg_kind::eltwise_log: return 5;
        case alg_kind::eltwise_clip: return 0;
        default: assert(!"unsupported eltwise algorithm");
    }

    return 0;
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_body(
        size_t start_idx, size_t end_idx) {
    using namespace alg_kind;
    for (size_t idx = start_idx; idx < end_idx; idx++) {
        switch (alg_) {
            case eltwise_relu:
                if (alpha_ == 0.f)
                    relu_zero_ns_compute_vector(Vmm(idx));
                else
                    relu_compute_vector(Vmm(idx));
                break;
            case eltwise_elu: elu_compute_vector(Vmm(idx)); break;
            case eltwise_tanh: tanh_compute_vector(Vmm(idx)); break;
            case eltwise_square: square_compute_vector(Vmm(idx)); break;
            case eltwise_abs: abs_compute_vector(Vmm(idx)); break;
            case eltwise_sqrt: sqrt_compute_vector(Vmm(idx)); break;
            case eltwise_swish: swish_compute_vector(Vmm(idx)); break;
            case eltwise_linear: linear_compute_vector(Vmm(idx)); break;
            case eltwise_bounded_relu:
                bounded_relu_compute_vector(Vmm(idx));
                break;
            case eltwise_soft_relu: soft_relu_compute_vector(Vmm(idx)); break;
            case eltwise_logistic: logistic_compute_vector(Vmm(idx)); break;
            case eltwise_exp: exp_compute_vector(Vmm(idx)); break;
            case eltwise_gelu: gelu_compute_vector(Vmm(idx)); break;
            case eltwise_log: log_compute_vector(Vmm(idx)); break;
            case eltwise_clip: clip_compute_vector(Vmm(idx)); break;
            default: assert(!"unsupported eltwise algorithm");
        }
        if (scale_ != 1.f) {
            h->uni_vmulps(Vmm(idx), Vmm(idx), h->ptr[p_table]);
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
    using namespace alg_kind;

    h->align(64);
    h->L(l_table);

    if (gen_table) {
        for (size_t d = 0; d < vlen / sizeof(float); ++d)
            h->dd(float2int(scale_));

        switch (alg_) {
            case eltwise_bounded_relu:
            case eltwise_relu: relu_prepare_table(); break;
            case eltwise_elu:
            case eltwise_tanh:
            case eltwise_logistic:
            case eltwise_exp:
            case eltwise_swish:
            case eltwise_gelu: elu_prepare_table(); break;
            case eltwise_soft_relu: soft_relu_prepare_table(); break;
            case eltwise_abs: abs_prepare_table(); break;
            case eltwise_clip:
            case eltwise_linear: linear_prepare_table(); break;
            case eltwise_log: log_prepare_table(); break;
            case eltwise_sqrt:
            case eltwise_square: break;
            default: assert(!"unsupported eltwise algorithm");
        }
    }
}

template struct jit_uni_eltwise_injector_f32<avx512_common>;
template struct jit_uni_eltwise_injector_f32<avx512_core>;
template struct jit_uni_eltwise_injector_f32<avx2>;
template struct jit_uni_eltwise_injector_f32<sse41>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
