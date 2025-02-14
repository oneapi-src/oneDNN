/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include "gpu/intel/jit/eltwise_injector.hpp"
#include "common/impl_registration.hpp"
#include "gpu/intel/jit/codegen/ngen_helpers.hpp"

#include <limits>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

using namespace ngen;

template <gpu_gen_t hw>
int eltwise_injector_f32_t<hw>::min_scratch_regs() {
    using namespace alg_kind;
    if (is_fwd_) {
        switch ((int)alg_) {
            case eltwise_elu:
            case eltwise_elu_use_dst_for_bwd: return 1;
            case eltwise_exp:
            case eltwise_exp_use_dst_for_bwd: return 0;
            case eltwise_gelu_erf: return 4;
            case eltwise_hardsigmoid: return 0;
            case eltwise_hardswish: return 1;
            case eltwise_log: return 0;
            case eltwise_mish: return 4;
            case eltwise_pow: return 1;
            case eltwise_relu:
            case eltwise_relu_use_dst_for_bwd: return 1;
            case eltwise_abs: return 0;
            case eltwise_soft_relu: return 1;
            case eltwise_sqrt:
            case eltwise_sqrt_use_dst_for_bwd: return 0;
            case eltwise_square: return 0;
            case eltwise_swish: return 1;
            case eltwise_tanh:
            case eltwise_tanh_use_dst_for_bwd: return 2;
            case eltwise_round: return 0;
            case eltwise_linear: return 0;
            case eltwise_clip:
            case eltwise_clip_v2:
            case eltwise_clip_v2_use_dst_for_bwd: return 0;
            case eltwise_gelu_tanh: return 2;
            case eltwise_logistic:
            case eltwise_logistic_use_dst_for_bwd: return 0;
            case eltwise_stochastic_round: return 6;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu: return 1;
            case eltwise_abs: return 1;
            case eltwise_square: return 0;
            case eltwise_linear: return 0;
            case eltwise_clip: return 1;
            case eltwise_gelu_tanh: return 2;
            default: assert(!"unsupported eltwise algorithm");
        }
    }
    return 0;
}

template <gpu_gen_t hw>
int eltwise_injector_f32_t<hw>::preferred_scratch_regs() {
    using namespace alg_kind;
    if (is_fwd_) {
        switch (alg_) {
            case eltwise_elu:
            case eltwise_elu_use_dst_for_bwd: return 8;
            case eltwise_gelu_erf: return 8;
            case eltwise_hardswish: return 8;
            case eltwise_mish: return 8;
            case eltwise_relu:
            case eltwise_relu_use_dst_for_bwd: return (alpha_ == 0.f) ? 1 : 8;
            case eltwise_tanh: return 8;
            case eltwise_gelu_tanh: return 8;
            case eltwise_soft_relu: return 8;
            case eltwise_swish: return 8;
            default: break;
        }
    } else {
        switch (alg_) {
            case eltwise_gelu_tanh: return 8;
            default: break;
        }
    }
    return min_scratch_regs();
}

template <gpu_gen_t hw>
int eltwise_injector_f32_t<hw>::max_batch_size() {
    using namespace alg_kind;
    auto ss = scratch_.getLen();

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu:
            case eltwise_relu_use_dst_for_bwd:
                if (alpha_ == 0.)
                    break;
                else
                    return ss;
            case eltwise_elu:
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_hardswish:
            case eltwise_pow:
            case eltwise_soft_relu:
            case eltwise_swish: return ss;
            case eltwise_tanh:
            case eltwise_mish:
            case eltwise_gelu_erf: return ss / min_scratch_regs();
            case eltwise_gelu_tanh: return ss & ~1;
            default: break;
        }
    } else {
        switch (alg_) {
            case eltwise_gelu_tanh: return ss / 2;
            default: break;
        }
    }

    return 128;
}

template <gpu_gen_t hw>
int eltwise_injector_f32_t<hw>::phase_count(alg_kind_t alg) {
    using namespace alg_kind;

    if (is_fwd_) {
        switch (alg) {
            case eltwise_elu:
            case eltwise_elu_use_dst_for_bwd: return 5;
            case eltwise_exp:
            case eltwise_exp_use_dst_for_bwd: return 2;
            case eltwise_gelu_erf: return 25;
            case eltwise_hardsigmoid: return 4;
            case eltwise_hardswish: return 5;
            case eltwise_log: return 2;
            case eltwise_mish:
                return phase_count(alg_kind::eltwise_soft_relu)
                        + phase_count(alg_kind::eltwise_tanh) + 1;
            case eltwise_pow: return 6;
            case eltwise_relu:
            case eltwise_relu_use_dst_for_bwd: return (alpha_ == 0) ? 1 : 2;
            case eltwise_soft_relu: return 10;
            case eltwise_swish: return 5;
            case eltwise_tanh:
            case eltwise_tanh_use_dst_for_bwd:
                return (use_tanh_compat()) ? 9 : 6;
            case eltwise_linear: return (beta_ == 0) ? 1 : 2;
            case eltwise_clip:
            case eltwise_clip_v2:
            case eltwise_clip_v2_use_dst_for_bwd: return 2;
            case eltwise_gelu_tanh: return 8;
            case eltwise_logistic:
            case eltwise_logistic_use_dst_for_bwd: return 4;
            default: break;
        }
    } else {
        switch (alg) {
            case eltwise_abs: return 2;
            case eltwise_clip: return 4;
            case eltwise_gelu_tanh: return 14;
            default: break;
        }
    }

    return 1;
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::relu_zero_ns_prepare_fwd() {
    h->mov(1, scratch_[0].f(0), 0.f);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::relu_zero_ns_compute_fwd(
        int simd, const ngen::GRF &r) {
    /* use csel instead of max to propagate NaNs*/
    h->csel(simd | le | f0[0], r, scratch_[0].f(0), r, r);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::relu_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    auto temp = scratch_[off].f();
    switch (phase) {
        case 0: h->mul(simd, temp, r, alpha_); break;
        case 1: h->csel(simd | le | f0[0], r, temp, r, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::abs_compute_fwd(int simd, const ngen::GRF &r) {
    h->mov(simd, r, abs(r));
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::soft_relu_compute_fwd_inner(int simd,
        const ngen::GRF &input, const ngen::GRF &temp, const ngen::GRF &dest,
        int phase, int off, float alpha) {
    const float exp_overflow_bound = 88.72283172607421875f;
    const float log2e = 1.44269502162933349609375f;
    const float reciproc_log2e = 1.f / log2e; // 1 / log_2(e)
    switch (phase) {
        case 0: h->mul(simd, temp, input, alpha); break;
        case 1: h->add(simd, dest, input, -exp_overflow_bound); break;
        case 2: h->csel(simd | le | f0[0], dest, dest, temp, dest); break;
        case 3: h->mul(simd, temp, temp, log2e); break;
        case 4: h->exp(simd, temp, temp); break;
        case 5: h->add(simd, temp, temp, 1.f); break;
        case 6: h->log(simd, temp, temp); break;
        case 7: h->mul(simd, temp, temp, reciproc_log2e); break;
        case 8: h->csel(simd | le | f0[0], temp, temp, dest, dest); break;
        case 9: h->mul(simd, dest, temp, 1.f / alpha); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::soft_relu_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    auto temp = scratch_[off].f();
    soft_relu_compute_fwd_inner(simd, r, temp, r, phase, off, alpha_);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::sqrt_compute_fwd(
        int simd, const ngen::GRF &r) {
    h->sqt(simd, r, r);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::square_compute_fwd(
        int simd, const ngen::GRF &r) {
    h->mul(simd, r, r, r);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::tanh_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off, int batch) {
    const float log2e = 1.44269502162933349609375f; // log_2(e)
    auto one_half = scratch_[0].f(7);
    auto a = scratch_[off + batch].f();
    switch (phase) {
        case 0: h->mul(simd, a, abs(r), 2.f * log2e); break;
        case 1: h->exp(simd, a, a); break;
        case 2: h->mad(simd, a, one_half, a, one_half); break;
        case 3: h->inv(simd, a, a); break;
        case 4: h->add(simd, a, -a, 1.f); break;
        case 5: h->csel(simd | ge | f0[0], r, a, -a, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::tanh_compute_fwd_compat(
        int simd, const ngen::GRF &r, int phase, int off, int batch) {
    // This approximation of tanh(x) does not use the math.exp instruction
    // that seems to be faulty on DG2-128; the exact formula is as follows:
    // R = max(min(0.0519867*x*((x^2 + k)^2 + l)/((x^2 + m)^2 + n), 1), -1)
    // Both absolute and relative errors are <7*10^-5 \forall x \in \mathbb R
    auto k = scratch_[0].f(4);
    auto l = scratch_[0].f(5);
    auto m = scratch_[0].f(6);
    auto n = scratch_[0].f(7);
    auto a = scratch_[off + batch].f();
    switch (phase) {
        case 0: h->mad(simd, a, m, r, r); break;
        case 1: h->mad(simd, a, n, a, a); break;
        case 2: h->inv(simd, a, a); break;
        case 3: h->mul(simd, a, a, r); break;
        case 4: h->mad(simd, r, k, r, r); break;
        case 5: h->mad(simd, r, l, r, r); break;
        case 6: h->mul(simd, r, r, 0.0519867f); break; // 0.051986694f
        case 7: h->mul(simd | sat, r, r, abs(a)); break;
        case 8: h->csel(simd | ge | f0[0], r, r, -r, a); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::round_compute_fwd(
        int simd, const ngen::GRF &r) {
    h->rnde(simd, r, r);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::sround_compute_fwd(int simd,
        const ngen::GRF &r, int phase, const ngen::Subregister &seed,
        const ngen::DataType dst_dt, int off) {
    // 2 regs for bias.
    auto bias = scratch_[0].ud();
    auto u_r = r[0].ud()(8, 8, 1);
    auto u_f = r[0].f()(8, 8, 1);

    // Initialize indices in counter.
    int base_idx = off * simd;
    h->template mov<uint16_t>(8, bias.uw(0)(1), Immediate::uv(0x76543210));
    h->template mov<uint16_t>(8, bias.uw(8)(1), Immediate::uv(0xfedcba98));
    auto imm = Immediate::ud(base_idx);
    if (hw >= gpu_xe_hpc)
        h->add(16, bias.ud(), bias.uw(), imm);
    else {
        auto extra = scratch_[1].ud();
        h->add(8, extra.ud(0)(1), bias.uw(8)(8, 8, 1), imm);
        h->add(8, bias.ud(0)(1), bias.uw(0)(8, 8, 1), imm);
    }

    const uint32_t dst_dt_digits = dnnl::impl::types::digits<uint32_t>(
            convert_ngen_type_to_dnnl(dst_dt));
    assert(dst_dt_digits <= 24);

    data_type_t dnnl_t = to_dnnl(to_ir(dst_dt));
    const float f_min = types::min_value<float>(dnnl_t);
    const float max = types::max_value<float>(dnnl_t);
    const float lowest = types::lowest_value<float>(dnnl_t);
    auto bia_scratch = scratch_[4].ud();

    // Mask for preserving inf, NaN.
    // u_r & 0x7F800000 != 0x7F800000 implies (~u_r) & 0x7F800000 != 0
    h->and_(simd | h->nz | f0[0], h->null.ud(), ~u_r, 0x7F800000);

    const int truncation_mask = (0xffffffff << (24 - dst_dt_digits));

    philox_4x32(simd, seed, bias);

    if (getBytes(dst_dt) == 2) {
        h->mov(simd, bia_scratch.ud(0), bias.ub(0)(simd, simd, 1));
    } else {
        h->mov(simd, bia_scratch.ud(0), bias.uw(0)(simd, simd, 1));
    }

    h->and_(simd, bia_scratch, bia_scratch, ~truncation_mask);

    h->add(simd | f0[0], u_r, u_r, bia_scratch);
    h->and_(simd | f0[0], u_r, u_r, truncation_mask);

    // Enforce dst data type range.
    h->max_(simd, u_f, u_f, lowest);
    h->min_(simd, u_f, u_f, max);
    // Enforce minimum precision.
    h->cmp(simd | lt | f0[0], abs(u_f), f_min);
    h->mov(simd | f0[0], u_r, 0);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::philox_4x32(
        int simd, const ngen::Subregister &seed, const ngen::GRF &bias) {
    auto sround_seed = seed;
    auto ctr = bias.ud(0);

    auto key = scratch_[2].ud();
    auto ctr_mul = scratch_[3].ud();
    auto offs = scratch_[4].ud();
    auto off_inc = scratch_[5].uw();
    auto addr = h->indirect[h->a0].ud(0)(0, 1, 0);

    // Compute key.
    if (hw >= gpu_xe_hpc)
        h->mov(4, key.uq(0)(4, 4, 1), uint64_t(0xBB67AE859E3779B9uLL));
    else {
        h->mov(4, key.ud(0)(4, 4, 2), uint32_t(0x9E3779B9u));
        h->mov(4, key.ud(1)(4, 4, 2), uint32_t(0xBB67AE85u));
    }

    h->template mov<uint16_t>(4, offs.uw(16)(1), Immediate::uv(0x00009988));
    h->template mul<uint32_t>(
            4, offs.ud(8)(1), key.ud(0)(4, 4, 1), offs.uw(16)(4, 4, 1));
    h->add(4, offs.ud(8)(1), offs.ud(8)(4, 4, 1), sround_seed);

    h->template mov<uint16_t>(8, offs.uw(8)(1), Immediate::uv(0x77665544));
    h->template mov<uint16_t>(8, offs.uw(0)(1), Immediate::uv(0x33221100));
    h->template mul<uint32_t>(
            8, key.ud(8)(1), key.ud(0)(8, 8, 1), offs.uw(8)(8, 8, 1));
    h->template mul<uint32_t>(
            8, key.ud(0)(1), key.ud(0)(8, 8, 1), offs.uw(0)(8, 8, 1));
    h->add(16, key, key, sround_seed);
    // Compute ctr_mul.
    h->mov(4, ctr_mul.ud(2)(4), 0xCD9E8D57);
    h->mov(4, ctr_mul.ud(0)(4), 0xD2511F53);
    auto ctr_base_sub = offs.uw(8)(8, 8, 1);
    h->mov(8, ctr_base_sub, (ctr.getBase() * GRF::bytes(hw)) + ctr.getOffset());

    // Prepare first iter idx swizzle
    h->template mov<uint16_t>(8, off_inc.uw(0)(1), Immediate::uv(0x56741230));
    h->template mul<uint16_t>(8, off_inc.uw(0)(1), off_inc.uw(0)(8, 8, 1), 4);

    //as_uint4(convert_ulong2(ctr.s31) * mul) ^ (uint4)(ctr.s20 ^ key, 0, 0).s3120
    auto philox_round = [&](ngen::Subregister &ctr, ngen::GRF &ctr_mul,
                                ngen::GRF &key, int idx) {
        // TODO: what if offsets in different operands can differ?

        // Apply idx swizzle.
        h->add(8, h->a0, ctr_base_sub, off_inc);
        h->template movi<uint32_t>(8, ctr.ud(0)(1), addr);
        h->add(8, h->a0, h->a0, 32);
        h->template movi<uint32_t>(8, ctr.ud(8)(1), addr);

        // KEY packed with mul in ctr_mul, key in odd indices, ctr_mul in even.
        // Swizzle Key to avoid double swizzle as in ocl ((uint4)(ctr.s20 ^ key, 0, 0).s3120).
        h->mov(4, ctr_mul.ud(1)(4), key.ud(idx + 1)(0, 1, 0));
        h->mov(4, ctr_mul.ud(3)(4), key.ud(idx)(0, 1, 0));
        // END SWIZZLE CTR_MUL

        // xor ctr.s02 ^ key.s10
        h->xor_(8, ctr_mul.ud(1)(2), ctr_mul.ud(1)(8, 4, 2),
                ctr.ud(1)(8, 4, 2));

        // EMULATE QW <- DW X DW
        // mul ctr.s31 * ctr_mul
        auto ctrLo = ctr.ud(0)(8, 4, 2);
        auto ctrHi = ctr.ud(1)(8, 4, 2);

        const auto grf_size = ngen::GRF::bytes(hw);
        const int esize = grf_size / 8; // 8 = 2 * dword bytes
        const int steps = utils::div_up(8, esize);

        auto acc = h->acc0.retype(DataType::ud);
        for (int i = 0, off = 0; i < steps; ++i, off += 2 * esize)
            h->mul(esize, acc[i](2), ctr.ud(off)(8, 4, 2),
                    ctr_mul.uw(off)(8, 2, 4));
        h->mach(8, ctrLo, ctr.ud(0)(8, 4, 2), ctr_mul.ud(0)(8, 4, 2));
        h->mov(8, ctrHi, ctrLo);
        for (int i = 0, off = 0; i < steps; ++i, off += 2 * esize)
            h->mov(esize, ctr.ud(off)(2), acc[i](2));

        // xor results
        h->xor_(8, ctr.ud(1)(2), ctr.ud(1)(8, 4, 2), ctr_mul.ud(1)(8, 4, 2));

        // Set idx swizzle for subsequent iterations.
        if (idx == 0) {
            h->template mov<uint16_t>(
                    8, off_inc.uw(0)(1), Immediate::uv(0x65472103));
            h->template mul<uint16_t>(
                    8, off_inc.uw(0)(1), off_inc.uw(0)(8, 8, 1), 4);
        }
    };
    philox_round(ctr, ctr_mul, key, 0);
    philox_round(ctr, ctr_mul, key, 2);
    philox_round(ctr, ctr_mul, key, 4);
    philox_round(ctr, ctr_mul, key, 6);
    philox_round(ctr, ctr_mul, key, 8);
    philox_round(ctr, ctr_mul, key, 10);
    philox_round(ctr, ctr_mul, key, 12);
    philox_round(ctr, ctr_mul, key, 14);
    philox_round(ctr, ctr_mul, offs, 8);
    philox_round(ctr, ctr_mul, offs, 10);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::swish_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    const float log2e = 1.442695f; // log_2(e)
    auto temp = scratch_[off].f();
    switch (phase) {
        case 0: h->mul(simd, temp, r, -1.f * log2e * alpha_); break;
        case 1: h->exp(simd, temp, temp); break;
        case 2: h->add(simd, temp, temp, 1.f); break;
        case 3: h->inv(simd, temp, temp); break;
        case 4: h->mul(simd, r, r, temp); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::linear_compute_fwd(
        int simd, const ngen::GRF &r, int phase) {
    switch (phase) {
        case 0: h->mul(simd, r, r, alpha_); break;
        case 1: h->add(simd, r, r, beta_); break; /* skipped if beta_ = 0 */
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::clip_compute_fwd(
        int simd, const ngen::GRF &r, int phase, float alpha, float beta) {
    switch (phase) {
        case 0: h->max_(simd, r, r, alpha); break;
        case 1: h->min_(simd, r, r, beta); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::gelu_tanh_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {

    const float k = 0.044715f;
    const float sqrt_2_over_pi = 0.7978845f; // sqrt(2/pi)
    const float log2e = 1.442695f; // log_2(e)

    int msimd = simd;
    if (hw == gpu_xe_hp)
        msimd = 16; // workaround for intermittent hang with DPAS+EM

    auto a = scratch_[off].f();
    switch (phase) {
        case 0: h->mul(simd, a, r, r); break;
        case 1: h->mul(simd, a, a, k); break;
        case 2: h->mad(simd, a, r, a, r); break;
        case 3: h->mul(simd, a, a, -2 * sqrt_2_over_pi * log2e); break;
        case 4: h->exp(msimd, a, a); break;
        case 5: h->add(simd, a, a, 1.0f); break;
        case 6: h->inv(msimd, a, a); break;
        case 7: h->mul(simd, r, a, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::logistic_compute_fwd(
        int simd, const ngen::GRF &r, int phase) {
    const float log2e = 1.442695f; // log_2(e)
    switch (phase) {
        case 0: h->mul(simd, r, r, -1.f * log2e); break;
        case 1: h->exp(simd, r, r); break;
        case 2: h->add(simd, r, r, 1.f); break;
        case 3: h->inv(simd, r, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::relu_prepare_bwd() {
    auto neg_slope = scratch_[0].f(0);
    auto pos_slope = scratch_[0].f(4);
    h->mov(1, neg_slope, alpha_);
    h->mov(1, pos_slope, 1.f);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::relu_compute_bwd(
        int simd, const ngen::GRF &r) {
    auto neg_slope = scratch_[0].f(0);
    auto pos_slope = scratch_[0].f(4);
    h->csel(simd | le | f0[0], r, neg_slope, pos_slope, r);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::abs_prepare_bwd() {
    auto neg_one = scratch_[0].f(0);
    auto pos_one = scratch_[0].f(4);
    h->mov(1, neg_one, -1.f);
    h->mov(1, pos_one, 1.f);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::clip_prepare_bwd() {
    auto pos_inf_imm = Immediate(std::numeric_limits<float>::infinity());
    auto zero = scratch_[0].f(0);
    auto one = scratch_[0].f(1);
    auto pos_inf = scratch_[0].f(2);
    h->mov(1, zero, 0.f);
    h->mov(1, one, 1.f);
    h->mov(1, pos_inf, pos_inf_imm);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::tanh_prepare_fwd() {
    auto one_half = scratch_[0].f(7);
    h->mov(1, one_half, 0.5f);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::tanh_prepare_fwd_compat() {
    auto k = scratch_[0].f(4);
    auto l = scratch_[0].f(5);
    auto m = scratch_[0].f(6);
    auto n = scratch_[0].f(7);
    h->mov(1, k, 77.0954f); //  77.095392909578f
    h->mov(1, l, -4435.55f); // -4435.54623970169f
    h->mov(1, m, 17.06396f); //  17.06396485f
    h->mov(1, n, -212.7724f); // -212.772646402036f
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::abs_compute_bwd(
        int simd, const ngen::GRF &r, int phase) {
    auto neg_one = scratch_[0].f(0);
    auto pos_one = scratch_[0].f(4);
    switch (phase) {
        case 0: h->csel(simd | lt | f0[0], r, neg_one, r, r); break;
        case 1: h->csel(simd | gt | f0[0], r, pos_one, r, r); break;
        default: break;
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::square_compute_bwd(
        int simd, const ngen::GRF &r) {
    h->add(simd, r, r, r);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::linear_compute_bwd(
        int simd, const ngen::GRF &r) {
    h->mov(simd, r, alpha_);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::clip_compute_bwd(
        int simd, const ngen::GRF &r, int phase, float alpha, float beta) {
    auto zero = scratch_[0].f(0);
    auto one = scratch_[0].f(1);
    auto pos_inf = scratch_[0].f(2);
    switch (phase) {
        // r[i] = r[i] - alpha
        case 0: h->add(simd, r, r, -alpha); break;
        // r[i] <= 0 => r[i] = infinity
        case 1: h->csel(simd | le | f0[0], r, pos_inf, r, r); break;
        // r[i] = (r[i] + alpha) - beta
        case 2: h->add(simd, r, r, alpha - beta); break;
        // r[i] = (r[i] <= 0 ? 1 : 0)
        case 3: h->csel(simd | le | f0[0], r, one, zero, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::gelu_tanh_compute_bwd(
        int simd, const ngen::GRF &r, int phase, int off, int batch) {

    const float k = 0.044715f;
    const float sqrt_2_over_pi = 0.7978845f; // sqrt(2/pi)
    const float log2e = 1.442695f; // log_2(e)

    int msimd = simd;
    if (hw == gpu_xe_hp) msimd = 16;

    auto a = scratch_[off].f();
    auto b = scratch_[off + batch].f();
    switch (phase) {
        case 0: h->mul(simd, a, r, r); break;
        case 1: h->mul(simd, b, a, 3.0f * k); break;
        case 2: h->mul(simd, a, a, k); break;
        case 3: h->mad(simd, a, r, a, r); break;
        case 4: h->mad(simd, b, r, b, r); break;
        case 5: h->mul(simd, a, a, -2 * sqrt_2_over_pi * log2e); break;
        case 6: h->mul(simd, b, b, 2 * sqrt_2_over_pi); break;
        case 7: h->exp(msimd, a, a); break;
        case 8: h->add(simd, r, a, 1.0f); break;
        case 9: h->inv(msimd, r, r); break;
        case 10: h->mul(simd, a, a, r); break;
        case 11: h->mul(simd, a, a, b); break;
        case 12: h->add(simd, a, a, 1.0f); break;
        case 13: h->mul(simd, r, r, a); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::elu_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    auto temp = scratch_[off].f();
    const float log2e = 1.442695f; // log_2(e)
    switch (phase) {
        case 0: h->mul(simd, temp, r, log2e); break;
        case 1: h->exp(simd, temp, temp); break;
        case 2: h->add(simd, temp, temp, -1.f); break;
        case 3: h->mul(simd, temp, temp, alpha_); break;
        case 4: h->csel(simd | le | f0[0], r, temp, r, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::exp_compute_fwd(
        int simd, const ngen::GRF &r, int phase) {
    const float log2e = 1.442695f; // log_2(e)
    switch (phase) {
        case 0: h->mul(simd, r, r, log2e); break;
        case 1: h->exp(simd, r, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::gelu_erf_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off, int batch) {
    auto temp = scratch_[off].f();
    auto at_accum = scratch_[off + batch].f();
    auto tpow = scratch_[off + 2 * batch].f();
    auto temp2 = scratch_[off + 3 * batch].f();
    const float log2e = 1.442695f; // log_2(e)
    const float reciproc_sqrt_2 = 0.707106769084930419921875f; // 1/sqrt(2)
    const float p = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    switch (phase) {
        case 0: h->mul(simd, temp, abs(r), reciproc_sqrt_2); break;
        case 1: h->mul(simd, temp, temp, p); break;
        case 2: h->add(simd, temp, temp, 1.f); break;
        case 3: h->inv(simd, temp, temp); break;
        case 4: h->mul(simd, at_accum, temp, a1); break;
        case 5: h->mul(simd, tpow, temp, temp); break;
        case 6: h->mul(simd, temp2, tpow, a2); break;
        case 7: h->add(simd, at_accum, temp2, at_accum); break;
        case 8: h->mul(simd, tpow, tpow, temp); break;
        case 9: h->mul(simd, temp2, tpow, a3); break;
        case 10: h->add(simd, at_accum, temp2, at_accum); break;
        case 11: h->mul(simd, tpow, tpow, temp); break;
        case 12: h->mul(simd, temp2, tpow, a4); break;
        case 13: h->add(simd, at_accum, temp2, at_accum); break;
        case 14: h->mul(simd, tpow, tpow, temp); break;
        case 15: h->mul(simd, temp2, tpow, a5); break;
        case 16: h->add(simd, at_accum, temp2, at_accum); break;
        case 17: h->mul(simd, temp, r, r); break;
        case 18: h->mul(simd, temp, temp, -log2e * 0.5f); break;
        case 19: h->exp(simd, temp, temp); break;
        case 20: h->mul(simd, temp, temp, at_accum); break;
        case 21: h->mul(simd, temp, temp, r); break;
        case 22: h->mul(simd, temp, temp, 0.5f); break;
        case 23: h->add(simd, temp2, r, -temp); break;
        case 24: h->csel(simd | le | f0[0], r, temp, temp2, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::hardsigmoid_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    switch (phase) {
        case 0: h->mul(simd, r, r, alpha_); break;
        case 1: h->add(simd, r, r, beta_); break;
        case 2: h->min_(simd, r, r, 1.f); break;
        case 3: h->max_(simd, r, r, 0.f); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::hardswish_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    auto temp = scratch_[off].f();
    switch (phase) {
        case 0: h->mul(simd, temp, r, alpha_); break;
        case 1: h->add(simd, temp, temp, beta_); break;
        case 2: h->min_(simd, temp, temp, 1.f); break;
        case 3: h->max_(simd, temp, temp, 0.f); break;
        case 4: h->mul(simd, r, r, temp); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::log_compute_fwd(
        int simd, const ngen::GRF &r, int phase) {
    const float reciproc_log2e = 1.f / 1.442695f; // 1 / log_2(e)
    switch (phase) {
        case 0: h->log(simd, r, r); break;
        case 1: h->mul(simd, r, r, reciproc_log2e); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::mish_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off, int batch) {
    auto temp = scratch_[off + batch].f();
    auto temp2 = scratch_[off + 2 * batch].f();
    const int srelu_phases = phase_count(alg_kind::eltwise_soft_relu);
    const int tanh_phases = phase_count(alg_kind::eltwise_tanh);
    // note tanh_compute_fwd_* clobbers scratch_[off] and scratch_[off + batch]
    if (phase < srelu_phases)
        soft_relu_compute_fwd_inner(simd, r, temp, temp2, phase, off, 1.f);
    if (phase >= srelu_phases && phase < srelu_phases + tanh_phases) {
        if (use_tanh_compat())
            tanh_compute_fwd_compat(
                    simd, temp2, phase - srelu_phases, off, batch);
        else
            tanh_compute_fwd(simd, temp2, phase - srelu_phases, off, batch);
    }
    if (phase == srelu_phases + tanh_phases) h->mul(simd, r, r, temp2);
    if (phase > srelu_phases + tanh_phases) assert(!"invalid phase");
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::pow_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    auto temp = scratch_[off].f();
    switch (phase) {
        case 0:
            if (float((long long int)beta_) == beta_) {
                h->mov(simd, temp, abs(r));
            } else {
                h->mov(simd, temp, r);
            }
            break;
        case 1: h->log(simd, temp, temp); break;
        case 2: h->mul(simd, temp, temp, beta_); break;
        case 3: h->exp(simd, temp, temp); break;
        case 4:
            if (((long long int)beta_) & 0x1)
                h->csel(simd | lt | f0[0], temp, -temp, temp, r);
            break;
        case 5: h->mul(simd, r, temp, alpha_); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::compute(const int *grfs, int ngrf,
        const int seed, const int off, const ngen::DataType dt) {
    using namespace alg_kind;

    auto bmax = max_batch_size();
    auto phases = phase_count(alg_);

    for (int idx0 = 0; idx0 < ngrf; idx0 += bmax) {
        auto batch = nstl::min(ngrf - idx0, bmax);

        for (int phase = 0; phase < phases; phase++) {
            for (int ii = 0, nreg = 0; ii < batch; ii += nreg) {
                auto grf0 = grfs[idx0 + ii];
                auto base = GRF(grf0).f();

                nreg = 1;
                if (ii + 1 < batch)
                    if (grf0 + 1 == grfs[idx0 + ii + 1]) nreg = 2;

                int simd = nreg * GRF::bytes(hw) / sizeof(float);

                if (is_fwd_) {
                    switch ((int)alg_) {
                        case eltwise_elu:
                        case eltwise_elu_use_dst_for_bwd:
                            elu_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_exp:
                        case eltwise_exp_use_dst_for_bwd:
                            exp_compute_fwd(simd, base, phase);
                            break;
                        case eltwise_gelu_erf:
                            gelu_erf_compute_fwd(simd, base, phase, ii, batch);
                            break;
                        case eltwise_hardsigmoid:
                            hardsigmoid_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_hardswish:
                            hardswish_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_log:
                            log_compute_fwd(simd, base, phase);
                            break;
                        case eltwise_mish:
                            mish_compute_fwd(simd, base, phase, ii, batch);
                            break;
                        case eltwise_pow:
                            pow_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_relu:
                        case eltwise_relu_use_dst_for_bwd:
                            if (alpha_ == 0.f)
                                relu_zero_ns_compute_fwd(simd, base);
                            else
                                relu_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_abs: abs_compute_fwd(simd, base); break;
                        case eltwise_soft_relu:
                            soft_relu_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_sqrt:
                        case eltwise_sqrt_use_dst_for_bwd:
                            sqrt_compute_fwd(simd, base);
                            break;
                        case eltwise_square:
                            square_compute_fwd(simd, base);
                            break;
                        case eltwise_tanh:
                        case eltwise_tanh_use_dst_for_bwd:
                            if (use_tanh_compat())
                                tanh_compute_fwd_compat(
                                        simd, base, phase, ii, batch);
                            else
                                tanh_compute_fwd(simd, base, phase, ii, batch);
                            break;
                        case eltwise_round:
                            round_compute_fwd(simd, base);
                            break;
                        case eltwise_swish:
                            swish_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_linear:
                            linear_compute_fwd(simd, base, phase);
                            break;
                        case eltwise_clip:
                        case eltwise_clip_v2:
                        case eltwise_clip_v2_use_dst_for_bwd:
                            clip_compute_fwd(simd, base, phase, alpha_, beta_);
                            break;
                        case eltwise_gelu_tanh:
                            gelu_tanh_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_logistic:
                        case eltwise_logistic_use_dst_for_bwd:
                            logistic_compute_fwd(simd, base, phase);
                            break;
                        case eltwise_stochastic_round:
                            sround_compute_fwd(simd, base, phase,
                                    GRF(seed).ud(off), dt, ii);
                            break;
                        default: assert(!"unsupported eltwise algorithm");
                    }
                } else {
                    switch (alg_) {
                        case eltwise_relu: relu_compute_bwd(simd, base); break;
                        case eltwise_abs:
                            abs_compute_bwd(simd, base, phase);
                            break;
                        case eltwise_square:
                            square_compute_bwd(simd, base);
                            break;
                        case eltwise_linear:
                            linear_compute_bwd(simd, base);
                            break;
                        case eltwise_clip:
                            clip_compute_bwd(simd, base, phase, alpha_, beta_);
                            break;
                        case eltwise_gelu_tanh:
                            gelu_tanh_compute_bwd(simd, base, phase, ii, batch);
                            break;
                        default: assert(!"unsupported eltwise algorithm");
                    }
                }
                // Apply scale.
                if (phase == phases - 1 && scale_ != 1.f) {
                    h->mul(simd, base, base, scale_);
                }
            }
        }
    }
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::compute(const ngen::GRFRange &regs,
        const int seed, const int off, const ngen::DataType dt) {
    int grfs[ngen::GRF::maxRegs()];

    for (int i = 0; i < regs.getLen(); i++)
        grfs[i] = regs.getBase() + i;

    compute(grfs, regs.getLen(), seed, off, dt);
}

template <gpu_gen_t hw>
void eltwise_injector_f32_t<hw>::prepare() {
    using namespace alg_kind;

    assert(scratch_.getLen() >= min_scratch_regs());

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu:
            case eltwise_relu_use_dst_for_bwd:
                if (alpha_ == 0.f) relu_zero_ns_prepare_fwd();
                break;
            case eltwise_mish:
            case eltwise_tanh:
                if (use_tanh_compat())
                    tanh_prepare_fwd_compat();
                else
                    tanh_prepare_fwd();
                break;
            default: break;
        }
    } else {
        switch (alg_) {
            case eltwise_relu: relu_prepare_bwd(); break;
            case eltwise_abs: abs_prepare_bwd(); break;
            case eltwise_clip: clip_prepare_bwd(); break;
            default: break;
        }
    }
}

REG_GEN9_ISA(template struct eltwise_injector_f32_t<gpu_gen9>);
REG_GEN11_ISA(template struct eltwise_injector_f32_t<gpu_gen11>);
REG_XELP_ISA(template struct eltwise_injector_f32_t<gpu_xe_lp>);
REG_XEHP_ISA(template struct eltwise_injector_f32_t<gpu_xe_hp>);
REG_XEHPG_ISA(template struct eltwise_injector_f32_t<gpu_xe_hpg>);
REG_XEHPC_ISA(template struct eltwise_injector_f32_t<gpu_xe_hpc>);
REG_XE2_ISA(template struct eltwise_injector_f32_t<gpu_xe2>);
REG_XE3_ISA(template struct eltwise_injector_f32_t<gpu_xe3>);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
