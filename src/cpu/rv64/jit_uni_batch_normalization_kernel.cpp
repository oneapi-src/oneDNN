/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

#include "cpu/rv64/jit_uni_batch_normalization_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace {

template <data_type_t data_type, bool per_elem_params, bool with_relu>
void dispatch_jit_batch_normalization_fwd(
        const jit_uni_batch_normalization_fwd_kernel_t::call_params_t *p) {
    static const jit_uni_batch_normalization_fwd_kernel_t kernel(
            data_type, per_elem_params, with_relu);
    kernel(p);
}

// The per-combination kernel lives in a function-local static, so every
// (dtype, per_elem_params, with_relu) triple needs its own instantiation; these
// helpers turn the runtime flags into those template arguments.
template <data_type_t data_type>
void dispatch_jit_batch_normalization_fwd_dt(
        const jit_uni_batch_normalization_fwd_kernel_t::call_params_t *p,
        bool per_elem_params, bool with_relu) {
    if (per_elem_params) {
        if (with_relu)
            dispatch_jit_batch_normalization_fwd<data_type, true, true>(p);
        else
            dispatch_jit_batch_normalization_fwd<data_type, true, false>(p);
    } else {
        if (with_relu)
            dispatch_jit_batch_normalization_fwd<data_type, false, true>(p);
        else
            dispatch_jit_batch_normalization_fwd<data_type, false, false>(p);
    }
}

template <data_type_t data_type, bool per_elem_params>
void dispatch_jit_batch_normalization_bwd_reduce(
        const jit_uni_batch_normalization_bwd_reduce_kernel_t::call_params_t
                *p) {
    static const jit_uni_batch_normalization_bwd_reduce_kernel_t kernel(
            data_type, per_elem_params);
    kernel(p);
}

template <data_type_t data_type>
void dispatch_jit_batch_normalization_bwd_reduce_dt(
        const jit_uni_batch_normalization_bwd_reduce_kernel_t::call_params_t *p,
        bool per_elem_params) {
    if (per_elem_params)
        dispatch_jit_batch_normalization_bwd_reduce<data_type, true>(p);
    else
        dispatch_jit_batch_normalization_bwd_reduce<data_type, false>(p);
}

template <data_type_t data_type, bool per_elem_params>
void dispatch_jit_batch_normalization_bwd_apply(
        const jit_uni_batch_normalization_bwd_apply_kernel_t::call_params_t
                *p) {
    static const jit_uni_batch_normalization_bwd_apply_kernel_t kernel(
            data_type, per_elem_params);
    kernel(p);
}

template <data_type_t data_type>
void dispatch_jit_batch_normalization_bwd_apply_dt(
        const jit_uni_batch_normalization_bwd_apply_kernel_t::call_params_t *p,
        bool per_elem_params) {
    if (per_elem_params)
        dispatch_jit_batch_normalization_bwd_apply<data_type, true>(p);
    else
        dispatch_jit_batch_normalization_bwd_apply<data_type, false>(p);
}

template <data_type_t data_type, bool calculate_variance>
const jit_uni_batch_normalization_fwd_stat_kernel_t &
get_jit_batch_normalization_fwd_stat_kernel() {
    static const jit_uni_batch_normalization_fwd_stat_kernel_t kernel(
            data_type, calculate_variance);
    return kernel;
}

template <data_type_t data_type>
const jit_uni_batch_normalization_fwd_stat_kernel_t &
get_jit_batch_normalization_fwd_stat_kernel_dt(bool calculate_variance) {
    if (calculate_variance)
        return get_jit_batch_normalization_fwd_stat_kernel<data_type, true>();
    return get_jit_batch_normalization_fwd_stat_kernel<data_type, false>();
}

} // namespace

jit_uni_batch_normalization_fwd_stat_kernel_t::
        jit_uni_batch_normalization_fwd_stat_kernel_t(
                data_type_t data_type, bool calculate_variance)
    : jit_generator_t("jit_uni_batch_normalization_fwd_stat_kernel")
    , data_type_(data_type)
    , calculate_variance_(calculate_variance) {
    create_kernel();
}

const jit_uni_batch_normalization_fwd_stat_kernel_t &
get_jit_uni_batch_normalization_fwd_stat_kernel(
        data_type_t dt, bool calculate_variance) {
    if (dt == data_type::f16)
        return get_jit_batch_normalization_fwd_stat_kernel_dt<data_type::f16>(
                calculate_variance);
    return get_jit_batch_normalization_fwd_stat_kernel_dt<data_type::f32>(
            calculate_variance);
}

void jit_uni_batch_normalization_fwd_stat_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_len = a2;
    const Reg reg_result = a3;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;

    const FReg f_mean = fa0;
    const FReg f_result = fa1;

    const VReg v_src16(2);
    const VReg v_src(4);
    const VReg v_reduction(8);

    ld(reg_src, reg_param, 0);
    ld(reg_len, reg_param, 8);
    ld(reg_result, reg_param, 16);
    if (calculate_variance_) flw(f_mean, reg_param, 24);

    // Seed the reduction from the caller-provided scalar. This permits an ncsp
    // channel to be accumulated across its non-contiguous minibatch slices.
    flw(f_result, reg_result, 0);

    Label loop, done;
    if (data_type_ == data_type::f32) {
        // Accumulate element-wise into a VLMAX-wide vector so no scalar FP
        // dependency chain is formed; tail-undisturbed keeps the partial sums
        // of lanes beyond a short final vl. Reduce once per call.
        const VReg v_acc(8);
        const VReg v_data(16);
        const VReg v_sum(4);

        vsetvli(reg_vl, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
        vmv_v_i(v_acc, 0);

        L(loop);
        beqz(reg_len, done);
        vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4, VTA::tu, VMA::ma);
        vle32_v(v_data, reg_src);
        if (calculate_variance_) {
            vfsub_vf(v_data, v_data, f_mean);
            vfmacc_vv(v_acc, v_data, v_data);
        } else {
            vfadd_vv(v_acc, v_acc, v_data);
        }
        slli(reg_bytes, reg_vl, 2);
        add(reg_src, reg_src, reg_bytes);
        sub(reg_len, reg_len, reg_vl);
        j_(loop);

        L(done);
        vsetvli(reg_vl, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
        vfmv_s_f(v_sum, f_result);
        vfredusum_vs(v_sum, v_acc, v_sum);
        vfmv_f_s(f_result, v_sum);
        fsw(f_result, reg_result, 0);
        ret();
        return;
    }

    L(loop);
    beqz(reg_len, done);

    vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
    vle16_v(v_src16, reg_src);
    vfwcvt_f_f_v(v_src, v_src16);
    vsetvli(reg_vl, reg_vl, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
    if (calculate_variance_) {
        vfsub_vf(v_src, v_src, f_mean);
        vfmul_vv(v_src, v_src, v_src);
    }

    // Use an ordered reduction so each chunk is folded into the scalar seed in
    // input order. Variance is a separate pass to avoid E[x^2] - E[x]^2
    // cancellation for inputs with a large mean and small variance.
    vfmv_v_f(v_reduction, f_result);
    vfredosum_vs(v_reduction, v_src, v_reduction);
    vfmv_f_s(f_result, v_reduction);

    slli(reg_bytes, reg_vl, 1);
    add(reg_src, reg_src, reg_bytes);
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    fsw(f_result, reg_result, 0);
    ret();
#else
    ret();
#endif
}

jit_uni_batch_normalization_fwd_kernel_t::
        jit_uni_batch_normalization_fwd_kernel_t(
                data_type_t data_type, bool per_elem_params, bool with_relu)
    : jit_generator_t("jit_uni_batch_normalization_fwd_kernel")
    , data_type_(data_type)
    , per_elem_params_(per_elem_params)
    , with_relu_(with_relu) {
    create_kernel();
}

void jit_uni_batch_normalization_apply(const void *src, void *dst, dim_t len,
        const float *mean, const float *scale_mul, const float *scale_add,
        data_type_t dt, bool per_elem_params, bool with_relu) {
    const jit_uni_batch_normalization_fwd_kernel_t::call_params_t p {
            src, dst, len, mean, scale_mul, scale_add};
    switch (dt) {
        case data_type::f16:
            dispatch_jit_batch_normalization_fwd_dt<data_type::f16>(
                    &p, per_elem_params, with_relu);
            break;
        case data_type::bf16:
            dispatch_jit_batch_normalization_fwd_dt<data_type::bf16>(
                    &p, per_elem_params, with_relu);
            break;
        default:
            dispatch_jit_batch_normalization_fwd_dt<data_type::f32>(
                    &p, per_elem_params, with_relu);
            break;
    }
}

void jit_uni_batch_normalization_fwd_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    // bf16 converts through Zvfbfmin, f16 through Zvfh. Both are issued under
    // the narrow e16 vtype and only the mnemonics differ, so the two xf16
    // dtypes share one widen-compute-narrow body.
    const bool is_bf16 = data_type_ == data_type::bf16;
    const bool is_xf16 = is_bf16 || data_type_ == data_type::f16;
    auto widen = [&](const VReg &vd, const VReg &vs) {
        if (is_bf16)
            vfwcvtbf16_f_f_v(vd, vs); // e16m1 -> e32m2
        else
            vfwcvt_f_f_v(vd, vs);
    };
    auto narrow = [&](const VReg &vd, const VReg &vs) {
        if (is_bf16)
            vfncvtbf16_f_f_w(vd, vs); // e32m2 -> e16m1
        else
            vfncvt_f_f_w(vd, vs);
    };
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_mean = a4;
    const Reg reg_sm = a5;
    const Reg reg_sv = a6;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;

    const FReg f_mean = fa0;
    const FReg f_sm = fa1;
    const FReg f_sv = fa2;
    const FReg f_zero = fa3;

    const VReg v_mask(0);
    const VReg v_src16(4);
    const VReg v_src(8);
    const VReg v_mean(10);
    const VReg v_sm(12);
    const VReg v_sv(14);

    ld(reg_src, reg_param, 0);
    ld(reg_dst, reg_param, 8);
    ld(reg_len, reg_param, 16);
    ld(reg_mean, reg_param, 24);
    ld(reg_sm, reg_param, 32);
    ld(reg_sv, reg_param, 40);

    if (!per_elem_params_) {
        flw(f_mean, reg_mean, 0);
        flw(f_sm, reg_sm, 0);
        flw(f_sv, reg_sv, 0);
    }
    fmv_w_x(f_zero, x0);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);
    if (is_xf16) {
        vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        vle16_v(v_src16, reg_src);
        widen(v_src, v_src16);
        vsetvli(reg_vl, reg_vl, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
    } else {
        vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        vle32_v(v_src, reg_src);
    }
    if (per_elem_params_) {
        vle32_v(v_mean, reg_mean);
        vle32_v(v_sm, reg_sm);
        vle32_v(v_sv, reg_sv);
        vfsub_vv(v_src, v_src, v_mean);
        vfmul_vv(v_src, v_src, v_sm);
        vfadd_vv(v_src, v_src, v_sv);
    } else {
        vfsub_vf(v_src, v_src, f_mean);
        vfmul_vf(v_src, v_src, f_sm);
        vfadd_vf(v_src, v_src, f_sv);
    }
    if (with_relu_) {
        vmflt_vf(v_mask, v_src, f_zero);
        vfmerge_vfm(v_src, v_src, f_zero);
    }
    if (is_xf16) {
        vsetvli(reg_vl, reg_vl, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        narrow(v_src16, v_src);
        vse16_v(v_src16, reg_dst);
    } else {
        vse32_v(v_src, reg_dst);
    }

    slli(reg_bytes, reg_vl, is_xf16 ? 1 : 2);
    add(reg_src, reg_src, reg_bytes);
    add(reg_dst, reg_dst, reg_bytes);
    if (per_elem_params_) {
        slli(reg_bytes, reg_vl, 2);
        add(reg_mean, reg_mean, reg_bytes);
        add(reg_sm, reg_sm, reg_bytes);
        add(reg_sv, reg_sv, reg_bytes);
    }
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    ret();
#else
    ret();
#endif
}

jit_uni_batch_normalization_bwd_reduce_kernel_t::
        jit_uni_batch_normalization_bwd_reduce_kernel_t(
                data_type_t data_type, bool per_elem_params)
    : jit_generator_t("jit_uni_batch_normalization_bwd_reduce_kernel")
    , data_type_(data_type)
    , per_elem_params_(per_elem_params) {
    create_kernel();
}

void jit_uni_batch_normalization_bwd_reduce(const void *src,
        const void *diff_dst, dim_t len, const float *mean, float *diff_scale,
        float *diff_shift, data_type_t dt, bool per_elem_params) {
    const jit_uni_batch_normalization_bwd_reduce_kernel_t::call_params_t p {
            src, diff_dst, len, mean, diff_scale, diff_shift};
    switch (dt) {
        case data_type::f16:
            dispatch_jit_batch_normalization_bwd_reduce_dt<data_type::f16>(
                    &p, per_elem_params);
            break;
        case data_type::bf16:
            dispatch_jit_batch_normalization_bwd_reduce_dt<data_type::bf16>(
                    &p, per_elem_params);
            break;
        default:
            dispatch_jit_batch_normalization_bwd_reduce_dt<data_type::f32>(
                    &p, per_elem_params);
            break;
    }
}

void jit_uni_batch_normalization_bwd_reduce_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    // bf16 converts through Zvfbfmin, f16 through Zvfh. Both are issued under
    // the narrow e16 vtype and only the mnemonics differ, so the two xf16
    // dtypes share one widen-compute-narrow body.
    const bool is_bf16 = data_type_ == data_type::bf16;
    const bool is_xf16 = is_bf16 || data_type_ == data_type::f16;
    auto widen = [&](const VReg &vd, const VReg &vs) {
        if (is_bf16)
            vfwcvtbf16_f_f_v(vd, vs); // e16m1 -> e32m2
        else
            vfwcvt_f_f_v(vd, vs);
    };
    const LMUL compute_lmul = is_xf16 ? LMUL::m2 : LMUL::m1;

    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_diff_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_mean = a4;
    const Reg reg_diff_scale = a5;
    const Reg reg_diff_shift = a6;
    const Reg reg_vl = t0;
    const Reg reg_data_bytes = t1;
    const Reg reg_param_bytes = t2;

    const FReg f_mean = fa0;
    const FReg f_diff_scale = fa1;
    const FReg f_diff_shift = fa2;
    const FReg f_reduce = fa3;

    const VReg v_src16(2);
    const VReg v_diff_dst16(4);
    const VReg v_src(8);
    const VReg v_diff_dst(10);
    const VReg v_mean(12);
    const VReg v_diff_scale(14);
    const VReg v_diff_shift(16);
    const VReg v_work(18);
    const VReg v_zero(20);
    const VReg v_reduce(21);

    ld(reg_src, reg_param, 0);
    ld(reg_diff_dst, reg_param, 8);
    ld(reg_len, reg_param, 16);
    ld(reg_mean, reg_param, 24);
    ld(reg_diff_scale, reg_param, 32);
    ld(reg_diff_shift, reg_param, 40);

    if (!per_elem_params_) {
        flw(f_mean, reg_mean, 0);
        flw(f_diff_scale, reg_diff_scale, 0);
        flw(f_diff_shift, reg_diff_shift, 0);
        vsetvli(reg_param_bytes, x0, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        vmv_v_x(v_zero, x0);
    }

    Label loop, done;
    L(loop);
    beqz(reg_len, done);
    if (is_xf16) {
        vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        vle16_v(v_src16, reg_src);
        vle16_v(v_diff_dst16, reg_diff_dst);
        widen(v_src, v_src16);
        widen(v_diff_dst, v_diff_dst16);
        vsetvli(reg_vl, reg_vl, SEW::e32, compute_lmul, VTA::ta, VMA::ma);
    } else {
        vsetvli(reg_vl, reg_len, SEW::e32, compute_lmul, VTA::ta, VMA::ma);
        vle32_v(v_src, reg_src);
        vle32_v(v_diff_dst, reg_diff_dst);
    }

    if (per_elem_params_) {
        vle32_v(v_mean, reg_mean);
        vle32_v(v_diff_scale, reg_diff_scale);
        vle32_v(v_diff_shift, reg_diff_shift);
        vfsub_vv(v_work, v_src, v_mean);
        vfmul_vv(v_work, v_work, v_diff_dst);
        vfadd_vv(v_diff_scale, v_diff_scale, v_work);
        vfadd_vv(v_diff_shift, v_diff_shift, v_diff_dst);
        vse32_v(v_diff_scale, reg_diff_scale);
        vse32_v(v_diff_shift, reg_diff_shift);
    } else {
        vfsub_vf(v_work, v_src, f_mean);
        vfmul_vv(v_work, v_work, v_diff_dst);

        vfredusum_vs(v_reduce, v_work, v_zero);
        vfmv_f_s(f_reduce, v_reduce);
        fadd_s(f_diff_scale, f_diff_scale, f_reduce);
        vfredusum_vs(v_reduce, v_diff_dst, v_zero);
        vfmv_f_s(f_reduce, v_reduce);
        fadd_s(f_diff_shift, f_diff_shift, f_reduce);
    }

    slli(reg_data_bytes, reg_vl, is_xf16 ? 1 : 2);
    add(reg_src, reg_src, reg_data_bytes);
    add(reg_diff_dst, reg_diff_dst, reg_data_bytes);
    if (per_elem_params_) {
        slli(reg_param_bytes, reg_vl, 2);
        add(reg_mean, reg_mean, reg_param_bytes);
        add(reg_diff_scale, reg_diff_scale, reg_param_bytes);
        add(reg_diff_shift, reg_diff_shift, reg_param_bytes);
    }
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    if (!per_elem_params_) {
        fsw(f_diff_scale, reg_diff_scale, 0);
        fsw(f_diff_shift, reg_diff_shift, 0);
    }
    ret();
#else
    ret();
#endif
}

jit_uni_batch_normalization_bwd_apply_kernel_t::
        jit_uni_batch_normalization_bwd_apply_kernel_t(
                data_type_t data_type, bool per_elem_params)
    : jit_generator_t("jit_uni_batch_normalization_bwd_apply_kernel")
    , data_type_(data_type)
    , per_elem_params_(per_elem_params) {
    create_kernel();
}

void jit_uni_batch_normalization_bwd_apply(const void *src,
        const void *diff_dst, void *diff_src, dim_t len, const float *mean,
        const float *scale_mul, const float *diff_scale_mul,
        const float *diff_shift_add, data_type_t dt, bool per_elem_params) {
    const jit_uni_batch_normalization_bwd_apply_kernel_t::call_params_t p {src,
            diff_dst, diff_src, len, mean, scale_mul, diff_scale_mul,
            diff_shift_add};
    switch (dt) {
        case data_type::f16:
            dispatch_jit_batch_normalization_bwd_apply_dt<data_type::f16>(
                    &p, per_elem_params);
            break;
        case data_type::bf16:
            dispatch_jit_batch_normalization_bwd_apply_dt<data_type::bf16>(
                    &p, per_elem_params);
            break;
        default:
            dispatch_jit_batch_normalization_bwd_apply_dt<data_type::f32>(
                    &p, per_elem_params);
            break;
    }
}

void jit_uni_batch_normalization_bwd_apply_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    // bf16 converts through Zvfbfmin, f16 through Zvfh. Both are issued under
    // the narrow e16 vtype and only the mnemonics differ, so the two xf16
    // dtypes share one widen-compute-narrow body.
    const bool is_bf16 = data_type_ == data_type::bf16;
    const bool is_xf16 = is_bf16 || data_type_ == data_type::f16;
    auto widen = [&](const VReg &vd, const VReg &vs) {
        if (is_bf16)
            vfwcvtbf16_f_f_v(vd, vs); // e16m1 -> e32m2
        else
            vfwcvt_f_f_v(vd, vs);
    };
    auto narrow = [&](const VReg &vd, const VReg &vs) {
        if (is_bf16)
            vfncvtbf16_f_f_w(vd, vs); // e32m2 -> e16m1
        else
            vfncvt_f_f_w(vd, vs);
    };
    const LMUL compute_lmul = is_xf16 ? LMUL::m2 : LMUL::m1;

    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_diff_dst = a2;
    const Reg reg_diff_src = a3;
    const Reg reg_len = a4;
    const Reg reg_mean = a5;
    const Reg reg_scale_mul = a6;
    const Reg reg_diff_scale_mul = a7;
    const Reg reg_diff_shift_add = t0;
    const Reg reg_vl = t1;
    const Reg reg_data_bytes = t2;
    const Reg reg_param_bytes = t3;

    const FReg f_mean = fa0;
    const FReg f_scale_mul = fa1;
    const FReg f_diff_scale_mul = fa2;
    const FReg f_diff_shift_add = fa3;

    const VReg v_src16(2);
    const VReg v_diff_dst16(4);
    const VReg v_diff_src16(6);
    const VReg v_src(8);
    const VReg v_diff_dst(10);
    const VReg v_mean(12);
    const VReg v_scale_mul(14);
    const VReg v_diff_scale_mul(16);
    const VReg v_diff_shift_add(18);
    const VReg v_work(20);

    ld(reg_src, reg_param, 0);
    ld(reg_diff_dst, reg_param, 8);
    ld(reg_diff_src, reg_param, 16);
    ld(reg_len, reg_param, 24);
    ld(reg_mean, reg_param, 32);
    ld(reg_scale_mul, reg_param, 40);
    ld(reg_diff_scale_mul, reg_param, 48);
    ld(reg_diff_shift_add, reg_param, 56);

    if (!per_elem_params_) {
        flw(f_mean, reg_mean, 0);
        flw(f_scale_mul, reg_scale_mul, 0);
        flw(f_diff_scale_mul, reg_diff_scale_mul, 0);
        flw(f_diff_shift_add, reg_diff_shift_add, 0);
    }

    Label loop, done;
    L(loop);
    beqz(reg_len, done);
    if (is_xf16) {
        vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        vle16_v(v_src16, reg_src);
        vle16_v(v_diff_dst16, reg_diff_dst);
        widen(v_src, v_src16);
        widen(v_diff_dst, v_diff_dst16);
        vsetvli(reg_vl, reg_vl, SEW::e32, compute_lmul, VTA::ta, VMA::ma);
    } else {
        vsetvli(reg_vl, reg_len, SEW::e32, compute_lmul, VTA::ta, VMA::ma);
        vle32_v(v_src, reg_src);
        vle32_v(v_diff_dst, reg_diff_dst);
    }

    if (per_elem_params_) {
        vle32_v(v_mean, reg_mean);
        vle32_v(v_scale_mul, reg_scale_mul);
        vle32_v(v_diff_scale_mul, reg_diff_scale_mul);
        vle32_v(v_diff_shift_add, reg_diff_shift_add);
        vfsub_vv(v_work, v_src, v_mean);
        vfmul_vv(v_work, v_work, v_diff_scale_mul);
        vfadd_vv(v_work, v_work, v_diff_shift_add);
        vfsub_vv(v_diff_dst, v_diff_dst, v_work);
        vfmul_vv(v_diff_dst, v_diff_dst, v_scale_mul);
    } else {
        vfsub_vf(v_work, v_src, f_mean);
        vfmul_vf(v_work, v_work, f_diff_scale_mul);
        vfadd_vf(v_work, v_work, f_diff_shift_add);
        vfsub_vv(v_diff_dst, v_diff_dst, v_work);
        vfmul_vf(v_diff_dst, v_diff_dst, f_scale_mul);
    }

    if (is_xf16) {
        vsetvli(reg_vl, reg_vl, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        narrow(v_diff_src16, v_diff_dst);
        vse16_v(v_diff_src16, reg_diff_src);
    } else {
        vse32_v(v_diff_dst, reg_diff_src);
    }

    slli(reg_data_bytes, reg_vl, is_xf16 ? 1 : 2);
    add(reg_src, reg_src, reg_data_bytes);
    add(reg_diff_dst, reg_diff_dst, reg_data_bytes);
    add(reg_diff_src, reg_diff_src, reg_data_bytes);
    if (per_elem_params_) {
        slli(reg_param_bytes, reg_vl, 2);
        add(reg_mean, reg_mean, reg_param_bytes);
        add(reg_scale_mul, reg_scale_mul, reg_param_bytes);
        add(reg_diff_scale_mul, reg_diff_scale_mul, reg_param_bytes);
        add(reg_diff_shift_add, reg_diff_shift_add, reg_param_bytes);
    }
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    ret();
#else
    ret();
#endif
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
