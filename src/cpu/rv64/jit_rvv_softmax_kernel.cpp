/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
* Copyright 2026 SpacemiT Corporation
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

#include <cstddef>

#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_rvv_softmax_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define XF16_AFFINE_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_softmax_xf16_affine_kernel_t::call_params_t, field))
#define XF16_STRIDED_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_softmax_xf16_strided_kernel_t::call_params_t, field))
#define XF16_EXP_SUB_SUM_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_softmax_xf16_exp_sub_sum_kernel_t::call_params_t, field))
#define F32_EXP_SUB_SUM_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_softmax_f32_exp_sub_sum_kernel_t::call_params_t, field))
#define XF16_REDUCE_MAX_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_softmax_xf16_reduce_max_kernel_t::call_params_t, field))
#define F32_REDUCE_MAX_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_softmax_f32_reduce_max_kernel_t::call_params_t, field))

template <bool gather>
const jit_rvv_softmax_xf16_strided_kernel_t &get_xf16_strided_kernel() {
    static const jit_rvv_softmax_xf16_strided_kernel_t kernel(gather);
    return kernel;
}

template const jit_rvv_softmax_xf16_strided_kernel_t &
get_xf16_strided_kernel<true>();
template const jit_rvv_softmax_xf16_strided_kernel_t &
get_xf16_strided_kernel<false>();

namespace {

template <bool src_f32, data_type_t dt>
void dispatch_xf16_affine(
        const jit_rvv_softmax_xf16_affine_kernel_t::call_params_t *p) {
    static const jit_rvv_softmax_xf16_affine_kernel_t kernel(src_f32, dt);
    kernel(p);
}

template <bool gather>
void dispatch_xf16_strided(
        const jit_rvv_softmax_xf16_strided_kernel_t::call_params_t *p) {
    get_xf16_strided_kernel<gather>()(p);
}

template <data_type_t dt>
void dispatch_xf16_exp_sub_sum(
        const jit_rvv_softmax_xf16_exp_sub_sum_kernel_t::call_params_t *p) {
    static const jit_rvv_softmax_xf16_exp_sub_sum_kernel_t kernel(dt);
    kernel(p);
}

template <bool store_exp>
void dispatch_f32_exp_sub_sum(
        const jit_rvv_softmax_f32_exp_sub_sum_kernel_t::call_params_t *p) {
    static const jit_rvv_softmax_f32_exp_sub_sum_kernel_t kernel(store_exp);
    kernel(p);
}

template <data_type_t dt>
void dispatch_xf16_reduce_max(
        const jit_rvv_softmax_xf16_reduce_max_kernel_t::call_params_t *p) {
    static const jit_rvv_softmax_xf16_reduce_max_kernel_t kernel(dt);
    kernel(p);
}

void dispatch_f32_reduce_max(
        const jit_rvv_softmax_f32_reduce_max_kernel_t::call_params_t *p) {
    static const jit_rvv_softmax_f32_reduce_max_kernel_t kernel;
    kernel(p);
}

} // namespace

jit_rvv_softmax_affine_kernel_t::jit_rvv_softmax_affine_kernel_t()
    : jit_generator_t("jit_rvv_softmax_affine_kernel") {
    create_kernel();
}

void jit_rvv_softmax_affine_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_tmp = t2;

    const FReg f_sub = fa0;
    const FReg f_mul = fa1;

    const VReg v_src(0);

    // call_params_t layout:
    //  0: src, 8: dst, 16: len, 24: sub, 28: mul
    ld(reg_src, reg_param, 0);
    ld(reg_dst, reg_param, 8);
    ld(reg_len, reg_param, 16);

    lw(reg_tmp, reg_param, 24);
    fmv_w_x(f_sub, reg_tmp);
    lw(reg_tmp, reg_param, 28);
    fmv_w_x(f_mul, reg_tmp);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    vle32_v(v_src, reg_src);
    vfsub_vf(v_src, v_src, f_sub);
    vfmul_vf(v_src, v_src, f_mul);
    vse32_v(v_src, reg_dst);
    slli(reg_bytes, reg_vl, 2);
    add(reg_src, reg_src, reg_bytes);
    add(reg_dst, reg_dst, reg_bytes);
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    ret();
#else
    ret();
#endif
}

jit_rvv_softmax_xf16_affine_kernel_t::jit_rvv_softmax_xf16_affine_kernel_t(
        bool src_f32, data_type_t dt)
    : jit_generator_t("jit_rvv_softmax_xf16_affine_kernel")
    , src_f32_(src_f32)
    , dt_(dt) {
    create_kernel();
}

jit_rvv_softmax_xf16_strided_kernel_t::jit_rvv_softmax_xf16_strided_kernel_t(
        bool gather)
    : jit_generator_t("jit_rvv_softmax_xf16_strided_kernel"), gather_(gather) {
    create_kernel();
}

void jit_rvv_softmax_xf16_affine_from_xf16(data_type_t dt, const void *src,
        void *dst, dim_t len, float sub, float mul) {
    const jit_rvv_softmax_xf16_affine_kernel_t::call_params_t p {
            src, dst, len, sub, mul};
    if (dt == data_type::bf16)
        dispatch_xf16_affine<false, data_type::bf16>(&p);
    else
        dispatch_xf16_affine<false, data_type::f16>(&p);
}

void jit_rvv_softmax_xf16_affine_from_f32(data_type_t dt, const float *src,
        void *dst, dim_t len, float sub, float mul) {
    const jit_rvv_softmax_xf16_affine_kernel_t::call_params_t p {
            src, dst, len, sub, mul};
    if (dt == data_type::bf16)
        dispatch_xf16_affine<true, data_type::bf16>(&p);
    else
        dispatch_xf16_affine<true, data_type::f16>(&p);
}

void jit_rvv_softmax_xf16_gather(
        const void *src, void *dst, dim_t len, dim_t stride_bytes) {
    const jit_rvv_softmax_xf16_strided_kernel_t::call_params_t p {
            src, dst, len, stride_bytes};
    dispatch_xf16_strided<true>(&p);
}

void jit_rvv_softmax_xf16_scatter(
        const void *src, void *dst, dim_t len, dim_t stride_bytes) {
    const jit_rvv_softmax_xf16_strided_kernel_t::call_params_t p {
            src, dst, len, stride_bytes};
    dispatch_xf16_strided<false>(&p);
}

void jit_rvv_softmax_xf16_exp_sub_sum(data_type_t dt, const void *src,
        float *tmp, dim_t len, float sub, float *sum) {
    const jit_rvv_softmax_xf16_exp_sub_sum_kernel_t::call_params_t p {
            src, tmp, len, sub, sum};
    if (dt == data_type::bf16)
        dispatch_xf16_exp_sub_sum<data_type::bf16>(&p);
    else
        dispatch_xf16_exp_sub_sum<data_type::f16>(&p);
}

jit_rvv_softmax_xf16_reduce_max_kernel_t::
        jit_rvv_softmax_xf16_reduce_max_kernel_t(data_type_t dt)
    : jit_generator_t("jit_rvv_softmax_xf16_reduce_max_kernel"), dt_(dt) {
    create_kernel();
}

void jit_rvv_softmax_xf16_reduce_max(data_type_t dt, const void *src, dim_t len,
        float *max_val, uint32_t *has_nan) {
    const jit_rvv_softmax_xf16_reduce_max_kernel_t::call_params_t p {
            src, len, max_val, has_nan};
    if (dt == data_type::bf16)
        dispatch_xf16_reduce_max<data_type::bf16>(&p);
    else
        dispatch_xf16_reduce_max<data_type::f16>(&p);
}

void jit_rvv_softmax_f32_reduce_max(
        const float *src, dim_t len, float *max_val) {
    const jit_rvv_softmax_f32_reduce_max_kernel_t::call_params_t p {
            src, len, max_val};
    dispatch_f32_reduce_max(&p);
}

void jit_rvv_softmax_xf16_reduce_max_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_len = a2;
    const Reg reg_max_ptr = a3;
    const Reg reg_nan_ptr = a4;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_tmp = t2;
    const Reg reg_has = t3;
    const Reg reg_imm = t4;

    const FReg f_seed = ft1;
    const FReg f_max = ft2;

    const VReg v_f16(4);
    const VReg v_nan_acc(2);
    const VReg v_x(8);
    const VReg v_red(28);

    auto load_f32_bits = [&](const FReg &freg, uint32_t bits) {
        li(reg_imm, static_cast<int64_t>(bits));
        fmv_w_x(freg, reg_imm);
    };

    ld(reg_src, reg_param, XF16_REDUCE_MAX_OFF(src));
    ld(reg_len, reg_param, XF16_REDUCE_MAX_OFF(len));
    ld(reg_max_ptr, reg_param, XF16_REDUCE_MAX_OFF(max_val));
    ld(reg_nan_ptr, reg_param, XF16_REDUCE_MAX_OFF(has_nan));

    // Seed for the max reduction: (float)numeric_limits<T>::lowest(), i.e.
    // -65504.0f (0xC77FE000u) for f16 and -3.38953139e38f (0xFF7F0000u) for
    // bf16. It only ever wins the reduction when every element is that lowest
    // value, -inf or NaN, which matches the scalar `> max_val` loop seeded the
    // same way.
    const bool is_bf16 = dt_ == data_type::bf16;
    load_f32_bits(f_seed, is_bf16 ? 0xFF7F0000u : 0xC77FE000u);

    // Clear the xf16 tail lanes to +0.0 so a partial final chunk does not feed
    // stale NaNs into the NaN-detection pass (the loop below loads with
    // VTA::tu, i.e. tail undisturbed).
    vsetvli(reg_vl, x0, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
    vmv_v_i(v_f16, 0);
    // Accumulator for the OR of all per-chunk NaN masks.
    vmclr_m(v_nan_acc);

    // Seed the vector max accumulator with the dtype's lowest value.
    vsetvli(reg_vl, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    vfmv_v_f(v_red, f_seed);

    Label loop, finish;
    L(loop);
    beqz(reg_len, finish);

    vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m2, VTA::tu, VMA::ma);
    vle16_v(v_f16, reg_src);
    slli(reg_bytes, reg_vl, 1);
    add(reg_src, reg_src, reg_bytes);

    // Accumulate the per-chunk NaN masks (x != x). f16 tests at e16 directly;
    // bf16 has no e16 arithmetic, so it widens first -- the widen is exact and
    // preserves NaN, so the f32 test is equivalent.
    if (is_bf16) {
        vfwcvtbf16_f_f_v(v_x, v_f16);
        vsetvli(reg_vl, reg_vl, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
        vmfne_vv(v0, v_x, v_x);
        vmor_mm(v_nan_acc, v_nan_acc, v0);
    } else {
        vmfne_vv(v0, v_f16, v_f16);
        vmor_mm(v_nan_acc, v_nan_acc, v0);
        vfwcvt_f_f_v(v_x, v_f16);
        vsetvli(reg_vl, reg_vl, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    }
    // Exclude NaN lanes from the max (matches the scalar `val > max_val` test,
    // under which a NaN never updates max_val).
    vfmerge_vfm(v_x, v_x, f_seed);
    vfredmax_vs(v_red, v_x, v_red);

    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(finish);
    vfmv_f_s(f_max, v_red);
    fsw(f_max, reg_max_ptr, 0);

    // has_nan = any set bit in the accumulated NaN mask.
    vsetvli(reg_vl, x0, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
    vfirst_m(reg_tmp, v_nan_acc);
    slti(reg_has, reg_tmp, 0);
    xori(reg_has, reg_has, 1);
    sw(reg_has, reg_nan_ptr, 0);
    ret();
#else
    ret();
#endif
}

jit_rvv_softmax_f32_reduce_max_kernel_t::
        jit_rvv_softmax_f32_reduce_max_kernel_t()
    : jit_generator_t("jit_rvv_softmax_f32_reduce_max_kernel") {
    create_kernel();
}

// f32 Stage-1 max reduction, mirroring jit_rvv_softmax_f16_reduce_max. The
// scalar loop seeds max_val with -INFINITY and only ever replaces it under
// `val > max_val`; a NaN never wins. vfredmax_vs therefore needs its own
// -INFINITY seed, and NaN lanes must be merged to that seed before the
// reduction so they cannot displace a valid maximum (vfredmax alone would
// propagate a NaN). vfredmax_vs reduces only the vl active elements, so a
// short final chunk needs no explicit tail handling.
void jit_rvv_softmax_f32_reduce_max_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_len = a2;
    const Reg reg_max_ptr = a3;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_imm = t4;

    const FReg f_seed = ft1;
    const FReg f_max = ft2;

    const VReg v_x(8);
    const VReg v_red(28);

    auto load_f32_bits = [&](const FReg &freg, uint32_t bits) {
        li(reg_imm, static_cast<int64_t>(bits));
        fmv_w_x(freg, reg_imm);
    };

    ld(reg_src, reg_param, F32_REDUCE_MAX_OFF(src));
    ld(reg_len, reg_param, F32_REDUCE_MAX_OFF(len));
    ld(reg_max_ptr, reg_param, F32_REDUCE_MAX_OFF(max_val));

    // Seed for the max reduction: -INFINITY (0xFF800000), matching the scalar
    // `> max_val` loop. It only ever wins when every element is -Inf or NaN.
    load_f32_bits(f_seed, 0xFF800000u);

    // Seed the vector max accumulator with -INFINITY.
    vsetvli(reg_vl, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    vfmv_v_f(v_red, f_seed);

    Label loop, finish;
    L(loop);
    beqz(reg_len, finish);

    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    vle32_v(v_x, reg_src);
    slli(reg_bytes, reg_vl, 2);
    add(reg_src, reg_src, reg_bytes);

    // Exclude NaN lanes from the max: a NaN is x != x, and merging it to the
    // -INFINITY seed reproduces the scalar `val > max_val` contract under
    // which a NaN never updates max_val.
    vmfne_vv(v0, v_x, v_x);
    vfmerge_vfm(v_x, v_x, f_seed);
    vfredmax_vs(v_red, v_x, v_red);

    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(finish);
    vfmv_f_s(f_max, v_red);
    fsw(f_max, reg_max_ptr, 0);
    ret();
#else
    ret();
#endif
}

void jit_rvv_softmax_xf16_affine_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_vl = t0;
    const Reg reg_src_bytes = t1;
    const Reg reg_dst_bytes = t2;

    const FReg f_sub = fa0;
    const FReg f_mul = fa1;

    const VReg v_src(4);
    const VReg v_f32(8);
    const VReg v_dst(4);

    ld(reg_src, reg_param, XF16_AFFINE_OFF(src));
    ld(reg_dst, reg_param, XF16_AFFINE_OFF(dst));
    ld(reg_len, reg_param, XF16_AFFINE_OFF(len));
    flw(f_sub, reg_param, XF16_AFFINE_OFF(sub));
    flw(f_mul, reg_param, XF16_AFFINE_OFF(mul));

    Label loop, done;
    L(loop);
    beqz(reg_len, done);

    const bool is_bf16 = dt_ == data_type::bf16;

    if (src_f32_) {
        vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        vle32_v(v_f32, reg_src);
    } else {
        vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        vle16_v(v_src, reg_src);
        if (is_bf16)
            vfwcvtbf16_f_f_v(v_f32, v_src);
        else
            vfwcvt_f_f_v(v_f32, v_src);
        vsetvli(reg_vl, reg_vl, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
    }

    vfsub_vf(v_f32, v_f32, f_sub);
    vfmul_vf(v_f32, v_f32, f_mul);
    vsetvli(reg_vl, reg_vl, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
    if (is_bf16)
        vfncvtbf16_f_f_w(v_dst, v_f32);
    else
        vfncvt_f_f_w(v_dst, v_f32);
    vse16_v(v_dst, reg_dst);

    slli(reg_dst_bytes, reg_vl, 1);
    add(reg_dst, reg_dst, reg_dst_bytes);
    if (src_f32_) {
        slli(reg_src_bytes, reg_vl, 2);
    } else {
        slli(reg_src_bytes, reg_vl, 1);
    }
    add(reg_src, reg_src, reg_src_bytes);
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    ret();
#else
    ret();
#endif
}

void jit_rvv_softmax_xf16_strided_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_stride = a4;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;

    const VReg v_data(4);

    ld(reg_src, reg_param, XF16_STRIDED_OFF(src));
    ld(reg_dst, reg_param, XF16_STRIDED_OFF(dst));
    ld(reg_len, reg_param, XF16_STRIDED_OFF(len));
    ld(reg_stride, reg_param, XF16_STRIDED_OFF(stride_bytes));

    Label loop, done;
    L(loop);
    beqz(reg_len, done);

    vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
    if (gather_) {
        vlse16_v(v_data, reg_src, reg_stride);
        vse16_v(v_data, reg_dst);
        mul(reg_bytes, reg_vl, reg_stride);
        add(reg_src, reg_src, reg_bytes);
        slli(reg_bytes, reg_vl, 1);
        add(reg_dst, reg_dst, reg_bytes);
    } else {
        vle16_v(v_data, reg_src);
        vsse16_v(v_data, reg_dst, reg_stride);
        slli(reg_bytes, reg_vl, 1);
        add(reg_src, reg_src, reg_bytes);
        mul(reg_bytes, reg_vl, reg_stride);
        add(reg_dst, reg_dst, reg_bytes);
    }

    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    ret();
#else
    ret();
#endif
}

jit_rvv_softmax_xf16_exp_sub_sum_kernel_t::
        jit_rvv_softmax_xf16_exp_sub_sum_kernel_t(data_type_t dt)
    : jit_generator_t("jit_rvv_softmax_xf16_exp_sub_sum_kernel"), dt_(dt) {
    create_kernel();
}

void jit_rvv_softmax_xf16_exp_sub_sum_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const bool hoist_poly_coeffs = get_platform_vlen() >= 256;
    // Share exact coefficient bits between hoisted and in-loop loads.
    constexpr uint32_t poly_coeffs[]
            = {0x3c092f6eu, 0x3d2aadadu, 0x3e2aaa28u, 0x3efffffbu, 0x3f800000u};

    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_tmp = a2;
    const Reg reg_len = a3;
    const Reg reg_sum = a4;
    const Reg reg_sub_tmp = t2;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_maxexp = t4;
    const Reg reg_minexp = t5;
    const Reg reg_imm = t6;

    const FReg f_zero = ft0;
    const FReg f_lower = ft1;
    const FReg f_upper = ft2;
    const FReg f_round = ft3;
    const FReg f_log2_recip = ft4;
    const FReg f_log2_high = ft5;
    const FReg f_log2_low = ft6;
    const FReg f_sub = fa0;
    const FReg f_poly = ft7;
    const FReg f_poly_coeff = ft8;
    const FReg f_poly_coeff0 = ft8;
    const FReg f_poly_coeff1 = ft9;
    const FReg f_poly_coeff2 = ft11;
    const FReg f_poly_coeff3 = fa1;
    const FReg f_poly_coeff4 = fa2;
    const FReg f_sum = ft10;

    const VReg v_in16(0);
    const VReg v_x(8);
    const VReg v_bias(12);
    const VReg v_poly(16);
    const VReg v_tmpv(20);
    const VReg v_acc(24);
    const VReg v_red(28);
    const VReg v_mask(0);

    auto load_f32_bits = [&](const FReg &freg, uint32_t bits) {
        li(reg_imm, static_cast<int64_t>(bits));
        fmv_w_x(freg, reg_imm);
    };
    auto load_f32 = [&](const FReg &freg, float value) {
        load_f32_bits(freg, utils::bit_cast<uint32_t>(value));
    };
    auto add_poly_coeff = [&](const FReg &hoisted_coeff, uint32_t bits) {
        const FReg &coeff = hoist_poly_coeffs ? hoisted_coeff : f_poly_coeff;
        if (!hoist_poly_coeffs) load_f32_bits(coeff, bits);
        vfadd_vf(v_poly, v_poly, coeff);
    };

    ld(reg_src, reg_param, XF16_EXP_SUB_SUM_OFF(src));
    ld(reg_tmp, reg_param, XF16_EXP_SUB_SUM_OFF(tmp));
    ld(reg_len, reg_param, XF16_EXP_SUB_SUM_OFF(len));
    ld(reg_sum, reg_param, XF16_EXP_SUB_SUM_OFF(sum));
    lw(reg_sub_tmp, reg_param, XF16_EXP_SUB_SUM_OFF(sub));
    fmv_w_x(f_sub, reg_sub_tmp);

    fmv_w_x(f_zero, x0);
    load_f32(f_lower, -103.9720840454f);
    load_f32(f_upper, 88.7762626647950f);
    load_f32_bits(f_round, 0x4b400000u);
    load_f32_bits(f_log2_recip, 0x3fb8aa3bu);
    load_f32_bits(f_log2_high, 0xbf317200u);
    load_f32_bits(f_log2_low, 0xb5bfbe8eu);
    load_f32_bits(f_poly, 0x3ab4a000u);
    if (hoist_poly_coeffs) {
        load_f32_bits(f_poly_coeff0, poly_coeffs[0]);
        load_f32_bits(f_poly_coeff1, poly_coeffs[1]);
        load_f32_bits(f_poly_coeff2, poly_coeffs[2]);
        load_f32_bits(f_poly_coeff3, poly_coeffs[3]);
        load_f32_bits(f_poly_coeff4, poly_coeffs[4]);
    }
    li(reg_minexp, static_cast<int64_t>(0xC1000000u));
    li(reg_maxexp, static_cast<int64_t>(0x3F800000u));

    vsetvli(reg_vl, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    vfmv_v_f(v_acc, f_zero);

    Label loop, finish;
    L(loop);
    beqz(reg_len, finish);

    vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
    vle16_v(v_in16, reg_src);
    slli(reg_bytes, reg_vl, 1);
    add(reg_src, reg_src, reg_bytes);
    if (dt_ == data_type::bf16)
        vfwcvtbf16_f_f_v(v_x, v_in16);
    else
        vfwcvt_f_f_v(v_x, v_in16);

    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4, VTA::tu, VMA::ma);
    vfsub_vf(v_x, v_x, f_sub);
    vfmv_v_f(v_bias, f_round);
    vmflt_vf(v_mask, v_x, f_lower);
    vfmerge_vfm(v_x, v_x, f_lower);
    vmfgt_vf(v_mask, v_x, f_upper);
    vfmerge_vfm(v_x, v_x, f_upper);
    vfmacc_vf(v_bias, f_log2_recip, v_x);
    vfsub_vf(v_tmpv, v_bias, f_round);
    vfmacc_vf(v_x, f_log2_high, v_tmpv);
    vfmacc_vf(v_x, f_log2_low, v_tmpv);
    vfmv_v_f(v_poly, f_poly);
    vfmul_vv(v_poly, v_poly, v_x);
    add_poly_coeff(f_poly_coeff0, poly_coeffs[0]);
    vfmul_vv(v_poly, v_poly, v_x);
    add_poly_coeff(f_poly_coeff1, poly_coeffs[1]);
    vfmul_vv(v_poly, v_poly, v_x);
    add_poly_coeff(f_poly_coeff2, poly_coeffs[2]);
    vfmul_vv(v_poly, v_poly, v_x);
    add_poly_coeff(f_poly_coeff3, poly_coeffs[3]);
    vfmul_vv(v_poly, v_poly, v_x);
    add_poly_coeff(f_poly_coeff4, poly_coeffs[4]);
    vsll_vi(v_bias, v_bias, 23);
    vmin_vx(v_tmpv, v_bias, reg_maxexp);
    vmax_vx(v_tmpv, v_tmpv, reg_minexp);
    vsub_vv(v_bias, v_bias, v_tmpv);
    vadd_vx(v_bias, v_bias, reg_maxexp);
    vadd_vx(v_tmpv, v_tmpv, reg_maxexp);
    vfmul_vv(v_x, v_x, v_bias);
    vfmadd_vv(v_poly, v_x, v_bias);
    vfmul_vv(v_poly, v_poly, v_tmpv);
    vfadd_vv(v_acc, v_acc, v_poly);
    vse32_v(v_poly, reg_tmp);
    slli(reg_bytes, reg_vl, 2);
    add(reg_tmp, reg_tmp, reg_bytes);
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(finish);
    vsetvli(reg_vl, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    vfmv_v_f(v_red, f_zero);
    vfredosum_vs(v_red, v_acc, v_red);
    vfmv_f_s(f_sum, v_red);
    fsw(f_sum, reg_sum, 0);
    ret();
#else
    ret();
#endif
}

jit_rvv_softmax_f32_exp_sub_sum_kernel_t::
        jit_rvv_softmax_f32_exp_sub_sum_kernel_t(bool store_exp)
    : jit_generator_t("jit_rvv_softmax_f32_exp_sub_sum_kernel")
    , store_exp_(store_exp) {
    create_kernel();
}

// Same range-reduced polynomial exp as the f16 kernel (which already computes
// in f32 internally); the input is simply loaded as f32 instead of widened
// from f16. The accumulator uses VTA::tu so a final short tail folds into the
// low lanes and the closing vfredosum still produces the exact ordered sum.
void jit_rvv_softmax_f32_exp_sub_sum_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_sum = a4;
    const Reg reg_sub_tmp = t2;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_maxexp = t4;
    const Reg reg_minexp = t5;
    const Reg reg_imm = t6;

    const FReg f_zero = ft0;
    const FReg f_lower = ft1;
    const FReg f_upper = ft2;
    const FReg f_round = ft3;
    const FReg f_log2_recip = ft4;
    const FReg f_log2_high = ft5;
    const FReg f_log2_low = ft6;
    const FReg f_sub = fa0;
    const FReg f_poly = ft7;
    const FReg f_poly_coeff = ft8;
    const FReg f_sum = ft10;

    const VReg v_x(8);
    const VReg v_bias(12);
    const VReg v_poly(16);
    const VReg v_tmpv(20);
    const VReg v_acc(24);
    const VReg v_red(28);
    const VReg v_mask(0);

    auto load_f32_bits = [&](const FReg &freg, uint32_t bits) {
        li(reg_imm, static_cast<int64_t>(bits));
        fmv_w_x(freg, reg_imm);
    };
    auto load_f32 = [&](const FReg &freg, float value) {
        load_f32_bits(freg, utils::bit_cast<uint32_t>(value));
    };

    ld(reg_src, reg_param, F32_EXP_SUB_SUM_OFF(src));
    ld(reg_dst, reg_param, F32_EXP_SUB_SUM_OFF(dst));
    ld(reg_len, reg_param, F32_EXP_SUB_SUM_OFF(len));
    ld(reg_sum, reg_param, F32_EXP_SUB_SUM_OFF(sum));
    lw(reg_sub_tmp, reg_param, F32_EXP_SUB_SUM_OFF(sub));
    fmv_w_x(f_sub, reg_sub_tmp);

    fmv_w_x(f_zero, x0);
    load_f32(f_lower, -103.9720840454f);
    load_f32(f_upper, 88.7762626647950f);
    load_f32_bits(f_round, 0x4b400000u);
    load_f32_bits(f_log2_recip, 0x3fb8aa3bu);
    load_f32_bits(f_log2_high, 0xbf317200u);
    load_f32_bits(f_log2_low, 0xb5bfbe8eu);
    load_f32_bits(f_poly, 0x3ab4a000u);
    li(reg_minexp, static_cast<int64_t>(0xC1000000u));
    li(reg_maxexp, static_cast<int64_t>(0x3F800000u));

    vsetvli(reg_vl, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    vfmv_v_f(v_acc, f_zero);

    Label loop, finish;
    L(loop);
    beqz(reg_len, finish);

    // tu (not ta): on a short final vector v_poly's tail is stale; under tu the
    // accumulator update vfadd_vv(v_acc, v_acc, v_poly) keeps v_acc's tail, so
    // the closing vfredosum sums only the valid lanes.
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4, VTA::tu, VMA::ma);
    vle32_v(v_x, reg_src);
    slli(reg_bytes, reg_vl, 2);
    add(reg_src, reg_src, reg_bytes);

    vfsub_vf(v_x, v_x, f_sub);
    vfmv_v_f(v_bias, f_round);
    vmflt_vf(v_mask, v_x, f_lower);
    vfmerge_vfm(v_x, v_x, f_lower);
    vmfgt_vf(v_mask, v_x, f_upper);
    vfmerge_vfm(v_x, v_x, f_upper);
    vfmacc_vf(v_bias, f_log2_recip, v_x);
    vfsub_vf(v_tmpv, v_bias, f_round);
    vfmacc_vf(v_x, f_log2_high, v_tmpv);
    vfmacc_vf(v_x, f_log2_low, v_tmpv);
    vfmv_v_f(v_poly, f_poly);
    load_f32_bits(f_poly_coeff, 0x3c092f6eu);
    vfmul_vv(v_poly, v_poly, v_x);
    vfadd_vf(v_poly, v_poly, f_poly_coeff);
    load_f32_bits(f_poly_coeff, 0x3d2aadadu);
    vfmul_vv(v_poly, v_poly, v_x);
    vfadd_vf(v_poly, v_poly, f_poly_coeff);
    load_f32_bits(f_poly_coeff, 0x3e2aaa28u);
    vfmul_vv(v_poly, v_poly, v_x);
    vfadd_vf(v_poly, v_poly, f_poly_coeff);
    load_f32_bits(f_poly_coeff, 0x3efffffbu);
    vfmul_vv(v_poly, v_poly, v_x);
    vfadd_vf(v_poly, v_poly, f_poly_coeff);
    load_f32_bits(f_poly_coeff, 0x3f800000u);
    vfmul_vv(v_poly, v_poly, v_x);
    vfadd_vf(v_poly, v_poly, f_poly_coeff);
    vsll_vi(v_bias, v_bias, 23);
    vmin_vx(v_tmpv, v_bias, reg_maxexp);
    vmax_vx(v_tmpv, v_tmpv, reg_minexp);
    vsub_vv(v_bias, v_bias, v_tmpv);
    vadd_vx(v_bias, v_bias, reg_maxexp);
    vadd_vx(v_tmpv, v_tmpv, reg_maxexp);
    vfmul_vv(v_x, v_x, v_bias);
    vfmadd_vv(v_poly, v_x, v_bias);
    vfmul_vv(v_poly, v_poly, v_tmpv);
    vfadd_vv(v_acc, v_acc, v_poly);

    if (store_exp_) {
        vse32_v(v_poly, reg_dst);
        add(reg_dst, reg_dst, reg_bytes);
    }
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(finish);
    vsetvli(reg_vl, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    vfmv_v_f(v_red, f_zero);
    vfredosum_vs(v_red, v_acc, v_red);
    vfmv_f_s(f_sum, v_red);
    fsw(f_sum, reg_sum, 0);
    ret();
#else
    ret();
#endif
}

void jit_rvv_softmax_f32_exp_sub_sum(const float *src, float *dst, dim_t len,
        float sub, float *sum, bool store_exp) {
    const jit_rvv_softmax_f32_exp_sub_sum_kernel_t::call_params_t p {
            src, dst, len, sub, sum};
    if (store_exp)
        dispatch_f32_exp_sub_sum<true>(&p);
    else
        dispatch_f32_exp_sub_sum<false>(&p);
}

jit_rvv_softmax_inner_kernel_t::jit_rvv_softmax_inner_kernel_t(bool apply,
        bool logsoftmax, bool inf_as_zero, int lmul, bool axis_vectorized,
        bool packed_axis)
    : jit_generator_t("jit_rvv_softmax_inner_kernel")
    , apply_(apply)
    , logsoftmax_(logsoftmax)
    , inf_as_zero_(inf_as_zero)
    , axis_vectorized_(axis_vectorized)
    , packed_axis_(packed_axis)
    , lmul_(lmul) {
    assert(lmul == 1 || lmul == 2 || lmul == 4);
    status_ = create_kernel();
}

void jit_rvv_softmax_inner_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const LMUL data_lmul = lmul_ == 1 ? LMUL::m1
            : lmul_ == 2              ? LMUL::m2
                                      : LMUL::m4;
    const LMUL sum_lmul = lmul_ == 1 ? LMUL::m2
            : lmul_ == 2             ? LMUL::m4
                                     : LMUL::m8;
    const Reg src = a1, dst = a2, axis = a3, stride = a4, columns = a5;
    const Reg max_ptr = a6, sum_ptr = a7;
    const Reg vl = t0, src_row = t1, dst_row = t2, rows_left = t3;
    const Reg minexp = t4, maxexp = t5, imm = t6;
    const FReg zero = ft0, lower = ft1, upper = ft2, round = ft3;
    const FReg log2_recip = ft4, log2_high = ft5, log2_low = ft6;
    const FReg poly = ft7, one = ft9, neg_inf = fa0;
    const FReg coefficients[] = {ft8, fa1, fa2, fa3, one};
    const VReg mask(0), maximum(4), x(8), bias(12), p(16), tmp(20), sum(24);
    auto load_bits = [&](const FReg &f, uint32_t bits) {
        li(imm, static_cast<int64_t>(bits));
        fmv_w_x(f, imm);
    };
#define INNER_OFF(field) static_cast<int32_t>(offsetof(call_params_t, field))
    ld(src, a0, INNER_OFF(src));
    ld(dst, a0, INNER_OFF(dst));
    ld(axis, a0, INNER_OFF(axis_len));
    ld(stride, a0, INNER_OFF(stride_bytes));
    ld(columns, a0, INNER_OFF(columns));
    ld(max_ptr, a0, INNER_OFF(max));
    ld(sum_ptr, a0, INNER_OFF(sum));
#undef INNER_OFF
    fmv_w_x(zero, x0);
    load_bits(one, 0x3f800000u);
    if (!apply_) {
        load_bits(neg_inf, 0xff800000u);
        load_bits(lower, utils::bit_cast<uint32_t>(-103.9720840454f));
        load_bits(upper, utils::bit_cast<uint32_t>(88.7762626647950f));
        load_bits(round, 0x4b400000u);
        load_bits(log2_recip, 0x3fb8aa3bu);
        load_bits(log2_high, 0xbf317200u);
        load_bits(log2_low, 0xb5bfbe8eu);
        load_bits(poly, 0x3ab4a000u);
        load_bits(coefficients[0], 0x3c092f6eu);
        load_bits(coefficients[1], 0x3d2aadadu);
        load_bits(coefficients[2], 0x3e2aaa28u);
        load_bits(coefficients[3], 0x3efffffbu);
        li(minexp, static_cast<int64_t>(0xc1000000u));
        li(maxexp, static_cast<int64_t>(0x3f800000u));
    }

    auto emit_exp = [&]() {
        // Shared polynomial for both orientations; keep the same coefficients
        // and evaluation order as the existing f32 softmax kernel.
        vfmv_v_f(bias, round);
        vmflt_vf(mask, x, lower);
        vfmerge_vfm(x, x, lower);
        vmfgt_vf(mask, x, upper);
        vfmerge_vfm(x, x, upper);
        vfmacc_vf(bias, log2_recip, x);
        vfsub_vf(tmp, bias, round);
        vfmacc_vf(x, log2_high, tmp);
        vfmacc_vf(x, log2_low, tmp);
        vfmv_v_f(p, poly);
        for (const auto &coeff : coefficients) {
            vfmul_vv(p, p, x);
            vfadd_vf(p, p, coeff);
        }
        vsll_vi(bias, bias, 23);
        vmin_vx(tmp, bias, maxexp);
        vmax_vx(tmp, tmp, minexp);
        vsub_vv(bias, bias, tmp);
        vadd_vx(bias, bias, maxexp);
        vadd_vx(tmp, tmp, maxexp);
        vfmul_vv(x, x, bias);
        vfmadd_vv(p, x, bias);
        vfmul_vv(p, p, tmp);
        if (inf_as_zero_ && !logsoftmax_) {
            vmfeq_vf(mask, maximum, neg_inf);
            vfmerge_vfm(p, p, zero);
        }
    };

    if (axis_vectorized_) {
        // A call handles one column. Actual VL determines the byte advance;
        // Packed calls use unit-stride memory operations on caller scratch.
        const FReg scalar_max = ft10, scalar_sum = ft11;
        auto load_axis = [&]() {
            // The softmax apply pass reads per-thread contiguous exponentials.
            if (packed_axis_ || (apply_ && !logsoftmax_))
                vle32_v(x, src_row);
            else
                vlse32_v(x, src_row, stride);
        };
        auto store_axis = [&](const VReg &value) {
            if (packed_axis_ || (!apply_ && !logsoftmax_))
                vse32_v(value, dst_row);
            else
                vsse32_v(value, dst_row, stride);
        };
        if (!apply_) {
            mv(src_row, src);
            mv(rows_left, axis);
            vsetvli(vl, rows_left, SEW::e32, data_lmul, VTA::ta, VMA::ma);
            vfmv_s_f(maximum, neg_inf);
            Label max_loop;
            L(max_loop);
            vsetvli(vl, rows_left, SEW::e32, data_lmul, VTA::ta, VMA::ma);
            load_axis();
            vfredmax_vs(maximum, x, maximum);
            mul(imm, vl, stride);
            add(src_row, src_row, imm);
            sub(rows_left, rows_left, vl);
            bnez(rows_left, max_loop);
            vfmv_f_s(scalar_max, maximum);
            fsw(scalar_max, max_ptr);
            // Keep f64 partial sums in every lane and reduce once per column.
            // All lanes are initialized; the last partial chunk preserves
            // inactive accumulator lanes until the VLMAX reduction.
            vsetvli(vl, x0, SEW::e64, sum_lmul, VTA::tu, VMA::ma);
            vmv_v_i(sum, 0);
            mv(src_row, src);
            mv(dst_row, dst);
            mv(rows_left, axis);
            Label exp_loop;
            L(exp_loop);
            vsetvli(vl, rows_left, SEW::e32, data_lmul, VTA::tu, VMA::ma);
            load_axis();
            vfmv_v_f(maximum, scalar_max);
            vfsub_vv(x, x, maximum);
            emit_exp();
            vfwadd_wv(sum, sum, p);
            if (!logsoftmax_) store_axis(p);
            mul(imm, vl, stride);
            add(src_row, src_row, imm);
            if (!logsoftmax_) slli(imm, vl, 2);
            add(dst_row, dst_row, imm);
            sub(rows_left, rows_left, vl);
            bnez(rows_left, exp_loop);
            vsetvli(vl, x0, SEW::e64, sum_lmul, VTA::ta, VMA::ma);
            fmv_d_x(scalar_sum, x0);
            vfmv_s_f(maximum, scalar_sum);
            vfredusum_vs(maximum, sum, maximum);
            vfmv_f_s(scalar_sum, maximum);
            fcvt_s_d(scalar_sum, scalar_sum);
            fsw(scalar_sum, sum_ptr);
        } else {
            flw(scalar_max, max_ptr);
            flw(scalar_sum, sum_ptr);
            if (!logsoftmax_) {
                // One reciprocal per column, rather than a vector division
                // for every axis chunk. Match the zero-sum convention.
                Label nonzero;
                feq_s(imm, scalar_sum, zero);
                beqz(imm, nonzero);
                fsgnj_s(scalar_sum, one, one);
                L(nonzero);
                fdiv_s(scalar_sum, one, scalar_sum);
            }
            mv(src_row, src);
            mv(dst_row, dst);
            mv(rows_left, axis);
            Label apply_loop;
            L(apply_loop);
            vsetvli(vl, rows_left, SEW::e32, data_lmul, VTA::ta, VMA::ma);
            load_axis();
            if (logsoftmax_) {
                vfsub_vf(x, x, scalar_max);
                vfsub_vf(x, x, scalar_sum);
            } else {
                vfmul_vf(x, x, scalar_sum);
            }
            store_axis(x);
            mul(imm, vl, stride);
            add(dst_row, dst_row, imm);
            if (!logsoftmax_) slli(imm, vl, 2);
            add(src_row, src_row, imm);
            sub(rows_left, rows_left, vl);
            bnez(rows_left, apply_loop);
        }
        ret();
        return;
    }

    Label column_loop, done;
    L(column_loop);
    beqz(columns, done);
    // Strip-mine in the kernel as well as the caller: the caller's tile can
    // exceed the executing core's VLMAX on a heterogeneous system.
    vsetvli(vl, columns, SEW::e32, data_lmul, VTA::ta, VMA::ma);
    if (apply_) {
        if (logsoftmax_) {
            // Subtract max before log(sum) to avoid rounding their sum when
            // the input has a large common offset.
            vle32_v(maximum, max_ptr);
            vle32_v(sum, sum_ptr);
        } else {
            vle32_v(sum, sum_ptr);
            vmfeq_vf(mask, sum, zero);
            vfmerge_vfm(sum, sum, one);
            vfrdiv_vf(sum, sum, one);
        }
        mv(src_row, src);
        mv(dst_row, dst);
        mv(rows_left, axis);
        Label apply_loop;
        L(apply_loop);
        vle32_v(x, src_row);
        if (logsoftmax_) {
            vfsub_vv(x, x, maximum);
            vfsub_vv(x, x, sum);
        } else
            vfmul_vv(x, x, sum);
        vse32_v(x, dst_row);
        add(src_row, src_row, stride);
        add(dst_row, dst_row, stride);
        addi(rows_left, rows_left, -1);
        bnez(rows_left, apply_loop);
    } else {
        vfmv_v_f(maximum, neg_inf);
        mv(src_row, src);
        mv(rows_left, axis);
        Label max_loop;
        L(max_loop);
        vle32_v(x, src_row);
        vfmax_vv(maximum, maximum, x);
        add(src_row, src_row, stride);
        addi(rows_left, rows_left, -1);
        bnez(rows_left, max_loop);
        vse32_v(maximum, max_ptr);

        // Each lane sums a full axis, unlike an axis-vectorized reduction.
        // Widen only the accumulator so long axes do not lose small terms.
        // Doubling SEW and LMUL retains the same lane capacity.
        vsetvli(vl, vl, SEW::e64, sum_lmul, VTA::ta, VMA::ma);
        vmv_v_i(sum, 0);
        vsetvli(vl, vl, SEW::e32, data_lmul, VTA::ta, VMA::ma);
        mv(src_row, src);
        mv(dst_row, dst);
        mv(rows_left, axis);
        Label exp_loop;
        L(exp_loop);
        vle32_v(x, src_row);
        vfsub_vv(x, x, maximum);
        emit_exp();
        vfwadd_wv(sum, sum, p);
        if (!logsoftmax_) vse32_v(p, dst_row);
        add(src_row, src_row, stride);
        add(dst_row, dst_row, stride);
        addi(rows_left, rows_left, -1);
        bnez(rows_left, exp_loop);
        vfncvt_f_f_w(p, sum);
        vse32_v(p, sum_ptr);
    }
    slli(imm, vl, 2);
    add(src, src, imm);
    add(dst, dst, imm);
    add(max_ptr, max_ptr, imm);
    add(sum_ptr, sum_ptr, imm);
    sub(columns, columns, vl);
    j_(column_loop);
    L(done);
#endif
    ret();
}

namespace {
template <int lmul, bool apply, bool log, bool inf, bool axis, bool packed>
const jit_rvv_softmax_inner_kernel_t &get_kernel() {
    static const jit_rvv_softmax_inner_kernel_t kernel(
            apply, log, inf, lmul, axis, packed);
    return kernel;
}
template <int lmul, bool axis = false, bool packed = false>
const jit_rvv_softmax_inner_kernel_t &get_kernel(
        bool apply, bool logsoftmax, bool inf_as_zero) {
    if (apply)
        return logsoftmax
                ? get_kernel<lmul, true, true, false, axis, packed>()
                : get_kernel<lmul, true, false, false, axis, packed>();
    if (logsoftmax) return get_kernel<lmul, false, true, false, axis, packed>();
    return inf_as_zero ? get_kernel<lmul, false, false, true, axis, packed>()
                       : get_kernel<lmul, false, false, false, axis, packed>();
}
} // namespace

const jit_rvv_softmax_inner_kernel_t &get_softmax_inner_kernel(bool apply,
        bool logsoftmax, bool inf_as_zero, int lmul, bool axis_vectorized,
        bool packed_axis) {
    if (packed_axis) {
        assert(axis_vectorized);
        return get_kernel<4, true, true>(apply, logsoftmax, inf_as_zero);
    }
    if (axis_vectorized)
        return get_kernel<4, true>(apply, logsoftmax, inf_as_zero);
    switch (lmul) {
        case 1: return get_kernel<1>(apply, logsoftmax, inf_as_zero);
        case 2: return get_kernel<2>(apply, logsoftmax, inf_as_zero);
        default:
            assert(lmul == 4);
            return get_kernel<4>(apply, logsoftmax, inf_as_zero);
    }
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
