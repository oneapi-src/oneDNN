/*******************************************************************************
* Copyright 2025 FUJITSU LIMITED
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

#include <cassert>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ranges>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/tag_traits.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/aarch64/matmul/jit_int8_matmul.hpp"

#define GET_OFF(field) (uint32_t) offsetof(dyn_params_t, field)

#define LDR_IMM(reg, addr, off) \
    { \
        const uint64_t IMM12_MASK = ~uint64_t(0xfff); \
        if ((off & IMM12_MASK) == 0) { \
            ldr(reg, ptr(addr, off)); \
        } else { \
            add_imm(X_DEFAULT_ADDR, addr, off, X_TMP_0); \
            ldr(reg, ptr(X_DEFAULT_ADDR)); \
        } \
    }

#define VCHECK_BG(f, msg, ...) \
    VCHECK(primitive, create, dispatch, brgemm_matmul, f, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using namespace data_type;

void jit_dyn_quant_src_kernel_t::reo_A_8x8(int lp, int kt) {
    mov(reg_tmp_1, reg_tmp);
    if (kt > 0) {
        for (int i = 0; i < lp; i++) {
            ld1b(ZRegB(i), prd_ld, ptr(reg_tmp_1));
            add_imm(reg_tmp_1, reg_tmp_1, dyn.K, X_TMP_0);
            st1b(ZRegB(i), prd_st, ptr(reg_dst));
            add_imm(reg_dst, reg_dst, dyn.k_blk, X_TMP_0);
        }
        for (int i = 0; i < dyn.m_blk - lp; i++) {
            mov(ZRegB(i), 0);
            st1b(ZRegB(i), prd_st, ptr(reg_dst));
            add_imm(reg_dst, reg_dst, dyn.k_blk, X_TMP_0);
        }
    } else {
        for (int i = 0; i < lp; i++) {
            ldr(DReg(i), ptr(reg_tmp_1));
            add_imm(reg_tmp_1, reg_tmp_1, dyn.K, X_TMP_0);
            str(DReg(i), ptr(reg_dst));
            add_imm(reg_dst, reg_dst, dyn.k_blk, X_TMP_0);
        }
        for (int i = 0; i < dyn.m_blk - lp; i++) {
            mov(ZRegB(i), 0);
            st1b(ZRegB(i), prd_st, ptr(reg_dst));
            add_imm(reg_dst, reg_dst, dyn.k_blk, X_TMP_0);
        }
    }
}

void jit_dyn_quant_src_kernel_t::reo_B_8x24(int lp, int nt) {
    auto p = (nt > 0) ? prd_p3 : prd_ld;
    mov(reg_tmp, reg_aux_a);
    for (int i = 0; i < lp; i++) {
        ld1b(ZRegB(i), p, ptr(reg_tmp));
        add_imm(reg_tmp, reg_tmp, dyn.N, X_TMP_4);
    }
    for (int i = lp; i < dyn.k_blk; i++) {
        mov(ZRegB(i), 0);
    }

    zip2(ZRegB(8), ZRegB(0), ZRegB(1));
    zip1(ZRegB(0), ZRegB(0), ZRegB(1));
    zip2(ZRegB(10), ZRegB(2), ZRegB(3));
    zip1(ZRegB(2), ZRegB(2), ZRegB(3));
    zip2(ZRegB(12), ZRegB(4), ZRegB(5));
    zip1(ZRegB(4), ZRegB(4), ZRegB(5));
    zip2(ZRegB(14), ZRegB(6), ZRegB(7));
    zip1(ZRegB(6), ZRegB(6), ZRegB(7));

    zip2(ZRegH(1), ZRegH(0), ZRegH(2));
    zip1(ZRegH(0), ZRegH(0), ZRegH(2));
    zip2(ZRegH(5), ZRegH(4), ZRegH(6));
    zip1(ZRegH(4), ZRegH(4), ZRegH(6));
    zip1(ZRegH(8), ZRegH(8), ZRegH(10));
    zip1(ZRegH(12), ZRegH(12), ZRegH(14));

    zip2(ZRegS(2), ZRegS(0), ZRegS(4));
    zip1(ZRegS(0), ZRegS(0), ZRegS(4));
    zip2(ZRegS(6), ZRegS(1), ZRegS(5));
    zip1(ZRegS(1), ZRegS(1), ZRegS(5));
    zip2(ZRegS(10), ZRegS(8), ZRegS(12));
    zip1(ZRegS(8), ZRegS(8), ZRegS(12));

    str(ZReg(0), ptr(reg_aux_b, 0, MUL_VL));
    str(ZReg(2), ptr(reg_aux_b, 1, MUL_VL));
    str(ZReg(1), ptr(reg_aux_b, 2, MUL_VL));
    str(ZReg(6), ptr(reg_aux_b, 3, MUL_VL));
    str(ZReg(8), ptr(reg_aux_b, 4, MUL_VL));
    str(ZReg(10), ptr(reg_aux_b, 5, MUL_VL));

    add_imm(reg_aux_b, reg_aux_b, dyn.n_blk * dyn.k_blk, X_TMP_4);
}

void jit_dyn_quant_src_kernel_t::calc_scales(int lp, int tail) {
    for (int i = 0; i < lp; i++) {
        auto p = (tail > 0 && i == lp - 1) ? prd_ld : P_ALL_ONE;
        ld1w(z2.s, p, ptr(reg_src, i, MUL_VL));
        fmax(z0.s, p / T_m, z2.s);
        fmin(z1.s, p / T_m, z2.s);
    }
}
void jit_dyn_quant_src_kernel_t::gen_scl() {
    int len = get_sve_length() / f32_dt_sz;
    int unroll = 8;
    Label L_tail, L_end, m_loop, last_m;
    dim_t noe = dyn.M * dyn.K * dyn.B;
    int res_m = div_up((noe % (unroll * len)), unroll);
    int k_tail = (noe % (unroll * len)) % len;
    if (k_tail == 0) k_tail = len;
    set_preg(prd_ld.s, k_tail, X_TMP_0, X_TMP_1);

    LDR_IMM(reg_tmp_2, reg_param, GET_OFF(nm));
    ldr(WReg(reg_m_loop.getIdx()), ptr(reg_tmp_2));

    LDR_IMM(reg_tmp_2, reg_param, GET_OFF(tl));
    ldr(WReg(reg_tail.getIdx()), ptr(reg_tmp_2));

    ld1rw(z0.s, P_ALL_ONE, ptr(reg_src));
    ld1rw(z1.s, P_ALL_ONE, ptr(reg_src));

    cmp(reg_m_loop, 1);
    b(EQ, last_m);
    L(m_loop);
    calc_scales(unroll, 0);
    add_imm(reg_src, reg_src, (unroll * len) * f32_dt_sz, X_TMP_0);
    sub(reg_m_loop, reg_m_loop, 1);
    cmp(reg_m_loop, 1);
    b(GT, m_loop);
    b(EQ, last_m);

    L(last_m);
    cmp(reg_tail, 1);
    b(EQ, L_tail);
    calc_scales(unroll, 0);
    bl(L_end);
    L(L_tail);
    calc_scales(res_m, 1);
    L(L_end);

    auto p = (noe > len) ? P_ALL_ONE : prd_ld;
    fmaxv(s7, p, z0.s);
    fminv(s8, p, z1.s);
    str(s7, ptr(reg_max));
    str(s8, ptr(reg_min));
}
void jit_dyn_quant_src_kernel_t::mul_scl(int lp, int tl) {
    auto p = (tl > 0) ? prd_ld : P_ALL_ONE;
    for (int i = 0; i < lp; i++) {
        if (i == lp - 1 && tl > 0)
            ld1w(ZRegS(i), p, ptr(reg_tmp, i, MUL_VL));
        else
            ld1w(ZRegS(i), P_ALL_ONE, ptr(reg_tmp, i, MUL_VL));
    }

    for (int i = 0; i < lp; i++) {
        fmul(ZRegS(i), P_ALL_ONE, z29.s);
    }
    for (int i = 0; i < lp; i++) {
        fadd(ZRegS(i), ZRegS(i), z28.s);
    }
    for (int i = 0; i < lp; i++) {
        fcvtzs(ZRegS(i), P_ALL_ONE, ZRegS(i));
    }
    for (int i = 0; i < lp; i++) {
        smin(ZRegS(i), P_ALL_ONE, z31.s);
        smax(ZRegS(i), P_ALL_ONE, z30.s);
    }
    for (int i = 0; i < lp; i++) {
        uzp1(ZRegH(i), ZRegH(i), ZRegH(i));
        uzp1(ZRegB(i), ZRegB(i), ZRegB(i));
    }
    for (int i = 0; i < lp; i++) {
        if (i == lp - 1 && tl > 0)
            st1b(ZRegB(i), prd_st, ptr(reg_tmp_1));
        else {
            str(DReg(i), ptr(reg_tmp_1));
            add_imm(reg_tmp_1, reg_tmp_1, 8, X_TMP_0);
        }
    }
}

void jit_dyn_quant_src_kernel_t::gen_src() {
    Label L_tail, L_end, m_loop, last_m;
    int len = get_sve_length() / f32_dt_sz;
    int unroll = 8;
    dim_t noe = dyn.M * dyn.K * dyn.B;
    int res_m = div_up((noe % (unroll * len)), unroll);
    int k_tail = (noe % (unroll * len)) % len;
    if (k_tail == 0) k_tail = len;
    set_preg(prd_ld.s, k_tail, X_TMP_0, X_TMP_1);
    set_preg(prd_st.b, k_tail, X_TMP_0, X_TMP_1);
    if (dyn.is_s8) {
        dupm(z31.s, 127);
        dupm(z30.s, -127);
    } else {
        dupm(z31.s, 255);
        dup(z30.s, 0);
    }
    LDR_IMM(reg_tmp_2, reg_param, GET_OFF(nm));
    ldr(WReg(reg_m_loop.getIdx()), ptr(reg_tmp_2));

    LDR_IMM(reg_tmp_2, reg_param, GET_OFF(tl));
    ldr(WReg(reg_tail.getIdx()), ptr(reg_tmp_2));

    ld1rw(z29.s, P_ALL_ONE, ptr(reg_max));
    ld1rw(z28.s, P_ALL_ONE, ptr(reg_min));
    scvtf(z28.s, P_ALL_ONE, z28.s);

    mov(reg_tmp, reg_src);

    cmp(reg_m_loop, 1);
    b(EQ, last_m);
    L(m_loop);
    mov(reg_tmp_1, reg_dst);
    mul_scl(unroll, 0);
    add_imm(reg_tmp, reg_tmp, (unroll * len) * f32_dt_sz, X_TMP_0);
    add_imm(reg_dst, reg_dst, (unroll * len), X_TMP_0);
    sub(reg_m_loop, reg_m_loop, 1);
    cmp(reg_m_loop, 1);
    b(EQ, last_m);
    b(GT, m_loop);

    L(last_m);
    mov(reg_tmp_1, reg_dst);
    cmp(reg_tail, 1);
    b(EQ, L_tail);
    mul_scl(unroll, 0);
    bl(L_end);
    L(L_tail);
    mul_scl(res_m, 1);
    L(L_end);

    // set $pc=0xfffff5b90060
}

void jit_dyn_quant_src_kernel_t::gen_reo_a() {

    int ktl = (dyn.ktail) ? dyn.ktail : dyn.k_blk;

    set_preg(prd_ld.b, ktl, X_TMP_0, X_TMP_1);
    set_preg(prd_st.b, dyn.k_blk, X_TMP_0, X_TMP_1);

    int lp = (dyn.mtail) ? dyn.mtail : dyn.m_blk;

    Label m_loop, last_m, m_end, k_loop, last_k, k_end, k_loop_1, last_k_1,
            k_end_1;

    LDR_IMM(reg_max, reg_param, GET_OFF(nk));
    LDR_IMM(reg_min, reg_param, GET_OFF(nm));

    LDR_IMM(reg_tmp_2, reg_param, GET_OFF(tl));
    ldr(WReg(reg_tail.getIdx()), ptr(reg_tmp_2));

    LDR_IMM(reg_tmp_2, reg_param, GET_OFF(mtl));
    ldr(WReg(reg_m_tail.getIdx()), ptr(reg_tmp_2));

    ldr(WReg(reg_m_loop.getIdx()), ptr(reg_min));

    cmp(reg_m_loop, 1);
    b(EQ, last_m);
    L(m_loop);
    ldr(WReg(reg_k_loop.getIdx()), ptr(reg_max));
    mov(reg_tmp, reg_src);
    cmp(reg_k_loop, 1);
    b(EQ, last_k);
    L(k_loop);
    reo_A_8x8(dyn.m_blk, 0);
    add_imm(reg_tmp, reg_tmp, dyn.k_blk, X_TMP_0);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_k_loop, 1);
    b(GT, k_loop);
    b(LT, k_end);
    L(last_k);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_tail, 0);
    b(EQ, k_loop);
    reo_A_8x8(dyn.m_blk, 1);
    L(k_end);
    add_imm(reg_src, reg_src, dyn.K * dyn.m_blk, X_TMP_0);
    sub(reg_m_loop, reg_m_loop, 1);
    cmp(reg_m_loop, 1);
    b(GT, m_loop);
    b(LT, m_end);

    L(last_m);
    sub(reg_m_loop, reg_m_loop, 1);
    cmp(reg_m_tail, 0);
    b(EQ, m_loop);
    ldr(WReg(reg_k_loop.getIdx()), ptr(reg_max));
    mov(reg_tmp, reg_src);
    cmp(reg_k_loop, 1);
    b(EQ, last_k_1);
    L(k_loop_1);
    reo_A_8x8(lp, 0);
    add_imm(reg_tmp, reg_tmp, dyn.k_blk, X_TMP_0);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_k_loop, 1);
    b(GT, k_loop_1);
    b(LT, k_end_1);
    L(last_k_1);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_tail, 0);
    b(EQ, k_loop_1);
    reo_A_8x8(lp, 1);
    L(k_end_1);
    L(m_end);
}

void jit_dyn_quant_src_kernel_t::gen_reo_b() {

    int lp = (dyn.ktail > 0) ? dyn.ktail : 8;

    set_preg(prd_ld.b, 24, X_TMP_4, X_TMP_1);
    set_preg(prd_p3.b, dyn.ntail, X_TMP_4, X_TMP_1);

    LDR_IMM(reg_max, reg_param, GET_OFF(nn));
    LDR_IMM(reg_min, reg_param, GET_OFF(nk));

    LDR_IMM(reg_tmp_2, reg_param, GET_OFF(tl));
    ldr(WReg(reg_tail.getIdx()), ptr(reg_tmp_2));

    LDR_IMM(reg_tmp_2, reg_param, GET_OFF(ntl));
    ldr(WReg(reg_n_tail.getIdx()), ptr(reg_tmp_2));

    ldr(WReg(reg_n_loop.getIdx()), ptr(reg_max));
    ldr(WReg(reg_k_loop.getIdx()), ptr(reg_min));

    mov(reg_aux_a, reg_src);
    mov(reg_aux_b, reg_dst);

    Label n_loop, last_n, n_end, k_loop, last_k, k_end, k_loop_1, last_k_1,
            k_end_1;

    cmp(reg_n_loop, 1);
    b(EQ, last_n);
    L(n_loop);
    ldr(WReg(reg_k_loop.getIdx()), ptr(reg_min));
    mov(reg_aux_a, reg_src);
    cmp(reg_k_loop, 1);
    b(EQ, last_k);
    L(k_loop);
    reo_B_8x24(8, 0);
    add_imm(reg_aux_a, reg_aux_a, 8 * dyn.N, X_TMP_4);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_k_loop, 1);
    b(GT, k_loop);
    b(LT, k_end);
    L(last_k);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_tail, 0);
    b(EQ, k_loop);
    reo_B_8x24(lp, 0);
    L(k_end);
    add_imm(reg_src, reg_src, 24, X_TMP_4);
    sub(reg_n_loop, reg_n_loop, 1);
    cmp(reg_n_loop, 1);
    b(GT, n_loop);
    b(LT, n_end);

    L(last_n);
    sub(reg_n_loop, reg_n_loop, 1);
    cmp(reg_n_tail, 0);
    b(EQ, n_loop);
    ldr(WReg(reg_k_loop.getIdx()), ptr(reg_min));
    mov(reg_aux_a, reg_src);
    cmp(reg_k_loop, 1);
    b(EQ, last_k_1);
    L(k_loop_1);
    reo_B_8x24(8, dyn.ntail);
    add_imm(reg_aux_a, reg_aux_a, 8 * dyn.N, X_TMP_4);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_k_loop, 1);
    b(GT, k_loop_1);
    b(LT, k_end_1);
    L(last_k_1);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_tail, 0);
    b(EQ, k_loop_1);
    reo_B_8x24(lp, dyn.ntail);
    L(k_end_1);
    L(n_end);
}

void jit_dyn_quant_src_kernel_t::generate() {

    preamble();

    if (dyn.get_min_max == 1) {
        LDR_IMM(reg_src, reg_param, GET_OFF(dyn_src));
        LDR_IMM(reg_max, reg_param, GET_OFF(max));
        LDR_IMM(reg_min, reg_param, GET_OFF(min));
        gen_scl();
    } else if (dyn.cal_src == 1) {
        LDR_IMM(reg_src, reg_param, GET_OFF(dyn_src));
        LDR_IMM(reg_dst, reg_param, GET_OFF(dst));
        LDR_IMM(reg_max, reg_param, GET_OFF(src_scls));
        LDR_IMM(reg_min, reg_param, GET_OFF(zp));
        gen_src();
    } else if (dyn.reorder_a == 1) {
        LDR_IMM(reg_src, reg_param, GET_OFF(src));
        LDR_IMM(reg_dst, reg_param, GET_OFF(dst));
        gen_reo_a();
    } else if (dyn.reorder_b == 1) {
        LDR_IMM(reg_src, reg_param, GET_OFF(src));
        LDR_IMM(reg_dst, reg_param, GET_OFF(dst));
        gen_reo_b();
    }

    postamble();
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
