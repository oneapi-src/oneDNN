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

#include "common/math_utils.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/matmul/jit_int8_matmul_utils.hpp"

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
namespace matmul {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using namespace data_type;

void jit_int8_matmul_utils_kernel_t::reo_A_8x8(int lp, int kt) {
    mov(reg_tmp_1, reg_tmp);
    if (kt > 0) {
        for (int i = 0; i < lp; i++) {
            ld1b(ZRegB(i), prd_ld, ptr(reg_tmp_1));
            add_imm(reg_tmp_1, reg_tmp_1, dyn_.K, X_TMP_0);
            st1b(ZRegB(i), prd_st, ptr(reg_dst));
            add_imm(reg_dst, reg_dst, dyn_.k_blk, X_TMP_0);
        }
        for (int i = 0; i < dyn_.m_blk - lp; i++) {
            mov(ZRegB(i), 0);
            st1b(ZRegB(i), prd_st, ptr(reg_dst));
            add_imm(reg_dst, reg_dst, dyn_.k_blk, X_TMP_0);
        }
    } else {
        for (int i = 0; i < lp; i++) {
            ldr(DReg(i), ptr(reg_tmp_1));
            add_imm(reg_tmp_1, reg_tmp_1, dyn_.K, X_TMP_0);
            str(DReg(i), ptr(reg_dst));
            add_imm(reg_dst, reg_dst, dyn_.k_blk, X_TMP_0);
        }
        for (int i = 0; i < dyn_.m_blk - lp; i++) {
            mov(ZRegB(i), 0);
            st1b(ZRegB(i), prd_st, ptr(reg_dst));
            add_imm(reg_dst, reg_dst, dyn_.k_blk, X_TMP_0);
        }
    }
}

void jit_int8_matmul_utils_kernel_t::reo_B_8x24(int lp, int nt) {
    auto p = (nt > 0) ? prd_p3 : prd_ld;
    mov(reg_tmp, reg_aux_a);
    for (int i = 0; i < lp; i++) {
        ld1b(ZRegB(i), p, ptr(reg_tmp));
        add_imm(reg_tmp, reg_tmp, dyn_.N, X_TMP_4);
    }
    for (int i = lp; i < dyn_.k_blk; i++) {
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

    add_imm(reg_aux_b, reg_aux_b, dyn_.n_blk * dyn_.k_blk, X_TMP_4);
}

void jit_int8_matmul_utils_kernel_t::gen_reo_a() {

    int ktl = (dyn_.ktail) ? dyn_.ktail : dyn_.k_blk;

    set_preg(prd_ld.b, ktl, X_TMP_0, X_TMP_1);
    set_preg(prd_st.b, dyn_.k_blk, X_TMP_0, X_TMP_1);

    int lp = (dyn_.mtail) ? dyn_.mtail : dyn_.m_blk;

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
    reo_A_8x8(dyn_.m_blk, 0);
    add_imm(reg_tmp, reg_tmp, dyn_.k_blk, X_TMP_0);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_k_loop, 1);
    b(GT, k_loop);
    b(LT, k_end);
    L(last_k);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_tail, 0);
    b(EQ, k_loop);
    reo_A_8x8(dyn_.m_blk, 1);
    L(k_end);
    add_imm(reg_src, reg_src, dyn_.K * dyn_.m_blk, X_TMP_0);
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
    add_imm(reg_tmp, reg_tmp, dyn_.k_blk, X_TMP_0);
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

void jit_int8_matmul_utils_kernel_t::gen_reo_b() {

    int lp = (dyn_.ktail > 0) ? dyn_.ktail : dyn_.k_blk;

    set_preg(prd_ld.b, dyn_.n_blk, X_TMP_4, X_TMP_1);
    set_preg(prd_p3.b, dyn_.ntail, X_TMP_4, X_TMP_1);

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
    reo_B_8x24(dyn_.k_blk, 0);
    add_imm(reg_aux_a, reg_aux_a, dyn_.k_blk * dyn_.N, X_TMP_4);
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
    add_imm(reg_src, reg_src, dyn_.n_blk, X_TMP_4);
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
    reo_B_8x24(dyn_.k_blk, dyn_.ntail);
    add_imm(reg_aux_a, reg_aux_a, dyn_.k_blk * dyn_.N, X_TMP_4);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_k_loop, 1);
    b(GT, k_loop_1);
    b(LT, k_end_1);
    L(last_k_1);
    sub(reg_k_loop, reg_k_loop, 1);
    cmp(reg_tail, 0);
    b(EQ, k_loop_1);
    reo_B_8x24(lp, dyn_.ntail);
    L(k_end_1);
    L(n_end);
}

void jit_int8_matmul_utils_kernel_t::generate() {

    preamble();

    if (dyn_.reorder_a == 1) {
        LDR_IMM(reg_src, reg_param, GET_OFF(src));
        LDR_IMM(reg_dst, reg_param, GET_OFF(dst));
        gen_reo_a();
    } else if (dyn_.reorder_b == 1) {
        LDR_IMM(reg_src, reg_param, GET_OFF(src));
        LDR_IMM(reg_dst, reg_param, GET_OFF(dst));
        gen_reo_b();
    }

    postamble();
}
} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
