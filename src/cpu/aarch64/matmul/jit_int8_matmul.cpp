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

#define GET_OFF(field) (uint32_t) offsetof(call_params_t, field)

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
using namespace dnnl::impl::cpu::matmul;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using namespace data_type;

ZReg jit_int8_matmul_kernel_t::loadb(int ld) {
    return ZReg(ld + 1);
}
ZReg jit_int8_matmul_kernel_t::acc(int bd, int ld) {
    return ZReg(bd * brg.ld_block + ld + brg.ld_block + 1);
}
void jit_int8_matmul_kernel_t::zero_regs() {
    for (int a = 0; a < brg.bd_block / 2; a++)
        for (int b = 0; b < brg.ld_block; b++)
            eor(acc(a, b).d, acc(a, b).d, acc(a, b).d);
}
void jit_int8_matmul_kernel_t::store_regs(int bdb, int ldb, int tail) {
    for (int a = 0; a < bdb; a++) {
        for (int b = 0; b < ldb; b++) {
            if (brg.is_s8)
                scvtf(acc(a, b).s, P_ALL_ONE, acc(a, b).s);
            else
                ucvtf(acc(a, b).s, P_ALL_ONE, acc(a, b).s);
        }
    }

    for (int a = 0; a < bdb; a++) {
        for (int b = 0; b < ldb; b += 2) {
            if (b + 1 < ldb) {
                uzp1(z31.d, acc(a, b).d, acc(a, b + 1).d);
                uzp2(acc(a, b + 1).d, acc(a, b).d, acc(a, b + 1).d);
                mov(acc(a, b).d, z31.d);
            } else {
                uzp1(z31.d, acc(a, b).d, acc(a, b).d);
                uzp2(acc(a, b + 1).d, acc(a, b).d, acc(a, b).d);
                mov(acc(a, b).d, z31.d);
            }
        }
    }

    if (brg.zp_type_a != jit_int8_broadcast_t::none || brg.is_src_f32) {
        for (int b = 0; b < ldb; b += 2) {
            PReg p = (brg.is_n_tail && b >= ldb - 2) ? prd_b : P_ALL_ONE;
            ld1w(z31.s, p, ptr(reg_zp_a, b / 2, MUL_VL));
            for (int a = 0; a < bdb; a++) {
                fsub(acc(a, b).s, acc(a, b).s, z31.s);
                fsub(acc(a, b + 1).s, acc(a, b + 1).s, z31.s);
            }
        }
    }

    if (brg.zp_type_b != jit_int8_broadcast_t::none) {
        int ao = 0;
        if (brg.is_zp_b_int8) {
            mov(reg_tmp_1, reg_zp_aux_b_buf);
            int ilp = (brg.is_n_tail) ? n_blks : 3;
            for (int i = 0; i < ilp; i++) {
                PReg p = (brg.is_n_tail && i == ilp - 1) ? prd_zp_b_tl : prd_8;
                ld1b(ZRegB(i + 1), p, ptr(reg_tmp_1));
                if (brg.zp_b_dt == u8) {
                    uunpklo(ZRegH(i + 1), ZRegB(i + 1));
                    uunpklo(ZRegS(i + 1), ZRegH(i + 1));
                    ucvtf(ZRegS(i + 1), P_ALL_ONE, ZRegS(i + 1));
                } else {
                    sunpklo(ZRegH(i + 1), ZRegB(i + 1));
                    sunpklo(ZRegS(i + 1), ZRegH(i + 1));
                    scvtf(ZRegS(i + 1), P_ALL_ONE, ZRegS(i + 1));
                }
                add_imm(reg_tmp_1, reg_tmp_1, 8, X_TMP_0);
            }
        }
        for (int a = 0; a < bdb; a++) {
            ld1rw(z31.s, P_ALL_ONE, ptr(reg_zp_aux_b, ao * 4));
            ld1rw(z0.s, P_ALL_ONE, ptr(reg_zp_aux_b, (ao + 1) * 4));
            for (int b = 0; b < ldb; b += 2) {
                if (brg.is_zp_b_int8) {
                    fmul(z4.s, z31.s, ZRegS(b / 2 + 1));
                    fmul(z5.s, z0.s, ZRegS(b / 2 + 1));
                    fsub(acc(a, b).s, acc(a, b).s, z4.s);
                    fsub(acc(a, b + 1).s, acc(a, b + 1).s, z5.s);
                } else {
                    fsub(acc(a, b).s, acc(a, b).s, z31.s);
                    fsub(acc(a, b + 1).s, acc(a, b + 1).s, z0.s);
                }
            }
            ao += 2;
        }
    }

    if (brg.is_src_f32) {
        for (int b = 0; b < ldb; b += 2) {
            ld1rw(z31.s, P_ALL_ONE, ptr(reg_dyn_scl));
            for (int a = 0; a < bdb; a++) {
                fmul(acc(a, b).s, acc(a, b).s, z31.s);
                fmul(acc(a, b + 1).s, acc(a, b + 1).s, z31.s);
            }
        }
    }

    if (brg.with_scales) {
        for (int b = 0; b < ldb; b += 2) {
            PReg p = (brg.is_n_tail && b >= ldb - 2) ? prd_b : P_ALL_ONE;
            if (brg.is_oc_scales) {
                ld1w(z31.s, p, ptr(reg_scales, b / 2, MUL_VL));
            } else {
                ld1w(z31.s, p, ptr(reg_scales));
            }

            for (int a = 0; a < bdb; a++) {
                fmul(acc(a, b).s, acc(a, b).s, z31.s);
                fmul(acc(a, b + 1).s, acc(a, b + 1).s, z31.s);
            }
        }
    }

    if (brg.is_bias) {
        for (int b = 0; b < ldb; b += 2) {
            PReg p = (brg.is_n_tail && b >= ldb - 2) ? prd_b : P_ALL_ONE;
            ld1w(z31.s, p, ptr(reg_bias, b / 2, MUL_VL));
            for (int a = 0; a < bdb; a++) {
                fadd(acc(a, b).s, acc(a, b).s, z31.s);
                fadd(acc(a, b + 1).s, acc(a, b + 1).s, z31.s);
            }
        }
    }

    if (brg.with_dst_scales) {
        ld1rw(z31.s, P_ALL_ONE, ptr(reg_aux_scales));
        for (int b = 0; b < ldb; b += 2) {
            for (int a = 0; a < bdb; a++) {
                fmul(acc(a, b).s, acc(a, b).s, z31.s);
                fmul(acc(a, b + 1).s, acc(a, b + 1).s, z31.s);
            }
        }
    }

    if (brg.zp_type_c != jit_int8_broadcast_t::none) {
        LDR_IMM(reg_zp_val_c, reg_param, GET_OFF(dst_zero_point));
        ldr(W_TMP_0, ptr(reg_zp_val_c));
        dup(z0.s, W_TMP_0);
        scvtf(z0.s, P_ALL_ONE, z0.s);
        for (int b = 0; b < ldb; b += 2) {
            for (int a = 0; a < bdb; a++) {
                fadd(acc(a, b).s, acc(a, b).s, z0.s);
                fadd(acc(a, b + 1).s, acc(a, b + 1).s, z0.s);
            }
        }
    }

    mov(reg_tmp, reg_aux_c);
    add_imm(reg_tmp_1, reg_aux_c, brg.N * brg.dst_dt_sz, X_TMP_0);
    for (int a = 0; a < bdb; a++) {
        for (int b = 0; b < ldb; b += 2) {
            PReg p = (brg.is_n_tail && b >= ldb - 2) ? prd_st : P_ALL_ONE;
            int vl = b / 2;
            st1w(acc(a, b).s, p, ptr(reg_tmp, vl, MUL_VL));
            if (a >= bdb - 1 && brg.is_m_tail) {
                if (brg.m_tail % 2 == 0)
                    st1w(acc(a, b + 1).s, p, ptr(reg_tmp_1, vl, MUL_VL));
            } else {
                st1w(acc(a, b + 1).s, p, ptr(reg_tmp_1, vl, MUL_VL));
            }
        }
        add_imm(reg_tmp, reg_tmp, 2 * brg.N * brg.dst_dt_sz, X_TMP_0);
        add_imm(reg_tmp_1, reg_tmp_1, 2 * brg.N * brg.dst_dt_sz, X_TMP_0);
    }
}

void jit_int8_matmul_kernel_t::microkernel(
        int rdb, int bdb, int ldb, int tail) {
    int a_off = 0, rd, ld, bd;
    mov(reg_tmp, reg_aux_b);
    for (rd = 0; rd < rdb; rd++) {
        int ao = 0;

        for (ld = 0; ld < ldb; ld++) {
            PReg p = (brg.is_n_tail && ld == ldb - 1) ? prd_ld : P_ALL_ONE;
            ld1b(loadb(ld).b, p, ptr(reg_tmp, ld, MUL_VL));
        }
        for (bd = 0; bd < bdb; bd++) {
            add_imm(X_DEFAULT_ADDR, reg_aux_a, a_off + ao, X_TMP_0);
            ld1rqb(z0.b, P_ALL_ONE, ptr(X_DEFAULT_ADDR));
            ao += brg.m_blk * 2;

            for (ld = 0; ld < ldb; ld++) {
                if (brg.is_s8)
                    smmla(acc(bd, ld).s, z0.b, loadb(ld).b);
                else
                    ummla(acc(bd, ld).s, z0.b, loadb(ld).b);
            }
        }
        a_off += brg.m_blk * brg.k_blk;
        add_imm(reg_tmp, reg_tmp, brg.k_blk * brg.n_blk * brg.ld_block,
                X_TMP_0);
    }
}

void jit_int8_matmul_kernel_t::loop_k(int bdb, int ldb, int tail) {
    zero_regs();
    mov(reg_aux_a, reg_aux_a1);
    mov(reg_aux_b, reg_b);
    if (k_full_blks > 0) {
        mov(reg_rd_loop, k_full_blks);
        Label l0;
        L(l0);
        microkernel(brg.rd_block, bdb, ldb, tail);
        add_imm(reg_aux_a, reg_aux_a, brg.m_blk * brg.k_blk * brg.rd_block,
                X_TMP_0);
        add_imm(reg_aux_b, reg_aux_b,
                brg.k_blk * brg.n_blk * brg.ld_block * brg.rd_block, X_TMP_0);
        sub(reg_rd_loop, reg_rd_loop, 1);
        cmp(reg_rd_loop, 0);
        b(GT, l0);
    }
    if (k_tail_blk > 0) {
        microkernel(k_tail_blk, bdb, ldb, tail);
        add_imm(reg_aux_a, reg_aux_a, brg.m_blk * brg.k_blk * k_tail_blk,
                X_TMP_0);
        add_imm(reg_aux_b, reg_aux_b,
                brg.k_blk * brg.n_blk * brg.ld_block * k_tail_blk, X_TMP_0);
    }
    if (k_residual_blk > 0) { microkernel(1, bdb, ldb, tail); }
    store_regs(bdb, ldb, tail);
}

void jit_int8_matmul_kernel_t::loop_k_zp(int bdb, int ldb, int is_a, int is_b) {
    eor(z3.d, z3.d, z3.d);
    eor(z4.d, z4.d, z4.d);
    for (int i = 0; i < 6; i++)
        eor(acc(2, i).d, acc(2, i).d, acc(2, i).d);
    mov(reg_aux_a, reg_aux_a1);
    mov(reg_aux_b, reg_b);
    if (k_full_blks > 0) {
        mov(reg_rd_loop, k_full_blks);
        Label l0;
        L(l0);
        zp_comp(brg.rd_block, bdb, ldb, is_a, is_b);
        add_imm(reg_aux_a, reg_aux_a, brg.m_blk * brg.k_blk * brg.rd_block,
                X_TMP_0);
        add_imm(reg_aux_b, reg_aux_b,
                brg.k_blk * brg.n_blk * brg.ld_block * brg.rd_block, X_TMP_0);
        sub(reg_rd_loop, reg_rd_loop, 1);
        cmp(reg_rd_loop, 0);
        b(GT, l0);
    }
    if (k_tail_blk > 0) {
        zp_comp(k_tail_blk, bdb, ldb, is_a, is_b);
        add_imm(reg_aux_a, reg_aux_a, brg.m_blk * brg.k_blk * k_tail_blk,
                X_TMP_0);
        add_imm(reg_aux_b, reg_aux_b,
                brg.k_blk * brg.n_blk * brg.ld_block * k_tail_blk, X_TMP_0);
    }
    if (k_residual_blk > 0) { zp_comp(1, bdb, ldb, is_a, is_b); }

    if (brg.zp_type_b != jit_int8_broadcast_t::none && is_b == 1) {
        uzp1(z3.d, z3.d, z4.d);
        scvtf(z3.s, P_ALL_ONE, z3.s);
        if (!brg.is_zp_b_int8) {
            ldr(W_TMP_0, ptr(reg_zp_val_b));
            dup(z0.s, W_TMP_0);
            scvtf(z0.s, P_ALL_ONE, z0.s);
            fmul(z3.s, P_ALL_ONE, z0.s);
        } else {
            if (brg.zp_type_a != jit_int8_broadcast_t::none) {
                ldr(W_TMP_0, ptr(reg_zp_val_a));
                dup(z0.s, W_TMP_0);
                mov_imm(W_TMP_0, brg.K);
                dup(z1.s, W_TMP_0);
                scvtf(z0.s, P_ALL_ONE, z0.s);
                scvtf(z1.s, P_ALL_ONE, z1.s);
                fmul(z0.s, z1.s, z0.s);
                fsub(z3.s, z3.s, z0.s);
            }
        }
        st1w(z3.s, P_ALL_ONE, ptr(reg_zp_b));
    }

    if ((brg.zp_type_a != jit_int8_broadcast_t::none || brg.is_src_f32)
            && is_a == 1) {
        ldr(W_TMP_0, ptr(reg_zp_val_a));
        dup(z2.s, W_TMP_0);
        scvtf(z2.s, P_ALL_ONE, z2.s);
        uzp1(acc(2, 0).d, acc(2, 0).d, acc(2, 1).d);
        uzp1(acc(2, 2).d, acc(2, 2).d, acc(2, 3).d);
        uzp1(acc(2, 4).d, acc(2, 4).d, acc(2, 5).d);

        scvtf(acc(2, 0).s, P_ALL_ONE, acc(2, 0).s);
        scvtf(acc(2, 2).s, P_ALL_ONE, acc(2, 2).s);
        scvtf(acc(2, 4).s, P_ALL_ONE, acc(2, 4).s);
        if (brg.zp_type_b != jit_int8_broadcast_t::none && !brg.is_zp_b_int8) {
            ldr(W_TMP_0, ptr(reg_zp_val_b));
            dup(z0.s, W_TMP_0);
            mov_imm(W_TMP_0, brg.K);
            dup(z1.s, W_TMP_0);
            scvtf(z0.s, P_ALL_ONE, z0.s);
            scvtf(z1.s, P_ALL_ONE, z1.s);
            fmul(z0.s, z1.s, z0.s);
            fsub(acc(2, 0).s, acc(2, 0).s, z0.s);
            fsub(acc(2, 2).s, acc(2, 2).s, z0.s);
            fsub(acc(2, 4).s, acc(2, 4).s, z0.s);
        }
        fmul(acc(2, 0).s, P_ALL_ONE, z2.s);
        fmul(acc(2, 2).s, P_ALL_ONE, z2.s);
        fmul(acc(2, 4).s, P_ALL_ONE, z2.s);

        st1w(acc(2, 0).s, P_ALL_ONE, ptr(reg_zp_a));
        st1w(acc(2, 2).s, P_ALL_ONE, ptr(reg_zp_a, 1, MUL_VL));
        st1w(acc(2, 4).s, P_ALL_ONE, ptr(reg_zp_a, 2, MUL_VL));
    }
}

void jit_int8_matmul_kernel_t::han_blk() {
    Label ld_loop, bd_loop;
    LDR_IMM(reg_tmp, reg_param, GET_OFF(nb));
    LDR_IMM(reg_na, reg_param, GET_OFF(na));
    ldr(WReg(reg_ld_loop.getIdx()), ptr(reg_tmp));
    mov(reg_aux_a1, reg_a);
    // mov(reg_b,reg_b);
    mov(reg_aux_c1, reg_c);
    mov(reg_aux_c, reg_aux_c1);
    mov(reg_zp_aux_b, reg_zp_b);
    L(ld_loop);
    ldr(WReg(reg_bd_loop.getIdx()), ptr(reg_na));
    L(bd_loop);
    loop_k(bdb, ldb, 0);
    add_imm(reg_aux_a1, reg_aux_a1,
            div_up(brg.K, brg.k_blk) * brg.k_blk * brg.bd_block, X_TMP_0);
    add_imm(reg_aux_c, reg_aux_c, brg.N * brg.bd_block * brg.dst_dt_sz,
            X_TMP_0);
    add_imm(reg_zp_aux_b, reg_zp_aux_b, brg.m_blk * brg.dst_dt_sz, X_TMP_0);
    sub(reg_bd_loop, reg_bd_loop, 1);
    cmp(reg_bd_loop, 0);
    b(GT, bd_loop);
    mov(reg_aux_a1, reg_a);
    mov(reg_zp_aux_b, reg_zp_b);
    add_imm(reg_b, reg_b,
            (brg.n_blk * brg.ld_block) * div_up(brg.K, brg.k_blk) * brg.k_blk,
            X_TMP_0);
    add_imm(reg_aux_c1, reg_aux_c1, brg.dst_dt_sz * (brg.n_blk * brg.ld_block),
            X_TMP_0);
    add_imm(reg_zp_a, reg_zp_a, brg.n_blk * brg.ld_block * brg.dst_dt_sz,
            X_TMP_0);
    if (brg.is_oc_scales)
        add_imm(reg_scales, reg_scales,
                brg.dst_dt_sz * (brg.n_blk * brg.ld_block), X_TMP_0);
    add_imm(reg_bias, reg_bias, brg.dst_dt_sz * (brg.n_blk * brg.ld_block),
            X_TMP_0);
    mov(reg_aux_c, reg_aux_c1);
    sub(reg_ld_loop, reg_ld_loop, 1);
    cmp(reg_ld_loop, 0);
    b(GT, ld_loop);
}

void jit_int8_matmul_kernel_t::han_blk_zp() {
    Label ld_loop, bd_loop, skip_ld_loop, skip_bd_loop;
    LDR_IMM(reg_tmp, reg_param, GET_OFF(nb));
    LDR_IMM(reg_na, reg_param, GET_OFF(na));
    ldr(WReg(reg_ld_loop.getIdx()), ptr(reg_tmp));
    ldr(WReg(reg_bd_loop.getIdx()), ptr(reg_na));
    mov(reg_aux_a1, reg_a);
    // mov(reg_b,reg_b);
    if (brg.zp_type_b != jit_int8_broadcast_t::none) {
        cmp(reg_bd_loop, 0);
        b(EQ, skip_bd_loop);
        L(bd_loop);
        loop_k_zp(bdb, ldb, 0, 1);
        add_imm(reg_aux_a1, reg_aux_a1,
                div_up(brg.K, brg.k_blk) * brg.k_blk * brg.bd_block, X_TMP_0);
        add_imm(reg_zp_b, reg_zp_b, brg.m_blk * brg.dst_dt_sz, X_TMP_0);
        sub(reg_bd_loop, reg_bd_loop, 1);
        cmp(reg_bd_loop, 0);
        b(GT, bd_loop);
        L(skip_bd_loop);
    }
    if (brg.zp_type_a != jit_int8_broadcast_t::none || brg.is_src_f32) {
        cmp(reg_ld_loop, 0);
        b(EQ, skip_ld_loop);
        L(ld_loop);
        loop_k_zp(bdb, ldb, 1, 0);
        add_imm(reg_zp_a, reg_zp_a, brg.n_blk * brg.ld_block * brg.dst_dt_sz,
                X_TMP_0);
        add_imm(reg_b, reg_b,
                (brg.n_blk * brg.ld_block) * div_up(brg.K, brg.k_blk)
                        * brg.k_blk,
                X_TMP_0);
        sub(reg_ld_loop, reg_ld_loop, 1);
        cmp(reg_ld_loop, 0);
        b(GT, ld_loop);
        L(skip_ld_loop);
    }
}

void jit_int8_matmul_kernel_t::zp_comp(
        int rdb, int bdb, int ldb, int is_a, int is_b) {

    dup(z0.b, 1);
    int rd, ld;
    if (brg.zp_type_b != jit_int8_broadcast_t::none && is_b == 1) {
        mov(reg_tmp, reg_aux_a);
        for (rd = 0; rd < rdb; rd++) {
            ld1b(z1.b, P_ALL_ONE / T_z, ptr(reg_tmp));
            ld1b(z2.b, P_ALL_ONE / T_z, ptr(reg_tmp, 1, MUL_VL));
            add_imm(reg_tmp, reg_tmp, brg.k_blk * brg.m_blk, X_TMP_0);
            if (brg.is_s8) {
                smmla(z3.s, z0.b, z1.b);
                smmla(z4.s, z0.b, z2.b);
            } else {
                ummla(z3.s, z0.b, z1.b);
                ummla(z4.s, z0.b, z2.b);
            }
        }
    }
    if ((brg.zp_type_a != jit_int8_broadcast_t::none || brg.is_src_f32)
            && is_a == 1) {
        mov(reg_tmp, reg_aux_b);

        for (rd = 0; rd < rdb; rd++) {
            for (ld = 0; ld < ldb; ld++) {
                PReg p = (brg.is_n_tail && ld == ldb - 1) ? prd_ld : P_ALL_ONE;
                ld1b(acc(1, ld).b, p, ptr(reg_tmp, ld, MUL_VL));
            }
            add_imm(reg_tmp, reg_tmp, brg.k_blk * brg.n_blk * brg.ld_block,
                    X_TMP_0);
            for (ld = 0; ld < ldb; ld++) {
                if (brg.is_s8) {
                    smmla(acc(2, ld).s, z0.b, acc(1, ld).b);
                } else {
                    ummla(acc(2, ld).s, z0.b, acc(1, ld).b);
                }
            }
        }
    }
}

void jit_int8_matmul_kernel_t::config() {
    int m, pred_st = 0, pred_ld = 0, sv_len = 8, pred_b = 8;
    n_blks = div_up(brg.n_tail, 8);
    k_full_blks = brg.K / (brg.k_blk * brg.rd_block);
    m = brg.K % (brg.k_blk * brg.rd_block);
    k_tail_blk = m / brg.k_blk;
    k_residual_blk = m % brg.k_blk;
    ldb = (brg.is_n_tail) ? div_up(brg.n_tail, 4) : brg.ld_block;
    bdb = (brg.is_m_tail) ? div_up(brg.m_tail, 2) : brg.bd_block / 2;
    rdb = (brg.is_k_tail) ? div_up(brg.k_tail, brg.k_blk) : 4;

    int pred_zp_b_tl = (brg.n_tail % 8 == 0) ? 8 : brg.n_tail % 8;
    set_preg(prd_8.b, 8, X_TMP_0, X_TMP_1);
    set_preg(prd_zp_b_tl.b, pred_zp_b_tl, X_TMP_0, X_TMP_1);

    if (brg.is_n_tail) {
        pred_b = (brg.n_tail % 8 == 0) ? sv_len : (brg.n_tail % 8);
        if (brg.n_tail % brg.n_blk == 0) {
            pred_st = (brg.n_tail % (brg.n_blk * 2) == 0) ? sv_len : sv_len / 2;
            pred_ld = sv_len * brg.dst_dt_sz;
        } else {
            pred_ld = (brg.n_tail % brg.n_blk) * brg.k_blk;
            pred_st = (ldb % 2 == 0) ? (sv_len / 2) + (brg.n_tail % brg.n_blk)
                                     : (brg.n_tail % brg.n_blk);
        }
    }
    set_preg(prd_ld.b, pred_ld, X_TMP_0, X_TMP_1);
    set_preg(prd_st.s, pred_st, X_TMP_0, X_TMP_1);
    set_preg(prd_b.s, pred_b, X_TMP_0, X_TMP_1);
}

void jit_int8_matmul_kernel_t::generate() {
    preamble();
    config();

    LDR_IMM(reg_a, reg_param, GET_OFF(src));
    LDR_IMM(reg_b, reg_param, GET_OFF(wei));
    LDR_IMM(reg_c, reg_param, GET_OFF(dst));
    LDR_IMM(reg_zp_b, reg_param, GET_OFF(zp_b_ptr));
    LDR_IMM(reg_zp_a, reg_param, GET_OFF(zp_a_ptr));
    if (brg.is_zp_cal) {
        LDR_IMM(reg_zp_val_b, reg_param, GET_OFF(wei_zero_point));
        if (brg.is_src_f32) {
            LDR_IMM(reg_zp_val_a, reg_param, GET_OFF(dyn_src_zp));
        } else {
            LDR_IMM(reg_zp_val_a, reg_param, GET_OFF(src_zero_point));
        }
        han_blk_zp();
    } else {

        LDR_IMM(reg_bias, reg_param, GET_OFF(bias));
        LDR_IMM(reg_scales, reg_param, GET_OFF(scales));
        LDR_IMM(reg_aux_scales, reg_param, GET_OFF(dst_scales));
        LDR_IMM(reg_dyn_scl, reg_param, GET_OFF(dyn_src_scl));
        LDR_IMM(reg_dyn_scl, reg_param, GET_OFF(dyn_src_scl));
        LDR_IMM(reg_zp_aux_b_buf, reg_param, GET_OFF(wei_zero_point_buf));
        han_blk();
    }

    postamble();
}

status_t jit_int8_matmul_t::pd_t::init(engine_t *engine) {

    const auto src_type = src_md(0)->data_type;
    const auto wei_type = weights_md(0)->data_type;
    const auto dst_type = dst_md(0)->data_type;

    const memory_desc_wrapper src_d(src_md_);
    const memory_desc_wrapper weights_d(weights_md_);
    const memory_desc_wrapper dst_d(dst_md_);

    if (src_d.has_runtime_dims_or_strides()
            || weights_d.has_runtime_dims_or_strides())
        return status::unimplemented;

    bool is_s8_wei = utils::everyone_is(s8, wei_type);
    bool is_u8 = utils::everyone_is(u8, src_type, wei_type);
    bool is_s8 = utils::everyone_is(s8, src_type, wei_type);
    bool is_src_f32 = (utils::everyone_is(u8, wei_type)
                              || utils::everyone_is(s8, wei_type))
            && utils::everyone_is(f32, src_type);

    static std::string val = getenv_string_user("AARCH64_MATMUL_SRC_QUANT");
    if (is_src_f32) {
        if (!val.empty()) {
            if (val.compare("on") != 0) return status::unimplemented;
        } else {
            return status::unimplemented;
        }
    }

    int dims = src_d.ndims();

    auto check_attr_scales = [&]() -> bool {
        if (is_src_f32
                && !attr()->scales_.get(DNNL_ARG_SRC).has_default_values())
            return false;

        const std::vector<int> supported_args
                = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
        bool ok = attr_scales_ok(supported_args);
        auto is_src_scl
                = !attr()->scales_.get(DNNL_ARG_SRC).has_default_values();
        auto is_dst_scl
                = !attr()->scales_.get(DNNL_ARG_DST).has_default_values();
        auto is_wei_scl
                = !attr()->scales_.get(DNNL_ARG_WEIGHTS).has_default_values();
        auto dst_scl_msk = attr()->scales_.get(DNNL_ARG_DST).get_mask();
        auto wei_scl_msk = attr()->scales_.get(DNNL_ARG_WEIGHTS).get_mask();
        auto src_scl_msk = attr()->scales_.get(DNNL_ARG_SRC).get_mask();

        if ((is_src_scl && src_scl_msk != 0)
                || (is_wei_scl && wei_scl_msk != 0
                        && wei_scl_msk != 1 << (dims - 1))
                || (is_dst_scl && dst_scl_msk != 0))
            return false;

        if (is_src_scl && is_src_scl && wei_scl_msk != 0) {
            // This case requires scratchpad
            if (N() == DNNL_RUNTIME_DIM_VAL) ok = false;
        }
        return ok;
    };

    auto check_bias = [&]() -> bool {
        const auto bia_dt = weights_md(1)->data_type;

        const bool is_bia_dt_correct = IMPLICATION(true, one_of(bia_dt, f32))
                && IMPLICATION(false, one_of(bia_dt, f32, src_type));
        return IMPLICATION(with_bias(), is_bia_dt_correct && is_bias_1xN());
    };

    auto init_zp_type = [&](brg_int8_t *brg) -> bool {
        auto zero_points = attr()->zero_points_;

        if (is_src_f32 && !zero_points.has_default_values(DNNL_ARG_SRC))
            return false;

        auto wt_int8 = zero_points.get_data_type(DNNL_ARG_WEIGHTS) == u8
                || zero_points.get_data_type(DNNL_ARG_WEIGHTS) == s8;
        if (!zero_points.has_default_data_type(DNNL_ARG_SRC)
                || !zero_points.has_default_data_type(DNNL_ARG_DST)
                || (!zero_points.has_default_data_type(DNNL_ARG_WEIGHTS)
                        && !wt_int8))
            return false;

        if (!zero_points.has_default_data_type(DNNL_ARG_WEIGHTS) && wt_int8) {
            brg->is_zp_b_int8 = true;
            if (zero_points.get_data_type(DNNL_ARG_WEIGHTS) == u8)
                brg->zp_b_dt = u8;
            else
                brg->zp_b_dt = s8;
        }

        if (!zero_points.common(DNNL_ARG_SRC)
                || !zero_points.common(DNNL_ARG_DST)
                || (!zero_points.common(DNNL_ARG_WEIGHTS)
                        && !zero_points.per_ocic(
                                DNNL_ARG_WEIGHTS, weights_d.ndims())))
            return false;

        brg->zp_type_a = zero_points.has_default_values(DNNL_ARG_SRC)
                ? jit_int8_broadcast_t::none
                : jit_int8_broadcast_t::per_tensor;

        brg->zp_type_b = zero_points.has_default_values(DNNL_ARG_WEIGHTS)
                ? jit_int8_broadcast_t::none
                : jit_int8_broadcast_t::per_tensor;

        brg->zp_type_c = zero_points.has_default_values(DNNL_ARG_DST)
                ? jit_int8_broadcast_t::none
                : jit_int8_broadcast_t::per_tensor;

        return true;
    };

    VDISPATCH_MATMUL(init_zp_type(&brg), VERBOSE_UNSUPPORTED_ZP_CFG);

    if (is_src_f32 && brg.zp_type_a != jit_int8_broadcast_t::none)
        return status::unimplemented;

    VDISPATCH_MATMUL(check_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_MATMUL(check_attr_scales(), VERBOSE_UNSUPPORTED_SCALES_CFG);

    bool no_post_ops = attr()->post_ops_.has_default_values();
    const bool ok = (is_s8 || is_u8 || is_src_f32)
            && utils::everyone_is(f32, dst_type) && no_post_ops
            && (get_sve_length() == 32);
    if (!ok) return status::unimplemented;

    switch (dims) {
        case 2: {
            VCHECK_BG(memory_desc_init_by_tag(src_md_, format_tag::ab),
                    VERBOSE_UNSUPPORTED_TAG);
            VCHECK_BG(memory_desc_init_by_tag(dst_md_, format_tag::ab),
                    VERBOSE_UNSUPPORTED_TAG);
            if (!weights_d.matches_tag(format_tag::ab)) {
                brg.b_reo = false;
                VCHECK_BG(memory_desc_init_by_tag(
                                  weights_md_, format_tag::BA24b8a),
                        VERBOSE_UNSUPPORTED_TAG);
            } else {
                VCHECK_BG(memory_desc_init_by_tag(weights_md_, format_tag::ab),
                        VERBOSE_UNSUPPORTED_TAG);
            }
            break;
        }
        case 3: {
            VCHECK_BG(memory_desc_init_by_tag(src_md_, format_tag::abc),
                    VERBOSE_UNSUPPORTED_TAG);
            VCHECK_BG(memory_desc_init_by_tag(dst_md_, format_tag::abc),
                    VERBOSE_UNSUPPORTED_TAG);
            if (!weights_d.matches_tag(format_tag::abc)) {
                brg.b_reo = false;
                VCHECK_BG(memory_desc_init_by_tag(
                                  weights_md_, format_tag::aCB24c8b),
                        VERBOSE_UNSUPPORTED_TAG);
            } else {
                VCHECK_BG(memory_desc_init_by_tag(weights_md_, format_tag::abc),
                        VERBOSE_UNSUPPORTED_TAG);
            }
            if (src_d.dims()[0] != weights_d.dims()[0])
                return status::unimplemented;
            break;
        }
        case 4: {
            VCHECK_BG(memory_desc_init_by_tag(src_md_, format_tag::abcd),
                    VERBOSE_UNSUPPORTED_TAG);
            VCHECK_BG(memory_desc_init_by_tag(dst_md_, format_tag::abcd),
                    VERBOSE_UNSUPPORTED_TAG);
            if (!weights_d.matches_tag(format_tag::abcd)) {
                brg.b_reo = false;
                VCHECK_BG(memory_desc_init_by_tag(
                                  weights_md_, format_tag::abDC24d8c),
                        VERBOSE_UNSUPPORTED_TAG);
            } else {
                VCHECK_BG(
                        memory_desc_init_by_tag(weights_md_, format_tag::abcd),
                        VERBOSE_UNSUPPORTED_TAG);
            }
            if (src_d.dims()[0] != weights_d.dims()[0]
                    || src_d.dims()[1] != weights_d.dims()[1])
                return status::unimplemented;
            break;
        }
        default: return status::unimplemented;
    }

    bool is_scales = !attr()->scales_.get(DNNL_ARG_SRC).has_default_values()
            || !attr()->scales_.get(DNNL_ARG_WEIGHTS).has_default_values();

    bool is_dst_scales
            = !attr()->scales_.get(DNNL_ARG_DST).has_default_values();

    const auto &wei_scales = attr()->scales_.get(DNNL_ARG_WEIGHTS);

    matmul_helper_t helper(src_d, weights_d, dst_d);
    brg.K = helper.K();
    brg.M = helper.M();
    brg.N = helper.N();
    brg.dst_dt_sz = 4;
    brg.na = 1;
    brg.nb = 1;
    brg.m_tail = brg.M % brg.m_blk;
    brg.k_tail = brg.K % (brg.k_blk * brg.rd_block);
    brg.n_tail = brg.N % (brg.n_blk * brg.ld_block);
    brg.is_s8 = is_s8_wei;
    brg.is_src_f32 = is_src_f32;
    brg.is_bias = with_bias();
    brg.B = batch();
    brg.with_scales = is_scales;
    brg.with_dst_scales = is_dst_scales;
    brg.is_oc_scales = wei_scales.get_mask() != 0;
    dyn.K = brg.K;
    dyn.N = brg.N;
    dyn.M = brg.M;
    dyn.B = brg.B;
    dyn.mtail = brg.m_tail;
    dyn.m_blk = brg.m_blk;
    dyn.k_blk = brg.k_blk;
    dyn.n_blk = brg.n_blk * brg.ld_block;
    dyn.ntail = brg.n_tail;
    dyn.ktail = dyn.K % brg.k_blk;

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(key_brgemm_primitive_zp_comp_a,
            div_up(brg.N, (brg.n_blk * brg.ld_block))
                    * (brg.n_blk * brg.ld_block) * brg.dst_dt_sz * brg.B,
            sizeof(char));
    scratchpad.book(key_brgemm_primitive_zp_comp_b,
            div_up(brg.M, brg.m_blk) * brg.m_blk * brg.dst_dt_sz * brg.B,
            sizeof(char));
    scratchpad.book(key_brgemm_primitive_buffer_a,
            brg.B * div_up(brg.M, brg.m_blk) * div_up(brg.K, brg.k_blk)
                    * brg.m_blk * brg.k_blk,
            sizeof(char));
    scratchpad.book(
            key_brgemm_primitive_buffer_b, brg.B * brg.M * brg.K, sizeof(char));
    if (brg.b_reo)
        scratchpad.book(key_gemm_blocked_b,
                brg.B * div_up(brg.N, (brg.n_blk * brg.ld_block))
                        * (brg.n_blk * brg.ld_block) * div_up(brg.K, brg.k_blk)
                        * brg.k_blk,
                sizeof(char));
    book_precomputed_scales(scratchpad, attr()->scales_, N());

    return status::success;
}

status_t jit_int8_matmul_t::init(engine_t *engine) {
    if (mayiuse(sve_512)) { assert(!"unreachable"); }
    auto &b1 = pd()->get_b();
    auto &d1 = pd()->get_d();

    dyn_vals_t d;
    d.K = d1.K;
    d.M = d1.M;
    d.B = d1.B;
    d.N = d1.N;
    d.mtail = d1.mtail;
    d.ktail = d1.ktail;
    d.ntail = d1.ntail;
    d.k_blk = d1.k_blk;
    d.m_blk = d1.m_blk;
    d.n_blk = d1.n_blk;

    dnnl::impl::cpu::aarch64::matmul::brg_int8_t b;
    b.M = b1.M;
    b.K = b1.K;
    b.N = b1.N;
    b.na = b1.na;
    b.nb = b1.nb;
    b.m_tail = b1.m_tail;
    b.n_tail = b1.n_tail;
    b.k_tail = b1.k_tail;
    b.dst_dt_sz = b1.dst_dt_sz;
    b.is_s8 = b1.is_s8;
    b.is_src_f32 = b1.is_src_f32;
    b.B = b1.B;
    b.is_bias = b1.is_bias;
    b.zp_type_a = b1.zp_type_a;
    b.zp_type_b = b1.zp_type_b;
    b.zp_type_c = b1.zp_type_c;
    b.is_zp_b_int8 = b1.is_zp_b_int8;
    b.zp_b_dt = b1.zp_b_dt;
    b.with_scales = b1.with_scales;
    b.with_dst_scales = b1.with_dst_scales;
    b.is_oc_scales = b1.is_oc_scales;
    b.b_reo = b1.b_reo;

    for (int z = 0; z < 2; z++)
        for (int m = 0; m < 2; m++)
            for (int n = 0; n < 2; n++)
                for (int k = 0; k < 2; k++) {
                    int idx = pd()->get_idx(z, m, k, n, b);
                    if (idx == -1) continue;
                    b.is_m_tail = m;
                    b.is_k_tail = k;
                    b.is_n_tail = n;
                    b.is_zp_cal = z;
                    int8_kernels_[idx]
                            = std::unique_ptr<jit_int8_matmul_kernel_t> {
                                    new jit_int8_matmul_kernel_t(b)};
                    if (!int8_kernels_[idx]) return status::runtime_error;
                    CHECK(int8_kernels_[idx]->create_kernel());
                }

    d.reorder_a = 1;
    d.get_min_max = 0;
    d.cal_src = 0;
    d.reorder_b = 0;
    reo_ker_a_ = std::unique_ptr<jit_dyn_quant_src_kernel_t> {
            new jit_dyn_quant_src_kernel_t(d)};
    CHECK(reo_ker_a_->create_kernel());

    d.reorder_b = 1;
    d.reorder_a = 0;
    reo_ker_b_ = std::unique_ptr<jit_dyn_quant_src_kernel_t> {
            new jit_dyn_quant_src_kernel_t(d)};
    CHECK(reo_ker_b_->create_kernel());

    if (b.is_src_f32) {
        d.reorder_a = 0;
        d.reorder_b = 0;
        d.get_min_max = 0;
        d.cal_src = 1;
        d.is_s8 = b.is_s8;
        dyn_src_ker = std::unique_ptr<jit_dyn_quant_src_kernel_t> {
                new jit_dyn_quant_src_kernel_t(d)};
        CHECK(dyn_src_ker->create_kernel());
        d.reorder_a = 0;
        d.get_min_max = 1;
        d.cal_src = 0;
        min_max_ker = std::unique_ptr<jit_dyn_quant_src_kernel_t> {
                new jit_dyn_quant_src_kernel_t(d)};
        CHECK(min_max_ker->create_kernel());
    }
    return status::success;
}

jit_int8_matmul_t::jit_int8_matmul_t(const pd_t *apd) : primitive_t(apd) {}
jit_int8_matmul_t::~jit_int8_matmul_t() = default;

status_t jit_int8_matmul_t::execute(const exec_ctx_t &ctx) const {
    const auto *weights_b = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    const auto *src_b = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);
    const auto *bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);

    DEFINE_ZERO_POINT_VALUE(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINT_VALUE(wei_zero_point, DNNL_ARG_WEIGHTS);
    DEFINE_ZERO_POINT_VALUE(dst_zero_point, DNNL_ARG_DST);
    DEFINE_ZERO_POINTS_BUFFER(wei_zero_point_buf, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    auto &b = pd()->get_b();
    auto &d = pd()->get_d();

    auto &scratchpad = ctx.get_scratchpad_grantor();

    int num_threads = dnnl_get_current_num_threads();
    char *src = scratchpad.template get<char>(key_brgemm_primitive_buffer_a);
    char *weights = (b.b_reo)
            ? scratchpad.template get<char>(key_gemm_blocked_b)
            : (char *)weights_b;
    char *zp_ptr_a
            = scratchpad.template get<char>(key_brgemm_primitive_zp_comp_a);
    char *zp_ptr_b
            = scratchpad.template get<char>(key_brgemm_primitive_zp_comp_b);
    char *dyn_src
            = scratchpad.template get<char>(key_brgemm_primitive_buffer_b);
    float dyn_src_scl = 0;
    int32_t dyn_src_zp = 0;
    const float *oscales = precompute_scales(
            scratchpad, src_scales, wei_scales, pd()->N(), pd()->attr());

    const dim_t B = b.B;
    const dim_t M = b.M;
    const dim_t N = b.N;
    const dim_t K = b.K;

    auto get_min_max = [&](float *scl, float *res_scl, int32_t *zp) {
        float fmax = src_b[0], fmin = src_b[0];
        int imin, imax;
        if (b.is_s8) {
            imin = 127;
            imax = -127;
        } else {
            imin = 0;
            imax = 255;
        }
        int blks = div_up(B * M * K, 64);
        int parallel_work = blks;
        int nt = std::min(num_threads, parallel_work);
        int src_tail = (B * M * K) % 64;
        float min_arr[nt], max_arr[nt];

        parallel(nt, [&](const int ithr, const int nthr) {
            int start {0}, end {0};
            balance211(parallel_work, nt, ithr, start, end);

            int ad = start * 64;
            int nob = end - start;
            int tl = 0;
            if (end == blks && src_tail > 0) tl = 1;
            dyn_params_t k;
            k.dyn_src = (float *)src_b + ad;
            k.min = &min_arr[ithr];
            k.max = &max_arr[ithr];
            k.nm = &nob;
            k.tl = &tl;
            (*min_max_ker)(&k);
        });

        for (int i = 0; i < nt; i++)
            if (fmax < max_arr[i]) fmax = max_arr[i];
        for (int i = 0; i < nt; i++)
            if (fmin > min_arr[i]) fmin = min_arr[i];

        if (fmin == fmax) {
            *scl = 1;
            *res_scl = 1;
            *zp = 0;
        } else {
            *scl = (fmax - fmin) / (imax - imin);
            *res_scl = (imax - imin) / (fmax - fmin);
            *zp = std::round((fmax * imin - fmin * imax) / (fmax - fmin));
        }
    };

    auto quantise_src = [&]() {
        float scl;

        get_min_max(&dyn_src_scl, &scl, &dyn_src_zp);

        int blks = div_up(B * M * K, 64);
        int parallel_work = blks;
        int nt = std::min(num_threads, parallel_work);
        int src_tail = (B * M * K) % 64;

        parallel(nt, [&](const int ithr, const int nthr) {
            int start {0}, end {0};
            balance211(parallel_work, nt, ithr, start, end);

            int ad = start * 64;
            int nob = end - start;
            int tl = 0;
            if (end == blks && src_tail > 0) tl = 1;
            dyn_params_t k;
            k.dyn_src = (float *)src_b + ad;
            k.dst = (int8_t *)dyn_src + ad;
            k.src_scls = &scl;
            k.zp = &dyn_src_zp;
            k.tl = &tl;
            k.nm = &nob;
            (*dyn_src_ker)(&k);
        });
    };

    auto reorder_a = [&]() {
        int m_blks = div_up(M, b.m_blk);
        int k_blks = div_up(K, b.k_blk);
        int n_blks = div_up(N, (b.n_blk * b.ld_block));
        int parallel_work = B * m_blks * k_blks;
        int parallel_work_mn = B * m_blks * n_blks;
        int blk_per_bt = m_blks * k_blks;
        int nt = std::min(num_threads, parallel_work);
        nt = std::min(parallel_work_mn, nt);
        auto tmp_src = (b.is_src_f32) ? (float *)dyn_src : src_b;

        parallel(nt, [&](const int ithr, const int nthr) {
            int start {0}, end {0};
            balance211(parallel_work, nt, ithr, start, end);

            int bt = start / blk_per_bt;
            int bs = start % blk_per_bt;
            int nobl = end - start;
            int nobt = 1;
            int noblf = end - start, nobll;

            if (bs + nobl > blk_per_bt) {
                nobt += div_up(nobl - (blk_per_bt - bs), blk_per_bt);
                noblf = blk_per_bt - bs;
                nobll = (nobl - (blk_per_bt - bs)) % blk_per_bt;
                if (nobll == 0) nobll = blk_per_bt;
            }
            int nob;
            for (int i = 0; i < nobt; i++) {
                nob = (i == 0) ? noblf : ((i == nobt - 1) ? nobll : blk_per_bt);
                bs = start % blk_per_bt;
                int m_blk_src = bs / k_blks;
                int k_blk_src = bs % k_blks;
                int m_blk_dst = bs / k_blks;
                int k_blk_dst = bs % k_blks;

                int k1 = std::min(k_blks - k_blk_src, nob);
                int k_tmp = nob - k1;
                int m1 = (k_tmp > 0) ? k_tmp / k_blks : 0;
                int k2 = (k_tmp > 0) ? k_tmp % k_blks : 0;
                int src_ad = (bt * M * K) + (m_blk_src * b.m_blk * K)
                        + (k_blk_src * b.k_blk);
                int dst_ad = (bt * m_blks * k_blks * b.m_blk * b.k_blk)
                        + (m_blk_dst * k_blks * b.m_blk * b.k_blk)
                        + (k_blk_dst * b.m_blk * b.k_blk);
                int src_new = src_ad, dst_new = dst_ad;

                dyn_params_t k;

                if (k1 > 0) {
                    int a = 1;
                    int mtl = (d.mtail > 0) ? 1 : 0;
                    int tl = (d.ktail > 0) ? 1 : 0;
                    if (k1 + k_blk_src < k_blks) tl = 0;
                    if (1 + m_blk_src < m_blks) mtl = 0;
                    k.src = (int8_t *)tmp_src + src_ad;
                    k.dst = (int8_t *)src + dst_ad;
                    k.nm = &a;
                    k.nk = &k1;
                    k.tl = &tl;
                    k.mtl = &mtl;
                    (*reo_ker_a_)(&k);
                }

                if (m1 > 0) {
                    int mtl = (d.mtail > 0) ? 1 : 0;
                    int tl = (d.ktail > 0) ? 1 : 0;
                    if (1 + m1 + m_blk_src < m_blks) mtl = 0;
                    if (k1 != k_blks) {
                        src_new = src_ad - b.k_blk * (k_blks - k1)
                                + b.m_blk * K;
                    } else {
                        src_new = src_ad + b.m_blk * K;
                    }
                    dst_new = dst_ad + b.m_blk * b.k_blk * k1;
                    k.src = (int8_t *)tmp_src + src_new;
                    k.dst = (int8_t *)src + dst_new;
                    k.nm = &m1;
                    k.nk = &k_blks;
                    k.tl = &tl;
                    k.mtl = &mtl;
                    (*reo_ker_a_)(&k);
                }
                if (k2 > 0) {
                    int a = 1, tl = 0;
                    int mtl = (d.mtail > 0) ? 1 : 0;
                    if (1 + 1 + m1 + m_blk_src < m_blks) mtl = 0;
                    if (m1 < 1) {
                        src_new = src_ad - b.k_blk * (k_blks - k1)
                                + (b.m_blk * K);
                        dst_new = dst_ad + b.m_blk * b.k_blk * k1;
                    } else {
                        src_new += K * m1 * b.m_blk;
                        dst_new += b.m_blk * b.k_blk * k_blks * m1;
                    }
                    k.src = (int8_t *)tmp_src + src_new;
                    k.dst = (int8_t *)src + dst_new;
                    k.nm = &a;
                    k.nk = &k2;
                    k.tl = &tl;
                    k.mtl = &mtl;
                    (*reo_ker_a_)(&k);
                }
                bt++;
                start += nob;
            }
        });
    };

    auto reorder_b = [&]() {
        int k_blks = div_up(K, d.k_blk);
        int n_blks = div_up(N, d.n_blk);
        int parallel_work = B * n_blks * k_blks;
        int blk_per_bt = n_blks * k_blks;
        int nt = std::min(num_threads, parallel_work);

        parallel(nt, [&](const int ithr, const int nthr) {
            int start {0}, end {0};
            balance211(parallel_work, nt, ithr, start, end);

            int bt = start / blk_per_bt;
            int bs = start % blk_per_bt;
            int nobl = end - start;
            int nobt = 1;
            int noblf = end - start, nobll;

            if (bs + nobl > blk_per_bt) {
                nobt += div_up(nobl - (blk_per_bt - bs), blk_per_bt);
                noblf = blk_per_bt - bs;
                nobll = (nobl - (blk_per_bt - bs)) % blk_per_bt;
                if (nobll == 0) nobll = blk_per_bt;
            }
            int nob;
            for (int i = 0; i < nobt; i++) {
                nob = (i == 0) ? noblf : ((i == nobt - 1) ? nobll : blk_per_bt);
                bs = start % blk_per_bt;
                int n_blk_src = bs / k_blks;
                int k_blk_src = bs % k_blks;
                int n_blk_dst = bs / k_blks;
                int k_blk_dst = bs % k_blks;

                int k1 = std::min(k_blks - k_blk_src, nob);
                int k_tmp = nob - k1;
                int n1 = (k_tmp > 0) ? k_tmp / k_blks : 0;
                int k2 = (k_tmp > 0) ? k_tmp % k_blks : 0;
                int src_ad = (bt * N * K) + (n_blk_src * d.n_blk)
                        + (k_blk_src * d.k_blk * N);
                int dst_ad = (bt * n_blks * k_blks * d.k_blk * d.n_blk)
                        + (n_blk_dst * k_blks * d.k_blk * d.n_blk)
                        + (k_blk_dst * d.k_blk * d.n_blk);
                int src_new = src_ad, dst_new = dst_ad;

                dyn_params_t k;

                if (k1 > 0) {
                    int a = 1;
                    int ntl = (d.ntail > 0) ? 1 : 0;
                    int tl = (d.ktail > 0) ? 1 : 0;

                    if (k1 + k_blk_src < k_blks) tl = 0;
                    if (1 + n_blk_src < n_blks) ntl = 0;
                    k.src = (int8_t *)weights_b + src_ad;
                    k.dst = (int8_t *)weights + dst_ad;
                    k.nn = &a;
                    k.nk = &k1;
                    k.tl = &tl;
                    k.ntl = &ntl;
                    (*reo_ker_b_)(&k);
                }

                if (n1 > 0) {
                    int ntl = (d.ntail > 0) ? 1 : 0;
                    int tl = (d.ktail > 0) ? 1 : 0;
                    if (1 + n1 + n_blk_src < n_blks) ntl = 0;

                    if (k1 != k_blks) {
                        src_new = src_ad - d.k_blk * N * (k_blks - k1)
                                + d.n_blk;
                    } else {
                        src_new = src_ad + d.n_blk;
                    }
                    dst_new = dst_ad + d.k_blk * d.n_blk * k1;
                    k.src = (int8_t *)weights_b + src_new;
                    k.dst = (int8_t *)weights + dst_new;
                    k.nn = &n1;
                    k.nk = &k_blks;
                    k.tl = &tl;
                    k.ntl = &ntl;
                    (*reo_ker_b_)(&k);
                }
                if (k2 > 0) {
                    int a = 1, tl = 0;
                    int ntl = (d.ntail > 0) ? 1 : 0;
                    if (1 + 1 + n1 + n_blk_src < n_blks) ntl = 0;
                    if (n1 < 1) {
                        src_new = src_ad - d.k_blk * N * (k_blks - k1)
                                + d.n_blk;
                        dst_new = dst_ad + d.k_blk * d.n_blk * k1;
                    } else {
                        src_new += n1 * d.n_blk;
                        dst_new += d.k_blk * d.n_blk * k_blks * n1;
                    }
                    k.src = (int8_t *)weights_b + src_new;
                    k.dst = (int8_t *)weights + dst_new;
                    k.nn = &a;
                    k.nk = &k2;
                    k.tl = &tl;
                    k.ntl = &ntl;
                    (*reo_ker_b_)(&k);
                }
                bt++;
                start += nob;
            }
        });
    };

    auto kernel_execute = [&](int idx, int na, int nb, int m_blk_adr,
                                  int n_blk_adr, int dst_adr, int bias_addr,
                                  int scl_addr, int zp_ptr_a_adr,
                                  int zp_ptr_b_adr, int zp_b_buf) {
        jit_int8_matmul_kernel_t::call_params_t p;
        p.na = &na;
        p.nb = &nb;
        p.src = (uint8_t *)src + m_blk_adr;
        p.wei = (uint8_t *)weights + n_blk_adr;
        p.dst = dst + dst_adr;
        p.bias = bias + bias_addr;
        p.scales = oscales + scl_addr;
        p.dst_scales = dst_scales;
        p.src_zero_point = &src_zero_point;
        if (b.is_zp_b_int8)
            p.wei_zero_point_buf = (int8_t *)wei_zero_point_buf + zp_b_buf;
        else
            p.wei_zero_point = &wei_zero_point;
        p.dst_zero_point = &dst_zero_point;
        p.M = M;
        p.N = N;
        p.K = K;
        p.zp_a_ptr = (float *)zp_ptr_a + zp_ptr_a_adr;
        p.zp_b_ptr = (float *)zp_ptr_b + zp_ptr_b_adr;
        p.dyn_src_scl = &dyn_src_scl;
        p.dyn_src_zp = &dyn_src_zp;
        (*int8_kernels_[idx])(&p);
    };

    auto kernel_execute_zp = [&]() {
        int num_a_blocks = div_up(M, b.m_blk);
        int num_b_blocks = div_up(N, (b.n_blk * b.ld_block));
        int ktail = (b.k_tail == 0) ? 0 : 1;
        int parallel_work = B * num_a_blocks;
        int nt = std::min(num_threads, parallel_work);
        if (b.zp_type_b != jit_int8_broadcast_t::none)
            parallel(nt, [&](const int ithr, const int nthr) {
                int start {0}, end {0};
                balance211(parallel_work, nt, ithr, start, end);
                int batch = start / num_a_blocks;
                int m_st = start % num_a_blocks;
                int m_ed = end - start + m_st;
                int mtail
                        = (m_ed == num_a_blocks) ? ((b.m_tail > 0) ? 1 : 0) : 0;
                int m_blk_adr = (batch
                                        * (num_a_blocks * b.m_blk
                                                * div_up(K, b.k_blk) * b.k_blk))
                        + m_st * b.m_blk * div_up(K, b.k_blk) * b.k_blk;
                int zp_ptr_b_adr
                        = (batch * (num_a_blocks * b.m_blk)) + m_st * b.m_blk;

                int idx = pd()->get_idx(1, 0, ktail, 0, b);
                int n_a = m_ed - m_st;
                if (mtail) n_a -= 1;
                kernel_execute(
                        idx, n_a, 0, m_blk_adr, 0, 0, 0, 0, 0, zp_ptr_b_adr, 0);

                if (mtail) {
                    idx = pd()->get_idx(1, mtail, ktail, 0, b);
                    m_blk_adr += n_a * b.m_blk * div_up(K, b.k_blk) * b.k_blk;
                    zp_ptr_b_adr += n_a * b.m_blk;
                    kernel_execute(idx, 1, 0, m_blk_adr, 0, 0, 0, 0, 0,
                            zp_ptr_b_adr, 0);
                }
                start++;
            });

        parallel_work = B * num_b_blocks;
        nt = std::min(num_threads, parallel_work);
        if (b.zp_type_a != jit_int8_broadcast_t::none || b.is_src_f32)
            parallel(nt, [&](const int ithr, const int nthr) {
                int start {0}, end {0};
                balance211(parallel_work, nt, ithr, start, end);
                int batch = start / num_b_blocks;
                int n_st = start % num_b_blocks;
                int n_ed = n_st + end - start;
                int ntail
                        = (n_ed == num_b_blocks) ? ((b.n_tail > 0) ? 1 : 0) : 0;
                int n_blk_adr = (batch
                                        * (num_b_blocks * (b.n_blk * b.ld_block)
                                                * div_up(K, b.k_blk) * b.k_blk))
                        + n_st * (b.n_blk * b.ld_block) * div_up(K, b.k_blk)
                                * b.k_blk;
                int zp_ptr_a_adr
                        = (batch * num_b_blocks * (b.n_blk * b.ld_block))
                        + n_st * (b.n_blk * b.ld_block);

                int idx = pd()->get_idx(1, 0, ktail, 0, b);
                int n_b = n_ed - n_st;
                if (ntail == 1) n_b -= 1;

                kernel_execute(
                        idx, 0, n_b, 0, n_blk_adr, 0, 0, 0, zp_ptr_a_adr, 0, 0);

                if (ntail) {
                    idx = pd()->get_idx(1, 0, ktail, 1, b);
                    n_blk_adr += n_b * (b.n_blk * b.ld_block)
                            * div_up(K, b.k_blk) * b.k_blk;
                    zp_ptr_a_adr += n_b * (b.n_blk * b.ld_block);
                    kernel_execute(idx, 0, 1, 0, n_blk_adr, 0, 0, 0,
                            zp_ptr_a_adr, 0, 0);
                }

                start++;
            });
    };

    if (b.b_reo) reorder_b();

    if (b.is_src_f32) quantise_src();

    reorder_a();

    if (b.zp_type_a != jit_int8_broadcast_t::none
            || b.zp_type_b != jit_int8_broadcast_t::none || b.is_src_f32)
        kernel_execute_zp();

    int m_block_sz = 32;
    int n_block_sz = 24;
    int m_block1 = div_up(m_block_sz, b.m_blk);
    int n_block1 = div_up(n_block_sz, (b.n_blk * b.ld_block));
    int m_block1_rs = div_up(M % m_block_sz, b.m_blk);
    int n_block1_rs = div_up(N % n_block_sz, (b.n_blk * b.ld_block));

    int num_a_blocks_act = div_up(M, b.m_blk);
    int num_b_blocks_act = div_up(N, (b.n_blk * b.ld_block));
    int num_a_blocks = div_up(M, m_block_sz);
    int num_b_blocks = div_up(N, n_block_sz);
    int ktail = (b.k_tail == 0) ? 0 : 1;
    int parallel_work = B * num_a_blocks * num_b_blocks;
    int nt = std::min(num_threads, parallel_work);

    parallel(nt, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(parallel_work, nt, ithr, start, end);
        while (start < end) {
            int batch = start / (num_a_blocks * num_b_blocks);
            int batch_start = start % (num_a_blocks * num_b_blocks);
            int m_block = batch_start % num_a_blocks;
            int n_block = batch_start / num_a_blocks;
            int mtail
                    = (m_block1_rs != 0 && m_block == num_a_blocks - 1) ? 1 : 0;
            int ntail
                    = (n_block1_rs != 0 && n_block == num_b_blocks - 1) ? 1 : 0;
            int dst_adr = (batch * M * N) + m_block * b.m_blk * m_block1 * N
                    + n_block * (b.n_blk * b.ld_block) * n_block1;
            int m_blk_adr = (batch
                                    * (num_a_blocks_act * b.m_blk
                                            * div_up(K, b.k_blk) * b.k_blk))
                    + m_block * b.m_blk * m_block1 * div_up(K, b.k_blk)
                            * b.k_blk;
            int n_blk_adr = (batch
                                    * (num_b_blocks_act * (b.n_blk * b.ld_block)
                                            * div_up(K, b.k_blk) * b.k_blk))
                    + n_block * (b.n_blk * b.ld_block) * n_block1
                            * div_up(K, b.k_blk) * b.k_blk;
            int zp_ptr_a_adr
                    = (batch * (num_b_blocks_act * (b.n_blk * b.ld_block)))
                    + n_block * (b.n_blk * b.ld_block) * n_block1;
            int zp_ptr_b_adr = (batch * (num_a_blocks_act * b.m_blk))
                    + m_block * b.m_blk * m_block1;
            int bias_addr = n_block * (b.n_blk * b.ld_block) * n_block1;
            int zp_b_buf = n_block * (b.n_blk * b.ld_block) * n_block1;
            int scl_addr = (b.is_oc_scales)
                    ? (n_block * (b.n_blk * b.ld_block) * n_block1)
                    : 0;
            int idx = pd()->get_idx(0, 0, ktail, 0, b);
            int n_a = m_block1, n_b = n_block1;
            n_a = (mtail) ? ((b.m_tail) ? m_block1_rs - 1 : m_block1_rs)
                          : m_block1;
            n_b = (ntail) ? ((b.n_tail) ? n_block1_rs - 1 : n_block1_rs)
                          : n_block1;

            if (n_a > 0 && n_b > 0) {

                kernel_execute(idx, n_a, n_b, m_blk_adr, n_blk_adr, dst_adr,
                        bias_addr, scl_addr, zp_ptr_a_adr, zp_ptr_b_adr,
                        zp_b_buf);
            }

            if (mtail && b.m_tail > 0 && n_b > 0) {
                int new_dst_adr = dst_adr + b.m_blk * n_a * N;
                int new_m_blk_adr = m_blk_adr
                        + b.m_blk * n_a * div_up(K, b.k_blk) * b.k_blk;
                int new_zp_ptr_b_adr = zp_ptr_b_adr + b.m_blk * n_a;
                idx = pd()->get_idx(0, 1, ktail, 0, b);
                int na = 1;
                kernel_execute(idx, na, n_b, new_m_blk_adr, n_blk_adr,
                        new_dst_adr, bias_addr, scl_addr, zp_ptr_a_adr,
                        new_zp_ptr_b_adr, zp_b_buf);
            }

            if (ntail && b.n_tail > 0 && n_a > 0) {
                int new_dst_adr = dst_adr + (b.n_blk * b.ld_block) * n_b;
                int new_n_blk_adr = n_blk_adr
                        + (b.n_blk * b.ld_block) * n_b * div_up(K, b.k_blk)
                                * b.k_blk;
                int new_zp_b_buf = zp_b_buf + (b.n_blk * b.ld_block) * n_b;
                int new_zp_ptr_a_adr
                        = zp_ptr_a_adr + (b.n_blk * b.ld_block) * n_b;
                int new_bias_addr = bias_addr + (b.n_blk * b.ld_block) * n_b;
                int new_scl_addr = scl_addr
                        + ((b.is_oc_scales) ? ((b.n_blk * b.ld_block) * n_b)
                                            : 0);
                idx = pd()->get_idx(0, 0, ktail, 1, b);
                int nb = 1;

                kernel_execute(idx, n_a, nb, m_blk_adr, new_n_blk_adr,
                        new_dst_adr, new_bias_addr, new_scl_addr,
                        new_zp_ptr_a_adr, zp_ptr_b_adr, new_zp_b_buf);
            }

            if (mtail && b.m_tail > 0 && ntail && b.n_tail > 0) {
                int new_dst_adr = dst_adr + (b.n_blk * b.ld_block) * n_b
                        + b.m_blk * n_a * N;
                int new_m_blk_adr = m_blk_adr
                        + b.m_blk * n_a * div_up(K, b.k_blk) * b.k_blk;
                int new_n_blk_adr = n_blk_adr
                        + (b.n_blk * b.ld_block) * n_b * div_up(K, b.k_blk)
                                * b.k_blk;
                int new_zp_b_buf = zp_b_buf + (b.n_blk * b.ld_block) * n_b;
                int new_zp_ptr_a_adr
                        = zp_ptr_a_adr + (b.n_blk * b.ld_block) * n_b;
                int new_zp_ptr_b_adr = zp_ptr_b_adr + b.m_blk * n_a;
                int new_bias_addr = bias_addr + (b.n_blk * b.ld_block) * n_b;
                int new_scl_addr = scl_addr
                        + ((b.is_oc_scales) ? ((b.n_blk * b.ld_block) * n_b)
                                            : 0);
                idx = pd()->get_idx(0, 1, ktail, 1, b);
                int nb = 1, na = 1;
                kernel_execute(idx, na, nb, new_m_blk_adr, new_n_blk_adr,
                        new_dst_adr, new_bias_addr, new_scl_addr,
                        new_zp_ptr_a_adr, new_zp_ptr_b_adr, new_zp_b_buf);
            }
            start++;
        }
    });

    return status::success;
}

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl