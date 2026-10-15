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

#include "cpu/rv64/jit_rvv_layernorm_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define GET_FUSED_OFF(field) \
    static_cast<int32_t>( \
            offsetof(jit_rvv_layernorm_fused_kernel_t::call_params_t, field))
#define GET_DATA_OFF(field) \
    static_cast<int32_t>( \
            offsetof(jit_rvv_layernorm_data_kernel_t::call_params_t, field))

jit_rvv_layernorm_fused_kernel_t::jit_rvv_layernorm_fused_kernel_t(
        bool with_scale, bool with_shift)
    : jit_generator_t("jit_rvv_layernorm_fused_kernel")
    , with_scale_(with_scale)
    , with_shift_(with_shift) {
    create_kernel();
}

void jit_rvv_layernorm_fused_kernel_t::generate() {
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_len = a2;
    const Reg reg_dst = a3;
    const Reg reg_scale = a4;
    const Reg reg_shift = a5;
    const Reg reg_mean_ptr = a6;
    const Reg reg_var_ptr = a7;
    const Reg reg_vl = t0;
    const Reg reg_vlmax = t1;
    const Reg reg_bytes = t2;
    const Reg reg_bytes4 = t3;
    const Reg reg_block = t4;
    const Reg reg_tmp = t5;

    const FReg f_zero = ft0;
    const FReg f_sum = ft1;
    const FReg f_var_sum = ft2;
    const FReg f_len = ft3;
    const FReg f_mean = ft4;
    const FReg f_var = ft5;
    const FReg f_inv = ft6;
    const FReg f_aux = ft7;
    const FReg f_one = ft8;

    const VReg v_in0(0);
    const VReg v_in1(1);
    const VReg v_in2(2);
    const VReg v_in3(3);
    const VReg v_tmp0(4);
    const VReg v_tmp1(5);
    const VReg v_tmp2(6);
    const VReg v_tmp3(7);
    const VReg v_mean(12);
    const VReg v_inv(13);
    const VReg v_scale0(14);
    const VReg v_scale1(15);
    const VReg v_scale2(16);
    const VReg v_scale3(17);
    const VReg v_shift0(18);
    const VReg v_shift1(19);
    const VReg v_shift2(20);
    const VReg v_shift3(21);
    fmv_w_x(f_zero, x0);

    vsetvli(reg_vlmax, x0, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    slli(reg_bytes, reg_vlmax, 2);
    slli(reg_bytes4, reg_bytes, 2);
    slli(reg_block, reg_vlmax, 2);

    Label mean_main_loop, mean_vec_tail, mean_reduce_loop, mean_done;
    Label var_main_loop, var_vec_tail, var_reduce_loop, var_done;
    Label store_var, start_data, data_main_loop, data_tail_loop, done;

    ld(reg_src, reg_param, GET_FUSED_OFF(src));
    ld(reg_len, reg_param, GET_FUSED_OFF(len));
    fsgnj_s(f_sum, f_zero, f_zero);
    vfmv_v_f(v_tmp0, f_zero);
    vfmv_v_f(v_tmp1, f_zero);
    vfmv_v_f(v_tmp2, f_zero);
    vfmv_v_f(v_tmp3, f_zero);

    vsetvli(x0, reg_vlmax, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    L(mean_main_loop);
    blt(reg_len, reg_block, mean_vec_tail);
    vle32_v(v_in0, reg_src);
    add(reg_tmp, reg_src, reg_bytes);
    vle32_v(v_in1, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in2, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in3, reg_tmp);
    vfadd_vv(v_tmp0, v_tmp0, v_in0);
    vfadd_vv(v_tmp1, v_tmp1, v_in1);
    vfadd_vv(v_tmp2, v_tmp2, v_in2);
    vfadd_vv(v_tmp3, v_tmp3, v_in3);
    add(reg_src, reg_src, reg_bytes4);
    sub(reg_len, reg_len, reg_block);
    j_(mean_main_loop);

    L(mean_vec_tail);
    beqz(reg_len, mean_reduce_loop);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1, VTA::tu, VMA::ma);
    slli(reg_tmp, reg_vl, 2);
    vle32_v(v_in0, reg_src);
    vfadd_vv(v_tmp0, v_tmp0, v_in0);
    add(reg_src, reg_src, reg_tmp);
    sub(reg_len, reg_len, reg_vl);
    j_(mean_vec_tail);

    L(mean_reduce_loop);
    vsetvli(x0, reg_vlmax, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    vfadd_vv(v_tmp0, v_tmp0, v_tmp1);
    vfadd_vv(v_tmp2, v_tmp2, v_tmp3);
    vfadd_vv(v_tmp0, v_tmp0, v_tmp2);
    vfmv_v_f(v_in0, f_zero);
    vfredusum_vs(v_in1, v_tmp0, v_in0);
    vfmv_f_s(f_aux, v_in1);
    fadd_s(f_sum, f_sum, f_aux);

    L(mean_done);
    ld(reg_len, reg_param, GET_FUSED_OFF(len));
    fcvt_s_l(f_len, reg_len);
    fdiv_s(f_mean, f_sum, f_len);

    ld(reg_src, reg_param, GET_FUSED_OFF(src));
    ld(reg_len, reg_param, GET_FUSED_OFF(len));
    fsgnj_s(f_var_sum, f_zero, f_zero);
    vfmv_v_f(v_tmp0, f_zero);
    vfmv_v_f(v_tmp1, f_zero);
    vfmv_v_f(v_tmp2, f_zero);
    vfmv_v_f(v_tmp3, f_zero);
    vsetvli(x0, reg_vlmax, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    vfmv_v_f(v_mean, f_mean);

    L(var_main_loop);
    blt(reg_len, reg_block, var_vec_tail);
    vle32_v(v_in0, reg_src);
    add(reg_tmp, reg_src, reg_bytes);
    vle32_v(v_in1, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in2, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in3, reg_tmp);
    vfsub_vv(v_in0, v_in0, v_mean);
    vfsub_vv(v_in1, v_in1, v_mean);
    vfsub_vv(v_in2, v_in2, v_mean);
    vfsub_vv(v_in3, v_in3, v_mean);
    vfmacc_vv(v_tmp0, v_in0, v_in0);
    vfmacc_vv(v_tmp1, v_in1, v_in1);
    vfmacc_vv(v_tmp2, v_in2, v_in2);
    vfmacc_vv(v_tmp3, v_in3, v_in3);
    add(reg_src, reg_src, reg_bytes4);
    sub(reg_len, reg_len, reg_block);
    j_(var_main_loop);

    L(var_vec_tail);
    beqz(reg_len, var_reduce_loop);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1, VTA::tu, VMA::ma);
    slli(reg_tmp, reg_vl, 2);
    vle32_v(v_in0, reg_src);
    vfsub_vv(v_in0, v_in0, v_mean);
    vfmacc_vv(v_tmp0, v_in0, v_in0);
    add(reg_src, reg_src, reg_tmp);
    sub(reg_len, reg_len, reg_vl);
    j_(var_vec_tail);

    L(var_reduce_loop);
    vsetvli(x0, reg_vlmax, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    vfadd_vv(v_tmp0, v_tmp0, v_tmp1);
    vfadd_vv(v_tmp2, v_tmp2, v_tmp3);
    vfadd_vv(v_tmp0, v_tmp0, v_tmp2);
    vfmv_v_f(v_in0, f_zero);
    vfredusum_vs(v_in1, v_tmp0, v_in0);
    vfmv_f_s(f_aux, v_in1);
    fadd_s(f_var_sum, f_var_sum, f_aux);

    L(var_done);
    ld(reg_len, reg_param, GET_FUSED_OFF(len));
    fcvt_s_l(f_len, reg_len);
    fdiv_s(f_var, f_var_sum, f_len);

    ld(reg_mean_ptr, reg_param, GET_FUSED_OFF(mean));
    beqz(reg_mean_ptr, store_var);
    fsw(f_mean, reg_mean_ptr, 0);

    L(store_var);
    ld(reg_var_ptr, reg_param, GET_FUSED_OFF(variance));
    beqz(reg_var_ptr, start_data);
    fsw(f_var, reg_var_ptr, 0);

    L(start_data);
    flw(f_aux, reg_param, GET_FUSED_OFF(eps));
    fadd_s(f_aux, f_var, f_aux);
    fsqrt_s(f_inv, f_aux);
    li(reg_tmp, 0x3f800000);
    fmv_w_x(f_one, reg_tmp);
    fdiv_s(f_inv, f_one, f_inv);

    ld(reg_src, reg_param, GET_FUSED_OFF(src));
    ld(reg_dst, reg_param, GET_FUSED_OFF(dst));
    ld(reg_scale, reg_param, GET_FUSED_OFF(scale));
    ld(reg_shift, reg_param, GET_FUSED_OFF(shift));
    ld(reg_len, reg_param, GET_FUSED_OFF(len));

    vsetvli(reg_vlmax, x0, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    slli(reg_bytes, reg_vlmax, 2);
    slli(reg_bytes4, reg_bytes, 2);
    slli(reg_block, reg_vlmax, 2);
    vfmv_v_f(v_mean, f_mean);
    vfmv_v_f(v_inv, f_inv);

    L(data_main_loop);
    blt(reg_len, reg_block, data_tail_loop);

    vle32_v(v_in0, reg_src);
    add(reg_tmp, reg_src, reg_bytes);
    vle32_v(v_in1, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in2, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in3, reg_tmp);

    vfsub_vv(v_in0, v_in0, v_mean);
    vfsub_vv(v_in1, v_in1, v_mean);
    vfsub_vv(v_in2, v_in2, v_mean);
    vfsub_vv(v_in3, v_in3, v_mean);
    vfmul_vv(v_in0, v_in0, v_inv);
    vfmul_vv(v_in1, v_in1, v_inv);
    vfmul_vv(v_in2, v_in2, v_inv);
    vfmul_vv(v_in3, v_in3, v_inv);

    if (with_scale_) {
        vle32_v(v_scale0, reg_scale);
        add(reg_tmp, reg_scale, reg_bytes);
        vle32_v(v_scale1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_scale2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_scale3, reg_tmp);
        if (with_shift_) {
            vle32_v(v_shift0, reg_shift);
            add(reg_tmp, reg_shift, reg_bytes);
            vle32_v(v_shift1, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vle32_v(v_shift2, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vle32_v(v_shift3, reg_tmp);
            vfmacc_vv(v_shift0, v_scale0, v_in0);
            vfmacc_vv(v_shift1, v_scale1, v_in1);
            vfmacc_vv(v_shift2, v_scale2, v_in2);
            vfmacc_vv(v_shift3, v_scale3, v_in3);
            vse32_v(v_shift0, reg_dst);
            add(reg_tmp, reg_dst, reg_bytes);
            vse32_v(v_shift1, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_shift2, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_shift3, reg_tmp);
        } else {
            vfmul_vv(v_in0, v_in0, v_scale0);
            vfmul_vv(v_in1, v_in1, v_scale1);
            vfmul_vv(v_in2, v_in2, v_scale2);
            vfmul_vv(v_in3, v_in3, v_scale3);
            vse32_v(v_in0, reg_dst);
            add(reg_tmp, reg_dst, reg_bytes);
            vse32_v(v_in1, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_in2, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_in3, reg_tmp);
        }
    } else if (with_shift_) {
        vle32_v(v_shift0, reg_shift);
        add(reg_tmp, reg_shift, reg_bytes);
        vle32_v(v_shift1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_shift2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_shift3, reg_tmp);
        vfadd_vv(v_in0, v_in0, v_shift0);
        vfadd_vv(v_in1, v_in1, v_shift1);
        vfadd_vv(v_in2, v_in2, v_shift2);
        vfadd_vv(v_in3, v_in3, v_shift3);
        vse32_v(v_in0, reg_dst);
        add(reg_tmp, reg_dst, reg_bytes);
        vse32_v(v_in1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in3, reg_tmp);
    } else {
        vse32_v(v_in0, reg_dst);
        add(reg_tmp, reg_dst, reg_bytes);
        vse32_v(v_in1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in3, reg_tmp);
    }

    add(reg_src, reg_src, reg_bytes4);
    add(reg_dst, reg_dst, reg_bytes4);
    if (with_scale_) add(reg_scale, reg_scale, reg_bytes4);
    if (with_shift_) add(reg_shift, reg_shift, reg_bytes4);
    sub(reg_len, reg_len, reg_block);
    j_(data_main_loop);

    L(data_tail_loop);
    beqz(reg_len, done);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    slli(reg_tmp, reg_vl, 2);
    vle32_v(v_in0, reg_src);
    vfsub_vv(v_in0, v_in0, v_mean);
    vfmul_vv(v_in0, v_in0, v_inv);
    if (with_scale_) {
        vle32_v(v_scale0, reg_scale);
        if (with_shift_) {
            vle32_v(v_shift0, reg_shift);
            vfmacc_vv(v_shift0, v_scale0, v_in0);
            vse32_v(v_shift0, reg_dst);
        } else {
            vfmul_vv(v_in0, v_in0, v_scale0);
            vse32_v(v_in0, reg_dst);
        }
    } else if (with_shift_) {
        vle32_v(v_shift0, reg_shift);
        vfadd_vv(v_in0, v_in0, v_shift0);
        vse32_v(v_in0, reg_dst);
    } else {
        vse32_v(v_in0, reg_dst);
    }
    add(reg_src, reg_src, reg_tmp);
    add(reg_dst, reg_dst, reg_tmp);
    if (with_scale_) add(reg_scale, reg_scale, reg_tmp);
    if (with_shift_) add(reg_shift, reg_shift, reg_tmp);
    sub(reg_len, reg_len, reg_vl);
    j_(data_tail_loop);

    L(done);
    ret();
}

jit_rvv_layernorm_data_kernel_t::jit_rvv_layernorm_data_kernel_t(
        bool with_scale, bool with_shift)
    : jit_generator_t("jit_rvv_layernorm_data_kernel")
    , with_scale_(with_scale)
    , with_shift_(with_shift) {
    create_kernel();
}

void jit_rvv_layernorm_data_kernel_t::generate() {
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_scale = a3;
    const Reg reg_shift = a4;
    const Reg reg_len = a5;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_tmp = t2;
    const Reg reg_vlmax = t3;
    const Reg reg_block = t4;
    const Reg reg_bytes4 = t5;

    const FReg f_mean = fa0;
    const FReg f_inv = fa1;

    const VReg v_in0(0);
    const VReg v_in1(1);
    const VReg v_in2(2);
    const VReg v_in3(3);
    const VReg v_mean(4);
    const VReg v_inv(5);
    const VReg v_scale0(6);
    const VReg v_scale1(7);
    const VReg v_scale2(8);
    const VReg v_scale3(9);
    const VReg v_shift0(10);
    const VReg v_shift1(11);
    const VReg v_shift2(12);
    const VReg v_shift3(13);

    ld(reg_src, reg_param, GET_DATA_OFF(src));
    ld(reg_dst, reg_param, GET_DATA_OFF(dst));
    ld(reg_scale, reg_param, GET_DATA_OFF(scale));
    ld(reg_shift, reg_param, GET_DATA_OFF(shift));
    ld(reg_len, reg_param, GET_DATA_OFF(len));
    flw(f_mean, reg_param, GET_DATA_OFF(mean));
    flw(f_inv, reg_param, GET_DATA_OFF(inv_std));

    vsetvli(reg_vlmax, x0, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    slli(reg_bytes, reg_vlmax, 2);
    slli(reg_bytes4, reg_bytes, 2);
    slli(reg_block, reg_vlmax, 2);

    Label main_loop, tail_loop, done;

    L(main_loop);
    blt(reg_len, reg_block, tail_loop);

    vfmv_v_f(v_mean, f_mean);
    vfmv_v_f(v_inv, f_inv);

    vle32_v(v_in0, reg_src);
    add(reg_tmp, reg_src, reg_bytes);
    vle32_v(v_in1, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in2, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in3, reg_tmp);

    vfsub_vv(v_in0, v_in0, v_mean);
    vfsub_vv(v_in1, v_in1, v_mean);
    vfsub_vv(v_in2, v_in2, v_mean);
    vfsub_vv(v_in3, v_in3, v_mean);
    vfmul_vv(v_in0, v_in0, v_inv);
    vfmul_vv(v_in1, v_in1, v_inv);
    vfmul_vv(v_in2, v_in2, v_inv);
    vfmul_vv(v_in3, v_in3, v_inv);

    if (with_scale_) {
        vle32_v(v_scale0, reg_scale);
        add(reg_tmp, reg_scale, reg_bytes);
        vle32_v(v_scale1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_scale2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_scale3, reg_tmp);
        if (with_shift_) {
            vle32_v(v_shift0, reg_shift);
            add(reg_tmp, reg_shift, reg_bytes);
            vle32_v(v_shift1, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vle32_v(v_shift2, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vle32_v(v_shift3, reg_tmp);
            vfmacc_vv(v_shift0, v_scale0, v_in0);
            vfmacc_vv(v_shift1, v_scale1, v_in1);
            vfmacc_vv(v_shift2, v_scale2, v_in2);
            vfmacc_vv(v_shift3, v_scale3, v_in3);
            vse32_v(v_shift0, reg_dst);
            add(reg_tmp, reg_dst, reg_bytes);
            vse32_v(v_shift1, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_shift2, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_shift3, reg_tmp);
        } else {
            vfmul_vv(v_in0, v_in0, v_scale0);
            vfmul_vv(v_in1, v_in1, v_scale1);
            vfmul_vv(v_in2, v_in2, v_scale2);
            vfmul_vv(v_in3, v_in3, v_scale3);
            vse32_v(v_in0, reg_dst);
            add(reg_tmp, reg_dst, reg_bytes);
            vse32_v(v_in1, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_in2, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_in3, reg_tmp);
        }
    } else if (with_shift_) {
        vle32_v(v_shift0, reg_shift);
        add(reg_tmp, reg_shift, reg_bytes);
        vle32_v(v_shift1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_shift2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_shift3, reg_tmp);
        vfadd_vv(v_in0, v_in0, v_shift0);
        vfadd_vv(v_in1, v_in1, v_shift1);
        vfadd_vv(v_in2, v_in2, v_shift2);
        vfadd_vv(v_in3, v_in3, v_shift3);
        vse32_v(v_in0, reg_dst);
        add(reg_tmp, reg_dst, reg_bytes);
        vse32_v(v_in1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in3, reg_tmp);
    } else {
        vse32_v(v_in0, reg_dst);
        add(reg_tmp, reg_dst, reg_bytes);
        vse32_v(v_in1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in3, reg_tmp);
    }

    add(reg_src, reg_src, reg_bytes4);
    add(reg_dst, reg_dst, reg_bytes4);
    if (with_scale_) add(reg_scale, reg_scale, reg_bytes4);
    if (with_shift_) add(reg_shift, reg_shift, reg_bytes4);
    sub(reg_len, reg_len, reg_block);
    j_(main_loop);

    L(tail_loop);
    beqz(reg_len, done);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    slli(reg_tmp, reg_vl, 2);
    vfmv_v_f(v_mean, f_mean);
    vfmv_v_f(v_inv, f_inv);
    vle32_v(v_in0, reg_src);
    vfsub_vv(v_in0, v_in0, v_mean);
    vfmul_vv(v_in0, v_in0, v_inv);
    if (with_scale_) {
        vle32_v(v_scale0, reg_scale);
        if (with_shift_) {
            vle32_v(v_shift0, reg_shift);
            vfmacc_vv(v_shift0, v_scale0, v_in0);
            vse32_v(v_shift0, reg_dst);
        } else {
            vfmul_vv(v_in0, v_in0, v_scale0);
            vse32_v(v_in0, reg_dst);
        }
    } else if (with_shift_) {
        vle32_v(v_shift0, reg_shift);
        vfadd_vv(v_in0, v_in0, v_shift0);
        vse32_v(v_in0, reg_dst);
    } else {
        vse32_v(v_in0, reg_dst);
    }
    add(reg_src, reg_src, reg_tmp);
    add(reg_dst, reg_dst, reg_tmp);
    if (with_scale_) add(reg_scale, reg_scale, reg_tmp);
    if (with_shift_) add(reg_shift, reg_shift, reg_tmp);
    sub(reg_len, reg_len, reg_vl);
    j_(tail_loop);

    L(done);
    ret();
}

#undef GET_FUSED_OFF
#undef GET_DATA_OFF

#define GET_F16_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_layernorm_f16_fused_kernel_t::call_params_t, field))

jit_rvv_layernorm_f16_fused_kernel_t::jit_rvv_layernorm_f16_fused_kernel_t(
        bool with_scale, bool with_shift, bool weights_f16,
        bool use_widening_sum)
    : jit_generator_t("jit_rvv_layernorm_f16_fused_kernel")
    , with_scale_(with_scale)
    , with_shift_(with_shift)
    , weights_f16_(weights_f16)
    , use_widening_sum_(use_widening_sum) {
    create_kernel();
}

void jit_rvv_layernorm_f16_fused_kernel_t::generate() {
    const bool use_widening_sum = use_widening_sum_;
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_scale = a3;
    const Reg reg_shift = a4;
    const Reg reg_len = a5;
    const Reg reg_t1 = a6; // remaining element count
    const Reg reg_ptr = a7; // mean/variance pointer scratch
    const Reg reg_vl = t0; // current vsetvli element count
    const Reg reg_tmp = t1;
    const Reg reg_tmp2 = t2;

    const FReg f_zero = ft0;
    const FReg f_mean = ft1; // mean (also holds Sx before divide)
    const FReg f_var_sum = ft2;
    const FReg f_len = ft3;
    const FReg f_inv = ft4; // 1/sqrt(var+eps), f32
    const FReg f_var = ft5;
    const FReg f_eps = ft6;
    const FReg d_b = ft8;
    const FReg d_eps = ft9;
    const FReg d_one = ft10;

    const VReg v_sum(0); // m8: Sx accumulator
    const VReg v_sumsq(8); // m8: variance accumulator
    const VReg v_work(16); // m8: widened x / compute register
    const VReg v_ld(24); // m4: f16 load temp
    const VReg v_gamma(24); // m8: widened/loaded gamma
    const VReg v_gld(8); // m4: gamma f16 load temp
    const VReg v_beta(8); // m8: widened/loaded beta
    const VReg v_bld(4); // m4: beta f16 load temp

    fmv_w_x(f_zero, x0);

    Label sum_loop, sum_done, var_loop, var_done, skip_mean, skip_var;
    Label out_loop, out_done;

    // ---- pass 1: accumulate Sx in f32 ----
    ld(reg_src, reg_param, GET_F16_OFF(src));
    ld(reg_t1, reg_param, GET_F16_OFF(len));
    vsetvli(reg_tmp, x0, SEW::e32, LMUL::m8, VTA::ta, VMA::ma);
    vmv_v_x(v_sum, x0);
    if (!use_widening_sum) vmv_v_x(v_work, x0);

    L(sum_loop);
    beqz(reg_t1, sum_done);
    if (use_widening_sum) {
        // Preserve inactive accumulator lanes for the final full-width
        // reduction; they may contain sums from preceding full chunks.
        vsetvli(reg_vl, reg_t1, SEW::e16, LMUL::m4, VTA::tu, VMA::ma);
    } else {
        vsetvli(reg_vl, reg_t1, SEW::e16, LMUL::m4, VTA::ta, VMA::ma);
    }
    sub(reg_t1, reg_t1, reg_vl);
    vle16_v(v_ld, reg_src);
    slli(reg_tmp, reg_vl, 1);
    add(reg_src, reg_src, reg_tmp);
    if (use_widening_sum) {
        vfwadd_wv(v_sum, v_sum, v_ld);
    } else {
        vfwcvt_f_f_v(v_work, v_ld);
        vsetvli(reg_tmp, reg_vl, SEW::e32, LMUL::m8, VTA::tu, VMA::ma);
        vfadd_vv(v_sum, v_sum, v_work);
        vmv_v_x(v_work, x0);
    }
    j_(sum_loop);
    L(sum_done);

    // ---- finish mean ----
    vsetvli(reg_tmp, x0, SEW::e32, LMUL::m8, VTA::ta, VMA::ma);
    vmv_v_v(v_work, v_sum);
    vmv_v_x(v_sum, x0); // reuse as reduction init (elem0 = 0)
    vfredosum_vs(v_work, v_work, v_sum);
    vfmv_f_s(f_mean, v_work); // Sx

    ld(reg_len, reg_param, GET_F16_OFF(len));
    fcvt_s_l(f_len, reg_len);
    fdiv_s(f_mean, f_mean, f_len); // mean

    ld(reg_ptr, reg_param, GET_F16_OFF(mean));
    beqz(reg_ptr, skip_mean);
    fsw(f_mean, reg_ptr, 0);
    L(skip_mean);

    // ---- pass 2: accumulate variance as sum((x - mean)^2) ----
    ld(reg_src, reg_param, GET_F16_OFF(src));
    ld(reg_t1, reg_param, GET_F16_OFF(len));
    vsetvli(reg_tmp, x0, SEW::e32, LMUL::m8, VTA::ta, VMA::ma);
    vmv_v_x(v_sumsq, x0);
    vmv_v_x(v_work, x0);

    L(var_loop);
    beqz(reg_t1, var_done);
    vsetvli(reg_vl, reg_t1, SEW::e16, LMUL::m4, VTA::ta, VMA::ma);
    sub(reg_t1, reg_t1, reg_vl);
    vle16_v(v_ld, reg_src);
    slli(reg_tmp, reg_vl, 1);
    add(reg_src, reg_src, reg_tmp);
    vfwcvt_f_f_v(v_work, v_ld); // f16 -> f32
    vsetvli(reg_tmp, reg_vl, SEW::e32, LMUL::m8, VTA::tu, VMA::ma);
    vfsub_vf(v_work, v_work, f_mean);
    vfmacc_vv(v_sumsq, v_work, v_work);
    vmv_v_x(v_work, x0); // keep tail zero for next widen
    j_(var_loop);
    L(var_done);

    vsetvli(reg_tmp, x0, SEW::e32, LMUL::m8, VTA::ta, VMA::ma);
    vmv_v_x(v_sum, x0); // reuse as reduction init
    vfredosum_vs(v_sumsq, v_sumsq, v_sum);
    vfmv_f_s(f_var_sum, v_sumsq);
    fdiv_s(f_var, f_var_sum, f_len);
    ld(reg_ptr, reg_param, GET_F16_OFF(variance));
    beqz(reg_ptr, skip_var);
    fsw(f_var, reg_ptr, 0);
    L(skip_var);

    flw(f_eps, reg_param, GET_F16_OFF(eps));
    fcvt_d_s(d_b, f_var);
    fcvt_d_s(d_eps, f_eps);
    fadd_d(d_b, d_b, d_eps); // var + eps (f64)
    fsqrt_d(d_b, d_b);
    li(reg_tmp, 1);
    fcvt_d_w(d_one, reg_tmp);
    fdiv_d(d_b, d_one, d_b); // 1/sqrt
    fcvt_s_d(f_inv, d_b); // inv (f32)

    // ---- pass 3: normalize + affine, write f16 ----
    ld(reg_src, reg_param, GET_F16_OFF(src));
    ld(reg_dst, reg_param, GET_F16_OFF(dst));
    if (with_scale_) ld(reg_scale, reg_param, GET_F16_OFF(scale));
    if (with_shift_) ld(reg_shift, reg_param, GET_F16_OFF(shift));
    ld(reg_t1, reg_param, GET_F16_OFF(len));

    L(out_loop);
    beqz(reg_t1, out_done);
    vsetvli(reg_vl, reg_t1, SEW::e16, LMUL::m4, VTA::ta, VMA::ma);
    sub(reg_t1, reg_t1, reg_vl);

    vle16_v(v_ld, reg_src);
    vfwcvt_f_f_v(v_work, v_ld); // x -> f32

    if (weights_f16_) {
        if (with_scale_) {
            vle16_v(v_gld, reg_scale);
            vfwcvt_f_f_v(v_gamma, v_gld); // gamma -> f32 (v24 m8)
        }
        if (with_shift_) {
            vle16_v(v_bld, reg_shift);
            vfwcvt_f_f_v(v_beta, v_bld); // beta -> f32 (v8 m8)
        }
    } else if (with_scale_ || with_shift_) {
        vsetvli(reg_tmp, reg_vl, SEW::e32, LMUL::m8, VTA::ta, VMA::ma);
        if (with_scale_) vle32_v(v_gamma, reg_scale);
        if (with_shift_) vle32_v(v_beta, reg_shift);
    }

    vsetvli(reg_tmp, x0, SEW::e32, LMUL::m8, VTA::ta, VMA::ma);
    vfsub_vf(v_work, v_work, f_mean); // x - mean
    vfmul_vf(v_work, v_work, f_inv); // * inv
    if (with_scale_) vfmul_vv(v_work, v_work, v_gamma);
    if (with_shift_) vfadd_vv(v_work, v_work, v_beta);

    vsetvli(reg_tmp, reg_vl, SEW::e16, LMUL::m4, VTA::ta, VMA::ma);
    vfncvt_f_f_w(v_ld, v_work); // f32 -> f16
    vse16_v(v_ld, reg_dst);

    slli(reg_tmp, reg_vl, 1);
    add(reg_src, reg_src, reg_tmp);
    add(reg_dst, reg_dst, reg_tmp);
    if (with_scale_ || with_shift_) {
        slli(reg_tmp2, reg_vl, weights_f16_ ? 1 : 2);
        if (with_scale_) add(reg_scale, reg_scale, reg_tmp2);
        if (with_shift_) add(reg_shift, reg_shift, reg_tmp2);
    }
    j_(out_loop);
    L(out_done);
    ret();
}

#undef GET_F16_OFF

jit_rvv_layernorm_bwd_kernel_t::jit_rvv_layernorm_bwd_kernel_t(
        bool scale, bool stats, bool weights)
    : jit_generator_t("jit_rvv_layernorm_bwd_kernel")
    , scale_(scale)
    , stats_(stats)
    , weights_(weights) {
    status_ = create_kernel();
}

void jit_rvv_layernorm_bwd_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg src = a1, dd = a2, ds = a3, gamma = a4;
    const Reg dg = a5, db = a6, len = a7;
    const Reg vl = t0, bytes = t1;
    const FReg mean = ft0, inv = ft1, channels = ft2;
    const FReg sum = ft3, sum_x = ft4, zero = ft5;
    const VReg x(0), dy(1), u(2), tmp(3), acc(4), g(5);
    const VReg v_sum(8), v_sum_x(9), reduction(10);
#define BWD_OFF(field) static_cast<int32_t>(offsetof(call_params_t, field))
    auto load_row = [&]() {
        ld(src, a0, BWD_OFF(src));
        ld(dd, a0, BWD_OFF(diff_dst));
        ld(len, a0, BWD_OFF(len));
        if (scale_) ld(gamma, a0, BWD_OFF(scale));
    };
    flw(mean, a0, BWD_OFF(mean));
    flw(inv, a0, BWD_OFF(inv_std));
    flw(channels, a0, BWD_OFF(channels));
    fmv_w_x(zero, x0);
    if (stats_ || weights_) {
        load_row();
        if (weights_) {
            ld(dg, a0, BWD_OFF(diff_scale));
            ld(db, a0, BWD_OFF(diff_shift));
        }
        // Initialize all lanes, preserving inactive accumulator lanes in
        // the final partial vector before reducing at VLMAX.
        if (stats_) {
            vsetvli(vl, x0, SEW::e32, LMUL::m1, VTA::tu, VMA::ma);
            vmv_v_i(v_sum, 0);
            vmv_v_i(v_sum_x, 0);
        }
        Label reduce_loop;
        L(reduce_loop);
        vsetvli(vl, len, SEW::e32, LMUL::m1, VTA::tu, VMA::ma);
        vle32_v(x, src);
        vle32_v(dy, dd);
        vfsub_vf(x, x, mean);
        if (weights_) {
            vfmul_vv(tmp, x, dy);
            vfmul_vf(tmp, tmp, inv);
            vle32_v(acc, dg);
            vfadd_vv(acc, acc, tmp);
            vse32_v(acc, dg);
            vle32_v(acc, db);
            vfadd_vv(acc, acc, dy);
            vse32_v(acc, db);
        }
        if (stats_) {
            if (scale_) {
                vle32_v(g, gamma);
                vfmul_vv(u, dy, g);
            } else {
                vmv_v_v(u, dy);
            }
            vfadd_vv(v_sum, v_sum, u);
            vfmul_vv(tmp, u, x);
            vfadd_vv(v_sum_x, v_sum_x, tmp);
        }
        slli(bytes, vl, 2);
        add(src, src, bytes);
        add(dd, dd, bytes);
        if (scale_ && stats_) add(gamma, gamma, bytes);
        if (weights_) {
            add(dg, dg, bytes);
            add(db, db, bytes);
        }
        sub(len, len, vl);
        bnez(len, reduce_loop);
        if (stats_) {
            vsetvli(vl, x0, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
            vfmv_s_f(reduction, zero);
            vfredusum_vs(reduction, v_sum, reduction);
            vfmv_f_s(sum, reduction);
            vfmv_s_f(reduction, zero);
            vfredusum_vs(reduction, v_sum_x, reduction);
            vfmv_f_s(sum_x, reduction);
            fdiv_s(sum, sum, channels);
            fmul_s(sum_x, sum_x, inv);
        }
    }
    load_row();
    ld(ds, a0, BWD_OFF(diff_src));
    Label apply_loop;
    L(apply_loop);
    vsetvli(vl, len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    vle32_v(dy, dd);
    if (scale_) {
        vle32_v(g, gamma);
        vfmul_vv(dy, dy, g);
    }
    if (stats_) {
        vle32_v(x, src);
        vfsub_vf(x, x, mean);
        vfmul_vf(x, x, sum_x);
        vfmul_vf(x, x, inv);
        vfdiv_vf(x, x, channels);
        vfsub_vf(dy, dy, sum);
        vfsub_vv(dy, dy, x);
    }
    vfmul_vf(dy, dy, inv);
    vse32_v(dy, ds);
    slli(bytes, vl, 2);
    add(src, src, bytes);
    add(dd, dd, bytes);
    add(ds, ds, bytes);
    if (scale_) add(gamma, gamma, bytes);
    sub(len, len, vl);
    bnez(len, apply_loop);
#undef BWD_OFF
#endif
    ret();
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
