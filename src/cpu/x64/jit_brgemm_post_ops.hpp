/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_POST_OPS_HPP
#define CPU_X64_JIT_BRGEMM_POST_OPS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_engine.hpp"

#include "cpu/x64/jit_brgemm_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct brgemm_kernel_diff_bias_t {
    void *ptr_diff_dst;
    void *ptr_diff_bias_acc;
    void *ptr_diff_bias;
    int flags;
};

#define GET_OFF(field) offsetof(brgemm_kernel_diff_bias_t, field)

struct jit_brgemm_kernel_diff_bias_t : public jit_generator {
    jit_brgemm_kernel_diff_bias_t(
            const jit_brgemm_primitive_conf_t &ajbgp, const brgemm_t &abrg)
        : brg_(abrg)
        , ddst_dt_(ajbgp.dst_dt)
        , bia_dt_(ajbgp.bia_dt)
        , acc_dt_(ajbgp.acc_dt)
        , ddst_typesize_(types::data_type_size(ddst_dt_))
        , bia_typesize_(types::data_type_size(bia_dt_))
        , acc_typesize_(types::data_type_size(acc_dt_)) {}

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_diff_bias_t)

private:
    brgemm_t brg_;
    data_type_t ddst_dt_;
    data_type_t bia_dt_;
    data_type_t acc_dt_;

    int ddst_typesize_;
    int bia_typesize_;
    int acc_typesize_;

    using reg64_t = const Xbyak::Reg64;
    // Register decomposition
    const reg64_t param1 = abi_param1;
    const reg64_t reg_ddst = r15;
    const reg64_t reg_bias = r14;
    const reg64_t reg_bias_acc = r13;
    const reg64_t aux_reg_ddst = r12;
    const reg64_t reg_k_iter = r11;
    const reg64_t reg_flag = r10;

    Xbyak::Opmask k_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_tail_mask = Xbyak::Opmask(3);
    Xbyak::Opmask k_tail_store_mask = Xbyak::Opmask(4);
    Xbyak::Zmm vreg_unit = Xbyak::Zmm(31);

    const int n_max_regs_ = 4;

    const Xbyak::Zmm zmm_mask(const Xbyak::Zmm zmm_in, bool mask_flag,
            bool store, Xbyak::Opmask ktail_mask) {
        return mask_flag
                ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                : zmm_in;
    }

    void loop_by_N(int n_loop, int nb_tail) {

        mov(aux_reg_ddst, reg_ddst);
        int mult = ddst_dt_ == data_type::bf16 ? 2 : 1;
        int n_iters = n_loop;
        if (nb_tail > 0) n_iters--;
        Xbyak::Label k_loop, init_zero, init_done;
        auto get_bias_reg = [=](int n) { return Xbyak::Zmm(n); };
        auto get_bias_reg_lower = [=](int n) { return Xbyak::Ymm(n); };
        auto get_ddst_reg = [=](int n) { return Xbyak::Zmm(n + n_max_regs_); };
        int n_ = 0;

        test(reg_flag, FLAG_REDUCE_FIRST);
        jnz(init_zero, T_NEAR); // FLAG_REDUCE_FIRST is set

        for (; n_ < n_iters; n_++) {
            auto vbias = get_bias_reg(n_);
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            vmovups(vbias, addr);
        }
        if (nb_tail > 0) {
            auto vbias = zmm_mask(get_bias_reg(n_), true, false, k_tail_mask);
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            vmovups(vbias, addr);
        }
        jmp(init_done, T_NEAR);
        L(init_zero);

        for (int n_ = 0; n_ < n_loop; n_++) {
            vxorpd(get_bias_reg(n_), get_bias_reg(n_), get_bias_reg(n_));
        }
        L(init_done);

        mov(reg_k_iter, utils::div_up(brg_.reduce_dim, mult));
        L(k_loop);
        {
            int n_ = 0;
            for (; n_ < n_iters; n_++) {
                auto vddst = get_ddst_reg(n_);
                auto vbias = get_bias_reg(n_);
                auto addr = ptr[aux_reg_ddst
                        + ddst_typesize_ * mult * n_ * brg_.ld_block];
                vmovups(vddst, addr);
                if (ddst_dt_ == data_type::bf16)
                    vdpbf16ps(vbias, vreg_unit, vddst);
                else
                    vaddps(vbias, vbias, vddst);
            }

            if (nb_tail > 0) {
                auto vddst = get_ddst_reg(n_);
                auto vddst_load = zmm_mask(vddst, true, false, k_tail_mask);
                auto vbias = get_bias_reg(n_);

                auto addr = ptr[aux_reg_ddst
                        + ddst_typesize_ * mult * n_ * brg_.ld_block];
                vmovups(vddst_load, addr);
                if (ddst_dt_ == data_type::bf16)
                    vdpbf16ps(vbias, vreg_unit, vddst);
                else
                    vaddps(vbias, vbias, vddst);
            }

            add(aux_reg_ddst, ddst_typesize_ * mult * brg_.LDB);

            sub(reg_k_iter, 1);
            jnz(k_loop, T_NEAR);
        }

        Xbyak::Label store_final, store_done;
        test(reg_flag, FLAG_REDUCE_LAST);
        jnz(store_final, T_NEAR); // FLAG_REDUCE_LAST is set

        n_ = 0;
        for (; n_ < n_iters; n_++) {
            auto vbias = get_bias_reg(n_);
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            vmovups(addr, vbias);
        }
        if (nb_tail > 0) {
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            auto vbias = zmm_mask(get_bias_reg(n_), true, true, k_tail_mask);
            vmovups(addr, vbias);
        }
        jmp(store_done, T_NEAR);

        L(store_final);
        n_ = 0;
        for (; n_ < n_iters; n_++) {
            auto vbias = get_bias_reg(n_);
            auto addr = ptr[reg_bias + bia_typesize_ * n_ * brg_.ld_block];
            if (bia_dt_ == data_type::bf16) {
                auto vbias_lower = get_bias_reg_lower(n_);
                vcvtneps2bf16(vbias_lower, vbias);
                vmovups(addr, vbias_lower);
            } else
                vmovups(addr, vbias);
        }
        if (nb_tail > 0) {
            auto addr = ptr[reg_bias + bia_typesize_ * n_ * brg_.ld_block];
            if (bia_dt_ == data_type::bf16) {
                auto vbias = get_bias_reg(n_);
                auto vbias_lower = get_bias_reg_lower(n_);
                vcvtneps2bf16(vbias_lower, vbias);
                auto vbias_store = zmm_mask(vbias, true, true, k_tail_mask);
                vmovdqu16(addr, vbias_store);
            } else {
                auto vbias
                        = zmm_mask(get_bias_reg(n_), true, true, k_tail_mask);
                vmovups(addr, vbias);
            }
        }
        L(store_done);
    }

    void generate() override {
        preamble();

        int nb = utils::div_up(brg_.load_dim, brg_.ld_block);
        int nb_tail = brg_.load_dim % brg_.ld_block;

        int n_loop = nb / n_max_regs_;
        int n_loop_tail = nb % n_max_regs_;
        if (n_loop_tail == 0 && nb_tail > 0) {
            n_loop--;
            n_loop_tail = n_max_regs_;
        }

        const auto full_mask = size_t {0xffffffffffffffff};
        const auto tail_mask = size_t((1 << nb_tail) - 1);
        reg64_t reg_mask = rax;

        mov(reg_mask, full_mask);
        kmovq(k_full_mask, reg_mask);
        mov(reg_mask, tail_mask);
        kmovq(k_tail_mask, reg_mask);

        if (ddst_dt_ == data_type::bf16) {
            auto reg_unit_val = reg_mask.cvt16();
            mov(reg_unit_val, 0x3f80); // bf16 value of 1.
            vpbroadcastw(vreg_unit, reg_unit_val);
        }

        mov(reg_ddst, ptr[param1 + GET_OFF(ptr_diff_dst)]);
        mov(reg_bias_acc, ptr[param1 + GET_OFF(ptr_diff_bias_acc)]);
        mov(reg_bias, ptr[param1 + GET_OFF(ptr_diff_bias)]);
        mov(reg_flag, ptr[param1 + GET_OFF(flags)]);

        int mult = ddst_dt_ == data_type::bf16 ? 2 : 1;
        for (int nb_ = 0; nb_ < n_loop; nb_++) {
            loop_by_N(n_max_regs_, 0);

            add(reg_ddst, ddst_typesize_ * mult * n_max_regs_ * brg_.ld_block);
            add(reg_bias, bia_typesize_ * n_max_regs_ * brg_.ld_block);
            add(reg_bias_acc, acc_typesize_ * n_max_regs_ * brg_.ld_block);
        }

        if (n_loop_tail > 0) loop_by_N(n_loop_tail, nb_tail);
        postamble();
    }
};

#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
