/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_engine.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_brgemm_primitive_conf.hpp"
#include "cpu/x64/jit_generator.hpp"

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

#define GET_OFF(field) offsetof(brgemm_kernel_post_ops_t, field)

struct brgemm_kernel_post_ops_t {
    void *ptr_in;
    void *ptr_out;
    void *ptr_bias;
    void *ptr_scales;
    const void *ptr_binary_post_ops_rhs;
    size_t oc_l_offset;
    size_t apply_comp = 0;
    int32_t a_comp_val = 1;
    int32_t *a_zp_compensation;
    int32_t *c_zp_values;
    int32_t *s8s8_compensation;
    const void *dst_orig;
};

struct jit_brgemm_kernel_post_ops : public jit_generator {

    jit_brgemm_kernel_post_ops(const jit_brgemm_conv_conf_t &ajcp,
            const brgemm_t &abrg, const primitive_attr_t &aattr)
        : brg(abrg)
        , jcp(ajcp)
        , attr(aattr)
        , postops_injector_(nullptr)
        , with_binary_per_oc_bcast_(brg.with_binary
                  && binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                          brg.attr->post_ops_,
                          memory_desc_wrapper(brg.dst_md))) {

        if ((jcp.with_sum && brg.beta != 0)
                || ((jcp.with_binary || jcp.with_eltwise) && brg.alpha != 0)) {
            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    static_cast<size_t>(Xbyak::Zmm(28).getIdx()), this->r14,
                    this->r15, preserve_gpr, preserve_vmm,
                    GET_OFF(ptr_binary_post_ops_rhs), GET_OFF(dst_orig),
                    memory_desc_wrapper(brg.dst_md),
                    static_cast<size_t>(brg.load_dim % brg.ld_block),
                    k_tail_mask, use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {this->param1, rhs_sp};

            const bool save_state = (brg.alpha != 0) && jcp.with_eltwise;
            const auto &reserved_eltwise_gpr = rax;
            const auto reserved_eltwise_maskr = Xbyak::Opmask(1);

            const eltwise_injector::static_params_t esp {
                    save_state, reserved_eltwise_gpr, reserved_eltwise_maskr};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<avx512_core>>(
                    this, attr.post_ops_, bsp, esp);
        }
        if (brg.is_bf16_emu)
            bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                    bf16_emu_reserv_1, bf16_emu_reserv_2, bf16_emu_reserv_3,
                    bf16_emu_scratch, bf16_emu_reserv_4, bf16_emu_reserv_4);

        const auto &oscales = attr.output_scales_;
        is_oc_scale_ = oscales.mask_ == 1 << 1;

        LDD_ = brg.LDD;
        inp_dt_ = brg.dt_c;
        out_dt_ = brg.dt_d;
        bia_dt_ = jcp.bia_dt;
        inp_typesize_ = types::data_type_size(inp_dt_);
        out_typesize_ = types::data_type_size(out_dt_);
        bia_typesize_ = (jcp.with_bias) ? types::data_type_size(bia_dt_) : 0;
    }

    ~jit_brgemm_kernel_post_ops() = default;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_post_ops)

    brgemm_t brg;
    jit_brgemm_conv_conf_t jcp;
    const primitive_attr_t &attr;

private:
    int LDD_;

    data_type_t inp_dt_;
    data_type_t out_dt_;
    data_type_t bia_dt_;

    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core>>
            postops_injector_;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    const bool with_binary_per_oc_bcast_;

    int inp_typesize_;
    int out_typesize_;
    int bia_typesize_;

    int is_oc_scale_;

    using reg64_t = const Xbyak::Reg64;

    // Register decomposition
    const reg64_t param1 = abi_param1;
    const reg64_t reg_in = r15;
    const reg64_t reg_out = r14;
    const reg64_t aux_reg_in = r13;
    const reg64_t aux_reg_out = r12;

    const reg64_t reg_bias = r11;
    const reg64_t aux_reg_bias = r10;

    const reg64_t reg_scales = r9;
    const reg64_t aux_reg_scales = r8;

    const reg64_t reg_ptr_sum_scale = rdx;
    const reg64_t reg_ptr_sum_zp = rsi;

    const reg64_t reg_oc_l_offset_ = abi_not_param1;
    const reg64_t aux_reg_oc_l_offset_ = rbx;

    const reg64_t reg_zp_c_values = rbx;
    const reg64_t aux_reg_zp_c_values = rbx;
    const reg64_t reg_zp_a_comp = rbx;
    const reg64_t aux_reg_zp_a_comp = rbx;
    const reg64_t reg_s8s8_comp = rbx;
    const reg64_t aux_reg_s8s8_comp = rbx;
    const reg64_t reg_zp_a_val = rbx;
    const reg64_t reg_apply_comp = rbx;

    constexpr static int reg_aux_oc_l_offset_offs_ = 0;
    constexpr static int reg_zp_c_values_offs_ = 8;
    constexpr static int aux_reg_zp_c_values_offs_ = 16;
    constexpr static int reg_zp_a_comp_offs_ = 24;
    constexpr static int aux_reg_zp_a_comp_offs_ = 32;
    constexpr static int reg_s8s8_comp_offs_ = 40;
    constexpr static int aux_reg_s8s8_comp_offs_ = 48;
    constexpr static int reg_zp_a_val_offs_ = 56;
    constexpr static int reg_apply_comp_offs_ = 64;
    constexpr static int stack_space_needed_ = 72;

    /* bf16 emulation */
    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(24);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(25);
    Xbyak::Zmm bf16_emu_reserv_4 = Xbyak::Zmm(26);
    reg64_t bf16_emu_scratch = rax;

    Xbyak::Opmask k_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_tail_mask = Xbyak::Opmask(3);

    const int n_block2_ = 4;

    int zp_c_values_offset(int n, bool is_tail = false) const noexcept {
        if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
            return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                             : sizeof(int32_t) * n * brg.ld_block;
        }

        return 0;
    }
    int zp_comp_a_vpad_offset(int n, int m, bool is_tail = false) const
            noexcept {
        return (is_tail) ? sizeof(int32_t) * (brg.ldb_tail + m * brg.LDB)
                         : sizeof(int32_t) * (n * brg.ld_block + m * brg.LDB);
    }
    int mb_zp_comp_a_offset(int m_block) const noexcept {
        return sizeof(int32_t) * m_block * brg.LDB;
    }
    int compensation_vpad_offset(int n, int m, bool is_tail = false) const
            noexcept {
        return (is_tail) ? sizeof(int32_t) * (brg.ldb_tail + m * brg.LDB)
                         : sizeof(int32_t) * (n * brg.ld_block + m * brg.LDB);
    }
    int mb_compensation_offset(int m_block) const noexcept {
        return sizeof(int32_t) * m_block * brg.LDB;
    }

    const Xbyak::Zmm zmm_mask(const Xbyak::Zmm zmm_in, bool mask_flag,
            bool store, Xbyak::Opmask ktail_mask) {
        return mask_flag
                ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                : zmm_in;
    }

    const Xbyak::Ymm ymm_mask(const Xbyak::Ymm ymm_in, bool mask_flag,
            bool store, Xbyak::Opmask ktail_mask) {
        return mask_flag
                ? (store ? ymm_in | ktail_mask : ymm_in | ktail_mask | T_z)
                : ymm_in;
    }

    void cvt2ps(data_type_t type_in, const Xbyak::Zmm zmm_in,
            const Xbyak::Operand &op, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) {
        const Xbyak::Zmm zmm = zmm_mask(zmm_in, mask_flag, store, ktail_mask);
        switch (type_in) {
            case data_type::f32:
            case data_type::s32: vmovups(zmm, op); break;
            case data_type::s8: vpmovsxbd(zmm, op); break;
            case data_type::u8: vpmovzxbd(zmm, op); break;
            case data_type::bf16:
                vpmovzxwd(zmm, op);
                vpslld(zmm, zmm, 16);
                break;
            default: assert(!"unsupported data type");
        }
        if (!utils::one_of(type_in, data_type::f32, data_type::bf16))
            vcvtdq2ps(zmm_in, zmm_in);
    }

    Xbyak::Zmm vector(int m, int n, int n_block) {
        return Xbyak::Zmm(m * n_block + n);
    };

    void advance_mb_post_ops_regs(int m_block) {
        if (brg.alpha != 0) {
            if (brg.zp_type_a != brgemm_broadcast_t::none) {
                mov(reg_zp_a_comp, ptr[rsp + reg_zp_a_comp_offs_]);
                add(reg_zp_a_comp, mb_zp_comp_a_offset(m_block));
                mov(ptr[rsp + reg_zp_a_comp_offs_], reg_zp_a_comp);
            }
            if (brg.req_s8s8_compensation) {
                mov(reg_s8s8_comp, ptr[rsp + reg_s8s8_comp_offs_]);
                add(reg_s8s8_comp, mb_compensation_offset(m_block));
                mov(ptr[rsp + reg_s8s8_comp_offs_], reg_s8s8_comp);
            }
        }
    }

    void inject_attr_postops(int m_block, int n_block, int tail = 0) {
        const auto &p = attr.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const auto k_mask = tail == 0 ? k_full_mask : k_tail_mask;
        const auto sum_dt = p.get_sum_dt(out_dt_);

        const auto sum_injector = [&] {
            const float *p_sum_scale = &p.entry_[sum_idx].sum.scale;
            const int32_t *p_sum_zp = &p.entry_[sum_idx].sum.zero_point;
            if (*p_sum_scale != 1.f)
                mov(reg_ptr_sum_scale, (size_t)p_sum_scale);
            auto zmm_sum_zp = Xbyak::Zmm(30);
            if (*p_sum_zp != 0) {
                mov(reg_ptr_sum_zp, (size_t)p_sum_zp);
                vcvtdq2ps(zmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
            }

            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const auto zmm = vector(m, n, n_block);
                const auto addr = ptr[aux_reg_out
                        + out_typesize_ * (m * LDD_ + n * brg.ld_block)];

                const auto zmm_prev_dst = Xbyak::Zmm(31);
                cvt2ps(sum_dt, zmm_prev_dst, addr, true, false, k_mask);
                if (*p_sum_zp != 0) vsubps(zmm_prev_dst, zmm_sum_zp);
                if (*p_sum_scale == 1.f)
                    vaddps(zmm, zmm_prev_dst);
                else
                    vfmadd231ps(zmm, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
            }
        };

        if (jcp.with_sum && brg.beta != 0) {
            postops_injector_->set_lambda_injector(
                    primitive_kind::sum, sum_injector);
        }

        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

        if (with_binary_per_oc_bcast_) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const auto zmm_idx = vector(m, n, n_block).getIdx();
                const size_t aux_output_offset
                        = out_typesize_ * (m * LDD_ + n * brg.ld_block);

                rhs_arg_params.vmm_idx_to_out_reg.emplace(zmm_idx, aux_reg_out);
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        zmm_idx, aux_output_offset);
                if (tail) rhs_arg_params.vmm_tail_idx_.emplace(zmm_idx);
            }
        }

        postops_injector_->compute_vector_range(
                0, m_block * n_block, rhs_arg_params);
    }

    void apply_comp(int m_block, int n_block, int tail = 0) {
        auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;

        if (brg.alpha != 0 && brg.zp_type_a != brgemm_broadcast_t::none) {
            auto zmm_zp_a_val = Xbyak::Zmm(30);
            mov(reg_zp_a_val, ptr[rsp + reg_zp_a_val_offs_]);
            vpbroadcastd(zmm_zp_a_val, reg_zp_a_val.cvt32());

            mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
            for (int n = 0; n < n_block; n++) {
                auto zmm_zp_comp_a = Xbyak::Zmm(31);
                auto zp_comp_a_addr = EVEX_compress_addr(aux_reg_zp_a_comp,
                        sizeof(int32_t) * (n * brg.ld_block));
                zmm_zp_comp_a = zmm_mask(zmm_zp_comp_a, true, false, k_mask);
                vmovups(zmm_zp_comp_a, zp_comp_a_addr);
                vpmulld(zmm_zp_comp_a, zmm_zp_a_val, zp_comp_a_addr);

                for (int m = 0; m < m_block; m++) {
                    if (brg.with_comp_pads) {
                        auto zp_comp_a_vpad_offs = zp_comp_a_vpad_offset(n, m);
                        auto zp_comp_a_vpad_addr = zword[aux_reg_zp_a_comp
                                + zp_comp_a_vpad_offs];
                        vmovups(zmm_zp_comp_a, zp_comp_a_vpad_addr);
                        vpmulld(zmm_zp_comp_a, zmm_zp_comp_a, zmm_zp_a_val);
                    }
                    auto zmm = vector(m, n, n_block);
                    vpaddd(zmm, zmm, zmm_zp_comp_a);
                }
            }
        }

        if (brg.alpha != 0 && brg.req_s8s8_compensation) {
            mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
            for (int n = 0; n < n_block; n++) {
                auto zmm_comp = Xbyak::Zmm(31);
                auto comp_addr = EVEX_compress_addr(aux_reg_s8s8_comp,
                        sizeof(int32_t) * (n * brg.ld_block));
                zmm_comp = zmm_mask(zmm_comp, true, false, k_mask);
                vmovups(zmm_comp, comp_addr);

                for (int m = 0; m < m_block; m++) {
                    if (brg.with_comp_pads) {
                        auto comp_vpad_offs = compensation_vpad_offset(n, m);
                        auto comp_vpad_addr
                                = zword[aux_reg_s8s8_comp + comp_vpad_offs];
                        vmovups(zmm_comp, comp_vpad_addr);
                    }
                    auto zmm = vector(m, n, n_block);
                    vpaddd(zmm, zmm, zmm_comp);
                }
            }
        }
    }

    void maybe_apply_comp(int m_block, int n_block, int tail = 0) {
        Xbyak::Label label_apply_without_comp;
        mov(reg_apply_comp, ptr[rsp + reg_apply_comp_offs_]);
        cmp(reg_apply_comp, 0);
        je(label_apply_without_comp, T_NEAR);
        apply_comp(m_block, n_block, tail);
        L_aligned(label_apply_without_comp);

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            vcvtdq2ps(vector(m, n, n_block), vector(m, n, n_block));
        }
    }

    void apply_post_ops(int m_block, int n_block, int tail = 0) {
        const auto vector
                = [=](int m, int n) { return Xbyak::Zmm(m * n_block + n); };
        auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;
        const auto &p = attr.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const auto maybe_req_comp = brg.is_int8 && brg.alpha != 0
                && (brg.req_s8s8_compensation
                        || brg.zp_type_a != brgemm_broadcast_t::none);

        // brg.alpha == 0 means no read from input, no bias, no eltwise - just
        // initialize registers by zero at the beginning of kernel
        // brg.beta == 0 means no sum - just registers write to output
        // maybe_req_comp == true -> convert accumulated values to f32 after apply
        // compensation to avoid the lost of accuracy when converting s32 to f32
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            if (brg.alpha == 0) {
                if (sum_idx != -1 && brg.beta != 0) {
                    // if sum then have to init zmm each time
                    vpxord(vector(m, n), vector(m, n), vector(m, n));
                }
            } else {
                auto inp_addr = ptr[aux_reg_in
                        + inp_typesize_ * (m * brg.LDC + n * brg.ld_block)];
                if (maybe_req_comp) {
                    const Xbyak::Zmm zmm
                            = zmm_mask(vector(m, n), true, false, k_mask);
                    vmovups(zmm, inp_addr);
                } else {
                    cvt2ps(inp_dt_, vector(m, n), inp_addr, true, false,
                            k_mask);
                }
            }
        }

        if (maybe_req_comp) maybe_apply_comp(m_block, n_block, tail);

        if (brg.alpha != 0 && jcp.with_bias) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                auto zmm_bias = Xbyak::Zmm(31);
                auto bias_addr = ptr[aux_reg_bias
                        + bia_typesize_ * (n * brg.ld_block)];

                cvt2ps(bia_dt_, zmm_bias, bias_addr, true, false, k_mask);
                vaddps(vector(m, n), zmm_bias);
            }
        }

        if (brg.alpha != 0) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const Xbyak::Zmm zmm
                        = zmm_mask(vector(m, n), true, false, k_mask);
                vmulps(zmm, zmm,
                        ptr[aux_reg_scales
                                + is_oc_scale_ * sizeof(float)
                                        * (n * brg.ld_block)]);
            }
        }

        if (postops_injector_) inject_attr_postops(m_block, n_block, tail);

        if (brg.alpha != 0 && brg.zp_type_c != brgemm_broadcast_t::none) {
            mov(aux_reg_zp_c_values, ptr[rsp + aux_reg_zp_c_values_offs_]);
            auto zmm_zp_c = Xbyak::Zmm(31);
            if (brg.zp_type_c == brgemm_broadcast_t::per_tensor) {
                vcvtdq2ps(zmm_zp_c,
                        EVEX_compress_addr(aux_reg_zp_c_values, 0, true));
            }
            for (int n = 0; n < n_block; n++) {
                if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
                    int zp_c_off = zp_c_values_offset(n);
                    auto zp_c_addr
                            = EVEX_compress_addr(aux_reg_zp_c_values, zp_c_off);
                    cvt2ps(data_type::s32, zmm_zp_c, zp_c_addr, true, false,
                            k_mask);
                }
                for (int m = 0; m < m_block; m++)
                    vaddps(vector(m, n), zmm_zp_c);
            }
        }

        const bool dt_requires_saturation = utils::one_of(
                brg.dt_d, data_type::u8, data_type::s8, data_type::s32);

        const reg64_t reg_tmp_gpr = rax;
        auto zmm_lbound = Xbyak::Zmm(31);
        auto zmm_ubound = Xbyak::Zmm(30);
        if (dt_requires_saturation) {
            init_saturate_f32(zmm_lbound, zmm_ubound, reg_tmp_gpr,
                    data_type::f32, brg.dt_d);
        }

        if (brg.is_bf16_emu) bf16_emu_->init_vcvtneps2bf16();

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto zmm = vector(m, n);
            auto addr = ptr[aux_reg_out
                    + out_typesize_ * (m * LDD_ + n * brg.ld_block)];

            if (out_dt_ == data_type::bf16) {
                Xbyak::Ymm ymm = Xbyak::Ymm(zmm.getIdx());
                if (brg.alpha != 0 || (sum_idx != -1 && brg.beta != 0)) {
                    if (brg.is_bf16_emu)
                        bf16_emu_->vcvtneps2bf16(ymm, zmm);
                    else
                        vcvtneps2bf16(ymm, zmm);
                }
                const Xbyak::Ymm r_ymm = ymm_mask(ymm, true, true, k_mask);
                vmovdqu16(addr, r_ymm);
            } else {
                if (brg.alpha != 0 || (sum_idx != -1 && brg.beta != 0)) {
                    saturate_f32(zmm, zmm_lbound, zmm_ubound, brg.dt_d);
                    if (out_dt_ != data_type::f32) vcvtps2dq(zmm, zmm);
                }

                const Xbyak::Zmm r_zmm = zmm_mask(zmm, true, true, k_mask);
                switch (out_dt_) {
                    case data_type::f32:
                    case data_type::s32: vmovups(addr, r_zmm); break;
                    case data_type::s8: vpmovsdb(addr, r_zmm); break;
                    case data_type::u8: vpmovusdb(addr, r_zmm); break;
                    default: assert(!"unknown dst_dt");
                }
            }
        }
    }

    void loop_by_N(int m_block, int nb2, int nb2_tail, int nb_tail) {

        if (brg.alpha) {
            mov(aux_reg_in, reg_in);
            if (jcp.with_bias) mov(aux_reg_bias, reg_bias);
            if (with_binary_per_oc_bcast_)
                mov(aux_reg_oc_l_offset_, reg_oc_l_offset_);
            if (brg.zp_type_c != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_c_values, ptr[rsp + reg_zp_c_values_offs_]);
                mov(ptr[rsp + aux_reg_zp_c_values_offs_], aux_reg_zp_c_values);
            }
            if (brg.zp_type_a != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_a_comp, ptr[rsp + reg_zp_a_comp_offs_]);
                mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
            }
            if (brg.req_s8s8_compensation) {
                mov(aux_reg_s8s8_comp, ptr[rsp + reg_s8s8_comp_offs_]);
                mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
            }
            mov(aux_reg_scales, reg_scales);
        }
        mov(aux_reg_out, reg_out);

        for (int n_loop_ = 0; n_loop_ < nb2; n_loop_++) {
            apply_post_ops(m_block, n_block2_);

            const auto oc_l_offset = n_block2_ * brg.ld_block;

            add(aux_reg_out, out_typesize_ * oc_l_offset);
            if (brg.alpha != 0) {
                add(aux_reg_in, inp_typesize_ * oc_l_offset);

                if (jcp.with_bias)
                    add(aux_reg_bias, bia_typesize_ * oc_l_offset);
                if (brg.zp_type_c != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_c_values,
                            ptr[rsp + aux_reg_zp_c_values_offs_]);
                    add(aux_reg_zp_c_values, zp_c_values_offset(n_block2_));
                    mov(ptr[rsp + aux_reg_zp_c_values_offs_],
                            aux_reg_zp_c_values);
                }
                if (brg.zp_type_a != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                    add(aux_reg_zp_a_comp, sizeof(int32_t) * oc_l_offset);
                    mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
                }
                if (brg.req_s8s8_compensation) {
                    mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                    add(aux_reg_s8s8_comp, sizeof(int32_t) * oc_l_offset);
                    mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
                }
                if (with_binary_per_oc_bcast_) {
                    mov(aux_reg_oc_l_offset_,
                            ptr[rsp + reg_aux_oc_l_offset_offs_]);
                    add(aux_reg_oc_l_offset_, oc_l_offset);
                    mov(ptr[rsp + reg_aux_oc_l_offset_offs_],
                            aux_reg_oc_l_offset_);
                }

                add(aux_reg_scales, is_oc_scale_ * sizeof(float) * oc_l_offset);
            }
        }
        if (nb2_tail > 0) {
            apply_post_ops(m_block, nb2_tail);
            const auto oc_l_offset = nb2_tail * brg.ld_block;

            add(aux_reg_out, out_typesize_ * oc_l_offset);
            if (brg.alpha != 0) {
                add(aux_reg_in, inp_typesize_ * oc_l_offset);
                if (jcp.with_bias)
                    add(aux_reg_bias, bia_typesize_ * oc_l_offset);
                if (brg.zp_type_c != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_c_values,
                            ptr[rsp + aux_reg_zp_c_values_offs_]);
                    add(aux_reg_zp_c_values, zp_c_values_offset(nb2_tail));
                    mov(ptr[rsp + aux_reg_zp_c_values_offs_],
                            aux_reg_zp_c_values);
                }
                if (brg.zp_type_a != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                    add(aux_reg_zp_a_comp, sizeof(int32_t) * oc_l_offset);
                    mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
                }
                if (brg.req_s8s8_compensation) {
                    mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                    add(aux_reg_s8s8_comp, sizeof(int32_t) * oc_l_offset);
                    mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
                }
                if (with_binary_per_oc_bcast_) {
                    mov(aux_reg_oc_l_offset_,
                            ptr[rsp + reg_aux_oc_l_offset_offs_]);
                    add(aux_reg_oc_l_offset_, oc_l_offset);
                    mov(ptr[rsp + reg_aux_oc_l_offset_offs_],
                            aux_reg_oc_l_offset_);
                }

                add(aux_reg_scales, is_oc_scale_ * sizeof(float) * oc_l_offset);
            }
        }
        if (nb_tail > 0) {
            apply_post_ops(m_block, 1, nb_tail);

            if (brg.alpha != 0) {
                add(aux_reg_in, inp_typesize_ * (nb_tail));
                if (jcp.with_bias) add(aux_reg_bias, bia_typesize_ * (nb_tail));
                if (brg.zp_type_c != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_c_values,
                            ptr[rsp + aux_reg_zp_c_values_offs_]);
                    add(aux_reg_zp_c_values, zp_c_values_offset(1, nb_tail));
                    mov(ptr[rsp + aux_reg_zp_c_values_offs_],
                            aux_reg_zp_c_values);
                }
                if (brg.zp_type_a != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                    add(aux_reg_zp_a_comp, sizeof(int32_t) * nb_tail);
                    mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
                }
                if (brg.req_s8s8_compensation) {
                    mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                    add(aux_reg_s8s8_comp, sizeof(int32_t) * nb_tail);
                    mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
                }
                if (with_binary_per_oc_bcast_) {
                    mov(aux_reg_oc_l_offset_,
                            ptr[rsp + reg_aux_oc_l_offset_offs_]);
                    add(aux_reg_oc_l_offset_, nb_tail);
                    mov(ptr[rsp + reg_aux_oc_l_offset_offs_],
                            aux_reg_oc_l_offset_);
                }
                add(aux_reg_scales, is_oc_scale_ * bia_typesize_ * (nb_tail));
            }
            add(aux_reg_out, out_typesize_ * (nb_tail));
        }
    }

    void generate() override {
        preamble();

        sub(rsp, stack_space_needed_);

        int nb = brg.load_dim / brg.ld_block;
        int nb_tail = brg.load_dim % brg.ld_block;

        int nb2 = nb / n_block2_;
        int nb2_tail = nb % n_block2_;
        int n_block = (nb2 == 0) ? nstl::max(1, nb2_tail) : n_block2_;

        int m_max_regs = (brg.is_bf16_emu ? 24 : 28) / n_block;
        int m_block = nstl::min(brg.bcast_dim, m_max_regs);

        int mb = brg.bcast_dim / m_block;
        int mb_tail = brg.bcast_dim % m_block;

        const auto full_mask = size_t {0xffffffffffffffff};
        const auto tail_mask = size_t((1 << nb_tail) - 1);

        reg64_t reg_mask = rax;

        mov(reg_mask, full_mask);
        kmovq(k_full_mask, reg_mask);
        mov(reg_mask, tail_mask);
        kmovq(k_tail_mask, reg_mask);

        if (brg.alpha != 0) {
            mov(reg_in, ptr[param1 + GET_OFF(ptr_in)]);
            mov(reg_scales, ptr[param1 + GET_OFF(ptr_scales)]);
            mov(reg_apply_comp, ptr[param1 + GET_OFF(apply_comp)]);
            mov(ptr[rsp + reg_apply_comp_offs_], reg_apply_comp);

            if (jcp.with_bias) mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
            if (brg.zp_type_c != brgemm_broadcast_t::none) {
                mov(reg_zp_c_values, ptr[param1 + GET_OFF(c_zp_values)]);
                mov(ptr[rsp + reg_zp_c_values_offs_], reg_zp_c_values);
            }
            if (brg.zp_type_a != brgemm_broadcast_t::none) {
                mov(reg_zp_a_comp, ptr[param1 + GET_OFF(a_zp_compensation)]);
                mov(ptr[rsp + reg_zp_a_comp_offs_], reg_zp_a_comp);

                mov(reg_zp_a_val, ptr[param1 + GET_OFF(a_comp_val)]);
                mov(ptr[rsp + reg_zp_a_val_offs_], reg_zp_a_val);
            }
            if (brg.req_s8s8_compensation) {
                mov(reg_s8s8_comp, ptr[param1 + GET_OFF(s8s8_compensation)]);
                mov(ptr[rsp + reg_s8s8_comp_offs_], reg_s8s8_comp);
            }
            if (with_binary_per_oc_bcast_)
                mov(reg_oc_l_offset_, ptr[param1 + GET_OFF(oc_l_offset)]);
        }
        mov(reg_out, ptr[param1 + GET_OFF(ptr_out)]);

        // brg.alpha == 0 means no read from input, no bias, no eltwise - just
        // initialize registers by zero
        // brg.beta == 0 means no sum - just registers write to output
        if (brg.alpha == 0) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                auto zmm = Xbyak::Zmm(m * n_block + n);
                vpxord(zmm, zmm, zmm);
            }
        }

        for (int mb_ = 0; mb_ < mb; mb_++) {
            loop_by_N(m_block, nb2, nb2_tail, nb_tail);

            if (brg.alpha != 0)
                add(reg_in, inp_typesize_ * (m_block * brg.LDC));
            advance_mb_post_ops_regs(m_block);
            add(reg_out, out_typesize_ * (m_block * LDD_));
        }
        if (mb_tail > 0) loop_by_N(mb_tail, nb2, nb2_tail, nb_tail);

        add(rsp, stack_space_needed_);

        postamble();

        if (brg.alpha != 0 && jcp.with_eltwise)
            postops_injector_->prepare_table();
    }
};

#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
