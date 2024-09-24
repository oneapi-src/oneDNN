/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
* Copyright 2024 Arm Ltd. and affiliates
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
#ifndef CPU_AARCH64_JIT_BRGEMM_POST_OPS_HPP
#define CPU_AARCH64_JIT_BRGEMM_POST_OPS_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_brgemm_primitive_conf.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/cpu_engine.hpp"

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct brgemm_kernel_diff_bias_t {
    brgemm_kernel_diff_bias_t()
        : ptr_diff_dst(nullptr)
        , ptr_diff_bias_acc(nullptr)
        , ptr_diff_bias(nullptr)
        , flags(0) {};

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
        , bia_typesize_(types::data_type_size(bia_dt_))
        , acc_typesize_(types::data_type_size(acc_dt_)) {

        ddst_dt_ = ajbgp.dst_dt;
        ddst_typesize_ = types::data_type_size(ddst_dt_);
        mult_ = data_type_vnni_granularity(ddst_dt_);
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_diff_bias_t)

private:
    brgemm_t brg_;
    data_type_t ddst_dt_;
    data_type_t bia_dt_;
    data_type_t acc_dt_;

    int ddst_typesize_;
    int bia_typesize_;
    int acc_typesize_;
    int mult_;

    // Register decomposition
    const XReg param1 = x7; //abi_param1
    const XReg reg_ddst = x15;
    const XReg reg_bias = x14;
    const XReg reg_bias_acc = x13;
    const XReg aux_reg_ddst = x12;
    const XReg reg_k_iter = x11;
    const XReg reg_flag = x10;

    PReg k_full_mask = PReg(2);
    PReg k_tail_mask = PReg(3);
    ZReg vreg_unit = ZReg(31);
    ZReg vreg_perm = ZReg(30);

    const int n_max_regs_ = 4;

    ZReg get_bias_reg(int n) const { return ZReg(n); }
    ZReg get_bias_reg_lower(int n) const { return ZReg(n); }
    ZReg get_ddst_reg(int n) const { return ZReg(n + n_max_regs_); }

    void accumulate_bias(int idx, bool mask_flag) {
        auto vddst = get_ddst_reg(idx);
        const auto k_mask = mask_flag ? k_tail_mask : k_full_mask;
        auto vbias = get_bias_reg(idx);
        if (ddst_dt_ == data_type::f16) {
            assert(!"unsupported\n");
        } else {
            auto addr = ptr(
                    aux_reg_ddst, ddst_typesize_ * mult_ * idx * brg_.ld_block);
            ld1w(vddst.s, k_mask / T_z, addr);
            if (ddst_dt_ == data_type::bf16)
                assert(!"unsupported\n");
            else
                fadd(vbias.s, vbias.s, vddst.s);
        }
    }

    void store(int idx, bool mask_flag) {
        auto addr = ptr(reg_bias, bia_typesize_ * idx * brg_.ld_block);
        const auto k_mask = mask_flag ? k_tail_mask : k_full_mask;
        switch (bia_dt_) {
            case data_type::bf16: assert(!"unsupported\n"); break;
            case data_type::f16: assert(!"unsupported\n"); break;
            case data_type::f32: st1w(get_bias_reg(idx).s, k_mask, addr); break;
            default: assert("Unsupported bias data type");
        }
    }

    void loop_by_N(int n_loop, int nb_tail) {

        mov(aux_reg_ddst, reg_ddst);

        int n_iters = n_loop;
        if (nb_tail > 0) n_iters--;
        Label k_loop, init_zero, init_done;
        int n_ = 0;

        tst(reg_flag, FLAG_REDUCE_FIRST);
        b(NE, init_zero); // FLAG_REDUCE_FIRST is set

        for (; n_ < n_iters; n_++) {
            ldr(get_bias_reg(n_),
                    ptr(reg_bias_acc, acc_typesize_ * n_ * brg_.ld_block));
        }
        if (nb_tail > 0) {
            ld1w(get_bias_reg(n_).s, k_tail_mask / T_z,
                    ptr(reg_bias_acc, acc_typesize_ * n_ * brg_.ld_block));
        }
        b(init_done);
        L(init_zero);

        for (int n_ = 0; n_ < n_loop; n_++) {
            eor(get_bias_reg(n_).d, get_bias_reg(n_).d, get_bias_reg(n_).d);
        }
        L(init_done);

        mov(reg_k_iter, utils::div_up(brg_.reduce_dim, mult_));
        L(k_loop);
        {
            int n_ = 0;
            for (; n_ < n_iters; n_++)
                accumulate_bias(n_, false);

            if (nb_tail > 0) accumulate_bias(n_, true);

            add_imm(aux_reg_ddst, aux_reg_ddst,
                    ddst_typesize_ * mult_ * brg_.LDB, X_TMP_0);

            sub(reg_k_iter, reg_k_iter, 1);
            b(NE, k_loop);
        }

        Label store_final, store_done;
        tst(reg_flag, FLAG_REDUCE_LAST);
        b(NE, store_final); // FLAG_REDUCE_LAST is set

        n_ = 0;
        for (; n_ < n_iters; n_++) {
            str(get_bias_reg(n_),
                    ptr(reg_bias_acc, acc_typesize_ * n_ * brg_.ld_block));
        }
        if (nb_tail > 0) {
            st1w(get_bias_reg(n_).s, k_tail_mask,
                    ptr(reg_bias_acc, acc_typesize_ * n_ * brg_.ld_block));
        }
        b(store_done);

        L(store_final);
        n_ = 0;

        for (; n_ < n_iters; n_++)
            store(n_, false);

        if (nb_tail > 0) store(n_, true);

        L(store_done);
    }

    void generate() override {
        size_t simd_w_ = 0;
        switch (brg_.isa_impl) {
            case sve_512:
                simd_w_ = cpu_isa_traits<sve_512>::vlen / sizeof(float);
                break;
            case sve_256:
                simd_w_ = cpu_isa_traits<sve_256>::vlen / sizeof(float);
                break;
            default: {
                assert(!"unsupported isa");
                return;
            }
        }
        preamble();
        if (simd_w_ != cpu_sveLen / sizeof(float)) {
            set_preg(P_ALL_ONE.b, simd_w_ * 4, X_TMP_0, X_TMP_1);
            set_preg(k_full_mask.b, simd_w_ * 4, X_TMP_0, X_TMP_1);
        } else
            ptrue(k_full_mask.b);

        int nb = utils::div_up(brg_.load_dim, brg_.ld_block);
        int nb_tail = brg_.load_dim % brg_.ld_block;

        int n_loop = nb / n_max_regs_;
        int n_loop_tail = nb % n_max_regs_;
        if (n_loop_tail == 0 && nb_tail > 0) {
            n_loop--;
            n_loop_tail = n_max_regs_;
        }

        set_preg(k_tail_mask.s, nb_tail, X_TMP_0, X_TMP_1);
        pfalse(P_TMP_0.b);
        zip1(k_tail_mask.b, k_tail_mask.b, P_TMP_0.b);
        zip1(k_tail_mask.h, k_tail_mask.h, P_TMP_0.h);

        if (ddst_dt_ == data_type::bf16) { assert(!"unsupported data type\n"); }

        if (ddst_dt_ == data_type::f16) { assert(!"unsupported data type\n"); }

        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_diff_dst), X_TMP_0);
        ldr(reg_ddst, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_diff_bias_acc), X_TMP_0);
        ldr(reg_bias_acc, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_diff_bias), X_TMP_0);
        ldr(reg_bias, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(flags), X_TMP_0);
        ldr(reg_flag, ptr(X_DEFAULT_ADDR));

        for (int nb_ = 0; nb_ < n_loop; nb_++) {
            loop_by_N(n_max_regs_, 0);

            add_imm(reg_ddst, reg_ddst,
                    ddst_typesize_ * mult_ * n_max_regs_ * brg_.ld_block,
                    X_TMP_0);
            add_imm(reg_bias, reg_bias,
                    bia_typesize_ * n_max_regs_ * brg_.ld_block, X_TMP_0);
            add_imm(reg_bias_acc, reg_bias_acc,
                    acc_typesize_ * n_max_regs_ * brg_.ld_block, X_TMP_0);
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
    size_t apply_comp = 0;
    int32_t a_comp_val = 1;
    int32_t *a_zp_compensation;
    int32_t *c_zp_values;
    int32_t *s8s8_compensation;
    const void *dst_orig;
    void *ptr_dst_scales;
};

template <cpu_isa_t isa>
struct jit_brgemm_kernel_post_ops : public jit_generator {

    jit_brgemm_kernel_post_ops(const jit_brgemm_conv_conf_t &ajcp,
            const brgemm_t &abrg, const primitive_attr_t &aattr)
        : brg(abrg)
        , jcp(ajcp)
        , attr(aattr)
        , postops_injector_(nullptr)
        , with_binary_non_scalar_bcast_(brg.with_binary
                  && binary_injector::
                          any_binary_postop_rhs_non_scalar_broadcast(
                                  brg.attr->post_ops_,
                                  memory_desc_wrapper(brg.dst_md))) {

        if (brg.beta != 0) {
            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    static_cast<size_t>(vmm_tmp(4).getIdx()), XReg(14),
                    XReg(15), XReg(13), preserve_gpr, preserve_vmm,
                    GET_OFF(ptr_binary_post_ops_rhs), GET_OFF(dst_orig),
                    memory_desc_wrapper(brg.dst_md),
                    static_cast<size_t>(brg.load_dim % brg.ld_block),
                    k_tail_mask, use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {this->param1, rhs_sp};

            const bool save_state = jcp.with_eltwise;
            const auto &reserved_eltwise_gpr = reg_reserved_eltwise;
            const auto reserved_eltwise_maskr = PReg(1);

            const eltwise_injector::static_params_t esp {
                    save_state, reserved_eltwise_gpr, reserved_eltwise_maskr};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<isa>>(
                    this, attr.post_ops_, bsp, esp);
        }

        const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
        // per_oc: conv: 1 << 0, (1 << 1) + (1 << 0) (with groups)
        // per_oc: ip: 1 << 0
        is_oc_scale_
                = utils::one_of(wei_scales.mask_, 1 << 0, (1 << 1) + (1 << 0));

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
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;

    const bool with_binary_non_scalar_bcast_;

    int inp_typesize_;
    int out_typesize_;
    int bia_typesize_;

    int is_oc_scale_;
    constexpr static int max_vregs_ = 32;

    // Register decomposition
    const XReg reg_reserved_eltwise = x0;
    ;
    const XReg param1 = x7; //abi_param1=rdi
    const XReg reg_in = x15;
    const XReg reg_out = x14;
    const XReg aux_reg_in = x13;
    const XReg aux_reg_out = x12;

    const XReg reg_bias = x11;
    const XReg aux_reg_bias = x10;

    const XReg reg_scales = x9;
    const XReg aux_reg_scales = x8;

    const XReg reg_ptr_sum_scale = x2;
    ;
    const XReg reg_ptr_sum_zp = x6;

    const XReg reg_zp_c_values = x3;
    const XReg aux_reg_zp_c_values = x3;
    const XReg reg_zp_a_comp = x3;
    const XReg aux_reg_zp_a_comp = x3;
    const XReg reg_s8s8_comp = x3;
    const XReg aux_reg_s8s8_comp = x3;
    const XReg reg_zp_a_val = x3;
    const XReg reg_apply_comp = x3;
    const XReg reg_dst_scales = x3;
    const XReg aux_reg_dst_scales = x3;
    const XReg reg_tmp = abi_not_param1;

    constexpr static int reg_zp_c_values_offs_ = 0;
    constexpr static int aux_reg_zp_c_values_offs_ = 8;
    constexpr static int reg_zp_a_comp_offs_ = 16;
    constexpr static int aux_reg_zp_a_comp_offs_ = 24;
    constexpr static int reg_s8s8_comp_offs_ = 32;
    constexpr static int aux_reg_s8s8_comp_offs_ = 40;
    constexpr static int reg_zp_a_val_offs_ = 48;
    constexpr static int reg_apply_comp_offs_ = 56;
    constexpr static int reg_dst_scales_offs_ = 64;
    constexpr static int stack_space_needed_ = 72;

    PReg k_full_mask = p2;
    PReg k_tail_mask = p3;

    const int n_block2_ = 4;

    ZReg vmm_tmp(int i) const { return ZReg(max_vregs_ - 1 - i); }

    int zp_c_values_offset(int n, bool is_tail = false) const noexcept {
        if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
            return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                             : sizeof(int32_t) * n * brg.ld_block;
        }

        return 0;
    }
    int zp_comp_a_vpad_offset(
            int n, int m, bool is_tail = false) const noexcept {
        return (is_tail) ? sizeof(int32_t) * (brg.ldb_tail + m * brg.LDB)
                         : sizeof(int32_t) * (n * brg.ld_block + m * brg.LDB);
    }
    int mb_zp_comp_a_offset(int m_block) const noexcept {
        return sizeof(int32_t) * m_block * brg.LDB;
    }
    int compensation_vpad_offset(
            int n, int m, bool is_tail = false) const noexcept {
        return (is_tail) ? sizeof(int32_t) * (brg.ldb_tail + m * brg.LDB)
                         : sizeof(int32_t) * (n * brg.ld_block + m * brg.LDB);
    }
    int mb_compensation_offset(int m_block) const noexcept {
        return sizeof(int32_t) * m_block * brg.LDB;
    }

    void cvt2ps(data_type_t type_in, const ZReg zmm_in, const AdrNoOfs &op,
            bool mask_flag, bool store, PReg ktail_mask) {
        switch (type_in) {
            case data_type::f32:
            case data_type::s32:
                if (mask_flag) {
                    if (store) {
                        st1w(zmm_in.s, ktail_mask / T_m, op);
                    } else {
                        ld1w(zmm_in.s, ktail_mask / T_z, op);
                    }
                } else {
                    ld1w(zmm_in.s, k_full_mask / T_z, op);
                }
                break;
            case data_type::s8: assert(!"unsupported data type\n"); break;
            case data_type::u8: assert(!"unsupported data type\n"); break;
            case data_type::bf16: assert(!"unsupported data type\n"); break;
            default: assert(!"unsupported data type");
        }
        if (!utils::one_of(type_in, data_type::f32, data_type::bf16))
            assert(!"unsupported data type\n");
    }

    ZReg vector(int m, int n, int n_block) { return ZReg(m * n_block + n); };

    void inject_attr_postops(int m_block, int n_block, int tail = 0) {
        const auto &p = attr.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const auto k_mask = tail == 0 ? k_full_mask : k_tail_mask;
        const auto sum_dt = p.get_sum_dt(out_dt_);

        const auto sum_injector = [&] {
            const float *p_sum_scale = &p.entry_[sum_idx].sum.scale;
            const int32_t *p_sum_zp = &p.entry_[sum_idx].sum.zero_point;
            if (*p_sum_scale != 1.f)
                mov_imm(reg_ptr_sum_scale, (size_t)p_sum_scale);
            auto vmm_sum_zp = vmm_tmp(1);
            if (*p_sum_zp != 0) {
                mov_imm(reg_ptr_sum_zp, (size_t)p_sum_zp);
                ld1rw(vmm_sum_zp.s, k_full_mask / T_z, ptr(reg_ptr_sum_zp));
                scvtf(vmm_sum_zp.s, k_full_mask / T_m, vmm_sum_zp.s);
            }

            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const auto vmm = vector(m, n, n_block);

                const auto vmm_prev_dst = vmm_tmp(0);
                const auto vmm_tmp2 = vmm_tmp(2);
                add_imm(X_DEFAULT_ADDR, aux_reg_out,
                        out_typesize_ * (m * LDD_ + n * brg.ld_block), X_TMP_0);
                cvt2ps(sum_dt, vmm_prev_dst, ptr(X_DEFAULT_ADDR), tail, false,
                        k_mask);
                if (*p_sum_zp != 0)
                    fsub(vmm_prev_dst.s, vmm_prev_dst.s, vmm_sum_zp.s);
                if (*p_sum_scale == 1.f)
                    fadd(vmm.s, vmm.s, vmm_prev_dst.s);
                else {
                    ld1rw(vmm_tmp2.s, k_full_mask / T_z,
                            ptr(reg_ptr_sum_scale));
                    fmla(vmm.s, k_full_mask / T_m, vmm_prev_dst.s, vmm_tmp2.s);
                }
            }
        };

        if (jcp.with_sum) {
            postops_injector_->set_lambda_injector(
                    primitive_kind::sum, sum_injector);
        }

        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

        if (with_binary_non_scalar_bcast_) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const auto vmm_idx = vector(m, n, n_block).getIdx();
                const size_t aux_output_offset
                        = out_typesize_ * (m * LDD_ + n * brg.ld_block);

                rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, aux_reg_out);
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        vmm_idx, aux_output_offset);
                if (tail) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
            }
        }

        postops_injector_->compute_vector_range(
                0, m_block * n_block, rhs_arg_params);
    }
    void apply_comp(int m_block, int n_block, int tail = 0) {
        auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;

        if (brg.zp_type_a != brgemm_broadcast_t::none) {
            auto vmm_zp_a_val = vmm_tmp(1);
            ldr(reg_zp_a_val, ptr(X_SP, reg_zp_a_val_offs_));
            dup(vmm_zp_a_val.s, WReg(reg_zp_a_val.getIdx()));

            ldr(aux_reg_zp_a_comp, ptr(X_SP, aux_reg_zp_a_comp_offs_));
            for (int n = 0; n < n_block; n++) {
                auto vmm_zp_comp_a = vmm_tmp(0);
                const size_t zp_comp_offset
                        = sizeof(int32_t) * (n * brg.ld_block);
                add_imm(X_DEFAULT_ADDR, aux_reg_zp_a_comp, zp_comp_offset,
                        X_TMP_0);
                ldr(vmm_zp_comp_a, ptr(X_DEFAULT_ADDR));
                mul(vmm_zp_comp_a.s, k_mask / T_m, vmm_zp_a_val.s);
                for (int m = 0; m < m_block; m++) {
                    auto vmm = vector(m, n, n_block);
                    add(vmm.s, vmm.s, vmm_zp_comp_a.s);
                }
            }
        }

        if (brg.req_s8s8_compensation) {
            ldr(aux_reg_s8s8_comp, ptr(X_SP, aux_reg_s8s8_comp_offs_));
            for (int n = 0; n < n_block; n++) {
                auto vmm_comp = vmm_tmp(0);
                const size_t s8s8_comp_offset
                        = sizeof(int32_t) * (n * brg.ld_block);
                add_imm(X_DEFAULT_ADDR, aux_reg_s8s8_comp, s8s8_comp_offset,
                        X_TMP_0);
                ld1w(vmm_comp.s, k_mask / T_z, ptr(X_DEFAULT_ADDR));
                for (int m = 0; m < m_block; m++) {
                    auto vmm = vector(m, n, n_block);
                    add(vmm.s, vmm.s, vmm_comp.s);
                }
            }
        }
    }
    void maybe_apply_comp(int m_block, int n_block, int tail = 0) {
        Label label_apply_without_comp;
        ldr(reg_apply_comp, ptr(X_SP, reg_apply_comp_offs_));
        cmp(reg_apply_comp, 0);
        b(EQ, label_apply_without_comp);
        apply_comp(m_block, n_block, tail);
        L_aligned(label_apply_without_comp);

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            scvtf(vector(m, n, n_block).s, k_full_mask / T_m,
                    vector(m, n, n_block).s);
        }
    }

    void apply_post_ops(int m_block, int n_block, int tail = 0) {
        const auto vector = [=](int m, int n) { return ZReg(m * n_block + n); };
        auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;
        const auto req_comp = brg.is_int8 && brg.beta != 0
                && (brg.req_s8s8_compensation
                        || brg.zp_type_a != brgemm_broadcast_t::none);

        // brg.alpha == 0 means initialize registers, 1 means read from input
        // brg.beta == 0 means skip postwork, 1 means do postwork
        // req_comp == true -> convert accumulated values to f32 after applying
        // compensation to avoid the loss of accuracy when converting s32 to f32
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            if (brg.alpha == 0 && brg.beta != 0) {
                // if postwork then have to init zmm each time
                eor(vector(m, n).d, vector(m, n).d, vector(m, n).d);
            } else if (brg.alpha != 0) {
                add_imm(X_DEFAULT_ADDR, aux_reg_in,
                        inp_typesize_ * (m * brg.LDC + n * brg.ld_block),
                        X_TMP_0);
                cvt2ps(inp_dt_, vector(m, n), ptr(X_DEFAULT_ADDR), true, false,
                        k_mask);
            }
        }

        if (req_comp) maybe_apply_comp(m_block, n_block, tail);

        if (brg.beta != 0) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                auto vmm = vector(m, n);
                auto vmm_tmp0 = vmm_tmp(0);
                const auto k_mask = tail > 0 ? k_tail_mask : k_full_mask;
                add_imm(X_DEFAULT_ADDR, aux_reg_scales,
                        is_oc_scale_ * sizeof(float) * (n * brg.ld_block),
                        X_TMP_0);
                ldr(vmm_tmp0, ptr(X_DEFAULT_ADDR));
                fmul(vmm.s, vmm.s, vmm_tmp0.s);
                if (tail > 0) mov(vmm.s, k_mask / T_m, vmm.s);
            }
        }

        if (brg.beta != 0 && jcp.with_bias) {
            for (int n = 0; n < n_block; n++) {
                auto vmm_bias = vmm_tmp(0);
                add_imm(X_DEFAULT_ADDR, aux_reg_bias,
                        bia_typesize_ * (n * brg.ld_block), X_TMP_0);
                cvt2ps(bia_dt_, vmm_bias, ptr(X_DEFAULT_ADDR), true, false,
                        k_mask);
                for (int m = 0; m < m_block; m++) {
                    fadd(vector(m, n).s, vector(m, n).s, vmm_bias.s);
                }
            }
        }

        if (postops_injector_) inject_attr_postops(m_block, n_block, tail);

        if (brg.beta != 0 && brg.with_dst_scales) {
            ldr(aux_reg_dst_scales, ptr(X_SP, reg_dst_scales_offs_));
            auto vmm_tmp1 = vmm_tmp(1);
            ldr(vmm_tmp1, ptr(aux_reg_dst_scales));

            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                auto vmm = vector(m, n);
                fmul(vmm.s, vmm.s, vmm_tmp1.s);
                if (tail > 0) mov(vmm.s, k_mask / T_m, vmm.s);
            }
        }

        if (brg.beta != 0 && brg.zp_type_c != brgemm_broadcast_t::none) {
            ldr(aux_reg_zp_c_values, ptr(X_SP, aux_reg_zp_c_values_offs_));
            auto vmm_zp_c = vmm_tmp(0);
            if (brg.zp_type_c == brgemm_broadcast_t::per_tensor) {
                auto vmm_tmp1 = vmm_tmp(1);
                ld1rw(vmm_tmp1.s, k_full_mask / T_z,
                        ptr(aux_reg_zp_c_values, 0));
                scvtf(vmm_zp_c.s, k_full_mask / T_m, vmm_tmp1.s);
            }
            for (int n = 0; n < n_block; n++) {
                if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
                    int zp_c_off = zp_c_values_offset(n);
                    add_imm(X_DEFAULT_ADDR, aux_reg_zp_c_values, zp_c_off,
                            X_TMP_0);
                    cvt2ps(data_type::s32, vmm_zp_c, ptr(X_DEFAULT_ADDR), tail,
                            false, k_mask);
                }
                for (int m = 0; m < m_block; m++) {
                    const auto vmm = vector(m, n);
                    fadd(vmm.s, vmm.s, vmm_zp_c.s);
                }
            }
        }

        const bool dt_requires_saturation = types::is_integral_dt(out_dt_);

        const XReg reg_tmp_gpr = reg_tmp;
        auto vmm_lbound = vmm_tmp(0);
        auto vmm_ubound = vmm_tmp(1);
        if (dt_requires_saturation) {
            init_saturate_f32(vmm_lbound, vmm_ubound, reg_tmp_gpr,
                    data_type::f32, out_dt_);
        }

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            // incase of tail, stores are unconditionally masked, regardless
            // of `n`, implying n_block must be equal to `1`.
            assert(IMPLICATION(tail > 0, n_block == 1));
            auto vmm = vector(m, n);

            if (dt_requires_saturation) {
                saturate_f32(vmm, vmm_lbound, vmm_ubound, out_dt_, k_full_mask);
                frintn(vmm.s, k_full_mask, vmm.s);
                fcvtzs(vmm.s, k_full_mask, vmm.s);
            }

            switch (out_dt_) {
                case data_type::f32:
                case data_type::s32:
                    add_imm(X_DEFAULT_ADDR, aux_reg_out,
                            out_typesize_ * (m * LDD_ + n * brg.ld_block),
                            X_TMP_0); //addr
                    st1w(vmm.s, k_mask / T_m, ptr(X_DEFAULT_ADDR));
                    break;
                case data_type::s8: assert(!"unsupported data type\n"); break;
                case data_type::u8: assert(!"unsupported data type\n"); break;
                default: assert(!"unknown dst_dt");
            }
        }
    }

    void loop_by_N(int m_block, int nb2, int nb2_tail, int nb_tail) {

        if (brg.alpha) { mov(aux_reg_in, reg_in); }
        if (brg.beta != 0) {
            if (jcp.with_bias) mov(aux_reg_bias, reg_bias);
            if (brg.zp_type_c != brgemm_broadcast_t::none) {
                ldr(aux_reg_zp_c_values, ptr(X_SP, reg_zp_c_values_offs_));
                str(aux_reg_zp_c_values, ptr(X_SP, aux_reg_zp_c_values_offs_));
            }
            if (brg.zp_type_a != brgemm_broadcast_t::none) {
                ldr(aux_reg_zp_a_comp, ptr(X_SP, reg_zp_a_comp_offs_));
                str(aux_reg_zp_a_comp, ptr(X_SP, aux_reg_zp_a_comp_offs_));
            }
            if (brg.req_s8s8_compensation) {
                ldr(aux_reg_s8s8_comp, ptr(X_SP, reg_s8s8_comp_offs_));
                str(aux_reg_s8s8_comp, ptr(X_SP, aux_reg_s8s8_comp_offs_));
            }
            mov(aux_reg_scales, reg_scales);
        }
        mov(aux_reg_out, reg_out);

        for (int n_loop_ = 0; n_loop_ < nb2; n_loop_++) {
            apply_post_ops(m_block, n_block2_);

            const auto oc_l_offset = n_block2_ * brg.ld_block;

            add_imm(aux_reg_out, aux_reg_out, out_typesize_ * oc_l_offset,
                    X_TMP_0);
            if (brg.alpha != 0) {
                add_imm(aux_reg_in, aux_reg_in, inp_typesize_ * oc_l_offset,
                        X_TMP_0);
            }
            if (brg.beta != 0) {
                if (jcp.with_bias)
                    add_imm(aux_reg_bias, aux_reg_bias,
                            bia_typesize_ * oc_l_offset, X_TMP_0);
                if (brg.zp_type_c != brgemm_broadcast_t::none) {
                    ldr(aux_reg_zp_c_values,
                            ptr(X_SP, aux_reg_zp_c_values_offs_));
                    add_imm(aux_reg_zp_c_values, aux_reg_zp_c_values,
                            zp_c_values_offset(n_block2_), X_TMP_0);
                    str(aux_reg_zp_c_values,
                            ptr(X_SP, aux_reg_zp_c_values_offs_));
                }
                if (brg.zp_type_a != brgemm_broadcast_t::none) {
                    ldr(aux_reg_zp_a_comp, ptr(X_SP, aux_reg_zp_a_comp_offs_));
                    add_imm(aux_reg_zp_a_comp, aux_reg_zp_a_comp,
                            sizeof(int32_t) * oc_l_offset, X_TMP_0);
                    str(aux_reg_zp_a_comp, ptr(X_SP, aux_reg_zp_a_comp_offs_));
                }
                if (brg.req_s8s8_compensation) {
                    ldr(aux_reg_s8s8_comp, ptr(X_SP, aux_reg_s8s8_comp_offs_));
                    add_imm(aux_reg_s8s8_comp, aux_reg_s8s8_comp,
                            sizeof(int32_t) * oc_l_offset, X_TMP_0);
                    str(aux_reg_s8s8_comp, ptr(X_SP, aux_reg_s8s8_comp_offs_));
                }

                add_imm(aux_reg_scales, aux_reg_scales,
                        is_oc_scale_ * sizeof(float) * oc_l_offset, X_TMP_0);
            }
        }
        if (nb2_tail > 0) {
            apply_post_ops(m_block, nb2_tail);
            const auto oc_l_offset = nb2_tail * brg.ld_block;

            add_imm(aux_reg_out, aux_reg_out, out_typesize_ * oc_l_offset,
                    X_TMP_0);
            if (brg.alpha != 0) {
                add_imm(aux_reg_in, aux_reg_in, inp_typesize_ * oc_l_offset,
                        X_TMP_0);
            }
            if (brg.beta != 0) {
                if (jcp.with_bias)
                    add_imm(aux_reg_bias, aux_reg_bias,
                            bia_typesize_ * oc_l_offset, X_TMP_0);
                if (brg.zp_type_c != brgemm_broadcast_t::none) {
                    ldr(aux_reg_zp_c_values,
                            ptr(X_SP, aux_reg_zp_c_values_offs_));
                    add_imm(aux_reg_zp_c_values, aux_reg_zp_c_values,
                            zp_c_values_offset(nb2_tail), X_TMP_0);
                    str(aux_reg_zp_c_values,
                            ptr(X_SP, aux_reg_zp_c_values_offs_));
                }
                if (brg.zp_type_a != brgemm_broadcast_t::none) {
                    ldr(aux_reg_zp_a_comp, ptr(X_SP, aux_reg_zp_a_comp_offs_));
                    add_imm(aux_reg_zp_a_comp, aux_reg_zp_a_comp,
                            sizeof(int32_t) * oc_l_offset, X_TMP_0);
                    str(aux_reg_zp_a_comp, ptr(X_SP, aux_reg_zp_a_comp_offs_));
                }
                if (brg.req_s8s8_compensation) {
                    ldr(aux_reg_s8s8_comp, ptr(X_SP, aux_reg_s8s8_comp_offs_));
                    add_imm(aux_reg_s8s8_comp, aux_reg_s8s8_comp,
                            sizeof(int32_t) * oc_l_offset, X_TMP_0);
                    str(aux_reg_s8s8_comp, ptr(X_SP, aux_reg_s8s8_comp_offs_));
                }

                add_imm(aux_reg_scales, aux_reg_scales,
                        is_oc_scale_ * sizeof(float) * oc_l_offset, X_TMP_0);
            }
        }
        if (nb_tail > 0) {
            apply_post_ops(m_block, 1, nb_tail);

            if (brg.alpha != 0) {
                add_imm(aux_reg_in, aux_reg_in, inp_typesize_ * (nb_tail),
                        X_TMP_0);
            }
            if (brg.beta != 0) {
                if (jcp.with_bias)
                    add_imm(aux_reg_bias, aux_reg_bias,
                            bia_typesize_ * (nb_tail), X_TMP_0);
                if (brg.zp_type_c != brgemm_broadcast_t::none) {
                    ldr(aux_reg_zp_c_values,
                            ptr(X_SP, aux_reg_zp_c_values_offs_));
                    add_imm(aux_reg_zp_c_values, aux_reg_zp_c_values,
                            zp_c_values_offset(1, nb_tail), X_TMP_0);
                    str(aux_reg_zp_c_values,
                            ptr(X_SP, aux_reg_zp_c_values_offs_));
                }
                if (brg.zp_type_a != brgemm_broadcast_t::none) {
                    ldr(aux_reg_zp_a_comp, ptr(X_SP, aux_reg_zp_a_comp_offs_));
                    add_imm(aux_reg_zp_a_comp, aux_reg_zp_a_comp,
                            sizeof(int32_t) * nb_tail, X_TMP_0);
                    str(aux_reg_zp_a_comp, ptr(X_SP, aux_reg_zp_a_comp_offs_));
                }
                if (brg.req_s8s8_compensation) {
                    ldr(aux_reg_s8s8_comp, ptr(X_SP, aux_reg_s8s8_comp_offs_));
                    add_imm(aux_reg_s8s8_comp, aux_reg_s8s8_comp,
                            sizeof(int32_t) * nb_tail, X_TMP_0);
                    str(aux_reg_s8s8_comp, ptr(X_SP, aux_reg_s8s8_comp_offs_));
                }
                add_imm(aux_reg_scales, aux_reg_scales,
                        is_oc_scale_ * bia_typesize_ * (nb_tail), X_TMP_0);
            }
            add_imm(aux_reg_out, aux_reg_out, out_typesize_ * (nb_tail),
                    X_TMP_0);
        }
    }

    void generate() override {
        size_t simd_w_ = 0;
        switch (brg.isa_impl) {
            case sve_512:
                simd_w_ = cpu_isa_traits<sve_512>::vlen / sizeof(float);
                break;
            case sve_256:
                simd_w_ = cpu_isa_traits<sve_256>::vlen / sizeof(float);
                break;
            default: {
                assert(!"unsupported isa");
                return;
            }
        }
        preamble();
        if (simd_w_ != cpu_sveLen / sizeof(float)) {
            set_preg(P_ALL_ONE.b, simd_w_ * 4, X_TMP_0, X_TMP_1);
            set_preg(k_full_mask.b, simd_w_ * 4, X_TMP_0, X_TMP_1);
        } else
            ptrue(k_full_mask.b);

        mov(x7, x0);
        mov(x6, x1);
        mov(x2, x2);
        mov(x1, x3);
        mov(x8, x4);
        mov(x9, x5);

        mov(x4, X_SP);
        int nb = brg.load_dim / brg.ld_block;
        int nb_tail = brg.load_dim % brg.ld_block;

        int nb2 = nb / n_block2_;
        int nb2_tail = nb % n_block2_;
        int n_block = (nb2 == 0) ? nstl::max(1, nb2_tail) : n_block2_;

        int m_max_regs = (max_vregs_ - 4) / n_block;
        int m_block = nstl::min(brg.bcast_dim, m_max_regs);

        int mb = brg.bcast_dim / m_block;
        int mb_tail = brg.bcast_dim % m_block;

        set_preg(k_tail_mask.s, nb_tail, X_TMP_0, X_TMP_1);

        if (brg.alpha != 0) {
            add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_in), X_TMP_0);
            ldr(reg_in, ptr(X_DEFAULT_ADDR));
        }
        if (brg.beta != 0) {
            add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_scales), X_TMP_0);
            ldr(reg_scales, ptr(X_DEFAULT_ADDR));
            add_imm(X_DEFAULT_ADDR, param1, GET_OFF(apply_comp), X_TMP_0);
            ldr(reg_apply_comp, ptr(X_DEFAULT_ADDR));
            str(reg_apply_comp, ptr(X_SP, reg_apply_comp_offs_));

            if (jcp.with_bias) {
                add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_bias), X_TMP_0);
                ldr(reg_bias, ptr(X_DEFAULT_ADDR));
            }
            if (brg.zp_type_c != brgemm_broadcast_t::none) {
                add_imm(X_DEFAULT_ADDR, param1, GET_OFF(c_zp_values), X_TMP_0);
                ldr(reg_zp_c_values, ptr(X_DEFAULT_ADDR));
                str(reg_zp_c_values, ptr(X_SP, reg_zp_c_values_offs_));
            }
            if (brg.zp_type_a != brgemm_broadcast_t::none) {
                add_imm(X_DEFAULT_ADDR, param1, GET_OFF(a_zp_compensation),
                        X_TMP_0);
                ldr(reg_zp_a_comp, ptr(X_DEFAULT_ADDR));
                str(reg_zp_a_comp, ptr(X_SP, reg_zp_a_comp_offs_));

                add_imm(X_DEFAULT_ADDR, param1, GET_OFF(a_comp_val), X_TMP_0);
                ldr(reg_zp_a_val, ptr(X_DEFAULT_ADDR));
                str(reg_zp_a_val, ptr(X_SP, reg_zp_a_val_offs_));
            }
            if (brg.req_s8s8_compensation) {
                add_imm(X_DEFAULT_ADDR, param1, GET_OFF(s8s8_compensation),
                        X_TMP_0);
                ldr(reg_s8s8_comp, ptr(X_DEFAULT_ADDR));
                str(reg_s8s8_comp, ptr(X_SP, reg_s8s8_comp_offs_));
            }
            if (brg.with_dst_scales) {
                add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_dst_scales),
                        X_TMP_0);
                ldr(reg_dst_scales, ptr(X_DEFAULT_ADDR));
                str(reg_dst_scales, ptr(X_SP, reg_dst_scales_offs_));
            }
        }
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_out), X_TMP_0);
        ldr(reg_out, ptr(X_DEFAULT_ADDR));

        // brg.alpha == 0 means initialize registers, 1 means read from input
        // brg.beta == 0 means skip postwork, 1 means do postwork
        if (brg.alpha == 0 && brg.beta == 0) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                auto zmm = ZReg(m * n_block + n);
                eor(zmm.d, zmm.d, zmm.d);
            }
        }

        for (int mb_ = 0; mb_ < mb; mb_++) {
            loop_by_N(m_block, nb2, nb2_tail, nb_tail);

            if (brg.alpha != 0)
                add_imm(reg_in, reg_in, inp_typesize_ * (m_block * brg.LDC),
                        X_TMP_0);
            add_imm(reg_out, reg_out, out_typesize_ * (m_block * LDD_),
                    X_TMP_0);
        }
        if (mb_tail > 0) loop_by_N(mb_tail, nb2, nb2_tail, nb_tail);

        add_imm(X_SP, X_SP, stack_space_needed_, X_TMP_0);

        postamble();

        if (postops_injector_) postops_injector_->prepare_table();
    }
};

#undef GET_OFF

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
