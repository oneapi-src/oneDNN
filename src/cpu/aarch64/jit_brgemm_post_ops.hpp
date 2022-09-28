/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2023 FUJITSU LIMITED
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

    // Register decomposition
    const XReg param1 = x7;
    const XReg reg_ddst = x15;
    const XReg reg_bias = x14;
    const XReg reg_bias_acc = x13;
    const XReg aux_reg_ddst = x12;
    const XReg reg_k_iter = x11;
    const XReg reg_flag = x10;

    PReg k_full_mask = PReg(2);
    PReg k_tail_mask = PReg(3);
    ZReg vreg_unit = ZReg(31);

    const int n_max_regs_ = 4;

    void loop_by_N(int n_loop, int nb_tail) { assert(!"unsupported\n"); }

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

        ptrue(k_full_mask.b);
        set_preg(k_tail_mask.s, nb_tail, X_TMP_0, X_TMP_1);
        pfalse(P_TMP_0.b);
        zip1(k_tail_mask.b, k_tail_mask.b, P_TMP_0.b);
        zip1(k_tail_mask.h, k_tail_mask.h, P_TMP_0.h);
        if (ddst_dt_ == data_type::bf16) { assert(!"unsupported data type\n"); }

        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_diff_dst), X_TMP_0);
        ldr(reg_ddst, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_diff_bias_acc), X_TMP_0);
        ldr(reg_bias_acc, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_diff_bias), X_TMP_0);
        ldr(reg_bias, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(flags), X_TMP_0);
        ldr(reg_flag, ptr(X_DEFAULT_ADDR));

        int mult = ddst_dt_ == data_type::bf16 ? 2 : 1;
        for (int nb_ = 0; nb_ < n_loop; nb_++) {
            loop_by_N(n_max_regs_, 0);

            add_imm(reg_ddst, reg_ddst,
                    ddst_typesize_ * mult * n_max_regs_ * brg_.ld_block,
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

struct jit_brgemm_kernel_post_ops : public jit_generator {

    jit_brgemm_kernel_post_ops(const jit_brgemm_conv_conf_t &ajcp,
            const brgemm_t &abrg, const primitive_attr_t &aattr)
        : brg(abrg)
        , jcp(ajcp)
        , attr(aattr)
#if WITH_POSTOPS_INJECTOR
        , postops_injector_(nullptr)
#endif
        , with_binary_per_oc_bcast_(brg.with_binary
#if WITH_POSTOPS_INJECTOR
                  && binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                          brg.attr->post_ops_, memory_desc_wrapper(brg.dst_md))
#endif
          ) {

        if ((jcp.with_sum && brg.beta != 0)
                || ((jcp.with_binary || jcp.with_eltwise) && brg.alpha != 0)) {
            assert(!"unsupported\n");
#if WITH_POSTOPS_INJECTOR
            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    static_cast<size_t>(Xbyak::Zmm(28).getIdx()), XReg(14),
                    XReg(15), preserve_gpr, preserve_vmm,
                    GET_OFF(ptr_binary_post_ops_rhs), GET_OFF(dst_orig),
                    memory_desc_wrapper(brg.dst_md),
                    static_cast<size_t>(brg.load_dim % brg.ld_block),
                    PReg(k_tail_mask.getIdx()), use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {
                    XReg(this->param1.getIdx()), rhs_sp};

            static constexpr bool save_state = true;
            const auto &reserved_eltwise_gpr = rax;
            const auto reserved_eltwise_maskr = Xbyak::Opmask(1);
            const eltwise_injector::static_params_t esp {save_state,
                    XReg(reserved_eltwise_gpr.getIdx()),
                    PReg(reserved_eltwise_maskr.getIdx())};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<avx512_common>>(
                    this, attr.post_ops_, bsp, esp);
#endif
        }

        const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
        is_oc_scale_ = wei_scales.mask_ == 1 << 1;

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
#if WITH_POSTOPS_INJECTOR
    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_common>>
            postops_injector_;
#endif

    const bool with_binary_per_oc_bcast_;

    int inp_typesize_;
    int out_typesize_;
    int bia_typesize_;

    int is_oc_scale_;

    // Register decomposition
    const XReg param1 = x7; //abi_param1=rdi
    const XReg reg_in = x15;
    const XReg reg_out = x14;
    const XReg aux_reg_in = x13;
    const XReg aux_reg_out = x12;

    const XReg reg_bias = x11;
    const XReg aux_reg_bias = x10;

    const XReg reg_scales = x9;
    const XReg aux_reg_scales = x8;

    const XReg reg_ptr_sum_scale = x2; //rdx;
    const XReg reg_ptr_sum_zp = x6; //rsi

    const XReg reg_oc_l_offset_ = x1; //rcx
    const XReg aux_reg_oc_l_offset_ = x3; //rbx

    PReg k_full_mask = p2;
    PReg k_tail_mask = p3;

    const int n_block2_ = 4;

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
                    ld1w(zmm_in.s, P_ALL_ONE / T_z, op);
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

    void cvt2ps(data_type_t type_in, const ZReg zmm_in, const ZReg &op,
            bool mask_flag, bool store, PReg ktail_mask) {
        switch (type_in) {
            case data_type::f32:
            case data_type::s32:
                if (mask_flag) {
                    if (store) {
                        mov(zmm_in.s, k_tail_mask / T_m, op.s);
                    } else {
                        mov(zmm_in.s, k_tail_mask / T_z, op.s);
                    }
                } else {
                    mov(zmm_in.s, P_ALL_ONE / T_z, op.s);
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
        assert(!"unsupported\n");
#if WITH_POSTOPS_INJECTOR
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
                    CodeGenerator::fadd(ZReg(zmm.getIdx()).s,
                            ZReg(zmm.getIdx()).s,
                            ZReg(zmm_prev_dst.getIdx()).s);
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

                rhs_arg_params.vmm_idx_to_out_reg.emplace(
                        zmm_idx, XReg(aux_reg_out.getIdx()));
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        zmm_idx, aux_output_offset);
                if (tail) rhs_arg_params.vmm_tail_idx_.emplace(zmm_idx);
            }
        }

        postops_injector_->compute_vector_range(
                0, m_block * n_block, rhs_arg_params);
#endif
    }

    void apply_post_ops(int m_block, int n_block, int tail = 0) {
        const auto vector = [=](int m, int n) { return ZReg(m * n_block + n); };
        auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;
        const auto &p = attr.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);

        // brg.alpha == 0 means no read from input, no bias, no eltwise - just
        // initialize registers by zero at the beginning of kernel
        // brg.beta == 0 means no sum - just registers write to output
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            if (brg.alpha == 0) {
                if (sum_idx != -1 && brg.beta != 0) {
                    // if sum then have to init zmm each time
                    eor(vector(m, n).d, vector(m, n).d, vector(m, n).d);
                }
            } else {
                add_imm(X_DEFAULT_ADDR, aux_reg_in,
                        inp_typesize_ * (m * brg.LDC + n * brg.ld_block),
                        X_TMP_0);
                cvt2ps(inp_dt_, vector(m, n), ptr(X_DEFAULT_ADDR), true, false,
                        k_mask);
            }
        }

        if (brg.alpha != 0 && jcp.with_bias) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                auto zmm_bias = ZReg(31);
                add_imm(X_DEFAULT_ADDR, aux_reg_bias,
                        bia_typesize_ * (n * brg.ld_block), X_TMP_0);

                cvt2ps(bia_dt_, zmm_bias, ptr(X_DEFAULT_ADDR), true, false,
                        k_mask);
                fadd(vector(m, n).s, vector(m, n).s, zmm_bias.s);
            }
        }

        if (brg.alpha != 0) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                CodeGenerator::add_imm(XReg(28), XReg(aux_reg_scales.getIdx()),
                        is_oc_scale_ * sizeof(float) * (n * brg.ld_block),
                        XReg(23));
                CodeGenerator::sub_imm(XReg(31), XReg(31), 64,
                        XReg(27)); // XReg(31)=sp
                CodeGenerator::str(ZReg(31), ptr(XReg(31)));
                CodeGenerator::ldr(ZReg(31), ptr(XReg(28)));
                CodeGenerator::fmul(
                        ZReg(31).s, ZReg(vector(m, n).getIdx()).s, ZReg(31).s);
                CodeGenerator::pfalse(PRegB(9));
                CodeGenerator::zip1(
                        PRegB(10), PRegB(k_mask.getIdx()), PRegB(9));
                CodeGenerator::zip1(
                        PRegH(10), PRegH(k_mask.getIdx()), PRegH(9));
                CodeGenerator::mov(ZRegS(vector(m, n).getIdx()), PReg(10) / T_m,
                        ZRegS(31));
                CodeGenerator::ldr(ZReg(31), ptr(XReg(31)));
                CodeGenerator::add_imm(XReg(31), XReg(31), 64,
                        XReg(27)); // XReg(31)=sp
            }
        }

#if WITH_POSTOPS_INJECTOR
        if (postops_injector_) inject_attr_postops(m_block, n_block, tail);
#endif

        const bool dt_requires_saturation = utils::one_of(
                brg.dt_d, data_type::u8, data_type::s8, data_type::s32);

        auto zmm_lbound = ZReg(31);
        auto zmm_ubound = ZReg(30);
        if (dt_requires_saturation) { assert(!"unsupported\n"); }

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto zmm = vector(m, n);

            if (out_dt_ == data_type::bf16) {
                assert(!"unsupported data type\n");
            } else {
                if (brg.alpha != 0 || (sum_idx != -1 && brg.beta != 0)) {
                    saturate_f32(
                            zmm, zmm_lbound, zmm_ubound, brg.dt_d, PReg(0));
                    if (out_dt_ != data_type::f32)
                        assert(!"unsupported data type\n");
                }

                switch (out_dt_) {
                    case data_type::f32:
                    case data_type::s32:
                        add_imm(X_DEFAULT_ADDR, aux_reg_out,
                                out_typesize_ * (m * LDD_ + n * brg.ld_block),
                                X_TMP_0); //addr
                        st1w(zmm.s, k_mask / T_m, ptr(X_DEFAULT_ADDR));
                        break;
                    case data_type::s8:
                        assert(!"unsupported data type\n");
                        break;
                    case data_type::u8:
                        assert(!"unsupported data type\n");
                        break;
                    default: assert(!"unknown dst_dt");
                }
            }
        }
    }

    void loop_by_N(int m_block, int nb2, int nb2_tail, int nb_tail) {

        if (brg.alpha) {
            mov(aux_reg_in, reg_in);
            if (jcp.with_bias) { mov(aux_reg_bias, reg_bias); }
            if (with_binary_per_oc_bcast_) { assert(!"unsupported\n"); }
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

                if (jcp.with_bias)
                    add_imm(aux_reg_bias, aux_reg_bias,
                            bia_typesize_ * oc_l_offset, X_TMP_0);
                if (with_binary_per_oc_bcast_) assert(!"unsupported\n");

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
                if (jcp.with_bias)
                    add_imm(aux_reg_bias, aux_reg_bias,
                            bia_typesize_ * oc_l_offset, X_TMP_0);
                if (with_binary_per_oc_bcast_) assert(!"unsupported\n");

                add_imm(aux_reg_scales, aux_reg_scales,
                        is_oc_scale_ * sizeof(float) * oc_l_offset, X_TMP_0);
            }
        }
        if (nb_tail > 0) {
            apply_post_ops(m_block, 1, nb_tail);

            if (brg.alpha != 0) {
                add_imm(aux_reg_in, aux_reg_in, inp_typesize_ * (nb_tail),
                        X_TMP_0);
                if (jcp.with_bias)
                    add_imm(aux_reg_bias, aux_reg_bias,
                            bia_typesize_ * (nb_tail), X_TMP_0);
                if (with_binary_per_oc_bcast_) assert(!"unsupported\n");
                add_imm(aux_reg_scales, aux_reg_scales,
                        is_oc_scale_ * bia_typesize_ * (nb_tail), X_TMP_0);
            }
            add_imm(aux_reg_out, aux_reg_out, out_typesize_ * (nb_tail),
                    X_TMP_0);
        }
    }

    void generate() override {
        preamble();

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

        int m_max_regs = 28 / n_block;
        int m_block = nstl::min(brg.bcast_dim, m_max_regs);

        int mb = brg.bcast_dim / m_block;
        int mb_tail = brg.bcast_dim % m_block;

        ptrue(k_full_mask.b);
        set_preg(k_tail_mask.s, nb_tail, X_TMP_0, X_TMP_1);

        if (brg.alpha != 0) {
            add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_in), X_TMP_0);
            ldr(reg_in, ptr(X_DEFAULT_ADDR));
            add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_scales), X_TMP_0);
            ldr(reg_scales, ptr(X_DEFAULT_ADDR));

            if (jcp.with_bias) {
                add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_bias), X_TMP_0);
                ldr(reg_bias, ptr(X_DEFAULT_ADDR));
            }
        }
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_out), X_TMP_0);
        ldr(reg_out, ptr(X_DEFAULT_ADDR));

        // brg.alpha == 0 means no read from input, no bias, no eltwise - just
        // initialize registers by zero
        // brg.beta == 0 means no sum - just registers write to output
        if (brg.alpha == 0) {
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

        postamble();

#if WITH_POSTOPS_INJECTOR
        if (brg.alpha != 0 && jcp.with_eltwise)
            postops_injector_->prepare_table();
#endif
    }
};

#undef GET_OFF

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
