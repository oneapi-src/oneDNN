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

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/brgemm/brgemm_amx.hpp"
#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_eltwise_injector.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

struct jit_brgemm_kernel_base_t : public jit_generator {
    jit_brgemm_kernel_base_t(const brgemm_t &abrg)
        : brg(abrg), eltwise_injector_(nullptr) {
        if (brg.with_eltwise) {
            const auto &p = brg.attr->post_ops_;
            const int eltwise_ind = p.find(primitive_kind::eltwise);

            post_ops_t::entry_t::eltwise_t eltwise;
            eltwise = p.entry_[eltwise_ind].eltwise;
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, eltwise, true, rax, Xbyak::Opmask(1));
        }
    }

    ~jit_brgemm_kernel_base_t() override { delete eltwise_injector_; }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_base_t)

    brgemm_t brg;

private:
    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;

    using reg64_t = const Xbyak::Reg64;

    // Register decomposition
    const reg64_t param1 = abi_param1;

    const reg64_t reg_C = r15;
    const reg64_t reg_aux_C = r14;

    const reg64_t reg_A = r13;
    const reg64_t reg_B = r12;

    const reg64_t reg_aux_A = r11;
    const reg64_t reg_aux_B = r10;

    const reg64_t reg_mb_loop = r9;
    const reg64_t reg_nb_loop = r8;

    const reg64_t reg_stride_lda = reg_mb_loop;
    const reg64_t reg_stride_ldb = reg_nb_loop;
    const reg64_t reg_stride_n_block = reg_nb_loop;

    const reg64_t reg_N_loop = rax;
    const reg64_t reg_kb_loop = rbx;
    const reg64_t reg_N = abi_not_param1;

    const reg64_t reg_a_offset = rdx;
    const reg64_t reg_b_offset = rsi;

    const reg64_t reg_aux1_A = rbp;
    const reg64_t reg_aux1_B = abi_param1;

    const reg64_t reg_offset_A = reg_aux1_A;
    const reg64_t reg_offset_B = reg_aux1_B;

    const reg64_t reg_bias = reg_kb_loop;
    const reg64_t reg_scales = reg_kb_loop;
    const reg64_t reg_aux_bias = reg_kb_loop;
    const reg64_t reg_aux_scales = reg_kb_loop;
    const reg64_t reg_do_post_ops = reg_kb_loop;
    const reg64_t reg_tmp_gpr = reg_kb_loop;
    const reg64_t reg_ptr_sum_scale = reg_kb_loop;

    const reg64_t reg_buf = reg_kb_loop;

    const reg64_t reg_D = reg_aux_A;
    const reg64_t reg_aux_D = reg_N_loop;

    constexpr static int origin_offset_A_offs_ = 0;
    constexpr static int origin_offset_B_offs_ = 8;
    constexpr static int reg_bias_offs_ = 16;
    constexpr static int reg_aux_bias_offs_ = 24;
    constexpr static int reg_do_post_ops_offs_ = 32;
    constexpr static int reg_D_offs_ = 40;
    constexpr static int reg_aux_D_offs_ = 48;
    constexpr static int reg_scales_offs_ = 56;
    constexpr static int reg_aux_scales_offs_ = 64;
    constexpr static int reg_mb_loop_offs_ = 72;
    constexpr static int reg_nb_loop_offs_ = 80;
    constexpr static int reg_buf_offs_ = 88;
    constexpr static int stack_space_needed_ = 128;

    Xbyak::Opmask k_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_tail_mask = Xbyak::Opmask(3);

    Xbyak::Zmm accm(int n_block, int m, int n) {
        return Xbyak::Zmm(31 - (m * n_block + n));
    }
#if _N_BCST_1_LOAD
    Xbyak::Zmm bcst(int m) {
        return Xbyak::Zmm(31 - (brg.n_block2 * brg.m_block + m));
    }
    Xbyak::Zmm load() { return Xbyak::Zmm(31); }
#endif
    Xbyak::Zmm load(int n) {
        assert(brg.n_block2 * brg.m_block < 31); // zmm0 is bcast ?
        return Xbyak::Zmm(31 - (brg.n_block2 * brg.m_block) - n);
    }

    Xbyak::Zmm bcst() { return Xbyak::Zmm(0); }

    Xbyak::Zmm zmm_tmp_1() { return Xbyak::Zmm(0); }
    Xbyak::Zmm zmm_tmp_2() { return Xbyak::Zmm(1); }
    Xbyak::Zmm zmm_tmp_3() { return Xbyak::Zmm(2); }

    Xbyak::Zmm zmm_mask(const Xbyak::Zmm zmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);
    Xbyak::Ymm ymm_mask(const Xbyak::Ymm ymm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);

    void cvt2ps(data_type_t type_in, const Xbyak::Zmm zmm_in,
            const Xbyak::Operand &op, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);

    void read_params();
    void load_accumulators(int m_block2, bool is_mb_tail, int n_block);

    void store_accumulators(
            int m_block2, bool is_mb_tail, int n_block, bool is_n_tail);
    void store_accumulators_without_post_ops(
            int m_block, int n_block, bool is_n_tail);
    void store_accumulators_apply_post_ops(
            int m_block, int n_block, bool is_n_tail);
    void apply_beta(int m_block, int n_block, bool is_n_tail);

    void restore_A_B_matrices();
    void restore_offsets();
    void set_A_B_matrices();

    void gemm_microkernel_avx512(int m_block2, bool is_mb_tail, int n_block,
            bool is_k_tail, bool is_n_tail);
    void gemm_microkernel_amx(int m_block2, bool is_mb_tail, int n_block,
            bool is_k_tail, bool is_nb_tail);
    void gemm_microkernel(int m_block2, bool is_mb_tail, int n_block,
            bool is_k_tail, bool is_nb_tail);

    void nb_loop(int m_block2, bool is_mb_tail, int n_block, int nb_loop_length,
            bool is_reg_tail, bool is_n_tail);
    void mb_loop();

    void generate() override;
};

Xbyak::Zmm jit_brgemm_kernel_base_t::zmm_mask(const Xbyak::Zmm zmm_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) {
    return mask_flag ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                     : zmm_in;
}

Xbyak::Ymm jit_brgemm_kernel_base_t::ymm_mask(const Xbyak::Ymm ymm_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) {
    return mask_flag ? (store ? ymm_in | ktail_mask : ymm_in | ktail_mask | T_z)
                     : ymm_in;
}

void jit_brgemm_kernel_base_t::cvt2ps(data_type_t type_in,
        const Xbyak::Zmm zmm_in, const Xbyak::Operand &op, bool mask_flag,
        bool store, Xbyak::Opmask ktail_mask) {
    const Xbyak::Zmm zmm = zmm_mask(zmm_in, mask_flag, store, ktail_mask);
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: vmovups(zmm, op); break;
        case data_type::bf16:
            vpmovzxwd(zmm, op);
            vpslld(zmm, zmm, 16);
            break;
        case data_type::s8: vpmovsxbd(zmm, op); break;
        case data_type::u8: vpmovzxbd(zmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (!one_of(type_in, data_type::f32, data_type::bf16))
        vcvtdq2ps(zmm_in, zmm_in);
}

void jit_brgemm_kernel_base_t::read_params() {
    Label label_done;

    mov(reg_A, ptr[param1 + GET_OFF(ptr_A)]);
    mov(reg_B, ptr[param1 + GET_OFF(ptr_B)]);
    mov(reg_C, ptr[param1 + GET_OFF(ptr_C)]);
    mov(reg_D, ptr[param1 + GET_OFF(ptr_D)]);
    mov(reg_N, ptr[param1 + GET_OFF(N)]);
    if (brg.is_int8_amx || brg.is_bf16_amx) {
        mov(reg_buf, ptr[param1 + GET_OFF(ptr_buf)]);
        mov(ptr[rsp + reg_buf_offs_], reg_buf);
    }

    if (brg.type == brgemm_offs) {
        mov(reg_offset_A, ptr[param1 + GET_OFF(offset_A)]);
        mov(reg_offset_B, ptr[param1 + GET_OFF(offset_B)]);

        mov(ptr[rsp + origin_offset_A_offs_], reg_offset_A);
        mov(ptr[rsp + origin_offset_B_offs_], reg_offset_B);
    }
    if (brg.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
        mov(ptr[rsp + reg_bias_offs_], reg_bias);
    }
    if (brg.with_scales) {
        mov(reg_scales, ptr[param1 + GET_OFF(ptr_scales)]);
        mov(ptr[rsp + reg_scales_offs_], reg_scales);
    }

    mov(reg_do_post_ops, ptr[param1 + GET_OFF(do_post_ops)]);
    mov(ptr[rsp + reg_do_post_ops_offs_], reg_do_post_ops);
}

void jit_brgemm_kernel_base_t::load_accumulators(
        int m_block2, bool is_mb_tail, int n_block2) {
    if (brg.is_int8_amx || brg.is_bf16_amx) {
        for_(int mb = 0; mb < m_block2; mb++)
        for (int nb = 0; nb < n_block2; nb++)
            tilezero(Tmm(brgemm_amx::get_C_tensor(mb, nb)));
    } else {
        int m_block = (is_mb_tail) ? brg.mb_tail : brg.m_block;
        for (int m = 0; m < m_block; m++) {
            for (int n = 0; n < n_block2; n++) {
                auto zmm = accm(n_block2, m, n);
                vxorps(zmm, zmm, zmm);
            }
        }
    }
}

void jit_brgemm_kernel_base_t::apply_beta(
        int m_block, int n_block2, bool is_n_tail) {
    auto k_mask = (!is_n_tail) ? k_full_mask : k_tail_mask;
    auto zmm_beta = zmm_tmp_1();
    auto zmm_alpha = zmm_tmp_2();
    auto zmm_prev_dst = zmm_tmp_3();

    const bool apply_alpha = (brg.alpha != 1.f && brg.alpha != 0.f);

    if (brg.beta != 1.f) {
        mov(reg_tmp_gpr, float2int((float)brg.beta));
        movq(Xmm(zmm_beta.getIdx()), reg_tmp_gpr);
        vbroadcastss(zmm_beta, Xmm(zmm_beta.getIdx()));
    }
    if (apply_alpha) {
        mov(reg_tmp_gpr, float2int((float)brg.alpha));
        movq(Xmm(zmm_alpha.getIdx()), reg_tmp_gpr);
        vbroadcastss(zmm_alpha, Xmm(zmm_alpha.getIdx()));
    }
    for (int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block2; n++) {
            int offset_C = brg.typesize_C * (m * brg.LDC + n * brg.n_block);
            auto zmm = accm(n_block2, m, n);
            if (brg.is_int8 && (apply_alpha || brg.beta != 1.f))
                vcvtdq2ps(zmm, zmm);
            if (apply_alpha) vmulps(zmm, zmm, zmm_alpha);
            if (brg.beta != 1.f) {
                cvt2ps(brg.dt_c, zmm_prev_dst, ptr[reg_aux_C + offset_C], true,
                        false, k_mask);
                vfmadd231ps(zmm, zmm_prev_dst, zmm_beta);
            } else {
                if (brg.is_int8)
                    vpaddd(zmm | k_mask | T_z, zmm, ptr[reg_aux_C + offset_C]);
                else
                    vaddps(zmm | k_mask | T_z, zmm, ptr[reg_aux_C + offset_C]);
            }
        }
}

void jit_brgemm_kernel_base_t::store_accumulators_apply_post_ops(
        int m_block, int n_block2, bool is_n_tail) {
    auto k_mask = (!is_n_tail) ? k_full_mask : k_tail_mask;

    if (brg.with_bias) { mov(reg_aux_bias, ptr[rsp + reg_aux_bias_offs_]); }
    for (int m = 0; m < m_block; m++) {
        for (int n = 0; n < n_block2; n++) {
            auto zmm = accm(n_block2, m, n);
            if (!brg.is_f32 && !brg.is_bf16
                    && ((brg.beta != 0.f) || (brg.beta != 1.f)))
                vcvtdq2ps(zmm, zmm);
            if (brg.with_bias) {
                auto zmm_bias = zmm_tmp_1();
                int bias_offset = brg.typesize_bias * n * brg.n_block;
                cvt2ps(brg.dt_bias, zmm_bias, ptr[reg_aux_bias + bias_offset],
                        true, false, k_mask);
                vaddps(zmm, zmm, zmm_bias);
            }
        }
    }
    if (brg.with_scales) {
        mov(reg_aux_scales, ptr[rsp + reg_aux_scales_offs_]);
        for (int m = 0; m < m_block; m++) {
            for (int n = 0; n < n_block2; n++) {
                const Xbyak::Zmm zmm
                        = zmm_mask(accm(n_block2, m, n), true, false, k_mask);
                vmulps(zmm, zmm,
                        ptr[reg_aux_scales
                                + brg.is_oc_scale * sizeof(float)
                                        * (n * brg.n_block)]);
            }
        }
    }

    bool sum_before_eltwise = false;
    if (brg.with_sum) {
        const auto &p = brg.attr->post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        sum_before_eltwise
                = (sum_idx == 0) && p.contain(primitive_kind::eltwise, 1);
    }

    if (brg.with_eltwise && !sum_before_eltwise)
        eltwise_injector_->compute_vector_range(32 - m_block * n_block2, 32);

    if (brg.with_sum) {
        const float *p_sum_scale = &brg.sum_scale;
        if (*p_sum_scale != 1.f) mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

        for (int m = 0; m < m_block; m++) {
            for (int n = 0; n < n_block2; n++) {
                auto zmm = accm(n_block2, m, n);
                auto addr = ptr[reg_aux_D
                        + brg.typesize_D * (m * brg.LDD + n * brg.n_block)];

                auto zmm_prev_dst = Xbyak::Zmm(0);
                cvt2ps(brg.dt_d, zmm_prev_dst, addr, true, false, k_mask);
                if (*p_sum_scale == 1.f)
                    vaddps(zmm, zmm_prev_dst);
                else
                    vfmadd231ps(zmm, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
            }
        }
    }

    if (brg.with_eltwise && sum_before_eltwise)
        eltwise_injector_->compute_vector_range(32 - m_block * n_block2, 32);

    auto zmm_zero = zmm_tmp_1();
    if (brg.dt_d == data_type::u8) vpxord(zmm_zero, zmm_zero, zmm_zero);

    for (int m = 0; m < m_block; m++) {
        for (int n = 0; n < n_block2; n++) {
            auto zmm = accm(n_block2, m, n);
            if (brg.dt_d == data_type::u8) vmaxps(zmm, zmm_zero, zmm);
            if (!one_of(brg.dt_d, data_type::f32, data_type::bf16))
                vcvtps2dq(zmm, zmm);
        }
        for (int n = 0; n < n_block2; n++) {
            auto addr = ptr[reg_aux_D
                    + brg.typesize_D * (m * brg.LDD + n * brg.n_block)];
            auto zmm = accm(n_block2, m, n);
            auto ymm = Xbyak::Ymm(zmm.getIdx());
            const Xbyak::Zmm r_zmm = zmm_mask(zmm, true, true, k_mask);
            const Xbyak::Ymm r_ymm = ymm_mask(ymm, true, true, k_mask);
            switch (brg.dt_d) {
                case data_type::f32:
                case data_type::s32: vmovups(addr, r_zmm); break;
                case data_type::bf16:
                    vcvtneps2bf16(ymm, zmm);
                    vmovdqu16(addr, r_ymm);
                    break;
                case data_type::s8: vpmovsdb(addr, r_zmm); break;
                case data_type::u8: vpmovusdb(addr, r_zmm); break;
                default: assert(!"unknown dst_dt");
            }
        }
    }
}

void jit_brgemm_kernel_base_t::store_accumulators_without_post_ops(
        int m_block, int n_block2, bool is_n_tail) {
    for (int m = 0; m < m_block; m++) {
        for (int n = 0; n < n_block2; n++) {
            auto zmm = accm(n_block2, m, n);
            int offset_C = brg.typesize_C * (m * brg.LDC + n * brg.n_block);
            if (!one_of(brg.beta, 1.f, 0.f) && (!brg.is_f32 && !brg.is_bf16))
                vcvtps2dq(zmm, zmm);
            if (is_n_tail)
                vmovups(ptr[reg_aux_C + offset_C] | k_tail_mask | T_z, zmm);
            else
                vmovups(ptr[reg_aux_C + offset_C], zmm);
        }
    }
}

void jit_brgemm_kernel_base_t::store_accumulators(
        int m_block2, bool is_mb_tail, int n_block2, bool is_n_tail) {
    if (brg.is_int8_amx || brg.is_bf16_amx) {
        mov(ptr[rsp + reg_nb_loop_offs_], reg_nb_loop);
        if (brg.beta != 0.f && brg.alpha != 0)
            mov(reg_stride_n_block, brg.n_block * brg.typesize_C);
        else
            mov(reg_stride_n_block, brg.LDC * brg.typesize_C);

        mov(reg_buf, ptr[rsp + reg_buf_offs_]);
        for (int mb = 0; mb < m_block2; mb++) {
            for (int nb = 0; nb < n_block2; nb++) {
                if (brg.beta != 0.f && brg.alpha != 0) {
                    tilestored(ptr[reg_buf + reg_stride_n_block],
                            Tmm(brgemm_amx::get_C_tensor(mb, nb)));
                    for (int m = 0; m < brg.m_block; m++) {
                        size_t buf_offset = (m * brg.n_block) * brg.typesize_C;
                        if (is_n_tail)
                            vmovups(accm(1, m, 0) | k_tail_mask | T_z,
                                    ptr[reg_buf + buf_offset]);
                        else
                            vmovups(accm(1, m, 0), ptr[reg_buf + buf_offset]);
                    }
                    apply_beta(brg.m_block, 1, is_n_tail);
                    store_accumulators_without_post_ops(
                            brg.m_block, 1, is_n_tail);
                } else {
                    tilestored(ptr[reg_aux_C + reg_stride_n_block],
                            Tmm(brgemm_amx::get_C_tensor(mb, nb)));
                }
                add(reg_aux_C, brg.typesize_C * brg.n_block);
            }
            sub(reg_aux_C, brg.typesize_C * n_block2 * brg.n_block);
            add(reg_aux_C, brg.typesize_C * brg.m_block * brg.LDC);
        }
        sub(reg_aux_C, brg.typesize_C * m_block2 * brg.m_block * brg.LDC);
        mov(reg_nb_loop, ptr[rsp + reg_nb_loop_offs_]);
    } else {
        int m_block = (is_mb_tail) ? brg.mb_tail : brg.m_block;
        if (brg.beta != 0.f && brg.alpha != 0) {
            apply_beta(m_block, n_block2, is_n_tail);
        }
        if (one_of(true, brg.with_eltwise, brg.with_scales, brg.with_bias,
                    brg.with_sum, brg.dt_d != brg.dt_c)) {
            Label label_done, label_store_without_post_ops;

            mov(reg_do_post_ops, ptr[rsp + reg_do_post_ops_offs_]);
            cmp(reg_do_post_ops, 0);
            jz(label_store_without_post_ops, T_NEAR);

            store_accumulators_apply_post_ops(m_block, n_block2, is_n_tail);
            jmp(label_done, T_NEAR);

            L(label_store_without_post_ops);
            store_accumulators_without_post_ops(m_block, n_block2, is_n_tail);

            L(label_done);
        } else {
            store_accumulators_without_post_ops(m_block, n_block2, is_n_tail);
        }
    }
}

void jit_brgemm_kernel_base_t::restore_A_B_matrices() {
    if (brg.type != brgemm_offs) {
        mov(reg_aux1_A, reg_A);
        mov(reg_aux1_B, reg_B);
    }
}
void jit_brgemm_kernel_base_t::restore_offsets() {
    if (brg.type == brgemm_offs) {
        mov(reg_offset_A, ptr[rsp + origin_offset_A_offs_]);
        mov(reg_offset_B, ptr[rsp + origin_offset_B_offs_]);
    }
}
void jit_brgemm_kernel_base_t::set_A_B_matrices() {
    if (brg.type == brgemm_addr) {
        mov(reg_aux_A, ptr[reg_aux1_A]);
        mov(reg_aux_B, ptr[reg_aux1_B]);

        add(reg_aux1_A, 8);
        add(reg_aux1_B, 8);
    } else if (brg.type == brgemm_strd) {
        mov(reg_aux_A, reg_aux1_A);
        mov(reg_aux_B, reg_aux1_B);

        add(reg_aux1_A, brg.stride_a);
        add(reg_aux1_B, brg.stride_b);
    } else if (brg.type == brgemm_offs) {
        mov(reg_aux_A, reg_A);
        mov(reg_aux_B, reg_B);

        add(reg_aux_A, ptr[reg_offset_A]);
        add(reg_aux_B, ptr[reg_offset_B]);
        add(reg_offset_A, 8);
        add(reg_offset_B, 8);
    }
    add(reg_aux_A, reg_a_offset);
    add(reg_aux_B, reg_b_offset);
}

void jit_brgemm_kernel_base_t::gemm_microkernel_amx(int m_block2,
        bool is_mb_tail, int n_block2, bool is_k_tail, bool is_n_tail) {
    MAYBE_UNUSED(is_k_tail);
    MAYBE_UNUSED(is_n_tail);
    auto offset_A = [=](int m) {
        return brg.typesize_A * (m * brg.m_block * brg.LDA);
    };
    auto offset_B = [=](int n) {
        return brg.typesize_B * (brg.k_step * n * brg.n_block);
    };
    auto tdpbxxd = [=](const Tmm &x1, const Tmm &x2, const Tmm &x3) {
        if (brg.dt_a == data_type::bf16 && brg.dt_b == data_type::bf16) {
            tdpbf16ps(x1, x2, x3);
        } else if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::u8) {
            tdpbuud(x1, x2, x3);
        } else if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::s8) {
            tdpbusd(x1, x2, x3);
        } else if (brg.dt_a == data_type::s8 && brg.dt_b == data_type::u8) {
            tdpbsud(x1, x2, x3);
        } else if (brg.dt_a == data_type::s8 && brg.dt_b == data_type::s8) {
            tdpbssd(x1, x2, x3);
        } else {
            assert(!"unsupported combination");
        }
    };

    mov(ptr[rsp + reg_mb_loop_offs_], reg_mb_loop);
    mov(ptr[rsp + reg_nb_loop_offs_], reg_nb_loop);

    mov(reg_stride_lda, brg.typesize_A * brg.LDA);
    mov(reg_stride_ldb, brg.k_step * brg.typesize_B * brg.LDB);

    for (int nb = 0; nb < n_block2; nb++) {
        tileloadd(Tmm(brgemm_amx::get_B_tensor(nb)),
                ptr[reg_aux_B + offset_B(nb) + reg_stride_ldb]);
    }
    for (int mb = 0; mb < m_block2; mb++) {
        tileloadd(Tmm(brgemm_amx::get_A_tensor(mb)),
                ptr[reg_aux_A + offset_A(mb) + reg_stride_lda]);
        for (int nb = 0; nb < n_block2; nb++) {
            tdpbxxd(Tmm(brgemm_amx::get_C_tensor(mb, nb)),
                    Tmm(brgemm_amx::get_A_tensor(mb)),
                    Tmm(brgemm_amx::get_B_tensor(nb)));
        }
    }
    mov(reg_mb_loop, ptr[rsp + reg_mb_loop_offs_]);
    mov(reg_nb_loop, ptr[rsp + reg_nb_loop_offs_]);
}

void jit_brgemm_kernel_base_t::gemm_microkernel_avx512(int m_block2,
        bool is_mb_tail, int n_block2, bool is_k_tail, bool is_n_tail) {
    MAYBE_UNUSED(m_block2);
    auto offset_A
            = [=](int m, int k) { return brg.typesize_A * (m * brg.LDA + k); };
    auto offset_B = [=](int n, int k) {
        return brg.typesize_B * (k * brg.LDB + brg.k_step * n * brg.n_block);
    };

    auto dot_product = [=](Zmm z1, Zmm z2, Zmm z3) {
        if (brg.is_f32)
            vfmadd231ps(z1, z2, z3);
        else if (brg.is_bf16)
            vdpbf16ps(z1, z2, z3);
        else if (brg.is_int8)
            vpdpbusd(z1, z3, z2);
    };
    auto broadcast = [=](Zmm z1, size_t offset) {
        if (brg.is_f32)
            vbroadcastss(z1, ptr[reg_aux_A + offset]);
        else if (brg.is_bf16 || brg.is_int8)
            vpbroadcastd(z1, ptr[reg_aux_A + offset]);
    };

    int m_block = (is_mb_tail) ? brg.mb_tail : brg.m_block;
    bool is_emdbd = brg.embd_bcst;

    int k_loop = 0, k_tail_size = 0;
    if (is_k_tail) {
        if (brg.is_bf16 || brg.is_int8) {
            k_tail_size = brg.kb_tail % brg.k_step;
            k_loop = (k_tail_size != 0)
                    ? ((brg.kb_tail / brg.k_step) + 1) * brg.k_step
                    : brg.kb_tail;
        } else
            k_loop = brg.kb_tail;
    } else
        k_loop = brg.k_block;

    for (int k = 0; k < k_loop; k += brg.k_step) {
        int prefetch_count_B = 0;
        for (int n = 0; n < n_block2; n++) {
            if (is_n_tail) {
                vmovups(load(n) | k_tail_mask | T_z,
                        ptr[reg_aux_B + offset_B(n, k)]);
            } else {
                vmovups(load(n), ptr[reg_aux_B + offset_B(n, k)]);
            }
        }
        for (int m = 0; m < m_block; m++) {
            if (!is_emdbd) {
                if (is_k_tail && k_tail_size != 0 && (k == k_loop - brg.k_step)
                        && (brg.is_bf16 || brg.is_int8)) {
                    Xmm xmm_tmp = Xmm(bcst().getIdx());
                    load_bytes(xmm_tmp, reg_aux_A, offset_A(m, k),
                            k_tail_size * brg.typesize_A);
                    vpbroadcastd(bcst(), xmm_tmp);
                } else
                    broadcast(bcst(), offset_A(m, k));
            }
            if (prefetch_count_B < n_block2) {
                prefetcht0(ptr[reg_aux_B + offset_B(prefetch_count_B++, k)
                        + brg.LDB * brg.k_block * brg.typesize_B]);
            }
            for (int n = 0; n < n_block2; n++) {
                auto zmm = accm(n_block2, m, n);
                if (is_emdbd)
                    vfmadd231ps(
                            zmm, load(n), zword_b[reg_aux_A + offset_A(m, k)]);
                else
                    dot_product(zmm, load(n), bcst());
            }
        }
    }
}

void jit_brgemm_kernel_base_t::gemm_microkernel(int m_block2, bool is_mb_tail,
        int n_block2, bool is_k_tail, bool is_n_tail) {
    if (brg.is_int8_amx || brg.is_bf16_amx) {
        gemm_microkernel_amx(
                m_block2, is_mb_tail, n_block2, is_k_tail, is_n_tail);
    } else {
        gemm_microkernel_avx512(
                m_block2, is_mb_tail, n_block2, is_k_tail, is_n_tail);
    }
}

void jit_brgemm_kernel_base_t::nb_loop(int m_block2, bool is_mb_tail,
        int n_block2, int nb_loop_length, bool is_reg_tail, bool is_n_tail) {
    Label nb_loop_label;
    Label kb_loop_label;
    Label N_loop_label;

    if (!is_reg_tail) {
        mov(reg_aux_C, reg_C);
        mov(reg_aux_D, reg_D);
        xor_(reg_b_offset, reg_b_offset);
        if (brg.with_bias) {
            mov(reg_bias, ptr[rsp + reg_bias_offs_]);
            mov(ptr[rsp + reg_aux_bias_offs_], reg_bias);
        }
        if (brg.with_scales) {
            mov(reg_scales, ptr[rsp + reg_scales_offs_]);
            mov(ptr[rsp + reg_aux_scales_offs_], reg_scales);
        }
    }

    mov(reg_nb_loop, nb_loop_length);
    L(nb_loop_label);
    {
        load_accumulators(m_block2, is_mb_tail, n_block2);

        mov(ptr[rsp + reg_D_offs_], reg_D);
        mov(ptr[rsp + reg_aux_D_offs_], reg_aux_D);

        restore_offsets();
        restore_A_B_matrices();

        if (brg.alpha != 0.f) {
            mov(reg_N_loop, reg_N);
            L(N_loop_label);
            {
                set_A_B_matrices();

                if (brg.kb > 0) {
                    mov(reg_kb_loop, brg.kb);
                    L(kb_loop_label);
                    {
                        const bool is_k_tail = false;
                        gemm_microkernel(m_block2, is_mb_tail, n_block2,
                                is_k_tail, is_n_tail);

                        add(reg_aux_A, brg.typesize_A * brg.k_block);
                        add(reg_aux_B, brg.typesize_B * brg.k_block * brg.LDB);

                        dec(reg_kb_loop);
                        cmp(reg_kb_loop, 0);
                    }
                    jg(kb_loop_label, T_NEAR);
                }
                if (brg.kb_tail != 0) {
                    const bool is_k_tail = true;
                    gemm_microkernel(m_block2, is_mb_tail, n_block2, is_k_tail,
                            is_n_tail);
                }

                dec(reg_N_loop);
                cmp(reg_N_loop, 0);
            }
            jg(N_loop_label, T_NEAR);
        }
        mov(reg_D, ptr[rsp + reg_D_offs_]);
        mov(reg_aux_D, ptr[rsp + reg_aux_D_offs_]);

        store_accumulators(m_block2, is_mb_tail, n_block2, is_n_tail);

        if (!is_n_tail) {
            add(reg_aux_C, brg.typesize_C * n_block2 * brg.n_block);
            add(reg_aux_D, brg.typesize_D * n_block2 * brg.n_block);
            add(reg_b_offset,
                    brg.typesize_B * n_block2 * brg.n_block * brg.n_step);
            if (brg.with_bias) {
                mov(reg_aux_bias, ptr[rsp + reg_aux_bias_offs_]);
                add(reg_aux_bias, brg.typesize_bias * n_block2 * brg.n_block);
                mov(ptr[rsp + reg_aux_bias_offs_], reg_aux_bias);
            }
            if (brg.with_scales) {
                mov(reg_aux_scales, ptr[rsp + reg_aux_scales_offs_]);
                add(reg_aux_scales,
                        brg.is_oc_scale * sizeof(float) * n_block2
                                * brg.n_block);
                mov(ptr[rsp + reg_aux_scales_offs_], reg_aux_scales);
            }
        } else {
            add(reg_aux_C, brg.typesize_C * brg.nb_tail);
            add(reg_aux_D, brg.typesize_D * brg.nb_tail);
            add(reg_b_offset, brg.typesize_B * brg.nb_tail * brg.n_step);
            if (brg.with_bias) {
                mov(reg_aux_bias, ptr[rsp + reg_aux_bias_offs_]);
                add(reg_aux_bias, brg.typesize_bias * brg.nb_tail);
                mov(ptr[rsp + reg_aux_bias_offs_], reg_aux_bias);
            }
            if (brg.with_scales) {
                mov(reg_aux_scales, ptr[rsp + reg_aux_scales_offs_]);
                add(reg_aux_scales,
                        brg.is_oc_scale * sizeof(float) * brg.nb_tail);
                mov(ptr[rsp + reg_aux_scales_offs_], reg_aux_scales);
            }
        }

        dec(reg_nb_loop);
        cmp(reg_nb_loop, 0);
    }
    jg(nb_loop_label, T_NEAR);
}

void jit_brgemm_kernel_base_t::mb_loop() {
    auto do_nb_loop = [=](int m_block2, bool is_mb_tail) {
        if (brg.nb2 > 0) {
            const bool is_Nreg_tail = false;
            const bool is_N_tail = false;
            nb_loop(m_block2, is_mb_tail, brg.n_block2, brg.nb2, is_Nreg_tail,
                    is_N_tail);
        }
        if (brg.nb2_tail > 0) {
            const bool is_Nreg_tail = (brg.nb2 == 0) ? false : true;
            const bool is_N_tail = false;
            nb_loop(m_block2, is_mb_tail, brg.nb2_tail, 1, is_Nreg_tail,
                    is_N_tail);
        }
        if (brg.nb_tail > 0) {
            const bool is_Nreg_tail
                    = (brg.nb2 == 0 && brg.nb2_tail == 0) ? false : true;
            const bool is_N_tail = true;
            nb_loop(m_block2, is_mb_tail, 1, 1, is_Nreg_tail, is_N_tail);
        }
    };

    auto mb_loop_body = [=](int m_block2, bool is_mb_tail) {
        do_nb_loop(m_block2, is_mb_tail);

        add(reg_C, brg.typesize_C * m_block2 * brg.m_block * brg.LDC);
        add(reg_D, brg.typesize_D * m_block2 * brg.m_block * brg.LDD);
        add(reg_a_offset, brg.typesize_A * m_block2 * brg.m_block * brg.LDA);
    };
    auto mb_loop_avx512 = [=]() {
        Label mb_loop_label;
        mov(reg_mb_loop, brg.mb);
        L(mb_loop_label);
        {
            mb_loop_body(1, false);

            dec(reg_mb_loop);
            cmp(reg_mb_loop, 0);
        }
        jg(mb_loop_label, T_NEAR);
    };
    auto mb_loop_amx = [=]() {
        Label mb_loop_label;
        if (brg.m_block2 > 1) {
            mov(reg_mb_loop, brg.mb2);
            L(mb_loop_label);
            {
                mb_loop_body(brg.m_block2, false);

                dec(reg_mb_loop);
                cmp(reg_mb_loop, 0);
            }
            jg(mb_loop_label, T_NEAR);
        }
        if (brg.mb2_tail > 0) mb_loop_body(brg.mb2_tail, false);
    };

    xor_(reg_a_offset, reg_a_offset);
    if (brg.is_int8_amx || brg.is_bf16_amx)
        mb_loop_amx();
    else
        mb_loop_avx512();
    if (brg.mb_tail > 0) do_nb_loop(1, true);
}

void jit_brgemm_kernel_base_t::generate() {
    preamble();

    sub(rsp, stack_space_needed_);

    const auto full_mask = size_t {0xffffffffffffffff};
    const auto tail_mask = size_t((1 << brg.nb_tail) - 1);

    reg64_t reg_mask = rax;

    mov(reg_mask, full_mask);
    kmovq(k_full_mask, reg_mask);
    mov(reg_mask, tail_mask);
    kmovq(k_tail_mask, reg_mask);

    read_params();

    if (!brg.embd_bcst && (brg.is_bf16 || brg.is_int8)) {
        Xmm xmm_tmp = Xmm(bcst().getIdx());
        vpxor(xmm_tmp, xmm_tmp, xmm_tmp);
    }

    mb_loop();

    add(rsp, stack_space_needed_);

    postamble();

    if (brg.with_eltwise) eltwise_injector_->prepare_table();
}

brgemm_kernel_t::brgemm_kernel_t(const brgemm_t abrd) {
    brgemm_kernel_ = new jit_brgemm_kernel_base_t(abrd);
}

status_t brgemm_kernel_t::create_kernel() {
    return brgemm_kernel_->create_kernel();
}

void brgemm_kernel_t::operator()(brgemm_kernel_params_t *params) const {
    (*brgemm_kernel_)(params);
}

brgemm_kernel_t::~brgemm_kernel_t() {
    delete brgemm_kernel_;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
