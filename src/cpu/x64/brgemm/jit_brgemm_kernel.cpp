/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include <memory>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/brgemm/brgemm_amx.hpp"
#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

struct jit_brgemm_kernel_base_t : public jit_generator {
    jit_brgemm_kernel_base_t(const brgemm_t &abrg)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_common)
        , brg(abrg)
        , postops_injector_(nullptr)
        , is_ldb_loop(false)
        , with_binary_per_oc_bcast_(brg.with_binary
                  && binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                          brg.attr->post_ops_,
                          memory_desc_wrapper(brg.dst_md))) {

        if (brg.with_eltwise || brg.with_binary || brg.with_sum) {

            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    static_cast<size_t>(Xbyak::Zmm(1).getIdx()), this->rdx,
                    this->r10, preserve_gpr, preserve_vmm,
                    GET_OFF(post_ops_binary_rhs_arg_vec),
                    memory_desc_wrapper(brg.dst_md),
                    static_cast<size_t>(brg.ldb_tail), ld_tail_mask,
                    use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {this->param1, rhs_sp};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<avx512_common>>(
                    this, brg.attr->post_ops_, bsp);
        }
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_base_t)

    brgemm_t brg;

private:
    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_common>>
            postops_injector_;

    using reg64_t = const Xbyak::Reg64;

    // Register decomposition
    const reg64_t param1 = abi_param1;

    const reg64_t reg_C = r15;
    const reg64_t reg_aux_C = r14;

    const reg64_t reg_addr_batch = r13;
    const reg64_t reg_A = r13;
    const reg64_t reg_B = r12;

    const reg64_t reg_aux_A = r11;
    const reg64_t reg_aux_B = r10;
    const reg64_t reg_aux_A_vpad = reg_aux_A;

    const reg64_t reg_bdb_loop = r9;
    const reg64_t reg_ldb_loop = r8;

    const reg64_t reg_stride_lda = reg_bdb_loop;
    const reg64_t reg_stride_ldb = reg_ldb_loop;
    const reg64_t reg_stride_ld_block = reg_ldb_loop;
    const reg64_t reg_s8_input_shift = reg_bdb_loop;

    const reg64_t reg_BS_loop = rax;
    const reg64_t reg_rdb_loop = rbx;
    const reg64_t reg_BS = abi_not_param1;

    const reg64_t reg_a_offset = rdx;
    const reg64_t reg_b_offset = rsi;

    const reg64_t reg_aux1_batch = rbp;
    const reg64_t reg_aux1_A = rbp;
    const reg64_t reg_aux1_B = abi_param1;

    const reg64_t reg_offs_batch = reg_aux1_A;
    const reg64_t reg_strd_batch = reg_rdb_loop;

    const reg64_t reg_bias = reg_rdb_loop;
    const reg64_t reg_scales = reg_rdb_loop;
    const reg64_t reg_aux_bias = reg_rdb_loop;
    const reg64_t reg_binary_postops_oc_l = reg_rdb_loop;
    const reg64_t reg_aux_binary_postops_oc_l = reg_rdb_loop;

    const reg64_t reg_aux_scales = reg_aux_B;
    const reg64_t reg_do_post_ops = reg_rdb_loop;
    const reg64_t reg_tmp_gpr = reg_rdb_loop;
    const reg64_t reg_ptr_sum_scale = reg_rdb_loop;

    const reg64_t reg_buf = reg_rdb_loop;
    const reg64_t reg_compensation = reg_bias;
    const reg64_t reg_aux_compensation = reg_aux_bias;

    const reg64_t reg_D = reg_aux_A;
    const reg64_t reg_aux_D = reg_BS_loop;

    constexpr static int origin_offs_batch_offs_ = 0;
    constexpr static int origin_strd_batch_offs_ = 0;
    constexpr static int reg_bias_offs_ = 8;
    constexpr static int reg_aux_bias_offs_ = 16;
    constexpr static int reg_do_post_ops_offs_ = 24;
    constexpr static int reg_D_offs_ = 32;
    constexpr static int reg_aux_D_offs_ = 40;
    constexpr static int reg_scales_offs_ = 48;
    constexpr static int reg_aux_scales_offs_ = 56;
    constexpr static int reg_bdb_loop_offs_ = 64;
    constexpr static int reg_ldb_loop_offs_ = 72;
    constexpr static int reg_buf_offs_ = 80;
    constexpr static int reg_comp_offs_ = reg_buf_offs_;
    constexpr static int reg_aux_comp_offs_ = 88;
    constexpr static int abi_param1_offs_ = 96;
    constexpr static int reg_binary_postops_oc_l_offs_ = 104;
    constexpr static int reg_aux_binary_postops_oc_l_offs_ = 112;
    constexpr static int stack_space_needed_ = 120;

    bool is_ldb_loop;
    const bool with_binary_per_oc_bcast_;

    Xbyak::Opmask ld_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask ld_tail_mask = Xbyak::Opmask(3);

    Xbyak::Zmm accm(int ld_block, int bd, int ld) {
        return Xbyak::Zmm(31 - (bd * ld_block + ld));
    }

    Xbyak::Zmm bcst(int bd = 0) {
        if (n_bcast_1_load) {
            int idx = 31 - (brg.ld_block2 * brg.bd_block) - bd;
            assert(idx > 0);
            return Xbyak::Zmm(idx);
        } else
            return this->zmm0;
    }

    Xbyak::Zmm load(int ld = 0) {
        if (n_bcast_1_load) {
            return this->zmm0;
        } else {
            int idx = 31 - (brg.ld_block2 * brg.bd_block) - ld;
            assert(idx > 0);
            return Xbyak::Zmm(idx);
        }
    }

    const Xbyak::Zmm &zmm_tmp_1() const noexcept { return this->zmm0; }
    const Xbyak::Zmm &zmm_tmp_2() const noexcept { return this->zmm1; }
    const Xbyak::Zmm &zmm_tmp_3() const noexcept { return this->zmm2; }
    const Xbyak::Zmm &zmm_inp_shift() const noexcept { return this->zmm1; }

    Xbyak::Zmm zmm_mask(const Xbyak::Zmm zmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) const;
    Xbyak::Ymm ymm_mask(const Xbyak::Ymm ymm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) const;

    void cvt2ps(data_type_t type_in, const Xbyak::Zmm zmm_in,
            const Xbyak::Operand &op, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);

    void read_params();
    void load_accumulators(
            int bd_block2, bool is_bdb_tail, int ld_block, bool is_ld_tail);

    void store_accumulators(
            int bd_block2, bool is_bdb_tail, int ld_block, bool is_ld_tail);
    void store_accumulators_without_post_ops(
            int bd_block, int ld_block, bool is_ld_tail);
    void store_accumulators_apply_post_ops(
            int bd_block, int ld_block, bool is_ld_tail);
    void apply_alpha_beta(int bd_block, int ld_block, bool is_ld_tail);
    void apply_post_ops(int bd_block, int ld_block2, bool is_ld_tail);
    void restore_A_B_matrices();
    void set_A_B_matrices();

    void gemm_microkernel_avx512(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_rd_tail, bool is_ld_tail, int vpad, int rows_for_rd_tail);
    void gemm_microkernel_amx(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_rd_tail, bool is_ldb_tail);
    void gemm_microkernel(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_rd_tail, bool is_ldb_tail, int vpad, int rows_for_rd_tail);

    void ldb_loop(int bd_block2, bool is_bdb_tail, int ld_block,
            int ldb_loop_length, bool is_reg_tail, bool is_ld_tail,
            bool check_top_vpad, bool check_bottom_vpad, int rows_for_rd_tail);
    void bdb_loop();

    void generate() override;

    int A_offset(int bd, int rd, bool is_amx = false) const noexcept;
    int B_offset(int ld, int rd, bool is_amx = false) const noexcept;
    int C_offset(int bd, int ld) const noexcept;
    int D_offset(int bd, int ld) const noexcept;

    int rdb_A_offset() const noexcept;
    int rdb_B_offset() const noexcept;

    int ldb_B_offset(int ld_block2, bool is_tail = false) const noexcept;
    int ldb_C_offset(int ld_block2, bool is_tail = false) const noexcept;
    int ldb_D_offset(int ld_block2, bool is_tail = false) const noexcept;

    int bdb_A_offset(int bd_block2) const noexcept;
    int bdb_C_offset(int bd_block2) const noexcept;
    int bdb_D_offset(int bd_block2) const noexcept;

    int bias_offset(int ld, bool is_tail = false) const noexcept;
    int oc_logical_offset(int ld, bool is_tail = false) const noexcept;

    int compensations_offset(int ld, bool is_tail = false) const noexcept;
    int scales_offset(int ld, bool is_tail = false) const noexcept;

    bool n_bcast_1_load = false;
    bool vpad_exist = false;
};

int jit_brgemm_kernel_base_t::A_offset(int bd, int rd, bool is_amx) const
        noexcept {
    return (is_amx) ? brg.typesize_A * (bd * brg.bd_block * brg.LDA)
                    : brg.typesize_A * (bd * brg.LDA + rd);
}
int jit_brgemm_kernel_base_t::B_offset(int ld, int rd, bool is_amx) const
        noexcept {
    return (is_amx)
            ? brg.typesize_B * (brg.rd_step * ld * brg.ld_block)
            : brg.typesize_B * (rd * brg.LDB + brg.rd_step * ld * brg.ld_block);
}
int jit_brgemm_kernel_base_t::C_offset(int bd, int ld) const noexcept {
    return brg.typesize_C * (bd * brg.LDC + ld * brg.ld_block);
}
int jit_brgemm_kernel_base_t::D_offset(int bd, int ld) const noexcept {
    return brg.typesize_D * (bd * brg.LDD + ld * brg.ld_block);
}

int jit_brgemm_kernel_base_t::rdb_A_offset() const noexcept {
    return brg.typesize_A * brg.rd_block;
}
int jit_brgemm_kernel_base_t::rdb_B_offset() const noexcept {
    return brg.typesize_B * brg.rd_block * brg.LDB;
}

int jit_brgemm_kernel_base_t::ldb_B_offset(int ld_block2, bool is_tail) const
        noexcept {
    return (is_tail) ? brg.typesize_B * brg.ldb_tail * brg.ld_step
                     : brg.typesize_B * ld_block2 * brg.ld_block * brg.ld_step;
}
int jit_brgemm_kernel_base_t::ldb_C_offset(int ld_block2, bool is_tail) const
        noexcept {
    return (is_tail) ? brg.typesize_C * brg.ldb_tail
                     : brg.typesize_C * ld_block2 * brg.ld_block;
}
int jit_brgemm_kernel_base_t::ldb_D_offset(int ld_block2, bool is_tail) const
        noexcept {
    return (is_tail) ? brg.typesize_D * brg.ldb_tail
                     : brg.typesize_D * ld_block2 * brg.ld_block;
}

int jit_brgemm_kernel_base_t::bdb_A_offset(int bd_block2) const noexcept {
    return brg.typesize_A * bd_block2 * brg.bd_block * brg.LDA;
}
int jit_brgemm_kernel_base_t::bdb_C_offset(int bd_block2) const noexcept {
    return brg.typesize_C * bd_block2 * brg.bd_block * brg.LDC;
}
int jit_brgemm_kernel_base_t::bdb_D_offset(int bd_block2) const noexcept {
    return brg.typesize_D * bd_block2 * brg.bd_block * brg.LDD;
}

int jit_brgemm_kernel_base_t::bias_offset(int ld, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_bias * brg.ldb_tail
                     : brg.typesize_bias * ld * brg.ld_block;
}

int jit_brgemm_kernel_base_t::oc_logical_offset(int ld, bool is_tail) const
        noexcept {
    return (is_tail) ? brg.ldb_tail : ld * brg.ld_block;
}

int jit_brgemm_kernel_base_t::compensations_offset(int ld, bool is_tail) const
        noexcept {
    return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                     : sizeof(int32_t) * ld * brg.ld_block;
}

int jit_brgemm_kernel_base_t::scales_offset(int ld, bool is_tail) const
        noexcept {
    return (is_tail) ? brg.is_oc_scale * sizeof(float) * brg.ldb_tail
                     : brg.is_oc_scale * sizeof(float) * ld * brg.ld_block;
}
Xbyak::Zmm jit_brgemm_kernel_base_t::zmm_mask(const Xbyak::Zmm zmm_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) const {
    return mask_flag ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                     : zmm_in;
}

Xbyak::Ymm jit_brgemm_kernel_base_t::ymm_mask(const Xbyak::Ymm ymm_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) const {
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

    if (brg.with_binary) mov(ptr[rsp + abi_param1_offs_], param1);

    if (brg.type == brgemm_addr) {
        mov(reg_addr_batch, ptr[param1 + GET_OFF(batch)]);
    } else {
        if (brg.layout == brgemm_row_major) {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_A)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_B)]);
        } else {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_B)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_A)]);
        }

        if (brg.type == brgemm_offs) {
            mov(reg_offs_batch, ptr[param1 + GET_OFF(batch)]);
            mov(ptr[rsp + origin_offs_batch_offs_], reg_offs_batch);
        } else {
            mov(reg_strd_batch, ptr[param1 + GET_OFF(batch)]);
            mov(ptr[rsp + origin_strd_batch_offs_], reg_strd_batch);
        }
    }

    mov(reg_C, ptr[param1 + GET_OFF(ptr_C)]);
    mov(reg_D, ptr[param1 + GET_OFF(ptr_D)]);
    mov(reg_BS, ptr[param1 + GET_OFF(BS)]);

    // ptr_buf is re-used for passing compensations for
    // brg.req_s8s8_compensation case
    if (brg.is_amx || brg.req_s8s8_compensation) {
        mov(reg_buf, ptr[param1 + GET_OFF(ptr_buf)]);
        mov(ptr[rsp + reg_buf_offs_], reg_buf);
    }

    if (brg.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
        mov(ptr[rsp + reg_bias_offs_], reg_bias);
    }
    if (brg.with_scales) {
        mov(reg_scales, ptr[param1 + GET_OFF(ptr_scales)]);
        mov(ptr[rsp + reg_scales_offs_], reg_scales);
    }
    if (with_binary_per_oc_bcast_) {
        mov(reg_binary_postops_oc_l, ptr[param1 + GET_OFF(oc_logical_off)]);
        mov(ptr[rsp + reg_binary_postops_oc_l_offs_], reg_binary_postops_oc_l);
    }

    mov(reg_do_post_ops, ptr[param1 + GET_OFF(do_post_ops)]);
    mov(ptr[rsp + reg_do_post_ops_offs_], reg_do_post_ops);
}

void jit_brgemm_kernel_base_t::load_accumulators(
        int bd_block2, bool is_bdb_tail, int ld_block2, bool is_ld_tail) {
    if (brg.is_amx) {
        for_(int bdb = 0; bdb < bd_block2; bdb++)
        for (int ldb = 0; ldb < ld_block2; ldb++) {
            int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
            tilezero(Tmm(brgemm_amx::get_C_tensor(bdb, idx)));
        }
    } else {
        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
        for_(int bd = 0; bd < bd_block; bd++)
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            vxorps(zmm, zmm, zmm);
        }
    }
}

void jit_brgemm_kernel_base_t::apply_alpha_beta(
        int bd_block, int ld_block2, bool is_ld_tail) {
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
    auto zmm_beta = zmm_tmp_1();
    auto zmm_alpha = zmm_tmp_2();
    auto zmm_prev_dst = zmm_tmp_3();

    const bool apply_alpha = brg.alpha != 1.f;
    const bool apply_beta = brg.beta != 0.f;
    if (!apply_alpha && !apply_beta) return;

    const bool dq2ps_required = brg.is_int8 && (apply_alpha || brg.beta != 1.f);
    const bool use_vadd_for_beta = brg.beta == 1.f && !dq2ps_required;

    if (apply_beta && !use_vadd_for_beta) {
        mov(reg_tmp_gpr, float2int((float)brg.beta));
        movq(Xmm(zmm_beta.getIdx()), reg_tmp_gpr);
        vbroadcastss(zmm_beta, Xmm(zmm_beta.getIdx()));
    }
    if (apply_alpha) {
        mov(reg_tmp_gpr, float2int((float)brg.alpha));
        movq(Xmm(zmm_alpha.getIdx()), reg_tmp_gpr);
        vbroadcastss(zmm_alpha, Xmm(zmm_alpha.getIdx()));
    }
    for_(int bd = 0; bd < bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        auto zmm = accm(ld_block2, bd, ld);
        if (dq2ps_required) vcvtdq2ps(zmm, zmm);
        if (apply_alpha) vmulps(zmm, zmm, zmm_alpha);
        if (apply_beta) {
            auto ptr_C = ptr[reg_aux_C + C_offset(bd, ld)];
            if (use_vadd_for_beta) {
                auto zmm_masked = zmm | k_mask | T_z;
                if (brg.is_int8)
                    vpaddd(zmm_masked, zmm, ptr_C);
                else
                    vaddps(zmm_masked, zmm, ptr_C);
            } else {
                cvt2ps(brg.dt_c, zmm_prev_dst, ptr_C, true, false, k_mask);
                vfmadd231ps(zmm, zmm_prev_dst, zmm_beta);
            }
        }
    }
}

void jit_brgemm_kernel_base_t::apply_post_ops(
        int bd_block, int ld_block2, bool is_ld_tail) {

    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

    const injector_utils::conditional_register_preserve_guard_t register_guard(
            brg.with_binary, this, {param1});
    const auto guard_space = register_guard.stack_space_occupied();
    if (brg.with_binary) {
        mov(param1, ptr[rsp + abi_param1_offs_ + guard_space]);

        if (with_binary_per_oc_bcast_) {
            mov(reg_aux_binary_postops_oc_l,
                    ptr[rsp + reg_aux_binary_postops_oc_l_offs_ + guard_space]);

            for_(int bd = 0; bd < bd_block; bd++)
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto zmm_idx = accm(ld_block2, bd, ld).getIdx();

                rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                        zmm_idx, reg_aux_binary_postops_oc_l);
                rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                        zmm_idx, oc_logical_offset(ld));
                if (is_ld_tail) rhs_arg_params.vmm_tail_idx_.emplace(zmm_idx);
            }
        }
    }
    const auto sum_injector = [&] {
        const float *p_sum_scale = &brg.sum_scale;
        const bool p_sum_scale_reg_set = *p_sum_scale != 1.f;

        const injector_utils::conditional_register_preserve_guard_t
                register_guard(with_binary_per_oc_bcast_ && p_sum_scale_reg_set,
                        this, {reg_ptr_sum_scale});

        if (p_sum_scale_reg_set) mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

        const auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

        for (int bd = 0; bd < bd_block; bd++) {
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto zmm = accm(ld_block2, bd, ld);
                const auto addr = ptr[reg_aux_D + D_offset(bd, ld)];
                const auto zmm_prev_dst = Xbyak::Zmm(0);
                cvt2ps(brg.dt_d, zmm_prev_dst, addr, true, false, k_mask);
                if (!p_sum_scale_reg_set)
                    vaddps(zmm, zmm_prev_dst);
                else
                    vfmadd231ps(zmm, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
            }
        }
    };

    if (brg.with_sum) {
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }

    postops_injector_->compute_vector_range(
            32 - bd_block * ld_block2, 32, rhs_arg_params);
}

void jit_brgemm_kernel_base_t::store_accumulators_apply_post_ops(
        int bd_block, int ld_block2, bool is_ld_tail) {
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are already converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dq2ps_required = brg.is_int8
            && IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);

    if (brg.with_bias) { mov(reg_aux_bias, ptr[rsp + reg_aux_bias_offs_]); }
    for_(int bd = 0; bd < bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        auto zmm = accm(ld_block2, bd, ld);
        if (dq2ps_required) vcvtdq2ps(zmm, zmm);
        if (brg.with_bias) {
            auto zmm_bias = zmm_tmp_1();
            auto ptr_bias = ptr[reg_aux_bias + bias_offset(ld)];
            cvt2ps(brg.dt_bias, zmm_bias, ptr_bias, true, false, k_mask);
            vaddps(zmm, zmm, zmm_bias);
        }
    }

    if (brg.req_s8s8_compensation) {
        mov(reg_aux_compensation, ptr[rsp + reg_aux_comp_offs_]);
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm_comp = zmm_tmp_1();
            int comp_offset = compensations_offset(ld);
            auto comp_addr
                    = EVEX_compress_addr(reg_aux_compensation, comp_offset);
            cvt2ps(data_type::s32, zmm_comp, comp_addr, true, false, k_mask);

            for (int bd = 0; bd < bd_block; bd++) {
                auto zmm = accm(ld_block2, bd, ld);
                vaddps(zmm, zmm, zmm_comp);
            }
        }
    }
    if (brg.with_scales) {
        mov(reg_aux_scales, ptr[rsp + reg_aux_scales_offs_]);
        for (int bd = 0; bd < bd_block; bd++) {
            for (int ld = 0; ld < ld_block2; ld++) {
                const Xbyak::Zmm zmm = zmm_mask(
                        accm(ld_block2, bd, ld), true, false, k_mask);
                vmulps(zmm, zmm, ptr[reg_aux_scales + scales_offset(ld)]);
            }
        }
    }

    if (postops_injector_) apply_post_ops(bd_block, ld_block2, is_ld_tail);

    const bool dt_requires_saturation
            = one_of(brg.dt_d, data_type::u8, data_type::s8, data_type::s32);
    auto zmm_lbound = zmm_tmp_1();
    auto zmm_ubound = zmm_tmp_2();
    if (dt_requires_saturation) {
        init_saturate_f32(
                zmm_lbound, zmm_ubound, reg_tmp_gpr, data_type::f32, brg.dt_d);
    }

    for (int bd = 0; bd < bd_block; bd++) {
        if (dt_requires_saturation) {
            for (int ld = 0; ld < ld_block2; ld++) {
                auto zmm = accm(ld_block2, bd, ld);
                saturate_f32(zmm, zmm_lbound, zmm_ubound, brg.dt_d);
                vcvtps2dq(zmm, zmm);
            }
        }
        for (int ld = 0; ld < ld_block2; ld++) {
            auto addr = ptr[reg_aux_D + D_offset(bd, ld)];
            auto zmm = accm(ld_block2, bd, ld);
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
        int bd_block, int ld_block2, bool is_ld_tail) {

    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dt_requires_saturation = brg.is_int8
            && !IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);
    auto zmm_lbound = zmm_tmp_1();
    auto zmm_ubound = zmm_tmp_2();
    if (dt_requires_saturation) {
        init_saturate_f32(
                zmm_lbound, zmm_ubound, reg_tmp_gpr, data_type::f32, brg.dt_d);
    }

    for (int bd = 0; bd < bd_block; bd++) {
        if (dt_requires_saturation) {
            for (int ld = 0; ld < ld_block2; ld++) {
                auto zmm = accm(ld_block2, bd, ld);
                saturate_f32(zmm, zmm_lbound, zmm_ubound, brg.dt_d);
                vcvtps2dq(zmm, zmm);
            }
        }
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            if (is_ld_tail)
                vmovups(ptr[reg_aux_C + C_offset(bd, ld)] | ld_tail_mask | T_z,
                        zmm);
            else
                vmovups(ptr[reg_aux_C + C_offset(bd, ld)], zmm);
        }
    }
}

void jit_brgemm_kernel_base_t::store_accumulators(
        int bd_block2, bool is_bdb_tail, int ld_block2, bool is_ld_tail) {
    const bool are_post_ops_applicable = one_of(true, brg.with_eltwise,
            brg.with_binary, brg.with_scales, brg.with_bias, brg.with_sum,
            brg.dt_d != brg.dt_c, brg.req_s8s8_compensation);
    const bool need_to_apply_alpha_beta = brg.beta != 0.f || brg.alpha != 1.f;

    if (brg.is_amx) {
        if (need_to_apply_alpha_beta || are_post_ops_applicable)
            mov(reg_stride_ld_block, brg.ld_block * brg.typesize_C);
        else
            mov(reg_stride_ld_block, brg.LDC * brg.typesize_C);

        auto store_accumulators_amx = [=](const bool apply_post_ops) {
            mov(reg_buf, ptr[rsp + reg_buf_offs_]);
            for (int bdb = 0; bdb < bd_block2; bdb++) {
                for (int ldb = 0; ldb < ld_block2; ldb++) {
                    int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
                    if (need_to_apply_alpha_beta || are_post_ops_applicable) {
                        tilestored(ptr[reg_buf + reg_stride_ld_block],
                                Tmm(brgemm_amx::get_C_tensor(bdb, idx)));
                        for (int bd = 0; bd < brg.bd_block; bd++) {
                            size_t buf_offset
                                    = (bd * brg.ld_block) * brg.typesize_C;
                            auto vreg_acc = is_ld_tail
                                    ? accm(1, bd, 0) | ld_tail_mask | T_z
                                    : accm(1, bd, 0);
                            vmovups(vreg_acc, ptr[reg_buf + buf_offset]);
                        }
                        if (need_to_apply_alpha_beta)
                            apply_alpha_beta(brg.bd_block, 1, is_ld_tail);

                        if (apply_post_ops) {
                            store_accumulators_apply_post_ops(
                                    brg.bd_block, 1, is_ld_tail);
                            if (ldb < ld_block2 - 1) {
                                if (brg.with_bias) {
                                    mov(reg_aux_bias,
                                            ptr[rsp + reg_aux_bias_offs_]);
                                    add(reg_aux_bias, bias_offset(1));
                                    mov(ptr[rsp + reg_aux_bias_offs_],
                                            reg_aux_bias);
                                }
                                if (brg.with_scales) {
                                    mov(reg_aux_scales,
                                            ptr[rsp + reg_aux_scales_offs_]);
                                    add(reg_aux_scales, scales_offset(1));
                                    mov(ptr[rsp + reg_aux_scales_offs_],
                                            reg_aux_scales);
                                }
                                if (with_binary_per_oc_bcast_) {
                                    mov(reg_aux_binary_postops_oc_l,
                                            ptr[rsp + reg_aux_binary_postops_oc_l_offs_]);
                                    add(reg_aux_binary_postops_oc_l,
                                            oc_logical_offset(1));
                                    mov(ptr[rsp + reg_aux_binary_postops_oc_l_offs_],
                                            reg_aux_binary_postops_oc_l);
                                }
                            }
                            mov(reg_buf, ptr[rsp + reg_buf_offs_]);
                            add(reg_aux_D, ldb_D_offset(1));
                        } else {
                            store_accumulators_without_post_ops(
                                    brg.bd_block, 1, is_ld_tail);
                        }
                    } else {
                        tilestored(ptr[reg_aux_C + reg_stride_ld_block],
                                Tmm(brgemm_amx::get_C_tensor(bdb, idx)));
                    }
                    add(reg_aux_C, ldb_C_offset(1));
                }
                sub(reg_aux_C, ldb_C_offset(ld_block2));
                add(reg_aux_C, bdb_C_offset(1));
                if (apply_post_ops) {
                    sub(reg_aux_D, ldb_D_offset(ld_block2));
                    add(reg_aux_D, bdb_D_offset(1));

                    if (ld_block2 > 1) {
                        bool post_processed = false;
                        if (brg.with_bias) {
                            post_processed = true;
                            mov(reg_aux_bias, ptr[rsp + reg_aux_bias_offs_]);
                            sub(reg_aux_bias, bias_offset(ld_block2 - 1));
                            mov(ptr[rsp + reg_aux_bias_offs_], reg_aux_bias);
                        }
                        if (brg.with_scales) {
                            post_processed = true;
                            mov(reg_aux_scales,
                                    ptr[rsp + reg_aux_scales_offs_]);
                            sub(reg_aux_scales, scales_offset(ld_block2 - 1));
                            mov(ptr[rsp + reg_aux_scales_offs_],
                                    reg_aux_scales);
                        }
                        if (with_binary_per_oc_bcast_) {
                            post_processed = true;
                            mov(reg_aux_binary_postops_oc_l,
                                    ptr[rsp + reg_aux_binary_postops_oc_l_offs_]);
                            sub(reg_aux_binary_postops_oc_l,
                                    oc_logical_offset(ld_block2 - 1));
                            mov(ptr[rsp + reg_aux_binary_postops_oc_l_offs_],
                                    reg_aux_binary_postops_oc_l);
                        }
                        if (post_processed)
                            mov(reg_buf, ptr[rsp + reg_buf_offs_]);
                    }
                }
            }
            sub(reg_aux_C, bdb_C_offset(bd_block2));
            if (apply_post_ops) sub(reg_aux_D, bdb_D_offset(bd_block2));
        };

        Label label_done;
        if (are_post_ops_applicable) {
            Label label_store_without_post_ops;
            mov(reg_do_post_ops, ptr[rsp + reg_do_post_ops_offs_]);
            cmp(reg_do_post_ops, 0);
            jz(label_store_without_post_ops, T_NEAR);

            store_accumulators_amx(true);
            jmp(label_done, T_NEAR);

            L_aligned(label_store_without_post_ops);
        }
        store_accumulators_amx(false);
        L_aligned(label_done);
    } else {
        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
        if (need_to_apply_alpha_beta)
            apply_alpha_beta(bd_block, ld_block2, is_ld_tail);

        Label label_done;
        if (are_post_ops_applicable) {
            Label label_store_without_post_ops;

            mov(reg_do_post_ops, ptr[rsp + reg_do_post_ops_offs_]);
            cmp(reg_do_post_ops, 0);
            jz(label_store_without_post_ops, T_NEAR);

            store_accumulators_apply_post_ops(bd_block, ld_block2, is_ld_tail);
            jmp(label_done, T_NEAR);

            L_aligned(label_store_without_post_ops);
        }
        store_accumulators_without_post_ops(bd_block, ld_block2, is_ld_tail);
        L_aligned(label_done);
    }
}

void jit_brgemm_kernel_base_t::restore_A_B_matrices() {
    auto restore_reg_batch = brg.brgattr.max_bs > 1 || vpad_exist;
    if (brg.type == brgemm_addr) {
        if (restore_reg_batch) mov(reg_aux1_batch, reg_addr_batch);
    } else {
        mov(reg_aux1_A, reg_A);
        mov(reg_aux1_B, reg_B);

        if (restore_reg_batch) {
            if (brg.type == brgemm_offs)
                mov(reg_offs_batch, ptr[rsp + origin_offs_batch_offs_]);
            else
                mov(reg_strd_batch, ptr[rsp + origin_strd_batch_offs_]);
        }
    }
}

void jit_brgemm_kernel_base_t::set_A_B_matrices() {
    if (brg.type == brgemm_addr) {
        if (brg.brgattr.max_bs > 1) {
            if (brg.layout == brgemm_row_major) {
                mov(reg_aux_A,
                        ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.A)]);
                mov(reg_aux_B,
                        ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.B)]);
            } else {
                mov(reg_aux_A,
                        ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.B)]);
                mov(reg_aux_B,
                        ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.A)]);
            }
        } else {
            // for max_batch == 1 we stored A and B pointers at the beginning
            // of kernel in reg_aux1_A and reg_aux1_B
            if (brg.layout == brgemm_row_major) {
                mov(reg_aux_A, reg_aux1_A);
                mov(reg_aux_B, reg_aux1_B);
            } else {
                mov(reg_aux_A, reg_aux1_B);
                mov(reg_aux_B, reg_aux1_A);
            }
        }

        if (brg.brgattr.max_bs > 1) {
            add(reg_aux1_batch, sizeof(brgemm_batch_element_t));
            prefetcht0(ptr[reg_aux1_batch]);
        }
    } else if (brg.type == brgemm_offs) {
        mov(reg_aux_A, reg_A);
        mov(reg_aux_B, reg_B);

        add(reg_aux_A, ptr[reg_offs_batch + GET_OFF_BATCH_ELEMENT(offset.A)]);
        add(reg_aux_B, ptr[reg_offs_batch + GET_OFF_BATCH_ELEMENT(offset.B)]);
        add(reg_offs_batch, sizeof(brgemm_batch_element_t));
    } else if (brg.type == brgemm_strd) {
        mov(reg_aux_A, reg_aux1_A);
        mov(reg_aux_B, reg_aux1_B);

        add(reg_aux1_A, brg.stride_a);
        add(reg_aux1_B, brg.stride_b);
        if (vpad_exist) {
            mov(reg_strd_batch, ptr[rsp + origin_strd_batch_offs_]);
            add(reg_strd_batch, sizeof(brgemm_batch_element_t));
            mov(ptr[rsp + origin_strd_batch_offs_], reg_strd_batch);
        }
    }

    add(reg_aux_A, reg_a_offset);
    add(reg_aux_B, reg_b_offset);
}

void jit_brgemm_kernel_base_t::gemm_microkernel_amx(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_rd_tail, bool is_ld_tail) {
    MAYBE_UNUSED(is_rd_tail);
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

    auto maybe_tileloadd_nt = [=](const Tmm &t1, int offset) {
        if (brg.brgattr.hint_expected_A_size != LLONG_MAX
                && brg.brgattr.hint_expected_B_size != LLONG_MAX
                && brg.brgattr.hint_expected_C_size != LLONG_MAX) {
            if (static_cast<size_t>(
                        brg.typesize_A * brg.brgattr.hint_expected_A_size
                        + brg.typesize_B * brg.brgattr.hint_expected_B_size
                        + brg.typesize_C * brg.brgattr.hint_expected_C_size)
                    >= platform::get_per_core_cache_size(1))
                tileloaddt1(t1, ptr[reg_aux_B + offset + reg_stride_ldb]);
            else
                tileloadd(t1, ptr[reg_aux_B + offset + reg_stride_ldb]);
        } else
            tileloaddt1(t1, ptr[reg_aux_B + offset + reg_stride_ldb]);
    };

    for (int bdb = 0; bdb < bd_block2; bdb++) {
        tileloadd(Tmm(brgemm_amx::get_A_tensor(bdb)),
                ptr[reg_aux_A + A_offset(bdb, 0, true) + reg_stride_lda]);
    }
    for (int ldb = 0; ldb < ld_block2; ldb++) {
        const int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
        maybe_tileloadd_nt(
                Tmm(brgemm_amx::get_B_tensor(idx)), B_offset(ldb, 0, true));
        for (int bdb = 0; bdb < bd_block2; bdb++) {
            tdpbxxd(Tmm(brgemm_amx::get_C_tensor(bdb, idx)),
                    Tmm(brgemm_amx::get_A_tensor(bdb)),
                    Tmm(brgemm_amx::get_B_tensor(idx)));
        }
    }
}

void jit_brgemm_kernel_base_t::gemm_microkernel_avx512(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_rd_tail, bool is_ld_tail,
        int vpad, int rows_for_rd_tail) {
    MAYBE_UNUSED(bd_block2);
    auto dot_product = [=](Zmm z1, Zmm z2, Zmm z3) {
        if (brg.is_f32)
            vfmadd231ps(z1, z2, z3);
        else if (brg.is_bf16)
            vdpbf16ps(z1, z2, z3);
        else if (brg.is_int8)
            vpdpbusd(z1, z3, z2);
    };

    int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
    const auto bd_b = nstl::max(0, vpad);
    const auto bd_e = nstl::min(bd_block, bd_block + vpad);
    if (bd_b >= bd_e) return;

    bool is_emdbd = brg.embd_bcst;

    int rd_loop = 0, rd_tail_size = 0;
    if (is_rd_tail) {
        if (brg.is_bf16 || brg.is_int8) {
            rd_tail_size = brg.rdb_tail % brg.rd_step;
            rd_loop = (rd_tail_size != 0)
                    ? ((brg.rdb_tail / brg.rd_step) + 1) * brg.rd_step
                    : brg.rdb_tail;
        } else
            rd_loop = brg.rdb_tail;
    } else
        rd_loop = brg.rd_block;

    auto broadcast = [=](Zmm z1, size_t offset, bool is_tail) {
        if (is_tail) {
            vpxord(z1, z1, z1);
            Xmm xmm_tmp = Xmm(z1.getIdx());
            load_bytes(
                    xmm_tmp, reg_aux_A, offset, rd_tail_size * brg.typesize_A);
            vpbroadcastd(z1, xmm_tmp);
        } else {
            if (brg.is_f32)
                vbroadcastss(z1, ptr[reg_aux_A + offset]);
            else if (brg.is_bf16 || brg.is_int8)
                vpbroadcastd(z1, ptr[reg_aux_A + offset]);
        }

        if (brg.req_s8s8_compensation) vpaddb(z1, z1, zmm_inp_shift());
    };

    bool maybe_load_bytes = (rows_for_rd_tail > 0 || brg.brgattr.wary_tail_read)
            && is_rd_tail && rd_tail_size != 0 && (brg.is_bf16 || brg.is_int8);
    if (n_bcast_1_load) {
        for (int rd = 0; rd < rd_loop; rd += brg.rd_step) {
            bool have_to_load_bytes
                    = maybe_load_bytes && (rd == rd_loop - brg.rd_step);

            auto rows_by_load_bytes = have_to_load_bytes ? rows_for_rd_tail : 0;
            for (int bd = bd_b; bd < bd_e && !is_emdbd; bd++) {
                const auto bd_by_load_bytes = (bd >= bd_e - rows_by_load_bytes
                        || brg.brgattr.wary_tail_read);
                broadcast(bcst(bd), A_offset(bd, rd),
                        have_to_load_bytes && bd_by_load_bytes);
            }
            for (int ld = 0; ld < ld_block2; ld++) {
                if (is_ld_tail) {
                    vmovups(load() | ld_tail_mask | T_z,
                            ptr[reg_aux_B + B_offset(ld, rd)]);
                } else {
                    vmovups(load(), ptr[reg_aux_B + B_offset(ld, rd)]);
                }
                for (int bd = bd_b; bd < bd_e; bd++) {
                    auto zmm = accm(ld_block2, bd, ld);
                    if (is_emdbd)
                        vfmadd231ps(zmm, load(),
                                zword_b[reg_aux_A + A_offset(bd, rd)]);
                    else
                        dot_product(zmm, load(), bcst(bd));
                }
            }
        }
    } else {
        for (int rd = 0; rd < rd_loop; rd += brg.rd_step) {
            int prefetch_count_B = 0;
            for (int ld = 0; ld < ld_block2; ld++) {
                if (is_ld_tail) {
                    vmovups(load(ld) | ld_tail_mask | T_z,
                            ptr[reg_aux_B + B_offset(ld, rd)]);
                } else {
                    vmovups(load(ld), ptr[reg_aux_B + B_offset(ld, rd)]);
                }
            }

            bool have_to_load_bytes
                    = maybe_load_bytes && (rd == rd_loop - brg.rd_step);

            auto rows_by_load_bytes = have_to_load_bytes ? rows_for_rd_tail : 0;
            for (int bd = bd_b; bd < bd_e; bd++) {
                if (!is_emdbd) {
                    const auto bd_by_load_bytes
                            = (bd >= bd_e - rows_by_load_bytes
                                    || brg.brgattr.wary_tail_read);
                    broadcast(bcst(), A_offset(bd, rd),
                            have_to_load_bytes && bd_by_load_bytes);
                }
                if (prefetch_count_B < ld_block2) {
                    prefetcht0(ptr[reg_aux_B + B_offset(prefetch_count_B++, rd)
                            + brg.LDB * brg.rd_block * brg.typesize_B]);
                }
                for (int ld = 0; ld < ld_block2; ld++) {
                    auto zmm = accm(ld_block2, bd, ld);
                    if (is_emdbd)
                        vfmadd231ps(zmm, load(ld),
                                zword_b[reg_aux_A + A_offset(bd, rd)]);
                    else
                        dot_product(zmm, load(ld), bcst());
                }
            }
        }
    }
}

void jit_brgemm_kernel_base_t::gemm_microkernel(int bd_block2, bool is_bdb_tail,
        int ld_block2, bool is_rd_tail, bool is_ld_tail, int vpad,
        int rows_for_rd_tail) {
    if (brg.is_amx) {
        gemm_microkernel_amx(
                bd_block2, is_bdb_tail, ld_block2, is_rd_tail, is_ld_tail);
    } else {
        gemm_microkernel_avx512(bd_block2, is_bdb_tail, ld_block2, is_rd_tail,
                is_ld_tail, vpad, rows_for_rd_tail);
    }
}

void jit_brgemm_kernel_base_t::ldb_loop(int bd_block2, bool is_bdb_tail,
        int ld_block2, int ldb_loop_length, bool is_reg_tail, bool is_ld_tail,
        bool check_top_vpad, bool check_bottom_vpad, int rows_for_rd_tail) {

    auto ldb_shift = [&](int ld_block2, bool is_tail = false) {
        int C_offset
                = (is_tail) ? ldb_C_offset(1, true) : ldb_C_offset(ld_block2);
        int D_offset
                = (is_tail) ? ldb_D_offset(1, true) : ldb_D_offset(ld_block2);
        add(reg_aux_C, C_offset);
        add(reg_aux_D, D_offset);

        add(reg_b_offset,
                (is_tail) ? ldb_B_offset(1, true) : ldb_B_offset(ld_block2));

        if (brg.with_bias) {
            mov(reg_aux_bias, ptr[rsp + reg_aux_bias_offs_]);
            add(reg_aux_bias,
                    (is_tail) ? bias_offset(1, true) : bias_offset(ld_block2));
            mov(ptr[rsp + reg_aux_bias_offs_], reg_aux_bias);
        }
        if (brg.req_s8s8_compensation) {
            mov(reg_aux_compensation, ptr[rsp + reg_aux_comp_offs_]);
            add(reg_aux_compensation,
                    (is_tail) ? compensations_offset(1, true)
                              : compensations_offset(ld_block2));
            mov(ptr[rsp + reg_aux_comp_offs_], reg_aux_compensation);
        }
        if (brg.with_scales) {
            mov(reg_aux_scales, ptr[rsp + reg_aux_scales_offs_]);
            add(reg_aux_scales,
                    (is_tail) ? scales_offset(1, true)
                              : scales_offset(ld_block2));
            mov(ptr[rsp + reg_aux_scales_offs_], reg_aux_scales);
        }
        if (with_binary_per_oc_bcast_) {
            mov(reg_aux_binary_postops_oc_l,
                    ptr[rsp + reg_aux_binary_postops_oc_l_offs_]);
            add(reg_aux_binary_postops_oc_l,
                    (is_tail) ? oc_logical_offset(1, true)
                              : oc_logical_offset(ld_block2));
            mov(ptr[rsp + reg_aux_binary_postops_oc_l_offs_],
                    reg_aux_binary_postops_oc_l);
        }
    };

    Label ldb_loop_label;
    Label BS_loop_label;

    if (!is_reg_tail) {
        mov(reg_aux_C, reg_C);
        mov(reg_aux_D, reg_D);
        xor_(reg_b_offset, reg_b_offset);
        if (brg.with_bias) {
            mov(reg_bias, ptr[rsp + reg_bias_offs_]);
            mov(ptr[rsp + reg_aux_bias_offs_], reg_bias);
        }
        if (brg.req_s8s8_compensation) {
            mov(reg_compensation, ptr[rsp + reg_comp_offs_]);
            mov(ptr[rsp + reg_aux_comp_offs_], reg_compensation);
        }
        if (brg.with_scales) {
            mov(reg_scales, ptr[rsp + reg_scales_offs_]);
            mov(ptr[rsp + reg_aux_scales_offs_], reg_scales);
        }
        if (with_binary_per_oc_bcast_) {
            mov(reg_binary_postops_oc_l,
                    ptr[rsp + reg_binary_postops_oc_l_offs_]);
            mov(ptr[rsp + reg_aux_binary_postops_oc_l_offs_],
                    reg_binary_postops_oc_l);
        }
    }

    auto ld_loop_body = [=](int vpad) {
        set_A_B_matrices();

        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
        const auto bd_b = nstl::max(0, vpad);
        const auto bd_e = nstl::min(bd_block, bd_block + vpad);
        if (bd_b >= bd_e) return;

        Label rdb_loop_label;
        if (brg.rdb > 0) {
            mov(reg_rdb_loop, brg.rdb);
            L_aligned(rdb_loop_label, 64);
            {
                const bool is_rd_tail = false;
                gemm_microkernel(bd_block2, is_bdb_tail, ld_block2, is_rd_tail,
                        is_ld_tail, vpad, rows_for_rd_tail);

                add(reg_aux_A, rdb_A_offset());
                add(reg_aux_B, rdb_B_offset());

                dec(reg_rdb_loop);
                cmp(reg_rdb_loop, 0);
            }
            jg(rdb_loop_label, T_NEAR);
        }
        if (brg.rdb_tail != 0) {
            const bool is_rd_tail = true;
            gemm_microkernel(bd_block2, is_bdb_tail, ld_block2, is_rd_tail,
                    is_ld_tail, vpad, rows_for_rd_tail);
        }
    };

    if (is_ldb_loop) {
        mov(reg_ldb_loop, ldb_loop_length);
        if (brg.is_amx) mov(ptr[rsp + reg_ldb_loop_offs_], reg_ldb_loop);
    }
    L_aligned(ldb_loop_label, 64);
    {
        load_accumulators(bd_block2, is_bdb_tail, ld_block2, is_ld_tail);

        if (is_ldb_loop)
            mov(ptr[rsp + reg_D_offs_], reg_D);
        else {
            mov(reg_ldb_loop, reg_D);
            if (brg.is_amx) mov(ptr[rsp + reg_ldb_loop_offs_], reg_ldb_loop);
        }
        if (brg.brgattr.max_bs > 1) mov(ptr[rsp + reg_aux_D_offs_], reg_aux_D);

        restore_A_B_matrices();
        if (brg.is_amx) {
            mov(reg_stride_lda, brg.typesize_A * brg.LDA);
            mov(reg_stride_ldb, brg.rd_step * brg.typesize_B * brg.LDB);
        }
        if (brg.req_s8s8_compensation) {
            mov(ptr[rsp + reg_bdb_loop_offs_], reg_bdb_loop);
            mov(reg_s8_input_shift, 128);
            vpbroadcastb(zmm_inp_shift(), reg_s8_input_shift.cvt8());
            mov(reg_bdb_loop, ptr[rsp + reg_bdb_loop_offs_]);
        }

        if (brg.alpha != 0.f) {
            if (brg.brgattr.max_bs > 1) mov(reg_BS_loop, reg_BS);
            L_aligned(BS_loop_label, 64);
            {
                if (check_top_vpad || check_bottom_vpad) {
                    const auto vpad_first = -brg.brgattr.max_bottom_vpad;
                    const auto vpad_last = brg.brgattr.max_top_vpad;
                    const auto n_vpads = vpad_last - vpad_first + 2;
                    constexpr auto MAX_N_VPADS = 2 * brgemm_t::MAX_VPAD;
                    assert(n_vpads < MAX_N_VPADS);

                    Label Vpad_loop_end_label;
                    Label Vpad_loop_iter_label[MAX_N_VPADS];
                    if (vpad_exist) {
                        reg64_t reg_batch = (brg.type == brgemm_addr)
                                ? reg_aux1_batch
                                : ((brg.type == brgemm_offs) ? reg_offs_batch
                                                             : reg_strd_batch);
                        if (brg.type == brgemm_strd)
                            mov(reg_strd_batch,
                                    ptr[rsp + origin_strd_batch_offs_]);

                        mov(reg_aux_A_vpad,
                                ptr[reg_batch
                                        + GET_OFF_BATCH_ELEMENT(vvpad.top)]);
                        sub(reg_aux_A_vpad,
                                ptr[reg_batch
                                        + GET_OFF_BATCH_ELEMENT(vvpad.bottom)]);
                    } else
                        xor_(reg_aux_A_vpad, reg_aux_A_vpad);

                    for (int vpad = vpad_first; vpad <= vpad_last; vpad++) {
                        const auto label_vpad = vpad - vpad_first;
                        L(Vpad_loop_iter_label[label_vpad]);
                        if (!check_top_vpad && vpad > 0) continue;
                        if (!check_bottom_vpad && vpad < 0) continue;
                        auto real_vpad = vpad;
                        if (check_bottom_vpad && !is_bdb_tail && brg.bdb_tail) {
                            // for last full block before
                            // bdb_tail && -vpad greater than bdb_tail
                            if (brg.bdb_tail < -vpad)
                                real_vpad += brg.bdb_tail;
                            else
                                continue;
                        }
                        cmp(reg_aux_A_vpad, vpad);
                        jne(Vpad_loop_iter_label[label_vpad + 1], T_NEAR);
                        ld_loop_body(real_vpad);
                        jmp(Vpad_loop_end_label, T_NEAR);
                    }
                    L(Vpad_loop_iter_label[n_vpads - 1]);
                    ld_loop_body(0);
                    L(Vpad_loop_end_label);
                } else {
                    ld_loop_body(0);
                }
                if (brg.brgattr.max_bs > 1) {
                    dec(reg_BS_loop);
                    cmp(reg_BS_loop, 0);
                    jg(BS_loop_label, T_NEAR);
                }
            }
        }

        if (is_ldb_loop)
            mov(reg_D, ptr[rsp + reg_D_offs_]);
        else {
            if (brg.is_amx) mov(reg_ldb_loop, ptr[rsp + reg_ldb_loop_offs_]);
            mov(reg_D, reg_ldb_loop);
        }
        if (brg.brgattr.max_bs > 1) mov(reg_aux_D, ptr[rsp + reg_aux_D_offs_]);

        store_accumulators(bd_block2, is_bdb_tail, ld_block2, is_ld_tail);
        if (is_ldb_loop) {
            if (brg.is_amx) mov(reg_ldb_loop, ptr[rsp + reg_ldb_loop_offs_]);
            if (!is_ld_tail)
                ldb_shift(ld_block2);
            else
                ldb_shift(1, true);
            dec(reg_ldb_loop);
            cmp(reg_ldb_loop, 0);
            if (brg.is_amx) mov(ptr[rsp + reg_ldb_loop_offs_], reg_ldb_loop);
            jg(ldb_loop_label, T_NEAR);
        }
    }
}

void jit_brgemm_kernel_base_t::bdb_loop() {
    auto do_ldb_loop = [=](int bd_block2, bool is_bdb_tail, bool check_top_vpad,
                               bool check_bottom_vpad, int rows_for_rd_tail) {
        if (brg.ldb2 > 0) {
            const bool is_ld_reg_tail = false;
            const bool is_ld_tail = false;
            ldb_loop(bd_block2, is_bdb_tail, brg.ld_block2, brg.ldb2,
                    is_ld_reg_tail, is_ld_tail, check_top_vpad,
                    check_bottom_vpad, rows_for_rd_tail);
        }
        if (brg.ldb2_tail > 0) {
            const bool is_ld_reg_tail = (brg.ldb2 == 0) ? false : true;
            const bool is_ld_tail = false;
            ldb_loop(bd_block2, is_bdb_tail, brg.ldb2_tail, 1, is_ld_reg_tail,
                    is_ld_tail, check_top_vpad, check_bottom_vpad,
                    rows_for_rd_tail);
        }
        if (brg.ldb_tail > 0) {
            const bool is_ld_reg_tail
                    = (brg.ldb2 == 0 && brg.ldb2_tail == 0) ? false : true;
            const bool is_ld_tail = true;
            ldb_loop(bd_block2, is_bdb_tail, 1, 1, is_ld_reg_tail, is_ld_tail,
                    check_top_vpad, check_bottom_vpad, rows_for_rd_tail);
        }
    };

    auto bdb_loop_body
            = [=](int bd_block2, bool is_bdb_tail, bool check_top_vpad,
                      bool check_bottom_vpad, int rows_for_rd_tail) {
                  do_ldb_loop(bd_block2, is_bdb_tail, check_top_vpad,
                          check_bottom_vpad, rows_for_rd_tail);

                  add(reg_C, bdb_C_offset(bd_block2));
                  add(reg_D, bdb_D_offset(bd_block2));
                  add(reg_a_offset, bdb_A_offset(bd_block2));
              };

    int rows_for_rd_tail, bd_blocks_for_rd_tail;

    if (brg.is_amx) {
        rows_for_rd_tail = 0;
        bd_blocks_for_rd_tail = 0;
        n_bcast_1_load = false;
    } else {
        rows_for_rd_tail = 0;
        if (brg.rdb_tail != 0 && (brg.is_bf16 || brg.is_int8)) {
            const auto rd_tail_size = brg.rdb_tail % brg.rd_step;
            rows_for_rd_tail = rd_tail_size
                    ? div_up(brg.rd_step - rd_tail_size, brg.reduce_dim)
                    : 0;
        }
        bd_blocks_for_rd_tail
                = div_up(nstl::max(0,
                                 rows_for_rd_tail - brg.bdb_tail
                                         + brg.brgattr.max_bottom_vpad),
                        brg.bd_block);

        auto ld_block2 = (brg.ldb2 > 0)
                ? brg.ld_block2
                : ((brg.ldb2_tail > 0) ? brg.ldb2_tail : 1);
        n_bcast_1_load = brg.is_int8
                && ((brg.bd_block * (ld_block2 + 1) < 32)
                        && (bd_blocks_for_rd_tail == 0)
                        && (rows_for_rd_tail == 0));
        // loop order may be specified in brgemm attributes
        if (brg.brgattr.hint_loop_order != brgemm_lo_default)
            n_bcast_1_load = (brg.brgattr.hint_loop_order == brgemm_lo_bl_1load)
                    ? true
                    : false;
    }

    auto bdb_loop_avx512 = [=]() {
        Label bdb_loop_end_label, no_vpad_label;
        if (vpad_exist) {
            // max_top_vp is restricted by bd_block due to
            // brgemm_kernel implementation. TODO: remove this restriction
            assert(brg.brgattr.max_top_vpad <= brg.bd_block
                    && brg.brgattr.max_bottom_vpad <= brg.bd_block);

            if (brg.type == brgemm_strd) {
                // if batch is nullptr then it means no vpadding in this call
                cmp(reg_offs_batch, 0);
                je(no_vpad_label, T_NEAR);
            }

            // first bd_block --------------
            auto bdblocks = brg.bdb;
            if (bdblocks >= 1) {
                bdb_loop_body(1, false, true, brg.bdb == 1 && brg.bdb_tail == 0,
                        brg.bdb - bd_blocks_for_rd_tail > 0 ? 0
                                                            : rows_for_rd_tail);
                bdblocks--;
            }
            if (bdblocks > 1) {
                // middle bd_blocks -----------
                Label bdb_loop_label;
                mov(reg_bdb_loop, bdblocks);
                L_aligned(bdb_loop_label, 64);
                {
                    bdb_loop_body(1, false, false, false,
                            bd_blocks_for_rd_tail <= 1 ? 0 : rows_for_rd_tail);

                    dec(reg_bdb_loop);
                    cmp(reg_bdb_loop, 1);
                    jg(bdb_loop_label, T_NEAR);
                }
                bdblocks = 1;
            }
            if (bdblocks == 1) {
                // last bd_block ------------
                bdb_loop_body(1, false, false, true,
                        bd_blocks_for_rd_tail == 0 ? 0 : rows_for_rd_tail);
            }
            if (brg.bdb_tail > 0)
                do_ldb_loop(1, true, brg.bdb < 1, true, rows_for_rd_tail);
            // for brgemm_strd "no vpadding" case may be implemented, so skip it
            if (brg.type == brgemm_strd) jmp(bdb_loop_end_label);
        }
        if (!vpad_exist || brg.type == brgemm_strd) {
            // for brgemm_strd batch may be null so we need this code path
            L_aligned(no_vpad_label, 64);
            if (brg.bdb > 0) {
                mov(reg_bdb_loop, brg.bdb);
                if (brg.bdb > (rows_for_rd_tail ? 1 : 0)) {
                    Label bdb_loop_label;
                    L_aligned(bdb_loop_label, 64);
                    {
                        bdb_loop_body(1, false, false, false,
                                bd_blocks_for_rd_tail <= 1 ? 0
                                                           : rows_for_rd_tail);
                        dec(reg_bdb_loop);
                        cmp(reg_bdb_loop, rows_for_rd_tail ? 1 : 0);
                        jg(bdb_loop_label, T_NEAR);
                    }
                }

                if (rows_for_rd_tail)
                    bdb_loop_body(1, false, false, true,
                            bd_blocks_for_rd_tail == 0 ? 0 : rows_for_rd_tail);
            }
            if (brg.bdb_tail > 0)
                do_ldb_loop(1, true, false, false, rows_for_rd_tail);
        }
        L_aligned(bdb_loop_end_label, 64);
    };
    auto bdb_loop_amx = [=]() {
        Label bdb_loop_label;
        if (brg.bd_block2 > 1) {
            mov(reg_bdb_loop, brg.bdb2);
            mov(ptr[rsp + reg_bdb_loop_offs_], reg_bdb_loop);
            L_aligned(bdb_loop_label, 64);
            {
                bdb_loop_body(brg.bd_block2, false, false, false, 0);
                mov(reg_bdb_loop, ptr[rsp + reg_bdb_loop_offs_]);
                dec(reg_bdb_loop);
                cmp(reg_bdb_loop, 0);
                mov(ptr[rsp + reg_bdb_loop_offs_], reg_bdb_loop);
            }
            jg(bdb_loop_label, T_NEAR);
        }
        if (brg.bdb2_tail > 0)
            bdb_loop_body(brg.bdb2_tail, false, false, false, 0);
        if (brg.bdb_tail > 0) do_ldb_loop(1, true, false, false, 0);
    };

    if (brg.type == brgemm_addr && brg.brgattr.max_bs == 1 && !vpad_exist) {
        mov(reg_aux1_A, ptr[reg_addr_batch + GET_OFF_BATCH_ELEMENT(ptr.A)]);
        mov(reg_aux1_B, ptr[reg_addr_batch + GET_OFF_BATCH_ELEMENT(ptr.B)]);
    }

    xor_(reg_a_offset, reg_a_offset);
    if (brg.is_amx)
        bdb_loop_amx();
    else
        bdb_loop_avx512();
}

void jit_brgemm_kernel_base_t::generate() {
    preamble();

    sub(rsp, stack_space_needed_);

    const auto full_mask = size_t {0xffffffffffffffff};
    const auto tail_mask = size_t((1 << brg.ldb_tail) - 1);

    int is_ldb2_tail = brg.ldb2_tail ? 1 : 0;
    int is_ldb_tail = brg.ldb_tail ? 1 : 0;
    is_ldb_loop = (brg.ldb2 + is_ldb2_tail + is_ldb_tail) > 1 ? true : false;
    vpad_exist
            = (brg.brgattr.max_top_vpad > 0 || brg.brgattr.max_bottom_vpad > 0)
            ? true
            : false;

    reg64_t reg_mask = rax;

    mov(reg_mask, full_mask);
    kmovq(ld_full_mask, reg_mask);
    mov(reg_mask, tail_mask);
    kmovq(ld_tail_mask, reg_mask);

    read_params();

    bdb_loop();

    add(rsp, stack_space_needed_);

    postamble();

    if (brg.with_eltwise) postops_injector_->prepare_table();
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
