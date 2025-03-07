/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
* Copyright 2024-2025 Arm Ltd. and affiliates
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

#include "cpu/aarch64/brgemm/brgemm_types.hpp"
#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/platform.hpp"

#define GET_OFF(field) (uint32_t) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) \
    (uint32_t) offsetof(brgemm_batch_element_t, field)
#define LD_MUL_VL(mn, op, mask, addr, off, size) \
    { \
        const int mul_vl_len = (cpu_sveLen / 4) * size; \
        const int off_mod = (off) % mul_vl_len; \
        const int off_mul_vl = (off) / mul_vl_len; \
        if (off_mod == 0 && -8 <= off_mul_vl && off_mul_vl <= 7) \
            mn(op, mask / T_z, ptr(addr, off_mul_vl, MUL_VL)); \
        else \
            mn(op, mask / T_z, \
                    ptr(addr_off(addr, off, X_DEFAULT_ADDR, X_TMP_0))); \
    }
#define ST_MUL_VL(mn, op, mask, addr, off, size) \
    { \
        const int mul_vl_len = (cpu_sveLen / 4) * size; \
        const int off_mod = (off) % mul_vl_len; \
        const int off_mul_vl = (off) / mul_vl_len; \
        if (off_mod == 0 && -8 <= off_mul_vl && off_mul_vl <= 7) \
            mn(op, mask, ptr(addr, off_mul_vl, MUL_VL)); \
        else \
            mn(op, mask, ptr(addr_off(addr, off, X_DEFAULT_ADDR, X_TMP_0))); \
    }
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
#define STR_IMM(reg, addr, off) \
    { \
        const uint64_t IMM12_MASK = ~uint64_t(0xfff); \
        if ((off & IMM12_MASK) == 0) { \
            str(reg, ptr(addr, off)); \
        } else { \
            add_imm(X_DEFAULT_ADDR, addr, off, X_TMP_0); \
            str(reg, ptr(X_DEFAULT_ADDR)); \
        } \
    }

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;

struct jit_brgemm_kernel_t : public jit_generator {
    jit_brgemm_kernel_t(const brgemm_desc_t &abrg)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, sve_512)
        , brg(abrg)
        , postops_injector_(nullptr)
        , max_effective_vregs(
                  max_vregs - ((brg.is_int8 && !brg.has_int8_vnni) ? 2 : 0)) {

        // The implementation uses is_superset(), is_subset() utilities.
        // So avoid isa_all, isa_undef in these comparisions.
        assert(!utils::one_of(brg.isa_impl, isa_all, isa_undef));
        const int is_ldb2_tail = brg.ldb2_tail ? 1 : 0;
        const int is_ldb_tail = brg.ldb_tail ? 1 : 0;
        is_ldb_loop_ = brg.ldb2 + is_ldb2_tail + is_ldb_tail > 1;

        if (brg.with_eltwise || brg.with_binary || brg.with_sum) {

            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;
            const auto dst_md_wrapper = memory_desc_wrapper(brg.dst_md);

            static const bcast_set_t enabled_bcast_strategy
                    = {broadcasting_strategy_t::scalar,
                            broadcasting_strategy_t::per_oc,
                            broadcasting_strategy_t::per_oc_spatial,
                            broadcasting_strategy_t::per_mb_spatial,
                            broadcasting_strategy_t::per_mb_w,
                            broadcasting_strategy_t::per_w,
                            broadcasting_strategy_t::no_broadcast};
            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    static_cast<size_t>(Xbyak_aarch64::ZReg(1).getIdx()),
                    XReg(14), XReg(15), XReg(13), preserve_gpr, preserve_vmm,
                    GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(data_C_ptr_),
                    dst_md_wrapper, static_cast<size_t>(brg.ldb_tail),
                    PReg(ld_tail_mask.getIdx()), use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {
                    XReg(this->param1.getIdx()), enabled_bcast_strategy,
                    rhs_sp};

            postops_injector_ = utils::make_unique<po_injector_t>(
                    this, brg.attr->post_ops_, bsp);

            with_binary_non_scalar_bcast_ = binary_injector::
                    any_binary_postop_rhs_non_scalar_broadcast(
                            brg.attr->post_ops_, dst_md_wrapper);
        }
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_t)

    brgemm_desc_t brg;

private:
    using po_injector_t = injector::jit_uni_postops_injector_t<sve_512>;
    std::unique_ptr<po_injector_t> postops_injector_;

    // Register decomposition
    const XReg param1 = XReg(7); //abi_param1_x64=r7

    const XReg reg_C = x15;
    const XReg reg_aux_C = x14;

    const XReg reg_addr_batch = x13;
    const XReg reg_A = x13;
    const XReg reg_B = x12;

    const XReg reg_aux_A = x11;
    const XReg reg_aux_B = x10;
    const XReg reg_aux_A_vpad = reg_aux_A;

    const XReg reg_bdb_loop = x9;
    const XReg reg_ldb_loop = x8;

    const XReg reg_stride_lda = reg_bdb_loop;
    const XReg reg_stride_ldb = reg_ldb_loop;
    const XReg reg_stride_ld_block = reg_ldb_loop;
    const XReg reg_s8_input_shift = reg_bdb_loop;
    const XReg reg_zp_a_input_shift = reg_bdb_loop;

    const XReg reg_BS_loop = x0;
    const XReg reg_rdb_loop = x3;
    const XReg reg_BS = x1; //from jit_generator.hpp in x64

    const XReg reg_a_offset = x2;
    const XReg reg_b_offset = x6;

    const XReg reg_aux1_batch = x5;
    const XReg reg_aux1_A = x5;
    const XReg reg_aux1_B = x7; //from jit_generator.hpp in x64

    const XReg reg_offs_batch = reg_aux1_A;
    const XReg reg_strd_batch = reg_rdb_loop;

    const XReg reg_bias = reg_rdb_loop;
    const XReg reg_scales = reg_rdb_loop;
    const XReg reg_aux_bias = reg_rdb_loop;
    const XReg reg_dst_scales = reg_rdb_loop;
    const XReg reg_zp_comp_a = reg_rdb_loop;
    const XReg reg_aux_zp_comp_a = reg_rdb_loop;
    const XReg reg_zp_comp_b = reg_rdb_loop;
    const XReg reg_aux_zp_comp_b = reg_rdb_loop;
    const XReg reg_zp_c_values = reg_rdb_loop;
    const XReg reg_aux_zp_c_values = reg_rdb_loop;

    const XReg reg_aux_scales = reg_aux_B;
    const XReg reg_aux_dst_scales = reg_aux_B;
    const XReg reg_do_post_ops = reg_rdb_loop;
    const XReg reg_do_comp = reg_rdb_loop;
    const XReg reg_skip_accm = reg_rdb_loop;
    const XReg reg_tmp_gpr = reg_rdb_loop;
    const XReg reg_ptr_sum_scale = reg_rdb_loop;
    const XReg reg_ptr_sum_zp = reg_bdb_loop;
    const XReg reg_zp_a_val = reg_rdb_loop;

    const XReg reg_buf = reg_rdb_loop;
    const XReg reg_compensation = reg_bias;
    const XReg reg_aux_compensation = reg_aux_bias;

    const XReg reg_D = reg_aux_A;
    const XReg reg_aux_D = reg_BS_loop;

    const XReg reg_tmp_ = x16;

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
    constexpr static int reg_zp_comp_a_offs_ = 104;
    constexpr static int reg_aux_zp_comp_a_offs_ = 112;
    constexpr static int reg_zp_comp_b_offs_ = 120;
    constexpr static int reg_aux_zp_comp_b_offs_ = 128;
    constexpr static int reg_zp_c_values_offs_ = 136;
    constexpr static int reg_aux_zp_c_values_offs_ = 144;
    constexpr static int reg_data_C_ptr_ = 152;
    constexpr static int reg_skip_accm_offs_ = 160;
    constexpr static int reg_zp_a_val_offs_ = 168;
    constexpr static int reg_do_comp_offs_ = 176;
    constexpr static int reg_dst_scales_offs_ = 184;
    constexpr static int stack_space_needed_ = 192;

    bool is_ldb_loop_ = false;
    bool with_binary_non_scalar_bcast_ = false;
    constexpr static int max_vregs = 32;
    const int max_effective_vregs;

    PReg ld_full_mask = PReg(2);
    PReg ld_tail_mask = PReg(3);

    ZReg accm(int ld_block, int bd, int ld) {
        return ZReg(max_effective_vregs - 1 - (bd * ld_block + ld));
    }

    ZReg bcst(int bd = 0) {
        if (n_bcast_1_load) {
            int idx = max_effective_vregs - 1 - (brg.ld_block2 * brg.bd_block)
                    - bd;
            assert(idx > 0);
            return ZReg(idx);
        } else
            return this->z0;
    }

    ZReg load(int ld = 0) {
        if (n_bcast_1_load) {
            return this->z0;
        } else {
            int idx = max_effective_vregs - 1 - (brg.ld_block2 * brg.bd_block)
                    - ld;
            assert(idx > 0);
            return ZReg(idx);
        }
    }
    const ZReg &z_tmp_1() const noexcept { return this->z0; }
    const ZReg &z_tmp_2() const noexcept { return this->z1; }
    const ZReg &z_tmp_3() const noexcept { return this->z2; }
    const ZReg &z_tail_mask() const noexcept { return this->z1; }
    const ZReg &z_one_bytes() const noexcept { return this->z3; }
    const ZReg &z_zp_a_shift() const noexcept { return this->z2; }
    const ZReg &z_inp_shift() const noexcept { return this->z1; }

    ZReg int8_ones_words() const noexcept { return ZReg(max_vregs - 1); }
    ZReg int8_dot_product_temp() const noexcept { return ZReg(max_vregs - 2); }

    void load_data(data_type_t type_in, const Xbyak_aarch64::ZReg &vmm,
            const Xbyak_aarch64::XReg &reg_addr, int load_size);

    void cvt2ps(data_type_t type_in, const ZReg zmm_in, const XReg &addr,
            bool mask_flag, bool store, PReg ktail_mask, const int offset,
            const int base_offset); //for only memory operand

    void advance_ldb_post_op_regs();
    void restore_ldb_post_op_regs(int ld_block2);
    void advance_bdb_post_op_regs(int adj_bd_block);
    void restore_bdb_post_op_regs(int bd_block2);
    void ldb_regs_shift(int ld_block2, bool is_tail = false);
    void advance_bd_block2_post_op_regs(int bd_block2);

    void copy_post_ops_stack_values_to_aux(bool is_reg_tail);
    void read_params();
    void zero_accumulators(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_ld_tail, bool skip_accumulation);

    void store_accumulators(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_ld_tail, bool skip_accumulation);
    void store_accumulators_without_post_ops(
            int bd_block, int ld_block, bool is_ld_tail);
    void store_accumulators_apply_post_ops(int bd_block, int ld_block,
            int ldb_and_bdb_offset, bool is_ld_tail);
    void apply_compensation(int bd_block, int ld_block, bool is_ld_tail);
    void apply_alpha_beta(int bd_block, int ld_block, bool is_ld_tail);
    void apply_post_ops(int bd_block, int ld_block2, int ldb_and_bdb_offset,
            bool is_ld_tail);
    void restore_A_B_matrices();
    void set_A_B_matrices();

    void compute_int8_compensation(int rd_loop, int bd_b, int bd_e,
            int bd_block, int ld_block2, bool is_ld_tail, int vpad);

    void dot_product(ZReg z1, ZReg z2, ZReg z3);
    void gemm_microkernel_sve512(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_rd_tail, bool is_ld_tail, int vpad, int rows_for_rd_tail);

    void ldb_loop(int bd_block2, bool is_bdb_tail, int ld_block,
            int ldb_loop_length, bool is_reg_tail, bool is_ld_tail,
            bool check_top_vpad, bool check_bottom_vpad, int rows_for_rd_tail,
            bool skip_accumulation);
    void bdb_loop();

    void generate() override;

    int A_offset(int bd, int rd) const noexcept;
    int B_offset(int ld, int rd) const noexcept;
    int C_offset(int bd, int ld) const noexcept;
    int D_offset(int bd, int ld) const noexcept;
    int po_offset(int bd, int ld) const noexcept;

    int rdb_A_offset() const noexcept;
    int rdb_B_offset() const noexcept;

    int ldb_B_offset(int ld_block2, bool is_tail = false) const noexcept;
    int ldb_C_offset(int ld_block2, bool is_tail = false) const noexcept;
    int ldb_D_offset(int ld_block2, bool is_tail = false) const noexcept;
    int ldb_po_offset(int ld_block2, bool is_tail = false) const noexcept;

    int bdb_A_offset(int bd_block2) const noexcept;
    int bdb_C_offset(int bd_block2) const noexcept;
    int bdb_D_offset(int bd_block2) const noexcept;
    int bdb_po_offset(int bd_block2) const noexcept;

    int bias_offset(int ld, bool is_tail = false) const noexcept;
    int oc_logical_offset(int ld, bool is_tail = false) const noexcept;

    int compensations_offset(int ld, bool is_tail = false) const noexcept;
    int bdb_compensation_offset(int bd_block2) const noexcept;
    int compensation_vpad_offset(int ld, int bd) const noexcept;
    int scales_offset(int ld, bool is_tail = false) const noexcept;
    int zp_comp_a_offset(int ld, bool is_tail = false) const noexcept;
    int zp_comp_a_vpad_offset(int ld, int bd) const noexcept;
    int bdb_zp_comp_a_offset(int bd_block2) const noexcept;
    int zp_comp_b_offset(int bd) const noexcept;
    int bdb_zp_comp_b_offset(int bd_block2) const noexcept;
    int zp_c_values_offset(int ld, bool is_tail = false) const noexcept;

    bool n_bcast_1_load = false;
    bool vpad_exist = false;
    bool need_comp_pads = false;
};

int jit_brgemm_kernel_t::A_offset(int bd, int rd) const noexcept {
    return brg.typesize_A * (bd * brg.LDA + rd);
}
int jit_brgemm_kernel_t::B_offset(int ld, int rd) const noexcept {
    const int data_vnni_granularity = brg.ld_step;
    const int rdb0 = rd / data_vnni_granularity;
    return brg.typesize_B
            * (rdb0 * data_vnni_granularity * brg.LDB
                    + data_vnni_granularity * ld * brg.ld_block);
}
int jit_brgemm_kernel_t::C_offset(int bd, int ld) const noexcept {
    return brg.typesize_C * (bd * brg.LDC + ld * brg.ld_block);
}
int jit_brgemm_kernel_t::D_offset(int bd, int ld) const noexcept {
    return brg.typesize_D * (bd * brg.LDD + ld * brg.ld_block);
}
int jit_brgemm_kernel_t::po_offset(int bd, int ld) const noexcept {
    return bd * brg.LDD + ld * brg.ld_block;
}

int jit_brgemm_kernel_t::rdb_A_offset() const noexcept {
    return brg.typesize_A * brg.rd_block;
}
int jit_brgemm_kernel_t::rdb_B_offset() const noexcept {
    return brg.typesize_B * brg.rd_block * brg.LDB;
}

int jit_brgemm_kernel_t::ldb_B_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_B * brg.ldb_tail * brg.ld_step
                     : brg.typesize_B * ld_block2 * brg.ld_block * brg.ld_step;
}
int jit_brgemm_kernel_t::ldb_C_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_C * brg.ldb_tail
                     : brg.typesize_C * ld_block2 * brg.ld_block;
}
int jit_brgemm_kernel_t::ldb_D_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_D * brg.ldb_tail
                     : brg.typesize_D * ld_block2 * brg.ld_block;
}
int jit_brgemm_kernel_t::ldb_po_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.ldb_tail : ld_block2 * brg.ld_block;
}

int jit_brgemm_kernel_t::bdb_A_offset(int bd_block2) const noexcept {
    return brg.typesize_A * bd_block2 * brg.bd_block * brg.LDA;
}
int jit_brgemm_kernel_t::bdb_C_offset(int bd_block2) const noexcept {
    return brg.typesize_C * bd_block2 * brg.bd_block * brg.LDC;
}
int jit_brgemm_kernel_t::bdb_D_offset(int bd_block2) const noexcept {
    return brg.typesize_D * bd_block2 * brg.bd_block * brg.LDD;
}
int jit_brgemm_kernel_t::bdb_po_offset(int bd_block2) const noexcept {
    return bd_block2 * brg.bd_block * brg.LDD;
}

int jit_brgemm_kernel_t::bias_offset(int ld, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_bias * brg.ldb_tail
                     : brg.typesize_bias * ld * brg.ld_block;
}

int jit_brgemm_kernel_t::oc_logical_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? brg.ldb_tail : ld * brg.ld_block;
}

int jit_brgemm_kernel_t::compensations_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                     : sizeof(int32_t) * ld * brg.ld_block;
}

int jit_brgemm_kernel_t::bdb_compensation_offset(int bd_block2) const noexcept {
    return sizeof(int32_t) * bd_block2 * brg.bd_block * brg.LDB;
}

int jit_brgemm_kernel_t::compensation_vpad_offset(
        int ld, int bd) const noexcept {
    return sizeof(int32_t) * (ld * brg.ld_block + bd * brg.LDB);
}

int jit_brgemm_kernel_t::scales_offset(int ld, bool is_tail) const noexcept {
    return (is_tail) ? brg.is_oc_scale * sizeof(float) * brg.ldb_tail
                     : brg.is_oc_scale * sizeof(float) * ld * brg.ld_block;
}

int jit_brgemm_kernel_t::zp_comp_a_offset(int ld, bool is_tail) const noexcept {
    return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                     : sizeof(int32_t) * ld * brg.ld_block;
}

int jit_brgemm_kernel_t::bdb_zp_comp_a_offset(int bd_block2) const noexcept {
    return sizeof(int32_t) * bd_block2 * brg.bd_block * brg.LDB;
}

int jit_brgemm_kernel_t::zp_comp_a_vpad_offset(int ld, int bd) const noexcept {
    return sizeof(int32_t) * (ld * brg.ld_block + bd * brg.LDB);
}

int jit_brgemm_kernel_t::zp_comp_b_offset(int bd) const noexcept {
    return sizeof(int32_t) * bd;
}

int jit_brgemm_kernel_t::bdb_zp_comp_b_offset(int bd_block2) const noexcept {
    return zp_comp_b_offset(bd_block2 * brg.bd_block);
}

int jit_brgemm_kernel_t::zp_c_values_offset(
        int ld, bool is_tail) const noexcept {
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                         : sizeof(int32_t) * ld * brg.ld_block;
    }

    return 0;
}

void jit_brgemm_kernel_t::load_data(data_type_t type_in,
        const Xbyak_aarch64::ZReg &vmm, const Xbyak_aarch64::XReg &reg_addr,
        int load_size) {
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: ld1w(vmm.s, P_ALL_ONE / T_z, ptr(reg_addr)); break;
        case data_type::s8:
        case data_type::u8: assert(!"unsupported\n"); break;
        case data_type::bf16: assert(!"unsupported\n"); break;
        case data_type::f16: assert(!"unsupported\n"); break;
        default: assert(!"unsupported source data type");
    }
}

void jit_brgemm_kernel_t::cvt2ps(data_type_t type_in, const ZReg zmm_in,
        const XReg &addr, bool mask_flag, bool store, PReg ktail_mask,
        const int offset, const int base_offset) {
    const auto mask = mask_flag ? ktail_mask : P_ALL_ONE;
    switch (type_in) {
        case data_type::f32:
        case data_type::s32:
            LD_MUL_VL(ld1w, z_tmp_1().s, mask, addr, offset - base_offset, 4);
            if (store) //Merging
                mov(zmm_in.s, ktail_mask / T_m, z_tmp_1().s);
            break;
        case data_type::bf16: assert(!"unsupported data type\n"); break;
        case data_type::s8: assert(!"unsupported data type\n"); break;
        case data_type::u8: assert(!"unsupported data type\n"); break;
        default: assert(!"unsupported data type");
    }
    if (!one_of(type_in, data_type::f32, data_type::bf16))
        assert(!"unsupported data type\n");
}

void jit_brgemm_kernel_t::advance_ldb_post_op_regs() {
    if (brg.with_bias) {
        LDR_IMM(reg_aux_bias, X_SP, reg_aux_bias_offs_);
        add_imm(reg_aux_bias, reg_aux_bias, bias_offset(1), X_TMP_0);
        STR_IMM(reg_aux_bias, X_SP, reg_aux_bias_offs_);
    }
    if (brg.with_scales) {
        LDR_IMM(reg_aux_scales, X_SP, reg_aux_scales_offs_);
        add_imm(reg_aux_scales, reg_aux_scales, scales_offset(1), X_TMP_0);
        STR_IMM(reg_aux_scales, X_SP, reg_aux_scales_offs_);
    }
    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        LDR_IMM(reg_aux_zp_comp_a, X_SP, reg_aux_zp_comp_a_offs_);
        add_imm(reg_aux_zp_comp_a, reg_aux_zp_comp_a, zp_comp_a_offset(1),
                X_TMP_0);
        STR_IMM(reg_aux_zp_comp_a, X_SP, reg_aux_zp_comp_a_offs_);
    }
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        LDR_IMM(reg_aux_zp_c_values, X_SP, reg_aux_zp_c_values_offs_);
        add_imm(reg_aux_zp_c_values, reg_aux_zp_c_values, zp_c_values_offset(1),
                X_TMP_0);
        STR_IMM(reg_aux_zp_c_values, X_SP, reg_aux_zp_c_values_offs_);
    }
}

void jit_brgemm_kernel_t::restore_ldb_post_op_regs(int ld_block2) {
    if (brg.with_bias) {
        LDR_IMM(reg_aux_bias, X_SP, reg_aux_bias_offs_);
        sub_imm(reg_aux_bias, reg_aux_bias, bias_offset(ld_block2 - 1),
                X_TMP_0);
        STR_IMM(reg_aux_bias, X_SP, reg_aux_bias_offs_);
    }
    if (brg.with_scales) {
        LDR_IMM(reg_aux_scales, X_SP, reg_aux_scales_offs_);
        sub_imm(reg_aux_scales, reg_aux_scales, scales_offset(ld_block2 - 1),
                X_TMP_0);
        STR_IMM(reg_aux_scales, X_SP, reg_aux_scales_offs_);
    }
    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        LDR_IMM(reg_aux_zp_comp_a, X_SP, reg_aux_zp_comp_a_offs_);
        sub_imm(reg_aux_zp_comp_a, reg_aux_zp_comp_a,
                zp_comp_a_offset(ld_block2 - 1), X_TMP_0);
        STR_IMM(reg_aux_zp_comp_a, X_SP, reg_aux_zp_comp_a_offs_);
    }
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        LDR_IMM(reg_aux_zp_c_values, X_SP, reg_aux_zp_c_values_offs_);
        sub_imm(reg_aux_zp_c_values, reg_aux_zp_c_values,
                zp_c_values_offset(ld_block2 - 1), X_TMP_0);
        STR_IMM(reg_aux_zp_c_values, X_SP, reg_aux_zp_c_values_offs_);
    }
}

void jit_brgemm_kernel_t::advance_bdb_post_op_regs(int adj_bd_block) {
    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        LDR_IMM(reg_aux_zp_comp_b, X_SP, reg_aux_zp_comp_b_offs_);
        add_imm(reg_aux_zp_comp_b, reg_aux_zp_comp_b, bdb_zp_comp_b_offset(1),
                X_TMP_0);
        STR_IMM(reg_aux_zp_comp_b, X_SP, reg_aux_zp_comp_b_offs_);
    }
}

void jit_brgemm_kernel_t::restore_bdb_post_op_regs(int bd_block2) {
    bool post_processed = false;
    if (bd_block2 > 1) {
        if (brg.zp_type_b != brgemm_broadcast_t::none) {
            post_processed = true;
            LDR_IMM(reg_aux_zp_comp_b, X_SP, reg_aux_zp_comp_b_offs_);
            sub_imm(reg_aux_zp_comp_b, reg_aux_zp_comp_b,
                    bdb_zp_comp_b_offset(bd_block2 - 1), X_TMP_0);
            STR_IMM(reg_aux_zp_comp_b, X_SP, reg_aux_zp_comp_b_offs_);
        }
    }
    if (post_processed) LDR_IMM(reg_buf, X_SP, reg_buf_offs_);
}

void jit_brgemm_kernel_t::ldb_regs_shift(int ld_block2, bool is_tail) {
    int C_offset = (is_tail) ? ldb_C_offset(1, true) : ldb_C_offset(ld_block2);
    int D_offset = (is_tail) ? ldb_D_offset(1, true) : ldb_D_offset(ld_block2);

    add_imm(reg_aux_C, reg_aux_C, C_offset, X_TMP_0);
    add_imm(reg_aux_D, reg_aux_D, D_offset, X_TMP_0);

    add_imm(reg_b_offset, reg_b_offset,
            (is_tail) ? ldb_B_offset(1, true) : ldb_B_offset(ld_block2),
            X_TMP_0);

    if (brg.with_bias) {
        LDR_IMM(reg_aux_bias, X_SP, reg_aux_bias_offs_);
        add_imm(reg_aux_bias, reg_aux_bias,
                (is_tail) ? bias_offset(1, true) : bias_offset(ld_block2),
                X_TMP_0);
        STR_IMM(reg_aux_bias, X_SP, reg_aux_bias_offs_);
    }
    if (brg.req_s8s8_compensation) {
        LDR_IMM(reg_aux_compensation, X_SP, reg_aux_comp_offs_);
        add_imm(reg_aux_compensation, reg_aux_compensation,
                (is_tail) ? compensations_offset(1, true)
                          : compensations_offset(ld_block2),
                X_TMP_0);
        STR_IMM(reg_aux_compensation, X_SP, reg_aux_comp_offs_);
    }
    if (brg.with_scales) {
        LDR_IMM(reg_aux_scales, X_SP, reg_aux_scales_offs_);
        add_imm(reg_aux_scales, reg_aux_scales,
                (is_tail) ? scales_offset(1, true) : scales_offset(ld_block2),
                X_TMP_0);
        STR_IMM(reg_aux_scales, X_SP, reg_aux_scales_offs_);
    }
    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        LDR_IMM(reg_aux_zp_comp_a, X_SP, reg_aux_zp_comp_a_offs_);
        add_imm(reg_aux_zp_comp_a, reg_aux_zp_comp_a,
                (is_tail) ? zp_comp_a_offset(1, true)
                          : zp_comp_a_offset(ld_block2),
                X_TMP_0);
        STR_IMM(reg_aux_zp_comp_a, X_SP, reg_aux_zp_comp_a_offs_);
    }
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        LDR_IMM(reg_aux_zp_c_values, X_SP, reg_aux_zp_c_values_offs_);
        add_imm(reg_aux_zp_c_values, reg_aux_zp_c_values,
                (is_tail) ? zp_c_values_offset(1, true)
                          : zp_c_values_offset(ld_block2),
                X_TMP_0);
        STR_IMM(reg_aux_zp_c_values, X_SP, reg_aux_zp_c_values_offs_);
    }
}

void jit_brgemm_kernel_t::advance_bd_block2_post_op_regs(int bd_block2) {
    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        LDR_IMM(reg_zp_comp_b, X_SP, reg_zp_comp_b_offs_);
        add_imm(reg_zp_comp_b, reg_zp_comp_b, bdb_zp_comp_b_offset(bd_block2),
                X_TMP_0);
        STR_IMM(reg_zp_comp_b, X_SP, reg_zp_comp_b_offs_);
    }
}

void jit_brgemm_kernel_t::copy_post_ops_stack_values_to_aux(bool is_reg_tail) {
    if (!is_reg_tail) {
        mov(reg_aux_C, reg_C);
        mov(reg_aux_D, reg_D);
        eor(reg_b_offset, reg_b_offset, reg_b_offset);
        if (brg.with_bias) {
            LDR_IMM(reg_bias, X_SP, reg_bias_offs_);
            STR_IMM(reg_bias, X_SP, reg_aux_bias_offs_);
        }
        if (brg.req_s8s8_compensation) {
            LDR_IMM(reg_compensation, X_SP, reg_comp_offs_);
            STR_IMM(reg_compensation, X_SP, reg_aux_comp_offs_);
        }
        if (brg.with_scales) {
            LDR_IMM(reg_scales, X_SP, reg_scales_offs_);
            STR_IMM(reg_scales, X_SP, reg_aux_scales_offs_);
        }

        if (brg.zp_type_a != brgemm_broadcast_t::none) {
            LDR_IMM(reg_zp_comp_a, X_SP, reg_zp_comp_a_offs_);
            STR_IMM(reg_zp_comp_a, X_SP, reg_aux_zp_comp_a_offs_);
        }

        if (brg.zp_type_c != brgemm_broadcast_t::none) {
            LDR_IMM(reg_zp_c_values, X_SP, reg_zp_c_values_offs_);
            STR_IMM(reg_zp_c_values, X_SP, reg_aux_zp_c_values_offs_);
        }
    }
    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        LDR_IMM(reg_zp_comp_b, X_SP, reg_zp_comp_b_offs_);
        STR_IMM(reg_zp_comp_b, X_SP, reg_aux_zp_comp_b_offs_);
    }
}

void jit_brgemm_kernel_t::read_params() {
    Label label_done;

    if (brg.with_binary) { STR_IMM(param1, X_SP, abi_param1_offs_); }

    if (brg.type == brgemm_addr) {
        LDR_IMM(reg_addr_batch, param1, GET_OFF(batch));
    } else {
        if (brg.layout == brgemm_row_major) {
            LDR_IMM(reg_A, param1, GET_OFF(ptr_A));
            LDR_IMM(reg_B, param1, GET_OFF(ptr_B));
        } else {
            LDR_IMM(reg_A, param1, GET_OFF(ptr_B));
            LDR_IMM(reg_B, param1, GET_OFF(ptr_A));
        }

        if (brg.type == brgemm_offs) {
            LDR_IMM(reg_offs_batch, param1, GET_OFF(batch));
            STR_IMM(reg_offs_batch, X_SP, origin_offs_batch_offs_);
        } else {
            LDR_IMM(reg_strd_batch, param1, GET_OFF(batch));
            STR_IMM(reg_strd_batch, X_SP, origin_strd_batch_offs_);
        }
    }

    ldr(reg_C, ptr(param1, GET_OFF(ptr_C)));
    ldr(reg_D, ptr(param1, GET_OFF(ptr_D)));
    ldr(reg_BS, ptr(param1, GET_OFF(BS)));

    // ptr_buf is re-used for passing compensations for
    // brg.req_s8s8_compensation case
    if (brg.req_s8s8_compensation) {
        ldr(reg_buf, ptr(param1, GET_OFF(ptr_buf)));
        str(reg_buf, ptr(X_SP, reg_buf_offs_));
    }

    if (brg.with_bias) {
        ldr(reg_bias, ptr(param1, GET_OFF(ptr_bias)));
        str(reg_bias, ptr(X_SP, reg_bias_offs_));
    }
    if (brg.with_scales) {
        ldr(reg_scales, ptr(param1, GET_OFF(ptr_scales)));
        str(reg_scales, ptr(X_SP, reg_scales_offs_));
    }

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        ldr(reg_zp_comp_a, ptr(param1, GET_OFF(a_zp_compensations)));
        str(reg_zp_comp_a, ptr(X_SP, reg_zp_comp_a_offs_));
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        ldr(reg_zp_comp_b, ptr(param1, GET_OFF(b_zp_compensations)));
        str(reg_zp_comp_b, ptr(X_SP, reg_zp_comp_b_offs_));
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        ldr(reg_zp_c_values, ptr(param1, GET_OFF(c_zp_values)));
        str(reg_zp_c_values, ptr(X_SP, reg_zp_c_values_offs_));
    }

    if (brg.with_dst_scales) {
        ldr(reg_dst_scales, ptr(param1, GET_OFF(ptr_dst_scales)));
        str(reg_dst_scales, ptr(X_SP, reg_dst_scales_offs_));
    }

    ldr(reg_do_post_ops, ptr(param1, GET_OFF(do_post_ops)));
    str(reg_do_post_ops, ptr(X_SP, reg_do_post_ops_offs_));

    ldr(reg_skip_accm, ptr(param1, GET_OFF(skip_accm)));
    str(reg_skip_accm, ptr(X_SP, reg_skip_accm_offs_));

    ldr(reg_zp_a_val, ptr(param1, GET_OFF(zp_a_val)));
    str(reg_zp_a_val, ptr(X_SP, reg_zp_a_val_offs_));

    ldr(reg_do_comp, ptr(param1, GET_OFF(do_apply_comp)));
    str(reg_do_comp, ptr(X_SP, reg_do_comp_offs_));
}

void jit_brgemm_kernel_t::zero_accumulators(int bd_block2, bool is_bdb_tail,
        int ld_block2, bool is_ld_tail, bool skip_accumulation) {
    int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
    const bool need_to_apply_beta = brg.beta != 0.f;
    for_(int bd = 0; bd < bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        auto zmm = accm(ld_block2, bd, ld);
        // This part is moved here from apply_alpha_beta function so that fadd instruction can be avoided.
        // This is also required only when K is blocked.
        if (need_to_apply_beta) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            const auto k_mask = is_tail ? ld_tail_mask : ld_full_mask;

            const int offset = C_offset(bd, ld);

            int base_offset = 0;
            auto x_addr = reg_aux_C;

            if ((unsigned)(offset - base_offset) > cpu_sveLen * 7) {
                add_imm(reg_tmp_, reg_aux_C, offset, X_TMP_0);
                base_offset = offset;
                x_addr = reg_tmp_;
            }
            LD_MUL_VL(ld1w, zmm.s, k_mask, x_addr, offset - base_offset, 4);

            const bool need_init_beta_vmm = brg.beta != 1.f;
            auto vmm_beta = z_tail_mask();
            if (need_init_beta_vmm) {
                auto wreg_tmp = WReg(reg_tmp_gpr.getIdx());
                mov_imm(wreg_tmp, float2int(static_cast<float>(brg.beta)));
                dup(vmm_beta.s, wreg_tmp);
                fmul(zmm.s, zmm.s, vmm_beta.s);
            }
        } else
            eor(zmm.d, zmm.d, zmm.d);
    }
}

void jit_brgemm_kernel_t::apply_alpha_beta(
        int bd_block, int ld_block2, bool is_ld_tail) {
    const bool apply_alpha = brg.alpha != 1.f;
    const bool dq2ps_required = brg.is_int8 && (apply_alpha || brg.beta != 1.f);

    auto vmm_alpha = z_tmp_1();
    if (apply_alpha) {
        auto wreg_tmp = WReg(reg_tmp_gpr.getIdx());
        mov_imm(wreg_tmp, float2int(static_cast<float>(brg.alpha)));
        dup(vmm_alpha.s, wreg_tmp);
    }
    for_(int bd = 0; bd < bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        auto vmm = accm(ld_block2, bd, ld);
        if (dq2ps_required) { scvtf(vmm.s, P_ALL_ONE / T_m, vmm.s); }
        if (apply_alpha) { fmul(vmm.s, vmm.s, vmm_alpha.s); }
    }
}

void jit_brgemm_kernel_t::apply_post_ops(
        int bd_block, int ld_block2, int ldb_and_bdb_offset, bool is_ld_tail) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

    const injector_utils::conditional_register_preserve_guard_t<sve_512>
            register_guard(brg.with_binary, this, {param1});
    const auto guard_space = register_guard.stack_space_occupied();
    if (brg.with_binary) {
        add_imm(X_DEFAULT_ADDR, X_SP, abi_param1_offs_ + guard_space, X_TMP_0);
        ldr(param1, ptr(X_DEFAULT_ADDR));

        if (with_binary_non_scalar_bcast_) {
            for_(int bd = 0; bd < bd_block; bd++)
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto vmm_idx = accm(ld_block2, bd, ld).getIdx();

                rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, reg_aux_D);
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        vmm_idx, D_offset(bd, ld));
                if (is_ld_tail) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
            }
        }
    }

    const auto sum_injector = [&] {
        const float *p_sum_scale = &brg.sum_scale;
        const int32_t *p_sum_zp = &brg.sum_zp;
        const bool p_sum_scale_reg_set = *p_sum_scale != 1.f;
        const bool p_sum_zp_reg_set = *p_sum_zp != 0;

        {
            const injector_utils::conditional_register_preserve_guard_t<sve_512>
                    register_guard_sum_scale((with_binary_non_scalar_bcast_)
                                    && p_sum_scale_reg_set,
                            this, {reg_ptr_sum_scale});
            const injector_utils::conditional_register_preserve_guard_t<sve_512>
                    register_guard_sum_zp(
                            p_sum_zp_reg_set, this, {reg_ptr_sum_zp});

            const auto &vmm_sum_zp = z_tmp_2();

            if (p_sum_zp_reg_set) {
                mov_imm(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
                ld1rw(z_tmp_3().s, P_ALL_ONE / T_z, ptr(reg_ptr_sum_zp));
                scvtf(vmm_sum_zp.s, P_ALL_ONE / T_m, z_tmp_3().s);
            }

            if (p_sum_scale_reg_set) {
                // embd bcast fma
                mov_imm(reg_ptr_sum_scale,
                        reinterpret_cast<size_t>(p_sum_scale));
            }

            for_(int bd = 0; bd < bd_block; bd++)
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto vmm = accm(ld_block2, bd, ld);
                const auto vmm_prev_dst = z_tmp_1();
                const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
                const auto k_mask = is_tail ? ld_tail_mask : ld_full_mask;
                add_imm(X_DEFAULT_ADDR, reg_aux_D, D_offset(bd, ld), X_TMP_0);
                ld1w(vmm_prev_dst.s, k_mask / T_z, ptr(X_DEFAULT_ADDR));
                if (p_sum_zp_reg_set)
                    fsub(vmm_prev_dst.s, vmm_prev_dst.s, vmm_sum_zp.s);
                if (p_sum_scale_reg_set) {
                    const auto ztmp2 = z_tmp_2();
                    ld1rw(ztmp2.s, P_ALL_ONE / T_z, ptr(reg_ptr_sum_scale));
                    fmla(vmm.s, P_ALL_ONE / T_m, vmm_prev_dst.s, ztmp2.s);
                } else
                    fadd(vmm.s, vmm.s, vmm_prev_dst.s);
            }
        }
    };

    if (brg.with_sum) {
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }

    postops_injector_->compute_vector_range(
            max_effective_vregs - bd_block * ld_block2, max_effective_vregs,
            rhs_arg_params);
}

static inline bool isa_has_masks(cpu_isa_t isa) {
    return is_superset(isa, sve_256);
}

void jit_brgemm_kernel_t::store_accumulators_apply_post_ops(
        int bd_block, int ld_block2, int ldb_and_bdb_offset, bool is_ld_tail) {

    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are already converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dq2ps_required = brg.is_int8
            && IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);

    if (brg.with_scales) {
        add_imm(X_DEFAULT_ADDR, X_SP, reg_aux_scales_offs_, X_TMP_0);
        ldr(reg_aux_scales, ptr(X_DEFAULT_ADDR));
        for (int ld = 0; ld < ld_block2; ld++) {
            const auto addr = X_DEFAULT_ADDR;
            add_imm(addr, reg_aux_scales, scales_offset(ld), X_TMP_0);
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            auto vmm_scales = z_tmp_1();
            if (IMPLICATION(is_tail, isa_has_masks(brg.isa_impl))) {
                ld1w(vmm_scales.s, k_mask / T_z, ptr(addr));
            } else {
                assert(!"Unreachable\n");
            }
            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (dq2ps_required) { scvtf(vmm.s, P_ALL_ONE / T_m, vmm.s); }
                fmul(vmm.s, vmm.s, vmm_scales.s);
            }
        }
    }

    if (brg.with_bias) { LDR_IMM(reg_aux_bias, X_SP, reg_aux_bias_offs_); }

    auto x_addr = reg_aux_bias;
    int base_offset = 0;
    for_(int ld = 0; ld < ld_block2; ld++)
    {
        auto zmm_bias = z_tmp_1();
        if (brg.with_bias) {
            const int offset = bias_offset(ld);
            if ((unsigned)(offset - base_offset) > cpu_sveLen * 7) {
                add_imm(reg_tmp_, reg_aux_bias, offset, X_TMP_0);
                base_offset = offset;
                x_addr = reg_tmp_;
            }
            cvt2ps(brg.dt_bias, zmm_bias, x_addr, true, false, k_mask, offset,
                    base_offset);
        }
        for (int bd = 0; bd < bd_block; bd++) {
            auto zmm = accm(ld_block2, bd, ld);
            if (dq2ps_required) { scvtf(zmm.s, P_ALL_ONE / T_m, zmm.s); }
            if (brg.with_bias) { fadd(zmm.s, zmm.s, zmm_bias.s); }
        }
    }

    if (postops_injector_)
        apply_post_ops(bd_block, ld_block2, ldb_and_bdb_offset, is_ld_tail);

    if (brg.with_dst_scales) {
        add_imm(X_DEFAULT_ADDR, X_SP, reg_dst_scales_offs_, X_TMP_0);
        ldr(reg_aux_dst_scales, ptr(X_DEFAULT_ADDR));
        auto vmm_dst_scales = z_tmp_1();
        ld1rw(vmm_dst_scales.s, P_ALL_ONE / T_z, ptr(reg_aux_dst_scales));

        for (int ld = 0; ld < ld_block2; ld++) {
            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                fmul(vmm.s, vmm.s, vmm_dst_scales.s);
            }
        }
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        add_imm(X_DEFAULT_ADDR, X_SP, reg_aux_zp_c_values_offs_, X_TMP_0);
        ldr(reg_aux_zp_c_values, ptr(X_DEFAULT_ADDR));
        auto vmm_zp_c = z_tmp_1();
        if (brg.zp_type_c == brgemm_broadcast_t::per_tensor) {
            add_imm(X_DEFAULT_ADDR, reg_aux_zp_c_values, 0, X_TMP_0);
            ldr(z_tmp_2(), ptr(X_DEFAULT_ADDR));
            scvtf(vmm_zp_c.s, k_mask / T_m, z_tmp_2().s);
        }
        for (int ld = 0; ld < ld_block2; ld++) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
                int zp_c_off = zp_c_values_offset(ld);
                add_imm(X_DEFAULT_ADDR, reg_aux_zp_c_values, zp_c_off, X_TMP_0);
                cvt2ps(data_type::s32, vmm_zp_c, X_DEFAULT_ADDR,
                        is_tail ? brg.ldb_tail : brg.ld_block, false, k_mask, 0,
                        0);
            }
            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                fadd(vmm.s, vmm.s, vmm_zp_c.s);
            }
        }
    }

    const bool dt_requires_saturation
            = one_of(brg.dt_d, data_type::u8, data_type::s8, data_type::s32);
    if (dt_requires_saturation) { assert(!"unsupported\n"); }

    x_addr = reg_aux_D;
    base_offset = 0;

    for (int bd = 0; bd < bd_block; bd++) {
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            const int offset = D_offset(bd, ld);
            if ((unsigned)(offset - base_offset) > cpu_sveLen * 7) {
                add_imm(reg_tmp_, reg_aux_D, offset, X_TMP_0);
                base_offset = offset;
                x_addr = reg_tmp_;
            }
            switch (brg.dt_d) {
                case data_type::f32:
                case data_type::s32:
                    ST_MUL_VL(st1w, zmm.s, k_mask, x_addr, offset - base_offset,
                            4);
                    break;
                case data_type::bf16: assert(!"unsupported\n"); break;
                case data_type::s8: assert(!"unsupported\n"); break;
                case data_type::u8: assert(!"unsupported\n"); break;
                default: assert(!"unknown dst_dt");
            }
        }
    }
}

void jit_brgemm_kernel_t::apply_compensation(
        int bd_block, int ld_block2, bool is_ld_tail) {
    // apply compensation to accumulated values
    // to avoid the loss of accuracy when converting s32 to f32
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

    if (!brg.req_cal_comp_pads && brg.zp_type_a != brgemm_broadcast_t::none) {
        auto vmm_zp_a_val = z_tmp_2();
        add_imm(X_DEFAULT_ADDR, X_SP, reg_zp_a_val_offs_, X_TMP_0);
        ldr(reg_zp_a_val, ptr(X_DEFAULT_ADDR));
        ldr(W_TMP_0, ptr(reg_zp_a_val));
        dup(vmm_zp_a_val.s, W_TMP_0);

        add_imm(X_DEFAULT_ADDR, X_SP, reg_aux_zp_comp_a_offs_, X_TMP_1);
        ldr(reg_aux_zp_comp_a, ptr(X_DEFAULT_ADDR));
        for (int ld = 0; ld < ld_block2; ld++) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            auto vmm_zp_comp_a = z_tmp_1();
            int zp_comp_a_off = zp_comp_a_offset(ld);
            // apply src zero points value to the accumulated values
            if (IMPLICATION(is_tail, isa_has_masks(brg.isa_impl))) {
                add_imm(X_DEFAULT_ADDR, reg_aux_zp_comp_a, zp_comp_a_off,
                        X_TMP_1);
                ld1w(vmm_zp_comp_a.s, k_mask / T_z, ptr(X_DEFAULT_ADDR));
            } else {
                assert(!"Unreachable\n");
            }
            mul(vmm_zp_comp_a.s, P_ALL_ONE / T_m, vmm_zp_a_val.s);

            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                add(vmm.s, vmm.s, vmm_zp_comp_a.s);
            }
        }
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        add_imm(X_DEFAULT_ADDR, X_SP, reg_aux_zp_comp_b_offs_, X_TMP_0);
        ldr(reg_aux_zp_comp_b, ptr(X_DEFAULT_ADDR));
        for (int bd = 0; bd < bd_block; bd++) {
            int zp_comp_b_off = zp_comp_b_offset(bd);
            for (int ld = 0; ld < ld_block2; ld++) {
                auto vmm = accm(ld_block2, bd, ld);
                ld1rw(vmm.s, P_ALL_ONE / T_z,
                        ptr(reg_aux_zp_comp_b, zp_comp_b_off));
            }
        }
    }

    if (!brg.req_cal_comp_pads && brg.req_s8s8_compensation) {
        ldr(reg_aux_compensation, ptr(X_SP, reg_aux_comp_offs_));
        for (int ld = 0; ld < ld_block2; ld++) {
            auto vmm_comp = z_tmp_1();
            int comp_offset = compensations_offset(ld);
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            if (IMPLICATION(is_tail, is_superset(brg.isa_impl, sve_512))) {
                const auto mask = is_tail ? k_mask : P_ALL_ONE;
                ld1w(vmm_comp.s, mask / T_z,
                        ptr(reg_aux_compensation, comp_offset));
            } else {
                not_(P_TMP.b, P_ALL_ONE, P_NOT_256.b);
                cmplt(P_TMP.s, P_TMP / T_z, z_tail_mask().s, 0);
                ld1w(vmm_comp.s, P_TMP / T_z,
                        ptr(reg_aux_compensation, comp_offset));
            }

            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                add(vmm.s, vmm.s, vmm_comp.s);
            }
        }
    }
}

void jit_brgemm_kernel_t::store_accumulators_without_post_ops(
        int bd_block, int ld_block2, bool is_ld_tail) {

    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dt_requires_saturation = brg.is_int8
            && !IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);
    if (dt_requires_saturation) { assert(!"unsupported\n"); }
    auto x_addr = reg_aux_C;
    int base_offset = 0;

    for (int bd = 0; bd < bd_block; bd++) {
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            const auto mask = is_ld_tail ? ld_tail_mask : P_ALL_ONE;
            const int offset = C_offset(bd, ld);

            if ((unsigned)(offset - base_offset) > cpu_sveLen * 7) {
                add_imm(reg_tmp_, reg_aux_C, offset, X_TMP_0);
                base_offset = offset;
                x_addr = reg_tmp_;
            }
            ST_MUL_VL(st1w, zmm.s, mask, x_addr, offset - base_offset, 4);
        }
    }
}

void jit_brgemm_kernel_t::store_accumulators(int bd_block2, bool is_bdb_tail,
        int ld_block2, bool is_ld_tail, bool skip_accumulation) {
    const bool has_zero_points = !everyone_is(brgemm_broadcast_t::none,
            brg.zp_type_a, brg.zp_type_b, brg.zp_type_c);
    const bool are_post_ops_applicable = one_of(true, brg.with_eltwise,
            brg.with_binary, brg.with_scales, brg.with_bias, brg.with_sum,
            brg.dt_d != brg.dt_c, brg.req_s8s8_compensation, has_zero_points);
    const bool need_to_apply_alpha_beta = brg.beta != 0.f || brg.alpha != 1.f;
    int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;

    if (brg.is_int8 && (brg.req_s8s8_compensation || has_zero_points)) {
        assert(!"unsupported\n");
    }

    if (need_to_apply_alpha_beta)
        apply_alpha_beta(bd_block, ld_block2, is_ld_tail);

    Label label_done;
    if (are_post_ops_applicable) {
        Label label_store_without_post_ops;

        LDR_IMM(reg_do_post_ops, X_SP, reg_do_post_ops_offs_);
        cmp_imm(reg_do_post_ops, 0, X_TMP_0);
        b(EQ, label_store_without_post_ops);
        store_accumulators_apply_post_ops(bd_block, ld_block2, 0, is_ld_tail);
        bl(label_done);

        L_aligned(label_store_without_post_ops);
    }
    store_accumulators_without_post_ops(bd_block, ld_block2, is_ld_tail);
    L_aligned(label_done);
}

void jit_brgemm_kernel_t::restore_A_B_matrices() {
    auto restore_reg_batch = brg.brgattr.max_bs > 1 || vpad_exist;
    if (brg.type == brgemm_addr) {
        if (restore_reg_batch) mov(reg_aux1_batch, reg_addr_batch);
    } else {
        mov(reg_aux1_A, reg_A);
        mov(reg_aux1_B, reg_B);

        if (restore_reg_batch) {
            if (brg.type == brgemm_offs) {
                ldr(reg_offs_batch, ptr(X_SP, origin_offs_batch_offs_));
            } else {
                ldr(reg_offs_batch, ptr(X_SP, origin_strd_batch_offs_));
            }
        }
    }
}

void jit_brgemm_kernel_t::set_A_B_matrices() {
    if (brg.type == brgemm_addr) {
        if (brg.brgattr.max_bs > 1) {
            if (brg.layout == brgemm_row_major) {
                ldr(reg_aux_A,
                        ptr(reg_aux1_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
                ldr(reg_aux_B,
                        ptr(reg_aux1_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
            } else {
                ldr(reg_aux_A,
                        ptr(reg_aux1_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
                ldr(reg_aux_B,
                        ptr(reg_aux1_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
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
            add_imm(reg_aux1_batch, reg_aux1_batch,
                    sizeof(brgemm_batch_element_t), X_TMP_0);
            prfm(PLDL1KEEP, ptr(reg_aux1_batch));
        }
    } else if (brg.type == brgemm_offs) {
        mov(reg_aux_A, reg_A);
        mov(reg_aux_B, reg_B);

        ldr(X_TMP_0, ptr(reg_offs_batch, GET_OFF_BATCH_ELEMENT(offset.A)));
        add(reg_aux_A, reg_aux_A, X_TMP_0);
        ldr(X_TMP_1, ptr(reg_offs_batch, GET_OFF_BATCH_ELEMENT(offset.B)));
        add(reg_aux_B, reg_aux_B, X_TMP_1);
        mov_imm(X_TMP_2, sizeof(brgemm_batch_element_t));
        add(reg_offs_batch, reg_offs_batch, X_TMP_2);
    } else if (brg.type == brgemm_strd) {
        mov(reg_aux_A, reg_aux1_A);
        mov(reg_aux_B, reg_aux1_B);

        mov_imm(reg_tmp_gpr, brg.stride_a);
        add(reg_aux1_A, reg_aux1_A, reg_tmp_gpr);
        mov_imm(reg_tmp_gpr, brg.stride_b);
        add(reg_aux1_B, reg_aux1_B, reg_tmp_gpr);
        if (vpad_exist) {
            ldr(reg_strd_batch, ptr(X_SP, origin_strd_batch_offs_));
            mov_imm(reg_strd_batch, sizeof(brgemm_batch_element_t));
            str(reg_strd_batch, ptr(X_SP, origin_strd_batch_offs_));
        }
    }

    add(reg_aux_A, reg_aux_A, reg_a_offset);
    add(reg_aux_B, reg_aux_B, reg_b_offset);
}

void jit_brgemm_kernel_t::dot_product(ZReg v1, ZReg v2, ZReg v3) {
    if (brg.is_f32) {
        fmla(v1.s, P_ALL_ONE / T_m, v2.s, v3.s);
    } else if (brg.is_bf16)
        assert(!"unsupported\n");
    else if (brg.is_int8)
        assert(!"unsupported\n");
    else
        assert(!"unsupported\n");
}

void jit_brgemm_kernel_t::compute_int8_compensation(int rd_loop, int bd_b,
        int bd_e, int bd_block, int ld_block2, bool is_ld_tail, int vpad) {
    assert(brg.is_int8);

    auto compensation_padding = [this, ld_block2](ZReg vmm_load, ZReg vmm_tmp,
                                        int ld, int bd_b, int bd_e) {
        // req_cal_comp_pads -> only calculate compensation along with
        // computation and do not use pre-calculated compensation.
        // Calculate comp padding as:
        // accum - inp_shift * conv(1, wei_s32)
        if (brg.req_s8s8_compensation) {
            if (brg.req_cal_comp_pads) {
                eor(vmm_tmp.d, vmm_tmp.d, vmm_tmp.d);
                dot_product(vmm_tmp, vmm_load, z_inp_shift());
            }

            for (int bd = bd_b; bd < bd_e; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (brg.req_cal_comp_pads) {
                    sub(vmm.s, vmm.s, vmm_tmp.s);
                } else {
                    dot_product(vmm, vmm_load, z_inp_shift());
                }
            }
        }

        if (brg.zp_type_a != brgemm_broadcast_t::none) {
            eor(vmm_tmp.d, vmm_tmp.d, vmm_tmp.d);
            dot_product(vmm_tmp, vmm_load, z_one_bytes());
            mul(vmm_tmp.s, P_ALL_ONE / T_m, z_inp_shift().s);

            for (int bd = bd_b; bd < bd_e; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (brg.req_cal_comp_pads) {
                    sub(vmm.s, vmm.s, vmm_tmp.s);
                } else {
                    add(vmm.s, vmm.s, vmm_tmp.s);
                }
            }
        }
    };

    if (n_bcast_1_load && brg.zp_type_a != brgemm_broadcast_t::none) {
        str(reg_bdb_loop, ptr(X_SP, reg_bdb_loop_offs_));
        const auto reg32_scratch = WReg(reg_zp_a_input_shift.getIdx());
        mov_imm(reg32_scratch, 0x1010101);
        dup(z_one_bytes().s, reg32_scratch);
        ldr(reg32_scratch, ptr(X_SP, reg_zp_a_val_offs_));
        dup(z_zp_a_shift().s, reg32_scratch);
        ldr(reg_bdb_loop, ptr(X_SP, reg_bdb_loop_offs_));
    }

    for_(int rd = 0; rd < rd_loop; rd += brg.rd_step)
    for (int ld = 0; ld < ld_block2; ++ld) {
        const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
        const auto mask = is_tail ? ld_tail_mask : P_ALL_ONE;
        ld1w(load().s, mask / T_z, ptr(reg_aux_B, B_offset(ld, rd)));

        if (brg.req_cal_comp_pads) {
            compensation_padding(load(), bcst(), ld, bd_b, bd_e);
        } else if (vpad != 0) {
            if (bd_b > 0) compensation_padding(load(), bcst(), ld, 0, bd_b);
            if (bd_e < bd_block)
                compensation_padding(load(), bcst(), ld, bd_e, bd_block);
        }
    }
}

void jit_brgemm_kernel_t::gemm_microkernel_sve512(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_rd_tail, bool is_ld_tail,
        int vpad, int rows_for_rd_tail) {
    MAYBE_UNUSED(bd_block2);
    int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
    const auto bd_b = nstl::max(0, vpad);
    const auto bd_e = nstl::min(bd_block, bd_block + vpad);
    const auto is_valid_bd
            = need_comp_pads && vpad != 0 ? bd_b <= bd_e : bd_b < bd_e;
    if (!is_valid_bd) return;

    bool is_emdbd = brg.embd_bcst;

    int rd_loop = 0, rd_tail_size = 0;
    if (is_rd_tail) {
        if (brg.is_bf16 || brg.is_int8) {
            assert(!"unsupported\n");
        } else
            rd_loop = brg.rdb_tail;
    } else
        rd_loop = brg.rd_block;

    auto broadcast = [=](const ZReg &z1, size_t offset, bool is_tail,
                             data_type_t dt) {
        if (is_tail) {
            eor(z1.d, z1.d, z1.d);
            auto xmm_tmp = z_tmp_1();
            add_imm(X_DEFAULT_ADDR, reg_aux_A, rd_tail_size * brg.typesize_A,
                    X_TMP_0);
            set_preg(P_TMP.b, offset);
            ld1b(xmm_tmp.b, P_TMP / T_z, ptr(X_DEFAULT_ADDR));
            dup(z1.s, xmm_tmp.s[0]);
        } else {
            if (dt == data_type::f32) {
                if (offset < (1 << 6)) {
                    ld1rw(z1.s, P_ALL_ONE / T_z,
                            ptr(reg_aux_A, (int32_t)offset));
                } else {
                    add_imm(X_DEFAULT_ADDR, reg_aux_A, offset, X_TMP_0);
                    ld1rw(z1.s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
                }
            } else if (dt == data_type::bf16) {
                assert(!"unsupported\n");
            } else if (one_of(dt, data_type::s8, data_type::u8)) {
                assert(!"unsupported\n");
            } else if (dt == data_type::f16) {
                assert(!"unsupported\n");
            }
        }

        if (brg.req_s8s8_compensation) assert(!"unsupported\n");
    };

    const bool comp_vpad = vpad != 0
            && (brg.req_s8s8_compensation
                    || brg.zp_type_a != brgemm_broadcast_t::none);
    if (brg.req_cal_comp_pads || comp_vpad) assert(!"unsupported\n");

    bool maybe_load_bytes
            = (rows_for_rd_tail > 0 || brg.brgattr.wary_A_k_tail_read)
            && is_rd_tail && rd_tail_size != 0 && (brg.is_bf16 || brg.is_int8);
    if (n_bcast_1_load) {
        for (int rd = 0; rd < rd_loop; rd += brg.rd_step) {
            bool have_to_load_bytes
                    = maybe_load_bytes && (rd == rd_loop - brg.rd_step);

            auto rows_by_load_bytes = have_to_load_bytes ? rows_for_rd_tail : 0;
            for (int bd = bd_b; bd < bd_e && !is_emdbd; bd++) {
                const auto bd_by_load_bytes = (bd >= bd_e - rows_by_load_bytes
                        || brg.brgattr.wary_A_k_tail_read);
                broadcast(bcst(bd), A_offset(bd, rd),
                        have_to_load_bytes && bd_by_load_bytes, brg.dt_a);
            }
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto addr = ptr(reg_aux_B, B_offset(ld, rd));
                const auto mask = is_ld_tail ? ld_tail_mask : P_ALL_ONE;
                if (brg.dt_b == data_type::f16) {
                    assert(!"unsupported\n");
                } else if (brg.dt_b == data_type::bf16
                        && brg.isa_impl == sve_256) {
                    assert(!"unsupported\n");
                } else if (is_ld_tail) {
                    ld1w(load().s, ld_tail_mask / T_z, addr);
                } else {
                    ld1w(load().s, P_ALL_ONE / T_z, addr);
                }
                for (int bd = bd_b; bd < bd_e; bd++) {
                    auto vmm = accm(ld_block2, bd, ld);
                    if (is_emdbd) {
                        if (A_offset(bd, rd) < (1 << 6)) {
                            ld1rw(load().s, mask / T_z,
                                    ptr(reg_aux_A, A_offset(bd, rd)));
                        } else {
                            add_imm(X_DEFAULT_ADDR, reg_aux_A, A_offset(bd, rd),
                                    X_TMP_0);
                            ld1rw(load().s, mask / T_z, ptr(X_DEFAULT_ADDR));
                        }
                        fmla(vmm.s, P_ALL_ONE / T_m, load(ld).s, load().s);
                    } else {
                        dot_product(vmm, load(), bcst(bd));
                    }
                }
            }
        }
    } else {
        auto x_addr = reg_aux_B;
        int base_offset = 0;

        for (int rd = 0; rd < rd_loop; rd += brg.rd_step) {
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto mask = is_ld_tail ? ld_tail_mask : P_ALL_ONE;
                if (brg.dt_b == data_type::f16) {
                    assert(!"unsupported\n");
                } else if (brg.dt_b == data_type::bf16
                        && brg.isa_impl == sve_256) {
                    assert(!"unsupported\n");
                } else {
                    const int offset = B_offset(ld, rd);
                    if ((unsigned)(offset - base_offset) > cpu_sveLen * 7) {
                        add_imm(reg_tmp_, reg_aux_B, offset, X_TMP_0);
                        base_offset = offset;
                        x_addr = reg_tmp_;
                    }
                    LD_MUL_VL(ld1w, load(ld).s, mask, x_addr,
                            offset - base_offset, 4);
                }
            }

            bool have_to_load_bytes
                    = maybe_load_bytes && (rd == rd_loop - brg.rd_step);

            auto rows_by_load_bytes = have_to_load_bytes ? rows_for_rd_tail : 0;
            for (int bd = bd_b; bd < bd_e; bd++) {
                if (!is_emdbd) {
                    const auto bd_by_load_bytes
                            = (bd >= bd_e - rows_by_load_bytes
                                    || brg.brgattr.wary_A_k_tail_read);
                    broadcast(bcst(), A_offset(bd, rd),
                            have_to_load_bytes && bd_by_load_bytes, brg.dt_a);
                }
                //The current implementaion of prefetch is not giving any gain in performance but is rather introducing some latency. Therefore it is removed util a new useful implementation is deviced.
                for (int ld = 0; ld < ld_block2; ld++) {
                    auto zmm = accm(ld_block2, bd, ld);
                    if (is_emdbd) {
                        if (A_offset(bd, rd) < (1 << 6)) {
                            ld1rw(z_tmp_1().s, P_ALL_ONE / T_z,
                                    ptr(reg_aux_A, A_offset(bd, rd)));
                        } else {
                            add_imm(X_DEFAULT_ADDR, reg_aux_A, A_offset(bd, rd),
                                    X_TMP_0);
                            ld1rw(z_tmp_1().s, P_ALL_ONE / T_z,
                                    ptr(X_DEFAULT_ADDR));
                        }
                        fmla(zmm.s, P_ALL_ONE / T_m, load(ld).s, z_tmp_1().s);
                    } else {
                        dot_product(zmm, load(ld), bcst());
                    }
                }
            }
        }
    }
}

void jit_brgemm_kernel_t::ldb_loop(int bd_block2, bool is_bdb_tail,
        int ld_block2, int ldb_loop_length, bool is_reg_tail, bool is_ld_tail,
        bool check_top_vpad, bool check_bottom_vpad, int rows_for_rd_tail,
        bool skip_accumulation) {

    Label ldb_loop_label;
    Label BS_loop_label;

    copy_post_ops_stack_values_to_aux(is_reg_tail);

    auto ld_loop_body = [=](int vpad) {
        set_A_B_matrices();

        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
        const auto bd_b = nstl::max(0, vpad);
        const auto bd_e = nstl::min(bd_block, bd_block + vpad);
        const auto is_valid_bd
                = need_comp_pads && vpad != 0 ? bd_b <= bd_e : bd_b < bd_e;
        if (!is_valid_bd) return;

        if (brg.rdb > 0) {
            Label rdb_loop_label;
            mov(reg_rdb_loop, brg.rdb);
            L_aligned(rdb_loop_label, 64);
            {
                const bool is_rd_tail = false;
                gemm_microkernel_sve512(bd_block2, is_bdb_tail, ld_block2,
                        is_rd_tail, is_ld_tail, vpad, rows_for_rd_tail);

                add_imm(reg_aux_A, reg_aux_A, rdb_A_offset(), X_TMP_0);
                add_imm(reg_aux_B, reg_aux_B, rdb_B_offset(), X_TMP_0);

                sub(reg_rdb_loop, reg_rdb_loop, 1);
                cmp_imm(reg_rdb_loop, 0, X_TMP_0);
            }
            b(GT, rdb_loop_label);
        }
        if (brg.rdb_tail != 0) {
            const bool is_rd_tail = true;

            gemm_microkernel_sve512(bd_block2, is_bdb_tail, ld_block2,
                    is_rd_tail, is_ld_tail, vpad, rows_for_rd_tail);
        }
    };
    if (is_ldb_loop_) { mov_imm(reg_ldb_loop, ldb_loop_length); }

    L_aligned(ldb_loop_label, 64);
    {
        zero_accumulators(bd_block2, is_bdb_tail, ld_block2, is_ld_tail,
                skip_accumulation);

        if (is_ldb_loop_) {
            STR_IMM(reg_D, X_SP, reg_D_offs_);
        } else {
            mov(reg_ldb_loop, reg_D);
        }
        if (brg.brgattr.max_bs > 1) {
            STR_IMM(reg_aux_D, X_SP, reg_aux_D_offs_);
        }

        if (brg.alpha != 0.f && !skip_accumulation) {
            restore_A_B_matrices();

            if (brg.req_s8s8_compensation) { assert(!"unsupported\n"); }
            if (need_comp_pads && brg.zp_type_a != brgemm_broadcast_t::none) {
                assert(!"unsupported\n");
            }

            if (brg.brgattr.max_bs > 1) { mov(reg_BS_loop, reg_BS); }
            L_aligned(BS_loop_label, 64);
            {
                if (check_top_vpad || check_bottom_vpad) {
                    const auto vpad_first = -brg.brgattr.max_bottom_vpad;
                    const auto vpad_last = brg.brgattr.max_top_vpad;
                    const auto n_vpads = vpad_last - vpad_first + 2;
                    constexpr auto MAX_N_VPADS = 2 * brgemm_desc_t::MAX_VPAD;
                    assert(n_vpads < MAX_N_VPADS);

                    Label Vpad_loop_end_label;
                    std::vector<Label> Vpad_loop_iter_label(MAX_N_VPADS);
                    if (vpad_exist) {
                        XReg reg_batch = (brg.type == brgemm_addr)
                                ? reg_aux1_batch
                                : ((brg.type == brgemm_offs) ? reg_offs_batch
                                                             : reg_strd_batch);
                        if (brg.type == brgemm_strd) {
                            LDR_IMM(reg_strd_batch, X_SP,
                                    origin_strd_batch_offs_);
                        }
                        ldr(reg_aux_A_vpad,
                                ptr(reg_batch,
                                        GET_OFF_BATCH_ELEMENT(vvpad.top)));

                        ldr(X_TMP_0,
                                ptr(reg_batch,
                                        GET_OFF_BATCH_ELEMENT(vvpad.bottom)));
                        sub(reg_aux_A_vpad, reg_aux_A_vpad, X_TMP_0);
                    } else {
                        eor(reg_aux_A_vpad, reg_aux_A_vpad, reg_aux_A_vpad);
                    }

                    for (int vpad = vpad_first; vpad <= vpad_last; vpad++) {
                        const auto label_vpad = vpad - vpad_first;
                        L(Vpad_loop_iter_label[label_vpad]);
                        if (!check_top_vpad && vpad > 0) continue;
                        if (!check_bottom_vpad && vpad < 0) continue;
                        auto real_vpad = vpad;
                        if (check_bottom_vpad && brg.bdb_tail) {
                            if (!is_bdb_tail) {
                                // for last full block before
                                // bdb_tail && -vpad greater than bdb_tail
                                if (brg.bdb_tail < -vpad)
                                    real_vpad += brg.bdb_tail;
                                else
                                    continue;
                            } else {
                                // for block with tail, call ldb_loop()
                                // to only calculate compensation for
                                // padding area when bdb_tail < -vpad for
                                // the cases using pre-cal compensation
                                if (brg.bdb_tail < -vpad && need_comp_pads
                                        && !brg.req_cal_comp_pads)
                                    real_vpad = -brg.bdb_tail;
                            }
                        }
                        cmp_imm(reg_aux_A_vpad, vpad, X_TMP_0);
                        b(NE, Vpad_loop_iter_label[label_vpad + 1]);
                        ld_loop_body(real_vpad);
                        b(Vpad_loop_end_label);
                    }
                    L(Vpad_loop_iter_label[n_vpads - 1]);
                    ld_loop_body(0);
                    L(Vpad_loop_end_label);
                } else {
                    ld_loop_body(0);
                }
                if (brg.brgattr.max_bs > 1) {
                    sub(reg_BS_loop, reg_BS_loop, 1);
                    cmp_imm(reg_BS_loop, 0, X_TMP_0);
                    b(GT, BS_loop_label);
                }
            }
        }

        if (is_ldb_loop_) {
            LDR_IMM(reg_D, X_SP, reg_D_offs_);
        } else {
            mov(reg_D, reg_ldb_loop);
        }
        if (brg.brgattr.max_bs > 1) {
            LDR_IMM(reg_aux_D, X_SP, reg_aux_D_offs_);
        }

        store_accumulators(bd_block2, is_bdb_tail, ld_block2, is_ld_tail,
                skip_accumulation);
        if (is_ldb_loop_) {
            if (!is_ld_tail) {
                ldb_regs_shift(ld_block2);
            } else {
                ldb_regs_shift(1, true);
            }
            sub(reg_ldb_loop, reg_ldb_loop, 1);
            cmp_imm(reg_ldb_loop, 0, X_TMP_0);
            b(GT, ldb_loop_label);
        }
    }
}

void jit_brgemm_kernel_t::bdb_loop() {
    auto do_ldb_loop = [=](int bd_block2, bool is_bdb_tail, bool check_top_vpad,
                               bool check_bottom_vpad, int rows_for_rd_tail,
                               bool skip_accumulation) {
        if (brg.ldb2 > 0) {
            const bool is_ld_reg_tail = false;
            const bool is_ld_tail = false;
            ldb_loop(bd_block2, is_bdb_tail, brg.ld_block2, brg.ldb2,
                    is_ld_reg_tail, is_ld_tail, check_top_vpad,
                    check_bottom_vpad, rows_for_rd_tail, skip_accumulation);
        }
        if (brg.ldb2_tail > 0) {
            const bool is_ld_reg_tail = (brg.ldb2 == 0) ? false : true;
            const bool is_ld_tail = false;
            ldb_loop(bd_block2, is_bdb_tail, brg.ldb2_tail, 1, is_ld_reg_tail,
                    is_ld_tail, check_top_vpad, check_bottom_vpad,
                    rows_for_rd_tail, skip_accumulation);
        }
        if (brg.ldb_tail > 0) {
            const bool is_ld_reg_tail
                    = (brg.ldb2 == 0 && brg.ldb2_tail == 0) ? false : true;
            const bool is_ld_tail = true;
            ldb_loop(bd_block2, is_bdb_tail, 1, 1, is_ld_reg_tail, is_ld_tail,
                    check_top_vpad, check_bottom_vpad, rows_for_rd_tail,
                    skip_accumulation);
        }
    };

    auto bdb_loop_body = [=](int bd_block2, bool is_bdb_tail,
                                 bool check_top_vpad, bool check_bottom_vpad,
                                 int rows_for_rd_tail, bool skip_accumulation) {
        do_ldb_loop(bd_block2, is_bdb_tail, check_top_vpad, check_bottom_vpad,
                rows_for_rd_tail, skip_accumulation);

        add_imm(reg_C, reg_C, bdb_C_offset(bd_block2), X_TMP_0);
        add_imm(reg_D, reg_D, bdb_D_offset(bd_block2), X_TMP_0);
        add_imm(reg_a_offset, reg_a_offset, bdb_A_offset(bd_block2), X_TMP_0);

        advance_bd_block2_post_op_regs(bd_block2);
    };

    int rows_for_rd_tail, bd_blocks_for_rd_tail;

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

    auto ld_block2 = (brg.ldb2 > 0) ? brg.ld_block2
                                    : ((brg.ldb2_tail > 0) ? brg.ldb2_tail : 1);
    const int free_vregs = max_effective_vregs - brg.req_s8s8_compensation;
    n_bcast_1_load = brg.is_int8
            && ((brg.bd_block * (ld_block2 + 1) < free_vregs)
                    && (bd_blocks_for_rd_tail == 0) && (rows_for_rd_tail == 0));
    // loop order may be specified in brgemm attributes
    if (brg.brgattr.hint_loop_order != brgemm_lo_default)
        n_bcast_1_load = (brg.brgattr.hint_loop_order == brgemm_lo_bl_1load)
                ? true
                : false;

    auto bdb_loop_sve512 = [=](bool skip_accumulation) {
        Label bdb_loop_end_label, no_vpad_label;
        if (vpad_exist) {
            // max_top_vp is restricted by bd_block due to
            // brgemm_kernel implementation. TODO: remove this restriction
            assert(brg.brgattr.max_top_vpad <= brg.bd_block
                    && brg.brgattr.max_bottom_vpad <= brg.bd_block);

            if (brg.type == brgemm_strd) {
                // if batch is nullptr then it means no vpadding in this call
                cmp_imm(reg_offs_batch, 0, X_TMP_0);
                b(EQ, no_vpad_label);
            }

            // first bd_block --------------
            auto bdblocks = brg.bdb;
            if (bdblocks >= 1) {
                bdb_loop_body(1, false, true,
                        (brg.bcast_dim - brg.brgattr.max_bottom_vpad)
                                < brg.bd_block,
                        brg.bdb - bd_blocks_for_rd_tail > 0 ? 0
                                                            : rows_for_rd_tail,
                        skip_accumulation);
                bdblocks--;
            }
            if (bdblocks > 1) {
                // middle bd_blocks -----------
                Label bdb_loop_label;
                mov_imm(reg_bdb_loop, bdblocks);
                L_aligned(bdb_loop_label, 64);
                {
                    bdb_loop_body(1, false, false, false,
                            bd_blocks_for_rd_tail <= 1 ? 0 : rows_for_rd_tail,
                            skip_accumulation);

                    sub(reg_bdb_loop, reg_bdb_loop, 1);
                    cmp_imm(reg_bdb_loop, 1, X_TMP_0);
                    b(GT, bdb_loop_label);
                }
                bdblocks = 1;
            }
            if (bdblocks == 1) {
                // last bd_block ------------
                bdb_loop_body(1, false, false, true,
                        bd_blocks_for_rd_tail == 0 ? 0 : rows_for_rd_tail,
                        skip_accumulation);
            }
            if (brg.bdb_tail > 0)
                do_ldb_loop(1, true, brg.bdb < 1, true, rows_for_rd_tail,
                        skip_accumulation);
            // for brgemm_strd "no vpadding" case may be implemented, so skip it
            if (brg.type == brgemm_strd) /*jmp(bdb_loop_end_label);*/
                b(bdb_loop_end_label);
        }
        if (!vpad_exist || brg.type == brgemm_strd) {
            // for brgemm_strd batch may be null so we need this code path
            L_aligned(no_vpad_label, 64);
            if (brg.bdb > 0) {
                mov_imm(reg_bdb_loop, brg.bdb);
                if (brg.bdb > (rows_for_rd_tail ? 1 : 0)) {
                    Label bdb_loop_label;
                    L_aligned(bdb_loop_label, 64);
                    {
                        bdb_loop_body(1, false, false, false,
                                bd_blocks_for_rd_tail <= 1 ? 0
                                                           : rows_for_rd_tail,
                                skip_accumulation);
                        sub(reg_bdb_loop, reg_bdb_loop, 1);
                        cmp_imm(reg_bdb_loop, rows_for_rd_tail ? 1 : 0,
                                X_TMP_0);
                        b(GT, bdb_loop_label);
                    }
                }

                if (rows_for_rd_tail)
                    bdb_loop_body(1, false, false, true,
                            bd_blocks_for_rd_tail == 0 ? 0 : rows_for_rd_tail,
                            skip_accumulation);
            }
            if (brg.bdb_tail > 0)
                do_ldb_loop(1, true, false, false, rows_for_rd_tail,
                        skip_accumulation);
        }
        L_aligned(bdb_loop_end_label, 64);
    };

    auto bdb_loop_general = [=](bool skip_accumulation) {
        if (brg.type == brgemm_addr && brg.brgattr.max_bs == 1 && !vpad_exist
                && !skip_accumulation) {
            ldr(reg_aux1_A, ptr(reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
            ldr(reg_aux1_B, ptr(reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
        }

        eor(reg_a_offset, reg_a_offset, reg_a_offset);
        bdb_loop_sve512(skip_accumulation);
    };

    if (brg.brgattr.generate_skip_accumulation) {
        Label bdb_loop_skip_acc_label, bdb_loop_done_label;
        LDR_IMM(reg_skip_accm, X_SP, reg_skip_accm_offs_);
        cmp_imm(reg_skip_accm, 0, X_TMP_0);
        b(NE, bdb_loop_skip_acc_label);

        bdb_loop_general(false);
        b(bdb_loop_done_label);

        L_aligned(bdb_loop_skip_acc_label, 64);
        bdb_loop_general(true);

        L_aligned(bdb_loop_done_label, 64);
    } else
        bdb_loop_general(false);
}

void jit_brgemm_kernel_t::generate() {
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
        set_preg(ld_full_mask.b, simd_w_ * 4, X_TMP_0, X_TMP_1);
    } else
        ptrue(ld_full_mask.b);

    mov(x7, x0);
    mov(x6, x1);
    mov(x2, x2);
    mov(x1, x3);
    mov(x8, x4);
    mov(x9, x5);

    sub_imm(X_SP, X_SP, stack_space_needed_,
            X_TMP_0); //rsp=X_SP

    vpad_exist
            = (brg.brgattr.max_top_vpad > 0 || brg.brgattr.max_bottom_vpad > 0)
            ? true
            : false;
    need_comp_pads = IMPLICATION(brg.zp_type_a == brgemm_broadcast_t::none,
                             brg.req_s8s8_compensation)
            && IMPLICATION(!vpad_exist, brg.req_cal_comp_pads);

    set_preg(ld_tail_mask.s, brg.ldb_tail, X_TMP_0, X_TMP_1);
    if (brg.is_int8 && !brg.has_int8_vnni) { assert(!"unsupported\n"); }

    read_params();

    bdb_loop();

    add_imm(X_SP, X_SP, stack_space_needed_, X_TMP_0);

    postamble();

    if (brg.with_eltwise) postops_injector_->prepare_table();
}

brgemm_attr_t::brgemm_attr_t()
    : max_bs(INT_MAX)
    , max_top_vpad(0)
    , max_bottom_vpad(0)
    , hint_expected_A_size(platform::get_per_core_cache_size(1))
    , hint_expected_B_size(platform::get_per_core_cache_size(1))
    , hint_expected_C_size(platform::get_per_core_cache_size(1))
    , hint_innermost_loop(brgemm_ld_loop_innermost)
    , hint_loop_order(brgemm_kernel_loop_order_t::brgemm_lo_default)
    , hint_prefetching(brgemm_kernel_prefetching_t::brgemm_prf_default)
    , wary_A_k_tail_read(true)
    , extendable_k(false)
    , generate_skip_accumulation(false)
    , bd_mask_level(0)
    , use_uker(false)
    , use_interleave_stores(false)
    , LDA2(0)
    , LDB2(0)
    , LDC2_M(0)
    , LDC2_N(0)
    , bd_mask(nullptr)
    , static_offsets(nullptr) {}

brgemm_kernel_common_t::brgemm_kernel_common_t(const brgemm_desc_t abrd) {
    brgemm_kernel_ = new jit_brgemm_kernel_t(abrd);
}

status_t brgemm_kernel_common_t::create_kernel() {
    return brgemm_kernel_->create_kernel();
}

void brgemm_kernel_common_t::operator()(brgemm_kernel_params_t *params) const {
    (*brgemm_kernel_)(params);
}

const jit_generator *brgemm_kernel_common_t::get_jit_generator() const {
    return brgemm_kernel_;
}

brgemm_kernel_common_t::~brgemm_kernel_common_t() {
    delete brgemm_kernel_;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
