/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
#include <vector>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;
template <cpu_isa_t isa, typename Wmm>
struct jit_brgemm_kernel_t : public jit_generator {
    jit_brgemm_kernel_t(const brgemm_t &abrg)
        : jit_generator(jit_name(), nullptr, MAX_CODE_SIZE, true, abrg.isa_impl)
        , brg(abrg)
        , postops_injector_(nullptr)
        , max_effective_vregs(
                  max_vregs - (brg.is_int8 && !brg.has_int8_vnni ? 2 : 0)) {

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
                    static_cast<size_t>(vmm_tmp(0).getIdx()), this->r14,
                    this->r15, this->r13, preserve_gpr, preserve_vmm,
                    GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(data_C_ptr_),
                    dst_md_wrapper, static_cast<size_t>(brg.ldb_tail),
                    ld_tail_mask, use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {
                    this->param1, enabled_bcast_strategy, rhs_sp};

            postops_injector_ = utils::make_unique<po_injector_t>(
                    this, brg.attr->post_ops_, bsp);

            with_binary_non_scalar_bcast_ = binary_injector::
                    any_binary_postop_rhs_non_scalar_broadcast(
                            brg.attr->post_ops_, dst_md_wrapper);
        }
        if (brg.is_bf16_emu)
            bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                    bf16_emu_reserv_1(), bf16_emu_reserv_2(),
                    bf16_emu_reserv_3(), bf16_emu_scratch, bf16_emu_reserv_4(),
                    bf16_emu_reserv_4());
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_t)

    brgemm_t brg;

private:
    using Vmm =
            typename utils::conditional<std::is_same<Wmm, Xbyak::Tmm>::value,
                    Xbyak::Zmm, Wmm>::type;
    using Vmm_lower_t = typename vreg_traits<Vmm>::Vmm_lower_t;
    static constexpr cpu_isa_t po_isa_t = utils::map(isa, avx512_core,
            avx512_core_amx_fp16, avx512_core_fp16, avx512_core_amx,
            avx512_core_fp16, avx512_core_fp16, avx512_core_fp16, avx2_vnni_2,
            avx2_vnni_2, avx2_vnni, avx2, avx2, avx2);
    using po_injector_t = injector::jit_uni_postops_injector_t<po_isa_t, Vmm>;
    std::unique_ptr<po_injector_t> postops_injector_;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    Xbyak::Label avx_tail_mask_;
    Xbyak::Label sum_zp_scale_data_;
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
    const reg64_t reg_zp_a_input_shift = reg_bdb_loop;

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
    const reg64_t reg_dst_scales = reg_rdb_loop;
    const reg64_t reg_zp_comp_a = reg_rdb_loop;
    const reg64_t reg_aux_zp_comp_a = reg_rdb_loop;
    const reg64_t reg_zp_comp_b = reg_rdb_loop;
    const reg64_t reg_aux_zp_comp_b = reg_rdb_loop;
    const reg64_t reg_zp_c_values = reg_rdb_loop;
    const reg64_t reg_aux_zp_c_values = reg_rdb_loop;

    const reg64_t reg_aux_scales = reg_aux_B;
    const reg64_t reg_aux_dst_scales = reg_aux_B;
    const reg64_t reg_do_post_ops = reg_rdb_loop;
    const reg64_t reg_do_comp = reg_rdb_loop;
    const reg64_t reg_skip_accm = reg_rdb_loop;
    const reg64_t reg_tmp_gpr = reg_rdb_loop;
    const reg64_t reg_ptr_sum_scale = reg_rdb_loop;
    const reg64_t reg_ptr_sum_zp = reg_bdb_loop;
    const reg64_t reg_zp_a_val = reg_rdb_loop;

    const reg64_t reg_buf = reg_rdb_loop;
    const reg64_t reg_compensation = reg_bias;
    const reg64_t reg_aux_compensation = reg_aux_bias;

    const reg64_t reg_D = reg_aux_A;
    const reg64_t reg_aux_D = reg_BS_loop;

    /* bf16 emulation */
    const reg64_t bf16_emu_scratch = reg_rdb_loop;

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
    constexpr static int max_vregs = cpu_isa_traits<po_isa_t>::n_vregs;
    const int max_effective_vregs;

    Xbyak::Opmask ld_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask ld_tail_mask = Xbyak::Opmask(3);

    Vmm accm(int ld_block, int bd, int ld) {
        return Vmm(max_effective_vregs - 1 - (bd * ld_block + ld));
    }

    Vmm bcst(int bd = 0) {
        if (n_bcast_1_load) {
            int idx = max_effective_vregs - 1 - (brg.ld_block2 * brg.bd_block)
                    - bd;
            assert(idx > 0);
            return Vmm(idx);
        } else
            return Vmm(0);
    }

    Vmm load(int ld = 0) {
        if (n_bcast_1_load) {
            return Vmm(0);
        } else {
            int idx = max_effective_vregs - 1 - (brg.ld_block2 * brg.bd_block)
                    - ld;
            assert(idx > 0);
            return Vmm(idx);
        }
    }

    Vmm vmm_tmp(int i) {
        assert(IMPLICATION(!brg.is_tmm,
                i >= 0
                        && i < max_effective_vregs
                                        - brg.bd_block * brg.ld_block2));
        return Vmm(i);
    }

    Vmm vmm_tail_mask() { return vmm_tmp(1); }
    Vmm vmm_one_bytes() const noexcept { return Vmm(3); }
    Vmm vmm_zp_a_shift() const noexcept { return Vmm(2); }
    Vmm vmm_inp_shift() const noexcept { return Vmm(1); }

    /* bf16 emulation */
    Zmm bf16_emu_reserv_1() const noexcept { return Zmm(0); }
    Zmm bf16_emu_reserv_2() const noexcept { return Zmm(1); }
    Zmm bf16_emu_reserv_3() const noexcept { return Zmm(2); }
    Zmm bf16_emu_reserv_4() const noexcept { return Zmm(3); }
    // note: zmm reserv_5 is not necessary since it's only used for 'vdpbf16ps'

    // Required in every dot product for INT8 non-VNNI computation.
    Vmm int8_ones_words() const noexcept { return Vmm(max_vregs - 1); }
    Vmm int8_dot_product_temp() const noexcept { return Vmm(max_vregs - 2); }

    Vmm vmm_mask(const Vmm vmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) const;
    Vmm_lower_t vmm_lower_mask(const Vmm_lower_t vmm_lower_in, bool mask_flag,
            bool store, Xbyak::Opmask ktail_mask) const;
    void maybe_set_avx_mask(bool is_ld_tail);

    void cvt2ps(data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
            bool mask_flag, bool store, Xbyak::Opmask ktail_mask,
            int tail_size);

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

    void dot_product(Vmm v1, Vmm v2, Vmm v3);
    void gemm_microkernel(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_rd_tail, bool is_ld_tail, int vpad, int rows_for_rd_tail);
    void gemm_microkernel_amx(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_rd_tail, bool is_ld_tail);

    void ldb_loop(int bd_block2, bool is_bdb_tail, int ld_block,
            int ldb_loop_length, bool is_reg_tail, bool is_ld_tail,
            bool check_top_vpad, bool check_bottom_vpad, int rows_for_rd_tail,
            bool skip_accumulation);
    void bdb_loop();

    void generate() override;

    int A_offset(int bd, int rd, bool is_amx = false) const noexcept;
    int B_offset(int ld, int rd, bool is_amx = false) const noexcept;
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
    int bd_compensation_offset(int ld, int bd) const noexcept;
    int scales_offset(int ld, bool is_tail = false) const noexcept;
    int zp_comp_a_offset(int ld, bool is_tail = false) const noexcept;
    int bd_zp_comp_a_offset(int ld, int bd) const noexcept;
    int bdb_zp_comp_a_offset(int bd_block2) const noexcept;
    int zp_comp_b_offset(int bd) const noexcept;
    int bdb_zp_comp_b_offset(int bd_block2) const noexcept;
    int zp_c_values_offset(int ld, bool is_tail = false) const noexcept;

    bool n_bcast_1_load = false;
    bool vpad_exist = false;
    bool need_comp_pads = false;
};

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::A_offset(
        int bd, int rd, bool is_amx) const noexcept {
    return (is_amx) ? brg.typesize_A * (bd * brg.bd_block * brg.LDA)
                    : brg.typesize_A * (bd * brg.LDA + rd);
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::B_offset(
        int ld, int rd, bool is_amx) const noexcept {
    if (is_amx) {
        return brg.typesize_B * (brg.rd_step * ld * brg.ld_block);
    } else {
        const int data_vnni_granularity = brg.ld_step;
        const int rdb0 = rd / data_vnni_granularity;
        // Note: Offsets for elements within vnni_granularity are expected to be
        // handled within gemm_microkernel (for ex: odd-even converts).
        // hence no `rd % data_vnni_granularity`
        return brg.typesize_B
                * (rdb0 * data_vnni_granularity * brg.LDB
                        + data_vnni_granularity * ld * brg.ld_block);
    }
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::C_offset(int bd, int ld) const noexcept {
    return brg.typesize_C * (bd * brg.LDC + ld * brg.ld_block);
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::D_offset(int bd, int ld) const noexcept {
    return brg.typesize_D * (bd * brg.LDD + ld * brg.ld_block);
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::po_offset(int bd, int ld) const noexcept {
    return bd * brg.LDD + ld * brg.ld_block;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::rdb_A_offset() const noexcept {
    return brg.typesize_A * brg.rd_block;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::rdb_B_offset() const noexcept {
    return brg.typesize_B * brg.rd_block * brg.LDB;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::ldb_B_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_B * brg.ldb_tail * brg.ld_step
                     : brg.typesize_B * ld_block2 * brg.ld_block * brg.ld_step;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::ldb_C_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_C * brg.ldb_tail
                     : brg.typesize_C * ld_block2 * brg.ld_block;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::ldb_D_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_D * brg.ldb_tail
                     : brg.typesize_D * ld_block2 * brg.ld_block;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::ldb_po_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.ldb_tail : ld_block2 * brg.ld_block;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::bdb_A_offset(int bd_block2) const noexcept {
    return brg.typesize_A * bd_block2 * brg.bd_block * brg.LDA;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::bdb_C_offset(int bd_block2) const noexcept {
    return brg.typesize_C * bd_block2 * brg.bd_block * brg.LDC;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::bdb_D_offset(int bd_block2) const noexcept {
    return brg.typesize_D * bd_block2 * brg.bd_block * brg.LDD;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::bdb_po_offset(int bd_block2) const noexcept {
    return bd_block2 * brg.bd_block * brg.LDD;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::bias_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_bias * brg.ldb_tail
                     : brg.typesize_bias * ld * brg.ld_block;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::oc_logical_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? brg.ldb_tail : ld * brg.ld_block;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::compensations_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                     : sizeof(int32_t) * ld * brg.ld_block;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::bdb_compensation_offset(
        int bd_block2) const noexcept {
    return sizeof(int32_t) * bd_block2 * brg.bd_block * brg.LDB;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::bd_compensation_offset(
        int ld, int bd) const noexcept {
    return sizeof(int32_t) * (ld * brg.ld_block + bd * brg.LDB);
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::scales_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? brg.is_oc_scale * sizeof(float) * brg.ldb_tail
                     : brg.is_oc_scale * sizeof(float) * ld * brg.ld_block;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::zp_comp_a_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                     : sizeof(int32_t) * ld * brg.ld_block;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::bdb_zp_comp_a_offset(
        int bd_block2) const noexcept {
    return sizeof(int32_t) * bd_block2 * brg.bd_block * brg.LDB;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::bd_zp_comp_a_offset(
        int ld, int bd) const noexcept {
    return sizeof(int32_t) * (ld * brg.ld_block + bd * brg.LDB);
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::zp_comp_b_offset(int bd) const noexcept {
    return sizeof(int32_t) * bd;
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::bdb_zp_comp_b_offset(
        int bd_block2) const noexcept {
    return zp_comp_b_offset(bd_block2 * brg.bd_block);
}

template <cpu_isa_t isa, typename Wmm>
int jit_brgemm_kernel_t<isa, Wmm>::zp_c_values_offset(
        int ld, bool is_tail) const noexcept {
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                         : sizeof(int32_t) * ld * brg.ld_block;
    }

    return 0;
}
template <cpu_isa_t isa, typename Wmm>
typename jit_brgemm_kernel_t<isa, Wmm>::Vmm
jit_brgemm_kernel_t<isa, Wmm>::vmm_mask(const Vmm vmm_in, bool mask_flag,
        bool store, Xbyak::Opmask ktail_mask) const {
    return mask_flag && is_superset(brg.isa_impl, avx512_core)
            ? (store ? vmm_in | ktail_mask : vmm_in | ktail_mask | T_z)
            : vmm_in;
}

template <cpu_isa_t isa, typename Wmm>
typename jit_brgemm_kernel_t<isa, Wmm>::Vmm_lower_t
jit_brgemm_kernel_t<isa, Wmm>::vmm_lower_mask(const Vmm_lower_t vmm_lower_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) const {
    return mask_flag && is_superset(brg.isa_impl, avx512_core)
            ? (store ? vmm_lower_in | ktail_mask
                     : vmm_lower_in | ktail_mask | T_z)
            : vmm_lower_in;
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::maybe_set_avx_mask(bool is_ld_tail) {
    if (IMPLICATION(is_ld_tail, isa_has_masks(brg.isa_impl))) return;
    mov(reg_tmp_gpr, avx_tail_mask_);
    vmovups(vmm_tail_mask(), ptr[reg_tmp_gpr]);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::cvt2ps(data_type_t type_in,
        const Vmm vmm_in, const Xbyak::Operand &op, bool mask_flag, bool store,
        Xbyak::Opmask ktail_mask, int tail_size) {
    Vmm vmm = vmm_in;
    const bool has_tail
            = op.isMEM() && tail_size != vreg_traits<Vmm>::vlen / sizeof(float);
    if (IMPLICATION(has_tail, is_superset(brg.isa_impl, avx512_core))) {
        vmm = vmm_mask(vmm_in, mask_flag, store, ktail_mask);
    } else {
        load_data(type_in, vmm_in, op.getAddress(), tail_size);
        if (types::is_integral_dt(type_in)) uni_vcvtdq2ps(vmm_in, vmm_in);
        return;
    }
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: uni_vmovups(vmm, op); break;
        case data_type::bf16:
            uni_vpmovzxwd(vmm, op);
            uni_vpslld(vmm, vmm, 16);
            break;
        case data_type::f16: vcvtph2ps(vmm, op); break;
        case data_type::s8: uni_vpmovsxbd(vmm, op); break;
        case data_type::u8: uni_vpmovzxbd(vmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (types::is_integral_dt(type_in)) uni_vcvtdq2ps(vmm_in, vmm_in);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::advance_ldb_post_op_regs() {
    if (brg.with_bias) {
        mov(reg_aux_bias, ptr[rsp + reg_aux_bias_offs_]);
        add(reg_aux_bias, bias_offset(1));
        mov(ptr[rsp + reg_aux_bias_offs_], reg_aux_bias);
    }
    if (brg.with_scales) {
        mov(reg_aux_scales, ptr[rsp + reg_aux_scales_offs_]);
        add(reg_aux_scales, scales_offset(1));
        mov(ptr[rsp + reg_aux_scales_offs_], reg_aux_scales);
    }
    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(reg_aux_zp_comp_a, ptr[rsp + reg_aux_zp_comp_a_offs_]);
        add(reg_aux_zp_comp_a, zp_comp_a_offset(1));
        mov(ptr[rsp + reg_aux_zp_comp_a_offs_], reg_aux_zp_comp_a);
    }
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        mov(reg_aux_zp_c_values, ptr[rsp + reg_aux_zp_c_values_offs_]);
        add(reg_aux_zp_c_values, zp_c_values_offset(1));
        mov(ptr[rsp + reg_aux_zp_c_values_offs_], reg_aux_zp_c_values);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::restore_ldb_post_op_regs(int ld_block2) {
    if (brg.with_bias) {
        mov(reg_aux_bias, ptr[rsp + reg_aux_bias_offs_]);
        sub(reg_aux_bias, bias_offset(ld_block2 - 1));
        mov(ptr[rsp + reg_aux_bias_offs_], reg_aux_bias);
    }
    if (brg.with_scales) {
        mov(reg_aux_scales, ptr[rsp + reg_aux_scales_offs_]);
        sub(reg_aux_scales, scales_offset(ld_block2 - 1));
        mov(ptr[rsp + reg_aux_scales_offs_], reg_aux_scales);
    }
    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(reg_aux_zp_comp_a, ptr[rsp + reg_aux_zp_comp_a_offs_]);
        sub(reg_aux_zp_comp_a, zp_comp_a_offset(ld_block2 - 1));
        mov(ptr[rsp + reg_aux_zp_comp_a_offs_], reg_aux_zp_comp_a);
    }
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        mov(reg_aux_zp_c_values, ptr[rsp + reg_aux_zp_c_values_offs_]);
        sub(reg_aux_zp_c_values, zp_c_values_offset(ld_block2 - 1));
        mov(ptr[rsp + reg_aux_zp_c_values_offs_], reg_aux_zp_c_values);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::advance_bdb_post_op_regs(int adj_bd_block) {
    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        mov(reg_aux_zp_comp_b, ptr[rsp + reg_aux_zp_comp_b_offs_]);
        add(reg_aux_zp_comp_b, bdb_zp_comp_b_offset(1));
        mov(ptr[rsp + reg_aux_zp_comp_b_offs_], reg_aux_zp_comp_b);
    }
    if (brg.req_comp_pads_with_bd
            && brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(reg_aux_zp_comp_a, ptr[rsp + reg_aux_zp_comp_a_offs_]);
        add(reg_aux_zp_comp_a, bdb_compensation_offset(1));
        mov(ptr[rsp + reg_aux_zp_comp_a_offs_], reg_aux_zp_comp_a);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::restore_bdb_post_op_regs(int bd_block2) {
    bool post_processed = false;
    if (bd_block2 > 1) {
        if (brg.zp_type_b != brgemm_broadcast_t::none) {
            post_processed = true;
            mov(reg_aux_zp_comp_b, ptr[rsp + reg_aux_zp_comp_b_offs_]);
            sub(reg_aux_zp_comp_b, bdb_zp_comp_b_offset(bd_block2 - 1));
            mov(ptr[rsp + reg_aux_zp_comp_b_offs_], reg_aux_zp_comp_b);
        }
        if (brg.req_comp_pads_with_bd
                && brg.zp_type_a != brgemm_broadcast_t::none) {
            mov(reg_aux_zp_comp_a, ptr[rsp + reg_aux_zp_comp_a_offs_]);
            sub(reg_aux_zp_comp_a, bdb_compensation_offset(bd_block2 - 1));
            mov(ptr[rsp + reg_aux_zp_comp_a_offs_], reg_aux_zp_comp_a);
        }
    }
    if (post_processed) mov(reg_buf, ptr[rsp + reg_buf_offs_]);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::ldb_regs_shift(
        int ld_block2, bool is_tail) {
    int C_offset = (is_tail) ? ldb_C_offset(1, true) : ldb_C_offset(ld_block2);
    int D_offset = (is_tail) ? ldb_D_offset(1, true) : ldb_D_offset(ld_block2);
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
                (is_tail) ? scales_offset(1, true) : scales_offset(ld_block2));
        mov(ptr[rsp + reg_aux_scales_offs_], reg_aux_scales);
    }
    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(reg_aux_zp_comp_a, ptr[rsp + reg_aux_zp_comp_a_offs_]);
        add(reg_aux_zp_comp_a,
                (is_tail) ? zp_comp_a_offset(1, true)
                          : zp_comp_a_offset(ld_block2));
        mov(ptr[rsp + reg_aux_zp_comp_a_offs_], reg_aux_zp_comp_a);
    }
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        mov(reg_aux_zp_c_values, ptr[rsp + reg_aux_zp_c_values_offs_]);
        add(reg_aux_zp_c_values,
                (is_tail) ? zp_c_values_offset(1, true)
                          : zp_c_values_offset(ld_block2));
        mov(ptr[rsp + reg_aux_zp_c_values_offs_], reg_aux_zp_c_values);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::advance_bd_block2_post_op_regs(
        int bd_block2) {
    if (brg.req_comp_pads_with_bd && brg.req_s8s8_compensation) {
        mov(reg_compensation, ptr[rsp + reg_comp_offs_]);
        add(reg_compensation, bdb_compensation_offset(bd_block2));
        mov(ptr[rsp + reg_comp_offs_], reg_compensation);
    }

    if (brg.req_comp_pads_with_bd
            && brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_a, ptr[rsp + reg_zp_comp_a_offs_]);
        add(reg_zp_comp_a, bdb_zp_comp_a_offset(bd_block2));
        mov(ptr[rsp + reg_zp_comp_a_offs_], reg_zp_comp_a);
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_b, ptr[rsp + reg_zp_comp_b_offs_]);
        add(reg_zp_comp_b, bdb_zp_comp_b_offset(bd_block2));
        mov(ptr[rsp + reg_zp_comp_b_offs_], reg_zp_comp_b);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::copy_post_ops_stack_values_to_aux(
        bool is_reg_tail) {
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

        if (brg.zp_type_a != brgemm_broadcast_t::none) {
            mov(reg_zp_comp_a, ptr[rsp + reg_zp_comp_a_offs_]);
            mov(ptr[rsp + reg_aux_zp_comp_a_offs_], reg_zp_comp_a);
        }

        if (brg.zp_type_c != brgemm_broadcast_t::none) {
            mov(reg_zp_c_values, ptr[rsp + reg_zp_c_values_offs_]);
            mov(ptr[rsp + reg_aux_zp_c_values_offs_], reg_zp_c_values);
        }
    }
    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_b, ptr[rsp + reg_zp_comp_b_offs_]);
        mov(ptr[rsp + reg_aux_zp_comp_b_offs_], reg_zp_comp_b);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::read_params() {
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
    if (brg.is_tmm || brg.req_s8s8_compensation) {
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

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_a, ptr[param1 + GET_OFF(a_zp_compensations)]);
        mov(ptr[rsp + reg_zp_comp_a_offs_], reg_zp_comp_a);
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_b, ptr[param1 + GET_OFF(b_zp_compensations)]);
        mov(ptr[rsp + reg_zp_comp_b_offs_], reg_zp_comp_b);
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        mov(reg_zp_c_values, ptr[param1 + GET_OFF(c_zp_values)]);
        mov(ptr[rsp + reg_zp_c_values_offs_], reg_zp_c_values);
    }

    if (brg.with_dst_scales) {
        mov(reg_dst_scales, ptr[param1 + GET_OFF(ptr_dst_scales)]);
        mov(ptr[rsp + reg_dst_scales_offs_], reg_dst_scales);
    }

    mov(reg_do_post_ops, ptr[param1 + GET_OFF(do_post_ops)]);
    mov(ptr[rsp + reg_do_post_ops_offs_], reg_do_post_ops);

    mov(reg_skip_accm, ptr[param1 + GET_OFF(skip_accm)]);
    mov(ptr[rsp + reg_skip_accm_offs_], reg_skip_accm);

    mov(reg_zp_a_val, ptr[param1 + GET_OFF(zp_a_val)]);
    mov(ptr[rsp + reg_zp_a_val_offs_], reg_zp_a_val);

    mov(reg_do_comp, ptr[param1 + GET_OFF(do_apply_comp)]);
    mov(ptr[rsp + reg_do_comp_offs_], reg_do_comp);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::zero_accumulators(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_ld_tail,
        bool skip_accumulation) {
    if (brg.is_tmm) {
        // avoid usage of tile registers if there is no accumulation
        if (skip_accumulation) return;
        for_(int bdb = 0; bdb < bd_block2; bdb++)
        for (int ldb = 0; ldb < ld_block2; ldb++) {
            int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
            tilezero(Tmm(brg.get_C_tensor(bdb, idx, is_bdb_tail, is_ld_tail)));
        }
    } else {
        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
        for_(int bd = 0; bd < bd_block; bd++)
        for (int ld = 0; ld < ld_block2; ld++) {
            auto vmm = accm(ld_block2, bd, ld);
            uni_vpxor(vmm, vmm, vmm);
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::apply_alpha_beta(
        int bd_block, int ld_block2, bool is_ld_tail) {
    const bool apply_alpha = brg.alpha != 1.f;
    const bool dq2ps_required = brg.is_int8 && (apply_alpha || brg.beta != 1.f);

    auto vmm_alpha = vmm_tmp(0);
    if (apply_alpha) {
        mov(reg_tmp_gpr, float2int(static_cast<float>(brg.alpha)));
        uni_vmovq(Xmm(vmm_alpha.getIdx()), reg_tmp_gpr);
        uni_vbroadcastss(vmm_alpha, Xmm(vmm_alpha.getIdx()));
    }
    for_(int bd = 0; bd < bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        auto vmm = accm(ld_block2, bd, ld);
        if (dq2ps_required) uni_vcvtdq2ps(vmm, vmm);
        if (apply_alpha) uni_vmulps(vmm, vmm, vmm_alpha);
    }

    if (brg.beta == 0.f) return;
    const bool use_vadd_for_beta = brg.beta == 1.f && !dq2ps_required;
    const bool need_init_beta_vmm = brg.beta != 1.f;
    auto vmm_prev_dst = vmm_tmp(0);
    auto vmm_beta = vmm_tail_mask();
    if (need_init_beta_vmm) {
        mov(reg_tmp_gpr, float2int(static_cast<float>(brg.beta)));
        uni_vmovq(Xmm(vmm_beta.getIdx()), reg_tmp_gpr);
        uni_vbroadcastss(vmm_beta, Xmm(vmm_beta.getIdx()));
    }

    for_(int bd = 0; bd < bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
        const auto k_mask = is_tail ? ld_tail_mask : ld_full_mask;
        auto vmm = accm(ld_block2, bd, ld);
        auto ptr_C = ptr[reg_aux_C + C_offset(bd, ld)];
        if (use_vadd_for_beta) {
            if (IMPLICATION(is_tail, is_superset(brg.isa_impl, avx512_core))) {
                auto vmm_masked = vmm_mask(vmm, is_tail, false, k_mask);
                if (brg.is_int8)
                    uni_vpaddd(vmm_masked, vmm, ptr_C);
                else
                    uni_vaddps(vmm_masked, vmm, ptr_C);
            } else {
                vmaskmovps(vmm_prev_dst, vmm_tail_mask(), ptr_C);
                if (brg.is_int8)
                    uni_vpaddd(vmm, vmm, vmm_prev_dst);
                else
                    uni_vaddps(vmm, vmm, vmm_prev_dst);
            }
        } else {
            const int ld_size = is_tail ? brg.ldb_tail : brg.ld_block;
            cvt2ps(brg.dt_c, vmm_prev_dst, ptr_C, is_tail, false, k_mask,
                    ld_size);
            if (brg.beta == 1.f)
                uni_vaddps(vmm, vmm, vmm_prev_dst);
            else
                uni_vfmadd231ps(vmm, vmm_prev_dst, vmm_beta);
        }
    }

    if (need_init_beta_vmm) maybe_set_avx_mask(is_ld_tail);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::apply_post_ops(
        int bd_block, int ld_block2, int ldb_and_bdb_offset, bool is_ld_tail) {

    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

    const injector_utils::conditional_register_preserve_guard_t register_guard(
            brg.with_binary, this, {param1});
    const auto guard_space = register_guard.stack_space_occupied();
    if (brg.with_binary) {
        mov(param1, ptr[rsp + abi_param1_offs_ + guard_space]);

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
        const bool reset_avx_tail_mask = p_sum_zp_reg_set;

        {
            const injector_utils::conditional_register_preserve_guard_t
                    register_guard_sum_scale((with_binary_non_scalar_bcast_)
                                    && p_sum_scale_reg_set,
                            this, {reg_ptr_sum_scale});
            const injector_utils::conditional_register_preserve_guard_t
                    register_guard_sum_zp(
                            p_sum_zp_reg_set, this, {reg_ptr_sum_zp});

            const auto &vmm_sum_zp = vmm_tmp(1);

            if (p_sum_zp_reg_set) {
                mov(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
                if (is_superset(brg.isa_impl, avx512_core)) {
                    vcvtdq2ps(vmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
                } else {
                    uni_vpbroadcastd(vmm_sum_zp, ptr[reg_ptr_sum_zp]);
                    uni_vcvtdq2ps(vmm_sum_zp, vmm_sum_zp);
                }
            }

            if (p_sum_scale_reg_set) {
                if (is_superset(brg.isa_impl, avx512_core)) {
                    // embd bcast fma
                    mov(reg_ptr_sum_scale,
                            reinterpret_cast<size_t>(p_sum_scale));
                } else {
                    mov(reg_ptr_sum_scale, sum_zp_scale_data_);
                }
            }

            for_(int bd = 0; bd < bd_block; bd++)
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto vmm = accm(ld_block2, bd, ld);
                const auto addr = ptr[reg_aux_D + D_offset(bd, ld)];
                const auto vmm_prev_dst = vmm_tmp(0);
                const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
                const auto k_mask = is_tail ? ld_tail_mask : ld_full_mask;
                const int ld_size = is_tail ? brg.ldb_tail : brg.ld_block;
                cvt2ps(brg.sum_dt, vmm_prev_dst, addr, is_tail, false, k_mask,
                        ld_size);
                if (p_sum_zp_reg_set)
                    uni_vsubps(vmm_prev_dst, vmm_prev_dst, vmm_sum_zp);
                if (p_sum_scale_reg_set) {
                    if (is_superset(brg.isa_impl, avx512_core))
                        uni_vfmadd231ps(
                                vmm, vmm_prev_dst, ptr_b[reg_ptr_sum_scale]);
                    else
                        uni_vfmadd231ps(
                                vmm, vmm_prev_dst, ptr[reg_ptr_sum_scale]);
                } else
                    uni_vaddps(vmm, vmm, vmm_prev_dst);
            }
        }

        if (reset_avx_tail_mask) maybe_set_avx_mask(is_ld_tail);
    };

    if (brg.with_sum) {
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }

    postops_injector_->compute_vector_range(
            max_effective_vregs - bd_block * ld_block2, max_effective_vregs,
            rhs_arg_params);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::store_accumulators_apply_post_ops(
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
        mov(reg_aux_scales, ptr[rsp + reg_aux_scales_offs_]);
        for (int ld = 0; ld < ld_block2; ld++) {
            const auto addr = ptr[reg_aux_scales + scales_offset(ld)];
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            auto vmm_scales = vmm_tmp(0);
            if (IMPLICATION(is_tail, isa_has_masks(brg.isa_impl))) {
                const Vmm vmm_masked
                        = vmm_mask(vmm_scales, is_tail, false, k_mask);
                uni_vmovups(vmm_masked, addr);
            } else {
                auto vmm_scales = vmm_tmp(0);
                vmaskmovps(vmm_scales, vmm_tail_mask(), addr);
            }
            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (dq2ps_required) uni_vcvtdq2ps(vmm, vmm);
                uni_vmulps(vmm, vmm, vmm_scales);
            }
        }
    }

    if (brg.with_bias) { mov(reg_aux_bias, ptr[rsp + reg_aux_bias_offs_]); }
    for (int ld = 0; ld < ld_block2; ld++) {
        auto vmm_bias = vmm_tmp(0);
        if (brg.with_bias) {
            auto ptr_bias = ptr[reg_aux_bias + bias_offset(ld)];
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            cvt2ps(brg.dt_bias, vmm_bias, ptr_bias, is_tail, false, k_mask,
                    is_tail ? brg.ldb_tail : brg.ld_block);
        }
        for (int bd = 0; bd < bd_block; bd++) {
            auto vmm = accm(ld_block2, bd, ld);
            if (dq2ps_required && !brg.with_scales) uni_vcvtdq2ps(vmm, vmm);
            if (brg.with_bias) uni_vaddps(vmm, vmm, vmm_bias);
        }
    }

    if (postops_injector_)
        apply_post_ops(bd_block, ld_block2, ldb_and_bdb_offset, is_ld_tail);

    if (brg.with_dst_scales) {
        mov(reg_aux_dst_scales, ptr[rsp + reg_dst_scales_offs_]);
        auto vmm_dst_scales = vmm_tmp(0);
        vbroadcastss(vmm_dst_scales, ptr[reg_aux_dst_scales]);

        for (int ld = 0; ld < ld_block2; ld++) {
            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                vmulps(vmm, vmm, vmm_dst_scales);
            }
        }
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        mov(reg_aux_zp_c_values, ptr[rsp + reg_aux_zp_c_values_offs_]);
        auto vmm_zp_c = vmm_tmp(0);
        if (brg.zp_type_c == brgemm_broadcast_t::per_tensor) {
            if (is_superset(brg.isa_impl, avx512_core)) {
                uni_vcvtdq2ps(vmm_zp_c,
                        EVEX_compress_addr(reg_aux_zp_c_values, 0, true));
            } else {
                uni_vpbroadcastd(vmm_zp_c, ptr[reg_aux_zp_c_values]);
                uni_vcvtdq2ps(vmm_zp_c, vmm_zp_c);
            }
        }
        for (int ld = 0; ld < ld_block2; ld++) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
                int zp_c_off = zp_c_values_offset(ld);
                if (is_superset(brg.isa_impl, avx512_core)) {
                    auto zp_c_addr
                            = EVEX_compress_addr(reg_aux_zp_c_values, zp_c_off);
                    cvt2ps(data_type::s32, vmm_zp_c, zp_c_addr, is_tail, false,
                            k_mask, is_tail ? brg.ldb_tail : brg.ld_block);
                } else {
                    cvt2ps(data_type::s32, vmm_zp_c,
                            ptr[reg_aux_zp_c_values + zp_c_off], is_tail, false,
                            k_mask, is_tail ? brg.ldb_tail : brg.ld_block);
                }
            }
            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                uni_vaddps(vmm, vmm, vmm_zp_c);
            }
        }
    }

    const bool dt_requires_saturation
            = one_of(brg.dt_d, data_type::u8, data_type::s8, data_type::s32);
    auto vmm_lbound = vmm_tail_mask();
    auto vmm_ubound = vmm_tmp(0);
    assert(vmm_lbound.getIdx() != vmm_ubound.getIdx());
    if (dt_requires_saturation) {
        init_saturate_f32(
                vmm_lbound, vmm_ubound, reg_tmp_gpr, data_type::f32, brg.dt_d);
        for (int bd = 0; bd < bd_block; bd++) {
            for (int ld = 0; ld < ld_block2; ld++) {
                auto vmm = accm(ld_block2, bd, ld);
                saturate_f32(vmm, vmm_lbound, vmm_ubound, brg.dt_d);
                uni_vcvtps2dq(vmm, vmm);
            }
        }
        // below call is not required as s32 doesn't use vmm_lbound
        // maybe_set_avx_mask(is_ld_tail);
    }

    if (brg.is_bf16_emu) bf16_emu_->init_vcvtneps2bf16();

    for (int bd = 0; bd < bd_block; bd++) {
        for (int ld = 0; ld < ld_block2; ld++) {
            auto addr = ptr[reg_aux_D + D_offset(bd, ld)];
            auto vmm = accm(ld_block2, bd, ld);
            auto vmm_lower = Vmm_lower_t(vmm.getIdx());
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            if (is_superset(brg.isa_impl, avx512_core)) {
                const Vmm r_vmm = vmm_mask(vmm, is_tail, true, k_mask);
                const Vmm_lower_t r_ymm
                        = vmm_lower_mask(vmm_lower, is_tail, true, k_mask);
                switch (brg.dt_d) {
                    case data_type::f32:
                    case data_type::s32: uni_vmovups(addr, r_vmm); break;
                    case data_type::bf16: // TODO - clean
                        if (brg.is_bf16_emu) {
                            bf16_emu_->vcvtneps2bf16(vmm_lower, vmm);
                        } else {
                            vcvtneps2bf16(vmm_lower, vmm);
                        }
                        vmovdqu16(addr, r_ymm);
                        break;
                    case data_type::f16:
                        vcvtps2ph(vmm_lower, vmm, _op_mxcsr);
                        vmovdqu16(addr, r_ymm);
                        break;
                    case data_type::s8: vpmovsdb(addr, r_vmm); break;
                    case data_type::u8: vpmovusdb(addr, r_vmm); break;
                    default: assert(!"unknown dst_dt");
                }
            } else {
                const int ld_block = is_tail ? brg.ldb_tail : brg.ld_block;
                if (is_tail && types::data_type_size(brg.dt_b) == sizeof(float))
                    vmaskmovps(addr, vmm_tail_mask(), vmm);
                else
                    store_data(brg.dt_d, vmm, reg_aux_D, D_offset(bd, ld),
                            ld_block);
            }
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::apply_compensation(
        int bd_block, int ld_block2, bool is_ld_tail) {
    // apply compensation to accumulated values
    // to avoid the loss of accuracy when converting s32 to f32
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

    if (!brg.req_cal_comp_pads && brg.zp_type_a != brgemm_broadcast_t::none) {
        auto vmm_zp_a_val = vmm_tmp(1);
        mov(reg_zp_a_val, ptr[rsp + reg_zp_a_val_offs_]);
        uni_vpbroadcastd(vmm_zp_a_val, reg_zp_a_val.cvt32());

        mov(reg_aux_zp_comp_a, ptr[rsp + reg_aux_zp_comp_a_offs_]);
        const auto vmm_zp_comp_a = vmm_tmp(0);
        for (int ld = 0; ld < ld_block2; ld++) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            for (int bd = 0; bd < bd_block; bd++) {
                if (IMPLICATION(!brg.req_comp_pads_with_bd, bd == 0)) {
                    const auto zp_comp_a_addr = ptr[reg_aux_zp_comp_a
                            + bd_zp_comp_a_offset(ld, bd)];
                    if (IMPLICATION(is_tail, isa_has_masks(brg.isa_impl))) {
                        auto vmm_zp_comp_a_masked = vmm_mask(
                                vmm_zp_comp_a, is_tail, false, k_mask);
                        uni_vmovups(vmm_zp_comp_a_masked, zp_comp_a_addr);
                    } else {
                        // cannot use vmaskmovps as vmm_zp_a_val clashes with
                        // vmm_tail_mask
                        load_data(data_type::s32, vmm_zp_comp_a, zp_comp_a_addr,
                                brg.ldb_tail);
                    }
                    uni_vpmulld(vmm_zp_comp_a, vmm_zp_comp_a, vmm_zp_a_val);
                }
                auto vmm = accm(ld_block2, bd, ld);
                uni_vpaddd(vmm, vmm, vmm_zp_comp_a);
            }
        }
        maybe_set_avx_mask(is_ld_tail);
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        mov(reg_aux_zp_comp_b, ptr[rsp + reg_aux_zp_comp_b_offs_]);
        for (int bd = 0; bd < bd_block; bd++) {
            int zp_comp_b_off = zp_comp_b_offset(bd);
            for (int ld = 0; ld < ld_block2; ld++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (is_superset(brg.isa_impl, avx512_core)) {
                    const auto zp_comp_b_addr = EVEX_compress_addr(
                            reg_aux_zp_comp_b, zp_comp_b_off, true);
                    uni_vpaddd(vmm, vmm, zp_comp_b_addr);
                } else {
                    const auto vmm_zp_comp_b = vmm_tmp(0);
                    uni_vpbroadcastd(vmm_zp_comp_b,
                            ptr[reg_aux_zp_comp_b + zp_comp_b_off]);
                    uni_vpaddd(vmm, vmm, vmm_zp_comp_b);
                }
            }
        }
    }

    if (!brg.req_cal_comp_pads && brg.req_s8s8_compensation) {
        mov(reg_aux_compensation, ptr[rsp + reg_aux_comp_offs_]);
        auto vmm_comp = vmm_tmp(0);
        for (int ld = 0; ld < ld_block2; ld++) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            for (int bd = 0; bd < bd_block; bd++) {
                if (IMPLICATION(!brg.req_comp_pads_with_bd, bd == 0)) {
                    const auto comp_addr = ptr[reg_aux_compensation
                            + bd_compensation_offset(ld, bd)];
                    if (IMPLICATION(is_tail,
                                is_superset(brg.isa_impl, avx512_core))) {
                        auto vmm_comp_masked
                                = vmm_mask(vmm_comp, is_tail, false, k_mask);
                        uni_vmovups(vmm_comp_masked, comp_addr);
                    } else
                        vmaskmovps(vmm_comp, vmm_tail_mask(), comp_addr);
                }
                auto vmm = accm(ld_block2, bd, ld);
                uni_vpaddd(vmm, vmm, vmm_comp);
            }
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::store_accumulators_without_post_ops(
        int bd_block, int ld_block2, bool is_ld_tail) {

    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dt_requires_saturation = brg.is_int8
            && !IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);

    if (dt_requires_saturation) {
        auto vmm_ubound = vmm_tmp(0);
        auto vmm_lbound = vmm_tmp(1);
        init_saturate_f32(
                vmm_lbound, vmm_ubound, reg_tmp_gpr, data_type::f32, brg.dt_d);
        for (int bd = 0; bd < bd_block; bd++) {
            for (int ld = 0; ld < ld_block2; ld++) {
                auto vmm = accm(ld_block2, bd, ld);
                saturate_f32(vmm, vmm_lbound, vmm_ubound, brg.dt_d);
                uni_vcvtps2dq(vmm, vmm);
            }
        }
        // below call is not required as s32 doesn't use vmm_lbound
        // maybe_set_avx_mask(is_ld_tail);
    }

    for (int bd = 0; bd < bd_block; bd++) {
        for (int ld = 0; ld < ld_block2; ld++) {
            auto vmm = accm(ld_block2, bd, ld);
            const auto addr_c = ptr[reg_aux_C + C_offset(bd, ld)];
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            if (!is_tail)
                uni_vmovups(addr_c, vmm);
            else if (isa_has_masks(brg.isa_impl)) { // is_tail
                uni_vmovups(addr_c | ld_tail_mask | T_z, vmm);
            } else {
                vmaskmovps(addr_c, vmm_tail_mask(), vmm);
            }
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::store_accumulators(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_ld_tail,
        bool skip_accumulation) {
    const bool has_zero_points = !everyone_is(brgemm_broadcast_t::none,
            brg.zp_type_a, brg.zp_type_b, brg.zp_type_c);
    const bool are_post_ops_applicable = one_of(true, brg.with_eltwise,
            brg.with_binary, brg.with_scales, brg.with_bias, brg.with_sum,
            brg.dt_d != brg.dt_c, brg.req_s8s8_compensation, has_zero_points,
            brg.with_dst_scales);
    const bool need_to_apply_alpha_beta = brg.beta != 0.f || brg.alpha != 1.f;
    const bool need_generate_zp_a_compensation
            = brg.is_int8 && (brg.req_s8s8_compensation || has_zero_points);

    maybe_set_avx_mask(is_ld_tail);

    if (brg.is_tmm) {
        if (need_to_apply_alpha_beta || are_post_ops_applicable
                || need_generate_zp_a_compensation)
            mov(reg_stride_ld_block, brg.ld_block * brg.typesize_C);
        else
            mov(reg_stride_ld_block, brg.LDC * brg.typesize_C);

        auto store_accumulators_amx = [&](const bool apply_post_ops,
                                              const bool apply_zp_a_compensation
                                              = false) {
            mov(reg_buf, ptr[rsp + reg_buf_offs_]);
            for (int bdb = 0; bdb < bd_block2; bdb++) {
                int adj_bd_block = (brg.is_M_tail && is_bdb_tail)
                        ? brg.bdb_tail
                        : brg.bd_block;
                for (int ldb = 0; ldb < ld_block2; ldb++) {
                    int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
                    if (need_to_apply_alpha_beta || are_post_ops_applicable
                            || apply_zp_a_compensation) {
                        if (skip_accumulation) {
                            for (int bd = 0; bd < adj_bd_block; bd++) {
                                auto vreg_acc = accm(1, bd, 0);
                                uni_vpxor(vreg_acc, vreg_acc, vreg_acc);
                            }
                        } else {
                            tilestored(ptr[reg_buf + reg_stride_ld_block],
                                    Tmm(brg.get_C_tensor(bdb, idx, is_bdb_tail,
                                            is_ld_tail)));
                            for (int bd = 0; bd < adj_bd_block; bd++) {
                                size_t buf_offset
                                        = (bd * brg.ld_block) * brg.typesize_C;
                                auto vreg_acc = is_ld_tail
                                        ? accm(1, bd, 0) | ld_tail_mask | T_z
                                        : accm(1, bd, 0);
                                uni_vmovups(
                                        vreg_acc, ptr[reg_buf + buf_offset]);
                            }
                        }

                        if (apply_zp_a_compensation) {
                            apply_compensation(adj_bd_block, 1, is_ld_tail);
                        }

                        if (need_to_apply_alpha_beta)
                            apply_alpha_beta(adj_bd_block, 1, is_ld_tail);

                        if (apply_post_ops) {
                            const size_t ldb_and_bdb_offset
                                    = ldb_po_offset(ldb) + bdb_po_offset(bdb);
                            store_accumulators_apply_post_ops(adj_bd_block, 1,
                                    ldb_and_bdb_offset, is_ld_tail);
                            if (ldb < ld_block2 - 1) advance_ldb_post_op_regs();
                            add(reg_aux_D, ldb_D_offset(1));
                        } else {
                            store_accumulators_without_post_ops(
                                    adj_bd_block, 1, is_ld_tail);
                        }
                        mov(reg_buf, ptr[rsp + reg_buf_offs_]);
                    } else {
                        auto tmm = Tmm(brg.get_C_tensor(
                                bdb, idx, is_bdb_tail, is_ld_tail));
                        if (skip_accumulation) tilezero(tmm);
                        tilestored(ptr[reg_aux_C + reg_stride_ld_block], tmm);
                    }
                    add(reg_aux_C, ldb_C_offset(1));
                }
                sub(reg_aux_C, ldb_C_offset(ld_block2));
                add(reg_aux_C, bdb_C_offset(1));
                if (apply_post_ops) {
                    sub(reg_aux_D, ldb_D_offset(ld_block2));
                    add(reg_aux_D, bdb_D_offset(1));

                    bool post_processed = false;
                    if (ld_block2 > 1) {
                        restore_ldb_post_op_regs(ld_block2);
                        post_processed |= utils::one_of(true, brg.with_bias,
                                brg.with_scales,
                                brg.zp_type_a != brgemm_broadcast_t::none,
                                brg.zp_type_c == brgemm_broadcast_t::per_n,
                                brg.with_dst_scales);
                    }
                    if (bdb < bd_block2 - 1) {
                        advance_bdb_post_op_regs(adj_bd_block);
                        post_processed
                                |= brg.zp_type_b != brgemm_broadcast_t::none;
                    }
                    if (post_processed) mov(reg_buf, ptr[rsp + reg_buf_offs_]);
                }
            }
            sub(reg_aux_C, bdb_C_offset(bd_block2));
            if (apply_post_ops) {
                sub(reg_aux_D, bdb_D_offset(bd_block2));
                restore_bdb_post_op_regs(bd_block2);
            }
        };

        Label label_done;
        if (are_post_ops_applicable) {
            Label label_skip_post_ops;
            mov(reg_do_post_ops, ptr[rsp + reg_do_post_ops_offs_]);
            cmp(reg_do_post_ops, 0);
            jz(label_skip_post_ops, T_NEAR);
            if (need_generate_zp_a_compensation) {
                Label label_skip_zp_comp_with_postops;
                mov(reg_do_comp, ptr[rsp + reg_do_comp_offs_]);
                cmp(reg_do_comp, 0);
                jz(label_skip_zp_comp_with_postops, T_NEAR);
                store_accumulators_amx(true, true);
                jmp(label_done, T_NEAR);

                L_aligned(label_skip_zp_comp_with_postops);
            }
            store_accumulators_amx(true);

            jmp(label_done, T_NEAR);

            L_aligned(label_skip_post_ops);
        }

        if (need_generate_zp_a_compensation) {
            Label label_skip_zp_comp;
            mov(reg_do_comp, ptr[rsp + reg_do_comp_offs_]);
            cmp(reg_do_comp, 0);
            jz(label_skip_zp_comp, T_NEAR);
            store_accumulators_amx(false, true);
            jmp(label_done, T_NEAR);

            L_aligned(label_skip_zp_comp);
        }

        store_accumulators_amx(false);
        L_aligned(label_done);
    } else {
        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;

        if (need_generate_zp_a_compensation) {
            Label label_store_without_comp;
            mov(reg_do_comp, ptr[rsp + reg_do_comp_offs_]);
            cmp(reg_do_comp, 0);
            jz(label_store_without_comp, T_NEAR);
            apply_compensation(bd_block, ld_block2, is_ld_tail);

            L_aligned(label_store_without_comp);
        }

        if (need_to_apply_alpha_beta)
            apply_alpha_beta(bd_block, ld_block2, is_ld_tail);

        Label label_done;
        if (are_post_ops_applicable) {
            Label label_store_without_post_ops;
            mov(reg_do_post_ops, ptr[rsp + reg_do_post_ops_offs_]);
            cmp(reg_do_post_ops, 0);
            jz(label_store_without_post_ops, T_NEAR);
            store_accumulators_apply_post_ops(
                    bd_block, ld_block2, 0, is_ld_tail);
            jmp(label_done, T_NEAR);

            L_aligned(label_store_without_post_ops);
        }
        store_accumulators_without_post_ops(bd_block, ld_block2, is_ld_tail);
        L_aligned(label_done);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::restore_A_B_matrices() {
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

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::set_A_B_matrices() {
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

        safe_add(reg_aux1_A, brg.stride_a, reg_tmp_gpr);
        safe_add(reg_aux1_B, brg.stride_b, reg_tmp_gpr);
        if (vpad_exist) {
            mov(reg_strd_batch, ptr[rsp + origin_strd_batch_offs_]);
            add(reg_strd_batch, sizeof(brgemm_batch_element_t));
            mov(ptr[rsp + origin_strd_batch_offs_], reg_strd_batch);
        }
    }

    add(reg_aux_A, reg_a_offset);
    add(reg_aux_B, reg_b_offset);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::gemm_microkernel_amx(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_rd_tail, bool is_ld_tail) {
    auto tdpbxxd = [this](const Tmm &x1, const Tmm &x2, const Tmm &x3) {
        if (brg.dt_a == data_type::bf16 && brg.dt_b == data_type::bf16) {
            tdpbf16ps(x1, x2, x3);
        } else if (brg.dt_a == data_type::f16 && brg.dt_b == data_type::f16) {
            tdpfp16ps(x1, x2, x3);
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

    auto maybe_tileloadd_nt = [this](const Tmm &t1, reg64_t base, int offset,
                                      reg64_t stride, bool try_load_nt) {
        if (try_load_nt
                && static_cast<size_t>(
                           brg.typesize_A * brg.brgattr.hint_expected_A_size
                           + brg.typesize_B * brg.brgattr.hint_expected_B_size
                           + brg.typesize_C * brg.brgattr.hint_expected_C_size)
                        >= platform::get_per_core_cache_size(1))
            tileloaddt1(t1, ptr[base + offset + stride]);
        else
            tileloadd(t1, ptr[base + offset + stride]);
    };

    int rbd_block = (is_rd_tail) ? 1 : brg.rdb;
    for (int rdb = 0; rdb < rbd_block; rdb++) {
        for (int bdb = 0; bdb < bd_block2; bdb++) {
            maybe_tileloadd_nt(Tmm(brg.get_A_tensor(bdb, is_bdb_tail)),
                    reg_aux_A, rdb * rdb_A_offset() + A_offset(bdb, 0, true),
                    reg_stride_lda,
                    brg.innermost_loop == brgemm_bd_loop_innermost);
        }
        for (int ldb = 0; ldb < ld_block2; ldb++) {
            const int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
            maybe_tileloadd_nt(Tmm(brg.get_B_tensor(idx, is_ld_tail)),
                    reg_aux_B, rdb * rdb_B_offset() + B_offset(ldb, 0, true),
                    reg_stride_ldb,
                    brg.innermost_loop == brgemm_ld_loop_innermost);
            for (int bdb = 0; bdb < bd_block2; bdb++) {
                tdpbxxd(Tmm(brg.get_C_tensor(
                                bdb, idx, is_bdb_tail, is_ld_tail)),
                        Tmm(brg.get_A_tensor(bdb, is_bdb_tail)),
                        Tmm(brg.get_B_tensor(idx, is_ld_tail)));
            }
        }
    }
    if (!is_rd_tail) {
        add(reg_aux_A, brg.rdb * rdb_A_offset());
        add(reg_aux_B, brg.rdb * rdb_B_offset());
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::dot_product(Vmm v1, Vmm v2, Vmm v3) {
    if (brg.is_f32 || brg.is_f16
            || (brg.is_bf16 && brg.isa_impl == avx2_vnni_2))
        uni_vfmadd231ps(v1, v2, v3);
    else if (brg.is_bf16)
        vdpbf16ps(v1, v2, v3);
    else if (brg.is_int8) {
        if (brg.isa_impl == avx2_vnni_2 && brg.dt_a == data_type::s8)
            vpdpbssd(v1, v3, v2);
        else if (brg.has_int8_vnni)
            vpdpbusd(v1, v3, v2,
                    is_superset(isa, avx512_core) ? EvexEncoding : VexEncoding);
        else {
            vpmaddubsw(int8_dot_product_temp(), v3, v2);
            vpmaddwd(int8_dot_product_temp(), int8_dot_product_temp(),
                    int8_ones_words());
            vpaddd(v1, v1, int8_dot_product_temp());
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::compute_int8_compensation(int rd_loop,
        int bd_b, int bd_e, int bd_block, int ld_block2, bool is_ld_tail,
        int vpad) {
    assert(brg.is_int8);

    auto compensation_padding = [this, ld_block2](Vmm vmm_load, Vmm vmm_tmp,
                                        int ld, int bd_b, int bd_e) {
        // req_cal_comp_pads -> only calculate compensation along with
        // computation and do not use pre-calculated compensation.
        // Calculate comp padding as:
        // accum - inp_shift * conv(1, wei_s32)
        if (brg.req_s8s8_compensation) {
            if (brg.req_cal_comp_pads) {
                uni_vpxor(vmm_tmp, vmm_tmp, vmm_tmp);
                dot_product(vmm_tmp, vmm_load, vmm_inp_shift());
            }

            for (int bd = bd_b; bd < bd_e; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (brg.req_cal_comp_pads) {
                    uni_vpsubd(vmm, vmm, vmm_tmp);
                } else {
                    dot_product(vmm, vmm_load, vmm_inp_shift());
                }
            }
        }

        if (brg.zp_type_a != brgemm_broadcast_t::none) {
            uni_vpxor(vmm_tmp, vmm_tmp, vmm_tmp);
            dot_product(vmm_tmp, vmm_load, vmm_one_bytes());
            uni_vpmulld(vmm_tmp, vmm_tmp, vmm_zp_a_shift());

            for (int bd = bd_b; bd < bd_e; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (brg.req_cal_comp_pads) {
                    uni_vpsubd(vmm, vmm, vmm_tmp);
                } else {
                    uni_vpaddd(vmm, vmm, vmm_tmp);
                }
            }
        }
    };

    if (n_bcast_1_load && brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(ptr[rsp + reg_bdb_loop_offs_], reg_bdb_loop);
        const auto reg32_scratch = reg_zp_a_input_shift.cvt32();
        mov(reg32_scratch, 0x1010101);
        uni_vpbroadcastd(vmm_one_bytes(), reg32_scratch);
        mov(reg32_scratch, ptr[rsp + reg_zp_a_val_offs_]);
        uni_vpbroadcastd(vmm_zp_a_shift(), reg32_scratch);
        mov(reg_bdb_loop, ptr[rsp + reg_bdb_loop_offs_]);
    }

    for_(int rd = 0; rd < rd_loop; rd += brg.rd_step)
    for (int ld = 0; ld < ld_block2; ++ld) {
        const auto addr = ptr[reg_aux_B + B_offset(ld, rd)];
        const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
        if (IMPLICATION(is_tail, is_superset(brg.isa_impl, avx512_core))) {
            auto vmm_store = vmm_mask(load(), is_tail, false, ld_tail_mask);
            uni_vmovups(vmm_store, addr);
        } else {
            load_bytes(
                    load(), addr, brg.typesize_B * brg.ldb_tail * brg.ld_step);
        }

        if (brg.req_cal_comp_pads) {
            compensation_padding(load(), bcst(), ld, bd_b, bd_e);
        } else if (vpad != 0) {
            if (bd_b > 0) compensation_padding(load(), bcst(), ld, 0, bd_b);
            if (bd_e < bd_block)
                compensation_padding(load(), bcst(), ld, bd_e, bd_block);
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::gemm_microkernel(int bd_block2,
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
            rd_tail_size = brg.rdb_tail % brg.rd_step;
            rd_loop = (rd_tail_size != 0)
                    ? ((brg.rdb_tail / brg.rd_step) + 1) * brg.rd_step
                    : brg.rdb_tail;
        } else
            rd_loop = brg.rdb_tail;
    } else
        rd_loop = brg.rd_block;

    auto broadcast = [this, rd_tail_size](Vmm v1, size_t offset, bool is_tail,
                             data_type_t dt) {
        if (is_tail) {
            Xmm xmm_tmp = Xmm(v1.getIdx());
            load_bytes(
                    xmm_tmp, reg_aux_A, offset, rd_tail_size * brg.typesize_A);
            uni_vpbroadcastd(v1, xmm_tmp);
        } else {
            if (dt == data_type::f32) {
                uni_vbroadcastss(v1, ptr[reg_aux_A + offset]);
            } else if (dt == data_type::bf16) {
                if (brg.isa_impl == avx2_vnni_2)
                    vbcstnebf162ps(v1, ptr[reg_aux_A + offset]);
                else
                    uni_vpbroadcastd(v1, ptr[reg_aux_A + offset]);
            } else if (one_of(dt, data_type::s8, data_type::u8)) {
                uni_vpbroadcastd(v1, ptr[reg_aux_A + offset]);
            } else if (dt == data_type::f16) {
                if (brg.isa_impl == avx2_vnni_2)
                    vbcstnesh2ps(v1, ptr[reg_aux_A + offset]);
                else
                    vcvtph2psx(v1, ptr_b[reg_aux_A + offset]);
            }
        }

        if (brg.req_s8s8_compensation) uni_vpaddb(v1, v1, vmm_inp_shift());
    };

    const bool comp_vpad = vpad != 0
            && (brg.req_s8s8_compensation
                    || brg.zp_type_a != brgemm_broadcast_t::none);
    if (brg.req_cal_comp_pads || comp_vpad)
        compute_int8_compensation(
                rd_loop, bd_b, bd_e, bd_block, ld_block2, is_ld_tail, vpad);

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
                        have_to_load_bytes && bd_by_load_bytes, brg.dt_a);
            }
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto addr = ptr[reg_aux_B + B_offset(ld, rd)];
                const Vmm vmm_load
                        = vmm_mask(load(), is_ld_tail, false, ld_tail_mask);
                // Note: Assuming the tails are properly padded/blocked for
                // avx2_vnni_2 with xf16 data type, as the B matrix is generally
                // at least double-blocked.
                if (brg.dt_b == data_type::f16) {
                    if (brg.isa_impl == avx2_vnni_2) {
                        if (rd % 2 == 0)
                            vcvtneeph2ps(vmm_load, addr);
                        else
                            vcvtneoph2ps(vmm_load, addr);
                    } else
                        vcvtph2psx(vmm_load, addr);
                } else if (brg.dt_b == data_type::bf16
                        && brg.isa_impl == avx2_vnni_2) {
                    if (rd % 2 == 0)
                        vcvtneebf162ps(vmm_load, addr);
                    else
                        vcvtneobf162ps(vmm_load, addr);
                } else if (is_ld_tail) {
                    if (is_superset(brg.isa_impl, avx512_core)) {
                        uni_vmovups(vmm_load, addr);
                    } else {
                        load_bytes(vmm_load, addr,
                                brg.typesize_B * brg.ldb_tail * brg.ld_step);
                    }
                } else {
                    uni_vmovups(vmm_load, addr);
                }
                for (int bd = bd_b; bd < bd_e; bd++) {
                    auto vmm = accm(ld_block2, bd, ld);
                    if (is_emdbd)
                        uni_vfmadd231ps(vmm, load(),
                                ptr_b[reg_aux_A + A_offset(bd, rd)]);
                    else
                        dot_product(vmm, load(), bcst(bd));
                }
            }
        }
    } else {
        for (int rd = 0; rd < rd_loop; rd += brg.rd_step) {
            int prefetch_count_B = 0;
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto addr = ptr[reg_aux_B + B_offset(ld, rd)];
                const Vmm vmm_load
                        = vmm_mask(load(ld), is_ld_tail, false, ld_tail_mask);
                // Note: Assuming the tails are properly padded/blocked for
                // avx2_vnni_2, as the B matrix is generally
                // at least double-blocked.
                if (brg.dt_b == data_type::f16) {
                    if (brg.isa_impl == avx2_vnni_2) {
                        if (rd % 2 == 0)
                            vcvtneeph2ps(vmm_load, addr);
                        else
                            vcvtneoph2ps(vmm_load, addr);
                    } else {
                        vcvtph2psx(vmm_load, addr);
                    }
                } else if (brg.dt_b == data_type::bf16
                        && brg.isa_impl == avx2_vnni_2) {
                    if (rd % 2 == 0)
                        vcvtneebf162ps(vmm_load, addr);
                    else
                        vcvtneobf162ps(vmm_load, addr);
                } else if (is_ld_tail) {
                    if (is_superset(brg.isa_impl, avx512_core)) {
                        uni_vmovups(vmm_load, addr);
                    } else {
                        load_bytes(vmm_load, addr,
                                brg.typesize_B * brg.ldb_tail * brg.ld_step);
                    }
                } else {
                    uni_vmovups(vmm_load, addr);
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
                            have_to_load_bytes && bd_by_load_bytes, brg.dt_a);
                }
                if (prefetch_count_B < ld_block2) {
                    prefetcht0(ptr[reg_aux_B + B_offset(prefetch_count_B++, rd)
                            + brg.LDB * brg.rd_block * brg.typesize_B]);
                }
                for (int ld = 0; ld < ld_block2; ld++) {
                    auto vmm = accm(ld_block2, bd, ld);
                    if (is_emdbd)
                        uni_vfmadd231ps(vmm, load(ld),
                                ptr_b[reg_aux_A + A_offset(bd, rd)]);
                    else
                        dot_product(vmm, load(ld), bcst());
                }
            }
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::ldb_loop(int bd_block2, bool is_bdb_tail,
        int ld_block2, int ldb_loop_length, bool is_reg_tail, bool is_ld_tail,
        bool check_top_vpad, bool check_bottom_vpad, int rows_for_rd_tail,
        bool skip_accumulation) {

    Label ldb_loop_label;
    Label BS_loop_label;

    copy_post_ops_stack_values_to_aux(is_reg_tail);

    auto ld_loop_body = [&](int vpad) {
        set_A_B_matrices();

        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
        const auto bd_b = nstl::max(0, vpad);
        const auto bd_e = nstl::min(bd_block, bd_block + vpad);
        const auto is_valid_bd
                = need_comp_pads && vpad != 0 ? bd_b <= bd_e : bd_b < bd_e;
        if (!is_valid_bd) return;

        if (brg.is_tmm) {
            const bool is_rd_tail = false;
            gemm_microkernel_amx(
                    bd_block2, is_bdb_tail, ld_block2, is_rd_tail, is_ld_tail);
        } else {
            if (brg.rdb > 0) {
                Label rdb_loop_label;
                mov(reg_rdb_loop, brg.rdb);
                L_aligned(rdb_loop_label, 64);
                {
                    const bool is_rd_tail = false;
                    gemm_microkernel(bd_block2, is_bdb_tail, ld_block2,
                            is_rd_tail, is_ld_tail, vpad, rows_for_rd_tail);

                    add(reg_aux_A, rdb_A_offset());
                    add(reg_aux_B, rdb_B_offset());

                    dec(reg_rdb_loop);
                    cmp(reg_rdb_loop, 0);
                }
                jg(rdb_loop_label, T_NEAR);
            }
        }
        if (brg.rdb_tail != 0) {
            const bool is_rd_tail = true;
            if (brg.is_tmm) {
                gemm_microkernel_amx(bd_block2, is_bdb_tail, ld_block2,
                        is_rd_tail, is_ld_tail);
            } else {
                gemm_microkernel(bd_block2, is_bdb_tail, ld_block2, is_rd_tail,
                        is_ld_tail, vpad, rows_for_rd_tail);
            }
        }
    };
    if (is_ldb_loop_) {
        mov(reg_ldb_loop, ldb_loop_length);
        if (brg.is_tmm) mov(ptr[rsp + reg_ldb_loop_offs_], reg_ldb_loop);
    }

    L_aligned(ldb_loop_label, 64);
    {
        zero_accumulators(bd_block2, is_bdb_tail, ld_block2, is_ld_tail,
                skip_accumulation);

        if (is_ldb_loop_)
            mov(ptr[rsp + reg_D_offs_], reg_D);
        else {
            mov(reg_ldb_loop, reg_D);
            if (brg.is_tmm) mov(ptr[rsp + reg_ldb_loop_offs_], reg_ldb_loop);
        }
        if (brg.brgattr.max_bs > 1) mov(ptr[rsp + reg_aux_D_offs_], reg_aux_D);

        if (brg.alpha != 0.f && !skip_accumulation) {
            restore_A_B_matrices();
            if (brg.is_tmm) {
                mov(reg_stride_lda, brg.typesize_A * brg.LDA);
                mov(reg_stride_ldb, brg.rd_step * brg.typesize_B * brg.LDB);
            }

            if (brg.req_s8s8_compensation) {
                mov(ptr[rsp + reg_bdb_loop_offs_], reg_bdb_loop);
                mov(reg_s8_input_shift, 128);
                uni_vpbroadcastb(vmm_inp_shift(), reg_s8_input_shift.cvt8());
                mov(reg_bdb_loop, ptr[rsp + reg_bdb_loop_offs_]);
            }
            if (need_comp_pads && brg.zp_type_a != brgemm_broadcast_t::none) {
                mov(ptr[rsp + reg_bdb_loop_offs_], reg_bdb_loop);
                const auto reg32_scratch = reg_zp_a_input_shift.cvt32();
                mov(reg32_scratch, 0x1010101);
                uni_vpbroadcastd(vmm_one_bytes(), reg32_scratch);
                mov(reg32_scratch, ptr[rsp + reg_zp_a_val_offs_]);
                uni_vpbroadcastd(vmm_zp_a_shift(), reg32_scratch);
                mov(reg_bdb_loop, ptr[rsp + reg_bdb_loop_offs_]);
            }

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
                    std::vector<Label> Vpad_loop_iter_label(MAX_N_VPADS);
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
                        if (check_bottom_vpad && brg.bdb_tail && vpad < 0) {
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

        if (is_ldb_loop_)
            mov(reg_D, ptr[rsp + reg_D_offs_]);
        else {
            if (brg.is_tmm) mov(reg_ldb_loop, ptr[rsp + reg_ldb_loop_offs_]);
            mov(reg_D, reg_ldb_loop);
        }
        if (brg.brgattr.max_bs > 1) mov(reg_aux_D, ptr[rsp + reg_aux_D_offs_]);

        store_accumulators(bd_block2, is_bdb_tail, ld_block2, is_ld_tail,
                skip_accumulation);
        if (is_ldb_loop_) {
            if (brg.is_tmm) mov(reg_ldb_loop, ptr[rsp + reg_ldb_loop_offs_]);
            if (!is_ld_tail)
                ldb_regs_shift(ld_block2);
            else
                ldb_regs_shift(1, true);
            dec(reg_ldb_loop);
            cmp(reg_ldb_loop, 0);
            if (brg.is_tmm) mov(ptr[rsp + reg_ldb_loop_offs_], reg_ldb_loop);
            jg(ldb_loop_label, T_NEAR);
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::bdb_loop() {
    auto do_ldb_loop = [this](int bd_block2, bool is_bdb_tail,
                               bool check_top_vpad, bool check_bottom_vpad,
                               int rows_for_rd_tail, bool skip_accumulation) {
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

    auto bdb_loop_body = [this, do_ldb_loop](int bd_block2, bool is_bdb_tail,
                                 bool check_top_vpad, bool check_bottom_vpad,
                                 int rows_for_rd_tail, bool skip_accumulation) {
        do_ldb_loop(bd_block2, is_bdb_tail, check_top_vpad, check_bottom_vpad,
                rows_for_rd_tail, skip_accumulation);

        add(reg_C, bdb_C_offset(bd_block2));
        add(reg_D, bdb_D_offset(bd_block2));
        add(reg_a_offset, bdb_A_offset(bd_block2));

        advance_bd_block2_post_op_regs(bd_block2);
    };

    int rows_for_rd_tail, bd_blocks_for_rd_tail;

    if (brg.is_tmm) {
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
        const int free_vregs = max_effective_vregs - brg.req_s8s8_compensation;
        n_bcast_1_load = brg.is_int8
                && ((brg.bd_block * (ld_block2 + 1) < free_vregs)
                        && (bd_blocks_for_rd_tail == 0)
                        && (rows_for_rd_tail == 0));
        if (brg.brgattr.hint_loop_order != brgemm_lo_default)
            n_bcast_1_load = (brg.brgattr.hint_loop_order == brgemm_lo_bl_1load)
                    ? true
                    : false;
    }

    auto bdb_loop_avx512 = [&](bool skip_accumulation) {
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
                mov(reg_bdb_loop, bdblocks);
                L_aligned(bdb_loop_label, 64);
                {
                    bdb_loop_body(1, false, false, false,
                            bd_blocks_for_rd_tail <= 1 ? 0 : rows_for_rd_tail,
                            skip_accumulation);
                    dec(reg_bdb_loop);
                    cmp(reg_bdb_loop, 1);
                    jg(bdb_loop_label, T_NEAR);
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
                                                           : rows_for_rd_tail,
                                skip_accumulation);
                        dec(reg_bdb_loop);
                        cmp(reg_bdb_loop, rows_for_rd_tail ? 1 : 0);
                        jg(bdb_loop_label, T_NEAR);
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
    auto bdb_loop_amx = [&](bool skip_accumulation) {
        Label bdb_loop_label;
        if (brg.bd_block2 >= 1) {
            mov(reg_bdb_loop, brg.bdb2);
            mov(ptr[rsp + reg_bdb_loop_offs_], reg_bdb_loop);
            L_aligned(bdb_loop_label, 64);
            {
                bdb_loop_body(brg.bd_block2, false, false, false, 0,
                        skip_accumulation);
                mov(reg_bdb_loop, ptr[rsp + reg_bdb_loop_offs_]);
                dec(reg_bdb_loop);
                cmp(reg_bdb_loop, 0);
                mov(ptr[rsp + reg_bdb_loop_offs_], reg_bdb_loop);
            }
            jg(bdb_loop_label, T_NEAR);
        }
        if (brg.bdb2_tail > 0)
            bdb_loop_body(
                    brg.bdb2_tail, false, false, false, 0, skip_accumulation);
        if (brg.bdb_tail > 0)
            do_ldb_loop(1, true, false, false, 0, skip_accumulation);
    };

    auto bdb_loop_general = [&](bool skip_accumulation) {
        if (brg.type == brgemm_addr && brg.brgattr.max_bs == 1 && !vpad_exist
                && !skip_accumulation) {
            mov(reg_aux1_A, ptr[reg_addr_batch + GET_OFF_BATCH_ELEMENT(ptr.A)]);
            mov(reg_aux1_B, ptr[reg_addr_batch + GET_OFF_BATCH_ELEMENT(ptr.B)]);
        }

        xor_(reg_a_offset, reg_a_offset);
        if (brg.is_tmm)
            bdb_loop_amx(skip_accumulation);
        else
            bdb_loop_avx512(skip_accumulation);
    };

    if (brg.brgattr.generate_skip_accumulation) {
        Label bdb_loop_skip_acc_label, bdb_loop_done_label;
        mov(reg_skip_accm, ptr[rsp + reg_skip_accm_offs_]);
        cmp(reg_skip_accm, 0);
        jnz(bdb_loop_skip_acc_label, T_NEAR);

        bdb_loop_general(false);
        jmp(bdb_loop_done_label, T_NEAR);

        L_aligned(bdb_loop_skip_acc_label, 64);
        bdb_loop_general(true);

        L_aligned(bdb_loop_done_label, 64);
    } else
        bdb_loop_general(false);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brgemm_kernel_t<isa, Wmm>::generate() {
    preamble();

    sub(rsp, stack_space_needed_);

    vpad_exist
            = (brg.brgattr.max_top_vpad > 0 || brg.brgattr.max_bottom_vpad > 0)
            ? true
            : false;
    need_comp_pads = IMPLICATION(brg.zp_type_a == brgemm_broadcast_t::none,
                             brg.req_s8s8_compensation)
            && IMPLICATION(!vpad_exist, brg.req_cal_comp_pads);

    if (is_superset(brg.isa_impl, avx512_core)) {
        const auto full_mask = size_t {0xffffffffffffffff};
        const auto tail_mask = size_t((1 << brg.ldb_tail) - 1);
        reg64_t reg_mask = rax;

        mov(reg_mask, full_mask);
        kmovq(ld_full_mask, reg_mask);
        mov(reg_mask, tail_mask);
        kmovq(ld_tail_mask, reg_mask);
    }

    if (brg.is_int8 && !brg.has_int8_vnni) {
        mov(reg_tmp_gpr.cvt16(), 0x1);
        vpbroadcastw(int8_ones_words(), reg_tmp_gpr.cvt16());
    }

    read_params();

    bdb_loop();

    add(rsp, stack_space_needed_);

    postamble();

    align(32);
    const int simd = vreg_traits<Vmm>::vlen / sizeof(float);
    if (!isa_has_masks(brg.isa_impl) && brg.ldb_tail > 0) {
        L(avx_tail_mask_);
        for (int i = 0; i < brg.ldb_tail; ++i)
            dd(0xffffffff);
        for (int i = brg.ldb_tail; i < simd; ++i)
            dd(0);
    }
    if (!is_superset(brg.isa_impl, avx512_core) && brg.with_sum
            && brg.sum_scale != 1.f) {
        L(sum_zp_scale_data_);
        const int scale_int = float2int(brg.sum_scale);
        for (int i = 0; i < simd; ++i)
            dd(scale_int);
    }

    if (brg.with_eltwise) postops_injector_->prepare_table();
}

brgemm_attr_t::brgemm_attr_t()
    : max_bs(INT_MAX)
    , max_top_vpad(0)
    , max_bottom_vpad(0)
    , max_top_bpad(0)
    , max_bottom_bpad(0)
    , hint_expected_A_size(platform::get_per_core_cache_size(1))
    , hint_expected_B_size(platform::get_per_core_cache_size(1))
    , hint_expected_C_size(platform::get_per_core_cache_size(1))
    , hint_innermost_loop(brgemm_ld_loop_innermost)
    , hint_loop_order(brgemm_kernel_loop_order_t::brgemm_lo_default)
    , hint_prefetching(brgemm_kernel_prefetching_t::brgemm_prf_default)
    , wary_tail_read(true)
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

template <cpu_isa_t isa, typename Wmm>
brgemm_kernel_common_t<isa, Wmm>::brgemm_kernel_common_t(const brgemm_t abrd) {
    brgemm_kernel_ = new jit_brgemm_kernel_t<isa, Wmm>(abrd);
}

template <cpu_isa_t isa, typename Wmm>
status_t brgemm_kernel_common_t<isa, Wmm>::create_kernel() {
    if (brgemm_kernel_) return brgemm_kernel_->create_kernel();
    return status::out_of_memory;
}

template <cpu_isa_t isa, typename Wmm>
void brgemm_kernel_common_t<isa, Wmm>::operator()(
        brgemm_kernel_params_t *params) const {
    (*brgemm_kernel_)(params);
}

template <cpu_isa_t isa, typename Wmm>
const jit_generator *
brgemm_kernel_common_t<isa, Wmm>::get_jit_generator() const {
    return brgemm_kernel_;
}

template <cpu_isa_t isa, typename Wmm>
brgemm_kernel_common_t<isa, Wmm>::~brgemm_kernel_common_t() {
    delete brgemm_kernel_;
}

// isa specific instantiations are required because
// post-ops require template isa param.
template struct brgemm_kernel_common_t<avx512_core_amx_fp16, Xbyak::Tmm>;
template struct brgemm_kernel_common_t<avx512_core_amx, Xbyak::Tmm>;
template struct brgemm_kernel_common_t<avx512_core_fp16, Xbyak::Zmm>;
template struct brgemm_kernel_common_t<avx512_core_bf16, Xbyak::Zmm>;
template struct brgemm_kernel_common_t<avx512_core_vnni, Xbyak::Zmm>;
template struct brgemm_kernel_common_t<avx512_core, Xbyak::Zmm>;
template struct brgemm_kernel_common_t<avx2_vnni, Xbyak::Ymm>;
template struct brgemm_kernel_common_t<avx2_vnni_2, Xbyak::Ymm>;
template struct brgemm_kernel_common_t<avx2, Xbyak::Ymm>;
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
