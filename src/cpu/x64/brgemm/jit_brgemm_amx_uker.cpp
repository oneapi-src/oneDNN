/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "cpu/platform.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_types.hpp"
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

struct jit_brgemm_amx_uker_base_t : public jit_generator {
    jit_brgemm_amx_uker_base_t(const brgemm_t &abrg)
        : jit_generator(jit_name(), nullptr, MAX_CODE_SIZE, true, avx512_core)
        , brg(abrg)
        , postops_injector_(nullptr) {

        if (brg.with_eltwise || brg.with_binary || brg.with_sum) {

            static constexpr bool preserve_gpr = true;
            // we don't use zmm1 for storing vectors
            // so we don't need to preserve vmm
            static constexpr bool preserve_vmm = false;
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
                    static_cast<size_t>(Xbyak::Zmm(1).getIdx()), this->r14,
                    this->r15, this->r13, preserve_gpr, preserve_vmm,
                    GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(data_C_ptr_),
                    dst_md_wrapper, static_cast<size_t>(brg.ldb_tail),
                    ld_tail_mask, use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {
                    this->param1, enabled_bcast_strategy, rhs_sp};

            eltwise_injector::static_params_t esp;
            esp.preserve_vmm = preserve_vmm;
            esp.preserve_p_table = false;

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<avx512_core>>(
                    this, brg.attr->post_ops_, bsp, esp);

            using namespace dnnl::impl::cpu::binary_injector_utils;
            std::tie(with_binary_per_oc_bcast_, with_binary_per_oc_sp_bcast_,
                    with_binary_channel_bcast_, with_binary_per_mb_w_bcast_,
                    with_binary_per_w_bcast_, with_binary_no_bcast_)
                    = bcast_strategies_present_tup(brg.attr->post_ops_.entry_,
                            dst_md_wrapper, broadcasting_strategy_t::per_oc,
                            broadcasting_strategy_t::per_oc_spatial,
                            broadcasting_strategy_t::per_mb_spatial,
                            broadcasting_strategy_t::per_mb_w,
                            broadcasting_strategy_t::per_w,
                            broadcasting_strategy_t::no_broadcast);
            handle_binary_po_offset_ = with_binary_per_oc_bcast_
                    || with_binary_per_oc_sp_bcast_
                    || with_binary_channel_bcast_ || with_binary_per_mb_w_bcast_
                    || with_binary_per_w_bcast_ || with_binary_no_bcast_;
        }
        use_ils_ = brg.brgattr.use_interleave_stores;
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_amx_uker_base_t)

    brgemm_t brg;

private:
    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core>>
            postops_injector_;

    using reg64_t = const Xbyak::Reg64;
    enum {
        simd_w = 16,
        zmm_width_in_bytes = cpu_isa_traits<avx512_core>::vlen,
        tile_size = 1024
    };
    enum matrix_kind_t { matrix_A, matrix_B };

    // Register decomposition
    const reg64_t param1 = abi_param1;

    const reg64_t reg_addr_batch = r13;
    const reg64_t reg_aux1_batch = rbp;
    const reg64_t reg_aux_A = r11;
    const reg64_t reg_aux_B = r10;
    const reg64_t reg_stride_lda = r14;
    const reg64_t reg_stride_ldb = abi_not_param1;
    const reg64_t reg_C = r15;
    const reg64_t reg_D = r12;

    const reg64_t reg_buf = r8;
    const reg64_t reg_BS = rbx;
    const reg64_t reg_BS_loop = r9;
    const reg64_t reg_bias = rbx;
    const reg64_t reg_scales = rbx;

    const reg64_t reg_stride_ld_block = rdx;
    const reg64_t reg_do_post_ops = rbx;
    const reg64_t reg_tmp_gpr = rbx;
    const reg64_t reg_ptr_sum_scale = rbx;

    const reg64_t reg_zp_comp_a = rbx;
    const reg64_t reg_aux_zp_comp_a = rbx;
    const reg64_t reg_zp_comp_b = rbx;
    const reg64_t reg_zp_c_values = rbx;
    const reg64_t reg_ptr_sum_zp = r9;
    const reg64_t reg_bf32_stride = rsi;

    constexpr static int abi_param1_offs_ = 0;
    constexpr static int reg_zp_comp_a_offs_ = 8;
    constexpr static int reg_zp_comp_b_offs_ = 16;
    constexpr static int reg_zp_c_values_offs_ = 24;
    constexpr static int stack_space_needed_ = 32;

    bool are_post_ops_applicable_ = false;
    bool need_to_apply_alpha_beta_ = false;
    bool may_load_accumulators_ = false;

    bool handle_binary_po_offset_ = false;
    bool with_binary_per_oc_bcast_ = false;
    bool with_binary_per_oc_sp_bcast_ = false;
    bool with_binary_channel_bcast_ = false;
    bool with_binary_per_mb_w_bcast_ = false;
    bool with_binary_per_w_bcast_ = false;
    bool with_binary_no_bcast_ = false;
    bool prepare_post_ops_registers_once_ = false;

    size_t b_offset_ = 0;

    char *bd_mask_buffer_ptr_ = nullptr;
    std::vector<size_t> adj_bd_mask_buffer_;
    size_t *adj_bd_mask_buffer_ptr_ = nullptr;
    std::vector<size_t> skipped_bd_mask_buffer_;
    size_t *skipped_bd_mask_buffer_ptr_ = nullptr;
    palette_config_t palette_;
    // used to store offsets within wsp buffer where the data is
    // transformed(downconverted), to reuse when needed.
    std::unordered_map<std::string, size_t> transform_buf_map_A_;
    std::unordered_map<std::string, size_t> transform_buf_map_B_;

    size_t LDA_size_ = 0, LDA2_size_ = 0;
    size_t LDB_size_ = 0, LDB2_size_ = 0;
    size_t LDC_size_ = 0, LDC2_size_M_ = 0, LDC2_size_N_ = 0;
    size_t LDD_size_ = 0;
    size_t ld_block_B_size_ = 0;
    size_t ld_block_C_size_ = 0;
    size_t ld_block_D_size_ = 0;
    size_t ld_block_bias_size_ = 0;
    size_t ld_block_scales_size_ = 0;
    size_t ld_block_zp_size_ = 0;

    size_t ldb_tail_B_size_ = 0;
    size_t ldb_tail_C_size_ = 0;
    size_t ldb_tail_D_size_ = 0;
    size_t ldb_tail_zp_size_ = 0;

    // variables of bd loop
    size_t inp_bd_ = 0;

    // interleave stores
    bool use_ils_ = false;
    int ils_store_ops_ = 0, ils_vecs_per_store_ = 0;
    bool ils_buffer_ready_ = false;
    // saved parameters for storing
    int ils_bd_block2_ = 0, ils_ld_block2_ = 0, ils_inp_bd_ = 0,
        ils_ldb_ind_ = 0, ils_apply_post_ops_ = 0, ils_is_ld_tail_ = 0;
    // current storing coordinates
    int ils_vec_ = 0, ils_bdb_ = 0, ils_ldb_ = 0, ils_bd_start_ = 0,
        ils_bd_step_ = 0;
    int pfo_vec_ = 0, pfo_vecs_per_store_ = 0;

    bool dt_requires_saturation_ = false;

    Xbyak::Opmask ld_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask ld_tail_mask = Xbyak::Opmask(3);
    Xbyak::Opmask bf32_col_mask = Xbyak::Opmask(4);

    int adjusted_bd_block(int bdb) const noexcept {
        return (brg.is_M_tail && bdb == brg.bd_block2 - 1) ? brg.bdb_tail
                                                           : brg.bd_block;
    }

    Xbyak::Zmm accm(int bd) {
        assert(bd < 16);
        return Xbyak::Zmm(31 - bd);
    }

    const Xbyak::Zmm &zmm_tmp_1() const noexcept { return this->zmm0; }
    const Xbyak::Zmm &zmm_tmp_2() const noexcept { return this->zmm1; }
    const Xbyak::Zmm &zmm_tmp_3() const noexcept { return this->zmm2; }

    Xbyak::Zmm zmm_bias(int ldb) {
        assert(ldb < 3);
        // zmm6, zmm7, zmm8
        return Xbyak::Zmm(6 + ldb);
    }
    Xbyak::Zmm zmm_scales(int ldb) {
        assert(ldb < 3);
        // zmm9, zmm10, zmm11
        return Xbyak::Zmm(9 + ldb);
    }
    const Xbyak::Zmm zmm_bf32_pemute = zmm12;
    const Xbyak::Zmm zmm_zp_comp_a = zmm12;
    const Xbyak::Zmm zmm_zp_c = zmm13;
    const Xbyak::Zmm zmm_lbound = zmm14;
    const Xbyak::Zmm zmm_ubound = zmm15;

    Xbyak::Zmm zmm_mask(const Xbyak::Zmm zmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) const;
    Xbyak::Ymm ymm_mask(const Xbyak::Ymm ymm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) const;

    void cvt2ps(data_type_t type_in, const Xbyak::Zmm zmm_in,
            const Xbyak::Operand &op, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);

    void read_params();
    void load_accumulators(
            int bd_block2, int ld_block, bool is_ld_tail, int ldb_ind);

    void maybe_saturation(Xbyak::Zmm &zmm);
    void apply_alpha_beta_to_vector(
            const int idx, const Address &addr, bool is_ld_tail);
    void apply_post_ops_to_vector(const int idx, const Address &addr,
            const int bd, const int ldb, bool is_ld_tail);
    void apply_post_ops_to_range(int bd_start, int bd_finish, int bd_inp_bdb,
            int ldb, bool is_ld_tail);
    void store_vector_with_post_ops(const int idx, const Address &addr,
            const int bd, const int ldb, bool is_ld_tail);
    void prepare_post_ops_registers_ldb(
            int ldb, bool is_ld_tail, bool apply_post_ops);
    void prepare_post_ops_registers(
            int ldb_ind, int ld_block2, bool is_ld_tail, bool apply_post_ops);
    void prefetch_output_range(int bd_start, int bd_finish, int bd_inp_bdb,
            int ldb, bool apply_post_ops);
    void process_output_range(int bd_start, int bd_finish, int bd_inp_bdb,
            int bdb, int ldb_ind, int ldb, bool is_ld_tail,
            bool apply_post_ops);
    void store_vector_without_post_ops(
            const int idx, const Address &addr, bool is_ld_tail);
    void store_vector(const int idx, const int bd, const int ldb,
            const bool apply_post_ops, bool is_ld_tail);

    void prefetch_output_data(int ldb_ind, int bd_block2, int ld_block2);
    void interleave_store(bool store_all);

    void store_accumulators(int bd_block2, int ld_block, bool is_ld_tail,
            int ldb_ind, bool apply_post_ops);

    void set_A_B_matrices(int bs);
    void set_A_B_matrices();

    void bf32_downconvert(int num_rows, int tile_num_col_bytes,
            reg64_t reg_data, int offset, reg64_t reg_data_stride,
            reg64_t reg_buf, bool is_rd_tail);

    void bf32_downconvert_to_vnni(int num_rows, int tile_num_col_bytes,
            reg64_t reg_data, int offset, reg64_t reg_data_stride,
            reg64_t reg_buf, bool is_rd_tail);

    void maybe_pre_process_data(const Tmm &t1, reg64_t reg_base, size_t offset,
            reg64_t reg_stride, int bs, int rbd, matrix_kind_t mk,
            bool is_rd_tail);

    void maybe_tileloadd_nt(matrix_kind_t mk, int bs, int xdb, int rdb,
            size_t offset, bool is_rd_tail);

    void tdpbxxd(int bdb_idx, int ldb_idx, int bd_block2, int ld_block2,
            int ldb_ind);

    void gemm_microkernel_amx(int bs, int bd_block2, int ldb_ind, int ld_block2,
            bool is_rd_tail, bool is_ld_tail);

    void ldb_loop(int bd_block2, int ld_block, int ldb_loop_length,
            bool is_reg_tail, bool is_ld_tail, size_t c_offset, size_t d_offset,
            int ldb_ind, bool apply_post_ops);
    void bdb_loop(bool apply_post_ops);

    void generate() override;

    void prepare_bd_mask() noexcept;
    int skipped_bd_mask(int inp_bd) noexcept;

    size_t A_offset(int bdb) const noexcept;

    size_t B_offset(int ldb) const noexcept;
    size_t C_offset(int bd, int ldb) const noexcept;
    size_t C_block_offset(int bd, int ldb) const noexcept;
    size_t D_offset(int bd, int ldb) const noexcept;

    size_t lda() const noexcept;
    size_t ldb() const noexcept;
    size_t rdb_A_offset() const noexcept;
    size_t rdb_B_offset() const noexcept;

    size_t ldb_B_offset(int ld_block2, bool is_tail = false) const noexcept;
    size_t ldb_C_offset(int ld_block2, bool is_tail = false) const noexcept;
    size_t ldb_D_offset(int ld_block2, bool is_tail = false) const noexcept;

    size_t bias_offset(int ldb) const noexcept;

    size_t scales_offset(int ldb) const noexcept;
    size_t zp_comp_a_offset(int ldb) const noexcept;
    size_t zp_comp_b_offset(int bd) const noexcept;
    size_t zp_c_values_offset(int ldb, bool is_tail = false) const noexcept;
    int get_out_bd(int bd_inp_bdb, int bd) const;
};

void jit_brgemm_amx_uker_base_t::prepare_bd_mask() noexcept {
    if (!brg.brgattr.bd_mask_level) return;
    bd_mask_buffer_ptr_ = brg.brgattr.bd_mask;
    const auto bd_mask_size = brg.bcast_dim;
    adj_bd_mask_buffer_.resize(bd_mask_size);
    adj_bd_mask_buffer_ptr_ = adj_bd_mask_buffer_.data();
    skipped_bd_mask_buffer_.resize(bd_mask_size);
    skipped_bd_mask_buffer_ptr_ = skipped_bd_mask_buffer_.data();
    if (!utils::any_null(bd_mask_buffer_ptr_, adj_bd_mask_buffer_ptr_)) {
        int out_ibd = 0;
        for (int i = 0; i < bd_mask_size; i++) {
            adj_bd_mask_buffer_ptr_[i] = out_ibd;
            out_ibd += bd_mask_buffer_ptr_[i];
            skipped_bd_mask_buffer_ptr_[i] = i;
            for (auto ii = i; ii < bd_mask_size; ii++) {
                if (bd_mask_buffer_ptr_[ii]) {
                    skipped_bd_mask_buffer_ptr_[i] = ii;
                    break;
                }
            }
        }
    } else
        assert(!"struct nullptr error");
}

int jit_brgemm_amx_uker_base_t::skipped_bd_mask(int inp_bd) noexcept {
    if (brg.brgattr.bd_mask_level != 2)
        return inp_bd;
    else
        return skipped_bd_mask_buffer_ptr_[inp_bd];
}

size_t jit_brgemm_amx_uker_base_t::A_offset(int bdb) const noexcept {
    return bdb * LDA2_size_;
}

size_t jit_brgemm_amx_uker_base_t::B_offset(int ldb) const noexcept {
    return (brg.is_blocked ? 1 : brg.rd_step) * ldb * ld_block_B_size_;
}

size_t jit_brgemm_amx_uker_base_t::C_offset(int bd, int ldb) const noexcept {
    return bd * LDC_size_ + ldb * ld_block_C_size_;
}

size_t jit_brgemm_amx_uker_base_t::C_block_offset(
        int bd, int ldb) const noexcept {
    return (size_t)bd * LDC2_size_M_ + (size_t)ldb * LDC2_size_N_;
}

size_t jit_brgemm_amx_uker_base_t::D_offset(int bd, int ldb) const noexcept {
    return bd * LDD_size_ + ldb * ld_block_D_size_;
}

size_t jit_brgemm_amx_uker_base_t::lda() const noexcept {
    return LDA_size_;
}

size_t jit_brgemm_amx_uker_base_t::ldb() const noexcept {
    return LDB_size_ * brg.rd_step;
}

size_t jit_brgemm_amx_uker_base_t::rdb_A_offset() const noexcept {
    return brg.typesize_A * brg.rd_block;
}

size_t jit_brgemm_amx_uker_base_t::rdb_B_offset() const noexcept {
    return brg.rd_block * LDB_size_;
}

size_t jit_brgemm_amx_uker_base_t::ldb_B_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? ldb_tail_B_size_ * brg.ld_step
                     : ld_block2 * ld_block_B_size_ * brg.ld_step;
}

size_t jit_brgemm_amx_uker_base_t::ldb_C_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? ldb_tail_C_size_ : ld_block2 * ld_block_C_size_;
}

size_t jit_brgemm_amx_uker_base_t::ldb_D_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? ldb_tail_D_size_ : ld_block2 * ld_block_D_size_;
}

size_t jit_brgemm_amx_uker_base_t::bias_offset(int ldb) const noexcept {
    return ldb * ld_block_bias_size_;
}

size_t jit_brgemm_amx_uker_base_t::scales_offset(int ldb) const noexcept {
    return brg.is_oc_scale * ldb * ld_block_scales_size_;
}

size_t jit_brgemm_amx_uker_base_t::zp_comp_a_offset(int ldb) const noexcept {
    return ldb * ld_block_zp_size_;
}

size_t jit_brgemm_amx_uker_base_t::zp_comp_b_offset(int bd) const noexcept {
    return sizeof(int32_t) * bd;
}

size_t jit_brgemm_amx_uker_base_t::zp_c_values_offset(
        int ldb, bool is_tail) const noexcept {
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        return (is_tail) ? ldb_tail_zp_size_ : ldb * ld_block_zp_size_;
    }

    return 0;
}

int jit_brgemm_amx_uker_base_t::get_out_bd(int bd_inp_bdb, int bd) const {
    const auto bd_out_bd = bd_inp_bdb + bd;
    if (brg.brgattr.bd_mask_level && !bd_mask_buffer_ptr_[bd_out_bd])
        return -1;
    else {
        if (brg.brgattr.bd_mask_level)
            return adj_bd_mask_buffer_ptr_[bd_out_bd];
        else
            return bd_out_bd;
    }
}

Xbyak::Zmm jit_brgemm_amx_uker_base_t::zmm_mask(const Xbyak::Zmm zmm_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) const {
    return mask_flag ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                     : zmm_in;
}

Xbyak::Ymm jit_brgemm_amx_uker_base_t::ymm_mask(const Xbyak::Ymm ymm_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) const {
    return mask_flag ? (store ? ymm_in | ktail_mask : ymm_in | ktail_mask | T_z)
                     : ymm_in;
}

void jit_brgemm_amx_uker_base_t::cvt2ps(data_type_t type_in,
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

void jit_brgemm_amx_uker_base_t::read_params() {
    Label label_done;

    mov(reg_C, ptr[param1 + GET_OFF(ptr_C)]);
    mov(reg_D, ptr[param1 + GET_OFF(ptr_D)]);
    mov(reg_BS, ptr[param1 + GET_OFF(BS)]);

    mov(reg_addr_batch, ptr[param1 + GET_OFF(batch)]);

    mov(reg_buf, ptr[param1 + GET_OFF(ptr_buf)]);

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
}

void jit_brgemm_amx_uker_base_t::load_accumulators(
        int bd_block2, int ld_block2, bool is_ld_tail, int ldb_ind) {
    assert(IMPLICATION(is_ld_tail, ld_block2 == 1));
    if (may_load_accumulators_) mov(reg_stride_ld_block, LDC_size_);

    int bd_inp_bdb = inp_bd_;

    for (int bdb = 0; bdb < bd_block2; bdb++) {
        for (int ldb = 0; ldb < ld_block2; ldb++) {
            int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
            if (may_load_accumulators_) {
                const auto bd_out_bdb = get_out_bd(bd_inp_bdb, 0);
                const auto c_offset = C_block_offset(bd_out_bdb, ldb_ind + ldb);
                tileloadd(Tmm(brg.get_C_tensor(bdb, idx)),
                        ptr[reg_C + c_offset + reg_stride_ld_block]);
            } else {
                tilezero(Tmm(brg.get_C_tensor(bdb, idx)));
            }
        }
        bd_inp_bdb += brg.bd_block;
        bd_inp_bdb = skipped_bd_mask(bd_inp_bdb);
    }
}

void jit_brgemm_amx_uker_base_t::apply_alpha_beta_to_vector(
        const int idx, const Address &addr, bool is_ld_tail) {
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
    auto zmm = Zmm(idx);
    auto zmm_beta = zmm_tmp_1();
    auto zmm_alpha = zmm_tmp_2();
    auto zmm_prev_dst = zmm_tmp_3();

    const bool apply_alpha = brg.alpha != 1.f;
    const bool apply_beta = brg.beta != 0.f;
    if (!apply_alpha && !apply_beta) return;

    const bool dq2ps_required = brg.is_int8 && (apply_alpha || brg.beta != 1.f);
    const bool use_vadd_for_beta = brg.beta == 1.f && !dq2ps_required;

    if (apply_beta && !use_vadd_for_beta) {
        mov(reg_tmp_gpr, float2int(static_cast<float>(brg.beta)));
        vmovq(Xmm(zmm_beta.getIdx()), reg_tmp_gpr);
        vbroadcastss(zmm_beta, Xmm(zmm_beta.getIdx()));
    }
    if (apply_alpha) {
        mov(reg_tmp_gpr, float2int(static_cast<float>(brg.alpha)));
        vmovq(Xmm(zmm_alpha.getIdx()), reg_tmp_gpr);
        vbroadcastss(zmm_alpha, Xmm(zmm_alpha.getIdx()));
    }
    if (dq2ps_required) vcvtdq2ps(zmm, zmm);
    if (apply_alpha) vmulps(zmm, zmm, zmm_alpha);
    if (apply_beta) {
        if (use_vadd_for_beta) {
            auto zmm_masked = zmm | k_mask | T_z;
            if (brg.is_int8)
                vpaddd(zmm_masked, zmm, addr);
            else
                vaddps(zmm_masked, zmm, addr);
        } else {
            cvt2ps(brg.dt_c, zmm_prev_dst, addr, true, false, k_mask);
            vfmadd231ps(zmm, zmm_prev_dst, zmm_beta);
        }
    }
}

void jit_brgemm_amx_uker_base_t::apply_post_ops_to_vector(const int idx,
        const Address &addr, const int bd, const int ldb, bool is_ld_tail) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    auto zmm = Zmm(idx);

    if (brg.with_binary) {
        if (handle_binary_po_offset_) {
            rhs_arg_params.vmm_idx_to_out_reg.emplace(idx, reg_D);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                    idx, D_offset(bd, ldb));
            if (is_ld_tail) rhs_arg_params.vmm_tail_idx_.emplace(idx);
        }
    }

    const auto sum_injector = [&] {
        const float *p_sum_scale = &brg.sum_scale;
        const bool p_sum_scale_reg_set = *p_sum_scale != 1.f;
        const int32_t *p_sum_zp = &brg.sum_zp;
        const bool p_sum_zp_reg_set = *p_sum_zp != 0;

        if (p_sum_scale_reg_set) mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

        const auto &zmm_sum_zp = zmm_tmp_2();
        if (p_sum_zp_reg_set) {
            mov(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
            vcvtdq2ps(zmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
        }

        const auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

        const auto zmm_prev_dst = Xbyak::Zmm(0);
        cvt2ps(brg.sum_dt, zmm_prev_dst, addr, true, false, k_mask);
        if (p_sum_zp_reg_set) vsubps(zmm_prev_dst, zmm_sum_zp);
        if (!p_sum_scale_reg_set)
            vaddps(zmm, zmm_prev_dst);
        else
            vfmadd231ps(zmm, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
    };

    if (brg.with_sum) {
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }

    postops_injector_->compute_vector(zmm.getIdx(), rhs_arg_params);
}

void jit_brgemm_amx_uker_base_t::apply_post_ops_to_range(
        int bd_start, int bd_finish, int bd_inp_bdb, int ldb, bool is_ld_tail) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

    if (brg.with_binary) {
        if (handle_binary_po_offset_) {
            for (int bd = bd_start; bd < bd_finish; bd++) {
                // We have no way to tell the injector to skip some vectors.
                // Therefore, we must set parameters correctly for all registers.
                // TODO: Make it possible to specify "skipped" vectors to injector
                const auto idx = accm(bd).getIdx();
                if (is_ld_tail) rhs_arg_params.vmm_tail_idx_.emplace(idx);
                rhs_arg_params.vmm_idx_to_out_reg.emplace(idx, reg_D);

                const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
                if (bd_out_bd == -1) continue;

                const auto d_offset = D_offset(bd_out_bd, ldb);
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        idx, d_offset);
            }
        }
    }

    const auto sum_injector = [&] {
        const float *p_sum_scale = &brg.sum_scale;
        const int32_t *p_sum_zp = &brg.sum_zp;
        const bool p_sum_scale_reg_set = *p_sum_scale != 1.f;
        const bool p_sum_zp_reg_set = *p_sum_zp != 0;

        {
            if (p_sum_scale_reg_set)
                mov(reg_ptr_sum_scale, reinterpret_cast<size_t>(p_sum_scale));

            const auto &zmm_sum_zp = zmm_tmp_2();
            if (p_sum_zp_reg_set) {
                mov(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
                vcvtdq2ps(zmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
            }

            const auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
            const auto zmm_prev_dst = Xbyak::Zmm(0);

            for (int bd = bd_start; bd < bd_finish; bd++) {
                const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
                if (bd_out_bd == -1) continue;

                auto zmm = accm(bd);
                const auto d_offset = D_offset(bd_out_bd, ldb);
                auto addr = EVEX_compress_addr(reg_D, d_offset);

                cvt2ps(brg.sum_dt, zmm_prev_dst, addr, true, false, k_mask);
                if (p_sum_zp_reg_set) vsubps(zmm_prev_dst, zmm_sum_zp);
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
            32 - bd_finish, 32 - bd_start, rhs_arg_params);
}

void jit_brgemm_amx_uker_base_t::maybe_saturation(Xbyak::Zmm &zmm) {
    if (!dt_requires_saturation_) return;
    saturate_f32(zmm, zmm_lbound, zmm_ubound, brg.dt_d);
    vcvtps2dq(zmm, zmm);
}

void jit_brgemm_amx_uker_base_t::prepare_post_ops_registers_ldb(
        int ldb, bool is_ld_tail, bool apply_post_ops) {
    if (!apply_post_ops) return;
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(reg_aux_zp_comp_a, ptr[rsp + reg_zp_comp_a_offs_]);

        int zp_comp_a_off = zp_comp_a_offset(ldb);
        auto zp_comp_a_addr
                = EVEX_compress_addr(reg_aux_zp_comp_a, zp_comp_a_off);
        cvt2ps(data_type::s32, zmm_zp_comp_a, zp_comp_a_addr, true, false,
                k_mask);
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        mov(reg_zp_c_values, ptr[rsp + reg_zp_c_values_offs_]);
        if (brg.zp_type_c == brgemm_broadcast_t::per_tensor) {
            vcvtdq2ps(zmm_zp_c, EVEX_compress_addr(reg_zp_c_values, 0, true));
        }
        if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
            int zp_c_off = zp_c_values_offset(ldb, is_ld_tail);
            auto zp_c_addr = EVEX_compress_addr(reg_zp_c_values, zp_c_off);
            cvt2ps(data_type::s32, zmm_zp_c, zp_c_addr, true, false, k_mask);
        }
    }
}

void jit_brgemm_amx_uker_base_t::prepare_post_ops_registers(
        int ldb_ind, int ld_block2, bool is_ld_tail, bool apply_post_ops) {
    if (!apply_post_ops) return;
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
    if (brg.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);

        for (int ldb = 0; ldb < ld_block2; ldb++) {
            auto ptr_bias
                    = EVEX_compress_addr(reg_bias, bias_offset(ldb_ind + ldb));
            cvt2ps(brg.dt_bias, zmm_bias(ldb), ptr_bias, true, false, k_mask);
        }
    }

    if (brg.with_scales) {
        mov(reg_scales, ptr[param1 + GET_OFF(ptr_scales)]);
        for (int ldb = 0; ldb < ld_block2; ldb++) {
            auto scales_ptr = EVEX_compress_addr(
                    reg_scales, scales_offset(ldb_ind + ldb));
            vmovups(zmm_scales(ldb) | k_mask | T_z, scales_ptr);
        }
    }
}

void jit_brgemm_amx_uker_base_t::prefetch_output_range(int bd_start,
        int bd_finish, int bd_inp_bdb, int ldb, bool apply_post_ops) {

    for (int bd = bd_start; bd < bd_finish; bd++) {
        const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
        if (bd_out_bd == -1) continue;
        if (apply_post_ops) {
            const auto d_offset = D_offset(bd_out_bd, ldb);
            auto ptr_D = EVEX_compress_addr(reg_D, d_offset);
            prefetcht1(ptr_D);
        } else if (are_post_ops_applicable_) {
            const auto c_offset = C_offset(bd_out_bd, ldb);
            auto ptr_C = EVEX_compress_addr(reg_C, c_offset);
            prefetcht1(ptr_C);
        } else {
            const auto d_offset = D_offset(bd_out_bd, ldb);
            auto ptr_D = EVEX_compress_addr(reg_D, d_offset);
            prefetcht1(ptr_D);
        }
    }
}

void jit_brgemm_amx_uker_base_t::process_output_range(int bd_start,
        int bd_finish, int bd_inp_bdb, int bdb, int ldb_ind, int ldb,
        bool is_ld_tail, bool apply_post_ops) {

    const int wsp_offset = use_ils_
            ? (bdb * ils_ld_block2_ + ldb) * brg.bd_block * ld_block_C_size_
            : 0;

    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are already converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dq2ps_required = brg.is_int8
            && IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);

    bool some_bd_mask = false;
    for (int bd = bd_start; bd < bd_finish; bd++) {
        auto zmm = accm(bd);
        const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
        if (bd_out_bd == -1) continue;

        auto vreg_acc = is_ld_tail ? accm(bd) | ld_tail_mask | T_z : accm(bd);
        some_bd_mask = true;

        size_t buf_offset = bd * ld_block_C_size_;
        vmovups(vreg_acc, ptr[reg_buf + buf_offset + wsp_offset]);

        const auto c_offset = C_offset(bd_out_bd, ldb_ind + ldb);
        auto ptr_C = EVEX_compress_addr(reg_C, c_offset);

        if (need_to_apply_alpha_beta_)
            apply_alpha_beta_to_vector(zmm.getIdx(), ptr_C, is_ld_tail);

        if (!apply_post_ops) continue;

        if (dq2ps_required) vcvtdq2ps(zmm, zmm);
    }

    if (!apply_post_ops || !some_bd_mask) return;

    if (brg.with_bias) {
        for (int bd = bd_start; bd < bd_finish; bd++) {
            const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
            if (bd_out_bd == -1) continue;

            auto zmm = accm(bd);
            vaddps(zmm, zmm, zmm_bias(ldb));
        }
    }

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        for (int bd = bd_start; bd < bd_finish; bd++) {
            const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
            if (bd_out_bd == -1) continue;

            auto zmm = accm(bd);
            vaddps(zmm, zmm, zmm_zp_comp_a);
        }
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_b, ptr[rsp + reg_zp_comp_b_offs_]);

        auto zmm_zp_comp_b = zmm_tmp_1();
        for (int bd = bd_start; bd < bd_finish; bd++) {
            const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
            if (bd_out_bd == -1) continue;

            auto zmm = accm(bd);

            int zp_comp_b_off = zp_comp_b_offset(bd_out_bd);
            vcvtdq2ps(zmm_zp_comp_b,
                    EVEX_compress_addr(reg_zp_comp_b, zp_comp_b_off, true));

            vaddps(zmm, zmm, zmm_zp_comp_b);
        }
    }

    if (brg.with_scales) {
        for (int bd = bd_start; bd < bd_finish; bd++) {
            const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
            if (bd_out_bd == -1) continue;

            auto zmm = accm(bd);
            const Xbyak::Zmm scaled_zmm = zmm_mask(zmm, true, false, k_mask);
            vmulps(scaled_zmm, scaled_zmm, zmm_scales(ldb));
        }
    }

    if (postops_injector_) {
        apply_post_ops_to_range(
                bd_start, bd_finish, bd_inp_bdb, ldb_ind + ldb, is_ld_tail);
    }
}

void jit_brgemm_amx_uker_base_t::store_vector_with_post_ops(const int idx,
        const Address &addr, const int bd, const int ldb, bool is_ld_tail) {
    auto zmm = Zmm(idx);
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

    if (brg.zp_type_c != brgemm_broadcast_t::none) vaddps(zmm, zmm, zmm_zp_c);

    maybe_saturation(zmm);

    auto ymm = Xbyak::Ymm(idx);
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

void jit_brgemm_amx_uker_base_t::store_vector_without_post_ops(
        const int idx, const Address &addr, bool is_ld_tail) {
    auto zmm = Zmm(idx);

    maybe_saturation(zmm);

    if (is_ld_tail)
        vmovups(addr | ld_tail_mask | T_z, zmm);
    else
        vmovups(addr, zmm);
}

void jit_brgemm_amx_uker_base_t::store_vector(const int idx, const int bd,
        const int ldb, const bool apply_post_ops, bool is_ld_tail) {
    const auto c_offset = C_offset(bd, ldb);
    const auto d_offset = D_offset(bd, ldb);

    auto ptr_C = EVEX_compress_addr(reg_C, c_offset);
    auto ptr_D = EVEX_compress_addr(reg_D, d_offset);

    if (apply_post_ops)
        store_vector_with_post_ops(idx, ptr_D, bd, ldb, is_ld_tail);
    else if (are_post_ops_applicable_)
        store_vector_without_post_ops(idx, ptr_C, is_ld_tail);
    else
        store_vector_without_post_ops(idx, ptr_D, is_ld_tail);
}

void jit_brgemm_amx_uker_base_t::prefetch_output_data(
        int ldb_ind, int bd_block2, int ld_block2) {

    if (brg.brgattr.hint_prefetching
            != brgemm_kernel_prefetching_t::brgemm_prf_output1)
        return;

    // last bd_block may be bd_tail
    const auto last_bd_block = adjusted_bd_block(bd_block2 - 1);
    const auto bdb_row = brg.bd_block * ld_block2;
    const auto tot_vecs = (bd_block2 - 1) * bdb_row + last_bd_block * ld_block2;
    const auto nvecs = nstl::min(pfo_vecs_per_store_, tot_vecs - pfo_vec_);

    for (int vec = 0; vec < nvecs && pfo_vec_ < tot_vecs; vec++) {
        int bdb = pfo_vec_ / bdb_row;
        auto adj_bd_block = adjusted_bd_block(bdb);
        auto vec_in_bdb_row = pfo_vec_ - bdb * bdb_row;
        int ldb = vec_in_bdb_row / adj_bd_block;
        int bd = vec_in_bdb_row % adj_bd_block;

        auto bd_inp_bdb = skipped_bd_mask(inp_bd_);
        for (int ibdb = 0; ibdb < bdb; ibdb++) {
            bd_inp_bdb += brg.bd_block;
            bd_inp_bdb = skipped_bd_mask(bd_inp_bdb);
        }
        prefetch_output_range(
                0, 1, bd_inp_bdb + bd, ldb_ind + ldb, ils_apply_post_ops_);
        pfo_vec_++;
    }
}

void jit_brgemm_amx_uker_base_t::interleave_store(bool store_all) {

    if (!use_ils_) return;
    if (!ils_buffer_ready_) return;
    const auto need_apply_post_ops
            = are_post_ops_applicable_ && ils_apply_post_ops_;
    if (!need_to_apply_alpha_beta_ && !need_apply_post_ops
            && !brg.brgattr.bd_mask_level)
        return;

    int bd_inp_bdb = ils_inp_bd_;

    auto cur_bdb = ils_bdb_;
    auto cur_ldb = ils_ldb_;

    ils_bd_step_ = 3; // heuristic value

    // if first block
    if (ils_vec_ == 0) {
        if (!prepare_post_ops_registers_once_)
            prepare_post_ops_registers(ils_ldb_ind_, ils_ld_block2_,
                    ils_is_ld_tail_, ils_apply_post_ops_);
        prepare_post_ops_registers_ldb(
                ils_ldb_ind_, ils_is_ld_tail_, ils_apply_post_ops_);
        ils_bd_start_ = 0;
        auto bd_finish = nstl::min(ils_bd_step_, adjusted_bd_block(cur_bdb));
        process_output_range(0, bd_finish, bd_inp_bdb, cur_bdb, ils_ldb_ind_,
                cur_ldb, ils_is_ld_tail_, ils_apply_post_ops_);
    }

    // last bd_block may be bd_tail
    const auto last_bd_block = adjusted_bd_block(ils_bd_block2_ - 1);
    const auto bdb_row = brg.bd_block * ils_ld_block2_;
    const auto total_vectors
            = (ils_bd_block2_ - 1) * bdb_row + last_bd_block * ils_ld_block2_;
    const auto nvecs = store_all ? total_vectors : ils_vecs_per_store_;
    for (int vec = 0; vec < nvecs && ils_vec_ < total_vectors; vec++) {
        int bdb = ils_vec_ / bdb_row;
        auto adj_bd_block = adjusted_bd_block(bdb);
        auto vec_in_bdb_row = ils_vec_ - bdb * bdb_row;
        int ldb = vec_in_bdb_row / adj_bd_block;
        int bd = vec_in_bdb_row % adj_bd_block;

        auto bd_inp_bdb = ils_inp_bd_;
        for (int ibdb = 0; ibdb < bdb; ibdb++) {
            bd_inp_bdb += brg.bd_block;
            bd_inp_bdb = skipped_bd_mask(bd_inp_bdb);
        }
        if (ldb != cur_ldb)
            prepare_post_ops_registers_ldb(
                    ils_ldb_ind_ + ldb, ils_is_ld_tail_, ils_apply_post_ops_);

        if (bdb != cur_bdb || ldb != cur_ldb
                || rnd_dn(bd, ils_bd_step_) != ils_bd_start_) {
            ils_bd_start_ = rnd_dn(bd, ils_bd_step_);
            auto bd_finish
                    = nstl::min(ils_bd_start_ + ils_bd_step_, adj_bd_block);
            process_output_range(ils_bd_start_, bd_finish, bd_inp_bdb, bdb,
                    ils_ldb_ind_, ldb, ils_is_ld_tail_, ils_apply_post_ops_);
        }

        const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
        if (bd_out_bd != -1) {
            auto vreg_acc = ils_is_ld_tail_ ? accm(bd) | ld_tail_mask | T_z
                                            : accm(bd);

            store_vector(vreg_acc.getIdx(), bd_out_bd, ils_ldb_ind_ + ldb,
                    ils_apply_post_ops_, ils_is_ld_tail_);
        }
        cur_bdb = bdb;
        cur_ldb = ldb;
        ils_vec_++;
    }
    ils_ldb_ = cur_ldb;
    ils_bdb_ = cur_bdb;
}

void jit_brgemm_amx_uker_base_t::store_accumulators(int bd_block2,
        int ld_block2, bool is_ld_tail, int ldb_ind, bool apply_post_ops) {

    const bool need_to_apply_post_ops
            = are_post_ops_applicable_ && apply_post_ops;
    const auto store_by_vectors = need_to_apply_alpha_beta_
            || need_to_apply_post_ops || brg.brgattr.bd_mask_level;

    if (store_by_vectors)
        mov(reg_stride_ld_block, ld_block_C_size_);
    else
        mov(reg_stride_ld_block, LDC_size_);

    int bd_inp_bdb = inp_bd_;

    ils_bd_block2_ = bd_block2;
    ils_ld_block2_ = ld_block2;
    ils_inp_bd_ = inp_bd_;
    ils_ldb_ind_ = ldb_ind;
    ils_apply_post_ops_ = apply_post_ops;
    ils_is_ld_tail_ = is_ld_tail;
    ils_vec_ = 0;
    ils_bdb_ = 0;
    ils_ldb_ = 0;
    ils_buffer_ready_ = true;
    ils_store_ops_ = ld_block2 * bd_block2 * brg.bd_block;
    pfo_vec_ = 0;

    if (store_by_vectors && !use_ils_ && !prepare_post_ops_registers_once_)
        prepare_post_ops_registers(
                ldb_ind, ld_block2, is_ld_tail, apply_post_ops);

    for (int bdb = 0; bdb < bd_block2; bdb++) {
        int adj_bd_block = adjusted_bd_block(bdb);

        for (int ldb = 0; ldb < ld_block2; ldb++) {
            int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
            const int wsp_offset = use_ils_
                    ? (bdb * ld_block2 + ldb) * brg.bd_block * ld_block_C_size_
                    : 0;
            if (store_by_vectors) {
                tilestored(ptr[reg_buf + reg_stride_ld_block + wsp_offset],
                        Tmm(brg.get_C_tensor(bdb, idx)));
                if (use_ils_) continue;

                prepare_post_ops_registers_ldb(
                        ldb_ind + ldb, is_ld_tail, apply_post_ops);

                process_output_range(0, adj_bd_block, bd_inp_bdb, bdb, ldb_ind,
                        ldb, is_ld_tail, apply_post_ops);

                for (int bd = 0; bd < adj_bd_block; bd++) {
                    const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
                    if (bd_out_bd == -1) continue;

                    auto vreg_acc = is_ld_tail ? accm(bd) | ld_tail_mask | T_z
                                               : accm(bd);
                    store_vector(vreg_acc.getIdx(), bd_out_bd, ldb_ind + ldb,
                            apply_post_ops, is_ld_tail);
                }
            } else {
                const auto bd_out_bdb = get_out_bd(bd_inp_bdb, 0);
                const auto c_offset = C_block_offset(bd_out_bdb, ldb_ind + ldb);
                tilestored(ptr[reg_C + c_offset + reg_stride_ld_block],
                        Tmm(brg.get_C_tensor(bdb, idx)));
            }
        }
        bd_inp_bdb += brg.bd_block;
        bd_inp_bdb = skipped_bd_mask(bd_inp_bdb);
    }
}

void jit_brgemm_amx_uker_base_t::set_A_B_matrices(int bs) {
    assert(brg.type == brgemm_addr);
    auto batch_offset = (size_t)bs * sizeof(brgemm_batch_element_t);
    if (brg.layout == brgemm_row_major) {
        mov(reg_aux_A,
                EVEX_compress_addr(reg_addr_batch,
                        batch_offset + GET_OFF_BATCH_ELEMENT(ptr.A)));
        mov(reg_aux_B,
                EVEX_compress_addr(reg_addr_batch,
                        batch_offset + GET_OFF_BATCH_ELEMENT(ptr.B)));
    } else {
        mov(reg_aux_A,
                EVEX_compress_addr(reg_addr_batch,
                        batch_offset + GET_OFF_BATCH_ELEMENT(ptr.B)));
        mov(reg_aux_B,
                EVEX_compress_addr(reg_addr_batch,
                        batch_offset + GET_OFF_BATCH_ELEMENT(ptr.A)));
    }
}

void jit_brgemm_amx_uker_base_t::set_A_B_matrices() {
    assert(brg.type == brgemm_addr);
    assert(brg.brgattr.var_bs);

    if (brg.layout == brgemm_row_major) {
        mov(reg_aux_A, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.A)]);
        mov(reg_aux_B, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.B)]);
    } else {
        mov(reg_aux_A, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.B)]);
        mov(reg_aux_B, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.A)]);
    }
}

void jit_brgemm_amx_uker_base_t::maybe_tileloadd_nt(matrix_kind_t mk, int bs,
        int xdb, int rdb, size_t offset, bool is_rd_tail) {
    const bool try_load_nt_A
            = (brg.brgattr.hint_innermost_loop == brgemm_bd_loop_innermost);
    const bool try_load_nt_B
            = (brg.brgattr.hint_innermost_loop == brgemm_ld_loop_innermost);
    const bool is_A = mk == matrix_kind_t::matrix_A;
    bool try_load_nt = is_A ? try_load_nt_A : try_load_nt_B;
    try_load_nt = try_load_nt
            && (static_cast<size_t>(brg.typesize_A)
                               * brg.brgattr.hint_expected_A_size
                       + static_cast<size_t>(brg.typesize_B)
                               * brg.brgattr.hint_expected_B_size
                       + static_cast<size_t>(brg.typesize_C)
                               * brg.brgattr.hint_expected_C_size)
                    >= platform::get_per_core_cache_size(1);

    auto t1 = Tmm(is_A ? brg.get_A_tensor(xdb) : brg.get_B_tensor(xdb));
    auto reg_base = is_A ? reg_aux_A : reg_aux_B;
    auto reg_stride = is_A ? reg_stride_lda : reg_stride_ldb;

    if (brg.is_bf32)
        // try_load_nt is not supported in maybe_pre_process_data as there is
        // no guarantee that the data is cacheline aligned.
        maybe_pre_process_data(
                t1, reg_base, offset, reg_stride, bs, rdb, mk, is_rd_tail);
    else if (try_load_nt)
        tileloaddt1(t1, ptr[reg_base + offset + reg_stride]);
    else
        tileloadd(t1, ptr[reg_base + offset + reg_stride]);
}

void jit_brgemm_amx_uker_base_t::tdpbxxd(
        int bdb_idx, int ldb_idx, int bd_block2, int ld_block2, int ldb_ind) {
    prefetch_output_data(ldb_ind, bd_block2, ld_block2);
    const Tmm &x1 = Tmm(brg.get_C_tensor(bdb_idx, ldb_idx));
    const Tmm &x2 = Tmm(brg.get_A_tensor(bdb_idx));
    const Tmm &x3 = Tmm(brg.get_B_tensor(ldb_idx));
    if (brg.is_bf32
            || (brg.dt_a == data_type::bf16 && brg.dt_b == data_type::bf16)) {
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
    interleave_store(false);
}

// This method down-converts the data from f32 to bf16 and saves at reg_buf.
// Generally used by matrix_A, where no vnni transformation of data is needed.
void jit_brgemm_amx_uker_base_t::bf32_downconvert(int num_rows,
        int tile_num_col_bytes, reg64_t reg_data, int offset,
        reg64_t reg_data_stride, reg64_t reg_buf, bool is_rd_tail) {
    const int rd_block = is_rd_tail ? brg.rdb_tail : brg.rd_block;
    const int max_num_cols
            = nstl::min<int>(tile_num_col_bytes / sizeof(bfloat16_t), rd_block);
    const int col_tail = max_num_cols % simd_w;
    auto zmm_1 = zmm_tmp_1();
    auto zmm_2 = zmm_tmp_2();
    auto zmm_2_masked = col_tail ? zmm_2 | bf32_col_mask | T_z : zmm_2;

    assert(max_num_cols > 0);

    if (col_tail) {
        const int tail_mask = (1 << col_tail) - 1;
        auto reg_tmp_32 = reg_tmp_gpr.cvt32();
        mov(reg_tmp_32, tail_mask);
        kmovw(bf32_col_mask, reg_tmp_32);
    }

    // Note: using the same register used in col_tail, so order is important
    const auto reg_data_aux = reg_tmp_gpr;
    lea(reg_data_aux, ptr[reg_data + offset]);

    for (int r = 0; r < num_rows; ++r) {
        if (max_num_cols > 16) {
            vmovups(zmm_1, ptr[reg_data_aux]);
            vmovups(zmm_2_masked, ptr[reg_data_aux + zmm_width_in_bytes]);
            vcvtne2ps2bf16(zmm_1, zmm_2, zmm_1);
            // we assume enough padding space is available.
            vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_1);
        } else {
            auto ymm_1 = Ymm(zmm_1.getIdx());
            auto ymm_1_masked
                    = max_num_cols == 16 ? ymm_1 : ymm_1 | bf32_col_mask | T_z;
            vcvtneps2bf16(ymm_1_masked, ptr[reg_data_aux]);
            vmovups(ptr[reg_buf + r * zmm_width_in_bytes], ymm_1);
        }
        add(reg_data_aux, reg_data_stride);
    }
}

// This method down-converts and transforms the data from f32 to bf16_vnni
// format. Generally used by matrix_B.
void jit_brgemm_amx_uker_base_t::bf32_downconvert_to_vnni(int num_rows,
        int tile_num_col_bytes, reg64_t reg_data, int offset,
        reg64_t reg_data_stride, reg64_t reg_buf, bool is_rd_tail) {
    const int num_cols_ele = tile_num_col_bytes / sizeof(bfloat16_t);
    const int num_N = num_cols_ele / sizeof(bfloat16_t);
    const int col_tail = num_N % simd_w;
    const auto zmm_1 = zmm_tmp_1();
    const auto zmm_2 = zmm_tmp_2();

    assert(num_N > 0);

    auto load = [&](Zmm zmm, Address addr) {
        if (col_tail)
            vmovups(zmm | bf32_col_mask | T_z, addr);
        else
            vmovups(zmm, addr);
    };

    if (col_tail) {
        const int tail_mask = (1 << col_tail) - 1;
        auto reg_tmp_32 = reg_tmp_gpr.cvt32();
        mov(reg_tmp_32, tail_mask);
        kmovw(bf32_col_mask, reg_tmp_32);
    }

    // Note: using the same register used in col_tail, so order is important
    const auto reg_data_aux = reg_tmp_gpr;
    lea(reg_data_aux, ptr[reg_data + offset]);

    const int rd_block = is_rd_tail ? brg.rdb_tail : brg.rd_block;
    const int vnni_granularity
            = data_type_vnni_granularity(data_type_t::dnnl_bf16);
    const int r_end
            = nstl::min(utils::div_up(rd_block, vnni_granularity), num_rows);

    for (int r = 0; r < r_end; ++r) {
        load(zmm_1, ptr[reg_data_aux]);

        if (r * vnni_granularity + 1 >= rd_block) {
            vpxord(zmm_2, zmm_2, zmm_2);
        } else {
            load(zmm_2, ptr[reg_data_aux + reg_data_stride]);
        }

        vcvtne2ps2bf16(zmm_1, zmm_2, zmm_1);
        vpermw(zmm_1, zmm_bf32_pemute, zmm_1);
        vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_1);
        lea(reg_data_aux,
                ptr[reg_data_aux + vnni_granularity * reg_data_stride]);
    }

    // zero rest of the tile data
    if (r_end < num_rows) {
        vpxord(zmm_2, zmm_2, zmm_2);
        for (int r = r_end; r < num_rows; ++r)
            vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_2);
    }
}

void jit_brgemm_amx_uker_base_t::maybe_pre_process_data(const Tmm &t1,
        reg64_t reg_base, size_t offset, reg64_t reg_stride, int bs, int rdb,
        matrix_kind_t mk, bool is_rd_tail) {

    auto should_save_transform = [&](matrix_kind_t mk) {
        // save if there is a reuse
        if (mk == matrix_A) {
            return brg.ldb + (brg.ldb_tail != 0) > brg.ld_block2;
        } else {
            return brg.bdb + (brg.bdb_tail != 0) > brg.bd_block2;
        }
    };

    const bool is_A = mk == matrix_A;
    auto &transform_buf = is_A ? transform_buf_map_A_ : transform_buf_map_B_;

    const int transform_offset
            = use_ils_ ? brg.get_num_C_tiles() * tile_size : 0;
    const int max_bdb2 = brg.bd_block2;
    const int max_rdb = brg.rdb + (brg.rdb_tail != 0);
    const int matrix_a_offset = transform_offset;
    const int matrix_b_offset = transform_offset
            + tile_size
                    * (nstl::max<int>(should_save_transform(mk),
                            should_save_transform(matrix_A) * brg.brgattr.max_bs
                                    * max_bdb2 * max_rdb));
    const int matrix_offset = is_A ? matrix_a_offset : matrix_b_offset;
    const std::string key = std::to_string(bs) + "_" + std::to_string(offset);

    if (transform_buf.find(key) != transform_buf.end()) {
        auto buf_idx = transform_buf[key];
        auto offt = matrix_offset + buf_idx * tile_size;
        tileloadd(t1, ptr[reg_buf + reg_bf32_stride + offt]);
        return;
    }

    int buf_offt = matrix_offset;
    // save offset of the transformation if required.
    if (should_save_transform(mk)) {
        auto buf_idx = transform_buf.size();
        buf_offt = matrix_offset + buf_idx * tile_size;
        transform_buf[key] = buf_idx;
    }

    if (buf_offt) add(reg_buf, buf_offt);
    mov(reg_bf32_stride, zmm_width_in_bytes);

    assert(t1.getIdx() >= 0 && t1.getIdx() < 16);
    const int num_rows = palette_.rows[t1.getIdx()];
    const int num_col_bytes = palette_.cols[t1.getIdx()];
    if (is_A) {
        bf32_downconvert(num_rows, num_col_bytes, reg_base, offset, reg_stride,
                reg_buf, is_rd_tail);
    } else {
        bf32_downconvert_to_vnni(num_rows, num_col_bytes, reg_base, offset,
                reg_stride, reg_buf, is_rd_tail);
    }

    // load into tmm from the transformed data.
    tileloadd(t1, ptr[reg_buf + reg_bf32_stride]);

    // reset buf pointer.
    if (buf_offt) sub(reg_buf, buf_offt);
}

void jit_brgemm_amx_uker_base_t::gemm_microkernel_amx(int bs, int bd_block2,
        int ldb_ind, int ld_block2, bool is_rd_tail, bool is_ld_tail) {

    int rbd_block = 0;
    size_t a_offset = 0, b_offset = 0;
    if (is_rd_tail) {
        rbd_block = 1;
        if (brg.rdb > 0) {
            a_offset = brg.rdb * rdb_A_offset();
            b_offset = brg.rdb * rdb_B_offset();
        }
    } else {
        rbd_block = brg.rdb;
        b_offset = a_offset = 0;
    }

    for (int rdb = 0; rdb < rbd_block; rdb++) {
        int bd_inp_bdb = inp_bd_;
        const size_t rdb_A_off
                = static_cast<size_t>(rdb) * rdb_A_offset() + a_offset;
        const size_t rdb_B_off = static_cast<size_t>(rdb) * rdb_B_offset()
                + b_offset_ + b_offset;

        for (int bdb = 0; bdb < bd_block2; bdb++) {
            maybe_tileloadd_nt(matrix_kind_t::matrix_A, bs, bdb, rdb,
                    rdb_A_off + A_offset(bd_inp_bdb), is_rd_tail);
            bd_inp_bdb += brg.bd_block;
            bd_inp_bdb = skipped_bd_mask(bd_inp_bdb);
            for (int ldb = 0; ldb < ld_block2; ldb++) {
                if (bdb == 0) {
                    const int ldb_idx = (is_ld_tail) ? brg.ld_block2 : ldb;
                    maybe_tileloadd_nt(matrix_kind_t::matrix_B, bs, ldb_idx,
                            rdb, rdb_B_off + B_offset(ldb), is_rd_tail);
                }
                if (ldb == 0) {
                    if (bdb > 0) {
                        const int ldb_idx
                                = (is_ld_tail) ? brg.ld_block2 : ld_block2 - 1;
                        const int bdb_idx = bdb - 1;
                        tdpbxxd(bdb_idx, ldb_idx, bd_block2, ld_block2,
                                ldb_ind);
                    }
                } else {
                    const int ldb_idx
                            = (is_ld_tail) ? brg.ld_block2 : (ldb - 1);
                    const int bdb_idx = bdb;
                    tdpbxxd(bdb_idx, ldb_idx, bd_block2, ld_block2, ldb_ind);
                }
            }
        }
        // last tdpbxxd
        {
            const int ldb_idx = (is_ld_tail) ? brg.ld_block2 : (ld_block2 - 1);
            const int bdb_idx = bd_block2 - 1;
            tdpbxxd(bdb_idx, ldb_idx, bd_block2, ld_block2, ldb_ind);
        }
    }
}

void jit_brgemm_amx_uker_base_t::ldb_loop(int bd_block2, int ld_block2,
        int ldb_loop_length, bool is_reg_tail, bool is_ld_tail, size_t c_offset,
        size_t d_offset, int ldb_ind, bool apply_post_ops) {

    if (!is_reg_tail) b_offset_ = 0;

    for (int l_ldb = 0; l_ldb < ldb_loop_length; l_ldb++) {
        int calc_ops = brg.brgattr.max_bs * (brg.rdb + (brg.rdb_tail ? 1 : 0))
                * ld_block2 * bd_block2;
        ils_vecs_per_store_ = (calc_ops) ? div_up(ils_store_ops_, calc_ops) : 0;
        pfo_vecs_per_store_ = ils_vecs_per_store_;

        const auto l_ldb_ind = l_ldb * (is_ld_tail ? brg.ldb_tail : ld_block2);

        load_accumulators(
                bd_block2, ld_block2, is_ld_tail, ldb_ind + l_ldb_ind);

        if (brg.alpha != 0.f) {
            if (brg.brgattr.var_bs) {
                Label BS_loop_label, end_BS_loop_label;

                mov(reg_BS_loop, reg_BS);
                cmp(reg_BS_loop, 0);
                jz(end_BS_loop_label, T_NEAR);
                mov(reg_aux1_batch, reg_addr_batch);

                L_aligned(BS_loop_label, 64);
                {
                    set_A_B_matrices();
                    add(reg_aux1_batch, sizeof(brgemm_batch_element_t));
                    prefetcht0(ptr[reg_aux1_batch]);
                    gemm_microkernel_amx(0, bd_block2, ldb_ind, ld_block2,
                            false, is_ld_tail);

                    if (brg.rdb_tail != 0) {
                        gemm_microkernel_amx(0, bd_block2, ldb_ind, ld_block2,
                                true, is_ld_tail);
                    }
                    dec(reg_BS_loop);
                    cmp(reg_BS_loop, 0);
                    jg(BS_loop_label, T_NEAR);
                }
                L(end_BS_loop_label);

            } else {
                for (int bs = 0; bs < brg.brgattr.max_bs; bs++) {
                    set_A_B_matrices(bs);
                    gemm_microkernel_amx(bs, bd_block2, ldb_ind, ld_block2,
                            false, is_ld_tail);

                    if (brg.rdb_tail != 0) {
                        gemm_microkernel_amx(bs, bd_block2, ldb_ind, ld_block2,
                                true, is_ld_tail);
                    }
                }
            }
        }
        store_accumulators(bd_block2, ld_block2, is_ld_tail,
                ldb_ind + l_ldb_ind, apply_post_ops);

        b_offset_ += (is_ld_tail) ? ldb_B_offset(1, true)
                                  : ldb_B_offset(ld_block2);
    }
}

void jit_brgemm_amx_uker_base_t::bdb_loop(bool apply_post_ops) {
    ils_buffer_ready_ = false;
    ils_apply_post_ops_ = apply_post_ops;

    // for many primitives which use brgemm the brg.ldb2 is equal or less than 1
    // so we can read post ops data only once per brgemm call
    if (brg.ldb2 > 1) {
        prepare_post_ops_registers_once_ = false;
    } else if (brg.ldb2 == 1) {
        if (brg.ldb2_tail == 0 && brg.ldb_tail == 0) {
            prepare_post_ops_registers_once_ = true;
            prepare_post_ops_registers(0, brg.ld_block2, false, apply_post_ops);
        }
    } else if (brg.ldb2_tail > 0) {
        if (brg.ldb_tail == 0) {
            prepare_post_ops_registers_once_ = true;
            prepare_post_ops_registers(0, brg.ldb2_tail, false, apply_post_ops);
        }
    } else {
        prepare_post_ops_registers_once_ = true;
        prepare_post_ops_registers(0, 1, true, apply_post_ops);
    }

    if (apply_post_ops)
        dt_requires_saturation_ = one_of(
                brg.dt_d, data_type::u8, data_type::s8, data_type::s32);
    else {
        // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
        // accumulated values are converted to ps in apply_alpha_beta()
        const bool alpha_or_beta_applicable
                = brg.alpha != 1.0f || brg.beta != 0.f;
        const bool beta_uses_vadd = brg.beta == 1.f
                && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
        dt_requires_saturation_ = brg.is_int8
                && !IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);
    }
    if (dt_requires_saturation_) {
        init_saturate_f32(
                zmm_lbound, zmm_ubound, reg_tmp_gpr, data_type::f32, brg.dt_d);
    }

    auto do_ldb_loop = [=](int bd_block2, bool apply_post_ops) {
        size_t c_offset = 0;
        size_t d_offset = 0;
        int ldb_ind = 0;
        // clear the transform cache for A, as the existing data is invalid as
        // we move to next bdb2 block.
        transform_buf_map_A_.clear();
        if (brg.ldb2 > 0) {
            const bool is_ld_reg_tail = false;
            const bool is_ld_tail = false;
            ldb_loop(bd_block2, brg.ld_block2, brg.ldb2, is_ld_reg_tail,
                    is_ld_tail, c_offset, d_offset, ldb_ind, apply_post_ops);
            c_offset += brg.ldb2 * ldb_C_offset(brg.ld_block2);
            d_offset += brg.ldb2 * ldb_D_offset(brg.ld_block2);
            ldb_ind += brg.ldb2 * brg.ld_block2;
        }
        if (brg.ldb2_tail > 0) {
            const bool is_ld_reg_tail = (brg.ldb2 == 0) ? false : true;
            const bool is_ld_tail = false;
            ldb_loop(bd_block2, brg.ldb2_tail, 1, is_ld_reg_tail, is_ld_tail,
                    c_offset, d_offset, ldb_ind, apply_post_ops);
            c_offset += ldb_C_offset(brg.ldb2_tail);
            d_offset += ldb_D_offset(brg.ldb2_tail);
            ldb_ind += brg.ldb2_tail;
        }
        if (brg.ldb_tail > 0) {
            const bool is_ld_reg_tail
                    = (brg.ldb2 == 0 && brg.ldb2_tail == 0) ? false : true;
            const bool is_ld_tail = true;
            ldb_loop(bd_block2, 1, 1, is_ld_reg_tail, is_ld_tail, c_offset,
                    d_offset, ldb_ind, apply_post_ops);
            c_offset += ldb_C_offset(1, true);
            d_offset += ldb_D_offset(1, true);
            ldb_ind += 1;
        }
        // rewind inp_bd
        for (int bdb = 0; bdb < bd_block2; bdb++) {
            inp_bd_ = skipped_bd_mask(inp_bd_);
            inp_bd_ += brg.bd_block;
        }
        inp_bd_ = skipped_bd_mask(inp_bd_);
    };

    inp_bd_ = skipped_bd_mask(0);
    for (int bdb2 = 0; bdb2 < brg.bdb2 && brg.bd_block2 > 1; bdb2++) {
        do_ldb_loop(brg.bd_block2, apply_post_ops);
    }
    if (brg.bdb2_tail > 0) { do_ldb_loop(brg.bdb2_tail, apply_post_ops); }
    if (!brg.is_M_tail && brg.bdb_tail > 0) do_ldb_loop(1, apply_post_ops);
    interleave_store(true);
}

void jit_brgemm_amx_uker_base_t::generate() {
    preamble();

    sub(rsp, stack_space_needed_);

    const auto full_mask = size_t {0xffffffffffffffff};
    const auto tail_mask = size_t((1 << brg.ldb_tail) - 1);
    LDA_size_ = brg.typesize_A * brg.LDA;
    LDB_size_ = brg.typesize_B * brg.LDB;
    LDC_size_ = brg.typesize_C * brg.LDC;
    LDD_size_ = brg.typesize_D * brg.LDD;

    LDA2_size_ = brg.typesize_A * brg.LDA2;
    LDB2_size_ = brg.typesize_B * brg.LDB2;
    LDC2_size_M_ = brg.typesize_C * brg.LDC2_M;
    LDC2_size_N_ = brg.typesize_C * brg.LDC2_N;

    ld_block_B_size_ = brg.typesize_B
            * ((brg.brgattr.LDB2 != 0) ? brg.brgattr.LDB2 : brg.ld_block);
    ld_block_C_size_ = brg.typesize_C * brg.ld_block;
    ld_block_D_size_ = brg.typesize_D * brg.ld_block;
    ld_block_bias_size_ = brg.typesize_bias * brg.ld_block;
    ld_block_scales_size_ = sizeof(float) * brg.ld_block;
    ld_block_zp_size_ = sizeof(int32_t) * brg.ld_block;
    ldb_tail_B_size_ = brg.typesize_B * brg.ldb_tail;
    ldb_tail_C_size_ = brg.typesize_C * brg.ldb_tail;
    ldb_tail_D_size_ = brg.typesize_D * brg.ldb_tail;
    ldb_tail_zp_size_ = sizeof(int32_t) * brg.ldb_tail;

    // if beta == 1 and C datatype is f32 it is better to perform addition by
    // reading tiles directly from C instead of by reading/writing by vectors
    may_load_accumulators_
            = (brg.beta == 1.f && brg.dt_c == data_type::f32 && !brg.is_bf32);
    need_to_apply_alpha_beta_
            = (brg.beta != 0.f && !may_load_accumulators_) || brg.alpha != 1.f;
    const bool has_zero_points = !everyone_is(brgemm_broadcast_t::none,
            brg.zp_type_a, brg.zp_type_b, brg.zp_type_c);
    are_post_ops_applicable_ = one_of(true, brg.with_eltwise, brg.with_binary,
            brg.with_scales, brg.with_bias, brg.with_sum, brg.dt_d != brg.dt_c,
            has_zero_points);

    // second level blocking eligible only if we don't use store by vectors for now

    assert(IMPLICATION(are_post_ops_applicable_ || need_to_apply_alpha_beta_
                    || brg.brgattr.bd_mask_level,
            !brg.is_blocked && !brg.brgattr.var_bs));

    //! 'maybe_pre_process_data' for 'is_bf32' uses 'bs' value,
    //! so we can't support var_bs for is_bf32 for this moment
    assert(IMPLICATION(brg.brgattr.var_bs, !brg.is_bf32));

    Label permute_index_table;
    if (brg.is_bf32) {
        brgemm_init_tiles(brg, (char *)(&palette_));
        mov(reg_tmp_gpr, permute_index_table);
        vmovups(zmm_bf32_pemute, ptr[reg_tmp_gpr]);
    }

    reg64_t reg_mask = rax;

    mov(reg_mask, full_mask);
    kmovq(ld_full_mask, reg_mask);
    mov(reg_mask, tail_mask);
    kmovq(ld_tail_mask, reg_mask);

    mov(reg_stride_lda, lda());
    mov(reg_stride_ldb, ldb());

    read_params();

    prepare_bd_mask();

    Label label_to_ret;
    if (are_post_ops_applicable_) {
        Label label_store_without_post_ops;
        mov(reg_do_post_ops, ptr[param1 + GET_OFF(do_post_ops)]);
        cmp(reg_do_post_ops, 0);
        jz(label_store_without_post_ops, T_NEAR);
        bdb_loop(true);
        jmp(label_to_ret, T_NEAR);
        transform_buf_map_A_.clear();
        transform_buf_map_B_.clear();
        L(label_store_without_post_ops);
    }
    bdb_loop(false);
    L(label_to_ret);

    add(rsp, stack_space_needed_);

    postamble();

    if (brg.with_eltwise) postops_injector_->prepare_table();

    if (brg.is_bf32) {
        align(64);
        L(permute_index_table);
        const uint16_t _idx[32] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6,
                22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30,
                15, 31};
        for (size_t i = 0; i < 32; ++i)
            dw(_idx[i]);
    }
}

brgemm_amx_uker_t::brgemm_amx_uker_t(const brgemm_t abrd) {
    brgemm_kernel_ = new jit_brgemm_amx_uker_base_t(abrd);
}

status_t brgemm_amx_uker_t::create_kernel() {
    return brgemm_kernel_->create_kernel();
}

void brgemm_amx_uker_t::operator()(brgemm_kernel_params_t *params) const {
    (*brgemm_kernel_)(params);
}

brgemm_amx_uker_t::~brgemm_amx_uker_t() {
    delete brgemm_kernel_;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
