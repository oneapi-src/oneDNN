/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
                    this->r15, preserve_gpr, preserve_vmm,
                    GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(data_C_ptr_),
                    dst_md_wrapper, static_cast<size_t>(brg.ldb_tail),
                    ld_tail_mask, use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {
                    this->param1, enabled_bcast_strategy, rhs_sp};

            eltwise_injector::static_params_t esp;
            esp.preserve_vmm = preserve_vmm;
            esp.preserve_p_table = false;

            postops_injector_ = utils::make_unique<po_injector_t>(
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
    static constexpr cpu_isa_t po_isa_ = avx512_core_fp16;
    using po_injector_t = injector::jit_uni_postops_injector_t<po_isa_>;
    std::unique_ptr<po_injector_t> postops_injector_;

    using reg64_t = const Xbyak::Reg64;
    enum {
        simd_w = 16,
        zmm_width_in_bytes = cpu_isa_traits<avx512_core>::vlen,
        tile_size = 1024
    };

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

    enum matrix_kind_t { matrix_A, matrix_B, matrix_C, matrix_D };

    // Loops in brgemm kernel are (two outermost loops depend on loop order):
    // by bd block2
    //     by ld block2
    //          by batch_size
    //              by rd block
    //                  gemm_microkernel
    // Structures below (dim_iteration_t, bd_iteration_t, bs_iteration_t and
    // iteration_map_t) describe the structure of cycles
    // and are used for JIT code generation
    struct dim_iteration_t {
        size_t idx = 0;
        size_t pos = 0;
        int block = 0;
        int block2 = 0;
        bool is_tail = false;
        dim_iteration_t() = default;
        dim_iteration_t(
                size_t pos_, int block_, int block2_, bool is_tail_ = false)
            : pos(pos_), block(block_), block2(block2_), is_tail(is_tail_) {}
    };

    struct bd_iteration_t : public dim_iteration_t {
        std ::vector<size_t> bdb_pos;
        bd_iteration_t() = default;
        bd_iteration_t(
                size_t pos_, int block_, int block2_, bool is_tail_ = false)
            : dim_iteration_t(pos_, block_, block2_, is_tail_) {}
    };

    struct bs_iteration_t {
        size_t idx = 0;
        size_t pos = 0;
        bool is_first = false;
        bool is_last = false;
        bs_iteration_t() = default;
        bs_iteration_t(
                size_t pos_, bool is_first_ = true, bool is_last_ = false)
            : pos(pos_), is_first(is_first_), is_last(is_last_) {}
    };

    struct iteration_map_t {
        std::vector<dim_iteration_t> ldis;
        std::vector<bd_iteration_t> bdis;
        std::vector<bs_iteration_t> bsis;
        std::vector<dim_iteration_t> rdis;
        bool is_last_ldi(const dim_iteration_t &ldi) const {
            return (ldi.idx == ldis.size() - 1);
        }
        bool is_last_bdi(const dim_iteration_t &bdi) const {
            return (bdi.idx == bdis.size() - 1);
        }
        bool is_last_rdi(const dim_iteration_t &rdi) const {
            return (rdi.idx == rdis.size() - 1);
        }
    };

    struct brgemm_iteration_t {
        bd_iteration_t bdi;
        dim_iteration_t ldi;
        bs_iteration_t bsi;
        dim_iteration_t rdi;
        brgemm_iteration_t *prev_iter = nullptr;
        bool apply_postops = false;
        brgemm_iteration_t() = default;
    };

    struct prf_t {
        int dist = -1;
        int pfo_vec = 0;
    };

    // iteration map
    iteration_map_t imap_;

    // interleave stores
    bool use_ils_ = false;
    int ils_store_ops_ = 0, ils_vecs_per_store_ = 0;
    bool ils_buffer_ready_ = false;
    // saved parameters for storing
    brgemm_iteration_t ils_bi_;
    // current storing coordinates
    int ils_vec_ = 0, ils_bdb_ = 0, ils_ldb_ = 0, ils_bd_start_ = 0,
        ils_bd_step_ = 0;
    int pfo_vec_ = 0, pfo_vecs_per_store_ = 0;

    bool dt_requires_saturation_ = false;

    Xbyak::Opmask ld_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask ld_tail_mask = Xbyak::Opmask(3);
    Xbyak::Opmask bf32_col_mask = Xbyak::Opmask(4);

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
    void load_accumulators(brgemm_iteration_t &bi);

    void maybe_saturation(Xbyak::Zmm &zmm);
    void apply_alpha_beta_to_vector(
            const int idx, const Address &addr, bool is_ld_tail);
    void apply_post_ops_to_range(brgemm_iteration_t &bi, int bd_start,
            int bd_finish, int bd_inp_bdb, int ldb);
    void store_vector_with_post_ops(const int idx, const Address &addr,
            const int bd, const int ldb, bool is_ld_tail);
    void prepare_post_ops_registers_ldb(brgemm_iteration_t &bi, int ldb);
    void prepare_post_ops_registers(brgemm_iteration_t &bi);

    bool bi_shift_output(
            brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi);
    bool bi_shift_A(
            brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi);
    bool bi_shift_B(
            brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi);

    void prefetch_output_range(brgemm_iteration_t &bi,
            brgemm_kernel_prefetching_t pft, int bd_start, int bd_finish,
            int bd_inp_bdb, int ldb);
    void prefetch_output_data(brgemm_iteration_t &bi, bool first_bs,
            bool last_bs, bool prefetch_all);

    void process_output_range(brgemm_iteration_t &bi, int bd_start,
            int bd_finish, int bd_inp_bdb, int bdb, int ldb);
    void store_vector_without_post_ops(
            const int idx, const Address &addr, bool is_ld_tail);
    void store_vector(
            brgemm_iteration_t &bi, const int idx, const int bd, const int ldb);

    void interleave_store(brgemm_iteration_t &bi, bool store_all);

    void store_accumulators(brgemm_iteration_t &bi);

    void set_A_B_matrices(int bs);
    void set_A_B_matrices();

    void bf32_downconvert(int num_rows, int tile_num_col_bytes,
            reg64_t reg_data, int offset, reg64_t reg_data_stride,
            reg64_t reg_buf, bool is_rd_tail);

    void bf32_downconvert_to_vnni(int num_rows, int tile_num_col_bytes,
            reg64_t reg_data, int offset, reg64_t reg_data_stride,
            reg64_t reg_buf, bool is_rd_tail);

    void maybe_pre_process_data(brgemm_iteration_t &bi, const Tmm &t1,
            reg64_t reg_base, size_t offset, reg64_t reg_stride,
            matrix_kind_t mk);

    void maybe_tileloadd_nt(
            brgemm_iteration_t &bi, matrix_kind_t mk, int xdb, size_t offset);

    void tdpbxxd(brgemm_iteration_t &bi, int bdb_idx, int ldb_idx);

    void gemm_microkernel_amx(brgemm_iteration_t &bi);

    void rdb_loop(brgemm_iteration_t &bi);

    void bs_loop_body(brgemm_iteration_t &bi);
    void bs_loop(brgemm_iteration_t &bi);

    void ldb_loop_body(brgemm_iteration_t &bi);
    void ldb_loop(brgemm_iteration_t &bi);

    void bdb_loop_body(brgemm_iteration_t &bi);
    void bdb_loop(brgemm_iteration_t &bi);

    void init(brgemm_iteration_t &bi);
    void generate() override;

    void prepare_bd_mask() noexcept;
    int skipped_bd_mask(int inp_bd) noexcept;

    bool get_store_by_vectors(bool apply_post_ops) const {
        const bool need_to_apply_post_ops
                = are_post_ops_applicable_ && apply_post_ops;
        const auto store_by_vectors = need_to_apply_alpha_beta_
                || need_to_apply_post_ops || brg.brgattr.bd_mask_level;
        return store_by_vectors;
    }

    // returns bd position in input for bd block based on base_bd
    // and defined by bdb_shift
    int get_bd_inp_bdb(int base_bd, int bdb_shift) {
        int res = base_bd;
        for (int bdb = 0; bdb < bdb_shift; bdb++) {
            res += brg.bd_block;
            res = skipped_bd_mask(res);
        }
        return res;
    }

    size_t A_offset(int bdb) const noexcept;

    size_t B_offset(int ldb) const noexcept;
    size_t C_offset(int bd, int ldb) const noexcept;
    size_t C_block_offset(int bd, int ldb) const noexcept;
    size_t D_offset(int bd, int ldb) const noexcept;

    size_t lda() const noexcept;
    size_t ldb() const noexcept;
    size_t rdb_A_offset(const brgemm_iteration_t &bi) const noexcept;
    size_t rdb_B_offset(const brgemm_iteration_t &bi) const noexcept;
    size_t ldb_B_offset(const brgemm_iteration_t &bi) const noexcept;

    size_t bias_offset(int ldb) const noexcept;

    size_t scales_offset(int ldb) const noexcept;
    size_t zp_comp_a_offset(int ldb) const noexcept;
    size_t zp_comp_b_offset(int bd) const noexcept;
    size_t zp_c_values_offset(brgemm_iteration_t &bi, int ldb) const noexcept;
    int get_out_bd(int bd_inp_bdb, int bd) const;

    int get_C_tensor(brgemm_iteration_t &bi, int m, int n) const noexcept;
    void top_loop(brgemm_iteration_t &bi);

    int calc_ops_bs_loop(brgemm_iteration_t &bi) const noexcept;
    void fill_imap();
};

bool jit_brgemm_amx_uker_base_t::bi_shift_output(
        brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi) {
    res_bi = bi;
    if (shift == 0) return true;
    size_t lidx = 0;
    size_t bd_idx = 0;
    size_t ld_idx = 0;
    if (brg.brgattr.hint_innermost_loop == brgemm_ld_loop_innermost) {
        lidx = bi.bdi.idx * imap_.ldis.size() + bi.ldi.idx;
        lidx += shift;
        bd_idx = lidx / imap_.ldis.size();
        ld_idx = lidx % imap_.ldis.size();
    } else if (brg.brgattr.hint_innermost_loop == brgemm_bd_loop_innermost) {
        lidx = bi.ldi.idx * imap_.bdis.size() + bi.bdi.idx;
        lidx += shift;
        ld_idx = lidx / imap_.bdis.size();
        bd_idx = lidx % imap_.bdis.size();
    } else
        assert(!"Unknown loop order!");
    if (lidx >= imap_.ldis.size() * imap_.bdis.size()) return false;
    res_bi.bdi = imap_.bdis[bd_idx];
    res_bi.ldi = imap_.ldis[ld_idx];

    return true;
}

int jit_brgemm_amx_uker_base_t::calc_ops_bs_loop(brgemm_iteration_t &bi) const
        noexcept {
    return (brg.rdb + (brg.rdb_tail ? 1 : 0)) * bi.ldi.block2 * bi.bdi.block2
            * (brg.brgattr.var_bs ? 1 : brg.brgattr.max_bs);
}

bool jit_brgemm_amx_uker_base_t::bi_shift_A(
        brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi) {
    res_bi = bi;
    size_t lidx = bi.bdi.idx * imap_.rdis.size() + bi.rdi.idx;
    lidx += shift;
    if (lidx >= imap_.rdis.size() * imap_.bdis.size()) return false;

    size_t bd_idx = lidx / imap_.rdis.size();
    size_t rd_idx = lidx % imap_.rdis.size();

    if (lidx >= imap_.rdis.size() * imap_.bdis.size()) return false;
    res_bi.bdi = imap_.bdis[bd_idx];
    res_bi.rdi = imap_.rdis[rd_idx];

    return true;
}

bool jit_brgemm_amx_uker_base_t::bi_shift_B(
        brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi) {
    res_bi = bi;
    size_t lidx = bi.ldi.idx * imap_.rdis.size() + bi.rdi.idx;
    lidx += shift;
    if (lidx >= imap_.rdis.size() * imap_.ldis.size()) return false;

    size_t ld_idx = lidx / imap_.rdis.size();
    size_t rd_idx = lidx % imap_.rdis.size();

    if (lidx >= imap_.rdis.size() * imap_.ldis.size()) return false;
    res_bi.ldi = imap_.ldis[ld_idx];
    res_bi.rdi = imap_.rdis[rd_idx];

    return true;
}

int jit_brgemm_amx_uker_base_t::get_C_tensor(
        brgemm_iteration_t &bi, int m, int n) const noexcept {
    return brg.get_C_tensor(m, n);
}

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

size_t jit_brgemm_amx_uker_base_t::C_block_offset(int bd, int ldb) const
        noexcept {
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

size_t jit_brgemm_amx_uker_base_t::rdb_A_offset(
        const brgemm_iteration_t &bi) const noexcept {
    return bi.rdi.pos * brg.typesize_A * brg.rd_block;
}

size_t jit_brgemm_amx_uker_base_t::rdb_B_offset(
        const brgemm_iteration_t &bi) const noexcept {
    return bi.rdi.pos * brg.rd_block * LDB_size_;
}

size_t jit_brgemm_amx_uker_base_t::ldb_B_offset(
        const brgemm_iteration_t &bi) const noexcept {
    return bi.ldi.pos * ld_block_B_size_ * brg.ld_step;
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
        brgemm_iteration_t &bi, int ldb) const noexcept {
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        return (bi.ldi.is_tail) ? ldb_tail_zp_size_
                                : (bi.ldi.pos + ldb) * ld_block_zp_size_;
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
        case data_type::f16: vcvtph2ps(zmm, op); break;
        case data_type::s8: vpmovsxbd(zmm, op); break;
        case data_type::u8: vpmovzxbd(zmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (types::is_integral_dt(type_in)) vcvtdq2ps(zmm_in, zmm_in);
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

void jit_brgemm_amx_uker_base_t::load_accumulators(brgemm_iteration_t &bi) {
    assert(IMPLICATION(bi.ldi.is_tail, bi.ldi.block2 == 1));
    if (may_load_accumulators_) mov(reg_stride_ld_block, LDC_size_);

    for (int bdb = 0; bdb < bi.bdi.block2; bdb++) {
        const auto bd_out_bdb = get_out_bd(bi.bdi.bdb_pos[bdb], 0);
        for (int ldb = 0; ldb < bi.ldi.block2; ldb++) {
            if (may_load_accumulators_) {
                const auto c_offset
                        = C_block_offset(bd_out_bdb, bi.ldi.pos + ldb);
                tileloadd(Tmm(get_C_tensor(bi, bdb, ldb)),
                        ptr[reg_C + c_offset + reg_stride_ld_block]);
            } else {
                tilezero(Tmm(get_C_tensor(bi, bdb, ldb)));
            }
        }
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

void jit_brgemm_amx_uker_base_t::apply_post_ops_to_range(brgemm_iteration_t &bi,
        int bd_start, int bd_finish, int bd_inp_bdb, int ldb) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

    auto ldb_pos = bi.ldi.pos + ldb;
    auto is_ld_tail = bi.ldi.is_tail;

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

                const auto d_offset = D_offset(bd_out_bd, ldb_pos);
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
                const auto d_offset = D_offset(bd_out_bd, ldb_pos);
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
        brgemm_iteration_t &bi, int ldb) {
    if (!bi.apply_postops) return;
    auto k_mask = (!bi.ldi.is_tail) ? ld_full_mask : ld_tail_mask;

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(reg_aux_zp_comp_a, ptr[rsp + reg_zp_comp_a_offs_]);

        int zp_comp_a_off = zp_comp_a_offset(bi.ldi.pos + ldb);
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
            int zp_c_off = zp_c_values_offset(bi, ldb);
            auto zp_c_addr = EVEX_compress_addr(reg_zp_c_values, zp_c_off);
            cvt2ps(data_type::s32, zmm_zp_c, zp_c_addr, true, false, k_mask);
        }
    }
}

void jit_brgemm_amx_uker_base_t::prepare_post_ops_registers(
        brgemm_iteration_t &bi) {
    if (!bi.apply_postops) return;
    dim_iteration_t &ldi = bi.ldi;
    auto k_mask = (!ldi.is_tail) ? ld_full_mask : ld_tail_mask;
    if (brg.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);

        for (int ldb = 0; ldb < ldi.block2; ldb++) {
            auto ptr_bias
                    = EVEX_compress_addr(reg_bias, bias_offset(ldi.pos + ldb));
            cvt2ps(brg.dt_bias, zmm_bias(ldb), ptr_bias, true, false, k_mask);
        }
    }

    if (brg.with_scales) {
        mov(reg_scales, ptr[param1 + GET_OFF(ptr_scales)]);
        for (int ldb = 0; ldb < ldi.block2; ldb++) {
            auto scales_ptr = EVEX_compress_addr(
                    reg_scales, scales_offset(ldi.pos + ldb));
            vmovups(zmm_scales(ldb) | k_mask | T_z, scales_ptr);
        }
    }
}

void jit_brgemm_amx_uker_base_t::prefetch_output_range(brgemm_iteration_t &bi,
        brgemm_kernel_prefetching_t pft, int bd_start, int bd_finish,
        int bd_inp_bdb, int ldb) {
    auto ldb_pos = bi.ldi.pos + ldb;
    for (int bd = bd_start; bd < bd_finish; bd++) {
        const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
        if (bd_out_bd == -1) continue;
        if (bi.apply_postops) {
            const auto d_offset = D_offset(bd_out_bd, ldb_pos);
            auto ptr_D = EVEX_compress_addr(reg_D, d_offset);
            prefetcht1(ptr_D);
        } else if (are_post_ops_applicable_) {
            const auto c_offset = C_offset(bd_out_bd, ldb_pos);
            auto ptr_C = EVEX_compress_addr(reg_C, c_offset);
            prefetcht1(ptr_C);
        } else {
            const auto d_offset = D_offset(bd_out_bd, ldb_pos);
            auto ptr_D = EVEX_compress_addr(reg_D, d_offset);
            prefetcht1(ptr_D);
        }
    }
}

void jit_brgemm_amx_uker_base_t::process_output_range(brgemm_iteration_t &bi,
        int bd_start, int bd_finish, int bd_inp_bdb, int bdb, int ldb) {

    const int wsp_offset = use_ils_
            ? (bdb * ils_bi_.ldi.block2 + ldb) * brg.bd_block * ld_block_C_size_
            : 0;

    auto k_mask = (!bi.ldi.is_tail) ? ld_full_mask : ld_tail_mask;

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

        auto vreg_acc
                = bi.ldi.is_tail ? accm(bd) | ld_tail_mask | T_z : accm(bd);
        some_bd_mask = true;

        size_t buf_offset = bd * ld_block_C_size_;
        vmovups(vreg_acc, ptr[reg_buf + buf_offset + wsp_offset]);

        const auto c_offset = C_offset(bd_out_bd, bi.ldi.pos + ldb);
        auto ptr_C = EVEX_compress_addr(reg_C, c_offset);

        if (need_to_apply_alpha_beta_)
            apply_alpha_beta_to_vector(zmm.getIdx(), ptr_C, bi.ldi.is_tail);

        if (!bi.apply_postops) continue;

        if (dq2ps_required) vcvtdq2ps(zmm, zmm);
    }

    if (!bi.apply_postops || !some_bd_mask) return;

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
        apply_post_ops_to_range(bi, bd_start, bd_finish, bd_inp_bdb, ldb);
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
        case data_type::f16:
            vcvtps2ph(ymm, zmm, _op_mxcsr);
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

void jit_brgemm_amx_uker_base_t::store_vector(
        brgemm_iteration_t &bi, const int idx, const int bd, const int ldb) {
    auto ldb_pos = bi.ldi.pos + ldb;
    auto is_ld_tail = bi.ldi.is_tail;
    const auto c_offset = C_offset(bd, ldb_pos);
    const auto d_offset = D_offset(bd, ldb_pos);

    auto ptr_C = EVEX_compress_addr(reg_C, c_offset);
    auto ptr_D = EVEX_compress_addr(reg_D, d_offset);

    if (bi.apply_postops)
        store_vector_with_post_ops(idx, ptr_D, bd, ldb_pos, is_ld_tail);
    else if (are_post_ops_applicable_)
        store_vector_without_post_ops(idx, ptr_C, is_ld_tail);
    else
        store_vector_without_post_ops(idx, ptr_D, is_ld_tail);
}

void jit_brgemm_amx_uker_base_t::prefetch_output_data(brgemm_iteration_t &bi,
        bool first_bs, bool last_bs, bool prefetch_all) {

    if (brg.brgattr.hint_prefetching
            != brgemm_kernel_prefetching_t::brgemm_prf_output1)
        return;

    // for var_bs we do prefetch on first iteration by bs only
    if (brg.brgattr.var_bs && !last_bs) return;

    brgemm_iteration_t pfo_bi = ils_bi_;

    // last bd_block may be bd_tail
    const auto bdb_row = brg.bd_block * pfo_bi.ldi.block2;
    const auto tot_vecs = (pfo_bi.bdi.block2 - 1) * bdb_row
            + pfo_bi.bdi.block * pfo_bi.ldi.block2;

    const auto nvecs = prefetch_all
            ? tot_vecs
            : nstl::min(pfo_vecs_per_store_, tot_vecs - pfo_vec_);

    int out_typesize = (are_post_ops_applicable_ && !ils_bi_.apply_postops)
            ? brg.typesize_C
            : brg.typesize_D;
    for (int vec = 0; vec < nvecs && pfo_vec_ < tot_vecs; vec++) {
        int bdb = pfo_vec_ / bdb_row;
        auto vec_in_bdb_row = pfo_vec_ - bdb * bdb_row;
        int ldb = vec_in_bdb_row / pfo_bi.bdi.block;
        // prefetch output cache lines only once
        if ((pfo_bi.ldi.pos + ldb) % (4 / out_typesize) == 0) {
            int bd = vec_in_bdb_row % pfo_bi.bdi.block;
            auto bd_inp_bdb
                    = get_bd_inp_bdb(skipped_bd_mask(pfo_bi.bdi.pos), bdb);
            prefetch_output_range(pfo_bi,
                    brgemm_kernel_prefetching_t::brgemm_prf_output1, 0, 1,
                    bd_inp_bdb + bd, ldb);
        }
        pfo_vec_++;
    }
}

void jit_brgemm_amx_uker_base_t::interleave_store(
        brgemm_iteration_t &bi, bool store_all) {

    if (!use_ils_) return;
    if (!ils_buffer_ready_) return;
    if (!get_store_by_vectors(bi.apply_postops)) return;

    if (store_all) prefetch_output_data(ils_bi_, true, true, true);

    int bd_inp_bdb = ils_bi_.bdi.pos;

    auto cur_bdb = ils_bdb_;
    auto cur_ldb = ils_ldb_;

    ils_bd_step_ = 3; // heuristic value

    // if first block
    if (ils_vec_ == 0) {
        if (!prepare_post_ops_registers_once_) {
            prepare_post_ops_registers(ils_bi_);
        }
        prepare_post_ops_registers_ldb(ils_bi_, 0);
        ils_bd_start_ = 0;
        auto bd_finish = nstl::min(ils_bd_step_, ils_bi_.bdi.block);
        process_output_range(
                ils_bi_, 0, bd_finish, bd_inp_bdb, cur_bdb, cur_ldb);
    }

    // last bd_block may be bd_tail
    const auto bdb_row = brg.bd_block * ils_bi_.ldi.block2;
    const auto total_vectors = (ils_bi_.bdi.block2 - 1) * bdb_row
            + ils_bi_.bdi.block * ils_bi_.ldi.block2;
    const auto nvecs = store_all ? total_vectors : ils_vecs_per_store_;
    for (int vec = 0; vec < nvecs && ils_vec_ < total_vectors; vec++) {
        int bdb = ils_vec_ / bdb_row;
        auto vec_in_bdb_row = ils_vec_ - bdb * bdb_row;
        int ldb = vec_in_bdb_row / ils_bi_.bdi.block;
        int bd = vec_in_bdb_row % ils_bi_.bdi.block;

        auto bd_inp_bdb = ils_bi_.bdi.bdb_pos[bdb];
        if (ldb != cur_ldb) prepare_post_ops_registers_ldb(ils_bi_, ldb);

        if (bdb != cur_bdb || ldb != cur_ldb
                || rnd_dn(bd, ils_bd_step_) != ils_bd_start_) {
            ils_bd_start_ = rnd_dn(bd, ils_bd_step_);
            auto bd_finish = nstl::min(
                    ils_bd_start_ + ils_bd_step_, ils_bi_.bdi.block);
            process_output_range(
                    ils_bi_, ils_bd_start_, bd_finish, bd_inp_bdb, bdb, ldb);
        }

        const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
        if (bd_out_bd != -1) {
            auto vreg_acc = ils_bi_.ldi.is_tail ? accm(bd) | ld_tail_mask | T_z
                                                : accm(bd);

            store_vector(ils_bi_, vreg_acc.getIdx(), bd_out_bd, ldb);
        }
        cur_bdb = bdb;
        cur_ldb = ldb;
        ils_vec_++;
    }
    ils_ldb_ = cur_ldb;
    ils_bdb_ = cur_bdb;
}

void jit_brgemm_amx_uker_base_t::store_accumulators(brgemm_iteration_t &bi) {

    const auto store_by_vectors = get_store_by_vectors(bi.apply_postops);

    if (store_by_vectors)
        mov(reg_stride_ld_block, ld_block_C_size_);
    else
        mov(reg_stride_ld_block, LDC_size_);

    ils_bi_ = bi;

    ils_vec_ = 0;
    ils_bdb_ = 0;
    ils_ldb_ = 0;
    ils_buffer_ready_ = true;
    ils_store_ops_ = bi.ldi.block2 * bi.bdi.block2 * brg.bd_block;
    pfo_vec_ = 0;

    if (store_by_vectors && !use_ils_ && !prepare_post_ops_registers_once_)
        prepare_post_ops_registers(bi);

    for (int bdb = 0; bdb < bi.bdi.block2; bdb++) {
        const auto bd_inp_bdb = bi.bdi.bdb_pos[bdb];

        for (int ldb = 0; ldb < bi.ldi.block2; ldb++) {
            if (store_by_vectors) {
                const int wsp_offset = use_ils_ ? (bdb * bi.ldi.block2 + ldb)
                                * brg.bd_block * ld_block_C_size_
                                                : 0;
                tilestored(ptr[reg_buf + reg_stride_ld_block + wsp_offset],
                        Tmm(get_C_tensor(bi, bdb, ldb)));
                if (use_ils_) continue;

                prepare_post_ops_registers_ldb(bi, ldb);

                process_output_range(bi, 0, bi.bdi.block, bd_inp_bdb, bdb, ldb);

                for (int bd = 0; bd < bi.bdi.block; bd++) {
                    const auto bd_out_bd = get_out_bd(bd_inp_bdb, bd);
                    if (bd_out_bd == -1) continue;

                    auto vreg_acc = bi.ldi.is_tail
                            ? accm(bd) | ld_tail_mask | T_z
                            : accm(bd);
                    store_vector(bi, vreg_acc.getIdx(), bd_out_bd, ldb);
                }
            } else {
                const auto bd_out_bdb = get_out_bd(bd_inp_bdb, 0);
                const auto c_offset
                        = C_block_offset(bd_out_bdb, bi.ldi.pos + ldb);
                tilestored(ptr[reg_C + reg_stride_ld_block + c_offset],
                        Tmm(get_C_tensor(bi, bdb, ldb)));
            }
        }
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

void jit_brgemm_amx_uker_base_t::maybe_tileloadd_nt(
        brgemm_iteration_t &bi, matrix_kind_t mk, int xdb, size_t offset) {
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
        maybe_pre_process_data(bi, t1, reg_base, offset, reg_stride, mk);
    else if (try_load_nt)
        tileloaddt1(t1, ptr[reg_base + offset + reg_stride]);
    else
        tileloadd(t1, ptr[reg_base + offset + reg_stride]);
}

void jit_brgemm_amx_uker_base_t::tdpbxxd(
        brgemm_iteration_t &bi, int bdb_idx, int ldb_idx) {
    prefetch_output_data(bi, bi.bsi.is_first, bi.bsi.is_last, false);

    const Tmm &x1 = Tmm(get_C_tensor(bi, bdb_idx, ldb_idx));
    const Tmm &x2 = Tmm(brg.get_A_tensor(bdb_idx));
    const Tmm &x3 = Tmm(brg.get_B_tensor(ldb_idx));
    if (brg.is_bf32
            || (brg.dt_a == data_type::bf16 && brg.dt_b == data_type::bf16)) {
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
    interleave_store(bi, false);
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

void jit_brgemm_amx_uker_base_t::maybe_pre_process_data(brgemm_iteration_t &bi,
        const Tmm &t1, reg64_t reg_base, size_t offset, reg64_t reg_stride,
        matrix_kind_t mk) {

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
    const std::string key
            = std::to_string(bi.bsi.pos) + "_" + std::to_string(offset);

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
                reg_buf, bi.rdi.is_tail);
    } else {
        bf32_downconvert_to_vnni(num_rows, num_col_bytes, reg_base, offset,
                reg_stride, reg_buf, bi.rdi.is_tail);
    }

    // load into tmm from the transformed data.
    tileloadd(t1, ptr[reg_buf + reg_bf32_stride]);

    // reset buf pointer.
    if (buf_offt) sub(reg_buf, buf_offt);
}

void jit_brgemm_amx_uker_base_t::gemm_microkernel_amx(brgemm_iteration_t &bi) {
    const auto store_by_vectors = get_store_by_vectors(bi.apply_postops);

    if (store_by_vectors)
        mov(reg_stride_ld_block, ld_block_C_size_);
    else
        mov(reg_stride_ld_block, LDC_size_);

    const auto rdb_A_off = rdb_A_offset(bi);
    const auto rdb_B_off = rdb_B_offset(bi) + ldb_B_offset(bi);

    for (int bdb = 0; bdb < bi.bdi.block2; bdb++) {
        const auto bd_inp_bdb = bi.bdi.bdb_pos[bdb];
        maybe_tileloadd_nt(bi, matrix_kind_t::matrix_A, bdb,
                rdb_A_off + A_offset(bd_inp_bdb));
        for (int ldb = 0; ldb < bi.ldi.block2; ldb++) {
            if (bdb == 0)
                maybe_tileloadd_nt(bi, matrix_kind_t::matrix_B, ldb,
                        rdb_B_off + B_offset(ldb));
            if (ldb == 0) {
                if (bdb > 0) tdpbxxd(bi, bdb - 1, bi.ldi.block2 - 1);
            } else
                tdpbxxd(bi, bdb, ldb - 1);
        }
    }
    // last tdpbxxd
    tdpbxxd(bi, bi.bdi.block2 - 1, bi.ldi.block2 - 1);
}

void jit_brgemm_amx_uker_base_t::rdb_loop(brgemm_iteration_t &bi) {
    for (size_t irdi = 0; irdi < imap_.rdis.size(); irdi++) {
        bi.rdi = imap_.rdis[irdi];
        gemm_microkernel_amx(bi);
    }
}

void jit_brgemm_amx_uker_base_t::bs_loop_body(brgemm_iteration_t &bi) {
    if (brg.brgattr.var_bs) {
        set_A_B_matrices();
        add(reg_aux1_batch, sizeof(brgemm_batch_element_t));
        prefetcht0(ptr[reg_aux1_batch]);
    } else {
        set_A_B_matrices(bi.bsi.pos);
    }

    rdb_loop(bi);
}

void jit_brgemm_amx_uker_base_t::bs_loop(brgemm_iteration_t &bi) {
    int calc_ops = brg.brgattr.max_bs * (brg.rdb + (brg.rdb_tail ? 1 : 0))
            * bi.ldi.block2 * bi.bdi.block2;
    ils_vecs_per_store_ = (calc_ops) ? div_up(ils_store_ops_, calc_ops) : 0;

    int store_ops = bi.ldi.block2 * bi.bdi.block2 * brg.bd_block;
    int calc_ops_for_prefteching
            = brg.brgattr.var_bs ? calc_ops / brg.brgattr.max_bs : calc_ops;
    pfo_vecs_per_store_ = (calc_ops_for_prefteching)
            ? div_up(store_ops, calc_ops_for_prefteching)
            : 0;

    load_accumulators(bi);

    if (brg.brgattr.var_bs) {
        if (brg.alpha != 0.f) {
            Label BS_loop_label, end_BS_loop_label, first_BS_loop_label,
                    last_BS_loop_label;

            mov(reg_BS_loop, reg_BS);
            cmp(reg_BS_loop, 0);
            jz(end_BS_loop_label, T_NEAR);

            mov(reg_aux1_batch, reg_addr_batch);
            // first bs iteration
            cmp(reg_BS_loop, 1);
            jg(first_BS_loop_label, T_NEAR);

            bi.bsi = imap_.bsis[0];
            // only one BS iteration: first and last
            bi.bsi.is_first = true;
            bi.bsi.is_last = true;
            bs_loop_body(bi);
            jmp(end_BS_loop_label, T_NEAR);

            // first BS iteration
            L_aligned(first_BS_loop_label, 64);
            bi.bsi.is_first = true;
            bi.bsi.is_last = false;
            bs_loop_body(bi);

            dec(reg_BS_loop);
            cmp(reg_BS_loop, 1);
            je(last_BS_loop_label, T_NEAR);

            // middle BS iterations
            L_aligned(BS_loop_label, 64);
            {
                bi.bsi.is_first = false;
                bi.bsi.is_last = false;
                bs_loop_body(bi);
                dec(reg_BS_loop);
                cmp(reg_BS_loop, 1);
                jg(BS_loop_label, T_NEAR);
            }
            // last BS iteration
            L_aligned(last_BS_loop_label, 64);
            bi.bsi.is_first = false;
            bi.bsi.is_last = true;
            bs_loop_body(bi);

            L_aligned(end_BS_loop_label, 64);
        }
        store_accumulators(bi);
    } else {
        if (brg.alpha != 0.f) {
            for (int bs = 0; bs < brg.brgattr.max_bs; bs++) {
                bi.bsi = imap_.bsis[bs];
                bs_loop_body(bi);
            }
        }
        store_accumulators(bi);
    }
}

void jit_brgemm_amx_uker_base_t::ldb_loop_body(brgemm_iteration_t &bi) {
    if (brg.brgattr.hint_innermost_loop == brgemm_bd_loop_innermost)
        bdb_loop(bi);
    else if (brg.brgattr.hint_innermost_loop == brgemm_ld_loop_innermost)
        bs_loop(bi);
    else
        assert(!"Unknown loop order!");
}

void jit_brgemm_amx_uker_base_t::ldb_loop(brgemm_iteration_t &bi) {
    // clear the transform cache for A, as the existing data is invalid as
    // we move to next bdb2 block.
    transform_buf_map_A_.clear();
    bi.ldi = imap_.ldis[0];
    for (size_t ildi = 0; ildi < imap_.ldis.size(); ildi++) {
        bi.ldi = imap_.ldis[ildi];
        ldb_loop_body(bi);
    }
}

void jit_brgemm_amx_uker_base_t::bdb_loop_body(brgemm_iteration_t &bi) {

    if (brg.brgattr.hint_innermost_loop == brgemm_ld_loop_innermost)
        ldb_loop(bi);
    else
        bs_loop(bi);
};

void jit_brgemm_amx_uker_base_t::bdb_loop(brgemm_iteration_t &bi) {
    bi.bdi = imap_.bdis[0];

    for (size_t ibdi = 0; ibdi < imap_.bdis.size(); ibdi++) {
        bi.bdi = imap_.bdis[ibdi];
        bdb_loop_body(bi);
    }
}

void jit_brgemm_amx_uker_base_t::top_loop(brgemm_iteration_t &bi) {
    init(bi);
    if (brg.brgattr.hint_innermost_loop == brgemm_ld_loop_innermost)
        bdb_loop(bi);
    else
        ldb_loop(bi);

    interleave_store(bi, true);
}

void jit_brgemm_amx_uker_base_t::fill_imap() {
    imap_.bdis.clear();
    imap_.ldis.clear();
    imap_.rdis.clear();
    imap_.bsis.clear();
    size_t bdi_pos = skipped_bd_mask(0);
    auto bdi = bd_iteration_t(bdi_pos, brg.bd_block, brg.bd_block2);
    for (int bdb2 = 0; bdb2 < brg.bdb2; bdb2++) {
        bdi.pos = bdi_pos;
        bdi.bdb_pos.clear();
        for (int bdb = 0; bdb < bdi.block2; bdb++) {
            bdi.bdb_pos.push_back(bdi_pos);
            bdi_pos += brg.bd_block;
            bdi_pos = skipped_bd_mask(bdi_pos);
        }
        bdi.idx = imap_.bdis.size();
        imap_.bdis.push_back(bdi);
    }
    if (brg.bdb2_tail > 0) {
        bdi.block2 = brg.bdb2_tail;
        bdi.pos = bdi_pos;
        bdi.bdb_pos.clear();
        for (int bdb = 0; bdb < bdi.block2; bdb++) {
            bdi.bdb_pos.push_back(bdi_pos);
            bdi_pos += brg.bd_block;
            bdi_pos = skipped_bd_mask(bdi_pos);
        }
        bdi.idx = imap_.bdis.size();
        imap_.bdis.push_back(bdi);
    }
    if (brg.bdb_tail > 0) {
        bdi.block2 = 1;
        bdi.block = brg.bdb_tail;
        bdi.is_tail = true;
        bdi.pos = bdi_pos;
        bdi.bdb_pos.clear();
        bdi.bdb_pos.push_back(bdi_pos);
        bdi.idx = imap_.bdis.size();
        imap_.bdis.push_back(bdi);
    }

    auto ldi = dim_iteration_t(0, brg.ld_block, brg.ld_block2);
    for (int ldb2 = 0; ldb2 < brg.ldb2; ldb2++) {
        ldi.idx = imap_.ldis.size();
        imap_.ldis.push_back(ldi);
        ldi.pos += ldi.block2;
    }
    if (brg.ldb2_tail > 0) {
        ldi.block2 = brg.ldb2_tail;
        ldi.idx = imap_.ldis.size();
        imap_.ldis.push_back(ldi);
        ldi.pos += ldi.block2;
    }
    if (brg.ldb_tail > 0) {
        ldi.block2 = 1;
        ldi.block = brg.ldb_tail;
        ldi.is_tail = true;
        ldi.idx = imap_.ldis.size();
        imap_.ldis.push_back(ldi);
    }

    auto rdi = dim_iteration_t(0, brg.rd_block, 1);
    for (int rdb = 0; rdb < brg.rdb; rdb++) {
        rdi.idx = imap_.rdis.size();
        imap_.rdis.push_back(rdi);
        rdi.pos++;
    }
    if (brg.rdb_tail > 0) {
        rdi.block = brg.rdb_tail;
        rdi.is_tail = true;
        rdi.idx = imap_.rdis.size();
        imap_.rdis.push_back(rdi);
    }

    bs_iteration_t bsi;
    for (int bs = 0; bs < brg.brgattr.max_bs; bs++) {
        bsi.pos = bs;
        bsi.is_first = (bs == 0);
        bsi.is_last = (bs == brg.brgattr.max_bs - 1);
        bsi.idx = imap_.bsis.size();
        imap_.bsis.push_back(bsi);
    }
}

void jit_brgemm_amx_uker_base_t::init(brgemm_iteration_t &bi) {
    ils_buffer_ready_ = false;

    if (brg.brgattr.max_bs == 1) {
        if (brg.layout == brgemm_row_major) {
            mov(reg_aux_A,
                    EVEX_compress_addr(
                            reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
            mov(reg_aux_B,
                    EVEX_compress_addr(
                            reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
        } else {
            mov(reg_aux_A,
                    EVEX_compress_addr(
                            reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
            mov(reg_aux_B,
                    EVEX_compress_addr(
                            reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
        }
    }

    // for many primitives which use brgemm the brg.ldb2 is equal or less than 1
    // so we can read post ops data only once per brgemm call

    if (brg.ldb2 > 1) {
        prepare_post_ops_registers_once_ = false;
    } else if (brg.ldb2 == 1) {
        if (brg.ldb2_tail == 0 && brg.ldb_tail == 0) {
            prepare_post_ops_registers_once_ = true;
            bi.ldi = dim_iteration_t(0, brg.ld_block, brg.ld_block2);
            prepare_post_ops_registers(bi);
        }
    } else if (brg.ldb2_tail > 0) {
        if (brg.ldb_tail == 0) {
            prepare_post_ops_registers_once_ = true;
            bi.ldi = dim_iteration_t(0, brg.ld_block, brg.ldb2_tail);
            prepare_post_ops_registers(bi);
        }
    } else {
        prepare_post_ops_registers_once_ = true;
        bi.ldi = dim_iteration_t(0, brg.ldb_tail, 1);
        bi.ldi.is_tail = true;
        prepare_post_ops_registers(bi);
    }
    if (bi.apply_postops)
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

    fill_imap();
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
    may_load_accumulators_ = (brg.beta == 1.f
            && (((brg.is_f32 || brg.is_bf16) && brg.dt_c == data_type::f32)
                    || (brg.is_int8 && brg.dt_c == data_type::s32)));
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
    assert(IMPLICATION(brg.brgattr.var_bs, !brg.is_bf32));
    read_params();
    prepare_bd_mask();
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

    brgemm_iteration_t bi;

    Label label_to_ret;
    if (are_post_ops_applicable_) {
        Label label_store_without_post_ops;
        mov(reg_do_post_ops, ptr[param1 + GET_OFF(do_post_ops)]);
        cmp(reg_do_post_ops, 0);
        jz(label_store_without_post_ops, T_NEAR);
        bi.apply_postops = true;
        top_loop(bi);
        transform_buf_map_A_.clear();
        transform_buf_map_B_.clear();
        L(label_store_without_post_ops);
    }
    bi.apply_postops = false;
    top_loop(bi);
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
