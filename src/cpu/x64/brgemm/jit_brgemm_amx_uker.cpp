/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

struct jit_brgemm_amx_uker_base_t : public jit_base_brgemm_kernel_t {
    jit_brgemm_amx_uker_base_t(const brgemm_desc_t &abrg)
        : jit_base_brgemm_kernel_t(jit_name(), abrg.isa_impl)
        , brg(abrg)
        , postops_injector_(nullptr) {

        bool has_f8_e5m2_binary_postops = false;
        bool has_f8_e4m3_binary_postops = false;
        if (brg.with_binary) {
            const auto &post_ops = brg.attr()->post_ops_;
            for (int i = 0; i < post_ops.len(); i++) {
                const auto &entry = post_ops.entry_[i];
                if (!entry.is_binary()) continue;
                has_f8_e5m2_binary_postops
                        = entry.binary.src1_desc.data_type == data_type::f8_e5m2
                        || has_f8_e5m2_binary_postops;
                has_f8_e4m3_binary_postops
                        = entry.binary.src1_desc.data_type == data_type::f8_e4m3
                        || has_f8_e4m3_binary_postops;
            }
        }

        if (brg.is_fp8_via_convert() || has_f8_e5m2_binary_postops
                || has_f8_e4m3_binary_postops) {
            if (one_of(data_type::f8_e5m2, brg.dt_a, brg.dt_b, brg.dt_d)
                    || has_f8_e5m2_binary_postops)
                f8_e5m2_emulator_ = utils::make_unique<fp8_emulation_e5m2_t>(
                        this, fp8_emu_xmm_1(), fp8_emu_xmm_2(), fp8_emu_xmm_3(),
                        fp8_tmp_mask, fp8_tmp_reg);
            if (one_of(data_type::f8_e4m3, brg.dt_a, brg.dt_b, brg.dt_d)
                    || has_f8_e4m3_binary_postops)
                f8_e4m3_emulator_ = utils::make_unique<fp8_emulation_e4m3_t>(
                        this, fp8_emu_xmm_1(), fp8_emu_xmm_2(), fp8_emu_xmm_3(),
                        fp8_emu_xmm_4(), fp8_emu_xmm_5(), fp8_tmp_reg);
        }

        if (brg.with_eltwise || brg.with_binary || brg.with_sum) {

            static constexpr bool preserve_gpr = true;
            // we don't use zmm1 for storing vectors
            // so we don't need to preserve vmm
            static constexpr bool preserve_vmm = false;
            static constexpr bool use_exact_tail_scalar_bcast = false;
            const auto dst_md_wrapper = memory_desc_wrapper(brg.dst_md());

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    static_cast<size_t>(Xbyak::Zmm(1).getIdx()), this->r14,
                    this->r15, this->r13, preserve_gpr, preserve_vmm,
                    GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(data_C_ptr_),
                    dst_md_wrapper, static_cast<size_t>(brg.ldb_tail),
                    ld_tail_mask, use_exact_tail_scalar_bcast};

            const binary_injector::static_params_t bsp(this->param1,
                    binary_injector::get_all_strategies_supported_by_injector(),
                    rhs_sp, f8_e5m2_emulator_.get(), f8_e4m3_emulator_.get());

            eltwise_injector::static_params_t esp;
            esp.preserve_vmm = preserve_vmm;
            esp.preserve_p_table = false;

            auto st = safe_ptr_assign(postops_injector_,
                    po_injector_t::create(this, brg.isa_impl,
                            brg.attr()->post_ops_, bsp, esp));
            if (st != status::success) {
                assert(!"postops_injector creation failed");
            }

            using namespace dnnl::impl::cpu::binary_injector_utils;
            std::tie(with_binary_per_oc_bcast_, with_binary_per_oc_sp_bcast_,
                    with_binary_per_mb_bcast_, with_binary_channel_bcast_,
                    with_binary_per_mb_w_bcast_, with_binary_per_w_bcast_,
                    with_binary_batch_bcast_, with_binary_spatial_bcast_,
                    with_binary_no_bcast_)
                    = bcast_strategies_present_tup(brg.attr()->post_ops_.entry_,
                            dst_md_wrapper, broadcasting_strategy_t::per_oc,
                            broadcasting_strategy_t::per_oc_spatial,
                            broadcasting_strategy_t::per_mb,
                            broadcasting_strategy_t::per_mb_spatial,
                            broadcasting_strategy_t::per_mb_w,
                            broadcasting_strategy_t::per_w,
                            broadcasting_strategy_t::batch,
                            broadcasting_strategy_t::spatial,
                            broadcasting_strategy_t::no_broadcast);
            handle_binary_po_offset_ = with_binary_per_oc_bcast_
                    || with_binary_per_oc_sp_bcast_ || with_binary_per_mb_bcast_
                    || with_binary_channel_bcast_ || with_binary_per_mb_w_bcast_
                    || with_binary_per_w_bcast_ || with_binary_batch_bcast_
                    || with_binary_spatial_bcast_ || with_binary_no_bcast_;
        }
        use_ils_ = brg.brgattr.use_interleave_stores;
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_amx_uker_base_t)

    brgemm_desc_t brg;

    const brgemm_desc_t &get_brg() const override { return brg; }

private:
    using po_injector_t = injector::jit_uni_postops_injector_base_t<Zmm>;
    std::unique_ptr<po_injector_t> postops_injector_;

    std::unique_ptr<fp8_emulation_e5m2_t> f8_e5m2_emulator_;
    std::unique_ptr<fp8_emulation_e4m3_t> f8_e4m3_emulator_;

    using reg64_t = const Xbyak::Reg64;
    enum {
        simd_w = 16,
        zmm_width_in_bytes = cpu_isa_traits<avx512_core>::vlen,
    };

    // Register decomposition
    const reg64_t param1 = abi_param1;

    const reg64_t reg_iter_label = r9;
    const reg64_t reg_iter_labels_list = rax;

    const reg64_t reg_addr_batch = r13;
    const reg64_t reg_aux1_batch = rbp;
    const reg64_t reg_A = r11;
    const reg64_t reg_B = r10;
    const reg64_t reg_stride_lda = r14;
    const reg64_t reg_stride_ldb = abi_not_param1;
    const reg64_t reg_C = r15;
    const reg64_t reg_D = r12;

    const reg64_t reg_buf = r8;
    const reg64_t reg_BS = rbx;
    const reg64_t reg_BS_loop = r9;
    const reg64_t reg_bias = rbx;
    const reg64_t reg_scales = rbx;
    const reg64_t reg_dst_scales = rbx;

    const reg64_t reg_stride_ld_block = rdx;
    const reg64_t reg_do_post_ops = rbx;
    const reg64_t reg_do_skip_accum = reg_do_post_ops;
    const reg64_t reg_tmp_gpr = rbx;
    const reg64_t reg_ptr_sum_scale = rbx;

    const reg64_t reg_zp_comp_a = rbx;
    const reg64_t reg_aux_zp_comp_a = rbx;
    const reg64_t reg_zp_a_values = rbx;
    const reg64_t reg_zp_comp_b = rbx;
    const reg64_t reg_zp_c_values = rbx;
    const reg64_t reg_ptr_sum_zp = rbx;
    const reg64_t reg_converted_stride = rsi;
    const reg64_t reg_zp_comp_pad_a = rsi;

    constexpr static int abi_param1_offs_ = 0;
    constexpr static int reg_zp_comp_a_offs_ = 8;
    constexpr static int reg_zp_comp_b_offs_ = 16;
    constexpr static int reg_zp_c_values_offs_ = 24;
    constexpr static int reg_iter_labels_list_offs_ = 32;
    constexpr static int reg_zp_a_values_offs_ = 40;
    constexpr static int stack_space_needed_ = 48;

    bool are_post_ops_applicable_ = false;
    bool need_to_apply_alpha_beta_ = false;
    bool may_load_accumulators_ = false;

    bool handle_binary_po_offset_ = false;
    bool with_binary_per_oc_bcast_ = false;
    bool with_binary_per_oc_sp_bcast_ = false;
    bool with_binary_channel_bcast_ = false;
    bool with_binary_per_mb_bcast_ = false;
    bool with_binary_per_mb_w_bcast_ = false;
    bool with_binary_per_w_bcast_ = false;
    bool with_binary_batch_bcast_ = false;
    bool with_binary_spatial_bcast_ = false;
    bool with_binary_no_bcast_ = false;
    bool prepare_post_ops_registers_once_ = false;

    const char *bd_mask_buffer_ptr_ = nullptr;
    std::vector<size_t> adj_bd_mask_buffer_;
    std::vector<size_t> skipped_bd_mask_buffer_;
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
    // Structures below (iteration_block_t, dim_iteration_t, bs_iteration_t and
    // iteration_map_t) describe the structure of cycles and are used for
    // JIT code generation
    struct iteration_block_t {
        int block = 0;
        size_t pos = 0;
        bool is_tail = false;
        iteration_block_t(size_t pos_, int block_, bool is_tail_ = false)
            : block(block_), pos(pos_), is_tail(is_tail_) {}
        bool operator==(const iteration_block_t &rhs) const {
            return block == rhs.block && is_tail == rhs.is_tail;
        }
    };

    struct dim_iteration_t {
        size_t idx = 0;
        std::vector<iteration_block_t> blocks;
        virtual bool operator==(const dim_iteration_t &rhs) const {
            return blocks == rhs.blocks;
        }
        virtual bool operator!=(const dim_iteration_t &rhs) const {
            return !operator==(rhs);
        }

        size_t pos(size_t b) const {
            assert(b < blocks.size());
            return blocks[b].pos;
        }

        size_t rel_pos(size_t b) const {
            assert(b < blocks.size());
            return (blocks[b].pos - blocks[0].pos);
        }

        int block(size_t b) const {
            assert(b < blocks.size());
            return blocks[b].block;
        }

        bool is_tail(size_t b) const {
            assert(b < blocks.size());
            return blocks[b].is_tail;
        }

        int block2() const { return static_cast<int>(blocks.size()); }

        int length() const {
            if (blocks.empty()) return 0;
            auto n = blocks.size();
            // only last block may be different
            return ((n - 1) * blocks[0].block + blocks[n - 1].block);
        }

        dim_iteration_t() = default;
        virtual ~dim_iteration_t() = default;
    };

    struct bd_iteration_t : public dim_iteration_t {
        size_t A_shift {0};
        size_t C_shift {0};
        size_t D_shift {0};
        size_t zp_comp_pad_a_shift {0};
        std::vector<char> bd_mask;
        std::vector<size_t> adj_bd_mask;
        bd_iteration_t *similar {nullptr};
        Label lstart;

        bool operator==(const dim_iteration_t &_rhs) const override {
            // `downcast` will catch a type mismatch in debug mode.
            // Note: it supports only a pointer type so far.
            const bd_iteration_t &rhs
                    = *utils::downcast<const bd_iteration_t *>(&_rhs);
            bool res = dim_iteration_t::operator==(rhs)
                    && A_shift == rhs.A_shift && C_shift == rhs.C_shift
                    && D_shift == rhs.D_shift && bd_mask == rhs.bd_mask
                    && zp_comp_pad_a_shift == rhs.zp_comp_pad_a_shift;
            return res;
        }
        bool operator!=(const dim_iteration_t &_rhs) const override {
            return !operator==(_rhs);
        }
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

    class iteration_map_t {
    public:
        struct top_loop_t {
            std::vector<dim_iteration_t> ldis;
            std::vector<bd_iteration_t> bdis;
            std::vector<bs_iteration_t> bsis;
            std::vector<dim_iteration_t> rdis;
            int duplicated {0};
            bool is_last_rdi(const dim_iteration_t *rdi) const {
                return (rdi->idx == rdis.size() - 1);
            }
        };

        iteration_map_t() : tloops(2) {}

        inline top_loop_t &operator[](bool bidx) {
            return tloops[static_cast<int>(bidx)];
        }
        inline const top_loop_t &operator[](bool bidx) const {
            return tloops[static_cast<int>(bidx)];
        }

    private:
        std::vector<top_loop_t> tloops;
    };

    struct brgemm_iteration_t {
        const bd_iteration_t *bdi {nullptr};
        const dim_iteration_t *ldi {nullptr};
        const bs_iteration_t *bsi {nullptr};
        const dim_iteration_t *rdi {nullptr};
        bool apply_postops {false};
        bool skip_accumulation {false};
        bool first_bsi {false};
        bool last_bsi {false};
        brgemm_iteration_t() = default;
    };

    struct prf_t {
        brgemm_kernel_prefetching_t pft = brgemm_prf_default;
        int dist = -1;
        int vec = 0;
        void set(brgemm_kernel_prefetching_t pft_, int dist_) {
            pft = pft_;
            dist = dist_;
            vec = 0;
        }
        void reset() { vec = 0; }
    };

    // iteration map
    iteration_map_t imap_;

    // interleave stores
    bool use_ils_ = false;
    bool was_prev_bi_ = false;
    // saved parameters for storing
    brgemm_iteration_t prev_bi_;
    // current storing coordinates
    int ils_vec_ = 0, ils_bdb_ = 0, ils_ldb_ = 0, ils_bd_start_ = 0;
    int ils_bd_step_ = 3; // heuristic value
    prf_t prf0A, prf1A, prf2A, prfntaA, prf0B, prf1B, prf2B, prfntaB, prf0C,
            prf1C;

    bool dt_requires_saturation_ = false;

    bool ununroll_bd_loop = false;

    Xbyak::Opmask ld_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask ld_tail_mask = Xbyak::Opmask(3);
    Xbyak::Opmask fp_col_mask = Xbyak::Opmask(4);
    Xbyak::Opmask rd_tail_mask = Xbyak::Opmask(5);

    // Zmm map below
    const Xbyak::Zmm &zmm_tmp_1() const noexcept { return this->zmm0; }
    const Xbyak::Zmm &zmm_tmp_2() const noexcept { return this->zmm1; }
    const Xbyak::Zmm &zmm_tmp_3() const noexcept { return this->zmm2; }

    /* fp8 emulation */
    Xmm fp8_emu_xmm_1() const noexcept { return Xmm(1); }
    Xmm fp8_emu_xmm_2() const noexcept { return Xmm(2); }
    Xmm fp8_emu_xmm_3() const noexcept { return Xmm(3); }
    Xmm fp8_emu_xmm_4() const noexcept { return Xmm(6); }
    Xmm fp8_emu_xmm_5() const noexcept { return Xmm(7); }
    Xbyak::Opmask fp8_tmp_mask = Xbyak::Opmask(6);
    const reg64_t fp8_tmp_reg = rax;

    const Xbyak::Zmm zmm_bf32_permute = zmm6;
    const Xbyak::Zmm zmm_zp_comp_a = zmm6;
    const Xbyak::Zmm zmm_zp_c = zmm7;
    const Xbyak::Zmm zmm_lbound = zmm8;
    const Xbyak::Zmm zmm_ubound = zmm9;

    // zmm_bias, zmm_bias and accm shouldn't be overlapped
    Xbyak::Zmm accm(int bd) const {
        assert(bd < 16);
        return Xbyak::Zmm(31 - (bd % ils_bd_step_));
    }

    Xbyak::Zmm zmm_bias(int ldb) const {
        assert(ldb < 5);
        // zmm10 - zmm14
        return Xbyak::Zmm(10 + ldb);
    }

    Xbyak::Zmm zmm_scales(int ldb) const {
        assert(ldb < 5);
        assert(ils_bd_step_ < 10);
        // zmm15 - zmm19
        return Xbyak::Zmm(15 + ldb);
    }

    Xbyak::Zmm zmm_mask(const Xbyak::Zmm &zmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) const;
    Xbyak::Ymm ymm_mask(const Xbyak::Ymm &ymm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) const;
    Xbyak::Xmm xmm_mask(const Xbyak::Xmm &xmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) const;

    void cvt2ps(data_type_t type_in, const Xbyak::Zmm &zmm_in,
            const Xbyak::Operand &op, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);

    void read_params();
    void load_accumulators(brgemm_iteration_t &bi);

    void maybe_saturation(Xbyak::Zmm &zmm);
    void apply_alpha_beta_to_vector(
            int idx, const Address &addr, bool is_ld_tail);
    void apply_post_ops_to_range(brgemm_iteration_t &bi, int bd_start,
            int bd_finish, int bdb, int ldb);
    void store_vector_with_post_ops(
            int idx, const Address &addr, bool is_ld_tail);
    void prepare_post_ops_registers_ldb(brgemm_iteration_t &bi, int ldb);
    void prepare_post_ops_registers(brgemm_iteration_t &bi);

    bool bi_shift_output(
            brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi);
    bool bi_shift_A(
            brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi);
    bool bi_shift_B(
            brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi);

    void uni_prefetch(const Address &addr, brgemm_kernel_prefetching_t pft,
            bool for_write);
    void prefetch_CD_range(brgemm_iteration_t &bi,
            brgemm_kernel_prefetching_t pft, int bd_start, int bd_finish,
            int bdb, int ldb);
    int calc_ops_CD(brgemm_iteration_t &bi) const noexcept;
    void prefetch_CD(brgemm_iteration_t &bi, brgemm_iteration_t &pfo_bi,
            prf_t &prf, bool prefetch_all);

    void prefetch_A(brgemm_iteration_t &bi, brgemm_iteration_t &pfo_bi,
            prf_t &prf, bool prefetch_all);
    void prefetch_B(brgemm_iteration_t &bi, brgemm_iteration_t &pfo_bi,
            prf_t &prf, bool prefetch_all);
    void prefetching(brgemm_iteration_t &bi, bool prefetch_all);

    void process_output_range(brgemm_iteration_t &bi, int bd_start,
            int bd_finish, int bdb, int ldb);
    void store_vector_without_post_ops(
            int idx, const Address &addr, bool is_ld_tail);
    void store_vector(brgemm_iteration_t &bi, int bdb, int bd, int ldb);
    void apply_comp_pad_to_vector(brgemm_iteration_t &bi, int bdb, int inp_bd,
            int ldb, const int idx);

    void interleave_store(brgemm_iteration_t &bi, bool store_all);

    void store_accumulators(brgemm_iteration_t &bi);

    void set_A_B_matrices(int bs);
    void set_A_B_matrices();

    void bf32_downconvert(brgemm_iteration_t &bi, int num_rows,
            int tile_num_col_bytes, reg64_t reg_data, int offset,
            reg64_t reg_data_stride, reg64_t reg_buf);
    void fp8_to_f16_upconvert(brgemm_iteration_t &bi, int num_rows,
            int tile_num_col_bytes, reg64_t reg_data, int offset,
            reg64_t reg_data_stride, reg64_t reg_buf, data_type_t dt);

    void fp8_to_f16_upconvert_to_vnni(brgemm_iteration_t &bi, int num_rows,
            int tile_num_col_bytes, reg64_t reg_data, int offset,
            reg64_t reg_data_stride, reg64_t reg_buf, data_type_t dt);

    void bf32_downconvert_to_vnni(brgemm_iteration_t &bi, int num_rows,
            int tile_num_col_bytes, reg64_t reg_data, int offset,
            reg64_t reg_data_stride, reg64_t reg_buf);

    void maybe_pre_process_data(brgemm_iteration_t &bi, const Tmm &t1,
            reg64_t reg_base, size_t offset, reg64_t reg_stride,
            matrix_kind_t mk);

    bool maybe_pre_process_k_tail(brgemm_iteration_t &bi, int bdb,
            const Tmm &t1, reg64_t reg_base, size_t offset, reg64_t reg_stride,
            matrix_kind_t mk);

    void maybe_tileloadd_nt(
            brgemm_iteration_t &bi, matrix_kind_t mk, int xdb, size_t offset);

    void tdpbxxd(brgemm_iteration_t &bi, int bdb_idx, int ldb_idx,
            bool do_pre_tilestore, bool do_post_tilestore);

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
    bool actual_ils(bool apply_post_ops, bool skip_accumulation = false) const {
        return (use_ils_ && get_store_by_vectors(apply_post_ops)
                && !skip_accumulation);
    }

    size_t A_offset(const brgemm_iteration_t &bi, int bdb) const noexcept;
    size_t B_offset(const brgemm_iteration_t &bi, int ldb) const noexcept;
    size_t C_offset(const brgemm_iteration_t &bi, int bdb, int inp_bd,
            int ldb) const noexcept;
    size_t D_offset(const brgemm_iteration_t &bi, int bdb, int inp_bd,
            int ldb) const noexcept;

    size_t lda() const noexcept;
    size_t ldb() const noexcept;

    size_t bias_offset(int ldb) const noexcept;

    size_t scales_offset(int ldb) const noexcept;
    size_t zp_comp_a_offset(int ldb) const noexcept;
    size_t zp_comp_pad_a_offset(const brgemm_iteration_t &bi, int bdb,
            int inp_bd, int ldb) const noexcept;
    size_t zp_comp_b_offset(int bd) const noexcept;
    size_t zp_c_values_offset(brgemm_iteration_t &bi, int ldb) const noexcept;
    bool is_out_bd(const bd_iteration_t *bdi, int bdb, int inp_bd) const;
    int get_out_bd(const bd_iteration_t *bdi, int bdb, int inp_bd) const;

    void maybe_tilestore(brgemm_iteration_t &bi, int bdb_idx, int ldb_idx,
            bool do_pre_tilestore, bool do_post_tilestore);
    int get_C_tensor(brgemm_iteration_t &bi, int m, int n) const noexcept;
    void top_loop(brgemm_iteration_t &bi);
    bd_iteration_t *find_similar(const bd_iteration_t *bdi, bool apply_postops);

    void fill_imap();
};

bool jit_brgemm_amx_uker_base_t::bi_shift_output(
        brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi) {
    res_bi = bi;
    if (shift == 0) return true;

    const auto &tloop = imap_[bi.apply_postops];
    const auto nldis = tloop.ldis.size();
    const auto nbdis = tloop.bdis.size();
    size_t lidx = 0;
    size_t bd_idx = 0;
    size_t ld_idx = 0;
    if (brg.innermost_loop == brgemm_ld_loop_innermost) {
        lidx = bi.bdi->idx * nldis + bi.ldi->idx;
        lidx += shift;
        bd_idx = lidx / nldis;
        ld_idx = lidx % nldis;
    } else if (brg.innermost_loop == brgemm_bd_loop_innermost) {
        lidx = bi.ldi->idx * nbdis + bi.bdi->idx;
        lidx += shift;
        ld_idx = lidx / nbdis;
        bd_idx = lidx % nbdis;
    } else
        assert(!"Unknown loop order!");
    if (lidx >= nldis * nbdis) return false;
    res_bi.bdi = &(tloop.bdis[bd_idx]);
    res_bi.ldi = &(tloop.ldis[ld_idx]);

    return true;
}

bool jit_brgemm_amx_uker_base_t::bi_shift_A(
        brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi) {
    res_bi = bi;
    const auto &tloop = imap_[bi.apply_postops];
    const auto nbdis = tloop.bdis.size();
    const auto nrdis = tloop.rdis.size();

    auto lidx = bi.bdi->idx * nrdis + bi.rdi->idx;
    lidx += shift;
    if (lidx >= nrdis * nbdis) return false;

    const auto bd_idx = lidx / nrdis;
    const auto rd_idx = lidx % nrdis;

    res_bi.bdi = &(tloop.bdis[bd_idx]);
    res_bi.rdi = &(tloop.rdis[rd_idx]);

    return true;
}

bool jit_brgemm_amx_uker_base_t::bi_shift_B(
        brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi) {
    res_bi = bi;
    const auto &tloop = imap_[bi.apply_postops];
    const auto nldis = tloop.ldis.size();
    const auto nrdis = tloop.rdis.size();

    auto lidx = bi.ldi->idx * nrdis + bi.rdi->idx;
    lidx += shift;
    if (lidx >= nrdis * nldis) return false;

    const auto ld_idx = lidx / nrdis;
    const auto rd_idx = lidx % nrdis;

    res_bi.ldi = &(tloop.ldis[ld_idx]);
    res_bi.rdi = &(tloop.rdis[rd_idx]);

    return true;
}

int jit_brgemm_amx_uker_base_t::get_C_tensor(
        brgemm_iteration_t &bi, int m, int n) const noexcept {
    return brg.get_C_tensor(m, n, bi.bdi->is_tail(m), bi.ldi->is_tail(n));
}

void jit_brgemm_amx_uker_base_t::prepare_bd_mask() noexcept {
    if (!brg.brgattr.bd_mask_level) return;
    bd_mask_buffer_ptr_ = brg.brgattr.bd_mask;
    const auto bd_mask_size = brg.bcast_dim;
    adj_bd_mask_buffer_.resize(bd_mask_size);
    skipped_bd_mask_buffer_.resize(bd_mask_size);
    if (bd_mask_buffer_ptr_ != nullptr) {
        int out_ibd = 0;
        for (int i = 0; i < bd_mask_size; i++) {
            adj_bd_mask_buffer_[i] = out_ibd;
            out_ibd += bd_mask_buffer_ptr_[i];
            skipped_bd_mask_buffer_[i] = i;
            for (auto ii = i; ii < bd_mask_size; ii++) {
                if (bd_mask_buffer_ptr_[ii]) {
                    skipped_bd_mask_buffer_[i] = ii;
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
        return skipped_bd_mask_buffer_[inp_bd];
}

size_t jit_brgemm_amx_uker_base_t::A_offset(
        const brgemm_iteration_t &bi, int bdb) const noexcept {
    const auto bs_offs = (brg.type == brgemm_static_offs)
            ? brg.brgattr.static_offsets[bi.bsi->idx].offset.A
            : 0;
    const auto bdb_offs
            = ununroll_bd_loop ? bi.bdi->rel_pos(bdb) : bi.bdi->pos(bdb);
    return bdb_offs * LDA2_size_ + bs_offs
            + bi.rdi->pos(0) * brg.rd_block * brg.typesize_A;
}

size_t jit_brgemm_amx_uker_base_t::B_offset(
        const brgemm_iteration_t &bi, int ldb) const noexcept {
    const auto bs_offs = (brg.type == brgemm_static_offs)
            ? brg.brgattr.static_offsets[bi.bsi->idx].offset.B
            : 0;

    const auto rdb_B_offset = bi.rdi->pos(0) * brg.rd_block * LDB_size_;

    const auto ldb_B_offset = bi.ldi->pos(0) * ld_block_B_size_ * brg.ld_step;

    return rdb_B_offset + ldb_B_offset
            + (brg.is_blocked ? 1 : brg.rd_step) * ldb * ld_block_B_size_
            + bs_offs;
}

size_t jit_brgemm_amx_uker_base_t::C_offset(const brgemm_iteration_t &bi,
        int bdb, int inp_bd, int ldb) const noexcept {
    const auto bi_bd_start = get_out_bd(bi.bdi, 0, 0);
    const auto bd = get_out_bd(bi.bdi, bdb, inp_bd);
    const auto bd_shift = bd - (ununroll_bd_loop ? bi_bd_start : 0);
    return (size_t)bd_shift * LDC2_size_M_ + (size_t)ldb * LDC2_size_N_;
}

size_t jit_brgemm_amx_uker_base_t::D_offset(const brgemm_iteration_t &bi,
        int bdb, int inp_bd, int ldb) const noexcept {
    const auto bi_bd_start = get_out_bd(bi.bdi, 0, 0);
    const auto bd = get_out_bd(bi.bdi, bdb, inp_bd);
    const auto bd_shift = bd - (ununroll_bd_loop ? bi_bd_start : 0);
    return (size_t)bd_shift * LDD_size_ + (size_t)ldb * ld_block_D_size_;
}

size_t jit_brgemm_amx_uker_base_t::lda() const noexcept {
    return LDA_size_;
}

size_t jit_brgemm_amx_uker_base_t::ldb() const noexcept {
    return LDB_size_ * brg.rd_step;
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

size_t jit_brgemm_amx_uker_base_t::zp_comp_pad_a_offset(
        const brgemm_iteration_t &bi, int bdb, int inp_bd,
        int ldb) const noexcept {
    const auto bi_bd_start = get_out_bd(bi.bdi, 0, 0);
    const auto bd = get_out_bd(bi.bdi, bdb, inp_bd);
    const auto bd_shift = bd - (ununroll_bd_loop ? bi_bd_start : 0);
    return (size_t)bd_shift * brg.LDB * sizeof(int32_t)
            + (size_t)ldb * ld_block_zp_size_;
}

size_t jit_brgemm_amx_uker_base_t::zp_comp_b_offset(int bd) const noexcept {
    return sizeof(int32_t) * bd;
}

size_t jit_brgemm_amx_uker_base_t::zp_c_values_offset(
        brgemm_iteration_t &bi, int ldb) const noexcept {
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        return (bi.ldi->is_tail(ldb)) ? ldb_tail_zp_size_
                                      : bi.ldi->pos(ldb) * ld_block_zp_size_;
    }

    return 0;
}

bool jit_brgemm_amx_uker_base_t::is_out_bd(
        const bd_iteration_t *bdi, int bdb, int inp_bd) const {
    const auto bd = bdi->pos(bdb) + inp_bd;
    return IMPLICATION(
            brg.brgattr.bd_mask_level, bdi->bd_mask[bd - bdi->pos(0)] != 0);
}

int jit_brgemm_amx_uker_base_t::get_out_bd(
        const bd_iteration_t *bdi, int bdb, int inp_bd) const {
    if (!is_out_bd(bdi, bdb, inp_bd)) return -1;
    const auto bd = bdi->pos(bdb) + inp_bd;
    if (brg.brgattr.bd_mask_level) {
        assert(bdi->adj_bd_mask[bd - bdi->pos(0)] == adj_bd_mask_buffer_[bd]);
        return bdi->adj_bd_mask[bd - bdi->pos(0)];
    } else
        return bd;
}

Xbyak::Zmm jit_brgemm_amx_uker_base_t::zmm_mask(const Xbyak::Zmm &zmm_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) const {
    return mask_flag ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                     : zmm_in;
}

Xbyak::Ymm jit_brgemm_amx_uker_base_t::ymm_mask(const Xbyak::Ymm &ymm_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) const {
    return mask_flag ? (store ? ymm_in | ktail_mask : ymm_in | ktail_mask | T_z)
                     : ymm_in;
}

Xbyak::Xmm jit_brgemm_amx_uker_base_t::xmm_mask(const Xbyak::Xmm &xmm_in,
        bool mask_flag, bool store, Xbyak::Opmask ktail_mask) const {
    return mask_flag ? (store ? xmm_in | ktail_mask : xmm_in | ktail_mask | T_z)
                     : xmm_in;
}

void jit_brgemm_amx_uker_base_t::cvt2ps(data_type_t type_in,
        const Xbyak::Zmm &zmm_in, const Xbyak::Operand &op, bool mask_flag,
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
        case data_type::f8_e5m2:
            f8_e5m2_emulator_->vcvt_f8_to_f32(zmm, op);
            break;
        case data_type::f8_e4m3:
            f8_e4m3_emulator_->vcvt_f8_to_f32(zmm, op);
            break;
        case data_type::s8: vpmovsxbd(zmm, op); break;
        case data_type::u8: vpmovzxbd(zmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (types::is_integral_dt(type_in)) vcvtdq2ps(zmm_in, zmm_in);
}

void jit_brgemm_amx_uker_base_t::read_params() {
    Label label_done;

    mov(reg_BS, ptr[param1 + GET_OFF(BS)]);

    mov(reg_addr_batch, ptr[param1 + GET_OFF(batch)]);

    mov(reg_buf, ptr[param1 + GET_OFF(ptr_buf)]);

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_a, ptr[param1 + GET_OFF(a_zp_compensations)]);
        mov(ptr[rsp + reg_zp_comp_a_offs_], reg_zp_comp_a);
        mov(reg_zp_a_values, ptr[param1 + GET_OFF(zp_a_val)]);
        mov(ptr[rsp + reg_zp_a_values_offs_], reg_zp_a_values);

        if (brg.req_comp_pads_with_bcast)
            mov(reg_zp_comp_pad_a, ptr[param1 + GET_OFF(a_zp_compensations)]);
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
    size_t ils_shift = 0;
    if (may_load_accumulators_) {
        mov(reg_stride_ld_block, LDC_size_);
        const auto need_ils_shift
                = (actual_ils(bi.apply_postops, bi.skip_accumulation)
                        && ununroll_bd_loop && bi.ldi->idx == 0);
        // if need_ils_shift then we have to add shift to C because reg_C points
        // to previous iteration in this case
        ils_shift = need_ils_shift ? bi.bdi->C_shift : 0;
    }

    for_(int bdb = 0; bdb < bi.bdi->block2(); bdb++)
    for (int ldb = 0; ldb < bi.ldi->block2(); ldb++) {
        if (may_load_accumulators_) {
            auto c_offset = C_offset(bi, bdb, 0, bi.ldi->pos(ldb)) + ils_shift;
            tileloadd(Tmm(get_C_tensor(bi, bdb, ldb)),
                    ptr[reg_C + c_offset + reg_stride_ld_block]);
        } else {
            // call tilezero on very first iteration
            if (!brg.interleave_tilestores_
                    || everyone_is(0u, bi.bdi->idx, bi.ldi->idx))
                tilezero(Tmm(get_C_tensor(bi, bdb, ldb)));
        }
    }
}

void jit_brgemm_amx_uker_base_t::apply_alpha_beta_to_vector(
        int idx, const Address &addr, bool is_ld_tail) {
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

void jit_brgemm_amx_uker_base_t::apply_post_ops_to_range(
        brgemm_iteration_t &bi, int bd_start, int bd_finish, int bdb, int ldb) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    const auto ldb_pos = bi.ldi->pos(ldb);
    const auto is_ld_tail = bi.ldi->is_tail(ldb);

    if (brg.with_binary) {
        if (handle_binary_po_offset_) {
            for (auto bd = bd_start; bd < bd_finish; bd++) {
                // We have no way to tell the injector to skip some vectors.
                // Therefore, we must set parameters correctly for all registers.
                // TODO: Make it possible to specify "skipped" vectors to injector
                const auto idx = accm(bd).getIdx();
                if (is_ld_tail) rhs_arg_params.vmm_tail_idx_.emplace(idx);
                rhs_arg_params.vmm_idx_to_out_reg.emplace(idx, reg_D);

                if (!is_out_bd(bi.bdi, bdb, bd)) continue;

                const auto d_offset = D_offset(bi, bdb, bd, ldb_pos);
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
            const auto &zmm_sum_zp = zmm_tmp_2();
            if (p_sum_zp_reg_set) {
                mov(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
                vcvtdq2ps(zmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
            }
            if (p_sum_scale_reg_set)
                mov(reg_ptr_sum_scale, reinterpret_cast<size_t>(p_sum_scale));

            const auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
            const auto zmm_prev_dst = Xbyak::Zmm(0);

            for (auto bd = bd_start; bd < bd_finish; bd++) {
                if (!is_out_bd(bi.bdi, bdb, bd)) continue;

                auto zmm = accm(bd);
                const auto d_offset = D_offset(bi, bdb, bd, ldb_pos);
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

    // Using knowledge how "accm" assign zmm registers.
    // TODO: make this code more clear
    const auto finish_idx = accm(bd_start).getIdx() + 1;
    const auto start_idx = accm(bd_finish - 1).getIdx();
    postops_injector_->compute_vector_range(
            start_idx, finish_idx, rhs_arg_params);
}

void jit_brgemm_amx_uker_base_t::maybe_saturation(Xbyak::Zmm &zmm) {
    if (!dt_requires_saturation_) return;
    saturate_cvt_f32(zmm, zmm_lbound, zmm_ubound, brg.dt_d);
}

void jit_brgemm_amx_uker_base_t::prepare_post_ops_registers_ldb(
        brgemm_iteration_t &bi, int ldb) {
    if (!bi.apply_postops) return;
    auto k_mask = (!bi.ldi->is_tail(ldb)) ? ld_full_mask : ld_tail_mask;

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        const auto zmm_zp_a_val = zmm_tmp_1();
        mov(reg_zp_a_values, ptr[rsp + reg_zp_a_values_offs_]);
        vpbroadcastd(zmm_zp_a_val, reg_zp_a_values.cvt32());
        vcvtdq2ps(zmm_zp_a_val, zmm_zp_a_val);
        mov(reg_aux_zp_comp_a, ptr[rsp + reg_zp_comp_a_offs_]);

        const auto zp_comp_a_off = zp_comp_a_offset(bi.ldi->pos(ldb));
        const auto zp_comp_a_addr
                = EVEX_compress_addr(reg_aux_zp_comp_a, zp_comp_a_off);
        cvt2ps(data_type::s32, zmm_zp_comp_a, zp_comp_a_addr, true, false,
                k_mask);
        vmulps(zmm_zp_comp_a, zmm_zp_comp_a, zmm_zp_a_val);
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        mov(reg_zp_c_values, ptr[rsp + reg_zp_c_values_offs_]);
        if (brg.zp_type_c == brgemm_broadcast_t::per_tensor) {
            vcvtdq2ps(zmm_zp_c, EVEX_compress_addr(reg_zp_c_values, 0, true));
        }
        if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
            const auto zp_c_off = zp_c_values_offset(bi, ldb);
            const auto zp_c_addr
                    = EVEX_compress_addr(reg_zp_c_values, zp_c_off);
            cvt2ps(data_type::s32, zmm_zp_c, zp_c_addr, true, false, k_mask);
        }
    }
}

void jit_brgemm_amx_uker_base_t::prepare_post_ops_registers(
        brgemm_iteration_t &bi) {
    if (!bi.apply_postops) return;
    const auto ldi = bi.ldi;

    if (brg.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);

        for (int ldb = 0; ldb < ldi->block2(); ldb++) {
            auto ptr_bias
                    = EVEX_compress_addr(reg_bias, bias_offset(ldi->pos(ldb)));
            auto k_mask = ldi->is_tail(ldb) ? ld_tail_mask : ld_full_mask;
            cvt2ps(brg.dt_bias, zmm_bias(ldb), ptr_bias, true, false, k_mask);
        }
    }

    if (brg.with_scales) {
        mov(reg_scales, ptr[param1 + GET_OFF(ptr_scales)]);
        for (int ldb = 0; ldb < ldi->block2(); ldb++) {
            auto scales_ptr = EVEX_compress_addr(
                    reg_scales, scales_offset(ldi->pos(ldb)));
            auto k_mask = ldi->is_tail(ldb) ? ld_tail_mask : ld_full_mask;
            vmovups(zmm_scales(ldb) | k_mask | T_z, scales_ptr);
        }
    }
}

void jit_brgemm_amx_uker_base_t::uni_prefetch(
        const Address &addr, brgemm_kernel_prefetching_t pft, bool for_write) {
    if (for_write) {
        switch (pft) {
            case brgemm_prf0: prefetchw(addr); break;
            default: break;
        }
    } else {
        switch (pft) {
            case brgemm_prf0: prefetcht0(addr); break;
            case brgemm_prf1: prefetcht1(addr); break;
            case brgemm_prf2: prefetcht2(addr); break;
            case brgemm_prfNTA: prefetchnta(addr); break;
            default: break;
        }
    }
}

void jit_brgemm_amx_uker_base_t::prefetch_CD_range(brgemm_iteration_t &bi,
        brgemm_kernel_prefetching_t pft, int bd_start, int bd_finish, int bdb,
        int ldb) {
    const auto ldb_pos = bi.ldi->pos(ldb);
    for (int bd = bd_start; bd < bd_finish; bd++) {
        if (!is_out_bd(bi.bdi, bdb, bd)) continue;
        if (bi.apply_postops) {
            const auto d_offset = D_offset(bi, bdb, bd, ldb_pos);
            auto ptr_D = EVEX_compress_addr(reg_D, d_offset);
            uni_prefetch(ptr_D, pft, true);
        } else if (are_post_ops_applicable_) {
            const auto c_offset = C_offset(bi, bdb, bd, ldb_pos);
            auto ptr_C = EVEX_compress_addr(reg_C, c_offset);
            uni_prefetch(ptr_C, pft, true);
        } else {
            const auto d_offset = D_offset(bi, bdb, bd, ldb_pos);
            auto ptr_D = EVEX_compress_addr(reg_D, d_offset);
            uni_prefetch(ptr_D, pft, true);
        }
    }
}

int jit_brgemm_amx_uker_base_t::calc_ops_CD(
        brgemm_iteration_t &bi) const noexcept {
    const auto &tloop = imap_[bi.apply_postops];
    return tloop.rdis.size() * bi.ldi->block2() * bi.bdi->block2()
            * (brg.brgattr.var_bs ? 1 : brg.brgattr.max_bs);
}

void jit_brgemm_amx_uker_base_t::prefetch_CD(brgemm_iteration_t &bi,
        brgemm_iteration_t &pfo_bi, prf_t &prf, bool prefetch_all) {

    const auto calc_ops = calc_ops_CD(bi);
    const auto bdb_row = pfo_bi.bdi->block(0) * pfo_bi.ldi->block2();
    const auto tot_vecs = pfo_bi.bdi->length() * pfo_bi.ldi->block2();
    const auto pfo_vecs_per_store = (calc_ops) ? div_up(tot_vecs, calc_ops) : 0;

    const auto nvecs = prefetch_all
            ? tot_vecs
            : nstl::min(pfo_vecs_per_store, tot_vecs - prf.vec);

    const auto out_typesize
            = (are_post_ops_applicable_ && !prev_bi_.apply_postops)
            ? brg.typesize_C
            : brg.typesize_D;
    for (int iv = 0; iv < nvecs && prf.vec < tot_vecs; iv++) {
        const auto bdb = prf.vec / bdb_row;
        const auto vec_in_bdb_row = prf.vec - bdb * bdb_row;
        const auto ldb = vec_in_bdb_row / pfo_bi.bdi->block(bdb);
        const auto bd = vec_in_bdb_row % pfo_bi.bdi->block(bdb);
        // prefetch output cache lines only once
        if (pfo_bi.ldi->pos(ldb) % (4 / out_typesize) == 0) {
            prefetch_CD_range(pfo_bi, prf.pft, bd, bd + 1, bdb, ldb);
        }
        prf.vec++;
    }
}

void jit_brgemm_amx_uker_base_t::prefetch_A(brgemm_iteration_t &bi,
        brgemm_iteration_t &pfo_bi, prf_t &prf, bool prefetch_all) {

    const auto calc_ops = bi.ldi->block2() * bi.bdi->block2();
    const auto tot_vecs = pfo_bi.bdi->length();
    const auto pfo_vecs_per_store = (calc_ops) ? div_up(tot_vecs, calc_ops) : 0;

    const auto nvecs = prefetch_all
            ? tot_vecs
            : nstl::min(pfo_vecs_per_store, tot_vecs - prf.vec);

    for (int iv = 0; iv < nvecs && prf.vec < tot_vecs; iv++) {
        const auto bdb = prf.vec / pfo_bi.bdi->block(0);
        const auto bd = prf.vec % pfo_bi.bdi->block(0);

        //TODO: looks like we have to prefetch in each bs separately
        const auto ptr_A = EVEX_compress_addr(
                reg_A, A_offset(pfo_bi, bdb) + bd * LDA_size_);
        uni_prefetch(ptr_A, prf.pft, false);
        prf.vec++;
    }
}

void jit_brgemm_amx_uker_base_t::prefetch_B(brgemm_iteration_t &bi,
        brgemm_iteration_t &pfo_bi, prf_t &prf, bool prefetch_all) {

    const auto calc_ops = bi.ldi->block2() * bi.bdi->block2();
    const auto tot_vecs = pfo_bi.ldi->length();
    const auto pfo_vecs_per_store = (calc_ops) ? div_up(tot_vecs, calc_ops) : 0;

    const auto nvecs = prefetch_all
            ? tot_vecs
            : nstl::min(pfo_vecs_per_store, tot_vecs - prf.vec);

    // TODO: check these addressing for correctness
    for (int iv = 0; iv < nvecs && prf.vec < tot_vecs; iv++) {

        const auto ldb = prf.vec / pfo_bi.rdi->block(0);
        const auto rb = prf.vec % pfo_bi.rdi->block(0);
        //TODO: looks like we have to prefetch in each bs separately
        const auto ptr_B = EVEX_compress_addr(
                reg_B, B_offset(pfo_bi, ldb) + rb * LDB_size_);

        uni_prefetch(ptr_B, prf.pft, false);
        prf.vec++;
    }
}

void jit_brgemm_amx_uker_base_t::prefetching(
        brgemm_iteration_t &bi, bool prefetch_all) {
    // for var_bs we do prefetch on last iteration by bs only
    if (brg.brgattr.var_bs && !bi.last_bsi) return;
    brgemm_iteration_t pfo_bi;
    auto maybe_prefetch_C = [&](prf_t &prf) {
        if (prf.dist < 0) return;
        bool is_pfo_bi = false;
        brgemm_iteration_t pfo_bi;
        if (actual_ils(bi.apply_postops, bi.skip_accumulation)) {
            if (was_prev_bi_ && prf.dist == 0) {
                is_pfo_bi = true;
                pfo_bi = prev_bi_;
            } else if (prf.dist > 0) {
                is_pfo_bi = bi_shift_output(bi, prf.dist - 1, pfo_bi);
            }
        } else {
            is_pfo_bi = bi_shift_output(bi, prf.dist, pfo_bi);
        }
        if (is_pfo_bi) prefetch_CD(bi, pfo_bi, prf, prefetch_all);
    };

    auto maybe_prefetch_A = [&](prf_t &prf) {
        if (prf.dist < 0) return;
        if (bi_shift_A(bi, prf.dist, pfo_bi))
            prefetch_A(bi, pfo_bi, prf, prefetch_all);
    };

    auto maybe_prefetch_B = [&](prf_t &prf) {
        if (prf.dist < 0) return;
        if (bi_shift_B(bi, prf.dist, pfo_bi))
            prefetch_B(bi, pfo_bi, prf, prefetch_all);
    };

    maybe_prefetch_C(prf0C);
    maybe_prefetch_C(prf1C);

    maybe_prefetch_A(prf0A);
    maybe_prefetch_A(prf1A);
    maybe_prefetch_A(prf2A);
    maybe_prefetch_A(prfntaA);

    maybe_prefetch_B(prf0B);
    maybe_prefetch_B(prf1B);
    maybe_prefetch_B(prf2B);
    maybe_prefetch_B(prfntaB);
}

void jit_brgemm_amx_uker_base_t::apply_comp_pad_to_vector(
        brgemm_iteration_t &bi, int bdb, int inp_bd, int ldb, const int idx) {
    const auto is_ld_tail = bi.ldi->is_tail(ldb);
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
    auto zmm = Zmm(idx);
    auto zmm_masked = zmm | k_mask | T_z;
    const auto zmm_zp_a_val = zmm_tmp_1();

    mov(reg_zp_a_values, ptr[rsp + reg_zp_a_values_offs_]);
    vpbroadcastd(zmm_zp_a_val, reg_zp_a_values.cvt32());
    vcvtdq2ps(zmm_zp_a_val, zmm_zp_a_val);
    mov(reg_aux_zp_comp_a, ptr[rsp + reg_zp_comp_a_offs_]);
    const auto comp_pad_offset
            = zp_comp_pad_a_offset(bi, bdb, inp_bd, bi.ldi->pos(ldb));
    const auto zp_comp_pad_a_addr
            = EVEX_compress_addr(reg_zp_comp_pad_a, comp_pad_offset);
    cvt2ps(data_type::s32, zmm_zp_comp_a, zp_comp_pad_a_addr, true, false,
            k_mask);
    vmulps(zmm_zp_comp_a, zmm_zp_comp_a, zmm_zp_a_val);
    vaddps(zmm_masked, zmm, zmm_zp_comp_a);
}

void jit_brgemm_amx_uker_base_t::process_output_range(
        brgemm_iteration_t &bi, int bd_start, int bd_finish, int bdb, int ldb) {

    const auto k_mask = bi.ldi->is_tail(ldb) ? ld_tail_mask : ld_full_mask;

    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are already converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dq2ps_required = brg.is_int8
            && IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);

    bool some_bd_mask = false;
    for (auto bd = bd_start; bd < bd_finish; bd++) {
        auto zmm = accm(bd);
        if (!is_out_bd(bi.bdi, bdb, bd)) continue;

        auto vreg_acc = bi.ldi->is_tail(ldb) ? accm(bd) | ld_tail_mask | T_z
                                             : accm(bd);
        some_bd_mask = true;

        if (bi.skip_accumulation) {
            vpxord(vreg_acc, vreg_acc, vreg_acc);
        } else {
            const auto wsp_offset = (use_ils_ || brg.interleave_tilestores_)
                    ? (bdb * prev_bi_.ldi->block2() + ldb)
                            * prev_bi_.bdi->block(0) * ld_block_C_size_
                    : 0;
            const auto buf_offset = bd * ld_block_C_size_;
            vmovups(vreg_acc, ptr[reg_buf + buf_offset + wsp_offset]);
        }

        const auto c_offset = C_offset(bi, bdb, bd, bi.ldi->pos(ldb));
        const auto ptr_C = EVEX_compress_addr(reg_C, c_offset);

        if (need_to_apply_alpha_beta_ || bi.skip_accumulation)
            apply_alpha_beta_to_vector(
                    zmm.getIdx(), ptr_C, bi.ldi->is_tail(ldb));

        if (!bi.apply_postops) continue;

        if (dq2ps_required) vcvtdq2ps(zmm, zmm);

        if (brg.req_comp_pads_with_bcast)
            apply_comp_pad_to_vector(bi, bdb, bd, ldb, zmm.getIdx());
    }

    if (!bi.apply_postops || !some_bd_mask) return;

    if (brg.zp_type_a != brgemm_broadcast_t::none
            && !brg.req_comp_pads_with_bcast) {
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);
            vaddps(zmm, zmm, zmm_zp_comp_a);
        }
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_b, ptr[rsp + reg_zp_comp_b_offs_]);

        auto zmm_zp_comp_b = zmm_tmp_1();
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);

            const auto zp_comp_b_off
                    = zp_comp_b_offset(get_out_bd(bi.bdi, bdb, bd));
            vcvtdq2ps(zmm_zp_comp_b,
                    EVEX_compress_addr(reg_zp_comp_b, zp_comp_b_off, true));

            vaddps(zmm, zmm, zmm_zp_comp_b);
        }
    }

    if (brg.with_scales) {
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);
            const Xbyak::Zmm scaled_zmm = zmm_mask(zmm, true, false, k_mask);
            vmulps(scaled_zmm, scaled_zmm, zmm_scales(ldb));
        }
    }

    if (brg.with_bias) {
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);
            vaddps(zmm, zmm, zmm_bias(ldb));
        }
    }

    if (postops_injector_) {
        apply_post_ops_to_range(bi, bd_start, bd_finish, bdb, ldb);
    }

    if (brg.with_dst_scales) {
        mov(reg_dst_scales, ptr[param1 + GET_OFF(ptr_dst_scales)]);
        auto zmm_dst_scales = zmm_tmp_1();
        vbroadcastss(zmm_dst_scales, ptr[reg_dst_scales]);
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);
            vmulps(zmm, zmm, zmm_dst_scales);
        }
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);
            vaddps(zmm, zmm, zmm_zp_c);
        }
    }
}

void jit_brgemm_amx_uker_base_t::store_vector_with_post_ops(
        int idx, const Address &addr, bool is_ld_tail) {
    auto zmm = Zmm(idx);

    maybe_saturation(zmm);

    auto ymm = Xbyak::Ymm(idx);
    auto xmm = Xbyak::Xmm(idx);
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
    const Xbyak::Zmm r_zmm = zmm_mask(zmm, true, true, k_mask);
    const Xbyak::Ymm r_ymm = ymm_mask(ymm, true, true, k_mask);
    const Xbyak::Xmm r_xmm = xmm_mask(xmm, true, true, k_mask);

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
        case data_type::f8_e5m2:
            f8_e5m2_emulator_->vcvt_f32_to_f8(xmm, zmm);
            vmovdqu8(addr, r_xmm);
            break;
        case data_type::f8_e4m3:
            f8_e4m3_emulator_->vcvt_f32_to_f8(xmm, zmm);
            vmovdqu8(addr, r_xmm);
            break;
        case data_type::s8: vpmovsdb(addr, r_zmm); break;
        case data_type::u8: vpmovusdb(addr, r_zmm); break;
        default: assert(!"unknown dst_dt");
    }
}

void jit_brgemm_amx_uker_base_t::store_vector_without_post_ops(
        int idx, const Address &addr, bool is_ld_tail) {
    auto zmm = Zmm(idx);

    maybe_saturation(zmm);

    if (is_ld_tail)
        vmovups(addr | ld_tail_mask | T_z, zmm);
    else
        vmovups(addr, zmm);
}

void jit_brgemm_amx_uker_base_t::store_vector(
        brgemm_iteration_t &bi, int bdb, int inp_bd, int ldb) {

    if (!is_out_bd(bi.bdi, bdb, inp_bd)) return;

    auto vreg_acc = bi.ldi->is_tail(ldb) ? accm(inp_bd) | ld_tail_mask | T_z
                                         : accm(inp_bd);

    auto ldb_pos = bi.ldi->pos(ldb);
    auto is_ld_tail = bi.ldi->is_tail(ldb);
    const auto c_offset = C_offset(bi, bdb, inp_bd, ldb_pos);
    const auto d_offset = D_offset(bi, bdb, inp_bd, ldb_pos);

    auto ptr_C = EVEX_compress_addr(reg_C, c_offset);
    auto ptr_D = EVEX_compress_addr(reg_D, d_offset);

    if (bi.apply_postops)
        store_vector_with_post_ops(vreg_acc.getIdx(), ptr_D, is_ld_tail);
    else if (are_post_ops_applicable_)
        store_vector_without_post_ops(vreg_acc.getIdx(), ptr_C, is_ld_tail);
    else
        store_vector_without_post_ops(vreg_acc.getIdx(), ptr_D, is_ld_tail);
}

void jit_brgemm_amx_uker_base_t::interleave_store(
        brgemm_iteration_t &bi, bool store_all) {

    if (store_all) { prev_bi_ = bi; }
    if (!was_prev_bi_) return;
    if (!actual_ils(prev_bi_.apply_postops, bi.skip_accumulation)) return;

    if (store_all) prefetching(prev_bi_, true);

    auto cur_bdb = ils_bdb_;
    auto cur_ldb = ils_ldb_;

    // if first block
    if (ils_vec_ == 0) {
        if (!prepare_post_ops_registers_once_) {
            prepare_post_ops_registers(prev_bi_);
        }
        prepare_post_ops_registers_ldb(prev_bi_, 0);
        ils_bd_start_ = 0;
        auto bd_finish = nstl::min(ils_bd_step_, prev_bi_.bdi->block(0));
        process_output_range(prev_bi_, 0, bd_finish, cur_bdb, cur_ldb);
    }

    const auto calc_ops = calc_ops_CD(bi);
    // we use maximum estimation (prev_bi_.bdi.block2() * prev_bi_.bdi.block())
    // to calculate ils_store_ops to avoid error when we didn't store all
    // vectors from tile buffer but it is already overwritten in a new iteration
    const auto ils_store_ops = prev_bi_.ldi->block2() * prev_bi_.bdi->block2()
            * prev_bi_.bdi->block(0);
    const auto ils_vecs_per_store
            = (calc_ops) ? div_up(ils_store_ops, calc_ops) : 0;

    // last bd_block may be bd_tail
    const auto bdb_row = prev_bi_.bdi->block(0) * prev_bi_.ldi->block2();
    const auto total_vectors = prev_bi_.bdi->length() * prev_bi_.ldi->block2();
    const auto nvecs = store_all ? total_vectors : ils_vecs_per_store;
    for (int vec = 0; vec < nvecs && ils_vec_ < total_vectors; vec++) {
        const auto bdb = ils_vec_ / bdb_row;
        const auto vec_in_bdb_row = ils_vec_ - bdb * bdb_row;
        const auto ldb = vec_in_bdb_row / prev_bi_.bdi->block(bdb);
        const auto bd = vec_in_bdb_row % prev_bi_.bdi->block(bdb);

        if (ldb != cur_ldb) prepare_post_ops_registers_ldb(prev_bi_, ldb);

        if (bdb != cur_bdb || ldb != cur_ldb
                || rnd_dn(bd, ils_bd_step_) != ils_bd_start_) {
            ils_bd_start_ = rnd_dn(bd, ils_bd_step_);
            auto bd_finish = nstl::min(
                    ils_bd_start_ + ils_bd_step_, prev_bi_.bdi->block(bdb));
            process_output_range(prev_bi_, ils_bd_start_, bd_finish, bdb, ldb);
        }

        store_vector(prev_bi_, bdb, bd, ldb);
        cur_bdb = bdb;
        cur_ldb = ldb;
        ils_vec_++;
    }
    ils_ldb_ = cur_ldb;
    ils_bdb_ = cur_bdb;
}

void jit_brgemm_amx_uker_base_t::store_accumulators(brgemm_iteration_t &bi) {

    const auto store_by_vectors = get_store_by_vectors(bi.apply_postops);

    if (store_by_vectors) {
        if (!brg.interleave_tilestores_)
            mov(reg_stride_ld_block, ld_block_C_size_);
    } else
        mov(reg_stride_ld_block, LDC_size_);

    prev_bi_ = bi;
    was_prev_bi_ = true;

    ils_vec_ = 0;
    ils_bdb_ = 0;
    ils_ldb_ = 0;

    prf0C.reset();
    prf1C.reset();

    const bool real_ils = actual_ils(bi.apply_postops, bi.skip_accumulation);
    if (store_by_vectors && !real_ils && !prepare_post_ops_registers_once_)
        prepare_post_ops_registers(bi);

    for_(int bdb = 0; bdb < bi.bdi->block2(); bdb++)
    for (int ldb = 0; ldb < bi.ldi->block2(); ldb++) {
        if (store_by_vectors) {
            if (!brg.interleave_tilestores_ && !bi.skip_accumulation) {
                const auto wsp_offset = use_ils_
                        ? (bdb * bi.ldi->block2() + ldb) * bi.bdi->block(0)
                                * ld_block_C_size_
                        : 0;
                tilestored(ptr[reg_buf + reg_stride_ld_block + wsp_offset],
                        Tmm(get_C_tensor(bi, bdb, ldb)));
            }
            if (real_ils) continue;

            prepare_post_ops_registers_ldb(bi, ldb);

            for (int bd_step = 0; bd_step < bi.bdi->block(bdb);
                    bd_step += ils_bd_step_) {
                auto bd_finish
                        = nstl::min(bd_step + ils_bd_step_, bi.bdi->block(bdb));
                process_output_range(bi, bd_step, bd_finish, bdb, ldb);

                for (auto bd = bd_step; bd < bd_finish; bd++)
                    store_vector(bi, bdb, bd, ldb);
            }
        } else if (!brg.interleave_tilestores_) {
            const auto c_offset = C_offset(bi, bdb, 0, bi.ldi->pos(ldb));
            tilestored(ptr[reg_C + reg_stride_ld_block + c_offset],
                    Tmm(get_C_tensor(bi, bdb, ldb)));
        }
    }
}

void jit_brgemm_amx_uker_base_t::set_A_B_matrices(int bs) {
    if (one_of(brg.type, brgemm_static_offs)) return;
    assert(one_of(brg.type, brgemm_addr, brgemm_offs));
    if (brg.brgattr.max_bs == 1) return;
    auto batch_offset = (size_t)bs * sizeof(brgemm_batch_element_t);
    if (brg.type == brgemm_addr) {
        if (brg.layout == brgemm_row_major) {
            mov(reg_A,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(ptr.A)));
            mov(reg_B,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(ptr.B)));
        } else {
            mov(reg_A,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(ptr.B)));
            mov(reg_B,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(ptr.A)));
        }
    } else if (brg.type == brgemm_offs) {
        if (brg.layout == brgemm_row_major) {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_A)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_B)]);
            add(reg_A,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(offset.A)));
            add(reg_B,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(offset.B)));
        } else {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_B)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_A)]);
            add(reg_A,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(offset.B)));
            add(reg_B,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(offset.A)));
        }
    }
}

void jit_brgemm_amx_uker_base_t::set_A_B_matrices() {
    if (one_of(brg.type, brgemm_static_offs)) return;
    assert(one_of(brg.type, brgemm_addr, brgemm_offs));
    assert(brg.brgattr.var_bs);
    if (brg.brgattr.max_bs == 1) return;

    if (brg.type == brgemm_addr) {
        if (brg.layout == brgemm_row_major) {
            mov(reg_A, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.A)]);
            mov(reg_B, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.B)]);
        } else {
            mov(reg_A, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.B)]);
            mov(reg_B, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.A)]);
        }
    } else if (brg.type == brgemm_offs) {
        if (brg.layout == brgemm_row_major) {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_A)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_B)]);
            add(reg_A, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(offset.A)]);
            add(reg_B, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(offset.B)]);
        } else {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_B)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_A)]);
            add(reg_A, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(offset.B)]);
            add(reg_B, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(offset.A)]);
        }
    }
}

void jit_brgemm_amx_uker_base_t::maybe_tileloadd_nt(
        brgemm_iteration_t &bi, matrix_kind_t mk, int xdb, size_t offset) {

    const bool is_A = mk == matrix_kind_t::matrix_A;
    bool load_nt = is_A ? brg.load_nt_A : brg.load_nt_B;

    auto t1 = Tmm(is_A ? brg.get_A_tensor(xdb, bi.bdi->is_tail(xdb))
                       : brg.get_B_tensor(xdb, bi.ldi->is_tail(xdb)));
    auto reg_base = is_A ? reg_A : reg_B;
    auto reg_stride = is_A ? reg_stride_lda : reg_stride_ldb;

    if (brg.is_input_convert()) {
        // try_load_nt is not supported in maybe_pre_process_data as there is
        // no guarantee that the data is cache line aligned.
        maybe_pre_process_data(bi, t1, reg_base, offset, reg_stride, mk);
        return;
    }

    if (maybe_pre_process_k_tail(bi, xdb, t1, reg_base, offset, reg_stride, mk))
        return;

    if (load_nt)
        tileloaddt1(t1, ptr[reg_base + offset + reg_stride]);
    else
        tileloadd(t1, ptr[reg_base + offset + reg_stride]);
}

void jit_brgemm_amx_uker_base_t::maybe_tilestore(brgemm_iteration_t &bi,
        int bdb_idx, int ldb_idx, bool do_pre_tilestore,
        bool do_post_tilestore) {
    if (bi.skip_accumulation) return;
    auto current_tensor_idx = get_C_tensor(bi, bdb_idx, ldb_idx);

    if (!brg.interleave_tilestores_) return;
    const auto current_tensor_number
            = current_tensor_idx - get_C_tensor(bi, 0, 0);
    const auto store_tensor_shift
            = do_pre_tilestore ? (bi.bdi->block2() == 1 ? 2 : 1) : 0;
    const auto store_tensor_idx = current_tensor_idx + store_tensor_shift;
    const auto store_tensor_number = current_tensor_number + store_tensor_shift;

    const auto &store_bi = do_pre_tilestore ? prev_bi_ : bi;
    const int max_store_tensor_number
            = store_bi.bdi->blocks.size() * store_bi.ldi->blocks.size();
    bool perform_store
            = (do_pre_tilestore
                      && (store_tensor_number >= 2
                              && store_tensor_number < max_store_tensor_number))
            || (do_post_tilestore && (store_tensor_number < 2));

    if (!perform_store) return;
    if (do_pre_tilestore) {
        bdb_idx = store_tensor_idx / bi.ldi->block2();
        ldb_idx = store_tensor_idx % bi.ldi->block2();
    }
    const bool store_by_vectors = get_store_by_vectors(bi.apply_postops);
    Tmm acc = Tmm(store_tensor_idx);
    if (store_by_vectors) {
        const auto wsp_offset = (use_ils_ || brg.interleave_tilestores_)
                ? (bdb_idx * bi.ldi->block2() + ldb_idx) * bi.bdi->block(0)
                        * ld_block_C_size_
                : 0;
        tilestored(ptr[reg_buf + reg_stride_ld_block + wsp_offset], acc);
    } else {
        const auto store_ldb_ind
                = do_pre_tilestore ? prev_bi_.ldi->pos(0) : bi.ldi->pos(0);
        const auto c_offset
                = C_offset(store_bi, bdb_idx, 0, store_ldb_ind + ldb_idx);
        tilestored(ptr[reg_C + reg_stride_ld_block + c_offset], acc);
    }
    tilezero(acc);
}

void jit_brgemm_amx_uker_base_t::tdpbxxd(brgemm_iteration_t &bi, int bdb_idx,
        int ldb_idx, bool do_pre_tilestore, bool do_post_tilestore) {
    prefetching(bi, false);
    maybe_tilestore(bi, bdb_idx, ldb_idx, do_pre_tilestore, false);

    const Tmm &x1 = Tmm(get_C_tensor(bi, bdb_idx, ldb_idx));
    const Tmm &x2 = Tmm(brg.get_A_tensor(bdb_idx, bi.bdi->is_tail(bdb_idx)));
    const Tmm &x3 = Tmm(brg.get_B_tensor(ldb_idx, bi.ldi->is_tail(ldb_idx)));

    if (brg.is_bf32
            || (brg.dt_a == data_type::bf16 && brg.dt_b == data_type::bf16)) {
        tdpbf16ps(x1, x2, x3);
    } else if (brg.dt_a == data_type::f16 && brg.dt_b == data_type::f16) {
        tdpfp16ps(x1, x2, x3);
    } else if (brg.is_fp8) {
        if (brg.is_fp8_via_convert())
            tdpfp16ps(x1, x2, x3);
        else
            assert(!"Not supported!");
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
    maybe_tilestore(bi, bdb_idx, ldb_idx, false, do_post_tilestore);
}

// This method up-converts the data from bf8 to f16 and saves at reg_buf.
// Generally used by matrix_A, where no vnni transformation of data is needed.
void jit_brgemm_amx_uker_base_t::fp8_to_f16_upconvert(brgemm_iteration_t &bi,
        int num_rows, int tile_num_col_bytes, reg64_t reg_data, int offset,
        reg64_t reg_data_stride, reg64_t reg_buf, data_type_t dt) {
    const auto rd_block = bi.rdi->block(0);
    const int max_num_cols
            = nstl::min<int>(tile_num_col_bytes / sizeof(float16_t), rd_block);
    const int col_tail = max_num_cols % 32;
    auto zmm_1 = zmm_tmp_1();
    auto zmm_1_masked = col_tail ? zmm_1 | fp_col_mask | T_z : zmm_1;

    assert(max_num_cols > 0);

    if (col_tail) {
        const auto tail_mask = (static_cast<size_t>(1) << col_tail) - 1;
        mov(reg_tmp_gpr, tail_mask);
        kmovq(fp_col_mask, reg_tmp_gpr);
    }

    // Note: using the same register used in col_tail, so order is important
    const auto reg_data_aux = reg_tmp_gpr;
    lea(reg_data_aux, ptr[reg_data + offset]);

    for (int r = 0; r < num_rows; ++r) {
        if (dt == data_type::f8_e5m2)
            f8_e5m2_emulator_->vcvt_f8_to_f16(zmm_1_masked, ptr[reg_data_aux]);
        else if (dt == data_type::f8_e4m3)
            f8_e4m3_emulator_->vcvt_f8_to_f16(zmm_1_masked, ptr[reg_data_aux]);
        else
            assert(!"unsupported data type");

        vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_1);
        add(reg_data_aux, reg_data_stride);
    }
}

// This method down-converts the data from f32 to bf16 and saves at reg_buf.
// Generally used by matrix_A, where no vnni transformation of data is needed.
void jit_brgemm_amx_uker_base_t::bf32_downconvert(brgemm_iteration_t &bi,
        int num_rows, int tile_num_col_bytes, reg64_t reg_data, int offset,
        reg64_t reg_data_stride, reg64_t reg_buf) {
    const auto rd_block = bi.rdi->block(0);
    const auto max_num_cols
            = nstl::min<int>(tile_num_col_bytes / sizeof(bfloat16_t), rd_block);
    const auto col_tail = max_num_cols % simd_w;
    auto zmm_1 = zmm_tmp_1();
    auto zmm_2 = zmm_tmp_2();
    auto zmm_2_masked = col_tail ? zmm_2 | fp_col_mask | T_z : zmm_2;

    assert(max_num_cols > 0);

    if (col_tail) {
        const auto tail_mask = (static_cast<size_t>(1) << col_tail) - 1;
        mov(reg_tmp_gpr, tail_mask);
        kmovq(fp_col_mask, reg_tmp_gpr);
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
                    = max_num_cols == 16 ? ymm_1 : ymm_1 | fp_col_mask | T_z;
            vcvtneps2bf16(ymm_1_masked, ptr[reg_data_aux]);
            vmovups(ptr[reg_buf + r * zmm_width_in_bytes], ymm_1);
        }
        add(reg_data_aux, reg_data_stride);
    }
}

// This method up-converts and transforms the data from fp8_vnni to f16_vnni
// format. Generally used by matrix_B.
void jit_brgemm_amx_uker_base_t::fp8_to_f16_upconvert_to_vnni(
        brgemm_iteration_t &bi, int num_rows, int tile_num_col_bytes,
        reg64_t reg_data, int offset, reg64_t reg_data_stride, reg64_t reg_buf,
        data_type_t dt) {
    const int num_cols_ele = tile_num_col_bytes / 2; // 32 for full tile
    const int num_N = num_cols_ele / 2; // 16 for full tile
    const auto zmm_2 = zmm_tmp_2();

    assert(num_N > 0 && "bad tile parameters");
    MAYBE_UNUSED(num_N);

    const auto rd_block = bi.rdi->block(0);
    const auto reg_data_aux = reg_tmp_gpr;
    lea(reg_data_aux, ptr[reg_data + offset]);

    const int vnni_granularity = 2;
    const int r_end = utils::div_up(rd_block, vnni_granularity);
    assert(r_end <= num_rows && "bad tile parameters");

    if (dt == data_type::f8_e5m2)
        f8_e5m2_emulator_->vcvt_f8_to_f16_vnni_block(
                r_end, reg_data_aux, reg_data_stride, reg_buf);
    else if (dt == data_type::f8_e4m3)
        f8_e4m3_emulator_->vcvt_f8_to_f16_vnni_block(
                r_end, reg_data_aux, reg_data_stride, reg_buf);
    else
        assert(!"unsupported data type");

    // zero rest of the tile data
    if (r_end < num_rows) {
        vpxord(zmm_2, zmm_2, zmm_2);
        for (int r = r_end; r < num_rows; ++r)
            vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_2);
    }
}

// This method down-converts and transforms the data from f32 to bf16_vnni
// format. Generally used by matrix_B.
void jit_brgemm_amx_uker_base_t::bf32_downconvert_to_vnni(
        brgemm_iteration_t &bi, int num_rows, int tile_num_col_bytes,
        reg64_t reg_data, int offset, reg64_t reg_data_stride,
        reg64_t reg_buf) {
    const auto num_cols_ele = tile_num_col_bytes / sizeof(bfloat16_t);
    const auto num_N = num_cols_ele / sizeof(bfloat16_t);
    const auto col_tail = num_N % simd_w;
    const auto zmm_1 = zmm_tmp_1();
    const auto zmm_2 = zmm_tmp_2();

    assert(num_N > 0);

    auto load = [&](Zmm zmm, Address addr) {
        if (col_tail)
            vmovups(zmm | fp_col_mask | T_z, addr);
        else
            vmovups(zmm, addr);
    };

    if (col_tail) {
        const auto tail_mask = (static_cast<size_t>(1) << col_tail) - 1;
        mov(reg_tmp_gpr, tail_mask);
        kmovq(fp_col_mask, reg_tmp_gpr);
    }

    // Note: using the same register used in col_tail, so order is important
    const auto reg_data_aux = reg_tmp_gpr;
    lea(reg_data_aux, ptr[reg_data + offset]);

    const auto rd_block = bi.rdi->block(0);
    const int vnni_granularity
            = data_type_vnni_granularity(data_type_t::dnnl_bf16);
    const auto r_end
            = nstl::min(utils::div_up(rd_block, vnni_granularity), num_rows);

    for (int r = 0; r < r_end; ++r) {
        load(zmm_1, ptr[reg_data_aux]);

        if (r * vnni_granularity + 1 >= rd_block) {
            vpxord(zmm_2, zmm_2, zmm_2);
        } else {
            load(zmm_2, ptr[reg_data_aux + reg_data_stride]);
        }

        vcvtne2ps2bf16(zmm_1, zmm_2, zmm_1);
        vpermw(zmm_1, zmm_bf32_permute, zmm_1);
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

    const auto &tloop = imap_[bi.apply_postops];
    auto should_save_transform = [&](matrix_kind_t mk) {
        // For fp8 via conversion we use temporal buffer heavily for conversion.
        // Therefore saved data may be overwritten
        // TODO: remove this restriction
        if (brg.is_fp8_via_convert()) return false;
        // save if there is a reuse
        if (mk == matrix_A) {
            return tloop.ldis.size() > 1;
        } else {
            return tloop.bdis.size() > 1;
        }
    };

    const auto dt = mk == matrix_A ? brg.dt_a : brg.dt_b;

    const bool is_A = mk == matrix_A;
    auto &transform_buf = is_A ? transform_buf_map_A_ : transform_buf_map_B_;

    const auto transform_offset
            = use_ils_ ? brg.get_num_C_tiles() * brgemm_desc_t::tilesize : 0;
    const auto max_bdb2 = tloop.bdis[0].block2();
    const auto max_rdb = tloop.rdis.size();
    const auto matrix_a_offset = transform_offset;
    const auto matrix_b_offset = transform_offset
            + brgemm_desc_t::tilesize
                    * (nstl::max<int>(should_save_transform(mk),
                            should_save_transform(matrix_A) * brg.brgattr.max_bs
                                    * max_bdb2 * max_rdb));
    const auto matrix_offset = is_A ? matrix_a_offset : matrix_b_offset;
    const std::string key
            = std::to_string(bi.bsi->pos) + "_" + std::to_string(offset);

    if (transform_buf.find(key) != transform_buf.end()) {
        auto buf_idx = transform_buf[key];
        auto offt = matrix_offset + buf_idx * brgemm_desc_t::tilesize;
        tileloadd(t1, ptr[reg_buf + reg_converted_stride + offt]);
        return;
    }

    auto buf_offt = matrix_offset;
    // save offset of the transformation if required.
    if (should_save_transform(mk)) {
        auto buf_idx = transform_buf.size();
        buf_offt = matrix_offset + buf_idx * brgemm_desc_t::tilesize;
        transform_buf[key] = buf_idx;
    }

    if (buf_offt) add(reg_buf, buf_offt);
    mov(reg_converted_stride, zmm_width_in_bytes);

    const int max_tiles = amx::get_max_tiles(amx::get_target_palette());
    JIT_ASSERT(t1.getIdx() >= 0 && t1.getIdx() < max_tiles);
    const auto num_rows = palette_.rows[t1.getIdx()];
    const auto num_col_bytes = palette_.cols[t1.getIdx()];
    if (is_A) {
        if (brg.is_bf32)
            bf32_downconvert(bi, num_rows, num_col_bytes, reg_base, offset,
                    reg_stride, reg_buf);
        else
            fp8_to_f16_upconvert(bi, num_rows, num_col_bytes, reg_base, offset,
                    reg_stride, reg_buf, dt);
    } else {
        if (brg.is_bf32)
            bf32_downconvert_to_vnni(bi, num_rows, num_col_bytes, reg_base,
                    offset, reg_stride, reg_buf);
        else
            fp8_to_f16_upconvert_to_vnni(bi, num_rows, num_col_bytes, reg_base,
                    offset, reg_stride, reg_buf, dt);
    }

    // load into tmm from the transformed data.
    tileloadd(t1, ptr[reg_buf + reg_converted_stride]);

    // reset buf pointer.
    if (buf_offt) sub(reg_buf, buf_offt);
}

bool jit_brgemm_amx_uker_base_t::maybe_pre_process_k_tail(
        brgemm_iteration_t &bi, int bdb, const Tmm &t1, reg64_t reg_base,
        size_t offset, reg64_t reg_stride, matrix_kind_t mk) {
    const auto &tloop = imap_[bi.apply_postops];

    const auto need_k_tail_processing = mk == matrix_A && brg.amx_wary_k_tail()
            && brg.rdb_tail != 0 && bi.bdi->idx == tloop.bdis.size() - 1
            && bdb == bi.bdi->block2() - 1 && bi.last_bsi
            && tloop.is_last_rdi(bi.rdi);

    if (!need_k_tail_processing) return false;

    auto transform_offset = brg.get_num_C_tiles() * brgemm_desc_t::tilesize
            + brg.get_convert_wsp_buffer_size();

    if (transform_offset) add(reg_buf, transform_offset);
    mov(reg_converted_stride, zmm_width_in_bytes);

    // reuse transformed data from matrix A for ldi > 0
    if (bi.ldi->idx == 0) {
        const auto num_rows = palette_.rows[t1.getIdx()];
        const auto num_col_bytes = palette_.cols[t1.getIdx()];

        const auto max_num_cols
                = nstl::min<int>(num_col_bytes / brg.typesize_A, brg.rdb_tail);
        const size_t col_tail
                = max_num_cols % (zmm_width_in_bytes / brg.typesize_A);
        if (col_tail) {
            const auto tail_mask = (static_cast<size_t>(1) << col_tail) - 1;
            mov(reg_tmp_gpr, tail_mask);
            kmovq(rd_tail_mask, reg_tmp_gpr);
        }
        auto zmm_1 = zmm_tmp_1();
        auto zmm_1_masked = col_tail ? zmm_1 | rd_tail_mask | T_z : zmm_1;

        assert(max_num_cols > 0);

        const auto reg_data_aux = reg_tmp_gpr;
        lea(reg_data_aux, ptr[reg_base + offset]);

        for (int r = 0; r < num_rows; ++r) {
            switch (brg.dt_a) {
                case data_type::bf16:
                case data_type::f16:
                    vmovdqu16(zmm_1_masked, ptr[reg_data_aux]);
                    break;
                case data_type::f8_e5m2:
                case data_type::f8_e4m3:
                case data_type::s8:
                case data_type::u8:
                    vmovdqu8(zmm_1_masked, ptr[reg_data_aux]);
                    break;
                default: assert(!"unsupported data type");
            }
            vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_1);
            add(reg_data_aux, reg_stride);
        }
    }
    // load into tmm from the transformed data.
    tileloadd(t1, ptr[reg_buf + reg_converted_stride]);

    // reset buf pointer
    if (transform_offset) sub(reg_buf, transform_offset);
    return true;
}

void jit_brgemm_amx_uker_base_t::gemm_microkernel_amx(brgemm_iteration_t &bi) {
    prf0A.reset();
    prf1A.reset();
    prf2A.reset();
    prfntaA.reset();
    prf0B.reset();
    prf1B.reset();
    prf2B.reset();
    prfntaB.reset();

    const auto store_by_vectors = get_store_by_vectors(bi.apply_postops);

    bool do_post_tilestore = (brg.interleave_tilestores_ && bi.last_bsi
            && imap_[bi.apply_postops].is_last_rdi(bi.rdi));

    bool do_pre_tilestore = (brg.interleave_tilestores_ && bi.first_bsi
            && bi.rdi->pos(0) == 0 && was_prev_bi_);

    if (store_by_vectors)
        mov(reg_stride_ld_block, ld_block_C_size_);
    else
        mov(reg_stride_ld_block, LDC_size_);

    for (int bdb = 0; bdb < bi.bdi->block2(); bdb++) {
        maybe_tileloadd_nt(bi, matrix_kind_t::matrix_A, bdb, A_offset(bi, bdb));
        for (int ldb = 0; ldb < bi.ldi->block2(); ldb++) {
            if (bdb == 0)
                maybe_tileloadd_nt(
                        bi, matrix_kind_t::matrix_B, ldb, B_offset(bi, ldb));
            if (ldb == 0) {
                if (bdb > 0)
                    tdpbxxd(bi, bdb - 1, bi.ldi->block2() - 1, do_pre_tilestore,
                            do_post_tilestore);
            } else
                tdpbxxd(bi, bdb, ldb - 1, do_pre_tilestore, do_post_tilestore);
        }
    }
    // last tdpbxxd
    tdpbxxd(bi, bi.bdi->block2() - 1, bi.ldi->block2() - 1, do_pre_tilestore,
            do_post_tilestore);
}

void jit_brgemm_amx_uker_base_t::rdb_loop(brgemm_iteration_t &bi) {
    const auto &tloop = imap_[bi.apply_postops];
    for (auto &rdi : tloop.rdis) {
        bi.rdi = &rdi;
        gemm_microkernel_amx(bi);
    }
}

void jit_brgemm_amx_uker_base_t::bs_loop_body(brgemm_iteration_t &bi) {
    if (brg.brgattr.var_bs) {
        set_A_B_matrices();
        add(reg_aux1_batch, sizeof(brgemm_batch_element_t));
        prefetcht0(ptr[reg_aux1_batch]);
    } else {
        set_A_B_matrices(bi.bsi->pos);
    }

    rdb_loop(bi);
}

void jit_brgemm_amx_uker_base_t::bs_loop(brgemm_iteration_t &bi) {
    if (ununroll_bd_loop && bi.bdi->similar != nullptr) {
        // there is code for this iteration already, so we need to store
        // prev_bi_ only
        prev_bi_ = bi;
        was_prev_bi_ = true;
        return;
    }

    const auto &tloop = imap_[bi.apply_postops];
    if (ununroll_bd_loop && was_prev_bi_) {
        if (bi.bdi->idx != prev_bi_.bdi->idx) add(reg_A, bi.bdi->A_shift);

        const auto real_ils
                = actual_ils(bi.apply_postops, bi.skip_accumulation);

        brgemm_iteration_t *bi_shift = nullptr;
        if (!real_ils && bi.bdi->idx != prev_bi_.bdi->idx)
            bi_shift = &bi;
        else if (real_ils && prev_bi_.bdi->idx > 0 && prev_bi_.ldi->idx == 0)
            bi_shift = &prev_bi_;
        if (bi_shift != nullptr) {
            add(reg_C, bi_shift->bdi->C_shift);
            add(reg_D, bi_shift->bdi->D_shift);
            if (brg.req_comp_pads_with_bcast)
                add(reg_zp_comp_pad_a, bi_shift->bdi->zp_comp_pad_a_shift);
        }
    }

    if (bi.skip_accumulation) {
        store_accumulators(bi);
        return;
    }

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

            bi.bsi = &(tloop.bsis[0]);
            // only one BS iteration: first and last
            bi.first_bsi = true;
            bi.last_bsi = true;
            bs_loop_body(bi);
            jmp(end_BS_loop_label, T_NEAR);

            // first BS iteration
            L_aligned(first_BS_loop_label, 64);
            bi.first_bsi = true;
            bi.last_bsi = false;
            bs_loop_body(bi);

            dec(reg_BS_loop);
            cmp(reg_BS_loop, 1);
            je(last_BS_loop_label, T_NEAR);

            // middle BS iterations
            L_aligned(BS_loop_label, 64);
            {
                bi.first_bsi = false;
                bi.last_bsi = false;
                bs_loop_body(bi);
                dec(reg_BS_loop);
                cmp(reg_BS_loop, 1);
                jg(BS_loop_label, T_NEAR);
            }
            // last BS iteration
            L_aligned(last_BS_loop_label, 64);
            bi.first_bsi = false;
            bi.last_bsi = true;
            bs_loop_body(bi);

            L_aligned(end_BS_loop_label, 64);
        }
        store_accumulators(bi);
    } else {
        if (brg.alpha != 0.f) {
            for (int bs = 0; bs < brg.brgattr.max_bs; bs++) {
                bi.bsi = &(tloop.bsis[bs]);
                bi.first_bsi = bi.bsi->is_first;
                bi.last_bsi = bi.bsi->is_last;
                bs_loop_body(bi);
            }
        }
        store_accumulators(bi);
    }
}

void jit_brgemm_amx_uker_base_t::ldb_loop_body(brgemm_iteration_t &bi) {
    if (brg.innermost_loop == brgemm_bd_loop_innermost)
        bdb_loop(bi);
    else if (brg.innermost_loop == brgemm_ld_loop_innermost)
        bs_loop(bi);
    else
        assert(!"Unknown loop order!");
}

void jit_brgemm_amx_uker_base_t::ldb_loop(brgemm_iteration_t &bi) {
    // clear the transform cache for A, as the existing data is invalid as
    // we move to next bdb2 block.
    const auto &tloop = imap_[bi.apply_postops];
    transform_buf_map_A_.clear();
    for (auto &ldi : tloop.ldis) {
        bi.ldi = &ldi;
        ldb_loop_body(bi);
    }
}

jit_brgemm_amx_uker_base_t::bd_iteration_t *
jit_brgemm_amx_uker_base_t::find_similar(
        const bd_iteration_t *bdi, bool apply_postops) {
    auto &tloop = imap_[apply_postops];
    const auto cidx = bdi->idx;
    // if wary_k_tail is true then last iteration is unique
    if (brg.amx_wary_k_tail() && cidx == tloop.bdis.size() - 1) return nullptr;

    for (size_t i = (actual_ils(apply_postops) ? 1 : 0); i < cidx; i++) {
        if (*bdi == tloop.bdis[i]
                && IMPLICATION(actual_ils(apply_postops),
                        tloop.bdis[cidx - 1] == tloop.bdis[i - 1])) {
            tloop.duplicated++;
            return &(tloop.bdis[i]);
        }
    }

    return nullptr;
}

void jit_brgemm_amx_uker_base_t::bdb_loop_body(brgemm_iteration_t &bi) {
    auto &tloop = imap_[bi.apply_postops];
    if (ununroll_bd_loop) {
        const auto cidx = bi.bdi->idx;
        if (bi.bdi->similar) {
            tloop.bdis[cidx].lstart = bi.bdi->similar->lstart;
        } else {
            align(64);
            L(tloop.bdis[cidx].lstart);
            mov(reg_iter_labels_list, ptr[rsp + reg_iter_labels_list_offs_]);
            mov(reg_iter_label, ptr[reg_iter_labels_list]);
            add(reg_iter_labels_list, 8);
            mov(ptr[rsp + reg_iter_labels_list_offs_], reg_iter_labels_list);
        }
    }

    if (brg.innermost_loop == brgemm_ld_loop_innermost)
        ldb_loop(bi);
    else if (brg.innermost_loop == brgemm_bd_loop_innermost)
        bs_loop(bi);
    else
        assert(!"Unknown loop order!");
    if (ununroll_bd_loop) { jmp(reg_iter_label); }
};

void jit_brgemm_amx_uker_base_t::bdb_loop(brgemm_iteration_t &bi) {
    const auto &tloop = imap_[bi.apply_postops];
    Label iteration_pointers;
    if (ununroll_bd_loop) {
        lea(reg_iter_labels_list, ptr[rip + iteration_pointers]);
        // shift to load address for jmp for next iteration
        add(reg_iter_labels_list, 8);
        mov(ptr[rsp + reg_iter_labels_list_offs_], reg_iter_labels_list);
    }

    for (auto &bdi : tloop.bdis) {
        bi.bdi = &bdi;
        bdb_loop_body(bi);
    }
    if (ununroll_bd_loop) {
        Label loop_end;
        jmp(loop_end, T_NEAR); //just skip list of iteration labels

        align(64);
        L(iteration_pointers);
        for (const auto &bdi : tloop.bdis) {
            putL(bdi.lstart);
        }
        putL(loop_end);
        L(loop_end);
    }
}

void jit_brgemm_amx_uker_base_t::top_loop(brgemm_iteration_t &bi) {
    mov(reg_C, ptr[param1 + GET_OFF(ptr_C)]);
    mov(reg_D, ptr[param1 + GET_OFF(ptr_D)]);
    init(bi);
    if (brg.innermost_loop == brgemm_ld_loop_innermost)
        bdb_loop(bi);
    else if (brg.innermost_loop == brgemm_bd_loop_innermost)
        ldb_loop(bi);
    else
        assert(!"Unknown loop order!");

    // bi is last iteration now
    if (brg.interleave_tilestores_) {
        prev_bi_ = bi;
        was_prev_bi_ = true;
        for_(int bdb = 0; bdb < prev_bi_.bdi->block2(); bdb++)
        for (int ldb = 0; ldb < prev_bi_.ldi->block2(); ldb++) {
            maybe_tilestore(prev_bi_, bdb, ldb, true, false);
        }
    }

    const auto &tloop = imap_[bi.apply_postops];
    if (actual_ils(bi.apply_postops, bi.skip_accumulation) && ununroll_bd_loop
            && tloop.ldis.size() == 1) {
        // update reg_C and reg_D if they they were not updated yet
        add(reg_C, bi.bdi->C_shift);
        add(reg_D, bi.bdi->D_shift);
        if (brg.req_comp_pads_with_bcast)
            add(reg_zp_comp_pad_a, bi.bdi->zp_comp_pad_a_shift);
    }
    interleave_store(bi, true);
}

void jit_brgemm_amx_uker_base_t::fill_imap() {
    for (bool apply_postops : {false, true}) {
        auto &tloop = imap_[apply_postops];

        tloop.bdis.clear();
        tloop.ldis.clear();
        tloop.rdis.clear();
        tloop.bsis.clear();
        tloop.bdis.reserve(brg.bdb2);
        tloop.ldis.reserve(brg.ldb2);
        tloop.rdis.reserve(brg.rdb);
        tloop.bsis.reserve(brg.brgattr.max_bs);

        auto bdi_pos = skipped_bd_mask(0);
        bd_iteration_t bdi;
        bdi.blocks.reserve(brg.bd_block2);
        for (int bdb = 0; bdb < brg.bdb; bdb += brg.bd_block2) {
            bdi.blocks.clear();
            for (int ibdb = 0; ibdb < brg.bd_block2; ibdb++) {
                auto abdb = bdb + ibdb;
                if (abdb >= brg.bdb) break;
                if (brg.bdb_tail && abdb == brg.bdb - 1)
                    bdi.blocks.emplace_back(bdi_pos, brg.bdb_tail, true);
                else
                    bdi.blocks.emplace_back(bdi_pos, brg.bd_block, false);
                bdi_pos += brg.bd_block;
                if (bdi_pos >= brg.bcast_dim) break;
                bdi_pos = skipped_bd_mask(bdi_pos);
            }
            bdi.idx = tloop.bdis.size();

            if (brg.brgattr.bd_mask_level > 0) {
                const auto lidx = bdi.blocks.size() - 1;
                const auto bdm_sz = bdi.rel_pos(lidx) + bdi.blocks[lidx].block;
                bdi.bd_mask.resize(bdm_sz);
                bdi.adj_bd_mask.resize(bdm_sz);
                for (size_t i = 0; i < bdm_sz; i++) {
                    bdi.bd_mask[i] = bd_mask_buffer_ptr_[bdi.pos(0) + i];
                    bdi.adj_bd_mask[i] = adj_bd_mask_buffer_[bdi.pos(0) + i];
                }
            }

            if (ununroll_bd_loop && bdi.idx > 0) {
                const auto prev_bdi = &tloop.bdis[bdi.idx - 1];
                const auto inp_shift = (bdi.pos(0) - prev_bdi->pos(0));
                bdi.A_shift = inp_shift * LDA2_size_;

                const auto out_shift
                        = (get_out_bd(&bdi, 0, 0) - get_out_bd(prev_bdi, 0, 0));
                bdi.C_shift = out_shift * LDC2_size_M_;
                bdi.D_shift = out_shift * LDD_size_;
                bdi.zp_comp_pad_a_shift = out_shift * brg.LDB * sizeof(int32_t);
            }
            tloop.bdis.push_back(bdi);
        }

        size_t ldi_pos = 0;
        dim_iteration_t ldi;
        ldi.blocks.reserve(brg.ld_block2);
        for (int ldb = 0; ldb < brg.ldb; ldb += brg.ld_block2) {
            ldi.blocks.clear();
            for (int ildb = 0; ildb < brg.ld_block2; ildb++) {
                auto aldb = ldb + ildb;
                if (aldb >= brg.ldb) break;
                if (brg.ldb_tail && aldb == brg.ldb - 1)
                    ldi.blocks.emplace_back(ldi_pos, brg.ldb_tail, true);
                else
                    ldi.blocks.emplace_back(ldi_pos, brg.ld_block, false);
                ldi_pos++;
            }
            ldi.idx = tloop.ldis.size();
            tloop.ldis.push_back(ldi);
        }

        size_t rdi_pos = 0;
        dim_iteration_t rdi;
        rdi.blocks.reserve(1);
        for (int rdb = 0; rdb < brg.rdb; rdb++) {
            rdi.blocks.clear();
            rdi.blocks.emplace_back(rdi_pos, brg.rd_block);
            rdi.idx = tloop.rdis.size();
            tloop.rdis.push_back(rdi);
            rdi_pos++;
        }
        if (brg.rdb_tail > 0) {
            rdi.blocks.clear();
            rdi.blocks.emplace_back(rdi_pos, brg.rdb_tail, true);
            rdi.idx = tloop.rdis.size();
            tloop.rdis.push_back(rdi);
        }

        bs_iteration_t bsi;
        for (int bs = 0; bs < brg.brgattr.max_bs; bs++) {
            bsi.pos = bs;
            bsi.is_first = (bs == 0);
            bsi.is_last = (bs == brg.brgattr.max_bs - 1);
            bsi.idx = tloop.bsis.size();
            tloop.bsis.push_back(bsi);
        }

        if (ununroll_bd_loop) {
            for (size_t ibdi = 0; ibdi < tloop.bdis.size(); ibdi++) {
                tloop.bdis[ibdi].similar
                        = find_similar(&(tloop.bdis[ibdi]), apply_postops);
            }
        }
    }
}

void jit_brgemm_amx_uker_base_t::init(brgemm_iteration_t &bi) {
    was_prev_bi_ = false;
    const auto bdb2_to_unroll = nstl::max(0,
            brg.bdb2
                    - (actual_ils(bi.apply_postops, bi.skip_accumulation) ? 1
                                                                          : 0));
    ununroll_bd_loop = brg.brgattr.hint_ununroll_bd_loop && bdb2_to_unroll > 1
            && (brg.innermost_loop == brgemm_ld_loop_innermost || brg.ldb2 == 1)
            && get_store_by_vectors(bi.apply_postops)
            && IMPLICATION(!bi.skip_accumulation,
                    (brg.brgattr.max_bs == 1 || brg.type == brgemm_static_offs)
                            && !brg.brgattr.var_bs);
    if (brg.type == brgemm_static_offs && !bi.skip_accumulation) {
        if (brg.layout == brgemm_row_major) {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_A)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_B)]);
        } else {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_B)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_A)]);
        }
    } else if (brg.brgattr.max_bs == 1 && !bi.skip_accumulation) {
        assert(one_of(brg.type, brgemm_addr, brgemm_offs));
        if (brg.type == brgemm_addr) {
            if (brg.layout == brgemm_row_major) {
                mov(reg_A,
                        EVEX_compress_addr(
                                reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
                mov(reg_B,
                        EVEX_compress_addr(
                                reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
            } else {
                mov(reg_A,
                        EVEX_compress_addr(
                                reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
                mov(reg_B,
                        EVEX_compress_addr(
                                reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
            }
        } else if (brg.type == brgemm_offs) {
            if (brg.layout == brgemm_row_major) {
                mov(reg_A, ptr[param1 + GET_OFF(ptr_A)]);
                mov(reg_B, ptr[param1 + GET_OFF(ptr_B)]);
            } else {
                mov(reg_A, ptr[param1 + GET_OFF(ptr_B)]);
                mov(reg_B, ptr[param1 + GET_OFF(ptr_A)]);
            }
        }
    }

    fill_imap();

    // for many primitives which use brgemm the brg.ldb2 is equal or less than 1
    // so we can read post ops data only once per brgemm call

    if (brg.ldb2 > 1) {
        prepare_post_ops_registers_once_ = false;
    } else if (brg.ldb2 == 1) {
        if (brg.ldb2_tail == 0 && brg.ldb_tail == 0) {
            prepare_post_ops_registers_once_ = true;
            bi.ldi = &(imap_[true].ldis[0]);
            prepare_post_ops_registers(bi);
        }
    } else if (brg.ldb2_tail > 0) {
        if (brg.ldb_tail == 0) {
            prepare_post_ops_registers_once_ = true;
            bi.ldi = &(imap_[true].ldis[0]);
            prepare_post_ops_registers(bi);
        }
    } else {
        prepare_post_ops_registers_once_ = true;
        bi.ldi = &(imap_[true].ldis[0]);
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

    if (bi.skip_accumulation) return;
    prf0A.set(brgemm_prf0, brg.prfA.dist0);
    prf1A.set(brgemm_prf1, brg.prfA.dist1);
    prf2A.set(brgemm_prf2, brg.prfA.dist2);
    prfntaA.set(brgemm_prfNTA, brg.prfA.distNTA);

    prf0B.set(brgemm_prf0, brg.prfB.dist0);
    prf1B.set(brgemm_prf1, brg.prfB.dist1);
    prf2B.set(brgemm_prf2, brg.prfB.dist2);
    prfntaB.set(brgemm_prfNTA, brg.prfB.distNTA);

    prf0C.set(brgemm_prf0, brg.prfC.dist0);
    prf1C.set(brgemm_prf1, brg.prfC.dist1);
}

void jit_brgemm_amx_uker_base_t::generate() {
    preamble();

    sub(rsp, stack_space_needed_);

    const auto full_mask = size_t {0xffffffffffffffff};
    const auto tail_mask = size_t((1 << brg.ldb_tail) - 1);
    reg64_t reg_mask = rbx;

    mov(reg_mask, full_mask);
    kmovq(ld_full_mask, reg_mask);
    mov(reg_mask, tail_mask);
    kmovq(ld_tail_mask, reg_mask);

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
    may_load_accumulators_ = one_of(brg.alpha, 0, 1) && brg.beta == 1.f
            && brg.dt_c == brg.dt_d
            && IMPLICATION(brg.is_input_convert(), brg.is_fp8_via_convert())
            && IMPLICATION(
                    brg.is_f32 || brg.is_bf16, brg.dt_c == data_type::f32)
            && IMPLICATION(brg.is_int8, brg.dt_c == data_type::s32)
            && brg.brgattr.bd_mask_level == 0;
    need_to_apply_alpha_beta_
            = (brg.beta != 0.f && !may_load_accumulators_) || brg.alpha != 1.f;
    are_post_ops_applicable_ = brg.are_post_ops_applicable();

    // second level blocking eligible only if we don't use store by vectors for now
    assert(IMPLICATION(are_post_ops_applicable_ || need_to_apply_alpha_beta_
                    || brg.brgattr.bd_mask_level,
            !brg.is_blocked && !brg.brgattr.var_bs));
    assert(IMPLICATION(brg.brgattr.var_bs,
            IMPLICATION(brg.is_input_convert(), brg.is_fp8_via_convert())));
    read_params();
    prepare_bd_mask();

    Label permute_index_table;
    if (brg.is_input_convert() || brg.amx_wary_k_tail()) {
        // save tiles description for later use
        brgemm_init_tiles(brg, (char *)(&palette_));
        // load permute indices
        if (brg.is_bf32)
            vmovups(zmm_bf32_permute, ptr[rip + permute_index_table]);
    }

    mov(reg_stride_lda, lda());
    mov(reg_stride_ldb, ldb());

    bool non_postops_generate
            = !are_post_ops_applicable_ || !brg.brgattr.postops_only;
    brgemm_iteration_t bi;

    Label label_to_ret;
    if (are_post_ops_applicable_) {
        Label label_store_without_post_ops;
        mov(reg_do_post_ops, ptr[param1 + GET_OFF(do_post_ops)]);
        cmp(reg_do_post_ops, 0);
        jz(label_store_without_post_ops, T_NEAR);
        bi.apply_postops = true;
        if (brg.brgattr.generate_skip_accumulation) {
            brgemm_iteration_t bi1;
            mov(reg_do_skip_accum, ptr[param1 + GET_OFF(skip_accm)]);
            cmp(reg_do_skip_accum, 0);
            Label label_do_not_skip_acc;
            jz(label_do_not_skip_acc, T_NEAR);

            bi1.skip_accumulation = true;
            bi1.apply_postops = true;
            top_loop(bi1);
            jmp(label_to_ret, T_NEAR);

            L(label_do_not_skip_acc);
        }
        top_loop(bi);
        if (non_postops_generate) jmp(label_to_ret, T_NEAR);
        transform_buf_map_A_.clear();
        transform_buf_map_B_.clear();
        L(label_store_without_post_ops);
    }
    if (non_postops_generate) {
        bi.apply_postops = false;
        top_loop(bi);
    }
    L(label_to_ret);

    add(rsp, stack_space_needed_);

    postamble();

    if (brg.with_eltwise)
        postops_injector_->prepare_table(/* generate = */ true);

    if (brg.is_fp8_via_convert()) {
        if (f8_e5m2_emulator_) f8_e5m2_emulator_->prepare_table();
        if (f8_e4m3_emulator_) f8_e4m3_emulator_->prepare_table();
    }

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

brgemm_amx_uker_t::brgemm_amx_uker_t(const brgemm_desc_t &abrd)
    : brgemm_kernel_(new jit_brgemm_amx_uker_base_t(abrd)) {}

status_t brgemm_amx_uker_t::create_kernel() {
    return brgemm_kernel_->create_kernel();
}

void brgemm_amx_uker_t::operator()(brgemm_kernel_params_t *params) const {
    (*brgemm_kernel_)(params);
}

const jit_generator *brgemm_amx_uker_t::get_jit_generator() const {
    return brgemm_kernel_;
}

brgemm_amx_uker_t::~brgemm_amx_uker_t() {
    delete brgemm_kernel_;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
