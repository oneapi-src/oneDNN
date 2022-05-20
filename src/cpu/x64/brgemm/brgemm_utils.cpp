/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "cpu/x64/brgemm/brgemm_utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;

enum {
    decomposition_2x2 = 101,
    decomposition_3x1_3,
    decomposition_3x1_2,
    undefined,
};

impl::data_type_t get_accum_datatype(brgemm_t *brg) {
    // this assert should check if 'init_kernel_datatype()' was previously
    // called.
    assert(brg->is_int8 || brg->is_bf16 || brg->is_f32);
    return brg->is_int8 ? data_type::s32 : data_type::f32;
}

void init_kernel_datatype(
        brgemm_t *brg, impl::data_type_t dt_a, impl::data_type_t dt_b) {
    assert(dt_a != data_type::undef && dt_b != data_type::undef);
    brg->is_int8 = utils::one_of(dt_a, data_type::u8, data_type::s8)
            && (dt_b == data_type::s8);
    brg->is_bf16 = (dt_a == data_type::bf16) && (dt_b == data_type::bf16);
    brg->is_f32 = (dt_a == data_type::f32) && (dt_b == data_type::f32);
    assert(brg->is_int8 || brg->is_bf16 || brg->is_f32);
}

void init_common_conf(brgemm_t *brg, brgemm_batch_kind_t type, float alpha,
        float beta, const brgemm_strides_t *strides) {
    brg->beta = beta;
    brg->alpha = alpha;
    brg->type = type;
    brg->with_bias = false;
    brg->with_eltwise = false;
    brg->with_sum = false;
    brg->sum_scale = 0;
    brg->sum_zp = 0;
    brg->with_scales = false;

    if (strides != nullptr) {
        brg->stride_a = strides->stride_a;
        brg->stride_b = strides->stride_b;
    } else {
        brg->stride_a = brg->stride_b = 0;
    }
}

namespace brgemm_utils {

bool can_dispatch_uker(const brgemm_t *brg) {
    return brg->is_amx && brg->type == brgemm_addr && brg->brgattr.max_bs >= 1
            && brg->brgattr.use_uker
            && !brg->brgattr.generate_skip_accumulation;
}

void maybe_try_bf32(brgemm_t *brg) {
    const bool try_bf32 = brg->is_f32
            && brg->brgattr.fpmath_mode == fpmath_mode::bf16
            && mayiuse(avx512_core_bf16_amx_bf16);
    if (try_bf32) {
        const bool is_amx = brg->is_amx;
        brg->is_amx = true;
        if (can_dispatch_uker(brg) /*Requires is_amx to be true*/) {
            brg->is_bf32 = true;
        } else {
            brg->is_bf32 = false;
            //  Restore
            brg->is_amx = is_amx;
        }
    }
}

status_t brgemm_blocking(brgemm_t *brg) {

    if (!brg->is_amx) {
        brg->ld_block = 16;
        brg->ldb = brg->load_dim / brg->ld_block;
        brg->ldb_tail = brg->load_dim % brg->ld_block;

        brg->ld_block2 = 4; // (M < 9) ? 2 : 4 | TODO - fix this for INT8
        brg->ldb2 = brg->ldb / brg->ld_block2;
        brg->ldb2_tail = brg->ldb % brg->ld_block2;

        if (brg->ldb2 == 0) brg->ld_block2 = nstl::max(1, brg->ldb2_tail);
        brg->embd_bcst = !brg->is_int8 && !brg->is_bf16
                && (brg->ldb2_tail <= 1 && brg->ldb2 == 0);

        int ld_block = (brg->ldb2 != 0) ? brg->ld_block2 : brg->ldb2_tail;
        int adj_ld_block = (ld_block == 0) ? (ld_block + 1) : ld_block;

        const int max_avx512_regs = 32;
        const int max_bcst_regs = 1;
        const bool req_compensation = brg->req_s8s8_compensation
                || brg->zp_type_a != brgemm_broadcast_t::none;
        const bool req_zp_a_comp_pads
                = (brg->req_cal_comp_pads || brg->brgattr.max_top_vpad > 0
                          || brg->brgattr.max_bottom_vpad > 0)
                && brg->zp_type_a != brgemm_broadcast_t::none;
        int max_regs = max_avx512_regs - (adj_ld_block + max_bcst_regs);
        int max_block
                = (brg->embd_bcst ? 28
                                  : ((brg->beta == 1.f || brg->beta == 0.f)
                                                  ? max_regs
                                                  : max_regs - 1));
        max_block -= req_compensation;
        max_block -= req_zp_a_comp_pads;
        if (req_zp_a_comp_pads) max_block = nstl::min(max_block, 27);
        if (brg->is_bf16_emu) max_block = nstl::min(max_block, 28);
        max_block /= adj_ld_block;
        int min_block = 1;
        float best_bd_block_eff = 0.f;
        brg->bd_block = 1;
        for (int bd_block = max_block; bd_block >= min_block; bd_block--) {
            const auto bd_block_disb = static_cast<float>(brg->bcast_dim)
                    / rnd_up(brg->bcast_dim, bd_block);
            const auto brgemm_microkernel_eff
                    = (static_cast<float>(adj_ld_block) * bd_block)
                    / (((adj_ld_block) + bd_block) * max_block);
            const auto bd_block_eff = bd_block_disb * brgemm_microkernel_eff;

            float block_foot_print = static_cast<float>(brg->typesize_A)
                    * (bd_block * brg->reduce_dim);
            if (block_foot_print <= static_cast<float>(
                        platform::get_per_core_cache_size(1))
                    && (bd_block_eff > best_bd_block_eff)) {
                brg->bd_block = bd_block;
                best_bd_block_eff = bd_block_eff;
            }
        }
        brg->bdb = brg->bcast_dim / brg->bd_block;
        brg->bdb_tail = brg->bcast_dim % brg->bd_block;

        brg->rd_block = 16 / brg->typesize_A;
        brg->rdb = brg->reduce_dim / brg->rd_block;
        brg->rdb_tail = brg->reduce_dim % brg->rd_block;

        brg->is_M_tail = false;
    } else {
        // Blocking configuration for AMX
        const int max_width = 16, min_width = 1;
        brg->ld_block = 16;
        brg->ldb = brg->load_dim / brg->ld_block;
        brg->ldb_tail = brg->load_dim % brg->ld_block;

        auto find_bd_block_for_bd_mask = [&]() {
            const auto bd_mask_size = brg->bcast_dim;
            if (brg->brgattr.bd_mask_level != 2 || bd_mask_size == 0)
                return false;

            const auto sm_buffer = brg->brgattr.bd_mask;
            auto min_bdb = INT_MAX;
            const auto start_bd_block = nstl::min(max_width, brg->bcast_dim);
            auto best_bd_block = start_bd_block;
            for (auto bd_block = start_bd_block; bd_block > 0; bd_block--) {
                auto bdb = 0;
                for (int i = 0; i < bd_mask_size;) {
                    if (brg->brgattr.bd_mask_level == 2 && sm_buffer[i] == 0) {
                        i++;
                    } else {
                        i += bd_block;
                        if (i > brg->bcast_dim) {
                            // bcast_dim not divided by bd_block
                            bdb = INT_MAX;
                        } else
                            bdb++;
                    }
                }
                if (bdb < min_bdb) {
                    min_bdb = bdb;
                    best_bd_block = bd_block;
                }
            }
            brg->bd_block = best_bd_block;
            brg->bdb_tail = 0;
            brg->bdb = min_bdb;
            return true;
        };

        auto set_decomposition_by_ld = [&]() {
            if (brg->bd_block2 == 1 && brg->ldb > 0 && brg->ldb_tail == 0) {
                if (brg->ldb % 3 == 0)
                    brg->ld_block2 = 3;
                else if (brg->ldb % 2 == 0)
                    brg->ld_block2 = 2;
                else
                    brg->ld_block2 = 1;
            } else {
                brg->ld_block2
                        = (brg->ldb > 0 && brg->ldb % 2 == 0
                                  && brg->ldb_tail == 0 && brg->bd_block2 < 3)
                        ? 2
                        : 1;
            }
            brg->ldb2 = brg->ldb / brg->ld_block2;
            brg->ldb2_tail = brg->ldb % brg->ld_block2;

            // Re-adjust the bd_block2 if possible
            if (brg->ld_block2 == 1 && !brg->is_M_tail && brg->ldb_tail == 0) {
                brg->bd_block2 = (brg->bdb >= 3) ? 3 : (brg->bdb >= 2) ? 2 : 1;
                brg->bdb2 = brg->bdb / brg->bd_block2;
                brg->bdb2_tail = (brg->bd_block2 == 1)
                        ? brg->bdb
                        : brg->bdb % brg->bd_block2;
            }
        };

        auto try_3x1_decomposition = [&](int width_step) {
            brg->is_M_tail = false;
            if (brg->bcast_dim > (width_step - 1) * max_width
                    && brg->bcast_dim < width_step * max_width
                    && brg->ldb_tail == 0) {
                if (!find_bd_block_for_bd_mask()) {
                    brg->bd_block = max_width;
                    brg->bdb = div_up(brg->bcast_dim, brg->bd_block);
                    brg->bdb_tail = brg->bcast_dim % brg->bd_block;
                    brg->is_M_tail = true;
                }
                brg->bd_block2 = width_step;
                brg->bdb2 = brg->bdb / brg->bd_block2;
                brg->bdb2_tail = brg->bdb % brg->bd_block2;
                set_decomposition_by_ld();
                return true;
            }
            return false;
        };

        auto try_2x2_decomposition = [&]() {
            if (!find_bd_block_for_bd_mask()) {
                for (int m_block = max_width; m_block >= min_width; m_block--) {
                    if (brg->bcast_dim % m_block == 0) {
                        brg->bd_block = m_block;
                        break;
                    }
                }
                if (brg->bd_block == 1) {
                    brg->bd_block = nstl::min(max_width, brg->bcast_dim);
                    brg->bdb_tail = brg->bcast_dim % max_width;
                    for (int i = max_width; i >= min_width; i--) {
                        int i_tail = brg->bcast_dim % i;
                        if (i_tail > brg->bdb_tail || i_tail == 0) {
                            brg->bd_block = i;
                            brg->bdb_tail = i_tail;
                            if (i_tail == 0) break;
                        }
                    }
                }
                brg->bdb = brg->bcast_dim / brg->bd_block;
                brg->bdb_tail = brg->bcast_dim % brg->bd_block;
            }

            brg->bd_block2 = (brg->bdb >= 2) ? 2 : 1;
            brg->bdb2 = brg->bdb / brg->bd_block2;
            brg->bdb2_tail = (brg->bd_block2 == 1) ? brg->bdb
                                                   : brg->bdb % brg->bd_block2;

            brg->is_M_tail = false;

            set_decomposition_by_ld();

            return !(brg->ld_block2 == 1 || brg->bd_block2 == 1
                    || brg->bd_block < 8);
        };

        bool is_decomposition_defined = false;
        for (int i = decomposition_2x2; i != undefined; i++) {
            switch (i) {
                case decomposition_2x2:
                    is_decomposition_defined = try_2x2_decomposition();
                    break;
                case decomposition_3x1_3:
                    is_decomposition_defined = try_3x1_decomposition(3);
                    break;
                case decomposition_3x1_2:
                    is_decomposition_defined = try_3x1_decomposition(2);
                    break;
                default: assert(!"invalid value"); break;
            };
            if (is_decomposition_defined) break;
        }
        if (!is_decomposition_defined) try_2x2_decomposition();

        brg->rd_block = (brg->is_bf16_amx || brg->is_bf32) ? 32 : 64;
        brg->rdb = brg->reduce_dim / brg->rd_block;
        brg->rdb_tail = brg->reduce_dim % brg->rd_block;

        // Remove these guard in the future (add tail processing by reduction dimension)
        if (!IMPLICATION(brg->rdb > 0 && brg->rdb_tail, brg->is_bf32))
            return status::unimplemented;
        if (!IMPLICATION((brg->rdb_tail % ((brg->is_bf16_amx) ? 2 : 4)) != 0,
                    brg->is_bf32))
            return status::unimplemented;
    }

    return status::success;
}

status_t brdgmm_blocking(brgemm_t *brg, const int max_zmm_accum) {

    constexpr int simd_w = 16;
    auto &M = brg->bcast_dim;
    auto &N = brg->load_dim;

    // In current implementation of dgmm, there is no reduce dim.
    auto &m_vlen_blk = brg->bd_block;
    auto &nb_m_vlen_blk = brg->bdb;
    auto &m_vlen_tail = brg->bdb_tail;
    auto &m_blocking = brg->bd_block2;
    auto &nb_m_blocking = brg->bdb2;
    auto &m_blocking_tail = brg->bdb2_tail;

    auto &n_vlen_blk = brg->ld_block;
    auto &nb_n_vlen_blk = brg->ldb;
    auto &n_vlen_tail = brg->ldb_tail;
    auto &n_blocking = brg->ld_block2;
    auto &nb_n_blocking = brg->ldb2;
    auto &n_blocking_tail = brg->ldb2_tail;

    // begin blocking
    n_vlen_blk = simd_w;
    nb_n_vlen_blk = div_up(N, n_vlen_blk);
    n_vlen_tail = N % n_vlen_blk;
    n_blocking = nstl::min(4, nb_n_vlen_blk);
    nb_n_blocking = div_up(nb_n_vlen_blk, n_blocking);
    n_blocking_tail = nb_n_vlen_blk % n_blocking;

    m_vlen_blk = 1;
    nb_m_vlen_blk = M / m_vlen_blk;
    m_vlen_tail = M % m_vlen_blk;
    m_blocking = nstl::min(nb_m_vlen_blk, max_zmm_accum / n_blocking);
    nb_m_blocking = div_up(nb_m_vlen_blk, m_blocking);
    m_blocking_tail = nb_m_vlen_blk % m_blocking;

    return status::success;
}

void init_brgemm_conf(brgemm_t *brg, cpu_isa_t isa, brgemm_batch_kind_t type,
        impl::data_type_t dt_a, impl::data_type_t dt_b, brgemm_layout_t layout,
        float alpha, float beta, dim_t LDA, dim_t LDB, dim_t LDC, dim_t M,
        dim_t N, dim_t K, const brgemm_strides_t *strides) {

    init_common_conf(brg, type, alpha, beta, strides);

    brg->layout = layout;

    brg->dt_a = brg->is_row_major() ? dt_a : dt_b;
    brg->dt_b = brg->is_row_major() ? dt_b : dt_a;
    init_kernel_datatype(brg, brg->dt_a, brg->dt_b);

    brg->dt_c = get_accum_datatype(brg);
    brg->dt_d = brg->dt_c;
    brg->dt_bias = brg->dt_c;

    brg->typesize_A = types::data_type_size(brg->dt_a);
    brg->typesize_B = types::data_type_size(brg->dt_b);
    brg->typesize_C = types::data_type_size(brg->dt_c);
    brg->typesize_D = types::data_type_size(brg->dt_d);

    brg->is_int8_amx = brg->is_int8 && mayiuse(avx512_core_bf16_amx_int8)
            && IMPLICATION(isa != isa_any, isa == avx512_core_bf16_amx_int8);
    brg->is_bf16_amx = brg->is_bf16 && mayiuse(avx512_core_bf16_amx_bf16)
            && IMPLICATION(isa != isa_any, isa == avx512_core_bf16_amx_bf16);
    brg->is_amx = (brg->is_int8_amx || brg->is_bf16_amx);

    brg->req_s8s8_compensation
            = brg->is_int8 && !brg->is_int8_amx && brg->dt_a == data_type::s8;

    brg->LDA = (brg->is_row_major()) ? static_cast<int>(LDA)
                                     : static_cast<int>(LDB);
    brg->LDB = (brg->is_row_major()) ? static_cast<int>(LDB)
                                     : static_cast<int>(LDA);
    brg->LDC = static_cast<int>(LDC);
    brg->LDD = static_cast<int>(LDC);

    brg->bcast_dim
            = (brg->is_row_major()) ? static_cast<int>(M) : static_cast<int>(N);
    brg->load_dim
            = (brg->is_row_major()) ? static_cast<int>(N) : static_cast<int>(M);
    brg->reduce_dim = static_cast<int>(K);

    brg->bd_block2 = 0;
    brg->bdb2 = 0;
    brg->bdb2_tail = 0;

    brg->ld_step = brg->rd_step = 4 / brg->typesize_A;
}

void init_brdgemm_conf(brgemm_t *brg, brgemm_batch_kind_t type,
        impl::data_type_t dt_a, impl::data_type_t dt_b, brgemm_layout_t layout,
        float alpha, float beta, dim_t LDA, dim_t LDC, dim_t M, dim_t N,
        const brgemm_strides_t *strides) {

    init_common_conf(brg, type, alpha, beta, strides);

    brg->layout = layout;

    brg->dt_a = dt_a;
    brg->dt_b = dt_b;
    init_kernel_datatype(brg, brg->dt_a, brg->dt_b);

    brg->dt_c = get_accum_datatype(brg);
    brg->dt_d = brg->dt_c;
    brg->dt_bias = brg->dt_c;

    brg->typesize_A = types::data_type_size(brg->dt_a);
    brg->typesize_B = types::data_type_size(brg->dt_b);
    brg->typesize_C = types::data_type_size(brg->dt_c);
    brg->typesize_D = types::data_type_size(brg->dt_d);

    brg->is_bf16_amx = brg->is_bf16 && mayiuse(avx512_core_bf16_amx_bf16);
    brg->is_dgmm = true;

    brg->LDA = static_cast<int>(LDA);
    brg->LDC = static_cast<int>(LDC);
    brg->LDD = static_cast<int>(LDC);

    brg->bcast_dim = M;
    brg->load_dim = N;
}

} // namespace brgemm_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
