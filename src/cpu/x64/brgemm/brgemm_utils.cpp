/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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
#include "cpu/x64/brgemm/jit_brdgmm_kernel.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
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
    assert(brg->is_int8 || brg->is_bf16 || brg->is_f32 || brg->is_f16);
    return brg->is_int8 ? data_type::s32 : data_type::f32;
}

void init_kernel_datatype(
        brgemm_t *brg, impl::data_type_t dt_a, impl::data_type_t dt_b) {
    assert(dt_a != data_type::undef && dt_b != data_type::undef);
    brg->is_int8 = utils::one_of(dt_a, data_type::u8, data_type::s8)
            && utils::one_of(dt_b, data_type::u8, data_type::s8);
    brg->is_bf16 = (dt_a == data_type::bf16) && (dt_b == data_type::bf16);
    brg->is_f32 = (dt_a == data_type::f32) && (dt_b == data_type::f32);
    brg->is_f16 = utils::one_of(data_type::f16, dt_a, dt_b);
    assert(brg->is_int8 || brg->is_bf16 || brg->is_f32 || brg->is_f16);
}

void init_common_conf(brgemm_t *brg, brgemm_batch_kind_t type, float alpha,
        float beta, const brgemm_strides_t *strides) {
    brg->beta = beta;
    brg->alpha = alpha;
    brg->type = type;
    brg->with_bias = false;
    brg->with_eltwise = false;
    brg->with_sum = false;
    brg->with_weights_scale_adjust = false;
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
    return brg->is_tmm
            && one_of(brg->type, brgemm_addr, brgemm_offs, brgemm_static_offs)
            && brg->brgattr.use_uker;
}

void maybe_try_bf32(brgemm_t *brg) {
    const bool try_bf32 = brg->is_f32
            && brg->brgattr.fpmath_mode == fpmath_mode::bf16
            && utils::one_of(brg->isa_user, isa_undef, avx512_core_amx)
            && mayiuse(avx512_core_amx);
    if (try_bf32) {
        const bool is_tmm = brg->is_tmm;
        brg->is_tmm = true;
        if (can_dispatch_uker(brg) /*Requires is_amx to be true*/) {
            brg->is_bf32 = true;
        } else {
            brg->is_bf32 = false;
            //  Restore
            brg->is_tmm = is_tmm;
        }
    }
}

void set_isa_impl(brgemm_t *brg) {
    auto is_isa_ok = [&](cpu_isa_t isa) {
        return mayiuse(isa) &&
                // maybe IMPLICATION(brg->isa_user != isa_undef,
                //  is_superset(brg->isa_user, isa)), but the API is not clear.
                one_of(brg->isa_user, isa_undef, isa);
    };

    if (brg->is_bf32) {
        brg->isa_impl = avx512_core_amx;
    } else if (brg->is_f32) {
        brg->isa_impl = utils::map(true, isa_undef,
                is_isa_ok(avx512_core) || is_isa_ok(avx512_core_amx) /*bf32*/,
                avx512_core, is_isa_ok(avx2), avx2,
                // Allow avx512_core_fp16 isa in case of a f16 primitive that
                // is implemented using pre-conversion of inputs to f32.
                // This is needed to support f16 binary post-ops.
                is_isa_ok(avx512_core_fp16), avx512_core_fp16, is_isa_ok(avx2),
                avx2);
    } else if (brg->is_bf16) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core_amx),
                avx512_core_amx, is_isa_ok(avx512_core_bf16), avx512_core_bf16,
                is_isa_ok(avx2_vnni_2), avx2_vnni_2);
    } else if (brg->is_f16) {
        if (everyone_is(data_type::f16, brg->dt_a, brg->dt_b)) {
            brg->isa_impl = utils::map(true, isa_undef,
                    is_isa_ok(avx512_core_amx_fp16), avx512_core_amx_fp16,
                    is_isa_ok(avx512_core_fp16), avx512_core_fp16,
                    is_isa_ok(avx2_vnni_2), avx2_vnni_2);
        } else {
            brg->isa_impl = utils::map(true, isa_undef,
                    is_isa_ok(avx512_core_fp16), avx512_core_fp16);
        }
    } else if (brg->is_int8) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core_amx),
                avx512_core_amx, is_isa_ok(avx512_core_vnni), avx512_core_vnni,
                is_isa_ok(avx512_core), avx512_core, is_isa_ok(avx2_vnni_2),
                avx2_vnni_2, is_isa_ok(avx2_vnni), avx2_vnni);
    }
}

void set_brg_vmm(brgemm_t *brg) {
    brg->is_tmm = brg->is_int8_tmm || brg->is_bf16_tmm || brg->is_f16_tmm
            || brg->is_bf32;
    brg->is_zmm = !brg->is_tmm && mayiuse(avx512_core)
            && is_superset(brg->isa_impl, avx512_core);
    brg->is_ymm
            = !brg->is_zmm && mayiuse(avx2) && is_superset(brg->isa_impl, avx2);
}

int calculate_ldb_params(brgemm_t *brg, const int try_ld_block2) {
    brg->ld_block2 = try_ld_block2;
    brg->ldb2 = brg->ldb / brg->ld_block2;
    brg->ldb2_tail = brg->ldb % brg->ld_block2;

    if (brg->ldb2 == 0) brg->ld_block2 = nstl::max(1, brg->ldb2_tail);
    brg->embd_bcst = brg->is_f32
            && (brg->ldb2_tail <= 1 && brg->ldb2 == 0)
            /*only avx512 or more can bcast*/
            && is_superset(brg->isa_impl, avx512_core);

    const int adj_ld_block2
            = (brg->ldb2 != 0) ? brg->ld_block2 : brg->ldb2_tail;
    return nstl::max(1, adj_ld_block2);
}

int calculate_max_bcast_block(brgemm_t *brg, const int adj_ld_block2) {

    constexpr int max_bcst_regs = 1;
    const bool req_compensation = brg->req_s8s8_compensation
            || brg->zp_type_a != brgemm_broadcast_t::none;
    const bool req_zp_a_comp_pads
            = (brg->req_cal_comp_pads || brg->brgattr.max_top_vpad > 0
                      || brg->brgattr.max_bottom_vpad > 0)
            && brg->zp_type_a != brgemm_broadcast_t::none;
    const int beta_regs = !one_of(brg->beta, 1.f, 0.f);

    const int max_isa_regs = isa_num_vregs(brg->isa_impl);
    // note: the 'adj_ld_block2' already removes the necessary registers
    // for 'embd_bcst'
    auto max_reg_count = max_isa_regs - max_bcst_regs - beta_regs
            - req_compensation - req_zp_a_comp_pads;
    if (req_zp_a_comp_pads)
        max_reg_count
                = nstl::min(max_reg_count, max_isa_regs - max_bcst_regs - 5);

    int max_bcast_block = max_reg_count - adj_ld_block2;

    if (brg->is_bf16_emu) {
        // in theory, vmm bf16_emu register indices overlap with other vmm
        // registers related to 'max_bcast_block'
        assert(is_superset(brg->isa_impl, avx512_core));
        constexpr int bf16_emu_reg_count = 28;
        max_bcast_block = nstl::min(max_bcast_block, bf16_emu_reg_count);
    }

    // non-VNNI INT8 dot product required 2 temp vectors
    if (brg->is_int8 && !brg->has_int8_vnni) max_bcast_block -= 2;

    max_bcast_block /= adj_ld_block2;

    return max_bcast_block;
}

status_t brgemm_blocking(brgemm_t *brg) {

    set_isa_impl(brg);
    if (brg->isa_impl == isa_undef) return status::unimplemented;
    assert(!brg->is_dgmm); // should not be called from brdgmm
    if (brg->is_dgmm) return status::unimplemented;
    set_brg_vmm(brg);
    if (!(brg->is_tmm || brg->is_zmm || brg->is_ymm))
        return status::unimplemented;

    if (!brg->is_tmm) {
        const int simd_w = is_superset(brg->isa_impl, avx512_core) ? 16 : 8;
        brg->ld_block = simd_w;
        brg->ldb = brg->load_dim / brg->ld_block;
        brg->ldb_tail = brg->load_dim % brg->ld_block;

        int adj_ld_block2 = calculate_ldb_params(brg, 4);
        int max_bcast_block = calculate_max_bcast_block(brg, adj_ld_block2);

        // reduce 'ld_block2' to allow a larger 'bd_block'
        const int max_vpad = nstl::max(
                brg->brgattr.max_top_vpad, brg->brgattr.max_bottom_vpad);
        if (is_superset(brg->isa_impl, avx2) && max_bcast_block < max_vpad) {
            adj_ld_block2 = calculate_ldb_params(brg, 2);
            max_bcast_block = calculate_max_bcast_block(brg, adj_ld_block2);
        }

        const int min_block = 1;
        float best_bd_block_eff = 0.f;
        brg->bd_block = 1;
        for (int bd_block = max_bcast_block; bd_block >= min_block;
                bd_block--) {
            const auto bd_block_disb = static_cast<float>(brg->bcast_dim)
                    / rnd_up(brg->bcast_dim, bd_block);
            const auto brgemm_microkernel_eff
                    = (static_cast<float>(adj_ld_block2) * bd_block)
                    / (((adj_ld_block2) + bd_block) * max_bcast_block);
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

        const int rd_unroll = 4;
        const int vnni_granularity
                = (brg->is_f16 && brg->isa_impl == avx512_core_fp16)
                ? 1
                : data_type_vnni_granularity(brg->dt_a);
        brg->rd_block = rd_unroll * vnni_granularity;
        brg->rdb = brg->reduce_dim / brg->rd_block;
        brg->rdb_tail = brg->reduce_dim % brg->rd_block;

        brg->is_M_tail = false;
    } else {
        // Blocking configuration for AMX
        const int max_width = 16, min_width = 1;
        brg->ld_block = 16;
        brg->ldb = brg->load_dim / brg->ld_block;
        brg->ldb_tail = brg->load_dim % brg->ld_block;

        auto find_bdb_bd_mask = [&](int bd_block, int &bdb, int &bdb_tail) {
            if (brg->brgattr.bd_mask_level != 2 || brg->bcast_dim == 0) {
                bdb = div_up(brg->bcast_dim, bd_block);
                bdb_tail = brg->bcast_dim % bd_block;
                return;
            }

            bdb = 0;
            bdb_tail = 0;
            for (int i = 0; i < brg->bcast_dim;) {
                if (brg->brgattr.bd_mask_level == 2
                        && brg->brgattr.bd_mask[i] == 0) {
                    i++;
                } else {
                    i += bd_block;
                    if (i > brg->bcast_dim) {
                        bdb_tail = brg->bcast_dim - i + bd_block;
                        if (brg->brgattr.use_uker) bdb++;
                    } else
                        bdb++;
                }
            }
        };

        auto find_bd_block_for_bd_mask = [&]() {
            if (brg->brgattr.bd_mask_level != 2 || brg->bcast_dim == 0)
                return false;

            auto min_bdb = INT_MAX;
            const auto start_bd_block = nstl::min(max_width, brg->bcast_dim);
            auto best_bd_block = start_bd_block;
            for (auto bd_block = start_bd_block; bd_block > 0; bd_block--) {
                int bdb = 0;
                int bdb_tail = 0;
                find_bdb_bd_mask(bd_block, bdb, bdb_tail);
                // bcast_dim should be divided by bd_block
                if (bdb < min_bdb && bdb_tail == 0) {
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
                        const auto i_tail = brg->bcast_dim % i;
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

        auto recalc_bd_block = [&](int new_bd_block) {
            if (new_bd_block == 0) return;
            brg->bd_block = new_bd_block;
            find_bdb_bd_mask(brg->bd_block, brg->bdb, brg->bdb_tail);
            brg->is_M_tail = (brg->bdb_tail != 0);
        };

        auto recalc_bd_block2 = [&](int new_bd_block2) {
            if (new_bd_block2 == 0) return;
            brg->bd_block2 = new_bd_block2;
            if (can_dispatch_uker(brg)) {
                brg->bdb2 = div_up(brg->bdb, brg->bd_block2);
                brg->bdb2_tail = 0;
            } else {
                if (brg->bdb_tail && brg->bd_block2 > 1) brg->bd_block2--;
                auto full_bd_blocks = brg->bdb - (brg->bdb_tail != 0 ? 1 : 0);
                brg->bdb2 = full_bd_blocks / brg->bd_block2;
                brg->bdb2_tail = full_bd_blocks % brg->bd_block2;
            }
        };

        auto recalc_ld_block = [&](int new_ld_block) {
            if (new_ld_block == 0) return;
            brg->ld_block = new_ld_block;
            brg->ldb = div_up(brg->load_dim, brg->ld_block);
            brg->ldb_tail = brg->load_dim % brg->ld_block;
        };

        auto recalc_ld_block2 = [&](int new_ld_block2) {
            if (new_ld_block2 == 0) return;
            brg->ld_block2 = new_ld_block2;
            if (can_dispatch_uker(brg)) {
                brg->ldb2 = div_up(brg->ldb, brg->ld_block2);
                brg->ldb2_tail = 0;
            } else {
                if (brg->ldb_tail && brg->ld_block2 > 1) brg->ld_block2--;
                auto full_ld_blocks = brg->ldb - (brg->ldb_tail != 0 ? 1 : 0);
                brg->ldb2 = full_ld_blocks / brg->ld_block2;
                brg->ldb2_tail = full_ld_blocks % brg->ld_block2;
            }
        };

        const bool try_load_nt_A
                = (brg->innermost_loop == brgemm_bd_loop_innermost);
        const bool try_load_nt_B
                = (brg->innermost_loop == brgemm_ld_loop_innermost);
        const bool try_load_nt
                = (static_cast<size_t>(brg->typesize_A)
                                  * brg->brgattr.hint_expected_A_size
                          + static_cast<size_t>(brg->typesize_B)
                                  * brg->brgattr.hint_expected_B_size
                          + static_cast<size_t>(brg->typesize_C)
                                  * brg->brgattr.hint_expected_C_size)
                >= platform::get_per_core_cache_size(1);
        brg->load_nt_A = try_load_nt_A && try_load_nt;
        brg->load_nt_B = try_load_nt_B && try_load_nt;

        recalc_bd_block(brg->bd_block);
        recalc_bd_block2(brg->bd_block2);
        recalc_ld_block(brg->ld_block);
        recalc_ld_block2(brg->ld_block2);

        if (brg->brgattr.use_uker) {
            // Blocking heuristics for some shapes
            // TODO: Review these criterias
            size_t eff_K
                    = brg->reduce_dim * brg->typesize_A * brg->brgattr.K_koef;
            auto L1 = platform::get_per_core_cache_size(1);
            auto low_K = (L1 - 4 * 1024) / (6 * 16);

            // TODO: if rdb_tail != 0 then we should limit
            // blocking because we need extra tiles for A and B to load rdb_tail
            // if bd_mask_level != 0 it means it aligned to 16

            bool bdb_block_tail = !(brg->bd_block > 12
                    && (brg->bcast_dim % brg->bd_block == 0
                            && brg->brgattr.bd_mask_level == 0));
            bool ldb_tail_16 = (brg->load_dim % 16 != 0);
            if (everyone_is(false, bdb_block_tail, ldb_tail_16)) {
                // try to use 1x(4|5) or (4|5)x1 decomposition for specific
                // range of K
                auto upper_K5 = (L1 - 5 * 1024) / (5 * 16);
                auto upper_K4 = (L1 - 4 * 1024) / (4 * 16);
                bool K5_fit_L1 = (low_K <= eff_K && eff_K < upper_K5);
                bool K4_fit_L1 = (low_K <= eff_K && eff_K < upper_K4);
                bool bd_big = (brg->bcast_dim > 32);
                bool ld_big = (brg->load_dim > 32);
                if (brg->load_dim % 80 == 0 && K5_fit_L1 && bd_big) {

                    recalc_ld_block(16);
                    recalc_bd_block2(1);
                    recalc_ld_block2(5);
                    brg->load_nt_A = true;
                    brg->load_nt_B = false;
                    brg->innermost_loop = brgemm_bd_loop_innermost;
                } else if (brg->load_dim % 64 == 0 && K4_fit_L1 && bd_big) {

                    recalc_ld_block(16);
                    recalc_bd_block2(1);
                    recalc_ld_block2(4);
                    brg->load_nt_A = true;
                    brg->load_nt_B = false;
                    brg->innermost_loop = brgemm_bd_loop_innermost;
                } else if ((brg->bcast_dim % 80 == 0
                                   || (brg->brgattr.bd_mask_level != 0
                                           && brg->bdb % 4 == 0))
                        && K5_fit_L1 && ld_big) {

                    recalc_ld_block(16);
                    recalc_bd_block2(5);
                    recalc_ld_block2(1);
                    brg->load_nt_A = false;
                    brg->load_nt_B = true;
                    brg->innermost_loop = brgemm_ld_loop_innermost;
                } else if ((brg->bcast_dim % 64 == 0
                                   || (brg->brgattr.bd_mask_level != 0
                                           && brg->bdb % 4 == 0))
                        && K4_fit_L1 && ld_big) {

                    recalc_bd_block(16);
                    recalc_ld_block(16);
                    recalc_bd_block2(4);
                    recalc_ld_block2(1);
                    brg->load_nt_A = false;
                    brg->load_nt_B = true;
                    brg->innermost_loop = brgemm_ld_loop_innermost;
                }
            }
            // Tile decomposition for shapes with small dimensions
            // or dimensions with tails
            if (ldb_tail_16 && !bdb_block_tail && brg->load_dim > 64
                    && brg->ld_block < 8) {
                recalc_ld_block(16);
                recalc_bd_block2(2);
                recalc_ld_block2(1);
            } else if (ldb_tail_16 && !bdb_block_tail
                    && rnd_up(brg->load_dim, 16) == 64
                    && (brg->ld_block < 8 || brg->ldb_tail > 0)) {
                recalc_ld_block(16);
                recalc_bd_block2(1);
                recalc_ld_block2(4);
            } else if (ldb_tail_16 && !bdb_block_tail
                    && rnd_up(brg->load_dim, 16) == 48
                    && (brg->ld_block < 8 || brg->ldb_tail > 0)) {
                recalc_ld_block(16);
                recalc_bd_block2(1);
                recalc_ld_block2(3);
            } else if (ldb_tail_16 && !bdb_block_tail
                    && rnd_up(brg->load_dim, 16) == 32
                    && (brg->ld_block < 8 || brg->ldb_tail > 0)) {
                recalc_ld_block(16);
                recalc_bd_block2(2);
                recalc_ld_block2(2);
            } else if (brg->bcast_dim <= 16) {
                recalc_bd_block(brg->bcast_dim);
                recalc_ld_block(16);
                recalc_bd_block2(1);
                recalc_ld_block2(
                        nstl::min(ldb_tail_16 ? ((brg->ldb > 4) ? 3 : 4) : 5,
                                div_up(brg->load_dim, 16)));
            } else if (bdb_block_tail && !ldb_tail_16 && brg->bcast_dim > 64
                    && (brg->bd_block < 8 || brg->bdb_tail > 0)) {

                recalc_bd_block(16);
                recalc_ld_block(16);
                recalc_bd_block2(1);
                recalc_ld_block2(2);
            } else if (bdb_block_tail && !ldb_tail_16
                    && rnd_up(brg->bcast_dim, 16) == 64
                    && (brg->bd_block < 8 || brg->bdb_tail > 0)) {
                recalc_bd_block(16);
                recalc_ld_block(16);
                recalc_bd_block2(4);
                recalc_ld_block2(1);
            } else if (bdb_block_tail && !ldb_tail_16
                    && rnd_up(brg->bcast_dim, 16) == 48
                    && (brg->bd_block < 8 || brg->bdb_tail > 0)) {
                recalc_bd_block(16);
                recalc_ld_block(16);
                recalc_bd_block2(3);
                recalc_ld_block2(1);
            } else if (bdb_block_tail && !ldb_tail_16
                    && rnd_up(brg->bcast_dim, 16) == 32
                    && (brg->bd_block < 8 || brg->bdb_tail > 0)
                    && (brg->load_dim % 32 == 0)) {

                recalc_bd_block(16);
                recalc_ld_block(16);
                recalc_bd_block2(2);
                recalc_ld_block2(2);
            } else if (brg->load_dim <= 16) {
                recalc_bd_block(16);
                recalc_ld_block(16); // we can't use ld_block other than 16
                recalc_bd_block2(
                        nstl::min(brg->bdb_tail ? (brg->bdb > 4 ? 3 : 4) : 5,
                                div_up(brg->bcast_dim, 16)));
                recalc_ld_block2(1);
            } else if (bdb_block_tail && ldb_tail_16
                    && rnd_up(brg->bcast_dim, 16) == 32
                    && rnd_up(brg->load_dim, 16) == 32
                    && (brg->ld_block < 8 || brg->ldb_tail > 0
                            || brg->bd_block < 8 || brg->bdb_tail > 0)) {
                recalc_bd_block(16);
                recalc_ld_block(16);
                recalc_bd_block2(2);
                recalc_ld_block2(2);
            }
            // if interleave stores and small number of iterations then
            // try to increase them
            auto n_iterations = brg->bdb2 * brg->bdb2;
            if (false && brg->brgattr.use_interleave_stores
                    && n_iterations < 4) {
                int k_it = div_up(4, n_iterations);
                if (brg->bdb2 > brg->ldb2)
                    recalc_bd_block2(div_up(brg->bdb2, k_it));
                else
                    recalc_ld_block2(div_up(brg->ldb2, k_it));
            }
        }

        if (brg->get_num_A_tiles() + brg->get_num_B_tiles()
                        + brg->get_num_C_tiles()
                > brgemm_t::AMX_TILES_NUM) {
            assert(!"brgemm internal error: invalid blocking");
            return status::runtime_error;
        }

        // check hints for blocking parameters
        recalc_bd_block(brg->brgattr.hint_bd_block);
        recalc_bd_block2(brg->brgattr.hint_bd_block2
                        ? brg->brgattr.hint_bd_block2
                        : brg->bd_block2);
        recalc_ld_block(brg->brgattr.hint_ld_block);
        recalc_ld_block2(brg->brgattr.hint_ld_block2
                        ? brg->brgattr.hint_ld_block2
                        : brg->ld_block2);

        if (brg->brgattr.hint_load_nt_A != brgemm_hint_nt_undef)
            brg->load_nt_A
                    = (brg->brgattr.hint_load_nt_A == brgemm_hint_nt_true);
        if (brg->brgattr.hint_load_nt_B != brgemm_hint_nt_undef)
            brg->load_nt_B
                    = (brg->brgattr.hint_load_nt_B == brgemm_hint_nt_true);

        const auto max_rd_block
                = (brg->is_bf16_tmm || brg->is_f16_tmm || brg->is_bf32) ? 32
                                                                        : 64;
        const auto rd_block_step
                = (brg->is_bf16_tmm || brg->is_f16_tmm || brg->is_bf32) ? 2 : 4;
        // TODO: if rd_block calculated is very small then maybe it makes
        // sense to use 1x2 or 2x1 blocking with supporting rd_block
        // and rdb_tail
        brg->rd_block = rd_block_step;
        for (int i = max_rd_block; i > 0; i -= rd_block_step) {
            if (brg->reduce_dim % i == 0) {
                brg->rd_block = i;
                break;
            }
        }
        brg->rdb = brg->reduce_dim / brg->rd_block;
        brg->rdb_tail = brg->reduce_dim % brg->rd_block;

        // Remove these guards in the future (add tail processing by reduction
        // dimension)
        if (!IMPLICATION(brg->rdb > 0 && brg->rdb_tail, brg->is_bf32))
            return status::unimplemented;
        if (!IMPLICATION(
                    (brg->rdb_tail
                            % ((brg->is_bf16_tmm || brg->is_f16_tmm) ? 2 : 4))
                            != 0,
                    brg->is_bf32))
            return status::unimplemented;

        //TODO: check this condition
        brg->interleave_tilestores_ = brg->beta == 0
                        && (brg->brgattr.use_interleave_stores
                                && (brg->bd_block2 * brg->ld_block2 == 4)
                                && !brg->brgattr.var_bs)
                ? true
                : false;
    }

    return status::success;
}

status_t brdgmm_blocking(brgemm_t *brg) {

    if (brg->isa_impl == isa_undef) return status::unimplemented;

    const int max_vregs = isa_num_vregs(brg->isa_impl);

    // Note: using avx512_core template, but calculation uses 'brg->isa_impl'
    // which is dynamic i.e. uses values AVX2, AVX2_VNNI, etc. depending on the
    // configuration.
    const int aux_vregs = jit_brdgmm_kernel_base_t<avx512_core_vnni,
            Xbyak::Zmm>::get_aux_vmm_count(*brg);
    const int compute_vregs = jit_brdgmm_kernel_base_t<avx512_core_vnni,
            Xbyak::Zmm>::get_compute_vmm_count(*brg);
    const int bf16_emu_vregs = brg->is_bf16_emu * 4;
    const int max_acc_vmms
            = max_vregs - nstl::max(compute_vregs + aux_vregs, bf16_emu_vregs);

    const int simd_w = isa_max_vlen(brg->isa_impl) / brg->typesize_C;
    const bool is_avx2_vnni_2_xf16
            = brg->is_xf16() && brg->isa_impl == avx2_vnni_2;

    auto &M = brg->bcast_dim;
    auto &N = brg->load_dim;

    // In current implementation of dgmm, there is no reduce dim.
    auto &m_block1 = brg->bd_block;
    auto &nb_m_block1 = brg->bdb;
    auto &m_block1_tail = brg->bdb_tail;
    auto &m_block2 = brg->bd_block2;
    auto &nb_m_block2 = brg->bdb2;
    auto &m_block2_tail = brg->bdb2_tail;

    auto &n_block1 = brg->ld_block;
    auto &nb_n_block1 = brg->ldb;
    auto &n_block1_tail = brg->ldb_tail;
    auto &n_block2 = brg->ld_block2;
    auto &nb_n_block2 = brg->ldb2;
    auto &n_block2_tail = brg->ldb2_tail;

    // begin blocking
    // for avx2_vnni_2_xf16, instead of processing a n_block1 at once, it is
    // processed as even/odd pair.
    const int n_block1_num_steps = is_avx2_vnni_2_xf16 ? 2 : 1;
    n_block1 = n_block1_num_steps * simd_w;
    nb_n_block1 = div_up(N, n_block1);
    n_block1_tail = N % n_block1;

    const auto min_possible_m_block2 = brg->brgattr.bs_group > 1
            ? (max_acc_vmms / (2 * n_block1_num_steps) - brg->brgattr.bs_group
                    + 1)
            : 1;

    if (min_possible_m_block2 < 1) brg->brgattr.bs_group = 1;

    if (brg->brgattr.bs_group > 1) {
        n_block2 = 1;
    } else {
        const int max_n_block2_vmms = 4;
        const int max_n_block2 = max_n_block2_vmms / n_block1_num_steps;
        n_block2 = nstl::min(max_n_block2, nb_n_block1);
    }

    nb_n_block2 = div_up(nb_n_block1, n_block2);
    n_block2_tail = nb_n_block1 % n_block2;

    m_block1 = 1;
    nb_m_block1 = M / m_block1;
    m_block1_tail = M % m_block1;
    m_block2 = nstl::min(nb_m_block1,
            brg->brgattr.bs_group > 1
                    ? max_acc_vmms / (2 * n_block2 * n_block1_num_steps)
                            - brg->brgattr.bs_group + 1
                    : max_acc_vmms / (n_block2 * n_block1_num_steps));
    nb_m_block2 = div_up(nb_m_block1, m_block2);
    m_block2_tail = nb_m_block1 % m_block2;

    return status::success;
}

void init_brgemm_conf(brgemm_t *brg, cpu_isa_t isa, brgemm_batch_kind_t type,
        impl::data_type_t dt_a, impl::data_type_t dt_b, brgemm_layout_t layout,
        float alpha, float beta, dim_t LDA, dim_t LDB, dim_t LDC, dim_t M,
        dim_t N, dim_t K, const brgemm_strides_t *strides, bool is_bf32) {

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

    brg->isa_user = isa;
    set_isa_impl(brg);
    brg->is_int8_tmm = brg->is_int8 && brg->isa_impl == avx512_core_amx;
    brg->is_bf16_tmm = brg->is_bf16 && brg->isa_impl == avx512_core_amx;
    brg->is_f16_tmm = brg->is_f16 && brg->isa_impl == avx512_core_amx_fp16;
    brg->is_bf32 = is_bf32
            && utils::one_of(brg->isa_user, isa_undef, avx512_core_amx)
            && mayiuse(avx512_core_amx);

    brg->has_int8_vnni = isa_has_int8_vnni(brg->isa_impl);

    set_brg_vmm(brg); // TODO: Investigate if it is really needed here.
    brg->req_s8s8_compensation = brg->is_int8 && !brg->is_int8_tmm
            && (brg->isa_impl != avx2_vnni_2) && brg->dt_a == data_type::s8;

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

    const bool is_b_in_vnni_format = !(
            brg->dt_b == data_type::f16 && brg->isa_impl == avx512_core_fp16);
    brg->ld_step
            = is_b_in_vnni_format ? data_type_vnni_granularity(brg->dt_b) : 1;

    const bool has_no_vnni_compute_instruction
            = (brg->is_f16
                      && one_of(brg->isa_impl, avx2_vnni_2, avx512_core_fp16))
            || (brg->is_bf16 && brg->isa_impl == avx2_vnni_2);
    brg->rd_step = has_no_vnni_compute_instruction
            ? 1
            : data_type_vnni_granularity(brg->dt_b);
}

void init_brdgmm_conf(brgemm_t *brg, cpu_isa_t isa, brgemm_batch_kind_t type,
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

    brg->isa_user = isa;
    auto is_isa_ok = [&](cpu_isa_t isa) {
        return mayiuse(isa) && one_of(brg->isa_user, isa_undef, isa);
    };

    if (brg->is_f32) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core),
                avx512_core, is_isa_ok(avx2), avx2);
    } else if (brg->is_bf16) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core_bf16),
                avx512_core_bf16, is_isa_ok(avx2_vnni_2), avx2_vnni_2);
    } else if (brg->is_f16) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core_fp16),
                avx512_core_fp16, is_isa_ok(avx2_vnni_2), avx2_vnni_2);
    } else if (brg->is_int8) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core_vnni),
                avx512_core_vnni, is_isa_ok(avx2_vnni_2), avx2_vnni_2,
                is_isa_ok(avx2_vnni), avx2_vnni);
    }

    brg->req_s8s8_compensation = brg->is_int8 && brg->dt_a == data_type::s8
            && !isa_has_s8s8(brg->isa_impl);

    brg->is_bf16_tmm = brg->is_bf16 && mayiuse(avx512_core_amx);
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
