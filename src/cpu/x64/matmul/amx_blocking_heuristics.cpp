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

#include "cpu/x64/matmul/amx_blocking_heuristics.hpp"
#include "cpu/matmul/gemm_based_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::cpu::matmul;

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace data_type;
using namespace format_tag;
void matmul_amx_blocking_params_t::update_configuration(
        brgemm_matmul_conf_t &bgmmc) const {
    bgmmc.nthr_k = nthr_k_;
    bgmmc.nthr_m = nthr_m_;
    bgmmc.nthr_n = nthr_n_;
    bgmmc.nthr_b = nthr_b_;
    bgmmc.nthr = nthr_;
    bgmmc.M_blk = m_blk_;
    bgmmc.M_chunk_size = m_chunk_size_;
    bgmmc.N_blk = n_blk_;
    bgmmc.N_chunk_size = n_chunk_size_;

    bgmmc.K_blk = k_blk_;
    bgmmc.K_chunk_size = k_chunk_size_;
    bgmmc.brgemm_batch_size = brgemm_batch_size_;

    bgmmc.use_buffer_c = need_buf_c_;
    bgmmc.use_buffer_a = need_buf_a_;
    bgmmc.extendable_k = extendable_k_;
    bgmmc.LDA = current_lda_;

    bgmmc.is_a_nt = is_a_nt_;
    bgmmc.is_b_nt = is_b_nt_;
    bgmmc.set_nt = set_nt_;
    bgmmc.is_macro_heuristics
            = dynamic_cast<const matmul_amx_blocking_params_macro_t *>(this)
            != nullptr;
}

dim_t matmul_amx_blocking_params_t::get_actual_lda() {
    if (!need_buf_a_)
        return treat_A_as_plain ? K : A_strides[1 - transposed_A] / a_dt_sz;

    constexpr int bytes_in_cacheline = 64;
    const int elems_in_cacheline = bytes_in_cacheline / a_dt_sz;
    dim_t lda = rnd_up(k_blk_, elems_in_cacheline);
    const bool is_big_2_pow = lda >= 512 && math::is_pow2(lda);
    if (is_big_2_pow) lda += elems_in_cacheline;
    return lda;
}

bool matmul_amx_blocking_params_t::is_buffer_c_required() {
    if (nthr_k_ > 1 && K > k_chunk_elems_) return true;

    return ((acc_dt != dst_dt || with_sum)
            && (K > k_chunk_elems_ || K % k_blk_ > 0));
}

size_t matmul_amx_blocking_params_t::L2_threshold() {
    return 3 * platform::get_per_core_cache_size(2) / 4;
}

size_t matmul_amx_blocking_params_t::L1_threshold() {
    return 5 * platform::get_per_core_cache_size(1) / 6;
}

bool matmul_amx_blocking_params_macro_t::is_supported(
        const brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils) {
    //todo: enable extendable_k optimization
    if (bgmmc.K < bgmmc.wei_k_blk
            || bgmmc.K % data_type_vnni_granularity(bgmmc.wei_dt) != 0) {
        return false;
    }

    bool a_dt_ok
            = one_of(bgmmc.orig_src_dt, dnnl_s8, dnnl_u8, dnnl_bf16, dnnl_f16);
    bool b_dt_ok
            = one_of(bgmmc.orig_wei_dt, dnnl_s8, dnnl_u8, dnnl_bf16, dnnl_f16);

    bool a_tag_ok = bgmmc.src_tag == dnnl_format_tag_any
            || bm_conf_utils.check_is_plain(bgmmc.src_tag);
    bool b_tag_ok = bm_conf_utils.is_any_B_layout()
            || bm_conf_utils.check_b_layout_blocked_32_by_n(bgmmc.wei_tag);

    bool has_zp = bgmmc.src_zp_type != brgemm_broadcast_t::none
            || bgmmc.wei_zp_type != brgemm_broadcast_t::none
            || bgmmc.dst_zp_type != brgemm_broadcast_t::none;

    return bgmmc.orig_src_dt == bgmmc.src_dt
            && bgmmc.orig_wei_dt == bgmmc.wei_dt && bgmmc.is_amx
            && !bgmmc.is_runtime_N && !bgmmc.is_runtime_M && a_dt_ok && a_tag_ok
            && (bgmmc.reduce_kind == matmul_reduce_kind::undef) && b_tag_ok
            && b_dt_ok && !has_zp;
}

bool matmul_amx_blocking_params_macro_t::divs_are_acceptable() {
    bool unacceptable_m_div = m_per_thread < min_m_dim && nthr_m_ > 1;
    bool unacceptable_k_div = k_per_thread < min_k_dim && nthr_k_ > 1;
    bool unacceptable_n_div;
    if (nthr_k_ == 1 && k_per_thread < k_threshold_write_bound_layer) {
        // The layer is write bound (small K) and no reduction (C becomes non consecutive)
        unacceptable_n_div
                = n_per_thread < min_n_dim_write_bound_layer && nthr_n_ > 1;
    } else {
        unacceptable_n_div = n_per_thread < min_n_dim && nthr_n_ > 1;
    }

    bool unacceptable_b_div = nthr_b_ > (size_t)batch;

    return !unacceptable_m_div && !unacceptable_k_div && !unacceptable_n_div
            && !unacceptable_b_div;
}

size_t determine_tmul_size(size_t num_elements, int full_tile_size) {
    size_t tmul_tiles = div_up(num_elements, full_tile_size);
    size_t tmul_size = div_up(num_elements, tmul_tiles);
    return tmul_size;
}

bool matmul_amx_blocking_params_macro_t::find_best_blocking(
        const brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils,
        matmul_amx_blocking_params_macro_t &best_blocking) {

    if (!matmul_amx_blocking_params_macro_t::is_supported(
                bgmmc, bm_conf_utils)) {
        return false;
    }

    best_blocking = matmul_amx_blocking_params_micro_t(bgmmc);

    matmul_amx_blocking_params_macro_t current_blocking(bgmmc);
    assert(bgmmc.tr_a_dt_sz == bgmmc.tr_b_dt_sz);
    current_blocking.gemm_dt_sz = bgmmc.tr_a_dt_sz;

    for (size_t nthr_to_check = bgmmc.nthr; nthr_to_check > 0;
            nthr_to_check--) {
        current_blocking.nthr_ = nthr_to_check;

        for (int b_div = 1; b_div <= current_blocking.nthr_; ++b_div) {
            if (current_blocking.nthr_ % b_div == 0) {
                for (int m_div = 1; m_div <= current_blocking.nthr_ / b_div;
                        ++m_div) {
                    if ((current_blocking.nthr_ / b_div) % m_div == 0) {
                        for (int k_div = 1; k_div
                                <= (current_blocking.nthr_ / b_div) / m_div;
                                ++k_div) {
                            if (((current_blocking.nthr_ / b_div) / m_div)
                                            % k_div
                                    == 0) {
                                int n_div = ((current_blocking.nthr_ / b_div)
                                                    / m_div)
                                        / k_div;
                                current_blocking.set_core_divs(
                                        b_div, m_div, k_div, n_div);
                                if (current_blocking.divs_are_acceptable()
                                        && current_blocking
                                                   .set_blocking_parameters()) {
                                    if (current_blocking > best_blocking) {
                                        best_blocking = current_blocking;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return true;
}

float matmul_amx_blocking_params_macro_t::calculate_blocking_scores() {

    size_t a_size = m_per_thread * k_per_thread * gemm_dt_sz;
    size_t b_size = n_per_thread * k_per_thread * gemm_dt_sz;
    size_t d_size = m_per_thread * n_per_thread * c_dt_sz;

    bw_map_t bw_interpulator;

    int macs_per_cycle_base = 1024;
    int max_k_tmul = 64;
    int max_n_tmul = 16;
    //Reducing k-tmul or n-tmul does not shorten the cycles.
    //However, reducing mtmul reduces the number of cycles required to execute a single tmul instruction.
    int num_cycles_per_tmul
            = m_tmul * max_k_tmul * max_n_tmul / macs_per_cycle_base;

    // Calculate reduction cycles
    float strip_1_size_shared, strip_1_size_private, strip_1_share_coef;
    float strip_mid_size_shared, strip_mid_size_private;
    float num_tmuls_per_strip, strip_mid_share_coef, num_strip, nt_mat_l1_miss;
    float l1_reuse;

    if (is_horizontal) {
        size_t strip_dst_size = m_decomposition * n_per_thread
                * (nthr_k_ == 1 ? c_dt_sz : acc_dt_sz);
        num_tmuls_per_strip = m_decomposition * k_per_thread * n_per_thread
                / (m_tmul * k_tmul * n_tmul);
        num_strip = div_up(m_per_thread, m_decomposition);
        nt_mat_l1_miss = b_size;
        l1_reuse = div_up(n_blk_, n_decomposition);

        // In strip 1 there is no sharing of A since there are no prefetches
        strip_1_size_shared = b_size;
        size_t strip_1_size_private_a
                = m_decomposition * k_per_thread * gemm_dt_sz;
        strip_1_size_private = strip_1_size_private_a + strip_dst_size;
        // the cores that share B
        strip_1_share_coef = nthr_m_;

        strip_mid_size_shared
                = m_decomposition * k_per_thread * gemm_dt_sz; // A
        strip_mid_size_private = strip_dst_size;
        // share_coeff - the cores that share A
        strip_mid_share_coef = std::max((size_t)1, nthr_n_);

    } else {
        size_t strip_dst_size = n_decomposition * m_per_thread
                * (nthr_k_ == 1 ? c_dt_sz : acc_dt_sz);
        num_tmuls_per_strip = n_decomposition * k_per_thread * m_per_thread
                / (m_tmul * k_tmul * n_tmul);
        num_strip = div_up(n_per_thread, n_decomposition);
        nt_mat_l1_miss = a_size;
        l1_reuse = div_up(m_blk_, m_decomposition);

        // In strip 1 there is no sharing of B since there are no prefetches
        strip_1_size_shared = a_size;
        size_t strip_1_size_private_b
                = n_decomposition * k_per_thread * gemm_dt_sz;
        strip_1_size_private = strip_1_size_private_b + strip_dst_size;
        // the cores that share A
        strip_1_share_coef = nthr_n_;

        strip_mid_size_shared
                = n_decomposition * k_per_thread * gemm_dt_sz; // B
        strip_mid_size_private = strip_dst_size;
        // share_coeff - the cores that share B
        strip_mid_share_coef = std::max((size_t)1, nthr_m_);
    }
    // there are 2 l1 misses for the l1 matrix:
    //   1. for the prefetch to the l2 (==l1 miss)
    //   2. for the read from l2
    float temporal_matrix_l1_miss = strip_mid_size_shared * 2;
    float temporal_matrix_l1_hit = strip_mid_size_shared * (l1_reuse - 1);

    float c_elem_per_strip = m_blk_ * n_blk_;

    // c post write miss in bytes = m_blk_ * (#n_decompositions in BRGEMM) * (#cache lines per n_decomposition) * 64
    float c_post_write_miss = m_blk_ * div_up(n_blk_, n_decomposition)
            * rnd_up(n_decomposition * c_dt_sz, 64);
    // c post write total in bytes = m_blk_ * (#n_decompositions in BRGEMM) * (#writes per n_decomposition) * 64
    float c_post_write_total = m_blk_ * div_up(n_blk_, n_decomposition)
            * rnd_up(n_decomposition * c_dt_sz, 16) * (4 / c_dt_sz);
    float c_post_write_hit = c_post_write_total - c_post_write_miss;

    float c_post_read_c_tmp = c_elem_per_strip * acc_dt_sz;

    float c_tmp_l1_cycles;
    if (k_blk_ == K) {
        c_tmp_l1_cycles = acc_dt_sz * c_elem_per_strip * k_chunk_size_
                / bw_interpulator.l1_load_hit_bw;
    } else {
        // todo: modify wrt wsp
        c_tmp_l1_cycles = acc_dt_sz * c_elem_per_strip * k_chunk_size_
                / bw_interpulator.l1_store_miss_bw;
    }

    float c_l1_cycles = c_post_write_miss / bw_interpulator.l1_store_miss_bw
            + c_post_write_hit / bw_interpulator.l1_store_hit_bw
            + c_post_read_c_tmp / bw_interpulator.l1_store_hit_bw
            + c_tmp_l1_cycles;
    float l1_cycles = temporal_matrix_l1_miss / bw_interpulator.l1_load_miss_bw
            + temporal_matrix_l1_hit / bw_interpulator.l1_load_hit_bw
            + nt_mat_l1_miss / bw_interpulator.l1_load_miss_bw + c_l1_cycles;

    float strip_1_cycles
            = strip_1_size_shared / bw_interpulator.get_bw(strip_1_share_coef)
            + strip_1_size_private / bw_interpulator.get_bw(1);

    float strip_mid_dram = strip_mid_size_shared
                    / bw_interpulator.get_bw(strip_mid_share_coef)
            + strip_mid_size_private / bw_interpulator.get_bw(1);
    float strip_mid_llc = (strip_mid_size_private + strip_mid_size_shared)
            / bw_interpulator.llc_bw;
    float strip_tmul = num_tmuls_per_strip * num_cycles_per_tmul;
    float strip_mid_cycles
            = std::max({strip_mid_dram, strip_mid_llc, l1_cycles, strip_tmul});

    float gemm_cycles = strip_1_cycles + (num_strip - 1) * strip_mid_cycles;

    // Calculate reduction cycles
    float reduction_cycles;
    size_t c_size_per_core = m_per_thread * n_per_thread * acc_dt_sz;

    if (nthr_k_ != 1) {
        if (c_size_per_core * 2 < L2_threshold() && batch == 1) {
            float reduction_read_bytes = (M * N * acc_dt_sz) * ((nthr_k_ - 1))
                    / (nthr_m_ * nthr_n_);
            float reduction_read_cycles;
            if (a_size + b_size + d_size < L2_threshold()) {
                reduction_read_cycles
                        = reduction_read_bytes / bw_interpulator.get_bw(2);
            } else {
                reduction_read_cycles
                        = reduction_read_bytes / bw_interpulator.llc_bw;
            }

            float reduction_write_bytes
                    = (M * N * c_dt_sz) / (nthr_m_ * nthr_n_);
            float reduction_write_cycles
                    = reduction_write_bytes / bw_interpulator.get_bw(1);
            // Add reduction const overhead - measured
            reduction_cycles
                    = reduction_read_cycles + reduction_write_cycles + 25000;
        } else {
            //don't do reduction if c tmp doesn't fit
            //also parallel reduction is not supported for large batch
            return 0;
        }
    } else {
        reduction_cycles = 0;
    }

    float total_macs = M * K * N * batch;
    float total_cycles = (gemm_cycles + reduction_cycles) * b_per_thread;
    float peak_macs_per_cycle = (macs_per_cycle_base / gemm_dt_sz) * nthr;
    float peak_cycles = total_macs / peak_macs_per_cycle;
    return peak_cycles / total_cycles;
}

bool matmul_amx_blocking_params_macro_t::operator>(
        matmul_amx_blocking_params_macro_t &other) {
    if (other.efficiency_score_ > this->efficiency_score_) { return false; }
    if (other.efficiency_score_ < this->efficiency_score_) { return true; }
    // Both efficiency scores are equal
    if (!this->is_horizontal && other.is_horizontal) {
        if (this->m_per_thread * K + (size_t)(this->m_per_thread * N)
                < L2_threshold()) {
            //vertical is an option. No l2 set issues for A
            if (other.is_a_nt_) {
                // horizontal uses precopy
                return true;
            }
        }
        return false;
    } else if (this->is_horizontal && !other.is_horizontal) {
        if (other.m_per_thread * K + (size_t)(other.m_per_thread * N)
                < L2_threshold()) {
            //vertical is an option. No l2 set issues for A
            if (this->is_a_nt_) {
                // horizontal uses precopy
                return false;
            }
        }
        return true;
    } else {
        // both are vertical or both are horizontal
        // Pick by l2 reuse - the one with the largest m/n_chunk
        // One of m_chunk_size_ and n_chunk_size is always 1
        return this->m_chunk_size_ * this->n_chunk_size_
                > other.m_chunk_size_ * other.n_chunk_size_;
    }
}

dim_t matmul_amx_blocking_params_macro_t::calc_k_blk(size_t l1_dim) {
    //Assuming 2x2 decomposition
    const size_t c_tiles = m_decomposition * n_decomposition * acc_dt_sz;
    const size_t d_tiles = m_decomposition
            * rnd_up(n_decomposition * c_dt_sz,
                    64); // rounded up to cache line size
    const size_t available_space_in_l1
            = L1_threshold() - (c_tiles * 2 + d_tiles);

    const dim_t largest_k = available_space_in_l1 / (l1_dim * gemm_dt_sz);
    const dim_t largest_k_tiles = largest_k / this->k_tmul;
    const dim_t k_tiles = div_up(K, this->k_tmul);
    const dim_t k_per_thread_tiles = div_up(k_tiles, nthr_k_);
    const dim_t num_K_blocks = div_up(k_per_thread_tiles, largest_k_tiles);
    return nstl::min(
            (dim_t)(div_up(k_per_thread_tiles, num_K_blocks) * this->k_tmul),
            K);
}

std::set<dim_t> matmul_amx_blocking_params_macro_t::blk_candidates(
        dim_t dim_per_thread, dim_t decomposition) {
    dim_t num_inner_blocks = div_up(dim_per_thread, decomposition);
    std::set<dim_t> dim_set;
    for (int num_groups = 1; num_groups <= num_inner_blocks; ++num_groups) {
        dim_t group_size = div_up(num_inner_blocks, num_groups);
        dim_set.insert(group_size);
    }

    return dim_set;
}

size_t matmul_amx_blocking_params_macro_t::l2_matrix_usage(size_t k_chunk_size,
        size_t m_or_n_blk, size_t k_blk, bool is_horizontal) {
    int decomposition = is_horizontal ? m_decomposition : n_decomposition;
    int l1_matrix_size = 2 * decomposition
            * nstl::min(k_blk * k_chunk_size, (size_t)k_per_thread)
            * gemm_dt_sz; // 2 for prefetch
    int l2_matrix_size = m_or_n_blk
            * nstl::min(k_blk * k_chunk_size, (size_t)k_per_thread)
            * gemm_dt_sz;
    int c_size = 2 * decomposition * m_or_n_blk
            * acc_dt_sz; // keep 2 c strips just to avoid evicting a
    return l1_matrix_size + l2_matrix_size + c_size;
}

size_t matmul_amx_blocking_params_macro_t::l2_matrix_and_c_usage(
        size_t k_chunk_size, size_t m_or_n_blk, size_t k_blk,
        bool is_horizontal) {
    size_t per_thread_for_l1_matrix
            = is_horizontal ? m_per_thread : n_per_thread;
    int l1_matrix_size = 2 * per_thread_for_l1_matrix
            * nstl::min(k_blk * k_chunk_size, (size_t)k_per_thread)
            * gemm_dt_sz; // 2x factor to make sure C is fresher than A,B in LRU
    int l2_matrix_size = 2 * m_or_n_blk
            * nstl::min(k_blk * k_chunk_size, (size_t)k_per_thread)
            * gemm_dt_sz; // 2x factor to make sure C is fresher than A,B in LRU
    int c_size
            = per_thread_for_l1_matrix * m_or_n_blk * acc_dt_sz; // keep c in l2
    return l1_matrix_size + l2_matrix_size + c_size;
}

int matmul_amx_blocking_params_macro_t::bw(size_t m_blk, size_t k_chunk_size,
        size_t k_blk, size_t n_blk, bool is_horizontal) {
    int a_bw = m_blk * nstl::min(k_blk * k_chunk_size, (size_t)k_per_thread)
            * gemm_dt_sz;
    int b_bw = n_blk * nstl::min(k_blk * k_chunk_size, (size_t)k_per_thread)
            * gemm_dt_sz;
    int c_bw;

    if ((l2_matrix_and_c_usage(k_chunk_size, is_horizontal ? n_blk : m_blk,
                 k_blk, is_horizontal)
                        < L2_threshold()
                || (dim_t)nstl::min(k_blk * k_chunk_size, (size_t)k_per_thread)
                        == K)
            && nthr_k_ == 1) {
        c_bw = 0;
    } else {
        c_bw = m_blk * n_blk * acc_dt_sz;
    }
    return a_bw + b_bw + c_bw;
}

int matmul_amx_blocking_params_macro_t::compute(
        size_t m_blk, size_t k_chunk_size, size_t k_blk, size_t n_blk) const {
    return m_blk * nstl::min(k_blk * k_chunk_size, (size_t)k_per_thread)
            * n_blk;
}

float matmul_amx_blocking_params_macro_t::ratio(size_t m_blk,
        size_t k_chunk_size, size_t k_blk, size_t n_blk, bool is_horizontal) {
    return static_cast<float>(compute(m_blk, k_chunk_size, k_blk, n_blk))
            / bw(m_blk, k_chunk_size, k_blk, n_blk, is_horizontal);
}

float matmul_amx_blocking_params_macro_t::evaluate_single_core_blocking(
        size_t k_chunk_size, size_t m_or_n_blk, size_t k_blk,
        bool is_horizontal) {
    if (l2_matrix_usage(k_chunk_size, m_or_n_blk, k_blk, is_horizontal)
            <= L2_threshold()) {
        size_t m_blk, n_blk;
        if (is_horizontal) {
            m_blk = m_decomposition;
            n_blk = m_or_n_blk;
        } else {
            m_blk = m_or_n_blk;
            n_blk = n_decomposition;
        }
        float ratio_score
                = ratio(m_blk, k_chunk_size, k_blk, n_blk, is_horizontal);
        return ratio_score;
    }
    return 0;
}

void matmul_amx_blocking_params_macro_t::set_tmul_sizes() {
    this->m_tmul = determine_tmul_size(this->m_per_thread, 16);
    this->n_tmul = 16; // B blocked layout is a multiply of 16
    this->k_tmul = nstl::min((size_t)wei_k_blk, (size_t)K);
}

void matmul_amx_blocking_params_macro_t::set_decomposition() {
    m_decomposition = nstl::min((size_t)m_per_thread, 2 * m_tmul);
    n_decomposition = nstl::min((size_t)n_per_thread, 2 * n_tmul);
}

bool matmul_amx_blocking_params_macro_t::is_horizontal_selected(
        bool horizontal_not_possible, bool vertical_not_possible,
        size_t best_m_v, size_t best_k_v, size_t k_blk_v) const {
    // Choose between horizontal and vertical

    bool is_horizontal_local;

    if (horizontal_not_possible) {
        is_horizontal_local = false;
    } else if (vertical_not_possible) {
        is_horizontal_local = true;
    } else if ((size_t)m_per_thread < m_tmul * 2) {
        // There are not enough tiles in M direction to go vertical
        is_horizontal_local = true;
    } else if ((size_t)n_per_thread < n_tmul * 2) {
        // There are not enough tiles in N direction to go horizontal
        is_horizontal_local = false;
    } else if (m_per_thread >= n_per_thread) {
        //choose horizontal
        is_horizontal_local = true;
    } else {
        //choose vertical
        is_horizontal_local = false;
    }
    return is_horizontal_local;
}

bool matmul_amx_blocking_params_macro_t::set_blocking_parameters() {
    set_tmul_sizes();
    set_decomposition();

    std::set<dim_t> m_candidates
            = blk_candidates(m_per_thread, m_decomposition);
    std::set<dim_t> n_candidates
            = blk_candidates(n_per_thread, n_decomposition);
    dim_t best_k_h, best_n_h;
    dim_t best_m_v, best_k_v;
    float best_score_h = 0, best_score_v = 0;
    bool horizontal_not_possible = false;
    bool vertical_not_possible = false;

    auto calc_horizontal = [&](size_t k_blk_h, dim_t min_k_chunk_size = 0) {
        if (rnd_up(m_per_thread, m_decomposition) * (nthr_m_ - 1) > (size_t)M) {
            horizontal_not_possible = true;
        } else if (rnd_up(k_per_thread, k_blk_h) * (nthr_k_ - 1) > (size_t)K) {
            // early exit: There is no possible division of work for nthr_k threads
            horizontal_not_possible = true;
        } else {
            std::set<dim_t> k_candidates_h
                    = blk_candidates(k_per_thread, k_blk_h);
            best_n_h = 0;
            for (std::set<dim_t>::reverse_iterator it_n = n_candidates.rbegin();
                    it_n != n_candidates.rend(); it_n++) {
                for (std::set<dim_t>::reverse_iterator it_k
                        = k_candidates_h.rbegin();
                        it_k != k_candidates_h.rend(); it_k++) {
                    float cur_score = evaluate_single_core_blocking(
                            *it_k, *it_n * n_decomposition, k_blk_h, true);
                    if (cur_score > best_score_h && *it_k >= min_k_chunk_size) {
                        best_score_h = cur_score;
                        best_k_h = *it_k;
                        best_n_h = *it_n;
                    }
                }
            }

            if (rnd_up(n_per_thread, best_n_h * n_decomposition) * (nthr_n_ - 1)
                    > (size_t)N) {
                horizontal_not_possible = true;
            }
            if (rnd_up(k_per_thread, best_k_h * k_blk_h) * (nthr_k_ - 1)
                    > (size_t)K) {
                // There is not enough work for nthr_k threads
                horizontal_not_possible = true;
            }
        }
    };
    // Calculate best score for horizontal traversal
    dim_t k_blk_h = calc_k_blk(m_decomposition);
    calc_horizontal(k_blk_h);

    auto calc_vertical = [&](size_t k_blk_v) {
        if (rnd_up(n_per_thread, n_decomposition) * (nthr_n_ - 1) > (size_t)N) {
            vertical_not_possible = true;
        } else if (rnd_up(k_per_thread, k_blk_v) * (nthr_k_ - 1) > (size_t)K) {
            // early exit: There is no possible division of work for nthr_k threads
            vertical_not_possible = true;
        } else {
            // Calculate best score for vertical traversal
            std::set<dim_t> k_candidates_v
                    = blk_candidates(k_per_thread, k_blk_v);
            for (std::set<dim_t>::reverse_iterator it_m = m_candidates.rbegin();
                    it_m != m_candidates.rend(); it_m++) {
                for (std::set<dim_t>::reverse_iterator it_k
                        = k_candidates_v.rbegin();
                        it_k != k_candidates_v.rend(); it_k++) {
                    float cur_score = evaluate_single_core_blocking(
                            *it_k, *it_m * m_decomposition, k_blk_v, false);
                    if (cur_score > best_score_v) {
                        best_score_v = cur_score;
                        best_k_v = *it_k;
                        best_m_v = *it_m;
                    }
                }
            }

            if (rnd_up(m_per_thread, best_m_v * m_decomposition) * (nthr_m_ - 1)
                    > (size_t)M) {
                vertical_not_possible = true;
            }
            if (rnd_up(k_per_thread, best_k_v * k_blk_v) * (nthr_k_ - 1)
                    > (size_t)K) {
                // There is not enough work for nthr_k threads
                vertical_not_possible = true;
            }
            size_t l2_util_v;

            if (!vertical_not_possible) {
                // Figure out if vertical is an option wrt l2 usage
                l2_util_v = l2_matrix_and_c_usage(
                        best_k_v, best_m_v, k_blk_v, false);
                if (l2_util_v > L2_threshold()) {
                    l2_util_v = l2_matrix_usage(
                            best_k_v, best_m_v, k_blk_v, false);
                }
            }
            bool repeat_loop_over_k = div_up(K, k_blk_v * best_k_v) != 1;
            bool critical_l2_set_issues_a
                    = div_up((size_t)K, k_blk_v * best_k_v) != nthr_k_
                    || (size_t)((l2_util_v * nthr_k_)) >= L2_threshold();

            if (repeat_loop_over_k && critical_l2_set_issues_a)
                vertical_not_possible = true;
        }
    };

    dim_t k_blk_v = calc_k_blk(n_decomposition);
    calc_vertical(k_blk_v);

    if (vertical_not_possible && horizontal_not_possible) { return false; }

    is_horizontal = is_horizontal_selected(horizontal_not_possible,
            vertical_not_possible, best_m_v, best_k_v, k_blk_v);

    if (is_horizontal) {
        size_t l1_eff_factor = div_up(K, k_blk_h);
        // this works for m > 32 in this case k_blk_h << 4096 =~ 512
        // for m <= the problem is heavily memory bound ==> don't care about the l1 and work completely from the l2

        size_t a_l1 = k_blk_h * m_decomposition * gemm_dt_sz;
        size_t c_l1 = n_decomposition * m_decomposition * acc_dt_sz;
        size_t d_post = m_decomposition * rnd_up(n_decomposition * c_dt_sz, 64);
        is_a_nt_ = false;
        is_b_nt_ = true;

        if (k_blk_h < K
                && l1_eff_factor * a_l1 + 2 * c_l1 + d_post > L1_threshold()) {
            best_score_h = 0;
            // Calculate k_blk_h and n_blk_h that can fit in the l2 when k_blk is wei_k_blk
            calc_horizontal(wei_k_blk, k_blk_h / wei_k_blk);
            // give up on the l1.
            k_blk_h = nstl::min(wei_k_blk * best_k_h, K);
            best_k_h = 1;
            is_a_nt_ = true;
            //todo: revive after precopy implementation
            //            need_buf_a_ = false;
            need_prefetch = false;
        } else {
            //todo: revive after precopy implementation
            //            need_buf_a_ = false;
            need_prefetch = true;
        }

        k_blk_ = k_blk_h;
        k_chunk_size_ = best_k_h;
        n_blk_ = nstl::min(best_n_h * n_decomposition, N);
        n_chunk_size_ = 1;
        m_blk_ = m_decomposition;
        m_chunk_size_ = div_up(m_per_thread, m_blk_);
    } else {
        k_blk_ = k_blk_v;
        k_chunk_size_ = best_k_v;
        n_blk_ = n_decomposition;
        n_chunk_size_ = div_up(n_per_thread, n_blk_);
        m_blk_ = nstl::min(best_m_v * m_decomposition, M);
        m_chunk_size_ = 1;
        is_a_nt_ = true;
        is_b_nt_ = false;
        need_prefetch = true;
    }

    extendable_k_ = K % data_type_vnni_granularity(wei_dt) != 0;

    brgemm_batch_size_ = 1;

    n_chunk_elems_ = nstl::min(n_per_thread, n_blk_ * n_chunk_size_);
    m_chunk_elems_ = nstl::min(m_per_thread, m_blk_ * m_chunk_size_);
    k_chunk_elems_ = nstl::min(k_per_thread, k_blk_ * k_chunk_size_);

    set_nt_ = true;

    current_lda_ = get_actual_lda();

    // Need a temp C buffer if a BRGEMM creates partial results
    need_buf_c_ = (nthr_k_ != 1) || (k_blk_ != K);

    efficiency_score_ = calculate_blocking_scores();

    return true;
}

void matmul_amx_blocking_params_macro_t::set_core_divs(
        int nthr_b, int nthr_m, int nthr_k, int nthr_n) {
    nthr_b_ = nthr_b;
    nthr_m_ = nthr_m;
    nthr_k_ = nthr_k;
    nthr_n_ = nthr_n;
    m_per_thread = div_up(M, nthr_m_);
    k_per_thread = div_up(K, nthr_k_);
    n_per_thread = div_up(N, nthr_n_);
    b_per_thread = div_up(this->batch, nthr_b_);

    nthr_mnb_ = nthr_ / nthr_k_;
}

void matmul_amx_blocking_params_micro_t::find_best_blocking(
        const brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils,
        matmul_amx_blocking_params_t &best_blocking) {

    matmul_amx_blocking_params_micro_t current_blocking(bgmmc);

    const int min_k_per_thread = 1024;
    const int max_k_parallel_work
            = div_up(static_cast<int>(bgmmc.K), min_k_per_thread);
    const bool is_amx_xf16 = bgmmc.is_amx
            && (bm_conf_utils.is_bf16() || bm_conf_utils.is_f16()
                    || bm_conf_utils.is_f32_f16() || bm_conf_utils.is_f32_bf16()
                    || bm_conf_utils.is_bf32()
                    || bm_conf_utils.is_bf16_with_int_wei()
                    || bm_conf_utils.is_f16_with_int_wei());
    const bool is_amx_int8 = bgmmc.is_amx && bm_conf_utils.is_int8();

    const bool runtime_dims
            = bgmmc.is_runtime_M || bgmmc.is_runtime_N || bgmmc.is_runtime_K;
    const int max_nthr_k = !runtime_dims && is_amx_xf16 && bgmmc.batch == 1
            ? nstl::min(saturate(1, 7, bgmmc.nthr / 8), max_k_parallel_work)
            : 1;
    int iter = 0;
    const int runtime_M_chunk = bgmmc.lda_big_pow2() ? 2 : 4;
    const int runtime_N_chunk = 2;

    // Disable skip configuration due to regressions for some cases.
    const bool disable_skip_config = bgmmc.M == 4
            && utils::one_of(true, bgmmc.N == 4096 && bgmmc.K == 4096,
                    bgmmc.N == 11008 && bgmmc.K == 4096,
                    bgmmc.N == 4096 && bgmmc.K == 11008);

    for (int nthr_k = 1; nthr_k <= max_nthr_k; nthr_k++) {
        int nthr_bmn = bgmmc.nthr / nthr_k;

        int num_M_blk = bgmmc.is_runtime_M ? 1 : div_up(bgmmc.M, bgmmc.M_blk);
        int num_N_blk = bgmmc.is_runtime_N ? 1 : div_up(bgmmc.N, bgmmc.N_blk);
        int k_parallel_work = nstl::min(max_k_parallel_work, nthr_k);
        int num_parallel_work
                = bgmmc.batch * num_M_blk * num_N_blk * k_parallel_work;
        const bool a_lot_of_parallel_work_lvl2
                = num_parallel_work > 16 * bgmmc.nthr;
        const bool low_parallelism
                = static_cast<float>(num_parallel_work) < 1.5f * bgmmc.nthr;
        const bool maybe_low_blocking
                = is_amx_int8 && bm_conf_utils.maybe_low_brg_blocking();
        const int min_M_blk = !bgmmc.is_runtime_M
                        && (maybe_low_blocking || low_parallelism)
                        && bgmmc.M_blk > 32
                ? div_up(bgmmc.M_blk, 2)
                : bgmmc.M_blk;
        const int min_N_blk = !bgmmc.is_runtime_N && low_parallelism
                        && is_amx_xf16 && !bm_conf_utils.check_n_blk_fixed()
                        && bgmmc.N_blk > 32 && !runtime_dims
                ? 32
                : bgmmc.N_blk;
        const int desired_M_chunk = bgmmc.is_runtime_M
                ? runtime_M_chunk
                : nstl::min(4, num_M_blk);
        const int desired_N_chunk = bgmmc.is_runtime_N
                ? runtime_N_chunk
                : nstl::min(a_lot_of_parallel_work_lvl2 ? 6 : 4, num_N_blk);

        std::unordered_set<int> mblk_candidates;
        for (int m_blk = bgmmc.M_blk; m_blk >= min_M_blk;
                m_blk = m_blk > 1 ? div_up(m_blk, 2) : m_blk - 1) {
            if (IMPLICATION(maybe_low_blocking, m_blk != bgmmc.M_blk))
                mblk_candidates.insert(m_blk);
        }

        if (!bgmmc.is_runtime_M && bgmmc.M > 16) {
            // Add multiple of 16 M block sizes for consideration
            const int mul16_m_blk_max
                    = nstl::min(rnd_dn(static_cast<int>(bgmmc.M), 16), 64);
            const int mul16_m_blk_min = rnd_up(min_M_blk, 16);
            for (int m_blk = mul16_m_blk_max; m_blk >= mul16_m_blk_min;
                    m_blk -= 16) {
                mblk_candidates.insert(m_blk);
            }
        }

        bool found_best_blocking = false;
        for_(int n_blk = bgmmc.N_blk; n_blk >= min_N_blk; n_blk -= 16)
        for_(int m_blk : mblk_candidates)
        for_(int n_ch_sz = desired_N_chunk; n_ch_sz >= 1; n_ch_sz--)
        for (int m_ch_sz = desired_M_chunk; m_ch_sz >= 1; m_ch_sz--, iter++) {
            current_blocking.set_blocking_parameters(
                    nthr_k, n_blk, n_ch_sz, m_blk, m_ch_sz);

            float cur_score = current_blocking.get_blocking_scores();
            float bst_score = best_blocking.get_blocking_scores();

            int m_chunks = div_up(bgmmc.M, m_blk * m_ch_sz);
            int n_chunks = div_up(bgmmc.N, n_blk * n_ch_sz);
            int work_amount = bgmmc.batch * m_chunks * n_chunks;

            bool skip_config = work_amount < nthr_bmn * 3
                    && work_amount % nthr_bmn != 0 && max_nthr_k == 1;
            if (skip_config && !disable_skip_config) continue;

            if (cur_score > bst_score) {
                best_blocking = current_blocking;
                found_best_blocking = true;
            }
        }

        if (!found_best_blocking) {
            current_blocking.set_blocking_parameters(
                    nthr_k, min_N_blk, 1, min_M_blk, 1);

            float cur_score = current_blocking.get_blocking_scores();
            float bst_score = best_blocking.get_blocking_scores();
            if (cur_score > bst_score) best_blocking = current_blocking;
        }
    }
}

void matmul_amx_blocking_params_micro_t::update_k_blocking_dependent_params() {
    k_chunk_elems_ = k_blk_ * k_chunk_size_ * brgemm_batch_size_;
    current_lda_ = get_actual_lda();
    need_buf_c_ = is_buffer_c_required();
}

void matmul_amx_blocking_params_micro_t::set_blocking_parameters(
        int nthr_k, int n_blk, int n_chunk_size, int m_blk, int m_chunk_size) {
    nthr_k_ = nstl::max(1, nthr_k);
    nthr_mnb_ = nthr / nthr_k_;
    nthr_ = nthr_mnb_ * nthr_k_;
    n_blk_ = n_blk;
    n_chunk_size_ = n_chunk_size;
    m_blk_ = m_blk;
    m_chunk_size_ = m_chunk_size;

    if (one_of(0, n_blk_, n_chunk_size_, m_blk_, m_chunk_size_)) {
        k_blk_ = k_chunk_size_ = k_chunk_elems_ = brgemm_batch_size_ = 0;
        efficiency_score_ = 0.0f;
        return;
    }

    n_chunk_elems_ = n_blk_ * n_chunk_size_;
    m_chunk_elems_ = m_blk_ * m_chunk_size_;

    if (K < wei_k_blk) {
        k_blk_ = is_amx ? rnd_up(K, required_k_granularity) : K;
        brgemm_batch_size_ = 1;
    } else {
        dim_t k_per_thr = div_up(K, nthr_k_);
        k_blk_ = nstl::min(rnd_up(k_per_thr, required_k_granularity),
                static_cast<dim_t>(wei_k_blk));
        const dim_t num_k_blk = div_up(K, k_blk_);
        const dim_t num_k_blk_per_thread = div_up(num_k_blk, nthr_k_);
        brgemm_batch_size_ = num_k_blk_per_thread;

        auto chunk_sz = calculate_chunk_memory_size();
        const dim_t div_min = chunk_sz / L2_threshold();
        const dim_t div_max = div_up(chunk_sz, L2_threshold());
        // for big pow2 lda prefer to increase area of linear memory access
        const dim_t adjust_k_divisor_threshold = lda_big_pow2() ? 2 : 0;
        // adjust k blocking values to fit into L2 cache
        if (div_min > adjust_k_divisor_threshold && brgemm_batch_size_ > 1) {
            const auto kc1 = nstl::max(
                    brgemm_batch_size_ / div_min, static_cast<dim_t>(1));
            const auto kc2 = div_up(brgemm_batch_size_, div_max);
            const auto tail1 = num_k_blk_per_thread % kc1;
            const auto tail2 = num_k_blk_per_thread % kc2;
            // prefer adjusted chunk size with more equal work distribution
            // across iterations
            brgemm_batch_size_
                    = IMPLICATION(tail1 == 0 || tail2 < tail1, tail2 == 0)
                    ? kc2
                    : kc1;
        }

        k_chunk_elems_ = k_blk_ * brgemm_batch_size_ * k_chunk_size_;
        dim_t brgemm_k_elems = k_blk_ * brgemm_batch_size_;
        const dim_t current_k_tail = K % k_blk_;

        // TODO: review extendable_k_ condition to cover more cases
        extendable_k_ = (K % wei_k_blk != 0) && (brgemm_k_elems > wei_k_blk)
                && wei_zp_type == none && !use_buffer_a
                && !packed_sparse_weights;

        if (extendable_k_) {
            if (brgemm_k_elems >= K) {
                k_blk_ = K;
                k_chunk_size_ = 1;
            } else {
                k_blk_ = brgemm_k_elems;
                k_chunk_size_ = 1;
            }
        } else if (current_k_tail == 0
                && K % (k_blk_ * brgemm_batch_size_) == 0) {
            k_blk_ = brgemm_k_elems;
            brgemm_batch_size_ = 1;
        } else if (nthr_k_ == 1
                && K == k_blk_ * brgemm_batch_size_ + current_k_tail) {
            k_blk_ = brgemm_k_elems;
            brgemm_batch_size_ = 2;
        }
    }
    need_buf_a_
            = use_buffer_a || (!extendable_k_ && K % required_k_granularity);

    blocking_chunk_mem_size_ = calculate_chunk_memory_size();

    efficiency_score_ = calculate_blocking_scores();
}

// returns score for current blocking parameters' values in range [0, 1]
// for parallel work over threads distribution score. Maximum scores - when
// all threads have the same work amount w/o tails
float matmul_amx_blocking_params_micro_t::get_thread_balance_scores() {
    assert(!(is_runtime_M && is_runtime_N)
            && "single runtime dim is supported");
    // Ignore M sizes in thread balance computation as actual M size is unknown
    if (is_runtime_M) return (float)N / rnd_up(N, n_chunk_elems_);
    // Ignore N sizes in thread balance computation as actual N size is unknown
    if (is_runtime_N) return (float)M / rnd_up(M, m_chunk_elems_);

    const dim_t num_M_chunks = div_up(M, m_chunk_elems_);
    const dim_t num_N_chunks = div_up(N, n_chunk_elems_);
    float mnb_parallel_score = batch * ((float)M / m_chunk_elems_)
            * ((float)N / n_chunk_elems_)
            / rnd_up(batch * num_M_chunks * num_N_chunks, nthr_mnb_)
            * nthr_mnb_;
    float k_parallel_score = 1.0f;
    if (nthr_k_ > 1) {
        const dim_t num_K_chunks = div_up(K, k_chunk_elems_);
        const float parallel_reduction_penalty = 0.8f;
        k_parallel_score = parallel_reduction_penalty
                * ((float)K / k_chunk_elems_) / rnd_up(num_K_chunks, nthr_k_)
                * nthr_k_;
    }

    return mnb_parallel_score * k_parallel_score / nthr;
}

// returns score for current blocking parameters' values in range [0, 1]
// for copied data reusage
float matmul_amx_blocking_params_micro_t::get_copied_data_reusage_scores() {
    const dim_t effective_m_chunk_sz = 64 * 4;
    const dim_t desired_M_chunk_size = is_runtime_M
            ? effective_m_chunk_sz
            : nstl::min(M, effective_m_chunk_sz);
    const dim_t effective_n_chunk_sz = 64 * (need_buf_a_ ? 4 : 1);
    const dim_t desired_N_chunk_size = is_runtime_N
            ? effective_n_chunk_sz
            : nstl::min(N, effective_n_chunk_sz);
    const float coef_M = nstl::min(
            static_cast<float>(m_chunk_elems_) / desired_M_chunk_size, 1.0f);
    const float coef_N = nstl::min(
            static_cast<float>(n_chunk_elems_) / desired_N_chunk_size, 1.0f);
    return 0.5f * (coef_M + coef_N);
}

// returns score for current blocking parameters' values in range [0, 1]
// for L2 utilization
float matmul_amx_blocking_params_micro_t::get_L2_utilization_scores() const {
    const float relative_difference_with_L2
            = fabsf((float)L2_threshold() - blocking_chunk_mem_size_)
            / nstl::max(L2_threshold(), blocking_chunk_mem_size_);
    return 1.0f - relative_difference_with_L2;
}

// returns score for current blocking parameters' values in range [0, 1]
// consists of 3 parts with its own weights:
// 	1) parallel work over threads distribution score
// 	2) L2 utilization score
// 	3) copied data re-usage score
float matmul_amx_blocking_params_micro_t::calculate_blocking_scores() {
    if (one_of(0, n_blk_, n_chunk_size_, m_blk_, m_chunk_size_, k_blk_,
                brgemm_batch_size_))
        return 0.0f;

    const float nthr_coeff = nstl::min(nthr, 100);
    const float reusage_factor = 1.0f;
    // for runtume M the actual size is unknown, use independent on num_threads
    // balance factors
    const float balance_factor
            = is_runtime_M ? 1.0f : (nthr_coeff - 1.0f) / nthr_coeff;
    const float cache_utilization_factor
            = is_runtime_M ? 1.0f : 1.0f / nthr_coeff;

    float scores = cache_utilization_factor * get_L2_utilization_scores()
            + reusage_factor * get_copied_data_reusage_scores();
    if (balance_factor > 0.0f)
        scores += balance_factor * get_thread_balance_scores();
    return scores
            / (reusage_factor + balance_factor + cache_utilization_factor);
}

size_t matmul_amx_blocking_params_micro_t::calculate_chunk_memory_size() {
    update_k_blocking_dependent_params();

    const size_t A_chunk_sz = a_dt_sz * k_chunk_elems_ * m_chunk_elems_;
    const size_t A_buf_sz = need_buf_a_
            ? tr_a_dt_sz * current_lda_ * brgemm_batch_size_ * m_chunk_elems_
            : 0;
    const size_t B_chunk_sz = b_dt_sz * k_chunk_elems_ * n_chunk_elems_;
    const size_t B_buf_sz
            = use_buffer_b ? tr_b_dt_sz * n_blk_ * k_chunk_elems_ : 0;
    const size_t C_chunk_sz = c_dt_sz * m_chunk_elems_ * n_chunk_elems_;
    const size_t C_buf_sz
            = need_buf_c_ ? acc_dt_sz * m_chunk_elems_ * n_chunk_elems_ : 0;
    return A_chunk_sz + A_buf_sz + B_chunk_sz + B_buf_sz + C_chunk_sz
            + C_buf_sz;
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
