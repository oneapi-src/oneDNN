/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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
#include <tuple>
#include <utility>
#include "common/dnnl_thread.hpp"
#include "cpu/rnn/rnn_utils.hpp"
#include "cpu/x64/rnn/rnn_brgemm_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace rnn_brgemm_utils {

namespace {

x64::cpu_isa_t brgemm_calc_isa(
        const cpu::rnn_utils::rnn_conf_t &rnn, dim_t K1, dim_t K2);
std::pair<dim_t, dim_t> brgemm_calc_k_block(
        const cpu::rnn_utils::rnn_conf_t &rnn, dim_t K1, dim_t K2, dim_t M,
        dim_t n_block, alg_kind_t cell_kind, dim_t src_layer_type_size,
        dim_t As, dim_t Bs, dim_t Cs, dim_t l2_cache_size, x64::cpu_isa_t isa);
std::pair<dim_t, dim_t> brgemm_calc_k_block_amx(
        dim_t K1, dim_t K2, bool is_int8);
std::pair<dim_t, dim_t> brgemm_calc_k_block_vanilla_rnn(dim_t K1, dim_t K2,
        dim_t M, dim_t n_block, dim_t src_layer_type_size, dim_t As, dim_t Bs,
        dim_t Cs, dim_t l2_cache_size, bool is_xf16, x64::cpu_isa_t isa);

dim_t brgemm_calc_m_block(alg_kind_t cell_kind, prop_kind_t aprop, dim_t nthr,
        dim_t M, dim_t N_blocks, bool is_f32, bool is_int8_amx,
        bool is_xf16_amx, float work_by_N, dim_t As, dim_t Bs, dim_t Cs,
        dim_t l2_cache_size);
dim_t brgemm_calc_m_block_vanilla_rnn(dim_t nthr, dim_t M, dim_t N_blocks,
        bool is_int8_amx, bool is_xf16_amx, float work_by_N, dim_t As, dim_t Bs,
        dim_t Cs, dim_t l2_cache_size);
dim_t brgemm_calc_m_block_lstm(dim_t nthr, dim_t M, dim_t N_blocks, bool is_f32,
        bool is_int8_amx, bool is_xf16_amx, float work_by_N, dim_t As, dim_t Cs,
        dim_t l2_cache_size);
dim_t adjust_m_block_lstm(dim_t nthr, dim_t M, dim_t N_blocks, bool is_int8_amx,
        bool is_xf16_amx);

bool has_amx_support(data_type_t dt) {
    switch (dt) {
        case data_type::u8:
        case data_type::s8:
        case data_type::bf16: return x64::mayiuse(x64::avx512_core_amx);
        case data_type::f16: return x64::mayiuse(x64::avx512_core_amx_fp16);
        default: break;
    }
    return false;
}

x64::cpu_isa_t brgemm_calc_isa(
        const cpu::rnn_utils::rnn_conf_t &rnn, dim_t K1, dim_t K2) {

    if (has_amx_support(rnn.cell_dt)) {
        const dim_t padding = data_type_vnni_granularity(rnn.cell_dt);
        const auto result
                = brgemm_calc_k_block_amx(K1, K2, rnn.is_cell_dt_int8());
        const auto k1_block_amx = result.first;
        const auto k2_block_amx = result.second;
        const auto k1_block_tail = K1 % k1_block_amx;
        const auto k2_block_tail = K2 % k2_block_amx;
        const bool amx_block_invalid = k1_block_tail % padding
                || k2_block_tail % padding || k1_block_amx % padding
                || k2_block_amx % padding;

        if (!amx_block_invalid)
            return rnn.is_cell_dt_f16() ? x64::avx512_core_amx_fp16
                                        : x64::avx512_core_amx;
    }

    if (rnn.is_cell_dt_int8()) {
        return x64::avx512_core_vnni;
    } else if (rnn.is_cell_dt_bf16()) {
        return x64::avx512_core_bf16;
    } else if (rnn.is_cell_dt_f16()) {
        return isa_undef;
    } else { // f32
        return utils::map(true, x64::isa_undef, mayiuse(avx512_core),
                avx512_core, mayiuse(avx2), avx2);
    }
}

std::pair<dim_t, dim_t> brgemm_calc_k_block(
        const cpu::rnn_utils::rnn_conf_t &rnn, dim_t K1, dim_t K2, dim_t M,
        dim_t n_block, alg_kind_t cell_kind, dim_t src_layer_type_size,
        dim_t As, dim_t Bs, dim_t Cs, dim_t l2_cache_size, x64::cpu_isa_t isa) {

    if (is_superset(isa, avx512_core_amx))
        return brgemm_calc_k_block_amx(K1, K2, rnn.is_cell_dt_int8());
    else if (cell_kind == alg_kind::vanilla_rnn)
        return brgemm_calc_k_block_vanilla_rnn(K1, K2, M, n_block,
                src_layer_type_size, As, Bs, Cs, l2_cache_size,
                rnn.is_cell_dt_xf16(), isa);

    return std::make_pair(K1, K2);
}

std::pair<dim_t, dim_t> brgemm_calc_k_block_amx(
        dim_t K1, dim_t K2, bool is_int8) {
    const bool is_amx_int8 = is_int8 && x64::mayiuse(x64::avx512_core_amx);
    const dim_t max_row_width = is_amx_int8 ? 64 : 32;

    dim_t k1_block = nstl::min(K1, max_row_width);
    dim_t k2_block = nstl::min(K2, max_row_width);

    if (k1_block <= K1 || k2_block <= K2) {
        const dim_t t_k_block = nstl::min(k1_block, k2_block);
        k2_block = k1_block = t_k_block;
    }

    return std::make_pair(k1_block, k2_block);
}

std::pair<dim_t, dim_t> brgemm_calc_k_block_vanilla_rnn(dim_t K1, dim_t K2,
        dim_t M, dim_t n_block, dim_t src_layer_type_size, dim_t As, dim_t Bs,
        dim_t Cs, dim_t l2_cache_size, bool is_xf16, x64::cpu_isa_t isa) {

    //Heuristics experimentally selected.
    // For avx2 (i.e., non-avx512_core) the value is chosen while tuning
    // deepspeech model. It might be applicable for avx512_core as well (TODO)
    const float l2_occupancy
            = is_superset(isa, x64::avx512_core) ? 0.25f : 0.75f;
    const bool should_adjust_by_l2 = static_cast<float>(As + Bs + Cs)
            >= l2_occupancy * static_cast<float>(l2_cache_size);
    dim_t k1_block = K1;
    dim_t k2_block = K2;

    if (should_adjust_by_l2) {
        int block_size = (l2_cache_size * l2_occupancy)
                / ((M + n_block) * src_layer_type_size);

        if (is_xf16) {
            // due to weights format ldgOI32o2i block_size should be even
            block_size -= (block_size % 2);
            block_size = nstl::max(block_size, 0);
        }
        if (block_size) {
            k1_block = nstl::min(K1, static_cast<dim_t>(block_size));
            k2_block = nstl::min(K2, static_cast<dim_t>(block_size));
        }
    }

    return std::make_pair(k1_block, k2_block);
}

dim_t brgemm_calc_m_block(alg_kind_t cell_kind, prop_kind_t aprop, dim_t nthr,
        dim_t M, dim_t N_blocks, bool is_f32, bool is_int8_amx,
        bool is_xf16_amx, float work_by_N, dim_t As, dim_t Bs, dim_t Cs,
        dim_t l2_cache_size) {
    if (cell_kind == alg_kind::vanilla_rnn
            || (cell_kind == alg_kind::vanilla_lstm
                    && aprop == prop_kind::backward))
        return brgemm_calc_m_block_vanilla_rnn(nthr, M, N_blocks, is_int8_amx,
                is_xf16_amx, work_by_N, As, Bs, Cs, l2_cache_size);
    else
        return brgemm_calc_m_block_lstm(nthr, M, N_blocks, is_f32, is_int8_amx,
                is_xf16_amx, work_by_N, As, Cs, l2_cache_size);
}

dim_t brgemm_calc_m_block_vanilla_rnn(dim_t nthr, dim_t M, dim_t N_blocks,
        bool is_int8_amx, bool is_xf16_amx, float work_by_N, dim_t As, dim_t Bs,
        dim_t Cs, dim_t l2_cache_size) {

    //Heuristics experimentally selected.
    const float decimal_n_factor = work_by_N - std::floor(work_by_N);
    static constexpr float thread_balance_threashold = 0.9f;

    dim_t m_block = M;

    if (work_by_N < 1.0)
        return adjust_m_block_lstm(nthr, M, N_blocks, is_int8_amx, is_xf16_amx);
    else if (decimal_n_factor < thread_balance_threashold
            && decimal_n_factor != 0.0f) {

        const dim_t m_block_start = M / 2;
        const dim_t m_block_end = 8;

        float max_decimal_mn = 0.0f;
        dim_t best_candidate = m_block_start;
        bool found_best_solution = false;

        for (dim_t m_block_it = m_block_start; m_block_it >= m_block_end;
                m_block_it--) {
            if (M % m_block_it == 0) {
                const auto m_blocks = M / m_block_it;
                const auto work_by_MN
                        = static_cast<float>(m_blocks * N_blocks) / nthr;

                const float work_by_MN_decimal
                        = work_by_MN - std::floor(work_by_MN);

                static constexpr float tolerance = 0.01f;
                if (work_by_MN_decimal > (max_decimal_mn + tolerance)) {
                    best_candidate = m_block_it;
                    max_decimal_mn = work_by_MN_decimal;
                }

                if (work_by_MN_decimal >= thread_balance_threashold
                        || work_by_MN_decimal == 0.0f) {
                    m_block = m_block_it;
                    found_best_solution = true;
                    break;
                }
            }
        }

        if (!found_best_solution) {
            if ((decimal_n_factor < max_decimal_mn)
                    || (static_cast<float>(As)
                            > (0.5f * static_cast<float>(l2_cache_size)))) {
                m_block = best_candidate;
            }
        }
    }

    return m_block;
}

dim_t brgemm_calc_m_block_lstm(dim_t nthr, dim_t M, dim_t N_blocks, bool is_f32,
        bool is_int8_amx, bool is_xf16_amx, float work_by_N, dim_t As, dim_t Cs,
        dim_t l2_cache_size) {
    const bool adj_by_l2 = is_f32
            ? true
            : (static_cast<float>(As + Cs)
                    < 0.6 * static_cast<float>(l2_cache_size));

    if (work_by_N > 2.0 || (work_by_N > 1.0 && adj_by_l2))
        return M;
    else
        return adjust_m_block_lstm(nthr, M, N_blocks, is_int8_amx, is_xf16_amx);
}

dim_t adjust_m_block_lstm(dim_t nthr, dim_t M, dim_t N_blocks, bool is_int8_amx,
        bool is_xf16_amx) {

    const bool is_amx = is_int8_amx || is_xf16_amx;

    const dim_t max_m_blocks = (is_amx ? 1 : 4) * utils::div_up(nthr, N_blocks);
    const dim_t max_m_value = is_amx ? 64 : 24;
    const dim_t max_M
            = nstl::min(max_m_value, nstl::max((dim_t)1, M / max_m_blocks));
    const dim_t min_M = 4;

    dim_t m_block = 1;
    for (dim_t m = max_M; m >= min_M; m--)
        if (M % m == 0) {
            m_block = m;
            break;
        }
    if (m_block == 1) m_block = M;

    return m_block;
}

x64::cpu_isa_t adjust_isa_by_m_block(
        x64::cpu_isa_t current_isa, dim_t m_block, bool is_int8_amx) {
    /*
     * If we have m<4 TMUL and AVX512 vnni calculate the same number of
     * operation per instruction but TMUL is 2x slower for int8 in terms of
     * throughput.
     */
    if (is_int8_amx && m_block < 4) {
        if (x64::mayiuse(x64::avx512_core_amx)) return x64::avx512_core_amx;
    }

    return current_isa;
}

} // namespace

void rnn_brgemm_base_t::init_scratchpad(const cpu::rnn_utils::rnn_conf_t &rnn,
        memory_tracking::registrar_t &scratchpad, dim_t gemm_acc_type_size,
        dim_t gemm_acc_align) {

    using namespace memory_tracking::names;

    if (rnn.is_cell_amx()) {
        const auto m_block = rnn.merge_gemm_layer
                ? nstl::max(rnn.m_block, rnn.mlayermerged_block)
                : rnn.m_block;
        size_t n_elements = m_block * rnn.n_block;
        scratchpad.book(key_brgemm_primitive_buffer, rnn.nthr * n_elements,
                gemm_acc_type_size, gemm_acc_align);
    }

    if (rnn.is_bf32()) {

        const dims_t wei_layer_dims
                = {rnn.n_layer, rnn.n_dir, rnn.n_gates, rnn.slc, rnn.dlc};
        const dims_t wei_iter_dims
                = {rnn.n_layer, rnn.n_dir, rnn.n_gates, rnn.sic, rnn.dic};

        memory_desc_t wei_layer_desc;
        const auto tag = rnn.n_block == 64 ? format_tag::ldgOI64o2i
                                           : format_tag::ldgOI32o2i;
        memory_desc_init_by_tag(
                wei_layer_desc, 5, wei_layer_dims, data_type::bf16, tag);

        memory_desc_t wei_iter_desc;
        memory_desc_init_by_tag(
                wei_iter_desc, 5, wei_iter_dims, data_type::bf16, tag);

        scratchpad.book(key_rnn_bf32_wei_layer_trans,
                memory_desc_wrapper(wei_layer_desc).size(), 64);

        scratchpad.book(key_rnn_bf32_wei_iter_trans,
                memory_desc_wrapper(wei_iter_desc).size(), 64);

        scratchpad.book(key_rnn_bf32_attention_trans,
                rnn.n_iter * rnn.mb * sizeof(bfloat16_t), 64);
    }

    const int max_K_Block
            = nstl::max(rnn.KB1_blocks + 1,
                      nstl::max(rnn.KBproj_blocks + 1, rnn.KB2_blocks + 1))
            * (rnn.brgemm_fwd_iter_layer_fuse_possible ? 2 : 1);
    scratchpad.template book<x64::brgemm_batch_element_t>(
            key_brgemm_primitive_batch, max_K_Block * rnn.nthr);
}

status_t rnn_brgemm_t<prop_kind::forward>::configure_brgemm(
        cpu::rnn_utils::rnn_conf_t &rnn, alg_kind_t cell_kind,
        dim_t src_layer_type_size, dim_t scratch_type_size) {
    using namespace cpu::rnn_utils;

    rnn.M = rnn.mb;
    rnn.N = rnn.dhc;
    rnn.K1 = rnn.slc;
    rnn.K2 = rnn.sic;
    const auto is_int8 = rnn.is_cell_dt_int8();
    const auto is_xf16 = rnn.is_cell_dt_xf16();

    const dim_t padding = (is_int8 ? 4 : (is_xf16 ? 2 : 1));
    rnn.K1padded = utils::rnd_up(rnn.K1, padding);
    rnn.K2padded = utils::rnd_up(rnn.K2, padding);

    rnn.brgemm_isa = brgemm_calc_isa(rnn, rnn.K1, rnn.K2);
    if (rnn.brgemm_isa == isa_undef) return status::unimplemented;
    const int bf32_reduction_dim_threshold = 128;
    const bool is_shape_ok_for_bf32 = rnn.K1 >= bf32_reduction_dim_threshold
            && rnn.K2 >= bf32_reduction_dim_threshold;
    const bool is_bf32 = rnn.is_cell_bf16_amx()
            && rnn.dt_conf == all_f32
            // workspace data type and layouts can differ between fwd and bwd
            // implementations during training.
            && !rnn.is_training
            // bf16 lstm_projection is not supported, so neither is bf32.
            && !rnn.is_lstm_projection && is_shape_ok_for_bf32;
    if (!IMPLICATION(rnn.is_f32_conf(), rnn.is_cell_dt_f32() || is_bf32))
        return status::unimplemented;

    rnn.nthr = dnnl_get_max_threads();
    const bool is_amx_isa_selected
            = rnn.is_cell_int8_amx() || rnn.is_cell_xf16_amx();
    const bool can_use_block64
            = is_amx_isa_selected && rnn.N % 64 == 0 && !rnn.is_lstm_projection;
    const int simd_w = isa_max_vlen(rnn.brgemm_isa) / sizeof(float);
    rnn.n_block = can_use_block64 ? 64 : 2 * simd_w;
    rnn.N_blocks = utils::div_up(rnn.N, rnn.n_block);
    rnn.n_tail = rnn.N % rnn.n_block;

    const float work_by_N
            = static_cast<float>(rnn.N_blocks) / static_cast<float>(rnn.nthr);

    const dim_t l2_cache_size = platform::get_per_core_cache_size(2);
    const dim_t As = src_layer_type_size * rnn.M * (nstl::max(rnn.K1, rnn.K2));
    const dim_t Bs
            = src_layer_type_size * (nstl::max(rnn.K1, rnn.K2)) * rnn.n_block;
    const dim_t Cs
            = scratch_type_size * (rnn.n_gates + 1) * (rnn.M * rnn.n_block);

    std::tie(rnn.k1_block, rnn.k2_block) = brgemm_calc_k_block(rnn, rnn.K1,
            rnn.K2, rnn.M, rnn.n_block, cell_kind, src_layer_type_size, As, Bs,
            Cs, l2_cache_size, rnn.brgemm_isa);
    rnn.KB1_blocks = rnn.K1 / rnn.k1_block;
    rnn.k1_tail = rnn.K1 % rnn.k1_block;
    rnn.KB2_blocks = rnn.K2 / rnn.k2_block;
    rnn.k2_tail = rnn.K2 % rnn.k2_block;
    rnn.m_block = brgemm_calc_m_block(cell_kind, prop_kind::forward, rnn.nthr,
            rnn.M, rnn.N_blocks, rnn.is_cell_dt_f32(), rnn.is_cell_int8_amx(),
            rnn.is_cell_xf16_amx(), work_by_N, As, Bs, Cs, l2_cache_size);

    rnn.M_blocks = rnn.M / rnn.m_block;

    rnn.brgemm_isa = adjust_isa_by_m_block(
            rnn.brgemm_isa, rnn.m_block, rnn.is_cell_int8_amx());
    // Unfused post-gemm for lstm cell allows to parallelize across gates loop
    // and reduces brgemm problem size for the single iteration of parallel loop
    rnn.unfused_post_gemm = cell_kind == alg_kind::vanilla_lstm
            ? IMPLICATION(rnn.M_blocks > 1, rnn.is_cell_xf16_amx())
            : false;

    rnn.LDA1[0] = rnn.src_layer_ld_;
    rnn.LDA1[1] = rnn.dst_iter_ld_;
    rnn.LDA1[2] = rnn.ws_states_layer_ld;

    rnn.LDA2[0] = rnn.src_iter_ld_;
    rnn.LDA2[1] = rnn.dst_layer_ld_;
    rnn.LDA2[2] = rnn.ws_states_iter_ld;

    rnn.LDA2_2[0] = rnn.dst_layer_ld_;
    rnn.LDA2_2[1] = rnn.dst_iter_ld_;
    rnn.LDA2_2[2] = rnn.ws_states_layer_ld;
    rnn.LDA2_2[3] = rnn.ws_states_iter_ld;

    rnn.LDB1 = rnn.n_block;
    rnn.LDB2 = rnn.n_block;
    rnn.LDC = rnn.scratch_gates_ld;

    auto get_dim = [&](dim_t block, dim_t tail) {
        return (block == 0) ? tail : block;
    };

    dim_t n_block = nstl::min(rnn.N, rnn.n_block);
    dim_t n_tail = nstl::min(rnn.N, rnn.nproj_tail);
    if (rnn.LDA1[0] < rnn.k1_block && rnn.LDA1[1] < rnn.k1_block
            && rnn.LDA1[2] < rnn.k1_block)
        return status::unimplemented;
    if (rnn.LDA2[0] < rnn.k2_block && rnn.LDA2[1] < rnn.k2_block
            && rnn.LDA2[2] < rnn.k2_block)
        return status::unimplemented;
    if (rnn.LDB1 < get_dim(n_block, n_tail)
            && rnn.LDB2 < get_dim(n_block, n_tail))
        return status::unimplemented;
    if (rnn.LDC < get_dim(n_block, n_tail)) return status::unimplemented;

    rnn.KBproj_blocks = 0;
    rnn.kproj_tail = 0;
    rnn.kproj_block = 0;

    if (rnn.is_lstm_projection) {
        rnn.Nproj = rnn.dic;
        rnn.Nproj_blocks = utils::div_up(rnn.Nproj, rnn.n_block);
        rnn.nproj_tail = rnn.Nproj % rnn.n_block;

        rnn.Kproj = rnn.dhc;
        rnn.Kprojpadded = utils::rnd_up(rnn.Kproj, padding);
        if (rnn.is_cell_amx()) {
            const dim_t max_row_width = rnn.is_cell_int8_amx() ? 64 : 32;
            rnn.kproj_block = nstl::min(rnn.Kproj, (dim_t)max_row_width);

            rnn.KBproj_blocks = rnn.Kproj / rnn.kproj_block;
            rnn.kproj_tail = rnn.Kproj % rnn.kproj_block;

            if ((rnn.kproj_tail % padding) || (rnn.kproj_block % padding)) {
                rnn.kproj_block = rnn.Kproj;
                rnn.kproj_tail = 0;
                rnn.brgemm_isa = rnn.is_cell_dt_int8() ? x64::avx512_core_vnni
                                                       : x64::avx512_core_bf16;
                if (rnn.is_cell_dt_f16()) return status::unimplemented;
            }
        } else {
            rnn.kproj_block = rnn.Kproj;
            rnn.KBproj_blocks = rnn.Kproj / rnn.kproj_block;
        }
        rnn.LDAproj = rnn.proj_ht_ld;
        rnn.LDBproj = rnn.n_block;
        if (rnn.dt_conf != cpu::rnn_utils::all_f32) {
            rnn.LDCproj[0] = rnn.scratch_gates_ld;
        } else {
            rnn.LDCproj[0] = rnn.scratch_ht_ld;
            rnn.LDCproj[1] = rnn.dst_layer_ld_;
            rnn.LDCproj[2] = rnn.dst_iter_ld_;
            rnn.LDCproj[3] = rnn.ws_states_layer_ld;
        }

        dim_t n_block = nstl::min(rnn.Nproj, rnn.n_block);
        dim_t n_tail = nstl::min(rnn.Nproj, rnn.nproj_tail);
        bool check_LDC = false;
        if (rnn.dt_conf != cpu::rnn_utils::all_f32) {
            check_LDC = rnn.LDCproj[0] < get_dim(n_block, n_tail);
        } else {
            check_LDC = rnn.LDCproj[0] < get_dim(n_block, n_tail)
                    && rnn.LDCproj[1] < get_dim(n_block, n_tail)
                    && rnn.LDCproj[2] < get_dim(n_block, n_tail)
                    && rnn.LDCproj[3] < get_dim(n_block, n_tail);
        }
        if (rnn.LDAproj < rnn.kproj_block
                || rnn.LDBproj < get_dim(n_block, n_tail) || check_LDC)
            return status::unimplemented;
    }

    // enable merged across n_iter dimension layer part of the cell computation
    // TODO: extend coverage for other problem types
    const bool mlc_cell_type_ok = cell_kind == alg_kind::vanilla_lstm
            && !rnn.is_lstm_projection && !rnn.is_lstm_peephole;
    const int mlc_mb_max_threshold = 1;
    const int mlc_n_iter_min_threshold = 2;
    const int mlc_n_layer_max_threshold = 1;
    const bool mlc_problem_shape_ok = rnn.mb <= mlc_mb_max_threshold
            && rnn.n_iter >= mlc_n_iter_min_threshold
            && rnn.n_layer <= mlc_n_layer_max_threshold;
    // if rnn.skip_dst_iter_copy() == false we might need to reduce number of
    // merged cells by 1 (rnn.n_iter - 1)
    // but if in addition rnn.skip_src_layer_copy() == true then on cell
    // position 'first_layer' we still should merge for all the n_iter cells
    // so current support is limited by the case when layer computation is
    // merged for all rnn.n_iter cells
    const bool mlc_m_dim_adjustment_not_required
            = IMPLICATION(rnn.skip_dst_iter_copy(),
                    rnn.skip_src_layer_copy() && rnn.n_layer == 1);
    const bool merged_layer_compute_applicable = rnn.src_layer_is_trivial_stride
            && mlc_cell_type_ok && mlc_problem_shape_ok
            && mlc_m_dim_adjustment_not_required;
    if (merged_layer_compute_applicable) {
        rnn.merge_gemm_layer = true;

        // required adjustment if mlc_m_dim_adjustment_not_required = false
        const int n_iters_to_merge = rnn.n_iter;
        rnn.Mlayermerged = rnn.mb * n_iters_to_merge;
        rnn.mlayermerged_block = brgemm_calc_m_block(cell_kind,
                prop_kind::forward, rnn.nthr, rnn.Mlayermerged, rnn.N_blocks,
                rnn.is_cell_dt_f32(), rnn.is_cell_int8_amx(),
                rnn.is_cell_xf16_amx(), work_by_N, As, Bs, Cs, l2_cache_size);

        rnn.Mlayermerged_blocks = rnn.Mlayermerged / rnn.mlayermerged_block;
    }

    rnn.brgemm_fwd_iter_layer_fuse_possible
            = rnn.slc == rnn.sic && !rnn.merge_gemm_layer;

    if (!rnn.is_orig_gru) {
        rnn.loop_order = rnn.is_cell_int8_amx() || rnn.is_cell_xf16_amx()
                ? brgemm_rnn_execute_loop_order_t::mblk_nblk
                : brgemm_rnn_execute_loop_order_t::nblk_mblk;
    }
    return status::success;
}

status_t init_brgemm_kernel(x64::brgemm_desc_t *desc, x64::cpu_isa_t isa,
        impl::data_type_t src_type, impl::data_type_t weights_type,
        std::unique_ptr<x64::brgemm_kernel_t> &ker, dim_t M, dim_t N, dim_t K,
        dim_t LDA, dim_t LDB, dim_t LDC, float beta, dim_t max_bs,
        dim_t hint_expected_A_size = LLONG_MAX,
        dim_t hint_expected_B_size = LLONG_MAX,
        dim_t hint_expected_C_size = LLONG_MAX) {
    bool transA = false;
    bool transB = false;
    x64::brgemm_layout_t layout = x64::brgemm_row_major;
    CHECK(brgemm_desc_init(desc, isa, x64::brgemm_addr, src_type, weights_type,
            transA, transB, layout, 1.0, beta, LDA, LDB, LDC, M, N, K));

    x64::brgemm_attr_t brgattr;

    brgattr.hint_expected_A_size = hint_expected_A_size;
    brgattr.hint_expected_B_size = hint_expected_B_size;
    brgattr.hint_expected_C_size = hint_expected_C_size;
    brgattr.max_bs = max_bs;
    brgattr.max_top_vpad = 0;
    brgattr.max_bottom_vpad = 0;
    CHECK(brgemm_desc_set_attr(desc, brgattr));

    x64::brgemm_kernel_t *_t_ptr;
    CHECK(brgemm_kernel_create(&_t_ptr, *desc));
    CHECK(safe_ptr_assign<x64::brgemm_kernel_t>(ker, _t_ptr));

    return status::success;
};

status_t rnn_brgemm_t<prop_kind::forward>::brgemm_rnn_init_tiles(
        brgemm_desc_t *desc_array, dim_t size, brgemm_pallete_t pallete) {

    for (dim_t it = 0; it < size; ++it) {
        const auto &desc = desc_array[it];
        const bool desc_empty
                = utils::everyone_is(0, desc.LDA, desc.LDB, desc.LDC);
        if (!desc_empty) return brgemm_init_tiles(desc, pallete);
    }

    return status::unimplemented;
}

status_t rnn_brgemm_t<prop_kind::forward>::brgemm_rnn_init_tiles(
        brgemm_desc_t *desc_array, brgemm_pallete_t pallete) {
    return brgemm_rnn_init_tiles(desc_array, num_base_kernels_, pallete);
}
status_t rnn_brgemm_t<prop_kind::forward>::brgemm_rnn_init_tiles_proj(
        brgemm_desc_t *desc_array, brgemm_pallete_t pallete) {
    return brgemm_rnn_init_tiles(desc_array, num_proj_kernels_, pallete);
}

status_t rnn_brgemm_t<prop_kind::forward>::init_kernels(
        const cpu::rnn_utils::rnn_conf_t &rnn, data_type_t src_type,
        data_type_t weights_type) {

    const auto init_brgemm
            = [&](x64::brgemm_desc_t *desc, x64::cpu_isa_t isa,
                      std::unique_ptr<x64::brgemm_kernel_t> &ker, dim_t M,
                      dim_t N, dim_t K, dim_t LDA, dim_t LDB, dim_t LDC,
                      float beta, dim_t max_bs) {
                  return init_brgemm_kernel(desc, isa, src_type, weights_type,
                          ker, M, N, K, LDA, LDB, LDC, beta, max_bs);
              };

    const int brgemm_n = nstl::min(rnn.N, rnn.n_block);
    const int brgemm_n_tail = nstl::min(rnn.N, rnn.n_tail);
    const int max_bs_factor = rnn.brgemm_fwd_iter_layer_fuse_possible ? 2 : 1;

    for (int i = 0; i < num_base_kernels_; i++) {
        if (rnn.merge_gemm_layer) {
            CHECK(init_brgemm(&desc_layermerged_b0_[i], rnn.brgemm_isa,
                    kernel_layermerged_b0_[i], rnn.mlayermerged_block, brgemm_n,
                    rnn.k1_block, rnn.LDA1[i], rnn.LDB1, rnn.LDC, 0.0,
                    rnn.KB1_blocks));
        } else {
            CHECK(init_brgemm(&desc_layer_b0_[i], rnn.brgemm_isa,
                    kernel_layer_b0_[i], rnn.m_block, brgemm_n, rnn.k1_block,
                    rnn.LDA1[i], rnn.LDB1, rnn.LDC, 0.0,
                    max_bs_factor * rnn.KB1_blocks));
        }

        CHECK(init_brgemm(&desc_iter_b0_[i], rnn.brgemm_isa, kernel_iter_b0_[i],
                rnn.m_block, brgemm_n, rnn.k2_block, rnn.LDA2[i], rnn.LDB2,
                rnn.LDC, 0.0, rnn.KB2_blocks));
        CHECK(init_brgemm(&desc_iter_b1_[i], rnn.brgemm_isa, kernel_iter_b1_[i],
                rnn.m_block, brgemm_n, rnn.k2_block, rnn.LDA2[i], rnn.LDB2,
                rnn.LDC, 1.0, rnn.KB2_blocks));
        if (rnn.n_tail) {
            if (rnn.merge_gemm_layer) {
                CHECK(init_brgemm(&desc_layermerged_N_tail_b0_[i],
                        rnn.brgemm_isa, kernel_layermerged_N_tail_b0_[i],
                        rnn.mlayermerged_block, brgemm_n_tail, rnn.k1_block,
                        rnn.LDA1[i], rnn.LDB1, rnn.LDC, 0.0, rnn.KB1_blocks));
            } else {
                CHECK(init_brgemm(&desc_layer_N_tail_b0_[i], rnn.brgemm_isa,
                        kernel_layer_N_tail_b0_[i], rnn.m_block, brgemm_n_tail,
                        rnn.k1_block, rnn.LDA1[i], rnn.LDB1, rnn.LDC, 0.0,
                        max_bs_factor * rnn.KB1_blocks));
            }

            CHECK(init_brgemm(&desc_iter_N_tail_b0_[i], rnn.brgemm_isa,
                    kernel_iter_N_tail_b0_[i], rnn.m_block, brgemm_n_tail,
                    rnn.k2_block, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 0.0,
                    rnn.KB2_blocks));
            CHECK(init_brgemm(&desc_iter_N_tail_b1_[i], rnn.brgemm_isa,
                    kernel_iter_N_tail_b1_[i], rnn.m_block, brgemm_n_tail,
                    rnn.k2_block, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 1.0,
                    rnn.KB2_blocks));
        }
        if (rnn.k1_tail) {
            if (rnn.merge_gemm_layer) {
                CHECK(init_brgemm(&desc_layermerged_K1_tail_b1_[i],
                        rnn.brgemm_isa, kernel_layermerged_K1_tail_b1_[i],
                        rnn.mlayermerged_block, brgemm_n, rnn.k1_tail,
                        rnn.LDA1[i], rnn.LDB1, rnn.LDC, 1.0, 1));
            } else {
                CHECK(init_brgemm(&desc_layer_K1_tail_b1_[i], rnn.brgemm_isa,
                        kernel_layer_K1_tail_b1_[i], rnn.m_block, brgemm_n,
                        rnn.k1_tail, rnn.LDA1[i], rnn.LDB1, rnn.LDC, 1.0,
                        max_bs_factor * 1));
            }
        }
        if (rnn.k2_tail)
            CHECK(init_brgemm(&desc_iter_K2_tail_b1_[i], rnn.brgemm_isa,
                    kernel_iter_K2_tail_b1_[i], rnn.m_block, brgemm_n,
                    rnn.k2_tail, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 1.0, 1));
        if (rnn.k1_tail && rnn.n_tail) {
            if (rnn.merge_gemm_layer) {
                CHECK(init_brgemm(&desc_layermerged_NK1_tail_b1_[i],
                        rnn.brgemm_isa, kernel_layermerged_NK1_tail_b1_[i],
                        rnn.mlayermerged_block, brgemm_n_tail, rnn.k1_tail,
                        rnn.LDA1[i], rnn.LDB1, rnn.LDC, 1.0, 1));
            } else {
                CHECK(init_brgemm(&desc_layer_NK1_tail_b1_[i], rnn.brgemm_isa,
                        kernel_layer_NK1_tail_b1_[i], rnn.m_block,
                        brgemm_n_tail, rnn.k1_tail, rnn.LDA1[i], rnn.LDB1,
                        rnn.LDC, 1.0, max_bs_factor * 1));
            }
        }
        if (rnn.k2_tail && rnn.n_tail)
            CHECK(init_brgemm(&desc_iter_NK2_tail_b1_[i], rnn.brgemm_isa,
                    kernel_iter_NK2_tail_b1_[i], rnn.m_block, brgemm_n_tail,
                    rnn.k2_tail, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 1.0, 1));
    }
    if (rnn.is_orig_gru) {
        for (int i = 0; i < num_vanilla_gru_iter_part2_kernels_; i++) {
            CHECK(init_brgemm(&desc_iter_p2_b1_[i], rnn.brgemm_isa,
                    kernel_iter_p2_b1_[i], rnn.m_block, brgemm_n, rnn.k2_block,
                    rnn.LDA2_2[i], rnn.LDB2, rnn.LDC, 1.0, rnn.KB2_blocks));
            if (rnn.n_tail)
                CHECK(init_brgemm(&desc_iter_p2_N_tail_b1_[i], rnn.brgemm_isa,
                        kernel_iter_p2_N_tail_b1_[i], rnn.m_block,
                        brgemm_n_tail, rnn.k2_block, rnn.LDA2_2[i], rnn.LDB2,
                        rnn.LDC, 1.0, rnn.KB2_blocks));
            if (rnn.k2_tail)
                CHECK(init_brgemm(&desc_iter_p2_K2_tail_b1_[i], rnn.brgemm_isa,
                        kernel_iter_p2_K2_tail_b1_[i], rnn.m_block, brgemm_n,
                        rnn.k2_tail, rnn.LDA2_2[i], rnn.LDB2, rnn.LDC, 1.0, 1));
            if (rnn.k2_tail && rnn.n_tail)
                CHECK(init_brgemm(&desc_iter_p2_NK2_tail_b1_[i], rnn.brgemm_isa,
                        kernel_iter_p2_NK2_tail_b1_[i], rnn.m_block,
                        brgemm_n_tail, rnn.k2_tail, rnn.LDA2_2[i], rnn.LDB2,
                        rnn.LDC, 1.0, 1));
        }
    }
    if (rnn.is_lstm_projection) {
        const dim_t brgemm_np = nstl::min(rnn.Nproj, rnn.n_block);
        const dim_t brgemm_np_tail = nstl::min(rnn.Nproj, rnn.nproj_tail);
        const int n_kernel = (rnn.dt_conf == cpu::rnn_utils::all_f32)
                ? num_proj_kernels_
                : 1;
        for (int i = 0; i < n_kernel; i++) {
            CHECK(init_brgemm(&desc_proj_b0_[i], rnn.brgemm_isa,
                    kernel_proj_b0_[i], rnn.m_block, brgemm_np, rnn.kproj_block,
                    rnn.LDAproj, rnn.LDBproj, rnn.LDCproj[i], 0.0,
                    rnn.KBproj_blocks));
            if (rnn.nproj_tail) {
                CHECK(init_brgemm(&desc_proj_N_tail_b0_[i], rnn.brgemm_isa,
                        kernel_proj_N_tail_b0_[i], rnn.m_block, brgemm_np_tail,
                        rnn.kproj_block, rnn.LDAproj, rnn.LDBproj,
                        rnn.LDCproj[i], 0.0, rnn.KBproj_blocks));
                CHECK(init_brgemm(&desc_proj_N_tail_b1_[i], rnn.brgemm_isa,
                        kernel_proj_N_tail_b1_[i], rnn.m_block, brgemm_np_tail,
                        rnn.kproj_block, rnn.LDAproj, rnn.LDBproj,
                        rnn.LDCproj[i], 1.0, rnn.KBproj_blocks));
            }
            if (rnn.is_cell_int8_amx() || rnn.is_cell_xf16_amx()) {
                if (rnn.kproj_tail)
                    CHECK(init_brgemm(&desc_proj_K_tail_b1_[i], rnn.brgemm_isa,
                            kernel_proj_K_tail_b1_[i], rnn.m_block, brgemm_np,
                            rnn.kproj_tail, rnn.LDAproj, rnn.LDBproj,
                            rnn.LDCproj[i], 1.0, 1));
                if (rnn.kproj_tail && rnn.nproj_tail)
                    CHECK(init_brgemm(&desc_proj_NK_tail_b1_[i], rnn.brgemm_isa,
                            kernel_proj_NK_tail_b1_[i], rnn.m_block,
                            brgemm_np_tail, rnn.kproj_tail, rnn.LDAproj,
                            rnn.LDBproj, rnn.LDCproj[i], 1.0, 1));
            }
        }
    }

    if (rnn.is_cell_amx()) {
        if (rnn.merge_gemm_layer)
            CHECK(brgemm_rnn_init_tiles(
                    desc_layermerged_b0_, pallete_buff_layermerged_));
        else
            CHECK(brgemm_rnn_init_tiles(desc_layer_b0_, pallete_buff_layer_));
        CHECK(brgemm_rnn_init_tiles(desc_iter_b0_, pallete_buff_iter_));

        if (rnn.n_tail) {
            if (rnn.merge_gemm_layer)
                CHECK(brgemm_rnn_init_tiles(desc_layermerged_N_tail_b0_,
                        pallete_buff_layermerged_n_tail_));
            else
                CHECK(brgemm_rnn_init_tiles(
                        desc_layer_N_tail_b0_, pallete_buff_layer_n_tail_));
            CHECK(brgemm_rnn_init_tiles(
                    desc_iter_N_tail_b0_, pallete_buff_iter_n_tail_));
        }
        if (rnn.k1_tail) {
            if (rnn.merge_gemm_layer)
                CHECK(brgemm_rnn_init_tiles(desc_layermerged_K1_tail_b1_,
                        pallete_buff_layermerged_k1_tail_));
            else
                CHECK(brgemm_rnn_init_tiles(
                        desc_layer_K1_tail_b1_, pallete_buff_k1_tail_));
        }
        if (rnn.k2_tail)
            CHECK(brgemm_rnn_init_tiles(
                    desc_iter_K2_tail_b1_, pallete_buff_k2_tail_));
        if (rnn.k1_tail && rnn.n_tail) {
            if (rnn.merge_gemm_layer)
                CHECK(brgemm_rnn_init_tiles(desc_layermerged_NK1_tail_b1_,
                        pallete_buff_layermerged_nk1_tail_));
            else
                CHECK(brgemm_rnn_init_tiles(
                        desc_layer_NK1_tail_b1_, pallete_buff_nk1_tail_));
        }
        if (rnn.k2_tail && rnn.n_tail)
            CHECK(brgemm_rnn_init_tiles(
                    desc_iter_NK2_tail_b1_, pallete_buff_nk2_tail_));
        if (rnn.is_lstm_projection) {
            CHECK(brgemm_rnn_init_tiles_proj(
                    desc_proj_b0_, pallete_buff_proj_));
            if (rnn.nproj_tail)
                CHECK(brgemm_rnn_init_tiles_proj(
                        desc_proj_N_tail_b0_, pallete_buff_nproj_tail_));
            if (rnn.kproj_tail)
                CHECK(brgemm_rnn_init_tiles_proj(
                        desc_proj_K_tail_b1_, pallete_buff_kproj_tail_));
            if (rnn.kproj_tail && rnn.nproj_tail)
                CHECK(brgemm_rnn_init_tiles_proj(
                        desc_proj_NK_tail_b1_, pallete_buff_nkproj_tail_));
        }
    }

    return status::success;
}

void rnn_brgemm_t<prop_kind::backward>::init_scratchpad(
        const cpu::rnn_utils::rnn_conf_t &rnn,
        memory_tracking::registrar_t &scratchpad, dim_t gemm_acc_type_size,
        dim_t gemm_acc_align) {

    rnn_brgemm_base_t::init_scratchpad(
            rnn, scratchpad, gemm_acc_type_size, gemm_acc_align);

    using namespace memory_tracking::names;

    // init scratchpad for internal reorders:
    const auto data_size
            = rnn.is_xf16_conf() ? sizeof(bfloat16_t) : sizeof(float);
    const auto &d_wei = rnn.diff_wei_brgemm;
    const auto scratch_gates_blocked_per_thr = d_wei.Kpadded * d_wei.n_block;
    const auto scratch_gates_blocked_size
            = rnn.nthr * scratch_gates_blocked_per_thr;
    scratchpad.book(key_rnn_gates_blocked, scratch_gates_blocked_size,
            data_size, gemm_acc_align);

    const auto scratch_src_layer_size = d_wei.global_transpose
            ? d_wei.M_layer * d_wei.Kpadded
            : rnn.nthr * std::min(d_wei.m_block, d_wei.M_layer) * d_wei.Kpadded;
    scratchpad.book(key_rnn_src_layer_trans, scratch_src_layer_size, data_size,
            gemm_acc_align);

    const auto scratch_src_iter_size = d_wei.global_transpose
            ? d_wei.M_iter * d_wei.Kpadded
            : rnn.nthr * std::min(d_wei.m_block, d_wei.M_iter) * d_wei.Kpadded;
    scratchpad.book(key_rnn_src_iter_trans, scratch_src_iter_size, data_size,
            gemm_acc_align);
}

status_t rnn_brgemm_t<prop_kind::backward>::configure_brgemm(
        cpu::rnn_utils::rnn_conf_t &rnn, alg_kind_t cell_kind,
        dim_t src_layer_type_size, dim_t scratch_type_size) {
    using namespace cpu::rnn_utils;

    if (rnn.is_int8_conf() || rnn.is_cell_dt_int8())
        return status::unimplemented;

    auto &diff_src_conf = rnn.diff_src_brgemm;

    diff_src_conf.M = rnn.mb;
    diff_src_conf.N_iter = rnn.sic;
    diff_src_conf.N_layer = rnn.slc;
    diff_src_conf.N = nstl::max(diff_src_conf.N_iter, diff_src_conf.N_layer);
    diff_src_conf.K = rnn.dhc;

    rnn.nthr = dnnl_get_max_threads();
    diff_src_conf.n_block = 32;
    diff_src_conf.N_blocks
            = utils::div_up(diff_src_conf.N, diff_src_conf.n_block);
    diff_src_conf.n_tail = diff_src_conf.N % diff_src_conf.n_block;
    diff_src_conf.N_layer_blocks
            = utils::div_up(diff_src_conf.N_layer, diff_src_conf.n_block);
    diff_src_conf.n_layer_tail = diff_src_conf.N_layer % diff_src_conf.n_block;
    diff_src_conf.N_iter_blocks
            = utils::div_up(diff_src_conf.N_iter, diff_src_conf.n_block);
    diff_src_conf.n_iter_tail = diff_src_conf.N_iter % diff_src_conf.n_block;

    const float work_by_N = static_cast<float>(diff_src_conf.N_blocks)
            / static_cast<float>(rnn.nthr);

    const dim_t l2_cache_size = platform::get_per_core_cache_size(2);
    const dim_t As = src_layer_type_size * diff_src_conf.M * diff_src_conf.K;
    const dim_t Bs
            = src_layer_type_size * diff_src_conf.K * diff_src_conf.n_block;
    const dim_t Cs = scratch_type_size * (rnn.n_gates + 1)
            * (diff_src_conf.M * diff_src_conf.n_block);

    const auto is_xf16 = rnn.is_cell_dt_xf16();

    const dim_t padding = is_xf16 ? 2 : 1;
    diff_src_conf.Kpadded = utils::rnd_up(diff_src_conf.K, padding);

    diff_src_conf.isa = brgemm_calc_isa(rnn, diff_src_conf.K, diff_src_conf.K);
    const bool is_xf16_amx
            = is_xf16 && is_superset(diff_src_conf.isa, x64::avx512_core_amx);
    const bool split_gates_computation = is_xf16_amx && diff_src_conf.K >= 1024
            && diff_src_conf.n_tail == 0;
    diff_src_conf.gates_block = split_gates_computation ? 1 : rnn.n_gates;

    std::tie(diff_src_conf.k_block, std::ignore) = brgemm_calc_k_block(rnn,
            diff_src_conf.K, diff_src_conf.K, diff_src_conf.M,
            diff_src_conf.n_block, cell_kind, src_layer_type_size, As, Bs, Cs,
            l2_cache_size, diff_src_conf.isa);

    diff_src_conf.K_blocks = diff_src_conf.K / diff_src_conf.k_block;
    diff_src_conf.K_blocks *= rnn.n_gates;
    diff_src_conf.k_tail = diff_src_conf.K % diff_src_conf.k_block;

    diff_src_conf.m_block = brgemm_calc_m_block(cell_kind, prop_kind::backward,
            rnn.nthr, diff_src_conf.M, diff_src_conf.N_blocks,
            rnn.is_cell_dt_f32(), false, is_xf16_amx, work_by_N, As, Bs, Cs,
            l2_cache_size);

    diff_src_conf.M_blocks = diff_src_conf.M / diff_src_conf.m_block;
    diff_src_conf.LDA = rnn.scratch_gates_ld;
    diff_src_conf.LDB = diff_src_conf.n_block;
    diff_src_conf.LDC = rnn.ws_diff_states_iter_ld;

    if (diff_src_conf.LDA < diff_src_conf.k_block) return status::unimplemented;

    const dim_t n_block = nstl::min(diff_src_conf.N, diff_src_conf.n_block);

    if (diff_src_conf.LDB < n_block) return status::unimplemented;
    if (diff_src_conf.LDC < n_block) return status::unimplemented;

    rnn.KBproj_blocks = 0;
    rnn.kproj_tail = 0;
    rnn.kproj_block = 0;

    auto &diff_wei_conf = rnn.diff_wei_brgemm;
    diff_wei_conf.global_transpose = rnn.mb > 1;
    diff_wei_conf.M_iter = rnn.sic;
    diff_wei_conf.M_layer = rnn.slc;
    diff_wei_conf.M = nstl::max(rnn.sic, rnn.slc);
    diff_wei_conf.N = rnn.dhc * rnn.n_gates;
    diff_wei_conf.K = (scratch_type_size != sizeof(float))
            ? utils::rnd_up(rnn.mb, 2)
            : rnn.mb;
    diff_wei_conf.Kpadded = utils::rnd_up(diff_wei_conf.K, padding);

    diff_wei_conf.isa = brgemm_calc_isa(rnn, diff_wei_conf.K, diff_wei_conf.K);

    const bool is_wei_xf16_amx = rnn.is_cell_dt_xf16()
            && is_superset(diff_wei_conf.isa, x64::avx512_core_amx);
    const bool diff_wei_can_use_nblock64 = is_wei_xf16_amx
            && diff_wei_conf.N % 64 == 0 && !rnn.is_lstm_peephole;
    diff_wei_conf.n_block = diff_wei_can_use_nblock64 ? 64 : 32;
    diff_wei_conf.N_blocks
            = utils::div_up(diff_wei_conf.N, diff_wei_conf.n_block);
    diff_wei_conf.n_tail = diff_wei_conf.N % diff_wei_conf.n_block;

    const dim_t As_wei
            = src_layer_type_size * diff_wei_conf.M * diff_wei_conf.K;
    const dim_t Bs_wei
            = src_layer_type_size * diff_wei_conf.K * diff_wei_conf.n_block;
    const dim_t Cs_wei = scratch_type_size * (rnn.n_gates + 1)
            * (diff_wei_conf.M * diff_wei_conf.n_block);

    std::tie(diff_wei_conf.k_block, std::ignore) = brgemm_calc_k_block(rnn,
            diff_wei_conf.K, diff_wei_conf.K, diff_wei_conf.M,
            diff_wei_conf.n_block, cell_kind, src_layer_type_size, As_wei,
            Bs_wei, Cs_wei, l2_cache_size, diff_wei_conf.isa);

    diff_wei_conf.K_blocks = diff_wei_conf.K / diff_wei_conf.k_block;
    diff_wei_conf.k_tail = diff_wei_conf.K % diff_wei_conf.k_block;

    if (diff_wei_conf.M_iter != diff_wei_conf.M_layer) {
        diff_wei_conf.m_block = diff_wei_conf.M;
        diff_wei_conf.M_blocks = 1;
    } else {
        const float work_by_N_wei = static_cast<float>(diff_wei_conf.N_blocks)
                / static_cast<float>(rnn.nthr);

        diff_wei_conf.m_block
                = brgemm_calc_m_block(cell_kind, prop_kind::backward, rnn.nthr,
                        diff_wei_conf.M, diff_wei_conf.N_blocks,
                        rnn.is_cell_dt_f32(), false, is_wei_xf16_amx,
                        work_by_N_wei, As_wei, Bs_wei, Cs_wei, l2_cache_size);
        diff_wei_conf.M_blocks = diff_wei_conf.M / diff_wei_conf.m_block;
    }

    diff_wei_conf.LDA_layer = diff_wei_conf.K;
    diff_wei_conf.LDA_iter = diff_wei_conf.K;
    diff_wei_conf.LDB = diff_wei_conf.n_block;
    diff_wei_conf.LDC_iter = rnn.diff_weights_iter_ld;
    diff_wei_conf.LDC_layer = rnn.diff_weights_layer_ld;

    if (diff_wei_conf.LDA_layer < diff_wei_conf.k_block
            || diff_wei_conf.LDA_iter < diff_wei_conf.k_block)
        return status::unimplemented;

    if (rnn.is_lstm_peephole) { configure_brgemm_peephole(rnn); }

    rnn.M = nstl::max(diff_wei_conf.M, diff_src_conf.M);
    rnn.N = nstl::max(diff_wei_conf.N, diff_src_conf.N);
    rnn.K1 = nstl::max(diff_wei_conf.K, diff_src_conf.K);
    rnn.K2 = rnn.K1;
    rnn.m_block = nstl::max(diff_wei_conf.m_block, diff_src_conf.m_block);
    rnn.M_blocks = nstl::max(diff_wei_conf.M_blocks, diff_src_conf.M_blocks);
    rnn.n_block = nstl::max(diff_wei_conf.n_block, diff_src_conf.n_block);
    rnn.N_blocks = nstl::max(diff_wei_conf.N_blocks, diff_src_conf.N_blocks);
    rnn.n_tail = nstl::max(diff_wei_conf.n_tail, diff_src_conf.n_tail);
    rnn.k1_block = nstl::max(diff_wei_conf.k_block, diff_src_conf.k_block);
    rnn.k2_block = rnn.k1_block;
    rnn.k1_tail = nstl::max(diff_wei_conf.k_tail, diff_src_conf.k_tail);
    rnn.k2_tail = rnn.k1_tail;
    rnn.KB1_blocks = nstl::max(diff_wei_conf.K_blocks, diff_src_conf.K_blocks);
    rnn.KB2_blocks = rnn.KB1_blocks;
    rnn.K1padded = nstl::max(diff_wei_conf.Kpadded, diff_src_conf.Kpadded);
    rnn.K2padded = rnn.K1padded;
    rnn.unfused_post_gemm = true;

    if (utils::one_of(true,
                is_superset(diff_wei_conf.isa, x64::avx512_core_amx)
                        || is_superset(
                                diff_src_conf.isa, x64::avx512_core_amx))) {
        // pick the greater isa
        rnn.brgemm_isa = is_superset(diff_wei_conf.isa, diff_src_conf.isa)
                ? diff_wei_conf.isa
                : diff_src_conf.isa;
    } else {
        rnn.brgemm_isa = diff_wei_conf.isa;
    }

    if (!rnn.is_orig_gru) {
        rnn.diff_src_brgemm.loop_order = is_xf16
                        && is_superset(diff_src_conf.isa, x64::avx512_core_amx)
                ? brgemm_rnn_execute_loop_order_t::mblk_nblk
                : brgemm_rnn_execute_loop_order_t::nblk_mblk;
        rnn.diff_wei_brgemm.loop_order = is_xf16
                        && is_superset(diff_wei_conf.isa, x64::avx512_core_amx)
                ? brgemm_rnn_execute_loop_order_t::mblk_nblk
                : brgemm_rnn_execute_loop_order_t::nblk_mblk;
    }

    return status::success;
}

static dim_t divide_block_to_improve_thread_balance(
        const dim_t initial_work_amount, const dim_t division_block,
        const dim_t nthr) {

    const float nthr_f = static_cast<float>(nthr);
    const float initial_work = static_cast<float>(initial_work_amount) / nthr_f;
    const float decimal_initial_factor
            = initial_work - std::floor(initial_work);
    static constexpr float thread_balance_threashold = 0.8f;
    static constexpr float tolerance = 0.01f;

    float max_decimal_factor = -1.0f;
    dim_t best_candidate = -1.0;
    bool found_best_solution = false;

    if (decimal_initial_factor < thread_balance_threashold
            && decimal_initial_factor != 0.0f) {

        for (const int block_size : {4096, 2048, 1024, 512, 256, 128, 64, 32}) {

            if (division_block <= block_size) continue;

            const auto blocks = utils::div_up(division_block, block_size);

            const float work
                    = static_cast<float>(initial_work_amount * blocks) / nthr_f;
            const float work_decimal = work - std::floor(work);

            if (work_decimal == 0.0f
                    || (max_decimal_factor != 0.0f
                                    ? work_decimal
                                            > (max_decimal_factor + tolerance)
                                    : work_decimal >= thread_balance_threashold)

            ) {
                best_candidate = block_size;
                max_decimal_factor = work_decimal;
            }

            if (work >= nthr_f
                    && (work_decimal >= thread_balance_threashold
                            || work_decimal == 0.0f)) {
                found_best_solution = true;
                break;
            }
        }
    }

    if (found_best_solution
            || (!found_best_solution
                    && max_decimal_factor
                            > decimal_initial_factor + tolerance)) {
        return best_candidate;
    }

    return division_block;
}

void rnn_brgemm_t<prop_kind::backward>::configure_brgemm_peephole(
        cpu::rnn_utils::rnn_conf_t &rnn) {
    static constexpr dim_t n_gates = 3;
    rnn.dhc_block_peephole = divide_block_to_improve_thread_balance(
            n_gates, rnn.dhc, rnn.nthr);
    rnn.dhc_blocks_peephole = utils::div_up(rnn.dhc, rnn.dhc_block_peephole);
    rnn.dhc_tail_peephole = rnn.dhc % rnn.dhc_block_peephole;
}

static status_t init_kernels_diff_src(rnn_diff_src_brgemm_t &diff_src,
        const cpu::rnn_utils::rnn_conf_t &rnn, data_type_t src_type,
        data_type_t weights_type) {

    const auto init_brgemm_diff_src
            = [&](x64::brgemm_desc_t *desc, x64::cpu_isa_t isa,
                      std::unique_ptr<x64::brgemm_kernel_t> &ker, dim_t M,
                      dim_t N, dim_t K, dim_t LDA, dim_t LDB, dim_t LDC,
                      float beta, dim_t max_bs) {
                  const dim_t A_size
                          = rnn.diff_src_brgemm.M * rnn.diff_src_brgemm.Kpadded;
                  const dim_t B_size
                          = rnn.diff_src_brgemm.Kpadded * rnn.diff_src_brgemm.N;
                  const dim_t C_size
                          = rnn.diff_src_brgemm.M * rnn.diff_src_brgemm.N;
                  return init_brgemm_kernel(desc, isa, src_type, weights_type,
                          ker, M, N, K, LDA, LDB, LDC, beta, max_bs, A_size,
                          B_size, C_size);
              };

    const auto &diff_src_conf = rnn.diff_src_brgemm;
    const int n_diff_src = nstl::min(diff_src_conf.N, diff_src_conf.n_block);
    const int n_diff_src_iter_tail
            = nstl::min(diff_src_conf.N_iter, diff_src_conf.n_iter_tail);
    const int n_diff_src_layer_tail
            = nstl::min(diff_src_conf.N_layer, diff_src_conf.n_layer_tail);
    const auto K_batch_size = rnn.n_gates * diff_src_conf.K_blocks;
    const auto split_gates_computation
            = diff_src_conf.gates_block != rnn.n_gates;
    CHECK(init_brgemm_diff_src(&diff_src.desc_iter_layer_beta0_,
            diff_src_conf.isa, diff_src.kernel_iter_layer_beta0_,
            diff_src_conf.m_block, n_diff_src, diff_src_conf.k_block,
            diff_src_conf.LDA, diff_src_conf.LDB, diff_src_conf.LDC, 0.0,
            K_batch_size));
    if (split_gates_computation)
        CHECK(init_brgemm_diff_src(&diff_src.desc_iter_layer_beta1_,
                diff_src_conf.isa, diff_src.kernel_iter_layer_beta1_,
                diff_src_conf.m_block, n_diff_src, diff_src_conf.k_block,
                diff_src_conf.LDA, diff_src_conf.LDB, diff_src_conf.LDC, 1.0,
                K_batch_size));

    if (n_diff_src_layer_tail) {
        CHECK(init_brgemm_diff_src(&diff_src.desc_layer_N_tail_beta0_,
                diff_src_conf.isa, diff_src.kernel_layer_N_tail_beta0_,
                diff_src_conf.m_block, n_diff_src_layer_tail,
                diff_src_conf.k_block, diff_src_conf.LDA, diff_src_conf.LDB,
                diff_src_conf.LDC, 0.0, K_batch_size));
        if (split_gates_computation)
            CHECK(init_brgemm_diff_src(&diff_src.desc_layer_N_tail_beta1_,
                    diff_src_conf.isa, diff_src.kernel_layer_N_tail_beta1_,
                    diff_src_conf.m_block, n_diff_src_layer_tail,
                    diff_src_conf.k_block, diff_src_conf.LDA, diff_src_conf.LDB,
                    diff_src_conf.LDC, 1.0, K_batch_size));
    }

    if (n_diff_src_iter_tail) {
        CHECK(init_brgemm_diff_src(&diff_src.desc_iter_N_tail_beta0_,
                diff_src_conf.isa, diff_src.kernel_iter_N_tail_beta0_,
                diff_src_conf.m_block, n_diff_src_iter_tail,
                diff_src_conf.k_block, diff_src_conf.LDA, diff_src_conf.LDB,
                diff_src_conf.LDC, 0.0, K_batch_size));
        if (split_gates_computation)
            CHECK(init_brgemm_diff_src(&diff_src.desc_iter_N_tail_beta1_,
                    diff_src_conf.isa, diff_src.kernel_iter_N_tail_beta1_,
                    diff_src_conf.m_block, n_diff_src_iter_tail,
                    diff_src_conf.k_block, diff_src_conf.LDA, diff_src_conf.LDB,
                    diff_src_conf.LDC, 1.0, K_batch_size));
    }

    if (diff_src_conf.k_tail) {
        CHECK(init_brgemm_diff_src(&diff_src.desc_iter_layer_K_tail_beta1_,
                diff_src_conf.isa, diff_src.kernel_iter_layer_K_tail_beta1_,
                diff_src_conf.m_block, n_diff_src, diff_src_conf.k_tail,
                diff_src_conf.LDA, diff_src_conf.LDB, diff_src_conf.LDC, 1.0,
                rnn.n_gates));

        if (n_diff_src_layer_tail) {
            CHECK(init_brgemm_diff_src(&diff_src.desc_layer_NK_tail_beta1_,
                    diff_src_conf.isa, diff_src.kernel_layer_NK_tail_beta1_,
                    diff_src_conf.m_block, n_diff_src_layer_tail,
                    diff_src_conf.k_tail, diff_src_conf.LDA, diff_src_conf.LDB,
                    diff_src_conf.LDC, 1.0, rnn.n_gates));
        }

        if (n_diff_src_iter_tail) {
            CHECK(init_brgemm_diff_src(&diff_src.desc_iter_NK_tail_beta1_,
                    diff_src_conf.isa, diff_src.kernel_iter_NK_tail_beta1_,
                    diff_src_conf.m_block, n_diff_src_iter_tail,
                    diff_src_conf.k_tail, diff_src_conf.LDA, diff_src_conf.LDB,
                    diff_src_conf.LDC, 1.0, rnn.n_gates));
        }
    }

    const bool is_xf16_amx = rnn.is_cell_dt_xf16()
            && is_superset(diff_src_conf.isa, x64::avx512_core_amx);

    if (is_xf16_amx) {
        CHECK(brgemm_init_tiles(diff_src.desc_iter_layer_beta0_,
                diff_src.pallete_buff_iter_layer_));

        if (n_diff_src_layer_tail)
            CHECK(brgemm_init_tiles(diff_src.desc_layer_N_tail_beta0_,
                    diff_src.pallete_buff_layer_n_tail_));

        if (n_diff_src_iter_tail)
            CHECK(brgemm_init_tiles(diff_src.desc_iter_N_tail_beta0_,
                    diff_src.pallete_buff_iter_n_tail_));

        if (diff_src_conf.k_tail) {
            CHECK(brgemm_init_tiles(diff_src.desc_iter_layer_K_tail_beta1_,
                    diff_src.pallete_buff_iter_layer_k_tail_));

            if (n_diff_src_layer_tail)
                CHECK(brgemm_init_tiles(diff_src.desc_layer_NK_tail_beta1_,
                        diff_src.pallete_buff_layer_nk_tail_));

            if (n_diff_src_iter_tail)
                CHECK(brgemm_init_tiles(diff_src.desc_iter_NK_tail_beta1_,
                        diff_src.pallete_buff_iter_nk_tail_));
        }
    }

    return status::success;
}

static status_t init_kernels_diff_wei(rnn_diff_wei_brgemm_t &diff_wei,
        const cpu::rnn_utils::rnn_conf_t &rnn, data_type_t src_type,
        data_type_t weights_type) {

    const auto init_brgemm_diff_wei
            = [&](x64::brgemm_desc_t *desc, x64::cpu_isa_t isa,
                      std::unique_ptr<x64::brgemm_kernel_t> &ker, dim_t M,
                      dim_t N, dim_t K, dim_t LDA, dim_t LDB, dim_t LDC,
                      float beta, dim_t max_bs) {
                  const dim_t A_size
                          = rnn.diff_wei_brgemm.M * rnn.diff_wei_brgemm.Kpadded;
                  const dim_t B_size
                          = rnn.diff_wei_brgemm.Kpadded * rnn.diff_wei_brgemm.N;
                  const dim_t C_size
                          = rnn.diff_wei_brgemm.M * rnn.diff_wei_brgemm.N;
                  return init_brgemm_kernel(desc, isa, src_type, weights_type,
                          ker, M, N, K, LDA, LDB, LDC, beta, max_bs, A_size,
                          B_size, C_size);
              };

    const auto &diff_wei_conf = rnn.diff_wei_brgemm;
    const bool is_m_block_equal = rnn.slc == rnn.sic;
    const auto m_block_iter
            = is_m_block_equal ? diff_wei_conf.m_block : diff_wei_conf.M_iter;
    const auto m_block_layer
            = is_m_block_equal ? diff_wei_conf.m_block : diff_wei_conf.M_layer;
    const auto n_diff_wei = nstl::min(diff_wei_conf.N, diff_wei_conf.n_block);
    const auto n_diff_wei_tail
            = nstl::min(diff_wei_conf.N, diff_wei_conf.n_tail);

    init_brgemm_diff_wei(&diff_wei.desc_iter_beta1_, diff_wei_conf.isa,
            diff_wei.kernel_iter_beta1_, m_block_iter, n_diff_wei,
            diff_wei_conf.k_block, diff_wei_conf.LDA_iter, diff_wei_conf.LDB,
            diff_wei_conf.LDC_iter, 1.0, diff_wei_conf.K_blocks);
    init_brgemm_diff_wei(&diff_wei.desc_layer_beta1_, diff_wei_conf.isa,
            diff_wei.kernel_layer_beta1_, m_block_layer, n_diff_wei,
            diff_wei_conf.k_block, diff_wei_conf.LDA_layer, diff_wei_conf.LDB,
            diff_wei_conf.LDC_layer, 1.0, diff_wei_conf.K_blocks);

    if (n_diff_wei_tail) {
        init_brgemm_diff_wei(&diff_wei.desc_iter_N_tail_beta1_,
                diff_wei_conf.isa, diff_wei.kernel_iter_N_tail_beta1_,
                m_block_iter, n_diff_wei_tail, diff_wei_conf.k_block,
                diff_wei_conf.LDA_iter, diff_wei_conf.LDB,
                diff_wei_conf.LDC_iter, 1.0, diff_wei_conf.K_blocks);
        init_brgemm_diff_wei(&diff_wei.desc_layer_N_tail_beta1_,
                diff_wei_conf.isa, diff_wei.kernel_layer_N_tail_beta1_,
                m_block_layer, n_diff_wei_tail, diff_wei_conf.k_block,
                diff_wei_conf.LDA_layer, diff_wei_conf.LDB,
                diff_wei_conf.LDC_layer, 1.0, diff_wei_conf.K_blocks);

        if (diff_wei_conf.k_tail) {
            init_brgemm_diff_wei(&diff_wei.desc_iter_NK_tail_beta1_,
                    diff_wei_conf.isa, diff_wei.kernel_iter_NK_tail_beta1_,
                    m_block_iter, n_diff_wei_tail, diff_wei_conf.k_tail,
                    diff_wei_conf.LDA_iter, diff_wei_conf.LDB,
                    diff_wei_conf.LDC_iter, 1.0, 1);
            init_brgemm_diff_wei(&diff_wei.desc_layer_NK_tail_beta1_,
                    diff_wei_conf.isa, diff_wei.kernel_layer_NK_tail_beta1_,
                    m_block_layer, n_diff_wei_tail, diff_wei_conf.k_tail,
                    diff_wei_conf.LDA_layer, diff_wei_conf.LDB,
                    diff_wei_conf.LDC_layer, 1.0, 1);
        }
    }

    if (diff_wei_conf.k_tail) {
        init_brgemm_diff_wei(&diff_wei.desc_iter_K_tail_beta1_,
                diff_wei_conf.isa, diff_wei.kernel_iter_K_tail_beta1_,
                m_block_iter, n_diff_wei, diff_wei_conf.k_tail,
                diff_wei_conf.LDA_iter, diff_wei_conf.LDB,
                diff_wei_conf.LDC_iter, 1.0, 1);
        init_brgemm_diff_wei(&diff_wei.desc_layer_K_tail_beta1_,
                diff_wei_conf.isa, diff_wei.kernel_layer_K_tail_beta1_,
                m_block_layer, n_diff_wei, diff_wei_conf.k_tail,
                diff_wei_conf.LDA_layer, diff_wei_conf.LDB,
                diff_wei_conf.LDC_layer, 1.0, 1);
    }

    const bool is_xf16_amx_wei = rnn.is_cell_dt_xf16()
            && is_superset(diff_wei_conf.isa, x64::avx512_core_amx);

    if (is_xf16_amx_wei) {
        CHECK(brgemm_init_tiles(
                diff_wei.desc_iter_beta1_, diff_wei.pallete_buff_iter_));
        CHECK(brgemm_init_tiles(
                diff_wei.desc_layer_beta1_, diff_wei.pallete_buff_layer_));
        if (n_diff_wei_tail) {
            CHECK(brgemm_init_tiles(diff_wei.desc_iter_N_tail_beta1_,
                    diff_wei.pallete_buff_iter_n_tail_));
            CHECK(brgemm_init_tiles(diff_wei.desc_layer_N_tail_beta1_,
                    diff_wei.pallete_buff_layer_n_tail_));

            if (diff_wei_conf.k_tail) {
                CHECK(brgemm_init_tiles(diff_wei.desc_iter_NK_tail_beta1_,
                        diff_wei.pallete_buff_iter_nk_tail_));
                CHECK(brgemm_init_tiles(diff_wei.desc_layer_NK_tail_beta1_,
                        diff_wei.pallete_buff_layer_nk_tail_));
            }
        }

        if (diff_wei_conf.k_tail) {
            CHECK(brgemm_init_tiles(diff_wei.desc_iter_K_tail_beta1_,
                    diff_wei.pallete_buff_iter_k_tail_));
            CHECK(brgemm_init_tiles(diff_wei.desc_layer_K_tail_beta1_,
                    diff_wei.pallete_buff_layer_k_tail_));
        }
    }

    // Creating temporary matmul configuration descriptor to use copy_B jit
    // kernels from brgemm matmul copy routines for reordering scratch gates in
    // diff_wei rnn brgemm implementation.
    // TODO: provide unification of jit-based copy routines with implementation
    // independent interface
    matmul::brgemm_matmul_conf_t tmp_matmul_conf_for_reorder;
    tmp_matmul_conf_for_reorder.isa = rnn.brgemm_isa;
    tmp_matmul_conf_for_reorder.wei_tag = format_tag::ab;
    tmp_matmul_conf_for_reorder.N = rnn.scratch_gates_ld;
    tmp_matmul_conf_for_reorder.K = rnn.mb;
    tmp_matmul_conf_for_reorder.wei_n_blk = tmp_matmul_conf_for_reorder.N_blk
            = diff_wei_conf.n_block;
    tmp_matmul_conf_for_reorder.N_tail = diff_wei_conf.n_tail;
    tmp_matmul_conf_for_reorder.LDB = diff_wei_conf.LDB;
    tmp_matmul_conf_for_reorder.src_dt = tmp_matmul_conf_for_reorder.wei_dt
            = rnn.cell_dt;
    tmp_matmul_conf_for_reorder.a_dt_sz = tmp_matmul_conf_for_reorder.tr_a_dt_sz
            = types::data_type_size(tmp_matmul_conf_for_reorder.src_dt);
    tmp_matmul_conf_for_reorder.b_dt_sz = tmp_matmul_conf_for_reorder.tr_b_dt_sz
            = types::data_type_size(tmp_matmul_conf_for_reorder.wei_dt);
    tmp_matmul_conf_for_reorder.copy_B_wei_stride
            = tmp_matmul_conf_for_reorder.N
            * tmp_matmul_conf_for_reorder.b_dt_sz;
    tmp_matmul_conf_for_reorder.transposed_B = false;
    tmp_matmul_conf_for_reorder.is_bf16_with_int_wei = false;
    tmp_matmul_conf_for_reorder.with_wei_decompression = false;
    CHECK(matmul::create_brgemm_matmul_copy_b(
            diff_wei.srcatch_gates_reorder_kernel_,
            &tmp_matmul_conf_for_reorder));

    return status::success;
}

status_t rnn_brgemm_t<prop_kind::backward>::init_kernels(
        const cpu::rnn_utils::rnn_conf_t &rnn, data_type_t src_type,
        data_type_t weights_type) {

    CHECK(init_kernels_diff_src(diff_src_, rnn, src_type, weights_type));
    CHECK(init_kernels_diff_wei(diff_wei_, rnn, src_type, weights_type));
    if (rnn.is_lstm_peephole) CHECK(init_peephole_kernels(rnn));

    const auto n_diff_wei_tail
            = nstl::min(rnn.diff_wei_brgemm.N, rnn.diff_wei_brgemm.n_tail);
    kernel_gates_reduction_
            = utils::make_unique<jit_gates_reduction_t>(rnn, false /*n_tail*/);
    kernel_gates_reduction_->create_kernel();

    if (n_diff_wei_tail) {
        kernel_gates_reduction_tail_
                = utils::make_unique<jit_gates_reduction_t>(
                        rnn, true /*n_tail*/);
        kernel_gates_reduction_tail_->create_kernel();
    }

    if (rnn.mb == 1) {
        if (utils::one_of(src_type, data_type::bf16, data_type::f16)) {
            const bool is_m_block_equal = rnn.slc == rnn.sic;
            const auto m_block_iter = is_m_block_equal
                    ? rnn.diff_wei_brgemm.m_block
                    : rnn.diff_wei_brgemm.M_iter;

            kernel_transpose_single_row_iter_
                    = utils::make_unique<jit_brgemm_transpose_single_row_t>(
                            m_block_iter);
            CHECK(kernel_transpose_single_row_iter_->create_kernel());

            if (!is_m_block_equal) {
                const auto m_block_layer = rnn.diff_wei_brgemm.M_layer;
                kernel_transpose_single_row_layer_
                        = utils::make_unique<jit_brgemm_transpose_single_row_t>(
                                m_block_layer);
                CHECK(kernel_transpose_single_row_layer_->create_kernel());
            }
        }
    } else {
        assert(rnn.diff_wei_brgemm.global_transpose);

        jit_brgemm_primitive_conf_t trans_conf;
        trans_conf.isa = rnn.brgemm_isa;
        trans_conf.prop_kind = dnnl_backward_weights;
        trans_conf.src_dt = src_type;
        static constexpr int blk_size = 16;
        trans_conf.os_block = blk_size; // src's rows block size
        trans_conf.ic_block = blk_size; // src's cols block size
        trans_conf.M = 0;
        const auto rnd_up_size = data_type_vnni_granularity(src_type);
        const auto os_padded = utils::rnd_up(rnn.mb, rnd_up_size);
        trans_conf.os = os_padded;
        trans_conf.LDA = os_padded; // dst's leading dim
        trans_conf.K_tail = rnn.mb % blk_size; // src's rows tail

        const int LDA_iter[]
                = {rnn.src_iter_ld_, rnn.dst_layer_ld_, rnn.ws_states_iter_ld};
        trans_conf.M_tail = rnn.sic % blk_size; // src's cols tail
        for (int i = 0; i < num_base_kernels_; i++) {
            trans_conf.ic = LDA_iter[i];
            CHECK(create_brgemm_trans_src(
                    kernel_transpose_iter_[i], &trans_conf));
        }

        const int LDA_layer[]
                = {rnn.src_layer_ld_, rnn.dst_iter_ld_, rnn.ws_states_layer_ld};
        trans_conf.M_tail = rnn.slc % blk_size; // src's cols tail
        for (int i = 0; i < num_base_kernels_; i++) {
            trans_conf.ic = LDA_layer[i];
            CHECK(create_brgemm_trans_src(
                    kernel_transpose_layer_[i], &trans_conf));
        }
    }

    return status::success;
}

status_t rnn_brgemm_t<prop_kind::backward>::init_peephole_kernels(
        const cpu::rnn_utils::rnn_conf_t &rnn) {

    if (rnn.dhc_blocks_peephole) {
        kernel_peephole_ = utils::make_unique<jit_diff_weights_peephole_t>(
                rnn, rnn.dhc_block_peephole);
        CHECK(kernel_peephole_->create_kernel());
    }

    if (rnn.dhc_tail_peephole) {
        kernel_peephole_tail_ = utils::make_unique<jit_diff_weights_peephole_t>(
                rnn, rnn.dhc_tail_peephole);
        CHECK(kernel_peephole_tail_->create_kernel());
    }

    return status::success;
}

} // namespace rnn_brgemm_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
