/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include "cpu/rnn/rnn_utils.hpp"
#include "cpu/x64/rnn/rnn_brgemm_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace rnn_brgemm_utils {

namespace {

std::tuple<dim_t, dim_t, x64::cpu_isa_t> brgemm_calc_k_block_with_isa(
        const cpu::rnn_utils::rnn_conf_t &rnn, alg_kind_t cell_kind,
        dim_t src_layer_type_size, dim_t padding, dim_t As, dim_t Bs, dim_t Cs,
        dim_t l2_cache_size);
std::pair<dim_t, dim_t> brgemm_calc_k_block_amx(
        const cpu::rnn_utils::rnn_conf_t &rnn);
std::pair<dim_t, dim_t> brgemm_calc_k_block_vanilla_rnn(
        const cpu::rnn_utils::rnn_conf_t &rnn, dim_t src_layer_type_size,
        dim_t As, dim_t Bs, dim_t Cs, dim_t l2_cache_size);

dim_t brgemm_calc_m_block(const cpu::rnn_utils::rnn_conf_t &rnn,
        alg_kind_t cell_kind, float work_by_N, dim_t As, dim_t Bs, dim_t Cs,
        dim_t l2_cache_size);
dim_t brgemm_calc_m_block_vanilla_rnn(const cpu::rnn_utils::rnn_conf_t &rnn,
        float work_by_N, dim_t As, dim_t Bs, dim_t Cs, dim_t l2_cache_size);
dim_t brgemm_calc_m_block_lstm(const cpu::rnn_utils::rnn_conf_t &rnn,
        float work_by_N, dim_t As, dim_t Cs, dim_t l2_cache_size);
dim_t adjust_m_block_lstm(const cpu::rnn_utils::rnn_conf_t &rnn);
dim_t brgemm_calc_m_block_vanilla_rnn(const cpu::rnn_utils::rnn_conf_t &rnn,
        float work_by_N, dim_t As, dim_t Bs, dim_t Cs, dim_t l2_cache_size);

std::tuple<dim_t, dim_t, x64::cpu_isa_t> brgemm_calc_k_block_with_isa(
        const cpu::rnn_utils::rnn_conf_t &rnn, alg_kind_t cell_kind,
        dim_t src_layer_type_size, dim_t padding, dim_t As, dim_t Bs, dim_t Cs,
        dim_t l2_cache_size) {
    const bool is_amx_int8
            = rnn.is_int8() && x64::mayiuse(x64::avx512_core_bf16_amx_int8);
    const bool is_amx_bf16
            = rnn.is_bf16() && x64::mayiuse(x64::avx512_core_bf16_amx_bf16);

    dim_t k1_block = rnn.K1;
    dim_t k2_block = rnn.K2;
    auto isa = x64::isa_any;

    if (is_amx_int8 || is_amx_bf16) {
        const auto result = brgemm_calc_k_block_amx(rnn);
        const auto k1_block_amx = result.first;
        const auto k2_block_amx = result.second;
        const auto k1_block_tail = rnn.K1 % k1_block_amx;
        const auto k2_block_tail = rnn.K2 % k2_block_amx;
        const bool amx_block_invalid = k1_block_tail % padding
                || k2_block_tail % padding || k1_block_amx % padding
                || k2_block_amx % padding;

        if (amx_block_invalid) {
            isa = is_amx_int8 ? x64::avx512_core_vnni : x64::avx512_core_bf16;
        } else {
            k1_block = k1_block_amx;
            k2_block = k2_block_amx;
            isa = is_amx_int8 ? x64::avx512_core_bf16_amx_int8
                              : x64::avx512_core_bf16_amx_bf16;
        }
    }

    if (cell_kind == alg_kind::vanilla_rnn)
        std::tie(k1_block, k2_block) = brgemm_calc_k_block_vanilla_rnn(
                rnn, src_layer_type_size, As, Bs, Cs, l2_cache_size);

    return std::make_tuple(k1_block, k2_block, isa);
}

std::pair<dim_t, dim_t> brgemm_calc_k_block_amx(
        const cpu::rnn_utils::rnn_conf_t &rnn) {
    const bool is_amx_int8
            = rnn.is_int8() && x64::mayiuse(x64::avx512_core_bf16_amx_int8);
    const dim_t max_row_width = is_amx_int8 ? 64 : 32;

    dim_t k1_block = nstl::min(rnn.K1, max_row_width);
    dim_t k2_block = nstl::min(rnn.K2, max_row_width);

    if (k1_block <= rnn.K1 || k2_block <= rnn.K2) {
        const dim_t t_k_block = nstl::min(k1_block, k2_block);
        k2_block = k1_block = t_k_block;
    }

    return std::make_pair(k1_block, k2_block);
}

std::pair<dim_t, dim_t> brgemm_calc_k_block_vanilla_rnn(
        const cpu::rnn_utils::rnn_conf_t &rnn, dim_t src_layer_type_size,
        dim_t As, dim_t Bs, dim_t Cs, dim_t l2_cache_size) {

    //Heuristics experimentally selected.
    const bool should_adjust_by_l2 = static_cast<float>(As + Bs + Cs)
            >= 0.25 * static_cast<float>(l2_cache_size);
    dim_t k1_block = rnn.K1;
    dim_t k2_block = rnn.K2;

    if (should_adjust_by_l2) {
        int block_size = (l2_cache_size * 0.25f)
                / ((rnn.M + rnn.n_block) * src_layer_type_size);

        if (rnn.is_bf16()) {
            // due to weights format ldgOI32o2i block_size should be even
            block_size -= (block_size % 2);
            block_size = nstl::max(block_size, 0);
        }
        if (block_size) {
            k1_block = nstl::min(rnn.K1, static_cast<dim_t>(block_size));
            k2_block = nstl::min(rnn.K2, static_cast<dim_t>(block_size));
        }
    }

    return std::make_pair(k1_block, k2_block);
}

dim_t brgemm_calc_m_block(const cpu::rnn_utils::rnn_conf_t &rnn,
        alg_kind_t cell_kind, float work_by_N, dim_t As, dim_t Bs, dim_t Cs,
        dim_t l2_cache_size) {
    if (cell_kind == alg_kind::vanilla_rnn)
        return brgemm_calc_m_block_vanilla_rnn(
                rnn, work_by_N, As, Bs, Cs, l2_cache_size);
    else
        return brgemm_calc_m_block_lstm(rnn, work_by_N, As, Cs, l2_cache_size);
}

dim_t brgemm_calc_m_block_vanilla_rnn(const cpu::rnn_utils::rnn_conf_t &rnn,
        float work_by_N, dim_t As, dim_t Bs, dim_t Cs, dim_t l2_cache_size) {

    //Heuristics experimentally selected.
    const float decimal_n_factor = work_by_N - std::floor(work_by_N);
    static constexpr float thread_balance_threashold = 0.9;

    dim_t m_block = rnn.M;

    if (work_by_N < 1.0)
        return adjust_m_block_lstm(rnn);
    else if (decimal_n_factor < thread_balance_threashold
            && decimal_n_factor != 0.0f) {

        const dim_t m_block_start = rnn.M / 2;
        const dim_t m_block_end = 4;

        float max_decimal_mn = 0.0;
        dim_t best_candidate = 0.0;
        bool found_best_solution = false;

        for (dim_t m_block_it = m_block_start; m_block_it >= m_block_end;
                m_block_it--) {
            if (rnn.M % m_block_it == 0) {
                const auto m_blocks = rnn.M / m_block_it;
                const auto work_by_MN
                        = static_cast<float>(m_blocks * rnn.N_blocks)
                        / rnn.nthr;

                const float work_by_MN_decimal
                        = work_by_MN - std::floor(work_by_MN);

                static constexpr float tolerance = 0.01;
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

dim_t brgemm_calc_m_block_lstm(const cpu::rnn_utils::rnn_conf_t &rnn,
        float work_by_N, dim_t As, dim_t Cs, dim_t l2_cache_size) {
    const bool adj_by_l2 = rnn.is_f32()
            ? true
            : (static_cast<float>(As + Cs)
                    < 0.6 * static_cast<float>(l2_cache_size));

    if (work_by_N > 2.0 || (work_by_N > 1.0 && adj_by_l2))
        return rnn.M;
    else
        return adjust_m_block_lstm(rnn);
}

dim_t adjust_m_block_lstm(const cpu::rnn_utils::rnn_conf_t &rnn) {

    const dim_t max_m_blocks
            = ((rnn.is_int8_amx() || rnn.is_bf16_amx()) ? 1 : 4)
            * utils::div_up(rnn.nthr, rnn.N_blocks);
    const dim_t max_m_value
            = (rnn.is_int8_amx() || rnn.is_bf16_amx()) ? 64 : 24;
    const dim_t max_M
            = nstl::min(max_m_value, nstl::max((dim_t)1, rnn.M / max_m_blocks));
    const dim_t min_M = 4;

    dim_t m_block = 1;
    for (dim_t m = max_M; m >= min_M; m--)
        if (rnn.M % m == 0) {
            m_block = m;
            break;
        }
    if (m_block == 1) m_block = rnn.M;

    return m_block;
}
} // namespace

void rnn_brgemm_t::init_scratchpad(const cpu::rnn_utils::rnn_conf_t &rnn,
        memory_tracking::registrar_t &scratchpad, dim_t gemm_acc_type_size,
        dim_t gemm_acc_align) {

    using namespace memory_tracking::names;

    if (rnn.is_int8_amx() || rnn.is_bf16_amx()) {
        size_t n_elements = rnn.m_block * rnn.n_block;
        scratchpad.book(key_brgemm_primitive_buffer, rnn.nthr * n_elements,
                gemm_acc_type_size, gemm_acc_align);
    }

    const int max_K_Block = nstl::max(rnn.KB1_blocks + 1,
            nstl::max(rnn.KBproj_blocks + 1, rnn.KB2_blocks + 1));
    scratchpad.template book<x64::brgemm_batch_element_t>(
            key_brgemm_primitive_batch, max_K_Block * rnn.nthr);
}

status_t rnn_brgemm_t::configure_brgemm(cpu::rnn_utils::rnn_conf_t &rnn,
        alg_kind_t cell_kind, dim_t src_layer_type_size,
        dim_t scratch_type_size) {

    rnn.M = rnn.mb;
    rnn.N = rnn.dhc;
    rnn.K1 = rnn.slc;
    rnn.K2 = rnn.sic;

    rnn.nthr = dnnl_get_max_threads();
    rnn.n_block = 32;
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

    const dim_t padding = (rnn.is_int8()) ? 4 : (rnn.is_bf16()) ? 2 : 1;
    rnn.K1padded = utils::rnd_up(rnn.K1, padding);
    rnn.K2padded = utils::rnd_up(rnn.K2, padding);

    std::tie(rnn.k1_block, rnn.k2_block, rnn.brgemm_isa)
            = brgemm_calc_k_block_with_isa(rnn, cell_kind, src_layer_type_size,
                    padding, As, Bs, Cs, l2_cache_size);
    rnn.KB1_blocks = rnn.K1 / rnn.k1_block;
    rnn.KB2_blocks = rnn.K2 / rnn.k2_block;
    rnn.k1_tail = rnn.K1 % rnn.k1_block;
    rnn.k2_tail = rnn.K2 % rnn.k2_block;
    rnn.m_block = brgemm_calc_m_block(
            rnn, cell_kind, work_by_N, As, Bs, Cs, l2_cache_size);
    rnn.M_blocks = rnn.M / rnn.m_block;
    rnn.unfused_post_gemm
            = cell_kind == alg_kind::vanilla_lstm ? (rnn.M_blocks == 1) : false;

    rnn.LDA1[0] = rnn.src_layer_ld_;
    rnn.LDA1[1] = rnn.dst_iter_ld_;
    rnn.LDA1[2] = rnn.ws_states_layer_ld;

    rnn.LDA2[0] = rnn.src_iter_ld_;
    rnn.LDA2[1] = rnn.dst_layer_ld_;
    rnn.LDA2[2] = rnn.ws_states_iter_ld;

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
        if (rnn.is_int8_amx() || rnn.is_bf16_amx()) {
            const dim_t max_row_width = rnn.is_int8_amx() ? 64 : 32;
            rnn.kproj_block = nstl::min(rnn.Kproj, (dim_t)max_row_width);

            rnn.KBproj_blocks = rnn.Kproj / rnn.kproj_block;
            rnn.kproj_tail = rnn.Kproj % rnn.kproj_block;

            if ((rnn.kproj_tail % padding) || (rnn.kproj_block % padding)) {
                rnn.kproj_block = rnn.Kproj;
                rnn.kproj_tail = 0;
                rnn.brgemm_isa = rnn.is_int8() ? x64::avx512_core_vnni
                                               : x64::avx512_core_bf16;
            } else {
                rnn.brgemm_isa = rnn.is_int8() ? x64::avx512_core_bf16_amx_int8
                                               : x64::avx512_core_bf16_amx_bf16;
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
    return status::success;
}

status_t init_brgemm_kernel(const cpu::rnn_utils::rnn_conf_t &rnn,
        x64::brgemm_t *desc, x64::cpu_isa_t isa, impl::data_type_t src_type,
        impl::data_type_t weights_type,
        std::unique_ptr<x64::brgemm_kernel_t> &ker, dim_t M, dim_t N, dim_t K,
        dim_t LDA, dim_t LDB, dim_t LDC, float beta, dim_t max_bs) {
    bool transA = false;
    bool transB = false;
    x64::brgemm_layout_t layout = x64::brgemm_row_major;
    CHECK(brgemm_desc_init(desc, isa, x64::brgemm_addr, src_type, weights_type,
            transA, transB, layout, 1.0, beta, LDA, LDB, LDC, M, N, K));

    if (!rnn.is_int8_amx() && !rnn.is_bf16_amx()) {
        x64::brgemm_attr_t brgattr;
        brgattr.max_bs = max_bs;
        brgattr.max_top_vpad = 0;
        brgattr.max_bottom_vpad = 0;
        brgemm_desc_set_attr(desc, brgattr);
    }

    x64::brgemm_kernel_t *_t_ptr;
    CHECK(brgemm_kernel_create(&_t_ptr, *desc));
    safe_ptr_assign<x64::brgemm_kernel_t>(ker, _t_ptr);

    return status::success;
};

void rnn_brgemm_t::init_kernels(const cpu::rnn_utils::rnn_conf_t &rnn,
        data_type_t src_type, data_type_t weights_type) {

    const auto init_brgemm
            = [&](x64::brgemm_t *desc, x64::cpu_isa_t isa,
                      std::unique_ptr<x64::brgemm_kernel_t> &ker, dim_t M,
                      dim_t N, dim_t K, dim_t LDA, dim_t LDB, dim_t LDC,
                      float beta, dim_t max_bs) {
                  init_brgemm_kernel(rnn, desc, isa, src_type, weights_type,
                          ker, M, N, K, LDA, LDB, LDC, beta, max_bs);
              };

    const int brgemm_n = nstl::min(rnn.N, rnn.n_block);
    const int brgemm_n_tail = nstl::min(rnn.N, rnn.n_tail);
    for (int i = 0; i < 3; i++) {
        init_brgemm(&desc_layer_b0_[i], rnn.brgemm_isa, kernel_layer_b0_[i],
                rnn.m_block, brgemm_n, rnn.k1_block, rnn.LDA1[i], rnn.LDB1,
                rnn.LDC, 0.0, rnn.KB1_blocks);
        init_brgemm(&desc_iter_b0_[i], rnn.brgemm_isa, kernel_iter_b0_[i],
                rnn.m_block, brgemm_n, rnn.k2_block, rnn.LDA2[i], rnn.LDB2,
                rnn.LDC, 0.0, rnn.KB2_blocks);
        init_brgemm(&desc_iter_b1_[i], rnn.brgemm_isa, kernel_iter_b1_[i],
                rnn.m_block, brgemm_n, rnn.k2_block, rnn.LDA2[i], rnn.LDB2,
                rnn.LDC, 1.0, rnn.KB2_blocks);
        if (rnn.n_tail) {
            init_brgemm(&desc_layer_N_tail_b0_[i], rnn.brgemm_isa,
                    kernel_layer_N_tail_b0_[i], rnn.m_block, brgemm_n_tail,
                    rnn.k1_block, rnn.LDA1[i], rnn.LDB1, rnn.LDC, 0.0,
                    rnn.KB1_blocks);
            init_brgemm(&desc_iter_N_tail_b0_[i], rnn.brgemm_isa,
                    kernel_iter_N_tail_b0_[i], rnn.m_block, brgemm_n_tail,
                    rnn.k2_block, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 0.0,
                    rnn.KB2_blocks);
            init_brgemm(&desc_iter_N_tail_b1_[i], rnn.brgemm_isa,
                    kernel_iter_N_tail_b1_[i], rnn.m_block, brgemm_n_tail,
                    rnn.k2_block, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 1.0,
                    rnn.KB2_blocks);
        }
        if (rnn.k1_tail)
            init_brgemm(&desc_layer_K1_tail_b1_[i], rnn.brgemm_isa,
                    kernel_layer_K1_tail_b1_[i], rnn.m_block, brgemm_n,
                    rnn.k1_tail, rnn.LDA1[i], rnn.LDB1, rnn.LDC, 1.0, 1);
        if (rnn.k2_tail)
            init_brgemm(&desc_iter_K2_tail_b1_[i], rnn.brgemm_isa,
                    kernel_iter_K2_tail_b1_[i], rnn.m_block, brgemm_n,
                    rnn.k2_tail, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 1.0, 1);
        if (rnn.k1_tail && rnn.n_tail)
            init_brgemm(&desc_layer_NK1_tail_b1_[i], rnn.brgemm_isa,
                    kernel_layer_NK1_tail_b1_[i], rnn.m_block, brgemm_n_tail,
                    rnn.k1_tail, rnn.LDA1[i], rnn.LDB1, rnn.LDC, 1.0, 1);
        if (rnn.k2_tail && rnn.n_tail)
            init_brgemm(&desc_iter_NK2_tail_b1_[i], rnn.brgemm_isa,
                    kernel_iter_NK2_tail_b1_[i], rnn.m_block, brgemm_n_tail,
                    rnn.k2_tail, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 1.0, 1);
    }
    if (rnn.is_lstm_projection) {
        const dim_t brgemm_np = nstl::min(rnn.Nproj, rnn.n_block);
        const dim_t brgemm_np_tail = nstl::min(rnn.Nproj, rnn.nproj_tail);
        const int n_kernel = (rnn.dt_conf == cpu::rnn_utils::all_f32) ? 4 : 1;
        for (int i = 0; i < n_kernel; i++) {
            init_brgemm(&desc_proj_b0_[i], rnn.brgemm_isa, kernel_proj_b0_[i],
                    rnn.m_block, brgemm_np, rnn.kproj_block, rnn.LDAproj,
                    rnn.LDBproj, rnn.LDCproj[i], 0.0, rnn.KBproj_blocks);
            if (rnn.nproj_tail) {
                init_brgemm(&desc_proj_N_tail_b0_[i], rnn.brgemm_isa,
                        kernel_proj_N_tail_b0_[i], rnn.m_block, brgemm_np_tail,
                        rnn.kproj_block, rnn.LDAproj, rnn.LDBproj,
                        rnn.LDCproj[i], 0.0, rnn.KBproj_blocks);
                init_brgemm(&desc_proj_N_tail_b1_[i], rnn.brgemm_isa,
                        kernel_proj_N_tail_b1_[i], rnn.m_block, brgemm_np_tail,
                        rnn.kproj_block, rnn.LDAproj, rnn.LDBproj,
                        rnn.LDCproj[i], 1.0, rnn.KBproj_blocks);
            }
            if (rnn.is_int8_amx() || rnn.is_bf16_amx()) {
                if (rnn.kproj_tail)
                    init_brgemm(&desc_proj_K_tail_b1_[i], rnn.brgemm_isa,
                            kernel_proj_K_tail_b1_[i], rnn.m_block, brgemm_np,
                            rnn.kproj_tail, rnn.LDAproj, rnn.LDBproj,
                            rnn.LDCproj[i], 1.0, 1);
                if (rnn.kproj_tail && rnn.nproj_tail)
                    init_brgemm(&desc_proj_NK_tail_b1_[i], rnn.brgemm_isa,
                            kernel_proj_NK_tail_b1_[i], rnn.m_block,
                            brgemm_np_tail, rnn.kproj_tail, rnn.LDAproj,
                            rnn.LDBproj, rnn.LDCproj[i], 1.0, 1);
            }
        }
    }
    if (rnn.is_int8_amx() || rnn.is_bf16_amx()) {
        brgemm_init_tiles(desc_layer_b0_[0], pallete_buff_);
        if (rnn.n_tail)
            brgemm_init_tiles(desc_layer_N_tail_b0_[0], pallete_buff_n_tail_);
        if (rnn.k1_tail)
            brgemm_init_tiles(desc_layer_K1_tail_b1_[0], pallete_buff_k1_tail_);
        if (rnn.k2_tail)
            brgemm_init_tiles(desc_iter_K2_tail_b1_[0], pallete_buff_k2_tail_);
        if (rnn.k1_tail && rnn.n_tail)
            brgemm_init_tiles(
                    desc_layer_NK1_tail_b1_[0], pallete_buff_nk1_tail_);
        if (rnn.k2_tail && rnn.n_tail)
            brgemm_init_tiles(
                    desc_iter_NK2_tail_b1_[0], pallete_buff_nk2_tail_);
        if (rnn.is_lstm_projection) {
            brgemm_init_tiles(desc_proj_b0_[0], pallete_buff_proj_);
            if (rnn.nproj_tail)
                brgemm_init_tiles(
                        desc_proj_N_tail_b0_[0], pallete_buff_nproj_tail_);
            if (rnn.kproj_tail)
                brgemm_init_tiles(
                        desc_proj_K_tail_b1_[0], pallete_buff_kproj_tail_);
            if (rnn.kproj_tail && rnn.nproj_tail)
                brgemm_init_tiles(
                        desc_proj_NK_tail_b1_[0], pallete_buff_nkproj_tail_);
        }
    }
}

} // namespace rnn_brgemm_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl