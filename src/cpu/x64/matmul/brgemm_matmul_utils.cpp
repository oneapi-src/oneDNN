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

#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"

#include "cpu/matmul/matmul_utils.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

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

// TODO: add support of post-ops with multiple binary and eltwise execution
bool post_ops_ok(brgemm_matmul_conf_t &bgmmc, const primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d) {
    using namespace injector;

    const auto &post_ops = attr.post_ops_;

    const bool is_binary_po_channel_bcast
            = binary_injector_utils::bcast_strategy_present(
                    binary_injector_utils::extract_bcast_strategies(
                            post_ops.entry_, dst_d),
                    broadcasting_strategy_t::per_mb_spatial);
    const bool supported_channel_bcast
            = IMPLICATION(is_binary_po_channel_bcast, (dst_d).ndims() == 4);

    return supported_channel_bcast
            && injector::post_ops_ok(post_ops_ok_args_t(get_max_cpu_isa(),
                    {sum, eltwise, binary}, post_ops, &dst_d,
                    false /*sum_at_pos_0_only*/,
                    false /*sum_requires_scale_one*/,
                    false /*sum_requires_zp_zero*/,
                    {broadcasting_strategy_t::per_oc,
                            broadcasting_strategy_t::scalar,
                            broadcasting_strategy_t::per_mb_spatial,
                            broadcasting_strategy_t::no_broadcast}));
}

brgemm_broadcast_t get_zp_type(const primitive_attr_t &attr, int arg) {
    return attr.zero_points_.has_default_values(arg)
            ? brgemm_broadcast_t::none
            : brgemm_broadcast_t::per_tensor;
}

void init_aux_values(brgemm_matmul_conf_t &bgmmc,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &wei_d,
        const memory_desc_wrapper &dst_d);

struct matmul_blocking_params_t : public brgemm_matmul_conf_t {
    matmul_blocking_params_t() { init_zero(); }
    matmul_blocking_params_t(const brgemm_matmul_conf_t &bgmmc)
        : brgemm_matmul_conf_t(bgmmc) {
        init_from_conf();
    }

    void set_blocking_parameters(int nthr_k, int n_blk, int n_chunk_size,
            int m_blk, int m_chunk_size);
    void update_configuration(brgemm_matmul_conf_t &bgmmc);
    float get_blocking_scores() { return efficiency_score_; }

    static size_t L2_threshold;

private:
    // num threads for parallelism wrt k dimension
    int nthr_k_;
    // num threads for parallelism wrt m, n and batch dimensions
    int nthr_mnb_;
    int nthr_;
    dim_t n_blk_, n_chunk_size_, k_chunk_elems_;
    dim_t m_blk_, m_chunk_size_, m_chunk_elems_;
    dim_t k_blk_, k_chunk_size_, n_chunk_elems_;

    dim_t current_lda_;
    bool need_buf_c_;
    size_t blocking_chunk_mem_size_;
    float efficiency_score_;

    void init_zero();
    void init_from_conf();
    void update_k_blocking_dependent_params();
    dim_t get_actual_lda();
    bool is_buffer_c_required();
    size_t calculate_chunk_memory_size();
    float get_thread_balance_scores();
    float get_copied_data_reusage_scores();
    float get_L2_utilization_scores();
    float calculate_blocking_scores();
};

size_t matmul_blocking_params_t::L2_threshold
        = 3 * platform::get_per_core_cache_size(2) / 4;

status_t init_brgemm_matmul_conf(cpu_isa_t isa, brgemm_matmul_conf_t &bgmmc,
        const matmul_desc_t &mmd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);

    bgmmc = zero<decltype(bgmmc)>();
    bgmmc.isa = isa;
    bgmmc.nthr = dnnl_get_max_threads();
    bgmmc.brg_type = brgemm_addr;

    bgmmc.src_dt = src_d.data_type();
    bgmmc.dst_dt = dst_d.data_type();
    bgmmc.wei_dt = weights_d.data_type();

    bgmmc.with_bias = mmd.bias_desc.format_kind != format_kind::undef;
    bgmmc.bia_dt = bgmmc.with_bias ? mmd.bias_desc.data_type : data_type::undef;
    bgmmc.s8s8_compensation_required
            = isa == avx512_core_vnni && bgmmc.src_dt == s8;
    bgmmc.ndims = dst_d.ndims();

    const bool is_int8 = one_of(bgmmc.src_dt, u8, s8) && bgmmc.wei_dt == s8
            && one_of(bgmmc.dst_dt, u8, s8, s32, f32, bf16);
    const bool is_bf16 = everyone_is(bf16, bgmmc.src_dt, bgmmc.wei_dt)
            && one_of(bgmmc.dst_dt, bf16, f32);
    const bool is_amx_int8 = isa == avx512_core_bf16_amx_int8;
    const bool is_amx_bf16 = isa == avx512_core_bf16_amx_bf16;
    bgmmc.is_amx = is_amx_int8 || is_amx_bf16;

    bgmmc.acc_dt = is_int8 ? s32 : f32;

    bgmmc.a_dt_sz = types::data_type_size(bgmmc.src_dt);
    bgmmc.b_dt_sz = types::data_type_size(bgmmc.wei_dt);
    bgmmc.c_dt_sz = types::data_type_size(bgmmc.dst_dt);
    bgmmc.acc_dt_sz = types::data_type_size(bgmmc.acc_dt);
    if (bgmmc.with_bias) bgmmc.bias_dt_sz = types::data_type_size(bgmmc.bia_dt);

    bgmmc.with_scales = !attr.output_scales_.has_default_values();
    if (bgmmc.with_scales) {
        const auto &oscales = attr.output_scales_;
        bgmmc.is_oscale_per_n = oscales.mask_ == 1 << (bgmmc.ndims - 1);

        // only common and per-oc-channel scales are supported
        const bool oscales_ok = oscales.mask_ == 0 || bgmmc.is_oscale_per_n;
        if (!oscales_ok) return status::unimplemented;
    }

    const auto &p = attr.post_ops_;
    bgmmc.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    bgmmc.with_eltwise = eltwise_ind != -1;
    const int binary_ind = p.find(primitive_kind::binary);
    bgmmc.with_binary = binary_ind != -1;

    if (!post_ops_ok(bgmmc, attr, dst_d)) return status::unimplemented;

    bgmmc.src_zp_type = get_zp_type(attr, DNNL_ARG_SRC);
    bgmmc.wei_zp_type = get_zp_type(attr, DNNL_ARG_WEIGHTS);
    bgmmc.dst_zp_type = get_zp_type(attr, DNNL_ARG_DST);

    if (!IMPLICATION(is_int8,
                one_of(isa, avx512_core_vnni, avx512_core_bf16_amx_int8)))
        return status::unimplemented;

    if (!IMPLICATION(is_bf16, isa == avx512_core_bf16_amx_bf16))
        return status::unimplemented;

    matmul_helper_t helper(src_d, weights_d, dst_d);

    bgmmc.batch_ndims = bgmmc.ndims - 2;

    auto init_bcast_desc = [&](brgemm_matmul_bcast_desc_t &bd,
                                   const dims_t &inp_dims) {
        const int ndims = bgmmc.batch_ndims;
        const int mask = 1 << (ndims - 1);
        bd.first_bcast_dim_to_last_batch_dim_prod = bgmmc.batch;
        for (int d = 0; d < ndims; ++d) {
            bd.batch_dims[d] = dst_d.dims()[d];
            bd.gb_off[d] = (d == 0 ? bgmmc.batch : bd.gb_off[d - 1])
                    / dst_d.dims()[d];
            if (dst_d.dims()[d] != 1 && inp_dims[d] == 1) { // broadcast
                bd.bcast_mask |= (mask >> d);
                if (bd.first_bcast_dim == -1) {
                    bd.first_bcast_dim = d;
                    if (d == 0) // broadcast_dim == B0
                        bd.first_bcast_dim_to_last_batch_dim_prod = bgmmc.batch;
                }
                bd.last_bcast_dim = d;
                bd.bcast_dims_prod *= dst_d.dims()[d];
            }
            if (bd.first_bcast_dim == -1) // broadcast_dim > B0
                bd.first_bcast_dim_to_last_batch_dim_prod /= dst_d.dims()[d];
        }
    };

    bgmmc.M = helper.M();
    bgmmc.N = helper.N();
    bgmmc.K = helper.K();
    bgmmc.batch = helper.batch();
    bgmmc.batch_without_first_dim
            = bgmmc.batch_ndims > 1 ? helper.batch() / dst_d.dims()[0] : 0;

    init_bcast_desc(bgmmc.bcast_A_desc, src_d.dims());
    init_bcast_desc(bgmmc.bcast_B_desc, weights_d.dims());

    // required granularity for k dimension
    bgmmc.required_k_granularity = is_amx_int8 ? 4 : (is_amx_bf16 ? 2 : 1);
    const int required_k_block_granularity
            = is_amx_int8 ? 64 : (is_amx_bf16 ? 32 : 16);

    auto pick_blocked_B_layout = [&](int n_blk) -> format_tag_t {
        if (bgmmc.ndims > 2) return format_tag::undef;

        if (isa == avx512_core_bf16_amx_int8) {
            switch (n_blk) {
                case 64: return BA16a64b4a;
                case 48: return BA16a48b4a;
                case 32: return BA16a32b4a;
                case 16: return BA16a16b4a;
                default: return format_tag::undef;
            }
        }

        if (isa == avx512_core_bf16_amx_bf16) {
            switch (n_blk) {
                case 64: return BA16a64b2a;
                case 48: return BA16a48b2a;
                case 32: return BA16a32b2a;
                case 16: return BA16a16b2a;
                default: return format_tag::undef;
            }
        }

        return format_tag::undef;
    };

    auto get_default_n_block = [&](format_tag_t matrix_b_tag) -> int {
        switch (matrix_b_tag) {
            case BA16a48b2a:
            case BA16a48b4a: return 48;
            case BA16a32b2a:
            case BA16a32b4a: return 32;
            case BA16a16b2a:
            case BA16a16b4a: return 16;
            default: return 64;
        }
    };

    const format_tag_t plain_tensor_layout_tag
            = pick(bgmmc.batch_ndims, ab, abc, abcd, abcde, abcdef, abcdefg,
                    abcdefgh, abcdefghi, abcdefghij, abcdefghijk, abcdefghijkl);
    const format_tag_t transposed_tensor_layout_tag
            = pick(bgmmc.batch_ndims, ba, acb, abdc, abced, abcdfe, abcdegf,
                    abcdefhg, abcdefgih, abcdefghji, abcdefghikj, abcdefghijlk);
    const format_tag_t blocked_64n_B_layout_tag = pick_blocked_B_layout(64);
    const format_tag_t blocked_48n_B_layout_tag = pick_blocked_B_layout(48);
    const format_tag_t blocked_32n_B_layout_tag = pick_blocked_B_layout(32);
    const format_tag_t blocked_16n_B_layout_tag = pick_blocked_B_layout(16);
    const bool blocked_B_layouts_allowed = !one_of(format_tag::undef,
            blocked_64n_B_layout_tag, blocked_48n_B_layout_tag,
            blocked_32n_B_layout_tag, blocked_16n_B_layout_tag);

    auto b_layout_blocked_by_n = [&](format_tag_t matrix_b_tag) -> bool {
        return blocked_B_layouts_allowed
                && one_of(matrix_b_tag, blocked_64n_B_layout_tag,
                        blocked_48n_B_layout_tag, blocked_32n_B_layout_tag,
                        blocked_16n_B_layout_tag);
    };

    bool n_blk_fixed = false;
    const bool any_B_layout = weights_d.format_kind() == format_kind::any;

    auto set_or_check_B_tag = [&](bool init, int n_blk_size = -1) -> status_t {
        if (n_blk_fixed && n_blk_size != bgmmc.wei_n_blk)
            return status::unimplemented;

        bgmmc.wei_k_blk = required_k_block_granularity;
        bgmmc.wei_n_blk
                = init ? get_default_n_block(format_tag::undef) : n_blk_size;

        if (!IMPLICATION(!init, any_B_layout && blocked_B_layouts_allowed))
            return status::success;

        if (any_B_layout) {
            bgmmc.wei_tag = blocked_B_layouts_allowed
                    ? pick_blocked_B_layout(bgmmc.wei_n_blk)
                    : plain_tensor_layout_tag;
            if (format_tag::undef == bgmmc.wei_tag)
                return status::unimplemented;

            CHECK(memory_desc_init_by_tag(weights_md, bgmmc.wei_tag));
        } else {
            bgmmc.wei_tag = blocked_B_layouts_allowed
                    ? memory_desc_matches_one_of_tag(weights_md,
                            plain_tensor_layout_tag,
                            transposed_tensor_layout_tag,
                            blocked_64n_B_layout_tag, blocked_48n_B_layout_tag,
                            blocked_32n_B_layout_tag, blocked_16n_B_layout_tag)
                    : memory_desc_matches_one_of_tag(weights_md,
                            plain_tensor_layout_tag,
                            transposed_tensor_layout_tag);
            if (format_tag::undef == bgmmc.wei_tag)
                return status::unimplemented;

            n_blk_fixed = blocked_B_layouts_allowed
                    && b_layout_blocked_by_n(bgmmc.wei_tag);
        }

        bgmmc.blocked_B = blocked_B_layouts_allowed
                && b_layout_blocked_by_n(bgmmc.wei_tag);
        bgmmc.use_buffer_b = !bgmmc.blocked_B;

        memory_desc_t want_wei_md = weights_md;
        if (bgmmc.src_zp_type != brgemm_broadcast_t::none && bgmmc.blocked_B) {
            want_wei_md.extra.flags
                    |= memory_extra_flags::compensation_conv_asymmetric_src;
            want_wei_md.extra.asymm_compensation_mask = (1 << 1);
        }

        if (any_B_layout) {
            weights_md = want_wei_md;
            return status::success;
        }

        return weights_md == want_wei_md ? status::success
                                         : status::unimplemented;
    };

    auto set_or_check_tags = [&]() -> status_t {
        if (src_d.format_kind() == format_kind::any) {
            const format_tag_t desired_src_tag = plain_tensor_layout_tag;
            CHECK(memory_desc_init_by_tag(src_md, desired_src_tag));
            bgmmc.src_tag = desired_src_tag;
        } else {
            bgmmc.src_tag = is_amx_bf16 ? memory_desc_matches_one_of_tag(src_md,
                                    plain_tensor_layout_tag,
                                    transposed_tensor_layout_tag)
                                        : memory_desc_matches_one_of_tag(src_md,
                                                plain_tensor_layout_tag);
        }

        if (dst_d.format_kind() == format_kind::any) {
            const format_tag_t desired_dst_tag = plain_tensor_layout_tag;
            CHECK(memory_desc_init_by_tag(dst_md, desired_dst_tag));
            bgmmc.dst_tag = desired_dst_tag;
        } else {
            bgmmc.dst_tag = memory_desc_matches_one_of_tag(
                    dst_md, plain_tensor_layout_tag);
        }

        if (one_of(format_tag::undef, bgmmc.src_tag, bgmmc.dst_tag))
            return status::unimplemented;

        if (bgmmc.with_bias && bias_md.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, plain_tensor_layout_tag));

        return set_or_check_B_tag(true);
    };

    CHECK(set_or_check_tags());
    CHECK(attr.set_default_formats(&dst_md));

    // Configure matrix sizes
    const dim_t max_M = 64, min_M = 32;
    bgmmc.M_blk = 1;
    for (dim_t m_ = max_M; m_ >= min_M; m_--) {
        if (bgmmc.M % m_ == 0) {
            bgmmc.M_blk = m_;
            break;
        }
    }
    if (bgmmc.M_blk == 1) bgmmc.M_blk = nstl::min(bgmmc.M, max_M);

    bgmmc.N_blk = nstl::min(
            static_cast<dim_t>(bgmmc.wei_n_blk == 1 ? 64 : bgmmc.wei_n_blk),
            bgmmc.N);
    bgmmc.M_chunk_size = bgmmc.N_chunk_size = 1;

    // AMX BRGEMM kernel requires (K_brgemm % 64 == 0 || K_brgemm < 64) for
    // for K_brgemm reduction value to avoid AMX tiles re-configuration.
    // To satisfy this condition K_tail value is fixed to K % wei_k_blk here.
    const bool fixed_K_tail_size = bgmmc.is_amx && bgmmc.K % bgmmc.wei_k_blk > 0
            && bgmmc.K > bgmmc.wei_k_blk;
    bgmmc.K_blk = IMPLICATION(bgmmc.is_amx, bgmmc.K < bgmmc.wei_k_blk)
            ? rnd_up(bgmmc.K, bgmmc.required_k_granularity)
            : fixed_K_tail_size ? bgmmc.wei_k_blk : bgmmc.K;
    bgmmc.brgemm_batch_size
            = nstl::max(bgmmc.K / bgmmc.K_blk, static_cast<dim_t>(1));

    bgmmc.transposed_A = bgmmc.src_tag == transposed_tensor_layout_tag;
    const bool lda_is_big_2pow = is_bf16 && !bgmmc.transposed_A
            && math::is_pow2(bgmmc.K) && bgmmc.K >= 4096 && bgmmc.M >= 1024;
    const bool is_copy_a_required = bgmmc.K % bgmmc.required_k_granularity != 0
            || bgmmc.wei_zp_type != brgemm_broadcast_t::none
            || bgmmc.transposed_A || lda_is_big_2pow;
    bgmmc.use_buffer_a = is_copy_a_required;
    // Supported computation with copy only part of A related to K_tail if
    // is_copy_a_required == true, but the current performance measurements
    // show worse performance for it in comparison with copy whole A approach
    // (especially for big K sizes).
    bgmmc.use_buffer_a_tail_only = false;

    const int min_k_per_thread = 1024;
    const int max_k_parallel_work
            = div_up(static_cast<int>(bgmmc.K), min_k_per_thread);
    const int max_nthr_k = is_amx_bf16 && bgmmc.batch == 1
            ? nstl::min(saturate(1, 7, bgmmc.nthr / 8), max_k_parallel_work)
            : 1;
    int iter = 0;
    matmul_blocking_params_t best_blocking(bgmmc);
    matmul_blocking_params_t current_blocking(bgmmc);
    for (int nthr_k = 1; nthr_k <= max_nthr_k; nthr_k++) {
        int num_M_blk = div_up(bgmmc.M, bgmmc.M_blk);
        int num_N_blk = div_up(bgmmc.N, bgmmc.N_blk);
        int k_parallel_work = nstl::min(max_k_parallel_work, nthr_k);
        int num_parallel_work
                = bgmmc.batch * num_M_blk * num_N_blk * k_parallel_work;
        const bool a_lot_of_parallel_work = num_parallel_work > 8 * bgmmc.nthr;
        const bool a_lot_of_parallel_work_lvl2
                = num_parallel_work > 16 * bgmmc.nthr;
        const bool low_parallelism
                = static_cast<float>(num_parallel_work) < 1.5f * bgmmc.nthr;
        const int min_M_blk = low_parallelism && bgmmc.M_blk > 32
                ? div_up(bgmmc.M_blk, 2)
                : bgmmc.M_blk;
        const int min_N_blk = low_parallelism && is_amx_bf16 && !n_blk_fixed
                        && bgmmc.N_blk > 32
                ? 32
                : bgmmc.N_blk;
        const int desired_M_chunk = nstl::min(
                (bgmmc.use_buffer_b || a_lot_of_parallel_work ? 4 : 1),
                num_M_blk);
        const int desired_N_chunk = nstl::min(a_lot_of_parallel_work_lvl2
                        ? 6
                        : (bgmmc.use_buffer_a || a_lot_of_parallel_work ? 4
                                                                        : 1),
                num_N_blk);

        for_(int n_blk = bgmmc.N_blk; n_blk >= min_N_blk; n_blk -= 16)
        for_(int m_blk = bgmmc.M_blk; m_blk >= min_M_blk;
                m_blk = m_blk > 1 ? div_up(m_blk, 2) : m_blk - 1)
        for_(int n_ch_sz = desired_N_chunk; n_ch_sz >= 1; n_ch_sz--)
        for (int m_ch_sz = desired_M_chunk; m_ch_sz >= 1; m_ch_sz--, iter++) {
            current_blocking.set_blocking_parameters(
                    nthr_k, n_blk, n_ch_sz, m_blk, m_ch_sz);
            if (current_blocking.get_blocking_scores()
                    > best_blocking.get_blocking_scores())
                best_blocking = current_blocking;
        }
    }

    if (best_blocking.get_blocking_scores() == 0.0f)
        return status::unimplemented;
    best_blocking.update_configuration(bgmmc);
    if (bgmmc.wei_n_blk > bgmmc.N_blk && bgmmc.N >= bgmmc.wei_n_blk) {
        bgmmc.wei_n_blk = bgmmc.N_blk;
        CHECK(set_or_check_B_tag(false, bgmmc.wei_n_blk));
    }

    bgmmc.M_tail = bgmmc.M % bgmmc.M_blk;
    bgmmc.N_tail = bgmmc.N % bgmmc.N_blk;
    bgmmc.K_tail = bgmmc.K > bgmmc.K_blk
            ? rnd_up(bgmmc.K % bgmmc.K_blk, bgmmc.required_k_granularity)
            : 0;

    bgmmc.LDB = bgmmc.wei_n_blk;
    bgmmc.LDD = bgmmc.N;
    bgmmc.LDC
            = bgmmc.use_buffer_c && bgmmc.nthr_k <= 1 ? bgmmc.N_blk : bgmmc.LDD;

    init_aux_values(bgmmc, src_d, weights_d, dst_d);

    return status::success;
}

void init_aux_values(brgemm_matmul_conf_t &bgmmc,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &wei_d,
        const memory_desc_wrapper &dst_d) {
    bgmmc.wsp_tile_per_thr_bytes = 1024;

    bgmmc.M_chunk_elems = bgmmc.M_blk * bgmmc.M_chunk_size;
    bgmmc.N_chunk_elems = bgmmc.N_blk * bgmmc.N_chunk_size;
    bgmmc.K_chunk_elems = bgmmc.K_blk * bgmmc.brgemm_batch_size;
    bgmmc.M_chunks = div_up(bgmmc.M, bgmmc.M_chunk_elems);
    bgmmc.N_chunks = div_up(bgmmc.N, bgmmc.N_chunk_elems);
    bgmmc.K_chunks = div_up(bgmmc.K, bgmmc.K_chunk_elems);
    bgmmc.num_M_blocks = div_up(bgmmc.M, bgmmc.M_blk);
    bgmmc.num_N_blocks = div_up(bgmmc.N, bgmmc.N_blk);

    bgmmc.buffer_c_chunk_sz = bgmmc.acc_dt_sz * bgmmc.LDC
            * (bgmmc.nthr_k > 1 ? bgmmc.M : bgmmc.M_blk);
    bgmmc.buffer_c_per_thread_sz = bgmmc.buffer_c_chunk_sz
            * (bgmmc.nthr_k > 1 ? 1 : bgmmc.M_chunk_size * bgmmc.N_chunk_size);

    bgmmc.buffer_a_chunk_sz = bgmmc.a_dt_sz * bgmmc.M_blk
            * (bgmmc.use_buffer_a_tail_only ? bgmmc.wei_k_blk : bgmmc.LDA);
    bgmmc.buffer_a_chunk_shift_along_m = bgmmc.buffer_a_chunk_sz
            * (bgmmc.use_buffer_a_tail_only ? 1 : bgmmc.brgemm_batch_size);
    bgmmc.buffer_a_per_thread_sz
            = bgmmc.buffer_a_chunk_shift_along_m * bgmmc.M_chunk_size;

    bgmmc.buffer_b_chunk_sz
            = bgmmc.b_dt_sz * bgmmc.LDB * rnd_up(bgmmc.K_blk, bgmmc.wei_k_blk);
    bgmmc.buffer_b_per_thread_sz
            = bgmmc.buffer_b_chunk_sz * bgmmc.brgemm_batch_size;

    bgmmc.s8s8_comp_ithr_str
            = bgmmc.use_buffer_b ? bgmmc.wei_n_blk * bgmmc.N_chunk_size : 0;
    bgmmc.s8s8_comp_b_str = bgmmc.use_buffer_b
            ? 0
            : div_up(bgmmc.N, bgmmc.wei_n_blk) * bgmmc.wei_n_blk;
    bgmmc.s8s8_comp_n_str = bgmmc.wei_n_blk;

    const int dmax = nstl::min(bgmmc.ndims, 3);
    for (int d = 0; d < dmax; d++) {
        int dim = bgmmc.ndims - 1 - d;
        bgmmc.A_strides[d] = bgmmc.a_dt_sz * src_d.blocking_desc().strides[dim];
        bgmmc.B_strides[d] = bgmmc.b_dt_sz * wei_d.blocking_desc().strides[dim];
        bgmmc.C_strides[d] = bgmmc.c_dt_sz * dst_d.blocking_desc().strides[dim];
    }

    bgmmc.has_zero_point_a = bgmmc.src_zp_type != brgemm_broadcast_t::none;
    bgmmc.has_zero_point_b = bgmmc.wei_zp_type != brgemm_broadcast_t::none;
    bgmmc.has_zero_point_c = bgmmc.dst_zp_type != brgemm_broadcast_t::none;
    bgmmc.post_ops_applicable = one_of(true, bgmmc.with_sum, bgmmc.with_bias,
            bgmmc.with_scales, bgmmc.with_eltwise, bgmmc.with_binary,
            bgmmc.acc_dt != bgmmc.dst_dt, bgmmc.s8s8_compensation_required,
            bgmmc.has_zero_point_a, bgmmc.has_zero_point_b,
            bgmmc.has_zero_point_c);

    bgmmc.zp_a_comp_shift_n = bgmmc.wei_n_blk;
    bgmmc.zp_a_comp_elems_per_thr
            = bgmmc.N_chunk_size * bgmmc.zp_a_comp_shift_n;

    const int s32_elems_in_cacheline = 16;
    bgmmc.zp_b_comp_result_shift_m = bgmmc.M_blk;
    bgmmc.zp_b_comp_buffer_start
            = bgmmc.M_chunk_size * bgmmc.zp_b_comp_result_shift_m;
    bgmmc.zp_b_comp_buffer_shift_m = s32_elems_in_cacheline * bgmmc.M_blk;
    bgmmc.zp_b_comp_elems_per_thr = bgmmc.M_chunk_size
            * (bgmmc.zp_b_comp_result_shift_m + bgmmc.zp_b_comp_buffer_shift_m);

    bgmmc.brgemm_batch_element_per_thr_sz = 16 * bgmmc.brgemm_batch_size;
}

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const brgemm_matmul_conf_t &bgmmc) {
    const size_t default_data_align = sizeof(char);
    if (bgmmc.brg_type == brgemm_addr)
        scratchpad.book(key_brgemm_primitive_batch,
                bgmmc.nthr * bgmmc.brgemm_batch_element_per_thr_sz,
                sizeof(brgemm_batch_element_t), 64);

    if (bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only)
        scratchpad.book(key_brgemm_primitive_buffer_a,
                bgmmc.nthr * bgmmc.buffer_a_per_thread_sz, default_data_align);

    if (bgmmc.use_buffer_b) {
        scratchpad.book(key_brgemm_primitive_buffer_b,
                bgmmc.nthr * bgmmc.buffer_b_per_thread_sz, default_data_align);

        if (bgmmc.s8s8_compensation_required)
            scratchpad.book(key_brgemm_primitive_buffer_comp,
                    bgmmc.nthr * bgmmc.s8s8_comp_ithr_str,
                    types::data_type_size(f32));
    }

    if (bgmmc.use_buffer_c)
        scratchpad.book(key_brgemm_primitive_buffer,
                bgmmc.nthr * bgmmc.buffer_c_per_thread_sz, default_data_align);

    if (bgmmc.has_zero_point_a) {
        int num_elems = bgmmc.blocked_B
                ? bgmmc.num_N_blocks * bgmmc.zp_a_comp_shift_n
                : bgmmc.nthr * bgmmc.zp_a_comp_elems_per_thr;
        scratchpad.book(key_brgemm_primitive_zp_comp_a, num_elems,
                types::data_type_size(s32));
    }

    if (bgmmc.has_zero_point_b)
        scratchpad.book(key_brgemm_primitive_zp_comp_b,
                bgmmc.nthr * bgmmc.zp_b_comp_elems_per_thr,
                types::data_type_size(s32));

    if (one_of(bgmmc.isa, avx512_core_bf16_amx_int8, avx512_core_bf16_amx_bf16))
        scratchpad.book(key_conv_amx_tile_buffer,
                bgmmc.nthr * bgmmc.wsp_tile_per_thr_bytes, default_data_align);
}

void matmul_blocking_params_t::init_zero() {
    nthr_k_ = nthr_mnb_ = nthr_ = 0;
    n_blk_ = n_chunk_size_ = n_chunk_elems_ = 0;
    m_blk_ = m_chunk_size_ = m_chunk_elems_ = 0;
    k_blk_ = k_chunk_size_ = k_chunk_elems_ = 0;
    need_buf_c_ = false;
    current_lda_ = 0;
    efficiency_score_ = 0.0f;
}

void matmul_blocking_params_t::init_from_conf() {
    nthr_k_ = nstl::max(nthr_k, 1);
    nthr_mnb_ = nthr / nthr_k_;
    nthr_ = nthr_mnb_ * nthr_k_;
    n_blk_ = N_blk;
    n_chunk_size_ = N_chunk_size;
    m_blk_ = M_blk;
    m_chunk_size_ = M_chunk_size;
    k_blk_ = K_blk;
    k_chunk_size_ = brgemm_batch_size;
    n_chunk_elems_ = n_blk_ * n_chunk_size_;
    m_chunk_elems_ = m_blk_ * m_chunk_size_;
    k_chunk_elems_ = k_blk_ * k_chunk_size_;
    need_buf_c_ = use_buffer_c;
    current_lda_ = LDA;
    efficiency_score_ = 0.0f;
}

void matmul_blocking_params_t::update_k_blocking_dependent_params() {
    k_chunk_elems_ = k_blk_ * k_chunk_size_;
    current_lda_ = get_actual_lda();
    need_buf_c_ = is_buffer_c_required();
}

void matmul_blocking_params_t::set_blocking_parameters(
        int nthr_k, int n_blk, int n_chunk_size, int m_blk, int m_chunk_size) {
    nthr_k_ = nstl::max(1, nthr_k);
    nthr_mnb_ = nthr / nthr_k_;
    nthr_ = nthr_mnb_ * nthr_k_;
    n_blk_ = n_blk;
    n_chunk_size_ = n_chunk_size;
    m_blk_ = m_blk;
    m_chunk_size_ = m_chunk_size;
    if (one_of(0, n_blk_, n_chunk_size_, m_blk_, m_chunk_size_)) {
        k_blk_ = k_chunk_size_ = k_chunk_elems_ = 0;
        efficiency_score_ = 0.0f;
        return;
    }

    n_chunk_elems_ = n_blk_ * n_chunk_size_;
    m_chunk_elems_ = m_blk_ * m_chunk_size_;

    if (K < wei_k_blk) {
        k_blk_ = is_amx ? rnd_up(K, required_k_granularity) : K;
        k_chunk_size_ = 1;
    } else {
        dim_t k_per_thr = div_up(K, nthr_k_);
        k_blk_ = nstl::min(
                is_amx ? rnd_up(k_per_thr, required_k_granularity) : k_per_thr,
                static_cast<dim_t>(wei_k_blk));
        k_chunk_size_ = nstl::min(nstl::max(static_cast<dim_t>(1), K / k_blk_),
                div_up(k_per_thr, k_blk_));

        update_k_blocking_dependent_params();
        auto chunk_sz = calculate_chunk_memory_size();
        float k_div = (float)chunk_sz / L2_threshold;
        if (k_div > 1.0f)
            k_chunk_size_ = static_cast<int>(
                    static_cast<float>(k_chunk_size_) / k_div + 0.6f);

        const dim_t current_k_tail = K % k_blk_;
        if (current_k_tail == 0 && K % (k_blk_ * k_chunk_size_) == 0) {
            k_blk_ *= k_chunk_size_;
            k_chunk_size_ = 1;
        } else if (nthr_k_ == 1
                && K == k_blk_ * k_chunk_size_ + current_k_tail) {
            k_blk_ *= k_chunk_size_;
            k_chunk_size_ = 2;
        }
    }

    update_k_blocking_dependent_params();

    blocking_chunk_mem_size_ = calculate_chunk_memory_size();

    efficiency_score_ = calculate_blocking_scores();
}

// returns score for current blocking parameters' values in range [0, 1]
// for parallel work over threads distribution score. Maximum scores - when
// all threads have the same work amount w/o tails
float matmul_blocking_params_t::get_thread_balance_scores() {
    dim_t num_M_chunks = div_up(M, m_chunk_elems_);
    dim_t num_N_chunks = div_up(N, n_chunk_elems_);
    float mnb_parallel_score = batch * ((float)M / m_chunk_elems_)
            * ((float)N / n_chunk_elems_)
            / rnd_up(batch * num_M_chunks * num_N_chunks, nthr_mnb_)
            * nthr_mnb_;
    float k_parallel_score = 1.0f;
    if (nthr_k_ > 1) {
        dim_t num_K_chunks = div_up(K, k_chunk_elems_);
        const float parallel_reduction_penalty = 0.8f;
        k_parallel_score = parallel_reduction_penalty
                * ((float)K / k_chunk_elems_) / rnd_up(num_K_chunks, nthr_k_)
                * nthr_k_;
    }

    return mnb_parallel_score * k_parallel_score / nthr;
}

// returns score for current blocking parameters' values in range [0, 1]
// for copied data reusage
float matmul_blocking_params_t::get_copied_data_reusage_scores() {
    const int desired_M_chunk = use_buffer_b
            ? nstl::min(4, rnd_up(static_cast<int>(M), m_blk_))
            : 1;
    const int desired_N_chunk = use_buffer_a
            ? nstl::min(4, rnd_up(static_cast<int>(N), n_blk_))
            : 1;

    return 0.5f
            * (nstl::min((float)m_chunk_size_ / desired_M_chunk, 1.0f)
                    + nstl::min((float)n_chunk_size_ / desired_N_chunk, 1.0f));
}

// returns score for current blocking parameters' values in range [0, 1]
// for L2 utilization
float matmul_blocking_params_t::get_L2_utilization_scores() {
    const float relative_difference_with_L2
            = fabsf((float)L2_threshold - blocking_chunk_mem_size_)
            / nstl::max(L2_threshold, blocking_chunk_mem_size_);
    return 1.0f - relative_difference_with_L2;
}

// returns score for current blocking parameters' values in range [0, 1]
// consists of 3 parts with its own weights:
// 	1) parallel work over threads distribution score
// 	2) L2 utilization score
// 	3) copied data re-usage score
float matmul_blocking_params_t::calculate_blocking_scores() {
    if (one_of(0, n_blk_, n_chunk_size_, m_blk_, m_chunk_size_, k_blk_,
                k_chunk_size_))
        return 0.0f;

    const float nthr_coeff = nstl::min(nthr, 100);
    const float reusage_factor = 1.0f;
    const float balance_factor = (nthr_coeff - 1.0f) / nthr_coeff;
    const float cache_utilization_factor = 1.0f / nthr_coeff;

    float scores = cache_utilization_factor * get_L2_utilization_scores()
            + reusage_factor * get_copied_data_reusage_scores();
    if (balance_factor > 0.0f)
        scores += balance_factor * get_thread_balance_scores();
    return scores
            / (reusage_factor + balance_factor + cache_utilization_factor);
}

void matmul_blocking_params_t::update_configuration(
        brgemm_matmul_conf_t &bgmmc) {
    bgmmc.nthr_k = nthr_k_;
    bgmmc.M_blk = m_blk_;
    bgmmc.M_chunk_size = m_chunk_size_;
    bgmmc.N_blk = n_blk_;
    bgmmc.N_chunk_size = n_chunk_size_;

    bgmmc.K_blk = k_blk_;
    bgmmc.brgemm_batch_size = k_chunk_size_;

    bgmmc.use_buffer_c = need_buf_c_;
    bgmmc.LDA = current_lda_;
}

dim_t matmul_blocking_params_t::get_actual_lda() {
    if (!use_buffer_a) return K;

    constexpr int bytes_in_cacheline = 64;
    const int elems_in_cacheline = bytes_in_cacheline / a_dt_sz;
    dim_t lda = rnd_up(k_blk_, elems_in_cacheline);
    const bool is_big_2_pow = lda >= 512 && math::is_pow2(lda);
    if (is_big_2_pow) lda += elems_in_cacheline;
    return lda;
}

bool matmul_blocking_params_t::is_buffer_c_required() {
    if (nthr_k_ > 1 && K > k_chunk_elems_) return true;

    return ((acc_dt != dst_dt || with_sum)
            && (K > k_chunk_elems_ || K % k_blk_ > 0));
}

size_t matmul_blocking_params_t::calculate_chunk_memory_size() {
    size_t A_chunk_sz = a_dt_sz * k_chunk_elems_ * m_chunk_elems_;
    size_t A_buf_sz = use_buffer_a
            ? a_dt_sz * current_lda_ * k_chunk_size_ * m_chunk_elems_
            : 0;
    size_t B_chunk_sz = b_dt_sz * k_chunk_elems_ * n_chunk_elems_;
    size_t B_buf_sz = use_buffer_b ? b_dt_sz * n_blk_ * k_chunk_elems_ : 0;
    size_t C_chunk_sz = c_dt_sz * m_chunk_elems_ * n_chunk_elems_;
    size_t C_buf_sz
            = need_buf_c_ ? acc_dt_sz * m_chunk_elems_ * n_chunk_elems_ : 0;
    return A_chunk_sz + A_buf_sz + B_chunk_sz + B_buf_sz + C_chunk_sz
            + C_buf_sz;
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
