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
            && one_of(bgmmc.dst_dt, u8, s8, s32, f32);
    const bool is_bf16 = everyone_is(bf16, bgmmc.src_dt, bgmmc.wei_dt)
            && one_of(bgmmc.dst_dt, bf16, f32);
    const bool is_amx_int8 = isa == avx512_core_bf16_amx_int8;
    const bool is_amx_bf16 = isa == avx512_core_bf16_amx_bf16;
    const bool is_amx = is_amx_int8 || is_amx_bf16;

    bgmmc.acc_dt = is_int8 ? s32 : f32;

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
    const int k_gran = is_amx_int8 ? 4 : (is_amx_bf16 ? 2 : 1);
    const int k_blk_gran = is_amx_int8 ? 64 : (is_amx_bf16 ? 32 : 4);

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

        bgmmc.wei_k_blk = k_blk_gran;
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

    if (is_amx_bf16 && !n_blk_fixed) {
        // reduce N block size for bf16 problems if number of parallel work
        // is small
        const auto num_parallel_work = bgmmc.batch
                * div_up(bgmmc.M, bgmmc.M_blk)
                * div_up(bgmmc.N, bgmmc.wei_n_blk);
        if ((float)num_parallel_work < 1.5f * bgmmc.nthr) {
            CHECK(set_or_check_B_tag(false, bgmmc.wei_n_blk / 2));
        }
    }

    bgmmc.N_blk = nstl::min(
            (dim_t)(bgmmc.wei_n_blk == 1 ? 64 : bgmmc.wei_n_blk), bgmmc.N);
    bgmmc.M_chunk_size = bgmmc.N_chunk_size = 1;

    // AMX BRGEMM kernel requires (K_brgemm % 64 == 0 || K_brgemm < 64) for
    // for K_brgemm reduction value to avoid AMX tiles re-configuration.
    // To satisfy this condition K_tail value is fixed to K % wei_k_blk here.
    const bool fixed_K_tail_size = is_amx && bgmmc.K % bgmmc.wei_k_blk > 0
            && bgmmc.K > bgmmc.wei_k_blk;
    bgmmc.K_blk = IMPLICATION(is_amx, bgmmc.K < bgmmc.wei_k_blk)
            ? rnd_up(bgmmc.K, k_gran)
            : fixed_K_tail_size ? bgmmc.wei_k_blk : bgmmc.K;
    bgmmc.brgemm_batch_size = nstl::max(bgmmc.K / bgmmc.K_blk, (dim_t)1);

    auto get_chunk_size = [&]() -> size_t {
        size_t A_chunk_sz = types::data_type_size(bgmmc.src_dt) * bgmmc.K_blk
                * bgmmc.brgemm_batch_size * bgmmc.M_blk * bgmmc.M_chunk_size;
        size_t A_buf_sz = 0;
        if (bgmmc.use_buffer_a)
            A_buf_sz = types::data_type_size(bgmmc.src_dt) * bgmmc.LDA
                    * bgmmc.brgemm_batch_size * bgmmc.M_blk
                    * bgmmc.M_chunk_size;
        size_t B_chunk_sz = types::data_type_size(bgmmc.wei_dt) * bgmmc.K_blk
                * bgmmc.brgemm_batch_size * bgmmc.N_blk * bgmmc.N_chunk_size;
        size_t B_buf_sz = 0;
        if (bgmmc.use_buffer_b)
            B_buf_sz = types::data_type_size(bgmmc.wei_dt) * bgmmc.wei_n_blk
                    * bgmmc.K_blk * bgmmc.brgemm_batch_size;
        size_t C_chunk_sz = types::data_type_size(bgmmc.dst_dt) * bgmmc.M_blk
                * bgmmc.M_chunk_size * bgmmc.N_blk * bgmmc.N_chunk_size;
        size_t C_buf_sz = 0;
        if (bgmmc.use_buffer_c)
            C_buf_sz = types::data_type_size(bgmmc.acc_dt) * bgmmc.M_blk
                    * bgmmc.M_chunk_size * bgmmc.N_blk * bgmmc.N_chunk_size;
        return A_chunk_sz + A_buf_sz + B_chunk_sz + B_buf_sz + C_chunk_sz
                + C_buf_sz;
    };

    auto get_actual_LDA = [&]() -> dim_t {
        if (bgmmc.use_buffer_a) {
            constexpr int bytes_in_cacheline = 64;
            const int elems_in_cacheline
                    = bytes_in_cacheline / types::data_type_size(bgmmc.src_dt);
            dim_t lda = rnd_up(bgmmc.K_blk, elems_in_cacheline);
            const bool is_big_2_pow = lda >= 512 && (lda & (lda - 1)) == 0;
            if (is_big_2_pow) lda += elems_in_cacheline;
            return lda;
        }
        return bgmmc.K;
    };

    auto is_buffer_c_required = [&]() -> bool {
        return (bgmmc.acc_dt != bgmmc.dst_dt || bgmmc.with_sum)
                && (bgmmc.K > bgmmc.K_blk * bgmmc.brgemm_batch_size
                        || bgmmc.K % bgmmc.K_blk > 0);
    };

    bgmmc.transposed_A = bgmmc.src_tag == transposed_tensor_layout_tag;
    const auto L2_treshold = 3 * platform::get_per_core_cache_size(2) / 4;
    const bool is_copy_a_required = bgmmc.K % k_gran != 0
            || bgmmc.wei_zp_type != brgemm_broadcast_t::none
            || bgmmc.transposed_A;
    bgmmc.use_buffer_a = is_copy_a_required;
    // Supported computation with copy only part of A related to K_tail if
    // is_copy_a_required == true, but the current performance measurements
    // show worse performance for it in comparison with copy whole A approach
    // (especially for big K sizes).
    bgmmc.use_buffer_a_tail_only = false;
    int attempts = 3;
    // Try to improve blocking wrt L2 size
    // TODO: improve blocking algorithm
    while (attempts > 0) {
        bgmmc.use_buffer_c = is_buffer_c_required();
        bgmmc.LDA = get_actual_LDA();

        int num_M_blk = div_up(bgmmc.M, bgmmc.M_blk);
        int num_N_blk = div_up(bgmmc.N, bgmmc.N_blk);
        if (4 * bgmmc.nthr < bgmmc.batch) {
            bgmmc.M_chunk_size = num_M_blk;
            bgmmc.N_chunk_size = num_N_blk;
        } else {
            // bgmmc.N_chunk_size and bgmmc.M_chunk_size parameters allow to
            // reuse copied / transformed chunks for A and B matrixes.
            // It reduces overhead on copy routines.
            const bool a_lot_of_parallel_work
                    = bgmmc.batch * num_M_blk * num_N_blk > 16 * bgmmc.nthr;
            const int desired_M_chunk = nstl::min(
                    bgmmc.use_buffer_b || a_lot_of_parallel_work ? 4 : 1,
                    num_M_blk);
            const int desired_N_chunk
                    = nstl::min(bgmmc.transposed_A ? 4 : 1, num_N_blk);
            const int min_m_blk_threshold = 32;
            int M_chunk = desired_M_chunk;
            int N_chunk = desired_N_chunk;
            while (M_chunk >= 1 && N_chunk >= 1) {
                // Trying to find balance between M/N work distribution across
                // threads and reusage of copied parts of matrixes A & B.
                const int par_work_along_n = div_up(num_N_blk, N_chunk);
                const int par_work_along_m = div_up(num_M_blk, M_chunk);
                const int par_work_total
                        = bgmmc.batch * par_work_along_n * par_work_along_m;
                if ((float)par_work_total < 0.7f * bgmmc.nthr
                        && bgmmc.M_blk > min_m_blk_threshold) {
                    bgmmc.M_blk = div_up(bgmmc.M_blk, 2);
                    num_M_blk = div_up(bgmmc.M, bgmmc.M_blk);
                    continue;
                }
                if ((float)par_work_total > bgmmc.nthr
                        && (float)par_work_total < 1.2f * bgmmc.nthr
                        && M_chunk == desired_M_chunk
                        && N_chunk == desired_N_chunk) {
                    if (num_M_blk % M_chunk != 0 || desired_N_chunk == 1)
                        M_chunk++;
                    else
                        N_chunk++;
                    bgmmc.M_chunk_size = M_chunk;
                    bgmmc.N_chunk_size = N_chunk;
                    break;
                }
                if ((float)par_work_total >= 0.85f * bgmmc.nthr) {
                    bgmmc.M_chunk_size = M_chunk;
                    bgmmc.N_chunk_size = N_chunk;
                    break;
                }

                const int allowed_chunk_size_diff
                        = num_M_blk % M_chunk == 0 ? 1 : 0;
                const bool reduce_N_chunk = N_chunk > 1
                        && N_chunk + allowed_chunk_size_diff >= M_chunk;
                if (reduce_N_chunk)
                    N_chunk--;
                else
                    M_chunk--;
            }
        }

        auto chunk_sz = get_chunk_size();
        if ((float)chunk_sz <= 1.1f * L2_treshold) break;
        int k_div = div_up(chunk_sz, L2_treshold);

        if (fixed_K_tail_size) // try to ajust brgemm_batch_size, not K_blk
            bgmmc.brgemm_batch_size
                    = nstl::max(bgmmc.brgemm_batch_size / k_div, 1);
        else
            bgmmc.K_blk
                    = nstl::min(rnd_up(bgmmc.K_blk / k_div, bgmmc.wei_k_blk),
                            rnd_up(bgmmc.K, k_gran));
        attempts--;
    }

    // try to refine K_blk size
    if (fixed_K_tail_size) {
        // K_tail might be different from bgmmc.K_tail
        const dim_t K_tail = bgmmc.K % bgmmc.K_blk;
        const dim_t K_no_tail = bgmmc.K - K_tail;
        const dim_t K_chunk_size = bgmmc.brgemm_batch_size * bgmmc.K_blk;
        const bool the_same_num_blocks_in_all_chunks
                = K_no_tail == K_no_tail / K_chunk_size * K_chunk_size;
        // it's better to avoid using of too small K_blk values like wei_k_blk
        if (the_same_num_blocks_in_all_chunks) {
            bgmmc.K_blk = K_chunk_size;
            // do not separate K_tail calculation to another chunk
            const bool use_single_K_chunk = K_no_tail == K_chunk_size;
            bgmmc.brgemm_batch_size = use_single_K_chunk ? 2 : 1;
            bgmmc.LDA = get_actual_LDA();
        }
    }
    bgmmc.use_buffer_c = is_buffer_c_required();
    bgmmc.LDA = get_actual_LDA();

    bgmmc.M_tail = bgmmc.M % bgmmc.M_blk;
    bgmmc.N_tail = bgmmc.N % bgmmc.N_blk;
    bgmmc.K_tail
            = bgmmc.K > bgmmc.K_blk ? rnd_up(bgmmc.K % bgmmc.K_blk, k_gran) : 0;

    bgmmc.LDB = bgmmc.wei_n_blk;
    bgmmc.LDD = bgmmc.N;
    bgmmc.LDC = bgmmc.use_buffer_c ? bgmmc.N_blk : bgmmc.LDD;

    init_aux_values(bgmmc, src_d, weights_d, dst_d);

    return status::success;
}

void init_aux_values(brgemm_matmul_conf_t &bgmmc,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &wei_d,
        const memory_desc_wrapper &dst_d) {
    bgmmc.wsp_tile_per_thr_bytes = 1024;
    bgmmc.a_dt_sz = types::data_type_size(bgmmc.src_dt);
    bgmmc.b_dt_sz = types::data_type_size(bgmmc.wei_dt);
    bgmmc.c_dt_sz = types::data_type_size(bgmmc.dst_dt);
    bgmmc.acc_dt_sz = types::data_type_size(bgmmc.acc_dt);

    bgmmc.M_chunk_elems = bgmmc.M_blk * bgmmc.M_chunk_size;
    bgmmc.N_chunk_elems = bgmmc.N_blk * bgmmc.N_chunk_size;
    bgmmc.K_chunk_elems = bgmmc.K_blk * bgmmc.brgemm_batch_size;
    bgmmc.M_chunks = div_up(bgmmc.M, bgmmc.M_chunk_elems);
    bgmmc.N_chunks = div_up(bgmmc.N, bgmmc.N_chunk_elems);
    bgmmc.K_chunks = div_up(bgmmc.K, bgmmc.K_chunk_elems);
    bgmmc.num_M_blocks = div_up(bgmmc.M, bgmmc.M_blk);
    bgmmc.num_N_blocks = div_up(bgmmc.N, bgmmc.N_blk);

    if (bgmmc.with_bias) bgmmc.bias_dt_sz = types::data_type_size(bgmmc.bia_dt);

    bgmmc.buffer_c_chunk_sz = bgmmc.acc_dt_sz * bgmmc.LDC * bgmmc.M_blk;
    bgmmc.buffer_c_per_thread_sz
            = bgmmc.buffer_c_chunk_sz * bgmmc.M_chunk_size * bgmmc.N_chunk_size;

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

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
