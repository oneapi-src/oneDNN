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

    return injector::post_ops_ok(post_ops_ok_args_t(avx512_common,
            {sum, eltwise, binary}, post_ops, &dst_d,
            false /*sum_at_pos_0_only*/, false /*sum_requires_scale_one*/,
            {broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::scalar}));
}

status_t init_brgemm_matmul_conf(cpu_isa_t isa, brgemm_matmul_conf_t &bgmmc,
        const matmul_desc_t &mmd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr) {
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
    bgmmc.signed_input = isa == avx512_core_vnni && bgmmc.src_dt == s8;

    const bool is_int8 = one_of(bgmmc.src_dt, u8, s8) && bgmmc.wei_dt == s8
            && one_of(bgmmc.dst_dt, u8, s8, s32, f32);
    const bool is_amx = isa == avx512_core_bf16_amx_int8;

    if (is_int8) {
        bgmmc.acc_dt = s32;
        bgmmc.with_scales = true;
    }

    if (bgmmc.with_scales) {
        const auto &oscales = attr.output_scales_;
        bgmmc.is_oscale_per_n = oscales.mask_ == 1 << 1;

        // only common and per-oc-channel scales are supported
        const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
        if (!oscales_ok) return status::unimplemented;
    }

    const auto &p = attr.post_ops_;
    bgmmc.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    bgmmc.with_eltwise = eltwise_ind != -1;
    const int binary_ind = p.find(primitive_kind::binary);
    bgmmc.with_binary = binary_ind != -1;

    if (!post_ops_ok(bgmmc, attr, dst_d)) return status::unimplemented;

    if (IMPLICATION(is_int8,
                !one_of(isa, avx512_core_vnni, avx512_core_bf16_amx_int8)))
        return status::unimplemented;
    matmul_helper_t helper(src_d, weights_d, dst_d);
    bgmmc.ndims = dst_d.ndims();

    if (bgmmc.ndims > 3) return status::unimplemented;

    bgmmc.batch_ndims = bgmmc.ndims - 2;
    bgmmc.M = helper.M();
    bgmmc.N = helper.N();
    bgmmc.K = helper.K();
    bgmmc.batch = helper.batch();

    // required granularity for k dimension
    const int k_gran = is_amx ? 4 : 1;
    const int k_blk_gran = is_amx ? 64 : 4;

    auto set_or_check_tags = [&]() -> status_t {
        format_tag_t desired_src_tag = pick(bgmmc.ndims - 2, ab, abc);
        format_tag_t desired_dst_tag = desired_src_tag;

        if (src_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(src_md, desired_src_tag));
            bgmmc.src_tag = desired_src_tag;
        } else {
            bgmmc.src_tag
                    = memory_desc_matches_one_of_tag(src_md, desired_src_tag);
        }

        if (dst_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(dst_md, desired_dst_tag));
            bgmmc.dst_tag = desired_dst_tag;
        } else {
            bgmmc.dst_tag
                    = memory_desc_matches_one_of_tag(dst_md, desired_dst_tag);
        }

        format_tag_t desired_wei_tag = desired_src_tag;
        if (weights_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(weights_md, desired_wei_tag));
            bgmmc.wei_tag = desired_wei_tag;
        } else {
            format_tag_t transposed_wei_tag = pick(bgmmc.ndims - 2, ba, acb);
            bgmmc.wei_tag = memory_desc_matches_one_of_tag(
                    weights_md, desired_wei_tag, transposed_wei_tag);
        }
        bgmmc.wei_k_blk = k_blk_gran;
        bgmmc.wei_n_blk = 64;
        bgmmc.use_buffer_b = true;

        if (one_of(format_tag::undef, bgmmc.src_tag, bgmmc.dst_tag,
                    bgmmc.wei_tag))
            return status::unimplemented;

        if (bgmmc.with_bias && bias_md.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));

        return status::success;
    };

    CHECK(set_or_check_tags());

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
        if (bgmmc.use_buffer)
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

    const auto L2_treshold = 3 * platform::get_per_core_cache_size(2) / 4;
    const bool is_copy_a_required = bgmmc.K % k_gran != 0;
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
        bgmmc.use_buffer
                = bgmmc.acc_dt != bgmmc.dst_dt && bgmmc.K > bgmmc.K_blk;
        bgmmc.LDA = get_actual_LDA();

        int num_M_blk = div_up(bgmmc.M, bgmmc.M_blk);
        int num_N_blk = div_up(bgmmc.N, bgmmc.N_blk);
        if (4 * bgmmc.nthr < bgmmc.batch) {
            bgmmc.M_chunk_size = num_M_blk;
            bgmmc.N_chunk_size = num_N_blk;
        } else {
            int need_M_work_per_thr
                    = div_up(bgmmc.nthr, num_N_blk * bgmmc.batch);
            int max_num_M_blk = div_up(num_M_blk, need_M_work_per_thr);
            if (max_num_M_blk > 1)
                bgmmc.M_chunk_size = nstl::min(max_num_M_blk, 4);
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

    bgmmc.M_tail = bgmmc.M % bgmmc.M_blk;
    bgmmc.N_tail = bgmmc.N % bgmmc.N_blk;
    bgmmc.K_tail
            = bgmmc.K > bgmmc.K_blk ? rnd_up(bgmmc.K % bgmmc.K_blk, k_gran) : 0;

    bgmmc.LDB = bgmmc.wei_n_blk;
    bgmmc.LDD = bgmmc.N;
    bgmmc.LDC = bgmmc.use_buffer ? bgmmc.N_blk : bgmmc.LDD;

    return status::success;
}

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const brgemm_matmul_conf_t &bgmmc) {
    size_t sc_size = sizeof(brgemm_batch_element_t);
    size_t n_elems = bgmmc.nthr * 16 * bgmmc.brgemm_batch_size;
    if (bgmmc.brg_type == brgemm_addr) {
        scratchpad.book(key_brgemm_primitive_batch, n_elems, sc_size, 64);
    }
    if (bgmmc.use_buffer) {
        size_t nelements = (size_t)bgmmc.nthr * bgmmc.LDC * bgmmc.M_blk
                * bgmmc.M_chunk_size * bgmmc.N_chunk_size;
        scratchpad.book(key_brgemm_primitive_buffer, nelements,
                types::data_type_size(bgmmc.acc_dt));
    }

    if (bgmmc.use_buffer_b) {
        size_t nelements = (size_t)bgmmc.nthr * bgmmc.LDB
                * rnd_up(bgmmc.K_blk, bgmmc.wei_k_blk)
                * bgmmc.brgemm_batch_size;
        scratchpad.book(key_brgemm_primitive_buffer_b, nelements,
                types::data_type_size(bgmmc.wei_dt));
        if (bgmmc.signed_input) {
            size_t nelements_comp
                    = (size_t)bgmmc.nthr * bgmmc.wei_n_blk * bgmmc.N_chunk_size;
            scratchpad.book(key_brgemm_primitive_buffer_comp, nelements_comp,
                    types::data_type_size(bgmmc.acc_dt));
        }
    }

    if (bgmmc.use_buffer_a) {
        size_t nelements = (size_t)bgmmc.nthr * bgmmc.LDA
                * bgmmc.brgemm_batch_size * bgmmc.M_blk * bgmmc.M_chunk_size;
        scratchpad.book(key_brgemm_primitive_buffer_a, nelements,
                types::data_type_size(bgmmc.src_dt));
    } else if (bgmmc.use_buffer_a_tail_only) {
        size_t nelements = (size_t)bgmmc.nthr * bgmmc.wei_k_blk * bgmmc.M_blk
                * bgmmc.M_chunk_size;
        scratchpad.book(key_brgemm_primitive_buffer_a, nelements,
                types::data_type_size(bgmmc.src_dt));
    }

    if (bgmmc.isa == avx512_core_bf16_amx_int8)
        scratchpad.book(
                key_conv_amx_tile_buffer, bgmmc.nthr * 1024, sizeof(char));
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
