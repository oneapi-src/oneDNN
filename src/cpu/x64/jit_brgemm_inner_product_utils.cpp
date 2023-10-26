/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "cpu/x64/jit_brgemm_inner_product_utils.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace prop_kind;
using namespace data_type;

namespace brgemm_inner_product_utils {

// Returns amount of work on a thread when using parallel reduction.
static int comp_work(
        int nthrs, int nthr_k, int n_chunks, int n_reduction_blocks) {
    assert(nthrs >= nthr_k);
    int nthr_other = nthrs / nthr_k;

    // Work in reduction dimension.
    int reduction_work = div_up(n_reduction_blocks, nthr_k);

    // Work in other non-reduction dimensions.
    int other_work = div_up(n_chunks, nthr_other);

    return reduction_work * other_work;
}

int get_brg_kernel_index(bool is_bs_tail, bool do_initialization,
        bool is_M_tail, bool is_N_tail, bool is_K_tail) {
    int idx = 16 * (int)is_bs_tail + 8 * (int)do_initialization
            + 4 * (int)is_M_tail + 2 * (int)is_N_tail + (int)is_K_tail;

    assert(idx < max_num_brg_kernels_ip);
    return idx;
}

int jit_brgemm_ip_conf_t::get_os_block(
        bool try_to_adjust, bool is_adjustment) const {
    const auto &jbgp = *this;

    const bool is_amx_int8 = jbgp.is_amx && one_of(jbgp.wei_dt, s8, u8);
    const bool is_xf16 = one_of(jbgp.wei_dt, bf16, f16) || jbgp.is_bf32;
    const bool is_amx_xf16 = jbgp.is_amx && is_xf16;
    const bool is_avx512_bf16 = jbgp.isa == avx512_core_bf16;
    const bool is_avx512 = jbgp.isa == avx512_core;
    const bool is_f32_compute = !jbgp.is_bf32
            && everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);

    int max_os_block = 0;
    int min_os_block = 0;

    if (try_to_adjust
            || one_of(jbgp.prop_kind, forward_training, forward_inference)) {
        min_os_block = (is_amx_int8 || is_amx_xf16) ? 16 : 6;
        // Currently gigantic flag is used to separate out transformer_lt and
        // alexnet shapes for which larger os_block gives better performance.
        // TODO: Figure out how much the constraints for `gigantic-ness` can
        // be further loosened.
        const bool is_gigantic_shape
                = jbgp.ic >= 9216 && jbgp.oc >= 4096 && jbgp.os >= 512;
        const bool use_128_block_for_amx
                = is_amx_xf16 && jbgp.os % 128 == 0 && jbgp.oc > 128;
        const bool enable_128_os_blocking
                = use_128_block_for_amx || is_gigantic_shape;
        max_os_block = enable_128_os_blocking ? 128 : 64;
        // Work done by each thread is given by:
        //     (nb_oc / nb_oc_blocking) * (nb_os / nb_os_blocking)
        // As a first approximation we take nb_oc_blocking = nb_os_blocking = 1
        // Furthermore, we recall that
        //     nb_oc = oc / oc_block
        //     nb_os = os / os_block
        //
        // For f32 data type our objective is to determine the optimal value
        // of os_block such that the work amount per thread ~ 2
        if (is_f32_compute && jbgp.nb_oc != 0) {
            const bool small_work_amt_per_thread
                    = div_up(jbgp.os, max_os_block) * jbgp.nb_oc
                    < 1.8f * jbgp.nthr;
            if (small_work_amt_per_thread)
                max_os_block = saturate(16, max_os_block,
                        div_up(jbgp.os * jbgp.nb_oc, 2 * jbgp.nthr));
        }
    } else if (jbgp.prop_kind == backward_data) {
        int plat_max_os_block = 0;
        if (is_amx_xf16) {
            plat_max_os_block
                    = (jbgp.ic >= 512 && jbgp.oc / jbgp.ic <= 4) ? 128 : 64;
        } else if (is_avx512_bf16) {
            plat_max_os_block = (jbgp.ic > 256) ? 128 : 64;
        } else {
            plat_max_os_block = 64;
        }
        max_os_block = nstl::min(plat_max_os_block, jbgp.os);
        min_os_block = is_amx_xf16 ? 16 : is_avx512 ? 6 : 4;
        if (jbgp.isa == avx2 && jbgp.prop_kind == backward_data
                && jbgp.os * jbgp.oc > 512 * 1024)
            return jbgp.os;

    } else if (jbgp.prop_kind == backward_weights) {
        constexpr int amx_xf16_row = 64;
        constexpr int amx_xf16_half_row = amx_xf16_row / 2;
        // ensure that os_tail <= amx_xf16_half_row
        const bool use_large_os_block = (jbgp.os >= amx_xf16_row)
                && (jbgp.os % amx_xf16_row) <= amx_xf16_half_row;
        return is_amx_xf16
                ? (use_large_os_block ? amx_xf16_row : amx_xf16_half_row)
                : jbgp.isa == avx2 ? // taken from gemm impl for avx2
                rnd_up(nstl::min(jbgp.os, 192), jbgp.simd_w)
                                   : 16;
    } else
        assert(!"unsupported case");

    if (is_adjustment) max_os_block = nstl::max(max_os_block / 2, 1);
    assert(min_os_block > 0 && max_os_block > 0);
    int os_block = max_div(jbgp.os, max_os_block);
    if (os_block < min_os_block) os_block = nstl::min(jbgp.os, max_os_block);

    // Use large os-block to reduce bandwidth requirement.
    if (jbgp.use_small_os_kernels) os_block = jbgp.os;

    return os_block;
}

std::unordered_map<int, format_tag_t>
jit_brgemm_ip_conf_t::get_desired_weights_tag() const {
    const auto &jbgp = *this;

    using namespace format_tag;
    const int n_sp_dims = jbgp.ndims - 2;
    const bool is_xf16 = utils::one_of(jbgp.wei_dt, bf16, f16);
    const bool is_not_vnni_tag = jbgp.wei_dt == f32
            || (jbgp.wei_dt == f16 && jbgp.isa == avx512_core_fp16);
    if (is_not_vnni_tag) {
        if (is_superset(jbgp.isa, avx512_core))
            return {{64,
                            pick(n_sp_dims, OI16i64o, OIw16i64o, OIhw16i64o,
                                    OIdhw16i64o)},
                    {48,
                            pick(n_sp_dims, OI16i48o, OIw16i48o, OIhw16i48o,
                                    OIdhw16i48o)},
                    {32,
                            pick(n_sp_dims, OI16i32o, OIw16i32o, OIhw16i32o,
                                    OIdhw16i32o)},
                    {16,
                            pick(n_sp_dims, OI16i16o, OIw16i16o, OIhw16i16o,
                                    OIdhw16i16o)}};
        else
            return {{32,
                            pick(n_sp_dims, OI8i32o, OIw8i32o, OIhw8i32o,
                                    OIdhw8i32o)},
                    {24,
                            pick(n_sp_dims, OI8i24o, OIw8i24o, OIhw8i24o,
                                    OIdhw8i24o)},
                    {16,
                            pick(n_sp_dims, OI8i16o, OIw8i16o, OIhw8i16o,
                                    OIdhw8i16o)},
                    {8, pick(n_sp_dims, OI8i8o, OIw8i8o, OIhw8i8o, OIdhw8i8o)}};
    } else if (is_xf16) {
        if (jbgp.is_amx) {
            return {{64,
                            pick(n_sp_dims, OI16i64o2i, OIw16i64o2i,
                                    OIhw16i64o2i, OIdhw16i64o2i)},
                    {32,
                            pick(n_sp_dims, OI16i32o2i, OIw16i32o2i,
                                    OIhw16i32o2i, OIdhw16i32o2i)},
                    {16,
                            pick(n_sp_dims, OI16i16o2i, OIw16i16o2i,
                                    OIhw16i16o2i, OIdhw16i16o2i)}};
        } else {
            return {{64,
                            pick(n_sp_dims, OI8i64o2i, OIw8i64o2i, OIhw8i64o2i,
                                    OIdhw8i64o2i)},
                    {32,
                            pick(n_sp_dims, OI8i32o2i, OIw8i32o2i, OIhw8i32o2i,
                                    OIdhw8i32o2i)},
                    {24,
                            pick(n_sp_dims, OI8i24o2i, OIw8i24o2i, OIhw8i24o2i,
                                    OIdhw8i24o2i)},
                    {16,
                            pick(n_sp_dims, OI8i16o2i, OIw8i16o2i, OIhw8i16o2i,
                                    OIdhw8i16o2i)},
                    {8,
                            pick(n_sp_dims, OI8i8o2i, OIw8i8o2i, OIhw8i8o2i,
                                    OIdhw8i8o2i)}};
        }
    } else if (jbgp.wei_dt == data_type::s8) {
        if (jbgp.is_amx) {
            return {{64,
                            pick(n_sp_dims, OI16i64o4i, OIw16i64o4i,
                                    OIhw16i64o4i, OIdhw16i64o4i)},
                    {32,
                            pick(n_sp_dims, OI16i32o4i, OIw16i32o4i,
                                    OIhw16i32o4i, OIdhw16i32o4i)},
                    {16,
                            pick(n_sp_dims, OI16i16o4i, OIw16i16o4i,
                                    OIhw16i16o4i, OIdhw16i16o4i)}};
        } else {
            return {{64,
                            pick(n_sp_dims, OI4i64o4i, OIw4i64o4i, OIhw4i64o4i,
                                    OIdhw4i64o4i)},
                    {32,
                            pick(n_sp_dims, OI4i32o4i, OIw4i32o4i, OIhw4i32o4i,
                                    OIdhw4i32o4i)},
                    {24,
                            pick(n_sp_dims, OI4i24o4i, OIw4i24o4i, OIhw4i24o4i,
                                    OIdhw4i24o4i)},
                    {16,
                            pick(n_sp_dims, OI4i16o4i, OIw4i16o4i, OIhw4i16o4i,
                                    OIdhw4i16o4i)},
                    {8,
                            pick(n_sp_dims, OI4i8o4i, OIw4i8o4i, OIhw4i8o4i,
                                    OIdhw4i8o4i)}};
        }
    } else {
        return {{0, format_tag::undef}};
    }
}

int jit_brgemm_ip_conf_t::get_oc_block(bool try_to_adjust) const {
    const auto &jbgp = *this;

    const bool amx_xf16_bwd_d_noadjust = !try_to_adjust
            && jbgp.prop_kind == backward_data && jbgp.is_amx && !jbgp.is_bf32;
    if (amx_xf16_bwd_d_noadjust) {
        constexpr int amx_xf16_row = 64;
        return amx_xf16_row;
    } else if (!jbgp.is_wei_layout_any) {
        const auto weights_tags = get_desired_weights_tag();
        for (const auto &k : weights_tags)
            if (jbgp.wei_tag == k.second) return k.first;
        assert(!"invalid_tag");
        return 0;
    } else {
        int oc_block = 0;
        const int max_block = is_superset(jbgp.isa, avx512_core) ? 4 : 3;
        if (jbgp.oc >= max_block * jbgp.simd_w) {
            oc_block = max_block * jbgp.simd_w;
        } else if (jbgp.oc >= 2 * jbgp.simd_w) {
            oc_block = 2 * jbgp.simd_w;
        } else {
            oc_block = jbgp.simd_w;
        }

        // Use smaller oc-block to reduce bandwidth requirement for weights and
        // increase parallelism, since no threading will be done in
        // os-direction.
        if (jbgp.use_small_os_kernels) oc_block = 2 * jbgp.simd_w;

        return oc_block;
    }
}

int jit_brgemm_ip_conf_t::get_nb_oc_blocking(bool is_adjustment) const {
    const auto &jbgp = *this;

    const int small_oc_threshold
            = is_superset(jbgp.isa, avx512_core) ? 256 : 128;
    const int small_os_threshold = 8;
    if (jbgp.os <= small_os_threshold && jbgp.oc <= small_oc_threshold) {
        // For small problems compute all oc blocks as a single chunk to avoid
        // parallel section
        return div_up(jbgp.oc,
                is_adjustment ? get_adjusted_oc_block() : get_oc_block());
    } else
        return 1;
}

// Check if work amount is balanced between threads.
static bool is_balanced(int work, int min_work, int nthrs, int goal_nthrs = 0) {
    if (goal_nthrs <= 0) goal_nthrs = nthrs;
    int eff_nthrs = work % nthrs;
    if (!eff_nthrs) return true;
    int work_per_thread = work / nthrs;

    bool imbalanced = work_per_thread <= min_work && eff_nthrs < goal_nthrs;

    return !imbalanced;
}

bool jit_brgemm_ip_conf_t::adjust_thread_balance() const {
    const auto &jbgp = *this;
    const bool is_f32_compute = !jbgp.is_bf32
            && everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);
    const bool is_avx512 = is_superset(jbgp.isa, avx512_core);
    const bool is_f32_compute_avx512 = is_f32_compute && is_avx512;

    const bool skip_thread_balancing = !jbgp.is_amx && !is_f32_compute_avx512;
    if (IMPLICATION(jbgp.is_wei_layout_any, skip_thread_balancing))
        return false;

    int os_chunks = div_up(jbgp.os, get_os_block(true, false));

    int nb_oc = div_up(jbgp.oc, get_oc_block(true));
    int nb_oc_blocking = get_nb_oc_blocking();
    int oc_chunks = div_up(nb_oc, nb_oc_blocking);

    int work_amount = oc_chunks * os_chunks;

    int min_work = 2; // Minimum work per thread.
    int goal_nthrs = jbgp.nthr / 2;

    // f32 case uses different threshold values.
    if (is_f32_compute_avx512) {
        min_work = 3;
        goal_nthrs = jbgp.nthr;
    }

    return !is_balanced(work_amount, min_work, jbgp.nthr, goal_nthrs);
}

int jit_brgemm_ip_conf_t::get_adjusted_oc_block() const {
    const auto &jbgp = *this;
    const bool is_amx_xf16 = jbgp.is_amx && !jbgp.is_bf32;
    const bool is_f32_compute = !jbgp.is_bf32
            && everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);
    const bool is_avx512 = is_superset(jbgp.isa, avx512_core);
    const bool is_f32_compute_avx512 = is_f32_compute && is_avx512;

    // we can't change block size on forward and weights update (external)
    // if layout is set by user, for backward data it can be chosen different
    // from external in this case because copy routine
    const bool not_adjustable_oc_block_size
            = !jbgp.is_wei_layout_any && jbgp.prop_kind != backward_data;

    const bool try_to_adjust
            = is_amx_xf16 || jbgp.is_bf32 || is_f32_compute_avx512;
    if (IMPLICATION(try_to_adjust, not_adjustable_oc_block_size))
        return get_oc_block();

    int oc_block = get_oc_block(true);
    if (adjust_thread_balance()) {
        if (is_f32_compute_avx512) {
            int n = oc_block / jbgp.simd_w;
            bool do_adjust = n > 1 && !jbgp.use_small_os_kernels;
            oc_block = do_adjust ? (n - 1) * jbgp.simd_w : oc_block;
        } else {
            oc_block = (oc_block > 16) ? oc_block / 2 : oc_block;
        }
    }

    constexpr int amx_bf16_half_row = 32;
    // ensure that oc_tail <= amx_bf16_half_row (requirement for brgemm kernel)
    while (jbgp.oc % oc_block > amx_bf16_half_row && !is_f32_compute_avx512)
        oc_block /= 2;
    return oc_block;
}

format_tag_t jit_brgemm_ip_conf_t::get_brgemm_ip_weights_tag(
        const memory_desc_t &weights_md) const {
    const auto &jbgp = *this;

    auto weights_tags = get_desired_weights_tag();
    if (!jbgp.is_wei_layout_any) {
        for (const auto &k : weights_tags) {
            if (memory_desc_matches_tag(weights_md, k.second)) return k.second;
        }
        return format_tag::undef;
    } else {
        const int oc_block = get_adjusted_oc_block();
        return weights_tags[oc_block];
    }
}

static bool post_ops_ok(
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d) {
    using namespace injector;

    const auto &post_ops = attr.post_ops_;

    return injector::post_ops_ok(post_ops_ok_args_t(get_max_cpu_isa(),
            {sum, eltwise, binary}, post_ops, &dst_d,
            false /*sum_at_pos_0_only*/, false /*sum_requires_scale_one*/,
            true /*sum_requires_zp_zero*/, true /*sum_requires_same_params*/,
            {broadcasting_strategy_t::per_oc, broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::no_broadcast}));
}

status_t jit_brgemm_ip_fwd_conf_t::init_conf(cpu_isa_t isa,
        const inner_product_desc_t &ipd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {
    CHECK(init_conf_base(
            isa, ipd, src_md, weights_md, dst_md, bias_md, attr, nthreads));
    assert(one_of(prop_kind, forward_training, forward_inference));

    auto &jbgp = *this;

    const bool is_amx_int8 = jbgp.is_amx && one_of(jbgp.wei_dt, s8, u8);
    const bool is_amx_xf16
            = jbgp.is_amx && one_of(jbgp.wei_dt, bf16, f16) && !jbgp.is_bf32;
    const bool is_int8 = one_of(jbgp.src_dt, u8, s8) && jbgp.wei_dt == s8;
    const bool is_f32_compute = !jbgp.is_bf32
            && everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);
    const auto &p = attr.post_ops_;
    jbgp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jbgp.with_eltwise = eltwise_ind != -1;
    const int binary_ind = p.find(primitive_kind::binary);
    const int prelu_ind = p.find(primitive_kind::prelu);
    jbgp.with_binary = !everyone_is(-1, binary_ind, prelu_ind);
    const memory_desc_wrapper dst_d(&dst_md);
    if (!post_ops_ok(attr, dst_d)) return status::unimplemented;
    if (jbgp.with_scales) {
        const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
        jbgp.is_oc_scale = wei_scales.mask_ != 0;
    }

    const int min_ic_divisor = is_amx_int8 ? 4 : is_amx_xf16 ? 2 : 1;

    jbgp.use_buffer_a = jbgp.ic % min_ic_divisor != 0;

    constexpr int amx_int8_row = 64;
    constexpr int amx_xf16_row = 32;
    jbgp.ic_block = (is_amx_int8) ? amx_int8_row
            : (is_amx_xf16)       ? amx_xf16_row
                                  : jbgp.simd_w;
    jbgp.nb_ic = div_up(jbgp.ic, jbgp.ic_block);

    // gemm-based inner product performs better when oc = 1
    if (is_f32_compute && jbgp.oc == 1) return status::unimplemented;

    jbgp.oc_block = get_adjusted_oc_block();
    jbgp.nb_oc = div_up(jbgp.oc, jbgp.oc_block);
    jbgp.nb_oc_blocking = get_nb_oc_blocking();

    // Use single a single chunk in oc dimension in case of a single main block
    // + tail a block to save bandwidth.
    if (jbgp.nb_oc == 2 && jbgp.oc % jbgp.oc_block != 0) {
        jbgp.nb_oc_blocking = jbgp.nb_oc;
    }

    jbgp.os_block = get_os_block(false, false);
    jbgp.nb_os = div_up(jbgp.os, jbgp.os_block);

    jbgp.nb_os_blocking = 1;
    // Work done by each thread is given by:
    //     (nb_oc / nb_oc_blocking) * (nb_os / nb_os_blocking)
    // For f32 data type we want to increase the nb_os_blocking such that
    //   * 1 <= nb_os_blocking <= 8 AND nb_os_blocking <= nb_os
    //   * Work amount per thread ~ 2
    //   * NOTE: here nb_oc_blocking = 1 as os is large
    if (jbgp.os > 256 && is_f32_compute) {
        jbgp.nb_os_blocking = saturate(1, nstl::min(8, jbgp.nb_os),
                nstl::min(nstl::max(jbgp.oc / jbgp.os / 2, 1),
                        div_up(jbgp.nb_os * jbgp.nb_oc, 2 * jbgp.nthr)));
        jbgp.nb_os_blocking = max_div(jbgp.nb_os, jbgp.nb_os_blocking);
    }

    if (jbgp.nthr == 1 && is_f32_compute) {
        // If a panel src doesn't fit in l2 cache, do all os-blocks at once to
        // improve bandwidth for weights.
        size_t src_panel_sz
                = jbgp.os_block * jbgp.ic * types::data_type_size(jbgp.src_dt);
        size_t l2_sz = platform::get_per_core_cache_size(2);
        if (src_panel_sz > l2_sz && jbgp.nb_oc_blocking == 1)
            jbgp.nb_os_blocking = jbgp.nb_os;
    }

    // NOTE: comment about is_gigantic_shape is in get_os_block()
    const bool is_gigantic_shape = jbgp.oc >= 4096 && jbgp.os >= 512;

    int oc_chunks = div_up(jbgp.nb_oc, jbgp.nb_oc_blocking);
    int os_chunks = div_up(jbgp.nb_os, jbgp.nb_os_blocking);
    int other_work = oc_chunks * os_chunks;

    const int max_nb_ic_blocking = nstl::min(64, jbgp.nb_ic);
    const int min_ic_chunks = jbgp.nb_ic / max_nb_ic_blocking;

    // Use parallel IC reduction for xf16 if we have:
    //  * Very large input channels.
    //  * Number of threads > 1.
    //  * AMX isa
    //  * Low amount of parallelism on os/oc dimensions compared to ic dimension
    //    with high level of thread imbalance
    //  * Single chunk wrt os dimension
    bool small_dst_chunk_size = jbgp.oc_block * jbgp.nb_oc_blocking
                    * jbgp.os_block * jbgp.nb_os_blocking
            <= 1024;
    bool low_parallelism_level = os_chunks == 1 && small_dst_chunk_size
            && !is_balanced(other_work, 2, jbgp.nthr, jbgp.nthr / 2);
    bool use_parallel_ic_reduction_for_bf16
            = is_amx_xf16 && jbgp.ic > 4 * 1024 && low_parallelism_level;
    bool low_work_amount = other_work < 2 * min_ic_chunks * jbgp.nthr;

    // Use parallel IC reduction for f32 if we have:
    //  * Very large input channels.
    //  * Low amount of parallelism on os/oc dimensions compared to ic dimension.
    //  * Number of threads > 1.
    //  * Not a "gigantic shape" since it already has a lot of parallelism
    //    in os and oc dimensions w/o enabling IC parallelism.
    bool use_parallel_ic_reduction_for_f32 = is_f32_compute && jbgp.ic > 1024
            && low_work_amount && jbgp.nthr > 1 && !is_gigantic_shape;
    bool use_parallel_ic_reduction = use_parallel_ic_reduction_for_f32
            || use_parallel_ic_reduction_for_bf16;

    // For os > 256, compute all os blocks as a single chunk when performing
    // IC reduction. Note that this condition is empirical
    if (use_parallel_ic_reduction && jbgp.os > 256 && jbgp.nb_os_blocking > 1)
        jbgp.nb_os_blocking = jbgp.nb_os;

    jbgp.nb_ic_blocking = 1;
    jbgp.nthr_ic_b = 1;
    const int k_blk = jbgp.is_bf32 ? amx_xf16_row : jbgp.ic_block;
    const bool trivial_shape
            = everyone_is(1, jbgp.kw, jbgp.kh, jbgp.kd) && !jbgp.use_buffer_a;
    const bool small_ic = jbgp.ic <= max_nb_ic_blocking * jbgp.ic_block;
    const bool avx2_small_os = jbgp.isa == avx2 && jbgp.nb_os == 1;
    if (trivial_shape && (is_int8 || small_ic || avx2_small_os)) {
        // Optimization: data & weights layouts allow to generate
        // brgemm kernel with K = ic & batch = 1
        // (K = rnd_dn(ic, ic_block), K_tail = ic % ic_block & batch = 1)
        // instead of K = ic_block & batch = nb_ic_blocking
        jbgp.K = jbgp.ic <= jbgp.ic_block ? jbgp.ic : rnd_dn(jbgp.ic, k_blk);
        jbgp.nb_ic_blocking = jbgp.nb_ic;
        jbgp.gemm_batch_size = 1;
    } else if (!jbgp.use_buffer_a && use_parallel_ic_reduction) {
        const int min_chunk_sz = 16;
        const int num_min_chunk_sz = div_up(jbgp.nb_ic, min_chunk_sz);
        int reduce_work = int(0.5f * num_min_chunk_sz * jbgp.nb_os
                + (float)num_min_chunk_sz / jbgp.nb_oc + 0.5f);
        const int max_nthr_ic_b = nstl::min(
                nstl::min(
                        jbgp.nb_ic >= 1024 || use_parallel_ic_reduction_for_bf16
                                ? 8
                                : 4,
                        num_min_chunk_sz),
                jbgp.nthr);

        // Don't sacrifice reduction threads if other dimension will
        // not benefit.
        int nthr_other = jbgp.nthr / reduce_work;
        if (reduce_work < max_nthr_ic_b && nthr_other <= 1) {
            reduce_work = max_nthr_ic_b;
        }

        jbgp.nthr_ic_b = saturate(1, max_nthr_ic_b, reduce_work);

        int prev_work
                = comp_work(jbgp.nthr, jbgp.nthr_ic_b, other_work, jbgp.nb_ic);
        while (jbgp.nthr_ic_b > 1) {
            int kthr = jbgp.nthr_ic_b - 1;
            int nthr_other = jbgp.nthr / kthr;

            int work = comp_work(jbgp.nthr, kthr, other_work, jbgp.nb_ic);

            // Sacrifice a thread in reduce dimension if work amount will be
            // reduced on the thread with most work.
            bool less_work = prev_work > work && nthr_other > 1;

            // Reduce number of reduction threads if non-reduction dimension
            // still have opportunities for parallelism.
            const int chunks_per_thr = 15;
            int min_other_work = chunks_per_thr * nthr_other;
            bool prefer_other = other_work > min_other_work;

            // Update previous work for next iteration if needed.
            prev_work = work;

            if (less_work || prefer_other)
                jbgp.nthr_ic_b = kthr;
            else
                break;
        }

        assert(jbgp.nthr_ic_b >= 1);
        jbgp.nb_ic_blocking = div_up(jbgp.nb_ic, jbgp.nthr_ic_b);
        jbgp.nb_ic_blocking /= div_up(jbgp.nb_ic_blocking, 64);

        jbgp.gemm_batch_size = jbgp.nb_ic_blocking;
        jbgp.K = jbgp.ic_block;
    } else {
        // Note: Here, ic divided into K_blocks of gemm_batch
        const int ic_blks_per_k = div_up(k_blk, jbgp.ic_block);
        const int nb_k_blk = div_up(jbgp.ic, k_blk);
        const int max_nb_k_blocking = div_up(max_nb_ic_blocking, ic_blks_per_k);
        int nb_k_blocking = max_div(nb_k_blk, max_nb_k_blocking);
        const bool small_nb_k_blk = nb_k_blk <= max_nb_k_blocking;
        if (small_nb_k_blk && nb_k_blocking == 1)
            nb_k_blocking = max_nb_k_blocking;

        // For non small_nb_ic [i.e. that has nb_ic > 64] shape that has
        // gcd(nb_ic, 64) < 16, we manually set nb_ic_blocking = 64
        // the coefficients 64 [used in max_nb_ic_blocking] and 16 are empirical
        const int min_nb_k_blocking = small_nb_k_blk ? 1 : 16;
        if (nb_k_blocking < min_nb_k_blocking)
            nb_k_blocking = max_nb_k_blocking;

        jbgp.nb_ic_blocking = nb_k_blocking * ic_blks_per_k;
        jbgp.K = k_blk;
        jbgp.gemm_batch_size = nb_k_blocking;
    }

    const int nthrs_other = jbgp.nthr / jbgp.nthr_ic_b;
    const int min_work = 15;

    bool balanced = is_balanced(other_work, min_work, nthrs_other);

    // Reduce os-block as needed for better thread balance for f32 case.
    const bool is_avx512 = is_superset(jbgp.isa, avx512_core);
    const int min_os_block = 6;
    while (!balanced && jbgp.os_block > min_os_block && is_f32_compute
            && is_avx512) {
        int max_os_block = jbgp.os_block - 1;
        jbgp.os_block = max_div(jbgp.os, max_os_block);
        jbgp.os_block = nstl::max(jbgp.os_block, min_os_block);
        jbgp.nb_os = div_up(jbgp.os, jbgp.os_block);
        jbgp.nb_os_blocking = max_div(jbgp.nb_os, jbgp.nb_os_blocking);
        os_chunks = div_up(jbgp.nb_os, jbgp.nb_os_blocking);
        other_work = os_chunks * oc_chunks;
        balanced = is_balanced(other_work, min_work, nthrs_other);
    }

    // to avoid cache concurrent write access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jbgp.adjusted_batch_size
            = div_up(rnd_up(jbgp.gemm_batch_size * sc_size, 4096), sc_size);

    if (is_amx_xf16 || jbgp.is_bf32) {
        if (adjust_thread_balance()) {
            // Adjust oc_block to improve thread balancing
            jbgp.oc_block = get_adjusted_oc_block();
            jbgp.nb_oc = div_up(jbgp.oc, jbgp.oc_block);
            jbgp.nb_oc_blocking = get_nb_oc_blocking(true);

            // Adjust os_block to improve thread balancing
            if (jbgp.oc <= 16
                    || types::data_type_size(jbgp.src_dt) * jbgp.mb * jbgp.ic
                            <= (size_t)platform::get_per_core_cache_size(2)) {
                jbgp.os_block = get_os_block(false, true);
                jbgp.nb_os = div_up(jbgp.os, jbgp.os_block);
            }
        }
    }

    jbgp.use_buffer = (IMPLICATION(jbgp.dst_dt == jbgp.acc_dt, jbgp.with_sum))
            || (jbgp.nthr_ic_b > 1);

    // NOTE: Choose loop order before setting brgemm buffer leading dimensions,
    // since buffer size might depend on loop order chosen.
    choose_loop_order();

    // If innermost loop on driver around the kernel matches kernel outermost
    // loop dimension (m-dim), we can reduce blocking to the same used in
    // register blocking in m-dim to force most efficient register
    // decomposition and avoid tail cases.
    bool no_inner_blocking_loops
            = jbgp.nb_os_blocking == 1 && jbgp.nb_oc_blocking == 1;
    bool use_min_os_block
            = no_inner_blocking_loops && jbgp.loop_order == icc_occ_osc_ocb_osb;

    // TODO: Consider expanding to other arches and data types.
    use_min_os_block &= is_f32_compute && is_avx512;

    if (use_min_os_block) {
        // Get potential bd_block from main kernel.
        brgemm_t brg_desc;
        CHECK(brgemm_desc_init(&brg_desc, isa, jbgp.brg_type, jbgp.src_dt,
                jbgp.wei_dt, false, false, brgemm_row_major, 1.0f, 1.0f,
                jbgp.ic_without_padding, jbgp.oc_block, jbgp.oc_without_padding,
                jbgp.os_block, jbgp.oc_block, jbgp.K));
        int bd_block = brg_desc.bd_block;

        if (jbgp.oc_block == 64 && bd_block != 6) jbgp.os_block = 6;
        if (jbgp.oc_block == 48 && bd_block != 8) jbgp.os_block = 8;
        jbgp.nb_os = div_up(jbgp.os, jbgp.os_block);
    }

    // Configure matrix sizes
    jbgp.M = jbgp.os_block;
    jbgp.M_tail = jbgp.os % jbgp.os_block;

    jbgp.N = jbgp.oc_block;
    jbgp.N_tail = jbgp.oc % jbgp.oc_block;
    jbgp.K_tail = jbgp.use_buffer_a ? 0 : jbgp.ic % jbgp.K;

    jbgp.LDA = jbgp.use_buffer_a ? jbgp.K * jbgp.gemm_batch_size
                                 : jbgp.ic_without_padding;
    jbgp.LDB = jbgp.N;
    jbgp.LDD = jbgp.oc_without_padding;
    jbgp.LDC = jbgp.LDD;
    if (jbgp.use_buffer && jbgp.nthr_ic_b == 1) {
        // Adjust LDC according to buffer size used at execute stage.
        switch (jbgp.loop_order) {
            case osc_occ_osb_ocb_icc: jbgp.LDC = jbgp.N; break;
            case osc_occ_icc_osb_ocb:
                jbgp.LDC = jbgp.oc_block * jbgp.nb_oc_blocking;
                break;
            case icc_osc_occ_osb_ocb:
            case icc_occ_osc_ocb_osb: jbgp.LDC = jbgp.LDD; break;
        }
    }

    if (jbgp.is_bf32) {
        const float M = static_cast<float>(jbgp.M);
        const float N = nstl::min<float>(jbgp.N, jbgp.oc);
        const float K
                = nstl::min<float>(jbgp.K * jbgp.gemm_batch_size, jbgp.ic);
        const float tmul_efficiency = (M / 16) * (N / 16) * (K / 32);
        // TODO: Adjust blocking such that bigger M, N, K are generated.
        if (one_of(true, M <= 8, K <= 8, N < 16, tmul_efficiency <= 2.25))
            return status::unimplemented;
    }

    return status::success;
}

status_t jit_brgemm_ip_bwd_d_conf_t::init_conf(cpu_isa_t isa,
        const inner_product_desc_t &ipd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {
    CHECK(init_conf_base(
            isa, ipd, src_md, weights_md, dst_md, bias_md, attr, nthreads));
    assert(prop_kind == backward_data);

    auto &jbgp = *this;

    const bool is_amx_xf16 = jbgp.is_amx && !jbgp.is_bf32;
    const bool is_avx512_bf16 = jbgp.isa == avx512_core_bf16;
    const bool is_f32_compute = !jbgp.is_bf32
            && everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);
    const bool is_bf16 = everyone_is(bf16, jbgp.wei_dt, jbgp.dst_dt);

    constexpr int amx_xf16_granularity = 2;
    jbgp.use_buffer_a = is_amx_xf16 && jbgp.oc % amx_xf16_granularity != 0;
    jbgp.use_buffer_b = true;
    jbgp.global_b_transpose = false;

    jbgp.oc_block = get_adjusted_oc_block();

    const int n_sp_dims = jbgp.ndims - 2;
    const format_tag_t wei_blk_8
            = pick(n_sp_dims, OI8i8o2i, OIw8i8o2i, OIhw8i8o2i, OIdhw8i8o2i);
    const format_tag_t wei_blk_24
            = pick(n_sp_dims, OI8i24o2i, OIw8i24o2i, OIhw8i24o2i, OIdhw8i24o2i);
    // Note: these wei tags are currently unsupported in the transform JIT
    // kernels.
    if (one_of(jbgp.wei_tag, wei_blk_8, wei_blk_24))
        return status::unimplemented;

    // Optimization: for small shape we avoid large ic_block
    // Thinking of os, ic, and oc as three dimensions, the boundary for small
    // shapes is heuristically chosen via the following constraints:
    //   os <= 128 && max(ic, oc) <= 2048 && min(ic, oc) <= 1000
    //
    // TODO: Will the optimization be useful for bf16 data type
    const bool avoid_max_ic_block = is_f32_compute && jbgp.os <= 128
            && nstl::max(jbgp.ic, jbgp.oc) <= 2048
            && nstl::min(jbgp.ic, jbgp.oc) <= 1000;
    const int max_ch_block_mult = is_superset(jbgp.isa, avx512_core) ? 4 : 3;
    const int max_ch_block = max_ch_block_mult * jbgp.simd_w;
    if (!avoid_max_ic_block
            && jbgp.ic >= (is_f32_compute ? 512 : max_ch_block)) {
        jbgp.ic_block = max_ch_block;
    } else if (jbgp.ic >= 2 * jbgp.simd_w) {
        jbgp.ic_block = 2 * jbgp.simd_w;
    } else {
        jbgp.ic_block = jbgp.simd_w;
    }

    jbgp.nb_ic = div_up(jbgp.ic, jbgp.ic_block);
    jbgp.nb_ic_blocking = 1;
    jbgp.nb_oc = div_up(jbgp.oc, jbgp.oc_block);

    jbgp.os_block = get_os_block(false, false);

    jbgp.nb_os = div_up(jbgp.os, jbgp.os_block);
    jbgp.nb_os_blocking = 1;
    int os_blocking_max = 2;
    jbgp.nb_os_blocking = max_div(jbgp.nb_os, os_blocking_max);

    if (is_amx_xf16 || jbgp.is_bf32) {
        const int os_chunks = div_up(jbgp.nb_os, jbgp.nb_os_blocking);
        const int work_amount = jbgp.nb_ic * os_chunks;
        float wb_ratio = (float)work_amount / (float)jbgp.nthr;
        if (wb_ratio != 1.f && wb_ratio < 2.f) {
            jbgp.ic_block
                    = (jbgp.ic_block > 16) ? jbgp.ic_block / 2 : jbgp.ic_block;
            jbgp.nb_ic = div_up(jbgp.ic, jbgp.ic_block);
        }
    }

    jbgp.nb_oc_blocking = 1;
    const int oc_chunk_max_size = max_ch_block;
    jbgp.nb_oc_blocking = max_div(jbgp.nb_oc, oc_chunk_max_size);

    if (jbgp.isa == avx2) {
        const auto L2_size = platform::get_per_core_cache_size(2) * jbgp.nthr;
        for (int bl = jbgp.nb_oc; bl >= 1; bl--) {
            jbgp.nb_oc_blocking = bl;
            if (L2_size >= types::data_type_size(jbgp.src_dt) * jbgp.os_block
                                    * jbgp.nb_os_blocking
                            + types::data_type_size(jbgp.wei_dt) * jbgp.oc
                                    * jbgp.nb_oc_blocking)
                break;
        }
    }

    jbgp.nthr_oc_b = 1;
    const int ic_chunks = div_up(jbgp.nb_ic, jbgp.nb_ic_blocking);
    const int os_chunks = div_up(jbgp.nb_os, jbgp.nb_os_blocking);
    const int other_work = ic_chunks * os_chunks;
    // Use oc reduction if we have
    //   * very large output channels
    //   * small work amount available to each thread
    if ((other_work < 2 * jbgp.nthr
                || jbgp.oc > (is_bf16 || jbgp.is_bf32 ? 4096 : 1024))) {
        const int min_chunk_sz
                = (is_avx512_bf16) ? 2 * jbgp.simd_w : jbgp.simd_w;
        const int num_min_chunk_sz = div_up(jbgp.nb_oc, min_chunk_sz);
        int reduce_work = int(0.5f * num_min_chunk_sz * jbgp.nb_os
                + (float)num_min_chunk_sz / jbgp.nb_ic + 0.5f);

        // optimization for transformer_lt on CPX/SKX
        const int max_nthr_oc_b = nstl::min(
                nstl::min((!is_amx_xf16 && !jbgp.is_bf32 && jbgp.oc > 32000)
                                ? jbgp.nthr / 2
                                : 4,
                        num_min_chunk_sz),
                jbgp.nthr);

        if (is_f32_compute) {
            // Don't sacrifice reduction threads if other dimension will
            // not benefit.
            int nthr_other = jbgp.nthr / reduce_work;
            if (reduce_work < max_nthr_oc_b && nthr_other <= 1) {
                reduce_work = max_nthr_oc_b;
            }
        }

        jbgp.nthr_oc_b = saturate(1, max_nthr_oc_b, reduce_work);

        bool is_1d_ic = os_chunks == 1 && ic_chunks > 1;
        bool is_1d_os = ic_chunks == 1 && os_chunks > 1;
        bool is_1d = is_1d_ic || is_1d_os;
        if (is_f32_compute && is_1d && jbgp.nthr_oc_b > 1) {
            int n_chunks = is_1d_ic ? ic_chunks : os_chunks;
            int nthr_1 = jbgp.nthr_oc_b;
            int nthr_2 = nthr_1 - 1;
            int nthr_other = jbgp.nthr / nthr_2;

            int work_1 = comp_work(jbgp.nthr, nthr_1, n_chunks, jbgp.nb_oc);
            int work_2 = comp_work(jbgp.nthr, nthr_2, n_chunks, jbgp.nb_oc);

            // Sacrifice a thread in reduce dimension if work amount will be
            // reduce on the thread with most work.
            if (work_1 >= work_2 && nthr_other > 1) jbgp.nthr_oc_b--;
        }

        if (jbgp.nthr_oc_b > 1) {
            jbgp.nb_oc_blocking = div_up(jbgp.nb_oc, jbgp.nthr_oc_b);
            jbgp.nb_oc_blocking
                    /= div_up(jbgp.nb_oc_blocking, oc_chunk_max_size);
        }
    }
    jbgp.gemm_batch_size = jbgp.nb_oc_blocking;
    // to avoid cache concurrent write access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jbgp.adjusted_batch_size
            = div_up(rnd_up(jbgp.gemm_batch_size * sc_size, 4096), sc_size);

    jbgp.use_buffer = jbgp.src_dt != jbgp.acc_dt || jbgp.nthr_oc_b > 1;

    jbgp.M = jbgp.os_block;
    jbgp.M_tail = jbgp.os % jbgp.os_block;

    jbgp.K = jbgp.oc_block;
    jbgp.N = jbgp.ic_block;
    jbgp.N_tail = jbgp.ic % jbgp.ic_block;
    jbgp.K_tail = jbgp.use_buffer_a ? 0 : jbgp.oc % jbgp.oc_block;

    jbgp.LDA = jbgp.use_buffer_a ? jbgp.K * jbgp.nb_oc_blocking
                                 : jbgp.oc_without_padding;
    jbgp.LDB = jbgp.N;
    jbgp.LDD = jbgp.ic_without_padding;
    jbgp.LDC = jbgp.use_buffer && jbgp.nthr_oc_b == 1 ? jbgp.N : jbgp.LDD;

    if (jbgp.is_bf32) {
        const float M = static_cast<float>(jbgp.M);
        const float N = nstl::min<float>(jbgp.N, jbgp.ic);
        const float K
                = nstl::min<float>(jbgp.K * jbgp.gemm_batch_size, jbgp.oc);
        const float tmul_efficiency = (M / 16) * (N / 16) * (K / 32);
        // TODO: Adjust blocking such that bigger M, N, K are generated.
        if (one_of(true, M <= 8, K <= 8, N < 16, tmul_efficiency <= 2.25))
            return status::unimplemented;
    }

    return status::success;
}

void jit_brgemm_ip_bwd_w_conf_t::thread_balance(int &nb_os_blocking_,
        int &nb_oc_blocking_, int &nb_ic_blocking_, int &nthr_, int &nthr_mb_,
        int &nthr_oc_b_, int &nthr_ic_b_) const {
    const auto &j = *this;

    nthr_ = nthr_mb_ = nthr_oc_b_ = nthr_ic_b_ = 1;
    nb_os_blocking_ = j.nb_os_blocking;
    nb_oc_blocking_ = j.nb_oc_blocking;
    nb_ic_blocking_ = j.nb_ic_blocking;

    const bool is_f32 = everyone_is(f32, j.src_dt, j.wei_dt, j.dst_dt);
    const bool is_xf16 = one_of(j.src_dt, bf16, f16) && (j.src_dt == j.dst_dt);

    const int max_threads = j.nthr;
    const int nthr = max_threads;
    auto calc_mem_cost = [=](int nb_os_blocking, int nb_oc_blocking,
                                 int nb_ic_blocking, int nthr_mb, int nthr_oc,
                                 int nthr_ic) {
        float src_size = static_cast<float>(j.ic) * j.mb;
        float dst_size = static_cast<float>(j.oc) * j.mb;
        float wei_size = static_cast<float>(j.ic) * j.oc;
        int os_chunks = div_up(j.nb_os, nb_os_blocking);
        int oc_chunks = div_up(j.nb_oc, nb_oc_blocking);
        int ic_chunks = div_up(j.nb_ic, nb_ic_blocking);

        float wei_compensation_scale = 0.5f * (dst_size + src_size) / wei_size;

        float oi_channels_ratio = 0;
        if (is_xf16) {
            oi_channels_ratio = ((j.oc > 3 * j.ic && os_chunks > 1)
                                        || (os_chunks == 1 && j.ic > j.oc))
                    ? src_size / dst_size
                    : dst_size / src_size;
        } else {
            oi_channels_ratio = src_size / dst_size;
        }

        auto get_src_coef = [=]() {
            if (is_f32) {
                float src_coef = nstl::max(1.0f / oi_channels_ratio, 1.0f);
                src_coef *= types::data_type_size(j.src_dt);
                src_coef *= 4 * saturate(1, 4, div_up(j.ic, 1024));
                if (wei_compensation_scale < 2.0f)
                    src_coef += sqrtf(2.0f / wei_compensation_scale);
                return src_coef;
            }
            float src_coef = nstl::max(1.0f / oi_channels_ratio, 1.0f);
            src_coef *= 4 * types::data_type_size(j.src_dt);
            if (wei_compensation_scale < 1.0f) src_coef *= 4.0f;

            return src_coef;
        };

        auto get_dst_coef = [=]() {
            if (is_f32) {
                float dst_coef = types::data_type_size(j.dst_dt)
                        * nstl::max(oi_channels_ratio, 1.0f);
                return dst_coef;
            }

            return 2 * types::data_type_size(j.dst_dt)
                    * nstl::max(oi_channels_ratio, 1.0f);
        };

        auto get_wei_coef = [=]() {
            if (is_f32) {
                return nstl::max(
                        4.0f - j.mb / 2048 * wei_compensation_scale, 1.0f);
            }

            // limit the range of coefficient values to have more stable behavior
            // for extreme cases
            const float low_limit = 1.0f;
            const float upper_limit = 1024.0f;
            return utils::saturate(
                    low_limit, upper_limit, wei_compensation_scale);
        };

        float src_tr = 0.0f;
        if (j.use_buffer_a && !is_f32) {
            int src_tr_oc_par_work = div_up(os_chunks, nthr_mb)
                    * div_up(ic_chunks, nthr_ic) * nb_ic_blocking;
            src_tr = get_src_coef() * div_up(src_tr_oc_par_work, nthr_oc)
                    * nb_os_blocking * j.os_block * j.ic_block;
        }

        float dst_tr = 0.0f;
        if (j.use_buffer_b && !is_f32) {
            int dst_tr_ic_par_work = div_up(os_chunks, nthr_mb)
                    * div_up(oc_chunks, nthr_oc) * nb_oc_blocking;
            dst_tr = get_dst_coef() * div_up(dst_tr_ic_par_work, nthr_ic)
                    * nb_os_blocking * j.os_block * j.oc_block;
        }

        float src_v = get_src_coef() * div_up(os_chunks, nthr_mb)
                * div_up(ic_chunks, nthr_ic) * nb_os_blocking * j.os_block
                * nb_ic_blocking * j.ic_block;
        float dst_v = get_dst_coef() * div_up(os_chunks, nthr_mb)
                * div_up(oc_chunks, nthr_oc) * nb_os_blocking * j.os_block
                * nb_oc_blocking * j.oc_block;

        auto acc_dt_sz = types::data_type_size(j.acc_dt);
        float wei_v = get_wei_coef() * acc_dt_sz * div_up(oc_chunks, nthr_oc)
                * div_up(ic_chunks, nthr_ic) * nb_oc_blocking * j.oc_block
                * nb_ic_blocking * j.ic_block;

        float wei_r = 0;
        if (nthr_mb > 1) {
            auto wei_dt_sz = types::data_type_size(j.wei_dt);
            int wei_r_mb_par_work = div_up(oc_chunks, nthr_oc)
                    * div_up(ic_chunks, nthr_ic) * nb_oc_blocking
                    * nb_ic_blocking;
            wei_r = get_wei_coef() * div_up(wei_r_mb_par_work, nthr_mb)
                    * j.oc_block * j.ic_block
                    * (wei_dt_sz
                            + (is_f32 ? div_up(j.os, 1024) : 1) * nthr_mb
                                    * acc_dt_sz);
        }

        return src_tr + dst_tr + src_v + dst_v + wei_v + wei_r;
    };

    float best_mem_cost = calc_mem_cost(nb_os_blocking_, nb_oc_blocking_,
            nb_ic_blocking_, nthr_mb_, nthr_oc_b_, nthr_ic_b_);

    /* Set range of values for nb_oc_blocking/nb_ic_blocking parameters to try.
       Use powers-of-2 values to avoid potential issues on converting to
       blocked weights layout stage
    */
    auto get_blk_values
            = [](int max_blk_value, int init_blk, int dim_blk_limit) {
                  int val_1st = rnd_up_pow2(init_blk);
                  int val_end = nstl::min(max_blk_value, dim_blk_limit);
                  std::vector<int> values;
                  for (int val = val_1st; val <= val_end; val <<= 1)
                      values.push_back(val);
                  return values;
              };

    const int max_nb_oc_blocking_pow
            = j.local_buffers_for_input_tensors ? 4 : j.nb_oc_blocking;
    auto nb_oc_blocking_values
            = get_blk_values(max_nb_oc_blocking_pow, j.nb_oc_blocking, j.nb_oc);
    const int max_nb_ic_blocking_pow
            = j.local_buffers_for_input_tensors ? 4 : j.nb_ic_blocking;
    auto nb_ic_blocking_values
            = get_blk_values(max_nb_ic_blocking_pow, j.nb_ic_blocking, j.nb_ic);

    /* find the best thread distribution with lowest memory cost */
    const int min_osb_chunk = is_f32 ? 32 : is_xf16 ? 8 : 1;
    const int nthr_mb_max = nstl::min(nthr, div_up(j.nb_os, min_osb_chunk));
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        int nb_os_blocking = j.nb_os_blocking;
        int os_chunks = div_up(j.nb_os, nb_os_blocking);
        if (os_chunks < nthr_mb) {
            int coef = saturate(1, 4, 2 * j.mb / (j.oc + j.ic));
            int os_blocking_max = div_up(div_up(j.nb_os, coef), nthr_mb);
            nb_os_blocking = max_div(j.nb_os, os_blocking_max);
        }

        const int nthr_par = nthr / nthr_mb;

        for (auto nb_oc_blocking : nb_oc_blocking_values) {
            int num_oc_chunks = div_up(j.nb_oc, nb_oc_blocking);
            const int nthr_oc_b_max = nstl::min(nthr_par, num_oc_chunks);
            for_(int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b)
            for (auto nb_ic_blocking : nb_ic_blocking_values) {
                int num_ic_chunks = div_up(j.nb_ic, nb_ic_blocking);

                int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, num_ic_chunks);
                float mem_cost = calc_mem_cost(nb_os_blocking, nb_oc_blocking,
                        nb_ic_blocking, nthr_mb, nthr_oc_b, nthr_ic_b);
                if (mem_cost <= best_mem_cost) {
                    best_mem_cost = mem_cost;
                    nb_os_blocking_ = nb_os_blocking;
                    nb_oc_blocking_ = nb_oc_blocking;
                    nb_ic_blocking_ = nb_ic_blocking;
                    nthr_mb_ = nthr_mb;
                    nthr_oc_b_ = nthr_oc_b;
                    nthr_ic_b_ = nthr_ic_b;
                }
            }
        }
    }

    nthr_ = nthr_mb_ * nthr_oc_b_ * nthr_ic_b_;
}

status_t jit_brgemm_ip_bwd_w_conf_t::init_conf(cpu_isa_t isa,
        const inner_product_desc_t &ipd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {
    CHECK(init_conf_base(
            isa, ipd, src_md, weights_md, dst_md, bias_md, attr, nthreads));
    assert(prop_kind == backward_weights);

    auto &jbgp = *this;

    const bool is_amx_xf16 = jbgp.is_amx && !jbgp.is_bf32;
    const bool is_f32 = everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);
    const bool has_weights_buffer = jbgp.wei_dt != jbgp.acc_dt;

    const int amx_xf16_row = 64;
    const bool big_ic_blk_ok = is_f32 && jbgp.ic % (4 * jbgp.simd_w) == 0
            && (jbgp.mb <= 128
                    || (jbgp.isa == avx2 && jbgp.oc <= 256 /*avx2 ncf model*/));
    const int max_c_mult = is_superset(jbgp.isa, avx512_core) ? 4 : 3;
    jbgp.ic_block = big_ic_blk_ok && !is_amx_xf16 ? max_c_mult * jbgp.simd_w
            : (is_amx_xf16 && has_weights_buffer) ? amx_xf16_row
                                                  : jbgp.simd_w;
    jbgp.ic_block_ext
            = is_amx_xf16 || (jbgp.wei_dt == dnnl::impl::data_type::bf16)
            ? 32
            : jbgp.simd_w;

    jbgp.oc_block
            = has_weights_buffer ? get_oc_block() : get_adjusted_oc_block();
    jbgp.oc_block_ext = get_adjusted_oc_block();

    const int n_sp_dims = jbgp.ndims - 2;
    const format_tag_t wei_blk_8
            = pick(n_sp_dims, OI8i8o2i, OIw8i8o2i, OIhw8i8o2i, OIdhw8i8o2i);
    const format_tag_t wei_blk_24
            = pick(n_sp_dims, OI8i24o2i, OIw8i24o2i, OIhw8i24o2i, OIdhw8i24o2i);
    // Note: these wei tags are currently unsupported in the transform JIT
    // kernels.
    if (one_of(jbgp.wei_tag, wei_blk_8, wei_blk_24))
        return status::unimplemented;

    jbgp.os_block = get_os_block(false, false);
    jbgp.nb_os = div_up(jbgp.os, jbgp.os_block);

    jbgp.nb_ic = div_up(jbgp.ic, jbgp.ic_block);
    jbgp.nb_oc = div_up(jbgp.oc, jbgp.oc_block);
    jbgp.nb_oc_blocking = 1;
    jbgp.nb_ic_blocking = jbgp.nb_ic % 2 ? 1 : 2;

    // Configure matrix sizes
    jbgp.M = jbgp.ic_block;
    jbgp.M_tail = jbgp.ic % jbgp.ic_block;

    jbgp.N = jbgp.oc_block;
    jbgp.N_tail = jbgp.oc % jbgp.oc_block;

    constexpr int amx_xf16_granularity = 2;
    // sanity check, must hold for transpose routines to work fine
    assert(IMPLICATION(is_amx_xf16, jbgp.os_block % amx_xf16_granularity == 0));
    const bool do_rnd_os = is_amx_xf16 && jbgp.os % amx_xf16_granularity != 0;

    jbgp.K = jbgp.os_block;
    jbgp.K_tail = (jbgp.os % jbgp.os_block) + (do_rnd_os ? 1 : 0);

    jbgp.nb_os_blocking = 1;
    int os_blocking_max = (is_amx_xf16 && jbgp.nb_os >= 64)
            ? (types::data_type_size(jbgp.src_dt) * jbgp.mb * jbgp.ic
                      < platform::get_per_core_cache_size(2))
                    ? 8
                    : 4
            : nstl::min(64, jbgp.nb_os);
    jbgp.nb_os_blocking = max_div(jbgp.nb_os, os_blocking_max);

    jbgp.use_buffer_a = true;
    const bool is_oc_big_2_pow = jbgp.oc >= 512 && math::is_pow2(jbgp.oc);
    const bool is_huge_oc = jbgp.oc >= (jbgp.isa == avx2 ? 2 : 4) * 1024;
    jbgp.use_buffer_b = jbgp.dst_dt != f32 || is_oc_big_2_pow || is_huge_oc;
    const bool os_dim_dominating = jbgp.os >= 5 * (jbgp.ic + jbgp.oc);
    const int big_nb_os_threshold = is_amx_xf16 ? 64 : 256;
    jbgp.local_buffers_for_input_tensors
            = is_amx_xf16 && jbgp.nb_os >= big_nb_os_threshold;
    jbgp.harness = (jbgp.isa == avx2)
                    || (os_dim_dominating && jbgp.nb_os >= big_nb_os_threshold)
            ? harness_mb_reduction
            : harness_2d_reduction;

    int nb_os_blocking, nb_oc_blocking, nb_ic_blocking, nthr, nthr_mb, nthr_oc,
            nthr_ic;
    // Caution: thread_balance requires `use_buffer_a` and `use_buffer_b`
    // fields of jbgp to be properly set
    thread_balance(nb_os_blocking, nb_oc_blocking, nb_ic_blocking, nthr,
            nthr_mb, nthr_oc, nthr_ic);

    jbgp.nb_os_blocking = jbgp.isa == avx2 ? 1 : nb_os_blocking;
    jbgp.nb_oc_blocking = nb_oc_blocking;
    jbgp.nb_ic_blocking = nb_ic_blocking;
    jbgp.nthr = nthr;
    jbgp.nthr_mb = nthr_mb;
    jbgp.nthr_oc_b = nthr_oc;
    jbgp.nthr_ic_b = nthr_ic;

    jbgp.gemm_batch_size = jbgp.nb_os_blocking;
    // to avoid cache concurrent write access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jbgp.adjusted_batch_size
            = div_up(rnd_up(jbgp.gemm_batch_size * sc_size, 4096), sc_size);

    jbgp.use_buffer = IMPLICATION(!has_weights_buffer, jbgp.nthr_mb > 1);

    // NOTE: Choose loop order before setting brgemm buffer leading dimensions,
    // since buffer size might depend on loop order chosen.
    choose_loop_order();

    jbgp.LDA = jbgp.K;
    jbgp.LDB = (jbgp.use_buffer_b) ? jbgp.N * jbgp.nb_oc_blocking
                                   : jbgp.oc_without_padding;
    jbgp.LDC = jbgp.LDD = jbgp.N;

    if (jbgp.is_bf32) {
        const float M = static_cast<float>(jbgp.M);
        const float N = nstl::min<float>(jbgp.N, jbgp.oc);
        const float K
                = nstl::min<float>(jbgp.K * jbgp.gemm_batch_size, jbgp.os);
        const float tmul_efficiency = (M / 16) * (N / 16) * (K / 32);
        // TODO: Adjust blocking such that bigger M, N, K are generated.
        if (one_of(true, M <= 8, K <= 8, N < 16, tmul_efficiency <= 2.25))
            return status::unimplemented;
    }

    return status::success;
}

size_t buf_dt_size(data_type_t dt, cpu_isa_t isa) {
    const auto buf_dt = isa == avx512_core_fp16 && dt == data_type::f16
            ? data_type::f32
            : dt;
    return types::data_type_size(buf_dt);
}

status_t jit_brgemm_ip_conf_t::init_conf_base(cpu_isa_t isa,
        const inner_product_desc_t &ipd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);

    using namespace prop_kind;

    int ndims = src_d.ndims();
    if (weights_d.ndims() != ndims || dst_d.ndims() != 2)
        return status::unimplemented;

    auto &jbgp = *this;
    jbgp = zero<decltype(jbgp)>();
    jbgp.ndims = ndims;
    jbgp.isa = isa;
    jbgp.is_amx = is_superset(jbgp.isa, avx512_core_amx);
    jbgp.prop_kind = ipd.prop_kind;
    jbgp.ngroups = 1;
    jbgp.mb = src_d.dims()[0];
    jbgp.os = jbgp.mb;
    jbgp.oc_without_padding = dst_d.dims()[1];
    jbgp.oc = jbgp.oc_without_padding;
    jbgp.ic_without_padding = src_d.dims()[1];
    jbgp.ic = jbgp.ic_without_padding;
    jbgp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jbgp.ih = (ndims < 4) ? 1 : src_d.dims()[ndims - 2];
    jbgp.iw = (ndims < 3) ? 1 : src_d.dims()[ndims - 1];
    jbgp.od = jbgp.oh = jbgp.ow = 1;
    jbgp.kd = (ndims == 5) ? weights_d.dims()[2] : 1;
    jbgp.kh = (ndims < 4) ? 1 : weights_d.dims()[ndims - 2];
    jbgp.kw = (ndims < 3) ? 1 : weights_d.dims()[ndims - 1];
    jbgp.stride_d = jbgp.stride_h = jbgp.stride_w = 1;

    if (!everyone_is(1, jbgp.ow, jbgp.oh, jbgp.od))
        return status::unimplemented;
    if (jbgp.kw != jbgp.iw || jbgp.kh != jbgp.ih || jbgp.kd != jbgp.id)
        return status::unimplemented;
    if (!everyone_is(1, jbgp.kw, jbgp.kh, jbgp.kd))
        return status::unimplemented;

    jbgp.with_bias
            = pick_by_prop_kind(jbgp.prop_kind, ipd.bias_desc.format_kind,
                      format_kind::undef, ipd.diff_bias_desc.format_kind)
            != format_kind::undef;

    jbgp.src_dt = src_d.data_type();
    jbgp.dst_dt = dst_d.data_type();
    jbgp.wei_dt = weights_d.data_type();
    jbgp.bia_dt = jbgp.with_bias
            ? pick_by_prop_kind(jbgp.prop_kind, ipd.bias_desc.data_type,
                    data_type::undef, ipd.diff_bias_desc.data_type)
            : data_type::undef;
    jbgp.req_s8s8_compensation
            = one_of(isa, avx512_core, avx512_core_vnni, avx2_vnni)
            && jbgp.src_dt == s8;
    const bool is_int8 = one_of(jbgp.src_dt, u8, s8) && jbgp.wei_dt == s8;
    const bool is_bf16
            = everyone_is(bf16, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt)
            || pick_by_prop_kind(jbgp.prop_kind,
                    everyone_is(bf16, jbgp.src_dt, jbgp.wei_dt)
                            && jbgp.dst_dt == f32,
                    everyone_is(bf16, jbgp.wei_dt, jbgp.dst_dt)
                            && jbgp.src_dt == f32,
                    everyone_is(bf16, jbgp.src_dt, jbgp.dst_dt)
                            && jbgp.wei_dt == f32);
    const bool is_f16 = everyone_is(f16, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt)
            || pick_by_prop_kind(jbgp.prop_kind,
                    everyone_is(f16, jbgp.src_dt, jbgp.wei_dt)
                            && jbgp.dst_dt == f32,
                    everyone_is(f16, jbgp.wei_dt, jbgp.dst_dt)
                            && jbgp.src_dt == f32,
                    everyone_is(f16, jbgp.src_dt, jbgp.dst_dt)
                            && jbgp.wei_dt == f32);
    const bool is_f32 = everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);
    jbgp.is_bf32
            = is_f32 && attr.fpmath_mode_ == fpmath_mode::bf16 && jbgp.is_amx;

    if (!IMPLICATION(is_int8,
                one_of(isa, avx2_vnni, avx2_vnni_2, avx512_core,
                        avx512_core_vnni, avx512_core_amx)))
        return status::unimplemented;
    if (!IMPLICATION(is_bf16,
                one_of(isa, avx2_vnni_2, avx512_core_bf16, avx512_core_amx)))
        return status::unimplemented;
    if (!IMPLICATION(is_f32, jbgp.is_bf32 || one_of(isa, avx512_core, avx2)))
        return status::unimplemented;
    if (!IMPLICATION(is_f16,
                one_of(isa, avx2_vnni_2, avx512_core_fp16,
                        avx512_core_amx_fp16)))
        return status::unimplemented;

    if (!one_of(true, is_int8, is_bf16, is_f16, is_f32))
        return status::unimplemented;
    if (is_int8) {
        jbgp.acc_dt = s32;
        jbgp.with_scales = true;
        jbgp.with_dst_scales = true;
    } else
        jbgp.acc_dt = f32;

    jbgp.simd_w = isa_max_vlen(jbgp.isa) / types::data_type_size(jbgp.acc_dt);

    // Dispatch small shapes to VNNI for better performance
    const bool is_amx_int8 = jbgp.is_amx && one_of(jbgp.wei_dt, s8, u8);
    const auto amx_row
            = static_cast<int32_t>(data_type_vnni_granularity(jbgp.src_dt))
            * jbgp.simd_w;
    const auto max_size = is_amx_int8 ? 1024 : 512;
    const bool is_small_shapes
            = (jbgp.os <= 16 && jbgp.ic <= amx_row && jbgp.oc <= amx_row)
            || (jbgp.ic <= max_size && jbgp.oc <= max_size && jbgp.mb == 1
                    && jbgp.ic % amx_row != 0);
    if (one_of(jbgp.isa, avx512_core_amx, avx512_core_amx) && is_small_shapes)
        return status::unimplemented;

    auto set_or_check_tags = [&]() -> status_t {
        using namespace format_tag;
        format_tag_t desired_src_tag = pick(ndims - 2, nc, ncw, nchw, ncdhw);
        format_tag_t desired_dst_tag = nc;

        if (src_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(src_md, desired_src_tag));
            jbgp.src_tag = desired_src_tag;
        } else {
            jbgp.src_tag
                    = memory_desc_matches_one_of_tag(src_md, desired_src_tag);
        }

        if (dst_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(dst_md, desired_dst_tag));
            jbgp.dst_tag = desired_dst_tag;
        } else {
            jbgp.dst_tag = memory_desc_matches_one_of_tag(dst_md, nc);
        }

        if (one_of(format_tag::undef, jbgp.src_tag, jbgp.dst_tag))
            return status::unimplemented;

        if (jbgp.with_bias && bias_md.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));

        jbgp.is_wei_layout_any = weights_d.format_kind() == format_kind::any;

        memory_desc_t want_wei_md = weights_md;
        jbgp.wei_tag = get_brgemm_ip_weights_tag(weights_md);
        if (jbgp.wei_tag == format_tag::undef) return status::unimplemented;
        CHECK(memory_desc_init_by_tag(want_wei_md, jbgp.wei_tag));

        if (jbgp.req_s8s8_compensation) {
            want_wei_md.extra.flags
                    |= memory_extra_flags::compensation_conv_s8s8;
            want_wei_md.extra.compensation_mask = (1 << 0);
            if (weights_md.format_kind != format_kind::any
                    && want_wei_md != weights_md)
                return status::unimplemented;
        }
        weights_md = want_wei_md;
        return status::success;
    };

    jbgp.brg_type = brgemm_addr;
    jbgp.nthr = nthreads;

    // Use blocking and kernels that reduce bandwidth requirement for small os
    // sizes.
    // TODO: Evaluate for other precisions, testing only done for f32.
    bool small_os = jbgp.os <= 80;

    // If os is tiny, less than # registers for broadcast in kernel, we only
    // read the weights once in the kernel.
    small_os &= jbgp.os > 6;
    jbgp.use_small_os_kernels = is_f32 && small_os && jbgp.oc % 32 == 0;

    jbgp.use_uker = true;
    jbgp.use_interleave_stores = jbgp.use_uker;
    if (jbgp.use_uker)
        jbgp.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf0;
    CHECK(set_or_check_tags());
    CHECK(attr.set_default_formats(&dst_md));

    return status::success;
}

void jit_brgemm_ip_conf_t::init_scratchpad_base(
        memory_tracking::registrar_t &scratchpad) const {

    auto &jbgp = *this;

    if (jbgp.brg_type == brgemm_addr) {
        size_t sc_size = sizeof(brgemm_batch_element_t);
        size_t n_elems = (size_t)jbgp.nthr * jbgp.adjusted_batch_size;
        scratchpad.book(key_brgemm_primitive_batch, n_elems, sc_size, 64);
    }

    if (jbgp.is_amx)
        scratchpad.book(key_conv_amx_tile_buffer,
                (size_t)jbgp.nthr * jbgp.amx_buf_size_per_thread, sizeof(char));
}

void jit_brgemm_ip_fwd_conf_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    assert(one_of(prop_kind, forward_training, forward_inference));

    init_scratchpad_base(scratchpad);

    const auto &jbgp = *this;

    if (jbgp.use_buffer) {
        size_t nelements = 0;
        size_t nrows = 0;

        // Number of reduction, per thread or shared buffers.
        size_t nbuffers = 0;

        if (jbgp.nthr_ic_b > 1) {
            const bool need_extra_buffer
                    = IMPLICATION(jbgp.dst_dt == jbgp.acc_dt, jbgp.with_sum);
            int n_reduction_buffers = jbgp.nthr_ic_b - !need_extra_buffer;
            nbuffers = n_reduction_buffers;
            nrows = jbgp.os;
        } else {
            switch (jbgp.loop_order) {
                case osc_occ_osb_ocb_icc:
                    nbuffers = jbgp.nthr;
                    nrows = jbgp.M;
                    break;
                case osc_occ_icc_osb_ocb:
                    nbuffers = jbgp.nthr;
                    nrows = jbgp.os_block * nb_os_blocking;
                    break;
                case icc_osc_occ_osb_ocb:
                case icc_occ_osc_ocb_osb:
                    nbuffers = 1;
                    nrows = jbgp.os;
                    break;
            }
        }
        nelements = nbuffers * nrows * jbgp.LDC;
        scratchpad.book(key_brgemm_primitive_buffer, nelements,
                types::data_type_size(jbgp.acc_dt));
    }

    if (jbgp.use_buffer_a) {
        scratchpad.book(key_brgemm_primitive_buffer_a,
                (size_t)jbgp.nthr * jbgp.LDA * jbgp.os_block
                        * jbgp.nb_os_blocking,
                buf_dt_size(jbgp.src_dt, jbgp.isa));
    }
}

void jit_brgemm_ip_bwd_d_conf_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    assert(prop_kind == backward_data);

    init_scratchpad_base(scratchpad);

    const auto &jbgp = *this;

    if (jbgp.use_buffer) {
        size_t nelements = (size_t)jbgp.nthr * jbgp.LDC * jbgp.M;
        if (jbgp.nthr_oc_b > 1) {
            const int adj_buffers = (jbgp.src_dt == f32) ? 1 : 0;
            int n_reduction_buffers = jbgp.nthr_oc_b - adj_buffers;
            nelements = (size_t)n_reduction_buffers * jbgp.LDC * jbgp.os;
        }
        scratchpad.book(key_brgemm_primitive_buffer, nelements,
                types::data_type_size(jbgp.acc_dt));
    }

    if (jbgp.use_buffer_a) {
        scratchpad.book(key_brgemm_primitive_buffer_a,
                (size_t)jbgp.nthr * jbgp.os_block * jbgp.LDA,
                buf_dt_size(jbgp.dst_dt, jbgp.isa));
    }

    if (jbgp.use_buffer_b) {
        auto size_B = (size_t)jbgp.LDB * rnd_up(jbgp.K, 2);

        if (!jbgp.global_b_transpose)
            scratchpad.book(key_brgemm_primitive_buffer_b,
                    (dim_t)jbgp.nthr * jbgp.gemm_batch_size * size_B,
                    buf_dt_size(jbgp.wei_dt, jbgp.isa));
        else
            scratchpad.book(key_brgemm_primitive_buffer_b,
                    (dim_t)jbgp.nb_oc * jbgp.nb_ic * size_B,
                    buf_dt_size(jbgp.wei_dt, jbgp.isa));
    }
}

void jit_brgemm_ip_bwd_w_conf_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    assert(prop_kind == backward_weights);

    init_scratchpad_base(scratchpad);

    const auto &jbgp = *this;

    if (jbgp.use_buffer) {
        size_t nelements = (size_t)jbgp.nthr * jbgp.LDC * jbgp.M;
        if (jbgp.nthr_mb > 1 || jbgp.harness == harness_mb_reduction) {
            const size_t n_reduction_buffers = jbgp.nthr_mb > 1
                    ? jbgp.nthr_mb - (jbgp.wei_dt == f32)
                    : 1;
            const size_t num_ic_chunks
                    = div_up(jbgp.nb_ic, jbgp.nb_ic_blocking);
            const size_t num_oc_chunks
                    = div_up(jbgp.nb_oc, jbgp.nb_oc_blocking);
            nelements = (size_t)n_reduction_buffers * num_ic_chunks
                    * num_oc_chunks * jbgp.nb_ic_blocking * jbgp.nb_oc_blocking
                    * jbgp.ic_block * jbgp.oc_block;
        } else if (jbgp.nthr_mb == 1) {
            nelements = (size_t)jbgp.nthr * jbgp.nb_ic_blocking * jbgp.ic_block
                    * jbgp.nb_oc_blocking * jbgp.oc_block;
        }
        scratchpad.book(key_brgemm_primitive_buffer, nelements,
                types::data_type_size(jbgp.acc_dt));
    }

    if (jbgp.use_buffer_a) {
        const dim_t num_ic_chunks_per_thread
                = jbgp.local_buffers_for_input_tensors
                ? 1
                : div_up(div_up(jbgp.nb_ic, jbgp.nb_ic_blocking),
                        jbgp.nthr_ic_b);
        const dim_t num_os_chunks_per_thread
                = jbgp.local_buffers_for_input_tensors
                ? 1
                : div_up(div_up(jbgp.nb_os, jbgp.nb_os_blocking), jbgp.nthr_mb);
        const dim_t num_elems_per_thread = num_ic_chunks_per_thread
                * num_os_chunks_per_thread * jbgp.gemm_batch_size
                * jbgp.os_block * jbgp.ic_block * jbgp.nb_ic_blocking;
        scratchpad.book(key_brgemm_primitive_buffer_a,
                jbgp.nthr * num_elems_per_thread,
                buf_dt_size(jbgp.src_dt, jbgp.isa));
    }

    if (jbgp.use_buffer_b) {
        int num_os_chunks_per_thread = jbgp.local_buffers_for_input_tensors
                ? 1
                : div_up(div_up(jbgp.nb_os, jbgp.nb_os_blocking), jbgp.nthr_mb);
        const dim_t num_elems_per_thread
                = static_cast<dim_t>(num_os_chunks_per_thread)
                * jbgp.gemm_batch_size * jbgp.os_block * jbgp.LDB;
        scratchpad.book(key_brgemm_primitive_buffer_b,
                (size_t)jbgp.nthr * num_elems_per_thread,
                buf_dt_size(jbgp.dst_dt, jbgp.isa));
    }

    if (jbgp.with_bias && (jbgp.bia_dt != f32 || jbgp.nthr_mb > 1)) {
        int nbuffers = jbgp.nthr_mb - (jbgp.bia_dt == f32);
        scratchpad.book(key_iprod_bias_bf16_convert_wsp,
                (size_t)nbuffers * jbgp.oc, types::data_type_size(jbgp.acc_dt));
    }

    if (dnnl_thr_syncable())
        scratchpad.book<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx, 1);
}

void jit_brgemm_ip_fwd_conf_t::choose_loop_order() {
    const bool is_f32 = everyone_is(f32, src_dt, wei_dt, dst_dt);
    const bool is_f32_compute = is_f32 && !is_bf32;

    // Optimize loop order for f32, if buffer is not required.
    const bool ocb_inner_most = is_f32_compute;
    if (ocb_inner_most) {
        loop_order = osc_occ_icc_osb_ocb;

        // Use icc loop as outer-most to save bandwidth when os is small.
        if (use_small_os_kernels) loop_order = icc_osc_occ_osb_ocb;
    }

    const int nthr_ic = nthr_ic_b <= nthr ? nthr_ic_b : 1;
    const int nthr_oc_mb = nthr / nthr_ic;

    const int os_chunks = div_up(nb_os, nb_os_blocking);
    const int oc_chunks = div_up(nb_oc, nb_oc_blocking);
    const int ic_chunks = div_up(nb_ic, nb_ic_blocking);
    const int work_amount = oc_chunks * os_chunks;

    const int os_chunk_sz = os_block * nb_os_blocking;
    const int oc_chunk_sz = oc_block * nb_oc_blocking;
    const int ic_chunk_sz = ic_block * nb_ic_blocking;
    const int n_blocks = div_up(work_amount, nthr_oc_mb);
    const int n_ic_chunks = div_up(ic_chunks, nthr_ic);

    int oc_span_osc_occ = nstl::min(n_blocks, oc_chunks) * oc_chunk_sz;
    int os_span_osc_occ = div_up(n_blocks, oc_chunks) * os_chunk_sz;
    oc_span_osc_occ = nstl::min(oc_span_osc_occ, oc);
    os_span_osc_occ = nstl::min(os_span_osc_occ, os);

    int os_span_occ_osc = nstl::min(n_blocks, os_chunks) * os_chunk_sz;
    int oc_span_occ_osc = div_up(n_blocks, os_chunks) * oc_chunk_sz;
    os_span_occ_osc = nstl::min(os_span_occ_osc, os);
    oc_span_occ_osc = nstl::min(oc_span_occ_osc, oc);

    int ic_span = nstl::min(n_ic_chunks * ic_chunk_sz, ic);

    auto eff = [](dim_t m, dim_t n, dim_t k) {
        return 2 * m * n * k / float(m * k + n * k + 2 * m * n);
    };

    // Prefer to use occ_osc_... instead of osc_occ_... if compute
    // intensity increases more than a threshold.
    float eff_osc_occ = eff(os_span_osc_occ, oc_span_osc_occ, ic_span);
    float eff_occ_osc = eff(os_span_occ_osc, oc_span_occ_osc, ic_span);
    bool do_occ_osc = eff_occ_osc > 1.15 * eff_osc_occ;

    // Enable occ_osc_... for f32 and with small os-blocks.
    // TODO: Expand to other precisions and other blocks sizes.
    const bool is_avx512 = is_superset(isa, avx512_core);
    if ((os_block < 32 || do_occ_osc) && is_f32_compute && is_avx512)
        loop_order = icc_occ_osc_ocb_osb;
}

void jit_brgemm_ip_bwd_w_conf_t::choose_loop_order() {
    loop_order = local_buffers_for_input_tensors ? osc_icc_occ
            : harness == harness_mb_reduction    ? osc_occ_icc
                                                 : occ_icc_osc;
}

} // namespace brgemm_inner_product_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
