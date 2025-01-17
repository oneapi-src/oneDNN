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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/tag_traits.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/matmul/brgemm_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::cpu::matmul;

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using namespace data_type;

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::pd_t::init(engine_t *engine) {
    const auto src_dt = src_md_.data_type;
    const auto wei_dt = weights_md_.data_type;
    const auto dst_dt = dst_md_.data_type;

    const bool is_f32 = everyone_is(f32, src_dt, wei_dt, dst_dt);
    const bool is_int8 = one_of(src_dt, u8, s8) && wei_dt == s8
            && one_of(dst_dt, u8, s8, s32, f32, f16, bf16);
    const bool is_f8 = one_of(src_dt, f8_e5m2, f8_e4m3)
            && one_of(wei_dt, f8_e5m2, f8_e4m3)
            && one_of(dst_dt, f32, f16, bf16, f8_e5m2, f8_e4m3);
    const bool is_bf16
            = everyone_is(bf16, src_dt, wei_dt) && one_of(dst_dt, bf16, f32);
    const bool is_f16
            = everyone_is(f16, src_dt, wei_dt) && one_of(dst_dt, f16, f32);
    const bool is_f32_f16
            = src_dt == f32 && wei_dt == f16 && one_of(dst_dt, f16, f32);
    const bool is_f32_bf16
            = src_dt == f32 && wei_dt == bf16 && one_of(dst_dt, bf16, f32);
    const bool is_bf16_with_int_wei = src_dt == bf16
            && one_of(wei_dt, s8, u8, s4, u4) && one_of(dst_dt, bf16, f32);
    const bool is_f16_with_int_wei = src_dt == f16
            && one_of(wei_dt, s8, u8, s4, u4) && one_of(dst_dt, f16, f32);

    auto check_bias = [&]() -> bool {
        const auto bia_dt = weights_md(1)->data_type;
        // The cause in IMPLICATION should be an expression to work around
        // ICE in GCC 7.4.
        const bool is_bia_dt_correct
                = IMPLICATION(is_int8 == true,
                          one_of(bia_dt, f32, s32, s8, u8, bf16))
                && IMPLICATION(
                        is_f8 == true, one_of(bia_dt, f32, f16, bf16, src_dt))
                && IMPLICATION(
                        !(is_int8 || is_f8), one_of(bia_dt, f32, src_dt));
        return IMPLICATION(with_bias(), is_bia_dt_correct && is_bias_1xN());
    };

    auto check_reduce = [&]() -> bool {
        if (!with_reduce()) return true;

        bool ok = reduce_kind() == matmul_reduce_kind::src;
        ok = ok && src_md()->ndims == 2;
        ok = ok && one_of(src_dt, f32, bf16, f16);

        const memory_desc_wrapper src_mdw(src_md_);
        ok = ok && !src_mdw.has_runtime_dims();
        ok = ok && src_mdw.matches_tag(format_tag::ba);

        const auto skip_mask = primitive_attr_t::skip_mask_t::fpmath_mode;
        ok = ok && attr()->has_default_values(skip_mask);

        return ok;
    };

    auto check_attr_scales = [&]() -> bool {
        const std::vector<int> supported_args
                = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
        bool ok = attr_scales_ok(supported_args);
        const auto &asc = attr()->scales_;
        if (!asc.has_default_values(DNNL_ARG_SRC)
                && !asc.has_default_values(DNNL_ARG_WEIGHTS)
                && asc.get_mask(DNNL_ARG_WEIGHTS) > 0) {
            // This case requires scratchpad
            if (N() == DNNL_RUNTIME_DIM_VAL) ok = false;
        }
        // Impl suppports f32 scales only for non-weight decompression
        if (!(is_bf16_with_int_wei || is_f16_with_int_wei)) {
            ok = ok && one_of(asc.get_data_type(DNNL_ARG_SRC), undef, f32);
            ok = ok && one_of(asc.get_data_type(DNNL_ARG_WEIGHTS), undef, f32);
            ok = ok && one_of(asc.get_data_type(DNNL_ARG_DST), undef, f32);
        }
        // Implementation has limited support w.r.t. scales groups.
        if (!asc.has_default_values(DNNL_ARG_WEIGHTS)) {
            if (!asc.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                // Only grouping over K is supported.
                ok = ok && asc.get_group(DNNL_ARG_WEIGHTS, 1) == 1;
                // Only 'per_ocic' mask is supported, but not 'per_tensor' in
                // benchdnn terms. In numbers, it's '12' is supported while for
                // 4D '15' is required.
                const int mask = asc.get_mask(DNNL_ARG_WEIGHTS);
                const int ndims = weights_md_.ndims;
                const int last_dim = (1 << (ndims - 1));
                const int prelast_dim = (1 << (ndims - 2));
                const bool mask_ok = (mask & ~(last_dim | prelast_dim)) == 0;
                ok = ok && mask_ok;
            }
        }
        return ok;
    };

    auto check_attr_zero_points = [&]() -> bool {
        const auto &zp = attr()->zero_points_;
        static const std::vector<int> supported_args {
                DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
        for (int arg : supported_args) {
            if (!zp.has_default_values(arg)) {
                const int mask = zp.get_mask(arg);
                if (mask > 0) return false;
            }
        }
        return true;
    };
    const bool problem_dt_correct
            = one_of(true, is_int8, is_f8, is_bf16, is_f32, is_f16, is_f32_f16,
                    is_f32_bf16, is_bf16_with_int_wei, is_f16_with_int_wei);

    auto src_d = memory_desc_wrapper(src_md_);
    auto weights_d = memory_desc_wrapper(weights_md_);
    auto bias_d = memory_desc_wrapper(bias_md_);
    auto dst_d = memory_desc_wrapper(dst_md_);
    const bool is_sparse_ok = is_dense_format_kind()
            || (!src_d.is_sparse_desc() && !bias_d.is_sparse_desc()
                    && !dst_d.is_sparse_desc()
                    && weights_d.is_sparse_packed_desc());
    // Disabling verbose dispatch messages for unsupported isa for better
    // readability.
    if (!mayiuse(isa)) return status::unimplemented;

    VDISPATCH_MATMUL(is_sparse_ok, VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_MATMUL(problem_dt_correct, VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_MATMUL(
            attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::scales_runtime_data_type
                            | primitive_attr_t::skip_mask_t::
                                    scales_runtime_groups
                            | primitive_attr_t::skip_mask_t::
                                    zero_points_runtime_data_type
                            | primitive_attr_t::skip_mask_t::post_ops
                            | primitive_attr_t::skip_mask_t::sum_dt
                            | primitive_attr_t::skip_mask_t::fpmath_mode,
                    dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_MATMUL(attr()->post_ops_.check_sum_consistency(dst_dt, is_int8),
            VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_MATMUL(check_attr_scales(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_MATMUL(check_attr_zero_points(), VERBOSE_UNSUPPORTED_ZP_CFG);
    VDISPATCH_MATMUL(check_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_MATMUL(check_reduce(), VERBOSE_UNSUPPORTED_FEATURE,
            "reduce is not supported");

    CHECK(init_brgemm_matmul_conf(isa, bgmmc_, *desc(), src_md_, weights_md_,
            dst_md_, bias_md_, attr_));

    // f32:f16 configuration on AVX2 doesn't support tails with proper
    // instruction sequence in copy routines. Anchor: F32_F16_AVX2_NO_TAIL.
    VDISPATCH_MATMUL(IMPLICATION((is_f32_f16 || is_f32_bf16) && isa == avx2,
                             bgmmc_.N % 8 == 0),
            "unsupported configuration");

    const float alpha = 1.0;
    const float beta = 1.0;
    const float beta_init = 0.0;

    const int max_m_ker_idx
            = bgmmc_.is_runtime_M ? max_num_dynamic_m_tails + 1 : 2;
    const int max_n_ker_idx
            = bgmmc_.is_runtime_N ? max_num_dynamic_n_tails + 1 : 2;

    const bool is_amx = is_superset(isa, avx512_core_amx);
    const bool is_s8s8 = src_dt == s8 && wei_dt == s8;
    // In the case of dynamic M for amx the last tail kernel generate using
    // non-amx isa. s8s8 proplem type is exception to avoid compensations
    // processing for tail kernel
    const auto backup_isa = is_amx && bgmmc_.is_runtime_M && !is_s8s8
            ? (is_f16 || is_f32_f16 || is_f16_with_int_wei
                            ? avx512_core_fp16
                            : (is_bf16 || is_f32_bf16 || is_bf16_with_int_wei
                                            ? avx512_core_bf16
                                            : (is_int8 ? avx512_core_vnni
                                                       : avx512_core)))
            : isa;

    const int i_bs_end = bgmmc_.brgemm_batch_tail_size ? 2 : 1;
    const int i_init_start = bgmmc_.K_blk != bgmmc_.K ? 0 : 1;
    const int i_K_end = bgmmc_.K_tail ? 2 : 1;

    for_(int i_bs = 0; i_bs < i_bs_end; i_bs++)
    for_(int i_init = i_init_start; i_init < 2; i_init++)
    for_(int i_M = 0; i_M < max_m_ker_idx; i_M++)
    for_(int i_N = 0; i_N < max_n_ker_idx; i_N++)
    for (int i_K = 0; i_K < i_K_end; i_K++) {
        auto vbeta = (i_init) ? beta_init : beta;
        auto vM = (i_M) == 0 ? bgmmc_.M_blk
                             : (bgmmc_.is_runtime_M ? dynamic_m_tails[i_M - 1]
                                                    : bgmmc_.M_tail);
        auto vN = (i_N) == 0 ? bgmmc_.N_blk
                             : (bgmmc_.is_runtime_N ? dynamic_n_tails[i_N - 1]
                                                    : bgmmc_.N_tail);
        auto vK = (i_K) ? bgmmc_.K_tail : bgmmc_.K_blk;

        int bs = get_brg_batchsize(bgmmc_, i_bs, i_K);
        int idx = get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K);
        if (idx < 0) continue;

        brgemm_desc_t &brg = brg_descs_[idx];
        auto LDA = i_K && bgmmc_.use_buffer_a_tail_only
                ? (dim_t)bgmmc_.wei_k_blk
                : bgmmc_.LDA;
        const auto kernel_isa = i_M == max_m_ker_idx - 1 ? backup_isa : isa;
        CHECK(brgemm_desc_init(&brg, kernel_isa, bgmmc_.brg_type, bgmmc_.src_dt,
                bgmmc_.wei_dt, false, false, brgemm_row_major, alpha, vbeta,
                LDA, bgmmc_.LDB, bgmmc_.LDC, vM, vN, vK));

        auto LDD = bgmmc_.LDD;
        if (bgmmc_.with_wei_decompression && bgmmc_.has_zero_point_b)
            brg.skip_zp_b_compensation = true;
        if (bgmmc_.apply_scales_in_buffer_b) brg.skip_scales = true;
        CHECK(brgemm_desc_set_postops(
                &brg, attr(), &dst_md_, LDD, bgmmc_.bia_dt));

        brgemm_attr_t brgattr;
        brgattr.generate_skip_accumulation
                = bgmmc_.post_ops_applicable && bgmmc_.nthr_k > 1;
        if (is_superset(kernel_isa, avx512_core_amx)) {
            brgattr.use_uker = true;
            brgattr.use_interleave_stores = true;
            brgattr.max_bs = bs;
            brgattr.wary_A_k_tail_read = bgmmc_.extendable_k;
            brgattr.extendable_k = bgmmc_.extendable_k;
            // TODO: change expected sizes to local chunks wrt L2 blocking
            brgattr.hint_expected_A_size = vM * vK * bs;
            brgattr.hint_expected_B_size = vN * vK * bs;
            brgattr.hint_expected_C_size = vM * vN * bs;
            brgattr.hint_innermost_loop = brgemm_innermost_undef;
            brgattr.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf0;
        }

        CHECK(brgemm_desc_set_attr(&brg, brgattr));
        CHECK(brgemm_desc_finalize(&brg));

        bgmmc_.wsp_tile_per_thr_bytes = nstl::max(
                brg.get_wsp_buffer_size(), bgmmc_.wsp_tile_per_thr_bytes);
    }

    auto scratchpad = scratchpad_registry().registrar();
    init_scratchpad(scratchpad, bgmmc_);
    const auto wei_scale_count = bgmmc_.is_oscale_per_k
            ? (bgmmc_.is_oscale_per_n ? N() * K() : K())
            : N();
    book_precomputed_scales(scratchpad, attr()->scales_, wei_scale_count);

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::init(engine_t *engine) {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const int max_m_ker_idx
            = bgmmc.is_runtime_M ? max_num_dynamic_m_tails + 1 : 2;
    const int max_n_ker_idx
            = bgmmc.is_runtime_N ? max_num_dynamic_n_tails + 1 : 2;

    const int i_bs_end = bgmmc.brgemm_batch_tail_size ? 2 : 1;
    const int i_init_start = bgmmc.K_blk != bgmmc.K ? 0 : 1;
    const int i_K_end = bgmmc.K_tail ? 2 : 1;

    for_(int i_bs = 0; i_bs < i_bs_end; i_bs++)
    for_(int i_M = 0; i_M < max_m_ker_idx; i_M++)
    for_(int i_N = 0; i_N < max_n_ker_idx; i_N++)
    for_(int i_K = 0; i_K < i_K_end; i_K++)
    for (int i_init = i_init_start; i_init < 2; i_init++) {
        int idx = pd()->get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K);
        if (idx < 0) continue;

        brgemm_kernel_t *ker = nullptr;
        CHECK(brgemm_kernel_create(&ker, pd()->get_brg_desc(idx)));
        CHECK(safe_ptr_assign(brg_kernels_[idx], ker));
        if (is_superset(pd()->get_brg_desc(idx).isa_impl, avx512_core_amx))
            brgemm_palettes_.insert(idx, pd()->get_brg_desc(idx));

        if (pd()->with_reduce()) {
            if (pd()->reduce_kind() == matmul_reduce_kind::src) {
                if (i_N == 0 && i_init == i_init_start) {
                    reducers_[i_M][i_K] = nullptr;
                    auto db_desc = pd()->get_brg_desc(idx);
                    db_desc.reduce_dim = i_K ? bgmmc.K_tail : bgmmc.K_blk;
                    db_desc.load_dim = i_M ? bgmmc.M_tail : bgmmc.M_blk;

                    if (db_desc.reduce_dim > 0 && db_desc.load_dim > 0) {
                        CHECK(safe_ptr_assign(reducers_[i_M][i_K],
                                new reducer_t(bgmmc, db_desc)));
                        CHECK(reducers_[i_M][i_K]->create_kernel());
                    }
                }
            } else {
                assert(!"unsupported reduce kind");
            }
        }
    }

    if (bgmmc.use_buffer_b && !bgmmc.packed_sparse_weights)
        CHECK(create_brgemm_matmul_copy_b(copy_B_kernel_, &bgmmc));

    if (bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only)
        CHECK(create_brgemm_matmul_copy_a(copy_A_kernel_, &bgmmc));

    if (pd()->with_reduce() || (bgmmc.nthr_k > 1 && bgmmc.acc_dt == f32)) {
        CHECK(safe_ptr_assign(
                acc_ker_f32_, new cpu_accumulator_1d_t<data_type::f32>()));
        CHECK(acc_ker_f32_->create_kernel());
    } else if (bgmmc.nthr_k > 1 && bgmmc.acc_dt == s32) {
        CHECK(safe_ptr_assign(
                acc_ker_s32_, new cpu_accumulator_1d_t<data_type::s32>()));
        CHECK(acc_ker_s32_->create_kernel());
    }

    if (bgmmc.packed_sparse_weights) {
        CHECK(safe_ptr_assign(sparse_decompress_kernel_,
                new jit_avx512_sparse_decompress_kernel_t(bgmmc)));
        CHECK(sparse_decompress_kernel_->create_kernel());
    }

    // JIT to precompute scales
    // TODO: enable transpose in JIT scales
    const bool is_jit_supported = mayiuse(avx512_core);
    const auto attr = pd()->attr();
    const auto wei_scale_count = bgmmc.is_oscale_per_k
            ? (bgmmc.is_oscale_per_n ? pd()->N() * pd()->K() : pd()->K())
            : pd()->N();
    if (is_jit_supported && wei_scale_count > 1 && req_copy_scales(attr)
            && !bgmmc.req_transpose_scales) {
        const auto &attr_scales = attr->scales_;
        int wei_scale_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
        if (wei_scale_mask > 0) {
            CHECK(safe_ptr_assign(jit_scale_precompute_,
                    new jit_avx512_core_scale_precompute_t(attr)));
            CHECK(jit_scale_precompute_->create_kernel());
        }
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::execute_body(const exec_ctx_t &ctx) const {
    DEFINE_ZERO_POINT_VALUE(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINT_VALUE(wei_zero_point, DNNL_ARG_WEIGHTS);
    DEFINE_ZERO_POINT_VALUE(dst_zero_point, DNNL_ARG_DST);
    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    matmul_helper_t helper(src_d, weights_d, dst_d);

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const bool has_wei_scales
            = !pd()->attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS);
    const int wei_scale_mask = pd()->attr()->scales_.get_mask(DNNL_ARG_WEIGHTS);
    const bool wei_scale_per_k
            = has_wei_scales && (wei_scale_mask & pd()->wei_qmask_K());
    const bool wei_scale_per_n
            = has_wei_scales && (wei_scale_mask & pd()->wei_qmask_N());
    const float *oscales = scale_utils::precompute_scales(
            ctx.get_scratchpad_grantor(), src_scales, wei_scales, pd()->K(),
            pd()->N(), wei_scale_per_k, wei_scale_per_n, pd()->attr(),
            jit_scale_precompute_.get(), 1.f, bgmmc.req_transpose_scales);

    brg_matmul_exec_ctx_t brgmm_ctx(ctx, pd(), oscales, src_zero_point,
            wei_zero_point, dst_zero_point, dst_scales, helper);

    const bool use_buffer_a
            = bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only;
    const bool is_amx = is_superset(isa, avx512_core_amx);
    const int num_threads = brgmm_ctx.get_num_threads_for_parallelization();

    const int M_chunks = brgmm_ctx.get_M_chunks();
    const int M_chunk_size = brgmm_ctx.get_M_chunk_size();
    const int M_chunk_tail = brgmm_ctx.get_M_chunk_tail();
    const int N_chunks = brgmm_ctx.get_N_chunks();
    const int N_chunk_tail = brgmm_ctx.get_N_chunk_tail();
    parallel(num_threads, [&](const int ithr, const int nthr) {
        const int ithr_bmn = brgmm_ctx.get_thread_idx_for_bmn(ithr);
        const int ithr_k = brgmm_ctx.get_thread_idx_for_k(ithr);
        if (ithr_bmn < 0 || ithr_k < 0) return;
        int start {0}, end {0};
        balance211(brgmm_ctx.get_parallel_work_amount(),
                brgmm_ctx.get_num_threads_for_bmn(), ithr_bmn, start, end);
        int kc_start {0}, kc_end {bgmmc.K_chunks};
        if (brgmm_ctx.parallel_reduction_is_used())
            balance211((int)bgmmc.K_chunks, brgmm_ctx.get_num_threads_for_k(),
                    ithr_k, kc_start, kc_end);

        int prev_ker_idx = -1;
        brgemm_palettes_.maybe_tile_configure(
                is_amx, prev_ker_idx, brgmm_ctx.get_base_brgemm_kernel_idx());

        int b {0}, mc {0}, nc {0};
        nd_iterator_init(start, b, bgmmc.batch, mc, M_chunks, nc, N_chunks);
        int mc_prev = -1;
        int nb_prev = -1;
        int b_prev = -1;
        const char *a_batch_ptr = nullptr;
        const char *b_batch_ptr = nullptr;
        while (start < end) {
            auto m_start = mc * M_chunk_size;
            const bool m_chunk_tail = mc == M_chunks - 1 && M_chunk_tail > 0;
            auto m_end = m_start + (m_chunk_tail ? M_chunk_tail : M_chunk_size);
            auto n_start = nc * bgmmc.N_chunk_size;
            const bool n_chunk_tail = nc == N_chunks - 1 && N_chunk_tail > 0;
            auto n_end = n_start
                    + (n_chunk_tail ? N_chunk_tail : bgmmc.N_chunk_size);
            int kc_prev = -1;
            if (b != b_prev) {
                a_batch_ptr = brgmm_ctx.get_data_A_batch_ptr(b);
                b_batch_ptr = brgmm_ctx.get_data_B_batch_ptr(b);
            }
            for_(int kc = kc_start; kc < kc_end; kc++)
            for (int nb = n_start; nb < n_end; nb++) {
                const bool bcast_across_all_batch_dims
                        = bgmmc.bcast_B_desc.bcast_across_all_batch_dims;
                const bool skip_copy_b
                        = (nb_prev == nb && kc_prev == kc
                                  && (b_prev == b
                                          || bcast_across_all_batch_dims))
                        && !bgmmc.packed_sparse_weights;
                if (bgmmc.use_buffer_b && !skip_copy_b)
                    copy_b_chunk_in_buffer(
                            brgmm_ctx, b_batch_ptr, ithr, b, nb, kc);
                for (int mb = m_start; mb < m_end; mb++) {
                    const bool skip_copy_a = mc_prev == mc && kc_prev == kc
                            && (b_prev == b
                                    || bgmmc.bcast_A_desc
                                               .bcast_across_all_batch_dims);
                    if (use_buffer_a && nb == n_start && !skip_copy_a)
                        copy_a_chunk_in_buffer(
                                brgmm_ctx, a_batch_ptr, ithr, mb, kc);
                    compute_kernel(brgmm_ctx, a_batch_ptr, b_batch_ptr, ithr, b,
                            mb, nb, kc, kc == kc_start, prev_ker_idx);
                }
                kc_prev = kc;
                nb_prev = nb;
            }
            mc_prev = mc;
            b_prev = b;
            ++start;
            nd_iterator_step(b, bgmmc.batch, mc, M_chunks, nc, N_chunks);
        }
        if (is_amx) { amx_tile_release(); }
    });

    maybe_reduce_and_convert_partial_results_A(brgmm_ctx);
    maybe_reduce_partial_results_and_apply_postops(brgmm_ctx);

    return status::success;
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::compute_kernel(
        const brg_matmul_exec_ctx_t &brgmm_ctx, const char *A_data_batch_ptr,
        const char *B_data_batch_ptr, int ithr, int b_idx, int m_blk_idx,
        int n_blk_idx, int k_chunk_idx, bool do_init, int &prev_ker_idx) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const auto addr_batch = brgmm_ctx.get_batch_elem_ptr(ithr);

    const auto wsp_tile = brgmm_ctx.get_tile_workspace(ithr);

    const dim_t n = brgmm_ctx.get_N_idx(n_blk_idx, true);
    const int k_blk_idx = k_chunk_idx * bgmmc.brgemm_batch_size;

    const dim_t M = brgmm_ctx.get_M();
    const dim_t N = brgmm_ctx.get_N();
    const int m_ker_idx = brgmm_ctx.get_M_kernel_idx(m_blk_idx);
    const int n_ker_idx = brgmm_ctx.get_N_kernel_idx(n_blk_idx);
    const bool is_last_K_chunk = brgmm_ctx.is_last_K_chunk(k_chunk_idx);

    const int remaining_k_blks
            = (bgmmc.use_buffer_a ? utils::rnd_up(bgmmc.K, bgmmc.K_blk)
                                  : bgmmc.K)
            - k_chunk_idx * bgmmc.K_chunk_elems;
    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_chunk_idx);
    const bool is_K_tail
            = is_last_K_chunk && (gemm_batch * bgmmc.K_blk) != remaining_k_blks;
    auto is_bs_tail = (gemm_batch != bgmmc.brgemm_batch_size);
    const int brg_ker_idx = pd()->get_brg_kernel_idx(
            is_bs_tail, do_init, m_ker_idx, n_ker_idx, false);
    const auto ptr_bias = brgmm_ctx.get_bias_ptr(n);
    auto ptr_D = brgmm_ctx.get_data_C_ptr(
            b_idx, brgmm_ctx.get_M_idx(m_blk_idx, true), n);
    auto ptr_C = (bgmmc.use_buffer_c)
            ? brgmm_ctx.get_buf_C_ptr(ithr, m_blk_idx, n_blk_idx)
            : ptr_D;

    const auto zp_comp_a
            = brgmm_ctx.get_zp_a_compensation_ptr(ithr, b_idx, n_blk_idx);
    const auto zp_comp_b
            = brgmm_ctx.get_zp_b_compensation_result_ptr(ithr, m_blk_idx);
    const auto zp_c_val_ptr = brgmm_ctx.get_zp_c_val_ptr();
    const auto &post_ops_binary_rhs_arg_vec
            = brgmm_ctx.get_post_ops_binary_rhs_arg_vec();
    const bool post_ops_applicable = bgmmc.post_ops_applicable
            && (brgmm_ctx.get_num_threads_for_k() <= 1 || bgmmc.K_chunks == 1);

    brgemm_dynamic_values_t leading_dimensions(
            bgmmc.LDA, bgmmc.LDB, brgmm_ctx.get_LDC(), brgmm_ctx.get_LDD());

    brgmm_ctx.maybe_backup_dst_values_to_buffer(
            ithr, b_idx, m_blk_idx, n_blk_idx);

    if (gemm_batch > 0 && brg_ker_idx >= 0) {
        const bool is_amx = is_superset(
                pd()->get_brg_desc(brg_ker_idx).isa_impl, avx512_core_amx);
        const auto brg_kernel = brg_kernels_[brg_ker_idx].get();
        assert(brg_kernel != nullptr);
        brgemm_palettes_.maybe_tile_configure(
                is_amx, prev_ker_idx, brg_ker_idx);

        brgmm_ctx.init_brgemm_batch_elements_values(ithr, 0, gemm_batch,
                A_data_batch_ptr, B_data_batch_ptr, b_idx, m_blk_idx, k_blk_idx,
                n_blk_idx);

        if (post_ops_applicable && is_last_K_chunk && !is_K_tail) {
            void *scratch = is_amx
                    ? static_cast<void *>(wsp_tile)
                    : static_cast<void *>(brgmm_ctx.get_s8s8_comp_ptr(
                            ithr, b_idx, n_blk_idx));

            const size_t dst_row_logical_off
                    = brgmm_ctx.get_M_idx(m_blk_idx, true);
            const size_t batch_first_dim_idx = bgmmc.batch_ndims > 1
                    ? b_idx / bgmmc.batch_without_first_dim
                    : 0;
            const size_t first_mb_matrix_addr_off
                    = batch_first_dim_idx * (M * N)
                    + (dst_row_logical_off * N + n);
            const char *dst_anchor_point = brgmm_ctx.get_data_C_ptr(0, 0, 0);
            const brgemm_post_ops_data_t post_ops_data {
                    static_cast<const void *>(ptr_bias),
                    brgmm_ctx.get_oscales_ptr(n),
                    post_ops_binary_rhs_arg_vec.data(), static_cast<size_t>(n),
                    dst_row_logical_off, dst_anchor_point,
                    first_mb_matrix_addr_off,
                    static_cast<const void *>(zp_comp_a),
                    static_cast<const void *>(zp_comp_b),
                    static_cast<const void *>(zp_c_val_ptr), false, 1, false,
                    false, brgmm_ctx.get_dst_scales_ptr()};
            brgemm_kernel_execute_postops(brg_kernel, gemm_batch, addr_batch,
                    (void *)ptr_C, (void *)ptr_D, post_ops_data, scratch,
                    &leading_dimensions);
        } else {
            brgemm_kernel_execute(brg_kernel, gemm_batch, addr_batch,
                    (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr,
                    &leading_dimensions);
        }

        maybe_reduce_A(brgmm_ctx, ithr, gemm_batch, m_blk_idx, n_blk_idx,
                k_chunk_idx, do_init, is_K_tail, /* do_K_tail */ false);
    }
    if (is_K_tail) {
        brgmm_ctx.init_brgemm_batch_elements_values(ithr, gemm_batch, 1,
                A_data_batch_ptr, B_data_batch_ptr, b_idx, m_blk_idx, k_blk_idx,
                n_blk_idx);

        const bool use_init_ker = (do_init && gemm_batch == 0);
        const int brg_ker_idx = pd()->get_brg_kernel_idx(
                false, use_init_ker, m_ker_idx, n_ker_idx, true);
        if (brg_ker_idx < 0) {
            assert(!"Requested brgemm kernel was not created.");
            return;
        }
        const bool is_amx = is_superset(
                pd()->get_brg_desc(brg_ker_idx).isa_impl, avx512_core_amx);
        brgemm_palettes_.maybe_tile_configure(
                is_amx, prev_ker_idx, brg_ker_idx);
        const auto brg_kernel_k_tail = brg_kernels_[brg_ker_idx].get();

        if (post_ops_applicable) {
            void *scratch = is_amx
                    ? static_cast<void *>(wsp_tile)
                    : static_cast<void *>(brgmm_ctx.get_s8s8_comp_ptr(
                            ithr, b_idx, n_blk_idx));

            const size_t dst_row_logical_off
                    = brgmm_ctx.get_M_idx(m_blk_idx, true);
            const size_t batch_first_dim_idx = bgmmc.batch_ndims > 1
                    ? b_idx / bgmmc.batch_without_first_dim
                    : 0;
            const size_t first_mb_matrix_addr_off
                    = batch_first_dim_idx * (M * N)
                    + (dst_row_logical_off * N + n);
            const char *dst_anchor_point = brgmm_ctx.get_data_C_ptr(0, 0, 0);
            const brgemm_post_ops_data_t post_ops_data {
                    static_cast<const void *>(ptr_bias),
                    brgmm_ctx.get_oscales_ptr(n),
                    post_ops_binary_rhs_arg_vec.data(), static_cast<size_t>(n),
                    dst_row_logical_off, dst_anchor_point,
                    first_mb_matrix_addr_off,
                    static_cast<const void *>(zp_comp_a),
                    static_cast<const void *>(zp_comp_b),
                    static_cast<const void *>(zp_c_val_ptr), false, 1, false,
                    false, brgmm_ctx.get_dst_scales_ptr()};

            brgemm_kernel_execute_postops(brg_kernel_k_tail, 1, addr_batch,
                    (void *)ptr_C, (void *)ptr_D, post_ops_data, scratch,
                    &leading_dimensions);
        } else {
            brgemm_kernel_execute(brg_kernel_k_tail, 1, addr_batch,
                    (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr,
                    &leading_dimensions);
        }

        maybe_reduce_A(brgmm_ctx, ithr, gemm_batch, m_blk_idx, n_blk_idx,
                k_chunk_idx, do_init, is_K_tail,
                /* do_K_tail */ true);
    }

    brgmm_ctx.maybe_restore_dst_values_from_buffer(
            ithr, b_idx, m_blk_idx, n_blk_idx);
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::maybe_reduce_A(
        const brg_matmul_exec_ctx_t &brgmm_ctx, int ithr, int gemm_batch,
        int m_blk_idx, int n_blk_idx, int k_chunk_idx, bool do_init,
        bool has_K_tail, bool do_K_tail) const {

    if (!pd()->with_reduce()) return;

    const bool reduce_a = pd()->reduce_kind() == matmul_reduce_kind::src;
    // Only `matmul_reduce_kind::src` is supported for now.
    assert(reduce_a);

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const auto *addr_batch = brgmm_ctx.get_batch_elem_ptr(ithr);

    if (reduce_a && n_blk_idx == 0) {
        const dim_t m = brgmm_ctx.get_M_idx(m_blk_idx, true);

        auto *reduce_ptr = bgmmc.use_buffer_reduce
                ? brgmm_ctx.get_buf_reduce_ptr(ithr, m)
                : brgmm_ctx.get_data_reduce_ptr(m);

        brgemm_kernel_diff_bias_t p;

        p.ptr_diff_bias_acc = (void *)reduce_ptr;
        p.ptr_diff_bias = (void *)brgmm_ctx.get_data_reduce_ptr(m);

        const int m_ker_idx = brgmm_ctx.get_M_kernel_idx(m_blk_idx);

        if (!do_K_tail) {
            for (int gb = 0; gb < gemm_batch; gb++) {
                p.ptr_diff_dst = (void *)addr_batch[gb].ptr.A;

                const bool is_first = do_init && gb == 0;
                const bool is_last = (bgmmc.nthr_k == 1 || bgmmc.K_chunks == 1)
                        && k_chunk_idx == bgmmc.K_chunks - 1
                        && gb == gemm_batch - 1 && !has_K_tail;

                p.flags = 0 | (is_first ? FLAG_REDUCE_FIRST : 0)
                        | (is_last ? FLAG_REDUCE_LAST : 0);

                (*reducers_[m_ker_idx][do_K_tail])(&p);
            }
        } else {
            p.ptr_diff_dst = (void *)addr_batch[0].ptr.A;

            const bool is_first = do_init && gemm_batch == 0;
            const bool is_last = (bgmmc.nthr_k == 1 || bgmmc.K_chunks == 1)
                    && k_chunk_idx == bgmmc.K_chunks - 1;

            p.flags = 0 | (is_first ? FLAG_REDUCE_FIRST : 0)
                    | (is_last ? FLAG_REDUCE_LAST : 0);

            (*reducers_[m_ker_idx][do_K_tail])(&p);
        }
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::maybe_reduce_and_convert_partial_results_A(
        const brg_matmul_exec_ctx_t &brgmm_ctx) const {
    // Partial results appear when parallel reduction is used.
    //
    // There are two cases that require slightly different handling.
    // - (Figure 1): when reduce data type is not f32. In this case there are
    //   three steps:
    //     * Step 1: add partial results from all reduce buffers except the last
    //       one to the first reduce buffer (reduce_buf_0).
    //     * Step 2 and step 3: add partial results from the first and the last
    //       reduce buffers, convert the result to the reduce data type and
    //       store it to the user provided reduce buffer.
    //
    // - (Figure 2): when reduce data type is f32. In this case the user
    //   provided reduce buffer is used as one of the reduce buffers and there
    //   is only 1 step:
    //     * Step 1: add partial results from all reduce buffers to the user
    //     provided reduce buffer.
    //       buffer.
    //
    //                    Figure 1.
    //             +--------------------+
    //             | reduce (bf16/f16)  |<------+ Step 3.
    //             +--------------------+       |
    //             +--------------------+       |
    //         +-->| reduce_buf_0 (f32) |--->   |
    // Step 1. |   +--------------------+   |   |
    //         |   +--------------------+   |   |
    //         +<--| reduce_buf_1 (f32) |   +---> Step 2.
    //             +--------------------+   |
    //             +--------------------+   |
    //             | reduce_buf_2 (f32) |--->
    //             +--------------------+
    //
    //                    Figure 2.
    //             +--------------------+
    //         +-->|    reduce (f32)    |
    //         |   +--------------------+
    //         |   +--------------------+
    // Step 1. +<--| reduce_buf_0 (f32) |
    //         |   +--------------------+
    //         |   +--------------------+
    //         +<--| reduce_buf_1 (f32) |
    //             +--------------------+

    if (!pd()->with_reduce() || !brgmm_ctx.parallel_reduction_is_used()) return;

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const int num_threads = brgmm_ctx.get_num_threads_for_parallelization();

    parallel(num_threads, [&](const int ithr, const int nthr) {
        const int ithr_bmn = brgmm_ctx.get_thread_idx_for_bmn(ithr);
        const int ithr_k = brgmm_ctx.get_thread_idx_for_k(ithr);
        if (ithr_bmn < 0 || ithr_k < 0) return;

        const int M_chunks = brgmm_ctx.get_M_chunks();

        int start_mc {0}, end_mc {0};
        balance211(M_chunks, brgmm_ctx.get_num_threads_for_bmn(), ithr_bmn,
                start_mc, end_mc);
        if (start_mc != end_mc && ithr_k == 0) {
            const size_t m = start_mc * bgmmc.M_chunk_elems;
            const size_t mc_work = end_mc - start_mc;
            const size_t acc_size
                    = std::min(mc_work * bgmmc.M_chunk_elems, bgmmc.M - m);

            const bool is_reduce_f32 = bgmmc.reduce_dt == f32;

            float *reduce_acc = is_reduce_f32
                    ? (float *)brgmm_ctx.get_data_reduce_ptr(m)
                    : (float *)brgmm_ctx.get_buf_reduce_ptr_by_index(0, m);

            int ibuf = !is_reduce_f32;
            for (; ibuf < bgmmc.nthr_k - 1; ibuf++) {
                float *reduce_buf
                        = (float *)brgmm_ctx.get_buf_reduce_ptr_by_index(
                                ibuf, m);
                acc_ker_f32_->accumulate(reduce_acc, reduce_buf, acc_size);
            }

            if (!is_reduce_f32) {
                float *reduce_buf
                        = (float *)brgmm_ctx.get_buf_reduce_ptr_by_index(
                                ibuf, m);
                switch (bgmmc.reduce_dt) {
                    case data_type::bf16:
                        add_floats_and_cvt_to_bfloat16(
                                (bfloat16_t *)brgmm_ctx.get_data_reduce_ptr(m),
                                reduce_acc, reduce_buf, acc_size);
                        break;
                    case data_type::f16:
                        add_floats_and_cvt_to_float16(
                                (float16_t *)brgmm_ctx.get_data_reduce_ptr(m),
                                reduce_acc, reduce_buf, acc_size);
                        break;
                    default: assert(!"invalid data type");
                }
            }
        }
    });
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::maybe_reduce_partial_results_and_apply_postops(
        const brg_matmul_exec_ctx_t &brgmm_ctx) const {
    if (!brgmm_ctx.parallel_reduction_is_used()) return;

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const int num_threads = brgmm_ctx.get_num_threads_for_parallelization();
    brgemm_dynamic_values_t leading_dimensions(
            bgmmc.LDA, bgmmc.LDB, brgmm_ctx.get_LDC(), brgmm_ctx.get_LDD());

    parallel(num_threads, [&](const int ithr, const int nthr) {
        const int nthr_k = brgmm_ctx.get_num_threads_for_k();
        const int ithr_bmn = brgmm_ctx.get_thread_idx_for_bmn(ithr);
        const int ithr_k = brgmm_ctx.get_thread_idx_for_k(ithr);
        if (ithr_bmn < 0 || ithr_k < 0) return;

        const int num_reduction_buffers = nstl::min(nthr_k, bgmmc.K_chunks);

        int bmn_start {0}, bmn_end {0};
        int start {0}, end {0};
        balance211(brgmm_ctx.get_parallel_work_amount(),
                brgmm_ctx.get_num_threads_for_bmn(), ithr_bmn, bmn_start,
                bmn_end);
        balance211(bmn_end - bmn_start, nthr_k, ithr_k, start, end);

        int prev_ker_idx = -1;

        int b {0}, mc {0}, nc {0};

        const dim_t M = brgmm_ctx.get_M();
        const int M_chunks = brgmm_ctx.get_M_chunks();
        const int M_chunk_size = brgmm_ctx.get_M_chunk_size();
        const int M_chunk_tail = brgmm_ctx.get_M_chunk_tail();
        const int N_chunks = brgmm_ctx.get_N_chunks();
        const int N_chunk_tail = brgmm_ctx.get_N_chunk_tail();
        const int N_chunk_tail_elems = brgmm_ctx.get_N_chunk_tail_elems();

        assert(bgmmc.batch == 1);
        nd_iterator_init(
                bmn_start + start, b, bgmmc.batch, mc, M_chunks, nc, N_chunks);
        while (start < end) {
            auto mb_start = mc * M_chunk_size;
            const bool m_chunk_tail = mc == M_chunks - 1 && M_chunk_tail > 0;
            auto mb_end
                    = mb_start + (m_chunk_tail ? M_chunk_tail : M_chunk_size);
            auto nb_start = nc * bgmmc.N_chunk_size;
            const bool n_chunk_tail = nc == N_chunks - 1 && N_chunk_tail > 0;
            auto nb_end = nb_start
                    + (n_chunk_tail ? N_chunk_tail : bgmmc.N_chunk_size);
            const bool n_chunk_has_tail
                    = nc == N_chunks - 1 && N_chunk_tail_elems > 0;
            const int curr_N_chunk_elems = n_chunk_has_tail
                    ? N_chunk_tail_elems
                    : bgmmc.N_chunk_elems;
            for (int mb = mb_start; mb < mb_end; mb++) {
                const int curr_M_blk = brgmm_ctx.get_M_kernel_size(mb);
                const int m_ker_idx = brgmm_ctx.get_M_kernel_idx(mb);
                char *buf_reduced_base = brgmm_ctx.get_buf_C_par_reduction_ptr(
                        0, mb, nb_start);
                const size_t m_offset = bgmmc.LDC * bgmmc.acc_dt_sz;
                for (int r = 1; r < num_reduction_buffers; r++) {
                    const char *buf_to_reduce_base
                            = brgmm_ctx.get_buf_C_par_reduction_ptr(
                                    r, mb, nb_start);
                    for (int m = 0; m < curr_M_blk; m++) {
                        accumulate(buf_reduced_base + m * m_offset,
                                buf_to_reduce_base + m * m_offset,
                                curr_N_chunk_elems);
                    }
                }
                if (bgmmc.post_ops_applicable) {
                    for (int nb = nb_start; nb < nb_end; nb++) {
                        const int n_ker_idx = brgmm_ctx.get_N_kernel_idx(nb);
                        const int brg_ker_idx = pd()->get_brg_kernel_idx(
                                false, false, m_ker_idx, n_ker_idx, false);
                        if (brg_ker_idx == -1) {
                            assert(!"Requested brgemm kernel was not created.");
                            return;
                        }
                        const bool is_amx = is_superset(
                                pd()->get_brg_desc(brg_ker_idx).isa_impl,
                                avx512_core_amx);
                        brgemm_palettes_.maybe_tile_configure(
                                is_amx, prev_ker_idx, brg_ker_idx);
                        const auto brg_kernel = brg_kernels_[brg_ker_idx].get();
                        const int m = brgmm_ctx.get_M_idx(mb);
                        const int n = nb * bgmmc.N_blk;
                        const auto ptr_bias = brgmm_ctx.get_bias_ptr(n);
                        auto ptr_D = brgmm_ctx.get_data_C_ptr(b, m, n);
                        auto ptr_C = brgmm_ctx.get_buf_C_par_reduction_ptr(
                                0, mb, nb);

                        // TODO: support reduction for zp/s8s8 compensations
                        // computed in copy routines
                        const auto zp_comp_a
                                = brgmm_ctx.get_zp_a_compensation_ptr(
                                        ithr, b, nb);
                        const auto zp_comp_b
                                = brgmm_ctx.get_zp_b_compensation_result_ptr(
                                        ithr, mb);
                        const auto zp_c_val_ptr = brgmm_ctx.get_zp_c_val_ptr();
                        const auto &post_ops_binary_rhs_arg_vec
                                = brgmm_ctx.get_post_ops_binary_rhs_arg_vec();

                        const size_t dst_row_logical_off
                                = brgmm_ctx.get_M_idx(mb, true);
                        const size_t batch_first_dim_idx = bgmmc.batch_ndims > 1
                                ? b / bgmmc.batch_without_first_dim
                                : 0;
                        const size_t first_mb_matrix_addr_off
                                = batch_first_dim_idx * (M * bgmmc.N)
                                + (m * bgmmc.N + n);
                        // apply post-ops and convert to dst data type only
                        constexpr bool skip_accumulation = true;
                        const char *dst_anchor_point
                                = brgmm_ctx.get_data_C_ptr(0, 0, 0);
                        const brgemm_post_ops_data_t post_ops_data {
                                static_cast<const void *>(ptr_bias),
                                brgmm_ctx.get_oscales_ptr(n),
                                post_ops_binary_rhs_arg_vec.data(),
                                static_cast<size_t>(n), dst_row_logical_off,
                                dst_anchor_point, first_mb_matrix_addr_off,
                                static_cast<const void *>(zp_comp_a),
                                static_cast<const void *>(zp_comp_b),
                                static_cast<const void *>(zp_c_val_ptr),
                                skip_accumulation, 1, false, false,
                                brgmm_ctx.get_dst_scales_ptr()};

                        brgemm_kernel_execute_postops(brg_kernel, 0, nullptr,
                                (void *)ptr_C, (void *)ptr_D, post_ops_data,
                                nullptr, &leading_dimensions);
                    }
                }
            }
            ++start;
            nd_iterator_step(b, bgmmc.batch, mc, M_chunks, nc, N_chunks);
        }
    });
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::copy_a_chunk_in_buffer(
        const brg_matmul_exec_ctx_t &brgmm_ctx, const char *A_data_batch_ptr,
        int ithr, int m_blk_idx, int k_chunk_idx) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();

    auto ctx = jit_brgemm_matmul_copy_a_t::ctx_t();
    const int k_start = k_chunk_idx * bgmmc.K_chunk_elems;
    const bool is_K_tail
            = brgmm_ctx.is_last_K_chunk(k_chunk_idx) && bgmmc.K_tail > 0;
    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_chunk_idx);
    const int gemm_batch_iters = bgmmc.use_buffer_a_tail_only ? 0 : gemm_batch;

    const dim_t m = brgmm_ctx.get_M_idx(m_blk_idx, true);

    ctx.current_M_blk = brgmm_ctx.get_M_kernel_size(m_blk_idx);
    ctx.zp_b_compensation_buffer_ptr
            = (void *)brgmm_ctx.get_zp_b_compensation_buffer_ptr(
                    ithr, m_blk_idx);
    ctx.zp_a_compensation_result_ptr
            = (void *)brgmm_ctx.get_zp_b_compensation_result_ptr(
                    ithr, m_blk_idx);
    ctx.zp_b_neg_value_ptr = (void *)brgmm_ctx.get_zp_b_neg_val_ptr();
    ctx.zp_ab_comp_ptr = (void *)brgmm_ctx.get_zp_ab_mixed_comp_ptr();
    ctx.dynamic_src_ld = brgmm_ctx.get_src_stride();

    for (int gb = 0; gb < gemm_batch_iters; gb++) {
        const int k = k_start + gb * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_A_mk_ptr(A_data_batch_ptr, m, k);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_A_ptr(ithr, m_blk_idx, gb);
        ctx.current_K_blk = nstl::min(bgmmc.K_blk, bgmmc.K);
        ctx.current_K_start = k;

        (*copy_A_kernel_)(&ctx);
    }
    if (is_K_tail) {
        const auto K_tail = bgmmc.K % bgmmc.K_blk;
        const int k = k_start + gemm_batch * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_A_mk_ptr(A_data_batch_ptr, m, k);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_A_ptr(
                ithr, m_blk_idx, gemm_batch_iters);
        ctx.current_K_blk = K_tail;
        ctx.current_K_start = k;

        (*copy_A_kernel_)(&ctx);
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::copy_b_chunk_in_buffer(
        const brg_matmul_exec_ctx_t &brgmm_ctx, const char *B_data_batch_ptr,
        int ithr, int b_idx, int n_blk_idx, int k_chunk_idx) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();

    const int k_start = k_chunk_idx * bgmmc.K_chunk_elems;
    const bool is_K_tail
            = brgmm_ctx.is_last_K_chunk(k_chunk_idx) && bgmmc.K_tail > 0;
    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_chunk_idx);

    const dim_t n = brgmm_ctx.get_N_idx(n_blk_idx, true);

    if (brgmm_ctx.packed_sparse_weights()) {
        for (int gb = 0; gb < gemm_batch + is_K_tail; gb++) {
            const int k = k_start + gb * bgmmc.K_blk;
            auto p = jit_avx512_sparse_decompress_kernel_t::call_params_t();
            const char *B_data_ptr
                    = brgmm_ctx.get_data_B_kn_ptr(B_data_batch_ptr, k, n);
            p.src_ptr = (void *)B_data_ptr;
            p.bitmask_ptr
                    = (void *)brgmm_ctx.get_data_B_bitmask_ptr(b_idx, k, n);
            p.dst_ptr = (void *)brgmm_ctx.get_buf_B_ptr(ithr, gb, n_blk_idx);
            (*sparse_decompress_kernel_)(&p);
        }
        return;
    }

    auto ctx = jit_brgemm_matmul_copy_b_t::ctx_t();
    ctx.current_N_blk = brgmm_ctx.get_N_kernel_size(n_blk_idx);

    ctx.zp_a_compensation_ptr = (void *)brgmm_ctx.get_zp_a_compensation_ptr(
            ithr, b_idx, n_blk_idx);
    ctx.zp_a_neg_value_ptr = (void *)brgmm_ctx.get_zp_a_neg_val_ptr();
    ctx.zp_b_value_ptr = (void *)brgmm_ctx.get_zp_b_val_ptr();
    ctx.dynamic_src_stride = brgmm_ctx.copy_B_wei_stride();

    int gb = 0;
    for (; gb < gemm_batch; gb++) {
        const int k = k_start + gb * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_B_kn_ptr(B_data_batch_ptr, k, n);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_B_ptr(ithr, gb, n_blk_idx);
        ctx.compensation_ptr
                = (void *)brgmm_ctx.get_s8s8_comp_ptr(ithr, b_idx, n_blk_idx);
        ctx.current_K_start = k;
        ctx.current_K_iters = nstl::min(bgmmc.K_blk, bgmmc.K);
        ctx.current_K_pad = brgmm_ctx.get_current_K_pad(ctx.current_K_iters);

        ctx.scales_ptr = (void *)brgmm_ctx.get_oscales_ptr(n, k);
        if (bgmmc.blocked_B && !bgmmc.is_f16_with_int_wei
                && isa == avx512_core_fp16) {
            cvt_float16_to_float((float *)ctx.tr_src, (float16_t *)ctx.src,
                    bgmmc.wei_n_blk * ctx.current_K_iters);
        } else {
            (*copy_B_kernel_)(&ctx);
        }
    }

    if (is_K_tail) {
        const int k = k_start + gb * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_B_kn_ptr(B_data_batch_ptr, k, n);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_B_ptr(ithr, gb, n_blk_idx);
        ctx.compensation_ptr
                = (void *)brgmm_ctx.get_s8s8_comp_ptr(ithr, b_idx, n_blk_idx);
        ctx.current_K_start = k;
        ctx.current_K_iters = bgmmc.K % bgmmc.K_blk;
        ctx.current_K_pad = brgmm_ctx.get_current_K_pad(ctx.current_K_iters);
        ctx.scales_ptr = (void *)brgmm_ctx.get_oscales_ptr(n, k);
        if (bgmmc.blocked_B && !bgmmc.is_f16_with_int_wei
                && isa == avx512_core_fp16) {
            cvt_float16_to_float((float *)ctx.tr_src, (float16_t *)ctx.src,
                    bgmmc.wei_n_blk * ctx.current_K_iters);
        } else {
            (*copy_B_kernel_)(&ctx);
        }
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::accumulate(
        char *result_ptr, const char *reduce_ptr, size_t size) const {
    if (pd()->get_brgemm_matmul_conf().acc_dt == f32)
        acc_ker_f32_->accumulate(
                (float *)result_ptr, (const float *)reduce_ptr, size);
    else if (pd()->get_brgemm_matmul_conf().acc_dt == s32)
        acc_ker_s32_->accumulate(
                (int32_t *)result_ptr, (const int32_t *)reduce_ptr, size);
    else
        assert(!"unsupported accumulation data type");
}

template <cpu_isa_t isa>
struct brgemm_matmul_t<isa>::brg_matmul_exec_ctx_t {
    brg_matmul_exec_ctx_t(const exec_ctx_t &ctx, const pd_t *pd,
            const float *oscales, int32_t src_zp, int32_t wei_zp,
            int32_t dst_zp, const float *dst_scales, matmul_helper_t &helper)
        : bgmmc_(pd->get_brgemm_matmul_conf())
        , src_d_(pd->src_md())
        , wei_d_(pd->weights_md())
        , dst_d_(pd->dst_md())
        , data_A_ptr_(CTX_IN_MEM(const char *, DNNL_ARG_SRC))
        , data_B_ptr_(CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS))
        , data_C_ptr_(CTX_OUT_MEM(char *, DNNL_ARG_DST))
        , data_reduce_ptr_(CTX_OUT_MEM(char *, DNNL_ARG_REDUCE)) {

        const memory_desc_wrapper weights_d(pd->weights_md(0));
        if (bgmmc_.packed_sparse_weights) {
            data_B_offsets_ptr_
                    = CTX_IN_MEM(const int64_t *, DNNL_ARG_WEIGHTS, 1);
            data_B_bitmask_ptr_ = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS, 2);
            B_packed_sparse_block_size_ = weights_d.blk_size();
        }

        bias_ptr_ = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);

        oscales_ptr_ = oscales;
        dst_scales_ptr_ = dst_scales;
        memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();
        const auto &bgmmc = pd->get_brgemm_matmul_conf();

        batch_element_ptr_ = scratchpad.template get<brgemm_batch_element_t>(
                key_brgemm_primitive_batch);

        const bool use_buffer_a
                = bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only;
        buf_A_ptr_ = (use_buffer_a)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_a)
                : nullptr;

        buf_B_ptr_ = (bgmmc.use_buffer_b)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_b)
                : nullptr;

        buf_C_ptr_ = (bgmmc.use_buffer_c)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
                : nullptr;

        buf_D_ptr_ = (bgmmc.is_runtime_M || bgmmc.is_runtime_N)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_d)
                : nullptr;

        buf_reduce_ptr_ = bgmmc.use_buffer_reduce
                ? scratchpad.template get<char>(
                        key_brgemm_primitive_buffer_reduce)
                : nullptr;

        is_amx_ = is_superset(isa, avx512_core_amx);
        wsp_tile_ptr_ = is_amx_
                ? ctx.get_scratchpad_grantor().template get<char>(
                        key_conv_amx_tile_buffer)
                : nullptr;

        const dim_t comp_offset = bgmmc_.b_dt_sz
                * (weights_d.size() - weights_d.additional_buffer_size());
        s8s8_compensation_ptr_ = (bgmmc.s8s8_compensation_required)
                ? ((bgmmc.use_buffer_b)
                                ? scratchpad.template get<int32_t>(
                                        key_brgemm_primitive_buffer_comp)
                                : const_cast<int32_t *>(
                                        reinterpret_cast<const int32_t *>(
                                                &data_B_ptr_[comp_offset])))
                : nullptr;
        assert(IMPLICATION(bgmmc.s8s8_compensation_required,
                bgmmc_.b_dt_sz == bgmmc_.tr_b_dt_sz));

        zero_point_a_compensations_ptr_ = bgmmc.has_zero_point_a
                ? scratchpad.template get<int32_t>(
                        key_brgemm_primitive_zp_comp_a)
                : nullptr;
        zero_point_b_compensations_ptr_ = bgmmc.has_zero_point_b
                ? scratchpad.template get<int32_t>(
                        key_brgemm_primitive_zp_comp_b)
                : nullptr;

        zero_point_a_negative_val_ = -src_zp;
        zero_point_b_val_ = wei_zp;
        zero_point_b_negative_val_ = -wei_zp;
        zero_point_mixed_ab_compensation_component_
                = bgmmc.K * zero_point_a_negative_val_;

        zero_point_c_val_ = dst_zp;

        post_ops_binary_rhs_arg_vec_ = binary_injector::prepare_binary_args(
                pd->attr()->post_ops_, ctx);
        base_brg_ker_idx_ = pd->get_brg_kernel_idx(false, true, 0, 0, false);
        vnni_factor = data_type_vnni_granularity(bgmmc.wei_dt);

        reorder_zp_a_comp_ptr_ = nullptr;
        if (bgmmc_.has_zero_point_a && bgmmc_.blocked_B) {
            // Store the pointer to computed in reorder compensation values to
            // scale them locally by zp_a value just before usage in post-ops.
            // Using the single global scaling before parallel section might
            // produce significant overhead for small problems running in
            // multitreaded execution mode
            const size_t reorder_zp_a_comp_offset
                    = weights_d.size() - weights_d.additional_buffer_size();
            const size_t b_batch
                    = get_bb_idx(bgmmc.batch - 1, bgmmc_.bcast_B_desc) + 1;
            const size_t s8s8_buffer_sz = bgmmc.s8s8_compensation_required
                    ? sizeof(int32_t) * b_batch * bgmmc.s8s8_comp_b_str
                    : 0;
            reorder_zp_a_comp_ptr_
                    = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(
                            &data_B_ptr_[reorder_zp_a_comp_offset
                                    + s8s8_buffer_sz]));
        }

        // Set last_chunk_brgemm_batch_size_ to brgemm_batch_size
        // when K_tail = 0 and brgemm_batch_tail_size = 0
        last_chunk_brgemm_batch_size_ = bgmmc.brgemm_batch_tail_size;
        if (bgmmc.K_tail == 0 && last_chunk_brgemm_batch_size_ == 0)
            last_chunk_brgemm_batch_size_ = bgmmc.brgemm_batch_size;

        LDD_ = is_runtime_value(bgmmc_.LDD) ? helper.ldc() : bgmmc_.LDD;
        LDC_ = is_runtime_value(bgmmc_.LDC) ? LDD_ : bgmmc_.LDC;
        copy_A_src_stride_ = bgmmc.copy_A_src_stride;
        is_A_batch_layout_trivial_ = bgmmc_.is_src_batch_layout_trivial;
        is_B_batch_layout_trivial_ = bgmmc_.is_wei_batch_layout_trivial;
        is_C_batch_layout_trivial_ = bgmmc_.is_dst_batch_layout_trivial;
        if (bgmmc.is_runtime_M) {
            M_ = helper.M();
            M_chunks_ = M_ / bgmmc.M_chunk_elems;
            M_chunk_tail_elements_ = M_ % bgmmc.M_chunk_elems;
            int tail = M_chunk_tail_elements_;
            dim_t m_idx = M_ - tail;
            int tail_idx = 0;
            dim_t m_c_buf_idx = 0;
            while (tail > 0) {
                int tail_ker_size = dynamic_m_tails[tail_idx];
                int ker_idx = tail_idx + 1;
                int prev_tail_ker_size = tail_idx > 0
                        ? dynamic_m_tails[tail_idx - 1]
                        : (int)bgmmc.M_blk;
                bool last_tail_kernel = tail_idx == max_num_dynamic_m_tails - 1;
                if (tail > tail_ker_size && M_ >= prev_tail_ker_size) {
                    tail_ker_size = prev_tail_ker_size;
                    ker_idx--;
                } else if (tail < tail_ker_size && !last_tail_kernel) {
                    // skip this tail kernel, try the next one
                    tail_idx++;
                    continue;
                }
                int kernel_m_shift = nstl::max(tail_ker_size - tail, 0);

                m_tail_processing_.push_back({m_idx, ker_idx, tail_ker_size,
                        kernel_m_shift, m_c_buf_idx});
                tail -= tail_ker_size;
                m_idx += tail_ker_size - kernel_m_shift;
                m_c_buf_idx += tail_ker_size;
                if (!last_tail_kernel && tail_ker_size != bgmmc.M_blk)
                    tail_idx++;
            }

            M_tail_block_start_ = M_chunks_ * get_M_chunk_size();
            M_chunk_tail_ = m_tail_processing_.size();
            if (M_chunk_tail_ > 0) M_chunks_++;
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                A_strides_[dim_idx] = bgmmc.a_dt_sz
                        * helper.get_a_stride(bgmmc.ndims - 1 - dim_idx);
            A_ptr_shift_b_ = bgmmc.A_ptr_shift_b;
            if (bgmmc.transposed_A)
                copy_A_src_stride_
                        = helper.get_a_stride(bgmmc.ndims - 1) * bgmmc.a_dt_sz;

            is_A_batch_layout_trivial_
                    = is_batch_layout_trivial(src_d_, bgmmc.batch);
            is_C_batch_layout_trivial_
                    = is_batch_layout_trivial(dst_d_, bgmmc.batch);
        } else {
            M_ = bgmmc.M;
            M_chunks_ = bgmmc.M_chunks;
            M_chunk_tail_ = bgmmc.num_M_blocks % get_M_chunk_size();
            M_chunk_tail_elements_ = M_ % bgmmc.M_chunk_elems;
            M_tail_block_start_ = bgmmc.num_M_blocks - (bgmmc.M_tail > 0);
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                A_strides_[dim_idx] = bgmmc.A_strides[dim_idx];
            A_ptr_shift_b_ = bgmmc.A_ptr_shift_b;
        }

        if (bgmmc.is_runtime_N) {
            N_ = helper.N();
            N_chunks_ = N_ / bgmmc.N_chunk_elems;
            N_chunk_tail_elems_ = N_ % bgmmc.N_chunk_elems;
            int tail = N_chunk_tail_elems_;
            dim_t n_idx = N_ - tail;
            int tail_idx = 0;
            dim_t n_c_buf_idx = 0;
            while (tail > 0) {
                int tail_ker_size = dynamic_n_tails[tail_idx];
                int ker_idx = tail_idx + 1;
                int prev_tail_ker_size = tail_idx > 0
                        ? dynamic_n_tails[tail_idx - 1]
                        : (int)bgmmc.N_blk;
                bool last_tail_kernel = tail_idx == max_num_dynamic_n_tails - 1;
                if (tail > tail_ker_size && N_ >= prev_tail_ker_size) {
                    tail_ker_size = prev_tail_ker_size;
                    ker_idx--;
                } else if (tail < tail_ker_size && !last_tail_kernel) {
                    // skip this tail kernel, try the next one
                    tail_idx++;
                    continue;
                }
                int kernel_n_shift = nstl::max(tail_ker_size - tail, 0);

                n_tail_processing_.push_back({n_idx, ker_idx, tail_ker_size,
                        kernel_n_shift, n_c_buf_idx});
                tail -= tail_ker_size;
                n_idx += tail_ker_size - kernel_n_shift;
                n_c_buf_idx += tail_ker_size;
                if (!last_tail_kernel && tail_ker_size != bgmmc.N_blk)
                    tail_idx++;
            }

            N_tail_block_start_ = N_chunks_ * bgmmc.N_chunk_size;
            N_chunk_tail_ = n_tail_processing_.size();
            if (N_chunk_tail_ > 0) N_chunks_++;

            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                B_strides_[dim_idx] = bgmmc.b_dt_sz
                        * helper.get_b_stride(bgmmc.ndims - 1 - dim_idx);

            is_B_batch_layout_trivial_
                    = is_batch_layout_trivial(wei_d_, bgmmc.batch);
            is_C_batch_layout_trivial_
                    = is_batch_layout_trivial(dst_d_, bgmmc.batch);
        } else {
            N_ = bgmmc.N;
            N_chunks_ = bgmmc.N_chunks;
            N_chunk_tail_ = bgmmc.num_N_blocks % bgmmc.N_chunk_size;
            N_chunk_tail_elems_ = N_ % bgmmc.N_chunk_elems;
            N_tail_block_start_ = bgmmc.num_N_blocks - (bgmmc.N_tail > 0);
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                B_strides_[dim_idx] = bgmmc.B_strides[dim_idx];
        }

        B_ptr_shift_b_ = bgmmc.B_ptr_shift_b;
        copy_B_wei_stride_ = is_runtime_value(bgmmc_.copy_B_wei_stride)
                ? helper.get_b_stride(bgmmc.ndims - 2) * bgmmc.b_dt_sz
                : bgmmc_.copy_B_wei_stride;
        if (bgmmc.is_runtime_M || bgmmc.is_runtime_N) {
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                C_strides_[dim_idx] = bgmmc.c_dt_sz
                        * helper.get_c_stride(bgmmc.ndims - 1 - dim_idx);
        } else {
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                C_strides_[dim_idx] = bgmmc.C_strides[dim_idx];
        }
        C_ptr_shift_b_ = bgmmc_.C_ptr_shift_b;

        // parallelization
        parallel_work_amount_ = bgmmc.batch * M_chunks_ * N_chunks_;

        // The number of threads available during primitive execution may
        // increase (ex. Eigen threadpool implementation) or decrease
        // (ex. nested parallelism) compared to the
        // number of threads available during primitive creation.
        // So we limit the total number of threads to the
        // minimum of these two values to prevent potential OOM issues.
        nthr_ = nstl::min(dnnl_get_current_num_threads(), bgmmc.nthr);

        nthr_k_ = bgmmc.nthr_k > 0 && bgmmc.nthr_k <= nthr_ ? bgmmc.nthr_k : 1;
        nthr_bmn_ = nthr_ / nthr_k_;

        // If parallel_work_amount_ == 1 and parallel reduction is not used, we
        // limit num threads to 1 as parallel(1, ...) does not create parallel
        // section at all. We do not limit number of threads for case
        // 1 < parallel_work_amount_ < dnnl_get_max_threads() to avoid potential
        // overhead on spawning different number of OMP threads from layer to
        // layer.
        if (parallel_work_amount_ == 1 && !parallel_reduction_is_used())
            nthr_ = nthr_bmn_ = nthr_k_ = 1;

        // For Eigen threadpool there is significant advantage to not spawn
        // useless threads.
        if (!dnnl_thr_syncable()) {
            nthr_bmn_ = nstl::min(nthr_bmn_, parallel_work_amount_);
        }

        num_threads_used_ = nthr_k_ * nthr_bmn_;

        const bool need_to_calculate_compensation_for_a
                = bgmmc.has_zero_point_b && !bgmmc.with_wei_decompression;
        const bool need_to_calculate_compensation_for_b = !IMPLICATION(
                (bgmmc.has_zero_point_a || bgmmc.s8s8_compensation_required),
                bgmmc.blocked_B);
        const bool calculate_compensations_in_copy_routines
                = need_to_calculate_compensation_for_a
                || need_to_calculate_compensation_for_b;
        // currently parallel reduction is supported only for case of
        // non-batched problems without computation of any compensations in
        // copy routines
        assert(IMPLICATION(parallel_reduction_is_used(),
                bgmmc.batch == 1 && !calculate_compensations_in_copy_routines));
        MAYBE_UNUSED(need_to_calculate_compensation_for_a);
        MAYBE_UNUSED(need_to_calculate_compensation_for_b);
        MAYBE_UNUSED(calculate_compensations_in_copy_routines);
    }

    // NOTE: gb --> generalized batch, bb --> broadcast batch
    int get_bb_idx(int gb_idx, const brgemm_matmul_bcast_desc_t &bd) const {
        if (!bd.bcast_mask) // no broadcast
            return gb_idx;

        if (bd.bcast_across_all_batch_dims) return 0;

        int gb_off_before_bcast = utils::rnd_dn(
                gb_idx, bd.first_bcast_dim_to_last_batch_dim_prod);
        int bb_idx = gb_off_before_bcast / (bd.bcast_dims_prod);

        dim_t cur_bcast_dims_prod = bd.bcast_dims_prod;
        int mask = 1 << (bgmmc_.batch_ndims - bd.first_bcast_dim - 1);
        for (int d = bd.first_bcast_dim; d < bd.last_bcast_dim; ++d) {
            if (bd.bcast_mask & mask) // broadcast
                cur_bcast_dims_prod /= bd.batch_dims[d];
            else {
                int cur_b = (gb_idx / bd.gb_off[d]) % bd.batch_dims[d];
                bb_idx += cur_b * (bd.gb_off[d] / cur_bcast_dims_prod);
            }
            mask >>= 1;
        }
        bb_idx += gb_idx % bd.gb_off[bd.last_bcast_dim];
        return bb_idx;
    }

    // Note: Minimize the calls to `X_batch_ptr` to reduce overhead.
    // Call it once the batch changes, later use get_data_mk_off
    const char *get_data_A_batch_ptr(int b_idx) const {
        using namespace format_tag;
        const int b = get_bb_idx(b_idx, bgmmc_.bcast_A_desc);
        dim_t b_off = 0;
        if (one_of(bgmmc_.src_tag, acbd, adbc)
                /* this is a special case when src can be represented
                   by plain and transposed tags due to a batch dim equal to 1 */
                || (one_of(bgmmc_.src_tag, abcd, abdc)
                        && bgmmc_.A_ptr_shift_b != 0)) {
            if (!bgmmc_.bcast_A_desc.bcast_mask) { // no broadcast
                const dim_t batch_dim1 = bgmmc_.bcast_A_desc.batch_dims[1];
                b_off = A_strides_[2] * (b % batch_dim1)
                        + (b / batch_dim1) * A_ptr_shift_b_;
            } else {
                b_off = b * A_ptr_shift_b_;
            }
        } else if (is_A_batch_layout_trivial_) {
            b_off = A_strides_[2] * b;
        } else {
            // slow code path
            b_off = src_d_.off_l(b * bgmmc_.M * bgmmc_.K) * bgmmc_.a_dt_sz;
        }
        return data_A_ptr_ + b_off;
    }

    const char *get_data_A_mk_ptr(const char *batch_ptr, int m, int k) const {
        return batch_ptr + A_strides_[1] * m + A_strides_[0] * k;
    }

    dim_t get_data_B_kn_off(int k, int n) const {
        int dt_b_k_blk = bgmmc_.is_bf32
                ? data_type_vnni_simd_elems(f32, bgmmc_.isa)
                : bgmmc_.wei_k_blk;
        int k_idx = bgmmc_.blocked_B ? k / dt_b_k_blk : k;
        int n_idx = bgmmc_.blocked_B ? n / bgmmc_.wei_n_blk : n;
        const int int4_fac = bgmmc_.is_int4_weights ? 2 : 1;
        return (B_strides_[1] * k_idx + B_strides_[0] * n_idx
                       + get_data_B_off_within_block(k, n))
                / int4_fac;
    }

    const char *get_data_B_kn_ptr(const char *batch_ptr, int k, int n) const {
        const char *b_ptr = batch_ptr + get_data_B_kn_off(k, n);
        if (bgmmc_.packed_sparse_weights) {
            const dim_t blk_num
                    = (b_ptr - data_B_ptr_) / B_packed_sparse_block_size_;
            const auto blk_off = data_B_offsets_ptr_[blk_num];
            return data_B_ptr_ + blk_off;
        }
        return b_ptr;
    }

    dim_t get_data_B_batch_off(int b) const {
        using namespace format_tag;
        dim_t b_off = 0;
        if (one_of(bgmmc_.wei_tag, acbd, adbc)
                /* this is a special case when weights can be represented
                   by plain and transposed tags due to a batch dim equal to 1 */
                || (one_of(bgmmc_.wei_tag, abcd, abdc)
                        && bgmmc_.B_ptr_shift_b != 0)) {
            if (!bgmmc_.bcast_B_desc.bcast_mask) { // no broadcast
                const dim_t batch_dim1 = bgmmc_.bcast_B_desc.batch_dims[1];
                b_off = B_strides_[2] * (b % batch_dim1)
                        + (b / batch_dim1) * B_ptr_shift_b_;
            } else {
                b_off = b * B_ptr_shift_b_;
            }
        } else if (is_B_batch_layout_trivial_) {
            b_off = B_strides_[2] * b;
        } else {
            b_off = wei_d_.off_l(b * bgmmc_.K * bgmmc_.N) * bgmmc_.b_dt_sz;
        }
        if (bgmmc_.is_int4_weights) b_off = b_off / 2;
        return b_off;
    }

    const char *get_data_B_batch_ptr(int b_idx) const {
        const int b = get_bb_idx(b_idx, bgmmc_.bcast_B_desc);
        return data_B_ptr_ + get_data_B_batch_off(b);
    }

    const char *get_data_B_bitmask_ptr(int b, int k, int n) const {
        assert(bgmmc_.packed_sparse_weights);
        const dim_t cur_data_B_off
                = get_data_B_batch_off(b) + get_data_B_kn_off(k, n);
        const auto bitmask_off = cur_data_B_off / CHAR_BIT;
        return data_B_bitmask_ptr_ + bitmask_off;
    }

    char *get_data_C_ptr(int b, int m, int n) const {
        return data_C_ptr_ + get_data_C_off(b, m, n);
    }

    brgemm_batch_element_t *get_batch_elem_ptr(int ithr) const {
        return batch_element_ptr_
                + ithr * bgmmc_.brgemm_batch_element_per_thr_sz;
    }

    void init_brgemm_batch_elements_values(int ithr, int brg_batch_start,
            int brg_batch_iters, const char *A_data_batch_ptr,
            const char *B_data_batch_ptr, int b_idx, int m_blk_idx,
            int k_blk_idx, int n_blk_idx) const {
        auto addr_batch = get_batch_elem_ptr(ithr);

        const dim_t m = get_M_idx(m_blk_idx, true);
        const int n = n_blk_idx * bgmmc_.N_blk;

        for (int b_iter = 0; b_iter < brg_batch_iters; b_iter++) {
            const int brg_batch_idx = brg_batch_start + b_iter;
            const int k = (k_blk_idx + brg_batch_idx) * bgmmc_.K_blk;
            addr_batch[b_iter].ptr.A = bgmmc_.use_buffer_a
                    ? get_buf_A_ptr(ithr, m_blk_idx, brg_batch_idx)
                    : get_data_A_mk_ptr(A_data_batch_ptr, m, k);
            addr_batch[b_iter].ptr.B = (bgmmc_.use_buffer_b)
                    ? get_buf_B_ptr(ithr, brg_batch_idx, n_blk_idx)
                    : get_data_B_kn_ptr(B_data_batch_ptr, k, n);
        }
    }

    char *get_buf_A_ptr(int ithr, int m_blk_idx, int k_blk_idx) const {
        if (!bgmmc_.use_buffer_a && !bgmmc_.use_buffer_a_tail_only)
            return nullptr;

        const int k_blk_local = bgmmc_.use_buffer_a_tail_only ? 0 : k_blk_idx;
        if (is_runtime_M_tail_chunk(m_blk_idx)) {
            const int tail_idx = get_M_tail_block_idx(m_blk_idx);
            const int curr_m_block_size
                    = m_tail_processing_[tail_idx].kernel_size;
            const dim_t curr_m_buf_shift
                    = m_tail_processing_[tail_idx].buf_dim_idx;
            const dim_t ld = bgmmc_.tr_a_dt_sz
                    * (bgmmc_.use_buffer_a_tail_only ? bgmmc_.wei_k_blk
                                                     : bgmmc_.LDA);
            const int batch = bgmmc_.use_buffer_a_tail_only
                    ? 1
                    : bgmmc_.brgemm_batch_size;
            const dim_t offset = curr_m_buf_shift * ld * batch
                    + k_blk_local * ld * curr_m_block_size;
            return buf_A_ptr_ + ithr * bgmmc_.buffer_a_per_thread_sz + offset;
        }

        const int m_blk_local = m_blk_idx % get_M_chunk_size();
        return buf_A_ptr_ + ithr * bgmmc_.buffer_a_per_thread_sz
                + m_blk_local * bgmmc_.buffer_a_chunk_shift_along_m
                + k_blk_local * bgmmc_.buffer_a_chunk_sz;
    }

    char *get_buf_B_ptr(int ithr, int k_blk_idx, int n_blk_idx) const {
        UNUSED(n_blk_idx);
        if (!bgmmc_.use_buffer_b) return nullptr;

        return buf_B_ptr_ + ithr * bgmmc_.buffer_b_per_thread_sz
                + k_blk_idx * bgmmc_.buffer_b_chunk_sz;
    }

    char *get_buf_C_ptr(int ithr, int m_blk_idx, int n_blk_idx) const {
        if (!bgmmc_.use_buffer_c) return nullptr;

        if (bgmmc_.nthr_k > 1) {
            const int ithr_k = get_thread_idx_for_k(ithr);
            return get_buf_C_par_reduction_ptr(ithr_k, m_blk_idx, n_blk_idx);
        }
        char *buf_C_ptr_local
                = buf_C_ptr_ + ithr * bgmmc_.buffer_c_per_thread_sz;
        const int n_blk_local = n_blk_idx % bgmmc_.N_chunk_size;
        const int m_blk_local = m_blk_idx % get_M_chunk_size();
        const bool runtime_M_tail = is_runtime_M_tail_chunk(m_blk_idx);
        const bool runtime_N_tail = is_runtime_N_tail_chunk(n_blk_idx);
        if (runtime_M_tail || runtime_N_tail) {
            const int curr_m_block_size = get_M_kernel_size(m_blk_idx);
            const dim_t curr_m_buf_shift = runtime_M_tail
                    ? m_tail_processing_[get_M_tail_block_idx(m_blk_idx)]
                              .buf_dim_idx
                    : m_blk_local;
            const dim_t curr_n_buf_shift = runtime_N_tail
                    ? n_tail_processing_[get_N_tail_block_idx(n_blk_idx)]
                              .buf_dim_idx
                    : n_blk_local;
            const dim_t m_elems_shift = curr_m_buf_shift * bgmmc_.N_chunk_elems;
            const dim_t n_elems_shift = curr_n_buf_shift
                    * (bgmmc_.is_runtime_N ? 1
                                           : curr_m_block_size * bgmmc_.N_blk);

            const dim_t offset
                    = bgmmc_.acc_dt_sz * (m_elems_shift + n_elems_shift);
            return buf_C_ptr_local + offset;
        }
        const dim_t m_shift
                = bgmmc_.N_chunk_size * m_blk_local * bgmmc_.buffer_c_chunk_sz;
        const dim_t n_shift = n_blk_local
                * (bgmmc_.is_runtime_N ? bgmmc_.acc_dt_sz * bgmmc_.N_blk
                                       : bgmmc_.buffer_c_chunk_sz);
        return buf_C_ptr_local + m_shift + n_shift;
    }

    char *get_buf_C_par_reduction_ptr(
            int ithr_k, int m_blk_idx, int n_blk_idx) const {
        if (bgmmc_.nthr_k <= 1) return nullptr;

        const int m = m_blk_idx * bgmmc_.M_blk;
        const int n = n_blk_idx * bgmmc_.N_blk;

        if (!bgmmc_.post_ops_applicable && ithr_k == 0)
            return get_data_C_ptr(0, m, n);

        int k_buf_idx = ithr_k - (!bgmmc_.post_ops_applicable ? 1 : 0);
        return buf_C_ptr_ + k_buf_idx * bgmmc_.buffer_c_per_thread_sz
                + get_data_C_off(0, m, n) * bgmmc_.acc_dt_sz / bgmmc_.c_dt_sz;
    }

    dim_t get_data_B_off_within_block(int k, int n) const {
        using namespace format_tag;

        if (!bgmmc_.blocked_B) return 0;

        int x0 = k % bgmmc_.wei_k_blk;
        int x1 = n % bgmmc_.wei_n_blk;
        dim_t offset = static_cast<dim_t>(x0 / vnni_factor) * vnni_factor
                        * bgmmc_.wei_n_blk
                + x1 * vnni_factor + x0 % vnni_factor;
        return bgmmc_.b_dt_sz * offset;
    }

    dim_t get_data_C_off(int b, int m, int n) const {
        using namespace format_tag;
        assert(bgmmc_.dst_tag != adbc);
        dim_t off = 0;
        if (bgmmc_.dst_tag == acbd
                || (one_of(bgmmc_.dst_tag, abcd, abdc)
                        && bgmmc_.C_ptr_shift_b != 0)) {
            const dim_t batch_dim1 = bgmmc_.bcast_A_desc.batch_dims[1];
            dim_t b_off = C_strides_[2] * (b % batch_dim1)
                    + (b / batch_dim1) * C_ptr_shift_b_;
            off = b_off + C_strides_[1] * m + C_strides_[0] * n;
        } else if (is_C_batch_layout_trivial_) {
            off = C_strides_[2] * b + C_strides_[1] * m + C_strides_[0] * n;
        } else {
            // slow code
            off = dst_d_.off_l(b * bgmmc_.M * bgmmc_.N) * bgmmc_.c_dt_sz
                    + C_strides_[1] * m + C_strides_[0] * n;
        }
        return off;
    }

    // Returns a pointer to the user-provided reduce buffer, shifted by
    // the specified offset @p off.
    char *get_data_reduce_ptr(int off) const {
        if (!bgmmc_.with_reduce) return nullptr;
        return data_reduce_ptr_ + off * bgmmc_.reduce_dt_sz;
    }

    // Returns a pointer to the scratchpad reduce buffer for the
    // corresponding @p ithr, shifted by the specified offset @p off.
    char *get_buf_reduce_ptr(int ithr, int off) const {
        if (!bgmmc_.with_reduce) return nullptr;
        assert(bgmmc_.acc_dt == f32);
        const int ithr_k = get_thread_idx_for_k(ithr);
        // Use the user-provided reduce buffer as one of the reduce buffers.
        const bool is_reduce_f32 = bgmmc_.reduce_dt == f32;
        if (is_reduce_f32 && ithr_k == 0) return get_data_reduce_ptr(off);

        return buf_reduce_ptr_
                + (ithr_k - is_reduce_f32) * bgmmc_.buffer_reduce_per_thread_sz
                + off * bgmmc_.acc_dt_sz;
    }

    // Returns a pointer to the scratchpad reduce buffer for the
    // corresponding index @p ibuf, shifted by the specified offset @p off.
    char *get_buf_reduce_ptr_by_index(int ibuf, int off) const {
        if (!bgmmc_.with_reduce) return nullptr;
        const size_t _off = bgmmc_.M * ibuf + off;
        return buf_reduce_ptr_ + _off * bgmmc_.acc_dt_sz;
    }

    const char *get_bias_ptr(int n) const {
        if (!bgmmc_.with_bias) return nullptr;

        return bias_ptr_ + n * bgmmc_.bias_dt_sz;
    }

    int32_t *get_s8s8_comp_ptr(int ithr, int b, int n_blk_idx) const {
        if (!bgmmc_.s8s8_compensation_required) return nullptr;

        const int n_blk_local = bgmmc_.use_buffer_b
                ? n_blk_idx % bgmmc_.N_chunk_size
                : n_blk_idx;
        return s8s8_compensation_ptr_ + ithr * bgmmc_.s8s8_comp_ithr_str
                + get_bb_idx(b, bgmmc_.bcast_B_desc) * bgmmc_.s8s8_comp_b_str
                + n_blk_local * bgmmc_.s8s8_comp_n_str;
    }

    const float *get_oscales_ptr(int n, int k = 0) const {
        const auto offset = bgmmc_.req_transpose_scales
                ? bgmmc_.is_oscale_per_k * k
                        + (bgmmc_.is_oscale_per_n * n
                                * (bgmmc_.is_oscale_per_k ? bgmmc_.K : 1))
                : bgmmc_.is_oscale_per_n * n
                        + (bgmmc_.is_oscale_per_k * k
                                * (bgmmc_.is_oscale_per_n ? bgmmc_.N : 1));
        return oscales_ptr_ + offset;
    }

    const float *get_dst_scales_ptr() const { return dst_scales_ptr_; }

    const int32_t *get_zp_a_neg_val_ptr() const {
        return &zero_point_a_negative_val_;
    }

    const int32_t *get_zp_b_neg_val_ptr() const {
        return &zero_point_b_negative_val_;
    }

    const int32_t *get_zp_b_val_ptr() const { return &zero_point_b_val_; }

    const int32_t *get_zp_ab_mixed_comp_ptr() const {
        return &zero_point_mixed_ab_compensation_component_;
    }

    const int32_t *get_zp_c_val_ptr() const { return &zero_point_c_val_; }

    int32_t *get_zp_a_compensation_ptr(
            int ithr, int b_idx, int n_blk_idx) const {
        if (!bgmmc_.has_zero_point_a) return nullptr;

        const int n_blk_local = n_blk_idx % bgmmc_.N_chunk_size;
        int32_t *zp_comp = zero_point_a_compensations_ptr_
                + ithr * bgmmc_.zp_a_comp_elems_per_thr
                + n_blk_local * bgmmc_.zp_a_comp_shift_n;

        if (bgmmc_.blocked_B) {
            // Scale computed in reorder compensation values by zp_a value
            // locally just before usage. Using the single global scaling before
            // parallel section might produce significant overhead for small
            // problems running in multitreaded execution mode
            const int base_offset = get_bb_idx(b_idx, bgmmc_.bcast_B_desc)
                            * rnd_up(bgmmc_.N, bgmmc_.wei_n_blk)
                    + n_blk_idx * bgmmc_.wei_n_blk;
            PRAGMA_OMP_SIMD()
            for (int b = 0; b < bgmmc_.wei_n_blk; b++)
                zp_comp[b] = -zero_point_a_negative_val_
                        * reorder_zp_a_comp_ptr_[base_offset + b];
        }
        return zp_comp;
    }

    int32_t *get_zp_b_compensation_result_ptr(int ithr, int m_blk_idx) const {
        if (!bgmmc_.has_zero_point_b) return nullptr;

        if (is_runtime_M_tail_chunk(m_blk_idx)) {
            const dim_t curr_m_buf_shift
                    = m_tail_processing_[get_M_tail_block_idx(m_blk_idx)]
                              .buf_dim_idx;
            return zero_point_b_compensations_ptr_
                    + ithr * bgmmc_.zp_b_comp_elems_per_thr + curr_m_buf_shift;
        }

        const int m_blk_local = m_blk_idx % get_M_chunk_size();
        return zero_point_b_compensations_ptr_
                + ithr * bgmmc_.zp_b_comp_elems_per_thr
                + m_blk_local * bgmmc_.zp_b_comp_result_shift_m;
    }

    int32_t *get_zp_b_compensation_buffer_ptr(int ithr, int m_blk_idx) const {
        if (!bgmmc_.has_zero_point_b) return nullptr;

        if (is_runtime_M_tail_chunk(m_blk_idx)) {
            const dim_t curr_m_buf_shift
                    = m_tail_processing_[get_M_tail_block_idx(m_blk_idx)]
                              .buf_dim_idx;
            return get_zp_b_compensation_result_ptr(ithr, 0)
                    + bgmmc_.zp_b_comp_buffer_start + curr_m_buf_shift;
        }

        const int m_blk_local = m_blk_idx % get_M_chunk_size();
        return get_zp_b_compensation_result_ptr(ithr, 0)
                + bgmmc_.zp_b_comp_buffer_start
                + m_blk_local * bgmmc_.zp_b_comp_buffer_shift_m;
    }

    char *get_tile_workspace(int ithr) const {
        return is_amx_ ? wsp_tile_ptr_ + ithr * bgmmc_.wsp_tile_per_thr_bytes
                       : nullptr;
    }

    const std::vector<const void *> &get_post_ops_binary_rhs_arg_vec() const {
        return post_ops_binary_rhs_arg_vec_;
    }

    int get_base_brgemm_kernel_idx() const { return base_brg_ker_idx_; }

    bool is_last_K_chunk(int k_chunk_idx) const {
        return k_chunk_idx == bgmmc_.K_chunks - 1;
    }

    int get_brgemm_batch_size(int k_chunk_idx) const {
        return is_last_K_chunk(k_chunk_idx) ? last_chunk_brgemm_batch_size_
                                            : bgmmc_.brgemm_batch_size;
    }

    int get_parallel_work_amount() const { return parallel_work_amount_; }
    int get_num_threads_for_k() const { return nthr_k_; }
    bool parallel_reduction_is_used() const {
        return nthr_k_ > 1 && bgmmc_.K_chunks > 1;
    }
    int get_num_threads_for_bmn() const { return nthr_bmn_; }
    // ithr = ithr_k * nthr_bmn + ithr_bmn
    int get_thread_idx_for_k(int ithr) const {
        if (ithr >= num_threads_used_) return -1;
        const int ithr_k = ithr / nthr_bmn_;
        return ithr_k < bgmmc_.K_chunks ? ithr_k : -1;
    }
    int get_thread_idx_for_bmn(int ithr) const {
        if (ithr >= num_threads_used_) return -1;
        const int ithr_bmn = ithr % nthr_bmn_;
        return ithr_bmn < parallel_work_amount_ ? ithr_bmn : -1;
    }
    int get_num_threads_for_parallelization() const {
        return num_threads_used_;
    }
    dim_t get_M() const { return M_; }
    int get_M_chunks() const { return M_chunks_; }
    int get_M_chunk_size() const { return bgmmc_.M_chunk_size; }
    int get_M_chunk_tail() const { return M_chunk_tail_; }

    int get_M_kernel_idx(int m_block_idx) const {
        if (!is_M_tail_processing(m_block_idx))
            return 0;
        else if (!bgmmc_.is_runtime_M)
            return 1;

        assert(is_runtime_M_tail_chunk(m_block_idx)
                && !m_tail_processing_.empty());
        return m_tail_processing_[get_M_tail_block_idx(m_block_idx)].kernel_idx;
    }

    int get_M_kernel_size(int m_block_idx) const {
        if (!is_M_tail_processing(m_block_idx))
            return bgmmc_.M_blk;
        else if (!bgmmc_.is_runtime_M)
            return bgmmc_.M_tail;

        assert(is_runtime_M_tail_chunk(m_block_idx)
                && !m_tail_processing_.empty());
        return m_tail_processing_[get_M_tail_block_idx(m_block_idx)]
                .kernel_size;
    }

    dim_t get_M_idx(
            int m_block_idx, bool adjust_for_kernel_overlap = false) const {
        if (is_runtime_M_tail_chunk(m_block_idx)) {
            const int tail_idx = get_M_tail_block_idx(m_block_idx);
            const int shift = adjust_for_kernel_overlap
                    ? m_tail_processing_[tail_idx].shift
                    : 0;
            return m_tail_processing_[tail_idx].idx - shift;
        }
        return m_block_idx * bgmmc_.M_blk;
    }

    dim_t get_N() const { return N_; }
    int get_N_chunks() const { return N_chunks_; }
    int get_N_chunk_tail() const { return N_chunk_tail_; }
    int get_N_chunk_tail_elems() const { return N_chunk_tail_elems_; }

    int get_N_kernel_idx(int n_block_idx) const {
        if (!is_N_tail_processing(n_block_idx))
            return 0;
        else if (!bgmmc_.is_runtime_N)
            return 1;

        assert(is_runtime_N_tail_chunk(n_block_idx)
                && !n_tail_processing_.empty());
        return n_tail_processing_[get_N_tail_block_idx(n_block_idx)].kernel_idx;
    }

    int get_N_kernel_size(int n_block_idx) const {
        if (!is_N_tail_processing(n_block_idx))
            return bgmmc_.N_blk;
        else if (!bgmmc_.is_runtime_N)
            return bgmmc_.N_tail;

        assert(is_runtime_N_tail_chunk(n_block_idx)
                && !n_tail_processing_.empty());
        return n_tail_processing_[get_N_tail_block_idx(n_block_idx)]
                .kernel_size;
    }

    dim_t get_N_idx(
            int n_block_idx, bool adjust_for_kernel_overlap = false) const {
        if (is_runtime_N_tail_chunk(n_block_idx)) {
            const int tail_idx = get_N_tail_block_idx(n_block_idx);
            const int shift = adjust_for_kernel_overlap
                    ? n_tail_processing_[tail_idx].shift
                    : 0;
            return n_tail_processing_[tail_idx].idx - shift;
        }
        return n_block_idx * bgmmc_.N_blk;
    }

    dim_t get_src_stride() const { return copy_A_src_stride_; }

    // For tail processing in the case of dynamic dimensions, it's possible to
    // have kernels overlap on the dst tensor when two different kernels
    // compute dst values for the same area. We need to backup/restore values
    // for overlapped area to avoid correctness issues.
    void maybe_backup_dst_values_to_buffer(
            int ithr, int b_idx, int m_blk_idx, int n_blk_idx) const {
        if (!copy_d_required(m_blk_idx, n_blk_idx)) return;

        const bool m_tail_overlapping = is_m_tail_overlap(m_blk_idx);
        dim_t m_start = m_tail_overlapping ? get_M_idx(m_blk_idx + 1, true)
                                           : get_M_idx(m_blk_idx);
        const int rows_to_copy = m_tail_overlapping
                ? m_tail_processing_[get_M_tail_block_idx(m_blk_idx + 1)].shift
                : get_M_kernel_size(m_blk_idx);

        const bool n_tail_overlapping = is_n_tail_overlap(n_blk_idx);
        dim_t n_start = n_tail_overlapping ? get_N_idx(n_blk_idx + 1, true)
                                           : get_N_idx(n_blk_idx);
        const int row_elems = n_tail_overlapping
                ? n_tail_processing_[get_N_tail_block_idx(n_blk_idx + 1)].shift
                : get_N_kernel_size(n_blk_idx);
        const dim_t bytes_to_copy = bgmmc_.c_dt_sz * row_elems;
        assert(!(n_tail_overlapping && m_tail_overlapping)
                && "dynamic tail processing for both M/N is not supported");

        auto copy_from = get_data_C_ptr(b_idx, m_start, n_start);
        auto copy_to = get_buf_D_ptr(ithr);
        const dim_t dst_ld = get_LDD() * bgmmc_.c_dt_sz;
        const dim_t buf_ld = bgmmc_.N_blk * bgmmc_.c_dt_sz;
        for (int r = 0; r < rows_to_copy; r++) {
            utils::array_copy(copy_to, copy_from, bytes_to_copy);
            copy_from += dst_ld;
            copy_to += buf_ld;
        }
    }

    void maybe_restore_dst_values_from_buffer(
            int ithr, int b_idx, int m_blk_idx, int n_blk_idx) const {
        if (!copy_d_required(m_blk_idx, n_blk_idx)) return;

        const bool m_tail_overlapping = is_m_tail_overlap(m_blk_idx);
        dim_t m_start = m_tail_overlapping ? get_M_idx(m_blk_idx + 1, true)
                                           : get_M_idx(m_blk_idx);
        const int rows_to_copy = m_tail_overlapping
                ? m_tail_processing_[get_M_tail_block_idx(m_blk_idx + 1)].shift
                : get_M_kernel_size(m_blk_idx);

        const bool n_tail_overlapping = is_n_tail_overlap(n_blk_idx);
        dim_t n_start = n_tail_overlapping ? get_N_idx(n_blk_idx + 1, true)
                                           : get_N_idx(n_blk_idx);
        const int row_elems = n_tail_overlapping
                ? n_tail_processing_[get_N_tail_block_idx(n_blk_idx + 1)].shift
                : get_N_kernel_size(n_blk_idx);
        const dim_t bytes_to_copy = bgmmc_.c_dt_sz * row_elems;

        assert(!(n_tail_overlapping && m_tail_overlapping)
                && "dynamic tail processing for both M/N is not supported");

        auto copy_from = get_buf_D_ptr(ithr);
        auto copy_to = get_data_C_ptr(b_idx, m_start, n_start);
        const dim_t dst_ld = get_LDD() * bgmmc_.c_dt_sz;
        const dim_t buf_ld = bgmmc_.N_blk * bgmmc_.c_dt_sz;
        for (int r = 0; r < rows_to_copy; r++) {
            utils::array_copy(copy_to, copy_from, bytes_to_copy);
            copy_from += buf_ld;
            copy_to += dst_ld;
        }
    }

    dim_t get_LDC() const { return LDC_; }

    dim_t get_LDD() const { return LDD_; }

    dim_t copy_B_wei_stride() const { return copy_B_wei_stride_; }

    bool packed_sparse_weights() const { return bgmmc_.packed_sparse_weights; }

    int get_current_K_pad(int current_K_iters) const {
        if (current_K_iters % bgmmc_.wei_k_blk == 0) return 0;
        return bgmmc_.extendable_k ? bgmmc_.wei_k_blk
                        - rnd_up(
                                current_K_iters % bgmmc_.wei_k_blk, vnni_factor)
                                   : 0;
    }

private:
    struct tail_processing_t {
        // dimension index kernel is applied to
        dim_t idx;
        // index of tail processing kernel, 0 is reserved for main block
        int kernel_idx;
        // block size of tail kernel
        int kernel_size;
        // shift wrt dimension index when kernel is applied w/ computational
        // overlapping with other kernel, dim_idx_to_apply_kernel = idx - shift
        int shift;
        // if shift > 0 (computational overlapping case) we have to use buffer
        // for kernel dst to avoid result values spoiling, this value
        // represents dimensional idx for dst buffer
        dim_t buf_dim_idx;
    };

    bool is_amx_;
    bool is_A_batch_layout_trivial_;
    bool is_B_batch_layout_trivial_;
    bool is_C_batch_layout_trivial_;
    const brgemm_matmul_conf_t &bgmmc_;
    const memory_desc_wrapper src_d_;
    const memory_desc_wrapper wei_d_;
    const memory_desc_wrapper dst_d_;
    const char *data_A_ptr_;
    const char *data_B_ptr_;
    // The offsets and bitmask pointers are only available when the weights
    // are sparse and packed.
    const dim_t *data_B_offsets_ptr_;
    const char *data_B_bitmask_ptr_;
    // The size of a packed saprse block. E.g. the block
    // for a tag 'BA16a64b4a' is 4096.
    int B_packed_sparse_block_size_;

    char *data_C_ptr_;
    char *data_reduce_ptr_;
    brgemm_batch_element_t *batch_element_ptr_;

    char *buf_A_ptr_;
    char *buf_B_ptr_;
    char *buf_C_ptr_;
    char *buf_D_ptr_;
    char *buf_reduce_ptr_;

    char *wsp_tile_ptr_;
    const char *bias_ptr_;
    const float *oscales_ptr_;
    const float *dst_scales_ptr_;
    int32_t *s8s8_compensation_ptr_;

    int32_t *zero_point_a_compensations_ptr_;
    int32_t *zero_point_b_compensations_ptr_;
    int32_t *reorder_zp_a_comp_ptr_;

    int32_t zero_point_a_negative_val_;
    int32_t zero_point_b_val_;
    int32_t zero_point_b_negative_val_;
    int32_t zero_point_mixed_ab_compensation_component_;
    int32_t zero_point_c_val_;
    std::vector<const void *> post_ops_binary_rhs_arg_vec_;

    int base_brg_ker_idx_;
    int vnni_factor;

    // parallelization parameters
    int parallel_work_amount_;
    int nthr_, nthr_k_, nthr_bmn_, num_threads_used_;
    int last_chunk_brgemm_batch_size_;
    dim_t M_;
    int M_chunks_;
    int M_chunk_tail_;
    int M_chunk_tail_elements_;
    int M_tail_block_start_;

    dim_t N_;
    int N_chunks_;
    int N_chunk_tail_;
    int N_chunk_tail_elems_;

    int N_tail_block_start_;
    dim_t A_strides_[3];
    dim_t A_ptr_shift_b_;
    dim_t copy_A_src_stride_;
    dim_t B_strides_[3];
    dim_t B_ptr_shift_b_;
    dim_t C_strides_[3];
    dim_t C_ptr_shift_b_;
    dim_t LDC_, LDD_;
    dim_t copy_B_wei_stride_;
    std::vector<tail_processing_t> m_tail_processing_;
    std::vector<tail_processing_t> n_tail_processing_;

    char *get_buf_D_ptr(int ithr) const {
        return buf_D_ptr_ + bgmmc_.c_dt_sz * bgmmc_.M_blk * bgmmc_.N_blk * ithr;
    }

    int get_M_tail_block_idx(int m_block_idx) const {
        const int tail_idx = m_block_idx - M_tail_block_start_;
        if (!bgmmc_.is_runtime_M) return tail_idx;
        return tail_idx < (int)m_tail_processing_.size() ? tail_idx : -1;
    }
    bool is_M_tail_processing(int m_block_idx) const {
        return get_M_tail_block_idx(m_block_idx) >= 0;
    }
    bool is_runtime_M_tail_chunk(int m_block_idx) const {
        return bgmmc_.is_runtime_M && is_M_tail_processing(m_block_idx);
    }

    bool is_m_tail_overlap(int m_block_idx) const {
        return is_runtime_M_tail_chunk(m_block_idx)
                && is_runtime_M_tail_chunk(m_block_idx + 1)
                && m_tail_processing_[get_M_tail_block_idx(m_block_idx + 1)]
                           .shift
                > 0;
    }

    int get_N_tail_block_idx(int n_block_idx) const {
        const int tail_idx = n_block_idx - N_tail_block_start_;
        if (!bgmmc_.is_runtime_N) return tail_idx;
        return tail_idx < (int)n_tail_processing_.size() ? tail_idx : -1;
    }
    bool is_N_tail_processing(int n_block_idx) const {
        return get_N_tail_block_idx(n_block_idx) >= 0;
    }

    bool is_runtime_N_tail_chunk(int n_block_idx) const {
        return bgmmc_.is_runtime_N && is_N_tail_processing(n_block_idx);
    }

    bool is_n_tail_overlap(int n_block_idx) const {
        return is_runtime_N_tail_chunk(n_block_idx)
                && is_runtime_N_tail_chunk(n_block_idx + 1)
                && n_tail_processing_[get_N_tail_block_idx(n_block_idx + 1)]
                           .shift
                > 0;
    }

    bool copy_d_required(int m_block_idx, int n_block_idx) const {
        if (!bgmmc_.with_sum) return false;
        return is_m_tail_overlap(m_block_idx) || is_n_tail_overlap(n_block_idx);
    }
};

template struct brgemm_matmul_t<avx512_core_amx_fp16>;
template struct brgemm_matmul_t<avx512_core_amx>;
template struct brgemm_matmul_t<avx512_core_fp16>;
template struct brgemm_matmul_t<avx512_core_bf16>;
template struct brgemm_matmul_t<avx512_core_vnni>;
template struct brgemm_matmul_t<avx2_vnni_2>;
template struct brgemm_matmul_t<avx2_vnni>;
template struct brgemm_matmul_t<avx2>;
template struct brgemm_matmul_t<avx512_core>;

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
