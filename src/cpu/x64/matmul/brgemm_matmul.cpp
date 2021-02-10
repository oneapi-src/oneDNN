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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/matmul/brgemm_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using namespace data_type;

#define get_blk_off(d, dt, b, idx1, idx0) \
    (types::data_type_size((dt)) \
            * ((bgmmc.ndims == 3) ? (d).blk_off(b, idx1, idx0) \
                                  : (d).blk_off(idx1, idx0)))

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::pd_t::init(engine_t *engine) {

    auto check_bias = [&]() -> bool {
        return IMPLICATION(with_bias(),
                utils::one_of(weights_md(1)->data_type, f32, s32, s8, u8)
                        && is_bias_1xN());
    };

    auto check_attr_oscale = [&]() -> bool {
        const auto &oscale = attr()->output_scales_;
        return IMPLICATION(oscale.mask_ != 0,
                oscale.mask_ == (1 << 1) && batched() == false);
    };

    auto check_attr_zero_points = [&]() -> bool {
        return attr()->zero_points_.has_default_values();
    };

    bool ok = true && mayiuse(isa) && !has_runtime_dims_or_strides()
            && attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::zero_points_runtime
                    | primitive_attr_t::skip_mask_t::post_ops)
            && check_attr_oscale() && check_attr_zero_points() && check_bias();
    if (!ok) return status::unimplemented;

    CHECK(init_brgemm_matmul_conf(isa, bgmmc_, *desc(), src_md_, weights_md_,
            dst_md_, bias_md_, *attr()));

    const float alpha = 1.0;
    const float beta = 1.0;
    const float beta_init = 0.0;
    for_(int i_init = 0; i_init < 2; i_init++)
    for_(int i_M = 0; i_M < 2; i_M++)
    for_(int i_N = 0; i_N < 2; i_N++)
    for (int i_K = 0; i_K < 2; i_K++) {
        auto vbeta = (i_init) ? beta_init : beta;
        auto vM = (i_M) ? bgmmc_.M_tail : bgmmc_.M_blk;
        auto vN = (i_N) ? bgmmc_.N_tail : bgmmc_.N_blk;
        auto vK = (i_K) ? bgmmc_.K_tail : bgmmc_.K_blk;

        int idx = get_brg_kernel_idx(i_init, i_M, i_N, i_K);
        if (idx < 0) continue;
        brgemm_t &brg = brg_descs_[idx];
        auto LDA = i_K && bgmmc_.use_buffer_a_tail_only
                ? (dim_t)bgmmc_.wei_k_blk
                : bgmmc_.LDA;
        CHECK(brgemm_desc_init(&brg, isa, bgmmc_.brg_type, bgmmc_.src_dt,
                bgmmc_.wei_dt, false, false, brgemm_row_major, alpha, vbeta,
                LDA, bgmmc_.LDB, bgmmc_.LDC, vM, vN, vK));

        auto LDD = bgmmc_.N;
        CHECK(brgemm_desc_set_postops(
                &brg, attr(), bgmmc_.dst_dt, LDD, bgmmc_.bia_dt));
    }

    auto scratchpad = scratchpad_registry().registrar();
    init_scratchpad(scratchpad, bgmmc_);

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::init(engine_t *engine) {
    for_(int i_M = 0; i_M < 2; i_M++)
    for_(int i_N = 0; i_N < 2; i_N++)
    for_(int i_K = 0; i_K < 2; i_K++)
    for (int i_init = 0; i_init < 2; i_init++) {
        int idx = pd()->get_brg_kernel_idx(i_init, i_M, i_N, i_K);
        if (idx < 0) continue;

        brgemm_kernel_t *ker = nullptr;
        CHECK(brgemm_kernel_create(&ker, pd()->get_brg_desc(idx)));
        CHECK(safe_ptr_assign(brg_kernels_[idx], ker));
        if (isa == avx512_core_bf16_amx_int8)
            CHECK(brgemm_init_tiles(
                    pd()->get_brg_desc(idx), &brg_kernel_palettes_[idx][0]));
    }

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    if (bgmmc.use_buffer_b)
        CHECK(create_brgemm_matmul_copy_B(copy_B_kernel_, &bgmmc));

    if (bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only)
        CHECK(create_brgemm_matmul_copy_A(copy_A_kernel_, &bgmmc));

    return status::success;
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::execute_body(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const float *oscales = pd()->attr()->output_scales_.scales_;

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const size_t bia_dt_size
            = bgmmc.with_bias ? types::data_type_size(bgmmc.bia_dt) : 0;

    auto addr_batch_global = scratchpad.template get<brgemm_batch_element_t>(
            key_brgemm_primitive_batch);
    auto c_buffer_global = (bgmmc.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;

    const size_t b_buffer_sz = types::data_type_size(bgmmc.wei_dt) * bgmmc.LDB
            * rnd_up(bgmmc.K_blk, bgmmc.wei_k_blk);
    const size_t b_buffer_per_thr = b_buffer_sz * bgmmc.brgemm_batch_size;
    auto b_buffer_global = (bgmmc.use_buffer_b)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer_b)
            : nullptr;

    const size_t a_buffer_sz = types::data_type_size(bgmmc.src_dt) * bgmmc.M_blk
            * (bgmmc.use_buffer_a_tail_only ? bgmmc.wei_k_blk : bgmmc.LDA);
    const size_t a_buffer_K_chunk_sz = a_buffer_sz
            * (bgmmc.use_buffer_a_tail_only ? 1 : bgmmc.brgemm_batch_size);
    const size_t a_buffer_per_thr = a_buffer_K_chunk_sz * bgmmc.M_chunk_size;
    const bool use_buffer_a
            = bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only;
    auto a_buffer_global = (use_buffer_a)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer_a)
            : nullptr;

    constexpr bool is_amx = isa == avx512_core_bf16_amx_int8;
    auto wsp_tile_base = is_amx
            ? ctx.get_scratchpad_grantor().template get<char>(
                    key_conv_amx_tile_buffer)
            : nullptr;

    int K_chunks = div_up(bgmmc.K, bgmmc.K_blk * bgmmc.brgemm_batch_size);
    bool are_post_ops_applicable = one_of(true, bgmmc.with_sum, bgmmc.with_bias,
            bgmmc.with_scales, bgmmc.with_eltwise, bgmmc.acc_dt != bgmmc.dst_dt,
            bgmmc.signed_input);

    size_t offset = types::data_type_size(bgmmc.wei_dt)
            * (weights_d.size() - weights_d.additional_buffer_size());
    auto compensation = (bgmmc.signed_input)
            ? ((bgmmc.use_buffer_b) ? scratchpad.template get<int32_t>(
                       key_brgemm_primitive_buffer_comp)
                                    : reinterpret_cast<const int32_t *>(
                                            &weights[offset]))
            : nullptr;

    const auto ker = [&](const int ithr, int b_idx, int m_blk_idx,
                             int n_blk_idx, int k_blk_idx) {
        auto addr_batch
                = addr_batch_global + ithr * 16 * bgmmc.brgemm_batch_size;

        const size_t c_buffer_sz
                = types::data_type_size(bgmmc.acc_dt) * bgmmc.LDC * bgmmc.M_blk;
        const size_t c_buffer_per_thr
                = c_buffer_sz * bgmmc.M_chunk_size * bgmmc.N_chunk_size;
        int c_buf_idx = bgmmc.N_chunk_size * (m_blk_idx % bgmmc.M_chunk_size)
                + (n_blk_idx % bgmmc.N_chunk_size);
        const size_t c_buffer_shift_within_ithr = c_buffer_sz * c_buf_idx;
        const size_t c_buffer_shift
                = ithr * c_buffer_per_thr + c_buffer_shift_within_ithr;
        auto c_buffer = (bgmmc.use_buffer) ? c_buffer_global + c_buffer_shift
                                           : nullptr;

        auto b_buffer = (bgmmc.use_buffer_b)
                ? b_buffer_global + ithr * b_buffer_per_thr
                : nullptr;

        auto a_buffer = (use_buffer_a)
                ? a_buffer_global + ithr * a_buffer_per_thr
                        + a_buffer_K_chunk_sz * (m_blk_idx % bgmmc.M_chunk_size)
                : nullptr;

        char *wsp_tile = is_amx ? wsp_tile_base + ithr * 1024 : nullptr;
        int m = m_blk_idx * bgmmc.M_blk;
        int n = n_blk_idx * bgmmc.N_blk;
        int k = k_blk_idx * bgmmc.K_blk * bgmmc.brgemm_batch_size;

        bool kernel_init = (k == 0);

        bool is_M_tail = (bgmmc.M - m < bgmmc.M_blk);
        bool is_N_tail = (bgmmc.N - n < bgmmc.N_blk);
        const bool is_last_K_chunk = k_blk_idx == K_chunks - 1;
        const bool is_K_tail = is_last_K_chunk && bgmmc.K_tail > 0;
        const int gemm_batch = is_last_K_chunk
                ? (nstl::max(bgmmc.K, bgmmc.K_blk) - k) / bgmmc.K_blk
                : bgmmc.brgemm_batch_size;
        int brg_ker_idx = pd()->get_brg_kernel_idx(
                kernel_init, is_M_tail, is_N_tail, false);
        auto brg_kernel = brg_kernels_[brg_ker_idx].get();
        auto ptr_bias = bgmmc.with_bias ? bias + bia_dt_size * n : nullptr;
        auto ptr_D = dst + get_blk_off(dst_d, bgmmc.dst_dt, b_idx, m, n);
        auto ptr_C = (bgmmc.use_buffer) ? c_buffer : ptr_D;

        dim_t buffer_comp_idx
                = (ithr * bgmmc.N_chunk_size + n_blk_idx % bgmmc.N_chunk_size)
                * bgmmc.wei_n_blk;
        dim_t weights_comp_idx
                = (b_idx * div_up(bgmmc.N, bgmmc.wei_n_blk) + n_blk_idx)
                * bgmmc.wei_n_blk;
        dim_t comp_idx
                = bgmmc.use_buffer_b ? buffer_comp_idx : weights_comp_idx;

        if (gemm_batch > 0 && brg_kernel != nullptr) {
            if (is_amx)
                amx_tile_configure(&brg_kernel_palettes_[brg_ker_idx][0]);
            for (int b = 0; b < gemm_batch; b++) {
                auto src_off = get_blk_off(
                        src_d, bgmmc.src_dt, b_idx, m, k + b * bgmmc.K_blk);
                addr_batch[b].ptr.A = (bgmmc.use_buffer_a)
                        ? a_buffer + b * a_buffer_sz
                        : src + src_off;
                auto wei_off = get_blk_off(weights_d, bgmmc.wei_dt, b_idx,
                        (k + b * bgmmc.K_blk) / bgmmc.wei_k_blk,
                        n / bgmmc.wei_n_blk);
                addr_batch[b].ptr.B = (bgmmc.use_buffer_b)
                        ? b_buffer + b * b_buffer_sz
                        : weights + wei_off;
            }

            if (are_post_ops_applicable && is_last_K_chunk && !is_K_tail) {
                brgemm_kernel_execute_postops(brg_kernel, gemm_batch,
                        addr_batch, (void *)ptr_C, (void *)ptr_D,
                        (void *)ptr_bias, &oscales[bgmmc.is_oscale_per_n * n],
                        is_amx ? (void *)wsp_tile
                               : (bgmmc.signed_input
                                               ? (void *)&compensation[comp_idx]
                                               : nullptr));
            } else {
                brgemm_kernel_execute(brg_kernel, gemm_batch, addr_batch,
                        (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
            }
        }
        if (is_K_tail) {
            auto src_off = get_blk_off(src_d, bgmmc.src_dt, b_idx, m,
                    k + gemm_batch * bgmmc.K_blk);
            const int a_buf_gemm_batch
                    = bgmmc.use_buffer_a_tail_only ? 0 : gemm_batch;
            addr_batch[0].ptr.A = (use_buffer_a)
                    ? a_buffer + a_buf_gemm_batch * a_buffer_sz
                    : src + src_off;
            auto wei_off = get_blk_off(weights_d, bgmmc.wei_dt, b_idx,
                    (k + gemm_batch * bgmmc.K_blk) / bgmmc.wei_k_blk,
                    n / bgmmc.wei_n_blk);
            addr_batch[0].ptr.B = (bgmmc.use_buffer_b)
                    ? b_buffer + gemm_batch * b_buffer_sz
                    : weights + wei_off;

            auto use_init_ker = (kernel_init && gemm_batch == 0);
            int brg_ker_idx = pd()->get_brg_kernel_idx(
                    use_init_ker, is_M_tail, is_N_tail, true);
            auto brg_kernel_k_tail = brg_kernels_[brg_ker_idx].get();
            if (is_amx)
                amx_tile_configure(&brg_kernel_palettes_[brg_ker_idx][0]);
            if (are_post_ops_applicable && k_blk_idx == K_chunks - 1) {
                brgemm_kernel_execute_postops(brg_kernel_k_tail, 1, addr_batch,
                        (void *)ptr_C, (void *)ptr_D, (void *)ptr_bias,
                        &oscales[bgmmc.is_oscale_per_n * n],
                        is_amx ? (void *)wsp_tile
                               : (bgmmc.signed_input
                                               ? (void *)&compensation[comp_idx]
                                               : nullptr));
            } else {
                brgemm_kernel_execute(brg_kernel_k_tail, 1, addr_batch,
                        (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
            }
        }
    };

    int num_M_blocks = div_up(bgmmc.M, bgmmc.M_blk);
    int num_N_blocks = div_up(bgmmc.N, bgmmc.N_blk);
    int M_chunks = div_up(bgmmc.M, bgmmc.M_blk * bgmmc.M_chunk_size);
    int N_chunks = div_up(bgmmc.N, bgmmc.N_blk * bgmmc.N_chunk_size);
    int work_amount = bgmmc.batch * M_chunks * N_chunks;

    // If work_amount == 1 we limit num threads to 1 as parallel(1, ...) does
    // not create parallel section at all. We do not limit number of threads
    // for 1 < work_amont < dnnl_get_max_threads() case to avoid potential
    // overhead on spawning different number of OMP threads from layer to layer.
    parallel(work_amount == 1 ? 1 : 0, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        int b {0}, mc {0}, nc {0};
        nd_iterator_init(start, b, bgmmc.batch, mc, M_chunks, nc, N_chunks);
        while (start < end) {
            auto m_start = mc * bgmmc.M_chunk_size;
            auto m_end = nstl::min((mc + 1) * bgmmc.M_chunk_size, num_M_blocks);
            auto n_start = nc * bgmmc.N_chunk_size;
            auto n_end = nstl::min((nc + 1) * bgmmc.N_chunk_size, num_N_blocks);
            for_(int kc = 0; kc < K_chunks; kc++)
            for (int nb = n_start; nb < n_end; nb++) {
                if (bgmmc.use_buffer_b) {
                    int n = nb * bgmmc.N_blk;
                    int k = kc * bgmmc.K_blk * bgmmc.brgemm_batch_size;
                    bool is_N_tail = (bgmmc.N - n < bgmmc.N_blk);
                    auto N_blk = is_N_tail ? bgmmc.N_tail : bgmmc.N_blk;
                    const bool is_last_K_chunk = kc == K_chunks - 1;
                    const bool is_K_tail = is_last_K_chunk && bgmmc.K_tail > 0;
                    const int gemm_batch = is_last_K_chunk
                            ? (nstl::max(bgmmc.K, bgmmc.K_blk) - k)
                                    / bgmmc.K_blk
                            : bgmmc.brgemm_batch_size;
                    auto ctx = jit_brgemm_matmul_copy_B_t::ctx_t();
                    auto b_buffer = b_buffer_global + ithr * b_buffer_per_thr;
                    dim_t buffer_comp_idx = (ithr * bgmmc.N_chunk_size
                                                    + nb % bgmmc.N_chunk_size)
                            * bgmmc.wei_n_blk;
                    auto compensation_local = bgmmc.signed_input
                            ? &compensation[buffer_comp_idx]
                            : nullptr;
                    int gb = 0;
                    for (; gb < gemm_batch; gb++) {
                        auto wei_off = get_blk_off(weights_d, bgmmc.wei_dt, b,
                                k + gb * bgmmc.K_blk, n);
                        ctx.src = (void *)(weights + wei_off);
                        ctx.tr_src = (void *)(b_buffer + gb * b_buffer_sz);
                        ctx.compensation_ptr = (void *)compensation_local;
                        ctx.current_K_start
                                = (kc * bgmmc.brgemm_batch_size + gb)
                                * bgmmc.K_blk;
                        ctx.current_K_iters = nstl::min(bgmmc.K_blk, bgmmc.K);
                        ctx.current_N_blk = N_blk;
                        (*copy_B_kernel_)(&ctx);
                    }

                    if (is_K_tail) {
                        auto wei_off = get_blk_off(weights_d, bgmmc.wei_dt, b,
                                k + gb * bgmmc.K_blk, n);
                        ctx.src = (void *)(weights + wei_off);
                        ctx.tr_src = (void *)(b_buffer + gb * b_buffer_sz);
                        ctx.compensation_ptr = (void *)compensation_local;
                        ctx.current_K_start
                                = (kc * bgmmc.brgemm_batch_size + gb)
                                * bgmmc.K_blk;
                        ctx.current_K_iters = bgmmc.K % bgmmc.K_blk;
                        ctx.current_N_blk = N_blk;
                        (*copy_B_kernel_)(&ctx);
                    }
                }
                for (int mb = m_start; mb < m_end; mb++) {
                    if (use_buffer_a && nb == n_start) {
                        auto ctx = jit_brgemm_matmul_copy_A_t::ctx_t();
                        int m = mb * bgmmc.M_blk;
                        bool is_M_tail = (bgmmc.M - m < bgmmc.M_blk);
                        auto M_blk = is_M_tail ? bgmmc.M_tail : bgmmc.M_blk;
                        int k = kc * bgmmc.K_blk * bgmmc.brgemm_batch_size;
                        const bool is_last_K_chunk = kc == K_chunks - 1;
                        const bool is_K_tail
                                = is_last_K_chunk && bgmmc.K_tail > 0;
                        const int gemm_batch = is_last_K_chunk
                                ? (nstl::max(bgmmc.K, bgmmc.K_blk) - k)
                                        / bgmmc.K_blk
                                : bgmmc.brgemm_batch_size;
                        const int gemm_batch_iters
                                = bgmmc.use_buffer_a_tail_only ? 0 : gemm_batch;

                        auto a_buffer = a_buffer_global
                                + ithr * a_buffer_per_thr
                                + a_buffer_K_chunk_sz
                                        * (mb % bgmmc.M_chunk_size);

                        for (int gb = 0; gb < gemm_batch_iters; gb++) {
                            auto src_off = get_blk_off(src_d, bgmmc.src_dt, b,
                                    m, k + gb * bgmmc.K_blk);
                            ctx.src = (void *)(src + src_off);
                            ctx.tr_src = (void *)(a_buffer + gb * a_buffer_sz);
                            ctx.current_K_blk = nstl::min(bgmmc.K_blk, bgmmc.K);
                            ctx.current_M_blk = M_blk;
                            (*copy_A_kernel_)(&ctx);
                        }
                        if (is_K_tail) {
                            auto K_tail = bgmmc.K % bgmmc.K_blk;
                            auto src_off = get_blk_off(src_d, bgmmc.src_dt, b,
                                    m, k + gemm_batch * bgmmc.K_blk);
                            ctx.src = (void *)(src + src_off);
                            ctx.tr_src = (void *)(a_buffer
                                    + gemm_batch_iters * a_buffer_sz);
                            ctx.current_K_blk = K_tail;
                            ctx.current_M_blk = M_blk;
                            (*copy_A_kernel_)(&ctx);
                        }
                    }
                    ker(ithr, b, mb, nb, kc);
                }
            }
            ++start;
            nd_iterator_step(b, bgmmc.batch, mc, M_chunks, nc, N_chunks);
        }
    });
}

template struct brgemm_matmul_t<avx512_core_bf16_amx_int8>;

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
