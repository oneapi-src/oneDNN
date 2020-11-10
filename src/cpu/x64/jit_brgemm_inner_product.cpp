/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/jit_brgemm_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using namespace nstl;

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void brgemm_inner_product_fwd_t<src_type, wei_type, dst_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src_ = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights_ = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst_ = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    auto src = const_cast<src_data_t *>(src_);
    auto weights = const_cast<wei_data_t *>(weights_);
    auto dst = const_cast<dst_data_t *>(dst_);

    memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();
    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type)
            : 0;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const float *oscales = pd()->attr()->output_scales_.scales_;

    const auto &jbgp = pd()->jbgp_;

    src_data_t **addr_A_global = scratchpad.template get<src_data_t *>(
            key_brgemm_primitive_addr_a);
    wei_data_t **addr_B_global = scratchpad.template get<wei_data_t *>(
            key_brgemm_primitive_addr_b);
    char *c_buffer_global = (jbgp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;

    int ic_chunks = jbgp.nb_ic / jbgp.nb_ic_blocking;
    bool are_post_ops_applicable = one_of(true, jbgp.with_sum, jbgp.with_bias,
            jbgp.with_scales, jbgp.with_eltwise, jbgp.acc_dt != jbgp.dst_dt,
            jbgp.signed_input);

    size_t offset = weights_d.size() - weights_d.additional_buffer_size();
    auto w = const_cast<wei_data_t *>(weights);
    int32_t *compensation = (jbgp.signed_input)
            ? reinterpret_cast<int32_t *>(&w[offset])
            : nullptr;

    const auto ker = [&](const int ithr, int n, int ocb, int icc) {
        src_data_t **addr_A = addr_A_global + ithr * 16 * jbgp.gemm_batch_size;
        wei_data_t **addr_B = addr_B_global + ithr * 16 * jbgp.gemm_batch_size;

        char *c_buffer = (jbgp.use_buffer) ? c_buffer_global
                        + ithr * types::data_type_size(jbgp.acc_dt) * jbgp.LDC
                                * jbgp.M
                                           : nullptr;

        int oc = ocb * jbgp.oc_block;
        int icb = icc * jbgp.nb_ic_blocking;
        int ic = icb * jbgp.ic_block;

        bool kernel_init = (icc == 0);

        bool is_os_tail = (jbgp.mb - n < jbgp.os_block);
        bool is_oc_tail = (jbgp.oc - oc < jbgp.oc_block);
        bool is_ic_tail = (jbgp.ic - ic < jbgp.ic_block * jbgp.nb_ic_blocking);
        auto nb_ic_b = is_ic_tail ? (jbgp.ic - ic) / jbgp.ic_block
                                  : jbgp.nb_ic_blocking;

        auto brg_kernel = brg_kernels_[pd()->get_brg_kernel_idx(kernel_init,
                                               is_os_tail, is_oc_tail, false)]
                                  .get();

        if (nb_ic_b > 0 && brg_kernel != nullptr) {
            for (int ic_block = 0; ic_block < nb_ic_b; ic_block++) {
                addr_A[ic_block]
                        = src + src_d.blk_off(n, ic + ic_block * jbgp.ic_block);
                addr_B[ic_block]
                        = weights + weights_d.blk_off(ocb, icb + ic_block);
            }

            if (are_post_ops_applicable && icc == ic_chunks - 1
                    && !is_ic_tail) {
                char *ptr_C = (jbgp.use_buffer) ? c_buffer
                                                : (char *)dst
                                + sizeof(dst_data_t) * dst_d.blk_off(n, oc);
                auto ptr_D = dst + dst_d.blk_off(n, oc);
                auto bias_w
                        = jbgp.with_bias ? bias + bia_dt_size * oc : nullptr;
                brgemm_kernel_execute_postops(brg_kernel, nb_ic_b,
                        (void **)addr_A, (void **)addr_B, (void *)ptr_C,
                        (void *)ptr_D, (void *)bias_w,
                        &oscales[jbgp.is_oc_scale * oc],
                        jbgp.signed_input ? &compensation[oc] : nullptr);
            } else {
                char *ptr_C = (jbgp.use_buffer) ? c_buffer
                                                : (char *)dst
                                + sizeof(dst_data_t) * dst_d.blk_off(n, oc);
                brgemm_kernel_execute(brg_kernel, nb_ic_b, (void **)addr_A,
                        (void **)addr_B, (void *)ptr_C);
            }
        }

        if (is_ic_tail) {
            int ic_block = jbgp.nb_ic_blocking - 1;
            addr_A[0] = src + src_d.blk_off(n, ic + ic_block * jbgp.ic_block);
            addr_B[0] = weights + weights_d.blk_off(ocb, icb + ic_block);

            auto use_init_ker = (kernel_init && nb_ic_b == 0);
            auto brg_kernel_ic_tail
                    = brg_kernels_[pd()->get_brg_kernel_idx(use_init_ker,
                                           is_os_tail, is_oc_tail, true)]
                              .get();

            if (are_post_ops_applicable && icc == ic_chunks - 1) {
                char *ptr_C = (jbgp.use_buffer) ? c_buffer
                                                : (char *)dst
                                + sizeof(dst_data_t) * dst_d.blk_off(n, oc);
                auto ptr_D = dst + dst_d.blk_off(n, oc);
                auto bias_w
                        = jbgp.with_bias ? bias + bia_dt_size * oc : nullptr;
                brgemm_kernel_execute_postops(brg_kernel_ic_tail, 1,
                        (void **)addr_A, (void **)addr_B, (void *)ptr_C,
                        (void *)ptr_D, (void *)bias_w,
                        &oscales[jbgp.is_oc_scale * oc],
                        jbgp.signed_input ? &compensation[oc] : nullptr);
            } else {
                char *ptr_C = (jbgp.use_buffer) ? c_buffer
                                                : (char *)dst
                                + sizeof(dst_data_t) * dst_d.blk_off(n, oc);
                brgemm_kernel_execute(brg_kernel_ic_tail, 1, (void **)addr_A,
                        (void **)addr_B, (void *)ptr_C);
            }
        }
    };

    int os_chunks = jbgp.nb_os / jbgp.nb_os_blocking;
    int work_amount = jbgp.nb_oc * os_chunks;

    parallel(0, [&](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;

        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        int ocb {0}, oss {0};

        nd_iterator_init(start, oss, os_chunks, ocb, jbgp.nb_oc);
        while (start < end) {
            for_(int osb = 0; osb < jbgp.nb_os_blocking; osb++)
            for (int icc = 0; icc < ic_chunks; icc++) {
                int n = (oss * jbgp.nb_os_blocking + osb) * jbgp.os_block;
                ker(ithr, n, ocb, icc);
            }
            ++start;
            nd_iterator_step(oss, os_chunks, ocb, jbgp.nb_oc);
        }
    });
}

template struct brgemm_inner_product_fwd_t<bf16>;
template struct brgemm_inner_product_fwd_t<bf16, bf16, f32>;
template struct brgemm_inner_product_fwd_t<u8, s8, f32>;
template struct brgemm_inner_product_fwd_t<u8, s8, s32>;
template struct brgemm_inner_product_fwd_t<u8, s8, u8>;
template struct brgemm_inner_product_fwd_t<u8, s8, s8>;
template struct brgemm_inner_product_fwd_t<s8, s8, f32>;
template struct brgemm_inner_product_fwd_t<s8, s8, s32>;
template struct brgemm_inner_product_fwd_t<s8, s8, u8>;
template struct brgemm_inner_product_fwd_t<s8, s8, s8>;

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void brgemm_inner_product_bwd_data_t<src_type, wei_type,
        dst_type>::execute_backward_data(const exec_ctx_t &ctx) const {

    auto diff_dst_ = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights_ = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src_ = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    auto diff_src = const_cast<diff_src_data_t *>(diff_src_);
    auto weights = const_cast<wei_data_t *>(weights_);
    auto diff_dst = const_cast<diff_dst_data_t *>(diff_dst_);

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jbgp = pd()->jbgp_;

    memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();
    diff_dst_data_t **addr_A_global
            = scratchpad.template get<diff_dst_data_t *>(
                    key_brgemm_primitive_addr_a);
    wei_data_t **addr_B_global = scratchpad.template get<wei_data_t *>(
            key_brgemm_primitive_addr_b);
    char *c_buffer_global = (jbgp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;
    wei_data_t *b_buffer_global = jbgp.use_buffer_b
            ? scratchpad.template get<wei_data_t>(key_brgemm_primitive_buffer_b)
            : nullptr;

    int oc_chunks = jbgp.nb_oc / jbgp.nb_oc_blocking;

    const auto get_weights_ptr = [&](int icb, int ocb) {
        int fwd_ic_block = jbgp.simd_w;
        int fwd_oc_block = 0;
        switch (jbgp.wei_tag) {
            case OI16i64o:
            case OI8i64o2i: fwd_oc_block = 4 * jbgp.simd_w; break;
            case OI16i32o:
            case OI8i32o2i: fwd_oc_block = 2 * jbgp.simd_w; break;
            default: fwd_oc_block = jbgp.simd_w;
        };
        int fwd_icb = icb * jbgp.ic_block / fwd_ic_block;
        int fwd_ocb = ocb * jbgp.oc_block / fwd_oc_block;
        wei_data_t *ptr_wei_local
                = weights + weights_d.blk_off(fwd_ocb, fwd_icb);

        int fwd_ocb_simd = (ocb * jbgp.oc_block) % fwd_oc_block;
        int fwd_icb_simd = (icb * jbgp.ic_block) % fwd_ic_block;
        int blk_sz = jbgp.wei_dt == data_type::bf16 ? 2 : 1;

        return ptr_wei_local + fwd_icb_simd / blk_sz * blk_sz * fwd_oc_block
                + blk_sz * fwd_ocb_simd;
    };

    const auto transform_b_chunk
            = [&](wei_data_t *tr_wei, const wei_data_t *wei, int trans_batch,
                      int current_N, int current_K) {
                  auto ctx = jit_brgemm_trans_wei_t::ctx_t();
                  ctx.src = (void *)wei;
                  ctx.tr_src = (void *)tr_wei;
                  ctx.current_gemm_batch = trans_batch;
                  ctx.current_N = current_N;
                  ctx.current_K = current_K;
                  (*trans_B_kernel_)(&ctx);
              };

    const auto ker = [&](const int ithr, int n, int icb, int occ) {
        diff_dst_data_t **addr_A
                = addr_A_global + ithr * 16 * jbgp.gemm_batch_size;
        wei_data_t **addr_B = addr_B_global + ithr * 16 * jbgp.gemm_batch_size;

        char *c_buffer = (jbgp.use_buffer) ? c_buffer_global
                        + ithr * types::data_type_size(jbgp.acc_dt) * jbgp.LDC
                                * jbgp.M
                                           : nullptr;

        int ic = icb * jbgp.ic_block;
        int ocb = occ * jbgp.nb_oc_blocking;
        int oc = ocb * jbgp.oc_block;

        bool kernel_init = (occ == 0);

        bool is_os_tail = (jbgp.mb - n < jbgp.os_block);
        bool is_ic_tail = (jbgp.ic - ic < jbgp.ic_block);
        bool is_oc_tail = (jbgp.oc - oc < jbgp.oc_block * jbgp.nb_oc_blocking);

        auto nb_oc_b = is_oc_tail ? (jbgp.oc - oc) / jbgp.oc_block
                                  : jbgp.nb_oc_blocking;

        auto brg_kernel = brg_kernels_[pd()->get_brg_kernel_idx(kernel_init,
                                               is_os_tail, is_ic_tail, false)]
                                  .get();

        const int size_B = jbgp.LDB * rnd_up(jbgp.K, 2);
#ifndef BRGEMM_IP_BWD_D_GLOBAL_B_TRANSPOSE
        wei_data_t *b_buffer
                = b_buffer_global + ithr * jbgp.gemm_batch_size * size_B;
#else
        wei_data_t *b_buffer = b_buffer_global
                + (dim_t)icb * jbgp.nb_oc * size_B + (dim_t)ocb * size_B;
#endif

        if (nb_oc_b > 0 && brg_kernel != nullptr) {
            for (int oc_block = 0; oc_block < nb_oc_b; oc_block++) {

                addr_A[oc_block] = diff_dst
                        + diff_dst_d.blk_off(n, oc + oc_block * jbgp.oc_block);
                addr_B[oc_block] = b_buffer + oc_block * size_B;
#ifndef BRGEMM_IP_BWD_D_GLOBAL_B_TRANSPOSE
                transform_b_chunk(addr_B[oc_block],
                        get_weights_ptr(icb, ocb + oc_block), 1,
                        is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block,
                        jbgp.oc_block);
#endif
            }

            if (jbgp.use_buffer && occ == oc_chunks - 1 && !is_oc_tail) {
                auto ptr_D = diff_src + diff_src_d.blk_off(n, ic);
                brgemm_kernel_execute_postops(brg_kernel, nb_oc_b,
                        (void **)addr_A, (void **)addr_B, (void *)c_buffer,
                        (void *)ptr_D, nullptr, nullptr);
            } else {
                char *ptr_C = (jbgp.use_buffer) ? c_buffer
                                                : (char *)diff_src
                                + sizeof(diff_src_data_t)
                                        * diff_src_d.blk_off(n, ic);
                brgemm_kernel_execute(brg_kernel, nb_oc_b, (void **)addr_A,
                        (void **)addr_B, (void *)ptr_C);
            }
        }
        if (is_oc_tail) {
            int oc_block = jbgp.nb_oc_blocking - 1;
            addr_A[0] = diff_dst
                    + diff_dst_d.blk_off(n, oc + oc_block * jbgp.oc_block);
#ifndef BRGEMM_IP_BWD_D_GLOBAL_B_TRANSPOSE
            addr_B[0] = b_buffer;
            transform_b_chunk(addr_B[0], get_weights_ptr(icb, ocb + oc_block),
                    1, is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block,
                    jbgp.K_tail);
#else
            addr_B[0] = b_buffer + oc_block * size_B;
#endif

            auto use_init_ker = (kernel_init && nb_oc_b == 0);
            auto brg_kernel_oc_tail
                    = brg_kernels_[pd()->get_brg_kernel_idx(use_init_ker,
                                           is_os_tail, is_ic_tail, true)]
                              .get();

            if (jbgp.use_buffer && occ == oc_chunks - 1) {
                auto ptr_D = diff_src + diff_src_d.blk_off(n, ic);
                brgemm_kernel_execute_postops(brg_kernel_oc_tail, 1,
                        (void **)addr_A, (void **)addr_B, (void *)c_buffer,
                        (void *)ptr_D, nullptr, nullptr);
            } else {
                char *ptr_C = (jbgp.use_buffer) ? c_buffer
                                                : (char *)diff_src
                                + sizeof(diff_src_data_t)
                                        * diff_src_d.blk_off(n, ic);
                brgemm_kernel_execute(brg_kernel_oc_tail, 1, (void **)addr_A,
                        (void **)addr_B, (void *)ptr_C);
            }
        }
    };

    int os_chunks = jbgp.nb_os / jbgp.nb_os_blocking;
    int work_amount = jbgp.nb_ic * os_chunks;
#if defined(BRGEMM_IP_BWD_D_GLOBAL_B_TRANSPOSE)
    if (jbgp.use_buffer_b) {
        parallel(0, [&](const int ithr, const int nthr) {
            int start {0}, end {0};
            int max_ch_block = nstl::max(jbgp.ic_block, jbgp.oc_block);
            int ic_chunk_sz = max_ch_block / jbgp.ic_block;
            int oc_chunk_sz = max_ch_block / jbgp.oc_block;
            int nc_ic = utils::div_up(jbgp.nb_ic, ic_chunk_sz);
            int nc_oc = utils::div_up(jbgp.nb_oc, oc_chunk_sz);
            int transp_work_amount = nc_ic * nc_oc;
            balance211(transp_work_amount, nthr, ithr, start, end);
            int icc, occ;
            nd_iterator_init(start, icc, nc_ic, occ, nc_oc);
            while (start < end) {
                int icb_start = icc * ic_chunk_sz;
                int icb_end = nstl::min((icc + 1) * ic_chunk_sz, jbgp.nb_ic);
                int ocb_start = occ * oc_chunk_sz;
                int ocb_end = nstl::min((occ + 1) * oc_chunk_sz, jbgp.nb_oc);
                for_(int icb = icb_start; icb < icb_end; icb++)
                for (int ocb = ocb_start; ocb < ocb_end; ocb++) {
                    int ic = icb * jbgp.ic_block;
                    int oc = ocb * jbgp.oc_block;
                    bool is_ic_tail = (jbgp.ic - ic < jbgp.ic_block);
                    bool is_oc_tail = (jbgp.oc - oc < jbgp.oc_block);
                    const int size_B = jbgp.LDB * rnd_up(jbgp.K, 2);
                    wei_data_t *b_buffer = b_buffer_global
                            + (dim_t)icb * jbgp.nb_oc * size_B
                            + (dim_t)ocb * size_B;

                    transform_b_chunk(b_buffer, get_weights_ptr(icb, ocb), 1,
                            is_ic_tail ? jbgp.ic % jbgp.ic_block
                                       : jbgp.ic_block,
                            is_oc_tail ? jbgp.oc % jbgp.oc_block
                                       : jbgp.oc_block);
                }
                ++start;
                nd_iterator_step(icc, nc_ic, occ, nc_oc);
            }
        });
    }
#endif

    parallel(0, [&](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;

        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        int icb {0}, oss {0};

        nd_iterator_init(start, oss, os_chunks, icb, jbgp.nb_ic);
        while (start < end) {
            for_(int osb = 0; osb < jbgp.nb_os_blocking; osb++)
            for (int occ = 0; occ < oc_chunks; occ++) {
                int n = (oss * jbgp.nb_os_blocking + osb) * jbgp.os_block;
                ker(ithr, n, icb, occ);
            }
            ++start;
            nd_iterator_step(oss, os_chunks, icb, jbgp.nb_ic);
        }
    });
}

template struct brgemm_inner_product_bwd_data_t<bf16>;
template struct brgemm_inner_product_bwd_data_t<f32, bf16, bf16>;

template <data_type_t src_type, data_type_t diff_wei_type,
        data_type_t diff_dst_type>
struct brgemm_inner_product_bwd_weights_t<src_type, diff_wei_type,
        diff_dst_type>::thread_info_t {
    const src_data_t *src;
    const diff_dst_data_t *diff_dst;
    diff_wei_data_t *diff_weights;
    char *diff_bias;

    const memory_tracking::grantor_t scratchpad;

    src_data_t *buffer_a = nullptr;
    diff_dst_data_t *buffer_b = nullptr;
    char *buffer_c = nullptr;
    char *buffer_bias = nullptr;

    int ithr;
    int ithr_ic_c, ithr_oc_c, ithr_os_c;
    int nthr;
    int nthr_ic_c, nthr_oc_c, nthr_os_c;

    int os_c_start = 0, os_c_end = 0, os_c_work;
    int oc_c_start = 0, oc_c_end = 0, oc_c_work;
    int ic_c_start = 0, ic_c_end = 0, ic_c_work;
    simple_barrier::ctx_t *barrier_ctx;

    thread_info_t(const brgemm_inner_product_bwd_weights_t *self,
            const exec_ctx_t &ctx, int ithr)
        : scratchpad(ctx.get_scratchpad_grantor()), ithr(ithr) {
        src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
        diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
        diff_weights = CTX_OUT_MEM(diff_wei_data_t *, DNNL_ARG_DIFF_WEIGHTS);
        diff_bias = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_BIAS);
        const auto &jbgp = self->pd()->jbgp_;

        buffer_c = (jbgp.use_buffer)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
                : nullptr;

        buffer_bias = (jbgp.with_bias
                              && (jbgp.bia_dt == data_type::bf16
                                      || jbgp.nthr_mb > 1))
                ? scratchpad.template get<char>(key_iprod_bias_bf16_convert_wsp)
                : nullptr;

        buffer_a = scratchpad.template get<src_data_t>(
                key_brgemm_primitive_buffer_a);
        buffer_b = jbgp.use_buffer_b ? scratchpad.template get<diff_dst_data_t>(
                           key_brgemm_primitive_buffer_b)
                                     : nullptr;

        nthr = jbgp.nthr;
        nthr_ic_c = jbgp.nthr_ic_b;
        nthr_oc_c = jbgp.nthr_oc_b;
        nthr_os_c = jbgp.nthr_mb;

        ithr_ic_c = ithr % nthr_ic_c;
        ithr_oc_c = ithr / nthr_ic_c % nthr_oc_c;
        ithr_os_c = ithr / nthr_ic_c / nthr_oc_c;

        int oc_chunks = jbgp.nb_oc / jbgp.nb_oc_blocking;
        int ic_chunks = jbgp.nb_ic / jbgp.nb_ic_blocking;
        int os_chunks = utils::div_up(jbgp.nb_os, jbgp.nb_os_blocking);

        /* reduction dimension */
        balance211(os_chunks, nthr_os_c, ithr_os_c, os_c_start, os_c_end);
        os_c_work = os_c_end - os_c_start;

        balance211(oc_chunks, nthr_oc_c, ithr_oc_c, oc_c_start, oc_c_end);
        oc_c_work = oc_c_end - oc_c_start;

        balance211(ic_chunks, nthr_ic_c, ithr_ic_c, ic_c_start, ic_c_end);
        ic_c_work = ic_c_end - ic_c_start;

        if (dnnl_thr_syncable())
            barrier_ctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_wei_bia_reduction_bctx);
    }
};

template <data_type_t src_type, data_type_t diff_wei_type,
        data_type_t diff_dst_type>
void brgemm_inner_product_bwd_weights_t<src_type, diff_wei_type,
        diff_dst_type>::transform_matrix_a_chunk(src_data_t *tr_src,
        const src_data_t *src, int trans_batch, int current_m,
        int current_k) const {
    auto ctx = jit_brgemm_trans_src_t::ctx_t();
    ctx.src = (void *)src;
    ctx.tr_src = (void *)tr_src;
    ctx.current_gemm_batch = trans_batch;
    ctx.current_M = current_m;
    ctx.current_K = current_k;
    (*trans_A_kernel_)(&ctx);
}

template <data_type_t src_type, data_type_t diff_wei_type,
        data_type_t diff_dst_type>
void brgemm_inner_product_bwd_weights_t<src_type, diff_wei_type,
        diff_dst_type>::transform_matrix_b_chunk(diff_dst_data_t *tr_diff_dst,
        const diff_dst_data_t *diff_dst, int trans_batch, int current_col_size,
        int current_row_size) const {
    auto ctx = jit_brgemm_trans_to_vnni_t::ctx_t();
    ctx.src = (void *)diff_dst;
    ctx.tr_src = (void *)tr_diff_dst;
    ctx.current_gemm_batch = trans_batch;
    ctx.current_col_size = current_col_size;
    ctx.current_row_size = current_row_size;
    (*trans_B_kernel_)(&ctx);
}

template <data_type_t src_type, data_type_t diff_wei_type,
        data_type_t diff_dst_type>
void brgemm_inner_product_bwd_weights_t<src_type, diff_wei_type,
        diff_dst_type>::compute_diff_weights_and_bias(const thread_info_t *ti)
        const {
    auto diff_dst = const_cast<diff_dst_data_t *>(ti->diff_dst);
    auto diff_weights = const_cast<diff_wei_data_t *>(ti->diff_weights);
    auto diff_bias = ti->diff_bias;

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jbgp = pd()->jbgp_;

    const size_t bia_dt_size
            = jbgp.with_bias ? types::data_type_size(jbgp.bia_dt) : 0;
    const size_t acc_dt_size = types::data_type_size(jbgp.acc_dt);

    src_data_t **addr_A_global = ti->scratchpad.template get<src_data_t *>(
            key_brgemm_primitive_addr_a);
    diff_dst_data_t **addr_B_global
            = ti->scratchpad.template get<diff_dst_data_t *>(
                    key_brgemm_primitive_addr_b);

    src_data_t *a_buffer_global = ti->buffer_a;
    diff_dst_data_t *b_buffer_global = ti->buffer_b;
    int os_chunks = utils::div_up(jbgp.nb_os, jbgp.nb_os_blocking);

    const auto get_wei_acc_ptr = [&](int ocb, int icb) {
        const int reduction_buf_start_idx = jbgp.wei_dt == f32;
        if (jbgp.use_buffer && jbgp.nthr_mb == 1) {
            UNUSED(icb);
            UNUSED(ocb);
            return ti->buffer_c + acc_dt_size * ti->ithr * jbgp.LDC * jbgp.M;
        } else if (jbgp.use_buffer && jbgp.nthr_mb > 1
                && ti->ithr_os_c >= reduction_buf_start_idx) {
            const size_t red_buf_elems = (size_t)jbgp.nb_ic * jbgp.ic_block
                    * jbgp.nb_oc * jbgp.oc_block;
            return ti->buffer_c
                    + acc_dt_size * (ti->ithr_os_c - reduction_buf_start_idx)
                    * red_buf_elems
                    + acc_dt_size * diff_weights_d.blk_off(ocb, icb);
        } else {
            return (char *)diff_weights
                    + acc_dt_size * diff_weights_d.blk_off(ocb, icb);
        }
    };

    const auto get_bia_acc_ptr = [&](int oc) {
        const int reduction_buf_start_idx = jbgp.bia_dt == f32;
        if (jbgp.bia_dt == data_type::bf16
                || (jbgp.nthr_mb > 1
                        && ti->ithr_os_c >= reduction_buf_start_idx)) {
            return ti->buffer_bias
                    + acc_dt_size * (ti->ithr_os_c - reduction_buf_start_idx)
                    * jbgp.oc
                    + acc_dt_size * oc;
        } else {
            return ti->diff_bias + bia_dt_size * oc;
        }
    };

    const auto ker = [&](const int osc, const int icb, const int ocb) {
        int os_chunks_per_thr = utils::div_up(os_chunks, jbgp.nthr_mb);
        int ic_chunks = utils::div_up(jbgp.nb_ic, jbgp.nb_ic_blocking);
        int ic_chunks_per_thr = utils::div_up(ic_chunks, jbgp.nthr_ic_b);
        int osc_l_idx = osc - ti->os_c_start;
        int icb_l_idx = icb - ti->ic_c_start * jbgp.nb_ic_blocking;
        int ocb_l_idx = ocb - ti->oc_c_start * jbgp.nb_oc_blocking;
        int a_buf_idx = osc_l_idx * ic_chunks_per_thr * jbgp.nb_ic_blocking
                + icb_l_idx;
        int b_buf_idx = osc_l_idx;

        src_data_t **addr_A
                = addr_A_global + ti->ithr * 16 * jbgp.gemm_batch_size;
        diff_dst_data_t **addr_B
                = addr_B_global + ti->ithr * 16 * jbgp.gemm_batch_size;
        const int size_A = jbgp.LDA * jbgp.M;
        const int size_B = jbgp.LDB * rnd_up(jbgp.K, 2);
        src_data_t *a_buffer = a_buffer_global
                + (ti->ithr * os_chunks_per_thr * ic_chunks_per_thr
                                  * jbgp.nb_ic_blocking
                          + a_buf_idx)
                        * jbgp.gemm_batch_size * jbgp.os_block * jbgp.ic_block;
        diff_dst_data_t *b_buffer = b_buffer_global
                + (ti->ithr * os_chunks_per_thr + b_buf_idx)
                        * jbgp.gemm_batch_size * jbgp.os_block * jbgp.oc_block;

        int ic = icb * jbgp.ic_block;
        int oc = ocb * jbgp.oc_block;
        int n = osc * jbgp.nb_os_blocking * jbgp.os_block;

        bool kernel_init = (osc == ti->os_c_start);

        bool is_os_tail = (jbgp.mb - n < jbgp.os_block * jbgp.nb_os_blocking);
        bool is_ic_tail = (jbgp.ic - ic < jbgp.ic_block);
        bool is_oc_tail = (jbgp.oc - oc < jbgp.oc_block);

        const bool transform_weights_to_vnni = jbgp.wei_dt == bf16
                && (jbgp.nthr_mb == 1 || os_chunks == 1)
                && osc == (os_chunks - 1);

        auto nb_os_b = is_os_tail ? (jbgp.mb - n) / jbgp.os_block
                                  : jbgp.nb_os_blocking;

        auto brg_kernel = brg_kernels_[pd()->get_brg_kernel_idx(kernel_init,
                                               is_ic_tail, is_oc_tail, false)]
                                  .get();

        if (kernel_init && (is_ic_tail || is_oc_tail))
            utils::array_set(get_wei_acc_ptr(ocb, icb), 0,
                    types::data_type_size(jbgp.acc_dt) * jbgp.ic_block
                            * jbgp.oc_block);
        if (nb_os_b > 0 && brg_kernel != nullptr) {
            if (jbgp.use_buffer_a && ocb_l_idx == 0) {
                const memory_desc_wrapper src_d(pd()->src_md());
                auto src_ptr = ti->src + src_d.blk_off(n, ic);
                transform_matrix_a_chunk(a_buffer, src_ptr, nb_os_b,
                        is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block,
                        jbgp.os_block);
            }

            if (jbgp.use_buffer_b && icb_l_idx == 0) {
                auto diff_dst_ptr = diff_dst + diff_dst_d.blk_off(n, oc);
                transform_matrix_b_chunk(b_buffer, diff_dst_ptr, nb_os_b,
                        is_oc_tail ? jbgp.oc % jbgp.oc_block : jbgp.oc_block,
                        jbgp.os_block);
            }

            for (int os_block = 0; os_block < nb_os_b; os_block++) {
                auto a_ptr = a_buffer + os_block * size_A;
                addr_A[os_block] = a_ptr;
                auto diff_dst_ptr = diff_dst
                        + diff_dst_d.blk_off(n + os_block * jbgp.os_block, oc);
                if (jbgp.use_buffer_b) {
                    auto b_ptr = b_buffer + os_block * size_B;
                    addr_B[os_block] = b_ptr;
                } else {
                    addr_B[os_block] = diff_dst_ptr;
                }
                if (jbgp.with_bias && icb == 0) {
                    brgemm_kernel_diff_bias_t p;
                    auto bias_ptr = diff_bias + bia_dt_size * oc;
                    p.ptr_diff_dst = (void *)addr_B[os_block];
                    p.ptr_diff_bias_acc = (void *)get_bia_acc_ptr(oc);
                    p.ptr_diff_bias = (void *)bias_ptr;
                    bool is_first = kernel_init && os_block == 0;
                    bool is_last = (jbgp.nthr_mb == 1 || os_chunks == 1)
                            && osc == os_chunks - 1 && os_block == nb_os_b - 1
                            && !is_os_tail;
                    p.flags = 0 | (is_first ? FLAG_REDUCE_FIRST : 0)
                            | (is_last ? FLAG_REDUCE_LAST : 0);

                    (*kernels_db_[false][is_oc_tail])(&p);
                }
            }
            brgemm_kernel_execute(brg_kernel, nb_os_b, (void **)addr_A,
                    (void **)addr_B, (void *)get_wei_acc_ptr(ocb, icb));
        }

        if (is_os_tail) {
            int os_block = nb_os_b;
            auto a_ptr = &a_buffer[os_block * jbgp.ic_block * jbgp.os_block];
            if (jbgp.use_buffer_a && ocb_l_idx == 0) {
                const memory_desc_wrapper src_d(pd()->src_md());
                auto src_ptr = ti->src
                        + src_d.blk_off(n + os_block * jbgp.os_block, ic);
                transform_matrix_a_chunk(a_ptr, src_ptr, 1,
                        is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block,
                        jbgp.mb % jbgp.os_block);
            }

            addr_A[0] = a_ptr;
            auto diff_dst_ptr = diff_dst
                    + diff_dst_d.blk_off(n + os_block * jbgp.os_block, oc);
            if (jbgp.use_buffer_b) {
                auto b_ptr
                        = &b_buffer[os_block * jbgp.oc_block * jbgp.os_block];
                if (icb_l_idx == 0)
                    transform_matrix_b_chunk(b_ptr, diff_dst_ptr, 1,
                            is_oc_tail ? jbgp.oc % jbgp.oc_block
                                       : jbgp.oc_block,
                            jbgp.mb % jbgp.os_block);

                addr_B[0] = b_ptr;
            } else {
                addr_B[0] = diff_dst_ptr;
            }

            if (jbgp.with_bias && icb == 0) {
                brgemm_kernel_diff_bias_t p;
                auto bias_ptr = diff_bias + bia_dt_size * oc;
                p.ptr_diff_dst = (void *)addr_B[0];
                p.ptr_diff_bias_acc = (void *)get_bia_acc_ptr(oc);
                p.ptr_diff_bias = (void *)bias_ptr;
                bool is_first = kernel_init && os_block == 0;
                bool is_last = (jbgp.nthr_mb == 1 || os_chunks == 1)
                        && osc == os_chunks - 1;
                p.flags = 0 | (is_first ? FLAG_REDUCE_FIRST : 0)
                        | (is_last ? FLAG_REDUCE_LAST : 0);

                (*kernels_db_[true][is_oc_tail])(&p);
            }

            auto use_init_ker = (kernel_init && nb_os_b == 0);
            auto brg_kernel_os_tail
                    = brg_kernels_[pd()->get_brg_kernel_idx(use_init_ker,
                                           is_ic_tail, is_oc_tail, true)]
                              .get();

            if (brg_kernel_os_tail != nullptr)
                brgemm_kernel_execute(brg_kernel_os_tail, 1, (void **)addr_A,
                        (void **)addr_B, (void *)get_wei_acc_ptr(ocb, icb));
        }

        if (transform_weights_to_vnni) {
            auto wei_ptr = diff_weights + diff_weights_d.blk_off(ocb, icb);
            auto ctx = jit_brgemm_trans_to_vnni_t::ctx_t();
            ctx.src = (void *)get_wei_acc_ptr(ocb, icb);
            ctx.tr_src = (void *)wei_ptr;
            ctx.current_gemm_batch = 1;
            ctx.current_col_size
                    = is_oc_tail ? jbgp.oc % jbgp.oc_block : jbgp.oc_block;
            ctx.current_row_size
                    = is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block;
            (*trans_C_kernel_)(&ctx);
        }
    };

    for_(int occ = ti->oc_c_start; occ < ti->oc_c_end; occ++)
    for_(int icc = ti->ic_c_start; icc < ti->ic_c_end; icc++)
    for_(int ocb = 0; ocb < jbgp.nb_oc_blocking; ocb++)
    for_(int icb = 0; icb < jbgp.nb_ic_blocking; icb++)
    for (int osc = ti->os_c_start; osc < ti->os_c_end; osc++) {
        ker(osc, icc * jbgp.nb_ic_blocking + icb,
                occ * jbgp.nb_oc_blocking + ocb);
    }
}

template <data_type_t src_type, data_type_t diff_wei_type,
        data_type_t diff_dst_type>
void brgemm_inner_product_bwd_weights_t<src_type, diff_wei_type,
        diff_dst_type>::
        reduce_and_convert_diff_weights_and_bias(
                const thread_info_t *ti) const {
    const auto &jbgp = pd()->jbgp_;

    if (dnnl_thr_syncable() && jbgp.nthr > 1)
        simple_barrier::barrier(ti->barrier_ctx, jbgp.nthr);

    if (ti->nthr_os_c == 1) return;
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const bool is_bf16_out = diff_weights_d.data_type() == data_type::bf16;
    const int icb_work = ti->ic_c_work * jbgp.nb_ic_blocking;
    const int ocb_work = ti->oc_c_work * jbgp.nb_oc_blocking;
    const int work = ocb_work * icb_work;

    int start {0}, end {0};
    balance211(work, ti->nthr_os_c, ti->ithr_os_c, start, end);
    if (start == end) return;

    int icb_l, ocb_l;
    int os_chunks = utils::div_up(jbgp.nb_os, jbgp.nb_os_blocking);
    int reduce_buffers = nstl::min(ti->nthr_os_c, os_chunks);
    if (reduce_buffers == 1) return;
    float *wei_reduced
            = is_bf16_out ? (float *)ti->buffer_c : (float *)ti->diff_weights;
    const size_t red_buf_elems
            = (size_t)jbgp.nb_ic * jbgp.ic_block * jbgp.nb_oc * jbgp.oc_block;

    int reduce_buf_idx_start = is_bf16_out;
    int reduce_buf_idx_end = reduce_buffers - !is_bf16_out;
    const int acc_size = jbgp.ic_block * jbgp.oc_block;
    for (int ir = reduce_buf_idx_start; ir < reduce_buf_idx_end; ++ir) {
        float *wei_to_reduce = (float *)ti->buffer_c + ir * red_buf_elems;

        int counter = start;
        nd_iterator_init(start, ocb_l, ocb_work, icb_l, icb_work);
        while (counter < end) {
            const int ocb = ti->oc_c_start * jbgp.nb_oc_blocking + ocb_l;
            const int icb = ti->ic_c_start * jbgp.nb_ic_blocking + icb_l;
            acc_ker_->accumulate(&wei_reduced[diff_weights_d.blk_off(ocb, icb)],
                    &wei_to_reduce[diff_weights_d.blk_off(ocb, icb)], acc_size);

            if (is_bf16_out && ir + 1 == reduce_buf_idx_end) {
                diff_wei_data_t *out_wei_ptr
                        = (diff_wei_data_t *)ti->diff_weights;
                auto ctx = jit_brgemm_trans_to_vnni_t::ctx_t();
                ctx.src = (void *)&wei_reduced[diff_weights_d.blk_off(
                        ocb, icb)];
                ctx.tr_src = (void *)&out_wei_ptr[diff_weights_d.blk_off(
                        ocb, icb)];
                ctx.current_gemm_batch = 1;
                ctx.current_col_size = jbgp.oc_block;
                ctx.current_row_size = jbgp.ic_block;
                (*trans_C_kernel_)(&ctx);
            }

            ++counter;
            nd_iterator_step(ocb_l, ocb_work, icb_l, icb_work);
        }
    }

    if (jbgp.with_bias && ti->ithr_ic_c == 0 && ti->ic_c_work > 0
            && ti->ithr_os_c == 0 && ti->os_c_work > 0 && ti->oc_c_work > 0) {
        const bool is_bf16_bias = jbgp.bia_dt == data_type::bf16;
        float *bias_reduced = is_bf16_bias ? (float *)ti->buffer_bias
                                           : (float *)ti->diff_bias;
        int reduce_buf_idx_start = is_bf16_bias;
        int reduce_buf_idx_end = reduce_buffers - 1;
        int oc_chunk_size = jbgp.nb_oc_blocking * jbgp.oc_block;
        int oc = ti->oc_c_start * oc_chunk_size;
        int acc_size = nstl::min(ti->oc_c_work * oc_chunk_size, jbgp.oc - oc);

        int ir = reduce_buf_idx_start;
        for (; ir < reduce_buf_idx_end; ++ir) {
            float *bias_to_reduce = (float *)ti->buffer_bias + ir * jbgp.oc;
            acc_ker_->accumulate(
                    &bias_reduced[oc], &bias_to_reduce[oc], acc_size);
        }

        if (is_bf16_bias) {
            float *bias_to_reduce = (float *)ti->buffer_bias + ir * jbgp.oc;
            add_floats_and_cvt_to_bfloat16((bfloat16_t *)(ti->diff_bias) + oc,
                    &bias_reduced[oc], &bias_to_reduce[oc], acc_size);
        }
    }
}

template <data_type_t src_type, data_type_t diff_wei_type,
        data_type_t diff_dst_type>
void brgemm_inner_product_bwd_weights_t<src_type, diff_wei_type,
        diff_dst_type>::execute_backward_weights(const exec_ctx_t &ctx) const {
    const auto &jbgp = pd()->jbgp_;
    if (dnnl_thr_syncable() && jbgp.nthr > 1) {
        auto scratchpad = ctx.get_scratchpad_grantor();
        simple_barrier::ctx_init(scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx));
    }

    parallel(jbgp.nthr, [&](const int ithr, const int nthr) {
        thread_info_t thread_info(this, ctx, ithr);
        compute_diff_weights_and_bias(&thread_info);

        if (dnnl_thr_syncable()) {
            reduce_and_convert_diff_weights_and_bias(&thread_info);
        }
    });

    if (!dnnl_thr_syncable()) {
        parallel(jbgp.nthr, [&](const int ithr, const int nthr) {
            thread_info_t thread_info(this, ctx, ithr);
            reduce_and_convert_diff_weights_and_bias(&thread_info);
        });
    }
}

template struct brgemm_inner_product_bwd_weights_t<bf16>;
template struct brgemm_inner_product_bwd_weights_t<bf16, f32, bf16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
