/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <cstdint>
#include <mutex>

#include "gemm_info.hpp"

#include "cpu_isa_traits.hpp"
#include "jit_generator.hpp"
#include "mkldnn_traits.hpp"
#include "mkldnn_types.h"
#include "bf16/common_s16.hpp"
#include "bf16/jit_avx512_core_gemm_bf16bf16f32_kern.hpp"
#include "common/bfloat16.hpp"
#include "f32/common_f32.hpp"
#include "f32/jit_avx2_kernel_sgemm_kern.hpp"
#include "f32/jit_avx_gemv_t_f32_kern.hpp"
#include "f32/jit_sse41_gemv_t_f32_kern.hpp"
#include "s8x8s32/common_u8.hpp"
#include "s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.hpp"
#include "s8x8s32/jit_avx512_core_kernel_gemv_s8u8s32_kern.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

static inline int decode_trans(char trans)
{
    switch (trans) {
        case 'T':
        case 't':
            return do_trans;
        case 'P':
        case 'p':
            return packed;
        default:
            return no_trans;
    }
}

namespace {
template <typename b_type>
void prepare_bo(uint8_t &bo_gemm_info, const b_type *bo_orig) {
    UNUSED(bo_orig);
    bo_gemm_info = 0;
}
template <> void prepare_bo(uint8_t &bo_gemm_info, const uint8_t *bo_orig) {
    bo_gemm_info = bo_orig ? *bo_orig : 0;
}
template <> void prepare_bo(uint8_t &bo_gemm_info, const int8_t *bo_orig) {
    int bo_s32 = bo_orig ? *bo_orig : 0;
    bo_gemm_info = (uint8_t)(bo_s32 + 128);
}
}

template <typename a_type, typename b_type, typename c_type>
gemm_info_t<a_type, b_type, c_type>::gemm_info_t(const char *transA,
        const char *transB, const char *offsetC, const int *m, const int *n,
        const int *k, const float *alpha, const a_type *a, const int *lda,
        const a_type *oa, const b_type *b, const int *ldb, const b_type *ob,
        const float *beta, c_type *c, const int *ldc, const c_type *oc,
        bool force_nocopy, pack_type packing, gemm_pack_storage_t *pack_dst,
        bool measure_only) {

    this->transa = decode_trans(*transA);
    this->transb = decode_trans(*transB);

    this->m = *m;
    this->n = *n;
    this->k = *k;

    this->a = a;
    this->b = b;
    this->c = c;

    this->lda = lda ? *lda : 0;
    this->ldb = ldb ? *ldb : 0;
    this->ldc = ldc ? *ldc : 0;

    this->ao = 0;
    this->bo = 0;
    this->co = NULL;

    this->alpha = alpha ? *alpha : 1.0f;
    this->beta = beta ? *beta : 1.0f;

    this->offsetc = offset_type::none;

    this->packing = packing;
    this->pack_dst = pack_dst;
    this->measure_only = measure_only && pack_dst
        && (packing != pack_type::none);

    if (this->transa == packed) {
        dim_t cols;

        this->a_packed.reset(new gemm_pack_storage_t(a));
        if (this->a_packed->get_nocopy(this->lda, cols)) {
            this->a = this->a_packed->template matrix<a_type>();
            this->transa = no_trans;
            this->a_packed = nullptr;
        }
    }
    if (this->transb == packed) {
        dim_t rows;

        this->b_packed.reset(new gemm_pack_storage_t(b));
        if (this->b_packed->get_nocopy(this->ldb, rows)) {
            this->b = this->b_packed->template matrix<b_type>();
            this->transb = no_trans;
            this->b_packed = nullptr;
        }
    }

    if (data_traits<a_type>::data_type == data_type::s8)
        this->ao = oa ? *oa : a_type(0);
    prepare_bo<b_type>(this->bo, ob);

    if (offsetC != NULL) {
        char offsetc = *offsetC;
        if (offsetc == 'F' || offsetc == 'f') {
            this->offsetc = offset_type::fixed;
        } else if (offsetc == 'R' || offsetc == 'r') {
            this->offsetc = offset_type::row;
        } else { // offsetc == 'C' || offsetc == 'c'
            this->offsetc = offset_type::column;
        }
        this->co = oc;
    }

    bool is_sgemm = data_traits<a_type>::data_type == data_type::f32;
    bool is_gemv = this->m == 1 || this->n == 1;

    // Copy-based sgemm doesn't support force-nocopy for ISAs older
    // than Intel AVX.
    this->force_nocopy = is_sgemm && force_nocopy && mayiuse(avx);
    this->force_nocopy |= mayiuse(avx512_mic);

    if (!this->force_nocopy || is_gemv) {
        this->jit_init();
    }
}

template<typename a_type, typename b_type, typename c_type>
void gemm_info_t<a_type, b_type, c_type>::jit_init(void) {

    // copyA[trans][sum]
    static void (*copyA[2][2])(const dim_t *m, const dim_t *n,
            const a_type *src, const dim_t *ldsrc, const float *alpha,
            a_type *dst, const dim_t *dummy1, const dim_t *dummy2,
            c_type *row_col_sum) = {{NULL}};

    // copyB[trans][sum]
    static void (*copyB[2][2])(const dim_t *m, const dim_t *n,
            const b_type *src, const dim_t *ldsrc, const float *alpha,
            b_type *dst, const dim_t *dummy1, const dim_t *dummy2,
            c_type *row_col_sum) = {{NULL}};

    // kern[beta0][col_off][row_off]
    static void (*kern[2][2][2])(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const a_type *a, const b_type *b, c_type *c,
            const dim_t ldc, const c_type *col_offset,
            const c_type *row_offset) = {{{NULL}}};

    // gemv_kern[trans]
    static void (*gemv_kern[2])(const dim_t *m, const dim_t *n,
            const float *alpha, const a_type*a, const dim_t *lda,
            const b_type *x, const dim_t *incy, c_type *y) = {NULL};

    static void (*gemv_s8u8s32_kern)(const dim_t, const dim_t, const float,
            const int8_t *, const dim_t, const uint8_t *, const float,
            int32_t *) = {NULL};

    static void (*gemv_u8s8s32_kern)(const dim_t, const dim_t, const float,
            const uint8_t *, const dim_t, const int8_t *, const float,
            int32_t *) = {NULL};

    switch (data_traits<a_type>::data_type) {
    case data_type::s8:
        if (mayiuse(avx512_core)) {
            this->um = 48;
            this->un = 8;
            this->uk = 1;
            this->bm = 9984;
            this->bn = 384;
            this->bk = mayiuse(avx512_core_vnni) ? 1536 : 768;

            this->bk_traditional = 384;
            this->blocking_small_k =  48;
            this->bn_small_k =  24;
        }
        break;

    case data_type::bf16:
        if (mayiuse(avx512_core)) {
            this->um = 48;
            this->un = 8;
            this->uk = 1;
            this->bm = 9984;
            this->bn = 384;
            this->bk = 768;

            this->bk_traditional = 384;
            this->blocking_small_k = 48;
            this->bn_small_k = 24;
        }
        break;

    case data_type::f32:
        if (mayiuse(avx512_core)) {
            this->um = 48;
            this->un = 8;
            this->uk = 1;
            this->bm = 9984;
            this->bn = 384;
            this->bk = 384;

            this->bk_traditional = 384;
            this->blocking_small_k = 48;
            this->bn_small_k = 24;
        } else if (mayiuse(avx2)) {
            this->um = 24;
            this->un = 4;
            this->uk = 1;
            this->bm = 10000;
            this->bn = 384;
            this->bk = 192;

            this->bk_traditional = 256;
            this->blocking_small_k = 48;
            this->bn_small_k = 24;
        } else if (mayiuse(avx)) {
            this->um = 16;
            this->un = 4;
            this->uk = 1;
            this->bm = 4096;
            this->bn = 96;
            this->bk = 256;

            this->bk_traditional = 256;
            this->blocking_small_k = 48;
            this->bn_small_k = 24;
        } else if (mayiuse(sse41)) {
            this->um = 8;
            this->un = 4;
            this->uk = 1;
            this->bm = 4096;
            this->bn = 96;
            this->bk = 256;

            this->bk_traditional = 256;
            this->blocking_small_k = 48;
            this->bn_small_k = 24;
        }
        break;
    }

    static std::once_flag initialized;
    std::call_once(initialized, []{
        const bool b_is_s8 = data_traits<b_type>::data_type == data_type::s8;

        static jit_generator *copy_a[2][2] = {{NULL}};
        static jit_generator *copy_b[2][2] = {{NULL}};

        switch (data_traits<a_type>::data_type) {
        case data_type::s8:
            if (mayiuse(avx512_core)) {
                copy_a[no_trans][no_sum] =
                    new jit_avx512_core_u8_copy_an_kern();
                copy_a[do_trans][no_sum] =
                    new jit_avx512_core_u8_copy_at_kern();

                copy_b[no_trans][no_sum] =
                    new jit_avx512_core_u8_copy_bn_kern(b_is_s8);
                copy_b[do_trans][no_sum] =
                    new jit_avx512_core_u8_copy_bt_kern(b_is_s8);

                copy_a[no_trans][do_sum] =
                    new jit_avx512_core_u8_copy_sum_an_kern();
                copy_a[do_trans][do_sum] =
                    new jit_avx512_core_u8_copy_sum_at_kern();

                copy_b[no_trans][do_sum] =
                    new jit_avx512_core_u8_copy_sum_bn_kern(b_is_s8);
                copy_b[do_trans][do_sum] =
                    new jit_avx512_core_u8_copy_sum_bt_kern(b_is_s8);
            }
            break;

        case data_type::bf16:
            if (mayiuse(avx512_core)) {
                copy_a[no_trans][no_sum] =
                    new jit_avx512_core_s16_copy_an_kern();
                copy_a[do_trans][no_sum] =
                    new jit_avx512_core_s16_copy_at_kern();

                copy_b[no_trans][no_sum] =
                    new jit_avx512_core_s16_copy_bn_kern();
                copy_b[do_trans][no_sum] =
                    new jit_avx512_core_s16_copy_bt_kern();
            }
            break;

        case data_type::f32:
            if (mayiuse(avx512_core)) {
                copy_a[no_trans][no_sum] =
                    new jit_avx512_core_f32_copy_an_kern();
                copy_a[do_trans][no_sum] =
                    new jit_avx512_core_f32_copy_at_kern();

                copy_b[no_trans][no_sum] =
                    new jit_avx512_core_f32_copy_bn_kern();
                copy_b[do_trans][no_sum] =
                    new jit_avx512_core_f32_copy_bt_kern();
            } else if (mayiuse(avx2)) {
                copy_a[no_trans][no_sum] = new jit_avx2_f32_copy_an_kern();
                copy_a[do_trans][no_sum] = new jit_avx2_f32_copy_at_kern();

                copy_b[no_trans][no_sum] = new jit_avx2_f32_copy_bn_kern();
                copy_b[do_trans][no_sum] = new jit_avx2_f32_copy_bt_kern();
            } else if (mayiuse(avx)) {
                copy_a[no_trans][no_sum] = new jit_avx_f32_copy_an_kern();
                copy_a[do_trans][no_sum] = new jit_avx_f32_copy_at_kern();

                copy_b[no_trans][no_sum] = new jit_avx_f32_copy_bn_kern();
                copy_b[do_trans][no_sum] = new jit_avx_f32_copy_bt_kern();
            } else if (mayiuse(sse41)) {
                copy_a[no_trans][no_sum] = new jit_sse41_f32_copy_an_kern();
                copy_a[do_trans][no_sum] = new jit_sse41_f32_copy_at_kern();

                copy_b[no_trans][no_sum] = new jit_sse41_f32_copy_bn_kern();
                copy_b[do_trans][no_sum] = new jit_sse41_f32_copy_bt_kern();
            }
            break;
        }

        static jit_generator *kernel[2][2][2] = {{{NULL}}};
        switch (data_traits<a_type>::data_type) {
        case data_type::s8:
            if (mayiuse(avx512_core)) {
                for (int isBeta0 : {no_beta0, do_beta0})
                    for (int isColOffset : {no_col_offset, do_col_offset})
                        for (int isRowOffset : {no_row_offset, do_row_offset}) {
                            kernel[isBeta0][isColOffset][isRowOffset] =
                                new jit_avx512_core_gemm_s8u8s32_kern(isBeta0,
                                        isColOffset, isRowOffset);
                        }
            }
            break;

        case data_type::bf16:
            if (mayiuse(avx512_core)) {
                for (int isBeta0 : {no_beta0, do_beta0}) {
                    kernel[isBeta0][no_col_offset][no_row_offset] =
                        new jit_avx512_core_gemm_bf16bf16f32_kern(isBeta0);
                }
            }
            break;

        case data_type::f32:
            if (mayiuse(avx2)) {
                for (int isBeta0 : {no_beta0, do_beta0}) {
                    kernel[isBeta0][no_col_offset][no_row_offset] =
                        new jit_avx2_kernel_sgemm_kern(isBeta0);
                }
            } else if (mayiuse(avx)) {
                kernel[no_beta0][no_col_offset][no_row_offset] =
                    new jit_avx_kernel_sgemm_kern;
                kernel[do_beta0][no_col_offset][no_row_offset] =
                    new jit_avx_kernel_b0_sgemm_kern();
            } else if (mayiuse(sse41)) {
                kernel[no_beta0][no_col_offset][no_row_offset] =
                    new jit_sse41_kernel_sgemm_kern;
                kernel[do_beta0][no_col_offset][no_row_offset] =
                    new jit_sse41_kernel_b0_sgemm_kern();
            }
            break;
        }

        static jit_generator *gemv_kernel[2] = {NULL};
        if (data_traits<a_type>::data_type == data_type::f32) {
            if (mayiuse(avx)) {
                gemv_kernel[do_trans] = new jit_avx_gemv_t_f32_kern();
            } else if (mayiuse(sse41)) {
                gemv_kernel[do_trans] = new jit_sse41_gemv_t_f32_kern();
            }
        }

        static jit_avx512_core_gemv_s8u8s32_kern *gemv_s8u8s32_kernel = NULL;
        static jit_avx512_core_gemv_s8u8s32_kern *gemv_u8s8s32_kernel = NULL;
        if (data_traits<a_type>::data_type == data_type::s8) {
            if (mayiuse(avx512_core)) {
                gemv_s8u8s32_kernel = new jit_avx512_core_gemv_s8u8s32_kern();
                gemv_u8s8s32_kernel = new jit_avx512_core_gemv_s8u8s32_kern();
            }
        }

        // Set copy kernels function pointer table
        for (int isTrans : {no_trans, do_trans})
            for (int isSum : {no_sum, do_sum}) {
                auto *p_copy_a = copy_a[isTrans][isSum];
                if (p_copy_a != NULL)
                    copyA[isTrans][isSum] = p_copy_a->getCode<
                        void (*)(const dim_t *, const dim_t *, const a_type *,
                                const dim_t *, const float *, a_type *,
                                const dim_t *, const dim_t *, c_type *)>();
                auto *p_copy_b = copy_b[isTrans][isSum];
                if (p_copy_b != NULL)
                    copyB[isTrans][isSum] = p_copy_b->getCode<
                        void (*)(const dim_t *, const dim_t *, const b_type *,
                                const dim_t *, const float *, b_type *,
                                const dim_t *, const dim_t *, c_type *)>();
            }

        // Set compute kernel function pointer table
        for (int isBeta0 : {no_beta0, do_beta0})
            for (int isColOffset : {no_col_offset, do_col_offset})
                for (int isRowOffset : {no_row_offset, do_row_offset}) {
                    auto *p_kernel = kernel[isBeta0][isColOffset][isRowOffset];
                    if (p_kernel != NULL)
                        kern[isBeta0][isColOffset][isRowOffset] =
                            p_kernel->getCode<
                            void (*)(const dim_t *, const dim_t *,
                                    const dim_t *, const float *,
                                    const a_type *, const b_type *, c_type *,
                                    const dim_t, const c_type *,
                                    const c_type *)>();
                }

        // Set gemv floating point kernels
        if (data_traits<a_type>::data_type == data_type::f32) {
            for (int isTrans : {no_trans, do_trans}) {
                auto *p_gemv_kernel = gemv_kernel[isTrans];
                if (p_gemv_kernel != NULL)
                    gemv_kern[isTrans] = p_gemv_kernel->getCode<
                void (*)(const dim_t *, const dim_t *, const float *,
                        const a_type *, const dim_t *, const b_type *,
                        const dim_t *, c_type *)>();

            }
        }

        // Set gemv integer gemm kernels
        if (data_traits<a_type>::data_type == data_type::s8) {
            gemv_s8u8s32_kern = gemv_s8u8s32_kernel->generate<
                jit_avx512_core_gemv_s8u8s32_kern::gemv_s8u8s32_kernel_t>(
                        mayiuse(avx512_core_vnni));

            gemv_u8s8s32_kern = gemv_u8s8s32_kernel->generate<
                jit_avx512_core_gemv_s8u8s32_kern::gemv_u8s8s32_kernel_t>(
                        mayiuse(avx512_core_vnni));
        }

    });

    int doSumA = this->bo != 0 ? do_sum : no_sum;
    int doSumB = this->ao != 0 ? do_sum : no_sum;

    int copy_trans_a = (this->transa == do_trans) ? do_trans : no_trans;
    int copy_trans_b = (this->transb == do_trans) ? do_trans : no_trans;

    this->copyA = copyA[copy_trans_a][doSumA];
    this->copyB = copyB[copy_trans_b][doSumB];

    for (int isBeta0 : {no_beta0, do_beta0})
        for (int isColOffset : {no_col_offset, do_col_offset})
            for (int isRowOffset : {no_row_offset, do_row_offset})
                this->kernel[isBeta0][isColOffset][isRowOffset] =
                    kern[isBeta0][isColOffset][isRowOffset];

    for (int isTrans : {no_trans, do_trans})
        this->gemv_kernel[isTrans] = gemv_kern[isTrans];

    this->gemv_s8u8s32_kernel = NULL;
    this->gemv_u8s8s32_kernel = NULL;
    if (data_traits<a_type>::data_type == data_type::s8) {
        this->gemv_s8u8s32_kernel = gemv_s8u8s32_kern;
        this->gemv_u8s8s32_kernel = gemv_u8s8s32_kern;
    }
}

// Check if copy algorithm kernels were generated on supported ISAs.
// Copy algorithm supported for:
//      s8  : Intel AVX512, Intel DL Boost
//      bf16 : Intel AVX512, Intel AVX512 BF16
//      f32 : Intel SSE4.1, Intel AVX, Intel AVX2, Intel AVX512
template <typename a_type, typename b_type, typename c_type>
bool gemm_info_t<a_type, b_type, c_type>::hasKernels(void) {
    switch (data_traits<a_type>::data_type) {
    case data_type::s8:
        if (mayiuse(avx512_core)) {
            for (int isBeta0 : {no_beta0, do_beta0})
                for (int isColOffset : {no_col_offset, do_col_offset})
                    for (int isRowOffset : {no_row_offset, do_row_offset})
                        if (!this->kernel[isBeta0][isColOffset][isRowOffset])
                            return false;

            if (!this->gemv_s8u8s32_kernel || !this->gemv_u8s8s32_kernel)
                return false;

            if (!this->copyA || !this->copyB)
                return false;
        }
        break;

    case data_type::bf16:
        if (mayiuse(avx512_core)) {
            for (int isBeta0 : {no_beta0, do_beta0})
                if (!this->kernel[isBeta0][no_col_offset][no_row_offset])
                    return false;

            if (!this->copyA || !this->copyB)
                return false;

        }
        break;

    case data_type::f32:
        if (mayiuse(sse41) && !this->force_nocopy) {
            for (int isBeta0 : {no_beta0, do_beta0})
                if (!this->kernel[isBeta0][no_col_offset][no_row_offset])
                    return false;

            if (!this->copyA || !this->copyB)
                return false;

            // We only need transpose case for performance.
            if (!this->gemv_kernel[do_trans])
                return false;

        }
        break;
    }

    // All kernels necessary have been found or ISA is not supported.
    return true;
}

// Override default blocking sizes with sizes specified in the gemm_threading_t
//  structure.
template <typename a_type, typename b_type, typename c_type>
void gemm_info_t<a_type, b_type, c_type>::update_blocking(
        const gemm_threading_t &thread_info) {

    if (thread_info.block_m > 0)
        this->bm = thread_info.block_m;
    if (thread_info.block_n > 0)
        this->bn = thread_info.block_n;
    if (thread_info.block_k > 0)
        this->bk = thread_info.block_k;
}

// Instantiate the gemm_info_t templates needed.
template // For gemm_s8u8s32
struct gemm_info_t<int8_t, uint8_t, int32_t>;

template // For gemm_s8s8s32
struct gemm_info_t<int8_t, int8_t, int32_t>;

template // For gemm_bf16bf16f32
struct gemm_info_t<bfloat16_t, bfloat16_t, float>;

template // For sgemm.
struct gemm_info_t<float, float, float>;

}
}
}
