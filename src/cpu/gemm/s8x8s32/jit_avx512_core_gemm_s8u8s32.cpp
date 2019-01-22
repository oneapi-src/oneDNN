/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "common.hpp"
#include "mkldnn_types.h"
#include "nstl.hpp"

#include "jit_avx512_core_gemm_s8u8s32.hpp"
#include "jit_avx512_core_gemm_s8u8s32_kern.hpp"

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

enum {
    PARTITION_1D_ROW,
    PARTITION_1D_COL,
    PARTITION_2D_COL_MAJOR,
    PARTITION_2D = PARTITION_2D_COL_MAJOR,
};

enum {
    COPY_NONE,
    COPY_A,
};

enum {
    NO_OFFSET,
    FIX_OFFSET,
    COL_OFFSET,
    ROW_OFFSET,
};

// Alias for any dimension related variable.
typedef long long int dim_t;

typedef struct {
    int8_t *a_buf;
    int32_t *a_row_sum;
} blas_buffers_t;

typedef struct {
    int nthrs_m, nthrs_n;
    int partition;
    int copy_type;
} blas_thread_t;

typedef struct {
    // Interface arguments.
    int transa, transb, offsetc;
    dim_t m, n, k;
    dim_t lda, ldb, ldc;
    const int8_t *a;
    const uint8_t *b;
    int32_t *c;
    const float *alpha, *beta;

    int8_t ao, bo;
    const int32_t *co;

    // Kernel parameters.
    dim_t um, un, uk, bm, bn, bk;
    dim_t bn_small_k, bk_traditional, blocking_small_k;

    int (*copyA)(const dim_t *m, const dim_t *n, const int8_t *a,
            const dim_t *lda, const int8_t *alpha, int8_t *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    int (*copyB)(const dim_t *m, const dim_t *n, const uint8_t *a,
            const dim_t *lda, const uint8_t *alpha, uint8_t *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    int (*kernel)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_b)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_r)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_c)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_b0)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_b0_b)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_b0_r)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_b0_c)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

} blas_t;


static inline void round_to_nearest(int32_t *rounded_val, double fp_val) {
    if (fp_val >= 0.) {
        fp_val += 0.5;
        if (fp_val > INT32_MAX) {
            fp_val = INT32_MAX;
        }
    } else {
        fp_val -= 0.5;
        if (fp_val < INT32_MIN) {
            fp_val = INT32_MIN;
        }
    }
    *rounded_val = (int32_t) fp_val;
}

static inline void add_results(const dim_t m, const dim_t n, const dim_t k,
        const float alpha, const float beta, const int32_t *c_partial_sum,
        const dim_t ldcp, int32_t *c_data, const dim_t ldc,
        const int32_t *a_row_sum, const int32_t *b_col_sum, const int8_t ao,
        const int8_t bo, const int32_t *co, const int offsetc)
{
    for (dim_t j = 0; j < n; ++j) {
        for (dim_t i = 0; i < m; ++i) {
            int32_t ctemp = c_partial_sum[i + j * ldcp];

            if (alpha == 1.0f) {
                if (beta == 0.0f) {
                    c_data[i + j * ldc] = ctemp;
                } else {
                    double c_float = (double) beta
                        * (double) c_data[i + j * ldc];
                    c_float += (double) ctemp;
                    round_to_nearest(&c_data[i + j * ldc], c_float);
                }
            } else if (alpha == -1.0f) {
                if (beta == 0.0f) {
                    c_data[i + j * ldc] = -ctemp;
                } else {
                    double c_float = (double) beta
                        * (double) c_data[i + j * ldc];
                    c_float -= (double) ctemp;
                    round_to_nearest(&c_data[i + j * ldc], c_float);
                }
            } else {
                if (beta == 0.0f) {
                    double c_float = alpha * (double) ctemp;
                    round_to_nearest(&c_data[i + j * ldc], c_float);
                } else {
                    double c_float = alpha * (double) ctemp +
                        beta * (double) c_data[i + j * ldc];
                    round_to_nearest(&c_data[i + j * ldc], c_float);
                }
            }

            if (offsetc == FIX_OFFSET) {
                c_data[i + j * ldc] += co[0];
            } else if (offsetc == ROW_OFFSET) {
                c_data[i + j * ldc] += co[j];
            } else if (offsetc == COL_OFFSET) {
                c_data[i + j * ldc] += co[i];
            }
        }
    }
}

// TODO Find a better place for those functions.
static inline dim_t val_padd(const dim_t x, const dim_t x1)
{
    return (x % x1) ? ((x / x1) + 1) * x1 : x;
}

static inline dim_t ld_padd(const dim_t x)
{
    return ((x + ((2048 / sizeof(int32_t)) - 1)) / (2048 / sizeof(int32_t)))
        * (2048 / sizeof(int32_t)) +  (64 / sizeof(int32_t));
}

static inline dim_t pad_unroll(const dim_t m, const dim_t m_unroll)
{
    return (m + m_unroll - 1) / m_unroll * m_unroll;
}

void igemm_inner_kernel(const dim_t m, const dim_t n, const dim_t k,
        const int8_t *a, const uint8_t *b, float beta, int32_t *c,
        const dim_t ldc, const int32_t *a_row_sum, const int32_t *b_col_sum,
        const int32_t *co, const int offsetc, const blas_t *arg)
{
    int8_t ao = arg->ao;
    int8_t bo = arg->bo;
    int32_t co_0 = (offsetc == NO_OFFSET)? 0 : co[0];

    // Since m and n are limited by blocking, stack overflow may not happen;
    // it's up to 32kB
#if !defined(_MSC_VER)
    int32_t col_offset[m];
    int32_t row_offset[n];
#else
    int32_t *col_offset = (int32_t *) _alloca(sizeof(*col_offset) * m);
    int32_t *row_offset = (int32_t *) _alloca(sizeof(*row_offset) * n);
#endif

    int col_req = 0;
    int row_req = 0;

    if ((bo != 0) || (offsetc == COL_OFFSET))
        col_req = 1;
    if ((ao != 0) || (offsetc == ROW_OFFSET))
        row_req = 1;

    // It needs one of colum or row offsets, but it doesn't need both
    if (((ao != 0) && (bo != 0)) || ((offsetc == FIX_OFFSET) && (co_0 != 0))) {
        if ((col_req == 0) && (row_req == 0)) {
            if (m <= n) {
                col_req = 1;
            } else {
                row_req = 1;
            }
        }
    }

    if (col_req) {
        for (dim_t i = 0; i < m; i++)
            col_offset[i] = 0;

        if (offsetc == COL_OFFSET) {
            for (dim_t i = 0; i < m; i++)
                col_offset[i] += co[i];
        }

        if (bo != 0) {
            for (dim_t i = 0; i < m; i++)
                col_offset[i] += bo * a_row_sum[i];
        }
    }

    if (row_req) {
        for (dim_t i = 0; i < n; i++)
            row_offset[i] = 0;

        if (offsetc == ROW_OFFSET) {
            for (dim_t i = 0; i < n; i++)
                row_offset[i] += co[i];
        }

        if (ao != 0) {
            for (dim_t i = 0; i < n; i++)
                row_offset[i] += ao * b_col_sum[i];
        }
    }

    if ((offsetc == FIX_OFFSET) && (co_0 != 0)) {
        if (col_req) {
            for (dim_t i = 0; i < m; i++)
                col_offset[i] += co_0;
        } else {
            for (dim_t i = 0; i < n; i++)
                row_offset[i] += co_0;
        }
    }

    if ((ao != 0) && (bo != 0)) {
        if (col_req) {
            for (dim_t i = 0; i < m; i++)
                col_offset[i] += (int32_t) k * ao * bo;
        } else {
            for (dim_t i = 0; i < n; i++)
                row_offset[i] += (int32_t) k * ao * bo;
        }
    }

    if (col_req == 0) {
        if (row_req == 0) {
            if (beta == 0.0) {
                arg->kernel_b0(&m, &n, &k, NULL, a, b, c, ldc, col_offset,
                        row_offset);
            } else {
                arg->kernel(&m, &n, &k, NULL, a, b, c, ldc, col_offset,
                        row_offset);
            }
        } else {
            if (beta == 0.0) {
                arg->kernel_b0_r(&m, &n, &k, NULL, a, b, c, ldc, col_offset,
                        row_offset);
            } else {
                arg->kernel_r(&m, &n, &k, NULL, a, b, c, ldc, col_offset,
                        row_offset);
            }
        }
    } else {
        if (row_req == 0) {
            if (beta == 0.0) {
                arg->kernel_b0_c(&m, &n, &k, NULL, a, b, c, ldc, col_offset,
                        row_offset);
            } else {
                arg->kernel_c(&m, &n, &k, NULL, a, b, c, ldc, col_offset,
                        row_offset);
            }
        } else {
            if (beta == 0.0) {
                arg->kernel_b0_b(&m, &n, &k, NULL, a, b, c, ldc, col_offset,
                        row_offset);
            } else {
                arg->kernel_b(&m, &n, &k, NULL, a, b, c, ldc, col_offset,
                        row_offset);
            }
        }
    }
}

static int gemm_kernel_driver(const dim_t m, const dim_t n, const dim_t k,
        const int8_t *a, const uint8_t *b, int32_t *c, const int32_t *co,
        const blas_t *arg)
{
    dim_t   lda   = arg->lda;
    dim_t   ldb   = arg->ldb;
    dim_t   ldc   = arg->ldc;
    int8_t  ao    = arg->ao;
    int8_t  bo    = arg->bo;
    float   alpha = *arg->alpha;
    float   beta  = *arg->beta;

    if (m <= 0 || n <= 0) {
        return 0;
    }

    // Padding along K dimension.
    dim_t k_padd = 0;
    if (k <= arg->bk_traditional) {
        k_padd = val_padd(k, arg->uk);
        k_padd = nstl::max(128LL, k_padd);
    } else if (k < 2 * arg->bk) {
        k_padd = val_padd(k / 2, arg->uk);
    } else {
        k_padd = arg->bk;
    }

    // Padding along M dimension.
    dim_t m_padd = val_padd(nstl::min(nstl::max(m, arg->um), arg->bm), arg->um);

    // Padding along N dimension.
    dim_t n_padd = 0;
    if (k < arg->blocking_small_k) {
        n_padd = val_padd(nstl::min(nstl::max(n, arg->un), arg->bn_small_k),
                arg->un);
    } else {
        n_padd = val_padd(nstl::min(nstl::max(n, arg->un), arg->bn), arg->un);
    }

    // Padding for temporary buffer for C
    dim_t ldc_buf = ld_padd(m_padd);

    dim_t strideAm = (arg->transa == 0)? 1 : lda;
    dim_t strideAn = (arg->transa != 0)? 1 : lda;
    dim_t strideBm = (arg->transb == 0)? 1 : ldb;
    dim_t strideBn = (arg->transb != 0)? 1 : ldb;

    int8_t *bufferA = (int8_t *) malloc(m_padd * k_padd * sizeof(*bufferA),
            PAGE_2M);
    if (!bufferA) {
        return -1;
    }

    uint8_t *bufferB = (uint8_t *) malloc(k_padd * n_padd * sizeof(*bufferB),
            PAGE_4K);
    if (!bufferB) {
        free(bufferA);
        return -1;
    }

    int32_t *bufferC = NULL;
    if (alpha != 1.0f || (beta != 1 && beta != 0)) {
        bufferC = (int32_t *) malloc(ldc_buf * n_padd * sizeof(*bufferC),
                PAGE_4K);
        if (!bufferC) {
            free(bufferA);
            free(bufferB);
            return -1;
        }
    }

    int32_t *a_row_sum = (int32_t *) malloc(m_padd * sizeof(*a_row_sum),
            PAGE_4K);
    if (!a_row_sum) {
        free(bufferA);
        free(bufferB);
        free(bufferC);
        return -1;
    }

    int32_t *b_col_sum = (int32_t *) malloc(n_padd * sizeof(*b_col_sum),
            PAGE_4K);
    if (!b_col_sum) {
        free(bufferA);
        free(bufferB);
        free(bufferC);
        free(a_row_sum);
        return -1;
    }

    float beta_saved = beta;

    int a_block_copied = 0;
    dim_t sizeM = 0;
    for (dim_t Bm = 0; Bm < m; Bm += sizeM) {
        sizeM = m - Bm;
        if (sizeM > m_padd)
            sizeM = m_padd;

        dim_t sizeK = 0;
        for (dim_t Bk = 0; Bk < k; Bk += sizeK) {
            sizeK = k - Bk;
            if (sizeK > k_padd)
                sizeK = k_padd;

            // Scale C blocks by beta only for the first time
            if (Bk == 0)
                beta = beta_saved;
            else
                beta = 1.0f;

            // Apply C offset when to the last k-block of the partial sum.
            int offsetc = NO_OFFSET;
            if (Bk + sizeK == k)
                offsetc = arg->offsetc;

            dim_t sizeN = 0;
            for (dim_t Bn = 0; Bn < n; Bn += sizeN) {
                sizeN = n - Bn;
                if (sizeN > n_padd)
                    sizeN = n_padd;

                const uint8_t *b_block = b + Bk * strideBm + Bn * strideBn;
                arg->copyB(&sizeK, &sizeN, b_block, &ldb, NULL, bufferB, NULL,
                        NULL, b_col_sum);

                dim_t sizeUM = 0;
                for (dim_t Um = 0; Um < sizeM; Um += sizeUM) {
                    sizeUM = sizeM - Um;
                    if (sizeUM > arg->um)
                        sizeUM = arg->um;

                    /*
                     * Use the whole A buffer only if we have multiple B blocks
                     * for k-dimension, otherwise we are wasting cache to store
                     * B and C blocks.
                     */
                    dim_t Um_forA = 0;
                    if (sizeN < n)
                        Um_forA = Um;

                    const int8_t *a_block = a + (Bm + Um) * strideAm
                        + Bk * strideAn;
                    if (!a_block_copied) {
                        arg->copyA(&sizeK, &sizeUM, a_block, &lda, NULL,
                                bufferA + Um_forA * sizeK, NULL, NULL,
                                a_row_sum + Um_forA);
                    }

                    int32_t *c_block = c + (Bm + Um) + Bn * ldc;
                    dim_t co_stride = 0;
                    if (offsetc == FIX_OFFSET) {
                        co_stride = 0;
                    } else if (offsetc == ROW_OFFSET) {
                        co_stride = Bn;
                    } else if (offsetc == COL_OFFSET) {
                        co_stride = Bm + Um;
                    }
                    if (bufferC) {
                        igemm_inner_kernel(sizeUM, sizeN, sizeK,
                                bufferA + Um_forA * sizeK, bufferB, 0.0f,
                                bufferC + Um, ldc_buf, a_row_sum + Um_forA,
                                b_col_sum, NULL, NO_OFFSET, arg);

                        // Finish the block adding the necessary alpha, beta
                        // and offsets.
                        add_results(sizeUM, sizeN, sizeK, alpha, beta,
                                bufferC + Um, ldc_buf, c_block, ldc,
                                a_row_sum + Um_forA, b_col_sum, ao, bo,
                                co + co_stride, offsetc);
                    } else {
                        igemm_inner_kernel(sizeUM, sizeN, sizeK,
                                bufferA + Um_forA * sizeK, bufferB, beta,
                                c_block, ldc, a_row_sum + Um_forA, b_col_sum,
                                co + co_stride, offsetc, arg);
                    }
                }
                a_block_copied = 1;
            }
            a_block_copied = 0;
        }
    }

    free(bufferA);
    free(bufferB);
    free(bufferC);
    free(a_row_sum);
    free(b_col_sum);

    return 0;
}

static int kernel_driver_parallel_acopiedbcopy(const dim_t m, const dim_t n,
        const dim_t k, const int8_t *bufferA, const uint8_t *b,
        const float beta, int32_t *c, const int offsetc, const int32_t *co,
        const int32_t *a_row_sum, const blas_t *arg)
{
    dim_t   ldb   = arg->ldb;
    dim_t   ldc   = arg->ldc;
    int8_t  ao    = arg->ao;
    int8_t  bo    = arg->bo;
    float   alpha = *arg->alpha;

    if (m <= 0 || n <= 0) {
        return 0;
    }

    // Padding along N dimension.
    dim_t n_padd = 0;
    if (k < arg->blocking_small_k) {
        n_padd = val_padd(nstl::min(nstl::max(n, arg->un), arg->bn_small_k),
                arg->un);
    } else {
        n_padd = val_padd(nstl::min(nstl::max(n, arg->un), arg->bn), arg->un);
    }

    // Padding for temporary buffer for C
    dim_t ldc_buf = ld_padd(m);

    dim_t strideBn = (arg->transb != 0)? 1 : ldb;

    uint8_t *bufferB = (uint8_t *) malloc(k * n_padd * sizeof(*bufferB),
            PAGE_4K);
    if (!bufferB) {
        return -1;
    }

    int32_t *bufferC = NULL;
    if (alpha != 1.0f || (beta != 1 && beta != 0)) {
        bufferC = (int32_t *) malloc(ldc_buf * n_padd * sizeof(*bufferC),
                PAGE_4K);
        if (!bufferC) {
            free(bufferB);
            return -1;
        }
    }

    int32_t *b_col_sum = (int32_t *) malloc(n_padd * sizeof(*b_col_sum),
            PAGE_4K);
    if (!b_col_sum) {
        free(bufferB);
        free(bufferC);
        return -1;
    }

    dim_t sizeN = 0;
    for (dim_t Bn = 0; Bn < n; Bn += sizeN) {
        sizeN = n - Bn;
        if (sizeN > n_padd)
            sizeN = n_padd;

        // Implement the kernel here.
        const uint8_t *b_block = b + Bn * strideBn;
        arg->copyB(&k, &sizeN, b_block, &ldb, NULL, bufferB, NULL, NULL,
                b_col_sum);

            dim_t co_stride = 0;
            if (offsetc == FIX_OFFSET) {
                co_stride = 0;
            } else if (offsetc == ROW_OFFSET) {
                co_stride = Bn;
            } else if (offsetc == COL_OFFSET) {
                co_stride = 0;
            }
        int32_t *c_block = c + Bn * ldc;
        if (bufferC) {
            igemm_inner_kernel(m, sizeN, k, bufferA, bufferB, 0.0f, bufferC,
                    ldc_buf, a_row_sum, b_col_sum, NULL, NO_OFFSET, arg);

            // Finish the block adding the necessary alpha, beta and offsets.
            add_results(m, sizeN, k, alpha, beta, bufferC, ldc_buf, c_block,
                    ldc, a_row_sum, b_col_sum, ao, bo, co + co_stride,
                    offsetc);
        } else {
            igemm_inner_kernel(m, sizeN, k, bufferA, bufferB, beta, c_block,
                    ldc, a_row_sum, b_col_sum, co + co_stride, offsetc, arg);
        }
    }

    free(bufferB);
    free(bufferC);
    free(b_col_sum);

    return 0;

}

#define N2D_MAX_AVX512 384
#define M2D_MIN_AVX512 384
#define VECLEN         16
#define NCONS          1
static inline void set_thread_opts_avx512(int *p_nthrs,
        blas_thread_t *thread_info, const blas_t *arg)
{
    int nthrs = *p_nthrs;
    dim_t m = arg->m;
    dim_t n = arg->n;

    thread_info->nthrs_m = 0;
    thread_info->nthrs_n = 0;
    thread_info->copy_type = COPY_NONE; // By default don't do parallel copy.

    int condition_2D_bsrc = -1;
    if ((256 * m > nthrs * n) && (nthrs * m < 256 * n)) {
        condition_2D_bsrc = 1;
    } else {
        condition_2D_bsrc = 0;
    }

    int condition_1D_copya = 0;
    if ((m >= 1000) && (n >= nthrs * N2D_MAX_AVX512 / 4)) {
        condition_2D_bsrc  = 0;
        condition_1D_copya = 1;
    }

    // If offset is non-zero, we need to keep 1D_copya to reduce update overhead
    if (arg->ao != 0 || arg->bo != 0 || arg->co[0] != 0
            || arg->offsetc != FIX_OFFSET) {
        condition_2D_bsrc  = 0;
        condition_1D_copya = 1;
    }

    if (condition_2D_bsrc == 1) {
        int nthrs_m = 1;
        int nthrs_n = nthrs;

        while ((nthrs_n % 2 == 0) &&
                (n / nthrs > N2D_MAX_AVX512 ||
                 n / nthrs_n <= N2D_MAX_AVX512 / 2) &&
                (m / nthrs_m >= 2 * M2D_MIN_AVX512) &&
                (nthrs_m < 4)) {
            nthrs_m *= 2;
            nthrs_n /= 2;
        }

        thread_info->nthrs_m = nthrs_m;
        thread_info->nthrs_n = nthrs_n;
        thread_info->partition = PARTITION_2D;

        // Reset the total number of threads that will be used.
        *p_nthrs = nthrs_m * nthrs_n;

    } else if (condition_1D_copya && mkldnn_thr_syncable()) {
        // Use parallel copy A algorithm
        thread_info->copy_type = COPY_A;
        thread_info->partition = PARTITION_1D_COL;
    } else {
        if ((m > n) && (m / nthrs >= VECLEN || n < NCONS * nthrs)) {
            thread_info->partition = PARTITION_1D_ROW;
        } else {
            thread_info->partition = PARTITION_1D_COL;
        }
    }
}
#undef N2D_MAX_AVX512
#undef M2D_MIN_AVX512
#undef VECLEN
#undef NCONS

static inline void partition_1d(const int ithr, const int nthrs, const dim_t n,
        dim_t *t_offset, dim_t *t_block)
{
    dim_t band = n / nthrs;

    dim_t tail = n - (nthrs - 1) * band;
    if (tail > (band + 1))
        band++;
    tail = n - (nthrs - 1) * band;

    if (ithr < (nthrs - 1))
        *t_block = band;
    else
        *t_block = tail;

    *t_offset = ithr * band;

    if (*t_offset >= n) {
        *t_block = 0;
        *t_offset = 0;
    } else if ((*t_offset + *t_block) > n) {
        *t_block = n - *t_offset;
    }
}

static inline void partition_2d(const int ithr, int *nthrs, const int ithr_i,
        const int ithr_j, const int nthrs_m, const int nthrs_n, const dim_t m,
        const dim_t n, dim_t *p_m_disp, dim_t *p_m_band, dim_t *p_n_disp,
        dim_t *p_n_band)
{
    dim_t m_disp = 0, n_disp = 0;
    dim_t m_band = 0, n_band = 0;

    int mdiv = nthrs_m;
    int ndiv = nthrs_n;

    dim_t m_bandt = m / mdiv; /* size per thread */
    dim_t n_bandt = n / ndiv; /* size per thread */
    int firstmgroup = mdiv - 1;
    int firstngroup = ndiv - 1;
    dim_t firstmval = m_bandt;
    dim_t firstnval = n_bandt;

    int mthr_used = mdiv;
    if (m - (mdiv - 1) * m_bandt > m_bandt + 1) {
        if (m - (mdiv - 1) * m_bandt > mdiv)
            ++m_bandt;

        firstmval = m_bandt + 1;
        mthr_used = (int) (m / firstmval);

        if (mthr_used * firstmval < m)
            ++mthr_used;

        firstmgroup = mthr_used - 1;
    }

    int nthr_used = ndiv;
    if (n - (ndiv - 1) * n_bandt > n_bandt + 1) {
        firstnval = n_bandt + 1;
        nthr_used = (int) (n / firstnval);

        if (nthr_used * firstnval < n)
            ++nthr_used;

        firstngroup = nthr_used - 1;
    }

    *nthrs = mthr_used * nthr_used;

    if (ithr < *nthrs) {
        if (ithr_i < firstmgroup) {
            m_band = firstmval;
            m_disp = ithr_i * firstmval;
        } else if (ithr_i <= mthr_used - 2) {
            m_band = m_bandt;
            m_disp = firstmgroup * firstmval + (ithr_i - firstmgroup) * m_bandt;
        } else {
            m_disp = firstmgroup * firstmval
                + (mthr_used - 1 - firstmgroup) * m_bandt;
            m_band = nstl::max(0LL, m - m_disp);
        }

        if (ithr_j < firstngroup) {
            n_band = firstnval;
            n_disp = ithr_j * firstnval;
        } else if (ithr_j <= nthr_used - 2) {
            n_band = n_bandt;
            n_disp = firstngroup * firstnval + (ithr_j - firstngroup) * n_bandt;
        } else {
            n_disp = firstngroup * firstnval
                + (nthr_used - 1 - firstngroup) * n_bandt;
            n_band = nstl::max(0LL, n - n_disp);
        }
        m_disp = nstl::max(nstl::min(m_disp, m - 1), 0LL);
        n_disp = nstl::max(nstl::min(n_disp, n - 1), 0LL);
    }

    if (ithr < *nthrs) {
        *p_m_disp = m_disp;
        *p_n_disp = n_disp;
        *p_m_band = m_band;
        *p_n_band = n_band;
    } else {
        *p_m_disp = 0;
        *p_n_disp = 0;
        *p_m_band = 0;
        *p_n_band = 0;
    }

    return;
}

static inline void decompose_matrices(const int ithr, int *nthrs, dim_t *m,
        dim_t *n, dim_t *k, const int8_t **a, const uint8_t **b, int32_t **c,
        const int32_t **co, const blas_thread_t *thread_info, const blas_t *arg)
{
    dim_t strideAm = (arg->transa == 0)? 1 : arg->lda;
    dim_t strideBn = (arg->transb != 0)? 1 : arg->ldb;
    int offsetc = arg->offsetc;

    switch (thread_info->partition) {
    case PARTITION_1D_ROW:
        {
            dim_t offset = 0;
            dim_t block = 0;
            partition_1d(ithr, *nthrs, arg->m, &offset, &block);

            *m = block;
            *n = arg->n;
            *k = arg->k;

            // Set matrix A.
            *a = arg->a + offset * strideAm;

            // Set matrix B.
            *b = arg->b;

            // Set matrix C.
            *c = arg->c + offset;

            // Set offset vector for C matrix
            dim_t co_stride = 0;
            if (offsetc == FIX_OFFSET) {
                co_stride = 0;
            } else if (offsetc == ROW_OFFSET) {
                co_stride = 0;
            } else if (offsetc == COL_OFFSET) {
                co_stride = offset;
            }
            *co = arg->co + co_stride;
            break;
        }

    case PARTITION_1D_COL:
        {
            dim_t offset = 0;
            dim_t block = 0;
            partition_1d(ithr, *nthrs, arg->n, &offset, &block);

            *m = arg->m;
            *n = block;
            *k = arg->k;

            // Set matrix A.
            *a = arg->a;

            // Set matrix B.
            *b = arg->b + offset * strideBn;

            // Set matrix C.
            *c = arg->c + offset * arg->ldc;

            // Set offset vector for C matrix
            dim_t co_stride = 0;
            if (offsetc == FIX_OFFSET) {
                co_stride = 0;
            } else if (offsetc == ROW_OFFSET) {
                co_stride = offset;
            } else if (offsetc == COL_OFFSET) {
                co_stride = 0;
            }
            *co = arg->co + co_stride;
            break;
        }

    case PARTITION_2D_COL_MAJOR:
        {
            int nthrs_m = thread_info->nthrs_m;
            int nthrs_n = thread_info->nthrs_n;
            int ithr_i = ithr % nthrs_m;
            int ithr_j = ithr / nthrs_m;

            dim_t m_disp = 0;
            dim_t m_band = 0;
            dim_t n_disp = 0;
            dim_t n_band = 0;

            partition_2d(ithr, nthrs, ithr_i, ithr_j, nthrs_m, nthrs_n,
                    arg->m, arg->n, &m_disp, &m_band, &n_disp, &n_band);

            *m = m_band;
            *n = n_band;
            *k = arg->k;

            // Set matrix A.
            *a = arg->a + m_disp * strideAm;

            // Set matrix B.
            *b = arg->b + n_disp * strideBn;

            // Set matrix C.
            *c = arg->c + m_disp + n_disp * arg->ldc;

            // Set offset vector for C matrix
            dim_t co_stride = 0;
            if (offsetc == FIX_OFFSET) {
                co_stride = 0;
            } else if (offsetc == ROW_OFFSET) {
                co_stride = n_disp;
            } else if (offsetc == COL_OFFSET) {
                co_stride = m_disp;
            }
            *co = arg->co + co_stride;
            break;
        }
    }
}

#define MULTIPLIER 10
static int parallel_a_copy(const int ithr, const int nthrs, const dim_t m,
        const dim_t n, const dim_t k, const int8_t *a, const uint8_t *b,
        int32_t *c, const int32_t *co, const blas_t *arg,
        blas_buffers_t *shared_buffers)
{
    const dim_t lda = arg->lda;
    const dim_t ldb = arg->ldb;
    const dim_t strideAm = (arg->transa == 0)? 1 : lda;
    const dim_t strideAn = (arg->transa != 0)? 1 : lda;
    const dim_t strideBm = (arg->transb == 0)? 1 : ldb;

    // Padding along M dimension.
    dim_t m_padd = val_padd(nstl::min(nstl::max(m, arg->um), arg->bm), arg->um);

    // Padding along K dimension.
    dim_t k_padd = 0;
    if (k <= arg->bk_traditional) {
        k_padd = val_padd(k, arg->uk);
        k_padd = nstl::max(128LL, k_padd);
    } else if (k < 2 * arg->bk) {
        k_padd = val_padd(k / 2, arg->uk);
    } else {
        k_padd = arg->bk;
    }

    m_padd *= nthrs > MULTIPLIER ? MULTIPLIER : nthrs;
    if (m_padd > m) {
        m_padd = pad_unroll(m, arg->um);
    }

    // Allocate buffers for A and it's row sum in master thread.
    if (ithr == 0) { // If thread master
        shared_buffers->a_buf = (int8_t *)
            malloc(m_padd * k_padd * sizeof(*a), PAGE_2M);

        shared_buffers->a_row_sum = (int32_t *)
            malloc(m_padd * sizeof(*c), PAGE_4K);
    }
    mkldnn_thr_barrier();

    int8_t *bufferA = shared_buffers->a_buf;
    int32_t *a_row_sum = shared_buffers->a_row_sum;

    if (!bufferA || !a_row_sum) {
        free(bufferA);
        free(a_row_sum);
        return -1;
    }

    int result = 0; // Return status

    dim_t sizeK = 0;
    for (dim_t Bk = 0; Bk < k; Bk += sizeK) {
        sizeK = k - Bk;
        if (sizeK > k_padd)
            sizeK = k_padd;

        // Scale C blocks by beta only for the first term of partial sum.
        float beta = 1.0f;
        if (Bk == 0)
            beta = *(arg->beta);

        // Apply C offset for the last k-block of the partial sum.
        int offsetc = NO_OFFSET;
        if (Bk + sizeK == k)
            offsetc = arg->offsetc;

        dim_t sizeM = 0;
        for (dim_t Bm = 0; Bm < m; Bm += sizeM) {
            sizeM = m - Bm;
            if (sizeM > m_padd)
                sizeM = m_padd;

            if (ithr < nthrs) {
                dim_t band = (sizeM + nthrs - 1) / nthrs;
                band = pad_unroll(band, arg->um);

                dim_t offset = band * ithr;

                // If offset is too large don't use that thread for copying.
                if (offset >= sizeM) {
                    offset = 0;
                    band = 0;
                }

                // Handle the tail of the copy.
                if (offset + band > sizeM) {
                    band = sizeM - offset;
                }

                if (band > 0) {
                    const int8_t *a_block = a + (Bm + offset) * strideAm
                        + Bk * strideAn;
                    arg->copyA(&sizeK, &band, a_block, &lda, NULL,
                            bufferA + offset * sizeK, NULL, NULL,
                            a_row_sum + offset);
                }
            }
            mkldnn_thr_barrier(); // Wait for finishing parallel copy.

            const uint8_t *b_block = b + Bk * strideBm;
            int32_t *c_block = c + Bm;
            dim_t co_stride = 0;
            if (offsetc == FIX_OFFSET) {
                co_stride = 0;
            } else if (offsetc == ROW_OFFSET) {
                co_stride = 0;
            } else if (offsetc == COL_OFFSET) {
                co_stride = Bm;
            }

            result = kernel_driver_parallel_acopiedbcopy(sizeM, n, sizeK,
                    bufferA, b_block, beta, c_block, offsetc, co + co_stride,
                    a_row_sum, arg);

            mkldnn_thr_barrier(); // Wait for kernel computations to finish.
        }
    }

    // Free memory allocated in master thread
    if (ithr == 0) {
        free(bufferA);
        free(a_row_sum);
    }

    return result;
}
#undef MULTIPLIER

static inline void get_omp_thread_count(dim_t m, dim_t n, dim_t k,
        double fp_per_cycle, int *nthrs)
{
    double omp_overhead_small_core = 3.0e+3;
    double omp_intercept_big_core = 4.0e+3;
    double omp_slope_big_core = 5.0e+2;

    double gemm_cycles = 8.0 * m * n * k / fp_per_cycle;

    int i = *nthrs;

    // Use a different model for omp overheads if nthrs is <= 4
    if (*nthrs <= 4 && omp_overhead_small_core > 0) {
        double omp_cycles = omp_overhead_small_core;
        if (gemm_cycles < omp_cycles) {
            *nthrs = 1;
            return;
        } else {
            while (i > 1) {
                if (omp_cycles * i < gemm_cycles * (i - 1)) break;
                --i;
            }
        }
    } else {
        if (gemm_cycles < (omp_intercept_big_core + 2 * omp_slope_big_core)) {
            *nthrs = 1;
            return;
        }

        // adaptive decrement to march faster·
        while (i > 1) {
            double omp_cycles = omp_intercept_big_core + i * omp_slope_big_core;
            if (omp_cycles * i < gemm_cycles * (i - 1))
                break;

            if (i < 10)
                i -= 2;
            else if (i < 30)
                i -= 4;
            else
                i -= 8;
        }
    }

    if (i < 1)
        i = 1;

    *nthrs = i;
}

#define CACHE_LINE_SIZE 64
static int gemm_threading_driver(blas_t *arg)
{
    if ((arg->m <= 0) || (arg->n <= 0))
        return mkldnn_success;

    int nthr = (mkldnn_in_parallel()) ? 1 : mkldnn_get_max_threads();
    get_omp_thread_count(arg->m, arg->n, arg->k, 64.0, &nthr);

    if (nthr == 1) {
        return gemm_kernel_driver(arg->m, arg->n, arg->k, arg->a, arg->b,
                arg->c, arg->co, arg);
    }

    int *results = (int *) malloc(sizeof(*results) * nthr * CACHE_LINE_SIZE,
            PAGE_4K);

    if (!results) {
        return -1;
    }

    for (int i = 0; i < nthr; i++) {
        results[i * CACHE_LINE_SIZE] = 0; // Initialize to success
    }

    blas_buffers_t shared_buffers;
    shared_buffers.a_buf = NULL;
    shared_buffers.a_row_sum = NULL;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int nthrs = nthr;
        if (nthrs == 1) {
            results[0] = gemm_kernel_driver(arg->m, arg->n, arg->k, arg->a,
                arg->b, arg->c, arg->co, arg);
        } else {
            blas_thread_t thread_info;
            set_thread_opts_avx512(&nthrs, &thread_info, arg);

            const int8_t *a = NULL;
            const uint8_t *b = NULL;
            int32_t *c = NULL;
            const int32_t *co = NULL;
            dim_t m = -1;
            dim_t n = -1;
            dim_t k = -1;
            decompose_matrices(ithr, &nthrs, &m, &n, &k, &a, &b, &c, &co,
                &thread_info, arg);

            if (ithr < nthrs) {
                switch (thread_info.copy_type) {
                case COPY_A:
                    results[ithr * CACHE_LINE_SIZE] =
                        parallel_a_copy(ithr, nthrs, m, n, k, a, b, c, co, arg,
                                &shared_buffers);
                    break;

                default:
                case COPY_NONE:
                    results[ithr * CACHE_LINE_SIZE] =
                        gemm_kernel_driver(m, n, k, a, b, c, co, arg);
                    break;
                }
            }
        }
    });

    int result = 0;  // Initialize to success
    for (int i = 0; i < nthr; i++) {
        if (results[i] != 0) {
            result = results[i * CACHE_LINE_SIZE];
            break;
        }
    }

    free(results);

    return result;
}
#undef CACHE_LINE_SIZE

static jit_avx512_core_u8_copy_an_kern *copy_an;
static jit_avx512_core_u8_copy_at_kern *copy_at;
static jit_avx512_core_u8_copy_bn_kern *copy_bn;
static jit_avx512_core_u8_copy_bt_kern *copy_bt;
static jit_avx512_core_u8_copy_sum_an_kern *copy_sum_an;
static jit_avx512_core_u8_copy_sum_at_kern *copy_sum_at;
static jit_avx512_core_u8_copy_sum_bn_kern *copy_sum_bn;
static jit_avx512_core_u8_copy_sum_bt_kern *copy_sum_bt;
static jit_avx512_core_gemm_s8u8s32_kern *kernel;
static jit_avx512_core_gemm_s8u8s32_kern *kernel_b;
static jit_avx512_core_gemm_s8u8s32_kern *kernel_r;
static jit_avx512_core_gemm_s8u8s32_kern *kernel_c;
static jit_avx512_core_gemm_s8u8s32_kern *kernel_b0;
static jit_avx512_core_gemm_s8u8s32_kern *kernel_b0_b;
static jit_avx512_core_gemm_s8u8s32_kern *kernel_b0_r;
static jit_avx512_core_gemm_s8u8s32_kern *kernel_b0_c;

static void jit_init(blas_t *arg)
{
    static int (*copyAn)(const dim_t *m, const dim_t *n, const int8_t *a,
            const dim_t *lda, const int8_t *alpha, int8_t *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    static int (*copyAt)(const dim_t *m, const dim_t *n, const int8_t  *a,
            const dim_t *lda, const int8_t  *alpha, int8_t  *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    static int (*copyBn)(const dim_t *m, const dim_t *n, const uint8_t *a,
            const dim_t *lda, const uint8_t *alpha, uint8_t *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    static int (*copyBt)(const dim_t *m, const dim_t *n, const uint8_t *a,
            const dim_t *lda, const uint8_t *alpha, uint8_t *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    static int (*copySumAn)(const dim_t *m, const dim_t *n, const int8_t  *a,
            const dim_t *lda, const int8_t  *alpha, int8_t  *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    static int (*copySumAt)(const dim_t *m, const dim_t *n, const int8_t  *a,
            const dim_t *lda, const int8_t  *alpha, int8_t  *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    static int (*copySumBn)(const dim_t *m, const dim_t *n, const uint8_t *a,
            const dim_t *lda, const uint8_t *alpha, uint8_t *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    static int (*copySumBt)(const dim_t *m, const dim_t *n, const uint8_t *a,
            const dim_t *lda, const uint8_t *alpha, uint8_t *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    static int (*kern)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    static int (*kern_b)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    static int (*kern_r)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    static int (*kern_c)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    static int (*kern_b0)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    static int (*kern_b0_b)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    static int (*kern_b0_r)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    static int (*kern_b0_c)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    if (mayiuse(avx512_core_vnni)) {
            arg->um = AVX512_UNROLL_M;
            arg->un = AVX512_UNROLL_N;
            arg->uk = AVX512_UNROLL_K;
            arg->bm = AVX512_BM;
            arg->bn = AVX512_BN;
            arg->bk = AVX512_BK_VNNI;

            arg->bk_traditional   = AVX512_BK_TRADITIONAL;
            arg->bn_small_k       = AVX512_BN_SMALL_K;
            arg->blocking_small_k = AVX512_BLOCKING_SMALL_K;
    } else {
            arg->um = AVX512_UNROLL_M;
            arg->un = AVX512_UNROLL_N;
            arg->uk = AVX512_UNROLL_K;
            arg->bm = AVX512_BM;
            arg->bn = AVX512_BN;
            arg->bk = AVX512_BK;

            arg->bk_traditional   = AVX512_BK_TRADITIONAL;
            arg->bn_small_k       = AVX512_BN_SMALL_K;
            arg->blocking_small_k = AVX512_BLOCKING_SMALL_K;
    }

    static std::once_flag initialized;
    std::call_once(initialized, []{
        copy_an = new jit_avx512_core_u8_copy_an_kern();
        copy_at = new jit_avx512_core_u8_copy_at_kern();
        copy_bn = new jit_avx512_core_u8_copy_bn_kern();
        copy_bt = new jit_avx512_core_u8_copy_bt_kern();

        copy_sum_an = new jit_avx512_core_u8_copy_sum_an_kern();
        copy_sum_at = new jit_avx512_core_u8_copy_sum_at_kern();
        copy_sum_bn = new jit_avx512_core_u8_copy_sum_bn_kern();
        copy_sum_bt = new jit_avx512_core_u8_copy_sum_bt_kern();

        kernel      = new jit_avx512_core_gemm_s8u8s32_kern(false, false, false);
        kernel_b    = new jit_avx512_core_gemm_s8u8s32_kern(false, true,  true);
        kernel_r    = new jit_avx512_core_gemm_s8u8s32_kern(false, false, true);
        kernel_c    = new jit_avx512_core_gemm_s8u8s32_kern(false, true,  false);
        kernel_b0   = new jit_avx512_core_gemm_s8u8s32_kern(true,  false, false);
        kernel_b0_b = new jit_avx512_core_gemm_s8u8s32_kern(true,  true,  true);
        kernel_b0_r = new jit_avx512_core_gemm_s8u8s32_kern(true,  false, true);
        kernel_b0_c = new jit_avx512_core_gemm_s8u8s32_kern(true,  true,  false);

        copyAn = copy_an->getCode<int (*)(const dim_t *, const dim_t *,
                const int8_t *, const dim_t *, const int8_t *, int8_t *,
                const dim_t *, const dim_t *, int32_t *)>();

        copyAt = copy_at->getCode<int (*)(const dim_t *, const dim_t *,
                const int8_t *, const dim_t *, const int8_t *, int8_t *,
                const dim_t *, const dim_t *, int32_t *)>();

        copyBn = copy_bn->getCode<int (*)(const dim_t *, const dim_t *,
                const uint8_t *, const dim_t *, const uint8_t *, uint8_t *,
                const dim_t *, const dim_t *, int32_t *)>();

        copyBt = copy_bt->getCode<int (*)(const dim_t *, const dim_t *,
                const uint8_t *, const dim_t *, const uint8_t *, uint8_t *,
                const dim_t *, const dim_t *, int32_t *)>();

        copySumAn = copy_sum_an->getCode<int (*)(const dim_t *, const dim_t *,
                const int8_t *, const dim_t *, const int8_t *, int8_t *,
                const dim_t *, const dim_t *, int32_t *)>();

        copySumAt = copy_sum_at->getCode<int (*)(const dim_t *, const dim_t *,
                const int8_t *, const dim_t *, const int8_t *, int8_t *,
                const dim_t *, const dim_t *, int32_t *)>();

        copySumBn = copy_sum_bn->getCode<int (*)(const dim_t *, const dim_t *,
                const uint8_t *, const dim_t *, const uint8_t *, uint8_t *,
                const dim_t *, const dim_t *, int32_t *)>();

        copySumBt = copy_sum_bt->getCode<int (*)(const dim_t *, const dim_t *,
                const uint8_t *, const dim_t *, const uint8_t *, uint8_t *,
                const dim_t *, const dim_t *, int32_t *)>();

        kern = kernel->getCode<int (*)(const dim_t *, const dim_t *,
                const dim_t *, const float *, const int8_t *, const uint8_t *,
                int32_t *, const dim_t, const int32_t *, const int32_t *)>();

        kern_b = kernel_b->getCode<int (*)(const dim_t *, const dim_t *,
                const dim_t *, const float *, const int8_t *, const uint8_t *,
                int32_t *, const dim_t, const int32_t *, const int32_t *)>();

        kern_r = kernel_r->getCode<int (*)(const dim_t *, const dim_t *,
                const dim_t *, const float *, const int8_t *, const uint8_t *,
                int32_t *, const dim_t, const int32_t *, const int32_t *)>();

        kern_c = kernel_c->getCode<int (*)(const dim_t *, const dim_t *,
                const dim_t *, const float *, const int8_t *, const uint8_t *,
                int32_t *, const dim_t, const int32_t *, const int32_t *)>();

        kern_b0 = kernel_b0->getCode<int (*)(const dim_t *, const dim_t *,
                const dim_t *, const float *, const int8_t *, const uint8_t *,
                int32_t *, const dim_t, const int32_t *, const int32_t *)>();

        kern_b0_b = kernel_b0_b->getCode<int (*)(const dim_t *, const dim_t *,
                const dim_t *, const float *, const int8_t *, const uint8_t *,
                int32_t *, const dim_t, const int32_t *, const int32_t *)>();

        kern_b0_r = kernel_b0_r->getCode<int (*)(const dim_t *, const dim_t *,
                const dim_t *, const float *, const int8_t *, const uint8_t *,
                int32_t *, const dim_t, const int32_t *, const int32_t *)>();

        kern_b0_c = kernel_b0_c->getCode<int (*)(const dim_t *, const dim_t *,
                const dim_t *, const float *, const int8_t *, const uint8_t *,
                int32_t *, const dim_t, const int32_t *, const int32_t *)>();
    });

    if (arg->bo == 0) { // No need to compute A row sum if bo is zero
        if (arg->transa == 0) {
            arg->copyA = copyAn;
        } else {
            arg->copyA = copyAt;
        }
    } else {
        if (arg->transa == 0) {
            arg->copyA = copySumAn;
        } else {
            arg->copyA = copySumAt;
        }
    }

    if (arg->ao == 0) { // No need to compute B column sum if ao is zero
        if (arg->transb == 0) {
            arg->copyB = copyBn;
        } else {
            arg->copyB = copyBt;
        }
    } else {
        if (arg->transb == 0) {
            arg->copyB = copySumBn;
        } else {
            arg->copyB = copySumBt;
        }
    }

    arg->kernel      = kern;
    arg->kernel_b    = kern_b;
    arg->kernel_r    = kern_r;
    arg->kernel_c    = kern_c;
    arg->kernel_b0   = kern_b0;
    arg->kernel_b0_b = kern_b0_b;
    arg->kernel_b0_r = kern_b0_r;
    arg->kernel_b0_c = kern_b0_c;
}

mkldnn_status_t jit_avx512_core_gemm_s8u8s32(
        const char *transA, const char *transB, const char *offsetC,
        const int *m, const int *n, const int *k,
        const float *alpha, const int8_t *a, const int *lda, const int8_t *oa,
        const uint8_t *b, const int *ldb, const int8_t *ob,
        const float *beta, int32_t *c, const int *ldc, const int32_t *oc)
{
    char transa  = *transA;
    char transb  = *transB;
    char offsetc = *offsetC;

    blas_t args;

    // Initialize blas structure
    args.m         = *m;
    args.n         = *n;
    args.k         = *k;
    args.alpha     = alpha;
    args.a         = a;
    args.lda       = *lda;
    args.b         = b;
    args.ldb       = *ldb;
    args.beta      = beta;
    args.c         = c;
    args.ldc       = *ldc;
    args.transa    = (transa == 'N' || transa == 'n') ? 0 : 1;
    args.transb    = (transb == 'N' || transb == 'n') ? 0 : 1;
    args.um        = 0;
    args.un        = 0;
    args.bm        = 0;
    args.bn        = 0;
    args.bk        = 0;
    args.copyA     = NULL;
    args.copyB     = NULL;
    args.kernel    = NULL;
    args.kernel_b0 = NULL;
    args.ao        = *oa;
    args.bo        = *ob;
    args.co        = oc;

    if (offsetc == 'F' || offsetc == 'f') {
        args.offsetc = FIX_OFFSET;
    } else if (offsetc == 'R' || offsetc == 'r') {
        args.offsetc = ROW_OFFSET;
    } else { // offsetc == 'C' || offsetc == 'c'
        args.offsetc = COL_OFFSET;
    }

    jit_init(&args);
    int result = gemm_threading_driver(&args);

    return (result < 0 ) ? mkldnn_out_of_memory : mkldnn_success;
}

}
}
}
