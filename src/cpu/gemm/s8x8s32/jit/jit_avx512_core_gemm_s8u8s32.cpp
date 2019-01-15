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

// Alias for any dimension related variable.
typedef long long int dim_t;

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

    int (*copyA)(const dim_t *m, const dim_t *n, const int8_t  *a, const dim_t *lda, const int8_t  *alpha, int8_t  *b);
    int (*copyB)(const dim_t *m, const dim_t *n, const uint8_t *a, const dim_t *lda, const uint8_t *alpha, uint8_t *b);

    int (*kernel)   (const dim_t *m, const dim_t *n, const dim_t *k, const float *alpha, const int8_t *a, const uint8_t *b, int *c, const dim_t ldc);
    int (*kernel_b0)(const dim_t *m, const dim_t *n, const dim_t *k, const float *alpha, const int8_t *a, const uint8_t *b, int *c, const dim_t ldc);

    // Threading parameters.
    int nthrs;
    int nthrs_m, nthrs_n;
    int thread_partition;
    int thread_copy;

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
        const int8_t bo, const int32_t *co, int offsetc)
{
    for (dim_t j = 0; j < n; ++j) {
        for (dim_t i = 0; i < m; ++i) {
            int32_t ctemp = c_partial_sum[i + j * ldcp];
            if (ao != 0 || bo != 0)
                ctemp += a_row_sum[i] * bo + b_col_sum[j] * ao + ao * bo * (int32_t) k;

            if (alpha == 1.0f) {
                if (beta == 0.0f) {
                    c_data[i + j * ldc] = ctemp;
                } else {
                    double c_float = (double) beta * (double) c_data[i + j * ldc];
                    c_float += (double) ctemp;
                    round_to_nearest(&c_data[i + j * ldc], c_float);
                }
            } else if (alpha == -1.0f) {
                if (beta == 0.0f) {
                    c_data[i + j * ldc] = -ctemp;
                } else {
                    double c_float = (double) beta * (double) c_data[i + j * ldc];
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

            if (offsetc == 0) { // Fix offset.
                c_data[i + j * ldc] += co[0];
            } else if (offsetc == 1) { // Row offset.
                c_data[i + j * ldc] += co[j];
            } else if (offsetc == 2) { // Col offset.
                c_data[i + j * ldc] += co[i];
            }
        }
    }
}

static inline void get_a_row_sum(const int transa, const dim_t nrows,
        const dim_t ncols, const int8_t *a, const dim_t lda, const int8_t bo,
        int32_t *a_row_sum)
{
    if (bo != 0) {
        dim_t strideAm = (transa == 0)? 1 : lda;
        dim_t strideAn = (transa != 0)? 1 : lda;

        for (dim_t i = 0; i < nrows; i++) {
            a_row_sum[i] = 0;
            for (dim_t j = 0; j < ncols; j++) {
                a_row_sum[i] += a[i * strideAm + j * strideAn];
            }
        }
    }
}

static inline void get_b_col_sum(const int transb, const dim_t nrows,
        const dim_t ncols, const uint8_t *b, const dim_t ldb, const int8_t ao,
        int32_t *b_col_sum)
{
    if (ao != 0) {
        dim_t strideBm = (transb == 0)? 1 : ldb;
        dim_t strideBn = (transb != 0)? 1 : ldb;

        for (dim_t j = 0; j < ncols; j++) {
            b_col_sum[j] = 0;
            for (dim_t i = 0; i < nrows; i++) {
                b_col_sum[j] += b[i * strideBm + j * strideBn];
            }
        }
    }
}

// TODO Find a better place for those macros.
#define VAL_PADD(y, x, x1)    y = ((x) % (x1)) ? (((x) / (x1)) + 1) * (x1) : (x)
#define LD_PADD(y,x)  (y) = ((((x) + ((2048 / sizeof(int8_t)) - 1)) / (2048 / sizeof(int8_t))) * (2048 / sizeof(int8_t)) +  (512 / sizeof(int8_t)));

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

    // Padding along K dimension.
    dim_t k_padd = 0;
    if (k <= arg->bk_traditional) {
        VAL_PADD(k_padd, k, arg->uk);
        k_padd = nstl::max(128LL, k_padd);
    } else if (k < 2 * arg->bk) {
        k_padd = k / 2;
        VAL_PADD(k_padd, k_padd, arg->uk);
    } else {
        k_padd = arg->bk;
    }

    // Padding along M dimension.
    dim_t m_padd = 0;
    VAL_PADD(m_padd, nstl::min(nstl::max(m, arg->um), arg->bm), arg->um);

    // Padding along N dimension.
    dim_t n_padd = 0;
    if (k < arg->blocking_small_k) {
        VAL_PADD(n_padd, nstl::min(nstl::max(n, arg->un), arg->bn_small_k), arg->un);
    } else {
        VAL_PADD(n_padd, nstl::min(nstl::max(n, arg->un), arg->bn), arg->un);
    }

    // Padding for temporary buffer for C
    dim_t ldc_buf = m_padd;
    LD_PADD(ldc_buf, m_padd);

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
    if (arg->offsetc != 0 || ao != 0 || bo != 0 || co[0] != 0
            || alpha != 1.0 || (beta != 1.0 && beta != 0.0)) {
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
            int offsetc = -1;
            if (Bk + sizeK == k)
                offsetc = arg->offsetc;

            dim_t sizeN = 0;
            for (dim_t Bn = 0; Bn < n; Bn += sizeN) {
                sizeN = n - Bn;
                if (sizeN > n_padd)
                    sizeN = n_padd;

                const uint8_t *b_block = b + Bk * strideBm + Bn * strideBn;
                arg->copyB(&sizeK, &sizeN, b_block, &ldb, NULL, bufferB);
                get_b_col_sum(arg->transb, sizeK, sizeN, b_block, ldb, ao, b_col_sum);

                dim_t sizeUM = 0;
                for (dim_t Um = 0; Um < sizeM; Um += sizeUM) {
                    sizeUM = sizeM - Um;
                    if (sizeUM > arg->um)
                        sizeUM = arg->um;

                    const int8_t *a_block = a + (Bm + Um) * strideAm + Bk * strideAn;
                    if (!a_block_copied) {
                        arg->copyA(&sizeK, &sizeUM, a_block, &lda, NULL, bufferA + Um * sizeK);
                        get_a_row_sum(arg->transa, sizeUM, sizeK, a_block, lda, bo, a_row_sum + Um);
                    }

                    int32_t *c_block = c + (Bm + Um) + Bn * ldc;
                    if (bufferC) {
                        arg->kernel_b0(&sizeUM, &sizeN, &sizeK, NULL, bufferA + Um * sizeK, bufferB, bufferC + Um, ldc_buf);

                        // Finish the block adding the necessary alpha, beta
                        // and offsets.
                        dim_t co_stride = 0;
                        if (offsetc == 0) { // Fix offset.
                            co_stride = 0;
                        } else if (offsetc == 1) { // Row offset.
                            co_stride = Bn;
                        } else if (offsetc == 2) { // Column offset.
                            co_stride = Bm + Um;
                        }
                        add_results(sizeUM, sizeN, sizeK, alpha, beta, bufferC + Um, ldc_buf, c_block, ldc, a_row_sum + Um, b_col_sum, ao, bo, co + co_stride, offsetc);
                    } else {
                        if (beta == 0.0f)
                            arg->kernel_b0(&sizeUM, &sizeN, &sizeK, NULL, bufferA + Um * sizeK, bufferB, c_block, ldc);
                        else
                            arg->kernel(&sizeUM, &sizeN, &sizeK, NULL, bufferA + Um * sizeK, bufferB, c_block, ldc);
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
#undef VAL_PADD
#undef LD_PADD

#define N2D_MAX_AVX512 384
#define M2D_MIN_AVX512 384
#define VECLEN         16
#define NCONS          1
static inline void set_thread_opts_avx512(int *p_nthrs, blas_t *arg)
{
    int nthrs = *p_nthrs;
    dim_t m = arg->m;
    dim_t n = arg->n;

    int condition_2D_bsrc = -1;
    if ((256 * m > nthrs * n) && (nthrs * m < 256 * n)) {
        condition_2D_bsrc = 1;
    } else {
        condition_2D_bsrc = 0;
    }

    arg->thread_copy = COPY_NONE; // By default don't do parallel copy.

    if (condition_2D_bsrc == 1) {
        int nthrs_m = 1;
        int nthrs_n = nthrs;

        while ((nthrs_n % 2 == 0) &&
                (n / nthrs > N2D_MAX_AVX512 || n / nthrs_n <= N2D_MAX_AVX512 / 2) &&
                (m / nthrs_m >= 2 * M2D_MIN_AVX512) &&
                (nthrs_m < 4)) {
            nthrs_m *= 2;
            nthrs_n /= 2;
        }

        arg->nthrs_m = nthrs_m;
        arg->nthrs_n = nthrs_n;
        arg->thread_partition = PARTITION_2D;

        // Reset the total number of threads that will be used.
        *p_nthrs = nthrs_m * nthrs_n;
    } else {
        if ((m > n) && (m / nthrs >= VECLEN || n < NCONS * nthrs)) {
            arg->thread_partition = PARTITION_1D_ROW;
        } else {
            arg->thread_partition = PARTITION_1D_COL;
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
            m_disp = firstmgroup * firstmval + (mthr_used - 1 - firstmgroup) * m_bandt;
            m_band = nstl::max(0LL, m - m_disp);
        }

        if (ithr_j < firstngroup) {
            n_band = firstnval;
            n_disp = ithr_j * firstnval;
        } else if (ithr_j <= nthr_used - 2) {
            n_band = n_bandt;
            n_disp = firstngroup * firstnval + (ithr_j - firstngroup) * n_bandt;
        } else {
            n_disp = firstngroup * firstnval + (nthr_used - 1 - firstngroup) * n_bandt;
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
        const int32_t **co, const blas_t *arg)
{
    dim_t strideAm = (arg->transa == 0)? 1 : arg->lda;
    dim_t strideBn = (arg->transb != 0)? 1 : arg->ldb;
    int offsetc = arg->offsetc;

    switch (arg->thread_partition) {
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
                if (offsetc == 0) { // Fix offset.
                    co_stride = 0;
                } else if (offsetc == 1) { // Row offset.
                    co_stride = 0;
                } else if (offsetc == 2) { // Column offset.
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
                if (offsetc == 0) { // Fix offset.
                    co_stride = 0;
                } else if (offsetc == 1) { // Row offset.
                    co_stride = offset;
                } else if (offsetc == 2) { // Column offset.
                    co_stride = 0;
                }
                *co = arg->co + co_stride;
                break;
            }

        case PARTITION_2D_COL_MAJOR:
            {
                int nthrs_m = arg->nthrs_m;
                int nthrs_n = arg->nthrs_n;
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
                if (offsetc == 0) { // Fix offset.
                    co_stride = 0;
                } else if (offsetc == 1) { // Row offset.
                    co_stride = n_disp;
                } else if (offsetc == 2) { // Column offset.
                    co_stride = m_disp;
                }
                *co = arg->co + co_stride;
                break;
            }
    }
}

static int gemm_threading_driver(blas_t *arg)
{
    if ((arg->m <= 0) || (arg->n <= 0))
        return mkldnn_success;

    const int nthr = (mkldnn_in_parallel()) ? 1 : mkldnn_get_max_threads();
    /*
     * TODO Add a thread checker.
     */

    if (nthr == 1) {
        return gemm_kernel_driver(arg->m, arg->n, arg->k, arg->a, arg->b,
                arg->c, arg->co, arg);
    }

    int status = 0;
    parallel(nthr, [&](const int ithr, const int nthr) {
        int nthrs = nthr;
        if (nthrs == 1) {
            status = gemm_kernel_driver(arg->m, arg->n, arg->k, arg->a, arg->b,
                arg->c, arg->co, arg);
        } else {
            set_thread_opts_avx512(&nthrs, arg);

            const int8_t *a = NULL;
            const uint8_t *b = NULL;
            int32_t *c = NULL;
            const int32_t *co = NULL;
            dim_t m = -1;
            dim_t n = -1;
            dim_t k = -1;
            decompose_matrices(ithr, &nthrs, &m, &n, &k, &a, &b, &c, &co, arg);

            if (ithr < nthrs) {
                int result = gemm_kernel_driver(m, n, k, a, b, c, co, arg);

                if (result < 0) {
                    status = result;
                }
            }
        }
    });

    return status;
}

static jit_avx512_core_u8_copy_an_kern *copy_an;
static jit_avx512_core_u8_copy_at_kern *copy_at;
static jit_avx512_core_u8_copy_bn_kern *copy_bn;
static jit_avx512_core_u8_copy_bt_kern *copy_bt;
static jit_avx512_core_kernel_gemm_s8u8s32_kern *kernel;
static jit_avx512_core_kernel_b0_gemm_s8u8s32_kern *kernel_b0;

static void jit_init(blas_t *arg)
{
    static int (*copyAn )(const dim_t *m, const dim_t *n, const int8_t *a , const dim_t *lda, const int8_t *alpha, int8_t *b);
    static int (*copyAt )(const dim_t *m, const dim_t *n, const int8_t *a , const dim_t *lda, const int8_t *alpha, int8_t *b);
    static int (*copyBn )(const dim_t *m, const dim_t *n, const uint8_t *a, const dim_t *lda, const uint8_t *alpha, uint8_t *b);
    static int (*copyBt )(const dim_t *m, const dim_t *n, const uint8_t *a, const dim_t *lda, const uint8_t *alpha, uint8_t *b);
    static int (*kern   )(const dim_t *m, const dim_t *n, const dim_t *k, const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c, const dim_t ldc);
    static int (*kern_b0)(const dim_t *m, const dim_t *n, const dim_t *k, const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c, const dim_t ldc);

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
        copy_an   = new jit_avx512_core_u8_copy_an_kern();
        copy_at   = new jit_avx512_core_u8_copy_at_kern();
        copy_bn   = new jit_avx512_core_u8_copy_bn_kern();
        copy_bt   = new jit_avx512_core_u8_copy_bt_kern();
        kernel    = new jit_avx512_core_kernel_gemm_s8u8s32_kern();
        kernel_b0 = new jit_avx512_core_kernel_b0_gemm_s8u8s32_kern();

        copyAn  = copy_an   -> getCode<int (*)(const dim_t *, const dim_t *, const int8_t  *, const dim_t *, const int8_t  *, int8_t  *)>();
        copyAt  = copy_at   -> getCode<int (*)(const dim_t *, const dim_t *, const int8_t  *, const dim_t *, const int8_t  *, int8_t  *)>();
        copyBn  = copy_bn   -> getCode<int (*)(const dim_t *, const dim_t *, const uint8_t *, const dim_t *, const uint8_t *, uint8_t *)>();
        copyBt  = copy_bt   -> getCode<int (*)(const dim_t *, const dim_t *, const uint8_t *, const dim_t *, const uint8_t *, uint8_t *)>();
        kern    = kernel    -> getCode<int (*)(const dim_t *, const dim_t *, const dim_t *, const float *, const int8_t *, const uint8_t *, int32_t *, const dim_t)>();
        kern_b0 = kernel_b0 -> getCode<int (*)(const dim_t *, const dim_t *, const dim_t *, const float *, const int8_t *, const uint8_t *, int32_t *, const dim_t)>();
    });

    if (arg->transa == 0) {
        arg->copyA = copyAn;
    } else {
        arg->copyA = copyAt;
    }

    if (arg->transb == 0) {
        arg->copyB = copyBn;
    } else {
        arg->copyB = copyBt;
    }

    arg->kernel    = kern;
    arg->kernel_b0 = kern_b0;
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
        args.offsetc = 0;
    } else if (offsetc == 'R' || offsetc == 'r') {
        args.offsetc = 1;
    } else { // offsetc == 'C' || offsetc == 'c'
        args.offsetc = 2;
    }

    jit_init(&args);
    int result = gemm_threading_driver(&args);

    return (result < 0 ) ? mkldnn_out_of_memory : mkldnn_success;
}

}
}
}
