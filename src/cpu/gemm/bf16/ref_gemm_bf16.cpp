/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dnnl/dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/gemm/bf16/ref_gemm_bf16.hpp"
#include "cpu/gemm/f32/gemm_utils_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace gemm_utils;

namespace {

void copy_A(bool isTransA, dim_t K, const bfloat16_t *A, const dim_t lda,
        bfloat16_t *ws) {
    for (dim_t k = 0; k < K; k++) {
        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < unroll_factor<bfloat16_t>::m; i++) {
            ws[i] = isTransA ? A[i * lda + k] : A[i + k * lda];
        }
        ws += unroll_factor<bfloat16_t>::m;
    }
}

template <bool isTransA, bool isTransB>
void kernel_mxn(dim_t K, const bfloat16_t *A, const dim_t lda,
        const bfloat16_t *B, const dim_t ldb, float *C, const dim_t ldc,
        const float alpha, const float beta) {
    float c[unroll_factor<bfloat16_t>::m * unroll_factor<bfloat16_t>::n]
            = {0.f};
    for (dim_t k = 0; k < K; k++) {
        for (dim_t j = 0; j < unroll_factor<bfloat16_t>::n; j++) {
            bfloat16_t b = isTransB ? B[j + k * ldb] : B[k + j * ldb];
            PRAGMA_OMP_SIMD()
            for (dim_t i = 0; i < unroll_factor<bfloat16_t>::m; i++) {
                bfloat16_t a = isTransA ? A[i * lda + k] : A[i + lda * k];
                c[i + unroll_factor<bfloat16_t>::m * j] += a * b;
            }
        }
    }
    for (dim_t j = 0; j < unroll_factor<bfloat16_t>::n; j++) {
        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < unroll_factor<bfloat16_t>::m; i++) {
            C[i + j * ldc] = (beta == 0.f)
                    ? alpha * c[i + unroll_factor<bfloat16_t>::m * j]
                    : alpha * c[i + unroll_factor<bfloat16_t>::m * j]
                            + beta * C[i + j * ldc];
        }
    }
}

template <bool isTransA, bool isTransB>
void block_ker(const dim_t M, const dim_t N, const dim_t K, const bfloat16_t *A,
        const dim_t lda, const bfloat16_t *B, const dim_t ldb, float *C,
        const dim_t ldc, const float alpha, const float beta, bfloat16_t *ws,
        bool do_copy) {
    dim_t Nu = rnd_dn(N, unroll_factor<bfloat16_t>::n);
    dim_t Mu = rnd_dn(M, unroll_factor<bfloat16_t>::m);
    for (dim_t i = 0; i < Mu; i += unroll_factor<bfloat16_t>::m) {
        for (dim_t j = 0; j < Nu; j += unroll_factor<bfloat16_t>::n) {
            const bfloat16_t *b = isTransB ? &B[j] : &B[j * ldb];
            const bfloat16_t *a = isTransA ? &A[i * lda] : &A[i];
            if (do_copy) {
                if (j == 0) { copy_A(isTransA, K, a, lda, ws); }
                kernel_mxn<false, isTransB>(K, ws, unroll_factor<bfloat16_t>::m,
                        b, ldb, &C[i + j * ldc], ldc, alpha, beta);
            } else {
                kernel_mxn<isTransA, isTransB>(
                        K, a, lda, b, ldb, &C[i + j * ldc], ldc, alpha, beta);
            }
        }
    }
    // tail processing
    for (dim_t i = 0; i < M; i++) {
        for (dim_t j = Nu; j < N; j++) {
            float c = beta == 0.f ? 0.f : beta * C[i + j * ldc];
            for (dim_t p = 0; p < K; p++) {
                bfloat16_t b = isTransB ? B[j + p * ldb] : B[p + j * ldb];
                bfloat16_t a = isTransA ? A[p + i * lda] : A[i + p * lda];
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
    for (dim_t i = Mu; i < M; i++) {
        for (dim_t j = 0; j < Nu; j++) {
            float c = beta == 0.f ? 0.f : beta * C[i + j * ldc];
            for (dim_t p = 0; p < K; p++) {
                bfloat16_t b = isTransB ? B[j + p * ldb] : B[p + j * ldb];
                bfloat16_t a = isTransA ? A[p + i * lda] : A[i + p * lda];
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
}

template <bool isTransA, bool isTransB>
void gemm_ithr(const dim_t M, const dim_t N, const dim_t K, const float alpha,
        const bfloat16_t *A, const dim_t lda, const bfloat16_t *B,
        const dim_t ldb, const float beta, float *C, const dim_t ldc,
        bool do_copy, bfloat16_t *ws) {
    constexpr dim_t BM = gemm_traits<bfloat16_t, isTransA, isTransB>::BM;
    constexpr dim_t BN = gemm_traits<bfloat16_t, isTransA, isTransB>::BN;
    constexpr dim_t BK = gemm_traits<bfloat16_t, isTransA, isTransB>::BK;

    const bfloat16_t *curA;
    const bfloat16_t *curB;
    float *curC;

    if ((M <= 0) || (N <= 0)) return;

    if ((K <= 0) || (alpha == 0.f)) {
        dim_t MN = N * M;
        if (beta == 0.f) {
            for (dim_t j = 0; j < MN; j++)
                C[j] = 0.f;
        } else if (beta != 1.f) {
            for (dim_t j = 0; j < MN; j++)
                C[j] *= beta;
        }
        return;
    }

    for (dim_t Bk = 0; Bk < K; Bk += BK) {
        dim_t kb = nstl::min(K - Bk, BK);
        for (dim_t Bm = 0; Bm < M; Bm += BM) {
            dim_t mb = nstl::min(M - Bm, BM);
            for (dim_t Bn = 0; Bn < N; Bn += BN) {
                dim_t nb = nstl::min(N - Bn, BN);
                curA = isTransA ? A + Bk + Bm * lda : A + Bm + Bk * lda;
                curB = isTransB ? B + Bn + Bk * ldb : B + Bk + Bn * ldb;
                curC = C + Bm + Bn * ldc;
                if (Bk == 0) {
                    block_ker<isTransA, isTransB>(mb, nb, kb, curA, lda, curB,
                            ldb, curC, ldc, alpha, beta, ws, do_copy);
                } else {
                    block_ker<isTransA, isTransB>(mb, nb, kb, curA, lda, curB,
                            ldb, curC, ldc, alpha, 1.f, ws, do_copy);
                }
            }
        }
    }
}

} // namespace

dnnl_status_t ref_gemm_bf16bf16f32(const char *transa_, const char *transb_,
        const dim_t *M_, const dim_t *N_, const dim_t *K_, const float *alpha_,
        const bfloat16_t *A, const dim_t *lda_, const bfloat16_t *B,
        const dim_t *ldb_, const float *beta_, float *C, const dim_t *ldc_) {

    if (!(utils::one_of(*transa_, 'n', 'N', 't', 'T')
                && utils::one_of(*transb_, 'n', 'N', 't', 'T')))
        return dnnl_unimplemented;

    bool isTransA = (*transa_ == 'T' || *transa_ == 't');
    bool isTransB = (*transb_ == 'T' || *transb_ == 't');
    const dim_t M = *M_, N = *N_, K = *K_;
    const dim_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const float alpha = *alpha_, beta = *beta_;

    // early out and avoid division by zero in partitioning
    if (utils::one_of(0, M, N)) return dnnl_success;

    int max_nthr = dnnl_get_current_num_threads();
    int nthr_m, nthr_n, nthr_k;
    dim_t MB, NB, KB;
    // thread balancing over M, N, K & size of blocking dimensions
    calc_nthr_nocopy_avx(
            M, N, K, max_nthr, &nthr_m, &nthr_n, &nthr_k, &MB, &NB, &KB);
    assert(IMPLICATION(!dnnl_thr_syncable(), nthr_k == 1));

    float *c_buffers = nullptr;
    bfloat16_t *ws_buffers = nullptr;
    if (nthr_k > 1) {
        c_buffers = (float *)malloc(
                sizeof(*c_buffers) * nthr_m * nthr_n * (nthr_k - 1) * MB * NB,
                PAGE_4K);
        if (!c_buffers) {
            nthr_k = 1;
            KB = K;
        }
    }

    bool do_copy = (NB / unroll_factor<bfloat16_t>::n > 3);
    const int nthr_mn = nthr_m * nthr_n;
    const int nthr_to_use = nthr_mn * nthr_k;
    const size_t ws_elems_per_thr = K * unroll_factor<bfloat16_t>::m;
    const size_t ws_size_per_thr
            = rnd_up(ws_elems_per_thr * sizeof(float), PAGE_4K);
    if (do_copy) {
        ws_buffers
                = (bfloat16_t *)malloc(nthr_to_use * ws_size_per_thr, PAGE_4K);
        if (!ws_buffers) do_copy = false;
    }

    auto get_thr_block = [&](dim_t &from, dim_t &to, dim_t &myN, dim_t NB,
                                 dim_t N, int ithr) {
        from = NB * (ithr);
        to = NB * (ithr + 1);
        if (to > N) to = N;
        myN = to - from;
    };

    parallel(nthr_to_use, [&](int ithr, int nthr) {
        assert(nthr_to_use == nthr);
        MAYBE_UNUSED(nthr);

        int ithr_mn = ithr % nthr_mn;
        int ithr_m = ithr_mn % nthr_m;
        int ithr_n = ithr_mn / nthr_m;
        int ithr_k = ithr / nthr_mn;

        int cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

        bfloat16_t *ws = do_copy
                ? ws_buffers + ithr * ws_size_per_thr / sizeof(float)
                : nullptr;

        dim_t m_from = 0, m_to = 0, myM = 0, n_from = 0, n_to = 0, myN = 0,
              k_from = 0, k_to = 0, myK = 0;

        get_thr_block(m_from, m_to, myM, MB, M, ithr_m);
        get_thr_block(n_from, n_to, myN, NB, N, ithr_n);
        get_thr_block(k_from, k_to, myK, KB, K, ithr_k);

        if (myM > 0 && myN > 0) {
            float myBeta, *myC;
            dim_t ld;
            if (ithr_k == 0) {
                myC = &(C[m_from + n_from * ldc]);
                myBeta = beta;
                ld = ldc;
            } else {
                myC = c_buffers + MB * NB * (cbase + ithr_k - 1);
                myBeta = 0.0f;
                ld = MB;
            }
            const bfloat16_t *myA = isTransA ? &(A[k_from + m_from * lda])
                                             : &(A[m_from + k_from * lda]);
            const bfloat16_t *myB = isTransB ? &(B[n_from + k_from * ldb])
                                             : &(B[k_from + n_from * ldb]);

            if (!isTransA) {
                if (!isTransB) {
                    gemm_ithr<false, false>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws);
                } else {
                    gemm_ithr<false, true>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws);
                }
            } else {
                if (!isTransB) {
                    gemm_ithr<true, false>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws);
                } else {
                    gemm_ithr<true, true>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws);
                }
            }
        }
    });

    if (nthr_k > 1) {
        parallel(nthr_to_use, [&](int ithr, int nthr) {
            assert(nthr_to_use == nthr);
            MAYBE_UNUSED(nthr);

            int ithr_mn = ithr % nthr_mn;
            int ithr_m = ithr_mn % nthr_m;
            int ithr_k = ithr / nthr_mn;
            int ithr_n = ithr_mn / nthr_m;

            dim_t n_from = 0, n_to = 0, myN = 0;
            dim_t m_from = 0, m_to = 0, myM = 0;

            int cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

            get_thr_block(n_from, n_to, myN, NB, N, ithr_n);
            get_thr_block(m_from, m_to, myM, MB, M, ithr_m);

            // sum matrices partitioned along K dimension
            dim_t offset = 0, block = 0;
            partition_unit_diff(ithr_k, nthr_k, myN, &offset, &block);
            for (int ik = 1; ik < nthr_k; ++ik) {
                float *myC = c_buffers
                        + MB * ((dim_t)NB * (cbase + ik - 1) + offset);

                gemm_utils::sum_two_matrices(myM, block, myC, MB,
                        &C[m_from + (n_from + offset) * ldc], ldc);
            }
        });
    }

    free(ws_buffers);
    free(c_buffers);

    return dnnl_success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
