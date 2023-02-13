/*******************************************************************************
* Copyright 2023 IBM Corporation
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

#if defined(__VX__)

#include <cstdint>
#include "common/dnnl_thread.hpp"
#include "cpu/s390x/kernel_s16s16s32.hpp"
#include "cpu/s390x/pack.hpp"
#include "cpu/simple_q10n.hpp"
#include "gemm.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace s390x {

constexpr dim_t MC = 96 * 8;
constexpr dim_t KC = 164 * 4 * 2;
constexpr dim_t NC = 512;

enum class offset_type {
    none,
    fixed,
    column,
    row,
};

__attribute__((noinline)) void addResults(offset_type offsetType, dim_t m,
        dim_t n, double alpha, double beta, int32_t *__restrict C, dim_t ldC,
        int32_t *__restrict Ctemp, dim_t ldCtemp,
        const int32_t *__restrict co) {

    if (offsetType == offset_type::fixed) {
        if (beta == 0) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val = alpha * (double)Ctemp[j * ldCtemp + i] + co[0];
                    gPtr(i, j) = static_cast<int32_t>(
                            nearbyint(saturate<int32_t, double>(val)));
                }
            }
        } else {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val = beta * (double)gPtr(i, j)
                            + alpha * (double)Ctemp[j * ldCtemp + i] + co[0];
                    gPtr(i, j) = static_cast<int32_t>(
                            nearbyint(saturate<int32_t, double>(val)));
                }
            }
        }
    } else if (offsetType == offset_type::column) {
        if (beta == 0) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val = alpha * (double)Ctemp[j * ldCtemp + i] + co[i];
                    gPtr(i, j) = static_cast<int32_t>(
                            nearbyint(saturate<int32_t, double>(val)));
                }
            }
        } else {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val = beta * (double)gPtr(i, j)
                            + alpha * (double)Ctemp[j * ldCtemp + i] + co[i];
                    gPtr(i, j) = static_cast<int32_t>(
                            nearbyint(saturate<int32_t, double>(val)));
                }
            }
        }

    } else if (offsetType == offset_type::row) {
        if (beta == 0) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val = alpha * (double)Ctemp[j * ldCtemp + i] + co[j];
                    gPtr(i, j) = static_cast<int32_t>(
                            nearbyint(saturate<int32_t, double>(val)));
                }
            }
        } else {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val = beta * (double)gPtr(i, j)
                            + alpha * (double)Ctemp[j * ldCtemp + i] + co[j];
                    gPtr(i, j) = static_cast<int32_t>(
                            nearbyint(saturate<int32_t, double>(val)));
                }
            }
        }
    } else {
        if (beta == 0) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    gPtr(i, j) = static_cast<int32_t>(
                            nearbyint(saturate<int32_t, double>(
                                    alpha * (double)Ctemp[j * ldCtemp + i])));
                }
            }
        } else {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val = beta * (double)gPtr(i, j)
                            + alpha * (double)Ctemp[j * ldCtemp + i];
                    gPtr(i, j) = static_cast<int32_t>(
                            nearbyint(saturate<int32_t, double>(val)));
                }
            }
        }
    }
}

template <typename TA, typename TB>
inline void LoopKC(bool transA, bool transB, dim_t m, dim_t n, dim_t k,
        const TA *A, dim_t ldA, const TA *ao, const TB *B, dim_t ldB,
        const TB *bo, int32_t *C, dim_t ldC, int16_t *Apacked,
        int16_t *Bpacked) {
    for (dim_t p = 0; p < k; p += KC) {
        dim_t pb = nstl::min(KC, k - p);
        dim_t kk = (pb + 1) & -2;
        if (bo) {
            int16_t add_val = -(*bo);
            if (transB) {
                pack_K<TB, int16_t, NR, 2, true>(
                        pb, n, &bPtr(0, p), ldB, Bpacked, add_val);
            } else {
                pack_K<TB, int16_t, NR, 2, false>(
                        pb, n, &bPtr(p, 0), ldB, Bpacked, add_val);
            }
        } else {
            if (transB) {
                pack_K<TB, int16_t, NR, 2, true>(
                        pb, n, &bPtr(0, p), ldB, Bpacked);
            } else {
                pack_K<TB, int16_t, NR, 2, false>(
                        pb, n, &bPtr(p, 0), ldB, Bpacked);
            }
        }

        if (ao) {
            int16_t add_val = -(*ao);
            if (transA) {
                pack_K<TA, int16_t, MR, 2, false>(
                        pb, m, &aPtr(p, 0), ldA, Apacked, add_val);
            } else {
                pack_K<TA, int16_t, MR, 2, true>(
                        pb, m, &aPtr(0, p), ldA, Apacked, add_val);
            }
        } else {
            if (transA) {
                pack_K<TA, int16_t, MR, 2, false>(
                        pb, m, &aPtr(p, 0), ldA, Apacked);
            } else {
                pack_K<TA, int16_t, MR, 2, true>(
                        pb, m, &aPtr(0, p), ldA, Apacked);
            }
        }

        LoopTwo<NR>(m, n, kk, Apacked, Bpacked, C, ldC);
    }
}

template <typename TA, typename TB>
inline void LoopMC(offset_type offsetType, bool transA, bool transB, dim_t m,
        dim_t n, dim_t k, float alpha, const TA *A, dim_t ldA, const TA *ao,
        const TB *B, dim_t ldB, const TB *bo, float beta, int32_t *C, dim_t ldC,
        int16_t *Apacked, int16_t *Bpacked, int32_t *Ctemp, dim_t ldCtemp,
        const int32_t *co) {
    for (dim_t i = 0; i < m; i += MC) {
        dim_t ib = nstl::min(MC, m - i);

        for (dim_t u = 0; u < ib * n; u++) {
            Ctemp[u] = 0;
        }
        LoopKC(transA, transB, ib, n, k, transA ? &aPtr(0, i) : &aPtr(i, 0),
                ldA, ao, B, ldB, bo, Ctemp, ib, Apacked, Bpacked);
        auto localCo = (offsetType == offset_type::column) ? &co[i] : co;
        addResults(offsetType, ib, n, (double)alpha, (double)beta, &gPtr(i, 0),
                ldC, Ctemp, ib, localCo);
    }
}

template <typename TA, typename TB>
inline void LoopNC(offset_type offsetType, bool transA, bool transB, dim_t m,
        dim_t n, dim_t k, float alpha, const TA *A, dim_t ldA, const TA *ao,
        const TB *B, dim_t ldB, const TB *bo, float beta, int32_t *C, dim_t ldC,
        const int32_t *co) {

    //lets restrict sizes by KC
    int kC = (k + 4) > KC ? KC : ((k + 3) & -4);

    auto Bpack = (int16_t *)malloc((kC * NC) * sizeof(int16_t) + 16, 4096);
    auto Apack = (int16_t *)malloc((MC * kC) * sizeof(int16_t) + 16, 4096);
    // unfortunately we have create memory for C as well for the correctness
    // scaling C with beta beforehand is not possible here
    // and also we have k blocked which makes it safer to allocate for C
    int mC = m + 16 > MC ? MC : (m + 15) & (-16);
    int nC = n + 16 > NC ? NC : (n + 15) & (-16);
    auto Ctemp = (int32_t *)malloc((mC * nC) * sizeof(int32_t) + 16, 4096);

    //align
    auto AP = utils::align_ptr(Apack, 16);
    auto BP = utils::align_ptr(Bpack, 16);
    auto CP = utils::align_ptr(Ctemp, 16);

    if (utils::any_null(Apack, Bpack, Ctemp)) {
        free(Apack);
        free(Bpack);
        free(Ctemp);
        return;
    }
    //we will use (NC->MC->KC) blocking  instead of (NC->KC->MC )to control memory for C temp
    for (dim_t j = 0; j < n; j += NC) {

        dim_t jb = nstl::min(
                NC, n - j); /* Last loop may not involve a full block */
        auto localCo = (offsetType == offset_type::row) ? &co[j] : co;
        LoopMC(offsetType, transA, transB, m, jb, k, alpha, A, ldA, ao,
                transB ? &bPtr(j, 0) : &bPtr(0, j), ldB, bo, beta, &gPtr(0, j),
                ldC, AP, BP, CP, mC, localCo);
    }

    free(Apack);
    free(Bpack);
    free(Ctemp);
}

template <typename TA, typename TB>
dnnl_status_t gemmX8X8s32(const char *transa, const char *transb,
        const char *offsetc, dim_t M, dim_t N, dim_t K, float alpha,
        const TA *A, dim_t ldA, const TA *ao, const TB *B, dim_t ldB,
        const TB *bo, float beta, int32_t *C, dim_t ldC, const int32_t *co) {

    offset_type offType = offset_type::none;
    if (*offsetc == 'F' || *offsetc == 'f') offType = offset_type::fixed;
    if (*offsetc == 'R' || *offsetc == 'r') offType = offset_type::row;
    if (*offsetc == 'C' || *offsetc == 'c') offType = offset_type::column;
    bool trA = *transa == 't' || *transa == 'T';
    bool trB = *transb == 't' || *transb == 'T';
    int thr_count = dnnl_get_current_num_threads();
    int nC = thr_count > 1 && N > (NC / 4) ? ((N / thr_count + NR - 1) & (-NR))
                                           : N;
    const dim_t nPanels = (N + nC - 1) / nC;
    const dim_t tileY = N - (nPanels - 1) * nC;
    dnnl::impl::parallel_nd(nPanels, [&](int64_t n) {
        dim_t localN = n + 1 == nPanels ? tileY : nC;
        auto j = n * nC;
        auto localB = trB ? &bPtr(j, 0) : &bPtr(0, j);
        auto localA = A;
        auto localC = &gPtr(0, j);
        auto localCo = (offType == offset_type::row) ? &co[j] : co;

        LoopNC<TA, TB>(offType, trA, trB, M, localN, K, alpha, localA, ldA, ao,
                localB, ldB, bo, beta, localC, ldC, localCo);
    });
    return dnnl_success;
}

dnnl_status_t gemmx8x8s32(const char *transa, const char *transb,
        const char *offsetc, dim_t M, dim_t N, dim_t K, float alpha,
        const int8_t *A, dim_t ldA, const int8_t *ao, const uint8_t *B,
        dim_t ldB, const uint8_t *bo, float beta, int32_t *C, dim_t ldC,
        const int32_t *co) {

    return gemmX8X8s32<int8_t, uint8_t>(transa, transb, offsetc, M, N, K, alpha,
            A, ldA, ao, B, ldB, bo, beta, C, ldC, co);
}

dnnl_status_t gemmx8x8s32(const char *transa, const char *transb,
        const char *offsetc, dim_t M, dim_t N, dim_t K, float alpha,
        const int8_t *A, dim_t ldA, const int8_t *ao, const int8_t *B,
        dim_t ldB, const int8_t *bo, float beta, int32_t *C, dim_t ldC,
        const int32_t *co) {

    return gemmX8X8s32<int8_t, int8_t>(transa, transb, offsetc, M, N, K, alpha,
            A, ldA, ao, B, ldB, bo, beta, C, ldC, co);
}

} // namespace s390x
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
