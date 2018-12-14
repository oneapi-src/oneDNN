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

#include <cassert>
#include <mutex>

#include "mkldnn.h"

#include "mkldnn_traits.hpp"
#include "math_utils.hpp"
#include "nstl.hpp"
#include "verbose.hpp"

#include "os_blas.hpp"
#include "gemm.hpp"

#include "f32/jit_avx_gemm_f32.hpp"
#include "f32/jit_avx512_common_gemm_f32.hpp"
#include "s8x8s32/jit_avx512_core_gemm_s8u8s32.hpp"

/* USE_MKL      USE_CBLAS       effect
 * -------      ---------       ------
 * yes          yes             use Intel(R) MKL CBLAS
 * yes          no              use jit
 * no           yes             system-dependent CBLAS
 * no           no              use jit
 */

namespace mkldnn {
namespace impl {
namespace cpu {

mkldnn_status_t check_gemm_input(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const int *lda,
        const int *ldb, const int *ldc, const float *alpha, const float *beta,
        const bool with_bias) {
    if (utils::any_null(transa, transb, M, N, K, lda, ldb, ldc, alpha, beta))
        return mkldnn_invalid_arguments;
    if (with_bias && *beta != 0)
        return mkldnn_unimplemented;
    bool consistency = true
        && utils::one_of(*transa, 'T', 't', 'N', 'n')
        && utils::one_of(*transb, 'T', 't', 'N', 'n')
        && *M >= 0
        && *N >= 0
        && *K >= 0;

    if (!consistency)
        return mkldnn_invalid_arguments;
    bool isTransA = utils::one_of(*transa, 'T', 't');
    bool isTransB = utils::one_of(*transb, 'T', 't');
    int nrowA = isTransA ? *K : *M;
    int nrowB = isTransB ? *N : *K;
    consistency = true
        && *lda >= nstl::max(1, nrowA)
        && *ldb >= nstl::max(1, nrowB)
        && *ldc >= nstl::max(1, *M);
    if (!consistency)
        return mkldnn_invalid_arguments;

    return mkldnn_success;
}

mkldnn_status_t check_gemm_x8x8x32_input(const char *offsetc,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, const int *ldc,
        const float *alpha, const float *beta, const bool with_bias) {
    if (offsetc == nullptr)
        return mkldnn_invalid_arguments;
    if (!utils::one_of(*offsetc, 'F', 'f', 'C', 'c', 'R', 'r'))
        return mkldnn_invalid_arguments;

    return check_gemm_input(transa, transb, M, N, K, lda, ldb, ldc, alpha,
        beta, with_bias);
}

struct gemm_impl_t {
    gemm_impl_t(char transa, char transb, bool zero_beta, bool with_bias) {
        //jit kernel has three codepaths: beta is 0, 1 or arbitrary
        //we will generate kernel for 0 and arbitrary beta
        float zero = 0.0f, arbitrary_float = 2.0f;
        if (mayiuse(avx512_common)) {
            isa_ = avx512_common;
            ker_ = (void *)new jit_avx512_common_gemm_f32(
                    transa, transb, zero_beta ? zero : arbitrary_float,
                    with_bias);
        }
        else if (mayiuse(avx)) {
            isa_ = avx;
            ker_ = (void *)new jit_avx_gemm_f32(
                    transa, transb, zero_beta ? zero : arbitrary_float,
                    with_bias);
        }
    }

    mkldnn_status_t call(const char *transa, const char *transb, const int *M,
            const int *N, const int *K, const float *alpha, const float *A,
            const int *lda, const float *B, const int *ldb, const float *beta,
            float *C, const int *ldc, const float *bias = nullptr) {
        switch (isa_) {
            case avx:
                ((jit_avx_gemm_f32*)ker_)->sgemm(transa, transb, M, N, K,
                    alpha, A, lda, B, ldb, beta, C, ldc, bias);
                break;
            case avx512_common:
                ((jit_avx512_common_gemm_f32*)ker_)->sgemm(transa, transb,
                    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, bias);
                break;
            default:
                ref_gemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta,
                        C, ldc, bias);
                break;
        }
        return mkldnn_success;
    }

    void *ker_;
    cpu_isa_t isa_;
};

//Gemm implementations for: zero/nonzero beta, transA, transB
static gemm_impl_t *gemm_impl[2][2][2];
//Gemm with bias implementations for: transA, transB
//Gemm with bias for beta!=0. is not supported
static gemm_impl_t *gemm_bias_impl[2][2];

void initialize() {
    for (int i = 0; i < 2; ++i) {
        gemm_impl[i][0][0] = new gemm_impl_t('n', 'n', (bool)i, false);
        gemm_impl[i][0][1] = new gemm_impl_t('n', 't', (bool)i, false);
        gemm_impl[i][1][0] = new gemm_impl_t('t', 'n', (bool)i, false);
        gemm_impl[i][1][1] = new gemm_impl_t('t', 't', (bool)i, false);
    }
    gemm_bias_impl[0][0] = new gemm_impl_t('n', 'n', true, true);
    gemm_bias_impl[0][1] = new gemm_impl_t('n', 't', true, true);
    gemm_bias_impl[1][0] = new gemm_impl_t('t', 'n', true, true);
    gemm_bias_impl[1][1] = new gemm_impl_t('t', 't', true, true);
}

mkldnn_status_t extended_sgemm(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *alpha,
        const float *A, const int *lda, const float *B, const int *ldb,
        const float *beta, float *C, const int *ldc,
        const float *bias, const bool force_jit_gemm) {
    mkldnn_status_t status = check_gemm_input(transa, transb, M, N, K,
            lda, ldb, ldc, alpha, beta, bias != nullptr);
    if (status != mkldnn_success)
        return status;

    int trA = *transa == 't' || *transa == 'T';
    int trB = *transb == 't' || *transb == 'T';
#ifdef USE_CBLAS
    if (!force_jit_gemm) {
        CBLAS_TRANSPOSE Cblas_trA = trA ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE Cblas_trB = trB ? CblasTrans : CblasNoTrans;
        cblas_sgemm(CblasColMajor, Cblas_trA, Cblas_trB,
                *M, *N, *K, *alpha, A, *lda, B, *ldb, *beta, C, *ldc);

        if (bias) {
            // Add bias if necessary (bias is applied to columns of C)
            cblas_int incx = 1, incy = 1;
            parallel_nd(*N, [&](int n) {
                ptrdiff_t offset = (ptrdiff_t)n * (*ldc);
                cblas_saxpy(*M, 1.0, bias, incx, C + offset, incy);
            });
        }
        return mkldnn_success;
    }
#endif
    //Generate jit kernel and call sgemm with bias
    static std::once_flag initialized;
    std::call_once(initialized, [] { mkldnn::impl::cpu::initialize(); });

    if (bias)
        gemm_bias_impl[trA][trB]->call(
                transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
                bias);
    else
        gemm_impl[*beta == 0.f][trA][trB]->call(
                transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    return mkldnn_success;
}

template <typename b_dt>
mkldnn_status_t gemm_s8x8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *LDA, const int8_t *ao,
        const b_dt *B, const int *LDB, const int8_t *bo, const float *beta,
        int32_t *C, const int *LDC, const int32_t *co) {
    mkldnn_status_t status = check_gemm_x8x8x32_input(offsetc, transa, transb,
        M, N, K, LDA, LDB, LDC, alpha, beta, false);
    if (status != mkldnn_success)
        return status;

#if USE_MKL_IGEMM
    if (data_traits<b_dt>::data_type == data_type::u8) {
        bool OCisR = (*offsetc == 'R' || *offsetc == 'r');
        bool OCisC = (*offsetc == 'C' || *offsetc == 'c');
        bool AisN = (*transa == 'N' || *transa == 'n');
        bool BisN = (*transb == 'N' || *transb == 'n');

        CBLAS_TRANSPOSE Cblas_trA = AisN ? CblasNoTrans : CblasTrans;
        CBLAS_TRANSPOSE Cblas_trB = BisN ? CblasNoTrans : CblasTrans;
        CBLAS_OFFSET Cblas_offsetc =
            OCisR
            ? CblasRowOffset
            : OCisC
            ? CblasColOffset
            : CblasFixOffset;
        cblas_gemm_s8u8s32(CblasColMajor, Cblas_trA, Cblas_trB, Cblas_offsetc,
                *M, *N, *K, *alpha, A, *LDA, *ao, (uint8_t *)B, *LDB, *bo,
                *beta, C, *LDC, co);
        return mkldnn_success;
    } else {
        assert(data_traits<b_dt>::data_type == data_type::s8);
        // TODO CBLAS implementation of gemm_s8s8s32 goes here.
        // Calling reference implementation.
        return ref_gemm_s8x8s32(transa, transb, offsetc, M, N, K, alpha, A,
                LDA, ao, B, LDB, bo, beta, C, LDC, co);
    }
#else
    if (data_traits<b_dt>::data_type == data_type::u8) {
        cpu_isa_t isa = isa_any;

        if (mayiuse(avx512_core_vnni)) {
            isa = avx512_core_vnni;
        } else if (mayiuse(avx512_core)) {
            isa = avx512_core;
        }

        switch (isa) {
            case avx512_core:
            case avx512_core_vnni:
                return jit_avx512_core_gemm_s8u8s32(transa, transb, offsetc, M,
                        N, K, alpha, A, LDA, ao, (uint8_t *)B, LDB, bo, beta,
                        C, LDC, co, isa);
                break;

            default:
                return ref_gemm_s8x8s32(transa, transb, offsetc, M, N, K,
                        alpha, A, LDA, ao, B, LDB, bo, beta, C, LDC, co);
        }
    } else {
        assert(data_traits<b_dt>::data_type == data_type::s8);
        return ref_gemm_s8x8s32(transa, transb, offsetc, M, N, K, alpha, A,
                LDA, ao, B, LDB, bo, beta, C, LDC, co);
    }
#endif
}

}
}
}

using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;

mkldnn_status_t mkldnn_sgemm(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *alpha,
        const float *A, const int *lda, const float *B, const int *ldb,
        const float *beta, float *C, const int *ldc) {
    return extended_sgemm(
            transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

mkldnn_status_t mkldnn_gemm_s8u8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *lda, const int8_t *ao,
        const uint8_t *B, const int *ldb, const int8_t *bo, const float *beta,
        int32_t *C, const int *ldc, const int32_t *co) {
    return gemm_s8x8s32(
        transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo,
        beta, C, ldc, co);
}

mkldnn_status_t mkldnn_gemm_s8s8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *lda, const int8_t *ao,
        const int8_t *B, const int *ldb, const int8_t *bo, const float *beta,
        int32_t *C, const int *ldc, const int32_t *co) {
    return gemm_s8x8s32(
        transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo,
        beta, C, ldc, co);
}

