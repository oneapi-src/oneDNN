/*******************************************************************************
* Copyright 2022 IBM Corporation
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

#ifdef __MMA__
#include <altivec.h>
#include "cpu/ppc64/ppc64_gemm_s8x8s32.hpp"
#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace ppc64 {

dnnl_status_t cblas_gemm_s8x8s32_ppc64(int ATflag, int BTflag,
        char const *offsetc, dim_t m, dim_t n, dim_t k, float alpha,
        int8_t const *A, dim_t lda, int8_t const *ao, uint8_t const *B,
        dim_t ldb, uint8_t const *bo, int *C, float beta, dim_t ldc,
        int const *co, int flipB_flag) {

    int m_cap, n_cap, k_cap;
    m_cap = (m + 3) & (~3);
    n_cap = (n + 3) & (~3);
    k_cap = (k + 3) & (~3);

    if ((*ao != 0) || (*bo != 0)) {
        short *Ashort, *AP, *APraw;
        short *Bshort, *BP, *BPraw;
        int a_size = lda * (ATflag ? m - 1 : k - 1) + (ATflag ? k : m);
        int b_size = ldb * (BTflag ? k - 1 : n - 1) + (BTflag ? n : k);
        Ashort = (short *)malloc(a_size * sizeof(short), 4096);
        Bshort = (short *)malloc(b_size * sizeof(short), 4096);
        if (utils::any_null(Ashort, Bshort)) {
            free(Ashort);
            free(Bshort);
            return dnnl_out_of_memory;
        }
        for (int i = 0; i < a_size; ++i)
            Ashort[i] = ((short)A[i]) - (short)*ao;
        if (flipB_flag) {
            const int8_t *Bflip = (const int8_t *)B;
            const int8_t *bo_flip = (const int8_t *)bo;
            for (int i = 0; i < b_size; ++i)
                Bshort[i] = ((short)(Bflip[i])) - (short)*bo_flip;
        } else {
            for (int i = 0; i < b_size; ++i)
                Bshort[i] = ((short)B[i]) - (short)*bo;
        }
        APraw = (short *)malloc((m_cap * k_cap + 15) * sizeof(short), 4096);
        BPraw = (short *)malloc((k_cap * n_cap + 15) * sizeof(short), 4096);
        if (utils::any_null(APraw, BPraw)) {
            free(Ashort);
            free(Bshort);
            free(APraw);
            free(BPraw);
            return dnnl_out_of_memory;
        }
        AP = (short *)((((unsigned long)APraw) + 15) & (~15));
        BP = (short *)((((unsigned long)BPraw) + 15) & (~15));
        if (ATflag)
            pack_N16_16bit(k, m, Ashort, lda, AP);
        else
            pack_T16_16bit(k, m, Ashort, lda, AP);
        if (BTflag)
            pack_T8_16bit(k, n, Bshort, ldb, BP);
        else
            pack_N8_16bit(k, n, Bshort, ldb, BP);
        gemm_kernel_16bit(m, n, k, (float)alpha, AP, BP, C, beta, ldc);
        free(Ashort);
        free(Bshort);
        free(APraw);
        free(BPraw);
    } else {
        int8_t *AP, *APraw;
        uint8_t *BP, *BPraw;
        APraw = (int8_t *)malloc((m_cap * k_cap + 3) * sizeof(uint8_t), 4096);
        BPraw = (uint8_t *)malloc((k_cap * n_cap + 3) * sizeof(int8_t), 4096);
        if (utils::any_null(APraw, BPraw)) {
            free(APraw);
            free(BPraw);
            return dnnl_out_of_memory;
        }
        AP = (int8_t *)((((unsigned long)APraw) + 3) & (~3));
        BP = (uint8_t *)((((unsigned long)BPraw) + 3) & (~3));
        if (ATflag)
            pack_N16_8bit(k, m, A, lda, AP);
        else
            pack_T16_8bit(k, m, A, lda, AP);
        if (flipB_flag) {
            int b_size = ldb * (BTflag ? k - 1 : n - 1) + (BTflag ? n : k);
            uint8_t *Bflip = (uint8_t *)malloc(b_size * sizeof(uint8_t), 4096);
            if (utils::any_null(Bflip)) {
                free(APraw);
                free(BPraw);
                free(Bflip);
                return dnnl_out_of_memory;
            }
            for (int i = 0; i < b_size; ++i)
                Bflip[i] = B[i] ^ 0x80;
            if (BTflag)
                pack_T8_8bit(k, n, Bflip, ldb, BP);
            else
                pack_N8_8bit(k, n, Bflip, ldb, BP);
            free(Bflip);
        } else {
            if (BTflag)
                pack_T8_8bit(k, n, B, ldb, BP);
            else
                pack_N8_8bit(k, n, B, ldb, BP);
        }
        gemm_kernel_8bit(m, n, k, (float)alpha, AP, BP, C, beta, ldc);
        if (flipB_flag) {
            int *comparray = (int *)malloc(m * sizeof(int), 4096);
            if (utils::any_null(comparray)) {
                free(APraw);
                free(BPraw);
                free(comparray);
                return dnnl_out_of_memory;
            }
            for (int i = 0; i < m; ++i)
                comparray[i] = 0;
            if (ATflag) {
                for (int i = 0; i < m; ++i) {
                    int ca = 0;
                    const int8_t *at = &A[lda * i];
                    for (int j = 0; j < k; ++j) {
                        ca += (int)*at++;
                    }
                    comparray[i] = ca;
                }
            } else {
                for (int j = 0; j < k; ++j) {
                    int *ca = comparray;
                    const int8_t *at = &A[lda * j];
                    for (int i = 0; i < m; ++i) {
                        *ca++ += (int)*at++;
                    }
                }
            }
            for (int i = 0; i < m; ++i) {
                comparray[i] = out_round<int32_t>(saturate<int32_t>(
                        ((double)comparray[i]) * alpha * -128.0));
            }
            for (int j = 0; j < n; ++j) {
                int *ca = comparray;
                int *ct = &C[ldc * j];
                for (int i = 0; i < m; ++i) {
                    *ct++ += *ca++;
                }
            }
            free(comparray);
        }
        free(APraw);
        free(BPraw);
    }
    if (*offsetc == 'F' || *offsetc == 'f')
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                C[ldc * i + j] += co[0];
    if (*offsetc == 'R' || *offsetc == 'r')
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                C[ldc * i + j] += co[i];
    if (*offsetc == 'C' || *offsetc == 'c')
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                C[ldc * i + j] += co[j];

    return dnnl_success;
}

} // namespace ppc64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // __MMA__
