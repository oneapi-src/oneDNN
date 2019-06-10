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

#ifndef GEMM_PACK_HPP
#define GEMM_PACK_HPP

#include "mkldnn_config.h"
#include "mkldnn_types.h"

namespace mkldnn {
namespace impl {
namespace cpu {

mkldnn_status_t MKLDNN_API sgemm_pack_get_size(const char *identifier,
        const char *transa, const char *transb,  const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, size_t *size,
        bool *pack = nullptr);

mkldnn_status_t MKLDNN_API gemm_s8u8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, size_t *size,
        bool *pack = nullptr);

mkldnn_status_t MKLDNN_API sgemm_pack(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, const float *src,
        float *dst);

mkldnn_status_t MKLDNN_API gemm_s8u8s32_pack(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, const void *src,
        void *dst);

mkldnn_status_t MKLDNN_API sgemm_compute(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *A,
        const int *lda, const float *B, const int *ldb, const float *beta,
        float *C, const int *ldc);

mkldnn_status_t MKLDNN_API gemm_s8u8s32_compute(const char *transa,
        const char *transb, const char *offsetc, const int *M, const int *N,
        const int *K, const int8_t *A, const int *lda, const uint8_t *B,
        const int *ldb, const float *beta, int32_t *C, const int *ldc,
        const int32_t *co);

}
}
}

#endif // GEMM_PACK_HPP
