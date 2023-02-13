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
#ifndef CPU_S390X_GEMM_H
#define CPU_S390X_GEMM_H
namespace dnnl {
namespace impl {
namespace cpu {
namespace s390x {

dnnl_status_t gemmx8x8s32(const char *transa, const char *transb,
        const char *offsetc, dim_t M, dim_t N, dim_t K, float alpha,
        const int8_t *A, dim_t ldA, const int8_t *ao, const uint8_t *B,
        dim_t ldB, const uint8_t *bo, float beta, int32_t *C, dim_t ldC,
        const int32_t *co);

dnnl_status_t gemmx8x8s32(const char *transa, const char *transb,
        const char *offsetc, dim_t M, dim_t N, dim_t K, float alpha,
        const int8_t *A, dim_t ldA, const int8_t *ao, const int8_t *B,
        dim_t ldB, const int8_t *bo, float beta, int32_t *C, dim_t ldC,
        const int32_t *co);

} // namespace s390x
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
