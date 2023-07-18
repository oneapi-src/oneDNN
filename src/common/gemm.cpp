/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#endif

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "cpu/gemm/gemm.hpp"
#endif

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/profiler.hpp"
#include "common/stack_checker.hpp"
#include "common/verbose.hpp"

using namespace dnnl::impl;

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
namespace {
const char *c2f_offsetC(const char *offC) {
    if (offC) {
        if (offC[0] == 'R' || offC[0] == 'r') return "C";
        if (offC[0] == 'C' || offC[0] == 'c') return "R";
    }
    return offC;
}

std::string get_descriptor(dim_t M, dim_t N, dim_t K) {
    std::string s_ = std::to_string(M);
    s_ += "x";
    s_ += std::to_string(K);
    s_ += ":";
    s_ += std::to_string(K);
    s_ += "x";
    s_ += std::to_string(N);
    return s_;
}

} // namespace
#endif

#ifdef DNNL_ENABLE_STACK_CHECKER
#define MAYBE_RUN_STACK_CHECKER(api_name, ...) \
    stack_checker::stack_checker_t(#api_name).check(__VA_ARGS__)
#else
#define MAYBE_RUN_STACK_CHECKER(_, func, ...) func(__VA_ARGS__)
#endif

#define MAYBE_VERBOSE(status, sdt_, wdt_, ddt_, ...) \
    if (get_verbose(verbose_t::exec_profile, component_t::gemm_api)) { \
        double start_ms = get_msec(); \
        status = __VA_ARGS__; \
        double duration_ms = get_msec() - start_ms; \
        std::stringstream ss; \
        ss << "cpu,gemm_api,,undef,"; \
        const bool is_src_ab = (transa == 'N' || transa == 'n'); \
        ss << "src_" << sdt_ << "::blocked:" << (is_src_ab ? "ab" : "ba") \
           << ":f0 "; \
        const bool is_wei_ab = (transb == 'N' || transb == 'n'); \
        ss << "wei_" << wdt_ << "::blocked:" << (is_wei_ab ? "ab" : "ba") \
           << ":f0 "; \
        ss << "dst_" << ddt_ << "::blocked:ab:f0,"; \
        if (is_src_ab && lda != K) ss << "lda:" << lda << " "; \
        if (!is_src_ab && lda != M) ss << "lda:" << lda << " "; \
        if (is_wei_ab && ldb != N) ss << "ldb:" << ldb << " "; \
        if (!is_wei_ab && ldb != K) ss << "ldb:" << ldb << " "; \
        if (alpha != 1.f) ss << "attr-oscale:common:" << alpha << " "; \
        if (beta != 0.f) ss << "attr-post-ops:sum:" << beta << " "; \
        ss << ",," << get_descriptor(M, N, K); \
        VPROF(start_ms, primitive, exec, VERBOSE_profile, ss.str().c_str(), \
                duration_ms); \
    } else { \
        status = __VA_ARGS__; \
    }

dnnl_status_t dnnl_sgemm(char transa, char transb, dim_t M, dim_t N, dim_t K,
        float alpha, const float *A, dim_t lda, const float *B, const dim_t ldb,
        float beta, float *C, dim_t ldc) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    status_t status = dnnl_success;
    MAYBE_VERBOSE(status, "f32", "f32", "f32",
            MAYBE_RUN_STACK_CHECKER(dnnl_sgemm, cpu::extended_sgemm, &transb,
                    &transa, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C,
                    &ldc, nullptr, false));
    return status;
#else
    return dnnl::impl::status::unimplemented;
#endif
}

dnnl_status_t dnnl_gemm_u8s8s32(char transa, char transb, char offsetc, dim_t M,
        dim_t N, dim_t K, float alpha, const uint8_t *A, dim_t lda, uint8_t ao,
        const int8_t *B, dim_t ldb, int8_t bo, float beta, int32_t *C,
        dim_t ldc, const int32_t *co) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    status_t status = dnnl_success;
    MAYBE_VERBOSE(status, "u8", "s8", "s32",
            MAYBE_RUN_STACK_CHECKER(dnnl_gemm_u8s8s32,
                    cpu::gemm_s8x8s32<uint8_t>, &transb, &transa,
                    c2f_offsetC(&offsetc), &N, &M, &K, &alpha, B, &ldb, &bo, A,
                    &lda, &ao, &beta, C, &ldc, co));
    return status;
#else
    return dnnl::impl::status::unimplemented;
#endif
}

dnnl_status_t dnnl_gemm_s8s8s32(char transa, char transb, char offsetc, dim_t M,
        dim_t N, dim_t K, float alpha, const int8_t *A, dim_t lda, int8_t ao,
        const int8_t *B, dim_t ldb, int8_t bo, float beta, int32_t *C,
        dim_t ldc, const int32_t *co) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    status_t status = dnnl_success;
    MAYBE_VERBOSE(status, "s8", "s8", "s32",
            MAYBE_RUN_STACK_CHECKER(dnnl_gemm_s8s8s32,
                    cpu::gemm_s8x8s32<int8_t>, &transb, &transa,
                    c2f_offsetC(&offsetc), &N, &M, &K, &alpha, B, &ldb, &bo, A,
                    &lda, &ao, &beta, C, &ldc, co));
    return status;
#else
    return dnnl::impl::status::unimplemented;
#endif
}

extern "C" dnnl_status_t DNNL_API dnnl_gemm_bf16bf16f32(char transa,
        char transb, dim_t M, dim_t N, dim_t K, float alpha,
        const bfloat16_t *A, dim_t lda, const bfloat16_t *B, dim_t ldb,
        float beta, float *C, dim_t ldc) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    status_t status = dnnl_success;
    MAYBE_VERBOSE(status, "bf16", "bf16", "f32",
            MAYBE_RUN_STACK_CHECKER(dnnl_gemm_bf16bf16f32,
                    cpu::gemm_bf16bf16f32, &transb, &transa, &N, &M, &K, &alpha,
                    B, &ldb, A, &lda, &beta, C, &ldc));
    return status;
#else
    return dnnl::impl::status::unimplemented;
#endif
}

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
dnnl_status_t dnnl_threadpool_interop_sgemm(char transa, char transb, dim_t M,
        dim_t N, dim_t K, float alpha, const float *A, dim_t lda,
        const float *B, const dim_t ldb, float beta, float *C, dim_t ldc,
        void *th) {
    threadpool_utils::activate_threadpool(
            (dnnl::threadpool_interop::threadpool_iface *)th);
    status_t status = dnnl_success;
    MAYBE_VERBOSE(status, "f32", "f32", "f32",
            MAYBE_RUN_STACK_CHECKER(dnnl_threadpool_interop_sgemm,
                    cpu::extended_sgemm, &transb, &transa, &N, &M, &K, &alpha,
                    B, &ldb, A, &lda, &beta, C, &ldc, nullptr, false));
    threadpool_utils::deactivate_threadpool();
    return status;
}

dnnl_status_t dnnl_threadpool_interop_gemm_u8s8s32(char transa, char transb,
        char offsetc, dim_t M, dim_t N, dim_t K, float alpha, const uint8_t *A,
        dim_t lda, uint8_t ao, const int8_t *B, dim_t ldb, int8_t bo,
        float beta, int32_t *C, dim_t ldc, const int32_t *co, void *th) {
    threadpool_utils::activate_threadpool(
            (dnnl::threadpool_interop::threadpool_iface *)th);
    status_t status = dnnl_success;
    MAYBE_VERBOSE(status, "u8", "s8", "s32",
            MAYBE_RUN_STACK_CHECKER(dnnl_threadpool_interop_gemm_u8s8s32,
                    cpu::gemm_s8x8s32<uint8_t>, &transb, &transa,
                    c2f_offsetC(&offsetc), &N, &M, &K, &alpha, B, &ldb, &bo, A,
                    &lda, &ao, &beta, C, &ldc, co));
    threadpool_utils::deactivate_threadpool();
    return status;
}

dnnl_status_t dnnl_threadpool_interop_gemm_s8s8s32(char transa, char transb,
        char offsetc, dim_t M, dim_t N, dim_t K, float alpha, const int8_t *A,
        dim_t lda, int8_t ao, const int8_t *B, dim_t ldb, int8_t bo, float beta,
        int32_t *C, dim_t ldc, const int32_t *co, void *th) {
    threadpool_utils::activate_threadpool(
            (dnnl::threadpool_interop::threadpool_iface *)th);
    status_t status = dnnl_success;
    MAYBE_VERBOSE(status, "s8", "s8", "s32",
            MAYBE_RUN_STACK_CHECKER(dnnl_threadpool_interop_gemm_s8s8s32,
                    cpu::gemm_s8x8s32<int8_t>, &transb, &transa,
                    c2f_offsetC(&offsetc), &N, &M, &K, &alpha, B, &ldb, &bo, A,
                    &lda, &ao, &beta, C, &ldc, co));
    threadpool_utils::deactivate_threadpool();
    return status;
}

extern "C" dnnl_status_t DNNL_API dnnl_threadpool_interop_gemm_bf16bf16f32(
        char transa, char transb, dim_t M, dim_t N, dim_t K, float alpha,
        const bfloat16_t *A, dim_t lda, const bfloat16_t *B, dim_t ldb,
        float beta, float *C, dim_t ldc, void *th) {
    threadpool_utils::activate_threadpool(
            (dnnl::threadpool_interop::threadpool_iface *)th);
    status_t status = dnnl_success;
    MAYBE_VERBOSE(status, "bf16", "bf16", "f32",
            MAYBE_RUN_STACK_CHECKER(dnnl_threadpool_interop_gemm_bf16bf16f32,
                    cpu::gemm_bf16bf16f32, &transb, &transa, &N, &M, &K, &alpha,
                    B, &ldb, A, &lda, &beta, C, &ldc));
    threadpool_utils::deactivate_threadpool();
    return status;
}

#undef MAYBE_VERBOSE

#endif
