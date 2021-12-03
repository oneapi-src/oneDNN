/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_MICROKERNEL_CPU_MICROKERNEL_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_MICROKERNEL_CPU_MICROKERNEL_HPP

#include <stdint.h>
#include <util/def.hpp>
namespace sc {
namespace runtime {
struct stream_t;
}
} // namespace sc

extern "C" {
SC_API int dnnl_brgemm_init_update(const void *A, const void *B, void *C,
        int num, int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b, int dtypeA, int dtypeB, sc::runtime::stream_t *stream);
SC_API int dnnl_brgemm_init(
        void *C, int M, int N, int LDC, int dtypeC, float value = 0.f);
SC_API int dnnl_brgemm_update(const void *A, const void *B, void *C, int num,
        int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b, int dtypeA, int dtypeB, sc::runtime::stream_t *stream);
SC_API int dnnl_brgemm_list_update(const void **A_list, const void **B_list,
        void *C, int num, int M, int N, int K, int LDA, int LDB, int LDC,
        int stride_a, int stride_b, int len, int dtypeA, int dtypeB,
        sc::runtime::stream_t *stream);
SC_API void *dnnl_brgemm_list_func(int M, int N, int K, int LDA, int LDB,
        int LDC, float beta, int dtypeA, int dtypeB);

struct brgemm_kernel_info;
SC_API void dnnl_brgemm_list_call(brgemm_kernel_info *brg_desc,
        const void **A_list, const void **B_list, void *C, int len, int num,
        int stride_a, int stride_b, int dtypeA, int dtypeB,
        sc::runtime::stream_t *stream);
SC_API void *dnnl_brgemm_func(int M, int N, int K, int LDA, int LDB, int LDC,
        int stride_a, int stride_b, float beta, int dtypeA, int dtypeB);
SC_API void dnnl_brgemm_call(brgemm_kernel_info *brg_desc, const void *A,
        const void *B, void *C, int num, sc::runtime::stream_t *stream);
}

#endif
