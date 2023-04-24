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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MICROKERNEL_CPU_BRGEMM_RANGE_HANDLE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MICROKERNEL_CPU_BRGEMM_RANGE_HANDLE_HPP
#include <memory>
#include <vector>
#include <cpu/x64/brgemm/brgemm_types.hpp>
struct brgemm_kernel_info;
using namespace dnnl::impl::cpu::x64;

namespace brg_range_tail_value {
constexpr const int dyn_tail = -1; // dynamic tail range, from 1 to upper bound
constexpr const int no_tail = 0; // not a range, fixed to upper bound
// if the value is >0, it means the dimension has tail in static case and the
// value is the static tail.
} // namespace brg_range_tail_value
// For range shape brgemm process, use a linear cache for indexing when the
// number of total ranges is below 256. Currently we only consider contiguous
// range.
struct brg_range_handle_t {
    struct extra_arg_t {
        float beta;
        int LDA;
        int LDB;
        int LDC;
        int stride_a;
        int stride_b;
        int dtypeA;
        int dtypeB;
        const void *brg_attrs;
        extra_arg_t(float beta, int LDA, int LDB, int LDC, int stride_a,
                int stride_b, int dtypeA, int dtypeB, const void *brg_attrs)
            : beta(beta)
            , LDA(LDA)
            , LDB(LDB)
            , LDC(LDC)
            , stride_a(stride_a)
            , stride_b(stride_b)
            , dtypeA(dtypeA)
            , dtypeB(dtypeB)
            , brg_attrs(brg_attrs) {}
    };
    // record the upper bound of M, N, K.
    int M_upper_bound;
    int N_upper_bound;
    int K_upper_bound;
    int M_tail_value;
    int N_tail_value;
    int K_tail_value;
    // use linear when total range number is below 256.
    std::vector<brgemm_kernel_info *> linear_cache;
    // record extra args for runtime generation.
    std::shared_ptr<extra_arg_t> extra_args;
    static constexpr const int linear_cache_capacity = 256;
    virtual ~brg_range_handle_t() {}
    void init_func(brgemm_batch_kind_t brg_type, int M, int N, int K, int LDA,
            int LDB, int LDC, int stride_a, int stride_b, float beta,
            int dtypeA, int dtypeB, const void *brg_attrs, int M_tail_value,
            int N_tail_value, int K_tail_value);
    brg_range_handle_t(int M, int N, int K, int LDA, int LDB, int LDC,
            float beta, int dtypeA, int dtypeB, const void *brg_attrs,
            int M_tail_value, int N_tail_value, int K_tail_value);
    brg_range_handle_t(int M, int N, int K, int LDA, int LDB, int LDC,
            int stride_a, int stride_b, float beta, int dtypeA, int dtypeB,
            const void *brg_attrs, int M_tail_value, int N_tail_value,
            int K_tail_value);
    brgemm_kernel_info *get_linear_kernel(
            int M_real, int N_real, int K_real) const;
    void brg_list_call(int M_real, int N_real, int K_real, const void **A_list,
            const void **B_list, void *C, int num, int stride_a, int stride_b,
            int len, int dtypeA, int dtypeB,
            dnnl::impl::graph::gc::runtime::stream_t *stream);
    void brg_strd_call(int M_real, int N_real, int K_real, const void *A,
            const void *B, void *C, int num,
            dnnl::impl::graph::gc::runtime::stream_t *stream);
};
#endif
