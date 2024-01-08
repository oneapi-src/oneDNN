/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MICROKERNEL_CPU_MICROKERNEL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MICROKERNEL_CPU_MICROKERNEL_HPP

#include <stdint.h>
#include <util/def.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
struct stream_t;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

extern "C" {
SC_API int dnnl_brgemm_init_update(const void *A, const void *B, void *C,
        int num, int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b, int dtypeA, int dtypeB, const void *brg_attrs,
        char *bd_mask, const void *postops_setting, const void *top_pad,
        const void *bottom_pad, const void *postops_data, void *c_buf,
        dnnl::impl::graph::gc::runtime::stream_t *stream);
SC_API int dnnl_brgemm_init(
        void *C, int M, int N, int LDC, int dtypeC, float value = 0.f);
SC_API int dnnl_brgemm_update(const void *A, const void *B, void *C, int num,
        int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b, int dtypeA, int dtypeB, const void *brg_attrs,
        char *bd_mask, const void *postops_setting, const void *top_pad,
        const void *bottom_pad, const void *postops_data, void *c_buf,
        dnnl::impl::graph::gc::runtime::stream_t *stream);
SC_API int dnnl_brgemm_list_update(const void **A_list, const void **B_list,
        void *C, int num, int M, int N, int K, int LDA, int LDB, int LDC,
        int stride_a, int stride_b, int len, int dtypeA, int dtypeB,
        const void *brg_attrs, char *bd_mask, const void *postops_setting,
        const void *top_pad, const void *bottom_pad, const void *postops_data,
        void *c_buf, dnnl::impl::graph::gc::runtime::stream_t *stream);
SC_API int dnnl_brgemm_init_list_update(const void **A_list,
        const void **B_list, void *C, int num, int M, int N, int K, int LDA,
        int LDB, int LDC, int stride_a, int stride_b, int len, int dtypeA,
        int dtypeB, const void *brg_attrs, char *bd_mask,
        const void *postops_setting, const void *top_pad,
        const void *bottom_pad, const void *postops_data, void *c_buf,
        dnnl::impl::graph::gc::runtime::stream_t *stream);
SC_API void *dnnl_brgemm_list_func(int M, int N, int K, int LDA, int LDB,
        int LDC, float beta, int dtypeA, int dtypeB, const void *brg_attrs,
        char *bd_mask, const void *postops_setting);

struct brgemm_kernel_info;
struct brg_range_handle_t;
SC_API void dnnl_brgemm_list_call(brgemm_kernel_info *brg_desc,
        const void **A_list, const void **B_list, void *C, int len, int num,
        int stride_a, int stride_b, int dtypeA, int dtypeB, const void *top_pad,
        const void *bottom_pad,
        dnnl::impl::graph::gc::runtime::stream_t *stream);
SC_API void dnnl_brgemm_list_call_range(brg_range_handle_t *brg_range_desc,
        int M_real, int N_real, int K_real, const void **A_list,
        const void **B_list, void *C, int num, int stride_a, int stride_b,
        int len, int dtypeA, int dtypeB, const void *top_pad,
        const void *bottom_pad,
        dnnl::impl::graph::gc::runtime::stream_t *stream);
SC_API void dnnl_brgemm_list_call_postops(brgemm_kernel_info *brg_desc,
        const void **A_list, const void **B_list, void *C, int len, int num,
        int stride_a, int stride_b, int dtypeA, int dtypeB, const void *top_pad,
        const void *bottom_pad, const void *postops_data, void *c_buf,
        dnnl::impl::graph::gc::runtime::stream_t *stream);
SC_API void *dnnl_brgemm_func(int M, int N, int K, int LDA, int LDB, int LDC,
        int stride_a, int stride_b, float beta, int dtypeA, int dtypeB,
        const void *brg_attrs, char *bd_mask, const void *postops_setting);
SC_API void dnnl_brgemm_call(brgemm_kernel_info *brg_desc, const void *A,
        const void *B, void *C, int num, const void *top_pad,
        const void *bottom_pad,
        dnnl::impl::graph::gc::runtime::stream_t *stream);
SC_API void dnnl_brgemm_call_range(brg_range_handle_t *brg_range_desc,
        int M_real, int N_real, int K_real, const void *A, const void *B,
        void *C, int num, const void *top_pad, const void *bottom_pad,
        dnnl::impl::graph::gc::runtime::stream_t *stream);
SC_API void dnnl_brgemm_call_postops(brgemm_kernel_info *brg_desc,
        const void *A, const void *B, void *C, int num, const void *top_pad,
        const void *bottom_pad, const void *postops_data, void *c_buf,
        dnnl::impl::graph::gc::runtime::stream_t *stream);
SC_API void dnnl_brgemm_postops_data_init(void *dnnl_data = nullptr,
        void *bias = nullptr, void *scales = nullptr,
        void *binary_post_ops_rhs = nullptr, uint64_t oc_logical_off = 0UL,
        uint64_t dst_row_logical_off = 0, void *data_C_ptr_ = nullptr,
        uint64_t first_mb_matrix_addr_off = 0,
        void *a_zp_compensations = nullptr, void *b_zp_compensations = nullptr,
        void *c_zp_values = nullptr, bool skip_accumulation = false,
        int zp_a_val = 1, bool do_only_comp = false,
        bool do_only_zp_a_val = false);
}
#endif
