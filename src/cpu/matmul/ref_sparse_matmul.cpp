/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_io_helper.hpp"

#include "cpu/matmul/ref_sparse_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

status_t ref_sparse_matmul_t::execute(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    const dim_t M = dst_d.dims()[0];
    const dim_t N = dst_d.dims()[1];
    const dim_t K = src_d.dims()[1];

    const data_type_t mm_dt = src_d.data_type();
    auto scratchpad = ctx.get_scratchpad_grantor();

    parallel_nd(M, N, [&](dim_t i, dim_t j) {
        const dim_t dst_idx = i * N + j;
        io::store_float_value(dst_d.data_type(), 0.0f, dst, dst_idx);
    });

    if (weights_d.is_sparse_desc()) {

        const auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        const auto wei_values = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS, 0);
        auto wei_buffer_1 = CTX_IN_MEM(const int32_t *, DNNL_ARG_WEIGHTS, 1);
        auto wei_buffer_2 = CTX_IN_MEM(const int32_t *, DNNL_ARG_WEIGHTS, 2);

        // Both COO and CSR encoded data is operated on using CSR kernel for
        // matrix multiplication.
        // For COO encoding, data preparation includes using a temporary
        // buffer to convert the data to the CSR format.
        // Matrix multiplication is then carried out using the CSR encoded data.
        const int32_t *wei_indices = nullptr;
        const int32_t *wei_pointers = nullptr;

        if (weights_d.encoding() == sparse_encoding::csr) {
            // For CSR encodings, pointer and indices assignment is
            // staightforward as,
            // index 1 - index buffer, index 2 - pointer buffer.
            wei_indices = wei_buffer_1;
            wei_pointers = wei_buffer_2;
        } else if (weights_d.encoding() == sparse_encoding::coo) {
            // For COO encodings, the two index buffers hold the row and column
            // indices respectively. For CSR conversion, the row indices are
            // compressed to generate the CSR pointers.
            wei_indices = wei_buffer_2;

            int32_t *wei_row_pointers = scratchpad.template get<int32_t>(
                    memory_tracking::names::key_matmul_sparse_tmp_ptr);

            parallel_nd(K + 1, [&](dim_t k) {
                io::store_float_value(
                        weights_d.metadata_type(0), 0, wei_row_pointers, k);
            });

            cvt_coo_indices_to_csr_pointers(
                    wei_buffer_1, wei_row_pointers, weights_d.nnz(), K);

            wei_pointers = wei_row_pointers;
        }

        run_csr_kernel(src, wei_values, wei_indices, wei_pointers, dst, M, N, K,
                mm_dt, src_d.is_sparse_desc());

    } else if (src_d.is_sparse_desc()) {
        const auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
        const auto src_values = CTX_IN_MEM(const void *, DNNL_ARG_SRC, 0);
        auto src_buffer_1 = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 1);
        auto src_buffer_2 = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 2);

        // Both COO and CSR encoded data is operated on using CSR kernel for
        // matrix multiplication.
        // For COO encoding, data preparation includes using a temporary
        // buffer to convert the data to the CSR format.
        // Matrix multiplication is then carried out using the CSR encoded data.
        const int32_t *src_indices = nullptr;
        const int32_t *src_pointers = nullptr;

        if (src_d.encoding() == sparse_encoding::csr) {
            // For CSR encodings, pointer and indices assignment is
            // staightforward as
            // index 1 - index buffer, index 2 - pointer buffer.
            src_indices = src_buffer_1;
            src_pointers = src_buffer_2;
        } else if (src_d.encoding() == sparse_encoding::coo) {
            // For COO encodings, the two index buffers hold the row and column
            // indices respectively. For CSR conversion, the row indices are
            // compressed to generate the CSR pointers.
            src_indices = src_buffer_2;

            int32_t *src_row_pointers = scratchpad.template get<int32_t>(
                    memory_tracking::names::key_matmul_sparse_tmp_ptr);

            parallel_nd(M + 1, [&](dim_t m) {
                io::store_float_value(
                        src_d.metadata_type(0), 0, src_row_pointers, m);
            });

            cvt_coo_indices_to_csr_pointers(
                    src_buffer_1, src_row_pointers, src_d.nnz(), M);
            src_pointers = src_row_pointers;
        }

        run_csr_kernel(weights, src_values, src_indices, src_pointers, dst, M,
                N, K, mm_dt, src_d.is_sparse_desc());
    }
    return status::success;
}

void ref_sparse_matmul_t::cvt_coo_indices_to_csr_pointers(
        const int32_t *indices, int32_t *pointers, const int nnz,
        const int nrows) const {
    parallel_nd(
            nnz, [&](dim_t i) { fetch_and_add(&pointers[indices[i] + 1], 1); });
    for (int i = 0; i < nrows; ++i) {
        pointers[i + 1] += pointers[i];
    }
}

void ref_sparse_matmul_t::run_csr_kernel(const void *dmat, const void *values,
        const int32_t *indices, const int32_t *pointers, void *res,
        const dim_t M, const dim_t N, const dim_t K, const data_type_t mm_dt,
        bool is_src_sparse) const {

    if (is_src_sparse) {
        // With a sparse source tensor, the matrix multiplication is carried out
        // for a sparse multiplier with parallelization over the sparse rows
        // of the multiplier matrix.
        parallel_nd(M, [&](dim_t m) {
            const dim_t row_start = pointers[m];
            const dim_t row_end = pointers[m + 1];

            for (dim_t n = 0; n < N; n++) {
                const dim_t c_idx = m * N + n;
                float c_val = io::load_float_value(mm_dt, res, c_idx);

                for (dim_t k = row_start; k < row_end; k++) {
                    const dim_t b_idx = indices[k] * N + n;
                    const float a_val = io::load_float_value(mm_dt, values, k);
                    const float b_val
                            = io::load_float_value(mm_dt, dmat, b_idx);
                    c_val += a_val * b_val;
                }
                io::store_float_value(mm_dt, c_val, res, c_idx);
            }
        });
    } else {
        // With a sparse weights tensor, the matrix multiplication is carried
        // out for a sparse multiplicand with parallelization over the dense
        // rows of the multiplier matrix.
        parallel_nd(M, [&](dim_t m) {
            for (dim_t k = 0; k < K; k++) {
                const dim_t row_start = pointers[k];
                const dim_t row_end = pointers[k + 1];
                for (dim_t n = row_start; n < row_end; n++) {
                    const dim_t a_idx = m * K + k;
                    const dim_t c_idx = m * N + indices[n];
                    const float a_val
                            = io::load_float_value(mm_dt, dmat, a_idx);
                    const float b_val = io::load_float_value(mm_dt, values, n);
                    float c_val = io::load_float_value(mm_dt, res, c_idx);
                    c_val += a_val * b_val;
                    io::store_float_value(mm_dt, c_val, res, c_idx);
                }
            }
        });
    }
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
