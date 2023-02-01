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

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/matmul/ref_sparse_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

status_t ref_sparse_matmul_t::execute(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto dst = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DST, status);
    CHECK(status);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    const dim_t M = dst_d.dims()[0];
    const dim_t N = dst_d.dims()[1];
    const dim_t K = src_d.dims()[1];

    parallel_nd(M, N, [&](dim_t i, dim_t j) { dst[i * N + j] = 0.0f; });

    if (weights_d.is_sparse_desc()) {
        const auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
        const auto wei_values = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS, 0);
        const auto wei_indices
                = CTX_IN_MEM(const int32_t *, DNNL_ARG_WEIGHTS, 1);
        const auto wei_pointers
                = CTX_IN_MEM(const int32_t *, DNNL_ARG_WEIGHTS, 2);

        parallel_nd(M, [&](dim_t m) {
            for (dim_t k = 0; k < K; k++) {
                const dim_t row_start = wei_pointers[k];
                const dim_t row_end = wei_pointers[k + 1];
                for (dim_t n = row_start; n < row_end; n++) {
                    const dim_t src_idx = m * K + k;
                    const dim_t dst_idx = m * N + wei_indices[n];
                    dst[dst_idx] = dst[dst_idx] + src[src_idx] * wei_values[n];
                }
            }
        });
    } else if (src_d.is_sparse_desc()) {
        const auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
        const auto src_values = CTX_IN_MEM(const float *, DNNL_ARG_SRC, 0);
        const auto src_indices = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 1);
        const auto src_pointers = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 2);

        parallel_nd(M, [&](dim_t m) {
            const dim_t row_start = src_pointers[m];
            const dim_t row_end = src_pointers[m + 1];
            for (dim_t k = row_start; k < row_end; k++) {
                for (dim_t n = 0; n < N; n++) {
                    const dim_t dst_idx = m * N + n;
                    const dim_t wei_idx = src_indices[k] * N + n;
                    dst[dst_idx]
                            = dst[dst_idx] + src_values[k] * weights[wei_idx];
                }
            }
        });
    }

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
