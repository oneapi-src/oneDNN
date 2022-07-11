/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "cpu/x64/rnn/brgemm_cell_common_reorders.hpp"
#include "common/dnnl_thread.hpp"
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

src_layer_iter_transpose_t::src_layer_iter_transpose_t(const int src_ld,
        const int dst_ld, const int rows, const int cols,
        jit_brgemm_trans_src_t *const kernel_transpose)
    : src_ld_(src_ld)
    , dst_ld_(dst_ld)
    , src_rows_(rows)
    , src_cols_(cols)
    , kernel_transpose_(kernel_transpose) {};

template <typename Dt>
void src_layer_iter_transpose_t::execute(const Dt *src, Dt *dst) const {
    static constexpr int block_size = 16;
    const auto rows_div = std::div(src_rows_, block_size);
    const auto rows_tail = rows_div.rem;
    const auto rows_blks = rows_div.quot + (rows_tail > 0 ? 1 : 0);
    const auto cols_div = std::div(src_cols_, block_size);
    const auto cols_tail = cols_div.rem;
    const auto cols_blks = cols_div.quot + (cols_tail > 0 ? 1 : 0);

    parallel_nd(cols_blks, rows_blks, [&](dim_t c, dim_t r) {
        const auto current_rows
                = (rows_tail && r == rows_blks - 1) ? rows_tail : block_size;
        const auto current_cols
                = (cols_tail && c == cols_blks - 1) ? cols_tail : block_size;

        auto ctx = jit_brgemm_trans_src_t::ctx_t();
        ctx.src = (void *)(src + (r * src_ld_ + c) * block_size);
        ctx.tr_src = (void *)(dst + (c * dst_ld_ + r) * block_size);
        ctx.current_gemm_batch = 1;
        ctx.current_M = current_cols;
        ctx.current_K = current_rows;

        (*kernel_transpose_)(&ctx);
    });
}

template void src_layer_iter_transpose_t::execute<float>(
        const float *, float *) const;
template void src_layer_iter_transpose_t::execute<bfloat16_t>(
        const bfloat16_t *, bfloat16_t *) const;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
