/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gpu/jit/ir/reduce.hpp"

#include <vector>

#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

stmt_t create_reduce_stmt(const layout_t &src, const layout_t &dst,
        const expr_t &src_buf, const expr_t &dst_buf, const tensor_t &_subtile,
        uint32_t reduction_mask, bool drop_dims) {
    auto subtile = _subtile;
    if (subtile.is_empty()) subtile = tensor_t(src.dims());
    ir_assert(src.ndims() == subtile.ndims());
    int ndims = src.ndims();

    // Align dst layout with src layout according to the mask if needed.
    layout_t dst_aligned;
    if (drop_dims) {
        std::vector<int> dst2src(dst.ndims());
        int dst_dim_idx = 0;
        for (int i = 0; i < ndims; i++) {
            if ((reduction_mask & (1 << i)) != 0) {
                dst2src[dst_dim_idx] = i;
                dst_dim_idx++;
            }
        }
        ir_assert(dst_dim_idx == dst.ndims()) << "Incompatible reduction mask.";

        auto dst_blocks = dst.blocks();
        for (auto &b : dst_blocks)
            b.dim_idx = dst2src[b.dim_idx];

        // Create final layout.
        dst_aligned = layout_t(dst.type(), ndims, dst.offset(), dst_blocks);
    } else {
        dst_aligned = dst;
    }

    std::vector<dim_t> dst_tile_dims = subtile.dims();
    std::vector<expr_t> dst_tile_start = subtile.start();
    for (int i = 0; i < ndims; i++) {
        if ((reduction_mask & (1 << i)) == 0) {
            dst_tile_dims[i] = 1;
            dst_tile_start[i] = expr_t(0);
            continue;
        }
    }
    dst_aligned = dst_aligned.map(tensor_t(dst_tile_dims, dst_tile_start));

    auto func = reduce_t::make(src, dst_aligned);
    return func.call({dst_buf, src_buf});
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
