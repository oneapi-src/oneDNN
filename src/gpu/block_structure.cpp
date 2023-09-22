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

#include "gpu/block_structure.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

std::vector<block_t> normalize_blocks(
        const std::vector<block_t> &blocks, bool remove_size_1_blocks) {
    auto new_blocks = blocks;

    // Remove blocks of size 1.
    if (remove_size_1_blocks) {
        for (auto it = new_blocks.begin(); it != new_blocks.end();) {
            if (it->block == 1) {
                it = new_blocks.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Merge same dimension blocks.
    block_t prev_b;
    prev_b.dim_idx = undefined_dim_idx;
    for (auto it = new_blocks.begin(); it != new_blocks.end();) {
        if (it->dim_idx == prev_b.dim_idx
                && it->stride == (prev_b.stride * prev_b.block)) {
            auto &b = *(it - 1);
            b.block *= it->block;
            prev_b = b;
            it = new_blocks.erase(it);
        } else {
            prev_b = *it;
            ++it;
        }
    }

    return new_blocks;
}

std::vector<block_t> compute_block_structure(
        const memory_desc_wrapper &mdw, bool inner_only, bool do_normalize) {
    if (mdw.format_kind() == format_kind::undef) return {};

    const int ndims = mdw.ndims();
    auto &blocking = mdw.blocking_desc();
    auto *padded_dims = mdw.padded_dims();

    dim_t stride = 1;
    std::vector<dim_t> full_blocks(ndims, 1);
    std::vector<block_t> out;
    for (int i = blocking.inner_nblks - 1; i >= 0; i--) {
        int dim_idx = blocking.inner_idxs[i];
        dim_t block = blocking.inner_blks[i];
        out.emplace_back(dim_idx, block, stride);
        stride *= block;
        full_blocks[dim_idx] *= block;
    }

    if (!inner_only) {
        for (int i = 0; i < ndims; i++) {
            dim_t block = padded_dims[i] / full_blocks[i];
            out.emplace_back(i, block, blocking.strides[i]);
        }

        // Sort outer blocks by their stride.
        std::sort(out.begin() + blocking.inner_nblks, out.end(),
                [](const block_t &a, const block_t &b) {
                    return a.stride < b.stride
                            || (a.stride == b.stride && a.block < b.block);
                });
    }

    if (do_normalize) out = normalize_blocks(out);
    return out;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
