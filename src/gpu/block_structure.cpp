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

static bool can_combine(
        const block_t &a, const block_t &b, bool same_dim_only = true) {
    bool dim_ok = !same_dim_only || (a.dim_idx == b.dim_idx);
    bool a_then_b = (a.stride * a.block == b.stride);
    return dim_ok && a_then_b;
}

block_layout_t block_layout_t::normalized(bool remove_size_1_blocks) const {
    if (num_blocks == 0) return block_layout_t();
    block_layout_t res;

    res.append(blocks.front());
    auto cur = res.begin();
    for (size_t i = 1; i < num_blocks; i++) {
        const auto &block = blocks[i];
        if (block.block == 1 && remove_size_1_blocks) continue;
        if (can_combine(*cur, block)) {
            cur->stride = std::min(cur->stride, block.stride);
            cur->block = cur->block * block.block;
        } else {
            res.append(block);
            cur++;
        }
    }

    if (res.front().block == 1 && remove_size_1_blocks) res.erase(0);

    return res;
}

std::vector<block_t> normalize_blocks(
        const std::vector<block_t> &blocks, bool remove_size_1_blocks) {
    if (blocks.empty()) return {};
    std::vector<block_t> res;

    res.emplace_back(blocks.front());
    auto *cur = &res.back();
    for (size_t i = 1; i < blocks.size(); i++) {
        const auto &block = blocks[i];
        if (block.block == 1 && remove_size_1_blocks) continue;
        if (can_combine(*cur, block)) {
            cur->stride = std::min(cur->stride, block.stride);
            cur->block = cur->block * block.block;
        } else {
            res.emplace_back(block);
            cur = &res.back();
        }
    }

    if (res.front().block == 1 && remove_size_1_blocks) res.erase(res.begin());

    return res;
}

block_layout_t::block_layout_t(
        const memory_desc_wrapper &mdw, bool inner_only, bool do_normalize) {
    if (mdw.format_kind() == format_kind::undef) return;

    const size_t ndims = static_cast<size_t>(mdw.ndims());
    auto &blocking = mdw.blocking_desc();
    auto *padded_dims = mdw.padded_dims();

    dim_t stride = 1;
    std::vector<dim_t> full_blocks(ndims, 1);
    for (int i = blocking.inner_nblks - 1; i >= 0; i--) {
        dim_t dim_idx = blocking.inner_idxs[i];
        dim_t block = blocking.inner_blks[i];
        append(block_t(dim_idx, block, stride));
        stride *= block;
        full_blocks[static_cast<size_t>(dim_idx)] *= block;
    }

    if (!inner_only) {
        for (size_t i = 0; i < ndims; i++) {
            dim_t block = padded_dims[i] / full_blocks[i];
            append(block_t(static_cast<dim_t>(i), block, blocking.strides[i]));
        }

        // Sort outer blocks by their stride.
        std::sort(begin() + blocking.inner_nblks, end(),
                [](const block_t &a, const block_t &b) {
                    return a.stride < b.stride
                            || (a.stride == b.stride && a.block < b.block);
                });
    }

    if (do_normalize) *this = normalized();
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
