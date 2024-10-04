/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include "gpu/intel/block_structure.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

block_layout_t block_layout_t::normalized(bool remove_size_1_blocks) const {
    if (num_blocks == 0) return block_layout_t();
    block_layout_t res;

    std::vector<block_t> block_vec(num_blocks);
    memcpy(&block_vec[0], &blocks[0], num_blocks * sizeof(block_t));

    std::vector<block_t> new_blocks
            = normalize_blocks(block_vec, remove_size_1_blocks);
    for (const block_t &block : new_blocks) {
        res.append(block);
    }

    return res;
}

std::vector<block_t> normalize_blocks(
        const std::vector<block_t> &blocks, bool remove_size_1_blocks) {
    if (blocks.empty()) return {};
    std::vector<block_t> res;

    for (const block_t &block : blocks) {
        if (remove_size_1_blocks && block.block == 1) continue;

        if (!res.empty() && res.back().can_merge(block)) {
            res.back().block *= block.block;
        } else {
            res.emplace_back(block);
        }
    }

    return res;
}

block_layout_t::block_layout_t(
        const memory_desc_wrapper &mdw, bool inner_only, bool do_normalize) {
    if (mdw.format_kind() == format_kind::undef) return;

    const dim_idx_t ndims = into<uint32_t>(mdw.ndims());
    auto &blocking = mdw.blocking_desc();
    auto *padded_dims = mdw.padded_dims();

    dim_t stride = 1;
    std::vector<dim_t> full_blocks(ndims, 1);
    for (int i = blocking.inner_nblks - 1; i >= 0; i--) {
        dim_idx_t dim_idx = into<uint32_t>(blocking.inner_idxs[i]);
        dim_t block = blocking.inner_blks[i];
        append(block_t(dim_idx, block, stride));
        stride *= block;
        full_blocks[dim_idx] *= block;
    }

    if (!inner_only) {
        for (dim_idx_t i = 0; i < ndims; i++) {
            dim_t block = padded_dims[i] / full_blocks[i];
            append(block_t(i, block, blocking.strides[i]));
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

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
