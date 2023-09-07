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

#include "reduction_utils.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// Convert a block structure + dims to a list of zero-padding structs
// Note: Doesn't include blocking structures that don't require zero-padding.
std::vector<zero_padding_t> calc_zero_padding(
        const std::vector<block_t> &blocks, const memory_desc_wrapper &mdw) {
    std::vector<zero_padding_t> out;
    const blocking_desc_t &src_blocking = mdw.blocking_desc();
    const dim_t *dims = mdw.dims();
    for (int i = 0; i < src_blocking.inner_nblks; i++) {
        // Check if this needs zero-padding
        const dim_t dim_idx = src_blocking.inner_idxs[i];
        const dim_t blk_size = src_blocking.inner_blks[i];
        if (dims[dim_idx] % blk_size != 0) {
            // Needs zero-padding: Find the 1 or 2 blocks related to this zero-padding
            const block_t *inner_block = nullptr;
            const block_t *outer_block = nullptr;
            for (size_t j = 0; j < blocks.size(); j++) {
                const block_t &block = blocks[j];
                if (block.dim_idx == dim_idx) {
                    if (!inner_block) {
                        inner_block = &block;
                    } else {
                        outer_block = &block;
                        break;
                    }
                }
            }
            assert(inner_block);
            block_t default_outer = block_t(
                    static_cast<int>(dim_idx), 1, mdw.strides()[dim_idx]);
            if (!outer_block) outer_block = &default_outer;
            out.emplace_back(dims[dim_idx], *outer_block, *inner_block);
        }
    }
    return out;
}

block_t merge_blocks(
        std::vector<block_t> blocks, size_t start_idx, size_t end_idx) {
    block_t ret = blocks[start_idx];
    for (size_t i = start_idx + 1; i < end_idx; i++) {
        block_t &next_block = blocks[i];
        // Assumes they're ordered by increasing stride
        assert(ret.stride * ret.block == next_block.stride);
        ret.block *= next_block.block;
    }
    return ret;
}

// Produce a subproblem needed to perform the reduction of red_block after a given subproblem
reduction_subproblem_t chain_reductions(
        const reduction_subproblem_t &prev_subprb, const block_t &red_block) {
    // Copy shape/block layout to the next subproblem
    const dim_t outer_stride = red_block.stride * red_block.block;
    const dim_t nelems
            = prev_subprb.inner_block.block * prev_subprb.outer_block.block;
    reduction_subproblem_t ret(
            red_block.stride, red_block.block, nelems / outer_stride);

    ret.src_zpads = prev_subprb.dst_zpads;
    return ret;
}

// Convert a src/dst pair to a sequence of reduction subproblems
// by normalizing dimensions via combining blocks when possible
// Example: --stag=aBx16b --dtag=aBx16b 1x30x4x2x2:1x1x1x2x2
// 1) use compute_block_structure to get rearrange src to: 2'x'4x2x2x16'
//    (blocks with ' need to be reduced)
// 2) Create a subproblem to reduce the innermost block, combining
//    all other dims to one outer dim. This is equivalent to:
//       32x16x1:32x1x1 (+src zero padding)
//    After this step, we're left with the intermediate structure 2'x4'x2x2x1
// 3) Create another subproblem to reduce the remaining dims (combining to one
//    block because sequential blocks can be):
//       1x8x4:1x1x4 (+dst zero padding)
// 4) Attach src zero-padding to first problem and dst zero-padding to the last:
//      src: (idx / 1) % 16 + [(idx / 256) % 2] * 16 < 30 aren't zeros
//      dst: (idx / 1) % 16 + [(idx / 64) % 1] * 16 < 1 aren't zeros
status_t generate_reduction_phases(const memory_desc_t *src,
        const memory_desc_t *dst,
        std::vector<reduction_subproblem_t> &subprbs) {
    int reduced_dim_mask
            = ~utils::get_dims_mask(src->dims, dst->dims, src->ndims)
            & ((1 << src->ndims) - 1);
    auto is_masked
            = [](int mask, dim_t dim_idx) { return mask & (1 << dim_idx); };
    memory_desc_wrapper src_mdw(src);
    memory_desc_wrapper dst_mdw(dst);

    std::vector<block_t> src_blocks = compute_block_structure(src_mdw);
    std::vector<block_t> dst_blocks = compute_block_structure(dst_mdw);

    // Requirement: dst blocks match src blocks with the exception of reduced dims (these
    // blocks are removed) and dst zero-padding on reduced dims (these are added back in)
    std::vector<block_t> exp_dst_blocks;
    int dst_zpad_mask
            = ~utils::get_dims_mask(dst->dims, dst->padded_dims, dst->ndims);
    int stride = 1;
    for (const auto &block : src_blocks) {
        if (!is_masked(reduced_dim_mask, block.dim_idx)) {
            // Non-reduced dims get transferred directly to dst (no reorders)
            exp_dst_blocks.push_back(block);
            exp_dst_blocks.back().stride = stride;
            stride *= block.block;
        } else if (is_masked(dst_zpad_mask, block.dim_idx)) {
            // dst-zpadded, reduced dims get added to dst as well
            exp_dst_blocks.push_back(block);
            exp_dst_blocks.back().stride = stride;
            stride *= block.block;

            // Outer blocks are removed still (first encountered block is always the inner one)
            dst_zpad_mask &= ~(1 << block.dim_idx);
        } // Otherwise, it's reduced and removed, not added to dst
    }
    exp_dst_blocks = normalize_blocks(exp_dst_blocks);

    // Make sure dst matches the expected format
    if (dst_blocks.size() != exp_dst_blocks.size()) {
        return status::unimplemented;
    }
    for (size_t i = 0; i < dst_blocks.size(); i++) {
        const block_t dst_block = dst_blocks[i];
        const block_t exp_dst_block = exp_dst_blocks[i];
        if (dst_block != exp_dst_block) { return status::unimplemented; }
    }

    std::vector<block_t> reduction_blocks;
    static constexpr size_t DIM_NOT_FOUND = std::numeric_limits<size_t>::max();
    size_t first_reduction_dim = DIM_NOT_FOUND;
    for (size_t i = 0; i < src_blocks.size(); i++) {
        block_t block = src_blocks[i];
        if (first_reduction_dim == DIM_NOT_FOUND
                && is_masked(reduced_dim_mask, block.dim_idx)) {
            first_reduction_dim = i;
        } else if (first_reduction_dim != DIM_NOT_FOUND
                && !is_masked(reduced_dim_mask, block.dim_idx)) {
            reduction_blocks.push_back(
                    merge_blocks(src_blocks, first_reduction_dim, i));
            first_reduction_dim = DIM_NOT_FOUND;
        }
    }
    if (first_reduction_dim != DIM_NOT_FOUND) {
        reduction_blocks.push_back(merge_blocks(
                src_blocks, first_reduction_dim, src_blocks.size()));
    }

    // Sequentially create subproblems after a partial reduction
    const dim_t nelems = src_mdw.nelems(true);
    subprbs.emplace_back(nelems, 1, 1);
    reduction_subproblem_t &base_subprb = subprbs.back();

    base_subprb.dst_zpads = calc_zero_padding(src_blocks, src_mdw);

    for (const auto &red_block : reduction_blocks) {
        const reduction_subproblem_t &prev_subprb = subprbs.back();
        subprbs.push_back(chain_reductions(prev_subprb, red_block));

        // Update the strides of all remaining reduction blocks after subproblem-i
        for (block_t &other_block : reduction_blocks) {
            if (other_block.stride > red_block.stride) {
                other_block.stride /= red_block.block;
            }
        }
    }

    // Remove the base subproblem from the list
    subprbs.erase(subprbs.begin());

    // Step 7: Potentially add dst-zero-padding if needed for the final reduction dimensions.
    reduction_subproblem_t &last_subprb = subprbs.back();
    const auto &dst_blk = dst_mdw.blocking_desc();
    for (size_t i = 0; i < static_cast<size_t>(dst_blk.inner_nblks); i++) {
        const dim_t dim_idx = dst_blk.inner_idxs[i];
        const bool needs_zero_padding
                = (dst_mdw.dims()[dim_idx] < dst_mdw.padded_dims()[dim_idx]);
        bool accounted_for = false;
        for (const auto &zpad : last_subprb.dst_zpads) {
            if (zpad.dim_idx == dim_idx) {
                accounted_for = true;
                break;
            }
        }
        if (needs_zero_padding && !accounted_for) {
            const block_t default_outer(
                    static_cast<int>(dim_idx), 1, dst_mdw.strides()[dim_idx]);

            // Get the first (inner) and second (outer) block for this dim
            const block_t *inner = nullptr;
            const block_t *outer = &default_outer;
            for (const auto &block : dst_blocks) {
                if (block.dim_idx == dim_idx) {
                    if (!inner) {
                        inner = &block;
                    } else {
                        outer = &block;
                        break;
                    }
                }
            }
            assert(inner);

            zero_padding_t zpad(dst_mdw.dims()[dim_idx], *outer, *inner);
            last_subprb.dst_zpads.push_back(zpad);
        }
    }

    // Sort dst zpadding by increasing inner stride
    std::sort(last_subprb.dst_zpads.begin(), last_subprb.dst_zpads.end(),
            [](zero_padding_t &first, zero_padding_t &last) -> bool {
                return first.inner_stride < last.inner_stride;
            });

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
