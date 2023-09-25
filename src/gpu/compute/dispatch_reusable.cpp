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

#include "gpu/compute/dispatch_reusable.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

void named_buffer_t::disable_dim(size_t dim_idx) {
    disabled_dims.push_back(dim_idx);
}

status_t reusable_dispatch_config_t::define_dim_index(
        const char *dim_name, size_t dim_idx) {
    gpu_assert(indexed_dims.size() < NUM_INDEXED_DIMS
            && "Too many dim indices required.");
    gpu_assert(
            strlen(dim_name) < MAX_DIM_NAME_LENGTH && "Dim name is too long");

    named_dim_t dim(dim_name, dim_idx);

    indexed_dims.emplace_back(dim_name, dim_idx);
    return status::success;
}

// XXX: Due to the possibility of block_structure_t::normalize removing
// all blocks in a named dim (or even all blocks for a size-1 buffer),
// it is not supported to register a buffer that was created via a block_structure_t
// with size-1 blocks removed.
status_t reusable_dispatch_config_t::register_buffer(named_buffer_t &buffer) {
    if (buffers.size() >= MAX_REGISTERED_BUFFERS) return status::unimplemented;

    // Do not allow buffers with zero-padding
    if (!memory_desc_wrapper(buffer.md).is_dense()) {
        return status::unimplemented;
    }

    // Remove disabled dims from the buffer
    named_buffer_t newbuf = buffer;
    for (int i = static_cast<int>(newbuf.layout.size()) - 1; i >= 0; i--) {
        size_t idx = static_cast<size_t>(i);
        auto &block = newbuf.layout[idx];
        bool is_disabled = false;
        for (const auto &dim : newbuf.disabled_dims) {
            if (static_cast<size_t>(block.dim_idx) == dim) {
                is_disabled = true;
                break;
            }
        }
        if (is_disabled) { newbuf.layout.erase(idx); }
    }

    buffers.emplace_back(newbuf);
    return status::success;
}

// Fill the gws_mapped_layout_t struct using the first buffer
// XXX: This means that access patterns are most optimal for the 1st buffer
gws_mapped_layout_t reusable_dispatch_config_t::compute_gws_blocks(
        const gws_layout_t &base_layout, const lws_strategy_t &lws_strategy) {
    // Sort blocks into "combinable" bins
    std::vector<std::vector<const gws_mapped_block_t *>> block_bins;
    for (const auto &block : base_layout) {
        bool is_inserted = false;
        for (auto &bin : block_bins) {
            // Relies on blocks being sorted by stride
            const auto &binned_block = bin.back();
            if ((block.is_indexed || binned_block->is_indexed)
                    && block.dim_idx != binned_block->dim_idx)
                continue;
            if (block.stride != binned_block->stride * binned_block->block)
                continue;

            bin.emplace_back(&block);
            is_inserted = true;
            break;
        }

        if (!is_inserted) block_bins.push_back({&block});
    }

    // Put each bin into its own dimension (placing extras in the final gws dim)
    gws_mapped_layout_t gws_blocks(lws_strategy);
    for (size_t i = 0; i < block_bins.size(); i++) {
        size_t mapped_idx = std::min(i, size_t {GWS_MAX_NDIMS - 1});
        for (auto *block : block_bins[i]) {
            gws_blocks.add_block(static_cast<size_t>(block->block),
                    block->dim_idx, indexed_dims, mapped_idx);
        }
    }
    return gws_blocks;
}

// Depends on nd_range being set already
// Computes the mapping from each buffer into the given nd range,
// by setting each block's mapped_to_idx value
status_t reusable_dispatch_config_t::compute_gws_mapping(
        gws_layout_t &layout, const gws_mapped_layout_t &gws_blocks) {
    std::vector<bool> mapped_to(layout.size(), false);
    for (size_t j = 0; j < layout.size(); j++) {
        auto &block = layout[j];
        for (size_t k = 0; k < gws_blocks.get_blocks().size(); k++) {
            if (mapped_to[k]) continue;
            const auto &gws_block = gws_blocks.get_blocks()[k];
            if (gws_block.dim_idx == block.dim_idx) {
                // map block onto gws_block
                block.mapped_to_idx = k;
                block.gws_idx = gws_block.gws_idx;
                mapped_to[k] = true;
                break;
            }
        }
    }

    for (bool block_is_mapped : mapped_to) {
        if (!block_is_mapped) return status::unimplemented;
    }

    return status::success;
}

void reusable_dispatch_config_t::compute_dim_terms(const named_dim_t &dim,
        size_t dim_idx, const gws_mapped_layout_t &gws_blocks) {
    // Compute the terms needed for this dim
    dim_t blocking_size = 1;
    for (const auto &block : gws_blocks.get_blocks()) {
        if (static_cast<size_t>(block.dim_idx) != dim.idx) continue;

        gws_op op;
        size_t block_stride = static_cast<size_t>(block.stride);
        if (gws_blocks.get_num_blocks(block.gws_idx) == 1) {
            op = blocking_size > 1 ? gws_op::SOLO_BLOCK : gws_op::SOLO;
        } else {
            if (block_stride
                    == gws_blocks.nd_range().global_range()[block.gws_idx]) {
                op = blocking_size > 1 ? gws_op::FIRST_BLOCK : gws_op::FIRST;
            } else {
                op = blocking_size > 1 ? gws_op::MOD_BLOCK : gws_op::MOD;
            }
        }
        term_list.add_dim_term(dim_idx, op, block.gws_idx, block.block,
                block_stride, blocking_size);
        blocking_size *= block.block;
    }
}

// Loop over each gws dim and combine terms when sequential blocks form a dense
// block in the tensor - this generally depends on the strides of each block
// (which is a runtime value, and therefore shouldn't be relied upon via
// reusable structures), but in this case it's more closely related to the
// blocking structure than the actual block/dimension sizes. This is a balance
// between reusability and optimization.
void reusable_dispatch_config_t::compute_buffer_terms(
        const gws_layout_t &layout, size_t buffer_idx,
        const gws_mapped_layout_t &gws_blocks) {
    for (size_t i = 0; i < GWS_MAX_NDIMS; i++) {
        std::vector<gws_mapped_block_t> dim_blocks;
        std::vector<gws_mapped_block_t> gws_dim_blocks;
        // Pull out blocks (and gws-mapped blocks) for this dim
        for (const auto &block : layout) {
            if (block.gws_idx == i) {
                dim_blocks.emplace_back(block);
                gws_dim_blocks.emplace_back(
                        gws_blocks.get_blocks()[block.mapped_to_idx]);
            }
        }

        // Combine blocks when possible
        for (size_t j = 1; j < dim_blocks.size(); j++) {
            auto &block = dim_blocks[j];
            auto &prev_block = dim_blocks[j - 1];
            if (prev_block.stride * prev_block.block == block.stride) {
                prev_block.block *= block.block;
                auto &gws_block = gws_dim_blocks[j];
                auto &prev_gws_block = gws_dim_blocks[j - 1];
                prev_gws_block.block *= gws_block.block;
                dim_blocks.erase(dim_blocks.begin() + static_cast<int>(j));
                gws_dim_blocks.erase(
                        gws_dim_blocks.begin() + static_cast<int>(j));
                j--;
            }
        }

        // Create a term for each remaining block
        for (size_t j = 0; j < dim_blocks.size(); j++) {
            const auto &block = dim_blocks[j];
            const auto &gws_block = gws_dim_blocks[j];
            gws_op op;
            size_t block_stride = static_cast<size_t>(gws_block.stride);
            size_t num_blocks = gws_dim_blocks.size();
            if (num_blocks == 1) {
                op = block.stride > 1 ? gws_op::SOLO_BLOCK : gws_op::SOLO;
            } else {
                const gws_mapped_block_t &first_block = gws_dim_blocks.back();
                if (gws_block == first_block) {
                    op = block.stride > 1 ? gws_op::FIRST_BLOCK : gws_op::FIRST;
                } else {
                    op = block.stride > 1 ? gws_op::MOD_BLOCK : gws_op::MOD;
                }
            }
            term_list.add_buffer_term(buffer_idx, op, block.gws_idx,
                    block.block, block_stride, block.stride);
        }
    }
}

std::array<block_t, 2> split(const block_t &block, dim_t size) {
    gpu_assert(block.block % size == 0);
    return {block_t(block.dim_idx, size, block.stride),
            block_t(block.dim_idx, block.block / size, block.stride * size)};
}

// Make layoutA and layoutB match each other by subdividing blocks in each one as
// necessary, to get:
// 1. The same number of blocks for each dimension
// 2. Identical block sizes/ordering for each dimension
status_t reconcile_via_subdivide(
        block_layout_t &layoutA, block_layout_t &layoutB) {
    // Partition each layout by dimension
    std::array<std::vector<block_t>, DNNL_MAX_NDIMS> dim_blocksA;
    std::array<std::vector<size_t>, DNNL_MAX_NDIMS> dim_block_idxA;
    for (size_t i = 0; i < layoutA.size(); i++) {
        const auto &block = layoutA[i];
        dim_blocksA[static_cast<size_t>(block.dim_idx)].emplace_back(block);
        dim_block_idxA[static_cast<size_t>(block.dim_idx)].emplace_back(i);
    }
    std::array<std::vector<block_t>, DNNL_MAX_NDIMS> dim_blocksB;
    std::array<std::vector<size_t>, DNNL_MAX_NDIMS> dim_block_idxB;
    for (size_t i = 0; i < layoutB.size(); i++) {
        const auto &block = layoutB[i];
        dim_blocksB[static_cast<size_t>(block.dim_idx)].emplace_back(block);
        dim_block_idxB[static_cast<size_t>(block.dim_idx)].emplace_back(i);
    }

    // Iterate through blocks for each dimension and subdivide
    for (size_t idx = 0; idx < DNNL_MAX_NDIMS; idx++) {
        std::vector<block_t> &blocksA = dim_blocksA[idx];
        std::vector<block_t> &blocksB = dim_blocksB[idx];

        size_t block_idxA = 0;
        size_t block_idxB = 0;
        while (block_idxA < blocksA.size() || block_idxB < blocksB.size()) {
            // If subdivision results in a different number of blocks between
            // the 2 layouts, they cannot be reconciled
            if (block_idxA >= blocksA.size() || block_idxB >= blocksB.size()) {
                return status::unimplemented;
            }

            block_t &blockA = blocksA[block_idxA];
            block_t &blockB = blocksB[block_idxB];

            if (blockA.block == blockB.block) {
                // Blocks already match
                block_idxA++;
                block_idxB++;
            } else if (blockA.block % blockB.block == 0) {
                // BlockA can be subdivided by blockB
                std::array<block_t, 2> split_blocks
                        = split(blockA, blockB.block);

                size_t buffer_block_idx = dim_block_idxA[idx][block_idxA];
                blockA = split_blocks[0];
                blocksA.insert(
                        blocksA.begin() + block_idxA + 1, split_blocks[1]);
                layoutA[buffer_block_idx] = split_blocks[0];
                layoutA.insert(buffer_block_idx + 1, split_blocks[1]);

                block_idxA++;
                block_idxB++;
            } else if (blockB.block % blockA.block == 0) {
                // BlockB can be subdivided by blockA
                std::array<block_t, 2> split_blocks
                        = split(blockB, blockA.block);

                size_t layout_idx = dim_block_idxB[idx][block_idxB];
                blockB = split_blocks[0];
                blocksB.insert(
                        blocksB.begin() + block_idxB + 1, split_blocks[1]);
                layoutB[layout_idx] = split_blocks[0];
                layoutB.insert(layout_idx + 1, split_blocks[1]);

                block_idxA++;
                block_idxB++;
            } else {
                // Blocks don't match and we can't subdivide them: fail
                return status::unimplemented;
            }
        }
    }
    return status::success;
}

// XXX: Mapping blocks into the gws cannot happen until all necessary dim indices
// have been requested and all buffers have been registered. Only then can the terms
// be computed, thus it's all done in the generate function
status_t reusable_dispatch_config_t::generate(
        reusable_dispatch_t &dispatch, const lws_strategy_t &lws_strategy) {
    // The reusable dispatcher must have at least one buffer to dispatch against
    gpu_assert(!buffers.empty());

    // Add size-1 blocks for each indexed dim as needed
    for (auto &buffer : buffers) {
        const dims_t &padded_dims = buffer.md->padded_dims;
        const dims_t &strides = buffer.md->format_desc.blocking.strides;
        for (const auto &dim : indexed_dims) {
            if (padded_dims[dim.idx] == 1) {
                block_t added_block(static_cast<dim_t>(dim.idx),
                        padded_dims[dim.idx], strides[dim.idx]);
                bool is_added = false;
                for (size_t i = 0; i < buffer.layout.size(); i++) {
                    if (buffer.layout[i].stride > added_block.stride) {
                        buffer.layout.insert(i, added_block);
                        is_added = true;
                        break;
                    }
                }
                if (!is_added) buffer.layout.append(added_block);
            }
        }
    }

    // Generate an initial set of gws blocks
    block_layout_t gws_layout = buffers.front().layout;

    // Subdivide the gws layout to match all buffers
    for (auto &buffer : buffers) {
        CHECK(reconcile_via_subdivide(buffer.layout, gws_layout));
    }

    // gws_layout is guaranteed to be as subdivided as needed for every
    // registered buffer. Now do one final pass to update each buffer
    // to match
    for (auto &buffer : buffers) {
        CHECK(reconcile_via_subdivide(buffer.layout, gws_layout));
    }

    // TODO: gws_layout is no longer used at this point, but it could be used to reduce
    // similar work with gws_blocks below.

    std::vector<gws_layout_t> gws_layouts;
    for (const auto &buffer : buffers) {
        gws_layout_t layout;
        for (const auto &block : buffer.layout) {
            // XXX: gws_dim=0 is updated in compute_gws_mapping,
            // once the mapped gws block is known
            layout.emplace_back(block, indexed_dims, 0);
        }

        if (layout.empty()) {
            // size-1 buffer case: add a single size-1 block
            layout.emplace_back(block_t(0, 1, 1), indexed_dims, 0);
        }

        gws_layouts.emplace_back(layout);
    }

    // Somewhat arbitrarily pick the first layout to generate the gws mapping
    // This means that the first buffer will likely have more optimal indexing
    auto gws_blocks = compute_gws_blocks(gws_layouts.front(), lws_strategy);

    for (size_t i = 0; i < gws_layouts.size(); i++) {
        auto &layout = gws_layouts[i];
        if (layout.size() != gws_blocks.get_blocks().size())
            return status::unimplemented;
        CHECK(compute_gws_mapping(layout, gws_blocks));
        compute_buffer_terms(layout, i, gws_blocks);
    }

    for (size_t i = 0; i < indexed_dims.size(); i++) {
        const auto &dim = indexed_dims[i];
        compute_dim_terms(dim, i, gws_blocks);
    }

    if (term_list.terms.size() >= MAX_INDEXING_TERMS) {
        return status::unimplemented;
    }

    dispatch = reusable_dispatch_t(
            buffers, indexed_dims, term_list, gws_blocks.nd_range());
    return status::success;
}

void dispatch_compile_params_t::def_kernel_macros(
        kernel_ctx_t &kernel_ctx, const char *suffix) const {
    kernel_ctx.define_int("GWS_WITH_RUNTIME_PARAMS", 1);

    // Find a unique prefix (in case there are many kernels in a file).
    std::string gws_prefix;
    for (int i = 0; i < 4; i++) {
        if (!kernel_ctx.has_macro(utils::format("GWS%d_DEF", i))) {
            gws_prefix = "GWS" + std::to_string(i);
            break;
        }
    }
    gpu_assert(!gws_prefix.empty());
    kernel_ctx.define_int(utils::format("%s_DEF", gws_prefix.c_str()), 1);

    // For each term, define each parameter
    for (size_t i = 0; i < num_terms; i++) {
        const gws_indexing_term_t &term = terms[i];
        const char *gws_dim_op;
        switch (term.op) {
            case (gws_op::ZERO): gws_dim_op = "ZERO"; break;
            case (gws_op::SOLO): gws_dim_op = "SOLO"; break;
            case (gws_op::FIRST): gws_dim_op = "FIRST"; break;
            case (gws_op::MOD): gws_dim_op = "MOD"; break;
            case (gws_op::SOLO_BLOCK): gws_dim_op = "SOLO_BLOCK"; break;
            case (gws_op::FIRST_BLOCK): gws_dim_op = "FIRST_BLOCK"; break;
            case (gws_op::MOD_BLOCK): gws_dim_op = "MOD_BLOCK"; break;
            default: assert(!"Not expected");
        }
        // GWS<X>_OP<Y>
        kernel_ctx.add_option(utils::format(
                "-D%s_OP%d=GWS_OP_%s", gws_prefix, i, gws_dim_op));

        // GWS<X>_RT_IDX<Y>
        kernel_ctx.define_int(utils::format("%s_RT_IDX%d", gws_prefix, i),
                static_cast<dim_t>(term.rt_data_index));

        // GWS<X>_IDX<Y>
        kernel_ctx.define_int(utils::format("%s_IDX%d", gws_prefix, i),
                static_cast<dim_t>(term.gws_idx));
    }

    // For each buffer, define the sum that leads to the offset calculation
    for (size_t i = 0; i < num_buffers; i++) {
        const char *name = buffer_names[i];
        std::string equation;
        for (size_t j = 0; j < buffer_num_terms[i]; j++) {
            equation += utils::format("%s_GET_ID%d(rt_params)", gws_prefix,
                    buffer_term_index[i][j]);
            if (j != buffer_num_terms[i] - 1) { equation += "+"; }
        }
        // GWS_<BUFFER_NAME>_<SUFFIX>_OFF
        kernel_ctx.add_option(utils::format("-DGWS_%s_%s_OFF(rt_params)=%s",
                name, suffix, equation.c_str()));
    }

    // For each indexed dim, define the sum that leads to the index calculation
    for (size_t i = 0; i < num_indexed_dims; i++) {
        const char *name = dim_names[i];
        std::string equation;
        for (size_t j = 0; j < dim_num_terms[i]; j++) {
            equation += utils::format(
                    "%s_GET_ID%d(rt_params)", gws_prefix, dim_term_index[i][j]);
            if (j != dim_num_terms[i] - 1) { equation += "+"; }
        }
        // GWS_GET_<DIM_NAME>_<SUFFIX>
        kernel_ctx.add_option(utils::format("-DGWS_GET_%s_%s(rt_params)=%s",
                name, suffix, equation.c_str()));
    }

    // TODO: set to 1 when vectorization is needed
    kernel_ctx.define_int(utils::format("GWS_WITH_SG_%s", suffix), 0);
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl
