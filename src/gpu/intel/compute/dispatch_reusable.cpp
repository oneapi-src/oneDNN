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

#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "common/c_types_map.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/data_type_converter.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

// Enables the use of intel subgroups in the kernel.
// A buffer is supplied to specify which block (stride=1 block for the buffer)
// is guaranteed to be dispatched across in the subgroup. Memory access
// patterns may be non-contiguous in other buffers (i.e. block read/write is only guaranteed
// to be valid for this buffer)
status_t reusable_dispatch_config_t::use_subgroup(
        const std::string &buf_name, size_t size) {
    if (!engine->mayiuse_sub_group(static_cast<int>(size))) {
        return status::unimplemented;
    }

    // Cannot use a subgroup on two buffers
    gpu_assert(!subgroup.used());

    // Look for a registered buffer with the given name
    for (size_t i = 0; i < buffers.size(); i++) {
        if (buffers[i].get_name() == buf_name) {
            subgroup = subgroup_data_t(i, size);
            break;
        }
    }

    // If we couldn't find the buffer, something has gone wrong
    if (!subgroup.used()) { return status::runtime_error; }

    return status::success;
}

status_t reusable_dispatch_config_t::define_dim_index(
        const char *dim_name, dim_id_t dim_id, dim_t size) {
    memory_desc_t md = types::zero_md();
    md.ndims = 1;
    md.dims[0] = size;
    md.padded_dims[0] = size;
    md.data_type = data_type::f32; // doesn't matter
    md.format_kind = format_kind::blocked;
    md.format_desc.blocking.strides[0] = 1;
    md.format_desc.blocking.inner_nblks = 0;

    named_buffer_t buf(dim_name, md, {dim_id});
    CHECK(register_buffer(buf));
    return status::success;
}

// Validate whether the given buffer is consistent with existing buffer layouts,
// and then add to the internal buffer registry.
status_t reusable_dispatch_config_t::register_buffer(
        const named_buffer_t &buffer) {
    if (buffers.size() >= MAX_REGISTERED_BUFFERS) return status::unimplemented;

    // Don't allow zero-padding
    bool has_zero_padding = false;
    for (size_t dim_idx = 0; dim_idx < static_cast<size_t>(buffer.ndims);
            dim_idx++) {
        if (buffer.dims[dim_idx] < buffer.padded_dims[dim_idx]) {
            has_zero_padding = true;
        }
    }
    if (has_zero_padding) return status::unimplemented;

    // Validate dim sizes
    std::unordered_map<dim_id_t, bool, dim_id_hash_t> dim_seen;
    for (const auto &dim : dispatched_dims) {
        size_t canonical_idx = buffer.get_dim_idx(dim);
        if (canonical_idx == dim_not_found) {
            // broadcasted dimension - nothing to check
            continue;
        }

        dim_seen[dim] = (dim_sizes.find(dim) != dim_sizes.end());

        if (dim_seen[dim] && (dim_sizes[dim] != buffer.dims[canonical_idx])) {
            // Attempting to dispatch to multiple buffers with differently
            // sized dispatched dimensions. These buffers are incompatible.
            return status::runtime_error;
        }
    }

    // All validation complete - start updating this object
    for (const auto &dim : dispatched_dims) {
        size_t canonical_idx = buffer.get_dim_idx(dim);
        if (canonical_idx == dim_not_found) continue;

        // Save the dimension size if it hasn't been saved yet
        if (!dim_seen[dim]) { dim_sizes[dim] = buffer.dims[canonical_idx]; }
    }
    buffers.emplace_back(buffer);
    return status::success;
}

// ZERO: The block only has 1 element, with index 0
// SOLO: There is 1 block that accounts for the entire GWS dim
// FIRST: There are multiple blocks in the GWS dim, but this is the outermost
// MOD: There are multiple blocks in the GWS dim, and this is not
//       outermost (it needs a modulus)
// *_BLOCK variant: buffer stride is greater than 1, so we have to
//       multiply indices by a block size
gws_op_t get_op(size_t gws_size, stride_t gws_stride, const block_t &block) {
    if (block.block == 1) return gws_op_t::ZERO;

    if (static_cast<size_t>(block.block) == gws_size) {
        return block.stride > 1 ? gws_op_t::SOLO_BLOCK : gws_op_t::SOLO;
    }

    bool is_outermost = (gws_stride * block.block
            == stride_t(static_cast<dim_t>(gws_size)));
    if (is_outermost) {
        return block.stride > 1 ? gws_op_t::FIRST_BLOCK : gws_op_t::FIRST;
    }

    return block.stride > 1 ? gws_op_t::MOD_BLOCK : gws_op_t::MOD;
}

// Will mutate a vector of layouts as needed to make each dimension:
// 1. have the same number of blocks,
// 2. each with the same size,
// 3. in the same order
// by subdividing existing blocks
class layout_equalizer_t {
public:
    static constexpr int broadcasted_block = -1;
    layout_equalizer_t() = default;

    status_t register_layout(const block_layout_t &layout) {
        if (master_layout.empty()) {
            for (const block_t &block : layout) {
                master_layout.emplace_back(num_layouts, block);
            }
            num_layouts++;
            return status::success;
        }

        // subdivide the new and master layouts as needed to match
        block_layout_t new_layout;
        CHECK(subdivide(layout, new_layout));

        // For each block, find the correct master term to add to
        std::vector<bool> is_mapped_to(master_layout.size(), false);
        for (const block_t &block : new_layout) {
            bool is_mapped = false;
            for (size_t i = 0; i < master_layout.size(); i++) {
                if (is_mapped_to[i]) continue;

                auto &master_block = master_layout[i];
                if (master_block.matches(block)) {
                    is_mapped = true;
                    is_mapped_to[i] = true;
                    master_block.map(num_layouts, block);
                    break;
                }
            }
            if (!is_mapped) {
                master_layout.emplace_back(num_layouts, block);
                is_mapped_to.push_back(true);
            }
        }
        num_layouts++;

        return status::success;
    }

    const std::unordered_map<size_t, block_t> &buffer_blocks(size_t idx) {
        return master_layout[idx].get_buffer_blocks();
    }

    // mutates master_layout and returns a matching layout
    status_t subdivide(const block_layout_t &layout, block_layout_t &res) {
        // Can subdivide as long as all dims have the same size as master_layout
        // (or layout size is 1, as in broadcasted dims)
        std::array<size_t, DNNL_MAX_NDIMS> layout_dim_sizes;
        layout_dim_sizes.fill(1);
        for (const block_t &block : layout) {
            layout_dim_sizes[static_cast<size_t>(block.dim_idx)]
                    *= static_cast<size_t>(block.block);
        }

        std::array<size_t, DNNL_MAX_NDIMS> master_dim_sizes;
        master_dim_sizes.fill(1);
        for (const mapped_block_t &block : master_layout) {
            master_dim_sizes[block.get_dim_idx()] *= block.get_size();
        }

        for (size_t i = 0; i < DNNL_MAX_NDIMS; i++) {
            if (layout_dim_sizes[i] == 1 || master_dim_sizes[i] == 1) continue;
            if (layout_dim_sizes[i] != master_dim_sizes[i]) {
                return status::runtime_error;
            }
        }

        // Shapes are coherent, start subdividing
        res = layout;
        std::vector<bool> is_mapped_to(master_layout.size(), false);
        for (size_t i = 0; i < res.size(); i++) {
            block_t &block = res[i];
            size_t block_size = static_cast<size_t>(block.block);
            for (size_t j = 0; j < master_layout.size(); j++) {
                if (is_mapped_to[j]) continue;

                mapped_block_t &master_block = master_layout[j];
                if (master_block.get_dim_idx()
                        != static_cast<size_t>(block.dim_idx))
                    continue;

                size_t master_size = master_block.get_size();
                if (master_size == block_size) {
                    // Nothing to do, already matches
                } else if (block_size % master_size == 0) {
                    // subdivide block
                    block.block = static_cast<dim_t>(master_size);
                    block_t next_block(block.dim_idx,
                            static_cast<dim_t>(block_size / master_size),
                            block.stride * static_cast<dim_t>(master_size));
                    res.insert(i + 1, next_block);
                } else if (master_size % block_size == 0) {
                    // subdivide master block
                    mapped_block_t next_block = master_block.split(block_size);
                    master_layout.insert(
                            master_layout.begin() + j + 1, next_block);
                } else {
                    // Should never be able to reach this case...
                    return status::runtime_error;
                }
                is_mapped_to[j] = true;
                break;
            }
        }

        return status::success;
    }

    std::vector<block_bin_t> compute_block_bins(
            const lws_strategy_t &lws_strat, const subgroup_data_t &subgroup) {
        std::vector<block_bin_t> bins;
        for (size_t i = 0; i < master_layout.size(); i++) {
            const mapped_block_t &mapped_blocks = master_layout[i];

            // mapped_block_t that are in the lws have to be
            // at the start of a new bin
            if (subgroup.used()) {
                // The subgroup block has to be in the lws
                size_t sg_buf_idx = subgroup.buffer_idx();
                if (!mapped_blocks.is_broadcasted(sg_buf_idx)) {
                    const block_t &buf_block
                            = mapped_blocks.get_buffer_blocks().at(sg_buf_idx);
                    if (static_cast<size_t>(buf_block.stride * buf_block.block)
                            <= subgroup.size()) {
                        // This mapped_block_t corresponds to the subgroup block
                        bins.emplace_back(mapped_blocks, num_layouts, true);
                        continue;
                    }
                }
            }

            // The lws_strategy_t can specify other blocks to be in the lws as well
            if (lws_strat.is_included(mapped_blocks)) {
                bins.emplace_back(mapped_blocks, num_layouts, true);
                continue;
            }

            bool found_bin = false;
            for (block_bin_t &bin : bins) {
                if (bin.get_blocks().back().can_merge(mapped_blocks)) {
                    found_bin = true;
                    bin.append(mapped_blocks);
                    break;
                }
            }
            if (!found_bin) bins.emplace_back(mapped_blocks, num_layouts);
        }

        return bins;
    }

private:
    void split_block(size_t block_idx, size_t size) {
        mapped_block_t next_block = master_layout[block_idx].split(size);
        master_layout.insert(master_layout.begin() + block_idx + 1, next_block);
    }

    std::vector<mapped_block_t> master_layout;
    size_t num_layouts = 0;
};

// Used in compute_terms to store the block_t data and info about
// where it's mapped to in the GWS
struct gws_mapped_block_t : public gpu::intel::block_t {
    gws_mapped_block_t() = default;
    gws_mapped_block_t(
            const block_t &block, size_t gws_idx, stride_t gws_stride)
        : block_t(block), gws_idx(gws_idx), gws_stride(gws_stride) {}

    std::string str() const {
        std::ostringstream ss;
        ss << static_cast<const block_t *>(this)->str().c_str();
        ss << " , gws_stride=" << gws_stride.str();
        ss << " / gws_idx=" << gws_idx;
        return ss.str();
    }

    size_t gws_idx;
    stride_t gws_stride;
};

std::vector<gws_indexing_term_t> gws_bin_mapping_t::condense_terms(
        size_t buf_idx) const {
    std::vector<gws_indexing_term_t> ret;
    for (size_t gws_idx = 0; gws_idx < range_t::max_ndims; gws_idx++) {
        const std::vector<block_bin_t> &bins = map[gws_idx];

        std::vector<gws_mapped_block_t> gws_blocks;
        stride_t gws_stride = 1;
        for (size_t i = 0; i < bins.size(); i++) {
            const block_bin_t &bin = bins[i];
            if (!bin.is_broadcasted(buf_idx)) {
                block_t block = bin.combined_block(buf_idx);
                gws_blocks.emplace_back(block, gws_idx, gws_stride);
            };
            gws_stride *= static_cast<dim_t>(bin.size());
        }
        if (gws_blocks.empty()) continue;

        gws_mapped_block_t block = gws_blocks.front();
        for (size_t i = 1; i < gws_blocks.size(); i++) {
            // Check if it can be merged with the next one
            gws_mapped_block_t &next_block = gws_blocks[i];
            bool is_buffer_dense
                    = (block.stride * block.block == next_block.stride);
            bool is_gws_dense
                    = (block.gws_stride * block.block == next_block.gws_stride);

            if (is_buffer_dense && is_gws_dense) {
                // Merge
                block.block *= next_block.block;
            } else {
                // Create a term and reset the block
                gws_op_t op = get_op(gws_[gws_idx], block.gws_stride, block);

                ret.emplace_back(op, gws_idx, block.block, block.gws_stride,
                        block.stride);

                // Update values for the next block
                block = next_block;
            }
        }

        // Create the final term
        gws_op_t op = get_op(gws_[gws_idx], block.gws_stride, block);

        ret.emplace_back(
                op, gws_idx, block.block, block.gws_stride, block.stride);
    }

    if (ret.empty()) {
        // Size-1 buffer needs to have a zero term
        ret.emplace_back(gws_op_t::ZERO, 0, 0, 0, 0);
    }
    return ret;
}

// XXX: Mapping blocks into the gws cannot happen until all necessary dim indices
// have been requested and all buffers have been registered. Only then can the terms
// be computed, thus it's all done in the generate function
status_t reusable_dispatch_config_t::generate(
        reusable_dispatch_t &dispatch, const lws_strategy_t &lws_strategy) {
    // The reusable dispatcher must have at least one buffer to dispatch against
    gpu_assert(!buffers.empty());

    // Every dispatched dim must have a defined size
    for (dim_id_t id : dispatched_dims) {
        if (dim_sizes.find(id) == dim_sizes.end()) {
            return status::unimplemented;
        }
    }

    std::array<bool, DNNL_MAX_NDIMS> is_dispatched;
    is_dispatched.fill(false);
    for (dim_id_t dim : dispatched_dims) {
        is_dispatched[dim] = true;
    }

    // Store layouts for each buffer, since they'll be manipulated
    // for the rest of the generate function
    layout_equalizer_t equalizer;
    std::vector<block_layout_t> buf_layouts(buffers.size());
    for (size_t i = 0; i < buffers.size(); i++) {
        block_layout_t layout = buffers[i].layout();
        block_layout_t new_layout;
        // Only keep dispatched blocks
        for (const auto &block : layout) {
            if (is_dispatched[static_cast<size_t>(block.dim_idx)]) {
                new_layout.append(block);
            }
        }
        buf_layouts[i] = new_layout;
        CHECK(equalizer.register_layout(new_layout));
    }

    std::vector<block_bin_t> bins
            = equalizer.compute_block_bins(lws_strategy, subgroup);

    // Map bins into gws dims - start with lws bins, then map the rest
    gws_bin_mapping_t gws_map(subgroup);
    for (const block_bin_t &bin : bins) {
        if (bin.is_in_lws()) gws_map.add(bin);
    }
    for (const block_bin_t &bin : bins) {
        if (!bin.is_in_lws()) gws_map.add(bin);
    }

    std::vector<std::vector<size_t>> buffer_term_map(buffers.size());
    gws_term_list_t term_list;
    for (size_t buf_idx = 0; buf_idx < buffers.size(); buf_idx++) {
        std::vector<gws_indexing_term_t> terms
                = gws_map.condense_terms(buf_idx);
        for (const gws_indexing_term_t &term : terms) {
            buffer_term_map[buf_idx].emplace_back(term_list.append(term));
        }
    }

    if (term_list.terms.size() >= MAX_INDEXING_TERMS) {
        return status::unimplemented;
    }

    dispatch = reusable_dispatch_t(buffers, term_list,
            gws_map.nd_range(lws_strategy), subgroup, buffer_term_map);

    return status::success;
}

void dispatch_compile_params_t::def_kernel_macros(
        kernel_ctx_t &kernel_ctx, const char *suffix) const {
    kernel_ctx.define_int("GWS_WITH_RUNTIME_PARAMS", 1);
    if (use_int32_offset) kernel_ctx.add_option("-DUSE_INT32_OFFSET");

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
    for (size_t i = 0; i < gpu_utils::into<size_t>(num_terms); i++) {
        const gws_indexing_term_t::compile_params_t &term = terms[i];
        const char *gws_dim_op = [term]() -> const char * {
            switch (term.op) {
                case (gws_op_t::ZERO): return "ZERO";
                case (gws_op_t::SOLO): return "SOLO";
                case (gws_op_t::FIRST): return "FIRST";
                case (gws_op_t::MOD): return "MOD";
                case (gws_op_t::SOLO_BLOCK): return "SOLO_BLOCK";
                case (gws_op_t::FIRST_BLOCK): return "FIRST_BLOCK";
                case (gws_op_t::MOD_BLOCK): return "MOD_BLOCK";
                default:
                    gpu_assert(false) << "Unexpected GWS indexing operation";
            }
            return nullptr;
        }();
        if (!gws_dim_op) continue; // Will not be hit due to gpu_assert above

        // GWS<X>_OP<Y>
        kernel_ctx.add_option(utils::format(
                "-D%s_OP%zu=GWS_OP_%s", gws_prefix, i, gws_dim_op));

        // GWS<X>_RT_IDX<Y>
        kernel_ctx.define_int(utils::format("%s_RT_IDX%zu", gws_prefix, i),
                gpu_utils::into<dim_t>(i));

        // GWS<X>_IDX<Y>
        kernel_ctx.define_int(utils::format("%s_IDX%zu", gws_prefix, i),
                gpu_utils::into<dim_t>(term.gws_idx));
    }

    // Define data types for conversion (Ignore the default suffix)
    std::string conv_suff = (suffix == std::string("DEFAULT"))
            ? ""
            : utils::format("_%s", suffix);

    // For each buffer, define the sum that leads to the offset calculation
    data_type_converter_t converter;
    for (size_t i = 0; i < num_buffers; i++) {
        const char *name = buffer_names[i];
        if (buffer_types[i] != data_type::undef) {
            converter.register_type(name + conv_suff, buffer_types[i]);
        }
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
    converter.def_kernel_macros(kernel_ctx);

    kernel_ctx.define_int(
            utils::format("GWS_WITH_SG_%s", suffix), subgroup.used() ? 1 : 0);
    if (subgroup.used()) {
        kernel_ctx.define_int(utils::format("GWS_SGS_%s", suffix),
                static_cast<int64_t>(subgroup.size()));
    }
}

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
