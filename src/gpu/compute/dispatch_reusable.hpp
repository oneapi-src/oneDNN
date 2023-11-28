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

#ifndef GPU_COMPUTE_DISPATCH_REUSABLE_HPP
#define GPU_COMPUTE_DISPATCH_REUSABLE_HPP

#include <sstream>
#include <string>
#include <vector>
#include "common/c_types_map.hpp"
#include "gpu/block_structure.hpp"
#include "gpu/compute/compute_engine.hpp"
#include "gpu/compute/dispatch.hpp"
#include "gpu/compute/kernel_ctx.hpp"
#include "gpu/compute/utils.hpp"
#include "gpu/ocl/types_interop.h"
#include "gpu/serialization.hpp"
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

// How many dims the ndrange can assign to
#define GWS_MAX_NDIMS 3
// How many dims can be indexed simultaneously
#define NUM_INDEXED_DIMS (GWS_MAX_NDIMS - 1)
// How many buffers can be registered simultaneously
#define MAX_REGISTERED_BUFFERS 4
// Maximum length of each indexed dim's name
#define MAX_DIM_NAME_LENGTH 15
// Maximum length of each buffer's name
#define MAX_BUFFER_NAME_LENGTH 7

// Ensure that we don't have padding in our structures
static_assert(NUM_INDEXED_DIMS * (MAX_DIM_NAME_LENGTH + 1) % 4 == 0,
        "Padding will be introduced due to indexed dims.");
static_assert(MAX_REGISTERED_BUFFERS * (MAX_BUFFER_NAME_LENGTH + 1) % 4 == 0,
        "Padding will be introduced due to registered buffers.");

enum gws_op {
    ZERO,
    SOLO,
    FIRST,
    MOD,
    SOLO_BLOCK,
    FIRST_BLOCK,
    MOD_BLOCK,
    UNDEF,
};

// Encodes the information needed for one term like (idx / stride % max) * block
// - stride, max, and block are defined by an index into the runtime struct
// - idx is defined by an index into the gws indexing function
// - the gws_op is used to simplify the expression if stride/max/block are known
struct gws_indexing_term_t {
    gws_indexing_term_t() = default;
#if __cplusplus >= 202002L
    bool operator==(const gws_indexing_term_t &) const = default;
#endif
    gws_indexing_term_t(gws_op op, size_t data_idx, size_t gws_idx)
        : op(op), rt_data_index(data_idx), gws_idx(gws_idx) {};
    gws_op op;
    uint8_t padding[4] = {0, 0, 0, 0};
    size_t rt_data_index;
    size_t gws_idx;

    std::string str() const {
        std::stringstream ss;
        ss << "<gws_indexing_term_t op=" << op << ", rt_idx=" << rt_data_index
           << ", gws_idx=" << gws_idx << ">";
        return ss.str();
    }
};

struct gws_term_list_t {
    void add_buffer_term(size_t buf_idx, gws_op op, size_t gws_idx, dim_t size,
            size_t stride, dim_t block) {
        size_t idx = add_term(op, gws_idx, size, stride, block);
        buf_idxs[buf_idx].emplace_back(idx);
    }
    void add_dim_term(size_t dim_idx, gws_op op, size_t gws_idx, dim_t size,
            size_t stride, dim_t block) {
        size_t idx = add_term(op, gws_idx, size, stride, block);
        dim_idxs[dim_idx].emplace_back(idx);
    }
    std::vector<gws_indexing_term_t> terms;
    std::vector<dim_t> sizes;
    std::vector<size_t> strides;
    std::vector<dim_t> blocks;

    std::unordered_map<size_t, std::vector<size_t>> buf_idxs;
    std::unordered_map<size_t, std::vector<size_t>> dim_idxs;

private:
    size_t add_term(
            gws_op op, size_t gws_idx, dim_t size, size_t stride, dim_t block) {
        size_t ret = terms.size();
        terms.emplace_back(op, terms.size(), gws_idx);
        sizes.emplace_back(size);
        strides.emplace_back(stride);
        blocks.emplace_back(block);
        return ret;
    }
};

// The reusable dispatcher interface involves a number of terms like (idx / stride % max) * block,
// and a mapping from several equations into these terms. Equations can share terms,
// and generally correspond to offset calculation for a buffer or dimension index
// calculation. As long as the sharing of terms is reasonably generic, the compiled
// parameters encode just the block structure and therefore are able to be reused.
struct dispatch_compile_params_t {
    dispatch_compile_params_t() = default;
#if __cplusplus >= 202002L
    bool operator==(const dispatch_compile_params_t &) const = default;
#endif

    void def_kernel_macros(
            kernel_ctx_t &kernel_ctx, const char *suffix = "DEFAULT") const;

    std::string str() const {
        std::ostringstream ss;
        ss << "dispatch_compile_params_t<num_terms=" << num_terms;
        ss << ": [";
        for (size_t i = 0; i < num_terms; i++) {
            ss << terms[i].str() << ", ";
        }
        ss << "], num_buffers=" << num_buffers;
        ss << ": [";
        for (size_t i = 0; i < num_buffers; i++) {
            ss << buffer_names[i] << " - [";
            for (size_t j = 0; j < buffer_num_terms[i]; j++) {
                ss << buffer_term_index[i][j] << "/";
            }
            ss << "], ";
        }
        ss << "], num_indexed_dims=" << num_indexed_dims;
        ss << ": [";
        for (size_t i = 0; i < num_indexed_dims; i++) {
            ss << dim_names[i] << " - [";
            for (size_t j = 0; j < dim_num_terms[i]; j++) {
                ss << dim_term_index[i][j] << "/";
            }
            ss << "], ";
        }
        ss << "]>";
        return ss.str();
    }

    size_t num_terms = 0;
    gws_indexing_term_t terms[MAX_INDEXING_TERMS] = {{gws_op::SOLO, 0, 0}};

    // Buffer definitions (each buffer has a name, and a collection of terms
    // used to compute the offset)
    size_t num_buffers = 0;
    char buffer_names[MAX_REGISTERED_BUFFERS][MAX_BUFFER_NAME_LENGTH + 1]
            = {{'\0'}};
    size_t buffer_term_index[MAX_REGISTERED_BUFFERS][MAX_INDEXING_TERMS]
            = {{0}};
    size_t buffer_num_terms[MAX_REGISTERED_BUFFERS] = {0};

    // Indexed dim definitions (each indexed dim has a name, and a collection
    // of terms used to compute the index)
    size_t num_indexed_dims = 0;
    char dim_names[NUM_INDEXED_DIMS][MAX_DIM_NAME_LENGTH + 1] = {{'\0'}};
    size_t dim_term_index[NUM_INDEXED_DIMS][MAX_INDEXING_TERMS] = {{0}};
    size_t dim_num_terms[NUM_INDEXED_DIMS] = {0};
};
assert_trivially_serializable(dispatch_compile_params_t);

class dispatch_runtime_params_t {
public:
    dispatch_runtime_params_t() = default;
    dispatch_runtime_params_t(const nd_range_t &nd_range,
            const std::vector<size_t> &strides, const std::vector<dim_t> &sizes,
            const std::vector<dim_t> &blocks)
        : nd_range(nd_range), num_terms(sizes.size()) {
        gpu_assert(num_terms == strides.size());
        gpu_assert(num_terms == blocks.size());
        gpu_assert(num_terms <= MAX_INDEXING_TERMS);
        for (size_t i = 0; i < num_terms; i++) {
            rt_params.sizes[i] = sizes[i];
            rt_params.strides[i] = static_cast<int64_t>(strides[i]);
            rt_params.blocks[i] = blocks[i];
        }
        for (size_t i = num_terms; i < MAX_INDEXING_TERMS; i++) {
            rt_params.sizes[i] = 1;
            rt_params.strides[i] = 1;
            rt_params.blocks[i] = 1;
        }
    };
    dispatch_gws_rt_params_t get() const { return rt_params; };

    std::string str() const {
        std::stringstream ss;
        ss << "<dispatch_runtime_params_t (size/stride/block): ";
        for (size_t i = 0; i < num_terms; i++) {
            ss << rt_params.sizes[i] << "/" << rt_params.strides[i] << "/"
               << rt_params.blocks[i] << ", ";
        }
        ss << ">";
        return ss.str();
    }

    nd_range_t nd_range;

private:
    size_t num_terms;
    dispatch_gws_rt_params_t rt_params;
};

struct named_dim_t {
public:
    named_dim_t(const char *name, size_t idx) : name(name), idx(idx) {};

    const char *name;
    size_t idx;
};

struct gws_mapped_block_t : public gpu::block_t {
    gws_mapped_block_t() = default;
    gws_mapped_block_t(const block_t &block,
            const std::vector<named_dim_t> &indexed_dims, size_t gws_idx)
        : block_t(block), is_indexed(false), gws_idx(gws_idx) {
        for (const auto &dim : indexed_dims) {
            if (dim.idx == static_cast<size_t>(dim_idx)) { is_indexed = true; }
        }
    }

    std::string str() const {
        std::ostringstream ss;
        ss << static_cast<const block_t *>(this)->str().c_str();
        ss << ", indexed=" << is_indexed;
        ss << " / gws_idx=" << gws_idx;
        return ss.str();
    }

    bool is_indexed;
    size_t gws_idx;
    size_t mapped_to_idx;
};

using gws_layout_t = std::vector<gws_mapped_block_t>;
using work_size = std::array<size_t, GWS_MAX_NDIMS>;

struct lws_strategy_t {
    lws_strategy_t(const compute_engine_t *engine,
            const gpu_primitive_attr_t *gpu_attr)
        : engine(engine), gpu_attr(gpu_attr) {};
    virtual ~lws_strategy_t() = default;

    virtual work_size create_lws(const work_size &gws) const = 0;

    size_t get_max_wg_size() const {
        bool large_grf_mode = gpu_attr && gpu_attr->threads_per_eu() == 4;
        return engine->device_info()->max_wg_size(large_grf_mode);
    }

protected:
    const compute_engine_t *engine;
    const gpu_primitive_attr_t *gpu_attr;
};

// Balance lws size with occupation
struct default_lws_strategy_t : public lws_strategy_t {
    default_lws_strategy_t(const compute_engine_t *engine,
            const gpu_primitive_attr_t *gpu_attr)
        : lws_strategy_t(engine, gpu_attr) {};
    work_size create_lws(const work_size &gws) const override {
        work_size lws;
        get_optimal_lws(gws.data(), lws.data(), gws.size(), -1,
                engine->device_info()->gpu_arch());
        return lws;
    }
};

struct gws_mapped_layout_t {
    gws_mapped_layout_t(const lws_strategy_t &lws_strategy)
        : lws_strategy(lws_strategy) {
        gws_size.fill(1);
    }

    void add_block(size_t size, dim_t dim_idx,
            const std::vector<named_dim_t> &indexed_dims, size_t gws_idx) {
        gpu_assert(gws_idx < GWS_MAX_NDIMS);
        block_idxs[gws_idx].emplace_back(blocks.size());
        blocks.emplace_back(block_t(dim_idx, static_cast<dim_t>(size),
                                    static_cast<dim_t>(gws_size[gws_idx])),
                indexed_dims, gws_idx);
        gws_size[gws_idx] *= size;
    }

    size_t get_num_blocks(size_t gws_idx) const {
        return block_idxs[gws_idx].size();
    }

    const gws_mapped_block_t &get_block(
            size_t gws_idx, size_t block_idx) const {
        return blocks[block_idxs[gws_idx][block_idx]];
    }

    compute::nd_range_t nd_range() const {
        work_size lws = lws_strategy.create_lws(gws_size);
        return compute::nd_range_t(gws_size.data(), lws.data());
    }

    inline const std::vector<gws_mapped_block_t> &get_blocks() const {
        return blocks;
    }

private:
    std::array<std::vector<size_t>, GWS_MAX_NDIMS> block_idxs;
    std::vector<gws_mapped_block_t> blocks;
    work_size gws_size;
    const lws_strategy_t &lws_strategy;
};

struct dim_id_t {
    dim_id_t() = default;
    constexpr dim_id_t(size_t id) : id(id) {};
    size_t id;

    bool operator==(const dim_id_t &other) const { return id == other.id; }
    bool operator==(size_t other) const { return id == other; }
    operator size_t() const { return id; }
};

struct dim_id_hash_t {
    size_t operator()(const dim_id_t &id) const noexcept { return id.id; }
};

constexpr size_t dim_not_found = std::numeric_limits<size_t>::max();

struct named_buffer_t : public memory_desc_t {
    named_buffer_t(const char *name, const memory_desc_t &md,
            std::vector<dim_id_t> dims)
        : memory_desc_t(md), name(name), dim_ids(std::move(dims)) {
        gpu_assert(this->name.size() <= MAX_BUFFER_NAME_LENGTH);
        gpu_assert(format_kind == format_kind::blocked);
    };

    // Copy the named_buffer_t, while changing the name
    named_buffer_t(const char *name, const named_buffer_t &buf)
        : memory_desc_t(buf), name(name), dim_ids(buf.get_dim_ids()) {};

    const std::string &get_name() const { return name; }
    const std::vector<dim_id_t> &get_dim_ids() const { return dim_ids; }

    void remove_dim(dim_id_t dim) {
        size_t dim_idx = get_dim_idx(dim);
        if (dim_idx == dim_not_found) return;

        remove_blocking(dim);

        auto &blk = format_desc.blocking;
        dim_t dim_stride = blk.strides[dim_idx];
        dim_t dim_size = padded_dims[dim_idx];

        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            if (blk.strides[i] > dim_stride) blk.strides[i] /= dim_size;

            // Shift dims down
            if (i > dim_idx) {
                blk.strides[i - 1] = blk.strides[i];
                dims[i - 1] = dims[i];
                padded_dims[i - 1] = padded_dims[i];
            }
        }

        // Reindex blocks to reflect the shift
        for (size_t blk_idx = 0; blk_idx < static_cast<size_t>(blk.inner_nblks);
                blk_idx++) {
            if (static_cast<size_t>(blk.inner_idxs[blk_idx]) > dim_idx)
                blk.inner_idxs[blk_idx]--;
        }

        // Remove the dimension label
        dim_ids.erase(dim_ids.begin() + static_cast<dim_t>(dim_idx));

        // Decrement the number of dimensions
        ndims--;
    }

    // Appends a block for the given dimension, of the given size.
    // Will change dimension size, strides, and block layout
    void append_block(dim_id_t dim, dim_t size) {
        auto &blk = format_desc.blocking;

        size_t dim_idx = get_dim_idx(dim);
        if (dim_idx == dim_not_found) {
            // Add a new dimension
            assert(ndims < DNNL_MAX_NDIMS - 1);
            dims[ndims] = 1;
            padded_dims[ndims] = 1;
            dim_idx = static_cast<size_t>(ndims++);
            dim_ids.emplace_back(dim);
        }

        // Add the block
        blk.inner_idxs[blk.inner_nblks] = static_cast<dim_t>(dim_idx);
        blk.inner_blks[blk.inner_nblks++] = size;

        // Update the strides
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            blk.strides[i] *= size;
        }

        // Update the dimension size
        dims[dim_idx] *= size;
        padded_dims[dim_idx] *= size;
    }

    size_t get_dim_idx(dim_id_t dim) {
        for (size_t i = 0; i < dim_ids.size(); i++) {
            if (dim_ids[i] == dim) { return i; }
        }
        return dim_not_found;
    }

    block_layout_t layout() {
        // Create the block layout and reindex to the canonical dimension indexing
        block_layout_t layout(*this);
        for (auto &block : layout) {
            // Re-index the layout according to the included dims
            block.dim_idx = static_cast<dim_t>(
                    get_dim_ids()[static_cast<size_t>(block.dim_idx)]);
        }
        return layout;
    }

private:
    std::string name;
    std::vector<dim_id_t> dim_ids;

    void remove_blocking(dim_id_t dim) {
        auto &blk = format_desc.blocking;
        size_t dim_idx = get_dim_idx(dim);
        if (dim_idx == dim_not_found) return;

        // Tally up inner blocks that will be removed
        std::vector<block_t> blocks;
        dim_t stride = 1;
        for (int i = blk.inner_nblks - 1; i >= 0; i--) {
            if (static_cast<size_t>(blk.inner_idxs[i]) == dim_idx)
                blocks.emplace_back(dim_idx, blk.inner_blks[i], stride);
            stride *= blk.inner_blks[i];
        }

        // Remove the inner blocks
        size_t num_remaining_blocks = 0;
        for (size_t i = 0; i < static_cast<size_t>(blk.inner_nblks); i++) {
            if (static_cast<size_t>(blk.inner_idxs[i]) == dim_idx) continue;

            blk.inner_idxs[num_remaining_blocks] = blk.inner_idxs[i];
            blk.inner_blks[num_remaining_blocks++] = blk.inner_blks[i];
        }
        blk.inner_nblks = static_cast<int>(num_remaining_blocks);

        // Update strides
        dim_t outer_stride = blk.strides[dim_idx];
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            if (blk.strides[i] > outer_stride) continue;
            for (const auto &block : blocks) {
                if (blk.strides[i] >= block.stride)
                    blk.strides[i] /= block.block;
            }
        }
    }
};

// Approach: default to combining dims to simplify indexing, unless the user requests to be able
// to access a dim's index individually. Dims can be combined if they form a dense block,
// and this is done by re-indexing the dims followed by a normalization step
class reusable_dispatch_t {
public:
    reusable_dispatch_t() = default;
    reusable_dispatch_t(const std::vector<named_buffer_t> &buffers,
            const std::vector<named_dim_t> &indexed_dims,
            const gws_term_list_t &term_list,
            const compute::nd_range_t &nd_range) {
        compile_params.num_terms = term_list.terms.size();
        for (size_t i = 0; i < term_list.terms.size(); i++) {
            compile_params.terms[i] = term_list.terms[i];
        }

        // Save buffer information
        compile_params.num_buffers = term_list.buf_idxs.size();
        for (const auto &kv : term_list.buf_idxs) {
            const size_t buf_idx = kv.first;
            const auto &buf_term_idx = kv.second;

            // Copy buffer name into params
            const auto &buf_name = buffers[buf_idx].get_name();
            for (size_t i = 0; i < buf_name.size(); i++) {
                compile_params.buffer_names[buf_idx][i] = buf_name[i];
            }

            // Copy buffer terms into params
            compile_params.buffer_num_terms[buf_idx] = buf_term_idx.size();
            for (size_t j = 0; j < buf_term_idx.size(); j++) {
                compile_params.buffer_term_index[buf_idx][j] = buf_term_idx[j];
            }
        }
        compile_params.num_indexed_dims = term_list.dim_idxs.size();
        for (const auto &kv : term_list.dim_idxs) {
            const size_t dim_idx = kv.first;
            const auto &dim_term_idx = kv.second;

            const auto &dim_name = indexed_dims[dim_idx].name;

            // Copy dim name into params
            for (size_t i = 0; i < strlen(dim_name); i++) {
                compile_params.dim_names[dim_idx][i] = dim_name[i];
            }

            // Copy dim terms into params
            compile_params.dim_num_terms[dim_idx] = dim_term_idx.size();
            for (size_t j = 0; j < dim_term_idx.size(); j++) {
                compile_params.dim_term_index[dim_idx][j] = dim_term_idx[j];
            }
        }

        // Set runtime params
        runtime_params = dispatch_runtime_params_t(
                nd_range, term_list.strides, term_list.sizes, term_list.blocks);
    }

    const dispatch_compile_params_t &get_compile_params() const {
        return compile_params;
    };
    const dispatch_runtime_params_t &get_runtime_params() const {
        return runtime_params;
    };

private:
    dispatch_compile_params_t compile_params;
    dispatch_runtime_params_t runtime_params;
};

class reusable_dispatch_config_t {
public:
    reusable_dispatch_config_t(std::vector<dim_id_t> dims)
        : dispatched_dims(std::move(dims)) {};
    status_t generate(
            reusable_dispatch_t &dispatch, const lws_strategy_t &lws_strategy);
    status_t register_buffer(named_buffer_t &buffer);
    status_t define_dim_index(const char *dim_name, size_t dim_idx);

private:
    size_t get_dim_idx(size_t dim_idx);
    gws_mapped_layout_t compute_gws_blocks(
            const gws_layout_t &layout, const lws_strategy_t &lws_strategy);
    status_t compute_gws_mapping(
            gws_layout_t &layout, const gws_mapped_layout_t &gws_blocks);
    void compute_buffer_terms(const gws_layout_t &layout, size_t buffer_idx,
            const gws_mapped_layout_t &gws_blocks);
    void compute_dim_terms(const named_dim_t &dim, size_t dim_idx,
            const gws_mapped_layout_t &gws_blocks);
    status_t subdivide_to_match(
            block_layout_t &layoutA, block_layout_t &layoutB);

    std::vector<named_buffer_t> buffers;
    std::vector<named_dim_t> indexed_dims;

    std::vector<dim_id_t> dispatched_dims;
    std::unordered_map<dim_id_t, dim_t, dim_id_hash_t> dim_sizes;

    gws_term_list_t term_list;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
