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

#ifndef GPU_INTEL_COMPUTE_DISPATCH_REUSABLE_HPP
#define GPU_INTEL_COMPUTE_DISPATCH_REUSABLE_HPP

#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/block_manipulation.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/dispatch.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/ocl/types_interop.h"
#include "gpu/intel/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

// How many buffers can be registered simultaneously
#define MAX_REGISTERED_BUFFERS 4
// Maximum length of each indexed dim's name
#define MAX_DIM_NAME_LENGTH 15
// Maximum length of each buffer's name
#define MAX_BUFFER_NAME_LENGTH 7

// Ensure that we don't have padding in our structures
static_assert(MAX_REGISTERED_BUFFERS * (MAX_BUFFER_NAME_LENGTH + 1) % 4 == 0,
        "Padding will be introduced due to registered buffers.");

enum class gws_op_t : uint32_t {
    ZERO,
    SOLO,
    FIRST,
    MOD,
    SOLO_BLOCK,
    FIRST_BLOCK,
    MOD_BLOCK,
    UNDEF,
};

inline std::string to_string(gws_op_t op) {
    switch (op) {
#define CASE(x) \
    case gws_op_t::x: return #x;
        CASE(ZERO);
        CASE(SOLO);
        CASE(FIRST);
        CASE(MOD);
        CASE(SOLO_BLOCK);
        CASE(FIRST_BLOCK);
        CASE(MOD_BLOCK);
        CASE(UNDEF);
#undef CASE
    }
    return "invalid";
#undef CASE
}

// Encodes the information needed for one term like (idx / stride % max) * block
// - stride, max, and block are defined by an index into the runtime struct
// - idx is defined by an index into the gws indexing function
// - the gws_op is used to simplify the expression if stride/max/block are known
struct gws_indexing_term_t {
    struct compile_params_t {
        compile_params_t() = default;
        compile_params_t(gws_op_t op, size_t gws_idx)
            : op(op), gws_idx(gws_idx) {};

        bool operator==(const compile_params_t &other) const {
            return op == other.op && gws_idx == other.gws_idx;
        };

        std::string str() const {
            std::stringstream ss;
            ss << "<compile_params_t op=" << to_string(op)
               << ", gws_idx=" << gws_idx << ">";
            return ss.str();
        }

        gws_op_t op;
        uint8_t padding[4] = {0, 0, 0, 0};
        uint64_t gws_idx;
    };

    struct runtime_params_t {
        runtime_params_t() = default;
        runtime_params_t(dim_t size, stride_t stride, dim_t block)
            : size(size), stride(stride), block(block) {};

        bool operator==(const runtime_params_t &other) const {
            return size == other.size && stride == other.stride
                    && block == other.block;
        };
        dim_t size;
        stride_t stride;
        dim_t block;
    };

    gws_indexing_term_t() = default;

    bool operator==(const gws_indexing_term_t &other) const {
        return compile_params_ == other.compile_params_
                && runtime_params_ == other.runtime_params_;
    };
    gws_indexing_term_t(gws_op_t op, size_t gws_idx, dim_t size,
            stride_t stride, dim_t block)
        : compile_params_(op, gws_idx), runtime_params_(size, stride, block) {};

    std::string str() const {
        std::stringstream ss;
        ss << "<gws_indexing_term_t op=" << to_string(compile_params_.op)
           << ", gws_idx=" << compile_params_.gws_idx
           << ", size=" << runtime_params_.size
           << ", stride=" << runtime_params_.stride
           << ", block=" << runtime_params_.block << ">";
        return ss.str();
    }

    const compile_params_t &compile_params() const { return compile_params_; }
    const runtime_params_t &runtime_params() const { return runtime_params_; }

    compile_params_t compile_params_;
    runtime_params_t runtime_params_;
};

struct gws_term_list_t {
    size_t append(const gws_indexing_term_t &term) {
        size_t idx = add_term(term);
        gpu_assert(idx <= MAX_INDEXING_TERMS);
        return idx;
    }

    const gws_indexing_term_t &operator[](size_t idx) const {
        return terms[idx];
    }
    size_t size() const { return terms.size(); }

    std::vector<gws_indexing_term_t> terms;

    std::string str() const {
        std::ostringstream ss;
        for (size_t i = 0; i < terms.size(); i++) {
            ss << terms[i].str() << std::endl;
        }
        return ss.str();
    }

private:
    size_t add_term(const gws_indexing_term_t &term) {
        // Use an existing term if an exact match is found
        for (size_t i = 0; i < terms.size(); i++) {
            const gws_indexing_term_t &existing = terms[i];
            if (term == existing) return i;
        }

        // Create a new term
        size_t ret = terms.size();
        terms.emplace_back(term);
        return ret;
    }
};

struct subgroup_data_t {
public:
    subgroup_data_t() = default;
    subgroup_data_t(size_t buffer_idx, size_t size)
        : use_subgroup(true), buffer_idx_(buffer_idx), size_(size) {}
#if __cplusplus >= 202002L
    bool operator==(const subgroup_data_t &) const = default;
#endif

    bool used() const { return use_subgroup; }
    size_t buffer_idx() const {
        gpu_assert(use_subgroup);
        return buffer_idx_;
    }
    size_t size() const {
        gpu_assert(use_subgroup);
        return size_;
    }

protected:
    bool use_subgroup = false;
    int8_t padding[7] = {0};
    uint64_t buffer_idx_ = 0;
    uint64_t size_ = 0;
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
        for (size_t i = 0; i < static_cast<size_t>(num_terms); i++) {
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
        ss << "]>";
        return ss.str();
    }

    subgroup_data_t subgroup;
    int32_t num_terms = 0;
    bool use_int32_offset = false;
    uint8_t padding[3] = {0};
    gws_indexing_term_t::compile_params_t terms[MAX_INDEXING_TERMS]
            = {{gws_op_t::SOLO, 0}};

    // Buffer definitions (each buffer has a name, and a collection of terms
    // used to compute the offset)
    uint64_t num_buffers = 0;
    char buffer_names[MAX_REGISTERED_BUFFERS][MAX_BUFFER_NAME_LENGTH + 1]
            = {{'\0'}};
    uint64_t buffer_term_index[MAX_REGISTERED_BUFFERS][MAX_INDEXING_TERMS]
            = {{0}};
    uint64_t buffer_num_terms[MAX_REGISTERED_BUFFERS] = {0};
    data_type_t buffer_types[MAX_REGISTERED_BUFFERS] = {data_type::undef};
};
assert_trivially_serializable(dispatch_compile_params_t);

class dispatch_runtime_params_t {
public:
    dispatch_runtime_params_t() = default;
    dispatch_runtime_params_t(
            const nd_range_t &nd_range, const gws_term_list_t &terms)
        : nd_range(nd_range), num_terms(terms.size()) {
        for (size_t i = 0; i < num_terms; i++) {
            const gws_indexing_term_t::runtime_params_t &params
                    = terms[i].runtime_params();
            rt_params.sizes[i] = params.size;
            rt_params.strides[i] = static_cast<int64_t>(params.stride);
            rt_params.blocks[i] = params.block;
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
    size_t num_terms = 0;
    dispatch_gws_rt_params_t rt_params;
};

struct named_dim_t {
public:
    named_dim_t(const char *name, size_t idx) : name(name), idx(idx) {};

    const char *name;
    size_t idx;
};

class gws_bin_mapping_t;

struct lws_strategy_t {
    lws_strategy_t(const compute_engine_t *engine,
            const gpu_primitive_attr_t *gpu_attr)
        : engine(engine), gpu_attr(gpu_attr) {};
    virtual ~lws_strategy_t() = default;

    virtual range_t create_lws(
            range_t &gws, const gws_bin_mapping_t &mapper) const = 0;

    // Determine if a given block (mapped to each buffer) should be in the lws.
    // Gets called for each block dispatched to the GWS.
    // XXX: If a subgroup is used, its block must be added to the lws. It will not get
    // dispatched to this function, and will always be included.
    virtual bool is_included(const mapped_block_t &blocks) const = 0;

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
    range_t create_lws(
            range_t &gws, const gws_bin_mapping_t &mapper) const override {
        range_t lws
                = get_optimal_lws(gws, -1, engine->device_info()->gpu_arch());
        return lws;
    }

    // this strategy doesn't care which blocks are in the lws
    bool is_included(const mapped_block_t &blocks) const override {
        return false;
    }
};

struct dim_id_hash_t {
    size_t operator()(const dim_idx_t &id) const noexcept { return id; }
};

constexpr dim_idx_t dim_not_found = std::numeric_limits<dim_idx_t>::max();

struct named_buffer_t : public memory_desc_t {
    named_buffer_t(const char *name, const memory_desc_t &md,
            const std::vector<dim_idx_t> &dims)
        : memory_desc_t(md), name(name), dim_ids(dims) {
        gpu_assert(this->name.size() <= MAX_BUFFER_NAME_LENGTH);
        gpu_assert(format_kind == format_kind::blocked);
        gpu_assert(static_cast<size_t>(md.ndims) <= dim_ids.size());
    };
    named_buffer_t(const char *name) : name(name) {
        format_kind = format_kind::blocked;
    };

    // Copy the named_buffer_t, while changing the name
    named_buffer_t(const char *name, const named_buffer_t &buf)
        : memory_desc_t(buf), name(name), dim_ids(buf.get_dim_ids()) {};

    dim_t nelems(bool with_padding = false) const {
        return memory_desc_wrapper(static_cast<memory_desc_t>(*this))
                .nelems(with_padding);
    }

    const std::string &get_name() const { return name; }
    const std::vector<dim_idx_t> &get_dim_ids() const { return dim_ids; }

    void remove_dim(dim_idx_t dim, bool update_strides = true) {
        size_t dim_idx = get_dim_idx(dim);
        if (dim_idx == dim_not_found) return;

        remove_blocking(dim);

        auto &blk = format_desc.blocking;
        dim_t dim_stride = blk.strides[dim_idx];
        dim_t dim_size = padded_dims[dim_idx];

        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            if (update_strides && blk.strides[i] > dim_stride) {
                blk.strides[i] /= dim_size;
            }

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
    void append_block(dim_idx_t dim, dim_t size) {
        auto &blk = format_desc.blocking;

        size_t dim_idx = get_dim_idx(dim);
        if (dim_idx == dim_not_found) {
            // Add a new dimension
            assert(ndims < DNNL_MAX_NDIMS - 1);
            dims[ndims] = 1;
            padded_dims[ndims] = 1;
            blk.strides[ndims] = 1;
            dim_idx = static_cast<size_t>(ndims++);
            dim_ids.emplace_back(dim);
        }

        // Add the block, if it needs to be placed in front of an existing one
        if (blk.inner_nblks > 0) {
            blk.inner_idxs[blk.inner_nblks] = static_cast<dim_t>(dim_idx);
            blk.inner_blks[blk.inner_nblks++] = size;
        }

        // Update the strides
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            if (i == dim_idx) continue;
            blk.strides[i] *= size;
        }

        // Update the dimension size
        dims[dim_idx] *= size;
        padded_dims[dim_idx] *= size;
    }

    dim_idx_t get_dim_idx(dim_idx_t dim) const {
        for (dim_idx_t i = 0; i < into<dim_idx_t>(dim_ids.size()); i++) {
            if (dim_ids[i] == dim) { return i; }
        }
        return dim_not_found;
    }

    block_layout_t layout() const {
        // Create the block layout and reindex to the canonical dimension indexing
        block_layout_t layout(*this);
        for (auto &block : layout) {
            // Re-index the layout according to the included dims
            block.dim_idx = get_dim_ids()[static_cast<size_t>(block.dim_idx)];
        }
        return layout;
    }

private:
    std::string name;
    std::vector<dim_idx_t> dim_ids;

    void remove_blocking(dim_idx_t dim) {
        auto &blk = format_desc.blocking;
        dim_idx_t dim_idx = get_dim_idx(dim);
        if (dim_idx == dim_not_found) return;

        // Tally up inner blocks that will be removed
        std::vector<block_t> blocks;
        dim_t stride = 1;
        for (int i = blk.inner_nblks - 1; i >= 0; i--) {
            if (blk.inner_idxs[i] == dim_idx)
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

class reusable_dispatch_t {
public:
    reusable_dispatch_t() = default;
    reusable_dispatch_t(const std::vector<named_buffer_t> &buffers,
            const gws_term_list_t &term_list,
            const compute::nd_range_t &nd_range, subgroup_data_t subgroup,
            const std::vector<std::vector<size_t>> &buffer_term_map) {
        assert(buffers.size() == buffer_term_map.size());

        compile_params.num_terms = into<int>(term_list.terms.size());
        for (size_t i = 0; i < term_list.terms.size(); i++) {
            compile_params.terms[i] = term_list.terms[i].compile_params();
        }

        // Save buffer information
        dim_t max_buffer_size = 0;
        compile_params.num_buffers = buffers.size();
        for (size_t buf_idx = 0; buf_idx < buffers.size(); buf_idx++) {
            const named_buffer_t &buffer = buffers[buf_idx];
            // Copy buffer name into params
            const auto &buf_name = buffer.get_name();
            for (size_t i = 0; i < buf_name.size(); i++) {
                compile_params.buffer_names[buf_idx][i] = buf_name[i];
            }

            // Copy buffer terms into params
            std::vector<size_t> buf_terms = buffer_term_map[buf_idx];
            compile_params.buffer_num_terms[buf_idx] = buf_terms.size();
            for (size_t j = 0; j < buf_terms.size(); j++) {
                compile_params.buffer_term_index[buf_idx][j] = buf_terms[j];
            }

            // Save the data type
            compile_params.buffer_types[buf_idx] = buffer.data_type;

            // Check buffer sizes to see if we can use int32_t offsets
            max_buffer_size = std::max(max_buffer_size, buffer.nelems(true));
        }

        compile_params.use_int32_offset = max_buffer_size <= INT32_MAX;
        compile_params.subgroup = subgroup;

        // Set runtime params
        runtime_params = dispatch_runtime_params_t(nd_range, term_list);
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
// TODO: Add a strategy pattern for this, in case the mapping
// leads to performance degredation
class gws_bin_mapping_t {
public:
    gws_bin_mapping_t(subgroup_data_t sg) : sg(sg) {}
    void add(const block_bin_t &bin) {
        // If this bin has the subgroup block, it has to be mapped to
        // the first bin in the 0th gws dim
        if (sg.used()) {
            if (!bin.is_broadcasted(sg.buffer_idx())) {
                block_t block = bin.combined_block(sg.buffer_idx());
                if (block.stride == stride_t(1)) {
                    // Remove any existing bins in dim 0, and re-add them
                    std::vector<block_bin_t> displaced = map[0];
                    clear_(0);
                    add_(bin, 0);
                    for (const block_bin_t &old_bin : displaced) {
                        add(old_bin);
                    }
                    return;
                }
            }
        }

        // Insert into the first empty gws dim, if one exists
        for (size_t i = 0; i < map.size(); i++) {
            if (map[i].empty()) {
                add_(bin, i);
                return;
            }
        }

        // Insert into the first dim that will remove *some* divisions/modulus
        const mapped_block_t &first_new_block = bin.get_blocks().front();
        for (size_t i = 0; i < map.size(); i++) {
            block_bin_t &last = map[i].back();
            const mapped_block_t &last_old_block = last.get_blocks().back();

            if (last_old_block.can_merge(first_new_block, false)) {
                add_(bin, i);
                return;
            }
        }

        // Insert into the last dim
        add_(bin, gws_.ndims() - 1);
    }

    nd_range_t nd_range(const lws_strategy_t &lws_strategy) {
        range_t lws = lws_strategy.create_lws(gws_, *this);
        return compute::nd_range_t(gws_, lws);
    }

    const range_t &gws() const { return gws_; }

    const std::vector<block_bin_t> &get_bins(size_t idx) const {
        return map[idx];
    }

    std::vector<gws_indexing_term_t> condense_terms(size_t buffer_idx) const;

private:
    void add_(const block_bin_t &bin, size_t gws_dim) {
        map[gws_dim].emplace_back(bin);
        gws_[gws_dim] *= bin.size();
    }
    void clear_(size_t gws_idx) {
        map[gws_idx].clear();
        gws_[gws_idx] = 1;
    }
    subgroup_data_t sg;
    std::array<std::vector<block_bin_t>, range_t::max_ndims> map;
    range_t gws_ = range_t::one();
};

class reusable_dispatch_config_t {
public:
    reusable_dispatch_config_t(
            const compute_engine_t *engine, std::vector<dim_idx_t> dims)
        : dispatched_dims(std::move(dims)), engine(engine) {};
    status_t generate(
            reusable_dispatch_t &dispatch, const lws_strategy_t &lws_strategy);
    status_t register_buffer(const named_buffer_t &buffer);
    status_t define_dim_index(
            const char *dim_name, dim_idx_t dim_id, dim_t size);
    status_t use_subgroup(const std::string &buf_name, size_t size);

private:
    std::vector<named_buffer_t> buffers;
    std::vector<dim_idx_t> dispatched_dims;
    std::unordered_map<dim_idx_t, dim_t, dim_id_hash_t> dim_sizes;

    subgroup_data_t subgroup;
    const compute_engine_t *engine;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
