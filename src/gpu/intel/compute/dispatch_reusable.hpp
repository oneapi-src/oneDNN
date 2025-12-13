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

#ifndef GPU_INTEL_COMPUTE_DISPATCH_REUSABLE_HPP
#define GPU_INTEL_COMPUTE_DISPATCH_REUSABLE_HPP

#include <string>
#include <vector>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/serialization.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/block_manipulation.hpp"
#include "gpu/intel/compute/dispatch.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/compute/types_interop.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

// How many buffers can be registered simultaneously
#define MAX_REGISTERED_BUFFERS 16
#define MAX_EXPR_TERMS 128

enum class gws_op_t : uint8_t {
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
            : op(op), gws_idx(into<uint8_t>(gws_idx)) {};

        bool operator==(const compile_params_t &other) const {
            return op == other.op && gws_idx == other.gws_idx;
        }

        std::string str() const {
            stringstream_t ss;
            ss << "<compile_params_t op=" << to_string(op)
               << ", gws_idx=" << int(gws_idx) << ">";
            return ss.str();
        }

        gws_op_t op = {};
        uint8_t gws_idx = {};
    };

    struct runtime_params_t {
        runtime_params_t() = default;
        runtime_params_t(dim_t size, stride_t stride, dim_t block)
            : size(size), stride(stride), block(block) {};

        bool operator==(const runtime_params_t &other) const {
            return size == other.size && stride == other.stride
                    && block == other.block;
        }
        dim_t size;
        stride_t stride;
        dim_t block;
    };

    gws_indexing_term_t() = default;

    bool operator==(const gws_indexing_term_t &other) const {
        return compile_params_ == other.compile_params_
                && runtime_params_ == other.runtime_params_;
    }
    gws_indexing_term_t(gws_op_t op, size_t gws_idx, dim_t size,
            stride_t stride, dim_t block)
        : compile_params_(op, gws_idx), runtime_params_(size, stride, block) {};

    size_t get_hash() const { return serialization_stream_t::get_hash(*this); }

    std::string str() const {
        stringstream_t ss;
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
    uint8_t pad[6] = {};
    runtime_params_t runtime_params_;
};

struct gws_term_list_t {
    size_t append(const gws_indexing_term_t &term) {
        size_t idx = add_term(term);
        return idx;
    }

    const gws_indexing_term_t &operator[](size_t idx) const {
        return terms[idx];
    }
    size_t size() const { return terms.size(); }

    size_t get_hash() const { return serialization_stream_t::get_hash(terms); }

    bool operator==(const gws_term_list_t &other) const {
        return terms == other.terms;
    }

    std::vector<gws_indexing_term_t> terms;

    std::string str() const {
        ostringstream_t ss;
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

enum class name_id_t : uint64_t {
    // Common buffer names
    src = 1,
    wei = uint64_t(1) << 1,
    dst = uint64_t(1) << 2,
    diff_src = uint64_t(1) << 3,
    diff_wei = uint64_t(1) << 4,
    diff_dst = uint64_t(1) << 5,
    stat = uint64_t(1) << 6,

    // Common dimension names
    a = uint64_t(1) << 12,
    b = uint64_t(1) << 13,
    c = uint64_t(1) << 14,
    d = uint64_t(1) << 15,
    e = uint64_t(1) << 16,
    f = uint64_t(1) << 17,

    // Implementation Specific Names
    original = uint64_t(1) << 24,
    ss = uint64_t(1) << 25,
    buffer = uint64_t(1) << 26,
    atomic = uint64_t(1) << 27,
    local = uint64_t(1) << 28,
    ic_dim = uint64_t(1) << 29,

};

inline const std::string &to_string(name_id_t value) {
    static const std::unordered_map<name_id_t, std::string> names {
            {name_id_t::src, "SRC"},
            {name_id_t::wei, "WEI"},
            {name_id_t::dst, "DST"},
            {name_id_t::diff_src, "DIFF_SRC"},
            {name_id_t::diff_wei, "DIFF_WEI"},
            {name_id_t::diff_dst, "DIFF_DST"},
            {name_id_t::a, "A"},
            {name_id_t::b, "B"},
            {name_id_t::c, "C"},
            {name_id_t::d, "D"},
            {name_id_t::e, "E"},
            {name_id_t::f, "F"},
            {name_id_t::stat, "STAT"},
            {name_id_t::original, "ORIGINAL"},
            {name_id_t::ss, "SS"},
            {name_id_t::buffer, "BUFFER"},
            {name_id_t::atomic, "ATOMIC"},
            {name_id_t::local, "LOCAL"},
            {name_id_t::ic_dim, "IC_DIM"},
    };
    auto ret = names.find(value);
    gpu_assert(ret != names.end());
    return ret->second;
}

GPU_DEFINE_BIT_MASK_ENUM_OPS(name_id_t);

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
    std::string str() const;

    subgroup_data_t subgroup;
    bool use_int32_offset = false;
    bool require_stateless_addressing = true;
    uint8_t padding[6] = {0};

    name_id_t buffer_set = {}; // Bitset representing supported buffers

    // Offset into exprs where buffer offset expression is stored
    uint8_t buffer_off_expr[MAX_REGISTERED_BUFFERS] = {};
    uint8_t exprs[MAX_EXPR_TERMS] = {};
    data_type_t buffer_types[MAX_REGISTERED_BUFFERS] = {data_type::undef};
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(dispatch_compile_params_t);

class dispatch_runtime_params_t {
public:
    dispatch_runtime_params_t() = default;
    dispatch_runtime_params_t(const nd_range_t &nd_range, bool use_int32_offset,
            const std::vector<int64_t> &term_list)
        : nd_range(nd_range)
        , use_int32_offset(use_int32_offset)
        , num_terms(term_list.size()) {
        for (size_t i = 0; i < num_terms; i++) {
            if (use_int32_offset) {
                rt32.params[i]
                        = into<int32_t>(static_cast<int64_t>(term_list[i]));
            } else {
                rt64.params[i] = term_list[i];
            }
        }
        for (size_t i = num_terms; i < MAX_RUNTIME_TERMS; i++) {
            if (use_int32_offset)
                rt32.params[i] = 0;
            else
                rt64.params[i] = 0;
        }
    }
    dispatch_gws_rt_params64_t get64() const {
        gpu_assert(!use_int32_offset);
        return rt64;
    }
    dispatch_gws_rt_params32_t get32() const {
        gpu_assert(use_int32_offset);
        return rt32;
    }

    std::string str() const {
        stringstream_t ss;
        ss << "<dispatch_runtime_params_t: ";
        for (size_t i = 0; i < num_terms; i++) {
            if (use_int32_offset) {
                if (i > 0) ss << ", ";
                ss << "term[" << std::to_string(i) << "]: " << rt32.params[i];
            } else {
                if (i > 0) ss << ", ";
                ss << "term[" << std::to_string(i) << "]: " << rt64.params[i];
            }
        }
        ss << ">";
        return ss.str();
    }

    nd_range_t nd_range;
    bool use_int32_offset = false;

private:
    size_t num_terms = 0;
    union {
        dispatch_gws_rt_params64_t rt64;
        dispatch_gws_rt_params32_t rt32;
    };
};

inline void set_rt_params(compute::kernel_arg_list_t &arg_list, int index,
        const dispatch_runtime_params_t &params) {
    if (params.use_int32_offset)
        arg_list.set(index, params.get32());
    else
        arg_list.set(index, params.get64());
}

inline void append_rt_params(compute::kernel_arg_list_t &arg_list,
        const dispatch_runtime_params_t &params) {
    if (params.use_int32_offset)
        arg_list.append(params.get32());
    else
        arg_list.append(params.get64());
}

struct named_dim_t {
public:
    named_dim_t(const char *name, size_t idx) : name(name), idx(idx) {};

    const char *name;
    size_t idx;
};

class gws_bin_mapping_t;

struct lws_strategy_t {
    lws_strategy_t(const engine_t *engine, const gpu_primitive_attr_t *gpu_attr)
        : engine(engine), gpu_attr(gpu_attr) {};
    virtual ~lws_strategy_t() = default;

    virtual range_t create_lws(
            range_t &gws, const gws_bin_mapping_t &mapper) const
            = 0;

    // Determine if a given block (mapped to each buffer) should be in the lws.
    // Gets called for each block dispatched to the GWS.
    // XXX: If a subgroup is used, its block must be added to the lws. It will not get
    // dispatched to this function, and will always be included.
    virtual bool is_included(const mapped_block_t &blocks) const = 0;

    size_t get_max_wg_size() const {
        int grf_per_thread = gpu_attr ? gpu_attr->grf_per_thread() : 128;
        return engine->device_info()->max_wg_size(grf_per_thread);
    }

protected:
    const engine_t *engine;
    const gpu_primitive_attr_t *gpu_attr;
};

// Balance lws size with occupation
struct default_lws_strategy_t : public lws_strategy_t {
    default_lws_strategy_t(
            const engine_t *engine, const gpu_primitive_attr_t *gpu_attr)
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
    named_buffer_t(name_id_t name_id, const memory_desc_t &md,
            const std::vector<dim_idx_t> &dims)
        : memory_desc_t(md), name_id(name_id), dim_ids(dims) {
        gpu_assert(format_kind == format_kind::blocked);
        gpu_assert(static_cast<size_t>(md.ndims) <= dim_ids.size());
    }
    named_buffer_t(name_id_t name_id) : name_id(name_id) {
        format_kind = format_kind::blocked;
    }
    named_buffer_t(name_id_t name_id, const memory_desc_t &md)
        : named_buffer_t(name_id, md, default_dims(md.ndims)) {};

    // Copy the named_buffer_t, while changing the name
    named_buffer_t(name_id_t name_id, const named_buffer_t &buf)
        : memory_desc_t(buf), name_id(name_id), dim_ids(buf.get_dim_ids()) {};

    static std::vector<dim_idx_t> default_dims(int ndims) {
        std::vector<dim_idx_t> dims(ndims);
        for (int i = 0; i < ndims; i++)
            dims[i] = i;
        return dims;
    }
    dim_t nelems(bool with_padding = false) const {
        return memory_desc_wrapper(static_cast<memory_desc_t>(*this))
                .nelems(with_padding);
    }

    uint64_t size(int index = 0, bool include_additional_size = true,
            bool include_offset0 = false) const {
        return memory_desc_wrapper(static_cast<memory_desc_t>(*this))
                .size(index, include_additional_size, include_offset0);
    }

    const std::string &name() const { return to_string(get_name_id()); }
    const name_id_t &get_name_id() const { return name_id; }
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
    name_id_t name_id;
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
                if (stride_t(blk.strides[i]) >= block.stride)
                    blk.strides[i] /= block.block;
            }
        }
    }
};

class reusable_dispatch_t {
public:
    reusable_dispatch_t() = default;
    reusable_dispatch_t(const compute::nd_range_t &nd_range,
            subgroup_data_t subgroup,
            const std::vector<named_buffer_t> &buffers,
            const std::vector<uint8_t> &buffer_off_exprs,
            const std::vector<uint8_t> &expr_data,
            const std::vector<int64_t> &term_list) {

        for (size_t i = 0; i < expr_data.size(); i++)
            compile_params.exprs[i] = expr_data[i];
        for (size_t i = 0; i < buffer_off_exprs.size(); i++)
            compile_params.buffer_off_expr[i] = buffer_off_exprs[i];

        auto &buffer_set = compile_params.buffer_set;
        for (size_t i = 0; i < buffers.size(); i++)
            buffer_set = buffer_set | buffers[i].get_name_id();

        int term_idx = 0;
        dim_t max_buffer_size = 0;
        for (auto &buffer : buffers) {
            compile_params.buffer_types[term_idx] = buffer.data_type;
            max_buffer_size = std::max(max_buffer_size, buffer.nelems(true));
            term_idx++;
        }

        compile_params.use_int32_offset = max_buffer_size <= INT32_MAX;
        compile_params.subgroup = subgroup;

        // Set runtime params
        runtime_params = dispatch_runtime_params_t(
                nd_range, compile_params.use_int32_offset, term_list);
    }

    const dispatch_compile_params_t &get_compile_params() const {
        return compile_params;
    }
    const dispatch_runtime_params_t &get_runtime_params() const {
        return runtime_params;
    }

private:
    dispatch_compile_params_t compile_params;
    dispatch_runtime_params_t runtime_params;
};
// TODO: Add a strategy pattern for this, in case the mapping
// leads to performance degradation
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
            const engine_t *engine, std::vector<dim_idx_t> dims)
        : dispatched_dims(std::move(dims)), engine(engine) {};
    status_t generate(
            reusable_dispatch_t &dispatch, const lws_strategy_t &lws_strategy);
    status_t register_buffer(const named_buffer_t &buffer);
    status_t define_dim_index(name_id_t dim_name, dim_idx_t dim_id, dim_t size);
    status_t use_subgroup(name_id_t buf_name, size_t size);

private:
    std::vector<named_buffer_t> buffers;
    std::vector<dim_idx_t> dispatched_dims;
    std::unordered_map<dim_idx_t, dim_t, dim_id_hash_t> dim_sizes;

    subgroup_data_t subgroup;
    const engine_t *engine;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::impl::gpu::intel::compute::gws_term_list_t> {
    size_t operator()(
            const dnnl::impl::gpu::intel::compute::gws_term_list_t &list)
            const {
        return list.get_hash();
    }
};
template <>
struct hash<dnnl::impl::gpu::intel::compute::gws_indexing_term_t> {
    size_t operator()(
            const dnnl::impl::gpu::intel::compute::gws_indexing_term_t &term)
            const {
        return term.get_hash();
    }
};

} // namespace std

#endif
