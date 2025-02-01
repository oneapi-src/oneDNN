/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_IR_TENSOR_HPP
#define GPU_INTEL_JIT_V2_IR_TENSOR_HPP

#include "common/math_utils.hpp"
#include "gpu/intel/jit/ir/linear_expr.hpp"
#include "gpu/intel/jit/ir/problem.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/jit/v2/ir/reqs.hpp"

#include <cstring>
#include <functional>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {

// Stores upper bounds for variables.
class var_range_info_t {
public:
    void set_bound(const expr_t &var, int bound) {
        for (auto &e : entries_) {
            if (e.var.is_equal(var)) {
                e.bound = std::min(e.bound, bound);
                return;
            }
        }
        entries_.emplace_back(entry_t {var, bound});
    }

    int bound(const expr_t &var) const {
        for (auto &e : entries_) {
            if (e.var.is_equal(var)) return e.bound;
        }
        return default_bound;
    }

private:
    static const int default_bound = std::numeric_limits<int>::max();

    struct entry_t {
        expr_t var;
        int bound = default_bound;

        entry_t() = default;
        entry_t(const expr_t &var, int bound) : var(var), bound(bound) {}
    };

private:
    std::vector<entry_t> entries_;
};

struct block_t {
    block_t() = default;
    block_t(const pvar_t &dim, const expr_t &size,
            const expr_t &stride = expr_t())
        : dim(dim), size(size), stride(stride) {}

    bool is_empty() const { return dim.is_undef(); }
    bool has_const_size() const { return size.is<int_imm_t>(); }
    bool has_const_stride() const { return stride.is<int_imm_t>(); }
    int int_size() const { return to_int(size); }
    int int_stride() const { return to_int(stride); }

    bool has_same_size(const expr_t &other_size) const {
        auto *imm = other_size.as_ptr<int_imm_t>();
        if (imm && has_const_size()) return int_size() == (int)imm->value;
        return size.is_same(other_size);
    }

    bool has_same_size(const block_t &other) const {
        return has_same_size(other.size);
    }

    bool has_same_stride(const expr_t &other_stride) const {
        auto *imm = other_stride.as_ptr<int_imm_t>();
        if (imm && has_const_stride()) return int_stride() == (int)imm->value;
        return stride.is_same(other_stride);
    }

    bool has_same_stride(const block_t &other) const {
        return has_same_stride(other.stride);
    }

    bool operator==(const block_t &other) const {
        return (dim == other.dim) && (size.is_equal(other.size))
                && (stride.is_equal(other.stride));
    }
    bool operator!=(const block_t &other) const { return !operator==(other); }
    std::string brief_str() const;
    std::string str() const;
    IR_DEFINE_DUMP()

    pvar_t dim;
    expr_t size;
    expr_t stride;
};

class layout_desc_t {
public:
    layout_desc_t() = default;
    layout_desc_t(const pvar_map_t<char> &letter_map);
    char layout_letter(const pvar_t &dim) const;
    const std::string &canonical() const { return canonical_; }
    const pvar_map_t<char> &letter_map() const { return letter_map_; }
    int ndims() const { return letter_map_.size(); }
    pvar_t prb_dim(int idx) const;
    int dim_index(const pvar_t &dim) const;
    std::string to_abx_tag(const std::string &tag) const;

    template <typename T>
    T filter_dim_map(const T &map) const {
        T ret;
        for (auto &d : map) {
            if (!letter_map_.has(d)) continue;
            ret[d] = map[d];
        }
        return ret;
    }

    bool operator==(const layout_desc_t &other) const {
        return letter_map_ == other.letter_map_;
    }

    bool operator!=(const layout_desc_t &other) const {
        return !operator==(other);
    }

    std::string str() const;
    IR_DEFINE_DUMP()

private:
    pvar_map_t<char> letter_map_;
    std::string canonical_;
};

class dim_mapper_t {
public:
    void set_dim(const pvar_t &dim, const expr_t &expr = expr_t(),
            bool has_undeflow = false);
    void set_layout_desc(const layout_desc_t &desc) { layout_desc_ = desc; }
    bool is_empty() const { return map_.is_empty(); }
    bool has(const pvar_t &dim) const { return map_.has(dim); }
    const expr_t &expr(const pvar_t &dim) const;
    bool has_underflow(const pvar_t &dim) const;
    const layout_desc_t &layout_desc() const { return layout_desc_; }
    std::string str() const;
    IR_DEFINE_DUMP()

private:
    struct map_data_t {
        std::string str() const {
            std::ostringstream oss;
            oss << expr;
            if (has_underflow) oss << " (has_underflow)";
            return oss.str();
        }

        expr_t expr;
        bool has_underflow;
    };

    pvar_map_t<map_data_t> map_;
    layout_desc_t layout_desc_;
};

struct layout_raw_tag_entry_t {
    char letter = '?';
    int block = 0;
    bool is_blocked = false;

    layout_raw_tag_entry_t() = default;
    layout_raw_tag_entry_t(char letter, int block, bool is_blocked)
        : letter(letter), block(block), is_blocked(is_blocked) {}

    dim_idx_t index() const {
        gpu_assert(letter >= 'a' && letter < 'x');
        return letter - 'a';
    }
    bool is_outer() const { return !is_blocked || (is_blocked && block == 0); }
    bool is_x() const { return letter == 'x'; }

    std::string str() const {
        std::ostringstream oss;
        if (block != 0) oss << block;
        oss << std::string(1,
                (is_blocked && block == 0
                                ? static_cast<char>(std::toupper(letter))
                                : letter));
        return oss.str();
    }

    IR_DEFINE_DUMP()

    bool operator==(const layout_raw_tag_entry_t &other) const {
        return (letter == other.letter) && (block == other.block)
                && (is_blocked == other.is_blocked);
    }

    bool operator!=(const layout_raw_tag_entry_t &other) const {
        return !operator==(other);
    }
};

class layout_raw_tag_t {
public:
    static layout_raw_tag_t any() { return layout_raw_tag_t("any"); }

    layout_raw_tag_t() = default;
    explicit layout_raw_tag_t(const std::string &tag, dim_idx_t ndims = 0)
        : is_any_(tag == "any"), entries_(to_entries(tag)) {
        expand_x(ndims);
    }

    bool is_empty() const { return !is_any_ && entries_.empty(); }
    bool is_any() const { return is_any_; }
    const std::vector<layout_raw_tag_entry_t> &entries() const {
        return entries_;
    }
    int nentries() const { return (int)entries_.size(); }
    void add_entry(char letter, int block, bool is_blocked);
    int entry_index(char letter);
    void add_dim(char letter, int pos);
    void remove_dim(char letter);
    bool is_blocked(char letter) const;
    dim_idx_t ndims() const;
    dim_idx_t non_x_ndims() const;
    std::string str() const;
    IR_DEFINE_DUMP()

    bool matches(const layout_raw_tag_t &other, const layout_desc_t &desc,
            const pvar_tile_t &sizes) const;

    bool operator==(const layout_raw_tag_t &other) const {
        return (is_any_ == other.is_any_) && (entries_ == other.entries_);
    }

    bool operator!=(const layout_raw_tag_t &other) const {
        return !operator==(other);
    }

    void stringify(std::ostream &out) const { out << str(); }

    void parse(std::istream &in) {
        auto tag = jit::parse<std::string>(in);
        is_any_ = (tag == "any");
        entries_ = to_entries(tag);
    }

private:
    void init_entries(const std::string &s);
    bool has_x() const;
    void expand_x(dim_idx_t ndims);
    std::vector<bool> skip_mask(
            const layout_desc_t &desc, const pvar_tile_t &sizes) const;
    static std::vector<std::pair<char, int>> parse_letter_blocks(
            const std::string &tag);
    static std::vector<layout_raw_tag_entry_t> to_entries(
            const std::string &tag);

    bool is_any_ = false;
    std::vector<layout_raw_tag_entry_t> entries_;
};

class layout_tag_t {
public:
    layout_tag_t() = default;

    layout_tag_t(const layout_desc_t &desc, const type_t &type,
            const layout_raw_tag_t &raw_tag)
        : desc_(desc), type_(type), raw_tag_(raw_tag) {}
    layout_tag_t(const type_t &type, const std::string &str_tag)
        : layout_tag_t({}, type, layout_raw_tag_t(str_tag)) {}
    layout_tag_t(const layout_desc_t &desc, const type_t &type,
            const std::string &str_tag)
        : layout_tag_t(desc, type, layout_raw_tag_t(str_tag)) {}

    bool is_empty() const { return raw_tag_.is_empty(); }
    bool is_any() const { return raw_tag_.is_any(); }
    const layout_desc_t &desc() const { return desc_; }
    const type_t &type() const { return type_; }
    const layout_raw_tag_t &raw_tag() const { return raw_tag_; }
    bool matches(const layout_tag_t &other, const pvar_tile_t &sizes,
            bool check_type = true) const;
    layout_tag_t with_type(const type_t &new_type) const {
        return layout_tag_t(desc_, new_type, raw_tag_);
    }
    std::string str() const;
    IR_DEFINE_DUMP()

    bool operator==(const layout_tag_t &other) const {
        return (desc_ == other.desc_) && (type_ == other.type_)
                && (raw_tag_ == other.raw_tag_);
    }

    bool operator!=(const layout_tag_t &other) const {
        return !operator==(other);
    }

    void stringify(std::ostream &out) const {
        jit::stringify(out, raw_tag_);
        out << ":";
        jit::stringify(out, type_);
    }

    void parse(std::istream &in) {
        desc_ = layout_desc_t();
        auto s = stream_parse<std::string>(in);
        auto parts = gpu_utils::split(s, ":");
        gpu_assert(parts.size() == 2);
        jit::parse(parts[0], raw_tag_);
        jit::parse(parts[1], type_);
    }

private:
    layout_desc_t desc_;
    type_t type_;
    layout_raw_tag_t raw_tag_;
};

class layout_t {
public:
    layout_t() = default;
    layout_t(const layout_desc_t &desc, const type_t &type)
        : desc_(desc), type_(type), base_(0) {}
    layout_t(const layout_desc_t &desc, const type_t &type, const expr_t &base,
            const std::vector<block_t> &blocks)
        : desc_(desc), type_(type), base_(base), blocks_(blocks) {}

    bool is_empty() const { return type_.is_undef(); }
    bool is_scalar() const { return elems() == 1; }
    const layout_desc_t &desc() const { return desc_; }
    const type_t &type() const { return type_; }
    const expr_t &base() const { return base_; }
    void set_base(const expr_t &base) { base_ = base; }
    const std::vector<block_t> &blocks() const { return blocks_; }
    std::vector<pvar_t> dims() const {
        // TODO: Change to bool.
        pvar_map_t<int> seen;
        for (auto &b : blocks_)
            seen[b.dim] = 1;
        return seen.keys();
    }
    bool operator==(const layout_t &other) const {
        if (desc_ != other.desc_) return false;
        if (type_ != other.type_) return false;
        if (!base_.is_equal(other.base_)) return false;
        if (blocks_ != other.blocks_) return false;
        return true;
    }
    bool operator!=(const layout_t &other) const { return !operator==(other); }
    int elems() const;
    // Storage size in bytes.
    int size() const;
    int nblocks() const { return static_cast<int>(blocks().size()); }
    int nblocks(const pvar_t &dim) const;
    int int_base_in_bytes() const { return to_int(base_) * type_.size(); }
    int int_dim_size(const pvar_t &dim) const;
    bool has_zero_base() const { return is_zero(base_); }
    bool has_const_sizes() const;
    bool has_const_strides() const;
    pvar_tile_t int_dim_sizes() const;
    pvar_map_t<expr_t> dim_sizes() const;
    int inner_block(const pvar_t &dim, bool with_outer = true) const;
    int inner_stride() const;
    expr_t stride(const pvar_t &dim, int dim_block_idx = 0) const;
    expr_t shift_in_bytes(const std::vector<int> &block_off) const;
    dim_t offset_in_bytes(pvar_coord_t<dim_t> coord) const;
    bool is_blocked_by(const pvar_t &dim, int block) const;
    bool is_blocked_by(const layout_t &other) const;
    void add_block(const pvar_t &dim, const expr_t &size,
            const expr_t &_stride = expr_t());
    void remove(const pvar_t &dim);
    void block_by(const std::vector<block_t> &blocks);
    void pad(int elems) { stride_pad_ = elems; }
    void pad_bytes(int bytes) { pad(ir_utils::safe_div(bytes, type().size())); }
    void normalize();
    layout_t split_block(
            const block_t *block_ptr, dim_t inner, dim_t outer) const;

    template <typename T>
    layout_t map(const dim_mapper_t &dim_mapper, const pvar_coord_t<T> &coord,
            const pvar_tile_t &tile,
            const var_range_info_t &var_range_info = {}) const;

    template <typename T>
    layout_t map(const pvar_coord_t<T> &coord, const pvar_tile_t &tile) const {
        return map(dim_mapper_t(), coord, tile);
    }

    template <typename T = int>
    layout_t map(const pvar_tile_t &tile) const {
        dim_mapper_t mapper;
        mapper.set_layout_desc(desc_);
        return map(mapper, pvar_coord_t<T>(), tile);
    }

    layout_t make_dense() const;
    layout_t retype(const type_t &new_type, bool dense = false) const;
    pvar_coord_t<dim_t> to_coord(const std::vector<int> &block_idx) const;
    int to_linear_index(
            const pvar_tile_t &tile, const pvar_coord_t<dim_t> &coord) const;
    std::string blocks_str() const;
    std::string str() const;
    std::string str_with_size(const hw_t &hw) const;

    IR_DEFINE_DUMP()

private:
    layout_desc_t desc_;
    type_t type_;
    // Base offset in in elements of the layout type.
    expr_t base_;
    std::vector<block_t> blocks_;
    // All added blocks are to be aligned to this value in elements. Also see
    // add_block().
    int stride_pad_ = 1;
};

void for_each(const pvar_tile_t &base_tile, pvar_tile_t tile,
        const std::function<void(const pvar_coord_t<dim_t> &)> &func);

class block_iterator_t {
public:
    block_iterator_t() = default;
    block_iterator_t(const layout_t &layout, bool set_to_end = false);
    const layout_t &parent() const { return *parent_; }
    bool is_end() const { return block_idx_ == parent_->nblocks(); }
    bool has_next() const {
        return !is_end() && (!is_last_block() || next_factor() != -1);
    }

    block_iterator_t &operator++();

    block_iterator_t operator+(int inc) const {
        auto ret = *this;
        for (int i = 0; i < inc; i++)
            ++ret;
        return ret;
    }

    bool operator==(const block_iterator_t &other) const {
        if (is_end() || other.is_end()) return is_end() == other.is_end();
        return (parent_ == other.parent_) && block_idx_ == other.block_idx_
                && block_ == other.block_;
    }

    bool operator!=(const block_iterator_t &other) const {
        return !operator==(other);
    }

    bool is_compatible(const block_iterator_t &other) const {
        return parent_ == other.parent_;
    }

    const block_t &operator*() const {
        gpu_assert(!is_end());
        return block_;
    }

    int block_index() const { return block_idx_; }
    block_t remaining_block() const;
    bool is_dense(const prover_t &prover = prover_t::instance()) const;
    int elems(const pvar_t &dim = pvar_t()) const;
    layout_t sub_layout(int stride = 1) const;
    std::string str() const;

    IR_DEFINE_DUMP()

private:
    void set_to_end();
    int next_factor(bool is_first = false) const;
    bool is_last_block() const { return block_idx_ == parent_->nblocks() - 1; }

    const layout_t *parent_ = nullptr;
    // Index of the current block in parent's blocks.
    int block_idx_ = 0;
    // Current block, may be incomplete.
    block_t block_;
    // Number of inner elements (block_ is included).
    int elems_ = 1;
};

inline block_iterator_t begin(const layout_t &layout) {
    return block_iterator_t(layout);
}

inline block_iterator_t end(const layout_t &layout) {
    return block_iterator_t(layout, /*set_to_end=*/true);
}

void add_remaining_blocks(layout_t &layout, const block_iterator_t &it);

class layout_iterator_t {
public:
    layout_iterator_t() = default;
    layout_iterator_t(const layout_t &layout, bool is_end = false);
    const layout_t &parent() const { return *parent_; }
    int offset() const { return offset_; }
    const std::vector<int> &block_offset() const { return block_off_; }
    bool has_next(int elems) const { return offset_ + elems < total_elems_; }
    void next(int elems);
    int offset(const pvar_t &dim) const;
    pvar_coord_t<dim_t> coord() const;
    std::string str() const;
    IR_DEFINE_DUMP()

private:
    void set_to_end() { offset_ = total_elems_; }

    const layout_t *parent_ = nullptr;
    int total_elems_ = 0;
    int offset_ = 0;
    std::vector<int> block_off_;
};

// 0 <= a * x + b * y + c < C
class dim_mask_desc_t {
public:
    dim_mask_desc_t() = default;
    dim_mask_desc_t(const pvar_t &dim, const expr_t &expr, const expr_t &bound,
            int block, bool has_underflow);
    bool is_identity() const { return is_zero(c) && is_one(a) && y.is_empty(); }

    template <typename T>
    expr_t to_expr(const pvar_coord_t<T> &coord, bool with_const = true) const;

    dim_mask_desc_t map(const pvar_coord_t<expr_t> &coord) const;
    bool has(const pvar_t &dim) const;
    expr_t dim_stride(const pvar_t &dim) const;
    std::string str() const;
    IR_DEFINE_DUMP()

    pvar_t dim;
    expr_t bound;
    dim_t block = 0;
    bool has_underflow = false;

    expr_t base;
    expr_t a, b, c;
    expr_t x, y;
    pvar_t x_dim, y_dim;

private:
    void init_abc_xy(const expr_t &expr);
};

class mask_desc_t {
public:
    mask_desc_t() = default;
    mask_desc_t(const dim_mapper_t &dim_mapper, const layout_t &layout);
    int nmasks() const { return static_cast<int>(dim_masks_.size()); }
    const dim_mask_desc_t &operator[](int idx) const;
    dim_mask_desc_t &operator[](int idx);
    mask_desc_t map(const pvar_coord_t<expr_t> &coord) const;
    bool is_uniform(const block_iterator_t &it,
            const prover_t &prover = prover_t::instance()) const;
    std::string str() const;
    IR_DEFINE_DUMP()

private:
    std::vector<dim_mask_desc_t> dim_masks_;
};

struct plane_t {
    type_t type;
    // Width and height algorithmic dimensions.
    pvar_t w_dim, h_dim;
    // Width and height block size.
    int w = 0, h = 0;
    // Width, height, pitch of the plane.
    expr_t W, H, P;
    // Width and height layout dimensions.
    pvar_t x_dim, y_dim;
    // Width, height offsets for masks.
    expr_t x, y;
    // Height stride.
    expr_t y_stride;

    bool is_valid = false;

    plane_t() = default;
    plane_t(const layout_t &layout, const mask_desc_t &mask_desc);
    explicit operator bool() const { return is_valid; }
};

// Helper class for layout splitting across a grid.
class grid_splitter_t {
public:
    void add(const expr_t &idx, dim_t size);
    int size() const {
        int ret = 1;
        for (auto &idx : idxs_)
            ret *= idx.size;
        return ret;
    }
    bool is_empty() const;
    expr_t pop(int size);
    const object_map_t<expr_t, expr_t> &virt_grid_idxs() const {
        return virt_grid_idxs_;
    }

    const var_range_info_t &var_range_info() const { return var_range_info_; }

    std::string str() const {
        std::ostringstream oss;
        bool is_first = true;
        for (auto &kv : virt_grid_idxs_) {
            if (!is_first) oss << "\n";
            oss << kv.first << " -> " << kv.second;
            is_first = false;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    struct index_t {
        expr_t expr;
        dim_t size = 0;

        index_t(const expr_t &expr, dim_t size) : expr(expr), size(size) {}
        expr_t pop(int &n);
    };

    expr_t register_index(const expr_t &expr, int size);

    std::vector<index_t> idxs_;
    object_map_t<expr_t, expr_t> virt_grid_idxs_;
    var_range_info_t var_range_info_;
};

class view_t {
public:
    view_t() = default;
    view_t(const dim_mapper_t &dim_mapper, const layout_t &base_layout,
            const pvar_coord_t<expr_t> &coord, const pvar_tile_t &tile,
            const var_range_info_t &var_range_info = {});
    bool is_empty() const { return base_layout_.is_empty(); }
    const dim_mapper_t &dim_mapper() const { return dim_mapper_; }
    const layout_t &base_layout() const { return base_layout_; }
    const pvar_coord_t<expr_t> &coord() const { return coord_; }
    const pvar_tile_t &tile() const { return tile_; }
    const layout_t &layout() const { return layout_; }
    const mask_desc_t &mask_desc() const { return mask_desc_; }
    const plane_t &plane() const { return plane_; }
    const type_t &type() const { return layout_.type(); }
    // Transforms the view to a scattered version where elements are strided
    // by stride_bytes value. This is used to generate scattered messages
    // prefetch.
    view_t scatterize(int stride_bytes, const prover_t &prover) const;
    std::string str() const;
    IR_DEFINE_DUMP()

    static view_t split(const dim_mapper_t &dim_mapper,
            const layout_t &base_layout, const pvar_coord_t<expr_t> &coord,
            const pvar_tile_t &tile, grid_splitter_t &grid_splitter);

private:
    dim_mapper_t dim_mapper_;
    layout_t base_layout_;
    pvar_coord_t<expr_t> coord_;
    pvar_tile_t tile_;
    layout_t layout_;
    mask_desc_t mask_desc_;
    plane_t plane_;
};

} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
