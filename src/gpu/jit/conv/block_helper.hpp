/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_JIT_CONV_BLOCK_HELPER_HPP
#define GPU_JIT_CONV_BLOCK_HELPER_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>
#include <initializer_list>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "common/utils.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/jit/ir/core.hpp"
#include "gpu/jit/ir/fma.hpp"
#include "gpu/jit/ir/hw_config.hpp"
#include "gpu/jit/ngen/ngen.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Tile level - describes hierarchy of dimensions for blocking.
enum class tile_level_t {
    unknown,
    iter, // Number of elements per iteration.
    tg, // Number of threads per thread group.
    loop, // Number of iterations per loop.
    _last,
};

// Min/max integer indices of tile levels.
const int min_tile_level_idx = (int)tile_level_t::iter;
const int max_tile_level_idx = (int)tile_level_t::loop;

// Describes the dimension value, contains either an integer or a special
// "unlimited" value which behaves like infinity when mixing in operations with
// integer values.
class dim_value_t {
public:
    dim_value_t() = default;
    dim_value_t(int value) : value_(value) {}
    dim_value_t &operator=(int value) {
        is_unlimited_ = false;
        value_ = value;
        return *this;
    }

    bool is_unlimited() const { return is_unlimited_; }

    bool operator==(dim_value_t other) const {
        if (is_unlimited() && other.is_unlimited()) return true;
        return (is_unlimited_ == other.is_unlimited_)
                && (value_ == other.value_);
    }
    bool operator==(int value) const {
        return !is_unlimited() && value_ == value;
    }
    bool operator!=(dim_value_t other) const { return !(*this == other); }
    bool operator!=(int value) const { return !(*this == value); }

    operator int() const {
        if (is_unlimited_) {
            ir_error_not_expected() << "Can't convert unlimited value to int.";
            return -1;
        }
        return value_;
    }

    std::string str() const {
        std::ostringstream oss;
        if (is_unlimited()) {
            oss << "(unlimited)";
        } else {
            oss << value_;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

    static dim_value_t unlimited() {
        dim_value_t ret;
        ret.is_unlimited_ = true;
        return ret;
    }

private:
    bool is_unlimited_ = false;
    int value_ = 0;
};

inline dim_value_t min(dim_value_t a, dim_value_t b) {
    if (a.is_unlimited()) return b;
    if (b.is_unlimited()) return a;
    return std::min((int)a, (int)b);
}

// Stores information about dimension blocking in context of BMNK/GEMM
// notation.
class dim_info_t {
public:
    dim_info_t() {
        for (int i = min_tile_level_idx; i <= max_tile_level_idx; i++) {
            tile_dims_[i] = 1;
            max_tile_dims_[i] = dim_value_t::unlimited();
        }
    }

    dim_info_t(const std::string &name, int size) : dim_info_t() {
        name_ = name;
        size_ = size;
    }

    const std::string &name() const { return name_; }
    void set_name(const std::string &name) { name_ = name; }

    int size() const { return size_; }
    void set_size(int value) { size_ = value; }

    int padded_size() const {
        return utils::rnd_up(size(), math::lcm(tg_blk(), pad_blk_));
    }

    char bmnk() const { return bmnk_; }
    void set_bmnk(char value) { bmnk_ = value; }

    int base_iter_block() const { return base_iter_blk_; }
    void set_base_iter_block(int value) { base_iter_blk_ = value; }

    int pad_block() const { return pad_blk_; }
    void set_pad_block(int value) { pad_blk_ = value; }

    int order_key() const { return order_key_; }
    void set_order_key(int value) { order_key_ = value; }

    bool is_blocked() const { return is_blocked_; }
    void set_blocked(bool value = true) { is_blocked_ = value; }

    bool allow_fuse() const { return allow_fuse_; }
    bool allow_split() const { return allow_split_; }
    void set_allow_fuse(bool value = true) { allow_fuse_ = value; }
    void set_allow_split(bool value = true) { allow_split_ = value; }

    int grid_dim() const {
        return ir_utils::safe_divide(padded_size(), tg_blk());
    }
    dim_value_t tg_dim() const { return dim(tile_level_t::tg); }
    dim_value_t loop_dim() const { return dim(tile_level_t::loop); }
    dim_value_t iter_dim() const { return dim(tile_level_t::iter); }
    dim_value_t dim(tile_level_t level) const {
        int idx = (int)level;
        ir_assert(idx >= min_tile_level_idx && idx <= max_tile_level_idx);
        return tile_dims_[idx];
    }
    dim_value_t max_dim(tile_level_t level) const {
        int idx = (int)level;
        ir_assert(idx >= min_tile_level_idx && idx <= max_tile_level_idx);
        return max_tile_dims_[idx];
    }
    bool pref_tg_block() { return pref_tg_block_; }
    void set_pref_tg_block(bool value = true) { pref_tg_block_ = value; }
    void set_tg_dim(dim_value_t value) { set_dim(tile_level_t::tg, value); }
    void set_loop_dim(dim_value_t value) { set_dim(tile_level_t::loop, value); }
    void set_iter_dim(dim_value_t value) { set_dim(tile_level_t::iter, value); }
    void set_dim(tile_level_t level, dim_value_t value) {
        int idx = (int)level;
        ir_assert(idx >= min_tile_level_idx && idx <= max_tile_level_idx);
        tile_dims_[idx] = value;
    }

    void set_max_dim(tile_level_t level, dim_value_t value) {
        int idx = (int)level;
        ir_assert(idx >= min_tile_level_idx && idx <= max_tile_level_idx);
        max_tile_dims_[idx] = value;
    }

    int inner_dims() const { return inner_dims_; }
    void incr_inner_dims() { inner_dims_++; }

    int iter_blk() const { return iter_dim(); }
    int loop_blk() const { return loop_dim() * iter_blk(); }
    int tg_blk() const { return tg_dim() * loop_blk(); }

    bool has_any_blocking() const {
        if (iter_dim() != 1) return true;
        if (loop_dim() != 1) return true;
        if (tg_dim() != 1) return true;
        return false;
    }

    std::string str() const {
        using namespace ir_utils;
        std::ostringstream oss;
        oss << "Dimension " << name_ << std::endl;
        oss << "  Size:              " << size_ << std::endl;
        oss << "  Base iter block:   " << base_iter_blk_ << std::endl;
        oss << "  BMNK:              " << bmnk_ << std::endl;
        oss << "  Blocked:           " << to_string(is_blocked_) << std::endl;
        oss << "  Allow fuse:        " << to_string(allow_fuse_) << std::endl;
        oss << "  Allow split:       " << to_string(allow_split_) << std::endl;
        oss << "  Order key:         " << order_key_ << std::endl;

        const char *tags[] = {"Iteration", "Thread group", "Loop", nullptr};
        int level_id = min_tile_level_idx;
        for (const char **tag = tags; *tag; tag++) {
            tile_level_t level = (tile_level_t)level_id++;
            dim_value_t max_dim_val = max_dim(level);
            oss << "  " << pad_str(*tag + std::string(" dim:"), -19);
            oss << dim(level).str();
            if (!max_dim_val.is_unlimited()) {
                oss << " (max: " << max_dim_val.str() << ")";
            }
            oss << std::endl;
        }
        return oss.str();
    }

    std::string brief_str() const {
        std::ostringstream oss;
        oss << "Dimension " << name_ << pad_str(":", -18 + (int)name_.length());
        oss << "(grid:" << pad_int(grid_dim(), 5) << ") x ";
        oss << "(loop:" << pad_int(loop_dim(), 5) << ") x ";
        oss << "(tg:" << pad_int(tg_dim(), 5) << ") x ";
        oss << "(iter:" << pad_int(iter_dim(), 5) << ")";
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    static std::string pad_str(std::string s, int pad) {
        auto pos = (pad >= 0 ? 0 : s.length());
        s.insert(pos, std::abs(pad) - s.length(), ' ');
        return s;
    }

    static std::string pad_int(int i, int pad) {
        return pad_str(std::to_string(i), pad);
    }

    // Dimension name.
    std::string name_;

    // Dimension size.
    int size_ = 0;

    // Minimal block size for iteration blocking. Iteration-level block must be
    // divisible by this value.
    int base_iter_blk_ = 1;

    // Block size to ensure correct zero padding. Blocked memory layouts must
    // be fully covered to ensure they are zero-padded.
    int pad_blk_ = 1;

    // Dimension kind in terms of BMNK notation.
    char bmnk_ = ' ';

    // Number of prb dims for a BMNK dim
    int inner_dims_ = 0;

    // Whether the dimension can be blocked. Dimensions without blocking are
    // implicitly tiled on the grid level (handle one element per thread
    // group).
    bool is_blocked_ = false;

    // Whether the dimension can be fused with other "fused" dimensions on the
    // same tile level.
    bool allow_fuse_ = false;

    // Whether the dimension can be split between multiple tile levels.
    bool allow_split_ = false;

    bool pref_tg_block_ = false;

    // Dimensions with smaller order keys are tiled first.
    int order_key_ = -1;

    // Dimensions of tiles.
    dim_value_t tile_dims_[max_tile_level_idx + 1];

    // Max allowed dimensions of tiles.
    dim_value_t max_tile_dims_[max_tile_level_idx + 1];
};

// Block helper provides functionality to compute tiling/blocking for a
// GEMM-like problem (when problem dimensions can be classified in terms of
// BMNK dimensions). A typical flow consists of three steps:
// - Setting problem configuration:
//   - Problem dimensions (sizes, padding requirements, base block
//      requirements, BMNK behavior)
//   - HW details, FMA kind, data types, etc
// - Setting restrictions/hints, for example:
//   - Maximal block sizes (e.g. to limit GRF usage)
//   - Fuse/split settings - to get the desired blocking decomposition
//   - K-slicing settings (grid or thread group sliciing)
// - Computing block sizes for each tile: per iteration, per loop, per thread
//   group
class block_helper_t {
public:
    bool is_frozen() const { return is_frozen_; }

    const std::unordered_map<std::string, dim_info_t> &dims() const {
        return dims_;
    }

    dim_info_t &dim(const std::string &name) {
        ir_assert(dims_.count(name) != 0) << "Dimension not found: " << name;
        return dims_.at(name);
    }

    void set_hw_config(const hw_config_t &hw_cfg) {
        check_if_can_set();
        hw_cfg_ = hw_cfg;
    }

    void set_fma_kind(fma_kind_t fma_kind) {
        check_if_can_set();
        fma_kind_ = fma_kind;
    }

    void set_simd_size(int simd_size) {
        check_if_can_set();
        simd_size_ = simd_size;
    }

    void set_vec_size(int vec_size) {
        check_if_can_set();
        vec_size_ = vec_size;
    }

    void set_max_tg_size(int max_tg_size) {
        check_if_can_set();
        max_tg_size_ = max_tg_size;
    }

    void set_max_tg_overridden(bool max_tg_overridden) {
        check_if_can_set();
        max_tg_overridden_ = max_tg_overridden;
    }

    void set_abc_types(
            data_type_t a_type, data_type_t b_type, data_type_t c_type) {
        check_if_can_set();
        a_type_ = a_type;
        b_type_ = b_type;
        c_type_ = c_type;
    }

    void set_use_2d_send(bool use_a_2d_send, bool use_b_2d_send) {
        use_a_2d_send_ = use_a_2d_send;
        use_b_2d_send_ = use_b_2d_send;
    }

    void set_max_m_tg_dim(int value) {
        m_dim().set_max_dim(tile_level_t::tg, value);
    }

    void set_max_n_tg_dim(int value) {
        n_dim().set_max_dim(tile_level_t::tg, value);
    }

    void set_max_k_tg_dim(int value) {
        k_dim().set_max_dim(tile_level_t::tg, value);
    }

    void set_dims(std::initializer_list<std::string> names,
            std::initializer_list<int> sizes) {
        check_if_can_set();
        ir_assert(names.size() == sizes.size());
        for (size_t i = 0; i < names.size(); i++) {
            set_dim(*(names.begin() + i), *(sizes.begin() + i));
        }
    }

    void set_dim(const std::string &name, int size) {
        check_if_can_set();
        ir_assert(dims_.count(name) == 0)
                << "Dimension already exists: " << name;
        dims_.emplace(name, dim_info_t(name, size));
    }

    void set_b_dims(std::initializer_list<std::string> names) {
        set_bmnk_dims(names, 'B');
    }
    void set_m_dims(std::initializer_list<std::string> names) {
        set_bmnk_dims(names, 'M');
    }
    void set_n_dims(std::initializer_list<std::string> names) {
        set_bmnk_dims(names, 'N');
    }
    void set_k_dims(std::initializer_list<std::string> names) {
        set_bmnk_dims(names, 'K');
    }

    void set_block_dims(std::initializer_list<std::string> names) {
        check_if_can_set();
        for (auto &name : names) {
            dim(name).set_blocked();
        }
    }

    void set_loop_dim(const std::string &name, int value) {
        dim(name).set_loop_dim(value);
    }

    void set_tg_dim(const std::string &name, int value) {
        dim(name).set_tg_dim(value);
    }

    void set_max_iter_dim(const std::string &name, int value) {
        dim(name).set_max_dim(tile_level_t::iter, value);
    }

    void set_max_loop_dim(const std::string &name, int value) {
        dim(name).set_max_dim(tile_level_t::loop, value);
    }

    void set_max_tg_dim(const std::string &name, int value) {
        dim(name).set_max_dim(tile_level_t::tg, value);
    }

    void set_pref_tg_block(const std::string &name, bool value = true) {
        dim(name).set_pref_tg_block(value);
    }

    bool any_pref_tg_block() {
        for (auto &kv : dims_) {
            auto &d = kv.second;
            if (d.pref_tg_block()) return true;
        }
        return false;
    }

    void set_reduce_m_block_hint(bool value = true) {
        reduce_m_block_hint_ = value;
        reduce_m_block_hint_set_ = true;
    }

    void allow_fuse(std::initializer_list<std::string> names) {
        check_if_can_set();
        for (auto &name : names) {
            dim(name).set_allow_fuse();
        }
    }

    void allow_split(std::initializer_list<std::string> names) {
        check_if_can_set();
        for (auto &name : names) {
            dim(name).set_allow_split();
        }
    }

    void allow_k_tg_slicing() { allow_k_tg_slicing_ = true; }

    void allow_k_grid_slicing() { allow_k_grid_slicing_ = true; }

    void set_vector_dim(const std::string &name) {
        check_if_can_set();
        auto &d = dim(name);
        d.set_base_iter_block(math::lcm(vec_size_, d.base_iter_block()));
        vector_bmnk_ = d.bmnk();
    }

    void set_base_iter_block(const std::string &name, int block) {
        check_if_can_set();
        dim(name).set_base_iter_block(block);
    }

    void set_base_iter_block(const std::string &name, int block0, int block1) {
        set_base_iter_block(name, math::lcm(block0, block1));
    }

    void set_pad_block(const std::string &name, int block) {
        dim(name).set_pad_block(block);
    }

    void reorder(std::initializer_list<std::string> names) {
        check_if_can_set();
        int key = 0;
        for (auto &name : names)
            dim(name).set_order_key(key++);
    }

    void compute();

    bool has_dim(const std::string &name) const {
        return dims_.count(name) != 0;
    }

    int iter_dim(const std::string &name) const { return dim(name).iter_dim(); }
    int loop_dim(const std::string &name) const { return dim(name).loop_dim(); }
    int tg_dim(const std::string &name) const { return dim(name).tg_dim(); }
    int grid_dim(const std::string &name) const { return dim(name).grid_dim(); }

    int iter_blk(const std::string &name) const { return dim(name).iter_blk(); }
    int loop_blk(const std::string &name) const { return dim(name).loop_blk(); }
    int tg_blk(const std::string &name) const { return dim(name).tg_blk(); }

    dim_value_t max_iter_dim(const std::string &name) const {
        return dim(name).max_dim(tile_level_t::iter);
    }

    int padded_size(const std::string &name) const {
        return dim(name).padded_size();
    }

    std::string str() const {
        std::ostringstream oss;
        for (auto &kv : dims_) {
            auto &d = kv.second;
            if (!d.has_any_blocking()) continue;
            oss << d.str();
        }
        return oss.str();
    }

    std::string brief_str() {
        std::ostringstream oss;
        for (auto &kv : dims_) {
            auto &d = kv.second;
            if (!d.has_any_blocking()) continue;
            oss << "  " << d.brief_str() << std::endl;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    void check_if_can_set() const {
        ir_assert(!is_frozen_) << "Can't set: setup is already frozen.";
    }

    const dim_info_t &dim(const std::string &name) const {
        ir_assert(dims_.count(name) != 0) << "Dimension not found: " << name;
        return dims_.at(name);
    }

    dim_info_t &b_dim() { return bmnk_dims_[0]; }
    dim_info_t &m_dim() { return bmnk_dims_[1]; }
    dim_info_t &n_dim() { return bmnk_dims_[2]; }
    dim_info_t &k_dim() { return bmnk_dims_[3]; }
    const dim_info_t &b_dim() const { return bmnk_dims_[0]; }
    const dim_info_t &m_dim() const { return bmnk_dims_[1]; }
    const dim_info_t &n_dim() const { return bmnk_dims_[2]; }
    const dim_info_t &k_dim() const { return bmnk_dims_[3]; }

    dim_info_t &bmnk_dim(char bmnk) {
        auto &ret = const_cast<const block_helper_t *>(this)->bmnk_dim(bmnk);
        return const_cast<dim_info_t &>(ret);
    }

    const dim_info_t &bmnk_dim(char bmnk) const {
        switch (bmnk) {
            case 'B': return b_dim();
            case 'M': return m_dim();
            case 'N': return n_dim();
            case 'K': return k_dim();
            default: ir_error_not_expected();
        }
        return b_dim();
    }

    int prb_blocked_ndims(char bmnk) const {
        int ret = 0;
        for (auto &kv : dims_) {
            auto &d = kv.second;
            if (d.bmnk() != bmnk) continue;
            if (!d.is_blocked()) continue;
            if (d.size() == 1) continue;
            ret++;
        }
        return ret;
    }

    const dim_info_t &prb_blocked_dim(char bmnk) const {
        ir_assert(prb_blocked_ndims(bmnk) == 1);
        for (auto &kv : dims_) {
            auto &d = kv.second;
            if (d.bmnk() != bmnk) continue;
            if (!d.is_blocked()) continue;
            if (d.size() == 1) continue;
            return d;
        }
        return dim("");
    }

    int prb_max_dim(char bmnk, tile_level_t level) const {
        int ret = 1;
        for (auto &kv : dims_) {
            auto &d = kv.second;
            if (d.bmnk() != bmnk) continue;
            if (!d.is_blocked()) continue;
            ret *= min(d.size(), d.max_dim(level));
        }
        return ret;
    }

    void set_bmnk_dims(std::initializer_list<std::string> dims, char bmnk) {
        check_if_can_set();
        for (auto &d : dims)
            dims_.at(d).set_bmnk(bmnk);
    }

    bool is_x8x8s32() const {
        if (!utils::one_of(a_type_, data_type::s8, data_type::u8)) return false;
        if (!utils::one_of(b_type_, data_type::s8, data_type::u8)) return false;
        if (c_type_ != data_type::s32) return false;
        return true;
    }
    bool is_tf32() const {
        return a_type_ == data_type::tf32 && b_type_ == data_type::tf32
                && c_type_ == data_type::f32;
    }

    bool vectorize_by_b() const { return vectorize_by_bmnk('B'); }
    bool vectorize_by_m() const { return vectorize_by_bmnk('M'); }
    bool vectorize_by_n() const { return vectorize_by_bmnk('N'); }
    bool vectorize_by_k() const { return vectorize_by_bmnk('K'); }

    bool vectorize_by_bmnk(char bmnk) const { return vector_bmnk_ == bmnk; }

    int b_size() const { return b_dim().size(); }
    int m_size() const { return m_dim().size(); }
    int n_size() const { return n_dim().size(); }
    int k_size() const { return k_dim().size(); }

    void init_bmnk_dims() {
        for (char bmnk : {'B', 'M', 'N', 'K'}) {
            auto &d = bmnk_dim(bmnk);
            d.set_name(std::string(1, bmnk));
            d.set_size(1);
            d.set_bmnk(bmnk);
        }

        for (auto &kv : dims_) {
            auto &d = kv.second;
            auto &bmnk_d = bmnk_dim(d.bmnk());
            if (d.pref_tg_block()) bmnk_d.set_pref_tg_block();
            bmnk_d.set_size(bmnk_d.size() * d.size());
            bmnk_d.set_base_iter_block(
                    bmnk_d.base_iter_block() * d.base_iter_block());
            if (d.is_blocked() && d.size() != 1) bmnk_d.incr_inner_dims();
        }
    }

    void init_bmnk_blocks();
    void init_k_blocking();
    bool enable_k_tg_slicing() const;
    void init_prb_blocks();
    int compute_mad_k_block() const;

    static int compute_block(int dim, int target_blk, int base_iter_block,
            double target_eff = 0.75) {
        int nblks = ir_utils::safe_divide(target_blk, base_iter_block);
        while (nblks != 1) {
            int dim_padded = utils::rnd_up(dim, nblks * base_iter_block);
            double eff = (double)dim / dim_padded;
            if (eff >= target_eff) break;
            nblks--;
        }
        return nblks * base_iter_block;
    }

    // Whether compute() was already called.
    bool is_frozen_ = false;

    // BMNK kind of dimension to vectorize.
    char vector_bmnk_ = ' ';

    // General information about HW and computation.
    hw_config_t hw_cfg_;
    fma_kind_t fma_kind_ = fma_kind_t::unknown;
    int vec_size_ = -1;
    int simd_size_ = -1;
    int max_tg_size_ = 0;
    bool max_tg_overridden_ = false;
    data_type_t a_type_ = data_type::undef;
    data_type_t b_type_ = data_type::undef;
    data_type_t c_type_ = data_type::undef;

    bool use_a_2d_send_ = false;
    bool use_b_2d_send_ = false;

    // Whether K computation can be split across threads in thread group.
    bool allow_k_tg_slicing_ = false;

    // Whether K computation can be split across thread groups in the grid.
    bool allow_k_grid_slicing_ = false;

    // Problem dimensions.
    std::unordered_map<std::string, dim_info_t> dims_;

    // BMNK dimensions.
    static const int bmnk_length = 4;
    dim_info_t bmnk_dims_[bmnk_length];

    bool reduce_m_block_hint_;
    bool reduce_m_block_hint_set_ = false;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
