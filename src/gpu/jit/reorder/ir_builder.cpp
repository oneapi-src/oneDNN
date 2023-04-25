/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "gpu/jit/reorder/ir_builder.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "gpu/jit/ir/epilogue.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/post_ops.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/pass/pass.hpp"
#include "gpu/jit/utils/iterator.hpp"
#include "gpu/jit/utils/range.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

dim_t reorder_ir_builder_t::count_block_messages(
        const exec_config_t &exec_cfg, dim_t inner_bytes, dim_t iterations) {
    const auto max_block_owords = exec_cfg.grf_size() / 2;
    const auto oword_size = 16;
    const auto owords_per_grf = exec_cfg.grf_size() / oword_size;

    dim_t block_owords = max_block_owords / 2;
    auto inner_owords = inner_bytes / oword_size;
    dim_t messages = inner_owords / max_block_owords;
    inner_owords -= messages * max_block_owords;
    // If iterations != 1, tail block messages must end on a grf boundary
    const dim_t lower_bound = iterations == 1 ? 1 : owords_per_grf;
    for (; block_owords >= lower_bound; block_owords >>= 1) {
        if (inner_owords >= block_owords) {
            inner_owords -= block_owords;
            messages++;
        }
    }
    ir_assert(inner_owords == 0);
    return messages * iterations;
}

dim_t reorder_ir_builder_t::count_scattered_messages(
        const exec_config_t &exec_cfg, dim_t inner_bytes, dim_t iterations) {
    const auto max_block_items = exec_cfg.grf_size() / 2;
    int item_size = 8;

    // Find the largest uint size we can use
    for (; item_size > 1; item_size >>= 1) {
        if (inner_bytes % item_size == 0) break;
    }

    dim_t block_items = max_block_items / 2;
    auto inner_items = (iterations * inner_bytes) / item_size;
    dim_t messages = (inner_items + (block_items - 1)) / max_block_items;
    inner_items -= std::min(inner_items, messages * max_block_items);
    for (; block_items >= (dim_t)2; block_items >>= 1) {
        if (inner_items > block_items / 2) {
            inner_items -= std::min(inner_items, block_items);
            messages++;
        }
    }
    if (inner_items) messages++;
    return messages;
}

dim_t reorder_ir_builder_t::message_latency(
        const exec_config_t &exec_cfg, const layout_t &l, const tensor_t &t) {
    const auto grf_size = exec_cfg.grf_size();
    const int scattered_message_penalty = 4;
    bool can_use_block_messages = true;
    std::vector<dim_t> outer = t.dims();
    dim_t inner_elems = 1;

    for (auto &blk : l.blocks()) {
        auto block = blk.block;
        auto dim_idx = blk.dim_idx;
        if (block == 1) continue;
        if (outer[dim_idx] < block) {
            if (block % outer[dim_idx] == 0) {
                inner_elems *= outer[dim_idx];
                outer[dim_idx] = 1;
            }
            break;
        }

        can_use_block_messages &= (outer[dim_idx] % block == 0);
        inner_elems *= block;
        outer[dim_idx] = utils::div_up(outer[dim_idx], block);
    }

    auto type_size = l.type().scalar().size();
    auto inner_bytes = inner_elems * type_size;
    auto iterations = tensor_t(outer).elems();
    can_use_block_messages &= (inner_bytes % 16 == 0);
    can_use_block_messages &= (iterations == 1 || inner_bytes % grf_size == 0);

    if (inner_bytes == 0 || iterations == 0) return 0;

    return can_use_block_messages
            ? count_block_messages(exec_cfg, inner_bytes, iterations)
            : count_scattered_messages(exec_cfg, inner_bytes, iterations)
                    * scattered_message_penalty;
}

void reorder_ir_builder_t::compute_blocks(const exec_config_t &exec_cfg,
        const layout_t &src, const layout_t &dst, std::vector<int> &iter_blocks,
        std::vector<int> &loop_blocks, std::vector<int> &tg_blocks,
        dim_t max_iter_tile_bytes, dim_t max_thr_tile_bytes) {
    if (max_iter_tile_bytes <= 0)
        max_iter_tile_bytes = max_tile_size(exec_cfg.hw_cfg(), dst, src);
    if (max_thr_tile_bytes <= 0)
        max_thr_tile_bytes = max_tile_size(exec_cfg.hw_cfg(), dst, src);

    ir_assert(src.ndims() == dst.ndims());
    int ndims = src.ndims();
    std::vector<dim_t> dims(ndims);
    for (int i = 0; i < ndims; i++) {
        dims[i] = std::max(src.dim(i), dst.dim(i));
    }

    // Pad src/dst layouts to match each other.
    auto pad_layout = [&](const layout_t &l) {
        std::vector<block_t> padded_blocks;
        for (auto &eb : l.enumerated_blocks()) {
            auto b = eb.second;
            if (l.is_outermost(eb)) {
                dim_t inner = l.dim(b.dim_idx) / b.block;
                b.block = ir_utils::safe_divide(dims[b.dim_idx], inner);
            }
            padded_blocks.push_back(b);
        }
        return layout_t(
                l.type(), ndims, 0, padded_blocks, /*do_normalize=*/false);
    };
    layout_t padded_src = pad_layout(src);
    layout_t padded_dst = pad_layout(dst);
    ir_assert(ir_utils::is_equal(padded_src.dims(), padded_dst.dims()));

    dim_t elems = padded_src.elems();
    int max_type_size = std::max(src.type().size(), dst.type().size());
    dim_t max_iter_tile_elems
            = std::min(max_iter_tile_bytes / max_type_size, elems);
    dim_t max_thr_tile_elems
            = std::min(max_thr_tile_bytes / max_type_size, elems);

    using tile_pair_t = std::array<tensor_t, 2>;

    auto can_be_mapped = [](const layout_t &l, const tensor_t &t) {
        std::vector<dim_t> rem_dims = t.dims();
        for (auto &b : l.blocks()) {
            auto &rem_dim = rem_dims[b.dim_idx];
            if (rem_dim >= b.block) {
                if (rem_dim % b.block != 0) return false;
                rem_dim /= b.block;
                continue;
            }
            if (b.block % rem_dim != 0) return false;
            rem_dim = 1;
        }
        for (auto d : rem_dims)
            ir_assert(d == 1);
        return true;
    };

    auto add_pseudo_dimension = [](const layout_t &l) {
        auto layout_size = l.size();
        return [=](const tensor_t &t) {
            auto dims = t.dims();
            dims.push_back(layout_size);
            return tensor_t(dims);
        };
    };

    auto mappable_tiles = [&](const tensor_t &t) {
        return can_be_mapped(padded_src, t) && can_be_mapped(padded_dst, t);
    };

    auto merge_tiles = [](const tile_pair_t &p) {
        auto ndims = p[0].ndims() - 1;
        std::vector<dim_t> dims(ndims);
        for (int i = 0; i < ndims; ++i)
            dims[i] = std::max(p[0](i), p[1](i));
        return tensor_t(dims);
    };

    auto take_smaller = [](const tensor_t &a, const tensor_t &b) {
        return a.elems() < b.elems();
    };

    // Incrementally increase subtiles in src and dst. The goal is to find the
    // maximum src/dst tiles so that the final combined tile covers dense
    // regions as big as possible in src/dst layouts.
    std::vector<tensor_t> candidate_tiles;
    auto a_tiles = inner_tiles(padded_src.blocks(), padded_src.ndims())
            | filter(mappable_tiles)
            | transform(add_pseudo_dimension(padded_src));
    auto b_tiles = inner_tiles(padded_dst.blocks(), padded_dst.ndims())
            | filter(mappable_tiles)
            | transform(add_pseudo_dimension(padded_dst));
    auto tiles = merge(a_tiles, b_tiles, take_smaller) | transform(merge_tiles);
    for (auto tile : tiles) {
        if (tile.elems() > max_thr_tile_elems) break;
        candidate_tiles.push_back(tile);
    }
    ir_assert(!candidate_tiles.empty());

    const auto eu_count = exec_cfg.hw_cfg().eu_count();
    std::sort(candidate_tiles.begin(), candidate_tiles.end(),
            [&](const tensor_t &a, const tensor_t &b) {
                auto a_threads_reqd = padded_src.elems() / a.elems();
                auto b_threads_reqd = padded_src.elems() / b.elems();
                auto a_eu_util = utils::div_up(a_threads_reqd, eu_count);
                auto b_eu_util = utils::div_up(b_threads_reqd, eu_count);
                auto a_msg_load = message_latency(exec_cfg, padded_src, a)
                        + message_latency(exec_cfg, padded_dst, a);
                auto b_msg_load = message_latency(exec_cfg, padded_src, b)
                        + message_latency(exec_cfg, padded_dst, b);

                // Choose tiles with less message overhead per thread
                if (a_eu_util * a_msg_load != b_eu_util * b_msg_load)
                    return (a_eu_util * a_msg_load < b_eu_util * b_msg_load);

                // Choose tiles with more bytes per message
                if (a.elems() * b_msg_load != b.elems() * a_msg_load)
                    return (a.elems() * b_msg_load > b.elems() * a_msg_load);

                // If all else fails, go with the bigger tile
                return a.elems() > b.elems();
            });

    tensor_t thr_tile = candidate_tiles[0];
    tensor_t iter_tile;
    for (auto &tile : candidate_tiles) {
        if (tile.elems() > max_iter_tile_elems || !thr_tile.is_divisible(tile))
            continue;
        if (iter_tile.is_empty() || tile.elems() > iter_tile.elems())
            iter_tile = tile;
    }

    ir_assert(!iter_tile.is_empty());
    std::vector<int> thr_blocks(thr_tile.dims().begin(), thr_tile.dims().end());
    iter_blocks.assign(iter_tile.dims().begin(), iter_tile.dims().end());

    ir_assert(utils::array_product(iter_blocks) <= max_iter_tile_elems);
    ir_assert(utils::array_product(thr_blocks) <= max_thr_tile_elems);

    // Initialize loop blocks.
    loop_blocks.resize(ndims, 1);
    for (int i = 0; i < ndims; i++) {
        loop_blocks[i] = ir_utils::safe_divide(thr_blocks[i], iter_blocks[i]);
    }

    // Initialize thread group blocks.
    // Heuristic: try to split outer dimension and assign its
    // inner part to the thread group. This may give better
    // bandwidth utilization on XeHP/XeHPG.
    tg_blocks.resize(ndims, 1);
    const int tg_factor = 2;
    for (int i = 0; i < ndims; i++) {
        int outer = utils::div_up(dims[i], thr_blocks[i]);
        if (outer % tg_factor == 0) {
            tg_blocks[i] = tg_factor;
            break;
        }
    }
}

void reorder_ir_builder_t::compute_blocks(const exec_config_t &exec_cfg,
        const layout_t &src, const layout_t &dst, std::vector<int> &tile_blocks,
        std::vector<int> &tg_blocks) {
    std::vector<int> iter_blocks;
    std::vector<int> loop_blocks;
    compute_blocks(exec_cfg, src, dst, iter_blocks, loop_blocks, tg_blocks);
    size_t n = iter_blocks.size();
    tile_blocks.resize(n);
    for (size_t i = 0; i < n; i++) {
        tile_blocks[i] = iter_blocks[i] * loop_blocks[i];
    }
}

void reorder_ir_builder_t::compute_grid(const layout_t &src,
        const layout_t &dst, const std::vector<int> &iter_blocks,
        const std::vector<int> &loop_blocks, const std::vector<int> &tg_blocks,
        grid_info_t &kernel_grid, grid_info_t &tg_grid,
        std::vector<int> *dim2grid) {
    int ndims = src.ndims();
    std::vector<dim_t> dims(ndims);
    for (int i = 0; i < ndims; i++) {
        dims[i] = std::max(src.dim(i), dst.dim(i));
    }

    if (dim2grid) dim2grid->resize(ndims, -1);

    const int grid_ndims = 3;
    std::vector<int> kernel_grid_dims(grid_ndims, 1);
    std::vector<int> tg_grid_dims(grid_ndims, 1);
    int grid_idx = 0;
    int max_grid_idx = grid_ndims - 1;
    for (int i = 0; i < ndims; i++) {
        if (dim2grid) (*dim2grid)[i] = grid_idx;
        int outer = utils::div_up(
                dims[i], iter_blocks[i] * loop_blocks[i] * tg_blocks[i]);
        tg_grid_dims[grid_idx] *= tg_blocks[i];
        kernel_grid_dims[grid_idx] *= outer;
        if (outer != 1 && grid_idx != max_grid_idx) grid_idx++;
    }
    kernel_grid = grid_info_t(kernel_grid_dims, "grid_idx");
    tg_grid = grid_info_t(tg_grid_dims, "grid_idx");
}

compute::nd_range_t reorder_ir_builder_t::nd_range(
        const exec_config_t &exec_cfg, layout_t src, layout_t dst) {
    const int simd = exec_cfg.simd();
    std::vector<int> iter_blocks;
    std::vector<int> loop_blocks;
    std::vector<int> tg_blocks;
    normalize_reorder_layouts(src, dst);
    compute_blocks(exec_cfg, src, dst, iter_blocks, loop_blocks, tg_blocks);
    grid_info_t kernel_grid;
    grid_info_t tg_grid;
    compute_grid(src, dst, iter_blocks, loop_blocks, tg_blocks, kernel_grid,
            tg_grid);
    std::array<size_t, 3> global;
    std::array<size_t, 3> local;
    for (int i = 0; i < kernel_grid.ndims(); i++) {
        global[i] = kernel_grid[i] * tg_grid[i];
        local[i] = tg_grid[i];
        if (i == 0) {
            global[i] *= simd;
            local[i] *= simd;
        }
    }
    return compute::nd_range_t(global.data(), local.data());
}

struct normalization_stage_t {
    int idx;
    block_t curr, last;
    std::array<dim_t, 2> tile;

    bool is_dense() const { return curr.stride == last.stride * last.block; }

    dim_t elems() const { return tile[0]; }

    normalization_stage_t() = default;
    normalization_stage_t(int idx, const block_t &curr, const block_t &last,
            std::vector<dim_t> tile)
        : idx(idx)
        , curr(curr)
        , last(last)
        , tile({tile[curr.dim_idx], tile[last.dim_idx]}) {}
};

struct merge_info_t {
    enum class merge_direction_t { none = 0, forward, backward };

    int iter_idx;
    merge_direction_t direction;

    merge_info_t(int iter_idx, merge_direction_t direction)
        : iter_idx(iter_idx), direction(direction) {}
};

merge_info_t::merge_direction_t merge_direction(
        const normalization_stage_t &l, const normalization_stage_t &r) {
    using direction_t = merge_info_t::merge_direction_t;
    if (l.curr.dim_idx != r.curr.dim_idx) return direction_t::none;
    if (l.last.dim_idx != r.last.dim_idx) return direction_t::none;
    if (l.tile[0] != r.tile[0]) return direction_t::none;
    if (l.curr.block == r.curr.block
            && l.tile[1] * l.last.block == r.tile[1] * r.last.block)
        return direction_t::backward;
    if (l.tile[1] == r.tile[1] && l.last.block == r.last.block)
        return direction_t::forward;
    return direction_t::none;
}

struct layout_normalization_t {
    using blocks_t = std::vector<block_t>;
    using block_iterator_t = typename blocks_t::const_iterator;
    using stage_t = normalization_stage_t;

    struct iterator_t {
        bool operator==(const iterator_t &o) const { return curr_ == o.curr_; }
        bool operator!=(const iterator_t &o) const { return !operator==(o); }
        stage_t operator*() const { return {idx_, *curr_, *last_, tile_}; }
        iterator_t &operator++() {
            if (curr_ == end_) return *this;
            auto blk = *last_;
            tile_[blk.dim_idx] *= blk.block;
            last_ = curr_;
            ++curr_;
            ++idx_;
            return *this;
        }

        iterator_t(int ndims, block_iterator_t it, block_iterator_t end)
            : curr_(it == end ? end : it + 1)
            , last_(it)
            , end_(end)
            , idx_(0)
            , tile_(ndims, 1) {}

    private:
        block_iterator_t curr_, last_, end_;
        int idx_;
        std::vector<dim_t> tile_;
    };

    int ndims() const { return ndims_; }
    const blocks_t &blocks() const { return blocks_; }

    bool empty() const { return begin() == end(); }
    bool contains_dim(int dim_idx) const {
        for (auto &blk : blocks_)
            if (blk.dim_idx == dim_idx) return true;
        return false;
    }

    void merge(std::vector<merge_info_t> merges) {
        using direction_t = merge_info_t::merge_direction_t;
        if (empty()) {
            if (blocks_.empty()) blocks_.emplace_back(0, 1, 1);
            return;
        }

        std::sort(merges.begin(), merges.end(),
                [](const merge_info_t &l, const merge_info_t &r) {
                    return l.iter_idx < r.iter_idx;
                });
        auto merge_it = merges.begin();
        auto merge_end = merges.end();
        std::vector<block_t> blocks;
        block_t last = (*begin()).last;
        for (auto s : *this) {
            if (merge_it != merge_end && merge_it->iter_idx == s.idx) {
                if (merge_it->direction == direction_t::backward)
                    s.curr.dim_idx = last.dim_idx;
                s.curr.block *= last.block;
                s.curr.stride = last.stride;
                ++merge_it;
            } else
                blocks.push_back(last);
            last = s.curr;
        }
        blocks.push_back(last);
        blocks_ = blocks;
    }

    void reindex(int ndims, std::vector<int> map) {
        ndims_ = ndims;
        for (auto &blk : blocks_)
            blk.dim_idx = map[blk.dim_idx];
    }

    layout_t layout() const {
        return {type_, ndims_, offset_, blocks_, /*do_normalize=*/false};
    }

    iterator_t begin() const {
        return {ndims_, blocks_.begin(), blocks_.end()};
    }
    iterator_t end() const { return {ndims_, blocks_.end(), blocks_.end()}; }

    layout_normalization_t(
            const layout_t &layout, const std::vector<bool> &dim_empty)
        : type_(layout.type())
        , ndims_(layout.ndims())
        , offset_(layout.offset())
        , blocks_(normalized_blocks(layout, dim_empty)) {}

private:
    static std::vector<block_t> normalized_blocks(
            const layout_t &layout, std::vector<bool> dim_empty) {
        std::vector<block_t> normalized_blocks;
        for (auto &eb : layout.enumerated_blocks()) {
            auto &blk = eb.second;
            if (blk.block != 1
                    || (layout.is_outermost(eb) && !dim_empty[blk.dim_idx])) {
                if (normalized_blocks.empty()
                        || normalized_blocks.back().dim_idx != blk.dim_idx) {
                    normalized_blocks.push_back(blk);
                    dim_empty[blk.dim_idx] = true;
                } else {
                    normalized_blocks.back().block *= blk.block;
                }
            }
        }
        return normalized_blocks;
    }

    type_t type_;
    int ndims_;
    expr_t offset_;
    blocks_t blocks_;
};

// Given two layouts, finds an equivalent pair of simpler layouts by attempting
// to combine consecutive blocks that appear in both layouts at the same level
// of nesting for the dimensions to which the blocks belong. E.g.,
//
//             1.          2.
// 16a16b16c ---> 256a16c ---> 256a16b
// 16c16a16b ---> 16c256a ---> 16b256a
//
// 1. The consecutive blocks 16a16b are repeated. For the first layout it
//    appears with an inner tile 1x1x16, and 1x1x1 for the second. Because the
//    ab-subtile is 1x1 for both and  the inner block (16b) is the same for
//    both, we can combine these blocks.
// 2. The b dimension no longer appears, so we can remove it from the layout and
//    re-index the dimensions so that the new layouts are 2D.
void reorder_ir_builder_t::normalize_reorder_layouts(layout_t &a, layout_t &b) {
    using direction_t = merge_info_t::merge_direction_t;
    int ndims = a.ndims();
    auto cmp = [](const normalization_stage_t &a,
                       const normalization_stage_t &b) {
        return a.elems() <= b.elems();
    };
    auto dim_blocks = [](int dim_idx) {
        return [=](const normalization_stage_t &s) {
            return s.curr.dim_idx == dim_idx;
        };
    };

    std::vector<bool> empty_dimension(ndims, true);
    for (auto &blk : a.blocks())
        if (blk.block != 1) empty_dimension[blk.dim_idx] = false;
    for (auto &blk : b.blocks())
        if (blk.block != 1) empty_dimension[blk.dim_idx] = false;

    layout_normalization_t a_normalization {a, empty_dimension};
    layout_normalization_t b_normalization {b, empty_dimension};

    std::vector<merge_info_t> a_merges;
    std::vector<merge_info_t> b_merges;
    // Find pairs of consecutive blocks which can be combined
    for (int i = 0; i < ndims; ++i) {
        auto dim_i_blocks = dim_blocks(i);
        auto a_stages = a_normalization | filter(dim_i_blocks);
        auto b_stages = b_normalization | filter(dim_i_blocks);
        for (auto p : merge(a_stages, b_stages, cmp)) {
            if (!p[0].is_dense() || !p[1].is_dense()) continue;
            direction_t direction = merge_direction(p[0], p[1]);
            if (direction == direction_t::none) continue;
            a_merges.emplace_back(p[0].idx, direction);
            b_merges.emplace_back(p[1].idx, direction);
        }
    }
    a_normalization.merge(a_merges);
    b_normalization.merge(b_merges);

    // Find dimensions present in either normalized layout and construct map of
    // new dimension indices
    int curr_dim = 0;
    std::vector<int> dim_map(ndims);
    for (int i = 0; i < ndims; ++i)
        if (a_normalization.contains_dim(i) || b_normalization.contains_dim(i))
            dim_map[i] = curr_dim++;
    a_normalization.reindex(curr_dim, dim_map);
    b_normalization.reindex(curr_dim, dim_map);

    a = a_normalization.layout();
    b = b_normalization.layout();
}

void reorder_ir_builder_t::build() {
    std::vector<int> iter_blocks;
    std::vector<int> loop_blocks;
    std::vector<int> tg_blocks;
    compute_blocks(cfg_.exec_cfg(), src_layout_, dst_layout_, iter_blocks,
            loop_blocks, tg_blocks);

    int max_iters = 10;
    int cur_iter_bytes
            = max_tile_size(cfg_.exec_cfg().hw_cfg(), dst_layout_, src_layout_);
    for (int i = 0; i < max_iters; i++) {
        if (try_build(iter_blocks, loop_blocks, tg_blocks)) {
            ir_info() << "Reorder configuration:" << std::endl;
            ir_info() << "  Source layout:              " << src_layout_
                      << std::endl;
            ir_info() << "  Destination layout:         " << dst_layout_
                      << std::endl;
            ir_info() << "  Iteration blocks:           "
                      << ir_utils::make_seq_print_helper(iter_blocks, " x ")
                      << std::endl;
            ir_info() << "  Loop blocks:                "
                      << ir_utils::make_seq_print_helper(loop_blocks, " x ")
                      << std::endl;
            ir_info() << "  Thread group blocks:        "
                      << ir_utils::make_seq_print_helper(tg_blocks, " x ")
                      << std::endl;
            return;
        }

        cur_iter_bytes /= 2;
        while (cur_iter_bytes >= 1) {
            std::vector<int> new_iter_blocks;
            compute_blocks(cfg_.exec_cfg(), src_layout_, dst_layout_,
                    new_iter_blocks, loop_blocks, tg_blocks, cur_iter_bytes);
            if (!ir_utils::is_equal(new_iter_blocks, iter_blocks)) {
                iter_blocks = new_iter_blocks;
                break;
            }
            cur_iter_bytes /= 2;
        }
    }
    ir_error_not_expected();
}

bool reorder_ir_builder_t::try_build(const std::vector<int> &iter_blocks,
        const std::vector<int> &loop_blocks,
        const std::vector<int> &tg_blocks) {
    constraint_set_t init_cset;

    int ndims = src_layout_.ndims();
    std::vector<expr_t> vars;
    for (int i = 0; i < ndims; i++) {
        char letter = 'a' + i;
        vars.push_back(var_t::make(type_t::s32(), std::string(1, letter)));
    }

    std::vector<int> dim2grid;
    compute_grid(src_layout_, dst_layout_, iter_blocks, loop_blocks, tg_blocks,
            kernel_grid_, tg_grid_, &dim2grid);

    std::vector<stmt_t> init_stmts;
    init_kernel_grid(kernel_grid_, tg_grid_, cfg_.exec_cfg().simd(), init_cset,
            init_stmts);

    std::vector<dim_t> vdims(ndims);
    for (int i = 0; i < ndims; i++) {
        vdims[i] = std::max(src_layout_.dim(i), dst_layout_.dim(i));
    }
    std::unordered_map<std::string, int> vdim_map;
    for (int i = 0; i < ndims; i++) {
        vdim_map[vars[i].as<var_t>().name] = vdims[i];
    }

    view_t src_view(vars, ndims);
    for (int i = 0; i < ndims; i++) {
        src_view.set_vdim(vars[i], vdims[i]);
        src_view.set_tdim(i, vars[i]);
    }
    src_view.set_tlayout(src_layout_);
    src_view.set_tmasks(vdim_map);

    view_t dst_view(vars, ndims);
    for (int i = 0; i < ndims; i++) {
        dst_view.set_vdim(vars[i], vdims[i]);
        dst_view.set_tdim(i, vars[i]);
    }
    dst_view.set_tlayout(dst_layout_);
    dst_view.set_tmasks(vdim_map);

    gemm_schedule_t schedule(init_cset, kernel_grid_, tg_grid_);

    schedule.set_view(src_view);
    schedule.set_view(dst_view);

    std::array<std::vector<expr_t>, 3> fused_idxs;
    for (int i = 0; i < ndims; i++) {
        std::vector<expr_t> ordered;
        auto v = vars[i];
        if (iter_blocks[i] != 1) {
            expr_t outer, inner;
            schedule.split(v, iter_blocks[i], outer, inner);
            schedule.tensorize(inner);
            v = outer;
            ordered.insert(ordered.begin(), outer);
        }
        if (loop_blocks[i] != 1) {
            if (!ordered.empty()) ordered.erase(ordered.begin());
            expr_t outer, inner;
            schedule.split(v, loop_blocks[i], outer, inner);
            v = outer;
            ordered.insert(ordered.begin(), inner);
            ordered.insert(ordered.begin(), outer);
        }
        if (tg_blocks[i] != 1) {
            if (!ordered.empty()) ordered.erase(ordered.begin());
            expr_t outer, inner;
            schedule.split(v, tg_blocks[i], outer, inner);
            schedule.bind(inner, tg_grid_.idx(dim2grid[i]));
            v = outer;
            ordered.insert(ordered.begin(), inner);
            ordered.insert(ordered.begin(), outer);
        }
        fused_idxs[dim2grid[i]].push_back(v);
        schedule.reorder(ordered);
    }

    for (int i = 0; i < (int)fused_idxs.size(); i++) {
        auto &vec = fused_idxs[i];
        if (vec.empty()) continue;
        auto var = (vec.size() == 1 ? vec[0] : schedule.fuse(vec));
        schedule.bind(var, kernel_grid_.idx(i));
    }

    schedule.finalize();

    auto thr_tile = schedule.thr_view_tile(src_view, /*is_relative=*/false);

    auto src_thr_view = src_view.create_sub_view(thr_tile);
    auto dst_thr_view = dst_view.create_sub_view(thr_tile);

    auto src_buf = kernel_info_.arg_var(0);
    auto dst_buf = kernel_info_.arg_var(1);

    ir_context_t ir_ctx(cfg_.exec_cfg(), init_cset);
    auto reg_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "reg");

    std::vector<stmt_t> allocs;
    for (int i = 0; i < kernel_info_.nargs(); i++) {
        auto &var = kernel_info_.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }

    auto read_params = get_send_params(cfg_.exec_cfg(), send_op_t::load,
            send_address_t::a64, src_thr_view);
    read_params.try_legacy = false;
    auto read = make_access_builder(
            ir_ctx, src_thr_view, src_buf, reg_buf, read_params);
    auto &read_stmt = read.stmt();

    auto write_params = get_send_params(cfg_.exec_cfg(), send_op_t::store,
            send_address_t::a64, dst_thr_view);
    write_params.try_legacy = false;
    auto write = make_access_builder(
            ir_ctx, dst_thr_view, dst_buf, reg_buf, write_params);
    auto write_stmt = write.stmt();

    auto &read_layout = read.reg_layout();
    auto &write_layout = write.reg_layout();
    int read_buf_size = read.reg_buf_size();
    int write_buf_size = write.reg_buf_size();

    bool has_post_ops = dst_md_ && attr_
            && (!attr_->post_ops_.has_default_values()
                    || !attr_->zero_points_.has_default_values()
                    || !attr_->scales_.has_default_values());

    if (has_post_ops) {
        post_op_view_mapper_t view_mapper(dst_view);
        post_op_context_t post_op_ctx(*attr_, cfg_.zp_cfg(), schedule,
                kernel_info_, *dst_md_, *dst_md_, view_mapper);
        write_stmt = create_epilogue_stmt(cfg_.exec_cfg(), ir_ctx, schedule,
                /*force_c_reorder=*/true, post_op_ctx, thr_tile, read_layout,
                dst_buf, reg_buf, write_buf_size);
    } else if (read_layout != write_layout) {
        auto tmp_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "tmp");
        allocs.push_back(
                alloc_t::make(tmp_buf, write_buf_size, alloc_kind_t::grf));
        auto reorder_stmt = create_reorder_stmt(
                read_layout, write_layout, reg_buf, tmp_buf);
        write_stmt = substitute(write_stmt, reg_buf, tmp_buf);
        write_stmt = reorder_stmt.append(write_stmt);
    } else {
        read_buf_size = std::max(read_buf_size, write_buf_size);
    }

    allocs.push_back(alloc_t::make(reg_buf, read_buf_size, alloc_kind_t::grf));

    stmt_ = stmt_t();
    stmt_ = stmt_.append(read_stmt);
    stmt_ = stmt_.append(write_stmt);

    stmt_ = schedule.create_loop_nest(stmt_);
    stmt_ = schedule.create_bind_stmt(stmt_);
    stmt_ = inject_let_stmts(stmt_, init_stmts);
    stmt_ = inject_alloc_stmts(stmt_, allocs);
    stmt_ = inject_external_var_let(stmt_, ir_ctx);

    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = lift_buffer_offsets_in_send(stmt_, ir_ctx);
    stmt_ = inject_send(stmt_, ir_ctx);
    stmt_ = split_wide_stores(stmt_, ir_ctx);
    stmt_ = fix_int32_overflow(stmt_, ir_ctx);
    stmt_ = eliminate_common_subexprs(
            stmt_, ir_ctx, cfg_.exec_cfg().regs() * cfg_.exec_cfg().grf_size());
    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = optimize_alloc_let(stmt_, ir_ctx);
    stmt_ = stmt_group_t::make(stmt_label_t::kernel(), stmt_);

    int ir_regs = get_peak_regs(stmt_, cfg_.exec_cfg().grf_size());
    int reserved_regs = 16;
    int regs = ir_regs + reserved_regs;
    if (regs > cfg_.exec_cfg().regs()) {
        ir_warning() << "Estimated GRF usage is " << regs
                     << " registers which exceeds available space, retry with "
                        "a smaller tile."
                     << std::endl;

        return false;
    }

    ir_trace() << "Reorder kernel body:\n" << stmt_ << std::endl;
    return true;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
