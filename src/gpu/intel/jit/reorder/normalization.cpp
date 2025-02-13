/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/jit/reorder/normalization.hpp"

#include "gpu/intel/jit/utils/range.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace reorder {

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

        iterator_t(dim_idx_t ndims, block_iterator_t it, block_iterator_t end)
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
    bool contains_dim(dim_idx_t dim_idx) const {
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
        blocks_ = std::move(blocks);
    }

    void reindex(int ndims, const std::vector<int> &map) {
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
    dim_idx_t ndims_;
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
void normalize(layout_t &a, layout_t &b) {
    using direction_t = merge_info_t::merge_direction_t;
    int ndims = a.ndims();
    auto cmp = [](const normalization_stage_t &a,
                       const normalization_stage_t &b) {
        return a.elems() <= b.elems();
    };
    auto dim_blocks = [](dim_idx_t dim_idx) {
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
    a_normalization.merge(std::move(a_merges));
    b_normalization.merge(std::move(b_merges));

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

} // namespace reorder
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
