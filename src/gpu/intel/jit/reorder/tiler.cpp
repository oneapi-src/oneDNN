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

#include "gpu/intel/jit/reorder/tiler.hpp"

#include "gpu/intel/jit/utils/range.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace reorder {

enum class message_kind_t {
    block,
    scattered,
};

dim_t max_bytes(const hw_t &hw, const layout_t &a, const layout_t &b) {
    // XeHPC is fine with 2048 bytes, XeHPG and below can fit 2048 bytes if
    // reorder is a simple copy.
    return (hw <= ngen::HW::XeHPG && a != b) ? 1024 : 2048;
}

dim_t count_block_messages(
        const hw_t &hw, dim_t inner_bytes, dim_t iterations) {
    const auto max_block_owords = hw.grf_size() / 2;
    const auto oword_size = 16;
    const auto owords_per_grf = hw.grf_size() / oword_size;

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
    gpu_assert(inner_owords == 0);
    return messages * iterations;
}

dim_t count_scattered_messages(
        const hw_t &hw, dim_t inner_bytes, dim_t iterations, int item_size) {
    constexpr int scattered_message_penalty = 4;
    const int message_items = hw.grf_size() / 2;

    auto inner_items = (iterations * inner_bytes) / item_size;
    auto messages = utils::div_up(inner_items, message_items);
    return messages * scattered_message_penalty;
}

struct message_info_t {
    message_info_t() = default;
    message_info_t(message_kind_t kind, dim_t inner_bytes, dim_t iterations,
            int item_size)
        : kind(kind)
        , inner_bytes(inner_bytes)
        , iterations(iterations)
        , item_size(item_size) {}

    message_kind_t kind = message_kind_t::block;
    dim_t inner_bytes = 0;
    dim_t iterations = 0;
    int item_size = 16;

    dim_t latency(const hw_t &hw) const {
        if (inner_bytes == 0 || iterations == 0) return 0;
        return kind == message_kind_t::block
                ? count_block_messages(hw, inner_bytes, iterations)
                : count_scattered_messages(
                        hw, inner_bytes, iterations, item_size);
    }
};

message_info_t estimate_message_info(
        const hw_t &hw, const layout_t &layout, const tensor_t &tile) {
    const auto grf_size = hw.grf_size();
    bool can_use_block_messages = true;
    std::vector<dim_t> outer = tile.dims();
    dim_t inner_elems = 1;
    int item_size = 16;

    for (auto &blk : layout.blocks()) {
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

    auto inner_bytes = utils::div_up(
            layout.type().with_elems(8).size() * inner_elems, 8);
    auto iterations = tensor_t(outer).elems();
    can_use_block_messages &= (inner_bytes % 16 == 0);
    can_use_block_messages &= (iterations == 1 || inner_bytes % grf_size == 0);

    if (inner_bytes == 0 || iterations == 0) return {};

    auto message_kind = can_use_block_messages ? message_kind_t::block
                                               : message_kind_t::scattered;
    if (!can_use_block_messages)
        // Find the largest unit size we can use
        for (item_size = 8; item_size > 1; item_size >>= 1) {
            if (inner_bytes % item_size == 0) break;
        }
    return {message_kind, inner_bytes, iterations, item_size};
}

std::vector<tensor_t> tiles(const hw_t &hw, layout_t a, layout_t b) {
    using tile_pair_t = std::array<tensor_t, 2>;
    auto max_size = max_bytes(hw, a, b);

    std::vector<dim_t> dims(a.ndims());
    for (dim_idx_t i = 0; i < a.ndims(); ++i)
        dims[i] = std::max(a.dim(i), b.dim(i));

    // Pad src/dst layouts to match each other.
    auto pad_layout = [&](layout_t &l) {
        std::vector<block_t> padded_blocks;
        for (auto &eb : l.enumerated_blocks()) {
            auto b = eb.second;
            if (l.is_outermost(eb)) {
                dim_t inner = l.dim(b.dim_idx) / b.block;
                b.block = ir_utils::safe_divide(dims[b.dim_idx], inner);
            }
            padded_blocks.push_back(b);
        }
        l = {l.type(), l.ndims(), 0, padded_blocks, /*do_normalize=*/false};
    };
    pad_layout(a);
    pad_layout(b);
    gpu_assert(ir_utils::is_equal(a.dims(), b.dims()));

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
            gpu_assert(d == 1);
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
        return can_be_mapped(a, t) && can_be_mapped(b, t);
    };

    auto merge_tiles = [](const tile_pair_t &p) {
        auto ndims = p[0].ndims() - 1;
        std::vector<dim_t> dims(ndims);
        for (dim_idx_t i = 0; i < ndims; ++i)
            dims[i] = std::max(p[0](i), p[1](i));
        return tensor_t(dims);
    };

    auto take_smaller = [](const tensor_t &a, const tensor_t &b) {
        return a.elems() < b.elems();
    };

    const auto eu_count = hw.eu_count();
    auto cmp = [&](const tensor_t &l, const tensor_t &r) {
        auto l_threads_reqd = a.elems() / l.elems();
        auto r_threads_reqd = a.elems() / r.elems();
        auto l_eu_util = utils::div_up(l_threads_reqd, eu_count);
        auto r_eu_util = utils::div_up(r_threads_reqd, eu_count);
        auto l_a_msg = estimate_message_info(hw, a, l);
        auto l_b_msg = estimate_message_info(hw, b, l);
        auto r_a_msg = estimate_message_info(hw, a, r);
        auto r_b_msg = estimate_message_info(hw, b, r);
        auto l_msg_load = l_a_msg.latency(hw) + l_b_msg.latency(hw);
        auto r_msg_load = r_a_msg.latency(hw) + r_b_msg.latency(hw);

        // Choose tiles with less message overhead per thread
        if (l_eu_util * l_msg_load != r_eu_util * r_msg_load)
            return (l_eu_util * l_msg_load < r_eu_util * r_msg_load);

        // Choose tiles with more bytes per message
        if (l.elems() * r_msg_load != r.elems() * l_msg_load)
            return (l.elems() * r_msg_load > r.elems() * l_msg_load);

        // If all else fails, go with the bigger tile
        return l.elems() > r.elems();
    };

    // Incrementally increase subtiles in a and b. The goal is to find the
    // maximum tiles so that the final combined tile covers dense regions as big
    // as possible in a/b layouts.
    std::vector<tensor_t> candidate_tiles;
    auto a_tiles = inner_tiles(a.blocks(), a.ndims()) | filter(mappable_tiles)
            | transform(add_pseudo_dimension(a));
    auto b_tiles = inner_tiles(b.blocks(), b.ndims()) | filter(mappable_tiles)
            | transform(add_pseudo_dimension(b));
    auto tiles = merge(a_tiles, b_tiles, take_smaller) | transform(merge_tiles);

    const int unit_size = std::max(a.type().size(), b.type().size());
    const dim_t max_units = max_size / unit_size;

    auto get_grf_layout_size = [&](const tensor_t &tile) {
        auto elems = tile.elems();
        dim_t grf_layout_size = 0;
        for (const auto &l : {a, b}) {
            auto info = estimate_message_info(hw, l, tile);
            int elem_size = std::max(info.item_size, 4);
            int elem_packing = info.item_size / l.type().size();
            auto layout_size = elem_size * elems / elem_packing;
            if (layout_size > grf_layout_size) grf_layout_size = layout_size;
        }
        return grf_layout_size;
    };

    for (auto tile : tiles) {
        if (tile.elems() > max_units) break;
        if (get_grf_layout_size(tile) > max_size) continue;
        if (candidate_tiles.empty() || !tile.is_equal(candidate_tiles.back()))
            candidate_tiles.push_back(tile);
    }
    gpu_assert(!candidate_tiles.empty());

    size_t best_idx = 0;
    for (size_t i = 0; i < candidate_tiles.size(); ++i)
        if (cmp(candidate_tiles[i], candidate_tiles[best_idx])) best_idx = i;
    candidate_tiles.resize(best_idx + 1);
    auto best = candidate_tiles.back();
    candidate_tiles.erase(
            std::remove_if(candidate_tiles.begin(), candidate_tiles.end(),
                    [&](const tensor_t &t) { return !best.is_divisible(t); }),
            candidate_tiles.end());
    candidate_tiles.shrink_to_fit();
    return candidate_tiles;
}

} // namespace reorder
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
