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

#include <algorithm>
#include <functional>
#include <utility>
#include "binding_axis.hpp"
#include "fusible_op.hpp"
#include "fusible_op_utils.hpp"
#include "graph.hpp"
#include "pass/pass.hpp"
#include "visitor.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// only hash topology of graph, excluding attr and so on
graph_identity get_identity_by_topology(const sc_graph_t &g) {
    graph_identity identity;
    size_t seed = 0;
    op_visitor_t vis = op_visitor_t::bfs_topology_sort(g.ops_.size());
    vis.visit_graph(
            g, [&seed, &identity](op_visitor_t *vis, const sc_op_ptr &op) {
                for (auto &in : op->get_inputs()) {
                    hash_combine(seed, in->details_.hash());
                    identity.op_lts_.emplace_back(in->details_);
                }
                for (auto &out : op->get_outputs()) {
                    hash_combine(seed, out->details_.hash());
                    identity.op_lts_.emplace_back(out->details_);
                }
                hash_combine(seed, op->op_name_);
                identity.op_names_.emplace_back(op->op_name_);
            });
    identity.hash_ = seed;
    return identity;
}

bool graph_identity::operator==(const graph_identity &other) const {
    return hash_ == other.hash_ && op_names_ == other.op_names_
            && op_lts_ == other.op_lts_;
}

void query_binding_axis(sc_graph_t &g) {
    // get hash of graph
    auto g_identity = get_identity_by_topology(g);
    // get global map cache of graph
    auto cache = g.attrs_.get_or_null<global_map_cache>(
            binding_axis_attr::global_map_cache);
    // auto skip, when cache is found and dont need to update
    if (cache && cache->identity_ == g_identity) { return; }
    // make shared ptr of global map
    auto shared_map_ptr = std::make_shared<global_binding_axis_map>();
    // global map
    auto &gax_map = *shared_map_ptr;
    // query function
    auto query_based_on_gt = [&gax_map](const graph_tensor_ptr &gt) {
        auto key = (uintptr_t)gt.get();
        // avoid repeated query
        if (gax_map.find(key) != gax_map.end()) return;
        // skip any format
        if (gt->details_.get_format().is_any()) return;
        // set initial axis
        int rank = gt->details_.get_plain_dims().size();
        binding_axis_map bd_ax_map;
        binding_axis init_axis;
        init_axis.reserve(rank);
        for (int i = 0; i < rank; i++) {
            init_axis.emplace_back(std::vector<int> {i});
        }
        bd_ax_map.get(gt) = init_axis;

        // start node producer recursively call pre-infer binding axis
        auto &producer_op = gt->producer_owner_;
        if (auto cur
                = producer_op
                          ->dyn_cast<op_traits::mixed_partition_acceptable>()) {
            cur->pre_infer_binding_axis(bd_ax_map);
        }

        // start node user recursively call infer binding axis
        for (auto &user : gt->uses_) {
            if (auto cur = user.second->dyn_cast<
                           op_traits::mixed_partition_acceptable>()) {
                cur->infer_binding_axis(bd_ax_map);
            }
        }

        // tranform key type
        std::unordered_map<uintptr_t, binding_axis> value_map;
        std::transform(bd_ax_map.datamap_.begin(), bd_ax_map.datamap_.end(),
                std::inserter(value_map, value_map.end()),
                [](const std::pair<graph_tensor *, binding_axis> &kv) {
                    return std::make_pair((uintptr_t)kv.first, kv.second);
                });

        // insert to global map
        gax_map[key] = std::move(value_map);
    };

    // visit all graph tensor
    for (auto &op : g.ops_) {
        if (op->isa<input_op>() || op->isa<output_op>()
                || op->isa<constant_op_t>())
            continue;
        for (auto &inp : op->get_inputs()) {
            query_based_on_gt(inp);
        }
        for (auto &out : op->get_outputs()) {
            query_based_on_gt(out);
        }
    }

    // attach to graph attr
    g.attrs_[binding_axis_attr::global_map_cache]
            = global_map_cache {shared_map_ptr, g_identity};
}

std::shared_ptr<global_binding_axis_map> get_binding_axis_map(sc_graph_t &g) {
    auto cache = g.attrs_.get_or_null<global_map_cache>(
            binding_axis_attr::global_map_cache);
    return cache ? cache->global_map_ptr_ : nullptr;
}

void bind_loop_axis(const graph_tensor_ptr &gt, const for_loop &loop,
        const std::vector<int> &axis, bool is_block) {
    auto &owner_graph = gt->producer_owner_->owner_graph_;
    COMPILE_ASSERT(owner_graph, "No owner graph is found")
    // get global map ptr
    auto global_ptr = get_binding_axis_map(*owner_graph);
    COMPILE_ASSERT(global_ptr, "No global mapping ptr found, please check")
    // set attr
    loop->attr()[binding_axis_attr::loop_hint]
            = loop_binding_axis_hint {global_ptr, (uintptr_t)gt.get(),
                    is_block ? transform_axis_blocking2plain(gt->details_, axis)
                             : axis};
}

void bind_loop_axis(
        const graph_tensor_ptr &gt, const stmt &loop, int axis, bool is_block) {
    COMPILE_ASSERT(
            loop.isa<for_loop>(), "loop node is expected, but got " << loop)
    bind_loop_axis(
            gt, loop.static_as<for_loop>(), std::vector<int> {axis}, is_block);
}

void bind_loop_axis(const graph_tensor_ptr &gt,
        const std::vector<for_loop> &loops, const std::vector<int> &axis,
        bool is_block) {
    COMPILE_ASSERT(loops.size() == axis.size(),
            "axis size should be equal to loop size, but got "
                    << axis.size() << " VS " << loops.size())
    for (size_t i = 0; i < loops.size(); i++) {
        bind_loop_axis(gt, loops[i], axis[i], is_block);
    }
}

std::vector<int> transform_binding_axis_with_indices(
        const binding_axis &axis, const std::vector<int> &indices) {
    std::vector<int> ret;
    for (auto &ax : indices) {
        COMPILE_ASSERT(ax < (int)axis.size(),
                "ax exceed size: " << ax << " >= " << axis.size())
        ret.insert(ret.end(), axis[ax].begin(), axis[ax].end());
    }

    // check if empty to make g++12 happy
    if (!ret.empty()) {
        // sort axis
        std::sort(ret.begin(), ret.end());
        // erase possible repeated elements in axis B
        ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
    }
    return ret;
}

bool get_aligned_binding_axis(const loop_binding_axis_hint &hint_a,
        const loop_binding_axis_hint &hint_b, std::vector<int> &axis_a,
        std::vector<int> &axis_b) {
    // reset
    axis_a.clear();
    axis_b.clear();

    COMPILE_ASSERT(hint_a.global_map_ptr_ && hint_b.global_map_ptr_,
            "global mapping ptr is null")
    // global mapping ptr must be equal, which means they are quried from the
    // same graph
    if (hint_a.global_map_ptr_ != hint_b.global_map_ptr_) { return false; }

    // if two keys are same
    if (hint_a.key_ == hint_b.key_) {
        axis_a = hint_a.axis_;
        axis_b = hint_b.axis_;
        return true;
    }
    // get binding axis of A from hint A
    axis_a = hint_a.axis_;
    // get global map
    auto &global_map = *hint_a.global_map_ptr_;

    // get mapping from view of hint B
    if (global_map.find(hint_b.key_) == global_map.end()) { return false; }
    auto &map_b = global_map[hint_b.key_];

    // query binding axis of A from hint B
    if (map_b.find(hint_a.key_) == map_b.end()) { return false; }
    auto &value_a = map_b[hint_a.key_];
    // use axis_a as indices to get axis_b from value_a
    axis_b = transform_binding_axis_with_indices(value_a, hint_b.axis_);

    return true;
}

bool get_aligned_binding_axis_twice(const loop_binding_axis_hint &hint_a,
        const loop_binding_axis_hint &hint_b, std::vector<int> &axis_a,
        std::vector<int> &axis_b) {
    return get_aligned_binding_axis(hint_a, hint_b, axis_a, axis_b)
            || get_aligned_binding_axis(hint_b, hint_a, axis_b, axis_a);
}

bool check_loop_binding_axis(
        const for_loop_node_t *loop_a, const for_loop_node_t *loop_b) {
    // get hint A and B respectively
    auto hint_a = any_map_t::fetch_or_null<loop_binding_axis_hint>(
            loop_a->attr_.get(), binding_axis_attr::loop_hint);
    auto hint_b = any_map_t::fetch_or_null<loop_binding_axis_hint>(
            loop_b->attr_.get(), binding_axis_attr::loop_hint);
    if (!hint_a || !hint_b) return false;

    std::vector<int> axis_a, axis_b;
    if (get_aligned_binding_axis_twice(*hint_a, *hint_b, axis_a, axis_b)) {
        return axis_a == axis_b;
    } else {
        return false;
    }
}

bool check_loop_binding_axis(const for_loop &loop_a, const for_loop &loop_b) {
    return check_loop_binding_axis(loop_a.get(), loop_b.get());
}

int check_loop_binding_axis(const std::vector<for_loop> &loop_a,
        const std::vector<for_loop> &loop_b, int64_t check_loop_size) {
    size_t gcs = std::min(loop_a.size(), loop_b.size());
    if (check_loop_size >= 0) { gcs = std::min(gcs, (size_t)check_loop_size); }
    int aligned_loop_num = 0;
    for (size_t i = 0; i < gcs; i++) {
        if (check_loop_binding_axis(loop_a[i], loop_b[i])) {
            aligned_loop_num++;
        } else {
            break;
        }
    }
    return aligned_loop_num;
}

bool check_loop_has_axis(const for_loop &loop, const graph_tensor_ptr &gt,
        const std::vector<int> &axis) {
    // get hint
    auto hint = any_map_t::fetch_or_null<loop_binding_axis_hint>(
            loop->attr_.get(), binding_axis_attr::loop_hint);
    if (!hint) return false;
    // get binding axis
    auto &self_axis = hint->axis_;
    // get global map
    COMPILE_ASSERT(hint->global_map_ptr_, "global mapping ptr is null")
    auto &global_map = *(hint->global_map_ptr_);

    // get mapping based on self hint key
    if (global_map.find(hint->key_) == global_map.end()) return false;
    auto &self_map = global_map[hint->key_];

    // get target key
    auto target_key = (uintptr_t)gt.get();
    // query binding axis
    if (self_map.find(target_key) == self_map.end()) return false;
    auto &target_binding_axis = self_map[target_key];
    // get target_axis based on self_axis from view of target graph tensor
    std::vector<int> target_axis = transform_binding_axis_with_indices(
            target_binding_axis, self_axis);

    // check any element of argument `axis` would appear among target axis
    return std::any_of(axis.begin(), axis.end(), [&target_axis](const int &ax) {
        return std::find(target_axis.begin(), target_axis.end(), ax)
                != target_axis.end();
    });
}

void copy_binding_axis_hint(
        const for_loop_node_t *ths, for_loop_node_t *other) {
    auto ptr = any_map_t::fetch_or_null<loop_binding_axis_hint>(
            ths->attr_.get(), binding_axis_attr::loop_hint);
    if (ptr) {
        auto loop_hint = *ptr;
        other->attr().set(binding_axis_attr::loop_hint, loop_hint);
    }
}

void fuse_binding_axis_hint(
        const for_loop_node_t *ths, const for_loop_node_t *other) {
    auto ths_hint_ptr = any_map_t::fetch_or_null<loop_binding_axis_hint>(
            ths->attr_.get(), binding_axis_attr::loop_hint);
    auto other_hint_ptr = any_map_t::fetch_or_null<loop_binding_axis_hint>(
            other->attr_.get(), binding_axis_attr::loop_hint);
    if (!ths_hint_ptr || !other_hint_ptr) return;

    loop_binding_axis_hint hint_ths = *ths_hint_ptr,
                           hint_other = *other_hint_ptr;

    auto merge_axis = [](const std::vector<int> &axis_a,
                              const std::vector<int> &axis_b) {
        auto ret = axis_a;
        ret.insert(ret.end(), axis_b.begin(), axis_b.end());
        // check if empty to make g++12 happy
        if (!ret.empty()) {
            std::sort(ret.begin(), ret.end());
            ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
        }
        return ret;
    };

    std::vector<int> axis_ths, axis_other;
    if (get_aligned_binding_axis(hint_ths, hint_other, axis_ths, axis_other)) {
        auto merged_axis = merge_axis(axis_ths, axis_other);
        hint_ths.axis_ = merged_axis;
    } else if (get_aligned_binding_axis(
                       hint_other, hint_ths, axis_other, axis_ths)) {
        auto merged_axis = merge_axis(axis_ths, axis_other);
        hint_other.axis_ = merged_axis;
    }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
