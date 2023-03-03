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

#include <utility>

#include <functional>
#include <limits>
#include "../dynamic_dispatch_key.hpp"
#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../tunable_op.hpp"
#include "../visitor.hpp"
#include <compiler/ir/graph/dynamic_lower_info.hpp>
#include <ops/fusible/memory_movement.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(graph.shape_relationship_binding);
struct dyn_plhd_union_t {
    dyn_plhd_union_t(sc_dim cur_plhd) {
        count_ = dynamic_lower_info_t::init_placeholder - cur_plhd;
        parent_.resize(count_);
        for (size_t i = 0; i < parent_.size(); i++) {
            parent_[i] = i;
        }
    }
    sc_dim to_idx(sc_dim in) {
        return -1 * (in - dynamic_lower_info_t::init_placeholder);
    }
    sc_dim to_plhd(sc_dim in) {
        return -1 * in + dynamic_lower_info_t::init_placeholder;
    }
    sc_dim find(sc_dim in) { return to_plhd(find_impl(to_idx(in))); }
    void merge(sc_dim in1, sc_dim in2) {
        assert(is_dynamic_dim(in1) || is_dynamic_dim(in2));
        if (in1 > 0) {
            static_map_.insert(std::make_pair(find(in2), in1));
            return;
        }
        if (in2 > 0) {
            static_map_.insert(std::make_pair(find(in1), in2));
            return;
        }
        merge_impl(to_idx(in1), to_idx(in2));
    }
    sc_dim get_count() const { return count_; }
    const std::unordered_map<sc_dim, sc_dim> &get_static_map() {
        for (auto it = static_map_.begin(); it != static_map_.end();) {
            auto p = find(it->first);
            if (p != it->first) {
                static_map_.insert(std::make_pair(p, it->second));
                it = static_map_.erase(it);
            } else {
                it++;
            }
        };
        return static_map_;
    }

private:
    std::vector<sc_dim> parent_;
    sc_dim count_;
    // parent to static value map
    std::unordered_map<sc_dim, sc_dim> static_map_;
    sc_dim find_impl(sc_dim in) {
        if (parent_[in] == in) { return in; }
        parent_[in] = find_impl(parent_[in]);
        return parent_[in];
    }
    void merge_impl(sc_dim in1, sc_dim in2) {
        sc_dim p1 = find_impl(in1), p2 = find_impl(in2);
        if (p1 == p2) { return; }
        parent_[p2] = p1;
        count_--;
    }
};

SC_INTERNAL_API void shape_relationship_binding(
        sc_graph_t &graph, const context_ptr &ctx) {
    if (graph.empty() || !graph.is_dynamic() || !graph.dyn_info_) { return; }
    // stage 1, first satisfy all dynamic dimensions are placeholder not any.
    for (auto &node : graph.ops_) {
        auto ins = node->get_inputs();
        auto outs = node->get_outputs();
        for (auto &in : ins) {
            auto plain_dims = in->details_.get_plain_dims();
            for (auto &dim : plain_dims) {
                if (dim == dimensions::dynamic_any) {
                    dim = graph.get_next_dynamic_placeholder();
                }
            }
            in->details_.set_plain_dims(plain_dims);
        }
        for (auto &out : outs) {
            auto plain_dims = out->details_.get_plain_dims();
            for (auto &dim : plain_dims) {
                if (dim == dimensions::dynamic_any) {
                    dim = graph.get_next_dynamic_placeholder();
                }
            }
            out->details_.set_plain_dims(plain_dims);
        }
    }
    // stage 2, use disjoint-set to merge all dynamic placeholders.
    // initialize disjoint-set.
    int cur_old_plhd = graph.dyn_info_->cur_dynamic_placeholder_;
    dyn_plhd_union_t dyn_un(cur_old_plhd);
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(graph.ops_.size());
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        std::vector<std::pair<sc_dim, sc_dim>> shape_relations
                = node->get_dynamic_shape_relations();
        for (auto &it : shape_relations) {
            dyn_un.merge(it.first, it.second);
        }
    });
    // stage 3: construct a map of old placeholder to new.
    // map of parent to new.
    std::unordered_map<sc_dim, sc_dim> parent_map;
    // map of old placeholder to new.
    std::unordered_map<sc_dim, sc_dim> plhd_map;
    sc_dim cur_new_plhd = dynamic_lower_info_t::init_placeholder;
    auto plhd_count = dyn_un.get_count();
    auto static_map = dyn_un.get_static_map();

    parent_map.insert(static_map.begin(), static_map.end());
    plhd_count -= static_map.size();
    for (sc_dim old_plhd = dynamic_lower_info_t::init_placeholder;
            old_plhd > cur_old_plhd; old_plhd--) {
        auto it = plhd_map.find(old_plhd);
        if (it == plhd_map.end()) {
            auto p = dyn_un.find(old_plhd);
            auto pit = parent_map.find(p);
            if (pit == parent_map.end()) {
                parent_map[p] = cur_new_plhd;
                plhd_map[old_plhd] = cur_new_plhd;
                cur_new_plhd--;
                plhd_count--;
            } else {
                plhd_map[old_plhd] = pit->second;
            }
        }
    }
    assert(plhd_count == 0);
    SC_UNUSED(plhd_count);
    // stage 4: replace old placeholders in graph with map.
    std::unordered_map<graph_tensor_ptr, bool> visited;
    op_visitor_t vis2 = op_visitor_t::dfs_topology_sort(graph.ops_.size());
    vis2.visit_graph(graph, [&](op_visitor_t *vis2, const sc_op_ptr &node) {
        auto ins = node->get_inputs();
        auto outs = node->get_outputs();
        SC_MODULE_INFO << "Op " << node->op_name_ << "\n Inputs:";
        for (auto &in : ins) {
            auto plain_dims = in->details_.get_plain_dims();
            if (!visited[in]) {
                std::for_each(plain_dims.begin(), plain_dims.end(),
                        [&plhd_map](sc_dim &in) {
                            if (is_dynamic_dim(in)) {
                                auto it = plhd_map.find(in);
                                assert(it != plhd_map.end());
                                in = it->second;
                            }
                        });
                in->details_.set_plain_dims(plain_dims);
                visited[in] = true;
            }
            SC_MODULE_INFO << utils::print_vector(plain_dims);
        }
        SC_MODULE_INFO << "Outputs: ";
        for (auto &out : outs) {
            auto plain_dims = out->details_.get_plain_dims();
            if (!visited[out]) {
                std::for_each(plain_dims.begin(), plain_dims.end(),
                        [&plhd_map](sc_dim &in) {
                            if (is_dynamic_dim(in)) {
                                auto it = plhd_map.find(in);
                                assert(it != plhd_map.end());
                                in = it->second;
                            }
                        });
                out->details_.set_plain_dims(plain_dims);
                visited[out] = true;
            }
            SC_MODULE_INFO << utils::print_vector(plain_dims);
        }
    });
    // stage 5: remove old extra placeholder info.
    graph.dyn_info_->cur_dynamic_placeholder_ = cur_new_plhd;
    for (sc_dim old_plhd = cur_new_plhd; old_plhd > cur_old_plhd; old_plhd--) {
        graph.dyn_info_->dim2expr_map_.erase(old_plhd);
    }
    graph.reset_op_ids();
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
