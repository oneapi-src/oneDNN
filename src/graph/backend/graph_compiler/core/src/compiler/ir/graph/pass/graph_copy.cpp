/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include "../dynamic_lower_info.hpp"
#include "../fusible_op.hpp"
#include "../traits.hpp"
#include "../visitor.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
using namespace op_traits;
SC_INTERNAL_API sc_graph_t copy_graph(const sc_graph_t &graph) {
    for (auto &op : graph.ops_) {
        if (!op->is_removed_ && !op->dyn_cast<input_op>()
                && !op->dyn_cast<output_op>() && !op->dyn_cast<copyable_t>()) {
            return sc_graph_t();
        }
    }
    sc_graph_t copied_graph;
    op_visitor_t vis = op_visitor_t::bfs_topology_sort(graph.ops_.size());
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> old_new_lt_map;
    std::unordered_map<sc_op_ptr, int> op_id_map;
    // the map from old op id to new op
    std::vector<sc_op_ptr> old_id_2_new_op;
    old_id_2_new_op.resize(graph.ops_.size());
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        sc_op_ptr new_node;
        if (node->dyn_cast<input_op>()) {
            new_node = copied_graph.make_input(
                    copy_logical_tsr(node->get_outputs()));

            // "unique_id" for integration
            new_node->attrs_ = node->attrs_;
        } else {
            std::vector<graph_tensor_ptr> ins;
            ins.reserve(node->get_inputs().size());
            for (auto &t : node->get_inputs()) {
                ins.emplace_back(old_new_lt_map.at(t));
            }
            if (node->dyn_cast<output_op>()) {
                new_node = copied_graph.make_output(ins);
                // "unique_id" for integration
                new_node->attrs_ = node->attrs_;
            } else {
                new_node = node->dyn_cast<op_traits::copyable_t>()->copy(ins,
                        copy_logical_tsr(node->get_outputs()), copied_graph);
                new_node->info_.cur_impl_ = node->info_.cur_impl_;
            }
        }
        // recording old graph_tensor->new graph_tensor
        for (size_t i = 0; i < new_node->get_outputs().size(); ++i) {
            old_new_lt_map[node->get_outputs()[i]] = new_node->get_outputs()[i];
        }
        op_id_map[new_node] = node->logical_op_id_;
        old_id_2_new_op[node->logical_op_id_] = new_node;
    });
    // update the uses order, it is important for checking equality of the
    // copied graph
    for (auto &newop : copied_graph.ops_) {
        auto &mapped_old = graph.ops_[op_id_map[newop]];
        auto &newouts = newop->get_outputs();
        auto &oldouts = mapped_old->get_outputs();
        for (size_t i = 0; i < newouts.size(); i++) {
            // copy the old uses, and re-map to new ops
            newouts[i]->uses_ = oldouts.at(i)->uses_;
            for (auto &use : newouts[i]->uses_) {
                auto old_id = use.second.lock()->logical_op_id_;
                use.second = old_id_2_new_op[old_id];
            }
        }
    }
    copied_graph.attrs_ = graph.attrs_;
    // deep copy here.
    if (graph.dyn_info_) {
        copied_graph.dyn_info_
                = std::make_shared<dynamic_lower_info_t>(*graph.dyn_info_);
    }
    copied_graph.resort_op_ids(op_id_map);
    return copied_graph;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
