/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

namespace sc {
using namespace op_traits;
SC_INTERNAL_API sc_graph_t copy_graph(const sc_graph_t &graph) {
    for (auto &op : graph.ops_) {
        if (!op->is_removed_ && !op->dyn_cast<input_op>()
                && !op->dyn_cast<output_op>() && !op->dyn_cast<copyable_t>()) {
            return sc_graph_t();
        }
    }
    sc_graph_t copied_graph;
    op_visitor_t vis(op_visitor_t::dequeue_selector,
            op_visitor_t::create_DAG_updater(graph.ops_.size()));
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> old_new_lt_map;
    std::unordered_map<sc_op_ptr, int> op_id_map;
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
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
            }
        }
        // recording old graph_tensor->new graph_tensor
        for (size_t i = 0; i < new_node->get_outputs().size(); ++i) {
            old_new_lt_map[node->get_outputs()[i]] = new_node->get_outputs()[i];
        }
        op_id_map[new_node] = node->logical_op_id_;
    });
    copied_graph.attrs_ = graph.attrs_;
    // deep copy here.
    if (graph.dyn_info_) {
        copied_graph.dyn_info_
                = std::make_shared<dynamic_lower_info_t>(*graph.dyn_info_);
    }
    copied_graph.resort_op_ids(op_id_map);
    return copied_graph;
}
} // namespace sc
