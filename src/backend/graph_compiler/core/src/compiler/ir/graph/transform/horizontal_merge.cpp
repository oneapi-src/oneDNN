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

#include <map>
#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <compiler/ir/graph/fused_op.hpp>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static std::map<int, std::vector<sc_op_ptr>> get_merge_map(sc_graph_t &graph) {
    auto vis = op_visitor_t::bfs();
    std::map<int, std::vector<sc_op_ptr>> to_merge;
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (node->attrs_.get_or_else(
                    "horizontal_merge", horizontal_merge_type::no_merge)
                != horizontal_merge_type::no_merge) {
            to_merge[node->attrs_.get<int>("horizontal_merge")].push_back(node);
        }
    });
    return to_merge;
}

static void do_horizontal_merge(
        sc_graph_t &graph, const std::vector<sc_op_ptr> &merge_list) {
    std::vector<graph_tensor_ptr> merged_ins, merged_outs;
    std::vector<sc_op_ptr> copied_ops;
    std::string name;
    horizontal_ops_idx_list ops_idx_list;

    for (auto &op : merge_list) {
        auto ins = op->get_inputs();
        auto outs = op->get_outputs();
        auto new_outs = graph_op_t::remake_logical_tensors(outs);
        std::vector<int> ins_idx;
        std::vector<int> outs_idx;
        for (auto &in : ins) {
            auto it = std::find(merged_ins.begin(), merged_ins.end(), in);
            if (it == merged_ins.end()) {
                ins_idx.push_back(static_cast<int>(merged_ins.size()));
                merged_ins.push_back(in);
            } else {
                ins_idx.push_back(it - merged_ins.begin());
            }
        }
        for (auto &out : new_outs) {
            outs_idx.push_back(static_cast<int>(merged_outs.size()));
            merged_outs.push_back(out);
        }
        ops_idx_list.emplace_back(
                std::make_pair(op, std::make_tuple(ins_idx, outs_idx)));
        name += op->op_name_ + "_";
    }
    auto merged_op = std::make_shared<horizontal_fused_op_t>(
            "horizontal_fused_" + name, ops_idx_list, merged_ins, merged_outs,
            any_map_t());
    graph.add(merged_op);
    size_t output_offset = 0;
    for (auto &op : merge_list) {
        for (size_t i = 0; i < op->get_outputs().size(); i++) {
            auto &ths_out = op->get_outputs()[i];
            auto &replace_out = merged_op->get_outputs()[i + output_offset];
            ths_out->replace_with(replace_out);
        }
        output_offset += op->get_outputs().size();
        op->remove();
    }
}

void horizontal_merge(sc_graph_t &graph, const context_ptr &ctx) {
    if (!graph.attrs_.get_or_else("temp.fuse", 1)) { return; }
    auto to_merge = get_merge_map(graph);
    for (auto &merge_list : to_merge) {
        do_horizontal_merge(graph, merge_list.second);
    }
    graph.reset_op_ids();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
