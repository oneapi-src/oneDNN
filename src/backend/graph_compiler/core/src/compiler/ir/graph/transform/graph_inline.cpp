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

#include <set>
#include <vector>
#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <unordered_map>

namespace sc {

static void do_inline_graph(
        sc_graph_t &full_graph, sc_op_ptr &cur_node, sc_graph_t &sub_graph) {
    std::unordered_map<sc_op_ptr, std::vector<sc_op_ptr>> *tunable_op_map;
    sc_op_ptr corresponding_node;
    bool need_tuning = full_graph.attrs_.has_key("temp.op_map");
    if (need_tuning) {
        tunable_op_map = full_graph.attrs_.get_or_null<
                std::unordered_map<sc_op_ptr, std::vector<sc_op_ptr>>>(
                "temp.op_map");
        if (tunable_op_map->find(cur_node) != tunable_op_map->end()) {
            corresponding_node = (*tunable_op_map)[cur_node][0];
            (*tunable_op_map)[corresponding_node].clear();
        }
    }
    int cur_op_id = cur_node->logical_op_id_;
    for (auto &op : sub_graph.ops_) {
        if (op->isa<input_op>()) {
            auto cur_op_ori_ins = cur_node->get_inputs();
            COMPILE_ASSERT(cur_op_ori_ins.size() == op->get_outputs().size(),
                    "cur_node " << cur_node->op_name_
                                << " 's input size should be equal with its "
                                   "sub_graph input op 's output size");
            if (cur_node->attrs_.get_or_else(op_attr_key::break_pre_fuse, false)
                    || cur_node->attrs_.get_or_else(
                            op_attr_key::no_fuse, false)) {
                for (auto &cur : op->get_outputs()) {
                    for (auto &u : cur->uses_) {
                        auto user_op = u.second;
                        bool need_break = true;
                        for (auto &in : user_op->get_inputs()) {
                            if (!(in->producer_owner_->isa<input_op>()
                                        || in->producer_owner_
                                                   ->isa<constant_op_t>())) {
                                need_break = false;
                                break;
                            }
                        }
                        u.second->attrs_.set(
                                op_attr_key::break_pre_fuse, need_break);
                    }
                }
            }
            for (size_t i = 0; i < op->get_outputs().size(); ++i) {
                op->get_outputs()[i]->replace_with(cur_op_ori_ins.at(i));
            }
            op->remove();
        } else if (op->isa<output_op>()) {
            auto cur_op_ori_outs = cur_node->get_outputs();
            COMPILE_ASSERT(cur_op_ori_outs.size() == op->get_inputs().size(),
                    "cur_node " << cur_node->op_name_
                                << " 's output size should be equal with its "
                                   "sub_graph output op's input size");
            if (cur_node->attrs_.get_or_else(
                        op_attr_key::break_post_fuse, false)
                    || cur_node->attrs_.get_or_else(
                            op_attr_key::no_fuse, false)) {
                for (auto &cur : op->get_inputs()) {
                    cur->producer_owner_->attrs_.set(
                            op_attr_key::break_post_fuse, true);
                }
            }
            for (size_t i = 0; i < cur_op_ori_outs.size(); ++i) {
                for (auto &use = cur_op_ori_outs[i]->uses_.front();
                        !cur_op_ori_outs[i]->uses_.empty();) {
                    use.second->replace_input(use.first, op->get_inputs()[i]);
                }
            }
            op->remove();
        } else {
            if (op->isa<op_traits::configurable_t>() && need_tuning) {
                (*tunable_op_map)[corresponding_node].push_back(op);
            }
            full_graph.ops_.emplace_back(op);
        }
    }
}

const std::set<std::string> &get_op_blocked_lists() {
    static std::set<std::string> blocked_list {
            "quantize", "dequantize", "dynamic_quantize", "dynamic_dequantize"};
    return blocked_list;
}

void graph_inline(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    auto &blocked_list = get_op_blocked_lists();
    vis.visit_graph(graph, [&](sc_op_ptr node) {
        if (auto graph_node = node->dyn_cast<graph_op_t>()) {
            if (blocked_list.find(node->op_name_) == blocked_list.end()) {
                auto sub_graph = graph_node->get_graph();
                vis.update_state_for_visited(node);
                do_inline_graph(graph, node, *sub_graph);
                node->remove();
            }
        }
    });
    graph.reset_op_ids();
}

namespace quantize {
void quantize_inline(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](sc_op_ptr node) {
        if (auto graph_node = node->dyn_cast<graph_op_t>()) {
            auto sub_graph = graph_node->get_graph();
            vis.update_state_for_visited(node);
            do_inline_graph(graph, node, *sub_graph);
            node->remove();
        }
    });
    graph.reset_op_ids();
}
} // namespace quantize

} // namespace sc
