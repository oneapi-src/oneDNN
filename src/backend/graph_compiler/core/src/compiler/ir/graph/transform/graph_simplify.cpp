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

#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <unordered_map>

SC_MODULE(graph.simplify)

namespace sc {
struct hash_sc_op_t {
    std::size_t operator()(const sc_op_ptr &v) const {
        size_t hash_ = 0;
        hash_combine(hash_, v->op_name_);
        hash_combine(hash_, v->info_.outputs_[0]->details_);
        hash_combine(hash_, v->hash_contents());
        return hash_;
    }
};

struct compare_sc_op_t {
    bool operator()(const sc_op_ptr &v0, const sc_op_ptr &v1) const {
        return v0->op_name_ == v1->op_name_
                && v0->info_.outputs_[0]->details_
                == v1->info_.outputs_[0]->details_
                && v0->compare_contents(v1.get());
    }
};

static void insert_tensor_view_op(sc_graph_t &graph, graph_tensor_ptr in,
        size_t in_index, const sc_op_ptr &cur_op) {
    auto ret = graph.make("tensor_view", {std::move(in)}, {},
            {{"shape", in->details_.get_plain_dims()}});
    cur_op->replace_input(in_index, ret->get_outputs()[0]);
}

void drop_same_op_on_output(sc_graph_t &graph, const graph_tensor_ptr &output) {
    std::unordered_map<sc_op_ptr, std::vector<int>, hash_sc_op_t,
            compare_sc_op_t>
            same_op_map;
    for (size_t i = 0; i < output->uses_.size(); i++) {
        auto node = output->uses_[i];
        if (node.second->get_inputs().empty()
                || node.second->get_outputs().empty()) {
            continue;
        }
        if (node.second->get_inputs().size() > 1
                || node.second->get_outputs().size() > 1) {
            SC_MODULE_INFO
                    << "Currently we don't support multi-input/multi-output op "
                       "elimination.";
            continue;
        }
        same_op_map[node.second].push_back(i);
    }
    std::vector<std::pair<int, sc_op_ptr>> next_nodes(
            output->uses_.begin(), output->uses_.end());
    for (auto &it : same_op_map) {
        if (it.second.size() > 1) {
            auto reserve_node = next_nodes[it.second[0]].second;
            std::vector<sc_op_ptr> del_node_list;
            for (size_t i = 1; i < it.second.size(); i++) {
                if (it.second[i] >= static_cast<int>(next_nodes.size())
                        || next_nodes[it.second[i]]
                                   .second->get_outputs()[0]
                                   ->uses_.empty()) {
                    break;
                }
                auto del_node = next_nodes[it.second[i]].second;
                std::vector<std::pair<int, sc_op_weak_ptr_t>> del_uses
                        = del_node->get_outputs()[0]->uses_;
                for (size_t u = 0; u < del_uses.size(); u++) {
                    auto node_after_del = del_uses[u];
                    node_after_del.second->replace_input(node_after_del.first,
                            reserve_node->get_outputs()[0]);
                }
                del_node_list.push_back(del_node);
            }
            for (auto &del_node : del_node_list) {
                del_node->remove();
            }
        }
    }
}

// eliminate horizontal same ops, e.g. qkv input reorder
void horizontal_same_op_elimination(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        if (!node->isa<output_op>()) {
            for (size_t i = 0; i < node->get_outputs().size(); i++) {
                auto output = node->get_outputs()[i];
                drop_same_op_on_output(graph, output);
            }
        }
    });
    graph.reset_op_ids();
}

static bool is_single_use(const sc_op_ptr &node) {
    return node->get_outputs()[0]->uses_.size() == 1;
}

// eliminate excess tensor view, e.g. tensor_view->tensor_view->tensor_view
void excess_tensor_view_elimination(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        if (node->isa<tensor_view_op_t>() && is_single_use(node)) {
            sc_op_ptr next_node
                    = node->get_outputs()[0]->uses_[0].second.get_shared();
            sc_op_ptr pre_node = next_node;
            std::vector<sc_op_ptr> node_to_remove;
            while (next_node->isa<tensor_view_op_t>()
                    && is_single_use(next_node)) {
                node_to_remove.push_back(next_node);
                pre_node = next_node;
                next_node = next_node->get_outputs()[0]
                                    ->uses_[0]
                                    .second.get_shared();
            }
            if (next_node->isa<tensor_view_op_t>()) { pre_node = next_node; }
            if (pre_node != next_node || pre_node->isa<tensor_view_op_t>()) {
                node->get_outputs()[0]->details_
                        = pre_node->get_outputs()[0]->details_;
                std::vector<std::pair<int, sc_op_weak_ptr_t>> uses
                        = pre_node->get_outputs()[0]->uses_;
                for (size_t i = 0; i < uses.size(); i++) {
                    int pre_idx = uses[i].first;
                    uses[i].second->replace_input(
                            pre_idx, node->get_outputs()[0]);
                }
                for (auto &del_node : node_to_remove) {
                    del_node->remove();
                }
                if (pre_node->isa<tensor_view_op_t>()) { pre_node->remove(); }
            }
            vis.update_state_for_visited(node);
        }
    });
    graph.reset_op_ids();
}

// eliminate redundant binary op
void redundant_binary_op_elimination(
        sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        // x + 0 or 0 + x or x * 1 or 1 * x
        if (node->isa<add_op_t>() || node->isa<mul_op_t>()) {
            for (size_t i = 0; i < node->get_inputs().size(); i++) {
                if (node->get_inputs()[i]
                                ->producer_owner_->isa<constant_op_t>()) {
                    auto in_const_op = node->get_inputs()[i]
                                               ->producer_owner_
                                               ->dyn_cast<constant_op_t>();
                    float v = reinterpret_cast<float *>(
                            in_const_op->get_constant_values()->data_)[0];
                    if (((v == 0.f && node->isa<add_op_t>())
                                || (v == 1.f && node->isa<mul_op_t>()))
                            && in_const_op->get_constant_plain_dims().size()
                                    == 1
                            && in_const_op->get_constant_plain_dims()[0] == 1) {
                        size_t use_size = node->get_outputs()[0]->uses_.size();
                        for (size_t j = 0; j < use_size; j++) {
                            sc_op_ptr next_node = node->get_outputs()[0]
                                                          ->uses_[j]
                                                          .second.get_shared();
                            int idx = node->get_outputs()[0]->uses_[j].first;
                            if (next_node->isa<output_op>()
                                    && node->get_inputs()[1 - i]
                                               ->producer_owner_
                                               ->isa<input_op>()) {
                                insert_tensor_view_op(graph,
                                        node->get_inputs()[1 - i], idx,
                                        next_node);
                            } else {
                                next_node->replace_input(
                                        idx, node->get_inputs()[1 - i]);
                            }
                        }
                        node->remove();
                        if (in_const_op->get_outputs()[0]->uses_.empty()) {
                            in_const_op->remove();
                        }
                        break;
                    }
                }
            }
        }
        // x - 0 or x / 1
        else if (node->isa<sub_op_t>() || node->isa<div_op_t>()) {
            if (node->get_inputs()[1]->producer_owner_->isa<constant_op_t>()) {
                auto in_const_op
                        = node->get_inputs()[1]
                                  ->producer_owner_->dyn_cast<constant_op_t>();
                float v = reinterpret_cast<float *>(
                        in_const_op->get_constant_values()->data_)[0];
                if (((v == 0.f && node->isa<sub_op_t>())
                            || (v == 1.f && node->isa<div_op_t>()))
                        && in_const_op->get_constant_plain_dims().size() == 1
                        && in_const_op->get_constant_plain_dims()[0] == 1) {
                    size_t use_size = node->get_outputs()[0]->uses_.size();
                    for (size_t j = 0; j < use_size; j++) {
                        sc_op_ptr next_node = node->get_outputs()[0]
                                                      ->uses_[j]
                                                      .second.get_shared();
                        int idx = node->get_outputs()[0]->uses_[j].first;
                        if (next_node->isa<output_op>()
                                && node->get_inputs()[0]
                                           ->producer_owner_->isa<input_op>()) {
                            insert_tensor_view_op(graph, node->get_inputs()[0],
                                    idx, next_node);
                        } else {
                            next_node->replace_input(
                                    idx, node->get_inputs()[0]);
                        }
                    }
                    node->remove();
                    if (in_const_op->get_outputs()[0]->uses_.empty()) {
                        in_const_op->remove();
                    }
                }
            }
        }
    });
    graph.reset_op_ids();
}

void graph_simplify(sc_graph_t &graph, const context_ptr &ctx) {
    redundant_binary_op_elimination(graph, ctx);
    excess_tensor_view_elimination(graph, ctx);
    horizontal_same_op_elimination(graph, ctx);
}

} // namespace sc
