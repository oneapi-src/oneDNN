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

#include <utility>
#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../tunable_op.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/fusible/unary_elemwise.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

template <typename opT>
static void collect_linked_ops_nearby_input_until_opT(sc_graph_t &graph,
        const context_ptr &ctx,
        std::vector<std::vector<sc_op_ptr>> &target_ops_list) {
    auto vis = op_visitor_t::bfs();
    constexpr int max_step = 5;
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (!node->isa<input_op>()) return;
        for (auto &user : node->get_outputs()[0]->uses_) {
            sc_op_ptr next_node = user.second;
            int step = 1;
            std::vector<sc_op_ptr> target_ops;
            bool found = false;
            while (next_node->is_single_output_single_use()
                    && (!next_node->get_outputs()[0]
                                    ->uses_[0]
                                    .second->isa<output_op>())
                    && (next_node->attrs_.get_or_else(
                                "constant", const_kind::not_const)
                            == const_kind::not_const)) {
                target_ops.emplace_back(next_node);
                if (next_node->isa<opT>()) {
                    found = true;
                    break;
                }
                if ((step++) >= max_step) return;
                next_node = next_node->get_outputs()[0]->uses_[0].second;
            }
            if (found && target_ops.size() > 1) {
                target_ops_list.emplace_back(target_ops);
            }
        }
    });
}

/**
 * The reschedule bypass rule is:
 * 1. upwards visit.
 * 2. scan bypass ops list: find the first non-elementwise op and try to swap
 * reorder op position with it firstly in avoid of unnecessay fusion break.
 * */
static void reschedule_reorder_nearby_input(std::vector<sc_op_ptr> target_ops) {
    if (target_ops.empty()) return;
    // first op in bypass list
    auto begin_op = target_ops.front();
    auto end_op = target_ops.back();
    COMPILE_ASSERT(end_op->isa<reorder_op_t>(),
            "tensorview op is expected, but got " << end_op->op_name_ << "_"
                                                  << end_op->logical_op_id_)
    // support automatically position switch for unary and special
    // broadcast ops
    auto last_reo = end_op->dyn_cast<reorder_op_t>();
    sc_op *pre_op = end_op.get();
    sc_data_type_t cached_dtype = end_op->get_inputs()[0]->details_.dtype_;
    // search the swtichable ops
    int i = target_ops.size() - 1;
    int j = i;
    std::unordered_map<int, int> idx_mp;
    while (pre_op != begin_op.get()) {
        pre_op = target_ops[--i].get();
        if (pre_op->isa<unary_elementwise_op_t>()) {
            if (pre_op->isa<cast_op_t>()) {
                if (utils::get_sizeof_type(
                            pre_op->get_inputs()[0]->details_.dtype_)
                        > utils::get_sizeof_type(
                                pre_op->get_outputs()[0]->details_.dtype_)) {
                    break;
                }
                cached_dtype = pre_op->get_inputs()[0]->details_.dtype_;
            }
            idx_mp[i] = 0;
            j = i;
        } else if (pre_op->isa<binary_elementwise_op_t>()) {
            int bc_idx = pre_op->dyn_cast<binary_elementwise_op_t>()
                                 ->get_broadcast_input();
            if (bc_idx < 0) break;
            if (auto const_op
                    = pre_op->get_inputs()[bc_idx]
                              ->producer_owner_->dyn_cast<constant_op_t>()) {
                auto dims = const_op->get_constant_blocking_dims();
                if (dims.size() == 1 && dims[0] == 1) {
                    idx_mp[i] = 1 - bc_idx;
                    j = i;
                    continue;
                }
            }
            break;
        } else {
            break;
        }
    }
    pre_op = target_ops[j].get();

    // switch pre_op and last_reo position
    if (pre_op == begin_op.get()) {
        auto new_fmt = last_reo->get_outputs()[0]->details_.get_format();
        // disconnect reorder
        last_reo->get_outputs()[0]->replace_with(
                target_ops[target_ops.size() - 2]->get_outputs()[0]);
        // insert reorder
        last_reo->replace_input(0, pre_op->get_inputs()[idx_mp[j]]);
        pre_op->replace_input(idx_mp[j], last_reo->get_outputs()[0]);

        // reset attribute
        last_reo->info_.outputs_[0]->details_.dtype_ = cached_dtype;
        // update output format
        for (int k = j; k < static_cast<int>(target_ops.size()) - 1; k++) {
            target_ops[k]->get_outputs()[0]->details_.set_format(new_fmt);
        }
    }
}

static void nearby_input_reorder_rule(
        sc_graph_t &graph, const context_ptr &ctx) {
    std::vector<std::vector<sc_op_ptr>> target_ops_list;
    collect_linked_ops_nearby_input_until_opT<reorder_op_t>(
            graph, ctx, target_ops_list);
    // reschedule all bypass ops
    for (auto &target_ops : target_ops_list) {
        reschedule_reorder_nearby_input(target_ops);
    }
}

using gr_rule = std::function<void(sc_graph_t &graph, const context_ptr &ctx)>;

void global_reschedule(sc_graph_t &graph, const context_ptr &ctx) {
    if (!graph.attrs_.get_or_else("temp.fuse", 1)) { return; }
    if (graph.is_dynamic()) { return; }
    std::vector<gr_rule> gr_rule_list;
    gr_rule_list.emplace_back(nearby_input_reorder_rule);
    for (auto &gr_rule : gr_rule_list) {
        gr_rule(graph, ctx);
    }
    graph.reset_op_ids();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
