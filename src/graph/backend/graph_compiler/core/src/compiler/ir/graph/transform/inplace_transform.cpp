/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#include "../fusible_op.hpp"
#include "../graph.hpp"
#include "../visitor.hpp"
#include <ops/fusible/memory_movement.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
enum inplace_status : int { no_linked, linked_output, directly_linked_output };
bool is_copy_reorder(const sc_op_ptr &node) {
    return node->isa<reorder_op_t>()
            && node->attrs_.get_or_else("actually_copy", false);
}

sc_op_ptr skip_tensor_view(const sc_op_ptr &node, int &index) {
    auto cur_node = node;
    while (cur_node->isa<tensor_view_op_t>()
            && cur_node->is_single_output_single_use()) {
        index = cur_node->get_outputs()[0]->uses_[0].first;
        cur_node = cur_node->get_outputs()[0]->uses_[0].second.lock();
    }
    return cur_node;
}
bool has_linked_input(const sc_op_ptr &node) {
    if (node->isa<input_op>()) { return true; }
    auto cur_node = node.get();
    while (cur_node->isa<tensor_view_op_t>()) {
        cur_node = cur_node->get_inputs()[0]->producer_owner_;
    }
    return cur_node->isa<input_op>();
}

// return if the copy_reorder is linked to output.
inplace_status has_linked_output_and_modify_copy(int out_idx,
        const sc_op_ptr &node, bool is_del = true, bool force_insert = false,
        bool force_delete = false) {
    auto cur_node = skip_tensor_view(node, out_idx);
    if (is_del && is_copy_reorder(cur_node)) {
        assert(cur_node->get_outputs()[0]->uses_.size() == 1);
        auto next_node = cur_node->get_outputs()[0]->uses_[0].second.lock();
        // skip tensor view
        auto nnext_node = skip_tensor_view(next_node, out_idx);
        if (nnext_node->isa<output_op>() && !force_delete) {
            return linked_output;
        }
        // delete copy.
        next_node->replace_input(cur_node->get_outputs()[0]->uses_[0].first,
                cur_node->get_inputs()[0]);
        cur_node->remove();
        return nnext_node->isa<output_op>() ? linked_output : no_linked;
    } else if (!is_del && force_insert && cur_node->isa<output_op>()) {
        // insert copy
        sc_graph_t &graph = node->get_owner_graph();
        auto old_ltsr = cur_node->get_inputs()[out_idx];
        auto cp_reorder = graph.make("reorder", {old_ltsr}, {},
                {{"internal", true}, {"actually_copy", true},
                        {"out_format", old_ltsr->details_.get_format()}});
        cur_node->replace_input(out_idx, cp_reorder->get_outputs()[0]);
        return linked_output;
    }
    // judge cur node is a output op.
    return cur_node->isa<output_op>() ? directly_linked_output : no_linked;
}

// This function aims to automatically detect wrong inplacement of external
// input/output buffers and insert copying reorders between them.
void invalid_inplacement_detection(sc_graph_t &graph) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        // skip tensor view with single use.
        int dummy_idx = 0;
        auto cur_node = skip_tensor_view(node, dummy_idx);
        if (is_copy_reorder(cur_node) || cur_node->isa<output_op>()) { return; }
        size_t inp_count = 0, out_count = 0;
        if (has_linked_input(cur_node)) { inp_count++; }
        for (auto &customer_use : cur_node->get_outputs()[0]->uses_) {
            auto customer = customer_use.second.lock();
            int out_idx = customer_use.first;
            auto status = has_linked_output_and_modify_copy(
                    out_idx, customer, /*is_del*/ false);
            if (status == directly_linked_output) { out_count++; }
        }
        // If the buffer is linked to 2 or more inputs/outputs, need to
        // insert copy.
        if (inp_count + out_count >= 2) {
            // if linked to input, then all output needs copy, else one output
            // could directly use the buffer.
            int insert_num = inp_count > 0 ? out_count : out_count - 1;
            auto uses = cur_node->get_outputs()[0]->uses_;
            for (auto &customer_use : uses) {
                if (insert_num == 0) { break; }
                auto customer = customer_use.second.lock();
                int out_idx = customer_use.first;
                auto status
                        = has_linked_output_and_modify_copy(out_idx, customer,
                                /*is_del*/ false, /*force_insert*/ true);
                if (status == linked_output) { --insert_num; }
            }
        }
        vis->update_state_for_visited(cur_node);
    });
    graph.reset_op_ids();
}

// This function aims to do reduction copying op elimination, copying reorders
// could be inserted by users or past pass.
void redundant_copy_elimination(sc_graph_t &graph) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        // skip tensor view with single use.
        int dummy_idx = 0;
        auto cur_node = skip_tensor_view(node, dummy_idx);
        if (is_copy_reorder(cur_node) || cur_node->isa<output_op>()) { return; }
        size_t inp_count = 0, out_count = 0, direct_out_count = 0;
        if (has_linked_input(cur_node)) { inp_count++; }
        auto uses = cur_node->get_outputs()[0]->uses_;
        for (auto &customer_use : uses) {
            auto customer = customer_use.second.lock();
            int out_idx = customer_use.first;
            auto status = has_linked_output_and_modify_copy(out_idx, customer);
            if (utils::is_one_of(
                        status, linked_output, directly_linked_output)) {
                out_count++;
                if (status == directly_linked_output) { direct_out_count++; }
            }
        }
        assert(direct_out_count <= 1);
        // If the buffer is linked to single input/output, eliminate the copy,
        // else if linked to 2 or more outputs, eliminate the first copy.
        if ((inp_count + out_count < 2 && inp_count + out_count > 0)
                || (inp_count == 0 && out_count >= 2
                        && direct_out_count == 0)) {
            int delete_num = 1;
            auto uses = cur_node->get_outputs()[0]->uses_;
            for (auto &customer_use : uses) {
                if (delete_num == 0) { break; }
                auto customer = customer_use.second.lock();
                int out_idx = customer_use.first;
                auto status
                        = has_linked_output_and_modify_copy(out_idx, customer,
                                /*is_del*/ true,
                                /*force insert*/ false, /*force delete*/ true);
                if (status == linked_output) { delete_num--; }
            }
        }
        vis->update_state_for_visited(cur_node);
    });
    graph.reset_op_ids();
}

void inplace_transform(sc_graph_t &graph, const context_ptr &ctx) {
    invalid_inplacement_detection(graph);
    redundant_copy_elimination(graph);
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
