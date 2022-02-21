/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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
#include <utility>

#include "../fusible_op.hpp"
#include "../visitor.hpp"
#include <ops/fusible/memory_movement.hpp>

namespace sc {
// Helper function to judge if a reorder should be transformed to tensor view.
// If the reorder actually does not result in memory movement, we mark it as
// should_transform.
bool should_transform_reorder(const sc_op_ptr &node) {
    assert(node->isa<reorder_op_t>());
    auto &input_blocking_shapes
            = node->get_inputs()[0]->details_.get_blocking_dims();
    auto &output_blocking_shapes
            = node->get_outputs()[0]->details_.get_blocking_dims();
    auto &input_format = node->get_inputs()[0]->details_.get_format();
    auto &output_format = node->get_outputs()[0]->details_.get_format();
    // inp format is equal to out format
    if (input_format == output_format) { return true; }
    // reorder for padding
    if (sc_data_format_t::get_padded_plain_shapes(
                input_blocking_shapes, input_format)
            != sc_data_format_t::get_padded_plain_shapes(
                    output_blocking_shapes, output_format)) {
        return false;
    }
    int inp_idx = 0, out_idx = 0;
    auto &inp_code = input_format.format_code_;
    auto &out_code = output_format.format_code_;
    // orig axis -> vector of block idx
    auto inp_blocked_axis = input_format.get_blocked_axis();
    auto out_blocked_axis = output_format.get_blocked_axis();
    // orig axis -> current idx of block
    std::unordered_map<int, size_t> inp_block_idx, out_block_idx;
    while (inp_idx < sc_data_format_kind_t::MAX_DIMS
            && out_idx < sc_data_format_kind_t::MAX_DIMS
            && (inp_code.get(inp_idx) != sc_data_format_kind_t::UNDEF_DIM
                    || out_code.get(out_idx)
                            != sc_data_format_kind_t::UNDEF_DIM)) {
        // get next inp_block_idx
        auto inp_block_idx_itr = inp_block_idx.find(inp_code.get(inp_idx));
        auto out_block_idx_itr = out_block_idx.find(out_code.get(out_idx));
        if (inp_block_idx_itr == inp_block_idx.end()) {
            inp_block_idx[inp_code.get(inp_idx)] = 0;
            out_block_idx[inp_code.get(out_idx)] = 0;
        } else {
            inp_block_idx_itr->second++;
            out_block_idx_itr->second++;
        }
        // skip same axis
        while (inp_code.get(inp_idx + 1) != sc_data_format_kind_t::UNDEF_DIM
                && inp_code.get(inp_idx + 1) == inp_code.get(inp_idx)) {
            inp_block_idx[inp_code.get(inp_idx + 1)]++;
            inp_idx++;
        }
        while (out_code.get(out_idx + 1) != sc_data_format_kind_t::UNDEF_DIM
                && out_code.get(out_idx + 1) == out_code.get(out_idx)) {
            out_block_idx[out_code.get(out_idx + 1)]++;
            out_idx++;
        }
        int orig_inp_axis = inp_code.get(inp_idx);
        int orig_out_axis = out_code.get(out_idx);
        // different axis
        if (orig_inp_axis != orig_out_axis) { return false; }
        // different block
        auto inp_block_itr = inp_blocked_axis.find(orig_inp_axis);
        auto out_block_itr = out_blocked_axis.find(orig_out_axis);
        if (inp_block_itr != inp_blocked_axis.end()
                || out_block_itr != out_blocked_axis.end()) {
            // inp has no block, but out has blocks remained.
            if (inp_block_itr == inp_blocked_axis.end()
                    && out_block_idx[orig_out_axis]
                            < out_block_itr->second.size()) {
                return false;
            }
            // out has no block, but inp has blocks remained.
            if (out_block_itr == out_blocked_axis.end()
                    && inp_block_idx[orig_inp_axis]
                            < inp_block_itr->second.size()) {
                return false;
            }
            // inp and out both have blocks.
            if (inp_block_itr != inp_blocked_axis.end()
                    && out_block_itr != out_blocked_axis.end()) {
                // inp and out have blocks remained.
                if (inp_block_idx[orig_inp_axis] < inp_block_itr->second.size()
                        && out_block_idx[orig_out_axis]
                                < out_block_itr->second.size()) {
                    // inp and out block remained should be equal.
                    if (inp_block_itr->second[inp_block_idx[orig_inp_axis]]
                            != out_block_itr
                                       ->second[out_block_idx[orig_out_axis]]) {
                        return false;
                    }
                } else if (inp_block_idx[orig_inp_axis]
                        < inp_block_itr->second.size()) {
                    return false;
                } else if (out_block_idx[orig_out_axis]
                        < out_block_itr->second.size()) {
                    return false;
                }
                // else inp and out have no remained blocks.
            }
        }
        // increase inp_idx
        inp_idx++;
        out_idx++;
    }
    return true;
}

static bool should_transform_transpose(const sc_op_ptr &node) {
    assert(node->isa<transpose_op_t>());
    if (node->info_.inputs_[0]->details_.get_format()
            != node->info_.outputs_[0]->details_.get_format()) {
        // means that we have permuted the data format
        return true;
    }
    return false;
}

// Replace reorder op and transpose op that does not cause data movements with
// tensor_view op
void convert_to_tensor_view(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        if (node->isa<reorder_op_t>() && should_transform_reorder(node)) {
            auto tensor_view_out = node->get_outputs()[0]->copy();
            tensor_view_out->producer_owner_ = nullptr;
            auto view = graph.make("tensor_view", node->get_inputs(),
                    {tensor_view_out},
                    {{"shape", tensor_view_out->details_.get_blocking_dims()}});
            node->replace_uses_with_and_remove(view);
            vis.update_state_for_visited(view);
        } else if (node->isa<transpose_op_t>()
                && should_transform_transpose(node)) {
            auto tensor_view_out = node->get_outputs()[0]->copy();
            tensor_view_out->producer_owner_ = nullptr;
            auto view = graph.make("tensor_view", node->get_inputs(),
                    {tensor_view_out},
                    {{"shape", tensor_view_out->details_.get_blocking_dims()}});
            node->replace_uses_with_and_remove(view);
            vis.update_state_for_visited(view);
        }
    });
    graph.reset_op_ids();
}

void tensor_view_transform(sc_graph_t &graph, const context_ptr &ctx) {
    convert_to_tensor_view(graph, ctx);
}
} // namespace sc
