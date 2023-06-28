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
#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../quantization/quantize_op.hpp"
#include "../tunable_op.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <ops/reshape.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/// Push tensor_view/transpose op back to quantize/dequantize ops for better
/// fusion in dynamic cases.
void tensor_view_push_back(sc_graph_t &graph, const context_ptr &ctx) {
    constexpr const int max_try_times = 10;
    for (int i = 0; i < max_try_times; i++) {
        bool changed = false;
        auto vis = op_visitor_t::bfs_unchecked();
        vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
            if (node->isa<tensor_view_op_t>() || node->isa<transpose_op_t>()
                    || node->isa<ops::dynamic_reshape_op>()) {
                auto cur_node = node;
                auto details = node->get_inputs()[0]->details_;
                while (cur_node->is_single_output_single_use()) {
                    auto next_node
                            = cur_node->get_outputs()[0]->uses_[0].second;
                    int use_idx = cur_node->get_outputs()[0]->uses_[0].first;
                    if (!(next_node->isa<quantize::quantize_op_t>()
                                || next_node->isa<cast_op_t>())) {
                        break;
                    }
                    if (cur_node == node) {
                        next_node->replace_input(
                                use_idx, cur_node->get_inputs()[0]);
                    }
                    cur_node = next_node;
                    details.dtype_ = cur_node->get_inputs()[0]->details_.dtype_;
                    cur_node->get_inputs()[0]->details_ = details;
                    details.dtype_
                            = cur_node->get_outputs()[0]->details_.dtype_;
                    cur_node->get_outputs()[0]->details_ = details;
                }
                // if changed
                if (cur_node != node) {
                    changed = true;
                    // cache uses.
                    auto uses = cur_node->get_outputs()[0]->uses_;
                    node->replace_input(0, cur_node->get_outputs()[0]);
                    // correct datatype as we could meet a cast op.
                    auto out_tsr = node->get_outputs()[0];
                    out_tsr->details_.dtype_
                            = cur_node->get_outputs()[0]->details_.dtype_;
                    for (auto &use : uses) {
                        use.second->replace_input(use.first, out_tsr);
                    }
                }
            }
            vis->update_state_for_visited(node);
        });
        if (!changed) { break; }
    }
    graph.reset_op_ids();
}

void dynamic_graph_transform(sc_graph_t &graph, const context_ptr &ctx) {
    if (!graph.is_dynamic()) { return; }
    tensor_view_push_back(graph, ctx);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
