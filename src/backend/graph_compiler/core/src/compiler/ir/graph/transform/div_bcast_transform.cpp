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

#include "../fusible_op.hpp"
#include "../graph.hpp"
#include "../visitor.hpp"
#include <ops/fusible/binary_elemwise.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/*
[v0] [v1] <--broadcast tensor
  \   /
   div
    |
   [b1]
===============
     [v1]
       |
   reciprocal
       |
[v0] [v2] <--broadcast tensor
  \   /
   mul
    |
   [b1]
*/

sc_op_ptr insert_rcp(sc_graph_t &graph, div_op_t *node) {
    auto dtype = node->get_inputs()[1]->details_.dtype_;
    if (dtype.type_code_ == sc_data_etype::F32
            || dtype.type_code_ == sc_data_etype::BF16) {
        // next node uses div result
        graph_tensor_ptr v0 = node->get_inputs()[0];
        graph_tensor_ptr v1 = node->get_inputs()[1];
        auto v2 = graph.make("reciprocal", {v1}, {}, {})->get_outputs()[0];
        auto new_node = graph.make("mul", {v0, v2}, {}, node->attrs_);
        node->replace_uses_with_and_remove(new_node);
        return new_node;
    }
    return nullptr;
}

void div_bcast_transform(sc_graph_t &graph, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&graph](op_visitor_t *vis, const sc_op_ptr &node) {
        if (auto div_node = node->dyn_cast<div_op_t>()) {
            auto bcast_idx = div_node->get_broadcast_input();
            if (bcast_idx == 1) {
                assert(div_node->get_inputs().size() == 2);
                auto inserted_op = insert_rcp(graph, div_node);
                if (inserted_op) { vis->update_state_for_visited(inserted_op); }
            }
        }
    });
    graph.reset_op_ids();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
