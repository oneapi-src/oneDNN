/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#include <algorithm>
#include <unordered_set>

#include "../fusible_op.hpp"
#include "../visitor.hpp"
#include <compiler/ir/graph/fusible_op.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/ternary_elemwise.hpp>
#include <util/math_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/*
Insert broadcast after the input with largest size
*/
static size_t get_input_idx_to_broadcast(const sc_op_ptr &node) {
    // TODO(yifei): upgrade the logic here
    auto inputs = node->get_inputs();
    std::vector<int64_t> input_sizes(inputs.size());
    std::transform(inputs.begin(), inputs.end(), input_sizes.begin(),
            [](const graph_tensor_ptr &gt) -> int64_t {
                auto lt = gt->details_;
                if (lt.is_dynamic()) {
                    return 0;
                } else {
                    return math_utils::get_dims_product(lt.get_blocking_dims())
                            * static_cast<int64_t>(utils::get_sizeof_etype(
                                    lt.dtype_.type_code_));
                }
            });
    return std::max_element(input_sizes.begin(), input_sizes.end())
            - input_sizes.begin();
}

/*
Applies to multi-directional broadcast scenario
One side broadcast to the output shape
TODO(yifei): decide broadcast insertion rule
    - Which side to choose
(x,1,z,1)[v0] [v1](y,z,w)
           \   /
            add
             |
   (x,y,z,w)[v3]
===============
               [v1](y,z,w)
                |
            broadcast
                |
(x,1,z,1)[v0] [v4](x,y,z,w)
           \   /
            add
             |
   (x,y,z,w)[v3]
*/
void broadcast_transform(sc_graph_t &graph, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::dfs_topology_sort();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (auto broadcastable_node
                = node->dyn_cast<op_traits::may_broadcast_t>()) {
            if (broadcastable_node->get_non_broadcast_input_index(false)
                            .empty()) {
                auto inputs = node->info_.inputs_;
                auto &out_shape
                        = node->info_.outputs_[0]->details_.get_plain_dims();
                auto broadcast_input_idx = get_input_idx_to_broadcast(node);
                const auto &bc_axis = broadcastable_node->get_plain_bc_axis();
                auto broadcast = graph.make("broadcast",
                        {inputs[broadcast_input_idx]}, {},
                        {{"output_shape", out_shape},
                                {"bc_axis", bc_axis[broadcast_input_idx]}});
                std::vector<graph_tensor_ptr> new_inputs = inputs;
                new_inputs[broadcast_input_idx] = broadcast->get_outputs()[0];
                auto new_broadcastable
                        = graph.make(node->op_name_, new_inputs, {}, {});
                node->replace_uses_with_and_remove(new_broadcastable);
                vis->update_state_for_visited(broadcast);
                vis->update_state_for_visited(new_broadcastable);
            }
        }
    });
    graph.reset_op_ids();
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
