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

#include "../fusible_op.hpp"
#include "../visitor.hpp"
#include <ops/fusible/binary_elemwise.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/*
[v0] [v1]
  \   /
   add
    |
   [v3]  [v2] <--broadcast tensor
     \    /
      bcast
        |
      [b1]
===============
[v0] [v2] <--broadcast tensor
  \   /
   bcast
    |
   [v3]  [v1]
     \    /
       add
        |
      [b1]
*/
static bool do_swap(sc_graph_t &mgr, sc_op *bcast, sc_op *other,
        int bcast_tsr_idx, op_visitor_t *visitor) {
    assert(bcast->get_inputs().size() == 2 && other->get_inputs().size() == 2);
    assert(bcast->get_outputs().size() == 1
            && other->get_outputs().size() == 1);
    graph_tensor_ptr v2 = bcast->get_inputs()[bcast_tsr_idx];
    graph_tensor_ptr v0 = other->get_inputs()[0];
    graph_tensor_ptr v1 = other->get_inputs()[1];
    graph_tensor_ptr b1 = bcast->get_outputs()[0];
    graph_tensor_ptr v3 = other->get_outputs()[0];
    if (v3->uses_.size() > 1) {
        // v3 has multiple uses, cannot swap
        return false;
    }

    int idx_of_other_in_bcast = false ? 1 : (bcast_tsr_idx == 1);
    auto new_bcast = mgr.make(bcast->op_name_, {v0, v2}, {}, bcast->attrs_);
    auto new_other = mgr.make(other->op_name_,
            {new_bcast->get_outputs()[0], v1}, {}, other->attrs_);
    bcast->replace_uses_with_and_remove(new_other);
    other->remove();
    visitor->update_state_for_visited(new_other);
    return true;
}

static bool check_and_swap(sc_graph_t &mgr, add_op_t *bcast_node,
        int index_in_bcast, int bcast_input_idx, op_visitor_t *visitor) {
    auto parent = bcast_node->get_inputs()[index_in_bcast]->producer_owner_;
    if (auto parent_add = parent->dyn_cast<add_op_t>()) {
        if (parent_add->get_broadcast_input() == -1) {
            return do_swap(
                    mgr, bcast_node, parent_add, bcast_input_idx, visitor);
        }
    }
    return false;
}

void elemwise_bcast_swap(sc_graph_t &mgr, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::dfs_topology_sort();
    vis.visit_graph(mgr, [&mgr](op_visitor_t *vis, const sc_op_ptr &node) {
        if (auto add_node = node->dyn_cast<add_op_t>()) {
            auto non_bcast_idx = add_node->get_non_broadcast_input_index(false);
            if (non_bcast_idx.size() == 1) {
                assert(add_node->get_inputs().size() == 2);
                int bcast_idx = 1 - non_bcast_idx[0];
                if (!check_and_swap(mgr, add_node, 0, bcast_idx, vis)) {
                    check_and_swap(mgr, add_node, 1, bcast_idx, vis);
                }
            }
        }
    });
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
