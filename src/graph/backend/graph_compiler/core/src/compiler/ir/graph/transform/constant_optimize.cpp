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

#include <set>

#include "../visitor.hpp"
#include "transform.hpp"
#include <compiler/ir/graph/traits.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

void constant_optimization(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::dfs_topology_sort(graph.ops_.size());
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (auto graph_node
                = node->dyn_cast<op_traits::constant_optimizable_t>()) {
            auto ret = graph_node->constant_optimize(graph);
            if (ret) { vis->update_state_for_visited(ret); }
        }
    });
    graph.reset_op_ids();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
