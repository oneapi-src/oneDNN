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

#include <unordered_map>

#include "../graph.hpp"
#include "../visitor.hpp"
#include "pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// using pre-order and post-order visit graph, if any throw error and
// connection error, this function return false
bool check_graph_connection(sc_graph_t &graph) {
    try {
        op_visitor_t post_vis {op_visitor_t::pop_back_selector,
                op_visitor_t::create_DAG_updater_post(graph.ops_.size()), true};
        // {logical op id, visited num} map
        std::unordered_map<int, int> visited_ops;
        post_vis.post_visit_graph(
                graph, [&](op_visitor_t *post_vis, const sc_op_ptr &aop) {
                    visited_ops[aop->logical_op_id_]++;
                });

        op_visitor_t::dfs_topology_sort(graph.ops_.size())
                .visit_graph(
                        graph, [&](op_visitor_t *vis, const sc_op_ptr &aop) {
                            visited_ops[aop->logical_op_id_]--;
                        });
        for (auto op : visited_ops) {
            if (op.second) { return false; }
        }
    } catch (...) { return false; }
    return true;
}

// check config in tunable op is valid or not.
bool check_graph_config(sc_graph_t &graph, const context_ptr &ctx) {
    for (auto &node : graph.ops_) {
        if (!node->is_removed_ && !node->is_valid(ctx)) { return false; }
    }
    return true;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
