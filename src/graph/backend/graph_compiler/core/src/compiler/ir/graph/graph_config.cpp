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

#include "graph_config.hpp"
#include <utility>
#include <vector>
#include "graph_op.hpp"
#include "traits.hpp"
#include "tunable_op.hpp"
#include "util/utils.hpp"
#include "visitor.hpp"
#include <unordered_map>
#include <util/reflection.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// clang-format off
SC_CLASS(graph_config)
  SC_FIELD(op_cfgs_)
SC_CLASS_END();
// clang-format on

namespace graph {

void set_graph_config(sc_graph_t &g, const graph_config &tcfg) {
    size_t visited_num = 0;
    for (auto &op : g.ops_) {
        if (auto tune_op = op->dyn_cast<op_traits::configurable_t>()) {
            tune_op->set_config(tcfg.op_cfgs_.at(visited_num++));
        }
    }
}

graph_config get_graph_default_config(context_ptr ctx, const sc_graph_t &g) {
    graph_config cfg;
    op_visitor_t vis = op_visitor_t::bfs_topology_sort(g.ops_.size());
    vis.visit_graph(g, [&](op_visitor_t *vis, const sc_op_ptr &op) {
        if (auto tune_op = op->dyn_cast<op_traits::configurable_t>()) {
            auto obj = tune_op->get_default_config(ctx);
            cfg.op_cfgs_.emplace_back(std::move(obj));
        }
    });
    return cfg;
}
} // namespace graph

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
