/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#include "traits.hpp"
#include "tunable_op.hpp"
#include "util/utils.hpp"
#include "visitor.hpp"

namespace sc {

namespace graph {

void set_graph_config(sc_graph_t &g, const graph_config &tcfg) {
    size_t visited_num = 0;
    op_visitor_t vis(op_visitor_t::dequeue_selector,
            op_visitor_t::create_DAG_updater(g.ops_.size()));
    vis.visit_graph(g, [&](const sc_op_ptr &op) {
        // avoid out of range error due to some intunable graph op
        if (auto tune_op = op->dyn_cast<op_traits::configurable_t>()) {
            tune_op->set_config(tcfg.op_cfgs_.at(visited_num++).data_);
        }
    });
}
} // namespace graph

} // namespace sc
