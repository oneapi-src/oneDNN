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
#include <utility>

#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../tunable_op.hpp"
#include "../visitor.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static bool check_have_any(const sc_graph_t &graph) {
    bool have_any = false;
    for (const auto &op : graph.ops_) {
        for (auto &in : op->get_inputs()) {
            if (in->details_.get_format().is_any()) {
                have_any = true;
                break;
            }
        }
    }
    return have_any;
}

static void insert_reorder_op(sc_graph_t &graph, graph_tensor_ptr in,
        size_t in_index, const sc_data_format_t &out_format,
        const sc_op_ptr &cur_op) {
    auto ret = graph.make("reorder", {std::move(in)}, {},
            {{"out_format", out_format}, {"internal", true},
                    {op_attr_key::no_fuse, // walk around for conv graph. will
                            // be dropped after yijie's refactor
                            graph.attrs_.get_or_else(
                                    "reorder_not_to_fuse", false)}});
    cur_op->replace_input(in_index, ret->get_outputs()[0]);
}

static bool need_to_reorder(const sc_data_format_t &format) {
    bool needed = false;
    for (size_t i = 0; i < sc_data_format_kind_t::MAX_DIMS; ++i) {
        if (i > 0) {
            if ((format.format_code_.get(i) - format.format_code_.get(i - 1))
                    < 0) {
                needed = true;
                break;
            }
        }
    }
    return needed;
}

void permute_propagation(sc_graph_t &graph, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(graph.ops_.size());
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        for (size_t i = 0; i < node->get_inputs().size(); ++i) {
            auto in = node->get_inputs()[i];
            if (in->details_.get_format().is_any()) {
                in->details_.set_format(sc_data_format_t::get_plain_by_dims(
                        (int)in->details_.get_plain_dims().size()));
            } else if (in->details_.get_format().is_plain()
                    && need_to_reorder(in->details_.get_format())) {
                // todo: should query the Op if it accepts a permuted layout.
                // TODO(yifei): consider remove this entire pass
                // since we will not run into this condition
                insert_reorder_op(graph, in, i,
                        sc_data_format_t::get_plain_by_dims(
                                (int)in->details_.get_plain_dims().size()),
                        node);
            }
        }
    });
    graph.reset_op_ids();
    COMPILE_ASSERT(!check_have_any(graph),
            "After permute_propagation, each op's graph_tensor should have no "
            "any format");
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
