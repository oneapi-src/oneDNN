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

#include "pass.hpp"

#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <ops/fusible/broadcast.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/fusible/ternary_elemwise.hpp>
SC_MODULE(graph.pass.const_input_fold);

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
enum mask_etype : int {
    no_need_mask = 0,
    not_nan, // not produce nan/inf(div)
    has_nan
};
const std::unordered_set<std::string> ops_need_mask
        = {"exp", "select", "add", "sub", "div", "max"};
static int need_op_mask(const sc_op_ptr &op) {
    int pre_mask_type = op->attrs_.get_or_else(
            "temp.mask_type", (int)mask_etype::no_need_mask);
    if (ops_need_mask.find(op->op_name_) != ops_need_mask.end()) {
        // div may produce nan/inf, and nan/inf * 0 is nan
        if (op->op_name_ == "div") {
            return std::max(pre_mask_type, (int)mask_etype::has_nan);
        }
        // f(0, 0) = 0
        if (utils::is_one_of(op->op_name_, std::string("add"),
                    std::string("sub"), std::string("max"))) {
            if (op->get_inputs()[0]->details_.get_plain_dims()
                    != op->get_inputs()[1]->details_.get_plain_dims()) {
                return std::max(pre_mask_type, (int)mask_etype::not_nan);
            }
            return std::max(pre_mask_type, (int)mask_etype::no_need_mask);
        }
        if (op->op_name_ == "select") {
            if (op->get_inputs()[1]->details_.get_plain_dims()
                    != op->get_inputs()[2]->details_.get_plain_dims()) {
                return std::max(pre_mask_type, (int)mask_etype::not_nan);
            }
            return std::max(pre_mask_type, (int)mask_etype::no_need_mask);
        }

        return std::max(pre_mask_type, (int)mask_etype::not_nan);
    }
    return pre_mask_type;
}
static void mark_mask_attr(const sc_op_ptr &op) {
    if (!op->isa<unary_elementwise_op_t>()
            && !op->isa<binary_elementwise_op_t>() && !op->isa<select_op_t>()) {
        return;
    }
    int mask_type = need_op_mask(op);
    auto &uses = op->get_outputs()[0]->uses_;
    if (op->get_outputs()[0]->details_.get_format().is_blocking()
            && mask_type) {
        for (auto &use : uses) {
            if (use.second->isa<reduce_op_t>()
                    || use.second->isa<reduce_impl_op_t>()) {
                op->attrs_.set(op_attr_key::use_padded_mask, true);
                return;
            }
            if (use.second->isa<tunable_op_t>()
                    && mask_type == (int)mask_etype::has_nan) {
                op->attrs_.set(op_attr_key::use_padded_mask, true);
                return;
            }
            if (use.second->isa<movement_op_t>()) {
                if (use.second->isa<reorder_op_t>()
                        && use.second->get_inputs()[0]
                                   ->details_.get_format()
                                   .is_blocking()
                        && use.second->get_outputs()[0]
                                   ->details_.get_format()
                                   .is_plain()) {
                    continue;
                }
                if (use.second->isa<broadcast_op_t>()) { continue; }
                op->attrs_.set(op_attr_key::use_padded_mask, true);
                return;
            }
        }
        // pass down to its uses.
        for (auto &use : uses) {
            use.second->attrs_.set("temp.mask_type", mask_type);
        }
    }
    op->attrs_.set(op_attr_key::use_padded_mask, false);
}

SC_INTERNAL_API void padded_mask_mark(
        sc_graph_t &graph, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(graph.ops_.size());
    vis.visit_graph(graph, [](op_visitor_t *vis, const sc_op_ptr &node) {
        mark_mask_attr(node);
    });
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
