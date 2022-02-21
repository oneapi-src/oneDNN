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

#include "../fused_op.hpp"
#include "../fusible_op.hpp"
#include "../visitor.hpp"
#include "pass.hpp"
#include <ops/fusible/memory_movement.hpp>

namespace sc {

SC_INTERNAL_API void graph_constant_input_folding(
        sc_graph_t &mgr, const context_ptr &ctx) {
    op_visitor_t vis(op_visitor_t::pop_back_selector,
            op_visitor_t::create_DAG_updater(mgr.ops_.size()));
    vis.visit_graph(mgr, [](const sc_op_ptr &node) {
        if (node->isa<constant_op_t>()
                || node->attrs_.get_or_else(
                        "constant", const_kind::not_const)) {
            if (node->isa<constant_op_t>()) {
                node->attrs_.set("constant", const_kind::local_const);
            }
            bool all_constant_inputs = true;
            for (const auto &input : node->get_inputs()) {
                auto parent_node = input->producer_owner_;
                if (parent_node->attrs_.get_or_else(
                            "constant", const_kind::not_const)
                                == const_kind::not_const
                        && !parent_node->isa<constant_op_t>()) {
                    all_constant_inputs = false;
                    break;
                }
            }
            if (!all_constant_inputs) {
                for (const auto &input : node->get_inputs()) {
                    auto parent_node = input->producer_owner_;
                    if (parent_node->attrs_.get_or_else(
                                "constant", const_kind::not_const)
                            != const_kind::not_const) {
                        parent_node->attrs_.set(
                                "constant", const_kind::global_const);
                        if (parent_node->isa<tensor_view_op_t>()) {
                            parent_node->get_inputs()[0]
                                    ->producer_owner_->attrs_.set("constant",
                                            const_kind::global_const);
                        }
                    }
                }
                node->attrs_.set("constant", const_kind::not_const);
            } else {
                if (!node->isa<output_op>()) {
                    // Setting attrs here is intermediary status. Current node
                    // is constant node, its uses may also be constant, so we
                    // set `local_const` here temporarily meaning
                    // `may_constant`.
                    // Later when visiting its uses, we check their all inputs
                    // and decide whether we reserve this attr.
                    for (auto &out : node->get_outputs()) {
                        for (auto &cld_node : out->uses_) {
                            cld_node.second->attrs_.set(
                                    "constant", const_kind::local_const);
                        }
                    }
                }
            }
        }
    });
}
} // namespace sc
