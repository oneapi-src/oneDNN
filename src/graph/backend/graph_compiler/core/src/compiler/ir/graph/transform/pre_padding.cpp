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

#include <compiler/ir/graph/fused_op.hpp>
#include <ops/convolution.hpp>
#include <ops/templates/conv_rl.hpp>
#include <ops/templates/utils.hpp>
#include <unordered_map>
#include <unordered_set>

#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../visitor.hpp"
#include "transform.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(graph.pre_padding);

void pre_padding(sc_graph_t &graph, const context_ptr &ctx) {
    if (graph.is_dynamic()) { return; }
    COMPILE_ASSERT(!graph.is_dynamic(), "PrePadding only support static cases");
    auto visitor = op_visitor_t::dfs_topology_sort(graph.ops_.size());
    auto is_zero_paddings = [](sc_dims &paddings) {
        bool zero_paddings = true;
        for (auto &p : paddings) {
            if (p != 0) {
                zero_paddings = false;
                break;
            }
        }
        return zero_paddings;
    };
    visitor.visit_graph(
            graph, [&](op_visitor_t *visitor, const sc_op_ptr &node) {
                // if current node is a conv op with paddings,
                // insert a padding op before current node

                if (node->isa<ops::conv_fwd_core_op_t>()) {
                    auto input_dims = node->get_inputs()[0]
                                              ->details_.get_plain_dims()
                                              .size();
                    bool is_amx_dtype = ops::is_amx_dtype(
                            ctx, node->get_inputs()[0]->details_.dtype_);

                    // TODO(xurui)
                    // Only support extract padding op from 2d conv for now.
                    if (((input_dims != 4) || !is_amx_dtype)) { return; }
                    // Only apply to inference
                    bool is_weight_constant
                            = node->get_inputs()[1]
                                      ->producer_owner_->isa<constant_op_t>()
                            || node->get_inputs()[1]
                                       ->producer_owner_->attrs_.get_or_else(
                                               "constant",
                                               const_kind::not_const)
                            || node->get_inputs()[1]->attrs_.get_or_else(
                                    "constant", const_kind::not_const);
                    if (!is_weight_constant) { return; }
                    if (node->attrs_.get_or_else(
                                "use_rl", ops::rl_kind::NO_LOWERING)
                            == ops::rl_kind::FULL_LOWERING)
                        return;

                    auto pads_begin = node->attrs_.has_key("pads_begin")
                            ? node->attrs_.get<sc_dims>("pads_begin")
                            : node->attrs_.get<sc_dims>("paddings");

                    auto pads_end = node->attrs_.has_key("pads_end")
                            ? node->attrs_.get<sc_dims>("pads_end")
                            : node->attrs_.get<sc_dims>("paddings");

                    if (is_zero_paddings(pads_begin)) { return; }

                    auto parent_node = node->get_inputs()[0]->producer_owner_;

                    if (parent_node->isa<input_op>()) { return; }
                    auto padding_node
                            = graph.make("padding", {node->get_inputs()[0]}, {},
                                    {{"pads_begin", pads_begin},
                                            {"pads_end", pads_end}});
                    if (node->get_inputs()[0]->details_.get_plain_dims()[0]
                            == 1) {
                        padding_node->attrs_.set(
                                op_attr_key::break_post_fuse, true);
                    }

                    // clear paddings from original conv node
                    node->attrs_.set<sc_dims>("pads_begin", sc_dims {0});
                    node->attrs_.set<sc_dims>("pads_end", sc_dims {0});
                    node->attrs_.set<sc_dims>("paddings", sc_dims {0});

                    visitor->update_state_for_visited(node);
                    auto conv_new = graph.make(node->op_name_,
                            {padding_node->get_outputs()[0],
                                    node->get_inputs()[1]},
                            node->get_outputs(), node->attrs_);

                    // Copy configs from old node to new node
                    // TODO(xurui) A better way for doing this is put the config
                    // setting step after all grpah rewrite so that we do not
                    // need to copy config here
                    if (auto tunable_node = node->dyn_cast<tunable_op_t>()) {
                        conv_new->dyn_cast<tunable_op_t>()->set_config(
                                tunable_node->get_config());
                    }
                    node->remove();
                    visitor->update_state_for_visited(padding_node);
                }
            });
    graph.reset_op_ids();
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
