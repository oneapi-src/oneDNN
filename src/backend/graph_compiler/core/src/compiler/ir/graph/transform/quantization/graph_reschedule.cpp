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
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/quantization/quantize_info.hpp>
#include <compiler/ir/graph/quantization/quantize_op.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <util/math_utils.hpp>
namespace sc {
namespace quantize {
void dequantize_elimination(sc_graph_t &mgr, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::bfs();
    vis.visit_graph(mgr, [&](const sc_op_ptr &node) {
        if (node->isa<dequantize_op_t>()
                || (node->isa<cast_op_t>()
                        && node->attrs_.get_or_else("mixed_dtype", false))) {
            assert(node->get_inputs().size() == 1);
            auto dequantize_input = node->info_.inputs_[0];
            auto dequantize_output = node->get_outputs()[0];
            int use_count = dequantize_output->uses_.size();
            vis.update_state_for_visited(node);

            auto cld_nodes = dequantize_output->uses_;
            for (auto &cld_node : cld_nodes) {
                if ((cld_node.second->isa<op_traits::may_quantize_t>()
                            && cld_node.second
                                       ->dyn_cast<op_traits::may_quantize_t>()
                                       ->should_quantized_)
                        || cld_node.second->isa<cast_op_t>()) {
                    cld_node.second->replace_input(
                            cld_node.first, dequantize_input);
                    use_count--;
                }
            }
            if (!use_count) { node->remove(); }
        }
    });
    mgr.reset_op_ids();
}

void insert_back_dequantize(sc_graph_t &mgr, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(mgr.ops_.size());
    vis.visit_graph(mgr, [&](const sc_op_ptr &node) {
        if (node->dyn_cast<op_traits::may_quantize_t>()) {
            if (node->isa<tunable_op_t>()) {
                // create a new quantized node every time may confuse
                // `to_visit` list.
                auto may_quantize_node
                        = node->dyn_cast<op_traits::may_quantize_t>();
                if (!may_quantize_node->should_quantized_
                        || may_quantize_node->is_quantized_) {
                    return;
                }
                if (may_quantize_node->should_quantized_
                        && node->get_inputs()[0]->details_.dtype_.is_etype(
                                sc_data_etype::BF16)) {
                    may_quantize_node->is_quantized_ = true;
                    vis.update_state_for_visited(node);
                    return;
                }
                node->get_outputs()[0]->details_.dtype_.type_code_
                        = sc_data_etype::S32;
                may_quantize_node->is_quantized_ = true;
                node->op_name_ = "quantized_" + node->op_name_;
                assert(node->attrs_.has_key("data_scales")
                        && node->attrs_.has_key("weight_scales"));
                auto data_scales
                        = node->attrs_.get<std::vector<float>>("data_scales");
                auto weight_scales
                        = node->attrs_.get<std::vector<float>>("weight_scales");
                auto output_scales
                        = math_utils::vector_mul(data_scales, weight_scales);
                int output_channel_axis;
                if (node->attrs_.has_key("output_channel_axis")) {
                    output_channel_axis
                            = node->attrs_.get<int>("output_channel_axis");
                } else {
                    output_channel_axis
                            = node->attrs_.get<int>("weight_channel_axis");
                }
                for (auto &child : node->get_outputs()[0]->uses_) {
                    auto cur_child = child;
                    auto cur_parent = node;
                    while (cur_child.second
                                    ->dyn_cast<op_traits::may_quantize_t>()) {
                        cur_child.second->replace_input(
                                cur_child.first, cur_parent->get_outputs()[0]);
                        cur_parent = cur_child.second;
                        cur_child
                                = cur_child.second->get_outputs()[0]->uses_[0];
                        cur_parent->get_outputs()[0]->details_.dtype_.type_code_
                                = sc_data_etype::S32;
                    }
                    auto dequantize_node = mgr.make("dequantize",
                            cur_parent->get_outputs(),
                            std::vector<graph_tensor_ptr> {},
                            {{"dtype", datatypes::f32},
                                    {"scales", output_scales},
                                    {"channel_axis", output_channel_axis}});
                    cur_child.second->replace_input(
                            cur_child.first, dequantize_node->get_outputs()[0]);
                    vis.update_state_for_visited(dequantize_node);
                }
            } else {
                // align output datatype with input
                for (auto &out : node->get_outputs()) {
                    out->details_.dtype_
                            = node->get_inputs()[0]->details_.dtype_;
                    // insert dequantize if last op of pattern is output op, for
                    // pattern like `qua->deq->reshape->output`
                    std::vector<std::pair<int, sc_op_weak_ptr_t>> uses
                            = out->uses_;
                    for (auto &use : uses) {
                        auto cld_op = use.second.lock();
                        if (cld_op->isa<output_op>()
                                && node->attrs_.has_key("scales")
                                && node->attrs_.has_key("zero_points")) {
                            auto dequantize_node = mgr.make("dequantize", {out},
                                    std::vector<graph_tensor_ptr> {},
                                    node->attrs_);
                            cld_op->replace_input(use.first,
                                    dequantize_node->get_outputs()[0]);
                            vis.update_state_for_visited(dequantize_node);
                        }
                    }
                }
            }
        }
    });
    mgr.reset_op_ids();
}

/**
 * reschedule graph by add/remove dequantize op and replace calculate op
 * with quantize calculate op.
 *
 * for dequantize op, as quantize information is propagated by info
 * propagation, if it has no non-quantize uses, we remove the node.
 * before:
 *                |
 *          dequantize_op
 *                |
 *             conv_op
 * after:
 *                |
 *                |
 *              conv_op
 *                |
 *
 * for calculate op with quantize info, we replace it with a quantized
 * one and add a dequantize node after. before:
 *                  |
 *               conv_op(with quantize info)
 *                  |
 * after:
 *                  |
 *               qconv_op(u8/s8 in, s32 out)
 *                  |
 *             dequantize_op
 *                  |
 *
 * */
void graph_reschedule(sc_graph_t &mgr, const context_ptr &ctx) {
    if (!mgr.attrs_.get_or_else(sc_graph_t::attr_key_t::quantize, false))
        return;
    dequantize_elimination(mgr, ctx);
    insert_back_dequantize(mgr, ctx);
}
} // namespace quantize
} // namespace sc
