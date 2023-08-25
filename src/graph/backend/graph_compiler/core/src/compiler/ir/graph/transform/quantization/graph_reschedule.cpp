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
#include <string>

#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/quantization/quantize_info.hpp>
#include <compiler/ir/graph/quantization/quantize_op.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <ops/convolution.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/pooling.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <ops/reshape.hpp>
#include <util/math_utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace quantize {
bool reschedule_forbid_op(const sc_op_ptr &node) {
    // transpose could be processed as it only changes the order of dim.
    return (node->isa<ops::dynamic_reshape_op>()
                   || node->isa<tensor_view_op_t>() || node->isa<reshape_op_t>()
                   || node->isa<concat_op_t>() || node->isa<split_op_t>())
            && node->attrs_.get_or_else(attr_keys::per_channel, false);
}
void dequantize_elimination(sc_graph_t &mgr, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::bfs();
    vis.visit_graph(mgr, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (node->isa<dequantize_op_t>()
                || node->isa<dynamic_dequantize_op_t>()) {
            auto dequantize_input = node->info_.inputs_[0];
            auto dequantize_output = node->get_outputs()[0];
            vis->update_state_for_visited(node);

            auto cld_nodes = dequantize_output->uses_;
            std::vector<sc_op_ptr> node_to_remove = {node};
            // whether replace inputs with next node according to two op nodes
            // back check when int8-bf16 mixed type.
            // For the case like:
            //                    dequantize
            //                     /
            //           matmul  cast
            //               \   /
            //                add
            if (cld_nodes.size() == 1) {
                auto cast_op = cld_nodes[0].second;
                if (cast_op->isa<cast_op_t>()
                        && cast_op->attrs_.get_or_else(
                                attr_keys::mixed_dtype, false)) {
                    vis->update_state_for_visited(cast_op);
                    node_to_remove.emplace_back(cast_op);
                    cld_nodes = cast_op->get_outputs()[0]->uses_;
                }
            }
            int use_count = cld_nodes.size();
            for (auto &cld_node : cld_nodes) {
                if ((cld_node.second->isa<op_traits::may_quantize_t>()
                            && cld_node.second
                                       ->dyn_cast<op_traits::may_quantize_t>()
                                       ->should_quantized_
                            && !reschedule_forbid_op(cld_node.second))) {
                    cld_node.second->replace_input(
                            cld_node.first, dequantize_input);
                    use_count--;
                }
            }
            if (!use_count) {
                std::for_each(node_to_remove.begin(), node_to_remove.end(),
                        [](const sc_op_ptr &op) { op->remove(); });
            }
        }
    });
    mgr.reset_op_ids();
}

void insert_back_dequantize(sc_graph_t &mgr, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(mgr.ops_.size());
    vis.visit_graph(mgr, [&](op_visitor_t *vis, const sc_op_ptr &node) {
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
                    vis->update_state_for_visited(node);
                    return;
                }
                node->get_outputs()[0]->details_.dtype_.type_code_
                        = sc_data_etype::S32;
                may_quantize_node->is_quantized_ = true;
                node->op_name_ = "quantized_" + node->op_name_;
                assert((node->attrs_.has_key(attr_keys::data_scales)
                               && node->attrs_.has_key(
                                       attr_keys::weight_scales))
                        || (node->attrs_.has_key(attr_keys::dyn_data_scales)
                                && node->attrs_.has_key(
                                        attr_keys::dyn_weight_scales)));
                bool is_dyn_quan
                        = node->attrs_.has_key(attr_keys::dyn_data_scales);
                auto weight_scales = node->attrs_.get_or_else(
                        attr_keys::weight_scales, std::vector<float>());
                int output_channel_axis;
                if (node->attrs_.has_key(attr_keys::output_channel_axis)) {
                    output_channel_axis = node->attrs_.get<int>(
                            attr_keys::output_channel_axis);
                } else if (node->isa<ops::conv_fwd_core_op_t>()
                        && node->attrs_.has_key("data_format")) {
                    // infer output_channel_axis based on op attributes
                    auto data_format
                            = node->attrs_.get<std::string>("data_format");
                    auto ndims = node->get_outputs()[0]
                                         ->details_.get_plain_dims()
                                         .size();
                    output_channel_axis = data_format == "NCX" ? 1 : ndims - 1;
                } else {
                    output_channel_axis = node->attrs_.get<int>(
                            attr_keys::weight_channel_axis);
                    if (weight_scales.size() > 1
                            && node->attrs_.has_key(
                                    attr_keys::weight_channel_axis)) {
                        SC_WARN << "Weight_channel_axis is specified but "
                                   "output_channel_axis not specified. "
                                   "Assuming output_channel_axis == "
                                   "weight_channel_axis";
                    }
                }
                auto tunable_op_consumers = node->get_outputs()[0]->uses_;
                for (auto &child : tunable_op_consumers) {
                    auto cur_child = child;
                    auto cur_parent = node;
                    while (cur_child.second
                                    ->dyn_cast<op_traits::may_quantize_t>()
                            && !cur_child.second->isa<concat_op_t>()
                            // reserve dynamic version here for debug.
                            //     &&
                            //     !cur_child.second->isa<movement_op_t>())
                            //     {
                            && cur_child.second->get_outputs().size() == 1) {
                        cur_child.second->replace_input(
                                cur_child.first, cur_parent->get_outputs()[0]);
                        cur_parent = cur_child.second;
                        cur_child
                                = cur_child.second->get_outputs()[0]->uses_[0];
                        cur_parent->get_outputs()[0]->details_.dtype_.type_code_
                                = sc_data_etype::S32;
                        if (cur_parent->get_outputs()[0]->uses_.size() > 1) {
                            break;
                        }
                    }
                    auto cur_parent_uses = cur_parent == node
                            ? std::vector<
                                    std::pair<int, sc_op_weak_ptr_t>> {child}
                            : cur_parent->get_outputs()[0]->uses_;
                    // TODO(yifei): overcome the constraints here
                    for (const auto &use : cur_parent_uses) {
                        COMPILE_ASSERT(use.second->isa<concat_op_t>()
                                        || !use.second->dyn_cast<
                                                op_traits::may_quantize_t>(),
                                "may_quantize op with multiple consumers "
                                "shouldn't have any of may_quantize "
                                "consumers (except concat).");
                    }
                    sc_op_ptr dequantize_node;
                    if (is_dyn_quan) {
                        assert(node->attrs_.has_key(attr_keys::dyn_data_scales)
                                && (node->attrs_.has_key(
                                            attr_keys::dyn_weight_scales)
                                        || node->attrs_.has_key(
                                                attr_keys::weight_scales)));
                        auto data_scales_gt
                                = node->attrs_.get<graph_tensor_ptr>(
                                        attr_keys::dyn_data_scales);
                        auto weight_scales_gt = node->attrs_.get_or_else(
                                attr_keys::dyn_weight_scales,
                                graph_tensor_ptr());
                        sc_op_ptr out_scales;
                        if (weight_scales_gt) {
                            out_scales = mgr.make("mul",
                                    {data_scales_gt, weight_scales_gt}, {}, {});
                        } else {
                            auto wei_scale_const = mgr.make("constant", {}, {},
                                    {{"values",
                                             std::make_shared<static_data_t>(
                                                     weight_scales)},
                                            {"dtype", datatypes::f32},
                                            {"plain_dims",
                                                    sc_dims {static_cast<
                                                            sc_dim>(
                                                            weight_scales
                                                                    .size())}},
                                            {"format", sc_data_format_t()}});
                            auto &wei_details
                                    = wei_scale_const->get_outputs()[0]
                                              ->details_;
                            out_scales = mgr.make("mul",
                                    {data_scales_gt,
                                            wei_scale_const->get_outputs()[0]},
                                    {std::make_shared<graph_tensor>(nullptr,
                                            wei_details.get_format(),
                                            wei_details.get_plain_dims(),
                                            wei_details.dtype_)},
                                    {});
                        }
                        dequantize_node = mgr.make("dynamic_dequantize",
                                {cur_parent->get_outputs()[0],
                                        out_scales->get_outputs()[0]},
                                {},
                                {{attr_keys::quan_dtype, datatypes::f32},
                                        {attr_keys::channel_axis,
                                                output_channel_axis}});
                    } else {
                        auto data_scales = node->attrs_.get<std::vector<float>>(
                                attr_keys::data_scales);
                        auto output_scales = math_utils::vector_mul(
                                data_scales, weight_scales);
                        dequantize_node = mgr.make("dequantize",
                                cur_parent->get_outputs(),
                                std::vector<graph_tensor_ptr> {},
                                {{attr_keys::quan_dtype, datatypes::f32},
                                        {attr_keys::scales, output_scales},
                                        {attr_keys::channel_axis,
                                                output_channel_axis}});
                    }
                    if (node->attrs_.get_or_else(
                                attr_keys::mixed_dtype, false)) {
                        dequantize_node = mgr.make("cast",
                                dequantize_node->get_outputs(),
                                std::vector<graph_tensor_ptr> {},
                                {{"dtype", datatypes::bf16}});
                    }
                    for (const auto &use : cur_parent_uses) {
                        use.second->replace_input(
                                use.first, dequantize_node->get_outputs()[0]);
                    }
                    vis->update_state_for_visited(dequantize_node);
                }
            } else if (!reschedule_forbid_op(node)) {
                // align output datatype with input
                for (auto &out : node->get_outputs()) {
                    if (node->isa<pooling_avg_op_t>()) {
                        node->get_outputs()[0]->details_.dtype_.type_code_
                                = sc_data_etype::F32;
                    } else {
                        out->details_.dtype_
                                = node->get_inputs()[0]->details_.dtype_;
                    }
                    // insert dequantize if last op of pattern is output op,
                    // for pattern like `qua->deq->reshape->output`
                    std::vector<std::pair<int, sc_op_weak_ptr_t>> uses
                            = out->uses_;
                    for (auto &use : uses) {
                        auto cld_op = use.second.lock();
                        if ((!(cld_op->dyn_cast<op_traits::may_quantize_t>()
                                     && cld_op->get_outputs().size() == 1)
                                    || reschedule_forbid_op(cld_op))
                                && ((node->attrs_.has_key(attr_keys::scales)
                                            && node->attrs_.has_key(
                                                    attr_keys::zero_points))
                                        || (node->attrs_.has_key(
                                                attr_keys::dyn_scales)))) {
                            bool is_dyn_quan = node->attrs_.has_key(
                                    attr_keys::dyn_scales);
                            sc_op_ptr dequantize_node;
                            if (is_dyn_quan) {
                                auto &scales
                                        = node->attrs_.get<graph_tensor_ptr>(
                                                attr_keys::dyn_scales);
                                auto &zero_points
                                        = node->attrs_.get<graph_tensor_ptr>(
                                                attr_keys::dyn_zero_points);
                                std::vector<graph_tensor_ptr> ins
                                        = {out, scales};
                                if (zero_points) {
                                    ins.emplace_back(zero_points);
                                }
                                dequantize_node = mgr.make("dynamic_dequantize",
                                        ins, {}, node->attrs_);
                            } else {
                                dequantize_node = mgr.make("dequantize", {out},
                                        std::vector<graph_tensor_ptr> {},
                                        node->attrs_);
                            }
                            if (node->attrs_.get_or_else(
                                        attr_keys::mixed_dtype, false)) {
                                dequantize_node = mgr.make("cast",
                                        dequantize_node->get_outputs(),
                                        std::vector<graph_tensor_ptr> {},
                                        {{"dtype", datatypes::bf16}});
                            }
                            cld_op->replace_input(use.first,
                                    dequantize_node->get_outputs()[0]);
                            vis->update_state_for_visited(dequantize_node);
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
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
