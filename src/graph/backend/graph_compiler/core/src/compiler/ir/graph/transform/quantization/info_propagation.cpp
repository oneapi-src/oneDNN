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
#include <utility>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/quantization/quantize_info.hpp>
#include <compiler/ir/graph/quantization/quantize_op.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace quantize {

static std::unordered_set<std::string> data_wei_op_set
        = {"conv_fwd_core", "matmul_core", "managed_matmul_core"};
// quantize awared nodes search starts from dequantize node and end in
// quantize node.
static std::vector<std::pair<int, sc_op_ptr>> find_quantize_aware_nodes(
        const sc_op_ptr &node) {
    if (node == nullptr) { return std::vector<std::pair<int, sc_op_ptr>>(); }
    std::vector<std::pair<int, sc_op_ptr>> aware_nodes;
    for (auto &child_lt : node->get_outputs()) {
        for (const auto &child_op : child_lt->uses_) {
            if (child_op.second->isa<concat_op_t>()) {
                // Concat op has multiple inputs whose data types must be same.
                // To assure this constraint, we do not consider it as quantize
                // aware node.
                continue;
            }
            if ((child_op.second->dyn_cast<op_traits::may_quantize_t>()
                        && child_op.second->attrs_.get_or_else(
                                sc_graph_t::attr_key_t::quantize, true))
                    || child_op.second->isa<cast_op_t>()) {
                aware_nodes.emplace_back(child_op);
            }
        }
    }
    return aware_nodes;
}

template <typename T>
void has_key_and_set(any_map_t &attrs, const std::string &key,
        const sc_op_ptr &aware_node,
        const std::string &new_key = std::string()) {
    if (attrs.has_key(key)) {
        aware_node->attrs_.set(
                new_key.empty() ? key : new_key, attrs.get<T>(key));
    }
}

static void propagate_quantize_info(const sc_op_ptr &quantize_node,
        const std::pair<int, sc_op_ptr> &aware_node) {
    assert((!(aware_node.second->attrs_.has_key(attr_keys::data_scales)
                    && aware_node.second->attrs_.has_key(
                            attr_keys::dyn_data_scales))
                   || !(aware_node.second->attrs_.has_key(
                                attr_keys::weight_scales)
                           && aware_node.second->attrs_.has_key(
                                   attr_keys::dyn_weight_scales)))
            && "aware node has been set a quantized info");
    if (data_wei_op_set.find(aware_node.second->op_name_)
            != data_wei_op_set.end()) {
        const auto qinfos = get_quantize_info_from_attrs(quantize_node->attrs_);
        std::string prefix, cur_dyn_scales, cur_dyn_zero_points;
        switch (aware_node.first) {
            case 0:
                prefix = "data_";
                cur_dyn_scales = attr_keys::dyn_data_scales;
                cur_dyn_zero_points = attr_keys::dyn_data_zero_points;
                break;
            case 1:
                prefix = "weight_";
                cur_dyn_scales = attr_keys::dyn_weight_scales;
                cur_dyn_zero_points = attr_keys::dyn_weight_zero_points;
                break;
            default: assert(0 && "invalid tensor type!"); break;
        };
        aware_node.second->attrs_.set(
                prefix + attr_keys::scales, qinfos.scales_);
        aware_node.second->attrs_.set(
                prefix + attr_keys::zero_points, qinfos.zero_points_);
        aware_node.second->attrs_.set(
                prefix + attr_keys::per_channel, qinfos.per_channel_);
        aware_node.second->attrs_.set(
                prefix + attr_keys::channel_axis, qinfos.channel_axis_);
        // dynamic quantize
        if (quantize_node->isa<dynamic_dequantize_op_t>()) {
            auto &inputs = quantize_node->get_inputs();
            aware_node.second->attrs_.set(cur_dyn_scales, inputs[1]);
            if (inputs.size() == 3) {
                aware_node.second->attrs_.set(cur_dyn_zero_points, inputs[2]);
            }
        } else {
            has_key_and_set<graph_tensor_ptr>(quantize_node->attrs_,
                    attr_keys::dyn_scales, aware_node.second, cur_dyn_scales);
            has_key_and_set<graph_tensor_ptr>(quantize_node->attrs_,
                    attr_keys::dyn_zero_points, aware_node.second,
                    cur_dyn_zero_points);
        }
        if (quantize_node->isa<dequantize_op_t>()
                || quantize_node->isa<dynamic_dequantize_op_t>()
                || quantize_node->attrs_.get_or_else(
                        attr_keys::may_quantize, false)) {
            if (!aware_node.second->attrs_.has_key(attr_keys::may_quantize)) {
                aware_node.second->attrs_.set(attr_keys::may_quantize, true);
            }
            if (aware_node.second->attrs_.get<bool>(attr_keys::may_quantize)) {
                aware_node.second->dyn_cast<op_traits::may_quantize_t>()
                        ->should_quantized_
                        = true;
            }
        } else {
            aware_node.second->attrs_.set(attr_keys::may_quantize, false);
            aware_node.second->dyn_cast<op_traits::may_quantize_t>()
                    ->should_quantized_
                    = false;
        }
    } else {
        // dynamic quantize
        if (quantize_node->isa<dynamic_dequantize_op_t>()) {
            auto &inputs = quantize_node->get_inputs();
            aware_node.second->attrs_.set(attr_keys::dyn_scales, inputs[1]);
            aware_node.second->attrs_.set(attr_keys::dyn_zero_points,
                    inputs.size() == 3 ? inputs[2] : nullptr);
        } else {
            has_key_and_set<graph_tensor_ptr>(quantize_node->attrs_,
                    attr_keys::dyn_scales, aware_node.second);
            has_key_and_set<graph_tensor_ptr>(quantize_node->attrs_,
                    attr_keys::dyn_zero_points, aware_node.second);
        }
        // static quantize
        has_key_and_set<std::vector<float>>(
                quantize_node->attrs_, attr_keys::scales, aware_node.second);
        has_key_and_set<std::vector<int>>(quantize_node->attrs_,
                attr_keys::zero_points, aware_node.second);
        // common propagation.
        has_key_and_set<bool>(quantize_node->attrs_, attr_keys::per_channel,
                aware_node.second);
        has_key_and_set<int>(quantize_node->attrs_, attr_keys::channel_axis,
                aware_node.second);
        bool is_transpose = aware_node.second->isa<transpose_op_t>(); // NOLINT
        if (is_transpose
                && quantize_node->attrs_.has_key(attr_keys::channel_axis)) {
            int channel_axis
                    = quantize_node->attrs_.get<int>(attr_keys::channel_axis);
            std::unordered_map<int, int> axis_map;
            auto order
                    = aware_node.second->attrs_.get<std::vector<int>>("order");
            for (size_t i = 0; i < order.size(); ++i) {
                axis_map[order[i]] = i;
            }
            aware_node.second->attrs_.set(
                    attr_keys::channel_axis, axis_map[channel_axis]);
        }
        has_key_and_set<bool>(quantize_node->attrs_, attr_keys::mixed_dtype,
                aware_node.second);
        has_key_and_set<std::vector<float>>(quantize_node->attrs_,
                attr_keys::data_scales, aware_node.second);
        has_key_and_set<std::vector<float>>(quantize_node->attrs_,
                attr_keys::weight_scales, aware_node.second);
        has_key_and_set<std::vector<int>>(quantize_node->attrs_,
                attr_keys::data_zero_points, aware_node.second);
        has_key_and_set<std::vector<int>>(quantize_node->attrs_,
                attr_keys::weight_zero_points, aware_node.second);
        has_key_and_set<graph_tensor_ptr>(quantize_node->attrs_,
                attr_keys::dyn_data_scales, aware_node.second);
        has_key_and_set<graph_tensor_ptr>(quantize_node->attrs_,
                attr_keys::dyn_weight_scales, aware_node.second);
        has_key_and_set<graph_tensor_ptr>(quantize_node->attrs_,
                attr_keys::dyn_data_zero_points, aware_node.second);
        has_key_and_set<graph_tensor_ptr>(quantize_node->attrs_,
                attr_keys::dyn_weight_zero_points, aware_node.second);
        has_key_and_set<int>(quantize_node->attrs_,
                attr_keys::weight_channel_axis, aware_node.second);
        if (quantize_node->isa<dequantize_op_t>()
                || quantize_node->isa<dynamic_dequantize_op_t>()
                || quantize_node->attrs_.get_or_else(
                        attr_keys::may_quantize, false)) {
            aware_node.second->attrs_.set(attr_keys::may_quantize, true);
            if (auto may_quantize_node
                    = aware_node.second
                              ->dyn_cast<op_traits::may_quantize_t>()) {
                may_quantize_node->should_quantized_ = true;
            }
        }
    }
}

static void check_and_set_mixed_dtype(const sc_op_ptr &cast_node) {
    assert(cast_node->isa<cast_op_t>());
    auto &attrs = cast_node->attrs_;
    // if after tunable op
    if (attrs.has_key(attr_keys::data_scales)
            || attrs.has_key(attr_keys::dyn_data_scales)) {
        assert((attrs.has_key(attr_keys::weight_scales)
                       || attrs.has_key(attr_keys::dyn_weight_scales))
                && !(attrs.has_key(attr_keys::scales)
                        || attrs.has_key(attr_keys::dyn_scales)));
    } else if (attrs.has_key(attr_keys::scales)
            || attrs.has_key(attr_keys::dyn_scales)) {
        assert(attrs.get<sc_data_type_t>(attr_keys::quan_dtype)
                == datatypes::bf16);
        attrs.set(attr_keys::mixed_dtype, true);
    }
}

// Do cast u8/s8 => s32 for dynamic zero points
void change_dyn_zp_to_s32(sc_graph_t &mgr, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(mgr.ops_.size());
    vis.visit_graph(mgr, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (node->isa<dynamic_quantize_op_t>()
                || node->isa<dynamic_dequantize_op_t>()) {
            if (node->get_inputs().size() == 3
                    && node->get_inputs()[2]->details_.dtype_
                            != datatypes::s32) {
                auto &zp = node->get_inputs()[2];
                auto casts32 = mgr.make(
                        "cast", {zp}, {}, {{"dtype", datatypes::s32}});
                node->replace_input(2, casts32->get_outputs()[0]);
            }
        }
    });
    mgr.reset_op_ids();
}

// Currenly we change u8 to s8 for weight
void change_weight_u8_to_s8(sc_graph_t &mgr, const context_ptr &ctx) {
    if (ctx->use_amx()) { return; }
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(mgr.ops_.size());
    vis.visit_graph(mgr, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (node->isa<dequantize_op_t>()
                || node->isa<dynamic_dequantize_op_t>()) {
            bool dyn_quan_cur = node->isa<dynamic_dequantize_op_t>();
            if (node->get_inputs()[0]->details_.dtype_ == datatypes::u8) {
                bool need_s8 = false;
                for (auto cld : node->get_outputs()[0]->uses_) {
                    while (!cld.second->isa<output_op>()
                            && !cld.second->isa<tunable_op_t>()
                            && cld.second->get_outputs().size() == 1) {
                        cld = cld.second->get_outputs()[0]->uses_[0];
                    }
                    if (data_wei_op_set.find(cld.second->op_name_)
                                    != data_wei_op_set.end()
                            && cld.first == 1) {
                        need_s8 = true;
                        break;
                    }
                }
                if (need_s8) {
                    auto *node_before = node->get_inputs()[0]->producer_owner_;
                    if (node_before->isa<quantize_op_t>()
                            || node_before->isa<dynamic_quantize_op_t>()) {
                        bool dyn_quan_before
                                = node_before->isa<dynamic_quantize_op_t>();
                        assert(node_before->attrs_.get_or_else(
                                       attr_keys::quan_dtype, datatypes::s8)
                                == datatypes::u8);
                        node_before->attrs_.get<sc_data_type_t>(
                                attr_keys::quan_dtype)
                                = datatypes::s8;
                        node->get_inputs()[0]->details_.dtype_ = datatypes::s8;
                        assert(node->attrs_.get_or_else(
                                       attr_keys::quan_dtype, datatypes::s8)
                                == datatypes::f32);
                        node->attrs_.get<sc_data_type_t>(attr_keys::quan_dtype)
                                = datatypes::f32;
                        if (dyn_quan_before) {
                            assert(node_before->get_inputs().size() == 3);
                            auto &zp = node_before->get_inputs()[2];
                            int value = 128;
                            auto const_128 = mgr.make("constant", {}, {},
                                    {{"dtype", datatypes::s32},
                                            {"format", sc_data_format_t()},
                                            {"values",
                                                    std::make_shared<
                                                            static_data_t>(
                                                            &value,
                                                            sizeof(int))},
                                            {"plain_dims", sc_dims {1}}});
                            auto shift_128 = mgr.make("sub",
                                    {zp, const_128->get_outputs()[0]}, {}, {});
                            node_before->replace_input(
                                    2, shift_128->get_outputs()[0]);
                            node->replace_input(2, shift_128->get_outputs()[0]);
                        } else {
                            assert(node_before->attrs_
                                            .get_or_else(attr_keys::zero_points,
                                                    std::vector<int>())
                                            .size()
                                    == 1);
                            assert(node->attrs_
                                            .get_or_else(attr_keys::zero_points,
                                                    std::vector<int>())
                                            .size()
                                    == 1);
                            node_before->attrs_.get<std::vector<int>>(
                                    attr_keys::zero_points)[0]
                                    -= 128;
                            node->attrs_.get<std::vector<int>>(
                                    attr_keys::zero_points)[0]
                                    -= 128;
                        }
                    } else {
                        auto *node_before
                                = node->get_inputs()[0]->producer_owner_;
                        const auto &qinfos
                                = get_quantize_info_from_attrs(node->attrs_);
                        auto new_zero_point = qinfos.zero_points_[0] - 128;
                        auto casts32
                                = mgr.make("cast", node_before->get_outputs(),
                                        {}, {{"dtype", datatypes::s32}});
                        auto const128 = mgr.make("constant", {}, {},
                                {{"values",
                                         std::make_shared<static_data_t>(
                                                 std::vector<int> {128})},
                                        {attr_keys::quan_dtype, datatypes::s32},
                                        {"plain_dims", sc_dims {1}}});
                        auto sub128 = mgr.make("sub",
                                {casts32->get_outputs()[0],
                                        const128->get_outputs()[0]},
                                {}, {});
                        auto casts8 = mgr.make("cast", sub128->get_outputs(),
                                {}, {{"dtype", datatypes::s8}});
                        sc_op_ptr deq;
                        if (dyn_quan_cur) {
                            assert(node->get_inputs().size() == 3);
                            auto &scales = node->get_inputs()[1];
                            auto &zp = node->get_inputs()[2];
                            auto shift_128 = mgr.make("sub",
                                    {zp, const128->get_outputs()[0]}, {}, {});
                            deq = mgr.make("dynamic_dequantize",
                                    {casts8->get_outputs()[0], scales,
                                            shift_128->get_outputs()[0]},
                                    {},
                                    {{attr_keys::quan_dtype, datatypes::f32}});
                        } else {
                            deq = mgr.make("dequantize", casts8->get_outputs(),
                                    {},
                                    {{attr_keys::quan_dtype, datatypes::f32},
                                            {attr_keys::scales, qinfos.scales_},
                                            {attr_keys::zero_points,
                                                    std::vector<int> {
                                                            new_zero_point}}});
                        }
                        node->replace_uses_with_and_remove(deq);
                        vis->update_state_for_visited(deq);
                    }
                }
            }
        }
    });
    mgr.reset_op_ids();
}

// do two things: infer input tensor is data/weight;transfer quantize info
// to calculation op
SC_INTERNAL_API void quantize_info_propagation(
        sc_graph_t &mgr, const context_ptr &ctx) {
    if (!mgr.attrs_.get_or_else(sc_graph_t::attr_key_t::quantize, false))
        return;
    change_dyn_zp_to_s32(mgr, ctx);
    change_weight_u8_to_s8(mgr, ctx);
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(mgr.ops_.size());
    vis.visit_graph(mgr, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (node->isa<dequantize_op_t>() || node->isa<dynamic_dequantize_op_t>()
                || (node->isa<op_traits::may_quantize_t>()
                        && !node->isa<concat_op_t>())
                || node->isa<cast_op_t>()) {
            if (node->isa<cast_op_t>()) { check_and_set_mixed_dtype(node); }
            auto aware_ops = find_quantize_aware_nodes(node);
            for (const auto &aware_op : aware_ops) {
                propagate_quantize_info(node, aware_op);
            }
        }
    });
}
} // namespace quantize
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
