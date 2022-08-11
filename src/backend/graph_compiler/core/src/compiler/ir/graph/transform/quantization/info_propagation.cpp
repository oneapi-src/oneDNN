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

namespace sc {
namespace quantize {

static std::unordered_set<std::string> data_wei_op_set
        = {"conv_fwd_core", "matmul_core"};
// quantize awared nodes search starts from dequantize node and end in
// quantize node.
static std::vector<std::pair<int, sc_op_ptr>> find_quantize_aware_nodes(
        const sc_op_ptr &node) {
    if (node == nullptr) { return std::vector<std::pair<int, sc_op_ptr>>(); }
    std::vector<std::pair<int, sc_op_ptr>> aware_nodes;
    for (auto &child_lt : node->get_outputs()) {
        for (const auto &child_op : child_lt->uses_) {
            if ((child_op.second->dyn_cast<op_traits::may_quantize_t>()
                        && child_op.second->attrs_.get_or_else(
                                "quantize", true))
                    || child_op.second->isa<cast_op_t>()) {
                aware_nodes.emplace_back(child_op);
            }
        }
    }
    return aware_nodes;
}

template <typename T>
void has_key_and_set(
        any_map_t &attrs, const std::string &key, const sc_op_ptr &aware_node) {
    if (attrs.has_key(key)) { aware_node->attrs_.set(key, attrs.get<T>(key)); }
}

static void propagate_quantize_info(const sc_op_ptr &quantize_node,
        const std::pair<int, sc_op_ptr> &aware_node) {
    assert((!(aware_node.second->attrs_.has_key("data_scales")
                    && aware_node.second->attrs_.has_key("data_zero_points")
                    && aware_node.second->attrs_.has_key("data_channel_axis"))
                   || !(aware_node.second->attrs_.has_key("weight_scales")
                           && aware_node.second->attrs_.has_key(
                                   "weight_zero_points")
                           && aware_node.second->attrs_.has_key(
                                   "weight_channel_axis")))
            && "aware node has been set a quantized info");
    if (data_wei_op_set.find(aware_node.second->op_name_)
            != data_wei_op_set.end()) {
        const auto qinfos = get_quantize_info_from_attrs(quantize_node->attrs_);
        std::string prefix;
        switch (aware_node.first) {
            case 0: prefix = "data_"; break;
            case 1: prefix = "weight_"; break;
            default: assert(0 && "invalid tensor type!"); break;
        };
        aware_node.second->attrs_.set(prefix + "scales", qinfos.scales_);
        aware_node.second->attrs_.set(
                prefix + "zero_points", qinfos.zero_points_);
        aware_node.second->attrs_.set(
                prefix + "per_channel", qinfos.per_channel_);
        aware_node.second->attrs_.set(
                prefix + "channel_axis", qinfos.channel_axis_);
        if (quantize_node->isa<dequantize_op_t>()
                || quantize_node->attrs_.get_or_else("may_quantize", false)) {
            if (!aware_node.second->attrs_.has_key("may_quantize")) {
                aware_node.second->attrs_.set("may_quantize", true);
            }
            if (aware_node.second->attrs_.get<bool>("may_quantize")) {
                aware_node.second->dyn_cast<op_traits::may_quantize_t>()
                        ->should_quantized_
                        = true;
            }
        } else {
            aware_node.second->attrs_.set("may_quantize", false);
            aware_node.second->dyn_cast<op_traits::may_quantize_t>()
                    ->should_quantized_
                    = false;
        }
    } else {
        has_key_and_set<std::vector<float>>(
                quantize_node->attrs_, "scales", aware_node.second);
        has_key_and_set<std::vector<int>>(
                quantize_node->attrs_, "zero_points", aware_node.second);
        has_key_and_set<bool>(
                quantize_node->attrs_, "per_channel", aware_node.second);
        has_key_and_set<int>(
                quantize_node->attrs_, "channel_axis", aware_node.second);
        bool is_transpose = aware_node.second->isa<transpose_op_t>(); // NOLINT
        if (is_transpose && quantize_node->attrs_.has_key("channel_axis")) {
            int channel_axis = quantize_node->attrs_.get<int>("channel_axis");
            std::unordered_map<int, int> axes_map;
            auto order
                    = aware_node.second->attrs_.get<std::vector<int>>("order");
            for (size_t i = 0; i < order.size(); ++i) {
                axes_map[order[i]] = i;
            }
            aware_node.second->attrs_.set(
                    "channel_axis", axes_map[channel_axis]);
        }
        has_key_and_set<bool>(
                quantize_node->attrs_, "mixed_dtype", aware_node.second);

        has_key_and_set<std::vector<float>>(
                quantize_node->attrs_, "data_scales", aware_node.second);
        has_key_and_set<std::vector<float>>(
                quantize_node->attrs_, "weight_scales", aware_node.second);
        has_key_and_set<std::vector<int>>(
                quantize_node->attrs_, "data_zero_points", aware_node.second);
        has_key_and_set<std::vector<int>>(
                quantize_node->attrs_, "weight_zero_points", aware_node.second);
        has_key_and_set<int>(
                quantize_node->attrs_, "data_channel_axis", aware_node.second);
        has_key_and_set<int>(quantize_node->attrs_, "weight_channel_axis",
                aware_node.second);
        if (quantize_node->isa<dequantize_op_t>()
                || quantize_node->attrs_.get_or_else("may_quantize", false)) {
            aware_node.second->attrs_.set("may_quantize", true);
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
    if (attrs.has_key("data_scales")) {
        assert(attrs.has_key("weight_scales") && !attrs.has_key("scales"));
    } else if (attrs.has_key("scales")) {
        assert(attrs.get<sc_data_type_t>("dtype") == datatypes::bf16);
        attrs.set("mixed_dtype", true);
    }
}

// Currenly we change u8 to s8 for weight
void change_weight_u8_to_s8(sc_graph_t &mgr, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(mgr.ops_.size());
    vis.visit_graph(mgr, [&](const sc_op_ptr &node) {
        if (auto dequantize_node = node->dyn_cast<dequantize_op_t>()) {
            if (node->get_inputs()[0]->details_.dtype_ == datatypes::u8) {
                bool need_s8 = false;
                for (auto cld : node->get_outputs()[0]->uses_) {
                    while (!cld.second->isa<output_op>()
                            && cld.second->get_inputs().size() == 1
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
                    if (auto quantize_node
                            = node->get_inputs()[0]
                                      ->producer_owner_
                                      ->dyn_cast<quantize_op_t>()) {
                        assert(quantize_node->attrs_.get_or_else(
                                       "dtype", datatypes::s8)
                                == datatypes::u8);
                        assert(quantize_node->attrs_
                                        .get_or_else("zero_points",
                                                std::vector<int>())
                                        .size()
                                == 1);
                        quantize_node->attrs_.get<sc_data_type_t>("dtype")
                                = datatypes::s8;
                        quantize_node->attrs_.get<std::vector<int>>(
                                "zero_points")[0]
                                -= 128;
                        node->get_inputs()[0]->details_.dtype_ = datatypes::s8;
                        assert(dequantize_node->attrs_.get_or_else(
                                       "dtype", datatypes::s8)
                                == datatypes::f32);
                        assert(dequantize_node->attrs_
                                        .get_or_else("zero_points",
                                                std::vector<int>())
                                        .size()
                                == 1);
                        dequantize_node->attrs_.get<sc_data_type_t>("dtype")
                                = datatypes::f32;
                        dequantize_node->attrs_.get<std::vector<int>>(
                                "zero_points")[0]
                                -= 128;
                    } else {
                        auto *node_before
                                = node->get_inputs()[0]->producer_owner_;
                        const auto &qinfos = get_quantize_info_from_attrs(
                                dequantize_node->attrs_);
                        auto new_zero_point = qinfos.zero_points_[0] - 128;
                        auto casts32
                                = mgr.make("cast", node_before->get_outputs(),
                                        {}, {{"dtype", datatypes::s32}});
                        auto const128 = mgr.make("constant", {}, {},
                                {{"values",
                                         std::make_shared<static_data_t>(
                                                 std::vector<int> {128})},
                                        {"dtype", datatypes::s32},
                                        {"plain_dims", sc_dims {1}}});
                        auto sub128 = mgr.make("sub",
                                {casts32->get_outputs()[0],
                                        const128->get_outputs()[0]},
                                {}, {});
                        auto casts8 = mgr.make("cast", sub128->get_outputs(),
                                {}, {{"dtype", datatypes::s8}});
                        auto deq = mgr.make("dequantize", casts8->get_outputs(),
                                {},
                                {{"dtype", datatypes::f32},
                                        {"scales", qinfos.scales_},
                                        {"zero_points",
                                                std::vector<int> {
                                                        new_zero_point}}});
                        node->replace_uses_with_and_remove(deq);
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
    change_weight_u8_to_s8(mgr, ctx);
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(mgr.ops_.size());
    vis.visit_graph(mgr, [&](const sc_op_ptr &node) {
        if (node->isa<dequantize_op_t>()
                || node->isa<op_traits::may_quantize_t>()
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
} // namespace sc
