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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_FUSIONS_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_FUSIONS_HPP

#include <vector>

#include "graph/backend/graph_compiler/patterns/transformation_pattern.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {
namespace pass {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

template <graph::data_type_t DTYPE>
bool check_input_dtype(op_t *op) {
    for (size_t i = 0; i < op->num_inputs(); ++i) {
        const logical_tensor_t &iport
                = op->get_input_value(i)->get_logical_tensor();
        if (iport.data_type != DTYPE) return false;
    }

    return true;
}

bool check_reduce_attrs(op_t *op) {
    auto attrs = op->get_attributes();
    if (attrs.find(op_attr::axes) != attrs.end()
            && !attrs[op_attr::axes].get<std::vector<int64_t>>().empty()) {
        return true;
    }
    return false;
}

bool check_conv_attrs(op_t *op) {
    auto attrs = op->get_attributes();
    // dilations must be {1, 1, ...}
    if (attrs.find(graph::op_attr::dilations) != attrs.end()) {
        auto dilations
                = attrs[graph::op_attr::dilations].get<std::vector<int64_t>>();
        if (!std::all_of(dilations.begin(), dilations.end(),
                    [&](const int64_t &d) { return d == 1; })) {
            return false;
        }
    }
    // groups must be 1
    if (attrs.find(graph::op_attr::groups) != attrs.end()
            && attrs[graph::op_attr::groups].get<int64_t>() != 1) {
        return false;
    }
    // preferred to be a 2D conv
    auto strides = attrs[graph::op_attr::strides].get<std::vector<int64_t>>();
    if (strides.size() != 2) { return false; }
    // preferred to be symmetric padding
    // if no auto_pad set, needs to check pads_begin == pads_end
    if (attrs.find(op_attr::auto_pad) == attrs.end()) {
        auto pads_begin
                = attrs[graph::op_attr::pads_begin].get<std::vector<int64_t>>();
        auto pads_end
                = attrs[graph::op_attr::pads_end].get<std::vector<int64_t>>();
        if (pads_begin != pads_end) { return false; }
    }
    return true;
}

bool check_select(op_t *op) {
    auto inputs = op->get_input_values();
    if (inputs.size() != 3) return false;
    // input[1]'s shape shall be either {1} or a 0-dim tensor
    if (inputs[1]->get_logical_tensor().ndims == 0
            || (inputs[1]->get_logical_tensor().ndims == 1
                    && inputs[1]->get_logical_tensor().dims[0] == 1))
        return true;
    return false;
}

// checks whether an op has no producer or wildcard producer
bool check_if_null_producer(op_t *op) {
    bool null_producer = true;
    for (const auto &in_value : op->get_input_values()) {
        if (in_value->has_producer()) {
            null_producer = null_producer
                    && in_value->get_producer().get_kind()
                            == graph::op_kind::Wildcard;
        }
    }
    return null_producer;
}

pm::pb_node_t *append_single_op_repetition_subgraph(
        const std::shared_ptr<pb_graph_t> &pgraph, graph::op_kind_t kind,
        pm::pb_node_t *input, int rep_min = 0, int rep_max = 2) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    auto rep_subgraph = std::make_shared<pb_graph_t>();
    auto single_op = rep_subgraph->append_op(kind);
    rep_subgraph->create_input_port(0, single_op, 0);
    rep_subgraph->create_output_port(0, single_op, 0);
    auto rep = pgraph->append_repetition(
            rep_subgraph, {0, 0}, rep_min, rep_max, in_edges);
    return rep;
};

pm::pb_node_t *create_dequant_matmul(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_node_t *input, bool is_bf16 = false, bool is_int8 = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    if (is_int8) {
        auto dequantize_A
                = pgraph->append_op(graph::op_kind::Dequantize, in_edges);
        auto dequantize_B = pgraph->append_op(graph::op_kind::Dequantize);
        if (is_bf16) {
            auto typecast_A = pgraph->append_op(
                    graph::op_kind::TypeCast, {in_edge(0, dequantize_A, 0)});
            auto typecast_B = pgraph->append_op(
                    graph::op_kind::TypeCast, {in_edge(0, dequantize_B, 0)});
            in_edges = in_edges_t {
                    in_edge(0, typecast_A, 0), in_edge(1, typecast_B, 0)};
        } else {
            in_edges = in_edges_t {
                    in_edge(0, dequantize_A, 0), in_edge(1, dequantize_B, 0)};
        }
    }
    auto matmul = pgraph->append_op(graph::op_kind::MatMul, in_edges);
    matmul->append_decision_function(is_bf16
                    ? check_input_dtype<graph::data_type::bf16>
                    : check_input_dtype<graph::data_type::f32>);
    return matmul;
}

} // namespace pass
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
