/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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
#include "graph/backend/graph_compiler/target_machine.hpp"

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
inline bool check_input_dtype(op_t *op) {
    for (size_t i = 0; i < op->num_inputs(); ++i) {
        const logical_tensor_t &iport
                = op->get_input_value(i)->get_logical_tensor();
        if (iport.data_type != DTYPE) return false;
    }

    return true;
}

inline bool check_reduce_attrs(op_t *op) {
    auto attrs = op->get_attributes();
    if (attrs.find(op_attr::axes) != attrs.end()
            && !attrs[op_attr::axes].get<std::vector<int64_t>>().empty()) {
        return true;
    }
    return false;
}

inline bool check_conv_attrs(op_t *op) {
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

// checks whether an op has no producer or wildcard producer
inline bool check_if_null_producer(op_t *op) {
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

// checks datatype and isa
inline bool check_isa_compatibility(op_t *op) {
    bool require_vnni = false;
    bool require_bf16 = false;
    for (const auto &in_value : op->get_input_values()) {
        auto in_dtype = in_value->get_logical_tensor().data_type;
        if (in_dtype == data_type::bf16) {
            require_bf16 = true;
        } else if (in_dtype == data_type::s8 || in_dtype == data_type::u8) {
            require_vnni = true;
        }
    }
    if (require_bf16) { return support_bf16(); }
    if (require_vnni) { return support_vnni(); }
    return true;
}

template <size_t N>
bool check_input_num(op_t *op) {
    return op->num_inputs() == N;
}

template <size_t N>
bool check_output_num(op_t *op) {
    return op->num_outputs() == N;
}

inline bool check_pooling_input_num(op_t *op) {
    if (op->get_kind() == graph::op_kind::AvgPoolBackward)
        return check_input_num<1>(op);
    return true;
}

inline pm::pb_node_t *append_single_op_repetition_subgraph(
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

inline pm::pb_node_t *create_dequant_matmul(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input,
        bool is_bf16 = false, bool is_int8 = false) {
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

inline const std::vector<graph::op_kind_t> &get_conv_forward_ops() {
    const static std::vector<graph::op_kind_t> conv_ops
            = {graph::op_kind::Convolution};
    return conv_ops;
}

inline const std::vector<graph::op_kind_t> &get_conv_backward_ops() {
    const static std::vector<graph::op_kind_t> conv_ops
            = {graph::op_kind::ConvolutionBackwardData,
                    graph::op_kind::ConvolutionBackwardWeights};
    return conv_ops;
}

inline const std::vector<graph::op_kind_t> &get_matmul_op() {
    const static std::vector<graph::op_kind_t> matmul_op
            = {graph::op_kind::MatMul};
    return matmul_op;
}

inline const std::vector<graph::op_kind_t> &get_pooling_ops() {
    const static std::vector<graph::op_kind_t> pooling_ops
            = {graph::op_kind::AvgPool, graph::op_kind::AvgPoolBackward,
                    graph::op_kind::MaxPool, graph::op_kind::MaxPoolBackward};
    return pooling_ops;
}

inline const std::vector<graph::op_kind_t> &get_reduction_ops() {
    const static std::vector<graph::op_kind_t> reduction_ops
            = {graph::op_kind::ReduceL1, graph::op_kind::ReduceL2,
                    graph::op_kind::ReduceMax, graph::op_kind::ReduceMin,
                    graph::op_kind::ReduceSum, graph::op_kind::ReduceProd,
                    graph::op_kind::ReduceMean};
    return reduction_ops;
}

inline const std::vector<graph::op_kind_t> &get_bn_training_ops() {
    const static std::vector<graph::op_kind_t> bn_training_ops
            = {graph::op_kind::BatchNormForwardTraining,
                    graph::op_kind::BatchNormTrainingBackward};
    return bn_training_ops;
}

inline const std::vector<graph::op_kind_t> get_no_constraint_ops() {
    auto supported_kinds = compiler_graph_impl_t::get_supported_op_kinds();
    std::vector<graph::op_kind_t> no_constraint_ops;
    std::vector<graph::op_kind_t> constraint_ops;
    constraint_ops.insert(constraint_ops.begin(),
            get_conv_forward_ops().begin(), get_conv_forward_ops().end());
    constraint_ops.insert(constraint_ops.begin(),
            get_conv_backward_ops().begin(), get_conv_backward_ops().end());
    constraint_ops.insert(constraint_ops.begin(), get_reduction_ops().begin(),
            get_reduction_ops().end());
    constraint_ops.insert(constraint_ops.begin(), get_pooling_ops().begin(),
            get_pooling_ops().end());
    constraint_ops.insert(constraint_ops.begin(), get_matmul_op().begin(),
            get_matmul_op().end());
    constraint_ops.insert(constraint_ops.begin(), get_bn_training_ops().begin(),
            get_bn_training_ops().end());
    for (const auto &kind : supported_kinds) {
        if (std::find(constraint_ops.begin(), constraint_ops.end(), kind)
                == constraint_ops.end()) {
            no_constraint_ops.push_back(kind);
        }
    }
    return no_constraint_ops;
}

} // namespace pass
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
