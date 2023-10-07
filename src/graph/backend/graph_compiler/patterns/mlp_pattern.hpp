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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_MLP_PATTERN_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_MLP_PATTERN_HPP

#include <memory>
#include <utility>

#include "graph/backend/graph_compiler/patterns/fusions.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {
namespace pass {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = graph::utils::pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

#define MLP_NUM_LAYER_LOWER_BOUND 2
#define MLP_NUM_LAYER_UPPER_BOUND 11

std::pair<pm::pb_node_t *, pm::pb_node_t *> single_layer_mlp(
        const std::shared_ptr<pb_graph_t> &pgraph, bool is_bf16 = false,
        bool is_int8 = false) {
    pm::pb_node_t *layer_input, *layer_output;
    in_edges_t matmul_in_edges;
    if (is_int8) {
        auto dequantize_input
                = pgraph->append_alternation({graph::op_kind::Dequantize,
                        graph::op_kind::DynamicDequantize});
        auto dequantize_weight
                = pgraph->append_alternation({graph::op_kind::Dequantize,
                        graph::op_kind::DynamicDequantize});
        if (is_bf16) {
            auto typecast_input = pgraph->append_op(graph::op_kind::TypeCast,
                    {in_edge(0, dequantize_input, 0)});
            auto typecast_weight = pgraph->append_op(graph::op_kind::TypeCast,
                    {in_edge(0, dequantize_weight, 0)});
            matmul_in_edges = in_edges_t {in_edge(0, typecast_input, 0),
                    in_edge(1, typecast_weight, 0)};
        } else {
            matmul_in_edges = in_edges_t {in_edge(0, dequantize_input, 0),
                    in_edge(1, dequantize_weight, 0)};
        }
        layer_input = dequantize_input;
    }
    auto matmul = pgraph->append_op(graph::op_kind::MatMul, matmul_in_edges);
    matmul->append_decision_function(is_bf16
                    ? check_input_dtype<graph::data_type::bf16>
                    : check_input_dtype<graph::data_type::f32>);
    matmul->allow_external_outputs();
    if (!is_int8) { layer_input = matmul; }

    /* optional add/biasAdd after matmul */
    auto add_subgraph = std::make_shared<pb_graph_t>();
    auto add = add_subgraph->append_alternation(
            {graph::op_kind::Add, graph::op_kind::BiasAdd});
    add->allow_external_outputs();
    add_subgraph->create_input_port(0, add, 0);
    add_subgraph->create_output_port(0, add, 0);
    auto optional_add
            = pgraph->append_optional(add_subgraph, {in_edge(0, matmul, 0)});

    /* optional activation */
    auto activation_subgraph = std::make_shared<pb_graph_t>();
    auto activation = activation_subgraph->append_alternation(
            {graph::op_kind::ReLU, graph::op_kind::Sigmoid});
    activation->allow_external_outputs();
    activation_subgraph->create_input_port(0, activation, 0);
    activation_subgraph->create_output_port(0, activation, 0);
    auto optional_activation = pgraph->append_optional(
            activation_subgraph, {in_edge(0, optional_add, 0)});
    layer_output = optional_activation;

    if (is_int8) {
        if (is_bf16) {
            auto typecast_output = pgraph->append_op(
                    graph::op_kind::TypeCast, {in_edge(0, layer_output, 0)});
            layer_output = typecast_output;
        }
        auto quantize_output = pgraph->append_alternation(
                {graph::op_kind::Quantize, graph::op_kind::DynamicQuantize},
                {in_edge(0, layer_output, 0)});
        layer_output = quantize_output;
    }
    return std::make_pair(layer_input, layer_output);
}

pm::pb_node_t *weight_grad_alternation_unit(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *activation) {
    /* Create 2 subgraph for alternation */
    auto weight_grad_option1 = std::make_shared<pb_graph_t>();
    auto transpose_subgraph1 = std::make_shared<pb_graph_t>();
    auto transpose_x1
            = transpose_subgraph1->append_op(graph::op_kind::StaticTranspose);
    transpose_subgraph1->create_input_port(0, transpose_x1, 0);
    transpose_subgraph1->create_output_port(0, transpose_x1, 0);
    auto optional_transpose_x
            = weight_grad_option1->append_optional(transpose_subgraph1);
    auto matmul_grad_w1 = weight_grad_option1->append_op(
            graph::op_kind::MatMul, {in_edge(0, optional_transpose_x, 0)});
    weight_grad_option1->create_input_port(0, matmul_grad_w1, 1);
    weight_grad_option1->create_output_port(0, matmul_grad_w1, 0);

    auto weight_grad_option2 = std::make_shared<pb_graph_t>();
    auto transpose_x2
            = weight_grad_option2->append_op(graph::op_kind::StaticTranspose);
    auto matmul_grad_w2 = weight_grad_option2->append_op(
            graph::op_kind::MatMul, {in_edge(0, transpose_x2, 0)});
    weight_grad_option2->create_input_port(0, transpose_x2, 0);
    weight_grad_option2->create_output_port(0, matmul_grad_w2, 0);

    auto weight_grad = pgraph->append_alternation(
            {weight_grad_option1, weight_grad_option2},
            {in_edge(0, activation, 0)});
    return weight_grad;
};

pm::pb_node_t *append_optional_quant_dequant(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input,
        bool is_mixed_dtype = false) {
    auto quant_dequant_subgraph = std::make_shared<pb_graph_t>();
    pm::pb_op_t *typecast1 = nullptr, *typecast2 = nullptr;
    in_edges_t quant_in_edges;
    if (is_mixed_dtype) {
        typecast1 = quant_dequant_subgraph->append_op(graph::op_kind::TypeCast);
        quant_in_edges = in_edges_t {in_edge(0, typecast1, 0)};
    }
    auto quant = quant_dequant_subgraph->append_op(
            graph::op_kind::Quantize, quant_in_edges);
    auto dequant = quant_dequant_subgraph->append_op(
            graph::op_kind::Dequantize, {in_edge(0, quant, 0)});
    if (is_mixed_dtype) {
        typecast2 = quant_dequant_subgraph->append_op(
                graph::op_kind::TypeCast, {in_edge(0, dequant, 0)});
    }
    quant_dequant_subgraph->create_input_port(
            0, is_mixed_dtype ? typecast1 : quant, 0);
    quant_dequant_subgraph->create_output_port(
            0, is_mixed_dtype ? typecast2 : dequant, 0);
    auto optional_quant_dequant = pgraph->append_optional(
            quant_dequant_subgraph, {in_edge(0, input, 0)});
    return optional_quant_dequant;
};

pm::pb_node_t *append_gelu_subgraph(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input) {
    auto pow = pgraph->append_op(graph::op_kind::Pow, {in_edge(0, input, 0)});
    auto mul1
            = pgraph->append_op(graph::op_kind::Multiply, {in_edge(0, pow, 0)});
    auto add1 = pgraph->append_op(
            graph::op_kind::Add, {in_edge(0, input, 0), in_edge(1, mul1, 0)});
    auto mul2 = pgraph->append_op(
            graph::op_kind::Multiply, {in_edge(0, add1, 0)});
    auto tanh = pgraph->append_op(graph::op_kind::Tanh, {in_edge(0, mul2, 0)});
    auto add2 = pgraph->append_op(graph::op_kind::Add, {in_edge(0, tanh, 0)});
    auto mul3 = pgraph->append_op(
            graph::op_kind::Multiply, {in_edge(0, input, 0)});
    auto mul4 = pgraph->append_op(graph::op_kind::Multiply,
            {in_edge(0, add2, 0), in_edge(1, mul3, 0)});
    return mul4;
};

/*
    [IN0]    [IN1]
       \      /
        MatMul
          |
        GELU    [IN2]  [IN3]    [IN4]
           \     /        \      /
            MatMul         MatMul
                \___    ___/
                    Add
                     |
                    Add
                     |
                 LayerNorm
                     |
                   [OUT]
*/
void create_gpt_mlp(const std::shared_ptr<pb_graph_t> &pgraph,
        bool gelu_subgraph = false, bool is_bf16 = false, bool is_int8 = false,
        bool quantize_output = true) {
    auto matmul1 = create_dequant_matmul(pgraph, nullptr, is_bf16, is_int8);
    pm::pb_node_t *gelu;
    if (gelu_subgraph) {
        gelu = append_gelu_subgraph(pgraph, matmul1);
    } else {
        gelu = pgraph->append_op(
                graph::op_kind::GELU, {in_edge(0, matmul1, 0)});
    }

    if (is_int8) {
        if (is_bf16) {
            auto typecast1 = pgraph->append_op(
                    graph::op_kind::TypeCast, {in_edge(0, gelu, 0)});
            gelu = typecast1;
        }
        auto smooth_quant_mul1 = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::Multiply, gelu);
        auto extra_casts1 = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::TypeCast, smooth_quant_mul1, 0, 3);
        auto quant1 = pgraph->append_op(
                graph::op_kind::Quantize, {in_edge(0, extra_casts1, 0)});
        gelu = quant1;
    }
    auto matmul2 = create_dequant_matmul(pgraph, gelu, is_bf16, is_int8);
    auto matmul3 = create_dequant_matmul(pgraph, nullptr, is_bf16, is_int8);
    auto qdq_matmul2 = append_optional_quant_dequant(pgraph, matmul2, is_bf16);
    auto qdq_matmul3 = append_optional_quant_dequant(pgraph, matmul3, is_bf16);
    auto add3 = pgraph->append_op(graph::op_kind::Add,
            {in_edge(0, qdq_matmul2, 0), in_edge(1, qdq_matmul3, 0)});
    auto add4 = pgraph->append_op(graph::op_kind::Add, {in_edge(0, add3, 0)});
    add4->allow_external_outputs(); // residual edge to next mlp
    auto layernorm = pgraph->append_op(
            graph::op_kind::LayerNorm, {in_edge(0, add4, 0)});
    layernorm->allow_external_outputs();
    if (is_int8 && quantize_output) {
        if (is_bf16) {
            auto typecast4 = pgraph->append_op(
                    graph::op_kind::TypeCast, {in_edge(0, layernorm, 0)});
            typecast4->allow_external_outputs();
            layernorm = typecast4;
        }
        auto smooth_quant_mul4 = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::Multiply, layernorm);
        auto extra_casts2 = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::TypeCast, smooth_quant_mul4, 0, 3);
        auto quant4 = pgraph->append_op(
                graph::op_kind::Quantize, {in_edge(0, extra_casts2, 0)});
        UNUSED(quant4);
    }
};

pm::pb_node_t *append_rms_norm_option1(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input,
        bool is_bf16 = false, bool is_int8 = false) {
    if (is_bf16) {
        auto typecast = pgraph->append_op(
                graph::op_kind::TypeCast, {in_edge(0, input, 0)});
        input = typecast;
    }
    auto pow = pgraph->append_op(graph::op_kind::Pow, {in_edge(0, input, 0)});
    auto mean = pgraph->append_op(
            graph::op_kind::ReduceMean, {in_edge(0, pow, 0)});
    auto add = pgraph->append_op(graph::op_kind::Add, {in_edge(0, mean, 0)});
    auto rsqrt = pgraph->append_op(graph::op_kind::Pow, {in_edge(0, add, 0)});
    auto mul1 = pgraph->append_op(graph::op_kind::Multiply,
            {in_edge(0, input, 0), in_edge(1, rsqrt, 0)});
    auto cast1 = append_single_op_repetition_subgraph(
            pgraph, graph::op_kind::TypeCast, mul1, 0, 3);
    auto mul2 = pgraph->append_op(
            graph::op_kind::Multiply, {in_edge(0, cast1, 0)});
    mul2->allow_external_outputs();
    UNUSED(is_bf16);
    UNUSED(is_int8);
    return mul2;
};

pm::pb_node_t *append_rms_norm_option2(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input,
        bool is_bf16 = false, bool is_int8 = false) {
    pm::pb_node_t *pow_in = input;
    pm::pb_node_t *mul1_in = input;
    if (is_bf16) {
        auto typecast1 = pgraph->append_op(
                graph::op_kind::TypeCast, {in_edge(0, input, 0)});
        pow_in = typecast1;
    }
    if (is_bf16) {
        auto typecast2 = pgraph->append_op(
                graph::op_kind::TypeCast, {in_edge(0, input, 0)});
        mul1_in = typecast2;
    }
    auto pow = pgraph->append_op(graph::op_kind::Pow, {in_edge(0, pow_in, 0)});
    auto mean = pgraph->append_op(
            graph::op_kind::ReduceMean, {in_edge(0, pow, 0)});
    auto add = pgraph->append_op(graph::op_kind::Add, {in_edge(0, mean, 0)});
    auto rsqrt = pgraph->append_op(graph::op_kind::Pow, {in_edge(0, add, 0)});
    auto mul1 = pgraph->append_op(graph::op_kind::Multiply,
            {in_edge(0, mul1_in, 0), in_edge(1, rsqrt, 0)});
    auto cast1 = append_single_op_repetition_subgraph(
            pgraph, graph::op_kind::TypeCast, mul1, 0, 3);
    auto mul2 = pgraph->append_op(
            graph::op_kind::Multiply, {in_edge(0, cast1, 0)});
    mul2->allow_external_outputs();
    UNUSED(is_bf16);
    UNUSED(is_int8);
    return mul2;
};

/*
    [IN0]    [IN1]
       \      /
        MatMul
          |
         Add ______________________
          |                        |
          |                    RMSNorm_____     [IN2]
          |                   /             \    /
          |                  /               MatMul
          |                  \     [IN3]    /     \
          |                   \    /       |    Sigmoid
          |                   MatMul        \     /
          |                       \         Multiply
          |                        \        /
          |                         Multiply   [IN4]
          |                               \    /
          |                               MatMul
          | ________________________________|
         Add
          |
       RMSNorm
          |
        [OUT]
*/
void create_llama_mlp(const std::shared_ptr<pb_graph_t> &pgraph,
        bool is_bf16 = false, bool is_int8 = false,
        bool use_rms_norm_alternative = false,
        bool split_smooth_quant = false) {
    auto matmul1 = create_dequant_matmul(pgraph, nullptr, is_bf16, is_int8);
    auto add1
            = pgraph->append_op(graph::op_kind::Add, {in_edge(0, matmul1, 0)});
    add1->allow_external_outputs();
    auto norm1 = use_rms_norm_alternative
            ? append_rms_norm_option1(pgraph, add1, is_bf16, is_int8)
            : append_rms_norm_option2(pgraph, add1, is_bf16, is_int8);

    pm::pb_node_t *norm1_for_lhs = norm1, *norm1_for_rhs = norm1;
    if (is_int8) {
        auto extra_cast_before_mul = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::TypeCast, norm1);
        auto smooth_quant_mul1 = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::Multiply, extra_cast_before_mul);
        auto extra_cast_after_mul = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::TypeCast, smooth_quant_mul1, 0, 3);
        auto quant1 = pgraph->append_op(graph::op_kind::Quantize,
                {in_edge(0, extra_cast_after_mul, 0)});
        if (split_smooth_quant) {
            auto extra_cast_before_mul_rhs
                    = append_single_op_repetition_subgraph(
                            pgraph, graph::op_kind::TypeCast, norm1);
            auto smooth_quant_mul1_rhs = append_single_op_repetition_subgraph(
                    pgraph, graph::op_kind::Multiply,
                    extra_cast_before_mul_rhs);
            auto extra_cast_after_mul_rhs
                    = append_single_op_repetition_subgraph(pgraph,
                            graph::op_kind::TypeCast, smooth_quant_mul1_rhs, 0,
                            3);
            auto quant1_rhs = pgraph->append_op(graph::op_kind::Quantize,
                    {in_edge(0, extra_cast_after_mul_rhs, 0)});
            norm1_for_lhs = quant1;
            norm1_for_rhs = quant1_rhs;
        } else {
            norm1_for_lhs = quant1;
            norm1_for_rhs = quant1;
        }
    }

    auto matmul2
            = create_dequant_matmul(pgraph, norm1_for_lhs, is_bf16, is_int8);
    auto silu_sigmoid = pgraph->append_op(
            graph::op_kind::Sigmoid, {in_edge(0, matmul2, 0)});
    auto silu_mul = pgraph->append_op(graph::op_kind::Multiply,
            {in_edge(0, matmul2, 0), in_edge(1, silu_sigmoid, 0)});

    auto matmul3
            = create_dequant_matmul(pgraph, norm1_for_rhs, is_bf16, is_int8);

    pm::pb_node_t *mul = pgraph->append_op(graph::op_kind::Multiply,
            {in_edge(0, silu_mul, 0), in_edge(1, matmul3, 0)});

    if (is_int8) {
        auto extra_cast_before_mul = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::TypeCast, mul);
        auto smooth_quant_mul2 = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::Multiply, extra_cast_before_mul);
        auto extra_cast_after_mul = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::TypeCast, smooth_quant_mul2, 0, 3);
        auto quant2 = pgraph->append_op(graph::op_kind::Quantize,
                {in_edge(0, extra_cast_after_mul, 0)});
        mul = quant2;
    }

    auto matmul4 = create_dequant_matmul(pgraph, mul, is_bf16, is_int8);
    auto add2 = pgraph->append_op(
            graph::op_kind::Add, {in_edge(0, matmul4, 0), in_edge(1, add1, 0)});
    add2->allow_external_outputs();
    auto norm2 = use_rms_norm_alternative
            ? append_rms_norm_option1(pgraph, add2, is_bf16, is_int8)
            : append_rms_norm_option2(pgraph, add2, is_bf16, is_int8);
    if (is_int8) {
        auto extra_cast_before_mul = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::TypeCast, norm2);
        auto smooth_quant_mul3 = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::Multiply, extra_cast_before_mul);
        auto extra_cast_after_mul = append_single_op_repetition_subgraph(
                pgraph, graph::op_kind::TypeCast, smooth_quant_mul3, 0, 3);
        auto quant3 = pgraph->append_op(graph::op_kind::Quantize,
                {in_edge(0, extra_cast_after_mul, 0)});
        UNUSED(quant3);
    }
};

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(fp32_mlp_pattern)

/*
repetition unit:
  (f32)[REP_IN0]   [REP_IN1](f32)
              \     /
               MatMul
                 |
                Add (optional)
                 |
             Activation (optional)
                 |
             [REP_OUT0](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mlp_forward_pattern)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto mlp_layer = std::make_shared<pb_graph_t>();
                    pm::pb_node_t *matmul, *optional_activation;
                    std::tie(matmul, optional_activation)
                            = single_layer_mlp(mlp_layer, false, false);
                    mlp_layer->create_input_port(0, matmul, 0);
                    mlp_layer->create_output_port(0, optional_activation, 0);
                    // repeat layer for [LOWER_BOUND, UPPER_BOUND) times
                    pgraph->append_repetition(mlp_layer, {0, 0},
                            MLP_NUM_LAYER_LOWER_BOUND,
                            MLP_NUM_LAYER_UPPER_BOUND);
                });

/*
[repetition unit]:
          (f32)[x_next]     [gradient_x_next](f32)
                      \       /     [weight](f32)
                       \     /           |
                       Backward    StaticTranspose
                     /    |   \     /     (optional)
    [grad_w_subgraph]  Reduce  Matmul
             |            |      |
       [gradient_w](f32)  |  [gradient_x](f32)
                          |
                  [gradient_bias](f32)

[optional unit]:
          (f32)[x_next]   [repetition_unit_out](f32)
                      \       /
                       \     /
                      Backward
                     /    |
      [grad_w_subgraph] Reduce
             |            |
       [gradient_w](f32)  |
                          |
                  [gradient_bias](f32)

pattern:
        [repetition unit]*[1-10]
                |
        [optional unit] (optional)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mlp_backward_pattern)
        .set_priority(5.1f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bwd_mlp_layer = std::make_shared<pb_graph_t>();
                    auto activation_bwd = bwd_mlp_layer->append_alternation(
                            {graph::op_kind::ReLUBackward,
                                    graph::op_kind::SigmoidBackward});
                    activation_bwd->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);

                    weight_grad_alternation_unit(bwd_mlp_layer, activation_bwd);

                    auto transpose_subgraph2 = std::make_shared<pb_graph_t>();
                    auto transpose_w = transpose_subgraph2->append_op(
                            graph::op_kind::StaticTranspose);
                    transpose_subgraph2->create_input_port(0, transpose_w, 0);
                    transpose_subgraph2->create_output_port(0, transpose_w, 0);
                    auto optional_transpose_w = bwd_mlp_layer->append_optional(
                            transpose_subgraph2);
                    auto matmul_layer = bwd_mlp_layer->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, activation_bwd, 0),
                                    in_edge(1, optional_transpose_w, 0)});

                    auto reduce_bias = bwd_mlp_layer->append_op(
                            graph::op_kind::ReduceSum,
                            {in_edge(0, activation_bwd, 0)});
                    reduce_bias->append_decision_function(check_reduce_attrs);

                    bwd_mlp_layer->create_input_port(0, activation_bwd, 1);
                    bwd_mlp_layer->create_output_port(0, matmul_layer, 0);

                    // repeat layer for [LOWER_BOUND - 1, UPPER_BOUND) times
                    auto repetition = pgraph->append_repetition(bwd_mlp_layer,
                            {0, 0}, MLP_NUM_LAYER_LOWER_BOUND - 1,
                            MLP_NUM_LAYER_UPPER_BOUND);

                    // start append the optional last layer
                    auto last_layer = std::make_shared<pb_graph_t>();
                    auto activation_bwd_last = last_layer->append_alternation(
                            {graph::op_kind::ReLUBackward,
                                    graph::op_kind::SigmoidBackward});
                    activation_bwd_last->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    // still allow extra grad_input computation
                    activation_bwd_last->allow_external_outputs();

                    auto weight_grad = weight_grad_alternation_unit(
                            last_layer, activation_bwd_last);
                    auto reduce_bias_last
                            = last_layer->append_op(graph::op_kind::ReduceSum,
                                    {in_edge(0, activation_bwd_last, 0)});
                    reduce_bias_last->append_decision_function(
                            check_reduce_attrs);
                    last_layer->create_input_port(0, activation_bwd_last, 1);
                    last_layer->create_output_port(0, weight_grad, 0);
                    pgraph->append_optional(
                            last_layer, {in_edge(0, repetition, 0)});
                });

/*
[repetition unit]:
           (f32)[x_next]     [gradient_x_next](bf16)
        [x](f32)      \       /     [weight](f32)
            |          \     /           |
     StaticTranspose   Backward    StaticTranspose
(optional)     \     /        \     /     (optional)
                Matmul         Matmul
                   |             |
       [gradient_w](f32)     [gradient_x](f32)

[optional unit]:
           (f32)[x_next]   [repetition_unit_out](f32)
        [x](f32)      \       /
            |          \     /
     StaticTranspose  Backward
(optional)     \     /
                Matmul
                   |
          [gradient_w](f32)

pattern:
        [repetition unit]*[1-10]
                |
         [optional unit] (optional)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mlp_backward_pattern_v2)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bwd_mlp_layer = std::make_shared<pb_graph_t>();
                    auto activation_bwd = bwd_mlp_layer->append_alternation(
                            {graph::op_kind::ReLUBackward,
                                    graph::op_kind::SigmoidBackward});
                    activation_bwd->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    activation_bwd->allow_external_outputs();

                    auto transpose_subgraph1 = std::make_shared<pb_graph_t>();
                    auto transpose_x = transpose_subgraph1->append_op(
                            graph::op_kind::StaticTranspose);
                    transpose_subgraph1->create_input_port(0, transpose_x, 0);
                    transpose_subgraph1->create_output_port(0, transpose_x, 0);
                    auto optional_transpose_x = bwd_mlp_layer->append_optional(
                            transpose_subgraph1);

                    auto transpose_subgraph2 = std::make_shared<pb_graph_t>();
                    auto transpose_w = transpose_subgraph2->append_op(
                            graph::op_kind::StaticTranspose);
                    transpose_subgraph2->create_input_port(0, transpose_w, 0);
                    transpose_subgraph2->create_output_port(0, transpose_w, 0);
                    auto optional_transpose_w = bwd_mlp_layer->append_optional(
                            transpose_subgraph2);

                    bwd_mlp_layer->append_op(graph::op_kind::MatMul,
                            {in_edge(0, optional_transpose_x, 0),
                                    in_edge(1, activation_bwd, 0)});
                    auto matmul_layer = bwd_mlp_layer->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, activation_bwd, 0),
                                    in_edge(1, optional_transpose_w, 0)});

                    bwd_mlp_layer->create_input_port(0, activation_bwd, 1);
                    bwd_mlp_layer->create_output_port(0, matmul_layer, 0);

                    // repeat layer for [LOWER_BOUND - 1, UPPER_BOUND) times
                    auto repetition = pgraph->append_repetition(bwd_mlp_layer,
                            {0, 0}, MLP_NUM_LAYER_LOWER_BOUND - 1,
                            MLP_NUM_LAYER_UPPER_BOUND);

                    // start append the optional last layer
                    auto last_layer = std::make_shared<pb_graph_t>();
                    auto activation_bwd_last = last_layer->append_alternation(
                            {graph::op_kind::ReLUBackward,
                                    graph::op_kind::SigmoidBackward});
                    activation_bwd_last->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    // still allow extra grad_input computation
                    activation_bwd_last->allow_external_outputs();

                    auto transpose_subgraph_last
                            = std::make_shared<pb_graph_t>();
                    auto transpose_x_last = transpose_subgraph_last->append_op(
                            graph::op_kind::StaticTranspose);
                    transpose_subgraph_last->create_input_port(
                            0, transpose_x_last, 0);
                    transpose_subgraph_last->create_output_port(
                            0, transpose_x_last, 0);
                    auto optional_transpose_x_last
                            = last_layer->append_optional(
                                    transpose_subgraph_last);
                    auto matmul_last = last_layer->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, optional_transpose_x_last, 0),
                                    in_edge(1, activation_bwd_last, 0)});
                    last_layer->create_input_port(0, activation_bwd_last, 1);
                    last_layer->create_output_port(0, matmul_last, 0);
                    pgraph->append_optional(
                            last_layer, {in_edge(0, repetition, 0)});
                });

/*
mlp residual graph, having an extra edge from LayerNorm to Add.
[IN0](fp32)   [IN1](fp32)
       \      /
        MatMul
          |
         Add
          |
       LayerNorm_______________      [IN2](fp32)
          |                     \     /
          |                      MatMul
          |                        |
          |                      GELU   [IN3](fp32)
          |                        \     /
          |                        MatMul
          | _________________________|
         Add
          |
      LayerNorm
          |
        [OUT0](fp32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_bart_mlp_residual_pattern)
        .set_priority(5.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_layer1
                            = pgraph->append_op(graph::op_kind::MatMul);
                    matmul_layer1->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto add_layer1 = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, matmul_layer1, 0)});
                    auto layernorm_layer1
                            = pgraph->append_op(graph::op_kind::LayerNorm,
                                    {in_edge(0, add_layer1, 0)});
                    auto matmul_layer2
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    {in_edge(0, layernorm_layer1, 0)});
                    auto gelu_layer2 = pgraph->append_op(graph::op_kind::GELU,
                            {in_edge(0, matmul_layer2, 0)});
                    auto matmul_layer3
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    {in_edge(0, gelu_layer2, 0)});
                    auto add_layer3 = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, layernorm_layer1, 0),
                                    in_edge(1, matmul_layer3, 0)});
                    pgraph->append_op(graph::op_kind::LayerNorm,
                            {in_edge(0, add_layer3, 0)});
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, fp32_gpt_mlp)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, false, false, false);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, true, false, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, fp32_llama_mlp)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_llama_mlp(pgraph, false, false);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(int8_mlp_pattern)

/*
repetition unit:
 (int8)[REP_IN0]    [REP_IN1](int8)
           |            |
      Dequantize    Dequantize
              \     /
               MatMul
                 |
                Add (optional)
                 |
             Activation (optional)
                 |
              Quantize
                 |
             [REP_OUT0](int8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, int8_mlp_pattern)
        .set_priority(6.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto mlp_layer = std::make_shared<pb_graph_t>();
                    pm::pb_node_t *dequantize_input, *quantize_output;
                    std::tie(dequantize_input, quantize_output)
                            = single_layer_mlp(mlp_layer, false, true);

                    mlp_layer->create_input_port(0, dequantize_input, 0);
                    mlp_layer->create_output_port(0, quantize_output, 0);

                    // repeat layer for [LOWER_BOUND, UPPER_BOUND) times
                    pgraph->append_repetition(mlp_layer, {0, 0},
                            MLP_NUM_LAYER_LOWER_BOUND,
                            MLP_NUM_LAYER_UPPER_BOUND);
                });

// bart_mlp pattern is causing regression in bert_large model
// will be added back after resolving the performance issue
#if 0
/*
 mlp residual graph, having an extra edge from LayerNorm to Add.
  [IN0](int8)     [IN1](int8)
    |               |
Dequantize      Dequantize
       \      /
        MatMul
          |
         Add
          |
       LayerNorm_______________
          |                    |
          |                 Quantize    [IN2](int8)
          |                    |          |
          |              Dequantize   Dequantize
          |                     \     /
          |                      MatMul
          |                        |
          |                      GELU
          |                        |
          |                     Quantize        [IN3](int8)
          |                        |              |
          |                   Dequantize      Dequantize
          |                            \      /
          |                             MatMul
          |                                |
         Add_______________________________|
          |
      LayerNorm
          |
      Quantize (optional)
          |
        [OUT0](int8/f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_bart_mlp_residual_pattern)
        .set_priority(6.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_input_layer1 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize});
                    auto dequantize_weight_layer1 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize});
                    auto matmul_layer1 = pgraph->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, dequantize_input_layer1, 0),
                                    in_edge(1, dequantize_weight_layer1, 0)});
                    auto add_layer1 = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, matmul_layer1, 0)});
                    auto layernorm_layer1
                            = pgraph->append_op(graph::op_kind::LayerNorm,
                                    {in_edge(0, add_layer1, 0)});
                    // quantize is the second use of layernorm output
                    auto quantize_output_layer1 = pgraph->append_alternation(
                            {graph::op_kind::Quantize,
                                    graph::op_kind::DynamicQuantize},
                            {in_edge(0, layernorm_layer1, 0)});
                    auto dequantize_input_layer2 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize},
                            {in_edge(0, quantize_output_layer1, 0)});
                    auto dequantize_weight_layer2 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize});
                    auto matmul_layer2 = pgraph->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, dequantize_input_layer2, 0),
                                    in_edge(1, dequantize_weight_layer2, 0)});
                    auto gelu_layer2 = pgraph->append_op(graph::op_kind::GELU,
                            {in_edge(0, matmul_layer2, 0)});

                    auto quantize_output_layer2 = pgraph->append_alternation(
                            {graph::op_kind::Quantize,
                                    graph::op_kind::DynamicQuantize},
                            {in_edge(0, gelu_layer2, 0)});

                    auto dequantize_input_layer3 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize},
                            {in_edge(0, quantize_output_layer2, 0)});
                    auto dequantize_weight_layer3 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize});
                    auto matmul_layer3 = pgraph->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, dequantize_input_layer3, 0),
                                    in_edge(1, dequantize_weight_layer3, 0)});
                    auto add_layer3 = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, layernorm_layer1, 0),
                                    in_edge(1, matmul_layer3, 0)});
                    auto layernorm_layer3
                            = pgraph->append_op(graph::op_kind::LayerNorm,
                                    {in_edge(0, add_layer3, 0)});
                    layernorm_layer3->allow_external_outputs();
                    auto last_layer = std::make_shared<pb_graph_t>();
                    auto quantize_output_layer3
                            = last_layer->append_alternation(
                                    {graph::op_kind::Quantize,
                                            graph::op_kind::DynamicQuantize});
                    last_layer->create_input_port(0, quantize_output_layer3, 0);
                    last_layer->create_output_port(
                            0, quantize_output_layer3, 0);
                    pgraph->append_optional(
                            last_layer, {in_edge(0, layernorm_layer3, 0)});
                });

/*
mlp residual graph, having an extra edge from LayerNorm to Add.
[IN0](int8)     [IN1](int8)
    |             |
Dequantize    Dequantize
    |             |
 TypeCast      TypeCast
       \      /
        MatMul
          |
         Add
          |
       LayerNorm_______________
          |                    |
          |                 TypeCast
          |                    |
          |                 Quantize    [IN2](int8)
          |                    |          |
          |               Dequantize   Dequantize
          |                    |          |
          |                TypeCast     TypeCast
          |                       \     /
          |                        MatMul
          |                          |
          |                        GELU
          |                          |
          |                      TypeCast
          |                          |
          |                      Quantize        [IN3](int8)
          |                          |              |
          |                     Dequantize      Dequantize
          |                          |              |
          |                       TypeCast       TypeCast
          |                              \      /
          |                               MatMul
          |                                |
         Add_______________________________|
          |
      LayerNorm
          |
      TypeCast (optional)
          |
      Quantize (optional)
          |
        [OUT0](int8/f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_bf16_bart_mlp_residual_pattern)
        .set_priority(6.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_input_layer1 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize});
                    auto typecast_input_layer1
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_input_layer1, 0)});
                    auto dequantize_weight_layer1 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize});
                    auto typecast_weight_layer1
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_weight_layer1, 0)});
                    auto matmul_layer1 = pgraph->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, typecast_input_layer1, 0),
                                    in_edge(1, typecast_weight_layer1, 0)});
                    auto add_layer1 = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, matmul_layer1, 0)});
                    auto layernorm_layer1
                            = pgraph->append_op(graph::op_kind::LayerNorm,
                                    {in_edge(0, add_layer1, 0)});
                    auto typecast_output_layer1
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, layernorm_layer1, 0)});
                    auto quantize_output_layer1 = pgraph->append_alternation(
                            {graph::op_kind::Quantize,
                                    graph::op_kind::DynamicQuantize},
                            {in_edge(0, typecast_output_layer1, 0)});

                    auto dequantize_input_layer2 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize},
                            {in_edge(0, quantize_output_layer1, 0)});
                    auto typecast_input_layer2
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_input_layer2, 0)});
                    auto dequantize_weight_layer2 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize});
                    auto typecast_weight_layer2
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_weight_layer2, 0)});
                    auto matmul_layer2 = pgraph->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, typecast_input_layer2, 0),
                                    in_edge(1, typecast_weight_layer2, 0)});
                    auto gelu_layer2 = pgraph->append_op(graph::op_kind::GELU,
                            {in_edge(0, matmul_layer2, 0)});
                    auto typecast_output_layer2
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, gelu_layer2, 0)});
                    auto quantize_output_layer2 = pgraph->append_alternation(
                            {graph::op_kind::Quantize,
                                    graph::op_kind::DynamicQuantize},
                            {in_edge(0, typecast_output_layer2, 0)});

                    auto dequantize_input_layer3 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize},
                            {in_edge(0, quantize_output_layer2, 0)});
                    auto typecast_input_layer3
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_input_layer3, 0)});
                    auto dequantize_weight_layer3 = pgraph->append_alternation(
                            {graph::op_kind::Dequantize,
                                    graph::op_kind::DynamicDequantize});
                    auto typecast_weight_layer3
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_weight_layer3, 0)});
                    auto matmul_layer3 = pgraph->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, typecast_input_layer3, 0),
                                    in_edge(1, typecast_weight_layer3, 0)});
                    auto add_layer3 = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, layernorm_layer1, 0),
                                    in_edge(1, matmul_layer3, 0)});
                    auto layernorm_layer3
                            = pgraph->append_op(graph::op_kind::LayerNorm,
                                    {in_edge(0, add_layer3, 0)});
                    layernorm_layer3->allow_external_outputs();
                    auto last_layer = std::make_shared<pb_graph_t>();
                    auto typecast_output_layer3
                            = last_layer->append_op(graph::op_kind::TypeCast);
                    auto quantize_output_layer3
                            = last_layer->append_alternation(
                                    {graph::op_kind::Quantize,
                                            graph::op_kind::DynamicQuantize},
                                    {in_edge(0, typecast_output_layer3, 0)});
                    last_layer->create_input_port(0, typecast_output_layer3, 0);
                    last_layer->create_output_port(
                            0, quantize_output_layer3, 0);
                    pgraph->append_optional(
                            last_layer, {in_edge(0, layernorm_layer3, 0)});
                });
#endif

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, int8_gpt_mlp)
        .set_priority(6.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, false, false, true);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, true, false, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, int8_gpt_mlp_fp32_out)
        .set_priority(6.4f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, false, false, true, false);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, true, false, true, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, int8_bf16_gpt_mlp)
        .set_priority(6.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, false, true, true);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, true, true, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_bf16_gpt_mlp_fp32_out)
        .set_priority(6.4f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, false, true, true, false);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, true, true, true, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, int8_llama_mlp)
        .set_priority(6.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_llama_mlp(pgraph, false, true);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_llama_mlp(pgraph, false, true, false, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, int8_bf16_llama_mlp)
        .set_priority(6.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_llama_mlp(pgraph, true, true);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_llama_mlp(pgraph, true, true, true);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_llama_mlp(pgraph, true, true, false, true);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_llama_mlp(pgraph, true, true, true, true);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(bf16_mlp_pattern)

/*
repetition unit:
 (bf16)[REP_IN0]   [REP_IN1](bf16)
              \     /
               MatMul
                 |
                Add (optional)
                 |
             Activation
                 |
             [REP_OUT0](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mlp_forward_pattern)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto mlp_layer = std::make_shared<pb_graph_t>();
                    pm::pb_node_t *matmul, *optional_activation;
                    std::tie(matmul, optional_activation)
                            = single_layer_mlp(mlp_layer, true, false);
                    mlp_layer->create_input_port(0, matmul, 0);
                    mlp_layer->create_output_port(0, optional_activation, 0);

                    // repeat layer for [LOWER_BOUND, UPPER_BOUND) times
                    pgraph->append_repetition(mlp_layer, {0, 0},
                            MLP_NUM_LAYER_LOWER_BOUND,
                            MLP_NUM_LAYER_UPPER_BOUND);
                });

/*
[repetition unit]:
          (bf16)[x_next]     [gradient_x_next](bf16)
                      \       /     [weight](bf16)
                       \     /           |
                       Backward    StaticTranspose
                     /    |   \     /     (optional)
    [grad_w_subgraph]  Reduce  Matmul
             |            |      |
      [gradient_w](bf16)  |  [gradient_x](bf16)
                          |
                  [gradient_bias](bf16)

[optional unit]:
          (bf16)[x_next]   [repetition_unit_out](bf16)
                      \       /
                       \     /
                      Backward
                     /    |
      [grad_w_subgraph] Reduce
             |            |
      [gradient_w](bf16)  |
                          |
                  [gradient_bias](bf16)

pattern:
        [repetition unit]*[1-10]
                |
        [optional unit] (optional)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mlp_backward_pattern)
        .set_priority(5.1f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bwd_mlp_layer = std::make_shared<pb_graph_t>();
                    auto activation_bwd = bwd_mlp_layer->append_alternation(
                            {graph::op_kind::ReLUBackward,
                                    graph::op_kind::SigmoidBackward});
                    activation_bwd->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);

                    weight_grad_alternation_unit(bwd_mlp_layer, activation_bwd);

                    auto transpose_subgraph2 = std::make_shared<pb_graph_t>();
                    auto transpose_w = transpose_subgraph2->append_op(
                            graph::op_kind::StaticTranspose);
                    transpose_subgraph2->create_input_port(0, transpose_w, 0);
                    transpose_subgraph2->create_output_port(0, transpose_w, 0);
                    auto optional_transpose_w = bwd_mlp_layer->append_optional(
                            transpose_subgraph2);
                    auto matmul_layer = bwd_mlp_layer->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, activation_bwd, 0),
                                    in_edge(1, optional_transpose_w, 0)});

                    auto reduce_bias = bwd_mlp_layer->append_op(
                            graph::op_kind::ReduceSum,
                            {in_edge(0, activation_bwd, 0)});
                    reduce_bias->append_decision_function(check_reduce_attrs);

                    bwd_mlp_layer->create_input_port(0, activation_bwd, 1);
                    bwd_mlp_layer->create_output_port(0, matmul_layer, 0);

                    // repeat layer for [LOWER_BOUND - 1, UPPER_BOUND) times
                    auto repetition = pgraph->append_repetition(bwd_mlp_layer,
                            {0, 0}, MLP_NUM_LAYER_LOWER_BOUND - 1,
                            MLP_NUM_LAYER_UPPER_BOUND);

                    // start append the optional last layer
                    auto last_layer = std::make_shared<pb_graph_t>();
                    auto activation_bwd_last = last_layer->append_alternation(
                            {graph::op_kind::ReLUBackward,
                                    graph::op_kind::SigmoidBackward});
                    activation_bwd_last->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    // still allow extra grad_input computation
                    activation_bwd_last->allow_external_outputs();

                    auto weight_grad = weight_grad_alternation_unit(
                            last_layer, activation_bwd_last);
                    auto reduce_bias_last
                            = last_layer->append_op(graph::op_kind::ReduceSum,
                                    {in_edge(0, activation_bwd_last, 0)});
                    reduce_bias_last->append_decision_function(
                            check_reduce_attrs);
                    last_layer->create_input_port(0, activation_bwd_last, 1);
                    last_layer->create_output_port(0, weight_grad, 0);
                    pgraph->append_optional(
                            last_layer, {in_edge(0, repetition, 0)});
                });

/*
[repetition unit]:
          (bf16)[x_next]     [gradient_x_next](bf16)
       [x](bf16)      \       /     [weight](bf16)
            |          \     /           |
     StaticTranspose  Backward    StaticTranspose
(optional)     \     /        \     /     (optional)
                Matmul         Matmul
                   |             |
      [gradient_w](bf16)     [gradient_x](bf16)

[optional unit]:
          (bf16)[x_next]   [repetition_unit_out](bf16)
       [x](bf16)      \       /
            |          \     /
     StaticTranspose  Backward
(optional)     \     /
                Matmul
                   |
      [gradient_w](bf16)

pattern:
        [repetition unit]*[1-10]
                |
         [optional unit] (optional)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mlp_backward_pattern_v2)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bwd_mlp_layer = std::make_shared<pb_graph_t>();
                    auto activation_bwd = bwd_mlp_layer->append_alternation(
                            {graph::op_kind::ReLUBackward,
                                    graph::op_kind::SigmoidBackward});
                    activation_bwd->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    activation_bwd->allow_external_outputs();

                    auto transpose_subgraph1 = std::make_shared<pb_graph_t>();
                    auto transpose_x = transpose_subgraph1->append_op(
                            graph::op_kind::StaticTranspose);
                    transpose_subgraph1->create_input_port(0, transpose_x, 0);
                    transpose_subgraph1->create_output_port(0, transpose_x, 0);
                    auto optional_transpose_x = bwd_mlp_layer->append_optional(
                            transpose_subgraph1);

                    auto transpose_subgraph2 = std::make_shared<pb_graph_t>();
                    auto transpose_w = transpose_subgraph2->append_op(
                            graph::op_kind::StaticTranspose);
                    transpose_subgraph2->create_input_port(0, transpose_w, 0);
                    transpose_subgraph2->create_output_port(0, transpose_w, 0);
                    auto optional_transpose_w = bwd_mlp_layer->append_optional(
                            transpose_subgraph2);

                    bwd_mlp_layer->append_op(graph::op_kind::MatMul,
                            {in_edge(0, optional_transpose_x, 0),
                                    in_edge(1, activation_bwd, 0)});
                    auto matmul_layer = bwd_mlp_layer->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, activation_bwd, 0),
                                    in_edge(1, optional_transpose_w, 0)});

                    bwd_mlp_layer->create_input_port(0, activation_bwd, 1);
                    bwd_mlp_layer->create_output_port(0, matmul_layer, 0);

                    // repeat layer for [LOWER_BOUND - 1, UPPER_BOUND) times
                    auto repetition = pgraph->append_repetition(bwd_mlp_layer,
                            {0, 0}, MLP_NUM_LAYER_LOWER_BOUND - 1,
                            MLP_NUM_LAYER_UPPER_BOUND);

                    // start append the optional last layer
                    auto last_layer = std::make_shared<pb_graph_t>();
                    auto activation_bwd_last = last_layer->append_alternation(
                            {graph::op_kind::ReLUBackward,
                                    graph::op_kind::SigmoidBackward});
                    activation_bwd_last->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    // still allow extra grad_input computation
                    activation_bwd_last->allow_external_outputs();

                    auto transpose_subgraph_last
                            = std::make_shared<pb_graph_t>();
                    auto transpose_x_last = transpose_subgraph_last->append_op(
                            graph::op_kind::StaticTranspose);
                    transpose_subgraph_last->create_input_port(
                            0, transpose_x_last, 0);
                    transpose_subgraph_last->create_output_port(
                            0, transpose_x_last, 0);
                    auto optional_transpose_x_last
                            = last_layer->append_optional(
                                    transpose_subgraph_last);
                    auto matmul_last = last_layer->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, optional_transpose_x_last, 0),
                                    in_edge(1, activation_bwd_last, 0)});
                    last_layer->create_input_port(0, activation_bwd_last, 1);
                    last_layer->create_output_port(0, matmul_last, 0);
                    pgraph->append_optional(
                            last_layer, {in_edge(0, repetition, 0)});
                });
/*
mlp residual graph, having an extra edge from LayerNorm to Add.
[IN0](bf16)   [IN1](bf16)
       \      /
        MatMul
          |
         Add
          |
       LayerNorm_______________      [IN2](bf16)
          |                     \     /
          |                      MatMul
          |                        |
          |                      GELU   [IN3](bf16)
          |                        \     /
          |                        MatMul
          | _________________________|
         Add
          |
      LayerNorm
          |
        [OUT0](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_bart_mlp_residual_pattern)
        .set_priority(5.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_layer1
                            = pgraph->append_op(graph::op_kind::MatMul);
                    matmul_layer1->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto add_layer1 = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, matmul_layer1, 0)});
                    auto layernorm_layer1
                            = pgraph->append_op(graph::op_kind::LayerNorm,
                                    {in_edge(0, add_layer1, 0)});
                    auto matmul_layer2
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    {in_edge(0, layernorm_layer1, 0)});
                    auto gelu_layer2 = pgraph->append_op(graph::op_kind::GELU,
                            {in_edge(0, matmul_layer2, 0)});
                    auto matmul_layer3
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    {in_edge(0, gelu_layer2, 0)});
                    auto add_layer3 = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, layernorm_layer1, 0),
                                    in_edge(1, matmul_layer3, 0)});
                    pgraph->append_op(graph::op_kind::LayerNorm,
                            {in_edge(0, add_layer3, 0)});
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, bf16_gpt_mlp)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, false, true, false);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_mlp(pgraph, true, true, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, bf16_llama_mlp)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_llama_mlp(pgraph, true, false);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_llama_mlp(pgraph, true, false, true);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
