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

#include "backend/graph_compiler/patterns/fusions.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {
namespace pass {

using pb_graph_t = impl::utils::pm::pb_graph_t;
using FCreatePattern = impl::pass::FCreatePattern;

#define MLP_NUM_LAYER_LOWER_BOUND 2
#define MLP_NUM_LAYER_UPPER_BOUND 11

std::pair<pm::pb_node_t *, pm::pb_node_t *> single_layer_mlp(
        const std::shared_ptr<pb_graph_t> &pgraph, bool is_bf16 = false,
        bool is_int8 = false) {
    pm::pb_node_t *layer_input, *layer_output;
    in_edges_t matmul_in_edges;
    if (is_int8) {
        auto dequantize_input = pgraph->append_op(
                impl::op_kind::Dequantize, "dequantize_input");
        auto dequantize_weight = pgraph->append_op(
                impl::op_kind::Dequantize, "dequantize_weight");
        if (is_bf16) {
            auto typecast_input = pgraph->append_op(impl::op_kind::TypeCast,
                    {in_edge(0, dequantize_input, 0)}, "typecast_input");
            auto typecast_weight = pgraph->append_op(impl::op_kind::TypeCast,
                    {in_edge(0, dequantize_weight, 0)}, "typecast_weight");
            matmul_in_edges = in_edges_t {in_edge(0, typecast_input, 0),
                    in_edge(1, typecast_weight, 0)};
        } else {
            matmul_in_edges = in_edges_t {in_edge(0, dequantize_input, 0),
                    in_edge(1, dequantize_weight, 0)};
        }
        layer_input = dequantize_input;
    }
    auto matmul = pgraph->append_op(
            impl::op_kind::MatMul, matmul_in_edges, "matmul");
    matmul->append_decision_function(is_bf16
                    ? check_input_dtype<impl::data_type::bf16>
                    : check_input_dtype<impl::data_type::f32>);
    matmul->allow_external_outputs();
    if (!is_int8) { layer_input = matmul; }

    /* optional add/biasAdd after matmul */
    auto add_subgraph = std::make_shared<pb_graph_t>("add_subgraph");
    auto add = add_subgraph->append_alternation(
            {impl::op_kind::Add, impl::op_kind::BiasAdd}, "add");
    add->allow_external_outputs();
    add_subgraph->create_input_port(0, add, 0);
    add_subgraph->create_output_port(0, add, 0);
    auto optional_add = pgraph->append_optional(
            add_subgraph, {in_edge(0, matmul, 0)}, "optional_add");

    /* optional activation */
    auto activation_subgraph
            = std::make_shared<pb_graph_t>("activation_subgraph");
    auto activation = activation_subgraph->append_alternation(
            {impl::op_kind::ReLU, impl::op_kind::Sigmoid}, "activation");
    activation->allow_external_outputs();
    activation_subgraph->create_input_port(0, activation, 0);
    activation_subgraph->create_output_port(0, activation, 0);
    auto optional_activation = pgraph->append_optional(activation_subgraph,
            {in_edge(0, optional_add, 0)}, "optional_activation");
    layer_output = optional_activation;

    if (is_int8) {
        if (is_bf16) {
            auto typecast_output = pgraph->append_op(impl::op_kind::TypeCast,
                    {in_edge(0, layer_output, 0)}, "typecast_output");
            layer_output = typecast_output;
        }
        auto quantize_output = pgraph->append_op(impl::op_kind::Quantize,
                {in_edge(0, layer_output, 0)}, "quantize_output");
        layer_output = quantize_output;
    }
    return std::make_pair(layer_input, layer_output);
}

pm::pb_node_t *weight_grad_alternation_unit(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *activation) {
    /* Create 2 subgraph for alternation */
    auto weight_grad_option1
            = std::make_shared<pb_graph_t>("weight_grad_option1");
    auto transpose_subgraph1
            = std::make_shared<pb_graph_t>("transpose_subgraph1");
    auto transpose_x1 = transpose_subgraph1->append_op(
            impl::op_kind::StaticTranspose, "transpose_x");
    transpose_subgraph1->create_input_port(0, transpose_x1, 0);
    transpose_subgraph1->create_output_port(0, transpose_x1, 0);
    auto optional_transpose_x = weight_grad_option1->append_optional(
            transpose_subgraph1, "optional_transpose_x");
    auto matmul_grad_w1 = weight_grad_option1->append_op(impl::op_kind::MatMul,
            {in_edge(0, optional_transpose_x, 0)}, "matmul_weight");
    weight_grad_option1->create_input_port(0, matmul_grad_w1, 1);
    weight_grad_option1->create_output_port(0, matmul_grad_w1, 0);

    auto weight_grad_option2
            = std::make_shared<pb_graph_t>("weight_grad_option2");
    auto transpose_x2 = weight_grad_option2->append_op(
            impl::op_kind::StaticTranspose, "transpose_x");
    auto matmul_grad_w2 = weight_grad_option2->append_op(impl::op_kind::MatMul,
            {in_edge(0, transpose_x2, 0)}, "matmul_weight");
    weight_grad_option2->create_input_port(0, transpose_x2, 0);
    weight_grad_option2->create_output_port(0, matmul_grad_w2, 0);

    auto weight_grad = pgraph->append_alternation(
            {weight_grad_option1, weight_grad_option2},
            {in_edge(0, activation, 0)}, "weight_grad");
    return weight_grad;
};

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(fp32_mlp_pattern)

/*
      (f32)[IN0]   [IN1](f32)
              \     /
               MatMul
                 |
                Add (optional)
                 |
             Activation (optional)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, fp32_mlp_single_layer)
        .set_priority(3.5f)
        .set_kind(impl::partition_kind::mlp)
        .set_type(impl::pass::pattern_type_t::dynamic_shape_only)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    single_layer_mlp(pgraph, false, false);
                });

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
        .set_kind(impl::partition_kind::mlp)
        .set_type(impl::pass::pattern_type_t::static_and_dynamic_shape)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto mlp_layer = std::make_shared<pb_graph_t>("mlp_layer");
                    pm::pb_node_t *matmul, *optional_activation;
                    std::tie(matmul, optional_activation)
                            = single_layer_mlp(mlp_layer, false, false);
                    mlp_layer->create_input_port(0, matmul, 0);
                    mlp_layer->create_output_port(0, optional_activation, 0);
                    // repeat layer for [LOWER_BOUND, UPPER_BOUND) times
                    pgraph->append_repetition(mlp_layer, {0, 0},
                            MLP_NUM_LAYER_LOWER_BOUND,
                            MLP_NUM_LAYER_UPPER_BOUND, "rep_unit");
                });

/*
[repetition unit]:
          (f32)[x_next]     [gradient_x_next](f32)
                      \       /     [weight](f32)
                       \     /           |
                       Backprop    StaticTranspose
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
                      Backprop
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
        .set_kind(impl::partition_kind::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bwd_mlp_layer
                            = std::make_shared<pb_graph_t>("bwd_mlp_layer");
                    auto activation_bwd = bwd_mlp_layer->append_alternation(
                            {impl::op_kind::ReLUBackprop,
                                    impl::op_kind::SigmoidBackprop},
                            "activation_bwd");
                    activation_bwd->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);

                    weight_grad_alternation_unit(bwd_mlp_layer, activation_bwd);

                    auto transpose_subgraph2 = std::make_shared<pb_graph_t>(
                            "transpose_subgraph2");
                    auto transpose_w = transpose_subgraph2->append_op(
                            impl::op_kind::StaticTranspose, "transpose_w");
                    transpose_subgraph2->create_input_port(0, transpose_w, 0);
                    transpose_subgraph2->create_output_port(0, transpose_w, 0);
                    auto optional_transpose_w = bwd_mlp_layer->append_optional(
                            transpose_subgraph2, "optional_transpose_w");
                    auto matmul_layer = bwd_mlp_layer->append_op(
                            impl::op_kind::MatMul,
                            {in_edge(0, activation_bwd, 0),
                                    in_edge(1, optional_transpose_w, 0)},
                            "matmul_layer");

                    auto reduce_bias = bwd_mlp_layer->append_op(
                            impl::op_kind::ReduceSum,
                            {in_edge(0, activation_bwd, 0)}, "reduce_bias");
                    reduce_bias->append_decision_function(check_reduce_attrs);

                    bwd_mlp_layer->create_input_port(0, activation_bwd, 1);
                    bwd_mlp_layer->create_output_port(0, matmul_layer, 0);

                    // repeat layer for [LOWER_BOUND - 1, UPPER_BOUND) times
                    auto repetition = pgraph->append_repetition(bwd_mlp_layer,
                            {0, 0}, MLP_NUM_LAYER_LOWER_BOUND - 1,
                            MLP_NUM_LAYER_UPPER_BOUND, "rep_unit");

                    // start append the optional last layer
                    auto last_layer
                            = std::make_shared<pb_graph_t>("last_layer");
                    auto activation_bwd_last = last_layer->append_alternation(
                            {impl::op_kind::ReLUBackprop,
                                    impl::op_kind::SigmoidBackprop},
                            "activation_bwd");
                    activation_bwd_last->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    // still allow extra grad_input computation
                    activation_bwd_last->allow_external_outputs();

                    auto weight_grad = weight_grad_alternation_unit(
                            last_layer, activation_bwd_last);
                    auto reduce_bias_last
                            = last_layer->append_op(impl::op_kind::ReduceSum,
                                    {in_edge(0, activation_bwd_last, 0)},
                                    "reduce_bias");
                    reduce_bias_last->append_decision_function(
                            check_reduce_attrs);
                    last_layer->create_input_port(0, activation_bwd_last, 1);
                    last_layer->create_output_port(0, weight_grad, 0);
                    pgraph->append_optional(last_layer,
                            {in_edge(0, repetition, 0)}, "optional_last_layer");
                });

/*
[repetition unit]:
           (f32)[x_next]     [gradient_x_next](bf16)
        [x](f32)      \       /     [weight](f32)
            |          \     /           |
     StaticTranspose   Backprop    StaticTranspose
(optional)     \     /        \     /     (optional)
                Matmul         Matmul
                   |             |
       [gradient_w](f32)     [gradient_x](f32)

[optional unit]:
           (f32)[x_next]   [repetition_unit_out](f32)
        [x](f32)      \       /
            |          \     /
     StaticTranspose  Backprop
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
        .set_kind(impl::partition_kind::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bwd_mlp_layer
                            = std::make_shared<pb_graph_t>("bwd_mlp_layer");
                    auto activation_bwd = bwd_mlp_layer->append_alternation(
                            {impl::op_kind::ReLUBackprop,
                                    impl::op_kind::SigmoidBackprop},
                            "activation_bwd");
                    activation_bwd->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    activation_bwd->allow_external_outputs();

                    auto transpose_subgraph1 = std::make_shared<pb_graph_t>(
                            "transpose_subgraph1");
                    auto transpose_x = transpose_subgraph1->append_op(
                            impl::op_kind::StaticTranspose, "transpose_x");
                    transpose_subgraph1->create_input_port(0, transpose_x, 0);
                    transpose_subgraph1->create_output_port(0, transpose_x, 0);
                    auto optional_transpose_x = bwd_mlp_layer->append_optional(
                            transpose_subgraph1, "optional_transpose_x");

                    auto transpose_subgraph2 = std::make_shared<pb_graph_t>(
                            "transpose_subgraph2");
                    auto transpose_w = transpose_subgraph2->append_op(
                            impl::op_kind::StaticTranspose, "transpose_w");
                    transpose_subgraph2->create_input_port(0, transpose_w, 0);
                    transpose_subgraph2->create_output_port(0, transpose_w, 0);
                    auto optional_transpose_w = bwd_mlp_layer->append_optional(
                            transpose_subgraph2, "optional_transpose_w");

                    bwd_mlp_layer->append_op(impl::op_kind::MatMul,
                            {in_edge(0, optional_transpose_x, 0),
                                    in_edge(1, activation_bwd, 0)},
                            "matmul_weight");
                    auto matmul_layer = bwd_mlp_layer->append_op(
                            impl::op_kind::MatMul,
                            {in_edge(0, activation_bwd, 0),
                                    in_edge(1, optional_transpose_w, 0)},
                            "matmul_layer");

                    bwd_mlp_layer->create_input_port(0, activation_bwd, 1);
                    bwd_mlp_layer->create_output_port(0, matmul_layer, 0);

                    // repeat layer for [LOWER_BOUND - 1, UPPER_BOUND) times
                    auto repetition = pgraph->append_repetition(bwd_mlp_layer,
                            {0, 0}, MLP_NUM_LAYER_LOWER_BOUND - 1,
                            MLP_NUM_LAYER_UPPER_BOUND, "rep_unit");

                    // start append the optional last layer
                    auto last_layer
                            = std::make_shared<pb_graph_t>("last_layer");
                    auto activation_bwd_last = last_layer->append_alternation(
                            {impl::op_kind::ReLUBackprop,
                                    impl::op_kind::SigmoidBackprop},
                            "activation_bwd");
                    activation_bwd_last->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    // still allow extra grad_input computation
                    activation_bwd_last->allow_external_outputs();

                    auto transpose_subgraph_last = std::make_shared<pb_graph_t>(
                            "transpose_subgraph");
                    auto transpose_x_last = transpose_subgraph_last->append_op(
                            impl::op_kind::StaticTranspose, "transpose_x");
                    transpose_subgraph_last->create_input_port(
                            0, transpose_x_last, 0);
                    transpose_subgraph_last->create_output_port(
                            0, transpose_x_last, 0);
                    auto optional_transpose_x_last
                            = last_layer->append_optional(
                                    transpose_subgraph_last,
                                    "optional_transpose_x");
                    auto matmul_last
                            = last_layer->append_op(impl::op_kind::MatMul,
                                    {in_edge(0, optional_transpose_x_last, 0),
                                            in_edge(1, activation_bwd_last, 0)},
                                    "matmul_weight");
                    last_layer->create_input_port(0, activation_bwd_last, 1);
                    last_layer->create_output_port(0, matmul_last, 0);
                    pgraph->append_optional(last_layer,
                            {in_edge(0, repetition, 0)}, "optional_last_layer");
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(int8_mlp_pattern)

/*
   (int8)[IN0]        [IN1](int8)
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
              [OUT0](int8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, int8_mlp_single_layer)
        .set_priority(4.0f)
        .set_kind(impl::partition_kind::quantized_mlp)
        .set_type(impl::pass::pattern_type_t::dynamic_shape_only)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    single_layer_mlp(pgraph, false, true);
                });

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
        .set_kind(impl::partition_kind::quantized_mlp)
        .set_type(impl::pass::pattern_type_t::static_and_dynamic_shape)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto mlp_layer = std::make_shared<pb_graph_t>("mlp_layer");
                    pm::pb_node_t *dequantize_input, *quantize_output;
                    std::tie(dequantize_input, quantize_output)
                            = single_layer_mlp(mlp_layer, false, true);

                    mlp_layer->create_input_port(0, dequantize_input, 0);
                    mlp_layer->create_output_port(0, quantize_output, 0);

                    // repeat layer for [LOWER_BOUND, UPPER_BOUND) times
                    pgraph->append_repetition(mlp_layer, {0, 0},
                            MLP_NUM_LAYER_LOWER_BOUND,
                            MLP_NUM_LAYER_UPPER_BOUND, "rep_unit");
                });

/*
   (int8)[IN0]        [IN1](int8)
           |            |
      Dequantize    Dequantize
           |            |
        TypeCast    TypeCast
              \      /
               MatMul
                 |
                Add (optional)
                 |
             Activation (optional)
                 |
              TypeCast
                 |
              Quantize
                 |
              [OUT0](int8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_bf16_mlp_single_layer)
        .set_priority(4.0f)
        .set_kind(impl::partition_kind::quantized_mlp)
        .set_type(impl::pass::pattern_type_t::dynamic_shape_only)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    single_layer_mlp(pgraph, true, true);
                });

COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(bf16_mlp_pattern)

/*
     (bf16)[IN0]   [IN1](bf16)
              \     /
               MatMul
                 |
                Add (optional)
                 |
             Activation (optional)
                 |
               [OUT0](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, bf16_mlp_single_layer)
        .set_priority(3.5f)
        .set_kind(impl::partition_kind::mlp)
        .set_type(impl::pass::pattern_type_t::dynamic_shape_only)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    single_layer_mlp(pgraph, true, false);
                });

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
        .set_kind(impl::partition_kind::mlp)
        .set_type(impl::pass::pattern_type_t::static_and_dynamic_shape)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto mlp_layer = std::make_shared<pb_graph_t>("mlp_layer");
                    pm::pb_node_t *matmul, *optional_activation;
                    std::tie(matmul, optional_activation)
                            = single_layer_mlp(mlp_layer, true, false);
                    mlp_layer->create_input_port(0, matmul, 0);
                    mlp_layer->create_output_port(0, optional_activation, 0);
                    // repeat layer for [LOWER_BOUND, UPPER_BOUND) times
                    pgraph->append_repetition(mlp_layer, {0, 0},
                            MLP_NUM_LAYER_LOWER_BOUND,
                            MLP_NUM_LAYER_UPPER_BOUND, "rep_unit");
                });

/*
[repetition unit]:
          (bf16)[x_next]     [gradient_x_next](bf16)
                      \       /     [weight](bf16)
                       \     /           |
                       Backprop    StaticTranspose
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
                      Backprop
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
        .set_kind(impl::partition_kind::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bwd_mlp_layer
                            = std::make_shared<pb_graph_t>("bwd_mlp_layer");
                    auto activation_bwd = bwd_mlp_layer->append_alternation(
                            {impl::op_kind::ReLUBackprop,
                                    impl::op_kind::SigmoidBackprop},
                            "activation_bwd");
                    activation_bwd->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);

                    weight_grad_alternation_unit(bwd_mlp_layer, activation_bwd);

                    auto transpose_subgraph2 = std::make_shared<pb_graph_t>(
                            "transpose_subgraph2");
                    auto transpose_w = transpose_subgraph2->append_op(
                            impl::op_kind::StaticTranspose, "transpose_w");
                    transpose_subgraph2->create_input_port(0, transpose_w, 0);
                    transpose_subgraph2->create_output_port(0, transpose_w, 0);
                    auto optional_transpose_w = bwd_mlp_layer->append_optional(
                            transpose_subgraph2, "optional_transpose_w");
                    auto matmul_layer = bwd_mlp_layer->append_op(
                            impl::op_kind::MatMul,
                            {in_edge(0, activation_bwd, 0),
                                    in_edge(1, optional_transpose_w, 0)},
                            "matmul_layer");

                    auto reduce_bias = bwd_mlp_layer->append_op(
                            impl::op_kind::ReduceSum,
                            {in_edge(0, activation_bwd, 0)}, "reduce_bias");
                    reduce_bias->append_decision_function(check_reduce_attrs);

                    bwd_mlp_layer->create_input_port(0, activation_bwd, 1);
                    bwd_mlp_layer->create_output_port(0, matmul_layer, 0);

                    // repeat layer for [LOWER_BOUND - 1, UPPER_BOUND) times
                    auto repetition = pgraph->append_repetition(bwd_mlp_layer,
                            {0, 0}, MLP_NUM_LAYER_LOWER_BOUND - 1,
                            MLP_NUM_LAYER_UPPER_BOUND, "rep_unit");

                    // start append the optional last layer
                    auto last_layer
                            = std::make_shared<pb_graph_t>("last_layer");
                    auto activation_bwd_last = last_layer->append_alternation(
                            {impl::op_kind::ReLUBackprop,
                                    impl::op_kind::SigmoidBackprop},
                            "activation_bwd");
                    activation_bwd_last->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    // still allow extra grad_input computation
                    activation_bwd_last->allow_external_outputs();

                    auto weight_grad = weight_grad_alternation_unit(
                            last_layer, activation_bwd_last);
                    auto reduce_bias_last
                            = last_layer->append_op(impl::op_kind::ReduceSum,
                                    {in_edge(0, activation_bwd_last, 0)},
                                    "reduce_bias");
                    reduce_bias_last->append_decision_function(
                            check_reduce_attrs);
                    last_layer->create_input_port(0, activation_bwd_last, 1);
                    last_layer->create_output_port(0, weight_grad, 0);
                    pgraph->append_optional(last_layer,
                            {in_edge(0, repetition, 0)}, "optional_last_layer");
                });

/*
[repetition unit]:
          (bf16)[x_next]     [gradient_x_next](bf16)
       [x](bf16)      \       /     [weight](bf16)
            |          \     /           |
     StaticTranspose  Backprop    StaticTranspose
(optional)     \     /        \     /     (optional)
                Matmul         Matmul
                   |             |
      [gradient_w](bf16)     [gradient_x](bf16)

[optional unit]:
          (bf16)[x_next]   [repetition_unit_out](bf16)
       [x](bf16)      \       /
            |          \     /
     StaticTranspose  Backprop
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
        .set_kind(impl::partition_kind::mlp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bwd_mlp_layer
                            = std::make_shared<pb_graph_t>("bwd_mlp_layer");
                    auto activation_bwd = bwd_mlp_layer->append_alternation(
                            {impl::op_kind::ReLUBackprop,
                                    impl::op_kind::SigmoidBackprop},
                            "activation_bwd");
                    activation_bwd->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    activation_bwd->allow_external_outputs();

                    auto transpose_subgraph1 = std::make_shared<pb_graph_t>(
                            "transpose_subgraph1");
                    auto transpose_x = transpose_subgraph1->append_op(
                            impl::op_kind::StaticTranspose, "transpose_x");
                    transpose_subgraph1->create_input_port(0, transpose_x, 0);
                    transpose_subgraph1->create_output_port(0, transpose_x, 0);
                    auto optional_transpose_x = bwd_mlp_layer->append_optional(
                            transpose_subgraph1, "optional_transpose_x");

                    auto transpose_subgraph2 = std::make_shared<pb_graph_t>(
                            "transpose_subgraph2");
                    auto transpose_w = transpose_subgraph2->append_op(
                            impl::op_kind::StaticTranspose, "transpose_w");
                    transpose_subgraph2->create_input_port(0, transpose_w, 0);
                    transpose_subgraph2->create_output_port(0, transpose_w, 0);
                    auto optional_transpose_w = bwd_mlp_layer->append_optional(
                            transpose_subgraph2, "optional_transpose_w");

                    bwd_mlp_layer->append_op(impl::op_kind::MatMul,
                            {in_edge(0, optional_transpose_x, 0),
                                    in_edge(1, activation_bwd, 0)},
                            "matmul_weight");
                    auto matmul_layer = bwd_mlp_layer->append_op(
                            impl::op_kind::MatMul,
                            {in_edge(0, activation_bwd, 0),
                                    in_edge(1, optional_transpose_w, 0)},
                            "matmul_layer");

                    bwd_mlp_layer->create_input_port(0, activation_bwd, 1);
                    bwd_mlp_layer->create_output_port(0, matmul_layer, 0);

                    // repeat layer for [LOWER_BOUND - 1, UPPER_BOUND) times
                    auto repetition = pgraph->append_repetition(bwd_mlp_layer,
                            {0, 0}, MLP_NUM_LAYER_LOWER_BOUND - 1,
                            MLP_NUM_LAYER_UPPER_BOUND, "rep_unit");

                    // start append the optional last layer
                    auto last_layer
                            = std::make_shared<pb_graph_t>("last_layer");
                    auto activation_bwd_last = last_layer->append_alternation(
                            {impl::op_kind::ReLUBackprop,
                                    impl::op_kind::SigmoidBackprop},
                            "activation_bwd");
                    activation_bwd_last->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    // still allow extra grad_input computation
                    activation_bwd_last->allow_external_outputs();

                    auto transpose_subgraph_last = std::make_shared<pb_graph_t>(
                            "transpose_subgraph");
                    auto transpose_x_last = transpose_subgraph_last->append_op(
                            impl::op_kind::StaticTranspose, "transpose_x");
                    transpose_subgraph_last->create_input_port(
                            0, transpose_x_last, 0);
                    transpose_subgraph_last->create_output_port(
                            0, transpose_x_last, 0);
                    auto optional_transpose_x_last
                            = last_layer->append_optional(
                                    transpose_subgraph_last,
                                    "optional_transpose_x");
                    auto matmul_last
                            = last_layer->append_op(impl::op_kind::MatMul,
                                    {in_edge(0, optional_transpose_x_last, 0),
                                            in_edge(1, activation_bwd_last, 0)},
                                    "matmul_weight");
                    last_layer->create_input_port(0, activation_bwd_last, 1);
                    last_layer->create_output_port(0, matmul_last, 0);
                    pgraph->append_optional(last_layer,
                            {in_edge(0, repetition, 0)}, "optional_last_layer");
                });

COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
