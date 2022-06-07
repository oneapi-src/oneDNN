/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "backend/graph_compiler/patterns/fusions.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {
namespace pass {

using pb_graph_t = impl::utils::pm::pb_graph_t;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

#define DEFINE_MLP_LAYER_START(DTYPE) \
    auto matmul1 = pgraph->append_op(impl::op_kind::MatMul, "matmul"); \
    matmul1->append_decision_function( \
            check_input_dtype<impl::data_type::DTYPE>); \
    matmul1->allow_external_output(0); \
    auto activation1 = pgraph->append_alternation( \
            {impl::op_kind::ReLU, impl::op_kind::Sigmoid, \
                    impl::op_kind::GELU}, \
            {in_edge(0, matmul1, 0)}, "activation"); \
    activation1->allow_external_output(0);

#define DEFINE_MLP_LAYER(LAYER, PREV_LAYER) \
    auto matmul##LAYER = pgraph->append_op(impl::op_kind::MatMul, \
            {in_edge(0, activation##PREV_LAYER, 0)}, "matmul"); \
    matmul##LAYER->allow_external_output(0); \
    auto activation##LAYER = pgraph->append_alternation( \
            {impl::op_kind::ReLU, impl::op_kind::Sigmoid, \
                    impl::op_kind::GELU}, \
            {in_edge(0, matmul##LAYER, 0)}, "activation"); \
    activation##LAYER->allow_external_output(0);

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(fp32_mlp_pattern)

/*
repetition unit:
  (f32)[REP_IN0]   [REP_IN1](f32)
              \     /
               MatMul
                 |
             Activation
                 |
             [REP_OUT0](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, fp32_mlp_pattern)
        .set_priority(5.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto mlp_layer = std::make_shared<pb_graph_t>("mlp_layer");
                    auto matmul = mlp_layer->append_op(
                            impl::op_kind::MatMul, "matmul");
                    matmul->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);

                    auto activation = mlp_layer->append_alternation(
                            {impl::op_kind::ReLU, impl::op_kind::Sigmoid,
                                    impl::op_kind::GELU},
                            {in_edge(0, matmul, 0)}, "activation");

                    mlp_layer->create_input_port(0, matmul, 0);
                    mlp_layer->create_output_port(0, activation, 0);

                    // repeat layer for [2, 10) times
                    pgraph->append_repetition(
                            mlp_layer, {0, 0}, 2, 10, "rep_unit");
                });

/*
repetition unit of 3-6 layers
  (f32)[REP_IN0]   [REP_IN1](f32)
              \     /
               MatMul -- [EXTERNAL_OUT]
                 |
             Activation -- [EXTERNAL_OUT]
                 |
             [REP_OUT0](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mlp_forward_pattern_2layers)
        .set_priority(6.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    DEFINE_MLP_LAYER_START(f32)
                    DEFINE_MLP_LAYER(2, 1)
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mlp_forward_pattern_3layers)
        .set_priority(6.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    DEFINE_MLP_LAYER_START(f32)
                    DEFINE_MLP_LAYER(2, 1)
                    DEFINE_MLP_LAYER(3, 2)
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mlp_forward_pattern_4layers)
        .set_priority(6.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    DEFINE_MLP_LAYER_START(f32)
                    DEFINE_MLP_LAYER(2, 1)
                    DEFINE_MLP_LAYER(3, 2)
                    DEFINE_MLP_LAYER(4, 3)
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mlp_forward_pattern_5layers)
        .set_priority(6.3f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    DEFINE_MLP_LAYER_START(f32)
                    DEFINE_MLP_LAYER(2, 1)
                    DEFINE_MLP_LAYER(3, 2)
                    DEFINE_MLP_LAYER(4, 3)
                    DEFINE_MLP_LAYER(5, 4)
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mlp_forward_pattern_6layers)
        .set_priority(6.4f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    DEFINE_MLP_LAYER_START(f32)
                    DEFINE_MLP_LAYER(2, 1)
                    DEFINE_MLP_LAYER(3, 2)
                    DEFINE_MLP_LAYER(4, 3)
                    DEFINE_MLP_LAYER(5, 4)
                    DEFINE_MLP_LAYER(6, 5)
                });

/*
repetition unit:
  (f32)[gradient_x_next]     [gradient](f32)
        [x](f32)      \       /     [weight](f32)
            |          \     /           |
     StaticTranspose   Backprop    StaticTranspose
(optional)     \     /    |   \     /     (optional)
                Matmul Reduce  Matmul
                   | (optional)  |
       [gradient_w](f32)  |  [gradient_x](f32)
                          |
                  [gradient_bias](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mlp_backward_pattern)
        .set_priority(5.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bwd_mlp_layer
                            = std::make_shared<pb_graph_t>("bwd_mlp_layer");
                    auto activation_bwd = bwd_mlp_layer->append_alternation(
                            {impl::op_kind::ReLUBackprop,
                                    impl::op_kind::SigmoidBackprop,
                                    impl::op_kind::GELUBackprop},
                            "activation_bwd");
                    activation_bwd->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);

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

                    auto optional_reduce_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_reduce_subgraph");
                    auto reduce_bias = optional_reduce_subgraph->append_op(
                            impl::op_kind::ReduceSum, "reduce_bias");
                    reduce_bias->append_decision_function(check_reduce_attrs);
                    optional_reduce_subgraph->create_input_port(
                            0, reduce_bias, 0);
                    optional_reduce_subgraph->create_output_port(
                            0, reduce_bias, 0);
                    bwd_mlp_layer->append_optional(optional_reduce_subgraph,
                            {in_edge(0, activation_bwd, 0)}, "optional_reduce");

                    bwd_mlp_layer->create_input_port(0, activation_bwd, 1);
                    bwd_mlp_layer->create_output_port(0, matmul_layer, 0);

                    // repeat layer for [2, 10) times
                    pgraph->append_repetition(
                            bwd_mlp_layer, {0, 0}, 2, 10, "rep_unit");
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
             Activation
                 |
              Quantize
                 |
             [REP_OUT0](int8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, int8_mlp_pattern)
        .set_priority(6.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto mlp_layer = std::make_shared<pb_graph_t>("mlp_layer");

                    auto dequantize_input = mlp_layer->append_op(
                            impl::op_kind::Dequantize, "dequantize_input");
                    auto dequantize_weight = mlp_layer->append_op(
                            impl::op_kind::Dequantize, "dequantize_weight");
                    auto matmul = mlp_layer->append_op(impl::op_kind::MatMul,
                            {in_edge(0, dequantize_input, 0),
                                    in_edge(1, dequantize_weight, 0)},
                            "matmul");
                    auto activation = mlp_layer->append_alternation(
                            {impl::op_kind::ReLU, impl::op_kind::Sigmoid,
                                    impl::op_kind::GELU},
                            {in_edge(0, matmul, 0)}, "activation");
                    auto quantize_output = mlp_layer->append_op(
                            impl::op_kind::Quantize,
                            {in_edge(0, activation, 0)}, "quantize_output");

                    mlp_layer->create_input_port(0, dequantize_input, 0);
                    mlp_layer->create_output_port(0, quantize_output, 0);

                    // repeat layer for [2, 10) times
                    pgraph->append_repetition(
                            mlp_layer, {0, 0}, 2, 10, "rep_unit");
                });

COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(bf16_mlp_pattern)

/*
repetition unit:
 (bf16)[REP_IN0]   [REP_IN1](bf16)
              \     /
               MatMul
                 |
             Activation
                 |
             [REP_OUT0](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, bf16_mlp_pattern)
        .set_priority(5.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto mlp_layer = std::make_shared<pb_graph_t>("mlp_layer");
                    auto matmul = mlp_layer->append_op(
                            impl::op_kind::MatMul, "matmul");
                    matmul->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);

                    auto activation = mlp_layer->append_alternation(
                            {impl::op_kind::ReLU, impl::op_kind::Sigmoid,
                                    impl::op_kind::GELU},
                            {in_edge(0, matmul, 0)}, "activation");

                    mlp_layer->create_input_port(0, matmul, 0);
                    mlp_layer->create_output_port(0, activation, 0);

                    // repeat layer for [2, 10) times
                    pgraph->append_repetition(
                            mlp_layer, {0, 0}, 2, 10, "rep_unit");
                });

/*
repetition unit of 3-6 layers
 (bf16)[REP_IN0]   [REP_IN1](bf16)
              \     /
               MatMul -- [EXTERNAL_OUT]
                 |
             Activation -- [EXTERNAL_OUT]
                 |
             [REP_OUT0](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mlp_forward_pattern_2layers)
        .set_priority(6.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    DEFINE_MLP_LAYER_START(bf16)
                    DEFINE_MLP_LAYER(2, 1)
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mlp_forward_pattern_3layers)
        .set_priority(6.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    DEFINE_MLP_LAYER_START(bf16)
                    DEFINE_MLP_LAYER(2, 1)
                    DEFINE_MLP_LAYER(3, 2)
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mlp_forward_pattern_4layers)
        .set_priority(6.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    DEFINE_MLP_LAYER_START(bf16)
                    DEFINE_MLP_LAYER(2, 1)
                    DEFINE_MLP_LAYER(3, 2)
                    DEFINE_MLP_LAYER(4, 3)
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mlp_forward_pattern_5layers)
        .set_priority(6.3f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    DEFINE_MLP_LAYER_START(bf16)
                    DEFINE_MLP_LAYER(2, 1)
                    DEFINE_MLP_LAYER(3, 2)
                    DEFINE_MLP_LAYER(4, 3)
                    DEFINE_MLP_LAYER(5, 4)
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mlp_forward_pattern_6layers)
        .set_priority(6.4f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    DEFINE_MLP_LAYER_START(bf16)
                    DEFINE_MLP_LAYER(2, 1)
                    DEFINE_MLP_LAYER(3, 2)
                    DEFINE_MLP_LAYER(4, 3)
                    DEFINE_MLP_LAYER(5, 4)
                    DEFINE_MLP_LAYER(6, 5)
                });

/*
repetition unit:
 (bf16)[gradient_x_next]     [gradient](bf16)
       [x](bf16)      \       /     [weight](bf16)
            |          \     /           |
     StaticTranspose  Backprop    StaticTranspose
(optional)     \     /    |   \     /     (optional)
                Matmul Reduce  Matmul
                   | (optional)  |
      [gradient_w](bf16)  |  [gradient_x](bf16)
                          |
                  [gradient_bias](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mlp_backward_pattern)
        .set_priority(5.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bwd_mlp_layer
                            = std::make_shared<pb_graph_t>("bwd_mlp_layer");
                    auto activation_bwd = bwd_mlp_layer->append_alternation(
                            {impl::op_kind::ReLUBackprop,
                                    impl::op_kind::SigmoidBackprop,
                                    impl::op_kind::GELUBackprop},
                            "activation_bwd");
                    activation_bwd->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);

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

                    auto optional_reduce_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_reduce_subgraph");
                    auto reduce_bias = optional_reduce_subgraph->append_op(
                            impl::op_kind::ReduceSum, "reduce_bias");
                    reduce_bias->append_decision_function(check_reduce_attrs);
                    optional_reduce_subgraph->create_input_port(
                            0, reduce_bias, 0);
                    optional_reduce_subgraph->create_output_port(
                            0, reduce_bias, 0);

                    bwd_mlp_layer->append_optional(optional_reduce_subgraph,
                            {in_edge(0, activation_bwd, 0)}, "optional_reduce");

                    bwd_mlp_layer->create_input_port(0, activation_bwd, 1);
                    bwd_mlp_layer->create_output_port(0, matmul_layer, 0);

                    // repeat layer for [2, 10) times
                    pgraph->append_repetition(
                            bwd_mlp_layer, {0, 0}, 2, 10, "rep_unit");
                });

COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
