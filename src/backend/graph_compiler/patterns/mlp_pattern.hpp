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

COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
