/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_CONCAT_PATTERN_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_CONCAT_PATTERN_HPP

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

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(concat_patterns)

/* from GPT-J bf16
[IN0](dtype)  [IN1](dtype) [IN2](dtype)   [IN3](dtype)
      \            /            |              /
           add                 to1            /
            \                  /             /
                concat1                     /
                   |                       /
                permute                   /
                   \                     /
                         concat2
                            |
                           to2
                            |
                          [OUT0](dtype)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, add_to_concat_permute_concat_to)
        .set_priority(5.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(graph::partition_kind_t::concat_fusion_memory_optim)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto add_layer = pgraph->append_op(graph::op_kind::Add);
                    add_layer->allow_external_outputs();
                    auto to_layer1
                            = pgraph->append_op(graph::op_kind::TypeCast);
                    to_layer1->allow_external_outputs();
                    auto concat_layer1
                            = pgraph->append_op(graph::op_kind::Concat,
                                    {in_edge(0, add_layer, 0),
                                            in_edge(1, to_layer1, 0)});
                    auto permute_layer
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    {in_edge(0, concat_layer1, 0)});
                    permute_layer->allow_external_outputs();
                    auto concat_layer2
                            = pgraph->append_op(graph::op_kind::Concat,
                                    {in_edge(1, permute_layer, 0)});
                    concat_layer2->allow_external_outputs();
                    append_single_op_repetition_subgraph(pgraph,
                            graph::op_kind::TypeCast, concat_layer2, 0, 3);
                });

/* from GPT-J bf16
[IN0](dtype)  [IN1](dtype)  [IN2](dtype)
      \            /           |
           add                to
             \               /
                 concat
                    |
                 permute
                    |
                [OUT0](dtype)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, add_to_concat_permute)
        .set_priority(5.0f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(graph::partition_kind_t::concat_fusion_memory_optim)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto add_layer = pgraph->append_op(graph::op_kind::Add);
                    add_layer->allow_external_outputs();
                    auto to_layer1
                            = pgraph->append_op(graph::op_kind::TypeCast);
                    to_layer1->allow_external_outputs();
                    auto concat_layer1
                            = pgraph->append_op(graph::op_kind::Concat,
                                    {in_edge(0, add_layer, 0),
                                            in_edge(1, to_layer1, 0)});
                    auto permute_layer
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    {in_edge(0, concat_layer1, 0)});
                    permute_layer->allow_external_outputs();
                });

/* from GPT-J bf16
[IN0](dtype) [IN1](dtype)
    |             |
    |          permute
     \            /
         concat
           |
           to
           |
        [OUT0](dtype)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, permute_concat_to)
        .set_priority(5.0f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(graph::partition_kind_t::concat_fusion_memory_optim)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto permute_layer = pgraph->append_op(
                            graph::op_kind::StaticTranspose);
                    permute_layer->allow_external_outputs();
                    auto concat_layer
                            = pgraph->append_op(graph::op_kind::Concat,
                                    {in_edge(1, permute_layer, 0)});
                    concat_layer->allow_external_outputs();
                    append_single_op_repetition_subgraph(pgraph,
                            graph::op_kind::TypeCast, concat_layer, 0, 3);
                });

/* from Llama int8_bf16
typecast -> concat
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, typecast_concat)
        .set_priority(5.0f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(graph::partition_kind_t::concat_fusion_memory_optim)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto to_layer1
                            = pgraph->append_op(graph::op_kind::TypeCast);
                    to_layer1->allow_external_outputs();
                    pgraph->append_op(
                            graph::op_kind::Concat, {in_edge(1, to_layer1, 0)});
                });

/* from Llama int8_bf16
concat -> mul
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, concat_mul)
        .set_priority(5.0f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(graph::partition_kind_t::concat_fusion_memory_optim)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto concat_layer
                            = pgraph->append_op(graph::op_kind::Concat);
                    concat_layer->allow_external_outputs();
                    pgraph->append_op(graph::op_kind::Multiply,
                            {in_edge(0, concat_layer, 0)});
                });

COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(int8_concat_patterns)

/* from GPT-J int8-bf16
[IN0](dtype) [IN1](dtype)
    |             |
    |          permute
     \            /
         concat
           |
           to
           |
        quantize
           |
        [OUT0](dtype)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, permute_concat_to_quantize)
        .set_priority(5.6f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(graph::partition_kind_t::concat_fusion_memory_optim)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto permute_layer = pgraph->append_op(
                            graph::op_kind::StaticTranspose);
                    permute_layer->allow_external_outputs();
                    auto concat_layer
                            = pgraph->append_op(graph::op_kind::Concat,
                                    {in_edge(1, permute_layer, 0)});
                    concat_layer->allow_external_outputs();
                    auto typecast_rep = append_single_op_repetition_subgraph(
                            pgraph, graph::op_kind::TypeCast, concat_layer, 0,
                            3);
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, typecast_rep, 0)});
                });

/* from GPT-J int8_bf16
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, mul_mul_add_concat_permute_concat_quant)
        .set_priority(6.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(graph::partition_kind_t::concat_fusion_memory_optim)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto mul1 = pgraph->append_op(graph::op_kind::Multiply);
                    auto mul2 = pgraph->append_op(graph::op_kind::Multiply);
                    auto add_layer = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, mul1, 0), in_edge(1, mul2, 0)});
                    add_layer->allow_external_outputs();
                    auto concat_layer1 = pgraph->append_op(
                            graph::op_kind::Concat, {in_edge(0, add_layer, 0)});
                    auto permute_layer
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    {in_edge(0, concat_layer1, 0)});
                    permute_layer->allow_external_outputs();
                    auto concat_layer2
                            = pgraph->append_op(graph::op_kind::Concat,
                                    {in_edge(1, permute_layer, 0)});
                    concat_layer2->allow_external_outputs();
                    auto typecast_rep = append_single_op_repetition_subgraph(
                            pgraph, graph::op_kind::TypeCast, concat_layer2, 0,
                            3);
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, typecast_rep, 0)});
                });

/* from GPT-J int8_bf16
[IN0](dtype)  [IN1](dtype)  [IN2](dtype)
     |              \            /
     |                   add
      \                  /
             concat1
                |
             permute
                |
              quant
                |
            [OUT0](dtype)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, add_concat_permute_quant)
        .set_priority(6.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(graph::partition_kind_t::concat_fusion_memory_optim)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto add_layer = pgraph->append_op(graph::op_kind::Add);
                    add_layer->allow_external_outputs();
                    auto concat_layer = pgraph->append_op(
                            graph::op_kind::Concat, {in_edge(0, add_layer, 0)});
                    auto permute
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    {in_edge(0, concat_layer, 0)});
                    auto typecast_rep = append_single_op_repetition_subgraph(
                            pgraph, graph::op_kind::TypeCast, permute, 0, 3);
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, typecast_rep, 0)});
                });

/* from GPT-J int8_bf16
concat -> [typecasts ->] quant
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, concat_quant)
        .set_priority(5.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(graph::partition_kind_t::concat_fusion_memory_optim)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto concat_layer
                            = pgraph->append_op(graph::op_kind::Concat);
                    concat_layer->allow_external_outputs();
                    concat_layer->append_decision_function(
                            check_if_null_producer);
                    auto typecast_rep = append_single_op_repetition_subgraph(
                            pgraph, graph::op_kind::TypeCast, concat_layer, 0,
                            3);
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, typecast_rep, 0)});
                });

/* from Llama int8_bf16
add -> typecast -> concat -> typecasts -> quant
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, add_typecast_concat_typecasts_quant)
        .set_priority(6.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(graph::partition_kind_t::concat_fusion_memory_optim)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto add_layer = pgraph->append_op(graph::op_kind::Add);
                    auto to_layer1 = pgraph->append_op(graph::op_kind::TypeCast,
                            {in_edge(0, add_layer, 0)});
                    to_layer1->allow_external_outputs();
                    auto concat_layer = pgraph->append_op(
                            graph::op_kind::Concat, {in_edge(1, to_layer1, 0)});
                    concat_layer->allow_external_outputs();
                    auto typecast_rep = append_single_op_repetition_subgraph(
                            pgraph, graph::op_kind::TypeCast, concat_layer, 0,
                            3);
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, typecast_rep, 0)});
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
