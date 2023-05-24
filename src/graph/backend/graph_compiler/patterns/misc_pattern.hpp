/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_MISC_PATTERN_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_MISC_PATTERN_HPP

#include <memory>
#include <utility>

#include "graph/backend/graph_compiler/patterns/fusions.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {
namespace pass {

using pb_graph_t = graph::utils::pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(misc_pattern)

/*
(f32/bf16)[IN0]
            |
 TypeCast*[0-1]    [IN1](f32/bf16)
             \     /
             Multiply
                |
            TypeCast*[0-2]
                |
            Quantize
                |
           [output](int8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, mul_typecast_quantize)
        .set_priority(1.f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto mul = pgraph->append_op(graph::op_kind::Multiply);
                    mul->append_decision_function(check_if_null_producer);
                    auto prep = append_single_op_repetition_subgraph(
                            pgraph, graph::op_kind::TypeCast, mul, 0, 3);
                    auto quantize = pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, prep, 0)});
                    UNUSED(quantize);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto cast = pgraph->append_op(graph::op_kind::TypeCast);
                    cast->append_decision_function(check_if_null_producer);
                    auto mul = pgraph->append_op(graph::op_kind::Multiply,
                            in_edges_t {in_edge(0, cast, 0)});
                    auto prep = append_single_op_repetition_subgraph(
                            pgraph, graph::op_kind::TypeCast, mul, 0, 3);
                    auto quantize = pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, prep, 0)});
                    UNUSED(quantize);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
