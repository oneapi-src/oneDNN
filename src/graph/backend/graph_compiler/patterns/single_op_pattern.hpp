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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_SINGLE_OP_PATTERN_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_SINGLE_OP_PATTERN_HPP

#include <memory>
#include <utility>

#include "graph/backend/graph_compiler/compiler_graph.hpp"
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

#define DEFAULT_PRIORITY 1.f

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(single_op_pattern)
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, single_op_gc)
        .set_priority(DEFAULT_PRIORITY)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pgraph->append_alternation(get_no_constraint_ops());
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, single_op_reduce_gc)
        .set_priority(DEFAULT_PRIORITY)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *reduction
                            = pgraph->append_alternation(get_reduction_ops());
                    reduction->append_decision_function(check_input_num<1>);
                    reduction->append_decision_function(check_reduce_attrs);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, single_op_conv_gc)
        .set_priority(DEFAULT_PRIORITY)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *conv
                            = pgraph->append_alternation(
                                    get_conv_forward_ops());
                    conv->append_decision_function(check_conv_attrs);
                    conv->append_decision_function(check_isa_compatibility);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *conv_backward
                            = pgraph->append_alternation(
                                    get_conv_backward_ops());
                    conv_backward->append_decision_function(check_input_num<2>);
                    conv_backward->append_decision_function(check_conv_attrs);
                    conv_backward->append_decision_function(
                            check_isa_compatibility);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, single_op_batchnorm_training_gc)
        .set_priority(DEFAULT_PRIORITY)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *bn_fwd = pgraph->append_op(
                            graph::op_kind::BatchNormForwardTraining);
                    bn_fwd->append_decision_function(check_input_num<5>);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *bn_bwd = pgraph->append_op(
                            graph::op_kind::BatchNormForwardTraining);
                    bn_bwd->append_decision_function(check_input_num<5>);
                    bn_bwd->append_decision_function(check_output_num<3>);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, single_op_pooling_gc)
        .set_priority(DEFAULT_PRIORITY)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *pooling
                            = pgraph->append_alternation(get_pooling_ops());
                    pooling->append_decision_function(check_pooling_input_num);
                    pooling->append_decision_function(check_conv_attrs);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, single_op_matmul_gc)
        .set_priority(DEFAULT_PRIORITY)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *matmul
                            = pgraph->append_alternation(get_matmul_op());
                    matmul->append_decision_function(check_isa_compatibility);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
