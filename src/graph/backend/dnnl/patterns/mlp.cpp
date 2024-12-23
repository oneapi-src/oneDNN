/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "graph/backend/dnnl/kernels/large_partition.hpp"

#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

#include "graph/utils/pm/pbuilder.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(mlp)

/*
//        /      \
//  matmul (gt)  matmul (up)
//     |          |
//    unary*      |
//        \      /
//        multiply
//           |
//      matmul (down)
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, gated_mlp)
        .set_priority(22.0f)
        .set_kind(partition_kind_t::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *fc_up
                            = pgraph->append_op(graph::op_kind::MatMul);
                    pm::pb_op_t *fc_gt
                            = pgraph->append_op(graph::op_kind::MatMul);
                    pgraph->create_input_port(0, fc_up, 0);
                    pgraph->create_input_port(0, fc_gt, 0);

                    // activations after fc_gt
                    auto alt_graph = std::make_shared<pb_graph_t>();
                    auto palt = alt_graph->append_alternation(get_unary_ops());
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);
                    // The activation is optional
                    auto act = pgraph->append_optional(
                            alt_graph, in_edges_t {in_edge(0, fc_gt, 0)});

                    // binary: add/div/mul/sub
                    in_edges_t edges
                            = {in_edge(0, act, 0), in_edge(1, fc_up, 0)};
                    auto bin = pgraph->append_alternation(
                            get_binary_ops(), edges);

                    // fc_down
                    pgraph->append_op(graph::op_kind::MatMul,
                            in_edges_t {in_edge(0, bin, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// gated mlp with swish decomposed to sigmoid and multiply.
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, gated_mlp_v1)
        .set_priority(22.05f)
        .set_kind(partition_kind_t::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *fc_up
                            = pgraph->append_op(graph::op_kind::MatMul);
                    pm::pb_op_t *fc_gt
                            = pgraph->append_op(graph::op_kind::MatMul);
                    pgraph->create_input_port(0, fc_up, 0);
                    pgraph->create_input_port(0, fc_gt, 0);

                    // swish (sigmoid + mul) after fc_gt
                    pm::pb_op_t *swish_sig = pgraph->append_op(
                            graph::op_kind::Sigmoid, {in_edge(0, fc_gt, 0)});
                    in_edges_t swish_mul_edges
                            = {in_edge(0, fc_gt, 0), in_edge(1, swish_sig, 0)};
                    pm::pb_op_t *swish_mul = pgraph->append_op(
                            graph::op_kind::Multiply, swish_mul_edges);

                    // binary: add/div/mul/sub
                    in_edges_t edges
                            = {in_edge(0, swish_mul, 0), in_edge(1, fc_up, 0)};
                    auto bin = pgraph->append_alternation(
                            get_binary_ops(), edges);

                    // fc_down
                    pgraph->append_op(graph::op_kind::MatMul,
                            in_edges_t {in_edge(0, bin, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

/*
//        |          |
//       deq        deq
//  \   /        \  /
//  matmul (gt)  matmul (up)
//     |          |
//    unary*      |
//        \      /   deq
//         binary   /
//           |     /
//           matmul (down)
//              |
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, quantized_gated_mlp)
        .set_priority(22.1f)
        .set_kind(partition_kind_t::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *deq_up = pgraph->append_op(
                            graph::op_kind::DynamicDequantize);
                    pm::pb_op_t *deq_gate = pgraph->append_op(
                            graph::op_kind::DynamicDequantize);
                    pm::pb_op_t *fc_up = pgraph->append_op(
                            graph::op_kind::MatMul, {in_edge(1, deq_up, 0)});
                    pm::pb_op_t *fc_gt = pgraph->append_op(
                            graph::op_kind::MatMul, {in_edge(1, deq_gate, 0)});
                    pgraph->create_input_port(0, fc_up, 0);
                    pgraph->create_input_port(0, fc_gt, 0);

                    // activations after fc_gt
                    auto alt_graph = std::make_shared<pb_graph_t>();
                    auto palt = alt_graph->append_alternation(get_unary_ops());
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);
                    // The activation is optional
                    auto act = pgraph->append_optional(
                            alt_graph, in_edges_t {in_edge(0, fc_gt, 0)});

                    // binary: add/div/mul/sub
                    in_edges_t edges
                            = {in_edge(0, act, 0), in_edge(1, fc_up, 0)};
                    auto bin = pgraph->append_alternation(
                            get_binary_ops(), edges);

                    // fc_down
                    pm::pb_op_t *deq_down = pgraph->append_op(
                            graph::op_kind::DynamicDequantize);
                    in_edges_t fc_down_edges
                            = {in_edge(0, bin, 0), in_edge(1, deq_down, 0)};
                    pgraph->append_op(graph::op_kind::MatMul, fc_down_edges);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// quantized gated mlp with swish decomposed to sigmoid and multiply.
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, quantized_gated_mlp_v1)
        .set_priority(22.1f)
        .set_kind(partition_kind_t::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *deq_up = pgraph->append_op(
                            graph::op_kind::DynamicDequantize);
                    pm::pb_op_t *deq_gate = pgraph->append_op(
                            graph::op_kind::DynamicDequantize);
                    pm::pb_op_t *fc_up = pgraph->append_op(
                            graph::op_kind::MatMul, {in_edge(1, deq_up, 0)});
                    pm::pb_op_t *fc_gt = pgraph->append_op(
                            graph::op_kind::MatMul, {in_edge(1, deq_gate, 0)});
                    pgraph->create_input_port(0, fc_up, 0);
                    pgraph->create_input_port(0, fc_gt, 0);

                    // swish (sigmoid + mul) after fc_gt
                    pm::pb_op_t *swish_sig = pgraph->append_op(
                            graph::op_kind::Sigmoid, {in_edge(0, fc_gt, 0)});
                    in_edges_t swish_mul_edges
                            = {in_edge(0, fc_gt, 0), in_edge(1, swish_sig, 0)};
                    pm::pb_op_t *swish_mul = pgraph->append_op(
                            graph::op_kind::Multiply, swish_mul_edges);

                    // binary: add/div/mul/sub
                    in_edges_t edges
                            = {in_edge(0, swish_mul, 0), in_edge(1, fc_up, 0)};
                    auto bin = pgraph->append_alternation(
                            get_binary_ops(), edges);

                    // fc_down
                    pm::pb_op_t *deq_down = pgraph->append_op(
                            graph::op_kind::DynamicDequantize);
                    in_edges_t fc_down_edges
                            = {in_edge(0, bin, 0), in_edge(1, deq_down, 0)};
                    pgraph->append_op(graph::op_kind::MatMul, fc_down_edges);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
