/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
#include "graph/backend/dnnl/kernels/matmul.hpp"
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

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(matmul_post_ops)

/*
              \   /
              matmul
                |
             [bias]*
                |
            [BatchNorm]*
                |
        [unary/binary]*[0,4]
                |
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, fp_matmul_post_ops)
        .set_priority(8.8f)
        .set_kind(partition_kind_t::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(graph::op_kind::MatMul);

                    // Optional bias
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    // Optional BN
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    auto pbn = popt_graph->append_op(
                            graph::op_kind::BatchNormInference);
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, popt_bias, 0)});

                    auto alt_graph = std::make_shared<pb_graph_t>();
                    auto palt = alt_graph->append_alternation(
                            get_unary_binary_ops());
                    palt->allow_internal_inputs();
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    pgraph->append_repetition(alt_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_matmul>();
        });

/*
              \   /
              matmul
                |
             [bias]*
                |
            [Reshape]*
                |
            transpose
                |
            [Reshape]*
                |
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(
        dnnl, fp_matmul_reshape_transpose_reshape)
        .set_priority(9.f)
        .set_kind(partition_kind_t::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(graph::op_kind::MatMul);

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    // Optional pre reshape
                    auto popt_reshape_pre_graph
                            = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *preshape_pre
                            = popt_reshape_pre_graph->append_op(
                                    graph::op_kind::StaticReshape);
                    popt_reshape_pre_graph->create_input_port(
                            0, preshape_pre, 0);
                    popt_reshape_pre_graph->create_output_port(
                            0, preshape_pre, 0);
                    auto popt_reshape_pre
                            = pgraph->append_optional(popt_reshape_pre_graph,
                                    in_edges_t {in_edge(0, popt_bias, 0)});

                    // transpose
                    auto ptranspose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            in_edges_t {in_edge(0, popt_reshape_pre, 0)});

                    // Optional post reshape
                    auto popt_reshape_post_graph
                            = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *preshape_post
                            = popt_reshape_post_graph->append_op(
                                    graph::op_kind::StaticReshape);
                    popt_reshape_post_graph->create_input_port(
                            0, preshape_post, 0);
                    popt_reshape_post_graph->create_output_port(
                            0, preshape_post, 0);
                    pgraph->append_optional(popt_reshape_post_graph,
                            in_edges_t {in_edge(0, ptranspose, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_matmul>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        \_____       _____/
              matmul
                |
              [bias]*
                |
        [unary/binary]*[0,4]
                |
            [quant_out]*
                |      
*/
/*
MatMul: Currently DNNL Backend doesn't support below
features on GPU:
1. Reorder with zero points (used in weight u8->s8)
While CPU supports.

Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8x8x_matmul_post_ops_cpu)
        .set_priority(9.9f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    auto postop_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            get_unary_binary_ops());
                    pop->allow_internal_inputs();
                    postop_graph->create_input_port(0, pop, 0);
                    postop_graph->create_input_port(1, pop, 1);
                    postop_graph->create_output_port(0, pop, 0);

                    auto prep = pgraph->append_repetition(postop_graph, {0, 0},
                            0, MAX_REPETITION,
                            in_edges_t {in_edge(0, popt_bias, 0)});

                    // Optional quant_out
                    auto popt_qout_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant_out = popt_qout_graph->append_op(
                            graph::op_kind::Quantize);
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(
                            popt_qout_graph, in_edges_t {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
MatMul: Currently DNNL Backend doesn't support below
features on GPU:
1. Reorder with zero points (used in weight u8->s8)
While CPU supports.

Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8s8x_matmul_post_ops_gpu)
        .set_priority(9.9f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    auto postop_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            get_unary_binary_ops());
                    pop->allow_internal_inputs();
                    postop_graph->create_input_port(0, pop, 0);
                    postop_graph->create_input_port(1, pop, 1);
                    postop_graph->create_output_port(0, pop, 0);

                    auto prep = pgraph->append_repetition(postop_graph, {0, 0},
                            0, MAX_REPETITION,
                            in_edges_t {in_edge(0, popt_bias, 0)});

                    // Optional quant_out
                    auto popt_qout_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant_out = popt_qout_graph->append_op(
                            graph::op_kind::Quantize);
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(
                            popt_qout_graph, in_edges_t {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        \_____       _____/
              matmul
                |
              [bias]* dequant
                |       /
               add
                |
        [unary/binary]*[0,4]
                |
            quant_out
                |      
*/
/*
MatMul: Currently DNNL Backend doesn't support below
features on GPU:
1. Post-sum with zero points
2. Reorder with zero points (used in weight u8->s8)
While CPU supports.

Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8x8x8_matmul_add_post_ops_cpu)
        .set_priority(10.f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    // dequantize(rhs) -> add
                    auto prep = post_quantized_add(pgraph, popt_bias);

                    // quantize
                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
MatMul: Currently DNNL Backend doesn't support below
features on GPU:
1. Post-sum with zero points
2. Reorder with zero points (used in weight u8->s8)
While CPU supports.

Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8s8x8_matmul_add_post_ops_gpu)
        .set_priority(10.f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    // dequantize(rhs) -> add
                    auto prep = post_quantized_add(
                            pgraph, popt_bias, /*check_zps*/ true);

                    // quantize
                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        |                  |
   typecast_data    typecast_weight
        \_____       _____/
              matmul
                | [typecast]*
                |   /
              [bias]*
                |
        [unary/binary]*[0,4]
                |
          [typecast_out]*
                |
           [quant_out]*
*/
/*
MatMul: Currently DNNL Backend doesn't support Reorder with zero points
(used in weight u8->s8) on GPU, while CPU supports.

Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8x8x_tc_matmul_post_ops_cpu)
        .set_priority(10.4f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    // Optional quant_weight
                    auto popt_quant_wei_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant = popt_quant_wei_graph->append_op(
                            graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_quant_wei_graph->create_input_port(0, pquant, 0);
                    popt_quant_wei_graph->create_output_port(0, pquant, 0);
                    auto popt_quant_wei
                            = pgraph->append_optional(popt_quant_wei_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt_quant_wei, 0)});

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    pm::pb_op_t *matmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias
                    auto popt_bias = optional_bias_add(pgraph, matmul, true);

                    auto other_postop_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pop = other_postop_graph->append_alternation(
                            get_unary_binary_ops());
                    pop->allow_internal_inputs();
                    other_postop_graph->create_input_port(0, pop, 0);
                    other_postop_graph->create_input_port(1, pop, 1);
                    other_postop_graph->create_output_port(0, pop, 0);

                    auto prep = pgraph->append_repetition(other_postop_graph,
                            {0, 0}, 0, MAX_REPETITION,
                            in_edges_t {in_edge(0, popt_bias, 0)});

                    // Optional typecast_out
                    auto popt_tc_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *ptc_out = popt_tc_graph->append_op(
                            graph::op_kind::TypeCast);
                    popt_tc_graph->create_input_port(0, ptc_out, 0);
                    popt_tc_graph->create_output_port(0, ptc_out, 0);
                    auto tc_out = pgraph->append_optional(
                            popt_tc_graph, in_edges_t {in_edge(0, prep, 0)});

                    optional_smooth_quant(pgraph, tc_out, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
MatMul: Currently DNNL Backend doesn't support Reorder with zero points
(used in weight u8->s8) on GPU, while CPU supports.

Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8s8x_tc_matmul_post_ops_gpu)
        .set_priority(10.4f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    // Optional quant_weight
                    auto popt_quant_wei_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant = popt_quant_wei_graph->append_op(
                            graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_quant_wei_graph->create_input_port(0, pquant, 0);
                    popt_quant_wei_graph->create_output_port(0, pquant, 0);
                    auto popt_quant_wei
                            = pgraph->append_optional(popt_quant_wei_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt_quant_wei, 0)});
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    pm::pb_op_t *matmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias
                    auto popt_bias = optional_bias_add(pgraph, matmul, true);

                    auto other_postop_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pop = other_postop_graph->append_alternation(
                            get_unary_binary_ops());
                    pop->allow_internal_inputs();
                    other_postop_graph->create_input_port(0, pop, 0);
                    other_postop_graph->create_input_port(1, pop, 1);
                    other_postop_graph->create_output_port(0, pop, 0);

                    auto prep = pgraph->append_repetition(other_postop_graph,
                            {0, 0}, 0, MAX_REPETITION,
                            in_edges_t {in_edge(0, popt_bias, 0)});

                    // Optional typecast_out
                    auto popt_tc_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *ptc_out = popt_tc_graph->append_op(
                            graph::op_kind::TypeCast);
                    popt_tc_graph->create_input_port(0, ptc_out, 0);
                    popt_tc_graph->create_output_port(0, ptc_out, 0);
                    auto tc_out = pgraph->append_optional(
                            popt_tc_graph, in_edges_t {in_edge(0, prep, 0)});

                    optional_smooth_quant(pgraph, tc_out, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        |                  |
   typecast_data    typecast_weight
        \_____       _____/
              matmul
                | [typecast]*
                |   /
              [bias]*    dequant_other
                |            /
                |     typecast_other
                |     /
               Add
                |
        [unary/binary]*[0,4]
                |
          typecast_out
                |
            quant_out
*/
/*
MatMul: Currently DNNL Backend doesn't support below
features on GPU:
1. Reorder with zero points (used in weight u8->s8)
2. Post-sum with zero points
while CPU supports.

Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(
        dnnl, x8x8x8_tc_matmul_add_post_ops_cpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    // Optional quant_weight
                    auto popt_quant_wei_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant = popt_quant_wei_graph->append_op(
                            graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_quant_wei_graph->create_input_port(0, pquant, 0);
                    popt_quant_wei_graph->create_output_port(0, pquant, 0);
                    auto popt_quant_wei
                            = pgraph->append_optional(popt_quant_wei_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt_quant_wei, 0)});

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    pm::pb_op_t *matmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias
                    auto popt_bias = optional_bias_add(pgraph, matmul, true);

                    // post add with dequant->typecast
                    pm::pb_op_t *pdequant_add
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pm::pb_op_t *typecast_add
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, pdequant_add, 0)});
                    pm::pb_op_t *padd = pgraph->append_op(graph::op_kind::Add,
                            in_edges_t {in_edge(0, popt_bias, 0),
                                    in_edge(1, typecast_add, 0)});

                    auto other_postop_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pop = other_postop_graph->append_alternation(
                            get_unary_binary_ops());
                    pop->allow_internal_inputs();
                    other_postop_graph->create_input_port(0, pop, 0);
                    other_postop_graph->create_input_port(1, pop, 1);
                    other_postop_graph->create_output_port(0, pop, 0);

                    auto popt_post_ops = pgraph->append_repetition(
                            other_postop_graph, {0, 0}, 0, MAX_REPETITION,
                            in_edges_t {in_edge(0, padd, 0)});

                    // typecast_out + quant_out
                    pm::pb_op_t *ptc_out
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, popt_post_ops, 0)});
                    optional_smooth_quant(pgraph, ptc_out, false);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
MatMul: Currently DNNL Backend doesn't support below
features on GPU:
1. Reorder with zero points (used in weight u8->s8)
2. Post-sum with zero points
while CPU supports.

Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(
        dnnl, x8s8x8_tc_matmul_add_post_ops_gpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    // Optional quant_weight
                    auto popt_quant_wei_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant = popt_quant_wei_graph->append_op(
                            graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_quant_wei_graph->create_input_port(0, pquant, 0);
                    popt_quant_wei_graph->create_output_port(0, pquant, 0);
                    auto popt_quant_wei
                            = pgraph->append_optional(popt_quant_wei_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt_quant_wei, 0)});
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    pm::pb_op_t *matmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias
                    auto popt_bias = optional_bias_add(pgraph, matmul, true);

                    // post add with dequant->typecast
                    pm::pb_op_t *pdequant_add
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pdequant_add->append_decision_function(check_zps_values<0>);
                    pm::pb_op_t *typecast_add
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, pdequant_add, 0)});
                    pm::pb_op_t *padd = pgraph->append_op(graph::op_kind::Add,
                            in_edges_t {in_edge(0, popt_bias, 0),
                                    in_edge(1, typecast_add, 0)});

                    auto other_postop_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pop = other_postop_graph->append_alternation(
                            get_unary_binary_ops());
                    pop->allow_internal_inputs();
                    other_postop_graph->create_input_port(0, pop, 0);
                    other_postop_graph->create_input_port(1, pop, 1);
                    other_postop_graph->create_output_port(0, pop, 0);

                    auto popt_post_ops = pgraph->append_repetition(
                            other_postop_graph, {0, 0}, 0, MAX_REPETITION,
                            in_edges_t {in_edge(0, padd, 0)});

                    // typecast_out + quant_out
                    pm::pb_op_t *ptc_out
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, popt_post_ops, 0)});
                    optional_smooth_quant(pgraph, ptc_out, false);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        \_____       _____/
              matmul
                |
             [bias]*
                |
            [Reshape]*
                |
            transpose
                |
            [Reshape]*
                |
            quant_out
                |
*/
/*
Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(
        dnnl, x8x8x8_matmul_reshape_transpose_reshape)
        .set_priority(10.f)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    // Optional pre reshape
                    auto popt_reshape_pre_graph
                            = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *preshape_pre
                            = popt_reshape_pre_graph->append_op(
                                    graph::op_kind::StaticReshape);
                    popt_reshape_pre_graph->create_input_port(
                            0, preshape_pre, 0);
                    popt_reshape_pre_graph->create_output_port(
                            0, preshape_pre, 0);
                    auto popt_reshape_pre
                            = pgraph->append_optional(popt_reshape_pre_graph,
                                    in_edges_t {in_edge(0, popt_bias, 0)});

                    // transpose
                    auto ptranspose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            in_edges_t {in_edge(0, popt_reshape_pre, 0)});

                    // Optional post reshape
                    auto popt_reshape_post_graph
                            = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *preshape_post
                            = popt_reshape_post_graph->append_op(
                                    graph::op_kind::StaticReshape);
                    popt_reshape_post_graph->create_input_port(
                            0, preshape_post, 0);
                    popt_reshape_post_graph->create_output_port(
                            0, preshape_post, 0);
                    auto popt_reshape_post
                            = pgraph->append_optional(popt_reshape_post_graph,
                                    in_edges_t {in_edge(0, ptranspose, 0)});

                    // quant_out
                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, popt_reshape_post, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        |                  |
   typecast_data    typecast_weight
        \_____       _____/
              matmul
                |
             [bias]*
                |
            [Reshape]*
                |
            transpose
                |
            [Reshape]*
                |
            typecast
                |
            quant_out
                |
*/
/*
Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(
        dnnl, x8x8x8_tc_matmul_reshape_transpose_reshape)
        .set_priority(10.5f)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, true);

                    // Optional pre reshape
                    auto popt_reshape_pre_graph
                            = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *preshape_pre
                            = popt_reshape_pre_graph->append_op(
                                    graph::op_kind::StaticReshape);
                    popt_reshape_pre_graph->create_input_port(
                            0, preshape_pre, 0);
                    popt_reshape_pre_graph->create_output_port(
                            0, preshape_pre, 0);
                    auto popt_reshape_pre
                            = pgraph->append_optional(popt_reshape_pre_graph,
                                    in_edges_t {in_edge(0, popt_bias, 0)});

                    // transpose
                    auto ptranspose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            in_edges_t {in_edge(0, popt_reshape_pre, 0)});

                    // Optional post reshape
                    auto popt_reshape_post_graph
                            = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *preshape_post
                            = popt_reshape_post_graph->append_op(
                                    graph::op_kind::StaticReshape);
                    popt_reshape_post_graph->create_input_port(
                            0, preshape_post, 0);
                    popt_reshape_post_graph->create_output_port(
                            0, preshape_post, 0);
                    auto popt_reshape_post
                            = pgraph->append_optional(popt_reshape_post_graph,
                                    in_edges_t {in_edge(0, ptranspose, 0)});

                    pm::pb_op_t *typecast_dst = pgraph->append_op(
                            graph::op_kind::TypeCast,
                            in_edges_t {in_edge(0, popt_reshape_post, 0)});
                    typecast_dst->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);

                    // quant_out
                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, typecast_dst, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
               \  /
              matmul
                |
             [bias]*
                |
            transpose
                |
             reorder
                |
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, fp_matmul_transpose_reorder)
        .set_priority(9.1f)
        .set_kind(partition_kind_t::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(graph::op_kind::MatMul);

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    // transpose
                    auto ptranspose
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, popt_bias, 0)});

                    // reorder
                    pgraph->append_op(graph::op_kind::Reorder,
                            in_edges_t {in_edge(0, ptranspose, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_matmul>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        \_____       _____/
              matmul
                |
             [bias]*
                |
            transpose
                |
             reorder
                |
           [quant_out]*
                |
*/
/*
Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8x8x_matmul_transpose_reorder)
        .set_priority(10.f)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    // transpose
                    auto ptranspose
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, popt_bias, 0)});

                    // reorder
                    auto preorder = pgraph->append_op(graph::op_kind::Reorder,
                            in_edges_t {in_edge(0, ptranspose, 0)});

                    // Optional quant_out
                    auto popt_qout_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant_out = popt_qout_graph->append_op(
                            graph::op_kind::Quantize);
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, preorder, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        |                  |
   typecast_data    typecast_weight
        \_____       _____/
              matmul
                |
             [bias]*
                |
            transpose
                |
             reorder
                |
     [typecast-> quant_out]*
                |
*/
/*
Note: This pattern also accepts fp32 as weight input
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(
        dnnl, x8x8x_tc_matmul_transpose_reorder)
        .set_priority(10.5f)
        .set_kind(partition_kind_t::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, true);

                    // transpose
                    auto ptranspose
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, popt_bias, 0)});

                    // reorder
                    auto preorder = pgraph->append_op(graph::op_kind::Reorder,
                            in_edges_t {in_edge(0, ptranspose, 0)});

                    // Optional typecast + quant_out
                    auto popt_tc_qout_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *typecast_dst = popt_tc_qout_graph->append_op(
                            graph::op_kind::TypeCast);
                    typecast_dst->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    pm::pb_op_t *pquant_out = popt_tc_qout_graph->append_op(
                            graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, typecast_dst, 0)});
                    popt_tc_qout_graph->create_input_port(0, typecast_dst, 0);
                    popt_tc_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_tc_qout_graph,
                            in_edges_t {in_edge(0, preorder, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
