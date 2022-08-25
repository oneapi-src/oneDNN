/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "backend/dnnl/kernels/large_partition.hpp"
#include "backend/dnnl/kernels/matmul.hpp"
#include "backend/dnnl/patterns/fusions.hpp"
#include "backend/dnnl/patterns/transformation_pattern.hpp"
#include "backend/dnnl/patterns/utils.hpp"

#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreatePattern = impl::pass::FCreatePattern;

/*!
 * \brief This provides matmul-related fusion, i.e.
 *        matmul-relu fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(matmul_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, matmul_post_ops_chain_fusion)
        .set_priority(8.8f)
        .set_kind(impl::partition_kind::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul);
                    pmatmul->append_decision_function(check_input_num<2>);

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            impl::op_kind::BatchNormInference, "pbn");
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, pmatmul, 0)}, "popt");

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            get_unary_binary_ops(), "palt");
                    palt->allow_internal_inputs();
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    pgraph->append_repetition(alt_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_matmul>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, matmul_bias_post_ops_chain_fusion)
        .set_priority(8.9f)
        .set_kind(impl::partition_kind::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul);
                    pmatmul->append_decision_function(check_input_num<2>);
                    pm::pb_op_t *biasadd
                            = pgraph->append_op(impl::op_kind::BiasAdd,
                                    in_edges_t {in_edge(0, pmatmul, 0)});

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            impl::op_kind::BatchNormInference, "pbn");
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, biasadd, 0)}, "popt");

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            get_unary_binary_ops(), "palt");
                    palt->allow_internal_inputs();
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    pgraph->append_repetition(alt_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul);
                    pmatmul->append_decision_function(check_input_num<3>);

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            impl::op_kind::BatchNormInference, "pbn");
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, pmatmul, 0)}, "popt");

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            get_unary_binary_ops(), "palt");
                    palt->allow_internal_inputs();
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    pgraph->append_repetition(alt_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_matmul>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, matmul_transpose_optional_reshape_fusion)
        .set_priority(9.f)
        .set_kind(impl::partition_kind::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul);

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    // Optional pre reshape
                    auto popt_reshape_pre_graph = std::make_shared<pb_graph_t>(
                            "poptional_reshape_pre");
                    pm::pb_op_t *preshape_pre
                            = popt_reshape_pre_graph->append_op(
                                    impl::op_kind::StaticReshape,
                                    "preshape_pre");
                    popt_reshape_pre_graph->create_input_port(
                            0, preshape_pre, 0);
                    popt_reshape_pre_graph->create_output_port(
                            0, preshape_pre, 0);
                    auto popt_reshape_pre
                            = pgraph->append_optional(popt_reshape_pre_graph,
                                    in_edges_t {in_edge(0, popt_bias, 0)},
                                    "popt_reshape_pre");

                    // transpose
                    auto ptranspose = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            in_edges_t {in_edge(0, popt_reshape_pre, 0)},
                            "ptranspose");

                    // Optional post reshape
                    auto popt_reshape_post_graph = std::make_shared<pb_graph_t>(
                            "poptional_reshape_post");
                    pm::pb_op_t *preshape_post
                            = popt_reshape_post_graph->append_op(
                                    impl::op_kind::StaticReshape,
                                    "preshape_post");
                    popt_reshape_post_graph->create_input_port(
                            0, preshape_post, 0);
                    popt_reshape_post_graph->create_output_port(
                            0, preshape_post, 0);
                    pgraph->append_optional(popt_reshape_post_graph,
                            in_edges_t {in_edge(0, ptranspose, 0)},
                            "popt_reshape_post");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_matmul>();
        });

/*
MatMul: Currently DNNL Backend doesn't support Reorder with zero points
(used in weight u8->s8) on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_matmul_div_add_fusion_cpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op_t *matmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    matmul->append_decision_function(check_input_num<2>);

                    pm::pb_op_t *div = pgraph->append_op(impl::op_kind::Divide,
                            in_edges_t {in_edge(0, matmul, 0)});
                    pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, div, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
MatMul: Currently DNNL Backend doesn't support Reorder with zero points
(used in weight u8->s8) on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_matmul_div_add_fusion_gpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    dequant_weight->append_decision_function(
                            check_input_dtype<impl::data_type::s8>);
                    pm::pb_op_t *matmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    matmul->append_decision_function(check_input_num<2>);

                    pm::pb_op_t *div = pgraph->append_op(impl::op_kind::Divide,
                            in_edges_t {in_edge(0, matmul, 0)});
                    pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, div, 0)});
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
              [bias]*                      [dequant]*
                |                       for Add/Multiply/Maximum/
                |                        Minimum/Divide/Subtract
                |                             /
        [ Abs/Clamp/Elu/Exp/GELU/HardSwish/Log/Sigmoid/SoftPlus/
          Pow/ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/Maximum/Minimum/
          Divide/Subtract]*[0,3]
                |
            [quant_out]*  
                |      
*/
/*
MatMul: Currently DNNL Backend doesn't support below
features on GPU:
1. Post-sum/binary with zero points
2. Reorder with zero points (used in weight u8->s8)
While CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_matmul_post_ops_fusion_cpu)
        .set_priority(9.9f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "matmul");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    auto pint8_binary_graph = std::make_shared<pb_graph_t>(
                            "pint8_binary_graph");
                    pm::pb_op_t *pdequant_binary
                            = pint8_binary_graph->append_op(
                                    impl::op_kind::Dequantize, "dequant");
                    pm::pb_op_t *pbinary
                            = pint8_binary_graph->append_alternation(
                                    get_binary_ops(),
                                    in_edges_t {in_edge(1, pdequant_binary, 0)},
                                    "pbinary");
                    pint8_binary_graph->create_input_port(0, pbinary, 0);
                    pint8_binary_graph->create_input_port(
                            1, pdequant_binary, 0);
                    pint8_binary_graph->create_output_port(0, pbinary, 0);

                    auto postop_graph
                            = std::make_shared<pb_graph_t>("postop_graph");
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            get_unary_binary_ops(), "postop");
                    pop->allow_internal_inputs();
                    postop_graph->create_input_port(0, pop, 0);
                    postop_graph->create_input_port(1, pop, 1);
                    postop_graph->create_output_port(0, pop, 0);

                    auto prep_graph
                            = std::make_shared<pb_graph_t>("prep_graph");
                    auto palt = prep_graph->append_alternation(
                            {pint8_binary_graph, postop_graph}, "palternation");
                    prep_graph->create_input_port(0, palt, 0);
                    prep_graph->create_input_port(1, palt, 1);
                    prep_graph->create_output_port(0, palt, 0);

                    auto prep = pgraph->append_repetition(prep_graph, {0, 0}, 0,
                            MAX_REPETITION,
                            in_edges_t {in_edge(0, popt_bias, 0)},
                            "prepetition");

                    // Optional quant_out
                    auto popt_qout_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_out");
                    pm::pb_op_t *pquant_out = popt_qout_graph->append_op(
                            impl::op_kind::Quantize, "pquant_out");
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)}, "popt_quant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
MatMul: Currently DNNL Backend doesn't support below
features on GPU:
1. Post-sum/binary with zero points
2. Reorder with zero points (used in weight u8->s8)
While CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_matmul_post_ops_fusion_gpu)
        .set_priority(9.9f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");
                    dequant_weight->append_decision_function(
                            check_input_dtype<impl::data_type::s8>);

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "matmul");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    auto pint8_binary_graph = std::make_shared<pb_graph_t>(
                            "pint8_binary_graph");
                    pm::pb_op_t *pdequant_binary
                            = pint8_binary_graph->append_op(
                                    impl::op_kind::Dequantize, "dequant");
                    pdequant_binary->append_decision_function(
                            check_zps_values<0>);
                    pm::pb_op_t *pbinary
                            = pint8_binary_graph->append_alternation(
                                    get_binary_ops(),
                                    in_edges_t {in_edge(1, pdequant_binary, 0)},
                                    "pbinary");
                    pint8_binary_graph->create_input_port(0, pbinary, 0);
                    pint8_binary_graph->create_input_port(
                            1, pdequant_binary, 0);
                    pint8_binary_graph->create_output_port(0, pbinary, 0);

                    auto postop_graph
                            = std::make_shared<pb_graph_t>("postop_graph");
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            get_unary_binary_ops(), "postop");
                    pop->allow_internal_inputs();
                    postop_graph->create_input_port(0, pop, 0);
                    postop_graph->create_input_port(1, pop, 1);
                    postop_graph->create_output_port(0, pop, 0);

                    auto prep_graph
                            = std::make_shared<pb_graph_t>("prep_graph");
                    auto palt = prep_graph->append_alternation(
                            {pint8_binary_graph, postop_graph}, "palternation");
                    prep_graph->create_input_port(0, palt, 0);
                    prep_graph->create_input_port(1, palt, 1);
                    prep_graph->create_output_port(0, palt, 0);

                    auto prep = pgraph->append_repetition(prep_graph, {0, 0}, 0,
                            MAX_REPETITION,
                            in_edges_t {in_edge(0, popt_bias, 0)},
                            "prepetition");

                    // Optional quant_out
                    auto popt_qout_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_out");
                    pm::pb_op_t *pquant_out = popt_qout_graph->append_op(
                            impl::op_kind::Quantize, "pquant_out");
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)}, "popt_quant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
MatMul: Currently DNNL Backend doesn't support Reorder with zero points
(used in weight u8->s8) on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_bf16_matmul_scale_add_fusion_cpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *matmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});
                    matmul->append_decision_function(check_input_num<2>);

                    pm::pb_op_t *scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            in_edges_t {in_edge(0, matmul, 0)});
                    pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, scale, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
MatMul: Currently DNNL Backend doesn't support Reorder with zero points
(used in weight u8->s8) on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_bf16_matmul_scale_add_fusion_gpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    dequant_weight->append_decision_function(
                            check_input_dtype<impl::data_type::s8>);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *matmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});
                    matmul->append_decision_function(check_input_num<2>);

                    pm::pb_op_t *scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            in_edges_t {in_edge(0, matmul, 0)});
                    pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, scale, 0)});
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
              [bias]*    [dequant_other -> typecast_other]* for Add
                |          /
 [ ReLU/GELU/Divide/Multiply/Add ]
                |
  [typecast_out -> quant_out]*
*/
/*
MatMul: Currently DNNL Backend doesn't support Reorder with zero points
(used in weight u8->s8) on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_bf16_matmul_post_ops_fusion_cpu)
        .set_priority(10.4f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    // Optional quant_weight
                    auto popt_quant_wei_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_quant_wei_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_quant_wei_graph->create_input_port(0, pquant, 0);
                    popt_quant_wei_graph->create_output_port(0, pquant, 0);
                    auto popt_quant_wei = pgraph->append_optional(
                            popt_quant_wei_graph, "popt");

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt_quant_wei, 0)},
                                    "dequant_weight");

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *matmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias
                    auto popt_bias = optional_bias_add(pgraph, matmul, true);

                    // post add with dequant->typecast
                    auto padd_graph
                            = std::make_shared<pb_graph_t>("padd_graph");
                    pm::pb_op_t *pdequant_add = padd_graph->append_op(
                            impl::op_kind::Dequantize, "dequant_add");
                    pm::pb_op_t *typecast_add
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, pdequant_add, 0)});
                    pm::pb_op_t *padd = padd_graph->append_op(
                            impl::op_kind::Add,
                            in_edges_t {in_edge(1, typecast_add, 0)}, "padd");
                    padd_graph->create_input_port(0, padd, 0);
                    padd_graph->create_input_port(1, pdequant_add, 0);
                    padd_graph->create_output_port(0, padd, 0);

                    auto other_postop_graph = std::make_shared<pb_graph_t>(
                            "pother_postop_graph");
                    pm::pb_op_t *pop = other_postop_graph->append_alternation(
                            {impl::op_kind::ReLU, impl::op_kind::GELU,
                                    impl::op_kind::Divide,
                                    impl::op_kind::Multiply,
                                    impl::op_kind::Add},
                            "pother_postop");
                    other_postop_graph->create_input_port(0, pop, 0);
                    other_postop_graph->create_input_port(1, pop, 1);
                    other_postop_graph->create_output_port(0, pop, 0);

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            {padd_graph, other_postop_graph}, "palt");
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    auto prep = pgraph->append_optional(alt_graph,
                            in_edges_t {in_edge(0, popt_bias, 0)},
                            "prepetition");

                    // Optional typecast_out + quant_out
                    auto popt_qout_graph = std::make_shared<pb_graph_t>(
                            "poptional_tc_quant_out");
                    pm::pb_op_t *ptc_out = popt_qout_graph->append_op(
                            impl::op_kind::TypeCast, "ptc_out");
                    pm::pb_op_t *pquant_out = popt_qout_graph->append_op(
                            impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, ptc_out, 0)}, "pquant_out");
                    popt_qout_graph->create_input_port(0, ptc_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)},
                            "popt_tc_quant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

/*
MatMul: Currently DNNL Backend doesn't support Reorder with zero points
(used in weight u8->s8) on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_bf16_matmul_post_ops_fusion_gpu)
        .set_priority(10.4f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    // Optional quant_weight
                    auto popt_quant_wei_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_quant_wei_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_quant_wei_graph->create_input_port(0, pquant, 0);
                    popt_quant_wei_graph->create_output_port(0, pquant, 0);
                    auto popt_quant_wei = pgraph->append_optional(
                            popt_quant_wei_graph, "popt");

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt_quant_wei, 0)},
                                    "dequant_weight");
                    dequant_weight->append_decision_function(
                            check_input_dtype<impl::data_type::s8>);

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *matmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias
                    auto popt_bias = optional_bias_add(pgraph, matmul, true);

                    // post add with dequant->typecast
                    auto padd_graph
                            = std::make_shared<pb_graph_t>("padd_graph");
                    pm::pb_op_t *pdequant_add = padd_graph->append_op(
                            impl::op_kind::Dequantize, "dequant_add");
                    pm::pb_op_t *typecast_add
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, pdequant_add, 0)});
                    pm::pb_op_t *padd = padd_graph->append_op(
                            impl::op_kind::Add,
                            in_edges_t {in_edge(1, typecast_add, 0)}, "padd");
                    padd_graph->create_input_port(0, padd, 0);
                    padd_graph->create_input_port(1, pdequant_add, 0);
                    padd_graph->create_output_port(0, padd, 0);

                    auto other_postop_graph = std::make_shared<pb_graph_t>(
                            "pother_postop_graph");
                    pm::pb_op_t *pop = other_postop_graph->append_alternation(
                            {impl::op_kind::ReLU, impl::op_kind::GELU,
                                    impl::op_kind::Divide,
                                    impl::op_kind::Multiply,
                                    impl::op_kind::Add},
                            "pother_postop");
                    other_postop_graph->create_input_port(0, pop, 0);
                    other_postop_graph->create_input_port(1, pop, 1);
                    other_postop_graph->create_output_port(0, pop, 0);

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            {padd_graph, other_postop_graph}, "palt");
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    auto prep = pgraph->append_optional(alt_graph,
                            in_edges_t {in_edge(0, popt_bias, 0)},
                            "prepetition");

                    // Optional typecast_out + quant_out
                    auto popt_qout_graph = std::make_shared<pb_graph_t>(
                            "poptional_tc_quant_out");
                    pm::pb_op_t *ptc_out = popt_qout_graph->append_op(
                            impl::op_kind::TypeCast, "ptc_out");
                    pm::pb_op_t *pquant_out = popt_qout_graph->append_op(
                            impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, ptc_out, 0)}, "pquant_out");
                    popt_qout_graph->create_input_port(0, ptc_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)},
                            "popt_tc_quant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_matmul_transpose_optional_reshape_fusion)
        .set_priority(10.f)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "matmul");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    // Optional pre reshape
                    auto popt_reshape_pre_graph = std::make_shared<pb_graph_t>(
                            "poptional_reshape_pre");
                    pm::pb_op_t *preshape_pre
                            = popt_reshape_pre_graph->append_op(
                                    impl::op_kind::StaticReshape,
                                    "preshape_pre");
                    popt_reshape_pre_graph->create_input_port(
                            0, preshape_pre, 0);
                    popt_reshape_pre_graph->create_output_port(
                            0, preshape_pre, 0);
                    auto popt_reshape_pre
                            = pgraph->append_optional(popt_reshape_pre_graph,
                                    in_edges_t {in_edge(0, popt_bias, 0)},
                                    "popt_reshape_pre");

                    // transpose
                    auto ptranspose = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            in_edges_t {in_edge(0, popt_reshape_pre, 0)},
                            "ptranspose");

                    // Optional post reshape
                    auto popt_reshape_post_graph = std::make_shared<pb_graph_t>(
                            "poptional_reshape_post");
                    pm::pb_op_t *preshape_post
                            = popt_reshape_post_graph->append_op(
                                    impl::op_kind::StaticReshape,
                                    "preshape_post");
                    popt_reshape_post_graph->create_input_port(
                            0, preshape_post, 0);
                    popt_reshape_post_graph->create_output_port(
                            0, preshape_post, 0);
                    auto popt_reshape_post
                            = pgraph->append_optional(popt_reshape_post_graph,
                                    in_edges_t {in_edge(0, ptranspose, 0)},
                                    "popt_reshape_post");

                    // quant_out
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, popt_reshape_post, 0)},
                            "pquant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_bf16_matmul_transpose_optional_reshape_fusion)
        .set_priority(10.5f)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)},
                                    "matmul");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, true);

                    // Optional pre reshape
                    auto popt_reshape_pre_graph = std::make_shared<pb_graph_t>(
                            "poptional_reshape_pre");
                    pm::pb_op_t *preshape_pre
                            = popt_reshape_pre_graph->append_op(
                                    impl::op_kind::StaticReshape,
                                    "preshape_pre");
                    popt_reshape_pre_graph->create_input_port(
                            0, preshape_pre, 0);
                    popt_reshape_pre_graph->create_output_port(
                            0, preshape_pre, 0);
                    auto popt_reshape_pre
                            = pgraph->append_optional(popt_reshape_pre_graph,
                                    in_edges_t {in_edge(0, popt_bias, 0)},
                                    "popt_reshape_pre");

                    // transpose
                    auto ptranspose = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            in_edges_t {in_edge(0, popt_reshape_pre, 0)},
                            "ptranspose");

                    // Optional post reshape
                    auto popt_reshape_post_graph = std::make_shared<pb_graph_t>(
                            "poptional_reshape_post");
                    pm::pb_op_t *preshape_post
                            = popt_reshape_post_graph->append_op(
                                    impl::op_kind::StaticReshape,
                                    "preshape_post");
                    popt_reshape_post_graph->create_input_port(
                            0, preshape_post, 0);
                    popt_reshape_post_graph->create_output_port(
                            0, preshape_post, 0);
                    auto popt_reshape_post
                            = pgraph->append_optional(popt_reshape_post_graph,
                                    in_edges_t {in_edge(0, ptranspose, 0)},
                                    "popt_reshape_post");

                    pm::pb_op_t *typecast_dst = pgraph->append_op(
                            impl::op_kind::TypeCast,
                            in_edges_t {in_edge(0, popt_reshape_post, 0)});
                    typecast_dst->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);

                    // quant_out
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, typecast_dst, 0)},
                            "pquant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, matmul_transpose_reorder_fusion)
        .set_priority(9.1f)
        .set_kind(impl::partition_kind::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul);

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    // transpose
                    auto ptranspose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, popt_bias, 0)},
                                    "ptranspose");

                    // reorder
                    pgraph->append_op(impl::op_kind::Reorder,
                            in_edges_t {in_edge(0, ptranspose, 0)}, "preorder");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_matmul>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_matmul_transpose_reorder_fusion)
        .set_priority(10.f)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "matmul");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, false);

                    // transpose
                    auto ptranspose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, popt_bias, 0)},
                                    "ptranspose");

                    // reorder
                    auto preorder = pgraph->append_op(impl::op_kind::Reorder,
                            in_edges_t {in_edge(0, ptranspose, 0)}, "preorder");

                    // Optional quant_out
                    auto popt_qout_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_out");
                    pm::pb_op_t *pquant_out = popt_qout_graph->append_op(
                            impl::op_kind::Quantize, "pquant_out");
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, preorder, 0)},
                            "popt_quant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_bf16_matmul_transpose_reorder_fusion)
        .set_priority(10.5f)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)},
                                    "matmul");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pmatmul, true);

                    // transpose
                    auto ptranspose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, popt_bias, 0)},
                                    "ptranspose");

                    // reorder
                    auto preorder = pgraph->append_op(impl::op_kind::Reorder,
                            in_edges_t {in_edge(0, ptranspose, 0)}, "preorder");

                    // Optional typecast + quant_out
                    auto popt_tc_qout_graph = std::make_shared<pb_graph_t>(
                            "poptional_tc_quant_out");
                    pm::pb_op_t *typecast_dst = popt_tc_qout_graph->append_op(
                            impl::op_kind::TypeCast, "ptc_out");
                    typecast_dst->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    pm::pb_op_t *pquant_out = popt_tc_qout_graph->append_op(
                            impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, typecast_dst, 0)},
                            "pquant_out");
                    popt_tc_qout_graph->create_input_port(0, typecast_dst, 0);
                    popt_tc_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_tc_qout_graph,
                            in_edges_t {in_edge(0, preorder, 0)},
                            "popt_tc_quant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_MHA_fusion)
        .set_priority(22.0f)
        .set_kind(impl::partition_kind::quantized_mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_query");

                    auto dequantize_key = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_key");

                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, dequantize_query, 0),
                                    in_edge(1, dequantize_key, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            in_edges_t {in_edge(0, matmul_qk, 0)},
                            "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, fscore_scale, 0)},
                            "fscore_add");
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            in_edges_t {in_edge(0, fscore_add, 0)}, "softmax");
                    auto quantize_softmax
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, softmax, 0)},
                                    "quantize_softmax");
                    auto dequantize_softmax = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, quantize_softmax, 0)},
                            "dequantize_softmax");

                    auto dequantize_value = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_value");
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, dequantize_softmax, 0),
                                    in_edge(1, dequantize_value, 0)},
                            "matmul_v");

                    auto transpose_output
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, matmul_v, 0)},
                                    "transpose_output");
                    auto reshape_reorder_output = pgraph->append_alternation(
                            {impl::op_kind::Reorder,
                                    impl::op_kind::StaticReshape},
                            {in_edge(0, transpose_output, 0)},
                            "reshape_reorder_output");
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, reshape_reorder_output, 0)},
                            "quantize_output");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// fp32 MHA pattern
/*
   (f32)[Query]    [Key](f32)
              \     /
               MatMul  [fscore scale](f32)
                 \    /
[Attention Mask] Div|Mul
              \   /
                Add
                 |
              Softmax  [Value](f32)
                    \     /
                     MatMul
                        |
                    Transpose
                        |
                Reorder|StaticReshape
                        |
                     [output](f32)
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, f32_MHA_fusion)
        .set_priority(21.0f)
        .set_kind(impl::partition_kind::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            impl::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto fscore_scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, softmax, 0)}, "matmul_v");
                    matmul_v->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto transpose_output = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");
                    pgraph->append_alternation(
                            {impl::op_kind::Reorder,
                                    impl::op_kind::StaticReshape},
                            {in_edge(0, transpose_output, 0)},
                            "reshape_reorder_output");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

/*
        (int8)[Query]   [Key](int8)
                 |          |
             Dequantize Dequantize
                 |          |
              TypeCast   TypeCast
                   \     /
                    MatMul  [Fscore Scale](f32)
                      \    /
(bf16)[Attention Mask] Div|Mul
                   \   /
                     Add
                      |
                   Softmax
                      |
                   TypeCast
                      |
                   Quantize  [Value](int8)
                      |          |
                  Dequantize Dequantize
                      |          |
                   TypeCast   TypeCast
                         \     /
                          MatMul
                             |
                         Transpose
                             |
                Reorder|StaticReshape
                             |
                          TypeCast
                             |
                          Quantize
                             |
                        [output](int8)
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_bf16_MHA_fusion)
        .set_priority(22.0f)
        .set_kind(impl::partition_kind::quantized_mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_query");
                    auto cast_query = pgraph->append_op(impl::op_kind::TypeCast,
                            {in_edge(0, dequantize_query, 0)}, "cast_query");

                    auto dequantize_key = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_key");
                    auto cast_key = pgraph->append_op(impl::op_kind::TypeCast,
                            {in_edge(0, dequantize_key, 0)}, "cast_key");

                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, cast_query, 0),
                                    in_edge(1, cast_key, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto cast_softmax_fp32 = pgraph->append_op(
                            impl::op_kind::TypeCast, {in_edge(0, softmax, 0)},
                            "cast_softmax_fp32");
                    auto quantize_softmax
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    {in_edge(0, cast_softmax_fp32, 0)},
                                    "quantize_softmax");
                    auto dequantize_softmax
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)},
                                    "dequantize_softmax");
                    auto cast_softmax
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    {in_edge(0, dequantize_softmax, 0)},
                                    "cast_softmax");

                    auto dequantize_value = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_value");
                    auto cast_value = pgraph->append_op(impl::op_kind::TypeCast,
                            {in_edge(0, dequantize_value, 0)}, "cast_value");

                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, cast_softmax, 0),
                                    in_edge(1, cast_value, 0)},
                            "matmul_v");
                    auto transpose_output = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");
                    auto reshape_reorder_output = pgraph->append_alternation(
                            {impl::op_kind::Reorder,
                                    impl::op_kind::StaticReshape},
                            {in_edge(0, transpose_output, 0)},
                            "reshape_reorder_output");
                    auto cast_output_fp32
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    {in_edge(0, reshape_reorder_output, 0)},
                                    "cast_output_fp32");
                    pgraph->append_op(impl::op_kind::Quantize,
                            {in_edge(0, cast_output_fp32, 0)},
                            "quantize_output");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

/*
MatMul: Currently DNNL Backend doesn't support Reorder with zero points
(used in weight u8->s8) on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, x8x8bf16_div_matmul_fusion_cpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *div = pgraph->append_op(impl::op_kind::Divide,
                            in_edges_t {in_edge(0, typecast_data, 0)});
                    pm::pb_op_t *matmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, div, 0),
                                            in_edge(1, typecast_weight, 0)});
                    matmul->append_decision_function(check_input_num<2>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });
/*
MatMul: Currently DNNL Backend doesn't support Reorder with zero points
(used in weight u8->s8) on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, x8x8bf16_div_matmul_fusion_gpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(impl::partition_kind::quantized_matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    dequant_weight->append_decision_function(
                            check_input_dtype<impl::data_type::s8>);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op_t *div = pgraph->append_op(impl::op_kind::Divide,
                            in_edges_t {in_edge(0, typecast_data, 0)});

                    pm::pb_op_t *matmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, div, 0),
                                            in_edge(1, typecast_weight, 0)});
                    matmul->append_decision_function(check_input_num<2>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_matmul>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
