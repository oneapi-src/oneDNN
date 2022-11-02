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

#include "backend/dnnl/kernels/conv.hpp"
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
 * \brief This provides conv-related fusion, i.e.
 *        conv-relu fusion, conv-bn fusion, conv-sum fusion, conv-bn-sum fusion, etc.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(conv_post_ops_fusion)

// Conv: Currently DNNL backend doesn't support conv + depthwise conv
// post-op fusion on GPU, while CPU supports. Check engine_kind == cpu
// before matching
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, conv_depthwise_fusion_cpu)
        .set_priority(10.2f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(impl::partition_kind::convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op_t *depthwise
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, conv, 0)});
                    depthwise->append_decision_function(check_input_num<2>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_conv_fwd>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        \_____       _____/
               conv
                |
              [bias]*
                |
                |                         dequant_add
                |                             /
        [ Abs/Clamp/Elu/Exp/GELU/HardSwish/Log/Sigmoid/SoftPlus/
          Pow/ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/Maximum/Minimum/
          Divide/Subtract]*[0,3]
                |
            [quant_out]*  
                |      
*/
/*
Conv: Currently DNNL Backend doesn't support below
features on GPU:
1. Post-sum/binary with zero points
2. Reorder with zero points (used in weight u8->s8)
While CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_conv_post_ops_int8_add_fusion_cpu)
        .set_priority(10.6f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(impl::partition_kind::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op_t *pconv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "conv");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    auto pint8_add_graph
                            = std::make_shared<pb_graph_t>("pint8_add_graph");
                    pm::pb_op_t *pdequant_add = pint8_add_graph->append_op(
                            impl::op_kind::Dequantize, "dequant");
                    pm::pb_op_t *padd = pint8_add_graph->append_op(
                            impl::op_kind::Add,
                            in_edges_t {in_edge(1, pdequant_add, 0)}, "padd");
                    pint8_add_graph->create_input_port(0, padd, 0);
                    pint8_add_graph->create_input_port(1, pdequant_add, 0);
                    pint8_add_graph->create_output_port(0, padd, 0);

                    // unary + binary post ops exclude add
                    auto postop_graph
                            = std::make_shared<pb_graph_t>("postops_graph");
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Elu, impl::op_kind::Exp,
                                    impl::op_kind::GELU,
                                    impl::op_kind::HardSwish,
                                    impl::op_kind::LeakyReLU,
                                    impl::op_kind::Log, impl::op_kind::Mish,
                                    impl::op_kind::Sigmoid,
                                    impl::op_kind::SoftPlus, impl::op_kind::Pow,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sqrt, impl::op_kind::Square,
                                    impl::op_kind::Tanh,
                                    impl::op_kind::Multiply,
                                    impl::op_kind::Maximum,
                                    impl::op_kind::Minimum,
                                    impl::op_kind::Divide,
                                    impl::op_kind::Subtract},
                            "postop");
                    pop->allow_internal_inputs();
                    postop_graph->create_input_port(0, pop, 0);
                    postop_graph->create_input_port(1, pop, 1);
                    postop_graph->create_output_port(0, pop, 0);

                    auto prep_graph
                            = std::make_shared<pb_graph_t>("prep_graph");
                    auto palt = prep_graph->append_alternation(
                            {pint8_add_graph, postop_graph}, "palternation");
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
                    pm::pb_op_t *pquant_out = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant_out");
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)}, "popt_quant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_conv>();
        });

/*
Conv: Currently DNNL Backend doesn't support below
features on GPU:
1. Post-sum/binary with zero points
2. Reorder with zero points (used in weight u8->s8)
While CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_conv_post_ops_int8_add_fusion_gpu)
        .set_priority(10.6f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(impl::partition_kind::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");
                    dequant_weight->append_decision_function(
                            check_input_dtype<impl::data_type::s8>);

                    pm::pb_op_t *pconv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "conv");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    auto pint8_add_graph
                            = std::make_shared<pb_graph_t>("pint8_add_graph");
                    pm::pb_op_t *pdequant_add = pint8_add_graph->append_op(
                            impl::op_kind::Dequantize, "dequant");
                    pdequant_add->append_decision_function(check_zps_values<0>);
                    pm::pb_op_t *padd = pint8_add_graph->append_op(
                            impl::op_kind::Add,
                            in_edges_t {in_edge(1, pdequant_add, 0)}, "padd");
                    pint8_add_graph->create_input_port(0, padd, 0);
                    pint8_add_graph->create_input_port(1, pdequant_add, 0);
                    pint8_add_graph->create_output_port(0, padd, 0);

                    // unary + binary post ops exclude add
                    auto postop_graph
                            = std::make_shared<pb_graph_t>("postops_graph");
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Elu, impl::op_kind::Exp,
                                    impl::op_kind::GELU,
                                    impl::op_kind::HardSwish,
                                    impl::op_kind::LeakyReLU,
                                    impl::op_kind::Log, impl::op_kind::Mish,
                                    impl::op_kind::Sigmoid,
                                    impl::op_kind::SoftPlus, impl::op_kind::Pow,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sqrt, impl::op_kind::Square,
                                    impl::op_kind::Tanh,
                                    impl::op_kind::Multiply,
                                    impl::op_kind::Maximum,
                                    impl::op_kind::Minimum,
                                    impl::op_kind::Divide,
                                    impl::op_kind::Subtract},
                            "postop");
                    pop->allow_internal_inputs();
                    postop_graph->create_input_port(0, pop, 0);
                    postop_graph->create_input_port(1, pop, 1);
                    postop_graph->create_output_port(0, pop, 0);

                    auto prep_graph
                            = std::make_shared<pb_graph_t>("prep_graph");
                    auto palt = prep_graph->append_alternation(
                            {pint8_add_graph, postop_graph}, "palternation");
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
                    pm::pb_op_t *pquant_out = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant_out");
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)}, "popt_quant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_conv>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        \_____       _____/
               conv
                |
              [bias]*
                |           [dequant]* for Multiply/
                |        Maximum/Minimum/Divide/Subtract
                |                   /
        [ Abs/Clamp/Elu/Exp/GELU/HardSwish/Log/Sigmoid/SoftPlus/
          Pow/ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/Maximum/Minimum/
          Divide/Subtract]*[0,3]
                |
            [quant_out]*
                |
*/
/*
Conv: Currently DNNL Backend doesn't support below
features on GPU:
1. Post-sum/binary with zero points
2. Reorder with zero points (used in weight u8->s8)
While CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_conv_post_ops_fusion_cpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(impl::partition_kind::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op_t *pconv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "conv");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    auto pint8_binary_graph = std::make_shared<pb_graph_t>(
                            "pint8_binary_graph");
                    pm::pb_op_t *pdequant_binary
                            = pint8_binary_graph->append_op(
                                    impl::op_kind::Dequantize, "dequant");
                    pm::pb_op_t *pbinary
                            = pint8_binary_graph->append_alternation(
                                    {impl::op_kind::Multiply,
                                            impl::op_kind::Maximum,
                                            impl::op_kind::Minimum,
                                            impl::op_kind::Divide,
                                            impl::op_kind::Subtract},
                                    in_edges_t {in_edge(1, pdequant_binary, 0)},
                                    "pbinary");
                    pint8_binary_graph->create_input_port(0, pbinary, 0);
                    pint8_binary_graph->create_input_port(
                            1, pdequant_binary, 0);
                    pint8_binary_graph->create_output_port(0, pbinary, 0);

                    // post ops
                    auto postop_graph
                            = std::make_shared<pb_graph_t>("postops_graph");
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
                    pm::pb_op_t *pquant_out = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant_out");
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)}, "popt_quant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_conv>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_conv_post_ops_fusion_gpu)
        .set_priority(10.6f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(impl::partition_kind::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");
                    dequant_weight->append_decision_function(
                            check_input_dtype<impl::data_type::s8>);

                    pm::pb_op_t *pconv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "conv");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    auto pint8_binary_graph = std::make_shared<pb_graph_t>(
                            "pint8_binary_graph");
                    pm::pb_op_t *pdequant_binary
                            = pint8_binary_graph->append_op(
                                    impl::op_kind::Dequantize, "dequant");
                    pdequant_binary->append_decision_function(
                            check_zps_values<0>);
                    pm::pb_op_t *pbinary
                            = pint8_binary_graph->append_alternation(
                                    {impl::op_kind::Multiply,
                                            impl::op_kind::Maximum,
                                            impl::op_kind::Minimum,
                                            impl::op_kind::Divide,
                                            impl::op_kind::Subtract},
                                    in_edges_t {in_edge(1, pdequant_binary, 0)},
                                    "pbinary");
                    pint8_binary_graph->create_input_port(0, pbinary, 0);
                    pint8_binary_graph->create_input_port(
                            1, pdequant_binary, 0);
                    pint8_binary_graph->create_output_port(0, pbinary, 0);

                    // post ops
                    auto postop_graph
                            = std::make_shared<pb_graph_t>("postops_graph");
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
                    pm::pb_op_t *pquant_out = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant_out");
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)}, "popt_quant_out");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_conv>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        |                  |
    typecast             typecast
        \_____       _____/
               conv
                | [typecast]*
                |   /
              [bias]*       
                |
              [GeLU]*
                |
             typecast
                |
            quant_out
                |
*/
/*
Conv: Currently DNNL Backend doesn't support below
features on GPU:
1. Reorder with zero points (used in weight u8->s8)
while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_conv_bias_fusion_cpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(impl::partition_kind::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

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
                    pm::pb_op_t *convolution
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias_add
                    auto popt_bias
                            = optional_bias_add(pgraph, convolution, true);

                    // optional GELU
                    auto popt_gelu_graph
                            = std::make_shared<pb_graph_t>("poptional_gelu");
                    pm::pb_op_t *gelu
                            = popt_gelu_graph->append_op(impl::op_kind::GELU);
                    popt_gelu_graph->create_input_port(0, gelu, 0);
                    popt_gelu_graph->create_output_port(0, gelu, 0);
                    auto popt_gelu = pgraph->append_optional(popt_gelu_graph,
                            in_edges_t {in_edge(0, popt_bias, 0)}, "popt_gelu");

                    pm::pb_op_t *typecast_gelu
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, popt_gelu, 0)});
                    typecast_gelu->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, typecast_gelu, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_conv>();
        });

/*
Conv: Currently DNNL Backend doesn't support below
features on GPU:
1. Reorder with zero points (used in weight u8->s8)
while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_conv_bias_fusion_gpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(impl::partition_kind::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");
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
                    pm::pb_op_t *convolution
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias_add
                    auto popt_bias
                            = optional_bias_add(pgraph, convolution, true);

                    // optional GELU
                    auto popt_gelu_graph
                            = std::make_shared<pb_graph_t>("poptional_gelu");
                    pm::pb_op_t *gelu
                            = popt_gelu_graph->append_op(impl::op_kind::GELU);
                    popt_gelu_graph->create_input_port(0, gelu, 0);
                    popt_gelu_graph->create_output_port(0, gelu, 0);
                    auto popt_gelu = pgraph->append_optional(popt_gelu_graph,
                            in_edges_t {in_edge(0, popt_bias, 0)}, "popt_gelu");

                    pm::pb_op_t *typecast_gelu
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, popt_gelu, 0)});
                    typecast_gelu->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, typecast_gelu, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_conv>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        |                  |
   typecast_data    typecast_weight
        \_____       _____/
               conv
                | [typecast]*
                |   /
              [bias]*    [dequant_other -> typecast_other]* for Add
                |          /
 [ ReLU/GELU/Divide/Multiply/Add ]
                |
  [typecast_out -> quant_out]*
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_bf16_conv_post_ops_fusion)
        .set_priority(10.4f)
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

                    pm::pb_op_t *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias
                    auto popt_bias = optional_bias_add(pgraph, conv, true);

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

                    // post binary with typecast
                    auto other_binary_with_tc_graph
                            = std::make_shared<pb_graph_t>(
                                    "pother_binary_with_tc_graph");
                    pm::pb_op_t *typecast_binary = pgraph->append_op(
                            impl::op_kind::TypeCast, "typecast_binary");
                    pm::pb_op_t *pbinary_op
                            = other_binary_with_tc_graph->append_alternation(
                                    get_binary_ops(),
                                    in_edges_t {in_edge(1, typecast_binary, 0)},
                                    "pother_binary_postop");
                    other_binary_with_tc_graph->create_input_port(
                            0, pbinary_op, 0);
                    other_binary_with_tc_graph->create_input_port(
                            1, typecast_binary, 1);
                    other_binary_with_tc_graph->create_output_port(
                            0, pbinary_op, 0);

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
                            {padd_graph, other_binary_with_tc_graph,
                                    other_postop_graph},
                            "palt");
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
            return std::make_shared<quantized_conv>();
        });

/*
              \   /
              conv
                |
        [ Abs/Clamp/Elu/Exp/GELU/HardSwish/Log/Sigmoid/SoftPlus/
          Pow/ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/
          Maximum/Minimum/Divide/Subtract]*[0,3] 
                |
            [TypeCast]*
                |
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, conv_post_ops_fusion)
        .set_priority(9.7f)
        .set_kind(impl::partition_kind::convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *pconv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    pconv->append_decision_function(check_input_num<2>);

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            impl::op_kind::BatchNormInference, "pbn");
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, pconv, 0)}, "popt");

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            get_unary_binary_ops(), "palt");
                    palt->allow_internal_inputs();
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    auto prep = pgraph->append_repetition(alt_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");

                    // Optional typecast
                    auto popt_tc_graph
                            = std::make_shared<pb_graph_t>("poptional_tc");
                    auto ptc = popt_tc_graph->append_op(
                            impl::op_kind::TypeCast, "ptc");
                    ptc->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    ptc->append_decision_function(
                            check_output_dtype<impl::data_type::f32>);
                    popt_tc_graph->create_input_port(0, ptc, 0);
                    popt_tc_graph->create_output_port(0, ptc, 0);
                    pgraph->append_optional(
                            popt_tc_graph, {in_edge(0, prep, 0)}, "popt_tc");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_conv_fwd>();
        });

/*
              \   /
              conv
                |
              bias
                |
        [ Abs/Clamp/Elu/Exp/GELU/HardSwish/Log/Sigmoid/SoftPlus/
          Pow/ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/
          Maximum/Minimum/Divide/Subtract]*[0,3] 
                |
           [TypeCast]*
                |
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, conv_bias_post_ops_fusion)
        .set_priority(9.8f)
        .set_kind(impl::partition_kind::convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *conv = pgraph->append_op(
                            impl::op_kind::Convolution, "conv");
                    conv->append_decision_function(check_input_num<2>);
                    pm::pb_op_t *biasadd
                            = pgraph->append_op(impl::op_kind::BiasAdd,
                                    in_edges_t {in_edge(0, conv, 0)}, "bias");

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

                    auto prep = pgraph->append_repetition(alt_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");

                    // Optional typecast
                    auto popt_tc_graph
                            = std::make_shared<pb_graph_t>("poptional_tc");
                    auto ptc = popt_tc_graph->append_op(
                            impl::op_kind::TypeCast, "ptc");
                    ptc->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    ptc->append_decision_function(
                            check_output_dtype<impl::data_type::f32>);
                    popt_tc_graph->create_input_port(0, ptc, 0);
                    popt_tc_graph->create_output_port(0, ptc, 0);
                    pgraph->append_optional(
                            popt_tc_graph, {in_edge(0, prep, 0)}, "popt_tc");
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *conv = pgraph->append_op(
                            impl::op_kind::Convolution, "conv");
                    conv->append_decision_function(check_input_num<3>);

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            impl::op_kind::BatchNormInference, "pbn");
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, conv, 0)}, "popt");

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            get_unary_binary_ops(), "palt");
                    palt->allow_internal_inputs();
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    auto prep = pgraph->append_repetition(alt_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");

                    // Optional typecast
                    auto popt_tc_graph
                            = std::make_shared<pb_graph_t>("poptional_tc");
                    auto ptc = popt_tc_graph->append_op(
                            impl::op_kind::TypeCast, "ptc");
                    ptc->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    ptc->append_decision_function(
                            check_output_dtype<impl::data_type::f32>);
                    popt_tc_graph->create_input_port(0, ptc, 0);
                    popt_tc_graph->create_output_port(0, ptc, 0);
                    pgraph->append_optional(
                            popt_tc_graph, {in_edge(0, prep, 0)}, "popt_tc");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_conv_fwd>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, conv_bwd_weights_bwd_bias_fusion)
        .set_enable(false)
        .set_kind(impl::partition_kind::convolution_backprop_post_ops)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *wildcard = pgraph->append_op(
                            impl::op_kind::Wildcard, "pwild");
                    pgraph->append_op(impl::op_kind::ConvolutionBackpropFilters,
                            in_edges_t {in_edge(1, wildcard, 0)});
                    pgraph->append_op(impl::op_kind::BiasAddBackprop,
                            in_edges_t {in_edge(0, wildcard, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<conv_bwd_weights_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
