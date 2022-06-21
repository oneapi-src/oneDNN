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

#include "backend/dnnl/patterns/fusions.hpp"
#include "backend/dnnl/patterns/transformation_pattern.hpp"
#include "backend/dnnl/patterns/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

/*!
 * \brief This provides convtranspose-related fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(convtranspose_fusion)

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
         \              /
           convtranspose
                |
              [bias]*                      [dequant]*
                |                       for Add/Multiply/Maximum/
                |                        Minimum/Divide/Subtract
                |                             /
        [ Abs/Clamp/Elu/GELU/Log/Sigmoid/SoftPlus/
          Pow/ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/
          Maximum/Minimum/Divide/Subtract]*[0,3]
                |
            [quant_out]*  
                |      
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_convtranspose_post_ops_fusion)
        .set_priority(10.5f)
        .set_kind(impl::partition_kind::quantized_convtranspose_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
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

                    pm::pb_op_t *pconvtranspose
                            = pgraph->append_op(impl::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "conv");

                    // Optional bias_add
                    auto popt_bias_graph
                            = std::make_shared<pb_graph_t>("poptional_bias");
                    pm::pb_op_t *pbias = popt_graph->append_op(
                            impl::op_kind::BiasAdd, "pbias");
                    pbias->append_decision_function(
                            check_producer_input_num<2>);
                    popt_bias_graph->create_input_port(0, pbias, 0);
                    popt_bias_graph->create_output_port(0, pbias, 0);
                    auto popt_bias = pgraph->append_optional(popt_bias_graph,
                            in_edges_t {in_edge(0, pconvtranspose, 0)},
                            "popt_bias");

                    auto pint8_binary_graph = std::make_shared<pb_graph_t>(
                            "pint8_binary_graph");
                    pm::pb_op_t *pdequant_binary
                            = pint8_binary_graph->append_op(
                                    impl::op_kind::Dequantize, "dequant");
                    pm::pb_op_t *pbinary
                            = pint8_binary_graph->append_alternation(
                                    {impl::op_kind::Add,
                                            impl::op_kind::Multiply,
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

                    auto postop_graph
                            = std::make_shared<pb_graph_t>("postop_graph");
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Elu, impl::op_kind::Exp,
                                    impl::op_kind::GELU,
                                    impl::op_kind::HardSwish,
                                    impl::op_kind::LeakyReLU,
                                    impl::op_kind::Log, impl::op_kind::Sigmoid,
                                    impl::op_kind::SoftPlus, impl::op_kind::Pow,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sqrt, impl::op_kind::Square,
                                    impl::op_kind::Tanh, impl::op_kind::Add,
                                    impl::op_kind::Multiply,
                                    impl::op_kind::Maximum,
                                    impl::op_kind::Minimum,
                                    impl::op_kind::Divide,
                                    impl::op_kind::Subtract},
                            "postop");
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
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::quantized_convtranspose_fusion);
                    fused_op->set_attr<std::string>(op_attr::backend, "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, convtranspose_post_ops_fusion)
        .set_priority(10.4f)
        .set_kind(impl::partition_kind::convtranspose_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    // convtranspose
                    auto convtranspose = pgraph->append_op(
                            impl::op_kind::ConvTranspose, "convtranspose");

                    // optional biasadd
                    auto biasadd_subgraph
                            = std::make_shared<pb_graph_t>("biasadd_subgraph");
                    auto biasadd = biasadd_subgraph->append_op(
                            impl::op_kind::BiasAdd, "biasadd");
                    biasadd->append_decision_function(
                            check_producer_input_num<2>);
                    biasadd_subgraph->create_input_port(0, biasadd, 0);
                    biasadd_subgraph->create_output_port(0, biasadd, 0);
                    auto optional_biasadd
                            = pgraph->append_optional(biasadd_subgraph,
                                    {in_edge(0, convtranspose, 0)}, "biasadd");

                    auto post_ops = std::make_shared<pb_graph_t>("post_ops");
                    auto alternation = post_ops->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Add,
                                    impl::op_kind::Clamp, impl::op_kind::Divide,
                                    impl::op_kind::Elu, impl::op_kind::Exp,
                                    impl::op_kind::GELU,
                                    impl::op_kind::HardSwish,
                                    impl::op_kind::Log, impl::op_kind::Maximum,
                                    impl::op_kind::Minimum,
                                    impl::op_kind::Multiply, impl::op_kind::Pow,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sigmoid,
                                    impl::op_kind::SoftPlus,
                                    impl::op_kind::Sqrt, impl::op_kind::Square,
                                    impl::op_kind::Subtract,
                                    impl::op_kind::Tanh},
                            "alternation");
                    post_ops->create_input_port(0, alternation, 0);
                    post_ops->create_output_port(0, alternation, 0);

                    auto repetition_post_ops = pgraph->append_repetition(
                            post_ops, {0, 0}, 0, MAX_REPETITION,
                            {in_edge(0, optional_biasadd, 0)},
                            "repetition_post_ops");
                    pgraph->create_input_port(0, convtranspose, 0);
                    pgraph->create_output_port(0, repetition_post_ops, 0);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::convtranspose_post_ops_fusion);
                    fused_op->set_attr<std::string>(op_attr::backend, "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
