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

#include "graph/backend/dnnl/kernels/convtranspose.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

bool check_scales_equal_to_1(op_t *op) {
    auto scales = op->get_attr<std::vector<float>>(op_attr::scales);
    return std::all_of(scales.begin(), scales.end(),
            [](float val) { return val == 1.0f; });
}

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(convtranspose_fusion)

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
         \              /
           convtranspose
                |
              [bias]*
                |
        [ Abs/Clamp/Elu/GELU/Log/Sigmoid/SoftPlus/
          ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/
          Maximum/Minimum/Divide/Subtract]*[0,3]
                |
            [quant_out]*  
                |      
*/
/*
ConvTranspose: Currently DNNL Backend doesn't support below
features on GPU:
1. ConvTranspose with per_channel output scale
2. ConvTranspose with per_tensor output scale != 1
3. ConvTranspose with zero points
While CPU supports.
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(
        dnnl, int8_convtranspose_post_ops_fusion_cpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::quantized_convtranspose_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});
                    // Currently oneDNN ConvTranspose primitive only supports s8 weight
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);

                    pm::pb_op_t *pconvtranspose
                            = pgraph->append_op(graph::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});

                    // Optional bias_add
                    auto popt_bias
                            = optional_bias_add(pgraph, pconvtranspose, false);

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
                    pm::pb_op_t *pquant_out
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(
                            popt_qout_graph, in_edges_t {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_convtranspose>();
        });

/*
ConvTranspose: Currently DNNL Backend doesn't support below
features on GPU:
1. ConvTranspose with per_channel output scale
2. ConvTranspose with per_tensor output scale != 1
3. ConvTranspose with zero points
While CPU supports.
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(
        dnnl, int8_convtranspose_post_ops_fusion_gpu)
        .set_priority(10.5f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(partition_kind_t::quantized_convtranspose_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    dequant_data->append_decision_function(check_zps_values<0>);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(check_zps_values<0>);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);
                    dequant_weight->append_decision_function(
                            check_zps_values<0>);

                    pm::pb_op_t *pconvtranspose
                            = pgraph->append_op(graph::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});

                    // Optional bias_add
                    auto popt_bias
                            = optional_bias_add(pgraph, pconvtranspose, false);

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
                    pm::pb_op_t *pquant_out
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant_out->append_decision_function(
                            check_qtype_equal_to_per_tensor);
                    pquant_out->append_decision_function(
                            check_scales_equal_to_1);
                    pquant_out->append_decision_function(check_zps_values<0>);
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(
                            popt_qout_graph, in_edges_t {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_convtranspose>();
        });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
         \              /
           convtranspose
                |
              [bias]*  dequant_add
                |     /
               add
                |
        [ Abs/Clamp/Elu/GELU/Log/Sigmoid/SoftPlus/
          ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/
          Maximum/Minimum/Divide/Subtract]*[0,3]
                |
            quant_out 
                |      
*/
/*
ConvTranspose: Currently DNNL Backend doesn't support below
features on GPU:
1. ConvTranspose with per_channel output scale
2. ConvTranspose with per_tensor output scale != 1
3. ConvTranspose with zero points
4. Post-sum with zero points
While CPU supports.
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(
        dnnl, int8_convtranspose_add_post_ops_fusion_cpu)
        .set_priority(10.6f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::quantized_convtranspose_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});
                    // Currently oneDNN ConvTranspose primitive only supports s8 weight
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);

                    pm::pb_op_t *pconvtranspose
                            = pgraph->append_op(graph::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});

                    // Optional bias_add
                    auto popt_bias
                            = optional_bias_add(pgraph, pconvtranspose, false);

                    // dequantize(rhs) -> add
                    auto prep = post_quantized_add(pgraph, popt_bias);

                    // quantize
                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_convtranspose>();
        });

/*
ConvTranspose: Currently DNNL Backend doesn't support below
features on GPU:
1. ConvTranspose with per_channel output scale
2. ConvTranspose with per_tensor output scale != 1
3. ConvTranspose with zero points
4. Post-sum with zero points
While CPU supports.
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(
        dnnl, int8_convtranspose_add_post_ops_fusion_gpu)
        .set_priority(10.6f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(partition_kind_t::quantized_convtranspose_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    dequant_data->append_decision_function(check_zps_values<0>);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(check_zps_values<0>);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);
                    dequant_weight->append_decision_function(
                            check_zps_values<0>);

                    pm::pb_op_t *pconvtranspose
                            = pgraph->append_op(graph::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});

                    // Optional bias_add
                    auto popt_bias
                            = optional_bias_add(pgraph, pconvtranspose, false);

                    // dequantize(rhs) -> add
                    auto prep = post_quantized_add(
                            pgraph, popt_bias, /*check_zps*/ true);

                    // quantize
                    pm::pb_op_t *pquant_out
                            = pgraph->append_op(graph::op_kind::Quantize,
                                    in_edges_t {in_edge(0, prep, 0)});
                    pquant_out->append_decision_function(
                            check_qtype_equal_to_per_tensor);
                    pquant_out->append_decision_function(
                            check_scales_equal_to_1);
                    pquant_out->append_decision_function(check_zps_values<0>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_convtranspose>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, convtranspose_post_ops_fusion)
        .set_priority(10.4f)
        .set_kind(partition_kind_t::convtranspose_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    // convtranspose
                    auto convtranspose
                            = pgraph->append_op(graph::op_kind::ConvTranspose);

                    // optional biasadd
                    auto optional_biasadd
                            = optional_bias_add(pgraph, convtranspose, false);

                    auto post_ops = std::make_shared<pb_graph_t>();
                    auto alternation = post_ops->append_alternation(
                            get_unary_binary_ops());
                    alternation->allow_internal_inputs();
                    post_ops->create_input_port(0, alternation, 0);
                    post_ops->create_output_port(0, alternation, 0);

                    auto repetition_post_ops = pgraph->append_repetition(
                            post_ops, {0, 0}, 0, MAX_REPETITION,
                            {in_edge(0, optional_biasadd, 0)});
                    pgraph->create_input_port(0, convtranspose, 0);
                    pgraph->create_output_port(0, repetition_post_ops, 0);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_convtranspose_fwd>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
