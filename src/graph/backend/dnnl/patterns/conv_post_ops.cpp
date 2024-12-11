/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include "graph/backend/dnnl/kernels/primitive_base/conv.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

#include "graph/utils/pm/pbuilder.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace {

bool check_s8_weight(op_t *op) {
    const op_kind_t kind = op->get_kind();
    if (kind != graph::op_kind::Dequantize) return true;
    return (!is_int8_quantization(op)
            || check_input_dtype<graph::data_type::s8>(op));
}

bool check_f8_conv_with_no_scale_and_zp(op_t *op) {
    const op_kind_t kind = op->get_kind();
    if (kind != graph::op_kind::Convolution) return true;

    for (const size_t index : {0, 1}) {
        auto &parent_op = op->get_input_value(index)->get_producer();
        if (parent_op.get_kind() != graph::op_kind::Dequantize) return true;
        if (is_f8_quantization(&parent_op)
                && !check_quant_with_no_effect(&parent_op))
            return false;
    }
    return true;
}

} // namespace

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(conv_post_ops)

// Conv: Currently DNNL backend doesn't support conv + depthwise conv
// post-op fusion on GPU, while CPU supports. Check engine_kind == cpu
// before matching
/*
              \   /
              conv
                |
             [bias]*
                |
    [unary/binary]*[0,MAX_REPETITION)
                |
          depthwise_conv
                |
             [bias]*
                |
    [unary/binary]*[0,MAX_REPETITION)
                |

  (TODO:Jiexin)limitation: For now, this pattern use decision function 
  check_conv_weight_size to avoid mismatching, but this implicitly requires user 
  to provide conv weight size when creating logical tensors if they want the 
  computation graph to benefit from this pattern.
*/
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(
        dnnl, fp_conv_postops_depthwise_postops_cpu)
        .set_priority(10.3f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *pconv
                            = pgraph->append_op(graph::op_kind::Convolution);
                    pconv->append_decision_function(check_conv_weight_size<1>);
                    // Optional bias_add
                    auto popt_bias1 = optional_bias_add(pgraph, pconv, false);

                    auto alt_graph1 = std::make_shared<pb_graph_t>();
                    auto palt1 = alt_graph1->append_alternation(
                            get_unary_binary_ops());
                    palt1->allow_internal_inputs();
                    alt_graph1->create_input_port(0, palt1, 0);
                    alt_graph1->create_output_port(0, palt1, 0);

                    auto prep1 = pgraph->append_repetition(alt_graph1, {0, 0},
                            0, MAX_REPETITION,
                            in_edges_t {in_edge(0, popt_bias1, 0)});

                    pm::pb_op_t *depthwise
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, prep1, 0)});
                    depthwise->append_decision_function(
                            check_conv_weight_size<3>);
                    // Optional bias_add
                    auto popt_bias2
                            = optional_bias_add(pgraph, depthwise, false);

                    auto alt_graph2 = std::make_shared<pb_graph_t>();
                    auto palt2 = alt_graph2->append_alternation(
                            get_unary_binary_ops());
                    palt2->allow_internal_inputs();
                    alt_graph2->create_input_port(0, palt2, 0);
                    alt_graph2->create_output_port(0, palt2, 0);

                    pgraph->append_repetition(alt_graph2, {0, 0}, 0,
                            MAX_REPETITION,
                            in_edges_t {in_edge(0, popt_bias2, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<float_conv_fwd>()};
            return kernels;
        });
#endif
/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        \_____       _____/
               conv
                |
              [bias]*
                |  dequant_add
                |  /
               add
                |
[unary/binary]*[0,MAX_REPETITION)
                |
            quant_out
                |
*/
/*
Conv: Currently DNNL Backend doesn't support below
features on GPU:
1. Post-sum with zero points
*/
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8x8x8_conv_add_post_ops_cpu)
        .set_priority(10.6f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(is_int8_quantization);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});
                    // Currently oneDNN convolution primitive only supports s8 weight
                    dequant_weight->append_decision_function(check_s8_weight);

                    pm::pb_op_t *pconv
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    pconv->append_decision_function(
                            check_f8_conv_with_no_scale_and_zp);

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    // dequantize(rhs) -> add
                    auto prep = post_quantized_add(pgraph, popt_bias);

                    // quantize
                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<quantized_conv>()};
            return kernels;
        });
#endif

#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8x8x8_conv_add_post_ops_gpu)
        .set_priority(10.6f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(partition_kind_t::quantized_convolution_post_ops)
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
                    // Currently oneDNN convolution primitive only supports s8 weight
                    dequant_weight->append_decision_function(check_s8_weight);

                    pm::pb_op_t *pconv
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    pconv->append_decision_function(
                            check_f8_conv_with_no_scale_and_zp);

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    // dequantize(rhs) -> add
                    auto prep = post_quantized_add(
                            pgraph, popt_bias, /*check_zps*/ true);

                    // quantize
                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<quantized_conv>()};
            return kernels;
        });
#endif

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        \_____       _____/
               conv
                |
              [bias]*
                |
[unary/binary]*[0,MAX_REPETITION)
                |
            [quant_out]*
                |
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8x8x_conv_post_ops)
        .set_priority(10.5f)
        .set_kind(partition_kind_t::quantized_convolution_post_ops)
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
                    // Currently oneDNN convolution primitive only supports s8 weight
                    dequant_weight->append_decision_function(check_s8_weight);

                    pm::pb_op_t *pconv
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    pconv->append_decision_function(
                            check_f8_conv_with_no_scale_and_zp);

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    // post ops
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
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<quantized_conv>()};
            return kernels;
        });

/*
                     staticreshape
                           |
                      quant_weight
                           |
                     dequant_weight
        |                  |
   dequant_data      staticreshape
        \_____       _____/
               conv
                |
              [bias]*
                |           
[unary/binary]*[0,MAX_REPETITION)
                |
            [quant_out]*
                | 

This pattern is defined for itex originally, the conv here is a depthwise 
convolution.
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8s8x_conv_reshape_post_ops)
        .set_priority(10.6f)
        .set_kind(partition_kind_t::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    dequant_data->append_decision_function(
                            is_int8_quantization);

                    pm::pb_op_t *weight_reshape1
                            = pgraph->append_op(graph::op_kind::StaticReshape);

                    pm::pb_op_t *quant_weight = pgraph->append_op(
                            graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, weight_reshape1, 0)});

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);
                    pm::pb_op_t *weight_reshape2
                            = pgraph->append_op(graph::op_kind::StaticReshape,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});

                    pm::pb_op_t *pconv
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {
                                            in_edge(0, dequant_data, 0),
                                            in_edge(1, weight_reshape2, 0),
                                    });
                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    // post ops
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
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<quantized_conv>()};
            return kernels;
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
              [bias]*  dequant_other
                |          /
                |  typecast_other
                |  /
               Add
                |
[unary/binary]*[0,MAX_REPETITION)
                |
            typecast_out
                |
            quant_out

Conv: Currently DNNL Backend doesn't support below
features on GPU:
1. Post-sum with zero points
*/
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8s8x_tc_conv_add_post_ops_cpu)
        .set_priority(10.6f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    dequant_data->append_decision_function(
                            is_int8_quantization);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    // TODO: remove bf16 datatype check when all float point
                    // datatypes are enabled
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
                    // TODO: remove bf16 datatype check when all float point
                    // datatypes are enabled
                    typecast_weight->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    pm::pb_op_t *conv
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias
                    auto popt_bias = optional_bias_add(pgraph, conv, true);

                    // post add with dequant->typecast
                    pm::pb_op_t *pdequant_add
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pdequant_add->append_decision_function(
                            is_int8_quantization);
                    pm::pb_op_t *typecast_add
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, pdequant_add, 0)});
                    pm::pb_op_t *padd = pgraph->append_op(graph::op_kind::Add,
                            in_edges_t {in_edge(0, popt_bias, 0),
                                    in_edge(1, typecast_add, 0)});

                    //post ops
                    auto postop_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            get_unary_binary_ops());
                    postop_graph->create_input_port(0, pop, 0);
                    postop_graph->create_input_port(1, pop, 1);
                    postop_graph->create_output_port(0, pop, 0);

                    auto prep = pgraph->append_repetition(postop_graph, {0, 0},
                            0, MAX_REPETITION,
                            in_edges_t {in_edge(0, padd, 0)});

                    pm::pb_op_t *ptc_out
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, prep, 0)});
                    // TODO: remove bf16 datatype check when all float point
                    // datatypes are enabled
                    ptc_out->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    ptc_out->append_decision_function(
                            check_output_dtype<graph::data_type::f32>);
                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, ptc_out, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<quantized_conv>()};
            return kernels;
        });
#endif

#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8s8x_tc_conv_add_post_ops_gpu)
        .set_priority(10.6f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(graph::partition_kind_t::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    dequant_data->append_decision_function(
                            is_int8_quantization);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    // TODO: remove bf16 datatype check when all float point
                    // datatypes are enabled
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
                    // TODO: remove bf16 datatype check when all float point
                    // datatypes are enabled
                    typecast_weight->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    pm::pb_op_t *conv
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias
                    auto popt_bias = optional_bias_add(pgraph, conv, true);

                    // post add with dequant->typecast
                    pm::pb_op_t *pdequant_add
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pdequant_add->append_decision_function(check_zps_values<0>);
                    pdequant_add->append_decision_function(
                            is_int8_quantization);
                    pm::pb_op_t *typecast_add
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, pdequant_add, 0)});
                    pm::pb_op_t *padd = pgraph->append_op(graph::op_kind::Add,
                            in_edges_t {in_edge(0, popt_bias, 0),
                                    in_edge(1, typecast_add, 0)});

                    //post ops
                    auto postop_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            get_unary_binary_ops());
                    postop_graph->create_input_port(0, pop, 0);
                    postop_graph->create_input_port(1, pop, 1);
                    postop_graph->create_output_port(0, pop, 0);

                    auto prep = pgraph->append_repetition(postop_graph, {0, 0},
                            0, MAX_REPETITION,
                            in_edges_t {in_edge(0, padd, 0)});

                    pm::pb_op_t *ptc_out
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, prep, 0)});
                    // TODO: remove bf16 datatype check when all float point
                    // datatypes are enabled
                    ptc_out->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    ptc_out->append_decision_function(
                            check_output_dtype<graph::data_type::f32>);
                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, ptc_out, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<quantized_conv>()};
            return kernels;
        });
#endif
/*
TODO(Jiexin): Currently, we found conv+relu+multiply/divide/subtract lacks 
optimization in oneDNN post-ops fusion, so we don't plan to extend the post-op 
chain of this pattern, we will extend it when corresponding optimization is 
enabled.
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        |                  |
    typecast             typecast
        \_____       _____/
               conv
                | [typecast_bias]*
                |   /
              [bias]*
                |
            [unary/binary]*[0,MAX_REPETITION)
                |
    [typecast_out -> quant_out]*
                |
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8s8x_tc_conv_post_ops)
        .set_priority(10.5f)
        .set_kind(partition_kind_t::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    dequant_data->append_decision_function(
                            is_int8_quantization);
                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquant
                            = popt_graph->append_op(graph::op_kind::Quantize);
                    pquant->append_decision_function(check_if_constant_weight);
                    pquant->append_decision_function(is_int8_quantization);
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph);
                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt, 0)});
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);
                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    // TODO: remove bf16 datatype check when all float point
                    // datatypes are enabled
                    typecast_data->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);
                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    // TODO: remove bf16 datatype check when all float point
                    // datatypes are enabled
                    typecast_weight->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);
                    pm::pb_op_t *convolution
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});
                    // Optional bias_add
                    auto popt_bias
                            = optional_bias_add(pgraph, convolution, true);
                    // post ops
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

                    // Optional typecast_out + quant_out
                    auto popt_tcout_qout_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *ptc_out = popt_tcout_qout_graph->append_op(
                            graph::op_kind::TypeCast);
                    // TODO: remove bf16 datatype check when all float point
                    // datatypes are enabled
                    ptc_out->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    ptc_out->append_decision_function(
                            check_output_dtype<graph::data_type::f32>);
                    pm::pb_op_t *pquant_out = popt_tcout_qout_graph->append_op(
                            graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, ptc_out, 0)});
                    popt_tcout_qout_graph->create_input_port(0, ptc_out, 0);
                    popt_tcout_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_tcout_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<quantized_conv>()};
            return kernels;
        });

/*
              \   /
              conv
                |
             [bias]*
                |
              [BN]*
                |
[unary/binary]*[0,MAX_REPETITION)
                |
           [TypeCast]*
                |
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, fp_conv_post_ops)
        .set_priority(9.8f)
        .set_kind(partition_kind_t::convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *conv
                            = pgraph->append_op(graph::op_kind::Convolution);

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, conv, false);
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

                    auto prep = pgraph->append_repetition(alt_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)});

                    // Optional typecast
                    auto popt_tc_graph = std::make_shared<pb_graph_t>();
                    auto ptc = popt_tc_graph->append_op(
                            graph::op_kind::TypeCast);
                    // TODO: remove bf16 datatype check when all float point
                    // datatypes are enabled
                    ptc->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    ptc->append_decision_function(
                            check_output_dtype<graph::data_type::f32>);
                    popt_tc_graph->create_input_port(0, ptc, 0);
                    popt_tc_graph->create_output_port(0, ptc, 0);
                    pgraph->append_optional(
                            popt_tc_graph, {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<float_conv_fwd>()};
            return kernels;
        });

/*
                  wildcard
              \   /      \
      conv_bwd_weight  biasadd_bwd
                |          |
*/
#if BUILD_TRAINING
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, fp_conv_bwd_weights_bias)
        .set_enable(false)
        .set_kind(partition_kind_t::convolution_backward_post_ops)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *wildcard
                            = pgraph->append_op(graph::op_kind::Wildcard);
                    graph::utils::pm::pb_op_t *p_conv_backward_weights
                            = pgraph->append_op(
                                    graph::op_kind::ConvolutionBackwardWeights,
                                    in_edges_t {in_edge(1, wildcard, 0)});
                    p_conv_backward_weights->append_decision_function(
                            check_input_num<2>);
                    pgraph->append_op(graph::op_kind::BiasAddBackward,
                            in_edges_t {in_edge(0, wildcard, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels
                    = {std::make_shared<conv_bwd_weights_t>()};
            return kernels;
        });
#endif

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
