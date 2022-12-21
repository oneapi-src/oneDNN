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

#include "graph/backend/dnnl/kernels/conv.hpp"
#include "graph/backend/dnnl/kernels/large_partition.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/transformation_pattern.hpp"
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

namespace {
template <bool GROUPED>
bool check_grouped(op_t *op) {
    if (GROUPED) {
        return op->has_attr(op_attr::groups)
                && op->get_attr<int64_t>(op_attr::groups) > 1;
    } else {
        return !op->has_attr(op_attr::groups)
                || op->get_attr<int64_t>(op_attr::groups) <= 1;
    }
}

// Block creators used to construct large patterns
pm::pb_op_t *conv_bias(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool grouped = false, bool use_biasadd = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op_t *conv
            = pgraph->append_op(graph::op_kind::Convolution, in_edges);
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                graph::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    return conv_bias_dst;
};

pm::pb_op_t *conv_bias_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool grouped = false, bool use_biasadd = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }

    pm::pb_op_t *conv
            = pgraph->append_op(graph::op_kind::Convolution, in_edges);
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                graph::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op_t *relu = pgraph->append_op(
            graph::op_kind::ReLU, in_edges_t {in_edge(0, conv_bias_dst, 0)});
    return relu;
};

pm::pb_op_t *conv_bias_add_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, pm::pb_op_t *post_src, bool grouped = false,
        bool use_biasadd = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op_t *conv
            = pgraph->append_op(graph::op_kind::Convolution, in_edges);
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                graph::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);

    in_edges_t add_in_edges = in_edges_t {in_edge(0, conv_bias_dst, 0)};
    if (post_src) { add_in_edges.emplace_back(in_edge(1, post_src, 0)); }
    pm::pb_op_t *add = pgraph->append_op(graph::op_kind::Add, add_in_edges);

    pm::pb_op_t *relu = pgraph->append_op(
            graph::op_kind::ReLU, in_edges_t {in_edge(0, add, 0)});
    return relu;
};

pm::pb_op_t *int8_conv_bias(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op_t *dequant_src
            = pgraph->append_op(graph::op_kind::Dequantize, in_edges);
    pm::pb_op_t *dequant_wei = nullptr;
    if (use_quant_wei) {
        pm::pb_op_t *quant_wei = pgraph->append_op(graph::op_kind::Quantize);
        dequant_wei = pgraph->append_op(graph::op_kind::Dequantize,
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_op(graph::op_kind::Dequantize);
    }
    pm::pb_op_t *conv = pgraph->append_op(graph::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                graph::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op_t *quant_dst = pgraph->append_op(graph::op_kind::Quantize,
            in_edges_t {in_edge(0, conv_bias_dst, 0)});
    return quant_dst;
};

pm::pb_op_t *int8_conv_bias_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op_t *dequant_src
            = pgraph->append_op(graph::op_kind::Dequantize, in_edges);
    pm::pb_op_t *dequant_wei = nullptr;
    if (use_quant_wei) {
        pm::pb_op_t *quant_wei = pgraph->append_op(graph::op_kind::Quantize);
        dequant_wei = pgraph->append_op(graph::op_kind::Dequantize,
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_op(graph::op_kind::Dequantize);
    }
    pm::pb_op_t *conv = pgraph->append_op(graph::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                graph::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op_t *relu = pgraph->append_op(
            graph::op_kind::ReLU, in_edges_t {in_edge(0, conv_bias_dst, 0)});
    pm::pb_op_t *quant_dst = pgraph->append_op(
            graph::op_kind::Quantize, in_edges_t {in_edge(0, relu, 0)});
    return quant_dst;
};

pm::pb_op_t *int8_conv_bias_add_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, pm::pb_op_t *post_src, bool grouped = false,
        bool use_biasadd = false, bool use_quant_wei = false,
        bool f32_output = false) {
    in_edges_t in_edges, post_src_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    if (post_src) { post_src_edges = in_edges_t {in_edge(0, post_src, 0)}; }
    pm::pb_op_t *dequant_src
            = pgraph->append_op(graph::op_kind::Dequantize, in_edges);
    pm::pb_op_t *dequant_wei = nullptr;
    if (use_quant_wei) {
        pm::pb_op_t *quant_wei = pgraph->append_op(graph::op_kind::Quantize);
        dequant_wei = pgraph->append_op(graph::op_kind::Dequantize,
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_op(graph::op_kind::Dequantize);
    }
    pm::pb_op_t *dequant_other
            = pgraph->append_op(graph::op_kind::Dequantize, post_src_edges);
    pm::pb_op_t *conv = pgraph->append_op(graph::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                graph::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op_t *add = pgraph->append_op(graph::op_kind::Add,
            in_edges_t {in_edge(0, conv_bias_dst, 0),
                    in_edge(1, dequant_other, 0)});
    pm::pb_op_t *relu = pgraph->append_op(
            graph::op_kind::ReLU, in_edges_t {in_edge(0, add, 0)});
    if (f32_output) {
        return relu;
    } else {
        pm::pb_op_t *quant_dst = pgraph->append_op(
                graph::op_kind::Quantize, in_edges_t {in_edge(0, relu, 0)});
        return quant_dst;
    }
};

// The F(x)+x basic residual block
pm::pb_op_t *int8_identical_basic_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op_t *quant_dst0
            = int8_conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *quant_dst1 = int8_conv_bias_add_relu(
            pgraph, quant_dst0, input, grouped, use_biasadd);
    return quant_dst1;
};

// The F(x)+G(x) basic residual block
pm::pb_op_t *int8_convolutional_basic_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op_t *quant_dst0
            = int8_conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *quant_dst1
            = int8_conv_bias(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *quant_dst2 = int8_conv_bias_add_relu(
            pgraph, quant_dst0, quant_dst1, grouped, use_biasadd);
    return quant_dst2;
};

// The F(x)+x bottleneck residual block
pm::pb_op_t *int8_identical_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false, bool f32_output = false) {
    pm::pb_op_t *quant_dst0 = int8_conv_bias_relu(
            pgraph, input, false, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst1 = int8_conv_bias_relu(
            pgraph, quant_dst0, grouped, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst2 = int8_conv_bias_add_relu(pgraph, quant_dst1, input,
            false, use_biasadd, use_quant_wei, f32_output);
    return quant_dst2;
};

// The F(x)+G(x) bottleneck residual block
pm::pb_op_t *int8_convolutional_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    pm::pb_op_t *quant_dst0 = int8_conv_bias_relu(
            pgraph, input, false, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst1 = int8_conv_bias_relu(
            pgraph, quant_dst0, grouped, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst2
            = int8_conv_bias(pgraph, input, false, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst3 = int8_conv_bias_add_relu(
            pgraph, quant_dst1, quant_dst2, false, use_biasadd, use_quant_wei);
    return quant_dst3;
};

pm::pb_op_t *int8_convolutional_bottleneck_resblock_v2(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    pm::pb_op_t *quant_dst0 = int8_conv_bias_relu(
            pgraph, input, grouped, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst1 = int8_conv_bias_relu(
            pgraph, quant_dst0, grouped, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst2 = int8_conv_bias(
            pgraph, quant_dst1, grouped, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst3 = int8_conv_bias_add_relu(
            pgraph, input, quant_dst2, grouped, use_biasadd, use_quant_wei);
    return quant_dst3;
};

pm::pb_op_t *convolutional_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op_t *dst0 = conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *dst1 = conv_bias_relu(pgraph, dst0, grouped, use_biasadd);
    pm::pb_op_t *dst2 = conv_bias(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *dst3
            = conv_bias_add_relu(pgraph, dst1, dst2, grouped, use_biasadd);
    return dst3;
};

pm::pb_op_t *identical_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op_t *dst0 = conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *dst1 = conv_bias_relu(pgraph, dst0, grouped, use_biasadd);
    pm::pb_op_t *dst2
            = conv_bias_add_relu(pgraph, dst1, input, grouped, use_biasadd);
    return dst2;
};

} // namespace

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(conv_fusion)

// Conv: Currently DNNL backend doesn't support conv + depthwise conv
// post-op fusion on GPU, while CPU supports. Check engine_kind == cpu
// before matching
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, conv_depthwise_fusion_cpu)
        .set_priority(10.2f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *conv
                            = pgraph->append_op(graph::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op_t *depthwise
                            = pgraph->append_op(graph::op_kind::Convolution,
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
          ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/Maximum/Minimum/
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
        .set_kind(partition_kind_t::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            graph::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            graph::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op_t *pconv
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "conv");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    auto pint8_add_graph
                            = std::make_shared<pb_graph_t>("pint8_add_graph");
                    pm::pb_op_t *pdequant_add = pint8_add_graph->append_op(
                            graph::op_kind::Dequantize, "dequant");
                    pm::pb_op_t *padd = pint8_add_graph->append_op(
                            graph::op_kind::Add,
                            in_edges_t {in_edge(1, pdequant_add, 0)}, "padd");
                    pint8_add_graph->create_input_port(0, padd, 0);
                    pint8_add_graph->create_input_port(1, pdequant_add, 0);
                    pint8_add_graph->create_output_port(0, padd, 0);

                    // unary + binary post ops exclude add
                    auto postop_graph
                            = std::make_shared<pb_graph_t>("postops_graph");
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            {
                                    graph::op_kind::Abs,
                                    graph::op_kind::Clamp,
                                    graph::op_kind::Elu,
                                    graph::op_kind::Exp,
                                    graph::op_kind::GELU,
                                    graph::op_kind::HardSwish,
                                    graph::op_kind::LeakyReLU,
                                    graph::op_kind::Log,
                                    graph::op_kind::Mish,
                                    graph::op_kind::Sigmoid,
                                    graph::op_kind::SoftPlus,
                                    graph::op_kind::ReLU,
                                    graph::op_kind::Round,
                                    graph::op_kind::Sqrt,
                                    graph::op_kind::Square,
                                    graph::op_kind::Tanh,
                                    graph::op_kind::Multiply,
                                    graph::op_kind::Maximum,
                                    graph::op_kind::Minimum,
                                    graph::op_kind::Divide,
                                    graph::op_kind::Subtract,
                            },
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
                            graph::op_kind::Quantize, "pquant_out");
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
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_conv_post_ops_int8_add_fusion_gpu)
        .set_priority(10.6f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(partition_kind_t::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            graph::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            graph::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);

                    pm::pb_op_t *pconv
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "conv");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    auto pint8_add_graph
                            = std::make_shared<pb_graph_t>("pint8_add_graph");
                    pm::pb_op_t *pdequant_add = pint8_add_graph->append_op(
                            graph::op_kind::Dequantize, "dequant");
                    pdequant_add->append_decision_function(check_zps_values<0>);
                    pm::pb_op_t *padd = pint8_add_graph->append_op(
                            graph::op_kind::Add,
                            in_edges_t {in_edge(1, pdequant_add, 0)}, "padd");
                    pint8_add_graph->create_input_port(0, padd, 0);
                    pint8_add_graph->create_input_port(1, pdequant_add, 0);
                    pint8_add_graph->create_output_port(0, padd, 0);

                    // unary + binary post ops exclude add
                    auto postop_graph
                            = std::make_shared<pb_graph_t>("postops_graph");
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            {
                                    graph::op_kind::Abs,
                                    graph::op_kind::Clamp,
                                    graph::op_kind::Elu,
                                    graph::op_kind::Exp,
                                    graph::op_kind::GELU,
                                    graph::op_kind::HardSwish,
                                    graph::op_kind::LeakyReLU,
                                    graph::op_kind::Log,
                                    graph::op_kind::Mish,
                                    graph::op_kind::Sigmoid,
                                    graph::op_kind::SoftPlus,
                                    graph::op_kind::ReLU,
                                    graph::op_kind::Round,
                                    graph::op_kind::Sqrt,
                                    graph::op_kind::Square,
                                    graph::op_kind::Tanh,
                                    graph::op_kind::Multiply,
                                    graph::op_kind::Maximum,
                                    graph::op_kind::Minimum,
                                    graph::op_kind::Divide,
                                    graph::op_kind::Subtract,
                            },
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
                            graph::op_kind::Quantize, "pquant_out");
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
          ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/Maximum/Minimum/
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
        .set_kind(partition_kind_t::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            graph::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            graph::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op_t *pconv
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "conv");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    auto pint8_binary_graph = std::make_shared<pb_graph_t>(
                            "pint8_binary_graph");
                    pm::pb_op_t *pdequant_binary
                            = pint8_binary_graph->append_op(
                                    graph::op_kind::Dequantize, "dequant");
                    pm::pb_op_t *pbinary
                            = pint8_binary_graph->append_alternation(
                                    {graph::op_kind::Multiply,
                                            graph::op_kind::Maximum,
                                            graph::op_kind::Minimum,
                                            graph::op_kind::Divide,
                                            graph::op_kind::Subtract},
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
                            graph::op_kind::Quantize, "pquant_out");
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
        .set_kind(partition_kind_t::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            graph::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            graph::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);

                    pm::pb_op_t *pconv
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "conv");

                    // Optional bias_add
                    auto popt_bias = optional_bias_add(pgraph, pconv, false);

                    auto pint8_binary_graph = std::make_shared<pb_graph_t>(
                            "pint8_binary_graph");
                    pm::pb_op_t *pdequant_binary
                            = pint8_binary_graph->append_op(
                                    graph::op_kind::Dequantize, "dequant");
                    pdequant_binary->append_decision_function(
                            check_zps_values<0>);
                    pm::pb_op_t *pbinary
                            = pint8_binary_graph->append_alternation(
                                    {graph::op_kind::Multiply,
                                            graph::op_kind::Maximum,
                                            graph::op_kind::Minimum,
                                            graph::op_kind::Divide,
                                            graph::op_kind::Subtract},
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
                            graph::op_kind::Quantize, "pquant_out");
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
        .set_kind(partition_kind_t::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            graph::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            graph::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);
                    pm::pb_op_t *convolution
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias_add
                    auto popt_bias
                            = optional_bias_add(pgraph, convolution, true);

                    // optional GELU
                    auto popt_gelu_graph
                            = std::make_shared<pb_graph_t>("poptional_gelu");
                    pm::pb_op_t *gelu
                            = popt_gelu_graph->append_op(graph::op_kind::GELU);
                    popt_gelu_graph->create_input_port(0, gelu, 0);
                    popt_gelu_graph->create_output_port(0, gelu, 0);
                    auto popt_gelu = pgraph->append_optional(popt_gelu_graph,
                            in_edges_t {in_edge(0, popt_bias, 0)}, "popt_gelu");

                    pm::pb_op_t *typecast_gelu
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, popt_gelu, 0)});
                    typecast_gelu->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    pgraph->append_op(graph::op_kind::Quantize,
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
        .set_kind(partition_kind_t::quantized_convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            graph::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            graph::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);

                    pm::pb_op_t *typecast_data
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);
                    pm::pb_op_t *convolution
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias_add
                    auto popt_bias
                            = optional_bias_add(pgraph, convolution, true);

                    // optional GELU
                    auto popt_gelu_graph
                            = std::make_shared<pb_graph_t>("poptional_gelu");
                    pm::pb_op_t *gelu
                            = popt_gelu_graph->append_op(graph::op_kind::GELU);
                    popt_gelu_graph->create_input_port(0, gelu, 0);
                    popt_gelu_graph->create_output_port(0, gelu, 0);
                    auto popt_gelu = pgraph->append_optional(popt_gelu_graph,
                            in_edges_t {in_edge(0, popt_bias, 0)}, "popt_gelu");

                    pm::pb_op_t *typecast_gelu
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, popt_gelu, 0)});
                    typecast_gelu->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    pgraph->append_op(graph::op_kind::Quantize,
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
        .set_kind(graph::partition_kind_t::quantized_matmul_post_ops)
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
                    auto popt_quant_wei_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_quant_wei_graph->append_op(
                            graph::op_kind::Quantize, "pquant");
                    pquant->append_decision_function(check_if_constant_weight);
                    popt_quant_wei_graph->create_input_port(0, pquant, 0);
                    popt_quant_wei_graph->create_output_port(0, pquant, 0);
                    auto popt_quant_wei = pgraph->append_optional(
                            popt_quant_wei_graph, "popt");

                    pm::pb_op_t *dequant_weight
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, popt_quant_wei, 0)},
                                    "dequant_weight");
                    dequant_weight->append_decision_function(
                            check_input_dtype<graph::data_type::s8>);

                    pm::pb_op_t *typecast_weight
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<graph::data_type::bf16>);

                    pm::pb_op_t *conv
                            = pgraph->append_op(graph::op_kind::Convolution,
                                    in_edges_t {in_edge(0, typecast_data, 0),
                                            in_edge(1, typecast_weight, 0)});

                    // Optional bias
                    auto popt_bias = optional_bias_add(pgraph, conv, true);

                    // post add with dequant->typecast
                    auto padd_graph
                            = std::make_shared<pb_graph_t>("padd_graph");
                    pm::pb_op_t *pdequant_add = padd_graph->append_op(
                            graph::op_kind::Dequantize, "dequant_add");
                    pm::pb_op_t *typecast_add
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, pdequant_add, 0)});
                    pm::pb_op_t *padd = padd_graph->append_op(
                            graph::op_kind::Add,
                            in_edges_t {in_edge(1, typecast_add, 0)}, "padd");
                    padd_graph->create_input_port(0, padd, 0);
                    padd_graph->create_input_port(1, pdequant_add, 0);
                    padd_graph->create_output_port(0, padd, 0);

                    // post binary with typecast
                    auto other_binary_with_tc_graph
                            = std::make_shared<pb_graph_t>(
                                    "pother_binary_with_tc_graph");
                    pm::pb_op_t *typecast_binary = pgraph->append_op(
                            graph::op_kind::TypeCast, "typecast_binary");
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
                            {graph::op_kind::ReLU, graph::op_kind::GELU,
                                    graph::op_kind::Divide,
                                    graph::op_kind::Multiply,
                                    graph::op_kind::Add},
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
                            graph::op_kind::TypeCast, "ptc_out");
                    pm::pb_op_t *pquant_out = popt_qout_graph->append_op(
                            graph::op_kind::Quantize,
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

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_resnet50_stage_1_4_fusion)
        .set_priority(22.f) // high priority to support lz models
        .set_kind(partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    output = int8_identical_bottleneck_resblock(pgraph, output);
                    output = int8_identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_resnet50_stage_2_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_kind(partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_resnet50_stage_3_fusion)
        .set_priority(22.2f) // high priority to support lz models
        .set_kind(partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_resnet34_stage_1_4_fusion)
        .set_priority(22.f) // high priority to support lz models
        .set_kind(partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_identical_basic_resblock(pgraph, nullptr);
                    output = int8_identical_basic_resblock(pgraph, output);
                    output = int8_identical_basic_resblock(pgraph, output);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_resnet34_stage_2_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_kind(partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_basic_resblock(pgraph, nullptr);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_basic_resblock(pgraph, output);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_resnet34_stage_3_fusion)
        .set_priority(22.2f) // high priority to support lz models
        .set_kind(partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_basic_resblock(pgraph, nullptr);
                    // 5 F(x)+x blocks
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_basic_resblock(pgraph, output);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, f32_resnet50_stage_1_4_fusion)
        .set_priority(22.f) // high priority to support itex
        .set_kind(partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    output = identical_bottleneck_resblock(pgraph, output);
                    output = identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    output = identical_bottleneck_resblock(
                            pgraph, output, false, true);
                    output = identical_bottleneck_resblock(
                            pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, f32_resnet50_stage_2_fusion)
        .set_priority(22.1f) // high priority to support itex
        .set_kind(partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, f32_resnet50_stage_3_fusion)
        .set_priority(22.2f) // high priority to support itex
        .set_kind(partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// For itex int8 rn50 only (include the weight quantize into pattern)
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, itex_int8_resnet50_stage_1_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_kind(partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// For itex int8 rn50 only (include the weight quantize into pattern)
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, itex_int8_resnet50_stage_2_fusion)
        .set_priority(22.2f) // high priority to support lz models
        .set_kind(partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// For itex int8 rn50 only (include the weight quantize into pattern)
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, itex_int8_resnet50_stage_3_fusion)
        .set_priority(22.3f) // high priority to support lz models
        .set_kind(partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, itex_int8_resnet50_stage_4_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_kind(partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true, true);
                    output = int8_identical_bottleneck_resblock(pgraph, output,
                            false, true, true, /* f32 output */ true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

/*
              \   /
              conv
                |
        [ Abs/Clamp/Elu/Exp/GELU/HardSwish/Log/Sigmoid/SoftPlus/
          ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/
          Maximum/Minimum/Divide/Subtract]*[0,3] 
                |
            [TypeCast]*
                |
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, conv_post_ops_fusion)
        .set_priority(9.7f)
        .set_kind(partition_kind_t::convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *pconv
                            = pgraph->append_op(graph::op_kind::Convolution);
                    pconv->append_decision_function(check_input_num<2>);

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            graph::op_kind::BatchNormInference, "pbn");
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
                            graph::op_kind::TypeCast, "ptc");
                    ptc->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    ptc->append_decision_function(
                            check_output_dtype<graph::data_type::f32>);
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
          ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/
          Maximum/Minimum/Divide/Subtract]*[0,3] 
                |
           [TypeCast]*
                |
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, conv_bias_post_ops_fusion)
        .set_priority(9.8f)
        .set_kind(partition_kind_t::convolution_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *conv = pgraph->append_op(
                            graph::op_kind::Convolution, "conv");
                    conv->append_decision_function(check_input_num<2>);
                    pm::pb_op_t *biasadd
                            = pgraph->append_op(graph::op_kind::BiasAdd,
                                    in_edges_t {in_edge(0, conv, 0)}, "bias");

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            graph::op_kind::BatchNormInference, "pbn");
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
                            graph::op_kind::TypeCast, "ptc");
                    ptc->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    ptc->append_decision_function(
                            check_output_dtype<graph::data_type::f32>);
                    popt_tc_graph->create_input_port(0, ptc, 0);
                    popt_tc_graph->create_output_port(0, ptc, 0);
                    pgraph->append_optional(
                            popt_tc_graph, {in_edge(0, prep, 0)}, "popt_tc");
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *conv = pgraph->append_op(
                            graph::op_kind::Convolution, "conv");
                    conv->append_decision_function(check_input_num<3>);

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            graph::op_kind::BatchNormInference, "pbn");
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
                            graph::op_kind::TypeCast, "ptc");
                    ptc->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    ptc->append_decision_function(
                            check_output_dtype<graph::data_type::f32>);
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
        .set_kind(partition_kind_t::convolution_backward_post_ops)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *wildcard = pgraph->append_op(
                            graph::op_kind::Wildcard, "pwild");
                    pgraph->append_op(
                            graph::op_kind::ConvolutionBackwardWeights,
                            in_edges_t {in_edge(1, wildcard, 0)});
                    pgraph->append_op(graph::op_kind::BiasAddBackward,
                            in_edges_t {in_edge(0, wildcard, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<conv_bwd_weights_t>();
        });

// ResNeXt101 backbone is the composition of 4 stages, which has 102 conv inside
// it. The convolution's bias can be connected to conv op directly as an
// optional input, or it also can be performed by using a separated biasadd op
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_resnext101_backbone_fusion)
        .set_enable(true)
        .set_priority(23.f) // high priority to support lz models
        .set_kind(partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    // stage 1
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, true);
                    for (size_t i = 0; i < 2; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true);
                    // stage 2
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true);
                    for (size_t i = 0; i < 3; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true);
                    // stage 3
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true);
                    for (size_t i = 0; i < 22; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true);
                    // stage 4
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true);
                    for (size_t i = 0; i < 2; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    // stage 1
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, true, true);
                    for (size_t i = 0; i < 2; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
                    // stage 2
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true, true);
                    for (size_t i = 0; i < 3; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
                    // stage 3
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true, true);
                    for (size_t i = 0; i < 22; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
                    // stage 4
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true, true);
                    for (size_t i = 0; i < 2; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
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
