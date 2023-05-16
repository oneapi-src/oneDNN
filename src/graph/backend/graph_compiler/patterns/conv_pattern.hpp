/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_CONV_PATTERN_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_CONV_PATTERN_HPP

#include <memory>
#include <utility>

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

// conv_bias_relu
/*
         [input]   [filter]
              \     /
            Convolution      [bias]
                 \           /
                BiasAdd (optional)
                   |
                  Relu
                   |
                [output]
*/
pm::pb_node_t *conv_bias_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_node_t *input, bool has_relu = false, bool is_bf16 = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }

    pm::pb_op_t *conv
            = pgraph->append_op(graph::op_kind::Convolution, in_edges);
    conv->append_decision_function(check_conv_attrs);
    if (is_bf16) {
        conv->append_decision_function(
                check_input_dtype<graph::data_type::bf16>);
    } else {
        conv->append_decision_function(
                check_input_dtype<graph::data_type::f32>);
    }

    auto biasadd_subgraph = std::make_shared<pb_graph_t>();
    auto biasadd = biasadd_subgraph->append_op(graph::op_kind::BiasAdd);
    biasadd_subgraph->create_input_port(0, biasadd, 0);
    biasadd_subgraph->create_output_port(0, biasadd, 0);
    auto optional_biasadd = pgraph->append_optional(
            biasadd_subgraph, in_edges_t {in_edge(0, conv, 0)});

    if (has_relu) {
        pm::pb_op_t *relu = pgraph->append_op(graph::op_kind::ReLU,
                in_edges_t {in_edge(0, optional_biasadd, 0)});
        return relu;
    } else {
        return optional_biasadd;
    }
};

std::pair<pm::pb_op_t *, pm::pb_op_t *> conv_bias_relu_subgraph(
        const std::shared_ptr<pb_graph_t> &pgraph, bool is_bf16 = false) {
    pm::pb_op_t *conv = pgraph->append_op(graph::op_kind::Convolution);
    conv->append_decision_function(check_conv_attrs);
    if (is_bf16) {
        conv->append_decision_function(
                check_input_dtype<graph::data_type::bf16>);
    } else {
        conv->append_decision_function(
                check_input_dtype<graph::data_type::f32>);
    }

    auto biasadd_subgraph = std::make_shared<pb_graph_t>();
    auto biasadd = biasadd_subgraph->append_op(graph::op_kind::BiasAdd);
    biasadd_subgraph->create_input_port(0, biasadd, 0);
    biasadd_subgraph->create_output_port(0, biasadd, 0);
    auto optional_biasadd = pgraph->append_optional(
            biasadd_subgraph, in_edges_t {in_edge(0, conv, 0)});

    pm::pb_op_t *relu = pgraph->append_op(
            graph::op_kind::ReLU, in_edges_t {in_edge(0, optional_biasadd, 0)});
    return {conv, relu};
};

// conv_bias_add_relu
/*
         [input]   [filter]
              \     /
            Convolution   [bias]
                 \        /
       (optional) BiasAdd   [other]
                        \   /
                         Add
                          |
                         Relu
                          |
                       [output]
*/
pm::pb_node_t *conv_bias_add_relu_flex(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input,
        pm::pb_node_t *post_src, bool is_bf16 = false) {
    in_edges_t in_edges, post_src_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }

    pm::pb_op_t *conv = pgraph->append_op(
            graph::op_kind::Convolution, in_edges_t {in_edge(0, input, 0)});
    conv->append_decision_function(check_conv_attrs);
    if (is_bf16) {
        conv->append_decision_function(
                check_input_dtype<graph::data_type::bf16>);
    } else {
        conv->append_decision_function(
                check_input_dtype<graph::data_type::f32>);
    }

    auto biasadd_subgraph = std::make_shared<pb_graph_t>();
    auto biasadd = biasadd_subgraph->append_op(graph::op_kind::BiasAdd);
    biasadd_subgraph->create_input_port(0, biasadd, 0);
    biasadd_subgraph->create_output_port(0, biasadd, 0);
    auto optional_biasadd = pgraph->append_optional(
            biasadd_subgraph, in_edges_t {in_edge(0, conv, 0)});

    in_edges_t add_in_edges = in_edges_t {in_edge(0, optional_biasadd, 0)};
    if (post_src) { add_in_edges.emplace_back(in_edge(1, post_src, 0)); }
    pm::pb_op_t *add = pgraph->append_op(graph::op_kind::Add, add_in_edges);
    pm::pb_op_t *relu = pgraph->append_op(
            graph::op_kind::ReLU, in_edges_t {in_edge(0, add, 0)});
    return relu;
};

pm::pb_node_t *conv_bias_add_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_node_t *input, pm::pb_node_t *post_src, bool use_biasadd = false,
        bool is_bf16 = false) {
    in_edges_t in_edges, post_src_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }

    pm::pb_op_t *conv = pgraph->append_op(
            graph::op_kind::Convolution, in_edges_t {in_edge(0, input, 0)});
    conv->append_decision_function(check_conv_attrs);
    if (is_bf16) {
        conv->append_decision_function(
                check_input_dtype<graph::data_type::bf16>);
    } else {
        conv->append_decision_function(
                check_input_dtype<graph::data_type::f32>);
    }

    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        pm::pb_op_t *biasadd = pgraph->append_op(
                graph::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv_bias_dst = conv;
    }

    in_edges_t add_in_edges = in_edges_t {in_edge(0, conv_bias_dst, 0)};
    if (post_src) { add_in_edges.emplace_back(in_edge(1, post_src, 0)); }
    pm::pb_op_t *add = pgraph->append_op(graph::op_kind::Add, add_in_edges);
    pm::pb_op_t *relu = pgraph->append_op(
            graph::op_kind::ReLU, in_edges_t {in_edge(0, add, 0)});
    return relu;
};

/*
                    [filter]
                       |
         [input]    Quantize (optional)
            |          |
      Dequantize   Dequantize
              \     /
            Convolution      [bias]
                 \           /
                BiasAdd (optional)
                   |
                  Relu
                   |
                Quantize
                   |
                [output]
*/
pm::pb_node_t *int8_conv_bias_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_node_t *input, bool has_relu = false,
        bool use_quant_wei = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }

    pm::pb_op_t *dequant_src = pgraph->append_alternation(
            {graph::op_kind::Dequantize, graph::op_kind::DynamicDequantize},
            in_edges);
    pm::pb_op_t *dequant_wei;
    if (use_quant_wei) {
        pm::pb_op_t *quant_wei = pgraph->append_alternation(
                {graph::op_kind::Quantize, graph::op_kind::DynamicQuantize});
        dequant_wei = pgraph->append_alternation(
                {graph::op_kind::Dequantize, graph::op_kind::DynamicDequantize},
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_alternation({graph::op_kind::Dequantize,
                graph::op_kind::DynamicDequantize});
    }

    pm::pb_op_t *conv = pgraph->append_op(graph::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    conv->append_decision_function(check_conv_attrs);

    auto biasadd_subgraph = std::make_shared<pb_graph_t>();
    auto biasadd = biasadd_subgraph->append_op(graph::op_kind::BiasAdd);
    biasadd_subgraph->create_input_port(0, biasadd, 0);
    biasadd_subgraph->create_output_port(0, biasadd, 0);
    auto optional_biasadd = pgraph->append_optional(
            biasadd_subgraph, in_edges_t {in_edge(0, conv, 0)});

    in_edges_t quant_in_edges {in_edge(0, optional_biasadd, 0)};
    if (has_relu) {
        pm::pb_op_t *relu = pgraph->append_op(graph::op_kind::ReLU,
                in_edges_t {in_edge(0, optional_biasadd, 0)});
        quant_in_edges = {in_edge(0, relu, 0)};
    }
    pm::pb_op_t *quant_dst = pgraph->append_alternation(
            {graph::op_kind::Quantize, graph::op_kind::DynamicQuantize},
            quant_in_edges);
    return quant_dst;
};

std::pair<pm::pb_op_t *, pm::pb_op_t *> int8_conv_bias_relu_subgraph(
        const std::shared_ptr<pb_graph_t> &pgraph) {
    pm::pb_op_t *dequant_src = pgraph->append_alternation(
            {graph::op_kind::Dequantize, graph::op_kind::DynamicDequantize});
    pm::pb_op_t *dequant_wei = pgraph->append_alternation(
            {graph::op_kind::Dequantize, graph::op_kind::DynamicDequantize});

    pm::pb_op_t *conv = pgraph->append_op(graph::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    conv->append_decision_function(check_conv_attrs);

    auto biasadd_subgraph = std::make_shared<pb_graph_t>();
    auto biasadd = biasadd_subgraph->append_op(graph::op_kind::BiasAdd);
    biasadd_subgraph->create_input_port(0, biasadd, 0);
    biasadd_subgraph->create_output_port(0, biasadd, 0);
    auto optional_biasadd = pgraph->append_optional(
            biasadd_subgraph, in_edges_t {in_edge(0, conv, 0)});

    pm::pb_op_t *relu = pgraph->append_op(
            graph::op_kind::ReLU, in_edges_t {in_edge(0, optional_biasadd, 0)});
    pm::pb_op_t *quant_dst = pgraph->append_alternation(
            {graph::op_kind::Quantize, graph::op_kind::DynamicQuantize},
            in_edges_t {in_edge(0, relu, 0)});
    return {dequant_src, quant_dst};
};

// int8_conv_bias_add_relu
/*
                    [filter]
                       |
         [input]    Quantize (optional)
            |          |
      Dequantize   Dequantize
              \       /
            Convolution   [bias]
                 \        /
       (optional) BiasAdd   [other]
                        \   /
                         Add
                          |
                         Relu
                          |
                       Quantize
                          |
                        [output]
*/
pm::pb_node_t *int8_conv_bias_add_relu_flex(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input,
        pm::pb_node_t *post_src, bool use_quant_wei = false,
        bool f32_output = false) {
    in_edges_t in_edges, post_src_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    if (post_src) { post_src_edges = in_edges_t {in_edge(0, post_src, 0)}; }

    pm::pb_op_t *dequant_src = pgraph->append_alternation(
            {graph::op_kind::Dequantize, graph::op_kind::DynamicDequantize},
            in_edges);
    pm::pb_op_t *dequant_wei;
    if (use_quant_wei) {
        pm::pb_op_t *quant_wei = pgraph->append_alternation(
                {graph::op_kind::Quantize, graph::op_kind::DynamicQuantize});
        dequant_wei = pgraph->append_alternation(
                {graph::op_kind::Dequantize, graph::op_kind::DynamicDequantize},
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_alternation({graph::op_kind::Dequantize,
                graph::op_kind::DynamicDequantize});
    }
    pm::pb_op_t *dequant_other = pgraph->append_alternation(
            {graph::op_kind::Dequantize, graph::op_kind::DynamicDequantize},
            post_src_edges);

    pm::pb_op_t *conv = pgraph->append_op(graph::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    conv->append_decision_function(check_conv_attrs);

    auto biasadd_subgraph = std::make_shared<pb_graph_t>();
    auto biasadd = biasadd_subgraph->append_op(graph::op_kind::BiasAdd);
    biasadd_subgraph->create_input_port(0, biasadd, 0);
    biasadd_subgraph->create_output_port(0, biasadd, 0);
    auto optional_biasadd = pgraph->append_optional(
            biasadd_subgraph, in_edges_t {in_edge(0, conv, 0)});

    pm::pb_op_t *add = pgraph->append_op(graph::op_kind::Add,
            in_edges_t {in_edge(0, optional_biasadd, 0),
                    in_edge(1, dequant_other, 0)});
    pm::pb_op_t *relu = pgraph->append_op(
            graph::op_kind::ReLU, in_edges_t {in_edge(0, add, 0)});

    // deal with itex int8 last bottleneck
    if (f32_output) { return relu; }

    pm::pb_op_t *quant_dst = pgraph->append_alternation(
            {graph::op_kind::Quantize, graph::op_kind::DynamicQuantize},
            in_edges_t {in_edge(0, relu, 0)});
    return quant_dst;
};

pm::pb_node_t *int8_conv_bias_add_relu(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input,
        pm::pb_node_t *post_src, bool use_biasadd = false,
        bool use_quant_wei = false, bool f32_output = false) {
    in_edges_t in_edges, post_src_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    if (post_src) { post_src_edges = in_edges_t {in_edge(0, post_src, 0)}; }

    pm::pb_op_t *dequant_src = pgraph->append_alternation(
            {graph::op_kind::Dequantize, graph::op_kind::DynamicDequantize},
            in_edges);
    pm::pb_op_t *dequant_wei;
    if (use_quant_wei) {
        pm::pb_op_t *quant_wei = pgraph->append_alternation(
                {graph::op_kind::Quantize, graph::op_kind::DynamicQuantize});
        dequant_wei = pgraph->append_alternation(
                {graph::op_kind::Dequantize, graph::op_kind::DynamicDequantize},
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_alternation({graph::op_kind::Dequantize,
                graph::op_kind::DynamicDequantize});
    }
    pm::pb_op_t *dequant_other = pgraph->append_alternation(
            {graph::op_kind::Dequantize, graph::op_kind::DynamicDequantize},
            post_src_edges);

    pm::pb_op_t *conv = pgraph->append_op(graph::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    conv->append_decision_function(check_conv_attrs);

    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        pm::pb_op_t *biasadd = pgraph->append_op(
                graph::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv_bias_dst = conv;
    }

    pm::pb_op_t *add = pgraph->append_op(graph::op_kind::Add,
            in_edges_t {in_edge(0, conv_bias_dst, 0),
                    in_edge(1, dequant_other, 0)});
    pm::pb_op_t *relu = pgraph->append_op(
            graph::op_kind::ReLU, in_edges_t {in_edge(0, add, 0)});

    // deal with itex int8 last bottleneck
    if (f32_output) { return relu; }

    pm::pb_op_t *quant_dst = pgraph->append_alternation(
            {graph::op_kind::Quantize, graph::op_kind::DynamicQuantize},
            in_edges_t {in_edge(0, relu, 0)});
    return quant_dst;
};

pm::pb_node_t *convolutional_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input,
        bool is_bf16 = false) {
    pm::pb_node_t *dst0 = conv_bias_relu(pgraph, input, true, is_bf16);
    pm::pb_node_t *dst1 = conv_bias_relu(pgraph, dst0, true, is_bf16);
    pm::pb_node_t *dst2 = conv_bias_relu(pgraph, input, false, is_bf16);
    pm::pb_node_t *dst3 = conv_bias_add_relu_flex(pgraph, dst1, dst2, is_bf16);
    return dst3;
};

pm::pb_node_t *identical_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input,
        bool is_bf16 = false) {
    pm::pb_node_t *dst0 = conv_bias_relu(pgraph, input, true, is_bf16);
    pm::pb_node_t *dst1 = conv_bias_relu(pgraph, dst0, true, is_bf16);
    pm::pb_node_t *dst2 = conv_bias_add_relu_flex(pgraph, dst1, input, is_bf16);
    return dst2;
};

pm::pb_node_t *int8_identical_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input,
        bool use_biasadd = false, bool use_quant_wei = false,
        bool output_f32 = false) {
    pm::pb_node_t *quant_dst0
            = int8_conv_bias_relu(pgraph, input, true, use_quant_wei);
    pm::pb_node_t *quant_dst1
            = int8_conv_bias_relu(pgraph, quant_dst0, true, use_quant_wei);
    pm::pb_node_t *quant_dst2 = int8_conv_bias_add_relu(
            pgraph, quant_dst1, input, use_biasadd, use_quant_wei, output_f32);
    return quant_dst2;
};

pm::pb_node_t *int8_convolutional_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_node_t *input,
        bool use_quant_wei = false) {
    pm::pb_node_t *quant_dst0
            = int8_conv_bias_relu(pgraph, input, true, use_quant_wei);
    pm::pb_node_t *quant_dst1
            = int8_conv_bias_relu(pgraph, quant_dst0, true, use_quant_wei);
    pm::pb_node_t *quant_dst2
            = int8_conv_bias_relu(pgraph, input, false, use_quant_wei);
    pm::pb_node_t *quant_dst3
            = int8_conv_bias_add_relu(pgraph, quant_dst1, quant_dst2);
    return quant_dst3;
};

pm::pb_node_t *int8_convolutional_bottleneck_resblock_v2(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool use_biasadd = false, bool use_quant_wei = false) {
    pm::pb_node_t *quant_dst0
            = int8_conv_bias_relu(pgraph, input, true, use_quant_wei);
    pm::pb_node_t *quant_dst1
            = int8_conv_bias_relu(pgraph, quant_dst0, true, use_quant_wei);
    pm::pb_node_t *quant_dst2
            = int8_conv_bias_relu(pgraph, quant_dst1, false, use_quant_wei);
    pm::pb_node_t *quant_dst3 = int8_conv_bias_add_relu(
            pgraph, input, quant_dst2, use_biasadd, use_quant_wei);
    return quant_dst3;
};

pm::pb_node_t *general_identical_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, bool is_bf16 = false) {
    auto subgraph = std::make_shared<pb_graph_t>();
    auto ports = conv_bias_relu_subgraph(subgraph, is_bf16);
    subgraph->create_input_port(0, ports.first, 0);
    subgraph->create_output_port(0, ports.second, 0);

    pm::pb_node_t *dst1 = pgraph->append_repetition(subgraph, {0, 0}, 1, 4);

    pm::pb_node_t *dst2
            = conv_bias_add_relu_flex(pgraph, dst1, nullptr, is_bf16);
    return dst2;
};

pm::pb_node_t *general_convolutional_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, bool is_bf16 = false) {
    auto subgraph = std::make_shared<pb_graph_t>();
    auto ports = conv_bias_relu_subgraph(subgraph, is_bf16);
    subgraph->create_input_port(0, ports.first, 0);
    subgraph->create_output_port(0, ports.second, 0);
    pm::pb_node_t *dst1 = pgraph->append_repetition(subgraph, {0, 0}, 1, 4);

    pm::pb_node_t *dst2 = conv_bias_relu(pgraph, nullptr, false, is_bf16);
    pm::pb_node_t *dst3 = conv_bias_add_relu_flex(pgraph, dst1, dst2, is_bf16);
    return dst3;
};

pm::pb_node_t *general_int8_identical_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph) {
    auto subgraph = std::make_shared<pb_graph_t>();
    auto ports = int8_conv_bias_relu_subgraph(subgraph);
    subgraph->create_input_port(0, ports.first, 0);
    subgraph->create_output_port(0, ports.second, 0);
    pm::pb_node_t *quant_dst1
            = pgraph->append_repetition(subgraph, {0, 0}, 1, 4);

    pm::pb_node_t *quant_dst2
            = int8_conv_bias_add_relu_flex(pgraph, quant_dst1, nullptr);
    return quant_dst2;
};

pm::pb_node_t *general_int8_convolutional_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph) {
    auto subgraph = std::make_shared<pb_graph_t>();
    auto ports = int8_conv_bias_relu_subgraph(subgraph);
    subgraph->create_input_port(0, ports.first, 0);
    subgraph->create_output_port(0, ports.second, 0);
    pm::pb_node_t *quant_dst1
            = pgraph->append_repetition(subgraph, {0, 0}, 1, 4);

    pm::pb_node_t *quant_dst2
            = int8_conv_bias_relu(pgraph, nullptr, false, false);
    pm::pb_node_t *quant_dst3
            = int8_conv_bias_add_relu_flex(pgraph, quant_dst1, quant_dst2);
    return quant_dst3;
};

pm::pb_op_t *conv_bn_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool has_relu = false, bool is_bf16 = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }

    pm::pb_op_t *conv
            = pgraph->append_op(graph::op_kind::Convolution, in_edges);
    conv->allow_external_outputs();
    conv->append_decision_function(check_conv_attrs);
    if (is_bf16) {
        conv->append_decision_function(
                check_input_dtype<graph::data_type::bf16>);
    } else {
        conv->append_decision_function(
                check_input_dtype<graph::data_type::f32>);
    }

    pm::pb_op_t *bn
            = pgraph->append_op(graph::op_kind::BatchNormForwardTraining,
                    in_edges_t {in_edge(0, conv, 0)});
    bn->allow_external_outputs();
    pm::pb_op_t *output = bn;
    if (has_relu) {
        output = pgraph->append_op(graph::op_kind::ReLU, {in_edge(0, bn, 0)});
        output->allow_external_outputs();
    }
    return output;
}

pm::pb_op_t *conv_bn_relu_bwd(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool has_relu = false, bool is_bf16 = false) {
    in_edges_t in_edges;
    if (input) {
        // delta is the second input of both bn_bwd and relu_bwd
        in_edges = in_edges_t {in_edge(1, input, 0)};
    }

    pm::pb_op_t *relu_bwd;
    if (has_relu) {
        relu_bwd = pgraph->append_op(graph::op_kind::ReLUBackward, in_edges);
        relu_bwd->allow_external_outputs();
        in_edges = in_edges_t {in_edge(1, relu_bwd, 0)};
    }
    pm::pb_op_t *bn_bwd = pgraph->append_op(
            graph::op_kind::BatchNormTrainingBackward, in_edges);
    bn_bwd->allow_external_outputs();
    pm::pb_op_t *conv_bwd_data
            = pgraph->append_op(graph::op_kind::ConvolutionBackwardData,
                    in_edges_t {in_edge(0, bn_bwd, 0)});
    conv_bwd_data->allow_external_outputs();
    conv_bwd_data->append_decision_function(check_conv_attrs);
    if (is_bf16) {
        conv_bwd_data->append_decision_function(
                check_input_dtype<graph::data_type::bf16>);
    } else {
        conv_bwd_data->append_decision_function(
                check_input_dtype<graph::data_type::f32>);
    }

    pm::pb_op_t *conv_bwd_filter
            = pgraph->append_op(graph::op_kind::ConvolutionBackwardWeights,
                    in_edges_t {in_edge(1, bn_bwd, 0)});
    conv_bwd_filter->allow_external_outputs();
    conv_bwd_filter->append_decision_function(check_conv_attrs);
    if (is_bf16) {
        conv_bwd_filter->append_decision_function(
                check_input_dtype<graph::data_type::bf16>);
    } else {
        conv_bwd_filter->append_decision_function(
                check_input_dtype<graph::data_type::f32>);
    }
    return conv_bwd_data;
}

pm::pb_op_t *convolutional_bottleneck_training_forward(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    pm::pb_op_t *dst0 = conv_bn_relu(pgraph, nullptr, true, is_bf16);
    pm::pb_op_t *dst1 = conv_bn_relu(pgraph, dst0, true, is_bf16);
    pm::pb_op_t *dst2 = conv_bn_relu(pgraph, dst1, false, is_bf16);
    pm::pb_op_t *dst3 = conv_bn_relu(pgraph, nullptr, false, is_bf16);
    auto bottleneck_add = pgraph->append_op(
            graph::op_kind::Add, {in_edge(0, dst2, 0), in_edge(1, dst3, 0)});
    auto relu = pgraph->append_op(
            graph::op_kind::ReLU, {in_edge(0, bottleneck_add, 0)});
    return relu;
};

pm::pb_op_t *identical_bottleneck_training_forward(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    pm::pb_op_t *dst0 = conv_bn_relu(pgraph, input, true, is_bf16);
    pm::pb_op_t *dst1 = conv_bn_relu(pgraph, dst0, true, is_bf16);
    pm::pb_op_t *dst2 = conv_bn_relu(pgraph, dst1, false, is_bf16);
    auto bottleneck_add
            = pgraph->append_op(graph::op_kind::Add, {in_edge(0, dst2, 0)});
    auto relu = pgraph->append_op(
            graph::op_kind::ReLU, {in_edge(0, bottleneck_add, 0)});
    return relu;
};

pm::pb_op_t *convolutional_bottleneck_training_backward(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    pm::pb_op_t *relu_bwd_top = pgraph->append_op(graph::op_kind::ReLUBackward);
    pm::pb_op_t *dst0 = conv_bn_relu_bwd(pgraph, relu_bwd_top, false, is_bf16);
    pm::pb_op_t *dst1 = conv_bn_relu_bwd(pgraph, dst0, true, is_bf16);
    pm::pb_op_t *dst2 = conv_bn_relu_bwd(pgraph, dst1, true, is_bf16);
    pm::pb_op_t *dst3 = conv_bn_relu_bwd(pgraph, relu_bwd_top, false, is_bf16);
    pm::pb_op_t *bottleneck_add = pgraph->append_op(
            graph::op_kind::Add, {in_edge(0, dst2, 0), in_edge(1, dst3, 0)});
    return bottleneck_add;
};

pm::pb_op_t *identical_bottleneck_training_backward(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    pm::pb_op_t *relu_bwd_top = pgraph->append_op(graph::op_kind::ReLUBackward);
    pm::pb_op_t *dst0 = conv_bn_relu_bwd(pgraph, relu_bwd_top, false, is_bf16);
    pm::pb_op_t *dst1 = conv_bn_relu_bwd(pgraph, dst0, true, is_bf16);
    pm::pb_op_t *dst2 = conv_bn_relu_bwd(pgraph, dst1, true, is_bf16);
    pm::pb_op_t *bottleneck_add = pgraph->append_op(graph::op_kind::Add,
            {in_edge(0, dst2, 0), in_edge(1, relu_bwd_top, 0)});
    return bottleneck_add;
};

pm::pb_op_t *convolutional_bottleneck_training_backward_v2(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    auto dst0 = conv_bn_relu_bwd(pgraph, nullptr, false, is_bf16);
    auto dst1 = conv_bn_relu_bwd(pgraph, dst0, true, is_bf16);
    auto dst2 = conv_bn_relu_bwd(pgraph, dst1, true, is_bf16);
    auto dst3 = conv_bn_relu_bwd(pgraph, nullptr, false, is_bf16);
    auto bottleneck_add = pgraph->append_op(
            graph::op_kind::Add, {in_edge(0, dst2, 0), in_edge(1, dst3, 0)});
    auto relu_bwd = pgraph->append_op(
            graph::op_kind::ReLUBackward, {in_edge(1, bottleneck_add, 0)});
    return relu_bwd;
};

pm::pb_op_t *identical_bottleneck_training_backward_v2(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    pm::pb_op_t *dst0 = conv_bn_relu_bwd(pgraph, nullptr, false, is_bf16);
    pm::pb_op_t *dst1 = conv_bn_relu_bwd(pgraph, dst0, true, is_bf16);
    pm::pb_op_t *dst2 = conv_bn_relu_bwd(pgraph, dst1, true, is_bf16);
    pm::pb_op_t *bottleneck_add
            = pgraph->append_op(graph::op_kind::Add, {in_edge(0, dst2, 0)});
    pm::pb_op_t *relu_bwd = pgraph->append_op(
            graph::op_kind::ReLUBackward, {in_edge(1, bottleneck_add, 0)});
    return relu_bwd;
};

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(fp32_conv_inference_pattern)
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_identical_bottleneck)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    general_identical_bottleneck_resblock(pgraph, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_convolutional_bottleneck)
        .set_priority(5.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    general_convolutional_bottleneck_resblock(pgraph, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_resnet50_stage_1_4_fusion_gc)
        .set_priority(22.f) // high priority to support itex
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    output = identical_bottleneck_resblock(pgraph, output);
                    output = identical_bottleneck_resblock(pgraph, output);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_resnet50_stage_2_fusion_gc)
        .set_priority(22.1f) // high priority to support itex
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(pgraph, output);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_resnet50_stage_3_fusion_gc)
        .set_priority(22.2f) // high priority to support itex
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(pgraph, output);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(fp32_conv_training_pattern)
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_identical_bottleneck_forward)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_forward(
                            pgraph, nullptr, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_convolutional_bottleneck_forward)
        .set_priority(5.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_forward(
                            pgraph, nullptr, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_identical_bottleneck_backward_v1)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_backward(
                            pgraph, nullptr, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_convolutional_bottleneck_backward_v1)
        .set_priority(5.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_backward(
                            pgraph, nullptr, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_identical_bottleneck_backward_v2)
        .set_priority(4.0f) // set to lower priority as backup
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_backward_v2(
                            pgraph, nullptr, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_convolutional_bottleneck_backward_v2)
        .set_priority(4.5f) // set to lower priority as backup
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_backward_v2(
                            pgraph, nullptr, false);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(bf16_conv_inference_pattern)
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_identical_bottleneck)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    general_identical_bottleneck_resblock(pgraph, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_convolutional_bottleneck)
        .set_priority(5.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    general_convolutional_bottleneck_resblock(pgraph, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_resnet50_stage_1_4_fusion_gc)
        .set_priority(22.f) // high priority to support itex
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, true);
                    output = identical_bottleneck_resblock(
                            pgraph, output, true);
                    output = identical_bottleneck_resblock(
                            pgraph, output, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_resnet50_stage_2_fusion_gc)
        .set_priority(22.1f) // high priority to support itex
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, true);
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(
                                pgraph, output, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_resnet50_stage_3_fusion_gc)
        .set_priority(22.2f) // high priority to support itex
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, true);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(
                                pgraph, output, true);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(bf16_conv_training_pattern)
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_identical_bottleneck_forward)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_forward(
                            pgraph, nullptr, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_convolutional_bottleneck_forward)
        .set_priority(5.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_forward(
                            pgraph, nullptr, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_identical_bottleneck_backward_v1)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_backward(
                            pgraph, nullptr, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_convolutional_bottleneck_backward_v1)
        .set_priority(5.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_backward(
                            pgraph, nullptr, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_identical_bottleneck_backward_v2)
        .set_priority(4.0f) // set to lower priority as backup
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_backward_v2(
                            pgraph, nullptr, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_convolutional_bottleneck_backward_v2)
        .set_priority(4.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_backward_v2(
                            pgraph, nullptr, true);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(int8_conv_inference_pattern)
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_identical_bottleneck)
        .set_priority(5.0f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    general_int8_identical_bottleneck_resblock(pgraph);
                });

// conv bottleneck inference
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_convolutional_bottleneck)
        .set_priority(5.5f)
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    general_int8_convolutional_bottleneck_resblock(pgraph);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_resnet50_stage_1_4_fusion_gc)
        .set_priority(22.f) // high priority to support lz models
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    output = int8_identical_bottleneck_resblock(pgraph, output);
                    output = int8_identical_bottleneck_resblock(pgraph, output);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_resnet50_stage_2_fusion_gc)
        .set_priority(22.1f) // high priority to support lz models
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_resnet50_stage_3_fusion_gc)
        .set_priority(22.2f) // high priority to support lz models
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, itex_int8_resnet50_stage_1_fusion_gc)
        .set_priority(22.1f) // high priority to support lz models
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, true, true);
                });

// For itex int8 rn50 only (include the weight quantize into pattern)
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, itex_int8_resnet50_stage_2_fusion_gc)
        .set_priority(22.2f) // high priority to support lz models
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, true, true);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
                });

// For itex int8 rn50 only (include the weight quantize into pattern)
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, itex_int8_resnet50_stage_3_fusion_gc)
        .set_priority(22.3f) // high priority to support lz models
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, true, true);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
                });

// For itex int8 rn50 only (include the weight quantize & output has no quant)
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, itex_int8_resnet50_stage_4_fusion_gc)
        .set_priority(22.1f) // high priority to support lz models
        .set_engine_kind(graph::engine_kind::cpu)
        .set_kind(graph::partition_kind_t::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_node_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, true, true, /* f32 output */ true);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
