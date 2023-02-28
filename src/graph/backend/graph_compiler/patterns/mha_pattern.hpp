/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_MHA_PATTERN_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_MHA_PATTERN_HPP

#include <memory>

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

// [rep_min, rep_max)
pm::repetition_t *create_append_transpose_repetition_subgraph(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        int rep_min, int rep_max) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    auto transpose_subgraph
            = std::make_shared<pb_graph_t>("transpose_subgraph");
    auto transpose_rep = transpose_subgraph->append_op(
            graph::op_kind::StaticTranspose, "transpose_rep");
    transpose_subgraph->create_input_port(0, transpose_rep, 0);
    transpose_subgraph->create_output_port(0, transpose_rep, 0);
    auto transpose = pgraph->append_repetition(transpose_subgraph, {0, 0},
            rep_min, rep_max, in_edges, "transpose");
    return transpose;
};

pm::repetition_t *create_optional_mul_subgraph(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool allow_external = false) {
    auto optional_mul_subgraph
            = std::make_shared<pb_graph_t>("optional_mul_subgraph");
    auto optional_mul = optional_mul_subgraph->append_op(
            graph::op_kind::Multiply, "optional_mul");
    if (allow_external) { optional_mul->allow_external_outputs(); }

    optional_mul_subgraph->create_input_port(0, optional_mul, 0);
    optional_mul_subgraph->create_output_port(0, optional_mul, 0);
    auto mul = pgraph->append_optional(
            optional_mul_subgraph, {in_edge(0, input, 0)}, "optional_mul");
    return mul;
};

pm::alternation_t *create_alternative_mul_subgraph(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input) {
    // 2 alternations: 1) a single mul; 2) 2 consecutive mul
    auto successive_mul_subgraph
            = std::make_shared<pb_graph_t>("successive_mul_subgraph");
    auto mul1 = successive_mul_subgraph->append_op(
            graph::op_kind::Multiply, "mul1");
    auto mul2 = successive_mul_subgraph->append_op(
            graph::op_kind::Multiply, {in_edge(0, mul1, 0)}, "mul2");
    successive_mul_subgraph->create_input_port(0, mul1, 0);
    successive_mul_subgraph->create_output_port(0, mul2, 0);

    auto single_mul_subgraph
            = std::make_shared<pb_graph_t>("single_mul_subgraph");
    auto mul = single_mul_subgraph->append_op(graph::op_kind::Multiply, "mul");
    single_mul_subgraph->create_input_port(0, mul, 0);
    single_mul_subgraph->create_output_port(0, mul, 0);

    auto mul_subgraph = pgraph->append_alternation(
            {successive_mul_subgraph, single_mul_subgraph},
            {in_edge(0, input, 0)}, "mul_subgraph");
    return mul_subgraph;
}

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(fp32_mha_pattern)
// fp32 MHA pattern
/*
   (f32)[Query]    [Key](f32)
            |         |
        Reshape    Reshape
            |         |
  [1-2]*Transpose Transpose*[1-2]
              \     /
               MatMul  [fscore scale](f32)
                 \    /
[Attention Mask] Div|Mul  [Value](f32)
              \   /        |
                Add     Reshape
                 |         |
              Softmax  Transpose*[1-2]
                    \     /
                     MatMul
                        |
                    Transpose
                        |
                  Reshape (optional)
                        |
                     [output](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, fp32_mha_pattern)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "query_reshape");
                    query_reshape->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto query_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, query_reshape, 1, 3);
                    auto key_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, key_reshape, 1, 3);
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, query_transpose, 0),
                                    in_edge(1, key_transpose, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto value_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, value_reshape, 1, 3);
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, softmax, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");
                    auto transpose_output = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");

                    auto optional_reshape_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_reshape_subgraph");
                    auto optional_reshape
                            = optional_reshape_subgraph->append_op(
                                    graph::op_kind::StaticReshape,
                                    "optional_reshape");
                    optional_reshape_subgraph->create_input_port(
                            0, optional_reshape, 0);
                    optional_reshape_subgraph->create_output_port(
                            0, optional_reshape, 0);

                    pgraph->append_optional(optional_reshape_subgraph,
                            {in_edge(0, transpose_output, 0)},
                            "reshape_output");
                });

// fp32 MHA pattern alternative
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
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mha_pattern_alternative)
        .set_priority(4.5f) // lower priority than non-alternative
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            graph::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, softmax, 0)}, "matmul_v");
                    matmul_v->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto transpose_output = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");
                    pgraph->append_alternation(
                            {graph::op_kind::Reorder,
                                    graph::op_kind::StaticReshape},
                            {in_edge(0, transpose_output, 0)},
                            "reshape_reorder_output");
                });

// fp32 MHA training forward pattern
/*
    (f32)[QueryTrans]   [KeyTrans](f32)
                  \      /
                   MatMul  [FscoreScale](f32)
                     \    /
(f32)[AttentionMask] Div|Mul
                  \   /
                    Add
                     |
                  Softmax   [Dropout](f32)
                      \     /
                      Multiply  [Select](f32)
                         \      /
               (optional)Multiply  [ValueTrans](f32)
                            \      /
                             MatMul
                                |
                            Transpose
                                |
                             Reshape
                                |
                            [output](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mha_forward_pattern)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            graph::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    fscore_scale->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    fscore_add->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    softmax->allow_external_outputs();
                    auto dropout = pgraph->append_op(graph::op_kind::Multiply,
                            {in_edge(0, softmax, 0)}, "dropout");
                    dropout->allow_external_outputs();
                    auto select = create_optional_mul_subgraph(
                            pgraph, dropout, true);
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, select, 0)}, "matmul_v");
                    matmul_v->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto reshape
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    {in_edge(0, matmul_v, 0)}, "reshape");
                    pgraph->append_op(graph::op_kind::StaticReshape,
                            {in_edge(0, reshape, 0)}, "transpose_output");
                });

// fp32 MHA training backward pattern
/*
                [BackwardIn](f32)
                        |
                     Reshape
                        |
(f32)[DrouputOut]   Transpose   [ValueTrans](f32)
          \       /         \    /
            MatMul           MatMul    [Select](f32)
              |                  \     /
        [output](f32)   (optional)Multiply    [Dropout](f32)
                                       \     /
                                      Multiply     [SoftmaxOut](f32)
                                        /  \        /
                                       /     Multiply
                                       |      |
                                       |  ReduceSum
                                        \   /
                                         Sub  [SoftmaxOut](f32)
                                          \   /
                                       Multiply  [Fscore](f32)
                                            \    /
                                           Div|Mul  [QueryTrans](f32)
                         ___________________/    \   /
                         \   [KeyTrans](f32)     MatMul
                          \      /                 |
                           MatMul             [output](f32)
                             |
                         [output](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mha_backward_pattern)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto in_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "in_reshape");
                    in_reshape->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto in_transpose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, in_reshape, 0)}, "in_transpose");
                    auto bmm_v_grad_weight = pgraph->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(1, in_transpose, 0)}, "bmm_v_grad_weight");
                    bmm_v_grad_weight->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto bmm_v_grad_data = pgraph->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, in_transpose, 0)}, "bmm_v_grad_data");
                    bmm_v_grad_data->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto dropout_grad = create_alternative_mul_subgraph(
                            pgraph, bmm_v_grad_data);
                    auto softmax_mul = pgraph->append_op(
                            graph::op_kind::Multiply,
                            {in_edge(0, dropout_grad, 0)}, "softmax_mul");
                    softmax_mul->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto softmax_sum = pgraph->append_op(
                            graph::op_kind::ReduceSum,
                            {in_edge(0, softmax_mul, 0)}, "softmax_sum");
                    softmax_sum->append_decision_function(check_reduce_attrs);
                    auto softmax_sub
                            = pgraph->append_op(graph::op_kind::Subtract,
                                    {in_edge(0, dropout_grad, 0),
                                            in_edge(1, softmax_sum, 0)},
                                    "softmax_sub");
                    /* Create 2 subgraph for alternation */
                    auto successive_mul_subgraph = std::make_shared<pb_graph_t>(
                            "successive_mul_subgraph");
                    auto softmax_mul2 = successive_mul_subgraph->append_op(
                            graph::op_kind::Multiply, "softmax_mul2");
                    auto fscore_grad_alter1
                            = successive_mul_subgraph->append_alternation(
                                    {graph::op_kind::Divide,
                                            graph::op_kind::Multiply},
                                    {in_edge(0, softmax_mul2, 0)},
                                    "fscore_grad");
                    successive_mul_subgraph->create_input_port(
                            0, softmax_mul2, 0);
                    successive_mul_subgraph->create_output_port(
                            0, fscore_grad_alter1, 0);

                    auto single_fscore_grad_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "single_fscore_grad_subgraph");
                    auto fscore_grad_alter2
                            = single_fscore_grad_subgraph->append_alternation(
                                    {graph::op_kind::Divide,
                                            graph::op_kind::Multiply},
                                    "fscore_grad");
                    single_fscore_grad_subgraph->create_input_port(
                            0, fscore_grad_alter2, 0);
                    single_fscore_grad_subgraph->create_output_port(
                            0, fscore_grad_alter2, 0);

                    auto softmax_grad_alter = pgraph->append_alternation(
                            {successive_mul_subgraph,
                                    single_fscore_grad_subgraph},
                            {in_edge(0, softmax_sub, 0)}, "softmax_grad_alter");
                    auto bmm_q_grad_weight
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    {in_edge(0, softmax_grad_alter, 0)},
                                    "bmm_q_grad_weight");
                    bmm_q_grad_weight->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                    auto bmm_k_grad_weight
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    {in_edge(0, softmax_grad_alter, 0)},
                                    "bmm_k_grad_weight");
                    bmm_k_grad_weight->append_decision_function(
                            check_input_dtype<graph::data_type::f32>);
                });

// fp32 MHA pattern with special reshape for softmax
/*
   (f32)[Query]    [Key](f32)
            |         |
        Reshape    Reshape
            |         |
        Transpose Transpose
              \     /
               MatMul  [Attention Mask](f32)
                  \   /
                   Add
                    |
                 Reshape   [Value](f32)
                    |        |
                 Softmax   Reshape
                    |        |
                 Reshape  Transpose
                     \     /
                      MatMul
                        |
                    Transpose
                        |
                     Reshape
                        |
                     [output](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mha_pattern_alternative2)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "query_reshape");
                    auto query_transpose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, query_reshape, 0)}, "query_transpose");

                    auto key_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, key_reshape, 0)}, "key_transpose");

                    auto mul = create_optional_mul_subgraph(
                            pgraph, key_transpose);

                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, query_transpose, 0),
                                    in_edge(1, mul, 0)},
                            "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);

                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, matmul_qk, 0)}, "fscore_add");

                    auto optional_pre_reshape_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_pre_reshape_subgraph");
                    auto optional_pre_reshape
                            = optional_pre_reshape_subgraph->append_op(
                                    graph::op_kind::StaticReshape,
                                    "optional_pre_reshape");
                    optional_pre_reshape_subgraph->create_input_port(
                            0, optional_pre_reshape, 0);
                    optional_pre_reshape_subgraph->create_output_port(
                            0, optional_pre_reshape, 0);
                    auto pre_reshape = pgraph->append_optional(
                            optional_pre_reshape_subgraph,
                            {in_edge(0, fscore_add, 0)}, "pre_reshape");

                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, pre_reshape, 0)}, "softmax");

                    auto optional_post_reshape_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_post_reshape_subgraph");
                    auto optional_post_reshape
                            = optional_post_reshape_subgraph->append_op(
                                    graph::op_kind::StaticReshape,
                                    "optional_post_reshape");
                    optional_post_reshape_subgraph->create_input_port(
                            0, optional_post_reshape, 0);
                    optional_post_reshape_subgraph->create_output_port(
                            0, optional_post_reshape, 0);
                    auto post_reshape = pgraph->append_optional(
                            optional_post_reshape_subgraph,
                            {in_edge(0, softmax, 0)}, "post_reshape");

                    auto value_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, value_reshape, 0)}, "value_transpose");

                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, post_reshape, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");

                    auto post_transpose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "post_transpose");
                    pgraph->append_op(graph::op_kind::StaticReshape,
                            {in_edge(0, post_transpose, 0)}, "reshape_output");
                });

// fake int8 MHA pattern corresponding to fp32_mha_pattern_alternative2
/*
   (f32)[Query]    [Key](f32)
            |         |
        Reshape    Reshape
            |         |
        Transpose Transpose
              \     /
               MatMul
                 |
              Quantize
                 |
             Dequantize    [Attention Mask](f32) 
                  \__    __/
                      Add    [Fscore Scale](f32) 
                        \      /
                        Multiply
                           |
                        Reshape   [Value](f32)
                           |        |
                        Softmax   Reshape
                           |        |
                        Reshape  Transpose
                            \     /
                             MatMul
                               |
                            [output](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, fake_int8_mha_pattern)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "query_reshape");
                    auto query_transpose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, query_reshape, 0)}, "query_transpose");

                    auto key_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, key_reshape, 0)}, "key_transpose");
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, query_transpose, 0),
                                    in_edge(1, key_transpose, 0)},
                            "matmul_qk");

                    auto fquantize = pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, matmul_qk, 0)}, "fquantize");
                    auto fdequantize
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    {in_edge(0, fquantize, 0)}, "fdequantize");
                    auto radd = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fdequantize, 0)}, "radd");
                    auto rmultiply = pgraph->append_op(graph::op_kind::Multiply,
                            {in_edge(0, radd, 0)}, "rmultiply");

                    auto pre_softmax_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape,
                            {in_edge(0, rmultiply, 0)}, "pre_softmax_reshape");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, pre_softmax_reshape, 0)}, "softmax");
                    auto post_softmax_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape,
                            {in_edge(0, softmax, 0)}, "post_softmax_reshape");

                    auto value_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, value_reshape, 0)}, "value_transpose");

                    pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, post_softmax_reshape, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");
                });

/*
   (f32)[Query]     [Key](f32)
              \     /
               MatMul
                 |
        Divide|Multiply (optional)
                 |
               Add (optional)
                 |
              Softmax    [Value](f32)
                    \     /
                     MatMul
                       |
                    [output](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mha_pattern_alternative3)
        .set_priority(4.4f) // lower priority than mha_pattern_alternative
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            graph::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto optional_postops_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_postops_subgraph");
                    auto scale = optional_postops_subgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            "scale");
                    auto add = optional_postops_subgraph->append_op(
                            graph::op_kind::Add, {in_edge(0, scale, 0)}, "add");
                    optional_postops_subgraph->create_input_port(0, scale, 0);
                    optional_postops_subgraph->create_output_port(0, add, 0);
                    auto postops
                            = pgraph->append_optional(optional_postops_subgraph,
                                    {in_edge(0, matmul_qk, 0)}, "postops");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, postops, 0)}, "softmax");
                    pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, softmax, 0)}, "matmul_v");
                });

/*
   (f32)[Query]     [Key](f32)
              \     /
               MatMul
                 |
         Divide|Multiply
                 |
                Add
                 |
              Softmax
                 |
            [output](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_matmul_softmax_fusion)
        .set_priority(4.0f)
        .set_kind(graph::partition_kind_t::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            graph::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                });

// fp32 MHA pattern with special view and max
/*
   (f32)[Query]    [Key](f32)
              \     /
               MatMul
                 |
              Reshape  [Attention Mask](f32)
                  \    /
                   Add  [Max In](f32)
                    \   /
                     Max
                      |
                   Reshape
                      |
                   Softmax   [Value](f32)
                       \     /
                        MatMul
                          |
                       [output](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mha_pattern_alternative4)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            graph::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto reshape1
                            = pgraph->append_op(graph::op_kind::StaticReshape,
                                    {in_edge(0, matmul_qk, 0)}, "reshape1");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, reshape1, 0)}, "fscore_add");
                    auto fscore_max = pgraph->append_op(graph::op_kind::Maximum,
                            {in_edge(0, fscore_add, 0)}, "fscore_max");
                    auto reshape2
                            = pgraph->append_op(graph::op_kind::StaticReshape,
                                    {in_edge(0, fscore_max, 0)}, "reshape2");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, reshape2, 0)}, "softmax");

                    pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, softmax, 0)}, "matmul_v");
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(bf16_mha_pattern)
// bf16 MHA pattern (it is the same as fp32 pattern except dtype)
/*
  (bf16)[Query]    [Key](bf16)
            |         |
        Reshape    Reshape
            |         |
   [1-2]*Transpose Transpose*[1-2]
              \     /
               MatMul  [fscore scale]
                 \    /
[Attention Mask] Div|Mul  [Value](bf16)
              \   /        |
                Add     Reshape
                 |         |
              Softmax  Transpose*[1-2]
                    \     /
                     MatMul
                        |
                    Transpose
                        |
                  Reshape (optional)
                        |
                     [output](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, bf16_mha_pattern)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "query_reshape");
                    query_reshape->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto query_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, query_reshape, 1, 3);
                    auto key_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, key_reshape, 1, 3);
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, query_transpose, 0),
                                    in_edge(1, key_transpose, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto value_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, value_reshape, 1, 3);
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, softmax, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");
                    auto transpose_output = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");

                    auto optional_reshape_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_reshape_subgraph");
                    auto optional_reshape
                            = optional_reshape_subgraph->append_op(
                                    graph::op_kind::StaticReshape,
                                    "optional_reshape");
                    optional_reshape_subgraph->create_input_port(
                            0, optional_reshape, 0);
                    optional_reshape_subgraph->create_output_port(
                            0, optional_reshape, 0);

                    pgraph->append_optional(optional_reshape_subgraph,
                            {in_edge(0, transpose_output, 0)},
                            "reshape_output");
                });

// bf16 MHA pattern alternative
/*
  (bf16)[Query]    [Key](bf16)
              \     /
               MatMul  [fscore scale]
                 \    /
[Attention Mask] Div|Mul
              \   /
                Add
                 |
              Softmax  [Value](bf16)
                    \     /
                     MatMul
                        |
                    Transpose
                        |
                Reorder|StaticReshape
                        |
                     [output](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mha_pattern_alternative)
        .set_priority(4.5f) // lower priority than non-alternative
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            graph::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, softmax, 0)}, "matmul_v");
                    matmul_v->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto transpose_output = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");
                    pgraph->append_alternation(
                            {graph::op_kind::Reorder,
                                    graph::op_kind::StaticReshape},
                            {in_edge(0, transpose_output, 0)},
                            "reshape_reorder_output");
                });

// bf16 MHA training forward pattern
/*
   (bf16)[QueryTrans]   [KeyTrans](bf16)
                  \      /
                   MatMul  [FscoreScale](f32/bf16)
                     \    /
(bf16)[AttentionMask] Div|Mul
                  \   /
                    Add
                     |
                  Softmax   [Dropout](bf16)
                      \     /
                      Multiply  [Select](bf16)
                         \      /
               (optional)Multiply  [ValueTrans](bf16)
                            \      /
                             MatMul
                                |
                            Transpose
                                |
                             Reshape
                                |
                            [output](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mha_forward_pattern)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            graph::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    fscore_add->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    softmax->allow_external_outputs();
                    auto dropout = pgraph->append_op(graph::op_kind::Multiply,
                            {in_edge(0, softmax, 0)}, "dropout");
                    dropout->allow_external_outputs();
                    auto select = create_optional_mul_subgraph(
                            pgraph, dropout, true);
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, select, 0)}, "matmul_v");
                    matmul_v->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto reshape
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    {in_edge(0, matmul_v, 0)}, "reshape");
                    pgraph->append_op(graph::op_kind::StaticReshape,
                            {in_edge(0, reshape, 0)}, "transpose_output");
                });

// bf16 MHA training backward pattern
/*
                [BackwardIn](bf16)
                        |
                     Reshape
                        |
(bf16)[DrouputOut]  Transpose   [ValueTrans](bf16)
          \       /         \    /
            MatMul           MatMul    [Select](bf16)
              |                  \     /
        [output](bf16)  (optional)Multiply    [Dropout](bf16)
                                       \     /
                                      Multiply     [SoftmaxOut](bf16)
                                        /  \        /
                                       /     Multiply
                                       |      |
                                       |  ReduceSum
                                        \   /
                                         Sub  [SoftmaxOut](bf16)
                                          \   /
                                       Multiply  [Fscore](bf16)
                                            \    /
                                           Div|Mul  [QueryTrans](bf16)
                         ___________________/    \   /
                         \   [KeyTrans](bf16)     MatMul
                          \      /                 |
                           MatMul             [output](bf16)
                             |
                         [output](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mha_backward_pattern)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto in_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape, "in_reshape");
                    in_reshape->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto in_transpose = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, in_reshape, 0)}, "in_transpose");
                    auto bmm_v_grad_weight = pgraph->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(1, in_transpose, 0)}, "bmm_v_grad_weight");
                    bmm_v_grad_weight->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto bmm_v_grad_data = pgraph->append_op(
                            graph::op_kind::MatMul,
                            {in_edge(0, in_transpose, 0)}, "bmm_v_grad_data");
                    bmm_v_grad_data->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto dropout_grad = create_alternative_mul_subgraph(
                            pgraph, bmm_v_grad_data);
                    auto softmax_mul = pgraph->append_op(
                            graph::op_kind::Multiply,
                            {in_edge(0, dropout_grad, 0)}, "softmax_mul");
                    softmax_mul->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto softmax_sum = pgraph->append_op(
                            graph::op_kind::ReduceSum,
                            {in_edge(0, softmax_mul, 0)}, "softmax_sum");
                    softmax_sum->append_decision_function(check_reduce_attrs);
                    auto softmax_sub
                            = pgraph->append_op(graph::op_kind::Subtract,
                                    {in_edge(0, dropout_grad, 0),
                                            in_edge(1, softmax_sum, 0)},
                                    "softmax_sub");
                    /* Create 2 subgraph for alternation */
                    auto successive_mul_subgraph = std::make_shared<pb_graph_t>(
                            "successive_mul_subgraph");
                    auto softmax_mul2 = successive_mul_subgraph->append_op(
                            graph::op_kind::Multiply, "softmax_mul2");
                    auto fscore_grad_alter1
                            = successive_mul_subgraph->append_alternation(
                                    {graph::op_kind::Divide,
                                            graph::op_kind::Multiply},
                                    {in_edge(0, softmax_mul2, 0)},
                                    "fscore_grad");
                    successive_mul_subgraph->create_input_port(
                            0, softmax_mul2, 0);
                    successive_mul_subgraph->create_output_port(
                            0, fscore_grad_alter1, 0);

                    auto single_fscore_grad_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "single_fscore_grad_subgraph");
                    auto fscore_grad_alter2
                            = single_fscore_grad_subgraph->append_alternation(
                                    {graph::op_kind::Divide,
                                            graph::op_kind::Multiply},
                                    "fscore_grad");
                    single_fscore_grad_subgraph->create_input_port(
                            0, fscore_grad_alter2, 0);
                    single_fscore_grad_subgraph->create_output_port(
                            0, fscore_grad_alter2, 0);

                    auto softmax_grad_alter = pgraph->append_alternation(
                            {successive_mul_subgraph,
                                    single_fscore_grad_subgraph},
                            {in_edge(0, softmax_sub, 0)}, "softmax_grad_alter");
                    auto bmm_q_grad_weight
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    {in_edge(0, softmax_grad_alter, 0)},
                                    "bmm_q_grad_weight");
                    bmm_q_grad_weight->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto bmm_k_grad_weight
                            = pgraph->append_op(graph::op_kind::MatMul,
                                    {in_edge(0, softmax_grad_alter, 0)},
                                    "bmm_k_grad_weight");
                    bmm_k_grad_weight->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                });

/*
   (bf16)[Query]   [Key](bf16)
              \     /
               MatMul
                 |
              Softmax    [Value](bf16)
                    \     /
                     MatMul
                       |
                   [output](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mha_pattern_alternative3)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            graph::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<graph::data_type::bf16>);
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, matmul_qk, 0)}, "softmax");
                    pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, softmax, 0)}, "matmul_v");
                });

/*
   (bf16)[Query]     [Key](bf16)
              \     /
               MatMul
                 |
         Divide|Multiply
                 |
                Add
                 |
              Softmax
                 |
            [output](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_matmul_softmax_fusion)
        .set_priority(4.0f)
        .set_kind(graph::partition_kind_t::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            graph::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(int8_mha_pattern)
// int8 MHA pattern
/*
       (u8/s8)[Query]   [Key](u8/s8)
                 |         |
            Dequantize  Dequantize
                 |         |
             Reshape    Reshape
                 |         |
        [1-2]*Transpose Transpose*[1-2]
                   \     /
                    MatMul  [Fscore Scale](f32)
                      \    /
(f32)[Attention Mask] Div|Mul
                   \   /
                     Add    [Value](u8/s8)
                      |         |
                   Softmax   Dequantize
                      |         |
                   Quantize   Reshape
                      |         |
                  Dequantize Transpose*[1-2]
                         \     /
                          MatMul
                             |
                         Transpose
                             |
                          Reshape (optional)
                             |
                          Quantize
                             |
                        [output](u8/s8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, int8_mha_pattern)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::quantized_mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_query");
                    auto dequantize_key = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_key");
                    auto dequantize_value = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_value");
                    auto query_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape,
                            {in_edge(0, dequantize_query, 0)}, "query_reshape");
                    auto query_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, query_reshape, 1, 3);
                    auto key_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape,
                            {in_edge(0, dequantize_key, 0)}, "key_reshape");
                    auto key_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, key_reshape, 1, 3);
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, query_transpose, 0),
                                    in_edge(1, key_transpose, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto quantize_softmax = pgraph->append_op(
                            graph::op_kind::Quantize, {in_edge(0, softmax, 0)},
                            "quantize_softmax");
                    auto dequantize_softmax
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)},
                                    "dequantize_softmax");
                    auto value_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape,
                            {in_edge(0, dequantize_value, 0)}, "value_reshape");
                    auto value_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, value_reshape, 1, 3);
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, dequantize_softmax, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");

                    auto transpose_output = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");

                    auto optional_reshape_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_reshape_subgraph");
                    auto optional_reshape
                            = optional_reshape_subgraph->append_op(
                                    graph::op_kind::StaticReshape,
                                    "optional_reshape");
                    optional_reshape_subgraph->create_input_port(
                            0, optional_reshape, 0);
                    optional_reshape_subgraph->create_output_port(
                            0, optional_reshape, 0);

                    auto reshape_output
                            = pgraph->append_optional(optional_reshape_subgraph,
                                    {in_edge(0, transpose_output, 0)},
                                    "reshape_output");
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, reshape_output, 0)}, "quantize_output");
                });

// int8-bf16 MHA pattern
/*
       (u8/s8)[Query]   [Key](u8/s8)
                 |         |
             Dequantize Dequantize
                 |         |
              Typecast  Typecast
                 |         |
              Reshape   Reshape
                 |         |
        [1-2]*Transpose Transpose*[1-2]
                   \     /
                    MatMul  [Fscore Scale](f32)
                      \    /
(b16)[Attention Mask] Div|Mul
                   \   /
                     Add
                      |
                   Softmax   [Value](u8/s8)
                      |         |
                   Typecast   Dequantize
                      |         |
                   Quantize   Typecast
                      |         |
                  Dequantize  Reshape
                      |         |
                   Typecast   Transpose*[1-2]
                         \     /
                          MatMul
                             |
                         Transpose
                             |
                          Reshape
                             |
                          Typecast
                             |
                          Quantize
                             |
                        [output](u8/s8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, int8_bf16_mha_pattern)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::quantized_mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_query");
                    auto typecast_query
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_query, 0)},
                                    "typecast_query");
                    auto query_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape,
                            {in_edge(0, typecast_query, 0)}, "query_reshape");
                    auto query_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, query_reshape, 1, 3);

                    auto dequantize_key = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_key");
                    auto typecast_key = pgraph->append_op(
                            graph::op_kind::TypeCast,
                            {in_edge(0, dequantize_key, 0)}, "typecast_key");
                    auto key_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape,
                            {in_edge(0, typecast_key, 0)}, "key_reshape");
                    auto key_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, key_reshape, 1, 3);
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, query_transpose, 0),
                                    in_edge(1, key_transpose, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto typecast_softmax = pgraph->append_op(
                            graph::op_kind::TypeCast, {in_edge(0, softmax, 0)},
                            "typecast_softmax");
                    auto quantize_softmax
                            = pgraph->append_op(graph::op_kind::Quantize,
                                    {in_edge(0, typecast_softmax, 0)},
                                    "quantize_softmax");
                    auto dequantize_softmax
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)},
                                    "dequantize_softmax");
                    auto typecast_softmax_bf16
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_softmax, 0)},
                                    "typecast_softmax");

                    auto dequantize_value = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_value");
                    auto typecast_value
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_value, 0)},
                                    "typecast_value");
                    auto value_reshape = pgraph->append_op(
                            graph::op_kind::StaticReshape,
                            {in_edge(0, typecast_value, 0)}, "value_reshape");
                    auto value_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, value_reshape, 1, 3);
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, typecast_softmax_bf16, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");

                    auto transpose_output = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");
                    auto reshape_output
                            = pgraph->append_op(graph::op_kind::StaticReshape,
                                    {in_edge(0, transpose_output, 0)},
                                    "reshape_output");
                    auto typecast_output = pgraph->append_op(
                            graph::op_kind::TypeCast,
                            {in_edge(0, reshape_output, 0)}, "typecast_output");
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, typecast_output, 0)},
                            "quantize_output");
                });

/*
        (int8)[Query]   [Key](int8)
                 |          |
             Dequantize Dequantize
                   \     /
                    MatMul  [Fscore Scale](f32)
                      \    /
(f32)[Attention Mask] Div|Mul
                   \   /
                     Add
                      |
                   Softmax
                      |
                   Quantize  [Value](int8)
                      |          |
                  Dequantize Dequantize
                         \     /
                          MatMul
                             |
                         Transpose
                             |
                Reorder|StaticReshape
                             |
                          Quantize
                             |
                        [output](int8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_mha_pattern_alternative)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::quantized_mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_query");

                    auto dequantize_key = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_key");

                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, dequantize_query, 0),
                                    in_edge(1, dequantize_key, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto quantize_softmax = pgraph->append_op(
                            graph::op_kind::Quantize, {in_edge(0, softmax, 0)},
                            "quantize_softmax");
                    auto dequantize_softmax
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)},
                                    "dequantize_softmax");

                    auto dequantize_value = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_value");
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, dequantize_softmax, 0),
                                    in_edge(1, dequantize_value, 0)},
                            "matmul_v");

                    auto transpose_output = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");
                    auto reshape_reorder_output = pgraph->append_alternation(
                            {graph::op_kind::Reorder,
                                    graph::op_kind::StaticReshape},
                            {in_edge(0, transpose_output, 0)},
                            "reshape_reorder_output");
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, reshape_reorder_output, 0)},
                            "quantize_output");
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
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_bf16_mha_pattern_alternative)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::quantized_mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_query");
                    auto cast_query = pgraph->append_op(
                            graph::op_kind::TypeCast,
                            {in_edge(0, dequantize_query, 0)}, "cast_query");

                    auto dequantize_key = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_key");
                    auto cast_key = pgraph->append_op(graph::op_kind::TypeCast,
                            {in_edge(0, dequantize_key, 0)}, "cast_key");

                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, cast_query, 0),
                                    in_edge(1, cast_key, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto cast_softmax_fp32 = pgraph->append_op(
                            graph::op_kind::TypeCast, {in_edge(0, softmax, 0)},
                            "cast_softmax_fp32");
                    auto quantize_softmax
                            = pgraph->append_op(graph::op_kind::Quantize,
                                    {in_edge(0, cast_softmax_fp32, 0)},
                                    "quantize_softmax");
                    auto dequantize_softmax
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)},
                                    "dequantize_softmax");
                    auto cast_softmax
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_softmax, 0)},
                                    "cast_softmax");

                    auto dequantize_value = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_value");
                    auto cast_value = pgraph->append_op(
                            graph::op_kind::TypeCast,
                            {in_edge(0, dequantize_value, 0)}, "cast_value");

                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, cast_softmax, 0),
                                    in_edge(1, cast_value, 0)},
                            "matmul_v");
                    auto transpose_output = pgraph->append_op(
                            graph::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");
                    auto reshape_reorder_output = pgraph->append_alternation(
                            {graph::op_kind::Reorder,
                                    graph::op_kind::StaticReshape},
                            {in_edge(0, transpose_output, 0)},
                            "reshape_reorder_output");
                    auto cast_output_fp32
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, reshape_reorder_output, 0)},
                                    "cast_output_fp32");
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, cast_output_fp32, 0)},
                            "quantize_output");
                });

/*
        (int8)[Query]   [Key](int8)
                 |          |
             Dequantize Dequantize
                   \     /
                    MatMul
                      |
                   Softmax
                      |
                   Quantize  [Value](int8)
                      |          |
                  Dequantize Dequantize
                         \     /
                          MatMul
                            |
                         Quantize
                            |
                        [output](int8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_mha_pattern_alternative3)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::quantized_mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_query");
                    auto dequantize_key = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_key");

                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, dequantize_query, 0),
                                    in_edge(1, dequantize_key, 0)},
                            "matmul_qk");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, matmul_qk, 0)}, "softmax");
                    auto quantize_softmax = pgraph->append_op(
                            graph::op_kind::Quantize, {in_edge(0, softmax, 0)},
                            "quantize_softmax");
                    auto dequantize_softmax
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)},
                                    "dequantize_softmax");

                    auto dequantize_value = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_value");

                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, dequantize_softmax, 0),
                                    in_edge(1, dequantize_value, 0)},
                            "matmul_v");
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, matmul_v, 0)}, "quantize_output");
                });

/*
        (int8)[Query]    [Key](int8)
                 |          |
             Dequantize Dequantize
                 |          |
              TypeCast   TypeCast
                   \     /
                    MatMul
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
                          TypeCast
                             |
                          Quantize
                             |
                        [output](int8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_bf16_mha_pattern_alternative3)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::quantized_mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_query");
                    auto cast_query = pgraph->append_op(
                            graph::op_kind::TypeCast,
                            {in_edge(0, dequantize_query, 0)}, "cast_query");

                    auto dequantize_key = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_key");
                    auto cast_key = pgraph->append_op(graph::op_kind::TypeCast,
                            {in_edge(0, dequantize_key, 0)}, "cast_key");

                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, cast_query, 0),
                                    in_edge(1, cast_key, 0)},
                            "matmul_qk");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, matmul_qk, 0)}, "softmax");
                    auto cast_softmax_fp32 = pgraph->append_op(
                            graph::op_kind::TypeCast, {in_edge(0, softmax, 0)},
                            "cast_softmax_fp32");
                    auto quantize_softmax
                            = pgraph->append_op(graph::op_kind::Quantize,
                                    {in_edge(0, cast_softmax_fp32, 0)},
                                    "quantize_softmax");
                    auto dequantize_softmax
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)},
                                    "dequantize_softmax");
                    auto cast_softmax
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_softmax, 0)},
                                    "cast_softmax");

                    auto dequantize_value = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_value");
                    auto cast_value = pgraph->append_op(
                            graph::op_kind::TypeCast,
                            {in_edge(0, dequantize_value, 0)}, "cast_value");

                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, cast_softmax, 0),
                                    in_edge(1, cast_value, 0)},
                            "matmul_v");
                    auto cast_output
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, matmul_v, 0)}, "cast_output");
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, cast_output, 0)}, "quantize_output");
                });

/*
   (int8)[Query]     [Key](int8)
            |          |
      Dequantize   Dequantize
              \     /
               MatMul
                 |
         Divide|Multiply
                 |
                Add
                 |
              Softmax
                 |
             Quantize (optional)
                 |
            [output](int8/f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_matmul_softmax_fusion)
        .set_priority(4.1f)
        .set_kind(graph::partition_kind_t::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_query");

                    auto dequantize_key = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_key");
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, dequantize_query, 0),
                                    in_edge(1, dequantize_key, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");

                    auto optional_quantize_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_quantize_subgraph");
                    auto optional_quantize
                            = optional_quantize_subgraph->append_op(
                                    graph::op_kind::Quantize,
                                    "optional_quantize");
                    optional_quantize_subgraph->create_input_port(
                            0, optional_quantize, 0);
                    optional_quantize_subgraph->create_output_port(
                            0, optional_quantize, 0);
                    pgraph->append_optional(optional_quantize_subgraph,
                            {in_edge(0, softmax, 0)}, "quantize_softmax");
                });

/*
   (int8)[Query]     [Key](int8)
            |          |
      Dequantize   Dequantize
            |          |
       TypeCast     TypeCast
              \     /
               MatMul
                 |
          Divide|Multiply
                 |
                Add
                 |
              Softmax
                 |
              TypeCast (optional)
                 |
              Quantize (optional)
                 |
            [output](int8/bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_bf16_matmul_softmax_fusion)
        .set_priority(4.1f)
        .set_kind(graph::partition_kind_t::matmul_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_query");
                    auto cast_query = pgraph->append_op(
                            graph::op_kind::TypeCast,
                            {in_edge(0, dequantize_query, 0)}, "cast_query");

                    auto dequantize_key = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_key");
                    auto cast_key = pgraph->append_op(graph::op_kind::TypeCast,
                            {in_edge(0, dequantize_key, 0)}, "cast_key");

                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, cast_query, 0),
                                    in_edge(1, cast_key, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");

                    auto optional_output_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_output_subgraph");
                    auto optional_typecast
                            = optional_output_subgraph->append_op(
                                    graph::op_kind::TypeCast,
                                    "optional_typecast");
                    auto optional_quantize
                            = optional_output_subgraph->append_op(
                                    graph::op_kind::Quantize,
                                    {in_edge(0, optional_typecast, 0)},
                                    "optional_quantize");

                    optional_output_subgraph->create_input_port(
                            0, optional_typecast, 0);
                    optional_output_subgraph->create_output_port(
                            0, optional_quantize, 0);
                    pgraph->append_optional(optional_output_subgraph,
                            {in_edge(0, softmax, 0)}, "cast_softmax");
                });

// int8 version of MHA pattern with special view and max
// (a.k.a int8_mha_pattern_alternative4)
/*
    (int8)[Query]   [Key](int8)
            |          |
      Dequantize   Dequantize
              \     /
               MatMul
                 |
              Reshape  [Attention Mask](f32)
                  \    /
                   Add  [Max In](f32)
                    \   /
                     Max
                      |
                   Reshape
                      |
                   Softmax
                      |
                  Quantize   [Value](int8)
                      |         |
                 Dequantize   Dequantize
                         \     /
                          MatMul
                            |
                         Quantize
                            |
                       [output](int8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_mha_pattern_alternative4)
        .set_priority(5.0f)
        .set_kind(graph::partition_kind_t::quantized_mha)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_query");
                    auto dequantize_key = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_key");

                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, dequantize_query, 0),
                                    in_edge(1, dequantize_key, 0)},
                            "matmul_qk");
                    auto reshape1
                            = pgraph->append_op(graph::op_kind::StaticReshape,
                                    {in_edge(0, matmul_qk, 0)}, "reshape1");
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, reshape1, 0)}, "fscore_add");
                    auto fscore_max = pgraph->append_op(graph::op_kind::Maximum,
                            {in_edge(0, fscore_add, 0)}, "fscore_max");
                    auto reshape2
                            = pgraph->append_op(graph::op_kind::StaticReshape,
                                    {in_edge(0, fscore_max, 0)}, "reshape2");
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, reshape2, 0)}, "softmax");
                    auto quantize_softmax = pgraph->append_op(
                            graph::op_kind::Quantize, {in_edge(0, softmax, 0)},
                            "quantize_softmax");
                    auto dequantize_softmax
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)},
                                    "dequantize_softmax");
                    auto dequantize_value = pgraph->append_op(
                            graph::op_kind::Dequantize, "dequantize_value");
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, dequantize_softmax, 0),
                                    in_edge(1, dequantize_value, 0)},
                            "matmul_v");
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, matmul_v, 0)}, "quantize_output");
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
