/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "backend/graph_compiler/patterns/fusions.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {
namespace pass {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = impl::utils::pm::pb_graph_t;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

// [rep_min, rep_max)
pm::repetition_t *create_append_transpose_repetition_subgraph(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        int rep_min, int rep_max) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    auto transpose_subgraph
            = std::make_shared<pb_graph_t>("transpose_subgraph");
    auto transpose_rep = transpose_subgraph->append_op(
            impl::op_kind::StaticTranspose, "transpose_rep");
    transpose_subgraph->create_input_port(0, transpose_rep, 0);
    transpose_subgraph->create_output_port(0, transpose_rep, 0);
    auto transpose = pgraph->append_repetition(transpose_subgraph, {0, 0},
            rep_min, rep_max, in_edges, "transpose");
    return transpose;
};

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
        .set_kind(impl::partition_kind::mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "query_reshape");
                    query_reshape->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto query_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, query_reshape, 1, 3);
                    auto key_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, key_reshape, 1, 3);
                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, query_transpose, 0),
                                    in_edge(1, key_transpose, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto value_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, value_reshape, 1, 3);
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, softmax, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");
                    auto transpose_output = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");

                    auto optional_reshape_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_reshape_subgraph");
                    auto optional_reshape
                            = optional_reshape_subgraph->append_op(
                                    impl::op_kind::StaticReshape,
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
                     Reorder
                        |
                     [output](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_mha_pattern_alternative)
        .set_priority(5.0f)
        .set_kind(impl::partition_kind::mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
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
                    pgraph->append_op(impl::op_kind::Reorder,
                            {in_edge(0, transpose_output, 0)},
                            "reorder_output");
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
                  Softmax [Dropout](f32)
                       \  /
                        Mul  [ValueTrans](f32)
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
        compiler, fp32_mha_forward_pattern)
        .set_priority(5.0f)
        .set_kind(impl::partition_kind::mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            impl::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto fscore_scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    fscore_scale->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    fscore_add->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    softmax->allow_external_output(0);
                    auto dropout = pgraph->append_op(impl::op_kind::Multiply,
                            {in_edge(0, softmax, 0)}, "dropout");
                    dropout->allow_external_output(0);
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, dropout, 0)}, "matmul_v");
                    matmul_v->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto reshape
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    {in_edge(0, matmul_v, 0)}, "reshape");
                    pgraph->append_op(impl::op_kind::StaticReshape,
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
            MatMul           MatMul  [Dropout](f32)
              |                  \  /
        [output](f32)            Mul [SoftmaxOut](f32)
                                /  \    /
                               /     Mul
                               |      |
                               |  ReduceSum
                                \   /
                                 Sub  [SoftmaxOut](f32)
                                  \   /
                                   Mul  [Fscore](f32)
                                    \  /
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
        .set_kind(impl::partition_kind::mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto in_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "in_reshape");
                    in_reshape->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto in_transpose = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            {in_edge(0, in_reshape, 0)}, "in_transpose");
                    auto bmm_v_grad_weight = pgraph->append_op(
                            impl::op_kind::MatMul,
                            {in_edge(1, in_transpose, 0)}, "bmm_v_grad_weight");
                    bmm_v_grad_weight->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto bmm_v_grad_data = pgraph->append_op(
                            impl::op_kind::MatMul,
                            {in_edge(0, in_transpose, 0)}, "bmm_v_grad_data");
                    bmm_v_grad_data->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto dropout_grad = pgraph->append_op(
                            impl::op_kind::Multiply,
                            {in_edge(0, bmm_v_grad_data, 0)}, "dropout_grad");
                    dropout_grad->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto softmax_mul = pgraph->append_op(
                            impl::op_kind::Multiply,
                            {in_edge(0, dropout_grad, 0)}, "softmax_mul");
                    softmax_mul->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto softmax_sum = pgraph->append_op(
                            impl::op_kind::ReduceSum,
                            {in_edge(0, softmax_mul, 0)}, "softmax_sum");
                    softmax_sum->append_decision_function(check_reduce_attrs);
                    auto softmax_sub
                            = pgraph->append_op(impl::op_kind::Subtract,
                                    {in_edge(0, dropout_grad, 0),
                                            in_edge(1, softmax_sum, 0)},
                                    "softmax_sub");
                    /* Create 2 subgraph for alternation */
                    auto successive_mul_subgraph = std::make_shared<pb_graph_t>(
                            "successive_mul_subgraph");
                    auto softmax_mul2 = successive_mul_subgraph->append_op(
                            impl::op_kind::Multiply, "softmax_mul2");
                    auto fscore_grad_alter1
                            = successive_mul_subgraph->append_alternation(
                                    {impl::op_kind::Divide,
                                            impl::op_kind::Multiply},
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
                                    {impl::op_kind::Divide,
                                            impl::op_kind::Multiply},
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
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    {in_edge(0, softmax_grad_alter, 0)},
                                    "bmm_q_grad_weight");
                    bmm_q_grad_weight->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
                    auto bmm_k_grad_weight
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    {in_edge(0, softmax_grad_alter, 0)},
                                    "bmm_k_grad_weight");
                    bmm_k_grad_weight->append_decision_function(
                            check_input_dtype<impl::data_type::f32>);
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
        .set_kind(impl::partition_kind::mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "query_reshape");
                    query_reshape->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    auto query_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, query_reshape, 1, 3);
                    auto key_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, key_reshape, 1, 3);
                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, query_transpose, 0),
                                    in_edge(1, key_transpose, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto value_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, value_reshape, 1, 3);
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, softmax, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");
                    auto transpose_output = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");

                    auto optional_reshape_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_reshape_subgraph");
                    auto optional_reshape
                            = optional_reshape_subgraph->append_op(
                                    impl::op_kind::StaticReshape,
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
                     Reorder
                        |
                     [output](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mha_pattern_alternative)
        .set_priority(5.0f)
        .set_kind(impl::partition_kind::mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            impl::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
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
                            check_input_dtype<impl::data_type::bf16>);
                    auto transpose_output = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");
                    pgraph->append_op(impl::op_kind::Reorder,
                            {in_edge(0, transpose_output, 0)},
                            "reorder_output");
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
                  Softmax [Dropout](bf16)
                       \  /
                        Mul  [ValueTrans](bf16)
                            \     /
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
        .set_kind(impl::partition_kind::mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(
                            impl::op_kind::MatMul, "matmul_qk");
                    matmul_qk->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    auto fscore_scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    fscore_add->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    softmax->allow_external_output(0);
                    auto dropout = pgraph->append_op(impl::op_kind::Multiply,
                            {in_edge(0, softmax, 0)}, "dropout");
                    dropout->allow_external_output(0);
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, dropout, 0)}, "matmul_v");
                    matmul_v->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    auto reshape
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    {in_edge(0, matmul_v, 0)}, "reshape");
                    pgraph->append_op(impl::op_kind::StaticReshape,
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
            MatMul           MatMul  [Dropout](bf16)
              |                  \  /
        [output](bf16)           Mul [SoftmaxOut](bf16)
                                /  \    /
                               /     Mul
                               |      |
                               |  ReduceSum
                                \   /
                                 Sub  [SoftmaxOut](bf16)
                                  \   /
                                   Mul  [Fscore](f32/bf16)
                                    \  /
                                   Div|Mul  [QueryTrans](bf16)
                 ___________________/    \   /
                 \   [KeyTrans](bf16)    MatMul
                  \      /                 |
                   MatMul             [output](bf16)
                     |
                 [output](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_mha_backward_pattern)
        .set_priority(5.0f)
        .set_kind(impl::partition_kind::mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto in_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "in_reshape");
                    in_reshape->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    auto in_transpose = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            {in_edge(0, in_reshape, 0)}, "in_transpose");
                    auto bmm_v_grad_weight = pgraph->append_op(
                            impl::op_kind::MatMul,
                            {in_edge(1, in_transpose, 0)}, "bmm_v_grad_weight");
                    bmm_v_grad_weight->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    auto bmm_v_grad_data = pgraph->append_op(
                            impl::op_kind::MatMul,
                            {in_edge(0, in_transpose, 0)}, "bmm_v_grad_data");
                    bmm_v_grad_data->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    auto dropout_grad = pgraph->append_op(
                            impl::op_kind::Multiply,
                            {in_edge(0, bmm_v_grad_data, 0)}, "dropout_grad");
                    dropout_grad->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    auto softmax_mul = pgraph->append_op(
                            impl::op_kind::Multiply,
                            {in_edge(0, dropout_grad, 0)}, "softmax_mul");
                    softmax_mul->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    auto softmax_sum = pgraph->append_op(
                            impl::op_kind::ReduceSum,
                            {in_edge(0, softmax_mul, 0)}, "softmax_sum");
                    softmax_sum->append_decision_function(check_reduce_attrs);
                    auto softmax_sub
                            = pgraph->append_op(impl::op_kind::Subtract,
                                    {in_edge(0, dropout_grad, 0),
                                            in_edge(1, softmax_sum, 0)},
                                    "softmax_sub");
                    /* Create 2 subgraph for alternation */
                    auto successive_mul_subgraph = std::make_shared<pb_graph_t>(
                            "successive_mul_subgraph");
                    auto softmax_mul2 = successive_mul_subgraph->append_op(
                            impl::op_kind::Multiply, "softmax_mul2");
                    auto fscore_grad_alter1
                            = successive_mul_subgraph->append_alternation(
                                    {impl::op_kind::Divide,
                                            impl::op_kind::Multiply},
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
                                    {impl::op_kind::Divide,
                                            impl::op_kind::Multiply},
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
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    {in_edge(0, softmax_grad_alter, 0)},
                                    "bmm_q_grad_weight");
                    bmm_q_grad_weight->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
                    auto bmm_k_grad_weight
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    {in_edge(0, softmax_grad_alter, 0)},
                                    "bmm_k_grad_weight");
                    bmm_k_grad_weight->append_decision_function(
                            check_input_dtype<impl::data_type::bf16>);
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
        .set_kind(impl::partition_kind::quantized_mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_query");
                    auto dequantize_key = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_key");
                    auto dequantize_value = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_value");
                    auto query_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape,
                            {in_edge(0, dequantize_query, 0)}, "query_reshape");
                    auto query_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, query_reshape, 1, 3);
                    auto key_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape,
                            {in_edge(0, dequantize_key, 0)}, "key_reshape");
                    auto key_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, key_reshape, 1, 3);
                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, query_transpose, 0),
                                    in_edge(1, key_transpose, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto quantize_softmax = pgraph->append_op(
                            impl::op_kind::Quantize, {in_edge(0, softmax, 0)},
                            "quantize_softmax");
                    auto dequantize_softmax
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)},
                                    "dequantize_softmax");
                    auto value_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape,
                            {in_edge(0, dequantize_value, 0)}, "value_reshape");
                    auto value_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, value_reshape, 1, 3);
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, dequantize_softmax, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");

                    auto transpose_output = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");

                    auto optional_reshape_subgraph
                            = std::make_shared<pb_graph_t>(
                                    "optional_reshape_subgraph");
                    auto optional_reshape
                            = optional_reshape_subgraph->append_op(
                                    impl::op_kind::StaticReshape,
                                    "optional_reshape");
                    optional_reshape_subgraph->create_input_port(
                            0, optional_reshape, 0);
                    optional_reshape_subgraph->create_output_port(
                            0, optional_reshape, 0);

                    auto reshape_output
                            = pgraph->append_optional(optional_reshape_subgraph,
                                    {in_edge(0, transpose_output, 0)},
                                    "reshape_output");
                    pgraph->append_op(impl::op_kind::Quantize,
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
        .set_kind(impl::partition_kind::quantized_mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_query");
                    auto typecast_query
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    {in_edge(0, dequantize_query, 0)},
                                    "typecast_query");
                    auto query_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape,
                            {in_edge(0, typecast_query, 0)}, "query_reshape");
                    auto query_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, query_reshape, 1, 3);

                    auto dequantize_key = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_key");
                    auto typecast_key = pgraph->append_op(
                            impl::op_kind::TypeCast,
                            {in_edge(0, dequantize_key, 0)}, "typecast_key");
                    auto key_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape,
                            {in_edge(0, typecast_key, 0)}, "key_reshape");
                    auto key_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, key_reshape, 1, 3);
                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, query_transpose, 0),
                                    in_edge(1, key_transpose, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto typecast_softmax = pgraph->append_op(
                            impl::op_kind::TypeCast, {in_edge(0, softmax, 0)},
                            "typecast_softmax");
                    auto quantize_softmax
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    {in_edge(0, typecast_softmax, 0)},
                                    "quantize_softmax");
                    auto dequantize_softmax
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)},
                                    "dequantize_softmax");
                    auto typecast_softmax_bf16
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    {in_edge(0, dequantize_softmax, 0)},
                                    "typecast_softmax");

                    auto dequantize_value = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_value");
                    auto typecast_value
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    {in_edge(0, dequantize_value, 0)},
                                    "typecast_value");
                    auto value_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape,
                            {in_edge(0, typecast_value, 0)}, "value_reshape");
                    auto value_transpose
                            = create_append_transpose_repetition_subgraph(
                                    pgraph, value_reshape, 1, 3);
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, typecast_softmax_bf16, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");

                    auto transpose_output = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");
                    auto reshape_output
                            = pgraph->append_op(impl::op_kind::StaticReshape,
                                    {in_edge(0, transpose_output, 0)},
                                    "reshape_output");
                    auto typecast_output = pgraph->append_op(
                            impl::op_kind::TypeCast,
                            {in_edge(0, reshape_output, 0)}, "typecast_output");
                    pgraph->append_op(impl::op_kind::Quantize,
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
                          Reorder
                             |
                          Quantize
                             |
                        [output](int8)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, int8_mha_pattern_alternative)
        .set_priority(5.0f)
        .set_kind(impl::partition_kind::quantized_mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_query");

                    auto dequantize_key = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_key");

                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, dequantize_query, 0),
                                    in_edge(1, dequantize_key, 0)},
                            "matmul_qk");
                    auto fscore_scale = pgraph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)}, "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, fscore_scale, 0)}, "fscore_add");
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)}, "softmax");
                    auto quantize_softmax = pgraph->append_op(
                            impl::op_kind::Quantize, {in_edge(0, softmax, 0)},
                            "quantize_softmax");
                    auto dequantize_softmax
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)},
                                    "dequantize_softmax");

                    auto dequantize_value = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequantize_value");
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            {in_edge(0, dequantize_softmax, 0),
                                    in_edge(1, dequantize_value, 0)},
                            "matmul_v");

                    auto transpose_output = pgraph->append_op(
                            impl::op_kind::StaticTranspose,
                            {in_edge(0, matmul_v, 0)}, "transpose_output");
                    auto reorder_output
                            = pgraph->append_op(impl::op_kind::Reorder,
                                    {in_edge(0, transpose_output, 0)},
                                    "reorder_output");
                    pgraph->append_op(impl::op_kind::Quantize,
                            {in_edge(0, reorder_output, 0)}, "quantize_output");
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
                          Reorder
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
        .set_kind(impl::partition_kind::quantized_mha)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
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
                    auto reorder_output
                            = pgraph->append_op(impl::op_kind::Reorder,
                                    {in_edge(0, transpose_output, 0)},
                                    "reorder_output");
                    auto cast_output_fp32
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    {in_edge(0, reorder_output, 0)},
                                    "cast_output_fp32");
                    pgraph->append_op(impl::op_kind::Quantize,
                            {in_edge(0, cast_output_fp32, 0)},
                            "quantize_output");
                });

COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
