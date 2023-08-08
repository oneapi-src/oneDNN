/*******************************************************************************
* Copyright 2023 Intel Corporation
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

// To be more specific,
// we use "sdp", the acronym of "scaled_dot_product"
// to replace former name "mha"
DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(sdp)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, float_sdp_fusion)
        .set_priority(21.0f)
        .set_kind(partition_kind_t::sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul);
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)});
                    auto fscore_add = pgraph->append_op(
                            graph::op_kind::Add, {in_edge(0, fscore_scale, 0)});
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)});
                    auto matmul_v = pgraph->append_op(
                            graph::op_kind::MatMul, {in_edge(0, softmax, 0)});
                    auto transpose_output
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    {in_edge(0, matmul_v, 0)});
                    pgraph->append_alternation(
                            {graph::op_kind::Reorder,
                                    graph::op_kind::StaticReshape},
                            {in_edge(0, transpose_output, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, int8_sdp_fusion)
        .set_priority(22.0f)
        .set_kind(partition_kind_t::quantized_sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    auto dequantize_key
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            in_edges_t {in_edge(0, dequantize_query, 0),
                                    in_edge(1, dequantize_key, 0)});
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            in_edges_t {in_edge(0, matmul_qk, 0)});
                    auto fscore_add = pgraph->append_op(graph::op_kind::Add,
                            in_edges_t {in_edge(0, fscore_scale, 0)});
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            in_edges_t {in_edge(0, fscore_add, 0)});
                    auto quantize_softmax
                            = pgraph->append_op(graph::op_kind::Quantize,
                                    in_edges_t {in_edge(0, softmax, 0)});
                    auto dequantize_softmax = pgraph->append_op(
                            graph::op_kind::Dequantize,
                            in_edges_t {in_edge(0, quantize_softmax, 0)});

                    auto dequantize_value
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            in_edges_t {in_edge(0, dequantize_softmax, 0),
                                    in_edge(1, dequantize_value, 0)});

                    auto transpose_output
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, matmul_v, 0)});
                    auto reshape_reorder_output = pgraph->append_alternation(
                            {graph::op_kind::Reorder,
                                    graph::op_kind::StaticReshape},
                            {in_edge(0, transpose_output, 0)});
                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, reshape_reorder_output, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, int8_bf16_sdp_fusion)
        .set_priority(22.0f)
        .set_kind(partition_kind_t::quantized_sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_query
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    auto cast_query
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_query, 0)});

                    auto dequantize_key
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    auto cast_key = pgraph->append_op(graph::op_kind::TypeCast,
                            {in_edge(0, dequantize_key, 0)});

                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, cast_query, 0),
                                    in_edge(1, cast_key, 0)});
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)});
                    auto fscore_add = pgraph->append_op(
                            graph::op_kind::Add, {in_edge(0, fscore_scale, 0)});
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, fscore_add, 0)});
                    auto cast_softmax_fp32 = pgraph->append_op(
                            graph::op_kind::TypeCast, {in_edge(0, softmax, 0)});
                    auto quantize_softmax
                            = pgraph->append_op(graph::op_kind::Quantize,
                                    {in_edge(0, cast_softmax_fp32, 0)});
                    auto dequantize_softmax
                            = pgraph->append_op(graph::op_kind::Dequantize,
                                    {in_edge(0, quantize_softmax, 0)});
                    auto cast_softmax
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_softmax, 0)});

                    auto dequantize_value
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    auto cast_value
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, dequantize_value, 0)});

                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, cast_softmax, 0),
                                    in_edge(1, cast_value, 0)});
                    auto transpose_output
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    {in_edge(0, matmul_v, 0)});
                    auto reshape_reorder_output = pgraph->append_alternation(
                            {graph::op_kind::Reorder,
                                    graph::op_kind::StaticReshape},
                            {in_edge(0, transpose_output, 0)});
                    auto cast_output_fp32
                            = pgraph->append_op(graph::op_kind::TypeCast,
                                    {in_edge(0, reshape_reorder_output, 0)});
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, cast_output_fp32, 0)});
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
