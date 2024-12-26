/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "graph/backend/dnnl/kernels/decomp/mqa_decomp.hpp"
#include "graph/backend/dnnl/kernels/decomp/sdp_decomp.hpp"
#include "graph/backend/dnnl/kernels/prim/large_partition.hpp"
#include "graph/backend/dnnl/kernels/prim/matmul.hpp"
#include "graph/backend/dnnl/kernels/uker/sdp_primitive.hpp"

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

/*
 [query]    [key]
      \     /
[cond] MatMul [mask]
     \   |   /
       Select   [scale]
            \   /
             Div   [add in]
               \   /
                Add
                 |
               Softmax   [value]
                    \     /
                     MatMul
                       |
                StaticTranspose
                       |
                    Reorder
                       |
                    [output]
*/
void create_gpt_sdp(
        const std::shared_ptr<pb_graph_t> &pgraph, bool is_bf16, bool is_int8) {
    auto matmul_qk = create_dequant_matmul(pgraph, nullptr, is_bf16, is_int8);
    auto select = pgraph->append_op(
            graph::op_kind::Select, {in_edge(1, matmul_qk, 0)});
    auto div = pgraph->append_op(
            graph::op_kind::Divide, {in_edge(0, select, 0)});
    auto add = pgraph->append_op(graph::op_kind::Add, {in_edge(0, div, 0)});
    auto softmax
            = pgraph->append_op(graph::op_kind::SoftMax, {in_edge(0, add, 0)});

    pm::pb_node_t *matmul_v_input = softmax;

    if (is_bf16) {
        // This means [0,3) TypeCast ops after softmax
        // For int8-bf16 pattern, there is 1 TypeCast op after softmax
        // For bf16 pattern, there is 2 TypeCast ops after softmax
        auto extra_casts = append_siso_repetition_subgraph(
                pgraph, graph::op_kind::TypeCast, softmax, 0, 3);
        matmul_v_input = extra_casts;
    }

    if (is_int8) {
        auto quantize_softmax = pgraph->append_op(
                graph::op_kind::Quantize, {in_edge(0, matmul_v_input, 0)});
        matmul_v_input = quantize_softmax;
    }

    auto matmul_v
            = create_dequant_matmul(pgraph, matmul_v_input, is_bf16, is_int8);
    auto transpose_output = pgraph->append_op(
            graph::op_kind::StaticTranspose, {in_edge(0, matmul_v, 0)});
    auto reorder_output = pgraph->append_alternation(
            {graph::op_kind::Reorder, graph::op_kind::StaticReshape},
            {in_edge(0, transpose_output, 0)});
    if (is_int8) {
        append_optional_typecast_quantize(pgraph, reorder_output, is_bf16);
    }
}

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(sdp)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, float_sdp_fusion)
        .set_priority(21.0f)
        .set_kind(partition_kind_t::sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul);

                    std::shared_ptr<pb_graph_t> scale_graph;
                    scale_graph = std::make_shared<pb_graph_t>();
                    auto scale = scale_graph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply});
                    scale_graph->create_input_port(0, scale, 0);
                    scale_graph->create_output_port(0, scale, 0);
                    auto optional_scale = pgraph->append_optional(
                            scale_graph, {in_edge(0, matmul_qk, 0)});

                    auto optional_mask = std::make_shared<pb_graph_t>();
                    auto fscore_add
                            = optional_mask->append_op(graph::op_kind::Add);
                    optional_mask->create_input_port(0, fscore_add, 0);
                    optional_mask->create_output_port(0, fscore_add, 0);
                    auto mask = pgraph->append_optional(
                            optional_mask, {in_edge(0, optional_scale, 0)});

                    // Optional select for distilbert
                    auto p_select2 = optional_select(pgraph, mask, 2);
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, p_select2, 0)});
                    auto matmul_v = pgraph->append_op(
                            graph::op_kind::MatMul, {in_edge(0, softmax, 0)});
                    // Optional transpose + reshape/reorder
                    optional_transpose_reshape(pgraph, matmul_v, 0);
                })
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_primitive_kernel_t<>>();
                }))
         DNNL_CPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_decomp_kernel_t<>>();
                }))
        // clang-format on
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, float_gqa_fusion)
        .set_priority(21.1f)
        .set_kind(partition_kind_t::sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto reshape1
                            = pgraph->append_op(graph::op_kind::StaticReshape);
                    auto reshape2
                            = pgraph->append_op(graph::op_kind::StaticReshape);
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {{in_edge(0, reshape1, 0),
                                    in_edge(1, reshape2, 0)}});
                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)});
                    auto optional_mask = std::make_shared<pb_graph_t>();
                    auto mask_reshape = optional_mask->append_op(
                            graph::op_kind::StaticReshape);
                    auto fscore_add = optional_mask->append_op(
                            graph::op_kind::Add, {in_edge(1, mask_reshape, 0)});
                    optional_mask->create_input_port(0, fscore_add, 0);
                    optional_mask->create_output_port(0, fscore_add, 0);
                    auto mask = pgraph->append_optional(
                            optional_mask, {in_edge(0, fscore_scale, 0)});

                    // Optional select for distilbert
                    auto p_select2 = optional_select(pgraph, mask, 2);
                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, p_select2, 0)});
                    auto reshape3
                            = pgraph->append_op(graph::op_kind::StaticReshape);
                    auto matmul_v = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, softmax, 0), in_edge(1, reshape3, 0)});
                    auto reshape4
                            = pgraph->append_op(graph::op_kind::StaticReshape,
                                    {in_edge(0, matmul_v, 0)});

                    // Optional transpose + reshape/reorder
                    optional_transpose_reshape(pgraph, reshape4, 0);
                })
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_primitive_kernel_t<>>();
                }))
        DNNL_CPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return DNNL_CPU_ONLY(std::make_shared<sdp_decomp_kernel_t<>>());
                }))
        // clang-format on
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, float_sdp_jax_fusion)
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
                    auto s_transpose
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, softmax, 0)});
                    auto s_reorder = pgraph->append_op(graph::op_kind::Reorder,
                            in_edges_t {in_edge(0, s_transpose, 0)});
                    auto matmul_v = pgraph->append_op(
                            graph::op_kind::MatMul, {in_edge(1, s_reorder, 0)});
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

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, float_mqa_jax_fusion)
        .set_priority(21.0f)
        .set_kind(partition_kind_t::sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul);
                    auto reshape1
                            = pgraph->append_op(graph::op_kind::StaticReshape,
                                    {in_edge(0, matmul_qk, 0)});
                    auto transpose1
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    {in_edge(0, reshape1, 0)});
                    auto post_add = pgraph->append_op(
                            graph::op_kind::Add, {in_edge(0, transpose1, 0)});
                    auto softmax = pgraph->append_op(
                            graph::op_kind::SoftMax, {in_edge(0, post_add, 0)});
                    auto transpose2
                            = pgraph->append_op(graph::op_kind::StaticTranspose,
                                    {in_edge(0, softmax, 0)});
                    auto reshape2
                            = pgraph->append_op(graph::op_kind::StaticReshape,
                                    {in_edge(0, transpose2, 0)});
                    pgraph->append_op(
                            graph::op_kind::MatMul, {in_edge(1, reshape2, 0)});
                })
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<mqa_decomp_kernel_t<>>();
                }))
        // clang-format on
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

                    std::shared_ptr<pb_graph_t> scale_graph;
                    scale_graph = std::make_shared<pb_graph_t>();
                    auto scale = scale_graph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply});
                    scale_graph->create_input_port(0, scale, 0);
                    scale_graph->create_output_port(0, scale, 0);
                    auto optional_scale = pgraph->append_optional(
                            scale_graph, {in_edge(0, matmul_qk, 0)});

                    auto optional_mask = std::make_shared<pb_graph_t>();
                    auto fscore_add
                            = optional_mask->append_op(graph::op_kind::Add);
                    optional_mask->create_input_port(0, fscore_add, 0);
                    optional_mask->create_output_port(0, fscore_add, 0);
                    auto mask = pgraph->append_optional(
                            optional_mask, {in_edge(0, optional_scale, 0)});

                    // Optional select for distilbert
                    auto p_select2 = optional_select(pgraph, mask, 2);

                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            in_edges_t {in_edge(0, p_select2, 0)});
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
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_primitive_kernel_t<true>>();
                }))
        DNNL_CPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_decomp_kernel_t<true>>();
                }))
        // clang-format on
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

                    std::shared_ptr<pb_graph_t> scale_graph;
                    scale_graph = std::make_shared<pb_graph_t>();
                    auto scale = scale_graph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply});
                    scale_graph->create_input_port(0, scale, 0);
                    scale_graph->create_output_port(0, scale, 0);
                    auto optional_scale = pgraph->append_optional(
                            scale_graph, {in_edge(0, matmul_qk, 0)});

                    auto optional_mask = std::make_shared<pb_graph_t>();
                    auto fscore_add
                            = optional_mask->append_op(graph::op_kind::Add);
                    optional_mask->create_input_port(0, fscore_add, 0);
                    optional_mask->create_output_port(0, fscore_add, 0);
                    auto mask = pgraph->append_optional(
                            optional_mask, {in_edge(0, optional_scale, 0)});

                    // Optional select for distilbert
                    auto p_select2 = optional_select(pgraph, mask, 2);

                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, p_select2, 0)});
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
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_primitive_kernel_t<true>>();
                }))
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<
                            sdp_decomp_kernel_t<true, memory::data_type::bf16>>();
                }))
        // clang-format on
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, sdp_with_compressed_kv_fusion)
        .set_priority(22.0f)
        .set_kind(partition_kind_t::quantized_sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_key = pgraph->append_op(
                            graph::op_kind::DynamicDequantize);
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(1, dequantize_key, 0)});

                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)});

                    auto optional_mask = std::make_shared<pb_graph_t>();
                    auto fscore_add
                            = optional_mask->append_op(graph::op_kind::Add);
                    optional_mask->create_input_port(0, fscore_add, 0);
                    optional_mask->create_output_port(0, fscore_add, 0);
                    auto mask = pgraph->append_optional(
                            optional_mask, {in_edge(0, fscore_scale, 0)});

                    // Optional select for distilbert
                    auto p_select2 = optional_select(pgraph, mask, 2);

                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, p_select2, 0)});
                    auto dequantize_value = pgraph->append_op(
                            graph::op_kind::DynamicDequantize);
                    pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, softmax, 0),
                                    in_edge(1, dequantize_value, 0)});
                })
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_primitive_kernel_t<true>>();
                }))
        DNNL_CPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<
                            sdp_decomp_kernel_t<true, memory::data_type::bf16>>();
                }))
        // clang-format on
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, sdp_with_compressed_v_fusion)
        .set_priority(22.0f)
        .set_kind(partition_kind_t::quantized_sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul);

                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)});

                    auto optional_mask = std::make_shared<pb_graph_t>();
                    auto fscore_add
                            = optional_mask->append_op(graph::op_kind::Add);
                    optional_mask->create_input_port(0, fscore_add, 0);
                    optional_mask->create_output_port(0, fscore_add, 0);
                    auto mask = pgraph->append_optional(
                            optional_mask, {in_edge(0, fscore_scale, 0)});

                    // Optional select for distilbert
                    auto p_select2 = optional_select(pgraph, mask, 2);

                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, p_select2, 0)});
                    auto dequantize_value = pgraph->append_op(
                            graph::op_kind::DynamicDequantize);
                    pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(0, softmax, 0),
                                    in_edge(1, dequantize_value, 0)});
                })
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_primitive_kernel_t<true>>();
                }))
        DNNL_CPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<
                            sdp_decomp_kernel_t<true, memory::data_type::bf16>>();
                }))
        // clang-format on
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, sdp_with_compressed_k_fusion)
        .set_priority(22.0f)
        .set_kind(partition_kind_t::quantized_sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto dequantize_key = pgraph->append_op(
                            graph::op_kind::DynamicDequantize);
                    auto matmul_qk = pgraph->append_op(graph::op_kind::MatMul,
                            {in_edge(1, dequantize_key, 0)});

                    auto fscore_scale = pgraph->append_alternation(
                            {graph::op_kind::Divide, graph::op_kind::Multiply},
                            {in_edge(0, matmul_qk, 0)});

                    auto optional_mask = std::make_shared<pb_graph_t>();
                    auto fscore_add
                            = optional_mask->append_op(graph::op_kind::Add);
                    optional_mask->create_input_port(0, fscore_add, 0);
                    optional_mask->create_output_port(0, fscore_add, 0);
                    auto mask = pgraph->append_optional(
                            optional_mask, {in_edge(0, fscore_scale, 0)});

                    // Optional select for distilbert
                    auto p_select2 = optional_select(pgraph, mask, 2);

                    auto softmax = pgraph->append_op(graph::op_kind::SoftMax,
                            {in_edge(0, p_select2, 0)});
                    pgraph->append_op(
                            graph::op_kind::MatMul, {in_edge(0, softmax, 0)});
                })
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_primitive_kernel_t<true>>();
                }))
        DNNL_CPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<
                            sdp_decomp_kernel_t<true, memory::data_type::bf16>>();
                }))
        // clang-format on
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, float_gpt_sdp)
        .set_priority(22.0f)
        .set_kind(partition_kind_t::sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_sdp(pgraph, /*bf16=*/false, /*int8=*/false);
                })
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_primitive_kernel_t<>>();
                }))
        DNNL_CPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_decomp_kernel_t<>>();
                }))
        // clang-format on
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, bfloat16_gpt_sdp)
        .set_priority(22.0f)
        .set_kind(partition_kind_t::sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_sdp(pgraph, /*bf16=*/true, /*int8=*/false);
                })
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_primitive_kernel_t<>>();
                }))
        DNNL_CPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_decomp_kernel_t<>>();
                }))
        // clang-format on
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, int8_fp32_gpt_sdp)
        .set_priority(22.0f)
        .set_kind(partition_kind_t::quantized_sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_sdp(pgraph, /*bf16=*/false, /*int8=*/true);
                })
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_primitive_kernel_t<true>>();
                }))
        DNNL_CPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_decomp_kernel_t<true>>();
                }))
        // clang-format on
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, int8_bf16_gpt_sdp)
        .set_priority(22.0f)
        .set_kind(partition_kind_t::quantized_sdp)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    create_gpt_sdp(pgraph, /*bf16=*/true, /*int8=*/true);
                })
        // clang-format off
        DNNL_GPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<sdp_primitive_kernel_t<true>>();
                }))
        DNNL_CPU_ONLY(.set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr {
                    return std::make_shared<
                            sdp_decomp_kernel_t<true, memory::data_type::bf16>>();
                }))
        // clang-format on    
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });
DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
