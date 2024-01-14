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

#include <iostream>
#include <sstream>
#include "compiler/ir/graph/fusible_op.hpp"
#include "compiler/jit/jit.hpp"
#include "context.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <ops/templates/matmul_core.hpp>
#include <util/reflection.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_graph_reshedule, TestGraphReschedule5) {
    SET_THREADS_OR_SKIP(8);

    auto get_test_graph = []() {
        sc_graph_t mgr;
        auto in_a = mgr.make_input({graph_tensor::make({4, 16, 1, 34},
                sc_data_format_t(format_kinds::ABCD), datatypes::u8)});
        auto in_b = mgr.make_input({graph_tensor::make(
                {4, 16, 34, 256}, sc_data_format_t(format_kinds::ABCD))});

        auto quant_b = mgr.make("quantize", in_b->get_outputs(), {},
                {{"dtype", datatypes::u8},
                        {"scales", std::vector<float> {1.1f}},
                        {"zero_points", std::vector<int> {2}},
                        {"channel_axis", 0}});

        auto dequant_a = mgr.make("dequantize", in_a->get_outputs(), {},
                {{"dtype", datatypes::f32},
                        {"scales", std::vector<float> {1.1f}},
                        {"zero_points", std::vector<int> {0}},
                        {"channel_axis", 0}});
        auto dequant_b = mgr.make("dequantize", quant_b->get_outputs(), {},
                {{"dtype", datatypes::f32},
                        {"scales", std::vector<float> {1.1f}},
                        {"zero_points", std::vector<int> {2}},
                        {"channel_axis", 0}});

        auto matmul = mgr.make("matmul",
                {dequant_a->get_outputs()[0], dequant_b->get_outputs()[0]}, {},
                {});

        auto quant_output = mgr.make("quantize", matmul->get_outputs(), {},
                {{"dtype", datatypes::u8},
                        {"scales", std::vector<float> {1.1f}},
                        {"zero_points", std::vector<int> {2}},
                        {"channel_axis", 0}});
        mgr.make_output(quant_output->get_outputs());
        return mgr;
    };

    {
        sc_graph_t graph = get_test_graph();
        auto ctx = std::make_shared<context_t>(*get_test_ctx());
        graph_driver(graph, ctx);

        std::stringstream ss;
        print_graph(graph, ss, true);
        std::string expected_str_spr
                = R"(graph(v0: u8[4, 16, 1, 34], v1: f32[4, 16, 34, 256]) -> [v2: u8[4, 16, 1, 256]] {
  [v3: s32[1]] = constant([1])
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = mul(v6, v5)
  [v8: f32[1]] = constant([1])
  [v9: f32[1]] = constant([1])
  [v10: u8[4, 16, 34, 256]] = outerloop_4X16X34_partition_mul_add_cast(v1, v9, v8)
  [v2: u8[4, 16, 1, 256]] = outerloop_4X16_partition_cast_reduce_compute_reduce_collect_mul_reorder_reorder_quantized_matmul_core_sub_cast_mul_add_cast(v10, v0, v3, v7, v4)
}
)";
        std::string expected_str_clx
                = R"(graph(v0: u8[4, 16, 1, 34], v1: f32[4, 16, 34, 256]) -> [v2: u8[4, 16, 1, 256]] {
  [v3: s32[1]] = constant([1])
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = mul(v6, v5)
  [v8: f32[1]] = constant([1])
  [v9: f32[1]] = constant([1])
  [v10: s8[4, 16, 34, 256]] = outerloop_4X16X34_partition_mul_add_cast(v1, v9, v8)
  [v2: u8[4, 16, 1, 256]] = outerloop_4X16_partition_cast_reduce_compute_reduce_collect_mul_reorder_reorder_quantized_matmul_core_sub_cast_mul_add_cast(v10, v0, v3, v7, v4)
}
)";
        if (IS_AMX_AVAILABLE()) {
            EXPECT_EQ(ss.str(), expected_str_spr);
        } else {
            EXPECT_EQ(ss.str(), expected_str_clx);
        }
    }
}

static void add_single_matmul_to_graph(sc_graph_t &graph, const sc_dims &A_dims,
        const sc_dims &B_dims, const sc_dims &bias_dims,
        const sc_dims &out_dims) {
    auto A_fmt = A_dims.size() == 3 ? sc_data_format_t(format_kinds::ABC)
                                    : sc_data_format_t(format_kinds::AB);

    auto ins0 = graph.make_input(
            {graph_tensor::make(A_dims, A_fmt, datatypes::u8)});
    auto ins1 = graph.make_input({graph_tensor::make(
            B_dims, sc_data_format_t(format_kinds::AB), datatypes::s8)});
    auto ins2 = graph.make_input({graph_tensor::make(
            bias_dims, sc_data_format_t(format_kinds::A), datatypes::bf16)});
    auto dequant0 = graph.make("dequantize", ins0->get_outputs(), {},
            {{"channel_axis", 1},
                    {"scales", std::vector<float> {0.00391666731f}},
                    {"zero_points", std::vector<int> {0}},
                    {"per_channel", false}, {"dtype", datatypes::f32}});
    auto cast0 = graph.make(
            "cast", dequant0->get_outputs(), {}, {{"dtype", datatypes::bf16}});
    auto dequant1 = graph.make("dequantize", ins1->get_outputs(), {},
            {{"channel_axis", 0},
                    {"scales", std::vector<float>(768, 0.00148682f)},
                    {"zero_points", std::vector<int>(768, 0)},
                    {"per_channel", true}, {"dtype", datatypes::f32}});
    auto cast1 = graph.make(
            "cast", dequant1->get_outputs(), {}, {{"dtype", datatypes::bf16}});
    any_map_t attrs({{"transpose_a", false}, {"transpose_b", false}});
    auto matmul = graph.make("matmul",
            {cast0->get_outputs()[0], cast1->get_outputs()[0],
                    ins2->get_outputs()[0]},
            {}, attrs);
    auto cast2 = graph.make(
            "cast", matmul->get_outputs(), {}, {{"dtype", datatypes::f32}});
    auto outs0 = std::make_shared<graph_tensor>(
            nullptr, A_fmt, out_dims, datatypes::u8);
    auto quant2 = graph.make("quantize", cast2->get_outputs(), {outs0},
            {{"channel_axis", 1},
                    {"scales", std::vector<float> {0.00391666731f}},
                    {"zero_points", std::vector<int> {0}},
                    {"per_channel", false}, {"dtype", datatypes::u8}});
    auto output = graph.make_output(quant2->get_outputs());
}

TEST(GCCore_CPU_graph_reshedule, TestNDx2DMatMulWithBias1) {
    REQUIRE_AVX512();
    SET_THREADS_OR_SKIP(8);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());

    sc_graph_t graph_2D;
    sc_dims A_dims_2D {133 * 197, 768}, B_dims_2D {768, 768},
            bias_dims_2D {768}, out_dims_2D {133 * 197, 768};
    add_single_matmul_to_graph(
            graph_2D, A_dims_2D, B_dims_2D, bias_dims_2D, out_dims_2D);
    graph_driver(graph_2D, ctx);
    auto mod_2D = lower_graph(ctx, graph_2D,
            {graph_2D.get_output_ops()[0], graph_2D.get_input_ops()[0],
                    graph_2D.get_input_ops()[1], graph_2D.get_input_ops()[2]});
    auto fptr_2D = jit_engine_t::make(ctx)->get_entry_func(mod_2D);

    sc_graph_t graph_3D;
    sc_dims A_dims_3D {133, 197, 768}, B_dims_3D {768, 768}, bias_dims_3D {768},
            out_dims_3D {133, 197, 768};
    add_single_matmul_to_graph(
            graph_3D, A_dims_3D, B_dims_3D, bias_dims_3D, out_dims_3D);
    graph_driver(graph_3D, ctx);
    std::stringstream ss;
    print_graph(graph_3D, ss, true);
    // {reorder, tensorview(v13)} is moved to last
    std::string expected
            = R"(graph(v0: u8[133, 197, 768], v1: s8[768, 768], v2: bf16[768]) -> [v3: u8[133, 197, 768]] {
  [v4: f32[1]] = constant([1])
  [v5: f32[1, 1, 768]] = constant([1, 1, 768])
  [v6: f32[1, 768]] = tensor_view(v5)
  [v7: f32[1, 12, 1, 64]] = reorder(v6)
  [v8: bf16[1, 1, 768]] = tensor_view(v2)
  [v9: bf16[1, 768]] = tensor_view(v8)
  [v10: bf16[1, 12, 1, 64]] = reorder(v9)
  [v11: s8[12, 12, 16, 64, 4]] = reorder(v1)
  [v12: u8[26201, 768]] = tensor_view(v0)
  [v13: u8[26201, 768]] = outerloop_8X1X1_partition_reorder_quantized_managed_matmul_core_cast_mul_cast_add_cast_mul_cast_reorder(v12, v11, v7, v10, v4)
  [v3: u8[133, 197, 768]] = tensor_view(v13)
}
)";
    EXPECT_EQ(ss.str(), expected);
    auto mod_3D = lower_graph(ctx, graph_3D,
            {graph_3D.get_output_ops()[0], graph_3D.get_input_ops()[0],
                    graph_3D.get_input_ops()[1], graph_3D.get_input_ops()[2]});
    auto fptr_3D = jit_engine_t::make(ctx)->get_entry_func(mod_3D);

    const int A_size = test_utils::product(A_dims_2D);
    const int B_size = test_utils::product(B_dims_2D);
    const int out_size = test_utils::product(out_dims_2D);
    test_buffer<uint8_t> A_data(A_size);
    test_buffer<int8_t> B_data(B_size);
    test_buffer<fp16_t> bias_data(bias_dims_2D[0]);
    test_buffer<uint8_t> out_data_2D(out_size);
    test_buffer<uint8_t> out_data_3D(out_size);
    test_utils::fill_data(&A_data[0], A_size);
    test_utils::fill_data(&B_data[0], B_size);
    test_utils::fill_data(&bias_data[0], bias_dims_2D[0]);

    fptr_2D->call_default(
            &out_data_2D[0], &A_data[0], &B_data[0], &bias_data[0]);
    fptr_3D->call_default(
            &out_data_3D[0], &A_data[0], &B_data[0], &bias_data[0]);
    test_utils::compare_data(out_data_3D, out_data_2D, 1e-4f, 1e-5f);
}

static void build_two_matmul_graph(sc_graph_t &graph, const sc_dims &A_dims,
        const sc_dims &B_dims, const sc_dims &bias_dims,
        const sc_dims &out_dims) {
    for (size_t i = 0; i < 2; ++i) {
        add_single_matmul_to_graph(graph, A_dims, B_dims, bias_dims, out_dims);
    }
    auto add = graph.make("add",
            {graph.get_output_ops()[0]->get_inputs()[0],
                    graph.get_output_ops()[1]->get_inputs()[0]},
            {}, {});
    auto add_output = graph.make_output(add->get_outputs());
    graph.get_output_ops()[0]->remove();
    graph.get_output_ops()[1]->remove();
    graph.reset_op_ids();
}

TEST(GCCore_CPU_graph_reshedule, TestNDx2DMatMulWithBias2) {
    REQUIRE_AVX512();
    SET_THREADS_OR_SKIP(8);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());

    sc_graph_t graph_2D;
    sc_dims A_dims_2D {133 * 197, 768}, B_dims_2D {768, 768},
            bias_dims_2D {768}, out_dims_2D {133 * 197, 768};
    build_two_matmul_graph(
            graph_2D, A_dims_2D, B_dims_2D, bias_dims_2D, out_dims_2D);
    graph_driver(graph_2D, ctx);
    auto mod_2D = lower_graph(ctx, graph_2D,
            {graph_2D.get_output_ops()[0], graph_2D.get_input_ops()[0],
                    graph_2D.get_input_ops()[1], graph_2D.get_input_ops()[2],
                    graph_2D.get_input_ops()[3], graph_2D.get_input_ops()[4],
                    graph_2D.get_input_ops()[5]});
    auto fptr_2D = jit_engine_t::make(ctx)->get_entry_func(mod_2D);

    sc_graph_t graph_3D;
    sc_dims A_dims_3D {133, 197, 768}, B_dims_3D {768, 768}, bias_dims_3D {768},
            out_dims_3D {133, 197, 768};
    build_two_matmul_graph(
            graph_3D, A_dims_3D, B_dims_3D, bias_dims_3D, out_dims_3D);
    graph_driver(graph_3D, ctx);
    std::stringstream ss;
    print_graph(graph_3D, ss, true);
    // {reorder, tensorview(v27))} and {reorder, tensor_view(v20)} are moved
    // just before the add6
    std::string expected
            = R"(graph(v0: u8[133, 197, 768], v1: s8[768, 768], v2: bf16[768], v3: u8[133, 197, 768], v4: s8[768, 768], v5: bf16[768]) -> [v6: u8[133, 197, 768]] {
  [v7: f32[1]] = constant([1])
  [v8: f32[1, 1, 768]] = constant([1, 1, 768])
  [v9: f32[1, 768]] = tensor_view(v8)
  [v10: f32[1, 12, 1, 64]] = reorder(v9)
  [v11: f32[1]] = constant([1])
  [v12: f32[1, 1, 768]] = constant([1, 1, 768])
  [v13: f32[1, 768]] = tensor_view(v12)
  [v14: f32[1, 12, 1, 64]] = reorder(v13)
  [v15: bf16[1, 1, 768]] = tensor_view(v5)
  [v16: bf16[1, 768]] = tensor_view(v15)
  [v17: bf16[1, 12, 1, 64]] = reorder(v16)
  [v18: s8[12, 12, 16, 64, 4]] = reorder(v4)
  [v19: u8[26201, 768]] = tensor_view(v3)
  [v20: u8[26201, 768]] = outerloop_8X1X1_partition_reorder_quantized_managed_matmul_core_cast_mul_cast_add_cast_mul_cast_reorder(v19, v18, v10, v17, v7)
  [v21: u8[133, 197, 768]] = tensor_view(v20)
  [v22: bf16[1, 1, 768]] = tensor_view(v2)
  [v23: bf16[1, 768]] = tensor_view(v22)
  [v24: bf16[1, 12, 1, 64]] = reorder(v23)
  [v25: s8[12, 12, 16, 64, 4]] = reorder(v1)
  [v26: u8[26201, 768]] = tensor_view(v0)
  [v27: u8[26201, 768]] = outerloop_8X1X1_partition_reorder_quantized_managed_matmul_core_cast_mul_cast_add_cast_mul_cast_reorder(v26, v25, v14, v24, v11)
  [v28: u8[133, 197, 768]] = tensor_view(v27)
  [v6: u8[133, 197, 768]] = add(v28, v21)
}
)";
    EXPECT_EQ(ss.str(), expected);
    auto mod_3D = lower_graph(ctx, graph_3D,
            {graph_3D.get_output_ops()[0], graph_3D.get_input_ops()[0],
                    graph_3D.get_input_ops()[1], graph_3D.get_input_ops()[2],
                    graph_3D.get_input_ops()[3], graph_3D.get_input_ops()[4],
                    graph_3D.get_input_ops()[5]});
    auto fptr_3D = jit_engine_t::make(ctx)->get_entry_func(mod_3D);

    const int A_size = test_utils::product(A_dims_2D);
    const int B_size = test_utils::product(B_dims_2D);
    const int out_size = test_utils::product(out_dims_2D);
    test_buffer<uint8_t> A0_data(A_size);
    test_buffer<int8_t> B0_data(B_size);
    test_buffer<fp16_t> bias0_data(bias_dims_2D[0]);
    test_buffer<uint8_t> A1_data(A_size);
    test_buffer<int8_t> B1_data(B_size);
    test_buffer<fp16_t> bias1_data(bias_dims_2D[0]);
    test_buffer<uint8_t> out_data_2D(out_size);
    test_buffer<uint8_t> out_data_3D(out_size);
    test_utils::fill_data(&A0_data[0], A_size);
    test_utils::fill_data(&B0_data[0], B_size);
    test_utils::fill_data(&bias0_data[0], bias_dims_2D[0]);
    test_utils::fill_data(&A1_data[0], A_size);
    test_utils::fill_data(&B1_data[0], B_size);
    test_utils::fill_data(&bias1_data[0], bias_dims_2D[0]);

    fptr_2D->call_default(&out_data_2D[0], &A0_data[0], &B0_data[0],
            &bias0_data[0], &A1_data[0], &B1_data[0], &bias1_data[0]);
    fptr_3D->call_default(&out_data_3D[0], &A0_data[0], &B0_data[0],
            &bias0_data[0], &A1_data[0], &B1_data[0], &bias1_data[0]);
    test_utils::compare_data(out_data_3D, out_data_2D, 1e-4f, 1e-5f);
}
