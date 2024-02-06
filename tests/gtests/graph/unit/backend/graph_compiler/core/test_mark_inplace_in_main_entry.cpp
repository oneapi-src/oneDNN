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

#include <iostream>
#include <memory>
#include "context.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/jit/jit.hpp>
#ifdef DO_BENCH
#include <tuner/time_evaluator.hpp>
#endif

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_mark_inplace_in_main_entry_cpp, AddAdd) {
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    builder::ir_builder_t bld;

    sc_graph_t graph0;
    int N = 64, Cin = 128, Hin = 56, Win = 56;
    auto in0 = graph0.make_input({graph_tensor::make({N, Cin, Hin, Win},
            sc_data_format_t(format_kinds::NCHW), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make({N, Cin, Hin, Win},
            sc_data_format_t(format_kinds::NCHW), datatypes::f32)});
    auto add0 = graph0.make(
            "add", {in0->get_outputs()[0], in1->get_outputs()[0]}, {}, {});
    auto add1 = graph0.make(
            "add", {in0->get_outputs()[0], in1->get_outputs()[0]}, {}, {});
    auto out0 = graph0.make_output(add0->get_outputs());
    auto out1 = graph0.make_output(add1->get_outputs());
    // out1 can inplace inputs, but out0 can not inplace inputs.

    graph_driver(graph0, ctx);
    auto ir_mod = lower_graph(ctx, graph0, {in0, in1, out0, out1});
    auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod);

    /*
    main_entry(buffer_1: [f32 * 25690112UL], buffer_0: [f32 * 25690112UL],
            buffer_3: [f32 * 25690112UL], buffer_2: [f32 * 25690112UL])

    buffer_3 -> buffer_0
    */
    EXPECT_EQ(fptr0->inplace_pairs_.size(), 0UL);
    /*
    EXPECT_EQ(fptr0->inplace_pairs_[0].first, 1UL); // input id
    EXPECT_EQ(fptr0->inplace_pairs_[0].second, 2UL); // output id

    auto in0_data = alloc_array<float>(N * Cin * Hin * Win);
    auto in1_data = alloc_array<float>(N * Cin * Hin * Win);
    auto out0_data = alloc_array<float>(N * Cin * Hin * Win);
    auto out1_data = alloc_array<float>(N * Cin * Hin * Win);
    fptr0->call_default(
            &in0_data[0], &in1_data[0], &out0_data[0], &out1_data[0]);
    fptr0->call_default(
            &in0_data[0], &in1_data[0], &in1_data[0], &out1_data[0]);
    for (size_t i = 0; i < size_t(N * Cin * Hin * Win); ++i) {
        EXPECT_FLOAT_EQ(out0_data[i], in1_data[i]);
    }
    */
}

TEST(GCCore_CPU_mark_inplace_in_main_entry_cpp, ConvAdd0) {
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    builder::ir_builder_t bld;

    sc_graph_t graph0;
    int N = 64, Cin = 128, Hin = 56, Win = 56; // input feature
    int Cout0 = 32, k0 = 3, stride0 = 1, padding0 = 1; // conv0
    auto in0 = graph0.make_input({graph_tensor::make({N, Cout0, Hin, Win},
            sc_data_format_t(format_kinds::NCHW), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make({N, Cin, Hin, Win},
            sc_data_format_t(format_kinds::NCHW), datatypes::f32)});
    auto in2 = graph0.make_input({graph_tensor::make({Cout0, Cin, k0, k0},
            sc_data_format_t(format_kinds::KCRS), datatypes::f32)});
    sc_dims strides0 = {stride0, stride0}, paddings0 = {padding0, padding0};
    auto conv = graph0.make("conv_fwd_core",
            {in1->get_outputs()[0], in2->get_outputs()[0]}, {},
            {{"strides", strides0}, {"paddings", paddings0}});
    auto add = graph0.make(
            "add", {conv->get_outputs()[0], in0->get_outputs()[0]}, {}, {});
    auto out = graph0.make_output(add->get_outputs());
    /*
    The graph with topo:
    in0          in1      in2
     \            |        |
      \        reorder4  reorder5
       \        \         /
     reorder3      conv6
         \         /
            add7
             |
          reorder8
             |
            out
    */

    graph_driver(graph0, ctx);
    auto ir_mod = lower_graph(ctx, graph0, {in0, in1, in2, out});
    auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod, true);

    /*
    main_entry(buffer_2: [f32 * 6422528UL], buffer_1: [f32 * 25690112UL],
            buffer_0: [f32 * 36864UL], buffer_5: [f32 * 6422528UL])

    buffer_5 can not inplace buffer_1 because their sizes are not equal.
    */

    EXPECT_EQ(fptr0->inplace_pairs_.size(), 0UL);
}

TEST(GCCore_CPU_mark_inplace_in_main_entry_cpp, ConvAdd1) {
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    builder::ir_builder_t bld;

    sc_graph_t graph0;
    int N = 1, Cin = 1, Hin = 4, Win = 4; // input feature
    int Cout0 = 1, k0 = 1, stride0 = 1, padding0 = 0; // conv0
    auto in0 = graph0.make_input({graph_tensor::make({N, Cout0, Hin, Win},
            sc_data_format_t(format_kinds::NCHW), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make({N, Cin, Hin, Win},
            sc_data_format_t(format_kinds::NCHW), datatypes::f32)});
    auto in2 = graph0.make_input({graph_tensor::make({Cout0, Cin, k0, k0},
            sc_data_format_t(format_kinds::KCRS), datatypes::f32)});
    sc_dims strides0 = {stride0, stride0}, paddings0 = {padding0, padding0};
    auto conv = graph0.make("conv_fwd_core",
            {in1->get_outputs()[0], in2->get_outputs()[0]}, {},
            {{"strides", strides0}, {"paddings", paddings0}});
    auto add = graph0.make(
            "add", {conv->get_outputs()[0], in0->get_outputs()[0]}, {}, {});
    auto out = graph0.make_output(add->get_outputs());
    /*
    The graph with topo:
    in0   in1   in2
      \    \     /
       \    conv6
        \   /
         add7
          |
         out
    */

    graph_driver(graph0, ctx);
    auto ir_mod = lower_graph(ctx, graph0, {in0, in1, in2, out});
    auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod, true);

    /*
    * @param buffer_2 [f32 [1, 1, 4, 4] @ ABCD]
    * @param buffer_1 [f32 [1, 1, 4, 4] @ ABCD]
    * @param buffer_0 [f32 [1, 1, 1, 1] @ ABCD]
    * @param buffer_3 [f32 [1, 1, 4, 4] @ ABCD]
    func main_entry(buffer_2: [f32 * 16UL], buffer_1: [f32 * 16UL],
            buffer_0: [f32 * 1UL], buffer_3: [f32 * 16UL]): void {
        evaluate{outerloop_1X1X1X4_partition_conv_fwd_core_add_8(&buffer_3[0UL],
                &buffer_1[0UL], &buffer_0[0UL], &buffer_2[0UL])}
    }
    buffer_3 -> buffer_2
    */
    EXPECT_EQ(fptr0->inplace_pairs_.size(), 0UL);
    /*
    EXPECT_EQ(fptr0->inplace_pairs_[0].first, 0UL); // input id
    EXPECT_EQ(fptr0->inplace_pairs_[0].second, 3UL); // output id

    auto in0_data = alloc_array<float>(N * Cout0 * Hin * Win);
    auto in1_data = alloc_array<float>(N * Cin * Hin * Win);
    auto in2_data = alloc_array<float>(Cout0 * Cin * k0 * k0);
    auto out_data = alloc_array<float>(N * Cout0 * Hin * Win);
    fptr0->call_default(&in0_data[0], &in1_data[0], &in2_data[0], &out_data[0]);
    fptr0->call_default(&in0_data[0], &in1_data[0], &in2_data[0], &in0_data[0]);
    for (size_t i = 0; i < size_t(N * Cout0 * Hin * Win); ++i) {
        EXPECT_FLOAT_EQ(out_data[i], in0_data[i]);
    }
    */
}

TEST(GCCore_CPU_mark_inplace_in_main_entry_cpp, MatmulAdd) {
    REQUIRE_AVX2();
    REQUIRE_BF16();
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    builder::ir_builder_t bld;

    int M = 384, K = 1024, N = 768;
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make(
            {M, N}, sc_data_format_t(format_kinds::MN), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make(
            {M, K}, sc_data_format_t(format_kinds::MK), datatypes::bf16)});
    auto in2 = graph0.make_input({graph_tensor::make(
            {K, N}, sc_data_format_t(format_kinds::KN), datatypes::bf16)});
    auto matmul = graph0.make("matmul_core",
            {in1->get_outputs()[0], in2->get_outputs()[0]}, {}, {});
    auto add = graph0.make(
            "add", {in0->get_outputs()[0], matmul->get_outputs()[0]}, {}, {});
    auto out = graph0.make_output(add->get_outputs());
    /*
    The graph with topo:
    in0      in1      in2
     \        |        |
      \       |    reorder3
       \      \       /
        \      matmul4
         \     /
          add5
           |
          out
    add5's output, output of the graph, will inplace in0, input of the graph.
    */

    graph_driver(graph0, ctx);
    auto ir_mod = lower_graph(ctx, graph0, {in0, in1, in2, out});
    auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod, true);

    /*
    main_entry(buffer_2: [f32 * 4096UL], buffer_1: [bf16 * 8192UL],
            buffer_0: [bf16 * 131072UL], buffer_4: [f32 * 4096UL])

    buffer_4 -> buffer_2
    */
    EXPECT_EQ(fptr0->inplace_pairs_.size(), 0UL);
    /*
    EXPECT_EQ(fptr0->inplace_pairs_[0].first, 0UL); // input id
    EXPECT_EQ(fptr0->inplace_pairs_[0].second, 3UL); // output id

    auto in0_data = alloc_array<float>(M * N);
    auto in1_data = alloc_array<bf16_t>(M * K);
    auto in2_data = alloc_array<bf16_t>(K * N);
    auto out_data = alloc_array<float>(M * N);
    fptr0->call_default(&in0_data[0], &in1_data[0], &in2_data[0], &out_data[0]);
    fptr0->call_default(&in0_data[0], &in1_data[0], &in2_data[0], &in0_data[0]);
    for (size_t i = 0; i < size_t(M * N); ++i) {
        EXPECT_FLOAT_EQ(out_data[i], in0_data[i]);
    }
    */
}

TEST(GCCore_CPU_mark_inplace_in_main_entry_cpp, MatmulMulAdd) {
    REQUIRE_AVX2();
    REQUIRE_VNNI();
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    builder::ir_builder_t bld;

    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make(
            {16, 512}, sc_data_format_t(format_kinds::MK), datatypes::u8)});
    auto in1 = graph0.make_input({graph_tensor::make(
            {512, 256}, sc_data_format_t(format_kinds::KN), datatypes::s8)});
    auto in2 = graph0.make_input({graph_tensor::make(
            {16, 256}, sc_data_format_t(format_kinds::MN), datatypes::f32)});
    auto in3 = graph0.make_input({graph_tensor::make(
            {16, 256}, sc_data_format_t(format_kinds::MN), datatypes::f32)});
    auto deq0 = graph0.make("dequantize", {in0->get_outputs()[0]}, {},
            {{"dtype", datatypes::f32}, {"scales", std::vector<float> {1.f}},
                    {"zero_points", std::vector<int> {0}}});
    auto deq1 = graph0.make("dequantize", {in1->get_outputs()[0]}, {},
            {{"dtype", datatypes::f32}, {"scales", std::vector<float> {1.f}},
                    {"zero_points", std::vector<int> {0}}});
    auto matmul = graph0.make("matmul_core",
            {deq0->get_outputs()[0], deq1->get_outputs()[0]}, {}, {});
    auto mul = graph0.make(
            "mul", {matmul->get_outputs()[0], in2->get_outputs()[0]}, {}, {});
    auto add = graph0.make(
            "add", {mul->get_outputs()[0], in3->get_outputs()[0]}, {}, {});
    auto out = graph0.make_output(add->get_outputs());
    /*
    The graph with topo:
    in0      in1    in2    in3
    |        |       |      |
    |    reorder5    |      |
     \       /       |      |
    quant_matmul6   /      /
            \      /      /
              mul7       /
                \       /
                  add8
                   |
                  out
    Ideally, out will inplace use in3.
    */

    graph_driver(graph0, ctx);
    auto ir_mod = lower_graph(ctx, graph0, {in0, in1, in2, in3, out});
    auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod, true);

    /*
    main_entry(buffer_3: [u8 * 8192UL], buffer_2: [s8 * 131072UL],
            buffer_1: [f32 * 4096UL], buffer_0: [f32 * 4096UL],
            buffer_5: [f32 * 4096UL])

    no output arg can inplace input arg.
    */
    EXPECT_EQ(fptr0->inplace_pairs_.size(), 0UL);
}
