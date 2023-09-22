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

#include <iostream>
#include "context.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>

using namespace dnnl::impl::graph::gc;

static const int A = 4, A0 = 4, A1 = 6, A2 = 8; // for concat at axis #0
static const int B = 8;
static const int C = 16, C0 = 16, C1 = 32, C2 = 64; // for concat at axis #2
static const int D = 32;

TEST(GCCore_CPU_concat_op_t_cpp, FourDimsConcatAxis0) {
    REQUIRE_AVX2();
    std::vector<float> input0_data(A0 * B * C * D);
    test_utils::fill_data(&input0_data[0], A0 * B * C * D);
    std::vector<float> input1_data(A1 * B * C * D);
    test_utils::fill_data(&input1_data[0], A1 * B * C * D);
    std::vector<float> input2_data(A2 * B * C * D);
    test_utils::fill_data(&input2_data[0], A2 * B * C * D);

    // concat at axis 0
    std::vector<float> ref_output0_data = input0_data;
    ref_output0_data.insert(
            ref_output0_data.end(), input1_data.begin(), input1_data.end());
    ref_output0_data.insert(
            ref_output0_data.end(), input2_data.begin(), input2_data.end());
    for (auto &e : ref_output0_data) {
        e *= 4;
    }

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    builder::ir_builder_t bld;
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make({A0, B, C, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make({A1, B, C, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto in2 = graph0.make_input({graph_tensor::make({A2, B, C, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});

    auto add0 = graph0.make(
            "add", {in0->get_outputs()[0], in0->get_outputs()[0]}, {}, {});
    auto add1 = graph0.make(
            "add", {in1->get_outputs()[0], in1->get_outputs()[0]}, {}, {});
    auto add2 = graph0.make(
            "add", {in2->get_outputs()[0], in2->get_outputs()[0]}, {}, {});

    auto concat0 = graph0.make("concat",
            {add0->get_outputs()[0], add1->get_outputs()[0],
                    add2->get_outputs()[0]},
            {}, {{"axis", 0}});

    auto add3 = graph0.make("add",
            {concat0->get_outputs()[0], concat0->get_outputs()[0]}, {}, {});
    auto out = graph0.make_output(add3->get_outputs());

    graph_driver(graph0, ctx);
    auto ir_mod = lower_graph(ctx, graph0, {in0, in1, in2, out});
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
    std::vector<float> graph_output0_data((A0 + A1 + A2) * B * C * D);
    fptr->call_default(&input0_data[0], &input1_data[0], &input2_data[0],
            &graph_output0_data[0]);
    test_utils::compare_data(
            graph_output0_data, ref_output0_data, 1e-4f, 1e-5f);
}

// Note: the input shapes of this function are fixed
static std::vector<float> calc_ref_output(std::vector<float> &input0_data,
        std::vector<float> &input1_data, std::vector<float> &input2_data) {
    std::vector<uint64_t> strides_in0 = {B * C0 * D, C0 * D, D, 1};
    std::vector<uint64_t> strides_in1 = {B * C1 * D, C1 * D, D, 1};
    std::vector<uint64_t> strides_in2 = {B * C2 * D, C2 * D, D, 1};
    std::vector<uint64_t> strides_concat
            = {B * (C0 + C1 + C2) * D, (C0 + C1 + C2) * D, D, 1};

    // concat at axis 2
    std::vector<float> ref_output0_data(A * B * (C0 + C1 + C2) * D);
    for (int i0 = 0; i0 < A; ++i0) {
        for (int i1 = 0; i1 < B; ++i1) {
            for (int i3 = 0; i3 < D; ++i3) {
                for (int j = 0; j < C0; ++j) {
                    ref_output0_data[i0 * strides_concat[0]
                            + i1 * strides_concat[1] + j * strides_concat[2]
                            + i3 * strides_concat[3]]
                            = 4
                            * input0_data[i0 * strides_in0[0]
                                    + i1 * strides_in0[1] + j * strides_in0[2]
                                    + i3 * strides_in0[3]];
                }
                for (int j = 0; j < C1; ++j) {
                    ref_output0_data[i0 * strides_concat[0]
                            + i1 * strides_concat[1]
                            + (j + C0) * strides_concat[2]
                            + i3 * strides_concat[3]]
                            = 4
                            * input1_data[i0 * strides_in1[0]
                                    + i1 * strides_in1[1] + j * strides_in1[2]
                                    + i3 * strides_in1[3]];
                }
                for (int j = 0; j < 64; ++j) {
                    ref_output0_data[i0 * strides_concat[0]
                            + i1 * strides_concat[1]
                            + (j + C0 + C1) * strides_concat[2]
                            + i3 * strides_concat[3]]
                            = 4
                            * input2_data[i0 * strides_in2[0]
                                    + i1 * strides_in2[1] + j * strides_in2[2]
                                    + i3 * strides_in2[3]];
                }
            }
        }
    }

    return ref_output0_data;
}

TEST(GCCore_CPU_concat_op_t_cpp, FourDimsConcatAxis2) {
    REQUIRE_AVX2();
    std::vector<float> input0_data(A * B * C0 * D);
    test_utils::fill_data(&input0_data[0], A * B * C0 * D);
    std::vector<float> input1_data(A * B * C1 * D);
    test_utils::fill_data(&input1_data[0], A * B * C1 * D);
    std::vector<float> input2_data(A * B * C2 * D);
    test_utils::fill_data(&input2_data[0], A * B * C2 * D);
    std::vector<float> ref_output0_data
            = calc_ref_output(input0_data, input1_data, input2_data);

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    builder::ir_builder_t bld;
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make({A, B, C0, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make({A, B, C1, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto in2 = graph0.make_input({graph_tensor::make({A, B, C2, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});

    auto add0 = graph0.make(
            "add", {in0->get_outputs()[0], in0->get_outputs()[0]}, {}, {});
    auto add1 = graph0.make(
            "add", {in1->get_outputs()[0], in1->get_outputs()[0]}, {}, {});
    auto add2 = graph0.make(
            "add", {in2->get_outputs()[0], in2->get_outputs()[0]}, {}, {});

    auto concat0 = graph0.make("concat",
            {add0->get_outputs()[0], add1->get_outputs()[0],
                    add2->get_outputs()[0]},
            {}, {{"axis", 2}});

    auto add3 = graph0.make("add",
            {concat0->get_outputs()[0], concat0->get_outputs()[0]}, {}, {});
    auto out = graph0.make_output(add3->get_outputs());

    graph_driver(graph0, ctx);

    auto ir_mod = lower_graph(ctx, graph0, {in0, in1, in2, out});
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
    std::vector<float> graph_output0_data(A * B * (C0 + C1 + C2) * D);
    fptr->call_default(&input0_data[0], &input1_data[0], &input2_data[0],
            &graph_output0_data[0]);
    test_utils::compare_data(
            graph_output0_data, ref_output0_data, 1e-4f, 1e-5f);
}

TEST(GCCore_CPU_concat_op_t_cpp, ConcatManagedMatmulAxis0) {
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    builder::ir_builder_t bld;

    int M0 = 32, M1 = 256, K = 128, N = 64;
    std::vector<float> input0_data(M0 * K);
    test_utils::fill_data(&input0_data[0], M0 * K);
    std::vector<float> input1_data(M1 * K);
    test_utils::fill_data(&input1_data[0], M1 * K);
    std::vector<float> weight0_data(K * N);
    test_utils::fill_data(&weight0_data[0], K * N);
    std::vector<float> graph_output0_data((M0 + M1) * N);
    std::vector<float> ref_output0_data((M0 + M1) * N);

    {
        /*
        A1 * B = C1, A2 * B = C2,
        D = [C1]
            [C2] (concat C1 and C2 at axis #0)
        */
        sc_graph_t graph0;
        auto in0 = graph0.make_input({graph_tensor::make(
                {M0, K}, sc_data_format_t(format_kinds::MK), datatypes::f32)});
        auto in1 = graph0.make_input({graph_tensor::make(
                {M1, K}, sc_data_format_t(format_kinds::MK), datatypes::f32)});
        auto weight0 = graph0.make_input({graph_tensor::make(
                {K, N}, sc_data_format_t(format_kinds::NK), datatypes::f32)});
        auto mm0 = graph0.make("managed_matmul_core",
                {in0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
        auto relu0 = graph0.make("relu", {mm0->get_outputs()[0]}, {}, {});
        auto mm1 = graph0.make("managed_matmul_core",
                {in1->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
        auto relu1 = graph0.make("relu", {mm1->get_outputs()[0]}, {}, {});
        auto concat1 = graph0.make("concat",
                {relu0->get_outputs()[0], relu1->get_outputs()[0]}, {},
                {{"axis", 0}});
        // concat1 output: [M0 + M1, N]
        auto out = graph0.make_output(concat1->get_outputs());
        graph_driver(graph0, ctx);
        auto ir_mod = lower_graph(ctx, graph0, {in0, in1, weight0, out});
        auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
        fptr0->call_default(&input0_data[0], &input1_data[0], &weight0_data[0],
                &graph_output0_data[0]);
    }

    {
        /*
        concat A1 and A2 at axis #0,
        [A1] * B = [C1] = D
        [A2]       [C2]
        */
        sc_graph_t graph0;
        auto in0 = graph0.make_input({graph_tensor::make(
                {M0, K}, sc_data_format_t(format_kinds::MK), datatypes::f32)});
        auto in1 = graph0.make_input({graph_tensor::make(
                {M1, K}, sc_data_format_t(format_kinds::MK), datatypes::f32)});
        auto weight0 = graph0.make_input({graph_tensor::make(
                {K, N}, sc_data_format_t(format_kinds::NK), datatypes::f32)});
        auto concat1 = graph0.make("concat",
                {in0->get_outputs()[0], in1->get_outputs()[0]}, {},
                {{"axis", 0}});
        // concat1 output: [M0 + M1, K]
        auto mm0 = graph0.make("managed_matmul_core",
                {concat1->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
        auto relu0 = graph0.make("relu", {mm0->get_outputs()[0]}, {}, {});
        auto out = graph0.make_output(relu0->get_outputs());
        graph_driver(graph0, ctx);
        auto ir_mod = lower_graph(ctx, graph0, {in0, in1, weight0, out});
        auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
        fptr0->call_default(&input0_data[0], &input1_data[0], &weight0_data[0],
                &ref_output0_data[0]);
    }

    test_utils::compare_data(
            graph_output0_data, ref_output0_data, 1e-4f, 1e-5f);
}

TEST(GCCore_CPU_concat_op_t_cpp, ConcatManagedMatmulAxis1) {
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    builder::ir_builder_t bld;

    int M = 64, K = 128, N0 = 256, N1 = 32;
    std::vector<float> input0_data(M * K);
    test_utils::fill_data(&input0_data[0], M * K);
    std::vector<float> weight0_data(K * N0);
    test_utils::fill_data(&weight0_data[0], K * N0);
    std::vector<float> weight1_data(K * N1);
    test_utils::fill_data(&weight1_data[0], K * N1);
    std::vector<float> graph_output0_data(M * (N0 + N1));
    std::vector<float> ref_output0_data(M * (N0 + N1));

    {
        // A * B1 = C1, A * B2 = C2, D = [C1 C2] (concat at axis #1)
        sc_graph_t graph0;
        auto in0 = graph0.make_input({graph_tensor::make(
                {M, K}, sc_data_format_t(format_kinds::MK), datatypes::f32)});
        auto weight0 = graph0.make_input({graph_tensor::make(
                {K, N0}, sc_data_format_t(format_kinds::KN), datatypes::f32)});
        auto mm0 = graph0.make("managed_matmul_core",
                {in0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
        auto relu0 = graph0.make("relu", {mm0->get_outputs()[0]}, {}, {});
        auto weight1 = graph0.make_input({graph_tensor::make(
                {K, N1}, sc_data_format_t(format_kinds::KN), datatypes::f32)});
        auto mm1 = graph0.make("managed_matmul_core",
                {in0->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
        auto relu1 = graph0.make("relu", {mm1->get_outputs()[0]}, {}, {});
        auto concat1 = graph0.make("concat",
                {relu0->get_outputs()[0], relu1->get_outputs()[0]}, {},
                {{"axis", 1}});
        // concat1 output: [M, N0 + N1]
        auto out = graph0.make_output(concat1->get_outputs());
        graph_driver(graph0, ctx);
        auto ir_mod = lower_graph(ctx, graph0, {in0, weight0, weight1, out});
        auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
        fptr0->call_default(&input0_data[0], &weight0_data[0], &weight1_data[0],
                &graph_output0_data[0]);
    }

    {
        // concat B1 and B2 at axis #1, A * [B1 B2] = [C1 C2] = D
        sc_graph_t graph0;
        auto in0 = graph0.make_input({graph_tensor::make(
                {M, K}, sc_data_format_t(format_kinds::MK), datatypes::f32)});
        auto weight0 = graph0.make_input({graph_tensor::make(
                {K, N0}, sc_data_format_t(format_kinds::KN), datatypes::f32)});
        auto weight1 = graph0.make_input({graph_tensor::make(
                {K, N1}, sc_data_format_t(format_kinds::KN), datatypes::f32)});
        auto concat1 = graph0.make("concat",
                {weight0->get_outputs()[0], weight1->get_outputs()[0]}, {},
                {{"axis", 1}});
        // concat1 output: [K, N0 + N1]
        auto mm0 = graph0.make("managed_matmul_core",
                {in0->get_outputs()[0], concat1->get_outputs()[0]}, {}, {});
        auto relu0 = graph0.make("relu", {mm0->get_outputs()[0]}, {}, {});
        auto out = graph0.make_output(relu0->get_outputs());
        graph_driver(graph0, ctx);
        auto ir_mod = lower_graph(ctx, graph0, {in0, weight0, weight1, out});
        auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
        fptr0->call_default(&input0_data[0], &weight0_data[0], &weight1_data[0],
                &ref_output0_data[0]);
    }

    test_utils::compare_data(
            graph_output0_data, ref_output0_data, 1e-4f, 1e-5f);
}

/*
         -- conv1x1 relu --
        /                  \
input ----- conv3x3 relu ------ concat -- output
        \                  /
         -- avg_pool ------
*/
TEST(GCCore_CPU_concat_op_t_cpp, InceptionLikeTopoConv) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(16);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    builder::ir_builder_t bld;

    sc_graph_t graph0;
    int N = 16, Cin = 128, Hin = 56, Win = 56;
    auto in0 = graph0.make_input({graph_tensor::make({N, Cin, Hin, Win},
            sc_data_format_t(format_kinds::NCHW), datatypes::f32)});

    int Cout0 = 32, kw0 = 1, kh0 = 1, stride0 = 1, padding0 = 0;
    auto weight0 = graph0.make_input({graph_tensor::make({Cout0, Cin, kw0, kh0},
            sc_data_format_t(format_kinds::KCRS), datatypes::f32)});
    sc_dims strides0 = {stride0, stride0}, paddings0 = {padding0, padding0};
    auto conv0 = graph0.make("conv_fwd_core",
            {in0->get_outputs()[0], weight0->get_outputs()[0]}, {},
            {{"strides", strides0}, {"paddings", paddings0}});
    auto relu0 = graph0.make("relu", {conv0->get_outputs()[0]}, {}, {});

    int Cout1 = 32, kw1 = 3, kh1 = 3, stride1 = 1, padding1 = 1;
    auto weight1 = graph0.make_input({graph_tensor::make({Cout1, Cin, kw1, kh1},
            sc_data_format_t(format_kinds::KCRS), datatypes::f32)});
    sc_dims strides1 = {stride1, stride1}, paddings1 = {padding1, padding1};
    auto conv1 = graph0.make("conv_fwd_core",
            {in0->get_outputs()[0], weight1->get_outputs()[0]}, {},
            {{"strides", strides1}, {"paddings", paddings1}});
    auto relu1 = graph0.make("relu", {conv1->get_outputs()[0]}, {}, {});

    // PreCI do not support avg_pooling_fwd op. Use conv.
    int Cout2 = 32, kw2 = 3, kh2 = 3, stride2 = 1, padding2 = 1;
    auto weight2 = graph0.make_input({graph_tensor::make({Cout2, Cin, kw2, kh2},
            sc_data_format_t(format_kinds::KCRS), datatypes::f32)});
    sc_dims strides2 = {stride2, stride2}, paddings2 = {padding2, padding2};
    auto conv2 = graph0.make("conv_fwd_core",
            {in0->get_outputs()[0], weight2->get_outputs()[0]}, {},
            {{"strides", strides2}, {"paddings", paddings2}});
    auto relu2 = graph0.make("relu", {conv2->get_outputs()[0]}, {}, {});

    auto concat1 = graph0.make("concat",
            {relu0->get_outputs()[0], relu1->get_outputs()[0],
                    relu2->get_outputs()[0]},
            {}, {{"axis", 1}});
    // concat1 output: [N, Cout0 + Cout1 + Cout2, Hin, Win]
    auto out = graph0.make_output(concat1->get_outputs());
    auto graph1 = copy_graph(graph0);

    std::vector<float> input0_data(N * Cin * Hin * Win);
    test_utils::fill_data(&input0_data[0], N * Cin * Hin * Win);
    std::vector<float> weight0_data(Cout0 * Cin * kw0 * kh0);
    test_utils::fill_data(&weight0_data[0], Cout0 * Cin * kw0 * kh0);
    std::vector<float> weight1_data(Cout1 * Cin * kw1 * kh1);
    test_utils::fill_data(&weight1_data[0], Cout1 * Cin * kw1 * kh1);
    std::vector<float> weight2_data(Cout2 * Cin * kw1 * kh1);
    test_utils::fill_data(&weight2_data[0], Cout2 * Cin * kw2 * kh2);
    std::vector<float> graph_output0_data(
            N * (Cout0 + Cout1 + Cout2) * Hin * Win);
    std::vector<float> ref_output0_data(
            N * (Cout0 + Cout1 + Cout2) * Hin * Win);

    ctx->flags_.mixed_fusion_ = true;
    graph_driver(graph0, ctx);
    auto ir_mod
            = lower_graph(ctx, graph0, {in0, weight0, weight1, weight2, out});
    auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
    fptr0->call_default(&input0_data[0], &weight0_data[0], &weight1_data[0],
            &weight2_data[0], &graph_output0_data[0]);

    ctx->flags_.mixed_fusion_ = false;
    graph_driver(graph1, ctx);
    auto ir_mod1 = lower_graph(ctx, graph1,
            {graph1.get_input_ops()[0], graph1.get_input_ops()[1],
                    graph1.get_input_ops()[2], graph1.get_input_ops()[3],
                    graph1.get_output_ops()[0]});
    auto fptr1 = jit_engine_t::make(ctx)->get_entry_func(ir_mod1);
    fptr1->call_default(&input0_data[0], &weight0_data[0], &weight1_data[0],
            &weight2_data[0], &ref_output0_data[0]);
    test_utils::compare_data(
            graph_output0_data, ref_output0_data, 1e-4f, 1e-5f);
}

TEST(GCCore_CPU_concat_op_t_cpp, ConcatPermuteConcat) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(16);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;

    int N = 32, L = 1024, D = 256;
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make(
            {N, L, D}, sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make(
            {N, L, D}, sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    auto in2 = graph0.make_input({graph_tensor::make(
            {N, L, D}, sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    // in3: plain dims and blocking format
    auto in3 = graph0.make_input({graph_tensor::make({N, 2 * L, D},
            sc_data_format_t(format_kinds::ACB), datatypes::f32)});

    auto add1 = graph0.make(
            "add", {in1->get_outputs()[0], in2->get_outputs()[0]}, {}, {});
    auto concat2 = graph0.make("concat",
            {in0->get_outputs()[0], add1->get_outputs()[0]}, {}, {{"axis", 1}});
    // concat2 output: (N, 2*L, D) @ ABC
    auto permute3 = graph0.make("reorder", {concat2->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t(format_kinds::ACB)},
                    {"internal", true}});
    // permute3 output: (N, D, 2*L) @ACB
    auto concat4 = graph0.make("concat",
            {permute3->get_outputs()[0], in3->get_outputs()[0]}, {},
            {{"axis", 2}}); // Note: axis = 2 in plain format
    // concat4 output: (N, 2*D, 2*L) @ACB
    auto out = graph0.make_output(concat4->get_outputs());
    // output: (N, 2*L, 2*D) @ABC

    graph_driver(graph0, ctx);
    auto ir_mod = lower_graph(ctx, graph0,
            {graph0.get_input_ops()[0], graph0.get_input_ops()[1],
                    graph0.get_input_ops()[2], graph0.get_input_ops()[3],
                    graph0.get_output_ops()[0]});
    std::stringstream ss;
    ss << ir_mod->get_entry_func();
    // Note the shapes should be right.
    expr out_buf = ir_mod->get_entry_func()->params_.back();
    std::vector<int64_t> expected_shape = {32, 2048, 512};
    auto shape = out_buf.checked_as<tensor>()->dims_;
    for (size_t i = 0; i < shape.size(); ++i) {
        EXPECT_EQ(get_const_as_int(shape[i].static_as<constant>()),
                expected_shape[i]);
    }
}

static sc_graph_t build_single_concat_graph(bool inner_most) {
    sc_data_format_t input_format = inner_most
            ? sc_data_format_t(format_kinds::ACDB)
            : sc_data_format_t(format_kinds::ABCD);
    int A = 112, B0 = 28, B1 = 56, B2 = 64, B3 = 112, C = 28, D = 56;
    sc_graph_t graph0;
    auto in0 = graph0.make_input(
            {graph_tensor::make({A, B0, C, D}, input_format, datatypes::f32)});
    auto in1 = graph0.make_input(
            {graph_tensor::make({A, B1, C, D}, input_format, datatypes::f32)});
    auto in2 = graph0.make_input(
            {graph_tensor::make({A, B2, C, D}, input_format, datatypes::f32)});
    auto in3 = graph0.make_input(
            {graph_tensor::make({A, B3, C, D}, input_format, datatypes::f32)});

    // concat at B-axis. If in plain format, concat axis is not the inner-most.
    // If in permuted format, concat axis is the inner-most axis.
    auto concat = graph0.make("concat",
            {in0->get_outputs()[0], in1->get_outputs()[0],
                    in2->get_outputs()[0], in3->get_outputs()[0]},
            {}, {{"axis", 1}});
    // concat output: (A, C, D, B0+B1+B2+B3) @ ACDB, if inner_most.
    // concat output: (A, B0+B1+B2+B3, C, D) @ ABCD, if not inner_most.
    auto out = graph0.make_output(concat->get_outputs());
    graph0.attrs_["is_input_plain"] = !inner_most;
    graph0.attrs_["is_output_plain"] = !inner_most;
    return graph0;
}

// print IR to check if vectorization works when concat axis is/isnot
// the inner-most.
TEST(GCCore_CPU_concat_op_t_cpp, CheckVectorized) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(16);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;

    for (bool inner_most : std::vector<bool> {true, false}) {
        sc_graph_t graph0 = build_single_concat_graph(inner_most);
        graph_driver(graph0, ctx);
        auto ir_mod = lower_graph(ctx, graph0,
                {graph0.get_input_ops()[0], graph0.get_input_ops()[1],
                        graph0.get_input_ops()[2], graph0.get_input_ops()[3],
                        graph0.get_output_ops()[0]});
        auto fptr = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
    }
}

TEST(GCCore_CPU_concat_op_t_cpp, DeqConv_Concat) {
    REQUIRE_AMX();
    SET_THREADS_OR_SKIP(16);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t graph0;
    // concat input
    auto in0 = graph0.make_input({graph_tensor::make({1, 64, 56, 56},
            sc_data_format_t(format_kinds::ABCD), datatypes::u8)});
    // conv input feature
    auto in1 = graph0.make_input({graph_tensor::make({1, 128, 56, 56},
            sc_data_format_t(format_kinds::ABCD), datatypes::u8)});
    // conv input weight
    auto in2 = graph0.make_input({graph_tensor::make({32, 128, 3, 3},
            sc_data_format_t(format_kinds::ABCD), datatypes::s8)});

    auto dequant0 = graph0.make("dequantize", in1->get_outputs(), {},
            {{"dtype", datatypes::f32}, {"scales", std::vector<float> {1.f}},
                    {"zero_points", std::vector<int> {0}}});
    auto dequant1 = graph0.make("dequantize", in2->get_outputs(), {},
            {{"dtype", datatypes::f32}, {"scales", std::vector<float> {1.f}},
                    {"zero_points", std::vector<int> {0}}});
    sc_dims strides = {1, 1}, paddings = {1, 1};
    auto conv = graph0.make("conv_fwd_core",
            {dequant0->get_outputs()[0], dequant1->get_outputs()[0]}, {},
            {{"strides", strides}, {"paddings", paddings}});
    auto dequant2 = graph0.make("dequantize", in0->get_outputs(), {},
            {{"dtype", datatypes::f32}, {"scales", std::vector<float> {1.f}},
                    {"zero_points", std::vector<int> {0}}});
    auto concat = graph0.make("concat",
            {dequant2->get_outputs()[0], conv->get_outputs()[0]}, {},
            {{"axis", 1}});
    auto out0 = graph0.make_output(conv->get_outputs());
    auto out1 = graph0.make_output(concat->get_outputs());

    graph_driver(graph0, ctx);
    std::stringstream ss;
    print_graph(graph0, ss, true);
    std::string expected_str
            = R"(graph(v0: u8[1, 64, 56, 56], v1: u8[1, 128, 56, 56], v2: s8[32, 128, 3, 3]) -> [v3: f32[1, 32, 56, 56], v4: f32[1, 96, 56, 56]] {
  [v5: s8[1, 1, 3, 3, 32, 32, 4]] = reorder(v2)
  [v6: u8[1, 56, 56, 128]] = reorder(v1)
  [v3: f32[1, 32, 56, 56], v4: f32[1, 96, 56, 56]] = partition_cast_quantized_conv_fwd_core_cast_reorder_concat_reorder(v6, v5, v0)
}
)";
    EXPECT_EQ(ss.str(), expected_str);

    std::vector<float> input0_data(1 * 64 * 56 * 56);
    test_utils::fill_data(&input0_data[0], 1 * 64 * 56 * 56);
    std::vector<float> input1_data(1 * 128 * 56 * 56);
    test_utils::fill_data(&input1_data[0], 1 * 128 * 56 * 56);
    std::vector<float> weight0_data(32 * 128 * 3 * 3);
    test_utils::fill_data(&weight0_data[0], 32 * 128 * 3 * 3);
    std::vector<float> graph_output0_data(1 * 32 * 56 * 56);
    std::vector<float> graph_output1_data(1 * (64 + 32) * 56 * 56);
    auto ir_mod = lower_graph(ctx, graph0, {in0, in1, in2, out0, out1});
    auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
    fptr0->call_default(&input0_data[0], &input1_data[0], &weight0_data[0],
            &graph_output0_data[0], &graph_output1_data[0]);
}

TEST(GCCore_CPU_concat_op_t_cpp, DeqConv_DeqConv_Concat) {
    REQUIRE_AMX();
    SET_THREADS_OR_SKIP(16);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;

    sc_graph_t graph0;
    // conv0 input feature
    auto in0 = graph0.make_input({graph_tensor::make({1, 128, 56, 56},
            sc_data_format_t(format_kinds::ABCD), datatypes::u8)});
    // conv0 input weight
    auto in1 = graph0.make_input({graph_tensor::make({32, 128, 3, 3},
            sc_data_format_t(format_kinds::ABCD), datatypes::s8)});

    // conv1 input feature
    auto in2 = graph0.make_input({graph_tensor::make({1, 64, 56, 56},
            sc_data_format_t(format_kinds::ABCD), datatypes::u8)});
    // conv1 input weight
    auto in3 = graph0.make_input({graph_tensor::make({16, 64, 3, 3},
            sc_data_format_t(format_kinds::ABCD), datatypes::s8)});

    auto dequant0 = graph0.make("dequantize", in0->get_outputs(), {},
            {{"dtype", datatypes::f32}, {"scales", std::vector<float> {1.f}},
                    {"zero_points", std::vector<int> {0}}});
    auto dequant1 = graph0.make("dequantize", in1->get_outputs(), {},
            {{"dtype", datatypes::f32}, {"scales", std::vector<float> {1.f}},
                    {"zero_points", std::vector<int> {0}}});
    sc_dims strides = {1, 1}, paddings = {1, 1};
    auto conv0 = graph0.make("conv_fwd_core",
            {dequant0->get_outputs()[0], dequant1->get_outputs()[0]}, {},
            {{"strides", strides}, {"paddings", paddings}});

    auto dequant2 = graph0.make("dequantize", in2->get_outputs(), {},
            {{"dtype", datatypes::f32}, {"scales", std::vector<float> {1.f}},
                    {"zero_points", std::vector<int> {0}}});
    auto dequant3 = graph0.make("dequantize", in3->get_outputs(), {},
            {{"dtype", datatypes::f32}, {"scales", std::vector<float> {1.f}},
                    {"zero_points", std::vector<int> {0}}});
    auto conv1 = graph0.make("conv_fwd_core",
            {dequant2->get_outputs()[0], dequant3->get_outputs()[0]}, {},
            {{"strides", strides}, {"paddings", paddings}});

    auto concat = graph0.make("concat",
            {conv0->get_outputs()[0], conv1->get_outputs()[0]}, {},
            {{"axis", 1}});
    auto out0 = graph0.make_output(concat->get_outputs());

    graph_driver(graph0, ctx);
    std::stringstream ss;
    print_graph(graph0, ss, true);
    std::string expected_str
            = R"(graph(v0: u8[1, 128, 56, 56], v1: s8[32, 128, 3, 3], v2: u8[1, 64, 56, 56], v3: s8[16, 64, 3, 3]) -> [v4: f32[1, 48, 56, 56]] {
  [v5: s8[1, 1, 3, 3, 16, 16, 4]] = reorder(v3)
  [v6: u8[1, 56, 56, 64]] = reorder(v2)
  [v7: f32[1, 16, 56, 56]] = partition_quantized_conv_fwd_core_cast_reorder(v6, v5)
  [v8: s8[1, 1, 3, 3, 32, 32, 4]] = reorder(v1)
  [v9: u8[1, 56, 56, 128]] = reorder(v0)
  [v4: f32[1, 48, 56, 56]] = partition_quantized_conv_fwd_core_cast_reorder_concat(v9, v8, v7)
}
)";
    EXPECT_EQ(ss.str(), expected_str);

    std::vector<float> input0_data(1 * 128 * 56 * 56);
    test_utils::fill_data(&input0_data[0], 1 * 128 * 56 * 56);
    std::vector<float> input1_data(32 * 128 * 3 * 3);
    test_utils::fill_data(&input1_data[0], 32 * 128 * 3 * 3);
    std::vector<float> input2_data(1 * 64 * 56 * 56);
    test_utils::fill_data(&input2_data[0], 1 * 64 * 56 * 56);
    std::vector<float> input3_data(16 * 64 * 3 * 3);
    test_utils::fill_data(&input3_data[0], 16 * 64 * 3 * 3);
    std::vector<float> graph_output0_data(1 * (32 + 16) * 56 * 56);
    auto ir_mod = lower_graph(ctx, graph0, {in0, in1, in2, in3, out0});
    auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
    fptr0->call_default(&input0_data[0], &input1_data[0], &input2_data[0],
            &input3_data[0], &graph_output0_data[0]);
}
