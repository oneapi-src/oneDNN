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

TEST(GCCore_CPU_test_eliminate_zero_shaped_tensors, ConcatInputShapeZero1) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(16);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;

    sc_graph_t graph0;
    int A = 32, B0 = 32, B1 = 0, C = 128;
    auto in0 = graph0.make_input({graph_tensor::make(
            {A, B0, C}, sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make(
            {A, B0, C}, sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    auto in2 = graph0.make_input({graph_tensor::make(
            {A, B1, C}, sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    auto add = graph0.make(
            "add", {in0->get_outputs()[0], in1->get_outputs()[0]}, {}, {});
    // break pre_fuse to show the inputs of concat op.
    auto concat = graph0.make("concat",
            {add->get_outputs()[0], in2->get_outputs()[0]}, {},
            {{"axis", 1}, {op_attr_key::break_pre_fuse, true}});
    auto out = graph0.make_output(concat->get_outputs());

    graph_driver(graph0, ctx);
    EXPECT_TRUE(std::any_of(
            graph0.ops_.begin(), graph0.ops_.end(), [](const sc_op_ptr &op) {
                return op->isa<concat_op_t>() && op->info_.inputs_.size() == 1;
            }));

    auto ir_mod = lower_graph(ctx, graph0, {in0, in1, in2, out});
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
    auto input0_data = alloc_array<float>(A * B0 * C);
    auto input1_data = alloc_array<float>(A * B0 * C);
    // For tensor whose shape contains 0, use alloc_size 1
    auto input2_data = alloc_array<float>(1);
    auto graph_output0_data = alloc_array<float>(A * (B0 + B1) * C);
    fptr->call_default(&input0_data[0], &input1_data[0], &input2_data[0],
            &graph_output0_data[0]);

    auto ref_output0_data = alloc_array<float>(A * (B0 + B1) * C);
    for (size_t i = 0; i < ref_output0_data.size(); ++i) {
        ref_output0_data[i] = input0_data[i] + input1_data[i];
    }
    test_utils::compare_data(
            graph_output0_data, ref_output0_data, 1e-4f, 1e-5f);
}

TEST(GCCore_CPU_test_eliminate_zero_shaped_tensors, ConcatInputShapeZero2) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(16);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;

    sc_graph_t graph0;
    int A = 4, B0 = 34, B1 = 0, C = 32, D = 128;
    auto in0 = graph0.make_input({graph_tensor::make({A, B0, C, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make({A, C, B1, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto trans = graph0.make("transpose", {in0->get_outputs()[0]}, {},
            {{"order", std::vector<int> {0, 2, 1, 3}}});
    auto concat = graph0.make("concat",
            {in1->get_outputs()[0], trans->get_outputs()[0]}, {},
            {{"axis", 2}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    auto quant = graph0.make("quantize", concat->get_outputs(), {},
            {{"channel_axis", 1}, {"zero_points", std::vector<int> {99}},
                    {"scales", std::vector<float> {0.00773799f}},
                    {"per_channel", false}, {"dtype", datatypes::u8}});
    auto out = graph0.make_output(quant->get_outputs());

    graph_driver(graph0, ctx);
    EXPECT_TRUE(std::any_of(
            graph0.ops_.begin(), graph0.ops_.end(), [](const sc_op_ptr &op) {
                return op->isa<concat_op_t>() && op->info_.inputs_.size() == 1;
            }));

    auto ir_mod = lower_graph(ctx, graph0, {in0, in1, out});
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
}

TEST(GCCore_CPU_test_eliminate_zero_shaped_tensors, AddInputShapeZero) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(16);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;

    int A = 32, B0 = 0, B1 = 32, C = 128;
    auto input0_data = alloc_array<float>(1);
    auto input1_data = alloc_array<float>(1);
    auto input2_data = alloc_array<float>(A * B1 * C);
    auto graph_output0_data = alloc_array<float>(A * (B0 + B1) * C);
    auto ref_output0_data = alloc_array<float>(A * (B0 + B1) * C);

    // graph with zero shaped input tensor
    {
        sc_graph_t graph0;
        auto in0 = graph0.make_input({graph_tensor::make({A, B0, C},
                sc_data_format_t(format_kinds::ABC), datatypes::f32)});
        auto in1 = graph0.make_input({graph_tensor::make({A, B0, C},
                sc_data_format_t(format_kinds::ABC), datatypes::f32)});
        auto in2 = graph0.make_input({graph_tensor::make({A, B1, C},
                sc_data_format_t(format_kinds::ABC), datatypes::f32)});
        auto add = graph0.make(
                "add", {in0->get_outputs()[0], in1->get_outputs()[0]}, {}, {});
        auto concat = graph0.make("concat",
                {add->get_outputs()[0], in2->get_outputs()[0]}, {},
                {{"axis", 1}, {op_attr_key::break_pre_fuse, true}});
        auto out = graph0.make_output(concat->get_outputs());

        graph_driver(graph0, ctx);
        // add op is deleted and concat op only has one input left.
        EXPECT_TRUE(std::none_of(graph0.ops_.begin(), graph0.ops_.end(),
                [](const sc_op_ptr &op) { return op->isa<add_op_t>(); }));
        EXPECT_TRUE(std::any_of(graph0.ops_.begin(), graph0.ops_.end(),
                [](const sc_op_ptr &op) {
                    return op->isa<concat_op_t>()
                            && op->info_.inputs_.size() == 1;
                }));

        auto ir_mod = lower_graph(ctx, graph0, {in0, in1, in2, out});
        auto fptr = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
        fptr->call_default(&input0_data[0], &input1_data[0], &input2_data[0],
                &graph_output0_data[0]);
    }

    // equivalent graph
    {
        sc_graph_t graph0;
        auto in0 = graph0.make_input({graph_tensor::make({A, B1, C},
                sc_data_format_t(format_kinds::ABC), datatypes::f32)});
        auto concat = graph0.make(
                "concat", {in0->get_outputs()[0]}, {}, {{"axis", 1}});
        auto out = graph0.make_output(concat->get_outputs());

        graph_driver(graph0, ctx);
        auto ir_mod = lower_graph(ctx, graph0, {in0, out});
        auto fptr = jit_engine_t::make(ctx)->get_entry_func(ir_mod);
        fptr->call_default(&input2_data[0], &ref_output0_data[0]);
    }

    test_utils::compare_data(
            graph_output0_data, ref_output0_data, 1e-4f, 1e-5f);
}
