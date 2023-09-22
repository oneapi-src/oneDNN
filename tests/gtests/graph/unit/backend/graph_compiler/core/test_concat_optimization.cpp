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

#include "context.hpp"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/concat_memory_planning.hpp>
#include <compiler/ir/transform/index_flatten.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <runtime/runtime.hpp>
#ifdef DO_BENCH
#include <tuner/time_evaluator.hpp>
#endif
#include <iostream>
#include <memory>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

static void ir_compare_test_on_graph(
        std::function<sc_graph_t(void)> graph_builder,
        std::string &expected_ir) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(16);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.concat_optimization_ = true;
    builder::ir_builder_t bld;
    auto graph = graph_builder();
    graph_driver(graph, ctx);

    std::vector<sc_op_ptr> graph_args;
    for (auto &op : graph.get_input_ops()) {
        graph_args.push_back(op);
    }
    for (auto &op : graph.get_output_ops()) {
        graph_args.push_back(op);
    }
    auto ir_mod = lower_graph(ctx, graph, graph_args);

    concat_memory_planning_t pass;
    auto ret_mod = pass(ir_mod);
    std::stringstream ss;
    ss << ret_mod->get_entry_func();

    EXPECT_EQ(ss.str(), expected_ir);
}

// Test the accuracy after concat optimization.
// All tensor are float. Other dtypes should not use this function.
static void accuracy_test_on_graph(
        std::function<sc_graph_t(void)> graph_builder) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(56);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    builder::ir_builder_t bld;
    sc_graph_t graph0 = graph_builder();
    sc_graph_t graph1 = graph_builder();

    std::vector<test_buffer<float>> arg_buffers;
    std::vector<sc_op_ptr> graph0_args;
    for (auto &op : graph0.get_input_ops()) {
        graph0_args.push_back(op);
        auto dims = op->get_outputs()[0]->details_.get_plain_dims();
        size_t num_elements = 1;
        for (auto dim : dims) {
            num_elements *= dim;
        }
        arg_buffers.push_back(alloc_array<float>(num_elements));
    }
    for (auto &op : graph0.get_output_ops()) {
        graph0_args.push_back(op);
        auto dims = op->get_inputs()[0]->details_.get_plain_dims();
        size_t num_elements = 1;
        for (auto dim : dims) {
            num_elements *= dim;
        }
        arg_buffers.push_back(alloc_array<float>(num_elements));
    }
    arg_buffers.back().zeroout();

    ctx->flags_.concat_optimization_ = false;
    graph_driver(graph0, ctx);
    auto f0 = lower_graph(ctx, graph0, graph0_args);
    auto fptr0 = jit_engine_t::make(ctx)->get_entry_func(f0, true);
    std::vector<generic_val> generic_args0;
    for (unsigned i = 0; i < arg_buffers.size(); ++i) {
        generic_args0.emplace_back(arg_buffers.at(i).data());
    }
    fptr0->call_generic_default(generic_args0.data());
    test_buffer<float> disable_output = arg_buffers.back().copy();

    arg_buffers.back().zeroout();
    std::vector<sc_op_ptr> graph1_args;
    for (auto &op : graph1.get_input_ops()) {
        graph1_args.push_back(op);
    }
    for (auto &op : graph1.get_output_ops()) {
        graph1_args.push_back(op);
    }
    ctx->flags_.concat_optimization_ = true;
    graph_driver(graph1, ctx);
    auto f1 = lower_graph(ctx, graph1, graph1_args);
    auto fptr1 = jit_engine_t::make(ctx)->get_entry_func(f1, true);
    std::vector<generic_val> generic_args1;
    for (unsigned i = 0; i < arg_buffers.size(); ++i) {
        generic_args1.emplace_back(arg_buffers.at(i).data());
    }
    fptr1->call_generic_default(generic_args1.data());
    test_buffer<float> enable_output = arg_buffers.back().copy();

    test_utils::compare_data(enable_output, disable_output, 1e-3f);

#ifdef DO_BENCH
    auto exec0 = [&]() { fptr0->call_generic_default(generic_args0.data()); };
    auto exec1 = [&]() { fptr1->call_generic_default(generic_args1.data()); };
    const int repeat = 5, warm_up = 10, loop = 100;
    double cost0 = 1e12, cost1 = 1e12;
    for (int r = 0; r < repeat; r++) {
        double cost0_r = 0.f, cost1_r = 0.f;
        for (int t = 0; t < warm_up + loop; t++) {
            auto time0 = evaluate_time(exec0);
            if (t >= warm_up) cost0_r += time0;
            auto time1 = evaluate_time(exec1);
            if (t >= warm_up) cost1_r += time1;
        }
        cost0 = std::min(cost0_r, cost0);
        cost1 = std::min(cost1_r, cost1);
    }
    printf("@Time cost: not optimized %f ms vs optimized %f ms\n", cost0 / loop,
            cost1 / loop);
#endif
}

static const int A = 4, A0 = 4, A1 = 8;
static const int B = 8;
static const int C0 = 16, C1 = 32, C2 = 64;
static const int D = 32;

TEST(GCCore_CPU_concat_optimization_cpp, MergeConsecutiveConcats) {
    SET_THREADS_OR_SKIP(16);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    builder::ir_builder_t bld;

    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make({A0, B, C0, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make({A1, B, C1, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto in2 = graph0.make_input({graph_tensor::make({A1, B, C2, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});

    auto add0 = graph0.make(
            "add", {in0->get_outputs()[0], in0->get_outputs()[0]}, {}, {});
    auto mul0 = graph0.make(
            "mul", {in0->get_outputs()[0], in0->get_outputs()[0]}, {}, {});
    auto add1 = graph0.make(
            "add", {in1->get_outputs()[0], in1->get_outputs()[0]}, {}, {});
    auto add2 = graph0.make(
            "add", {in2->get_outputs()[0], in2->get_outputs()[0]}, {}, {});

    auto concat0 = graph0.make("concat",
            {mul0->get_outputs()[0], add0->get_outputs()[0]}, {},
            {{"axis", 0}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat0 output: [A0+A0, B, C0, D], A0+A0 = A1

    auto concat1 = graph0.make("concat",
            {concat0->get_outputs()[0], add1->get_outputs()[0]}, {},
            {{"axis", 2}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat1 output: [A1, B, C0+C1, D]

    auto concat2 = graph0.make("concat",
            {concat1->get_outputs()[0], add2->get_outputs()[0]}, {},
            {{"axis", 2}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat2 output: [A1, B, C0+C1+C2, D]

    auto concat3 = graph0.make("concat",
            {concat2->get_outputs()[0], add2->get_outputs()[0]}, {},
            {{"axis", 2}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat3 output: [A1, B, C0+C1+C2+C2, D]

    auto add3 = graph0.make("add",
            {concat3->get_outputs()[0], concat3->get_outputs()[0]}, {}, {});
    auto out = graph0.make_output(add3->get_outputs());

    graph_driver(graph0, ctx);
    // concat1 and concat2 is merged into concat3
    std::stringstream ss;
    print_graph(graph0, ss, true);

    std::string expected_str
            = R"(graph(v0: f32[4, 8, 16, 32], v1: f32[8, 8, 32, 32], v2: f32[8, 8, 64, 32]) -> [v3: f32[8, 8, 176, 32]] {
  [v4: f32[8, 8, 64, 32]] = add(v2, v2)
  [v5: f32[8, 8, 32, 32]] = add(v1, v1)
  [v6: f32[4, 8, 16, 32]] = mul(v0, v0)
  [v7: f32[4, 8, 16, 32]] = add(v0, v0)
  [v8: f32[8, 8, 16, 32]] = concat(v6, v7)
  [v9: f32[8, 8, 176, 32]] = concat(v8, v5, v4, v4)
  [v3: f32[8, 8, 176, 32]] = add(v9, v9)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

// DenseNet-like topo, concat ops are standalone
static sc_graph_t build_sequential_standalone_concats() {
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make({A, B, C0, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto add0 = graph0.make(
            "add", {in0->get_outputs()[0], in0->get_outputs()[0]}, {}, {});

    auto relu1 = graph0.make("tanh", {add0->get_outputs()[0]}, {}, {});
    auto concat1 = graph0.make("concat",
            {add0->get_outputs()[0], relu1->get_outputs()[0]}, {},
            {{"axis", 2}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat1 output: [A, B, C0*2, D]

    auto relu2 = graph0.make("sigmoid", {concat1->get_outputs()[0]}, {}, {});
    auto concat2 = graph0.make("concat",
            {concat1->get_outputs()[0], relu2->get_outputs()[0]}, {},
            {{"axis", 2}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat2 output: [A, B, C0*4, D]

    auto relu3 = graph0.make("relu", {concat2->get_outputs()[0]}, {}, {});
    auto concat3 = graph0.make("concat",
            {concat2->get_outputs()[0], relu3->get_outputs()[0]}, {},
            {{"axis", 2}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat3 output: [A, B, C0*8, D]

    auto add1 = graph0.make("add",
            {concat3->get_outputs()[0], concat3->get_outputs()[0]}, {}, {});
    auto out = graph0.make_output(add1->get_outputs());
    return graph0;
}

TEST(GCCore_CPU_concat_optimization_cpp, SequentialStandaloneConcats) {
    accuracy_test_on_graph(build_sequential_standalone_concats);
    std::string expected_str = R"(/**
 * main_entry
 * @param buffer_0 [f32 [4, 8, 16, 32] @ ABCD]
 * @param buffer_8 [f32 [4, 8, 128, 32] @ ABCD]
*/
func main_entry(buffer_0: [f32 * 4UL * 8UL * 16UL * 32UL], buffer_8: [f32 * 4UL * 8UL * 128UL * 32UL]): void {
  // [f32 [4, 8, 128, 32] @ ABCD]
  tensor buffer_7: [f32 * 4UL * 8UL * 128UL * 32UL]
  evaluate{outerloop_4X8X16_partition_add_tanh_8(&buffer_7[0UL, 0UL, 0UL, 0UL], &buffer_7[0UL, 0UL, 16UL, 0UL], buffer_0)}
  evaluate{sigmoid_2(&buffer_7[0UL, 0UL, 32UL, 0UL], &buffer_7[0UL, 0UL, 0UL, 0UL])}
  evaluate{relu_4(&buffer_7[0UL, 0UL, 64UL, 0UL], &buffer_7[0UL, 0UL, 0UL, 0UL])}
  evaluate{add_6(buffer_8, buffer_7, buffer_7)}
})";
    ir_compare_test_on_graph(build_sequential_standalone_concats, expected_str);
}

// DenseNet-like topo, concat ops are fused into one mixed partition
static sc_graph_t build_sequential_concats_in_one_partition() {
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make({A, B, C0, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto add0 = graph0.make(
            "add", {in0->get_outputs()[0], in0->get_outputs()[0]}, {}, {});

    auto relu1 = graph0.make("relu", {add0->get_outputs()[0]}, {}, {});
    auto concat1 = graph0.make("concat",
            {add0->get_outputs()[0], relu1->get_outputs()[0]}, {},
            {{"axis", 2}});
    // concat1 output: [A, B, C0*2, D]

    auto relu2 = graph0.make("relu", {concat1->get_outputs()[0]}, {}, {});
    auto concat2 = graph0.make("concat",
            {concat1->get_outputs()[0], relu2->get_outputs()[0]}, {},
            {{"axis", 2}});
    // concat2 output: [A, B, C0*4, D]

    auto relu3 = graph0.make("relu", {concat2->get_outputs()[0]}, {}, {});
    auto concat3 = graph0.make("concat",
            {concat2->get_outputs()[0], relu3->get_outputs()[0]}, {},
            {{"axis", 2}});
    // concat3 output: [A, B, C0*8, D]

    auto add1 = graph0.make("add",
            {concat3->get_outputs()[0], concat3->get_outputs()[0]}, {}, {});
    auto out = graph0.make_output(add1->get_outputs());
    return graph0;
}

TEST(GCCore_CPU_concat_optimization_cpp, SequentialConcatsInOnePartition) {
    accuracy_test_on_graph(build_sequential_concats_in_one_partition);
    // The added new buffer arguemnt is wrapped into inlined function
    // illustrated below
    std::string expected_str = R"(/**
 * main_entry
 * @param buffer_0 [f32 [4, 8, 16, 32] @ ABCD]
 * @param buffer_1 [f32 [4, 8, 128, 32] @ ABCD]
*/
func main_entry(buffer_0: [f32 * 4UL * 8UL * 16UL * 32UL], buffer_1: [f32 * 4UL * 8UL * 128UL * 32UL]): void {
  evaluate{outerloop_4X8_partition_add_relu_concat_relu_concat_relu_concat_add(buffer_1, buffer_0)}
})";
    ir_compare_test_on_graph(
            build_sequential_concats_in_one_partition, expected_str);
}

// Both standalone and fused concats exist in this graph.
static sc_graph_t build_sequential_concats_standalone_and_in_one_partition() {
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make({A, B, C0, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto add0 = graph0.make(
            "add", {in0->get_outputs()[0], in0->get_outputs()[0]}, {}, {});

    auto relu1 = graph0.make("tanh", {add0->get_outputs()[0]}, {}, {});
    auto concat1 = graph0.make("concat",
            {add0->get_outputs()[0], relu1->get_outputs()[0]}, {},
            {{"axis", 2}});
    // concat1 output: [A, B, C0*2, D]

    auto relu2 = graph0.make("relu", {concat1->get_outputs()[0]}, {}, {});
    auto concat2 = graph0.make("concat",
            {concat1->get_outputs()[0], relu2->get_outputs()[0]}, {},
            {{"axis", 2}});
    // concat2 output: [A, B, C0*4, D]

    auto relu3 = graph0.make("sigmoid", {concat2->get_outputs()[0]}, {}, {});
    auto concat3 = graph0.make("concat",
            {concat2->get_outputs()[0], relu3->get_outputs()[0]}, {},
            {{"axis", 2}});
    // concat3 output: [A, B, C0*8, D]

    auto relu4 = graph0.make("tanh", {concat3->get_outputs()[0]}, {}, {});
    auto concat4 = graph0.make("concat",
            {concat3->get_outputs()[0], relu4->get_outputs()[0]}, {},
            {{"axis", 2}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat4 output: [A, B, C0*16, D], standalone

    auto relu5 = graph0.make("relu", {concat4->get_outputs()[0]}, {}, {});
    auto concat5 = graph0.make("concat",
            {concat4->get_outputs()[0], relu5->get_outputs()[0]}, {},
            {{"axis", 2}});
    // concat5 output: [A, B, C0*32, D]

    auto relu6 = graph0.make("tanh", {concat5->get_outputs()[0]}, {}, {});
    auto concat6 = graph0.make("concat",
            {concat5->get_outputs()[0], relu6->get_outputs()[0]}, {},
            {{"axis", 2}});
    // concat6 output: [A, B, C0*64, D]

    auto relu7 = graph0.make("sigmoid", {concat6->get_outputs()[0]}, {}, {});
    auto concat7 = graph0.make("concat",
            {concat6->get_outputs()[0], relu7->get_outputs()[0]}, {},
            {{"axis", 2}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat7 output: [A, B, C0*128, D], standalone

    auto add1 = graph0.make("add",
            {concat7->get_outputs()[0], concat7->get_outputs()[0]}, {}, {});
    auto out = graph0.make_output(add1->get_outputs());
    return graph0;
}

TEST(GCCore_CPU_concat_optimization_cpp,
        SequentialStandaloneAndInOnePartitionConcats) {
    accuracy_test_on_graph(
            build_sequential_concats_standalone_and_in_one_partition);

    std::string expected_str = R"(/**
 * main_entry
 * @param buffer_0 [f32 [4, 8, 16, 32] @ ABCD]
 * @param buffer_7 [f32 [4, 8, 2048, 32] @ ABCD]
*/
func main_entry(buffer_0: [f32 * 4UL * 8UL * 16UL * 32UL], buffer_7: [f32 * 4UL * 8UL * 2048UL * 32UL]): void {
  // [f32 [4, 8, 256, 32] @ ABCD]
  tensor buffer_3: [f32 * 4UL * 8UL * 256UL * 32UL]
  evaluate{outerloop_4X8_partition_add_tanh_concat_relu_concat_sigmoid_concat_tanh(&buffer_3[0UL, 0UL, 0UL, 0UL], &buffer_3[0UL, 0UL, 128UL, 0UL], buffer_0)}
  // [f32 [4, 8, 2048, 32] @ ABCD]
  tensor buffer_6: [f32 * 4UL * 8UL * 2048UL * 32UL]
  evaluate{outerloop_4X8_partition_relu_concat_tanh_concat_sigmoid(&buffer_6[0UL, 0UL, 0UL, 0UL], &buffer_6[0UL, 0UL, 1024UL, 0UL], buffer_3)}
  evaluate{add_3(buffer_7, buffer_6, buffer_6)}
})";
    ir_compare_test_on_graph(
            build_sequential_concats_standalone_and_in_one_partition,
            expected_str);
}

static sc_graph_t build_reduce_reduce_concat() {
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make({A, B, C0, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make({A, B, C0, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto reduce0 = graph0.make("reduce", {in0->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int> {2}}, {"rd_op", 0}});
    auto reduce1 = graph0.make("reduce", {in1->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int> {2}}, {"rd_op", 0}});
    auto concat1 = graph0.make("concat",
            {reduce0->get_outputs()[0], reduce1->get_outputs()[0]}, {},
            {{"axis", 2}});
    // concat1 output: [A, B, 2, D]
    auto out = graph0.make_output(concat1->get_outputs());
    return graph0;
}

TEST(GCCore_CPU_concat_optimization_cpp, ReduceReduceConcat) {
    // In this case, reduce0 is a standalone op, reduce1 and concat are in one
    // partition. So the concat operation of reduce0's output is remained, the
    // concat operation of reduce1's output is deleted.
    accuracy_test_on_graph(build_reduce_reduce_concat);
}

static sc_graph_t build_tensorview_add_concat() {
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make({A, B * C0, D},
            sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make({A, B, C0, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto tv0 = graph0.make("tensor_view", {in0->get_outputs()[0]}, {},
            {{"shape", sc_dims {A, B, C0, D}}});
    auto add1 = graph0.make(
            "add", {in1->get_outputs()[0], in1->get_outputs()[0]}, {}, {});
    auto concat1 = graph0.make("concat",
            {tv0->get_outputs()[0], add1->get_outputs()[0]}, {}, {{"axis", 2}});
    // concat1 output: [A, B, 2 * C0, D]
    auto out = graph0.make_output(concat1->get_outputs());
    return graph0;
}

TEST(GCCore_CPU_concat_optimization_cpp, TensorviewAddConcat) {
    accuracy_test_on_graph(build_tensorview_add_concat);

    std::string expected_str = R"(/**
 * main_entry
 * @param buffer_1 [f32 [4, 128, 32] @ ABC]
 * @param buffer_0 [f32 [4, 8, 16, 32] @ ABCD]
 * @param buffer_2 [f32 [4, 8, 32, 32] @ ABCD]
*/
func main_entry(buffer_1: [f32 * 4UL * 128UL * 32UL], buffer_0: [f32 * 4UL * 8UL * 16UL * 32UL], buffer_2: [f32 * 4UL * 8UL * 32UL * 32UL]): void {
  evaluate{outerloop_4X8_partition_add_concat(buffer_2, buffer_0, &buffer_1[0, 0, 0])}
})";
    ir_compare_test_on_graph(build_tensorview_add_concat, expected_str);
}

/*
     -- B --
    /       \
A ----- C ---- Concat, all these ops are in one partition
    \       /
     -- D --
*/
static sc_graph_t build_inception_block_with_adds() {
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make({A, B, C0, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto add0 = graph0.make(
            "add", {in0->get_outputs()[0], in0->get_outputs()[0]}, {}, {});
    auto relu1 = graph0.make("tanh", {add0->get_outputs()[0]}, {}, {});
    auto relu2 = graph0.make("relu", {add0->get_outputs()[0]}, {}, {});
    auto relu3 = graph0.make("sigmoid", {add0->get_outputs()[0]}, {}, {});
    auto concat1 = graph0.make("concat",
            {relu1->get_outputs()[0], relu2->get_outputs()[0],
                    relu3->get_outputs()[0]},
            {}, {{"axis", 2}});
    // concat1 output: [A, B, C0*3, D]
    auto out = graph0.make_output(concat1->get_outputs());
    return graph0;
}

TEST(GCCore_CPU_concat_optimization_cpp, InceptionLikeTopoAdd) {
    // In this case, all the parent ops of concat op are in the same partition,
    // so all the concat operations are deleted.
    accuracy_test_on_graph(build_inception_block_with_adds);
}

struct inception_block_config {
    // input feature
    int N = 64;
    int Cin = 128;
    int Hin = 56;
    int Win = 56;

    // conv0
    int Cout0 = 32;
    int k0 = 1;
    int stride0 = 1;
    int padding0 = 0;

    // conv1
    int Cout1 = 32;
    int k1 = 3;
    int stride1 = 1;
    int padding1 = 1;

    // pool
    int k2 = 3;
    int stride2 = 1;
    int padding2 = 1;
};

/*
     -- B --
    /       \
A ----- C ---- Concat, these ops are in different partitions
    \       /
     -- D --
We can only optimize the input tensors whose producers are in the same partition
with the concat op.
*/
static sc_graph_t build_inception_block() {
    inception_block_config config; // use default config

    sc_graph_t graph0;
    auto in0 = graph0.make_input(
            {graph_tensor::make({config.N, config.Cin, config.Hin, config.Win},
                    sc_data_format_t(format_kinds::NCHW), datatypes::f32)});

    auto weight0 = graph0.make_input({graph_tensor::make(
            {config.Cout0, config.Cin, config.k0, config.k0},
            sc_data_format_t(format_kinds::KCRS), datatypes::f32)});
    sc_dims strides0 = {config.stride0, config.stride0},
            paddings0 = {config.padding0, config.padding0};
    auto conv0 = graph0.make("conv_fwd_core",
            {in0->get_outputs()[0], weight0->get_outputs()[0]}, {},
            {{"strides", strides0}, {"paddings", paddings0}});
    auto relu0 = graph0.make("relu", {conv0->get_outputs()[0]}, {}, {});

    auto weight1 = graph0.make_input({graph_tensor::make(
            {config.Cout1, config.Cin, config.k1, config.k1},
            sc_data_format_t(format_kinds::KCRS), datatypes::f32)});
    sc_dims strides1 = {config.stride1, config.stride1},
            paddings1 = {config.padding1, config.padding1};
    auto conv1 = graph0.make("conv_fwd_core",
            {in0->get_outputs()[0], weight1->get_outputs()[0]}, {},
            {{"strides", strides1}, {"paddings", paddings1}});
    auto relu1 = graph0.make("relu", {conv1->get_outputs()[0]}, {}, {});

    // PreCI do not support pooling for now. Omit the third branch.

    auto concat1 = graph0.make("concat",
            {relu0->get_outputs()[0], relu1->get_outputs()[0]}, {},
            {{"axis", 1}});
    // concat1 output: [N, Cout0 + Cout1, Hin, Win]
    auto out = graph0.make_output(concat1->get_outputs());

    return graph0;
}

TEST(GCCore_CPU_concat_optimization_cpp, InceptionLikeTopoConv) {
    accuracy_test_on_graph(build_inception_block);
}

struct conv_config_t {
    int Cin;
    int Cout;
    int kernel;
    int stride;
    int padding;
};

static graph_tensor_ptr add_conv(sc_graph_t &graph,
        const graph_tensor_ptr &data, const conv_config_t &config) {
    auto weight = graph.make_input({graph_tensor::make(
            {config.Cout, config.Cin, config.kernel, config.kernel},
            sc_data_format_t(format_kinds::KCRS), datatypes::f32)});
    sc_dims strides = {config.stride, config.stride};
    sc_dims paddings = {config.padding, config.padding};
    auto conv = graph.make("conv_fwd_core", {data, weight->get_outputs()[0]},
            {}, {{"strides", strides}, {"paddings", paddings}});
    return conv->get_outputs()[0];
}

static const std::vector<int> bn_bc_axis = {1}; // NCHW-format
static graph_tensor_ptr add_bn(sc_graph_t &graph, const graph_tensor_ptr &data,
        int K, std::vector<int> bc_axis) {
    auto bn_mul = graph.make_input(
            {graph_tensor::make({K})}, {{"constant", const_kind::local_const}});
    auto bn_add = graph.make_input(
            {graph_tensor::make({K})}, {{"constant", const_kind::local_const}});

    auto out = graph.make("mul", {data, bn_mul->get_outputs()[0]}, {},
            {{"bc_axis", bn_bc_axis}});
    out = graph.make("add", {out->get_outputs()[0], bn_add->get_outputs()[0]},
            {}, {{"bc_axis", bn_bc_axis}});
    return out->get_outputs()[0];
}

// each conv block contains several conv layers and one concat
// param growth: how many filters to add after a conv block
// param bn_size: factor of bottleneck layer in the conv block
static graph_tensor_ptr build_conv_block(sc_graph_t &graph,
        graph_tensor_ptr data, int num_input_features, int growth_rate,
        int bn_size, float drop_rate) {
    graph_tensor_ptr ori_data = data;
    data = add_bn(graph, ori_data, num_input_features, bn_bc_axis);
    data = graph.make("relu", {data}, {}, {})->get_outputs()[0];
    data = add_conv(
            graph, data, {num_input_features, bn_size * growth_rate, 1, 1, 0});

    data = add_bn(graph, data, bn_size * growth_rate, bn_bc_axis);
    data = graph.make("relu", {data}, {}, {})->get_outputs()[0];
    data = add_conv(graph, data, {bn_size * growth_rate, growth_rate, 3, 1, 1});

    // do not fuse concat
    data = graph.make("concat", {ori_data, data}, {},
                        {{"axis", 1}, {op_attr_key::break_pre_fuse, true},
                                {op_attr_key::break_post_fuse, true}})
                   ->get_outputs()[0];

    // currently do not support dropout layer

    return data;
}

// each dense block contains several conv blocks
static graph_tensor_ptr build_dense_block(sc_graph_t &graph,
        graph_tensor_ptr data, int num_layers, int num_input_features,
        int bn_size, int growth_rate, float drop_rate) {
    for (int i = 0; i < num_layers; ++i) {
        data = build_conv_block(graph, data,
                num_input_features + i * growth_rate, growth_rate, bn_size,
                drop_rate);
    }
    return data;
}

static graph_tensor_ptr add_transition_module(sc_graph_t &graph,
        graph_tensor_ptr data, int num_input_features,
        int num_output_features) {
    data = add_bn(graph, data, num_input_features, bn_bc_axis);
    data = graph.make("relu", {data}, {}, {})->get_outputs()[0];
    data = add_conv(
            graph, data, {num_input_features, num_output_features, 1, 1, 0});
    // PreCI do not support pooling for now. Use conv to downsample.
    data = add_conv(
            graph, data, {num_output_features, num_output_features, 3, 2, 1});
    return data;
}

static sc_graph_t build_densenet() {
    int growth_rate = 32;
    // std::vector<int> block_config = {6, 12, 24, 16}; // Densenet121
    // std::vector<int> block_config = {6, 12, 36, 24}; // Densenet161
    // std::vector<int> block_config = {6, 12, 32, 32}; // Densenet169
    // std::vector<int> block_config = {6, 12, 48, 32}; // Densenet201
    // std::vector<int> block_config = {2, 4, 8, 5};
    std::vector<int> block_config = {2}; // fast for test
    int num_init_features = 64;
    int bn_size = 4;
    float drop_rate = 0;

    sc_graph_t graph;
    int N = 16, Cin = 3, Hin = 224, Win = 224;
    auto in = graph.make_input({graph_tensor::make({N, Cin, Hin, Win},
            sc_data_format_t(format_kinds::NCHW), datatypes::f32)});

    // first laysers: conv-bn-relu-pool
    auto conv1_out = add_conv(
            graph, in->get_outputs()[0], {Cin, num_init_features, 3, 1, 1});
    auto bn1_out = add_bn(graph, conv1_out, num_init_features, bn_bc_axis);
    auto relu1_out = graph.make("relu", {bn1_out}, {}, {})->get_outputs()[0];
    // PreCI do not support pooling for now. Use conv to downsample.
    auto max_pool1_out = add_conv(
            graph, relu1_out, {num_init_features, num_init_features, 3, 2, 1});

    // dense blocks
    graph_tensor_ptr data = max_pool1_out;
    int num_features = num_init_features;
    for (size_t i = 0; i < block_config.size(); ++i) {
        int num_layers = block_config[i];
        data = build_dense_block(graph, data, num_layers, num_features, bn_size,
                growth_rate, drop_rate);
        num_features = num_features + num_layers * growth_rate;
        // add transition module between dense blocks
        if (i != block_config.size() - 1) {
            data = add_transition_module(
                    graph, data, num_features, num_features / 2);
            num_features /= 2;
        }
    }

    // we omit the final avg_pooling, flattening and classifier layers
    graph.make_output({data}, {});
    return graph;
}

TEST(GCCore_CPU_concat_optimization_cpp, Densenet) {
    accuracy_test_on_graph(build_densenet);
}

static sc_graph_t build_gptj_subgraph() {
    int N = 32, L = 1024, D = 256;
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make(
            {N, L, D}, sc_data_format_t(format_kinds::ABC), datatypes::bf16)});
    auto in1 = graph0.make_input({graph_tensor::make(
            {N, L, D}, sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    auto in2 = graph0.make_input({graph_tensor::make(
            {N, L, D}, sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    auto in3 = graph0.make_input({graph_tensor::make({N, 2 * L, D},
            sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    auto to0 = graph0.make(
            "cast", {in0->get_outputs()[0]}, {}, {{"dtype", datatypes::f32}});
    auto add1 = graph0.make(
            "add", {in1->get_outputs()[0], in2->get_outputs()[0]}, {}, {});
    auto concat2 = graph0.make("concat",
            {to0->get_outputs()[0], add1->get_outputs()[0]}, {},
            {{"axis", 1}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat2 output: (N, 2*L, D) @ ABC
    auto permute3 = graph0.make("reorder", {concat2->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t(format_kinds::ACB)},
                    {"internal", true}});
    // permute3 output: (N, D, 2*L) @ACB
    auto permute_in3 = graph0.make("reorder", {in3->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t(format_kinds::ACB)},
                    {"internal", true}});
    // permute_in3 output: (N, D, 2*L) @ACB
    auto concat4 = graph0.make("concat",
            {permute3->get_outputs()[0], permute_in3->get_outputs()[0]}, {},
            {{"axis", 1}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat4 output: (N, 2*D, 2*L) @ACB
    auto to5 = graph0.make("cast", {concat4->get_outputs()[0]}, {},
            {{"dtype", datatypes::bf16}});
    auto out = graph0.make_output(to5->get_outputs());
    return graph0;
}

TEST(GCCore_CPU_concat_optimization_cpp, GPTJ) {
    REQUIRE_BF16();
    accuracy_test_on_graph(build_gptj_subgraph);
}

// Both standalone and fused concats exist in this simple graph.
// Check ir to see if the vectorization and memory optimization can co-work.
static sc_graph_t build_concats_standalone_and_in_one_partition() {
    static const int A = 4, B = 8, C = 16, D = 56;
    sc_graph_t graph0;
    auto in0 = graph0.make_input({graph_tensor::make({A, B, C, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto in1 = graph0.make_input({graph_tensor::make({A, B, C, D},
            sc_data_format_t(format_kinds::ABCD), datatypes::f32)});
    auto add0 = graph0.make(
            "add", {in0->get_outputs()[0], in0->get_outputs()[0]}, {}, {});
    auto relu1 = graph0.make("relu", {add0->get_outputs()[0]}, {}, {});
    auto concat2 = graph0.make("concat",
            {in1->get_outputs()[0], relu1->get_outputs()[0]}, {},
            {{"axis", 2}});
    // concat2 output: [A, B, C*2, D]
    auto relu3 = graph0.make("relu", {concat2->get_outputs()[0]}, {}, {});
    auto concat4 = graph0.make("concat",
            {concat2->get_outputs()[0], relu3->get_outputs()[0]}, {},
            {{"axis", 2}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::break_post_fuse, true}});
    // concat4 output: [A, B, C*4, D], standalone
    auto relu5 = graph0.make("relu", {concat4->get_outputs()[0]}, {}, {});
    auto out = graph0.make_output(relu5->get_outputs());
    return graph0;
}

TEST(GCCore_CPU_concat_optimization_cpp, StandaloneAndInOnePartitionConcats) {
    accuracy_test_on_graph(build_concats_standalone_and_in_one_partition);
}
