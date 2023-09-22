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
#include "exception_util.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <ops/matmul_core.hpp>
#include <ops/templates/matmul_core.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_fusion_cost_model_cpp, TestBroadcastOp1) {
    sc_graph_t graph;
    SET_THREADS_OR_SKIP(28);

    auto input0 = graph.make_input(
            {graph_tensor::make({1, 64}, sc_data_format_t::MK())});
    auto input1 = graph.make_input(
            {graph_tensor::make({32, 64}, sc_data_format_t::MKmk(16, 16))});

    auto reo_node = graph.make("reorder", {input0->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(1, 16)}});
    // mul op still can be added into reorder partition although mul has more
    // loop parallelism than reorder, because it is small op workload
    auto mul_node = graph.make("mul",
            {reo_node->get_outputs()[0], input1->get_outputs()[0]}, {}, {});

    auto output0 = graph.make_output(mul_node->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    // turn on cost model
    ctx->flags_.use_cost_model_ = true;
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[1, 64], v1: f32[2, 4, 16, 16]) -> [v2: f32[2, 4, 16, 16]] {
  [v2: f32[2, 4, 16, 16]] = outerloop_1X4X1_partition_reorder_mul(v0, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_fusion_cost_model_cpp, TestBroadcastOp2) {
    sc_graph_t graph;
    auto ctx = std::make_shared<context_t>(*get_test_ctx());

    SET_THREADS_OR_SKIP(28);
    if (vectorize_step(ctx, sc_data_etype::F32) > 16) { GTEST_SKIP(); }

    // build N more than small op workload threshold
    int N = (mixed_partition_hint::small_op_workload_threshold / 2 + 1) * 16;
    auto input0 = graph.make_input(
            {graph_tensor::make({16, N}, sc_data_format_t::MKmk(16, 16))});
    auto input1 = graph.make_input(
            {graph_tensor::make({16, N}, sc_data_format_t::MKmk(16, 16))});

    auto red_node = graph.make("reduce", {input0->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});
    // mul op could not be added into reduce partition due to mul has more loop
    // parallelism than reduce.
    auto mul_node = graph.make("mul",
            {red_node->get_outputs()[0], input1->get_outputs()[0]}, {}, {});
    auto relu_node = graph.make("relu", mul_node->get_outputs(), {}, {});

    auto output0 = graph.make_output(relu_node->get_outputs());

    // turn on cost model
    ctx->flags_.use_cost_model_ = true;
    mixed_partition(graph, ctx);
    mixed_fuse_op_t *fused_op = get_mixed_op_from_graph(graph);
    COMPILE_ASSERT(fused_op, "No mixed fused op is found, please check")
    // fused op should have two partition due to mul op is expected to break
    // fusion by cost model
    EXPECT_EQ(fused_op->parti_list_.size(), (size_t)2);
    std::stringstream ss;
    print_graph(graph, ss, true);
    // multi_partitions prefix means it finally contains more than one
    // partition: `reduce_compute+reduce_collect` and `mul+relu`
    std::string expected_str
            = R"(graph(v0: f32[1, 845, 16, 16], v1: f32[1, 845, 16, 16]) -> [v2: f32[1, 845, 16, 16]] {
  [v2: f32[1, 845, 16, 16]] = multi_partitions_mul_relu_reduce_compute_reduce_collect(v0, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_fusion_cost_model_cpp, TestFusePreLoadBufferCheck) {
    sc_graph_t graph;

    int run_threads = 28;
    SET_THREADS_OR_SKIP(run_threads);

    int BS = run_threads, M = 384, K = 1024, N = 1024;

    auto input0 = graph.make_input({graph_tensor::make(
            {BS, M, K}, sc_data_format_t(format_kinds::ABC))});
    auto weight0 = graph.make_input({graph_tensor::make(
            {BS, K, N}, sc_data_format_t(format_kinds::ABC))});

    auto cast0 = graph.make("cast", {input0->get_outputs()[0]}, {},
            {{"dtype", datatypes::s32}});
    auto cast1 = graph.make("cast", {weight0->get_outputs()[0]}, {},
            {{"dtype", datatypes::s32}});
    auto cast2 = graph.make(
            "cast", {cast1->get_outputs()[0]}, {}, {{"dtype", datatypes::f32}});

    auto matmul0 = graph.make("matmul_core",
            {cast0->get_outputs()[0], cast2->get_outputs()[0]}, {}, {});

    auto output0 = graph.make_output(matmul0->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    graph_driver_before_fusion(graph, ctx);
    // turn on cost model
    ctx->flags_.use_cost_model_ = true;
    // Simulate the pre-load weight size of matmul is larger than L2 cache *
    // run_threads, as the result, the weight branch will not be merged with
    // input branch.
    ctx->machine_.cpu_flags_.dataCacheSize_[2]
            = N * utils::get_sizeof_type(datatypes::s32) / run_threads - 1;
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[28, 384, 1024], v1: f32[28, 1024, 1024]) -> [v2: f32[28, 384, 1024]] {
  [v3: f32[28, 1024, 1024]] = outerloop_28X1024_partition_cast_cast(v1)
  [v2: f32[28, 384, 1024]] = outerloop_28_partition_cast_matmul_core(v0, v3)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_fusion_cost_model_cpp, TestVerticalMergeForImageAffine) {
    sc_graph_t graph;

    auto get_conv_block_graph = [](int BS) {
        sc_graph_t g;
        int C = 64, H = 56, W = 56, K = 128;
        auto input = g.make_input({graph_tensor::make({BS, C, H, W})});
        auto weight0 = g.make_input({graph_tensor::make({K, C, 1, 1})});
        auto weight1 = g.make_input({graph_tensor::make({K, C, 1, 1})});

        auto conv_data0 = g.make("conv_fwd_core",
                {input->get_outputs()[0], weight0->get_outputs()[0]}, {},
                {{"strides", sc_dims {1, 1}}, {"paddings", sc_dims {0, 0}}});
        auto relu_out0 = g.make("relu", {conv_data0->get_outputs()[0]}, {}, {});

        auto conv_data1 = g.make("conv_fwd_core",
                {input->get_outputs()[0], weight1->get_outputs()[0]}, {},
                {{"strides", sc_dims {1, 1}}, {"paddings", sc_dims {0, 0}}});
        // break normal vertical merge
        auto relu_out1 = g.make("relu", {conv_data1->get_outputs()[0]}, {},
                {{op_attr_key::break_post_fuse, true}});
        auto mul0 = g.make("mul",
                {relu_out0->get_outputs()[0], relu_out1->get_outputs()[0]}, {},
                {});
        g.make_output(mul0->get_outputs());
        return g;
    };

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.use_cost_model_ = true;
    int num_threads = 28;
    SET_THREADS_OR_SKIP(num_threads);

    // case 1: can be merged
    graph = get_conv_block_graph(num_threads);
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[28, 64, 56, 56], v1: f32[128, 64, 1, 1], v2: f32[128, 64, 1, 1]) -> [v3: f32[28, 128, 56, 56]] {
  [v3: f32[28, 128, 56, 56]] = outerloop_28_partition_conv_fwd_core_relu_conv_fwd_core_relu_mul(v0, v2, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);

    // case 2: can not be merged
    graph = get_conv_block_graph(1);
    mixed_partition(graph, ctx);
    ss.str("");
    print_graph(graph, ss, true);
    expected_str
            = R"(graph(v0: f32[1, 64, 56, 56], v1: f32[128, 64, 1, 1], v2: f32[128, 64, 1, 1]) -> [v3: f32[1, 128, 56, 56]] {
  [v4: f32[1, 128, 56, 56]] = outerloop_1X1X1X56_partition_conv_fwd_core_relu(v0, v2)
  [v3: f32[1, 128, 56, 56]] = outerloop_1X1X1X56_partition_conv_fwd_core_relu_mul(v0, v1, v4)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_fusion_cost_model_cpp, TestTunableOp) {
    sc_graph_t graph;
    SET_THREADS_OR_SKIP(56);

    int BS = 64, M = 64, K = 1024, N = 64;

    auto input0 = graph.make_input({graph_tensor::make(
            {BS, M, K}, sc_data_format_t(format_kinds::ABC))});
    auto weight0 = graph.make_input({graph_tensor::make(
            {BS, K, N}, sc_data_format_t(format_kinds::ABC))});

    auto cast0 = graph.make("cast", {input0->get_outputs()[0]}, {},
            {{"dtype", datatypes::s32}});

    auto matmul0 = graph.make("matmul_core",
            {cast0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});

    graph.make_output(matmul0->get_outputs());

    ops::matmul_core_config_t cfg = {32, 32, 32};
    matmul0->stc_cast<ops::matmul_core_op_t>()->set_config(
            reflection::general_object_t::make(cfg));

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    graph_driver_before_fusion(graph, ctx);
    // turn on cost model
    ctx->flags_.use_cost_model_ = true;
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    // Fuse `BMM0` will reduce `cast0` loop parallelism from 64*64 to 64, which
    // would be rejected by cost model. However, if not fused, `BMM0` could only
    // get 64*2*2 of parallel loop by itself, which does not reach parallelism
    // requirement yet. As the result, cost model suggests to fuse them in order
    // to acheive better cache efficiency.
    std::string expected_str
            = R"(graph(v0: f32[64, 64, 1024], v1: f32[64, 1024, 64]) -> [v2: f32[64, 64, 64]] {
  [v2: f32[64, 64, 64]] = outerloop_64_partition_cast_matmul_core(v0, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}
