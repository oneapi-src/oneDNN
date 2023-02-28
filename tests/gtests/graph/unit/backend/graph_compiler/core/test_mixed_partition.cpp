/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/managed_matmul_core.hpp>
#include <ops/matmul_core.hpp>
#include <ops/templates/managed_matmul_core.hpp>
#include <ops/templates/matmul_core.hpp>
#include <reference/act_ref.hpp>
#include <reference/gemm_ref.hpp>
#include <runtime/config.hpp>
#ifdef DO_BENCH
#include <tuner/time_evaluator.hpp>
#endif
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

static void gtest_graph_driver_before_fusion(
        sc_graph_t &graph, const context_ptr &ctx) {
    analysis_quantized(graph, ctx);
    graph_inline(graph, ctx);
    constant_optimization(graph, ctx);
    quantize::quantize_info_propagation(graph, ctx);

    quantize::graph_reschedule(graph, ctx);
    quantize::quantize_inline(graph, ctx);

    elemwise_bcast_swap(graph, ctx);
    shape_relationship_binding(graph, ctx);
    permute_propagation(graph, ctx);

    quantize::calculate_op_compensation(graph, ctx);
    elemwise_dimension_alignment(graph, ctx);
    layout_propagation(graph, ctx);

    tensor_view_transform(graph, ctx);
    graph_simplify(graph, ctx);
    global_reschedule(graph, ctx);
    graph_constant_input_folding(graph, ctx);
}

static mixed_fuse_op_t *get_mixed_op_from_graph(sc_graph_t &graph) {
    mixed_fuse_op_t *mixed_op = nullptr;
    for (auto &op : graph.ops_) {
        if (auto mx_op = op->dyn_cast<mixed_fuse_op_t>()) {
            COMPILE_ASSERT(!mixed_op, "Only one fused op is expected")
            mixed_op = mx_op;
        } else {
            COMPILE_ASSERT(op->isa<input_op>() || op->isa<output_op>()
                            || op->isa<constant_op_t>(),
                    "Unexpected op type found " << op->op_name_)
        }
    }
    return mixed_op;
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphFuseOpPass) {
    sc_graph_t graph;

    auto in_a = graph.make_input({graph_tensor::make(
            {28, 64, 16, 16}, sc_data_format_t::NCHWc(32))});
    auto in_2 = graph.make_input({graph_tensor::make(
            {28, 64, 16, 16}, sc_data_format_t::NCHWc(32))});
    auto in_weight1 = graph.make_input({graph_tensor::make(
            {64, 64, 1, 1}, sc_data_format_t::KCRSck(32, 32))});

    auto conv = graph.make("conv_fwd_core",
            {in_a->get_outputs()[0], in_weight1->get_outputs()[0]}, {},
            {{"strides", sc_dims {1, 1}}, {"paddings", sc_dims {0, 0}}});
    conv->get_outputs()[0]->details_.set_format(sc_data_format_t::NCHWc(32));
    auto addop1 = graph.make(
            "add", {conv->get_outputs()[0], in_2->get_outputs()[0]}, {}, {});
    auto relu1 = graph.make("relu", addop1->get_outputs(), {}, {});
    auto raddop = graph.make("reduce", relu1->get_outputs(), {},
            {{"rd_axis", std::vector<int> {0}}, {"rd_op", 0}});
    auto in_3 = graph.make_input({std::make_shared<graph_tensor>(
            nullptr, raddop->get_outputs()[0]->details_)});
    auto addop = graph.make(
            "add", {raddop->get_outputs()[0], in_3->get_outputs()[0]}, {}, {});
    auto relu2 = graph.make("relu", addop->get_outputs(), {}, {});
    graph.make_output(relu2->get_outputs());

    auto ctx = get_test_ctx();

    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[28, 2, 16, 16, 32], v1: f32[28, 2, 16, 16, 32], v2: f32[2, 2, 1, 1, 32, 32], v3: f32[1, 2, 16, 16, 32]) -> [v4: f32[1, 2, 16, 16, 32]] {
  [v5: f32[28, 2, 16, 16, 32]] = outerloop_28X1X16_partition_conv_fwd_core_add_relu(v0, v2, v1)
  [v6: f32[1, 2, 16, 16, 32]] = reduce(v5)
  [v4: f32[1, 2, 16, 16, 32]] = outerloop_1X2X16X16_partition_add_relu(v6, v3)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp, TestFuseOpBreakAndNoFuse) {
    auto get_test_graph = [](const char *fuse_type = nullptr) {
        int M = 32, K = 64, N = 32;
        sc_graph_t mgr;
        auto in_a = mgr.make_input(
                {graph_tensor::make({M, K}, sc_data_format_t::MK())});
        auto in_b = mgr.make_input(
                {graph_tensor::make({K, N}, sc_data_format_t::KN())});
        auto in_c = mgr.make_input(
                {graph_tensor::make({M, N}, sc_data_format_t::MK())});
        in_a->attrs_.set("constant", const_kind::local_const);
        in_b->attrs_.set("constant", const_kind::local_const);

        auto gemm = mgr.make("matmul_core",
                {in_a->get_outputs()[0], in_b->get_outputs()[0]}, {}, {});
        auto bias = mgr.make("add",
                {gemm->get_outputs()[0], in_c->get_outputs()[0]},
                {graph_tensor::make({M, N}, sc_data_format_t::MK())},
                {{"bc_axis", std::vector<int> {1}}});
        auto relu = mgr.make("relu", {bias->get_outputs()[0]},
                {graph_tensor::make({M, N}, sc_data_format_t::MK())}, {});

        if (fuse_type) { relu->attrs_.set(fuse_type, true); }
        auto quan = mgr.make("quantize", relu->get_outputs(),
                {graph_tensor::make(
                        {M, N}, sc_data_format_t::MK(), datatypes::s8)},
                {{"dtype", datatypes::s8},
                        {"scales", std::vector<float> {1.2f}},
                        {"zero_points", std::vector<int> {2}},
                        {"channel_axis", 0}});

        mgr.make_output(quan->get_outputs());
        return mgr;
    };

    auto ctx = get_test_ctx();

    // full fusion version
    {
        sc_graph_t graph = get_test_graph();
        graph_inline(graph);
        quantize::quantize_inline(graph);
        elemwise_dimension_alignment(graph);
        layout_propagation(graph);
        mixed_partition(graph, ctx);
        std::stringstream ss;
        print_graph(graph, ss, true);
        std::string expected_str
                = R"(graph(v0: f32[32, 64], v1: f32[64, 32], v2: f32[32, 32]) -> [v3: s8[32, 32]] {
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: f32[1, 1, 64, 32]] = reorder(v1)
  [v3: s8[32, 32]] = outerloop_1_partition_reorder_reorder_matmul_core_add_relu_mul_add_cast_reorder(v2, v0, v6, v5, v4)
}
)";
        EXPECT_EQ(ss.str(), expected_str);
    }

    // break pre fusion version
    {
        sc_graph_t graph = get_test_graph(op_attr_key::break_pre_fuse);
        graph_inline(graph);
        quantize::quantize_inline(graph);
        elemwise_dimension_alignment(graph);
        layout_propagation(graph);
        mixed_partition(graph, ctx);
        std::stringstream ss;
        print_graph(graph, ss, true);
        std::string expected_str
                = R"(graph(v0: f32[32, 64], v1: f32[64, 32], v2: f32[32, 32]) -> [v3: s8[32, 32]] {
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: f32[1, 1, 64, 32]] = reorder(v1)
  [v7: f32[1, 1, 32, 32]] = outerloop_1_partition_reorder_reorder_matmul_core_add(v2, v0, v6)
  [v3: s8[32, 32]] = outerloop_1X1X32_partition_relu_mul_add_cast_reorder(v7, v5, v4)
}
)";
        EXPECT_EQ(ss.str(), expected_str);
    }

    // break post fusion version
    {
        sc_graph_t graph = get_test_graph(op_attr_key::break_post_fuse);
        graph_inline(graph);
        quantize::quantize_inline(graph);
        elemwise_dimension_alignment(graph);
        layout_propagation(graph);
        mixed_partition(graph, ctx);

        std::stringstream ss;
        print_graph(graph, ss, true);
        std::string expected_str
                = R"(graph(v0: f32[32, 64], v1: f32[64, 32], v2: f32[32, 32]) -> [v3: s8[32, 32]] {
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: f32[1, 1, 64, 32]] = reorder(v1)
  [v7: f32[1, 1, 32, 32]] = outerloop_1_partition_reorder_reorder_matmul_core_add_relu(v2, v0, v6)
  [v3: s8[32, 32]] = outerloop_1X1X32_partition_mul_add_cast_reorder(v7, v5, v4)
}
)";
        EXPECT_EQ(ss.str(), expected_str);
    }

    // no fuse fusion version
    {
        sc_graph_t graph = get_test_graph(op_attr_key::no_fuse);
        graph_inline(graph);
        quantize::quantize_inline(graph);
        elemwise_dimension_alignment(graph);
        layout_propagation(graph);
        mixed_partition(graph, ctx);

        std::stringstream ss;
        print_graph(graph, ss, true);
        std::string expected_str
                = R"(graph(v0: f32[32, 64], v1: f32[64, 32], v2: f32[32, 32]) -> [v3: s8[32, 32]] {
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: f32[1, 1, 64, 32]] = reorder(v1)
  [v7: f32[1, 1, 32, 32]] = outerloop_1_partition_reorder_reorder_matmul_core_add(v2, v0, v6)
  [v8: f32[1, 1, 32, 32]] = relu(v7)
  [v3: s8[32, 32]] = outerloop_1X1X32_partition_mul_add_cast_reorder(v8, v5, v4)
}
)";
        EXPECT_EQ(ss.str(), expected_str);
    }
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphBatchWiseFuse) {
    sc_graph_t graph;
    auto input_A = graph.make_input({graph_tensor::make({16, 32, 384, 1024})});
    auto input_B = graph.make_input({graph_tensor::make({16, 32, 384, 1024})});
    auto input_C = graph.make_input({graph_tensor::make({16, 1, 384, 1})});

    auto add0 = graph.make("add",
            {input_A->get_outputs()[0], input_B->get_outputs()[0]}, {}, {});
    auto red1 = graph.make("reduce", {add0->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int> {1, 3}}, {"rd_op", 0}});
    auto add1 = graph.make(
            "add", {add0->get_outputs()[0], red1->get_outputs()[0]}, {}, {});
    auto add2 = graph.make(
            "add", {add1->get_outputs()[0], input_C->get_outputs()[0]}, {}, {});

    auto tv0 = graph.make("tensor_view", {add2->get_outputs()[0]}, {},
            {{"shape", sc_dims {8, 2, 32, 384, 1024}}});

    auto output = graph.make_output(tv0->get_outputs());

    auto ctx = get_test_ctx();

    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[16, 32, 384, 1024], v1: f32[16, 32, 384, 1024], v2: f32[16, 1, 384, 1]) -> [v3: f32[8, 2, 32, 384, 1024]] {
  [v4: f32[16, 32, 384, 1024]] = outerloop_16X384_partition_add_reduce_add_add(v0, v1, v2)
  [v3: f32[8, 2, 32, 384, 1024]] = tensor_view(v4)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphHorizontalMerge) {
    sc_graph_t graph;
    int M = 384, K = 1024, N = 1024;
    auto input = graph.make_input({graph_tensor::make({M, K})});
    auto weight0 = graph.make_input({graph_tensor::make({K, N})});
    auto weight1 = graph.make_input({graph_tensor::make({K, N})});
    auto weight2 = graph.make_input({graph_tensor::make({K, N})});
    auto weight3 = graph.make_input({graph_tensor::make({K, N})});
    auto weight4 = graph.make_input({graph_tensor::make({K, N})});

    auto matmul0 = graph.make("matmul_core",
            {input->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    auto matmul1 = graph.make("matmul_core",
            {input->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    auto matmul2 = graph.make("matmul_core",
            {input->get_outputs()[0], weight2->get_outputs()[0]}, {}, {});
    auto matmul3 = graph.make("matmul_core",
            {input->get_outputs()[0], weight3->get_outputs()[0]}, {}, {});
    auto matmul4 = graph.make("matmul_core",
            {input->get_outputs()[0], weight4->get_outputs()[0]}, {}, {});
    auto output0 = graph.make_output(matmul0->get_outputs());
    auto output1 = graph.make_output(matmul1->get_outputs());
    auto output2 = graph.make_output(matmul2->get_outputs());
    auto output3 = graph.make_output(matmul3->get_outputs());
    auto output4 = graph.make_output(matmul4->get_outputs());
    auto ctx = get_test_ctx();

    gtest_graph_driver_before_fusion(graph, ctx);
    auto cfg = matmul0->stc_cast<ops::matmul_core_op_t>()
                       ->get_default_config(ctx)
                       .get_as<ops::matmul_core_config_t>();
    int M_num_block = M / cfg->M_block;
    int N_num_block = N / cfg->N_block;
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[384, 1024], v1: f32[1024, 1024], v2: f32[1024, 1024], v3: f32[1024, 1024], v4: f32[1024, 1024], v5: f32[1024, 1024]) -> [v6: f32[384, 1024], v7: f32[384, 1024], v8: f32[384, 1024], v9: f32[384, 1024], v10: f32[384, 1024]] {
  [v10: f32[384, 1024], v9: f32[384, 1024], v8: f32[384, 1024], v7: f32[384, 1024], v6: f32[384, 1024]] = outerloop_)"
            + std::to_string(M_num_block * N_num_block * 5) +
            R"(_partition_matmul_core_matmul_core_matmul_core_matmul_core_matmul_core(v0, v5, v4, v3, v2, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphRunSingleThreads) {
    sc_graph_t graph;

    auto input = graph.make_input({graph_tensor::make({10, 20, 30})});
    auto weight = graph.make_input({graph_tensor::make({10, 20, 30})});
    auto cast_node = graph.make(
            "cast", {input->get_outputs()[0]}, {}, {{"dtype", datatypes::s32}});
    auto reduce_node = graph.make("reduce", {cast_node->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});
    auto mul_node = graph.make("mul",
            {reduce_node->get_outputs()[0], weight->get_outputs()[0]}, {}, {});
    auto output = graph.make_output(mul_node->get_outputs());

    auto ctx = get_test_ctx();
    thread_num_reset reseter;
    // set single threads envoriment
    runtime_config_t::get().set_num_threads(1);
    gtest_graph_driver_before_fusion(graph, ctx);
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[10, 20, 30], v1: f32[10, 20, 30]) -> [v2: f32[10, 20, 30]] {
  [v2: f32[10, 20, 30]] = outerloop_10_partition_cast_reduce_compute_reduce_collect_mul(v0, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp,
        TestGraphFuseFakeTensorviewSharedWithOutput) {
    sc_graph_t graph;

    auto input0 = graph.make_input({graph_tensor::make({10, 20, 30})});
    auto input1 = graph.make_input({graph_tensor::make({10, 20, 30})});
    auto mul_node = graph.make("mul",
            {input0->get_outputs()[0], input1->get_outputs()[0]}, {}, {});
    // Although this tensorview op is shared with output op, it is actually fake
    // one which has same input and output dims. It is expected to fuse rather
    // than break.
    auto tv_node = graph.make("tensor_view", {mul_node->get_outputs()[0]}, {},
            {{"shape", sc_dims {10, 20, 30}}});
    auto cast_node = graph.make("cast", {tv_node->get_outputs()[0]}, {},
            {{"dtype", datatypes::s32}});
    auto output0 = graph.make_output(tv_node->get_outputs());
    auto output1 = graph.make_output(cast_node->get_outputs());

    auto ctx = get_test_ctx();
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[10, 20, 30], v1: f32[10, 20, 30]) -> [v2: f32[10, 20, 30], v3: s32[10, 20, 30]] {
  [v2: f32[10, 20, 30], v3: s32[10, 20, 30]] = outerloop_10X20_partition_mul_tensor_view_cast(v0, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphSpecialReorder) {
    sc_graph_t graph;

    auto input0 = graph.make_input(
            {graph_tensor::make({10, 64, 32, 32}, sc_data_format_t::NCHW())});
    auto input1 = graph.make_input(
            {graph_tensor::make({10, 64, 32, 32}, sc_data_format_t::NCHW())});

    auto mul_node = graph.make("mul",
            {input0->get_outputs()[0], input1->get_outputs()[0]}, {}, {});

    // This reorder is speical reoder case which can run into opmtimized reorder
    // kernel, but require larger fusion anchor
    auto reo_node = graph.make("reorder", {mul_node->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::NCHWc(32)}});

    auto output0 = graph.make_output(reo_node->get_outputs());

    auto ctx = get_test_ctx();
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[10, 64, 32, 32], v1: f32[10, 64, 32, 32]) -> [v2: f32[10, 2, 32, 32, 32]] {
  [v2: f32[10, 2, 32, 32, 32]] = outerloop_10_partition_mul_reorder(v0, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphBroadcastOp) {
    sc_graph_t graph;
    thread_num_reset reseter;
    // set threads envoriment
    runtime_config_t::get().set_num_threads(28);

    auto input0 = graph.make_input(
            {graph_tensor::make({1, 64}, sc_data_format_t::MK())});
    auto input1 = graph.make_input(
            {graph_tensor::make({512, 64}, sc_data_format_t::MKmk(16, 16))});

    auto reo_node = graph.make("reorder", {input0->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(1, 16)}});
    // mul op could not be added into reorder partition due to mul has more loop
    // parallelism than reorder.
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
            = R"(graph(v0: f32[1, 64], v1: f32[32, 4, 16, 16]) -> [v2: f32[32, 4, 16, 16]] {
  [v3: f32[1, 4, 1, 16]] = reorder(v0)
  [v2: f32[32, 4, 16, 16]] = mul(v3, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphFusePreLoadBufferCheck) {
    sc_graph_t graph;

    int run_threads = 28;
    thread_num_reset reseter;
    // set threads envoriment
    runtime_config_t::get().set_num_threads(run_threads);

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
    gtest_graph_driver_before_fusion(graph, ctx);
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

TEST(GCCore_graph_mixed_partition_cpp, TestGraphFuseBRGemmPreOpFusion) {
    sc_graph_t graph;

    auto run_threads = runtime_config_t::get().get_num_threads();

    int BS = run_threads, C = 64, H = 56, W = 56, K = 64;
    auto input0 = graph.make_input({graph_tensor::make({BS, C, H, W})});

    auto weight0 = graph.make_input({graph_tensor::make({K, C, 1, 1})});

    auto input1 = graph.make_input({graph_tensor::make({BS, C, H, W})});

    auto input2 = graph.make("constant", {}, {graph_tensor::make({1})},
            {{"values",
                     std::make_shared<static_data_t>(
                             std::vector<float> {1.0f})},
                    {"dtype", datatypes::f32}, {"plain_dims", sc_dims {1}}});

    auto cast0 = graph.make("cast", {input1->get_outputs()[0]}, {},
            {{"dtype", datatypes::s32}});

    auto add0 = graph.make(
            "add", {cast0->get_outputs()[0], input2->get_outputs()[0]}, {}, {});

    auto conv0 = graph.make("conv_fwd_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {},
            {{"strides", sc_dims {1, 1}}, {"paddings", sc_dims {0, 0}}});

    auto add1 = graph.make(
            "add", {conv0->get_outputs()[0], add0->get_outputs()[0]}, {}, {});

    auto output0 = graph.make_output(add1->get_outputs());

    auto ctx = get_test_ctx();
    mixed_partition(graph, ctx);
    mixed_fuse_op_t *fused_op = get_mixed_op_from_graph(graph);
    COMPILE_ASSERT(fused_op, "No mixed fused op is found, please check")

    // Due to brgemm pre-op fusion, the number of outer loops is equal to what
    // is written in conv template
    EXPECT_EQ(fused_op->parti_list_[0]->get_outer_loops().size(), (size_t)3);
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphPartitionRingRiskCheck1) {
    sc_graph_t graph;

    thread_num_reset reseter;
    // set single threads envoriment
    runtime_config_t::get().set_num_threads(1);

    int M, N, K;
    M = N = K = 256;

    auto input0 = graph.make_input(
            {graph_tensor::make({M, K}, sc_data_format_t(format_kinds::MN))});
    auto bias = graph.make_input(
            {graph_tensor::make({N}, sc_data_format_t(format_kinds::A))});
    auto weight0 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});
    auto weight1 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});
    auto weight2 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});
    // mmm1
    auto ret = graph.make("matmul",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    ret = graph.make("add", {ret->get_outputs()[0], bias->get_outputs()[0]}, {},
            {{"bc_axis", std::vector<int> {1}}});
    // mm0
    ret = graph.make("matmul_core",
            {ret->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    ret = graph.make("add", {ret->get_outputs()[0], bias->get_outputs()[0]}, {},
            {{"bc_axis", std::vector<int> {1}}});
    // mmm2
    ret = graph.make("matmul",
            {ret->get_outputs()[0], weight2->get_outputs()[0]}, {}, {});
    ret = graph.make("add", {ret->get_outputs()[0], bias->get_outputs()[0]}, {},
            {{"bc_axis", std::vector<int> {1}}});
    ret = graph.make_output(ret->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    graph_inline(graph, ctx);
    ops::managed_matmul_core_config_t cfg = {1, 1, 1, 1, 2, 0};
    for (auto &op : graph.ops_) {
        if (op->op_name_ == "managed_matmul_core") {
            auto matmul_op = op->dyn_cast<ops::managed_matmul_core_op_t>();
            matmul_op->set_config(reflection::general_object_t::make(cfg));
        }
    }
    graph_driver(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    // mmm1 and mmm2 could not be parallel merged due to paritition ring risk
    std::string expected_str
            = R"(graph(v0: f32[256, 256], v1: f32[256], v2: f32[256, 256], v3: f32[256, 256], v4: f32[256, 256]) -> [v5: f32[256, 256]] {
  [v6: f32[1, 256]] = tensor_view(v1)
  [v7: f32[256, 256]] = outerloop_1X1X1X1X1_partition_managed_matmul_core_add(v0, v2, v6)
  [v8: f32[256, 256]] = outerloop_4X4_partition_matmul_core_add(v7, v3, v6)
  [v5: f32[256, 256]] = outerloop_1X1X1X1X1_partition_managed_matmul_core_add(v8, v4, v6)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphBreakOpPreFusion1) {
    sc_graph_t graph;
    int BS = 6, M = 384, K = 1024, N = 1024;
    auto input = graph.make_input({graph_tensor::make({BS, M, K})});
    auto weight0 = graph.make_input({graph_tensor::make({BS, K, N})});

    auto cast0 = graph.make(
            "cast", {input->get_outputs()[0]}, {}, {{"dtype", datatypes::s32}});
    // break pre fuse for relu0, but it is expected to be pre-op fused yet
    auto relu0 = graph.make("relu", {cast0->get_outputs()[0]}, {},
            {{op_attr_key::break_pre_fuse, true}});
    auto matmul0 = graph.make("matmul_core",
            {cast0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    auto add0 = graph.make("add",
            {matmul0->get_outputs()[0], relu0->get_outputs()[0]}, {}, {});

    auto output0 = graph.make_output(add0->get_outputs());

    auto ctx = get_test_ctx();
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[6, 384, 1024], v1: f32[6, 1024, 1024]) -> [v2: f32[6, 384, 1024]] {
  [v2: f32[6, 384, 1024]] = outerloop_6_partition_cast_matmul_core_relu_add(v0, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphBreakOpPreFusion2) {
    sc_graph_t graph;

    int BS = 28, C = 64, H = 56, W = 56, K = 64;

    thread_num_reset reseter;
    // set threads envoriment
    runtime_config_t::get().set_num_threads(BS);

    auto input0 = graph.make_input({graph_tensor::make({BS, C, H, W})});
    auto weight0 = graph.make_input({graph_tensor::make({K, C, 1, 1})});
    auto weight1 = graph.make_input({graph_tensor::make({K, C, 1, 1})});

    weight0->attrs_.set("constant", const_kind::local_const);
    weight1->attrs_.set("constant", const_kind::local_const);

    auto relu0 = graph.make("relu", {input0->get_outputs()[0]}, {}, {});

    auto conv0 = graph.make("conv_fwd_core",
            {relu0->get_outputs()[0], weight0->get_outputs()[0]}, {},
            {{"strides", sc_dims {1, 1}}, {"paddings", sc_dims {0, 0}}});

    auto cast0 = graph.make(
            "cast", {conv0->get_outputs()[0]}, {}, {{"dtype", datatypes::s32}});
    // break pre fuse
    auto cast1 = graph.make("cast", {cast0->get_outputs()[0]}, {},
            {{"dtype", datatypes::s32}, {op_attr_key::break_pre_fuse, true}});

    auto conv1 = graph.make("conv_fwd_core",
            {relu0->get_outputs()[0], weight1->get_outputs()[0]}, {},
            {{"strides", sc_dims {1, 1}}, {"paddings", sc_dims {0, 0}}});

    auto relu1 = graph.make("relu", {conv1->get_outputs()[0]}, {}, {});

    auto add0 = graph.make(
            "add", {cast1->get_outputs()[0], relu1->get_outputs()[0]}, {}, {});

    graph.make_output(add0->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = false;
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[28, 64, 56, 56], v1: f32[64, 64, 1, 1], v2: f32[64, 64, 1, 1]) -> [v3: s32[28, 64, 56, 56]] {
  [v3: s32[28, 64, 56, 56]] = outerloop_28_partition_relu_conv_fwd_core_relu_conv_fwd_core_cast_cast_add(v0, v2, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp, TestaxisBinding1) {
    sc_graph_t graph;

    auto input0 = graph.make_input({graph_tensor::make({100, 100, 200})});
    auto input1 = graph.make_input({graph_tensor::make({100, 100, 200})});
    // broadcast side input
    auto input2 = graph.make_input({graph_tensor::make({100, 200})});

    // add0 owns 'for i in range(100)' outer loop, `100` represents first axis
    // for dims {100, 100, 200}
    auto add0 = graph.make("add",
            {input0->get_outputs()[0], input1->get_outputs()[0]}, {}, {});
    auto relu0 = graph.make("relu", {add0->get_outputs()[0]}, {}, {});

    // relu1 also owns 'for i in range(100)' outer loop, but here `100` is
    // broadcast axis for second axis for dims {100, 100, 200}
    auto relu1 = graph.make("relu", {input2->get_outputs()[0]}, {}, {});

    // add1 is broadcast add, the two input relu0 and relu1 should not be merged
    // with `outer_loop = 100` due to axis binding conflict
    auto add1 = graph.make(
            "add", {relu0->get_outputs()[0], relu1->get_outputs()[0]}, {}, {});

    graph.make_output(add1->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    graph_driver(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    /** the expected graph could not be like below, which is actully wrong and
     * may cause correctness issue:
     * graph(v0: f32[100, 100, 200], v1: f32[100, 100, 200], v2: f32[100, 200])
     * -> [v3: f32[100, 100, 200]] {
     * [v3: f32[100, 100, 200]] =
     * outerloop_100_partition_relu_tensor_view_add_relu_add(v2, v0, v1)
     *  }
     * */
    std::string expected_str
            = R"(graph(v0: f32[100, 100, 200], v1: f32[100, 100, 200], v2: f32[100, 200]) -> [v3: f32[100, 100, 200]] {
  [v4: f32[100, 200]] = relu(v2)
  [v5: f32[1, 100, 200]] = tensor_view(v4)
  [v3: f32[100, 100, 200]] = outerloop_100X100_partition_add_relu_add(v0, v1, v5)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp, TestaxisBinding2) {
    sc_graph_t graph;

    thread_num_reset reseter;
    // set single threads envoriment
    runtime_config_t::get().set_num_threads(2);

    int M, N, K;
    M = N = K = 256;

    auto input0 = graph.make_input(
            {graph_tensor::make({M, K}, sc_data_format_t(format_kinds::MN))});
    auto bias = graph.make_input(
            {graph_tensor::make({N}, sc_data_format_t(format_kinds::A))});
    auto weight0 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});
    auto weight1 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});

    // mmm0
    auto mmm0 = graph.make("matmul",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    mmm0 = graph.make("add", {mmm0->get_outputs()[0], bias->get_outputs()[0]},
            {}, {{"bc_axis", std::vector<int> {1}}});
    // mmm1
    auto mmm1 = graph.make("matmul",
            {input0->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    mmm1 = graph.make("add", {mmm1->get_outputs()[0], bias->get_outputs()[0]},
            {}, {{"bc_axis", std::vector<int> {1}}});
    graph.make_output(mmm0->get_outputs());
    graph.make_output(mmm1->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    graph_driver(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    // mmm0 and mmm1 are exactly two forking partitions
    // three levels of loops are merged
    std::string expected_str
            = R"(graph(v0: f32[256, 256], v1: f32[256], v2: f32[256, 256], v3: f32[256, 256]) -> [v4: f32[256, 256], v5: f32[256, 256]] {
  [v6: f32[1, 256]] = tensor_view(v1)
  [v5: f32[256, 256], v4: f32[256, 256]] = outerloop_2X1X1X1X1_partition_managed_matmul_core_add_managed_matmul_core_add(v0, v3, v6, v2)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

/* Case: same weight
out1 = x1 * w, out2 = x2 * w
*/
TEST(GCCore_graph_mixed_partition_cpp, SplitAndMergeInners_Accuracy0) {
    thread_num_reset reseter;
    runtime_config_t::get().set_num_threads(16);

    int M0 = 1024, M1 = 512, K = 2048, N = 256;

    sc_graph_t graph;
    auto input0 = graph.make_input(
            {graph_tensor::make({M0, K}, sc_data_format_t(format_kinds::MK))});
    auto weight0 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});
    auto input1 = graph.make_input(
            {graph_tensor::make({M1, K}, sc_data_format_t(format_kinds::MK))});

    // mmm0
    auto mmm0 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    auto relu0 = graph.make("relu", {mmm0->get_outputs()[0]}, {}, {});
    // mmm1, same weight with mmm0
    auto mmm1 = graph.make("managed_matmul_core",
            {input1->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    auto relu1 = graph.make("relu", {mmm1->get_outputs()[0]}, {}, {});
    auto out0 = graph.make_output(relu0->get_outputs());
    auto out1 = graph.make_output(relu1->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;

    auto f1 = lower_graph(ctx, graph, {input0, weight0, input1, out0, out1});
    auto fptr1 = jit_engine_t::make(ctx)->get_entry_func(f1);

    // split outmost and merge inners, outerloop_8X2
    mixed_partition(graph, ctx);

    auto f2 = lower_graph(ctx, graph, {input0, weight0, input1, out0, out1});
    auto fptr2 = jit_engine_t::make(ctx)->get_entry_func(f2);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[1024, 2048], v1: f32[2048, 256], v2: f32[512, 2048]) -> [v3: f32[1024, 256], v4: f32[512, 256]] {
  [v4: f32[512, 256], v3: f32[1024, 256]] = outerloop_8X2_partition_managed_matmul_core_relu_managed_matmul_core_relu(v2, v1, v0)
}
)";
    EXPECT_EQ(ss.str(), expected_str);

    std::vector<float> input0_data(M0 * K);
    test_utils::fill_data(&input0_data[0], M0 * K);
    std::vector<float> weight0_data(K * N);
    test_utils::fill_data(&weight0_data[0], K * N);
    std::vector<float> input1_data(M1 * K);
    test_utils::fill_data(&input1_data[0], M1 * K);

    std::vector<float> ori_output0_data(M0 * N);
    std::vector<float> ori_output1_data(M1 * N);
    fptr1->call_default(&input0_data[0], &weight0_data[0], &input1_data[0],
            &ori_output0_data[0], &ori_output1_data[0]);

    std::vector<float> pass_output0_data(M0 * N);
    std::vector<float> pass_output1_data(M1 * N);
    fptr2->call_default(&input0_data[0], &weight0_data[0], &input1_data[0],
            &pass_output0_data[0], &pass_output1_data[0]);

    test_utils::compare_data(ori_output0_data, pass_output0_data, 1e-4, 1e-5);
    test_utils::compare_data(ori_output1_data, pass_output1_data, 1e-4, 1e-5);
}

/* Case: same input_data
out1 = x1 * w1, out2 = x1 * w2
*/
TEST(GCCore_graph_mixed_partition_cpp, SplitAndMergeInners_Accuracy1) {
    thread_num_reset reseter;
    runtime_config_t::get().set_num_threads(16);

    int M = 1024, K = 2048, N = 256;

    sc_graph_t graph;
    auto input0 = graph.make_input(
            {graph_tensor::make({M, K}, sc_data_format_t(format_kinds::MK))});
    auto weight0 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});
    auto weight1 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});

    // mmm0
    auto mmm0 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    auto relu0 = graph.make("relu", {mmm0->get_outputs()[0]}, {}, {});
    // mmm1, irrelevant with mmm0
    auto mmm1 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    auto relu1 = graph.make("relu", {mmm1->get_outputs()[0]}, {}, {});
    auto out0 = graph.make_output(relu0->get_outputs());
    auto out1 = graph.make_output(relu1->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;

    auto f1 = lower_graph(ctx, graph, {input0, weight0, weight1, out0, out1});
    auto fptr1 = jit_engine_t::make(ctx)->get_entry_func(f1);

    // merge inners, outerloop_16x1x1
    mixed_partition(graph, ctx);

    auto f2 = lower_graph(ctx, graph, {input0, weight0, weight1, out0, out1});
    auto fptr2 = jit_engine_t::make(ctx)->get_entry_func(f2);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[1024, 2048], v1: f32[2048, 256], v2: f32[2048, 256]) -> [v3: f32[1024, 256], v4: f32[1024, 256]] {
  [v4: f32[1024, 256], v3: f32[1024, 256]] = outerloop_16X1X1X1X1_partition_managed_matmul_core_relu_managed_matmul_core_relu(v0, v2, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);

    std::vector<float> input0_data(M * K);
    test_utils::fill_data(&input0_data[0], M * K);
    std::vector<float> weight0_data(K * N);
    test_utils::fill_data(&weight0_data[0], K * N);
    std::vector<float> weight1_data(K * N);
    test_utils::fill_data(&weight1_data[0], K * N);

    std::vector<float> ori_output0_data(M * N);
    std::vector<float> ori_output1_data(M * N);
    fptr1->call_default(&input0_data[0], &weight0_data[0], &weight1_data[0],
            &ori_output0_data[0], &ori_output1_data[0]);

    std::vector<float> pass_output0_data(M * N);
    std::vector<float> pass_output1_data(M * N);
    fptr2->call_default(&input0_data[0], &weight0_data[0], &weight1_data[0],
            &pass_output0_data[0], &pass_output1_data[0]);

    test_utils::compare_data(ori_output0_data, pass_output0_data, 1e-4, 1e-5);
    test_utils::compare_data(ori_output1_data, pass_output1_data, 1e-4, 1e-5);
}

/* Case: two consective MMMs
out1 = x1 * w1, out2 = out1 * w2
*/
static ir_module_ptr get_two_consective_mmm(
        int M, int K1, int K2, int K3, context_ptr ctx, sc_graph_t &graph) {
    auto input0 = graph.make_input(
            {graph_tensor::make({M, K1}, sc_data_format_t(format_kinds::MK))});
    auto weight0 = graph.make_input(
            {graph_tensor::make({K1, K2}, sc_data_format_t(format_kinds::KN))});
    auto weight1 = graph.make_input(
            {graph_tensor::make({K2, K3}, sc_data_format_t(format_kinds::KN))});

    // mmm0
    auto mmm0 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    auto relu0 = graph.make("relu", {mmm0->get_outputs()[0]}, {}, {});
    // mmm1, using mmm0's output
    auto mmm1 = graph.make("managed_matmul_core",
            {relu0->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    auto relu1 = graph.make("relu", {mmm1->get_outputs()[0]}, {}, {});
    auto out0 = graph.make_output(relu0->get_outputs());
    auto out1 = graph.make_output(relu1->get_outputs());

    graph_driver(graph, ctx);

    auto f1 = lower_graph(ctx, graph, {input0, weight0, weight1, out0, out1});

    return f1;
}

TEST(GCCore_graph_mixed_partition_cpp, SplitAndMergeInners_Accuracy2) {
    thread_num_reset reseter;
    runtime_config_t::get().set_num_threads(16);

    int M = 256, K1 = 2048, K2 = 512, K3 = 1024;
    std::vector<float> input0_data(M * K1);
    test_utils::fill_data(&input0_data[0], M * K1);
    std::vector<float> weight0_data(K1 * K2);
    test_utils::fill_data(&weight0_data[0], K1 * K2);
    std::vector<float> weight1_data(K2 * K3);
    test_utils::fill_data(&weight1_data[0], K2 * K3);

    std::vector<float> ori_output0_data(M * K2);
    std::vector<float> ori_output1_data(M * K3);
    std::vector<float> pass_output0_data(M * K2);
    std::vector<float> pass_output1_data(M * K3);

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = false;
    sc_graph_t graph1;
    auto f1 = get_two_consective_mmm(M, K1, K2, K3, ctx, graph1);
    auto fptr1 = jit_engine_t::make(ctx)->get_entry_func(f1);
    fptr1->call_default(&input0_data[0], &weight0_data[0], &weight1_data[0],
            &ori_output0_data[0], &ori_output1_data[0]);

    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t graph2;
    auto f2 = get_two_consective_mmm(M, K1, K2, K3, ctx, graph2);
    std::stringstream ss;
    print_graph(graph2, ss, true);
    // split and merge the outer-most
    std::string expected_str
            = R"(graph(v0: f32[256, 2048], v1: f32[2048, 512], v2: f32[512, 1024]) -> [v3: f32[256, 512], v4: f32[256, 1024]] {
  [v3: f32[256, 512], v4: f32[256, 1024]] = outerloop_2_partition_managed_matmul_core_relu_managed_matmul_core_relu(v0, v1, v2)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
    auto fptr2 = jit_engine_t::make(ctx)->get_entry_func(f2);
    fptr2->call_default(&input0_data[0], &weight0_data[0], &weight1_data[0],
            &pass_output0_data[0], &pass_output1_data[0]);

    test_utils::compare_data(ori_output0_data, pass_output0_data, 1e-4, 1e-5);
    test_utils::compare_data(ori_output1_data, pass_output1_data, 1e-4, 1e-5);

#ifdef DO_BENCH
    auto exec = [&]() {
        fptr2->call_default(&input0_data[0], &weight0_data[0], &weight1_data[0],
                &pass_output0_data[0], &pass_output1_data[0]);
    };
    const int warm_up = 5;
    const int repeat = 5;
    const int loop = 20;
    double cost1 = 1e12;
    for (int r = 0; r < repeat; r++) {
        double cost1_r = 0.f;
        for (int t = 0; t < loop + warm_up; t++) {
            auto time1 = evaluate_time(exec);
            if (t >= warm_up) cost1_r += time1;
        }
        cost1 = std::min(cost1_r, cost1);
    }
    printf("\n@mlp cost: %f ms\n", cost1 / loop);
#endif
}

static ir_module_ptr get_three_consective_mmm(int M, int K1, int K2, int K3,
        int K4, context_ptr ctx, sc_graph_t &graph) {
    auto input0 = graph.make_input(
            {graph_tensor::make({M, K1}, sc_data_format_t(format_kinds::MK))});
    auto weight0 = graph.make_input(
            {graph_tensor::make({K1, K2}, sc_data_format_t(format_kinds::KN))});
    auto weight1 = graph.make_input(
            {graph_tensor::make({K2, K3}, sc_data_format_t(format_kinds::KN))});
    auto weight2 = graph.make_input(
            {graph_tensor::make({K3, K4}, sc_data_format_t(format_kinds::KN))});

    // mmm0
    auto mmm0 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    auto relu0 = graph.make("relu", {mmm0->get_outputs()[0]}, {}, {});
    // mmm1, using mmm0's output
    auto mmm1 = graph.make("managed_matmul_core",
            {relu0->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    auto relu1 = graph.make("relu", {mmm1->get_outputs()[0]}, {}, {});
    // mmm2, using mmm2's output
    auto mmm2 = graph.make("managed_matmul_core",
            {relu1->get_outputs()[0], weight2->get_outputs()[0]}, {}, {});
    auto relu2 = graph.make("relu", {mmm2->get_outputs()[0]}, {}, {});
    auto out0 = graph.make_output(relu0->get_outputs());
    auto out1 = graph.make_output(relu1->get_outputs());
    auto out2 = graph.make_output(relu2->get_outputs());

    graph_driver(graph, ctx);

    auto f1 = lower_graph(
            ctx, graph, {input0, weight0, weight1, weight2, out0, out1, out2});

    return f1;
}

TEST(GCCore_graph_mixed_partition_cpp, SplitAndMergeInners_Accuracy3) {
    thread_num_reset reseter;
    runtime_config_t::get().set_num_threads(56);

    int M = 1024, K1 = 13, K2 = 512, K3 = 256, K4 = 128;
    std::vector<float> input0_data(M * K1);
    test_utils::fill_data(&input0_data[0], M * K1);
    std::vector<float> weight0_data(K1 * K2);
    test_utils::fill_data(&weight0_data[0], K1 * K2);
    std::vector<float> weight1_data(K2 * K3);
    test_utils::fill_data(&weight1_data[0], K2 * K3);
    std::vector<float> weight2_data(K2 * K3);
    test_utils::fill_data(&weight2_data[0], K3 * K4);

    std::vector<float> ori_output0_data(M * K2);
    std::vector<float> ori_output1_data(M * K3);
    std::vector<float> ori_output2_data(M * K4);
    std::vector<float> pass_output0_data(M * K2);
    std::vector<float> pass_output1_data(M * K3);
    std::vector<float> pass_output2_data(M * K4);

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    // uncomment to dump output
    // ctx->flags_.graph_dump_results_ = "path=./dump";
    // old fusion mgr has correctness issue when facing imbalance under some
    // specific configs, here we use ref model for validation
    gemm_params gemm_param0 {
            false, false, 1024, 512, 13, 1.0, 0.0, 13, 512, 512};
    gemm_params gemm_param1 {
            false, false, 1024, 256, 512, 1.0, 0.0, 512, 256, 256};
    gemm_params gemm_param2 {
            false, false, 1024, 128, 256, 1.0, 0.0, 256, 128, 128};
    ref_gemm(gemm_param0, &input0_data[0], &weight0_data[0],
            &ori_output0_data[0]);
    ref_relu(ori_output0_data.data(), ori_output0_data.data(),
            ori_output0_data.size());
    ref_gemm(gemm_param1, &ori_output0_data[0], &weight1_data[0],
            &ori_output1_data[0]);
    ref_relu(ori_output1_data.data(), ori_output1_data.data(),
            ori_output1_data.size());
    ref_gemm(gemm_param2, &ori_output1_data[0], &weight2_data[0],
            &ori_output2_data[0]);
    ref_relu(ori_output2_data.data(), ori_output2_data.data(),
            ori_output2_data.size());

    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t graph;
    auto f = get_three_consective_mmm(M, K1, K2, K3, K4, ctx, graph);

    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
    fptr->call_default(&input0_data[0], &weight0_data[0], &weight1_data[0],
            &weight2_data[0], &pass_output0_data[0], &pass_output1_data[0],
            &pass_output2_data[0]);

    // fix-me (xxx): a special iim_block=19 will be given in this ut, making it
    // unable to converge with rtol=1e-4, atol=1e-5
    test_utils::compare_data(pass_output0_data, ori_output0_data, 1e-3, 1e-4);
    test_utils::compare_data(pass_output1_data, ori_output1_data, 1e-3, 1e-4);
    test_utils::compare_data(pass_output2_data, ori_output2_data, 1e-3, 1e-4);
}

TEST(GCCore_graph_mixed_partition_cpp, SplitOuterMostLoopWithTensorShrink) {
    thread_num_reset reseter;
    runtime_config_t::get().set_num_threads(16);
    int M = 256, K1 = 2048, K2 = 512, N = 1024;

    sc_graph_t graph;
    auto input0 = graph.make_input(
            {graph_tensor::make({M, K1}, sc_data_format_t(format_kinds::MK))});
    auto weight0 = graph.make_input(
            {graph_tensor::make({K1, K2}, sc_data_format_t(format_kinds::KN))});
    auto weight1 = graph.make_input(
            {graph_tensor::make({K2, N}, sc_data_format_t(format_kinds::KN))});

    // mmm0
    auto mmm0 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    // mmm1
    auto mmm1 = graph.make("managed_matmul_core",
            {mmm0->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    auto out0 = graph.make_output(mmm1->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    // split outmost and merge inners
    mixed_partition(graph, ctx);
    auto mixed_op = get_mixed_op_from_graph(graph);
    COMPILE_ASSERT(mixed_op, "No mixed fused op is found, please check")

    auto body = mixed_op->parti_list_[0]
                        ->get_outer_loops()
                        .back()
                        ->body_.checked_as<stmts>()
                        ->seq_;
    auto mm0_out = body[0].checked_as<define>()->var_;
    EXPECT_TRUE(mm0_out->attr().has_key(tensor_shrinker_attrs::should_shrink));
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphVerticalMergeForImageAffine) {
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
        auto relu_out1 = g.make("relu", {conv_data1->get_outputs()[0]}, {}, {});

        if (BS == 1) {
            auto mul0 = g.make("mul",
                    {relu_out0->get_outputs()[0], relu_out1->get_outputs()[0]},
                    {}, {});
            g.make_output(mul0->get_outputs());
        } else {
            g.make_output(relu_out0->get_outputs());
            g.make_output(relu_out1->get_outputs());
        }
        return g;
    };

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.use_cost_model_ = true;
    thread_num_reset reseter;
    int num_threads = 28;
    runtime_config_t::get().set_num_threads(num_threads);

    // case 1: can be merged
    graph = get_conv_block_graph(num_threads);
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[28, 64, 56, 56], v1: f32[128, 64, 1, 1], v2: f32[128, 64, 1, 1]) -> [v3: f32[28, 128, 56, 56], v4: f32[28, 128, 56, 56]] {
  [v4: f32[28, 128, 56, 56], v3: f32[28, 128, 56, 56]] = outerloop_28_partition_conv_fwd_core_relu_conv_fwd_core_relu(v0, v2, v1)
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
  [v4: f32[1, 128, 56, 56]] = outerloop_1X1X56_partition_conv_fwd_core_relu(v0, v2)
  [v3: f32[1, 128, 56, 56]] = outerloop_1X1X56_partition_conv_fwd_core_relu_mul(v0, v1, v4)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphMarkInplaceHint1) {
    sc_graph_t graph;
    int batch_size = 56;
    thread_num_reset reseter;
    runtime_config_t::get().set_num_threads(batch_size);
    auto input0 = graph.make_input({graph_tensor::make(
            {batch_size, 64, 32, 32}, sc_data_format_t::NCHW())});
    auto input1 = graph.make_input({graph_tensor::make(
            {batch_size, 64, 32, 32}, sc_data_format_t::NCHW())});
    auto input2 = graph.make_input({graph_tensor::make(
            {batch_size, 64, 32, 32}, sc_data_format_t::NCHW())});

    // cast0 is f32
    auto cast0 = graph.make("cast", {input0->get_outputs()[0]}, {},
            {{"dtype", datatypes::f32}});
    auto cast1 = graph.make(
            "cast", {cast0->get_outputs()[0]}, {}, {{"dtype", datatypes::s32}});
    auto mul0 = graph.make(
            "mul", {cast1->get_outputs()[0], input1->get_outputs()[0]}, {}, {});
    auto add0 = graph.make(
            "add", {mul0->get_outputs()[0], input2->get_outputs()[0]}, {}, {});
    // cast2 is also f32, can be inplaced with cast0
    auto cast2 = graph.make(
            "cast", {add0->get_outputs()[0]}, {}, {{"dtype", datatypes::f32}});
    auto cast3 = graph.make(
            "cast", {cast2->get_outputs()[0]}, {}, {{"dtype", datatypes::s32}});
    graph.make_output(cast3->get_outputs());

    auto ctx = get_test_ctx();
    mixed_partition(graph, ctx);

    mixed_fuse_op_t *fused_op = get_mixed_op_from_graph(graph);
    COMPILE_ASSERT(fused_op, "No mixed fused op is found, please check")

    auto body = fused_op->parti_list_[0]
                        ->get_outer_loops()
                        .back()
                        ->body_.checked_as<stmts>()
                        ->seq_;

    // search cast tensor define node.
    EXPECT_TRUE(std::any_of(body.begin(), body.end(), [](const stmt &s) {
        return s.cast<define>()
                .map([](const define &d) { return d->var_.as<tensor>(); })
                .filter([](const tensor &t) {
                    // check cast_7_outs_0 inplace attr
                    return t->name_ == "cast_7_outs_0"
                            && t->attr().has_key(
                                    attr_keys::tensor_inplace_hint);
                })
                .has_value();
    }));
}

TEST(GCCore_graph_mixed_partition_cpp, TestGraphMarkInplaceHint2) {
    sc_graph_t graph;
    int batch_size = 28;
    thread_num_reset reseter;
    runtime_config_t::get().set_num_threads(batch_size);
    auto input0 = graph.make_input({graph_tensor::make({batch_size, 64, 32})});

    auto relu0 = graph.make("relu", {input0->get_outputs()[0]}, {}, {});
    auto relu1 = graph.make("relu", {relu0->get_outputs()[0]}, {}, {});

    auto relu2 = graph.make("relu", {relu1->get_outputs()[0]}, {}, {});
    // relu3 should not inplace relu0, due to relu1 share same buffer with
    // relu0, which is not truely last use for relu_0_out buffer
    auto relu3 = graph.make("relu", {relu1->get_outputs()[0]}, {}, {});
    auto mul0 = graph.make(
            "mul", {relu2->get_outputs()[0], relu3->get_outputs()[0]}, {}, {});
    graph.make_output(mul0->get_outputs());

    auto ctx = get_test_ctx();
    mixed_partition(graph, ctx);

    mixed_fuse_op_t *fused_op = get_mixed_op_from_graph(graph);
    COMPILE_ASSERT(fused_op, "No mixed fused op is found, please check")

    auto body = fused_op->parti_list_[0]
                        ->get_outer_loops()
                        .back()
                        ->body_.checked_as<stmts>()
                        ->seq_;

    // search relu tensor define node.
    EXPECT_TRUE(std::any_of(body.begin(), body.end(), [](const stmt &s) {
        return s.cast<define>()
                .map([](const define &d) { return d->var_.as<tensor>(); })
                .filter([](const tensor &t) {
                    // check relu_4_outs_0 inplace attr
                    return t->name_ == "relu_4_outs_0"
                            && !t->attr().has_key(
                                    attr_keys::tensor_inplace_hint);
                })
                .has_value();
    }));
}

TEST(GCCore_graph_mixed_partition_cpp, TestUserDefinedShrintTensor) {
    sc_graph_t graph;
    auto ctx = get_test_ctx();
    auto input = graph.make_input({graph_tensor::make({56, 16, 196})});
    auto weight0 = graph.make_input({graph_tensor::make({32, 16, 1})});
    auto weight1 = graph.make_input({graph_tensor::make({64, 32, 1})});

    auto conv0 = graph.make("conv_fwd_core",
            {input->get_outputs()[0], weight0->get_outputs()[0]}, {},
            {{"strides", sc_dims {2}}, {"paddings", sc_dims {0}}});
    auto conv1 = graph.make("conv_fwd_core",
            {conv0->get_outputs()[0], weight1->get_outputs()[0]}, {},
            {{"strides", sc_dims {2}}, {"paddings", sc_dims {0}}});
    graph.make_output(conv1->get_outputs());
    layout_propagation(graph, ctx);
    mixed_partition(graph, ctx);
    auto mod = lower_graph(ctx, graph, {});
    auto jitf = jit_engine_t::make(mod->ctx_)->get_entry_func(mod, true);
    ASSERT_TRUE(jitf);
}

TEST(GCCore_graph_mixed_partition_cpp, TestInputFusionAnchor) {
    sc_graph_t graph;
    auto ctx = get_test_ctx();
    auto input = graph.make_input(
            {graph_tensor::make({128, 128}, sc_data_format_t::MK())});
    auto weight = graph.make_input(
            {graph_tensor::make({128, 128}, sc_data_format_t::KN())});

    auto reo0 = graph.make("reorder", {input->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(64, 64)}});
    auto reo1 = graph.make("reorder", {weight->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::NKkn(64, 32)}});

    auto gemm = graph.make("managed_matmul_core",
            {reo0->get_outputs()[0], reo1->get_outputs()[0]}, {}, {});
    graph.make_output(gemm->get_outputs());
    thread_num_reset reseter;
    runtime_config_t::get().set_num_threads(32);
    ops::managed_matmul_core_config_t cfg = {32, 1, 1, 1, 1, 0};
    for (auto &op : graph.ops_) {
        if (op->op_name_ == "managed_matmul_core") {
            auto matmul_op = op->dyn_cast<ops::managed_matmul_core_op_t>();
            matmul_op->set_config(reflection::general_object_t::make(cfg));
        }
    }
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[128, 128], v1: f32[128, 128]) -> [v2: f32[128, 128]] {
  [v2: f32[128, 128]] = outerloop_32X1X1_partition_reorder_reorder_managed_matmul_core(v1, v0)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

class test_prefetchable_op : public tunable_op_t,
                             public op_traits::may_prefetch_t {
public:
    std::vector<int> indices_;
    test_prefetchable_op(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &producer_lt,
            const std::vector<graph_tensor_ptr> &consumer_lt,
            const any_map_t &attrs)
        : tunable_op_t(op_name, producer_lt, {producer_lt[0]->copy()}, attrs) {}

    body_generator_ptr create_generator() override { return nullptr; }
    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override {}

    std::vector<int> query_prefetch(const context_ptr &ctx, bool is_global,
            const std::vector<tensor_slice> &ins) override {
        return {0, 1};
    }

    void generate_prefetcher_body_for_tensor(const context_ptr &ctx,
            const std::vector<expr> &func_args, const std::vector<expr> &ins,
            const std::vector<int> &indices) override {
        indices_ = indices;
    }

    virtual sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override {
        return mgr.make<test_prefetchable_op>("tperf", ins, outs, any_map_t {});
    }
    ir_module_ptr get_func(context_ptr ctx) override { return nullptr; }
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override {}
};

TEST(GCCore_graph_mixed_partition_cpp, TestPrefetchNone) {
    sc_graph_t graph;
    auto ctx = get_test_ctx();

    auto input = graph.make_input(
            {graph_tensor::make({128, 128}, sc_data_format_t::MK())});
    auto weight = graph.make_input(
            {graph_tensor::make({128, 128}, sc_data_format_t::KN())});

    auto reo0 = graph.make("reorder", {input->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(64, 64)}});
    auto reo1 = graph.make("reorder", {weight->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::NKkn(64, 32)}});

    auto gemm = graph.make<test_prefetchable_op>("tperf",
            std::vector<graph_tensor_ptr> {
                    reo0->get_outputs()[0], reo1->get_outputs()[0]},
            std::vector<graph_tensor_ptr> {}, any_map_t {});
    graph.make_output(gemm->get_outputs());

    mixed_fuse_op_t tester {"test", {}, nullptr, graph,
            {graph_tensor::make({128, 128}, sc_data_format_t::MK()),
                    graph_tensor::make({128, 128}, sc_data_format_t::MK())},
            {graph_tensor::make({128, 128}, sc_data_format_t::MK())}, {}};
    EXPECT_EQ(tester.query_prefetch(ctx, true, {}), std::vector<int>());
}

TEST(GCCore_graph_mixed_partition_cpp, TestPrefetchSelected) {
    sc_graph_t graph;
    auto ctx = get_test_ctx();

    auto input = graph.make_input(
            {graph_tensor::make({128, 128}, sc_data_format_t::MK())});
    auto weight = graph.make_input(
            {graph_tensor::make({128, 128}, sc_data_format_t::KN())});

    auto reo1 = graph.make("reorder", {weight->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::NKkn(64, 32)}});

    auto gemm = graph.make<test_prefetchable_op>("tperf",
            std::vector<graph_tensor_ptr> {
                    reo1->get_outputs()[0], input->get_outputs()[0]},
            std::vector<graph_tensor_ptr> {}, any_map_t {});
    graph.make_output(gemm->get_outputs());

    mixed_fuse_op_t tester {"test", {}, nullptr, graph,
            {graph_tensor::make({128, 128}, sc_data_format_t::MK()),
                    graph_tensor::make({128, 128}, sc_data_format_t::MK())},
            {graph_tensor::make({128, 128}, sc_data_format_t::MK())}, {}};

    // prefetch op returns {0,1}, and only 1 is input. Input 1 of the op maps to
    // the input 0 of the graph
    EXPECT_EQ(tester.query_prefetch(ctx, true, {}), std::vector<int>({0}));

    tester.generate_prefetcher_body_for_tensor(ctx, {}, {}, {0});
    bool found = false;
    for (auto &op : tester.sub_graph_.ops_) {
        if (auto theop = op->dyn_cast<test_prefetchable_op>()) {
            EXPECT_EQ(theop->indices_, std::vector<int>({1}));
            found = true;
        }
    }
    EXPECT_TRUE(found);
}
