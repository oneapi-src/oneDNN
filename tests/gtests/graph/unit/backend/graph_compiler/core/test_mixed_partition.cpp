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
#include "test_graph.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/simplify.hpp>
#include <compiler/ir/viewer.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/convolution.hpp>
#include <ops/fusible/padding.hpp>
#include <ops/fusible/pooling.hpp>
#include <ops/managed_matmul_core.hpp>
#include <ops/matmul_core.hpp>
#include <ops/templates/conv_fwd.hpp>
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphFuseOpPass) {
    SET_THREADS_OR_SKIP(28);

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

    ops::conv_fwd_config_t cfg = {32, 32, 1, 1, 1, 1, 0, 1};
    conv->dyn_cast<ops::conv_fwd_core_op_t>()->set_config(
            reflection::general_object_t::make(cfg));
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[28, 2, 16, 16, 32], v1: f32[28, 2, 16, 16, 32], v2: f32[2, 2, 1, 1, 32, 32], v3: f32[1, 2, 16, 16, 32]) -> [v4: f32[1, 2, 16, 16, 32]] {
  [v5: f32[28, 2, 16, 16, 32]] = outerloop_28X1X2X16_partition_conv_fwd_core_add_relu(v0, v2, v1)
  [v6: f32[1, 2, 16, 16, 32]] = reduce(v5)
  [v4: f32[1, 2, 16, 16, 32]] = outerloop_1X2X16X16_partition_add_relu(v6, v3)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestFuseOpBreakAndNoFuse) {
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
                {graph_tensor::make({M, N}, sc_data_format_t::MK())}, {});
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphBatchWiseFuse) {
    REQUIRE_PARALLEL();
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
  [v4: f32[16, 32, 384, 1024]] = outerloop_16X384_partition_add_reduce_compute_reduce_collect_add_add(v0, v1, v2)
  [v3: f32[8, 2, 32, 384, 1024]] = tensor_view(v4)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphHorizontalMerge) {
    SET_THREADS_OR_SKIP(32);

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

    graph_driver_before_fusion(graph, ctx);
    ops::matmul_core_config_t cfg = {32, 32, 32};
    for (auto &op : graph.ops_) {
        if (op->op_name_ == "matmul_core") {
            auto matmul_op = op->dyn_cast<ops::matmul_core_op_t>();
            matmul_op->set_config(reflection::general_object_t::make(cfg));
        }
    }

    int M_num_block = M / cfg.M_block;
    int N_num_block = N / cfg.N_block;
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphRunSingleThreads) {
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
    SET_THREADS_OR_SKIP(1);
    graph_driver_before_fusion(graph, ctx);
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

TEST(GCCore_CPU_graph_mixed_partition_cpp,
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphOptimizedReorder) {
    REQUIRE_AVX2();
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphLastDimPaddedReorder) {
    REQUIRE_AVX2();
    sc_graph_t graph;

    auto input0 = graph.make_input(
            {graph_tensor::make({32, 31}, sc_data_format_t::MKmk(4, 48))});
    auto input1 = graph.make_input(
            {graph_tensor::make({32, 31}, sc_data_format_t::MKmk(4, 32))});

    auto relu_node = graph.make("relu", {input0->get_outputs()[0]}, {}, {});
    // This reorder is speical reoder case with last dim padded and would not
    // break following `mul` fusion
    auto reo_node = graph.make("reorder", {relu_node->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(4, 32)}});
    auto mul_node = graph.make("mul",
            {reo_node->get_outputs()[0], input1->get_outputs()[0]}, {}, {});

    auto output0 = graph.make_output(mul_node->get_outputs());

    auto ctx = get_test_ctx();
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[8, 1, 4, 48], v1: f32[8, 1, 4, 32]) -> [v2: f32[8, 1, 4, 32]] {
  [v2: f32[8, 1, 4, 32]] = outerloop_8X1X4_partition_relu_reorder_mul(v0, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphFuseBRGemmPreOpFusion) {
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
    ASSERT_TRUE(fused_op && fused_op->parti_list_.size() == 1);
    // Due to brgemm pre-op fusion, the number of outer loops is equal to what
    // is written in conv template
    EXPECT_EQ(fused_op->parti_list_[0]->get_outer_loops().size(), (size_t)4);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphFuseOptimizedReduce2) {
    sc_graph_t graph;

    SET_THREADS_OR_SKIP(56);

    auto input0 = graph.make_input({graph_tensor::make({1024, 1024})});
    // This reduce op does not satisfy register requirement of `tsr2var`
    // optimization due to 1024 is larger than max tolerence.
    auto reduce0 = graph.make("reduce", input0->get_outputs(), {},
            {{"rd_axis", std::vector<int> {0}}, {"rd_op", 0}});
    graph.make_output(reduce0->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    graph_driver_before_fusion(graph, ctx);
    bool found = false;
    for (auto &op : graph.ops_) {
        if (op->op_name_ == "reduce_compute") {
            auto mod = op->get_func(ctx);
            // this reduce_compute_op should not be split
            found = (mod->get_func("reduce_compute_2") != nullptr);
        }
    }
    ASSERT_TRUE(found);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphPartitionRingRiskCheck1) {
    sc_graph_t graph;

    SET_THREADS_OR_SKIP(1);

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

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphBreakOpPreFusion1) {
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphBreakOpPreFusion2) {
    sc_graph_t graph;

    int BS = 28, C = 64, H = 56, W = 56, K = 64;

    SET_THREADS_OR_SKIP(BS);

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
            = R"(graph(v0: f32[28, 64, 56, 56], v1: f32[64, 64, 1, 1], v2: f32[64, 64, 1, 1]) -> [v3: f32[28, 64, 56, 56]] {
  [v3: f32[28, 64, 56, 56]] = outerloop_28_partition_relu_conv_fwd_core_relu_conv_fwd_core_cast_cast_add(v0, v2, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestaxisBinding1) {
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestaxisBinding2) {
    sc_graph_t graph;

    SET_THREADS_OR_SKIP(2);

    int M, N, K;
    M = N = 256;
    K = 64;

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
            = R"(graph(v0: f32[256, 64], v1: f32[256], v2: f32[64, 256], v3: f32[64, 256]) -> [v4: f32[256, 256], v5: f32[256, 256]] {
  [v6: f32[1, 256]] = tensor_view(v1)
  [v5: f32[256, 256], v4: f32[256, 256]] = outerloop_2X1X1X1X1_partition_managed_matmul_core_add_managed_matmul_core_add(v0, v3, v6, v2)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

/* Case: same weight
out1 = x1 * w, out2 = x2 * w
*/
TEST(GCCore_CPU_graph_mixed_partition_cpp, SplitAndMergeInners_Accuracy0) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(16);

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
    bool is_scpi = ctx->machine_.cpu_flags_.is_skx_like();
    if (is_scpi) {
        // managed matmul core will have different config under such machine
        // Only compare result in this case
        EXPECT_EQ(ss.str(), expected_str);
    }

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

    test_utils::compare_data(ori_output0_data, pass_output0_data, 1e-4f, 1e-5f);
    test_utils::compare_data(ori_output1_data, pass_output1_data, 1e-4f, 1e-5f);
}

/* Case: same input_data
out1 = x1 * w1, out2 = x1 * w2
*/
TEST(GCCore_CPU_graph_mixed_partition_cpp, SplitAndMergeInners_Accuracy1) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(16);

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
    ops::managed_matmul_core_config_t cfg = {16, 1, 1, 1, 1, 0};
    for (auto &op : graph.ops_) {
        if (op->op_name_ == "managed_matmul_core") {
            auto matmul_op = op->dyn_cast<ops::managed_matmul_core_op_t>();
            matmul_op->set_config(reflection::general_object_t::make(cfg));
        }
    }

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

    test_utils::compare_data(ori_output0_data, pass_output0_data, 1e-4f, 1e-5f);
    test_utils::compare_data(ori_output1_data, pass_output1_data, 1e-4f, 1e-5f);
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, SplitAndMergeInners_Accuracy2) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(16);

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
    bool is_scpi = ctx->machine_.cpu_flags_.is_skx_like();
    if (is_scpi) {
        // managed matmul core will have different config under such machine
        // Only compare result in this case
        EXPECT_EQ(ss.str(), expected_str);
    }
    auto fptr2 = jit_engine_t::make(ctx)->get_entry_func(f2);
    fptr2->call_default(&input0_data[0], &weight0_data[0], &weight1_data[0],
            &pass_output0_data[0], &pass_output1_data[0]);

    test_utils::compare_data(ori_output0_data, pass_output0_data, 1e-4f, 1e-5f);
    test_utils::compare_data(ori_output1_data, pass_output1_data, 1e-4f, 1e-5f);

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
    // mmm2, using mmm1's output
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, SplitAndMergeInners_Accuracy3) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(56);

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
    // unable to converge with rtol=1e-4f, atol=1e-5f
    test_utils::compare_data(pass_output0_data, ori_output0_data, 5e-2f, 1e-4f);
    test_utils::compare_data(pass_output1_data, ori_output1_data, 5e-2f, 1e-4f);
    test_utils::compare_data(pass_output2_data, ori_output2_data, 5e-2f, 1e-4f);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, SplitOuterMostLoopWithTensorShrink) {
    SET_THREADS_OR_SKIP(16);
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
    ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
    auto body = mixed_op->parti_list_[0]
                        ->get_outer_loops()
                        .back()
                        ->body_.checked_as<stmts>()
                        ->seq_;
    auto mm0_out = body[0].checked_as<define>()->var_;
    bool is_scpi = ctx->machine_.cpu_flags_.is_skx_like();
    if (is_scpi) {
        // managed matmul core will have different config under such machine
        // Only compare result in this case
        EXPECT_TRUE(
                mm0_out->attr().has_key(tensor_shrinker_attrs::should_shrink));
    }
}

TEST(GCCore_CPU_graph_mixed_partition_cpp,
        ParitialReduceMatmulTensorViewShrink) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(16);
    int M = 256, K = 256, N = 320;

    sc_graph_t graph;
    auto input = graph.make_input(
            {graph_tensor::make({M, K}, sc_data_format_t(format_kinds::MK))});
    auto weight = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});

    // mmm + tensor_view + post_op
    auto mmm = graph.make("managed_matmul_core",
            {input->get_outputs()[0], weight->get_outputs()[0]}, {}, {});
    auto tv = graph.make("tensor_view", {mmm->get_outputs()[0]},
            {graph_tensor::make(sc_dims {1, M, N}, sc_data_format_t(),
                    mmm->get_outputs()[0]->details_.dtype_)},
            {{"shape", sc_dims {1, M, N}}, {"format", sc_data_format_t()}});
    auto relu = graph.make("relu", {tv->get_outputs()[0]}, {}, {});
    auto out = graph.make_output(relu->get_outputs());

    ops::managed_matmul_core_config_t cfg = {1, 16, 1, 1, 4, 0};
    for (auto &op : graph.ops_) {
        if (op->op_name_ == "managed_matmul_core") {
            auto matmul_op = op->dyn_cast<ops::managed_matmul_core_op_t>();
            matmul_op->set_config(reflection::general_object_t::make(cfg));
        }
    }

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;

    auto f1 = lower_graph(ctx, graph, {input, weight, out});
    auto fptr1 = jit_engine_t::make(ctx)->get_entry_func(f1);

    graph_driver(graph, ctx);

    auto f2 = lower_graph(ctx, graph, {input, weight, out});
    auto fptr2 = jit_engine_t::make(ctx)->get_entry_func(f2);

    std::vector<float> input_data(M * K);
    test_utils::fill_data(&input_data[0], M * K);
    std::vector<float> weight_data(K * N);
    test_utils::fill_data(&weight_data[0], K * N);

    std::vector<float> ori_output_data(M * N);
    fptr1->call_default(&input_data[0], &weight_data[0], &ori_output_data[0]);

    std::vector<float> pass_output_data(M * N);
    fptr2->call_default(&input_data[0], &weight_data[0], &pass_output_data[0]);

    test_utils::compare_data(ori_output_data, pass_output_data, 1e-4f, 1e-5f);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphMarkInplaceHint1) {
    sc_graph_t graph;
    int batch_size = 56;
    SET_THREADS_OR_SKIP(batch_size);
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
    ASSERT_TRUE(fused_op && fused_op->parti_list_.size() == 1);
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphMarkInplaceHint2) {
    sc_graph_t graph;
    int batch_size = 28;
    SET_THREADS_OR_SKIP(batch_size);
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
    ASSERT_TRUE(fused_op && fused_op->parti_list_.size() == 1);
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphMarkInplaceHint4) {
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
    auto relu0 = graph.make("relu", {cast0->get_outputs()[0]}, {}, {});
    // matmul0 will use relu0_out in larger anchor, as the result, it should not
    // be set inplace hint
    auto matmul0 = graph.make("matmul_core",
            {relu0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});

    auto output0 = graph.make_output(matmul0->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    graph_driver_before_fusion(graph, ctx);
    mixed_partition(graph, ctx);

    mixed_fuse_op_t *fused_op = get_mixed_op_from_graph(graph);
    ASSERT_TRUE(fused_op && fused_op->parti_list_.size() == 1);
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
                    // check relu_3_outs_0 inplace attr
                    return t->name_ == "relu_3_outs_0"
                            && !t->attr().has_key(
                                    attr_keys::tensor_inplace_hint);
                })
                .has_value();
    }));
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphMarkInplaceHint5) {
    sc_graph_t graph;

    int run_threads = 28;
    SET_THREADS_OR_SKIP(run_threads);

    auto input0 = graph.make_input({graph_tensor::make({28, 100, 200})});
    auto input1 = graph.make_input({graph_tensor::make({28, 200, 100})});

    auto relu0 = graph.make("relu", {input0->get_outputs()[0]}, {}, {});
    // tensorptr
    auto tv0 = graph.make("tensor_view", {relu0->get_outputs()[0]}, {},
            {{"shape", sc_dims {28, 200, 100}}});
    // add0 out will try to inplace tensorptr buffer
    auto add0 = graph.make(
            "add", {tv0->get_outputs()[0], input1->get_outputs()[0]}, {}, {});
    // add0 is also output of the graph
    auto output0 = graph.make_output(add0->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    mixed_partition(graph, ctx);

    mixed_fuse_op_t *fused_op = get_mixed_op_from_graph(graph);
    EXPECT_TRUE(fused_op && fused_op->parti_list_.size() == 1);
    // get output buffer named `add_4_outs_0`
    auto output_buf = fused_op->parti_list_[0]->func_->params_[0];
    // the output buffer should not do inplace due to tensorptr found
    EXPECT_TRUE(!output_buf->attr().has_key(attr_keys::tensor_inplace_hint));
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphMarkInplaceHint7) {
    sc_graph_t graph;
    int batch_size = 56;
    SET_THREADS_OR_SKIP(batch_size);
    auto input0 = graph.make_input({graph_tensor::make(
            {batch_size, 64, 32, 32}, sc_data_format_t::NCHW())});
    auto input1 = graph.make_input({graph_tensor::make(
            {batch_size, 64, 32, 32}, sc_data_format_t::NCHW())});
    auto input2 = graph.make_input({graph_tensor::make(
            {batch_size, 64, 32, 32}, sc_data_format_t::NCHW())});

    auto mul0 = graph.make("mul",
            {input0->get_outputs()[0], input1->get_outputs()[0]}, {}, {});
    // add0 could not do inplace on mul0, due to it is not last user
    auto add0 = graph.make(
            "add", {mul0->get_outputs()[0], input2->get_outputs()[0]}, {}, {});
    auto sub0 = graph.make(
            "sub", {add0->get_outputs()[0], add0->get_outputs()[0]}, {}, {});
    auto div0 = graph.make(
            "div", {mul0->get_outputs()[0], input1->get_outputs()[0]}, {}, {});
    auto add1 = graph.make(
            "add", {div0->get_outputs()[0], sub0->get_outputs()[0]}, {}, {});
    auto relu0 = graph.make("relu", {add1->get_outputs()[0]}, {}, {});
    graph.make_output(relu0->get_outputs());

    auto ctx = get_test_ctx();
    mixed_partition(graph, ctx);

    mixed_fuse_op_t *fused_op = get_mixed_op_from_graph(graph);
    ASSERT_TRUE(fused_op && fused_op->parti_list_.size() == 1);
    auto body = fused_op->parti_list_[0]
                        ->get_outer_loops()
                        .back()
                        ->body_.checked_as<stmts>()
                        ->seq_;

    // add_4_outs_0 could not inplace mul_3_outs_0
    EXPECT_TRUE(std::any_of(body.begin(), body.end(), [](const stmt &s) {
        return s.cast<define>()
                .map([](const define &d) { return d->var_.as<tensor>(); })
                .filter([](const tensor &t) {
                    return t->name_ == "add_4_outs_0"
                            && !t->attr().has_key(
                                    attr_keys::tensor_inplace_hint);
                })
                .has_value();
    }));
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestGraphMarkInplaceHint8) {
    int run_threads = 28;
    SET_THREADS_OR_SKIP(run_threads);

    sc_graph_t graph;
    auto input_shape = sc_dims {28, 16, 56, 56};
    auto weight_shape = sc_dims {16, 16, 3, 3};
    const sc_dims stride = {1, 1};
    const sc_dims padding_conv = {0, 0};
    const sc_dims padding_pad = {1, 1};

    auto in_data = graph.make_input({std::make_shared<graph_tensor>(
            nullptr, sc_data_format_t::NCHW(), input_shape, datatypes::f32)});
    auto in_weight = graph.make_input({std::make_shared<graph_tensor>(
            nullptr, sc_data_format_t::KCRS(), weight_shape, datatypes::f32)});
    auto conv_out = graph.make("conv_fwd_core",
            {in_data->get_outputs()[0], in_weight->get_outputs()[0]}, {},
            {{"strides", stride}, {"pads_begin", padding_conv},
                    {"pads_end", padding_conv}});
    auto relu0 = graph.make("relu", conv_out->get_outputs(), {}, {});
    // relu1 output could have inplace on relu0 output
    auto relu1 = graph.make("relu", relu0->get_outputs(), {}, {});
    // pre-padding replace the output of relu1, as the result, it should
    // remove related inpalce hint at the same time
    auto pad_out = graph.make("padding", {relu1->get_outputs()[0]}, {},
            {{"pads_begin", padding_pad}, {"pads_end", padding_pad},
                    {"break_post_fuse", true}});
    auto output0 = graph.make_output(pad_out->get_outputs());

    auto ctx = get_test_ctx();
    mixed_partition(graph, ctx);

    mixed_fuse_op_t *fused_op = get_mixed_op_from_graph(graph);
    ASSERT_TRUE(fused_op && fused_op->parti_list_.size() == 1);
    // get inplace map
    auto inplace_map = fused_op->parti_list_[0]->buf_alloc_.inplace_map_;
    // there is expected nothing in inplace map, due to `relu_4_outs_0 ==>
    // relu_3_outs_0` hint is removed
    EXPECT_EQ(inplace_map.size(), (size_t)0);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestUserDefinedShrintTensor) {
    REQUIRE_AVX2();
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

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestInputFusionAnchor1) {
    sc_graph_t graph;
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    auto input = graph.make_input(
            {graph_tensor::make({128, 32}, sc_data_format_t::MK())});
    auto weight = graph.make_input(
            {graph_tensor::make({32, 128}, sc_data_format_t::KN())});

    auto reo0 = graph.make("reorder", {input->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(16, 16)}});
    auto reo1 = graph.make("reorder", {weight->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::NKkn(16, 16)}});

    auto gemm = graph.make("managed_matmul_core",
            {reo0->get_outputs()[0], reo1->get_outputs()[0]}, {}, {});
    graph.make_output(gemm->get_outputs());
    SET_THREADS_OR_SKIP(32);
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
            = R"(graph(v0: f32[128, 32], v1: f32[32, 128]) -> [v2: f32[128, 128]] {
  [v2: f32[128, 128]] = outerloop_32X1X1_partition_reorder_reorder_managed_matmul_core(v1, v0)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestInputFusionAnchor2) {
    sc_graph_t graph;
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    auto input = graph.make_input(
            {graph_tensor::make({128, 16}, sc_data_format_t::MK())});
    auto weight = graph.make_input(
            {graph_tensor::make({1, 32}, sc_data_format_t::MKmk(4, 16))});

    auto reo0 = graph.make("reorder", {input->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(16, 2)}});
    // This reorder could not be executed input fusion because it use input loop
    auto reo1 = graph.make("reorder", {weight->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::NKkn(1, 16)}});

    auto gemm = graph.make("managed_matmul_core",
            {reo0->get_outputs()[0], reo1->get_outputs()[0]}, {}, {});
    graph.make_output(gemm->get_outputs());
    SET_THREADS_OR_SKIP(32);
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
    // reorder1 should not be fused
    std::string expected_str
            = R"(graph(v0: f32[128, 16], v1: f32[1, 2, 4, 16]) -> [v2: f32[128, 32]] {
  [v3: f32[2, 1, 1, 16]] = reorder(v1)
  [v2: f32[128, 32]] = outerloop_32X1X1_partition_reorder_managed_matmul_core(v0, v3)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestMergeMixedPartiVertically1) {
    sc_graph_t graph;
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    auto input0 = graph.make_input(
            {graph_tensor::make({4, 4096}, sc_data_format_t::MK())});
    auto weight0 = graph.make_input(
            {graph_tensor::make({4096, 11008}, sc_data_format_t::KN())});
    auto input1 = graph.make_input(
            {graph_tensor::make({4, 4096}, sc_data_format_t::MK())});
    auto weight1 = graph.make_input(
            {graph_tensor::make({4096, 11008}, sc_data_format_t::KN())});

    auto gemm0 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    auto sig = graph.make("sigmoid", {gemm0->get_outputs()[0]}, {}, {});
    auto reo0 = graph.make("reorder", {sig->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(4, 16)},
                    {"internal", true}});
    auto gemm1 = graph.make("managed_matmul_core",
            {input1->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    auto relu = graph.make("sigmoid", {gemm1->get_outputs()[0]}, {}, {});
    auto reo1 = graph.make("reorder", {relu->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(4, 32)},
                    {"internal", true}});

    auto add = graph.make(
            "add", {reo1->get_outputs()[0], reo0->get_outputs()[0]}, {}, {});
    graph.make_output(add->get_outputs());
    SET_THREADS_OR_SKIP(56);
    ops::managed_matmul_core_config_t cfg = {1, 56, 1, 1, 1, 0};
    for (auto &op : graph.ops_) {
        if (op->op_name_ == "managed_matmul_core") {
            auto matmul_op = op->dyn_cast<ops::managed_matmul_core_op_t>();
            matmul_op->set_config(reflection::general_object_t::make(cfg));
        }
    }
    graph_driver(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[4, 4096], v1: f32[4096, 11008], v2: f32[4, 4096], v3: f32[4096, 11008]) -> [v4: f32[4, 11008]] {
  [v4: f32[4, 11008]] = outerloop_1X56X1X1X1_partition_managed_matmul_core_sigmoid_reorder_managed_matmul_core_sigmoid_reorder_add_reorder(v2, v3, v0, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestMergeMixedPartiVertically2) {
    sc_graph_t graph;
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    auto input0 = graph.make_input(
            {graph_tensor::make({4, 4096}, sc_data_format_t::MK())});
    auto weight0 = graph.make_input(
            {graph_tensor::make({4096, 11008}, sc_data_format_t::KN())});
    auto input1 = graph.make_input(
            {graph_tensor::make({4, 4096}, sc_data_format_t::MK())});
    auto weight1 = graph.make_input(
            {graph_tensor::make({4096, 11008}, sc_data_format_t::KN())});

    auto gemm0 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    auto sig = graph.make("sigmoid", {gemm0->get_outputs()[0]}, {}, {});
    auto reo0 = graph.make("reorder", {sig->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(4, 16)},
                    {"internal", true}});
    auto gemm1 = graph.make("managed_matmul_core",
            {input1->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    auto reo1 = graph.make("reorder", {gemm1->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(4, 32)},
                    {"internal", true}});

    auto add = graph.make(
            "add", {reo0->get_outputs()[0], reo1->get_outputs()[0]}, {}, {});
    graph.make_output(add->get_outputs());
    SET_THREADS_OR_SKIP(56);
    ops::managed_matmul_core_config_t cfg = {1, 56, 1, 1, 1, 0};
    for (auto &op : graph.ops_) {
        if (op->op_name_ == "managed_matmul_core") {
            auto matmul_op = op->dyn_cast<ops::managed_matmul_core_op_t>();
            matmul_op->set_config(reflection::general_object_t::make(cfg));
        }
    }
    graph_driver(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[4, 4096], v1: f32[4096, 11008], v2: f32[4, 4096], v3: f32[4096, 11008]) -> [v4: f32[4, 11008]] {
  [v4: f32[4, 11008]] = outerloop_1X56X1X1X1_partition_managed_matmul_core_managed_matmul_core_sigmoid_reorder_add_reorder(v2, v3, v0, v1)
}
)";
    bool is_special_fm = ctx->machine_.cpu_flags_.is_spr_like();
    if (is_special_fm) {
        // managed matmul core will have different config under such machine
        // Only compare result in this case
        EXPECT_EQ(ss.str(), expected_str);
    }
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestMergeMixedPartiVertically3) {
    sc_graph_t graph;

    SET_THREADS_OR_SKIP(8);

    int M, N, K;
    M = N = 256;
    K = 64;

    auto input0 = graph.make_input(
            {graph_tensor::make({M, K}, sc_data_format_t(format_kinds::MN))});
    auto bias = graph.make_input(
            {graph_tensor::make({N}, sc_data_format_t(format_kinds::A))});
    auto weight0 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});
    auto weight1 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});

    ops::matmul_core_config_t cfg = {32, 32, 32};
    // mm0
    auto mm0 = graph.make("matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    // set config
    mm0->dyn_cast<ops::matmul_core_op_t>()->set_config(
            reflection::general_object_t::make(cfg));
    mm0 = graph.make("add", {mm0->get_outputs()[0], bias->get_outputs()[0]}, {},
            {{"bc_axis", std::vector<int> {1}}});
    // mm1
    auto mm1 = graph.make("matmul_core",
            {input0->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    // set config
    mm1->dyn_cast<ops::matmul_core_op_t>()->set_config(
            reflection::general_object_t::make(cfg));
    mm1 = graph.make("add", {mm1->get_outputs()[0], bias->get_outputs()[0]}, {},
            {{"bc_axis", std::vector<int> {1}}});
    graph.make_output(mm0->get_outputs());
    graph.make_output(mm1->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;
    graph_driver(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    // mm0 and mm1 are exactly two forking partitions, as the result, they
    // should not be merged vertically, due to no benifit expected.
    std::string expected_str
            = R"(graph(v0: f32[256, 64], v1: f32[256], v2: f32[64, 256], v3: f32[64, 256]) -> [v4: f32[256, 256], v5: f32[256, 256]] {
  [v6: f32[1, 256]] = tensor_view(v1)
  [v5: f32[256, 256]] = outerloop_8X8_partition_matmul_core_add(v0, v3, v6)
  [v4: f32[256, 256]] = outerloop_8X8_partition_matmul_core_add(v0, v2, v6)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestMergeMixedPartiVertically4) {
    REQUIRE_BF16();
    int num_threads = 56;
    SET_THREADS_OR_SKIP(num_threads);

    sc_graph_t graph;
    auto input0 = graph.make_input({graph_tensor::make(
            {dimensions::dynamic_any, dimensions::dynamic_any, 16, 256},
            sc_data_format_t(), datatypes::bf16)});
    auto weight0 = graph.make_input({graph_tensor::make(
            {dimensions::dynamic_any, dimensions::dynamic_any, 16, 256},
            sc_data_format_t(), datatypes::bf16)});
    auto trans0 = graph.make("transpose", input0->get_outputs(), {},
            {{"order", std::vector<int> {0, 2, 1, 3}}});
    auto trans1 = graph.make("transpose", weight0->get_outputs(), {},
            {{"order", std::vector<int> {0, 2, 3, 1}}});
    // bmm0 may find no suitable anchor after two input partitions merged
    auto bmm0 = graph.make("matmul_core",
            {trans0->get_outputs()[0], trans1->get_outputs()[0]}, {}, {});
    graph.make_output(bmm0->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;
    graph_driver(graph, ctx);
    std::vector<sc_op_ptr> lower_args(graph.get_output_ops());
    auto input_ops = graph.get_input_ops();
    lower_args.insert(lower_args.end(), input_ops.begin(), input_ops.end());
    // During dynamic dispatch stage, some `block` may cause exception if no
    // fall-back mechanism supported
    auto mod = lower_graph(ctx, graph, lower_args);
    // the bmm will be fused into one of suitable input partition instead of
    // merging them
    EXPECT_TRUE(mod);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestMergeMixedPartiVertically5) {
    REQUIRE_AMX();
    SET_THREADS_OR_SKIP(16);

    int M = 4, N = 11008, K = 4096;

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
    {
        ops::managed_matmul_core_config_t cfg = {1, 16, 1, 16, 4, 0};
        mmm0->dyn_cast<op_traits::configurable_t>()->set_config(
                reflection::general_object_t::make(cfg));
    }
    auto relu0 = graph.make("relu", mmm0->get_outputs(), {}, {});
    // mmm1
    auto mmm1 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    {
        ops::managed_matmul_core_config_t cfg = {1, 16, 1, 16, 4, 0};
        mmm1->dyn_cast<op_traits::configurable_t>()->set_config(
                reflection::general_object_t::make(cfg));
    }
    auto relu1 = graph.make("relu", mmm1->get_outputs(), {}, {});
    // Although mmm0 and mmm1 are both inputs of add0, they should not be merged
    // due to nested parallel loop found
    auto add0 = graph.make(
            "add", {relu0->get_outputs()[0], relu1->get_outputs()[0]}, {}, {});
    auto out0 = graph.make_output(add0->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.opt_level_ = sc_opt_level::lv2;
    graph_driver(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    // Actually, mmm0 and mmm1 would still be merged during later parallel merge
    std::string expected_str
            = R"(graph(v0: f32[4, 4096], v1: f32[4096, 11008], v2: f32[4096, 11008]) -> [v3: f32[4, 11008]] {
  [v4: f32[4, 11008]] = outerloop_1X16X1X1X16_partition_managed_matmul_core_relu(v0, v2)
  [v3: f32[4, 11008]] = outerloop_1X16X1X1X16_partition_managed_matmul_core_relu_add(v0, v1, v4)
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
        return mgr.make<test_prefetchable_op>("tperf", ins, outs, attrs_);
    }
    ir_module_ptr get_func(context_ptr ctx) override { return nullptr; }
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override {}
};

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestPrefetchNone) {
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
    gemm->attrs_.set(mixed_partition_hint::first_prefetch_op, true);
    graph.make_output(gemm->get_outputs());

    std::vector<mixed_parti_t::ptr> par(graph.ops_.size());

    mixed_fuse_op_t tester {"test", par, nullptr, graph,
            {graph_tensor::make({128, 128}, sc_data_format_t::MK()),
                    graph_tensor::make({128, 128}, sc_data_format_t::MK())},
            {graph_tensor::make({128, 128}, sc_data_format_t::MK())}, {}};
    EXPECT_EQ(tester.query_prefetch(ctx, true, {}), std::vector<int>());
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, TestPrefetchSelected) {
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
    gemm->attrs_.set(mixed_partition_hint::first_prefetch_op, true);
    graph.make_output(gemm->get_outputs());

    std::vector<mixed_parti_t::ptr> par(graph.ops_.size());

    mixed_fuse_op_t tester {"test", par, nullptr, graph,
            {graph_tensor::make({128, 128}, sc_data_format_t::MK()),
                    graph_tensor::make({128, 128}, sc_data_format_t::MK())},
            {graph_tensor::make({128, 128}, sc_data_format_t::MK())}, {}};

    // prefetch op returns {0,1}, and only 1 is input. Input 1 of the op
    // maps to the input 0 of the graph
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

// loop finder
class loop_finder_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;
    void operator()(stmt_c v) { ir_viewer_t::dispatch(std::move(v)); }
    bool has_illegal_var() const { return illegal_loop_var_; }
    bool has_dummy_range() const { return dummy_loop_range_; }
    bool has_no_barrier_attr() const { return no_barrier_attr_; }
    void view(for_loop_c f) override {
        // check `var_` if var type
        if (!f->var_.isa<var>()) { illegal_loop_var_ = true; }
        // check loop range if dummy
        if (f->iter_begin_.isa<constant>() && f->iter_end_.isa<constant>()
                && get_expr_as_int(f->iter_begin_) == 0
                && get_expr_as_int(f->iter_end_) == 1) {
            dummy_loop_range_ = true;
        }
        if (f->attr_
                && f->attr_->get_or_else(
                        stmt_attr_key::no_post_barrier, false)) {
            no_barrier_attr_ = true;
        }
        ir_viewer_t::view(f);
    }

private:
    bool illegal_loop_var_ = false;
    bool dummy_loop_range_ = false;
    bool no_barrier_attr_ = false;
};

TEST(GCCore_CPU_graph_mixed_partition_cpp, ParallelMergeAndBarrier) {
    SET_THREADS_OR_SKIP(16);

    auto make_parallel_mmm_graph = [](bool barrier) {
        int M = 256, K1 = 2048, N = 1024;
        sc_graph_t graph;
        auto input0 = graph.make_input({graph_tensor::make(
                {M, K1}, sc_data_format_t(format_kinds::MK))});
        auto weight0 = graph.make_input({graph_tensor::make(
                {K1, N}, sc_data_format_t(format_kinds::KN))});
        auto weight1 = graph.make_input({graph_tensor::make(
                {K1, N}, sc_data_format_t(format_kinds::KN))});
        ops::managed_matmul_core_config_t cfg0 = {2, 8, 1, 1, 1, 0},
                                          cfg1 = {2, 4, 1, 1, 1, 0};
        // mmm0
        auto mmm0 = graph.make("managed_matmul_core",
                {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
        mmm0->dyn_cast<op_traits::configurable_t>()->set_config(
                reflection::general_object_t::make(barrier ? cfg1 : cfg0));
        // mmm1
        auto mmm1 = graph.make("managed_matmul_core",
                {input0->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
        mmm1->dyn_cast<op_traits::configurable_t>()->set_config(
                reflection::general_object_t::make(cfg1));
        graph.make_output({mmm0->get_outputs()[0]});
        graph.make_output({mmm1->get_outputs()[0]});
        return graph;
    };

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    bool barrier;
    {
        /* Case 0: need barrier */
        barrier = true;
        auto graph = make_parallel_mmm_graph(barrier);
        mixed_partition(graph, ctx);
        auto mixed_op = get_mixed_op_from_graph(graph);
        ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
        auto &inner_body
                = mixed_op->parti_list_[0]->get_outer_loops().back()->body_;
        // loop attr finder
        loop_finder_t la_finder;
        la_finder(inner_body);
        // `no_post_barrier attr` is not expected
        ASSERT_FALSE(la_finder.has_no_barrier_attr());
    }
    {
        /* Case 1: remove barrier */
        barrier = false;
        auto graph = make_parallel_mmm_graph(barrier);
        mixed_partition(graph, ctx);
        auto mixed_op = get_mixed_op_from_graph(graph);
        ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
        auto &body = mixed_op->parti_list_[0]->func_->body_;
        auto inner_loop = body.cast<stmts>()
                                  .map([](const stmts &v) {
                                      return v->seq_.at(0).as<for_loop>();
                                  })
                                  .map([](const for_loop &v) {
                                      return v->body_.as<stmts>()
                                              ->seq_.at(0)
                                              .as<for_loop>();
                                  })
                                  .get_or_else(for_loop());
        ASSERT_TRUE(inner_loop.defined());
        // `no_post_barrier` attr is expected
        ASSERT_TRUE(inner_loop->attr().get_or_else(
                stmt_attr_key::no_post_barrier, false));
    }
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, ParallelMergeNotAppendInputAnchor) {
    SET_THREADS_OR_SKIP(4);
    int M = 256, K = 252, N = 256;

    sc_graph_t graph;
    auto input0 = graph.make_input({graph_tensor::make(
            {M, K}, sc_data_format_t(format_kinds::MK), sc_data_type_t::u8())});
    auto weight0 = graph.make_input({graph_tensor::make(
            {K, N}, sc_data_format_t(format_kinds::KN), sc_data_type_t::s8())});
    auto weight1 = graph.make_input({graph_tensor::make(
            {K, N}, sc_data_format_t(format_kinds::KN), sc_data_type_t::s8())});

    ops::managed_matmul_core_config_t cfg = {4, 1, 1, 1, 1, 0};
    // mmm0
    auto mmm0 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {},
            {{op_attr_key::break_pre_fuse, true}});
    mmm0->dyn_cast<op_traits::configurable_t>()->set_config(
            reflection::general_object_t::make(cfg));
    //  mmm1
    auto mmm1 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight1->get_outputs()[0]}, {},
            {{op_attr_key::break_pre_fuse, true}});
    mmm1->dyn_cast<op_traits::configurable_t>()->set_config(
            reflection::general_object_t::make(cfg));
    graph.make_output({mmm0->get_outputs()[0]});
    graph.make_output({mmm1->get_outputs()[0]});

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    graph_driver(graph, ctx);
    auto mixed_op = get_mixed_op_from_graph(graph);
    // mmm0 and mmm1 both have input anchor, which is under outer loop when
    // parallel merged. As the result, it could not be straightfowardly append
    // to target partition fanchor list as inner loop anchor. Otherwise, fusion
    // partition will finally try to remove an unattached anchor in TIR and
    // throw assertion error.
    ASSERT_TRUE(mixed_op);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp,
        ReplaceBufferForTensorviewPaddingPattern) {
    SET_THREADS_OR_SKIP(4);

    // cached gt
    graph_tensor_ptr relu0_out;

    auto make_test_graph = [&relu0_out](bool shared_with_tv = false) {
        int BS = 1, H = 46, W = 46, C = 16;
        sc_graph_t graph;
        auto input0 = graph.make_input({graph_tensor::make(
                {BS, H, W, C}, sc_data_format_t(), sc_data_type_t::u8())});
        // relu
        auto relu0 = graph.make("relu", input0->get_outputs(), {}, {});
        // cache output of relu0
        relu0_out = relu0->get_outputs()[0];
        sc_op_ptr tv0 = nullptr;
        if (shared_with_tv) {
            // tensorview0
            tv0 = graph.make("tensor_view", {relu0->get_outputs()[0]}, {},
                    {{"shape", sc_dims {BS, H, W, C}}});
        }
        auto relu1 = graph.make("relu",
                tv0 ? tv0->get_outputs() : relu0->get_outputs(), {}, {});
        graph.make_output({relu1->get_outputs()[0]});
        // tensorview1
        auto tv1 = graph.make("tensor_view", {relu0->get_outputs()[0]}, {},
                {{"shape", sc_dims {BS, H, W, C}}});
        // padding
        auto padding0 = graph.make("padding", {tv1->get_outputs()[0]}, {},
                {{"pads_begin", sc_dims {1, 1}}, {"pads_end", sc_dims {1, 1}}});
        // relu
        auto relu2 = graph.make("relu", padding0->get_outputs(), {}, {});
        graph.make_output({relu2->get_outputs()[0]});
        return graph;
    };

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;

    // Case 1
    {
        auto graph = make_test_graph(false);
        mixed_partition(graph, ctx);
        // The output buffer of `relu0` should be replaced by padding op
        auto mixed_op = get_mixed_op_from_graph(graph);
        ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
        auto allocator = mixed_op->parti_list_[0]->buf_alloc_;
        auto relu_base_tsr = get_real_tensor(allocator.g2b_map_.get(relu0_out));
        ASSERT_TRUE(relu_base_tsr->name_ == "padding_5_outs_0");
    }

    // Case 2
    {
        auto graph = make_test_graph(true);
        mixed_partition(graph, ctx);
        // The output buffer of `relu0` should not be replaced by padding op
        // because it is shared with another tensorview op
        auto mixed_op = get_mixed_op_from_graph(graph);
        ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
        auto allocator = mixed_op->parti_list_[0]->buf_alloc_;
        auto relu_base_tsr = get_real_tensor(allocator.g2b_map_.get(relu0_out));
        ASSERT_TRUE(relu_base_tsr->name_ == "relu_1_outs_0");
    }
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, CommitPaddingToContentOfAnchor) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(4);

    int BS = 4, H = 46, W = 46, C = 16, K = 16;
    sc_graph_t graph;
    auto input0 = graph.make_input({graph_tensor::make(
            {BS, C, H, W}, sc_data_format_t::NCHW(), sc_data_type_t::s8())});
    auto weight0 = graph.make_input({graph_tensor::make(
            {K, C, 1, 1}, sc_data_format_t::KCRS(), sc_data_type_t::s8())});
    weight0->attrs_.set("constant", const_kind::local_const);
    auto weight1 = graph.make_input({graph_tensor::make(
            {K, C, 1, 1}, sc_data_format_t::KCRS(), sc_data_type_t::f32())});
    weight1->attrs_.set("constant", const_kind::local_const);

    // conv0
    auto conv0 = graph.make("conv_fwd_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {},
            {{"strides", sc_dims {1, 1}}, {"paddings", sc_dims {0, 0}}});
    // relu0
    auto relu0 = graph.make("relu", conv0->get_outputs(), {}, {});
    // padding0
    auto padding0 = graph.make("padding", {relu0->get_outputs()[0]}, {},
            {{"pads_begin", sc_dims {1, 1}}, {"pads_end", sc_dims {1, 1}}});
    // cast0
    auto cast0 = graph.make("cast", {padding0->get_outputs()[0]}, {},
            {{"dtype", datatypes::f32}});
    // conv1
    auto conv1 = graph.make("conv_fwd_core",
            {cast0->get_outputs()[0], weight1->get_outputs()[0]}, {},
            {{"strides", sc_dims {1, 1}}, {"paddings", sc_dims {0, 0}}});
    // relu1
    auto relu1 = graph.make("relu", padding0->get_outputs(), {},
            {{op_attr_key::break_pre_fuse, true}});
    // relu2
    auto relu2 = graph.make("relu", relu1->get_outputs(), {}, {});
    // relu3
    auto relu3 = graph.make("relu", conv1->get_outputs(), {}, {});
    // add0
    auto addd0 = graph.make(
            "add", {relu3->get_outputs()[0], relu2->get_outputs()[0]}, {}, {});
    graph.make_output(addd0->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;

    graph_driver_before_fusion(graph, ctx);
    mixed_partition(graph, ctx);

    auto mixed_op = get_mixed_op_from_graph(graph);
    ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
    ASSERT_TRUE(padding0);
    auto &op_anchor_map = mixed_op->parti_list_[0]->op_anchor_map_;
    auto iter = op_anchor_map.find(padding0.get());
    ASSERT_TRUE(iter != op_anchor_map.end());
    auto &content_number_map = iter->second->content_number_map_;
    // padding0 shoud be committed to content number map of committed anchor
    ASSERT_TRUE(content_number_map.find(padding0.get())
            != content_number_map.end());
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, CleanFusibleInnerLoop1) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(28);
    int BS = 28, H = 32, W = 32, C = 64;

    sc_graph_t graph;
    auto input0 = graph.make_input(
            {graph_tensor::make({BS, C, H, W}, sc_data_format_t::NCHW())});
    auto input1 = graph.make_input({graph_tensor::make({1, 1, H, 1})});
    // relu
    auto relu0 = graph.make("relu", input0->get_outputs(), {}, {});
    // add
    auto add0 = graph.make(
            "add", {relu0->get_outputs()[0], input1->get_outputs()[0]}, {}, {});
    // reorder
    auto reo0 = graph.make("reorder", add0->get_outputs(), {},
            {{"out_format", sc_data_format_t::NCHWc(32)}});
    graph.make_output({reo0->get_outputs()[0]});

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;
    mixed_partition(graph, ctx);

    auto mixed_op = get_mixed_op_from_graph(graph);
    ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
    auto func = mixed_op->parti_list_[0]->func_;
    ASSERT_TRUE(func && func->body_.isa<stmts>());
    auto body = func->body_.static_as<stmts>();
    ASSERT_TRUE(body->seq_.size() == 1 && body->seq_[0].isa<for_loop>());
    auto outer_loop = body->seq_[0].static_as<for_loop>();
    ASSERT_TRUE(outer_loop->body_.isa<stmts>());
    auto loop_body = outer_loop->body_.static_as<stmts>();
    ASSERT_TRUE(loop_body->seq_.size() == 3 && loop_body->seq_[2].isa<stmts>());
    auto second_anchor = loop_body->seq_[2].static_as<stmts>();
    ASSERT_TRUE(second_anchor->seq_.size() == 1
            && second_anchor->seq_[0].isa<for_loop>());
    auto reo_inner_loop = second_anchor->seq_[0].static_as<for_loop>();
    // The first inner loop of reorder op should have range of [0, 32] rather
    // than [0, 1], and own `merge_loop` attr
    ASSERT_TRUE(get_expr_as_int(reo_inner_loop->iter_end_) == 32
            && reo_inner_loop->attr_
            && reo_inner_loop->attr_->get_or_else(
                    stmt_attr_key::merge_loop, false));
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, CleanFusibleInnerLoop2) {
    SET_THREADS_OR_SKIP(1);
    int BS = 1, H = 8, W = 8, C = 64;

    sc_graph_t graph;
    auto input0 = graph.make_input({graph_tensor::make({BS, C, H, W})});
    // relu
    auto relu0 = graph.make("relu", input0->get_outputs(), {}, {});
    // reduce
    auto radd0 = graph.make("reduce", relu0->get_outputs(), {},
            {{"rd_axis", std::vector<int> {0, 2, 3}}, {"rd_op", 0},
                    {"keep_dims", false}});
    graph.make_output(radd0->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;
    mixed_partition(graph, ctx);

    auto mixed_op = get_mixed_op_from_graph(graph);
    ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
    auto &func = mixed_op->parti_list_[0]->func_;
    // loop var finder
    loop_finder_t lv_finder;
    lv_finder(func->body_);
    // All loop var should be `var` type. `for 0 in (0, 1, 1)` is not expected.
    EXPECT_FALSE(lv_finder.has_illegal_var());
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, CleanFusibleInnerLoop3) {
    SET_THREADS_OR_SKIP(28);
    int BS = 28, M = 32, N = 32;

    sc_graph_t graph;
    auto input0 = graph.make_input({graph_tensor::make(
            {BS, M, N}, sc_data_format_t(format_kinds::ABC), datatypes::u8)});
    auto input1 = graph.make_input({graph_tensor::make(
            {BS, M, N}, sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    auto input2 = graph.make_input({graph_tensor::make(
            {BS, M, N}, sc_data_format_t(format_kinds::ABC), datatypes::f32)});
    // select0
    auto select0 = graph.make("select",
            {input0->get_outputs()[0], input1->get_outputs()[0],
                    input2->get_outputs()[0]},
            {}, {});
    // relu0
    auto relu0 = graph.make("relu", select0->get_outputs(), {}, {});
    graph.make_output({relu0->get_outputs()[0]});

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;
    mixed_partition(graph, ctx);

    auto mixed_op = get_mixed_op_from_graph(graph);
    ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
    auto &func = mixed_op->parti_list_[0]->func_;
    // loop range finder
    loop_finder_t lr_finder;
    lr_finder(func->body_);
    // All loop range should not be dummp like (0, 1, 1)
    EXPECT_FALSE(lr_finder.has_dummy_range());
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, PoolingLoopReSchedule) {
    int num_threads = 4;
    SET_THREADS_OR_SKIP(num_threads);
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    auto dtype = sc_data_type_t::f32();
    auto lanes = vectorize_step(ctx, dtype.as_etype());
    int N = 1, H = 56, W = 56, C = num_threads * lanes;

    sc_graph_t graph;
    auto input0 = graph.make_input(
            {graph_tensor::make({N, H, W, C}, sc_data_format_t(), dtype)});
    // pooling
    auto pooling_out = graph.make("pooling_max", input0->get_outputs(), {},
            {{pooling_attr_key::strides, sc_dims {1, 1}},
                    {pooling_attr_key::pads_begin, sc_dims {0, 0}},
                    {pooling_attr_key::pads_end, sc_dims {0, 0}},
                    {pooling_attr_key::kernel, sc_dims {3, 3}},
                    {pooling_attr_key::data_format, data_format_options::NXC}});
    // relu
    auto relu0 = graph.make("relu", pooling_out->get_outputs(), {}, {});
    graph.make_output({relu0->get_outputs()[0]});

    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;
    layout_propagation(graph, ctx);
    mixed_partition(graph, ctx);

    auto mixed_op = get_mixed_op_from_graph(graph);
    ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
    auto parti = mixed_op->parti_list_[0];
    ASSERT_TRUE(parti->get_outer_loops().size() == 1);
    auto outer_loop = parti->get_outer_loops()[0].checked_as<for_loop>();
    // pooling op should reschedule its outer loop to seek more parallelism
    EXPECT_TRUE(outer_loop->kind_ == for_type::PARALLEL
            && get_expr_as_int(outer_loop->iter_end_) == num_threads);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, ComplexTensorViewInferSlice) {
    SET_THREADS_OR_SKIP(4);

    sc_graph_t graph;
    auto input0 = graph.make_input({graph_tensor::make(
            {224, 256}, sc_data_format_t(format_kinds::MK))});
    auto weight0 = graph.make_input({graph_tensor::make(
            {256, 256}, sc_data_format_t(format_kinds::KN))});

    // mmm0
    auto mmm0 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    ops::managed_matmul_core_config_t cfg = {4, 1, 1, 1, 1, 0};
    mmm0->dyn_cast<op_traits::configurable_t>()->set_config(
            reflection::general_object_t::make(cfg));
    auto relu0 = graph.make("relu", {mmm0->get_outputs()[0]}, {}, {});
    auto tv0 = graph.make("tensor_view", {relu0->get_outputs()[0]}, {},
            {{"shape", sc_dims {4, 56, 256}}});
    auto relu1 = graph.make("relu", {tv0->get_outputs()[0]}, {}, {});
    auto radd0 = graph.make("reduce", relu1->get_outputs(), {},
            {{"rd_axis", std::vector<int> {2}}, {"rd_op", 0}});
    auto relu2 = graph.make("relu", {radd0->get_outputs()[0]}, {}, {});
    graph.make_output({relu2->get_outputs()[0]});
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;

    graph_driver_before_fusion(graph, ctx);
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    // The reduce op could not find suitable anchor to commit, because of fusion
    // flow was breaked by tensorview op
    std::string expected_str
            = R"(graph(v0: f32[224, 256], v1: f32[256, 256]) -> [v2: f32[4, 56, 1]] {
  [v3: f32[4, 56, 256]] = outerloop_4X1X1X1X1_partition_managed_matmul_core_relu_tensor_view_relu(v0, v1)
  [v2: f32[4, 56, 1]] = outerloop_4X56_partition_reduce_compute_reduce_collect_relu(v3)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp,
        DoNotSplitReduceInNestedParallelTemplate) {
    SET_THREADS_OR_SKIP(8);

    sc_graph_t graph;
    auto input0 = graph.make_input({graph_tensor::make(
            {224, 256}, sc_data_format_t(format_kinds::MK))});
    auto weight0 = graph.make_input({graph_tensor::make(
            {256, 256}, sc_data_format_t(format_kinds::KN))});

    // mmm0
    auto mmm0 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    ops::managed_matmul_core_config_t cfg = {4, 2, 1, 1, 1, 0};
    mmm0->dyn_cast<op_traits::configurable_t>()->set_config(
            reflection::general_object_t::make(cfg));
    auto relu0 = graph.make("relu", {mmm0->get_outputs()[0]}, {}, {});
    auto radd0 = graph.make("reduce", relu0->get_outputs(), {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});
    auto relu1 = graph.make("relu", {radd0->get_outputs()[0]}, {}, {});
    graph.make_output({relu1->get_outputs()[0]});
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;

    graph_driver_before_fusion(graph, ctx);
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    // The reduce op could not be split
    std::string expected_str
            = R"(graph(v0: f32[224, 256], v1: f32[256, 256]) -> [v2: f32[224, 1]] {
  [v2: f32[224, 1]] = outerloop_4_partition_managed_matmul_core_relu_reduce_relu(v0, v1)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp,
        TransposeSemanticTensorViewBindAxis) {
    SET_THREADS_OR_SKIP(64);
    REQUIRE_AMX();

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;

    sc_graph_t mlp_graph;
    auto sigmoid_backprop_in0 = mlp_graph.make_input(
            {graph_tensor::make({128, 1}, sc_data_format_t())});
    auto sigmoid_backprop_in1 = mlp_graph.make_input(
            {graph_tensor::make({128, 1}, sc_data_format_t())});
    auto transpose_in0 = mlp_graph.make_input(
            {graph_tensor::make({256, 1}, sc_data_format_t())});
    auto relu_backprop_in0 = mlp_graph.make_input(
            {graph_tensor::make({128, 256}, sc_data_format_t())});
    auto transpose_in1 = mlp_graph.make_input(
            {graph_tensor::make({512, 256}, sc_data_format_t())});
    auto transpose_in2 = mlp_graph.make_input(
            {graph_tensor::make({1024, 512}, sc_data_format_t())});
    auto relu_backprop_in2 = mlp_graph.make_input(
            {graph_tensor::make({128, 512}, sc_data_format_t())});
    auto matmul_in1 = mlp_graph.make_input(
            {graph_tensor::make({1024, 1024}, sc_data_format_t())});
    auto matmul_in2 = mlp_graph.make_input(
            {graph_tensor::make({128, 1024}, sc_data_format_t())});

    auto sigmoid_backprop = mlp_graph.make("sigmoid_backprop",
            {sigmoid_backprop_in0->get_outputs()[0],
                    sigmoid_backprop_in1->get_outputs()[0]},
            {}, {{"use_dst", true}});
    auto static_transpose0
            = mlp_graph.make("transpose", {transpose_in0->get_outputs()[0]}, {},
                    {{"order", std::vector<int> {1, 0}}});
    auto matmul0 = mlp_graph.make("matmul",
            {sigmoid_backprop->get_outputs()[0],
                    static_transpose0->get_outputs()[0]},
            {}, {});
    auto static_transpose0_0
            = mlp_graph.make("transpose", {matmul0->get_outputs()[0]}, {},
                    {{"order", std::vector<int> {1, 0}}});
    auto matmul0_0 = mlp_graph.make("matmul",
            {static_transpose0_0->get_outputs()[0],
                    relu_backprop_in0->get_outputs()[0]},
            {}, {});
    auto relu_backprop1 = mlp_graph.make("relu_backprop",
            {relu_backprop_in0->get_outputs()[0], matmul0->get_outputs()[0]},
            {}, {{"use_dst", true}});
    auto static_transpose1
            = mlp_graph.make("transpose", {transpose_in1->get_outputs()[0]}, {},
                    {{"order", std::vector<int> {1, 0}}});
    auto static_transpose2
            = mlp_graph.make("transpose", {relu_backprop1->get_outputs()[0]},
                    {}, {{"order", std::vector<int> {1, 0}}});
    auto matmul1 = mlp_graph.make("matmul",
            {relu_backprop1->get_outputs()[0],
                    static_transpose1->get_outputs()[0]},
            {}, {});
    auto matmul2 = mlp_graph.make("matmul",
            {static_transpose2->get_outputs()[0],
                    relu_backprop_in2->get_outputs()[0]},
            {}, {});
    auto relu_backprop2 = mlp_graph.make("relu_backprop",
            {relu_backprop_in2->get_outputs()[0], matmul1->get_outputs()[0]},
            {}, {{"use_dst", true}});
    auto static_transpose3
            = mlp_graph.make("transpose", {transpose_in2->get_outputs()[0]}, {},
                    {{"order", std::vector<int> {1, 0}}});
    auto static_transpose4
            = mlp_graph.make("transpose", {relu_backprop2->get_outputs()[0]},
                    {}, {{"order", std::vector<int> {1, 0}}});
    auto matmul3 = mlp_graph.make("matmul",
            {relu_backprop2->get_outputs()[0],
                    static_transpose3->get_outputs()[0]},
            {}, {});
    auto matmul4 = mlp_graph.make("matmul",
            {static_transpose4->get_outputs()[0], matmul_in2->get_outputs()[0]},
            {}, {});
    auto matmul5 = mlp_graph.make("matmul",
            {matmul3->get_outputs()[0], matmul_in1->get_outputs()[0]}, {}, {});
    auto static_transpose5
            = mlp_graph.make("transpose", {matmul3->get_outputs()[0]}, {},
                    {{"order", std::vector<int> {1, 0}}});
    auto matmul6 = mlp_graph.make("matmul",
            {static_transpose5->get_outputs()[0], matmul_in2->get_outputs()[0]},
            {}, {});
    auto matmul7 = mlp_graph.make("matmul",
            {matmul5->get_outputs()[0], matmul_in2->get_outputs()[0]}, {}, {});

    mlp_graph.make_output({matmul0_0->get_outputs()[0]});
    mlp_graph.make_output({matmul2->get_outputs()[0]});
    mlp_graph.make_output({matmul4->get_outputs()[0]});
    mlp_graph.make_output({matmul6->get_outputs()[0]});
    mlp_graph.make_output({matmul7->get_outputs()[0]});

    graph_driver(mlp_graph, ctx);
    std::stringstream ss;
    print_graph(mlp_graph, ss, true);
    // The reduce op could not be split
    std::string expected_str_spr
            = R"(graph(v0: f32[128, 1], v1: f32[128, 1], v2: f32[256, 1], v3: f32[128, 256], v4: f32[512, 256], v5: f32[1024, 512], v6: f32[128, 512], v7: f32[1024, 1024], v8: f32[128, 1024]) -> [v9: f32[256, 256], v10: f32[256, 512], v11: f32[512, 1024], v12: f32[1024, 1024], v13: f32[128, 1024]] {
  [v14: f32[32, 32, 4, 16]] = outerloop_32X32X4_partition_reorder_select_one(v6)
  [v15: f32[1024, 512]] = tensor_view(v5)
  [v16: f32[512, 1024]] = reorder(v15)
  [v17: f32[512, 256]] = tensor_view(v4)
  [v18: f32[256, 512]] = reorder(v17)
  [v19: f32[32, 16, 4, 16]] = outerloop_32X16X4_partition_reorder_select_one(v3)
  [v20: f32[1, 256]] = tensor_view(v2)
  [v21: f32[128, 1]] = outerloop_128_partition_mul_sub_mul(v0, v1)
  [v22: f32[64, 8, 4, 16], v23: f32[64, 8, 4, 16], v24: f32[32, 32, 4, 16]] = outerloop_32_partition_managed_matmul_core_tensor_view_reorder_mul_tensor_view_reorder_managed_matmul_core_mul(v21, v20, v19, v18, v14)
  [v25: f32[64, 8, 16, 16], v13: f32[128, 1024]] = outerloop_8_partition_reorder_managed_matmul_core_tensor_view_reorder_managed_matmul_core_managed_matmul_core_reorder(v24, v16, v7, v8)
  [v26: f32[32, 32, 4, 16]] = tensor_view(v24)
  [v11: f32[512, 1024], v12: f32[1024, 1024]] = outerloop_32X2X1_partition_reorder_managed_matmul_core_reorder_managed_matmul_core_reorder(v26, v8, v25)
  [v10: f32[256, 512]] = outerloop_64X1X1X1X1_partition_managed_matmul_core_reorder(v23, v6)
  [v9: f32[256, 256]] = outerloop_64X1X1X1X1_partition_managed_matmul_core_reorder(v22, v3)
}
)";
    EXPECT_EQ(ss.str(), expected_str_spr);
}

TEST(GCCore_CPU_graph_mixed_partition_cpp, InferSliceForBMMWithBroadcast) {
    SET_THREADS_OR_SKIP(8);

    sc_graph_t graph;
    auto input0 = graph.make_input({graph_tensor::make(
            {8, 16, 64, 128}, sc_data_format_t(), sc_data_type_t::u8())});
    // explicit broadcast semantic on weight side
    auto weight0 = graph.make_input({graph_tensor::make(
            {8, 1, 128, 64}, sc_data_format_t(), sc_data_type_t::s8())});

    any_map_t attrs({{"transpose_a", false}, {"transpose_b", false},
            {"output2d", false}, {"use_mmm", false}});
    // bmm
    auto bmm = graph.make("matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, attrs);
    ops::matmul_core_config_t cfg = {32, 32, 32};
    bmm->dyn_cast<ops::matmul_core_op_t>()->set_config(
            reflection::general_object_t::make(cfg));
    graph.make_output({bmm->get_outputs()[0]});
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.use_cost_model_ = true;

    graph_driver_before_fusion(graph, ctx);
    mixed_partition(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    // The matmul op could not be fused into reorder op along weight side
    std::string expected_str
            = R"(graph(v0: u8[8, 16, 64, 128], v1: s8[8, 1, 128, 64]) -> [v2: s32[8, 16, 64, 64]] {
  [v3: s8[8, 1, 2, 4, 8, 32, 4]] = reorder(v1)
  [v2: s32[8, 16, 64, 64]] = matmul_core(v0, v3)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}
