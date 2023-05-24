/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/unary_elemwise.hpp>

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_graph_simplify_cpp, TestSameOpElimination) {
    auto graph = sc_graph_t();
    auto data = graph.make_input({graph_tensor::make(
            {2, 64, 128}, sc_data_format_t(format_kinds::ABC))});
    auto weight0 = graph.make_input({graph_tensor::make(
            {2, 128, 256}, sc_data_format_t(format_kinds::ABC))});
    auto weight1 = graph.make_input({graph_tensor::make(
            {2, 128, 512}, sc_data_format_t(format_kinds::ABC))});
    auto weight2 = graph.make_input({graph_tensor::make(
            {2, 128, 1024}, sc_data_format_t(format_kinds::ABC))});
    auto gemm0 = graph.make("matmul_core",
            {data->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    auto gemm1 = graph.make("matmul_core",
            {data->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    auto gemm2 = graph.make("matmul_core",
            {data->get_outputs()[0], weight2->get_outputs()[0]}, {}, {});
    auto relu0 = graph.make("relu", data->get_outputs(), {}, {});
    auto relu1 = graph.make("relu", data->get_outputs(), {}, {});
    auto gemm3 = graph.make("matmul_core",
            {relu0->get_outputs()[0], relu1->get_outputs()[0]}, {}, {});
    auto output0 = graph.make_output(gemm3->get_outputs());
    auto output1 = graph.make_output({gemm0->get_outputs()[0],
            gemm1->get_outputs()[0], gemm2->get_outputs()[0]});
    graph.attrs_[sc_graph_t::attr_key_t::is_output_plain] = true;
    permute_propagation(graph);
    layout_propagation(graph);
    EXPECT_EQ(graph.ops_.size(), 12UL);
    graph_simplify(graph);
    EXPECT_EQ(check_graph_connection(graph), true);
    EXPECT_EQ(graph.ops_.size(), 11UL);

    int reorder_count = 0, relu_count = 0;
    for (auto &op : graph.ops_) {
        if (op->isa<reorder_op_t>()) {
            reorder_count++;
            if (op->get_inputs()[0]->producer_owner_->isa<input_op>()
                    && op->get_outputs()[0]->uses_[0].first == 0) {
                EXPECT_EQ(op->get_outputs()[0]->uses_.size(), 3UL);
            }
        } else if (op->isa<relu_op_t>()) {
            relu_count++;
            EXPECT_EQ(op->get_outputs()[0]->uses_.size(), 2UL);
        }
    }
    EXPECT_EQ(reorder_count, 0);
    EXPECT_EQ(relu_count, 1);
}

TEST(GCCore_CPU_graph_simplify_cpp, TestTensorViewElimination) {
    sc_dims in_plain_dims {128, 64, 256};
    sc_dims inter_plain_dims0 {128, 64, 32, 8};
    sc_dims inter_plain_dims1 {32, 128, 64, 8};
    sc_dims out_plain_dims0 {1024, 2048};
    sc_dims out_plain_dims1 {512, 4096};

    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(in_plain_dims)});
    auto tv0 = graph.make("tensor_view", inp->get_outputs(),
            {graph_tensor::make(inter_plain_dims0)},
            {{"shape", inter_plain_dims0}});
    auto tv1 = graph.make("tensor_view", tv0->get_outputs(),
            {graph_tensor::make(
                    out_plain_dims0, sc_data_format_t(format_kinds::BA))},
            {{"shape", out_plain_dims0}});
    auto tv2 = graph.make("tensor_view", inp->get_outputs(),
            {graph_tensor::make(inter_plain_dims1)},
            {{"shape", inter_plain_dims1}});
    auto tv3 = graph.make("tensor_view", tv2->get_outputs(),
            {graph_tensor::make(
                    out_plain_dims1, sc_data_format_t(format_kinds::BA))},
            {{"shape", out_plain_dims1}});
    auto relu0 = graph.make("relu", tv3->get_outputs(), {}, {});
    auto relu1 = graph.make("relu", tv3->get_outputs(), {}, {});

    auto output0 = graph.make_output(tv1->get_outputs());
    auto output1 = graph.make_output(relu0->get_outputs());
    auto output2 = graph.make_output(relu1->get_outputs());
    graph_simplify(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    auto &ops = graph.ops_;
    EXPECT_EQ(ops.size(), 8UL);
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), tv0) != ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), tv1) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), tv2) != ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), tv3) == ops.end());

    EXPECT_EQ(tv0->get_inputs()[0]->details_.get_plain_dims(), in_plain_dims);
    EXPECT_EQ(tv0->get_inputs()[0]->details_.get_format(), sc_data_format_t());
    EXPECT_EQ(
            tv0->get_outputs()[0]->details_.get_plain_dims(), out_plain_dims0);
    EXPECT_EQ(tv0->get_outputs()[0]->details_.get_format(),
            sc_data_format_t(format_kinds::BA));
    EXPECT_EQ(
            tv2->get_outputs()[0]->details_.get_plain_dims(), out_plain_dims1);
    EXPECT_EQ(tv2->get_outputs()[0]->details_.get_format(),
            sc_data_format_t(format_kinds::BA));
}

TEST(GCCore_CPU_graph_simplify_cpp, TestBinaryOpElimination) {
    sc_dims in_plain_dims {128, 64, 256};

    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(in_plain_dims)});
    auto zero = graph.make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {0.f}),
            datatypes::f32, sc_dims {1});
    auto one = graph.make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {1.f}),
            datatypes::f32, sc_dims {1});
    auto two = graph.make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {2.f}),
            datatypes::f32, sc_dims {1});
    auto add1 = graph.make(
            "add", {inp->get_outputs()[0], zero->get_outputs()[0]}, {}, {});
    auto add2 = graph.make(
            "add", {two->get_outputs()[0], add1->get_outputs()[0]}, {}, {});
    auto mul1 = graph.make(
            "mul", {add1->get_outputs()[0], add2->get_outputs()[0]}, {}, {});
    auto div1 = graph.make(
            "div", {mul1->get_outputs()[0], one->get_outputs()[0]}, {}, {});

    auto output0 = graph.make_output({div1->get_outputs()[0]});

    graph_simplify(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    auto &ops = graph.ops_;
    EXPECT_EQ(ops.size(), 5UL);
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), zero) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), one) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), add1) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), div1) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), two) != ops.end());
}

TEST(GCCore_CPU_graph_simplify_cpp, TestGraphConstantFoldingF32) {
    sc_dims in_plain_dims {128, 64, 256};
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(in_plain_dims)});
    auto const_inp = graph.make_input({graph_tensor::make(in_plain_dims)},
            {{"constant", const_kind::local_const}, {"all_positive", true}});
    float v = 2.f;
    auto const_op = graph.make("constant", {}, {},
            {{"all_positive", true}, {"plain_dims", sc_dims {1}},
                    {"format", sc_data_format_t()}, {"dtype", datatypes::f32},
                    {"values",
                            std::make_shared<static_data_t>(
                                    (void *)&v, sizeof(float))}});
    auto &const_inp_tsr = const_inp->get_outputs()[0];
    auto &const_op_tsr = const_op->get_outputs()[0];
    auto sub0
            = graph.make("sub", {inp->get_outputs()[0], const_op_tsr}, {}, {});
    auto add0 = graph.make(
            "add", {const_inp_tsr, sub0->get_outputs()[0]}, {}, {});
    auto mul0 = graph.make(
            "mul", {add0->get_outputs()[0], const_inp_tsr}, {}, {});
    auto div0
            = graph.make("div", {mul0->get_outputs()[0], const_op_tsr}, {}, {});
    auto sub1 = graph.make(
            "sub", {div0->get_outputs()[0], const_inp_tsr}, {}, {});
    auto relu = graph.make("relu", {sub1->get_outputs()[0]}, {}, {});
    auto mul1
            = graph.make("mul", {relu->get_outputs()[0], const_op_tsr}, {}, {});
    auto div1 = graph.make(
            "div", {mul1->get_outputs()[0], const_inp_tsr}, {}, {});

    auto output0 = graph.make_output(div1->get_outputs());

    graph_simplify(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: f32[128, 64, 256], v1: f32[128, 64, 256]) -> [v2: f32[128, 64, 256]] {
  [v3: f32[1]] = constant([1])
  [v4: f32[128, 64, 256]] = mul(v1, v3)
  [v5: f32[128, 64, 256]] = div(v4, v1)
  [v6: f32[128, 64, 256]] = div(v1, v3)
  [v7: f32[128, 64, 256]] = mul(v6, v3)
  [v8: f32[128, 64, 256]] = div(v7, v1)
  [v9: f32[128, 64, 256]] = sub(v3, v1)
  [v10: f32[128, 64, 256]] = sub(v0, v9)
  [v11: f32[128, 64, 256]] = mul(v10, v8)
  [v12: f32[128, 64, 256]] = sub(v11, v5)
  [v2: f32[128, 64, 256]] = relu(v12)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

TEST(GCCore_CPU_graph_simplify_cpp, TestGraphConstantFoldingS32) {
    sc_dims in_plain_dims {128, 64, 256};
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            in_plain_dims, sc_data_format_t(), datatypes::s32)});
    auto const_inp = graph.make_input(
            {graph_tensor::make(
                    in_plain_dims, sc_data_format_t(), datatypes::s32)},
            {{"constant", const_kind::local_const}, {"all_positive", true}});
    int v = 2;
    auto const_op = graph.make("constant", {}, {},
            {{"all_positive", true}, {"plain_dims", sc_dims {1}},
                    {"format", sc_data_format_t()}, {"dtype", datatypes::s32},
                    {"values",
                            std::make_shared<static_data_t>(
                                    (void *)&v, sizeof(int))}});
    auto &const_inp_tsr = const_inp->get_outputs()[0];
    auto &const_op_tsr = const_op->get_outputs()[0];
    auto sub0
            = graph.make("sub", {inp->get_outputs()[0], const_op_tsr}, {}, {});
    auto add0 = graph.make(
            "add", {const_inp_tsr, sub0->get_outputs()[0]}, {}, {});
    auto mul0 = graph.make(
            "mul", {add0->get_outputs()[0], const_inp_tsr}, {}, {});
    auto div0
            = graph.make("div", {mul0->get_outputs()[0], const_op_tsr}, {}, {});
    auto sub1 = graph.make(
            "sub", {div0->get_outputs()[0], const_inp_tsr}, {}, {});
    auto relu = graph.make("relu", {sub1->get_outputs()[0]}, {}, {});
    auto mul1
            = graph.make("mul", {relu->get_outputs()[0], const_op_tsr}, {}, {});
    auto div1 = graph.make(
            "div", {mul1->get_outputs()[0], const_inp_tsr}, {}, {});

    auto output0 = graph.make_output(div1->get_outputs());

    graph_simplify(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: s32[128, 64, 256], v1: s32[128, 64, 256]) -> [v2: s32[128, 64, 256]] {
  [v3: s32[1]] = constant([1])
  [v4: s32[128, 64, 256]] = sub(v3, v1)
  [v5: s32[128, 64, 256]] = sub(v0, v4)
  [v6: s32[128, 64, 256]] = mul(v5, v1)
  [v7: s32[128, 64, 256]] = div(v6, v3)
  [v8: s32[128, 64, 256]] = sub(v7, v1)
  [v9: s32[128, 64, 256]] = mul(v8, v3)
  [v10: s32[128, 64, 256]] = div(v9, v1)
  [v2: s32[128, 64, 256]] = relu(v10)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

TEST(GCCore_CPU_graph_simplify_cpp, TestGraphConstantFoldingRecursiveStop) {
    sc_dims in_plain_dims {128, 64, 256};
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            in_plain_dims, sc_data_format_t(), datatypes::s32)});
    auto const_inp = graph.make_input(
            {graph_tensor::make(
                    in_plain_dims, sc_data_format_t(), datatypes::s32)},
            {{"constant", const_kind::local_const}, {"all_positive", true}});
    int v = 2;
    auto const_op = graph.make("constant", {}, {},
            {{"all_positive", true}, {"plain_dims", sc_dims {1}},
                    {"format", sc_data_format_t()}, {"dtype", datatypes::s32},
                    {"values",
                            std::make_shared<static_data_t>(
                                    (void *)&v, sizeof(int))}});
    auto &const_inp_tsr = const_inp->get_outputs()[0];
    auto &const_op_tsr = const_op->get_outputs()[0];
    auto mul0
            = graph.make("mul", {inp->get_outputs()[0], const_op_tsr}, {}, {});
    auto mul1 = graph.make(
            "mul", {const_inp_tsr, mul0->get_outputs()[0]}, {}, {});

    auto output0 = graph.make_output(mul1->get_outputs());

    graph_simplify(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: s32[128, 64, 256], v1: s32[128, 64, 256]) -> [v2: s32[128, 64, 256]] {
  [v3: s32[1]] = constant([1])
  [v4: s32[128, 64, 256]] = mul(v3, v1)
  [v2: s32[128, 64, 256]] = mul(v0, v4)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

TEST(GCCore_CPU_graph_simplify_cpp, TestGraphConstantFoldingMultiOutput) {
    sc_dims in_plain_dims {128, 64, 256};
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            in_plain_dims, sc_data_format_t(), datatypes::s32)});
    auto const_inp = graph.make_input(
            {graph_tensor::make(
                    in_plain_dims, sc_data_format_t(), datatypes::s32)},
            {{"constant", const_kind::local_const}, {"all_positive", true}});
    int v = 2;
    auto const_op = graph.make("constant", {}, {},
            {{"all_positive", true}, {"plain_dims", sc_dims {1}},
                    {"format", sc_data_format_t()}, {"dtype", datatypes::s32},
                    {"values",
                            std::make_shared<static_data_t>(
                                    (void *)&v, sizeof(int))}});
    auto &const_inp_tsr = const_inp->get_outputs()[0];
    auto &const_op_tsr = const_op->get_outputs()[0];
    auto mul0
            = graph.make("mul", {inp->get_outputs()[0], const_op_tsr}, {}, {});
    auto add0 = graph.make(
            "add", {const_inp_tsr, mul0->get_outputs()[0]}, {}, {});
    auto mul1 = graph.make(
            "mul", {add0->get_outputs()[0], const_inp_tsr}, {}, {});
    auto relu0 = graph.make("relu", add0->get_outputs(), {}, {});
    auto output0 = graph.make_output(
            {relu0->get_outputs()[0], mul1->get_outputs()[0]});

    graph_simplify(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: s32[128, 64, 256], v1: s32[128, 64, 256]) -> [v2: s32[128, 64, 256], v3: s32[128, 64, 256]] {
  [v4: s32[1]] = constant([1])
  [v5: s32[128, 64, 256]] = mul(v0, v4)
  [v6: s32[128, 64, 256]] = add(v5, v1)
  [v2: s32[128, 64, 256]] = relu(v6)
  [v3: s32[128, 64, 256]] = mul(v6, v1)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

TEST(GCCore_CPU_graph_simplify_cpp, TestGraphPushReluBackNegative) {
    sc_dims in_plain_dims {128, 64, 256};
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            in_plain_dims, sc_data_format_t(), datatypes::s32)});
    auto const_inp = graph.make_input(
            {graph_tensor::make(
                    in_plain_dims, sc_data_format_t(), datatypes::s32)},
            {{"constant", const_kind::local_const}, {"all_positive", false}});
    auto &const_inp_tsr = const_inp->get_outputs()[0];
    auto relu = graph.make("relu", {inp->get_outputs()[0]}, {}, {});
    auto mul1 = graph.make(
            "mul", {relu->get_outputs()[0], const_inp_tsr}, {}, {});
    auto div1 = graph.make(
            "div", {mul1->get_outputs()[0], const_inp_tsr}, {}, {});

    auto output0 = graph.make_output(div1->get_outputs());

    graph_simplify(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: s32[128, 64, 256], v1: s32[128, 64, 256]) -> [v2: s32[128, 64, 256]] {
  [v3: s32[128, 64, 256]] = relu(v0)
  [v4: s32[128, 64, 256]] = mul(v3, v1)
  [v2: s32[128, 64, 256]] = div(v4, v1)
}
)";
    EXPECT_EQ(ss.str(), expected);
}
