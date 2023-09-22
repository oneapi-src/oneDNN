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
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/unary_elemwise.hpp>

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_graph_inplace_transform_cpp, TestInputOutputInplacement1) {
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            {2, 64, 128}, sc_data_format_t(format_kinds::ABC))});
    auto out = graph.make_output(inp->get_outputs());
    inplace_transform(graph);
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: f32[2, 64, 128]) -> [v1: f32[2, 64, 128]] {
  [v1: f32[2, 64, 128]] = reorder(v0)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

TEST(GCCore_CPU_graph_inplace_transform_cpp, TestInputOutputInplacement2) {
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            {2, 64, 128}, sc_data_format_t(format_kinds::ABC))});
    auto tv = graph.make("tensor_view", inp->get_outputs(), {},
            {{"shape", sc_dims {2, 64, 32, 4}}});
    auto out = graph.make_output(tv->get_outputs());
    inplace_transform(graph);
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: f32[2, 64, 128]) -> [v1: f32[2, 64, 32, 4]] {
  [v2: f32[2, 64, 32, 4]] = tensor_view(v0)
  [v1: f32[2, 64, 32, 4]] = reorder(v2)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

TEST(GCCore_CPU_graph_inplace_transform_cpp, TestInputOutputInplacement3) {
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            {2, 64, 128}, sc_data_format_t(format_kinds::ABC))});
    auto tv1 = graph.make("tensor_view", inp->get_outputs(), {},
            {{"shape", sc_dims {2, 64, 32, 4}}});
    auto tv2 = graph.make("tensor_view", inp->get_outputs(), {},
            {{"shape", sc_dims {4, 16, 8, 32}}});
    auto tv3 = graph.make("tensor_view", tv1->get_outputs(), {},
            {{"shape", sc_dims {128, 4, 32}}});
    auto tv4 = graph.make("tensor_view", tv1->get_outputs(), {},
            {{"shape", sc_dims {128, 128}}});
    auto tv5 = graph.make("tensor_view", tv2->get_outputs(), {},
            {{"shape", sc_dims {256, 4, 16}}});
    auto relu = graph.make("relu", tv2->get_outputs(), {}, {});
    auto tv6 = graph.make("tensor_view", relu->get_outputs(), {},
            {{"shape", sc_dims {512, 32}}});
    auto out = graph.make_output({tv3->get_outputs()[0], tv4->get_outputs()[0],
            tv2->get_outputs()[0], tv5->get_outputs()[0],
            tv6->get_outputs()[0]});
    inplace_transform(graph);
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: f32[2, 64, 128]) -> [v1: f32[128, 4, 32], v2: f32[128, 128], v3: f32[4, 16, 8, 32], v4: f32[256, 4, 16], v5: f32[512, 32]] {
  [v6: f32[4, 16, 8, 32]] = tensor_view(v0)
  [v3: f32[4, 16, 8, 32]] = reorder(v6)
  [v7: f32[4, 16, 8, 32]] = relu(v6)
  [v5: f32[512, 32]] = tensor_view(v7)
  [v8: f32[256, 4, 16]] = tensor_view(v6)
  [v4: f32[256, 4, 16]] = reorder(v8)
  [v9: f32[2, 64, 32, 4]] = tensor_view(v0)
  [v10: f32[128, 128]] = tensor_view(v9)
  [v2: f32[128, 128]] = reorder(v10)
  [v11: f32[128, 4, 32]] = tensor_view(v9)
  [v1: f32[128, 4, 32]] = reorder(v11)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

TEST(GCCore_CPU_graph_inplace_transform_cpp, TestMultiOutputInplacement1) {
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            {2, 64, 128}, sc_data_format_t(format_kinds::ABC))});
    auto relu = graph.make("relu", inp->get_outputs(), {}, {});
    auto tv1 = graph.make("tensor_view", relu->get_outputs(), {},
            {{"shape", sc_dims {512, 32}}});
    auto out = graph.make_output(
            {relu->get_outputs()[0], tv1->get_outputs()[0]});
    inplace_transform(graph);
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: f32[2, 64, 128]) -> [v1: f32[2, 64, 128], v2: f32[512, 32]] {
  [v1: f32[2, 64, 128]] = relu(v0)
  [v3: f32[512, 32]] = tensor_view(v1)
  [v2: f32[512, 32]] = reorder(v3)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

TEST(GCCore_CPU_graph_inplace_transform_cpp, TestRedudantCopy1) {
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            {2, 64, 128}, sc_data_format_t(format_kinds::ABC))});
    auto relu = graph.make("relu", inp->get_outputs(), {}, {});
    auto cp_reo1 = graph.make("reorder", relu->get_outputs(), {},
            {{"out_format", sc_data_format_t(format_kinds::ABC)},
                    {"actually_copy", true}});
    auto out = graph.make_output({cp_reo1->get_outputs()[0]});
    inplace_transform(graph);
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: f32[2, 64, 128]) -> [v1: f32[2, 64, 128]] {
  [v1: f32[2, 64, 128]] = relu(v0)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

TEST(GCCore_CPU_graph_inplace_transform_cpp, TestRedudantCopy2) {
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            {2, 64, 128}, sc_data_format_t(format_kinds::ABC))});
    auto relu1 = graph.make("relu", inp->get_outputs(), {}, {});
    auto relu2 = graph.make("relu", inp->get_outputs(), {}, {});
    auto cp_reo1 = graph.make("reorder", relu1->get_outputs(), {},
            {{"out_format", sc_data_format_t(format_kinds::ABC)},
                    {"actually_copy", true}});
    auto cp_reo2 = graph.make("reorder", relu1->get_outputs(), {},
            {{"out_format", sc_data_format_t(format_kinds::ABC)},
                    {"actually_copy", true}});
    auto relu3 = graph.make("relu", cp_reo2->get_outputs(), {}, {});
    auto cp_reo3 = graph.make("reorder", relu2->get_outputs(), {},
            {{"out_format", sc_data_format_t(format_kinds::ABC)},
                    {"actually_copy", true}});
    auto cp_reo4 = graph.make("reorder", relu2->get_outputs(), {},
            {{"out_format", sc_data_format_t(format_kinds::ABC)},
                    {"actually_copy", true}});
    auto out = graph.make_output(
            {cp_reo1->get_outputs()[0], relu3->get_outputs()[0],
                    cp_reo3->get_outputs()[0], cp_reo4->get_outputs()[0]});
    inplace_transform(graph);
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: f32[2, 64, 128]) -> [v1: f32[2, 64, 128], v2: f32[2, 64, 128], v3: f32[2, 64, 128], v4: f32[2, 64, 128]] {
  [v3: f32[2, 64, 128]] = relu(v0)
  [v4: f32[2, 64, 128]] = reorder(v3)
  [v1: f32[2, 64, 128]] = relu(v0)
  [v2: f32[2, 64, 128]] = relu(v1)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

TEST(GCCore_CPU_graph_inplace_transform_cpp, TestRedudantCopy3) {
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            {2, 64, 128}, sc_data_format_t(format_kinds::ABC))});
    auto tv1 = graph.make("tensor_view", inp->get_outputs(), {},
            {{"shape", sc_dims {64, 4, 64}}});
    auto cp_reo1 = graph.make("reorder", tv1->get_outputs(), {},
            {{"out_format", sc_data_format_t(format_kinds::ABC)},
                    {"actually_copy", true}});
    auto tv2 = graph.make("tensor_view", cp_reo1->get_outputs(), {},
            {{"shape", sc_dims {2, 64, 128}}});
    auto tv3 = graph.make("tensor_view", inp->get_outputs(), {},
            {{"shape", sc_dims {64, 4, 64}}});
    auto cp_reo2 = graph.make("reorder", tv3->get_outputs(), {},
            {{"out_format", sc_data_format_t(format_kinds::ABC)},
                    {"actually_copy", true}});
    auto tv4 = graph.make("tensor_view", cp_reo2->get_outputs(), {},
            {{"shape", sc_dims {2, 64, 128}}});
    auto relu1 = graph.make("relu", tv4->get_outputs(), {}, {});
    auto relu2 = graph.make("relu", inp->get_outputs(), {}, {});
    auto tv5 = graph.make("tensor_view", relu2->get_outputs(), {},
            {{"shape", sc_dims {64, 4, 64}}});
    auto cp_reo3 = graph.make("reorder", tv5->get_outputs(), {},
            {{"out_format", sc_data_format_t(format_kinds::ABC)},
                    {"actually_copy", true}});
    auto tv6 = graph.make("tensor_view", cp_reo3->get_outputs(), {},
            {{"shape", sc_dims {2, 64, 128}}});
    auto out = graph.make_output({tv2->get_outputs()[0],
            relu1->get_outputs()[0], tv6->get_outputs()[0]});
    inplace_transform(graph);
    EXPECT_EQ(check_graph_connection(graph), true);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: f32[2, 64, 128]) -> [v1: f32[2, 64, 128], v2: f32[2, 64, 128], v3: f32[2, 64, 128]] {
  [v4: f32[2, 64, 128]] = relu(v0)
  [v5: f32[64, 4, 64]] = tensor_view(v4)
  [v3: f32[2, 64, 128]] = tensor_view(v5)
  [v6: f32[64, 4, 64]] = tensor_view(v0)
  [v7: f32[2, 64, 128]] = tensor_view(v6)
  [v2: f32[2, 64, 128]] = relu(v7)
  [v8: f32[64, 4, 64]] = tensor_view(v0)
  [v9: f32[64, 4, 64]] = reorder(v8)
  [v1: f32[2, 64, 128]] = tensor_view(v9)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

TEST(GCCore_CPU_graph_inplace_transform_cpp, TestBatchNormNoCopy) {
    auto graph = sc_graph_t();
    sc_dims input_dims {64, 32, 32, 16};
    int channel_axis = 3;
    auto ins0 = graph.make_input({graph_tensor::make(input_dims)});
    auto ins1 = graph.make_input(
            {graph_tensor::make({input_dims[channel_axis]})});
    auto ins2 = graph.make_input(
            {graph_tensor::make({input_dims[channel_axis]})});
    auto ins3 = graph.make_input(
            {graph_tensor::make({input_dims[channel_axis]})});
    auto ins4 = graph.make_input(
            {graph_tensor::make({input_dims[channel_axis]})});
    std::unordered_map<std::string, any_t> attrs = {{"epsilon", 1e-5f},
            {"momentum", 1.f}, {"data_format", std::string("NXC")}};
    auto relu0 = graph.make("relu", ins0->get_outputs(), {}, {});
    auto relu1 = graph.make("relu", ins1->get_outputs(), {}, {});
    auto relu2 = graph.make("relu", ins2->get_outputs(), {}, {});
    auto relu3 = graph.make("relu", ins3->get_outputs(), {}, {});
    auto relu4 = graph.make("relu", ins4->get_outputs(), {}, {});
    auto bn = graph.make("batchnorm_forward_training",
            {relu0->get_outputs()[0], relu1->get_outputs()[0],
                    relu2->get_outputs()[0], relu3->get_outputs()[0],
                    relu4->get_outputs()[0]},
            {}, any_map_t(attrs));
    auto out = graph.make_output(bn->get_outputs());
    graph_inline(graph);
    inplace_transform(graph);
    EXPECT_EQ(check_graph_connection(graph), true);
    int reorder_count = 0;
    for (auto &op : graph.ops_) {
        if (op->isa<reorder_op_t>()) { reorder_count++; }
    }
    EXPECT_EQ(reorder_count, 0);
}

TEST(GCCore_CPU_graph_inplace_transform_cpp, TestBatchNormCopy) {
    auto graph = sc_graph_t();
    sc_dims input_dims {64, 32, 32, 16};
    int channel_axis = 3;
    auto ins0 = graph.make_input({graph_tensor::make(input_dims)});
    auto ins1 = graph.make_input(
            {graph_tensor::make({input_dims[channel_axis]})});
    auto ins2 = graph.make_input(
            {graph_tensor::make({input_dims[channel_axis]})});
    auto ins3 = graph.make_input(
            {graph_tensor::make({input_dims[channel_axis]})});
    auto ins4 = graph.make_input(
            {graph_tensor::make({input_dims[channel_axis]})});
    std::unordered_map<std::string, any_t> attrs = {{"epsilon", 1e-5f},
            {"momentum", 1.f}, {"data_format", std::string("NXC")}};
    auto bn = graph.make("batchnorm_forward_training",
            {ins0->get_outputs()[0], ins1->get_outputs()[0],
                    ins2->get_outputs()[0], ins3->get_outputs()[0],
                    ins4->get_outputs()[0]},
            {}, any_map_t(attrs));
    auto out = graph.make_output(bn->get_outputs());
    graph_inline(graph);
    inplace_transform(graph);
    EXPECT_EQ(check_graph_connection(graph), true);
    int reorder_count = 0;
    for (auto &op : graph.ops_) {
        if (op->isa<reorder_op_t>()) { reorder_count++; }
    }
    EXPECT_EQ(reorder_count, 2);
}
