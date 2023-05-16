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

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_graph_tensor_view_transform, TestReorderToTransform0) {
    auto graph = sc_graph_t();
    auto inp = graph.make_input(
            {graph_tensor::make({64}, sc_data_format_t(format_kinds::A))});
    auto reorder = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(
                    {64}, sc_data_format_t(format_kinds::Aa, {16, 0, 0, 0}))},
            {});
    auto output = graph.make_output(reorder->get_outputs());
    tensor_view_transform(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    EXPECT_EQ(graph.ops_.size(), 3UL);
    for (auto &op : graph.ops_) {
        if (!op->isa<input_op>() && !op->isa<output_op>()) {
            EXPECT_EQ(op->get_outputs()[0]->details_.get_format(),
                    sc_data_format_t(format_kinds::Aa, {16, 0, 0, 0}));
        }
    }
}

constexpr sc_data_format_kind_t fmtMmKk = sc_data_format_kind_t {0, 0, 1, 1};
TEST(GCCore_CPU_graph_tensor_view_transform, TestReorderToTransform1) {
    auto graph = sc_graph_t();
    auto inp = graph.make_input(
            {graph_tensor::make({64, 128}, sc_data_format_t::MK())});
    auto reorder0 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make({64, 128}, sc_data_format_t::MKmk(16, 16))},
            {});
    auto reorder1 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(
                    {64, 128}, sc_data_format_t(fmtMmKk, {16, 16, 0, 0}))},
            {});
    auto output0 = graph.make_output(reorder0->get_outputs());
    auto output1 = graph.make_output(reorder1->get_outputs());
    tensor_view_transform(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    EXPECT_EQ(graph.ops_.size(), 5UL);
    for (auto &op : graph.ops_) {
        if (op->isa<reorder_op_t>()) {
            EXPECT_EQ(op->get_outputs()[0]->details_.get_format(),
                    sc_data_format_t::MKmk(16, 16));
        } else if (op->isa<tensor_view_op_t>()) {
            EXPECT_EQ(op->get_outputs()[0]->details_.get_format(),
                    sc_data_format_t(fmtMmKk, {16, 16, 0, 0}));
        }
    }
}

constexpr sc_data_format_kind_t fmtAaBCbbc
        = sc_data_format_kind_t {0, 0, 1, 2, 1, 1, 2};
constexpr sc_data_format_kind_t fmtABCbc
        = sc_data_format_kind_t {0, 1, 2, 1, 2};
constexpr sc_data_format_kind_t fmtAaBbCbc
        = sc_data_format_kind_t {0, 0, 1, 1, 2, 1, 2};
constexpr sc_data_format_kind_t fmtABCbcb
        = sc_data_format_kind_t {0, 1, 2, 1, 2, 1};
TEST(GCCore_CPU_graph_tensor_view_transform, TestReorderToTransform2) {
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            {128, 64, 256}, sc_data_format_t(fmtAaBCbbc, {32, 64, 16, 128}))});
    // can eliminate
    auto reorder0 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make({128, 64, 256},
                    sc_data_format_t(fmtAaBCbbc, {32, 64, 32, 128}))},
            {});
    // can not eliminate
    auto reorder1 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make({128, 64, 256},
                    sc_data_format_t(fmtAaBCbbc, {32, 32, 16, 128}))},
            {});
    // can eliminate
    auto reorder2 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make({128, 64, 256},
                    sc_data_format_t(fmtABCbc, {64, 128, 0, 0}))},
            {});
    // can not eliminate
    auto reorder3 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make({128, 64, 256},
                    sc_data_format_t(fmtABCbc, {32, 128, 0, 0}))},
            {});
    // can not eliminate
    auto reorder4 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make({128, 64, 256},
                    sc_data_format_t(fmtAaBbCbc, {32, 64, 16, 128}))},
            {});
    // can not eliminate
    auto reorder5 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make({128, 64, 256},
                    sc_data_format_t(fmtABCbcb, {64, 128, 16, 0}))},
            {});
    auto output0 = graph.make_output(reorder0->get_outputs());
    auto output1 = graph.make_output(reorder1->get_outputs());
    auto output2 = graph.make_output(reorder2->get_outputs());
    auto output3 = graph.make_output(reorder3->get_outputs());
    auto output4 = graph.make_output(reorder4->get_outputs());
    auto output5 = graph.make_output(reorder5->get_outputs());

    tensor_view_transform(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    auto &ops = graph.ops_;
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder0) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder1) != ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder2) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder3) != ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder4) != ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder5) != ops.end());
    int tensor_view_op_count = 0;
    for (auto &op : ops) {
        if (op->isa<tensor_view_op_t>()) { tensor_view_op_count++; }
    }
    EXPECT_EQ(tensor_view_op_count, 2);
}

// Ones at begin of shapes.
TEST(GCCore_CPU_graph_tensor_view_transform, TestReorderToTransform3) {
    auto graph = sc_graph_t();
    auto inp = graph.make_input({graph_tensor::make(
            {128, 256}, sc_data_format_t(format_kinds::KN))});
    // can eliminate
    auto reorder0 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make({128, 256},
                    sc_data_format_t(format_kinds::NKkn, {32, 256}))},
            {});
    // can not eliminate
    auto reorder1 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make({128, 256},
                    sc_data_format_t(format_kinds::NKkn, {32, 32}))},
            {});
    // can not eliminate
    auto reorder2 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make({128, 256},
                    sc_data_format_t(format_kinds::NKknk, {32, 256, 4}))},
            {});
    // can eliminate
    auto reorder3 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(
                    {128, 256}, sc_data_format_t(format_kinds::MKmk, {1, 64}))},
            {});
    // can eliminate
    auto reorder4 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make({128, 256},
                    sc_data_format_t(format_kinds::MKmk, {32, 256}))},
            {});
    auto output0 = graph.make_output(reorder0->get_outputs());
    auto output1 = graph.make_output(reorder1->get_outputs());
    auto output2 = graph.make_output(reorder2->get_outputs());
    auto output3 = graph.make_output(reorder3->get_outputs());
    auto output4 = graph.make_output(reorder4->get_outputs());

    tensor_view_transform(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    auto &ops = graph.ops_;
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder0) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder1) != ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder2) != ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder3) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder4) == ops.end());

    int tensor_view_op_count = 0;
    for (auto &op : ops) {
        if (op->isa<tensor_view_op_t>()) { tensor_view_op_count++; }
    }
    EXPECT_EQ(tensor_view_op_count, 3);
}

// Ones at middle of shapes.
TEST(GCCore_CPU_graph_tensor_view_transform, TestReorderToTransform4) {
    auto graph = sc_graph_t();
    sc_dims plain_dims = {128, 1, 1, 384};
    auto inp = graph.make_input({graph_tensor::make(
            plain_dims, sc_data_format_t(format_kinds::ABCD))});
    // can eliminate
    auto reorder0 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(plain_dims,
                    sc_data_format_t(format_kinds::ABCDcd, {1, 64}))},
            {});
    // can eliminate
    auto reorder1 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(plain_dims,
                    sc_data_format_t(format_kinds::ABCDcd, {1, 1}))},
            {});
    // can eliminate
    auto reorder2 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(plain_dims,
                    sc_data_format_t(format_kinds::ABCDcd, {1, 384}))},
            {});
    // can not eliminate
    auto reorder3 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(plain_dims,
                    sc_data_format_t(format_kinds::ABCDcd, {32, 64}))},
            {});
    auto output0 = graph.make_output(reorder0->get_outputs());
    auto output1 = graph.make_output(reorder1->get_outputs());
    auto output2 = graph.make_output(reorder2->get_outputs());
    auto output3 = graph.make_output(reorder3->get_outputs());

    tensor_view_transform(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    auto &ops = graph.ops_;
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder0) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder1) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder2) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder3) != ops.end());
    int tensor_view_op_count = 0;
    for (auto &op : ops) {
        if (op->isa<tensor_view_op_t>()) { tensor_view_op_count++; }
    }
    EXPECT_EQ(tensor_view_op_count, 3);
}

constexpr sc_data_format_kind_t fmtABaCa
        = sc_data_format_kind_t {0, 1, 0, 2, 0};
// Ones at end of shapes
TEST(GCCore_CPU_graph_tensor_view_transform, TestReorderToTransform5) {
    auto graph = sc_graph_t();
    sc_dims plain_dims = {128, 256, 384};
    auto inp = graph.make_input({graph_tensor::make(
            plain_dims, sc_data_format_t(format_kinds::ABC))});
    // can eliminate
    auto reorder0 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(
                    plain_dims, sc_data_format_t(fmtABaCa, {1, 1}))},
            {});
    // can not eliminate
    auto reorder1 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(
                    plain_dims, sc_data_format_t(fmtABaCa, {64, 64}))},
            {});
    auto output0 = graph.make_output(reorder0->get_outputs());
    auto output1 = graph.make_output(reorder1->get_outputs());

    tensor_view_transform(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    auto &ops = graph.ops_;
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder0) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder1) != ops.end());

    int tensor_view_op_count = 0;
    for (auto &op : ops) {
        if (op->isa<tensor_view_op_t>()) { tensor_view_op_count++; }
    }
    EXPECT_EQ(tensor_view_op_count, 1);
}

constexpr sc_data_format_kind_t fmtABDC = sc_data_format_kind_t {0, 1, 3, 2};
constexpr sc_data_format_kind_t fmtADCB = sc_data_format_kind_t {0, 3, 2, 1};
constexpr sc_data_format_kind_t fmtDABC = sc_data_format_kind_t {3, 0, 1, 2};
TEST(GCCore_CPU_graph_tensor_view_transform, TestReorderToTransform6) {
    auto graph = sc_graph_t();
    sc_dims plain_dims = {128, 1, 1, 384};
    auto inp = graph.make_input({graph_tensor::make(
            plain_dims, sc_data_format_t(format_kinds::ABCD))});
    // can eliminate
    auto reorder0 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(
                    plain_dims, sc_data_format_t(format_kinds::ACBD))},
            {});
    // can eliminate
    auto reorder1 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(plain_dims, sc_data_format_t(fmtABDC))}, {});
    // can eliminate
    auto reorder2 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(plain_dims, sc_data_format_t(fmtADCB))}, {});
    // can not eliminate
    auto reorder3 = graph.make("reorder", inp->get_outputs(),
            {graph_tensor::make(plain_dims, sc_data_format_t(fmtDABC))}, {});
    auto output0 = graph.make_output(reorder0->get_outputs());
    auto output1 = graph.make_output(reorder1->get_outputs());
    auto output2 = graph.make_output(reorder2->get_outputs());
    auto output3 = graph.make_output(reorder3->get_outputs());

    tensor_view_transform(graph, get_test_ctx());
    EXPECT_EQ(check_graph_connection(graph), true);
    auto &ops = graph.ops_;
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder0) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder1) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder2) == ops.end());
    EXPECT_TRUE(std::find(ops.begin(), ops.end(), reorder3) != ops.end());

    int tensor_view_op_count = 0;
    for (auto &op : ops) {
        if (op->isa<tensor_view_op_t>()) { tensor_view_op_count++; }
    }
    EXPECT_EQ(tensor_view_op_count, 3);
}
