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
#include <vector>
#include "context.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/unary_elemwise.hpp>

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_graph_padded_mask_mark_cpp, TestPaddedMaskMark) {
    auto graph = sc_graph_t();
    auto inp1 = graph.make_input(
            {graph_tensor::make({64, 128}, sc_data_format_t::MKmk(16, 32))});
    auto inp2 = graph.make_input({graph_tensor::make({1})});
    auto inp3 = graph.make_input({graph_tensor::make(
            {128, 64}, sc_data_format_t(format_kinds::AB))});
    auto relu = graph.make("relu", inp1->get_outputs(), {}, {});
    auto add1 = graph.make(
            "add", {inp1->get_outputs()[0], inp2->get_outputs()[0]}, {}, {});
    auto mul1 = graph.make(
            "mul", {relu->get_outputs()[0], add1->get_outputs()[0]}, {}, {});
    auto div1 = graph.make(
            "div", {mul1->get_outputs()[0], add1->get_outputs()[0]}, {}, {});
    auto exp1 = graph.make("exp", {div1->get_outputs()[0]}, {}, {});
    auto matmul = graph.make("matmul_core",
            {exp1->get_outputs()[0], inp3->get_outputs()[0]}, {}, {});
    auto exp2 = graph.make("exp", relu->get_outputs(), {}, {});
    auto select = graph.make("select",
            {inp1->get_outputs()[0], inp1->get_outputs()[0],
                    inp2->get_outputs()[0]},
            {}, {});
    // output
    auto mul2 = graph.make(
            "mul", {mul1->get_outputs()[0], exp1->get_outputs()[0]}, {}, {});
    auto reduce1 = graph.make("reduce", add1->get_outputs(), {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});
    auto reduce2 = graph.make("reduce", select->get_outputs(), {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});
    auto tv1 = graph.make("tensor_view", relu->get_outputs(), {},
            {{"shape", sc_dims {2, 64, 64}}});
    auto tv2 = graph.make("tensor_view", exp2->get_outputs(), {},
            {{"shape", sc_dims {2, 32, 128}}});
    auto outs = graph.make_output({mul2->get_outputs()[0],
            reduce1->get_outputs()[0], reduce2->get_outputs()[0],
            tv1->get_outputs()[0], tv2->get_outputs()[0]});
    padded_mask_mark(graph, get_test_ctx());
    EXPECT_EQ(relu->attrs_.get_or_else(op_attr_key::use_padded_mask, true),
            false);
    EXPECT_EQ(
            add1->attrs_.get_or_else(op_attr_key::use_padded_mask, true), true);
    EXPECT_EQ(mul1->attrs_.get_or_else(op_attr_key::use_padded_mask, true),
            false);
    EXPECT_EQ(div1->attrs_.get_or_else(op_attr_key::use_padded_mask, true),
            false);
    EXPECT_EQ(
            exp1->attrs_.get_or_else(op_attr_key::use_padded_mask, true), true);
    EXPECT_EQ(mul2->attrs_.get_or_else(op_attr_key::use_padded_mask, true),
            false);
    EXPECT_EQ(
            exp2->attrs_.get_or_else(op_attr_key::use_padded_mask, true), true);
    EXPECT_EQ(select->attrs_.get_or_else(op_attr_key::use_padded_mask, true),
            true);
}
