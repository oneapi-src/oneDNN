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

#include "test_utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/jit/jit.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "exception_util.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;
static sc_graph_t get_graph_to_merge() {
    sc_graph_t graph;
    auto input = graph.make_input({graph_tensor::make({384, 1024})});
    auto weight0 = graph.make_input({graph_tensor::make({1024, 1024})});
    auto weight1 = graph.make_input({graph_tensor::make({1024, 1024})});
    auto weight2 = graph.make_input({graph_tensor::make({1024, 1024})});
    auto weight3 = graph.make_input({graph_tensor::make({1024, 2048})});
    auto weight4 = graph.make_input({graph_tensor::make({1024, 4096})});

    auto matmul0 = graph.make("matmul_core",
            {input->get_outputs()[0], weight0->get_outputs()[0]}, {},
            {{"horizontal_merge", 1}});
    auto matmul1 = graph.make("matmul_core",
            {input->get_outputs()[0], weight1->get_outputs()[0]}, {},
            {{"horizontal_merge", 1}});
    auto matmul2 = graph.make("matmul_core",
            {input->get_outputs()[0], weight2->get_outputs()[0]}, {},
            {{"horizontal_merge", 1}});
    auto matmul3 = graph.make("matmul_core",
            {input->get_outputs()[0], weight3->get_outputs()[0]}, {},
            {{"horizontal_merge", 2}});
    auto matmul4 = graph.make("matmul_core",
            {input->get_outputs()[0], weight4->get_outputs()[0]}, {},
            {{"horizontal_merge", 2}});
    auto output0 = graph.make_output(matmul0->get_outputs());
    auto output1 = graph.make_output(matmul1->get_outputs());
    auto output2 = graph.make_output(matmul2->get_outputs());
    auto output3 = graph.make_output(matmul3->get_outputs());
    auto output4 = graph.make_output(matmul4->get_outputs());
    return graph;
}

TEST(GCCore_CPU_graph_horizontal_merge_cpp, TestGraphHorizontalMerge) {
    sc_graph_t graph = get_graph_to_merge();
    //     layout_propagation(graph);
    horizontal_merge(graph);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected
            = R"(graph(v0: f32[384, 1024], v1: f32[1024, 1024], v2: f32[1024, 1024], v3: f32[1024, 1024], v4: f32[1024, 2048], v5: f32[1024, 4096]) -> [v6: f32[384, 1024], v7: f32[384, 1024], v8: f32[384, 1024], v9: f32[384, 2048], v10: f32[384, 4096]] {
  [v9: f32[384, 2048], v10: f32[384, 4096]] = horizontal_fused_matmul_core_matmul_core_(v0, v4, v5)
  [v6: f32[384, 1024], v7: f32[384, 1024], v8: f32[384, 1024]] = horizontal_fused_matmul_core_matmul_core_matmul_core_(v0, v1, v2, v3)
}
)";
    EXPECT_EQ(ss.str(), expected);
    auto horizontal_fused_op0 = graph.ops_[11];
    auto horizontal_fused_op1 = graph.ops_[12];
    EXPECT_EQ(horizontal_fused_op0->get_inputs().size(), 4UL);
    EXPECT_EQ(horizontal_fused_op0->get_outputs().size(), 3UL);
    EXPECT_EQ(horizontal_fused_op1->get_inputs().size(), 3UL);
    EXPECT_EQ(horizontal_fused_op1->get_outputs().size(), 2UL);

    // test lower
    sc_graph_t graph1 = get_graph_to_merge();
    std::vector<sc_op_ptr> args;
    std::vector<sc_op_ptr> ins = graph1.get_input_ops();
    std::vector<sc_op_ptr> outs = graph1.get_output_ops();
    args.insert(args.end(), ins.begin(), ins.end());
    args.insert(args.end(), outs.begin(), outs.end());
    graph_driver(graph1);
    ir_module_ptr f = lower_graph(get_default_context(), graph1, args);
    REQUIRE_AVX2();
    EXPECT_NO_FATAL_FAILURE(
            jit_engine_t::make(get_default_context())->get_entry_func(f));
}
