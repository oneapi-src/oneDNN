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
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/jit/jit.hpp>
#include <runtime/config.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

TEST(GCCore_CPU_graph_batchwise_merge_cpp, TestGraphBatchWiseMultiLevel) {
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

    std::stringstream ss;
    batchwise_merge(graph);
    print_graph(graph, ss, 1);
    std::string expected_str;
    const int run_threads = runtime_config_t::get().get_num_threads();
    int top_level_prod = 16 * 32 * 384, least_level_prod = 16;

    auto parallelism_check = [&run_threads](int prod) {
        if (prod == 1) return false;
        return (prod / run_threads > 8
                || (prod % run_threads == 0 && (prod / run_threads) >= 1));
    };

    if (parallelism_check(least_level_prod)
            && parallelism_check(top_level_prod)) {
        expected_str
                = R"(graph(v0: f32[16, 32, 384, 1024], v1: f32[16, 32, 384, 1024], v2: f32[16, 1, 384, 1]) -> [v3: f32[8, 2, 32, 384, 1024]] {
  [v4: f32[16, 32, 384, 1024], v5: f32[16, 1, 384, 1]] = batchwise_16_fused_add_reduce(v0, v1)
  [v6: f32[16, 32, 384, 1024]] = batchwise_16X32X384_fused_add_add(v4, v5, v2)
  [v3: f32[8, 2, 32, 384, 1024]] = tensor_view(v6)
}
)";
    } else if (parallelism_check(top_level_prod)) {
        expected_str
                = R"(graph(v0: f32[16, 32, 384, 1024], v1: f32[16, 32, 384, 1024], v2: f32[16, 1, 384, 1]) -> [v3: f32[8, 2, 32, 384, 1024]] {
  [v4: f32[16, 32, 384, 1024]] = add(v0, v1)
  [v5: f32[16, 1, 384, 1]] = reduce(v4)
  [v6: f32[16, 32, 384, 1024]] = batchwise_16X32X384_fused_add_add(v4, v5, v2)
  [v3: f32[8, 2, 32, 384, 1024]] = tensor_view(v6)
}
)";
    } else {
        expected_str
                = R"(graph(v0: f32[16, 32, 384, 1024], v1: f32[16, 32, 384, 1024], v2: f32[16, 1, 384, 1]) -> [v3: f32[8, 2, 32, 384, 1024]] {
  [v4: f32[16, 32, 384, 1024]] = add(v0, v1)
  [v5: f32[16, 1, 384, 1]] = reduce(v4)
  [v6: f32[16, 32, 384, 1024]] = add(v4, v5)
  [v7: f32[16, 32, 384, 1024]] = add(v6, v2)
  [v3: f32[8, 2, 32, 384, 1024]] = tensor_view(v7)
}
)";
    }

    EXPECT_EQ(ss.str(), expected_str);
}

TEST(GCCore_CPU_graph_batchwise_merge_cpp, TestGraphBatchWiseAggresiveShrink) {
    sc_graph_t graph;
    auto input_A = graph.make_input({graph_tensor::make({64, 16, 384, 64})});
    auto input_B = graph.make_input({graph_tensor::make({64, 16, 64, 384})});
    auto input_C = graph.make_input({graph_tensor::make({64, 16, 384, 64})});
    auto input_D = graph.make_input({graph_tensor::make({64, 1, 1, 384})});
    auto matmul0 = graph.make("matmul_core",
            {input_A->get_outputs()[0], input_B->get_outputs()[0]}, {}, {});
    auto add0 = graph.make("add",
            {matmul0->get_outputs()[0], input_D->get_outputs()[0]}, {}, {});
    auto tv0 = graph.make("tensor_view", {add0->get_outputs()[0]}, {},
            {{"shape", sc_dims {64 * 16 * 384, 384}}});
    auto softmax = graph.make("softmax", {tv0->get_outputs()[0]}, {},
            {{"axis", std::vector<int> {1}}});
    auto tv1 = graph.make("tensor_view", {softmax->get_outputs()[0]}, {},
            {{"shape", sc_dims {64, 16, 384, 384}},
                    {"format", sc_data_format_t(format_kinds::ABCD)}});
    auto matmul1 = graph.make("matmul_core",
            {tv1->get_outputs()[0], input_C->get_outputs()[0]}, {}, {});
    auto output = graph.make_output(matmul1->get_outputs());

    std::stringstream ss;
    graph_driver(graph, get_test_ctx());
    print_graph(graph, ss, 1);
    std::string expected_str;
    const int run_threads = runtime_config_t::get().get_num_threads();
    int bwise_prod = 64 * 16;

    auto parallelism_check = [&run_threads](int prod) {
        if (prod == 1) return false;
        return (prod / run_threads > 8
                || (prod % run_threads == 0 && (prod / run_threads) >= 1));
    };

    if (parallelism_check(bwise_prod)) {
        expected_str
                = R"(graph(v0: f32[64, 16, 384, 64], v1: f32[64, 16, 64, 384], v2: f32[64, 16, 384, 64], v3: f32[64, 1, 1, 384]) -> [v4: f32[64, 16, 384, 64]] {
  [v4: f32[64, 16, 384, 64]] = batchwise_64X16_fused_matmul_core_add_tensor_view_reduce_sub_exp_reduce_reciprocal_mul_tensor_view_matmul_core(v0, v1, v3, v2)
}
)";
    } else {
        expected_str
                = R"(graph(v0: f32[64, 16, 384, 64], v1: f32[64, 16, 64, 384], v2: f32[64, 16, 384, 64], v3: f32[64, 1, 1, 384]) -> [v4: f32[64, 16, 384, 64]] {
  [v5: f32[64, 16, 384, 384]] = matmul_core_add_tensor_view_reduce_sub_exp_reduce_reciprocal_mul(v0, v1, v3)
  [v6: f32[64, 16, 384, 384]] = tensor_view(v5)
  [v4: f32[64, 16, 384, 64]] = matmul_core(v6, v2)
}
)";
    }

    EXPECT_EQ(ss.str(), expected_str);
}
