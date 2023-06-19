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

#include <cmath>
#include <iostream>
#include <vector>
#include "context.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/jit/jit.hpp>
#include <reference/act_ref.hpp>
#include <reference/softmax_ref.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;
static bool verbose = false;

static void check_reciprocal(const sc_dims &input_dims) {
    REQUIRE_AVX2();
    sc_graph_t g;
    auto ins = g.make_input({graph_tensor::make(input_dims)});
    auto op = g.make("reciprocal", ins->get_outputs(), {}, {});
    g.make_output(op->get_outputs());
    graph_driver(g, get_test_ctx());

    auto f = lower_graph(get_test_ctx(), g, {});
    if (verbose) std::cout << f << std::endl;
    const auto input_size = test_utils::product(input_dims);
    std::vector<float> input_data(input_size);
    test_utils::fill_data(&input_data[0], input_size);
    std::vector<float> ref_output(input_size);
    ref_reciprocal(ref_output.data(), input_data.data(), input_size);
    std::vector<float> sc_output(input_size);
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f);
    fptr->call_default(&input_data[0], &sc_output[0]);
    test_utils::compare_data(sc_output, ref_output, 1e-4f, 1e-5f);
}

TEST(GCCore_CPU_reciprocal_test, TestReciprocalOp) {
    check_reciprocal({32, 64, 48});
    check_reciprocal({1, 12, 128, 128});
}

TEST(GCCore_CPU_reciprocal_test, TestAttentionSubGraph) {
    sc_graph_t g;
    sc_dims output_dims = {12, 128, 128};
    sc_dims input_dims_13 = {12, 128, 128};
    sc_dims input_dims_4 = {12, 128, 1};
    sc_dims input_dims_3 = {1, 1, 128};

    ///////////////
    auto v13 = g.make_input({graph_tensor::make(
            sc_dims(input_dims_13), sc_data_format_t(), datatypes::f32)});
    auto v4 = g.make_input({graph_tensor::make(
            sc_dims(input_dims_4), sc_data_format_t(), datatypes::f32)});
    auto v3 = g.make_input({graph_tensor::make(
            sc_dims(input_dims_3), sc_data_format_t(), datatypes::f32)});

    ///////////////
    auto v14 = g.make("div", //
            {v13->get_outputs()[0], v4->get_outputs()[0]}, {}, {});
    auto v15 = g.make("add", //
            {v14->get_outputs()[0], v3->get_outputs()[0]},
            {graph_tensor::make(
                    output_dims, sc_data_format_t(), datatypes::f32)},
            {});
    auto v16 = g.make("softmax", //
            {v15->get_outputs()[0]},
            {graph_tensor::make(
                    output_dims, sc_data_format_t(), datatypes::f32)},
            {{"axis", std::vector<int> {2}}});
    auto out = g.make_output(v16->get_outputs());

    graph_inline(g);
    div_bcast_transform(g);
    std::stringstream ss;
    print_graph(g, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[12, 128, 128], v1: f32[12, 128, 1], v2: f32[1, 1, 128]) -> [v3: f32[12, 128, 128]] {
  [v4: f32[12, 128, 1]] = reciprocal(v1)
  [v5: f32[12, 128, 128]] = mul(v0, v4)
  [v6: f32[12, 128, 128]] = add(v5, v2)
  [v7: f32[12, 128, 1]] = reduce(v6)
  [v8: f32[12, 128, 128]] = sub(v6, v7)
  [v9: f32[12, 128, 128]] = exp(v8)
  [v10: f32[12, 128, 1]] = reduce(v9)
  [v11: f32[12, 128, 1]] = reciprocal(v10)
  [v3: f32[12, 128, 128]] = mul(v9, v11)
}
)";
    EXPECT_EQ(ss.str(), expected_str);
}
