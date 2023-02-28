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

#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <runtime/context.hpp>

#include <iostream>
#include "exception_util.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
TEST(GCCore_graph_reshape_cpp, TestGraphReshapeCreation) {
    sc_graph_t g;
    graph_tensor_ptr in0, in1;
    auto make_graph = [&](int32_t *values, size_t num_ele) {
        sc_graph_t newg;
        in0 = newg.make_input({graph_tensor::make({100, 200})})
                      ->get_outputs()[0];
        in1 = newg.make_input({graph_tensor::make(
                                      {4}, sc_data_format_t(), datatypes::s32)},
                          {{"values",
                                  std::make_shared<static_data_t>(
                                          values, num_ele * sizeof(int32_t))}})
                      ->get_outputs()[0];
        g = std::move(newg);
    };
    any_map_t attr = {{"special_zero", true}};
    {
        int32_t values[] = {1, 2, 3};
        make_graph(values, 3);
        EXPECT_SC_ERROR(g.make("dynamic_reshape", {in0, in1}, {}, attr),
                "Bad shape data");
    }

    {
        sc_graph_t g;
        auto in0 = g.make_input({graph_tensor::make({100, 200})})
                           ->get_outputs()[0];
        auto in1 = g.make_input({graph_tensor::make({4}, sc_data_format_t(),
                                        datatypes::s32)})
                           ->get_outputs()[0];
        EXPECT_SC_ERROR(g.make("dynamic_reshape", {in0, in1}, {}, attr),
                "Since dynamic shape is not supported yet, we are expecting "
                "the");
    }

    {
        sc_graph_t g;
        auto in0 = g.make_input({graph_tensor::make({100, 200})})
                           ->get_outputs()[0];
        auto in1 = g.make_input({graph_tensor::make({1, 4}, sc_data_format_t(),
                                        datatypes::s32)},
                            {{"values",
                                    std::make_shared<static_data_t>(20,
                                            runtime::get_default_stream()
                                                    ->engine_)}})
                           ->get_outputs()[0];
        EXPECT_SC_ERROR(g.make("dynamic_reshape", {in0, in1}, {}, attr),
                "Expecting 1D and int32/int64 tensor for input 2 of");
    }

    {
        sc_graph_t g;
        auto in0 = g.make_input({graph_tensor::make({10}, sc_data_format_t(),
                                        datatypes::s32)})
                           ->get_outputs()[0];
        auto in1 = g.make("add", {in0, in0}, {}, {})->get_outputs()[0];
        EXPECT_SC_ERROR(g.make("dynamic_reshape", {in0, in1}, {}, attr),
                "Reshape requires compile-time constant shape for now");
    }

    {
        int32_t values[] = {100, -1, 2, -1};
        make_graph(values, 4);
        EXPECT_SC_ERROR(g.make("dynamic_reshape", {in0, in1}, {}, attr),
                "reshape only support one -1 shape");
    }

    {
        int32_t values[] = {0, 2, 10, 0};
        make_graph(values, 4);
        EXPECT_SC_ERROR(g.make("dynamic_reshape", {in0, in1}, {}, attr),
                "The special zero at ");
    }

    {
        int32_t values[] = {100, 500, 1, -1};
        make_graph(values, 4);
        EXPECT_SC_ERROR(g.make("dynamic_reshape", {in0, in1}, {}, attr),
                "Reshape: The input tensor size does not match the given "
                "shape");
    }

    {
        int32_t values[] = {100, 1, 1, 1};
        make_graph(values, 4);
        EXPECT_SC_ERROR(g.make("dynamic_reshape", {in0, in1}, {}, attr),
                "Reshape: The input tensor size does not match the given "
                "shape");
    }

    {
        int32_t values[] = {100, 200, 1, 1};
        make_graph(values, 4);
        EXPECT_SC_ERROR(g.make("dynamic_reshape", {in0, in1},
                                {graph_tensor::make({200, 100, 1, 1})}, attr),
                "Reshape: Expecting output shape = ");
    }

    // tests for inferring the output shape

    {
        int32_t values[] = {100, 2, 10, -1};
        make_graph(values, 4);
        auto ret = g.make("dynamic_reshape", {in0, in1}, {}, attr);
        EXPECT_EQ(ret->get_outputs()[0]->details_.get_plain_dims(),
                (sc_dims {100, 2, 10, 10}));
    }
    {
        int32_t values[] = {0, 2, 10, 10};
        make_graph(values, 4);
        auto ret = g.make("dynamic_reshape", {in0, in1}, {}, attr);
        EXPECT_EQ(ret->get_outputs()[0]->details_.get_plain_dims(),
                (sc_dims {100, 2, 10, 10}));
    }
}

TEST(GCCore_graph_reshape_cpp, TestConstOptimize) {
    std::vector<bool> reshape_type {true, false};
    for (auto dynamic_reshape : reshape_type) {
        auto get_graph = [](bool dynamic_reshape) {
            sc_graph_t g;

            auto in0 = g.make_input({graph_tensor::make({100, 200})})
                               ->get_outputs()[0];
            auto in1 = g.make_input({graph_tensor::make({100, 200})})
                               ->get_outputs()[0];
            auto in2 = g.make_input({graph_tensor::make({10, 10, 20, 10})})
                               ->get_outputs()[0];

            auto addout = g.make("add", {in0, in1}, {}, {})->get_outputs()[0];
            graph_tensor_ptr shape_in;
            if (dynamic_reshape) {
                int32_t shapes[] = {10, 10, 20, 10};
                shape_in = g.make_input(
                                    {graph_tensor::make({4}, sc_data_format_t(),
                                            datatypes::s32)},
                                    {{"values",
                                            std::make_shared<static_data_t>(
                                                    shapes,
                                                    4 * sizeof(int32_t))}})
                                   ->get_outputs()[0];
            }
            any_map_t attr = {{"special_zero", true}};
            if (!dynamic_reshape) {
                attr.set("shape", sc_dims {10, 10, 20, 10});
            }
            std::string op_name
                    = dynamic_reshape ? "dynamic_reshape" : "static_reshape";
            std::vector<graph_tensor_ptr> ins = {addout};
            if (dynamic_reshape) { ins.emplace_back(shape_in); }
            auto reshape_out = g.make(op_name, ins, {}, attr)->get_outputs()[0];
            auto addout2 = g.make("add", {reshape_out, in2}, {}, {})
                                   ->get_outputs()[0];
            g.make_output({addout2});
            return g;
        };
        sc_graph_t g = get_graph(dynamic_reshape);

        constant_optimization(g, get_default_context());

        sc_graph_t expected;
        {
            auto in0 = expected.make_input({graph_tensor::make({100, 200})})
                               ->get_outputs()[0];
            auto in1 = expected.make_input({graph_tensor::make({100, 200})})
                               ->get_outputs()[0];
            auto in2 = expected.make_input(
                                       {graph_tensor::make({10, 10, 20, 10})})
                               ->get_outputs()[0];

            auto addout = expected.make("add", {in0, in1}, {}, {})
                                  ->get_outputs()[0];
            graph_tensor_ptr shape_in;
            if (dynamic_reshape) {
                int32_t shapes[] = {10, 10, 20, 10};
                auto data = std::make_shared<static_data_t>(
                        shapes, 4 * sizeof(int32_t));
                shape_in = expected.make_input({graph_tensor::make({4},
                                                       sc_data_format_t(),
                                                       datatypes::s32)},
                                           {{"values", data}})
                                   ->get_outputs()[0];
            }
            any_map_t attr = {{"special_zero", true}};
            auto reshape_out
                    = expected.make("tensor_view", {addout}, {},
                                      {{"shape", sc_dims {10, 10, 20, 10}}})
                              ->get_outputs()[0];
            auto addout2 = expected.make("add", {reshape_out, in2}, {}, {})
                                   ->get_outputs()[0];
            expected.make_output({addout2});
        }
        EXPECT_TRUE(compare_graph(g, expected));

        g = get_graph(dynamic_reshape);
        graph_driver(g);
        lower_graph(get_default_context(), g, {});
    }
}

TEST(GCCore_graph_reshape_cpp, TestSingleOptimize) {
    std::vector<bool> reshape_type {true, false};
    for (auto dynamic_reshape : reshape_type) {
        auto get_graph = [](bool dynamic_reshape) {
            sc_graph_t g;

            auto in0 = g.make_input({graph_tensor::make({100, 200})})
                               ->get_outputs()[0];
            graph_tensor_ptr shape_in;
            if (dynamic_reshape) {
                int32_t shapes[] = {10, 10, 20, 10};
                shape_in = g.make_input(
                                    {graph_tensor::make({4}, sc_data_format_t(),
                                            datatypes::s32)},
                                    {{"values",
                                            std::make_shared<static_data_t>(
                                                    shapes,
                                                    4 * sizeof(int32_t))}})
                                   ->get_outputs()[0];
            }
            any_map_t attr = {{"special_zero", true}};
            if (!dynamic_reshape) {
                attr.set("shape", sc_dims {10, 10, 20, 10});
            }
            std::string op_name
                    = dynamic_reshape ? "dynamic_reshape" : "static_reshape";
            std::vector<graph_tensor_ptr> ins = {in0};
            if (dynamic_reshape) { ins.emplace_back(shape_in); }
            auto reshape_out = g.make(op_name, ins, {}, attr)->get_outputs()[0];
            g.make_output({reshape_out});
            return g;
        };
        sc_graph_t g = get_graph(dynamic_reshape);

        constant_optimization(g, get_default_context());
        inplace_transform(g, get_default_context());

        sc_graph_t expected;
        {
            auto in0 = expected.make_input({graph_tensor::make({100, 200})})
                               ->get_outputs()[0];
            graph_tensor_ptr shape_in;
            if (dynamic_reshape) {
                int32_t shapes[] = {10, 10, 20, 10};
                auto data = std::make_shared<static_data_t>(
                        shapes, 4 * sizeof(int32_t));
                shape_in = expected.make_input({graph_tensor::make({4},
                                                       sc_data_format_t(),
                                                       datatypes::s32)},
                                           {{"values", data}})
                                   ->get_outputs()[0];
            }
            any_map_t attr = {{"special_zero", true}};
            auto reshape_out
                    = expected.make("reshape", {in0},
                                      {graph_tensor::make({10, 10, 20, 10})},
                                      {{"shape", sc_dims {10, 10, 20, 10}}})
                              ->get_outputs()[0];
            expected.make_output({reshape_out});
        }
        EXPECT_TRUE(compare_graph(g, expected));

        lower_graph(get_default_context(), g, {});
    }
}

TEST(GCCore_graph_reshape_cpp, TestSingleOptimizeMultipleUse) {
    std::vector<bool> reshape_type {true, false};
    for (auto dynamic_reshape : reshape_type) {
        auto get_graph = [](bool dynamic_reshape) {
            sc_graph_t g;

            auto in0 = g.make_input({graph_tensor::make({100, 200})})
                               ->get_outputs()[0];
            graph_tensor_ptr shape_in;
            if (dynamic_reshape) {
                int32_t shapes[] = {10, 10, 20, 10};
                shape_in = g.make_input(
                                    {graph_tensor::make({4}, sc_data_format_t(),
                                            datatypes::s32)},
                                    {{"values",
                                            std::make_shared<static_data_t>(
                                                    shapes,
                                                    4 * sizeof(int32_t))}})
                                   ->get_outputs()[0];
            }
            any_map_t attr = {{"special_zero", true}};
            if (!dynamic_reshape) {
                attr.set("shape", sc_dims {10, 10, 20, 10});
            }
            std::string op_name
                    = dynamic_reshape ? "dynamic_reshape" : "static_reshape";
            std::vector<graph_tensor_ptr> ins = {in0};
            if (dynamic_reshape) { ins.emplace_back(shape_in); }
            auto reshape_out = g.make(op_name, ins, {}, attr)->get_outputs()[0];
            auto relu_out
                    = g.make("relu", {reshape_out}, {}, {})->get_outputs()[0];
            g.make_output({reshape_out, relu_out});
            return g;
        };
        sc_graph_t g = get_graph(dynamic_reshape);

        constant_optimization(g, get_default_context());
        inplace_transform(g, get_default_context());

        sc_graph_t expected;
        {
            auto in0 = expected.make_input({graph_tensor::make({100, 200})})
                               ->get_outputs()[0];
            graph_tensor_ptr shape_in;
            if (dynamic_reshape) {
                int32_t shapes[] = {10, 10, 20, 10};
                auto data = std::make_shared<static_data_t>(
                        shapes, 4 * sizeof(int32_t));
                shape_in = expected.make_input({graph_tensor::make({4},
                                                       sc_data_format_t(),
                                                       datatypes::s32)},
                                           {{"values", data}})
                                   ->get_outputs()[0];
            }
            any_map_t attr = {{"special_zero", true}};
            auto reshape_out
                    = expected.make("reshape", {in0},
                                      {graph_tensor::make({10, 10, 20, 10})},
                                      {{"shape", sc_dims {10, 10, 20, 10}}})
                              ->get_outputs()[0];
            auto relu_out = expected.make("relu", {reshape_out}, {}, {})
                                    ->get_outputs()[0];
            expected.make_output({reshape_out, relu_out});
        }
        EXPECT_TRUE(compare_graph(g, expected));

        lower_graph(get_default_context(), g, {});
    }
}
