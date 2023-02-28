
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
 ******************************************************************************/
#include <iostream>
#include "compiler/ir/graph/driver.hpp"
#include "compiler/ir/graph/fusible_op.hpp"
#include "compiler/ir/graph/graph.hpp"
#include "compiler/ir/graph/graph_op.hpp"
#include "compiler/ir/graph/lowering.hpp"
#include "compiler/ir/graph/pass/pass.hpp"
#include "compiler/ir/graph/transform/transform.hpp"
#include "compiler/jit/jit.hpp"
#include "context.hpp"
#include "reference/conv_ref.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

TEST(GCCore_graph_conv_test, TestGraphConvolutionWithBias) {
    int N = 64, IC = 16, OC = 64, H = 32, W = 32, R = 1, S = 1;
    sc_dims input_dims {N, IC, H, W};
    sc_dims filter_dims {OC, IC, R, S};
    sc_dims output_dims {N, OC, H, W};
    auto ins0 = graph_tensor::make(input_dims);
    auto ins1 = graph_tensor::make(filter_dims);
    auto ins2 = graph_tensor::make({OC});
    auto out0 = graph_tensor::make(output_dims);

    std::unordered_map<std::string, any_t> attrs = {{"data_format", "NCX"},
            {"weights_format", "OIX"}, {"strides", sc_dims {1, 1}},
            {"pads_begin", sc_dims {0, 0}}, {"pads_end", sc_dims {0, 0}}};

    sc_graph_t graph;
    auto in = graph.make_input({ins0, ins1, ins2});
    auto conv = graph.make(
            "conv_fwd", in->get_outputs(), {out0}, any_map_t(attrs));
    auto out = graph.make_output(conv->get_outputs());
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = 1;
    // The conv1d graph is different from the reference
    graph.attrs_["no_conv1d"] = true;
    graph_driver(graph, ctx);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected_graph
            = R"(graph(v0: f32[64, 16, 32, 32], v1: f32[64, 16, 1, 1], v2: f32[64]) -> [v3: f32[64, 64, 32, 32]] {
  [v4: f32[1, 1, 1, 64]] = tensor_view(v2)
  [v5: f32[1, 1, 1, 1, 16, 64]] = reorder(v1)
  [v6: f32[64, 32, 32, 16]] = reorder(v0)
  [v3: f32[64, 64, 32, 32]] = outerloop_64X1_partition_conv_fwd_core_add_reorder(v6, v5, v4)
}
)";
    EXPECT_EQ(ss.str(), expected_graph);
}

TEST(GCCore_graph_conv_test, TestGraphConvolutionNXCXIO) {
    int N = 64, IC = 16, OC = 64, H = 32, W = 32, R = 1, S = 1;
    sc_dims input_dims {N, H, W, IC};
    sc_dims filter_dims {R, S, IC, OC};
    sc_dims output_dims {N, H, W, OC};
    auto ins0 = graph_tensor::make(
            input_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins1 = graph_tensor::make(
            filter_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins2 = graph_tensor::make({OC});
    auto out0 = graph_tensor::make(output_dims);

    std::unordered_map<std::string, any_t> attrs = {{"strides", sc_dims {1, 1}},
            {"pads_begin", sc_dims {0, 0}}, {"pads_end", sc_dims {0, 0}}};

    sc_graph_t graph;
    auto in = graph.make_input({ins0, ins1, ins2});
    auto conv = graph.make(
            "conv_fwd", in->get_outputs(), {out0}, any_map_t(attrs));
    auto out = graph.make_output(conv->get_outputs());
    // The conv1d graph is different from the reference
    graph.attrs_["no_conv1d"] = true;
    graph_driver(graph, get_test_ctx());
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected_graph
            = R"(graph(v0: f32[64, 32, 32, 16], v1: f32[1, 1, 16, 64], v2: f32[64]) -> [v3: f32[64, 32, 32, 64]] {
  [v4: f32[1, 1, 1, 64]] = tensor_view(v2)
  [v5: f32[1, 1, 1, 1, 16, 64]] = tensor_view(v1)
  [v6: f32[64, 32, 32, 16]] = tensor_view(v0)
  [v3: f32[64, 32, 32, 64]] = conv_fwd_core_tensor_view_add(v6, v5, v4)
}
)";
    EXPECT_EQ(ss.str(), expected_graph);
}

TEST(GCCore_graph_conv_test, TestGraphConvolutionNXCXIODifferentHW) {
    int N = 64, IC = 16, OC = 64, H = 128, W = 32, R = 1, S = 1;
    sc_dims input_dims {N, H, W, IC};
    sc_dims filter_dims {R, S, IC, OC};
    sc_dims output_dims {N, H, W, OC};
    auto ins0 = graph_tensor::make(
            input_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins1 = graph_tensor::make(
            filter_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins2 = graph_tensor::make({OC});
    auto out0 = graph_tensor::make(output_dims);

    std::unordered_map<std::string, any_t> attrs = {{"strides", sc_dims {1, 1}},
            {"pads_begin", sc_dims {0, 0}}, {"pads_end", sc_dims {0, 0}}};

    sc_graph_t graph;
    auto in = graph.make_input({ins0, ins1, ins2});
    auto conv = graph.make(
            "conv_fwd", in->get_outputs(), {out0}, any_map_t(attrs));
    auto out = graph.make_output(conv->get_outputs());
    // The conv1d graph is different from the reference
    graph.attrs_["no_conv1d"] = true;
    graph_driver(graph, get_test_ctx());
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected_graph
            = R"(graph(v0: f32[64, 128, 32, 16], v1: f32[1, 1, 16, 64], v2: f32[64]) -> [v3: f32[64, 128, 32, 64]] {
  [v4: f32[1, 1, 1, 64]] = tensor_view(v2)
  [v5: f32[1, 1, 1, 1, 16, 64]] = tensor_view(v1)
  [v6: f32[64, 128, 32, 16]] = tensor_view(v0)
  [v3: f32[64, 128, 32, 64]] = conv_fwd_core_tensor_view_add(v6, v5, v4)
}
)";
    EXPECT_EQ(ss.str(), expected_graph);
}

TEST(GCCore_graph_conv_test, TestGraphConvolutionAutoPad) {
    {
        int N = 64, IC = 16, OC = 64, H = 32, W = 32, R = 3, S = 3;
        sc_dims input_dims {N, H, W, IC};
        sc_dims filter_dims {R, S, IC, OC};
        sc_dims output_dims {N, H - 2, W - 2, OC};
        auto ins0 = graph_tensor::make(
                input_dims, sc_data_format_t(format_kinds::ABCD));
        auto ins1 = graph_tensor::make(
                filter_dims, sc_data_format_t(format_kinds::ABCD));
        auto ins2 = graph_tensor::make({OC});
        auto out0 = graph_tensor::make(output_dims);

        std::unordered_map<std::string, any_t> attrs = {
                {"strides", sc_dims {1, 1}}, {"auto_pad", "VALID"},
                {"pads_begin", sc_dims {0, 0}}, {"pads_end", sc_dims {0, 0}}};

        sc_graph_t graph;
        auto in = graph.make_input({ins0, ins1, ins2});
        auto conv = graph.make(
                "conv_fwd", in->get_outputs(), {out0}, any_map_t(attrs));
        auto out = graph.make_output(conv->get_outputs());
        graph.attrs_["temp.fuse"] = 0;
        graph_driver(graph, get_test_ctx());

        for (auto op : graph.ops_) {
            if (op->op_name_ == "conv_fwd_core") {
                EXPECT_TRUE(op->attrs_.has_key("pads_begin"));
                EXPECT_EQ(
                        op->attrs_.get<sc_dims>("pads_begin"), sc_dims({0, 0}));
                EXPECT_TRUE(op->attrs_.has_key("pads_end"));
                EXPECT_EQ(op->attrs_.get<sc_dims>("pads_end"), sc_dims({0, 0}));
            }
        }
    }
    {
        int N = 64, IC = 16, OC = 64, H = 32, W = 32, R = 3, S = 3;
        sc_dims input_dims {N, H, W, IC};
        sc_dims filter_dims {R, S, IC, OC};
        sc_dims output_dims {N, H, W, OC};
        auto ins0 = graph_tensor::make(
                input_dims, sc_data_format_t(format_kinds::ABCD));
        auto ins1 = graph_tensor::make(
                filter_dims, sc_data_format_t(format_kinds::ABCD));
        auto ins2 = graph_tensor::make({OC});
        auto out0 = graph_tensor::make(output_dims);

        std::unordered_map<std::string, any_t> attrs = {
                {"strides", sc_dims {1, 1}}, {"auto_pad", "SAME_UPPER"},
                {"pads_begin", sc_dims {0, 0}}, {"pads_end", sc_dims {0, 0}}};

        sc_graph_t graph;
        auto in = graph.make_input({ins0, ins1, ins2});
        auto conv = graph.make(
                "conv_fwd", in->get_outputs(), {out0}, any_map_t(attrs));
        auto out = graph.make_output(conv->get_outputs());
        graph.attrs_["temp.fuse"] = 0;
        graph_driver(graph, get_test_ctx());

        for (auto op : graph.ops_) {
            if (op->op_name_ == "conv_fwd_core") {
                EXPECT_TRUE(op->attrs_.has_key("pads_begin"));
                EXPECT_EQ(
                        op->attrs_.get<sc_dims>("pads_begin"), sc_dims({1, 1}));
                EXPECT_TRUE(op->attrs_.has_key("pads_end"));
                EXPECT_EQ(op->attrs_.get<sc_dims>("pads_end"), sc_dims({1, 1}));
            }
        }
    }
}

TEST(GCCore_graph_conv_test, TestGraphConvolutionAutoPadSameUpperStride2x2) {
    int N = 1, IC = 1024, OC = 2048, IH = 13, IW = 13, OH = 7, OW = 7, R = 5,
        S = 5;
    sc_dims input_dims {N, IH, IW, IC};
    sc_dims filter_dims {R, S, IC, OC};
    sc_dims output_dims {N, OH, OW, OC};
    auto ins0 = graph_tensor::make(
            input_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins1 = graph_tensor::make(
            filter_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins2 = graph_tensor::make({OC});
    auto out0 = graph_tensor::make(output_dims);

    std::unordered_map<std::string, any_t> attrs = {{"strides", sc_dims {2, 2}},
            {"auto_pad", "SAME_UPPER"}, {"pads_begin", sc_dims {0, 0}},
            {"pads_end", sc_dims {0, 0}}};

    sc_graph_t graph;
    auto in = graph.make_input({ins0, ins1, ins2});
    auto conv = graph.make(
            "conv_fwd", in->get_outputs(), {out0}, any_map_t(attrs));
    auto out = graph.make_output(conv->get_outputs());
    graph.attrs_["temp.fuse"] = 0;
    graph_driver(graph, get_test_ctx());

    for (const auto &op : graph.ops_) {
        if (op->op_name_ == "conv_fwd_core") {
            EXPECT_TRUE(op->attrs_.has_key("pads_begin"));
            EXPECT_EQ(op->attrs_.get<sc_dims>("pads_begin"), sc_dims({2, 2}));
            EXPECT_TRUE(op->attrs_.has_key("pads_end"));
            EXPECT_EQ(op->attrs_.get<sc_dims>("pads_end"), sc_dims({2, 2}));
        }
    }
}

TEST(GCCore_graph_conv_test, TestGraphConvolutionBwdData) {
    int N = 64, IC = 16, OC = 64, H = 32, W = 32, R = 1, S = 1;
    sc_dims output_delta_dims {N, H, W, OC};
    sc_dims filter_dims {R, S, IC, OC};
    sc_dims input_delta_dims {N, H, W, IC};
    auto ins0 = graph_tensor::make(
            output_delta_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins1 = graph_tensor::make(
            filter_dims, sc_data_format_t(format_kinds::ABCD));

    std::unordered_map<std::string, any_t> attrs = {{"strides", sc_dims {1, 1}},
            {"pads_begin", sc_dims {0, 0}}, {"pads_end", sc_dims {0, 0}},
            {"dst_shape", input_delta_dims}};

    sc_graph_t graph;
    auto in = graph.make_input({ins0, ins1});
    auto conv = graph.make(
            "conv_bwd_data", in->get_outputs(), {}, any_map_t(attrs));
    auto out = graph.make_output(conv->get_outputs());
    graph_inline(graph);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected_graph
            = R"(graph(v0: f32[64, 32, 32, 64], v1: f32[1, 1, 16, 64]) -> [v2: f32[64, 32, 32, 16]] {
  [v3: f32[64, 16, 1, 1]] = transpose(v1)
  [v4: f32[64, 64, 32, 32]] = transpose(v0)
  [v5: f32[64, 16, 32, 32]] = conv_bwd_data_core(v4, v3)
  [v2: f32[64, 32, 32, 16]] = transpose(v5)
}
)";
    EXPECT_EQ(ss.str(), expected_graph);

    for (auto op : graph.ops_) {
        if (op->op_name_ == "conv_bwd_data_core") {
            sc_dims out_shape = op->attrs_.get<sc_dims>("dst_shape");
            sc_dims expected_out_shape {N, IC, H, W};
            EXPECT_EQ(out_shape, expected_out_shape);
            break;
        }
    }
}

TEST(GCCore_graph_conv_test, TestGraphConvolutionBwdWeight) {
    int N = 64, IC = 16, OC = 64, H = 32, W = 32, R = 1, S = 1;
    sc_dims output_delta_dims {N, H, W, OC};
    sc_dims filter_dims {R, S, IC, OC};
    sc_dims input_dims {N, H, W, IC};
    auto ins0 = graph_tensor::make(
            input_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins1 = graph_tensor::make(
            output_delta_dims, sc_data_format_t(format_kinds::ABCD));

    std::unordered_map<std::string, any_t> attrs = {{"strides", sc_dims {1, 1}},
            {"pads_begin", sc_dims {0, 0}}, {"pads_end", sc_dims {0, 0}},
            {"weights_shape", filter_dims}};

    sc_graph_t graph;
    auto in = graph.make_input({ins0, ins1});
    auto conv = graph.make(
            "conv_bwd_weight", in->get_outputs(), {}, any_map_t(attrs));
    auto out = graph.make_output(conv->get_outputs());
    graph_inline(graph);
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected_graph
            = R"(graph(v0: f32[64, 32, 32, 16], v1: f32[64, 32, 32, 64]) -> [v2: f32[1, 1, 16, 64]] {
  [v3: f32[64, 64, 32, 32]] = transpose(v1)
  [v4: f32[64, 16, 32, 32]] = transpose(v0)
  [v5: f32[64, 16, 1, 1]] = conv_bwd_weight_core(v4, v3)
  [v2: f32[1, 1, 16, 64]] = transpose(v5)
}
)";
    EXPECT_EQ(ss.str(), expected_graph);

    for (auto op : graph.ops_) {
        if (op->op_name_ == "conv_bwd_data_core") {
            sc_dims filter_shape = op->attrs_.get<sc_dims>("weights_shape");
            sc_dims expected_filter_shape {OC, IC, R, S};
            EXPECT_EQ(filter_shape, expected_filter_shape);
            break;
        }
    }
}
