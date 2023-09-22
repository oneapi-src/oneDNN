
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
#include "compiler/ir/graph/quantization/quantize_info.hpp"
#include "compiler/ir/graph/transform/transform.hpp"
#include "compiler/jit/jit.hpp"
#include "context.hpp"
#include "reference/conv_ref.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_graph_conv_test, TestGraphConvolutionWithBias) {
    REQUIRE_AVX2();
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
  [v3: f32[64, 64, 32, 32]] = outerloop_64X1X1_partition_conv_fwd_core_add_reorder(v6, v5, v4)
}
)";
    EXPECT_EQ(ss.str(), expected_graph);
}

TEST(GCCore_CPU_graph_conv_test, TestGraphConvolutionNXCXIO) {
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

TEST(GCCore_CPU_graph_conv_test, TestGraphConvolution7x1) {
    int N = 112, IC = 128, OC = 128, H = 12, W = 12, R = 7, S = 1;
    sc_dims input_dims {N, H, W, IC};
    sc_dims filter_dims {R, S, IC, OC};
    sc_dims output_dims {N, H, W, OC};
    auto ins0 = graph_tensor::make(
            input_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins1 = graph_tensor::make(
            filter_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins2 = graph_tensor::make({OC});

    std::unordered_map<std::string, any_t> attrs = {{"strides", sc_dims {1, 1}},
            {"pads_begin", sc_dims {3, 0}}, {"pads_end", sc_dims {3, 0}}};

    sc_graph_t graph;
    auto in = graph.make_input({ins0, ins1, ins2});
    auto conv = graph.make("conv_fwd", in->get_outputs(), {}, any_map_t(attrs));
    auto out = graph.make_output(conv->get_outputs());
    // The conv1d graph is different from the reference
    graph.attrs_["no_conv1d"] = true;
    graph_driver(graph, get_test_ctx());
}

TEST(GCCore_CPU_graph_conv_test, TestGraphConvolution1DFusion) {
    REQUIRE_AMX();
    SET_THREADS_OR_SKIP(112);
    int N = 128, IC = 64, OC = 64, H = 56, W = 56, R = 3, S = 3;
    any_map_t dqinfo = {
            {attr_keys::quan_dtype, datatypes::f32},
            {attr_keys::scales, std::vector<float>({1.f})},
            {attr_keys::per_channel, false},
            {attr_keys::channel_axis, 1},
            {attr_keys::zero_points, std::vector<int>({0})},
            {attr_keys::asymmetric, false},
    };
    any_map_t qinfo = {
            {attr_keys::quan_dtype, datatypes::s8},
            {attr_keys::scales, std::vector<float>({1.f})},
            {attr_keys::per_channel, false},
            {attr_keys::channel_axis, 1},
            {attr_keys::zero_points, std::vector<int>({0})},
            {attr_keys::asymmetric, false},
    };
    sc_graph_t g;
    // input
    sc_data_type_t src_dtype = datatypes::s8;
    sc_data_type_t weight_dtype = datatypes::s8;
    // input dequantize
    sc_op_ptr data_input = g.make_input({graph_tensor::make(
            {N, IC, H, W}, sc_data_format_t(format_kinds::ABCD), src_dtype)});
    sc_op_ptr weight_input1 = g.make_input({graph_tensor::make({OC, IC, 1, 1},
            sc_data_format_t(format_kinds::ABCD), weight_dtype)});
    weight_input1->attrs_.set("constant", const_kind::local_const);
    sc_op_ptr weight_input2 = g.make_input({graph_tensor::make({OC, IC, R, S},
            sc_data_format_t(format_kinds::ABCD), weight_dtype)});
    weight_input2->attrs_.set("constant", const_kind::local_const);
    auto deq_data1
            = g.make("dequantize", data_input->get_outputs(), {}, dqinfo);
    auto deq_weight1
            = g.make("dequantize", weight_input1->get_outputs(), {}, dqinfo);
    // conv relu quantize
    sc_dims paddings1 = {0, 0};
    sc_op_ptr conv_op1 = g.make("conv_fwd_core",
            {deq_data1->get_outputs()[0], deq_weight1->get_outputs()[0]}, {},
            {{"strides", sc_dims {1, 1}}, {"pads_begin", paddings1},
                    {"pads_end", paddings1}, {"data_format", "NCX"},
                    {"weights_format", "OIX"}});
    auto conv_relu_quan_out1
            = g.make("quantize", conv_op1->get_outputs(), {}, qinfo);

    auto deq_data2 = g.make(
            "dequantize", conv_relu_quan_out1->get_outputs(), {}, dqinfo);
    auto deq_weight2
            = g.make("dequantize", weight_input2->get_outputs(), {}, dqinfo);
    sc_dims paddings2 = {1, 1};
    // conv relu quantize
    sc_op_ptr conv_op2 = g.make("conv_fwd_core",
            {deq_data2->get_outputs()[0], deq_weight2->get_outputs()[0]}, {},
            {{"strides", sc_dims {1, 1}}, {"pads_begin", paddings2},
                    {"pads_end", paddings2}, {"data_format", "NCX"},
                    {"weights_format", "OIX"}});
    auto conv_relu_quan_out2
            = g.make("quantize", conv_op2->get_outputs(), {}, qinfo);
    const sc_op_ptr &final_out = conv_relu_quan_out2;

    sc_op_ptr out = g.make_output(final_out->get_outputs());
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = 1;
    graph_driver(g, ctx);
    std::stringstream ss;
    print_graph(g, ss, true, false, true, false);
    const char *expected_graph
            = R"(graph(v0: s8[128, 64, 56, 56], v1: s8[64, 64, 1, 1], v2: s8[64, 64, 3, 3]) -> [v3: s8[128, 64, 56, 56]] {
  [v4: s8[1, 1, 3, 3, 16, 64, 4]] = reorder(v2)
  [v5: s8[64, 64, 1]] = tensor_view(v1)
  [v6: s8[1, 1, 1, 16, 64, 4]] = reorder(v5)
  [v7: s8[128, 56, 56, 64]] = reorder(v0)
  [v8: s8[1, 401408, 64]] = tensor_view(v7)
  [v9: s8[128, 56, 56, 64]] = outerloop_1X112X1X1X1X1X1_partition_quantized_conv_fwd_core_tensor_view_cast_cast(v8, v6)
  [v3: s8[128, 64, 56, 56]] = outerloop_128_partition_padding_conv_fwd_core_cast_cast_reorder(v9, v4)
}
)";
    EXPECT_TRUE(ss.str() == expected_graph);
}

TEST(GCCore_CPU_graph_conv_test, TestGraphConvolutionWithDilation) {
    int N = 64, IC = 16, OC = 64, H = 32, W = 32, R = 3, S = 3, dilation = 2;
    sc_dims input_dims {N, H, W, IC};
    sc_dims filter_dims {R, S, IC, OC};
    sc_dims output_dims {N, H - 2 * dilation, W - 2 * dilation, OC};
    auto ins0 = graph_tensor::make(
            input_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins1 = graph_tensor::make(
            filter_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins2 = graph_tensor::make({OC});
    auto out0 = graph_tensor::make(output_dims);

    std::unordered_map<std::string, any_t> attrs = {{"strides", sc_dims {1, 1}},
            {"pads_begin", sc_dims {0, 0}}, {"pads_end", sc_dims {0, 0}},
            {"dilations", sc_dims {dilation, dilation}}};

    sc_graph_t graph;
    auto in = graph.make_input({ins0, ins1, ins2});
    auto conv = graph.make(
            "conv_fwd", in->get_outputs(), {out0}, any_map_t(attrs));
    auto out = graph.make_output(conv->get_outputs());
    graph_driver(graph, get_test_ctx());
    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected_graph_blocking =
            R"(graph(v0: f32[64, 32, 32, 16], v1: f32[3, 3, 16, 64], v2: f32[64]) -> [v3: f32[64, 28, 28, 64]] {
  [v4: f32[1, 1, 1, 1, 64]] = tensor_view(v2)
  [v5: f32[1, 1, 3, 3, 16, 64]] = tensor_view(v1)
  [v6: f32[64, 1, 32, 32, 16]] = tensor_view(v0)
  [v7: f32[64, 1, 28, 28, 64]] = conv_fwd_core_tensor_view_add(v6, v5, v4)
  [v3: f32[64, 28, 28, 64]] = tensor_view(v7)
}
)";
    const char *expected_graph_plain =
            R"(graph(v0: f32[64, 32, 32, 16], v1: f32[3, 3, 16, 64], v2: f32[64]) -> [v3: f32[64, 28, 28, 64]] {
  [v4: f32[1, 1, 1, 64]] = tensor_view(v2)
  [v5: f32[1, 1, 3, 3, 16, 64]] = tensor_view(v1)
  [v6: f32[64, 32, 32, 16]] = tensor_view(v0)
  [v3: f32[64, 28, 28, 64]] = conv_fwd_core_tensor_view_add(v6, v5, v4)
}
)";
    EXPECT_TRUE(ss.str() == expected_graph_plain
            || ss.str() == expected_graph_blocking);
}

TEST(GCCore_CPU_graph_conv_test, TestGraphConvolutionNXCXIODifferentHW) {
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

TEST(GCCore_CPU_graph_conv_test, TestGraphConvolutionAutoPad) {
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

TEST(GCCore_CPU_graph_conv_test,
        TestGraphConvolutionAutoPadSameUpperStride2x2) {
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

TEST(GCCore_CPU_graph_conv_test, TestGraphConvolutionBwdData) {
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

TEST(GCCore_CPU_graph_conv_test, TestGraphConvolutionBwdDataWithInverseWeight) {
    REQUIRE_AMX();
    int N = 8, IC = 64, OC = 64, H = 32, W = 32, R = 3, S = 3;
    auto ctx = get_test_ctx();
    sc_dims ginput_dims {N, H, W, IC};
    sc_dims filter_dims {R, S, OC, IC}; // bwd's semantic
    sc_dims goutput_dims {N, H, W, OC}; // valid_pad mode
    auto ins0 = graph_tensor::make(
            ginput_dims, sc_data_format_t(format_kinds::ABCD));
    auto ins1 = graph_tensor::make(
            filter_dims, sc_data_format_t(format_kinds::ABCD));

    std::unordered_map<std::string, any_t> attrs
            = {{"strides", sc_dims {1, 1}}, {"pads_begin", sc_dims {1, 1}},
                    {"pads_end", sc_dims {1, 1}}, {"dst_shape", goutput_dims}};

    sc_graph_t graph;
    auto in = graph.make_input({ins0, ins1});
    auto conv = graph.make(
            "conv_bwd_data", in->get_outputs(), {}, any_map_t(attrs));
    auto out = graph.make_output(conv->get_outputs());
    graph_driver(graph, ctx);

    std::stringstream ss;
    print_graph(graph, ss, true);
    const char *expected_graph
            = R"(graph(v0: f32[8, 32, 32, 64], v1: f32[3, 3, 64, 64]) -> [v2: f32[8, 32, 32, 64]] {
  [v3: f32[3, 3, 64, 64]] = tensor_view(v1)
  [v4: f32[1, 1, 3, 3, 64, 64]] = reorder(v3)
  [v5: f32[8, 32, 32, 64]] = tensor_view(v0)
  [v6: f32[8, 32, 32, 64]] = conv_fwd_core(v5, v4)
  [v2: f32[8, 32, 32, 64]] = tensor_view(v6)
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
    auto f = lower_graph(ctx, graph, {in, out});
    ASSERT_TRUE(f);
}

TEST(GCCore_CPU_graph_conv_test, TestGraphConvolutionBwdWeight) {
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
