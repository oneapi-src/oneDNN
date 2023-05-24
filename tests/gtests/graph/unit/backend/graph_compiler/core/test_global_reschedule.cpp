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
#include <sstream>
#include "compiler/ir/graph/fusible_op.hpp"
#include "context.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <ops/templates/matmul_core.hpp>
#include <util/reflection.hpp>

using namespace dnnl::impl::graph::gc;

/**
 * E.g.
 *       in_a  in_b
 *         \    /
 *        matmul2d   in_c
 *           |      /
 *          bias
 *           |
 *    in_d  quan
 *      \    |       \
 *        matmul2d   dequan
 *           |       /
 *          add
 *           |
 *         output
 *
 * the above graph will be transformed into:
 *
 *       in_a    in_b    in_c
 *         \      |       /
 *    matmul2d_bias_quan_reorder
 *                |          /
 *    in_d        |        /
 *       \        |      /
 *       matmul2d_dequan_add_reorder
 *                |
 *                |
 *              output
 * */
// fix-me(brgemm-fuse): recover the following tests when postop is fixed
#if 0
TEST(GCCore_CPU_graph_reshedule, TestGraphReschedule1) {
    auto get_test_graph = []() {
        sc_graph_t mgr;
        auto in_a = mgr.make_input({graph_tensor::make(
                {6, 32, 64}, sc_data_format_t(format_kinds::ABC))});
        auto in_b = mgr.make_input({graph_tensor::make(
                {6, 64, 32}, sc_data_format_t(format_kinds::ABC))});
        auto in_d = mgr.make_input({graph_tensor::make({6, 32, 32},
                sc_data_format_t(format_kinds::ABC), sc_data_type_t::s8())});

        auto gemm1 = mgr.make("matmul_core",
                {in_a->get_outputs()[0], in_b->get_outputs()[0]},
                {graph_tensor::make(
                        {6, 32, 32}, sc_data_format_t(format_kinds::ABC))},
                {});

        auto quan = mgr.make("quantize", gemm1->get_outputs(),
                {graph_tensor::make({6, 32, 32},
                        sc_data_format_t(format_kinds::ABC), datatypes::s8)},
                {{"dtype", datatypes::s8},
                        {"scales", std::vector<float> {1.1f}},
                        {"zero_points", std::vector<int> {0}},
                        {"channel_axis", 0}});

        auto dequan = mgr.make("dequantize", quan->get_outputs(),
                {graph_tensor::make(
                        {6, 32, 32}, sc_data_format_t(format_kinds::ABC))},
                {{"dtype", datatypes::f32},
                        {"scales", std::vector<float> {1.1f}},
                        {"zero_points", std::vector<int> {0}},
                        {"channel_axis", 0}});

        auto gemm2 = mgr.make("matmul_core",
                {quan->get_outputs()[0], in_d->get_outputs()[0]},
                {graph_tensor::make({6, 32, 32},
                        sc_data_format_t(format_kinds::ABC), datatypes::s32)},
                {});

        auto cast = mgr.make("cast", gemm2->get_outputs(), {},
                {{"dtype", sc_data_type_t::f32()}});

        auto add_out = mgr.make("add",
                {cast->get_outputs()[0], dequan->get_outputs()[0]}, {}, {});

        mgr.make_output(add_out->get_outputs());
        return mgr;
    };

    {
        sc_graph_t graph = get_test_graph();
        graph_driver(graph, get_test_ctx());

        std::stringstream ss;
        print_graph(graph, ss, true);

        std::string expected_str1
                = R"(graph(v0: f32[6, 32, 64], v1: f32[6, 64, 32], v2: s8[6, 32, 32]) -> [v3: f32[6, 32, 32]] {
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: s8[6, 1, 1, 8, 32, 4]] = reorder(v2)
  [v7: s8[6, 32, 32]] = matmul_core_mul_cast(v0, v1, v5)
  [v3: f32[6, 32, 32]] = matmul_core_cast_mul_cast_add(v7, v6, v7, v4)
}
)";
        std::string expected_str2
                = R"(graph(v0: f32[6, 32, 64], v1: f32[6, 64, 32], v2: s8[6, 32, 32]) -> [v3: f32[6, 32, 32]] {
  [v4: f32[1, 1, 32]] = constant([1, 1, 32])
  [v5: f32[1]] = constant([1])
  [v6: s8[6, 1, 1, 8, 32, 4]] = reorder(v2)
  [v7: s8[6, 32, 32]] = matmul_core_mul_cast(v0, v1, v4)
  [v3: f32[6, 32, 32]] = matmul_core_cast_mul_cast_add(v7, v6, v7, v5)
}
)";
        if (IS_AMX_AVAILABLE()) {
            EXPECT_EQ(ss.str(), expected_str1);
        } else {
            EXPECT_EQ(ss.str(), expected_str1);
        }
    }
}
#endif

TEST(GCCore_CPU_graph_reshedule, TestGraphReschedule2) {
    SET_THREADS_OR_SKIP(8);

    auto get_test_graph = []() {
        sc_graph_t mgr;
        auto in_a = mgr.make_input({graph_tensor::make(
                {6, 32, 64}, sc_data_format_t(format_kinds::ABC))});
        auto in_b = mgr.make_input({graph_tensor::make(
                {6, 64, 32}, sc_data_format_t(format_kinds::ABC))});
        auto in_c = mgr.make_input({graph_tensor::make(
                {6, 32, 32}, sc_data_format_t(format_kinds::ABC))});

        auto c_1 = mgr.make("constant", {}, {graph_tensor::make({1})},
                {{"values",
                         std::make_shared<static_data_t>(
                                 std::vector<float> {1.0f})},
                        {"dtype", datatypes::f32},
                        {"plain_dims", sc_dims {1}}});

        auto c_2 = mgr.make("constant", {}, {graph_tensor::make({1})},
                {{"values",
                         std::make_shared<static_data_t>(
                                 std::vector<float> {1.5f})},
                        {"dtype", datatypes::f32},
                        {"plain_dims", sc_dims {1}}});

        auto gemm1 = mgr.make("matmul_core",
                {in_a->get_outputs()[0], in_b->get_outputs()[0]},
                {graph_tensor::make(
                        {6, 32, 32}, sc_data_format_t(format_kinds::ABC))},
                {});

        auto add_1 = mgr.make("add",
                {gemm1->get_outputs()[0], c_1->get_outputs()[0]}, {}, {});
        auto mul_1 = mgr.make("mul",
                {add_1->get_outputs()[0], c_2->get_outputs()[0]}, {}, {});
        auto sub_1 = mgr.make("sub",
                {mul_1->get_outputs()[0], c_2->get_outputs()[0]}, {}, {});
        auto gemm2 = mgr.make("matmul_core",
                {add_1->get_outputs()[0], in_c->get_outputs()[0]},
                {graph_tensor::make(
                        {6, 32, 32}, sc_data_format_t(format_kinds::ABC))},
                {});

        auto add_out = mgr.make("add",
                {gemm2->get_outputs()[0], sub_1->get_outputs()[0]}, {}, {});

        mgr.make_output(add_out->get_outputs());
        return mgr;
    };

    {
        sc_graph_t graph = get_test_graph();
        graph_driver(graph, get_test_ctx());

        std::stringstream ss;
        print_graph(graph, ss, true);
        std::string expected_str
                = R"(graph(v0: f32[6, 32, 64], v1: f32[6, 64, 32], v2: f32[6, 32, 32]) -> [v3: f32[6, 32, 32]] {
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: f32[6, 32, 32]] = matmul_core_add(v0, v1, v5)
  [v3: f32[6, 32, 32]] = matmul_core_mul_sub_add(v6, v2, v6, v4)
}
)";
        EXPECT_EQ(ss.str(), expected_str);
    }
}

TEST(GCCore_CPU_graph_reshedule, TestGraphReschedule3) {
    SET_THREADS_OR_SKIP(8);

    auto get_test_graph = []() {
        sc_graph_t mgr;
        auto in_a = mgr.make_input({graph_tensor::make(
                {6, 32, 64}, sc_data_format_t(format_kinds::ABC))});
        auto in_b = mgr.make_input({graph_tensor::make(
                {6, 64, 32}, sc_data_format_t(format_kinds::ABC))});
        auto in_c = mgr.make_input({graph_tensor::make(
                {6, 32, 32}, sc_data_format_t(format_kinds::ABC))});

        auto c_1 = mgr.make("constant", {}, {graph_tensor::make({1})},
                {{"values",
                         std::make_shared<static_data_t>(
                                 std::vector<float> {1.0f})},
                        {"dtype", datatypes::f32},
                        {"plain_dims", sc_dims {1}}});

        auto c_2 = mgr.make("constant", {}, {graph_tensor::make({1})},
                {{"values",
                         std::make_shared<static_data_t>(
                                 std::vector<float> {1.5f})},
                        {"dtype", datatypes::f32},
                        {"plain_dims", sc_dims {1}}});

        auto c_3 = mgr.make("constant", {}, {graph_tensor::make({1})},
                {{"values",
                         std::make_shared<static_data_t>(
                                 std::vector<float> {1.0f})},
                        {"dtype", datatypes::f32},
                        {"plain_dims", sc_dims {1}}});

        auto gemm1 = mgr.make("matmul_core",
                {in_a->get_outputs()[0], in_b->get_outputs()[0]},
                {graph_tensor::make(
                        {6, 32, 32}, sc_data_format_t(format_kinds::ABC))},
                {});

        auto add_1 = mgr.make("add",
                {gemm1->get_outputs()[0], c_1->get_outputs()[0]}, {}, {});
        auto mul_1 = mgr.make("mul",
                {add_1->get_outputs()[0], c_2->get_outputs()[0]}, {}, {});
        auto rd_1 = mgr.make("reduce", {mul_1->get_outputs()[0]}, {},
                {{"rd_axis", std::vector<int> {2}}, {"rd_op", 0},
                        {"keep_dims", true}});

        auto sub_1 = mgr.make(
                "sub", {rd_1->get_outputs()[0], c_3->get_outputs()[0]}, {}, {});

        auto gemm2 = mgr.make("matmul_core",
                {add_1->get_outputs()[0], in_c->get_outputs()[0]},
                {graph_tensor::make(
                        {6, 32, 32}, sc_data_format_t(format_kinds::ABC))},
                {});

        auto add_out = mgr.make("add",
                {gemm2->get_outputs()[0], sub_1->get_outputs()[0]}, {}, {});

        mgr.make_output(add_out->get_outputs());
        return mgr;
    };

    {
        sc_graph_t graph = get_test_graph();
        graph_driver(graph, get_test_ctx());

        std::stringstream ss;
        print_graph(graph, ss, true);
        std::string expected_str
                = R"(graph(v0: f32[6, 32, 64], v1: f32[6, 64, 32], v2: f32[6, 32, 32]) -> [v3: f32[6, 32, 32]] {
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[6, 32, 32], v8: f32[6, 32, 1]] = matmul_core_add_mul_reduce_sub(v0, v1, v6, v5, v4)
  [v3: f32[6, 32, 32]] = matmul_core_add(v7, v2, v8)
}
)";
        EXPECT_EQ(ss.str(), expected_str);
    }
}

TEST(GCCore_CPU_graph_reshedule, TestGraphReschedule4) {
    SET_THREADS_OR_SKIP(8);

    auto get_test_graph = []() {
        sc_graph_t mgr;
        auto in_a = mgr.make_input({graph_tensor::make(
                {6, 32, 64}, sc_data_format_t(format_kinds::ABC))});
        auto in_b = mgr.make_input({graph_tensor::make(
                {6, 64, 32}, sc_data_format_t(format_kinds::ABC))});
        auto in_c = mgr.make_input({graph_tensor::make(
                {6, 32, 32}, sc_data_format_t(format_kinds::ABC))});

        auto c_1 = mgr.make("constant", {}, {graph_tensor::make({1})},
                {{"values",
                         std::make_shared<static_data_t>(
                                 std::vector<float> {1.0f})},
                        {"dtype", datatypes::f32},
                        {"plain_dims", sc_dims {1}}});

        auto c_2 = mgr.make("constant", {}, {graph_tensor::make({1})},
                {{"values",
                         std::make_shared<static_data_t>(
                                 std::vector<float> {1.5f})},
                        {"dtype", datatypes::f32},
                        {"plain_dims", sc_dims {1}}});

        auto gemm1 = mgr.make("matmul_core",
                {in_a->get_outputs()[0], in_b->get_outputs()[0]},
                {graph_tensor::make(
                        {6, 32, 32}, sc_data_format_t(format_kinds::ABC))},
                {});

        auto add_1 = mgr.make("add",
                {gemm1->get_outputs()[0], c_1->get_outputs()[0]}, {}, {});
        auto mul_1 = mgr.make("mul",
                {add_1->get_outputs()[0], c_2->get_outputs()[0]}, {}, {});

        auto sub_1 = mgr.make("sub",
                {mul_1->get_outputs()[0], c_2->get_outputs()[0]}, {}, {});

        auto gemm2 = mgr.make("matmul_core",
                {add_1->get_outputs()[0], in_c->get_outputs()[0]},
                {graph_tensor::make(
                        {6, 32, 32}, sc_data_format_t(format_kinds::ABC))},
                {});

        ops::matmul_core_config_t cfg = {16, 16, 16};

        gemm2->stc_cast<tunable_op_t>()->set_config(
                reflection::general_object_t::make(cfg));

        auto add_out = mgr.make("add",
                {gemm2->get_outputs()[0], sub_1->get_outputs()[0]}, {}, {});

        mgr.make_output(add_out->get_outputs());
        return mgr;
    };

    {
        sc_graph_t graph = get_test_graph();
        graph_driver(graph, get_test_ctx());

        std::stringstream ss;
        print_graph(graph, ss, true);
        std::string expected_str
                = R"(graph(v0: f32[6, 32, 64], v1: f32[6, 64, 32], v2: f32[6, 32, 32]) -> [v3: f32[6, 32, 32]] {
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: f32[6, 32, 32]] = matmul_core_add(v0, v1, v5)
  [v3: f32[6, 32, 32]] = matmul_core_mul_sub_add(v6, v2, v6, v4)
}
)";
        EXPECT_EQ(ss.str(), expected_str);
    }
}

TEST(GCCore_CPU_graph_reshedule, TestGraphReschedule5) {
    SET_THREADS_OR_SKIP(8);

    auto get_test_graph = []() {
        sc_graph_t mgr;
        auto in_a = mgr.make_input({graph_tensor::make({4, 16, 1, 34},
                sc_data_format_t(format_kinds::ABCD), datatypes::u8)});
        auto in_b = mgr.make_input({graph_tensor::make(
                {4, 16, 34, 256}, sc_data_format_t(format_kinds::ABCD))});

        auto quant_b = mgr.make("quantize", in_b->get_outputs(), {},
                {{"dtype", datatypes::u8},
                        {"scales", std::vector<float> {1.1f}},
                        {"zero_points", std::vector<int> {2}},
                        {"channel_axis", 0}});

        auto dequant_a = mgr.make("dequantize", in_a->get_outputs(), {},
                {{"dtype", datatypes::f32},
                        {"scales", std::vector<float> {1.1f}},
                        {"zero_points", std::vector<int> {0}},
                        {"channel_axis", 0}});
        auto dequant_b = mgr.make("dequantize", quant_b->get_outputs(), {},
                {{"dtype", datatypes::f32},
                        {"scales", std::vector<float> {1.1f}},
                        {"zero_points", std::vector<int> {2}},
                        {"channel_axis", 0}});

        auto matmul = mgr.make("matmul",
                {dequant_a->get_outputs()[0], dequant_b->get_outputs()[0]}, {},
                {});

        auto quant_output = mgr.make("quantize", matmul->get_outputs(), {},
                {{"dtype", datatypes::u8},
                        {"scales", std::vector<float> {1.1f}},
                        {"zero_points", std::vector<int> {2}},
                        {"channel_axis", 0}});
        mgr.make_output(quant_output->get_outputs());
        return mgr;
    };

    {
        sc_graph_t graph = get_test_graph();
        auto ctx = std::make_shared<context_t>(*get_test_ctx());
        ctx->flags_.mixed_fusion_ = true;
        graph_driver(graph, ctx);

        std::stringstream ss;
        print_graph(graph, ss, true);
        std::string expected_str_spr
                = R"(graph(v0: u8[4, 16, 1, 34], v1: f32[4, 16, 34, 256]) -> [v2: u8[4, 16, 1, 256]] {
  [v3: s32[1]] = constant([1])
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = mul(v6, v5)
  [v8: f32[1]] = constant([1])
  [v9: f32[1]] = constant([1])
  [v10: u8[4, 16, 34, 256]] = outerloop_4X16X34_partition_mul_add_cast(v1, v9, v8)
  [v2: u8[4, 16, 1, 256]] = outerloop_4X16_partition_cast_reduce_compute_reduce_collect_mul_reorder_reorder_quantized_matmul_core_sub_cast_mul_add_cast(v10, v0, v3, v7, v4)
}
)";
        std::string expected_str_clx
                = R"(graph(v0: u8[4, 16, 1, 34], v1: f32[4, 16, 34, 256]) -> [v2: u8[4, 16, 1, 256]] {
  [v3: s32[1]] = constant([1])
  [v4: f32[1]] = constant([1])
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = mul(v6, v5)
  [v8: f32[1]] = constant([1])
  [v9: f32[1]] = constant([1])
  [v10: s8[4, 16, 34, 256]] = outerloop_4X16X34_partition_mul_add_cast(v1, v9, v8)
  [v2: u8[4, 16, 1, 256]] = outerloop_4X16_partition_cast_reduce_compute_reduce_collect_mul_reorder_reorder_quantized_matmul_core_sub_cast_mul_add_cast(v10, v0, v3, v7, v4)
}
)";
        if (IS_AMX_AVAILABLE()) {
            EXPECT_EQ(ss.str(), expected_str_spr);
        } else {
            EXPECT_EQ(ss.str(), expected_str_clx);
        }
    }
}
