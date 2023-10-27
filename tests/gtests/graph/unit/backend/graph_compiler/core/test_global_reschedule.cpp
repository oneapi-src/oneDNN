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
