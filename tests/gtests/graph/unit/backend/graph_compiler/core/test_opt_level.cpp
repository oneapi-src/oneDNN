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
#include <fstream>
#include <iostream>
#include <utility>
#include "context.hpp"
#include "test_graph.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/jit/jit.hpp>

using namespace std;

static sc_graph_t get_quantized_matmul_graph(const sc_dims &data_dims,
        const sc_dims &weight_dims, const sc_dims &gamma_dims,
        const sc_dims &beta_dims) {
    sc_graph_t graph;
    auto data = graph.make_input(
            {graph_tensor::make(data_dims, sc_data_format_t(), datatypes::u8)});
    auto weight = graph.make_input({graph_tensor::make(
            weight_dims, sc_data_format_t(), datatypes::s8)});
    data = graph.make("dequantize", data->get_outputs(), {},
            {{"dtype", datatypes::f32}, {"scales", std::vector<float> {1.2f}},
                    {"zero_points", std::vector<int> {0}}});
    weight = graph.make("dequantize", weight->get_outputs(), {},
            {{"dtype", datatypes::f32}, {"scales", std::vector<float> {1.2f}},
                    {"zero_points", std::vector<int> {0}}});
    auto mm = graph.make("matmul",
            {data->get_outputs()[0], weight->get_outputs()[0]}, {}, {});
    auto gelu = graph.make("gelu", mm->get_outputs(), {}, {});
    auto gamma = graph.make_input({graph_tensor::make(gamma_dims)});
    auto beta = graph.make_input({graph_tensor::make(beta_dims)});
    auto lnorm = graph.make("layernorm",
            {gelu->get_outputs()[0], gamma->get_outputs()[0],
                    beta->get_outputs()[0]},
            {},
            {{"use_affine", true}, {"begin_norm_axis", 1}, {"epsilon", 1e-5f},
                    {"keep_stats", false}});
    auto output = graph.make("quantize", lnorm->get_outputs(), {},
            {{"dtype", datatypes::u8}, {"scales", std::vector<float> {1.3f}},
                    {"zero_points", std::vector<int> {5}}});
    output = graph.make_output(output->get_outputs());
    return graph;
}

TEST(GCCore_CPU_opt_level_cpp, TestCompilerOptLevel) {
    REQUIRE_VNNI();
    SKIP_AMX();
    sc_dims data_dims = {16, 64}, weight_dims = {64, 256}, gamma_dims = {256},
            beta_dims = {256}, out_dims = {16, 256};
    test_buffer<uint8_t> data(test_utils::product(data_dims));
    test_buffer<int8_t> weight(test_utils::product(weight_dims));
    test_buffer<float> gamma(test_utils::product(gamma_dims));
    test_buffer<float> beta(test_utils::product(beta_dims));
    test_utils::fill_data<uint8_t>(data.get(), test_utils::product(data_dims));
    test_utils::fill_data<int8_t>(
            weight.get(), test_utils::product(weight_dims));
    test_utils::fill_data<float>(gamma.get(), test_utils::product(gamma_dims));
    test_utils::fill_data<float>(beta.get(), test_utils::product(beta_dims));
    std::vector<generic_val> gargs;
    gargs.emplace_back((void *)data.get());
    gargs.emplace_back((void *)weight.get());
    gargs.emplace_back((void *)gamma.get());
    gargs.emplace_back((void *)beta.get());
    gargs.emplace_back((void *)nullptr);
    test_buffer<uint8_t> output_lv0(test_utils::product(out_dims));
    test_buffer<uint8_t> output_lv1(test_utils::product(out_dims));
    test_buffer<uint8_t> output_lv2(test_utils::product(out_dims));
    test_buffer<uint8_t> output_lv3(test_utils::product(out_dims));
    {
        // opt level 0
        auto graph = get_quantized_matmul_graph(
                data_dims, weight_dims, gamma_dims, beta_dims);
        auto temp_ctx = std::make_shared<context_t>(*get_test_ctx());
        temp_ctx->flags_.opt_level_ = sc_opt_level::lv0;
        temp_ctx->flags_.mixed_fusion_ = true;
        graph_driver(graph, temp_ctx);
        std::stringstream ss;
        print_graph(graph, ss, true);
        constexpr const char *expected_0
                = R"(graph(v0: u8[16, 64], v1: s8[64, 256], v2: f32[256], v3: f32[256]) -> [v4: u8[16, 256]] {
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = constant([1])
  [v8: f32[1]] = constant([1])
  [v9: f32[1]] = constant([1])
  [v10: f32[1]] = constant([1])
  [v11: f32[1]] = constant([1])
  [v12: f32[1]] = constant([1])
  [v13: f32[1]] = constant([1])
  [v14: f32[1]] = constant([1])
  [v15: f32[1, 256]] = tensor_view(v3)
  [v16: f32[1, 256]] = tensor_view(v2)
  [v17: s8[4, 4, 4, 64, 4]] = reorder(v1)
  [v18: s32[16, 256]] = quantized_managed_matmul_core(v0, v17)
  [v19: f32[16, 256]] = cast(v18)
  [v20: f32[16, 256]] = mul(v19, v7)
  [v21: f32[16, 256]] = cast(v18)
  [v22: f32[16, 256]] = mul(v21, v8)
  [v23: f32[16, 256]] = mul(v22, v13)
  [v24: f32[16, 256]] = erf(v23)
  [v25: f32[16, 256]] = add(v24, v12)
  [v26: f32[16, 256]] = mul(v25, v20)
  [v27: f32[16, 256]] = mul(v26, v11)
  [v28: f32[16, 1]] = reduce(v27)
  [v29: f32[16, 1]] = div(v28, v10)
  [v30: f32[16, 256]] = sub(v27, v29)
  [v31: f32[16, 1]] = mul(v29, v29)
  [v32: f32[16, 256]] = mul(v27, v27)
  [v33: f32[16, 1]] = reduce(v32)
  [v34: f32[16, 1]] = div(v33, v9)
  [v35: f32[16, 1]] = sub(v34, v31)
  [v36: f32[16, 1]] = add(v35, v14)
  [v37: f32[16, 1]] = squared_root(v36)
  [v38: f32[16, 256]] = mul(v30, v37)
  [v39: f32[16, 256]] = mul(v38, v16)
  [v40: f32[16, 256]] = add(v39, v15)
  [v41: f32[16, 256]] = mul(v40, v6)
  [v42: f32[16, 256]] = add(v41, v5)
  [v4: u8[16, 256]] = cast(v42)
}
)";
        EXPECT_EQ(ss.str(), expected_0);
        std::vector<sc_op_ptr> op_args = graph.get_input_ops();
        const std::vector<sc_op_ptr> &out_args = graph.get_output_ops();
        op_args.insert(op_args.end(), out_args.begin(), out_args.end());
        auto ir_mod = lower_graph(temp_ctx, graph, op_args);
        auto jitf = jit_engine_t::make(temp_ctx)->get_entry_func(ir_mod, true);
        gargs.back() = (void *)output_lv0.data();
        jitf->call_generic_default(gargs.data());
    }
    {
        // opt level 1
        auto graph = get_quantized_matmul_graph(
                data_dims, weight_dims, gamma_dims, beta_dims);
        auto temp_ctx = std::make_shared<context_t>(*get_test_ctx());
        temp_ctx->flags_.opt_level_ = sc_opt_level::lv1;
        temp_ctx->flags_.mixed_fusion_ = true;
        graph_driver(graph, temp_ctx);
        std::stringstream ss;
        print_graph(graph, ss, true);
        constexpr const char *expected_1
                = R"(graph(v0: u8[16, 64], v1: s8[64, 256], v2: f32[256], v3: f32[256]) -> [v4: u8[16, 256]] {
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = constant([1])
  [v8: f32[1]] = constant([1])
  [v9: f32[1]] = constant([1])
  [v10: f32[1]] = constant([1])
  [v11: f32[1]] = constant([1])
  [v12: f32[1]] = constant([1])
  [v13: f32[1]] = constant([1])
  [v14: f32[1]] = constant([1])
  [v15: f32[1, 256]] = tensor_view(v3)
  [v16: f32[1, 256]] = tensor_view(v2)
  [v17: s8[4, 4, 4, 64, 4]] = reorder(v1)
  [v18: s32[16, 256]] = quantized_managed_matmul_core(v0, v17)
  [v19: f32[16, 256]] = outerloop_16_partition_cast_mul(v18, v7)
  [v20: f32[16, 256]] = outerloop_16_partition_cast_mul(v18, v8)
  [v21: f32[16, 256]] = outerloop_16_partition_mul_erf_add_mul_mul(v20, v13, v12, v19, v11)
  [v22: f32[16, 1]] = outerloop_16_partition_reduce_compute_reduce_collect_div(v21, v10)
  [v23: f32[16, 256]] = mul(v21, v21)
  [v24: f32[16, 1]] = outerloop_16_partition_reduce_compute_reduce_collect_div(v23, v9)
  [v25: f32[16, 256]] = outerloop_16_partition_sub_mul_sub_add_squared_root_mul_mul_add(v21, v22, v24, v14, v16, v15)
  [v4: u8[16, 256]] = outerloop_16_partition_mul_add_cast(v25, v6, v5)
}
)";
        EXPECT_EQ(ss.str(), expected_1);
        std::vector<sc_op_ptr> op_args = graph.get_input_ops();
        const std::vector<sc_op_ptr> &out_args = graph.get_output_ops();
        op_args.insert(op_args.end(), out_args.begin(), out_args.end());
        auto ir_mod = lower_graph(temp_ctx, graph, op_args);
        auto jitf = jit_engine_t::make(temp_ctx)->get_entry_func(ir_mod, true);
        gargs.back() = (void *)output_lv1.data();
        jitf->call_generic_default(gargs.data());
    }
    {
        // opt level 2
        auto graph = get_quantized_matmul_graph(
                data_dims, weight_dims, gamma_dims, beta_dims);
        auto temp_ctx = std::make_shared<context_t>(*get_test_ctx());
        temp_ctx->flags_.opt_level_ = sc_opt_level::lv2;
        temp_ctx->flags_.mixed_fusion_ = true;
        graph_driver(graph, temp_ctx);
        std::stringstream ss;
        print_graph(graph, ss, true);
        constexpr const char *expected_2
                = R"(graph(v0: u8[16, 64], v1: s8[64, 256], v2: f32[256], v3: f32[256]) -> [v4: u8[16, 256]] {
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = constant([1])
  [v8: f32[1]] = constant([1])
  [v9: f32[1]] = constant([1])
  [v10: f32[1]] = reciprocal(v9)
  [v11: f32[1]] = constant([1])
  [v12: f32[1]] = reciprocal(v11)
  [v13: f32[1]] = constant([1])
  [v14: f32[1]] = constant([1])
  [v15: f32[1]] = constant([1])
  [v16: f32[1]] = mul(v8, v15)
  [v17: f32[1]] = constant([1])
  [v18: f32[1, 256]] = tensor_view(v3)
  [v19: f32[1, 256]] = tensor_view(v2)
  [v4: u8[16, 256]] = outerloop_4X1X1X1X1_partition_reorder_quantized_managed_matmul_core_cast_mul_erf_add_mul_mul_mul_reduce_mul_sub_mul_mul_reduce_mul_sub_add_squared_root_mul_mul_add_mul_add_cast(v1, v0, v16, v14, v7, v13, v12, v10, v17, v19, v18, v6, v5)
}
)";
        EXPECT_EQ(ss.str(), expected_2);
        std::vector<sc_op_ptr> op_args = graph.get_input_ops();
        const std::vector<sc_op_ptr> &out_args = graph.get_output_ops();
        op_args.insert(op_args.end(), out_args.begin(), out_args.end());
        auto ir_mod = lower_graph(temp_ctx, graph, op_args);
        auto jitf = jit_engine_t::make(temp_ctx)->get_entry_func(ir_mod, true);
        gargs.back() = (void *)output_lv2.data();
        jitf->call_generic_default(gargs.data());
    }
    {
        // opt level 3
        auto graph = get_quantized_matmul_graph(
                data_dims, weight_dims, gamma_dims, beta_dims);
        auto temp_ctx = std::make_shared<context_t>(*get_test_ctx());
        temp_ctx->flags_.opt_level_ = sc_opt_level::lv3;
        temp_ctx->flags_.mixed_fusion_ = true;
        graph_driver(graph, temp_ctx);
        std::stringstream ss;
        print_graph(graph, ss, true);
        constexpr const char *expected_3
                = R"(graph(v0: u8[16, 64], v1: s8[64, 256], v2: f32[256], v3: f32[256]) -> [v4: u8[16, 256]] {
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = constant([1])
  [v8: f32[1]] = constant([1])
  [v9: f32[1]] = constant([1])
  [v10: f32[1]] = reciprocal(v9)
  [v11: f32[1]] = constant([1])
  [v12: f32[1]] = reciprocal(v11)
  [v13: f32[1]] = constant([1])
  [v14: f32[1]] = constant([1])
  [v15: f32[1]] = constant([1])
  [v16: f32[1]] = mul(v8, v15)
  [v17: f32[1]] = constant([1])
  [v18: f32[1, 256]] = tensor_view(v3)
  [v19: f32[1, 256]] = tensor_view(v2)
  [v4: u8[16, 256]] = outerloop_4X1X1X1X1_partition_reorder_quantized_managed_matmul_core_cast_mul_erf_add_mul_mul_mul_reduce_mul_sub_mul_mul_reduce_mul_sub_add_squared_root_mul_mul_add_mul_add_cast(v1, v0, v16, v14, v7, v13, v12, v10, v17, v19, v18, v6, v5)
}
)";
        EXPECT_EQ(ss.str(), expected_3);
        std::vector<sc_op_ptr> op_args = graph.get_input_ops();
        const std::vector<sc_op_ptr> &out_args = graph.get_output_ops();
        op_args.insert(op_args.end(), out_args.begin(), out_args.end());
        auto ir_mod = lower_graph(temp_ctx, graph, op_args);
        auto jitf = jit_engine_t::make(temp_ctx)->get_entry_func(ir_mod, true);
        gargs.back() = (void *)output_lv3.data();
        jitf->call_generic_default(gargs.data());
    }
    test_utils::compare_data(output_lv0.data(), output_lv1.data(),
            test_utils::product(out_dims), 1e-4f, 1.0f);
    test_utils::compare_data(output_lv0.data(), output_lv2.data(),
            test_utils::product(out_dims), 1e-4f, 1.0f);
    test_utils::compare_data(output_lv0.data(), output_lv3.data(),
            test_utils::product(out_dims), 1e-4f, 1.0f);
}

TEST(GCCore_CPU_opt_level_cpp, TestCompilerOptLevel2) {
    SET_THREADS_OR_SKIP(28);
    {
        // opt level 0
        auto graph = get_parallel_merge_mlp_graph();
        auto temp_ctx = std::make_shared<context_t>(*get_test_ctx());
        temp_ctx->flags_.opt_level_ = sc_opt_level::lv0;
        temp_ctx->flags_.mixed_fusion_ = true;
        graph_driver(graph, temp_ctx);
        std::stringstream ss;
        print_graph(graph, ss, true);
        constexpr const char *expected_0
                = R"(graph(v0: f32[10752, 1024], v1: f32[1024, 1024], v2: f32[1024, 1024], v3: f32[1024, 1024]) -> [v4: f32[10752, 1024]] {
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = constant([1])
  [v8: f32[168, 16, 64, 64]] = managed_matmul_core(v0, v1)
  [v9: f32[10752, 1024]] = managed_matmul_core(v8, v2)
  [v10: f32[10752, 1024]] = mul(v9, v7)
  [v11: f32[10752, 1024]] = erf(v10)
  [v12: f32[10752, 1024]] = add(v11, v6)
  [v13: f32[10752, 1024]] = mul(v12, v9)
  [v14: f32[10752, 1024]] = mul(v13, v5)
  [v15: f32[10752, 1024]] = managed_matmul_core(v14, v3)
  [v4: f32[10752, 1024]] = add(v14, v15)
}
)";
        EXPECT_EQ(ss.str(), expected_0);
    }
    {
        // opt level 1
        auto graph = get_parallel_merge_mlp_graph();
        auto temp_ctx = std::make_shared<context_t>(*get_test_ctx());
        temp_ctx->flags_.opt_level_ = sc_opt_level::lv1;
        temp_ctx->flags_.mixed_fusion_ = true;
        graph_driver(graph, temp_ctx);
        std::stringstream ss;
        print_graph(graph, ss, true);
        constexpr const char *expected_1
                = R"(graph(v0: f32[10752, 1024], v1: f32[1024, 1024], v2: f32[1024, 1024], v3: f32[1024, 1024]) -> [v4: f32[10752, 1024]] {
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = constant([1])
  [v8: f32[168, 16, 64, 64]] = managed_matmul_core(v0, v1)
  [v9: f32[10752, 1024]] = managed_matmul_core(v8, v2)
  [v10: f32[10752, 1024]] = outerloop_10752_partition_mul_erf_add_mul_mul(v9, v7, v6, v5)
  [v4: f32[10752, 1024]] = outerloop_14X2X1X1X1_partition_managed_matmul_core_add(v10, v3)
}
)";
        EXPECT_EQ(ss.str(), expected_1);
    }
    {
        // opt level 2
        auto graph = get_parallel_merge_mlp_graph();
        auto temp_ctx = std::make_shared<context_t>(*get_test_ctx());
        temp_ctx->flags_.opt_level_ = sc_opt_level::lv2;
        temp_ctx->flags_.mixed_fusion_ = true;
        graph_driver(graph, temp_ctx);
        std::stringstream ss;
        print_graph(graph, ss, true);
        constexpr const char *expected_2
                = R"(graph(v0: f32[10752, 1024], v1: f32[1024, 1024], v2: f32[1024, 1024], v3: f32[1024, 1024]) -> [v4: f32[10752, 1024]] {
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = constant([1])
  [v8: f32[168, 16, 64, 64]] = managed_matmul_core(v0, v1)
  [v9: f32[10752, 1024]] = outerloop_14X2X1X1X1_partition_managed_matmul_core_mul_erf_add_mul_mul(v8, v2, v7, v6, v5)
  [v4: f32[10752, 1024]] = outerloop_14X2X1X1X1_partition_managed_matmul_core_add(v9, v3)
}
)";
        EXPECT_EQ(ss.str(), expected_2);
    }
    {
        // opt level 3
        auto graph = get_parallel_merge_mlp_graph();
        auto temp_ctx = std::make_shared<context_t>(*get_test_ctx());
        temp_ctx->flags_.opt_level_ = sc_opt_level::lv3;
        temp_ctx->flags_.mixed_fusion_ = true;
        graph_driver(graph, temp_ctx);
        std::stringstream ss;
        print_graph(graph, ss, true);
        constexpr const char *expected_3
                = R"(graph(v0: f32[10752, 1024], v1: f32[1024, 1024], v2: f32[1024, 1024], v3: f32[1024, 1024]) -> [v4: f32[10752, 1024]] {
  [v5: f32[1]] = constant([1])
  [v6: f32[1]] = constant([1])
  [v7: f32[1]] = constant([1])
  [v4: f32[10752, 1024]] = outerloop_14_partition_managed_matmul_core_managed_matmul_core_mul_erf_add_mul_mul_managed_matmul_core_add(v0, v1, v2, v7, v6, v5, v3)
}
)";
        EXPECT_EQ(ss.str(), expected_3);
    }
}
