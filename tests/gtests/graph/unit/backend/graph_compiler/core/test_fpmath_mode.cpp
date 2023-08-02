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
#include "context.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/jit/jit.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_fpmath_mode_cpp, TestAddCast2bf16) {
    REQUIRE_BF16();
    sc_dims input_dims = {100, 200};
    const auto input_size = test_utils::product(input_dims);
    test_buffer<float> input_data0(input_size);
    test_buffer<float> input_data1(input_size);
    test_buffer<float> input_data2(input_size);
    test_utils::fill_data(&input_data0[0], input_size);
    test_utils::fill_data(&input_data1[0], input_size);
    test_utils::fill_data(&input_data2[0], input_size);
    test_buffer<float> ref_output(input_size);
    test_buffer<float> sc_output(input_size);

    auto get_graph = [input_dims]() {
        sc_graph_t g;

        auto in0 = g.make_input({graph_tensor::make(input_dims)})
                           ->get_outputs()[0];
        auto in1 = g.make_input({graph_tensor::make(input_dims)})
                           ->get_outputs()[0];
        auto in2 = g.make_input({graph_tensor::make(input_dims)})
                           ->get_outputs()[0];
        auto addout1 = g.make("add", {in0, in1}, {}, {})->get_outputs()[0];
        auto addout2 = g.make("add", {addout1, in2}, {}, {})->get_outputs()[0];
        auto out = g.make_output({addout2});

        return g;
    };
    auto ctx = get_test_ctx();
    sc_graph_t g = get_graph();
    g.attrs_["fpmath_mode"] = 1;
    fpmath_mode(g, ctx);

    std::stringstream ss;
    print_graph(g, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[100, 200], v1: f32[100, 200], v2: f32[100, 200]) -> [v3: f32[100, 200]] {
  [v4: bf16[100, 200]] = cast(v2)
  [v5: bf16[100, 200]] = cast(v1)
  [v6: bf16[100, 200]] = cast(v0)
  [v7: bf16[100, 200]] = add(v6, v5)
  [v8: bf16[100, 200]] = add(v7, v4)
  [v3: f32[100, 200]] = cast(v8)
}
)";
    // Check graph
    EXPECT_EQ(ss.str(), expected_str);

    // Check accuracy
    {
        g = get_graph();
        g.attrs_["fpmath_mode"] = 1;
        graph_driver(g, ctx);
        std::vector<sc_op_ptr> ins_out = g.get_input_ops();
        ins_out.insert(ins_out.begin(), g.get_output_ops()[0]);
        auto f = lower_graph(ctx, g, ins_out);
        auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
        fptr->call_default(&sc_output[0], &input_data0[0], &input_data1[0],
                &input_data2[0]);
    }
    // ref
    {
        g = get_graph();
        auto ref_ctx = get_test_ctx();
        graph_driver(g, ref_ctx);
        std::vector<sc_op_ptr> ins_out = g.get_input_ops();
        ins_out.insert(ins_out.begin(), g.get_output_ops()[0]);
        auto ref_f = lower_graph(ref_ctx, g, ins_out);
        auto ref_fptr = jit_engine_t::make(ref_ctx)->get_entry_func(ref_f);
        ref_fptr->call_default(&ref_output[0], &input_data0[0], &input_data1[0],
                &input_data2[0]);
    }
    test_utils::compare_data(sc_output, ref_output, 1e-2f, 1e-2f);
}

TEST(GCCore_CPU_fpmath_mode_cpp, TestMatmulCast2bf16) {
    REQUIRE_BF16();
    sc_dims A_dims = {32, 64};
    sc_dims B_dims = {64, 64};
    sc_dims bias_dims = {64};
    sc_dims C_dims = {32, 64};
    test_buffer<float> input_A(test_utils::product(A_dims));
    test_buffer<float> input_B(test_utils::product(B_dims));
    test_buffer<float> input_bias(test_utils::product(bias_dims));
    test_utils::fill_data(&input_A[0], test_utils::product(A_dims));
    test_utils::fill_data(&input_B[0], test_utils::product(B_dims));
    test_utils::fill_data(&input_bias[0], test_utils::product(bias_dims));
    std::vector<float> ref_output(test_utils::product(C_dims));
    std::vector<float> sc_output(test_utils::product(C_dims));

    auto get_graph = [A_dims, B_dims, bias_dims]() {
        sc_graph_t g;
        auto in0 = g.make_input({graph_tensor::make(A_dims)})->get_outputs()[0];
        auto in1 = g.make_input({graph_tensor::make(B_dims)})->get_outputs()[0];
        auto bias = g.make_input({graph_tensor::make(bias_dims)})
                            ->get_outputs()[0];
        auto mmout
                = g.make("matmul", {in0, in1, bias}, {}, {})->get_outputs()[0];
        auto reluout = g.make("relu", {mmout}, {}, {})->get_outputs()[0];
        auto out = g.make_output({reluout});
        return g;
    };
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t g = get_graph();
    g.attrs_["fpmath_mode"] = 1;
    graph_inline(g, ctx);
    fpmath_mode(g, ctx);
    std::stringstream ss;
    print_graph(g, ss, true);
    // mmm1 and mmm2 could not be parallel merged due to paritition ring risk
    std::string expected_str
            = R"(graph(v0: f32[32, 64], v1: f32[64, 64], v2: f32[64]) -> [v3: f32[32, 64]] {
  [v4: bf16[64]] = cast(v2)
  [v5: bf16[64, 64]] = cast(v1)
  [v6: bf16[32, 64]] = cast(v0)
  [v7: f32[32, 64]] = managed_matmul_core(v6, v5)
  [v8: bf16[32, 64]] = cast(v7)
  [v9: bf16[32, 64]] = add(v8, v4)
  [v10: bf16[32, 64]] = relu(v9)
  [v3: f32[32, 64]] = cast(v10)
}
)";
    // Check graph
    EXPECT_EQ(ss.str(), expected_str);
    // Check accuracy
    {
        g = get_graph();
        g.attrs_["fpmath_mode"] = 1;
        graph_driver(g, ctx);
        std::vector<sc_op_ptr> ins_out = g.get_input_ops();
        ins_out.insert(ins_out.begin(), g.get_output_ops()[0]);
        auto f = lower_graph(ctx, g, ins_out);
        auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
        fptr->call_default(
                &sc_output[0], &input_A[0], &input_B[0], &input_bias[0]);
    }
    {
        g = get_graph();
        auto &ref_ctx = ctx;
        graph_driver(g, ref_ctx);
        std::vector<sc_op_ptr> ins_out = g.get_input_ops();
        ins_out.insert(ins_out.begin(), g.get_output_ops()[0]);
        auto ref_f = lower_graph(ref_ctx, g, ins_out);
        auto ref_fptr = jit_engine_t::make(ref_ctx)->get_entry_func(ref_f);
        ref_fptr->call_default(
                &ref_output[0], &input_A[0], &input_B[0], &input_bias[0]);
    }

    auto rmse = test_utils::cal_rmse(sc_output, ref_output);
    EXPECT_TRUE(rmse < 1e-2f);
}

TEST(GCCore_CPU_fpmath_mode_cpp, TestMLPInstanceNorm) {
    REQUIRE_BF16();
    sc_dims A_dims = {1, 64, 128};
    sc_dims B_dims = {128, 128};
    sc_dims bias_dims = {128};
    sc_dims C_dims = {1, 64, 128};
    test_buffer<float> input_0(test_utils::product(A_dims));
    test_buffer<float> input_1(test_utils::product(B_dims));
    test_buffer<float> input_2(test_utils::product(bias_dims));
    test_buffer<float> input_3(test_utils::product(A_dims));
    test_buffer<float> input_4(test_utils::product(bias_dims));
    test_buffer<float> input_5(test_utils::product(bias_dims));
    test_utils::fill_data(&input_0[0], test_utils::product(A_dims));
    test_utils::fill_data(&input_1[0], test_utils::product(B_dims));
    test_utils::fill_data(&input_2[0], test_utils::product(bias_dims));
    test_utils::fill_data(&input_3[0], test_utils::product(A_dims));
    test_utils::fill_data(&input_4[0], test_utils::product(bias_dims));
    test_utils::fill_data(&input_5[0], test_utils::product(bias_dims));
    std::vector<float> ref_output(test_utils::product(C_dims));
    std::vector<float> sc_output(test_utils::product(C_dims));

    auto get_graph = [A_dims, B_dims, bias_dims]() {
        sc_graph_t g;
        auto in0 = g.make_input({graph_tensor::make(A_dims)})->get_outputs()[0];
        auto in1 = g.make_input({graph_tensor::make(B_dims)})->get_outputs()[0];
        auto in2 = g.make_input({graph_tensor::make(bias_dims)})
                           ->get_outputs()[0];
        auto in3 = g.make_input({graph_tensor::make(A_dims)})->get_outputs()[0];
        auto in4 = g.make_input({graph_tensor::make(bias_dims)})
                           ->get_outputs()[0];
        auto in5 = g.make_input({graph_tensor::make(bias_dims)})
                           ->get_outputs()[0];
        auto mmout
                = g.make("matmul", {in0, in1, in2}, {}, {{"transpose_b", true}})
                          ->get_outputs()[0];
        auto addout = g.make("add", {mmout, in3}, {}, {})->get_outputs()[0];
        auto lnout
                = g.make("layernorm", {addout, in4, in5}, {},
                           {{"begin_norm_axis", -1}, {"rd_axis", 2},
                                   {"epsilon", 9.9996e-13f},
                                   {"use_affine", true}, {"keep_stats", false}})
                          ->get_outputs()[0];
        auto out = g.make_output({lnout});
        return g;
    };
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    sc_graph_t g = get_graph();
    g.attrs_["fpmath_mode"] = 1;
    graph_inline(g, ctx);
    fpmath_mode(g, ctx);
    std::stringstream ss;
    print_graph(g, ss, true);
    // mmm1 and mmm2 could not be parallel merged due to paritition ring risk
    std::string expected_str
            = R"(graph(v0: f32[1, 64, 128], v1: f32[128, 128], v2: f32[128], v3: f32[1, 64, 128], v4: f32[128], v5: f32[128]) -> [v6: f32[1, 64, 128]] {
  [v7: f32[1]] = constant([1])
  [v8: f32[1]] = constant([1])
  [v9: f32[1]] = constant([1])
  [v10: bf16[128]] = cast(v5)
  [v11: bf16[128]] = cast(v4)
  [v12: bf16[1, 64, 128]] = cast(v3)
  [v13: bf16[128]] = cast(v2)
  [v14: bf16[128, 128]] = cast(v1)
  [v15: bf16[128, 128]] = transpose(v14)
  [v16: bf16[1, 64, 128]] = cast(v0)
  [v17: bf16[64, 128]] = tensor_view(v16)
  [v18: f32[64, 128]] = managed_matmul_core(v17, v15)
  [v19: bf16[64, 128]] = cast(v18)
  [v20: bf16[1, 64, 128]] = tensor_view(v19)
  [v21: bf16[1, 64, 128]] = add(v20, v13)
  [v22: bf16[1, 64, 128]] = add(v21, v12)
  [v23: bf16[1, 64, 1]] = reduce(v22)
  [v24: f32[1, 64, 1]] = cast(v23)
  [v25: f32[1, 64, 1]] = div(v24, v8)
  [v26: bf16[1, 64, 1]] = cast(v25)
  [v27: bf16[1, 64, 128]] = sub(v22, v26)
  [v28: bf16[1, 64, 1]] = cast(v25)
  [v29: bf16[1, 64, 1]] = cast(v25)
  [v30: bf16[1, 64, 1]] = mul(v29, v28)
  [v31: bf16[1, 64, 128]] = mul(v22, v22)
  [v32: bf16[1, 64, 1]] = reduce(v31)
  [v33: f32[1, 64, 1]] = cast(v32)
  [v34: f32[1, 64, 1]] = div(v33, v7)
  [v35: bf16[1, 64, 1]] = cast(v34)
  [v36: bf16[1, 64, 1]] = sub(v35, v30)
  [v37: f32[1, 64, 1]] = cast(v36)
  [v38: f32[1, 64, 1]] = add(v37, v9)
  [v39: bf16[1, 64, 1]] = cast(v38)
  [v40: bf16[1, 64, 1]] = squared_root(v39)
  [v41: bf16[1, 64, 128]] = mul(v27, v40)
  [v42: bf16[1, 64, 128]] = mul(v41, v11)
  [v43: bf16[1, 64, 128]] = add(v42, v10)
  [v6: f32[1, 64, 128]] = cast(v43)
}
)";
    // Check graph
    EXPECT_EQ(ss.str(), expected_str);
    // Check accuracy
    {
        g = get_graph();
        g.attrs_["fpmath_mode"] = 1;
        graph_driver(g, ctx);
        std::vector<sc_op_ptr> ins_out = g.get_input_ops();
        ins_out.insert(ins_out.begin(), g.get_output_ops()[0]);
        auto f = lower_graph(ctx, g, ins_out);
        auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
        fptr->call_default(&sc_output[0], &input_0[0], &input_1[0], &input_2[0],
                &input_3[0], &input_4[0], &input_5[0]);
    }
    {
        g = get_graph();
        auto &ref_ctx = ctx;
        graph_driver(g, ref_ctx);
        std::vector<sc_op_ptr> ins_out = g.get_input_ops();
        ins_out.insert(ins_out.begin(), g.get_output_ops()[0]);
        auto ref_f = lower_graph(ref_ctx, g, ins_out);
        auto ref_fptr = jit_engine_t::make(ref_ctx)->get_entry_func(ref_f);
        ref_fptr->call_default(&ref_output[0], &input_0[0], &input_1[0],
                &input_2[0], &input_3[0], &input_4[0], &input_5[0]);
    }

    auto rmse = test_utils::cal_rmse(sc_output, ref_output);
    EXPECT_TRUE(rmse < 1e-2f);
}

TEST(GCCore_CPU_fpmath_mode_cpp, TestConvCast2bf16) {
    REQUIRE_BF16();
    int N = 32, IC = 128, OC = 128, H = 12, W = 12, R = 1, S = 1;
    sc_dims input_dims = {N, H, W, IC};
    sc_dims filter_dims = {R, S, IC, OC};
    sc_dims bias_dims = {OC};
    sc_dims output_dims = {N, H, W, OC};
    test_buffer<float> input_A(test_utils::product(input_dims));
    test_buffer<float> input_B1(test_utils::product(filter_dims));
    test_buffer<float> input_B2(test_utils::product(filter_dims));
    test_buffer<float> input_bias(test_utils::product(bias_dims));
    test_utils::fill_data(&input_A[0], test_utils::product(input_dims));
    test_utils::fill_data(&input_B1[0], test_utils::product(filter_dims));
    test_utils::fill_data(&input_B2[0], test_utils::product(filter_dims));
    test_utils::fill_data(&input_bias[0], test_utils::product(bias_dims));
    std::vector<float> ref_output(test_utils::product(output_dims));
    std::vector<float> sc_output(test_utils::product(output_dims));

    auto get_graph = [input_dims, filter_dims, bias_dims]() {
        sc_graph_t g;
        auto data = g.make_input({graph_tensor::make(input_dims,
                                         sc_data_format_t(format_kinds::ABCD))})
                            ->get_outputs()[0];
        auto w1 = g.make_input({graph_tensor::make(filter_dims,
                                       sc_data_format_t(format_kinds::ABCD))})
                          ->get_outputs()[0];
        auto bias = g.make_input({graph_tensor::make(bias_dims)})
                            ->get_outputs()[0];
        auto w2 = g.make_input({graph_tensor::make(filter_dims,
                                       sc_data_format_t(format_kinds::ABCD))})
                          ->get_outputs()[0];
        std::unordered_map<std::string, any_t> attrs
                = {{"strides", sc_dims {1, 1}}, {"pads_begin", sc_dims {0, 0}},
                        {"pads_end", sc_dims {0, 0}}};

        auto conv1_out
                = g.make("conv_fwd", {data, w1, bias}, {}, any_map_t(attrs))
                          ->get_outputs()[0];
        auto relu1_out = g.make("relu", {conv1_out}, {}, {})->get_outputs()[0];
        auto conv2_out = g.make("conv_fwd", {data, w2}, {}, any_map_t(attrs))
                                 ->get_outputs()[0];
        auto relu2_out = g.make("relu", {conv2_out}, {}, {})->get_outputs()[0];
        auto out = g.make("add", {relu1_out, relu2_out}, {}, {})
                           ->get_outputs()[0];
        g.make_output({out});
        return g;
    };
    auto ctx = get_test_ctx();
    sc_graph_t g = get_graph();
    g.attrs_["fpmath_mode"] = 1;
    graph_inline(g, ctx);
    fpmath_mode(g, ctx);
    std::stringstream ss;
    print_graph(g, ss, true);
    // mmm1 and mmm2 could not be parallel merged due to paritition ring risk
    std::string expected_str
            = R"(graph(v0: f32[32, 12, 12, 128], v1: f32[1, 1, 128, 128], v2: f32[128], v3: f32[1, 1, 128, 128]) -> [v4: f32[32, 12, 12, 128]] {
  [v5: bf16[1, 1, 128, 128]] = cast(v3)
  [v6: bf16[128, 128, 1, 1]] = transpose(v5)
  [v7: bf16[128]] = cast(v2)
  [v8: bf16[1, 1, 128, 128]] = cast(v1)
  [v9: bf16[128, 128, 1, 1]] = transpose(v8)
  [v10: bf16[32, 12, 12, 128]] = cast(v0)
  [v11: bf16[32, 128, 12, 12]] = transpose(v10)
  [v12: f32[32, 128, 12, 12]] = conv_fwd_core(v11, v6)
  [v13: bf16[32, 128, 12, 12]] = cast(v12)
  [v14: bf16[32, 12, 128, 12]] = transpose(v13)
  [v15: bf16[32, 12, 12, 128]] = relu(v14)
  [v16: bf16[32, 12, 12, 128]] = cast(v0)
  [v17: bf16[32, 128, 12, 12]] = transpose(v16)
  [v18: f32[32, 128, 12, 12]] = conv_fwd_core(v17, v9)
  [v19: bf16[32, 128, 12, 12]] = cast(v18)
  [v20: bf16[32, 12, 128, 12]] = transpose(v19)
  [v21: bf16[32, 12, 128, 12]] = add(v20, v7)
  [v22: bf16[32, 12, 12, 128]] = relu(v21)
  [v23: bf16[32, 12, 12, 128]] = add(v22, v15)
  [v4: f32[32, 12, 12, 128]] = cast(v23)
}
)";
    // Check graph
    EXPECT_EQ(ss.str(), expected_str);
    // Check accuracy
    {
        g = get_graph();
        g.attrs_["fpmath_mode"] = 1;
        graph_driver(g, ctx);
        std::vector<sc_op_ptr> ins_out = g.get_input_ops();
        ins_out.insert(ins_out.begin(), g.get_output_ops()[0]);
        auto f = lower_graph(ctx, g, ins_out);
        auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);
        fptr->call_default(&sc_output[0], &input_A[0], &input_B1[0],
                &input_bias[0], &input_B2[0]);
    }
    {
        g = get_graph();
        auto ref_ctx = get_test_ctx();
        graph_driver(g, ref_ctx);
        std::vector<sc_op_ptr> ins_out = g.get_input_ops();
        ins_out.insert(ins_out.begin(), g.get_output_ops()[0]);
        auto ref_f = lower_graph(ref_ctx, g, ins_out);
        auto ref_fptr = jit_engine_t::make(ref_ctx)->get_entry_func(ref_f);
        ref_fptr->call_default(&ref_output[0], &input_A[0], &input_B1[0],
                &input_bias[0], &input_B2[0]);
    }

    auto rmse = test_utils::cal_rmse(sc_output, ref_output);
    EXPECT_TRUE(rmse < 1e-2f);
}
