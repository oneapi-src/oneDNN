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
#include <numeric>
#include <vector>

#include "context.hpp"
#include "reference/act_ref.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/jit/jit.hpp>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>

using namespace dnnl::impl::graph::gc;

static void binary_add_ref(const sc_dims &lhs_plain_dims,
        const sc_dims &rhs_plain_dims, const sc_dims &out_plain_dims,
        const std::vector<std::vector<int>> &plain_bc_axis,
        std::vector<float> &lhs, std::vector<float> &rhs,
        std::vector<float> &out) {
    auto &lhs_plain_axis = plain_bc_axis[0];
    auto &rhs_plain_axis = plain_bc_axis[1];
    auto extended_lhs_plain_dims = test_utils::get_extended_plain_dims(
            lhs_plain_axis, lhs_plain_dims, out_plain_dims);
    auto extended_rhs_plain_dims = test_utils::get_extended_plain_dims(
            rhs_plain_axis, rhs_plain_dims, out_plain_dims);
    sc_dims lhs_strides
            = test_utils::compute_dense_stride(extended_lhs_plain_dims);
    sc_dims rhs_strides
            = test_utils::compute_dense_stride(extended_rhs_plain_dims);
    sc_dims output_strides = test_utils::compute_dense_stride(out_plain_dims);
    const size_t total_size = out.size();
    utils::parallel_for(0, total_size, 1, [&](int64_t i) {
        size_t lhs_idx_flattened = 0, rhs_idx_flattened = 0;
        size_t idx = i;
        for (size_t d = 0; d < output_strides.size(); ++d) {
            auto output_idx = idx / output_strides[d];
            idx -= output_idx * output_strides[d];
            auto lhs_idx = extended_lhs_plain_dims[d] == 1 ? 0 : output_idx;
            lhs_idx_flattened += lhs_idx * lhs_strides[d];
            auto rhs_idx = extended_rhs_plain_dims[d] == 1 ? 0 : output_idx;
            rhs_idx_flattened += rhs_idx * rhs_strides[d];
        }
        out[i] = lhs[lhs_idx_flattened] + rhs[rhs_idx_flattened];
    });
}

static void check_broadcast_correctness(const sc_dims &lhs_plain_dims,
        const sc_dims &rhs_plain_dims,
        sc_data_format_t lhs_format = sc_data_format_t(),
        sc_data_format_t rhs_format = sc_data_format_t(),
        const std::vector<int> &bc_axis = {}) {
    REQUIRE_AVX2();
    sc_graph_t graph;
    auto input = graph.make_input({std::make_shared<graph_tensor>(nullptr,
                                           lhs_format.to_plain(),
                                           lhs_plain_dims, datatypes::f32),
            std::make_shared<graph_tensor>(nullptr, rhs_format.to_plain(),
                    rhs_plain_dims, datatypes::f32)});
    sc_op_ptr reorder_lhs, reorder_rhs;
    reorder_lhs = graph.make("reorder", {input->get_outputs()[0]}, {},
            {{"out_format", lhs_format}, {"internal", true}});
    reorder_rhs = graph.make("reorder", {input->get_outputs()[1]}, {},
            {{"out_format", rhs_format}, {"internal", true}});
    auto add = graph.make("add",
            {reorder_lhs->get_outputs()[0], reorder_rhs->get_outputs()[0]}, {},
            {{"bc_axis", bc_axis}});
    auto output = graph.make_output(add->get_outputs());
    graph_driver(graph, get_test_ctx());
    ir_module_ptr mod = lower_graph(get_test_ctx(), graph, {input, output});
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(mod, true);
    std::vector<generic_val> gargs;
    sc_dim lhs_size = test_utils::product(lhs_plain_dims);
    sc_dim rhs_size = test_utils::product(rhs_plain_dims);
    const auto &out_plain_dims
            = output->get_inputs()[0]->details_.get_plain_dims();
    const auto &plain_bc_axis
            = dynamic_cast<op_traits::may_broadcast_t *>(add.get())
                      ->get_plain_bc_axis();
    sc_dim out_size = test_utils::product(out_plain_dims);
    std::vector<float> lhs(lhs_size, 0.f);
    std::iota(lhs.begin(), lhs.end(), 0);
    std::vector<float> rhs(rhs_size, 0.f);
    std::iota(rhs.begin(), rhs.end(), 0);
    std::vector<float> out(out_size, 0.f);
    std::vector<float> ref_out(out_size, 0.f);
    gargs.emplace_back(lhs.data());
    gargs.emplace_back(rhs.data());
    gargs.emplace_back(out.data());
    fptr->call_generic_default(gargs.data());
    binary_add_ref(lhs_plain_dims, rhs_plain_dims, out_plain_dims,
            plain_bc_axis, lhs, rhs, ref_out);
    test_utils::compare_data<float>(out.data(), ref_out.data(), ref_out.size());
}

TEST(GCCore_CPU_binary_elementwise_test, TestCorrectnessNonBlocking) {
    // 4D + 1D
    check_broadcast_correctness({2, 3, 5, 7}, {1},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 3, 2)),
            sc_data_format_t(format_kinds::A));
    check_broadcast_correctness({2, 3, 5, 7}, {1},
            sc_data_format_t(sc_data_format_kind_t(3, 2, 1, 0)),
            sc_data_format_t(format_kinds::A));
    check_broadcast_correctness({2, 3, 5, 7}, {7},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 3, 2)),
            sc_data_format_t(format_kinds::A));
    check_broadcast_correctness({2, 3, 5, 7}, {7},
            sc_data_format_t(sc_data_format_kind_t(3, 1, 0, 2)),
            sc_data_format_t(format_kinds::A));
    check_broadcast_correctness({2, 3, 5, 7}, {7},
            sc_data_format_t(sc_data_format_kind_t(1, 0, 3, 2)),
            sc_data_format_t(format_kinds::A));
    check_broadcast_correctness({2, 3, 5, 7}, {3},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 2, 3)),
            sc_data_format_t(format_kinds::A), {1});
    // 4D + 2D
    check_broadcast_correctness({2, 3, 5, 7}, {5, 7},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 3, 2)),
            sc_data_format_t(format_kinds::AB));
    check_broadcast_correctness({2, 3, 5, 7}, {5, 7},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::AB));
    check_broadcast_correctness({2, 3, 5, 7}, {5, 7},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::AB));
    check_broadcast_correctness({2, 3, 5, 7}, {5, 7},
            sc_data_format_t(sc_data_format_kind_t(3, 2, 1, 0)),
            sc_data_format_t(format_kinds::AB));
    check_broadcast_correctness({2, 3, 5, 7}, {5, 7},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 3, 2)),
            sc_data_format_t(format_kinds::BA));
    check_broadcast_correctness({2, 3, 5, 7}, {5, 7},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::BA));
    check_broadcast_correctness({2, 3, 5, 7}, {5, 7},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::BA));
    check_broadcast_correctness({2, 3, 5, 7}, {5, 7},
            sc_data_format_t(sc_data_format_kind_t(3, 2, 1, 0)),
            sc_data_format_t(format_kinds::BA));
    // 4D + 3D
    check_broadcast_correctness({2, 3, 5, 7}, {3, 5, 7},
            sc_data_format_t(sc_data_format_kind_t(3, 2, 1, 0)),
            sc_data_format_t(sc_data_format_kind_t(2, 1, 0)));
    check_broadcast_correctness({2, 3, 5, 7}, {3, 5, 7},
            sc_data_format_t(sc_data_format_kind_t(3, 2, 1, 0)),
            sc_data_format_t(sc_data_format_kind_t(1, 2, 0)));
    check_broadcast_correctness({2, 3, 5, 7}, {3, 5, 7},
            sc_data_format_t(sc_data_format_kind_t(3, 2, 1, 0)),
            sc_data_format_t(sc_data_format_kind_t(0, 2, 1)));
    check_broadcast_correctness({2, 3, 5, 7}, {3, 5, 7},
            sc_data_format_t(sc_data_format_kind_t(3, 2, 1, 0)),
            sc_data_format_t(sc_data_format_kind_t(2, 0, 1)));
    check_broadcast_correctness({2, 3, 5, 7}, {3, 5, 7},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(sc_data_format_kind_t(2, 0, 1)));
    check_broadcast_correctness({2, 3, 5, 7}, {3, 5, 7},
            sc_data_format_t(sc_data_format_kind_t(0, 3, 1, 2)),
            sc_data_format_t(sc_data_format_kind_t(0, 1, 2)));
    // 4D + 4D
    check_broadcast_correctness({2, 3, 5, 7}, {2, 3, 5, 7},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::ABCD));
    check_broadcast_correctness({2, 3, 5, 7}, {2, 1, 1, 7},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(sc_data_format_kind_t(3, 0, 2, 1)));
    // 3D + 4D
    check_broadcast_correctness({96, 1, 1}, {2, 96, 56, 56},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 2)),
            sc_data_format_t(sc_data_format_kind_t(0, 1, 2, 3)));
}

TEST(GCCore_CPU_binary_elementwise_test, TestCorrectnessSingleSideBlockingLHS) {
    // lhs blocking
    check_broadcast_correctness({1, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::ACBDcd, {4, 16}),
            sc_data_format_t(format_kinds::A));
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ACBDcd, {4, 16}),
            sc_data_format_t(format_kinds::AB));
    check_broadcast_correctness({1, 128, 16, 64}, {1, 128, 1, 64},
            sc_data_format_t(format_kinds::ACBDcd, {4, 16}),
            sc_data_format_t(format_kinds::ABCD));
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ACBDcdc, {16, 16, 4}),
            sc_data_format_t(format_kinds::AB));
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ACBDcdc, {16, 16, 4}),
            sc_data_format_t(format_kinds::BA));
    check_broadcast_correctness({64, 64, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ABCDba, {4, 16}),
            sc_data_format_t(format_kinds::AB));
    check_broadcast_correctness({64, 64, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ABCDba, {4, 16}),
            sc_data_format_t(format_kinds::BA));
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 2, 3, 1, 3), {4, 16}),
            sc_data_format_t(format_kinds::AB));
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 2, 3, 1, 3), {4, 16}),
            sc_data_format_t(format_kinds::BA));
    check_broadcast_correctness({128 * 16, 64}, {1, 64},
            sc_data_format_t(sc_data_format_kind_t(0, 0, 0, 1), {128 * 16, 16}),
            sc_data_format_t(format_kinds::AB));
}

TEST(GCCore_CPU_binary_elementwise_test, TestCorrectnessSingleSideBlockingRHS) {
    // rhs blocking
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::AB));
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::ABba, {16, 4}));
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::BAab, {4, 16}));
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(sc_data_format_kind_t(1, 0, 1, 0), {16, 4}));
}

TEST(GCCore_CPU_binary_elementwise_test, TestCorrectnessBlocking) {
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ACBDcd, {4, 16}),
            sc_data_format_t(format_kinds::ABab, {4, 16}));
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ACBDcd, {4, 16}),
            sc_data_format_t(format_kinds::ABab, {8, 8}));
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(format_kinds::ACBDcd, {8, 16}),
            sc_data_format_t(sc_data_format_kind_t(0, 1, 0, 1, 0), {4, 16, 2}));
    check_broadcast_correctness({1, 128, 16, 64}, {16, 64},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 2, 3, 1, 3), {4, 16}),
            sc_data_format_t(format_kinds::ABab, {4, 16}));
}

TEST(GCCore_CPU_binary_elementwise_test, TestConstInferShape) {
    sc_graph_t graph;
    std::vector<float> const_data(320);
    auto const1 = graph.make("constant", {}, {},
            {{"values", std::make_shared<static_data_t>(const_data)},
                    {"dtype", datatypes::f32},
                    {"plain_dims", sc_dims {1, 1, 1, 320}},
                    {"format", sc_data_format_t()}});
    auto input = graph.make_input({std::make_shared<graph_tensor>(
            nullptr, sc_data_format_t(), sc_dims {1}, datatypes::f32)});
    auto add = graph.make(
            "add", {const1->get_outputs()[0], input->get_outputs()[0]}, {}, {});
    auto output = graph.make_output(add->get_outputs());
    std::stringstream ss;
    print_graph(graph, ss, true);
    constexpr const char *expected
            = R"(graph(v0: f32[1]) -> [v1: f32[1, 1, 1, 320]] {
  [v2: f32[1, 1, 1, 320]] = constant([1, 1, 1, 320])
  [v1: f32[1, 1, 1, 320]] = add(v2, v0)
}
)";
    EXPECT_EQ(ss.str(), expected);
}

template <typename T,
        typename dummy = typename std::enable_if<
                std::is_same<typename std::decay<T>::type, float>::value
                || std::is_same<typename std::decay<T>::type, bf16_t>::value>>
static void check_binary_elementwise(const std::string &op_name,
        const sc_dims &input_dims,
        void (*ref_func)(T *, const T *, const T *, size_t)) {
    sc_graph_t g;
    sc_op_ptr ins;
    bool is_bf16 = std::is_same<typename std::decay<T>::type, bf16_t>::value;
    if (is_bf16
            && !::dnnl::impl::graph::gc::get_default_context()
                        ->machine_.cpu_flags_.fAVX512F) {
        return;
    }
    auto dtype = is_bf16 ? datatypes::bf16 : datatypes::f32;
    ins = g.make_input(
            {graph_tensor::make(input_dims, sc_data_format_t(), dtype),
                    graph_tensor::make(input_dims, sc_data_format_t(), dtype)});

    auto op = g.make(op_name, ins->get_outputs(), {}, {});
    g.make_output(op->get_outputs());
    graph_driver(g, get_test_ctx());

    auto f = lower_graph(get_test_ctx(), g, {});
    const auto input_size = test_utils::product(input_dims);
    std::vector<T> input_data1(input_size), input_data2(input_size);
    if (utils::is_one_of(op_name, std::string("pow"))) {
        test_utils::fill_data(&input_data1[0], input_size,
                static_cast<T>(1e-4f), static_cast<T>(1.f));
    } else {
        test_utils::fill_data(&input_data1[0], input_size);
    }
    test_utils::fill_data(&input_data2[0], input_size);
    std::vector<T> ref_output(input_size);
    ref_func(ref_output.data(), input_data1.data(), input_data2.data(),
            input_size);
    std::vector<T> sc_output(input_size);
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f);
    fptr->call_default(&input_data1[0], &input_data2[0], &sc_output[0]);
    if (!is_bf16) {
        test_utils::compare_data(sc_output, ref_output, 1e-4f, 1e-5f);
    } else {
        float sum = 0.f;
        std::for_each(sc_output.begin(), sc_output.end(),
                [&sum](const T &n) { sum += std::abs(float(n)); });
        EXPECT_TRUE(test_utils::cal_rmse(sc_output, ref_output) / sum < 1e-4f);
    }
}

static std::vector<sc_dims> test_shapes
        = {{16, 63}, {2, 8, 4}, {4, 16, 256, 1024}};
TEST(GCCore_CPU_binary_elementwise_test, TestPReluOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_binary_elementwise<float>("prelu", shape, ref_prelu);
        check_binary_elementwise<bf16_t>("prelu", shape, ref_prelu);
    }
}

TEST(GCCore_CPU_binary_elementwise_test, TestCorrectnessBidirectional) {
    check_broadcast_correctness({1, 64, 16, 1}, {16, 64},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::AB));
    check_broadcast_correctness({4, 1, 1, 1}, {16, 32, 16},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABC));
    check_broadcast_correctness({1, 16, 64}, {1, 64, 16, 1},
            sc_data_format_t(format_kinds::ABC),
            sc_data_format_t(format_kinds::ABCD));
    check_broadcast_correctness({4, 16, 1, 1}, {16, 1, 16},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABC));
}
