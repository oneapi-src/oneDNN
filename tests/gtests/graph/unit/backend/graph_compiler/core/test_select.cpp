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

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <time.h>
#include <vector>
#include "context.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/fusible/ternary_elemwise.hpp>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc;

static void select_ref(const sc_dims &cond_plain_dims,
        const sc_dims &then_plain_dims, const sc_dims &else_plain_dims,
        const sc_dims &out_plain_dims,
        const std::vector<std::vector<int>> &ternary_bc_axis,
        test_buffer<uint8_t> &cond, test_buffer<float> &then,
        test_buffer<float> &els, test_buffer<float> &out) {
    auto &cond_plain_axis = ternary_bc_axis[0];
    auto &then_plain_axis = ternary_bc_axis[1];
    auto &else_plain_axis = ternary_bc_axis[2];

    auto extended_cond_plain_dims = test_utils::get_extended_plain_dims(
            cond_plain_axis, cond_plain_dims, out_plain_dims);
    auto extended_then_plain_dims = test_utils::get_extended_plain_dims(
            then_plain_axis, then_plain_dims, out_plain_dims);
    auto extended_else_plain_dims = test_utils::get_extended_plain_dims(
            else_plain_axis, else_plain_dims, out_plain_dims);

    sc_dims cond_strides
            = test_utils::compute_dense_stride(extended_cond_plain_dims);
    sc_dims then_strides
            = test_utils::compute_dense_stride(extended_then_plain_dims);
    sc_dims else_strides
            = test_utils::compute_dense_stride(extended_else_plain_dims);
    sc_dims output_strides = test_utils::compute_dense_stride(out_plain_dims);

    const size_t total_size = out.size();
    utils::parallel_for(0, total_size, 1, [&](int64_t i) {
        size_t cond_idx_flattened = 0, then_idx_flattened = 0,
               else_idx_flattened = 0;
        size_t idx = i;
        for (size_t d = 0; d < output_strides.size(); ++d) {
            auto output_idx = idx / output_strides[d];
            idx -= output_idx * output_strides[d];
            size_t cond_idx = extended_cond_plain_dims[d] == 1 ? 0 : output_idx;
            cond_idx_flattened += cond_idx * cond_strides[d];
            cond_idx = extended_then_plain_dims[d] == 1 ? 0 : output_idx;
            then_idx_flattened += cond_idx * then_strides[d];
            cond_idx = extended_else_plain_dims[d] == 1 ? 0 : output_idx;
            else_idx_flattened += cond_idx * else_strides[d];
        }
        out[i] = cond[cond_idx_flattened] > 0UL ? then[then_idx_flattened]
                                                : els[else_idx_flattened];
    });
}

static void check_select_correctness(const sc_dims &cond_plain_dims,
        const sc_dims &else_plain_dims, const sc_dims &then_plain_dims,
        sc_data_format_t cond_format = sc_data_format_t(),
        sc_data_format_t else_format = sc_data_format_t(),
        sc_data_format_t then_format = sc_data_format_t()) {
    REQUIRE_AVX2(); // llvm stuck when SSE
    sc_graph_t graph;
    auto input = graph.make_input({std::make_shared<graph_tensor>(nullptr,
                                           cond_format.to_plain(),
                                           cond_plain_dims, datatypes::u8),
            std::make_shared<graph_tensor>(nullptr, then_format.to_plain(),
                    then_plain_dims, datatypes::f32),
            std::make_shared<graph_tensor>(nullptr, else_format.to_plain(),
                    else_plain_dims, datatypes::f32)});
    sc_op_ptr reorder_cond, reorder_then, reorder_else, select, reorder_out;
    reorder_cond = graph.make("reorder", {input->get_outputs()[0]}, {},
            {{"out_format", cond_format}, {"internal", true}});
    reorder_then = graph.make("reorder", {input->get_outputs()[1]}, {},
            {{"out_format", then_format}, {"internal", true}});
    reorder_else = graph.make("reorder", {input->get_outputs()[2]}, {},
            {{"out_format", else_format}, {"internal", true}});
    select = graph.make("select",
            {reorder_cond->get_outputs()[0], reorder_then->get_outputs()[0],
                    reorder_else->get_outputs()[0]},
            {}, {});
    reorder_out = graph.make("reorder", {select->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t(format_kinds::ABCD)},
                    {"internal", true}});

    auto output = graph.make_output(reorder_out->get_outputs());
    graph_driver(graph, get_test_ctx());
    ir_module_ptr mod = lower_graph(get_test_ctx(), graph, {input, output});
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(mod, true);
    std::vector<generic_val> gargs;
    sc_dim cond_size = test_utils::product(cond_plain_dims);
    sc_dim then_size = test_utils::product(then_plain_dims);
    sc_dim else_size = test_utils::product(else_plain_dims);
    auto out_plain_dims = output->get_inputs()[0]->details_.get_plain_dims();
    sc_dim out_size = test_utils::product(out_plain_dims);

    const auto &ternary_bc_axis
            = dynamic_cast<select_op_t *>(select.get())->get_plain_bc_axis();

    auto cond = alloc_array<uint8_t>(cond_size, INIT_RANDOM);
    auto then = alloc_array<float>(then_size, INIT_RANDOM);
    auto els = alloc_array<float>(else_size, INIT_ZERO);
    auto out = alloc_array<float>(out_size, INIT_ZERO);
    auto ref_out = alloc_array<float>(out_size, INIT_ZERO);

    gargs.emplace_back(cond.data());
    gargs.emplace_back(then.data());
    gargs.emplace_back(els.data());
    gargs.emplace_back(out.data());
    fptr->call_generic_default(gargs.data());

    select_ref(cond_plain_dims, then_plain_dims, else_plain_dims,
            out_plain_dims, ternary_bc_axis, cond, then, els, ref_out);
    test_utils::compare_data<float>(out.data(), ref_out.data(), ref_out.size());
}

static void check_distill_bert_mha(const sc_dims &feature_plain_dims,
        const sc_dims &weight_plain_dims, const sc_dims &cond_plain_dims,
        const sc_dims &then_plain_dims, const sc_dims &feature2_plain_dims,
        sc_data_format_t feature_format = sc_data_format_t(),
        sc_data_format_t weight_format = sc_data_format_t(),
        sc_data_format_t cond_format = sc_data_format_t(),
        sc_data_format_t then_format = sc_data_format_t(),
        sc_data_format_t feature2_format = sc_data_format_t(),
        bool use_then_as_else = false) {
    REQUIRE_AVX2(); // llvm stuck when SSE
    sc_graph_t graph;
    auto input = graph.make_input(
            {std::make_shared<graph_tensor>(nullptr, feature_format,
                     feature_plain_dims, datatypes::f32),
                    std::make_shared<graph_tensor>(nullptr, weight_format,
                            weight_plain_dims, datatypes::f32),
                    std::make_shared<graph_tensor>(nullptr, cond_format,
                            cond_plain_dims, datatypes::u8),
                    std::make_shared<graph_tensor>(nullptr, then_format,
                            then_plain_dims, datatypes::f32),
                    std::make_shared<graph_tensor>(nullptr, feature2_format,
                            feature2_plain_dims, datatypes::f32)});

    sc_op_ptr matmul, select, softmax, matmul2, transpose, reorder;
    matmul = graph.make("matmul_core",
            {input->get_outputs()[0], input->get_outputs()[1]}, {}, {});
    if (use_then_as_else) {
        select = graph.make("select",
                {input->get_outputs()[2], matmul->get_outputs()[0],
                        input->get_outputs()[3]},
                {}, {});
    } else {
        select = graph.make("select",
                {input->get_outputs()[2], input->get_outputs()[3],
                        matmul->get_outputs()[0]},
                {}, {});
    }
    softmax = graph.make("softmax", {select->get_outputs()[0]}, {},
            {{"axis", std::vector<int> {3}}});
    matmul2 = graph.make("matmul_core",
            {softmax->get_outputs()[0], input->get_outputs()[4]}, {}, {});
    transpose = graph.make("transpose", {matmul2->get_outputs()[0]}, {},
            {{"order", std::vector<int> {0, 2, 1, 3}}});
    reorder = graph.make("reorder", {transpose->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t(format_kinds::ABCD)},
                    {"internal", true}});

    auto output = graph.make_output(reorder->get_outputs());
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    graph_driver(graph, ctx);
    ir_module_ptr mod = lower_graph(ctx, graph, {input, output});
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(mod, true);
    std::vector<generic_val> gargs;
    sc_dim feature_size = test_utils::product(feature_plain_dims);
    sc_dim weight_size = test_utils::product(weight_plain_dims);
    sc_dim cond_size = test_utils::product(cond_plain_dims);
    sc_dim then_size = test_utils::product(then_plain_dims);
    sc_dim feature2_size = test_utils::product(feature2_plain_dims);
    sc_dim out_size = test_utils::product(
            output->get_inputs()[0]->details_.get_plain_dims());

    std::vector<float> feature(feature_size, 0.f);
    std::iota(feature.begin(), feature.end(), 0);
    std::vector<float> weight(weight_size, 1.f);
    std::vector<uint8_t> cond(cond_size, 1);
    std::vector<float> then(then_size, 0.f);
    std::vector<float> feature2(feature2_size, 1.f);
    std::vector<float> out(out_size, 0.f);
    gargs.emplace_back(feature.data());
    gargs.emplace_back(weight.data());
    gargs.emplace_back(cond.data());
    gargs.emplace_back(then.data());
    gargs.emplace_back(feature2.data());
    gargs.emplace_back(out.data());
    fptr->call_generic_default(gargs.data());
}

TEST(GCCore_CPU_select_test, TestCorrectnessNonBlocking) {
    check_select_correctness({1}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A));
    // 1D + 4D + 1D
    check_select_correctness({68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A));
    // 2D + 4D + 1D
    check_select_correctness({16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::AB),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::BA),
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::A));
    //  3D + 4D + 1D
    check_select_correctness({128, 16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::ABC),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({128, 16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::ACB),
            sc_data_format_t(format_kinds::ACDB),
            sc_data_format_t(format_kinds::A));
    //  4D + 4D + 1D
    check_select_correctness({2, 128, 16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({205, 1, 1, 132}, {205, 12, 132, 132}, {1},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::A));
    //  4D + 1D + 4D
    check_select_correctness({2, 128, 16, 68}, {1}, {2, 128, 16, 68},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ABCD));
    check_select_correctness({205, 1, 1, 132}, {1}, {205, 12, 132, 132},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ACBD));
    //  4D + 4D + 4D
    check_select_correctness({205, 1, 1, 132}, {205, 1, 1, 132},
            {205, 1, 1, 132}, sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABCD));
    check_select_correctness({16, 1, 1, 32}, {16, 1, 1, 32}, {16, 1, 1, 32},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABCD));
    check_select_correctness({16, 16, 1, 32}, {16, 16, 16, 32}, {16, 1, 16, 32},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABCD));
    check_select_correctness({16, 16, 1, 32}, {16, 16, 16, 32}, {16, 1, 16, 32},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABCD));
    check_select_correctness({16, 16, 1, 32}, {16, 16, 16, 32}, {16, 1, 16, 32},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABCD));
}

TEST(GCCore_CPU_select_test, TestCorrectnessSingleSideBlockingThen) {
    check_select_correctness({64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ABCDcd, {4, 16}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ACBDcd, {4, 16}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::AB),
            sc_data_format_t(format_kinds::ABCDcd, {4, 16}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::BA),
            sc_data_format_t(format_kinds::ACBDcd, {4, 16}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::BA),
            sc_data_format_t(format_kinds::ACBDcdc, {4, 16, 4}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::AB),
            sc_data_format_t(format_kinds::ABCDdc, {4, 16}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {4, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::BA),
            sc_data_format_t(format_kinds::ABCDba, {4, 4}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {4, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::BA),
            sc_data_format_t(format_kinds::ABCDb, {4}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({205, 1, 1, 132}, {205, 12, 128, 132}, {1},
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::ABCDcd, {64, 66}),
            sc_data_format_t(format_kinds::A));
}

TEST(GCCore_CPU_select_test, TestCorrectnessSingleSideBlockingCond) {
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::ABab, {4, 16}),
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::ABba, {4, 16}),
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::BAab, {4, 16}),
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::A));
}

TEST(GCCore_CPU_select_test, TestCorrectnessBlocking) {
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::ABab, {4, 16}),
            sc_data_format_t(format_kinds::ACBDcd, {4, 16}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::ABba, {4, 16}),
            sc_data_format_t(format_kinds::ACBDcd, {4, 16}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::BAab, {4, 16}),
            sc_data_format_t(format_kinds::ACBDcd, {4, 16}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::ABab, {4, 16}),
            sc_data_format_t(format_kinds::ABCDcd, {4, 16}),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 64}, {2, 128, 16, 64}, {1},
            sc_data_format_t(format_kinds::ABba, {4, 16}),
            sc_data_format_t(format_kinds::ACBDcdc, {4, 16, 4}),
            sc_data_format_t(format_kinds::A));
}

TEST(GCCore_CPU_distill_bert_test, TestFuntionality) {
    check_distill_bert_mha({205, 12, 132, 64}, {205, 12, 64, 132},
            {205, 1, 1, 132}, {1}, {205, 12, 132, 64},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_kind_t(0, 3, 1, 2),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ACBD));
    check_distill_bert_mha({16, 16, 1, 256}, {16, 16, 256, 48}, {1, 1, 1, 48},
            {1}, {16, 16, 48, 256}, sc_data_format_t(format_kinds::ACBD),
            sc_data_format_kind_t(0, 3, 1, 2),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ACBD), true);
    check_distill_bert_mha({16, 16, 1, 256}, {16, 16, 256, 48}, {16, 16, 1, 48},
            {1}, {16, 16, 48, 256}, sc_data_format_t(format_kinds::ACBD),
            sc_data_format_kind_t(0, 3, 1, 2),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ACBD));
    check_distill_bert_mha({16, 16, 1, 256}, {16, 16, 256, 48}, {16, 16, 1, 48},
            {1}, {16, 16, 48, 256}, sc_data_format_t(format_kinds::ACBD),
            sc_data_format_kind_t(0, 3, 1, 2),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ACBD), true);
}
