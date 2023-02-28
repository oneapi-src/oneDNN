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
#include <util/any_map.hpp>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc;

// select_ref function calculates (cond?then:else)
template <class T>
T get_dims_product(const std::vector<T> &dims) {
    T ret = 1;
    for (unsigned i = 0; i < dims.size(); ++i) {
        ret *= dims[i];
    }
    assert(ret > 0 && "Overflow or non-constant shape detected");
    return ret;
}

static std::vector<int> get_stride_size(const sc_dims &src_dims) {
    std::vector<int> stride_size(src_dims.size());
    int product = 1;
    for (size_t i = 0; i < src_dims.size(); i++) {
        stride_size[src_dims.size() - 1 - i] = product;
        product *= src_dims[src_dims.size() - 1 - i];
    }
    return stride_size;
}

static int get_broadcast_input(
        const sc_dims &lhs_dims, const sc_dims &rhs_dims) {
    if (lhs_dims == rhs_dims) {
        return -1;
    } else {
        auto lhs_dp = get_dims_product(lhs_dims);
        auto rhs_dp = get_dims_product(rhs_dims);
        if (lhs_dp == rhs_dp) {
            COMPILE_ASSERT(lhs_dims.size() != rhs_dims.size(),
                    "Unexpected dims of bianry elementwise inputs are found: "
                            << utils::print_vector(lhs_dims) << " and "
                            << utils::print_vector(rhs_dims))
            return lhs_dims.size() > rhs_dims.size() ? 1 : 0;
        } else {
            return lhs_dp > rhs_dp ? 1 : 0;
        }
    }
}

static std::vector<int> get_bc_stride_size(
        const sc_dims &lhs_dims, const sc_dims &rhs_dims) {
    int bc_input_idx = get_broadcast_input(lhs_dims, rhs_dims);
    if (bc_input_idx == -1) return {};

    sc_dims elt_dims, bc_dims;
    if (bc_input_idx == 1) {
        elt_dims = lhs_dims;
        bc_dims = rhs_dims;
    } else if (bc_input_idx == 0) {
        elt_dims = rhs_dims;
        bc_dims = lhs_dims;
    }

    std::vector<int> bc_axis(elt_dims.size(), 0);
    for (size_t i = 0; i < bc_dims.size(); i++) {
        if (elt_dims[elt_dims.size() - 1 - i]
                == bc_dims[bc_dims.size() - 1 - i]) {
            bc_axis[bc_axis.size() - 1 - i] = 1;
        }
    }

    auto stride_size = get_stride_size(bc_dims);
    int base = elt_dims.size() - bc_dims.size();

    std::vector<int> bc_stride_size(bc_axis.size(), 0);
    for (size_t i = 0; i < bc_axis.size(); i++) {
        if (bc_axis[i] == 1) { bc_stride_size[i] = stride_size[i - base]; }
    }
    return bc_stride_size;
}

static void select_ref(const sc_dims &cond_plain_dims,
        const sc_dims &then_plain_dims, const sc_dims &else_plain_dims,
        test_buffer<uint8_t> &cond, test_buffer<float> &then,
        test_buffer<float> &els, test_buffer<float> &out) {
    int bc_input_idx = get_broadcast_input(cond_plain_dims, else_plain_dims);
    sc_dims out_dims;
    std::vector<int> cond_stride(4, 0);
    std::vector<int> else_stride(4, 0);
    std::vector<int> out_stride(4, 0);
    if (bc_input_idx == 0) {
        cond_stride = get_bc_stride_size(cond_plain_dims, else_plain_dims);
        else_stride = get_stride_size(else_plain_dims);
        out_stride = else_stride;
        out_dims.assign(else_plain_dims.begin(), else_plain_dims.end());
    } else if (bc_input_idx == 1) {
        cond_stride = get_stride_size(cond_plain_dims);
        else_stride = get_bc_stride_size(cond_plain_dims, else_plain_dims);
        out_stride = cond_stride;
        out_dims.assign(cond_plain_dims.begin(), cond_plain_dims.end());
    } else {
        cond_stride = get_stride_size(cond_plain_dims);
        else_stride = get_stride_size(else_plain_dims);
        out_stride = else_stride;
        out_dims.assign(else_plain_dims.begin(), else_plain_dims.end());
    }

    for (int i = 0; i < out_dims[0]; i++) {
        for (int j = 0; j < out_dims[1]; j++) {
            for (int k = 0; k < out_dims[2]; k++) {
                for (int l = 0; l < out_dims[3]; l++) {
                    int out_idx = i * out_stride[0] + j * out_stride[1]
                            + k * out_stride[2] + l * out_stride[3];
                    int cond_idx = i * cond_stride[0] + j * cond_stride[1]
                            + k * cond_stride[2] + l * cond_stride[3];
                    int else_idx = i * else_stride[0] + j * else_stride[1]
                            + k * else_stride[2] + l * else_stride[3];
                    out[out_idx]
                            = cond[cond_idx] > 0UL ? then[0] : els[else_idx];
                }
            }
        }
    }
    return;
}

static void check_select_correctness(const sc_dims &cond_plain_dims,
        const sc_dims &else_plain_dims, const sc_dims &then_plain_dims,
        sc_data_format_t cond_format = sc_data_format_t(),
        sc_data_format_t else_format = sc_data_format_t(),
        sc_data_format_t then_format = sc_data_format_t()) {
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
    sc_dim out_size = test_utils::product(
            output->get_inputs()[0]->details_.get_plain_dims());

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

    select_ref(cond_plain_dims, then_plain_dims, else_plain_dims, cond, then,
            els, ref_out);
    test_utils::compare_data<float>(out.data(), ref_out.data(), ref_out.size());
}

static void check_distill_bert_mha(const sc_dims &feature_plain_dims,
        const sc_dims &weight_plain_dims, const sc_dims &cond_plain_dims,
        const sc_dims &then_plain_dims, const sc_dims &feature2_plain_dims,
        sc_data_format_t feature_format = sc_data_format_t(),
        sc_data_format_t weight_format = sc_data_format_t(),
        sc_data_format_t cond_format = sc_data_format_t(),
        sc_data_format_t then_format = sc_data_format_t(),
        sc_data_format_t feature2_format = sc_data_format_t()) {
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
    select = graph.make("select",
            {input->get_outputs()[2], input->get_outputs()[3],
                    matmul->get_outputs()[0]},
            {}, {});
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
    graph_driver(graph, get_test_ctx());
    ir_module_ptr mod = lower_graph(get_test_ctx(), graph, {input, output});
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(mod, true);
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

TEST(GCCore_select_test, TestCorrectnessNonBlocking) {
    check_select_correctness({1}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A));
    // 1D + 4D + 1D
    check_select_correctness({68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ACDB),
            sc_data_format_t(format_kinds::A));
    // 2D + 4D + 1D
    check_select_correctness({16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::AB),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::AB),
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::AB),
            sc_data_format_t(format_kinds::ACDB),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::BA),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::BA),
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::BA),
            sc_data_format_t(format_kinds::ACDB),
            sc_data_format_t(format_kinds::A));
    //  3D + 4D + 1D
    check_select_correctness({128, 16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::ABC),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({128, 16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::ABC),
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({128, 16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::ABC),
            sc_data_format_t(format_kinds::ACDB),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({128, 16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::ACB),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A));
    check_select_correctness({128, 16, 68}, {2, 128, 16, 68}, {1},
            sc_data_format_t(format_kinds::ACB),
            sc_data_format_t(format_kinds::ACBD),
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
}

TEST(GCCore_select_test, TestCorrectnessSingleSideBlockingThen) {
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

TEST(GCCore_select_test, TestCorrectnessSingleSideBlockingCond) {
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

TEST(GCCore_select_test, TestCorrectnessBlocking) {
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

TEST(GCCore_distill_bert_test, TestCorrectness) {
    check_distill_bert_mha({205, 12, 132, 64}, {205, 12, 64, 132},
            {205, 1, 1, 132}, {1}, {205, 12, 132, 64},
            sc_data_format_t(format_kinds::ACBD),
            sc_data_format_kind_t(0, 3, 1, 2),
            sc_data_format_t(format_kinds::ABCD),
            sc_data_format_t(format_kinds::A),
            sc_data_format_t(format_kinds::ACBD));
}
