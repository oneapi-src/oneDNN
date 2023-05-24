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

#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/graph.hpp>

#include <iostream>
#include "exception_util.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_combined_dispatch_key_cpp, TestCombinedKeyEmptyInput) {
    combined_dispatch_key_set_t combined_set(
            (std::vector<std::shared_ptr<dispatch_key_set_base_t>>()));
    EXPECT_EQ(combined_set.size(), UINT64_C(0));
}

TEST(GCCore_CPU_combined_dispatch_key_cpp, TestCombinedKeyOneInput) {
    auto in_set = std::make_shared<dispatch_key_set_t>();
    in_set->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(32, 16), sc_data_format_t::MK()}),
            op_dispatch_key_t(
                    std::vector<sc_data_format_t> {
                            sc_data_format_t::MKmk(32, 16),
                            sc_data_format_t::MK()},
                    impl_kind_t::no_padding)};
    combined_dispatch_key_set_t::inner_set_t expected {
            combined_op_dispatch_key_t {
                    op_dispatch_key_t(std::vector<sc_data_format_t> {
                            sc_data_format_t::MKmk(32, 16),
                            sc_data_format_t::MK()})},
            combined_op_dispatch_key_t {op_dispatch_key_t(
                    std::vector<sc_data_format_t> {
                            sc_data_format_t::MKmk(32, 16),
                            sc_data_format_t::MK()},
                    impl_kind_t::no_padding)}};
    combined_dispatch_key_set_t combined_set({in_set});
    EXPECT_EQ(combined_set.size(), UINT64_C(2));
    EXPECT_EQ(combined_set.set_, expected);
}

TEST(GCCore_CPU_combined_dispatch_key_cpp, TestCombinedKeyMultiInput) {
    auto in_set1 = std::make_shared<dispatch_key_set_t>();
    auto in_set2 = std::make_shared<dispatch_key_set_t>();
    in_set1->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(
                    std::vector<sc_data_format_t> {sc_data_format_t::MK()}),
            op_dispatch_key_t(
                    std::vector<sc_data_format_t> {
                            sc_data_format_t::MKmk(32, 16)},
                    impl_kind_t::no_padding)};
    in_set2->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(
                    std::vector<sc_data_format_t> {sc_data_format_t::NCHWc(16),
                            sc_data_format_t::KCRSck(16, 32)}),
            op_dispatch_key_t(
                    std::vector<sc_data_format_t> {sc_data_format_t::NCHWc(16),
                            sc_data_format_t::KCRSck(32, 32)}),
            op_dispatch_key_t(
                    std::vector<sc_data_format_t> {sc_data_format_t::NCHWc(32),
                            sc_data_format_t::KCRS()},
                    impl_kind_t::no_padding)};
    combined_dispatch_key_set_t::inner_set_t expected {
            combined_op_dispatch_key_t {
                    op_dispatch_key_t(std::vector<sc_data_format_t> {
                            sc_data_format_t::MK()}),
                    op_dispatch_key_t(std::vector<sc_data_format_t> {
                            sc_data_format_t::NCHWc(16),
                            sc_data_format_t::KCRSck(16, 32)})},
            combined_op_dispatch_key_t {
                    op_dispatch_key_t(std::vector<sc_data_format_t> {
                            sc_data_format_t::MK()}),
                    op_dispatch_key_t(std::vector<sc_data_format_t> {
                            sc_data_format_t::NCHWc(16),
                            sc_data_format_t::KCRSck(32, 32)})},
            combined_op_dispatch_key_t {
                    op_dispatch_key_t(std::vector<sc_data_format_t> {
                            sc_data_format_t::MK()}),
                    op_dispatch_key_t(
                            std::vector<sc_data_format_t> {
                                    sc_data_format_t::NCHWc(32),
                                    sc_data_format_t::KCRS()},
                            impl_kind_t::no_padding)},
            combined_op_dispatch_key_t {
                    op_dispatch_key_t(
                            std::vector<sc_data_format_t> {
                                    sc_data_format_t::MKmk(32, 16)},
                            impl_kind_t::no_padding),
                    op_dispatch_key_t(std::vector<sc_data_format_t> {
                            sc_data_format_t::NCHWc(16),
                            sc_data_format_t::KCRSck(16, 32)})},
            combined_op_dispatch_key_t {
                    op_dispatch_key_t(
                            std::vector<sc_data_format_t> {
                                    sc_data_format_t::MKmk(32, 16)},
                            impl_kind_t::no_padding),
                    op_dispatch_key_t(std::vector<sc_data_format_t> {
                            sc_data_format_t::NCHWc(16),
                            sc_data_format_t::KCRSck(32, 32)})},
            combined_op_dispatch_key_t {
                    op_dispatch_key_t(
                            std::vector<sc_data_format_t> {
                                    sc_data_format_t::MKmk(32, 16)},
                            impl_kind_t::no_padding),
                    op_dispatch_key_t(
                            std::vector<sc_data_format_t> {
                                    sc_data_format_t::NCHWc(32),
                                    sc_data_format_t::KCRS()},
                            impl_kind_t::no_padding)},
    };
    combined_dispatch_key_set_t combined_set(
            std::vector<dispatch_set_ptr> {in_set1, in_set2});
    EXPECT_EQ(combined_set.size(), UINT64_C(6));
    EXPECT_EQ(combined_set.set_, expected);
}

TEST(GCCore_CPU_combined_dispatch_key_cpp, TestCombinedKeyGraphLinked) {
    auto in_set1 = std::make_shared<dispatch_key_set_t>(); // data reorder
    auto in_set2 = std::make_shared<dispatch_key_set_t>(); // weight_reorder
    auto in_set3 = std::make_shared<dispatch_key_set_t>(); // matmul1
    auto in_set4 = std::make_shared<dispatch_key_set_t>(); // bias reorder
    auto in_set5 = std::make_shared<dispatch_key_set_t>(); // weight 2 reorder
    auto in_set6 = std::make_shared<dispatch_key_set_t>(); // matmul2
    auto in_set7 = std::make_shared<dispatch_key_set_t>(); // last reorder

    in_set1->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::MK()}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::MKmk(16, 32)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::MKmk(32, 32)})};
    in_set2->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::NKkn(32, 64)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::NKkn(32, 32)})};
    in_set3->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::NKkn(32, 64),
                    sc_data_format_t::MKmk(4, 64)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(16, 32),
                    sc_data_format_t::NKkn(32, 32),
                    sc_data_format_t::MKmk(16, 32)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(32, 32),
                    sc_data_format_t::NKkn(32, 64),
                    sc_data_format_t::MKmk(32, 64)})};
    in_set4->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::MKmk(1, 32)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::MKmk(1, 64)})};
    in_set5->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::KN(), sc_data_format_t::KN()}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::KN(), sc_data_format_t::NKkn(32, 32)})};
    in_set6->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(4, 64), sc_data_format_t::KN(),
                    sc_data_format_t::MKmk(4, 64)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(16, 32), sc_data_format_t::KN(),
                    sc_data_format_t::MKmk(16, 32)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(16, 32),
                    sc_data_format_t::NKkn(32, 32),
                    sc_data_format_t::MKmk(16, 32)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(32, 64), sc_data_format_t::KN(),
                    sc_data_format_t::MKmk(32, 64)})};
    in_set7->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(4, 64), sc_data_format_t::MK()}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(16, 32), sc_data_format_t::MK()}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(32, 64), sc_data_format_t::MK()})};
    sc_graph_t graph;
    auto input1 = graph.make_input({graph_tensor::make({-1, 32})}); // data
    auto input2 = graph.make_input({graph_tensor::make({32, 64})}); // weight
    auto input3 = graph.make_input({graph_tensor::make({1, 64})}); // bias
    auto input4 = graph.make_input({graph_tensor::make({64, 8})}); // weight2
    auto data_reorder = graph.make("reorder", input1->get_outputs(),
            {graph_tensor::make({16, 32})}, {});
    data_reorder->get_dispatch_key_set() = in_set1;
    auto weight_reorder = graph.make("reorder", input2->get_outputs(),
            {graph_tensor::make({32, 64})}, {});
    weight_reorder->get_dispatch_key_set() = in_set2;
    auto matmul1 = graph.make("matmul_core",
            {data_reorder->get_outputs()[0], weight_reorder->get_outputs()[0]},
            {}, {});
    matmul1->get_dispatch_key_set() = in_set3;
    auto bias_reorder = graph.make("reorder", input3->get_outputs(),
            {graph_tensor::make({1, 64})}, {});
    bias_reorder->get_dispatch_key_set() = in_set4;
    auto bias_add = graph.make("add",
            {matmul1->get_outputs()[0], bias_reorder->get_outputs()[0]}, {},
            {});
    auto weight_2_reorder = graph.make("reorder", input4->get_outputs(),
            {graph_tensor::make({64, 8})}, {});
    weight_2_reorder->get_dispatch_key_set() = in_set5;
    auto matmul2 = graph.make("matmul_core",
            {bias_add->get_outputs()[0], weight_2_reorder->get_outputs()[0]},
            {}, {});
    matmul2->get_dispatch_key_set() = in_set6;
    auto last_reorder = graph.make("reorder", matmul2->get_outputs(),
            {graph_tensor::make({64, 8})}, {});
    last_reorder->get_dispatch_key_set() = in_set7;
    auto out = graph.make_output(last_reorder->get_outputs());
    std::vector<sc_op_ptr> input_ops = {data_reorder, weight_reorder, matmul1,
            bias_reorder, weight_2_reorder, matmul2, last_reorder};
    combined_dispatch_key_set_t combined_set(input_ops);
    EXPECT_EQ(combined_set.size(), UINT64_C(4));
}

TEST(GCCore_CPU_combined_dispatch_key_cpp, TestCombinedKeyGraphLinked2) {
    auto in_set1 = std::make_shared<dispatch_key_set_t>(); // data reorder
    auto in_set2 = std::make_shared<dispatch_key_set_t>(); // weight_reorder
    auto in_set3 = std::make_shared<dispatch_key_set_t>(); // matmul1

    in_set1->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::NKkn(32, 64),
                    sc_data_format_t::MKmk(4, 64)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(16, 32),
                    sc_data_format_t::NKkn(32, 32),
                    sc_data_format_t::MKmk(16, 32)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(32, 32),
                    sc_data_format_t::NKkn(32, 64),
                    sc_data_format_t::MKmk(32, 64)})};
    in_set2->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::MKmk(4, 64)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::MKmk(16, 32)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::MKmk(32, 64)}),
    };
    in_set3->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(4, 64), sc_data_format_t::MK()}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(16, 32), sc_data_format_t::MK()}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(32, 64), sc_data_format_t::MK()})};
    sc_graph_t graph;
    auto input1 = graph.make_input({graph_tensor::make({-1, 32})}); // data
    auto input2 = graph.make_input({graph_tensor::make({32, 64})}); // weight
    auto input3
            = graph.make_input({graph_tensor::make({16, 64})}); // add input 1

    auto matmul1 = graph.make("matmul_core",
            {input1->get_outputs()[0], input2->get_outputs()[0]}, {}, {});
    matmul1->get_dispatch_key_set() = in_set1;
    auto reorder1 = graph.make("reorder", input3->get_outputs(),
            {graph_tensor::make({16, 64})}, {});
    reorder1->get_dispatch_key_set() = in_set2;
    auto right_add = graph.make("add",
            {reorder1->get_outputs()[0], matmul1->get_outputs()[0]}, {},
            {{op_attr_key::layout_input_index, 1}});
    auto last_reorder = graph.make("reorder", right_add->get_outputs(),
            {graph_tensor::make({16, 64})}, {});
    last_reorder->get_dispatch_key_set() = in_set3;
    auto out = graph.make_output(last_reorder->get_outputs());
    std::vector<sc_op_ptr> input_ops = {matmul1, reorder1, last_reorder};
    combined_dispatch_key_set_t combined_set(input_ops);
    EXPECT_EQ(combined_set.size(), UINT64_C(3));
}

TEST(GCCore_CPU_combined_dispatch_key_cpp, TestCombinedKeyGraphCheck) {
    auto in_set1 = std::make_shared<dispatch_key_set_t>(); // reorder 1
    auto in_set2 = std::make_shared<dispatch_key_set_t>(); // reorder 2

    in_set1->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MK(), sc_data_format_t::NKkn(32, 64),
                    sc_data_format_t::MKmk(4, 64)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(16, 32),
                    sc_data_format_t::NKkn(32, 32),
                    sc_data_format_t::MKmk(16, 32)}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(32, 32),
                    sc_data_format_t::NKkn(32, 64),
                    sc_data_format_t::MKmk(32, 64)})};
    in_set2->set_ = dispatch_key_set_t::inner_set_t {
            op_dispatch_key_t(
                    std::vector<sc_data_format_t> {
                            sc_data_format_t::MKmk(4, 64),
                            sc_data_format_t::MK()},
                    impl_kind_t::no_padding),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(16, 32), sc_data_format_t::MK()}),
            op_dispatch_key_t(std::vector<sc_data_format_t> {
                    sc_data_format_t::MKmk(32, 64), sc_data_format_t::MK()})};
    sc_graph_t graph;
    auto input1 = graph.make_input({graph_tensor::make({-1, 32})}); // data
    auto reorder1 = graph.make("reorder", input1->get_outputs(),
            {graph_tensor::make({-1, 32})}, {});
    reorder1->get_dispatch_key_set() = in_set1;
    auto reorder2 = graph.make("reorder", reorder1->get_outputs(),
            {graph_tensor::make({-1, 32})}, {});
    reorder2->get_dispatch_key_set() = in_set2;
    auto out = graph.make_output(reorder2->get_outputs());
    std::vector<sc_op_ptr> input_ops = {reorder1, reorder2};
    EXPECT_SC_ERROR(combined_dispatch_key_set_t combined_set((input_ops)),
            "Wrong dispatch key sets, could not construct combined "
            "dispatch key.");
}
