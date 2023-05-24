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
#include "gtest/gtest.h"
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_elemwise_bcast_swap, TestElemwiseBcastSwap) {
    sc_graph_t mgr;
    auto in_a = mgr.make_input({graph_tensor::make({28, 64, 16, 16})});
    auto in_b = mgr.make_input({graph_tensor::make({28, 64, 16, 16})});
    auto in_c = mgr.make_input({graph_tensor::make({28, 64, 1, 16})});
    auto normal_add = mgr.make(
            "add", {in_a->get_outputs()[0], in_b->get_outputs()[0]}, {}, {});
    auto bcast_add = mgr.make("add",
            {normal_add->get_outputs()[0], in_c->get_outputs()[0]}, {}, {});
    auto normal_add2 = mgr.make("add",
            {bcast_add->get_outputs()[0], in_c->get_outputs()[0]}, {}, {});
    auto out = mgr.make_output(
            {bcast_add->get_outputs()[0], normal_add2->get_outputs()[0]});
    elemwise_bcast_swap(mgr);
    EXPECT_EQ(check_graph_connection(mgr), true);
    std::stringstream ss;
    print_graph(mgr, ss, true);

    sc_graph_t mgr2;
    {
        auto in_a = mgr2.make_input({graph_tensor::make({28, 64, 16, 16})});
        auto in_b = mgr2.make_input({graph_tensor::make({28, 64, 16, 16})});
        auto in_c = mgr2.make_input({graph_tensor::make({28, 64, 1, 16})});
        auto b_add = mgr2.make("add",
                {in_a->get_outputs()[0], in_c->get_outputs()[0]}, {}, {});
        auto add1 = mgr2.make("add",
                {b_add->get_outputs()[0], in_b->get_outputs()[0]}, {}, {});
        auto add2 = mgr2.make("add",
                {add1->get_outputs()[0], in_c->get_outputs()[0]}, {}, {});
        auto out = mgr2.make_output(
                {add1->get_outputs()[0], add2->get_outputs()[0]});
        std::swap(in_c->get_outputs()[0]->uses_[0],
                in_c->get_outputs()[0]->uses_[1]);
    }

    EXPECT_TRUE(compare_graph(mgr, mgr2));
}

// should not swap if parent is also bcast
TEST(GCCore_CPU_elemwise_bcast_swap, TestFailParentBCast) {
    auto make_graph = []() {
        sc_graph_t mgr;
        auto in_a = mgr.make_input({graph_tensor::make({28, 64, 16, 16})});
        auto in_b = mgr.make_input({graph_tensor::make({28, 64, 1, 16})});
        auto in_c = mgr.make_input({graph_tensor::make({28, 64, 1, 16})});
        // parent is also bcast
        auto normal_add = mgr.make("add",
                {in_a->get_outputs()[0], in_b->get_outputs()[0]}, {}, {});
        auto bcast_add = mgr.make("add",
                {normal_add->get_outputs()[0], in_c->get_outputs()[0]}, {}, {});
        auto normal_add2 = mgr.make("add",
                {bcast_add->get_outputs()[0], in_c->get_outputs()[0]}, {}, {});
        auto out = mgr.make_output(
                {bcast_add->get_outputs()[0], normal_add2->get_outputs()[0]});
        return mgr;
    };

    auto mgr = make_graph();
    elemwise_bcast_swap(mgr);
    EXPECT_EQ(check_graph_connection(mgr), true);
    EXPECT_TRUE(compare_graph(mgr, make_graph()));
}
