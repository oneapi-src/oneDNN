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
TEST(GCCore_CPU_shape_relationship_binding,
        TestDynamicShapeRelationshipBinding1) {
    sc_graph_t mgr;
    auto in_a = mgr.make_input({graph_tensor::make({28, -1, -1, -1})});
    auto in_b = mgr.make_input({graph_tensor::make({28, -1, -1, -1})});
    auto in_c = mgr.make_input({graph_tensor::make({28, -1, -1, -1})});
    auto matmul1 = mgr.make("matmul_core",
            {in_b->get_outputs()[0], in_c->get_outputs()[0]}, {}, {});
    auto matmul2 = mgr.make("matmul_core",
            {in_a->get_outputs()[0], matmul1->get_outputs()[0]}, {}, {});
    auto out = mgr.make_output({matmul2->get_outputs()[0]});
    shape_relationship_binding(mgr);
    EXPECT_EQ(check_graph_connection(mgr), true);
    sc_dims expected_a_dims {28, -2, -3, -4};
    sc_dims expected_b_dims {28, -2, -4, -5};
    sc_dims expected_c_dims {28, -2, -5, -6};
    sc_dims expected_out_dims {28, -2, -3, -6};
    EXPECT_EQ(
            in_a->get_outputs()[0]->details_.get_plain_dims(), expected_a_dims);
    EXPECT_EQ(
            in_b->get_outputs()[0]->details_.get_plain_dims(), expected_b_dims);
    EXPECT_EQ(
            in_c->get_outputs()[0]->details_.get_plain_dims(), expected_c_dims);
    EXPECT_EQ(
            out->get_inputs()[0]->details_.get_plain_dims(), expected_out_dims);
}

TEST(GCCore_CPU_shape_relationship_binding,
        TestDynamicShapeRelationshipBinding2) {
    sc_graph_t mgr;
    auto in_a = mgr.make_input({graph_tensor::make({28, -1, -1, -1})});
    auto in_b = mgr.make_input({graph_tensor::make({28, -1, -1, -1})});
    auto in_c = mgr.make_input({graph_tensor::make({28, -1, -1, -1})});
    auto in_d = mgr.make_input({graph_tensor::make({28, 16, -1, -1})});
    auto matmul1 = mgr.make("matmul_core",
            {in_a->get_outputs()[0], in_b->get_outputs()[0]}, {}, {});
    auto matmul2 = mgr.make("matmul_core",
            {in_c->get_outputs()[0], in_d->get_outputs()[0]}, {}, {});
    auto add = mgr.make("add",
            {matmul1->get_outputs()[0], matmul2->get_outputs()[0]}, {}, {});
    auto out = mgr.make_output({add->get_outputs()[0]});
    shape_relationship_binding(mgr);
    EXPECT_EQ(check_graph_connection(mgr), true);
    sc_dims expected_a_dims {28, 16, -2, -3};
    sc_dims expected_b_dims {28, 16, -3, -4};
    sc_dims expected_c_dims {28, 16, -2, -5};
    sc_dims expected_d_dims {28, 16, -5, -4};
    sc_dims expected_out_dims {28, 16, -2, -4};
    EXPECT_EQ(
            in_a->get_outputs()[0]->details_.get_plain_dims(), expected_a_dims);
    EXPECT_EQ(
            in_b->get_outputs()[0]->details_.get_plain_dims(), expected_b_dims);
    EXPECT_EQ(
            in_c->get_outputs()[0]->details_.get_plain_dims(), expected_c_dims);
    EXPECT_EQ(
            in_d->get_outputs()[0]->details_.get_plain_dims(), expected_d_dims);
    EXPECT_EQ(
            out->get_inputs()[0]->details_.get_plain_dims(), expected_out_dims);
}
