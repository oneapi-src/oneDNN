/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <string>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "utils.hpp"

TEST(graph_test, create) {
    using namespace dnnl::graph::impl;

    graph_t g_default_engine;
    ASSERT_EQ(g_default_engine.get_engine_kind(), engine_kind::cpu);

    graph_t g_gpu {engine_kind::gpu};
    ASSERT_EQ(g_gpu.get_engine_kind(), engine_kind::gpu);
}

TEST(graph_test, add_op) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv2d")};
    op_t op1 {1, ReLU, std::string("relu")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    ASSERT_EQ(agraph.add_op(&op0), status::invalid_op);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.num_ops(), 1);

    auto ret = agraph.get_ops()[0];
    ASSERT_EQ(*ret, op1);
}

TEST(graph_test, add_null_op) {
    using namespace dnnl::graph::impl;

    graph_t agraph;
    ASSERT_EQ(agraph.add_op(nullptr), status::invalid_op);
    ASSERT_EQ(agraph.num_ops(), 0);
    ASSERT_EQ(agraph.get_ops().size(), 0);
}

TEST(graph_test, delete_op) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv2d")};
    op_t op1 {1, ReLU, std::string("relu")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    ASSERT_EQ(agraph.add_op(&op0), status::invalid_op);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.num_ops(), 1);
    ASSERT_EQ(agraph.get_ops().size(), 1);

    agraph.delete_op(&op1);
    ASSERT_EQ(agraph.num_ops(), 0);
    ASSERT_EQ(agraph.get_ops().size(), 0);
}

TEST(graph_test, get_output_ops) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, ReLU, std::string("relu1")};
    op_t op1 {1, ReLU, std::string("relu2")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.num_ops(), 2);
    agraph.build_graph();
    ASSERT_EQ(agraph.get_output_ops().size(), 1);
    ASSERT_EQ(*(agraph.get_output_ops()[0]), op1);
}

TEST(graph_test, get_output_ops_2) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, ReLU, std::string("relu1")};
    op_t op1 {1, ReLU, std::string("relu2")};
    op_t op2 {2, ReLU, std::string("relu3")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    logical_tensor_t t3 = logical_tensor_init(3, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    op2.add_input(t2);
    op2.add_output(t3);
    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    ASSERT_EQ(agraph.num_ops(), 3);
    agraph.build_graph();
    ASSERT_EQ(agraph.get_ops()[1]
                      ->get_output_value(0)
                      ->get_consumers()[0]
                      .get_op(),
            *agraph.get_ops()[2]);
    ASSERT_EQ(agraph.get_ops()[0]
                      ->get_output_value(0)
                      ->get_consumers()[0]
                      .get_op(),
            *agraph.get_ops()[1]);
    ASSERT_EQ(agraph.get_ops()[2]->get_input_value(0)->has_producer(), true);
    ASSERT_EQ(agraph.get_ops()[1]->get_input_value(0)->has_producer(), true);
}

TEST(graph_test, build_graph) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv2d")};
    op_t op1 {1, ReLU, std::string("relu")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    ASSERT_EQ(agraph.add_op(&op0), status::invalid_op);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.num_ops(), 1);

    ASSERT_EQ(agraph.build_graph(), status::success);
    ASSERT_EQ(agraph.num_ops(), 1);
    ASSERT_EQ(agraph.build_graph(), status::success);
    ASSERT_EQ(agraph.num_ops(), 1);
}

TEST(graph_test, run_pass) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv2d")};
    op_t op1 {1, ReLU, std::string("relu")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    ASSERT_EQ(agraph.add_op(&op0), status::invalid_op);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.num_ops(), 1);

    ASSERT_EQ(agraph.run_pass(partition_policy::fusion), status::invalid_graph);
    ASSERT_EQ(agraph.build_graph(), status::success);
    // ASSERT_EQ(agraph.run_pass(partition_policy::fusion), status::success);
}

TEST(graph_test, get_partitions) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv2d")};
    op_t op1 {1, ReLU, std::string("relu")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    ASSERT_EQ(agraph.add_op(&op0), status::invalid_op);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.num_ops(), 1);

    // ASSERT_EQ(agraph.build_graph(), status::success);
    // ASSERT_EQ(agraph.run_pass(partition_policy::fusion), status::success);
    // ASSERT_EQ(agraph.get_num_partitions(), 1);
}

TEST(graph_test, wildcard) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op {0, Wildcard, std::string("wildcard")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    op.add_input(t0);
    op.add_output(t1);
    ASSERT_EQ(agraph.add_op(&op), status::success);
    ASSERT_EQ(agraph.num_ops(), 1);
}
