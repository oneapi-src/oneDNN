/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "interface/common.hpp"
#include "interface/graph.hpp"
#include "interface/ir.hpp"

#include "utils.hpp"

/**
 * 1. Create a dnnl::graph::impl::node_t object
 * 2. Validate if dnnl::graph::impl::node_t has expected contents
 */
TEST(ir_test, node_op) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    node_t cur_node_conv2d = node_t(Convolution);
    node_t cur_node_relu = node_t(ReLU);
    node_t cur_node_matmul = node_t(MatMul);

    ASSERT_EQ(cur_node_conv2d.get_op_kind(), Convolution);
    ASSERT_EQ(cur_node_relu.get_op_kind(), ReLU);
    ASSERT_EQ(cur_node_matmul.get_op_kind(), MatMul);
}

TEST(ir_test, node_inputs) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    node_t in_node = node_t(Convolution);
    node_t cur_node = node_t(Convolution);
    cur_node.set_input(0, &in_node, 0);

    ASSERT_EQ(cur_node.num_inputs(), 1);
    ASSERT_EQ(cur_node.get_input_node(0), &in_node);
}

TEST(ir_test, node_outputs) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    node_t cur_node = node_t(Convolution);
    node_t out_node = node_t(Convolution);
    out_node.set_input(0, &cur_node, 0);

    ASSERT_EQ(cur_node.num_outputs(), 1);
    ASSERT_EQ(cur_node.get_output_node(0), &out_node);
}

TEST(ir_test, node_input_offset) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    node_t in_node = node_t(Convolution);
    node_t cur_node = node_t(Convolution);

    cur_node.set_input(0, &in_node, 0);
    ASSERT_EQ(cur_node.get_input_offset(0), 0);

    cur_node.set_input(0, &in_node, 1);
    ASSERT_EQ(cur_node.get_input_offset(0), 1);
}

TEST(ir_test, node_attrs) {
    using namespace dnnl::graph::impl;

    op_t op(kAdd, "kAdd");
    constexpr int64_t int_attr_value {123};
    op.set_attr("value", int_attr_value);

    node_t cur_node = node_t(op.kind());
    cur_node.parse_op_attr(&op);
    cur_node.set_attr<bool>("is_inplace", true);

    ASSERT_EQ(cur_node.get_attr<int64_t>("value"), int_attr_value);
    ASSERT_EQ(cur_node.get_attr<bool>("is_inplace"), true);
    ASSERT_EQ(cur_node.num_attrs(), 2);
}

TEST(ir_test, cmp_attrs) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    node_t node_a = node_t(Convolution);
    node_t node_b = node_t(Convolution);
    node_t node_c = node_t(Convolution);

    node_a.set_attr<int64_t>("groups", 123);
    node_b.set_attr<int64_t>("groups", 123);
    node_c.set_attr<float>("groups", 123);
    ASSERT_EQ(node_a.is_same_attr_value(node_b, "groups"), true);
    ASSERT_EQ(node_a.is_same_attr_value(node_c, "groups"), false);
    node_b.set_attr<int64_t>("groups", 125);
    ASSERT_EQ(node_a.is_same_attr_value(node_b, "groups"), false);

    node_a.set_attr<std::string>("name", "node_a");
    node_b.set_attr<std::string>("name", "node_a");
    ASSERT_EQ(node_a.is_same_attr_value(node_b, "name"), true);
    node_b.set_attr<std::string>("name", "node_b");
    ASSERT_EQ(node_a.is_same_attr_value(node_b, "name"), false);

    node_a.set_attr<std::vector<int64_t>>("strides", {1, 2});
    node_b.set_attr<std::vector<int64_t>>("strides", {1, 2});
    ASSERT_EQ(node_a.is_same_attr_value(node_b, "strides"), true);
    node_b.set_attr<std::vector<int64_t>>("strides", {1, 2, 3});
    ASSERT_EQ(node_a.is_same_attr_value(node_b, "strides"), false);

    node_a.set_attr<bool>("is_inplace", true);
    node_b.set_attr<bool>("is_inplace", true);
    ASSERT_EQ(node_a.is_same_attr_value(node_b, "is_inplace"), true);
    node_b.set_attr<bool>("is_inplace", false);
    ASSERT_EQ(node_a.is_same_attr_value(node_b, "is_inplace"), false);
}

TEST(ir_test, merge_attrs) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    node_t cur_node = node_t(Convolution);
    node_t next_node = node_t(BatchNormInference);
    cur_node.set_attr<bool>("conv_attr", true);
    cur_node.set_attr<bool>("duplicated_attr", true);
    next_node.set_attr<bool>("bn_attr", true);
    next_node.set_attr<bool>("duplicated_attr", false);
    node_t fused_node = node_t(conv_bn);
    fused_node.merge_attrs_map(cur_node.get_attrs_map());
    fused_node.merge_attrs_map(next_node.get_attrs_map());

    ASSERT_EQ(fused_node.get_attr<bool>("conv_attr"), true);
    ASSERT_EQ(fused_node.get_attr<bool>("bn_attr"), true);
    ASSERT_EQ(fused_node.get_attr<bool>("duplicated_attr"), true);
}

TEST(ir_test, graph_create_node) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(Convolution);
    node_t *out_node = agraph.create_node(Convolution);
    out_node->set_input(0, in_node, 0);

    ASSERT_EQ(agraph.num_nodes(), 2);
    ASSERT_EQ(agraph.get_inputs().size(), 1);
    ASSERT_EQ(agraph.get_outputs().size(), 1);
}

TEST(ir_test, graph_delete_node) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(Convolution);
    node_t *out_node = agraph.create_node(Convolution);
    out_node->set_input(0, in_node, 0);

    ASSERT_EQ(agraph.num_nodes(), 2);
    agraph.delete_node(out_node);
    ASSERT_EQ(agraph.num_nodes(), 1);
}

TEST(ir_test, add_op) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv2d")};
    op_t op1 {1, ReLU, std::string("relu")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(&t0);
    op0.add_output(&t1);
    op1.add_input(&t1);
    op1.add_output(&t2);
    ASSERT_EQ(agraph.add_op(&op0), status::invalid_op);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 1);
}

TEST(ir_test, wildcard) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op {0, Wildcard, std::string("wildcard")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    op.add_input(&t0);
    op.add_output(&t1);
    ASSERT_EQ(agraph.add_op(&op), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 1);
}
