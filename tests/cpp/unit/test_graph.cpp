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
#include "interface/logical_tensor.hpp"
#include "interface/value.hpp"
#include "utils.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/fake/fake_backend.hpp"

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

    /*
    ASSERT_EQ(agraph.run_pass(partition_policy::fusion), status::invalid_graph);
    ASSERT_EQ(agraph.build_graph(), status::success);
    ASSERT_EQ(agraph.run_pass(partition_policy::fusion), status::success);
    */
}

TEST(graph_test, get_partitions) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t conv {0, op_kind::Convolution, std::string("conv2d")};
    op_t relu {1, op_kind::ReLU, std::string("relu")};
    op_t end {2, op_kind::End, std::string("end")};
    logical_tensor_t conv_src = logical_tensor_init(0, data_type::f32);
    logical_tensor_t conv_wei = logical_tensor_init(1, data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(2, data_type::f32);
    logical_tensor_t relu_dst = logical_tensor_init(3, data_type::f32);
    conv.add_input(conv_src);
    conv.add_input(conv_wei);
    conv.add_output(conv_dst);
    conv.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv.set_attr<int64_t>("groups", 1);
    conv.set_attr<std::string>("data_format", "NCX");
    conv.set_attr<std::string>("filter_format", "OIX");
    relu.add_input(conv_dst);
    relu.add_output(relu_dst);
    end.add_input(relu_dst);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&end), status::success);
    ASSERT_EQ(agraph.num_ops(), 3);

    ASSERT_EQ(agraph.build_graph(), status::success);
    auto &dnnl_bkd = dnnl_impl::dnnl_backend::get_singleton();
    dnnl_bkd.get_partitions(agraph, partition_policy::fusion);
    auto &fake_bkd = fake_impl::fake_backend::get_singleton();
    fake_bkd.get_partitions(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 2);
    auto p1 = agraph.get_partitions()[0].get();
    ASSERT_NE(p1->get_assigned_backend()->get_name(),
            std::string("fake_backend"));
    auto p2 = agraph.get_partitions()[1].get();
    ASSERT_EQ(p2->get_assigned_backend()->get_name(),
            std::string("fake_backend"));
}

TEST(graph_test, get_partitions_fake) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t wildcard {0, op_kind::Wildcard, std::string("wildcard")};
    op_t index {1, op_kind::Index, std::string("index")};
    op_t end {2, op_kind::End, std::string("end")};
    logical_tensor_t input = logical_tensor_init(0, data_type::f32);
    logical_tensor_t wildcard_out = logical_tensor_init(1, data_type::f32);
    logical_tensor_t index_in_2 = logical_tensor_init(2, data_type::f32);
    logical_tensor_t index_out = logical_tensor_init(3, data_type::f32);

    wildcard.add_input(input);
    wildcard.add_output(wildcard_out);
    index.add_input(wildcard_out);
    index.add_input(index_in_2);
    index.add_output(index_out);
    end.add_input(index_out);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);
    ASSERT_EQ(agraph.add_op(&index), status::success);
    ASSERT_EQ(agraph.add_op(&end), status::success);
    ASSERT_EQ(agraph.num_ops(), 3);

    auto &bkd = fake_impl::fake_backend::get_singleton();
    bkd.get_partitions(agraph, partition_policy::fusion);
    ASSERT_EQ(agraph.get_num_partitions(), 3);
    auto partition = agraph.get_partitions()[0].get();
    ASSERT_EQ(partition->get_assigned_backend()->get_name(),
            std::string("fake_backend"));
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

TEST(graph_test, get_input_output_edges) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;
    using ltw = dnnl::graph::impl::logical_tensor_wrapper;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv0")};
    op_t op1 {1, Add, std::string("add0")};
    op_t op2 {2, ReLU, std::string("relu0")};
    op_t op3 {3, Wildcard, std::string("wildcard0")};

    op0.set_attr<std::vector<int64_t>>("strides", {4, 4});
    op0.set_attr<std::vector<int64_t>>("pads_begin", {111, 111});
    op0.set_attr<std::vector<int64_t>>("pads_end", {111, 111});
    op0.set_attr<std::string>("auto_pad", "VALID");
    op0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    op0.set_attr<std::string>("data_format", "NCX");
    op0.set_attr<std::string>("filter_format", "OIX");
    op0.set_attr<int64_t>("groups", 1);

    // prepare logical tensor
    logical_tensor_t src = logical_tensor_init(0, impl::data_type::f32);
    logical_tensor_t weight = logical_tensor_init(1, impl::data_type::f32);
    logical_tensor_t bias = logical_tensor_init(2, impl::data_type::f32);
    logical_tensor_t other = logical_tensor_init(3, impl::data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(4, impl::data_type::f32);
    logical_tensor_t add_dst = logical_tensor_init(5, impl::data_type::f32);
    logical_tensor_t dst = logical_tensor_init(6, impl::data_type::f32);

    op0.add_input(src);
    op0.add_input(weight);
    op0.add_input(bias);
    op0.add_output(conv_dst);

    op1.add_input(conv_dst);
    op1.add_input(other);
    op1.add_output(add_dst);

    op2.add_input(add_dst);
    op2.add_output(dst);

    op3.add_input(conv_dst);

    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    ASSERT_EQ(agraph.add_op(&op3), status::success);
    ASSERT_EQ(agraph.num_ops(), 4);
    agraph.build_graph();

    auto ops = agraph.get_ops();
    ops.pop_back();
    graph_t subgraph(ops);
    ASSERT_EQ(subgraph.num_ops(), 3);

    auto in_vals = subgraph.get_input_values();
    ASSERT_EQ(in_vals.size(), 4);
    ASSERT_EQ(ltw(in_vals[0]->get_logical_tensor()), ltw(src));
    ASSERT_EQ(ltw(in_vals[1]->get_logical_tensor()), ltw(weight));
    ASSERT_EQ(ltw(in_vals[2]->get_logical_tensor()), ltw(bias));
    ASSERT_EQ(ltw(in_vals[3]->get_logical_tensor()), ltw(other));

    auto out_vals = subgraph.get_output_values();
    ASSERT_EQ(out_vals.size(), 2);
    logical_tensor_t out_lt1 = out_vals[0]->get_logical_tensor();
    logical_tensor_t out_lt2 = out_vals[1]->get_logical_tensor();
    if (out_lt1.id == 6)
        ASSERT_EQ(out_lt2.id, 4);
    else if (out_lt1.id == 4)
        ASSERT_EQ(out_lt2.id, 6);
    else
        ASSERT_TRUE(false);
}

TEST(graph_pass_test, infer_shape) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;
    using ltw = dnnl::graph::impl::logical_tensor_wrapper;

    std::vector<int64_t> src_shape {8, 3, 227, 227};
    std::vector<int64_t> weight_shape {96, 3, 11, 11};
    std::vector<int64_t> bias_shape {96};
    std::vector<int64_t> dst_shape {8, 96, 55, 55};

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv0")};
    op_t op1 {1, Add, std::string("add0")};
    op_t op2 {2, ReLU, std::string("relu0")};

    op0.set_attr<std::vector<int64_t>>("strides", {4, 4});
    op0.set_attr<std::vector<int64_t>>("pads_begin", {111, 111});
    op0.set_attr<std::vector<int64_t>>("pads_end", {111, 111});
    op0.set_attr<std::string>("auto_pad", "VALID");
    op0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    op0.set_attr<std::string>("data_format", "NCX");
    op0.set_attr<std::string>("filter_format", "OIX");
    op0.set_attr<int64_t>("groups", 1);

    // prepare logical tensor
    logical_tensor_t src
            = logical_tensor_init(0, src_shape, impl::data_type::f32);
    logical_tensor_t weight
            = logical_tensor_init(1, weight_shape, impl::data_type::f32);
    logical_tensor_t bias
            = logical_tensor_init(2, bias_shape, impl::data_type::f32);
    logical_tensor_t other
            = logical_tensor_init(3, dst_shape, impl::data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(4, impl::data_type::f32);
    logical_tensor_t add_dst = logical_tensor_init(5, impl::data_type::f32);
    logical_tensor_t dst = logical_tensor_init(6, impl::data_type::f32);

    op0.add_input(src);
    op0.add_input(weight);
    op0.add_input(bias);
    op0.add_output(conv_dst);

    op1.add_input(conv_dst);
    op1.add_input(other);
    op1.add_output(add_dst);

    op2.add_input(add_dst);
    op2.add_output(dst);

    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    ASSERT_EQ(agraph.num_ops(), 3);
    agraph.build_graph();

    auto in_vals = agraph.get_input_values();
    ASSERT_EQ(in_vals.size(), 4);
    ASSERT_EQ(ltw(in_vals[0]->get_logical_tensor()), ltw(src));
    ASSERT_EQ(ltw(in_vals[1]->get_logical_tensor()), ltw(weight));
    ASSERT_EQ(ltw(in_vals[2]->get_logical_tensor()), ltw(bias));
    ASSERT_EQ(ltw(in_vals[3]->get_logical_tensor()), ltw(other));

    ASSERT_EQ(agraph.infer_shape(), status::success);

    auto out_vals = agraph.get_output_values();
    ASSERT_EQ(out_vals.size(), 1);
    logical_tensor_t out_lt = out_vals[0]->get_logical_tensor();
    ASSERT_EQ(out_lt.id, 6);
    ASSERT_EQ(out_lt.ndims, 4);
    ASSERT_EQ(out_lt.dims[0], 8);
    ASSERT_EQ(out_lt.dims[1], 96);
    ASSERT_EQ(out_lt.dims[2], 55);
    ASSERT_EQ(out_lt.dims[3], 55);
}
