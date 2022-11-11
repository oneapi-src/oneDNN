/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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
#include <string>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/logical_tensor.hpp"
#include "interface/value.hpp"

#include "cpp/unit/utils.hpp"

TEST(Graph, Create) {
    using namespace dnnl::graph::impl;

    graph_t g_default_engine;
    ASSERT_EQ(g_default_engine.get_engine_kind(), engine_kind::cpu);

    graph_t g_gpu {engine_kind::gpu};
    ASSERT_EQ(g_gpu.get_engine_kind(), engine_kind::gpu);
}

TEST(Graph, AddOp) {
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
    ASSERT_EQ(agraph.num_ops(), 1U);

    auto ret = agraph.get_ops()[0];
    ASSERT_EQ(*ret, op1);
}

TEST(Graph, FailAddOpWithInvalidAttrValue) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv0")};

    op0.set_attr<std::vector<int64_t>>(op_attr::strides, {4, 4});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_begin, {111, 111});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_end, {111, 111});
    op0.set_attr<std::string>(op_attr::auto_pad, "VALID");
    op0.set_attr<std::vector<int64_t>>(op_attr::dilations, {1, 1});
    op0.set_attr<std::string>(op_attr::data_format, "NCX");
    op0.set_attr<std::string>(op_attr::filter_format, "OIX");
    op0.set_attr<int64_t>(op_attr::groups, 1);

    // prepare logical tensor
    logical_tensor_t src = logical_tensor_init(0, impl::data_type::f32);
    logical_tensor_t weight = logical_tensor_init(1, impl::data_type::f32);
    logical_tensor_t bias = logical_tensor_init(2, impl::data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(4, impl::data_type::f32);

    op0.add_input(src);
    op0.add_input(weight);
    op0.add_input(bias);
    op0.add_output(conv_dst);

    ASSERT_EQ(agraph.add_op(&op0), status::success);

    op0.set_attr<std::string>(op_attr::filter_format, "IOX");
    graph_t agraph1;
    ASSERT_EQ(agraph1.add_op(&op0), status::invalid_op);
}

TEST(Graph, AddNullOp) {
    using namespace dnnl::graph::impl;

    graph_t agraph;
    ASSERT_EQ(agraph.add_op(nullptr), status::invalid_op);
    ASSERT_EQ(agraph.num_ops(), 0U);
    ASSERT_EQ(agraph.get_ops().size(), 0U);
}

TEST(Graph, DeleteOp) {
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
    ASSERT_EQ(agraph.num_ops(), 1U);
    ASSERT_EQ(agraph.get_ops().size(), 1U);

    agraph.delete_op(&op1);
    ASSERT_EQ(agraph.num_ops(), 0U);
    ASSERT_EQ(agraph.get_ops().size(), 0U);
}

TEST(Graph, GetOutputOps) {
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
    ASSERT_EQ(agraph.num_ops(), 2U);
    agraph.build_graph();
    ASSERT_EQ(agraph.get_output_ops().size(), 1U);
    ASSERT_EQ(*(agraph.get_output_ops()[0]), op1);
}

TEST(Graph, GetOutputOps2) {
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
    ASSERT_EQ(agraph.num_ops(), 3U);
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

TEST(Graph, BuildGraph) {
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
    ASSERT_EQ(agraph.num_ops(), 1U);

    ASSERT_EQ(agraph.build_graph(), status::success);
    ASSERT_EQ(agraph.num_ops(), 1U);
    ASSERT_EQ(agraph.build_graph(), status::success);
    ASSERT_EQ(agraph.num_ops(), 1U);
}

TEST(Graph, InvalidOp) {
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
    ASSERT_EQ(agraph.num_ops(), 1U);

    /*
    ASSERT_EQ(agraph.run_pass(partition_policy::fusion), status::invalid_graph);
    ASSERT_EQ(agraph.build_graph(), status::success);
    ASSERT_EQ(agraph.run_pass(partition_policy::fusion), status::success);
    */
}

TEST(Graph, Wildcard) {
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
    ASSERT_EQ(agraph.num_ops(), 1U);
}

TEST(Graph, GetInputOutputEdges) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;
    using ltw = dnnl::graph::impl::logical_tensor_wrapper_t;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv0")};
    op_t op1 {1, Add, std::string("add0")};
    op_t op2 {2, ReLU, std::string("relu0")};
    op_t op3 {3, Wildcard, std::string("wildcard0")};
    op_t op4 {4, Wildcard, std::string("wildcard1")};
    op_t op5 {5, Wildcard, std::string("wildcard2")};

    op0.set_attr<std::vector<int64_t>>(op_attr::strides, {4, 4});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_begin, {111, 111});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_end, {111, 111});
    op0.set_attr<std::string>(op_attr::auto_pad, "VALID");
    op0.set_attr<std::vector<int64_t>>(op_attr::dilations, {1, 1});
    op0.set_attr<std::string>(op_attr::data_format, "NCX");
    op0.set_attr<std::string>(op_attr::filter_format, "OIX");
    op0.set_attr<int64_t>(op_attr::groups, 1);

    // prepare logical tensor
    logical_tensor_t src = logical_tensor_init(0, impl::data_type::f32);
    logical_tensor_t weight = logical_tensor_init(1, impl::data_type::f32);
    logical_tensor_t bias = logical_tensor_init(2, impl::data_type::f32);
    logical_tensor_t other = logical_tensor_init(3, impl::data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(4, impl::data_type::f32);
    logical_tensor_t add_dst = logical_tensor_init(5, impl::data_type::f32);
    logical_tensor_t dst = logical_tensor_init(6, impl::data_type::f32);

    logical_tensor_t wild_val = logical_tensor_init(7, impl::data_type::f32);

    op4.add_output(src);
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
    op3.add_output(wild_val);

    op5.add_input(wild_val);

    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    ASSERT_EQ(agraph.add_op(&op3), status::success);
    ASSERT_EQ(agraph.num_ops(), 4U);
    agraph.build_graph();

    auto ops = agraph.get_ops();
    ops.pop_back();
    graph_t subgraph(ops);
    ASSERT_EQ(subgraph.num_ops(), 3U);

    auto in_vals = subgraph.get_input_values();
    ASSERT_EQ(in_vals.size(), 4U);
    ASSERT_EQ(ltw(in_vals[0]->get_logical_tensor()), ltw(src));
    ASSERT_EQ(ltw(in_vals[1]->get_logical_tensor()), ltw(weight));
    ASSERT_EQ(ltw(in_vals[2]->get_logical_tensor()), ltw(bias));
    ASSERT_EQ(ltw(in_vals[3]->get_logical_tensor()), ltw(other));

    auto out_vals = subgraph.get_output_values();
    ASSERT_EQ(out_vals.size(), 2U);
    logical_tensor_t out_lt1 = out_vals[0]->get_logical_tensor();
    logical_tensor_t out_lt2 = out_vals[1]->get_logical_tensor();
    if (out_lt1.id == 6)
        ASSERT_EQ(out_lt2.id, 4U);
    else if (out_lt1.id == 4)
        ASSERT_EQ(out_lt2.id, 6U);
    else
        ASSERT_TRUE(false);
}

TEST(Graph, InferShape) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;
    using ltw = dnnl::graph::impl::logical_tensor_wrapper_t;

    std::vector<int64_t> src_shape {8, 3, 227, 227};
    std::vector<int64_t> weight_shape {96, 3, 11, 11};
    std::vector<int64_t> bias_shape {96};
    std::vector<int64_t> dst_shape {8, 96, 55, 55};

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv0")};
    op_t op1 {1, Add, std::string("add0")};
    op_t op2 {2, ReLU, std::string("relu0")};

    op0.set_attr<std::vector<int64_t>>(op_attr::strides, {4, 4});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_begin, {111, 111});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_end, {111, 111});
    op0.set_attr<std::string>(op_attr::auto_pad, "VALID");
    op0.set_attr<std::vector<int64_t>>(op_attr::dilations, {1, 1});
    op0.set_attr<std::string>(op_attr::data_format, "NCX");
    op0.set_attr<std::string>(op_attr::filter_format, "OIX");
    op0.set_attr<int64_t>(op_attr::groups, 1);

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
    ASSERT_EQ(agraph.num_ops(), 3U);
    agraph.build_graph();

    auto in_vals = agraph.get_input_values();
    ASSERT_EQ(in_vals.size(), 4U);
    ASSERT_EQ(ltw(in_vals[0]->get_logical_tensor()), ltw(src));
    ASSERT_EQ(ltw(in_vals[1]->get_logical_tensor()), ltw(weight));
    ASSERT_EQ(ltw(in_vals[2]->get_logical_tensor()), ltw(bias));
    ASSERT_EQ(ltw(in_vals[3]->get_logical_tensor()), ltw(other));

    ASSERT_EQ(agraph.infer_shape(), status::success);

    auto out_vals = agraph.get_output_values();
    ASSERT_EQ(out_vals.size(), 1U);
    logical_tensor_t out_lt = out_vals[0]->get_logical_tensor();
    ASSERT_EQ(out_lt.id, 6U);
    ASSERT_EQ(out_lt.ndims, 4);
    ASSERT_EQ(out_lt.dims[0], 8);
    ASSERT_EQ(out_lt.dims[1], 96);
    ASSERT_EQ(out_lt.dims[2], 55);
    ASSERT_EQ(out_lt.dims[3], 55);
}

TEST(Graph, Rewrite) {
    /*
          mm
          |
         relu
          |  \
          mm  mm
          |
         relu
    */
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    impl::op_t mm0(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t relu1(1, impl::op_kind::ReLU, "relu_op");
    impl::op_t mm2(2, impl::op_kind::MatMul, "matmul_op");
    impl::op_t relu3(3, impl::op_kind::ReLU, "relu_op");
    impl::op_t mm4(4, impl::op_kind::MatMul, "matmul_op");

    // prepare logical tensor
    impl::logical_tensor_t mm0_src
            = logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t mm0_weight
            = logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t mm0_dst
            = logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t relu1_dst
            = logical_tensor_init(3, {1, 1}, impl::data_type::f32);

    impl::logical_tensor_t mm2_weight
            = logical_tensor_init(4, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t mm2_dst
            = logical_tensor_init(5, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t relu3_dst
            = logical_tensor_init(6, {1, 1}, impl::data_type::f32);

    impl::logical_tensor_t mm4_weight
            = logical_tensor_init(7, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t mm4_dst
            = logical_tensor_init(8, {1, 1}, impl::data_type::f32);

    mm0.add_input(mm0_src);
    mm0.add_input(mm0_weight);
    mm0.add_output(mm0_dst);
    relu1.add_input(mm0_dst);
    relu1.add_output(relu1_dst);
    mm2.add_input(relu1_dst);
    mm2.add_input(mm2_weight);
    mm2.add_output(mm2_dst);
    relu3.add_input(mm2_dst);
    relu3.add_output(relu3_dst);
    mm4.add_input(relu1_dst);
    mm4.add_input(mm4_weight);
    mm4.add_output(mm4_dst);

    impl::graph_t g(impl::engine_kind::cpu);
    ASSERT_EQ(g.add_op(&mm0), impl::status::success);
    ASSERT_EQ(g.add_op(&relu1), impl::status::success);
    ASSERT_EQ(g.add_op(&mm2), impl::status::success);
    ASSERT_EQ(g.add_op(&relu3), impl::status::success);
    ASSERT_EQ(g.add_op(&mm4), impl::status::success);

    g.build_graph();
    auto all_ops = g.get_ops();

    std::vector<impl::op_t *> fusion_ops = {all_ops[0].get(), all_ops[1].get(),
            all_ops[2].get(), all_ops[3].get()};

    impl::rewrite(g, {fusion_ops});
    ASSERT_EQ(g.num_ops(), 2U);

    auto fused_op = g.get_ops()[1];
    // The inputs are mm0_src, mm0_weight, mm2_weight
    ASSERT_EQ(fused_op->num_inputs(), 3U);
    // The outputs are mm0_dst, mm2_dst
    ASSERT_EQ(fused_op->num_outputs(), 2U);
}

TEST(Graph, SetFpmathMode) {
    ASSERT_EQ(impl::get_default_fpmath_mode(), impl::fpmath_mode::strict);

    impl::graph_t graph;
    ASSERT_EQ(graph.get_fpmath_mode(), impl::fpmath_mode::strict);

    for (auto m : {impl::fpmath_mode::strict, impl::fpmath_mode::bf16,
                 impl::fpmath_mode::f16, impl::fpmath_mode::any}) {
        impl::graph_t graph2 {impl::engine_kind::cpu, m};
        ASSERT_EQ(graph2.get_fpmath_mode(), m);
    }
}
