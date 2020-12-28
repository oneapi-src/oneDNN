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

#include <vector>
#include "gtest/gtest.h"

#include <algorithm>
#include "backend/pass/pass_base.hpp"
#include "backend/pass/pass_manager.hpp"
#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "utils.hpp"

using node_ptr = std::unique_ptr<llga::impl::node_t>;

namespace {
llga::impl::pass::pass_base_ptr get_pass(const std::string &pass_name) {
    auto pm = llga::impl::pass::pass_manager();
    auto &passes = pm.get_passes();
    auto find = std::find_if(passes.begin(), passes.end(),
            [&pass_name](const llga::impl::pass::pass_base_ptr &p) -> bool {
                return p->get_pass_name() == pass_name;
            });

    return *find;
}
} // namespace

/**
 * 1. Query the registered conv_bn_fusion pass
 * 2. Test conv_bn_fusion pass name
 * 3. Create a graph with conv_bn pattern
 * 4. Pass the graph to the pass
 * 5. Check if conv_bn can be fused
 */
TEST(pass_test, conv_bn_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;
    node_t *in_node = agraph.create_node(Convolution);
    in_node->set_attr<int>("groups", 0);
    node_t *out_node = agraph.create_node(BatchNormInference);
    out_node->set_input(0, in_node, 0);

    pass::pass_base_ptr conv_bn_fusion_pass = get_pass("conv_bn_fusion");
    conv_bn_fusion_pass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bn);
    ASSERT_TRUE(fused_node->has_attr("groups"));
    ASSERT_EQ(fused_node->get_attr<int>("groups"), 0);
}

TEST(pass_test, conv_bias_sum_relu6_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;

    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Wildcard);
    node_t *node4 = agraph.create_node(Add);
    node_t *node5 = agraph.create_node(HardTanh);
    node5->set_attr<float>("min", 0);
    node5->set_attr<float>("max", 6);
    node2->set_input(0, node1, 0);
    node4->set_input(0, node2, 0);
    node4->set_input(1, node3, 0);
    node5->set_input(0, node4, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_relu6_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != conv_bias_add_relu6) {
        ASSERT_EQ(fnode2->get_op_kind(), conv_bias_add_relu6);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), conv_bias_add_relu6);
    }
}

TEST(pass_test, conv_bias_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;

    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(ReLU);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias_relu);
}

TEST(pass_test, conv_bias_relu6_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;

    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(HardTanh);
    node3->set_attr<float>("min", 0);
    node3->set_attr<float>("max", 6);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_relu6_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias_relu6);
}

TEST(pass_test, conv_bias_relu6_fusion_negative) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;

    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(HardTanh);
    node3->set_attr<float>("min", 0);
    node3->set_attr<float>("max", 5);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_relu6_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 3);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    const node_ptr &fnode3 = agraph.get_nodes()[2];
    ASSERT_EQ(fnode1->get_op_kind(), Convolution);
    ASSERT_EQ(fnode2->get_op_kind(), BiasAdd);
    ASSERT_EQ(fnode3->get_op_kind(), HardTanh);
}

TEST(pass_test, bn_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;

    node_t *node1 = agraph.create_node(BatchNormInference);
    node_t *node2 = agraph.create_node(ReLU);
    node2->set_input(0, node1, 0);

    pass::pass_base_ptr apass = get_pass("bn_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), bn_relu);
}

TEST(pass_test, bn_bwd_relu_bwd_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;

    node_t *node1 = agraph.create_node(ReLUBackprop);
    node_t *node2 = agraph.create_node(BatchNormTrainingBackprop);
    node2->set_input(0, node1, 0);

    pass::pass_base_ptr apass = get_pass("bn_bwd_relu_bwd_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), bn_bwd_relu_bwd);
}
/*
TEST(pass_test, bn_fwd_train_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;

    node_t *node1 = agraph.create_node(BatchNormForwardTraining);
    node_t *node2 = agraph.create_node(ReLU);
    node2->set_input(0, node1, 0);

    pass::pass_base_ptr apass = get_pass("bn_fwd_train_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), bn_fwd_train_relu);
}*/

TEST(pass_test, conv_bn_fusion_exception) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(Convolution);
    node_t *out_node_1 = agraph.create_node(BatchNormInference);
    node_t *out_node_2 = agraph.create_node(ReLU);
    out_node_1->set_input(0, in_node, 0);
    out_node_2->set_input(0, in_node, 1);

    pass::pass_base_ptr conv_bn_fusion_pass = get_pass("conv_bn_fusion");
    conv_bn_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 3);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    const node_ptr &unfused_node1 = agraph.get_nodes()[0];
    const node_ptr &unfused_node2 = agraph.get_nodes()[1];
    const node_ptr &unfused_node3 = agraph.get_nodes()[2];

    ASSERT_EQ(unfused_node1->get_op_kind(), Convolution);
    ASSERT_EQ(unfused_node2->get_op_kind(), BatchNormInference);
    ASSERT_EQ(unfused_node3->get_op_kind(), ReLU);
}

TEST(pass_test, bn_conv) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(BatchNormInference);
    node_t *out_node = agraph.create_node(Convolution);
    out_node->set_input(0, in_node, 0);

    pass::pass_base_ptr conv_bn_fusion_pass = get_pass("conv_bn_fusion");
    conv_bn_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    const node_ptr &unfused_node1 = agraph.get_nodes()[0];
    const node_ptr &unfused_node2 = agraph.get_nodes()[1];
    ASSERT_EQ(unfused_node2->get_input_node(0), unfused_node1.get());
    ASSERT_EQ(unfused_node1->get_output_node(0), unfused_node2.get());
}

TEST(pass_test, conv_bias_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;
    node_t *in_node = agraph.create_node(Convolution);
    in_node->set_attr<int>("groups", 0);
    node_t *out_node = agraph.create_node(BiasAdd);
    out_node->set_input(0, in_node, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias);
    ASSERT_TRUE(fused_node->has_attr("groups"));
    ASSERT_EQ(fused_node->get_attr<int>("groups"), 0);
}

TEST(pass_test, conv_bias_bn_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(BatchNormInference);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias_bn);
}

TEST(pass_test, conv_sum_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(Wildcard);
    node_t *node3 = agraph.create_node(Add);
    node_t *node4 = agraph.create_node(ReLU);
    node3->set_input(0, node1, 0);
    node3->set_input(1, node2, 0);
    node4->set_input(0, node3, 0);

    pass::pass_base_ptr apass = get_pass("conv_sum_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != conv_add_relu) {
        ASSERT_EQ(fnode2->get_op_kind(), conv_add_relu);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), conv_add_relu);
    }
}

TEST(pass_test, conv_bias_sum_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Wildcard);
    node_t *node4 = agraph.create_node(Add);
    node2->set_input(0, node1, 0);
    node4->set_input(0, node2, 0);
    node4->set_input(1, node3, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != conv_bias_add) {
        ASSERT_EQ(fnode2->get_op_kind(), conv_bias_add);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), conv_bias_add);
    }
}

TEST(pass_test, conv_sum_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    using namespace llga::tests::unit::utils;

    graph_t agraph;
    dnnl_graph_op conv {0, Convolution, "conv"};
    dnnl_graph_op relu {1, ReLU, "relu"};
    dnnl_graph_op add {2, Add, "add"};

    // create lt
    logical_tensor_t conv_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t conv_weight = logical_tensor_init(1, data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(2, data_type::f32);
    logical_tensor_t relu_data = logical_tensor_init(3, data_type::f32);
    logical_tensor_t relu_dst = logical_tensor_init(4, data_type::f32);
    logical_tensor_t add_dst = logical_tensor_init(5, data_type::f32);

    // set_attrs
    conv.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv.set_attr<std::string>("data_format", "NXC");
    conv.set_attr<std::string>("weight_format", "XIO");
    conv.set_attr<int64_t>("groups", 1);

    conv.add_input(&conv_data);
    conv.add_input(&conv_weight);
    conv.add_output(&conv_dst);
    relu.add_input(&relu_data);
    relu.add_output(&relu_dst);
    add.add_input(&conv_dst);
    add.add_input(&relu_dst);
    add.add_output(&add_dst);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);
}

TEST(pass_test, conv_sum_sum) {
    using namespace llga::impl;
    using namespace llga::impl::pass;
    using namespace llga::impl::op_kind;

    graph_t agraph;

    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(Wildcard);
    node_t *node3 = agraph.create_node(Add);
    node_t *node4 = agraph.create_node(Wildcard);
    node_t *node5 = agraph.create_node(Add);

    node3->set_input(0, node1, 0);
    node3->set_input(1, node2, 0);
    node5->set_input(1, node3, 0);
    node5->set_input(0, node4, 0);

    DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_sum_fusion)
            .set_priority(9.7f)
            .set_attr<FCreatePattern>("FCreatePattern",
                    [](pattern *apattern) -> void {
                        node_t *any1 = apattern->create_node(op_kind::any);
                        node_t *conv1
                                = apattern->create_node(op_kind::Convolution);
                        node_t *add1 = apattern->create_node(op_kind::Add);
                        node_t *any2 = apattern->create_node(op_kind::any);
                        node_t *add2 = apattern->create_node(op_kind::Add);
                        add1->set_input(0, any1, 0);
                        add1->set_input(1, conv1, 0);
                        add2->set_input(1, any2, 0);
                        add2->set_input(0, add1, 0);
                    })
            .set_attr<FCreateOptPattern>("FCreateOptPattern",
                    [](pattern *optimized_pattern) -> void {
                        node_t *fused_node = optimized_pattern->create_node(
                                op_kind::conv_add);
                        fused_node->set_attr<std::string>("backend", "dnnl");
                    });

    pass::pass_base_ptr apass = get_pass("conv_sum_sum_fusion");

    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 3);

    const node_ptr &fnode3 = agraph.get_nodes()[2];
    ASSERT_EQ(fnode3->get_op_kind(), conv_add);
}

TEST(pass_test, conv_bias_sum_sum) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;

    node_t *node3 = agraph.create_node(Wildcard);
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node4 = agraph.create_node(Add);
    node_t *node5 = agraph.create_node(Convolution);
    node_t *node6 = agraph.create_node(BiasAdd);
    node_t *node7 = agraph.create_node(Add);
    node2->set_input(0, node1, 0);
    node4->set_input(0, node3, 0);
    node4->set_input(1, node2, 0);
    node6->set_input(0, node5, 0);
    node7->set_input(0, node4, 0);
    node7->set_input(1, node6, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_fusion");

    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 3);

    const node_ptr &fnode2 = agraph.get_nodes()[1];
    const node_ptr &fnode3 = agraph.get_nodes()[2];
    ASSERT_EQ(fnode2->get_op_kind(), conv_bias_add);
    ASSERT_EQ(fnode3->get_op_kind(), conv_bias_add);
}

TEST(pass_test, conv_bias_sum_elu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Wildcard);
    node_t *node4 = agraph.create_node(Add);
    node_t *node5 = agraph.create_node(Elu);
    node2->set_input(0, node1, 0);
    node4->set_input(0, node2, 0);
    node4->set_input(1, node3, 0);
    node5->set_input(0, node4, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_elu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != conv_bias_add_elu) {
        ASSERT_EQ(fnode2->get_op_kind(), conv_bias_add_elu);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), conv_bias_add_elu);
    }
}

TEST(pass_test, conv_bias_sum_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Wildcard);
    node_t *node4 = agraph.create_node(Add);
    node_t *node5 = agraph.create_node(ReLU);
    node2->set_input(0, node1, 0);
    node4->set_input(0, node2, 0);
    node4->set_input(1, node3, 0);
    node5->set_input(0, node4, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != conv_bias_add_relu) {
        ASSERT_EQ(fnode2->get_op_kind(), conv_bias_add_relu);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), conv_bias_add_relu);
    }
}

TEST(pass_test, conv_bias_elu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Elu);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_elu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias_elu);
}

TEST(pass_test, conv_bias_sigmoid_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Sigmoid);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_sigmoid_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias_sigmoid);
}

TEST(pass_test, conv_bias_swish_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;

    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Sigmoid);
    node_t *node4 = agraph.create_node(Multiply);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);
    node4->set_input(0, node3, 0);
    node4->set_input(1, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_swish_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias_swish);
}

TEST(pass_test, conv_bias_hardtanh_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(HardTanh);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_hardtanh_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias_hardtanh);
}

TEST(pass_test, conv_bias_square_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Square);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_square_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias_square);
}

TEST(pass_test, conv_bias_tanh_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Tanh);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_tanh_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias_tanh);
}

TEST(pass_test, conv_bias_abs_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Abs);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_abs_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias_abs);
}

TEST(pass_test, conv_bias_sqrt_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Sqrt);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_sqrt_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bias_sqrt);
}

TEST(pass_test, conv_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(Convolution);
    node_t *out_node = agraph.create_node(ReLU);
    out_node->set_input(0, in_node, 0);
    pass::pass_base_ptr conv_relu_fusion_pass = get_pass("conv_relu_fusion");
    conv_relu_fusion_pass->run(agraph);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_relu);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
}

TEST(pass_test, conv_relu_fusion_exception) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(Convolution);
    node_t *out_node = agraph.create_node(ReLU);
    node_t *out_node_2 = agraph.create_node(ReLU);
    out_node->set_input(0, in_node, 0);
    out_node_2->set_input(0, in_node, 0);
    pass::pass_base_ptr conv_relu_fusion_pass = get_pass("conv_relu_fusion");
    conv_relu_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 3);
    ASSERT_EQ(in_node->num_outputs(), 2);
    ASSERT_EQ(agraph.get_num_partitions(), 0);
}

TEST(pass_test, matmul_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(MatMul);
    node_t *out_node = agraph.create_node(ReLU);
    out_node->set_input(0, in_node, 0);

    pass::pass_base_ptr matmul_relu_fusion_pass
            = get_pass("matmul_relu_fusion");
    matmul_relu_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), matmul_relu);
}

TEST(pass_test, matmul_elu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(MatMul);
    node_t *out_node = agraph.create_node(Elu);
    out_node->set_input(0, in_node, 0);

    pass::pass_base_ptr matmul_elu_fusion_pass = get_pass("matmul_elu_fusion");
    matmul_elu_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), matmul_elu);
}

TEST(pass_test, matmul_sigmoid_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(MatMul);
    node_t *out_node = agraph.create_node(Sigmoid);
    out_node->set_input(0, in_node, 0);

    pass::pass_base_ptr matmul_sigmoid_fusion_pass
            = get_pass("matmul_sigmoid_fusion");
    matmul_sigmoid_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), matmul_sigmoid);
}

TEST(pass_test, matmul_hardtanh_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(MatMul);
    node_t *out_node = agraph.create_node(HardTanh);
    out_node->set_input(0, in_node, 0);

    pass::pass_base_ptr matmul_hardtanh_fusion_pass
            = get_pass("matmul_hardtanh_fusion");
    matmul_hardtanh_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), matmul_hardtanh);
}

TEST(pass_test, matmul_gelu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(MatMul);
    node_t *out_node = agraph.create_node(GELU);
    out_node->set_input(0, in_node, 0);

    pass::pass_base_ptr matmul_gelu_fusion_pass
            = get_pass("matmul_gelu_fusion");
    matmul_gelu_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), matmul_gelu);
}

TEST(pass_test, matmul_sum_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node1 = agraph.create_node(MatMul);
    node_t *in_node2 = agraph.create_node(Wildcard);

    node_t *out_node = agraph.create_node(Add);
    out_node->set_input(0, in_node1, 0);
    out_node->set_input(1, in_node2, 0);

    pass::pass_base_ptr matmul_sum_fusion_pass = get_pass("matmul_sum_fusion");
    matmul_sum_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != matmul_add) {
        ASSERT_EQ(fnode2->get_op_kind(), matmul_add);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), matmul_add);
    }
}

TEST(pass_test, matmul_sum_fusion_opposite_order) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node1 = agraph.create_node(MatMul);
    node_t *in_node2 = agraph.create_node(Wildcard);

    node_t *out_node = agraph.create_node(Add);
    out_node->set_input(1, in_node1, 0);
    out_node->set_input(0, in_node2, 0);

    pass::pass_base_ptr matmul_sum_fusion_pass = get_pass("matmul_sum_fusion");
    matmul_sum_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != matmul_add) {
        ASSERT_EQ(fnode2->get_op_kind(), matmul_add);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), matmul_add);
    }
}

TEST(pass_test, matmul_sum_gelu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node1 = agraph.create_node(MatMul);
    node_t *in_node2 = agraph.create_node(Wildcard);

    node_t *add_node = agraph.create_node(Add);
    add_node->set_input(0, in_node1, 0);
    add_node->set_input(1, in_node2, 0);

    node_t *gelu_node = agraph.create_node(GELU);
    gelu_node->set_input(0, add_node, 0);

    pass::pass_base_ptr matmul_sum_gelu_fusion_pass
            = get_pass("matmul_sum_gelu_fusion");
    matmul_sum_gelu_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != matmul_add_gelu) {
        ASSERT_EQ(fnode2->get_op_kind(), matmul_add_gelu);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), matmul_add_gelu);
    }
}

TEST(pass_test, matmul_sum_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node1 = agraph.create_node(MatMul);
    node_t *in_node2 = agraph.create_node(Wildcard);

    node_t *add_node = agraph.create_node(Add);
    add_node->set_input(0, in_node1, 0);
    add_node->set_input(1, in_node2, 0);

    node_t *relu_node = agraph.create_node(ReLU);
    relu_node->set_input(0, add_node, 0);

    pass::pass_base_ptr matmul_sum_relu_fusion_pass
            = get_pass("matmul_sum_relu_fusion");
    matmul_sum_relu_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != matmul_add_relu) {
        ASSERT_EQ(fnode2->get_op_kind(), matmul_add_relu);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), matmul_add_relu);
    }
}

TEST(pass_test, conv_bwd_f_biasadd_bwd_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(ConvolutionBackpropFilters);
    node_t *node2 = agraph.create_node(BiasAddBackprop);
    node2->set_input(0, node1, 0);

    pass::pass_base_ptr apass = get_pass("conv_bwd_f_biasadd_bwd_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bwd_f_biasadd_bwd);
}

TEST(pass_test, relu_matmul) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(ReLU);
    node_t *out_node = agraph.create_node(MatMul);
    out_node->set_input(0, in_node, 0);

    pass::pass_base_ptr matmul_relu_fusion_pass
            = get_pass("matmul_relu_fusion");
    matmul_relu_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &unfused_node1 = agraph.get_nodes()[0];
    const node_ptr &unfused_node2 = agraph.get_nodes()[1];
    ASSERT_EQ(unfused_node2->get_input_node(0), unfused_node1.get());
    ASSERT_EQ(unfused_node1->get_output_node(0), unfused_node2.get());
}

TEST(pass_test, matmul_bias_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *in_node = agraph.create_node(MatMul);
    node_t *out_node = agraph.create_node(BiasAdd);
    out_node->set_input(0, in_node, 0);

    pass::pass_base_ptr apass = get_pass("matmul_bias_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), matmul_bias);
}

TEST(pass_test, matmul_bias_sigmoid_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(MatMul);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Sigmoid);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr matmul_bias_sigmoid_pass
            = get_pass("matmul_bias_sigmoid_fusion");
    matmul_bias_sigmoid_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fnode = agraph.get_nodes()[0];
    ASSERT_EQ(fnode->get_op_kind(), matmul_bias_sigmoid);
}

TEST(pass_test, matmul_bias_elu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(MatMul);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Elu);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr matmul_bias_elu_pass
            = get_pass("matmul_bias_elu_fusion");
    matmul_bias_elu_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), matmul_bias_elu);
}

TEST(pass_test, matmul_bias_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(MatMul);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(ReLU);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr matmul_bias_relu_pass
            = get_pass("matmul_bias_relu_fusion");
    matmul_bias_relu_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fnode = agraph.get_nodes()[0];
    ASSERT_EQ(fnode->get_op_kind(), matmul_bias_relu);
}

TEST(pass_test, matmul_bias_hardtanh_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(MatMul);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(HardTanh);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr matmul_bias_hardtanh_pass
            = get_pass("matmul_bias_hardtanh_fusion");
    matmul_bias_hardtanh_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fnode = agraph.get_nodes()[0];
    ASSERT_EQ(fnode->get_op_kind(), matmul_bias_hardtanh);
}

TEST(pass_test, matmul_bias_sum_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(MatMul);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Wildcard);
    node_t *node4 = agraph.create_node(Add);
    node2->set_input(0, node1, 0);
    node4->set_input(0, node2, 0);
    node4->set_input(1, node3, 0);

    pass::pass_base_ptr matmul_bias_sum_pass
            = get_pass("matmul_bias_sum_fusion");
    matmul_bias_sum_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != matmul_bias_add) {
        ASSERT_EQ(fnode2->get_op_kind(), matmul_bias_add);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), matmul_bias_add);
    }
}

TEST(pass_test, matmul_bias_sum_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(MatMul);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Wildcard);
    node_t *node4 = agraph.create_node(Add);
    node_t *node5 = agraph.create_node(ReLU);
    node2->set_input(0, node1, 0);
    node4->set_input(0, node2, 0);
    node4->set_input(1, node3, 0);
    node5->set_input(0, node4, 0);

    pass::pass_base_ptr matmul_bias_sum_relu_pass
            = get_pass("matmul_bias_sum_relu_fusion");
    matmul_bias_sum_relu_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != matmul_bias_add_relu) {
        ASSERT_EQ(fnode2->get_op_kind(), matmul_bias_add_relu);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), matmul_bias_add_relu);
    }
}

TEST(pass_test, matmul_bias_swish_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(MatMul);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(Sigmoid);
    node_t *node4 = agraph.create_node(Multiply);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);
    node4->set_input(0, node3, 0);
    node4->set_input(1, node2, 0);

    pass::pass_base_ptr matmul_bias_swish_pass
            = get_pass("matmul_bias_swish_fusion");
    matmul_bias_swish_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    ASSERT_EQ(fnode1->get_op_kind(), matmul_bias_swish);
}

TEST(pass_test, matmul_bias_bn_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(MatMul);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(BatchNormInference);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr matmul_bias_bn_pass = get_pass("matmul_bias_bn_fusion");
    matmul_bias_bn_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fnode = agraph.get_nodes()[0];
    ASSERT_EQ(fnode->get_op_kind(), matmul_bias_bn);
}

TEST(pass_test, matmul_bias_relu6_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(MatMul);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(HardTanh);
    node3->set_attr<float>("min", 0);
    node3->set_attr<float>("max", 6);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr apass = get_pass("matmul_bias_relu6_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), matmul_bias_relu6);
}

TEST(pass_test, conv_bn_sum_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BatchNormInference);
    node_t *node3 = agraph.create_node(Wildcard);
    node_t *node4 = agraph.create_node(Add);
    node2->set_input(0, node1, 0);
    node4->set_input(0, node2, 0);
    node4->set_input(1, node3, 0);

    pass::pass_base_ptr conv_bn_sum_fusion_pass
            = get_pass("conv_bn_sum_fusion");
    conv_bn_sum_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != conv_bn_add) {
        ASSERT_EQ(fnode2->get_op_kind(), conv_bn_add);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), conv_bn_add);
    }
}

TEST(pass_test, conv_bn_sum_fusion_exception) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BatchNormInference);
    node_t *node3 = agraph.create_node(Wildcard);
    node_t *node4 = agraph.create_node(Add);
    node_t *node5 = agraph.create_node(MaxPool);
    node2->set_input(0, node1, 0);
    node4->set_input(0, node2, 0);
    node4->set_input(1, node3, 0);
    node5->set_input(0, node2, 0);

    pass::pass_base_ptr conv_bn_sum_fusion_pass
            = get_pass("conv_bn_sum_fusion");
    conv_bn_sum_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 5);
}

TEST(pass_test, conv_bias_bn_sum_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(BatchNormInference);
    node_t *node4 = agraph.create_node(Wildcard);
    node_t *node5 = agraph.create_node(Add);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);
    node5->set_input(0, node3, 0);
    node5->set_input(1, node4, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != conv_bias_bn_add) {
        ASSERT_EQ(fnode2->get_op_kind(), conv_bias_bn_add);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), conv_bias_bn_add);
    }
}

TEST(pass_test, conv_bn_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BatchNormInference);
    node_t *node3 = agraph.create_node(ReLU);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);

    pass::pass_base_ptr conv_bn_relu_fusion_pass
            = get_pass("conv_bn_relu_fusion");
    conv_bn_relu_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    ASSERT_EQ(fnode1->get_op_kind(), conv_bn_relu);
}

TEST(pass_test, conv_bias_bn_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(BatchNormInference);
    node_t *node4 = agraph.create_node(ReLU);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);
    node4->set_input(0, node3, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    ASSERT_EQ(fnode1->get_op_kind(), conv_bias_bn_relu);
}

TEST(pass_test, conv_bn_sum_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BatchNormInference);
    node_t *node3 = agraph.create_node(Wildcard);
    node_t *node4 = agraph.create_node(Add);
    node_t *node5 = agraph.create_node(ReLU);
    node2->set_input(0, node1, 0);
    node4->set_input(0, node2, 0);
    node4->set_input(1, node3, 0);
    node5->set_input(0, node4, 0);

    pass::pass_base_ptr conv_bn_sum_relu_fusion_pass
            = get_pass("conv_bn_sum_relu_fusion");
    conv_bn_sum_relu_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != conv_bn_add_relu) {
        ASSERT_EQ(fnode2->get_op_kind(), conv_bn_add_relu);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), conv_bn_add_relu);
    }
}

TEST(pass_test, conv_bias_bn_sum_relu_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(BiasAdd);
    node_t *node3 = agraph.create_node(BatchNormInference);
    node_t *node4 = agraph.create_node(Wildcard);
    node_t *node5 = agraph.create_node(Add);
    node_t *node6 = agraph.create_node(ReLU);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);
    node5->set_input(0, node3, 0);
    node5->set_input(1, node4, 0);
    node6->set_input(0, node5, 0);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_sum_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != conv_bias_bn_add_relu) {
        ASSERT_EQ(fnode2->get_op_kind(), conv_bias_bn_add_relu);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), conv_bias_bn_add_relu);
    }
}
/*
TEST(pass_test, layernorm_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Reshape);
    node_t *node2 = agraph.create_node(BatchNormForwardTraining);
    node_t *node3 = agraph.create_node(Reshape);
    node_t *node4 = agraph.create_node(Multiply);
    node_t *node5 = agraph.create_node(Wildcard);
    node_t *node6 = agraph.create_node(Add);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);
    node4->set_input(0, node3, 0);
    node6->set_input(0, node4, 0);
    node6->set_input(1, node5, 0);

    pass::pass_base_ptr apass = get_pass("layernorm_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);

    const node_ptr &fnode1 = agraph.get_nodes()[0];
    const node_ptr &fnode2 = agraph.get_nodes()[1];
    if (fnode1->get_op_kind() != LayerNorm) {
        ASSERT_EQ(fnode2->get_op_kind(), LayerNorm);
    } else {
        ASSERT_EQ(fnode1->get_op_kind(), LayerNorm);
    }
}*/

TEST(pass_test, gelu_erf_based_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    graph_t agraph;
    node_t *node1 = agraph.create_node(Wildcard);
    node_t *node2 = agraph.create_node(Divide);
    node_t *node3 = agraph.create_node(Erf);
    node_t *node4 = agraph.create_node(Wildcard);
    node_t *node5 = agraph.create_node(Add);
    node_t *node6 = agraph.create_node(Wildcard);
    node_t *node7 = agraph.create_node(Multiply);
    node_t *node8 = agraph.create_node(Wildcard);
    node_t *node9 = agraph.create_node(Multiply);
    node2->set_input(0, node1, 0);
    node3->set_input(0, node2, 0);
    node5->set_input(0, node3, 0);
    node5->set_input(1, node4, 0);
    node7->set_input(0, node5, 0);
    node7->set_input(1, node6, 0);
    node9->set_input(0, node7, 0);
    node9->set_input(1, node8, 0);

    pass::pass_base_ptr apass = get_pass("gelu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 5);
}

TEST(pass_test, gelu_erf_based_tensor_input_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    using namespace llga::tests::unit::utils;

    graph_t agraph;
    dnnl_graph_op divide {0, Divide, std::string("divide")};
    dnnl_graph_op erf {1, Erf, std::string("erf")};
    dnnl_graph_op add {2, Add, std::string("add")};
    dnnl_graph_op multiply_1 {3, Multiply, std::string("multiply")};
    dnnl_graph_op multiply_2 {4, Multiply, std::string("multiply")};

    // create lt
    logical_tensor_t divide_in_a_tensor
            = logical_tensor_init(0, data_type::f32);
    logical_tensor_t divide_in_b_tensor
            = logical_tensor_init(1, data_type::f32);
    logical_tensor_t divide_dst = logical_tensor_init(2, data_type::f32);
    logical_tensor_t erf_dst = logical_tensor_init(3, data_type::f32);
    logical_tensor_t add_in_tensor = logical_tensor_init(4, data_type::f32);
    logical_tensor_t add_dst = logical_tensor_init(5, data_type::f32);
    logical_tensor_t multiply_1_in_tensor
            = logical_tensor_init(6, data_type::f32);
    logical_tensor_t multiply_1_dst = logical_tensor_init(7, data_type::f32);
    logical_tensor_t multiply_2_in_tensor
            = logical_tensor_init(8, data_type::f32);
    logical_tensor_t multiply_2_dst = logical_tensor_init(9, data_type::f32);

    divide.add_input(&divide_in_a_tensor);
    divide.add_input(&divide_in_b_tensor);
    divide.add_output(&divide_dst);
    erf.add_input(&divide_dst);
    erf.add_output(&erf_dst);
    add.add_input(&erf_dst);
    add.add_input(&add_in_tensor);
    add.add_output(&add_dst);
    multiply_1.add_input(&add_dst);
    multiply_1.add_input(&multiply_1_in_tensor);
    multiply_1.add_output(&multiply_1_dst);
    multiply_2.add_input(&multiply_1_dst);
    multiply_2.add_input(&multiply_2_in_tensor);
    multiply_2.add_output(&multiply_2_dst);

    ASSERT_EQ(agraph.add_op(&divide), status::success);
    ASSERT_EQ(agraph.add_op(&erf), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&multiply_1), status::success);
    ASSERT_EQ(agraph.add_op(&multiply_2), status::success);

    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 5);

    pass::pass_base_ptr apass = get_pass("gelu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);
}

struct ut_gelu_params {
    size_t node4_idx;
    size_t node5_idx;
    size_t node9_idx;
    size_t node10_idx;
};

class gelu_test : public ::testing::TestWithParam<ut_gelu_params> {};

TEST_P(gelu_test, gelu_tanh_based_fusion) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    const ut_gelu_params &params = GetParam();

    graph_t agraph;
    node_t *node1 = agraph.create_node(Wildcard);
    node_t *node2 = agraph.create_node(Pow);
    node_t *node3 = agraph.create_node(Wildcard);
    node_t *node4 = agraph.create_node(Multiply);
    node_t *node5 = agraph.create_node(Wildcard);
    node_t *node6 = agraph.create_node(Add);
    node_t *node7 = agraph.create_node(Wildcard);
    node_t *node8 = agraph.create_node(Multiply);
    node_t *node9 = agraph.create_node(Tanh);
    node_t *node10 = agraph.create_node(Wildcard);
    node_t *node11 = agraph.create_node(Add);
    node_t *node12 = agraph.create_node(Wildcard);
    node_t *node13 = agraph.create_node(Multiply);
    node_t *node14 = agraph.create_node(Wildcard);
    node_t *node15 = agraph.create_node(Multiply);
    node2->set_input(0, node1, 0);
    node4->set_input(0, node2, 0);
    node4->set_input(1, node3, 0);
    node6->set_input(params.node4_idx, node4, 0);
    node6->set_input(params.node5_idx, node5, 0);
    node8->set_input(0, node6, 0);
    node8->set_input(1, node7, 0);
    node9->set_input(0, node8, 0);
    node11->set_input(params.node9_idx, node9, 0);
    node11->set_input(params.node10_idx, node10, 0);
    node13->set_input(0, node11, 0);
    node13->set_input(1, node12, 0);
    node15->set_input(0, node13, 0);
    node15->set_input(1, node14, 0);

    pass::pass_base_ptr gelu_pass = get_pass("gelu_fusion");
    gelu_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 8);
}

// Parameters below correspond to the indices of the node inputs for Add ops.
INSTANTIATE_TEST_SUITE_P(gelu_test_instance, gelu_test,
        testing::Values(ut_gelu_params {/*multiply_1*/ 0, /*any_2*/ 1,
                                /*tanh*/ 0, /*any_4*/ 1},
                ut_gelu_params {0, 1, 1, 0}, ut_gelu_params {1, 0, 0, 1},
                ut_gelu_params {1, 0, 1, 0}));

TEST(pass_test, single_node_replacement) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;

    pass::pass_manager pm;
    std::vector<op_kind_t> single_node_set = {Convolution, BatchNormInference,
            Add, ReLU, MatMul, AvgPool, MaxPool, AvgPoolBackprop,
            BatchNormTrainingBackprop, ConvolutionBackpropData,
            ConvolutionBackpropFilters, MaxPoolBackprop, ReLUBackprop,
            GELUBackprop, LogSoftmax, LogSoftmaxBackprop, SoftMax, LayerNorm};
    for (auto akind : single_node_set) {
        graph_t agraph;
        node_t *node = agraph.create_node(akind);
        ASSERT_EQ(node->get_op_kind(), akind);
        pm.run_passes(agraph, "no_config");

        const node_ptr &replaced_node = agraph.get_nodes()[0];
        ASSERT_EQ(replaced_node->get_op_kind(), akind);
        ASSERT_EQ(replaced_node->has_attr("backend"), true);
        ASSERT_EQ(replaced_node->get_attr<std::string>("backend"), "dnnl");
    }
}

TEST(pass_test, merge_logical_tensor) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    using namespace llga::tests::unit::utils;
    // tensor dim
    int32_t N = 3, // batch size
            IC = 32, // input channels
            IH = 13, // input height
            IW = 13, // input width
            OC = 64, // output channels
            KH = 3, // weights height
            KW = 3, // weights width
            PH_L = 1, // height padding: left
            PH_R = 1, // height padding: right
            PW_L = 1, // width padding: left
            PW_R = 1, // width padding: right
            SH = 4, // height-wise stride
            SW = 4, // width-wise stride
            OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
            OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

    graph_t agraph;
    dnnl_graph_op op0 {0, Convolution, std::string("conv2d")};
    op0.set_attr("strides", std::vector<int64_t> {4, 4});
    op0.set_attr("pads_begin", std::vector<int64_t> {0, 0});
    op0.set_attr("pads_end", std::vector<int64_t> {0, 0});
    op0.set_attr("dilations", std::vector<int64_t> {1, 1});
    op0.set_attr("data_format", std::string("NCX"));
    op0.set_attr("filter_format", std::string("OIX"));
    dnnl_graph_op op1 {1, BatchNormInference, std::string("bn")};
    op1.set_attr("epsilon", 0.001f);
    // create conv inputs tensor
    logical_tensor_t conv_data
            = logical_tensor_init(0, {N, IC, IH, IW}, data_type::f32);
    logical_tensor_t conv_weight
            = logical_tensor_init(1, {OC, IC, KH, KW}, data_type::f32);
    // create conv inputs tensor
    logical_tensor_t conv_dst
            = logical_tensor_init(4, {N, OC, OH, OW}, data_type::f32);
    op0.add_input(&conv_data);
    op0.add_input(&conv_weight);
    op0.add_output(&conv_dst);
    //create bn inputs tensor
    logical_tensor_t bn_scale
            = logical_tensor_init(5, std::vector<dim_t> {OC}, data_type::f32);
    logical_tensor_t bn_shift
            = logical_tensor_init(6, std::vector<dim_t> {OC}, data_type::f32);
    logical_tensor_t bn_mean = logical_tensor_init(7, data_type::f32);
    logical_tensor_t bn_variance = logical_tensor_init(8, data_type::f32);
    logical_tensor_t bn_dst
            = logical_tensor_init(9, {N, OC, OH, OW}, data_type::f32);
    //create bn outputs tensor
    op1.add_input(&conv_dst);
    op1.add_input(&bn_scale);
    op1.add_input(&bn_shift);
    op1.add_input(&bn_mean);
    op1.add_input(&bn_variance);
    op1.add_output(&bn_dst);
    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);
    for (auto &node : agraph.get_nodes()) {
        if (node->get_op_kind() == Convolution) {
            ASSERT_EQ(node->num_inputs_tensor(), 2);
            ASSERT_EQ(node->num_outputs_tensor(), 1);
        } else if (node->get_op_kind() == BatchNormInference) {
            ASSERT_EQ(node->num_inputs_tensor(), 5);
            ASSERT_EQ(node->num_outputs_tensor(), 1);
        }
    }
    pass::pass_base_ptr conv_bn_fusion_pass = get_pass("conv_bn_fusion");
    conv_bn_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    const node_ptr &fused_node = agraph.get_nodes()[0];
    ASSERT_EQ(fused_node->get_op_kind(), conv_bn);
    ASSERT_EQ(fused_node->num_inputs_tensor(), 6);
    ASSERT_EQ(fused_node->num_outputs_tensor(), 1);
    ASSERT_EQ(fused_node->get_input_tensor_map().size(), 6);
    ASSERT_EQ(fused_node->get_output_tensor_map().size(), 1);
}

TEST(pass_test, save_load_json) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    pass::pass_manager pm;
    pm.print_passes("passes.json");
    graph_t agraph;
    node_t *node1 = agraph.create_node(Convolution);
    node_t *node2 = agraph.create_node(Convolution);
    node_t *node3 = agraph.create_node(BatchNormInference);
    node_t *node4 = agraph.create_node(ReLU);
    node_t *node5 = agraph.create_node(Add);
    node3->set_input(0, node2, 0);
    node4->set_input(0, node3, 0);
    node5->set_input(0, node4, 0);
    node5->set_input(1, node1, 0);
    pm.run_passes(agraph, "passes.json");
    ASSERT_EQ(agraph.num_nodes(), 2);
    ASSERT_EQ(agraph.get_num_partitions(), 2);
}

TEST(pass_test, add_with_two_inputnode) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    using namespace llga::tests::unit::utils;
    // tensor dim
    int32_t N = 3, // batch size
            IC = 32, // input channels
            IH = 13, // input height
            IW = 13, // input width
            OC = 64, // output channels
            KH = 3, // weights height
            KW = 3, // weights width
            PH_L = 1, // height padding: left
            PH_R = 1, // height padding: right
            PW_L = 1, // width padding: left
            PW_R = 1, // width padding: right
            SH = 4, // height-wise stride
            SW = 4, // width-wise stride
            OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
            OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

    graph_t agraph;
    dnnl_graph_op conv0 {0, Convolution, std::string("conv0")};
    conv0.set_attr("strides", std::vector<int64_t> {4, 4});
    conv0.set_attr("pads_begin", std::vector<int64_t> {0, 0});
    conv0.set_attr("pads_end", std::vector<int64_t> {0, 0});
    conv0.set_attr("dilations", std::vector<int64_t> {1, 1});
    conv0.set_attr("data_format", std::string("NCX"));
    conv0.set_attr("filter_format", std::string("OIX"));
    dnnl_graph_op conv1 {1, Convolution, std::string("conv1")};
    conv1.set_attr("strides", std::vector<int64_t> {4, 4});
    conv1.set_attr("pads_begin", std::vector<int64_t> {0, 0});
    conv1.set_attr("pads_end", std::vector<int64_t> {0, 0});
    conv1.set_attr("dilations", std::vector<int64_t> {1, 1});
    conv1.set_attr("data_format", std::string("NXC"));
    conv1.set_attr("filter_format", std::string("OIX"));
    dnnl_graph_op bn {2, BatchNormInference, std::string("bn")};
    bn.set_attr("epsilon", 0.001f);
    dnnl_graph_op add {3, Add, std::string("add")};

    // create conv0 inputs tensor
    logical_tensor_t conv0_data
            = logical_tensor_init(0, {N, IC, IH, IW}, data_type::f32);
    logical_tensor_t conv0_weight
            = logical_tensor_init(1, {OC, IC, KH, KW}, data_type::f32);
    // create conv0 outputs tensor
    logical_tensor_t conv0_dst
            = logical_tensor_init(2, {N, OC, OH, OW}, data_type::f32);

    conv0.add_input(&conv0_data);
    conv0.add_input(&conv0_weight);
    conv0.add_output(&conv0_dst);
    // create conv1 inputs tensor
    logical_tensor_t conv1_data
            = logical_tensor_init(3, {N, IC, IH, IW}, data_type::f32);
    logical_tensor_t conv1_weight
            = logical_tensor_init(4, {OC, IC, KH, KW}, data_type::f32);
    // create conv1 outputs tensor
    logical_tensor_t conv1_dst
            = logical_tensor_init(5, {N, OC, OH, OW}, data_type::f32);

    conv1.add_input(&conv1_data);
    conv1.add_input(&conv1_weight);
    conv1.add_output(&conv1_dst);

    //create bn inputs tensor
    logical_tensor_t bn_scale
            = logical_tensor_init(6, std::vector<dim_t> {OC}, data_type::f32);
    logical_tensor_t bn_shift
            = logical_tensor_init(7, std::vector<dim_t> {OC}, data_type::f32);
    logical_tensor_t bn_mean = logical_tensor_init(8, data_type::f32);
    logical_tensor_t bn_variance = logical_tensor_init(9, data_type::f32);
    //create bn outputs tensor
    logical_tensor_t bn_dst
            = logical_tensor_init(10, {N, OC, OH, OW}, data_type::f32);

    bn.add_input(&conv0_dst);
    bn.add_input(&bn_scale);
    bn.add_input(&bn_shift);
    bn.add_input(&bn_mean);
    bn.add_input(&bn_variance);
    bn.add_output(&bn_dst);

    // create add outputs tensor
    logical_tensor_t add_dst
            = logical_tensor_init(11, {N, OC, OH, OW}, data_type::f32);

    add.add_input(&bn_dst);
    add.add_input(&conv1_dst);
    add.add_output(&add_dst);

    ASSERT_EQ(agraph.add_op(&conv0), status::success);
    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 4);

    pass::pass_base_ptr conv_add_fusion_pass = get_pass("conv_bn_sum_fusion");
    pass::pass_base_ptr conv_pass = get_pass("conv_pass");

    conv_add_fusion_pass->run(agraph);
    conv_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);
    ASSERT_EQ(agraph.get_num_partitions(), 2);

    for (auto &node : agraph.get_nodes()) {
        if (node->get_op_kind() == conv_bn_add) {
            ASSERT_EQ(node->num_inputs_tensor(), 7);
            ASSERT_EQ(node->num_outputs_tensor(), 1);
        } else if (node->get_op_kind() == Convolution) {
            ASSERT_EQ(node->num_inputs_tensor(), 2);
            ASSERT_EQ(node->num_outputs_tensor(), 1);
        }
    }
}

TEST(pass_test, two_conv_with_shared_weight) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    using namespace llga::tests::unit::utils;

    graph_t agraph;
    dnnl_graph_op conv0 {0, Convolution, std::string("conv0")};
    conv0.set_attr("strides", std::vector<int64_t> {1, 1});
    conv0.set_attr("pads_begin", std::vector<int64_t> {0, 0});
    conv0.set_attr("pads_end", std::vector<int64_t> {0, 0});
    conv0.set_attr("dilations", std::vector<int64_t> {1, 1});
    conv0.set_attr("data_format", std::string("NCX"));
    conv0.set_attr("filter_format", std::string("OIX"));
    dnnl_graph_op relu0 {1, ReLU, std::string("relu0")};
    dnnl_graph_op conv1 {2, Convolution, std::string("conv1")};
    conv1.set_attr("strides", std::vector<int64_t> {1, 1});
    conv1.set_attr("pads_begin", std::vector<int64_t> {0, 0});
    conv1.set_attr("pads_end", std::vector<int64_t> {0, 0});
    conv1.set_attr("dilations", std::vector<int64_t> {1, 1});
    conv1.set_attr("data_format", std::string("NXC"));
    conv1.set_attr("filter_format", std::string("OIX"));
    dnnl_graph_op relu1 {3, ReLU, std::string("relu1")};

    // create conv0 inputs tensor
    logical_tensor_t conv0_data
            = logical_tensor_init(0, {3, 32, 13, 13}, data_type::f32);
    logical_tensor_t conv0_weight
            = logical_tensor_init(1, {32, 32, 3, 3}, data_type::f32);
    // create conv0 outputs tensor
    logical_tensor_t conv0_dst
            = logical_tensor_init(2, {3, 32, 11, 11}, data_type::f32);

    conv0.add_input(&conv0_data);
    conv0.add_input(&conv0_weight);
    conv0.add_output(&conv0_dst);

    logical_tensor_t relu0_dst
            = logical_tensor_init(3, {3, 32, 11, 11}, data_type::f32);
    relu0.add_input(&conv0_dst);
    relu0.add_output(&relu0_dst);

    // create conv1 inputs tensor
    logical_tensor_t conv1_dst
            = logical_tensor_init(4, {3, 32, 9, 9}, data_type::f32);
    conv1.add_input(&relu0_dst);
    conv1.add_input(&conv0_weight);
    conv1.add_output(&conv1_dst);

    //create bn inputs tensor
    logical_tensor_t relu1_dst
            = logical_tensor_init(5, {3, 32, 9, 9}, data_type::f32);
    relu1.add_input(&conv1_dst);
    relu1.add_output(&relu1_dst);

    ASSERT_EQ(agraph.add_op(&conv0), status::success);
    ASSERT_EQ(agraph.add_op(&relu0), status::success);
    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&relu1), status::success);

    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 4);

    pass::pass_base_ptr conv_relu_fusion_pass = get_pass("conv_relu_fusion");

    conv_relu_fusion_pass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 2);
    ASSERT_EQ(agraph.get_num_partitions(), 2);

    for (auto &node : agraph.get_nodes()) {
        if (node->get_op_kind() == conv_relu) {
            ASSERT_EQ(node->num_inputs_tensor(), 2);
            ASSERT_EQ(node->num_outputs_tensor(), 1);
        }
    }
}

TEST(pass_test, add_with_tensor_input) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    using namespace llga::tests::unit::utils;

    graph_t agraph;
    dnnl_graph_op conv {0, Convolution, std::string("conv")};
    conv.set_attr("strides", std::vector<int64_t> {4, 4});
    conv.set_attr("pads_begin", std::vector<int64_t> {0, 0});
    conv.set_attr("pads_end", std::vector<int64_t> {0, 0});
    conv.set_attr("dilations", std::vector<int64_t> {1, 1});
    conv.set_attr("data_format", std::string("NCX"));
    conv.set_attr("filter_format", std::string("OIX"));
    dnnl_graph_op bias {1, BiasAdd, std::string("bias")};
    dnnl_graph_op add {2, Add, std::string("add")};

    // create lt
    logical_tensor_t conv_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t conv_weight = logical_tensor_init(1, data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(2, data_type::f32);
    logical_tensor_t bias_input_tensor = logical_tensor_init(3, data_type::f32);
    logical_tensor_t bias_dst = logical_tensor_init(4, data_type::f32);
    logical_tensor_t add_input_tensor = logical_tensor_init(5, data_type::f32);
    logical_tensor_t add_dst = logical_tensor_init(6, data_type::f32);

    conv.add_input(&conv_data);
    conv.add_input(&conv_weight);
    conv.add_output(&conv_dst);
    bias.add_input(&conv_dst);
    bias.add_input(&bias_input_tensor);
    bias.add_output(&bias_dst);
    add.add_input(&bias_dst);
    add.add_input(&add_input_tensor);
    add.add_output(&add_dst);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_add);
}

TEST(pass_test, convolution_with_tensor_shape_check) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    using namespace llga::impl::pass;
    using namespace llga::tests::unit::utils;

    graph_t agraph;
    std::vector<dnnl_graph_op> conv;
    conv.emplace_back(0, Convolution, std::string("conv0"));
    conv.emplace_back(1, Convolution, std::string("conv1"));
    conv.emplace_back(2, Convolution, std::string("conv2"));

    for (auto &convi : conv) {
        convi.set_attr("strides", std::vector<int64_t> {1, 1});
        convi.set_attr("pads_begin", std::vector<int64_t> {0, 0});
        convi.set_attr("pads_end", std::vector<int64_t> {0, 0});
        convi.set_attr("dilations", std::vector<int64_t> {1, 1});
        convi.set_attr("data_format", std::string("NCX"));
        convi.set_attr("weight_format", std::string("OIX"));
    }

    // create conv0 inputs tensor
    logical_tensor_t conv0_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t conv0_weight
            = logical_tensor_init(1, {32, 32, 3, 3}, data_type::f32);
    // create conv0 outputs tensor
    logical_tensor_t conv0_dst = logical_tensor_init(2, data_type::f32);

    conv[0].add_input(&conv0_data);
    conv[0].add_input(&conv0_weight);
    conv[0].add_output(&conv0_dst);

    // create conv1 inputs tensor
    logical_tensor_t conv1_data = logical_tensor_init(3, data_type::f32);
    logical_tensor_t conv1_weight
            = logical_tensor_init(4, {32, 64, 32, 32, 3, 3}, data_type::f32);
    // create conv1 outputs tensor
    logical_tensor_t conv1_dst = logical_tensor_init(5, data_type::f32);

    conv[1].add_input(&conv1_data);
    conv[1].add_input(&conv1_weight);
    conv[1].add_output(&conv1_dst);

    // create conv2 inputs tensor with unknown ndims;
    logical_tensor_t conv2_data = logical_tensor_init(6, data_type::f32);
    logical_tensor_t conv2_weight = logical_tensor_init(7, data_type::f32);

    // create conv2 outputs tensor
    logical_tensor_t conv2_dst = logical_tensor_init(8, data_type::f32);

    conv[2].add_input(&conv2_data);
    conv[2].add_input(&conv2_weight);
    conv[2].add_output(&conv2_dst);

    ASSERT_EQ(agraph.add_op(&conv[0]), status::success);
    ASSERT_EQ(agraph.add_op(&conv[1]), status::success);
    ASSERT_EQ(agraph.add_op(&conv[2]), status::success);

    DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_with_requirement_pass)
            .set_priority(8.0f)
            .set_attr<FCreatePattern>("FCreatePattern",
                    [](pattern *apattern) -> void {
                        node_t *conv
                                = apattern->create_node(op_kind::Convolution);
                        //check if ndims of weight is supported by dnnl
                        conv->set_attr<FRequirement>(
                                "FRequirement", [](node_t *graph_node) -> bool {
                                    auto weight = logical_tensor_wrapper(
                                            graph_node->get_input_tensor(1));
                                    std::set<int> right_ndims {-1, 4, 5};
                                    if (right_ndims.count(weight.ndims()))
                                        return true;
                                    return false;
                                });
                    })
            .set_attr<FCreateOptPattern>("FCreateOptPattern",
                    [](pattern *optimized_pattern) -> void {
                        node_t *anode = optimized_pattern->create_node(
                                op_kind::Convolution);
                        anode->set_attr<std::string>("backend", "dnnl");
                    });

    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr conv_pass = get_pass("conv_with_requirement_pass");
    conv_pass->run(agraph);

    //conv1 don't meet the requirement of oneDNN
    ASSERT_EQ(agraph.get_num_partitions(), 2);
}

TEST(pass_test, multi_values_between_two_nodes) {
    using namespace llga::impl;
    using namespace llga::impl::op_kind;
    using namespace llga::tests::unit::utils;

    graph_t agraph;
    dnnl_graph_op conv {0, Convolution, std::string("conv")};
    conv.set_attr("strides", std::vector<int64_t> {4, 4});
    conv.set_attr("pads_begin", std::vector<int64_t> {0, 0});
    conv.set_attr("pads_end", std::vector<int64_t> {0, 0});
    conv.set_attr("dilations", std::vector<int64_t> {1, 1});
    conv.set_attr("data_format", std::string("NCX"));
    conv.set_attr("filter_format", std::string("OIX"));
    dnnl_graph_op add {1, Add, std::string("add")};

    // create lt
    logical_tensor_t conv_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t conv_weight = logical_tensor_init(1, data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(2, data_type::f32);
    logical_tensor_t add_dst = logical_tensor_init(3, data_type::f32);

    conv.add_input(&conv_data);
    conv.add_input(&conv_weight);
    conv.add_output(&conv_dst);
    add.add_input(&conv_dst);
    add.add_input(&conv_dst);
    add.add_output(&add_dst);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);
    pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(agraph);
    apass = get_pass("sum_pass");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 2);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), Convolution);
    ASSERT_EQ(agraph.get_nodes()[0]->num_outputs_tensor(), 1);
    ASSERT_EQ(agraph.get_nodes()[1]->get_op_kind(), Add);
    ASSERT_EQ(agraph.get_nodes()[1]->num_inputs_tensor(), 2);
}
