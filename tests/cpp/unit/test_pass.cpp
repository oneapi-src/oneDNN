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

using namespace dnnl::graph::impl;
using namespace dnnl::graph::impl::op_kind;
using namespace dnnl::graph::tests::unit::utils;

using node_ptr = std::unique_ptr<dnnl::graph::impl::node_t>;

namespace {
dnnl::graph::impl::pass::pass_base_ptr get_pass(const std::string &pass_name) {
    auto pm = dnnl::graph::impl::pass::pass_manager();
    auto &passes = pm.get_passes();
    auto find = std::find_if(passes.begin(), passes.end(),
            [&pass_name](const dnnl::graph::impl::pass::pass_base_ptr &p)
                    -> bool { return p->get_pass_name() == pass_name; });

    return *find;
}

std::vector<logical_tensor_t> create_logical_tensors(size_t num_lt) {
    size_t count = 0;
    std::vector<logical_tensor_t> lt_vec;
    lt_vec.reserve(num_lt);
    while (count < num_lt) {
        lt_vec.emplace_back(logical_tensor_init(count, data_type::f32));
        count++;
    }
    return lt_vec;
}

void set_conv_common_attr(op_t &conv, std::vector<int64_t> strides = {1, 1},
        std::vector<int64_t> pads_begin = {0, 0},
        std::vector<int64_t> pads_end = {0, 0},
        std::vector<int64_t> dilations = {1, 1},
        std::string data_format = "NXC", std::string filter_format = "XIO",
        int64_t groups = 1, std::string auto_pad = "None") {
    conv.set_attr("strides", strides);
    conv.set_attr("pads_begin", pads_begin);
    conv.set_attr("pads_end", pads_end);
    conv.set_attr("dilations", dilations);
    conv.set_attr("data_format", data_format);
    conv.set_attr("filter_format", filter_format);
    conv.set_attr("groups", groups);
    conv.set_attr("auto_pad", auto_pad);
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
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(8);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bn.add_input(&lt_vec[2]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_output(&lt_vec[7]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), Convolution);
    ASSERT_EQ(agraph.get_nodes()[0]->num_inputs_tensor(), 2);
    ASSERT_EQ(agraph.get_nodes()[0]->num_outputs_tensor(), 1);
    ASSERT_EQ(agraph.get_nodes()[1]->get_op_kind(), BatchNormInference);
    ASSERT_EQ(agraph.get_nodes()[1]->num_inputs_tensor(), 5);
    ASSERT_EQ(agraph.get_nodes()[1]->num_outputs_tensor(), 1);

    pass::pass_base_ptr apass = get_pass("conv_bn_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bn);
    ASSERT_EQ(agraph.get_nodes()[0]->num_inputs_tensor(), 6);
    ASSERT_EQ(agraph.get_nodes()[0]->num_outputs_tensor(), 1);
}

TEST(pass_test, conv_bn_fusion_fail) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(9);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); //conv with bias
    conv.add_output(&lt_vec[3]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_input(&lt_vec[7]);
    bn.add_output(&lt_vec[8]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bn_fusion");
    apass->run(agraph);
    // conv with bias cannot be fused via conv_bn_fusion pass
    ASSERT_EQ(agraph.num_nodes(), 2);
}

TEST(pass_test, conv_bn_fusion_fail_case2) {
    /*   conv
        /    \
       bn   relu
    */
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t relu {2, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(9);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bn.add_input(&lt_vec[2]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_output(&lt_vec[7]);
    relu.add_input(&lt_vec[2]);
    relu.add_output(&lt_vec[8]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bn_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 3);
}

TEST(pass_test, conv_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {1, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    relu.add_input(&lt_vec[2]);
    relu.add_output(&lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_relu);
}

TEST(pass_test, conv_relu_fusion_fail) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {1, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    relu.add_input(&lt_vec[3]);
    relu.add_output(&lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 2);
}

TEST(pass_test, conv_relu_fusion_fail_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu1 {1, ReLU, "relu"};
    op_t relu2 {2, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    relu1.add_input(&lt_vec[3]);
    relu1.add_output(&lt_vec[4]);
    relu2.add_input(&lt_vec[3]);
    relu2.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu1), status::success);
    ASSERT_EQ(agraph.add_op(&relu2), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 3);
}

TEST(pass_test, conv_bias_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
}

TEST(pass_test, conv_bias_fusion_fail) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    bias.add_input(&lt_vec[3]);
    bias.add_input(&lt_vec[4]);
    bias.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_fusion");
    apass->run(agraph);
    // conv with bias cannot be fused via conv_bias_fusion pass
    ASSERT_EQ(agraph.num_nodes(), 2);
}

TEST(pass_test, conv_sum_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    add.add_input(&lt_vec[2]);
    add.add_input(&lt_vec[3]);
    add.add_output(&lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
}

TEST(pass_test, conv_sum_fusion_fail) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    add.add_input(&lt_vec[3]);
    add.add_input(&lt_vec[4]);
    add.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 2);
}

TEST(pass_test, conv_bias_bn_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t bn {2, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(10);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_input(&lt_vec[7]);
    bn.add_input(&lt_vec[8]);
    bn.add_output(&lt_vec[9]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_bn);
}

TEST(pass_test, conv_bias_bn_fusion_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(9);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_input(&lt_vec[7]);
    bn.add_output(&lt_vec[8]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_bn);
}

TEST(pass_test, conv_bias_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t relu {2, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    relu.add_input(&lt_vec[4]);
    relu.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_relu);
}

TEST(pass_test, conv_bias_relu_fusion_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {1, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]);
    conv.add_output(&lt_vec[3]);
    relu.add_input(&lt_vec[3]);
    relu.add_output(&lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_relu);
}

TEST(pass_test, conv_bias_relu6_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t hardtanh {2, HardTanh, "hardtanh"};
    hardtanh.set_attr("min", 0.f);
    hardtanh.set_attr("max", 6.f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    hardtanh.add_input(&lt_vec[4]);
    hardtanh.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&hardtanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_relu6_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_relu6);
}

TEST(pass_test, conv_bias_relu6_fusion_fail) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t hardtanh {2, HardTanh, "hardtanh"};
    hardtanh.set_attr("min", 0.f);
    hardtanh.set_attr("max", 5.f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    hardtanh.add_input(&lt_vec[4]);
    hardtanh.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&hardtanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_relu6_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 3);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), Convolution);
    ASSERT_EQ(agraph.get_nodes()[1]->get_op_kind(), BiasAdd);
    ASSERT_EQ(agraph.get_nodes()[2]->get_op_kind(), HardTanh);
}

TEST(pass_test, conv_bias_elu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t elu {1, Elu, "elu"};
    elu.set_attr("alpha", 0.1f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    elu.add_input(&lt_vec[3]);
    elu.add_output(&lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&elu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_elu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_elu);
}

TEST(pass_test, conv_bias_sigmoid_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t sigmoid {1, Sigmoid, "sigmoid"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    sigmoid.add_input(&lt_vec[3]);
    sigmoid.add_output(&lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_sigmoid_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_sigmoid);
}

TEST(pass_test, conv_bias_swish_fusion) {
    // swish: f(x) = x * sigmoid(x)
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t sigmoid {1, Sigmoid, "sigmoid"};
    op_t multiply {2, Multiply, "multiply"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    sigmoid.add_input(&lt_vec[3]);
    sigmoid.add_output(&lt_vec[4]);
    multiply.add_input(&lt_vec[4]);
    multiply.add_input(&lt_vec[3]);
    multiply.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_swish_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_swish);
}

TEST(pass_test, conv_bias_hardtanh_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t hardtanh {2, HardTanh, "hardtanh"};
    hardtanh.set_attr("min", 0.f);
    hardtanh.set_attr("max", 100.f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    hardtanh.add_input(&lt_vec[4]);
    hardtanh.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&hardtanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_hardtanh_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_hardtanh);
}

TEST(pass_test, conv_bias_square_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t square {2, Square, "square"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    square.add_input(&lt_vec[4]);
    square.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&square), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_square_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_square);
}

TEST(pass_test, conv_bias_tanh_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t tanh {2, Tanh, "tanh"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    tanh.add_input(&lt_vec[4]);
    tanh.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&tanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_tanh_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_tanh);
}

TEST(pass_test, conv_bias_abs_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t abs {2, Abs, "abs"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    abs.add_input(&lt_vec[4]);
    abs.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&abs), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_abs_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_abs);
}

TEST(pass_test, conv_bias_sqrt_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t sqrt {2, Sqrt, "sqrt"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    sqrt.add_input(&lt_vec[4]);
    sqrt.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&sqrt), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_sqrt_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_sqrt);
}

TEST(pass_test, conv_bias_sum_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t add {2, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(7);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    add.add_input(&lt_vec[4]);
    add.add_input(&lt_vec[5]);
    add.add_output(&lt_vec[6]);

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

TEST(pass_test, conv_bias_sum_fusion_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]);
    conv.add_output(&lt_vec[3]);
    add.add_input(&lt_vec[3]);
    add.add_input(&lt_vec[4]);
    add.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_add);
}

TEST(pass_test, conv_bias_sum_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t add {2, Add, "add"};
    op_t relu {3, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(8);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    add.add_input(&lt_vec[4]);
    add.add_input(&lt_vec[5]);
    add.add_output(&lt_vec[6]);
    relu.add_input(&lt_vec[6]);
    relu.add_output(&lt_vec[7]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_add_relu);
}

TEST(pass_test, conv_bias_sum_elu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t add {2, Add, "add"};
    op_t elu {3, Elu, "elu"};
    elu.set_attr("alpha", 0.1f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(8);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    add.add_input(&lt_vec[4]);
    add.add_input(&lt_vec[5]);
    add.add_output(&lt_vec[6]);
    elu.add_input(&lt_vec[6]);
    elu.add_output(&lt_vec[7]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&elu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_elu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_add_elu);
}

TEST(pass_test, conv_bias_sum_relu6_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t add {2, Add, "add"};
    op_t hardtanh {3, HardTanh, "hardtanh"};
    hardtanh.set_attr("min", 0.f);
    hardtanh.set_attr("max", 6.f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(8);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    add.add_input(&lt_vec[4]);
    add.add_input(&lt_vec[5]);
    add.add_output(&lt_vec[6]);
    hardtanh.add_input(&lt_vec[6]);
    hardtanh.add_output(&lt_vec[7]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&hardtanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_relu6_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_add_relu6);
}

TEST(pass_test, bn_relu_fusion) {
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

TEST(pass_test, conv_sum_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};
    op_t relu {2, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    add.add_input(&lt_vec[2]);
    add.add_input(&lt_vec[3]);
    add.add_output(&lt_vec[4]);
    relu.add_input(&lt_vec[4]);
    relu.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_sum_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_add_relu);
}

TEST(pass_test, conv_sum_elu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};
    op_t elu {2, Elu, "elu"};
    elu.set_attr("alpha", 0.2f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    add.add_input(&lt_vec[2]);
    add.add_input(&lt_vec[3]);
    add.add_output(&lt_vec[4]);
    elu.add_input(&lt_vec[4]);
    elu.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&elu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_sum_elu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_add_elu);
}

TEST(pass_test, conv_sum_relu6_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};
    op_t relu6 {2, HardTanh, "relu6"};
    relu6.set_attr("min", 0.f);
    relu6.set_attr("max", 6.f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    add.add_input(&lt_vec[2]);
    add.add_input(&lt_vec[3]);
    add.add_output(&lt_vec[4]);
    relu6.add_input(&lt_vec[4]);
    relu6.add_output(&lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu6), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_sum_relu6_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_add_relu6);
}

TEST(pass_test, conv_bias_sum_sum) {
    /*  conv
          |
        bias   conv
          |      |
         add   bias
           \   /
            add
    */
    graph_t agraph;
    op_t conv1 {0, Convolution, "conv"};
    set_conv_common_attr(conv1);
    op_t bias1 {1, BiasAdd, "bias"};
    op_t add1 {2, Add, "add"};
    op_t conv2 {3, Convolution, "conv"};
    set_conv_common_attr(conv2);
    op_t bias2 {4, BiasAdd, "bias"};
    op_t add2 {5, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(13);
    conv1.add_input(&lt_vec[0]);
    conv1.add_input(&lt_vec[1]);
    conv1.add_output(&lt_vec[2]);
    bias1.add_input(&lt_vec[2]);
    bias1.add_input(&lt_vec[3]);
    bias1.add_output(&lt_vec[4]);
    add1.add_input(&lt_vec[4]);
    add1.add_input(&lt_vec[5]);
    add1.add_output(&lt_vec[6]);
    conv2.add_input(&lt_vec[7]);
    conv2.add_input(&lt_vec[8]);
    conv2.add_output(&lt_vec[9]);
    bias2.add_input(&lt_vec[9]);
    bias2.add_input(&lt_vec[10]);
    bias2.add_output(&lt_vec[11]);
    add2.add_input(&lt_vec[6]);
    add2.add_input(&lt_vec[11]);
    add2.add_output(&lt_vec[12]);

    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&bias1), status::success);
    ASSERT_EQ(agraph.add_op(&add1), status::success);
    ASSERT_EQ(agraph.add_op(&conv2), status::success);
    ASSERT_EQ(agraph.add_op(&bias2), status::success);
    ASSERT_EQ(agraph.add_op(&add2), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 6);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 2);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_add);
    ASSERT_EQ(agraph.get_nodes()[1]->get_op_kind(), conv_bias_add);
}

TEST(pass_test, conv_bn_sum_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t add {2, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(10);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bn.add_input(&lt_vec[2]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_output(&lt_vec[7]);
    add.add_input(&lt_vec[7]);
    add.add_input(&lt_vec[8]);
    add.add_output(&lt_vec[9]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bn_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bn_add);
}

TEST(pass_test, conv_bn_sum_fusion_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t add {2, Add, "add"};
    op_t relu {3, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(11);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bn.add_input(&lt_vec[2]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_output(&lt_vec[7]);
    relu.add_input(&lt_vec[8]);
    relu.add_output(&lt_vec[9]);
    add.add_input(&lt_vec[7]);
    add.add_input(&lt_vec[9]);
    add.add_output(&lt_vec[10]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bn_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 2);
    ASSERT_EQ(agraph.get_nodes()[1]->get_op_kind(), conv_bn_add);
}

TEST(pass_test, conv_bn_sum_fusion_fail) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t add {2, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(11);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_input(&lt_vec[7]);
    bn.add_output(&lt_vec[8]);
    add.add_input(&lt_vec[8]);
    add.add_input(&lt_vec[9]);
    add.add_output(&lt_vec[10]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bn_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 3);
}

TEST(pass_test, conv_bias_bn_sum_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t add {2, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(11);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_input(&lt_vec[7]);
    bn.add_output(&lt_vec[8]);
    add.add_input(&lt_vec[8]);
    add.add_input(&lt_vec[9]);
    add.add_output(&lt_vec[10]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_bn_add);
}

TEST(pass_test, conv_bn_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t relu {2, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(9);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bn.add_input(&lt_vec[2]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_output(&lt_vec[7]);
    relu.add_input(&lt_vec[7]);
    relu.add_output(&lt_vec[8]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bn_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bn_relu);
}

TEST(pass_test, conv_bias_bn_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t bn {2, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t relu {3, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(11);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bias.add_input(&lt_vec[2]);
    bias.add_input(&lt_vec[3]);
    bias.add_output(&lt_vec[4]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_input(&lt_vec[7]);
    bn.add_input(&lt_vec[8]);
    bn.add_output(&lt_vec[9]);
    relu.add_input(&lt_vec[9]);
    relu.add_output(&lt_vec[10]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_bn_relu);
}

TEST(pass_test, conv_bias_bn_relu_fusion_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t relu {2, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(10);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_input(&lt_vec[7]);
    bn.add_output(&lt_vec[8]);
    relu.add_input(&lt_vec[8]);
    relu.add_output(&lt_vec[9]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_bn_relu);
}

TEST(pass_test, conv_bn_sum_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t add {2, Add, "add"};
    op_t relu {3, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(11);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    bn.add_input(&lt_vec[2]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_output(&lt_vec[7]);
    add.add_input(&lt_vec[7]);
    add.add_input(&lt_vec[8]);
    add.add_output(&lt_vec[9]);
    relu.add_input(&lt_vec[9]);
    relu.add_output(&lt_vec[10]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bn_sum_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bn_add_relu);
}

TEST(pass_test, conv_bias_bn_sum_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t add {2, Add, "add"};
    op_t relu {3, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(12);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]); // conv with bias
    conv.add_output(&lt_vec[3]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_input(&lt_vec[7]);
    bn.add_output(&lt_vec[8]);
    add.add_input(&lt_vec[8]);
    add.add_input(&lt_vec[9]);
    add.add_output(&lt_vec[10]);
    relu.add_input(&lt_vec[10]);
    relu.add_output(&lt_vec[11]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_sum_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    ASSERT_EQ(agraph.get_nodes()[0]->get_op_kind(), conv_bias_bn_add_relu);
}

TEST(pass_test, matmul_relu_fusion) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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

/*
TEST(pass_test, layernorm_fusion) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t divide {0, Divide, std::string("divide")};
    op_t erf {1, Erf, std::string("erf")};
    op_t add {2, Add, std::string("add")};
    op_t multiply_1 {3, Multiply, std::string("multiply")};
    op_t multiply_2 {4, Multiply, std::string("multiply")};

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

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
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    pass::pass_manager pm;
    std::vector<op_kind_t> single_node_set = {BatchNormInference, Add, ReLU,
            MatMul, AvgPool, MaxPool, AvgPoolBackprop,
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

TEST(pass_test, conv_single_node_replacement) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(3);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 1);

    pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    const node_ptr &replaced_node = agraph.get_nodes()[0];
    ASSERT_EQ(replaced_node->get_op_kind(), Convolution);
    ASSERT_EQ(replaced_node->has_attr("backend"), true);
    ASSERT_EQ(replaced_node->get_attr<std::string>("backend"), "dnnl");
}

TEST(pass_test, conv_single_node_replacement_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_input(&lt_vec[2]);
    conv.add_output(&lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 1);

    pass::pass_base_ptr apass = get_pass("conv_bias_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.num_nodes(), 1);
    const node_ptr &replaced_node = agraph.get_nodes()[0];
    ASSERT_EQ(replaced_node->get_op_kind(), conv_bias);
    ASSERT_EQ(replaced_node->has_attr("backend"), true);
    ASSERT_EQ(replaced_node->get_attr<std::string>("backend"), "dnnl");
}

TEST(pass_test, save_load_json) {
    graph_t agraph;
    op_t conv1 {0, Convolution, "conv"};
    set_conv_common_attr(conv1);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t relu {2, ReLU, "relu"};
    op_t conv2 {3, Convolution, "conv"};
    set_conv_common_attr(conv2);
    op_t add {4, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(13);
    conv1.add_input(&lt_vec[0]);
    conv1.add_input(&lt_vec[1]);
    conv1.add_output(&lt_vec[2]);
    bn.add_input(&lt_vec[2]);
    bn.add_input(&lt_vec[3]);
    bn.add_input(&lt_vec[4]);
    bn.add_input(&lt_vec[5]);
    bn.add_input(&lt_vec[6]);
    bn.add_output(&lt_vec[7]);
    relu.add_input(&lt_vec[7]);
    relu.add_output(&lt_vec[8]);
    conv2.add_input(&lt_vec[9]);
    conv2.add_input(&lt_vec[10]);
    conv2.add_output(&lt_vec[11]);
    add.add_input(&lt_vec[11]);
    add.add_input(&lt_vec[8]);
    add.add_output(&lt_vec[12]);

    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&conv2), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 5);

    pass::pass_manager pm;
    pm.print_passes("passes.json");
    pm.run_passes(agraph, "passes.json");
    ASSERT_EQ(agraph.num_nodes(), 2);
    ASSERT_EQ(agraph.get_num_partitions(), 2);
}

TEST(pass_test, two_conv_with_shared_weight) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t conv0 {0, Convolution, "conv0"};
    set_conv_common_attr(conv0);
    op_t relu0 {1, ReLU, std::string("relu0")};
    op_t conv1 {2, Convolution, std::string("conv1")};
    set_conv_common_attr(conv1);
    op_t relu1 {3, ReLU, std::string("relu1")};

    // create conv0 inputs tensor
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv0.add_input(&lt_vec[0]);
    conv0.add_input(&lt_vec[1]);
    conv0.add_output(&lt_vec[2]);
    relu0.add_input(&lt_vec[2]);
    relu0.add_output(&lt_vec[3]);

    conv1.add_input(&lt_vec[3]);
    conv1.add_input(&lt_vec[1]);
    conv1.add_output(&lt_vec[4]);
    relu1.add_input(&lt_vec[4]);
    relu1.add_output(&lt_vec[5]);

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
        ASSERT_EQ(node->get_op_kind(), conv_relu);
        ASSERT_EQ(node->num_inputs_tensor(), 2);
        ASSERT_EQ(node->num_outputs_tensor(), 1);
    }
}

TEST(pass_test, multi_values_between_two_nodes) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};

    // create lt
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);

    conv.add_input(&lt_vec[0]);
    conv.add_input(&lt_vec[1]);
    conv.add_output(&lt_vec[2]);
    add.add_input(&lt_vec[2]);
    add.add_input(&lt_vec[2]);
    add.add_output(&lt_vec[3]);

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
