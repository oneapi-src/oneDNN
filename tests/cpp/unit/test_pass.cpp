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

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "utils/pm/pass_base.hpp"
#include "utils/pm/pass_manager.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"

#include "backend/fake/fake_backend.hpp"
#include "backend/fake/fake_partition_impl.hpp"

#include "utils.hpp"

using namespace dnnl::graph::impl;
using namespace dnnl::graph::impl::op_kind;
using namespace dnnl::graph::tests::unit::utils;

using op_ptr = std::shared_ptr<dnnl::graph::impl::op_t>;

namespace {
dnnl::graph::impl::pass::pass_base_ptr get_pass(const std::string &pass_name) {
    auto &backend_ptr
            = dnnl::graph::impl::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::graph::impl::pass::pass_manager(
            backend_ptr.get_pass_registry());
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

const impl::op_t *get_fused_op(
        const std::shared_ptr<dnnl::graph::impl::partition_impl_t> &part) {
    return dynamic_cast<
            const dnnl::graph::impl::dnnl_impl::dnnl_partition_impl_t *>(
            part.get())
            ->get_fused_op()
            .get();
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    //bn.internal_connect_input(0, conv, 0);
    bn.add_input(lt_vec[2]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_output(lt_vec[7]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);

    agraph.build_graph();

    ASSERT_EQ(agraph.num_ops(), 2);
    ASSERT_EQ(agraph.get_ops()[0]->get_kind(), Convolution);
    ASSERT_EQ(agraph.get_ops()[0]->num_inputs(), 2);
    ASSERT_EQ(agraph.get_ops()[0]->num_outputs(), 1);
    ASSERT_EQ(agraph.get_ops()[1]->get_kind(), BatchNormInference);
    ASSERT_EQ(agraph.get_ops()[1]->num_inputs(), 5);
    ASSERT_EQ(agraph.get_ops()[1]->num_outputs(), 1);

    pass::pass_base_ptr apass = get_pass("conv_bn_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(), conv_bn);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
}

TEST(pass_test, conv_bn_fusion_fail) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(9);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); //conv with bias
    conv.add_output(lt_vec[3]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_input(lt_vec[7]);
    bn.add_output(lt_vec[8]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bn_fusion");
    apass->run(agraph);
    // conv with bias cannot be fused via conv_bn_fusion pass,
    // so num partitions is zero
    ASSERT_EQ(agraph.get_num_partitions(), 0);
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bn.add_input(lt_vec[2]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_output(lt_vec[7]);
    relu.add_input(lt_vec[2]);
    relu.add_output(lt_vec[8]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bn_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);
}

TEST(pass_test, conv_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {1, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    relu.add_input(lt_vec[2]);
    relu.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(), conv_relu);
}

TEST(pass_test, conv_relu_fusion_fail) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {1, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    relu.add_input(lt_vec[3]);
    relu.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);
}

TEST(pass_test, conv_relu_fusion_fail_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu1 {1, ReLU, "relu"};
    op_t relu2 {2, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    relu1.add_input(lt_vec[3]);
    relu1.add_output(lt_vec[4]);
    relu2.add_input(lt_vec[3]);
    relu2.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu1), status::success);
    ASSERT_EQ(agraph.add_op(&relu2), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);
}

TEST(pass_test, conv_bias_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
}

TEST(pass_test, conv_bias_fusion_fail) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    bias.add_input(lt_vec[3]);
    bias.add_input(lt_vec[4]);
    bias.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_fusion");
    apass->run(agraph);
    // conv with bias cannot be fused via conv_bias_fusion pass,
    // so only one partition
    ASSERT_EQ(agraph.get_num_partitions(), 1);
}

TEST(pass_test, conv_sum_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    add.add_input(lt_vec[2]);
    add.add_input(lt_vec[3]);
    add.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
}

TEST(pass_test, conv_sum_fusion_fail) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    add.add_input(lt_vec[3]);
    add.add_input(lt_vec[4]);
    add.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);
}

TEST(pass_test, conv_bias_bn_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t bn {2, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(10);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_input(lt_vec[7]);
    bn.add_input(lt_vec[8]);
    bn.add_output(lt_vec[9]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), conv_bias_bn);
}

TEST(pass_test, conv_bias_bn_fusion_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(9);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_input(lt_vec[7]);
    bn.add_output(lt_vec[8]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), conv_bias_bn);
}

TEST(pass_test, conv_bias_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t relu {2, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    relu.add_input(lt_vec[4]);
    relu.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_relu);
}

TEST(pass_test, conv_bias_relu_fusion_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {1, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]);
    conv.add_output(lt_vec[3]);
    relu.add_input(lt_vec[3]);
    relu.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_relu);
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    hardtanh.add_input(lt_vec[4]);
    hardtanh.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&hardtanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_relu6_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_relu6);
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    hardtanh.add_input(lt_vec[4]);
    hardtanh.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&hardtanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_relu6_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);
}

TEST(pass_test, conv_bias_elu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t elu {1, Elu, "elu"};
    elu.set_attr("alpha", 0.1f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    elu.add_input(lt_vec[3]);
    elu.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&elu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_elu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_elu);
}

TEST(pass_test, conv_bias_sigmoid_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t sigmoid {1, Sigmoid, "sigmoid"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    sigmoid.add_input(lt_vec[3]);
    sigmoid.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_sigmoid_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_sigmoid);
}

TEST(pass_test, conv_bias_swish_fusion) {
    // swish: f(x) = x * sigmoid(x)
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t sigmoid {1, Sigmoid, "sigmoid"};
    op_t multiply {2, Multiply, "multiply"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    sigmoid.add_input(lt_vec[3]);
    sigmoid.add_output(lt_vec[4]);
    multiply.add_input(lt_vec[4]);
    multiply.add_input(lt_vec[3]);
    multiply.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_swish_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_swish);
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    hardtanh.add_input(lt_vec[4]);
    hardtanh.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&hardtanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_hardtanh_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_hardtanh);
}

TEST(pass_test, conv_bias_square_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t square {2, Square, "square"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    square.add_input(lt_vec[4]);
    square.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&square), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_square_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_square);
}

TEST(pass_test, conv_bias_tanh_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t tanh {2, Tanh, "tanh"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    tanh.add_input(lt_vec[4]);
    tanh.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&tanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_tanh_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_tanh);
}

TEST(pass_test, conv_bias_abs_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t abs {2, Abs, "abs"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    abs.add_input(lt_vec[4]);
    abs.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&abs), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_abs_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_abs);
}

TEST(pass_test, conv_bias_sqrt_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t sqrt {2, Sqrt, "sqrt"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    sqrt.add_input(lt_vec[4]);
    sqrt.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&sqrt), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_sqrt_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_sqrt);
}

TEST(pass_test, conv_bias_sum_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t add {2, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(7);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    add.add_input(lt_vec[4]);
    add.add_input(lt_vec[5]);
    add.add_output(lt_vec[6]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_add);
}

TEST(pass_test, conv_bias_sum_fusion_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]);
    conv.add_output(lt_vec[3]);
    add.add_input(lt_vec[3]);
    add.add_input(lt_vec[4]);
    add.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_add);
}

TEST(pass_test, conv_bias_sum_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t add {2, Add, "add"};
    op_t relu {3, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(8);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    add.add_input(lt_vec[4]);
    add.add_input(lt_vec[5]);
    add.add_output(lt_vec[6]);
    relu.add_input(lt_vec[6]);
    relu.add_output(lt_vec[7]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_add_relu);
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    add.add_input(lt_vec[4]);
    add.add_input(lt_vec[5]);
    add.add_output(lt_vec[6]);
    elu.add_input(lt_vec[6]);
    elu.add_output(lt_vec[7]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&elu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_elu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_add_elu);
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    add.add_input(lt_vec[4]);
    add.add_input(lt_vec[5]);
    add.add_output(lt_vec[6]);
    hardtanh.add_input(lt_vec[6]);
    hardtanh.add_output(lt_vec[7]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&hardtanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_relu6_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_add_relu6);
}

TEST(pass_test, binary_sum_fusion_no_broadcast) {
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, op_kind_t>> opkind_pair {
            {Multiply, multiply_add}, {Maximum, maximum_add},
            {Minimum, minimum_add}};

    for (auto &p : opkind_pair) {
        graph_t agraph;
        auto binary_kind = p.first;
        op_t binary {0, binary_kind, "binary"};
        op_t add {1, Add, "add"};

        std::vector<logical_tensor_t> lt_vec;
        lt_vec.reserve(5);
        for (size_t i = 0; i < 5; i++)
            lt_vec.emplace_back(
                    logical_tensor_init(i, {2, 3, 4, 5}, data_type::f32));

        binary.add_input(lt_vec[0]);
        binary.add_input(lt_vec[1]);
        binary.add_output(lt_vec[2]);
        add.add_input(lt_vec[2]);
        add.add_input(lt_vec[3]);
        add.add_output(lt_vec[4]);

        ASSERT_EQ(agraph.add_op(&binary), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        agraph.build_graph();

        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1);

        auto fused_op = get_fused_op(agraph.get_partitions()[0]);
        ASSERT_EQ(fused_op->get_kind(), p.second);
    }
}

TEST(pass_test, binary_sum_fusion_supported_broadcast) {
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, op_kind_t>> opkind_pair {
            {Multiply, multiply_add}, {Maximum, maximum_add},
            {Minimum, minimum_add}};

    for (auto &p : opkind_pair) {
        graph_t agraph;
        auto binary_kind = p.first;
        op_t binary {0, binary_kind, "binary"};
        op_t add {1, Add, "add"};

        std::vector<logical_tensor_t> lt_vec;
        lt_vec.reserve(5);
        for (size_t i = 0; i < 5; i++)
            lt_vec.emplace_back(
                    logical_tensor_init(i, {2, 3, 4, 5}, data_type::f32));

        // set add's src1 shape to be {1,1,4,5}
        lt_vec[3].dims[0] = 1;
        lt_vec[3].dims[1] = 1;

        binary.add_input(lt_vec[0]);
        binary.add_input(lt_vec[1]);
        binary.add_output(lt_vec[2]);
        add.add_input(lt_vec[2]);
        add.add_input(lt_vec[3]);
        add.add_output(lt_vec[4]);

        ASSERT_EQ(agraph.add_op(&binary), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        agraph.build_graph();

        pm.run_passes(agraph, "no_config");

        // should not be fused
        ASSERT_EQ(agraph.get_num_partitions(), 1);
    }
}

TEST(pass_test, binary_sum_fusion_unsupported_broadcast) {
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, op_kind_t>> opkind_pair {
            {Multiply, multiply_add}, {Maximum, maximum_add},
            {Minimum, minimum_add}};

    for (auto &p : opkind_pair) {
        graph_t agraph;
        auto binary_kind = p.first;
        op_t binary {0, binary_kind, "binary"};
        op_t add {1, Add, "add"};

        std::vector<logical_tensor_t> lt_vec;
        lt_vec.reserve(5);
        for (size_t i = 0; i < 5; i++)
            lt_vec.emplace_back(
                    logical_tensor_init(i, {1, 1, 1, 1}, data_type::f32));

        lt_vec[1].dims[2] = 28;
        lt_vec[1].dims[3] = 28;

        lt_vec[2].dims[2] = 28;
        lt_vec[2].dims[3] = 28;

        lt_vec[3].dims[1] = 32;
        lt_vec[3].dims[2] = 28;
        lt_vec[3].dims[2] = 28;

        lt_vec[4].dims[1] = 32;
        lt_vec[4].dims[2] = 28;
        lt_vec[4].dims[3] = 28;

        binary.add_input(lt_vec[0]);
        binary.add_input(lt_vec[1]);
        binary.add_output(lt_vec[2]);
        add.add_input(lt_vec[2]);
        add.add_input(lt_vec[3]);
        add.add_output(lt_vec[4]);

        ASSERT_EQ(agraph.add_op(&binary), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        agraph.build_graph();

        pm.run_passes(agraph, "no_config");

        // should not be fused
        ASSERT_EQ(agraph.get_num_partitions(), 2);
    }
}

TEST(pass_test, binary_sum_fusion_unknown_shape) {
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, op_kind_t>> opkind_pair {
            {Multiply, multiply_add}, {Maximum, maximum_add},
            {Minimum, minimum_add}};

    for (auto &p : opkind_pair) {
        graph_t agraph;
        auto binary_kind = p.first;
        op_t binary {0, binary_kind, "binary"};
        op_t add {1, Add, "add"};

        std::vector<logical_tensor_t> lt_vec;
        lt_vec.reserve(5);
        // valid ndims, invalid shape
        for (size_t i = 0; i < 5; i++)
            lt_vec.emplace_back(
                    logical_tensor_init(i, {-1, -1, -1, -1}, data_type::f32));

        binary.add_input(lt_vec[0]);
        binary.add_input(lt_vec[1]);
        binary.add_output(lt_vec[2]);
        add.add_input(lt_vec[2]);
        add.add_input(lt_vec[3]);
        add.add_output(lt_vec[4]);

        ASSERT_EQ(agraph.add_op(&binary), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        agraph.build_graph();

        pm.run_passes(agraph, "no_config");

        // should not be fused
        ASSERT_EQ(agraph.get_num_partitions(), 2);
    }
}

TEST(pass_test, binary_eltwise_fusion) {
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    std::vector<std::pair<std::pair<op_kind_t, op_kind_t>, op_kind_t>>
            opkind_pair {{{Add, Sigmoid}, add_sigmoid}, {{Add, ReLU}, add_relu},
                    {{Multiply, Sigmoid}, multiply_sigmoid},
                    {{Multiply, ReLU}, multiply_relu},
                    {{Maximum, Sigmoid}, maximum_sigmoid},
                    {{Maximum, ReLU}, maximum_relu},
                    {{Minimum, Sigmoid}, minimum_sigmoid},
                    {{Minimum, ReLU}, minimum_relu}};

    for (auto &p : opkind_pair) {
        graph_t agraph;
        auto binary_kind = p.first.first;
        auto eltwise_kind = p.first.second;

        op_t *op1 = agraph.create_op(binary_kind);
        op_t *op2 = agraph.create_op(eltwise_kind);
        op2->fill_and_connect_input(0, *op1, 0);

        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1);

        auto fused_op = get_fused_op(agraph.get_partitions()[0]);
        ASSERT_EQ(fused_op->get_kind(), p.second);
    }
}

TEST(pass_test, bn_relu_fusion) {
    graph_t agraph;

    op_t *op1 = agraph.create_op(BatchNormInference);
    op_t *op2 = agraph.create_op(ReLU);
    op2->fill_and_connect_input(0, *op1, 0);

    pass::pass_base_ptr apass = get_pass("bn_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), bn_relu);
}

TEST(pass_test, bn_bwd_relu_bwd_fusion) {
    graph_t agraph;

    op_t *op1 = agraph.create_op(ReLUBackprop);
    op_t *op2 = agraph.create_op(BatchNormTrainingBackprop);
    op2->fill_and_connect_input(0, *op1, 0);

    pass::pass_base_ptr apass = get_pass("bn_bwd_relu_bwd_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), bn_bwd_relu_bwd);
}

TEST(pass_test, conv_sum_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};
    op_t relu {2, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    add.add_input(lt_vec[2]);
    add.add_input(lt_vec[3]);
    add.add_output(lt_vec[4]);
    relu.add_input(lt_vec[4]);
    relu.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_sum_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_add_relu);
}

TEST(pass_test, conv_sum_elu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};
    op_t elu {2, Elu, "elu"};
    elu.set_attr("alpha", 0.2f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    add.add_input(lt_vec[2]);
    add.add_input(lt_vec[3]);
    add.add_output(lt_vec[4]);
    elu.add_input(lt_vec[4]);
    elu.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&elu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_sum_elu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), conv_add_elu);
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    add.add_input(lt_vec[2]);
    add.add_input(lt_vec[3]);
    add.add_output(lt_vec[4]);
    relu6.add_input(lt_vec[4]);
    relu6.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu6), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_sum_relu6_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_add_relu6);
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
    conv1.add_input(lt_vec[0]);
    conv1.add_input(lt_vec[1]);
    conv1.add_output(lt_vec[2]);
    bias1.add_input(lt_vec[2]);
    bias1.add_input(lt_vec[3]);
    bias1.add_output(lt_vec[4]);
    add1.add_input(lt_vec[4]);
    add1.add_input(lt_vec[5]);
    add1.add_output(lt_vec[6]);
    conv2.add_input(lt_vec[7]);
    conv2.add_input(lt_vec[8]);
    conv2.add_output(lt_vec[9]);
    bias2.add_input(lt_vec[9]);
    bias2.add_input(lt_vec[10]);
    bias2.add_output(lt_vec[11]);
    add2.add_input(lt_vec[6]);
    add2.add_input(lt_vec[11]);
    add2.add_output(lt_vec[12]);

    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&bias1), status::success);
    ASSERT_EQ(agraph.add_op(&add1), status::success);
    ASSERT_EQ(agraph.add_op(&conv2), status::success);
    ASSERT_EQ(agraph.add_op(&bias2), status::success);
    ASSERT_EQ(agraph.add_op(&add2), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 6);

    pass::pass_base_ptr apass = get_pass("conv_bias_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 2);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_add);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[1])->get_kind(),
            conv_bias_add);
}

TEST(pass_test, conv_bn_sum_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t add {2, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(10);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bn.add_input(lt_vec[2]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_output(lt_vec[7]);
    add.add_input(lt_vec[7]);
    add.add_input(lt_vec[8]);
    add.add_output(lt_vec[9]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bn_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), conv_bn_add);
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bn.add_input(lt_vec[2]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_output(lt_vec[7]);
    relu.add_input(lt_vec[8]);
    relu.add_output(lt_vec[9]);
    add.add_input(lt_vec[7]);
    add.add_input(lt_vec[9]);
    add.add_output(lt_vec[10]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bn_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), conv_bn_add);
}

TEST(pass_test, conv_bn_sum_fusion_fail) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t add {2, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(11);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_input(lt_vec[7]);
    bn.add_output(lt_vec[8]);
    add.add_input(lt_vec[8]);
    add.add_input(lt_vec[9]);
    add.add_output(lt_vec[10]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bn_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);
}

TEST(pass_test, conv_bias_bn_sum_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t add {2, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(11);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_input(lt_vec[7]);
    bn.add_output(lt_vec[8]);
    add.add_input(lt_vec[8]);
    add.add_input(lt_vec[9]);
    add.add_output(lt_vec[10]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_bn_add);
}

TEST(pass_test, conv_bn_relu_fusion) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t relu {2, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(9);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bn.add_input(lt_vec[2]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_output(lt_vec[7]);
    relu.add_input(lt_vec[7]);
    relu.add_output(lt_vec[8]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bn_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), conv_bn_relu);
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_input(lt_vec[7]);
    bn.add_input(lt_vec[8]);
    bn.add_output(lt_vec[9]);
    relu.add_input(lt_vec[9]);
    relu.add_output(lt_vec[10]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_bn_relu);
}

TEST(pass_test, conv_bias_bn_relu_fusion_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t relu {2, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(10);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_input(lt_vec[7]);
    bn.add_output(lt_vec[8]);
    relu.add_input(lt_vec[8]);
    relu.add_output(lt_vec[9]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_bn_relu);
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bn.add_input(lt_vec[2]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_output(lt_vec[7]);
    add.add_input(lt_vec[7]);
    add.add_input(lt_vec[8]);
    add.add_output(lt_vec[9]);
    relu.add_input(lt_vec[9]);
    relu.add_output(lt_vec[10]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bn_sum_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bn_add_relu);
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
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_input(lt_vec[7]);
    bn.add_output(lt_vec[8]);
    add.add_input(lt_vec[8]);
    add.add_input(lt_vec[9]);
    add.add_output(lt_vec[10]);
    relu.add_input(lt_vec[10]);
    relu.add_output(lt_vec[11]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);

    pass::pass_base_ptr apass = get_pass("conv_bias_bn_sum_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            conv_bias_bn_add_relu);
}

TEST(pass_test, matmul_relu_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t relu {1, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    relu.add_input(lt_vec[2]);
    relu.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_relu);
}

TEST(pass_test, matmul_relu_fail_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t relu {1, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_input(lt_vec[2]);
    matmul.add_output(lt_vec[3]);
    relu.add_input(lt_vec[3]);
    relu.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("matmul_bias_relu_fusion");
    apass2->run(agraph);
    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_bias_relu);
}

TEST(pass_test, relu_matmul) {
    graph_t agraph;
    op_t relu {0, ReLU, "relu"};
    op_t matmul {1, MatMul, "matmul"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    relu.add_input(lt_vec[0]);
    relu.add_output(lt_vec[1]);
    matmul.add_input(lt_vec[1]);
    matmul.add_input(lt_vec[2]);
    matmul.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 0);
}

TEST(pass_test, matmul_elu_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t elu {1, Elu, "elu"};
    elu.set_attr("alpha", 0.1f);
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    elu.add_input(lt_vec[2]);
    elu.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&elu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_elu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_elu);
}

TEST(pass_test, matmul_sigmoid_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t sigmoid {1, Sigmoid, "sigmoid"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    sigmoid.add_input(lt_vec[2]);
    sigmoid.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_sigmoid_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_sigmoid);
}

TEST(pass_test, matmul_hardtanh_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t hardtanh {1, HardTanh, "hardtanh"};
    hardtanh.set_attr("min", -1.f);
    hardtanh.set_attr("max", 1.f);
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    hardtanh.add_input(lt_vec[2]);
    hardtanh.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&hardtanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_hardtanh_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_hardtanh);
}

TEST(pass_test, matmul_gelu_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t gelu {1, GELU, "gelu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    gelu.add_input(lt_vec[2]);
    gelu.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&gelu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_gelu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_gelu);
}

TEST(pass_test, matmul_sum_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t wildcard {1, Wildcard, "wildcard"};
    op_t add {2, Add, "add"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    wildcard.add_output(lt_vec[3]);
    add.add_input(lt_vec[2]);
    add.add_input(lt_vec[3]);
    add.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("matmul_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_add);
}

TEST(pass_test, matmul_sum_fusion_opposite_order) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t wildcard {1, Wildcard, "wildcard"};
    op_t add {2, Add, "add"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    wildcard.add_output(lt_vec[3]);
    add.add_input(lt_vec[3]);
    add.add_input(lt_vec[2]);
    add.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("matmul_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_add);
}

TEST(pass_test, matmul_sum_gelu_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t wildcard {1, Wildcard, "wildcard"};
    op_t add {2, Add, "add"};
    op_t gelu {3, GELU, "add"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    wildcard.add_output(lt_vec[3]);
    add.add_input(lt_vec[2]);
    add.add_input(lt_vec[3]);
    add.add_output(lt_vec[4]);
    gelu.add_input(lt_vec[4]);
    gelu.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&gelu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);

    pass::pass_base_ptr apass = get_pass("matmul_sum_gelu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_add_gelu);
}

TEST(pass_test, matmul_sum_relu_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t wildcard {1, Wildcard, "wildcard"};
    op_t add {2, Add, "add"};
    op_t relu {3, ReLU, "add"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    wildcard.add_output(lt_vec[3]);
    add.add_input(lt_vec[2]);
    add.add_input(lt_vec[3]);
    add.add_output(lt_vec[4]);
    relu.add_input(lt_vec[4]);
    relu.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);

    pass::pass_base_ptr apass = get_pass("matmul_sum_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_add_relu);
}

TEST(pass_test, conv_bwd_f_biasadd_bwd_fusion) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    graph_t agraph;
    op_t *op1 = agraph.create_op(ConvolutionBackpropFilters);
    op_t *op2 = agraph.create_op(BiasAddBackprop);
    op2->fill_and_connect_input(0, *op1, 0);

    pass::pass_base_ptr apass = get_pass("conv_bwd_f_biasadd_bwd_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), conv_bwd_f_biasadd_bwd);
}

TEST(pass_test, matmul_bias_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t bias {1, BiasAdd, "bias"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_bias_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_bias);
}

TEST(pass_test, matmul_bias_fusion_case2) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_input(lt_vec[2]);
    matmul.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 1);

    pass::pass_base_ptr apass = get_pass("matmul_bias_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_bias);
}

TEST(pass_test, matmul_bias_sigmoid_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t sigmoid {1, Sigmoid, "sigmoid"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_input(lt_vec[2]);
    matmul.add_output(lt_vec[3]);
    sigmoid.add_input(lt_vec[3]);
    sigmoid.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_sigmoid_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("matmul_bias_sigmoid_fusion");
    apass2->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_bias_sigmoid);
}

TEST(pass_test, matmul_bias_elu_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t bias {1, BiasAdd, "bias"};
    op_t elu {2, Elu, "elu"};
    elu.set_attr("alpha", 0.1f);
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    elu.add_input(lt_vec[4]);
    elu.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&elu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("matmul_bias_elu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_bias_elu);
}

TEST(pass_test, matmul_bias_relu_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t bias {1, BiasAdd, "bias"};
    op_t relu {2, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    relu.add_input(lt_vec[4]);
    relu.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("matmul_bias_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_bias_relu);
}

TEST(pass_test, matmul_bias_hardtanh_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t hardtanh {1, HardTanh, "hardtanh"};
    hardtanh.set_attr("min", 0.1f);
    hardtanh.set_attr("max", 0.2f);
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_input(lt_vec[2]);
    matmul.add_output(lt_vec[3]);
    hardtanh.add_input(lt_vec[3]);
    hardtanh.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&hardtanh), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_hardtanh_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("matmul_bias_hardtanh_fusion");
    apass2->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_bias_hardtanh);
}

TEST(pass_test, matmul_bias_sum_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t relu {1, ReLU, "relu"};
    op_t add {2, Add, "add"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(7);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_input(lt_vec[2]);
    matmul.add_output(lt_vec[3]);
    relu.add_input(lt_vec[4]);
    relu.add_output(lt_vec[5]);
    add.add_input(lt_vec[5]);
    add.add_input(lt_vec[3]);
    add.add_output(lt_vec[6]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    pass::pass_base_ptr apass = get_pass("matmul_bias_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_bias_add);
}

TEST(pass_test, matmul_bias_sum_relu_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t wildcard {1, Wildcard, "wildcard"};
    op_t add {2, Add, "add"};
    op_t relu {3, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(8);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_input(lt_vec[2]);
    matmul.add_output(lt_vec[3]);
    wildcard.add_input(lt_vec[4]);
    wildcard.add_output(lt_vec[5]);
    add.add_input(lt_vec[5]);
    add.add_input(lt_vec[3]);
    add.add_output(lt_vec[6]);
    relu.add_input(lt_vec[6]);
    relu.add_output(lt_vec[7]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);

    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);

    pass::pass_base_ptr apass = get_pass("matmul_bias_sum_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_bias_add_relu);
}

TEST(pass_test, matmul_bias_swish_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t bias {1, BiasAdd, "bias"};
    op_t sigmoid {2, Sigmoid, "sigmoid"};
    op_t mul {3, Multiply, "mul"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(7);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    sigmoid.add_input(lt_vec[4]);
    sigmoid.add_output(lt_vec[5]);
    mul.add_input(lt_vec[5]);
    mul.add_input(lt_vec[4]);
    mul.add_output(lt_vec[6]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&mul), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);

    pass::pass_base_ptr apass = get_pass("matmul_bias_swish_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_bias_swish);
}

TEST(pass_test, matmul_bias_relu6_fusion) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t relu6 {1, HardTanh, "hardtanh"};
    relu6.set_attr("min", 0.f);
    relu6.set_attr("max", 6.f);
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_input(lt_vec[2]);
    matmul.add_output(lt_vec[3]);
    relu6.add_input(lt_vec[3]);
    relu6.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu6), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_bias_relu6_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), matmul_bias_relu6);
}

TEST(pass_test, matmul_bias_relu6_fusion_fail) {
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t relu6 {1, HardTanh, "hardtanh"};
    relu6.set_attr("min", 0.2f);
    relu6.set_attr("max", 6.f);
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_input(lt_vec[2]);
    matmul.add_output(lt_vec[3]);
    relu6.add_input(lt_vec[3]);
    relu6.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu6), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("matmul_bias_relu6_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 0);
}
/*
TEST(pass_test, layernorm_fusion) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    graph_t agraph;
    op_t *op1 = agraph.create_op(Reshape);
    op_t *op2 = agraph.create_op(BatchNormForwardTraining);
    op_t *op3 = agraph.create_op(Reshape);
    op_t *op4 = agraph.create_op(Multiply);
    op_t *op5 = agraph.create_op(Wildcard);
    op_t *op6 = agraph.create_op(Add);
    op2->fill_and_connect_input(0, op1, 0);
    op3->fill_and_connect_input(0, op2, 0);
    op4->fill_and_connect_input(0, op3, 0);
    op6->fill_and_connect_input(0, op4, 0);
    op6->fill_and_connect_input(1, op5, 0);

    pass::pass_base_ptr apass = get_pass("layernorm_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 2);

    auto fop1 = agraph.get_ops()[0];
    auto fop2 = agraph.get_ops()[1];
    if (fop1->get_kind() != LayerNorm) {
        ASSERT_EQ(fop2->get_kind(), LayerNorm);
    } else {
        ASSERT_EQ(fop1->get_kind(), LayerNorm);
    }
}*/

TEST(pass_test, gelu_erf_based_fusion) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    graph_t agraph;
    op_t *op1 = agraph.create_op(Wildcard);
    op_t *op2 = agraph.create_op(Divide);
    op_t *op3 = agraph.create_op(Erf);
    op_t *op4 = agraph.create_op(Wildcard);
    op_t *op5 = agraph.create_op(Add);
    op_t *op6 = agraph.create_op(Wildcard);
    op_t *op7 = agraph.create_op(Multiply);
    op_t *op8 = agraph.create_op(Wildcard);
    op_t *op9 = agraph.create_op(Multiply);
    op2->fill_and_connect_input(0, *op1, 0);
    op3->fill_and_connect_input(0, *op2, 0);
    op5->fill_and_connect_input(0, *op3, 0);
    op5->fill_and_connect_input(1, *op4, 0);
    op7->fill_and_connect_input(0, *op5, 0);
    op7->fill_and_connect_input(1, *op6, 0);
    op9->fill_and_connect_input(0, *op7, 0);
    op9->fill_and_connect_input(1, *op8, 0);

    pass::pass_base_ptr apass = get_pass("gelu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);
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

    divide.add_input(divide_in_a_tensor);
    divide.add_input(divide_in_b_tensor);
    divide.add_output(divide_dst);
    erf.add_input(divide_dst);
    erf.add_output(erf_dst);
    add.add_input(erf_dst);
    add.add_input(add_in_tensor);
    add.add_output(add_dst);
    multiply_1.add_input(add_dst);
    multiply_1.add_input(multiply_1_in_tensor);
    multiply_1.add_output(multiply_1_dst);
    multiply_2.add_input(multiply_1_dst);
    multiply_2.add_input(multiply_2_in_tensor);
    multiply_2.add_output(multiply_2_dst);

    ASSERT_EQ(agraph.add_op(&divide), status::success);
    ASSERT_EQ(agraph.add_op(&erf), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&multiply_1), status::success);
    ASSERT_EQ(agraph.add_op(&multiply_2), status::success);

    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 5);

    pass::pass_base_ptr apass = get_pass("gelu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);
}

struct ut_gelu_params {
    size_t op4_idx;
    size_t op5_idx;
    size_t op9_idx;
    size_t op10_idx;
};

class gelu_test : public ::testing::TestWithParam<ut_gelu_params> {};

TEST_P(gelu_test, gelu_tanh_based_fusion) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    const ut_gelu_params &params = GetParam();

    graph_t agraph;
    op_t *op1 = agraph.create_op(Wildcard);
    op_t *op2 = agraph.create_op(Pow);
    op_t *op3 = agraph.create_op(Wildcard);
    op_t *op4 = agraph.create_op(Multiply);
    op_t *op5 = agraph.create_op(Wildcard);
    op_t *op6 = agraph.create_op(Add);
    op_t *op7 = agraph.create_op(Wildcard);
    op_t *op8 = agraph.create_op(Multiply);
    op_t *op9 = agraph.create_op(Tanh);
    op_t *op10 = agraph.create_op(Wildcard);
    op_t *op11 = agraph.create_op(Add);
    op_t *op12 = agraph.create_op(Wildcard);
    op_t *op13 = agraph.create_op(Multiply);
    op_t *op14 = agraph.create_op(Wildcard);
    op_t *op15 = agraph.create_op(Multiply);
    op2->fill_and_connect_input(0, *op1, 0);
    op4->fill_and_connect_input(0, *op2, 0);
    op4->fill_and_connect_input(1, *op3, 0);
    op6->fill_and_connect_input(params.op4_idx, *op4, 0);
    op6->fill_and_connect_input(params.op5_idx, *op5, 0);
    op8->fill_and_connect_input(0, *op6, 0);
    op8->fill_and_connect_input(1, *op7, 0);
    op9->fill_and_connect_input(0, *op8, 0);
    op11->fill_and_connect_input(params.op9_idx, *op9, 0);
    op11->fill_and_connect_input(params.op10_idx, *op10, 0);
    op13->fill_and_connect_input(0, *op11, 0);
    op13->fill_and_connect_input(1, *op12, 0);
    op15->fill_and_connect_input(0, *op13, 0);
    op15->fill_and_connect_input(1, *op14, 0);

    pass::pass_base_ptr gelu_pass = get_pass("gelu_fusion");
    gelu_pass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);
}

// Parameters below correspond to the indices of the op inputs for Add ops.
INSTANTIATE_TEST_SUITE_P(gelu_test_instance, gelu_test,
        testing::Values(ut_gelu_params {/*multiply_1*/ 0, /*any_2*/ 1,
                                /*tanh*/ 0, /*any_4*/ 1},
                ut_gelu_params {0, 1, 1, 0}, ut_gelu_params {1, 0, 0, 1},
                ut_gelu_params {1, 0, 1, 0}));

TEST(pass_test, single_op_replacement) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    std::vector<op_kind_t> single_op_set_supported = {BatchNormInference, Add,
            ReLU, MatMul, AvgPool, MaxPool, AvgPoolBackprop,
            BatchNormTrainingBackprop, ConvolutionBackpropData,
            ConvolutionBackpropFilters, MaxPoolBackprop, ReLUBackprop,
            GELUBackprop, LogSoftmax, LogSoftmaxBackprop, SoftMax, LayerNorm,
            BatchNormForwardTraining, Elu, Exp, HardTanh, Log, Multiply,
            Maximum, Minimum, Pow, Sqrt, Square, Tanh, SoftMaxBackprop};
    for (auto akind : single_op_set_supported) {
        graph_t agraph;
        op_t *op = agraph.create_op(akind);
        ASSERT_EQ(op->get_kind(), akind);
        pm.run_passes(agraph, "no_config");

        auto orig_op = agraph.get_ops()[0];
        ASSERT_NE(orig_op->get_partition(), nullptr);
        ASSERT_EQ(orig_op->get_partition()->get_assigned_backend()->get_name(),
                std::string("dnnl_backend"));

        auto replaced_op = get_fused_op(agraph.get_partitions()[0]);
        ASSERT_EQ(replaced_op->get_kind(), akind);
    }

    auto &fake_backend_ptr = fake_impl::fake_backend::get_singleton();
    auto fake_pm = pass::pass_manager(fake_backend_ptr.get_pass_registry());
    std::vector<op_kind_t> single_op_set_unsupported = {
            /* not enabling ops = */ Concat, Divide, EluBackprop,
            LayerNormBackprop, Reshape, Round, Sigmoid, SigmoidBackprop,
            SqrtBackprop, TanhBackprop,
            /* no dnnl primitive support = */ BiasAdd, BiasAddBackprop, Clamp,
            ClampBackprop, Erf, HardTanhBackprop, PowBackprop, ReduceSum,
            SoftPlus, SoftPlusBackprop, Wildcard, End, Interpolate,
            InterpolateBackprop, Transpose, Index, PowBackpropExponent};
    for (auto akind : single_op_set_unsupported) {
        graph_t agraph;
        op_t *op = agraph.create_op(akind);
        ASSERT_EQ(op->get_kind(), akind);
        fake_pm.run_passes(agraph, "no_config");

        auto orig_op = agraph.get_ops()[0];
        ASSERT_NE(orig_op->get_partition(), nullptr);
        ASSERT_EQ(orig_op->get_partition()->get_assigned_backend()->get_name(),
                std::string("fake_backend"));

        ASSERT_EQ(agraph.get_partitions().size(), 1);
        auto replaced_op = static_cast<fake_impl::fake_partition_impl_t *>(
                agraph.get_partitions()[0].get())
                                   ->get_fused_op();
        ASSERT_EQ(replaced_op->get_kind(), akind);
    }
}

TEST(pass_test, conv_single_op_replacement) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(3);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 1);

    pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    auto &orig_op = agraph.get_ops()[0];
    ASSERT_NE(orig_op->get_partition(), nullptr);

    auto replaced_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(replaced_op->get_kind(), Convolution);
}

TEST(pass_test, conv_single_op_replacement_case2) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]);
    conv.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 1);

    pass::pass_base_ptr apass = get_pass("conv_bias_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto &orig_op = agraph.get_ops()[0];
    ASSERT_NE(orig_op->get_partition(), nullptr);

    auto replaced_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(replaced_op->get_kind(), conv_bias);
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
    conv1.add_input(lt_vec[0]);
    conv1.add_input(lt_vec[1]);
    conv1.add_output(lt_vec[2]);
    bn.add_input(lt_vec[2]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_input(lt_vec[5]);
    bn.add_input(lt_vec[6]);
    bn.add_output(lt_vec[7]);
    relu.add_input(lt_vec[7]);
    relu.add_output(lt_vec[8]);
    conv2.add_input(lt_vec[9]);
    conv2.add_input(lt_vec[10]);
    conv2.add_output(lt_vec[11]);
    add.add_input(lt_vec[11]);
    add.add_input(lt_vec[8]);
    add.add_output(lt_vec[12]);

    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&conv2), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 5);

    auto &backend_ptr
            = dnnl::graph::impl::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::graph::impl::pass::pass_manager(
            backend_ptr.get_pass_registry());

    pm.print_passes("passes.json");
    pm.run_passes(agraph, "passes.json");
    ASSERT_EQ(agraph.num_ops(), 5);
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
    conv0.add_input(lt_vec[0]);
    conv0.add_input(lt_vec[1]);
    conv0.add_output(lt_vec[2]);
    relu0.add_input(lt_vec[2]);
    relu0.add_output(lt_vec[3]);

    conv1.add_input(lt_vec[3]);
    conv1.add_input(lt_vec[1]);
    conv1.add_output(lt_vec[4]);
    relu1.add_input(lt_vec[4]);
    relu1.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv0), status::success);
    ASSERT_EQ(agraph.add_op(&relu0), status::success);
    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&relu1), status::success);

    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 4);
    pass::pass_base_ptr conv_relu_fusion_pass = get_pass("conv_relu_fusion");
    conv_relu_fusion_pass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 2);
    ASSERT_EQ(agraph.get_num_partitions(), 2);
    for (auto &part : agraph.get_partitions()) {
        ASSERT_EQ(get_fused_op(part)->get_kind(), conv_relu);
        ASSERT_EQ(part->get_inputs().size(), 2);
        ASSERT_EQ(part->get_outputs().size(), 1);
    }
}

TEST(pass_test, multi_values_between_two_ops) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {1, Add, "add"};

    // create lt
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);

    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    add.add_input(lt_vec[2]);
    add.add_input(lt_vec[2]);
    add.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);
    pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(agraph);
    apass = get_pass("sum_pass");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 2);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), Convolution);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[1])->get_kind(), Add);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 2);
}

TEST(pass_test, int8_conv_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);
    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_conv_out = logical_tensor_init(4, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_output(fp32_conv_out);

    logical_tensor_t int8_out = logical_tensor_init(5, data_type::u8);
    quant.add_input(fp32_conv_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_conv_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(), int8_conv);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
}

TEST(pass_test, int8_conv_fail_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
      /    conv
wildcard     | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);
    op_t wildcard {5, Wildcard, "wildcard"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);
    wildcard.add_input(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_conv_out = logical_tensor_init(4, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_output(fp32_conv_out);

    logical_tensor_t int8_out = logical_tensor_init(5, data_type::u8);
    quant.add_input(fp32_conv_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);

    agraph.build_graph();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 4);
}

TEST(pass_test, int8_conv_bias_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(5, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t int8_out = logical_tensor_init(6, data_type::u8);
    quant.add_input(fp32_conv_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_conv_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("int8_conv_bias_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_conv_bias);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
}

TEST(pass_test, int8_conv_relu_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_conv_out = logical_tensor_init(4, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_output(fp32_conv_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(5, data_type::f32);
    relu.add_input(fp32_conv_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(6, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_conv_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_conv_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
}

TEST(pass_test, int8_conv_bias_relu_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(5, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(6, data_type::f32);
    relu.add_input(fp32_conv_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(7, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_conv_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("int8_conv_bias_relu_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_conv_bias_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
}

TEST(pass_test, int8_conv_bias_add_relu_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            conv_with_bias
             | (f32)
             |     | (s8)
             |   dequant
             |  / (f32)
            add
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr("scales", scales);
    dequant3.set_attr("zps", zps);
    op_t conv {3, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {5, Add, "add"};
    set_conv_common_attr(conv);
    op_t relu {6, ReLU, "relu"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t int8_other = logical_tensor_init(4, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(5, data_type::f32);
    dequant3.add_input(int8_other);
    dequant3.add_output(fp32_other);

    logical_tensor_t fp32_bias = logical_tensor_init(6, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(7, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t fp32_add_out = logical_tensor_init(8, data_type::f32);
    add.add_input(fp32_conv_out);
    add.add_input(fp32_other);
    add.add_output(fp32_add_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(9, data_type::f32);
    relu.add_input(fp32_add_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_conv_add_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("int8_conv_bias_add_relu_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_conv_bias_add_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
}

TEST(pass_test, int8_conv_add_test) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            conv_with_bias
             | (f32)
             |     | (s8)
             |   dequant
             |  / (f32)
            add
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    //asymmetric zps
    std::vector<int64_t> zps = {0, 0, 0, 1};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr("scales", scales);
    dequant3.set_attr("zps", zps);
    op_t conv {3, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {5, Add, "add"};
    set_conv_common_attr(conv);
    op_t relu {6, ReLU, "relu"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t int8_other = logical_tensor_init(4, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(5, data_type::f32);
    dequant3.add_input(int8_other);
    dequant3.add_output(fp32_other);

    logical_tensor_t fp32_bias = logical_tensor_init(6, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(7, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t fp32_add_out = logical_tensor_init(8, data_type::f32);
    add.add_input(fp32_conv_out);
    add.add_input(fp32_other);
    add.add_output(fp32_add_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(9, data_type::f32);
    relu.add_input(fp32_add_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_conv_add_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("int8_conv_bias_add_relu_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_conv_bias_add_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
}

TEST(pass_test, x8s8f32_conv_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_conv_out = logical_tensor_init(4, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_output(fp32_conv_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8f32_conv_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), x8s8f32_conv);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
}

TEST(pass_test, x8s8f32_conv_bias_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(5, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8f32_conv_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("x8s8f32_conv_bias_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            x8s8f32_conv_bias);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
}

TEST(pass_test, x8s8f32_conv_relu_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
            relu
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_conv_out = logical_tensor_init(4, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_output(fp32_conv_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(5, data_type::f32);
    relu.add_input(fp32_conv_out);
    relu.add_output(fp32_relu_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8f32_conv_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            x8s8f32_conv_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
}

TEST(pass_test, x8s8f32_conv_bias_relu_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
            relu
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(5, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(6, data_type::f32);
    relu.add_input(fp32_conv_out);
    relu.add_output(fp32_relu_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8f32_conv_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("x8s8f32_conv_bias_relu_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            x8s8f32_conv_bias_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
}

TEST(pass_test, x8s8f32_conv_bias_add_relu_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            conv_with_bias
             | (f32)
             |     | (s8)
             |   dequant
             |  / (f32)
            add
             | (f32)
            relu
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr("scales", scales);
    dequant3.set_attr("zps", zps);
    op_t conv {3, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {5, Add, "add"};
    set_conv_common_attr(conv);
    op_t relu {6, ReLU, "relu"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t int8_other = logical_tensor_init(4, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(5, data_type::f32);
    dequant3.add_input(int8_other);
    dequant3.add_output(fp32_other);

    logical_tensor_t fp32_bias = logical_tensor_init(6, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(7, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t fp32_add_out = logical_tensor_init(8, data_type::f32);
    add.add_input(fp32_conv_out);
    add.add_input(fp32_other);
    add.add_output(fp32_add_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(9, data_type::f32);
    relu.add_input(fp32_add_out);
    relu.add_output(fp32_relu_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8f32_conv_add_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("x8s8f32_conv_bias_add_relu_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            x8s8f32_conv_bias_add_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
}

TEST(pass_test, x8s8f32_conv_add_test) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            conv_with_bias
             | (f32)
             |     | (s8)
             |   dequant
             |  / (f32)
            add
             | (f32)
            relu
             | (f32)
    */
    graph_t agraph;
    //asymmetric zps
    std::vector<int64_t> zps = {0, 0, 0, 1};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr("scales", scales);
    dequant3.set_attr("zps", zps);
    op_t conv {3, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {5, Add, "add"};
    set_conv_common_attr(conv);
    op_t relu {6, ReLU, "relu"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t int8_other = logical_tensor_init(4, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(5, data_type::f32);
    dequant3.add_input(int8_other);
    dequant3.add_output(fp32_other);

    logical_tensor_t fp32_bias = logical_tensor_init(6, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(7, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t fp32_add_out = logical_tensor_init(8, data_type::f32);
    add.add_input(fp32_conv_out);
    add.add_input(fp32_other);
    add.add_output(fp32_add_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(9, data_type::f32);
    relu.add_input(fp32_add_out);
    relu.add_output(fp32_relu_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8f32_conv_add_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("x8s8f32_conv_bias_add_relu_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            x8s8f32_conv_bias_add_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
}

TEST(pass_test, quantized_conv_priority_test) {
    /*
        | (u8/s8)  | (s8)   | (u8/s8)  | (s8)
     dequant    dequant   dequant    dequant
    (f32) \     / (f32)       \ (f32) / (f32)
            conv_with_bias   / conv
             | (f32)        / (f32)
             |            quantize
             |           / (s8)
             |    dequant
             |   / (f32)
            add
             | (f32)
            relu
             | (f32)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    graph_t agraph;
    //asymmetric zps
    std::vector<int64_t> zps = {0, 0, 0, 0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr("scales", scales);
    dequant3.set_attr("zps", zps);
    op_t conv {3, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {4, Add, "add"};
    set_conv_common_attr(conv);
    op_t relu {5, ReLU, "relu"};
    op_t dequant4 {6, Dequantize, "dequant"};
    dequant4.set_attr("scales", scales);
    dequant4.set_attr("zps", zps);
    op_t dequant5 {7, Dequantize, "dequant"};
    dequant5.set_attr("scales", scales);
    dequant5.set_attr("zps", zps);
    op_t quant {8, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);
    op_t conv1 {9, Convolution, "conv"};
    set_conv_common_attr(conv1);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t int8_other = logical_tensor_init(4, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(5, data_type::f32);
    dequant3.add_input(int8_other);
    dequant3.add_output(fp32_other);

    logical_tensor_t fp32_bias = logical_tensor_init(6, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(7, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t fp32_add_out = logical_tensor_init(8, data_type::f32);
    add.add_input(fp32_conv_out);
    add.add_input(fp32_other);
    add.add_output(fp32_add_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(9, data_type::f32);
    relu.add_input(fp32_add_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_data2 = logical_tensor_init(10, data_type::u8);
    logical_tensor_t fp32_data2 = logical_tensor_init(11, data_type::f32);
    dequant4.add_input(int8_data2);
    dequant4.add_output(fp32_data2);

    logical_tensor_t s8_weight2 = logical_tensor_init(12, data_type::s8);
    logical_tensor_t fp32_weight2 = logical_tensor_init(13, data_type::f32);
    dequant5.add_input(s8_weight2);
    dequant5.add_output(fp32_weight2);

    logical_tensor_t fp32_conv_out2 = logical_tensor_init(14, data_type::f32);
    conv1.add_input(fp32_data2);
    conv1.add_input(fp32_weight2);
    conv1.add_output(fp32_conv_out2);

    quant.add_input(fp32_conv_out2);
    quant.add_output(int8_other);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&dequant4), status::success);
    ASSERT_EQ(agraph.add_op(&dequant5), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);

    agraph.build_graph();

    // run all the pass to check if the priority is correct
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 2);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[1])->get_kind(),
            x8s8f32_conv_bias_add_relu);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 4);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(), int8_conv);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
}

TEST(pass_test, int8_matmul_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
           matmul
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t matmul {2, MatMul, "matmul"};
    op_t quant {3, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);
    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_matmul_out = logical_tensor_init(4, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t int8_out = logical_tensor_init(5, data_type::u8);
    quant.add_input(fp32_matmul_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_matmul_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), int8_matmul);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
}

TEST(pass_test, int8_matmul_bias_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
        matmul (w/ bias)
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t matmul {2, MatMul, "matmul"};
    op_t quant {3, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);
    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
    logical_tensor_t fp32_matmul_out = logical_tensor_init(5, data_type::f32);

    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t int8_out = logical_tensor_init(6, data_type::u8);
    quant.add_input(fp32_matmul_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_matmul_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("int8_matmul_bias_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_matmul_bias);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
}

TEST(pass_test, int8_matmul_relu_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
           matmul
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t matmul {2, MatMul, "conv"};
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_matmul_out = logical_tensor_init(4, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(5, data_type::f32);
    relu.add_input(fp32_matmul_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(6, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_matmul_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_matmul_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
}

TEST(pass_test, int8_matmul_bias_relu_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
        matmul (w/ bias)
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t matmul {2, MatMul, "matmul"};
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
    logical_tensor_t fp32_matmul_out = logical_tensor_init(5, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(6, data_type::f32);
    relu.add_input(fp32_matmul_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(7, data_type::u8);
    quant.add_input(fp32_relu_out);

    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_matmul_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("int8_matmul_bias_relu_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_matmul_bias_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
}

TEST(pass_test, x8s8f32_matmul_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
           matmul
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t matmul {2, MatMul, "matmul"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_matmul_out = logical_tensor_init(4, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_output(fp32_matmul_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8f32_matmul_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            x8s8f32_matmul);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
}

TEST(pass_test, x8s8f32_matmul_bias_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
        matmul (w/ bias)
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t matmul {2, MatMul, "matmul"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
    logical_tensor_t fp32_matmul_out = logical_tensor_init(5, data_type::f32);

    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(fp32_matmul_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8f32_matmul_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("x8s8f32_matmul_bias_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            x8s8f32_matmul_bias);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
}

TEST(pass_test, x8s8f32_matmul_eltwise_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
           matmul
             | (f32)
           eltwise
             | (f32)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, op_kind_t>> opkind_pair {
            {ReLU, x8s8f32_matmul_relu}, {Sigmoid, x8s8f32_matmul_sigmoid},
            {GELU, x8s8f32_matmul_gelu}};
    for (auto &p : opkind_pair) {
        graph_t agraph;
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr("scales", scales);
        dequant1.set_attr("zps", zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr("scales", scales);
        dequant2.set_attr("zps", zps);
        op_t matmul {2, MatMul, "conv"};
        op_t eltwise {3, p.first, "eltwise"};

        logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
        logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
        dequant1.add_input(int8_data);
        dequant1.add_output(fp32_data);

        logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
        logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
        dequant2.add_input(s8_weight);
        dequant2.add_output(fp32_weight);

        logical_tensor_t fp32_matmul_out
                = logical_tensor_init(4, data_type::f32);
        matmul.add_input(fp32_data);
        matmul.add_input(fp32_weight);
        matmul.add_output(fp32_matmul_out);

        logical_tensor_t fp32_eltwise_out
                = logical_tensor_init(5, data_type::f32);
        eltwise.add_input(fp32_matmul_out);
        eltwise.add_output(fp32_eltwise_out);

        ASSERT_EQ(agraph.add_op(&dequant1), status::success);
        ASSERT_EQ(agraph.add_op(&dequant2), status::success);
        ASSERT_EQ(agraph.add_op(&matmul), status::success);
        ASSERT_EQ(agraph.add_op(&eltwise), status::success);

        agraph.build_graph();

        pm.run_passes(agraph, "no_config");
        ASSERT_EQ(agraph.get_num_partitions(), 1);
        ASSERT_EQ(
                get_fused_op(agraph.get_partitions()[0])->get_kind(), p.second);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    }
}

TEST(pass_test, x8s8f32_matmul_bias_eltwise_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
        matmul (w/ bias)
             | (f32)
           eltwise
             | (f32)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, op_kind_t>> opkind_pair {
            {ReLU, x8s8f32_matmul_bias_relu},
            {Sigmoid, x8s8f32_matmul_bias_sigmoid},
            {GELU, x8s8f32_matmul_bias_gelu}};
    for (auto &p : opkind_pair) {
        graph_t agraph;
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr("scales", scales);
        dequant1.set_attr("zps", zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr("scales", scales);
        dequant2.set_attr("zps", zps);
        op_t matmul {2, MatMul, "matmul"};
        op_t eltwise {3, p.first, "eltwise"};

        logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
        logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
        dequant1.add_input(int8_data);
        dequant1.add_output(fp32_data);

        logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
        logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
        dequant2.add_input(s8_weight);
        dequant2.add_output(fp32_weight);

        logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
        logical_tensor_t fp32_matmul_out
                = logical_tensor_init(5, data_type::f32);
        matmul.add_input(fp32_data);
        matmul.add_input(fp32_weight);
        matmul.add_input(fp32_bias);
        matmul.add_output(fp32_matmul_out);

        logical_tensor_t fp32_eltwise_out
                = logical_tensor_init(6, data_type::f32);
        eltwise.add_input(fp32_matmul_out);
        eltwise.add_output(fp32_eltwise_out);

        ASSERT_EQ(agraph.add_op(&dequant1), status::success);
        ASSERT_EQ(agraph.add_op(&dequant2), status::success);
        ASSERT_EQ(agraph.add_op(&matmul), status::success);
        ASSERT_EQ(agraph.add_op(&eltwise), status::success);

        agraph.build_graph();

        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1);
        ASSERT_EQ(
                get_fused_op(agraph.get_partitions()[0])->get_kind(), p.second);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    }
}

TEST(pass_test, int8_maxpool_fusion) {
    /*
             | (u8/s8)
          dequant
             | (f32)
           maxpool
             | (f32)
           quant
             | (u8/s8)
    */

    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant {0, Dequantize, "dequant"};
    dequant.set_attr("scales", scales);
    dequant.set_attr("zps", zps);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0};
    std::vector<int64_t> pads_end = {0, 0};
    std::vector<int64_t> kernel = {2, 2};
    op_t maxpool {1, MaxPool, "maxpool"};
    maxpool.set_attr("strides", strides);
    maxpool.set_attr("pads_begin", pads_begin);
    maxpool.set_attr("pads_end", pads_end);
    maxpool.set_attr("kernel", kernel);

    op_t quant {2, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant.add_input(int8_data);
    dequant.add_output(fp32_data);

    logical_tensor_t fp32_maxpool_out = logical_tensor_init(2, data_type::f32);
    maxpool.add_input(fp32_data);
    maxpool.add_output(fp32_maxpool_out);

    logical_tensor_t int8_out = logical_tensor_init(3, data_type::u8);
    quant.add_input(fp32_maxpool_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant), status::success);
    ASSERT_EQ(agraph.add_op(&maxpool), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_maxpool_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), int8_maxpool);
}

TEST(pass_test, int8_avgpool_fusion) {
    /*
             | (u8/s8)
          dequant
             | (f32)
           avgpool
             | (f32)
           quant
             | (u8/s8)
    */

    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant {0, Dequantize, "dequant"};
    dequant.set_attr("scales", scales);
    dequant.set_attr("zps", zps);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0};
    std::vector<int64_t> pads_end = {0, 0};
    std::vector<int64_t> kernel = {2, 2};
    op_t avgpool {1, AvgPool, "avgpool"};
    avgpool.set_attr("strides", strides);
    avgpool.set_attr("pads_begin", pads_begin);
    avgpool.set_attr("pads_end", pads_end);
    avgpool.set_attr("kernel", kernel);
    avgpool.set_attr<bool>("exclude_pad", false);

    op_t quant {2, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant.add_input(int8_data);
    dequant.add_output(fp32_data);

    logical_tensor_t fp32_avgpool_out = logical_tensor_init(2, data_type::f32);
    avgpool.add_input(fp32_data);
    avgpool.add_output(fp32_avgpool_out);

    logical_tensor_t int8_out = logical_tensor_init(3, data_type::u8);
    quant.add_input(fp32_avgpool_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant), status::success);
    ASSERT_EQ(agraph.add_op(&avgpool), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_avgpool_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), int8_avgpool);
}

TEST(pass_test, int8_matmul_bias_add_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            matmul_with_bias
             | (f32)
             |     | (s8)
             |   dequant
             |  / (f32)
            add
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps {0, 1};
    std::vector<float> scales {3.1f, 3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr("scales", scales);
    dequant3.set_attr("zps", zps);
    op_t matmul {3, MatMul, "matmul"};
    matmul.set_attr<bool>("transpose_a", false);
    matmul.set_attr<bool>("transpose_b", false);
    op_t add {4, Add, "add"};
    op_t quant {5, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t int8_other = logical_tensor_init(4, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(5, data_type::f32);
    dequant3.add_input(int8_other);
    dequant3.add_output(fp32_other);

    logical_tensor_t fp32_bias = logical_tensor_init(6, data_type::f32);
    logical_tensor_t fp32_matmul_out = logical_tensor_init(7, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t fp32_add_out = logical_tensor_init(8, data_type::f32);
    add.add_input(fp32_matmul_out);
    add.add_input(fp32_other);
    add.add_output(fp32_add_out);

    logical_tensor_t int8_out = logical_tensor_init(9, data_type::u8);
    quant.add_input(fp32_add_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_matmul_add_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("int8_matmul_bias_add_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_matmul_bias_add);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
}

TEST(pass_test, relu_add_fusion) {
    graph_t agraph;

    op_t relu {0, ReLU, "relu"};
    op_t add {1, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    relu.add_input(lt_vec[0]);
    relu.add_output(lt_vec[1]);
    add.add_input(lt_vec[1]);
    add.add_input(lt_vec[2]);
    add.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    auto &backend_ptr
            = dnnl::graph::impl::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::graph::impl::pass::pass_manager(
            backend_ptr.get_pass_registry());

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(), relu_add);
}

TEST(pass_test, x8s8f32_matmul_bias_add_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            matmul_with_bias
             | (f32)
             |     | (s8)
             |   dequant
             |  / (f32)
            add
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps {0, 1};
    std::vector<float> scales {3.1f, 3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr("scales", scales);
    dequant3.set_attr("zps", zps);
    op_t matmul {3, MatMul, "matmul"};
    matmul.set_attr<bool>("transpose_a", false);
    matmul.set_attr<bool>("transpose_b", false);
    op_t add {4, Add, "add"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t int8_other = logical_tensor_init(4, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(5, data_type::f32);
    dequant3.add_input(int8_other);
    dequant3.add_output(fp32_other);

    logical_tensor_t fp32_bias = logical_tensor_init(6, data_type::f32);
    logical_tensor_t fp32_matmul_out = logical_tensor_init(7, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t fp32_add_out = logical_tensor_init(8, data_type::f32);
    add.add_input(fp32_matmul_out);
    add.add_input(fp32_other);
    add.add_output(fp32_add_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8f32_matmul_add_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("x8s8f32_matmul_bias_add_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            x8s8f32_matmul_bias_add);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
}

TEST(pass_test, x8s8f32_matmul_bias_add_fusion_with_dtype_check) {
    /*
        | (u8/s8)  | (u8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            matmul_with_bias
             | (f32)
             |     | (s8)
             |   dequant
             |  / (f32)
            add
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps {0, 0};
    std::vector<float> scales {3.1f, 3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr("scales", scales);
    dequant3.set_attr("zps", zps);
    op_t matmul {3, MatMul, "matmul"};
    matmul.set_attr<bool>("transpose_a", false);
    matmul.set_attr<bool>("transpose_b", false);
    op_t add {4, Add, "add"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t u8_weight = logical_tensor_init(2, data_type::u8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(u8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t int8_other = logical_tensor_init(4, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(5, data_type::f32);
    dequant3.add_input(int8_other);
    dequant3.add_output(fp32_other);

    logical_tensor_t fp32_bias = logical_tensor_init(6, data_type::f32);
    logical_tensor_t fp32_matmul_out = logical_tensor_init(7, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t fp32_add_out = logical_tensor_init(8, data_type::f32);
    add.add_input(fp32_matmul_out);
    add.add_input(fp32_other);
    add.add_output(fp32_add_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8f32_matmul_bias_add_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);
}
