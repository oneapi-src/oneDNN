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
    /*   conv
          |
         bn
    */
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(8);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::conv_bn);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7);
}

TEST(pass_test, conv_bn_fusion_fail) {
    /*   conv
          |
        bias
          |
         bn
    */
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
    /*   conv
          |
         relu
    */
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::conv_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
}

TEST(pass_test, conv_relu_fusion_fail) {
    /*   conv
          |
         bias
          |
         relu
    */
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
    /*   conv
        /   \
     relu   relu
    */
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
    /*   conv
          |
         bias
    */
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

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, conv_bias_fusion_case2) {
    /*   conv
          |
         bias
          |
         bias
    */
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
    // bias op can't be fused since the post conv already has bias input.
    // so only three inputs
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
}

TEST(pass_test, conv_sum_fusion) {
    /*   conv
           \  /
           add
    */
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

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, conv_sum_fusion_fail) {
    /*   conv
           |
         bias
           \  /
           add
    */
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
    /*   conv
          |
         bias
          |
         bn
    */
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::conv_bias_bn);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 8);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9);
}

TEST(pass_test, conv_bias_bn_fusion_case2) {
    /*   conv
          |
         bias
          |
         bn
    */
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::conv_bias_bn);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 7);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8);
}

TEST(pass_test, conv_bias_relu_fusion) {
    /*   conv
          |
         bias
          |
         relu
    */
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
            dnnl_impl::op_kind::conv_bias_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, conv_bias_relu_fusion_case2) {
    /*   conv
          |
         bias
          |
         relu
    */
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
            dnnl_impl::op_kind::conv_bias_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, conv_bias_relu6_fusion) {
    /*   conv
          |
         bias
          |
         relu6
    */
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
            dnnl_impl::op_kind::conv_bias_relu6);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, conv_bias_relu6_fusion_fail) {
    /*   conv
          |
         bias
          |
         hardtanh
    */
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

TEST(pass_priority_test, conv_bias_relu6_fusion) {
    pass::pass_base_ptr pass1 = get_pass("conv_bias_relu6_fusion");
    pass::pass_base_ptr pass2 = get_pass("conv_bias_hardtanh_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_bias_fusion");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
}

TEST(pass_test, conv_bias_elu_fusion) {
    /*   conv
          |
         bias
          |
         elu
    */
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
            dnnl_impl::op_kind::conv_bias_elu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, conv_bias_sigmoid_fusion) {
    /*   conv
          |
         bias
          |
         sigmoid
    */
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
            dnnl_impl::op_kind::conv_bias_sigmoid);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, conv_bias_swish_fusion) {
    // swish: f(x) = x * sigmoid(x)
    /*   conv
          |
         bias
        /    |
    sigmoid  |
        \    |
        multiply

    */
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
            dnnl_impl::op_kind::conv_bias_swish);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, conv_bias_swish_fail_fusion) {
    // swish: f(x) = x * sigmoid(x)
    /*   conv
          |
         bias
        /    |
    sigmoid  |
        \    |
        multiply

    */
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

    pass::pass_base_ptr apass = get_pass("conv_bias_sigmoid_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);
}

TEST(pass_test, conv_bias_hardtanh_fusion) {
    /*   conv
          |
         bias
          |
       hardtanh
    */
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
            dnnl_impl::op_kind::conv_bias_hardtanh);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, conv_bias_square_fusion) {
    /*   conv
          |
         bias
          |
        square
    */
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
            dnnl_impl::op_kind::conv_bias_square);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, conv_bias_tanh_fusion) {
    /*   conv
          |
         bias
          |
         tanh
    */
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
            dnnl_impl::op_kind::conv_bias_tanh);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, conv_bias_abs_fusion) {
    /*   conv
          |
         bias
          |
         abs
    */
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
            dnnl_impl::op_kind::conv_bias_abs);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, conv_bias_sqrt_fusion) {
    /*   conv
          |
         bias
          |
         sqrt
    */
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
            dnnl_impl::op_kind::conv_bias_sqrt);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, conv_bias_sum_fusion) {
    /*   conv
          |
         bias
           \   /
            add
    */
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
            dnnl_impl::op_kind::conv_bias_add);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6);
}

TEST(pass_test, conv_bias_sum_fusion_case2) {
    /*   conv
          |
         bias
           \   /
            add
    */
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
            dnnl_impl::op_kind::conv_bias_add);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, conv_bias_sum_relu_fusion) {
    /*   conv
          |
         bias
           \   /
            add
             |
            relu
    */
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
            dnnl_impl::op_kind::conv_bias_add_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7);
}

TEST(pass_priority_test, conv_bias_sum_relu) {
    /*   conv
          |
         bias conv
           \   /
            add
             |
            relu    should be fused to conv-bias-add-relu + conv
    */
    pass::pass_base_ptr pass1 = get_pass("conv_bias_sum_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("conv_bias_sum_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_bias_fusion");
    pass::pass_base_ptr pass4 = get_pass("conv_sum_relu_fusion");
    pass::pass_base_ptr pass5 = get_pass("binary_add_relu_fusion");
    pass::pass_base_ptr pass6 = get_pass("conv_sum_fusion");
    pass::pass_base_ptr pass7 = get_pass("conv_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass4->get_priority() > pass5->get_priority());
    ASSERT_TRUE(pass4->get_priority() > pass6->get_priority());
    ASSERT_TRUE(pass6->get_priority() > pass7->get_priority());
    ASSERT_TRUE(pass3->get_priority() > pass7->get_priority());
}

TEST(pass_test, conv_bias_sum_elu_fusion) {
    /*   conv
          |
         bias
           \   /
            add
             |
            elu
    */
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
            dnnl_impl::op_kind::conv_bias_add_elu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7);
}

TEST(pass_priority_test, conv_bias_sum_elu) {
    /*   conv
          |
         bias conv
           \   /
            add
             |
            elu    should be fused to conv-bias-add-elu + conv
    */
    pass::pass_base_ptr pass1 = get_pass("conv_bias_sum_elu_fusion");
    pass::pass_base_ptr pass2 = get_pass("conv_bias_sum_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_bias_fusion");
    pass::pass_base_ptr pass4 = get_pass("conv_sum_elu_fusion");
    pass::pass_base_ptr pass5 = get_pass("conv_sum_fusion");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass4->get_priority() > pass5->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass5->get_priority());
}

TEST(pass_test, conv_bias_sum_relu6_fusion) {
    /*   conv
          |
         bias
           \   /
            add
             |
            relu6
    */
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
            dnnl_impl::op_kind::conv_bias_add_relu6);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7);
}

TEST(pass_priority_test, conv_bias_sum_relu6) {
    /*   conv
          |
         bias conv
           \   /
            add
             |
            relu6    should be fused to conv-bias-add-relu6 + conv
    */
    pass::pass_base_ptr pass1 = get_pass("conv_bias_sum_relu6_fusion");
    pass::pass_base_ptr pass2 = get_pass("conv_bias_sum_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_bias_fusion");
    pass::pass_base_ptr pass4 = get_pass("conv_sum_relu6_fusion");
    pass::pass_base_ptr pass5 = get_pass("conv_sum_fusion");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass4->get_priority() > pass5->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass5->get_priority());
}

TEST(pass_test, binary_sum_fusion) {
    /* binary here represents Multiply, Minimum, Maximum

        binary
           \   /
            add
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, op_kind_t>> opkind_pair {
            {Multiply, dnnl_impl::op_kind::multiply_add},
            {Maximum, dnnl_impl::op_kind::maximum_add},
            {Minimum, dnnl_impl::op_kind::minimum_add}};

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

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
    }
}

TEST(pass_priority_test, binary_sum_fusion) {
    /* binary here represents Multiply, Minimum, Maximum

        binary conv
           \   /
            add        should be fused to conv-add + binary
    */
    pass::pass_base_ptr pass1 = get_pass("conv_sum_fusion");
    pass::pass_base_ptr pass2 = get_pass("binary_multiply_add_fusion");
    pass::pass_base_ptr pass3 = get_pass("sum_pass");
    pass::pass_base_ptr pass4 = get_pass("mul_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());

    pass::pass_base_ptr pass5 = get_pass("binary_maximum_add_fusion");
    pass::pass_base_ptr pass6 = get_pass("max_pass");
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
    ASSERT_TRUE(pass5->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass5->get_priority() > pass6->get_priority());

    pass::pass_base_ptr pass7 = get_pass("binary_minimum_add_fusion");
    pass::pass_base_ptr pass8 = get_pass("min_pass");
    ASSERT_TRUE(pass1->get_priority() > pass7->get_priority());
    ASSERT_TRUE(pass7->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass7->get_priority() > pass8->get_priority());
}

TEST(pass_test, binary_sum_fusion_supported_broadcast) {
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, op_kind_t>> opkind_pair {
            {Multiply, dnnl_impl::op_kind::multiply_add},
            {Maximum, dnnl_impl::op_kind::maximum_add},
            {Minimum, dnnl_impl::op_kind::minimum_add}};

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
            {Multiply, dnnl_impl::op_kind::multiply_add},
            {Maximum, dnnl_impl::op_kind::maximum_add},
            {Minimum, dnnl_impl::op_kind::minimum_add}};

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
            {Multiply, dnnl_impl::op_kind::multiply_add},
            {Maximum, dnnl_impl::op_kind::maximum_add},
            {Minimum, dnnl_impl::op_kind::minimum_add}};

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

TEST(pass_test, binary_add_mul_fusion) {
    /*
         \  /
          add
           \   /
            mul
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());

    graph_t agraph;
    op_t add {0, Add, "add"};
    op_t mul {1, Multiply, "mul"};

    std::vector<logical_tensor_t> lt_vec;
    lt_vec.reserve(5);
    for (size_t i = 0; i < 5; i++)
        lt_vec.emplace_back(
                logical_tensor_init(i, {2, 3, 4, 5}, data_type::f32));

    add.add_input(lt_vec[0]);
    add.add_input(lt_vec[1]);
    add.add_output(lt_vec[2]);
    mul.add_input(lt_vec[3]); // need swap input
    mul.add_input(lt_vec[2]);
    mul.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&mul), status::success);
    agraph.build_graph();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::add_multiply);
}

TEST(pass_test, binary_eltwise_fusion) {
    /* binary here represents Add, Multiply, Minimum, Maximum
       eltwise here represents Sigmoid, ReLU

         \  /
        binary
           |
        eltwise
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    std::vector<std::pair<std::pair<op_kind_t, op_kind_t>, op_kind_t>>
            opkind_pair {{{Add, Sigmoid}, dnnl_impl::op_kind::add_sigmoid},
                    {{Add, ReLU}, dnnl_impl::op_kind::add_relu},
                    {{Multiply, Sigmoid}, dnnl_impl::op_kind::multiply_sigmoid},
                    {{Multiply, ReLU}, dnnl_impl::op_kind::multiply_relu},
                    {{Maximum, Sigmoid}, dnnl_impl::op_kind::maximum_sigmoid},
                    {{Maximum, ReLU}, dnnl_impl::op_kind::maximum_relu},
                    {{Minimum, Sigmoid}, dnnl_impl::op_kind::minimum_sigmoid},
                    {{Minimum, ReLU}, dnnl_impl::op_kind::minimum_relu}};

    for (auto &p : opkind_pair) {
        graph_t agraph;
        auto binary_kind = p.first.first;
        auto eltwise_kind = p.first.second;

        op_t binary {0, binary_kind, "binary"};
        op_t eltwise {1, eltwise_kind, "eltwise"};

        std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
        binary.add_input(lt_vec[0]);
        binary.add_input(lt_vec[1]);
        binary.add_output(lt_vec[2]);
        eltwise.add_input(lt_vec[2]);
        eltwise.add_output(lt_vec[3]);

        ASSERT_EQ(agraph.add_op(&binary), status::success);
        ASSERT_EQ(agraph.add_op(&eltwise), status::success);
        agraph.build_graph();

        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1);

        auto fused_op = get_fused_op(agraph.get_partitions()[0]);
        ASSERT_EQ(fused_op->get_kind(), p.second);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
    }
}

TEST(pass_priority_test, binary_eltwise_fusion) {
    /* binary here represents Add, Multiply, Minimum, Maximum
       eltwise here represents Sigmoid, ReLU

        \    /
        binary
           |
        eltwise
    */
    pass::pass_base_ptr pass1 = get_pass("binary_add_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("conv_sum_relu_fusion");
    pass::pass_base_ptr pass3 = get_pass("sum_pass");
    pass::pass_base_ptr pass4 = get_pass("relu_pass");
    ASSERT_TRUE(pass1->get_priority() < pass2->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass4->get_priority());

    pass::pass_base_ptr pass5 = get_pass("binary_add_sigmoid_fusion");
    ASSERT_TRUE(pass5->get_priority() > pass3->get_priority());

    pass::pass_base_ptr pass6 = get_pass("binary_multiply_relu_fusion");
    pass::pass_base_ptr pass7 = get_pass("mul_pass");
    ASSERT_TRUE(pass6->get_priority() > pass7->get_priority());
    ASSERT_TRUE(pass6->get_priority() > pass4->get_priority());

    pass::pass_base_ptr pass8 = get_pass("binary_mul_sigmoid_fusion");
    ASSERT_TRUE(pass8->get_priority() > pass7->get_priority());

    pass::pass_base_ptr pass9 = get_pass("binary_maximum_relu_fusion");
    pass::pass_base_ptr pass10 = get_pass("max_pass");
    ASSERT_TRUE(pass9->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass9->get_priority() > pass10->get_priority());

    pass::pass_base_ptr pass11 = get_pass("binary_max_sigmoid_fusion");
    ASSERT_TRUE(pass11->get_priority() > pass10->get_priority());
}

TEST(pass_test, bn_relu_fusion) {
    /*
         bn
         |
        relu
    */
    graph_t agraph;
    op_t bn {0, BatchNormInference, "bn"};
    bn.set_attr("epsilon", 0.001f);
    op_t relu {1, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(7);
    bn.add_input(lt_vec[0]);
    bn.add_input(lt_vec[1]);
    bn.add_input(lt_vec[2]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[4]);
    bn.add_output(lt_vec[5]);
    relu.add_input(lt_vec[5]);
    relu.add_output(lt_vec[6]);

    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("bn_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::bn_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6);
}

TEST(pass_priority_test, bn_relu) {
    /*
         bn
         |
        relu
    */
    pass::pass_base_ptr pass1 = get_pass("bn_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("bn_pass");
    pass::pass_base_ptr pass3 = get_pass("relu_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass3->get_priority());
}

TEST(pass_test, bn_bwd_relu_bwd_fusion) {
    /*
        ReLUBackprop
         |
        BatchNormTrainingBackprop
    */
    graph_t agraph;
    op_t op1 {0, ReLUBackprop, "op1"};
    op_t op2 {1, BatchNormTrainingBackprop, "op2"};
    op2.set_attr("epsilon", 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(11);
    op1.add_input(lt_vec[0]);
    op1.add_input(lt_vec[1]);
    op1.add_output(lt_vec[2]);

    op2.add_input(lt_vec[2]);
    op2.add_input(lt_vec[3]);
    op2.add_input(lt_vec[4]);
    op2.add_input(lt_vec[5]);
    op2.add_input(lt_vec[6]);
    op2.add_input(lt_vec[7]);
    op2.add_output(lt_vec[8]);
    op2.add_output(lt_vec[9]);
    op2.add_output(lt_vec[10]);

    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("bn_bwd_relu_bwd_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::bn_bwd_relu_bwd);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 7);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[1].id, 9);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[2].id, 10);
}

TEST(pass_priority_test, bn_bwd_relu_bwd) {
    /*
        ReLUBackprop
         |
        BatchNormTrainingBackprop
    */
    pass::pass_base_ptr pass1 = get_pass("bn_bwd_relu_bwd_fusion");
    pass::pass_base_ptr pass2 = get_pass("bn_bw_pass");
    pass::pass_base_ptr pass3 = get_pass("relu_bw_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass3->get_priority());
}

TEST(pass_test, conv_sum_relu_fusion) {
    /*   conv
           \   /
            add
             |
            relu
    */
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
            dnnl_impl::op_kind::conv_add_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, conv_sum_elu_fusion) {
    /*   conv
           \   /
            add
             |
            elu
    */
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::conv_add_elu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, conv_sum_relu6_fusion) {
    /*   conv
           \   /
            add
             |
            relu6
    */
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
            dnnl_impl::op_kind::conv_add_relu6);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
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
            dnnl_impl::op_kind::conv_bias_add);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6);

    ASSERT_EQ(get_fused_op(agraph.get_partitions()[1])->get_kind(),
            dnnl_impl::op_kind::conv_bias_add);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[0].id, 7);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[1].id, 8);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[2].id, 10);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[3].id, 6);

    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs()[0].id, 12);
}

TEST(pass_test, conv_bn_sum_fusion) {
    /*   conv
          |
          bn
           \   /
            add
    */
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::conv_bn_add);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 8);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9);
}

TEST(pass_test, conv_bn_sum_fusion_case2) {
    /*   conv
          |
          bn   relu
           \   /
            add
    */
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::conv_bn_add);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 9);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10);
}

TEST(pass_test, conv_bn_sum_fusion_fail) {
    /*   conv
          |
         bias
          |
          bn
           \   /
            add
    */
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
    /*   conv
          |
         bias
          |
          bn
           \   /
            add
    */
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
            dnnl_impl::op_kind::conv_bias_bn_add);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 8);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[7].id, 9);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10);
}

TEST(pass_priority_test, conv_bias_bn_sum) {
    /*   conv
          |
         bias  conv
          |     |
          bn   bn
           \   /
            add   should be fused to conv_bias_bn_add + conv_bn
    */
    pass::pass_base_ptr pass1 = get_pass("conv_bias_bn_sum_fusion");
    pass::pass_base_ptr pass2 = get_pass("conv_bias_bn_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_bias_fusion");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
}

TEST(pass_test, conv_bn_relu_fusion) {
    /*   conv
          |
          bn
          |
         relu
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::conv_bn_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8);
}

TEST(pass_priority_test, conv_bn_relu) {
    /*   conv
          |
         bn
          |
         relu
    */
    pass::pass_base_ptr pass1 = get_pass("conv_bn_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("conv_bn_fusion");
    pass::pass_base_ptr pass3 = get_pass("bn_relu_fusion");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass3->get_priority());
}

TEST(pass_test, conv_bias_bn_relu_fusion) {
    /*   conv
          |
         bias
          |
         bn
          |
         relu
    */
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
            dnnl_impl::op_kind::conv_bias_bn_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 8);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10);
}

TEST(pass_test, conv_bias_bn_relu_fusion_case2) {
    /*   conv
          |
         bias
          |
         bn
          |
         relu
    */
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
            dnnl_impl::op_kind::conv_bias_bn_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 7);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9);
}

TEST(pass_priority_test, conv_bias_bn_relu) {
    /*   conv
          |
         bias
          |
         bn
          |
         relu
    */
    pass::pass_base_ptr pass1 = get_pass("conv_bias_bn_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("conv_bias_bn_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_bias_fusion");
    pass::pass_base_ptr pass4 = get_pass("bn_relu_fusion");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass4->get_priority());
}

TEST(pass_test, conv_bn_sum_relu_fusion) {
    /*   conv
          |
         bn
           \  /
           add
            |
          relu
    */
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
            dnnl_impl::op_kind::conv_bn_add_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 8);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10);
}

TEST(pass_test, conv_bias_bn_sum_relu_fusion) {
    /*   conv
          |
         bias
          |
         bn
           \  /
           add
            |
          relu
    */
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
            dnnl_impl::op_kind::conv_bias_bn_add_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 8);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[7].id, 9);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 11);
}

TEST(pass_priority_test, conv_bias_bn_sum_relu) {
    /*   conv
          |
         bias
          |
         bn
           \  /
           add
            |
          relu
    */
    pass::pass_base_ptr pass1 = get_pass("conv_bias_bn_sum_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("conv_bias_bn_sum_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_bias_bn_fusion");
    pass::pass_base_ptr pass4 = get_pass("conv_bias_fusion");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass3->get_priority() > pass4->get_priority());
}

TEST(pass_test, convtranspose_bias_fusion) {
    /*   conv
          |
         bias
    */
    graph_t agraph;
    op_t convtranspose {0, ConvTranspose, "convtranspose"};
    set_conv_common_attr(convtranspose);
    op_t bias {1, BiasAdd, "bias"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    convtranspose.add_input(lt_vec[0]);
    convtranspose.add_input(lt_vec[1]);
    convtranspose.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&convtranspose), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("convtranspose_bias_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, matmul_relu_fusion) {
    /*  matmul
          |
        relu
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
}

TEST(pass_test, matmul_relu_fail_fusion) {
    /*  matmul
          |
        relu
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_bias_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, relu_matmul) {
    /*  relu
          |
        matmul
    */
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
    /*  matmul
          |
        elu
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_elu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
}

TEST(pass_test, matmul_sigmoid_fusion) {
    /*  matmul
          |
        sigmoid
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_sigmoid);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
}

TEST(pass_test, matmul_hardtanh_fusion) {
    /*  matmul
          |
        hardtanh
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_hardtanh);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
}

TEST(pass_test, matmul_gelu_fusion) {
    /*  matmul
          |
        gelu
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_gelu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
}

TEST(pass_test, matmul_sum_fusion) {
    /*  matmul  wildcard
          \    /
            add
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_add);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, matmul_sum_fusion_opposite_order) {
    /* wildcard matmul
          \    /
            add
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_add);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, matmul_sum_gelu_fusion) {
    /*  matmul  wildcard
          \    /
            add
             |
            gelu
    */
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t wildcard {1, Wildcard, "wildcard"};
    op_t add {2, Add, "add"};
    op_t gelu {3, GELU, "gelu"};
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_add_gelu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_priority_test, matmul_sum_gelu) {
    /*  matmul
          \    /
            add
             |
            gelu
    */
    pass::pass_base_ptr pass1 = get_pass("matmul_sum_gelu_fusion");
    pass::pass_base_ptr pass2 = get_pass("matmul_sum_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
}

TEST(pass_test, matmul_sum_relu_fusion) {
    /*  matmul wildcard
          \    /
            add
             |
            relu
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_add_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_priority_test, matmul_sum_relu) {
    /*  matmul
          \    /
            add
             |
            relu
    */
    pass::pass_base_ptr pass1 = get_pass("matmul_sum_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("matmul_sum_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
}

TEST(pass_test, conv_bwd_f_biasadd_bwd_fusion) {
    /*  ConvolutionBackpropFilters
              |
        BiasAddBackprop
    */
    graph_t agraph;
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    op_t op1 {0, ConvolutionBackpropFilters, "op1"};
    set_conv_common_attr(op1);
    op_t op2 {1, BiasAddBackprop, "op2"};

    op1.add_input(lt_vec[0]);
    op1.add_input(lt_vec[1]);
    op1.add_input(lt_vec[2]);
    op1.add_output(lt_vec[3]);
    op2.add_input(lt_vec[3]);
    op2.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    pass::pass_base_ptr apass = get_pass("conv_bwd_f_biasadd_bwd_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    auto fused_op = get_fused_op(agraph.get_partitions()[0]);
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::conv_bwd_f_biasadd_bwd);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, matmul_bias_fusion) {
    /*  matmul
           |
         bias
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_bias);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, matmul_bias_fusion_case2) {
    /*  matmul
           |
         bias
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_bias);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
}

TEST(pass_test, matmul_bias_sigmoid_fusion) {
    /*  matmul
           |
         bias
           |
        sigmoid
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_bias_sigmoid);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_priority_test, matmul_bias_sigmoid) {
    /*  matmul
           |
         bias
           |
        sigmoid
    */
    pass::pass_base_ptr pass1 = get_pass("matmul_bias_sigmoid_fusion");
    pass::pass_base_ptr pass2 = get_pass("matmul_bias_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
}

TEST(pass_test, matmul_bias_elu_fusion) {
    /*  matmul
           |
         bias
           |
          elu
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_bias_elu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, matmul_bias_relu_fusion) {
    /*  matmul
           |
         bias
           |
          relu
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_bias_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_test, matmul_bias_hardtanh_fusion) {
    /*  matmul
           |
         bias
           |
        hardtanh
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_bias_hardtanh);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
}

TEST(pass_test, matmul_relu_sum_fusion) {
    /*  matmul
           |
         bias  relu
           \   /
            add
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_bias_add);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6);
}

TEST(pass_test, matmul_bias_sum_relu_fusion) {
    /*  matmul
           |
         bias  wildcard
           \   /
            add
             |
            relu
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_bias_add_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7);
}

TEST(pass_priority_test, matmul_bias_sum_relu) {
    /*  matmul
           |
         bias  wildcard
           \   /
            add
             |
            relu
    */
    pass::pass_base_ptr pass1 = get_pass("matmul_bias_sum_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("matmul_bias_sum_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_bias_fusion");
    pass::pass_base_ptr pass4 = get_pass("binary_add_relu_fusion");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass4->get_priority());
}

TEST(pass_test, matmul_bias_swish_fusion) {
    /*       matmul
               |
              bias
              /  \
        sigmoid   |
              \  /
            multiply
                |
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_bias_swish);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6);
}

TEST(pass_priority_test, matmul_bias_swish) {
    /*       matmul
               |
              bias
              /  \
        sigmoid   |
              \  /
            multiply
                |
    */
    pass::pass_base_ptr pass1 = get_pass("matmul_bias_swish_fusion");
    pass::pass_base_ptr pass2 = get_pass("matmul_bias_sigmoid_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_bias_fusion");
    pass::pass_base_ptr pass4 = get_pass("matmul_pass");
    pass::pass_base_ptr pass5 = get_pass("mul_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass3->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
}

TEST(pass_test, matmul_bias_relu6_fusion) {
    /*  matmul
           |
         bias
           |
         relu6
    */
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
    ASSERT_EQ(fused_op->get_kind(), dnnl_impl::op_kind::matmul_bias_relu6);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
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

TEST(pass_priority_test, matmul_bias_relu6) {
    /*  matmul
           |
         bias
           |
         relu6
    */
    pass::pass_base_ptr pass1 = get_pass("matmul_bias_relu6_fusion");
    pass::pass_base_ptr pass2 = get_pass("matmul_bias_hardtanh_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_bias_fusion");
    pass::pass_base_ptr pass4 = get_pass("matmul_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass3->get_priority() > pass4->get_priority());
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
    /*   \  /
        Divide
           |
          Erf
           \ /
           Add
             \      /
             Multiply
                \     /
                Multiply
                   |
    */
    graph_t agraph;
    op_t op0 {0, Divide, "op0"};
    op_t op1 {1, Erf, "op1"};
    op_t op2 {2, Add, "op2"};
    op_t op3 {3, Multiply, "op3"};
    op_t op4 {4, Multiply, "op4"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(10);
    op0.add_input(lt_vec[0]);
    op0.add_input(lt_vec[1]);
    op0.add_output(lt_vec[2]);
    op1.add_input(lt_vec[2]);
    op1.add_output(lt_vec[3]);
    op2.add_input(lt_vec[3]);
    op2.add_input(lt_vec[4]);
    op2.add_output(lt_vec[5]);
    op3.add_input(lt_vec[5]);
    op3.add_input(lt_vec[6]);
    op3.add_output(lt_vec[7]);
    op4.add_input(lt_vec[7]);
    op4.add_input(lt_vec[8]);
    op4.add_output(lt_vec[9]);

    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    ASSERT_EQ(agraph.add_op(&op3), status::success);
    ASSERT_EQ(agraph.add_op(&op4), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 5);

    pass::pass_base_ptr apass = get_pass("gelu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 8);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9);
}

TEST(pass_test, gelu_tanh_based_fusion) {
    /*   \  /
          Pow
            \    /
           Multiply
             \      /
                Add
                  \     /
                  Multiply
                     |
                    Tanh
                      \    /
                        Add
                         \      /
                          Multiply
                              \     /
                              Multiply
                                 |

    */
    graph_t agraph;
    op_t op0 {0, Pow, "op0"};
    op_t op1 {1, Multiply, "op1"};
    op_t op2 {2, Add, "op2"};
    op_t op3 {3, Multiply, "op3"};
    op_t op4 {4, Tanh, "op4"};
    op_t op5 {5, Add, "op5"};
    op_t op6 {6, Multiply, "op6"};
    op_t op7 {7, Multiply, "op7"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(16);
    op0.add_input(lt_vec[0]);
    op0.add_input(lt_vec[1]);
    op0.add_output(lt_vec[2]);
    op1.add_input(lt_vec[2]);
    op1.add_input(lt_vec[3]);
    op1.add_output(lt_vec[4]);
    op2.add_input(lt_vec[4]);
    op2.add_input(lt_vec[5]);
    op2.add_output(lt_vec[6]);
    op3.add_input(lt_vec[6]);
    op3.add_input(lt_vec[7]);
    op3.add_output(lt_vec[8]);
    op4.add_input(lt_vec[8]);
    op4.add_output(lt_vec[9]);
    op5.add_input(lt_vec[9]);
    op5.add_input(lt_vec[10]);
    op5.add_output(lt_vec[11]);
    op6.add_input(lt_vec[11]);
    op6.add_input(lt_vec[12]);
    op6.add_output(lt_vec[13]);
    op7.add_input(lt_vec[13]);
    op7.add_input(lt_vec[14]);
    op7.add_output(lt_vec[15]);

    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    ASSERT_EQ(agraph.add_op(&op3), status::success);
    ASSERT_EQ(agraph.add_op(&op4), status::success);
    ASSERT_EQ(agraph.add_op(&op5), status::success);
    ASSERT_EQ(agraph.add_op(&op6), status::success);
    ASSERT_EQ(agraph.add_op(&op7), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 8);

    pass::pass_base_ptr gelu_pass = get_pass("gelu_fusion");
    gelu_pass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 8);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 7);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 10);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 12);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[7].id, 14);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 15);
}

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

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2);
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
    ASSERT_EQ(replaced_op->get_kind(), dnnl_impl::op_kind::conv_bias);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
}

TEST(pass_test, save_load_json) {
    /*   \  /
          conv
            |
           bn
            |
           relu  conv
             \    /
               add
    */
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

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 9);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 10);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 8);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 12);

    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 6);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[1].id, 1);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[2].id, 3);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[3].id, 4);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[4].id, 5);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[5].id, 6);

    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs()[0].id, 8);
}

TEST(pass_test, valid_json) {
    /*   \   /
          conv
           |
          relu
    */
    graph_t agraph;
    op_t conv1 {0, Convolution, "conv"};
    set_conv_common_attr(conv1);
    op_t relu {1, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    conv1.add_input(lt_vec[0]);
    conv1.add_input(lt_vec[1]);
    conv1.add_output(lt_vec[2]);
    relu.add_input(lt_vec[2]);
    relu.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    auto &backend_ptr
            = dnnl::graph::impl::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::graph::impl::pass::pass_manager(
            backend_ptr.get_pass_registry());

    std::ostringstream valid_stream;
    std::string version = std::to_string(dnnl_graph_version()->major) + "."
            + std::to_string(dnnl_graph_version()->minor) + "."
            + std::to_string(dnnl_graph_version()->patch);
    valid_stream << "{\n"
                 << "\"version\": \"" << version << "\",\n"
                 << "\"hash\": \"" << dnnl_graph_version()->hash << "\",\n"
                 << "\"passes\": [\n"
                 << "  {\n"
                 << "  \"pass_name\": \"conv_pass\",\n"
                 << "  \"pass_type\": \"Transformation\",\n"
                 << "  \"pass_backend\": \"dnnl\",\n"
                 << "  \"priority\": 8,\n"
                 << "  \"enable\": 1\n"
                 << "  },\n"
                 << "  {\n"
                 << "  \"pass_name\": \"relu_pass\",\n"
                 << "  \"pass_type\": \"Transformation\",\n"
                 << "  \"pass_backend\": \"dnnl\",\n"
                 << "  \"priority\": 8,\n"
                 << "  \"enable\": 1\n"
                 << "  }\n"
                 << "]\n"
                 << "}\n";
    std::string valid_str = valid_stream.str();
    std::istringstream valid_is(valid_str);
    pm.run_passes(agraph, &valid_is);
    ASSERT_EQ(agraph.get_num_partitions(), 2);
}

TEST(pass_test, invalid_json_for_incompatible_hash) {
    /*   \   /
          conv
           |
          relu
    */
    graph_t agraph;
    op_t conv1 {0, Convolution, "conv"};
    set_conv_common_attr(conv1);
    op_t relu {1, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    conv1.add_input(lt_vec[0]);
    conv1.add_input(lt_vec[1]);
    conv1.add_output(lt_vec[2]);
    relu.add_input(lt_vec[2]);
    relu.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    auto &backend_ptr
            = dnnl::graph::impl::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::graph::impl::pass::pass_manager(
            backend_ptr.get_pass_registry());

    std::ostringstream invalid_stream;
    std::string version = std::to_string(dnnl_graph_version()->major) + "."
            + std::to_string(dnnl_graph_version()->minor) + "."
            + std::to_string(dnnl_graph_version()->patch);
    invalid_stream << "{\n"
                   << "\"version\": \"" << version << "\",\n"
                   << "\"hash\": \""
                   << "aninvalidcommitid"
                   << "\",\n"
                   << "\"passes\": [\n"
                   << "  {\n"
                   << "  \"pass_name\": \"conv_pass\",\n"
                   << "  \"pass_type\": \"Transformation\",\n"
                   << "  \"pass_backend\": \"dnnl\",\n"
                   << "  \"priority\": 8,\n"
                   << "  \"enable\": 1\n"
                   << "  },\n"
                   << "  {\n"
                   << "  \"pass_name\": \"relu_pass\",\n"
                   << "  \"pass_type\": \"Transformation\",\n"
                   << "  \"pass_backend\": \"dnnl\",\n"
                   << "  \"priority\": 8,\n"
                   << "  \"enable\": 1\n"
                   << "  }\n"
                   << "]\n"
                   << "}\n";
    std::string invalid_str = invalid_stream.str();
    std::istringstream invalid_is(invalid_str);
    pm.run_passes(agraph, &invalid_is);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
}

TEST(pass_test, invalid_json_for_missing_field) {
    /*   \   /
          conv
           |
          relu
    */
    graph_t agraph;
    op_t conv1 {0, Convolution, "conv"};
    set_conv_common_attr(conv1);
    op_t relu {1, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    conv1.add_input(lt_vec[0]);
    conv1.add_input(lt_vec[1]);
    conv1.add_output(lt_vec[2]);
    relu.add_input(lt_vec[2]);
    relu.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    auto &backend_ptr
            = dnnl::graph::impl::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::graph::impl::pass::pass_manager(
            backend_ptr.get_pass_registry());

    std::ostringstream invalid_stream;
    invalid_stream << "{\n"
                   << "\"passes\": [\n"
                   << "  {\n"
                   << "  \"pass_name\": \"conv_pass\",\n"
                   << "  \"pass_type\": \"Transformation\",\n"
                   << "  \"pass_backend\": \"dnnl\",\n"
                   << "  \"priority\": 8,\n"
                   << "  \"enable\": 1\n"
                   << "  },\n"
                   << "  {\n"
                   << "  \"pass_name\": \"relu_pass\",\n"
                   << "  \"pass_type\": \"Transformation\",\n"
                   << "  \"pass_backend\": \"dnnl\",\n"
                   << "  \"priority\": 8,\n"
                   << "  \"enable\": 1\n"
                   << "  }\n"
                   << "]\n"
                   << "}\n";
    std::string invalid_str = invalid_stream.str();
    std::istringstream invalid_is(invalid_str);
    pm.run_passes(agraph, &invalid_is);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
}

TEST(pass_test, invalid_json_for_wrong_format) {
    /*   \   /
          conv
           |
          relu
    */
    graph_t agraph;
    op_t conv1 {0, Convolution, "conv"};
    set_conv_common_attr(conv1);
    op_t relu {1, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    conv1.add_input(lt_vec[0]);
    conv1.add_input(lt_vec[1]);
    conv1.add_output(lt_vec[2]);
    relu.add_input(lt_vec[2]);
    relu.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&conv1), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 2);

    auto &backend_ptr
            = dnnl::graph::impl::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::graph::impl::pass::pass_manager(
            backend_ptr.get_pass_registry());

    std::ostringstream invalid_stream;
    std::string version = std::to_string(dnnl_graph_version()->major) + "."
            + std::to_string(dnnl_graph_version()->minor) + "."
            + std::to_string(dnnl_graph_version()->patch);
    invalid_stream << "\"version\": \"" << version << "\",\n"
                   << "\"hash\": \"" << dnnl_graph_version()->hash << "\",\n"
                   << "\"passes\": [\n"
                   << "  {\n"
                   << "  \"pass_name\": \"conv_pass\",\n"
                   << "  \"pass_type\": \"Transformation\",\n"
                   << "  \"pass_backend\": \"dnnl\",\n"
                   << "  \"priority\": 8,\n"
                   << "  \"enable\": 1\n"
                   << "  },\n";
    std::string invalid_str = invalid_stream.str();
    std::istringstream invalid_is(invalid_str);
    pm.run_passes(agraph, &invalid_is);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
}

TEST(pass_test, two_conv_with_shared_weight) {
    /*    \   /\    /
          conv  conv
            |     |
           relu relu

    */
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

    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::conv_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);

    ASSERT_EQ(get_fused_op(agraph.get_partitions()[1])->get_kind(),
            dnnl_impl::op_kind::conv_relu);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[0].id, 3);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[1].id, 1);

    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs()[0].id, 5);
}

TEST(pass_test, multi_values_between_two_ops) {
    /*     conv
            ||
           add
    */
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

    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(
            get_fused_op(agraph.get_partitions()[0])->get_kind(), Convolution);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2);
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::int8_conv);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_priority_test, int8_conv_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
           quant
             | (u8/s8)
    */
    pass::pass_base_ptr pass1 = get_pass("int8_conv_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_conv_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_pass");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
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
            dnnl_impl::op_kind::int8_conv_bias);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6);
}

TEST(pass_priority_test, int8_conv_bias_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
           quant
             | (u8/s8)
    */
    pass::pass_base_ptr pass1 = get_pass("int8_conv_bias_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_conv_bias_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_bias_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
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
            dnnl_impl::op_kind::int8_conv_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6);
}

TEST(pass_priority_test, int8_conv_relu_fusion) {
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
    pass::pass_base_ptr pass1 = get_pass("int8_conv_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_conv_relu_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_relu_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
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
            dnnl_impl::op_kind::int8_conv_bias_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7);
}

TEST(pass_priority_test, int8_conv_bias_relu_fusion) {
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
    pass::pass_base_ptr pass1 = get_pass("int8_conv_bias_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_conv_bias_relu_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_bias_relu_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
}

TEST(pass_test, int8_conv_bias_add_fusion) {
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
    op_t quant {6, Quantize, "quant"};
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

    logical_tensor_t int8_out = logical_tensor_init(9, data_type::u8);
    quant.add_input(fp32_add_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_conv_bias_add_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::int8_conv_bias_add);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9);
}

TEST(pass_priority_test, int8_conv_bias_add_fusion) {
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
           quant
             | (u8/s8)
    */
    pass::pass_base_ptr pass1 = get_pass("int8_conv_bias_add_fusion");
    pass::pass_base_ptr pass2 = get_pass("conv_bias_sum_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_bias_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
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
            dnnl_impl::op_kind::int8_conv_bias_add_relu);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10);
}

TEST(pass_priority_test, int8_conv_bias_add_relu_fusion) {
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
    pass::pass_base_ptr pass1 = get_pass("int8_conv_bias_add_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_conv_bias_add_relu_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_bias_sum_relu_fusion");
    pass::pass_base_ptr pass4 = get_pass("conv_bias_sum_fusion");
    pass::pass_base_ptr pass5 = get_pass("conv_bias_fusion");
    pass::pass_base_ptr pass6 = get_pass("quant_pass");
    pass::pass_base_ptr pass7 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass3->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass4->get_priority() > pass5->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass6->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass7->get_priority());
}

TEST(pass_priority_test, int8_conv_add_relu_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
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
    pass::pass_base_ptr pass1 = get_pass("int8_conv_add_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_conv_add_relu_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_sum_relu_fusion");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
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
            dnnl_impl::op_kind::int8_conv_bias_add_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10);
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::x8s8f32_conv);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
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
            dnnl_impl::op_kind::x8s8f32_conv_bias);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
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
            impl::dnnl_impl::op_kind::x8s8f32_conv_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
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
            impl::dnnl_impl::op_kind::x8s8f32_conv_bias_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6);
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
            impl::dnnl_impl::op_kind::x8s8f32_conv_bias_add_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9);
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
            impl::dnnl_impl::op_kind::x8s8f32_conv_bias_add_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9);
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
            dnnl_impl::op_kind::x8s8f32_conv_bias_add_relu);

    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[2].id, 6);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[3].id, 4);

    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs()[0].id, 9);

    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::int8_conv);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 10);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 12);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            impl::dnnl_impl::op_kind::int8_matmul);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
}

TEST(pass_priority_test, int8_matmul_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
           matmul
             | (f32)
           quant
             | (u8/s8)
    */
    pass::pass_base_ptr pass1 = get_pass("int8_matmul_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8x8f32_matmul_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_pass");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
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
            impl::dnnl_impl::op_kind::int8_matmul_bias);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6);
}

TEST(pass_priority_test, int8_matmul_bias_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)   / (f32)
           matmul (w/ bias)
             | (f32)
           quant
             | (u8/s8)
    */
    pass::pass_base_ptr pass1 = get_pass("int8_matmul_bias_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_matmul_bias_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_bias_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
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
            dnnl_impl::op_kind::int8_matmul_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6);
}

TEST(pass_priority_test, int8_matmul_relu_fusion) {
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
    pass::pass_base_ptr pass1 = get_pass("int8_matmul_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_matmul_relu_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_relu_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
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
            dnnl_impl::op_kind::int8_matmul_bias_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7);
}

TEST(pass_priority_test, int8_matmul_bias_relu_fusion) {
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
    pass::pass_base_ptr pass1 = get_pass("int8_matmul_bias_relu_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_matmul_bias_relu_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_bias_relu_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
}

TEST(pass_test, x8x8f32_matmul_fusion) {
    /*
        | (u8/s8)  | (u8/s8)
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

    pass::pass_base_ptr apass = get_pass("x8x8f32_matmul_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::x8x8f32_matmul);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4);
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

    pass::pass_base_ptr apass = get_pass("x8x8f32_matmul_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);

    pass::pass_base_ptr apass2 = get_pass("x8s8f32_matmul_bias_fusion");
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::x8s8float_matmul_bias);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
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
            {ReLU, dnnl_impl::op_kind::x8s8f32_matmul_relu},
            {Sigmoid, dnnl_impl::op_kind::x8s8f32_matmul_sigmoid},
            {GELU, dnnl_impl::op_kind::x8s8f32_matmul_gelu}};
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

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5);
    }
}

TEST(pass_priority_test, int8_matmul_sigmoid_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
           matmul
             | (f32)
           sigmoid
             | (f32)
           quant
             | (u8/s8)
    */
    pass::pass_base_ptr pass1 = get_pass("int8_matmul_sigmoid_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_matmul_sigmoid_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_sigmoid_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
}

TEST(pass_priority_test, int8_matmul_gelu_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
           matmul
             | (f32)
           gelu
             | (f32)
           quant
             | (u8/s8)
    */
    pass::pass_base_ptr pass1 = get_pass("int8_matmul_gelu_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_matmul_gelu_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_gelu_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
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
            {ReLU, dnnl_impl::op_kind::x8s8f32_matmul_bias_relu},
            {Sigmoid, dnnl_impl::op_kind::x8s8f32_matmul_bias_sigmoid},
            {GELU, dnnl_impl::op_kind::x8s8f32_matmul_bias_gelu}};
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

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6);
    }
}

TEST(pass_priority_test, int8_matmul_bias_sigmoid_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
           matmul
             | (f32)
           sigmoid
             | (f32)
           quant
             | (u8/s8)
    */
    pass::pass_base_ptr pass1 = get_pass("int8_matmul_bias_sigmoid_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_matmul_bias_sigmoid_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_bias_sigmoid_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
}

TEST(pass_priority_test, int8_matmul_bias_gelu_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
           matmul
             | (f32)
           gelu
             | (f32)
           quant
             | (u8/s8)
    */
    pass::pass_base_ptr pass1 = get_pass("int8_matmul_bias_gelu_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_matmul_bias_gelu_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_bias_gelu_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::int8_maxpool);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
}

TEST(pass_priority_test, int8_maxpool_fusion) {
    /*
             | (u8/s8)
          dequant
             | (f32)
           maxpool
             | (f32)
           quant
             | (u8/s8)
    */
    pass::pass_base_ptr pass1 = get_pass("int8_maxpool_fusion");
    pass::pass_base_ptr pass2 = get_pass("max_pool_pass");
    pass::pass_base_ptr pass3 = get_pass("quant_pass");
    pass::pass_base_ptr pass4 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass4->get_priority());
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::int8_avgpool);
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
            dnnl_impl::op_kind::int8_matmul_bias_add);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9);
}

TEST(pass_priority_test, int8_matmul_bias_add_fusion) {
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
    pass::pass_base_ptr pass1 = get_pass("int8_matmul_bias_add_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_matmul_bias_add_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_bias_sum_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
}

TEST(pass_priority_test, int8_matmul_add_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            matmul
             | (f32)
             |     | (s8)
             |   dequant
             |  / (f32)
            add
             | (f32)
           quant
             | (u8/s8)
    */
    pass::pass_base_ptr pass1 = get_pass("int8_matmul_add_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_matmul_add_fusion");
    pass::pass_base_ptr pass3 = get_pass("matmul_sum_fusion");
    pass::pass_base_ptr pass4 = get_pass("quant_pass");
    pass::pass_base_ptr pass5 = get_pass("dequant_pass");
    ASSERT_TRUE(pass1->get_priority() > pass2->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass3->get_priority());
    ASSERT_TRUE(pass2->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
}

TEST(pass_test, relu_add_fusion) {
    /*
         relu
           \  /
           add
    */
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
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::relu_add);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3);
}

TEST(pass_priority_test, relu_add_fusion) {
    /*
         relu
           \  /
           add
    */
    pass::pass_base_ptr pass1 = get_pass("relu_add_fusion");
    pass::pass_base_ptr pass2 = get_pass("matmul_sum_fusion");
    pass::pass_base_ptr pass3 = get_pass("conv_sum_fusion");
    pass::pass_base_ptr pass4 = get_pass("relu_pass");
    pass::pass_base_ptr pass5 = get_pass("sum_pass");
    ASSERT_TRUE(pass1->get_priority() < pass2->get_priority());
    ASSERT_TRUE(pass1->get_priority() < pass3->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass4->get_priority());
    ASSERT_TRUE(pass1->get_priority() > pass5->get_priority());
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
            dnnl_impl::op_kind::x8s8float_matmul_bias_add);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8);
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

TEST(pass_test, x8x8bf16_matmul_div_fusion) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
           matmul
             | (bf16)
            div
             | (bf16)
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
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t div {5, Divide, "divide"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t bf16_data = logical_tensor_init(2, data_type::bf16);
    typecast1.add_input(fp32_data);
    typecast1.add_output(bf16_data);

    logical_tensor_t int8_weight = logical_tensor_init(3, data_type::u8);
    logical_tensor_t fp32_weight = logical_tensor_init(4, data_type::f32);
    dequant2.add_input(int8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t bf16_weight = logical_tensor_init(5, data_type::bf16);
    typecast2.add_input(fp32_weight);
    typecast2.add_output(bf16_weight);

    logical_tensor_t bf16_matmul_out = logical_tensor_init(6, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t bf16_div_in = logical_tensor_init(7, data_type::bf16);
    logical_tensor_t bf16_div_out = logical_tensor_init(8, data_type::bf16);
    div.add_input(bf16_matmul_out);
    div.add_input(bf16_div_in);
    div.add_output(bf16_div_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8x8bf16_matmul_div_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::x8x8float_matmul_div);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 7);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8);
}

TEST(pass_test, x8x8bf16_matmul_div_fusion_fail) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (f16)
           matmul
             | (bf16)
            div
             | (bf16)
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
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t div {5, Divide, "divide"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t bf16_data = logical_tensor_init(2, data_type::bf16);
    typecast1.add_input(fp32_data);
    typecast1.add_output(bf16_data);

    logical_tensor_t int8_weight = logical_tensor_init(3, data_type::u8);
    logical_tensor_t fp32_weight = logical_tensor_init(4, data_type::f32);
    dequant2.add_input(int8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t f16_weight = logical_tensor_init(5, data_type::f16);
    typecast2.add_input(fp32_weight);
    typecast2.add_output(f16_weight);

    logical_tensor_t bf16_matmul_out = logical_tensor_init(6, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(f16_weight);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t bf16_div_in = logical_tensor_init(7, data_type::bf16);
    logical_tensor_t bf16_div_out = logical_tensor_init(8, data_type::bf16);
    div.add_input(bf16_matmul_out);
    div.add_input(bf16_div_in);
    div.add_output(bf16_div_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8x8bf16_matmul_div_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);
}

TEST(pass_test, x8s8bf16_matmul_bias_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16) / (f32/bf16)
           matmul_with_bias
             | (bf16)
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
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t bf16_data = logical_tensor_init(2, data_type::bf16);
    typecast1.add_input(fp32_data);
    typecast1.add_output(bf16_data);

    logical_tensor_t int8_weight = logical_tensor_init(3, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(4, data_type::f32);
    dequant2.add_input(int8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t bf16_weight = logical_tensor_init(5, data_type::bf16);
    typecast2.add_input(fp32_weight);
    typecast2.add_output(bf16_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(6, data_type::f32);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(7, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(bf16_matmul_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8bf16_matmul_bias_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::x8s8float_matmul_bias);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7);
}

TEST(pass_test, single_typecast_pass) {
    /*
        | (f32)
     typecast
        | (bf16) 
    */
    graph_t agraph;
    op_t typecast {0, TypeCast, "typecast"};

    logical_tensor_t f32_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t bf16_out = logical_tensor_init(1, data_type::bf16);
    typecast.add_input(f32_data);
    typecast.add_output(bf16_out);

    ASSERT_EQ(agraph.add_op(&typecast), status::success);

    ASSERT_EQ(agraph.build_graph(), status::success);

    pass::pass_base_ptr apass = get_pass("typecast_pass");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
}

TEST(pass_test, single_typecast_fail) {
    /*
        | (f16)
     typecast
        | (bf16) 
    */
    graph_t agraph;
    op_t typecast {0, TypeCast, "typecast"};

    logical_tensor_t f16_data = logical_tensor_init(0, data_type::f16);
    logical_tensor_t bf16_out = logical_tensor_init(1, data_type::bf16);
    typecast.add_input(f16_data);
    typecast.add_output(bf16_out);

    ASSERT_EQ(agraph.add_op(&typecast), status::success);

    ASSERT_EQ(agraph.build_graph(), status::success);

    pass::pass_base_ptr apass = get_pass("typecast_pass");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0);
}

TEST(pass_test, x8s8bf16_matmul_bias_add_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16) / (f32/bf16)
            matmul_with_bias
             | (bf16)
             |     | (s8)
             |   dequant
             |     | (f32)
             |   typecast
             |  / (bf16)
            add
             | (bf16)
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
    op_t typecast1 {3, TypeCast, "typecast"};
    op_t typecast2 {4, TypeCast, "typecast"};
    op_t matmul {5, MatMul, "matmul"};
    matmul.set_attr<bool>("transpose_a", false);
    matmul.set_attr<bool>("transpose_b", false);
    op_t typecast3 {6, TypeCast, "typecast"};
    op_t add {7, Add, "add"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t bf16_data = logical_tensor_init(2, data_type::bf16);
    typecast1.add_input(fp32_data);
    typecast1.add_output(bf16_data);

    logical_tensor_t int8_weight = logical_tensor_init(3, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(4, data_type::f32);
    dequant2.add_input(int8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t bf16_weight = logical_tensor_init(5, data_type::bf16);
    typecast2.add_input(fp32_weight);
    typecast2.add_output(bf16_weight);

    logical_tensor_t int8_other = logical_tensor_init(6, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(7, data_type::f32);
    dequant3.add_input(int8_other);
    dequant3.add_output(fp32_other);

    logical_tensor_t bf16_other = logical_tensor_init(8, data_type::bf16);
    typecast3.add_input(fp32_other);
    typecast3.add_output(bf16_other);

    logical_tensor_t fp32_bias = logical_tensor_init(9, data_type::f32);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(10, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t bf16_add_out = logical_tensor_init(11, data_type::bf16);
    add.add_input(bf16_matmul_out);
    add.add_input(bf16_other);
    add.add_output(bf16_add_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8bf16_matmul_bias_add_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::x8s8float_matmul_bias_add);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 9);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 6);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 11);
}

TEST(pass_test, x8s8bf16_matmul_add_fusion) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
            matmul
             | (bf16)
             |     | (s8)
             |   dequant
             |     | (f32)
             |   typecast
             |  / (bf16)
            add
             | (bf16)
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
    op_t typecast1 {3, TypeCast, "typecast"};
    op_t typecast2 {4, TypeCast, "typecast"};
    op_t matmul {5, MatMul, "matmul"};
    matmul.set_attr<bool>("transpose_a", false);
    matmul.set_attr<bool>("transpose_b", false);
    op_t typecast3 {6, TypeCast, "typecast"};
    op_t add {7, Add, "add"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t bf16_data = logical_tensor_init(2, data_type::bf16);
    typecast1.add_input(fp32_data);
    typecast1.add_output(bf16_data);

    logical_tensor_t int8_weight = logical_tensor_init(3, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(4, data_type::f32);
    dequant2.add_input(int8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t bf16_weight = logical_tensor_init(5, data_type::bf16);
    typecast2.add_input(fp32_weight);
    typecast2.add_output(bf16_weight);

    logical_tensor_t int8_other = logical_tensor_init(6, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(7, data_type::f32);
    dequant3.add_input(int8_other);
    dequant3.add_output(fp32_other);

    logical_tensor_t bf16_other = logical_tensor_init(8, data_type::bf16);
    typecast3.add_input(fp32_other);
    typecast3.add_output(bf16_other);

    logical_tensor_t bf16_matmul_out = logical_tensor_init(10, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t bf16_add_out = logical_tensor_init(11, data_type::bf16);
    add.add_input(bf16_matmul_out);
    add.add_input(bf16_other);
    add.add_output(bf16_add_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("x8s8bf16_matmul_add_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);

    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            dnnl_impl::op_kind::x8s8float_matmul_add);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 11);
}

TEST(pass_test, int8_mix_bf16_matmul_bias_gelu_fusion) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
        matmul_with_bias
             | (bf16)
            gelu
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t eltwise {5, GELU, "gelu"};
    op_t typecast3 {6, TypeCast, "typecast"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t bf16_data = logical_tensor_init(2, data_type::bf16);
    typecast1.add_input(fp32_data);
    typecast1.add_output(bf16_data);

    logical_tensor_t int8_weight = logical_tensor_init(3, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(4, data_type::f32);
    dequant2.add_input(int8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t bf16_weight = logical_tensor_init(5, data_type::bf16);
    typecast2.add_input(fp32_weight);
    typecast2.add_output(bf16_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(6, data_type::f32);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(7, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t bf16_eltwise_out = logical_tensor_init(8, data_type::bf16);
    eltwise.add_input(bf16_matmul_out);
    eltwise.add_output(bf16_eltwise_out);

    logical_tensor_t fp32_eltwise_out = logical_tensor_init(9, data_type::f32);
    typecast3.add_input(bf16_eltwise_out);
    typecast3.add_output(fp32_eltwise_out);

    logical_tensor_t int8_dst = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_eltwise_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&eltwise), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_matmul_bias_gelu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10);
}

TEST(pass_test, int8_mix_bf16_matmul_gelu_fusion) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
           matmul
             | (bf16)
            gelu
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t eltwise {5, GELU, "gelu"};
    op_t typecast3 {6, TypeCast, "typecast"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t bf16_data = logical_tensor_init(2, data_type::bf16);
    typecast1.add_input(fp32_data);
    typecast1.add_output(bf16_data);

    logical_tensor_t int8_weight = logical_tensor_init(3, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(4, data_type::f32);
    dequant2.add_input(int8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t bf16_weight = logical_tensor_init(5, data_type::bf16);
    typecast2.add_input(fp32_weight);
    typecast2.add_output(bf16_weight);

    logical_tensor_t bf16_matmul_out = logical_tensor_init(7, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t bf16_eltwise_out = logical_tensor_init(8, data_type::bf16);
    eltwise.add_input(bf16_matmul_out);
    eltwise.add_output(bf16_eltwise_out);

    logical_tensor_t fp32_eltwise_out = logical_tensor_init(9, data_type::f32);
    typecast3.add_input(bf16_eltwise_out);
    typecast3.add_output(fp32_eltwise_out);

    logical_tensor_t int8_dst = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_eltwise_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&eltwise), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_matmul_gelu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10);
}

// deq->matmul->gelu->q should have higher priority than deq->matmul->gelu
// deq->typecast->matmul->gelu->typecast->q should have higher priority than
// deq->typecast->matmul
TEST(pass_priority_test, int8_bf16_matmul_gelu) {
    pass::pass_base_ptr pass1 = get_pass("int8_matmul_bias_gelu_fusion");
    pass::pass_base_ptr pass2 = get_pass("x8s8f32_matmul_bias_gelu_fusion");
    pass::pass_base_ptr pass3 = get_pass("x8s8bf16_matmul_bias_fusion");
    ASSERT_GT(pass1->get_priority(), pass2->get_priority());
    ASSERT_GT(pass1->get_priority(), pass3->get_priority());
}
