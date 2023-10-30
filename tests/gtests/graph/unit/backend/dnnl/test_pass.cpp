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

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "utils/pm/pass_base.hpp"
#include "utils/pm/pass_manager.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"

#include "graph/unit/utils.hpp"

using namespace dnnl::impl::graph;
using namespace dnnl::impl::graph::op_kind;
using namespace dnnl::graph::tests::unit::utils;

using op_ptr = std::shared_ptr<dnnl::impl::graph::op_t>;

#define for_ for

namespace {
dnnl::impl::graph::pass::pass_base_ptr get_pass(const std::string &pass_name) {
    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());
    auto &passes = pm.get_passes();
    auto find = std::find_if(passes.begin(), passes.end(),
            [&pass_name](const dnnl::impl::graph::pass::pass_base_ptr &p)
                    -> bool { return p->get_pass_name() == pass_name; });

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
TEST(Pass, FuseConvBn) {
    /*   conv
          |
         bn
    */
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr(op_attr::epsilon, 0.001f);

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

    agraph.finalize();

    ASSERT_EQ(agraph.num_ops(), 2U);
    ASSERT_EQ(agraph.get_ops()[0]->get_kind(), Convolution);
    ASSERT_EQ(agraph.get_ops()[0]->num_inputs(), 2U);
    ASSERT_EQ(agraph.get_ops()[0]->num_outputs(), 1U);
    ASSERT_EQ(agraph.get_ops()[1]->get_kind(), BatchNormInference);
    ASSERT_EQ(agraph.get_ops()[1]->num_inputs(), 5U);
    ASSERT_EQ(agraph.get_ops()[1]->num_outputs(), 1U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(Pass, FuseConvBnWithSharedInputs) {
    /*   conv
          |
         bn
    */
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr(op_attr::epsilon, 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bn.add_input(lt_vec[2]);
    //assume gamma/beta/mean/var are using the same lt
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[3]);
    bn.add_input(lt_vec[3]);
    bn.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bn), status::success);

    agraph.finalize();

    ASSERT_EQ(agraph.num_ops(), 2U);
    ASSERT_EQ(agraph.get_ops()[0]->get_kind(), Convolution);
    ASSERT_EQ(agraph.get_ops()[0]->num_inputs(), 2U);
    ASSERT_EQ(agraph.get_ops()[0]->num_outputs(), 1U);
    ASSERT_EQ(agraph.get_ops()[1]->get_kind(), BatchNormInference);
    ASSERT_EQ(agraph.get_ops()[1]->num_inputs(), 5U);
    ASSERT_EQ(agraph.get_ops()[1]->num_outputs(), 1U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    // For a partition with N inputs that have the same id
    // It is required that those inputs are input N times
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FailToFuseConvBnWithConvSecondOutput) {
    /*   conv
        /    \
       bn   relu
    */
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bn {1, BatchNormInference, "bn"};
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2U);
}

TEST(Pass, FuseConvRelu) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseConvBiasadd) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseConvWithInputBias) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    // bias op can't be fused since the post conv already has bias input.
    // so only three inputs
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseConvSum) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseConvBiasaddBn) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);

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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 8U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9U);
}

TEST(Pass, FuseConvBiasBnWithInputBias) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);

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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 7U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8U);
}

TEST(Pass, FuseConvBiasaddRelu) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvBiasReluWithInputBias) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseConvBiasaddRelu6) {
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
    op_t clamp {2, Clamp, "clamp"};
    clamp.set_attr(op_attr::min, 0.f);
    clamp.set_attr(op_attr::max, 6.f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    clamp.add_input(lt_vec[4]);
    clamp.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&clamp), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvBiasElu) {
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
    elu.set_attr(op_attr::alpha, 0.1f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]); // conv with bias
    conv.add_output(lt_vec[3]);
    elu.add_input(lt_vec[3]);
    elu.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&elu), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseConvBiasSigmoid) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseConvBiasSwish) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvSwish) {
    // swish: f(x) = x * sigmoid(x)
    /*   conv
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

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    sigmoid.add_input(lt_vec[2]);
    sigmoid.add_output(lt_vec[3]);
    multiply.add_input(lt_vec[2]);
    multiply.add_input(lt_vec[3]);
    multiply.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseConvSwishSigmoid) {
    // swish: f(x) = x * sigmoid(x)
    /*   conv
        /    |
    sigmoid  |
        \    |
        multiply
           |
        sigmoid

    */
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t sigmoid {1, Sigmoid, "sigmoid"};
    op_t multiply {2, Multiply, "multiply"};
    op_t sigmoid2 {3, Sigmoid, "sigmoid"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    sigmoid.add_input(lt_vec[2]);
    sigmoid.add_output(lt_vec[3]);
    multiply.add_input(lt_vec[2]);
    multiply.add_input(lt_vec[3]);
    multiply.add_output(lt_vec[4]);
    sigmoid2.add_input(lt_vec[4]);
    sigmoid2.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid2), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvBiasClamp) {
    /*   conv
          |
         bias
          |
         clamp
    */
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t clamp {2, Clamp, "clamp"};
    clamp.set_attr(op_attr::min, 0.f);
    clamp.set_attr(op_attr::max, 100.f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    clamp.add_input(lt_vec[4]);
    clamp.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&clamp), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvBiasSquare) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvBiasTanh) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvBiasAbs) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvBiasSqrt) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvBiasaddSum) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FuseConvBiasSum) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvBiasaddSumRelu) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(PassSystem, TestConvRelated) {
    /*   conv
          |
         bias conv
           \   /
            add
             |
            relu    should be fused to conv-bias-add-relu + conv
    */
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t bias {1, BiasAdd, "bias"};
    op_t add {2, Add, "add"};
    op_t relu {3, ReLU, "relu"};
    op_t conv2 {4, Convolution, "conv2"};
    set_conv_common_attr(conv2);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(10);
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
    conv2.add_input(lt_vec[8]);
    conv2.add_input(lt_vec[9]);
    conv2.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&conv2), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 5U);

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 2U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);
    ASSERT_EQ((agraph.get_partitions()[1])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);

    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[0].id, 8U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[1].id, 9U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);

    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvBiasaddSumElu) {
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
    elu.set_attr(op_attr::alpha, 0.1f);

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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(Pass, FuseConvBiasaddSumRelu6) {
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
    op_t clamp {3, Clamp, "clamp"};
    clamp.set_attr(op_attr::min, 0.f);
    clamp.set_attr(op_attr::max, 6.f);

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
    clamp.add_input(lt_vec[6]);
    clamp.add_output(lt_vec[7]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&clamp), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(PassSystem, FuseConvDepthwise) {
    /*   conv
          |
         conv (depthwise)
    */
    const std::vector<engine_kind_t> engine_kinds
            = {engine_kind::cpu, engine_kind::gpu};
    const std::vector<std::string> dw_types {"k3s1p1", "k3s2p1"};
    // N, IC, IH, IW
    const std::vector<int64_t> conv_src_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    const std::vector<int64_t> conv_wei_shape {4, 4, 1, 1};
    // N, OC, OH, OW
    const std::vector<int64_t> conv_dst_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    const std::vector<int64_t> dw_wei_shape {4, 1, 3, 3};
    // N, OC, OH, OW
    const std::vector<int64_t> dw_dst_shape {4, 4, 4, 4};

    const auto apply_str_for_ncx = [](const std::vector<int64_t> &shape,
                                           const std::string &dw_type) {
        std::vector<int64_t> new_shape = shape;
        const int64_t str_val = (dw_type == "k3s1p1") ? 1 : 2;
        for (size_t i = 0; i < new_shape.size() - 2; ++i) {
            new_shape[2 + i] /= str_val;
        }
        return new_shape;
    };

    for_(const auto &engine_kind : engine_kinds)
    for (const auto &dw_type : dw_types) {
        op_t conv {0, Convolution, "conv"};
        set_conv_dw_base_op_attr(conv);

        op_t depthwise {1, Convolution, "depthwise"};
        set_conv_dw_post_op_attr(depthwise, dw_type);

        logical_tensor_t conv_src
                = logical_tensor_init(0, conv_src_shape, data_type::f32);
        logical_tensor_t conv_wei
                = logical_tensor_init(1, conv_wei_shape, data_type::f32);
        logical_tensor_t conv_dst
                = logical_tensor_init(2, conv_dst_shape, data_type::f32);

        logical_tensor_t dw_wei
                = logical_tensor_init(3, dw_wei_shape, data_type::f32);
        logical_tensor_t dw_dst = logical_tensor_init(
                4, apply_str_for_ncx(dw_dst_shape, dw_type), data_type::f32);

        conv.add_input(conv_src);
        conv.add_input(conv_wei);
        conv.add_output(conv_dst);

        depthwise.add_input(conv_dst);
        depthwise.add_input(dw_wei);
        depthwise.add_output(dw_dst);

        graph_t agraph(engine_kind);
        ASSERT_EQ(agraph.add_op(&conv), status::success);
        ASSERT_EQ(agraph.add_op(&depthwise), status::success);
        agraph.finalize();

        auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
        auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
        pm.run_passes(agraph, "no_config");

        if (engine_kind == engine_kind::cpu) {
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::convolution_post_ops);
        } else {
            ASSERT_EQ(agraph.get_num_partitions(), 2U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::convolution_post_ops);
            ASSERT_EQ((agraph.get_partitions()[1])->get_kind(),
                    partition_kind_t::convolution_post_ops);
        }
    }
}

TEST(Pass, FuseBinarySum) {
    /* binary here represents Multiply, Minimum, Maximum

        binary
           \   /
            add
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, partition_kind_t>> opkind_pair {
            {Multiply, partition_kind_t::binary_post_ops},
            {Maximum, partition_kind_t::binary_post_ops},
            {Minimum, partition_kind_t::binary_post_ops}};

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
        agraph.finalize();

        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(), p.second);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
    }
}

TEST(PassSystem, TestConvSumAndBinary) {
    /* binary here represents Multiply, Minimum, Maximum

        binary conv
           \   /
            add        should be fused to conv-add + binary
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, partition_kind_t>> opkind_pair {
            {Multiply, partition_kind_t::convolution_post_ops},
            {Maximum, partition_kind_t::convolution_post_ops},
            {Minimum, partition_kind_t::convolution_post_ops}};

    for (const auto &p : opkind_pair) {
        graph_t agraph;
        auto binary_kind = p.first;
        op_t binary {0, binary_kind, "binary"};
        op_t conv {1, Convolution, "convolution"};
        op_t add {2, Add, "add"};

        auto lt_vec = create_logical_tensors(7);

        binary.add_input(lt_vec[0]);
        binary.add_input(lt_vec[1]);
        binary.add_output(lt_vec[2]);
        conv.add_input(lt_vec[3]);
        conv.add_input(lt_vec[4]);
        conv.add_output(lt_vec[5]);
        set_conv_common_attr(conv);
        add.add_input(lt_vec[2]);
        add.add_input(lt_vec[5]);
        add.add_output(lt_vec[6]);

        ASSERT_EQ(agraph.add_op(&binary), status::success);
        ASSERT_EQ(agraph.add_op(&conv), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        agraph.finalize();

        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 2U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(), p.second);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 3U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 4U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
    }
}

TEST(Pass, FuseBinarySumWithSupportBroadcast) {
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, partition_kind_t>> opkind_pair {
            {Multiply, partition_kind_t::binary_post_ops},
            {Maximum, partition_kind_t::binary_post_ops},
            {Minimum, partition_kind_t::binary_post_ops}};

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
        agraph.finalize();

        pm.run_passes(agraph, "no_config");

        // should not be fused
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
    }
}

TEST(Pass, FailToFuseBinarySumWithUnsupportBroadcast) {
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, partition_kind_t>> opkind_pair {
            {Multiply, partition_kind_t::binary_post_ops},
            {Maximum, partition_kind_t::binary_post_ops},
            {Minimum, partition_kind_t::binary_post_ops}};

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
        agraph.finalize();

        pm.run_passes(agraph, "no_config");

        // could fuse now
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
    }
}

TEST(Pass, FailToFuseBinarySumWithUnknownShape) {
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, partition_kind_t>> opkind_pair {
            {Multiply, partition_kind_t::binary_post_ops},
            {Maximum, partition_kind_t::binary_post_ops},
            {Minimum, partition_kind_t::binary_post_ops}};

    for (auto &p : opkind_pair) {
        graph_t agraph;
        auto binary_kind = p.first;
        op_t binary {0, binary_kind, "binary"};
        op_t add {1, Add, "add"};

        std::vector<logical_tensor_t> lt_vec;
        lt_vec.reserve(5);
        // valid ndims, invalid shape
        for (size_t i = 0; i < 5; i++)
            lt_vec.emplace_back(logical_tensor_init(i, data_type::f32));

        binary.add_input(lt_vec[0]);
        binary.add_input(lt_vec[1]);
        binary.add_output(lt_vec[2]);
        add.add_input(lt_vec[2]);
        add.add_input(lt_vec[3]);
        add.add_output(lt_vec[4]);

        ASSERT_EQ(agraph.add_op(&binary), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        agraph.finalize();

        pm.run_passes(agraph, "no_config");

        // could fuse now
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
    }
}

TEST(Pass, FuseBinaryAddMul) {
    /*
         \  /
          add
           \   /
            mul
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());

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
    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::binary_post_ops);
}

TEST(Pass, FuseBinaryEltwise) {
    /* binary here represents Add, Multiply, Minimum, Maximum
       eltwise here represents Sigmoid, ReLU

         \  /
        binary
           |
        eltwise
    */
    std::vector<std::pair<std::pair<op_kind_t, op_kind_t>, partition_kind_t>>
            opkind_pair {{{Add, Sigmoid}, partition_kind_t::binary_post_ops},
                    {{Add, ReLU}, partition_kind_t::binary_post_ops},
                    {{Multiply, Sigmoid}, partition_kind_t::binary_post_ops},
                    {{Multiply, ReLU}, partition_kind_t::binary_post_ops},
                    {{Maximum, Sigmoid}, partition_kind_t::binary_post_ops},
                    {{Maximum, ReLU}, partition_kind_t::binary_post_ops},
                    {{Minimum, Sigmoid}, partition_kind_t::binary_post_ops},
                    {{Minimum, ReLU}, partition_kind_t::binary_post_ops}};

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
        agraph.finalize();

        auto apass = get_pass("binary_post_ops_fusion");
        apass->run(agraph);

        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(), p.second);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
    }
}

TEST(Pass, FuseEltwiseBinary3PostOps) {
    /*
           |
        eltwise
           \     /
            binary
    */
    graph_t agraph;

    op_t relu {0, ReLU, "relu"};
    op_t add {1, Add, "add"};
    op_t mul {2, Multiply, "mul"};
    op_t div {3, Divide, "div"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(8);
    relu.add_input(lt_vec[0]);
    relu.add_output(lt_vec[1]);
    add.add_input(lt_vec[1]);
    add.add_input(lt_vec[2]);
    add.add_output(lt_vec[3]);
    mul.add_input(lt_vec[3]);
    mul.add_input(lt_vec[4]);
    mul.add_output(lt_vec[5]);
    div.add_input(lt_vec[5]);
    div.add_input(lt_vec[6]);
    div.add_output(lt_vec[7]);

    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&mul), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);
    agraph.finalize();

    pass::pass_base_ptr pass = get_pass("eltwise_binary_fusion");
    pass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::unary_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(Pass, FuseEltwiseBinaryFail) {
    /*
           |
          eltwise
        \   /
         div
    */
    graph_t agraph;

    op_t relu {0, ReLU, "relu"};
    op_t div {1, Divide, "div"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    relu.add_input(lt_vec[0]);
    relu.add_output(lt_vec[1]);
    div.add_input(lt_vec[2]);
    div.add_input(lt_vec[1]);
    div.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);
    agraph.finalize();

    pass::pass_base_ptr pass = get_pass("eltwise_binary_fusion");
    pass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 0U);
}

TEST(Pass, ReciprocalMultiply2Divide) {
    /* convert the following pattern to division
                1
                /
        0    reciprocal
        \     /
        multiply
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());

    graph_t agraph;
    op_t reciprocal {0, Reciprocal, "reciprocal"};
    op_t multiply {1, Multiply, "multiply"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    reciprocal.add_input(lt_vec[0]);
    reciprocal.add_output(lt_vec[1]);
    multiply.add_input(lt_vec[1]);
    multiply.add_input(lt_vec[2]);
    multiply.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&reciprocal), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(PassSystem, TestBinaryEltwise) {
    /* binary here represents Add, Multiply, Minimum, Maximum
       eltwise here represents Sigmoid, ReLU

        \    /
        binary
           |
        eltwise
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    std::vector<std::pair<std::pair<op_kind_t, op_kind_t>, partition_kind_t>>
            opkind_pair {{{Add, Sigmoid}, partition_kind_t::binary_post_ops},
                    {{Add, ReLU}, partition_kind_t::binary_post_ops},
                    {{Multiply, Sigmoid}, partition_kind_t::binary_post_ops},
                    {{Multiply, ReLU}, partition_kind_t::binary_post_ops},
                    {{Maximum, Sigmoid}, partition_kind_t::binary_post_ops},
                    {{Maximum, ReLU}, partition_kind_t::binary_post_ops},
                    {{Minimum, Sigmoid}, partition_kind_t::binary_post_ops},
                    {{Minimum, ReLU}, partition_kind_t::binary_post_ops}};

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
        agraph.finalize();

        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(), p.second);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
    }
}

TEST(Pass, FuseBnRelu) {
    /*
         bn
         |
        relu
    */
    graph_t agraph;
    op_t bn {0, BatchNormInference, "bn"};
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("bn_relu_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::batch_norm_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(PassSystem, TestBnRelu) {
    /*
         bn
         |
        relu
    */
    graph_t agraph;
    op_t bn {0, BatchNormInference, "bn"};
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::batch_norm_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FuseBnBwdReluBwd) {
    /*
        ReLUBackward
         |
        BatchNormTrainingBackward
    */
    graph_t agraph;
    op_t op1 {0, ReLUBackward, "op1"};
    op_t op2 {1, BatchNormTrainingBackward, "op2"};
    op2.set_attr(op_attr::epsilon, 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(10);
    op1.add_input(lt_vec[0]);
    op1.add_input(lt_vec[1]);
    op1.add_output(lt_vec[2]);

    op2.add_input(lt_vec[2]);
    op2.add_input(lt_vec[3]);
    op2.add_input(lt_vec[4]);
    op2.add_input(lt_vec[5]);
    op2.add_input(lt_vec[6]);
    op2.add_output(lt_vec[7]);
    op2.add_output(lt_vec[8]);
    op2.add_output(lt_vec[9]);

    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("bn_bwd_relu_bwd_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[1].id, 8U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[2].id, 9U);
}

TEST(PassSystem, TestBnBwdReluBwd) {
    /*
        ReLUBackward
         |
        BatchNormTrainingBackward
    */
    graph_t agraph;
    op_t op1 {0, ReLUBackward, "op1"};
    op_t op2 {1, BatchNormTrainingBackward, "op2"};
    op2.set_attr(op_attr::epsilon, 0.001f);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(10);
    op1.add_input(lt_vec[0]);
    op1.add_input(lt_vec[1]);
    op1.add_output(lt_vec[2]);

    op2.add_input(lt_vec[2]);
    op2.add_input(lt_vec[3]);
    op2.add_input(lt_vec[4]);
    op2.add_input(lt_vec[5]);
    op2.add_input(lt_vec[6]);
    op2.add_output(lt_vec[7]);
    op2.add_output(lt_vec[8]);
    op2.add_output(lt_vec[9]);

    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("bn_bwd_relu_bwd_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::misc_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[1].id, 8U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[2].id, 9U);
}

TEST(Pass, FuseConvSumRelu) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvSumElu) {
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
    elu.set_attr(op_attr::alpha, 0.2f);

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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvSumRelu6) {
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
    op_t relu6 {2, Clamp, "relu6"};
    relu6.set_attr(op_attr::min, 0.f);
    relu6.set_attr(op_attr::max, 6.f);

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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseConvBiasaddSumSum) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 6U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 2U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 11U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 12U);

    ASSERT_EQ((agraph.get_partitions()[1])->get_kind(),
            partition_kind_t::convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[0].id, 7U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[1].id, 8U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[2].id, 10U);

    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs()[0].id, 11U);
}

TEST(Pass, FuseConvBnSum) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 8U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9U);
}

TEST(Pass, FuseConvBnSumWithRelu) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 9U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(Pass, FuseConvBiasBnSum) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 8U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[7].id, 9U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(Pass, FuseConvBnRelu) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8U);
}

TEST(PassSystem, TestConvBnRelu) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8U);
}

TEST(Pass, FuseConvBiasaddBnRelu) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 8U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(Pass, FuseConvBiasBnReluWithInputBias) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 7U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9U);
}

TEST(PassSystem, TestConvBiasBnRelu) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 8U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(Pass, FuseConvBnSumRelu) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 8U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(Pass, FuseConvBiasBnSumRelu) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 8U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[5].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[6].id, 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[7].id, 9U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 11U);
}

TEST(Pass, FuseConvBiasPostOpsChain) {
    size_t max_num_post_ops = 4;
    const std::vector<op_kind_t> two_inputs_ops {
            op_kind::Multiply,
            op_kind::Add,
            op_kind::Maximum,
            op_kind::Minimum,
            op_kind::Divide,
            op_kind::Subtract,
    };
    const std::vector<op_kind_t> supported_ops {
            op_kind::Abs,
            op_kind::Clamp,
            op_kind::Elu,
            op_kind::Exp,
            op_kind::GELU,
            op_kind::HardSigmoid,
            op_kind::HardSwish,
            op_kind::Log,
            op_kind::Sigmoid,
            op_kind::SoftPlus,
            op_kind::ReLU,
            op_kind::Round,
            op_kind::Sqrt,
            op_kind::Square,
            op_kind::Tanh,
            op_kind::Multiply,
            op_kind::Add,
            op_kind::Maximum,
            op_kind::Minimum,
            op_kind::Divide,
            op_kind::Subtract,
    };

    for (size_t k = 0; k < supported_ops.size(); ++k) {
        for (size_t num_chain_ops = 0; num_chain_ops <= max_num_post_ops;
                ++num_chain_ops) {
            graph_t agraph;
            size_t op_id = 0;
            size_t lt_id = 0;
            op_t conv {op_id++, Convolution, "conv"};
            set_conv_common_attr(conv);

            logical_tensor_t conv_in0
                    = logical_tensor_init(lt_id++, data_type::f32);
            logical_tensor_t conv_in1
                    = logical_tensor_init(lt_id++, data_type::f32);
            logical_tensor_t conv_in2
                    = logical_tensor_init(lt_id++, data_type::f32);
            logical_tensor_t conv_out
                    = logical_tensor_init(lt_id++, data_type::f32);

            conv.add_input(conv_in0);
            conv.add_input(conv_in1);
            conv.add_input(conv_in2);
            conv.add_output(conv_out);
            ASSERT_EQ(agraph.add_op(&conv), status::success);

            logical_tensor_t postop_output = conv_out;
            size_t num_two_inputs_post_ops = 0;
            for (size_t i = 0; i < num_chain_ops; ++i) {
                size_t idx = (k + i) % supported_ops.size();
                auto opkind = supported_ops[idx];
                if (std::find(two_inputs_ops.begin(), two_inputs_ops.end(),
                            opkind)
                        != two_inputs_ops.end()) {
                    // binary
                    num_two_inputs_post_ops++;
                    op_t binary {op_id++, opkind, "binary"};
                    if (i == 0) {
                        binary.add_input(conv_out);
                    } else {
                        binary.add_input(postop_output);
                    }

                    logical_tensor_t other_input
                            = logical_tensor_init(lt_id++, data_type::f32);
                    binary.add_input(other_input);
                    postop_output
                            = logical_tensor_init(lt_id++, data_type::f32);
                    binary.add_output(postop_output);

                    ASSERT_EQ(agraph.add_op(&binary), status::success);
                } else {
                    // eltwise
                    op_t eltwise {op_id++, opkind, "eltwise"};
                    switch (opkind) {
                        case graph::op_kind::Elu:
                            eltwise.set_attr<float>(graph::op_attr::alpha, 1.f);
                            break;
                        case graph::op_kind::Clamp:
                            eltwise.set_attr<float>(graph::op_attr::min, -1.f);
                            eltwise.set_attr<float>(graph::op_attr::max, 2.f);
                            break;
                        case graph::op_kind::HardSigmoid:
                            eltwise.set_attr<float>(
                                    graph::op_attr::alpha, 1.0f / 6);
                            eltwise.set_attr<float>(graph::op_attr::beta, 0.5f);
                            break;
                        default: break;
                    }

                    if (i == 0) {
                        eltwise.add_input(conv_out);
                    } else {
                        eltwise.add_input(postop_output);
                    }
                    postop_output
                            = logical_tensor_init(lt_id++, data_type::f32);
                    eltwise.add_output(postop_output);

                    ASSERT_EQ(agraph.add_op(&eltwise), status::success);
                }
            }

            agraph.finalize();
            ASSERT_EQ(agraph.num_ops(), num_chain_ops + 1);

            pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
            apass->run(agraph);
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(),
                    num_chain_ops + 1);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::convolution_post_ops);

            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(),
                    3 + num_two_inputs_post_ops);
            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                    postop_output.id);
        }
    }
}

TEST(Pass, FuseConvPostOpsChain) {
    size_t max_num_post_ops = 3;
    const std::vector<op_kind_t> two_inputs_ops {
            op_kind::Multiply,
            op_kind::Add,
            op_kind::Maximum,
            op_kind::Minimum,
            op_kind::Divide,
            op_kind::Subtract,
    };
    const std::vector<op_kind_t> supported_ops {
            op_kind::Abs,
            op_kind::Clamp,
            op_kind::Elu,
            op_kind::Exp,
            op_kind::GELU,
            op_kind::HardSigmoid,
            op_kind::HardSwish,
            op_kind::Log,
            op_kind::Sigmoid,
            op_kind::SoftPlus,
            op_kind::ReLU,
            op_kind::Round,
            op_kind::Sqrt,
            op_kind::Square,
            op_kind::Tanh,
            op_kind::Multiply,
            op_kind::Add,
            op_kind::Maximum,
            op_kind::Minimum,
            op_kind::Divide,
            op_kind::Subtract,
    };

    for (size_t k = 0; k < supported_ops.size(); ++k) {
        for (size_t num_chain_ops = 0; num_chain_ops <= max_num_post_ops;
                ++num_chain_ops) {
            graph_t agraph;
            size_t op_id = 0;
            size_t lt_id = 0;
            op_t conv {op_id++, Convolution, "conv"};
            set_conv_common_attr(conv);

            logical_tensor_t conv_in0
                    = logical_tensor_init(lt_id++, data_type::f32);
            logical_tensor_t conv_in1
                    = logical_tensor_init(lt_id++, data_type::f32);
            logical_tensor_t conv_out
                    = logical_tensor_init(lt_id++, data_type::f32);

            conv.add_input(conv_in0);
            conv.add_input(conv_in1);
            conv.add_output(conv_out);
            ASSERT_EQ(agraph.add_op(&conv), status::success);

            logical_tensor_t postop_output = conv_out;
            size_t num_two_inputs_post_ops = 0;
            for (size_t i = 0; i < num_chain_ops; ++i) {
                size_t idx = (k + i) % supported_ops.size();
                auto opkind = supported_ops[idx];
                if (std::find(two_inputs_ops.begin(), two_inputs_ops.end(),
                            opkind)
                        != two_inputs_ops.end()) {
                    // binary
                    num_two_inputs_post_ops++;
                    op_t binary {op_id++, opkind, "binary"};
                    if (i == 0) {
                        binary.add_input(conv_out);
                    } else {
                        binary.add_input(postop_output);
                    }

                    logical_tensor_t other_input
                            = logical_tensor_init(lt_id++, data_type::f32);
                    binary.add_input(other_input);
                    postop_output
                            = logical_tensor_init(lt_id++, data_type::f32);
                    binary.add_output(postop_output);

                    ASSERT_EQ(agraph.add_op(&binary), status::success);
                } else {
                    // eltwise
                    op_t eltwise {op_id++, opkind, "eltwise"};
                    switch (opkind) {
                        case graph::op_kind::Elu:
                            eltwise.set_attr<float>(graph::op_attr::alpha, 1.f);
                            break;
                        case graph::op_kind::Clamp:
                            eltwise.set_attr<float>(graph::op_attr::min, -1.f);
                            eltwise.set_attr<float>(graph::op_attr::max, 2.f);
                            break;
                        case graph::op_kind::HardSigmoid:
                            eltwise.set_attr<float>(
                                    graph::op_attr::alpha, 1.0f / 6);
                            eltwise.set_attr<float>(graph::op_attr::beta, 0.5f);
                            break;
                        default: break;
                    }

                    if (i == 0) {
                        eltwise.add_input(conv_out);
                    } else {
                        eltwise.add_input(postop_output);
                    }
                    postop_output
                            = logical_tensor_init(lt_id++, data_type::f32);
                    eltwise.add_output(postop_output);

                    ASSERT_EQ(agraph.add_op(&eltwise), status::success);
                }
            }

            agraph.finalize();
            ASSERT_EQ(agraph.num_ops(), num_chain_ops + 1);

            pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
            apass->run(agraph);
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::convolution_post_ops);

            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(),
                    2 + num_two_inputs_post_ops);
            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                    postop_output.id);
        }
    }
}

TEST(Pass, FuseConvtransposeBiasadd) {
    /*   conv
          |
         bias
    */
    graph_t agraph;
    op_t convtranspose {0, ConvTranspose, "convtranspose"};
    set_convtranspose_common_attr(convtranspose);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("convtranspose_post_ops_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseConvtransposeAdd) {
    /*   convtranspose
          |
         w/wo bias
          |
         add
    */
    std::vector<bool> with_biases {false, true};

    for (auto with_bias : with_biases) {
        graph_t agraph;
        op_t convtranspose {0, ConvTranspose, "convtranspose"};
        set_convtranspose_common_attr(convtranspose);
        op_t add {1, Add, "add"};

        std::vector<logical_tensor_t> lt_vec
                = create_logical_tensors(with_bias ? 6 : 5);
        int lt_id = -1;
        convtranspose.add_input(lt_vec[++lt_id]);
        convtranspose.add_input(lt_vec[++lt_id]);
        if (with_bias) convtranspose.add_input(lt_vec[++lt_id]);
        convtranspose.add_output(lt_vec[++lt_id]);
        add.add_input(lt_vec[lt_id]);
        add.add_input(lt_vec[++lt_id]);
        add.add_output(lt_vec[++lt_id]);

        ASSERT_EQ(agraph.add_op(&convtranspose), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        agraph.finalize();
        ASSERT_EQ(agraph.num_ops(), 2U);

        pass::pass_base_ptr apass = get_pass("convtranspose_post_ops_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(),
                with_bias ? 4U : 3U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
        if (with_bias) {
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
        } else {
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
        }

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                with_bias ? 5U : 4U);
    }
}

TEST(Pass, FuseConvtransposeAddTwoInputs) {
    /*   convtranspose
          |
         bias (is a convtranspose third input)
          |
         add
    */

    graph_t agraph;
    op_t convtranspose {0, ConvTranspose, "convtranspose"};
    set_convtranspose_common_attr(convtranspose);
    op_t bias {1, BiasAdd, "bias"};
    op_t add {2, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(7);
    convtranspose.add_input(lt_vec[0]);
    convtranspose.add_input(lt_vec[1]);
    convtranspose.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    add.add_input(lt_vec[4]);
    add.add_input(lt_vec[5]);
    add.add_output(lt_vec[6]);

    ASSERT_EQ(agraph.add_op(&convtranspose), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("convtranspose_post_ops_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FuseConvtransposeRelu) {
    /*   convtranspose
          |
         w/wo bias (is a convtranspose third input)
          |
         relu
    */
    std::vector<bool> with_biases {false, true};

    for (auto with_bias : with_biases) {
        graph_t agraph;
        op_t convtranspose {0, ConvTranspose, "convtranspose"};
        set_convtranspose_common_attr(convtranspose);
        op_t relu {1, ReLU, "relu"};

        std::vector<logical_tensor_t> lt_vec
                = create_logical_tensors(with_bias ? 5 : 4);
        int lt_id = -1;
        convtranspose.add_input(lt_vec[++lt_id]);
        convtranspose.add_input(lt_vec[++lt_id]);
        if (with_bias) convtranspose.add_input(lt_vec[++lt_id]);
        convtranspose.add_output(lt_vec[++lt_id]);
        relu.add_input(lt_vec[lt_id]);
        relu.add_output(lt_vec[++lt_id]);

        ASSERT_EQ(agraph.add_op(&convtranspose), status::success);
        ASSERT_EQ(agraph.add_op(&relu), status::success);
        agraph.finalize();
        ASSERT_EQ(agraph.num_ops(), 2U);

        pass::pass_base_ptr apass = get_pass("convtranspose_post_ops_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(),
                with_bias ? 3U : 2U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
        if (with_bias) {
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
        }

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                with_bias ? 4U : 3U);
    }
}

TEST(Pass, FuseConvtransposeReLUTwoInputs) {
    /*   convtranspose
          |
         bias
          |
         relu
    */
    graph_t agraph;
    op_t convtranspose {0, ConvTranspose, "convtranspose"};
    set_convtranspose_common_attr(convtranspose);
    op_t bias {1, BiasAdd, "bias"};
    op_t relu {2, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    convtranspose.add_input(lt_vec[0]);
    convtranspose.add_input(lt_vec[1]);
    convtranspose.add_output(lt_vec[2]);
    bias.add_input(lt_vec[2]);
    bias.add_input(lt_vec[3]);
    bias.add_output(lt_vec[4]);
    relu.add_input(lt_vec[4]);
    relu.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&convtranspose), status::success);
    ASSERT_EQ(agraph.add_op(&bias), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("convtranspose_post_ops_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseMatmulRelu) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseMatmulReluCase2) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    pass::pass_base_ptr apass2 = get_pass("fp_matmul_post_ops");
    apass2->run(agraph);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FailToFuseReluMatmul) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
}

TEST(Pass, FuseMatmulElu) {
    /*  matmul
          |
        elu
    */
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t elu {1, Elu, "elu"};
    elu.set_attr(op_attr::alpha, 0.1f);
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    elu.add_input(lt_vec[2]);
    elu.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&elu), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseMatmulSigmoid) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseMatmulClamp) {
    /*  matmul
          |
        clamp
    */
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t clamp {1, Clamp, "clamp"};
    clamp.set_attr(op_attr::min, -1.f);
    clamp.set_attr(op_attr::max, 1.f);
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    clamp.add_input(lt_vec[2]);
    clamp.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&clamp), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseMatmulGelu) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseMatmulSum) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseMatmulSumWithCommunicativeOrder) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseMatmulSumGelu) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseMatmulSumRelu) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseMatmulDiv) {
    /*  matmul  wildcard
          \    /
            div
    */
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t wildcard {1, Wildcard, "wildcard"};
    op_t div {2, Divide, "div"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    wildcard.add_output(lt_vec[3]);
    div.add_input(lt_vec[2]);
    div.add_input(lt_vec[3]);
    div.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(PassSystem, FuseMatmulDiv) {
    /*  matmul  wildcard
          \    /
            div
    */
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t wildcard {1, Wildcard, "wildcard"};
    op_t div {2, Divide, "div"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    wildcard.add_output(lt_vec[3]);
    div.add_input(lt_vec[2]);
    div.add_input(lt_vec[3]);
    div.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);
}

TEST(Pass, FuseMatmulDivAdd) {
    /*  matmul  wildcard
          \    /
            div wildcard
             | /
            add
    */
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t wildcard {1, Wildcard, "wildcard"};
    op_t wildcard2 {2, Wildcard, "wildcard2"};
    op_t div {3, Divide, "div"};
    op_t add {4, Add, "add"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(7);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    wildcard.add_output(lt_vec[3]);
    div.add_input(lt_vec[2]);
    div.add_input(lt_vec[3]);
    div.add_output(lt_vec[4]);
    wildcard2.add_output(lt_vec[5]);
    add.add_input(lt_vec[4]);
    add.add_input(lt_vec[5]);
    add.add_output(lt_vec[6]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard2), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 5U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(PassSystem, FuseMatmulDivAdd) {
    /*  matmul  wildcard
          \    /
            div wildcard
             | /
            add
    */
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t wildcard {1, Wildcard, "wildcard"};
    op_t wildcard2 {2, Wildcard, "wildcard2"};
    op_t div {3, Divide, "div"};
    op_t add {4, Add, "add"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(7);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    wildcard.add_output(lt_vec[3]);
    div.add_input(lt_vec[2]);
    div.add_input(lt_vec[3]);
    div.add_output(lt_vec[4]);
    wildcard2.add_output(lt_vec[5]);
    add.add_input(lt_vec[4]);
    add.add_input(lt_vec[5]);
    add.add_output(lt_vec[6]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);
    ASSERT_EQ(agraph.add_op(&wildcard2), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 5U);

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);
}

TEST(PassSystem, TestMatmulDivAdd) {
    /*  matmul
          \    /
            div
             | /
            add
    */
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t div {3, Divide, "div"};
    op_t add {4, Add, "add"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(7);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_output(lt_vec[2]);
    div.add_input(lt_vec[2]);
    div.add_input(lt_vec[3]);
    div.add_output(lt_vec[4]);
    add.add_input(lt_vec[4]);
    add.add_input(lt_vec[5]);
    add.add_output(lt_vec[6]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FuseMatmulBiasadd) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseMatmulBias) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 1U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseMatmulBiasSigmoid) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    pass::pass_base_ptr apass2 = get_pass("fp_matmul_post_ops");
    apass2->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseMatmulBiasaddElu) {
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
    elu.set_attr(op_attr::alpha, 0.1f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseMatmulBiasaddRelu) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseMatmulBiasaddClamp) {
    /*  matmul
           |
         bias
           |
        clamp
    */
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t clamp {1, Clamp, "clamp"};
    clamp.set_attr(op_attr::min, 0.1f);
    clamp.set_attr(op_attr::max, 0.2f);
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_input(lt_vec[2]);
    matmul.add_output(lt_vec[3]);
    clamp.add_input(lt_vec[3]);
    clamp.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&clamp), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    pass::pass_base_ptr apass2 = get_pass("fp_matmul_post_ops");
    apass2->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseMatmulReluSum) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FuseMatmulBiasSumRelu) {
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

    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(PassSystem, TestMatmulBiasSumRelu) {
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

    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(Pass, FuseMatmulBiasaddSwish) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 4U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(PassSystem, FuseMatmulBiasaddSwish) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    // run all the pass to check if the priority is correct
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FuseMatmulBiasaddRelu6) {
    /*  matmul
           |
         bias
           |
         relu6
    */
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};
    op_t relu6 {1, Clamp, "clamp"};
    relu6.set_attr(op_attr::min, 0.f);
    relu6.set_attr(op_attr::max, 6.f);
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    matmul.add_input(lt_vec[0]);
    matmul.add_input(lt_vec[1]);
    matmul.add_input(lt_vec[2]);
    matmul.add_output(lt_vec[3]);
    relu6.add_input(lt_vec[3]);
    relu6.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu6), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, DnnlSingleOpReplacement) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    std::vector<op_kind_t> single_op_set_supported = {
            BatchNormInference,
            Add,
            ReLU,
            MatMul,
            AvgPool,
            MaxPool,
            Clamp,
            ConvolutionBackwardData,
            ConvolutionBackwardWeights,
            MaxPoolBackward,
            Elu,
            Exp,
            HardSigmoid,
            HardSwish,
            Log,
            LogSoftmax,
            SoftMax,
            Multiply,
            Maximum,
            Minimum,
            Mish,
            MishBackward,
            Sqrt,
            Square,
            Tanh,
            ClampBackward,
            EluBackward,
            GELUBackward,
            LogSoftmaxBackward,
            ReLUBackward,
            SigmoidBackward,
            SqrtBackward,
            TanhBackward,
            LayerNorm,
            LayerNormBackward,
            SoftMaxBackward,
            DynamicQuantize,
            DynamicDequantize,
    };
    for (auto akind : single_op_set_supported) {
        graph_t agraph;
        op_t *op = agraph.create_op(akind);
        // specially handle AvgPool
        // which requires certain combinations of attributes
        if (akind == AvgPool) {
            op->set_attr<bool>(op_attr::exclude_pad, false);
            op->set_attr<std::string>(op_attr::rounding_type, "floor");
        }
        ASSERT_EQ(op->get_kind(), akind);
        pm.run_passes(agraph, "no_config");

        auto orig_op = agraph.get_ops()[0];
        ASSERT_NE(orig_op->get_partition(), nullptr);
        ASSERT_EQ(orig_op->get_partition()->get_assigned_backend()->get_name(),
                std::string("dnnl_backend"));
        ASSERT_EQ((agraph.get_partitions()[0])->get_ops().size(), 1U);
    }
}

struct single_op_params_t {
    op_kind_t op_kind;
    size_t input_num;
    size_t output_num;
    graph::data_type_t data_type;
    size_t partition_num;
    std::vector<graph::op_attr_t> float_attrs;
};

class test_single_op_pass_t
    : public ::testing::TestWithParam<single_op_params_t> {
public:
    void Test_Single_Op_Pass() {
        const auto params
                = ::testing::TestWithParam<single_op_params_t>::GetParam();

        graph_t agraph;
        op_t aop {0, params.op_kind, "aop"};
        for (const auto &attr : params.float_attrs) {
            aop.set_attr(attr, 0.1f);
        }

        std::vector<logical_tensor_t> lts = create_logical_tensors(
                params.input_num + params.output_num, params.data_type);
        for (size_t i = 0; i < params.input_num; ++i) {
            aop.add_input(lts[i]);
        }
        for (size_t i = 0; i < params.output_num; ++i) {
            aop.add_output(lts[params.input_num + i]);
        }

        ASSERT_EQ(agraph.add_op(&aop), status::success);
        agraph.finalize();
        ASSERT_EQ(agraph.num_ops(), 1U);

        auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
        auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), params.partition_num);
    }
};

TEST_P(test_single_op_pass_t, Test_Single_Op_Pass) {
    Test_Single_Op_Pass();
}

INSTANTIATE_TEST_SUITE_P(Test_Single_Op_Pass, test_single_op_pass_t,
        ::testing::Values(single_op_params_t {graph::op_kind::Round, 1, 1,
                                  graph::data_type::f32, 1},
                single_op_params_t {
                        graph::op_kind::Round, 1, 1, graph::data_type::bf16, 0},
                single_op_params_t {
                        graph::op_kind::Round, 1, 1, graph::data_type::f16, 0},
                single_op_params_t {graph::op_kind::BatchNormInference, 5, 1,
                        graph::data_type::f32, 1, {graph::op_attr::epsilon}},
                single_op_params_t {graph::op_kind::BatchNormInference, 5, 1,
                        graph::data_type::bf16, 0, {graph::op_attr::epsilon}},
                single_op_params_t {graph::op_kind::BatchNormForwardTraining, 5,
                        5, graph::data_type::f32, 1, {graph::op_attr::epsilon}},
                single_op_params_t {graph::op_kind::BatchNormForwardTraining, 5,
                        5, graph::data_type::bf16, 0,
                        {graph::op_attr::epsilon}},
                single_op_params_t {graph::op_kind::BatchNormTrainingBackward,
                        5, 3, graph::data_type::f32, 1,
                        {graph::op_attr::epsilon}},
                single_op_params_t {graph::op_kind::BatchNormTrainingBackward,
                        5, 3, graph::data_type::bf16, 0,
                        {graph::op_attr::epsilon}},
                single_op_params_t {graph::op_kind::LayerNorm, 3, 3,
                        graph::data_type::f32, 1},
                single_op_params_t {graph::op_kind::LayerNorm, 3, 3,
                        graph::data_type::bf16, 0},
                single_op_params_t {graph::op_kind::LayerNormBackward, 6, 3,
                        graph::data_type::f32, 1},
                single_op_params_t {graph::op_kind::LayerNormBackward, 6, 3,
                        graph::data_type::bf16, 0}));

TEST(Pass, ConvSingleOpReplacement) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(3);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_output(lt_vec[2]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 1U);

    pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    auto &orig_op = agraph.get_ops()[0];
    ASSERT_NE(orig_op->get_partition(), nullptr);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2U);
}

TEST(Pass, ConvSingleOpReplacementWithBias) {
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    conv.add_input(lt_vec[0]);
    conv.add_input(lt_vec[1]);
    conv.add_input(lt_vec[2]);
    conv.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 1U);

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    auto &orig_op = agraph.get_ops()[0];
    ASSERT_NE(orig_op->get_partition(), nullptr);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, SaveLoadJson) {
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
    bn.set_attr(op_attr::epsilon, 0.001f);
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 5U);

    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());

    pm.print_passes("passes.json");
    pm.run_passes(agraph, "passes.json");
    ASSERT_EQ(agraph.num_ops(), 5U);
    ASSERT_EQ(agraph.get_num_partitions(), 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 9U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 10U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 8U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 12U);

    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 6U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[2].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[3].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[4].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[5].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs()[0].id, 8U);
}

TEST(Pass, InputJsonIsValid) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());

    std::ostringstream valid_stream;
    std::string version = std::to_string(dnnl_version()->major) + "."
            + std::to_string(dnnl_version()->minor) + "."
            + std::to_string(dnnl_version()->patch);
    valid_stream << "{\n"
                 << "\"version\": \"" << version << "\",\n"
                 << "\"hash\": \"" << dnnl_version()->hash << "\",\n"
                 << "\"passes\": [\n"
                 << "  {\n"
                 << "  \"pass_name\": \"conv_pass\",\n"
                 << "  \"pass_backend\": \"dnnl\",\n"
                 << "  \"priority\": 8,\n"
                 << "  \"enable\": 1\n"
                 << "  },\n"
                 << "  {\n"
                 << "  \"pass_name\": \"relu_pass\",\n"
                 << "  \"pass_backend\": \"dnnl\",\n"
                 << "  \"priority\": 8,\n"
                 << "  \"enable\": 1\n"
                 << "  }\n"
                 << "]\n"
                 << "}\n";
    std::string valid_str = valid_stream.str();
    std::istringstream valid_is(valid_str);
    pm.run_passes(agraph, &valid_is);
    ASSERT_EQ(agraph.get_num_partitions(), 2U);
}

TEST(Pass, InputJsonIsInvalidWithIncompleteHash) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());

    std::ostringstream invalid_stream;
    std::string version = std::to_string(dnnl_version()->major) + "."
            + std::to_string(dnnl_version()->minor) + "."
            + std::to_string(dnnl_version()->patch);
    invalid_stream << "{\n"
                   << "\"version\": \"" << version << "\",\n"
                   << "\"hash\": \""
                   << "aninvalidcommitid"
                   << "\",\n"
                   << "\"passes\": [\n"
                   << "  {\n"
                   << "  \"pass_name\": \"conv_pass\",\n"
                   << "  \"pass_backend\": \"dnnl\",\n"
                   << "  \"priority\": 8,\n"
                   << "  \"enable\": 1\n"
                   << "  },\n"
                   << "  {\n"
                   << "  \"pass_name\": \"relu_pass\",\n"
                   << "  \"pass_backend\": \"dnnl\",\n"
                   << "  \"priority\": 8,\n"
                   << "  \"enable\": 1\n"
                   << "  }\n"
                   << "]\n"
                   << "}\n";
    std::string invalid_str = invalid_stream.str();
    std::istringstream invalid_is(invalid_str);
    pm.run_passes(agraph, &invalid_is);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
}

TEST(Pass, InputJsonIsInvalidWithMissingFiled) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());

    std::ostringstream invalid_stream;
    invalid_stream << "{\n"
                   << "\"passes\": [\n"
                   << "  {\n"
                   << "  \"pass_name\": \"conv_pass\",\n"
                   << "  \"pass_backend\": \"dnnl\",\n"
                   << "  \"priority\": 8,\n"
                   << "  \"enable\": 1\n"
                   << "  },\n"
                   << "  {\n"
                   << "  \"pass_name\": \"relu_pass\",\n"
                   << "  \"pass_backend\": \"dnnl\",\n"
                   << "  \"priority\": 8,\n"
                   << "  \"enable\": 1\n"
                   << "  }\n"
                   << "]\n"
                   << "}\n";
    std::string invalid_str = invalid_stream.str();
    std::istringstream invalid_is(invalid_str);
    pm.run_passes(agraph, &invalid_is);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
}

TEST(Pass, InputJsonIsInvalidWithWrongFormat) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());

    std::ostringstream invalid_stream;
    std::string version = std::to_string(dnnl_version()->major) + "."
            + std::to_string(dnnl_version()->minor) + "."
            + std::to_string(dnnl_version()->patch);
    invalid_stream << "\"version\": \"" << version << "\",\n"
                   << "\"hash\": \"" << dnnl_version()->hash << "\",\n"
                   << "\"passes\": [\n"
                   << "  {\n"
                   << "  \"pass_name\": \"conv_pass\",\n"
                   << "  \"pass_backend\": \"dnnl\",\n"
                   << "  \"priority\": 8,\n"
                   << "  \"enable\": 1\n"
                   << "  },\n";
    std::string invalid_str = invalid_stream.str();
    std::istringstream invalid_is(invalid_str);
    pm.run_passes(agraph, &invalid_is);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
}

TEST(Pass, FuseTwoConvReluWithSharedWeight) {
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

    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);
    pass::pass_base_ptr conv_post_ops_fusion_pass
            = get_pass("fp_conv_post_ops");
    conv_post_ops_fusion_pass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 2U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);

    ASSERT_EQ((agraph.get_partitions()[1])->get_kind(),
            partition_kind_t::convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[0].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs()[0].id, 5U);
}

TEST(Pass, CheckSameInput) {
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

    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(agraph);
    apass = get_pass("sum_pass");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2U);

    // For a partition with N inputs that have the same id
    // It is required that those inputs are input N times
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[0].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs()[0].id, 3U);
}

TEST(PassSystem, FuseToInt8Conv) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
           quant
             | (u8/s8)
    */
    std::vector<engine_kind_t> engine_kinds
            = {engine_kind::cpu, engine_kind::gpu};
    for (const auto &engine_kind : engine_kinds) {
        graph_t agraph(engine_kind);
        std::vector<int64_t> zps = {2};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr(op_attr::scales, scales);
        dequant1.set_attr(op_attr::zps, zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales);
        dequant2.set_attr(op_attr::zps, zps);
        op_t conv {2, Convolution, "conv"};
        set_conv_common_attr(conv);
        op_t relu {3, ReLU, "relu"};
        op_t quant {4, Quantize, "quant"};
        quant.set_attr(op_attr::scales, scales);
        quant.set_attr(op_attr::zps, zps);
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

        agraph.finalize();

        auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
        auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::quantized_convolution_post_ops);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
    }
}

TEST(Pass, FuseToInt8Fp32Conv) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
            /  \
        relu   relu
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    op_t relu2 {4, ReLU, "relu"};
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

    logical_tensor_t fp32_relu2_out = logical_tensor_init(6, data_type::f32);
    relu2.add_input(fp32_conv_out);
    relu2.add_output(fp32_relu2_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&relu2), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(PassSystem, TestInt8) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t quant {3, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FailToFuseToInt8Conv) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 4U);
}

TEST(Pass, FuseToInt8ConvBias) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t quant {3, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(PassSystem, TestInt8ConvBias) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t quant {3, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FuseToInt8ConvRelu) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FuseToInt8ConvSwish) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
      (f32) / |
     sigmoid  |
           \  |
          multiply       
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t sigmoid {3, Sigmoid, "sigmoid"};
    op_t multiply {4, Multiply, "mul"};
    op_t quant {5, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t fp32_sigmoid_out = logical_tensor_init(5, data_type::f32);
    sigmoid.add_input(fp32_conv_out);
    sigmoid.add_output(fp32_sigmoid_out);

    logical_tensor_t fp32_mul_out = logical_tensor_init(6, data_type::f32);
    multiply.add_input(fp32_conv_out);
    multiply.add_input(fp32_sigmoid_out);
    multiply.add_output(fp32_mul_out);

    logical_tensor_t int8_out = logical_tensor_init(7, data_type::u8);
    quant.add_input(fp32_mul_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(PassSystem, TestInt8ConvRelu) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FuseToInt8ConvBiasRelu) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(PassSystem, TestInt8ConvBiasRelu) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(Pass, FuseToInt8ConvBiasAdd) {
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
    const std::vector<engine_kind_t> engine_kinds
            = {engine_kind::cpu, engine_kind::gpu};
    for (const auto &engine_kind : engine_kinds) {
        graph_t agraph(engine_kind);
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr(op_attr::scales, scales);
        dequant1.set_attr(op_attr::zps, zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales);
        dequant2.set_attr(op_attr::zps, zps);
        op_t dequant3 {2, Dequantize, "dequant"};
        dequant3.set_attr(op_attr::scales, scales);
        dequant3.set_attr(op_attr::zps, zps);
        op_t conv {3, Convolution, "conv"};
        set_conv_common_attr(conv);
        op_t add {5, Add, "add"};
        op_t quant {6, Quantize, "quant"};
        quant.set_attr(op_attr::scales, scales);
        quant.set_attr(op_attr::zps, zps);

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

        agraph.finalize();

        pass::pass_base_ptr apass = get_pass(engine_kind == engine_kind::cpu
                        ? "x8s8x8_conv_add_post_ops_cpu"
                        : "x8s8x8_conv_add_post_ops_gpu");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::quantized_convolution_post_ops);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9U);
    }
}

TEST(Pass, FuseToInt8ConvBinary) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            conv w/wo bias
             | (f32)
            binary
             | (f32)
           quant
             | (u8/s8)
    */
    const std::vector<graph::op_kind_t> binary_kinds = {
            Maximum,
            Minimum,
            Divide,
            Multiply,
            Subtract,
    };
    std::vector<bool> with_biases {false, true};

    for (auto &binary_kind : binary_kinds) {
        for (auto with_bias : with_biases) {
            graph_t agraph;
            std::vector<int64_t> zps = {0};
            std::vector<float> scales = {3.1f};
            op_t dequant1 {0, Dequantize, "dequant"};
            dequant1.set_attr(op_attr::scales, scales);
            dequant1.set_attr(op_attr::zps, zps);
            op_t dequant2 {1, Dequantize, "dequant"};
            dequant2.set_attr(op_attr::scales, scales);
            dequant2.set_attr(op_attr::zps, zps);
            op_t conv {2, Convolution, "conv"};
            set_conv_common_attr(conv);
            op_t binary {3, binary_kind, "binary"};
            op_t quant {4, Quantize, "quant"};
            quant.set_attr(op_attr::scales, scales);
            quant.set_attr(op_attr::zps, zps);

            int lt_id = -1;
            logical_tensor_t int8_data
                    = logical_tensor_init(++lt_id, data_type::u8);
            logical_tensor_t fp32_data
                    = logical_tensor_init(++lt_id, data_type::f32);
            dequant1.add_input(int8_data);
            dequant1.add_output(fp32_data);

            logical_tensor_t s8_weight
                    = logical_tensor_init(++lt_id, data_type::s8);
            logical_tensor_t fp32_weight
                    = logical_tensor_init(++lt_id, data_type::f32);
            dequant2.add_input(s8_weight);
            dequant2.add_output(fp32_weight);

            logical_tensor_t fp32_bias;
            if (with_bias)
                fp32_bias = logical_tensor_init(++lt_id, data_type::f32);

            logical_tensor_t fp32_conv_out
                    = logical_tensor_init(++lt_id, data_type::f32);
            conv.add_input(fp32_data);
            conv.add_input(fp32_weight);
            if (with_bias) conv.add_input(fp32_bias);
            conv.add_output(fp32_conv_out);

            logical_tensor_t fp32_binary_other
                    = logical_tensor_init(++lt_id, data_type::f32);
            logical_tensor_t fp32_binary_out
                    = logical_tensor_init(++lt_id, data_type::f32);
            binary.add_input(fp32_conv_out);
            binary.add_input(fp32_binary_other);
            binary.add_output(fp32_binary_out);

            logical_tensor_t int8_out
                    = logical_tensor_init(++lt_id, data_type::u8);
            quant.add_input(fp32_binary_out);
            quant.add_output(int8_out);

            ASSERT_EQ(agraph.add_op(&dequant1), status::success);
            ASSERT_EQ(agraph.add_op(&dequant2), status::success);
            ASSERT_EQ(agraph.add_op(&conv), status::success);
            ASSERT_EQ(agraph.add_op(&binary), status::success);
            ASSERT_EQ(agraph.add_op(&quant), status::success);

            agraph.finalize();

            pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
            apass->run(agraph);
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::quantized_convolution_post_ops);

            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(),
                    with_bias ? 4U : 3U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
            if (with_bias) {
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 6U);
            } else {
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 5U);
            }

            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                    with_bias ? 8U : 7U);
        }
    }
}

TEST(PassSystem, TestInt8ConvBiasAdd) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            conv_with_bias
             | (f32)
             |     | (s8)
             |   dequant*
             |  / (f32)
            add
             | (f32)
           quant
             | (u8/s8)
    */
    std::vector<bool> dequant_for_add = {true, false};
    for (auto with_dequant_for_add : dequant_for_add) {
        graph_t agraph;
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr(op_attr::scales, scales);
        dequant1.set_attr(op_attr::zps, zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales);
        dequant2.set_attr(op_attr::zps, zps);
        op_t dequant3 {2, Dequantize, "dequant"};
        dequant3.set_attr(op_attr::scales, scales);
        dequant3.set_attr(op_attr::zps, zps);
        op_t conv {3, Convolution, "conv"};
        set_conv_common_attr(conv);
        op_t add {5, Add, "add"};
        op_t quant {6, Quantize, "quant"};
        quant.set_attr(op_attr::scales, scales);
        quant.set_attr(op_attr::zps, zps);

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
        if (with_dequant_for_add) {
            dequant3.add_input(int8_other);
            dequant3.add_output(fp32_other);
        }

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
        if (with_dequant_for_add) {
            ASSERT_EQ(agraph.add_op(&dequant3), status::success);
        }
        ASSERT_EQ(agraph.add_op(&conv), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        ASSERT_EQ(agraph.add_op(&quant), status::success);

        agraph.finalize();

        auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
        auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::quantized_convolution_post_ops);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);
        if (with_dequant_for_add) {
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);
        } else {
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 5U);
        }

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9U);
    }
}

TEST(Pass, FuseToInt8ConvBiasAddRelu) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr(op_attr::scales, scales);
    dequant3.set_attr(op_attr::zps, zps);
    op_t conv {3, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {5, Add, "add"};
    op_t relu {6, ReLU, "relu"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x8_conv_add_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(PassSystem, TestInt8ConvBiasAddRelu) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr(op_attr::scales, scales);
    dequant3.set_attr(op_attr::zps, zps);
    op_t conv {3, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {5, Add, "add"};
    op_t relu {6, ReLU, "relu"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(Pass, FuseToInt8ConvBiasAddReluWithInputBias) {
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
    std::vector<int64_t> zps = {1};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr(op_attr::scales, scales);
    dequant3.set_attr(op_attr::zps, zps);
    op_t conv {3, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {5, Add, "add"};
    set_conv_common_attr(conv);
    op_t relu {6, ReLU, "relu"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x8_conv_add_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(Pass, FuseToX8s8f32Conv) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseToX8s8f32ConvBiasWithInputBias) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseToX8s8f32ConvReluWithInputBias) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseToX8s8f32ConvBiasReluWithInputBias) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, TestQuantizedConv) {
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
          quantize
             | (u8/s8)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    graph_t agraph;
    //asymmetric zps
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr(op_attr::scales, scales);
    dequant3.set_attr(op_attr::zps, zps);
    op_t conv {3, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t add {4, Add, "add"};
    set_conv_common_attr(conv);
    op_t relu {5, ReLU, "relu"};
    op_t dequant4 {6, Dequantize, "dequant"};
    dequant4.set_attr(op_attr::scales, scales);
    dequant4.set_attr(op_attr::zps, zps);
    op_t dequant5 {7, Dequantize, "dequant"};
    dequant5.set_attr(op_attr::scales, scales);
    dequant5.set_attr(op_attr::zps, zps);
    op_t quant {8, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);
    op_t conv1 {9, Convolution, "conv"};
    set_conv_common_attr(conv1);
    op_t quant1 {10, Quantize, "quant"};
    quant1.set_attr(op_attr::scales, scales);
    quant1.set_attr(op_attr::zps, zps);

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

    logical_tensor_t int8_out2 = logical_tensor_init(15, data_type::u8);
    quant1.add_input(fp32_relu_out);
    quant1.add_output(int8_out2);

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
    ASSERT_EQ(agraph.add_op(&quant1), status::success);

    agraph.finalize();

    // run all the pass to check if the priority is correct
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 2U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 15U);

    ASSERT_EQ((agraph.get_partitions()[1])->get_kind(),
            partition_kind_t::quantized_convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[0].id, 10U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[1].id, 12U);

    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseToInt8Matmul) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    op_t quant {3, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(PassSystem, TestInt8Matmul) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    op_t quant {3, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    // run all the pass to check if the priority is correct
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, OptionalQuantForInt8Matmul) {
    /*
    quant_wei has a producer, so it will not
    be fused into int8_matmul_post_ops
                matmul
                   |
                 quant
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

    op_t matmul0 {0, MatMul, "matmul"};
    op_t quant1 {1, Quantize, "quant"};
    quant1.set_attr(op_attr::scales, scales);
    quant1.set_attr(op_attr::zps, zps);

    op_t dequant1 {2, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);

    op_t dequant2 {3, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {4, MatMul, "matmul"};
    op_t quant2 {5, Quantize, "quant"};
    quant2.set_attr(op_attr::scales, scales);
    quant2.set_attr(op_attr::zps, zps);

    logical_tensor_t fp32_data0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t fp32_weight0 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t fp32_out0 = logical_tensor_init(2, data_type::f32);
    matmul0.add_input(fp32_data0);
    matmul0.add_input(fp32_weight0);
    matmul0.add_output(fp32_out0);

    logical_tensor_t s8_weight = logical_tensor_init(3, data_type::s8);
    quant1.add_input(fp32_out0);
    quant1.add_output(s8_weight);

    logical_tensor_t int8_data = logical_tensor_init(4, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(5, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t fp32_weight = logical_tensor_init(6, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_matmul_out = logical_tensor_init(7, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t int8_out = logical_tensor_init(8, data_type::u8);
    quant2.add_input(fp32_matmul_out);
    quant2.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&quant1), status::success);
    ASSERT_EQ(agraph.add_op(&matmul0), status::success);
    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&quant2), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8U);
}

TEST(Pass, OptionalQuantWith2ConsumersForInt8Matmul) {
    /*
    quant_wei has two consumers, so it will not
    be fused into int8_matmul_post_ops
                         |
                       quant
        |(u8/s8)  (s8) /   \
     dequant    dequant   dequant
    (f32) \     / (f32)
           matmul
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};

    op_t quant1 {0, Quantize, "quant"};
    quant1.set_attr(op_attr::scales, scales);
    quant1.set_attr(op_attr::zps, zps);

    op_t dequant1 {1, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);

    op_t dequant2 {2, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);

    op_t dequant3 {3, Dequantize, "dequant"};
    dequant3.set_attr(op_attr::scales, scales);
    dequant3.set_attr(op_attr::zps, zps);

    op_t matmul {4, MatMul, "matmul"};
    op_t quant2 {5, Quantize, "quant"};
    quant2.set_attr(op_attr::scales, scales);
    quant2.set_attr(op_attr::zps, zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t fp32_weight = logical_tensor_init(2, data_type::f32);
    logical_tensor_t s8_weight = logical_tensor_init(3, data_type::s8);
    quant1.add_input(fp32_weight);
    quant1.add_output(s8_weight);

    logical_tensor_t fp32_weight2 = logical_tensor_init(4, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight2);

    // another consumer of quant1's output
    logical_tensor_t fp32_weight3 = logical_tensor_init(5, data_type::f32);
    dequant3.add_input(s8_weight);
    dequant3.add_output(fp32_weight3);

    logical_tensor_t fp32_matmul_out = logical_tensor_init(6, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight2);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t int8_out = logical_tensor_init(7, data_type::u8);
    quant2.add_input(fp32_matmul_out);
    quant2.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&quant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&quant2), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 4U);
}

TEST(Pass, FuseToInt8MatMulBinary) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
          matmul w/wo bias
             | (f32)
            binary
             | (f32)
           quant
             | (u8/s8)
    */
    const std::vector<graph::op_kind_t> binary_kinds = {
            Maximum,
            Minimum,
            Divide,
            Multiply,
            Subtract,
    };
    std::vector<bool> with_biases {false, true};

    for (auto &binary_kind : binary_kinds) {
        for (auto with_bias : with_biases) {
            graph_t agraph;
            std::vector<int64_t> zps = {0};
            std::vector<float> scales = {3.1f};
            op_t dequant1 {0, Dequantize, "dequant"};
            dequant1.set_attr(op_attr::scales, scales);
            dequant1.set_attr(op_attr::zps, zps);
            op_t dequant2 {1, Dequantize, "dequant"};
            dequant2.set_attr(op_attr::scales, scales);
            dequant2.set_attr(op_attr::zps, zps);
            op_t matmul {2, MatMul, "matmul"};
            op_t binary {3, binary_kind, "binary"};
            op_t quant {4, Quantize, "quant"};
            quant.set_attr(op_attr::scales, scales);
            quant.set_attr(op_attr::zps, zps);

            int lt_id = -1;
            logical_tensor_t int8_data
                    = logical_tensor_init(++lt_id, data_type::u8);
            logical_tensor_t fp32_data
                    = logical_tensor_init(++lt_id, data_type::f32);
            dequant1.add_input(int8_data);
            dequant1.add_output(fp32_data);

            logical_tensor_t s8_weight
                    = logical_tensor_init(++lt_id, data_type::s8);
            logical_tensor_t fp32_weight
                    = logical_tensor_init(++lt_id, data_type::f32);
            dequant2.add_input(s8_weight);
            dequant2.add_output(fp32_weight);

            logical_tensor_t fp32_bias;
            if (with_bias)
                fp32_bias = logical_tensor_init(++lt_id, data_type::f32);

            logical_tensor_t fp32_matmul_out
                    = logical_tensor_init(++lt_id, data_type::f32);
            matmul.add_input(fp32_data);
            matmul.add_input(fp32_weight);
            if (with_bias) matmul.add_input(fp32_bias);
            matmul.add_output(fp32_matmul_out);

            logical_tensor_t fp32_binary_other
                    = logical_tensor_init(++lt_id, data_type::f32);
            logical_tensor_t fp32_binary_out
                    = logical_tensor_init(++lt_id, data_type::f32);
            binary.add_input(fp32_matmul_out);
            binary.add_input(fp32_binary_other);
            binary.add_output(fp32_binary_out);

            logical_tensor_t int8_out
                    = logical_tensor_init(++lt_id, data_type::u8);
            quant.add_input(fp32_binary_out);
            quant.add_output(int8_out);

            ASSERT_EQ(agraph.add_op(&dequant1), status::success);
            ASSERT_EQ(agraph.add_op(&dequant2), status::success);
            ASSERT_EQ(agraph.add_op(&matmul), status::success);
            ASSERT_EQ(agraph.add_op(&binary), status::success);
            ASSERT_EQ(agraph.add_op(&quant), status::success);

            agraph.finalize();

            pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops_cpu");
            apass->run(agraph);
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::quantized_matmul_post_ops);

            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(),
                    with_bias ? 4U : 3U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
            if (with_bias) {
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 6U);
            } else {
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 5U);
            }

            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                    with_bias ? 8U : 7U);
        }
    }
}

TEST(Pass, FailToFuseToInt8MatMulDivOrSubtract) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
          matmul
        \    / (f32)
         div/sub
             | (f32)
           quant
             | (u8/s8)
    */
    const std::vector<graph::op_kind_t> binary_kinds = {Divide, Subtract};

    for (auto &binary_kind : binary_kinds) {
        graph_t agraph;
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr(op_attr::scales, scales);
        dequant1.set_attr(op_attr::zps, zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales);
        dequant2.set_attr(op_attr::zps, zps);
        op_t matmul {2, MatMul, "matmul"};
        op_t binary {3, binary_kind, "binary"};
        op_t quant {4, Quantize, "quant"};
        quant.set_attr(op_attr::scales, scales);
        quant.set_attr(op_attr::zps, zps);

        int lt_id = -1;
        logical_tensor_t int8_data
                = logical_tensor_init(++lt_id, data_type::u8);
        logical_tensor_t fp32_data
                = logical_tensor_init(++lt_id, data_type::f32);
        dequant1.add_input(int8_data);
        dequant1.add_output(fp32_data);

        logical_tensor_t s8_weight
                = logical_tensor_init(++lt_id, data_type::s8);
        logical_tensor_t fp32_weight
                = logical_tensor_init(++lt_id, data_type::f32);
        dequant2.add_input(s8_weight);
        dequant2.add_output(fp32_weight);

        logical_tensor_t fp32_matmul_out
                = logical_tensor_init(++lt_id, data_type::f32);
        matmul.add_input(fp32_data);
        matmul.add_input(fp32_weight);
        matmul.add_output(fp32_matmul_out);

        logical_tensor_t fp32_binary_other
                = logical_tensor_init(++lt_id, data_type::f32);
        logical_tensor_t fp32_binary_out
                = logical_tensor_init(++lt_id, data_type::f32);
        // divide and subtract are not commutative
        binary.add_input(fp32_binary_other);
        binary.add_input(fp32_matmul_out);
        binary.add_output(fp32_binary_out);

        logical_tensor_t int8_out = logical_tensor_init(++lt_id, data_type::u8);
        quant.add_input(fp32_binary_out);
        quant.add_output(int8_out);

        ASSERT_EQ(agraph.add_op(&dequant1), status::success);
        ASSERT_EQ(agraph.add_op(&dequant2), status::success);
        ASSERT_EQ(agraph.add_op(&matmul), status::success);
        ASSERT_EQ(agraph.add_op(&binary), status::success);
        ASSERT_EQ(agraph.add_op(&quant), status::success);

        agraph.finalize();

        pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops_cpu");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 3U);
    }
}

TEST(PassSystem, FuseToInt8MatMulSwishReLU) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            matmul
      (f32) / |
     sigmoid  |
           \  |
          multiply     
             |
            relu  
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    op_t sigmoid {3, Sigmoid, "sigmoid"};
    op_t multiply {4, Multiply, "mul"};
    op_t relu {5, ReLU, "relu"};
    op_t quant {6, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t fp32_sigmoid_out = logical_tensor_init(5, data_type::f32);
    sigmoid.add_input(fp32_matmul_out);
    sigmoid.add_output(fp32_sigmoid_out);

    logical_tensor_t fp32_mul_out = logical_tensor_init(6, data_type::f32);
    multiply.add_input(fp32_matmul_out);
    multiply.add_input(fp32_sigmoid_out);
    multiply.add_output(fp32_mul_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(7, data_type::f32);
    relu.add_input(fp32_mul_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(8, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    // run all the pass to check if the priority is correct
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 7U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8U);
}

TEST(Pass, FuseToInt8MatmulBias) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    op_t quant {3, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(PassSystem, TestInt8MatmulBias) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    op_t quant {3, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FuseToInt8MatmulRelu) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "conv"};
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(PassSystem, FuseToInt8MatmulRelu) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "conv"};
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(Pass, FuseToInt8MatmulBiasRelu) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(Pass, FuseToX8s8f32Matmul) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseToX8s8f32MatmulBias) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseToX8s8f32MatmulEltwise) {
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
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, partition_kind_t>> opkind_pair {
            {ReLU, partition_kind_t::quantized_matmul_post_ops},
            {Sigmoid, partition_kind_t::quantized_matmul_post_ops},
            {GELU, partition_kind_t::quantized_matmul_post_ops}};
    for (auto &p : opkind_pair) {
        graph_t agraph;
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr(op_attr::scales, scales);
        dequant1.set_attr(op_attr::zps, zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales);
        dequant2.set_attr(op_attr::zps, zps);
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

        agraph.finalize();

        pm.run_passes(agraph, "no_config");
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(), p.second);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
    }
}

TEST(Pass, FuseToX8s8f32MatmulBiasEltwise) {
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
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    std::vector<std::pair<op_kind_t, partition_kind_t>> opkind_pair {
            {ReLU, partition_kind_t::quantized_matmul_post_ops},
            {Sigmoid, partition_kind_t::quantized_matmul_post_ops},
            {GELU, partition_kind_t::quantized_matmul_post_ops}};
    for (auto &p : opkind_pair) {
        graph_t agraph;
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr(op_attr::scales, scales);
        dequant1.set_attr(op_attr::zps, zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales);
        dequant2.set_attr(op_attr::zps, zps);
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

        agraph.finalize();

        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(), p.second);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
    }
}

TEST(Pass, FuseToInt8Maxpool) {
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
    dequant.set_attr(op_attr::scales, scales);
    dequant.set_attr(op_attr::zps, zps);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0};
    std::vector<int64_t> pads_end = {0, 0};
    std::vector<int64_t> kernel = {2, 2};
    op_t maxpool {1, MaxPool, "maxpool"};
    maxpool.set_attr(op_attr::strides, strides);
    maxpool.set_attr(op_attr::pads_begin, pads_begin);
    maxpool.set_attr(op_attr::pads_end, pads_end);
    maxpool.set_attr(op_attr::kernel, kernel);

    op_t quant {2, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8_pool_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_pooling_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(PassSystem, TestInt8Maxpool) {
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
    dequant.set_attr(op_attr::scales, scales);
    dequant.set_attr(op_attr::zps, zps);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0};
    std::vector<int64_t> pads_end = {0, 0};
    std::vector<int64_t> kernel = {2, 2};
    op_t maxpool {1, MaxPool, "maxpool"};
    maxpool.set_attr(op_attr::strides, strides);
    maxpool.set_attr(op_attr::pads_begin, pads_begin);
    maxpool.set_attr(op_attr::pads_end, pads_end);
    maxpool.set_attr(op_attr::kernel, kernel);

    op_t quant {2, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_pooling_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseToInt8Avgpool) {
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
    dequant.set_attr(op_attr::scales, scales);
    dequant.set_attr(op_attr::zps, zps);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0};
    std::vector<int64_t> pads_end = {0, 0};
    std::vector<int64_t> kernel = {2, 2};
    op_t avgpool {1, AvgPool, "avgpool"};
    avgpool.set_attr(op_attr::strides, strides);
    avgpool.set_attr(op_attr::pads_begin, pads_begin);
    avgpool.set_attr(op_attr::pads_end, pads_end);
    avgpool.set_attr(op_attr::kernel, kernel);
    avgpool.set_attr<bool>(op_attr::exclude_pad, false);

    op_t quant {2, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8_pool_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_pooling_post_ops);
}

TEST(PassSystem, FuseToInt8PoolAdd) {
    /*    
             | (u8/s8)
          dequant
             | (f32)
           pool
             | (f32)
             |     | (u8/s8)
             |   dequant
             |  / (f32)
            add
             | (f32)
           quant
             | (u8/s8)
    */
    std::vector<op_kind_t> pool_kinds = {AvgPool, MaxPool};
    std::vector<engine_kind_t> engine_kinds
            = {engine_kind::cpu, engine_kind::gpu};

    for_(const auto &engine_kind : engine_kinds)
    for (const auto &pool_kind : pool_kinds) {
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant {0, Dequantize, "dequant"};
        dequant.set_attr(op_attr::scales, scales);
        dequant.set_attr(op_attr::zps, zps);

        std::vector<int64_t> zps2 = {1};
        std::vector<float> scales2 = {3.3f};
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales2);
        dequant2.set_attr(op_attr::zps, zps2);

        std::vector<int64_t> strides = {2, 2};
        std::vector<int64_t> pads_begin = {0, 0};
        std::vector<int64_t> pads_end = {0, 0};
        std::vector<int64_t> kernel = {2, 2};
        op_t pool {2, pool_kind, "pool"};
        pool.set_attr(op_attr::strides, strides);
        pool.set_attr(op_attr::pads_begin, pads_begin);
        pool.set_attr(op_attr::pads_end, pads_end);
        pool.set_attr(op_attr::kernel, kernel);
        if (pool_kind == AvgPool)
            pool.set_attr<bool>(op_attr::exclude_pad, false);

        op_t add {3, Add, "add"};

        op_t quant {4, Quantize, "quant"};
        quant.set_attr(op_attr::scales, scales);
        quant.set_attr(op_attr::zps, zps);

        logical_tensor_t int8_data
                = logical_tensor_init(0, {1, 1, 4, 4}, data_type::u8);
        logical_tensor_t fp32_data
                = logical_tensor_init(1, {1, 1, 4, 4}, data_type::f32);
        logical_tensor_t int8_other
                = logical_tensor_init(2, {1, 1, 1, 1}, data_type::u8);
        logical_tensor_t fp32_other
                = logical_tensor_init(3, {1, 1, 1, 1}, data_type::f32);
        logical_tensor_t fp32_pool_out
                = logical_tensor_init(4, {1, 1, 2, 2}, data_type::f32);
        logical_tensor_t fp32_add_out
                = logical_tensor_init(5, {1, 1, 2, 2}, data_type::f32);
        logical_tensor_t int8_out
                = logical_tensor_init(6, {1, 1, 2, 2}, data_type::u8);

        dequant.add_input(int8_data);
        dequant.add_output(fp32_data);

        pool.add_input(fp32_data);
        pool.add_output(fp32_pool_out);

        dequant2.add_input(int8_other);
        dequant2.add_output(fp32_other);

        add.add_input(fp32_pool_out);
        add.add_input(fp32_other);
        add.add_output(fp32_add_out);

        quant.add_input(fp32_add_out);
        quant.add_output(int8_out);

        graph_t agraph(engine_kind);
        ASSERT_EQ(agraph.add_op(&dequant), status::success);
        ASSERT_EQ(agraph.add_op(&dequant2), status::success);
        ASSERT_EQ(agraph.add_op(&pool), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        ASSERT_EQ(agraph.add_op(&quant), status::success);
        agraph.finalize();

        auto &backend_ptr
                = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
        auto pm = dnnl::impl::graph::pass::pass_manager_t(
                backend_ptr.get_pass_registry());

        pm.run_passes(agraph, "no_config");

        if (engine_kind == engine_kind::cpu) {
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::quantized_pooling_post_ops);
        } else {
            ASSERT_EQ(agraph.get_num_partitions(), 2U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::quantized_pooling_post_ops);
            ASSERT_EQ((agraph.get_partitions()[1])->get_kind(),
                    partition_kind_t::misc_post_ops);
        }
    }
}

TEST(PassSystem, Quantize) {
    std::vector<engine_kind_t> engine_kinds
            = {engine_kind::cpu, engine_kind::gpu};
    for (const auto &engine_kind : engine_kinds) {
        graph_t agraph(engine_kind);
        std::vector<int64_t> zps = {1};
        std::vector<float> scales = {3.1f};
        op_t quant {0, Quantize, "quant"};
        quant.set_attr(op_attr::scales, scales);
        quant.set_attr(op_attr::zps, zps);

        logical_tensor_t in = logical_tensor_init(0, data_type::f32);
        logical_tensor_t out = logical_tensor_init(1, data_type::s8);

        quant.add_input(in);
        quant.add_output(out);
        ASSERT_EQ(agraph.add_op(&quant), status::success);
        ASSERT_EQ(agraph.finalize(), status::success);

        auto &backend_ptr
                = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
        auto pm = dnnl::impl::graph::pass::pass_manager_t(
                backend_ptr.get_pass_registry());
        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1U);
    }
}

TEST(Pass, FuseToInt8MatmulAdd) {
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
    graph_t agraph;
    std::vector<int64_t> zps {0};
    std::vector<float> scales {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr(op_attr::scales, scales);
    dequant3.set_attr(op_attr::zps, zps);
    op_t matmul {3, MatMul, "matmul"};
    matmul.set_attr<bool>(op_attr::transpose_a, false);
    matmul.set_attr<bool>(op_attr::transpose_b, false);
    op_t add {4, Add, "add"};
    op_t quant {5, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t fp32_matmul_out = logical_tensor_init(7, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x8_matmul_add_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9U);
}

TEST(Pass, FuseToInt8MatmulBiasAdd) {
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
    std::vector<int64_t> zps {0};
    std::vector<float> scales {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t dequant3 {2, Dequantize, "dequant"};
    dequant3.set_attr(op_attr::scales, scales);
    dequant3.set_attr(op_attr::zps, zps);
    op_t matmul {3, MatMul, "matmul"};
    matmul.set_attr<bool>(op_attr::transpose_a, false);
    matmul.set_attr<bool>(op_attr::transpose_b, false);
    op_t add {4, Add, "add"};
    op_t quant {5, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x8_matmul_add_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9U);
}

TEST(PassSystem, FuseReluAdd) {
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
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::unary_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseToX8x8f32MatmulDivAdd) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
    (f32) \     / (f32)
           matmul
             | (f32)
            div
             | (f32)
            add
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    op_t div {3, Divide, "divide"};
    op_t add {4, Add, "add"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t int8_weight = logical_tensor_init(2, data_type::u8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(int8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t f32_matmul_out = logical_tensor_init(4, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_output(f32_matmul_out);

    logical_tensor_t f32_div_in = logical_tensor_init(5, data_type::f32);
    logical_tensor_t f32_div_out = logical_tensor_init(6, data_type::f32);
    div.add_input(f32_matmul_out);
    div.add_input(f32_div_in);
    div.add_output(f32_div_out);

    logical_tensor_t f32_add_in = logical_tensor_init(7, data_type::f32);
    logical_tensor_t f32_add_out = logical_tensor_init(8, data_type::f32);
    add.add_input(f32_div_out);
    add.add_input(f32_add_in);
    add.add_output(f32_add_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 7U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8U);
}

TEST(PassSystem, FuseToX8x8f32MatmulDivAdd) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
    (f32) \     / (f32)
           matmul
             | (f32)
            div
             | (f32)
            add
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    op_t div {3, Divide, "divide"};
    op_t add {4, Add, "add"};

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t int8_weight = logical_tensor_init(2, data_type::u8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(int8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t f32_matmul_out = logical_tensor_init(4, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_output(f32_matmul_out);

    logical_tensor_t f32_div_in = logical_tensor_init(5, data_type::f32);
    logical_tensor_t f32_div_out = logical_tensor_init(6, data_type::f32);
    div.add_input(f32_matmul_out);
    div.add_input(f32_div_in);
    div.add_output(f32_div_out);

    logical_tensor_t f32_add_in = logical_tensor_init(7, data_type::f32);
    logical_tensor_t f32_add_out = logical_tensor_init(8, data_type::f32);
    add.add_input(f32_div_out);
    add.add_input(f32_add_in);
    add.add_output(f32_add_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
}

TEST(Pass, FuseToX8s8bf16Matmul) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
           matmul
             | (bf16)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(PassSystem, FuseToX8s8bf16Matmul) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
           matmul
             | (bf16)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
}

TEST(Pass, FuseToX8s8bf16MatmulDiv) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 7U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8U);
}

TEST(PassSystem, FuseToX8s8bf16MatmulDiv) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
}

TEST(Pass, FailToAddMatmul) {
    /*
    (bf16) \     / (f16)
           matmul
             | (bf16)
    */
    graph_t agraph;
    op_t matmul {0, MatMul, "matmul"};

    logical_tensor_t bf16_data = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t f16_data = logical_tensor_init(1, data_type::f16);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(2, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(f16_data);
    matmul.add_output(bf16_matmul_out);

    ASSERT_EQ(agraph.add_op(&matmul), status::invalid_graph_op);
}

TEST(Pass, FuseToX8s8bf16MatmulScaleAdd) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
           matmul
             | (bf16)
          div/mul
             | (bf16)
            add
             | (bf16)
    */
    std::vector<op_kind_t> scale_kinds {Multiply, Divide};
    for (auto scale_kind : scale_kinds) {
        graph_t agraph;
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr(op_attr::scales, scales);
        dequant1.set_attr(op_attr::zps, zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales);
        dequant2.set_attr(op_attr::zps, zps);
        op_t typecast1 {2, TypeCast, "typecast"};
        op_t typecast2 {3, TypeCast, "typecast"};
        op_t matmul {4, MatMul, "matmul"};
        op_t scale {5, scale_kind, "scale"};
        op_t add {6, Add, "add"};

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

        logical_tensor_t bf16_matmul_out
                = logical_tensor_init(6, data_type::bf16);
        matmul.add_input(bf16_data);
        matmul.add_input(bf16_weight);
        matmul.add_output(bf16_matmul_out);

        logical_tensor_t bf16_scale_in
                = logical_tensor_init(7, data_type::bf16);
        logical_tensor_t bf16_scale_out
                = logical_tensor_init(8, data_type::bf16);
        scale.add_input(bf16_matmul_out);
        scale.add_input(bf16_scale_in);
        scale.add_output(bf16_scale_out);

        logical_tensor_t bf16_add_in = logical_tensor_init(9, data_type::bf16);
        logical_tensor_t bf16_add_out
                = logical_tensor_init(10, data_type::bf16);
        add.add_input(bf16_scale_out);
        add.add_input(bf16_add_in);
        add.add_output(bf16_add_out);

        ASSERT_EQ(agraph.add_op(&dequant1), status::success);
        ASSERT_EQ(agraph.add_op(&dequant2), status::success);
        ASSERT_EQ(agraph.add_op(&matmul), status::success);
        ASSERT_EQ(agraph.add_op(&typecast1), status::success);
        ASSERT_EQ(agraph.add_op(&typecast2), status::success);
        ASSERT_EQ(agraph.add_op(&scale), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);

        agraph.finalize();

        pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops_cpu");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::quantized_matmul_post_ops);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 7U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 9U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
    }
}

TEST(PassSystem, FuseToX8s8bf16MatmulScaleAdd) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
           matmul
             | (bf16)
           div/mul
             | (bf16)
            add
             | (bf16)
    */
    std::vector<op_kind_t> scale_kinds {Multiply, Divide};
    for (auto scale_kind : scale_kinds) {
        graph_t agraph;
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr(op_attr::scales, scales);
        dequant1.set_attr(op_attr::zps, zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales);
        dequant2.set_attr(op_attr::zps, zps);
        op_t typecast1 {2, TypeCast, "typecast"};
        op_t typecast2 {3, TypeCast, "typecast"};
        op_t matmul {4, MatMul, "matmul"};
        op_t scale {5, scale_kind, "scale"};
        op_t add {6, Add, "add"};

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

        logical_tensor_t bf16_matmul_out
                = logical_tensor_init(6, data_type::bf16);
        matmul.add_input(bf16_data);
        matmul.add_input(bf16_weight);
        matmul.add_output(bf16_matmul_out);

        logical_tensor_t bf16_scale_in
                = logical_tensor_init(7, data_type::bf16);
        logical_tensor_t bf16_scale_out
                = logical_tensor_init(8, data_type::bf16);
        scale.add_input(bf16_matmul_out);
        scale.add_input(bf16_scale_in);
        scale.add_output(bf16_scale_out);

        logical_tensor_t bf16_add_in = logical_tensor_init(9, data_type::bf16);
        logical_tensor_t bf16_add_out
                = logical_tensor_init(10, data_type::bf16);
        add.add_input(bf16_scale_out);
        add.add_input(bf16_add_in);
        add.add_output(bf16_add_out);

        ASSERT_EQ(agraph.add_op(&dequant1), status::success);
        ASSERT_EQ(agraph.add_op(&dequant2), status::success);
        ASSERT_EQ(agraph.add_op(&matmul), status::success);
        ASSERT_EQ(agraph.add_op(&typecast1), status::success);
        ASSERT_EQ(agraph.add_op(&typecast2), status::success);
        ASSERT_EQ(agraph.add_op(&scale), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);

        agraph.finalize();

        auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
        auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
        pm.run_passes(agraph, "no_config");
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::quantized_matmul_post_ops);
    }
}

TEST(Pass, FuseToX8s8bf16MatmulBias) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    logical_tensor_t bf16_bias = logical_tensor_init(6, data_type::bf16);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(7, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(bf16_bias);
    matmul.add_output(bf16_matmul_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(PassSystem, FuseToX8s8bf16MatmulBias) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
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

    logical_tensor_t bf16_bias = logical_tensor_init(6, data_type::bf16);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(7, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(bf16_bias);
    matmul.add_output(bf16_matmul_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
}

TEST(Pass, FuseSingleTypecast) {
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

    ASSERT_EQ(agraph.finalize(), status::success);

    pass::pass_base_ptr apass = get_pass("typecast_pass");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
}

TEST(Pass, FuseToX8s8bf16MatmulBiasAddBF16) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16) / (f32/bf16)
            matmul_with_bias
             | (bf16)
             |
             |  / (bf16)
            add
             | (bf16)
    */
    graph_t agraph;
    std::vector<int64_t> zps {0};
    std::vector<float> scales {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    matmul.set_attr<bool>(op_attr::transpose_a, false);
    matmul.set_attr<bool>(op_attr::transpose_b, false);
    op_t add {5, Add, "add"};

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

    logical_tensor_t bf16_other = logical_tensor_init(6, data_type::bf16);
    logical_tensor_t bf16_bias = logical_tensor_init(7, data_type::bf16);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(8, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(bf16_bias);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t bf16_add_out = logical_tensor_init(11, data_type::bf16);
    add.add_input(bf16_matmul_out);
    add.add_input(bf16_other);
    add.add_output(bf16_add_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops_cpu");
    ASSERT_TRUE(apass);
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 7U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 11U);
}

TEST(PassSystem, FuseToX8s8bf16MatmulBiasAddBF16) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16) / (f32/bf16)
            matmul_with_bias
             | (bf16)
             |
             |  / (bf16)
            add
             | (bf16)
    */
    graph_t agraph;
    std::vector<int64_t> zps {0};
    std::vector<float> scales {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    matmul.set_attr<bool>(op_attr::transpose_a, false);
    matmul.set_attr<bool>(op_attr::transpose_b, false);
    op_t add {5, Add, "add"};

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

    logical_tensor_t bf16_other = logical_tensor_init(6, data_type::bf16);
    logical_tensor_t bf16_bias = logical_tensor_init(7, data_type::bf16);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(8, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(bf16_bias);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t bf16_add_out = logical_tensor_init(11, data_type::bf16);
    add.add_input(bf16_matmul_out);
    add.add_input(bf16_other);
    add.add_output(bf16_add_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
}

TEST(Pass, MixInt8AndBf16MatmulBiasGelu) {
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
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t eltwise {5, GELU, "gelu"};
    op_t typecast3 {6, TypeCast, "typecast"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t bf16_bias = logical_tensor_init(6, data_type::bf16);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(7, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(bf16_bias);
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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops_cpu");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(PassSystem, MixInt8AndBf16MatmulBiasGelu) {
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
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t eltwise {5, GELU, "gelu"};
    op_t typecast3 {6, TypeCast, "typecast"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t bf16_bias = logical_tensor_init(6, data_type::bf16);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(7, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(bf16_bias);
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

    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
}

TEST(Pass, MixInt8AndBf16MatmulGelu) {
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
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t eltwise {5, GELU, "gelu"};
    op_t typecast3 {6, TypeCast, "typecast"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops_cpu");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(PassSystem, MixInt8AndBf16MatmulGelu) {
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
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t eltwise {5, GELU, "gelu"};
    op_t typecast3 {6, TypeCast, "typecast"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 8U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(Pass, MixInt8AndBf16MatmulBias) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
        matmul_with_bias
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t typecast3 {5, TypeCast, "typecast"};
    op_t quant {6, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t bf16_bias = logical_tensor_init(6, data_type::bf16);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(7, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(bf16_bias);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t fp32_matmul_out = logical_tensor_init(9, data_type::f32);
    typecast3.add_input(bf16_matmul_out);
    typecast3.add_output(fp32_matmul_out);

    logical_tensor_t int8_dst = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_matmul_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops_cpu");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(PassSystem, MixInt8AndBf16MatmulBias) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
        matmul_with_bias
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t typecast3 {5, TypeCast, "typecast"};
    op_t quant {6, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t bf16_bias = logical_tensor_init(6, data_type::bf16);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(7, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(bf16_bias);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t fp32_matmul_out = logical_tensor_init(9, data_type::f32);
    typecast3.add_input(bf16_matmul_out);
    typecast3.add_output(fp32_matmul_out);

    logical_tensor_t int8_dst = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_matmul_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(Pass, MixInt8AndBf16Matmul) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
           matmul
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t typecast3 {5, TypeCast, "typecast"};
    op_t quant {6, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t fp32_matmul_out = logical_tensor_init(9, data_type::f32);
    typecast3.add_input(bf16_matmul_out);
    typecast3.add_output(fp32_matmul_out);

    logical_tensor_t int8_dst = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_matmul_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops_cpu");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(PassSystem, MixInt8AndBf16Matmul) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
           matmul
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t typecast3 {5, TypeCast, "typecast"};
    op_t quant {6, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t fp32_matmul_out = logical_tensor_init(9, data_type::f32);
    typecast3.add_input(bf16_matmul_out);
    typecast3.add_output(fp32_matmul_out);

    logical_tensor_t int8_dst = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_matmul_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 7U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(PassSystem, QuantWeiMixBf16MatmulBiasTransposeReshapeQuantize) {
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    std::vector<bool> with_bias_typecasts {false, true};
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    std::vector<int64_t> transpose_order {0, 2, 1, 3};
    std::vector<int64_t> reshape_shape {0, 0, 0, 0};
    for (auto with_bias_typecast : with_bias_typecasts) {
        graph_t agraph;
        op_t dqdata_op(1, Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::vector<int64_t>>(op_attr::zps, zps);
        dqdata_op.set_attr<std::vector<float>>(op_attr::scales, scales);
        op_t tcdata_op(2, TypeCast, "tcdata_op");
        op_t qweight_op(3, Quantize, "qweight_op");
        qweight_op.set_attr<std::vector<int64_t>>(op_attr::zps, zps);
        qweight_op.set_attr<std::vector<float>>(op_attr::scales, scales);
        op_t dqweight_op(4, Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::vector<int64_t>>(op_attr::zps, zps);
        dqweight_op.set_attr<std::vector<float>>(op_attr::scales, scales);
        op_t tcweight_op(5, TypeCast, "tcweight_op");
        op_t matmul_op(6, MatMul, "matmul_op");
        matmul_op.set_attr<bool>(op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(op_attr::transpose_b, false);
        op_t tc_bias_op(7, TypeCast, "tcbias_op");
        op_t biasadd_op(8, BiasAdd, "biasadd_op");
        biasadd_op.set_attr<std::string>(op_attr::data_format, "NXC");
        op_t transpose_op(9, StaticTranspose, "transpose_op");
        transpose_op.set_attr(op_attr::order, transpose_order);
        op_t reshape_op(10, StaticReshape, "reshape_op");
        reshape_op.set_attr(op_attr::shape, reshape_shape);
        reshape_op.set_attr(op_attr::special_zero, false);
        op_t tcout_op(11, TypeCast, "tcout_op");
        op_t qout_op(12, Quantize, "qout_op");
        qout_op.set_attr<std::vector<int64_t>>(op_attr::zps, zps);
        qout_op.set_attr<std::vector<float>>(op_attr::scales, scales);

        // prepare logical tensor
        logical_tensor_t src_u8 = logical_tensor_init(1, data_type::u8);
        logical_tensor_t src_f32_dq = logical_tensor_init(2, data_type::f32);
        logical_tensor_t src_bf16_dq = logical_tensor_init(3, data_type::bf16);
        logical_tensor_t weight_f32 = logical_tensor_init(4, data_type::f32);
        logical_tensor_t weight_s8 = logical_tensor_init(5, data_type::s8);
        logical_tensor_t weight_f32_dq = logical_tensor_init(6, data_type::f32);
        logical_tensor_t weight_bf16_dq
                = logical_tensor_init(7, data_type::bf16);
        logical_tensor_t bias_f32 = logical_tensor_init(8, data_type::f32);
        logical_tensor_t bias_bf16 = logical_tensor_init(9, data_type::bf16);
        logical_tensor_t biasadd_bf16
                = logical_tensor_init(10, data_type::bf16);
        logical_tensor_t dst_bf16 = logical_tensor_init(11, data_type::bf16);
        logical_tensor_t transpose_bf16
                = logical_tensor_init(12, data_type::bf16);
        logical_tensor_t reshape_bf16
                = logical_tensor_init(13, data_type::bf16);
        logical_tensor_t reshape_f32 = logical_tensor_init(14, data_type::f32);
        logical_tensor_t dst_s8 = logical_tensor_init(15, data_type::s8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        tcdata_op.add_input(src_f32_dq);
        tcdata_op.add_output(src_bf16_dq);

        qweight_op.add_input(weight_f32);
        qweight_op.add_output(weight_s8);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        tcweight_op.add_input(weight_f32_dq);
        tcweight_op.add_output(weight_bf16_dq);

        matmul_op.add_input(src_bf16_dq);
        matmul_op.add_input(weight_bf16_dq);
        matmul_op.add_output(dst_bf16);

        tc_bias_op.add_input(bias_f32);
        tc_bias_op.add_output(bias_bf16);

        biasadd_op.add_input(dst_bf16);
        biasadd_op.add_input(bias_bf16);
        biasadd_op.add_output(biasadd_bf16);

        transpose_op.add_input(biasadd_bf16);
        transpose_op.add_output(transpose_bf16);

        reshape_op.add_input(transpose_bf16);
        reshape_op.add_output(reshape_bf16);

        tcout_op.add_input(reshape_bf16);
        tcout_op.add_output(reshape_f32);

        qout_op.add_input(reshape_f32);
        qout_op.add_output(dst_s8);
        ASSERT_EQ(agraph.add_op(&dqdata_op), status::success);
        ASSERT_EQ(agraph.add_op(&tcdata_op), status::success);
        ASSERT_EQ(agraph.add_op(&qweight_op), status::success);
        ASSERT_EQ(agraph.add_op(&dqweight_op), status::success);
        ASSERT_EQ(agraph.add_op(&tcweight_op), status::success);
        ASSERT_EQ(agraph.add_op(&matmul_op), status::success);
        if (with_bias_typecast) {
            ASSERT_EQ(agraph.add_op(&tc_bias_op), status::success);
        }
        ASSERT_EQ(agraph.add_op(&biasadd_op), status::success);
        ASSERT_EQ(agraph.add_op(&transpose_op), status::success);
        ASSERT_EQ(agraph.add_op(&reshape_op), status::success);
        ASSERT_EQ(agraph.add_op(&tcout_op), status::success);
        ASSERT_EQ(agraph.add_op(&qout_op), status::success);
        ASSERT_EQ(agraph.finalize(), status::success);

        pm.run_passes(agraph, "no_config");
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        auto part = agraph.get_partitions()[0];

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 4U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id,
                with_bias_typecast ? 8U : 9U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 15U);
    }
}

TEST(Pass, MixInt8AndBf16ConvolutionBias) {
    /*
        | (u8/s8)  | s8
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
     convolution_with_bias
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t convolution {4, Convolution, "convolution"};
    set_conv_common_attr(convolution);
    op_t typecast3 {5, TypeCast, "typecast"};
    op_t quant {6, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t bf16_bias = logical_tensor_init(6, data_type::bf16);
    logical_tensor_t bf16_convolution_out
            = logical_tensor_init(7, data_type::bf16);
    convolution.add_input(bf16_data);
    convolution.add_input(bf16_weight);
    convolution.add_input(bf16_bias);
    convolution.add_output(bf16_convolution_out);

    logical_tensor_t fp32_convolution_out
            = logical_tensor_init(9, data_type::f32);
    typecast3.add_input(bf16_convolution_out);
    typecast3.add_output(fp32_convolution_out);

    logical_tensor_t int8_dst = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_convolution_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&convolution), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_tc_conv_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(PassSystem, MixInt8AndBf16ConvolutionBias) {
    /*
        | (u8/s8)  | s8
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
        convolution_with_bias
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t convolution {4, Convolution, "convolution"};
    set_conv_common_attr(convolution);
    op_t typecast3 {5, TypeCast, "typecast"};
    op_t quant {6, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t bf16_bias = logical_tensor_init(6, data_type::bf16);
    logical_tensor_t bf16_convolution_out
            = logical_tensor_init(7, data_type::bf16);
    convolution.add_input(bf16_data);
    convolution.add_input(bf16_weight);
    convolution.add_input(bf16_bias);
    convolution.add_output(bf16_convolution_out);

    logical_tensor_t fp32_convolution_out
            = logical_tensor_init(9, data_type::f32);
    typecast3.add_input(bf16_convolution_out);
    typecast3.add_output(fp32_convolution_out);

    logical_tensor_t int8_dst = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_convolution_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&convolution), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 7U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 10U);
}

TEST(Pass, FailToFuseMixInt8AndBf16ConvolutionWithoutQuantAfterTypecast) {
    /*
        | (u8/s8)  | s8
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
        convolution_with_bias
             | (bf16)
            Elu
             | (bf16)
           typecast
             | (f32)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    float alpha_value = 1.0f;
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t convolution {4, Convolution, "convolution"};
    set_conv_common_attr(convolution);
    op_t elu {5, Elu, "elu"};
    elu.set_attr(op_attr::alpha, alpha_value);
    op_t typecast3 {6, TypeCast, "typecast"};

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

    logical_tensor_t bf16_bias = logical_tensor_init(10, data_type::bf16);
    logical_tensor_t bf16_convolution_out
            = logical_tensor_init(6, data_type::bf16);
    convolution.add_input(bf16_data);
    convolution.add_input(bf16_weight);
    convolution.add_input(bf16_bias);
    convolution.add_output(bf16_convolution_out);

    logical_tensor_t bf16_elu_out = logical_tensor_init(7, data_type::bf16);
    elu.add_input(bf16_convolution_out);
    elu.add_output(bf16_elu_out);

    logical_tensor_t fp32_typecast3_out
            = logical_tensor_init(8, data_type::f32);
    typecast3.add_input(bf16_elu_out);
    typecast3.add_output(fp32_typecast3_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&convolution), status::success);
    ASSERT_EQ(agraph.add_op(&elu), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_tc_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 7U);
}

TEST(Pass, MixInt8AndBf16ConvolutionBiasGelu) {
    /*
        | (u8/s8)  | s8
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
        convolution_with_bias
             | (bf16)
            GeLU
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t convolution {4, Convolution, "convolution"};
    set_conv_common_attr(convolution);
    op_t gelu {5, GELU, "gelu"};
    op_t typecast3 {6, TypeCast, "typecast"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t bf16_bias = logical_tensor_init(10, data_type::bf16);
    logical_tensor_t bf16_convolution_out
            = logical_tensor_init(6, data_type::bf16);
    convolution.add_input(bf16_data);
    convolution.add_input(bf16_weight);
    convolution.add_input(bf16_bias);
    convolution.add_output(bf16_convolution_out);

    logical_tensor_t bf16_gelu_out = logical_tensor_init(7, data_type::bf16);
    gelu.add_input(bf16_convolution_out);
    gelu.add_output(bf16_gelu_out);

    logical_tensor_t fp32_typecast3_out
            = logical_tensor_init(8, data_type::f32);
    typecast3.add_input(bf16_gelu_out);
    typecast3.add_output(fp32_typecast3_out);

    logical_tensor_t int8_dst = logical_tensor_init(9, data_type::u8);
    quant.add_input(fp32_typecast3_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&convolution), status::success);
    ASSERT_EQ(agraph.add_op(&gelu), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("x8s8x_tc_conv_post_ops");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 10U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9U);
}

TEST(PassSystem, MixInt8AndBf16ConvolutionBiasGelu) {
    /*
        | (u8/s8)  | s8
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
        convolution_with_bias
             | (bf16)
            GeLU
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t convolution {4, Convolution, "convolution"};
    set_conv_common_attr(convolution);
    op_t gelu {5, GELU, "gelu"};
    op_t typecast3 {6, TypeCast, "typecast"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t bf16_bias = logical_tensor_init(10, data_type::bf16);
    logical_tensor_t bf16_convolution_out
            = logical_tensor_init(6, data_type::bf16);
    convolution.add_input(bf16_data);
    convolution.add_input(bf16_weight);
    convolution.add_input(bf16_bias);
    convolution.add_output(bf16_convolution_out);

    logical_tensor_t bf16_gelu_out = logical_tensor_init(7, data_type::bf16);
    gelu.add_input(bf16_convolution_out);
    gelu.add_output(bf16_gelu_out);

    logical_tensor_t fp32_convolution_out
            = logical_tensor_init(8, data_type::f32);
    typecast3.add_input(bf16_gelu_out);
    typecast3.add_output(fp32_convolution_out);

    logical_tensor_t int8_dst = logical_tensor_init(9, data_type::u8);
    quant.add_input(fp32_convolution_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&convolution), status::success);
    ASSERT_EQ(agraph.add_op(&gelu), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 10U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9U);
}

TEST(PassSystem, MixInt8AndBf16ConvolutionAdd) {
    /*
        | (u8/s8)  | s8
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
        convolution
      (bf16) | typecast
             | / (bf16)
            add
             | (bf16)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t convolution {4, Convolution, "convolution"};
    set_conv_common_attr(convolution);
    op_t typecast3 {5, TypeCast, "typecast"};
    op_t add {6, Add, "add"};

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
    logical_tensor_t bf16_bias = logical_tensor_init(7, data_type::bf16);
    typecast3.add_input(fp32_bias);
    typecast3.add_output(bf16_bias);

    logical_tensor_t bf16_convolution_out
            = logical_tensor_init(8, data_type::bf16);
    convolution.add_input(bf16_data);
    convolution.add_input(bf16_weight);
    convolution.add_output(bf16_convolution_out);

    logical_tensor_t bf16_bia_outs = logical_tensor_init(9, data_type::bf16);
    add.add_input(bf16_convolution_out);
    add.add_input(bf16_bias);
    add.add_output(bf16_bia_outs);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&convolution), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 7U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 6U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 9U);
}

TEST(Pass, FuseAddIntoSum) {
    /*
        \   /
         Add
          \   /
           Add
            \   /
             Add
             ...
    */
    graph_t agraph;

    const size_t rep_times = 3;
    logical_tensor_t input0 = empty_logical_tensor_with_default_id();
    logical_tensor_t input1 = empty_logical_tensor_with_default_id();
    logical_tensor_t output = empty_logical_tensor_with_default_id();
    for (size_t n = 0; n < rep_times; ++n) {
        op_t add {n, op_kind::Add, "add_" + std::to_string(n)};
        add.set_attr<std::string>(op_attr::auto_broadcast, "none");
        if (n == 0) {
            input0 = logical_tensor_init(n, data_type::f32);
        } else {
            input0 = output;
        }

        input1 = logical_tensor_init(n + 2 * rep_times, data_type::f32);
        output = logical_tensor_init(n + 1, data_type::f32);

        add.add_input(input0);
        add.add_input(input1);
        add.add_output(output);

        ASSERT_EQ(agraph.add_op(&add), status::success);
    }

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), rep_times + 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2 * rep_times);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, rep_times);
}

TEST(Pass, FuseBroadcastAddIntoSum) {
    /*
        \   /
         Add
          \   /
           Add
            ...
    */
    graph_t agraph;

    op_t add0 {0, op_kind::Add, "add0"};
    add0.set_attr<std::string>(op_attr::auto_broadcast, "none");
    logical_tensor_t input0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t input1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t output0 = logical_tensor_init(2, data_type::f32);

    op_t add1 {1, op_kind::Add, "add1"};
    add1.set_attr<std::string>(op_attr::auto_broadcast, "numpy");
    logical_tensor_t input2 = logical_tensor_init(3, data_type::f32);
    logical_tensor_t output1 = logical_tensor_init(4, data_type::f32);

    add0.add_input(input0);
    add0.add_input(input1);
    add0.add_output(output0);

    add1.add_input(output0);
    add1.add_input(input2);
    add1.add_output(output1);

    agraph.add_op(&add0);
    agraph.add_op(&add1);
    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("sum_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 0U);
}

TEST(Pass, FuseTypecaseQuantize) {
    /*
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t typecast {0, TypeCast, "typecast"};
    op_t quant {1, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t bf16_input = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t f32_input = logical_tensor_init(1, data_type::f32);
    typecast.add_input(bf16_input);
    typecast.add_output(f32_input);

    logical_tensor_t int8_dst = logical_tensor_init(2, data_type::u8);
    quant.add_input(f32_input);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&typecast), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("typecast_quantize_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2U);
}

TEST(PassSystem, FuseSoftmaxQuantize) {
    /*
             | (f32)
           softmax
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t softmax {0, SoftMax, "softmax"};
    op_t quant {1, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t src = logical_tensor_init(0, data_type::f32);
    logical_tensor_t softmax_dst = logical_tensor_init(1, data_type::f32);
    softmax.add_input(src);
    softmax.add_output(softmax_dst);

    logical_tensor_t int8_dst = logical_tensor_init(2, data_type::u8);
    quant.add_input(softmax_dst);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&softmax), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2U);
}

TEST(PassSystem, FuseLayernormQuantize) {
    /*
             | (f32)
           layernorm
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    bool keep_stats = false;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t layernorm {0, LayerNorm, "layernorm"};
    layernorm.set_attr(op_attr::keep_stats, keep_stats);
    op_t quant {1, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t src = logical_tensor_init(0, data_type::f32);
    logical_tensor_t scale = logical_tensor_init(1, data_type::f32);
    logical_tensor_t shift = logical_tensor_init(2, data_type::f32);
    logical_tensor_t layernorm_dst = logical_tensor_init(3, data_type::f32);
    layernorm.add_input(src);
    layernorm.add_input(scale);
    layernorm.add_input(shift);
    layernorm.add_output(layernorm_dst);

    logical_tensor_t int8_dst = logical_tensor_init(4, data_type::u8);
    quant.add_input(layernorm_dst);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&layernorm), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(PassSystem, FuseSoftmaxTypecast) {
    /*
             | (bf16)
           softmax
             | (bf16)
           typecast
             | (f32)
    */
    graph_t agraph;
    op_t softmax {0, SoftMax, "softmax"};
    op_t typecast {1, TypeCast, "typecast"};

    logical_tensor_t src = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t softmax_dst = logical_tensor_init(1, data_type::bf16);
    softmax.add_input(src);
    softmax.add_output(softmax_dst);

    logical_tensor_t dst = logical_tensor_init(2, data_type::f32);
    typecast.add_input(softmax_dst);
    typecast.add_output(dst);

    ASSERT_EQ(agraph.add_op(&softmax), status::success);
    ASSERT_EQ(agraph.add_op(&typecast), status::success);

    agraph.finalize();
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2U);
}

TEST(PassSystem, FuseLayernormTypecast) {
    /*
             | (bf16)
           layernorm
             | (bf16)
           typecast
             | (f32)
    */
    graph_t agraph;
    bool keep_stats = false;
    op_t layernorm {0, LayerNorm, "layernorm"};
    layernorm.set_attr(op_attr::keep_stats, keep_stats);
    op_t typecast {1, TypeCast, "typecast"};

    logical_tensor_t src = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t scale = logical_tensor_init(1, data_type::f32);
    logical_tensor_t shift = logical_tensor_init(2, data_type::f32);
    logical_tensor_t layernorm_dst = logical_tensor_init(3, data_type::bf16);

    layernorm.add_input(src);
    layernorm.add_input(scale);
    layernorm.add_input(shift);
    layernorm.add_output(layernorm_dst);

    logical_tensor_t dst = logical_tensor_init(4, data_type::f32);
    typecast.add_input(layernorm_dst);
    typecast.add_output(dst);

    ASSERT_EQ(agraph.add_op(&layernorm), status::success);
    ASSERT_EQ(agraph.add_op(&typecast), status::success);

    agraph.finalize();
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(PassSystem, FuseSoftmaxTypecastQuantize) {
    /*
             | (bf16)
           softmax
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t softmax {0, SoftMax, "softmax"};
    op_t typecast {1, TypeCast, "typecast"};
    op_t quant {2, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t src = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t softmax_dst = logical_tensor_init(1, data_type::bf16);
    softmax.add_input(src);
    softmax.add_output(softmax_dst);

    logical_tensor_t typecast_dst = logical_tensor_init(2, data_type::f32);
    typecast.add_input(softmax_dst);
    typecast.add_output(typecast_dst);

    logical_tensor_t quant_dst = logical_tensor_init(3, data_type::u8);
    quant.add_input(typecast_dst);
    quant.add_output(quant_dst);

    ASSERT_EQ(agraph.add_op(&softmax), status::success);
    ASSERT_EQ(agraph.add_op(&typecast), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(PassSystem, FuseLayernormTypecastQuantize) {
    /*
             | (bf16)
           layernorm
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    bool keep_stats = false;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t layernorm {0, LayerNorm, "layernorm"};
    layernorm.set_attr(op_attr::keep_stats, keep_stats);
    op_t typecast {1, TypeCast, "typecast"};
    op_t quant {2, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t src = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t scale_lt = logical_tensor_init(1, data_type::f32);
    logical_tensor_t shift_lt = logical_tensor_init(2, data_type::f32);
    logical_tensor_t layernorm_dst = logical_tensor_init(3, data_type::bf16);
    layernorm.add_input(src);
    layernorm.add_input(scale_lt);
    layernorm.add_input(shift_lt);
    layernorm.add_output(layernorm_dst);

    logical_tensor_t typecast_dst = logical_tensor_init(4, data_type::f32);
    typecast.add_input(layernorm_dst);
    typecast.add_output(typecast_dst);

    logical_tensor_t quant_dst = logical_tensor_init(5, data_type::u8);
    quant.add_input(typecast_dst);
    quant.add_output(quant_dst);

    ASSERT_EQ(agraph.add_op(&layernorm), status::success);
    ASSERT_EQ(agraph.add_op(&typecast), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(PassSystem, NotFuseLayernormTypecast) {
    /*
             | (bf16)
           layernorm
             | (bf16)
           typecast
             | (f32)
           quant (non-zero zps)
             | (u8/s8)
    */
    graph_t agraph;
    bool keep_stats = false;
    std::vector<int64_t> zps = {198};
    std::vector<float> scales = {3.1f};
    op_t layernorm {0, LayerNorm, "layernorm"};
    layernorm.set_attr(op_attr::keep_stats, keep_stats);
    op_t typecast {1, TypeCast, "typecast"};
    op_t quant {2, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t src = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t scale_lt = logical_tensor_init(1, data_type::f32);
    logical_tensor_t shift_lt = logical_tensor_init(2, data_type::f32);
    logical_tensor_t layernorm_dst = logical_tensor_init(3, data_type::bf16);
    layernorm.add_input(src);
    layernorm.add_input(scale_lt);
    layernorm.add_input(shift_lt);
    layernorm.add_output(layernorm_dst);

    logical_tensor_t typecast_dst = logical_tensor_init(4, data_type::f32);
    typecast.add_input(layernorm_dst);
    typecast.add_output(typecast_dst);

    logical_tensor_t quant_dst = logical_tensor_init(5, data_type::u8);
    quant.add_input(typecast_dst);
    quant.add_output(quant_dst);

    ASSERT_EQ(agraph.add_op(&layernorm), status::success);
    ASSERT_EQ(agraph.add_op(&typecast), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 2U);

    // partition 0: tc + quant
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 3U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);

    // partition 1: layernorm
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs().size(), 3U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_inputs()[2].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[1]->get_outputs()[0].id, 3U);
}

TEST(Pass, ShuffleFusion) {
    /*   reshape
            |
        transpose
            |
         reshape
    */
    const std::vector<int64_t> base_shape {8, 8, 8, 8};
    const std::vector<size_t> config_axis {1, 3};
    const int64_t g = 4;

    for (const size_t axis : config_axis) {
        const std::vector<int64_t> &reshape1_dst_shape = base_shape;

        std::vector<int64_t> reshape0_dst_shape = base_shape;
        reshape0_dst_shape[axis] /= g;
        reshape0_dst_shape.insert(reshape0_dst_shape.begin() + axis + 1, g);

        std::vector<int64_t> transpose_dst_shape = reshape0_dst_shape;
        std::swap(transpose_dst_shape[axis], transpose_dst_shape[axis + 1]);

        std::vector<int64_t> order(transpose_dst_shape.size());
        std::iota(order.begin(), order.end(), 0);
        std::swap(order[axis], order[axis + 1]);

        op_t reshape0 {0, StaticReshape, "reshape0"};
        reshape0.set_attr(op_attr::shape, reshape0_dst_shape);
        reshape0.set_attr(op_attr::special_zero, false);

        op_t transpose {1, StaticTranspose, "transpose"};
        transpose.set_attr(op_attr::order, order);

        op_t reshape1 {2, StaticReshape, "reshape1"};
        reshape1.set_attr(op_attr::shape, reshape1_dst_shape);
        reshape1.set_attr(op_attr::special_zero, false);

        logical_tensor_t reshape0_src = logical_tensor_init(0, data_type::f32);
        logical_tensor_t reshape0_dst = logical_tensor_init(1, data_type::f32);
        logical_tensor_t transpose_dst = logical_tensor_init(2, data_type::f32);
        logical_tensor_t reshape1_dst = logical_tensor_init(3, data_type::f32);

        reshape0.add_input(reshape0_src);
        reshape0.add_output(reshape0_dst);

        transpose.add_input(reshape0_dst);
        transpose.add_output(transpose_dst);

        reshape1.add_input(transpose_dst);
        reshape1.add_output(reshape1_dst);

        graph_t agraph;
        ASSERT_EQ(agraph.add_op(&reshape0), status::success);
        ASSERT_EQ(agraph.add_op(&transpose), status::success);
        ASSERT_EQ(agraph.add_op(&reshape1), status::success);
        agraph.finalize();

        pass::pass_base_ptr apass = get_pass("shuffle_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::misc_post_ops);
    }
}

TEST(PassSystem, FuseTypecaseQuantize) {
    /*
             | (bf16)
           typecast
             | (f32)
           quant
             | (u8/s8)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t typecast {0, TypeCast, "typecast"};
    op_t quant {1, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t bf16_input = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t f32_input = logical_tensor_init(1, data_type::f32);
    typecast.add_input(bf16_input);
    typecast.add_output(f32_input);

    logical_tensor_t int8_dst = logical_tensor_init(2, data_type::u8);
    quant.add_input(f32_input);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&typecast), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2U);
}

TEST(PassSystem, MixInt8AndBf16MatmulAdd) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)  / (u8/s8)
     typecast  typecast   dequant
    (bf16) \     / (bf16) / (fp32)
           matmul  typecast
         (bf16)\   / (bf16)
                add
                 | (bf16)
              typecast
                 | (f32)
               quant
                 | (u8/s8)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t add {5, Add, "add"};
    op_t dequant3 {6, Dequantize, "dequant"};
    dequant3.set_attr(op_attr::scales, scales);
    dequant3.set_attr(op_attr::zps, zps);
    op_t typecast3 {7, TypeCast, "typecast"};
    op_t typecast4 {8, TypeCast, "typecast"};
    op_t quant {9, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t bf16_bias = logical_tensor_init(7, data_type::bf16);
    logical_tensor_t bf16_matmul_out = logical_tensor_init(8, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_input(bf16_bias);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t int8_add_in = logical_tensor_init(9, data_type::u8);
    logical_tensor_t fp32_add_in = logical_tensor_init(10, data_type::f32);
    dequant3.add_input(int8_add_in);
    dequant3.add_output(fp32_add_in);

    logical_tensor_t bf16_add_in = logical_tensor_init(11, data_type::bf16);
    typecast3.add_input(fp32_add_in);
    typecast3.add_output(bf16_add_in);

    logical_tensor_t bf16_add_out = logical_tensor_init(12, data_type::bf16);
    add.add_input(bf16_matmul_out);
    add.add_input(bf16_add_in);
    add.add_output(bf16_add_out);

    logical_tensor_t fp32_add_out = logical_tensor_init(13, data_type::f32);
    typecast4.add_input(bf16_add_out);
    typecast4.add_output(fp32_add_out);

    logical_tensor_t int8_dst = logical_tensor_init(14, data_type::u8);
    quant.add_input(fp32_add_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&dequant3), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&typecast4), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
}

TEST(PassSystem, MixInt8AndBf16MatmulDiv) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
           matmul
         (bf16)\   /(bf16)
                div
                 | (bf16)
              typecast
                 | (f32)
               quant
                 | (u8/s8)
    */
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t typecast1 {2, TypeCast, "typecast"};
    op_t typecast2 {3, TypeCast, "typecast"};
    op_t matmul {4, MatMul, "matmul"};
    op_t div {5, Divide, "divide"};
    op_t typecast3 {6, TypeCast, "typecast"};
    op_t quant {7, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

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

    logical_tensor_t bf16_matmul_out = logical_tensor_init(6, data_type::bf16);
    matmul.add_input(bf16_data);
    matmul.add_input(bf16_weight);
    matmul.add_output(bf16_matmul_out);

    logical_tensor_t bf16_div_in = logical_tensor_init(7, data_type::bf16);
    logical_tensor_t bf16_div_out = logical_tensor_init(8, data_type::bf16);
    div.add_input(bf16_matmul_out);
    div.add_input(bf16_div_in);
    div.add_output(bf16_div_out);

    logical_tensor_t fp32_div_out = logical_tensor_init(9, data_type::f32);
    typecast3.add_input(bf16_div_out);
    typecast3.add_output(fp32_div_out);

    logical_tensor_t int8_dst = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_div_out);
    quant.add_output(int8_dst);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&typecast1), status::success);
    ASSERT_EQ(agraph.add_op(&typecast2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&typecast3), status::success);
    ASSERT_EQ(agraph.add_op(&div), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    pm.run_passes(agraph, "no_config");

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_matmul_post_ops);
}

TEST(Pass, FuseBnReLUWithSharedInputs) {
    /*   bn
          |
         relu
    */
    graph_t agraph;
    op_t bn {0, BatchNormInference, "bn"};
    bn.set_attr(op_attr::epsilon, 0.001f);
    op_t relu {1, ReLU, "relu"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    bn.add_input(lt_vec[0]);
    //assume gamma/beta/mean/var are using the same lt
    bn.add_input(lt_vec[1]);
    bn.add_input(lt_vec[1]);
    bn.add_input(lt_vec[1]);
    bn.add_input(lt_vec[1]);
    bn.add_output(lt_vec[2]);
    relu.add_input(lt_vec[2]);
    relu.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&bn), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("bn_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::batch_norm_post_ops);

    // For a partition with N inputs that have the same id
    // It is required that those inputs are input N times
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 5U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[4].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseReorderAdd) {
    /*
             |
         reorder
             |  /
            add
             |
    */
    const std::vector<data_type_t> dtypes {data_type::f32, data_type::bf16};
    for (auto &dtype : dtypes) {
        graph_t agraph;
        op_t reorder {0, Reorder, "reorder"};
        op_t add {1, Add, "add"};
        add.set_attr<std::string>(op_attr::auto_broadcast, "none");

        logical_tensor_t src_lt = logical_tensor_init(0, {2, 3}, {3, 1}, dtype);
        logical_tensor_t dst_lt = logical_tensor_init(1, {2, 3}, {1, 2}, dtype);
        reorder.add_input(src_lt);
        reorder.add_output(dst_lt);

        logical_tensor_t add_src_lt
                = logical_tensor_init(2, {2, 3}, {1, 2}, dtype);
        logical_tensor_t add_dst_lt = logical_tensor_init(3, dtype);
        add.add_input(dst_lt);
        add.add_input(add_src_lt);
        add.add_output(add_dst_lt);

        ASSERT_EQ(agraph.add_op(&reorder), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);

        ASSERT_EQ(agraph.finalize(), status::success);

        pass::pass_base_ptr apass = get_pass("reorder_sum_fusion");
        apass->run(agraph);

        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
    }
}

TEST(Pass, FailToFuseReorderAdd) {
    /*
             |
         reorder
             |  /
            add
             |
    */
    graph_t agraph;
    op_t reorder {0, Reorder, "reorder"};
    op_t add {1, Add, "add"};
    // if add support auto_broadcast, it cannot be fused
    add.set_attr<std::string>(op_attr::auto_broadcast, "numpy");

    logical_tensor_t src_lt
            = logical_tensor_init(0, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t dst_lt
            = logical_tensor_init(1, {2, 3}, {1, 2}, data_type::f32);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);

    logical_tensor_t add_src_lt
            = logical_tensor_init(2, {2, 3}, {1, 2}, data_type::f32);
    logical_tensor_t add_dst_lt = logical_tensor_init(3, data_type::f32);
    add.add_input(dst_lt);
    add.add_input(add_src_lt);
    add.add_output(add_dst_lt);

    ASSERT_EQ(agraph.add_op(&reorder), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);

    ASSERT_EQ(agraph.finalize(), status::success);

    pass::pass_base_ptr apass = get_pass("reorder_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 0U);
}

TEST(Pass, FuseInt8Reorder) {
    /*
         dequantize
             |
         reorder
             |
         quantize
             |
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant {0, Dequantize, "dequant"};
    dequant.set_attr(op_attr::scales, scales);
    dequant.set_attr(op_attr::zps, zps);
    op_t reorder {1, Reorder, "reorder"};
    op_t quant {2, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t int8_src_lt
            = logical_tensor_init(0, {2, 3}, {3, 1}, data_type::u8);
    logical_tensor_t src_lt
            = logical_tensor_init(1, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t dst_lt
            = logical_tensor_init(2, {2, 3}, {1, 2}, data_type::f32);
    logical_tensor_t int8_dst_lt
            = logical_tensor_init(3, data_type::u8, layout_type::strided);
    dequant.add_input(int8_src_lt);
    dequant.add_output(src_lt);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);
    quant.add_input(dst_lt);
    quant.add_output(int8_dst_lt);

    ASSERT_EQ(agraph.add_op(&dequant), status::success);
    ASSERT_EQ(agraph.add_op(&reorder), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    ASSERT_EQ(agraph.finalize(), status::success);

    pass::pass_base_ptr apass = get_pass("int8_reorder_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(PassSystem, FuseInt8Reorder) {
    /*
         dequantize
             |
         reorder
             |
         quantize
             |
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant {0, Dequantize, "dequant"};
    dequant.set_attr(op_attr::scales, scales);
    dequant.set_attr(op_attr::zps, zps);
    op_t reorder {1, Reorder, "reorder"};
    op_t quant {2, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t int8_src_lt
            = logical_tensor_init(0, {2, 3}, {3, 1}, data_type::u8);
    logical_tensor_t src_lt
            = logical_tensor_init(1, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t dst_lt
            = logical_tensor_init(2, {2, 3}, {1, 2}, data_type::f32);
    logical_tensor_t int8_dst_lt
            = logical_tensor_init(3, data_type::u8, layout_type::strided);
    dequant.add_input(int8_src_lt);
    dequant.add_output(src_lt);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);
    quant.add_input(dst_lt);
    quant.add_output(int8_dst_lt);

    ASSERT_EQ(agraph.add_op(&dequant), status::success);
    ASSERT_EQ(agraph.add_op(&reorder), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    ASSERT_EQ(agraph.finalize(), status::success);

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::misc_quantized_post_ops);
}

TEST(Pass, FuseInt8ReorderAdd) {
    /*
         dequantize
             |
         reorder dequantize
             |  /
            add
             |
         quantize
             |
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant {0, Dequantize, "dequant"};
    dequant.set_attr(op_attr::scales, scales);
    dequant.set_attr(op_attr::zps, zps);
    op_t reorder {1, Reorder, "reorder"};
    op_t dequant_other {2, Dequantize, "dequant_other"};
    dequant_other.set_attr(op_attr::scales, scales);
    dequant_other.set_attr(op_attr::zps, zps);
    op_t add {3, Add, "add"};
    add.set_attr<std::string>(op_attr::auto_broadcast, "none");
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t int8_src_lt
            = logical_tensor_init(0, {2, 3}, {3, 1}, data_type::u8);
    logical_tensor_t int8_src_other_lt
            = logical_tensor_init(1, {2, 3}, {3, 1}, data_type::u8);
    logical_tensor_t src_lt
            = logical_tensor_init(2, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t src_other_lt
            = logical_tensor_init(3, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t dst_lt
            = logical_tensor_init(4, {2, 3}, {1, 2}, data_type::f32);
    logical_tensor_t dst_add_lt
            = logical_tensor_init(5, {2, 3}, {1, 2}, data_type::f32);
    logical_tensor_t int8_dst_add_lt
            = logical_tensor_init(6, data_type::u8, layout_type::strided);
    dequant.add_input(int8_src_lt);
    dequant.add_output(src_lt);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);
    dequant_other.add_input(int8_src_other_lt);
    dequant_other.add_output(src_other_lt);
    add.add_input(dst_lt);
    add.add_input(src_other_lt);
    add.add_output(dst_add_lt);
    quant.add_input(dst_add_lt);
    quant.add_output(int8_dst_add_lt);

    ASSERT_EQ(agraph.add_op(&dequant), status::success);
    ASSERT_EQ(agraph.add_op(&dequant_other), status::success);
    ASSERT_EQ(agraph.add_op(&reorder), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    ASSERT_EQ(agraph.finalize(), status::success);

    pass::pass_base_ptr apass = get_pass("int8_reorder_sum_fusion_cpu");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 6U);
}

TEST(PassSystem, FuseInt8ReorderAdd) {
    /*
         dequantize
             |
         reorder dequantize
             |  /
            add
             |
         quantize
             |
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant {0, Dequantize, "dequant"};
    dequant.set_attr(op_attr::scales, scales);
    dequant.set_attr(op_attr::zps, zps);
    op_t reorder {1, Reorder, "reorder"};
    op_t dequant_other {2, Dequantize, "dequant_other"};
    dequant_other.set_attr(op_attr::scales, scales);
    dequant_other.set_attr(op_attr::zps, zps);
    op_t add {3, Add, "add"};
    add.set_attr<std::string>(op_attr::auto_broadcast, "none");
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t int8_src_lt
            = logical_tensor_init(0, {2, 3}, {3, 1}, data_type::u8);
    logical_tensor_t int8_src_other_lt
            = logical_tensor_init(1, {2, 3}, {3, 1}, data_type::u8);
    logical_tensor_t src_lt
            = logical_tensor_init(2, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t src_other_lt
            = logical_tensor_init(3, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t dst_lt
            = logical_tensor_init(4, {2, 3}, {1, 2}, data_type::f32);
    logical_tensor_t dst_add_lt
            = logical_tensor_init(5, {2, 3}, {1, 2}, data_type::f32);
    logical_tensor_t int8_dst_add_lt
            = logical_tensor_init(6, data_type::u8, layout_type::strided);
    dequant.add_input(int8_src_lt);
    dequant.add_output(src_lt);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);
    dequant_other.add_input(int8_src_other_lt);
    dequant_other.add_output(src_other_lt);
    add.add_input(dst_lt);
    add.add_input(src_other_lt);
    add.add_output(dst_add_lt);
    quant.add_input(dst_add_lt);
    quant.add_output(int8_dst_add_lt);

    ASSERT_EQ(agraph.add_op(&dequant), status::success);
    ASSERT_EQ(agraph.add_op(&dequant_other), status::success);
    ASSERT_EQ(agraph.add_op(&reorder), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    ASSERT_EQ(agraph.finalize(), status::success);

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::misc_quantized_post_ops);
}

TEST(Pass, SingleInterpolatePass) {
    graph_t agraph;
    op_t interpolate {0, Interpolate, "interpolate"};

    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_out = logical_tensor_init(1, data_type::f32);
    interpolate.add_input(lt_data);
    interpolate.add_output(lt_out);
    interpolate.set_attr(op_attr::sizes, std::vector<int64_t> {2, 3, 4});
    interpolate.set_attr(op_attr::scales, std::vector<float> {});
    interpolate.set_attr(op_attr::mode, std::string("linear"));
    interpolate.set_attr(
            op_attr::coordinate_transformation_mode, std::string("half_pixel"));

    ASSERT_EQ(agraph.add_op(&interpolate), status::success);
    ASSERT_EQ(agraph.finalize(), status::success);
    pass::pass_base_ptr apass = get_pass("interpolate_pass");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    interpolate.set_attr(op_attr::coordinate_transformation_mode,
            std::string("align_corners"));
    graph_t fgraph;
    ASSERT_EQ(fgraph.add_op(&interpolate), status::success);
    ASSERT_EQ(fgraph.finalize(), status::success);
    apass->run(fgraph);
    ASSERT_EQ(fgraph.get_num_partitions(), 0U);
}

TEST(Pass, FuseInterpolateRelu) {
    /* interpolate
            |
           relu
    */
    graph_t agraph;
    op_t interpolate {0, Interpolate, "interpolate"};
    interpolate.set_attr(op_attr::sizes, std::vector<int64_t> {2, 3, 4});
    interpolate.set_attr(op_attr::mode, std::string("linear"));
    interpolate.set_attr(
            op_attr::coordinate_transformation_mode, std::string("half_pixel"));
    op_t relu {1, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(3);
    interpolate.add_input(lt_vec[0]);
    interpolate.add_output(lt_vec[1]);
    relu.add_input(lt_vec[1]);
    relu.add_output(lt_vec[2]);

    ASSERT_EQ(agraph.add_op(&interpolate), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("interpolate_post_ops_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::interpolate_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2U);
}

TEST(Pass, FuseInterpolateSwish) {
    /*    interpolate
            /    |
      sigmoid    |
           \     /
           multiply
              |
            relu
    */
    graph_t agraph;
    op_t interpolate {0, Interpolate, "interpolate"};
    interpolate.set_attr(op_attr::sizes, std::vector<int64_t> {2, 3, 4});
    interpolate.set_attr(op_attr::mode, std::string("linear"));
    interpolate.set_attr(
            op_attr::coordinate_transformation_mode, std::string("half_pixel"));
    op_t sigmoid {1, Sigmoid, "sigmoid"};
    op_t multiply {2, Multiply, "mul"};
    op_t relu {3, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    interpolate.add_input(lt_vec[0]);
    interpolate.add_output(lt_vec[1]);
    sigmoid.add_input(lt_vec[1]);
    sigmoid.add_output(lt_vec[2]);
    multiply.add_input(lt_vec[1]);
    multiply.add_input(lt_vec[2]);
    multiply.add_output(lt_vec[3]);
    relu.add_input(lt_vec[3]);
    relu.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&interpolate), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("interpolate_post_ops_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 4U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::interpolate_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(PassSystem, FuseInterpolateSwish) {
    /*    interpolate
            /    |
      sigmoid    |
           \     /
           multiply
              |
            relu
    */
    graph_t agraph;
    op_t interpolate {0, Interpolate, "interpolate"};
    interpolate.set_attr(op_attr::sizes, std::vector<int64_t> {2, 3, 4});
    interpolate.set_attr(op_attr::mode, std::string("linear"));
    interpolate.set_attr(
            op_attr::coordinate_transformation_mode, std::string("half_pixel"));
    op_t sigmoid {1, Sigmoid, "sigmoid"};
    op_t multiply {2, Multiply, "mul"};
    op_t relu {3, ReLU, "relu"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    interpolate.add_input(lt_vec[0]);
    interpolate.add_output(lt_vec[1]);
    sigmoid.add_input(lt_vec[1]);
    sigmoid.add_output(lt_vec[2]);
    multiply.add_input(lt_vec[1]);
    multiply.add_input(lt_vec[2]);
    multiply.add_output(lt_vec[3]);
    relu.add_input(lt_vec[3]);
    relu.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&interpolate), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 4U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::interpolate_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(Pass, FuseInterpolate3PostOps) {
    /*    interpolate
               |
           sigmoid
               |
             relu
               |    /
              multiply
    */
    graph_t agraph;
    op_t interpolate {0, Interpolate, "interpolate"};
    interpolate.set_attr(op_attr::sizes, std::vector<int64_t> {2, 3, 4});
    interpolate.set_attr(op_attr::mode, std::string("linear"));
    interpolate.set_attr(
            op_attr::coordinate_transformation_mode, std::string("half_pixel"));
    op_t sigmoid {1, Sigmoid, "sigmoid"};
    op_t relu {2, ReLU, "relu"};
    op_t multiply {3, Multiply, "mul"};
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    interpolate.add_input(lt_vec[0]);
    interpolate.add_output(lt_vec[1]);
    sigmoid.add_input(lt_vec[1]);
    sigmoid.add_output(lt_vec[2]);
    relu.add_input(lt_vec[2]);
    relu.add_output(lt_vec[3]);
    multiply.add_input(lt_vec[3]);
    multiply.add_input(lt_vec[4]);
    multiply.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&interpolate), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 4U);

    pass::pass_base_ptr apass = get_pass("interpolate_post_ops_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::interpolate_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FuseInterpolateSum) {
    /*   interpolate
             \           /
               \        /
                  add
    */
    graph_t agraph;
    op_t interpolate {0, Interpolate, "interpolate"};
    interpolate.set_attr(op_attr::sizes, std::vector<int64_t> {2, 3, 4});
    interpolate.set_attr(op_attr::mode, std::string("linear"));
    interpolate.set_attr(
            op_attr::coordinate_transformation_mode, std::string("half_pixel"));
    op_t add {1, Add, "add"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    interpolate.add_input(lt_vec[0]);
    interpolate.add_output(lt_vec[1]);
    add.add_input(lt_vec[1]);
    add.add_input(lt_vec[2]);
    add.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&interpolate), status::success);
    ASSERT_EQ(agraph.add_op(&add), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("interpolate_post_ops_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, FuseInterpolateMul) {
    /*   interpolate
             \           /
               \        /
                Multiply
    */
    graph_t agraph;
    op_t interpolate {0, Interpolate, "interpolate"};
    interpolate.set_attr<std::vector<int64_t>>(op_attr::sizes, {2, 3, 4});
    interpolate.set_attr<std::string>(op_attr::mode, "linear");
    interpolate.set_attr<std::string>(
            op_attr::coordinate_transformation_mode, "half_pixel");
    op_t mul {1, Multiply, "mul"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    interpolate.add_input(lt_vec[0]);
    interpolate.add_output(lt_vec[1]);
    mul.add_input(lt_vec[1]);
    mul.add_input(lt_vec[2]);
    mul.add_output(lt_vec[3]);

    ASSERT_EQ(agraph.add_op(&interpolate), status::success);
    ASSERT_EQ(agraph.add_op(&mul), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 2U);

    pass::pass_base_ptr apass = get_pass("interpolate_post_ops_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
}

TEST(Pass, Int8MhaFusion) {
    dnnl::impl::graph::graph_t agraph;
    dnnl::graph::tests::unit::utils::construct_int8_MHA(&agraph);
    agraph.finalize();
    ASSERT_EQ(agraph.get_ops().size(), 13U);

    dnnl::impl::graph::pass::pass_base_ptr apass = get_pass("int8_sdp_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
}

TEST(Pass, F32MhaFusion) {
    dnnl::impl::graph::graph_t agraph;
    dnnl::graph::tests::unit::utils::construct_dnnl_float_MHA(&agraph);
    agraph.finalize();
    ASSERT_EQ(agraph.get_ops().size(), 7U);

    dnnl::impl::graph::pass::pass_base_ptr apass = get_pass("float_sdp_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
}

TEST(Pass, FuseReduceAdd) {
    /* reduce
          |
         add
    */
    const std::vector<op_kind_t> configs {ReduceL1, ReduceL2, ReduceMax,
            ReduceMean, ReduceMin, ReduceProd, ReduceSum};

    for (const auto &base_op : configs) {
        op_t reduce {0, base_op, "reduce"};
        reduce.set_attr<bool>(op_attr::keep_dims, false);
        reduce.set_attr<std::vector<int64_t>>(op_attr::axes, {0});
        op_t add {1, Add, "add"};

        logical_tensor_t reduce_src = logical_tensor_init(0, data_type::f32);
        logical_tensor_t reduce_dst = logical_tensor_init(2, data_type::f32);

        logical_tensor_t add_src1 = logical_tensor_init(3, data_type::f32);
        logical_tensor_t add_dst = logical_tensor_init(4, data_type::f32);

        reduce.add_input(reduce_src);
        reduce.add_output(reduce_dst);

        add.add_input(reduce_dst);
        add.add_input(add_src1);
        add.add_output(add_dst);

        graph_t agraph;
        ASSERT_EQ(agraph.add_op(&reduce), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        agraph.finalize();

        pass::pass_base_ptr apass = get_pass("reduction_post_ops_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::reduction_post_ops);
    }
}

TEST(Pass, FuseReduceRelu) {
    /* reduce
          |
        relu
    */
    const std::vector<op_kind_t> configs {ReduceL1, ReduceL2, ReduceMax,
            ReduceMean, ReduceMin, ReduceProd, ReduceSum};

    for (const auto &base_op : configs) {
        op_t reduce {0, base_op, "reduce"};
        reduce.set_attr<bool>(op_attr::keep_dims, false);
        reduce.set_attr<std::vector<int64_t>>(op_attr::axes, {0});
        op_t relu {1, ReLU, "relu"};

        logical_tensor_t reduce_src = logical_tensor_init(0, data_type::f32);
        logical_tensor_t reduce_dst = logical_tensor_init(2, data_type::f32);

        logical_tensor_t relu_dst = logical_tensor_init(3, data_type::f32);

        reduce.add_input(reduce_src);
        reduce.add_output(reduce_dst);

        relu.add_input(reduce_dst);
        relu.add_output(relu_dst);

        graph_t agraph;
        ASSERT_EQ(agraph.add_op(&reduce), status::success);
        ASSERT_EQ(agraph.add_op(&relu), status::success);
        agraph.finalize();

        pass::pass_base_ptr apass = get_pass("reduction_post_ops_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::reduction_post_ops);
    }
}

TEST(PassSystem, FuseReduceSwish) {
    /*       reduce
            /    |
        sigmoid  |
            \    |
            multiply
    */
    const std::vector<op_kind_t> configs {ReduceL1, ReduceL2, ReduceMax,
            ReduceMean, ReduceMin, ReduceProd, ReduceSum};

    for (const auto &base_op : configs) {
        op_t reduce {0, base_op, "reduce"};
        reduce.set_attr<bool>(op_attr::keep_dims, false);
        reduce.set_attr<std::vector<int64_t>>(op_attr::axes, {0});
        op_t sigmoid {1, Sigmoid, "sigmoid"};
        op_t multiply {2, Multiply, "mul"};

        logical_tensor_t reduce_src = logical_tensor_init(0, data_type::f32);
        logical_tensor_t reduce_dst = logical_tensor_init(1, data_type::f32);
        logical_tensor_t sigmoid_dst = logical_tensor_init(2, data_type::f32);
        logical_tensor_t mul_dst = logical_tensor_init(3, data_type::f32);

        reduce.add_input(reduce_src);
        reduce.add_output(reduce_dst);

        sigmoid.add_input(reduce_dst);
        sigmoid.add_output(sigmoid_dst);

        multiply.add_input(sigmoid_dst);
        multiply.add_input(reduce_dst);
        multiply.add_output(mul_dst);

        graph_t agraph;
        ASSERT_EQ(agraph.add_op(&reduce), status::success);
        ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
        ASSERT_EQ(agraph.add_op(&multiply), status::success);
        agraph.finalize();

        auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
        auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
        pm.run_passes(agraph, "no_config");
        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::reduction_post_ops);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 3U);
    }
}

TEST(PassSystem, FuseReduceWith3PostOps) {
    /*       reducel1
               |
             relu
               |
             sigmoid
               \     /
               multiply
    */
    op_t reduce {0, ReduceL1, "reduce"};
    reduce.set_attr<bool>(op_attr::keep_dims, false);
    reduce.set_attr<std::vector<int64_t>>(op_attr::axes, {0});
    op_t relu {1, ReLU, "relu"};
    op_t sigmoid {2, Sigmoid, "sigmoid"};
    op_t multiply {3, Multiply, "mul"};

    logical_tensor_t reduce_src = logical_tensor_init(0, data_type::f32);
    logical_tensor_t reduce_dst = logical_tensor_init(1, data_type::f32);
    logical_tensor_t relu_dst = logical_tensor_init(2, data_type::f32);
    logical_tensor_t sigmoid_dst = logical_tensor_init(3, data_type::f32);
    logical_tensor_t mul_src = logical_tensor_init(4, data_type::f32);
    logical_tensor_t mul_dst = logical_tensor_init(5, data_type::f32);

    reduce.add_input(reduce_src);
    reduce.add_output(reduce_dst);

    relu.add_input(reduce_dst);
    relu.add_output(relu_dst);

    sigmoid.add_input(relu_dst);
    sigmoid.add_output(sigmoid_dst);

    multiply.add_input(mul_src);
    multiply.add_input(sigmoid_dst);
    multiply.add_output(mul_dst);

    graph_t agraph;
    ASSERT_EQ(agraph.add_op(&reduce), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    agraph.finalize();

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::reduction_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 4U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 5U);
}

TEST(Pass, FailToFuseReduceWithEmptyScales) {
    const std::vector<op_kind_t> configs {ReduceL1, ReduceL2, ReduceMax,
            ReduceMean, ReduceMin, ReduceProd, ReduceSum};
    for (const auto base_op : configs) {
        graph_t agraph;
        op_t reduce {0, base_op, "reduce"};
        reduce.set_attr<std::vector<int64_t>>(op_attr::axes, {});

        std::vector<logical_tensor_t> lt_vec = create_logical_tensors(2);
        reduce.add_input(lt_vec[0]);
        reduce.add_output(lt_vec[1]);

        ASSERT_EQ(agraph.add_op(&reduce), status::success);
        agraph.finalize();
        ASSERT_EQ(agraph.num_ops(), 1U);

        pass::pass_base_ptr apass = get_pass("reduce_pass");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 0U);
    }
}

TEST(Pass, Int8Concat) {
    /*
         dq  dq dq  ..
          \  |  |  /
            concat
               |
           quantize
               |
    */
    const size_t max_num_dq = 32;
    for (size_t i = 1; i <= max_num_dq; ++i) {
        graph_t agraph;
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        // test concat with cur_num_dq inputs
        const size_t cur_num_dq = i;
        op_t concat {cur_num_dq, Concat, "concat"};
        concat.set_attr<int64_t>(op_attr::axis, 1);
        logical_tensor_t input = empty_logical_tensor_with_default_id();
        logical_tensor_t output = empty_logical_tensor_with_default_id();
        for (size_t j = 0; j < cur_num_dq; ++j) {
            op_t dequant {j, Dequantize, "dequant"};
            dequant.set_attr(op_attr::scales, scales);
            dequant.set_attr(op_attr::zps, zps);
            input = logical_tensor_init(2 * j, data_type::u8);
            output = logical_tensor_init(2 * j + 1, data_type::f32);
            dequant.add_input(input);
            dequant.add_output(output);
            ASSERT_EQ(agraph.add_op(&dequant), status::success);
            concat.add_input(output);
        }

        logical_tensor_t dst_lt
                = logical_tensor_init(2 * cur_num_dq, data_type::f32);
        concat.add_output(dst_lt);
        ASSERT_EQ(agraph.add_op(&concat), status::success);
        op_t quant {cur_num_dq + 1, Quantize, "quant"};
        quant.set_attr(op_attr::scales, scales);
        quant.set_attr(op_attr::zps, zps);
        logical_tensor_t int8_dst_lt
                = logical_tensor_init(2 * cur_num_dq + 1, data_type::u8);
        quant.add_input(dst_lt);
        quant.add_output(int8_dst_lt);
        ASSERT_EQ(agraph.add_op(&quant), status::success);

        ASSERT_EQ(agraph.finalize(), status::success);

        pass::pass_base_ptr apass = get_pass("int8_concat_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::misc_quantized_post_ops);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), cur_num_dq);
        for (size_t k = 0; k < cur_num_dq; ++k) {
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[k].id, 2 * k);
        }
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                2 * cur_num_dq + 1);
    }
}

TEST(Pass, FailToFuseInt8Concat) {
    /*
         dq  dq not_dq
          \  |    /
            concat
               |
           quantize
               |
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant1"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant2"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t concat {3, Concat, "concat"};
    concat.set_attr<int64_t>(op_attr::axis, 1);
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t int8_src_lt1
            = logical_tensor_init(0, {2, 3}, {3, 1}, data_type::u8);
    logical_tensor_t src_lt1
            = logical_tensor_init(1, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t int8_src_lt2
            = logical_tensor_init(2, {2, 3}, {3, 1}, data_type::u8);
    logical_tensor_t src_lt2
            = logical_tensor_init(3, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t src_lt3
            = logical_tensor_init(5, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t dst_lt = logical_tensor_init(6, data_type::f32);
    logical_tensor_t int8_dst_lt = logical_tensor_init(7, data_type::u8);
    dequant1.add_input(int8_src_lt1);
    dequant1.add_output(src_lt1);
    dequant2.add_input(int8_src_lt2);
    dequant2.add_output(src_lt2);
    concat.add_input(src_lt1);
    concat.add_input(src_lt2);
    // src_lt3 is not produced by a dequant
    concat.add_input(src_lt3);
    concat.add_output(dst_lt);
    quant.add_input(dst_lt);
    quant.add_output(int8_dst_lt);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&concat), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    ASSERT_EQ(agraph.finalize(), status::success);

    pass::pass_base_ptr apass = get_pass("int8_concat_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 0U);
}

TEST(Pass, FuseToInt8ConvTransposeAdd) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            deconv w/wo bias
             | (f32)
             |     | (s8)
             |   dequant
             |  / (f32)
            add
             | (f32)
           quant
             | (u8/s8)
    */
    std::vector<bool> with_biases {false, true};

    for (auto with_bias : with_biases) {
        graph_t agraph;
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr(op_attr::scales, scales);
        dequant1.set_attr(op_attr::zps, zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales);
        dequant2.set_attr(op_attr::zps, zps);
        op_t dequant3 {2, Dequantize, "dequant"};
        dequant3.set_attr(op_attr::scales, scales);
        dequant3.set_attr(op_attr::zps, zps);
        op_t deconv {3, ConvTranspose, "deconv"};
        set_convtranspose_common_attr(deconv);
        op_t add {5, Add, "add"};
        op_t quant {6, Quantize, "quant"};
        quant.set_attr(op_attr::scales, scales);
        quant.set_attr(op_attr::zps, zps);

        int lt_id = -1;
        logical_tensor_t int8_data
                = logical_tensor_init(++lt_id, data_type::u8);
        logical_tensor_t fp32_data
                = logical_tensor_init(++lt_id, data_type::f32);
        dequant1.add_input(int8_data);
        dequant1.add_output(fp32_data);

        logical_tensor_t s8_weight
                = logical_tensor_init(++lt_id, data_type::s8);
        logical_tensor_t fp32_weight
                = logical_tensor_init(++lt_id, data_type::f32);
        dequant2.add_input(s8_weight);
        dequant2.add_output(fp32_weight);

        logical_tensor_t fp32_bias;
        if (with_bias) fp32_bias = logical_tensor_init(++lt_id, data_type::f32);
        logical_tensor_t fp32_deconv_out
                = logical_tensor_init(++lt_id, data_type::f32);
        deconv.add_input(fp32_data);
        deconv.add_input(fp32_weight);
        if (with_bias) deconv.add_input(fp32_bias);
        deconv.add_output(fp32_deconv_out);

        logical_tensor_t int8_other
                = logical_tensor_init(++lt_id, data_type::u8);
        logical_tensor_t fp32_other
                = logical_tensor_init(++lt_id, data_type::f32);
        dequant3.add_input(int8_other);
        dequant3.add_output(fp32_other);

        logical_tensor_t fp32_add_out
                = logical_tensor_init(++lt_id, data_type::f32);
        add.add_input(fp32_deconv_out);
        add.add_input(fp32_other);
        add.add_output(fp32_add_out);

        logical_tensor_t int8_out = logical_tensor_init(++lt_id, data_type::u8);
        quant.add_input(fp32_add_out);
        quant.add_output(int8_out);

        ASSERT_EQ(agraph.add_op(&dequant1), status::success);
        ASSERT_EQ(agraph.add_op(&dequant2), status::success);
        ASSERT_EQ(agraph.add_op(&dequant3), status::success);
        ASSERT_EQ(agraph.add_op(&deconv), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        ASSERT_EQ(agraph.add_op(&quant), status::success);

        agraph.finalize();

        pass::pass_base_ptr apass
                = get_pass("int8_convtranspose_add_post_ops_fusion_cpu");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::quantized_convtranspose_post_ops);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(),
                with_bias ? 4U : 3U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
        if (with_bias) {
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 6U);
        } else {
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 5U);
        }

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                with_bias ? 9U : 8U);
    }
}

TEST(PassSystem, FuseToInt8ConvTransposeAdd) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            deconv w/wo bias
             | (f32)
             |     | (s8)
             |   dequant
             |  / (f32)
            add
             | (f32)
           quant
             | (u8/s8)
    */
    std::vector<bool> with_biases {false, true};
    std::vector<engine_kind_t> engine_kinds
            = {engine_kind::cpu, engine_kind::gpu};

    for_(const auto &engine_kind : engine_kinds)
    for (auto with_bias : with_biases) {
        graph_t agraph(engine_kind);
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = engine_kind == engine_kind::gpu
                ? std::vector<float> {1.f}
                : std::vector<float> {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr(op_attr::scales, scales);
        dequant1.set_attr(op_attr::zps, zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales);
        dequant2.set_attr(op_attr::zps, zps);
        op_t dequant3 {2, Dequantize, "dequant"};
        dequant3.set_attr(op_attr::scales, scales);
        dequant3.set_attr(op_attr::zps, zps);
        op_t deconv {3, ConvTranspose, "deconv"};
        set_convtranspose_common_attr(deconv);
        op_t add {5, Add, "add"};
        op_t quant {6, Quantize, "quant"};
        quant.set_attr(op_attr::scales, scales);
        quant.set_attr(op_attr::zps, zps);

        int lt_id = -1;
        logical_tensor_t int8_data
                = logical_tensor_init(++lt_id, data_type::u8);
        logical_tensor_t fp32_data
                = logical_tensor_init(++lt_id, data_type::f32);
        dequant1.add_input(int8_data);
        dequant1.add_output(fp32_data);

        logical_tensor_t s8_weight
                = logical_tensor_init(++lt_id, data_type::s8);
        logical_tensor_t fp32_weight
                = logical_tensor_init(++lt_id, data_type::f32);
        dequant2.add_input(s8_weight);
        dequant2.add_output(fp32_weight);

        logical_tensor_t fp32_bias;
        if (with_bias) fp32_bias = logical_tensor_init(++lt_id, data_type::f32);
        logical_tensor_t fp32_deconv_out
                = logical_tensor_init(++lt_id, data_type::f32);
        deconv.add_input(fp32_data);
        deconv.add_input(fp32_weight);
        if (with_bias) deconv.add_input(fp32_bias);
        deconv.add_output(fp32_deconv_out);

        logical_tensor_t int8_other
                = logical_tensor_init(++lt_id, data_type::u8);
        logical_tensor_t fp32_other
                = logical_tensor_init(++lt_id, data_type::f32);
        dequant3.add_input(int8_other);
        dequant3.add_output(fp32_other);

        logical_tensor_t fp32_add_out
                = logical_tensor_init(++lt_id, data_type::f32);
        add.add_input(fp32_deconv_out);
        add.add_input(fp32_other);
        add.add_output(fp32_add_out);

        logical_tensor_t int8_out = logical_tensor_init(++lt_id, data_type::u8);
        quant.add_input(fp32_add_out);
        quant.add_output(int8_out);

        ASSERT_EQ(agraph.add_op(&dequant1), status::success);
        ASSERT_EQ(agraph.add_op(&dequant2), status::success);
        ASSERT_EQ(agraph.add_op(&dequant3), status::success);
        ASSERT_EQ(agraph.add_op(&deconv), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);
        ASSERT_EQ(agraph.add_op(&quant), status::success);

        agraph.finalize();

        auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
        auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
        pm.run_passes(agraph, "no_config");
        if (engine_kind == engine_kind::cpu) {
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::quantized_convtranspose_post_ops);

            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(),
                    with_bias ? 4U : 3U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
            if (with_bias) {
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 6U);
            } else {
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 5U);
            }

            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                    with_bias ? 9U : 8U);
        } else {
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::quantized_convtranspose_post_ops);
        }
    }
}

TEST(Pass, FuseToInt8ConvtransposeEltwise) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            deconv w/wo bias
             | (f32)
            eltwise
             | (f32)
           quant
             | (u8/s8)
    */
    const std::vector<graph::op_kind_t> eltwise_kinds = {
            Abs,
            Clamp,
            Elu,
            Exp,
            GELU,
            Log,
            ReLU,
            Round,
            Sigmoid,
            Sqrt,
            Square,
            Tanh,
    };
    std::vector<bool> with_biases {false, true};

    for (auto &eltwise_kind : eltwise_kinds) {
        for (auto with_bias : with_biases) {
            graph_t agraph;
            std::vector<int64_t> zps = {0};
            std::vector<float> scales = {3.1f};
            op_t dequant1 {0, Dequantize, "dequant"};
            dequant1.set_attr(op_attr::scales, scales);
            dequant1.set_attr(op_attr::zps, zps);
            op_t dequant2 {1, Dequantize, "dequant"};
            dequant2.set_attr(op_attr::scales, scales);
            dequant2.set_attr(op_attr::zps, zps);
            op_t deconv {2, ConvTranspose, "deconv"};
            set_convtranspose_common_attr(deconv);
            op_t eltwise {3, eltwise_kind, "relu"};
            if (eltwise_kind == op_kind::Elu) {
                eltwise.set_attr<float>(op_attr::alpha, 1.f);
            } else if (eltwise_kind == op_kind::Clamp) {
                eltwise.set_attr<float>(op_attr::min, -1.f);
                eltwise.set_attr<float>(op_attr::max, 2.f);
            }
            op_t quant {4, Quantize, "quant"};
            quant.set_attr(op_attr::scales, scales);
            quant.set_attr(op_attr::zps, zps);

            int lt_id = -1;
            logical_tensor_t int8_data
                    = logical_tensor_init(++lt_id, data_type::u8);
            logical_tensor_t fp32_data
                    = logical_tensor_init(++lt_id, data_type::f32);
            dequant1.add_input(int8_data);
            dequant1.add_output(fp32_data);

            logical_tensor_t s8_weight
                    = logical_tensor_init(++lt_id, data_type::s8);
            logical_tensor_t fp32_weight
                    = logical_tensor_init(++lt_id, data_type::f32);
            dequant2.add_input(s8_weight);
            dequant2.add_output(fp32_weight);

            logical_tensor_t fp32_bias;
            if (with_bias)
                fp32_bias = logical_tensor_init(++lt_id, data_type::f32);

            logical_tensor_t fp32_deconv_out
                    = logical_tensor_init(++lt_id, data_type::f32);
            deconv.add_input(fp32_data);
            deconv.add_input(fp32_weight);
            if (with_bias) deconv.add_input(fp32_bias);
            deconv.add_output(fp32_deconv_out);

            logical_tensor_t fp32_eltwise_out
                    = logical_tensor_init(++lt_id, data_type::f32);
            eltwise.add_input(fp32_deconv_out);
            eltwise.add_output(fp32_eltwise_out);

            logical_tensor_t int8_out
                    = logical_tensor_init(++lt_id, data_type::u8);
            quant.add_input(fp32_eltwise_out);
            quant.add_output(int8_out);

            ASSERT_EQ(agraph.add_op(&dequant1), status::success);
            ASSERT_EQ(agraph.add_op(&dequant2), status::success);
            ASSERT_EQ(agraph.add_op(&deconv), status::success);
            ASSERT_EQ(agraph.add_op(&eltwise), status::success);
            ASSERT_EQ(agraph.add_op(&quant), status::success);

            agraph.finalize();

            pass::pass_base_ptr apass
                    = get_pass("int8_convtranspose_post_ops_fusion_cpu");
            apass->run(agraph);
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::quantized_convtranspose_post_ops);

            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(),
                    with_bias ? 3U : 2U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
            if (with_bias) {
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);
            }

            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                    with_bias ? 7U : 6U);
        }
    }
}

TEST(PassSystem, FuseToInt8ConvtransposeEltwise) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            deconv w/wo bias
             | (f32)
            eltwise
             | (f32)
           quant
             | (u8/s8)
    */
    const std::vector<graph::op_kind_t> eltwise_kinds = {
            Abs,
            Clamp,
            Elu,
            Exp,
            GELU,
            Log,
            ReLU,
            Round,
            Sigmoid,
            Sqrt,
            Square,
            Tanh,
    };
    std::vector<bool> with_biases {false, true};

    for (auto &eltwise_kind : eltwise_kinds) {
        for (auto with_bias : with_biases) {
            graph_t agraph;
            std::vector<int64_t> zps = {0};
            std::vector<float> scales = {3.1f};
            op_t dequant1 {0, Dequantize, "dequant"};
            dequant1.set_attr(op_attr::scales, scales);
            dequant1.set_attr(op_attr::zps, zps);
            op_t dequant2 {1, Dequantize, "dequant"};
            dequant2.set_attr(op_attr::scales, scales);
            dequant2.set_attr(op_attr::zps, zps);
            op_t deconv {2, ConvTranspose, "deconv"};
            set_convtranspose_common_attr(deconv);
            op_t eltwise {3, eltwise_kind, "relu"};
            if (eltwise_kind == op_kind::Elu) {
                eltwise.set_attr<float>(op_attr::alpha, 1.f);
            } else if (eltwise_kind == op_kind::Clamp) {
                eltwise.set_attr<float>(op_attr::min, -1.f);
                eltwise.set_attr<float>(op_attr::max, 2.f);
            }
            op_t quant {4, Quantize, "quant"};
            quant.set_attr(op_attr::scales, scales);
            quant.set_attr(op_attr::zps, zps);

            int lt_id = -1;
            logical_tensor_t int8_data
                    = logical_tensor_init(++lt_id, data_type::u8);
            logical_tensor_t fp32_data
                    = logical_tensor_init(++lt_id, data_type::f32);
            dequant1.add_input(int8_data);
            dequant1.add_output(fp32_data);

            logical_tensor_t s8_weight
                    = logical_tensor_init(++lt_id, data_type::s8);
            logical_tensor_t fp32_weight
                    = logical_tensor_init(++lt_id, data_type::f32);
            dequant2.add_input(s8_weight);
            dequant2.add_output(fp32_weight);

            logical_tensor_t fp32_bias;
            if (with_bias)
                fp32_bias = logical_tensor_init(++lt_id, data_type::f32);

            logical_tensor_t fp32_deconv_out
                    = logical_tensor_init(++lt_id, data_type::f32);
            deconv.add_input(fp32_data);
            deconv.add_input(fp32_weight);
            if (with_bias) deconv.add_input(fp32_bias);
            deconv.add_output(fp32_deconv_out);

            logical_tensor_t fp32_eltwise_out
                    = logical_tensor_init(++lt_id, data_type::f32);
            eltwise.add_input(fp32_deconv_out);
            eltwise.add_output(fp32_eltwise_out);

            logical_tensor_t int8_out
                    = logical_tensor_init(++lt_id, data_type::u8);
            quant.add_input(fp32_eltwise_out);
            quant.add_output(int8_out);

            ASSERT_EQ(agraph.add_op(&dequant1), status::success);
            ASSERT_EQ(agraph.add_op(&dequant2), status::success);
            ASSERT_EQ(agraph.add_op(&deconv), status::success);
            ASSERT_EQ(agraph.add_op(&eltwise), status::success);
            ASSERT_EQ(agraph.add_op(&quant), status::success);

            agraph.finalize();

            auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
            auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
            pm.run_passes(agraph, "no_config");
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::quantized_convtranspose_post_ops);

            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(),
                    with_bias ? 3U : 2U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
            if (with_bias) {
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);
            }

            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                    with_bias ? 7U : 6U);
        }
    }
}

TEST(Pass, FuseToInt8ConvtransposeBinary) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)    / (f32)
            deconv w/wo bias
             | (f32)
            binary
             | (f32)
           quant
             | (u8/s8)
    */
    const std::vector<graph::op_kind_t> binary_kinds = {
            Maximum,
            Minimum,
            Divide,
            Multiply,
            Subtract,
    };
    std::vector<bool> with_biases {false, true};

    for (auto &binary_kind : binary_kinds) {
        for (auto with_bias : with_biases) {
            graph_t agraph;
            std::vector<int64_t> zps = {0};
            std::vector<float> scales = {3.1f};
            op_t dequant1 {0, Dequantize, "dequant"};
            dequant1.set_attr(op_attr::scales, scales);
            dequant1.set_attr(op_attr::zps, zps);
            op_t dequant2 {1, Dequantize, "dequant"};
            dequant2.set_attr(op_attr::scales, scales);
            dequant2.set_attr(op_attr::zps, zps);
            op_t deconv {2, ConvTranspose, "deconv"};
            set_convtranspose_common_attr(deconv);
            op_t binary {3, binary_kind, "binary"};
            op_t quant {4, Quantize, "quant"};
            quant.set_attr(op_attr::scales, scales);
            quant.set_attr(op_attr::zps, zps);

            int lt_id = -1;
            logical_tensor_t int8_data
                    = logical_tensor_init(++lt_id, data_type::u8);
            logical_tensor_t fp32_data
                    = logical_tensor_init(++lt_id, data_type::f32);
            dequant1.add_input(int8_data);
            dequant1.add_output(fp32_data);

            logical_tensor_t s8_weight
                    = logical_tensor_init(++lt_id, data_type::s8);
            logical_tensor_t fp32_weight
                    = logical_tensor_init(++lt_id, data_type::f32);
            dequant2.add_input(s8_weight);
            dequant2.add_output(fp32_weight);

            logical_tensor_t fp32_bias;
            if (with_bias)
                fp32_bias = logical_tensor_init(++lt_id, data_type::f32);

            logical_tensor_t fp32_deconv_out
                    = logical_tensor_init(++lt_id, data_type::f32);
            deconv.add_input(fp32_data);
            deconv.add_input(fp32_weight);
            if (with_bias) deconv.add_input(fp32_bias);
            deconv.add_output(fp32_deconv_out);

            logical_tensor_t fp32_binary_other
                    = logical_tensor_init(++lt_id, data_type::f32);
            logical_tensor_t fp32_binary_out
                    = logical_tensor_init(++lt_id, data_type::f32);
            binary.add_input(fp32_deconv_out);
            binary.add_input(fp32_binary_other);
            binary.add_output(fp32_binary_out);

            logical_tensor_t int8_out
                    = logical_tensor_init(++lt_id, data_type::u8);
            quant.add_input(fp32_binary_out);
            quant.add_output(int8_out);

            ASSERT_EQ(agraph.add_op(&dequant1), status::success);
            ASSERT_EQ(agraph.add_op(&dequant2), status::success);
            ASSERT_EQ(agraph.add_op(&deconv), status::success);
            ASSERT_EQ(agraph.add_op(&binary), status::success);
            ASSERT_EQ(agraph.add_op(&quant), status::success);

            agraph.finalize();

            pass::pass_base_ptr apass
                    = get_pass("int8_convtranspose_post_ops_fusion_cpu");
            apass->run(agraph);
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    partition_kind_t::quantized_convtranspose_post_ops);

            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(),
                    with_bias ? 4U : 3U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);
            if (with_bias) {
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 4U);
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[3].id, 6U);
            } else {
                ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[2].id, 5U);
            }

            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id,
                    with_bias ? 8U : 7U);
        }
    }
}

TEST(Pass, FailToFuseInt8ConcatDifferentScales) {
    /*
          dq     dq
           \     /
            concat
              |
           quantize
              |
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    std::vector<float> other_scales = {1.3f};
    op_t dequant1 {0, Dequantize, "dequant1"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant2"};
    dequant2.set_attr(op_attr::scales, other_scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t concat {2, Concat, "concat"};
    concat.set_attr<int64_t>(op_attr::axis, 1);
    op_t quant {3, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t int8_src_lt1
            = logical_tensor_init(0, {2, 3}, {3, 1}, data_type::u8);
    logical_tensor_t src_lt1
            = logical_tensor_init(1, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t int8_src_lt2
            = logical_tensor_init(2, {2, 3}, {3, 1}, data_type::u8);
    logical_tensor_t src_lt2
            = logical_tensor_init(3, {2, 3}, {3, 1}, data_type::f32);
    logical_tensor_t dst_lt = logical_tensor_init(4, data_type::f32);
    logical_tensor_t int8_dst_lt = logical_tensor_init(5, data_type::u8);
    dequant1.add_input(int8_src_lt1);
    dequant1.add_output(src_lt1);
    dequant2.add_input(int8_src_lt2);
    dequant2.add_output(src_lt2);
    concat.add_input(src_lt1);
    concat.add_input(src_lt2);
    concat.add_output(dst_lt);
    quant.add_input(dst_lt);
    quant.add_output(int8_dst_lt);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&concat), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    ASSERT_EQ(agraph.finalize(), status::success);

    pass::pass_base_ptr apass = get_pass("int8_concat_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 0U);
}

TEST(Pass, SingleSoftPlusForwardAndBackwardPass) {
    std::vector<std::pair<op_kind_t, std::string>> op_infos {
            {SoftPlus, "softplus"}, {SoftPlusBackward, "softplus_bw"}};
    std::vector<float> beta_values {-3.f, -1.f, 0.f, 1.f, 3.f};
    for_(const auto &op_info : op_infos)
    for (auto beta : beta_values) {
        const auto &op_name = op_info.second;
        auto kind = op_info.first;

        logical_tensor_t lt_in = logical_tensor_init(0, data_type::f32);
        logical_tensor_t lt_out = logical_tensor_init(1, data_type::f32);
        logical_tensor_t lt_other = logical_tensor_init(2, data_type::f32);

        op_t softplus {0, kind, op_name};
        if (kind == SoftPlusBackward) softplus.add_input(lt_other);
        softplus.add_input(lt_in);
        softplus.add_output(lt_out);
        softplus.set_attr(op_attr::beta, beta);

        graph_t agraph;
        ASSERT_EQ(agraph.add_op(&softplus), status::success);
        ASSERT_EQ(agraph.finalize(), status::success);
        pass::pass_base_ptr apass = get_pass(op_name + "_pass");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
    }
}

TEST(Pass, FuseConvBwdBiasaddBwd) {
    /*       Wildcard
        \        /\
      Convolution  BiasAddBackward
    BackwardWeights
    */
    graph_t agraph;
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    op_t op0 {0, ReLU, "op0"};
    op_t op1 {1, ConvolutionBackwardWeights, "op1"};
    set_conv_common_attr(op1);
    op_t op2 {2, BiasAddBackward, "op2"};

    op0.add_input(lt_vec[0]);
    op0.add_output(lt_vec[1]);
    op1.add_input(lt_vec[2]);
    op1.add_input(lt_vec[1]);
    op1.add_output(lt_vec[4]);
    op2.add_input(lt_vec[1]);
    op2.add_output(lt_vec[5]);

    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    pass::pass_base_ptr apass = get_pass("fp_conv_bwd_weights_bwd_bias");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convolution_backward_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    std::unordered_set<size_t> input_ids;
    input_ids.insert(agraph.get_partitions()[0]->get_inputs()[0].id);
    input_ids.insert(agraph.get_partitions()[0]->get_inputs()[1].id);
    input_ids.insert(agraph.get_partitions()[0]->get_inputs()[2].id);
    ASSERT_TRUE(input_ids.find(2) != input_ids.end());
    ASSERT_TRUE(input_ids.find(1) != input_ids.end());

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 2U);
    std::unordered_set<size_t> output_ids;
    output_ids.insert(agraph.get_partitions()[0]->get_outputs()[0].id);
    output_ids.insert(agraph.get_partitions()[0]->get_outputs()[1].id);
    ASSERT_TRUE(output_ids.find(4) != output_ids.end());
    ASSERT_TRUE(output_ids.find(5) != output_ids.end());
}

// TODO(zitian): wait for the implementation of comparison ops:
//      Gt, Ge, Le, Lt, Eq, Ne
TEST(Pass, BinaryPostops) {
    /*
        0       1
        \       /
        [Add, Multiply, Maximum, Minimum, Divide, Subtract]
            |
        [Abs, Add, Clamp, Divide, Elu, Exp, GELU, Log, Maximum, Minimum, Multiply, ReLU, Round, Sigmoid, SoftPlus, Sqrt, Square, Subtract, Tanh] * [0, 1]
    */

    std::vector<op_kind_t> supported_binary_ops {
            Add, Divide, Maximum, Minimum, Multiply, Subtract};
    std::vector<op_kind_t> supported_post_ops {
            Abs,
            Add,
            Clamp,
            Divide,
            Elu,
            Exp,
            GELU,
            HardSigmoid,
            HardSwish,
            Log,
            Maximum,
            Minimum,
            Multiply,
            ReLU,
            Round,
            Sigmoid,
            SoftPlus,
            Sqrt,
            Square,
            Subtract,
            Tanh,
    };
    std::vector<op_kind_t> supported_binary_post_ops {
            Add,
            Divide,
            Maximum,
            Minimum,
            Multiply,
            Subtract,
    };
    for_(auto bop : supported_binary_ops)
    for (auto pop : supported_post_ops) {
        auto is_post_op_binary = (std::find(supported_binary_post_ops.begin(),
                                          supported_binary_post_ops.end(), pop)
                != supported_binary_post_ops.end());
        graph_t agraph;
        op_t binary_op {0, bop, "binary op"};
        op_t post_op {1, pop, "post op"};

        // set additional parameters for specific ops
        switch (pop) {
            case Elu: post_op.set_attr<float>(op_attr::alpha, 1.0f); break;
            case Clamp:
                post_op.set_attr<float>(op_attr::min, 1.0f);
                post_op.set_attr<float>(op_attr::max, 3.0f);
                break;
            case HardSigmoid:
                post_op.set_attr<float>(op_attr::alpha, 1.0f / 6);
                post_op.set_attr<float>(op_attr::beta, 0.5f);
                break;
            default: break;
        }

        std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
        size_t lt_idx = 0;
        std::vector<size_t> input_lts = {};
        std::vector<size_t> output_lts = {};
        binary_op.add_input(lt_vec[lt_idx]);
        input_lts.push_back(lt_idx);
        binary_op.add_input(lt_vec[++lt_idx]);
        input_lts.push_back(lt_idx);
        binary_op.add_output(lt_vec[++lt_idx]);

        post_op.add_input(lt_vec[lt_idx]);
        if (is_post_op_binary) {
            post_op.add_input(lt_vec[++lt_idx]);
            input_lts.push_back(lt_idx);
        }
        post_op.add_output(lt_vec[++lt_idx]);

        output_lts.push_back(lt_idx);

        ASSERT_EQ(agraph.add_op(&binary_op), status::success);

        ASSERT_EQ(agraph.add_op(&post_op), status::success);

        agraph.finalize();

        pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        auto partition = agraph.get_partitions()[0];
        ASSERT_EQ(partition->get_kind(), partition_kind_t::binary_post_ops);

        ASSERT_EQ(partition->get_inputs().size(), input_lts.size());
        for (size_t k = 0; k < input_lts.size(); ++k)
            ASSERT_EQ(partition->get_inputs()[k].id, input_lts[k]);
        ASSERT_EQ(partition->get_outputs().size(), output_lts.size());
        for (size_t k = 0; k < output_lts.size(); ++k)
            ASSERT_EQ(partition->get_outputs()[k].id, output_lts[k]);
    }
}

// TODO(zitian): wait for the implementation of comparison ops:
//      Gt, Ge, Le, Lt, Eq, Ne
TEST(Pass, Binary3Postops) {
    /*
        0       1
        \       /
        [Add, Multiply, Maximum, Minimum, Divide, Subtract]
            |
        [Abs, Add, Clamp, Divide, Elu, Exp, GELU, Log, Maximum, Minimum, Multiply, ReLU, Round, Sigmoid, SoftPlus, Sqrt, Square, Subtract, Tanh] * [0, 3]
    */

    std::vector<op_kind_t> supported_binary_ops {
            Add, Divide, Maximum, Minimum, Multiply, Subtract};
    std::vector<op_kind_t> supported_post_ops {
            Abs,
            Add,
            Clamp,
            Divide,
            Elu,
            Exp,
            GELU,
            HardSigmoid,
            HardSwish,
            Log,
            Maximum,
            Minimum,
            Multiply,
            ReLU,
            Round,
            Sigmoid,
            SoftPlus,
            Sqrt,
            Square,
            Subtract,
            Tanh,
    };
    std::vector<op_kind_t> supported_binary_post_ops {
            Add,
            Divide,
            Maximum,
            Minimum,
            Multiply,
            Subtract,
    };

    // select several combinations of post ops
    std::vector<std::vector<op_kind_t>> post_op_seqs {
            {Abs, Subtract, Divide},
            {Round, Multiply},
            {Add, Elu},
            {Clamp, Minimum},
            {HardSigmoid, ReLU},
    };
    for_(const auto &pop_seq : post_op_seqs)
    for (const auto &pop : pop_seq)
        ASSERT_NE(std::find(supported_post_ops.begin(),
                          supported_post_ops.end(), pop),
                supported_post_ops.end());

    for_(auto bop : supported_binary_ops)
    for (const auto &pop_seq : post_op_seqs) {
        graph_t agraph;
        std::vector<logical_tensor_t> lt_vec = create_logical_tensors(9);
        size_t lt_idx = 0;
        std::vector<size_t> input_lts = {};
        std::vector<size_t> output_lts = {};

        op_t binary_op {0, bop, "binary op"};
        binary_op.add_input(lt_vec[lt_idx]);
        input_lts.push_back(lt_idx);
        binary_op.add_input(lt_vec[++lt_idx]);
        input_lts.push_back(lt_idx);
        binary_op.add_output(lt_vec[++lt_idx]);

        std::vector<op_t> post_ops {};

        for (size_t i = 0; i < pop_seq.size(); ++i) {
            auto pop = pop_seq[i];
            post_ops.emplace_back(op_t {i + 1, pop, "post op"});

            // set additional parameters for specific ops
            switch (pop) {
                case Elu:
                    post_ops.back().set_attr<float>(op_attr::alpha, 1.0f);
                    break;
                case Clamp:
                    post_ops.back().set_attr<float>(op_attr::min, 1.0f);
                    post_ops.back().set_attr<float>(op_attr::max, 3.0f);
                    break;
                case HardSigmoid:
                    post_ops.back().set_attr<float>(op_attr::alpha, 1.0f / 6);
                    post_ops.back().set_attr<float>(op_attr::beta, 0.5f);
                    break;
                default: break;
            }

            post_ops.back().add_input(lt_vec[lt_idx]);
            if (std::find(supported_binary_post_ops.begin(),
                        supported_binary_post_ops.end(), pop)
                    != supported_binary_post_ops.end()) {
                post_ops.back().add_input(lt_vec[++lt_idx]);
                input_lts.push_back(lt_idx);
            }
            post_ops.back().add_output(lt_vec[++lt_idx]);
        }

        output_lts.push_back(lt_idx);

        ASSERT_EQ(agraph.add_op(&binary_op), status::success);

        for (size_t i = 0; i < post_ops.size(); ++i)
            ASSERT_EQ(agraph.add_op(&post_ops[i]), status::success);

        agraph.finalize();

        pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        auto partition = agraph.get_partitions()[0];
        ASSERT_EQ(partition->get_kind(), partition_kind_t::binary_post_ops);

        ASSERT_EQ(partition->get_inputs().size(), input_lts.size());
        for (size_t k = 0; k < input_lts.size(); ++k)
            ASSERT_EQ(partition->get_inputs()[k].id, input_lts[k]);
        ASSERT_EQ(partition->get_outputs().size(), output_lts.size());
        for (size_t k = 0; k < output_lts.size(); ++k)
            ASSERT_EQ(partition->get_outputs()[k].id, output_lts[k]);
    }
}

TEST(PassSystem, FuseBinarySwish) {
    /*       binary
            /    |
        sigmoid  |
            \    |
            multiply
    */
    const std::vector<op_kind_t> configs {
            Add, Divide, Maximum, Minimum, Multiply, Subtract};

    for (const auto &base_op : configs) {
        op_t binary {0, base_op, "binary"};
        op_t sigmoid {1, Sigmoid, "sigmoid"};
        op_t multiply {2, Multiply, "mul"};

        logical_tensor_t binary_src0 = logical_tensor_init(0, data_type::f32);
        logical_tensor_t binary_src1 = logical_tensor_init(1, data_type::f32);
        logical_tensor_t binary_dst = logical_tensor_init(2, data_type::f32);
        logical_tensor_t sigmoid_dst = logical_tensor_init(3, data_type::f32);
        logical_tensor_t mul_dst = logical_tensor_init(4, data_type::f32);

        binary.add_input(binary_src0);
        binary.add_input(binary_src1);
        binary.add_output(binary_dst);

        sigmoid.add_input(binary_dst);
        sigmoid.add_output(sigmoid_dst);

        multiply.add_input(sigmoid_dst);
        multiply.add_input(binary_dst);
        multiply.add_output(mul_dst);

        graph_t agraph;
        ASSERT_EQ(agraph.add_op(&binary), status::success);
        ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
        ASSERT_EQ(agraph.add_op(&multiply), status::success);
        agraph.finalize();

        auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
        auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
        pm.run_passes(agraph, "no_config");
        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::binary_post_ops);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
    }
}

TEST(Pass, ConvtransposePostops) {
    /*
        0       1
        \       /
        convtranspose
            |
        addbias * [0, 1]
            |
        [Abs, Add, Clamp, Divide, Elu, Exp, GELU, Log, Maximum, Minimum, Multiply, ReLU, Round, Sigmoid, SoftPlus, Sqrt, Square, Subtract, Tanh] * [0, 1]
    */

    std::vector<bool> with_conv_bias = {true, false};
    std::vector<bool> with_post_bias = {true, false};
    std::vector<bool> with_post_activation = {true, false};
    std::vector<op_kind_t> supported_ops {
            Abs,
            Add,
            Clamp,
            Divide,
            Elu,
            Exp,
            GELU,
            HardSigmoid,
            HardSwish,
            Log,
            Maximum,
            Minimum,
            Multiply,
            ReLU,
            Round,
            Sigmoid,
            SoftPlus,
            Sqrt,
            Square,
            Subtract,
            Tanh,
    };
    std::vector<op_kind_t> supported_binary_ops {
            Add,
            Divide,
            Maximum,
            Minimum,
            Multiply,
            Subtract,
    };

    for (auto conv_bias_on : with_conv_bias)
        for (auto post_bias_on : with_post_bias)
            for (auto post_activation_on : with_post_activation)
                for (auto Activation : supported_ops) {
                    auto is_binary_activation
                            = (std::find(supported_binary_ops.begin(),
                                       supported_binary_ops.end(), Activation)
                                    != supported_binary_ops.end());

                    graph_t agraph;
                    op_t convtranspose {0, ConvTranspose, "convtranspose"};
                    set_convtranspose_common_attr(convtranspose);
                    op_t biasadd {1, BiasAdd, "biasadd"};
                    op_t activation {2, Activation, "activation"};

                    // set additional parameters for specific ops
                    switch (Activation) {
                        case Elu:
                            activation.set_attr<float>(op_attr::alpha, 1.0f);
                            break;
                        case Clamp:
                            activation.set_attr<float>(op_attr::min, 1.0f);
                            activation.set_attr<float>(op_attr::max, 3.0f);
                            break;
                        case HardSigmoid:
                            activation.set_attr<float>(
                                    op_attr::alpha, 1.0f / 6);
                            activation.set_attr<float>(op_attr::beta, 0.5f);
                            break;
                        default: break;
                    }

                    std::vector<logical_tensor_t> lt_vec
                            = create_logical_tensors(8);
                    size_t lt_idx = 0;
                    std::vector<size_t> input_lts = {};
                    std::vector<size_t> output_lts = {};
                    convtranspose.add_input(lt_vec[lt_idx]);
                    input_lts.push_back(lt_idx);
                    convtranspose.add_input(lt_vec[++lt_idx]);
                    input_lts.push_back(lt_idx);
                    if (conv_bias_on) {
                        convtranspose.add_input(lt_vec[++lt_idx]);
                        input_lts.push_back(lt_idx);
                    }
                    convtranspose.add_output(lt_vec[++lt_idx]);
                    if (post_bias_on) {
                        biasadd.add_input(lt_vec[lt_idx]);
                        biasadd.add_input(lt_vec[++lt_idx]);
                        input_lts.push_back(lt_idx);
                        biasadd.add_output(lt_vec[++lt_idx]);
                    }
                    if (post_activation_on) {
                        activation.add_input(lt_vec[lt_idx]);
                        if (is_binary_activation) {
                            activation.add_input(lt_vec[++lt_idx]);
                            input_lts.push_back(lt_idx);
                        }
                        activation.add_output(lt_vec[++lt_idx]);
                    }
                    if (conv_bias_on && post_bias_on) {
                        // a special case where the matching process
                        // should terminate before biasadd
                        input_lts = std::vector<size_t>(
                                input_lts.begin(), input_lts.begin() + 3);
                        output_lts.push_back(3);
                    } else
                        output_lts.push_back(lt_idx);

                    ASSERT_EQ(agraph.add_op(&convtranspose), status::success);
                    if (post_bias_on) {
                        ASSERT_EQ(agraph.add_op(&biasadd), status::success);
                    }
                    if (post_activation_on) {
                        ASSERT_EQ(agraph.add_op(&activation), status::success);
                    }
                    agraph.finalize();

                    pass::pass_base_ptr apass
                            = get_pass("convtranspose_post_ops_fusion");
                    apass->run(agraph);
                    ASSERT_EQ(agraph.get_num_partitions(), 1U);

                    auto partition = agraph.get_partitions()[0];
                    ASSERT_EQ(partition->get_kind(),
                            partition_kind_t::convtranspose_post_ops);

                    ASSERT_EQ(partition->get_inputs().size(), input_lts.size());
                    for (size_t k = 0; k < input_lts.size(); ++k)
                        ASSERT_EQ(partition->get_inputs()[k].id, input_lts[k]);
                    ASSERT_EQ(
                            partition->get_outputs().size(), output_lts.size());
                    for (size_t k = 0; k < output_lts.size(); ++k)
                        ASSERT_EQ(
                                partition->get_outputs()[k].id, output_lts[k]);
                }
}

TEST(Pass, Convtranspose3Postops) {
    /*
        0       1
        \       /
        convtranspose
            |
        addbias * [0, 1]
            |
        [Abs, Add, Clamp, Divide, Elu, Exp, GELU, Log, Maximum, Minimum, Multiply, ReLU, Round, Sigmoid, SoftPlus, Sqrt, Square, Subtract, Tanh] * [0, 3]
    */

    std::vector<bool> with_conv_bias = {true, false};
    std::vector<bool> with_post_bias = {true, false};
    std::vector<bool> with_post_activation = {true, false};
    std::vector<op_kind_t> supported_ops {
            Abs,
            Add,
            Clamp,
            Divide,
            Elu,
            Exp,
            GELU,
            HardSigmoid,
            HardSwish,
            Log,
            Maximum,
            Minimum,
            Multiply,
            ReLU,
            Round,
            Sigmoid,
            SoftPlus,
            Sqrt,
            Square,
            Subtract,
            Tanh,
    };
    std::vector<op_kind_t> supported_binary_ops {
            Add,
            Divide,
            Maximum,
            Minimum,
            Multiply,
            Subtract,
    };
    std::vector<std::vector<op_kind_t>> post_op_seqs {
            {Abs, Subtract, Divide},
            {Round, Multiply},
            {Add, Elu},
            {Clamp, Minimum},
            {HardSigmoid, ReLU},
    };

    for_(auto conv_bias_on : with_conv_bias)
    for_(auto post_bias_on : with_post_bias)
    for_(auto post_activation_on : with_post_activation)
    for (auto pop_seq : post_op_seqs) {
        graph_t agraph;
        op_t convtranspose {0, ConvTranspose, "convtranspose"};
        set_convtranspose_common_attr(convtranspose);
        op_t biasadd {1, BiasAdd, "biasadd"};

        std::vector<logical_tensor_t> lt_vec = create_logical_tensors(11);
        size_t lt_idx = 0;
        std::vector<size_t> input_lts = {};
        std::vector<size_t> output_lts = {};

        convtranspose.add_input(lt_vec[lt_idx]);
        input_lts.push_back(lt_idx);
        convtranspose.add_input(lt_vec[++lt_idx]);
        input_lts.push_back(lt_idx);
        if (conv_bias_on) {
            convtranspose.add_input(lt_vec[++lt_idx]);
            input_lts.push_back(lt_idx);
        }
        convtranspose.add_output(lt_vec[++lt_idx]);
        ASSERT_EQ(agraph.add_op(&convtranspose), status::success);

        if (post_bias_on) {
            biasadd.add_input(lt_vec[lt_idx]);
            biasadd.add_input(lt_vec[++lt_idx]);
            input_lts.push_back(lt_idx);
            biasadd.add_output(lt_vec[++lt_idx]);
            ASSERT_EQ(agraph.add_op(&biasadd), status::success);
        }

        if (post_activation_on) {
            for (size_t i = 0; i < pop_seq.size(); ++i) {
                auto activation_t = pop_seq[i];
                op_t activation {i + 2, activation_t, "activation"};
                // set additional parameters for specific ops
                switch (activation_t) {
                    case Elu:
                        activation.set_attr<float>(op_attr::alpha, 1.0f);
                        break;
                    case Clamp:
                        activation.set_attr<float>(op_attr::min, 1.0f);
                        activation.set_attr<float>(op_attr::max, 3.0f);
                        break;
                    case HardSigmoid:
                        activation.set_attr<float>(op_attr::alpha, 1.0f / 6);
                        activation.set_attr<float>(op_attr::beta, 0.5f);
                        break;
                    default: break;
                }

                activation.add_input(lt_vec[lt_idx]);
                if (std::find(supported_binary_ops.begin(),
                            supported_binary_ops.end(), activation_t)
                        != supported_binary_ops.end()) {
                    activation.add_input(lt_vec[++lt_idx]);
                    input_lts.push_back(lt_idx);
                }
                activation.add_output(lt_vec[++lt_idx]);
                ASSERT_EQ(agraph.add_op(&activation), status::success);
            }
        }

        if (conv_bias_on && post_bias_on) {
            // a special case where the matching process
            // should terminate before biasadd
            input_lts = std::vector<size_t>(
                    input_lts.begin(), input_lts.begin() + 3);
            output_lts.push_back(3);
        } else
            output_lts.push_back(lt_idx);

        agraph.finalize();

        pass::pass_base_ptr apass = get_pass("convtranspose_post_ops_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        auto partition = agraph.get_partitions()[0];
        ASSERT_EQ(partition->get_kind(),
                partition_kind_t::convtranspose_post_ops);

        ASSERT_EQ(partition->get_inputs().size(), input_lts.size());
        for (size_t k = 0; k < input_lts.size(); ++k)
            ASSERT_EQ(partition->get_inputs()[k].id, input_lts[k]);
        ASSERT_EQ(partition->get_outputs().size(), output_lts.size());
        for (size_t k = 0; k < output_lts.size(); ++k)
            ASSERT_EQ(partition->get_outputs()[k].id, output_lts[k]);
    }
}

TEST(PassSystem, FuseConvTransposeSwish) {
    // swish: f(x) = x * sigmoid(x)
    /*convtranspose
        /    |
    sigmoid  |
        \    |
        multiply

    */
    graph_t agraph;
    op_t convtranspose {0, ConvTranspose, "convtranspose"};
    set_convtranspose_common_attr(convtranspose);
    op_t sigmoid {1, Sigmoid, "sigmoid"};
    op_t multiply {2, Multiply, "multiply"};

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    convtranspose.add_input(lt_vec[0]);
    convtranspose.add_input(lt_vec[1]);
    convtranspose.add_output(lt_vec[2]);
    sigmoid.add_input(lt_vec[2]);
    sigmoid.add_output(lt_vec[3]);
    multiply.add_input(lt_vec[2]);
    multiply.add_input(lt_vec[3]);
    multiply.add_output(lt_vec[4]);

    ASSERT_EQ(agraph.add_op(&convtranspose), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::convtranspose_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
}

TEST(PassSystem, FuseToInt8ConvTransposeSwishReLU) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
         convtranspose
      (f32) / |
     sigmoid  |
           \  |
          multiply     
             |
            relu  
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t convtranspose {2, ConvTranspose, "convtranspose"};
    set_convtranspose_common_attr(convtranspose);
    op_t sigmoid {3, Sigmoid, "sigmoid"};
    op_t multiply {4, Multiply, "mul"};
    op_t relu {5, ReLU, "relu"};
    op_t quant {6, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);

    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_matmul_out = logical_tensor_init(4, data_type::f32);
    convtranspose.add_input(fp32_data);
    convtranspose.add_input(fp32_weight);
    convtranspose.add_output(fp32_matmul_out);

    logical_tensor_t fp32_sigmoid_out = logical_tensor_init(5, data_type::f32);
    sigmoid.add_input(fp32_matmul_out);
    sigmoid.add_output(fp32_sigmoid_out);

    logical_tensor_t fp32_mul_out = logical_tensor_init(6, data_type::f32);
    multiply.add_input(fp32_matmul_out);
    multiply.add_input(fp32_sigmoid_out);
    multiply.add_output(fp32_mul_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(7, data_type::f32);
    relu.add_input(fp32_mul_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(8, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&convtranspose), status::success);
    ASSERT_EQ(agraph.add_op(&sigmoid), status::success);
    ASSERT_EQ(agraph.add_op(&multiply), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    // run all the pass to check if the priority is correct
    auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "no_config");
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 7U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            partition_kind_t::quantized_convtranspose_post_ops);

    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 2U);

    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 8U);
}

// TODO(zitian): wait for the implementation of comparison ops:
//      Gt, Ge, Le, Lt, Eq, Ne
TEST(Pass, Pool3Postops) {
    /*
        0       1
        \       /
        [AvgPool, MaxPool]
            |
        [Add, Divide, Maximum, Minimum, Multiply, Subtract] * [0, 3]
    */

    const std::vector<op_kind_t> pooling_op_ts {AvgPool, MaxPool};
    const std::vector<op_kind_t> binary_op_ts {
            Add, Divide, Maximum, Minimum, Multiply, Subtract};
    // select several combinations of post ops
    const std::vector<std::vector<op_kind_t>> post_op_t_seqs {
            {Add, Subtract, Divide}, {Minimum, Multiply}, {Add, Maximum},
            {Divide, Minimum, Multiply}};

    for_(auto &pool_op_t : pooling_op_ts)
    for (const auto &post_op_t_seq : post_op_t_seqs) {
        graph_t agraph;
        std::vector<logical_tensor_t> lt_vec = create_logical_tensors(9);
        size_t lt_idx = 0;
        std::vector<size_t> input_lts = {};
        std::vector<size_t> output_lts = {};

        std::vector<int64_t> strides = {2, 2};
        std::vector<int64_t> pads_begin = {0, 0};
        std::vector<int64_t> pads_end = {0, 0};
        std::vector<int64_t> kernel = {2, 2};
        op_t pool_op {0, pool_op_t, "pooling op"};
        pool_op.set_attr(op_attr::strides, strides);
        pool_op.set_attr(op_attr::pads_begin, pads_begin);
        pool_op.set_attr(op_attr::pads_end, pads_end);
        pool_op.set_attr(op_attr::kernel, kernel);
        if (pool_op_t == AvgPool)
            pool_op.set_attr<bool>(op_attr::exclude_pad, false);
        pool_op.add_input(lt_vec[lt_idx]);
        input_lts.push_back(lt_idx);
        pool_op.add_output(lt_vec[++lt_idx]);

        std::vector<op_t> post_ops {};
        for (size_t i = 0; i < post_op_t_seq.size(); ++i) {
            auto pop_t = post_op_t_seq[i];
            post_ops.emplace_back(op_t {i + 1, pop_t, "post op"});

            post_ops.back().add_input(lt_vec[lt_idx]);
            if (std::find(binary_op_ts.begin(), binary_op_ts.end(), pop_t)
                    != binary_op_ts.end()) {
                post_ops.back().add_input(lt_vec[++lt_idx]);
                input_lts.push_back(lt_idx);
            }
            post_ops.back().add_output(lt_vec[++lt_idx]);
        }

        output_lts.push_back(lt_idx);

        ASSERT_EQ(agraph.add_op(&pool_op), status::success);
        for (size_t i = 0; i < post_ops.size(); ++i)
            ASSERT_EQ(agraph.add_op(&post_ops[i]), status::success);

        agraph.finalize();

        pass::pass_base_ptr apass = get_pass("fp_pool_post_ops");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);

        auto partition = agraph.get_partitions()[0];
        ASSERT_EQ(partition->get_kind(), partition_kind_t::pooling_post_ops);

        ASSERT_EQ(partition->get_inputs().size(), input_lts.size());
        for (size_t k = 0; k < input_lts.size(); ++k)
            ASSERT_EQ(partition->get_inputs()[k].id, input_lts[k]);
        ASSERT_EQ(partition->get_outputs().size(), output_lts.size());
        for (size_t k = 0; k < output_lts.size(); ++k)
            ASSERT_EQ(partition->get_outputs()[k].id, output_lts[k]);
    }
}

TEST(PassSystem, PoolFusionWithInternalInputs) {
    /*
        AvgPool/MaxPool
                |  (both inputs come from the pooling op,
               / \  the 2nd input should be an internal input)
        Add/Divide/Maximum/Minimum/Multiply/Subtract
    */
    std::vector<op_kind_t> pooling_ts = {AvgPool, MaxPool};
    std::vector<op_kind_t> binary_ts
            = {Add, Divide, Maximum, Minimum, Multiply, Subtract};
    for_(auto pool_t : pooling_ts)
    for (auto bin_t : binary_ts) {
        std::vector<int64_t> strides = {1, 1};
        std::vector<int64_t> pads_begin = {0, 0};
        std::vector<int64_t> pads_end = {0, 0};
        std::vector<int64_t> kernel = {2, 2};
        op_t pool_op {0, pool_t, "pooling"};
        pool_op.set_attr(op_attr::strides, strides);
        pool_op.set_attr(op_attr::pads_begin, pads_begin);
        pool_op.set_attr(op_attr::pads_end, pads_end);
        pool_op.set_attr(op_attr::kernel, kernel);
        if (pool_t == AvgPool) pool_op.set_attr(op_attr::exclude_pad, false);
        op_t binary_op {1, bin_t, "binary"};

        std::vector<logical_tensor_t> lt_vec = create_logical_tensors(3);
        pool_op.add_input(lt_vec[0]);
        pool_op.add_output(lt_vec[1]);
        binary_op.add_input(lt_vec[1]);
        binary_op.add_input(lt_vec[1]);
        binary_op.add_output(lt_vec[2]);

        graph_t agraph;
        ASSERT_EQ(agraph.add_op(&pool_op), status::success);
        ASSERT_EQ(agraph.add_op(&binary_op), status::success);
        agraph.finalize();

        auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
        auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 2U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::pooling_post_ops);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2U);
    }
}

TEST(PassSystem, EltwiseFusionWithInternalInputs) {
    /*
        Abs/Clamp/Elu/Exp/GELU/HardSwish/LeakyReLU/Log/
        Mish/Sigmoid/SoftPlus/ReLU/Round/Sqrt/Square/Tanh
                |  (both inputs come from the eltwise op,
               / \  the 2nd input should be an internal input)
        Add/Divide/Maximum/Minimum/Multiply/Subtract
    */
    std::vector<op_kind_t> eltwise_ts
            = {Abs, Clamp, Elu, Exp, GELU, HardSwish, LeakyReLU, Log, Mish,
                    Sigmoid, SoftPlus, ReLU, Round, Sqrt, Square, Tanh};
    std::vector<op_kind_t> binary_ts
            = {Add, Divide, Maximum, Minimum, Multiply, Subtract};
    for_(auto elt_t : eltwise_ts)
    for (auto bin_t : binary_ts) {
        op_t eltwise_op {0, elt_t, "eltwise"};
        if (elt_t == Clamp) {
            eltwise_op.set_attr<float>(op_attr::min, 1.0f);
            eltwise_op.set_attr<float>(op_attr::max, 3.0f);
        } else if (elt_t == Elu) {
            eltwise_op.set_attr<float>(op_attr::alpha, 1.0f);
        } else if (elt_t == LeakyReLU)
            eltwise_op.set_attr<float>(op_attr::alpha, 0.02f);
        op_t binary_op {1, bin_t, "binary"};

        std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
        eltwise_op.add_input(lt_vec[0]);
        eltwise_op.add_output(lt_vec[1]);
        binary_op.add_input(lt_vec[1]);
        binary_op.add_input(lt_vec[1]);
        binary_op.add_output(lt_vec[2]);

        graph_t agraph;
        ASSERT_EQ(agraph.add_op(&eltwise_op), status::success);
        ASSERT_EQ(agraph.add_op(&binary_op), status::success);
        agraph.finalize();

        auto &backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
        auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
        pm.run_passes(agraph, "no_config");

        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 2U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                partition_kind_t::unary_post_ops);

        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);

        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 2U);
    }
}

TEST(Pass, BatchNormReluU8Unfuse) {
    using dims = graph::dnnl_impl::dims;
    namespace utils = dnnl::graph::tests::unit::utils;

    std::vector<std::pair<graph::data_type_t, graph::data_type_t>> vec {
            {graph::data_type::u8, graph::data_type::s8},
            {graph::data_type::u8, graph::data_type::u8},
            {graph::data_type::s8, graph::data_type::u8},
    };
    std::vector<int64_t> zps = {0};
    std::vector<float> scales_src = {2.1f};
    std::vector<float> scales_out = {3.1f};

    // Tensor dimensions.
    const graph::dim_t N = 1, // batch size
            IC = 256, // channels
            IH = 14, // tensor height
            IW = 14; // tensor width
    // Source (src) and destination (dst) tensors dimensions.
    dims src_dims = {N, IH, IW, IC};
    // Scale/shift tensor dimensions.
    dims scale_dims = {IC};
    dims shift_dims = {IC};
    dims mean_dims = {IC};
    dims variance_dims = {IC};

    for (auto &&par : vec) {
        graph::op_t bn_op(0, graph::op_kind::BatchNormInference, "batchnorm");
        bn_op.set_attr<float>(graph::op_attr::epsilon, 0.01f);
        bn_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");

        graph::op_t dequant {1, graph::op_kind::Dequantize, "dequant"};
        dequant.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dequant.set_attr(graph::op_attr::scales, scales_src);
        dequant.set_attr(graph::op_attr::zps, zps);

        graph::op_t quant {2, graph::op_kind::Quantize, "quant"};
        quant.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        quant.set_attr(graph::op_attr::scales, scales_out);
        quant.set_attr(graph::op_attr::zps, zps);

        graph::op_t relu_op(3, graph::op_kind::ReLU, "relu");

        // prepare logical tensor
        graph::logical_tensor_t src_int8
                = utils::logical_tensor_init(0, src_dims, par.first);
        graph::logical_tensor_t src_f32 = utils::logical_tensor_init(
                1, src_dims, graph::data_type::f32);
        graph::logical_tensor_t scale = utils::logical_tensor_init(
                2, scale_dims, graph::data_type::f32);
        graph::logical_tensor_t shift = utils::logical_tensor_init(
                3, shift_dims, graph::data_type::f32);
        graph::logical_tensor_t mean = utils::logical_tensor_init(
                4, mean_dims, graph::data_type::f32);
        graph::logical_tensor_t variance = utils::logical_tensor_init(
                5, variance_dims, graph::data_type::f32);
        graph::logical_tensor_t dst = utils::logical_tensor_init(
                6, src_dims, graph::data_type::f32);
        graph::logical_tensor_t relu_dst = utils::logical_tensor_init(
                7, src_dims, graph::data_type::f32);
        graph::logical_tensor_t dst_int8_out
                = utils::logical_tensor_init(8, src_dims, par.second);

        scale.property = graph::property_type::constant;
        shift.property = graph::property_type::constant;
        mean.property = graph::property_type::constant;
        variance.property = graph::property_type::constant;

        dequant.add_input(src_int8);
        dequant.add_output(src_f32);
        bn_op.add_input(src_f32);
        bn_op.add_input(scale);
        bn_op.add_input(shift);
        bn_op.add_input(mean);
        bn_op.add_input(variance);
        bn_op.add_output(dst);
        relu_op.add_input(dst);
        relu_op.add_output(relu_dst);
        quant.add_input(relu_dst);
        quant.add_output(dst_int8_out);

        graph::graph_t g;
        EXPECT_EQ(g.add_op(&dequant), graph::status::success);
        EXPECT_EQ(g.add_op(&bn_op), graph::status::success);
        EXPECT_EQ(g.add_op(&relu_op), graph::status::success);
        EXPECT_EQ(g.add_op(&quant), graph::status::success);
        g.finalize();
        graph::pass::pass_base_ptr apass = get_pass("int8_bn_fusion");
        apass->run(g);
        EXPECT_NE(g.get_num_partitions(), 1U);
    }
}

TEST(Pass, FuseMatmulSwish) {
    const std::vector<std::string> seqs_1 {"first", "second"};
    const std::vector<std::string> seqs_2 {"left", "right"};
    for (auto seq_1 : seqs_1)
        for (auto seq_2 : seqs_2) {
            graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul");
            graph::op_t sigmoid_op(1, graph::op_kind::Sigmoid, "sigmoid");
            graph::op_t multi_op(2, graph::op_kind::Multiply, "multiply");

            // prepare logical tensor
            std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);

            matmul_op.add_input(lt_vec[0]);
            matmul_op.add_input(lt_vec[1]);
            matmul_op.add_output(lt_vec[2]);
            sigmoid_op.add_input(lt_vec[2]);
            sigmoid_op.add_output(lt_vec[3]);
            if (seq_1 == "first") {
                multi_op.add_input(lt_vec[2]);
                multi_op.add_input(lt_vec[3]);
            } else {
                multi_op.add_input(lt_vec[3]);
                multi_op.add_input(lt_vec[2]);
            }

            multi_op.add_output(lt_vec[4]);

            graph::graph_t agraph;
            agraph.add_op(&matmul_op);

            if (seq_2 == "left") {
                agraph.add_op(&sigmoid_op);
                agraph.add_op(&multi_op);
            } else {
                agraph.add_op(&multi_op);
                agraph.add_op(&sigmoid_op);
            }

            agraph.finalize();

            graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
            apass->run(agraph);
            ASSERT_EQ(agraph.get_num_partitions(), 1U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_ops().size(), 3U);
            ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                    graph::partition_kind_t::matmul_post_ops);

            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 2U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[0].id, 0U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_inputs()[1].id, 1U);

            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
            ASSERT_EQ(agraph.get_partitions()[0]->get_outputs()[0].id, 4U);
        }
}
