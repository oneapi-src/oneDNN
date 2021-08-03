/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"

#include "backend/dnnl/subgraph/constant_propagation.hpp"
#include "backend/dnnl/subgraph/infer_type.hpp"
#include "backend/dnnl/subgraph/layout_propagation.hpp"
#include "backend/dnnl/subgraph/memory_binding.hpp"
#include "backend/dnnl/subgraph/op_executable.hpp"
#include "backend/dnnl/subgraph/passes.hpp"

#include "unit_test_common.hpp"
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

TEST(pass_test, int8_conv_lower_down_pass) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
            sum
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    graph_t agraph;
    std::vector<int64_t> zps {0};
    std::vector<float> scales {0.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t dequant_other {3, Dequantize, "dequant"};
    dequant_other.set_attr("scales", scales);
    dequant_other.set_attr("zps", zps);
    op_t sum {4, Add, "sum"};
    op_t relu {5, ReLU, "relu"};
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

    logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(5, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t s8_other = logical_tensor_init(6, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(7, data_type::f32);
    dequant_other.add_input(s8_other);
    dequant_other.add_output(fp32_other);

    logical_tensor_t fp32_sum_out = logical_tensor_init(8, data_type::f32);
    sum.add_input(fp32_conv_out);
    sum.add_input(fp32_other);
    sum.add_output(fp32_sum_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(9, data_type::f32);
    relu.add_input(fp32_sum_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&dequant_other), status::success);
    ASSERT_EQ(agraph.add_op(&sum), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_conv_bias_add_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_conv_bias_add_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4);

    std::vector<std::shared_ptr<op_t>> subgraph
            = agraph.get_partitions()[0]->get_ops();
    ASSERT_EQ(subgraph.size(), 7);

    dnnl_impl::split_quant_dequant(subgraph);
    ASSERT_EQ(subgraph.size(), 11);
    auto conv_op = std::find_if(subgraph.begin(), subgraph.end(),
            [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == op_kind::Convolution;
            });
    auto &producer0 = (*conv_op)->get_input_value(0)->get_producer();
    ASSERT_EQ(producer0.get_kind(), op_kind::mul_scales);
    ASSERT_EQ(producer0.get_attr<std::vector<float>>("scales")[0], scales[0]);
    auto &producer1 = (*conv_op)->get_input_value(1)->get_producer();
    ASSERT_EQ(producer1.get_kind(), op_kind::mul_scales);
    ASSERT_EQ(producer1.get_attr<std::vector<float>>("scales")[0], scales[0]);

    // 2. merge into int8 conv, change the input's scales to output scale
    dnnl_impl::fuse_to_int8_conv(subgraph);
    dnnl_impl::folding_mul_scales(subgraph);
    auto qconv_op = std::find_if(subgraph.begin(), subgraph.end(),
            [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == op_kind::dnnl_convolution;
            });
    auto &consumer
            = (*qconv_op)->get_output_value(0)->get_consumers()[0].get_op();
    ASSERT_EQ(consumer.get_kind(), op_kind::mul_scales);
    ASSERT_EQ(consumer.get_attr<std::vector<float>>("scales")[0],
            scales[0] * scales[0]);

    // 3. fuse output mul_scales op to conv's output scale
    dnnl_impl::primitive_attr_mgr prm_attr_mgr;
    dnnl_impl::fuse_output_scales(subgraph, prm_attr_mgr);

    // 4. fuse post ops to int8 conv
    ASSERT_EQ(
            dnnl_impl::fuse_post_ops(subgraph, prm_attr_mgr), status::success);

    qconv_op = std::find_if(subgraph.begin(), subgraph.end(),
            [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == op_kind::dnnl_convolution;
            });
    ASSERT_TRUE((*qconv_op)->has_attr("primitive_attr_key"));
    int64_t key = (*qconv_op)->get_attr<int64_t>("primitive_attr_key");
    dnnl::primitive_attr &prm_attr = prm_attr_mgr.get_attr(key);
    auto post_ops = prm_attr.get_post_ops();
    ASSERT_EQ(post_ops.len(), 2);
}

TEST(pass_test, int8_matmul_lower_down_pass) {
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
    std::vector<int64_t> zps {0};
    std::vector<float> scales {0.5f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t matmul {2, MatMul, "matmul"};
    matmul.set_attr<bool>("transpose_a", false);
    matmul.set_attr<bool>("transpose_b", false);
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

    pass::pass_base_ptr apass = get_pass("int8_matmul_bias_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_matmul_bias_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);

    std::vector<std::shared_ptr<op_t>> subgraph
            = agraph.get_partitions()[0]->get_ops();
    ASSERT_EQ(subgraph.size(), 5);

    dnnl_impl::split_quant_dequant(subgraph);
    ASSERT_EQ(subgraph.size(), 8);
    auto matmul_op = std::find_if(subgraph.begin(), subgraph.end(),
            [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == op_kind::MatMul;
            });
    auto &producer0 = (*matmul_op)->get_input_value(0)->get_producer();
    ASSERT_EQ(producer0.get_kind(), op_kind::mul_scales);
    ASSERT_EQ(producer0.get_attr<std::vector<float>>("scales")[0], scales[0]);
    auto &producer1 = (*matmul_op)->get_input_value(1)->get_producer();
    ASSERT_EQ(producer1.get_kind(), op_kind::mul_scales);
    ASSERT_EQ(producer1.get_attr<std::vector<float>>("scales")[0], scales[0]);

    // 2. merge into int8 matmul, change the input's scales to output scale
    dnnl_impl::fuse_to_int8_matmul(subgraph);
    dnnl_impl::folding_mul_scales(subgraph);
    auto qmatmul_op = std::find_if(subgraph.begin(), subgraph.end(),
            [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == op_kind::MatMul;
            });
    auto &consumer
            = (*qmatmul_op)->get_output_value(0)->get_consumers()[0].get_op();
    ASSERT_EQ(consumer.get_kind(), op_kind::mul_scales);
    ASSERT_EQ(consumer.get_attr<std::vector<float>>("scales")[0],
            scales[0] * scales[0]);

    // 3. fuse output mul_scales op to matmul's output scale
    dnnl_impl::primitive_attr_mgr prm_attr_mgr;
    dnnl_impl::fuse_output_scales(subgraph, prm_attr_mgr);

    // 4. fuse post ops to int8 matmul
    ASSERT_EQ(
            dnnl_impl::fuse_post_ops(subgraph, prm_attr_mgr), status::success);

    qmatmul_op = std::find_if(subgraph.begin(), subgraph.end(),
            [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == op_kind::MatMul;
            });
    ASSERT_TRUE((*qmatmul_op)->has_attr("primitive_attr_key"));
    int64_t key = (*qmatmul_op)->get_attr<int64_t>("primitive_attr_key");
    dnnl::primitive_attr &prm_attr = prm_attr_mgr.get_attr(key);
    auto post_ops = prm_attr.get_post_ops();
    ASSERT_EQ(post_ops.len(), 1);
}

TEST(pass_test, subgraph_passes) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
            sum
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    using dims = impl::dnnl_impl::dims;

    dnnl::engine p_eng(dnnl::engine::kind::cpu, 0);

    int64_t groups = 4;

    int64_t in_channel = 8, out_channel = 8;
    int64_t kernel_size = 3;
    std::vector<int64_t> src_shape {1, 112, 112, in_channel};
    std::vector<int64_t> weight_shape {
            kernel_size, kernel_size, in_channel / groups, out_channel};
    std::vector<int64_t> bias_shape {out_channel};
    std::vector<int64_t> dst_shape {1, 110, 110, out_channel};

    float scale_src = 1 / 255.f;
    float scale_other = 1 / 127.f;
    float scale_out = 1;
    int64_t zp_src = 0;
    int64_t zp_other = 0;
    int64_t zp_out = 78;
    std::vector<float> scale_wei(1, 1 / 127.f);
    std::vector<int64_t> zp_wei(1, 0);

    impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
    dqdata_node.set_attr<std::string>("qtype", "per_tensor");
    dqdata_node.set_attr<std::string>("out_type", "uint8");
    dqdata_node.set_attr<std::vector<int64_t>>("zps", {zp_src});
    dqdata_node.set_attr<std::vector<float>>("scales", {scale_src});
    dqdata_node.set_attr<int64_t>("axis", 0);

    impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
    dqweight_node.set_attr<std::string>("qtype", "per_tensor");
    dqweight_node.set_attr<std::string>("out_type", "int8");
    dqweight_node.set_attr<std::vector<int64_t>>("zps", zp_wei);
    dqweight_node.set_attr<std::vector<float>>("scales", scale_wei);
    dqweight_node.set_attr<int64_t>("axis", 0);

    impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
    conv_node.set_attr<dims>("strides", dims(2, 1));
    conv_node.set_attr<dims>("dilations", dims(2, 1));
    conv_node.set_attr<dims>("pads_begin", dims(2, 0));
    conv_node.set_attr<dims>("pads_end", dims(2, 0));
    conv_node.set_attr<int64_t>("groups", groups);
    conv_node.set_attr<std::string>("data_format", "NXC");
    conv_node.set_attr<std::string>("filter_format", "XIO");

    impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

    impl::op_t qout_node(6, impl::op_kind::Quantize, "qout_node");
    qout_node.set_attr<std::string>("qtype", "per_tensor");
    qout_node.set_attr<std::string>("out_type", "int8");
    qout_node.set_attr<std::vector<int64_t>>("zps", {zp_out});
    qout_node.set_attr<std::vector<float>>("scales", {scale_out});
    qout_node.set_attr<int64_t>("axis", 0);

    impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
    dqother_node.set_attr<std::string>("qtype", "per_tensor");
    dqother_node.set_attr<std::string>("in_type", "int8");
    dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
    dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
    dqother_node.set_attr<int64_t>("axis", 0);

    impl::op_t add_node(9, impl::op_kind::Add, "add_node");

    logical_tensor_t src_u8 = logical_tensor_init(1, impl::data_type::u8);
    logical_tensor_t src_f32_dq = logical_tensor_init(2, impl::data_type::f32);
    logical_tensor_t weight_s8 = logical_tensor_init(4, impl::data_type::s8);
    logical_tensor_t weight_f32_dq
            = logical_tensor_init(5, impl::data_type::f32);
    logical_tensor_t dst_f32 = logical_tensor_init(7, impl::data_type::f32);
    logical_tensor_t dst_relu_f32
            = logical_tensor_init(8, impl::data_type::f32);
    logical_tensor_t dst_s8 = logical_tensor_init(9, impl::data_type::s8);
    logical_tensor_t other_s8 = logical_tensor_init(11, impl::data_type::s8);
    logical_tensor_t other_f32_dq
            = logical_tensor_init(12, impl::data_type::f32);
    logical_tensor_t dst_add_f32
            = logical_tensor_init(13, impl::data_type::f32);
    logical_tensor_t bias_f32 = logical_tensor_init(6, impl::data_type::f32);

    dqdata_node.add_input(src_u8);
    dqdata_node.add_output(src_f32_dq);

    dqweight_node.add_input(weight_s8);
    dqweight_node.add_output(weight_f32_dq);

    conv_node.add_input(src_f32_dq);
    conv_node.add_input(weight_f32_dq);
    conv_node.add_input(bias_f32);
    conv_node.add_output(dst_f32);

    dqother_node.add_input(other_s8);
    dqother_node.add_output(other_f32_dq);

    add_node.add_input(dst_f32);
    add_node.add_input(other_f32_dq);
    add_node.add_output(dst_add_f32);

    relu_node.add_input(dst_add_f32);
    relu_node.add_output(dst_relu_f32);

    qout_node.add_input(dst_relu_f32);
    qout_node.add_output(dst_s8);

    impl::graph_t g;
    g.add_op(&dqdata_node);
    g.add_op(&dqweight_node);
    g.add_op(&conv_node);
    g.add_op(&dqother_node);
    g.add_op(&add_node);
    g.add_op(&relu_node);
    g.add_op(&qout_node);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("int8_conv_bias_add_relu_fusion");

    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    src_u8 = logical_tensor_init(1, src_shape, impl::data_type::u8);
    weight_s8 = logical_tensor_init(4, weight_shape, impl::data_type::s8);
    bias_f32 = logical_tensor_init(6, bias_shape, impl::data_type::f32);
    other_s8 = logical_tensor_init(11, dst_shape, impl::data_type::s8);
    dst_s8 = logical_tensor_init(9, dst_shape, impl::data_type::s8);

    dnnl_impl::primitive_attr_mgr prm_attr_mgr;
    dnnl_impl::execution_args_mgr exec_arg_mgr;
    std::vector<std::shared_ptr<impl::op_t>> subgraph = part->get_ops();

    // run lower down passes
    dnnl_impl::check_with_bias(subgraph);
    dnnl_impl::split_quant_dequant(subgraph);
    dnnl_impl::fuse_to_int8_conv(subgraph);
    dnnl_impl::folding_mul_scales(subgraph);
    dnnl_impl::fuse_output_scales(subgraph, prm_attr_mgr);
    dnnl_impl::fuse_post_ops(subgraph, prm_attr_mgr);
    dnnl_impl::fuse_zero_points(subgraph, prm_attr_mgr);
    ASSERT_EQ(subgraph.size(), 2);
    if (subgraph[0]->get_kind() == op_kind::dnnl_convolution) {
        ASSERT_EQ(subgraph[1]->get_kind(), op_kind::mul_scales);
    } else {
        ASSERT_EQ(subgraph[0]->get_kind(), op_kind::mul_scales);
        ASSERT_EQ(subgraph[1]->get_kind(), op_kind::dnnl_convolution);
    }

    // insert preprocess and reorder ops
    dnnl_impl::insert_permute(subgraph);
    ASSERT_EQ(subgraph.size(), 5);

    dnnl_impl::insert_to_group_for_conv(subgraph);
    ASSERT_EQ(subgraph.size(), 6);

    dnnl_impl::insert_reorder(subgraph);
    ASSERT_EQ(subgraph.size(), 10);

    std::vector<logical_tensor_t> inputs
            = {src_u8, weight_s8, bias_f32, other_s8};
    std::vector<logical_tensor_t> outputs = {dst_s8};

    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

    for (auto &val : impl::graph_t(subgraph).get_input_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper(lt).is_shape_unknown());
    }

    for (auto &val : impl::graph_t(subgraph).get_output_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper(lt).is_shape_unknown());
    }

    // infer shape/type, layout propagation and memory binding
    impl::graph_t agraph(subgraph);
    ASSERT_EQ(agraph.infer_shape(), impl::status::success);
    ASSERT_EQ(dnnl_impl::infer_type(agraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph, p_eng, prm_attr_mgr),
            impl::status::success);

    for (auto &cur_op : subgraph) {
        for (auto &val : cur_op->get_input_values()) {
            auto lt = val->get_logical_tensor();
            impl::logical_tensor_wrapper ltw(lt);
            ASSERT_FALSE(ltw.is_shape_unknown());
            ASSERT_NE(ltw.layout_type(), layout_type::undef);
            ASSERT_NE(ltw.layout_type(), layout_type::any);
            ASSERT_NE(ltw.data_type(), data_type::undef);
        }
        size_t idx = 0;
        for (auto &val : cur_op->get_output_values()) {
            auto lt = val->get_logical_tensor();
            impl::logical_tensor_wrapper ltw(lt);

            // skip shape and dtype check for conv's scratchpad output
            if (!(cur_op->get_kind() == op_kind::dnnl_convolution
                        && idx == cur_op->num_outputs() - 1)) {
                ASSERT_FALSE(ltw.is_shape_unknown());
                ASSERT_NE(ltw.data_type(), data_type::undef);
                ASSERT_NE(ltw.layout_type(), layout_type::undef);
                ASSERT_NE(ltw.layout_type(), layout_type::any);
            } else {
                ASSERT_EQ(ltw.layout_type(), layout_type::opaque);
            }

            idx++;
        }
    }

    ASSERT_EQ(dnnl_impl::memory_binding(subgraph, inputs, outputs, p_eng,
                      exec_arg_mgr, prm_attr_mgr),
            impl::status::success);
    for (auto &cur_op : subgraph) {
        ASSERT_TRUE(cur_op->has_attr("execution_args_key"));
        int64_t key = cur_op->get_attr<int64_t>("execution_args_key");
        std::unordered_map<int, dnnl::memory> exec_arg
                = exec_arg_mgr.get_args(key);
        ASSERT_FALSE(exec_arg.empty());

        if (cur_op->get_kind() == op_kind::dnnl_convolution) {
            ASSERT_NE(exec_arg.find(DNNL_ARG_SRC), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_WEIGHTS), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_BIAS), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_GRAPH_ARG_POST_SRC), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_DST), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_SCRATCHPAD), exec_arg.end());
        } else {
            ASSERT_NE(exec_arg.find(DNNL_ARG_FROM), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_TO), exec_arg.end());
        }
    }
}

struct ut_int8_matmul_params {
    std::vector<impl::dim_t> src_shape;
    std::vector<impl::dim_t> weight_shape;
    std::vector<impl::dim_t> bias_shape;
    std::vector<impl::dim_t> dst_shape;
    bool transpose_a;
    bool transpose_b;
    bool constant_weight;
    size_t final_subgraph_size;
};

class int8_matmul_pass_test
    : public ::testing::TestWithParam<ut_int8_matmul_params> {};

TEST_P(int8_matmul_pass_test, int8_matmul_layout_propagation) {
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
    const auto &params = GetParam();

    graph_t agraph;
    dnnl::engine p_eng(dnnl::engine::kind::cpu, 0);
    std::vector<int64_t> zps {0};
    std::vector<float> scales {0.5f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr("scales", scales);
    dequant1.set_attr("zps", zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr("scales", scales);
    dequant2.set_attr("zps", zps);
    op_t matmul {2, MatMul, "matmul"};
    matmul.set_attr<bool>("transpose_a", params.transpose_a);
    matmul.set_attr<bool>("transpose_b", params.transpose_b);
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

    pass::pass_base_ptr apass = get_pass("int8_matmul_bias_relu_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    ASSERT_EQ(get_fused_op(agraph.get_partitions()[0])->get_kind(),
            int8_matmul_bias_relu);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3);

    std::vector<std::shared_ptr<op_t>> subgraph
            = agraph.get_partitions()[0]->get_ops();
    ASSERT_EQ(subgraph.size(), 5);

    dnnl_impl::set_all_layout_to_any(subgraph);
    dnnl_impl::check_with_bias(subgraph);

    int8_data = logical_tensor_init(0, params.src_shape, impl::data_type::u8);
    s8_weight
            = logical_tensor_init(2, params.weight_shape, impl::data_type::s8);
    fp32_bias = logical_tensor_init(4, params.bias_shape, impl::data_type::f32);
    int8_out = logical_tensor_init(7, params.dst_shape, impl::data_type::u8);

    std::vector<logical_tensor_t> inputs = {int8_data, s8_weight, fp32_bias};
    std::vector<logical_tensor_t> outputs = {int8_out};

    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

    dnnl_impl::split_quant_dequant(subgraph);
    dnnl_impl::fuse_to_int8_matmul(subgraph);
    dnnl_impl::folding_mul_scales(subgraph);
    dnnl_impl::primitive_attr_mgr prm_attr_mgr;
    dnnl_impl::fuse_output_scales(subgraph, prm_attr_mgr);
    dnnl_impl::fuse_post_ops(subgraph, prm_attr_mgr);
    dnnl_impl::fuse_zero_points(subgraph, prm_attr_mgr);
    dnnl_impl::fuse_mul_scales_add_zps(subgraph);
    ASSERT_EQ(subgraph.size(), 2);

    impl::graph_t(subgraph).infer_shape();
    dnnl_impl::insert_transpose_for_matmul(subgraph);
    impl::graph_t(subgraph).infer_shape();
    dnnl_impl::insert_expand_for_matmul(subgraph);
    dnnl_impl::insert_reorder(subgraph);
    ASSERT_EQ(subgraph.size(), 10);

    for (auto &val : impl::graph_t(subgraph).get_input_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper(lt).is_shape_unknown());
    }

    for (auto &val : impl::graph_t(subgraph).get_output_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper(lt).is_shape_unknown());
    }

    impl::graph_t g(subgraph);
    ASSERT_EQ(g.infer_shape(), impl::status::success);
    ASSERT_EQ(dnnl_impl::infer_type(g), impl::status::success);

    if (params.constant_weight) {
        dnnl_impl::set_weight_bias_constant(subgraph);
        dnnl_impl::constant_propagation(subgraph);
    }
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph, p_eng, prm_attr_mgr),
            impl::status::success);
    ASSERT_EQ(subgraph.size(), params.final_subgraph_size);
}

INSTANTIATE_TEST_SUITE_P(int8_matmul_test_instance, int8_matmul_pass_test,
        testing::Values(ut_int8_matmul_params {{1, 1024}, {1000, 1024}, {1000},
                                {1, 1000}, false, true, false, 6},
                ut_int8_matmul_params {{1, 1024}, {1000, 1024}, {1000},
                        {1, 1000}, false, true, true, 7},
                ut_int8_matmul_params {{4, 3, 64}, {3, 64}, {3}, {4, 3, 3},
                        false, true, false, 6},
                ut_int8_matmul_params {{4, 3, 64}, {3, 64}, {3}, {4, 3, 3},
                        false, true, true, 7}));
