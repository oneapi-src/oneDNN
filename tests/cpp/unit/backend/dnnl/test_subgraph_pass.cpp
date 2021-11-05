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

#include "backend/dnnl/passes/constant_propagation.hpp"
#include "backend/dnnl/passes/infer_type.hpp"
#include "backend/dnnl/passes/insert_ops.hpp"
#include "backend/dnnl/passes/layout_propagation.hpp"
#include "backend/dnnl/passes/lower_down.hpp"
#include "backend/dnnl/passes/memory_planning.hpp"
#include "backend/dnnl/passes/op_executable.hpp"

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

using namespace dnnl::graph::impl;
using namespace dnnl::graph::impl::op_kind;
using namespace dnnl::graph::tests::unit::utils;

using op_ptr = std::shared_ptr<dnnl::graph::impl::op_t>;

namespace {
dnnl::graph::impl::pass::pass_base_ptr get_pass(const std::string &pass_name) {
    auto &backend_ptr
            = dnnl::graph::impl::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::graph::impl::pass::pass_manager_t(
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
            dnnl_impl::op_kind::int8_conv_bias_add_relu);
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
    ASSERT_EQ(producer0.get_kind(), dnnl_impl::op_kind::mul_scales);
    ASSERT_EQ(producer0.get_attr<std::vector<float>>("scales")[0], scales[0]);
    auto &producer1 = (*conv_op)->get_input_value(1)->get_producer();
    ASSERT_EQ(producer1.get_kind(), dnnl_impl::op_kind::mul_scales);
    ASSERT_EQ(producer1.get_attr<std::vector<float>>("scales")[0], scales[0]);

    // 2. merge into int8 conv, change the input's scales to output scale
    dnnl_impl::fuse_to_int8_conv_or_deconv(subgraph);
    dnnl_impl::folding_mul_scales(subgraph);
    auto qconv_op = std::find_if(subgraph.begin(), subgraph.end(),
            [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == dnnl_impl::op_kind::dnnl_convolution;
            });
    auto &consumer
            = (*qconv_op)->get_output_value(0)->get_consumers()[0].get_op();
    ASSERT_EQ(consumer.get_kind(), dnnl_impl::op_kind::mul_scales);
    ASSERT_EQ(consumer.get_attr<std::vector<float>>("scales")[0],
            scales[0] * scales[0]);

    // 3. fuse output mul_scales op to conv's output scale
    dnnl_impl::primitive_attr_mgr_t prm_attr_mgr;
    dnnl_impl::fuse_output_scales(subgraph, prm_attr_mgr);

    // 4. fuse post ops to int8 conv
    ASSERT_EQ(
            dnnl_impl::fuse_post_ops(subgraph, prm_attr_mgr), status::success);

    qconv_op = std::find_if(subgraph.begin(), subgraph.end(),
            [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == dnnl_impl::op_kind::dnnl_convolution;
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
            dnnl_impl::op_kind::int8_matmul_bias_relu);
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
    ASSERT_EQ(producer0.get_kind(), dnnl_impl::op_kind::mul_scales);
    ASSERT_EQ(producer0.get_attr<std::vector<float>>("scales")[0], scales[0]);
    auto &producer1 = (*matmul_op)->get_input_value(1)->get_producer();
    ASSERT_EQ(producer1.get_kind(), dnnl_impl::op_kind::mul_scales);
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
    ASSERT_EQ(consumer.get_kind(), dnnl_impl::op_kind::mul_scales);
    ASSERT_EQ(consumer.get_attr<std::vector<float>>("scales")[0],
            scales[0] * scales[0]);

    // 3. fuse output mul_scales op to matmul's output scale
    dnnl_impl::primitive_attr_mgr_t prm_attr_mgr;
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
                   | (f32, constant)
                 quant
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
    dqdata_node.set_attr<std::vector<int64_t>>("zps", {zp_src});
    dqdata_node.set_attr<std::vector<float>>("scales", {scale_src});
    dqdata_node.set_attr<int64_t>("axis", 0);

    impl::op_t qweight_node(10, impl::op_kind::Quantize, "qweight_node");
    qweight_node.set_attr<std::string>("qtype", "per_tensor");
    qweight_node.set_attr<std::vector<int64_t>>("zps", zp_wei);
    qweight_node.set_attr<std::vector<float>>("scales", scale_wei);
    qweight_node.set_attr<int64_t>("axis", 0);

    impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
    dqweight_node.set_attr<std::string>("qtype", "per_tensor");
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
    qout_node.set_attr<std::vector<int64_t>>("zps", {zp_out});
    qout_node.set_attr<std::vector<float>>("scales", {scale_out});
    qout_node.set_attr<int64_t>("axis", 0);

    impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
    dqother_node.set_attr<std::string>("qtype", "per_tensor");
    dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
    dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
    dqother_node.set_attr<int64_t>("axis", 0);

    impl::op_t add_node(9, impl::op_kind::Add, "add_node");

    logical_tensor_t src_u8 = logical_tensor_init(1, impl::data_type::u8);
    logical_tensor_t src_f32_dq = logical_tensor_init(2, impl::data_type::f32);
    logical_tensor_t weight_f32 = logical_tensor_init(20, impl::data_type::f32);
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

    qweight_node.add_input(weight_f32);
    qweight_node.add_output(weight_s8);

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
    g.add_op(&qweight_node);
    g.add_op(&dqweight_node);
    g.add_op(&conv_node);
    g.add_op(&dqother_node);
    g.add_op(&add_node);
    g.add_op(&relu_node);
    g.add_op(&qout_node);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("int8_quant_wei_conv_bias_add_relu_fusion");

    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    src_u8 = logical_tensor_init(1, src_shape, impl::data_type::u8);
    weight_f32 = logical_tensor_init(20, weight_shape, impl::data_type::f32);
    bias_f32 = logical_tensor_init(6, bias_shape, impl::data_type::f32);
    other_s8 = logical_tensor_init(11, dst_shape, impl::data_type::s8);
    dst_s8 = logical_tensor_init(9, dst_shape, impl::data_type::s8);

    weight_f32.property = impl::property_type::constant;
    bias_f32.property = impl::property_type::constant;

    dnnl_impl::primitive_attr_mgr_t prm_attr_mgr;
    dnnl_impl::memory_planner_t memory_planner;
    std::vector<std::shared_ptr<impl::op_t>> subgraph = part->get_ops();

    dnnl_impl::set_all_layout_to_any(subgraph);

    // run lower down passes
    dnnl_impl::check_with_bias(subgraph);
    dnnl_impl::split_quant_dequant(subgraph);
    dnnl_impl::fuse_to_int8_conv_or_deconv(subgraph);
    dnnl_impl::folding_mul_scales(subgraph);
    dnnl_impl::fuse_output_scales(subgraph, prm_attr_mgr);
    dnnl_impl::fuse_post_ops(subgraph, prm_attr_mgr);
    dnnl_impl::fuse_zero_points(subgraph, prm_attr_mgr);
    dnnl_impl::fuse_mul_scales_add_zps(subgraph);
    ASSERT_EQ(subgraph.size(), 3);
    if (subgraph[0]->get_kind() == dnnl_impl::op_kind::dnnl_convolution) {
        ASSERT_EQ(subgraph[1]->get_kind(), dnnl_impl::op_kind::mul_scales);
        ASSERT_EQ(subgraph[2]->get_kind(), op_kind::Reorder);
    } else {
        ASSERT_EQ(subgraph[0]->get_kind(), dnnl_impl::op_kind::mul_scales);
        ASSERT_EQ(
                subgraph[1]->get_kind(), dnnl_impl::op_kind::dnnl_convolution);
        ASSERT_EQ(subgraph[2]->get_kind(), op_kind::Reorder);
    }

    // insert preprocess and reorder ops
    dnnl_impl::insert_permute(subgraph);
    ASSERT_EQ(subgraph.size(), 7);

    dnnl_impl::insert_to_group_for_conv_or_deconv(subgraph);
    ASSERT_EQ(subgraph.size(), 8);

    dnnl_impl::insert_reorder(subgraph);
    ASSERT_EQ(subgraph.size(), 12);

    std::vector<logical_tensor_t> inputs
            = {src_u8, weight_f32, bias_f32, other_s8};
    std::vector<logical_tensor_t> outputs = {dst_s8};

    std::vector<logical_tensor_t> wrong_inputs = {src_u8, weight_s8, bias_f32};
    std::vector<logical_tensor_t> wrong_outputs = {};

    ASSERT_EQ(dnnl_impl::set_given_inputs_outputs(
                      subgraph, wrong_inputs, outputs),
            status::miss_ins_outs);
    ASSERT_EQ(dnnl_impl::set_given_inputs_outputs(
                      subgraph, inputs, wrong_outputs),
            status::miss_ins_outs);

    // output shape is not must
    ASSERT_EQ(dnnl_impl::set_given_inputs_outputs(subgraph, inputs,
                      {logical_tensor_init(9, impl::data_type::s8)}),
            status::success);

    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

    for (auto &val : impl::graph_t(subgraph).get_input_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    for (auto &val : impl::graph_t(subgraph).get_output_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    // infer shape/type, layout propagation and memory binding
    impl::graph_t agraph(subgraph);
    ASSERT_EQ(agraph.infer_shape(), impl::status::success);
    ASSERT_EQ(dnnl_impl::infer_type(agraph), impl::status::success);

    dnnl_impl::pd_cache_t pd_cache;
    ASSERT_EQ(dnnl_impl::layout_propagation(
                      subgraph, p_eng, prm_attr_mgr, pd_cache),
            impl::status::success);

    for (auto &cur_op : subgraph) {
        for (auto &val : cur_op->get_input_values()) {
            auto lt = val->get_logical_tensor();
            impl::logical_tensor_wrapper_t ltw(lt);
            ASSERT_FALSE(ltw.is_shape_unknown());
            ASSERT_NE(ltw.layout_type(), layout_type::undef);
            ASSERT_NE(ltw.layout_type(), layout_type::any);
            ASSERT_NE(ltw.data_type(), data_type::undef);
        }
        size_t idx = 0;
        for (auto &val : cur_op->get_output_values()) {
            auto lt = val->get_logical_tensor();
            impl::logical_tensor_wrapper_t ltw(lt);

            // skip shape and dtype check for conv's scratchpad output
            if (!(cur_op->get_kind() == dnnl_impl::op_kind::dnnl_convolution
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

    dnnl_impl::constant_propagation(subgraph);

    ASSERT_EQ(
            memory_planner.run(subgraph, inputs, outputs, p_eng, prm_attr_mgr),
            impl::status::success);

    ASSERT_GE(memory_planner.total_internal_persistent_size(), 0);
    ASSERT_GE(memory_planner.total_internal_temporary_size(), 0);

    // only the final weight and bias used by conv are cached
    auto cached_mem_offkeys = memory_planner.get_exec_args_set()
                                      .get_mems_use_internal_persistent();
    std::set<size_t> unique_offkeys;
    for (auto &mem_offkey : cached_mem_offkeys) {
        unique_offkeys.insert(mem_offkey.second);
    }
    ASSERT_EQ(unique_offkeys.size(), 2);

    std::vector<impl::op_t *> topo_ordered_ops;
    dnnl::graph::impl::topo_order_visit(
            impl::graph_t(subgraph).get_output_ops(), [&](impl::op_t *op) {
                topo_ordered_ops.emplace_back(op);
                return status::success;
            });

    auto topo_ordered_args = memory_planner.get_exec_args_set().get_exec_args();

    ASSERT_EQ(topo_ordered_ops.size(), topo_ordered_args.size());

    for (size_t i = 0; i < topo_ordered_args.size(); i++) {
        std::unordered_map<int, dnnl::memory> exec_arg = topo_ordered_args[i];
        ASSERT_FALSE(exec_arg.empty());

        auto cur_op = topo_ordered_ops[i];
        if (cur_op->get_kind() == dnnl_impl::op_kind::dnnl_convolution) {
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
    size_t subgraph_size_after_insertion;
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
            dnnl_impl::op_kind::int8_matmul_bias_relu);
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
    dnnl_impl::primitive_attr_mgr_t prm_attr_mgr;
    dnnl_impl::fuse_output_scales(subgraph, prm_attr_mgr);
    dnnl_impl::fuse_post_ops(subgraph, prm_attr_mgr);
    dnnl_impl::fuse_zero_points(subgraph, prm_attr_mgr);
    dnnl_impl::fuse_mul_scales_add_zps(subgraph);
    ASSERT_EQ(subgraph.size(), 2);

    impl::graph_t(subgraph).infer_shape();
    dnnl_impl::insert_transpose_for_matmul(subgraph);
    impl::graph_t(subgraph).infer_shape();
    dnnl_impl::insert_expand_and_squeeze_for_matmul(subgraph);
    dnnl_impl::insert_reorder(subgraph);
    ASSERT_EQ(subgraph.size(), params.subgraph_size_after_insertion);

    for (auto &val : impl::graph_t(subgraph).get_input_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    for (auto &val : impl::graph_t(subgraph).get_output_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    impl::graph_t g(subgraph);
    ASSERT_EQ(g.infer_shape(), impl::status::success);
    ASSERT_EQ(dnnl_impl::infer_type(g), impl::status::success);

    if (params.constant_weight) {
        dnnl_impl::set_weight_bias_constant(subgraph);
        dnnl_impl::constant_propagation(subgraph);
    }

    dnnl_impl::pd_cache_t pd_cache;
    ASSERT_EQ(dnnl_impl::layout_propagation(
                      subgraph, p_eng, prm_attr_mgr, pd_cache),
            impl::status::success);
    ASSERT_EQ(subgraph.size(), params.final_subgraph_size);
}

INSTANTIATE_TEST_SUITE_P(int8_matmul_test_instance, int8_matmul_pass_test,
        testing::Values(ut_int8_matmul_params {{1, 1024}, {1000, 1024}, {1000},
                                {1, 1000}, false, true, false, 8, 5},
                ut_int8_matmul_params {{1, 1024}, {1000, 1024}, {1000},
                        {1, 1000}, false, true, true, 8, 5},
                ut_int8_matmul_params {{4, 3, 64}, {3, 64}, {3}, {4, 3, 3},
                        false, true, false, 9, 6},
                ut_int8_matmul_params {{4, 3, 64}, {3, 64}, {3}, {4, 3, 3},
                        false, true, true, 9, 6}));

TEST(pass_test, subgraph_execution_args_set) {
    ///////////////////////////
    // val1    val2
    //   \     /
    //    \   /
    //     op1
    //      |
    //     val3   val4
    //       \    /
    //        \  /
    //         op2
    //          |
    //         val5
    ///////////////////////////
    using value_t = impl::value_t;
    using dtype = dnnl::memory::data_type;
    using ftag = dnnl::memory::format_tag;
    using engine = dnnl::engine;
    using exec_args = impl::dnnl_impl::exec_args;
    using execution_args_set = impl::dnnl_impl::execution_args_set_t;

    value_t *val1 = (value_t *)1;
    value_t *val2 = (value_t *)2;
    value_t *val3 = (value_t *)3;
    value_t *val4 = (value_t *)4;
    value_t *val5 = (value_t *)5;

    engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::memory mem1({{1, 2, 3, 4}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem2({{2, 3, 4, 5}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem3({{3, 4, 5, 6}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem4({{4, 5, 6, 7}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem5({{5, 6, 7, 8}, dtype::f32, ftag::abcd}, eng, nullptr);

    // construct the execution_args_set
    execution_args_set exec_args_set;
    exec_args_set.add_value_mem_map({val1, mem1});
    exec_args_set.add_value_mem_map({val2, mem2});
    exec_args_set.add_value_mem_map({val3, mem3});
    exec_args_set.add_value_mem_map({val4, mem4});
    exec_args_set.add_value_mem_map({val5, mem5});

    exec_args_set.add_mem_use_external_inputs(std::make_pair(mem1, 0));
    exec_args_set.add_mem_use_external_inputs(std::make_pair(mem2, 1));
    exec_args_set.add_mem_use_external_inputs(std::make_pair(mem4, 2));

    exec_args_set.add_mem_use_external_outputs(std::make_pair(mem5, 0));

    exec_args_set.add_mem_use_internal_temporary(std::make_pair(mem3, 0));

    exec_args op1_args;
    op1_args.insert({DNNL_ARG_SRC_0, mem1});
    op1_args.insert({DNNL_ARG_SRC_1, mem2});
    op1_args.insert({DNNL_ARG_DST, mem3});
    exec_args_set.add_exec_args(op1_args);

    exec_args op2_args;
    op2_args.insert({DNNL_ARG_SRC_0, mem3});
    op2_args.insert({DNNL_ARG_SRC_1, mem4});
    op2_args.insert({DNNL_ARG_DST, mem5});
    exec_args_set.add_exec_args(op2_args);

    // create the subgraph (will deep copy the exec_args_mgr implicitly)
    auto cloned_exec_args_set_ptr = exec_args_set.clone();
    const auto &cloned_exec_args_set = *cloned_exec_args_set_ptr;

    dnnl::memory cloned_mem1, cloned_mem2, cloned_mem3, cloned_mem4,
            cloned_mem5;
    ASSERT_TRUE(cloned_exec_args_set.find_value_mem_map(val1, cloned_mem1));
    ASSERT_TRUE(cloned_exec_args_set.find_value_mem_map(val2, cloned_mem2));
    ASSERT_TRUE(cloned_exec_args_set.find_value_mem_map(val3, cloned_mem3));
    ASSERT_TRUE(cloned_exec_args_set.find_value_mem_map(val4, cloned_mem4));
    ASSERT_TRUE(cloned_exec_args_set.find_value_mem_map(val5, cloned_mem5));

    // because of deep copy, the desc should be same but the address should be
    // different
    ASSERT_TRUE(cloned_mem1.get_desc() == mem1.get_desc()
            && cloned_mem1.get() != mem1.get());
    ASSERT_TRUE(cloned_mem2.get_desc() == mem2.get_desc()
            && cloned_mem2.get() != mem2.get());
    ASSERT_TRUE(cloned_mem3.get_desc() == mem3.get_desc()
            && cloned_mem3.get() != mem3.get());
    ASSERT_TRUE(cloned_mem4.get_desc() == mem4.get_desc()
            && cloned_mem4.get() != mem4.get());
    ASSERT_TRUE(cloned_mem5.get_desc() == mem5.get_desc()
            && cloned_mem5.get() != mem5.get());

    // the external mems and internal mems are just alias to the mem object in
    // val-mem map, so both of their desc and address should be same
    auto mems_use_external_inputs
            = cloned_exec_args_set.get_mems_use_external_inputs();
    ASSERT_TRUE(cloned_mem1.get_desc()
                    == mems_use_external_inputs[0].first.get_desc()
            && cloned_mem1.get() == mems_use_external_inputs[0].first.get());
    ASSERT_TRUE(cloned_mem2.get_desc()
                    == mems_use_external_inputs[1].first.get_desc()
            && cloned_mem2.get() == mems_use_external_inputs[1].first.get());
    ASSERT_TRUE(cloned_mem4.get_desc()
                    == mems_use_external_inputs[2].first.get_desc()
            && cloned_mem4.get() == mems_use_external_inputs[2].first.get());

    auto mems_use_external_outputs
            = cloned_exec_args_set.get_mems_use_external_outputs();
    ASSERT_TRUE(cloned_mem5.get_desc()
                    == mems_use_external_outputs[0].first.get_desc()
            && cloned_mem5.get() == mems_use_external_outputs[0].first.get());

    auto mems_use_internal_variables
            = cloned_exec_args_set.get_mems_use_internal_temporary();
    ASSERT_TRUE(cloned_mem3.get_desc()
                    == mems_use_internal_variables[0].first.get_desc()
            && cloned_mem3.get() == mems_use_internal_variables[0].first.get());

    auto args = cloned_exec_args_set.get_exec_args();

    // the mems in args should also be alias
    auto cloned_op1_args = args[0];
    ASSERT_TRUE(
            cloned_mem1.get_desc() == cloned_op1_args[DNNL_ARG_SRC_0].get_desc()
            && cloned_mem1.get() == cloned_op1_args[DNNL_ARG_SRC_0].get());
    ASSERT_TRUE(
            cloned_mem2.get_desc() == cloned_op1_args[DNNL_ARG_SRC_1].get_desc()
            && cloned_mem2.get() == cloned_op1_args[DNNL_ARG_SRC_1].get());
    ASSERT_TRUE(
            cloned_mem3.get_desc() == cloned_op1_args[DNNL_ARG_DST].get_desc()
            && cloned_mem3.get() == cloned_op1_args[DNNL_ARG_DST].get());

    auto cloned_op2_args = args[1];
    ASSERT_TRUE(
            cloned_mem3.get_desc() == cloned_op2_args[DNNL_ARG_SRC_0].get_desc()
            && cloned_mem3.get() == cloned_op2_args[DNNL_ARG_SRC_0].get());
    ASSERT_TRUE(
            cloned_mem4.get_desc() == cloned_op2_args[DNNL_ARG_SRC_1].get_desc()
            && cloned_mem4.get() == cloned_op2_args[DNNL_ARG_SRC_1].get());
    ASSERT_TRUE(
            cloned_mem5.get_desc() == cloned_op2_args[DNNL_ARG_DST].get_desc()
            && cloned_mem5.get() == cloned_op2_args[DNNL_ARG_DST].get());
}

TEST(pass_test, memory_planning) {
    /*
                / -> Reorder -> Reorder
               /
    mul_scales -> mul_scales -> permute -> mul_scales -> permute -> mul_scales
    -> mul_scales
    */
    using dims = impl::dnnl_impl::dims;

    dnnl::engine p_eng(dnnl::engine::kind::cpu, 0);

    std::vector<int64_t> shape_NCX {64, 32, 256, 256};
    std::vector<int64_t> shape_NXC {64, 256, 256, 32};

    impl::op_t op1(1, dnnl_impl::op_kind::mul_scales, "op1");
    impl::op_t op2(2, dnnl_impl::op_kind::mul_scales, "op2");
    impl::op_t op3(3, dnnl_impl::op_kind::permute, "op3");
    impl::op_t op4(4, dnnl_impl::op_kind::mul_scales, "op4");
    impl::op_t op5(5, dnnl_impl::op_kind::permute, "op5");
    impl::op_t op6(6, dnnl_impl::op_kind::mul_scales, "op6");
    impl::op_t op7(7, dnnl_impl::op_kind::mul_scales, "op7");
    impl::op_t op8(8, impl::op_kind::Reorder, "op8");
    impl::op_t op9(9, impl::op_kind::Reorder, "op9");

    op1.set_attr<std::vector<float>>("scales", {0.5});
    op2.set_attr<std::vector<float>>("scales", {0.5});
    op4.set_attr<std::vector<float>>("scales", {0.5});
    op6.set_attr<std::vector<float>>("scales", {0.5});
    op7.set_attr<std::vector<float>>("scales", {0.5});

    logical_tensor_t val0
            = logical_tensor_init(0, shape_NCX, impl::data_type::f32);
    logical_tensor_t val1
            = logical_tensor_init(1, shape_NCX, impl::data_type::f32);
    logical_tensor_t val2
            = logical_tensor_init(2, shape_NCX, impl::data_type::f32);
    logical_tensor_t val3
            = logical_tensor_init(3, shape_NXC, impl::data_type::f32);
    logical_tensor_t val4
            = logical_tensor_init(4, shape_NXC, impl::data_type::f32);
    logical_tensor_t val5
            = logical_tensor_init(5, shape_NCX, impl::data_type::f32);
    logical_tensor_t val6
            = logical_tensor_init(6, shape_NCX, impl::data_type::f32);
    logical_tensor_t val7
            = logical_tensor_init(7, shape_NCX, impl::data_type::f32);
    logical_tensor_t val8
            = logical_tensor_init(8, shape_NCX, impl::data_type::f32);
    logical_tensor_t val9
            = logical_tensor_init(9, shape_NCX, impl::data_type::f32);

    op1.add_input(val0);
    op1.add_output(val1);
    op2.add_input(val1);
    op2.add_output(val2);
    op3.add_input(val2);
    op3.add_output(val3);
    op4.add_input(val3);
    op4.add_output(val4);
    op5.add_input(val4);
    op5.add_output(val5);
    op6.add_input(val5);
    op6.add_output(val6);
    op7.add_input(val6);
    op7.add_output(val7);
    op8.add_input(val1);
    op8.add_output(val8);
    op9.add_input(val8);
    op9.add_output(val9);

    impl::graph_t g;
    g.add_op(&op1);
    g.add_op(&op2);
    g.add_op(&op3);
    g.add_op(&op4);
    g.add_op(&op5);
    g.add_op(&op6);
    g.add_op(&op7);
    g.add_op(&op8);
    g.add_op(&op9);
    g.build_graph();

    auto subgraph = g.get_ops();
    ASSERT_EQ(subgraph.size(), 9);

    std::vector<logical_tensor_t> inputs = {val0};
    std::vector<logical_tensor_t> outputs = {val7, val9};

    // the prm_attr_mgr is dummy here
    dnnl_impl::primitive_attr_mgr_t prm_attr_mgr;
    dnnl_impl::memory_planner_t memory_planner;

    ASSERT_EQ(
            memory_planner.run(subgraph, inputs, outputs, p_eng, prm_attr_mgr),
            impl::status::success);

    auto mem_offkeys = memory_planner.get_exec_args_set()
                               .get_mems_use_internal_temporary();
    std::set<size_t> unique_offkeys;
    for (auto &mem_offkey : mem_offkeys) {
        unique_offkeys.insert(mem_offkey.second);
    }

    // 2 buffers for val1, (val2 or val8)
    // - val1, val2 can't share buffer because val2 has two consumer and
    //   mul_scales will change the buffer content
    // - if compute reorder first, then val2 can reuse the val8 or val1 buffer.
    //   otherwise, val8 can reuse the val2 buffer
    ASSERT_EQ(unique_offkeys.size(), 2);
}
