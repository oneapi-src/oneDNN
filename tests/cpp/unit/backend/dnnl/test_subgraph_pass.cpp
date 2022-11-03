/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "backend/dnnl/kernels/large_partition.hpp"
#include "backend/dnnl/op_executable.hpp"
#include "backend/dnnl/passes/constant_propagation.hpp"
#include "backend/dnnl/passes/insert_ops.hpp"
#include "backend/dnnl/passes/layout_propagation.hpp"
#include "backend/dnnl/passes/lower.hpp"
#include "backend/dnnl/passes/memory_planning.hpp"
#include "backend/dnnl/passes/transform.hpp"

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
} // namespace

TEST(SubgraphPass, LowerDownToInt8Conv) {
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
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t dequant_other {3, Dequantize, "dequant"};
    dequant_other.set_attr(op_attr::scales, scales);
    dequant_other.set_attr(op_attr::zps, zps);
    op_t sum {4, Add, "sum"};
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

    pass::pass_base_ptr apass
            = get_pass("int8_conv_post_ops_int8_add_fusion_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            impl::partition_kind::quantized_convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops());
    ASSERT_EQ(subgraph->get_ops().size(), 7U);

    // lower the binary ops in oneDNN Graph (Add, Mul, ...) to DNNL backend
    // internal ops and canonicalize them
    int8_data = logical_tensor_init(0, {1, 112, 112, 8}, data_type::u8);
    s8_weight = logical_tensor_init(2, {3, 3, 8, 8}, data_type::s8);
    fp32_bias = logical_tensor_init(4, {8}, data_type::f32);
    s8_other = logical_tensor_init(6, {1, 110, 110, 8}, data_type::u8);

    dnnl_impl::set_given_inputs_outputs(
            subgraph, {int8_data, s8_weight, fp32_bias, s8_other}, {int8_out});

    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    validator.run(subgraph); // validate and set default param

    ASSERT_EQ(subgraph->get_ops().size(), 11U);
    auto conv_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == dnnl_impl::op_kind::dnnl_convolution;
            });
    ASSERT_NE(conv_op, subgraph->get_ops().end());
    auto &producer0 = (*conv_op)->get_input_value(0)->get_producer();
    ASSERT_EQ(producer0.get_kind(), dnnl_impl::op_kind::dnnl_mul_scales);
    ASSERT_EQ(producer0.get_attr<std::vector<float>>(op_attr::scales)[0],
            scales[0]);
    auto &producer1 = (*conv_op)->get_input_value(1)->get_producer();
    ASSERT_EQ(producer1.get_kind(), dnnl_impl::op_kind::dnnl_mul_scales);
    ASSERT_EQ(producer1.get_attr<std::vector<float>>(op_attr::scales)[0],
            scales[0]);

    // 2. merge into int8 conv, change the input's scales to output scale
    dnnl_impl::fuse_to_int8_conv_or_deconv(subgraph);
    dnnl_impl::fold_mul_scales(subgraph);
    auto qconv_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == dnnl_impl::op_kind::dnnl_convolution;
            });
    auto &consumer
            = (*qconv_op)->get_output_value(0)->get_consumers()[0].get_op();
    ASSERT_EQ(consumer.get_kind(), dnnl_impl::op_kind::dnnl_mul_scales);
    ASSERT_EQ(consumer.get_attr<std::vector<float>>(op_attr::scales)[0],
            scales[0] * scales[0]);

    // 3. fuse output mul_scales op to conv's output scale
    dnnl_impl::fuse_output_scales(subgraph);

    dnnl_impl::infer_shape(subgraph);
    dnnl_impl::binary_canonicalization(subgraph);
    dnnl_impl::infer_shape(subgraph);

    // 4. fuse post ops to int8 conv
    ASSERT_EQ(dnnl_impl::fuse_post_ops(subgraph), status::success);

    qconv_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == dnnl_impl::op_kind::dnnl_convolution;
            });
    ASSERT_TRUE((*qconv_op)->has_attr(dnnl_impl::op_attr::fusion_info_key));
    int64_t key = (*qconv_op)->get_attr<int64_t>(
            dnnl_impl::op_attr::fusion_info_key);
    auto &fusion_info = subgraph->fusion_info_mgr_.get_info(key);
    auto post_ops = fusion_info.get_post_ops();
    ASSERT_EQ(post_ops.size(), 2U);
}

TEST(SubgraphPass, LowerDownToInt8Matmul) {
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
    std::vector<int64_t> src_shape {1, 1024};
    std::vector<int64_t> weight_shape {1024, 512};
    std::vector<int64_t> bias_shape {1};
    std::vector<int64_t> dst_shape {1, 512};

    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    matmul.set_attr<bool>(op_attr::transpose_a, false);
    matmul.set_attr<bool>(op_attr::transpose_b, false);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);
    logical_tensor_t int8_data
            = logical_tensor_init(0, src_shape, data_type::u8);
    logical_tensor_t fp32_data
            = logical_tensor_init(1, src_shape, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight
            = logical_tensor_init(2, weight_shape, data_type::s8);
    logical_tensor_t fp32_weight
            = logical_tensor_init(3, weight_shape, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_bias
            = logical_tensor_init(4, bias_shape, data_type::f32);
    logical_tensor_t fp32_matmul_out
            = logical_tensor_init(5, bias_shape, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t fp32_relu_out
            = logical_tensor_init(6, dst_shape, data_type::f32);
    relu.add_input(fp32_matmul_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out
            = logical_tensor_init(7, dst_shape, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_matmul_post_ops_fusion_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            impl::partition_kind::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops());
    ASSERT_EQ(subgraph->get_ops().size(), 5U);

    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    validator.run(subgraph); // validate and set default param

    ASSERT_EQ(subgraph->get_ops().size(), 8U);
    auto matmul_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == dnnl_impl::op_kind::dnnl_matmul;
            });
    auto &producer0 = (*matmul_op)->get_input_value(0)->get_producer();
    ASSERT_EQ(producer0.get_kind(), dnnl_impl::op_kind::dnnl_mul_scales);
    ASSERT_EQ(producer0.get_attr<std::vector<float>>(op_attr::scales)[0],
            scales[0]);
    auto &producer1 = (*matmul_op)->get_input_value(1)->get_producer();
    ASSERT_EQ(producer1.get_kind(), dnnl_impl::op_kind::dnnl_mul_scales);
    ASSERT_EQ(producer1.get_attr<std::vector<float>>(op_attr::scales)[0],
            scales[0]);

    // 2. merge into int8 matmul, change the input's scales to output scale
    dnnl_impl::fuse_to_int8_matmul(subgraph);
    dnnl_impl::fold_mul_scales(subgraph);
    auto qmatmul_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == dnnl_impl::op_kind::dnnl_matmul;
            });
    auto &consumer
            = (*qmatmul_op)->get_output_value(0)->get_consumers()[0].get_op();
    ASSERT_EQ(consumer.get_kind(), dnnl_impl::op_kind::dnnl_mul_scales);
    ASSERT_EQ(consumer.get_attr<std::vector<float>>(op_attr::scales)[0],
            scales[0] * scales[0]);

    // 3. fuse output mul_scales op to matmul's output scale
    dnnl_impl::fuse_output_scales(subgraph);

    // 4. fuse post ops to int8 matmul
    ASSERT_EQ(dnnl_impl::fuse_post_ops(subgraph), status::success);

    qmatmul_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> op) {
                return op->get_kind() == dnnl_impl::op_kind::dnnl_matmul;
            });
    ASSERT_TRUE((*qmatmul_op)->has_attr(dnnl_impl::op_attr::fusion_info_key));
    int64_t key
            = (*qmatmul_op)
                      ->get_attr<int64_t>(dnnl_impl::op_attr::fusion_info_key);
    auto &fusion_info = subgraph->fusion_info_mgr_.get_info(key);
    auto post_ops = fusion_info.get_post_ops();
    ASSERT_EQ(post_ops.size(), 1U);
}

TEST(SubgraphPass, Int8ConvSumRelu) {
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

    impl::engine_t &eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(eng);

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
    dqdata_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    dqdata_node.set_attr<std::vector<int64_t>>(op_attr::zps, {zp_src});
    dqdata_node.set_attr<std::vector<float>>(op_attr::scales, {scale_src});
    dqdata_node.set_attr<int64_t>(op_attr::axis, 0);

    impl::op_t qweight_node(10, impl::op_kind::Quantize, "qweight_node");
    qweight_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    qweight_node.set_attr<std::vector<int64_t>>(op_attr::zps, zp_wei);
    qweight_node.set_attr<std::vector<float>>(op_attr::scales, scale_wei);
    qweight_node.set_attr<int64_t>(op_attr::axis, 0);

    impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
    dqweight_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    dqweight_node.set_attr<std::vector<int64_t>>(op_attr::zps, zp_wei);
    dqweight_node.set_attr<std::vector<float>>(op_attr::scales, scale_wei);
    dqweight_node.set_attr<int64_t>(op_attr::axis, 0);

    impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
    conv_node.set_attr<dims>(op_attr::strides, dims(2, 1));
    conv_node.set_attr<dims>(op_attr::dilations, dims(2, 1));
    conv_node.set_attr<dims>(op_attr::pads_begin, dims(2, 0));
    conv_node.set_attr<dims>(op_attr::pads_end, dims(2, 0));
    conv_node.set_attr<int64_t>(op_attr::groups, groups);
    conv_node.set_attr<std::string>(op_attr::data_format, "NXC");
    conv_node.set_attr<std::string>(op_attr::filter_format, "XIO");

    impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

    impl::op_t qout_node(6, impl::op_kind::Quantize, "qout_node");
    qout_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    qout_node.set_attr<std::vector<int64_t>>(op_attr::zps, {zp_out});
    qout_node.set_attr<std::vector<float>>(op_attr::scales, {scale_out});
    qout_node.set_attr<int64_t>(op_attr::axis, 0);

    impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
    dqother_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    dqother_node.set_attr<std::vector<int64_t>>(op_attr::zps, {zp_other});
    dqother_node.set_attr<std::vector<float>>(op_attr::scales, {scale_other});
    dqother_node.set_attr<int64_t>(op_attr::axis, 0);

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
            = get_pass("int8_conv_post_ops_int8_add_fusion_cpu");

    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    src_u8 = logical_tensor_init(1, src_shape, impl::data_type::u8);
    weight_f32 = logical_tensor_init(20, weight_shape, impl::data_type::f32);
    bias_f32 = logical_tensor_init(6, bias_shape, impl::data_type::f32);
    other_s8 = logical_tensor_init(11, dst_shape, impl::data_type::s8);
    dst_s8 = logical_tensor_init(9, dst_shape, impl::data_type::s8);

    weight_f32.property = impl::property_type::constant;
    bias_f32.property = impl::property_type::constant;

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            part->get_ops(), p_eng, impl::fpmath_mode::strict, false, true);

    std::vector<logical_tensor_t> inputs
            = {src_u8, weight_f32, bias_f32, other_s8};
    std::vector<logical_tensor_t> outputs = {dst_s8};

    std::vector<logical_tensor_t> wrong_inputs = {src_u8, weight_s8, bias_f32};
    std::vector<logical_tensor_t> wrong_outputs = {};

    ASSERT_EQ(dnnl_impl::set_given_inputs_outputs(
                      subgraph, wrong_inputs, outputs),
            status::invalid_arguments);
    ASSERT_EQ(dnnl_impl::set_given_inputs_outputs(
                      subgraph, inputs, wrong_outputs),
            status::invalid_arguments);

    // output shape is not must
    ASSERT_EQ(dnnl_impl::set_given_inputs_outputs(subgraph, inputs,
                      {logical_tensor_init(
                              9, impl::data_type::s8, impl::layout_type::any)}),
            status::success);

    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

    for (auto &val : subgraph->get_input_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    for (auto &val : subgraph->get_output_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    validator.run(subgraph); // validate and set default param

    dnnl_impl::infer_shape(subgraph);
    dnnl_impl::binary_canonicalization(subgraph);
    dnnl_impl::infer_shape(subgraph);

    // run lower down passes
    dnnl_impl::check_with_bias(subgraph);
    dnnl_impl::fuse_to_int8_conv_or_deconv(subgraph);
    dnnl_impl::fold_mul_scales(subgraph);
    dnnl_impl::fuse_output_scales(subgraph);
    dnnl_impl::fuse_post_ops(subgraph);
    dnnl_impl::fuse_zero_points(subgraph);
    dnnl_impl::fuse_static_mul_scales_add_zps(subgraph);
    dnnl_impl::fuse_static_sub_zps_mul_scales(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), 3U);
    if (subgraph->get_ops()[0]->get_kind()
            == dnnl_impl::op_kind::dnnl_convolution) {
        ASSERT_EQ(subgraph->get_ops()[1]->get_kind(),
                dnnl_impl::op_kind::dnnl_mul_scales);
        ASSERT_EQ(subgraph->get_ops()[2]->get_kind(),
                dnnl_impl::op_kind::dnnl_reorder);
    } else {
        ASSERT_EQ(subgraph->get_ops()[0]->get_kind(),
                dnnl_impl::op_kind::dnnl_mul_scales);
        ASSERT_EQ(subgraph->get_ops()[1]->get_kind(),
                dnnl_impl::op_kind::dnnl_convolution);
        ASSERT_EQ(subgraph->get_ops()[2]->get_kind(),
                dnnl_impl::op_kind::dnnl_reorder);
    }

    // insert preprocess and reorder ops
    dnnl_impl::insert_permute_for_conv_or_deconv(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), 7U);

    dnnl_impl::insert_to_group_for_conv_or_deconv(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), 8U);

    // infer shape/type, layout propagation and memory binding
    ASSERT_EQ(subgraph->infer_shape(), impl::status::success);

    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), impl::status::success);

    // since we insert Reorder ops during layout propagation, here need
    // do shape inference and type inference again
    ASSERT_EQ(subgraph->infer_shape(), impl::status::success);

    for (auto &cur_op : subgraph->get_ops()) {
        for (auto &val : cur_op->get_input_values()) {
            auto lt = val->get_logical_tensor();
            impl::logical_tensor_wrapper_t ltw(lt);
            ASSERT_FALSE(ltw.is_shape_unknown());
            ASSERT_NE(ltw.layout_type(), layout_type::undef);
            ASSERT_NE(ltw.layout_type(), layout_type::any);
            //     ASSERT_NE(ltw.data_type(), data_type::undef);
        }

        auto lt = cur_op->get_output_value(0)->get_logical_tensor();
        impl::logical_tensor_wrapper_t ltw(lt);

        ASSERT_FALSE(ltw.is_shape_unknown());
        // inserted reorder's logical tensor is not set to new data type
        // since didn't do type inference after layout propagation
        // ASSERT_NE(ltw.data_type(), data_type::undef);
        ASSERT_NE(ltw.layout_type(), layout_type::undef);
        ASSERT_NE(ltw.layout_type(), layout_type::any);
    }

    dnnl_impl::constant_propagation(subgraph);

    dnnl_impl::memory_planner_t memory_planner;
    ASSERT_EQ(memory_planner.run(subgraph), impl::status::success);

    ASSERT_GE(memory_planner.total_internal_persistent_size(), 0U);
    ASSERT_GE(memory_planner.total_internal_temporary_size(), 0U);

    // only the final weight and bias used by conv are cached
    auto cached_mem_offkeys = memory_planner.get_exec_args_set()
                                      .get_mems_use_internal_persistent();
    std::set<size_t> unique_offkeys;
    for (auto &mem_offkey : cached_mem_offkeys) {
        unique_offkeys.insert(mem_offkey.second);
    }
    ASSERT_EQ(unique_offkeys.size(), 2U);

    std::vector<impl::op_t *> topo_ordered_ops;
    dnnl::graph::impl::topo_order_visit(
            subgraph->get_output_ops(), [&](impl::op_t *op) {
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

struct ut_matmul_params {
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

class TestInt8MatmulPassesWithDiffInputs
    : public ::testing::TestWithParam<ut_matmul_params> {};

TEST_P(TestInt8MatmulPassesWithDiffInputs, Int8MatmulPasses) {
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
    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);
    std::vector<int64_t> zps {0};
    std::vector<float> scales {0.5f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    matmul.set_attr<bool>(op_attr::transpose_a, params.transpose_a);
    matmul.set_attr<bool>(op_attr::transpose_b, params.transpose_b);
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

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("int8_matmul_post_ops_fusion_cpu");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            impl::partition_kind::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops(), p_eng,
            impl::fpmath_mode::strict, false, true);
    ASSERT_EQ(subgraph->get_ops().size(), 5U);

    dnnl_impl::check_with_bias(subgraph);

    int8_data = logical_tensor_init(0, params.src_shape, impl::data_type::u8);
    s8_weight
            = logical_tensor_init(2, params.weight_shape, impl::data_type::s8);
    fp32_bias = logical_tensor_init(4, params.bias_shape, impl::data_type::f32);
    int8_out = logical_tensor_init(7, params.dst_shape, impl::data_type::u8);

    std::vector<logical_tensor_t> inputs = {int8_data, s8_weight, fp32_bias};
    std::vector<logical_tensor_t> outputs = {int8_out};

    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    validator.run(subgraph); // validate and set default param

    dnnl_impl::fuse_to_int8_matmul(subgraph);
    dnnl_impl::fold_mul_scales(subgraph);
    dnnl_impl::fuse_output_scales(subgraph);
    dnnl_impl::fuse_post_ops(subgraph);
    dnnl_impl::fuse_zero_points(subgraph);
    dnnl_impl::fuse_static_mul_scales_add_zps(subgraph);
    dnnl_impl::fuse_static_sub_zps_mul_scales(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), 2U);

    subgraph->infer_shape();
    dnnl_impl::insert_permute_for_matmul(subgraph);
    subgraph->infer_shape();
    dnnl_impl::insert_unsqueeze_and_squeeze_for_matmul(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), params.subgraph_size_after_insertion);

    for (auto &val : subgraph->get_input_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    for (auto &val : subgraph->get_output_values()) {
        auto lt = val->get_logical_tensor();
        if (lt.id == std::numeric_limits<size_t>::max()) continue;
        ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    ASSERT_EQ(subgraph->infer_shape(), impl::status::success);

    if (params.constant_weight) {
        dnnl_impl::set_weight_bias_constant(subgraph);
        dnnl_impl::constant_propagation(subgraph);
    }

    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), impl::status::success);
    ASSERT_EQ(subgraph->get_ops().size(), params.final_subgraph_size);
}

INSTANTIATE_TEST_SUITE_P(SubgraphPass, TestInt8MatmulPassesWithDiffInputs,
        testing::Values(ut_matmul_params {{1, 1024}, {1000, 1024}, {1000},
                                {1, 1000}, false, true, false, 4, 5},
                ut_matmul_params {{1, 1024}, {1000, 1024}, {1, 1000}, {1, 1000},
                        false, true, false, 3, 4},
                ut_matmul_params {{1, 1024}, {1000, 1024}, {1000}, {1, 1000},
                        false, true, true, 4, 5},
                ut_matmul_params {{1024, 1000}, {1024, 1000}, {1024},
                        {1024, 1024}, false, true, true, 4, 5},
                ut_matmul_params {{4, 3, 64}, {3, 64}, {3}, {4, 3, 3}, false,
                        true, false, 5, 6},
                ut_matmul_params {{4, 3, 64}, {3, 64}, {3}, {4, 3, 3}, false,
                        true, true, 5, 6},
                ut_matmul_params {{4, 3, 64}, {3, 64}, {1, 1, 1}, {4, 3, 3},
                        false, true, true, 4, 5},
                ut_matmul_params {{4, 3, 64}, {3, 64}, {4, 1, 1}, {4, 3, 3},
                        false, true, true, 4, 5},
                ut_matmul_params {{4, 3, 64}, {3, 64}, {1, 3, 1}, {4, 3, 3},
                        false, true, true, 4, 5},
                ut_matmul_params {{4, 3, 64}, {3, 64}, {1, 1, 3}, {4, 3, 3},
                        false, true, true, 4, 5},
                ut_matmul_params {{4, 3, 64}, {3, 64}, {4, 1, 3}, {4, 3, 3},
                        false, true, true, 4, 5},
                ut_matmul_params {{4, 3, 64}, {3, 64}, {4, 3, 3}, {4, 3, 3},
                        false, true, true, 4, 5}));

class TestMatmulPassesWithDiffInputs
    : public ::testing::TestWithParam<ut_matmul_params> {};

TEST_P(TestMatmulPassesWithDiffInputs, MatmulPasses) {
    /*
    (f32) \     / (f32)
            matmul
             | (f32)
            relu
             | (f32)
    */
    const auto &params = GetParam();

    graph_t agraph;
    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);
    op_t matmul {0, MatMul, "matmul"};
    matmul.set_attr<bool>(op_attr::transpose_a, params.transpose_a);
    matmul.set_attr<bool>(op_attr::transpose_b, params.transpose_b);
    op_t relu {1, ReLU, "relu"};

    logical_tensor_t fp32_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t fp32_weight = logical_tensor_init(1, data_type::f32);
    logical_tensor_t fp32_bias = logical_tensor_init(2, data_type::f32);
    logical_tensor_t fp32_matmul_out = logical_tensor_init(3, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(4, data_type::f32);
    relu.add_input(fp32_matmul_out);
    relu.add_output(fp32_relu_out);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("matmul_bias_post_ops_chain_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            impl::partition_kind::matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops(), p_eng,
            impl::fpmath_mode::strict, false, true);
    ASSERT_EQ(subgraph->get_ops().size(), 2U);

    dnnl_impl::check_with_bias(subgraph);

    fp32_data = logical_tensor_init(0, params.src_shape, impl::data_type::f32);
    fp32_weight
            = logical_tensor_init(1, params.weight_shape, impl::data_type::f32);
    fp32_bias = logical_tensor_init(2, params.bias_shape, impl::data_type::f32);
    fp32_relu_out
            = logical_tensor_init(4, params.dst_shape, impl::data_type::f32);

    std::vector<logical_tensor_t> inputs = {fp32_data, fp32_weight, fp32_bias};
    std::vector<logical_tensor_t> outputs = {fp32_relu_out};

    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    validator.run(subgraph); // validate and set default param

    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);
    subgraph->infer_shape();
    dnnl_impl::insert_permute_for_matmul(subgraph);
    subgraph->infer_shape();
    dnnl_impl::insert_reshape_for_ndx2d_matmul(subgraph);
    subgraph->infer_shape();
    dnnl_impl::insert_unsqueeze_and_squeeze_for_matmul(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), params.subgraph_size_after_insertion);

    for (auto &val : subgraph->get_input_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    for (auto &val : subgraph->get_output_values()) {
        auto lt = val->get_logical_tensor();
        if (lt.id == std::numeric_limits<size_t>::max()) continue;
        ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    ASSERT_EQ(subgraph->infer_shape(), impl::status::success);

    if (params.constant_weight) {
        dnnl_impl::set_weight_bias_constant(subgraph);
        dnnl_impl::constant_propagation(subgraph);
    }

    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), impl::status::success);
    auto final_subgraph_size = params.final_subgraph_size;
    if (params.transpose_a && p_eng.get_kind() == dnnl::engine::kind::gpu) {
        final_subgraph_size -= 1;
    }
    ASSERT_EQ(subgraph->get_ops().size(), final_subgraph_size);
}

INSTANTIATE_TEST_SUITE_P(SubgraphPass, TestMatmulPassesWithDiffInputs,
        testing::Values(ut_matmul_params {{1, 1024}, {1000, 1024}, {1000},
                                {1, 1000}, false, true, false, 4, 5},
                ut_matmul_params {{4, 3, 64}, {3, 64}, {3}, {4, 3, 3}, false,
                        true, false, 6, 7},
                ut_matmul_params {{4, 64, 3}, {3, 64}, {3}, {4, 3, 3}, true,
                        true, false, 6, 8}));

TEST(SubgraphPass, ExecutionArgsSet) {
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

    impl::engine_t &g_eng = get_engine();
    engine eng = impl::dnnl_impl::make_dnnl_engine(g_eng);
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

TEST(SubgraphPass, MemoryPlanning) {
    /*
                / -> dnnl_reorder -> dnnl_reorder
               /
    mul_scales -> mul_scales -> permute -> mul_scales -> permute -> mul_scales
    -> mul_scales
    */
    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);

    std::vector<int64_t> shape_NCX {64, 32, 256, 256};
    std::vector<int64_t> shape_NXC {64, 256, 256, 32};

    impl::op_t op1(1, dnnl_impl::op_kind::dnnl_mul_scales, "op1");
    impl::op_t op2(2, dnnl_impl::op_kind::dnnl_mul_scales, "op2");
    impl::op_t op3(3, dnnl_impl::op_kind::dnnl_permute, "op3");
    impl::op_t op4(4, dnnl_impl::op_kind::dnnl_mul_scales, "op4");
    impl::op_t op5(5, dnnl_impl::op_kind::dnnl_permute, "op5");
    impl::op_t op6(6, dnnl_impl::op_kind::dnnl_mul_scales, "op6");
    impl::op_t op7(7, dnnl_impl::op_kind::dnnl_mul_scales, "op7");
    impl::op_t op8(8, dnnl_impl::op_kind::dnnl_reorder, "op8");
    impl::op_t op9(9, dnnl_impl::op_kind::dnnl_reorder, "op9");

    op1.set_attr<std::vector<float>>(op_attr::scales, {0.5});
    op2.set_attr<std::vector<float>>(op_attr::scales, {0.5});
    op4.set_attr<std::vector<float>>(op_attr::scales, {0.5});
    op6.set_attr<std::vector<float>>(op_attr::scales, {0.5});
    op7.set_attr<std::vector<float>>(op_attr::scales, {0.5});

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

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(g.get_ops(), p_eng,
            impl::fpmath_mode::strict, false, /* reset_layout */ false);
    ASSERT_EQ(subgraph->get_ops().size(), 9U);

    std::vector<logical_tensor_t> inputs = {val0};
    std::vector<logical_tensor_t> outputs = {val7, val9};
    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

    // the fusion_info_mgr is dummy here
    dnnl_impl::memory_planner_t memory_planner;

    ASSERT_EQ(memory_planner.run(subgraph), impl::status::success);

    auto mem_offkeys = memory_planner.get_exec_args_set()
                               .get_mems_use_internal_temporary();
    ASSERT_TRUE(mem_offkeys.empty());
}

TEST(SubgraphPass, FusePostOpsForConvDepthwise) {
    /*   conv
          |
         conv (depthwise)
    */
    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);

    // N, IC, IH, IW
    std::vector<int64_t> conv_src_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    std::vector<int64_t> conv_wei_shape {4, 4, 1, 1};
    // N, OC, OH, OW
    std::vector<int64_t> conv_dst_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    std::vector<int64_t> dw_wei_shape {4, 1, 3, 3};
    // N, OC, OH, OW
    std::vector<int64_t> dw_dst_shape {4, 4, 2, 2};

    impl::op_t conv {0, impl::op_kind::Convolution, "conv"};
    set_conv_dw_base_op_attr(conv);

    impl::op_t depthwise {1, impl::op_kind::Convolution, "depthwise"};
    set_conv_dw_post_op_attr(depthwise, "k3s2p1");

    impl::logical_tensor_t conv_src
            = logical_tensor_init(0, conv_src_shape, impl::data_type::f32);
    impl::logical_tensor_t conv_wei
            = logical_tensor_init(1, conv_wei_shape, impl::data_type::f32);
    impl::logical_tensor_t conv_dst
            = logical_tensor_init(2, conv_dst_shape, impl::data_type::f32);

    impl::logical_tensor_t dw_wei
            = logical_tensor_init(3, dw_wei_shape, impl::data_type::f32);
    impl::logical_tensor_t dw_dst
            = logical_tensor_init(4, dw_dst_shape, impl::data_type::f32);

    conv.add_input(conv_src);
    conv.add_input(conv_wei);
    conv.add_output(conv_dst);

    depthwise.add_input(conv_dst);
    depthwise.add_input(dw_wei);
    depthwise.add_output(dw_dst);

    impl::graph_t g;
    g.add_op(&conv);
    g.add_op(&depthwise);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_depthwise_fusion_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            part->get_ops(), p_eng, impl::fpmath_mode::strict, false, true);
    dnnl_impl::subgraph_visualizer_t vis(part->id(), [](const value_t *val) {
        (void)val;
        return std::string();
    });
    dnnl_impl::pass_pipeline_t pipeline(vis, true, true);
    dnnl_impl::larger_partition_kernel_t::setup_pipeline_stage1(pipeline);
    ASSERT_EQ(pipeline.run(subgraph), impl::status::success);
    // fused conv and to_groupped ops
    ASSERT_EQ(subgraph->num_ops(), 2U);
}

TEST(SubgraphPass, FuseSigmoidMultiplyToSwish) {
    /*   
              /\
        sigmoid \
              \ /
             multiply
                |
    */
    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = dnnl_impl::make_dnnl_engine(g_eng);

    std::vector<int64_t> src_shape {1, 16, 4, 4};

    impl::op_t sigmoid {0, impl::op_kind::Sigmoid, "sigmoid"};
    impl::op_t multiply {1, impl::op_kind::Multiply, "multiply"};

    impl::logical_tensor_t sigmoid_src
            = logical_tensor_init(0, src_shape, impl::data_type::f32);
    impl::logical_tensor_t sigmoid_dst
            = logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t multiply_dst
            = logical_tensor_init(2, src_shape, impl::data_type::f32);

    sigmoid.add_input(sigmoid_src);
    sigmoid.add_output(sigmoid_dst);

    multiply.add_input(sigmoid_src);
    multiply.add_input(sigmoid_dst);
    multiply.add_output(multiply_dst);

    impl::graph_t g;
    g.add_op(&sigmoid);
    g.add_op(&multiply);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("eltwise_binary_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            part->get_ops(), p_eng, impl::fpmath_mode::strict, false, true);
    dnnl_impl::pass_pipeline_t pipeline(
            dnnl_impl::subgraph_visualizer_t(), true, false);
    dnnl_impl::larger_partition_kernel_t::setup_pipeline_stage1(pipeline);
    ASSERT_EQ(pipeline.run(subgraph), impl::status::success);
    ASSERT_EQ(subgraph->num_ops(), 1U);
    ASSERT_EQ(subgraph->get_ops()[0]->get_kind(),
            dnnl_impl::op_kind::dnnl_eltwise);
    ASSERT_EQ(static_cast<dnnl::algorithm>(
                      subgraph->get_ops()[0]->get_attr<int64_t>(
                              dnnl_impl::op_attr::alg_kind)),
            dnnl::algorithm::eltwise_swish);
}

TEST(TestInt8MatmulPassesWithDiffInputs, X8X8BF16MatmulScaleAddPasses) {
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
        impl::engine_t &g_eng = get_engine();
        dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);
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

        agraph.build_graph();

        pass::pass_base_ptr apass
                = get_pass("int8_bf16_matmul_scale_add_fusion_cpu");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                impl::partition_kind::quantized_matmul_post_ops);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);

        auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
                agraph.get_partitions()[0]->get_ops(), p_eng,
                impl::fpmath_mode::strict, false, true);
        // dequant, dequant, tc, tc, matmul, scale, add
        ASSERT_EQ(subgraph->get_ops().size(), 7U);

        dnnl_impl::check_with_bias(subgraph);

        int8_data = logical_tensor_init(0, {16, 8, 8, 8}, impl::data_type::u8);
        int8_weight
                = logical_tensor_init(3, {16, 8, 8, 8}, impl::data_type::u8);
        bf16_scale_in
                = logical_tensor_init(7, {1, 1, 1, 1}, impl::data_type::bf16);
        bf16_add_in
                = logical_tensor_init(9, {16, 1, 1, 8}, impl::data_type::bf16);
        bf16_add_out
                = logical_tensor_init(10, {16, 8, 8, 8}, impl::data_type::bf16);

        std::vector<logical_tensor_t> inputs
                = {int8_data, int8_weight, bf16_scale_in, bf16_add_in};
        std::vector<logical_tensor_t> outputs = {bf16_add_out};

        dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

        dnnl_impl::pass_pipeline_t pipeline(
                dnnl_impl::subgraph_visualizer_t(), true, false);
        dnnl_impl::larger_partition_kernel_t::setup_pipeline_stage1(pipeline);
        ASSERT_EQ(pipeline.run(subgraph), impl::status::success);

        // reorder, matmul
        ASSERT_EQ(subgraph->get_ops().size(), 2U);

        for (auto &val : subgraph->get_input_values()) {
            auto lt = val->get_logical_tensor();
            ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
        }

        for (auto &val : subgraph->get_output_values()) {
            auto lt = val->get_logical_tensor();
            if (lt.id == std::numeric_limits<size_t>::max()) continue;
            ASSERT_FALSE(impl::logical_tensor_wrapper_t(lt).is_shape_unknown());
        }

        ASSERT_EQ(subgraph->infer_shape(), impl::status::success);

        ASSERT_EQ(
                dnnl_impl::layout_propagation(subgraph), impl::status::success);
        // reorder, matmul
        ASSERT_EQ(subgraph->get_ops().size(), 2U);
    }
}

TEST(SubgraphPass, FuseTypecastToQuantize) {
    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);
    graph_t agraph;

    std::vector<int64_t> src_shape = {1, 8, 16};
    impl::op_t typecast(0, impl::op_kind::TypeCast, "typecast");
    impl::op_t quantize(1, impl::op_kind::Quantize, "quantize");
    quantize.set_attr<std::vector<float>>(op_attr::scales, {0.1f});
    quantize.set_attr<std::vector<int64_t>>(op_attr::zps, {10});
    quantize.set_attr<std::string>(op_attr::qtype, "per_tensor");
    quantize.set_attr<int64_t>(op_attr::axis, 0);

    impl::logical_tensor_t src_bf16
            = logical_tensor_init(0, src_shape, impl::data_type::bf16);
    impl::logical_tensor_t src_f32
            = logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_int8
            = logical_tensor_init(2, src_shape, impl::data_type::u8);

    typecast.add_input(src_bf16);
    typecast.add_output(src_f32);

    quantize.add_input(src_f32);
    quantize.add_output(dst_int8);

    ASSERT_EQ(agraph.add_op(&typecast), impl::status::success);
    ASSERT_EQ(agraph.add_op(&quantize), impl::status::success);
    agraph.build_graph();
    pass::pass_base_ptr apass = get_pass("typecast_quantize_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops(), p_eng,
            impl::fpmath_mode::strict, false, true);
    // tc, quant
    ASSERT_EQ(subgraph->get_ops().size(), 2U);

    dnnl_impl::pass_pipeline_t pipeline(
            dnnl_impl::subgraph_visualizer_t(), true, false);
    dnnl_impl::larger_partition_kernel_t::setup_pipeline_stage1(pipeline);
    ASSERT_EQ(pipeline.run(subgraph), impl::status::success);

    ASSERT_EQ(subgraph->get_ops().size(), 1U);
}

TEST(SubgraphPass, MemoryPlanningAllowReuseOutputBuffer) {
    impl::engine_t &eng = get_engine();

    id_generator id_gen;
    impl::graph_t g(eng.kind());
    construct_convolutional_bottleneck_resblock(&g, id_gen);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 8U);

    impl::pass::pass_base_ptr apass
            = get_pass("convolutional_bottleneck_resblock_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // prepare inputs/outputs
    impl::partition_t p;
    p.init(part);
    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 10U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<impl::logical_tensor_t> inputs, outputs;
    for (auto &lt : partition_inputs) {
        // skip alias inputs
        auto pos = std::find_if(inputs.begin(), inputs.end(),
                [&](const impl::logical_tensor_t &item) {
                    return item.id == lt.id;
                });
        if (pos != inputs.end()) continue;
        inputs.emplace_back(lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be any
        lt = logical_tensor_init(lt.id, lt.data_type, impl::layout_type::any);
        outputs.emplace_back(lt);
    }

    // run subgraph passes
    dnnl::engine p_eng = dnnl_impl::make_dnnl_engine(eng);
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            part->get_ops(), p_eng, impl::fpmath_mode::strict, false, true);

    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

    ASSERT_EQ(dnnl_impl::lower_down(subgraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::check_with_bias(subgraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::infer_shape(subgraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::binary_canonicalization(subgraph),
            impl::status::success);
    ASSERT_EQ(dnnl_impl::infer_shape(subgraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::fuse_post_ops(subgraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::insert_permute_for_conv_or_deconv(subgraph),
            impl::status::success);
    ASSERT_EQ(dnnl_impl::insert_to_group_for_conv_or_deconv(subgraph),
            impl::status::success);
    ASSERT_EQ(dnnl_impl::infer_shape(subgraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::common_reorder_elimination(subgraph),
            impl::status::success);
    ASSERT_EQ(dnnl_impl::constant_propagation(subgraph), impl::status::success);

    dnnl_impl::memory_planner_t memory_planner;
    ASSERT_EQ(memory_planner.run(subgraph), impl::status::success);

    dnnl_impl::subgraph_visualizer_t vis(0, [&](const value_t *val) {
        return memory_planner.get_memory_info(val);
    });
    vis.run(subgraph, "SubgraphPass.MemoryPlanningAllowReuseOutputBuffer", true,
            true);

    // external output buffer will be used for subgraph's output as well as the
    // conv-sum's post-src
    auto ext_out_mem_offkeys = memory_planner.get_exec_args_set()
                                       .get_mems_use_external_outputs();
    ASSERT_EQ(ext_out_mem_offkeys.size(), 2U);
}

TEST(SubgraphPass, CommonReorderElimination) {
    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);
    size_t id = 0;
    auto lt1 = logical_tensor_init(id++, {1, 3, 5, 5}, impl::data_type::f32);
    auto lt2 = logical_tensor_init(id++, {1, 3, 5, 5}, impl::data_type::f32);

    auto lt3 = logical_tensor_init(id++, {1, 3, 5, 5}, impl::data_type::f32);
    auto lt4 = logical_tensor_init(id++, {1, 3, 5, 5}, impl::data_type::f32);
    auto lt5 = logical_tensor_init(id++, {1, 3, 5, 5}, impl::data_type::f32);

    impl::op_t op0 {0, impl::op_kind::Wildcard, "wild_card"};
    op0.add_input(lt1);
    op0.add_output(lt2);

    impl::op_t reorder_op1 {1, impl::op_kind::Reorder, "reorder_op1"};
    reorder_op1.add_input(lt2);
    reorder_op1.add_output(lt3);

    impl::op_t reorder_op2 {2, impl::op_kind::Reorder, "reorder2"};
    reorder_op2.add_input(lt2);
    reorder_op2.add_output(lt4);

    impl::op_t op1 {id++, impl::op_kind::Wildcard, "op2"};
    op1.add_input(lt3);
    op1.add_input(lt4);
    op1.add_output(lt5);

    impl::graph_t g;
    g.add_op(&op0);
    g.add_op(&reorder_op1);
    g.add_op(&reorder_op2);
    g.add_op(&op1);
    g.build_graph();
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(g.get_ops(), p_eng,
            impl::fpmath_mode::any, /* reset_layout */ false, false);
    ASSERT_EQ(dnnl_impl::common_reorder_elimination(subgraph),
            impl::status::success);
    ASSERT_EQ(subgraph->get_ops().size(), 3U);
}

TEST(LayoutPropagation, ReshapeWithSpecifiedOutputLayout) {
    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);

    std::vector<int64_t> in_shape {1, 384, 16, 64};
    std::vector<int64_t> out_shape {1, 384, 1024};

    impl::op_t op1(1, impl::op_kind::StaticReshape, "op1");
    op1.set_attr<std::vector<int64_t>>(op_attr::shape, out_shape);
    op1.set_attr<bool>(op_attr::special_zero, true);

    auto in = logical_tensor_init(0, in_shape, impl::data_type::f32);
    // the output layout is specified to be channel last
    auto out = logical_tensor_init(1, out_shape,
            std::vector<int64_t> {384 * 1024, 1, 384}, impl::data_type::f32);
    op1.add_input(in);
    op1.add_output(out);

    impl::graph_t g;
    g.add_op(&op1);
    g.build_graph();

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(g.get_ops(), p_eng,
            impl::fpmath_mode::strict, false, /* reset_layout */ false);
    ASSERT_EQ(subgraph->get_ops().size(), 1U);

    ASSERT_EQ(dnnl_impl::lower_down(subgraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), impl::status::success);

    // A reorder should be inserted before reshape op
    ASSERT_EQ(subgraph->get_ops().size(), 2U);
    std::vector<impl::op_t *> sorted_ops;
    impl::topo_order_visit(subgraph->get_output_ops(), [&](impl::op_t *op) {
        sorted_ops.emplace_back(op);
        return impl::status::success;
    });
    ASSERT_EQ(sorted_ops[0]->get_kind(), dnnl_impl::op_kind::dnnl_reorder);
}

TEST(LayoutPropagation, ReshapeWithUnreshapableInputLayout) {
    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);

    std::vector<int64_t> in_shape {1, 384, 16, 64};
    std::vector<int64_t> out_shape {384 * 16, 64};

    impl::op_t op1(1, impl::op_kind::StaticReshape, "op1");
    op1.set_attr<std::vector<int64_t>>(op_attr::shape, out_shape);
    op1.set_attr<bool>(op_attr::special_zero, true);

    // the input layout is nhwc, which can't be directly reshaped to out_shape
    auto in = logical_tensor_init(0, in_shape,
            std::vector<int64_t> {384 * 16 * 64, 1, 384 * 64, 384},
            impl::data_type::f32);
    auto out = logical_tensor_init(
            1, out_shape, impl::data_type::f32, impl::layout_type::any);
    op1.add_input(in);
    op1.add_output(out);

    impl::graph_t g;
    g.add_op(&op1);
    g.build_graph();

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(g.get_ops(), p_eng,
            impl::fpmath_mode::strict, false, /* reset_layout */ false);
    ASSERT_EQ(subgraph->get_ops().size(), 1U);

    ASSERT_EQ(dnnl_impl::lower_down(subgraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), impl::status::success);

    // A reorder should be inserted before reshape op
    ASSERT_EQ(subgraph->get_ops().size(), 2U);
    std::vector<impl::op_t *> sorted_ops;
    impl::topo_order_visit(subgraph->get_output_ops(), [&](impl::op_t *op) {
        sorted_ops.emplace_back(op);
        return impl::status::success;
    });
    ASSERT_EQ(sorted_ops[0]->get_kind(), dnnl_impl::op_kind::dnnl_reorder);
}

TEST(LayoutPropagation, ReshapeWithReshapableInputLayout) {
    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);

    std::vector<int64_t> in_shape {1, 384, 16, 64};
    std::vector<int64_t> out_shape {384 * 16, 64};

    impl::op_t op1(1, impl::op_kind::StaticReshape, "op1");
    op1.set_attr<std::vector<int64_t>>(op_attr::shape, out_shape);
    op1.set_attr<bool>(op_attr::special_zero, true);

    auto in = logical_tensor_init(0, in_shape, impl::data_type::f32);
    auto out = logical_tensor_init(
            1, out_shape, impl::data_type::f32, impl::layout_type::any);
    op1.add_input(in);
    op1.add_output(out);

    impl::graph_t g;
    g.add_op(&op1);
    g.build_graph();

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(g.get_ops(), p_eng,
            impl::fpmath_mode::strict, false, /* reset_layout */ false);
    ASSERT_EQ(subgraph->get_ops().size(), 1U);

    ASSERT_EQ(dnnl_impl::lower_down(subgraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), impl::status::success);

    // No reorder
    ASSERT_EQ(subgraph->get_ops().size(), 1U);
}

TEST(LayoutPropagation, Transpose) {
    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);

    std::vector<int64_t> in_shape {1, 384, 16, 64};
    std::vector<int64_t> out_shape {1, 16, 64, 384};

    impl::op_t op1(1, impl::op_kind::StaticTranspose, "op1");
    op1.set_attr<std::vector<int64_t>>(
            op_attr::order, std::vector<int64_t> {0, 2, 3, 1});

    auto in = logical_tensor_init(0, in_shape, impl::data_type::f32);
    // the output layout is specified to be channel last
    auto out = logical_tensor_init(
            1, out_shape, impl::data_type::f32, impl::layout_type::any);
    op1.add_input(in);
    op1.add_output(out);

    impl::graph_t g;
    g.add_op(&op1);
    g.build_graph();

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(g.get_ops(), p_eng,
            impl::fpmath_mode::strict, false, /* reset_layout */ false);
    ASSERT_EQ(subgraph->get_ops().size(), 1U);

    ASSERT_EQ(dnnl_impl::lower_down(subgraph), impl::status::success);
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), impl::status::success);

    // the output value's layout type should be strided, and the corresponding
    // md shape should be equal to the out_shape, the stride should be
    // transposed one
    auto out_lt
            = subgraph->get_ops()[0]->get_output_value(0)->get_logical_tensor();
    ASSERT_EQ(out_lt.layout_type, impl::layout_type::strided);
    auto out_md = dnnl_impl::make_dnnl_memory_desc(out_lt);
    ASSERT_EQ(out_md.dims(), out_shape);
    std::vector<int64_t> out_stride {393216, 64, 1, 1024};
    const auto md_stride
            = std::vector<int64_t>(out_md.data.format_desc.blocking.strides,
                    out_md.data.format_desc.blocking.strides + 4);
    ASSERT_EQ(md_stride, out_stride);
}

TEST(SubgraphPass, FuseTypecastBeforeFusePostops) {
    impl::engine_t &engine = get_engine();

    // prepare fp32 data
    std::vector<int64_t> src_shape = {3, 8, 4};
    std::vector<int64_t> weight_shape = {4, 2};
    std::vector<int64_t> bias_shape {2};
    std::vector<int64_t> dst_shape = {3, 8, 2};

    float scale_src = 1 / 255.f; // map to 0~255
    float scale_dst = 1 / 255.f; // map to 0~255
    int64_t zp_src = 0;
    int64_t zp_dst = 6;

    size_t id = 0;

    size_t scales_wei_sizes = dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    impl::op_t dqdata_op(id++, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(op_attr::axis, 0);

    impl::op_t dqweight_op(id++, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(op_attr::qtype, "per_channel");
    dqweight_op.set_attr<std::vector<int64_t>>(op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(op_attr::axis, 1);

    impl::op_t matmul_op(id++, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(op_attr::transpose_b, false);

    impl::op_t gelu_op(id++, impl::op_kind::GELU, "gelu_op");

    impl::op_t tcdata_op {id++, impl::op_kind::TypeCast, "typecast_data"};
    impl::op_t tcweight_op {id++, impl::op_kind::TypeCast, "typecast_weight"};
    impl::op_t tcdst_op {id++, impl::op_kind::TypeCast, "typecast_dst"};

    impl::op_t qdst_op(id++, impl::op_kind::Quantize, "qdst_op");
    qdst_op.set_attr<std::string>(op_attr::qtype, "per_tensor");
    qdst_op.set_attr<std::vector<int64_t>>(op_attr::zps, {zp_dst});
    qdst_op.set_attr<std::vector<float>>(op_attr::scales, {scale_dst});
    qdst_op.set_attr<int64_t>(op_attr::axis, 0);

    // prepare logical tensor
    impl::logical_tensor_t src_u8
            = logical_tensor_init(id++, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = logical_tensor_init(id++, src_shape, impl::data_type::f32);
    impl::logical_tensor_t src_bf16
            = logical_tensor_init(id++, src_shape, impl::data_type::bf16);
    impl::logical_tensor_t weight_s8
            = logical_tensor_init(id++, weight_shape, impl::data_type::s8);
    impl::logical_tensor_t weight_bf16
            = logical_tensor_init(5, weight_shape, impl::data_type::bf16);
    impl::logical_tensor_t weight_f32_dq
            = logical_tensor_init(id++, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t bias_bf16
            = logical_tensor_init(id++, bias_shape, impl::data_type::bf16);
    impl::logical_tensor_t dst_bf16
            = logical_tensor_init(id++, dst_shape, impl::data_type::bf16);
    impl::logical_tensor_t gelu_bf16
            = logical_tensor_init(id++, dst_shape, impl::data_type::bf16);
    impl::logical_tensor_t gelu_f32
            = logical_tensor_init(id++, dst_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_u8
            = logical_tensor_init(id++, dst_shape, impl::data_type::u8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_input(bias_bf16);
    matmul_op.add_output(dst_bf16);

    gelu_op.add_input(dst_bf16);
    gelu_op.add_output(gelu_bf16);

    tcdst_op.add_input(gelu_bf16);
    tcdst_op.add_output(gelu_f32);

    qdst_op.add_input(gelu_f32);
    qdst_op.add_output(dst_u8);

    impl::graph_t g(engine.kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&tcdst_op);
    g.add_op(&gelu_op);
    g.add_op(&qdst_op);
    g.build_graph();

    pass::pass_base_ptr apass = get_pass(engine.kind() == impl::engine_kind::gpu
                    ? "int8_bf16_matmul_post_ops_fusion_gpu"
                    : "int8_bf16_matmul_post_ops_fusion_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);

    impl::engine_t &g_eng = get_engine();
    dnnl::engine p_eng = impl::dnnl_impl::make_dnnl_engine(g_eng);
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            g.get_partitions()[0]->get_ops(), p_eng, impl::fpmath_mode::strict,
            false, true);
    ASSERT_EQ(subgraph->get_ops().size(), 8U);

    dnnl_impl::subgraph_visualizer_t vis(0, [](const value_t *val) {
        (void)val;
        return std::string();
    });
    dnnl_impl::pass_pipeline_t pipeline(vis, true, true);
    dnnl_impl::larger_partition_kernel_t::setup_pipeline_stage1(pipeline);
    ASSERT_EQ(pipeline.run(subgraph), impl::status::success);
    // 1 bias scaling, 1 bias unsqueezing, 1 fused matmul, 2 reshape
    ASSERT_EQ(subgraph->num_ops(), 5U);
}

TEST(SubgraphPass, CheckUndefinedOpAttribute) {
    /*
    (f32) \     / (f32)
            conv
             | (f32)
            relu
             | (f32)
    */
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {1, ReLU, "relu"};

    logical_tensor_t fp32_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t fp32_weight = logical_tensor_init(1, data_type::f32);
    logical_tensor_t fp32_bias = logical_tensor_init(2, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(3, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(4, data_type::f32);
    relu.add_input(fp32_conv_out);
    relu.add_output(fp32_relu_out);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);

    agraph.build_graph();

    pass::pass_base_ptr apass = get_pass("conv_bias_post_ops_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops());

    fp32_data = logical_tensor_init(0, {1, 112, 112, 8}, data_type::f32);
    fp32_weight = logical_tensor_init(1, {3, 3, 8, 8}, data_type::f32);
    fp32_bias = logical_tensor_init(2, {8}, data_type::f32);

    // set incorrect op attribute to trigger the alarm.
    std::vector<int64_t> zps = {0};
    subgraph->get_ops()[0]->set_attr(op_attr::zps, zps);
    ASSERT_TRUE(subgraph->get_ops()[0]->has_attr(op_attr::zps));

    dnnl_impl::set_given_inputs_outputs(
            subgraph, {fp32_data, fp32_weight, fp32_bias}, {fp32_relu_out});
    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    ASSERT_EQ(validator.run(subgraph), status::invalid_op);
}

TEST(SubgraphPass, FuseAdjacentReorders) {
    dnnl_impl::dnnl_backend::get_singleton();
    size_t id = 0;
    impl::engine_t &eng = get_engine();
    dnnl::engine p_engine = dnnl_impl::make_dnnl_engine(eng);
    {
        graph_t agraph {eng.kind()};
        impl::op_t op1 {id++, impl::op_kind::TypeCast, "op1"};
        impl::op_t op2 {id++, impl::op_kind::TypeCast, "op2"};
        auto lt1 = logical_tensor_init(
                id++, {1, 2}, impl::data_type::f32, impl::layout_type::strided);
        auto lt2 = logical_tensor_init(
                id++, {1, 2}, impl::data_type::f16, impl::layout_type::strided);
        auto lt3 = logical_tensor_init(
                id++, {1, 2}, impl::data_type::f32, impl::layout_type::strided);
        op1.add_input(lt1);
        op1.add_output(lt2);
        op2.add_input(lt2);
        op2.add_output(lt3);
        agraph.add_op(&op1);
        agraph.add_op(&op2);
        ASSERT_EQ(agraph.build_graph(), impl::status::success);
        auto subgraph
                = std::make_shared<dnnl_impl::subgraph_t>(agraph.get_ops(),
                        p_engine, impl::fpmath_mode::any, false, false);
        ASSERT_EQ(dnnl_impl::lower_down(subgraph), impl::status::success);
        ASSERT_EQ(dnnl_impl::fuse_adjacent_reorders(subgraph),
                impl::status::success);
        ASSERT_EQ(subgraph->num_ops(), 1U);
    }
    {
#define type_list \
    std::vector<float>, std::vector<float>, std::vector<int64_t>, \
            std::vector<int64_t>
        std::vector<std::tuple<type_list>> vec {
                std::make_tuple<type_list>(std::vector<float> {0.7},
                        std::vector<float> {0.6}, std::vector<int64_t> {10},
                        std::vector<int64_t> {20}),
        };
#undef type_list
        for (auto &item : vec) {
            const auto &scales1 = std::get<0>(item);
            const auto &scales2 = std::get<1>(item);
            const auto &zps1 = std::get<2>(item);
            const auto &zps2 = std::get<3>(item);

            graph_t agraph {eng.kind()};
            impl::op_t op1 {id++, impl::op_kind::Dequantize, "op1"};

            op1.set_attr<int64_t>(impl::op_attr::axis, 1);
            op1.set_attr<std::vector<float>>(impl::op_attr::scales, scales1);
            op1.set_attr<std::vector<int64_t>>(impl::op_attr::zps, zps1);
            op1.set_attr<std::string>(impl::op_attr::qtype, "per_channel");

            impl::op_t op2 {id++, impl::op_kind::Quantize, "op2"};
            op2.set_attr<int64_t>(impl::op_attr::axis, 1);
            op2.set_attr<std::vector<float>>(impl::op_attr::scales, scales2);
            op2.set_attr<std::vector<int64_t>>(impl::op_attr::zps, zps2);
            op2.set_attr<std::string>(impl::op_attr::qtype, "per_channel");

            auto lt1 = logical_tensor_init(id++, {1, 2}, impl::data_type::u8,
                    impl::layout_type::strided);
            auto lt2 = logical_tensor_init(id++, {1, 2}, impl::data_type::f32,
                    impl::layout_type::strided);
            auto lt3 = logical_tensor_init(id++, {1, 2}, impl::data_type::u8,
                    impl::layout_type::strided);
            op1.add_input(lt1);
            op1.add_output(lt2);
            op2.add_input(lt2);
            op2.add_output(lt3);
            agraph.add_op(&op1);
            agraph.add_op(&op2);
            ASSERT_EQ(agraph.build_graph(), impl::status::success);
            auto subgraph
                    = std::make_shared<dnnl_impl::subgraph_t>(agraph.get_ops(),
                            p_engine, impl::fpmath_mode::any, false, false);
            ASSERT_EQ(dnnl_impl::lower_down(subgraph), impl::status::success);
            ASSERT_EQ(dnnl_impl::fuse_static_mul_scales_add_zps(subgraph),
                    impl::status::success);
            ASSERT_EQ(dnnl_impl::fuse_static_sub_zps_mul_scales(subgraph),
                    impl::status::success);
            ASSERT_EQ(dnnl_impl::fuse_adjacent_reorders(subgraph),
                    impl::status::success);
            ASSERT_EQ(subgraph->num_ops(), 1U);
        }
    }
}

TEST(SubgraphPass, CombineBinaryPostOpScales) {
    namespace utils = dnnl::graph::tests::unit::utils;
    dnnl_impl::dnnl_backend::get_singleton();
    using dims = impl::dnnl_impl::dims;
    using config_t = std::tuple<impl::op_kind_t, bool>;

    impl::engine_t &engine = get_engine();
    dnnl::engine p_engine = impl::dnnl_impl::make_dnnl_engine(engine);

    auto conf = config_t {impl::op_kind::AvgPool, true};
    std::string qtype = "symmetric";

    impl::op_kind_t base_op = impl::op_kind::Wildcard;
    bool per_channel_broadcast = false;
    std::tie(base_op, per_channel_broadcast) = conf;

    const std::string data_format {"NCX"};
    const int64_t channels = 2;
    std::vector<int64_t> src_shape {2, channels, 4, 4};
    std::vector<int64_t> dst_shape {2, channels, 2, 2};
    std::vector<int64_t> other_shape {1, 1, 1, 1};
    if (per_channel_broadcast) other_shape[1] = channels;

    const float scale_src = 5 / 127.f;
    const float scale_out = 10 / 127.f;
    const float scale_other = 2 / 127.f;
    const int64_t zp_src = (qtype == "symmetric") ? 0 : -2;
    const int64_t zp_out = (qtype == "symmetric") ? 0 : -2;
    const int64_t zp_other = (qtype == "symmetric") ? 0 : 4;

    impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(impl::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    impl::op_t pool_op(1, base_op, "pool_op");
    size_t spatial_size = src_shape.size() - 2;
    pool_op.set_attr<dims>(impl::op_attr::strides, dims(spatial_size, 2));
    pool_op.set_attr<dims>(impl::op_attr::kernel, dims(spatial_size, 2));
    pool_op.set_attr<dims>(impl::op_attr::pads_begin, dims(spatial_size, 0));
    pool_op.set_attr<dims>(impl::op_attr::pads_end, dims(spatial_size, 0));
    pool_op.set_attr<std::string>(impl::op_attr::data_format, data_format);
    pool_op.set_attr<bool>(impl::op_attr::exclude_pad, false);

    impl::op_t qout_op(2, impl::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_out});
    qout_op.set_attr<std::vector<float>>(impl::op_attr::scales, {scale_out});
    qout_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    impl::op_t dqother_op(3, impl::op_kind::Dequantize, "dqother_op");
    dqother_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dqother_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zp_other});
    dqother_op.set_attr<std::vector<float>>(
            impl::op_attr::scales, {scale_other});
    dqother_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    impl::op_t mul_op(4, impl::op_kind::Multiply, "mul_op");

    auto src_s8 = utils::logical_tensor_init(0, src_shape, impl::data_type::s8);
    auto src_f32_dq
            = utils::logical_tensor_init(1, src_shape, impl::data_type::f32);
    auto dst_f32
            = utils::logical_tensor_init(2, dst_shape, impl::data_type::f32);
    auto dst_s8 = utils::logical_tensor_init(3, dst_shape, impl::data_type::s8);
    auto other_s8
            = utils::logical_tensor_init(4, other_shape, impl::data_type::s8);
    auto other_f32_dq
            = utils::logical_tensor_init(5, other_shape, impl::data_type::f32);
    auto dst_mul_f32
            = utils::logical_tensor_init(6, dst_shape, impl::data_type::f32);

    dqdata_op.add_input(src_s8);
    dqdata_op.add_output(src_f32_dq);

    pool_op.add_input(src_f32_dq);
    pool_op.add_output(dst_f32);

    dqother_op.add_input(other_s8);
    dqother_op.add_output(other_f32_dq);

    mul_op.add_input(other_f32_dq);
    mul_op.add_input(dst_f32);

    mul_op.add_output(dst_mul_f32);

    qout_op.add_input(dst_mul_f32);
    qout_op.add_output(dst_s8);

    impl::graph_t g(engine.kind());
    g.add_op(&dqdata_op);
    g.add_op(&pool_op);
    g.add_op(&dqother_op);
    g.add_op(&mul_op);
    g.add_op(&qout_op);
    g.build_graph();

    auto subgraph = std::make_shared<impl::dnnl_impl::subgraph_t>(g.get_ops(),
            p_engine, impl::fpmath_mode::any, /* reset_layout */ false, false);
    ASSERT_EQ(impl::dnnl_impl::lower_down(subgraph), impl::status::success);
    ASSERT_EQ(impl::dnnl_impl::fuse_to_int8_pool(subgraph),
            impl::status::success);
    ASSERT_EQ(impl::dnnl_impl::combine_binary_post_op_scales(subgraph),
            impl::status::success);
    ASSERT_EQ(
            impl::dnnl_impl::fold_mul_scales(subgraph), impl::status::success);
    ASSERT_EQ(impl::dnnl_impl::remove_quant_data_with_no_effect(subgraph),
            impl::status::success);
    ASSERT_EQ(impl::dnnl_impl::replace_quant_data_with_binary_post_op(subgraph),
            impl::status::success);
    ASSERT_EQ(impl::dnnl_impl::fuse_static_mul_scales_add_zps(subgraph),
            impl::status::success);
    ASSERT_EQ(impl::dnnl_impl::fuse_static_sub_zps_mul_scales(subgraph),
            impl::status::success);
    ASSERT_EQ(impl::dnnl_impl::fuse_post_ops(subgraph), impl::status::success);
    ASSERT_EQ(subgraph->num_ops(), 2U);
}
