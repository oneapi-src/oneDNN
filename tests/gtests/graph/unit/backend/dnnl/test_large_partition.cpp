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

#include <functional>
#include <random>

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Execute, ConvResBlock) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::op_t conv0(0, graph::op_kind::Convolution, "conv0");
    graph::op_t relu0(1, graph::op_kind::ReLU, "relu0");
    graph::op_t conv1(2, graph::op_kind::Convolution, "conv1");
    graph::op_t relu1(3, graph::op_kind::ReLU, "relu1");
    graph::op_t conv2(4, graph::op_kind::Convolution, "conv2");
    graph::op_t relu2(5, graph::op_kind::ReLU, "relu2");
    graph::op_t add(6, graph::op_kind::Add, "add");

    std::vector<int64_t> v_1_1 = {1, 1}, v_0_0 = {0, 0};
    utils::set_conv_common_attr(
            conv0, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);
    utils::set_conv_common_attr(
            conv1, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);
    utils::set_conv_common_attr(
            conv2, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);

    // prepare logical tensor
    std::vector<graph::dim_t> src_shape = {8, 8, 32, 32};
    std::vector<graph::dim_t> wei_shape = {8, 8, 1, 1};
    graph::data_type_t dtype = graph::data_type::f32;
    auto conv0_src = utils::logical_tensor_init(0, src_shape, dtype);
    auto conv0_wei = utils::logical_tensor_init(1, wei_shape, dtype);
    auto relu0_src = utils::logical_tensor_init(2, src_shape, dtype);
    auto conv1_src = utils::logical_tensor_init(3, src_shape, dtype);
    auto conv1_wei = utils::logical_tensor_init(4, wei_shape, dtype);
    auto relu1_src = utils::logical_tensor_init(5, src_shape, dtype);
    auto conv2_src = utils::logical_tensor_init(6, src_shape, dtype);
    auto conv2_wei = utils::logical_tensor_init(7, wei_shape, dtype);
    auto add_src0 = utils::logical_tensor_init(8, src_shape, dtype);
    auto relu2_src = utils::logical_tensor_init(9, src_shape, dtype);
    auto relu2_dst = utils::logical_tensor_init(10, src_shape, dtype);

    conv0.add_input(conv0_src);
    conv0.add_input(conv0_wei);
    conv0.add_output(relu0_src);
    relu0.add_input(relu0_src);
    relu0.add_output(conv1_src);
    conv1.add_input(conv1_src);
    conv1.add_input(conv1_wei);
    conv1.add_output(relu1_src);
    relu1.add_input(relu1_src);
    relu1.add_output(conv2_src);
    conv2.add_input(conv2_src);
    conv2.add_input(conv2_wei);
    conv2.add_output(add_src0);
    add.add_input(add_src0);
    add.add_input(conv0_src);
    add.add_output(relu2_src);
    relu2.add_input(relu2_src);
    relu2.add_output(relu2_dst);

    graph::graph_t g(eng->kind());
    g.add_op(&conv0);
    g.add_op(&relu0);
    g.add_op(&conv1);
    g.add_op(&relu1);
    g.add_op(&conv2);
    g.add_op(&add);
    g.add_op(&relu2);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 7U);

    graph::pass::pass_base_ptr apass = get_pass("conv_simple_resblock_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &conv0_src, &conv0_wei, &conv1_wei, &conv2_wei, &conv0_src};
    std::vector<const graph::logical_tensor_t *> outputs {&relu2_dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    test::vector<float> conv0_src_data(8 * 8 * 32 * 32);
    test::vector<float> conv0_wei_data(8 * 8 * 1 * 1);
    test::vector<float> conv1_wei_data(8 * 8 * 1 * 1);
    test::vector<float> conv2_wei_data(8 * 8 * 1 * 1);
    test::vector<float> relu2_dst_data(8 * 8 * 32 * 32);
    test::vector<float> ref_dst_data(8 * 8 * 32 * 32);

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    std::generate(conv0_src_data.begin(), conv0_src_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv0_wei_data.begin(), conv0_wei_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv1_wei_data.begin(), conv1_wei_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv2_wei_data.begin(), conv2_wei_data.end(),
            [&]() { return distribution(generator); });

    graph::tensor_t conv0_src_ts(conv0_src, eng, conv0_src_data.data());
    graph::tensor_t conv0_wei_ts(conv0_wei, eng, conv0_wei_data.data());
    graph::tensor_t conv1_wei_ts(conv1_wei, eng, conv1_wei_data.data());
    graph::tensor_t conv2_wei_ts(conv2_wei, eng, conv2_wei_data.data());
    graph::tensor_t relu2_dst_ts(relu2_dst, eng, relu2_dst_data.data());
    graph::tensor_t ref_dst_ts(relu2_dst, eng, ref_dst_data.data());

    ASSERT_EQ(cp.execute(strm,
                      {conv0_src_ts, conv0_wei_ts, conv1_wei_ts, conv2_wei_ts},
                      {relu2_dst_ts}),
            graph::status::success);
    strm->wait();
}

TEST(Execute, ConvResBlockWithNhwcLayout) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::op_t conv0(0, graph::op_kind::Convolution, "conv0");
    graph::op_t relu0(1, graph::op_kind::ReLU, "relu0");
    graph::op_t conv1(2, graph::op_kind::Convolution, "conv1");
    graph::op_t relu1(3, graph::op_kind::ReLU, "relu1");
    graph::op_t conv2(4, graph::op_kind::Convolution, "conv2");
    graph::op_t relu2(5, graph::op_kind::ReLU, "relu2");
    graph::op_t add(6, graph::op_kind::Add, "add");

    std::vector<int64_t> v_1_1 = {1, 1}, v_0_0 = {0, 0};
    utils::set_conv_common_attr(
            conv0, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);
    utils::set_conv_common_attr(
            conv1, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);
    utils::set_conv_common_attr(
            conv2, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);

    // prepare logical tensor
    std::vector<graph::dim_t> src_shape = {8, 8, 32, 32};
    // strides for nhwc
    std::vector<graph::dim_t> src_strides_nhwc {
            src_shape[2] * src_shape[3] * src_shape[1], 1,
            src_shape[3] * src_shape[1], src_shape[1]};
    std::vector<graph::dim_t> wei_shape = {8, 8, 1, 1};
    graph::data_type_t dtype = graph::data_type::f32;
    auto conv0_src
            = utils::logical_tensor_init(0, src_shape, src_strides_nhwc, dtype);
    auto conv0_wei = utils::logical_tensor_init(1, wei_shape, dtype);
    auto relu0_src = utils::logical_tensor_init(2, src_shape, dtype);
    auto conv1_src = utils::logical_tensor_init(3, src_shape, dtype);
    auto conv1_wei = utils::logical_tensor_init(4, wei_shape, dtype);
    auto relu1_src = utils::logical_tensor_init(5, src_shape, dtype);
    auto conv2_src = utils::logical_tensor_init(6, src_shape, dtype);
    auto conv2_wei = utils::logical_tensor_init(7, wei_shape, dtype);
    auto add_src0 = utils::logical_tensor_init(8, src_shape, dtype);
    auto relu2_src = utils::logical_tensor_init(9, src_shape, dtype);
    auto relu2_dst = utils::logical_tensor_init(
            10, src_shape, dtype, graph::layout_type::any);

    conv0.add_input(conv0_src);
    conv0.add_input(conv0_wei);
    conv0.add_output(relu0_src);
    relu0.add_input(relu0_src);
    relu0.add_output(conv1_src);
    conv1.add_input(conv1_src);
    conv1.add_input(conv1_wei);
    conv1.add_output(relu1_src);
    relu1.add_input(relu1_src);
    relu1.add_output(conv2_src);
    conv2.add_input(conv2_src);
    conv2.add_input(conv2_wei);
    conv2.add_output(add_src0);
    add.add_input(add_src0);
    add.add_input(conv0_src);
    add.add_output(relu2_src);
    relu2.add_input(relu2_src);
    relu2.add_output(relu2_dst);

    graph::graph_t g(eng->kind());
    g.add_op(&conv0);
    g.add_op(&relu0);
    g.add_op(&conv1);
    g.add_op(&relu1);
    g.add_op(&conv2);
    g.add_op(&add);
    g.add_op(&relu2);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 7U);

    graph::pass::pass_base_ptr apass = get_pass("conv_simple_resblock_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &conv0_src, &conv0_wei, &conv1_wei, &conv2_wei, &conv0_src};
    std::vector<const graph::logical_tensor_t *> outputs {&relu2_dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t queried_relu2_dst;
    cp.query_logical_tensor(relu2_dst.id, &queried_relu2_dst);
    ASSERT_EQ(queried_relu2_dst.layout_type, graph::layout_type::strided);

    test::vector<float> conv0_src_data(8 * 8 * 32 * 32);
    test::vector<float> conv0_wei_data(8 * 8 * 1 * 1);
    test::vector<float> conv1_wei_data(8 * 8 * 1 * 1);
    test::vector<float> conv2_wei_data(8 * 8 * 1 * 1);
    test::vector<float> relu2_dst_data(8 * 8 * 32 * 32);
    test::vector<float> ref_dst_data(8 * 8 * 32 * 32);

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    std::generate(conv0_src_data.begin(), conv0_src_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv0_wei_data.begin(), conv0_wei_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv1_wei_data.begin(), conv1_wei_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv2_wei_data.begin(), conv2_wei_data.end(),
            [&]() { return distribution(generator); });

    graph::tensor_t conv0_src_ts(conv0_src, eng, conv0_src_data.data());
    graph::tensor_t conv0_wei_ts(conv0_wei, eng, conv0_wei_data.data());
    graph::tensor_t conv1_wei_ts(conv1_wei, eng, conv1_wei_data.data());
    graph::tensor_t conv2_wei_ts(conv2_wei, eng, conv2_wei_data.data());
    graph::tensor_t relu2_dst_ts(relu2_dst, eng, relu2_dst_data.data());
    graph::tensor_t ref_dst_ts(relu2_dst, eng, ref_dst_data.data());

    ASSERT_EQ(cp.execute(strm,
                      {conv0_src_ts, conv0_wei_ts, conv1_wei_ts, conv2_wei_ts},
                      {relu2_dst_ts}),
            graph::status::success);
    strm->wait();
}

TEST(Execute, F32ConvolutionalBottleneckResBlock) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator id_gen;
    graph::graph_t g(eng->kind());
    utils::construct_convolutional_bottleneck_resblock(&g, id_gen);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 8U);

    graph::pass::pass_base_ptr apass
            = get_pass("convolutional_bottleneck_resblock_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 10U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        // skip alias inputs
        auto pos = std::find_if(inputs.begin(), inputs.end(),
                [&](const graph::logical_tensor_t *item) {
                    return item->id == lt.id;
                });
        if (pos != inputs.end()) continue;
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    std::cout << "-------------------\n";
    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);

    strm->wait();
}

TEST(Execute, Int8IdenticalBottleneckResBlock) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator id_gen;
    graph::graph_t g(eng->kind());
    utils::construct_int8_identical_bottleneck_resblock(&g, id_gen);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 17U);

    graph::pass::pass_base_ptr apass
            = get_pass("int8_identical_bottleneck_resblock_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 8U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}

TEST(Execute, Int8Resnet50Stage2Block) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator id_gen;
    graph::graph_t g(eng->kind());
    utils::construct_int8_resnet50_stage2_block(&g, id_gen, 3);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 72U);

    graph::pass::pass_base_ptr apass = get_pass("int8_resnet50_stage_2_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 28U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}

TEST(Execute, F32Resnet50Stage2Block) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator id_gen;
    graph::graph_t g(eng->kind());
    utils::construct_f32_resnet50_stage2_block(
            &g, id_gen, 3, /* use biasadd */ true);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 42U);

    graph::pass::pass_base_ptr apass = get_pass("f32_resnet50_stage_2_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 28U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, eng, outputs_data.back().data());
    }

    std::cout << "----------------iter 1----------------\n";
    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    std::cout << "----------------iter 2----------------\n";
    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}

TEST(Execute, ItexInt8Resnet50Stage2Block) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator id_gen;
    graph::graph_t g(eng->kind());
    utils::construct_itex_int8_resnet50_stage2_block(&g, id_gen, 3);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 98U);

    graph::pass::pass_base_ptr apass
            = get_pass("itex_int8_resnet50_stage_2_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 28U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, eng, outputs_data.back().data());
    }

    std::cout << "----------------iter 1----------------\n";
    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    std::cout << "----------------iter 2----------------\n";
    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}

TEST(Execute, Int8ResneXt101Stage3Block) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator id_gen;
    graph::graph_t g(eng->kind());
    utils::construct_int8_resnext101_stage3_block(&g, id_gen, 22);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 395U);

    graph::pass::pass_base_ptr apass
            = get_pass("int8_resnext101_stage_3_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 142U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}

TEST(Execute, ChainedReLU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::graph_t g(eng->kind());
    utils::construct_chained_relu(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 3U);

    graph::pass::pass_base_ptr apass = get_pass("chained_relu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 1U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : partition_inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(lt, eng, inputs_data.back().data());
    }

    for (auto &lt : partition_outputs) {
        outputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        outputs_ts.emplace_back(lt, eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}

TEST(Execute, Int8ConvBiasReluConvBiasReluBlock) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator id_gen;
    graph::graph_t g(eng->kind());
    utils::construct_int8_conv_bias_relu_conv_bias_relu_block(&g, id_gen);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 10U);

    graph::pass::pass_base_ptr apass
            = get_pass("int8_conv_bias_relu_conv_bias_relu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}

TEST(Execute, Int8ConvBiasReluConvBiasReluConvBiasConvBiasAddReluBlock) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator id_gen;
    graph::graph_t g(eng->kind());
    utils::construct_int8_convolutional_bottleneck_resblock(&g, id_gen);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 21U);

    graph::pass::pass_base_ptr apass
            = get_pass("int8_convolutional_bottleneck_resblock_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 10U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}

TEST(Compile, Int8ConvBlockGetInplacePair) {
    graph::engine_t *eng = get_engine();

    utils::id_generator id_gen;
    graph::graph_t g(eng->kind());
    utils::construct_int8_convolutional_bottleneck_resblock(&g, id_gen);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 21U);

    graph::pass::pass_base_ptr apass1
            = get_pass("int8_identical_bottleneck_resblock_fusion");
    graph::pass::pass_base_ptr apass2
            = get_pass(eng->kind() == graph::engine_kind::gpu
                            ? "int8_conv_post_ops_int8_add_fusion_gpu"
                            : "int8_conv_post_ops_int8_add_fusion_cpu");
    apass1->run(g);
    apass2->run(g);
    ASSERT_EQ(g.get_num_partitions(), 2U);
    auto part0 = g.get_partitions()[0];
    auto part1 = g.get_partitions()[1];

    // compile part1 (single conv)
    graph::partition_t p1;
    p1.init(part1);

    auto partition_inputs1 = p1.get_inputs();
    auto partition_outputs1 = p1.get_outputs();
    ASSERT_EQ(partition_inputs1.size(), 3U);
    ASSERT_EQ(partition_outputs1.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs1, outputs1;
    for (auto &lt : partition_inputs1) {
        inputs1.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs1) {
        // set output to be any
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::any);
        outputs1.emplace_back(&lt);
    }

    graph::compiled_partition_t cp1(p1);
    ASSERT_EQ(p1.compile(&cp1, inputs1, outputs1, eng), graph::status::success);

    // compile part0 (three conv, one with sum)
    graph::partition_t p0;
    p0.init(part0);

    auto partition_inputs0 = p0.get_inputs();
    auto partition_outputs0 = p0.get_outputs();
    ASSERT_EQ(partition_inputs0.size(), 8U);
    ASSERT_EQ(partition_outputs0.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs0, outputs0;
    for (auto &lt : partition_inputs0) {
        if (lt.id == outputs1[0]->id) {
            // use precursor's output
            cp1.query_logical_tensor(lt.id, &lt);
        }
        inputs0.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs0) {
        // set output to be any
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::any);
        outputs0.emplace_back(&lt);
    }

    graph::compiled_partition_t cp0(p0);
    ASSERT_EQ(p0.compile(&cp0, inputs0, outputs0, eng), graph::status::success);
    auto pairs = cp0.get_inplace_pairs();
    ASSERT_EQ(pairs.size(), 1U);
    ASSERT_EQ(pairs[0].input_id, outputs1[0]->id);
    ASSERT_EQ(pairs[0].output_id, outputs0[0]->id);
}

TEST(Compile, ConvBiasReluAdd) {
    /* \  |  /
        Conv
          |
        ReLU
           \  /
           Add
    */
    using dims = graph::dnnl_impl::dims;

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            3, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t relu_dst_lt = utils::logical_tensor_init(
            4, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t add_src_lt = utils::logical_tensor_init(
            5, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            6, {1, 1, 4, 4}, graph::data_type::f32);

    // create op conv
    graph::op_t conv_op(0, graph::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::filter_format, "OIX");
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_input(bias_lt);
    conv_op.add_output(conv_dst_lt);
    //create op relu
    graph::op_t relu_op(1, graph::op_kind::ReLU, "ReLU");
    relu_op.add_input(conv_dst_lt);
    relu_op.add_output(relu_dst_lt);
    // create op add
    graph::op_t add_op(2, graph::op_kind::Add, "Add");
    add_op.add_input(relu_dst_lt);
    add_op.add_input(add_src_lt);
    add_op.add_output(add_dst_lt);
    // build graph
    graph::engine_t *eng = get_engine();
    graph::graph_t g(eng->kind());
    g.add_op(&conv_op);
    g.add_op(&relu_op);
    g.add_op(&add_op);
    g.finalize();

    // run pass
    graph::pass::pass_base_ptr apass = get_pass("conv_bias_post_ops_fusion");
    apass->run(g);

    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile conv+add partition
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);
    // arbitrary order of inputs
    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &bias_lt, &add_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);
}

TEST(Execute, Int8Mha) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    graph::graph_t g(eng->kind());
    utils::construct_int8_MHA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 21U);

    graph::pass::pass_base_ptr apass = get_pass("int8_MHA_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}

TEST(Execute, F32Mha) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::graph_t g(eng->kind());
    utils::construct_f32_MHA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 13U);

    graph::pass::pass_base_ptr apass = get_pass("f32_MHA_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}

TEST(Execute, Int8Bf16Mha) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu, "skip on gpu");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    graph::graph_t g(eng->kind());
    utils::construct_int8_bf16_MHA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 29U);

    graph::pass::pass_base_ptr apass = get_pass("int8_bf16_MHA_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<uint16_t>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : partition_inputs) {
        inputs_data.emplace_back(
                test::vector<uint16_t>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(lt, eng, inputs_data.back().data());
    }

    for (auto &lt : partition_outputs) {
        outputs_data.emplace_back(
                test::vector<uint16_t>(utils::product(ltw(lt).vdims())));
        outputs_ts.emplace_back(lt, eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}

TEST(Execute, F32MhaReshapeSoftMax) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::graph_t g(eng->kind());
    utils::construct_reshaped_softmax_f32_mha(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 14U);

    graph::pass::pass_base_ptr apass = get_pass("f32_MHA_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 4U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    using ltw = graph::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<graph::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(strm, inputs_ts, outputs_ts), graph::status::success);
    strm->wait();
}
