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

#include <functional>
#include <random>

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

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

TEST(Execute, Int8Resnet50Stage2BlockWithZeroZps) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator id_gen;
    graph::graph_t g(eng->kind());
    int64_t zps_postbinary = eng->kind() == graph::engine_kind::gpu ? 0 : 78;
    utils::construct_int8_resnet50_stage2_block(
            &g, id_gen, 3, false, false, 1 / 255.f, 0, zps_postbinary);
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

TEST(Execute, Int8Resnet50Stage2BlockWithQuantWei) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator id_gen;
    graph::graph_t g(eng->kind());
    utils::construct_int8_resnet50_stage2_block(&g, id_gen, 3, true, true);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 98U);

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
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
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

    ASSERT_EQ(g.get_ops().size(), 13U);

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
    utils::construct_dnnl_f32_MHA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 7U);

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
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core || isa == dnnl_cpu_isa_avx2_vnni)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    graph::graph_t g(eng->kind());
    utils::construct_int8_bf16_MHA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 19U);

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
