/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

static inline void custom_setenv(
        const char *name, const char *value, int overwrite) {
#ifdef _WIN32
    SetEnvironmentVariable(name, value);
#else
    ::setenv(name, value, overwrite);
#endif
}

static void fill_data(
        std::vector<float> &buffer, dnnl::impl::data_type_t dtype) {
    if (dtype == dnnl::impl::data_type::u8) {
        uint8_t *ptr = reinterpret_cast<uint8_t *>(buffer.data());
        std::generate(ptr, ptr + buffer.size(),
                [&]() { return static_cast<uint8_t>(2.f); });
    } else if (dtype == dnnl::impl::data_type::s8) {
        int8_t *ptr = reinterpret_cast<int8_t *>(buffer.data());
        std::generate(ptr, ptr + buffer.size(),
                [&]() { return static_cast<int8_t>(2.f); });
    } else if (dtype == dnnl::impl::data_type::f32) {
        std::generate(buffer.begin(), buffer.end(), [&]() { return 0.5f; });
    } else {
        throw std::runtime_error("unsupported data types");
    }
}

TEST(test_large_partition_execute, Int8Resnet50Stage2Block) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator_t id_gen;
    graph::graph_t g(eng->kind());
    utils::construct_int8_resnet50_stage2_block(
            &g, id_gen, 3, false, false, 1.2f, 1, 0);
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts, ref_outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<uint8_t>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
        ref_outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(run_graph(g, inputs_ts, ref_outputs_ts, *eng, *strm),
            graph::status::success);

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();

    ASSERT_TRUE(
            allclose<uint8_t>(outputs_ts[0], ref_outputs_ts[0], /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
}

TEST(test_large_partition_execute, Int8Resnet50Stage2BlockWithZeroZps) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator_t id_gen;
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts, ref_outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<uint8_t>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
        ref_outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(run_graph(g, inputs_ts, ref_outputs_ts, *eng, *strm),
            graph::status::success);

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();

    ASSERT_TRUE(
            allclose<uint8_t>(outputs_ts[0], ref_outputs_ts[0], /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
}

TEST(test_large_partition_execute, Int8Resnet50Stage2BlockWithQuantWei) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator_t id_gen;
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<uint8_t>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, F32Resnet50Stage2Block) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator_t id_gen;
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

    std::vector<std::vector<float>> inputs_data;
    std::vector<std::vector<float>> outputs_data, ref_outputs_data;
    std::vector<test_tensor_t> inputs_ts, outputs_ts, ref_outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(utils::product(ltw(lt).vdims()));
        fill_data(inputs_data.back(), ltw(lt).data_type());
        inputs_ts.emplace_back(*lt, eng, inputs_data.back());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        const std::vector<int64_t> dims = ltw(compiled_output).vdims();
        auto size = utils::product(dims);
        outputs_data.emplace_back(size);
        outputs_ts.emplace_back(compiled_output, eng, outputs_data.back());
        ref_outputs_data.emplace_back(size);
        ref_outputs_ts.emplace_back(
                compiled_output, eng, ref_outputs_data.back());
    }

    // set constant tensor cache capacity as 1GB
    dnnl::graph::set_constant_tensor_cache_capacity(
            static_cast<engine::kind>(eng->kind()), 1024);

    ASSERT_EQ(run_graph(g, inputs_ts, ref_outputs_ts, *eng, *strm),
            graph::status::success);

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    // execute another iteration to test constant cache hit
    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    // disable constant tensor cache and then to test constant cache miss
    dnnl::graph::set_constant_tensor_cache_capacity(
            static_cast<engine::kind>(eng->kind()), 0);
    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();

    ASSERT_TRUE(
            allclose<float>(outputs_ts[0], ref_outputs_ts[0], /*rtol*/ 1e-5f,
                    /*atol*/ 1e-5f));
}

TEST(test_large_partition_execute, ItexInt8Resnet50Stage2Block) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    utils::id_generator_t id_gen;
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<uint8_t>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    std::cout << "----------------iter 1----------------\n";
    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    std::cout << "----------------iter 2----------------\n";
    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_compile, ConvBiasReluAdd) {
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
    graph::pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
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

TEST(test_large_partition_execute, Int8Mha_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    graph::graph_t g(eng->kind());
    utils::construct_int8_MHA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 13U);

    graph::pass::pass_base_ptr apass = get_pass("int8_sdp_fusion");
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<uint8_t>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, Int8DistilBertMha) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::graph_t g(eng->kind());
    utils::construct_select_int8_MHA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 13U);

    graph::pass::pass_base_ptr apass = get_pass("int8_sdp_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 6U);
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<uint8_t>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, Int8GptMha) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::graph_t g(eng->kind());
    utils::construct_select_int8_MHA(&g, 1, 32, 16, 4096, false, false, true);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 14U);

    graph::pass::pass_base_ptr apass = get_pass("int8_fp32_gpt_sdp");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 7U);
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<uint8_t>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, F32Mha) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::graph_t g(eng->kind());
    utils::construct_dnnl_float_MHA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 7U);

    graph::pass::pass_base_ptr apass = get_pass(
            eng->kind() == graph::engine_kind::cpu ? "float_sdp_fusion_cpu"
                                                   : "float_sdp_fusion_gpu");
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<float>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, F32DistilBertMha) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::graph_t g(eng->kind());
    utils::construct_select_float_MHA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 7U);

    graph::pass::pass_base_ptr apass = get_pass(
            eng->kind() == graph::engine_kind::cpu ? "float_sdp_fusion_cpu"
                                                   : "float_sdp_fusion_gpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 6U);
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        // For select op's bool input
        if (ltw(lt).data_type() == graph::data_type::boolean)
            inputs_ts.back().fill<uint8_t>();
        else
            inputs_ts.back().fill<float>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, F32GptMha) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::graph_t g(eng->kind());
    utils::construct_select_float_MHA(
            &g, graph::data_type::f32, 1, 32, 16, 4096, false, false, true);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 8U);

    graph::pass::pass_base_ptr apass = get_pass("float_gpt_sdp");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 7U);
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        // For select op's bool input
        if (ltw(lt).data_type() == graph::data_type::boolean)
            inputs_ts.back().fill<uint8_t>();
        else
            inputs_ts.back().fill<float>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, F32JaxMha) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::graph_t g(eng->kind());
    utils::construct_dnnl_float_JAX_MHA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 9U);

    graph::pass::pass_base_ptr apass = get_pass("float_sdp_jax_fusion");
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<float>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, F32JaxMqa) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    graph::graph_t g(eng->kind());
    utils::construct_dnnl_float_JAX_MQA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 8U);

    graph::pass::pass_base_ptr apass = get_pass("float_mqa_jax_fusion");
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

    // Enable large partition test
    custom_setenv("_ONEDNN_GRAPH_SDPA_FORCE_PRIMITIVE", "1", 1);
    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    // Set back to avoid affecting other tests
    custom_setenv("_ONEDNN_GRAPH_SDPA_FORCE_PRIMITIVE", "0", 1);

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<float>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

namespace {
union bit32_t {
    float f32;
    uint32_t u32;
};

uint16_t f32_to_bf16(float f32_val) {
    bit32_t bit32;
    bit32.f32 = f32_val;
    return bit32.u32 >> 16;
}

float bf16_to_f32(uint16_t bf16_val) {
    bit32_t bit32;
    bit32.u32 = bf16_val << 16;
    return bit32.f32;
}
} // namespace

TEST(test_large_partition_execute, Bf16Mha_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu, "skip on gpu");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    graph::graph_t g(eng->kind());
    utils::construct_dnnl_float_MHA(&g, dnnl::impl::data_type::bf16);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 7U);

    graph::pass::pass_base_ptr apass = get_pass(
            eng->kind() == graph::engine_kind::cpu ? "float_sdp_fusion_cpu"
                                                   : "float_sdp_fusion_gpu");
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

    std::vector<std::vector<uint16_t>> inputs_data;
    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        // set all the input value to 1.f, then the value after softmax should
        // be 1.f/seq_len, and the final output should be 1.f
        inputs_data.emplace_back(
                utils::product(ltw(lt).vdims()), f32_to_bf16(1.f));
        inputs_ts.emplace_back(*lt, eng, inputs_data.back());
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();

    // correctness check
    auto out_data = outputs_ts[0].as_vec_type<uint16_t>();
    for (size_t i = 0; i < out_data.size(); i++) {
        ASSERT_LT(std::fabs(bf16_to_f32(out_data[i]) - 1.f), 1e-5);
    }
}

TEST(test_large_partition_execute, Bf16GptMha) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();
    SKIP_IF(eng->kind() == graph::engine_kind::gpu, "skip on gpu");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    graph::graph_t g(eng->kind());
    utils::construct_select_float_MHA(
            &g, graph::data_type::bf16, 1, 32, 16, 4096, false, false, true);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 10U);

    graph::pass::pass_base_ptr apass = get_pass("bfloat16_gpt_sdp");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 7U);
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        // For select op's bool input
        if (ltw(lt).data_type() == graph::data_type::boolean)
            inputs_ts.back().fill<uint8_t>();
        else
            inputs_ts.back().fill<bfloat16_t>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, Bf16DistilBertMha) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();
    SKIP_IF(eng->kind() == graph::engine_kind::gpu, "skip on gpu");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    graph::graph_t g(eng->kind());
    utils::construct_select_float_MHA(&g, dnnl::impl::data_type::bf16);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 7U);

    graph::pass::pass_base_ptr apass = get_pass(
            eng->kind() == graph::engine_kind::cpu ? "float_sdp_fusion_cpu"
                                                   : "float_sdp_fusion_gpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 6U);
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        // For select op's bool input
        if (ltw(lt).data_type() == graph::data_type::boolean)
            inputs_ts.back().fill<uint8_t>();
        else
            inputs_ts.back().fill<bfloat16_t>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, Int8Bf16Mha_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu, "skip on gpu");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    graph::graph_t g(eng->kind());
    utils::construct_int8_bf16_MHA(&g);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 19U);

    graph::pass::pass_base_ptr apass = get_pass("int8_bf16_sdp_fusion");
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : partition_inputs) {
        inputs_ts.emplace_back(lt, eng);
    }

    for (auto &lt : partition_outputs) {
        outputs_ts.emplace_back(lt, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, Int8Bf16DistilBertMha_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu, "skip on gpu");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    graph::graph_t g(eng->kind());
    utils::construct_int8_bf16_MHA(
            &g, 1, 128, 12, 768, false, true, true, false);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 19U);

    graph::pass::pass_base_ptr apass = get_pass("int8_bf16_sdp_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 6U);
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : partition_inputs) {
        inputs_ts.emplace_back(lt, eng);
    }

    for (auto &lt : partition_outputs) {
        outputs_ts.emplace_back(lt, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_large_partition_execute, Int8Bf16GptMha_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu, "skip on gpu");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    graph::graph_t g(eng->kind());
    utils::construct_int8_bf16_MHA(
            &g, 1, 32, 16, 4096, false, true, false, true);
    g.finalize();

    ASSERT_EQ(g.get_ops().size(), 20U);

    graph::pass::pass_base_ptr apass = get_pass("int8_bf16_gpt_sdp");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 7U);
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

    std::vector<test_tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : partition_inputs) {
        inputs_ts.emplace_back(lt, eng);
    }

    for (auto &lt : partition_outputs) {
        outputs_ts.emplace_back(lt, eng);
    }

    ASSERT_EQ(cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                      test_tensor_t::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}
