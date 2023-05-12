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

TEST(Execute, QuantizePerTensor) {
    graph::engine_t *engine = get_engine();

    std::vector<float> scales = {1.f, 0.1f};
    std::vector<int64_t> zps = {0, 10};

    for_(const auto &scale : scales)
    for (const auto &zp : zps) {
        graph::op_t quantize(graph::op_kind::Quantize);
        quantize.set_attr<std::vector<float>>(graph::op_attr::scales, {scale});
        quantize.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp});
        quantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        quantize.set_attr<int64_t>(graph::op_attr::axis, 0);

        test::vector<float> src {1.0, 0.0, 1.0, 2.0};
        test::vector<uint8_t> dst(src.size(), 0);

        // int8 = f32 / scales + zero_points
        test::vector<uint8_t> ref_dst = scale == 1.f
                ? test::vector<uint8_t> {1, 0, 1, 2}
                : test::vector<uint8_t> {10, 0, 10, 20};

        for (size_t i = 0; i < dst.size(); ++i)
            ref_dst[i] += zp;

        // prepare input/output logical tensor
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 2, 2}, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
                1, {1, 2, 2}, graph::data_type::u8);

        quantize.add_input(src_lt);
        quantize.add_output(dst_lt);

        graph::graph_t g(engine->kind());
        g.add_op(&quantize);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("quant_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);
        std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
        ASSERT_EQ(p.compile(&cp, inputs, outputs, engine),
                graph::status::success);

        graph::logical_tensor_t lt;
        cp.query_logical_tensor(dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(src_lt, engine, src.data());
        graph::tensor_t dst_ts(dst_lt, engine, dst.data());

        graph::stream_t *strm = get_stream();
        ASSERT_EQ(cp.execute(strm, {src_ts}, {dst_ts}), graph::status::success);
        strm->wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
        }
    }
}

TEST(Execute, QuantizePerTensorAnyLayout) {
    graph::engine_t *engine = get_engine();

    graph::op_t quantize(graph::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>(graph::op_attr::scales, {0.1f});
    int64_t zps = engine->kind() == graph::engine_kind::gpu ? 0 : 10;
    quantize.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zps});
    quantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    quantize.set_attr<int64_t>(graph::op_attr::axis, 0);

    test::vector<float> src {1.0, 0.0, 1.0, 2.0};
    test::vector<uint8_t> dst(src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst = engine->kind() == graph::engine_kind::gpu
            ? test::vector<uint8_t> {10, 0, 10, 20}
            : test::vector<uint8_t> {20, 10, 20, 30};

    // prepare input/output logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2}, graph::data_type::u8, graph::layout_type::any);

    quantize.add_input(src_lt);
    quantize.add_output(dst_lt);

    graph::graph_t g(engine->kind());
    g.add_op(&quantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);
    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, engine, src.data());
    graph::tensor_t dst_ts(dst_lt, engine, dst.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts}, {dst_ts}), graph::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, QuantizePerChannelSymmetric) {
    graph::engine_t *engine = get_engine();

    graph::op_t quantize(graph::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>(graph::op_attr::scales, {0.1f, 0.2f});
    quantize.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {0, 0});
    quantize.set_attr<std::string>(graph::op_attr::qtype, "per_channel");
    quantize.set_attr<int64_t>(graph::op_attr::axis, 1);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<int8_t> dst(src.size(), 0);

    // int8 = f32 / scales
    test::vector<int8_t> ref_dst {-10, 0, 5, 10};

    // prepare input/output logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, graph::data_type::s8);

    quantize.add_input(src_lt);
    quantize.add_output(dst_lt);

    graph::graph_t g(engine->kind());
    g.add_op(&quantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);
    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, engine, src.data());
    graph::tensor_t dst_ts(dst_lt, engine, dst.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts}, {dst_ts}), graph::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, TypecastQuantize) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> src_shape = {1, 8, 16};
    test::vector<float> src_data(product(src_shape));
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });
    graph::op_t typecast(0, graph::op_kind::TypeCast, "typecast");
    graph::op_t quantize(1, graph::op_kind::Quantize, "quantize");
    quantize.set_attr<std::vector<float>>(graph::op_attr::scales, {0.1f});
    int64_t zps = engine->kind() == graph::engine_kind::gpu ? 0 : 10;
    quantize.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zps});
    quantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    quantize.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(0, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t src_f32
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_int8
            = utils::logical_tensor_init(2, src_shape, graph::data_type::u8);

    typecast.add_input(src_bf16);
    typecast.add_output(src_f32);

    quantize.add_input(src_f32);
    quantize.add_output(dst_int8);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&typecast), graph::status::success);
    ASSERT_EQ(g.add_op(&quantize), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("typecast_quantize_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {&src_bf16};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_int8};

    ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine), graph::status::success);

    test::vector<int8_t> dst_data(product(src_shape));
    graph::tensor_t src_ts(src_bf16, engine, src_data.data());
    graph::tensor_t dst_ts(dst_int8, engine, dst_data.data());
    ASSERT_EQ(cp.execute(strm, {src_ts}, {dst_ts}), graph::status::success);
    strm->wait();
}

TEST(Execute, DynamicQuantizeS32ZpsPerTensor) {
    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    graph::op_t dync_quantize(graph::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dync_quantize.set_attr<int64_t>(graph::op_attr::axis, 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f};
    test::vector<int32_t> zps {10};
    test::vector<uint8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {1}, graph::data_type::s32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, graph::data_type::u8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_input(zps_lt);
    dync_quantize.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&dync_quantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("dync_quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &scales_lt, &zps_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t scales_ts(scales_lt, eng, scales.data());
    graph::tensor_t zps_ts(zps_lt, eng, zps.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts, scales_ts, zps_ts}, {dst_ts}),
            graph::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicQuantizeS32ZpsPerChannel) {
    // oneDNN reorder primitive didn't support per channel asymmetric quantize
    // regression?
    SKIP_IF(true,
            "oneDNN reorder primitive didn't support per channel asymmetric "
            "quantize");

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    graph::op_t dync_quantize(graph::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>(graph::op_attr::qtype, "per_channel");
    dync_quantize.set_attr<int64_t>(graph::op_attr::axis, 1);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f, 0.2f};
    test::vector<int32_t> zps {10, 20};
    test::vector<uint8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 25, 30};

    // prepare logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {2}, graph::data_type::f32);
    graph::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {2}, graph::data_type::s32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, graph::data_type::u8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_input(zps_lt);
    dync_quantize.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&dync_quantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("dync_quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &scales_lt, &zps_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t scales_ts(scales_lt, eng, scales.data());
    graph::tensor_t zps_ts(zps_lt, eng, zps.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts, scales_ts, zps_ts}, {dst_ts}),
            graph::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicQuantizeS8ZpsPerTensor) {
    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    graph::op_t dync_quantize(graph::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dync_quantize.set_attr<int64_t>(graph::op_attr::axis, 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f};
    test::vector<int8_t> zps {10};
    test::vector<uint8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {1}, graph::data_type::s8);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, graph::data_type::u8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_input(zps_lt);
    dync_quantize.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&dync_quantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("dync_quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_lt, &scales_lt, &zps_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t scales_ts(scales_lt, eng, scales.data());
    graph::tensor_t zps_ts(zps_lt, eng, zps.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts, scales_ts, zps_ts}, {dst_ts}),
            graph::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicQuantizeNoZpsPerTensor) {
    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    graph::op_t dync_quantize(graph::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dync_quantize.set_attr<int64_t>(graph::op_attr::axis, 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f};
    test::vector<int8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<int8_t> ref_dst {-10, 0, 10, 20};

    // prepare logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, graph::data_type::s8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&dync_quantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("dync_quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &scales_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t scales_ts(scales_lt, eng, scales.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts, scales_ts}, {dst_ts}),
            graph::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, QuantizeZeroVolume) {
    graph::engine_t *engine = get_engine();

    graph::op_t quantize(graph::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>(graph::op_attr::scales, {0.1f});
    quantize.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {0});
    quantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    quantize.set_attr<int64_t>(graph::op_attr::axis, 0);

    test::vector<float> src(0, 0);
    test::vector<uint8_t> dst(0, 0);

    // prepare input/output logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {0, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(1,
            {DNNL_GRAPH_UNKNOWN_DIM, DNNL_GRAPH_UNKNOWN_DIM,
                    DNNL_GRAPH_UNKNOWN_DIM},
            {DNNL_GRAPH_UNKNOWN_DIM, DNNL_GRAPH_UNKNOWN_DIM,
                    DNNL_GRAPH_UNKNOWN_DIM},
            graph::data_type::u8);

    quantize.add_input(src_lt);
    quantize.add_output(dst_lt);

    graph::graph_t g(engine->kind());
    g.add_op(&quantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);
    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);
    ASSERT_EQ(lt.dims[0], 0U);
    ASSERT_EQ(lt.dims[1], 2U);
    ASSERT_EQ(lt.dims[2], 2U);
    ASSERT_EQ(lt.layout.strides[0], 4U);
    ASSERT_EQ(lt.layout.strides[1], 2U);
    ASSERT_EQ(lt.layout.strides[2], 1U);
    ASSERT_EQ(graph::logical_tensor_wrapper_t(lt).size(), 0U);

    graph::tensor_t src_ts(src_lt, engine, src.data());
    graph::tensor_t dst_ts(dst_lt, engine, dst.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts}, {dst_ts}), graph::status::success);
    strm->wait();
}
