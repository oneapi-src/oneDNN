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

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Execute, DequantizePerTensor) {
    graph::engine_t *engine = get_engine();
    std::vector<float> scales = {1.f, 0.1f};
    std::vector<int64_t> zps = {0, 10};

    for_(const auto &scale : scales)
    for (const auto &zp : zps) {
        graph::op_t dequantize(graph::op_kind::Dequantize);
        dequantize.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale});
        dequantize.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp});
        dequantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dequantize.set_attr<int64_t>(graph::op_attr::axis, 0);

        test::vector<uint8_t> src {0, 10, 20, 30};
        test::vector<float> dst(src.size(), 0);

        // f32 = scales * (int8 - zero_points)
        test::vector<float> ref_dst = zp == 0
                ? test::vector<float> {0.0, 10.0, 20.0, 30.0}
                : test::vector<float> {-10.0, 0.0, 10.0, 20.0};
        for (size_t i = 0; i < ref_dst.size(); i++)
            ref_dst[i] *= scale;
        // prepare input/output logical tensor
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 2, 2}, graph::data_type::u8);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
                1, {1, 2, 2}, graph::data_type::f32);

        dequantize.add_input(src_lt);
        dequantize.add_output(dst_lt);

        graph::graph_t g(engine->kind());
        g.add_op(&dequantize);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("dequant_pass");
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

TEST(Execute, DequantizePerTensorAnyLayout) {
    graph::engine_t *engine = get_engine();

    graph::op_t dequantize(graph::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>(graph::op_attr::scales, {0.1f});
    int64_t zps = engine->kind() == graph::engine_kind::gpu ? 0 : 10;
    dequantize.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zps});
    dequantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dequantize.set_attr<int64_t>(graph::op_attr::axis, 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst = engine->kind() == graph::engine_kind::gpu
            ? test::vector<float> {0.0, 1.0, 2.0, 3.0}
            : test::vector<float> {-1.0, 0.0, 1.0, 2.0};

    // prepare input/output logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, graph::data_type::u8);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2}, graph::data_type::f32, graph::layout_type::any);

    dequantize.add_input(src_lt);
    dequantize.add_output(dst_lt);

    graph::graph_t g(engine->kind());
    g.add_op(&dequantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("dequant_pass");
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

TEST(Execute, DequantizePerChannelSymmetric) {
    graph::engine_t *engine = get_engine();

    graph::op_t dequantize(graph::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>(
            graph::op_attr::scales, {0.1f, 0.2f});
    dequantize.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {0, 0});
    dequantize.set_attr<std::string>(graph::op_attr::qtype, "per_channel");
    dequantize.set_attr<int64_t>(graph::op_attr::axis, 1);

    test::vector<int8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * int8
    test::vector<float> ref_dst {0.0, 1, 4, 6};

    // prepare input/output logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, graph::data_type::s8);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, graph::data_type::f32);

    dequantize.add_input(src_lt);
    dequantize.add_output(dst_lt);

    graph::graph_t g(engine->kind());
    g.add_op(&dequantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("dequant_pass");
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

TEST(Execute, DynamicDequantizeS32ZpsPerTensor) {
    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    graph::op_t dync_dequantize(graph::op_kind::DynamicDequantize);
    dync_dequantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dync_dequantize.set_attr<int64_t>(graph::op_attr::axis, 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> scales {0.1f};
    test::vector<int32_t> zps {10};
    test::vector<float> dst(src.size(), 0);
    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {-1.0, 0.0, 1.0, 2.0};

    // prepare logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, graph::data_type::u8);
    graph::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {1}, graph::data_type::s32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, graph::data_type::f32);

    dync_dequantize.add_input(src_lt);
    dync_dequantize.add_input(scales_lt);
    dync_dequantize.add_input(zps_lt);
    dync_dequantize.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&dync_dequantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("dync_dequant_pass");
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

TEST(Execute, DynamicDequantizeS8ZpsPerTensor) {
    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    graph::op_t dync_dequantize(graph::op_kind::DynamicDequantize);
    dync_dequantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dync_dequantize.set_attr<int64_t>(graph::op_attr::axis, 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> scales {0.1f};
    test::vector<int8_t> zps {10};
    test::vector<float> dst(src.size(), 0);
    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {-1.0, 0.0, 1.0, 2.0};

    // prepare logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, graph::data_type::u8);
    graph::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {1}, graph::data_type::s8);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, graph::data_type::f32);

    dync_dequantize.add_input(src_lt);
    dync_dequantize.add_input(scales_lt);
    dync_dequantize.add_input(zps_lt);
    dync_dequantize.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&dync_dequantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("dync_dequant_pass");
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

TEST(Execute, DynamicDequantizeNoZpsPerTensor) {
    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    graph::op_t dync_dequantize(graph::op_kind::DynamicDequantize);
    dync_dequantize.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dync_dequantize.set_attr<int64_t>(graph::op_attr::axis, 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> scales {0.1f};
    test::vector<float> dst(src.size(), 0);
    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {0, 1.0, 2.0, 3.0};

    // prepare logical tensor
    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, graph::data_type::u8);
    graph::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, graph::data_type::f32);

    dync_dequantize.add_input(src_lt);
    dync_dequantize.add_input(scales_lt);
    dync_dequantize.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&dync_dequantize);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("dync_dequant_pass");
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
