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
#include "gtest/gtest.h"

#include "cpp/unit/backend/dnnl/dnnl_test_common.hpp"
#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Execute, DequantizePerTensor) {
    impl::engine_t &engine = get_engine();

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>(impl::op_attr::scales, {0.1f});
    int64_t zps = engine.kind() == impl::engine_kind::gpu ? 0 : 10;
    dequantize.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zps});
    dequantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dequantize.set_attr<int64_t>(impl::op_attr::axis, 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst = engine.kind() == impl::engine_kind::gpu
            ? test::vector<float> {0.0, 1.0, 2.0, 3.0}
            : test::vector<float> {-1.0, 0.0, 1.0, 2.0};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::f32);

    dequantize.add_input(src_lt);
    dequantize.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&dequantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dequant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DequantizePerTensorAnyLayout) {
    impl::engine_t &engine = get_engine();

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>(impl::op_attr::scales, {0.1f});
    int64_t zps = engine.kind() == impl::engine_kind::gpu ? 0 : 10;
    dequantize.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zps});
    dequantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dequantize.set_attr<int64_t>(impl::op_attr::axis, 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst = engine.kind() == impl::engine_kind::gpu
            ? test::vector<float> {0.0, 1.0, 2.0, 3.0}
            : test::vector<float> {-1.0, 0.0, 1.0, 2.0};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    dequantize.add_input(src_lt);
    dequantize.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&dequantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dequant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DequantizePerChannelSymmetric) {
    impl::engine_t &engine = get_engine();

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>(
            impl::op_attr::scales, {0.1f, 0.2f});
    dequantize.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {0, 0});
    dequantize.set_attr<std::string>(impl::op_attr::qtype, "per_channel");
    dequantize.set_attr<int64_t>(impl::op_attr::axis, 1);

    test::vector<int8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * int8
    test::vector<float> ref_dst {0.0, 1, 4, 6};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::s8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::f32);

    dequantize.add_input(src_lt);
    dequantize.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&dequantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dequant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicDequantizeS32ZpsPerTensor) {
    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    SKIP_IF(eng.kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_dequantize(impl::op_kind::DynamicDequantize);
    dync_dequantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dync_dequantize.set_attr<int64_t>(impl::op_attr::axis, 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> scales {0.1f};
    test::vector<int32_t> zps {10};
    test::vector<float> dst(src.size(), 0);
    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {-1.0, 0.0, 1.0, 2.0};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {1}, impl::data_type::s32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::f32);

    dync_dequantize.add_input(src_lt);
    dync_dequantize.add_input(scales_lt);
    dync_dequantize.add_input(zps_lt);
    dync_dequantize.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&dync_dequantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_dequant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scales_lt, &zps_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scales_ts(scales_lt, &eng, scales.data());
    impl::tensor_t zps_ts(zps_lt, &eng, zps.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts, scales_ts, zps_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicDequantizeNoZpsPerTensor) {
    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    SKIP_IF(eng.kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_dequantize(impl::op_kind::DynamicDequantize);
    dync_dequantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dync_dequantize.set_attr<int64_t>(impl::op_attr::axis, 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> scales {0.1f};
    test::vector<float> dst(src.size(), 0);
    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {0, 1.0, 2.0, 3.0};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::f32);

    dync_dequantize.add_input(src_lt);
    dync_dequantize.add_input(scales_lt);
    dync_dequantize.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&dync_dequantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_dequant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &scales_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scales_ts(scales_lt, &eng, scales.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts, scales_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}
