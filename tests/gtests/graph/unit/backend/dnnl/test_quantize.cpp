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

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"

TEST(Execute, QuantizePerTensor) {
    impl::engine_t *engine = get_engine();

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>(impl::op_attr::scales, {0.1f});
    int64_t zps = engine->kind() == impl::engine_kind::gpu ? 0 : 10;
    quantize.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zps});
    quantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    quantize.set_attr<int64_t>(impl::op_attr::axis, 0);

    test::vector<float> src {1.0, 0.0, 1.0, 2.0};
    test::vector<uint8_t> dst(src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst = engine->kind() == impl::engine_kind::gpu
            ? test::vector<uint8_t> {10, 0, 10, 20}
            : test::vector<uint8_t> {20, 10, 20, 30};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::u8);

    quantize.add_input(src_lt);
    quantize.add_output(dst_lt);

    impl::graph_t g(engine->kind());
    g.add_op(&quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass(
            engine->kind() == impl::engine_kind::cpu ? "quant_pass_cpu"
                                                     : "quant_pass_gpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, engine, src.data());
    impl::tensor_t dst_ts(dst_lt, engine, dst.data());

    impl::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts}, {dst_ts}), impl::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, QuantizePerTensorAnyLayout) {
    impl::engine_t *engine = get_engine();

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>(impl::op_attr::scales, {0.1f});
    int64_t zps = engine->kind() == impl::engine_kind::gpu ? 0 : 10;
    quantize.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zps});
    quantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    quantize.set_attr<int64_t>(impl::op_attr::axis, 0);

    test::vector<float> src {1.0, 0.0, 1.0, 2.0};
    test::vector<uint8_t> dst(src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst = engine->kind() == impl::engine_kind::gpu
            ? test::vector<uint8_t> {10, 0, 10, 20}
            : test::vector<uint8_t> {20, 10, 20, 30};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2}, impl::data_type::u8, impl::layout_type::any);

    quantize.add_input(src_lt);
    quantize.add_output(dst_lt);

    impl::graph_t g(engine->kind());
    g.add_op(&quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass(
            engine->kind() == impl::engine_kind::cpu ? "quant_pass_cpu"
                                                     : "quant_pass_gpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, engine, src.data());
    impl::tensor_t dst_ts(dst_lt, engine, dst.data());

    impl::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts}, {dst_ts}), impl::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, QuantizePerChannelSymmetric) {
    impl::engine_t *engine = get_engine();

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>(impl::op_attr::scales, {0.1f, 0.2f});
    quantize.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {0, 0});
    quantize.set_attr<std::string>(impl::op_attr::qtype, "per_channel");
    quantize.set_attr<int64_t>(impl::op_attr::axis, 1);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<int8_t> dst(src.size(), 0);

    // int8 = f32 / scales
    test::vector<int8_t> ref_dst {-10, 0, 5, 10};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::s8);

    quantize.add_input(src_lt);
    quantize.add_output(dst_lt);

    impl::graph_t g(engine->kind());
    g.add_op(&quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass(
            engine->kind() == impl::engine_kind::cpu ? "quant_pass_cpu"
                                                     : "quant_pass_gpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, engine, src.data());
    impl::tensor_t dst_ts(dst_lt, engine, dst.data());

    impl::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts}, {dst_ts}), impl::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, TypecastQuantize) {
    impl::engine_t *engine = get_engine();
    impl::stream_t *strm = get_stream();

    std::vector<int64_t> src_shape = {1, 8, 16};
    test::vector<float> src_data(product(src_shape));
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });
    impl::op_t typecast(0, impl::op_kind::TypeCast, "typecast");
    impl::op_t quantize(1, impl::op_kind::Quantize, "quantize");
    quantize.set_attr<std::vector<float>>(impl::op_attr::scales, {0.1f});
    int64_t zps = engine->kind() == impl::engine_kind::gpu ? 0 : 10;
    quantize.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {zps});
    quantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    quantize.set_attr<int64_t>(impl::op_attr::axis, 0);

    impl::logical_tensor_t src_bf16
            = utils::logical_tensor_init(0, src_shape, impl::data_type::bf16);
    impl::logical_tensor_t src_f32
            = utils::logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_int8
            = utils::logical_tensor_init(2, src_shape, impl::data_type::u8);

    typecast.add_input(src_bf16);
    typecast.add_output(src_f32);

    quantize.add_input(src_f32);
    quantize.add_output(dst_int8);

    impl::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&typecast), impl::status::success);
    ASSERT_EQ(g.add_op(&quantize), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("typecast_quantize_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {&src_bf16};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_int8};

    ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine), impl::status::success);

    test::vector<int8_t> dst_data(product(src_shape));
    impl::tensor_t src_ts(src_bf16, engine, src_data.data());
    impl::tensor_t dst_ts(dst_int8, engine, dst_data.data());
    ASSERT_EQ(cp.execute(strm, {src_ts}, {dst_ts}), impl::status::success);
    strm->wait();
}

TEST(Execute, DynamicQuantizeS32ZpsPerTensor) {
    // default engine kind is cpu.
    impl::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_quantize(impl::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dync_quantize.set_attr<int64_t>(impl::op_attr::axis, 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f};
    test::vector<int32_t> zps {10};
    test::vector<uint8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {1}, impl::data_type::s32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::u8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_input(zps_lt);
    dync_quantize.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&dync_quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_quant_pass_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scales_lt, &zps_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, eng, src.data());
    impl::tensor_t scales_ts(scales_lt, eng, scales.data());
    impl::tensor_t zps_ts(zps_lt, eng, zps.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts, scales_ts, zps_ts}, {dst_ts}),
            impl::status::success);
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
    impl::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_quantize(impl::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>(impl::op_attr::qtype, "per_channel");
    dync_quantize.set_attr<int64_t>(impl::op_attr::axis, 1);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f, 0.2f};
    test::vector<int32_t> zps {10, 20};
    test::vector<uint8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 25, 30};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::s32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::u8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_input(zps_lt);
    dync_quantize.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&dync_quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_quant_pass_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scales_lt, &zps_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, eng, src.data());
    impl::tensor_t scales_ts(scales_lt, eng, scales.data());
    impl::tensor_t zps_ts(zps_lt, eng, zps.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts, scales_ts, zps_ts}, {dst_ts}),
            impl::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicQuantizeS8ZpsPerTensor) {
    // default engine kind is cpu.
    impl::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_quantize(impl::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dync_quantize.set_attr<int64_t>(impl::op_attr::axis, 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f};
    test::vector<int8_t> zps {10};
    test::vector<uint8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {1}, impl::data_type::s8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::u8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_input(zps_lt);
    dync_quantize.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&dync_quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_quant_pass_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scales_lt, &zps_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, eng, src.data());
    impl::tensor_t scales_ts(scales_lt, eng, scales.data());
    impl::tensor_t zps_ts(zps_lt, eng, zps.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts, scales_ts, zps_ts}, {dst_ts}),
            impl::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicQuantizeNoZpsPerTensor) {
    // default engine kind is cpu.
    impl::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_quantize(impl::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    dync_quantize.set_attr<int64_t>(impl::op_attr::axis, 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f};
    test::vector<int8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<int8_t> ref_dst {-10, 0, 10, 20};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::s8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&dync_quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_quant_pass_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &scales_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, eng, src.data());
    impl::tensor_t scales_ts(scales_lt, eng, scales.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts, scales_ts}, {dst_ts}),
            impl::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(ExecuteSubgraphInt8, SoftmaxTypecastQuant) {
    impl::engine_t *engine = get_engine();
    impl::stream_t *strm = get_stream();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && engine->kind() == impl::engine_kind::cpu,
            "Skip bf16 test for systems that do not support avx512_core.");

    std::vector<int64_t> softmax_shape {2, 2, 2};
    test::vector<float> src_data(product(softmax_shape));

    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> src_distribution(0.f, 1.f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return src_distribution(generator); });

    impl::op_t softmax_op(0, impl::op_kind::SoftMax, "softmax");
    softmax_op.set_attr<int64_t>(impl::op_attr::axis, 2);
    impl::op_t typecast(1, impl::op_kind::TypeCast, "typecast");
    impl::op_t quantize(2, impl::op_kind::Quantize, "quantize");
    quantize.set_attr<std::vector<float>>(impl::op_attr::scales, {0.1f});
    quantize.set_attr<std::vector<int64_t>>(impl::op_attr::zps, {0});
    quantize.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, softmax_shape, impl::data_type::bf16);
    impl::logical_tensor_t softmax_dst = utils::logical_tensor_init(
            1, softmax_shape, impl::data_type::bf16);
    impl::logical_tensor_t tc_dst = utils::logical_tensor_init(
            2, softmax_shape, impl::data_type::f32);
    impl::logical_tensor_t quant_dst
            = utils::logical_tensor_init(3, softmax_shape, impl::data_type::u8);

    softmax_op.add_input(src);
    softmax_op.add_output(softmax_dst);
    typecast.add_input(softmax_dst);
    typecast.add_output(tc_dst);
    quantize.add_input(tc_dst);
    quantize.add_output(quant_dst);

    impl::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&softmax_op), impl::status::success);
    ASSERT_EQ(g.add_op(&typecast), impl::status::success);
    ASSERT_EQ(g.add_op(&quantize), impl::status::success);
    ASSERT_EQ(g.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass = get_pass("softmax_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {&src};
    std::vector<const impl::logical_tensor_t *> lt_outs {&quant_dst};

    ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine), impl::status::success);

    test::vector<float> dst_data(product(softmax_shape));
    test::vector<float> ref_data(product(softmax_shape));
    impl::tensor_t src_ts(src, engine, src_data.data());
    impl::tensor_t dst_ts(quant_dst, engine, dst_data.data());
    impl::tensor_t ref_ts(quant_dst, engine, ref_data.data());

    ASSERT_EQ(run_graph(g, {src_ts}, {ref_ts}, *engine, *strm),
            impl::status::success);
    ASSERT_EQ(cp.execute(strm, {src_ts}, {dst_ts}), impl::status::success);
    strm->wait();
    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_FLOAT_EQ(ref_data[i], dst_data[i]);
    }
}
