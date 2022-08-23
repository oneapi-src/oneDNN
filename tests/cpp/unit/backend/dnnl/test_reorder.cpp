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

TEST(Execute, ReorderData) {
    impl::engine_t &engine = get_engine();

    test::vector<float> src {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    test::vector<float> ref_dst {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    test::vector<float> dst(src.size(), 10);

    impl::op_t reorder_op(impl::op_kind::Reorder);

    // prepare input/output logical tensor
    // [[1, 2, 3],
    //  [4, 5, 6]]
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, impl::data_type::f32);
    reorder_op.add_input(src_lt);
    reorder_op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&reorder_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("reorder_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    cp.execute(&stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, Int8Reorder) {
    /*
        dequant
        |
        reorder
        |
        quant
    */
    impl::engine_t &engine = get_engine();

    test::vector<uint8_t> int8_src {1, 2, 3, 4, 5, 6};
    test::vector<uint8_t> int8_dst(int8_src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {9, 36, 18, 45, 27, 54};

    impl::graph_t agraph(engine.kind());
    std::vector<int64_t> zps1 = {0};
    std::vector<int64_t> zps2 = {0};
    std::vector<float> scales1 = {3.f};
    std::vector<float> scales2 = {1 / 3.f};
    impl::op_t dequant {0, impl::op_kind::Dequantize, "dequant"};
    dequant.set_attr(impl::op_attr::scales, scales1);
    dequant.set_attr(impl::op_attr::zps, zps1);
    impl::op_t reorder {1, impl::op_kind::Reorder, "reorder"};
    impl::op_t quant {2, impl::op_kind::Quantize, "quant"};
    quant.set_attr(impl::op_attr::scales, scales2);
    quant.set_attr(impl::op_attr::zps, zps2);

    impl::logical_tensor_t int8_src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::u8);
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            1, {2, 3}, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {2, 3}, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t int8_dst_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, impl::data_type::u8);
    dequant.add_input(int8_src_lt);
    dequant.add_output(src_lt);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);
    quant.add_input(dst_lt);
    quant.add_output(int8_dst_lt);

    ASSERT_EQ(agraph.add_op(&dequant), impl::status::success);
    ASSERT_EQ(agraph.add_op(&reorder), impl::status::success);
    ASSERT_EQ(agraph.add_op(&quant), impl::status::success);

    ASSERT_EQ(agraph.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass = get_pass("int8_reorder_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    auto part = agraph.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&int8_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&int8_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(int8_src_lt, &engine, int8_src.data());
    impl::tensor_t dst_ts(int8_dst_lt, &engine, int8_dst.data());

    cp.execute(&stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < int8_src.size(); ++i) {
        ASSERT_EQ(int8_dst[i], ref_dst[i]);
    }
}

TEST(Compile, ReorderNegativeInput) {
    impl::engine_t &engine = get_engine();

    impl::op_t reorder_op(impl::op_kind::Reorder);

    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);
    // failure case: different shape (dims)
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, {1, 1, 2}, impl::data_type::f32);

    reorder_op.add_input(src_lt);
    reorder_op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&reorder_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("reorder_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine),
            impl::status::invalid_shape);
}

TEST(Execute, ReorderDataBf16) {
    impl::engine_t &engine = get_engine();

    test::vector<float> src {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    test::vector<float> ref_dst {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    test::vector<float> dst(src.size(), 10);

    impl::op_t reorder_op(impl::op_kind::Reorder);

    // prepare input/output logical tensor
    // [[1, 2, 3],
    //  [4, 5, 6]]
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::bf16);
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, impl::data_type::bf16);
    reorder_op.add_input(src_lt);
    reorder_op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&reorder_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("reorder_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    cp.execute(&stream, {src_ts}, {dst_ts});
    stream.wait();
}

TEST(Execute, ReorderAddBf16) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    test::vector<uint16_t> src {1, 2, 3, 4, 5, 6};
    test::vector<uint16_t> post_src {1, 2, 3, 4, 5, 6};
    test::vector<uint16_t> dst(src.size(), 0);

    impl::graph_t agraph(eng.kind());
    impl::op_t reorder {0, impl::op_kind::Reorder, "reorder"};
    impl::op_t add {1, impl::op_kind::Add, "add"};
    add.set_attr<std::string>(impl::op_attr::auto_broadcast, "none");

    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::bf16);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, impl::data_type::bf16);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);

    impl::logical_tensor_t add_src_lt = utils::logical_tensor_init(
            2, {2, 3}, {1, 2}, impl::data_type::bf16);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, impl::data_type::bf16);
    add.add_input(dst_lt);
    add.add_input(add_src_lt);
    add.add_output(add_dst_lt);

    ASSERT_EQ(agraph.add_op(&reorder), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add), impl::status::success);

    ASSERT_EQ(agraph.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass = get_pass("reorder_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);
    auto part = agraph.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    // compile reorder_add partition
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &add_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t post_src_ts(add_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(add_dst_lt, &eng, dst.data());
    cp.execute(&strm, {src_ts, post_src_ts}, {dst_ts});
    strm.wait();
}

TEST(Compile, ReorderAddGetInplacePair) {
    impl::engine_t &eng = get_engine();

    impl::graph_t agraph(eng.kind());
    impl::op_t reorder {0, impl::op_kind::Reorder, "reorder"};
    impl::op_t add {1, impl::op_kind::Add, "add"};
    add.set_attr<std::string>(impl::op_attr::auto_broadcast, "none");

    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, impl::data_type::f32);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);

    impl::logical_tensor_t add_src_lt = utils::logical_tensor_init(
            2, {2, 3}, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, impl::data_type::f32);
    add.add_input(dst_lt);
    add.add_input(add_src_lt);
    add.add_output(add_dst_lt);

    ASSERT_EQ(agraph.add_op(&reorder), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add), impl::status::success);

    ASSERT_EQ(agraph.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass = get_pass("reorder_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);
    auto part = agraph.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    // compile reorder_add partition
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &add_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    std::vector<impl::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1);
    ASSERT_EQ(inplace_pairs[0].input_id, add_src_lt.id);
    ASSERT_EQ(inplace_pairs[0].output_id, add_dst_lt.id);
}

TEST(Execute, Int8ReorderAdd) {
    /*
        dequant
        |
      reorder dequant
        |    /
        add
        |
        quant
    */
    impl::engine_t &engine = get_engine();

    test::vector<uint8_t> int8_src {1, 2, 3, 4, 5, 6};
    test::vector<uint8_t> int8_src_other {1, 2, 3, 4, 5, 6};
    test::vector<uint8_t> int8_dst(int8_src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {2, 6, 5, 9, 8, 12};

    impl::graph_t agraph(engine.kind());
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.f};
    impl::op_t dequant {0, impl::op_kind::Dequantize, "dequant"};
    dequant.set_attr(impl::op_attr::scales, scales);
    dequant.set_attr(impl::op_attr::zps, zps);
    impl::op_t reorder {1, impl::op_kind::Reorder, "reorder"};
    impl::op_t dequant_other {2, impl::op_kind::Dequantize, "dequant_other"};
    dequant_other.set_attr(impl::op_attr::scales, scales);
    dequant_other.set_attr(impl::op_attr::zps, zps);
    impl::op_t add {3, impl::op_kind::Add, "add"};
    add.set_attr<std::string>(impl::op_attr::auto_broadcast, "none");
    impl::op_t quant {4, impl::op_kind::Quantize, "quant"};
    quant.set_attr(impl::op_attr::scales, scales);
    quant.set_attr(impl::op_attr::zps, zps);

    // 1,2,3
    // 4,5,6
    impl::logical_tensor_t int8_src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::u8);
    // 1,3,5
    // 2,4,6
    impl::logical_tensor_t int8_src_other_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, impl::data_type::u8);
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            2, {2, 3}, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t src_other_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {2, 3}, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_add_lt = utils::logical_tensor_init(
            5, {2, 3}, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t int8_dst_add_lt = utils::logical_tensor_init(
            6, {2, 3}, {1, 2}, impl::data_type::u8);
    dequant.add_input(int8_src_lt);
    dequant.add_output(src_lt);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);
    dequant_other.add_input(int8_src_other_lt);
    dequant_other.add_output(src_other_lt);
    add.add_input(dst_lt);
    add.add_input(src_other_lt);
    add.add_output(dst_add_lt);
    quant.add_input(dst_add_lt);
    quant.add_output(int8_dst_add_lt);

    ASSERT_EQ(agraph.add_op(&dequant), impl::status::success);
    ASSERT_EQ(agraph.add_op(&dequant_other), impl::status::success);
    ASSERT_EQ(agraph.add_op(&reorder), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add), impl::status::success);
    ASSERT_EQ(agraph.add_op(&quant), impl::status::success);

    ASSERT_EQ(agraph.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass
            = get_pass(engine.kind() == impl::engine_kind::cpu
                            ? "int8_reorder_sum_fusion_cpu"
                            : "int8_reorder_sum_fusion_gpu");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);
    auto part = agraph.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &int8_src_lt, &int8_src_other_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&int8_dst_add_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(int8_src_lt, &engine, int8_src.data());
    impl::tensor_t src_other_ts(
            int8_src_other_lt, &engine, int8_src_other.data());
    impl::tensor_t dst_ts(int8_dst_add_lt, &engine, int8_dst.data());

    cp.execute(&stream, {src_ts, src_other_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < int8_src.size(); ++i) {
        ASSERT_EQ(int8_dst[i], ref_dst[i]);
    }
}
