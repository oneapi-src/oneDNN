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

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Execute, ReorderData) {
    graph::engine_t *engine = get_engine();

    test::vector<float> src {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    test::vector<float> ref_dst {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    test::vector<float> dst(src.size(), 10);

    graph::op_t reorder_op(graph::op_kind::Reorder);

    // prepare input/output logical tensor
    // [[1, 2, 3],
    //  [4, 5, 6]]
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, graph::data_type::f32);
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, graph::data_type::f32);
    reorder_op.add_input(src_lt);
    reorder_op.add_output(dst_lt);

    graph::graph_t g(engine->kind());
    g.add_op(&reorder_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("reorder_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    graph::stream_t *stream = get_stream();
    graph::tensor_t src_ts(src_lt, engine, src.data());
    graph::tensor_t dst_ts(dst_lt, engine, dst.data());

    cp.execute(stream, {src_ts}, {dst_ts});
    stream->wait();
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
    graph::engine_t *engine = get_engine();

    test::vector<uint8_t> int8_src {1, 2, 3, 4, 5, 6};
    test::vector<uint8_t> int8_dst(int8_src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {9, 36, 18, 45, 27, 54};

    graph::graph_t agraph(engine->kind());
    std::vector<int64_t> zps1 = {0};
    std::vector<int64_t> zps2 = {0};
    std::vector<float> scales1 = {3.f};
    std::vector<float> scales2 = {1 / 3.f};
    graph::op_t dequant {0, graph::op_kind::Dequantize, "dequant"};
    dequant.set_attr(graph::op_attr::scales, scales1);
    dequant.set_attr(graph::op_attr::zps, zps1);
    graph::op_t reorder {1, graph::op_kind::Reorder, "reorder"};
    graph::op_t quant {2, graph::op_kind::Quantize, "quant"};
    quant.set_attr(graph::op_attr::scales, scales2);
    quant.set_attr(graph::op_attr::zps, zps2);

    graph::logical_tensor_t int8_src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, graph::data_type::u8);
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            1, {2, 3}, {3, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {2, 3}, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t int8_dst_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, graph::data_type::u8);
    dequant.add_input(int8_src_lt);
    dequant.add_output(src_lt);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);
    quant.add_input(dst_lt);
    quant.add_output(int8_dst_lt);

    ASSERT_EQ(agraph.add_op(&dequant), graph::status::success);
    ASSERT_EQ(agraph.add_op(&reorder), graph::status::success);
    ASSERT_EQ(agraph.add_op(&quant), graph::status::success);

    ASSERT_EQ(agraph.finalize(), graph::status::success);

    graph::pass::pass_base_ptr apass = get_pass("int8_reorder_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    auto part = agraph.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&int8_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&int8_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    graph::stream_t *stream = get_stream();
    graph::tensor_t src_ts(int8_src_lt, engine, int8_src.data());
    graph::tensor_t dst_ts(int8_dst_lt, engine, int8_dst.data());

    cp.execute(stream, {src_ts}, {dst_ts});
    stream->wait();
    for (size_t i = 0; i < int8_src.size(); ++i) {
        ASSERT_EQ(int8_dst[i], ref_dst[i]);
    }
}

TEST(Compile, ReorderNegativeInput) {
    graph::engine_t *engine = get_engine();

    graph::op_t reorder_op(graph::op_kind::Reorder);

    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, graph::data_type::f32);
    // failure case: different shape (dims)
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, {1, 1, 2}, graph::data_type::f32);

    reorder_op.add_input(src_lt);
    reorder_op.add_output(dst_lt);

    graph::graph_t g(engine->kind());
    g.add_op(&reorder_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("reorder_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine),
            graph::status::invalid_shape);
}

TEST(Execute, ReorderDataBf16) {
    graph::engine_t *engine = get_engine();

    test::vector<float> src {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    test::vector<float> ref_dst {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    test::vector<float> dst(src.size(), 10);

    graph::op_t reorder_op(graph::op_kind::Reorder);

    // prepare input/output logical tensor
    // [[1, 2, 3],
    //  [4, 5, 6]]
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, graph::data_type::bf16);
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, graph::data_type::bf16);
    reorder_op.add_input(src_lt);
    reorder_op.add_output(dst_lt);

    graph::graph_t g(engine->kind());
    g.add_op(&reorder_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("reorder_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    graph::stream_t *stream = get_stream();
    graph::tensor_t src_ts(src_lt, engine, src.data());
    graph::tensor_t dst_ts(dst_lt, engine, dst.data());

    cp.execute(stream, {src_ts}, {dst_ts});
    stream->wait();
}

TEST(Execute, ReorderAddBf16) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    test::vector<uint16_t> src {1, 2, 3, 4, 5, 6};
    test::vector<uint16_t> post_src {1, 2, 3, 4, 5, 6};
    test::vector<uint16_t> dst(src.size(), 0);

    graph::graph_t agraph(eng->kind());
    graph::op_t reorder {0, graph::op_kind::Reorder, "reorder"};
    graph::op_t add {1, graph::op_kind::Add, "add"};
    add.set_attr<std::string>(graph::op_attr::auto_broadcast, "none");

    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, graph::data_type::bf16);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, graph::data_type::bf16);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);

    graph::logical_tensor_t add_src_lt = utils::logical_tensor_init(
            2, {2, 3}, {1, 2}, graph::data_type::bf16);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, graph::data_type::bf16);
    add.add_input(dst_lt);
    add.add_input(add_src_lt);
    add.add_output(add_dst_lt);

    ASSERT_EQ(agraph.add_op(&reorder), graph::status::success);
    ASSERT_EQ(agraph.add_op(&add), graph::status::success);

    ASSERT_EQ(agraph.finalize(), graph::status::success);

    graph::pass::pass_base_ptr apass = get_pass("reorder_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    auto part = agraph.get_partitions()[0];
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    // compile reorder_add partition
    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &add_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t post_src_ts(add_src_lt, eng, post_src.data());
    graph::tensor_t dst_ts(add_dst_lt, eng, dst.data());
    cp.execute(strm, {src_ts, post_src_ts}, {dst_ts});
    strm->wait();
}

TEST(Compile, ReorderAddGetInplacePair) {
    graph::engine_t *eng = get_engine();

    graph::graph_t agraph(eng->kind());
    graph::op_t reorder {0, graph::op_kind::Reorder, "reorder"};
    graph::op_t add {1, graph::op_kind::Add, "add"};
    add.set_attr<std::string>(graph::op_attr::auto_broadcast, "none");

    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, graph::data_type::f32);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);

    graph::logical_tensor_t add_src_lt = utils::logical_tensor_init(
            2, {2, 3}, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, graph::data_type::f32);
    add.add_input(dst_lt);
    add.add_input(add_src_lt);
    add.add_output(add_dst_lt);

    ASSERT_EQ(agraph.add_op(&reorder), graph::status::success);
    ASSERT_EQ(agraph.add_op(&add), graph::status::success);

    ASSERT_EQ(agraph.finalize(), graph::status::success);

    graph::pass::pass_base_ptr apass = get_pass("reorder_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    auto part = agraph.get_partitions()[0];
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    // compile reorder_add partition
    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &add_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    std::vector<graph::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1U);
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
    graph::engine_t *engine = get_engine();

    test::vector<uint8_t> int8_src {1, 2, 3, 4, 5, 6};
    test::vector<uint8_t> int8_src_other {1, 2, 3, 4, 5, 6};
    test::vector<uint8_t> int8_dst(int8_src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {2, 6, 5, 9, 8, 12};

    graph::graph_t agraph(engine->kind());
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.f};
    graph::op_t dequant {0, graph::op_kind::Dequantize, "dequant"};
    dequant.set_attr(graph::op_attr::scales, scales);
    dequant.set_attr(graph::op_attr::zps, zps);
    graph::op_t reorder {1, graph::op_kind::Reorder, "reorder"};
    graph::op_t dequant_other {2, graph::op_kind::Dequantize, "dequant_other"};
    dequant_other.set_attr(graph::op_attr::scales, scales);
    dequant_other.set_attr(graph::op_attr::zps, zps);
    graph::op_t add {3, graph::op_kind::Add, "add"};
    add.set_attr<std::string>(graph::op_attr::auto_broadcast, "none");
    graph::op_t quant {4, graph::op_kind::Quantize, "quant"};
    quant.set_attr(graph::op_attr::scales, scales);
    quant.set_attr(graph::op_attr::zps, zps);

    // 1,2,3
    // 4,5,6
    graph::logical_tensor_t int8_src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, graph::data_type::u8);
    // 1,3,5
    // 2,4,6
    graph::logical_tensor_t int8_src_other_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, graph::data_type::u8);
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            2, {2, 3}, {3, 1}, graph::data_type::f32);
    graph::logical_tensor_t src_other_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {2, 3}, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst_add_lt = utils::logical_tensor_init(
            5, {2, 3}, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t int8_dst_add_lt = utils::logical_tensor_init(
            6, {2, 3}, {1, 2}, graph::data_type::u8);
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

    ASSERT_EQ(agraph.add_op(&dequant), graph::status::success);
    ASSERT_EQ(agraph.add_op(&dequant_other), graph::status::success);
    ASSERT_EQ(agraph.add_op(&reorder), graph::status::success);
    ASSERT_EQ(agraph.add_op(&add), graph::status::success);
    ASSERT_EQ(agraph.add_op(&quant), graph::status::success);

    ASSERT_EQ(agraph.finalize(), graph::status::success);

    graph::pass::pass_base_ptr apass
            = get_pass(engine->kind() == graph::engine_kind::cpu
                            ? "int8_reorder_sum_fusion_cpu"
                            : "int8_reorder_sum_fusion_gpu");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    auto part = agraph.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &int8_src_lt, &int8_src_other_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&int8_dst_add_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    graph::stream_t *stream = get_stream();
    graph::tensor_t src_ts(int8_src_lt, engine, int8_src.data());
    graph::tensor_t src_other_ts(
            int8_src_other_lt, engine, int8_src_other.data());
    graph::tensor_t dst_ts(int8_dst_add_lt, engine, int8_dst.data());

    cp.execute(stream, {src_ts, src_other_ts}, {dst_ts});
    stream->wait();
    for (size_t i = 0; i < int8_src.size(); ++i) {
        ASSERT_EQ(int8_dst[i], ref_dst[i]);
    }
}
