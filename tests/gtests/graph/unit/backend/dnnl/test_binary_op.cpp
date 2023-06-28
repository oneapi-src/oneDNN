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

#include "interface/c_types_map.hpp"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

#include "gtest/gtest.h"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Execute, BinaryOp) {
    graph::engine_t *eng = get_engine();

    std::vector<graph::op_kind_t> op_kinds = {graph::op_kind::Multiply,
            graph::op_kind::Minimum, graph::op_kind::Maximum,
            graph::op_kind::Divide, graph::op_kind::Subtract,
            graph::op_kind::SquaredDifference};
    std::vector<std::string> pass_names = {"mul_pass", "min_pass", "max_pass",
            "div_pass", "sub_pass", "squareddifference_pass"};

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    auto src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, graph::data_type::f32);
    auto src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, graph::data_type::f32);
    auto dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, graph::data_type::f32);

    for (size_t i = 0; i < op_kinds.size(); i++) {
        graph::op_t binary_op(op_kinds[i]);

        binary_op.add_input(src0_lt);
        binary_op.add_input(src1_lt);
        binary_op.add_output(dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&binary_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass(pass_names[i]);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src0_lt, &src1_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

        graph::tensor_t src0_ts(src0_lt, eng, src0.data());
        graph::tensor_t src1_ts(src1_lt, eng, src1.data());
        graph::tensor_t dst_ts(dst_lt, eng, dst.data());

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(Execute, MulEltwise) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, graph::data_type::f32);

    std::vector<graph::op_kind_t> eltwise_ops
            = {graph::op_kind::ReLU, graph::op_kind::Sigmoid};

    for (size_t i = 0; i < eltwise_ops.size(); i++) {
        graph::op_t mul_op(0, graph::op_kind::Multiply, "mul");
        graph::op_t eltwise_op(1, eltwise_ops[i], "eltwise");

        mul_op.add_input(src0_lt);
        mul_op.add_input(src1_lt);
        mul_op.add_output(mul_dst_lt);
        eltwise_op.add_input(mul_dst_lt);
        eltwise_op.add_output(dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&mul_op);
        g.add_op(&eltwise_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        ASSERT_NE(apass.get(), nullptr);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);
        std::vector<const graph::logical_tensor_t *> inputs {
                &src0_lt, &src1_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

        graph::tensor_t src0_ts(src0_lt, eng, src0.data());
        graph::tensor_t src1_ts(src1_lt, eng, src1.data());
        graph::tensor_t dst_ts(dst_lt, eng, dst.data());

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(Execute, BinaryOpAddFusion) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t tmp_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 3, 3}, graph::data_type::f32);

    std::vector<graph::op_kind_t> bin_ops = {graph::op_kind::Multiply,
            graph::op_kind::Maximum, graph::op_kind::Minimum};

    for (size_t i = 0; i < bin_ops.size(); i++) {
        graph::op_t bin_op(0, bin_ops[i], "bin");
        graph::op_t add_op(1, graph::op_kind::Add, "add");

        bin_op.add_input(src0_lt);
        bin_op.add_input(src1_lt);
        bin_op.add_output(tmp_dst_lt);
        add_op.add_input(tmp_dst_lt);
        add_op.add_input(post_src_lt);
        add_op.add_output(dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&bin_op);
        g.add_op(&add_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);
        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src0_lt, &src1_lt, &post_src_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, eng);

        graph::tensor_t src0_ts(src0_lt, eng, src0.data());
        graph::tensor_t src1_ts(src1_lt, eng, src1.data());
        graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
        graph::tensor_t dst_ts(dst_lt, eng, dst.data());

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(Execute, BinarySub) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0};
    test::vector<float> src1 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> dst(src1.size(), 0.0);
    test::vector<float> ref {1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0};

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {2, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {2, 1, 2, 2}, graph::data_type::f32);

    graph::op_t bin_op(0, graph::op_kind::Subtract, "bin");

    bin_op.add_input(src0_lt);
    bin_op.add_input(src1_lt);
    bin_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&bin_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sub_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < ref.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref[i]);
    }
}

TEST(Execute, MinEltwise) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {-1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t min_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, graph::data_type::f32);

    std::vector<graph::op_kind_t> eltwise_ops
            = {graph::op_kind::ReLU, graph::op_kind::Sigmoid};

    for (size_t i = 0; i < eltwise_ops.size(); i++) {
        graph::op_t min_op(0, graph::op_kind::Minimum, "min");
        graph::op_t eltwise_op(1, eltwise_ops[i], "eltwise");

        min_op.add_input(src0_lt);
        min_op.add_input(src1_lt);
        min_op.add_output(min_dst_lt);
        eltwise_op.add_input(min_dst_lt);
        eltwise_op.add_output(dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&min_op);
        g.add_op(&eltwise_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        ASSERT_NE(apass.get(), nullptr);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);
        std::vector<const graph::logical_tensor_t *> inputs {
                &src0_lt, &src1_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, eng);

        graph::tensor_t src0_ts(src0_lt, eng, src0.data());
        graph::tensor_t src1_ts(src1_lt, eng, src1.data());
        graph::tensor_t dst_ts(dst_lt, eng, dst.data());

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(Execute, MaxEltwise) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {-2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {-1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> ref_dst {0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t max_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, graph::data_type::f32);

    std::vector<graph::op_kind_t> eltwise_ops
            = {graph::op_kind::ReLU, graph::op_kind::Sigmoid};

    for (size_t i = 0; i < eltwise_ops.size(); i++) {
        graph::op_t max_op(0, graph::op_kind::Maximum, "max");
        graph::op_t eltwise_op(1, eltwise_ops[i], "eltwise");

        max_op.add_input(src0_lt);
        max_op.add_input(src1_lt);
        max_op.add_output(max_dst_lt);
        eltwise_op.add_input(max_dst_lt);
        eltwise_op.add_output(dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&max_op);
        g.add_op(&eltwise_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        ASSERT_NE(apass.get(), nullptr);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);
        std::vector<const graph::logical_tensor_t *> inputs {
                &src0_lt, &src1_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, eng);

        graph::tensor_t src0_ts(src0_lt, eng, src0.data());
        graph::tensor_t src1_ts(src1_lt, eng, src1.data());
        graph::tensor_t dst_ts(dst_lt, eng, dst.data());

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(ExecuteSubgraphFp32, BinarySwish) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> binary_src_shape {2, 2, 2, 2};
    std::vector<int64_t> binary_dst_shape {2, 2, 2, 2};

    test::vector<float> src0_data(product(binary_src_shape));
    test::vector<float> src1_data(product(binary_src_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src0_data.begin(), src0_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(src1_data.begin(), src1_data.end(),
            [&]() { return f32_distribution(generator); });

    const std::vector<graph::op_kind_t> op_infos {graph::op_kind::Add,
            graph::op_kind::Divide, graph::op_kind::Maximum,
            graph::op_kind::Minimum, graph::op_kind::Multiply,
            graph::op_kind::Subtract};

    for (auto &akind : op_infos) {
        graph::op_t binary {0, akind, "binary"};

        graph::op_t sigmoid {1, graph::op_kind::Sigmoid, "sigmoid"};
        graph::op_t multiply {2, graph::op_kind::Multiply, "multiply"};

        graph::logical_tensor_t binary_src0 = utils::logical_tensor_init(
                0, binary_src_shape, graph::data_type::f32);

        graph::logical_tensor_t binary_src1 = utils::logical_tensor_init(
                1, binary_src_shape, graph::data_type::f32);

        graph::logical_tensor_t binary_dst = utils::logical_tensor_init(
                2, binary_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t sigmoid_dst = utils::logical_tensor_init(
                3, binary_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t mul_dst = utils::logical_tensor_init(
                4, binary_dst_shape, graph::data_type::f32);

        binary.add_input(binary_src0);
        binary.add_input(binary_src1);
        binary.add_output(binary_dst);
        sigmoid.add_input(binary_dst);
        sigmoid.add_output(sigmoid_dst);
        multiply.add_input(binary_dst);
        multiply.add_input(sigmoid_dst);
        multiply.add_output(mul_dst);

        graph::graph_t g(engine->kind());
        g.add_op(&binary);
        g.add_op(&sigmoid);
        g.add_op(&multiply);
        g.finalize();

        graph::tensor_t binary_src0_ts(binary_src0, engine, src0_data.data());
        graph::tensor_t binary_src1_ts(binary_src1, engine, src1_data.data());

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(binary_dst_shape));
        graph::tensor_t mul_dst_ts(mul_dst, engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {binary_src0_ts, binary_src1_ts}, {mul_dst_ts},
                          *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &binary_src0, &binary_src1};
        std::vector<const graph::logical_tensor_t *> lt_outs {&mul_dst};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> case2_out_data(product(binary_dst_shape));
        graph::tensor_t mul_dst_ts2(mul_dst, engine, case2_out_data.data());

        cp.execute(strm, {binary_src0_ts, binary_src1_ts}, {mul_dst_ts2});
        strm->wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(Execute, Eltwise3BinaryPostops) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src = {-2.0, -1.5, 1.0, 0.5};
    test::vector<float> binary_src1 = {1.0, 2.0, 3.0, 4.0};
    test::vector<float> binary_src2 = {2.0, 3.0, 4.0, 5.0};
    test::vector<float> binary_src3 = {3.0, 4.0, 5.0, 6.0};
    test::vector<float> dst = {0.0, 0.0, 0.0, 0.0};

    graph::op_t relu(0, graph::op_kind::ReLU, "relu");
    graph::op_t div(1, graph::op_kind::Divide, "div");
    graph::op_t max(2, graph::op_kind::Maximum, "max");
    graph::op_t sub(3, graph::op_kind::Subtract, "sub");

    graph::logical_tensor_t relu_src_lt = utils::logical_tensor_init(
            0, {1, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t relu_dst_lt = utils::logical_tensor_init(
            1, {1, 1, 2, 2}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t div_src_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t div_dst_lt = utils::logical_tensor_init(
            3, {1, 1, 2, 2}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t max_src_lt = utils::logical_tensor_init(
            4, {1, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t max_dst_lt = utils::logical_tensor_init(
            5, {1, 1, 2, 2}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t sub_src_lt = utils::logical_tensor_init(
            6, {1, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t sub_dst_lt = utils::logical_tensor_init(7,
            {1, 1, 2, 2}, graph::data_type::f32, graph::layout_type::strided);

    relu.add_input(relu_src_lt);
    relu.add_output(relu_dst_lt);
    div.add_input(relu_dst_lt);
    div.add_input(div_src_lt);
    div.add_output(div_dst_lt);
    max.add_input(div_dst_lt);
    max.add_input(max_src_lt);
    max.add_output(max_dst_lt);
    sub.add_input(max_dst_lt);
    sub.add_input(sub_src_lt);
    sub.add_output(sub_dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&relu);
    g.add_op(&div);
    g.add_op(&max);
    g.add_op(&sub);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("eltwise_binary_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &relu_src_lt, &div_src_lt, &max_src_lt, &sub_src_lt};

    std::vector<const graph::logical_tensor_t *> outputs {&sub_dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::tensor_t relu_src_ts(relu_src_lt, eng, src.data());
    graph::tensor_t div_src_ts(div_src_lt, eng, binary_src1.data());
    graph::tensor_t max_src_ts(max_src_lt, eng, binary_src2.data());
    graph::tensor_t sub_src_ts(sub_src_lt, eng, binary_src3.data());
    graph::tensor_t sub_dst_ts(sub_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();

    ASSERT_EQ(
            cp.execute(strm, {relu_src_ts, div_src_ts, max_src_ts, sub_src_ts},
                    {sub_dst_ts}),
            graph::status::success);
    strm->wait();
}

TEST(ExecuteSubgraphFp32, Binary3Postops) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> binary_src_shape {2, 2, 2, 2};
    std::vector<int64_t> binary_dst_shape {2, 2, 2, 2};

    test::vector<float> src0_data(product(binary_src_shape));
    test::vector<float> src1_data(product(binary_src_shape));
    std::vector<test::vector<float>> src_datas(
            10, test::vector<float>(product(binary_src_shape)));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src0_data.begin(), src0_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(src1_data.begin(), src1_data.end(),
            [&]() { return f32_distribution(generator); });
    for (auto &data : src_datas)
        std::generate(data.begin(), data.end(),
                [&]() { return f32_distribution(generator); });

    const std::vector<graph::op_kind_t> binary_op_ts {graph::op_kind::Add,
            graph::op_kind::Divide, graph::op_kind::Maximum,
            graph::op_kind::Minimum, graph::op_kind::Multiply,
            graph::op_kind::Subtract};
    const std::vector<std::vector<graph::op_kind_t>> post_op_t_seqs {
            {graph::op_kind::Abs, graph::op_kind::Sqrt},
            {graph::op_kind::ReLU, graph::op_kind::Log,
                    graph::op_kind::Subtract},
            {graph::op_kind::Multiply, graph::op_kind::HardSwish}};

    std::vector<graph::logical_tensor_t> lt_vec;
    for (size_t i = 0; i < 9; ++i)
        lt_vec.emplace_back(utils::logical_tensor_init(
                i, binary_src_shape, graph::data_type::f32));

    for_(auto &bop_t : binary_op_ts)
    for (auto &pop_ts : post_op_t_seqs) {
        graph::op_t binary_op {0, bop_t, "binary op"};
        size_t lt_idx = 0;
        std::vector<size_t> input_lts {};
        std::vector<size_t> output_lts {};
        binary_op.add_input(lt_vec[lt_idx]);
        input_lts.push_back(lt_idx);
        binary_op.add_input(lt_vec[++lt_idx]);
        input_lts.push_back(lt_idx);
        binary_op.add_output(lt_vec[++lt_idx]);

        std::vector<graph::op_t> post_ops {};
        for (size_t i = 0; i < pop_ts.size(); ++i) {
            auto pop_t = pop_ts[i];
            post_ops.emplace_back(graph::op_t {i + 1, pop_t, "post op"});

            // set additional parameters for specific ops
            if (pop_t == graph::op_kind::Elu) {
                post_ops.back().set_attr<float>(graph::op_attr::alpha, 1.0f);
            } else if (pop_t == graph::op_kind::Clamp) {
                post_ops.back().set_attr<float>(graph::op_attr::min, 1.0f);
                post_ops.back().set_attr<float>(graph::op_attr::max, 3.0f);
            }

            post_ops.back().add_input(lt_vec[lt_idx]);
            if (std::find(binary_op_ts.begin(), binary_op_ts.end(), pop_t)
                    != binary_op_ts.end()) {
                post_ops.back().add_input(lt_vec[++lt_idx]);
                input_lts.push_back(lt_idx);
            }
            post_ops.back().add_output(lt_vec[++lt_idx]);
        }

        output_lts.push_back(lt_idx);

        graph::graph_t g(engine->kind());
        g.add_op(&binary_op);
        for (const auto &pop : post_ops)
            g.add_op(&pop);
        g.finalize();

        graph::tensor_t binary_src0_ts(lt_vec[0], engine, src_datas[0].data());
        graph::tensor_t binary_src1_ts(lt_vec[1], engine, src_datas[1].data());
        std::vector<graph::tensor_t> src_tss {};
        for (size_t i = 0; i < input_lts.size(); ++i)
            src_tss.emplace_back(graph::tensor_t(
                    lt_vec[input_lts[i]], engine, src_datas[i].data()));

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(binary_src_shape));
        graph::tensor_t case1_dst_ts(
                lt_vec[lt_idx], engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, src_tss, {case1_dst_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins(input_lts.size());
        std::transform(input_lts.begin(), input_lts.end(), lt_ins.begin(),
                [&](size_t idx) -> graph::logical_tensor_t * {
                    return &lt_vec[idx];
                });
        std::vector<const graph::logical_tensor_t *> lt_outs {&lt_vec[lt_idx]};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> case2_out_data(product(binary_dst_shape));
        graph::tensor_t case2_dst_ts(
                lt_vec[lt_idx], engine, case2_out_data.data());

        cp.execute(strm, src_tss, {case2_dst_ts});
        strm->wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(Execute, Add) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, graph::data_type::f32, graph::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    ASSERT_EQ(dst_lt.layout_type, graph::layout_type::any);
    ASSERT_EQ(compiled_dst_lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    // the second iteration
    test::vector<float> src0_2nd {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1_2nd {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> ref_dst_2nd {
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst_2nd(src0_2nd.size(), 0.0);

    graph::tensor_t src0_2nd_ts(src0_lt, eng, src0_2nd.data());
    graph::tensor_t src1_2nd_ts(src1_lt, eng, src1_2nd.data());
    graph::tensor_t dst_2nd_ts(compiled_dst_lt, eng, dst_2nd.data());
    cp.execute(strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm->wait();

    for (size_t i = 0; i < src0_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(Execute, AddWithDifferentFormat) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt = utils::logical_tensor_init(
            0, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(1,
            {1, 1, 3, 3}, {3, 3, 1, 3}, graph::data_type::f32); // abdc format
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 3, 3}, graph::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, BroadcastAdd) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {
            2.0, 2.0, 2.0}; // bianary op's src1 support broadcast
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt = utils::logical_tensor_init(
            0, {1, 2, 3, 3}, graph::data_type::f32);
    // src1 will first be unsequeeze to {1,1,1,3} and then broadcast
    // to {1,2,3,3}
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<graph::dim_t> {3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 2, 3, 3}, graph::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    // the second iteration
    test::vector<float> src0_2nd {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1_2nd {1.0, 1.0, 1.0};
    test::vector<float> ref_dst_2nd {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst_2nd(src0_2nd.size(), 0.0);

    graph::tensor_t src0_2nd_ts(src0_lt, eng, src0_2nd.data());
    graph::tensor_t src1_2nd_ts(src1_lt, eng, src1_2nd.data());
    graph::tensor_t dst_2nd_ts(compiled_dst_lt, eng, dst_2nd.data());
    cp.execute(strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm->wait();

    for (size_t i = 0; i < src0_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(Execute, SwapBroadcastAdd) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src1.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt = utils::logical_tensor_init(
            0, {1, 1, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {1, 2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 2, 2, 2}, graph::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src1.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    // inplace
    auto inplace_pair = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pair[0].input_id, src1_lt.id);
    ASSERT_EQ(inplace_pair[0].output_id, dst_lt.id);

    graph::tensor_t dst_ts2(compiled_dst_lt, eng, src1.data());
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts2});
    strm->wait();

    for (size_t i = 0; i < src1.size(); ++i) {
        ASSERT_FLOAT_EQ(src1[i], ref_dst[i]);
    }
}

TEST(Execute, MultidirectionalBroadcastAddBA) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<graph::dim_t> {3, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {3, 3}, graph::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    // the second iteration
    test::vector<float> src0_2nd {1.0, 1.0, 1.0};
    test::vector<float> src1_2nd {1.0, 1.0, 1.0};
    test::vector<float> ref_dst_2nd {
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst_2nd(ref_dst_2nd.size(), 0.0);

    graph::tensor_t src0_2nd_ts(src0_lt, eng, src0_2nd.data());
    graph::tensor_t src1_2nd_ts(src1_lt, eng, src1_2nd.data());
    graph::tensor_t dst_2nd_ts(compiled_dst_lt, eng, dst_2nd.data());
    cp.execute(strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm->wait();

    for (size_t i = 0; i < ref_dst_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(Execute, multidirectionalbBroadcastAddAB) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {3, 1}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<graph::dim_t> {3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {3, 3}, graph::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MultidirectionalBroadcastAdd) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0(8, 1.0);
    test::vector<float> src1(3, 2.0);
    test::vector<float> ref_dst(24, 3.0);
    test::vector<float> dst(ref_dst.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {2, 1, 4}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<graph::dim_t> {1, 3, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {2, 3, 4}, graph::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MultidirectionalBroadcastAddExpandDim) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0(2, 1.0);
    test::vector<float> src1(12, 2.0);
    test::vector<float> ref_dst(24, 3.0);
    test::vector<float> dst(ref_dst.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {2, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<graph::dim_t> {3, 4}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {2, 3, 4}, graph::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Compile, AddShapeMismatchCase0) {
    graph::engine_t *eng = get_engine();

    graph::op_t add_op(0, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {8, 4, 256}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {2, 2, 512}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {8, 4, 256}, graph::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    auto ret = p.compile(&cp, inputs, outputs, eng);

    // compile the add operator
    ASSERT_EQ(ret, graph::status::invalid_shape);
}

TEST(Compile, AddShapeMismatch1) {
    graph::engine_t *eng = get_engine();

    graph::op_t add_op(graph::op_kind::Add);

    graph::logical_tensor_t src0_lt = utils::logical_tensor_init(
            0, {8, 15, 5, 7}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 15, 5, 7}, graph::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    auto ret = p.compile(&cp, inputs, outputs, eng);

    // compile the add operator
    ASSERT_EQ(ret, graph::status::success);
}

TEST(Compile, AddShapeMismatch2) {
    graph::engine_t *eng = get_engine();

    graph::op_t add_op(graph::op_kind::Add);

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {8, 15, 5, 7}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 15, 5, 7}, graph::data_type::f32);

    // compile the add operator
    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    auto ret = p.compile(&cp, inputs, outputs, eng);

    ASSERT_EQ(ret, graph::status::success);
}

TEST(Execute, ReversedDifferentFormatBroadcastAdd) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src1 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src0 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0,
            2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src1.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");

    // we reverse the order of src0 and src1
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            0, {1, 2, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src0_lt = utils::logical_tensor_init(
            1, {3, 3}, {1, 3}, graph::data_type::f32); // ba format
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 2, 3, 3}, graph::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);
    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, BiasAdd) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> bias {1.0, 2.0};
    test::vector<float> ref_dst1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> ref_dst2 {2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0,
            3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0};
    test::vector<float> dst(src.size(), 0.0);

    std::vector<std::vector<graph::dim_t>> src_shapes {
            {1, 2, 3, 3}, {1, 3, 3, 2}};
    std::vector<std::vector<graph::dim_t>> dst_shapes {
            {1, 2, 3, 3}, {1, 3, 3, 2}};
    std::vector<graph::dim_t> bias_shape {2};
    std::vector<std::string> data_formats {"NCX", "NXC"};

    for (size_t i = 0; i < data_formats.size(); i++) {
        graph::op_t bias_add_op(graph::op_kind::BiasAdd);
        bias_add_op.set_attr<std::string>(
                graph::op_attr::data_format, data_formats[i]);

        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, src_shapes[i], graph::data_type::f32);
        graph::logical_tensor_t bias_lt = utils::logical_tensor_init(
                1, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, dst_shapes[i], graph::data_type::f32);

        bias_add_op.add_input(src_lt);
        bias_add_op.add_input(bias_lt);
        bias_add_op.add_output(dst_lt);

        graph::graph_t g(eng->kind());
        ASSERT_EQ(g.add_op(&bias_add_op), graph::status::success);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("bias_add_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);
        std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &bias_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, eng);

        graph::tensor_t src_ts(src_lt, eng, src.data());
        graph::tensor_t bias_ts(bias_lt, eng, bias.data());
        graph::tensor_t dst_ts(dst_lt, eng, dst.data());

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src_ts, bias_ts}, {dst_ts});
        strm->wait();

        if (i == 0) {
            for (size_t i = 0; i < src.size(); ++i) {
                ASSERT_FLOAT_EQ(dst[i], ref_dst1[i]);
            }
        } else {
            for (size_t i = 0; i < src.size(); ++i) {
                ASSERT_FLOAT_EQ(dst[i], ref_dst2[i]);
            }
        }
    }
}

TEST(Execute, AddMul) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");
    graph::op_t mul_op(1, graph::op_kind::Multiply, "mul");

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {1, 3, 3}, graph::data_type::f32, graph::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    mul_op.add_input(add_dst_lt);
    mul_op.add_input(post_src_lt);
    mul_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.add_op(&mul_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddMulPostSrcAsNxc) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {
            3.0, 12.0, 21.0, 6.0, 15.0, 24.0, 9.0, 18.0, 27.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");
    graph::op_t mul_op(1, graph::op_kind::Multiply, "mul");

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t post_src_lt = utils::logical_tensor_init(
            3, {1, 3, 3}, {9, 1, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {1, 3, 3}, graph::data_type::f32, graph::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    mul_op.add_input(add_dst_lt);
    mul_op.add_input(post_src_lt);
    mul_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.add_op(&mul_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);
    std::vector<const graph::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddRelu) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {-2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");
    graph::op_t relu_op(1, graph::op_kind::ReLU, "relu");

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 3, 3}, graph::data_type::f32, graph::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    relu_op.add_input(add_dst_lt);
    relu_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.add_op(&relu_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    ASSERT_EQ(dst_lt.layout_type, graph::layout_type::any);
    ASSERT_EQ(compiled_dst_lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddSigmoid) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {
            -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0};
    test::vector<float> ref_dst {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst(src0.size(), 0.0);

    graph::op_t add_op(0, graph::op_kind::Add, "add");
    graph::op_t sigmoid_op(1, graph::op_kind::Sigmoid, "sigmoid");

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 3, 3}, graph::data_type::f32, graph::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    sigmoid_op.add_input(add_dst_lt);
    sigmoid_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.add_op(&sigmoid_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);
    std::vector<const graph::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    ASSERT_EQ(dst_lt.layout_type, graph::layout_type::any);
    ASSERT_EQ(compiled_dst_lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddAdd) {
    graph::engine_t *eng = get_engine();

    std::vector<int64_t> src_shape = {8, 128, 768};
    test::vector<float> src0(product(src_shape));
    test::vector<float> src1(product(src_shape));
    test::vector<float> post_src(product(src_shape));
    test::vector<float> dst(src0.size(), 0.0);

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(-1.0f, 1.0f);
    std::generate(src0.begin(), src0.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(src1.begin(), src1.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(post_src.begin(), post_src.end(),
            [&]() { return f32_distribution(generator); });

    graph::op_t add0_op(0, graph::op_kind::Add, "add0");
    graph::op_t add1_op(1, graph::op_kind::Add, "add1");

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(2, src_shape, graph::data_type::f32);
    graph::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, src_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, src_shape, graph::data_type::f32);

    add0_op.add_input(src0_lt);
    add0_op.add_input(src1_lt);
    add0_op.add_output(add_dst_lt);
    add1_op.add_input(add_dst_lt);
    add1_op.add_input(post_src_lt);
    add1_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&add0_op);
    g.add_op(&add1_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    // inplace
    auto inplace_pair = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pair[0].input_id, post_src_lt.id);
    ASSERT_EQ(inplace_pair[0].output_id, dst_lt.id);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    graph::tensor_t dst_inplace_ts(dst_lt, eng, post_src.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_inplace_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(post_src[i], dst[i]);
    }
}

TEST(Execute, ScalarScalarAdd) {
    graph::op_t add_op(graph::op_kind::Add);
    graph::engine_t *eng = get_engine();

    test::vector<float> src0_data {10.f};
    test::vector<float> src1_data {2.f};
    test::vector<float> ref_dst_data {12.f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src0
            = utils::logical_tensor_init(0, {}, graph::data_type::f32);
    graph::logical_tensor_t src1
            = utils::logical_tensor_init(1, {}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            2, graph::data_type::f32, graph::layout_type::any);

    add_op.add_input(src0);
    add_op.add_input(src1);
    add_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0, &src1};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    // output should be a scalar (ndims=0, layout_type=strided)
    graph::logical_tensor_t scalar_lt;
    cp.query_logical_tensor(dst.id, &scalar_lt);
    ASSERT_EQ(scalar_lt.layout_type, graph::layout_type::strided);
    ASSERT_EQ(scalar_lt.ndims, 0);

    graph::tensor_t src0_ts(src0, eng, src0_data.data());
    graph::tensor_t src1_ts(src1, eng, src1_data.data());
    graph::tensor_t dst_ts(scalar_lt, eng, dst_data.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src0_ts, src1_ts}, {dst_ts}),
            graph::status::success);
    strm->wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, ScalarVectorAdd) {
    graph::op_t add_op(graph::op_kind::Add);
    graph::engine_t *eng = get_engine();

    test::vector<float> src0_data {10.f};
    test::vector<float> src1_data {2.f};
    test::vector<float> ref_dst_data {12.f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src0
            = utils::logical_tensor_init(0, {}, graph::data_type::f32);
    graph::logical_tensor_t src1
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            2, graph::data_type::f32, graph::layout_type::any);

    add_op.add_input(src0);
    add_op.add_input(src1);
    add_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0, &src1};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    // output should be a scalar (ndims=0, layout_type=strided)
    graph::logical_tensor_t scalar_lt;
    cp.query_logical_tensor(dst.id, &scalar_lt);
    ASSERT_EQ(scalar_lt.layout_type, graph::layout_type::strided);
    ASSERT_EQ(scalar_lt.ndims, 1);

    graph::tensor_t src0_ts(src0, eng, src0_data.data());
    graph::tensor_t src1_ts(src1, eng, src1_data.data());
    graph::tensor_t dst_ts(scalar_lt, eng, dst_data.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src0_ts, src1_ts}, {dst_ts}),
            graph::status::success);
    strm->wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, MulAddPerTensorBroadcast) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0};
    test::vector<float> post_src {2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::op_t mul_op(0, graph::op_kind::Multiply, "mul");
    graph::op_t add_op(1, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt = utils::logical_tensor_init(
            0, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t mul_dst_lt = utils::logical_tensor_init(
            2, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {1, 1, 3, 3}, graph::data_type::f32);

    mul_op.add_input(src0_lt);
    mul_op.add_input(src1_lt);
    mul_op.add_output(mul_dst_lt);
    add_op.add_input(mul_dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&mul_op);
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MulAddPerHwBroadcast) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0(18, 2.0);
    test::vector<float> src1(1, 2.0);
    test::vector<float> post_src(6, 2.0);
    test::vector<float> ref_dst(18, 6.0);
    test::vector<float> dst(src0.size(), 0.0);

    graph::op_t mul_op(0, graph::op_kind::Multiply, "mul");
    graph::op_t add_op(1, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt = utils::logical_tensor_init(
            0, {1, 3, 2, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t mul_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 2, 3}, graph::data_type::f32);
    graph::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {2, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {1, 3, 2, 3}, graph::data_type::f32);

    mul_op.add_input(src0_lt);
    mul_op.add_input(src1_lt);
    mul_op.add_output(mul_dst_lt);
    add_op.add_input(mul_dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&mul_op);
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MulAddPerChannelBroadcast) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    graph::op_t mul_op(0, graph::op_kind::Multiply, "mul");
    graph::op_t add_op(1, graph::op_kind::Add, "add");

    graph::logical_tensor_t src0_lt = utils::logical_tensor_init(
            0, {1, 3, 1, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t mul_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 1, 3}, graph::data_type::f32);
    graph::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {3, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {1, 3, 1, 3}, graph::data_type::f32);

    mul_op.add_input(src0_lt);
    mul_op.add_input(src1_lt);
    mul_op.add_output(mul_dst_lt);
    add_op.add_input(mul_dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&mul_op);
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    graph::tensor_t src0_ts(src0_lt, eng, src0.data());
    graph::tensor_t src1_ts(src1_lt, eng, src1.data());
    graph::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MulAddAdd) {
    graph::engine_t *eng = get_engine();

    test::vector<float> mul_src1(1917, 2.5f);
    test::vector<float> mul_src2(1, 1.5f);
    test::vector<float> add1_src(1917, 1.15f);
    test::vector<float> add2_src(1917, 1.07f);
    test::vector<float> ref_dst(1917, 5.97f);
    test::vector<float> dst(1917, 0.f);

    graph::op_t mul_op(0, graph::op_kind::Multiply, "mul");
    graph::op_t add1_op(1, graph::op_kind::Add, "add");
    graph::op_t add2_op(2, graph::op_kind::Add, "add");
    add1_op.set_attr<std::string>(graph::op_attr::auto_broadcast, "none");
    add2_op.set_attr<std::string>(graph::op_attr::auto_broadcast, "none");

    graph::logical_tensor_t mul_src1_lt
            = utils::logical_tensor_init(0, {1917}, graph::data_type::f32);
    graph::logical_tensor_t mul_src2_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);
    graph::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(2, {1917}, graph::data_type::f32);
    graph::logical_tensor_t add1_src_lt
            = utils::logical_tensor_init(3, {1917}, graph::data_type::f32);
    graph::logical_tensor_t add1_dst_lt
            = utils::logical_tensor_init(4, {1917}, graph::data_type::f32);
    graph::logical_tensor_t add2_src_lt
            = utils::logical_tensor_init(5, {1917}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(6, {1917}, graph::data_type::f32);

    mul_op.add_input(mul_src1_lt);
    mul_op.add_input(mul_src2_lt);
    mul_op.add_output(mul_dst_lt);

    add1_op.add_input(mul_dst_lt);
    add1_op.add_input(add1_src_lt);
    add1_op.add_output(add1_dst_lt);

    add2_op.add_input(add1_dst_lt);
    add2_op.add_input(add2_src_lt);
    add2_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&mul_op);
    g.add_op(&add1_op);
    g.add_op(&add2_op);
    g.finalize();

    run_all_passes(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &mul_src1_lt, &mul_src2_lt, &add1_src_lt, &add2_src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    graph::tensor_t mul_src1_ts(mul_src1_lt, eng, mul_src1.data());
    graph::tensor_t mul_src2_ts(mul_src2_lt, eng, mul_src2.data());
    graph::tensor_t add1_src_ts(add1_src_lt, eng, add1_src.data());
    graph::tensor_t add2_src_ts(add2_src_lt, eng, add2_src.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {mul_src1_ts, mul_src2_ts, add1_src_ts, add2_src_ts},
            {dst_ts});
    strm->wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddEmptyInput) {
    graph::op_t add_op(graph::op_kind::Add);
    graph::engine_t *eng = get_engine();

    // prepare logical tensor
    graph::logical_tensor_t src0
            = utils::logical_tensor_init(0, {2, 3, 0}, graph::data_type::f32);
    graph::logical_tensor_t src1
            = utils::logical_tensor_init(1, {2, 3, 0}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            3, graph::data_type::f32, graph::layout_type::any);

    add_op.add_input(src0);
    add_op.add_input(src1);
    add_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src0, &src1};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t empty_lt;
    cp.query_logical_tensor(dst.id, &empty_lt);
    ASSERT_EQ(empty_lt.layout_type, graph::layout_type::strided);
    ASSERT_EQ(empty_lt.ndims, 3);
    ASSERT_EQ(empty_lt.dims[0], 2);
    ASSERT_EQ(empty_lt.dims[1], 3);
    ASSERT_EQ(empty_lt.dims[2], 0);

    graph::tensor_t src0_ts(src0, eng, nullptr);
    graph::tensor_t src1_ts(src1, eng, nullptr);
    graph::tensor_t dst_ts(dst, eng, nullptr);

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src0_ts, src1_ts}, {dst_ts}),
            graph::status::success);
    strm->wait();
}
