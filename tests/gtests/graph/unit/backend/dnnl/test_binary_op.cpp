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

TEST(Execute, BinaryOp) {
    impl::engine_t *eng = get_engine();

    std::vector<impl::op_kind_t> op_kinds
            = {impl::op_kind::Multiply, impl::op_kind::Minimum,
                    impl::op_kind::Maximum, impl::op_kind::Divide,
                    impl::op_kind::Subtract, impl::op_kind::SquaredDifference};
    std::vector<std::string> pass_names = {"mul_pass", "min_pass", "max_pass",
            "div_pass", "sub_pass", "squareddifference_pass"};

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    auto src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    auto src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    auto dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    for (size_t i = 0; i < op_kinds.size(); i++) {
        impl::op_t binary_op(op_kinds[i]);

        binary_op.add_input(src0_lt);
        binary_op.add_input(src1_lt);
        binary_op.add_output(dst_lt);

        impl::graph_t g(eng->kind());
        g.add_op(&binary_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass(pass_names[i]);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

        impl::tensor_t src0_ts(src0_lt, eng, src0.data());
        impl::tensor_t src1_ts(src1_lt, eng, src1.data());
        impl::tensor_t dst_ts(dst_lt, eng, dst.data());

        impl::stream_t *strm = get_stream();
        cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(Execute, MulEltwise) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);

    std::vector<impl::op_kind_t> eltwise_ops
            = {impl::op_kind::ReLU, impl::op_kind::Sigmoid};

    for (size_t i = 0; i < eltwise_ops.size(); i++) {
        impl::op_t mul_op(0, impl::op_kind::Multiply, "mul");
        impl::op_t eltwise_op(1, eltwise_ops[i], "eltwise");

        mul_op.add_input(src0_lt);
        mul_op.add_input(src1_lt);
        mul_op.add_output(mul_dst_lt);
        eltwise_op.add_input(mul_dst_lt);
        eltwise_op.add_output(dst_lt);

        impl::graph_t g(eng->kind());
        g.add_op(&mul_op);
        g.add_op(&eltwise_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        ASSERT_NE(apass.get(), nullptr);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

        impl::tensor_t src0_ts(src0_lt, eng, src0.data());
        impl::tensor_t src1_ts(src1_lt, eng, src1.data());
        impl::tensor_t dst_ts(dst_lt, eng, dst.data());

        impl::stream_t *strm = get_stream();
        cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(Execute, BinaryOpAddFusion) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t tmp_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 3, 3}, impl::data_type::f32);

    std::vector<impl::op_kind_t> bin_ops = {impl::op_kind::Multiply,
            impl::op_kind::Maximum, impl::op_kind::Minimum};

    for (size_t i = 0; i < bin_ops.size(); i++) {
        impl::op_t bin_op(0, bin_ops[i], "bin");
        impl::op_t add_op(1, impl::op_kind::Add, "add");

        bin_op.add_input(src0_lt);
        bin_op.add_input(src1_lt);
        bin_op.add_output(tmp_dst_lt);
        add_op.add_input(tmp_dst_lt);
        add_op.add_input(post_src_lt);
        add_op.add_output(dst_lt);

        impl::graph_t g(eng->kind());
        g.add_op(&bin_op);
        g.add_op(&add_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);
        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src0_lt, &src1_lt, &post_src_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, eng);

        impl::tensor_t src0_ts(src0_lt, eng, src0.data());
        impl::tensor_t src1_ts(src1_lt, eng, src1.data());
        impl::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
        impl::tensor_t dst_ts(dst_lt, eng, dst.data());

        impl::stream_t *strm = get_stream();
        cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(Execute, BinarySub) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0};
    test::vector<float> src1 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> dst(src1.size(), 0.0);
    test::vector<float> ref {1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0};

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {2, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {2, 1, 2, 2}, impl::data_type::f32);

    impl::op_t bin_op(0, impl::op_kind::Subtract, "bin");

    bin_op.add_input(src0_lt);
    bin_op.add_input(src1_lt);
    bin_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&bin_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sub_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();
    for (size_t i = 0; i < ref.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref[i]);
    }
}

TEST(Execute, MinEltwise) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {-1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t min_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);

    std::vector<impl::op_kind_t> eltwise_ops
            = {impl::op_kind::ReLU, impl::op_kind::Sigmoid};

    for (size_t i = 0; i < eltwise_ops.size(); i++) {
        impl::op_t min_op(0, impl::op_kind::Minimum, "min");
        impl::op_t eltwise_op(1, eltwise_ops[i], "eltwise");

        min_op.add_input(src0_lt);
        min_op.add_input(src1_lt);
        min_op.add_output(min_dst_lt);
        eltwise_op.add_input(min_dst_lt);
        eltwise_op.add_output(dst_lt);

        impl::graph_t g(eng->kind());
        g.add_op(&min_op);
        g.add_op(&eltwise_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        ASSERT_NE(apass.get(), nullptr);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, eng);

        impl::tensor_t src0_ts(src0_lt, eng, src0.data());
        impl::tensor_t src1_ts(src1_lt, eng, src1.data());
        impl::tensor_t dst_ts(dst_lt, eng, dst.data());

        impl::stream_t *strm = get_stream();
        cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(Execute, MaxEltwise) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {-2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {-1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> ref_dst {0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t max_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);

    std::vector<impl::op_kind_t> eltwise_ops
            = {impl::op_kind::ReLU, impl::op_kind::Sigmoid};

    for (size_t i = 0; i < eltwise_ops.size(); i++) {
        impl::op_t max_op(0, impl::op_kind::Maximum, "max");
        impl::op_t eltwise_op(1, eltwise_ops[i], "eltwise");

        max_op.add_input(src0_lt);
        max_op.add_input(src1_lt);
        max_op.add_output(max_dst_lt);
        eltwise_op.add_input(max_dst_lt);
        eltwise_op.add_output(dst_lt);

        impl::graph_t g(eng->kind());
        g.add_op(&max_op);
        g.add_op(&eltwise_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        ASSERT_NE(apass.get(), nullptr);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, eng);

        impl::tensor_t src0_ts(src0_lt, eng, src0.data());
        impl::tensor_t src1_ts(src1_lt, eng, src1.data());
        impl::tensor_t dst_ts(dst_lt, eng, dst.data());

        impl::stream_t *strm = get_stream();
        cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
        strm->wait();
    }
}

TEST(ExecuteSubgraphFp32, BinarySwish) {
    impl::engine_t *engine = get_engine();
    impl::stream_t *strm = get_stream();

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

    const std::vector<impl::op_kind_t> op_infos {impl::op_kind::Add,
            impl::op_kind::Divide, impl::op_kind::Maximum,
            impl::op_kind::Minimum, impl::op_kind::Multiply,
            impl::op_kind::Subtract};

    for (auto &akind : op_infos) {
        impl::op_t binary {0, akind, "binary"};

        impl::op_t sigmoid {1, impl::op_kind::Sigmoid, "sigmoid"};
        impl::op_t multiply {2, impl::op_kind::Multiply, "multiply"};

        impl::logical_tensor_t binary_src0 = utils::logical_tensor_init(
                0, binary_src_shape, impl::data_type::f32);

        impl::logical_tensor_t binary_src1 = utils::logical_tensor_init(
                1, binary_src_shape, impl::data_type::f32);

        impl::logical_tensor_t binary_dst = utils::logical_tensor_init(
                2, binary_dst_shape, impl::data_type::f32);
        impl::logical_tensor_t sigmoid_dst = utils::logical_tensor_init(
                3, binary_dst_shape, impl::data_type::f32);
        impl::logical_tensor_t mul_dst = utils::logical_tensor_init(
                4, binary_dst_shape, impl::data_type::f32);

        binary.add_input(binary_src0);
        binary.add_input(binary_src1);
        binary.add_output(binary_dst);
        sigmoid.add_input(binary_dst);
        sigmoid.add_output(sigmoid_dst);
        multiply.add_input(binary_dst);
        multiply.add_input(sigmoid_dst);
        multiply.add_output(mul_dst);

        impl::graph_t g(engine->kind());
        g.add_op(&binary);
        g.add_op(&sigmoid);
        g.add_op(&multiply);
        g.build_graph();

        impl::tensor_t binary_src0_ts(binary_src0, engine, src0_data.data());
        impl::tensor_t binary_src1_ts(binary_src1, engine, src1_data.data());

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(binary_dst_shape));
        impl::tensor_t mul_dst_ts(mul_dst, engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {binary_src0_ts, binary_src1_ts}, {mul_dst_ts},
                          *engine, *strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &binary_src0, &binary_src1};
        std::vector<const impl::logical_tensor_t *> lt_outs {&mul_dst};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> case2_out_data(product(binary_dst_shape));
        impl::tensor_t mul_dst_ts2(mul_dst, engine, case2_out_data.data());

        cp.execute(strm, {binary_src0_ts, binary_src1_ts}, {mul_dst_ts2});
        strm->wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(Execute, Eltwise3BinaryPostops) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src = {-2.0, -1.5, 1.0, 0.5};
    test::vector<float> binary_src1 = {1.0, 2.0, 3.0, 4.0};
    test::vector<float> binary_src2 = {2.0, 3.0, 4.0, 5.0};
    test::vector<float> binary_src3 = {3.0, 4.0, 5.0, 6.0};
    test::vector<float> dst = {0.0, 0.0, 0.0, 0.0};

    impl::op_t relu(0, impl::op_kind::ReLU, "relu");
    impl::op_t div(1, impl::op_kind::Divide, "div");
    impl::op_t max(2, impl::op_kind::Maximum, "max");
    impl::op_t sub(3, impl::op_kind::Subtract, "sub");

    impl::logical_tensor_t relu_src_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t relu_dst_lt = utils::logical_tensor_init(
            1, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t div_src_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t div_dst_lt = utils::logical_tensor_init(
            3, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t max_src_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t max_dst_lt = utils::logical_tensor_init(
            5, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t sub_src_lt
            = utils::logical_tensor_init(6, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t sub_dst_lt = utils::logical_tensor_init(
            7, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::strided);

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

    impl::graph_t g(eng->kind());
    g.add_op(&relu);
    g.add_op(&div);
    g.add_op(&max);
    g.add_op(&sub);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("eltwise_binary_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &relu_src_lt, &div_src_lt, &max_src_lt, &sub_src_lt};

    std::vector<const impl::logical_tensor_t *> outputs {&sub_dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    impl::tensor_t relu_src_ts(relu_src_lt, eng, src.data());
    impl::tensor_t div_src_ts(div_src_lt, eng, binary_src1.data());
    impl::tensor_t max_src_ts(max_src_lt, eng, binary_src2.data());
    impl::tensor_t sub_src_ts(sub_src_lt, eng, binary_src3.data());
    impl::tensor_t sub_dst_ts(sub_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();

    ASSERT_EQ(
            cp.execute(strm, {relu_src_ts, div_src_ts, max_src_ts, sub_src_ts},
                    {sub_dst_ts}),
            impl::status::success);
    strm->wait();
}

TEST(ExecuteSubgraphFp32, Binary3Postops) {
    impl::engine_t *engine = get_engine();
    impl::stream_t *strm = get_stream();

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

    const std::vector<impl::op_kind_t> binary_op_ts {impl::op_kind::Add,
            impl::op_kind::Divide, impl::op_kind::Maximum,
            impl::op_kind::Minimum, impl::op_kind::Multiply,
            impl::op_kind::Subtract};
    const std::vector<std::vector<impl::op_kind_t>> post_op_t_seqs {
            {impl::op_kind::Abs, impl::op_kind::Sqrt},
            {impl::op_kind::ReLU, impl::op_kind::Log, impl::op_kind::Subtract},
            {impl::op_kind::Multiply, impl::op_kind::HardSwish}};

    std::vector<impl::logical_tensor_t> lt_vec;
    for (size_t i = 0; i < 9; ++i)
        lt_vec.emplace_back(utils::logical_tensor_init(
                i, binary_src_shape, impl::data_type::f32));

    for_(auto &bop_t : binary_op_ts)
    for (auto &pop_ts : post_op_t_seqs) {
        impl::op_t binary_op {0, bop_t, "binary op"};
        size_t lt_idx = 0;
        std::vector<size_t> input_lts {};
        std::vector<size_t> output_lts {};
        binary_op.add_input(lt_vec[lt_idx]);
        input_lts.push_back(lt_idx);
        binary_op.add_input(lt_vec[++lt_idx]);
        input_lts.push_back(lt_idx);
        binary_op.add_output(lt_vec[++lt_idx]);

        std::vector<impl::op_t> post_ops {};
        for (size_t i = 0; i < pop_ts.size(); ++i) {
            auto pop_t = pop_ts[i];
            post_ops.emplace_back(impl::op_t {i + 1, pop_t, "post op"});

            // set additional parameters for specific ops
            if (pop_t == impl::op_kind::Elu) {
                post_ops.back().set_attr<float>(impl::op_attr::alpha, 1.0f);
            } else if (pop_t == impl::op_kind::Clamp) {
                post_ops.back().set_attr<float>(impl::op_attr::min, 1.0f);
                post_ops.back().set_attr<float>(impl::op_attr::max, 3.0f);
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

        impl::graph_t g(engine->kind());
        g.add_op(&binary_op);
        for (const auto &pop : post_ops)
            g.add_op(&pop);
        g.build_graph();

        impl::tensor_t binary_src0_ts(lt_vec[0], engine, src_datas[0].data());
        impl::tensor_t binary_src1_ts(lt_vec[1], engine, src_datas[1].data());
        std::vector<impl::tensor_t> src_tss {};
        for (size_t i = 0; i < input_lts.size(); ++i)
            src_tss.emplace_back(impl::tensor_t(
                    lt_vec[input_lts[i]], engine, src_datas[i].data()));

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(binary_src_shape));
        impl::tensor_t case1_dst_ts(
                lt_vec[lt_idx], engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, src_tss, {case1_dst_ts}, *engine, *strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins(input_lts.size());
        std::transform(input_lts.begin(), input_lts.end(), lt_ins.begin(),
                [&](size_t idx) -> impl::logical_tensor_t * {
                    return &lt_vec[idx];
                });
        std::vector<const impl::logical_tensor_t *> lt_outs {&lt_vec[lt_idx]};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> case2_out_data(product(binary_dst_shape));
        impl::tensor_t case2_dst_ts(
                lt_vec[lt_idx], engine, case2_out_data.data());

        cp.execute(strm, src_tss, {case2_dst_ts});
        strm->wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(Execute, Add) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::any);
    ASSERT_EQ(compiled_dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
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

    impl::tensor_t src0_2nd_ts(src0_lt, eng, src0_2nd.data());
    impl::tensor_t src1_2nd_ts(src1_lt, eng, src1_2nd.data());
    impl::tensor_t dst_2nd_ts(compiled_dst_lt, eng, dst_2nd.data());
    cp.execute(strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm->wait();

    for (size_t i = 0; i < src0_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(Execute, AddWithDifferentFormat) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, {3, 3, 1, 3}, impl::data_type::f32); // abdc format
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 3, 3}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, BroadcastAdd) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {
            2.0, 2.0, 2.0}; // bianary op's src1 support broadcast
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 2, 3, 3}, impl::data_type::f32);
    // src1 will first be unsequeeze to {1,1,1,3} and then broadcast
    // to {1,2,3,3}
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 3, 3}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
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

    impl::tensor_t src0_2nd_ts(src0_lt, eng, src0_2nd.data());
    impl::tensor_t src1_2nd_ts(src1_lt, eng, src1_2nd.data());
    impl::tensor_t dst_2nd_ts(compiled_dst_lt, eng, dst_2nd.data());
    cp.execute(strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm->wait();

    for (size_t i = 0; i < src0_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(Execute, SwapBroadcastAdd) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src1.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 2, 2}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src1.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    // inplace
    auto inplace_pair = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pair[0].input_id, src1_lt.id);
    ASSERT_EQ(inplace_pair[0].output_id, dst_lt.id);

    impl::tensor_t dst_ts2(compiled_dst_lt, eng, src1.data());
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts2});
    strm->wait();

    for (size_t i = 0; i < src1.size(); ++i) {
        ASSERT_FLOAT_EQ(src1[i], ref_dst[i]);
    }
}

TEST(Execute, MultidirectionalBroadcastAddBA) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {3, 3}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
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

    impl::tensor_t src0_2nd_ts(src0_lt, eng, src0_2nd.data());
    impl::tensor_t src1_2nd_ts(src1_lt, eng, src1_2nd.data());
    impl::tensor_t dst_2nd_ts(compiled_dst_lt, eng, dst_2nd.data());
    cp.execute(strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm->wait();

    for (size_t i = 0; i < ref_dst_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(Execute, multidirectionalbBroadcastAddAB) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {3, 3}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MultidirectionalBroadcastAdd) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0(8, 1.0);
    test::vector<float> src1(3, 2.0);
    test::vector<float> ref_dst(24, 3.0);
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {2, 1, 4}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {1, 3, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {2, 3, 4}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MultidirectionalBroadcastAddExpandDim) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0(2, 1.0);
    test::vector<float> src1(12, 2.0);
    test::vector<float> ref_dst(24, 3.0);
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {2, 3, 4}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Compile, AddShapeMismatchCase0) {
    impl::engine_t *eng = get_engine();

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {8, 4, 256}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {2, 2, 512}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {8, 4, 256}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    auto ret = p.compile(&cp, inputs, outputs, eng);

    // compile the add operator
    ASSERT_EQ(ret, impl::status::invalid_shape);
}

TEST(Compile, AddShapeMismatch1) {
    impl::engine_t *eng = get_engine();

    impl::op_t add_op(impl::op_kind::Add);

    impl::logical_tensor_t src0_lt = utils::logical_tensor_init(
            0, {8, 15, 5, 7}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 15, 5, 7}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    auto ret = p.compile(&cp, inputs, outputs, eng);

    // compile the add operator
    ASSERT_EQ(ret, impl::status::success);
}

TEST(Compile, AddShapeMismatch2) {
    impl::engine_t *eng = get_engine();

    impl::op_t add_op(impl::op_kind::Add);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {8, 15, 5, 7}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 15, 5, 7}, impl::data_type::f32);

    // compile the add operator
    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    auto ret = p.compile(&cp, inputs, outputs, eng);

    ASSERT_EQ(ret, impl::status::success);
}

TEST(Execute, ReversedDifferentFormatBroadcastAdd) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src1 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src0 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0,
            2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src1.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    // we reverse the order of src0 and src1
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(0, {1, 2, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src0_lt = utils::logical_tensor_init(
            1, {3, 3}, {1, 3}, impl::data_type::f32); // ba format
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 3, 3}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, BiasAdd) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> bias {1.0, 2.0};
    test::vector<float> ref_dst1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> ref_dst2 {2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0,
            3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0};
    test::vector<float> dst(src.size(), 0.0);

    std::vector<std::vector<impl::dim_t>> src_shapes {
            {1, 2, 3, 3}, {1, 3, 3, 2}};
    std::vector<std::vector<impl::dim_t>> dst_shapes {
            {1, 2, 3, 3}, {1, 3, 3, 2}};
    std::vector<impl::dim_t> bias_shape {2};
    std::vector<std::string> data_formats {"NCX", "NXC"};

    for (size_t i = 0; i < data_formats.size(); i++) {
        impl::op_t bias_add_op(impl::op_kind::BiasAdd);
        bias_add_op.set_attr<std::string>(
                impl::op_attr::data_format, data_formats[i]);

        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, src_shapes[i], impl::data_type::f32);
        impl::logical_tensor_t bias_lt = utils::logical_tensor_init(
                1, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, dst_shapes[i], impl::data_type::f32);

        bias_add_op.add_input(src_lt);
        bias_add_op.add_input(bias_lt);
        bias_add_op.add_output(dst_lt);

        impl::graph_t g(eng->kind());
        ASSERT_EQ(g.add_op(&bias_add_op), impl::status::success);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("bias_add_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &bias_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, eng);

        impl::tensor_t src_ts(src_lt, eng, src.data());
        impl::tensor_t bias_ts(bias_lt, eng, bias.data());
        impl::tensor_t dst_ts(dst_lt, eng, dst.data());

        impl::stream_t *strm = get_stream();
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
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");
    impl::op_t mul_op(1, impl::op_kind::Multiply, "mul");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    mul_op.add_input(add_dst_lt);
    mul_op.add_input(post_src_lt);
    mul_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.add_op(&mul_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddMulPostSrcAsNxc) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {
            3.0, 12.0, 21.0, 6.0, 15.0, 24.0, 9.0, 18.0, 27.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");
    impl::op_t mul_op(1, impl::op_kind::Multiply, "mul");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t post_src_lt = utils::logical_tensor_init(
            3, {1, 3, 3}, {9, 1, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    mul_op.add_input(add_dst_lt);
    mul_op.add_input(post_src_lt);
    mul_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.add_op(&mul_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddRelu) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {-2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");
    impl::op_t relu_op(1, impl::op_kind::ReLU, "relu");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    relu_op.add_input(add_dst_lt);
    relu_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.add_op(&relu_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::any);
    ASSERT_EQ(compiled_dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddSigmoid) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {
            -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0};
    test::vector<float> ref_dst {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");
    impl::op_t sigmoid_op(1, impl::op_kind::Sigmoid, "sigmoid");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    sigmoid_op.add_input(add_dst_lt);
    sigmoid_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.add_op(&sigmoid_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::any);
    ASSERT_EQ(compiled_dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddAdd) {
    impl::engine_t *eng = get_engine();

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

    impl::op_t add0_op(0, impl::op_kind::Add, "add0");
    impl::op_t add1_op(1, impl::op_kind::Add, "add1");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, src_shape, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(2, src_shape, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, src_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, src_shape, impl::data_type::f32);

    add0_op.add_input(src0_lt);
    add0_op.add_input(src1_lt);
    add0_op.add_output(add_dst_lt);
    add1_op.add_input(add_dst_lt);
    add1_op.add_input(post_src_lt);
    add1_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&add0_op);
    g.add_op(&add1_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    // inplace
    auto inplace_pair = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pair[0].input_id, post_src_lt.id);
    ASSERT_EQ(inplace_pair[0].output_id, dst_lt.id);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    impl::tensor_t dst_inplace_ts(dst_lt, eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_inplace_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(post_src[i], dst[i]);
    }
}

TEST(Execute, ScalarScalarAdd) {
    impl::op_t add_op(impl::op_kind::Add);
    impl::engine_t *eng = get_engine();

    test::vector<float> src0_data {10.f};
    test::vector<float> src1_data {2.f};
    test::vector<float> ref_dst_data {12.f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src0
            = utils::logical_tensor_init(0, {}, impl::data_type::f32);
    impl::logical_tensor_t src1
            = utils::logical_tensor_init(1, {}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0);
    add_op.add_input(src1);
    add_op.add_output(dst);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0, &src1};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    // output should be a scalar (ndims=0, layout_type=strided)
    impl::logical_tensor_t scalar_lt;
    cp.query_logical_tensor(dst.id, &scalar_lt);
    ASSERT_EQ(scalar_lt.layout_type, impl::layout_type::strided);
    ASSERT_EQ(scalar_lt.ndims, 0);

    impl::tensor_t src0_ts(src0, eng, src0_data.data());
    impl::tensor_t src1_ts(src1, eng, src1_data.data());
    impl::tensor_t dst_ts(scalar_lt, eng, dst_data.data());

    impl::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src0_ts, src1_ts}, {dst_ts}),
            impl::status::success);
    strm->wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, ScalarVectorAdd) {
    impl::op_t add_op(impl::op_kind::Add);
    impl::engine_t *eng = get_engine();

    test::vector<float> src0_data {10.f};
    test::vector<float> src1_data {2.f};
    test::vector<float> ref_dst_data {12.f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src0
            = utils::logical_tensor_init(0, {}, impl::data_type::f32);
    impl::logical_tensor_t src1
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0);
    add_op.add_input(src1);
    add_op.add_output(dst);

    impl::graph_t g(eng->kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0, &src1};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    // output should be a scalar (ndims=0, layout_type=strided)
    impl::logical_tensor_t scalar_lt;
    cp.query_logical_tensor(dst.id, &scalar_lt);
    ASSERT_EQ(scalar_lt.layout_type, impl::layout_type::strided);
    ASSERT_EQ(scalar_lt.ndims, 1);

    impl::tensor_t src0_ts(src0, eng, src0_data.data());
    impl::tensor_t src1_ts(src1, eng, src1_data.data());
    impl::tensor_t dst_ts(scalar_lt, eng, dst_data.data());

    impl::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src0_ts, src1_ts}, {dst_ts}),
            impl::status::success);
    strm->wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, MulAddPerTensorBroadcast) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0};
    test::vector<float> post_src {2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(0, impl::op_kind::Multiply, "mul");
    impl::op_t add_op(1, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(2, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 3, 3}, impl::data_type::f32);

    mul_op.add_input(src0_lt);
    mul_op.add_input(src1_lt);
    mul_op.add_output(mul_dst_lt);
    add_op.add_input(mul_dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&mul_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MulAddPerHwBroadcast) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0(18, 2.0);
    test::vector<float> src1(1, 2.0);
    test::vector<float> post_src(6, 2.0);
    test::vector<float> ref_dst(18, 6.0);
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(0, impl::op_kind::Multiply, "mul");
    impl::op_t add_op(1, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 3, 2, 3}, impl::data_type::f32);

    mul_op.add_input(src0_lt);
    mul_op.add_input(src1_lt);
    mul_op.add_output(mul_dst_lt);
    add_op.add_input(mul_dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&mul_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MulAddPerChannelBroadcast) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(0, impl::op_kind::Multiply, "mul");
    impl::op_t add_op(1, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 1, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 1, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {3, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 3, 1, 3}, impl::data_type::f32);

    mul_op.add_input(src0_lt);
    mul_op.add_input(src1_lt);
    mul_op.add_output(mul_dst_lt);
    add_op.add_input(mul_dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&mul_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    impl::tensor_t src0_ts(src0_lt, eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm->wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MulAddAdd) {
    impl::engine_t *eng = get_engine();

    test::vector<float> mul_src1(1917, 2.5);
    test::vector<float> mul_src2(1, 1.5);
    test::vector<float> add1_src(1917, 1.15);
    test::vector<float> add2_src(1917, 1.07);
    test::vector<float> ref_dst(1917, 5.97);
    test::vector<float> dst(1917, 0.0);

    impl::op_t mul_op(0, impl::op_kind::Multiply, "mul");
    impl::op_t add1_op(1, impl::op_kind::Add, "add");
    impl::op_t add2_op(2, impl::op_kind::Add, "add");
    add1_op.set_attr<std::string>(impl::op_attr::auto_broadcast, "none");
    add2_op.set_attr<std::string>(impl::op_attr::auto_broadcast, "none");

    impl::logical_tensor_t mul_src1_lt
            = utils::logical_tensor_init(0, {1917}, impl::data_type::f32);
    impl::logical_tensor_t mul_src2_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(2, {1917}, impl::data_type::f32);
    impl::logical_tensor_t add1_src_lt
            = utils::logical_tensor_init(3, {1917}, impl::data_type::f32);
    impl::logical_tensor_t add1_dst_lt
            = utils::logical_tensor_init(4, {1917}, impl::data_type::f32);
    impl::logical_tensor_t add2_src_lt
            = utils::logical_tensor_init(5, {1917}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(6, {1917}, impl::data_type::f32);

    mul_op.add_input(mul_src1_lt);
    mul_op.add_input(mul_src2_lt);
    mul_op.add_output(mul_dst_lt);

    add1_op.add_input(mul_dst_lt);
    add1_op.add_input(add1_src_lt);
    add1_op.add_output(add1_dst_lt);

    add2_op.add_input(add1_dst_lt);
    add2_op.add_input(add2_src_lt);
    add2_op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&mul_op);
    g.add_op(&add1_op);
    g.add_op(&add2_op);
    g.build_graph();

    run_all_passes(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &mul_src1_lt, &mul_src2_lt, &add1_src_lt, &add2_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    impl::tensor_t mul_src1_ts(mul_src1_lt, eng, mul_src1.data());
    impl::tensor_t mul_src2_ts(mul_src2_lt, eng, mul_src2.data());
    impl::tensor_t add1_src_ts(add1_src_lt, eng, add1_src.data());
    impl::tensor_t add2_src_ts(add2_src_lt, eng, add2_src.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();
    cp.execute(strm, {mul_src1_ts, mul_src2_ts, add1_src_ts, add2_src_ts},
            {dst_ts});
    strm->wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}
