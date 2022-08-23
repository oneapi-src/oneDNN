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

TEST(Execute, InterpolateForwardNearest) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_dst {
            -2.f, -1.5f, -1.5f, -1.f, -0.5f, -0.5f, -1.f, -0.5f, -0.5f};

    impl::op_t op(impl::op_kind::Interpolate);
    op.set_attr<std::string>(impl::op_attr::mode, "nearest");
    op.set_attr(impl::op_attr::sizes, std::vector<int64_t> {3, 3});
    op.set_attr<std::string>(
            impl::op_attr::coordinate_transformation_mode, "half_pixel");
    op.set_attr<std::string>(impl::op_attr::data_format, "NCX");

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, impl::data_type::f32, impl::layout_type::strided);

    op.add_input(src_lt);
    op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("interpolate_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> lt_ins {&src_lt};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_lt};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(*lt_outs[0], &engine, dst.data());
    cp.execute(&strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, InterpolateAddForwardNearest) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> src1 {
            0.f, 0.5f, 1.f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.f};
    test::vector<float> dst_add {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_dst {
            -2.f, -1.f, -0.5f, 0.5f, 1.5f, 2.f, 2.f, 3.f, 3.5f};

    impl::op_t interpolate_node(0, impl::op_kind::Interpolate, "interpolate");
    interpolate_node.set_attr<std::string>(impl::op_attr::mode, "nearest");
    interpolate_node.set_attr(
            impl::op_attr::sizes, std::vector<int64_t> {3, 3});
    interpolate_node.set_attr<std::string>(
            impl::op_attr::coordinate_transformation_mode, "half_pixel");
    interpolate_node.set_attr<std::string>(impl::op_attr::data_format, "NCX");

    impl::op_t add_node(1, impl::op_kind::Add, "add_node");

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            2, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t dst_add_lt = utils::logical_tensor_init(
            3, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);

    interpolate_node.add_input(src_lt);
    interpolate_node.add_output(dst_lt);

    add_node.add_input(dst_lt);
    add_node.add_input(src1_lt);
    add_node.add_output(dst_add_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&interpolate_node);
    g.add_op(&add_node);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("interpolate_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> lt_ins {&src_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_add_lt};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t src1_ts(src1_lt, &engine, src1.data());
    impl::tensor_t dst_add_ts(*lt_outs[0], &engine, dst_add.data());
    cp.execute(&strm, {src_ts, src1_ts}, {dst_add_ts});
    strm.wait();

    for (size_t i = 0; i < dst_add.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_add[i], ref_dst[i]);
    }
}

TEST(Execute, InterpolateSwish) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> dst_mul {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    impl::op_t interpolate_node(0, impl::op_kind::Interpolate, "interpolate");
    interpolate_node.set_attr<std::string>(impl::op_attr::mode, "nearest");
    interpolate_node.set_attr(
            impl::op_attr::sizes, std::vector<int64_t> {3, 3});
    interpolate_node.set_attr<std::string>(
            impl::op_attr::coordinate_transformation_mode, "half_pixel");
    interpolate_node.set_attr<std::string>(impl::op_attr::data_format, "NCX");

    impl::op_t sigmoid_node(1, impl::op_kind::Sigmoid, "sigmoid_node");
    impl::op_t mul_node(2, impl::op_kind::Multiply, "multiply_node");

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t dst_sigmoid_lt = utils::logical_tensor_init(
            2, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t dst_mul_lt = utils::logical_tensor_init(
            3, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);

    interpolate_node.add_input(src_lt);
    interpolate_node.add_output(dst_lt);
    sigmoid_node.add_input(dst_lt);
    sigmoid_node.add_output(dst_sigmoid_lt);
    mul_node.add_input(dst_sigmoid_lt);
    mul_node.add_input(dst_lt);
    mul_node.add_output(dst_mul_lt);

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&interpolate_node), impl::status::success);
    ASSERT_EQ(g.add_op(&sigmoid_node), impl::status::success);
    ASSERT_EQ(g.add_op(&mul_node), impl::status::success);
    ASSERT_EQ(g.build_graph(), impl::status::success);
    ASSERT_EQ(g.num_ops(), 3);

    impl::pass::pass_base_ptr apass = get_pass("interpolate_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> lt_ins {&src_lt};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_mul_lt};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_mul_ts(*lt_outs[0], &engine, dst_mul.data());
    cp.execute(&strm, {src_ts}, {dst_mul_ts});
    strm.wait();
}

TEST(Execute, Interpolate3PostOps) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> src_div {
            1.0, -1.0, -1.0, -1.5, 2.0, 3.0, 4.0, 5.0, 6.0};
    test::vector<float> dst_div(9, 1.0);

    impl::op_t interpolate_node(0, impl::op_kind::Interpolate, "interpolate");
    interpolate_node.set_attr<std::string>(impl::op_attr::mode, "nearest");
    interpolate_node.set_attr(
            impl::op_attr::sizes, std::vector<int64_t> {3, 3});
    interpolate_node.set_attr<std::string>(
            impl::op_attr::coordinate_transformation_mode, "half_pixel");
    interpolate_node.set_attr<std::string>(impl::op_attr::data_format, "NCX");

    impl::op_t relu_node(1, impl::op_kind::ReLU, "relu_node");
    impl::op_t sigmoid_node(2, impl::op_kind::Sigmoid, "sigmoid_node");
    impl::op_t div_node(3, impl::op_kind::Divide, "div_node");

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t dst_relu_lt = utils::logical_tensor_init(
            2, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t dst_sigmoid_lt = utils::logical_tensor_init(
            3, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t src_div_lt = utils::logical_tensor_init(
            4, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t dst_div_lt = utils::logical_tensor_init(
            5, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);

    interpolate_node.add_input(src_lt);
    interpolate_node.add_output(dst_lt);
    relu_node.add_input(dst_lt);
    relu_node.add_output(dst_relu_lt);
    sigmoid_node.add_input(dst_relu_lt);
    sigmoid_node.add_output(dst_sigmoid_lt);
    div_node.add_input(dst_sigmoid_lt);
    div_node.add_input(src_div_lt);
    div_node.add_output(dst_div_lt);

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&interpolate_node), impl::status::success);
    ASSERT_EQ(g.add_op(&relu_node), impl::status::success);
    ASSERT_EQ(g.add_op(&sigmoid_node), impl::status::success);
    ASSERT_EQ(g.add_op(&div_node), impl::status::success);
    ASSERT_EQ(g.build_graph(), impl::status::success);
    ASSERT_EQ(g.num_ops(), 4);

    impl::pass::pass_base_ptr apass = get_pass("interpolate_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> lt_ins {&src_lt, &src_div_lt};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_div_lt};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t src_div_ts(src_div_lt, &engine, src_div.data());
    impl::tensor_t dst_div_ts(dst_div_lt, &engine, dst_div.data());
    cp.execute(&strm, {src_ts, src_div_ts}, {dst_div_ts});
    strm.wait();
}

TEST(Execute, InterpolatePostOps) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    const std::vector<impl::op_kind_t> supported_post_ops = {impl::op_kind::Abs,
            impl::op_kind::Clamp, impl::op_kind::Elu, impl::op_kind::Exp,
            impl::op_kind::GELU, impl::op_kind::HardSwish, impl::op_kind::Log,
            impl::op_kind::Sigmoid, impl::op_kind::SoftPlus, impl::op_kind::Pow,
            impl::op_kind::ReLU, impl::op_kind::Round, impl::op_kind::Sqrt,
            impl::op_kind::Square, impl::op_kind::Tanh, impl::op_kind::Add,
            impl::op_kind::Multiply, impl::op_kind::Maximum,
            impl::op_kind::Minimum, impl::op_kind::Divide,
            impl::op_kind::Subtract};
    const std::vector<impl::op_kind_t> two_inputs_ops {impl::op_kind::Multiply,
            impl::op_kind::Add, impl::op_kind::Maximum, impl::op_kind::Minimum,
            impl::op_kind::Divide, impl::op_kind::Subtract, impl::op_kind::Pow};

    for (const auto &post_op_kind : supported_post_ops) {
        test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
        test::vector<float> src1 {
                0.f, 0.5f, 1.f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.f};
        test::vector<float> dst_add {
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

        impl::op_t interpolate_node(
                0, impl::op_kind::Interpolate, "interpolate");
        interpolate_node.set_attr<std::string>(impl::op_attr::mode, "nearest");
        interpolate_node.set_attr(
                impl::op_attr::sizes, std::vector<int64_t> {3, 3});
        interpolate_node.set_attr<std::string>(
                impl::op_attr::coordinate_transformation_mode, "half_pixel");
        interpolate_node.set_attr<std::string>(
                impl::op_attr::data_format, "NCX");

        impl::op_t post_node(1, post_op_kind, "post_op_node");
        if (post_op_kind == impl::op_kind::Elu) {
            post_node.set_attr<float>(impl::op_attr::alpha, 1.0f);
        } else if (post_op_kind == impl::op_kind::Clamp) {
            post_node.set_attr<float>(impl::op_attr::min, 1.0f);
            post_node.set_attr<float>(impl::op_attr::max, 3.0f);
        }

        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                1, impl::data_type::f32, impl::layout_type::strided);

        impl::logical_tensor_t src1_lt = utils::logical_tensor_init(2,
                {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t dst_post_lt = utils::logical_tensor_init(3,
                {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);

        interpolate_node.add_input(src_lt);
        interpolate_node.add_output(dst_lt);

        post_node.add_input(dst_lt);
        if (std::find(
                    two_inputs_ops.begin(), two_inputs_ops.end(), post_op_kind)
                != two_inputs_ops.end()) {
            post_node.add_input(src1_lt);
        }
        post_node.add_output(dst_post_lt);

        impl::graph_t g(engine.kind());
        g.add_op(&interpolate_node);
        g.add_op(&post_node);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("interpolate_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];
        ASSERT_TRUE(part != nullptr);

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> lt_ins {&src_lt};
        if (std::find(
                    two_inputs_ops.begin(), two_inputs_ops.end(), post_op_kind)
                != two_inputs_ops.end()) {
            lt_ins.emplace_back(&src1_lt);
        }
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_post_lt};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        impl::tensor_t src_ts(src_lt, &engine, src.data());
        impl::tensor_t src1_ts(src1_lt, &engine, src1.data());
        impl::tensor_t dst_add_ts(*lt_outs[0], &engine, dst_add.data());
        if (std::find(
                    two_inputs_ops.begin(), two_inputs_ops.end(), post_op_kind)
                != two_inputs_ops.end()) {
            cp.execute(&strm, {src_ts, src1_ts}, {dst_add_ts});
        } else {
            cp.execute(&strm, {src_ts}, {dst_add_ts});
        }

        strm.wait();
    }
}

TEST(Execute, InterpolateForwardLinear) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_dst {
            -2.f, -1.75f, -1.5f, -1.5f, -1.25f, -1.f, -1.f, -0.75f, -0.5f};

    impl::op_t op(impl::op_kind::Interpolate);
    op.set_attr<std::string>(impl::op_attr::mode, "linear");
    op.set_attr(impl::op_attr::sizes, std::vector<int64_t> {3, 3});
    op.set_attr<std::string>(
            impl::op_attr::coordinate_transformation_mode, "half_pixel");
    op.set_attr<std::string>(impl::op_attr::data_format, "NXC");

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, impl::data_type::f32, impl::layout_type::strided);

    op.add_input(src_lt);
    op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("interpolate_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> lt_ins {&src_lt};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_lt};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(*lt_outs[0], &engine, dst.data());
    cp.execute(&strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, InterpolateBackwardNearest) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> diff_src {0.0, 0.0, 0.0, 0.0};
    test::vector<float> ref_diff_src {0.f, 3.f, 9.f, 24.f};

    impl::op_t op(impl::op_kind::InterpolateBackprop);
    op.set_attr<std::string>(impl::op_attr::mode, "nearest");
    op.set_attr<std::string>(impl::op_attr::data_format, "NCX");

    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(1, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::strided);

    op.add_input(src_lt);
    op.add_input(diff_dst_lt);
    op.add_output(diff_src_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("interpolate_bwd_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(diff_src_lt, &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, InterpolateBackwardLinear) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> diff_src {0.0, 0.0, 0.0, 0.0};
    test::vector<float> ref_diff_src {3.f, 6.f, 12.f, 15.f};

    impl::op_t op(impl::op_kind::InterpolateBackprop);
    op.set_attr<std::string>(impl::op_attr::mode, "linear");
    op.set_attr<std::string>(impl::op_attr::data_format, "NCX");

    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(1, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::strided);

    op.add_input(src_lt);
    op.add_input(diff_dst_lt);
    op.add_output(diff_src_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("interpolate_bwd_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(diff_src_lt, &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}
