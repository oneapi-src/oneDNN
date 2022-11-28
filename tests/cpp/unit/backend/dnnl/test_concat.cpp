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

#include <functional>
#include <random>

#include "gtest/gtest.h"

#include "cpp/unit/backend/dnnl/dnnl_test_common.hpp"
#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

struct dnnl_graph_test_concat_params {
    std::vector<impl::dim_t> src0_shape;
    std::vector<impl::dim_t> src1_shape;
    std::vector<impl::dim_t> dst_shape;
    int64_t axis;
    bool is_nhwc;
};

class Concat : public ::testing::TestWithParam<dnnl_graph_test_concat_params> {
public:
    void TestConcat() {
        const auto params = ::testing::TestWithParam<
                dnnl_graph_test_concat_params>::GetParam();

        std::vector<impl::dim_t> src0_dims = params.src0_shape;
        std::vector<impl::dim_t> src1_dims = params.src1_shape;
        std::vector<impl::dim_t> dst_dims = params.dst_shape;
        test::vector<float> src0_data(product(src0_dims));
        test::vector<float> src1_data(product(src1_dims));
        test::vector<float> case1_out_data(product(dst_dims));
        test::vector<float> case2_out_data(product(dst_dims));
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src0_data.begin(), src0_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(src1_data.begin(), src1_data.end(),
                [&]() { return f32_distribution(generator); });

        const auto &axis = params.axis;

        impl::op_t concat_op(impl::op_kind::Concat, "concat");
        concat_op.set_attr<int64_t>(impl::op_attr::axis, axis);

        impl::logical_tensor_t src0_lt = utils::logical_tensor_init(
                0, src0_dims, impl::data_type::f32, impl::layout_type::strided);

        impl::logical_tensor_t src1_lt;
        if (!params.is_nhwc) {
            src1_lt = utils::logical_tensor_init(1, src1_dims,
                    impl::data_type::f32, impl::layout_type::strided);
        } else {
            auto permuted_strides = utils::compute_dense_strides(src1_dims);
            // convert strides to nhwc format
            permuted_strides.insert(
                    permuted_strides.begin() + 1, permuted_strides.back());
            permuted_strides.pop_back();
            src1_lt = utils::logical_tensor_init(
                    1, src1_dims, permuted_strides, impl::data_type::f32);
        }

        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, dst_dims, impl::data_type::f32, impl::layout_type::strided);

        concat_op.add_input(src0_lt);
        concat_op.add_input(src1_lt);
        concat_op.add_output(dst_lt);

        auto &eng = get_engine();
        auto &strm = get_stream();
        impl::graph_t g(eng.kind());
        g.add_op(&concat_op);
        g.build_graph();

        impl::tensor_t src0_ts(src0_lt, &eng, src0_data.data());
        impl::tensor_t src1_ts(src1_lt, &eng, src1_data.data());
        impl::tensor_t case1_dst_ts(dst_lt, &eng, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {src0_ts, src1_ts}, {case1_dst_ts}, eng, strm),
                impl::status::success);

        impl::pass::pass_base_ptr apass = get_pass("concat_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

        p.compile(&cp, inputs, outputs, &eng);

        impl::tensor_t case2_dst_ts(dst_lt, &eng, case2_out_data.data());
        cp.execute(&strm, {src0_ts, src1_ts}, {case2_dst_ts});
        strm.wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
};

TEST_P(Concat, TestConcat) {
    TestConcat();
}

INSTANTIATE_TEST_SUITE_P(Execute, Concat,
        ::testing::Values(
                // 2D, axis = 0
                dnnl_graph_test_concat_params {
                        {1, 2}, {1, 2}, {2, 2}, 0, false},
                // 2D, axis = 1
                dnnl_graph_test_concat_params {
                        {1, 2}, {1, 2}, {1, 4}, 1, false},
                // 3D, axis = 0
                dnnl_graph_test_concat_params {
                        {1, 2, 2}, {1, 2, 2}, {2, 2, 2}, 0, false},
                // 3D, axis = 1
                dnnl_graph_test_concat_params {
                        {1, 2, 2}, {1, 2, 2}, {1, 4, 2}, 1, false},
                // 3D, axis = 2
                dnnl_graph_test_concat_params {
                        {1, 2, 2}, {1, 2, 2}, {1, 2, 4}, 2, false},
                // 4D, axis = 0
                dnnl_graph_test_concat_params {
                        {1, 2, 2, 2}, {1, 2, 2, 2}, {2, 2, 2, 2}, 0, false},
                // 4D, axis = 0
                dnnl_graph_test_concat_params {
                        {1, 2, 2, 2}, {1, 2, 2, 2}, {2, 2, 2, 2}, 0, true},
                // 4D, axis = 1
                dnnl_graph_test_concat_params {
                        {1, 2, 2, 2}, {1, 2, 2, 2}, {1, 4, 2, 2}, 1, false},
                // 4D, axis = 2
                dnnl_graph_test_concat_params {
                        {1, 2, 2, 2}, {1, 2, 2, 2}, {1, 2, 4, 2}, 2, false},
                // 4D, axis = 3
                dnnl_graph_test_concat_params {
                        {1, 2, 2, 2}, {1, 2, 2, 2}, {1, 2, 2, 4}, 3, false},
                // 4D, axis = -1
                dnnl_graph_test_concat_params {
                        {1, 2, 2, 2}, {1, 2, 2, 2}, {1, 2, 2, 4}, -1, false}));

TEST(Compile, ConcatWithMoreInputs) {
    size_t num_inputs = 64;
    const std::vector<impl::dim_t> src_dims = {1, 2, 2, 2};
    const std::vector<impl::dim_t> dst_dims
            = {1, 2, 2, 2 * static_cast<impl::dim_t>(num_inputs)};
    test::vector<float> dst_data(dst_dims.size(), 0.);
    const auto axis = 3;

    impl::op_t concat_op(impl::op_kind::Concat, "concat");
    concat_op.set_attr<int64_t>(impl::op_attr::axis, axis);

    std::vector<impl::logical_tensor_t> input_lts;
    for (size_t i = 0; i < num_inputs; ++i) {
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                i, src_dims, impl::data_type::f32, impl::layout_type::strided);
        input_lts.push_back(src_lt);
        concat_op.add_input(input_lts[i]);
    }
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(num_inputs,
            dst_dims, impl::data_type::f32, impl::layout_type::strided);
    concat_op.add_output(dst_lt);

    auto &eng = get_engine();
    impl::graph_t g(eng.kind());
    g.add_op(&concat_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("concat_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs;
    for (size_t i = 0; i < num_inputs; ++i) {
        inputs.push_back(&input_lts[i]);
    }
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    auto ret = p.compile(&cp, inputs, outputs, &eng);
    ASSERT_EQ(ret, impl::status::success);
}

TEST(Compile, ConcatWithMoreInputsFail) {
    size_t num_inputs = VARIADIC_INPUT_NUM + 1;
    const std::vector<impl::dim_t> src_dims = {1, 2, 2, 2};
    const std::vector<impl::dim_t> dst_dims
            = {1, 2, 2, 2 * static_cast<impl::dim_t>(num_inputs)};
    test::vector<float> dst_data(dst_dims.size(), 0.);
    const auto axis = 3;

    impl::op_t concat_op(impl::op_kind::Concat, "concat");
    concat_op.set_attr<int64_t>(impl::op_attr::axis, axis);

    std::vector<impl::logical_tensor_t> input_lts;
    for (size_t i = 0; i < num_inputs; ++i) {
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                i, src_dims, impl::data_type::f32, impl::layout_type::strided);
        input_lts.push_back(src_lt);
        concat_op.add_input(input_lts[i]);
    }
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(num_inputs,
            dst_dims, impl::data_type::f32, impl::layout_type::strided);
    concat_op.add_output(dst_lt);

    auto &eng = get_engine();
    impl::graph_t g(eng.kind());
    ASSERT_EQ(g.add_op(&concat_op), impl::status::invalid_op);
}

TEST(ExecuteSubgraphInt8, Concat) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    const int64_t channels = 2;
    const int64_t n_inputs = 3;
    std::vector<int64_t> in_shape {2, channels, 4, 4};
    std::vector<int64_t> out_shape {2, channels * n_inputs, 4, 4};

    std::vector<float> scales = {5.f / 127.f};
    std::vector<int64_t> zps = {0};

    test::vector<int8_t> src0_s8_data(utils::product(in_shape));
    test::vector<int8_t> src1_s8_data(utils::product(in_shape));
    test::vector<int8_t> src2_s8_data(utils::product(in_shape));
    test::vector<int8_t> case1_dst_s8_data(utils::product(out_shape));
    test::vector<int8_t> case2_dst_s8_data(utils::product(out_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
    std::generate(src0_s8_data.begin(), src0_s8_data.end(),
            [&]() { return static_cast<uint8_t>(s8_distribution(generator)); });
    std::generate(src1_s8_data.begin(), src1_s8_data.end(),
            [&]() { return static_cast<uint8_t>(s8_distribution(generator)); });
    std::generate(src2_s8_data.begin(), src2_s8_data.end(),
            [&]() { return static_cast<uint8_t>(s8_distribution(generator)); });

    impl::op_t dq0_op(0, impl::op_kind::Dequantize, "dq0_op");
    impl::op_t dq1_op(1, impl::op_kind::Dequantize, "dq1_op");
    impl::op_t dq2_op(2, impl::op_kind::Dequantize, "dq2_op");
    for (auto dq_op : {&dq0_op, &dq1_op, &dq2_op}) {
        dq_op->set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
        dq_op->set_attr<std::vector<int64_t>>(impl::op_attr::zps, zps);
        dq_op->set_attr<std::vector<float>>(impl::op_attr::scales, scales);
    }

    impl::op_t concat_op(3, impl::op_kind::Concat, "concat_op");
    concat_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    impl::op_t q_op(4, impl::op_kind::Quantize, "q_op");
    q_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    q_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, zps);
    q_op.set_attr<std::vector<float>>(impl::op_attr::scales, scales);

    auto src0_s8 = utils::logical_tensor_init(0, in_shape, impl::data_type::s8);
    auto src0_f32_dq
            = utils::logical_tensor_init(1, in_shape, impl::data_type::f32);

    auto src1_s8 = utils::logical_tensor_init(2, in_shape, impl::data_type::s8);
    auto src1_f32_dq
            = utils::logical_tensor_init(3, in_shape, impl::data_type::f32);

    auto src2_s8 = utils::logical_tensor_init(4, in_shape, impl::data_type::s8);
    auto src2_f32_dq
            = utils::logical_tensor_init(5, in_shape, impl::data_type::f32);

    auto dst_f32
            = utils::logical_tensor_init(6, out_shape, impl::data_type::f32);
    auto dst_s8_q
            = utils::logical_tensor_init(7, out_shape, impl::data_type::s8);

    dq0_op.add_input(src0_s8);
    dq0_op.add_output(src0_f32_dq);

    dq1_op.add_input(src1_s8);
    dq1_op.add_output(src1_f32_dq);

    dq2_op.add_input(src2_s8);
    dq2_op.add_output(src2_f32_dq);

    concat_op.add_input(src0_f32_dq);
    concat_op.add_input(src1_f32_dq);
    concat_op.add_input(src2_f32_dq);
    concat_op.add_output(dst_f32);

    q_op.add_input(dst_f32);
    q_op.add_output(dst_s8_q);

    impl::graph_t g(engine.kind());
    g.add_op(&dq0_op);
    g.add_op(&dq1_op);
    g.add_op(&dq2_op);
    g.add_op(&concat_op);
    g.add_op(&q_op);
    g.build_graph();

    impl::tensor_t src0_s8_ts(src0_s8, &engine, src0_s8_data.data());
    impl::tensor_t src1_s8_ts(src1_s8, &engine, src1_s8_data.data());
    impl::tensor_t src2_s8_ts(src2_s8, &engine, src2_s8_data.data());
    impl::tensor_t case1_dst_s8_ts(dst_s8_q, &engine, case1_dst_s8_data.data());
    impl::tensor_t case2_dst_s8_ts(dst_s8_q, &engine, case2_dst_s8_data.data());

    // -------------------------case 1----------------------------------
    ASSERT_EQ(run_graph(g, {src0_s8_ts, src1_s8_ts, src2_s8_ts},
                      {case1_dst_s8_ts}, engine, strm),
            impl::status::success);

    // -------------------------case 2----------------------------------
    impl::pass::pass_base_ptr apass = get_pass("int8_concat_fusion");
    apass->run(g);

    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src0_s8, &src1_s8, &src2_s8};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8_q};
    p.compile(&cp, lt_ins, lt_outs, &engine);

    cp.execute(&strm, {src0_s8_ts, src1_s8_ts, src2_s8_ts}, {case2_dst_s8_ts});
    strm.wait();

    ASSERT_TRUE(allclose(case1_dst_s8_data, case2_dst_s8_data,
            /*rtol*/ 0.01f,
            /*atol*/ 1.f));
}

TEST(Execute, ConcatEmptyInput) {
    impl::op_t concat_op(impl::op_kind::Concat);
    concat_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    impl::engine_t &eng = get_engine();

    // prepare logical tensor
    impl::logical_tensor_t src0
            = utils::logical_tensor_init(0, {2, 3, 4}, impl::data_type::f32);
    impl::logical_tensor_t src1
            = utils::logical_tensor_init(1, {2, 0, 4}, impl::data_type::f32);
    impl::logical_tensor_t src2
            = utils::logical_tensor_init(2, {2, 3, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, impl::data_type::f32, impl::layout_type::any);

    concat_op.add_input(src0);
    concat_op.add_input(src1);
    concat_op.add_input(src2);
    concat_op.add_output(dst);

    impl::graph_t g(eng.kind());
    g.add_op(&concat_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("concat_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0, &src1, &src2};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);
    ASSERT_EQ(lt.ndims, 3);
    ASSERT_EQ(lt.dims[0], 2);
    ASSERT_EQ(lt.dims[1], 6);
    ASSERT_EQ(lt.dims[2], 4);

    test::vector<float> src0_data(24);
    test::vector<float> src2_data(24);
    test::vector<float> dst_data(48);

    impl::tensor_t src0_ts(src0, &eng, src0_data.data());
    impl::tensor_t src1_ts(src1, &eng, nullptr);
    impl::tensor_t src2_ts(src2, &eng, src2_data.data());
    impl::tensor_t dst_ts(dst, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src0_ts, src1_ts, src2_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
}

TEST(ExecuteSubgraphInt8, ConcatWithManyInputs) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    const int64_t channels = 2;
    const int64_t n_inputs = VARIADIC_INPUT_NUM;
    std::vector<int64_t> in_shape {2, channels, 4, 4};
    std::vector<int64_t> out_shape {2, channels * n_inputs, 4, 4};

    std::vector<float> scales = {5.f / 127.f};
    std::vector<int64_t> zps = {0};

    test::vector<test::vector<int8_t>> src_s8_data(
            n_inputs, test::vector<int8_t>(utils::product(in_shape)));

    test::vector<int8_t> case1_dst_s8_data(utils::product(out_shape));
    test::vector<int8_t> case2_dst_s8_data(utils::product(out_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
    for (auto i = 0; i < n_inputs; ++i) {
        std::generate(src_s8_data[i].begin(), src_s8_data[i].end(), [&]() {
            return static_cast<uint8_t>(s8_distribution(generator));
        });
    }
    int64_t id = 0;
    std::vector<std::shared_ptr<impl::op_t>> dqs;
    for (auto i = 0; i < n_inputs; ++i) {
        dqs.emplace_back(
                std::make_shared<impl::op_t>(id++, impl::op_kind::Dequantize,
                        std::string("dq") + std::to_string(i) + "_op"));
        auto &op = dqs.back();
        op->set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
        op->set_attr<std::vector<int64_t>>(impl::op_attr::zps, zps);
        op->set_attr<std::vector<float>>(impl::op_attr::scales, scales);
    }

    impl::op_t concat_op(id++, impl::op_kind::Concat, "concat_op");
    concat_op.set_attr<int64_t>(impl::op_attr::axis, 1);

    impl::op_t q_op(id++, impl::op_kind::Quantize, "q_op");
    q_op.set_attr<std::string>(impl::op_attr::qtype, "per_tensor");
    q_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, zps);
    q_op.set_attr<std::vector<float>>(impl::op_attr::scales, scales);
    std::vector<impl::logical_tensor_t> lts;
    for (auto i = 0; i < n_inputs; ++i) {
        lts.emplace_back(utils::logical_tensor_init(
                id++, in_shape, impl::data_type::s8));
    }
    for (auto i = 0; i < n_inputs; ++i) {
        auto src0_f32_dq = utils::logical_tensor_init(
                id++, in_shape, impl::data_type::f32);
        dqs[i]->add_input(lts[i]);
        dqs[i]->add_output(src0_f32_dq);
        concat_op.add_input(src0_f32_dq);
    }

    auto dst_f32
            = utils::logical_tensor_init(id++, out_shape, impl::data_type::f32);
    auto dst_s8_q
            = utils::logical_tensor_init(id++, out_shape, impl::data_type::s8);
    concat_op.add_output(dst_f32);

    q_op.add_input(dst_f32);
    q_op.add_output(dst_s8_q);

    impl::graph_t g(engine.kind());
    for (auto i = 0; i < n_inputs; ++i) {
        g.add_op(dqs[i].get());
    }
    g.add_op(&concat_op);
    g.add_op(&q_op);
    g.build_graph();

    std::vector<impl::tensor_t> src_s8_ts;
    for (auto i = 0; i < n_inputs; ++i) {
        impl::tensor_t src0_s8_ts(lts[i], &engine, src_s8_data[i].data());
        src_s8_ts.push_back(std::move(src0_s8_ts));
    }

    impl::tensor_t case1_dst_s8_ts(dst_s8_q, &engine, case1_dst_s8_data.data());
    impl::tensor_t case2_dst_s8_ts(dst_s8_q, &engine, case2_dst_s8_data.data());

    // -------------------------case 1----------------------------------
    ASSERT_EQ(run_graph(g, src_s8_ts, {case1_dst_s8_ts}, engine, strm),
            impl::status::success);

    // -------------------------case 2----------------------------------
    impl::pass::pass_base_ptr apass = get_pass("int8_concat_fusion");
    apass->run(g);

    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins;
    for (auto i = 0; i < n_inputs; ++i) {
        lt_ins.push_back(&lts[i]);
    }

    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8_q};
    p.compile(&cp, lt_ins, lt_outs, &engine);

    cp.execute(&strm, src_s8_ts, {case2_dst_s8_ts});
    strm.wait();

    ASSERT_TRUE(allclose(case1_dst_s8_data, case2_dst_s8_data,
            /*rtol*/ 0.01f,
            /*atol*/ 1.f));
}
