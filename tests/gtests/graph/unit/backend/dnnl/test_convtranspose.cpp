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

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

struct convtranspose_params_t {
    std::string data_format;
    std::string filter_format;
    std::vector<int64_t> src_shape;
    std::vector<int64_t> weight_shape;
    bool with_bias;
    std::vector<int64_t> bias_shape;
    std::vector<int64_t> dst_shape;
    graph::dnnl_impl::dims strides;
    graph::dnnl_impl::dims dilations;
    graph::dnnl_impl::dims pads_begin;
    graph::dnnl_impl::dims pads_end;
};

class convtranspose_4d_5d_t
    : public ::testing::TestWithParam<convtranspose_params_t> {
public:
    void TestConvtranspose() {
        using dims = graph::dnnl_impl::dims;

        auto params
                = ::testing::TestWithParam<convtranspose_params_t>::GetParam();

        graph::engine_t *eng = get_engine();
        std::vector<graph::dim_t> src_dims = params.src_shape;
        std::vector<graph::dim_t> weight_dims = params.weight_shape;
        std::vector<graph::dim_t> bias_dims = params.bias_shape;
        std::vector<graph::dim_t> dst_dims = params.dst_shape;
        test::vector<float> src_data(product(src_dims));
        test::vector<float> weight_data(product(weight_dims));
        test::vector<float> bias_data(product(bias_dims));
        test::vector<float> case1_out_data(product(dst_dims));
        test::vector<float> case2_out_data(product(dst_dims));
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });

        graph::op_t convtranspose_op(graph::op_kind::ConvTranspose);
        convtranspose_op.set_attr<dims>(
                graph::op_attr::strides, params.strides);
        convtranspose_op.set_attr<dims>(
                graph::op_attr::dilations, params.dilations);
        convtranspose_op.set_attr<dims>(
                graph::op_attr::pads_begin, params.pads_begin);
        convtranspose_op.set_attr<dims>(
                graph::op_attr::pads_end, params.pads_end);
        convtranspose_op.set_attr<int64_t>(graph::op_attr::groups, 1);
        convtranspose_op.set_attr<std::string>(
                graph::op_attr::data_format, params.data_format);
        convtranspose_op.set_attr<std::string>(
                graph::op_attr::weights_format, params.filter_format);

        // prepare logical tensor
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, params.src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, params.weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, params.dst_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_lt
                = utils::logical_tensor_init(3, graph::data_type::f32);

        convtranspose_op.add_input(src_lt);
        convtranspose_op.add_input(weight_lt);
        if (params.with_bias) {
            bias_lt = utils::logical_tensor_init(
                    3, params.bias_shape, graph::data_type::f32);
            convtranspose_op.add_input(bias_lt);
        }
        convtranspose_op.add_output(dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&convtranspose_op);
        g.finalize();

        graph::pass::pass_base_ptr apass
                = get_pass("convtranspose_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src_lt, &weight_lt};
        if (params.with_bias) inputs.push_back(&bias_lt);
        std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

        p.compile(&cp, inputs, outputs, eng);
        ASSERT_EQ(dst_lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(src_lt, eng, src_data.data());
        graph::tensor_t weight_ts(weight_lt, eng, weight_data.data());
        graph::tensor_t dst1_ts(dst_lt, eng, case1_out_data.data());
        graph::tensor_t dst2_ts(dst_lt, eng, case2_out_data.data());
        graph::tensor_t bias_ts;
        if (params.with_bias) {
            bias_ts = graph::tensor_t(bias_lt, eng, bias_data.data());
        }
        graph::stream_t *strm = get_stream();
        if (params.with_bias) {
            ASSERT_EQ(run_graph(g, {src_ts, weight_ts, bias_ts}, {dst1_ts},
                              *eng, *strm),
                    graph::status::success);
            cp.execute(strm, {src_ts, weight_ts, bias_ts}, {dst2_ts});
        } else {
            ASSERT_EQ(run_graph(g, {src_ts, weight_ts}, {dst1_ts}, *eng, *strm),
                    graph::status::success);
            cp.execute(strm, {src_ts, weight_ts}, {dst2_ts});
        }
        strm->wait();
        for (size_t i = 0; i < case2_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
};

struct convtranspose_bwd_params_t {
    std::vector<int64_t> src_dims;
    std::vector<int64_t> wei_dims;
    std::vector<int64_t> dst_dims;
    graph::dnnl_impl::dims strides;
    graph::dnnl_impl::dims pads_begin;
    graph::dnnl_impl::dims pads_end;
    graph::dnnl_impl::dims dilations;
    std::string data_format;
    std::string filter_format;
};

class convtranspose_backprop_data_t
    : public ::testing::TestWithParam<convtranspose_bwd_params_t> {
public:
    void TestConvTransposeBackwardData() {
        using dims = graph::dnnl_impl::dims;

        auto params = ::testing::TestWithParam<
                convtranspose_bwd_params_t>::GetParam();

        // default engine kind is cpu.
        graph::engine_t *eng = get_engine();
        std::vector<graph::dim_t> src_dims = params.src_dims;
        std::vector<graph::dim_t> wei_dims = params.wei_dims;
        std::vector<graph::dim_t> dst_dims = params.dst_dims;
        test::vector<float> diff_dst(product(dst_dims));
        test::vector<float> weight(product(wei_dims));
        test::vector<float> case1_out_data(product(src_dims));
        test::vector<float> case2_out_data(product(src_dims));
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(diff_dst.begin(), diff_dst.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(weight.begin(), weight.end(),
                [&]() { return f32_distribution(generator); });

        graph::op_t deconv_op(graph::op_kind::ConvTransposeBackwardData);
        deconv_op.set_attr<dims>(graph::op_attr::strides, params.strides);
        deconv_op.set_attr<dims>(graph::op_attr::dilations, params.dilations);
        deconv_op.set_attr<dims>(graph::op_attr::pads_begin, params.pads_begin);
        deconv_op.set_attr<dims>(graph::op_attr::pads_end, params.pads_end);
        deconv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
        deconv_op.set_attr<std::string>(
                graph::op_attr::data_format, params.data_format);
        deconv_op.set_attr<std::string>(
                graph::op_attr::weights_format, params.filter_format);

        // prepare logical tensor
        graph::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
                0, dst_dims, graph::data_type::f32);
        graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, wei_dims, graph::data_type::f32);
        graph::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
                2, src_dims, graph::data_type::f32);

        deconv_op.add_input(diff_dst_lt);
        deconv_op.add_input(weight_lt);
        deconv_op.add_output(diff_src_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&deconv_op);
        g.finalize();

        graph::pass::pass_base_ptr apass
                = get_pass("convtranspose_data_bwd_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &diff_dst_lt, &weight_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&diff_src_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

        graph::logical_tensor_t lt;
        cp.query_logical_tensor(diff_src_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t diff_dst_ts(diff_dst_lt, eng, diff_dst.data());
        graph::tensor_t weight_ts(weight_lt, eng, weight.data());
        graph::tensor_t diff_src1_ts(diff_src_lt, eng, case1_out_data.data());
        graph::tensor_t diff_src2_ts(diff_src_lt, eng, case2_out_data.data());

        graph::stream_t *strm = get_stream();
        ASSERT_EQ(run_graph(g, {diff_dst_ts, weight_ts}, {diff_src1_ts}, *eng,
                          *strm),
                graph::status::success);
        cp.execute(strm, {diff_dst_ts, weight_ts}, {diff_src2_ts});
        strm->wait();
        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
};

class convtranspose_backprop_filters_t
    : public ::testing::TestWithParam<convtranspose_bwd_params_t> {
public:
    void TestConvTransposeBackwardWeights() {
        using dims = graph::dnnl_impl::dims;

        auto params = ::testing::TestWithParam<
                convtranspose_bwd_params_t>::GetParam();

        graph::engine_t *eng = get_engine();
        std::vector<graph::dim_t> src_dims = params.src_dims;
        std::vector<graph::dim_t> wei_dims = params.wei_dims;
        std::vector<graph::dim_t> dst_dims = params.dst_dims;
        test::vector<float> src(product(src_dims));
        test::vector<float> diff_dst(product(dst_dims));
        test::vector<float> case1_out_data(product(wei_dims));
        test::vector<float> case2_out_data(product(wei_dims));
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src.begin(), src.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(diff_dst.begin(), diff_dst.end(),
                [&]() { return f32_distribution(generator); });

        graph::op_t deconv_op(graph::op_kind::ConvTransposeBackwardWeights);
        deconv_op.set_attr<dims>(graph::op_attr::strides, params.strides);
        deconv_op.set_attr<dims>(graph::op_attr::dilations, params.dilations);
        deconv_op.set_attr<dims>(graph::op_attr::pads_begin, params.pads_begin);
        deconv_op.set_attr<dims>(graph::op_attr::pads_end, params.pads_end);
        deconv_op.set_attr<dims>(
                graph::op_attr::weights_shape, params.wei_dims);
        deconv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
        deconv_op.set_attr<std::string>(
                graph::op_attr::data_format, params.data_format);
        deconv_op.set_attr<std::string>(
                graph::op_attr::weights_format, params.filter_format);

        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, src_dims, graph::data_type::f32);
        graph::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
                1, dst_dims, graph::data_type::f32);
        graph::logical_tensor_t diff_wei_lt = utils::logical_tensor_init(
                2, wei_dims, graph::data_type::f32);

        deconv_op.add_input(src_lt);
        deconv_op.add_input(diff_dst_lt);
        deconv_op.add_output(diff_wei_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&deconv_op);
        g.finalize();

        graph::pass::pass_base_ptr apass
                = get_pass("convtranspose_filter_bwd_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src_lt, &diff_dst_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&diff_wei_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

        graph::logical_tensor_t lt;
        cp.query_logical_tensor(diff_wei_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(src_lt, eng, src.data());
        graph::tensor_t diff_dst_ts(diff_dst_lt, eng, diff_dst.data());
        graph::tensor_t diff_wei_ts1(diff_wei_lt, eng, case1_out_data.data());
        graph::tensor_t diff_wei_ts2(diff_wei_lt, eng, case2_out_data.data());

        graph::stream_t *strm = get_stream();
        ASSERT_EQ(run_graph(g, {src_ts, diff_dst_ts}, {diff_wei_ts1}, *eng,
                          *strm),
                graph::status::success);
        cp.execute(strm, {src_ts, diff_dst_ts}, {diff_wei_ts2});
        strm->wait();
        for (size_t i = 0; i < case2_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
};

struct convtranspose_add_params_t {
    std::vector<graph::dim_t> add_src_shape;
    bool swap;
    bool with_bias;
};

class test_convtranspose_add_compile_t
    : public ::testing::TestWithParam<convtranspose_add_params_t> {
public:
    void TestConvTransposeAdd() {
        const auto params = ::testing::TestWithParam<
                convtranspose_add_params_t>::GetParam();
        using dims = graph::dnnl_impl::dims;

        graph::engine_t *eng = get_engine();
        std::vector<graph::dim_t> add_dims = params.add_src_shape;
        test::vector<float> src_data {-1.0, 2.5, 5.0, 1.5};
        test::vector<float> weight_data {
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
        test::vector<float> bias_data = {1.0};
        test::vector<float> add_src_data(product(add_dims));
        test::vector<float> case1_out_data(16);
        test::vector<float> case2_out_data(16);
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(add_src_data.begin(), add_src_data.end(),
                [&]() { return f32_distribution(generator); });

        graph::op_t convtranspose_op(
                0, graph::op_kind::ConvTranspose, "ConvTranspose");
        convtranspose_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
        convtranspose_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
        convtranspose_op.set_attr<dims>(
                graph::op_attr::pads_begin, dims {0, 0});
        convtranspose_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
        convtranspose_op.set_attr<int64_t>(graph::op_attr::groups, 1);
        convtranspose_op.set_attr<std::string>(
                graph::op_attr::data_format, "NXC");
        convtranspose_op.set_attr<std::string>(
                graph::op_attr::weights_format, "IOX");
        graph::op_t add_op(1, graph::op_kind::Add, "Add");

        // prepare logical tensor
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 2, 2, 1}, graph::data_type::f32);
        graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {1, 1, 3, 3}, graph::data_type::f32);
        graph::logical_tensor_t bias_lt
                = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
        graph::logical_tensor_t add_src_lt = utils::logical_tensor_init(
                3, add_dims, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(4,
                {1, 4, 4, 1}, graph::data_type::f32, graph::layout_type::any);
        graph::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
                5, {1, 4, 4, 1}, graph::data_type::f32);

        convtranspose_op.add_input(src_lt);
        convtranspose_op.add_input(weight_lt);
        if (params.with_bias) { convtranspose_op.add_input(bias_lt); }
        convtranspose_op.add_output(dst_lt);
        if (params.swap) {
            add_op.add_input(add_src_lt);
            add_op.add_input(dst_lt);
        } else {
            add_op.add_input(dst_lt);
            add_op.add_input(add_src_lt);
        }
        add_op.add_output(add_dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&convtranspose_op);
        g.add_op(&add_op);
        g.finalize();

        graph::pass::pass_base_ptr apass
                = get_pass("convtranspose_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);
        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src_lt, &weight_lt};
        if (params.with_bias) { inputs.push_back(&bias_lt); }
        inputs.push_back(&add_src_lt);
        std::vector<const graph::logical_tensor_t *> outputs {&add_dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);
        graph::logical_tensor_t lt;
        cp.query_logical_tensor(add_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(src_lt, eng, src_data.data());
        graph::tensor_t weight_ts(weight_lt, eng, weight_data.data());
        graph::tensor_t bias_ts;
        if (params.with_bias)
            bias_ts = graph::tensor_t(bias_lt, eng, bias_data.data());
        graph::tensor_t add_src_ts(add_src_lt, eng, add_src_data.data());
        graph::tensor_t add_dst_ts1(add_dst_lt, eng, case1_out_data.data());
        graph::tensor_t add_dst_ts2(add_dst_lt, eng, case2_out_data.data());

        graph::stream_t *strm = get_stream();
        if (params.with_bias) {
            ASSERT_EQ(run_graph(g, {src_ts, weight_ts, bias_ts, add_src_ts},
                              {add_dst_ts1}, *eng, *strm),
                    graph::status::success);
            cp.execute(strm, {src_ts, weight_ts, bias_ts, add_src_ts},
                    {add_dst_ts2});
        } else {
            ASSERT_EQ(run_graph(g, {src_ts, weight_ts, add_src_ts},
                              {add_dst_ts1}, *eng, *strm),
                    graph::status::success);
            cp.execute(strm, {src_ts, weight_ts, add_src_ts}, {add_dst_ts2});
        }
        strm->wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
};

TEST(Compile, ConvtransposeFp32) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *eng = get_engine();

    graph::op_t convtranspose_op(graph::op_kind::ConvTranspose);
    convtranspose_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    convtranspose_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    convtranspose_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    convtranspose_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    convtranspose_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    convtranspose_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    convtranspose_op.set_attr<std::string>(
            graph::op_attr::weights_format, "IOX");

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 222, 222}, graph::data_type::f32);
    graph::logical_tensor_t weight = utils::logical_tensor_init(
            1, {3, 16, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(2,
            {8, 16, 224, 224}, graph::data_type::f32, graph::layout_type::any);

    convtranspose_op.add_input(src);
    convtranspose_op.add_input(weight);
    convtranspose_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&convtranspose_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("convtranspose_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);
    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type,
            eng->kind() == graph::engine_kind::gpu
                    ? graph::layout_type::opaque
                    : graph::layout_type::strided);
}

TEST_P(convtranspose_4d_5d_t, TestConvtranspose) {
    TestConvtranspose();
}

INSTANTIATE_TEST_SUITE_P(Execute, convtranspose_4d_5d_t,
        ::testing::Values(convtranspose_params_t {"NXC", "IOX", {1, 2, 2, 1},
                                  {1, 1, 3, 3}, false, {1}, {1, 4, 4, 1},
                                  {1, 1}, {1, 1}, {0, 0}, {0, 0}},
                convtranspose_params_t {"NCX", "IOX", {1, 1, 2, 2},
                        {1, 1, 3, 3}, true, {1}, {1, 1, 4, 4}, {1, 1}, {1, 1},
                        {0, 0}, {0, 0}},
                convtranspose_params_t {"NXC", "XOI", {1, 2, 2, 1},
                        {3, 3, 1, 1}, false, {1}, {1, 4, 4, 1}, {1, 1}, {1, 1},
                        {0, 0}, {0, 0}},
                convtranspose_params_t {"NCX", "XOI", {1, 1, 2, 2},
                        {3, 3, 1, 1}, true, {1}, {1, 1, 4, 4}, {1, 1}, {1, 1},
                        {0, 0}, {0, 0}},
                convtranspose_params_t {"NCX", "XOI", {1, 1, 2, 2},
                        {3, 3, 2, 1}, true, {2}, {1, 2, 4, 4}, {1, 1}, {1, 1},
                        {0, 0}, {0, 0}},
                convtranspose_params_t {"NXC", "IOX", {1, 1, 2, 2, 1},
                        {1, 1, 1, 3, 3}, false, {1}, {1, 1, 4, 4, 1}, {1, 1, 1},
                        {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
                convtranspose_params_t {"NCX", "IOX", {1, 1, 1, 2, 2},
                        {1, 2, 1, 3, 3}, true, {2}, {1, 2, 1, 4, 4}, {1, 1, 1},
                        {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
                convtranspose_params_t {"NCX", "IOX", {1, 1, 1, 2, 2},
                        {1, 1, 1, 3, 3}, false, {1}, {1, 1, 1, 4, 4}, {1, 1, 1},
                        {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
                convtranspose_params_t {"NXC", "XOI", {1, 1, 2, 2, 1},
                        {1, 3, 3, 1, 1}, false, {1}, {1, 1, 4, 4, 1}, {1, 1, 1},
                        {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
                convtranspose_params_t {"NCX", "XOI", {1, 1, 1, 2, 2},
                        {1, 3, 3, 1, 1}, true, {1}, {1, 1, 1, 4, 4}, {1, 1, 1},
                        {1, 1, 1}, {0, 0, 0}, {0, 0, 0}}));

TEST_P(convtranspose_backprop_data_t, TestConvTransposeBackwardData) {
    TestConvTransposeBackwardData();
}

INSTANTIATE_TEST_SUITE_P(Execute, convtranspose_backprop_data_t,
        ::testing::Values(
                // NCX, IOX
                convtranspose_bwd_params_t {{1, 1, 2, 2}, {1, 1, 3, 3},
                        {1, 1, 4, 4}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NCX",
                        "IOX"},
                // NCX, IOX
                convtranspose_bwd_params_t {{1, 1, 2, 2}, {1, 6, 3, 3},
                        {1, 6, 4, 4}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NCX",
                        "IOX"},
                // 3d, NCX, IOX
                convtranspose_bwd_params_t {{1, 1, 1, 2, 2}, {1, 1, 1, 3, 3},
                        {1, 1, 1, 4, 4}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                        {1, 1, 1}, "NCX", "IOX"},
                // NCX, XOI
                convtranspose_bwd_params_t {{1, 1, 2, 2}, {3, 3, 1, 1},
                        {1, 1, 4, 4}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NCX",
                        "XOI"},
                // 3d, NCX, XOI
                convtranspose_bwd_params_t {{1, 1, 1, 2, 2}, {1, 3, 3, 1, 1},
                        {1, 1, 1, 4, 4}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                        {1, 1, 1}, "NCX", "XOI"},
                // 3d, NCX, XOI
                convtranspose_bwd_params_t {{1, 1, 1, 2, 2}, {1, 3, 3, 2, 1},
                        {1, 2, 1, 4, 4}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                        {1, 1, 1}, "NCX", "XOI"},
                // NXC, XOI
                convtranspose_bwd_params_t {{1, 1, 1, 1}, {3, 4, 1, 1},
                        {1, 3, 4, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NXC",
                        "XOI"},
                // NXC, XOI
                convtranspose_bwd_params_t {{1, 1, 1, 1}, {3, 4, 2, 1},
                        {1, 3, 4, 2}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NXC",
                        "XOI"},
                // 3d, NXC, XOI
                convtranspose_bwd_params_t {{1, 1, 1, 1, 1}, {1, 3, 4, 1, 1},
                        {1, 1, 3, 4, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                        {1, 1, 1}, "NXC", "XOI"},
                // NXC, IOX
                convtranspose_bwd_params_t {{1, 2, 2, 1}, {1, 1, 3, 3},
                        {1, 4, 4, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NXC",
                        "IOX"},
                // NXC, IOX
                convtranspose_bwd_params_t {{1, 2, 2, 1}, {1, 2, 3, 3},
                        {1, 4, 4, 2}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NXC",
                        "IOX"},
                // 3d, NXC, IOX
                convtranspose_bwd_params_t {{1, 1, 2, 2, 1}, {1, 1, 1, 3, 3},
                        {1, 1, 4, 4, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                        {1, 1, 1}, "NXC", "IOX"}));

TEST_P(convtranspose_backprop_filters_t, TestConvTransposeBackwardWeights) {
    TestConvTransposeBackwardWeights();
}

INSTANTIATE_TEST_SUITE_P(Execute, convtranspose_backprop_filters_t,
        ::testing::Values(convtranspose_bwd_params_t {{1, 1, 2, 2},
                                  {1, 1, 3, 3}, {1, 1, 4, 4}, {1, 1}, {0, 0},
                                  {0, 0}, {1, 1}, "NCX", "IOX"},
                convtranspose_bwd_params_t {{1, 1, 2, 2}, {1, 2, 3, 3},
                        {1, 2, 4, 4}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NCX",
                        "IOX"},
                convtranspose_bwd_params_t {{1, 1, 1, 2, 2}, {1, 1, 1, 3, 3},
                        {1, 1, 1, 4, 4}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                        {1, 1, 1}, "NCX", "IOX"},
                convtranspose_bwd_params_t {{1, 2, 2, 1}, {3, 3, 1, 1},
                        {1, 4, 4, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NXC",
                        "XOI"},
                convtranspose_bwd_params_t {{1, 1, 2, 2, 1}, {1, 3, 3, 1, 1},
                        {1, 1, 4, 4, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                        {1, 1, 1}, "NXC", "XOI"},
                convtranspose_bwd_params_t {{1, 1, 2, 2, 1}, {1, 3, 3, 2, 1},
                        {1, 1, 4, 4, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                        {1, 1, 1}, "NXC", "XOI"}));

TEST(Compile, ConvTransposeBackwardWeightsWithGroupsAndFiltersAnyLayout) {
    using dims = graph::dnnl_impl::dims;

    const dims src_dims {2, 4, 2};
    const dims diff_dst_dims {2, 4, 2};
    const dims diff_wei_dims {4, 2, 3};

    const dims strides {1};
    const dims pads_begin {1};
    const dims pads_end {1};
    const dims dilations {1};
    const int64_t groups {2};
    const std::string auto_pad {"None"};
    const std::string data_format {"NCX"};
    const std::string filter_format {"IOX"};

    graph::op_t deconv_op(graph::op_kind::ConvTransposeBackwardWeights);
    deconv_op.set_attr(graph::op_attr::strides, strides)
            .set_attr(graph::op_attr::pads_begin, pads_begin)
            .set_attr(graph::op_attr::pads_end, pads_end)
            .set_attr(graph::op_attr::dilations, dilations)
            .set_attr(graph::op_attr::groups, groups)
            .set_attr(graph::op_attr::auto_pad, auto_pad)
            .set_attr(graph::op_attr::data_format, data_format)
            .set_attr(graph::op_attr::weights_format, filter_format);

    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, src_dims, graph::data_type::f32);
    graph::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
            1, diff_dst_dims, graph::data_type::f32);
    graph::logical_tensor_t diff_wei_lt = utils::logical_tensor_init(
            2, diff_wei_dims, graph::data_type::f32, graph::layout_type::any);

    deconv_op.add_input(src_lt);
    deconv_op.add_input(diff_dst_lt);
    deconv_op.add_output(diff_wei_lt);

    graph::engine_t *eng = get_engine();
    graph::graph_t g(eng->kind());
    g.add_op(&deconv_op);
    g.finalize();

    graph::pass::pass_base_ptr apass
            = get_pass("convtranspose_filter_bwd_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&diff_wei_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(diff_wei_lt.id, &lt);
    // if layout queried from the primitive will make descriptor impossible
    // to reshape (with groups -> no groups), we make it strided (via reorder)
    ASSERT_TRUE(lt.layout_type == graph::layout_type::opaque
            || lt.layout_type == graph::layout_type::strided);
}

TEST_P(test_convtranspose_add_compile_t, TestConvTransposeAddCompile) {
    TestConvTransposeAdd();
}

INSTANTIATE_TEST_SUITE_P(TestConvTransposeAddCompile,
        test_convtranspose_add_compile_t,
        ::testing::Values(
                // with broadcast add, no swap inputs, without bias
                convtranspose_add_params_t {{1, 1, 1, 1}, false, false},
                // with broadcast add, swap inputs, without bias
                convtranspose_add_params_t {{1, 1, 1, 1}, true, false},
                // no broadcast add (sum), no swap inputs, without bias
                convtranspose_add_params_t {{1, 4, 4, 1}, false, false},
                // no broadcast add (sum), swap inputs, without bias
                convtranspose_add_params_t {{1, 4, 4, 1}, true, false},
                // with broadcast add, no swap inputs, with bias
                convtranspose_add_params_t {{1, 1, 1, 1}, false, true},
                // with broadcast add, swap inputs, with bias
                convtranspose_add_params_t {{1, 1, 1, 1}, true, true},
                // no broadcast add (sum), no swap inputs, with bias
                convtranspose_add_params_t {{1, 4, 4, 1}, false, true},
                // no broadcast add (sum), swap inputs, with bias
                convtranspose_add_params_t {{1, 4, 4, 1}, true, true}));

TEST(operator_kernel, convtranspose_relu) {
    using dims = graph::dnnl_impl::dims;

    std::vector<bool> with_biases = {false, true};

    for (auto with_bias : with_biases) {
        graph::engine_t *eng = get_engine();
        test::vector<float> src_data {-1.0, 2.5, 5.0, 1.5};
        test::vector<float> weight_data {
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
        test::vector<float> bias_data = {2.0};
        test::vector<float> ref_dst_data = with_bias
                ? test::vector<float> {1.0, 4.5, 1.0, 4.5, 7.0, 2.5, 9.5, 3.5,
                        1.0, 9.5, 2.5, 4.5, 7.0, 3.5, 7.0, 3.5}
                : test::vector<float> {0.0, 2.5, 0.0, 2.5, 5.0, 0.5, 7.5, 1.5,
                        0.0, 7.5, 0.5, 2.5, 5.0, 1.5, 5.0, 1.5};
        test::vector<float> dst_data(ref_dst_data.size(), 0.0);

        graph::op_t convtranspose_op(
                0, graph::op_kind::ConvTranspose, "ConvTranspose");
        convtranspose_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
        convtranspose_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
        convtranspose_op.set_attr<dims>(
                graph::op_attr::pads_begin, dims {0, 0});
        convtranspose_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
        convtranspose_op.set_attr<int64_t>(graph::op_attr::groups, 1);
        convtranspose_op.set_attr<std::string>(
                graph::op_attr::data_format, "NXC");
        convtranspose_op.set_attr<std::string>(
                graph::op_attr::weights_format, "XOI");
        graph::op_t relu_op(1, graph::op_kind::ReLU, "ReLU");

        // prepare logical tensor
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 2, 2, 1}, graph::data_type::f32);
        graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {3, 3, 1, 1}, graph::data_type::f32);
        graph::logical_tensor_t bias_lt
                = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(4,
                {1, 4, 4, 1}, graph::data_type::f32, graph::layout_type::any);
        graph::logical_tensor_t relu_dst_lt = utils::logical_tensor_init(
                5, {1, 4, 4, 1}, graph::data_type::f32);

        convtranspose_op.add_input(src_lt);
        convtranspose_op.add_input(weight_lt);
        if (with_bias) { convtranspose_op.add_input(bias_lt); }
        convtranspose_op.add_output(dst_lt);

        relu_op.add_input(dst_lt);
        relu_op.add_output(relu_dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&convtranspose_op);
        g.add_op(&relu_op);
        g.finalize();

        graph::pass::pass_base_ptr apass
                = get_pass("convtranspose_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);
        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src_lt, &weight_lt};
        if (with_bias) { inputs.push_back(&bias_lt); }
        std::vector<const graph::logical_tensor_t *> outputs {&relu_dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);
        graph::logical_tensor_t lt;
        cp.query_logical_tensor(relu_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(src_lt, eng, src_data.data());
        graph::tensor_t weight_ts(weight_lt, eng, weight_data.data());
        graph::tensor_t bias_ts;
        if (with_bias)
            bias_ts = graph::tensor_t(bias_lt, eng, bias_data.data());
        graph::tensor_t relu_dst_ts(relu_dst_lt, eng, dst_data.data());

        graph::stream_t *strm = get_stream();
        if (with_bias)
            cp.execute(strm, {src_ts, weight_ts, bias_ts}, {relu_dst_ts});
        else
            cp.execute(strm, {src_ts, weight_ts}, {relu_dst_ts});
        strm->wait();

        for (size_t i = 0; i < dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
        }
    }
}

TEST(operator_kernel, convtranspose_swish) {
    using dims = graph::dnnl_impl::dims;

    std::vector<bool> with_biases = {false, true};

    for (auto with_bias : with_biases) {
        graph::engine_t *eng = get_engine();
        test::vector<float> src_data {-1.0, 2.5, 5.0, 1.5};
        test::vector<float> weight_data {
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
        test::vector<float> bias_data = {2.0};
        test::vector<float> dst_data(16, 0.0);

        graph::op_t convtranspose_op(
                0, graph::op_kind::ConvTranspose, "ConvTranspose");
        convtranspose_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
        convtranspose_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
        convtranspose_op.set_attr<dims>(
                graph::op_attr::pads_begin, dims {0, 0});
        convtranspose_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
        convtranspose_op.set_attr<int64_t>(graph::op_attr::groups, 1);
        convtranspose_op.set_attr<std::string>(
                graph::op_attr::data_format, "NXC");
        convtranspose_op.set_attr<std::string>(
                graph::op_attr::weights_format, "XOI");
        graph::op_t sigmoid_op(1, graph::op_kind::Sigmoid, "Sigmoid");
        graph::op_t multiply_op(2, graph::op_kind::Multiply, "Multiply");

        // prepare logical tensor
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 2, 2, 1}, graph::data_type::f32);
        graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {3, 3, 1, 1}, graph::data_type::f32);
        graph::logical_tensor_t bias_lt
                = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(3,
                {1, 4, 4, 1}, graph::data_type::f32, graph::layout_type::any);
        graph::logical_tensor_t sigmoid_dst_lt = utils::logical_tensor_init(
                4, {1, 4, 4, 1}, graph::data_type::f32);
        graph::logical_tensor_t multiply_dst_lt = utils::logical_tensor_init(
                5, {1, 4, 4, 1}, graph::data_type::f32);

        convtranspose_op.add_input(src_lt);
        convtranspose_op.add_input(weight_lt);
        if (with_bias) { convtranspose_op.add_input(bias_lt); }
        convtranspose_op.add_output(dst_lt);

        sigmoid_op.add_input(dst_lt);
        sigmoid_op.add_output(sigmoid_dst_lt);
        multiply_op.add_input(dst_lt);
        multiply_op.add_input(sigmoid_dst_lt);
        multiply_op.add_output(multiply_dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&convtranspose_op);
        g.add_op(&sigmoid_op);
        g.add_op(&multiply_op);
        g.finalize();

        graph::pass::pass_base_ptr apass
                = get_pass("convtranspose_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);
        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src_lt, &weight_lt};
        if (with_bias) { inputs.push_back(&bias_lt); }
        std::vector<const graph::logical_tensor_t *> outputs {&multiply_dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);
        graph::logical_tensor_t lt;
        cp.query_logical_tensor(multiply_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(src_lt, eng, src_data.data());
        graph::tensor_t weight_ts(weight_lt, eng, weight_data.data());
        graph::tensor_t bias_ts;
        if (with_bias)
            bias_ts = graph::tensor_t(bias_lt, eng, bias_data.data());
        graph::tensor_t multiply_dst_ts(multiply_dst_lt, eng, dst_data.data());

        graph::stream_t *strm = get_stream();
        if (with_bias)
            cp.execute(strm, {src_ts, weight_ts, bias_ts}, {multiply_dst_ts});
        else
            cp.execute(strm, {src_ts, weight_ts}, {multiply_dst_ts});
        strm->wait();
    }
}

TEST(ExecuteSubgraphInt8, GroupConvTransposeWeightOC1) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();
    SKIP_IF(engine->kind() == graph::engine_kind::gpu,
            "Skip group convtranspose on GPU");

    size_t nd = 2;
    int64_t g = 17;
    std::string wei_qtype = "per_channel";
    std::string src_qtype = "symmetric";

    static auto isa = dnnl_get_effective_cpu_isa();

    // prepare data
    int64_t in_channel = 17, out_channel = 17;
    int64_t kernel_size = 3;
    std::vector<int64_t> src_shape = {2, in_channel, 5, 5};
    std::vector<int64_t> weight_shape
            = {in_channel, out_channel / g, kernel_size, kernel_size};
    std::vector<int64_t> dst_shape = {2, out_channel, 7, 7};

    test::vector<uint8_t> src_u8_data(product(src_shape));
    test::vector<int8_t> weight_s8_data(product(weight_shape));
    test::vector<int8_t> case1_out_data(product(dst_shape));
    test::vector<int8_t> case2_out_data(product(dst_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
    std::generate(src_u8_data.begin(), src_u8_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_s8_data.begin(), weight_s8_data.end(),
            [&]() { return static_cast<int8_t>(s8_distribution(generator)); });

    float scale_src = 1 / 255.f; // map to 0~255
    float scale_out = 1.f;
    int64_t zp_src = 0;
    int64_t zp_out = 0;

    std::vector<float> scale_wei(1, 1 / 127.f);
    std::vector<int64_t> zp_wei(1, 0);

    graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
    SET_Q_DQ_DATA_ATTR(dqdata_node)

    graph::op_t dqweight_node(3, graph::op_kind::Dequantize, "dqweight_node");
    SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 1)

    graph::op_t convtranspose_node(
            4, graph::op_kind::ConvTranspose, "convtranspose_node");
    SET_CONVTRANSPOSE_ATTR(convtranspose_node, nd)

    graph::op_t qout_node(5, graph::op_kind::Quantize, "qout_node");
    SET_Q_DQ_OUT_ATTR(qout_node)

    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(4, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_s8
            = utils::logical_tensor_init(8, dst_shape, graph::data_type::s8);

    dqdata_node.add_input(src_u8);
    dqdata_node.add_output(src_f32_dq);

    dqweight_node.add_input(weight_s8);
    dqweight_node.add_output(weight_f32_dq);

    convtranspose_node.add_input(src_f32_dq);
    convtranspose_node.add_input(weight_f32_dq);
    convtranspose_node.add_output(dst_f32);

    qout_node.add_input(dst_f32);
    qout_node.add_output(dst_s8);

    graph::graph_t agraph(engine->kind());
    agraph.add_op(&dqdata_node);
    agraph.add_op(&dqweight_node);
    agraph.add_op(&convtranspose_node);
    agraph.add_op(&qout_node);
    agraph.finalize();

    graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
    graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
    graph::tensor_t dst_s8_ts(dst_s8, engine, case1_out_data.data());
    graph::tensor_t dst_s8_case2_ts(dst_s8, engine, case2_out_data.data());

    // -------------------------case 1----------------------------------
    ASSERT_EQ(run_graph(agraph, {src_u8_ts, weight_s8_ts}, {dst_s8_ts}, *engine,
                      *strm),
            graph::status::success);

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass
            = get_pass(engine->kind() == graph::engine_kind::gpu
                            ? "int8_convtranspose_post_ops_fusion_gpu"
                            : "int8_convtranspose_post_ops_fusion_cpu");
    ASSERT_TRUE(apass != nullptr);
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    auto part = agraph.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {&src_u8, &weight_s8};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    cp.execute(strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
    strm->wait();

    if (engine->kind() == graph::engine_kind::cpu
            && isa < dnnl_cpu_isa_avx512_core_vnni)
        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    else
        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                /*atol*/ 1.f));
}

TEST(ExecuteSubgraphInt8, GroupConvTransposePerTensorScale) {

    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(engine->kind() == graph::engine_kind::gpu,
            "Skip group convtranspose on GPU");
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core_vnni
                    && engine->kind() == graph::engine_kind::cpu,
            "Skip u8+s8 examples for systems that do not support "
            "avx512_core_vnni.");

    size_t nd = 2;
    int64_t g = 4;
    std::string wei_qtype = "per_tensor";
    std::string src_qtype = "symmetric";

    // prepare data
    int64_t in_channel = 16, out_channel = 20;
    int64_t kernel_size = 3;
    std::vector<int64_t> src_shape = {2, in_channel, 5, 5};
    std::vector<int64_t> weight_shape
            = {in_channel, out_channel / g, kernel_size, kernel_size};
    std::vector<int64_t> dst_shape = {2, out_channel, 7, 7};

    test::vector<uint8_t> src_s8_data(product(src_shape));
    test::vector<int8_t> weight_s8_data(product(weight_shape));
    test::vector<int8_t> case1_out_data(product(dst_shape));
    test::vector<int8_t> case2_out_data(product(dst_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
    std::generate(src_s8_data.begin(), src_s8_data.end(),
            [&]() { return static_cast<int8_t>(s8_distribution(generator)); });
    std::generate(weight_s8_data.begin(), weight_s8_data.end(),
            [&]() { return static_cast<int8_t>(s8_distribution(generator)); });

    float scale_src = 1.f;
    float scale_out = 1.f;
    int64_t zp_src = 0;
    int64_t zp_out = 0;

    std::vector<float> scale_wei(1, 1 / 127.f);
    std::vector<int64_t> zp_wei(1, 0);

    graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
    SET_Q_DQ_DATA_ATTR(dqdata_node)

    graph::op_t dqweight_node(3, graph::op_kind::Dequantize, "dqweight_node");
    SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 1)

    graph::op_t convtranspose_node(
            4, graph::op_kind::ConvTranspose, "convtranspose_node");
    SET_CONVTRANSPOSE_ATTR(convtranspose_node, nd)

    graph::op_t qout_node(5, graph::op_kind::Quantize, "qout_node");
    SET_Q_DQ_OUT_ATTR(qout_node)

    graph::logical_tensor_t src_s8
            = utils::logical_tensor_init(1, src_shape, graph::data_type::s8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(4, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_s8
            = utils::logical_tensor_init(8, dst_shape, graph::data_type::s8);

    dqdata_node.add_input(src_s8);
    dqdata_node.add_output(src_f32_dq);

    dqweight_node.add_input(weight_s8);
    dqweight_node.add_output(weight_f32_dq);

    convtranspose_node.add_input(src_f32_dq);
    convtranspose_node.add_input(weight_f32_dq);
    convtranspose_node.add_output(dst_f32);

    qout_node.add_input(dst_f32);
    qout_node.add_output(dst_s8);

    graph::graph_t agraph(engine->kind());
    agraph.add_op(&dqdata_node);
    agraph.add_op(&dqweight_node);
    agraph.add_op(&convtranspose_node);
    agraph.add_op(&qout_node);
    agraph.finalize();

    graph::tensor_t src_s8_ts(src_s8, engine, src_s8_data.data());
    graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
    graph::tensor_t dst_s8_case1_ts(dst_s8, engine, case1_out_data.data());
    graph::tensor_t dst_s8_case2_ts(dst_s8, engine, case2_out_data.data());

    // -------------------------case 1----------------------------------
    ASSERT_EQ(run_graph(agraph, {src_s8_ts, weight_s8_ts}, {dst_s8_case1_ts},
                      *engine, *strm),
            graph::status::success);

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass
            = get_pass("int8_convtranspose_post_ops_fusion_cpu");
    ASSERT_TRUE(apass != nullptr);
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    auto part = agraph.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);
    std::vector<const graph::logical_tensor_t *> lt_ins {&src_s8, &weight_s8};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};
    p.compile(&cp, lt_ins, lt_outs, engine);
    cp.execute(strm, {src_s8_ts, weight_s8_ts}, {dst_s8_case2_ts});
    strm->wait();

    ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
            /*atol*/ 1.f));
}

TEST(ExecuteSubgraphInt8, ConvTranspose1d2d3d) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};

    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for_(const auto &src_qtype : src_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (((isa < dnnl_cpu_isa_avx512_core_vnni
                     && engine->kind() == graph::engine_kind::cpu)
                    || engine->kind() == graph::engine_kind::gpu)
                && src_qtype == "asymmetric")
            continue;

        if (engine->kind() == graph::engine_kind::gpu && g == 4) continue;

        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {in_channel, out_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {in_channel, out_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {in_channel, out_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 14}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 14, 14}
                          : std::vector<int64_t> {1, out_channel, 14, 14, 14};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        // The following cmd will be skiped by benchdnn, since oneDNN didn't
        // support int deconv with oscales: "./tests/benchdnn/benchdnn --deconv
        // --engine=gpu --mode=c --api=P --dir=FWD_B --cfg=u8s8s8 --stag=acdb
        // --wtag=ABcd2b8a4b --dtag=acdb --attr-oscale=common:0.000031
        // mb1_ic8oc8_ih12oh14kh3sh1dh0ph0_iw12ow14kw3sw1dw0pw0"
        float scale_src = engine->kind() == graph::engine_kind::gpu
                ? 1.f
                : 1 / 255.f; // map to 0~255
        float scale_out = 1.f;
        int64_t zp_src = src_qtype == "symmetric" ? 0 : 128;
        // The following cmd will be skiped by benchdnn, since oneDNN didn't
        // support reorder with zps on GPU: "./tests/benchdnn/benchdnn --reorder
        // --engine=gpu --mode=C --sdt=f32 --ddt=s8
        // --attr-zero-points=dst:common:78 --stag=aBc8b --dtag=abc 1x8x10"
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : (out_channel / g);
        std::vector<float> scale_wei(scale_size,
                engine->kind() == graph::engine_kind::gpu ? 1.f : 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                3, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 1)

        graph::op_t convtranspose_node(
                4, graph::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONVTRANSPOSE_ATTR(convtranspose_node, nd)

        graph::op_t qout_node(5, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, graph::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        if (with_bias) convtranspose_node.add_input(bias_f32);
        convtranspose_node.add_output(dst_f32);

        qout_node.add_input(dst_f32);
        qout_node.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&convtranspose_node);
        g.add_op(&qout_node);
        g.finalize();

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }
        graph::tensor_t dst_s8_ts(dst_s8, engine, case1_out_data.data());
        graph::tensor_t dst_s8_case2_ts(dst_s8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "int8_convtranspose_post_ops_fusion_gpu"
                                : "int8_convtranspose_post_ops_fusion_cpu");
        ASSERT_TRUE(apass != nullptr);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        if (with_bias)
            cp.execute(strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, ConvTranspose2dEltwise) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // some cases with Exp post-ops can't pass correctness check on GPU
    SKIP_IF(engine->kind() == graph::engine_kind::gpu, "skip on gpu");

    const std::vector<graph::op_kind_t> eltwise_kinds = {
            graph::op_kind::Abs,
            graph::op_kind::Elu,
            graph::op_kind::Exp,
            graph::op_kind::GELU,
            graph::op_kind::Clamp,
            graph::op_kind::HardSigmoid,
            graph::op_kind::HardSwish,
            graph::op_kind::Log,
            graph::op_kind::ReLU,
            graph::op_kind::Round,
            graph::op_kind::Sigmoid,
            graph::op_kind::Tanh,
    };

    std::vector<size_t> nds = {2};
    std::vector<int64_t> groups = {1};
    std::vector<bool> with_biases = {true};
    std::vector<std::string> weight_qtypes = {"per_channel"};
    std::vector<std::string> src_qtypes = {"asymmetric"};

    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &eltwise_kind : eltwise_kinds)
    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for_(const auto &src_qtype : src_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (((isa < dnnl_cpu_isa_avx512_core_vnni
                     && engine->kind() == graph::engine_kind::cpu)
                    || engine->kind() == graph::engine_kind::gpu)
                && src_qtype == "asymmetric")
            continue;

        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {in_channel, out_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {in_channel, out_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {in_channel, out_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 14}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 14, 14}
                          : std::vector<int64_t> {1, out_channel, 14, 14, 14};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(1.0f, 25.0f);
        std::uniform_real_distribution<float> s8_distribution(-1.0f, 25.0f);
        std::uniform_real_distribution<float> f32_distribution(1.0f, 25.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        // The following cmd will be skiped by benchdnn, since oneDNN didn't
        // support int deconv with oscales: "./tests/benchdnn/benchdnn --deconv
        // --engine=gpu --mode=c --api=P --dir=FWD_B --cfg=u8s8s8 --stag=acdb
        // --wtag=ABcd2b8a4b --dtag=acdb --attr-oscale=common:0.000031
        // mb1_ic8oc8_ih12oh14kh3sh1dh0ph0_iw12ow14kw3sw1dw0pw0"
        float scale_src = engine->kind() == graph::engine_kind::gpu
                ? 1.f
                : 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = src_qtype == "symmetric" ? 0 : -4;
        // The following cmd will be skiped by benchdnn, since oneDNN didn't
        // support reorder with zps on GPU: "./tests/benchdnn/benchdnn --reorder
        // --engine=gpu --mode=C --sdt=f32 --ddt=s8
        // --attr-zero-points=dst:common:78 --stag=aBc8b --dtag=abc 1x8x10"
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : (out_channel / g);
        std::vector<float> scale_wei(scale_size,
                engine->kind() == graph::engine_kind::gpu ? 1.f : 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(0, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                1, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 1)

        graph::op_t convtranspose_node(
                2, graph::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONVTRANSPOSE_ATTR(convtranspose_node, nd)

        graph::op_t eltwise_node(3, eltwise_kind, "eltwise_node");

        switch (eltwise_kind) {
            case graph::op_kind::Elu:
                eltwise_node.set_attr<float>(graph::op_attr::alpha, 1.f);
                break;
            case graph::op_kind::Clamp:
                eltwise_node.set_attr<float>(graph::op_attr::min, -1.f);
                eltwise_node.set_attr<float>(graph::op_attr::max, 2.f);
                break;
            case graph::op_kind::HardSigmoid:
                eltwise_node.set_attr<float>(graph::op_attr::alpha, 1.0f / 6);
                eltwise_node.set_attr<float>(graph::op_attr::beta, 0.5f);
                break;
            default: break;
        }

        graph::op_t qout_node(4, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                0, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                1, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                2, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                5, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_eltwise_f32 = utils::logical_tensor_init(
                6, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    4, bias_shape, graph::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        if (with_bias) convtranspose_node.add_input(bias_f32);
        convtranspose_node.add_output(dst_f32);

        eltwise_node.add_input(dst_f32);
        eltwise_node.add_output(dst_eltwise_f32);

        qout_node.add_input(dst_eltwise_f32);
        qout_node.add_output(dst_s8);

        graph::graph_t graph(engine->kind());
        graph.add_op(&dqdata_node);
        graph.add_op(&dqweight_node);
        graph.add_op(&convtranspose_node);
        graph.add_op(&eltwise_node);
        graph.add_op(&qout_node);
        graph.finalize();

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }
        graph::tensor_t dst_s8_ts(dst_s8, engine, case1_out_data.data());
        graph::tensor_t dst_s8_case2_ts(dst_s8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(graph, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass("int8_convtranspose_post_ops_fusion_cpu");
        ASSERT_TRUE(apass != nullptr);
        apass->run(graph);
        ASSERT_EQ(graph.get_num_partitions(), 1U);
        auto part = graph.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        if (with_bias)
            cp.execute(strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, X8X8F32ConvTranspose1d2d3dEltwise) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // some cases with Exp post-ops can't pass correctness check on GPU
    SKIP_IF(engine->kind() == graph::engine_kind::gpu, "skip on gpu");

    const std::vector<graph::op_kind_t> eltwise_kinds = {
            graph::op_kind::Abs,
            graph::op_kind::Clamp,
            graph::op_kind::Elu,
            graph::op_kind::Exp,
            graph::op_kind::GELU,
            graph::op_kind::HardSwish,
            graph::op_kind::Log,
            graph::op_kind::ReLU,
            graph::op_kind::Round,
            graph::op_kind::Sigmoid,
            graph::op_kind::Tanh,
    };

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1};
    std::vector<bool> with_biases = {true};
    std::vector<std::string> weight_qtypes = {"per_channel"};
    std::vector<std::string> src_qtypes = {"asymmetric"};

    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &eltwise_kind : eltwise_kinds)
    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for_(const auto &src_qtype : src_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (((isa < dnnl_cpu_isa_avx512_core_vnni
                     && engine->kind() == graph::engine_kind::cpu)
                    || engine->kind() == graph::engine_kind::gpu)
                && src_qtype == "asymmetric")
            continue;

        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {in_channel, out_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {in_channel, out_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {in_channel, out_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 14}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 14, 14}
                          : std::vector<int64_t> {1, out_channel, 14, 14, 14};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<float> case1_out_data(product(dst_shape));
        test::vector<float> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(1.0f, 25.0f);
        std::uniform_real_distribution<float> s8_distribution(-1.0f, 25.0f);
        std::uniform_real_distribution<float> f32_distribution(1.0f, 25.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = engine->kind() == graph::engine_kind::gpu
                ? 1.f
                : 1 / 255.f; // map to 0~255
        int64_t zp_src = src_qtype == "symmetric" ? 0 : -4;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : (out_channel / g);
        std::vector<float> scale_wei(scale_size,
                engine->kind() == graph::engine_kind::gpu ? 1.f : 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(0, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                1, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 1)

        graph::op_t convtranspose_node(
                2, graph::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONVTRANSPOSE_ATTR(convtranspose_node, nd)

        graph::op_t eltwise_node(3, eltwise_kind, "eltwise_node");

        if (eltwise_kind == graph::op_kind::Elu) {
            eltwise_node.set_attr<float>(graph::op_attr::alpha, 1.f);
        } else if (eltwise_kind == graph::op_kind::Clamp) {
            eltwise_node.set_attr<float>(graph::op_attr::min, -1.f);
            eltwise_node.set_attr<float>(graph::op_attr::max, 2.f);
        }

        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                0, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                1, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                2, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                5, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_eltwise_f32 = utils::logical_tensor_init(
                6, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    4, bias_shape, graph::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        if (with_bias) convtranspose_node.add_input(bias_f32);
        convtranspose_node.add_output(dst_f32);

        eltwise_node.add_input(dst_f32);
        eltwise_node.add_output(dst_eltwise_f32);

        graph::graph_t graph(engine->kind());
        graph.add_op(&dqdata_node);
        graph.add_op(&dqweight_node);
        graph.add_op(&convtranspose_node);
        graph.add_op(&eltwise_node);
        // graph.add_op(&qout_node);
        graph.finalize();

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }
        graph::tensor_t dst_f32_ts(
                dst_eltwise_f32, engine, case1_out_data.data());
        graph::tensor_t dst_f32_case2_ts(
                dst_eltwise_f32, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(graph, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_f32_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass("int8_convtranspose_post_ops_fusion_cpu");
        ASSERT_TRUE(apass != nullptr);
        apass->run(graph);
        ASSERT_EQ(graph.get_num_partitions(), 1U);
        auto part = graph.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_eltwise_f32};

        p.compile(&cp, lt_ins, lt_outs, engine);

        if (with_bias)
            cp.execute(strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_f32_case2_ts});
        else
            cp.execute(strm, {src_u8_ts, weight_s8_ts}, {dst_f32_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, X8X8F32ConvTransposeSwish) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};

    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &src_qtype : src_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (((isa < dnnl_cpu_isa_avx512_core_vnni
                     && engine->kind() == graph::engine_kind::cpu)
                    || engine->kind() == graph::engine_kind::gpu)
                && src_qtype == "asymmetric")
            continue;

        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3, g = 1;
        std::vector<int64_t> src_shape
                = std::vector<int64_t> {1, in_channel, 12, 12};
        std::vector<int64_t> weight_shape = std::vector<int64_t> {
                in_channel, out_channel, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape
                = std::vector<int64_t> {1, out_channel, 14, 14};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<float> case1_out_data(product(dst_shape));
        test::vector<float> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(1.0f, 25.0f);
        std::uniform_real_distribution<float> s8_distribution(-1.0f, 25.0f);
        std::uniform_real_distribution<float> f32_distribution(1.0f, 25.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });

        float scale_src = engine->kind() == graph::engine_kind::gpu
                ? 1.f
                : 1 / 255.f; // map to 0~255
        int64_t zp_src = src_qtype == "symmetric" ? 0 : -4;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size,
                engine->kind() == graph::engine_kind::gpu ? 1.f : 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(0, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                1, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 1)

        graph::op_t convtranspose_node(
                2, graph::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONVTRANSPOSE_ATTR(convtranspose_node, 2)

        graph::op_t sigmoid_node(3, graph::op_kind::Sigmoid, "sigmoid_node");
        graph::op_t multiply_node {
                4, graph::op_kind::Multiply, "multiply_node"};

        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                0, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                1, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                2, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                5, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_sigmoid_f32 = utils::logical_tensor_init(
                6, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_multiply_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        convtranspose_node.add_output(dst_f32);

        sigmoid_node.add_input(dst_f32);
        sigmoid_node.add_output(dst_sigmoid_f32);
        multiply_node.add_input(dst_sigmoid_f32);
        multiply_node.add_input(dst_f32);
        multiply_node.add_output(dst_multiply_f32);

        graph::graph_t graph(engine->kind());
        graph.add_op(&dqdata_node);
        graph.add_op(&dqweight_node);
        graph.add_op(&convtranspose_node);
        graph.add_op(&sigmoid_node);
        graph.add_op(&multiply_node);
        graph.finalize();

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t dst_f32_ts(
                dst_multiply_f32, engine, case1_out_data.data());
        graph::tensor_t dst_f32_case2_ts(
                dst_multiply_f32, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(graph, {src_u8_ts, weight_s8_ts}, {dst_f32_ts},
                          *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "int8_convtranspose_post_ops_fusion_gpu"
                                : "int8_convtranspose_post_ops_fusion_cpu");
        ASSERT_TRUE(apass != nullptr);
        apass->run(graph);
        ASSERT_EQ(graph.get_num_partitions(), 1U);
        auto part = graph.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins
                = {&src_u8, &weight_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {
                &dst_multiply_f32};

        p.compile(&cp, lt_ins, lt_outs, engine);
        cp.execute(strm, {src_u8_ts, weight_s8_ts}, {dst_f32_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, ConvTranspose1d2d3dAdd) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1, 4};
    std::vector<float> scales = {1.f, 1 / 127.f};
    std::vector<int64_t> zps = {0, 78};
    std::vector<bool> with_biases = {true};
    // swap add's two inputs
    std::vector<bool> swaps = {true, false};
    std::vector<bool> with_broadcasts = {true, false};
    std::vector<std::string> weight_qtypes = {"per_channel"};
    std::vector<std::string> src_qtypes = {"asymmetric"};
    std::vector<std::string> other_qtypes = {"asymmetric"};

    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &nd : nds)
    for_(const auto &scale : scales)
    for_(const auto &zp : zps)
    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for_(const auto swap : swaps)
    for_(const auto with_broadcast : with_broadcasts)
    for_(const auto &src_qtype : src_qtypes)
    for_(const auto &other_qtype : other_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (((isa < dnnl_cpu_isa_avx512_core_vnni
                     && engine->kind() == graph::engine_kind::cpu)
                    || engine->kind() == graph::engine_kind::gpu)
                && src_qtype == "asymmetric")
            continue;

        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {in_channel, out_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {in_channel, out_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {in_channel, out_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 14}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 14, 14}
                          : std::vector<int64_t> {1, out_channel, 14, 14, 14};
        std::vector<int64_t> other_shape
                = with_broadcast ? std::vector<int64_t> {1, 1, 1} : dst_shape;

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(other_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 25.0f);
        std::uniform_real_distribution<float> s8_distribution(-1.0f, 25.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        // The following cmd will be skiped by benchdnn, since oneDNN didn't
        // support int deconv with oscales: "./tests/benchdnn/benchdnn --deconv
        // --engine=gpu --mode=c --api=P --dir=FWD_B --cfg=u8s8s8 --stag=acdb
        // --wtag=ABcd2b8a4b --dtag=acdb --attr-oscale=common:0.000031
        // mb1_ic8oc8_ih12oh14kh3sh1dh0ph0_iw12ow14kw3sw1dw0pw0"
        float scale_src = engine->kind() == graph::engine_kind::gpu
                ? 1.f
                : scale; // map to 0~255
        float scale_other = scale;
        float scale_out = scale;
        int64_t zp_src = src_qtype == "symmetric"
                        || engine->kind() == graph::engine_kind::gpu
                ? 0
                : zp;
        int64_t zp_other = other_qtype == "symmetric"
                        || engine->kind() == graph::engine_kind::gpu
                ? 0
                : zp;
        // The following cmd will be skiped by benchdnn, since oneDNN didn't
        // support reorder with zps on GPU: "./tests/benchdnn/benchdnn --reorder
        // --engine=gpu --mode=C --sdt=f32 --ddt=s8
        // --attr-zero-points=dst:common:78 --stag=aBc8b --dtag=abc 1x8x10"
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : zp;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : (out_channel / g);
        std::vector<float> scale_wei(scale_size,
                engine->kind() == graph::engine_kind::gpu ? 1.f : scale);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(0, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                1, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 1)

        graph::op_t convtranspose_node(
                2, graph::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONVTRANSPOSE_ATTR(convtranspose_node, nd)

        graph::op_t dqother_node(3, graph::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        dqother_node.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        dqother_node.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t add_node(4, graph::op_kind::Add, "add_node");

        graph::op_t qout_node(5, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        auto src_u8 = utils::logical_tensor_init(
                0, src_shape, graph::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(
                1, src_shape, graph::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(
                2, weight_shape, graph::data_type::s8);
        auto weight_f32_dq = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                5, dst_shape, graph::data_type::f32);
        auto other_s8 = utils::logical_tensor_init(
                6, other_shape, graph::data_type::s8);
        auto other_f32_dq = utils::logical_tensor_init(
                7, other_shape, graph::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::f32);
        auto dst_s8 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    4, bias_shape, graph::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        if (with_bias) convtranspose_node.add_input(bias_f32);
        convtranspose_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);
        if (swap) {
            add_node.add_input(dst_f32);
            add_node.add_input(other_f32_dq);
        } else {
            add_node.add_input(other_f32_dq);
            add_node.add_input(dst_f32);
        }
        add_node.add_output(dst_add_f32);

        qout_node.add_input(dst_add_f32);
        qout_node.add_output(dst_s8);

        graph::graph_t graph(engine->kind());
        graph.add_op(&dqdata_node);
        graph.add_op(&dqweight_node);
        graph.add_op(&convtranspose_node);
        graph.add_op(&dqother_node);
        graph.add_op(&add_node);
        graph.add_op(&qout_node);
        graph.finalize();

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t other_s8_ts(other_s8, engine, other_s8_data.data());
        graph::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = graph::tensor_t(bias_f32, engine, bias_data.data());
        }
        graph::tensor_t dst_s8_ts(dst_s8, engine, case1_out_data.data());
        graph::tensor_t dst_s8_case2_ts(dst_s8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(graph,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "int8_convtranspose_add_post_ops_fusion_gpu"
                                : "int8_convtranspose_add_post_ops_fusion_cpu");
        ASSERT_TRUE(apass != nullptr);
        apass->run(graph);
        ASSERT_EQ(graph.get_num_partitions(), 1U);
        auto part = graph.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_s8, &other_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        if (with_bias)
            cp.execute(strm,
                    {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(strm, {src_u8_ts, weight_s8_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, ConvTranspose1d2d3dBinary) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1};
    std::vector<bool> with_broadcasts = {false, true};
    std::vector<std::string> weight_qtypes = {"per_channel"};
    std::vector<std::string> src_qtypes = {"asymmetric"};
    std::vector<graph::op_kind_t> binary_kinds = {graph::op_kind::Multiply,
            graph::op_kind::Maximum, graph::op_kind::Minimum,
            graph::op_kind::Divide, graph::op_kind::Subtract};

    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto with_broadcast : with_broadcasts)
    for_(const auto &src_qtype : src_qtypes)
    for_(const auto &wei_qtype : weight_qtypes)
    for (const auto &binary_kind : binary_kinds) {
        // gpu engine cannot support non-broadcast binary post-op for deconv
        if (engine->kind() == graph::engine_kind::gpu && !with_broadcast)
            continue;
        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {in_channel, out_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {in_channel, out_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {in_channel, out_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 14}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 14, 14}
                          : std::vector<int64_t> {1, out_channel, 14, 14, 14};
        std::vector<int64_t> other_shape
                = with_broadcast ? std::vector<int64_t> {1, 1, 1} : dst_shape;

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<float> other_f32_data(product(other_shape));
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 25.0f);
        std::uniform_real_distribution<float> s8_distribution(-1.0f, 25.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(other_f32_data.begin(), other_f32_data.end(),
                [&]() { return 1.0f; });

        // The following cmd will be skiped by benchdnn, since oneDNN didn't
        // support int deconv with oscales: "./tests/benchdnn/benchdnn --deconv
        // --engine=gpu --mode=c --api=P --dir=FWD_B --cfg=u8s8s8 --stag=acdb
        // --wtag=ABcd2b8a4b --dtag=acdb --attr-oscale=common:0.000031
        // mb1_ic8oc8_ih12oh14kh3sh1dh0ph0_iw12ow14kw3sw1dw0pw0"
        float scale_src = engine->kind() == graph::engine_kind::gpu
                ? 1.f
                : 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = src_qtype == "symmetric"
                        || engine->kind() == graph::engine_kind::gpu
                ? 0
                : -4;
        // The following cmd will be skiped by benchdnn, since oneDNN didn't
        // support reorder with zps on GPU: "./tests/benchdnn/benchdnn --reorder
        // --engine=gpu --mode=C --sdt=f32 --ddt=s8
        // --attr-zero-points=dst:common:78 --stag=aBc8b --dtag=abc 1x8x10"
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : (out_channel / g);
        std::vector<float> scale_wei(scale_size,
                engine->kind() == graph::engine_kind::gpu ? 1.f : 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(0, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                1, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 1)

        graph::op_t convtranspose_node(
                2, graph::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONVTRANSPOSE_ATTR(convtranspose_node, nd)

        graph::op_t binary_node(3, binary_kind, "binary_node");

        graph::op_t qout_node(4, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        auto src_u8 = utils::logical_tensor_init(
                0, src_shape, graph::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(
                1, src_shape, graph::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(
                2, weight_shape, graph::data_type::s8);
        auto weight_f32_dq = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                5, dst_shape, graph::data_type::f32);
        auto other_f32 = utils::logical_tensor_init(
                6, other_shape, graph::data_type::f32);
        auto dst_binary_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        auto dst_s8 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::s8);

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        convtranspose_node.add_output(dst_f32);

        binary_node.add_input(dst_f32);
        binary_node.add_input(other_f32);
        binary_node.add_output(dst_binary_f32);

        qout_node.add_input(dst_binary_f32);
        qout_node.add_output(dst_s8);

        graph::graph_t graph(engine->kind());
        graph.add_op(&dqdata_node);
        graph.add_op(&dqweight_node);
        graph.add_op(&convtranspose_node);
        graph.add_op(&binary_node);
        graph.add_op(&qout_node);
        graph.finalize();

        graph::tensor_t src_u8_ts(src_u8, engine, src_u8_data.data());
        graph::tensor_t weight_s8_ts(weight_s8, engine, weight_s8_data.data());
        graph::tensor_t other_f32_ts(other_f32, engine, other_f32_data.data());
        graph::tensor_t dst_s8_ts(dst_s8, engine, case1_out_data.data());
        graph::tensor_t dst_s8_case2_ts(dst_s8, engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(graph, {src_u8_ts, weight_s8_ts, other_f32_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "int8_convtranspose_post_ops_fusion_gpu"
                                : "int8_convtranspose_post_ops_fusion_cpu");
        ASSERT_TRUE(apass != nullptr);
        apass->run(graph);
        ASSERT_EQ(graph.get_num_partitions(), 1U);
        ASSERT_EQ(graph.get_partitions()[0]->get_ops().size(), 5U);

        auto part = graph.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &other_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);
        cp.execute(strm, {src_u8_ts, weight_s8_ts, other_f32_ts},
                {dst_s8_case2_ts});
        strm->wait();

        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni) {
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        } else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, ConvTranspose2dAddGetInplacePair) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();

    std::vector<size_t> nds = {2};
    std::vector<int64_t> groups = {1};
    std::vector<std::string> weight_qtypes = {"per_channel"};
    std::vector<std::string> src_qtypes = {"asymmetric"};
    std::vector<std::string> other_qtypes = {"asymmetric"};

    // the two deconvs has different optimal layout on GPU, so can't construct a
    // inplaced pattern
    SKIP_IF(engine->kind() == graph::engine_kind::gpu, "skip on gpu");

    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto &src_qtype : src_qtypes)
    for_(const auto &other_qtype : other_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (((isa < dnnl_cpu_isa_avx512_core_vnni
                     && engine->kind() == graph::engine_kind::cpu)
                    || engine->kind() == graph::engine_kind::gpu)
                && src_qtype == "asymmetric")
            continue;

        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {in_channel, out_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {in_channel, out_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {in_channel, out_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 14}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 14, 14}
                          : std::vector<int64_t> {1, out_channel, 14, 14, 14};
        std::vector<int64_t> other_shape = dst_shape;

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(other_shape));
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });

        float scale_src = engine->kind() == graph::engine_kind::gpu
                ? 1.f
                : 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = src_qtype == "symmetric"
                        || engine->kind() == graph::engine_kind::gpu
                ? 0
                : 128;
        int64_t zp_other = other_qtype == "symmetric"
                        || engine->kind() == graph::engine_kind::gpu
                ? 0
                : 128;
        // The following cmd will be skiped by benchdnn, since oneDNN didn't
        // support reorder with zps on GPU: "./tests/benchdnn/benchdnn --reorder
        // --engine=gpu --mode=C --sdt=f32 --ddt=s8
        // --attr-zero-points=dst:common:78 --stag=aBc8b --dtag=abc 1x8x10"
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : (out_channel / g);
        std::vector<float> scale_wei(scale_size,
                engine->kind() == graph::engine_kind::gpu ? 1.f : 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        graph::op_t dqdata_node(0, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        graph::op_t dqweight_node(
                1, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node, 1)

        graph::op_t convtranspose_node(
                2, graph::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONVTRANSPOSE_ATTR(convtranspose_node, nd)

        graph::op_t dqother_node(3, graph::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        dqother_node.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        dqother_node.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t add_node(4, graph::op_kind::Add, "add_node");

        graph::op_t qout_node(5, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        graph::op_t dqdata_node2(6, graph::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node2)

        graph::op_t dqweight_node2(
                7, graph::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node2, 1)

        graph::op_t convtranspose_node2(
                8, graph::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONVTRANSPOSE_ATTR(convtranspose_node2, nd)

        graph::op_t qout_node2(9, graph::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node2)

        auto src_u8 = utils::logical_tensor_init(
                0, src_shape, graph::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(
                1, src_shape, graph::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(
                2, weight_shape, graph::data_type::s8);
        auto weight_f32_dq = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                4, dst_shape, graph::data_type::f32);
        auto other_f32_dq = utils::logical_tensor_init(
                6, other_shape, graph::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        auto dst_s8 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::s8);

        auto src_u8_2 = utils::logical_tensor_init(
                9, src_shape, graph::data_type::u8);
        auto src_f32_dq_2 = utils::logical_tensor_init(
                10, src_shape, graph::data_type::f32);
        auto weight_s8_2 = utils::logical_tensor_init(
                11, weight_shape, graph::data_type::s8);
        auto weight_f32_dq_2 = utils::logical_tensor_init(
                12, weight_shape, graph::data_type::f32);
        auto dst_f32_2 = utils::logical_tensor_init(
                13, dst_shape, graph::data_type::f32);
        auto dst_s8_2 = utils::logical_tensor_init(
                14, dst_shape, graph::data_type::s8);

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        convtranspose_node.add_output(dst_f32);

        dqother_node.add_input(dst_s8_2);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        qout_node.add_input(dst_add_f32);
        qout_node.add_output(dst_s8);

        dqdata_node2.add_input(src_u8_2);
        dqdata_node2.add_output(src_f32_dq_2);

        dqweight_node2.add_input(weight_s8_2);
        dqweight_node2.add_output(weight_f32_dq_2);

        convtranspose_node2.add_input(src_f32_dq_2);
        convtranspose_node2.add_input(weight_f32_dq_2);
        convtranspose_node2.add_output(dst_f32_2);

        qout_node2.add_input(dst_f32_2);
        qout_node2.add_output(dst_s8_2);

        graph::graph_t graph(engine->kind());
        graph.add_op(&dqdata_node);
        graph.add_op(&dqweight_node);
        graph.add_op(&convtranspose_node);
        graph.add_op(&dqother_node);
        graph.add_op(&add_node);
        graph.add_op(&qout_node);
        graph.add_op(&dqdata_node2);
        graph.add_op(&dqweight_node2);
        graph.add_op(&convtranspose_node2);
        graph.add_op(&qout_node2);
        graph.finalize();

        graph::pass::pass_base_ptr apass1
                = get_pass("int8_convtranspose_add_post_ops_fusion_cpu");
        graph::pass::pass_base_ptr apass2
                = get_pass("int8_convtranspose_post_ops_fusion_cpu");
        apass1->run(graph);
        apass2->run(graph);
        ASSERT_EQ(graph.get_num_partitions(), 2U);
        auto part2 = graph.get_partitions()[0]; // int8_convtranspose
        auto part1 = graph.get_partitions()[1]; // int8_convtranspose_add

        // compile
        graph::partition_t p1, p2;
        p1.init(part1);
        p2.init(part2);

        graph::compiled_partition_t cp1(p1);
        graph::compiled_partition_t cp2(p2);

        std::vector<const graph::logical_tensor_t *> lt_ins1 {
                &src_u8_2, &weight_s8_2};

        dst_s8_2.layout_type = graph::layout_type::any;
        std::vector<const graph::logical_tensor_t *> lt_outs1 {&dst_s8_2};

        p1.compile(&cp1, lt_ins1, lt_outs1, engine);

        cp1.query_logical_tensor(dst_s8_2.id, &dst_s8_2);

        std::vector<const graph::logical_tensor_t *> lt_ins;
        lt_ins = {&src_u8, &weight_s8, &dst_s8_2};

        dst_s8.layout_type = graph::layout_type::any;
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p2.compile(&cp2, lt_ins, lt_outs, engine);

        std::vector<graph::inplace_pair_t> inplace_pairs
                = cp2.get_inplace_pairs();

        ASSERT_EQ(inplace_pairs.size(), 1U);
        ASSERT_EQ(inplace_pairs[0].input_id, dst_s8_2.id);
        ASSERT_EQ(inplace_pairs[0].output_id, dst_s8.id);
    }
}

TEST(ExecuteSubgraphFp32, Convtranspose3Postops) {
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    const std::vector<graph::op_kind_t> supported_binary_ops {
            graph::op_kind::Add,
            graph::op_kind::Divide,
            graph::op_kind::Maximum,
            graph::op_kind::Minimum,
            graph::op_kind::Multiply,
            graph::op_kind::Subtract,
    };
    std::vector<std::vector<graph::op_kind_t>> post_op_t_seqs {
            {graph::op_kind::Abs, graph::op_kind::Sqrt},
            {graph::op_kind::Elu, graph::op_kind::SoftPlus,
                    graph::op_kind::Tanh},
            {graph::op_kind::ReLU, graph::op_kind::Log,
                    graph::op_kind::Subtract},
            {graph::op_kind::Multiply, graph::op_kind::HardSwish},
    };

    std::vector<size_t> nds = {2};
    std::vector<int64_t> groups = {1};
    std::vector<bool> with_biases = {true};

    for_(const auto &post_op_ts : post_op_t_seqs)
    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for (const auto with_bias : with_biases) {
        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {in_channel, out_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {in_channel, out_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {in_channel, out_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 14}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 14, 14}
                          : std::vector<int64_t> {1, out_channel, 14, 14, 14};
        // GPU does not support non-broadcastable binary post ops currently
        std::vector<int64_t> other_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 1}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 1, 14}
                          : std::vector<int64_t> {1, out_channel, 1, 14, 14};

        std::vector<test::vector<float>> datas {};
        datas.emplace_back(product(src_shape));
        datas.emplace_back(product(weight_shape));
        datas.emplace_back(product(bias_shape));
        for (size_t i = 0; i < 3; ++i)
            datas.emplace_back(product(dst_shape));
        test::vector<float> case1_out_data(product(dst_shape));
        test::vector<float> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(1.0f, 25.0f);
        for (auto &data : datas) {
            std::generate(data.begin(), data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        graph::graph_t agraph(engine->kind());
        std::vector<graph::logical_tensor_t> lt_vec {};
        size_t lt_idx = 0;
        std::vector<size_t> input_lts = {};
        std::vector<size_t> output_lts = {};

        graph::op_t convtranspose_node(
                0, graph::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONVTRANSPOSE_ATTR(convtranspose_node, nd)
        lt_vec.emplace_back(utils::logical_tensor_init(
                0, src_shape, graph::data_type::f32));
        convtranspose_node.add_input(lt_vec.back());
        input_lts.push_back(lt_idx);
        ++lt_idx;
        lt_vec.emplace_back(utils::logical_tensor_init(
                lt_idx, weight_shape, graph::data_type::f32));
        convtranspose_node.add_input(lt_vec.back());
        input_lts.push_back(lt_idx);
        if (!with_bias) {
            ++lt_idx;
            lt_vec.emplace_back(utils::logical_tensor_init(
                    lt_idx, bias_shape, graph::data_type::f32));
            convtranspose_node.add_input(lt_vec.back());
            input_lts.push_back(lt_idx);
        }
        ++lt_idx;
        lt_vec.emplace_back(utils::logical_tensor_init(
                lt_idx, dst_shape, graph::data_type::f32));
        convtranspose_node.add_output(lt_vec.back());
        ASSERT_EQ(agraph.add_op(&convtranspose_node), graph::status::success);

        graph::op_t bias_node(1, graph::op_kind::BiasAdd, "bias_node");
        bias_node.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        if (with_bias) {
            bias_node.add_input(lt_vec.back());
            ++lt_idx;
            lt_vec.emplace_back(utils::logical_tensor_init(
                    lt_idx, bias_shape, graph::data_type::f32));
            bias_node.add_input(lt_vec.back());
            input_lts.push_back(lt_idx);
            ++lt_idx;
            lt_vec.emplace_back(utils::logical_tensor_init(
                    lt_idx, dst_shape, graph::data_type::f32));
            bias_node.add_output(lt_vec.back());
            ASSERT_EQ(agraph.add_op(&bias_node), graph::status::success);
        }

        for (size_t i = 0; i < post_op_ts.size(); ++i) {
            auto activation_t = post_op_ts[i];
            graph::op_t activation {i + 2, activation_t, "post_op_node"};
            // set additional parameters for specific ops
            if (activation_t == graph::op_kind::Elu) {
                activation.set_attr<float>(graph::op_attr::alpha, 1.0f);
            } else if (activation_t == graph::op_kind::Clamp) {
                activation.set_attr<float>(graph::op_attr::min, 1.0f);
                activation.set_attr<float>(graph::op_attr::max, 3.0f);
            }

            activation.add_input(lt_vec[lt_idx]);
            if (std::find(supported_binary_ops.begin(),
                        supported_binary_ops.end(), activation_t)
                    != supported_binary_ops.end()) {
                ++lt_idx;
                lt_vec.emplace_back(utils::logical_tensor_init(
                        lt_idx, other_shape, graph::data_type::f32));
                activation.add_input(lt_vec.back());
                input_lts.push_back(lt_idx);
            }
            ++lt_idx;
            lt_vec.emplace_back(utils::logical_tensor_init(
                    lt_idx, dst_shape, graph::data_type::f32));
            activation.add_output(lt_vec.back());
            ASSERT_EQ(agraph.add_op(&activation), graph::status::success);
        }

        output_lts.push_back(lt_idx);

        agraph.finalize();

        graph::tensor_t dst_case1_ts(
                lt_vec[output_lts[0]], engine, case1_out_data.data());
        graph::tensor_t dst_case2_ts(
                lt_vec[output_lts[0]], engine, case2_out_data.data());
        std::vector<graph::tensor_t> src_tss {};
        for (size_t i = 0; i < input_lts.size(); ++i)
            src_tss.emplace_back(lt_vec[input_lts[i]], engine, datas[i].data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(agraph, src_tss, {dst_case1_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass("convtranspose_post_ops_fusion");
        ASSERT_TRUE(apass != nullptr);
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        auto part = agraph.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins(input_lts.size());
        std::transform(input_lts.begin(), input_lts.end(), lt_ins.begin(),
                [&](size_t idx) -> graph::logical_tensor_t * {
                    return &lt_vec[idx];
                });
        std::vector<const graph::logical_tensor_t *> lt_outs {
                &lt_vec[output_lts[0]]};

        p.compile(&cp, lt_ins, lt_outs, engine);

        cp.execute(strm, src_tss, {dst_case2_ts});
        strm->wait();

        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                /*atol*/ 1.f));
    }
}

TEST(Execute, ConvtransposeWithCache) {
    using dims = graph::dnnl_impl::dims;

    // default engine kind is cpu.
    graph::engine_t *eng = get_engine();
    test::vector<float> src {1.0, 2.0, 3.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst {3.0, 0.0, 3.0, 0.0, 7.0, 0.0, 7.0, 0.0};
    test::vector<float> dst(ref_dst.size(), 0);
    graph::op_t convtranspose_op(graph::op_kind::ConvTranspose);
    convtranspose_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    convtranspose_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    convtranspose_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    convtranspose_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    convtranspose_op.set_attr<int64_t>(graph::op_attr::groups, 2);
    convtranspose_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    convtranspose_op.set_attr<std::string>(
            graph::op_attr::weights_format, "IOX");

    // prepare logical tensor
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 4, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {4, 4, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 8, 1, 1}, graph::data_type::f32);

    convtranspose_op.add_input(src_lt);
    convtranspose_op.add_input(weight_lt);
    convtranspose_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&convtranspose_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("convtranspose_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);
    ASSERT_EQ(dst_lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t weight_ts(weight_lt, eng, weight.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts, weight_ts}, {dst_ts}),
            graph::status::success);
    strm->wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    //test with same data and stream to see
    //if the memory cache runs correctly
    test::vector<float> src2 {1.0, 2.0, 3.0, 4.0};
    test::vector<float> weight2 {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst2 {3.0, 0.0, 3.0, 0.0, 7.0, 0.0, 7.0, 0.0};
    test::vector<float> dst2(ref_dst.size(), 0);

    graph::tensor_t src_ts2(src_lt, eng, src2.data());
    graph::tensor_t weight_ts2(weight_lt, eng, weight2.data());
    graph::tensor_t dst_ts2(dst_lt, eng, dst2.data());

    ASSERT_EQ(cp.execute(strm, {src_ts2, weight_ts2}, {dst_ts2}),
            graph::status::success);
    strm->wait();
    for (size_t i = 0; i < dst2.size(); ++i) {
        ASSERT_FLOAT_EQ(dst2[i], ref_dst2[i]);
    }
}
