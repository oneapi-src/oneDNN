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

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

struct prelu_params_t {
    dnnl::impl::graph::dims wei_dims;
    std::string data_format;
    bool per_channel_broadcast;
};

class prelu_t : public ::testing::TestWithParam<prelu_params_t> {
public:
    void TestPrelu() {
        const auto params
                = ::testing::TestWithParam<prelu_params_t>::GetParam();
        graph::engine_t *eng = get_engine();

        graph::op_t op(graph::op_kind::PReLU, "prelu");
        op.set_attr<std::string>(
                graph::op_attr::data_format, params.data_format);
        op.set_attr<bool>(graph::op_attr::per_channel_broadcast,
                params.per_channel_broadcast);

        dnnl::impl::graph::dims dims {1, 2, 2, 2};
        test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5, -1.0, 1.0};
        std::vector<graph::dim_t> wei_dims = params.wei_dims;
        test::vector<float> wei(product(wei_dims));
        test::vector<float> case1_out_data(product(dims));
        test::vector<float> case2_out_data(product(dims));
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(wei.begin(), wei.end(),
                [&]() { return f32_distribution(generator); });

        graph::logical_tensor_t src_lt
                = utils::logical_tensor_init(0, dims, graph::data_type::f32);
        graph::logical_tensor_t wei_lt = utils::logical_tensor_init(
                1, wei_dims, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, dims, graph::data_type::f32, graph::layout_type::any);

        op.add_input(src_lt);
        op.add_input(wei_lt);
        op.add_output(dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("prelu_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &wei_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

        p.compile(&cp, inputs, outputs, eng);

        graph::logical_tensor_t lt;
        cp.query_logical_tensor(dst_lt.id, &lt);

        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(src_lt, eng, src.data());
        graph::tensor_t wei_ts(wei_lt, eng, wei.data());
        graph::tensor_t dst_ts1(dst_lt, eng, case1_out_data.data());
        graph::tensor_t dst_ts2(dst_lt, eng, case2_out_data.data());

        graph::stream_t *strm = get_stream();

        ASSERT_EQ(run_graph(g, {src_ts, wei_ts}, {dst_ts1}, *eng, *strm),
                graph::status::success);
        cp.execute(strm, {src_ts, wei_ts}, {dst_ts2});
        strm->wait();

        for (size_t i = 0; i < src.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
};

struct prelu_bwd_params_t {
    dnnl::impl::graph::dims data_dims;
    dnnl::impl::graph::dims wei_dims;
    std::string data_format;
};

class prelu_backprop_t : public ::testing::TestWithParam<prelu_bwd_params_t> {
public:
    void TestPreluBackprop() {
        const auto params
                = ::testing::TestWithParam<prelu_bwd_params_t>::GetParam();
        graph::engine_t *eng = get_engine();
        graph::stream_t *strm = get_stream();

        std::vector<graph::dim_t> data_dims = params.data_dims;
        std::vector<graph::dim_t> wei_dims = params.wei_dims;
        test::vector<float> src(product(data_dims));
        test::vector<float> wei(product(wei_dims));
        test::vector<float> diff_dst(product(data_dims));
        test::vector<float> diff_src1(product(data_dims));
        test::vector<float> diff_src2(product(data_dims));
        test::vector<float> diff_wei1(product(wei_dims));
        test::vector<float> diff_wei2(product(wei_dims));
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src.begin(), src.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(wei.begin(), wei.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(diff_dst.begin(), diff_dst.end(),
                [&]() { return f32_distribution(generator); });

        graph::op_t prelu_op(graph::op_kind::PReLUBackprop, "prelu_bwd");
        prelu_op.set_attr<std::string>(
                graph::op_attr::data_format, params.data_format);

        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, data_dims, graph::data_type::f32);
        graph::logical_tensor_t wei_lt = utils::logical_tensor_init(
                1, wei_dims, graph::data_type::f32);
        graph::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
                2, data_dims, graph::data_type::f32);
        graph::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
                3, data_dims, graph::data_type::f32, graph::layout_type::any);
        graph::logical_tensor_t diff_wei_lt = utils::logical_tensor_init(
                4, wei_dims, graph::data_type::f32, graph::layout_type::any);

        prelu_op.add_input(src_lt);
        prelu_op.add_input(wei_lt);
        prelu_op.add_input(diff_dst_lt);
        prelu_op.add_output(diff_src_lt);
        prelu_op.add_output(diff_wei_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&prelu_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("prelu_bwd_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src_lt, &wei_lt, &diff_dst_lt};
        std::vector<const graph::logical_tensor_t *> outputs {
                &diff_src_lt, &diff_wei_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

        graph::tensor_t src_ts(src_lt, eng, src.data());
        graph::tensor_t wei_ts(wei_lt, eng, wei.data());
        graph::tensor_t diff_dst_ts(diff_dst_lt, eng, diff_dst.data());
        graph::tensor_t diff_src_ts1(diff_src_lt, eng, diff_src1.data());
        graph::tensor_t diff_wei_ts1(diff_wei_lt, eng, diff_wei1.data());
        graph::tensor_t diff_src_ts2(diff_src_lt, eng, diff_src2.data());
        graph::tensor_t diff_wei_ts2(diff_wei_lt, eng, diff_wei2.data());

        ASSERT_EQ(run_graph(g, {src_ts, wei_ts, diff_dst_ts},
                          {diff_src_ts1, diff_wei_ts1}, *eng, *strm),
                graph::status::success);

        ASSERT_EQ(cp.execute(strm, {src_ts, wei_ts, diff_dst_ts},
                          {diff_src_ts2, diff_wei_ts2}),
                graph::status::success);
        strm->wait();

        for (size_t i = 0; i < diff_src1.size(); ++i) {
            ASSERT_FLOAT_EQ(diff_src1[i], diff_src2[i]);
        }
        for (size_t i = 0; i < diff_wei1.size(); ++i) {
            ASSERT_FLOAT_EQ(diff_wei1[i], diff_wei2[i]);
        }
    }
};

TEST_P(prelu_t, TestPrelu) {
    TestPrelu();
}

INSTANTIATE_TEST_SUITE_P(Execute, prelu_t,
        ::testing::Values(
                // no broadcast
                prelu_params_t {{1, 2, 2, 2}, "NXC", false},
                // channel-shared broadcast
                prelu_params_t {{1, 1, 1, 1}, "NXC", false},
                // shared-axes broadcast
                prelu_params_t {{1, 2, 2, 1}, "NCX", false},
                // channel-wise broadcast, NCX
                prelu_params_t {{1, 2, 1, 1}, "NCX", true},
                // channel-wise broadcast, NXC
                prelu_params_t {{1, 1, 1, 2}, "NXC", true},
                // 1D weights broadcast, NXC
                prelu_params_t {{2}, "NXC", true},
                // 1d weights, no channel-wise broadcast, NCX
                prelu_params_t {{2}, "NCX", false},
                // 1d weights, channel-wise broadcast, NCX
                prelu_params_t {{2}, "NCX", true}));

TEST_P(prelu_backprop_t, TestPreluBackprop) {
    TestPreluBackprop();
}

INSTANTIATE_TEST_SUITE_P(Execute, prelu_backprop_t,
        ::testing::Values(
                // NCX, 1d slope, per tensor broadcast, pytorch case
                prelu_bwd_params_t {{1, 2, 2, 2}, {1}, "NCX"},
                // NCX, 1d slope, per channel broadcast, pytorch case
                prelu_bwd_params_t {{1, 2, 1, 1}, {2}, "NCX"},
                // NCX, tensorflow case
                prelu_bwd_params_t {{1, 2, 2, 2}, {2, 2, 2}, "NCX"},
                // NXC, 1d slope, per tensor broadcast, pytorch case
                prelu_bwd_params_t {{1, 2, 2, 2}, {1}, "NXC"},
                // NXC, 1d slope, per channel broadcast, pytorch case
                prelu_bwd_params_t {{1, 1, 1, 2}, {2}, "NXC"},
                // 2d input, per tensor broadcast
                prelu_bwd_params_t {{1, 2}, {1}, "NCX"},
                // 2d input, per channel broadcast
                prelu_bwd_params_t {{1, 2}, {2}, "NCX"}));
