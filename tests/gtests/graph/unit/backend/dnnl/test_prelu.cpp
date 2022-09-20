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
    std::vector<float> wei;
    std::vector<float> ref_dst;
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

        test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5, -1.0, 1.0};
        test::vector<float> wei;
        copy_data(params.wei, wei);
        test::vector<float> dst(src.size(), 0.0);
        dnnl::impl::graph::dims dims {1, 2, 2, 2};

        graph::logical_tensor_t src_lt
                = utils::logical_tensor_init(0, dims, graph::data_type::f32);
        graph::logical_tensor_t wei_lt = utils::logical_tensor_init(
                1, params.wei_dims, graph::data_type::f32);
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
        graph::tensor_t dst_ts(dst_lt, eng, dst.data());
        graph::stream_t *strm = get_stream();

        cp.execute(strm, {src_ts, wei_ts}, {dst_ts});
        strm->wait();

        for (size_t i = 0; i < src.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], params.ref_dst[i]);
        }
    }
};

struct prelu_bwd_params_t {
    dnnl::impl::graph::dims data_dims;
    dnnl::impl::graph::dims wei_dims;
    dnnl::impl::graph::dims diff_wei_dims;
    std::vector<float> src;
    std::vector<float> wei;
    std::vector<float> diff_dst;
    std::vector<float> ref_diff_src;
    std::vector<float> ref_diff_wei;
    std::string data_format;
};

class prelu_backprop_t : public ::testing::TestWithParam<prelu_bwd_params_t> {
public:
    void TestPreluBackward() {
        const auto params
                = ::testing::TestWithParam<prelu_bwd_params_t>::GetParam();
        graph::engine_t *eng = get_engine();
        graph::stream_t *strm = get_stream();

        test::vector<float> src;
        test::vector<float> wei;
        test::vector<float> diff_dst;
        copy_data(params.src, src);
        copy_data(params.wei, wei);
        copy_data(params.diff_dst, diff_dst);

        test::vector<float> diff_src(src.size(), 0.f);

        size_t diff_wei_size = 1;
        for (auto dim : params.diff_wei_dims) {
            diff_wei_size *= dim;
        }
        test::vector<float> diff_wei(diff_wei_size, 0.f);

        graph::op_t prelu_op(graph::op_kind::PReLUBackward, "prelu_bwd");
        prelu_op.set_attr<std::string>(
                graph::op_attr::data_format, params.data_format);

        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, params.data_dims, graph::data_type::f32);
        graph::logical_tensor_t wei_lt = utils::logical_tensor_init(
                1, params.wei_dims, graph::data_type::f32);
        graph::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
                2, params.data_dims, graph::data_type::f32);
        graph::logical_tensor_t diff_src_lt
                = utils::logical_tensor_init(3, params.data_dims,
                        graph::data_type::f32, graph::layout_type::any);
        graph::logical_tensor_t diff_wei_lt
                = utils::logical_tensor_init(4, params.diff_wei_dims,
                        graph::data_type::f32, graph::layout_type::any);

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
        graph::tensor_t diff_src_ts(diff_src_lt, eng, diff_src.data());
        graph::tensor_t diff_wei_ts(diff_wei_lt, eng, diff_wei.data());

        ASSERT_EQ(cp.execute(strm, {src_ts, wei_ts, diff_dst_ts},
                          {diff_src_ts, diff_wei_ts}),
                graph::status::success);
        strm->wait();

        for (size_t i = 0; i < params.ref_diff_src.size(); ++i) {
            ASSERT_FLOAT_EQ(diff_src[i], params.ref_diff_src[i]);
        }
        for (size_t i = 0; i < params.ref_diff_wei.size(); ++i) {
            ASSERT_FLOAT_EQ(diff_wei[i], params.ref_diff_wei[i]);
        }
    }
};

TEST_P(prelu_t, TestPrelu) {
    TestPrelu();
}

INSTANTIATE_TEST_SUITE_P(Execute, prelu_t,
        ::testing::Values(
                // no broadcast
                prelu_params_t {{1, 2, 2, 2},
                        {2.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                        {-4.0, -3.0, -2.0, 0.0, 0.0, 3.5, -1.0, 1.0}, "NXC",
                        false},
                // channel-shared broadcast
                prelu_params_t {{1, 1, 1, 1}, {2.0},
                        {-4.0, -3.0, -2.0, -1.0, 0.0, 3.5, -2.0, 1.0}, "NXC",
                        false},
                // shared-axes broadcast
                prelu_params_t {{1, 2, 2, 1}, {2.0, 1.0, 1.0, 2.0},
                        {-4.0, -3.0, -1.0, -0.5, 0.0, 3.5, -2.0, 1.0}, "NCX",
                        false},
                // channel-wise broadcast, NCX
                prelu_params_t {{1, 2, 1, 1}, {1.0, 0.0},
                        {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5, 0.0, 1.0}, "NCX",
                        true},
                // channel-wise broadcast, NXC
                prelu_params_t {{1, 1, 1, 2}, {1.0, 0.0},
                        {-2.0, 0.0, -1.0, 0.0, 0.0, 3.5, -1.0, 1.0}, "NXC",
                        true},
                // 1D weights broadcast, NXC
                prelu_params_t {{2}, {1.0, 2.0},
                        {-2.0, -3.0, -1.0, -1.0, 0.0, 3.5, -1.0, 1.0}, "NXC",
                        true},
                // 1d weights, no channel-wise broadcast, NCX
                prelu_params_t {{2}, {1.0, 2.0},
                        {-2.0, -3.0, -1.0, -1.0, 0.0, 3.5, -1.0, 1.0}, "NCX",
                        false},
                // 1d weights, channel-wise broadcast, NCX
                prelu_params_t {{2}, {1.0, 2.0},
                        {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5, -2.0, 1.0}, "NCX",
                        true}));

TEST_P(prelu_backprop_t, TestPreluBackward) {
    TestPreluBackward();
}

INSTANTIATE_TEST_SUITE_P(Execute, prelu_backprop_t,
        ::testing::Values(
                // NCX, 1d slope, per tensor broadcast, pytorch case
                prelu_bwd_params_t {{1, 2, 2, 2}, {1}, {1},
                        {-0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, 2.0}, {-4.0},
                        {-0.0625, 0.125, 0.0, 0.0625, -0.125, 0.0, 0.0625,
                                0.125},
                        {0.25, -0.5, -0.0, 0.0625, 0.5, 0.0, 0.0625, 0.125},
                        {0.125}, "NCX"},
                // NCX, 1d slope, per channel broadcast, pytorch case
                prelu_bwd_params_t {{1, 2, 1, 1}, {2}, {2}, {-0.0, 2.0},
                        {-4.0, 2.0}, {-0.0625, 0.125}, {0.25, 0.125},
                        {0.0, 0.0}, "NCX"},
                // NCX, tensorflow case
                prelu_bwd_params_t {{1, 2, 2, 2}, {2, 2, 2}, {2, 2, 2},
                        {-0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, 2.0},
                        {-4.0, 2.0, 1.0, 0.5, -0.25, 8.0, 4.0, 2.0},
                        {-0.0625, 0.125, 0.0, 0.0625, -0.125, 0.0, 0.0625,
                                0.125},
                        {0.25, 0.25, 0.0, 0.0625, 0.03125, 0.0, 0.0625, 0.125},
                        {0.0, 0.0, 0.0, 0.0, 0.125, 0.0, 0.0, 0.0}, "NCX"},
                // NXC, 1d slope, per tensor broadcast, pytorch case
                prelu_bwd_params_t {{1, 2, 2, 2}, {1}, {1},
                        {-0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0}, {-4.0},
                        {-0.0625, -0.125, 0.125, 0.0, 0.0, 0.0625, 0.0625,
                                0.125},
                        {0.25, 0.5, -0.5, 0.0, -0.0, 0.0625, 0.0625, 0.125},
                        {0.125}, "NXC"},
                // NXC, 1d slope, per channel broadcast, pytorch case
                prelu_bwd_params_t {{1, 1, 1, 2}, {2}, {2}, {-0.0, -1.0},
                        {-4.0, 1.0}, {-0.0625, -0.125}, {0.25, -0.125},
                        {0.0, 0.125}, "NXC"},
                // 2d input, per tensor broadcast
                prelu_bwd_params_t {{1, 2}, {1}, {1}, {-0.0, 0.0}, {-4.0},
                        {-0.0625, 0.125}, {0.25, -0.5}, {0.0}, "NCX"},
                // 2d input, per channel broadcast
                prelu_bwd_params_t {{1, 2}, {2}, {2}, {-0.0, 0.0}, {-4.0, 2.0},
                        {-0.0625, 0.125}, {0.25, 0.25}, {0.0, 0.0}, "NCX"}));
