/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "test_api_common.hpp"
#include "gtest/gtest.h"

struct dnnl_graph_test_pool_params_t {
    std::vector<int64_t> input_dims;
    std::vector<int64_t> ref_dst_dims;
    std::vector<int64_t> kernel;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    std::vector<int64_t> pads_begin;
    std::vector<int64_t> pads_end;
    bool exclude_pad;
    std::string data_format;
    std::string rounding_type;

    std::vector<float> src;
    std::vector<float> ref_dst;
};

class test_average_pool_compile_t
    : public ::testing::TestWithParam<dnnl_graph_test_pool_params_t> {
public:
    void Test_Average_Pool() {
        auto params = ::testing::TestWithParam<
                dnnl_graph_test_pool_params_t>::GetParam();

        using namespace dnnl::graph;
        engine::kind engine_kind = engine::kind::cpu;
        engine eng = cpp_api_test_dnnl_graph_engine_create(engine_kind);
        stream strm {eng};

        graph g(engine_kind);

        const auto &input_dims = params.input_dims;
        const auto &ref_dst_dims = params.ref_dst_dims;
        const auto &kernel = params.kernel;
        const auto &strides = params.strides;
        const auto &pads_begin = params.pads_begin;
        const auto &pads_end = params.pads_end;
        const auto &exclude_pad = params.exclude_pad;
        const auto &data_format = params.data_format;
        const auto &rounding_type = params.rounding_type;
        auto &src = params.src;
        auto &ref_dst = params.ref_dst;

        std::vector<int64_t> infer_dst_dims {-1, -1, -1, -1};

        logical_tensor lt1 {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt2 {1, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};

        op pool_op(0, op::kind::AvgPool, "pool");

        pool_op.set_attr<std::vector<int64_t>>("kernel", kernel);
        pool_op.set_attr<std::vector<int64_t>>("strides", strides);
        pool_op.set_attr<std::vector<int64_t>>("pads_begin", pads_begin);
        pool_op.set_attr<std::vector<int64_t>>("pads_end", pads_end);
        pool_op.set_attr<std::string>("data_format", data_format);
        pool_op.set_attr<std::string>("rounding_type", rounding_type);
        pool_op.set_attr<bool>("exclude_pad", exclude_pad);

        pool_op.add_input(lt1);
        pool_op.add_output(lt2);

        g.add_op(pool_op);

        //create_partition
        auto partitions = g.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions.size(), 1);

        // check partition engine kind
        ASSERT_EQ(partitions[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops = partitions[0].get_ops();
        ASSERT_EQ(ops.size(), 1);
        ASSERT_EQ(partitions[0].get_ops_num(), 1);

        logical_tensor lt1_plain {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt2_plain {1, logical_tensor::data_type::f32,
                infer_dst_dims, logical_tensor::layout_type::strided};

        //compile partition
        std::vector<logical_tensor> in0({lt1_plain});
        std::vector<logical_tensor> out0({lt2_plain});

        auto cp = partitions[0].compile(in0, out0, eng);

        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[0], ref_dst_dims[0]);
        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[1], ref_dst_dims[1]);
        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[2], ref_dst_dims[2]);
        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[3], ref_dst_dims[3]);

        std::vector<float> dst(ref_dst.size(), 0.0);

        tensor src_ts(lt1_plain, eng, src.data());
        tensor dst_ts(out0[0], eng, dst.data());

        cp.execute(strm, {src_ts}, {dst_ts});

        strm.wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
        }
    }
};

TEST_P(test_average_pool_compile_t, Test_Average_Pool_Compile) {
    Test_Average_Pool();
}

INSTANTIATE_TEST_SUITE_P(Test_Average_Pool_Compile, test_average_pool_compile_t,
        ::testing::Values(dnnl_graph_test_pool_params_t {{1, 1, 4, 4},
                                  {1, 1, 2, 3}, {2, 2}, {3, 1}, {0, 0}, {0, 0},
                                  {0, 0}, false, "NCX", "ceil",
                                  {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f,
                                          8.0f, 9.0f, 11.0f, 10.0f, 12.0f,
                                          13.0f, 15.0f, 14.0f, 16.0f},
                                  {4.0f, 4.5f, 5.0f, 14.0f, 14.5f, 15.0f}},
                dnnl_graph_test_pool_params_t {{1, 1, 4, 4}, {1, 1, 2, 3},
                        {2, 2}, {3, 1}, {0, 0}, {0, 0}, {0, 0}, true, "NCX",
                        "ceil",
                        {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f, 8.0f, 9.0f,
                                11.0f, 10.0f, 12.0f, 13.0f, 15.0f, 14.0f,
                                16.0f},
                        {4.0f, 4.5f, 5.0f, 14.0f, 14.5f, 15.0f}}));

class test_max_pool_compile_t
    : public ::testing::TestWithParam<dnnl_graph_test_pool_params_t> {
public:
    void Test_Max_Pool() {
        auto params = ::testing::TestWithParam<
                dnnl_graph_test_pool_params_t>::GetParam();

        using namespace dnnl::graph;
        engine::kind engine_kind = engine::kind::cpu;
        engine eng = cpp_api_test_dnnl_graph_engine_create(engine_kind);
        stream strm {eng};

        graph g(engine_kind);

        const auto &input_dims = params.input_dims;
        const auto &ref_dst_dims = params.ref_dst_dims;
        const auto &kernel = params.kernel;
        const auto &strides = params.strides;
        const auto &dilations = params.dilations;
        const auto &pads_begin = params.pads_begin;
        const auto &pads_end = params.pads_end;
        const auto &data_format = params.data_format;
        const auto &rounding_type = params.rounding_type;
        auto &src = params.src;
        auto &ref_dst = params.ref_dst;

        std::vector<int64_t> infer_dst_dims {-1, -1, -1, -1};

        logical_tensor lt1 {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt2 {1, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};

        op pool_op(0, op::kind::MaxPool, "pool");

        pool_op.set_attr<std::vector<int64_t>>("kernel", kernel);
        pool_op.set_attr<std::vector<int64_t>>("strides", strides);
        pool_op.set_attr<std::vector<int64_t>>("dilations", dilations);
        pool_op.set_attr<std::vector<int64_t>>("pads_begin", pads_begin);
        pool_op.set_attr<std::vector<int64_t>>("pads_end", pads_end);
        pool_op.set_attr<std::string>("data_format", data_format);
        pool_op.set_attr<std::string>("rounding_type", rounding_type);

        pool_op.add_input(lt1);
        pool_op.add_output(lt2);

        g.add_op(pool_op);

        //create_partition
        auto partitions = g.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions.size(), 1);

        // check partition engine kind
        ASSERT_EQ(partitions[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops = partitions[0].get_ops();
        ASSERT_EQ(ops.size(), 1);
        ASSERT_EQ(partitions[0].get_ops_num(), 1);

        logical_tensor lt1_plain {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt2_plain {1, logical_tensor::data_type::f32,
                infer_dst_dims, logical_tensor::layout_type::strided};

        //compile partition
        std::vector<logical_tensor> in0({lt1_plain});
        std::vector<logical_tensor> out0({lt2_plain});

        auto cp = partitions[0].compile(in0, out0, eng);

        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[0], ref_dst_dims[0]);
        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[1], ref_dst_dims[1]);
        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[2], ref_dst_dims[2]);
        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[3], ref_dst_dims[3]);

        std::vector<float> dst(ref_dst.size(), 0.0);

        tensor src_ts(lt1_plain, eng, src.data());
        tensor dst_ts(out0[0], eng, dst.data());

        cp.execute(strm, {src_ts}, {dst_ts});

        strm.wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
        }
    }
};

TEST_P(test_max_pool_compile_t, Test_Max_Pool_Compile) {
    Test_Max_Pool();
}

INSTANTIATE_TEST_SUITE_P(Test_Max_Pool_Compile, test_max_pool_compile_t,
        ::testing::Values(
                dnnl_graph_test_pool_params_t {{1, 1, 3, 3}, {1, 1, 2, 2},
                        {2, 2}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false, "NCX",
                        "ceil",
                        {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f, 8.0f, 9.0f},
                        {5.0f, 7.0f, 8.0f, 9.0f}},
                dnnl_graph_test_pool_params_t {{1, 1, 3, 3}, {1, 1, 2, 2},
                        {2, 2}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false, "NCX",
                        "floor",
                        {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f, 8.0f, 9.0f},
                        {5.0f, 7.0f, 8.0f, 9.0f}}));

struct dnnl_graph_test_bn_params_t {
    std::vector<int64_t> input_dims;
    float epsilon;
    std::string data_format;
};

class test_bn_compile_t
    : public ::testing::TestWithParam<dnnl_graph_test_bn_params_t> {
public:
    void Test_BatchNormTraining() {
        auto params = ::testing::TestWithParam<
                dnnl_graph_test_bn_params_t>::GetParam();

        using namespace dnnl::graph;
        engine::kind engine_kind = engine::kind::cpu;
        engine eng = cpp_api_test_dnnl_graph_engine_create(engine_kind);
        stream strm {eng};

        graph g(engine_kind);

        const auto &input_dims = params.input_dims;
        const auto &epsilon = params.epsilon;
        const auto &data_format = params.data_format;

        const auto &channels
                = (data_format == "NCX") ? input_dims[1] : input_dims.back();

        std::vector<int64_t> infer_dst_dims {-1, -1, -1, -1};
        std::vector<int64_t> infer_weight_dims {-1};

        std::vector<int64_t> gamma_dims = {channels};
        std::vector<int64_t> beta_dims = {channels};
        std::vector<int64_t> mean_dims = {channels};
        std::vector<int64_t> variance_dims = {channels};

        logical_tensor lt0 {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt1 {1, logical_tensor::data_type::f32, gamma_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt2 {2, logical_tensor::data_type::f32, beta_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt3 {3, logical_tensor::data_type::f32, mean_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt4 {4, logical_tensor::data_type::f32, variance_dims,
                logical_tensor::layout_type::strided};

        logical_tensor lt5 {5, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt6 {6, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::strided};
        logical_tensor lt7 {7, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::strided};
        logical_tensor lt8 {8, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::strided};
        logical_tensor lt9 {9, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::strided};

        // BatchNormForwardTraining
        op bn_fwd_op(0, op::kind::BatchNormForwardTraining, "bn_fwd");

        bn_fwd_op.set_attr<std::string>("data_format", data_format);
        bn_fwd_op.set_attr<float>("epsilon", epsilon);
        bn_fwd_op.set_attr<float>("momentum", 0.1f);

        bn_fwd_op.add_input(lt0);
        bn_fwd_op.add_input(lt1);
        bn_fwd_op.add_input(lt2);
        bn_fwd_op.add_input(lt3);
        bn_fwd_op.add_input(lt4);
        bn_fwd_op.add_output(lt5);
        bn_fwd_op.add_output(lt6);
        bn_fwd_op.add_output(lt7);
        bn_fwd_op.add_output(lt8);
        bn_fwd_op.add_output(lt9);

        g.add_op(bn_fwd_op);

        //create_partition
        auto partitions = g.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions.size(), 1);

        // check partition engine kind
        ASSERT_EQ(partitions[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops = partitions[0].get_ops();
        ASSERT_EQ(ops.size(), 1);
        ASSERT_EQ(partitions[0].get_ops_num(), 1);

        //compile partition
        std::vector<logical_tensor> in0({lt0, lt1, lt2, lt3, lt4});
        std::vector<logical_tensor> out0({lt5, lt6, lt7, lt8, lt9});

        auto cp = partitions[0].compile(in0, out0, eng);

        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[0], input_dims[0]);
        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[1], input_dims[1]);
        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[2], input_dims[2]);
        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[3], input_dims[3]);

        std::vector<float> src(product(input_dims), 0.0);
        std::vector<float> gamma(channels, 0.0);
        std::vector<float> beta(channels, 0.0);
        std::vector<float> mean(channels, 0.0);
        std::vector<float> variance(channels, 0.0);
        std::vector<float> dst(product(input_dims), 0.0);
        std::vector<float> running_mean(channels, 0.0);
        std::vector<float> running_variance(channels, 0.0);
        std::vector<float> batch_mean(channels, 0.0);
        std::vector<float> batch_variance(channels, 0.0);

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(-1.0f, 1.0f);
        std::generate(src.begin(), src.end(),
                [&]() { return distribution(generator); });
        std::generate(gamma.begin(), gamma.end(), []() {
            static int i = 0;
            return std::sin(static_cast<float>(i++) * 2.f);
        });
        std::generate(beta.begin(), beta.end(), []() {
            static int i = 0;
            return std::tan(i++);
        });
        std::generate(mean.begin(), mean.end(), []() {
            static int i = 0;
            return std::sin(static_cast<float>(i++) * 2.f);
        });
        std::generate(variance.begin(), variance.end(), []() {
            static int i = 0;
            return std::tan(i++);
        });

        logical_tensor lt0_plain {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt1_plain {1, logical_tensor::data_type::f32, gamma_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt2_plain {2, logical_tensor::data_type::f32, beta_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt3_plain {3, logical_tensor::data_type::f32, mean_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt4_plain {4, logical_tensor::data_type::f32,
                variance_dims, logical_tensor::layout_type::strided};

        tensor src_ts(lt0_plain, eng, src.data());
        tensor gamma_ts(lt1_plain, eng, gamma.data());
        tensor beta_ts(lt2_plain, eng, beta.data());
        tensor mean_ts(lt3_plain, eng, mean.data());
        tensor variance_ts(lt4_plain, eng, variance.data());

        logical_tensor dst0_lt_infered {5, logical_tensor::data_type::f32,
                input_dims, logical_tensor::layout_type::strided};
        logical_tensor dst1_lt_infered {6, logical_tensor::data_type::f32,
                mean_dims, logical_tensor::layout_type::strided};
        logical_tensor dst2_lt_infered {7, logical_tensor::data_type::f32,
                variance_dims, logical_tensor::layout_type::strided};
        logical_tensor dst3_lt_infered {8, logical_tensor::data_type::f32,
                mean_dims, logical_tensor::layout_type::strided};
        logical_tensor dst4_lt_infered {9, logical_tensor::data_type::f32,
                variance_dims, logical_tensor::layout_type::strided};
        tensor dst_ts(dst0_lt_infered, eng, dst.data());
        tensor running_mean_ts(dst1_lt_infered, eng, running_mean.data());
        tensor running_variance_ts(
                dst2_lt_infered, eng, running_variance.data());
        tensor batch_mean_ts(dst3_lt_infered, eng, batch_mean.data());
        tensor batch_variance_ts(dst4_lt_infered, eng, batch_variance.data());

        cp.execute(strm, {src_ts, gamma_ts, beta_ts, mean_ts, variance_ts},
                {dst_ts, running_mean_ts, running_variance_ts, batch_mean_ts,
                        batch_variance_ts});

        strm.wait();

        // BatchNormTrainingBackprop
        graph g2(engine_kind);
        op bn_bwd_op(0, op::kind::BatchNormTrainingBackprop, "bn_bwd");

        bn_bwd_op.set_attr<std::string>("data_format", data_format);
        bn_bwd_op.set_attr<float>("epsilon", epsilon);

        logical_tensor lt10 {10, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt11 {11, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt12 {12, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::strided};
        logical_tensor lt13 {13, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::strided};

        bn_bwd_op.add_input(lt0);
        bn_bwd_op.add_input(lt10);
        bn_bwd_op.add_input(lt1);
        bn_bwd_op.add_input(dst3_lt_infered);
        bn_bwd_op.add_input(dst4_lt_infered);
        bn_bwd_op.add_output(lt11);
        bn_bwd_op.add_output(lt12);
        bn_bwd_op.add_output(lt13);

        g2.add_op(bn_bwd_op);

        auto partitions2 = g2.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions2.size(), 1);

        //get_ops
        std::vector<size_t> ops2 = partitions2[0].get_ops();
        ASSERT_EQ(ops2.size(), 1);
        ASSERT_EQ(partitions2[0].get_ops_num(), 1);

        //compile partition
        std::vector<logical_tensor> in1(
                {lt0, lt10, lt1, dst3_lt_infered, dst4_lt_infered});
        std::vector<logical_tensor> out1({lt11, lt12, lt13});

        auto cp2 = partitions2[0].compile(in1, out1, eng);

        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[0], input_dims[0]);
        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[1], input_dims[1]);
        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[2], input_dims[2]);
        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[3], input_dims[3]);

        logical_tensor lt10_plain {10, logical_tensor::data_type::f32,
                input_dims, logical_tensor::layout_type::strided};
        tensor outgrad_ts(lt10_plain, eng, dst.data());

        logical_tensor dst0_lt_infered2 {11, logical_tensor::data_type::f32,
                input_dims, logical_tensor::layout_type::strided};
        logical_tensor dst1_lt_infered2 {12, logical_tensor::data_type::f32,
                gamma_dims, logical_tensor::layout_type::strided};
        logical_tensor dst2_lt_infered2 {13, logical_tensor::data_type::f32,
                beta_dims, logical_tensor::layout_type::strided};

        tensor ingrad_ts(dst0_lt_infered2, eng, src.data());
        tensor gammagrad_ts(dst1_lt_infered2, eng, gamma.data());
        tensor betagrad_ts(dst2_lt_infered2, eng, beta.data());

        cp2.execute(strm,
                {src_ts, outgrad_ts, gamma_ts, batch_mean_ts,
                        batch_variance_ts},
                {ingrad_ts, gammagrad_ts, betagrad_ts});

        strm.wait();
    }

    void Test_BatchNormTraining_Opaque() {
        auto params = ::testing::TestWithParam<
                dnnl_graph_test_bn_params_t>::GetParam();

        using namespace dnnl::graph;
        engine::kind engine_kind = engine::kind::cpu;
        engine eng = cpp_api_test_dnnl_graph_engine_create(engine_kind);
        stream strm {eng};

        graph g(engine_kind);

        const auto &input_dims = params.input_dims;
        const auto &epsilon = params.epsilon;
        const auto &data_format = params.data_format;

        const auto &channels
                = (data_format == "NCX") ? input_dims[1] : input_dims.back();

        std::vector<int64_t> infer_dst_dims {-1, -1, -1, -1};
        std::vector<int64_t> infer_weight_dims {-1};

        std::vector<int64_t> gamma_dims = {channels};
        std::vector<int64_t> beta_dims = {channels};
        std::vector<int64_t> mean_dims = {channels};
        std::vector<int64_t> variance_dims = {channels};

        logical_tensor lt0 {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt1 {1, logical_tensor::data_type::f32, gamma_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt2 {2, logical_tensor::data_type::f32, beta_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt3 {3, logical_tensor::data_type::f32, mean_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt4 {4, logical_tensor::data_type::f32, variance_dims,
                logical_tensor::layout_type::undef};

        logical_tensor lt5 {5, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt6 {6, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::undef};
        logical_tensor lt7 {7, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::undef};
        logical_tensor lt8 {8, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::undef};
        logical_tensor lt9 {9, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::undef};

        // BatchNormForwardTraining
        op bn_fwd_op(0, op::kind::BatchNormForwardTraining, "bn_fwd");

        bn_fwd_op.set_attr<std::string>("data_format", data_format);
        bn_fwd_op.set_attr<float>("epsilon", epsilon);
        bn_fwd_op.set_attr<float>("momentum", 0.1f);

        bn_fwd_op.add_input(lt0);
        bn_fwd_op.add_input(lt1);
        bn_fwd_op.add_input(lt2);
        bn_fwd_op.add_input(lt3);
        bn_fwd_op.add_input(lt4);
        bn_fwd_op.add_output(lt5);
        bn_fwd_op.add_output(lt6);
        bn_fwd_op.add_output(lt7);
        bn_fwd_op.add_output(lt8);
        bn_fwd_op.add_output(lt9);

        g.add_op(bn_fwd_op);

        //create_partition
        auto partitions = g.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions.size(), 1);

        // check partition engine kind
        ASSERT_EQ(partitions[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops = partitions[0].get_ops();
        ASSERT_EQ(ops.size(), 1);
        ASSERT_EQ(partitions[0].get_ops_num(), 1);

        logical_tensor lt0_plain {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt1_plain {1, logical_tensor::data_type::f32, gamma_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt2_plain {2, logical_tensor::data_type::f32, beta_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt3_plain {3, logical_tensor::data_type::f32, mean_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt4_plain {4, logical_tensor::data_type::f32,
                variance_dims, logical_tensor::layout_type::strided};

        logical_tensor lt5_any {5, logical_tensor::data_type::f32,
                infer_dst_dims, logical_tensor::layout_type::any};
        logical_tensor lt6_plain {6, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::strided};
        logical_tensor lt7_plain {7, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::strided};
        logical_tensor lt8_any {8, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::any};
        logical_tensor lt9_any {9, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::any};

        //compile partition
        std::vector<logical_tensor> in0(
                {lt0_plain, lt1_plain, lt2_plain, lt3_plain, lt4_plain});
        std::vector<logical_tensor> out0(
                {lt5_any, lt6_plain, lt7_plain, lt8_any, lt9_any});

        auto cp = partitions[0].compile(in0, out0, eng);

        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[0], input_dims[0]);
        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[1], input_dims[1]);
        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[2], input_dims[2]);
        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[3], input_dims[3]);

        ASSERT_EQ(cp.query_logical_tensor(6).get_dims()[0], channels);
        ASSERT_EQ(cp.query_logical_tensor(7).get_dims()[0], channels);
        ASSERT_EQ(cp.query_logical_tensor(8).get_dims()[0], channels);
        ASSERT_EQ(cp.query_logical_tensor(9).get_dims()[0], channels);

        std::vector<float> src(product(input_dims), 0.0);
        std::vector<float> gamma(channels, 0.0);
        std::vector<float> beta(channels, 0.0);
        std::vector<float> mean(channels, 0.0);
        std::vector<float> variance(channels, 0.0);
        std::vector<float> dst(product(input_dims), 0.0);
        std::vector<float> running_mean(channels, 0.0);
        std::vector<float> running_variance(channels, 0.0);
        std::vector<float> batch_mean(channels, 0.0);
        std::vector<float> batch_variance(channels, 0.0);

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(-1.0f, 1.0f);
        std::generate(src.begin(), src.end(),
                [&]() { return distribution(generator); });
        std::generate(gamma.begin(), gamma.end(), []() {
            static int i = 0;
            return std::sin(static_cast<float>(i++) * 2.f);
        });
        std::generate(beta.begin(), beta.end(), []() {
            static int i = 0;
            return std::tan(i++);
        });
        std::generate(mean.begin(), mean.end(), []() {
            static int i = 0;
            return std::sin(static_cast<float>(i++) * 2.f);
        });
        std::generate(variance.begin(), variance.end(), []() {
            static int i = 0;
            return std::tan(i++);
        });

        tensor src_ts(lt0_plain, eng, src.data());
        tensor gamma_ts(lt1_plain, eng, gamma.data());
        tensor beta_ts(lt2_plain, eng, beta.data());
        tensor mean_ts(lt3_plain, eng, mean.data());
        tensor variance_ts(lt4_plain, eng, variance.data());

        tensor dst_ts(cp.query_logical_tensor(5), eng, dst.data());
        tensor running_mean_ts(
                cp.query_logical_tensor(6), eng, running_mean.data());
        tensor running_variance_ts(
                cp.query_logical_tensor(7), eng, running_variance.data());
        tensor batch_mean_ts(
                cp.query_logical_tensor(8), eng, batch_mean.data());
        tensor batch_variance_ts(
                cp.query_logical_tensor(9), eng, batch_variance.data());

        cp.execute(strm, {src_ts, gamma_ts, beta_ts, mean_ts, variance_ts},
                {dst_ts, running_mean_ts, running_variance_ts, batch_mean_ts,
                        batch_variance_ts});

        strm.wait();

        // BatchNormTrainingBackprop
        graph g2(engine_kind);
        op bn_bwd_op(0, op::kind::BatchNormTrainingBackprop, "bn_bwd");

        bn_bwd_op.set_attr<std::string>("data_format", data_format);
        bn_bwd_op.set_attr<float>("epsilon", epsilon);

        logical_tensor lt10 {10, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt11 {11, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt12 {12, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::undef};
        logical_tensor lt13 {13, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::undef};

        bn_bwd_op.add_input(lt0);
        bn_bwd_op.add_input(lt10);
        bn_bwd_op.add_input(lt1);
        bn_bwd_op.add_input(cp.query_logical_tensor(8));
        bn_bwd_op.add_input(cp.query_logical_tensor(9));
        bn_bwd_op.add_output(lt11);
        bn_bwd_op.add_output(lt12);
        bn_bwd_op.add_output(lt13);

        g2.add_op(bn_bwd_op);

        auto partitions2 = g2.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions2.size(), 1);

        //get_ops
        std::vector<size_t> ops2 = partitions2[0].get_ops();
        ASSERT_EQ(ops2.size(), 1);
        ASSERT_EQ(partitions2[0].get_ops_num(), 1);

        logical_tensor lt10_plain {10, logical_tensor::data_type::f32,
                input_dims, logical_tensor::layout_type::strided};
        logical_tensor lt11_any {11, logical_tensor::data_type::f32,
                infer_dst_dims, logical_tensor::layout_type::any};
        logical_tensor lt12_plain {12, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::strided};
        logical_tensor lt13_plain {13, logical_tensor::data_type::f32,
                infer_weight_dims, logical_tensor::layout_type::strided};

        //compile partition
        std::vector<logical_tensor> in1({lt0_plain, lt10_plain, lt1_plain,
                cp.query_logical_tensor(8), cp.query_logical_tensor(9)});
        std::vector<logical_tensor> out1({lt11_any, lt12_plain, lt13_plain});

        auto cp2 = partitions2[0].compile(in1, out1, eng);

        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[0], input_dims[0]);
        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[1], input_dims[1]);
        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[2], input_dims[2]);
        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[3], input_dims[3]);

        tensor outgrad_ts(lt10_plain, eng, dst.data());

        tensor ingrad_ts(cp2.query_logical_tensor(11), eng, src.data());
        tensor gammagrad_ts(cp2.query_logical_tensor(12), eng, gamma.data());
        tensor betagrad_ts(cp2.query_logical_tensor(13), eng, beta.data());

        cp2.execute(strm,
                {src_ts, outgrad_ts, gamma_ts, batch_mean_ts,
                        batch_variance_ts},
                {ingrad_ts, gammagrad_ts, betagrad_ts});

        strm.wait();
    }
};

TEST_P(test_bn_compile_t, Test_BatchNorm_Compile) {
    Test_BatchNormTraining();
    Test_BatchNormTraining_Opaque();
}

INSTANTIATE_TEST_SUITE_P(Test_BatchNorm_Compile, test_bn_compile_t,
        ::testing::Values(
                dnnl_graph_test_bn_params_t {{1, 3, 3, 10}, 0.001f, "NXC"}));
