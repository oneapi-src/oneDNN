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

#ifndef DNNL_GRAPH_CPU_SYCL
struct dnnl_graph_test_conv_params_t {
    std::vector<int64_t> input_dims;
    std::vector<int64_t> ref_dst_dims;
    std::vector<int64_t> kernel;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    std::vector<int64_t> pads_begin;
    std::vector<int64_t> pads_end;
    int64_t out_channels;
    std::string auto_pad;
    std::string data_format;
    std::string filter_format;
};

class test_conv_compile_t
    : public ::testing::TestWithParam<dnnl_graph_test_conv_params_t> {
public:
    void Test_Conv() {
        auto params = ::testing::TestWithParam<
                dnnl_graph_test_conv_params_t>::GetParam();

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
        const auto &out_channels = params.out_channels;
        const auto &auto_pad = params.auto_pad;
        const auto &data_format = params.data_format;
        const auto &filter_format = params.filter_format;
        const auto &in_channels
                = (data_format == "NCX") ? input_dims[1] : input_dims.back();

        std::vector<int64_t> filter_dims {-1, -1, -1, -1};
        if (filter_format == "XIO") {
            filter_dims[0] = kernel[0];
            filter_dims[1] = kernel[1];
            filter_dims[2] = in_channels;
            filter_dims[3] = out_channels;
        } else {
            filter_dims[0] = out_channels;
            filter_dims[1] = in_channels;
            filter_dims[2] = kernel[0];
            filter_dims[3] = kernel[1];
        }

        std::vector<int64_t> infer_dst_dims {-1, -1, -1, -1};

        // forward
        logical_tensor lt1 {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt2 {1, logical_tensor::data_type::f32, filter_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt3 {2, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};

        op convfwd_op(0, op::kind::Convolution, "conv_fwd");

        convfwd_op.set_attr<std::vector<int64_t>>("strides", strides);
        convfwd_op.set_attr<std::vector<int64_t>>("dilations", dilations);
        convfwd_op.set_attr<std::vector<int64_t>>("pads_begin", pads_begin);
        convfwd_op.set_attr<std::vector<int64_t>>("pads_end", pads_end);
        convfwd_op.set_attr<std::string>("auto_pad", auto_pad);
        convfwd_op.set_attr<std::string>("data_format", data_format);
        convfwd_op.set_attr<std::string>("filter_format", filter_format);

        convfwd_op.add_input(lt1);
        convfwd_op.add_input(lt2);
        convfwd_op.add_output(lt3);

        g.add_op(convfwd_op);

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
                filter_dims, logical_tensor::layout_type::strided};
        logical_tensor lt3_plain {2, logical_tensor::data_type::f32,
                infer_dst_dims, logical_tensor::layout_type::strided};

        //compile partition
        std::vector<logical_tensor> in0({lt1_plain, lt2_plain});
        std::vector<logical_tensor> out0({lt3_plain});

        auto cp = partitions[0].compile(in0, out0, eng);

        ASSERT_EQ(cp.query_logical_tensor(2).get_dims()[0], ref_dst_dims[0]);
        ASSERT_EQ(cp.query_logical_tensor(2).get_dims()[1], ref_dst_dims[1]);
        ASSERT_EQ(cp.query_logical_tensor(2).get_dims()[2], ref_dst_dims[2]);
        ASSERT_EQ(cp.query_logical_tensor(2).get_dims()[3], ref_dst_dims[3]);

        std::vector<float> src(product(input_dims), 0.0);
        std::vector<float> filter(product(filter_dims), 0.0);
        std::vector<float> dst(product(input_dims), 0.0);

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(-1.0f, 1.0f);
        std::generate(src.begin(), src.end(),
                [&]() { return distribution(generator); });
        std::generate(filter.begin(), filter.end(),
                [&]() { return distribution(generator); });

        tensor src_ts(lt1_plain, eng, src.data());
        tensor filter_ts(lt2_plain, eng, filter.data());
        tensor dst_ts(out0[0], eng, dst.data());

        cp.execute(strm, {src_ts, filter_ts}, {dst_ts});

        strm.wait();

        // backward filter
        graph g1(engine_kind);
        std::vector<int64_t> infer_filter_dims {-1, -1, -1, -1};
        logical_tensor lt4 {3, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt5 {4, logical_tensor::data_type::f32, ref_dst_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt6 {5, logical_tensor::data_type::f32,
                infer_filter_dims, logical_tensor::layout_type::undef};

        op convbwd_filters_op(
                1, op::kind::ConvolutionBackpropFilters, "conv_bwd_filters");

        convbwd_filters_op.set_attr<std::vector<int64_t>>("strides", strides);
        convbwd_filters_op.set_attr<std::vector<int64_t>>(
                "dilations", dilations);
        convbwd_filters_op.set_attr<std::vector<int64_t>>(
                "pads_begin", pads_begin);
        convbwd_filters_op.set_attr<std::vector<int64_t>>("pads_end", pads_end);
        convbwd_filters_op.set_attr<std::string>("auto_pad", auto_pad);
        convbwd_filters_op.set_attr<std::string>("data_format", data_format);
        convbwd_filters_op.set_attr<std::string>(
                "filter_format", filter_format);
        convbwd_filters_op.set_attr<std::vector<int64_t>>(
                "filter_shape", filter_dims);

        convbwd_filters_op.add_input(lt4);
        convbwd_filters_op.add_input(lt5);
        convbwd_filters_op.add_output(lt6);

        g1.add_op(convbwd_filters_op);

        auto partitions1 = g1.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions1.size(), 1);
        std::vector<size_t> ops1 = partitions1[0].get_ops();
        ASSERT_EQ(ops1.size(), 1);
        ASSERT_EQ(partitions1[0].get_ops_num(), 1);

        logical_tensor lt4_plain {3, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt5_plain {4, logical_tensor::data_type::f32,
                ref_dst_dims, logical_tensor::layout_type::strided};
        logical_tensor lt6_plain {5, logical_tensor::data_type::f32,
                infer_filter_dims, logical_tensor::layout_type::strided};

        //compile partition
        std::vector<logical_tensor> in1({lt4_plain, lt5_plain});
        std::vector<logical_tensor> out1({lt6_plain});

        auto cp1 = partitions1[0].compile(in1, out1, eng);

        ASSERT_EQ(cp1.query_logical_tensor(5).get_dims()[0], filter_dims[0]);
        ASSERT_EQ(cp1.query_logical_tensor(5).get_dims()[1], filter_dims[1]);
        ASSERT_EQ(cp1.query_logical_tensor(5).get_dims()[2], filter_dims[2]);
        ASSERT_EQ(cp1.query_logical_tensor(5).get_dims()[3], filter_dims[3]);

        logical_tensor lt6_plain_infered {5, logical_tensor::data_type::f32,
                filter_dims, logical_tensor::layout_type::strided};

        tensor input_ts(lt4_plain, eng, src.data());
        tensor outgrad_ts(lt5_plain, eng, dst.data());
        tensor filtergrad_ts(lt6_plain_infered, eng, filter.data());

        cp1.execute(strm, {input_ts, outgrad_ts}, {filtergrad_ts});

        strm.wait();

        // backward data
        graph g2(engine_kind);
        std::vector<int64_t> infer_data_dims {-1, -1, -1, -1};
        logical_tensor lt7 {6, logical_tensor::data_type::f32, ref_dst_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt8 {7, logical_tensor::data_type::f32, filter_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt9 {8, logical_tensor::data_type::f32, infer_data_dims,
                logical_tensor::layout_type::undef};

        op convbwd_data_op(
                2, op::kind::ConvolutionBackpropData, "conv_bwd_data");

        convbwd_data_op.set_attr<std::vector<int64_t>>("strides", strides);
        convbwd_data_op.set_attr<std::vector<int64_t>>("dilations", dilations);
        convbwd_data_op.set_attr<std::vector<int64_t>>(
                "pads_begin", pads_begin);
        convbwd_data_op.set_attr<std::vector<int64_t>>("pads_end", pads_end);
        convbwd_data_op.set_attr<std::string>("auto_pad", auto_pad);
        convbwd_data_op.set_attr<std::string>("data_format", data_format);
        convbwd_data_op.set_attr<std::string>("filter_format", filter_format);
        convbwd_data_op.set_attr<std::vector<int64_t>>(
                "output_shape", input_dims);

        convbwd_data_op.add_input(lt7);
        convbwd_data_op.add_input(lt8);
        convbwd_data_op.add_output(lt9);

        g2.add_op(convbwd_data_op);

        auto partitions2 = g2.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions2.size(), 1);
        std::vector<size_t> ops2 = partitions2[0].get_ops();
        ASSERT_EQ(ops2.size(), 1);
        ASSERT_EQ(partitions2[0].get_ops_num(), 1);

        logical_tensor lt7_plain {6, logical_tensor::data_type::f32,
                ref_dst_dims, logical_tensor::layout_type::strided};
        logical_tensor lt8_plain {7, logical_tensor::data_type::f32,
                filter_dims, logical_tensor::layout_type::strided};
        logical_tensor lt9_plain {8, logical_tensor::data_type::f32,
                infer_data_dims, logical_tensor::layout_type::strided};

        //compile partition
        std::vector<logical_tensor> in2({lt7_plain, lt8_plain});
        std::vector<logical_tensor> out2({lt9_plain});

        auto cp2 = partitions2[0].compile(in2, out2, eng);

        ASSERT_EQ(cp2.query_logical_tensor(8).get_dims()[0], input_dims[0]);
        ASSERT_EQ(cp2.query_logical_tensor(8).get_dims()[1], input_dims[1]);
        ASSERT_EQ(cp2.query_logical_tensor(8).get_dims()[2], input_dims[2]);
        ASSERT_EQ(cp2.query_logical_tensor(8).get_dims()[3], input_dims[3]);

        tensor outputgrad_ts(lt7_plain, eng, dst.data());
        tensor filter_ts2(lt8_plain, eng, filter.data());
        tensor inputgrad_ts(out2[0], eng, src.data());

        cp2.execute(strm, {outputgrad_ts, filter_ts2}, {inputgrad_ts});

        strm.wait();
    }
};

TEST_P(test_conv_compile_t, Test_Conv_Compile) {
    Test_Conv();
}

INSTANTIATE_TEST_SUITE_P(Test_Conv_Compile, test_conv_compile_t,
        ::testing::Values(
                dnnl_graph_test_conv_params_t {{1, 56, 56, 64}, {1, 56, 56, 64},
                        {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 64,
                        "SAME_UPPER", "NXC", "XIO"},
                dnnl_graph_test_conv_params_t {{1, 14, 14, 1024},
                        {1, 7, 7, 2048}, {1, 1}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                        2048, "SAME_UPPER", "NXC", "XIO"}));

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
    std::string auto_pad;
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
        const auto &auto_pad = params.auto_pad;

        std::vector<float> src(product(input_dims), 0.0);
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(-1.0f, 1.0f);
        std::generate(src.begin(), src.end(),
                [&]() { return distribution(generator); });

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
        pool_op.set_attr<std::string>("auto_pad", auto_pad);

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

        std::vector<float> dst(product(ref_dst_dims), 0.0);

        tensor src_ts(lt1_plain, eng, src.data());
        tensor dst_ts(out0[0], eng, dst.data());

        cp.execute(strm, {src_ts}, {dst_ts});

        strm.wait();
    }
};

TEST_P(test_average_pool_compile_t, Test_Average_Pool_Compile) {
    Test_Average_Pool();
}

INSTANTIATE_TEST_SUITE_P(Test_Average_Pool_Compile, test_average_pool_compile_t,
        ::testing::Values(dnnl_graph_test_pool_params_t {{1, 1, 4, 4},
                                  {1, 1, 2, 3}, {2, 2}, {3, 1}, {0, 0}, {0, 0},
                                  {0, 0}, false, "NCX", "ceil", "None"},
                dnnl_graph_test_pool_params_t {{1, 1, 4, 4}, {1, 1, 2, 3},
                        {2, 2}, {3, 1}, {0, 0}, {0, 0}, {0, 0}, true, "NCX",
                        "ceil", "None"},
                dnnl_graph_test_pool_params_t {{1, 1, 60, 60}, {1, 1, 30, 30},
                        {3, 3}, {2, 2}, {0, 0}, {1, 1}, {0, 0}, false, "NCX",
                        "ceil", "None"}));

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
        const auto &auto_pad = params.auto_pad;

        std::vector<float> src(product(input_dims), 0.0);
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(-1.0f, 1.0f);
        std::generate(src.begin(), src.end(),
                [&]() { return distribution(generator); });

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
        pool_op.set_attr<std::string>("auto_pad", auto_pad);

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

        std::vector<float> dst(product(ref_dst_dims), 0.0);

        tensor src_ts(lt1_plain, eng, src.data());
        tensor dst_ts(out0[0], eng, dst.data());

        cp.execute(strm, {src_ts}, {dst_ts});

        strm.wait();
    }

    void Test_Max_Pool_FWD_BWD() {
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
        const auto &auto_pad = params.auto_pad;

        std::vector<float> src(product(input_dims), 0.0);
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(-1.0f, 1.0f);
        std::generate(src.begin(), src.end(),
                [&]() { return distribution(generator); });

        std::vector<int64_t> infer_dst_dims {-1, -1, -1, -1};

        logical_tensor lt0 {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt1 {1, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};

        op pool_op(0, op::kind::MaxPool, "pool");

        pool_op.set_attr<std::vector<int64_t>>("kernel", kernel);
        pool_op.set_attr<std::vector<int64_t>>("strides", strides);
        pool_op.set_attr<std::vector<int64_t>>("dilations", dilations);
        pool_op.set_attr<std::vector<int64_t>>("pads_begin", pads_begin);
        pool_op.set_attr<std::vector<int64_t>>("pads_end", pads_end);
        pool_op.set_attr<std::string>("data_format", data_format);
        pool_op.set_attr<std::string>("rounding_type", rounding_type);
        pool_op.set_attr<std::string>("auto_pad", auto_pad);

        pool_op.add_input(lt0);
        pool_op.add_output(lt1);

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

        logical_tensor lt0_plain {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt1_plain {1, logical_tensor::data_type::f32,
                infer_dst_dims, logical_tensor::layout_type::strided};

        //compile partition
        std::vector<logical_tensor> in0({lt0_plain});
        std::vector<logical_tensor> out0({lt1_plain});

        auto cp = partitions[0].compile(in0, out0, eng);

        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[0], ref_dst_dims[0]);
        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[1], ref_dst_dims[1]);
        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[2], ref_dst_dims[2]);
        ASSERT_EQ(cp.query_logical_tensor(1).get_dims()[3], ref_dst_dims[3]);

        std::vector<float> dst(product(ref_dst_dims), 0.0);

        tensor src_ts(lt1_plain, eng, src.data());
        tensor dst_ts(out0[0], eng, dst.data());

        cp.execute(strm, {src_ts}, {dst_ts});

        strm.wait();

        graph g1(engine_kind);

        logical_tensor lt2 {2, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt3 {3, logical_tensor::data_type::f32, ref_dst_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt4 {4, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};

        op pool_op1(1, op::kind::MaxPoolBackprop, "poolbackprop");

        pool_op1.set_attr<std::vector<int64_t>>("kernel", kernel);
        pool_op1.set_attr<std::vector<int64_t>>("strides", strides);
        pool_op1.set_attr<std::vector<int64_t>>("dilations", dilations);
        pool_op1.set_attr<std::vector<int64_t>>("pads_begin", pads_begin);
        pool_op1.set_attr<std::vector<int64_t>>("pads_end", pads_end);
        pool_op1.set_attr<std::string>("data_format", data_format);
        pool_op1.set_attr<std::string>("auto_pad", auto_pad);

        pool_op1.add_input(lt2);
        pool_op1.add_input(lt3);
        pool_op1.add_output(lt4);

        g1.add_op(pool_op1);

        //create_partition
        auto partitions1 = g1.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions1.size(), 1);

        // check partition engine kind
        ASSERT_EQ(partitions1[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops1 = partitions1[0].get_ops();
        ASSERT_EQ(ops1.size(), 1);
        ASSERT_EQ(partitions1[0].get_ops_num(), 1);

        logical_tensor lt2_plain {2, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt3_plain {3, logical_tensor::data_type::f32,
                ref_dst_dims, logical_tensor::layout_type::strided};
        logical_tensor lt4_plain {4, logical_tensor::data_type::f32,
                infer_dst_dims, logical_tensor::layout_type::any};

        //compile partition
        std::vector<logical_tensor> in1({lt2_plain, lt3_plain});
        std::vector<logical_tensor> out1({lt4_plain});

        auto cp1 = partitions1[0].compile(in1, out1, eng);

        ASSERT_EQ(cp1.query_logical_tensor(4).get_dims()[0], input_dims[0]);
        ASSERT_EQ(cp1.query_logical_tensor(4).get_dims()[1], input_dims[1]);
        ASSERT_EQ(cp1.query_logical_tensor(4).get_dims()[2], input_dims[2]);
        ASSERT_EQ(cp1.query_logical_tensor(4).get_dims()[3], input_dims[3]);

        std::vector<float> outgrad(src.size(), 0.0);
        tensor src_ts2(lt2_plain, eng, src.data());
        tensor dst_ts2(lt3_plain, eng, dst.data());
        tensor outgrad_ts(cp1.query_logical_tensor(4), eng, outgrad.data());

        cp1.execute(strm, {src_ts2, dst_ts2}, {outgrad_ts});

        strm.wait();
    }
};

TEST_P(test_max_pool_compile_t, Test_Max_Pool_Compile) {
    Test_Max_Pool();
    Test_Max_Pool_FWD_BWD();
}

INSTANTIATE_TEST_SUITE_P(Test_Max_Pool_Compile, test_max_pool_compile_t,
        ::testing::Values(dnnl_graph_test_pool_params_t {{1, 1, 3, 3},
                                  {1, 1, 2, 2}, {2, 2}, {1, 1}, {1, 1}, {0, 0},
                                  {0, 0}, false, "NCX", "ceil", "None"},
                dnnl_graph_test_pool_params_t {{1, 1, 3, 3}, {1, 1, 2, 2},
                        {2, 2}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false, "NCX",
                        "floor", "None"},
                dnnl_graph_test_pool_params_t {{1, 4, 4, 3}, {1, 1, 1, 3},
                        {3, 3}, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, "NXC",
                        "floor", "None"},
                dnnl_graph_test_pool_params_t {{1, 4, 4, 3}, {1, 2, 2, 3},
                        {3, 3}, {2, 2}, {1, 1}, {0, 0}, {0, 0}, false, "NXC",
                        "floor", "SAME_UPPER"}));

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
        logical_tensor lt1 {1, logical_tensor::data_type::f32, mean_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt2 {2, logical_tensor::data_type::f32, variance_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt3 {3, logical_tensor::data_type::f32, gamma_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt4 {4, logical_tensor::data_type::f32, beta_dims,
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
        logical_tensor lt1_plain {1, logical_tensor::data_type::f32, mean_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt2_plain {2, logical_tensor::data_type::f32,
                variance_dims, logical_tensor::layout_type::strided};
        logical_tensor lt3_plain {3, logical_tensor::data_type::f32, gamma_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt4_plain {4, logical_tensor::data_type::f32, beta_dims,
                logical_tensor::layout_type::strided};

        tensor src_ts(lt0_plain, eng, src.data());
        tensor mean_ts(lt1_plain, eng, gamma.data());
        tensor variance_ts(lt2_plain, eng, beta.data());
        tensor gamma_ts(lt3_plain, eng, mean.data());
        tensor beta_ts(lt4_plain, eng, variance.data());

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

        cp.execute(strm, {src_ts, mean_ts, variance_ts, gamma_ts, beta_ts},
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
        bn_bwd_op.add_input(dst3_lt_infered);
        bn_bwd_op.add_input(dst4_lt_infered);
        bn_bwd_op.add_input(lt3);
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
                {lt0, lt10, dst3_lt_infered, dst4_lt_infered, lt3});
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
                {src_ts, outgrad_ts, batch_mean_ts, batch_variance_ts,
                        gamma_ts},
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
        logical_tensor lt1 {1, logical_tensor::data_type::f32, mean_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt2 {2, logical_tensor::data_type::f32, variance_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt3 {3, logical_tensor::data_type::f32, gamma_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt4 {4, logical_tensor::data_type::f32, beta_dims,
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
        logical_tensor lt1_plain {1, logical_tensor::data_type::f32, mean_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt2_plain {2, logical_tensor::data_type::f32,
                variance_dims, logical_tensor::layout_type::strided};
        logical_tensor lt3_plain {3, logical_tensor::data_type::f32, gamma_dims,
                logical_tensor::layout_type::strided};
        logical_tensor lt4_plain {4, logical_tensor::data_type::f32, beta_dims,
                logical_tensor::layout_type::strided};

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
        tensor mean_ts(lt1_plain, eng, gamma.data());
        tensor variance_ts(lt2_plain, eng, beta.data());
        tensor gamma_ts(lt3_plain, eng, mean.data());
        tensor beta_ts(lt4_plain, eng, variance.data());

        tensor dst_ts(cp.query_logical_tensor(5), eng, dst.data());
        tensor running_mean_ts(
                cp.query_logical_tensor(6), eng, running_mean.data());
        tensor running_variance_ts(
                cp.query_logical_tensor(7), eng, running_variance.data());
        tensor batch_mean_ts(
                cp.query_logical_tensor(8), eng, batch_mean.data());
        tensor batch_variance_ts(
                cp.query_logical_tensor(9), eng, batch_variance.data());

        cp.execute(strm, {src_ts, mean_ts, variance_ts, gamma_ts, beta_ts},
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
        bn_bwd_op.add_input(cp.query_logical_tensor(8));
        bn_bwd_op.add_input(cp.query_logical_tensor(9));
        bn_bwd_op.add_input(lt3);
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
        std::vector<logical_tensor> in1(
                {lt0_plain, lt10_plain, cp.query_logical_tensor(8),
                        cp.query_logical_tensor(9), lt3_plain});
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
                {src_ts, outgrad_ts, batch_mean_ts, batch_variance_ts,
                        gamma_ts},
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
#endif
