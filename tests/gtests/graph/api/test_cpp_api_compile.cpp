/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

struct test_conv_params_t {
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
    : public ::testing::TestWithParam<test_conv_params_t> {
public:
    void Test_Conv() {
        auto params = ::testing::TestWithParam<test_conv_params_t>::GetParam();

        using namespace dnnl::graph;
        dnnl::engine::kind engine_kind
                = static_cast<dnnl::engine::kind>(api_test_engine_kind);
        dnnl::engine eng = cpp_api_test_dnnl_engine_create(engine_kind);

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

        std::vector<int64_t> filter_dims(4, DNNL_GRAPH_UNKNOWN_DIM);
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

        std::vector<int64_t> infer_dst_dims(4, DNNL_GRAPH_UNKNOWN_DIM);

        // forward
        logical_tensor lt1 {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt2 {1, logical_tensor::data_type::f32, filter_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt3 {2, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};

        op convfwd_op(0, op::kind::Convolution);

        convfwd_op.set_attr<std::vector<int64_t>>(op::attr::strides, strides);
        convfwd_op.set_attr<std::vector<int64_t>>(
                op::attr::dilations, dilations);
        convfwd_op.set_attr<std::vector<int64_t>>(
                op::attr::pads_begin, pads_begin);
        convfwd_op.set_attr<std::vector<int64_t>>(op::attr::pads_end, pads_end);
        convfwd_op.set_attr<std::string>(op::attr::auto_pad, auto_pad);
        convfwd_op.set_attr<std::string>(op::attr::data_format, data_format);
        convfwd_op.set_attr<std::string>(
                op::attr::weights_format, filter_format);

        convfwd_op.add_input(lt1);
        convfwd_op.add_input(lt2);
        convfwd_op.add_output(lt3);

        g.add_op(convfwd_op);
        g.finalize();

        //create_partition
        auto partitions = g.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions.size(), 1U);

        // check partition engine kind
        ASSERT_EQ(partitions[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops = partitions[0].get_ops();
        ASSERT_EQ(ops.size(), 1U);
        ASSERT_EQ(partitions[0].get_ops_num(), 1U);

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

        // backward filter
        graph g1(engine_kind);
        std::vector<int64_t> infer_filter_dims(4, DNNL_GRAPH_UNKNOWN_DIM);
        logical_tensor lt4 {3, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt5 {4, logical_tensor::data_type::f32, ref_dst_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt6 {5, logical_tensor::data_type::f32,
                infer_filter_dims, logical_tensor::layout_type::undef};

        op convbwd_filters_op(
                1, op::kind::ConvolutionBackwardWeights, "conv_bwd_filters");

        convbwd_filters_op.set_attr<std::vector<int64_t>>(
                op::attr::strides, strides);
        convbwd_filters_op.set_attr<std::vector<int64_t>>(
                op::attr::dilations, dilations);
        convbwd_filters_op.set_attr<std::vector<int64_t>>(
                op::attr::pads_begin, pads_begin);
        convbwd_filters_op.set_attr<std::vector<int64_t>>(
                op::attr::pads_end, pads_end);
        convbwd_filters_op.set_attr<std::string>(op::attr::auto_pad, auto_pad);
        convbwd_filters_op.set_attr<std::string>(
                op::attr::data_format, data_format);
        convbwd_filters_op.set_attr<std::string>(
                op::attr::weights_format, filter_format);
        convbwd_filters_op.set_attr<std::vector<int64_t>>(
                op::attr::weights_shape, filter_dims);

        convbwd_filters_op.add_input(lt4);
        convbwd_filters_op.add_input(lt5);
        convbwd_filters_op.add_output(lt6);

        g1.add_op(convbwd_filters_op);
        g1.finalize();

        auto partitions1 = g1.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions1.size(), 1U);
        std::vector<size_t> ops1 = partitions1[0].get_ops();
        ASSERT_EQ(ops1.size(), 1U);
        ASSERT_EQ(partitions1[0].get_ops_num(), 1U);

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

        // backward data
        graph g2(engine_kind);
        std::vector<int64_t> infer_data_dims(4, DNNL_GRAPH_UNKNOWN_DIM);
        logical_tensor lt7 {6, logical_tensor::data_type::f32, ref_dst_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt8 {7, logical_tensor::data_type::f32, filter_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt9 {8, logical_tensor::data_type::f32, infer_data_dims,
                logical_tensor::layout_type::undef};

        op convbwd_data_op(
                2, op::kind::ConvolutionBackwardData, "conv_bwd_data");

        convbwd_data_op.set_attr<std::vector<int64_t>>(
                op::attr::strides, strides);
        convbwd_data_op.set_attr<std::vector<int64_t>>(
                op::attr::dilations, dilations);
        convbwd_data_op.set_attr<std::vector<int64_t>>(
                op::attr::pads_begin, pads_begin);
        convbwd_data_op.set_attr<std::vector<int64_t>>(
                op::attr::pads_end, pads_end);
        convbwd_data_op.set_attr<std::string>(op::attr::auto_pad, auto_pad);
        convbwd_data_op.set_attr<std::string>(
                op::attr::data_format, data_format);
        convbwd_data_op.set_attr<std::string>(
                op::attr::weights_format, filter_format);
        convbwd_data_op.set_attr<std::vector<int64_t>>(
                op::attr::dst_shape, input_dims);

        convbwd_data_op.add_input(lt7);
        convbwd_data_op.add_input(lt8);
        convbwd_data_op.add_output(lt9);

        g2.add_op(convbwd_data_op);
        g2.finalize();

        auto partitions2 = g2.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions2.size(), 1U);
        std::vector<size_t> ops2 = partitions2[0].get_ops();
        ASSERT_EQ(ops2.size(), 1U);
        ASSERT_EQ(partitions2[0].get_ops_num(), 1U);

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
    }
};

TEST_P(test_conv_compile_t, Test_Conv_Compile) {
    Test_Conv();
}

INSTANTIATE_TEST_SUITE_P(Test_Conv_Compile, test_conv_compile_t,
        ::testing::Values(test_conv_params_t {{1, 56, 56, 64}, {1, 56, 56, 64},
                                  {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 64,
                                  "SAME_UPPER", "NXC", "XIO"},
                test_conv_params_t {{1, 14, 14, 1024}, {1, 7, 7, 2048}, {1, 1},
                        {2, 2}, {1, 1}, {0, 0}, {0, 0}, 2048, "SAME_UPPER",
                        "NXC", "XIO"}));

struct pool_params_t {
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
    : public ::testing::TestWithParam<pool_params_t> {
public:
    void Test_Average_Pool() {
        auto params = ::testing::TestWithParam<pool_params_t>::GetParam();

        using namespace dnnl::graph;
        dnnl::engine::kind engine_kind
                = static_cast<dnnl::engine::kind>(api_test_engine_kind);
        dnnl::engine eng = cpp_api_test_dnnl_engine_create(engine_kind);

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

        std::vector<int64_t> infer_dst_dims(4, DNNL_GRAPH_UNKNOWN_DIM);

        logical_tensor lt1 {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt2 {1, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};

        op pool_op(0, op::kind::AvgPool, "pool");

        pool_op.set_attr<std::vector<int64_t>>(op::attr::kernel, kernel);
        pool_op.set_attr<std::vector<int64_t>>(op::attr::strides, strides);
        pool_op.set_attr<std::vector<int64_t>>(
                op::attr::pads_begin, pads_begin);
        pool_op.set_attr<std::vector<int64_t>>(op::attr::pads_end, pads_end);
        pool_op.set_attr<std::string>(op::attr::data_format, data_format);
        pool_op.set_attr<std::string>(op::attr::rounding_type, rounding_type);
        pool_op.set_attr<bool>(op::attr::exclude_pad, exclude_pad);
        pool_op.set_attr<std::string>(op::attr::auto_pad, auto_pad);

        pool_op.add_input(lt1);
        pool_op.add_output(lt2);

        g.add_op(pool_op);
        g.finalize();

        //create_partition
        auto partitions = g.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions.size(), 1U);

        // check partition engine kind
        ASSERT_EQ(partitions[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops = partitions[0].get_ops();
        ASSERT_EQ(ops.size(), 1U);
        ASSERT_EQ(partitions[0].get_ops_num(), 1U);

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
    }
};

TEST_P(test_average_pool_compile_t, Test_Average_Pool_Compile) {
    Test_Average_Pool();
}

INSTANTIATE_TEST_SUITE_P(Test_Average_Pool_Compile, test_average_pool_compile_t,
        ::testing::Values(
                pool_params_t {{1, 1, 4, 4}, {1, 1, 1, 3}, {2, 2}, {3, 1},
                        {0, 0}, {0, 0}, {0, 0}, false, "NCX", "floor", "None"},
                pool_params_t {{1, 1, 4, 4}, {1, 1, 2, 3}, {2, 2}, {3, 1},
                        {0, 0}, {0, 0}, {0, 0}, true, "NCX", "ceil", "None"},
                pool_params_t {{1, 1, 60, 60}, {1, 1, 30, 30}, {3, 3}, {2, 2},
                        {0, 0}, {1, 1}, {0, 0}, false, "NCX", "floor",
                        "None"}));

class test_max_pool_compile_t : public ::testing::TestWithParam<pool_params_t> {
public:
    void Test_Max_Pool() {
        auto params = ::testing::TestWithParam<pool_params_t>::GetParam();

        using namespace dnnl::graph;
        dnnl::engine::kind engine_kind
                = static_cast<dnnl::engine::kind>(api_test_engine_kind);
        dnnl::engine eng = cpp_api_test_dnnl_engine_create(engine_kind);

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

        std::vector<int64_t> infer_dst_dims(4, DNNL_GRAPH_UNKNOWN_DIM);

        logical_tensor lt1 {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt2 {1, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};

        op pool_op(0, op::kind::MaxPool, "pool");

        pool_op.set_attr<std::vector<int64_t>>(op::attr::kernel, kernel);
        pool_op.set_attr<std::vector<int64_t>>(op::attr::strides, strides);
        pool_op.set_attr<std::vector<int64_t>>(op::attr::dilations, dilations);
        pool_op.set_attr<std::vector<int64_t>>(
                op::attr::pads_begin, pads_begin);
        pool_op.set_attr<std::vector<int64_t>>(op::attr::pads_end, pads_end);
        pool_op.set_attr<std::string>(op::attr::data_format, data_format);
        pool_op.set_attr<std::string>(op::attr::rounding_type, rounding_type);
        pool_op.set_attr<std::string>(op::attr::auto_pad, auto_pad);

        pool_op.add_input(lt1);
        pool_op.add_output(lt2);

        g.add_op(pool_op);
        g.finalize();

        //create_partition
        auto partitions = g.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions.size(), 1U);

        // check partition engine kind
        ASSERT_EQ(partitions[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops = partitions[0].get_ops();
        ASSERT_EQ(ops.size(), 1U);
        ASSERT_EQ(partitions[0].get_ops_num(), 1U);

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
    }

    void Test_Max_Pool_FWD_BWD() {
        auto params = ::testing::TestWithParam<pool_params_t>::GetParam();

        using namespace dnnl::graph;
        dnnl::engine::kind engine_kind
                = static_cast<dnnl::engine::kind>(api_test_engine_kind);
        dnnl::engine eng = cpp_api_test_dnnl_engine_create(engine_kind);

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

        std::vector<int64_t> infer_dst_dims(4, DNNL_GRAPH_UNKNOWN_DIM);

        logical_tensor lt0 {0, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt1 {1, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};

        op pool_op(0, op::kind::MaxPool);

        pool_op.set_attr<std::vector<int64_t>>(op::attr::kernel, kernel);
        pool_op.set_attr<std::vector<int64_t>>(op::attr::strides, strides);
        pool_op.set_attr<std::vector<int64_t>>(op::attr::dilations, dilations);
        pool_op.set_attr<std::vector<int64_t>>(
                op::attr::pads_begin, pads_begin);
        pool_op.set_attr<std::vector<int64_t>>(op::attr::pads_end, pads_end);
        pool_op.set_attr<std::string>(op::attr::data_format, data_format);
        pool_op.set_attr<std::string>(op::attr::rounding_type, rounding_type);
        pool_op.set_attr<std::string>(op::attr::auto_pad, auto_pad);

        pool_op.add_input(lt0);
        pool_op.add_output(lt1);

        g.add_op(pool_op);
        g.finalize();

        //create_partition
        auto partitions = g.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions.size(), 1U);

        // check partition engine kind
        ASSERT_EQ(partitions[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops = partitions[0].get_ops();
        ASSERT_EQ(ops.size(), 1U);
        ASSERT_EQ(partitions[0].get_ops_num(), 1U);

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

        graph g1(engine_kind);

        logical_tensor lt2 {2, logical_tensor::data_type::f32, input_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt3 {3, logical_tensor::data_type::f32, ref_dst_dims,
                logical_tensor::layout_type::undef};
        logical_tensor lt4 {4, logical_tensor::data_type::f32, infer_dst_dims,
                logical_tensor::layout_type::undef};

        op pool_op1(1, op::kind::MaxPoolBackward, "poolbackprop");

        pool_op1.set_attr<std::vector<int64_t>>(op::attr::kernel, kernel);
        pool_op1.set_attr<std::vector<int64_t>>(op::attr::strides, strides);
        pool_op1.set_attr<std::vector<int64_t>>(op::attr::dilations, dilations);
        pool_op1.set_attr<std::vector<int64_t>>(
                op::attr::pads_begin, pads_begin);
        pool_op1.set_attr<std::vector<int64_t>>(op::attr::pads_end, pads_end);
        pool_op1.set_attr<std::string>(op::attr::data_format, data_format);
        pool_op1.set_attr<std::string>(op::attr::auto_pad, auto_pad);

        pool_op1.add_input(lt2);
        pool_op1.add_input(lt3);
        pool_op1.add_output(lt4);

        g1.add_op(pool_op1);
        g1.finalize();

        //create_partition
        auto partitions1 = g1.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions1.size(), 1U);

        // check partition engine kind
        ASSERT_EQ(partitions1[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops1 = partitions1[0].get_ops();
        ASSERT_EQ(ops1.size(), 1U);
        ASSERT_EQ(partitions1[0].get_ops_num(), 1U);

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
    }
};

TEST_P(test_max_pool_compile_t, Test_Max_Pool_Compile) {
    Test_Max_Pool();
    Test_Max_Pool_FWD_BWD();
}

INSTANTIATE_TEST_SUITE_P(Test_Max_Pool_Compile, test_max_pool_compile_t,
        ::testing::Values(
                pool_params_t {{1, 1, 3, 3}, {1, 1, 2, 2}, {2, 2}, {1, 1},
                        {1, 1}, {0, 0}, {0, 0}, false, "NCX", "ceil", "None"},
                pool_params_t {{1, 1, 3, 3}, {1, 1, 2, 2}, {2, 2}, {1, 1},
                        {1, 1}, {0, 0}, {0, 0}, false, "NCX", "floor", "None"},
                pool_params_t {{1, 4, 4, 3}, {1, 1, 1, 3}, {3, 3}, {2, 2},
                        {1, 1}, {0, 0}, {0, 0}, false, "NXC", "floor", "None"},
                pool_params_t {{1, 4, 4, 3}, {1, 2, 2, 3}, {3, 3}, {2, 2},
                        {1, 1}, {0, 0}, {0, 0}, false, "NXC", "floor",
                        "SAME_UPPER"}));

struct bn_params_t {
    std::vector<int64_t> input_dims;
    float epsilon;
    std::string data_format;
};

class test_bn_compile_t : public ::testing::TestWithParam<bn_params_t> {
public:
    void Test_BatchNormTraining() {
        auto params = ::testing::TestWithParam<bn_params_t>::GetParam();

        using namespace dnnl::graph;
        dnnl::engine::kind engine_kind
                = static_cast<dnnl::engine::kind>(api_test_engine_kind);
        dnnl::engine eng = cpp_api_test_dnnl_engine_create(engine_kind);

        graph g(engine_kind);

        const auto &input_dims = params.input_dims;
        const auto &epsilon = params.epsilon;
        const auto &data_format = params.data_format;

        const auto &channels
                = (data_format == "NCX") ? input_dims[1] : input_dims.back();

        std::vector<int64_t> infer_dst_dims(4, DNNL_GRAPH_UNKNOWN_DIM);
        std::vector<int64_t> infer_weight_dims(1, DNNL_GRAPH_UNKNOWN_DIM);

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

        bn_fwd_op.set_attr<std::string>(op::attr::data_format, data_format);
        bn_fwd_op.set_attr<float>(op::attr::epsilon, epsilon);
        bn_fwd_op.set_attr<float>(op::attr::momentum, 0.1f);

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
        g.finalize();

        //create_partition
        auto partitions = g.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions.size(), 1U);

        // check partition engine kind
        ASSERT_EQ(partitions[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops = partitions[0].get_ops();
        ASSERT_EQ(ops.size(), 1U);
        ASSERT_EQ(partitions[0].get_ops_num(), 1U);

        //compile partition
        std::vector<logical_tensor> in0({lt0, lt1, lt2, lt3, lt4});
        std::vector<logical_tensor> out0({lt5, lt6, lt7, lt8, lt9});

        auto cp = partitions[0].compile(in0, out0, eng);

        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[0], input_dims[0]);
        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[1], input_dims[1]);
        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[2], input_dims[2]);
        ASSERT_EQ(cp.query_logical_tensor(5).get_dims()[3], input_dims[3]);

        logical_tensor dst3_lt_inferred {8, logical_tensor::data_type::f32,
                mean_dims, logical_tensor::layout_type::strided};
        logical_tensor dst4_lt_inferred {9, logical_tensor::data_type::f32,
                variance_dims, logical_tensor::layout_type::strided};

        // BatchNormTrainingBackward
        graph g2(engine_kind);
        op bn_bwd_op(0, op::kind::BatchNormTrainingBackward, "bn_bwd");

        bn_bwd_op.set_attr<std::string>(op::attr::data_format, data_format);
        bn_bwd_op.set_attr<float>(op::attr::epsilon, epsilon);

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
        bn_bwd_op.add_input(dst3_lt_inferred);
        bn_bwd_op.add_input(dst4_lt_inferred);
        bn_bwd_op.add_input(lt3);
        bn_bwd_op.add_output(lt11);
        bn_bwd_op.add_output(lt12);
        bn_bwd_op.add_output(lt13);

        g2.add_op(bn_bwd_op);
        g2.finalize();

        auto partitions2 = g2.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions2.size(), 1U);

        //get_ops
        std::vector<size_t> ops2 = partitions2[0].get_ops();
        ASSERT_EQ(ops2.size(), 1U);
        ASSERT_EQ(partitions2[0].get_ops_num(), 1U);

        //compile partition
        std::vector<logical_tensor> in1(
                {lt0, lt10, dst3_lt_inferred, dst4_lt_inferred, lt3});
        std::vector<logical_tensor> out1({lt11, lt12, lt13});

        auto cp2 = partitions2[0].compile(in1, out1, eng);

        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[0], input_dims[0]);
        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[1], input_dims[1]);
        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[2], input_dims[2]);
        ASSERT_EQ(cp2.query_logical_tensor(11).get_dims()[3], input_dims[3]);
    }

    void Test_BatchNormTraining_Opaque() {
        auto params = ::testing::TestWithParam<bn_params_t>::GetParam();

        using namespace dnnl::graph;
        dnnl::engine::kind engine_kind
                = static_cast<dnnl::engine::kind>(api_test_engine_kind);
        dnnl::engine eng = cpp_api_test_dnnl_engine_create(engine_kind);

        graph g(engine_kind);

        const auto &input_dims = params.input_dims;
        const auto &epsilon = params.epsilon;
        const auto &data_format = params.data_format;

        const auto &channels
                = (data_format == "NCX") ? input_dims[1] : input_dims.back();

        std::vector<int64_t> infer_dst_dims(4, DNNL_GRAPH_UNKNOWN_DIM);
        std::vector<int64_t> infer_weight_dims(1, DNNL_GRAPH_UNKNOWN_DIM);

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

        bn_fwd_op.set_attr<std::string>(op::attr::data_format, data_format);
        bn_fwd_op.set_attr<float>(op::attr::epsilon, epsilon);
        bn_fwd_op.set_attr<float>(op::attr::momentum, 0.1f);

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
        g.finalize();

        //create_partition
        auto partitions = g.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions.size(), 1U);

        // check partition engine kind
        ASSERT_EQ(partitions[0].get_engine_kind(), engine_kind);

        //get_ops
        std::vector<size_t> ops = partitions[0].get_ops();
        ASSERT_EQ(ops.size(), 1U);
        ASSERT_EQ(partitions[0].get_ops_num(), 1U);

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

        // BatchNormTrainingBackward
        graph g2(engine_kind);
        op bn_bwd_op(0, op::kind::BatchNormTrainingBackward, "bn_bwd");

        bn_bwd_op.set_attr<std::string>(op::attr::data_format, data_format);
        bn_bwd_op.set_attr<float>(op::attr::epsilon, epsilon);

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
        g2.finalize();

        auto partitions2 = g2.get_partitions(partition::policy::fusion);
        ASSERT_EQ(partitions2.size(), 1U);

        //get_ops
        std::vector<size_t> ops2 = partitions2[0].get_ops();
        ASSERT_EQ(ops2.size(), 1U);
        ASSERT_EQ(partitions2[0].get_ops_num(), 1U);

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
    }
};

TEST_P(test_bn_compile_t, Test_BatchNorm_Compile) {
    Test_BatchNormTraining();
    Test_BatchNormTraining_Opaque();
}

INSTANTIATE_TEST_SUITE_P(Test_BatchNorm_Compile, test_bn_compile_t,
        ::testing::Values(bn_params_t {{1, 3, 3, 10}, 0.001f, "NXC"}));
