/*******************************************************************************
* Copyright 2021 Intel Corporation
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

struct dnnl_graph_test_average_pool_params_t {
    std::vector<int64_t> input_dims;
    std::vector<int64_t> ref_dst_dims;
    std::vector<int64_t> kernel;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads_begin;
    std::vector<int64_t> pads_end;
    bool exclude_pad;
    std::string data_format;
    std::string rounding_type;

    std::vector<float> src;
    std::vector<float> ref_dst;
};

class test_average_pool_compile_t
    : public ::testing::TestWithParam<dnnl_graph_test_average_pool_params_t> {
public:
    void Test_Average_Pool() {
        auto params = ::testing::TestWithParam<
                dnnl_graph_test_average_pool_params_t>::GetParam();

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
        ::testing::Values(dnnl_graph_test_average_pool_params_t {{1, 1, 4, 4},
                                  {1, 1, 2, 3}, {2, 2}, {3, 1}, {0, 0}, {0, 0},
                                  false, "NCX", "ceil",
                                  {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f,
                                          8.0f, 9.0f, 11.0f, 10.0f, 12.0f,
                                          13.0f, 15.0f, 14.0f, 16.0f},
                                  {4.0f, 4.5f, 5.0f, 14.0f, 14.5f, 15.0f}},
                dnnl_graph_test_average_pool_params_t {{1, 1, 4, 4},
                        {1, 1, 2, 3}, {2, 2}, {3, 1}, {0, 0}, {0, 0}, true,
                        "NCX", "ceil",
                        {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f, 8.0f, 9.0f,
                                11.0f, 10.0f, 12.0f, 13.0f, 15.0f, 14.0f,
                                16.0f},
                        {4.0f, 4.5f, 5.0f, 14.0f, 14.5f, 15.0f}}));
