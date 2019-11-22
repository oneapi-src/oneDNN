/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "dnnl.hpp"

namespace dnnl {

class attr_test : public ::testing::Test {
protected:
    virtual void SetUp() {}
};

TEST_F(attr_test, TestScratchpadMode) {
    dnnl::primitive_attr attr;
    for (auto m : {scratchpad_mode::library, scratchpad_mode::user}) {
        attr.set_scratchpad_mode(m);
        ASSERT_EQ(m, attr.get_scratchpad_mode());
    }
}

TEST_F(attr_test, TestScratchpadModeEx) {
    engine eng = get_test_engine();

    const memory::dim N = 2, C = 2, W = 2;

    memory::desc data_md(
            {N, C, W}, memory::data_type::f32, memory::format_tag::ncw);

    dnnl::primitive_attr attr;
    auto softmax_d
            = softmax_forward::desc(prop_kind::forward_inference, data_md, 1);
    for (auto m : {scratchpad_mode::library, scratchpad_mode::user}) {
        attr.set_scratchpad_mode(m);
        auto softmax_pd = softmax_forward::primitive_desc(softmax_d, attr, eng);
        auto scratchpad_size = (long)softmax_pd.scratchpad_desc().get_size();
        auto mem_consumption
                = (long)softmax_pd.query_s64(query::memory_consumption_s64);

        // printf("scratchpad_size: %ld\n", scratchpad_size);
        // printf("mem consumption: %ld\n", mem_consumption);

        if (m == scratchpad_mode::library) {
            ASSERT_EQ(scratchpad_size, 0L);
        } else {
            ASSERT_EQ(mem_consumption, 0L);
        }
    }
}

TEST_F(attr_test, TestScratchpadArg) {
    engine eng = get_test_engine();

    const memory::dim N = 2, C = 2, W = 2;

    memory::desc data_md(
            {N, C, W}, memory::data_type::f32, memory::format_tag::ncw);

    dnnl::primitive_attr attr;
    auto softmax_d
            = softmax_forward::desc(prop_kind::forward_inference, data_md, 1);
    for (auto m : {scratchpad_mode::library, scratchpad_mode::user}) {
        attr.set_scratchpad_mode(m);
        auto softmax_pd = softmax_forward::primitive_desc(softmax_d, attr, eng);

        memory src(softmax_pd.src_desc(), eng);
        memory dst(softmax_pd.dst_desc(), eng);
        memory scratchpad(softmax_pd.scratchpad_desc(), eng);

        fill_data<float>(src.get_desc().get_size() / sizeof(float), src);

        stream s(eng);

        softmax_forward softmax_p(softmax_pd);
        softmax_p.execute(s,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst},
                        {DNNL_ARG_SCRATCHPAD, scratchpad}});
        s.wait();
    }
}

TEST_F(attr_test, TestIntOutputScales) {
    dnnl::primitive_attr attr;

    int mask;
    std::vector<float> scales;

    // default scales
    attr.get_output_scales(mask, scales);
    ASSERT_EQ(mask, 0);
    ASSERT_EQ(scales.size(), 1U);
    ASSERT_EQ(scales[0], 1.);

    // single non-default scale
    attr.set_output_scales(0, {2.});
    attr.get_output_scales(mask, scales);
    ASSERT_EQ(mask, 0);
    ASSERT_EQ(scales.size(), 1U);
    ASSERT_EQ(scales[0], 2.);

    // multiple scales
    attr.set_output_scales(1 << 1, {1., 2., 3.});
    attr.get_output_scales(mask, scales);
    ASSERT_EQ(mask, 1 << 1);
    ASSERT_EQ(scales.size(), 3U);
    ASSERT_EQ(scales[0], 1.);
    ASSERT_EQ(scales[1], 2.);
    ASSERT_EQ(scales[2], 3.);
}

TEST_F(attr_test, TestZeroPoints) {
    dnnl::primitive_attr attr;

    const std::vector<int> supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
    const std::vector<int> unsupported_args = {DNNL_ARG_BIAS, DNNL_ARG_DST_2,
            DNNL_ARG_MEAN, DNNL_ARG_WORKSPACE, DNNL_ARG_SCRATCHPAD};
    int zero_points_mask;
    std::vector<int> zero_points;

    // default zero points
    for (int arg : supported_args) {
        attr.get_zero_points(arg, zero_points_mask, zero_points);
        ASSERT_EQ(zero_points_mask, 0);
        ASSERT_EQ(zero_points.size(), 1U);
        ASSERT_EQ(zero_points[0], 0);
    }

    for (int arg : unsupported_args) {
        attr.get_zero_points(arg, zero_points_mask, zero_points);
        ASSERT_EQ(zero_points_mask, 0);
        ASSERT_EQ(zero_points.size(), 1U);
        ASSERT_EQ(zero_points[0], 0);
    }

    // single non-default zero_point for supported arg
    attr.set_zero_points(supported_args[0], 0, {2});
    attr.get_zero_points(supported_args[0], zero_points_mask, zero_points);
    ASSERT_EQ(zero_points_mask, 0);
    ASSERT_EQ(zero_points.size(), 1U);
    ASSERT_EQ(zero_points[0], 2);

    // single **default** zero_point for **unsupported** arg
    attr.set_zero_points(unsupported_args[0], 0, {0});
    attr.get_zero_points(unsupported_args[0], zero_points_mask, zero_points);
    ASSERT_EQ(zero_points_mask, 0);
    ASSERT_EQ(zero_points.size(), 1U);
    ASSERT_EQ(zero_points[0], 0);

    // multiple zero_points not implemented yet ...
}

TEST_F(attr_test, TestZeroPointsExpectFailure) {
    dnnl::primitive_attr attr;

    const int supported_arg = DNNL_ARG_SRC;
    const int unsupported_arg = DNNL_ARG_MEAN;

    // single non-default zero_point for unsupported arg
    EXPECT_ANY_THROW(attr.set_zero_points(unsupported_arg, 0, {2}));

    // multiple zero points for supported and unsupported args
    EXPECT_ANY_THROW(attr.set_zero_points(supported_arg, 1 << 1, {1, 2, 3}));
    EXPECT_ANY_THROW(attr.set_zero_points(unsupported_arg, 1 << 1, {1, 2, 3}));
}

TEST_F(attr_test, TestPostOps) {
    dnnl::primitive_attr attr;
    dnnl::post_ops ops;

    algorithm alg;
    float scale, alpha, beta;

    ASSERT_EQ(ops.len(), 0);
    ASSERT_EQ(attr.get_post_ops().len(), 0);

    ops.append_sum(1.1f);
    attr.set_post_ops(ops);

    ASSERT_EQ(attr.get_post_ops().len(), 1);
    ASSERT_EQ(attr.get_post_ops().kind(0), primitive::kind::sum);
    attr.get_post_ops().get_params_sum(0, scale);
    ASSERT_FLOAT_EQ(scale, 1.1f);

    ops.append_eltwise(2.2f, algorithm::eltwise_bounded_relu, 3.3f, 4.4f);
    attr.set_post_ops(ops);

    ASSERT_EQ(attr.get_post_ops().len(), 2);
    ASSERT_EQ(attr.get_post_ops().kind(0), primitive::kind::sum);
    ASSERT_EQ(attr.get_post_ops().kind(1), primitive::kind::eltwise);
    attr.get_post_ops().get_params_eltwise(1, scale, alg, alpha, beta);
    ASSERT_FLOAT_EQ(scale, 2.2f);
    ASSERT_EQ(alg, algorithm::eltwise_bounded_relu);
    ASSERT_FLOAT_EQ(alpha, 3.3f);
    ASSERT_FLOAT_EQ(beta, 4.4f);
}

} // namespace dnnl
