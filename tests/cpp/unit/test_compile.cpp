/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include "interface/backend.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/operators/batchnorm.hpp"
#include "backend/dnnl/operators/conv.hpp"
#include "backend/dnnl/operators/eltwise.hpp"
#include "backend/dnnl/operators/inner_product.hpp"
#include "backend/dnnl/operators/layernorm.hpp"
#include "backend/dnnl/operators/matmul.hpp"
#include "backend/dnnl/operators/pool.hpp"
#include "backend/dnnl/operators/reorder.hpp"
#include "backend/dnnl/operators/softmax.hpp"

#include "unit_test_common.hpp"

#include "dnnl.h"
#include "utils.hpp"

namespace impl = dnnl::graph::impl;
namespace dnnl_impl = impl::dnnl_impl;
namespace utils = dnnl::graph::tests::unit::utils;

static dnnl_impl::kernel_registry &get_dnnl_kernel_registry() {
    return dnnl_impl::dnnl_backend::get_singleton().get_kernel_registry();
}

TEST(operator_compile, convolution_compile_fp32) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();

    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, impl::data_type::f32);
    impl::logical_tensor_t weight = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {8, 16, 222, 222}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    size_t operator_num = get_dnnl_kernel_registry().get_register_kernels_num();
    ASSERT_NE(operator_num, 0);

    auto kernel = get_dnnl_kernel_registry().create_kernel(conv_op);
    ASSERT_TRUE(kernel);

    kernel->compile(&conv_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);
}

TEST(operator_compile, convolution_backward_data_compile_fp32) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::ConvolutionBackpropData);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t diff_src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, impl::data_type::f32);
    impl::logical_tensor_t weights = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            2, {8, 16, 222, 222}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {diff_dst, weights};
    std::vector<impl::logical_tensor_t> outputs {diff_src};

    size_t operator_num = get_dnnl_kernel_registry().get_register_kernels_num();
    ASSERT_NE(operator_num, 0);

    auto conv_bwd_kernel = get_dnnl_kernel_registry().create_kernel(conv_op);
    ASSERT_TRUE(conv_bwd_kernel);

    conv_bwd_kernel->compile(&conv_op, &eng, inputs, outputs);

    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);
}

TEST(operator_compile, convolution_backward_weights_compile_fp32) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::conv_bwd_f_biasadd_bwd);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, impl::data_type::f32);
    impl::logical_tensor_t diff_weights = utils::logical_tensor_init(
            2, {16, 3, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_bias = utils::logical_tensor_init(
            3, dims {16}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            4, {8, 16, 222, 222}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src, diff_dst};
    std::vector<impl::logical_tensor_t> outputs {diff_weights, diff_bias};

    size_t operator_num = get_dnnl_kernel_registry().get_register_kernels_num();
    ASSERT_NE(operator_num, 0);

    auto conv_bwd_kernel = get_dnnl_kernel_registry().create_kernel(conv_op);
    ASSERT_TRUE(conv_bwd_kernel);

    conv_bwd_kernel->compile(&conv_op, &eng, inputs, outputs);

    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);
    ASSERT_EQ(outputs[1].layout_type, impl::layout_type::opaque);
}

void ref_batchnorm_fwd(impl::dim_t mb, impl::dim_t ic, impl::dim_t ih,
        impl::dim_t iw, impl::tensor_t *src, impl::tensor_t *dst,
        impl::tensor_t *scale, impl::tensor_t *shift,
        impl::tensor_t *mean = nullptr, impl::tensor_t *variance = nullptr,
        float epsilon = 0.001f,
        std::function<float(const float)> activation = nullptr,
        bool channel_last = false) {
    if (!activation) activation = [](const float v) { return v; };

    float *mean_ptr = nullptr;
    float *variance_ptr = nullptr;

    size_t mb_ = static_cast<size_t>(mb);
    size_t ic_ = static_cast<size_t>(ic);
    size_t ih_ = static_cast<size_t>(ih);
    size_t iw_ = static_cast<size_t>(iw);

    std::unique_ptr<float[]> mean_data;
    std::unique_ptr<float[]> variance_data;
    if (!mean) {
        mean_data.reset(new float[static_cast<size_t>(ic)]);
        mean_ptr = mean_data.get();
    } else {
        mean_ptr = mean->get_data_handle<float>();
    }
    if (!variance) {
        variance_data.reset(new float[static_cast<size_t>(ic)]);
        variance_ptr = variance_data.get();
    } else {
        variance_ptr = variance->get_data_handle<float>();
    }

    float *src_ptr = src->get_data_handle<float>();

    // derives ref mean
    for (size_t channel = 0; channel < ic_; ++channel) {
        mean_ptr[channel] = 0.f;
        for (size_t batch = 0; batch < mb_; ++batch) {
            for (size_t h = 0; h < ih_; ++h) {
                for (size_t w = 0; w < iw_; ++w) {
                    size_t idx = channel_last
                            ? channel + w * ic_ + h * iw_ * ic_
                                    + batch * ih_ * iw_ * ic_
                            : w + h * iw_ + channel * ih_ * iw_
                                    + batch * ic_ * ih_ * iw_;
                    mean_ptr[channel] += src_ptr[idx];
                }
            }
        }
        mean_ptr[channel] /= static_cast<float>(mb_ * ih_ * iw_);
    }
    // derives ref variance
    for (size_t channel = 0; channel < ic_; ++channel) {
        variance_ptr[channel] = 0.f;
        for (size_t batch = 0; batch < mb_; ++batch) {
            for (size_t h = 0; h < ih_; ++h) {
                for (size_t w = 0; w < iw_; ++w) {
                    size_t idx = channel_last
                            ? channel + w * ic_ + h * iw_ * ic_
                                    + batch * ih_ * iw_ * ic_
                            : w + h * iw_ + channel * ih_ * iw_
                                    + batch * ic_ * ih_ * iw_;
                    float variance_tmp = src_ptr[idx] - mean_ptr[channel];
                    variance_ptr[channel] += variance_tmp * variance_tmp;
                }
            }
        }
        variance_ptr[channel] /= static_cast<float>(mb * ih * iw);
    }

    float *dst_ptr = dst->get_data_handle<float>();
    float *scale_ptr = scale->get_data_handle<float>();
    float *shift_ptr = shift->get_data_handle<float>();
    for (size_t batch = 0; batch < mb_; ++batch) {
        for (size_t channel = 0; channel < ic_; ++channel) {
            for (size_t h = 0; h < ih_; ++h) {
                for (size_t w = 0; w < iw_; ++w) {
                    size_t idx = channel_last
                            ? channel + w * ic_ + h * iw_ * ic_
                                    + batch * ih_ * iw_ * ic_
                            : w + h * iw_ + channel * ih_ * iw_
                                    + batch * ic_ * ih_ * iw_;
                    dst_ptr[idx] = activation(scale_ptr[channel]
                                    * (src_ptr[idx] - mean_ptr[channel])
                                    / sqrtf(variance_ptr[channel] + epsilon)
                            + shift_ptr[channel]);
                }
            }
        }
    }
}

struct dnnl_graph_test_batchnorm_params {
    impl::op_kind_t op_kind;
    impl::dim_t N;
    impl::dim_t IC;
    impl::dim_t IH;
    impl::dim_t IW;
    float epsilon;
    std::string data_format;
    std::string backend_name;
    impl::data_type_t data_type;
};

class test_batchnorm_compile
    : public ::testing::TestWithParam<dnnl_graph_test_batchnorm_params> {
public:
    void TestBatchnorm(bool dims_in_order = true) {
        using dims = dnnl::graph::impl::dnnl_impl::dims;

        auto params = ::testing::TestWithParam<
                dnnl_graph_test_batchnorm_params>::GetParam();

        impl::op_t batchnorm_op(params.op_kind);
        batchnorm_op.set_attr("epsilon", params.epsilon);
        batchnorm_op.set_attr("data_format", params.data_format);

        impl::engine_t &engine = get_engine();

        // Tensor dimensions.
        const impl::dim_t N = params.N, // batch size
                IC = params.IC, // channels
                IH = params.IH, // tensor height
                IW = params.IW; // tensor width
        // Source (src) and destination (dst) tensors dimensions.
        dims src_dims;

        // |                     | == NCX?   | == NXC?   |
        // |---------------------|-----------|-----------|
        // |in-order (true)      | NCHW      | NHWC      |
        // |out-of-order (false) | NHWC      | HCHW      |
        if ((params.data_format == "NCX") ^ dims_in_order)
            src_dims = {N, IH, IW, IC};
        else
            src_dims = {N, IC, IH, IW};
        // Scale/shift tensor dimensions.
        dims scale_dims = {IC};
        dims shift_dims = {IC};

        // prepare logical tensor
        impl::logical_tensor_t src
                = utils::logical_tensor_init(0, src_dims, impl::data_type::f32);
        impl::logical_tensor_t scale = utils::logical_tensor_init(
                1, scale_dims, impl::data_type::f32);
        impl::logical_tensor_t shift = utils::logical_tensor_init(
                2, shift_dims, impl::data_type::f32);
        impl::logical_tensor_t dst
                = utils::logical_tensor_init(3, src_dims, impl::data_type::f32);

        // Allocate buffers.
        test::vector<float> src_data(static_cast<size_t>(N * IC * IH * IW));
        test::vector<float> scale_data(static_cast<size_t>(IC));
        test::vector<float> shift_data(static_cast<size_t>(IC));
        test::vector<float> dst_data(src_data.size(), 0.0);
        test::vector<float> ref_dst_data(src_data.size(), 0.0);
        // Initialize src.
        std::generate(src_data.begin(), src_data.end(), []() {
            static int i = 0;
            return std::cos(static_cast<float>(i++) / 10.f);
        });
        // Initialize scale.
        std::generate(scale_data.begin(), scale_data.end(), []() {
            static int i = 0;
            return std::sin(static_cast<float>(i++) * 2.f);
        });
        // Initialize shift.
        std::generate(shift_data.begin(), shift_data.end(), []() {
            static int i = 0;
            return std::tan(i++);
        });

        auto &op_factory = get_dnnl_kernel_registry();
        auto bn_kernel = op_factory.create_kernel(batchnorm_op);
        ASSERT_TRUE(bn_kernel);

        std::vector<impl::logical_tensor_t> inputs {src, scale, shift};
        std::vector<impl::logical_tensor_t> outputs {dst};

        // compile the bn operator
        bn_kernel->compile(&batchnorm_op, &engine, inputs, outputs);

        ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src, src_data.data());
        impl::tensor_t scale_ts(scale, scale_data.data());
        impl::tensor_t shift_ts(shift, shift_data.data());
        impl::tensor_t dst_ts(dst, dst_data.data());
        impl::tensor_t ref_dst_ts(dst, ref_dst_data.data());

        impl::stream_t &strm = get_stream();

        if (dims_in_order) {
            bn_kernel->execute(&batchnorm_op, &strm,
                    {src_ts, scale_ts, shift_ts}, {dst_ts});
            strm.wait();

            std::function<float(const float)> activation = nullptr;
            if (params.op_kind == impl::op_kind::bn_relu) {
                activation = [](const float v) { return v < 0.f ? 0.f : v; };
            }
            ref_batchnorm_fwd(N, IC, IH, IW, &src_ts, &ref_dst_ts, &scale_ts,
                    &shift_ts, nullptr, nullptr, params.epsilon, activation,
                    params.data_format == "NXC");

            for (size_t i = 0; i < dst_data.size(); ++i) {
                ASSERT_NEAR(dst_data[i], ref_dst_data[i], 1.e-6f);
            }
        } else {
            EXPECT_THROW(bn_kernel->execute(&batchnorm_op, &strm,
                                 {src_ts, scale_ts, shift_ts}, {dst_ts}),
                    std::runtime_error);
            strm.wait();
        };
    }

    void Test() {
        TestBatchnorm(true);
        TestBatchnorm(false);
    }
};

TEST_P(test_batchnorm_compile, TestBatchnormCompile) {}

INSTANTIATE_TEST_SUITE_P(TestBatchnormCompile, test_batchnorm_compile,
        ::testing::Values(
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, 3, 3, 2, 2, 0.001f,
                        "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, 3, 3, 2, 2, 0.001f,
                        "NXC", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {impl::op_kind::bn_relu, 3, 3,
                        2, 2, 0.001f, "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {impl::op_kind::bn_relu, 3, 3,
                        2, 2, 0.001f, "NXC", "dnnl", impl::data_type::f32}));

TEST(operator_compile, bn_compile_bwd_fp32) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::op_t bn_op(impl::op_kind::BatchNormTrainingBackprop);
    bn_op.set_attr<float>("epsilon", 0.0);
    bn_op.set_attr<std::string>("data_format", "NCX");

    impl::engine_t &engine = get_engine();

    // Tensor dimensions.
    const impl::dim_t N = 1, // batch size
            IC = 1, // channels
            IH = 2, // tensor height
            IW = 2; // tensor width
    // Source (src) and destination (dst) tensors dimensions.
    dims src_dims = {N, IC, IH, IW};
    // Scale/shift tensor dimensions.
    dims scale_dims = {IC};
    dims shift_dims = {IC};
    dims mean_dims = {IC};
    dims varience_dims = {IC};

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t scale = utils::logical_tensor_init(
            1, scale_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t mean = utils::logical_tensor_init(
            2, mean_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t varience = utils::logical_tensor_init(
            3, varience_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            4, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_src = utils::logical_tensor_init(
            5, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_scale = utils::logical_tensor_init(
            6, scale_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_shift = utils::logical_tensor_init(
            7, shift_dims, impl::data_type::f32, impl::layout_type::strided);

    // Allocate buffers.
    test::vector<float> src_data {1.0, 2.0, 3.0, 4.0};
    test::vector<float> scale_data {2.0, 0.0, 0.0, 0.0};
    test::vector<float> mean_data {2.5, 0.0, 0.0, 0.0};
    test::vector<float> varience_data {1.25, 0.0, 0.0, 0.0};
    test::vector<float> diff_dst_data {0.1f, 0.1f, 0.2f, 0.2f};
    test::vector<float> diff_src_data(src_data.size(), 0.0);
    test::vector<float> diff_scale_data(scale_data.size(), 0.0);
    test::vector<float> diff_shift_data(scale_data.size(), 0.0);

    // expected results
    test::vector<float> ref_diff_src_data {
            0.17888544f, 0.17888544f, 0.35777088f, 0.35777088f};
    test::vector<float> ref_diff_scale_data {0.178885f, 0.0, 0.0, 0.0};
    test::vector<float> ref_diff_shift_data {0.6f, 0.0, 0.0, 0.0};

    auto &op_factory = get_dnnl_kernel_registry();
    auto bn_kernel = op_factory.create_kernel(bn_op);
    ASSERT_TRUE(bn_kernel);

    std::vector<impl::logical_tensor_t> inputs {
            src, diff_dst, scale, mean, varience};
    std::vector<impl::logical_tensor_t> outputs {
            diff_src, diff_scale, diff_shift};

    // compile the bn operator
    bn_kernel->compile(&bn_op, &engine, inputs, outputs);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t scale_ts(scale, scale_data.data());
    impl::tensor_t mean_ts(mean, mean_data.data());
    impl::tensor_t varience_ts(varience, varience_data.data());
    impl::tensor_t diff_dst_ts(diff_dst, diff_dst_data.data());
    impl::tensor_t diff_src_ts(diff_src, diff_src_data.data());
    impl::tensor_t diff_scale_ts(diff_scale, diff_scale_data.data());
    impl::tensor_t diff_shift_ts(diff_shift, diff_shift_data.data());

    impl::stream_t &strm = get_stream();
    bn_kernel->execute(&bn_op, &strm,
            {src_ts, diff_dst_ts, scale_ts, mean_ts, varience_ts},
            {diff_src_ts, diff_scale_ts, diff_shift_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src_data.size(); ++i) {
        auto absolute_error
                = std::fabs(diff_src_data[i] - ref_diff_src_data[i]);
        ASSERT_TRUE(absolute_error < 0.001);
    }
    for (size_t i = 0; i < diff_scale_data.size(); ++i) {
        auto absolute_error
                = std::fabs(diff_scale_data[i] - ref_diff_scale_data[i]);
        ASSERT_TRUE(absolute_error < 0.001);
    }
    for (size_t i = 0; i < diff_shift_data.size(); ++i) {
        auto absolute_error
                = std::fabs(diff_shift_data[i] - ref_diff_shift_data[i]);
        ASSERT_TRUE(absolute_error < 0.001);
    }
}

TEST(operator_kernel, relu) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> ref_dst {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t relu_op(impl::op_kind::ReLU);

    auto &op_factory = get_dnnl_kernel_registry();
    auto relu_kernel = op_factory.create_kernel(relu_op);
    ASSERT_TRUE(relu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);

    impl::engine_t &eng = get_engine();
    // compile the relu operator
    relu_kernel->compile(&relu_op, &eng, {src_lt}, {dst_lt});

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    relu_kernel->execute(&relu_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, relu_backward) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_diff_src {
            0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> diff_src(src.size(), 0.0);

    impl::op_t relu_op(impl::op_kind::ReLUBackprop);

    auto &op_factory = get_dnnl_kernel_registry();
    auto relu_kernel = op_factory.create_kernel(relu_op);
    ASSERT_TRUE(relu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the relu backward operator
    relu_kernel->compile(&relu_op, &eng, {diff_dst_lt, src_lt}, {diff_src_lt});

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_src_ts(diff_src_lt, diff_src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());

    impl::stream_t &strm = get_stream();
    relu_kernel->execute(&relu_op, &strm, {diff_dst_ts, src_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, gelu) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> ref_dst {-0.0455001f, -0.10021085f, -0.15865527f,
            -0.15426877f, 0.f, 0.3457312f, 0.84134465f, 1.399789f, 1.9544998f};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t gelu_op(impl::op_kind::GELU);

    auto &op_factory = get_dnnl_kernel_registry();
    auto gelu_kernel = op_factory.create_kernel(gelu_op);
    ASSERT_TRUE(gelu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);

    gelu_kernel->compile(&gelu_op, &eng, {src_lt}, {dst_lt});

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    gelu_kernel->execute(&gelu_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_NEAR(dst[i], ref_dst[i], 1.e-6f);
    }
}

TEST(operator_kernel, gelu_backward) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_diff_src {0.0f, -0.1274692f, -0.1666309f,
            0.3975146f, 2.0f, 4.3374756f, 6.4998928f, 7.8922843f, 8.6818544f};
    test::vector<float> diff_src(src.size(), 0.0);

    impl::op_t gelu_op(impl::op_kind::GELUBackprop);

    auto &op_factory = get_dnnl_kernel_registry();
    auto gelu_kernel = op_factory.create_kernel(gelu_op);
    ASSERT_TRUE(gelu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the helu backward operator
    gelu_kernel->compile(&gelu_op, &eng, {diff_dst_lt, src_lt}, {diff_src_lt});

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_src_ts(diff_src_lt, diff_src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());

    impl::stream_t &strm = get_stream();
    gelu_kernel->execute(&gelu_op, &strm, {diff_dst_ts, src_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    // compile the add operator
    std::vector<impl::logical_tensor_t> inputs {src0_lt, src1_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};
    add_kernel->compile(&add_op, &eng, inputs, outputs);

    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::any);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    add_kernel->execute(&add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    // the second iteration
    test::vector<float> src0_2nd {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1_2nd {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> ref_dst_2nd {
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst_2nd(src0_2nd.size(), 0.0);

    impl::tensor_t src0_2nd_ts(src0_lt, src0_2nd.data());
    impl::tensor_t src1_2nd_ts(src1_lt, src1_2nd.data());
    impl::tensor_t dst_2nd_ts(outputs[0], dst_2nd.data());
    add_kernel->execute(
            &add_op, &strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm.wait();

    for (size_t i = 0; i < src0_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(operator_kernel, different_format_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, {3, 3, 1, 3}, impl::data_type::f32); // abdc format
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 3, 3}, impl::data_type::f32);
    // compile the add operator
    add_kernel->compile(&add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    add_kernel->execute(&add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, broadcast_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {
            2.0, 2.0, 2.0}; // bianary op's src1 support broadcast
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t broadcast_add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto broadcast_add_kernel = op_factory.create_kernel(broadcast_add_op);
    ASSERT_TRUE(broadcast_add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 2, 3, 3}, impl::data_type::f32);
    // src1 will first be unsequeeze to {1,1,1,3} and then broadcast
    // to {1,2,3,3}
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 3, 3}, impl::data_type::f32);

    // compile the add operator
    broadcast_add_kernel->compile(
            &broadcast_add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, add_shape_mismatch_0) {
    impl::engine_t &eng = get_engine();

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {8, 4, 256}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {2, 2, 512}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {8, 4, 256}, impl::data_type::f32);

    // compile the add operator
    auto ret = add_kernel->compile(&add_op, &eng, {src0_lt, src1_lt}, {dst_lt});
    ASSERT_EQ(ret, impl::status::invalid_shape);
}

TEST(operator_kernel, add_shape_mismatch_1) {
    impl::engine_t &eng = get_engine();

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt = utils::logical_tensor_init(
            0, {8, 15, 5, 7}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 15, 5, 7}, impl::data_type::f32);

    // compile the add operator
    auto ret = add_kernel->compile(&add_op, &eng, {src0_lt, src1_lt}, {dst_lt});
    ASSERT_EQ(ret, impl::status::success);
}

TEST(operator_kernel, add_shape_mismatch_2) {
    impl::engine_t &eng = get_engine();

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {8, 15, 5, 7}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 15, 5, 7}, impl::data_type::f32);

    // compile the add operator
    auto ret = add_kernel->compile(&add_op, &eng, {src0_lt, src1_lt}, {dst_lt});
    ASSERT_EQ(ret, impl::status::success);
}

TEST(operator_kernel, reversed_different_format_broadcast_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src1 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src0 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0,
            2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src1.size(), 0.0);

    impl::op_t broadcast_add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto broadcast_add_kernel = op_factory.create_kernel(broadcast_add_op);
    ASSERT_TRUE(broadcast_add_kernel);

    // we reverse the order of src0 and src1
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(0, {1, 2, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src0_lt = utils::logical_tensor_init(
            1, {3, 3}, {1, 3}, impl::data_type::f32); // ba format
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 3, 3}, impl::data_type::f32);

    // compile the add operator
    broadcast_add_kernel->compile(
            &broadcast_add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, bias_add) {
    impl::engine_t &eng = get_engine();
    auto &op_factory = get_dnnl_kernel_registry();

    test::vector<float> src {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> bias {1.0, 2.0};
    test::vector<float> ref_dst1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> ref_dst2 {2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0,
            3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0};
    test::vector<float> dst(src.size(), 0.0);

    std::vector<std::vector<impl::dim_t>> src_shapes {
            {1, 2, 3, 3}, {1, 3, 3, 2}};
    std::vector<std::vector<impl::dim_t>> dst_shapes {
            {1, 2, 3, 3}, {1, 3, 3, 2}};
    std::vector<impl::dim_t> bias_shape {2};
    std::vector<std::string> data_formats {"NCX", "NXC"};

    for (size_t i = 0; i < data_formats.size(); i++) {
        impl::op_t bias_add_op(impl::op_kind::BiasAdd);
        bias_add_op.set_attr<std::string>("data_format", data_formats[i]);

        auto bias_add_kernel = op_factory.create_kernel(bias_add_op);
        ASSERT_TRUE(bias_add_kernel);

        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, src_shapes[i], impl::data_type::f32);
        impl::logical_tensor_t bias_lt = utils::logical_tensor_init(
                1, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, dst_shapes[i], impl::data_type::f32);

        // compile the add operator
        bias_add_kernel->compile(
                &bias_add_op, &eng, {src_lt, bias_lt}, {dst_lt});

        impl::tensor_t src_ts(src_lt, src.data());
        impl::tensor_t bias_ts(bias_lt, bias.data());
        impl::tensor_t dst_ts(dst_lt, dst.data());

        impl::stream_t &strm = get_stream();
        bias_add_kernel->execute(
                &bias_add_op, &strm, {src_ts, bias_ts}, {dst_ts});
        strm.wait();

        if (i == 0) {
            for (size_t i = 0; i < src.size(); ++i) {
                ASSERT_FLOAT_EQ(dst[i], ref_dst1[i]);
            }
        } else {
            for (size_t i = 0; i < src.size(); ++i) {
                ASSERT_FLOAT_EQ(dst[i], ref_dst2[i]);
            }
        }
    }
}

TEST(operator_kernel, mul) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(impl::op_kind::Multiply);

    auto &op_factory = get_dnnl_kernel_registry();
    auto mul_kernel = op_factory.create_kernel(mul_op);
    ASSERT_TRUE(mul_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    mul_kernel->compile(&mul_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    mul_kernel->execute(&mul_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, min) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> ref_dst {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t max_op(impl::op_kind::Minimum);

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_kernel = op_factory.create_kernel(max_op);
    ASSERT_TRUE(max_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    max_kernel->compile(&max_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    max_kernel->execute(&max_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, max) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> ref_dst {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t min_op(impl::op_kind::Maximum);

    auto &op_factory = get_dnnl_kernel_registry();
    auto min_kernel = op_factory.create_kernel(min_op);
    ASSERT_TRUE(min_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    min_kernel->compile(&min_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, src0.data());
    impl::tensor_t src1_ts(src1_lt, src1.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    min_kernel->execute(&min_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, matmul_compile_fwd_fp32) {
    impl::op_t matmul_op(impl::op_kind::MatMul);
    matmul_op.set_attr<bool>("transpose_b", true);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {-2.0, -1.5};
    test::vector<float> ref_dst_data {6.25};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_compile_fwd_f16f16f16) {
    impl::op_t matmul_op(impl::op_kind::MatMul);

    impl::engine_t &engine = get_engine();
    SKIP_IF(engine.kind() != impl::engine_kind::gpu,
            "Skip fp16 test for non-GPU device.");

    // in real use case, the data type should be half
    test::vector<float> src_data {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5};
    test::vector<float> weight_data {
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, -2.0, -1.5};
    test::vector<float> dst_data(12, 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2}, impl::data_type::f16);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2}, impl::data_type::f16);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {1, 1}, impl::data_type::f16, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

TEST(operator_kernel, matmul_compile_fwd_bf16bf16bf16) {
    impl::op_t matmul_op(impl::op_kind::MatMul);

    impl::engine_t &engine = get_engine();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && engine.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    // in real use case, the data type should be half
    test::vector<float> src_data {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5};
    test::vector<float> weight_data {
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, -2.0, -1.5};
    test::vector<float> dst_data(12, 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2}, impl::data_type::bf16);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2}, impl::data_type::bf16);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {1, 1}, impl::data_type::bf16, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

static size_t product(std::vector<int64_t> &in) {
    if (in.empty()) return 0;
    int64_t prod = std::accumulate(in.begin(), in.end(),
            static_cast<int64_t>(1), std::multiplies<int64_t>());
    return static_cast<size_t>(prod);
}

TEST(operator_kernel, matmul_ndx2d) {
    impl::op_t matmul_op(impl::op_kind::MatMul);
    impl::engine_t &engine = get_engine();
    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);

    std::vector<std::vector<int64_t>> weight_shapes {{16}, {16, 1}};
    std::vector<std::vector<int64_t>> src_shapes {
            {4, 2, 16}, {6, 4, 2, 16}, {8, 6, 4, 2, 16}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {4, 2, 1}, {6, 4, 2, 1}, {8, 6, 4, 2, 1}};

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    for (size_t w_idx = 0; w_idx < weight_shapes.size(); ++w_idx) {
        for (size_t idx = 0; idx < src_shapes.size(); idx++) {
            auto src_shape = src_shapes[idx];
            auto dst_shape = dst_shapes[idx];

            test::vector<float> src_data(product(src_shape));
            test::vector<float> weight_data(product(weight_shapes[w_idx]));
            test::vector<float> dst_data(product(dst_shape));
            test::vector<float> ref_dst_data(product(dst_shape), 0.0);

            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return distribution(generator); });

            // prepare logical tensor
            impl::logical_tensor_t src = utils::logical_tensor_init(
                    0, src_shape, impl::data_type::f32);
            impl::logical_tensor_t weight = utils::logical_tensor_init(
                    1, weight_shapes[w_idx], impl::data_type::f32);
            impl::logical_tensor_t dst = utils::logical_tensor_init(
                    2, dst_shape, impl::data_type::f32, impl::layout_type::any);

            std::vector<impl::logical_tensor_t> inputs {src, weight};
            std::vector<impl::logical_tensor_t> outputs {dst};

            matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
            ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

            impl::tensor_t src_ts(src, src_data.data());
            impl::tensor_t weight_ts(weight, weight_data.data());
            impl::tensor_t dst_ts(outputs[0], dst_data.data());

            impl::stream_t &strm = get_stream();
            matmul_kernel->execute(
                    &matmul_op, &strm, {src_ts, weight_ts}, {dst_ts});
            strm.wait();

            // compute the ref results
            size_t M = product(src_shape)
                    / static_cast<size_t>(src_shape.back());
            size_t K = static_cast<size_t>(src_shape.back());
            size_t N = weight_shapes[w_idx].size() == 1
                    ? static_cast<size_t>(1)
                    : static_cast<size_t>(weight_shapes[w_idx].back());
            // size_t N = static_cast<size_t>(1);
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    for (size_t k = 0; k < K; k++)
                        ref_dst_data[m * N + n]
                                += src_data[m * K + k] * weight_data[k * N + n];
                }
            }

            // FIXME(qun) the max error will be bigger than 1e-6
            for (size_t i = 0; i < ref_dst_data.size(); ++i) {
                ASSERT_NEAR(dst_data[i], ref_dst_data[i], 3e-6f);
            }
        }
    }
}

TEST(operator_kernel, matmul_3dx3d) {
    impl::op_t matmul_op(impl::op_kind::MatMul);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> weight_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> ref_dst_data {2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {4, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {4, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {4, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_relu_fusion) {
    impl::op_t matmul_relu_op(impl::op_kind::matmul_relu);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, 1.5};
    test::vector<float> ref_dst_data {0.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_relu_kernel = op_factory.create_kernel(matmul_relu_op);

    matmul_relu_kernel->compile(&matmul_relu_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_relu_kernel->execute(
            &matmul_relu_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    test::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {7.25, 4.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_fusion_broadcast_1d) {
    impl::op_t matmul_op(impl::op_kind::matmul_add);

    test::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    test::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    test::vector<float> add_src1_data {1.0, 1.0};
    test::vector<float> ref_dst_data {5.0, 4.0, 4.0, 3.25, 4.0, 4.0, 1.5, 1.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_src1
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, add_src1};
    std::vector<impl::logical_tensor_t> outputs {dst};

    impl::engine_t &engine = get_engine();
    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t add_src1_ts(add_src1, add_src1_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, add_src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_add);

    test::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    test::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    test::vector<float> add_src1_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> ref_dst_data {5.0, 4.0, 4.0, 3.25, 4.0, 4.0, 1.5, 1.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_src1
            = utils::logical_tensor_init(2, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, add_src1};
    std::vector<impl::logical_tensor_t> outputs {dst};

    impl::engine_t &engine = get_engine();
    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t add_src1_ts(add_src1, add_src1_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, add_src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_gelu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_add_gelu);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {1.0, 1.0, 1.0, 1.0};
    test::vector<float> weight_data {0.5, 0.5, 0.5, 0.5};
    test::vector<float> other_data {1.0};
    test::vector<float> ref_dst_data {1.9544998f, 1.9544998f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t other
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, other};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t other_ts(other, other_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, other_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_relu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_add_relu);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {1.0, 2.0, 3.0, 4.0};
    test::vector<float> weight_data {-1.0, -2.0, 3.0, 4.0};
    test::vector<float> other_data {2.5};
    test::vector<float> ref_dst_data {0, 27.5f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t other
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, other};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t other_ts(other, other_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, other_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_relu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_relu);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {0.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_gelu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_gelu);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {1.0, 1.0, 1.0, 1.0};
    test::vector<float> weight_data {0.5, 0.5, 0.5, 0.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {1.9544998f, 1.9544998f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {2, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_relu6_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_relu6);
    matmul_op.set_attr<float>("min", 0.0);
    matmul_op.set_attr<float>("max", 6.0);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {-2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {6.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_hardtanh_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_hardtanh);
    matmul_op.set_attr<float>("min", -3.0);
    matmul_op.set_attr<float>("max", 3.0);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {-2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {3.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_elu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_elu);
    matmul_op.set_attr<float>("alpha", 1.f);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {-0.75};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);
    ref_dst_data[0] = static_cast<float>(exp(-0.75) - 1);
    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_sigmoid_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_sigmoid);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {-0.75};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);
    ref_dst_data[0] = static_cast<float>(1 / (exp(-ref_dst_data[0]) + 1));
    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(
            &matmul_op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_add_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_add);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> post_src_data {-2.0};
    test::vector<float> ref_dst_data {-2.75};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t post_src
            = utils::logical_tensor_init(3, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            4, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias, post_src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t post_src_ts(post_src, post_src_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm,
            {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_per_tensor_broadcast_add_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_add);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0, 2.0};
    test::vector<float> post_src_data {-2.0};
    test::vector<float> ref_dst_data {-5.0, 3.0, -4.0, 2.25};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    std::vector<impl::dims> post_src_shapes = {{1}, {1, 1}};

    for (auto &post_src_shape : post_src_shapes) {
        // prepare logical tensor
        impl::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 1}, impl::data_type::f32);
        impl::logical_tensor_t weight
                = utils::logical_tensor_init(1, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t bias
                = utils::logical_tensor_init(2, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t post_src = utils::logical_tensor_init(
                3, post_src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst
                = utils::logical_tensor_init(4, {2, 2}, impl::data_type::f32);

        std::vector<impl::logical_tensor_t> inputs {
                src, weight, bias, post_src};
        std::vector<impl::logical_tensor_t> outputs {dst};

        auto &op_factory = get_dnnl_kernel_registry();
        auto matmul_kernel = op_factory.create_kernel(matmul_op);
        matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);

        impl::tensor_t src_ts(src, src_data.data());
        impl::tensor_t weight_ts(weight, weight_data.data());
        impl::tensor_t bias_ts(bias, bias_data.data());
        impl::tensor_t post_src_ts(post_src, post_src_data.data());
        impl::tensor_t dst_ts(outputs[0], dst_data.data());

        impl::stream_t &strm = get_stream();
        matmul_kernel->execute(&matmul_op, &strm,
                {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
        strm.wait();
        for (size_t i = 0; i < ref_dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
        }
    }
}

TEST(operator_kernel, matmul_bias_per_channel_broadcast_add_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_add);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0, 2.0};
    test::vector<float> post_src_data {-2.0, -1.0};
    test::vector<float> ref_dst_data {-5.0, 4.0, -4.0, 3.25};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    std::vector<impl::dims> post_src_shapes = {{2}, {1, 2}};

    for (auto &post_src_shape : post_src_shapes) {
        // prepare logical tensor
        impl::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 1}, impl::data_type::f32);
        impl::logical_tensor_t weight
                = utils::logical_tensor_init(1, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t bias
                = utils::logical_tensor_init(2, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t post_src = utils::logical_tensor_init(
                3, post_src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst
                = utils::logical_tensor_init(4, {2, 2}, impl::data_type::f32);

        std::vector<impl::logical_tensor_t> inputs {
                src, weight, bias, post_src};
        std::vector<impl::logical_tensor_t> outputs {dst};

        auto &op_factory = get_dnnl_kernel_registry();
        auto matmul_kernel = op_factory.create_kernel(matmul_op);
        ASSERT_EQ(matmul_kernel->compile(&matmul_op, &engine, inputs, outputs),
                impl::status::success);

        impl::tensor_t src_ts(src, src_data.data());
        impl::tensor_t weight_ts(weight, weight_data.data());
        impl::tensor_t bias_ts(bias, bias_data.data());
        impl::tensor_t post_src_ts(post_src, post_src_data.data());
        impl::tensor_t dst_ts(outputs[0], dst_data.data());

        impl::stream_t &strm = get_stream();
        matmul_kernel->execute(&matmul_op, &strm,
                {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
        strm.wait();
        for (size_t i = 0; i < ref_dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
        }
    }
}

TEST(operator_kernel, matmul_bias_unsupported_broadcast_add_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_add);
    impl::engine_t &engine = get_engine();

    std::vector<impl::dims> post_src_shapes = {{3}, {1, 3}, {2, 1}};

    for (auto &post_src_shape : post_src_shapes) {
        // prepare logical tensor
        impl::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 1}, impl::data_type::f32);
        impl::logical_tensor_t weight
                = utils::logical_tensor_init(1, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t bias
                = utils::logical_tensor_init(2, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t post_src = utils::logical_tensor_init(
                3, post_src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst
                = utils::logical_tensor_init(4, {2, 2}, impl::data_type::f32);

        std::vector<impl::logical_tensor_t> inputs {
                src, weight, bias, post_src};
        std::vector<impl::logical_tensor_t> outputs {dst};

        auto &op_factory = get_dnnl_kernel_registry();
        auto matmul_kernel = op_factory.create_kernel(matmul_op);
        ASSERT_EQ(matmul_kernel->compile(&matmul_op, &engine, inputs, outputs),
                impl::status::compile_fail);
    }
}

TEST(operator_kernel, matmul_bias_add_relu_fusion) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_add_relu);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> post_src_data {-2.0};
    test::vector<float> ref_dst_data {0.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t post_src
            = utils::logical_tensor_init(3, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            4, {1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src, weight, bias, post_src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t post_src_ts(post_src, post_src_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm,
            {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_bn) {
    impl::op_t matmul_op(impl::op_kind::matmul_bias_bn);
    matmul_op.set_attr<float>("epsilon", 0.f);
    matmul_op.set_attr<bool>("transpose_b", true);
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5, -3.0, 0.5};
    test::vector<float> weight_data {2.0, -1.5, 1.0, -1.0};
    test::vector<float> bias_data {1.0, -0.5};
    test::vector<float> scale_data {2.0, 1.0};
    test::vector<float> shift_data {-1.5, 0.5};
    test::vector<float> mean_data {1.0, -2.0};
    test::vector<float> varience_data {1.0, 0.25};
    test::vector<float> ref_dst_data {-5.0, -3.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t scale
            = utils::logical_tensor_init(3, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t shift
            = utils::logical_tensor_init(4, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t mean
            = utils::logical_tensor_init(5, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t varience
            = utils::logical_tensor_init(6, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            7, {2, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {
            src, weight, bias, scale, shift, mean, varience};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto matmul_kernel = op_factory.create_kernel(matmul_op);
    matmul_kernel->compile(&matmul_op, &engine, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t weight_ts(weight, weight_data.data());
    impl::tensor_t bias_ts(bias, bias_data.data());
    impl::tensor_t scale_ts(scale, scale_data.data());
    impl::tensor_t shift_ts(shift, shift_data.data());
    impl::tensor_t mean_ts(mean, mean_data.data());
    impl::tensor_t varience_ts(varience, varience_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    matmul_kernel->execute(&matmul_op, &strm,
            {src_ts, weight_ts, bias_ts, scale_ts, shift_ts, mean_ts,
                    varience_ts},
            {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, max_pool) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {-0.5, 2.0, 3.0, 4.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t max_pool_op(impl::op_kind::MaxPool);
    max_pool_op.set_attr<dims>("strides", {2, 2});
    max_pool_op.set_attr<dims>("kernel", {2, 2});
    max_pool_op.set_attr<dims>("pads_begin", {0, 0});
    max_pool_op.set_attr<dims>("pads_end", {0, 0});
    max_pool_op.set_attr<std::string>("data_format", "NCX");
    max_pool_op.set_attr<dims>("dilations", {1, 1});

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_pool_kernel = op_factory.create_kernel(max_pool_op);

    max_pool_kernel->compile(&max_pool_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    max_pool_kernel->execute(&max_pool_op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, avg_pool_exclude_pad) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {
            -2.0, 0.25, 0.5, 0.75, 0.5, 0.75, 3.0, -1.5, 4.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t avg_pool_op(impl::op_kind::AvgPool);
    avg_pool_op.set_attr<dims>("strides", {2, 2});
    avg_pool_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_op.set_attr<bool>("exclude_pad", true);
    avg_pool_op.set_attr<std::string>("data_format", "NCX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_kernel = op_factory.create_kernel(avg_pool_op);

    avg_pool_kernel->compile(&avg_pool_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    avg_pool_kernel->execute(&avg_pool_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, avg_pool_include_pad) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {
            -0.5, 0.125, 0.125, 0.375, 0.5, 0.375, 0.75, -0.75, 1.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t avg_pool_op(impl::op_kind::AvgPool);
    avg_pool_op.set_attr<dims>("strides", {2, 2});
    avg_pool_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_op.set_attr<std::string>("data_format", "NCX");
    avg_pool_op.set_attr<bool>("exclude_pad", false);

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_kernel = op_factory.create_kernel(avg_pool_op);

    avg_pool_kernel->compile(&avg_pool_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    avg_pool_kernel->execute(&avg_pool_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NCX_OIX) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution3D_NCX_OIX) {
    using dims = std::vector<int64_t>;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NCX_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {3, 3, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution3D_NCX_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 3, 3, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NXC_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {
            -3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5, -1.0, 0};
    test::vector<float> weight {
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst {0.5};
    test::vector<float> dst {0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {3, 4, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 1, 1}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution3D_NXC_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {
            -3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5, -1.0, 0};
    test::vector<float> weight {
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst {0.5};
    test::vector<float> dst {0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 3, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 3, 4, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 1, 1}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NXC_OIX) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 4, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 2, 1}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution3D_NXC_OIX) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2, 1}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, ConvolutionF16F16F16) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();
    SKIP_IF(eng.kind() != impl::engine_kind::gpu,
            "Skip fp16 test for non-GPU device.");
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f16);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f16);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f16);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

TEST(operator_compile, ConvolutionBF16BF16BF16) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::bf16);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::bf16);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, impl::data_type::bf16);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

TEST(operator_compile, ConvolutionBF16BF16F32) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::bf16);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::bf16);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(&conv_op, &strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

TEST(operator_kernel, group_convolution) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 4);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, impl::data_type::f32);
    impl::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 8, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> conv_inputs {
            conv_src_lt, conv_weight_lt};
    std::vector<impl::logical_tensor_t> conv_outputs {conv_dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, conv_inputs, conv_outputs);
    ASSERT_EQ(conv_outputs[0].layout_type, impl::layout_type::strided);

    test::vector<float> conv_src(8 * 32 * 16 * 16, 1);
    test::vector<float> conv_weight(32 * 8 * 1 * 1, 1);
    test::vector<float> relu_dst(8 * 32 * 16 * 16, 1);
    size_t conv_dst_size
            = impl::dnnl_impl::dnnl_backend::get_singleton().get_mem_size(
                    conv_outputs[0]);
    test::vector<float> conv_dst(conv_dst_size / sizeof(float), 1);

    impl::tensor_t conv_src_ts(conv_src_lt, conv_src.data());
    impl::tensor_t conv_weight_ts(conv_weight_lt, conv_weight.data());
    impl::tensor_t conv_dst_ts(conv_outputs[0], conv_dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(
            &conv_op, &strm, {conv_src_ts, conv_weight_ts}, {conv_dst_ts});
    strm.wait();

    for (size_t i = 0; i < relu_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(conv_dst[i], 8);
    }
}

TEST(operator_kernel, ConvolutionBackpropData) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_diff_src {0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0,
            0.0, 3.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0};
    test::vector<float> diff_src(src.size(), 0.0);

    impl::op_t conv_op(impl::op_kind::ConvolutionBackpropData);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(1, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(2, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_bwd_kernel = op_factory.create_kernel(conv_op);

    conv_bwd_kernel->compile(
            &conv_op, &eng, {diff_dst_lt, weight_lt}, {diff_src_lt});
    ASSERT_EQ(diff_src_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(diff_src_lt, diff_src.data());

    impl::stream_t &strm = get_stream();
    conv_bwd_kernel->execute(
            &conv_op, &strm, {diff_dst_ts, weight_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, conv_bwd_weight_bias) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0};
    test::vector<float> ref_diff_weights {
            -5.5, 3.0, 7.0, 10.5, 3.0, -0.5, 2.5, -8.0, 10.0};
    test::vector<float> ref_diff_bias {6.0};
    test::vector<float> diff_weights(ref_diff_weights.size(), 0.0);
    test::vector<float> diff_bias(ref_diff_bias.size(), 0.0);

    impl::op_t conv_op(impl::op_kind::conv_bwd_f_biasadd_bwd);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare input/output logical_tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_weights_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_bwd_kernel = op_factory.create_kernel(conv_op);

    conv_bwd_kernel->compile(&conv_op, &eng, {src_lt, diff_dst_lt},
            {diff_weights_lt, diff_bias_lt});
    ASSERT_EQ(diff_weights_lt.layout_type, impl::layout_type::strided);
    ASSERT_EQ(diff_bias_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_weights_ts(diff_weights_lt, diff_weights.data());
    impl::tensor_t diff_bias_ts(diff_bias_lt, diff_bias.data());

    impl::stream_t &strm = get_stream();
    conv_bwd_kernel->execute(&conv_op, &strm, {src_ts, diff_dst_ts},
            {diff_weights_ts, diff_bias_ts});
    strm.wait();
    for (size_t i = 0; i < diff_weights.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_weights[i], ref_diff_weights[i]);
    }
    for (size_t i = 0; i < diff_bias.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_bias[i], ref_diff_bias[i]);
    }
}

TEST(operator_compile, conv_relu_unfused) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    impl::op_t relu_op(impl::op_kind::ReLU);

    // prepare logical tensor
    impl::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, impl::data_type::f32);
    impl::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 32, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> conv_inputs {
            conv_src_lt, conv_weight_lt};
    std::vector<impl::logical_tensor_t> conv_outputs {conv_dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);

    conv_kernel->compile(&conv_op, &eng, conv_inputs, conv_outputs);
    ASSERT_EQ(conv_outputs[0].layout_type, impl::layout_type::opaque);

    impl::logical_tensor_t relu_dst_lt = utils::logical_tensor_init(
            3, {8, 32, 16, 16}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> relu_inputs {conv_outputs[0]};
    std::vector<impl::logical_tensor_t> relu_outputs {relu_dst_lt};

    auto relu_kernel = op_factory.create_kernel(relu_op);

    relu_kernel->compile(&relu_op, &eng, relu_inputs, relu_outputs);
    ASSERT_EQ(relu_outputs[0].layout_type, impl::layout_type::strided);

    test::vector<float> conv_src(8 * 32 * 16 * 16, 1);
    test::vector<float> conv_weight(32 * 32 * 1 * 1, 1);
    test::vector<float> relu_dst(8 * 32 * 16 * 16, 1);
    size_t conv_dst_size
            = impl::dnnl_impl::dnnl_backend::get_singleton().get_mem_size(
                    conv_outputs[0]);
    test::vector<float> conv_dst(conv_dst_size / sizeof(float), 1);

    impl::tensor_t conv_src_ts(conv_src_lt, conv_src.data());
    impl::tensor_t conv_weight_ts(conv_weight_lt, conv_weight.data());
    impl::tensor_t conv_dst_ts(conv_outputs[0], conv_dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(
            &conv_op, &strm, {conv_src_ts, conv_weight_ts}, {conv_dst_ts});

    impl::tensor_t relu_src_ts(conv_outputs[0], conv_dst.data());
    impl::tensor_t relu_dst_ts(relu_outputs[0], relu_dst.data());

    relu_kernel->execute(&relu_op, &strm, {relu_src_ts}, {relu_dst_ts});
    strm.wait();

    for (size_t i = 0; i < relu_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(relu_dst[i], 32);
    }
}

TEST(operator_compile, convolution_bn_fp32) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    impl::op_t bn_op(impl::op_kind::BatchNormInference);
    bn_op.set_attr<std::string>("data_format", "NCX");
    bn_op.set_attr("epsilon", 1e-6f);

    // prepare logical tensor
    impl::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, impl::data_type::f32);
    impl::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 32, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, impl::data_type::f32, impl::layout_type::any);
    std::vector<impl::logical_tensor_t> conv_inputs {
            conv_src_lt, conv_weight_lt};
    std::vector<impl::logical_tensor_t> conv_outputs {conv_dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_kernel = op_factory.create_kernel(conv_op);
    conv_kernel->compile(&conv_op, &eng, conv_inputs, conv_outputs);

    impl::logical_tensor_t gamma
            = utils::logical_tensor_init(3, {32}, impl::data_type::f32);
    impl::logical_tensor_t beta
            = utils::logical_tensor_init(4, {32}, impl::data_type::f32);
    impl::logical_tensor_t scale
            = utils::logical_tensor_init(5, {32}, impl::data_type::f32);
    impl::logical_tensor_t shift
            = utils::logical_tensor_init(6, {32}, impl::data_type::f32);
    impl::logical_tensor_t bn_dst_lt = utils::logical_tensor_init(
            7, {8, 32, 16, 16}, impl::data_type::f32);
    std::vector<impl::logical_tensor_t> bn_inputs {
            conv_outputs[0], gamma, beta, scale, shift};
    std::vector<impl::logical_tensor_t> bn_outputs {bn_dst_lt};

    auto bn_kernel = op_factory.create_kernel(bn_op);
    bn_kernel->compile(&bn_op, &eng, bn_inputs, bn_outputs);

    test::vector<float> conv_src(8 * 32 * 16 * 16);
    test::vector<float> conv_weight(32 * 32 * 1 * 1);
    test::vector<float> bn_gamma(32);
    test::vector<float> bn_beta(32);
    test::vector<float> bn_scale(32);
    test::vector<float> bn_shift(32);

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    std::generate(conv_src.begin(), conv_src.end(),
            [&]() { return distribution(generator); });
    std::generate(conv_weight.begin(), conv_weight.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_gamma.begin(), bn_gamma.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_beta.begin(), bn_beta.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_scale.begin(), bn_scale.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_shift.begin(), bn_shift.end(),
            [&]() { return distribution(generator); });

    auto &dnnl_backend = impl::dnnl_impl::dnnl_backend::get_singleton();
    size_t conv_dst_size = dnnl_backend.get_mem_size(conv_outputs[0]);
    test::vector<float> conv_dst(conv_dst_size / sizeof(float), 0.0);
    size_t bn_dst_size = dnnl_backend.get_mem_size(bn_outputs[0]);
    test::vector<float> bn_dst(bn_dst_size / sizeof(float), 0.0);

    impl::tensor_t conv_src_ts(conv_src_lt, conv_src.data());
    impl::tensor_t conv_weight_ts(conv_weight_lt, conv_weight.data());
    impl::tensor_t conv_dst_ts(conv_outputs[0], conv_dst.data());

    impl::stream_t &strm = get_stream();
    conv_kernel->execute(
            &conv_op, &strm, {conv_src_ts, conv_weight_ts}, {conv_dst_ts});

    impl::tensor_t bn_src_ts(conv_outputs[0], conv_dst.data());
    impl::tensor_t bn_gamma_ts(gamma, bn_gamma.data());
    impl::tensor_t bn_beta_ts(beta, bn_beta.data());
    impl::tensor_t bn_scale_ts(scale, bn_scale.data());
    impl::tensor_t bn_shift_ts(shift, bn_shift.data());
    impl::tensor_t bn_dst_ts(bn_outputs[0], bn_dst.data());

    bn_kernel->execute(&bn_op, &strm,
            {bn_src_ts, bn_gamma_ts, bn_beta_ts, bn_scale_ts, bn_shift_ts},
            {bn_dst_ts});

    impl::op_t convbn_op(impl::op_kind::conv_bn);
    convbn_op.set_attr<dims>("strides", dims {1, 1});
    convbn_op.set_attr<dims>("dilations", dims {1, 1});
    convbn_op.set_attr<dims>("pads_begin", dims {0, 0});
    convbn_op.set_attr<dims>("pads_end", dims {0, 0});
    convbn_op.set_attr<int64_t>("groups", 0);
    convbn_op.set_attr<std::string>("data_format", "NCX");
    convbn_op.set_attr<std::string>("filter_format", "OIX");
    convbn_op.set_attr("epsilon", 1e-6f);
    impl::logical_tensor_t conv_bn_dst_lt = utils::logical_tensor_init(
            8, {8, 32, 16, 16}, impl::data_type::f32);
    std::vector<impl::logical_tensor_t> convbn_inputs {
            conv_src_lt, conv_weight_lt, gamma, beta, scale, shift};
    std::vector<impl::logical_tensor_t> convbn_outputs {conv_bn_dst_lt};
    auto convbn_kernel = op_factory.create_kernel(convbn_op);
    convbn_kernel->compile(&convbn_op, &eng, convbn_inputs, convbn_outputs);

    size_t convbn_dst_size = dnnl_backend.get_mem_size(convbn_outputs[0]);
    test::vector<float> convbn_dst(convbn_dst_size / sizeof(float), 0.0);
    impl::tensor_t convbn_dst_ts(convbn_outputs[0], convbn_dst.data());

    convbn_kernel->execute(&convbn_op, &strm,
            {conv_src_ts, conv_weight_ts, bn_gamma_ts, bn_beta_ts, bn_scale_ts,
                    bn_shift_ts},
            {convbn_dst_ts});
    strm.wait();

    float max_diff = 0;
    for (size_t i = 0; i < bn_dst.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(bn_dst[i] - convbn_dst[i]));
    }
    ASSERT_LT(max_diff, 1e-6f);
}

TEST(operator_kernel, conv_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {1.0, 2.0, 3.0, 4.0};
    test::vector<float> ref_dst {0.0, 4.5, 8.0, 5.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_add_op(impl::op_kind::conv_add);
    conv_add_op.set_attr<dims>("strides", dims {1, 1});
    conv_add_op.set_attr<dims>("dilations", dims {1, 1});
    conv_add_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_add_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_add_op.set_attr<int64_t>("groups", 1);
    conv_add_op.set_attr<std::string>("data_format", "NCX");
    conv_add_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_add_kernel = op_factory.create_kernel(conv_add_op);

    conv_add_kernel->compile(&conv_add_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(post_src_lt, post_src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_add_kernel->execute(
            &conv_add_op, &strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, conv_per_tensor_broadcast_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0};
    test::vector<float> ref_dst {2.0, 5.5, 8.0, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_add_op(impl::op_kind::conv_add);
    conv_add_op.set_attr<dims>("strides", {1, 1});
    conv_add_op.set_attr<dims>("dilations", {1, 1});
    conv_add_op.set_attr<dims>("pads_begin", {0, 0});
    conv_add_op.set_attr<dims>("pads_end", {0, 0});
    conv_add_op.set_attr<int64_t>("groups", 1);
    conv_add_op.set_attr<std::string>("data_format", "NCX");
    conv_add_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    // post_src will first be unsequeeze to {1,1,1,1} and then broadcast
    // to {1,1,2,2}
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_add_kernel = op_factory.create_kernel(conv_add_op);

    conv_add_kernel->compile(&conv_add_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(post_src_lt, post_src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_add_kernel->execute(
            &conv_add_op, &strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, conv_expanded_per_tensor_broadcast_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0};
    test::vector<float> ref_dst {2.0, 5.5, 8.0, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_add_op(impl::op_kind::conv_add);
    conv_add_op.set_attr<dims>("strides", {1, 1});
    conv_add_op.set_attr<dims>("dilations", {1, 1});
    conv_add_op.set_attr<dims>("pads_begin", {0, 0});
    conv_add_op.set_attr<dims>("pads_end", {0, 0});
    conv_add_op.set_attr<int64_t>("groups", 1);
    conv_add_op.set_attr<std::string>("data_format", "NCX");
    conv_add_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 1, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_add_kernel = op_factory.create_kernel(conv_add_op);

    conv_add_kernel->compile(&conv_add_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(post_src_lt, post_src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_add_kernel->execute(
            &conv_add_op, &strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, conv_per_channel_broadcast_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0, 3.0};
    test::vector<float> ref_dst {2.0, 5.5, 8.0, 4.5, 2.0, 5.5, 8.0, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_add_op(impl::op_kind::conv_add);
    conv_add_op.set_attr<dims>("strides", {1, 1});
    conv_add_op.set_attr<dims>("dilations", {1, 1});
    conv_add_op.set_attr<dims>("pads_begin", {0, 0});
    conv_add_op.set_attr<dims>("pads_end", {0, 0});
    conv_add_op.set_attr<int64_t>("groups", 1);
    conv_add_op.set_attr<std::string>("data_format", "NCX");
    conv_add_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {2, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_add_kernel = op_factory.create_kernel(conv_add_op);

    conv_add_kernel->compile(&conv_add_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(post_src_lt, post_src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_add_kernel->execute(
            &conv_add_op, &strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, conv_nxc_per_channel_broadcast_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};
    test::vector<float> post_src {3.0, 3.0};
    test::vector<float> ref_dst {2.0, 2.0, 5.5, 5.5, 8.0, 8.0, 4.5, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_add_op(impl::op_kind::conv_add);
    conv_add_op.set_attr<dims>("strides", {1, 1});
    conv_add_op.set_attr<dims>("dilations", {1, 1});
    conv_add_op.set_attr<dims>("pads_begin", {0, 0});
    conv_add_op.set_attr<dims>("pads_end", {0, 0});
    conv_add_op.set_attr<int64_t>("groups", 1);
    conv_add_op.set_attr<std::string>("data_format", "NXC");
    conv_add_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 4, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {3, 3, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 1, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_add_kernel = op_factory.create_kernel(conv_add_op);

    conv_add_kernel->compile(&conv_add_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(post_src_lt, post_src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_add_kernel->execute(
            &conv_add_op, &strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, conv_unsupported_broadcast_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0, 3.0};
    impl::op_t conv_add_op(impl::op_kind::conv_add);
    conv_add_op.set_attr<dims>("strides", {1, 1});
    conv_add_op.set_attr<dims>("dilations", {1, 1});
    conv_add_op.set_attr<dims>("pads_begin", {0, 0});
    conv_add_op.set_attr<dims>("pads_end", {0, 0});
    conv_add_op.set_attr<int64_t>("groups", 1);
    conv_add_op.set_attr<std::string>("data_format", "NCX");
    conv_add_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_add_kernel = op_factory.create_kernel(conv_add_op);

    ASSERT_EQ(conv_add_kernel->compile(&conv_add_op, &eng, inputs, outputs),
            impl::status::compile_fail);
}

TEST(operator_kernel, conv_add_relu) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-1.0, -2.0, -3.0, -4.0};
    test::vector<float> ref_dst {0.0, 0.5, 2.0, 0.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_add_relu_op(impl::op_kind::conv_add_relu);
    conv_add_relu_op.set_attr<dims>("strides", dims {1, 1});
    conv_add_relu_op.set_attr<dims>("dilations", dims {1, 1});
    conv_add_relu_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_add_relu_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_add_relu_op.set_attr<int64_t>("groups", 1);
    conv_add_relu_op.set_attr<std::string>("data_format", "NCX");
    conv_add_relu_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_add_relu_kernel = op_factory.create_kernel(conv_add_relu_op);

    conv_add_relu_kernel->compile(&conv_add_relu_op, &eng, inputs, outputs);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(dst_lt, post_src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    impl::stream_t &strm = get_stream();
    conv_add_relu_kernel->execute(&conv_add_relu_op, &strm,
            {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, abs) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Abs);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, elu) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Elu);
    op.set_attr<float>("alpha", 1.f);
    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp;
        if (src[i] > 0) {
            temp = src[i];
        } else {
            temp = static_cast<float>(exp(src[i]) - 1);
        }
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, exp) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Exp);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(exp(src[i]));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, hardtanh) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {-1.0, -1.0, -1.0, -0.5, 0.0, 2.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::HardTanh);
    op.set_attr<float>("max", 2.f);
    op.set_attr<float>("min", -1.f);
    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, sqrt) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Sqrt);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(sqrt(src[i]));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, square) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Square);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = src[i] * src[i];
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, pow) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Pow);
    op.set_attr<float>("alpha", 1.f);
    op.set_attr<float>("beta", 3.f);
    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(pow(src[i], 3.0));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, log) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.f, 1.5f, 1.f, 0.5f, 0.8f, 3.5f};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Log);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(log(src[i]));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_TRUE(std::fabs(dst[i] - ref_dst[i]) < 0.00001);
    }
}

TEST(operator_kernel, tanh) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Tanh);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(
                (exp(src[i]) - exp(-src[i])) / (exp(src[i]) + exp(-src[i])));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_abs) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {2.0, 1.5, 4.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    impl::op_t op(impl::op_kind::conv_bias_abs);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");
    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_elu) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {-2.0, 1.5, 4.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    ref_dst[0] = static_cast<float>(exp(-2) - 1);
    impl::op_t op(impl::op_kind::conv_bias_elu);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);
    op.set_attr<float>("alpha", 1.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");
    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_hardtanh) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {0.0, 1.5, 3.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t op(impl::op_kind::conv_bias_hardtanh);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);
    op.set_attr<float>("min", 0.f);
    op.set_attr<float>("max", 3.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");
    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_relu6) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {2.0};
    test::vector<float> ref_dst {1.0, 4.5, 6.0, 3.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t op(impl::op_kind::conv_bias_relu6);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);
    op.set_attr<float>("min", 0.f);
    op.set_attr<float>("max", 6.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");
    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_sigmoid) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {-2.0, 1.5, 4.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    for (auto &rdst : ref_dst) {
        rdst = static_cast<float>(1 / (exp(-rdst) + 1));
    }
    impl::op_t op(impl::op_kind::conv_bias_sigmoid);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");
    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_sqrt) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {1.0};
    test::vector<float> ref_dst {0.0, 3.5, 6.0, 2.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    for (auto &rdst : ref_dst) {
        rdst = static_cast<float>(sqrt(rdst));
    }
    impl::op_t op(impl::op_kind::conv_bias_sqrt);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_square) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {4.0, 2.25, 16.0, 0.25};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    impl::op_t op(impl::op_kind::conv_bias_square);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_tanh) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {-2.0, 1.5, 4.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    for (auto &rdst : ref_dst) {
        rdst = static_cast<float>(
                (exp(rdst) - exp(-rdst)) / (exp(rdst) + exp(-rdst)));
    }
    impl::op_t op(impl::op_kind::conv_bias_tanh);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, bias_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_add_elu) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> bias {-1.0};
    test::vector<float> ref_dst {-4.0, 2.5, 3.0, 0.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    ref_dst[0] = static_cast<float>(exp(-4) - 1);
    impl::op_t op(impl::op_kind::conv_bias_add_elu);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<float>("alpha", 1.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t post_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {
            src_lt, weight_lt, bias_lt, post_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t post_src_ts(outputs[0], post_src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(
            &op, &strm, {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_bias_add_relu6) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> bias {3.0};
    test::vector<float> ref_dst {0.0, 6.f, 6.f, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    impl::op_t op(impl::op_kind::conv_bias_add_relu6);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<float>("min", 0.f);
    op.set_attr<float>("max", 6.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t post_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {
            src_lt, weight_lt, bias_lt, post_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t bias_ts(bias_lt, bias.data());
    impl::tensor_t post_src_ts(outputs[0], post_src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(
            &op, &strm, {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_add_elu) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> ref_dst {-3.0, 3.5, 4.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    for (auto &rdst : ref_dst) {
        rdst = rdst > 0 ? rdst : static_cast<float>((exp(rdst) - 1));
    }

    impl::op_t op(impl::op_kind::conv_add_elu);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<float>("alpha", 1.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(outputs[0], post_src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, conv_add_relu6) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> ref_dst {0.0, 3.5, 4.f, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    impl::op_t op(impl::op_kind::conv_add_relu6);

    op.set_attr<dims>("strides", {1, 1});
    op.set_attr<dims>("dilations", {1, 1});
    op.set_attr<dims>("pads_begin", {0, 0});
    op.set_attr<dims>("pads_end", {0, 0});
    op.set_attr<int64_t>("groups", 1);

    op.set_attr<float>("min", 0.f);
    op.set_attr<float>("max", 6.f);
    op.set_attr<std::string>("data_format", "NCX");
    op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, weight_lt, post_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t weight_ts(weight_lt, weight.data());
    impl::tensor_t post_src_ts(outputs[0], post_src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, softmax) {
    impl::op_t softmax_op(impl::op_kind::SoftMax);
    impl::engine_t &eng = get_engine();

    softmax_op.set_attr<int64_t>("axis", 0);

    test::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    test::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(softmax_op);

    softmax_kernel->compile(&softmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(&softmax_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, softmax_with_last_dim) {
    impl::op_t softmax_op(impl::op_kind::SoftMax);
    impl::engine_t &eng = get_engine();

    softmax_op.set_attr<int64_t>("axis", -1);

    test::vector<float> src_data {3.0, 3.0, 1.0, 1.0};
    test::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(softmax_op);

    softmax_kernel->compile(&softmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(&softmax_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, softmax_backward) {
    impl::op_t softmax_op(impl::op_kind::SoftMaxBackprop);
    impl::engine_t &eng = get_engine();

    softmax_op.set_attr<int64_t>("axis", 1);

    test::vector<float> dst {0.5, 0.5, 0.5, 0.5};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-1, 1, -0.5, 0.5};
    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            3, {2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {diff_dst_lt, dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(softmax_op);

    softmax_kernel->compile(&softmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t dst_ts(dst_lt, dst.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(
            &softmax_op, &strm, {diff_dst_ts, dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < ref_diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, logsoftmax) {
    impl::op_t logsoftmax_op(impl::op_kind::LogSoftmax);
    impl::engine_t &eng = get_engine();

    logsoftmax_op.set_attr<int64_t>("axis", 0);

    test::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    test::vector<float> ref_dst_data {-0.6931472, -0.6931472, -0.6931472,
            -0.6931472, -0.6931472, -0.6931472};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(logsoftmax_op);

    softmax_kernel->compile(&logsoftmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, src_data.data());
    impl::tensor_t dst_ts(outputs[0], dst_data.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(&logsoftmax_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, logsoftmax_backward) {
    impl::op_t logsoftmax_op(impl::op_kind::LogSoftmaxBackprop);
    impl::engine_t &eng = get_engine();

    logsoftmax_op.set_attr<int64_t>("axis", 1);

    test::vector<float> dst {-0.6931472, -0.6931472, -0.6931472, -0.6931472};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-2, 2, -1, 1};
    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            3, {2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {diff_dst_lt, dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(logsoftmax_op);

    softmax_kernel->compile(&logsoftmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t dst_ts(dst_lt, dst.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(
            &logsoftmax_op, &strm, {diff_dst_ts, dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < ref_diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, avg_pool_backward_exclude_pad) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {-1.0, 1.5, 1.5, 10.0, 2.0, 4.0, 4.0, 4.0,
            2.0, 4.0, 4.0, 4.0, 12.0, -2.5, -2.5, -3.0};
    test::vector<float> diff_dst {
            -1.0, 3.0, 10.0, 4.0, 16.0, 8.0, 12.0, -5.0, -3.0};

    impl::op_t avg_pool_bwd_op(impl::op_kind::AvgPoolBackprop);
    avg_pool_bwd_op.set_attr<dims>("strides", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_bwd_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_bwd_op.set_attr<bool>("exclude_pad", true);

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, diff_dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_bwd_kernel = op_factory.create_kernel(avg_pool_bwd_op);

    avg_pool_bwd_kernel->compile(&avg_pool_bwd_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());

    impl::stream_t &strm = get_stream();
    avg_pool_bwd_kernel->execute(
            &avg_pool_bwd_op, &strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, avg_pool_backward_include_pad) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {-0.25, 0.75, 0.75, 2.5, 1.0, 4.0, 4.0,
            2.0, 1.0, 4.0, 4.0, 2.0, 3.0, -1.25, -1.25, -3.0 / 4};
    test::vector<float> diff_dst {
            -1.0, 3.0, 10.0, 4.0, 16.0, 8.0, 12.0, -5.0, -3.0};

    impl::op_t avg_pool_bwd_op(impl::op_kind::AvgPoolBackprop);
    avg_pool_bwd_op.set_attr<dims>("strides", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_bwd_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_bwd_op.set_attr<bool>("exclude_pad", false);

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, diff_dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_bwd_kernel = op_factory.create_kernel(avg_pool_bwd_op);

    avg_pool_bwd_kernel->compile(&avg_pool_bwd_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());

    impl::stream_t &strm = get_stream();
    avg_pool_bwd_kernel->execute(
            &avg_pool_bwd_op, &strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, max_pool_backward_with_incides) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0.0, 1.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {0.0, 0.0, 16.0, 0.0, 4.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0};

    test::vector<float> diff_dst {4.0, 16.0, 8.0, 12.0};
#if DNNL_GRAPH_WITH_SYCL
    test::vector<int32_t> indices {2, 0, 1, 3};
#else
    test::vector<uint8_t> indices {2, 0, 1, 3};
#endif

    impl::op_t max_pool_bwd_op(impl::op_kind::MaxPoolBackprop);

    max_pool_bwd_op.set_attr<dims>("strides", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("kernel", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("pads_begin", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("pads_end", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("dilations", {1, 1});

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
#if DNNL_GRAPH_WITH_SYCL
    impl::logical_tensor_t indices_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::s32);
#else
    impl::logical_tensor_t indices_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::u8);
#endif

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_pool_bwd_kernel = op_factory.create_kernel(max_pool_bwd_op);
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};
    max_pool_bwd_kernel->compile(
            &max_pool_bwd_op, &eng, {src_lt, diff_dst_lt}, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());
    impl::tensor_t indices_ts(indices_lt, indices.data());

    impl::stream_t &strm = get_stream();
    max_pool_bwd_kernel->execute(&max_pool_bwd_op, &strm,
            {src_ts, indices_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, max_pool_backward_without_incides) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0.0, 1.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {0.0, 0.0, 16.0, 0.0, 4.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0};

    test::vector<float> diff_dst {4.0, 16.0, 8.0, 12.0};

    impl::op_t max_pool_bwd_op(impl::op_kind::MaxPoolBackprop);

    max_pool_bwd_op.set_attr<dims>("strides", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("kernel", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("pads_begin", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("pads_end", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("dilations", {1, 1});

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};
    auto &op_factory = get_dnnl_kernel_registry();
    auto max_pool_bwd_kernel = op_factory.create_kernel(max_pool_bwd_op);
    max_pool_bwd_kernel->compile(
            &max_pool_bwd_op, &eng, {src_lt, diff_dst_lt}, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], diff_src.data());

    impl::stream_t &strm = get_stream();
    max_pool_bwd_kernel->execute(
            &max_pool_bwd_op, &strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, layernorm_training) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 5.0, 2.0, 3.0, 5.0};
    test::vector<float> scale {1.0, 2.0};
    test::vector<float> shift {0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 3.0, 1.0, -1.0, -1.0, 3.0};
    test::vector<float> ref_mean {3.0, 3.5, 4.0};
    test::vector<float> ref_var {1.0, 2.25, 1.0};
    test::vector<float> dst(src.size(), 0.0);
    test::vector<float> mean(ref_mean.size(), 0.0);
    test::vector<float> var(ref_var.size(), 0.0);

    impl::op_t op(impl::op_kind::LayerNorm);

    op.set_attr<float>("epsilon", 0);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 2}, impl::data_type::f32);
    impl::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 3, 2}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t mean_lt = utils::logical_tensor_init(
            4, {3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t variance_lt = utils::logical_tensor_init(
            5, {3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt, scale_lt, shift_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt, mean_lt, variance_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);

    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);
    ASSERT_EQ(outputs[1].layout_type, impl::layout_type::opaque);
    ASSERT_EQ(outputs[2].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t scale_ts(scale_lt, scale.data());
    impl::tensor_t shift_ts(shift_lt, shift.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());
    impl::tensor_t mean_ts(outputs[1], mean.data());
    impl::tensor_t var_ts(outputs[2], var.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, scale_ts, shift_ts},
            {dst_ts, mean_ts, var_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    for (size_t i = 0; i < ref_mean.size(); ++i) {
        ASSERT_FLOAT_EQ(mean[i], ref_mean[i]);
        ASSERT_FLOAT_EQ(var[i], ref_var[i]);
    }
}

TEST(operator_kernel, layernorm_inference) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    test::vector<float> scale {1.0, 2.0};
    test::vector<float> shift {0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 3.0, -1.0, 3.0, 1.0, -1.0, -1.0, 3.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::LayerNorm);

    op.set_attr<float>("epsilon", 0);
    op.set_attr<bool>("keep_stats", false); //inference

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {2, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt, scale_lt, shift_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);

    kernel->compile(&op, &eng, inputs, outputs);

    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t scale_ts(scale_lt, scale.data());
    impl::tensor_t shift_ts(shift_lt, shift.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, scale_ts, shift_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, layernorm_inference_without_scale_shift) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    test::vector<float> ref_dst {-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::LayerNorm);

    op.set_attr<float>("epsilon", 0);
    op.set_attr<bool>("keep_stats", false); //inference
    op.set_attr<bool>("use_affine", false);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {2, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);

    kernel->compile(&op, &eng, inputs, outputs);

    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(outputs[0], dst.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, convert_data) {
    impl::engine_t &engine = get_engine();

    test::vector<int8_t> src {1, 2, 3, 4, 5, 6};
    test::vector<int32_t> ref_dst {1, 2, 3, 4, 5, 6};
    test::vector<int32_t> dst(src.size(), 0);

    impl::op_t convert_op(impl::op_kind::convert);

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::s8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 3}, impl::data_type::s32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(convert_op);

    kernel->compile(&convert_op, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    kernel->execute(&convert_op, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, quantize_per_tensor) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>("scales", {0.1f});
    quantize.set_attr<std::vector<int64_t>>("zps", {10});
    quantize.set_attr<std::string>("qtype", "per_tensor");
    quantize.set_attr<int64_t>("axis", 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<uint8_t> dst(src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::u8);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(quantize);

    op->compile(&quantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&quantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, quantize_per_tensor_any) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>("scales", {0.1f});
    quantize.set_attr<std::vector<int64_t>>("zps", {10});
    quantize.set_attr<std::string>("qtype", "per_tensor");
    quantize.set_attr<int64_t>("axis", 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<uint8_t> dst(src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2}, impl::data_type::u8, impl::layout_type::any);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(quantize);

    op->compile(&quantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&quantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, quantize_per_channel_symmetric) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>("scales", {0.1f, 0.2f});
    quantize.set_attr<std::vector<int64_t>>("zps", {0, 0});
    quantize.set_attr<std::string>("qtype", "per_channel");
    quantize.set_attr<int64_t>("axis", 1);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<int8_t> dst(src.size(), 0);

    // int8 = f32 / scales
    test::vector<int8_t> ref_dst {-10, 0, 5, 10};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::s8);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(quantize);

    op->compile(&quantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&quantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, dequantize_per_tensor) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>("scales", {0.1f});
    dequantize.set_attr<std::vector<int64_t>>("zps", {10});
    dequantize.set_attr<std::string>("qtype", "per_tensor");
    dequantize.set_attr<int64_t>("axis", 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {-1.0, 0.0, 1.0, 2.0};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(dequantize);

    op->compile(&dequantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&dequantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, dequantize_per_tensor_any) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>("scales", {0.1f});
    dequantize.set_attr<std::vector<int64_t>>("zps", {10});
    dequantize.set_attr<std::string>("qtype", "per_tensor");
    dequantize.set_attr<int64_t>("axis", 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {-1.0, 0.0, 1.0, 2.0};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(dequantize);

    op->compile(&dequantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&dequantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, dequantize_per_channel_symmetric) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>("scales", {0.1f, 0.2f});
    dequantize.set_attr<std::vector<int64_t>>("zps", {0, 0});
    dequantize.set_attr<std::string>("qtype", "per_channel");
    dequantize.set_attr<int64_t>("axis", 1);

    test::vector<int8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * int8
    test::vector<float> ref_dst {0.0, 1, 4, 6};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::s8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(dequantize);

    op->compile(&dequantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, src.data());
    impl::tensor_t dst_ts(dst_lt, dst.data());

    op->execute(&dequantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

template <typename T>
static bool allclose(const test::vector<T> &a, const test::vector<T> &b,
        float rtol, float atol) {
    if (a.size() != b.size()) return false;
    bool flag = true;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]))
                > (atol + rtol * std::abs(static_cast<float>(b[i])))) {
            flag = false;
            break;
        }
    }
    return flag;
}

namespace {
dnnl::graph::impl::pass::pass_base_ptr get_pass(const std::string &pass_name) {
    auto &backend_ptr
            = dnnl::graph::impl::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::graph::impl::pass::pass_manager(
            backend_ptr.get_pass_registry());
    auto &passes = pm.get_passes();
    auto find = std::find_if(passes.begin(), passes.end(),
            [&pass_name](const dnnl::graph::impl::pass::pass_base_ptr &p)
                    -> bool { return p->get_pass_name() == pass_name; });

    return *find;
}
} // namespace

// FIXME(qun) If the atol and rtol in the following cases are too small, then
// these cases may can't pass when using AVX2 or AVX512, because of the DNNL
// issue: https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html
// Please use machine with DL Boost, such as clx or cpx, to avoid this issue.

#define for_ for
#define SET_Q_DQ_DATA_ATTR(q_dq_data) \
    q_dq_data.set_attr<std::string>("qtype", "per_tensor"); \
    q_dq_data.set_attr<std::string>("out_type", "uint8"); \
    q_dq_data.set_attr<std::vector<int64_t>>("zps", {zp_src}); \
    q_dq_data.set_attr<std::vector<float>>("scales", {scale_src}); \
    q_dq_data.set_attr<int64_t>("axis", 0);

#define SET_Q_DQ_WEIGHT_ATTR(q_dq_weight) \
    q_dq_weight.set_attr<std::string>("qtype", wei_qtype); \
    q_dq_weight.set_attr<std::string>("out_type", "int8"); \
    q_dq_weight.set_attr<std::vector<int64_t>>("zps", zp_wei); \
    q_dq_weight.set_attr<std::vector<float>>("scales", scale_wei); \
    q_dq_weight.set_attr<int64_t>("axis", 0);

#define SET_CONV_ATTR(conv, nd) \
    conv.set_attr<dims>("strides", dims(nd, 1)); \
    conv.set_attr<dims>("dilations", dims(nd, 1)); \
    conv.set_attr<dims>("pads_begin", dims(nd, 0)); \
    conv.set_attr<dims>("pads_end", dims(nd, 0)); \
    conv.set_attr<int64_t>("groups", g); \
    conv.set_attr<std::string>("data_format", "NCX"); \
    conv.set_attr<std::string>("filter_format", "OIX");

#define SET_Q_DQ_OUT_ATTR(q_dq_out) \
    q_dq_out.set_attr<std::string>("qtype", "per_tensor"); \
    q_dq_out.set_attr<std::string>("out_type", "int8"); \
    q_dq_out.set_attr<std::vector<int64_t>>("zps", {zp_out}); \
    q_dq_out.set_attr<std::vector<float>>("scales", {scale_out}); \
    q_dq_out.set_attr<int64_t>("axis", 0);

// For asymmetric quantization, the following case will only pass on machine
// with DL Boost, such as clx or cpx,
TEST(int8_subgraph_mode, int8_conv1d_conv2d_conv3d) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && engine.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for_(const auto &src_qtype : src_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (isa < dnnl_cpu_isa_avx512_core_vnni && src_qtype == "asymmetric")
            continue;

        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {out_channel, in_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {out_channel, in_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {out_channel, in_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 10}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 10, 10}
                          : std::vector<int64_t> {1, out_channel, 10, 10, 10};

        test::vector<float> src_data(product(src_shape));
        test::vector<float> weight_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(),
                [&]() { return distribution(generator); });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return distribution(generator); });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return distribution(generator); });
        }
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = src_qtype == "symmetric" ? 0 : 128;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        // -------------------------case 1----------------------------------
        impl::op_t qdata_node(0, impl::op_kind::Quantize, "qdata_node");
        SET_Q_DQ_DATA_ATTR(qdata_node)

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t qweight_node(2, impl::op_kind::Quantize, "qweight_node");
        SET_Q_DQ_WEIGHT_ATTR(qweight_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, nd)

        impl::op_t qout_node(5, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        // create kernels
        auto kernel_qdata
                = get_dnnl_kernel_registry().create_kernel(qdata_node);
        auto kernel_dqdata
                = get_dnnl_kernel_registry().create_kernel(dqdata_node);
        auto kernel_qweight
                = get_dnnl_kernel_registry().create_kernel(qweight_node);
        auto kernel_dqweight
                = get_dnnl_kernel_registry().create_kernel(dqweight_node);
        auto kernel_conv = get_dnnl_kernel_registry().create_kernel(conv_node);
        auto kernel_qout = get_dnnl_kernel_registry().create_kernel(qout_node);

        // prepare logical tensor
        impl::logical_tensor_t src_f32 = utils::logical_tensor_init(
                0, src_shape, impl::data_type::f32);
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(8, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }

        // compile
        kernel_qdata->compile(&qdata_node, &engine, {src_f32}, {src_u8});
        kernel_dqdata->compile(&dqdata_node, &engine, {src_u8}, {src_f32_dq});
        kernel_qweight->compile(
                &qweight_node, &engine, {weight_f32}, {weight_s8});
        kernel_dqweight->compile(
                &dqweight_node, &engine, {weight_s8}, {weight_f32_dq});
        if (with_bias)
            kernel_conv->compile(&conv_node, &engine,
                    {src_f32_dq, weight_f32_dq, bias_f32}, {dst_f32});
        else
            kernel_conv->compile(&conv_node, &engine,
                    {src_f32_dq, weight_f32_dq}, {dst_f32});
        kernel_qout->compile(&qout_node, &engine, {dst_f32}, {dst_s8});

        // execute
        test::vector<uint8_t> src_u8_data(product(src_shape));
        impl::tensor_t src_f32_ts(src_f32, src_data.data());
        impl::tensor_t src_u8_ts(src_u8, src_u8_data.data());
        kernel_qdata->execute(&qdata_node, &strm, {src_f32_ts}, {src_u8_ts});
        strm.wait();

        test::vector<float> src_f32_dq_data(product(src_shape));
        impl::tensor_t src_f32_dq_ts(src_f32_dq, src_f32_dq_data.data());
        kernel_dqdata->execute(
                &dqdata_node, &strm, {src_u8_ts}, {src_f32_dq_ts});
        strm.wait();

        test::vector<int8_t> weight_s8_data(product(weight_shape));
        impl::tensor_t weight_f32_ts(weight_f32, weight_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, weight_s8_data.data());
        kernel_qweight->execute(
                &qweight_node, &strm, {weight_f32_ts}, {weight_s8_ts});
        strm.wait();

        test::vector<float> weight_f32_dq_data(product(weight_shape));
        impl::tensor_t weight_f32_dq_ts(
                weight_f32_dq, weight_f32_dq_data.data());
        kernel_dqweight->execute(
                &dqweight_node, &strm, {weight_s8_ts}, {weight_f32_dq_ts});
        strm.wait();

        test::vector<float> out_f32_data(product(dst_shape));
        impl::tensor_t bias_f32_ts;
        impl::tensor_t dst_f32_ts(dst_f32, out_f32_data.data());
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, bias_data.data());
            kernel_conv->execute(&conv_node, &strm,
                    {src_f32_dq_ts, weight_f32_dq_ts, bias_f32_ts},
                    {dst_f32_ts});
        } else {
            kernel_conv->execute(&conv_node, &strm,
                    {src_f32_dq_ts, weight_f32_dq_ts}, {dst_f32_ts});
        }
        strm.wait();

        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, case1_out_data.data());
        kernel_qout->execute(&qout_node, &strm, {dst_f32_ts}, {dst_s8_ts});
        strm.wait();

        // -------------------------case 2----------------------------------
        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        qout_node.add_input(dst_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&qout_node);
        g.build_graph();

        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("int8_conv_bias_fusion")
                : get_pass("int8_conv_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_s8_case2_ts(dst_s8, case2_out_data.data());
        if (with_bias)
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
        strm.wait();

        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_conv2d_relu) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    if (engine.kind() == impl::engine_kind::gpu) return;

    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        test::vector<float> src_data(product(src_shape));
        test::vector<float> weight_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(),
                [&]() { return distribution(generator); });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return distribution(generator); });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return distribution(generator); });
        }
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        // -------------------------case 1----------------------------------
        impl::op_t qdata_node(0, impl::op_kind::Quantize, "qdata_node");
        SET_Q_DQ_DATA_ATTR(qdata_node)

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t qweight_node(2, impl::op_kind::Quantize, "qweight_node");
        SET_Q_DQ_WEIGHT_ATTR(qweight_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        impl::op_t qout_node(6, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        // create kernels
        auto kernel_qdata
                = get_dnnl_kernel_registry().create_kernel(qdata_node);
        auto kernel_dqdata
                = get_dnnl_kernel_registry().create_kernel(dqdata_node);
        auto kernel_qweight
                = get_dnnl_kernel_registry().create_kernel(qweight_node);
        auto kernel_dqweight
                = get_dnnl_kernel_registry().create_kernel(dqweight_node);
        auto kernel_conv = get_dnnl_kernel_registry().create_kernel(conv_node);
        auto kernel_relu = get_dnnl_kernel_registry().create_kernel(relu_node);
        auto kernel_qout = get_dnnl_kernel_registry().create_kernel(qout_node);

        // prepare logical tensor
        impl::logical_tensor_t src_f32 = utils::logical_tensor_init(
                0, src_shape, impl::data_type::f32);
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }

        // compile
        kernel_qdata->compile(&qdata_node, &engine, {src_f32}, {src_u8});
        kernel_dqdata->compile(&dqdata_node, &engine, {src_u8}, {src_f32_dq});
        kernel_qweight->compile(
                &qweight_node, &engine, {weight_f32}, {weight_s8});
        kernel_dqweight->compile(
                &dqweight_node, &engine, {weight_s8}, {weight_f32_dq});
        if (with_bias)
            kernel_conv->compile(&conv_node, &engine,
                    {src_f32_dq, weight_f32_dq, bias_f32}, {dst_f32});
        else
            kernel_conv->compile(&conv_node, &engine,
                    {src_f32_dq, weight_f32_dq}, {dst_f32});
        kernel_relu->compile(&relu_node, &engine, {dst_f32}, {dst_relu_f32});
        kernel_qout->compile(&qout_node, &engine, {dst_relu_f32}, {dst_s8});

        // execute
        test::vector<uint8_t> src_u8_data(product(src_shape));
        impl::tensor_t src_f32_ts(src_f32, src_data.data());
        impl::tensor_t src_u8_ts(src_u8, src_u8_data.data());
        kernel_qdata->execute(&qdata_node, &strm, {src_f32_ts}, {src_u8_ts});
        strm.wait();

        test::vector<float> src_f32_dq_data(product(src_shape));
        impl::tensor_t src_f32_dq_ts(src_f32_dq, src_f32_dq_data.data());
        kernel_dqdata->execute(
                &dqdata_node, &strm, {src_u8_ts}, {src_f32_dq_ts});
        strm.wait();

        test::vector<int8_t> weight_s8_data(product(weight_shape));
        impl::tensor_t weight_f32_ts(weight_f32, weight_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, weight_s8_data.data());
        kernel_qweight->execute(
                &qweight_node, &strm, {weight_f32_ts}, {weight_s8_ts});
        strm.wait();

        test::vector<float> weight_f32_dq_data(product(weight_shape));
        impl::tensor_t weight_f32_dq_ts(
                weight_f32_dq, weight_f32_dq_data.data());
        kernel_dqweight->execute(
                &dqweight_node, &strm, {weight_s8_ts}, {weight_f32_dq_ts});
        strm.wait();

        test::vector<float> out_f32_data(product(dst_shape));
        impl::tensor_t bias_f32_ts;
        impl::tensor_t dst_f32_ts(dst_f32, out_f32_data.data());
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, bias_data.data());
            kernel_conv->execute(&conv_node, &strm,
                    {src_f32_dq_ts, weight_f32_dq_ts, bias_f32_ts},
                    {dst_f32_ts});
        } else {
            kernel_conv->execute(&conv_node, &strm,
                    {src_f32_dq_ts, weight_f32_dq_ts}, {dst_f32_ts});
        }
        strm.wait();

        test::vector<float> out_relu_f32_data(product(dst_shape));
        impl::tensor_t dst_relu_f32_ts(dst_relu_f32, out_relu_f32_data.data());
        kernel_relu->execute(
                &relu_node, &strm, {dst_f32_ts}, {dst_relu_f32_ts});
        strm.wait();

        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, case1_out_data.data());
        kernel_qout->execute(&qout_node, &strm, {dst_relu_f32_ts}, {dst_s8_ts});
        strm.wait();

        // -------------------------case 2----------------------------------
        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        relu_node.add_input(dst_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.build_graph();

        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("int8_conv_bias_relu_fusion")
                : get_pass("int8_conv_relu_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_s8_case2_ts(dst_s8, case2_out_data.data());
        if (with_bias)
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
        strm.wait();

        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_conv2d_sum_relu) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        test::vector<float> src_data(product(src_shape));
        test::vector<float> weight_data(product(weight_shape));
        test::vector<float> other_data(product(dst_shape));

        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(),
                [&]() { return distribution(generator); });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return distribution(generator); });
        std::generate(other_data.begin(), other_data.end(),
                [&]() { return distribution(generator); });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return distribution(generator); });
        }
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        // -------------------------case 1----------------------------------
        impl::op_t qdata_node(0, impl::op_kind::Quantize, "qdata_node");
        SET_Q_DQ_DATA_ATTR(qdata_node)

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t qweight_node(2, impl::op_kind::Quantize, "qweight_node");
        SET_Q_DQ_WEIGHT_ATTR(qweight_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        impl::op_t qout_node(6, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        impl::op_t qother_node(7, impl::op_kind::Quantize, "qother_node");
        qother_node.set_attr<std::string>("qtype", "per_tensor");
        qother_node.set_attr<std::string>("out_type", "int8");
        qother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        qother_node.set_attr<std::vector<float>>("scales", {scale_other});
        qother_node.set_attr<int64_t>("axis", 0);

        impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>("qtype", "per_tensor");
        dqother_node.set_attr<std::string>("in_type", "int8");
        dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_node.set_attr<int64_t>("axis", 0);

        impl::op_t add_node(9, impl::op_kind::Add, "add_node");

        // create kernels
        auto kernel_qdata
                = get_dnnl_kernel_registry().create_kernel(qdata_node);
        auto kernel_dqdata
                = get_dnnl_kernel_registry().create_kernel(dqdata_node);
        auto kernel_qweight
                = get_dnnl_kernel_registry().create_kernel(qweight_node);
        auto kernel_dqweight
                = get_dnnl_kernel_registry().create_kernel(dqweight_node);
        auto kernel_conv = get_dnnl_kernel_registry().create_kernel(conv_node);
        auto kernel_relu = get_dnnl_kernel_registry().create_kernel(relu_node);
        auto kernel_qout = get_dnnl_kernel_registry().create_kernel(qout_node);
        auto kernel_qother
                = get_dnnl_kernel_registry().create_kernel(qother_node);
        auto kernel_dqother
                = get_dnnl_kernel_registry().create_kernel(dqother_node);
        auto kernel_add = get_dnnl_kernel_registry().create_kernel(add_node);

        // prepare logical tensor
        impl::logical_tensor_t src_f32 = utils::logical_tensor_init(
                0, src_shape, impl::data_type::f32);
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t other_f32 = utils::logical_tensor_init(
                10, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t other_s8 = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t other_f32_dq = utils::logical_tensor_init(
                12, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                13, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }

        // compile
        kernel_qdata->compile(&qdata_node, &engine, {src_f32}, {src_u8});
        kernel_dqdata->compile(&dqdata_node, &engine, {src_u8}, {src_f32_dq});
        kernel_qweight->compile(
                &qweight_node, &engine, {weight_f32}, {weight_s8});
        kernel_dqweight->compile(
                &dqweight_node, &engine, {weight_s8}, {weight_f32_dq});
        if (with_bias)
            kernel_conv->compile(&conv_node, &engine,
                    {src_f32_dq, weight_f32_dq, bias_f32}, {dst_f32});
        else
            kernel_conv->compile(&conv_node, &engine,
                    {src_f32_dq, weight_f32_dq}, {dst_f32});

        kernel_qother->compile(&qother_node, &engine, {other_f32}, {other_s8});
        kernel_dqother->compile(
                &dqother_node, &engine, {other_s8}, {other_f32_dq});
        kernel_add->compile(
                &add_node, &engine, {dst_f32, other_f32_dq}, {dst_add_f32});
        kernel_relu->compile(
                &relu_node, &engine, {dst_add_f32}, {dst_relu_f32});
        kernel_qout->compile(&qout_node, &engine, {dst_relu_f32}, {dst_s8});

        // execute
        test::vector<uint8_t> src_u8_data(product(src_shape));
        impl::tensor_t src_f32_ts(src_f32, src_data.data());
        impl::tensor_t src_u8_ts(src_u8, src_u8_data.data());
        kernel_qdata->execute(&qdata_node, &strm, {src_f32_ts}, {src_u8_ts});
        strm.wait();

        test::vector<float> src_f32_dq_data(product(src_shape));
        impl::tensor_t src_f32_dq_ts(src_f32_dq, src_f32_dq_data.data());
        kernel_dqdata->execute(
                &dqdata_node, &strm, {src_u8_ts}, {src_f32_dq_ts});
        strm.wait();

        test::vector<int8_t> weight_s8_data(product(weight_shape));
        impl::tensor_t weight_f32_ts(weight_f32, weight_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, weight_s8_data.data());
        kernel_qweight->execute(
                &qweight_node, &strm, {weight_f32_ts}, {weight_s8_ts});
        strm.wait();

        test::vector<float> weight_f32_dq_data(product(weight_shape));
        impl::tensor_t weight_f32_dq_ts(
                weight_f32_dq, weight_f32_dq_data.data());
        kernel_dqweight->execute(
                &dqweight_node, &strm, {weight_s8_ts}, {weight_f32_dq_ts});
        strm.wait();

        test::vector<float> out_f32_data(product(dst_shape));
        impl::tensor_t dst_f32_ts(dst_f32, out_f32_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, bias_data.data());
            kernel_conv->execute(&conv_node, &strm,
                    {src_f32_dq_ts, weight_f32_dq_ts, bias_f32_ts},
                    {dst_f32_ts});
        } else {
            kernel_conv->execute(&conv_node, &strm,
                    {src_f32_dq_ts, weight_f32_dq_ts}, {dst_f32_ts});
        }
        strm.wait();

        test::vector<int8_t> other_s8_data(product(dst_shape));
        impl::tensor_t other_f32_ts(other_f32, other_data.data());
        impl::tensor_t other_s8_ts(other_s8, other_s8_data.data());
        kernel_qother->execute(
                &qother_node, &strm, {other_f32_ts}, {other_s8_ts});
        strm.wait();

        test::vector<float> other_f32_dq_data(product(dst_shape));
        impl::tensor_t other_f32_dq_ts(other_f32_dq, other_f32_dq_data.data());
        kernel_dqother->execute(
                &dqother_node, &strm, {other_s8_ts}, {other_f32_dq_ts});
        strm.wait();

        test::vector<float> out_add_f32_data(product(dst_shape));
        impl::tensor_t dst_add_f32_ts(dst_add_f32, out_add_f32_data.data());
        kernel_add->execute(&add_node, &strm, {dst_f32_ts, other_f32_dq_ts},
                {dst_add_f32_ts});
        strm.wait();

        test::vector<float> out_relu_f32_data(product(dst_shape));
        impl::tensor_t dst_relu_f32_ts(dst_relu_f32, out_relu_f32_data.data());
        kernel_relu->execute(
                &relu_node, &strm, {dst_add_f32_ts}, {dst_relu_f32_ts});
        strm.wait();

        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, case1_out_data.data());
        kernel_qout->execute(&qout_node, &strm, {dst_relu_f32_ts}, {dst_s8_ts});
        strm.wait();

        // -------------------------case 2----------------------------------
        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.build_graph();

        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("int8_conv_bias_add_relu_fusion")
                : get_pass("int8_conv_add_relu_fusion");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_s8, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_s8_case2_ts(dst_s8, case2_out_data.data());
        if (with_bias)
            cp.execute(&strm,
                    {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        strm.wait();

        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_conv2d_sum_relu_NXC) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, 12, 12, in_channel};
        std::vector<int64_t> weight_shape {
                kernel_size, kernel_size, in_channel / g, out_channel};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, 10, 10, out_channel};

        test::vector<float> src_data(product(src_shape));
        test::vector<float> weight_data(product(weight_shape));
        test::vector<float> other_data(product(dst_shape));

        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(),
                [&]() { return distribution(generator); });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return distribution(generator); });
        std::generate(other_data.begin(), other_data.end(),
                [&]() { return distribution(generator); });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return distribution(generator); });
        }
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        // -------------------------case 1----------------------------------
        impl::op_t qdata_node(0, impl::op_kind::Quantize, "qdata_node");
        SET_Q_DQ_DATA_ATTR(qdata_node)

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t qweight_node(2, impl::op_kind::Quantize, "qweight_node");
        SET_Q_DQ_WEIGHT_ATTR(qweight_node)
        // for XIO weight, the output channel is 3rd one
        qweight_node.set_attr<int64_t>(
                "axis", wei_qtype == "per_tensor" ? 0 : 3);

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)
        dqweight_node.set_attr<int64_t>(
                "axis", wei_qtype == "per_tensor" ? 0 : 3);

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)
        conv_node.set_attr<std::string>("data_format", "NXC");
        conv_node.set_attr<std::string>("filter_format", "XIO");

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        impl::op_t qout_node(6, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        impl::op_t qother_node(7, impl::op_kind::Quantize, "qother_node");
        qother_node.set_attr<std::string>("qtype", "per_tensor");
        qother_node.set_attr<std::string>("out_type", "int8");
        qother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        qother_node.set_attr<std::vector<float>>("scales", {scale_other});
        qother_node.set_attr<int64_t>("axis", 0);

        impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>("qtype", "per_tensor");
        dqother_node.set_attr<std::string>("in_type", "int8");
        dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_node.set_attr<int64_t>("axis", 0);

        impl::op_t add_node(9, impl::op_kind::Add, "add_node");

        // create kernels
        auto kernel_qdata
                = get_dnnl_kernel_registry().create_kernel(qdata_node);
        auto kernel_dqdata
                = get_dnnl_kernel_registry().create_kernel(dqdata_node);
        auto kernel_qweight
                = get_dnnl_kernel_registry().create_kernel(qweight_node);
        auto kernel_dqweight
                = get_dnnl_kernel_registry().create_kernel(dqweight_node);
        auto kernel_conv = get_dnnl_kernel_registry().create_kernel(conv_node);
        auto kernel_relu = get_dnnl_kernel_registry().create_kernel(relu_node);
        auto kernel_qout = get_dnnl_kernel_registry().create_kernel(qout_node);
        auto kernel_qother
                = get_dnnl_kernel_registry().create_kernel(qother_node);
        auto kernel_dqother
                = get_dnnl_kernel_registry().create_kernel(dqother_node);
        auto kernel_add = get_dnnl_kernel_registry().create_kernel(add_node);

        // prepare logical tensor
        impl::logical_tensor_t src_f32 = utils::logical_tensor_init(
                0, src_shape, impl::data_type::f32);
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t other_f32 = utils::logical_tensor_init(
                10, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t other_s8 = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t other_f32_dq = utils::logical_tensor_init(
                12, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                13, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }

        // compile
        kernel_qdata->compile(&qdata_node, &engine, {src_f32}, {src_u8});
        kernel_dqdata->compile(&dqdata_node, &engine, {src_u8}, {src_f32_dq});
        kernel_qweight->compile(
                &qweight_node, &engine, {weight_f32}, {weight_s8});
        kernel_dqweight->compile(
                &dqweight_node, &engine, {weight_s8}, {weight_f32_dq});
        if (with_bias)
            kernel_conv->compile(&conv_node, &engine,
                    {src_f32_dq, weight_f32_dq, bias_f32}, {dst_f32});
        else
            kernel_conv->compile(&conv_node, &engine,
                    {src_f32_dq, weight_f32_dq}, {dst_f32});

        kernel_qother->compile(&qother_node, &engine, {other_f32}, {other_s8});
        kernel_dqother->compile(
                &dqother_node, &engine, {other_s8}, {other_f32_dq});
        kernel_add->compile(
                &add_node, &engine, {dst_f32, other_f32_dq}, {dst_add_f32});
        kernel_relu->compile(
                &relu_node, &engine, {dst_add_f32}, {dst_relu_f32});
        kernel_qout->compile(&qout_node, &engine, {dst_relu_f32}, {dst_s8});

        // execute
        test::vector<uint8_t> src_u8_data(product(src_shape));
        impl::tensor_t src_f32_ts(src_f32, src_data.data());
        impl::tensor_t src_u8_ts(src_u8, src_u8_data.data());
        kernel_qdata->execute(&qdata_node, &strm, {src_f32_ts}, {src_u8_ts});
        strm.wait();

        test::vector<float> src_f32_dq_data(product(src_shape));
        impl::tensor_t src_f32_dq_ts(src_f32_dq, src_f32_dq_data.data());
        kernel_dqdata->execute(
                &dqdata_node, &strm, {src_u8_ts}, {src_f32_dq_ts});
        strm.wait();

        test::vector<int8_t> weight_s8_data(product(weight_shape));
        impl::tensor_t weight_f32_ts(weight_f32, weight_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, weight_s8_data.data());
        kernel_qweight->execute(
                &qweight_node, &strm, {weight_f32_ts}, {weight_s8_ts});
        strm.wait();

        test::vector<float> weight_f32_dq_data(product(weight_shape));
        impl::tensor_t weight_f32_dq_ts(
                weight_f32_dq, weight_f32_dq_data.data());
        kernel_dqweight->execute(
                &dqweight_node, &strm, {weight_s8_ts}, {weight_f32_dq_ts});
        strm.wait();

        test::vector<float> out_f32_data(product(dst_shape));
        impl::tensor_t dst_f32_ts(dst_f32, out_f32_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, bias_data.data());
            kernel_conv->execute(&conv_node, &strm,
                    {src_f32_dq_ts, weight_f32_dq_ts, bias_f32_ts},
                    {dst_f32_ts});
        } else {
            kernel_conv->execute(&conv_node, &strm,
                    {src_f32_dq_ts, weight_f32_dq_ts}, {dst_f32_ts});
        }
        strm.wait();

        test::vector<int8_t> other_s8_data(product(dst_shape));
        impl::tensor_t other_f32_ts(other_f32, other_data.data());
        impl::tensor_t other_s8_ts(other_s8, other_s8_data.data());
        kernel_qother->execute(
                &qother_node, &strm, {other_f32_ts}, {other_s8_ts});
        strm.wait();

        test::vector<float> other_f32_dq_data(product(dst_shape));
        impl::tensor_t other_f32_dq_ts(other_f32_dq, other_f32_dq_data.data());
        kernel_dqother->execute(
                &dqother_node, &strm, {other_s8_ts}, {other_f32_dq_ts});
        strm.wait();

        test::vector<float> out_add_f32_data(product(dst_shape));
        impl::tensor_t dst_add_f32_ts(dst_add_f32, out_add_f32_data.data());
        kernel_add->execute(&add_node, &strm, {dst_f32_ts, other_f32_dq_ts},
                {dst_add_f32_ts});
        strm.wait();

        test::vector<float> out_relu_f32_data(product(dst_shape));
        impl::tensor_t dst_relu_f32_ts(dst_relu_f32, out_relu_f32_data.data());
        kernel_relu->execute(
                &relu_node, &strm, {dst_add_f32_ts}, {dst_relu_f32_ts});
        strm.wait();

        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, case1_out_data.data());
        kernel_qout->execute(&qout_node, &strm, {dst_relu_f32_ts}, {dst_s8_ts});
        strm.wait();

        // -------------------------case 2----------------------------------
        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.build_graph();

        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("int8_conv_bias_add_relu_fusion")
                : get_pass("int8_conv_add_relu_fusion");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_s8, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_s8_case2_ts(dst_s8, case2_out_data.data());
        if (with_bias)
            cp.execute(&strm,
                    {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        strm.wait();

        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_matmul_ndx2d) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {1, 2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<float> src_data(product(src_shape));
        test::vector<float> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(),
                [&]() { return distribution(generator); });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return distribution(generator); });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_out = 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        // -------------------------case 1----------------------------------
        impl::op_t qdata_op(0, impl::op_kind::Quantize, "qdata_op");
        qdata_op.set_attr<std::string>("qtype", "per_tensor");
        qdata_op.set_attr<std::string>("out_type", "uint8");
        qdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        qdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        qdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::string>("in_type", "uint8");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t qweight_op(2, impl::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>("qtype", qtype);
        qweight_op.set_attr<std::string>("out_type", "int8");
        qweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        qweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        qweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t dqweight_op(3, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::string>("in_type", "int8");
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", false);

        impl::op_t qout_op(5, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::string>("out_type", "int8");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        // create kernels
        auto kernel_qdata = get_dnnl_kernel_registry().create_kernel(qdata_op);
        auto kernel_dqdata
                = get_dnnl_kernel_registry().create_kernel(dqdata_op);
        auto kernel_qweight
                = get_dnnl_kernel_registry().create_kernel(qweight_op);
        auto kernel_dqweight
                = get_dnnl_kernel_registry().create_kernel(dqweight_op);
        auto kernel_matmul
                = get_dnnl_kernel_registry().create_kernel(matmul_op);
        auto kernel_qout = get_dnnl_kernel_registry().create_kernel(qout_op);

        // prepare logical tensor
        impl::logical_tensor_t src_f32 = utils::logical_tensor_init(
                0, src_shape, impl::data_type::f32);
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(8, dst_shape, impl::data_type::s8);

        // compile
        kernel_qdata->compile(&qdata_op, &engine, {src_f32}, {src_u8});
        kernel_dqdata->compile(&dqdata_op, &engine, {src_u8}, {src_f32_dq});
        kernel_qweight->compile(
                &qweight_op, &engine, {weight_f32}, {weight_s8});
        kernel_dqweight->compile(
                &dqweight_op, &engine, {weight_s8}, {weight_f32_dq});
        kernel_matmul->compile(&matmul_op, &engine,
                {src_f32_dq, weight_f32_dq, bias_f32}, {dst_f32});
        kernel_qout->compile(&qout_op, &engine, {dst_f32}, {dst_s8});

        // execute
        test::vector<uint8_t> src_u8_data(product(src_shape));
        impl::tensor_t src_f32_ts(src_f32, src_data.data());
        impl::tensor_t src_u8_ts(src_u8, src_u8_data.data());
        kernel_qdata->execute(&qdata_op, &strm, {src_f32_ts}, {src_u8_ts});
        strm.wait();

        test::vector<float> src_f32_dq_data(product(src_shape));
        impl::tensor_t src_f32_dq_ts(src_f32_dq, src_f32_dq_data.data());
        kernel_dqdata->execute(&dqdata_op, &strm, {src_u8_ts}, {src_f32_dq_ts});
        strm.wait();

        test::vector<int8_t> weight_s8_data(product(weight_shape));
        impl::tensor_t weight_f32_ts(weight_f32, weight_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, weight_s8_data.data());
        kernel_qweight->execute(
                &qweight_op, &strm, {weight_f32_ts}, {weight_s8_ts});
        strm.wait();

        test::vector<float> weight_f32_dq_data(product(weight_shape));
        impl::tensor_t weight_f32_dq_ts(
                weight_f32_dq, weight_f32_dq_data.data());
        kernel_dqweight->execute(
                &dqweight_op, &strm, {weight_s8_ts}, {weight_f32_dq_ts});
        strm.wait();

        test::vector<float> out_f32_data(product(dst_shape));
        impl::tensor_t bias_f32_ts(bias_f32, bias_data.data());
        impl::tensor_t dst_f32_ts(dst_f32, out_f32_data.data());
        kernel_matmul->execute(&matmul_op, &strm,
                {src_f32_dq_ts, weight_f32_dq_ts, bias_f32_ts}, {dst_f32_ts});
        strm.wait();

        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, case1_out_data.data());
        kernel_qout->execute(&qout_op, &strm, {dst_f32_ts}, {dst_s8_ts});
        strm.wait();

        // -------------------------case 2----------------------------------
        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("int8_matmul_bias_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_s8_case2_ts(dst_s8, case2_out_data.data());
        cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                {dst_s8_case2_ts});
        strm.wait();

        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_matmul_ndx1d) {
    // compare results between: case 1: [quantize] - [dequantize] -
    // [fp32_matmul] - [quantize] case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::vector<int64_t>> src_shapes {{3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 1}, {4}};
    std::vector<std::vector<int64_t>> dst_shapes {{3, 8, 1}, {8, 1}, {1, 1}};

    for (size_t i = 0; i < src_shapes.size(); ++i) {
        for (size_t j = 0; j < weight_shapes.size(); ++j) {
            // prepare fp32 data
            std::vector<int64_t> src_shape = src_shapes[i];
            std::vector<int64_t> weight_shape = weight_shapes[j];
            std::vector<int64_t> dst_shape = dst_shapes[i];

            test::vector<float> src_data(product(src_shape));
            test::vector<float> weight_data(product(weight_shape));

            // random generate src, weight and bias data random seed = 7
            std::default_random_engine generator(7);
            std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return distribution(generator); });
            float scale_src = 1 / 255.f; // map to 0~255
            float scale_wei = 1 / 127.f;
            float scale_out = 1;
            int64_t zp_src = 0;
            int64_t zp_wei = 0;
            int64_t zp_out = 78;

            // -------------------------case 1----------------------------------
            impl::op_t qdata_op(0, impl::op_kind::Quantize, "qdata_op");
            qdata_op.set_attr<std::string>("qtype", "per_tensor");
            qdata_op.set_attr<std::string>("out_type", "uint8");
            qdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
            qdata_op.set_attr<std::vector<float>>("scales", {scale_src});
            qdata_op.set_attr<int64_t>("axis", 0);

            impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
            dqdata_op.set_attr<std::string>("qtype", "per_tensor");
            dqdata_op.set_attr<std::string>("in_type", "uint8");
            dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
            dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
            dqdata_op.set_attr<int64_t>("axis", 0);

            impl::op_t qweight_op(2, impl::op_kind::Quantize, "qweight_op");
            qweight_op.set_attr<std::string>("qtype", "per_tensor");
            qweight_op.set_attr<std::string>("out_type", "int8");
            qweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
            qweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
            qweight_op.set_attr<int64_t>("axis", 0);

            impl::op_t dqweight_op(3, impl::op_kind::Dequantize, "dqweight_op");
            dqweight_op.set_attr<std::string>("qtype", "per_tensor");
            dqweight_op.set_attr<std::string>("in_type", "int8");
            dqweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
            dqweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
            dqweight_op.set_attr<int64_t>("axis", 0);

            impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
            matmul_op.set_attr<bool>("transpose_a", false);
            matmul_op.set_attr<bool>("transpose_b", false);

            impl::op_t qout_op(5, impl::op_kind::Quantize, "qout_op");
            qout_op.set_attr<std::string>("qtype", "per_tensor");
            qout_op.set_attr<std::string>("out_type", "int8");
            qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
            qout_op.set_attr<std::vector<float>>("scales", {scale_out});
            qout_op.set_attr<int64_t>("axis", 0);

            // create kernels
            auto kernel_qdata
                    = get_dnnl_kernel_registry().create_kernel(qdata_op);
            auto kernel_dqdata
                    = get_dnnl_kernel_registry().create_kernel(dqdata_op);
            auto kernel_qweight
                    = get_dnnl_kernel_registry().create_kernel(qweight_op);
            auto kernel_dqweight
                    = get_dnnl_kernel_registry().create_kernel(dqweight_op);
            auto kernel_matmul
                    = get_dnnl_kernel_registry().create_kernel(matmul_op);
            auto kernel_qout
                    = get_dnnl_kernel_registry().create_kernel(qout_op);

            // prepare logical tensor
            impl::logical_tensor_t src_f32 = utils::logical_tensor_init(
                    0, src_shape, impl::data_type::f32);
            impl::logical_tensor_t src_u8 = utils::logical_tensor_init(
                    1, src_shape, impl::data_type::u8);
            impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    2, src_shape, impl::data_type::f32);
            impl::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                    3, weight_shape, impl::data_type::f32);
            impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                    4, weight_shape, impl::data_type::s8);
            impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    5, weight_shape, impl::data_type::f32);
            impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                    7, dst_shape, impl::data_type::f32);
            impl::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                    8, dst_shape, impl::data_type::s8);

            // compile
            kernel_qdata->compile(&qdata_op, &engine, {src_f32}, {src_u8});
            kernel_dqdata->compile(&dqdata_op, &engine, {src_u8}, {src_f32_dq});
            kernel_qweight->compile(
                    &qweight_op, &engine, {weight_f32}, {weight_s8});
            kernel_dqweight->compile(
                    &dqweight_op, &engine, {weight_s8}, {weight_f32_dq});
            kernel_matmul->compile(&matmul_op, &engine,
                    {src_f32_dq, weight_f32_dq}, {dst_f32});
            kernel_qout->compile(&qout_op, &engine, {dst_f32}, {dst_s8});

            // execute
            test::vector<uint8_t> src_u8_data(product(src_shape));
            impl::tensor_t src_f32_ts(src_f32, src_data.data());
            impl::tensor_t src_u8_ts(src_u8, src_u8_data.data());
            kernel_qdata->execute(&qdata_op, &strm, {src_f32_ts}, {src_u8_ts});
            strm.wait();

            test::vector<float> src_f32_dq_data(product(src_shape));
            impl::tensor_t src_f32_dq_ts(src_f32_dq, src_f32_dq_data.data());
            kernel_dqdata->execute(
                    &dqdata_op, &strm, {src_u8_ts}, {src_f32_dq_ts});
            strm.wait();

            test::vector<int8_t> weight_s8_data(product(weight_shape));
            impl::tensor_t weight_f32_ts(weight_f32, weight_data.data());
            impl::tensor_t weight_s8_ts(weight_s8, weight_s8_data.data());
            kernel_qweight->execute(
                    &qweight_op, &strm, {weight_f32_ts}, {weight_s8_ts});
            strm.wait();

            test::vector<float> weight_f32_dq_data(product(weight_shape));
            impl::tensor_t weight_f32_dq_ts(
                    weight_f32_dq, weight_f32_dq_data.data());
            kernel_dqweight->execute(
                    &dqweight_op, &strm, {weight_s8_ts}, {weight_f32_dq_ts});
            strm.wait();

            test::vector<float> out_f32_data(product(dst_shape));
            impl::tensor_t dst_f32_ts(dst_f32, out_f32_data.data());
            kernel_matmul->execute(&matmul_op, &strm,
                    {src_f32_dq_ts, weight_f32_dq_ts}, {dst_f32_ts});
            strm.wait();

            test::vector<int8_t> case1_out_data(product(dst_shape));
            impl::tensor_t dst_s8_ts(dst_s8, case1_out_data.data());
            kernel_qout->execute(&qout_op, &strm, {dst_f32_ts}, {dst_s8_ts});
            strm.wait();

            // -------------------------case 2----------------------------------
            dqdata_op.add_input(src_u8);
            dqdata_op.add_output(src_f32_dq);

            dqweight_op.add_input(weight_s8);
            dqweight_op.add_output(weight_f32_dq);

            matmul_op.add_input(src_f32_dq);
            matmul_op.add_input(weight_f32_dq);
            //     conv_node.add_input(bias_f32);
            matmul_op.add_output(dst_f32);

            qout_op.add_input(dst_f32);
            qout_op.add_output(dst_s8);

            impl::graph_t g;
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&qout_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass = get_pass("int8_matmul_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {
                    &src_u8, &weight_s8};
            std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<int8_t> case2_out_data(product(dst_shape));
            impl::tensor_t dst_s8_case2_ts(dst_s8, case2_out_data.data());
            cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
            strm.wait();

            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        }
    }
}

TEST(int8_subgraph_mode, int8_matmul_ndx2d_with_transpose) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{2, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {1, 2}};

    for (size_t i = 0; i < src_shapes.size(); ++i) {
        for (size_t j = 0; j < weight_shapes.size(); ++j) {
            // prepare fp32 data
            std::vector<int64_t> src_shape = src_shapes[i];
            std::vector<int64_t> weight_shape = weight_shapes[j];
            std::vector<int64_t> bias_shape {1};
            std::vector<int64_t> dst_shape = dst_shapes[i];

            test::vector<float> src_data(product(src_shape));
            test::vector<float> weight_data(product(weight_shape));
            test::vector<float> bias_data(product(bias_shape));

            // random generate src, weight and bias data random seed = 7
            std::default_random_engine generator(7);
            std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return distribution(generator); });
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return distribution(generator); });
            float scale_src = 1 / 255.f; // map to 0~255
            float scale_wei = 1 / 127.f;
            float scale_out = 1;
            int64_t zp_src = 0;
            int64_t zp_wei = 0;
            int64_t zp_out = 78;

            // -------------------------case 1----------------------------------
            impl::op_t qdata_op(0, impl::op_kind::Quantize, "qdata_op");
            qdata_op.set_attr<std::string>("qtype", "per_tensor");
            qdata_op.set_attr<std::string>("out_type", "uint8");
            qdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
            qdata_op.set_attr<std::vector<float>>("scales", {scale_src});
            qdata_op.set_attr<int64_t>("axis", 0);

            impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
            dqdata_op.set_attr<std::string>("qtype", "per_tensor");
            dqdata_op.set_attr<std::string>("in_type", "uint8");
            dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
            dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
            dqdata_op.set_attr<int64_t>("axis", 0);

            impl::op_t qweight_op(2, impl::op_kind::Quantize, "qweight_op");
            qweight_op.set_attr<std::string>("qtype", "per_tensor");
            qweight_op.set_attr<std::string>("out_type", "int8");
            qweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
            qweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
            qweight_op.set_attr<int64_t>("axis", 0);

            impl::op_t dqweight_op(3, impl::op_kind::Dequantize, "dqweight_op");
            dqweight_op.set_attr<std::string>("qtype", "per_tensor");
            dqweight_op.set_attr<std::string>("in_type", "int8");
            dqweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
            dqweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
            dqweight_op.set_attr<int64_t>("axis", 0);

            impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
            matmul_op.set_attr<bool>("transpose_a", false);
            matmul_op.set_attr<bool>("transpose_b", true);

            impl::op_t qout_op(5, impl::op_kind::Quantize, "qout_op");
            qout_op.set_attr<std::string>("qtype", "per_tensor");
            qout_op.set_attr<std::string>("out_type", "int8");
            qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
            qout_op.set_attr<std::vector<float>>("scales", {scale_out});
            qout_op.set_attr<int64_t>("axis", 0);

            // create kernels
            auto kernel_qdata
                    = get_dnnl_kernel_registry().create_kernel(qdata_op);
            auto kernel_dqdata
                    = get_dnnl_kernel_registry().create_kernel(dqdata_op);
            auto kernel_qweight
                    = get_dnnl_kernel_registry().create_kernel(qweight_op);
            auto kernel_dqweight
                    = get_dnnl_kernel_registry().create_kernel(dqweight_op);
            auto kernel_matmul
                    = get_dnnl_kernel_registry().create_kernel(matmul_op);
            auto kernel_qout
                    = get_dnnl_kernel_registry().create_kernel(qout_op);

            // prepare logical tensor
            impl::logical_tensor_t src_f32 = utils::logical_tensor_init(
                    0, src_shape, impl::data_type::f32);
            impl::logical_tensor_t src_u8 = utils::logical_tensor_init(
                    1, src_shape, impl::data_type::u8);
            impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    2, src_shape, impl::data_type::f32);
            impl::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                    3, weight_shape, impl::data_type::f32);
            impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                    4, weight_shape, impl::data_type::s8);
            impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    5, weight_shape, impl::data_type::f32);
            impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
            impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                    7, dst_shape, impl::data_type::f32);
            impl::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                    8, dst_shape, impl::data_type::s8);

            // compile
            kernel_qdata->compile(&qdata_op, &engine, {src_f32}, {src_u8});
            kernel_dqdata->compile(&dqdata_op, &engine, {src_u8}, {src_f32_dq});
            kernel_qweight->compile(
                    &qweight_op, &engine, {weight_f32}, {weight_s8});
            kernel_dqweight->compile(
                    &dqweight_op, &engine, {weight_s8}, {weight_f32_dq});
            kernel_matmul->compile(&matmul_op, &engine,
                    {src_f32_dq, weight_f32_dq, bias_f32}, {dst_f32});
            kernel_qout->compile(&qout_op, &engine, {dst_f32}, {dst_s8});

            // execute
            test::vector<uint8_t> src_u8_data(product(src_shape));
            impl::tensor_t src_f32_ts(src_f32, src_data.data());
            impl::tensor_t src_u8_ts(src_u8, src_u8_data.data());
            kernel_qdata->execute(&qdata_op, &strm, {src_f32_ts}, {src_u8_ts});
            strm.wait();

            test::vector<float> src_f32_dq_data(product(src_shape));
            impl::tensor_t src_f32_dq_ts(src_f32_dq, src_f32_dq_data.data());
            kernel_dqdata->execute(
                    &dqdata_op, &strm, {src_u8_ts}, {src_f32_dq_ts});
            strm.wait();

            test::vector<int8_t> weight_s8_data(product(weight_shape));
            impl::tensor_t weight_f32_ts(weight_f32, weight_data.data());
            impl::tensor_t weight_s8_ts(weight_s8, weight_s8_data.data());
            kernel_qweight->execute(
                    &qweight_op, &strm, {weight_f32_ts}, {weight_s8_ts});
            strm.wait();

            test::vector<float> weight_f32_dq_data(product(weight_shape));
            impl::tensor_t weight_f32_dq_ts(
                    weight_f32_dq, weight_f32_dq_data.data());
            kernel_dqweight->execute(
                    &dqweight_op, &strm, {weight_s8_ts}, {weight_f32_dq_ts});
            strm.wait();

            test::vector<float> out_f32_data(product(dst_shape));
            impl::tensor_t bias_f32_ts(bias_f32, bias_data.data());
            impl::tensor_t dst_f32_ts(dst_f32, out_f32_data.data());
            kernel_matmul->execute(&matmul_op, &strm,
                    {src_f32_dq_ts, weight_f32_dq_ts, bias_f32_ts},
                    {dst_f32_ts});
            strm.wait();

            test::vector<int8_t> case1_out_data(product(dst_shape));
            impl::tensor_t dst_s8_ts(dst_s8, case1_out_data.data());
            kernel_qout->execute(&qout_op, &strm, {dst_f32_ts}, {dst_s8_ts});
            strm.wait();

            // -------------------------case 2----------------------------------
            dqdata_op.add_input(src_u8);
            dqdata_op.add_output(src_f32_dq);

            dqweight_op.add_input(weight_s8);
            dqweight_op.add_output(weight_f32_dq);

            matmul_op.add_input(src_f32_dq);
            matmul_op.add_input(weight_f32_dq);
            matmul_op.add_input(bias_f32);
            matmul_op.add_output(dst_f32);

            qout_op.add_input(dst_f32);
            qout_op.add_output(dst_s8);

            impl::graph_t g;
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&qout_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass
                    = get_pass("int8_matmul_bias_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {
                    &src_u8, &weight_s8, &bias_f32};
            std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<int8_t> case2_out_data(product(dst_shape));
            impl::tensor_t dst_s8_case2_ts(dst_s8, case2_out_data.data());
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
            strm.wait();

            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        }
    }
}

TEST(int8_subgraph_mode, int8_matmul_relu_fusion) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [relu] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    // prepare fp32 data
    std::vector<int64_t> src_shape {8, 6};
    std::vector<int64_t> weight_shape {6, 4};
    std::vector<int64_t> dst_shape {8, 4};

    test::vector<float> src_data(product(src_shape));
    test::vector<float> weight_data(product(weight_shape));

    // random generate src, weight and bias data random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    float scale_wei = 1 / 127.f;
    float scale_out = 1;
    int64_t zp_src = 0;
    int64_t zp_wei = 0;
    int64_t zp_out = 78;

    // -------------------------case 1----------------------------------
    impl::op_t qdata_op(0, impl::op_kind::Quantize, "qdata_op");
    qdata_op.set_attr<std::string>("qtype", "per_tensor");
    qdata_op.set_attr<std::string>("out_type", "uint8");
    qdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
    qdata_op.set_attr<std::vector<float>>("scales", {scale_src});
    qdata_op.set_attr<int64_t>("axis", 0);

    impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>("qtype", "per_tensor");
    dqdata_op.set_attr<std::string>("in_type", "uint8");
    dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
    dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
    dqdata_op.set_attr<int64_t>("axis", 0);

    impl::op_t qweight_op(2, impl::op_kind::Quantize, "qweight_op");
    qweight_op.set_attr<std::string>("qtype", "per_tensor");
    qweight_op.set_attr<std::string>("out_type", "int8");
    qweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
    qweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
    qweight_op.set_attr<int64_t>("axis", 0);

    impl::op_t dqweight_op(3, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>("qtype", "per_tensor");
    dqweight_op.set_attr<std::string>("in_type", "int8");
    dqweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
    dqweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
    dqweight_op.set_attr<int64_t>("axis", 0);

    impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>("transpose_a", false);
    matmul_op.set_attr<bool>("transpose_b", false);

    impl::op_t relu_op(5, impl::op_kind::ReLU, "relu_op");

    impl::op_t qout_op(6, impl::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>("qtype", "per_tensor");
    qout_op.set_attr<std::string>("out_type", "int8");
    qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
    qout_op.set_attr<std::vector<float>>("scales", {scale_out});
    qout_op.set_attr<int64_t>("axis", 0);

    // create kernels
    auto kernel_qdata = get_dnnl_kernel_registry().create_kernel(qdata_op);
    auto kernel_dqdata = get_dnnl_kernel_registry().create_kernel(dqdata_op);
    auto kernel_qweight = get_dnnl_kernel_registry().create_kernel(qweight_op);
    auto kernel_dqweight
            = get_dnnl_kernel_registry().create_kernel(dqweight_op);
    auto kernel_matmul = get_dnnl_kernel_registry().create_kernel(matmul_op);
    auto kernel_relu = get_dnnl_kernel_registry().create_kernel(relu_op);
    auto kernel_qout = get_dnnl_kernel_registry().create_kernel(qout_op);

    // prepare logical tensor
    impl::logical_tensor_t src_f32
            = utils::logical_tensor_init(0, src_shape, impl::data_type::f32);
    impl::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_f32
            = utils::logical_tensor_init(3, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_s8
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::s8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(5, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_relu_f32
            = utils::logical_tensor_init(8, dst_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_s8
            = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);

    // compile
    kernel_qdata->compile(&qdata_op, &engine, {src_f32}, {src_u8});
    kernel_dqdata->compile(&dqdata_op, &engine, {src_u8}, {src_f32_dq});
    kernel_qweight->compile(&qweight_op, &engine, {weight_f32}, {weight_s8});
    kernel_dqweight->compile(
            &dqweight_op, &engine, {weight_s8}, {weight_f32_dq});
    kernel_matmul->compile(
            &matmul_op, &engine, {src_f32_dq, weight_f32_dq}, {dst_f32});
    kernel_relu->compile(&relu_op, &engine, {dst_f32}, {dst_relu_f32});
    kernel_qout->compile(&qout_op, &engine, {dst_f32}, {dst_s8});

    // execute
    test::vector<uint8_t> src_u8_data(product(src_shape));
    impl::tensor_t src_f32_ts(src_f32, src_data.data());
    impl::tensor_t src_u8_ts(src_u8, src_u8_data.data());
    kernel_qdata->execute(&qdata_op, &strm, {src_f32_ts}, {src_u8_ts});
    strm.wait();

    test::vector<float> src_f32_dq_data(product(src_shape));
    impl::tensor_t src_f32_dq_ts(src_f32_dq, src_f32_dq_data.data());
    kernel_dqdata->execute(&dqdata_op, &strm, {src_u8_ts}, {src_f32_dq_ts});
    strm.wait();

    test::vector<int8_t> weight_s8_data(product(weight_shape));
    impl::tensor_t weight_f32_ts(weight_f32, weight_data.data());
    impl::tensor_t weight_s8_ts(weight_s8, weight_s8_data.data());
    kernel_qweight->execute(
            &qweight_op, &strm, {weight_f32_ts}, {weight_s8_ts});
    strm.wait();

    test::vector<float> weight_f32_dq_data(product(weight_shape));
    impl::tensor_t weight_f32_dq_ts(weight_f32_dq, weight_f32_dq_data.data());
    kernel_dqweight->execute(
            &dqweight_op, &strm, {weight_s8_ts}, {weight_f32_dq_ts});
    strm.wait();

    test::vector<float> out_f32_data(product(dst_shape));
    impl::tensor_t dst_f32_ts(dst_f32, out_f32_data.data());
    kernel_matmul->execute(
            &matmul_op, &strm, {src_f32_dq_ts, weight_f32_dq_ts}, {dst_f32_ts});
    strm.wait();

    test::vector<float> out_relu_f32_data(product(dst_shape));
    impl::tensor_t dst_relu_f32_ts(dst_relu_f32, out_relu_f32_data.data());
    kernel_relu->execute(&relu_op, &strm, {dst_f32_ts}, {dst_relu_f32_ts});
    strm.wait();

    test::vector<int8_t> case1_out_data(product(dst_shape));
    impl::tensor_t dst_s8_ts(dst_s8, case1_out_data.data());
    kernel_qout->execute(&qout_op, &strm, {dst_relu_f32_ts}, {dst_s8_ts});
    strm.wait();

    // -------------------------case 2----------------------------------
    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    //     conv_node.add_input(bias_f32);
    matmul_op.add_output(dst_f32);

    relu_op.add_input(dst_f32);
    relu_op.add_output(dst_relu_f32);

    qout_op.add_input(dst_relu_f32);
    qout_op.add_output(dst_s8);

    impl::graph_t g;
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&relu_op);
    g.add_op(&qout_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("int8_matmul_relu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {&src_u8, &weight_s8};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<int8_t> case2_out_data(product(dst_shape));
    impl::tensor_t dst_s8_case2_ts(dst_s8, case2_out_data.data());
    cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
    strm.wait();

    ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
            /*atol*/ 1.f));
}

TEST(int8_subgraph_mode, int8_matmul_bias_sum_ndx2d) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {1, 2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<float> src_data(product(src_shape));
        test::vector<float> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));
        test::vector<float> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(),
                [&]() { return distribution(generator); });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return distribution(generator); });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return distribution(generator); });
        std::generate(other_data.begin(), other_data.end(),
                [&]() { return distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        // -------------------------case 1----------------------------------
        impl::op_t qdata_op(0, impl::op_kind::Quantize, "qdata_op");
        qdata_op.set_attr<std::string>("qtype", "per_tensor");
        qdata_op.set_attr<std::string>("out_type", "uint8");
        qdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        qdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        qdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::string>("in_type", "uint8");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t qweight_op(2, impl::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>("qtype", qtype);
        qweight_op.set_attr<std::string>("out_type", "int8");
        qweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        qweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        qweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t dqweight_op(3, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::string>("in_type", "int8");
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", false);

        impl::op_t qout_op(5, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::string>("out_type", "int8");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        impl::op_t qother_op(6, impl::op_kind::Quantize, "qother_op");
        qother_op.set_attr<std::string>("qtype", "per_tensor");
        qother_op.set_attr<std::string>("out_type", "int8");
        qother_op.set_attr<std::vector<int64_t>>("zps", {zp_other});
        qother_op.set_attr<std::vector<float>>("scales", {scale_other});
        qother_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqother_op(7, impl::op_kind::Dequantize, "dqother_op");
        dqother_op.set_attr<std::string>("qtype", "per_tensor");
        dqother_op.set_attr<std::string>("in_type", "int8");
        dqother_op.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_op.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_op.set_attr<int64_t>("axis", 0);

        impl::op_t add_op(8, impl::op_kind::Add, "add_op");

        // create kernels
        auto kernel_qdata = get_dnnl_kernel_registry().create_kernel(qdata_op);
        auto kernel_dqdata
                = get_dnnl_kernel_registry().create_kernel(dqdata_op);
        auto kernel_qweight
                = get_dnnl_kernel_registry().create_kernel(qweight_op);
        auto kernel_dqweight
                = get_dnnl_kernel_registry().create_kernel(dqweight_op);
        auto kernel_matmul
                = get_dnnl_kernel_registry().create_kernel(matmul_op);
        auto kernel_qout = get_dnnl_kernel_registry().create_kernel(qout_op);
        auto kernel_qother
                = get_dnnl_kernel_registry().create_kernel(qother_op);
        auto kernel_dqother
                = get_dnnl_kernel_registry().create_kernel(dqother_op);
        auto kernel_add = get_dnnl_kernel_registry().create_kernel(add_op);

        // prepare logical tensor
        impl::logical_tensor_t src_f32 = utils::logical_tensor_init(
                0, src_shape, impl::data_type::f32);
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(8, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t other_f32 = utils::logical_tensor_init(
                9, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t other_s8 = utils::logical_tensor_init(
                10, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t other_f32_dq = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                12, dst_shape, impl::data_type::f32);

        // compile
        kernel_qdata->compile(&qdata_op, &engine, {src_f32}, {src_u8});
        kernel_dqdata->compile(&dqdata_op, &engine, {src_u8}, {src_f32_dq});
        kernel_qweight->compile(
                &qweight_op, &engine, {weight_f32}, {weight_s8});
        kernel_dqweight->compile(
                &dqweight_op, &engine, {weight_s8}, {weight_f32_dq});
        kernel_matmul->compile(&matmul_op, &engine,
                {src_f32_dq, weight_f32_dq, bias_f32}, {dst_f32});
        kernel_qother->compile(&qother_op, &engine, {other_f32}, {other_s8});
        kernel_dqother->compile(
                &dqother_op, &engine, {other_s8}, {other_f32_dq});
        kernel_add->compile(
                &add_op, &engine, {dst_f32, other_f32_dq}, {dst_add_f32});
        kernel_qout->compile(&qout_op, &engine, {dst_add_f32}, {dst_s8});

        // execute
        test::vector<uint8_t> src_u8_data(product(src_shape));
        impl::tensor_t src_f32_ts(src_f32, src_data.data());
        impl::tensor_t src_u8_ts(src_u8, src_u8_data.data());
        kernel_qdata->execute(&qdata_op, &strm, {src_f32_ts}, {src_u8_ts});
        strm.wait();

        test::vector<float> src_f32_dq_data(product(src_shape));
        impl::tensor_t src_f32_dq_ts(src_f32_dq, src_f32_dq_data.data());
        kernel_dqdata->execute(&dqdata_op, &strm, {src_u8_ts}, {src_f32_dq_ts});
        strm.wait();

        test::vector<int8_t> weight_s8_data(product(weight_shape));
        impl::tensor_t weight_f32_ts(weight_f32, weight_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, weight_s8_data.data());
        kernel_qweight->execute(
                &qweight_op, &strm, {weight_f32_ts}, {weight_s8_ts});
        strm.wait();

        test::vector<float> weight_f32_dq_data(product(weight_shape));
        impl::tensor_t weight_f32_dq_ts(
                weight_f32_dq, weight_f32_dq_data.data());
        kernel_dqweight->execute(
                &dqweight_op, &strm, {weight_s8_ts}, {weight_f32_dq_ts});
        strm.wait();

        test::vector<float> out_f32_data(product(dst_shape));
        impl::tensor_t bias_f32_ts(bias_f32, bias_data.data());
        impl::tensor_t dst_f32_ts(dst_f32, out_f32_data.data());
        kernel_matmul->execute(&matmul_op, &strm,
                {src_f32_dq_ts, weight_f32_dq_ts, bias_f32_ts}, {dst_f32_ts});
        strm.wait();

        test::vector<int8_t> other_s8_data(product(dst_shape));
        impl::tensor_t other_f32_ts(other_f32, other_data.data());
        impl::tensor_t other_s8_ts(other_s8, other_s8_data.data());
        kernel_qother->execute(
                &qother_op, &strm, {other_f32_ts}, {other_s8_ts});
        strm.wait();

        test::vector<float> other_f32_dq_data(product(dst_shape));
        impl::tensor_t other_f32_dq_ts(other_f32_dq, other_f32_dq_data.data());
        kernel_dqother->execute(
                &dqother_op, &strm, {other_s8_ts}, {other_f32_dq_ts});
        strm.wait();

        test::vector<float> out_add_f32_data(product(dst_shape));
        impl::tensor_t dst_add_f32_ts(dst_add_f32, out_add_f32_data.data());
        kernel_add->execute(&add_op, &strm, {dst_f32_ts, other_f32_dq_ts},
                {dst_add_f32_ts});
        strm.wait();

        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, case1_out_data.data());
        kernel_qout->execute(&qout_op, &strm, {dst_add_f32_ts}, {dst_s8_ts});
        strm.wait();

        // -------------------------case 2----------------------------------
        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        dqother_op.add_input(other_s8);
        dqother_op.add_output(other_f32_dq);

        add_op.add_input(dst_f32);
        add_op.add_input(other_f32_dq);
        add_op.add_output(dst_add_f32);

        qout_op.add_input(dst_add_f32);
        qout_op.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&dqother_op);
        g.add_op(&add_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("int8_matmul_bias_add_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_s8_case2_ts(dst_s8, case2_out_data.data());
        cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                {dst_s8_case2_ts});
        strm.wait();

        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_conv2d_sum_relu_get_inplace_pair) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<int64_t> groups = {1, 4};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    for_(const auto &g : groups)
    for (const auto &wei_qtype : weight_qtypes) {
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        impl::op_t qout_node(6, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>("qtype", "per_tensor");
        dqother_node.set_attr<std::string>("in_type", "int8");
        dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_node.set_attr<int64_t>("axis", 0);

        impl::op_t add_node(9, impl::op_kind::Add, "add_node");

        impl::op_t dqdata_node2(10, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node2)

        impl::op_t dqweight_node2(
                11, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node2)

        impl::op_t conv_node2(12, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node2, 2)

        impl::op_t relu_node2(13, impl::op_kind::ReLU, "relu_node");

        impl::op_t qout_node2(14, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node2)

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t other_f32_dq = utils::logical_tensor_init(
                12, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                13, dst_shape, impl::data_type::f32);

        impl::logical_tensor_t src_u8_2 = utils::logical_tensor_init(
                14, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq_2 = utils::logical_tensor_init(
                15, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8_2 = utils::logical_tensor_init(
                16, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq_2 = utils::logical_tensor_init(
                17, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32_2 = utils::logical_tensor_init(
                18, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8_2 = utils::logical_tensor_init(
                19, dst_shape, impl::data_type::s8);

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(dst_s8_2);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        dqdata_node2.add_input(src_u8_2);
        dqdata_node2.add_output(src_f32_dq_2);

        dqweight_node2.add_input(weight_s8_2);
        dqweight_node2.add_output(weight_f32_dq_2);

        conv_node2.add_input(src_f32_dq_2);
        conv_node2.add_input(weight_f32_dq_2);
        conv_node2.add_output(dst_f32_2);

        qout_node2.add_input(dst_f32_2);
        qout_node2.add_output(dst_s8_2);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.add_op(&dqdata_node2);
        g.add_op(&dqweight_node2);
        g.add_op(&conv_node2);
        g.add_op(&qout_node2);
        g.build_graph();

        impl::pass::pass_base_ptr apass1 = get_pass("int8_conv_fusion");
        impl::pass::pass_base_ptr apass2
                = get_pass("int8_conv_add_relu_fusion");

        apass1->run(g);
        apass2->run(g);
        ASSERT_EQ(g.get_num_partitions(), 2);
        auto part1 = g.get_partitions()[0]; // int8_conv
        auto part2 = g.get_partitions()[1]; // int8_conv_sum_elu

        // compile
        impl::partition_t p1, p2;
        p1.init(part1);
        p2.init(part2);

        impl::compiled_partition_t cp1(p1);
        impl::compiled_partition_t cp2(p2);

        std::vector<const impl::logical_tensor_t *> lt_ins1;
        lt_ins1 = {&src_u8_2, &weight_s8_2};

        dst_s8_2.layout_type = impl::layout_type::any;
        std::vector<const impl::logical_tensor_t *> lt_outs1 {&dst_s8_2};

        p1.compile(&cp1, lt_ins1, lt_outs1, &engine);

        cp1.query_logical_tensor(dst_s8_2.id, &dst_s8_2);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        lt_ins = {&src_u8, &weight_s8, &dst_s8_2};

        dst_s8.layout_type = impl::layout_type::any;
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p2.compile(&cp2, lt_ins, lt_outs, &engine);

        std::vector<impl::inplace_pair_t> inplace_pairs
                = cp2.get_inplace_pairs();

        ASSERT_EQ(inplace_pairs.size(), 1);
        ASSERT_EQ(inplace_pairs[0].input, dst_s8_2.id);
        ASSERT_EQ(inplace_pairs[0].output, dst_s8.id);
    }
}

TEST(int8_subgraph_mode, int8_maxpool) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_maxpool] - [quantize]
    // case 2: [quantize] - [int8_maxpool]
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> data_formats {"NCX", "NXC"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 4, 4, 4}, {3, 3, 4, 4}, {3, 3, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 2, 2, 2}, {3, 3, 2, 2}, {3, 3, 2}};
    for_(const auto &data_format : data_formats)
    for (size_t i = 0; i < src_shapes.size(); ++i) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> dst_shape = dst_shapes[i];

        if (data_format == "NXC") {
            src_shape.emplace_back(src_shape[1]);
            src_shape.erase(src_shape.begin() + 1);
            dst_shape.emplace_back(dst_shape[1]);
            dst_shape.erase(dst_shape.begin() + 1);
        }

        test::vector<float> src_data(product(src_shape));
        test::vector<float> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(),
                [&]() { return distribution(generator); });
        float scale_src = 1 / 127.f;
        float scale_out = 1 / 127.f;
        int64_t zp_src = 0;
        int64_t zp_out = 0;

        // -------------------------case 1----------------------------------
        impl::op_t qdata_op(0, impl::op_kind::Quantize, "qdata_op");
        qdata_op.set_attr<std::string>("qtype", "per_tensor");
        qdata_op.set_attr<std::string>("out_type", "uint8");
        qdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        qdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        qdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::string>("in_type", "uint8");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t maxpool_op(2, impl::op_kind::MaxPool, "maxpool_op");
        size_t spatial_size = src_shape.size() - 2;
        maxpool_op.set_attr<dims>("strides", dims(spatial_size, 2));
        maxpool_op.set_attr<dims>("kernel", dims(spatial_size, 2));
        maxpool_op.set_attr<dims>("pads_begin", dims(spatial_size, 0));
        maxpool_op.set_attr<dims>("pads_end", dims(spatial_size, 0));
        maxpool_op.set_attr<std::string>("data_format", data_format);
        maxpool_op.set_attr<dims>("dilations", dims(spatial_size, 1));

        impl::op_t qout_op(3, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::string>("out_type", "uint8");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        // create kernels
        auto kernel_qdata = get_dnnl_kernel_registry().create_kernel(qdata_op);
        auto kernel_dqdata
                = get_dnnl_kernel_registry().create_kernel(dqdata_op);
        auto kernel_maxpool
                = get_dnnl_kernel_registry().create_kernel(maxpool_op);
        auto kernel_qout = get_dnnl_kernel_registry().create_kernel(qout_op);

        // prepare logical tensor
        impl::logical_tensor_t src_f32 = utils::logical_tensor_init(
                0, src_shape, impl::data_type::f32);
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                3, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_u8
                = utils::logical_tensor_init(4, dst_shape, impl::data_type::u8);

        // compile
        kernel_qdata->compile(&qdata_op, &engine, {src_f32}, {src_u8});
        kernel_dqdata->compile(&dqdata_op, &engine, {src_u8}, {src_f32_dq});
        kernel_maxpool->compile(&maxpool_op, &engine, {src_f32_dq}, {dst_f32});
        kernel_qout->compile(&qout_op, &engine, {dst_f32}, {dst_u8});

        // execute
        test::vector<uint8_t> src_u8_data(product(src_shape));
        impl::tensor_t src_f32_ts(src_f32, src_data.data());
        impl::tensor_t src_u8_ts(src_u8, src_u8_data.data());
        kernel_qdata->execute(&qdata_op, &strm, {src_f32_ts}, {src_u8_ts});
        strm.wait();

        test::vector<float> src_f32_dq_data(product(src_shape));
        impl::tensor_t src_f32_dq_ts(src_f32_dq, src_f32_dq_data.data());
        kernel_dqdata->execute(&dqdata_op, &strm, {src_u8_ts}, {src_f32_dq_ts});
        strm.wait();

        test::vector<float> out_f32_data(product(dst_shape));
        impl::tensor_t dst_f32_ts(dst_f32, out_f32_data.data());
        kernel_maxpool->execute(
                &maxpool_op, &strm, {src_f32_dq_ts}, {dst_f32_ts});
        strm.wait();

        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_u8_ts(dst_u8, case1_out_data.data());
        kernel_qout->execute(&qout_op, &strm, {dst_f32_ts}, {dst_u8_ts});
        strm.wait();

        // -------------------------case 2----------------------------------
        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        maxpool_op.add_input(src_f32_dq);
        maxpool_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_u8);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&maxpool_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("int8_maxpool_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {&src_u8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_u8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_u8_case2_ts(dst_u8, case2_out_data.data());
        cp.execute(&strm, {src_u8_ts}, {dst_u8_case2_ts});
        strm.wait();

        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    }
}
