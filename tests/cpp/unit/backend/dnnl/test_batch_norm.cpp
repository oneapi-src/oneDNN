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
#include <memory>

#include "gtest/gtest.h"

#include "cpp/unit/backend/dnnl/dnnl_test_common.hpp"
#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

static inline void ref_batchnorm_fwd(impl::dim_t mb, impl::dim_t ic,
        impl::dim_t ih, impl::dim_t iw, impl::tensor_t *src,
        impl::tensor_t *dst, impl::tensor_t *scale, impl::tensor_t *shift,
        impl::tensor_t *mean = nullptr, impl::tensor_t *variance = nullptr,
        impl::tensor_t *running_mean = nullptr,
        impl::tensor_t *running_variance = nullptr,
        impl::tensor_t *batch_mean = nullptr,
        impl::tensor_t *batch_variance = nullptr, float epsilon = 0.001f,
        float momentum = 0.1f,
        std::function<float(const float)> activation = nullptr,
        bool channel_last = false, bool is_training = false) {
    if (!activation) activation = [](const float v) { return v; };

    size_t mb_ = static_cast<size_t>(mb);
    size_t ic_ = static_cast<size_t>(ic);
    size_t ih_ = static_cast<size_t>(ih);
    size_t iw_ = static_cast<size_t>(iw);

    float *src_ptr = src->get_data_handle<float>();
    float *dst_ptr = dst->get_data_handle<float>();

    if (is_training) {
        float *mean_ptr = batch_mean->get_data_handle<float>();
        float *variance_ptr = batch_variance->get_data_handle<float>();
        float *scale_ptr = scale->get_data_handle<float>();
        float *shift_ptr = shift->get_data_handle<float>();
        float *est_mean_ptr = mean->get_data_handle<float>();
        float *est_variance_ptr = variance->get_data_handle<float>();
        float *running_mean_ptr = running_mean->get_data_handle<float>();
        float *running_variance_ptr
                = running_variance->get_data_handle<float>();
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
        for (int k = 0; k < ic_; k++) {
            running_mean_ptr[k]
                    = momentum * est_mean_ptr[k] + (1 - momentum) * mean_ptr[k];
            running_variance_ptr[k] = momentum * est_variance_ptr[k]
                    + (1 - momentum) * variance_ptr[k];
        }
    } else {
        float *mean_ptr = nullptr;
        float *variance_ptr = nullptr;
        float *scale_ptr = scale->get_data_handle<float>();
        float *shift_ptr = shift->get_data_handle<float>();
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
}

struct dnnl_graph_test_batchnorm_params {
    impl::op_kind_t op_kind;
    bool with_relu;
    impl::dim_t N;
    impl::dim_t IC;
    impl::dim_t IH;
    impl::dim_t IW;
    float epsilon;
    std::string data_format;
    std::string backend_name;
    impl::data_type_t data_type;
};

class BatchNorm4D
    : public ::testing::TestWithParam<dnnl_graph_test_batchnorm_params> {
public:
    void TestBatchnorm(bool dims_in_order = true) {
        using dims = dnnl::graph::impl::dnnl_impl::dims;

        auto params = ::testing::TestWithParam<
                dnnl_graph_test_batchnorm_params>::GetParam();

        impl::op_t batchnorm_op(0, params.op_kind, "batchnorm");
        batchnorm_op.set_attr(impl::op_attr::epsilon, params.epsilon);
        batchnorm_op.set_attr(impl::op_attr::data_format, params.data_format);
        impl::op_t relu_op(1, impl::op_kind::ReLU, "relu");

        bool is_training
                = params.op_kind == impl::op_kind::BatchNormForwardTraining;
        const float momentum = 0.1f;
        if (is_training)
            batchnorm_op.set_attr(impl::op_attr::momentum, momentum);

        impl::engine_t &engine = get_engine();

        SKIP_IF(engine.kind() == impl::engine_kind::gpu && is_training,
                "Skip the case due to regression of sum primitive on GPU");

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
        // if ((params.data_format == "NCX") ^ dims_in_order)
        if (params.data_format == "NXC")
            src_dims = {N, IH, IW, IC};
        else
            src_dims = {N, IC, IH, IW};
        // Scale/shift tensor dimensions.
        dims scale_dims = {IC};
        dims shift_dims = {IC};
        dims mean_dims = {IC};
        dims variance_dims = {IC};

        // prepare logical tensor
        impl::logical_tensor_t src
                = utils::logical_tensor_init(0, src_dims, impl::data_type::f32);
        impl::logical_tensor_t scale = utils::logical_tensor_init(
                1, scale_dims, impl::data_type::f32);
        impl::logical_tensor_t shift = utils::logical_tensor_init(
                2, shift_dims, impl::data_type::f32);
        impl::logical_tensor_t mean = utils::logical_tensor_init(
                3, mean_dims, impl::data_type::f32);
        impl::logical_tensor_t variance = utils::logical_tensor_init(
                4, variance_dims, impl::data_type::f32);
        impl::logical_tensor_t running_mean = utils::logical_tensor_init(
                5, mean_dims, impl::data_type::f32);
        impl::logical_tensor_t running_variance = utils::logical_tensor_init(
                6, variance_dims, impl::data_type::f32);
        impl::logical_tensor_t batch_mean = utils::logical_tensor_init(
                7, mean_dims, impl::data_type::f32);
        impl::logical_tensor_t batch_variance = utils::logical_tensor_init(
                8, variance_dims, impl::data_type::f32);
        impl::logical_tensor_t dst
                = utils::logical_tensor_init(9, src_dims, impl::data_type::f32);
        impl::logical_tensor_t relu_dst = utils::logical_tensor_init(
                10, src_dims, impl::data_type::f32);

        batchnorm_op.add_input(src);
        if (!is_training) {
            batchnorm_op.add_input(scale);
            batchnorm_op.add_input(shift);
        }
        batchnorm_op.add_input(mean);
        batchnorm_op.add_input(variance);
        if (is_training) {
            batchnorm_op.add_input(scale);
            batchnorm_op.add_input(shift);
        }
        batchnorm_op.add_output(dst);
        if (is_training) {
            batchnorm_op.add_output(running_mean);
            batchnorm_op.add_output(running_variance);
            batchnorm_op.add_output(batch_mean);
            batchnorm_op.add_output(batch_variance);
        }

        relu_op.add_input(dst);
        relu_op.add_output(relu_dst);

        impl::graph_t g(engine.kind());
        g.add_op(&batchnorm_op);
        if (params.with_relu) g.add_op(&relu_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = params.with_relu
                ? get_pass("bn_relu_fusion")
                : (is_training ? get_pass("bn_fw_train_pass")
                               : get_pass("bn_pass"));
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);
        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {&src};
        if (!is_training) {
            inputs.emplace_back(&scale);
            inputs.emplace_back(&shift);
        }
        inputs.emplace_back(&mean);
        inputs.emplace_back(&variance);
        if (is_training) {
            inputs.emplace_back(&scale);
            inputs.emplace_back(&shift);
        }
        std::vector<const impl::logical_tensor_t *> outputs {&dst};
        if (params.with_relu) outputs[0] = &relu_dst;
        if (is_training) {
            outputs.emplace_back(&running_mean);
            outputs.emplace_back(&running_variance);
            outputs.emplace_back(&batch_mean);
            outputs.emplace_back(&batch_variance);
        }

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine),
                impl::status::success);

        // Allocate buffers.
        test::vector<float> src_data(static_cast<size_t>(N * IC * IH * IW));
        test::vector<float> scale_data(static_cast<size_t>(IC));
        test::vector<float> shift_data(static_cast<size_t>(IC));
        test::vector<float> mean_data(static_cast<size_t>(IC));
        test::vector<float> variance_data(static_cast<size_t>(IC));
        test::vector<float> running_mean_data(static_cast<size_t>(IC));
        test::vector<float> running_variance_data(static_cast<size_t>(IC));
        test::vector<float> batch_mean_data(static_cast<size_t>(IC));
        test::vector<float> batch_variance_data(static_cast<size_t>(IC));
        test::vector<float> dst_data(src_data.size(), 0.0);
        test::vector<float> ref_dst_data(src_data.size(), 0.0);
        test::vector<float> ref_running_mean_data(
                running_mean_data.size(), 0.0);
        test::vector<float> ref_running_variance_data(
                running_variance_data.size(), 0.0);
        test::vector<float> ref_batch_mean_data(batch_mean_data.size(), 0.0);
        test::vector<float> ref_batch_variance_data(
                batch_variance_data.size(), 0.0);
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
        if (!is_training) {
            // Initialize mean.
            std::generate(mean_data.begin(), mean_data.end(), []() {
                static int i = 0;
                return std::sin(static_cast<float>(i++) * 2.f);
            });
            // Initialize variance.
            std::generate(variance_data.begin(), variance_data.end(), []() {
                static int i = 0;
                return static_cast<float>(i++);
            });
        } else {
            // Initialize mean.
            std::generate(
                    running_mean_data.begin(), running_mean_data.end(), []() {
                        static int i = 0;
                        return std::sin(static_cast<float>(i++) * 2.f);
                    });
            // Initialize variance.
            std::generate(running_variance_data.begin(),
                    running_variance_data.end(), []() {
                        static int i = 0;
                        return static_cast<float>(i++);
                    });
        }

        impl::tensor_t src_ts(src, &engine, src_data.data());
        impl::tensor_t scale_ts(scale, &engine, scale_data.data());
        impl::tensor_t shift_ts(shift, &engine, shift_data.data());
        impl::tensor_t mean_ts(mean, &engine, mean_data.data());
        impl::tensor_t variance_ts(variance, &engine, variance_data.data());
        impl::tensor_t running_mean_ts(
                running_mean, &engine, running_mean_data.data());
        impl::tensor_t running_variance_ts(
                running_variance, &engine, running_variance_data.data());
        impl::tensor_t batch_mean_ts(
                batch_mean, &engine, batch_mean_data.data());
        impl::tensor_t batch_variance_ts(
                batch_variance, &engine, batch_variance_data.data());
        impl::tensor_t dst_ts = params.with_relu
                ? impl::tensor_t(relu_dst, &engine, dst_data.data())
                : impl::tensor_t(dst, &engine, dst_data.data());
        impl::tensor_t ref_dst_ts = params.with_relu
                ? impl::tensor_t(relu_dst, &engine, ref_dst_data.data())
                : impl::tensor_t(dst, &engine, ref_dst_data.data());
        impl::tensor_t ref_running_mean_ts(
                running_mean, &engine, ref_running_mean_data.data());
        impl::tensor_t ref_running_variance_ts(
                running_variance, &engine, ref_running_variance_data.data());
        impl::tensor_t ref_batch_mean_ts(
                batch_mean, &engine, ref_batch_mean_data.data());
        impl::tensor_t ref_batch_variance_ts(
                batch_variance, &engine, ref_batch_variance_data.data());

        impl::stream_t &strm = get_stream();

        if (!is_training) {
            cp.execute(&strm,
                    {src_ts, scale_ts, shift_ts, mean_ts, variance_ts},
                    {dst_ts});
        } else {
            cp.execute(&strm,
                    {src_ts, mean_ts, variance_ts, scale_ts, shift_ts},
                    {dst_ts, running_mean_ts, running_variance_ts,
                            batch_mean_ts, batch_variance_ts});
        }

        strm.wait();

        std::function<float(const float)> activation = nullptr;
        if (params.with_relu) {
            activation = [](const float v) { return v < 0.f ? 0.f : v; };
        }
        if (!is_training) {
            ref_batchnorm_fwd(N, IC, IH, IW, &src_ts, &ref_dst_ts, &scale_ts,
                    &shift_ts, &mean_ts, &variance_ts, nullptr, nullptr,
                    nullptr, nullptr, params.epsilon, momentum, activation,
                    params.data_format == "NXC",
                    params.op_kind == impl::op_kind::BatchNormForwardTraining);
        } else {
            ref_batchnorm_fwd(N, IC, IH, IW, &src_ts, &ref_dst_ts, &scale_ts,
                    &shift_ts, &mean_ts, &variance_ts, &ref_running_mean_ts,
                    &ref_running_variance_ts, &ref_batch_mean_ts,
                    &ref_batch_variance_ts, params.epsilon, momentum,
                    activation, params.data_format == "NXC",
                    params.op_kind == impl::op_kind::BatchNormForwardTraining);
            for (size_t i = 0; i < IC; ++i) {
                ASSERT_NEAR(
                        running_mean_data[i], ref_running_mean_data[i], 1.e-6f);
                ASSERT_NEAR(running_variance_data[i],
                        ref_running_variance_data[i], 1.e-6f);
                ASSERT_NEAR(batch_mean_data[i], ref_batch_mean_data[i], 1.e-6f);
                ASSERT_NEAR(batch_variance_data[i], ref_batch_variance_data[i],
                        1.e-6f);
            }
        }
        for (size_t i = 0; i < dst_data.size(); ++i) {
            ASSERT_NEAR(dst_data[i], ref_dst_data[i], 1.e-6f);
        }
    }

    void Test() {
        TestBatchnorm(true);
        TestBatchnorm(false);
    }
};

TEST_P(BatchNorm4D, TestBatchnorm) {
    Test();
}

INSTANTIATE_TEST_SUITE_P(Execute, BatchNorm4D,
        ::testing::Values(
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, false, 3, 3, 2, 2,
                        0.001f, "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, true, 3, 3, 2, 2,
                        0.001f, "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, false, 3, 3, 2, 2,
                        0.001f, "NXC", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormForwardTraining, false, 3, 3, 2,
                        2, 0.001f, "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormForwardTraining, false, 3, 3, 2,
                        2, 0.001f, "NXC", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormForwardTraining, true, 3, 3, 2,
                        2, 0.001f, "NXC", "dnnl", impl::data_type::f32}));

TEST(Compile, BatchNormBackpropFp32) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::op_t bn_op(impl::op_kind::BatchNormTrainingBackprop);
    bn_op.set_attr<float>(impl::op_attr::epsilon, 0.0);
    bn_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");

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
    dims variance_dims = {IC};

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t scale = utils::logical_tensor_init(
            1, scale_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t mean = utils::logical_tensor_init(
            2, mean_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t variance = utils::logical_tensor_init(
            3, variance_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            4, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_src = utils::logical_tensor_init(
            5, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_scale = utils::logical_tensor_init(
            6, scale_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_shift = utils::logical_tensor_init(
            7, shift_dims, impl::data_type::f32, impl::layout_type::strided);

    // Allocate buffers.
    test::vector<float> src_data {1.0f, 2.0f, 3.0f, 4.0f};
    test::vector<float> scale_data {2.0f};
    test::vector<float> mean_data {2.5f};
    test::vector<float> variance_data {1.25f};
    test::vector<float> diff_dst_data {0.1f, 0.1f, 0.2f, 0.2f};
    test::vector<float> diff_src_data(src_data.size(), 0.0);
    test::vector<float> diff_scale_data(scale_data.size(), 0.0);
    test::vector<float> diff_shift_data(scale_data.size(), 0.0);

    // expected results
    test::vector<float> ref_diff_src_data {
            0.01788854f, -0.05366563f, 0.05366563f, -0.01788854f};
    test::vector<float> ref_diff_scale_data {0.17888544f};
    test::vector<float> ref_diff_shift_data {0.6f};

    bn_op.add_input(src);
    bn_op.add_input(diff_dst);
    bn_op.add_input(mean);
    bn_op.add_input(variance);
    bn_op.add_input(scale);
    bn_op.add_output(diff_src);
    bn_op.add_output(diff_scale);
    bn_op.add_output(diff_shift);

    impl::graph_t g(engine.kind());
    g.add_op(&bn_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("bn_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src, &diff_dst, &mean, &variance, &scale};
    std::vector<const impl::logical_tensor_t *> outputs {
            &diff_src, &diff_scale, &diff_shift};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t scale_ts(scale, &engine, scale_data.data());
    impl::tensor_t mean_ts(mean, &engine, mean_data.data());
    impl::tensor_t variance_ts(variance, &engine, variance_data.data());
    impl::tensor_t diff_dst_ts(diff_dst, &engine, diff_dst_data.data());
    impl::tensor_t diff_src_ts(diff_src, &engine, diff_src_data.data());
    impl::tensor_t diff_scale_ts(diff_scale, &engine, diff_scale_data.data());
    impl::tensor_t diff_shift_ts(diff_shift, &engine, diff_shift_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, diff_dst_ts, mean_ts, variance_ts, scale_ts},
            {diff_src_ts, diff_scale_ts, diff_shift_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src_data.size(); ++i) {
        ASSERT_NEAR(diff_src_data[i], ref_diff_src_data[i], 1e-3);
    }
    for (size_t i = 0; i < diff_scale_data.size(); ++i) {
        ASSERT_NEAR(diff_scale_data[i], ref_diff_scale_data[i], 1e-3);
    }
    for (size_t i = 0; i < diff_shift_data.size(); ++i) {
        ASSERT_NEAR(diff_shift_data[i], ref_diff_shift_data[i], 1e-3);
    }
}
