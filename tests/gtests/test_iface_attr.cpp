/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
* Copyright 2020-2021 FUJITSU LIMITED
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

#include "oneapi/dnnl/dnnl.hpp"
#include "tests/test_isa_common.hpp"

namespace dnnl {

using data_type = memory::data_type;
using tag = memory::format_tag;

class attr_test_t : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(attr_test_t, TestFPMathMode) {
    dnnl::primitive_attr attr;
    ASSERT_EQ(attr.get_fpmath_mode(), fpmath_mode::strict);

    for (auto m : {fpmath_mode::strict, fpmath_mode::bf16, fpmath_mode::f16,
                 fpmath_mode::tf32, fpmath_mode::any}) {
        attr.set_fpmath_mode(m);
        ASSERT_EQ(m, attr.get_fpmath_mode());
    }
}

TEST_F(attr_test_t, TestFPMathModeDefault) {
    ASSERT_EQ(fpmath_mode::strict, get_default_fpmath_mode());

    for (auto m : {fpmath_mode::strict, fpmath_mode::bf16, fpmath_mode::f16,
                 fpmath_mode::tf32, fpmath_mode::any}) {
        set_default_fpmath_mode(m);
        ASSERT_EQ(m, get_default_fpmath_mode());
        dnnl::primitive_attr attr;
        ASSERT_EQ(m, attr.get_fpmath_mode());
    }
}

TEST_F(attr_test_t, TestScratchpadMode) {
    dnnl::primitive_attr attr;
    for (auto m : {scratchpad_mode::library, scratchpad_mode::user}) {
        attr.set_scratchpad_mode(m);
        ASSERT_EQ(m, attr.get_scratchpad_mode());
    }
}

TEST_F(attr_test_t, TestScratchpadModeEx) {
    engine eng = get_test_engine();

    const memory::dim N = 2, C = 2, W = 2;

    memory::desc data_md(
            {N, C, W}, memory::data_type::f32, memory::format_tag::ncw);

    dnnl::primitive_attr attr;
    for (auto m : {scratchpad_mode::library, scratchpad_mode::user}) {
        attr.set_scratchpad_mode(m);
        auto softmax_pd = softmax_forward::primitive_desc(eng,
                prop_kind::forward_inference, algorithm::softmax_accurate,
                data_md, data_md, 1, attr);
        auto scratchpad_size = (long)softmax_pd.scratchpad_desc().get_size();
        auto mem_consumption
                = (long)softmax_pd.query_s64(query::memory_consumption_s64);

        if (m == scratchpad_mode::library) {
            ASSERT_EQ(scratchpad_size, 0L);
        } else {
            ASSERT_EQ(mem_consumption, 0L);
        }
    }
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, TestScratchpadArg) {
    engine eng = get_test_engine();

    const memory::dim N = 2, C = 2, W = 2;

    memory::desc data_md(
            {N, C, W}, memory::data_type::f32, memory::format_tag::ncw);

    dnnl::primitive_attr attr;
    for (auto m : {scratchpad_mode::library, scratchpad_mode::user}) {
        attr.set_scratchpad_mode(m);
        auto softmax_pd = softmax_forward::primitive_desc(eng,
                prop_kind::forward_inference, algorithm::softmax_accurate,
                data_md, data_md, 1, attr);

        auto src = test::make_memory(softmax_pd.src_desc(), eng);
        auto dst = test::make_memory(softmax_pd.dst_desc(), eng);
        auto scratchpad = test::make_memory(softmax_pd.scratchpad_desc(), eng);

        fill_data<float>(src.get_desc().get_size() / sizeof(float), src);

        stream s(eng);

        softmax_forward softmax_p(softmax_pd);
        softmax_p.execute(s,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst},
                        {DNNL_ARG_SCRATCHPAD, scratchpad}});
        s.wait();
    }
}

TEST_F(attr_test_t, TestZeroPoints) {
    dnnl::primitive_attr attr;

    const std::vector<int> supported_args = {DNNL_ARG_SRC, DNNL_ARG_DST};
    const std::vector<int> unsupported_args = {DNNL_ARG_BIAS, DNNL_ARG_DST_2,
            DNNL_ARG_MEAN, DNNL_ARG_WORKSPACE, DNNL_ARG_SCRATCHPAD};

    for (auto arg : supported_args) {
        // single non-default zero_point for supported arg
        attr.set_zero_points_mask(arg, 0);
    }

    for (auto arg : unsupported_args) {
        // single **default** zero_point for **unsupported** arg
        EXPECT_ANY_THROW(attr.set_zero_points_mask(arg, 0));
    }

    // multiple zero_points not implemented yet ...
}

TEST_F(attr_test_t, TestZeroPointsExpectFailure) {
    dnnl::primitive_attr attr;

    const int unsupported_arg = DNNL_ARG_MEAN;

    // single non-default zero_point for unsupported arg
    EXPECT_ANY_THROW(attr.set_zero_points_mask(unsupported_arg, 0));

    // multiple zero points for unsupported args
    EXPECT_ANY_THROW(attr.set_zero_points_mask(unsupported_arg, 1 << 1));
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, TestScales) {
    dnnl::primitive_attr attr;

    const std::vector<int> supported_args = {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1,
            DNNL_ARG_MULTIPLE_SRC, DNNL_ARG_MULTIPLE_SRC + 42};
    const std::vector<int> unsupported_args = {DNNL_ARG_BIAS, DNNL_ARG_DST_2,
            DNNL_ARG_MEAN, DNNL_ARG_WORKSPACE, DNNL_ARG_SCRATCHPAD};

    for (auto arg : supported_args) {
        // single non-default scales for supported arg
        attr.set_scales_mask(arg, 0);
        // multiple scales
        attr.set_scales_mask(arg, 1 << 1);
    }

    for (auto arg : unsupported_args) {
        // single scales for unsupported args
        EXPECT_ANY_THROW(attr.set_scales_mask(arg, 0));
    }
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, TestRNNDataQuantization) {
    dnnl::primitive_attr attr;

    float scale = NAN, shift = NAN;

    // default scale and shift
    attr.get_rnn_data_qparams(scale, shift);
    ASSERT_EQ(scale, 1.f);
    ASSERT_EQ(shift, 0.f);

    // non-default
    attr.set_rnn_data_qparams(0.5f, 2.f);
    attr.get_rnn_data_qparams(scale, shift);
    ASSERT_EQ(scale, 0.5f);
    ASSERT_EQ(shift, 2.f);
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, TestRNNWeightsQuantization) {
    dnnl::primitive_attr attr;

    int scales_mask = INT_MAX;
    std::vector<float> scales;

    // default scale and shift
    attr.get_rnn_weights_qparams(scales_mask, scales);
    ASSERT_EQ(scales_mask, 0);
    ASSERT_EQ(scales.size(), 1U);
    ASSERT_EQ(scales[0], 1.f);

    // single non-default scales
    attr.set_rnn_weights_qparams(0, {2.f});
    attr.get_rnn_weights_qparams(scales_mask, scales);
    ASSERT_EQ(scales_mask, 0);
    ASSERT_EQ(scales.size(), 1U);
    ASSERT_EQ(scales[0], 2.f);

    // multiple scales
    attr.set_rnn_weights_qparams(1 << 1, {1.f, 2.f, 4.f});
    attr.get_rnn_weights_qparams(scales_mask, scales);
    ASSERT_EQ(scales_mask, 1 << 1);
    ASSERT_EQ(scales.size(), 3U);
    ASSERT_EQ(scales[0], 1.f);
    ASSERT_EQ(scales[1], 2.f);
    ASSERT_EQ(scales[2], 4.f);
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, TestRNNProjWeightsQuantization) {
    dnnl::primitive_attr attr;

    int scales_mask = INT_MAX;
    std::vector<float> scales;

    // default scale and shift
    attr.get_rnn_weights_projection_qparams(scales_mask, scales);
    ASSERT_EQ(scales_mask, 0);
    ASSERT_EQ(scales.size(), 1U);
    ASSERT_EQ(scales[0], 1.f);

    // single non-default scales
    attr.set_rnn_weights_projection_qparams(0, {2.f});
    attr.get_rnn_weights_projection_qparams(scales_mask, scales);
    ASSERT_EQ(scales_mask, 0);
    ASSERT_EQ(scales.size(), 1U);
    ASSERT_EQ(scales[0], 2.f);

    // multiple scales
    attr.set_rnn_weights_projection_qparams(1 << 1, {1.f, 2.f, 4.f});
    attr.get_rnn_weights_projection_qparams(scales_mask, scales);
    ASSERT_EQ(scales_mask, 1 << 1);
    ASSERT_EQ(scales.size(), 3U);
    ASSERT_EQ(scales[0], 1.f);
    ASSERT_EQ(scales[1], 2.f);
    ASSERT_EQ(scales[2], 4.f);
}

TEST_F(attr_test_t, TestScalesExpectFailure) {
    dnnl::primitive_attr attr;
    const int unsupported_arg = DNNL_ARG_MEAN;

    // non-default scales for unsupported arg
    EXPECT_ANY_THROW(attr.set_scales_mask(unsupported_arg, 0));
    EXPECT_ANY_THROW(attr.set_scales_mask(unsupported_arg, 1 << 1));
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, TestPostOps) {
    dnnl::primitive_attr attr;
    dnnl::post_ops ops;

    algorithm alg = algorithm::undef;
    float scale = NAN, alpha = NAN, beta = NAN;
    int32_t sum_zp = INT_MAX;
    data_type dt = data_type::undef;

    ASSERT_EQ(ops.len(), 0);
    ASSERT_EQ(attr.get_post_ops().len(), 0);

    ops.append_sum(1.1f, 1, data_type::f32);
    attr.set_post_ops(ops);

    ASSERT_EQ(attr.get_post_ops().len(), 1);
    ASSERT_EQ(attr.get_post_ops().kind(0), primitive::kind::sum);
    attr.get_post_ops().get_params_sum(0, scale, sum_zp, dt);
    ASSERT_FLOAT_EQ(scale, 1.1f);
    ASSERT_EQ(1, sum_zp);
    ASSERT_EQ(data_type::f32, dt);

    ops.append_eltwise(algorithm::eltwise_clip, 3.3f, 4.4f);
    attr.set_post_ops(ops);

    ASSERT_EQ(attr.get_post_ops().len(), 2);
    ASSERT_EQ(attr.get_post_ops().kind(0), primitive::kind::sum);
    ASSERT_EQ(attr.get_post_ops().kind(1), primitive::kind::eltwise);
    attr.get_post_ops().get_params_eltwise(1, alg, alpha, beta);
    ASSERT_EQ(alg, algorithm::eltwise_clip);
    ASSERT_FLOAT_EQ(alpha, 3.3f);
    ASSERT_FLOAT_EQ(beta, 4.4f);

    memory::desc src1_md({1}, data_type::f32, {1});
    ops.append_binary(algorithm::binary_add, src1_md);
    attr.set_post_ops(ops);

    ASSERT_EQ(attr.get_post_ops().len(), 3);
    ASSERT_EQ(attr.get_post_ops().kind(0), primitive::kind::sum);
    ASSERT_EQ(attr.get_post_ops().kind(1), primitive::kind::eltwise);
    ASSERT_EQ(attr.get_post_ops().kind(2), primitive::kind::binary);
    memory::desc src1_md_out;
    attr.get_post_ops().get_params_binary(2, alg, src1_md_out);
    ASSERT_EQ(alg, algorithm::binary_add);
    ASSERT_EQ(src1_md, src1_md_out);

    const int prelu_mask = 1;
    ops.append_prelu(prelu_mask);
    attr.set_post_ops(ops);
    ASSERT_EQ(attr.get_post_ops().len(), 4);
    ASSERT_EQ(attr.get_post_ops().kind(0), primitive::kind::sum);
    ASSERT_EQ(attr.get_post_ops().kind(1), primitive::kind::eltwise);
    ASSERT_EQ(attr.get_post_ops().kind(2), primitive::kind::binary);
    ASSERT_EQ(attr.get_post_ops().kind(3), primitive::kind::prelu);
    int mask = INT_MAX;
    attr.get_post_ops().get_params_prelu(3, mask);
    ASSERT_EQ(mask, prelu_mask);
}

TEST_F(attr_test_t, TestPostOpsCheckLimit) {
    dnnl::post_ops ops_sum, ops_eltwise, ops_binary, ops_prelu;

    for (int i = 0; i < 32; i++) {
        EXPECT_NO_THROW(ops_sum.append_sum(i + 1.f));
        EXPECT_NO_THROW(ops_eltwise.append_eltwise(
                algorithm::eltwise_relu, 2 * i, 0.f));
        EXPECT_NO_THROW(ops_binary.append_binary(algorithm::binary_add,
                memory::desc({i}, data_type::s8, memory::format_tag::a)));
        EXPECT_NO_THROW(ops_prelu.append_prelu(1));
    }

    EXPECT_ANY_THROW(ops_prelu.append_prelu(1));
    EXPECT_ANY_THROW(ops_sum.append_sum(1.f));
    EXPECT_ANY_THROW(
            ops_eltwise.append_eltwise(algorithm::eltwise_relu, 1.f, 0.f));
    EXPECT_ANY_THROW(ops_binary.append_binary(algorithm::binary_add,
            memory::desc({1}, data_type::s8, memory::format_tag::a)));
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, TestSumPostOpQuantization) {
#define ALLOW_UNIMPL(f) \
    EXPECT_NO_THROW( \
            catch_expected_failures([&]() { f; }, false, dnnl_success, true))

    auto engine_kind = get_test_engine_kind();
    engine e {engine_kind, 0};

    std::vector<data_type> test_dts {data_type::f32, data_type::s8};

    if (!unsupported_data_type(data_type::bf16))
        test_dts.push_back(data_type::bf16);

    auto create_pd = [&e](primitive::kind pk, data_type dt,
                             primitive_attr &attr) {
        switch (pk) {
            case primitive::kind::convolution: {
                memory::desc dat_md {
                        {2, 64, 3, 3}, dt, memory::format_tag::any};
                memory::desc wei_md {
                        {64, 64, 3, 3}, dt, memory::format_tag::any};

                auto pd = convolution_forward::primitive_desc(e,
                        prop_kind::forward_inference,
                        algorithm::convolution_direct, dat_md, wei_md, dat_md,
                        {1, 1}, {1, 1}, {1, 1}, attr);
                break;
            }
            case primitive::kind::deconvolution: {
                memory::desc dat_md {
                        {2, 64, 3, 3}, dt, memory::format_tag::any};
                memory::desc wei_md {
                        {64, 64, 3, 3}, dt, memory::format_tag::any};

                auto pd = deconvolution_forward::primitive_desc(e,
                        prop_kind::forward_inference,
                        algorithm::deconvolution_direct, dat_md, wei_md, dat_md,
                        {1, 1}, {1, 1}, {1, 1}, attr);
                break;
            }
            case primitive::kind::inner_product: {
                memory::desc dat_md {{2, 64}, dt, memory::format_tag::any};
                memory::desc wei_md {{64, 64}, dt, memory::format_tag::any};

                auto pd = inner_product_forward::primitive_desc(e,
                        prop_kind::forward_inference, dat_md, wei_md, dat_md,
                        attr);
                break;
            }
            case primitive::kind::matmul: {
                memory::desc dat_md {{2, 64}, dt, memory::format_tag::any};
                memory::desc wei_md {{64, 64}, dt, memory::format_tag::any};

                auto pd = matmul::primitive_desc(
                        e, dat_md, wei_md, dat_md, attr);
                break;
            }
            default: assert(!"not expected primitive kind");
        }
    };

    for (auto pk :
            {primitive::kind::convolution, primitive::kind::deconvolution,
                    primitive::kind::inner_product, primitive::kind::matmul})
        for (auto sum_dt :
                {data_type::f32, data_type::s8, data_type::u8, data_type::s32})
            for (float s : {1.0f, 0.5f})
                for (int32_t zp : {0, 1}) {
                    // GPU doesn't support zero point
                    if (zp != 0 && e.get_kind() == engine::kind::gpu) continue;

                    dnnl::primitive_attr attr;
                    dnnl::post_ops ops;
                    ops.append_sum(s, zp, sum_dt);
                    attr.set_post_ops(ops);

                    for (auto dt : test_dts) {
                        if (memory::data_type_size(dt)
                                == memory::data_type_size(sum_dt)) {
                            if (zp != 0) {
                                if (dt != data_type::s8) {
                                    // unsupported scenario
                                    EXPECT_ANY_THROW(create_pd(pk, dt, attr));
                                } else {
                                    if (impl::utils::one_of(sum_dt,
                                                data_type::s8, data_type::u8,
                                                data_type::s32))
                                        ALLOW_UNIMPL(create_pd(pk, dt, attr));
                                    else
                                        // unsupported scenario
                                        EXPECT_ANY_THROW(
                                                create_pd(pk, dt, attr));
                                }
                            } else {
                                // gemm-style beta support
                                ALLOW_UNIMPL(create_pd(pk, dt, attr));
                            }
                        } else {
                            // unsupported scenario
                            EXPECT_ANY_THROW(create_pd(pk, dt, attr));
                        }
                    }
                }
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, DepthwiseFusionPostop) {
    dnnl::primitive_attr attr;
    dnnl::post_ops ops;

    data_type wei_dt = data_type::undef;
    data_type bias_dt = data_type::undef;
    data_type dst_dt = data_type::undef;
    memory::dim kernel = -1;
    memory::dim stride = -1;
    memory::dim padding = -1;

    ASSERT_EQ(ops.len(), 0);
    ASSERT_EQ(attr.get_post_ops().len(), 0);

    ops.append_dw(memory::data_type::s8, memory::data_type::f32,
            memory::data_type::u8, 3, 1, 1);
    attr.set_post_ops(ops);

    ASSERT_EQ(attr.get_post_ops().kind(0), primitive::kind::convolution);
    attr.get_post_ops().get_params_dw(
            0, wei_dt, bias_dt, dst_dt, kernel, stride, padding);
    ASSERT_EQ(wei_dt, memory::data_type::s8);
    ASSERT_EQ(bias_dt, memory::data_type::f32);
    ASSERT_EQ(dst_dt, memory::data_type::u8);
    ASSERT_EQ(kernel, 3);
    ASSERT_EQ(stride, 1);
    ASSERT_EQ(padding, 1);

    kernel = stride = padding = -1;
    ops.append_dw(memory::data_type::u8, memory::data_type::s32,
            memory::data_type::f32, 3, 2, 1);
    attr.set_post_ops(ops);

    ASSERT_EQ(attr.get_post_ops().kind(0), primitive::kind::convolution);
    ASSERT_EQ(attr.get_post_ops().kind(1), primitive::kind::convolution);

    attr.get_post_ops().get_params_dw(
            1, wei_dt, bias_dt, dst_dt, kernel, stride, padding);

    ASSERT_EQ(wei_dt, memory::data_type::u8);
    ASSERT_EQ(bias_dt, memory::data_type::s32);
    ASSERT_EQ(dst_dt, memory::data_type::f32);
    ASSERT_EQ(kernel, 3);
    ASSERT_EQ(stride, 2);
    ASSERT_EQ(padding, 1);

    kernel = stride = padding = -1;
    ops.append_dw(memory::data_type::f32, memory::data_type::f32,
            memory::data_type::f32, 7, 3, 2);
    attr.set_post_ops(ops);

    ASSERT_EQ(attr.get_post_ops().kind(0), primitive::kind::convolution);
    ASSERT_EQ(attr.get_post_ops().kind(1), primitive::kind::convolution);
    ASSERT_EQ(attr.get_post_ops().kind(2), primitive::kind::convolution);

    attr.get_post_ops().get_params_dw(
            2, wei_dt, bias_dt, dst_dt, kernel, stride, padding);

    ASSERT_EQ(wei_dt, memory::data_type::f32);
    ASSERT_EQ(bias_dt, memory::data_type::f32);
    ASSERT_EQ(dst_dt, memory::data_type::f32);
    ASSERT_EQ(kernel, 7);
    ASSERT_EQ(stride, 3);
    ASSERT_EQ(padding, 2);

    kernel = stride = padding = -1;
    ops.append_dw(memory::data_type::s8, memory::data_type::f32,
            memory::data_type::u8, 5, 2, 1);
    attr.set_post_ops(ops);

    ASSERT_EQ(attr.get_post_ops().kind(3), primitive::kind::convolution);

    attr.get_post_ops().get_params_dw(
            3, wei_dt, bias_dt, dst_dt, kernel, stride, padding);

    ASSERT_EQ(wei_dt, memory::data_type::s8);
    ASSERT_EQ(bias_dt, memory::data_type::f32);
    ASSERT_EQ(dst_dt, memory::data_type::u8);
    ASSERT_EQ(kernel, 5);
    ASSERT_EQ(stride, 2);
    ASSERT_EQ(padding, 1);
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, DepthwiseFusion) {

    auto engine_kind = get_test_engine_kind();
    SKIP_IF(engine_kind != engine::kind::cpu,
            "Depthwise fusion is only supported on CPU engine");
#if DNNL_AARCH64
    SKIP_IF(true, "Depthwise fusion is not supported on AArch64 at this time");
#endif

    engine e {engine_kind, 0};

    std::vector<memory::data_type> test_dts {
            memory::data_type::f32, memory::data_type::s8};

    if (!unsupported_data_type(memory::data_type::bf16))
        test_dts.push_back(memory::data_type::bf16);

    for (auto dt : test_dts) {

        memory::desc dat_md {{1024, 512, 64, 64}, dt, memory::format_tag::any};
        memory::desc wht_md {{512, 512, 1, 1}, dt, memory::format_tag::any};

        std::string impl_info_unfused;

        auto pd = convolution_forward::primitive_desc(e,
                prop_kind::forward_inference, algorithm::convolution_auto,
                dat_md, wht_md, dat_md, {1, 1}, {0, 0}, {0, 0});

        ASSERT_NO_THROW(impl_info_unfused = pd.impl_info_str(););

        // skip if above unfused impl is not jitted.
        if (impl_info_unfused.compare(0, 3, "jit") != 0) continue;

        // skip if above unfused impl is jitted amx.
        if (impl_info_unfused.find("amx") != std::string::npos) continue;

        dnnl::primitive_attr attr;
        dnnl::post_ops ops;
        ops.append_dw(dt, dt, dt, 3, 1, 1);
        attr.set_post_ops(ops);

        std::string impl_info_fused;

        pd = convolution_forward::primitive_desc(e,
                prop_kind::forward_inference, algorithm::convolution_auto,
                dat_md, wht_md, dat_md, {1, 1}, {0, 0}, {0, 0}, attr);
        ASSERT_NO_THROW(impl_info_fused = pd.impl_info_str(););

        // Make sure ref fused impl is not deployed.
        // NOTE: When out_of_memory testing enabled, all implementations that
        // construct primitive attributes will fail, hence the ref
        // implementation is deployed.
        if (!test_out_of_memory()) {
            ASSERT_EQ(impl_info_fused, impl_info_unfused);
        }
    }
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, InnerProdBlockedWeights) {
    auto engine_kind = get_test_engine_kind();
    bool skip_test = !DNNL_X64 || (DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE)
            || (engine_kind != engine::kind::cpu);
#if DNNL_X64 && (DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE)
    skip_test = skip_test || !dnnl::mayiuse(cpu_isa::avx512_core);
#endif
    SKIP_IF(skip_test,
            "Inner product blocked weights test is supported only on "
            "avx512_core CPU");

    engine e {engine_kind, 0};

    std::vector<memory::format_tag> blocked_weights_tags {
            memory::format_tag::OIhw16i64o, memory::format_tag::OIhw16i48o,
            memory::format_tag::OIhw16i32o, memory::format_tag::OIhw16i16o};

    for (const auto &weights_tag : blocked_weights_tags) {
        memory::desc src_md {{1024, 512, 1, 1}, memory::data_type::f32,
                memory::format_tag::any};
        memory::desc wei_md {
                {256, 512, 1, 1}, memory::data_type::f32, weights_tag};
        memory::desc bia_md {
                {256}, memory::data_type::f32, memory::format_tag::any};
        memory::desc dst_md {
                {1024, 256}, memory::data_type::f32, memory::format_tag::any};

        auto pd = inner_product_forward::primitive_desc(
                e, prop_kind::forward_training, src_md, wei_md, bia_md, dst_md);

        std::string impl_info;
        ASSERT_NO_THROW(impl_info = pd.impl_info_str(););
        ASSERT_NE(impl_info, "ref:any");
    }
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, TestGetAttr) {
    auto engine_kind = get_test_engine_kind();
    SKIP_IF(engine_kind != engine::kind::cpu,
            "Depthwise fusion is only supported on CPU engine");

    engine eng {engine_kind, 0};

    auto dt = memory::data_type::s8;
    dnnl::primitive_attr attr_s, attr_os, attr_dw;
    dnnl::post_ops ops;
    ops.append_dw(dt, dt, dt, 3, 1, 1);
    attr_s.set_scales_mask(DNNL_ARG_SRC_0, 0);
    attr_os.set_scales_mask(DNNL_ARG_DST, 0);
    attr_dw.set_scales_mask(
            DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS, (1 << 1) + (1 << 0));
    attr_dw.set_post_ops(ops);

    memory::desc dat_md {{512, 512, 3, 3}, dt, memory::format_tag::nchw};
    memory::desc wht_md {{512, 512, 1, 1}, dt, memory::format_tag::nchw};
    auto bin_pd = binary::primitive_desc(
            eng, algorithm::binary_add, wht_md, wht_md, wht_md, attr_s);

    auto cd_pd_os = convolution_forward::primitive_desc(eng,
            prop_kind::forward_inference, algorithm::convolution_auto, dat_md,
            wht_md, dat_md, {1, 1}, {0, 0}, {0, 0}, attr_os);
    auto cd_pd_dw = convolution_forward::primitive_desc(eng,
            prop_kind::forward_inference, algorithm::convolution_auto, dat_md,
            wht_md, dat_md, {1, 1}, {0, 0}, {0, 0}, attr_dw);
    if (test_out_of_memory()) {
        attr_s = bin_pd.get_primitive_attr();
        attr_os = cd_pd_os.get_primitive_attr();
        attr_dw = cd_pd_dw.get_primitive_attr();
    } else {
        ASSERT_NO_THROW(attr_s = bin_pd.get_primitive_attr());
        ASSERT_NO_THROW(attr_os = cd_pd_os.get_primitive_attr());
        ASSERT_NO_THROW(attr_dw = cd_pd_dw.get_primitive_attr());
    }
}

HANDLE_EXCEPTIONS_FOR_TEST_F(attr_test_t, TestGetCppObjects) {
    SKIP_IF_CUDA(true, "Binary post-op is not supported for CUDA");
    SKIP_IF_HIP(true, "Binary post-op is not supported for HIP");

    auto engine_kind = get_test_engine_kind();
    engine eng {engine_kind, 0};

    // Post-ops is the only object that is returned from primitive attr, rest
    // calls are of `void` type. Lack of "cloning" for post-ops led to a problem
    // of using a dangling pointer from destroyed object via
    // `pd.get_primitive_attr().get_post_ops()` construction as attributes will
    // be destroyed once post-ops are saved on stack.
    // See https://github.com/oneapi-src/oneDNN/issues/1337 for details.
    dnnl::primitive_attr attr;
    dnnl::post_ops ops;
    memory::desc po_src1_md({1, 1, 1, 1}, data_type::f32, tag::abcd);
    ops.append_binary(algorithm::binary_add, po_src1_md);
    attr.set_post_ops(ops);

    memory::desc md {{512, 512, 3, 3}, data_type::f32, tag::abcd};
    auto bin_pd = binary::primitive_desc(
            eng, algorithm::binary_add, md, md, md, attr);

    const auto q_po = bin_pd.get_primitive_attr().get_post_ops();
    ASSERT_EQ(q_po.len(), 1);
    ASSERT_EQ(q_po.kind(0), primitive::kind::binary);

    algorithm q_alg;
    memory::desc q_po_src1_md;
    q_po.get_params_binary(0, q_alg, q_po_src1_md);
    ASSERT_EQ(q_alg, algorithm::binary_add);
    ASSERT_EQ(q_po_src1_md, po_src1_md);
}

} // namespace dnnl
