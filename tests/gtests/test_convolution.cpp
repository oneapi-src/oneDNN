/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

namespace mkldnn {

template <typename data_t>
void compute_ref_conv_fwd(test_convolution_descr_t c, memory src,
        memory weights, memory bias, memory dst)
{
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *weights_data = (data_t *)weights.get_data_handle();
    data_t *bias_data
            = (data_t *)(bias.get() ? bias.get_data_handle() : nullptr);
    data_t *dst_data = (data_t *)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

#pragma omp parallel for collapse(5)
    for (uint32_t n = 0; n < c.mb; n++) {
        for (uint32_t g = 0; g < c.ng; g++) {
            for (uint32_t oc = 0; oc < c.oc / c.ng; oc++) {
                for (uint32_t oh = 0; oh < c.oh; oh++) {
                    for (uint32_t ow = 0; ow < c.ow; ow++) {
                        uint32_t oidx = n * c.oc * c.oh * c.ow
                                + g * c.oc / c.ng * c.oh * c.ow
                                + oc * c.oh * c.ow + oh * c.ow + ow;
                        dst_data[map_index(dst_d, oidx)] = bias_data ?
                                bias_data[map_index(
                                        bias.get_primitive_desc().desc(),
                                        g * c.oc / c.ng + oc)] :
                                0.0;
                        for (uint32_t ic = 0; ic < c.ic / c.ng; ic++) {
                            for (uint32_t kh = 0; kh < c.kh; kh++) {
                                for (uint32_t kw = 0; kw < c.kw; kw++) {
                                    int32_t iw = ow * c.strw - c.padw + kw;
                                    int32_t ih = oh * c.strh - c.padh + kh;
                                    if (iw < 0 || iw >= (int32_t)c.iw || ih < 0
                                            || ih >= (int32_t)c.ih)
                                        continue;
                                    uint32_t iidx = n * c.ic * c.ih * c.iw
                                            + g * c.ic / c.ng * c.ih * c.iw
                                            + ic * c.ih * c.iw + ih * c.iw + iw;
                                    uint32_t widx = g * c.oc / c.ng * c.ic
                                                    / c.ng * c.kh * c.kw
                                            + oc * c.ic / c.ng * c.kh * c.kw
                                            + ic * c.kh * c.kw + kh * c.kw + kw;

                                    dst_data[map_index(dst_d, oidx)]
                                            += src_data[map_index(src_d, iidx)]
                                            * weights_data[map_index(
                                                      weights_d, widx)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

struct conv_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    convolution::algorithm aalgorithm;
    memory::format src_format;
    memory::format weights_format;
    memory::format bias_format;
    memory::format dst_format;
    test_convolution_descr_t test_cd;
};

template <typename data_t>
class convolution_test : public ::testing::TestWithParam<conv_test_params> {
protected:
    virtual void SetUp()
    {
        conv_test_params p
                = ::testing::TestWithParam<conv_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu
                || p.engine_kind == engine::kind::cpu_lazy);
        ASSERT_EQ(p.aprop_kind, prop_kind::forward);
        ASSERT_EQ(p.aalgorithm, convolution::direct);
        auto eng = engine(p.engine_kind, 0);
        memory::precision prec = data_traits<data_t>::prec;
        ASSERT_EQ(prec, mkldnn::memory::precision::f32);

        test_convolution_descr_t cd = p.test_cd;

        auto c_src_desc
                = create_md({ cd.mb, cd.ic, cd.ih, cd.iw }, prec, p.src_format);
        auto c_weights_desc = cd.ng > 1 ?
                create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                        prec, p.weights_format) :
                create_md(
                        { cd.oc, cd.ic, cd.kh, cd.kw }, prec, p.weights_format);
        auto c_dst_desc
                = create_md({ cd.mb, cd.oc, cd.oh, cd.ow }, prec, p.dst_format);

        auto c_src = memory(memory::primitive_desc(c_src_desc, eng));
        auto c_weights = memory(memory::primitive_desc(c_weights_desc, eng));
        auto c_dst = memory(memory::primitive_desc(c_dst_desc, eng));

        auto dst_ref = memory(memory::primitive_desc(c_dst_desc, eng));

        fill_data<data_t>(c_src.get_primitive_desc().get_number_of_elements(),
                (data_t *)c_src.get_data_handle());

        fill_data<data_t>(
                c_weights.get_primitive_desc().get_number_of_elements(),
                (data_t *)c_weights.get_data_handle());

        bool with_bias = p.bias_format != memory::format::format_undef;
        auto c_bias_desc = with_bias ?
                create_md({ cd.oc }, prec, p.bias_format) :
                create_md({}, prec, p.bias_format);
        auto c_bias = memory(memory::primitive_desc(c_bias_desc, eng));
        if (with_bias) {
            fill_data<data_t>(
                    c_bias.get_primitive_desc().get_number_of_elements(),
                    (data_t *)c_bias.get_data_handle());
        }

        auto c = with_bias ?
                convolution(p.aprop_kind, p.aalgorithm, c_src, c_weights,
                        c_bias, c_dst, { cd.strh, cd.strw },
                        { cd.padh, cd.padw }, padding_kind::zero) :
                convolution(p.aprop_kind, p.aalgorithm, c_src, c_weights, c_dst,
                        { cd.strh, cd.strw }, { cd.padh, cd.padw },
                        padding_kind::zero);

        std::vector<primitive> pipeline;
        pipeline.push_back(c);

        stream().submit(pipeline).wait();

        compute_ref_conv_fwd<data_t>(cd, c_src, c_weights, c_bias, dst_ref);
        compare_data<data_t>(dst_ref, c_dst);
    }
};

using convolution_test_float = convolution_test<float>;
using conv_test_params_float = conv_test_params;

TEST_P(convolution_test_float, TestsConvolution)
{
}
INSTANTIATE_TEST_CASE_P(
        TestConvolutionForward, convolution_test_float,
        ::testing::Values(
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nchw,
                        memory::format::oihw, memory::format::x,
                        memory::format::nchw,
                        { 2, 1, 4, 4, 4, 6, 4, 4, 3, 3, 1, 1, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nchw,
                        memory::format::oihw, memory::format::x,
                        memory::format::nchw,
                        { 2, 1, 4, 4, 4, 6, 2, 2, 3, 3, 0, 0, 1, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionForwardNoBias, convolution_test_float,
        ::testing::Values(
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nchw,
                        memory::format::oihw, memory::format::format_undef,
                        memory::format::nchw,
                        { 2, 1, 4, 4, 4, 6, 4, 4, 3, 3, 1, 1, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nchw,
                        memory::format::oihw, memory::format::format_undef,
                        memory::format::nchw,
                        { 2, 1, 4, 4, 4, 6, 2, 2, 3, 3, 0, 0, 1, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionForwardNHWC, convolution_test_float,
        ::testing::Values(
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nhwc,
                        memory::format::oihw, memory::format::x,
                        memory::format::nhwc,
                        { 2, 1, 4, 4, 4, 6, 4, 4, 3, 3, 1, 1, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nhwc,
                        memory::format::oihw, memory::format::x,
                        memory::format::nhwc,
                        { 2, 1, 4, 4, 4, 6, 2, 2, 3, 3, 0, 0, 1, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionForwardBlocked, convolution_test_float,
        ::testing::Values(
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nChw8c,
                        memory::format::OIhw8i8o, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 13, 13, 32, 12, 12, 3, 3, 0, 0, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nChw8c,
                        memory::format::OIhw8i8o, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 3, 3, 32, 4, 4, 3, 3, 1, 1, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nChw8c,
                        memory::format::OIhw8i8o, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 4, 4, 32, 4, 4, 3, 3, 0, 0, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nChw8c,
                        memory::format::OIhw8i8o, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 3, 3, 32, 2, 2, 3, 3, 0, 0, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nChw8c,
                        memory::format::OIhw8i8o, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 2, 2, 32, 2, 2, 3, 3, 1, 1, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nChw8c,
                        memory::format::OIhw8i8o, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 13, 13, 48, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nChw8c,
                        memory::format::OIhw8i8o, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 13, 13, 48, 11, 11, 3, 3, 0, 0, 1, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionAlexnetForwardNCHW, convolution_test_float,
        ::testing::Values(
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nchw,
                        memory::format::oihw, memory::format::x,
                        memory::format::nchw,
                        { 2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nchw,
                        memory::format::goihw, memory::format::x,
                        memory::format::nchw,
                        { 2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nchw,
                        memory::format::oihw, memory::format::x,
                        memory::format::nchw,
                        { 2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nchw,
                        memory::format::goihw, memory::format::x,
                        memory::format::nchw,
                        { 2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nchw,
                        memory::format::goihw, memory::format::x,
                        memory::format::nchw, { 2, 2, 384, 13, 13, 256, 13, 13,
                                                      3, 3, 1, 1, 1, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionAlexnetForwardBlocked, convolution_test_float,
        ::testing::Values(
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nchw,
                        memory::format::Ohwi8o, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nhwc,
                        memory::format::Ohwi8o, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nChw8c,
                        memory::format::gOIhw8i8o, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nChw8c,
                        memory::format::OIhw8i8o, memory::format::format_undef,
                        memory::format::nChw8c,
                        { 2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nChw8c,
                        memory::format::gOIhw8i8o, memory::format::format_undef,
                        memory::format::nChw8c,
                        { 2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        convolution::direct, memory::format::nChw8c,
                        memory::format::gOIhw8i8o, memory::format::format_undef,
                        memory::format::nChw8c,
                        { 2, 2, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1,
                                1 } }));
}
