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
void compute_ref_conv_bwd_bias(test_convolution_descr_t c, memory bias,
        memory dst)
{
    data_t *bias_data = (data_t *)bias.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();

    const memory::desc bias_d = bias.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

#   pragma omp parallel for collapse(2) schedule(static)
    for (int g = 0; g < c.ng; ++g) {
        for (int oc = 0; oc < c.oc / c.ng; ++oc) {
            int bidx = g * c.oc / c.ng + oc;
            bias_data[map_index(bias_d, bidx)] = 0.0;
            for (int mb = 0; mb < c.mb; ++mb) {
                for (int oh = 0; oh < c.oh; ++oh) {
                    for (int ow = 0; ow < c.ow; ++ow) {
                        int oidx = mb * c.oc * c.oh * c.ow
                                + g * c.oc / c.ng * c.oh * c.ow
                                + oc * c.oh * c.ow + oh * c.ow + ow;
                        bias_data[map_index(bias_d, bidx)]
                            += dst_data[map_index(dst_d, oidx)];
                    }
                }
            }
        }
    }
}

struct conv_bwd_bias_test_params {
    const engine::kind engine_kind;
    convolution::algorithm aalgorithm;
    memory::format bias_format;
    memory::format dst_format;
    test_convolution_descr_t test_cd;
};

template <typename data_t>
class convolution_backward_bias_test :
    public ::testing::TestWithParam<conv_bwd_bias_test_params> {
protected:
    virtual void SetUp()
    {
        conv_bwd_bias_test_params p
                = ::testing::TestWithParam<
                conv_bwd_bias_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu
                || p.engine_kind == engine::kind::cpu_lazy);
        ASSERT_EQ(p.aalgorithm, convolution::direct);
        auto eng = engine(p.engine_kind, 0);
        memory::precision prec = data_traits<data_t>::prec;
        ASSERT_EQ(prec, mkldnn::memory::precision::f32);

        test_convolution_descr_t cd = p.test_cd;
        auto aprop_kind = prop_kind::backward_bias;

        auto c_dst_desc
                = create_md({ cd.mb, cd.oc, cd.oh, cd.ow }, prec, p.dst_format);
        auto c_bias_desc = create_md({ cd.oc }, prec, p.bias_format);

        auto c_dst = memory(memory::primitive_desc(c_dst_desc, eng));
        auto c_bias = memory(memory::primitive_desc(c_bias_desc, eng));

        fill_data<data_t>(c_dst.get_primitive_desc().get_number_of_elements(),
                (data_t *)c_dst.get_data_handle());

        std::vector<primitive> pipeline;
        pipeline.push_back(convolution(aprop_kind, p.aalgorithm, c_bias,
                    c_dst, { cd.strh, cd.strw }, { cd.padh, cd.padw },
                padding_kind::zero));
        stream().submit(pipeline).wait();

        auto ref_memory = memory(memory::primitive_desc(c_bias_desc, eng));
        compute_ref_conv_bwd_bias<data_t>(cd, ref_memory, c_dst);
        compare_data<data_t>(ref_memory, c_bias);
    }
};

using convolution_backward_bias_test_float
    = convolution_backward_bias_test<float>;
using conv_bwd_bias_test_params_float = conv_bwd_bias_test_params;

TEST_P(convolution_backward_bias_test_float, TestsConvolution)
{
}

INSTANTIATE_TEST_CASE_P(
        TestConvolutionBackwardBias,
        convolution_backward_bias_test_float,
        ::testing::Values(
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nchw,
                        { 2, 1, 4, 4, 4, 6, 4, 4, 3, 3, 1, 1, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nchw,
                        { 2, 1, 4, 4, 4, 6, 2, 2, 3, 3, 0, 0, 1, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionBackwardBiasNHWC,
        convolution_backward_bias_test_float,
        ::testing::Values(
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nhwc,
                        { 2, 1, 4, 4, 4, 6, 4, 4, 3, 3, 1, 1, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nhwc,
                        { 2, 1, 4, 4, 4, 6, 2, 2, 3, 3, 0, 0, 1, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionBackwardBiasBlocked,
        convolution_backward_bias_test_float,
        ::testing::Values(
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 13, 13, 32, 12, 12, 3, 3, 0, 0, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 3, 3, 32, 4, 4, 3, 3, 1, 1, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 4, 4, 32, 4, 4, 3, 3, 0, 0, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 3, 3, 32, 2, 2, 3, 3, 0, 0, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 2, 2, 32, 2, 2, 3, 3, 1, 1, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 13, 13, 48, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 32, 13, 13, 48, 11, 11, 3, 3, 0, 0, 1, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionAlexnetBackwardBiasNCHW,
        convolution_backward_bias_test_float,
        ::testing::Values(
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nchw,
                        { 2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nchw,
                        { 2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nchw,
                        { 2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nchw,
                        { 2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nchw,
                        { 2, 2, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1,
                            1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionAlexnetBackwardBiasBlocked,
        convolution_backward_bias_test_float,
        ::testing::Values(
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1 } },
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 2, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1,
                                1 } }));

INSTANTIATE_TEST_CASE_P(
        TestConvolutionCifar10BackwardBiasBlocked,
        convolution_backward_bias_test_float,
        ::testing::Values(
                conv_bwd_bias_test_params_float{ engine::kind::cpu,
                        convolution::direct, memory::format::x,
                        memory::format::nChw8c,
                        { 2, 1, 3, 32, 32, 32, 32, 32, 5, 5, 2, 2, 1, 1 } }));
}
