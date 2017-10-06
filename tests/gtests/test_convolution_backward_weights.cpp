/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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
void compute_ref_conv_bwd_bias(const test_convolution_sizes_t &c,
        const memory &diff_dst, const memory &diff_bias)
{
    data_t *diff_bias_data = (data_t *)diff_bias.get_data_handle();
    data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();

    const memory::desc bias_d = diff_bias.get_primitive_desc().desc();
    const memory::desc dst_d = diff_dst.get_primitive_desc().desc();

#   pragma omp parallel for collapse(2) schedule(static)
    for (int g = 0; g < c.ng; ++g) {
        for (int oc = 0; oc < c.oc / c.ng; ++oc) {
            int bidx = g * c.oc / c.ng + oc;
            diff_bias_data[map_index(bias_d, bidx)] = 0.0;
            for (int mb = 0; mb < c.mb; ++mb) {
                for (int oh = 0; oh < c.oh; ++oh) {
                    for (int ow = 0; ow < c.ow; ++ow) {
                        int oidx = mb * c.oc * c.oh * c.ow
                                + g * c.oc / c.ng * c.oh * c.ow
                                + oc * c.oh * c.ow + oh * c.ow + ow;
                        diff_bias_data[map_index(bias_d, bidx)]
                            += diff_dst_data[map_index(dst_d, oidx)];
                    }
                }
            }
        }
    }
}

template <typename data_t>
void compute_ref_conv_bwd_weights(const test_convolution_sizes_t &c,
        const memory &src, const memory &diff_dst, const memory &diff_weights)
{
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *diff_weights_data = (data_t *)diff_weights.get_data_handle();
    data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = diff_weights.get_primitive_desc().desc();
    const memory::desc dst_d = diff_dst.get_primitive_desc().desc();

#   pragma omp parallel for collapse(5) schedule(static)
    for (int g = 0; g < c.ng; ++g) {
        for (int oc = 0; oc < c.oc / c.ng; oc++) {
            for (int ic = 0; ic < c.ic / c.ng; ++ic) {
                for (int kh = 0; kh < c.kh; kh++) {
                    for (int kw = 0; kw < c.kw; kw++) {
                        int widx = g * c.oc / c.ng * c.ic / c.ng * c.kh * c.kw
                                + oc * c.ic / c.ng * c.kh * c.kw
                                + ic * c.kh * c.kw + kh * c.kw + kw;
                        diff_weights_data[map_index(weights_d, widx)] = 0.0;
                        for (int mb = 0; mb < c.mb; ++mb) {
                            for (int oh = 0; oh < c.oh; ++oh) {
                                for (int ow = 0; ow < c.ow; ++ow) {
                                    if (ow*c.strw + kw *
                                        (1 + c.dilw) < c.padw ||
                                        oh*c.strh + kh *
                                        (1 + c.dilh) < c.padh ||
                                        ow*c.strw + kw *
                                        (1 + c.dilw) >= c.iw + c.padw ||
                                        oh*c.strh + kh *
                                        (1 + c.dilh)>= c.ih + c.padh)
                                        continue;

                                    int ih = oh * c.strh - c.padh + kh
                                            * (1 + c.dilh);
                                    int iw = ow * c.strw - c.padw + kw
                                            * (1 + c.dilw);
                                    int sidx = mb * c.ic * c.ih * c.iw
                                            + g * c.ic / c.ng * c.ih * c.iw
                                            + ic * c.ih * c.iw + ih * c.iw + iw;
                                    int didx = mb * c.oc * c.oh * c.ow
                                            + g * c.oc / c.ng * c.oh * c.ow
                                            + oc * c.oh * c.ow + oh * c.ow + ow;

                                    diff_weights_data[map_index(weights_d, widx)]
                                        += src_data[map_index(src_d, sidx)]
                                        * diff_dst_data[map_index(dst_d, didx)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename data_t>
class convolution_backward_weights_test
            : public ::testing::TestWithParam<test_convolution_params_t> {
protected:
    virtual void SetUp()
    {
        test_convolution_params_t p
                = ::testing::TestWithParam<
                test_convolution_params_t>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aalgorithm, convolution_direct);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        test_convolution_sizes_t cd = p.sizes;

        auto c_src_desc = create_md({ cd.mb, cd.ic, cd.ih, cd.iw }, data_type,
                p.formats.src_format);
        auto c_diff_weights_desc = cd.ng > 1 ?
                create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                        data_type, p.formats.weights_format) :
                create_md(
                        { cd.oc, cd.ic, cd.kh, cd.kw }, data_type,
                        p.formats.weights_format);
        auto c_diff_bias_desc
                = create_md({ cd.oc }, data_type, p.formats.bias_format);
        auto c_diff_dst_desc
                = create_md({ cd.mb, cd.oc, cd.oh, cd.ow }, data_type,
                        p.formats.dst_format);

        auto c_src = memory({c_src_desc, eng});
        auto c_diff_weights = memory({c_diff_weights_desc, eng});
        auto c_diff_bias = memory({c_diff_bias_desc, eng});
        auto c_diff_dst = memory({c_diff_dst_desc, eng});

        fill_data<data_t>(c_diff_dst.get_primitive_desc().get_size()
                / sizeof(data_t), (data_t *)c_diff_dst.get_data_handle());
        fill_data<data_t>(c_src.get_primitive_desc().get_size()
                / sizeof(data_t), (data_t *)c_src.get_data_handle());

        std::vector<int> padR = { cd.padh, cd.padw };
        for (int i = 0; i < 2; ++i) {
            if ((cd.ih - ((cd.kh - 1) * (cd.dilh + 1) + 1) + cd.padh + padR[0])
                / cd.strh + 1 != cd.oh)
                ++padR[0];
            if ((cd.iw - ((cd.kw - 1) * (cd.dilw + 1) + 1) + cd.padw + padR[1])
                / cd.strw + 1 != cd.ow)
                ++padR[1];
        }

        auto conv_desc = convolution_forward::desc(prop_kind::forward_training,
                p.aalgorithm, c_src_desc, c_diff_weights_desc, c_diff_bias_desc,
                c_diff_dst_desc, { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                { cd.padh, cd.padw }, padR, padding_kind::zero);

        auto conv_bwd_weights_desc = convolution_backward_weights::desc(
                p.aalgorithm, c_src_desc, c_diff_weights_desc, c_diff_bias_desc,
                c_diff_dst_desc, { cd.strh, cd.strw }, { cd.dilh, cd.dilw },
                { cd.padh, cd.padw }, padR, padding_kind::zero);

        auto conv_primitive_desc = convolution_forward::primitive_desc(
                conv_desc, eng);

        auto conv_bwd_weights_primitive_desc =
                convolution_backward_weights::primitive_desc(
                        conv_bwd_weights_desc, eng, conv_primitive_desc);

        auto conv_bwd_weights =
                convolution_backward_weights(conv_bwd_weights_primitive_desc,
                        c_src, c_diff_dst, c_diff_weights, c_diff_bias);

        std::vector<primitive> pipeline;
        pipeline.push_back(conv_bwd_weights);
        stream(stream::kind::lazy).submit(pipeline).wait();

        auto ref_diff_weights = memory({c_diff_weights_desc, eng});
        auto ref_diff_bias = memory({c_diff_bias_desc, eng});

        compute_ref_conv_bwd_weights<data_t>(cd, c_src, c_diff_dst,
                ref_diff_weights);
        compare_data<data_t>(ref_diff_weights, c_diff_weights);

        compute_ref_conv_bwd_bias<data_t>(cd, c_diff_dst,
                ref_diff_bias);
        compare_data<data_t>(ref_diff_bias, c_diff_bias);
    }
};

using convolution_test = convolution_backward_weights_test<float>;

TEST_P(convolution_test, TestConvolution)
{
}

#define DIRECTION_BACKWARD_WEIGHTS
#include "convolution_common.h"
#include "diluted_convolution.h"

}
