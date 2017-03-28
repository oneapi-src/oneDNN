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
void compute_ref_conv_fwd(const test_convolution_sizes_t &c,
        const memory::desc &src_d,
        const memory::desc &weights_d,
        const memory::desc &bias_d,
        const memory::desc &dst_d,
        const memory &src,
        const memory &weights,
        const memory &bias,
        const memory &dst)
{
    const bool w_bias = memory::convert_to_c(memory::format::format_undef)
        != bias_d.data.format;
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *weights_data = (data_t *)weights.get_data_handle();
    data_t *bias_data = w_bias ? (data_t *)bias.get_data_handle() : nullptr;
    data_t *dst_data = (data_t *)dst.get_data_handle();

#pragma omp parallel for collapse(5) schedule(static)
    for (int n = 0; n < c.mb; n++) {
        for (int g = 0; g < c.ng; g++) {
            for (int oc = 0; oc < c.oc / c.ng; oc++) {
                for (int oh = 0; oh < c.oh; oh++) {
                    for (int ow = 0; ow < c.ow; ow++) {
                        int oidx = n * c.oc * c.oh * c.ow
                                + g * c.oc / c.ng * c.oh * c.ow
                                + oc * c.oh * c.ow + oh * c.ow + ow;
                        dst_data[map_index(dst_d, oidx)] = bias_data ?
                                bias_data[map_index(bias_d,
                                        g * c.oc / c.ng + oc)] :
                                0.0;
                        for (int ic = 0; ic < c.ic / c.ng; ic++) {
                            for (int kh = 0; kh < c.kh; kh++) {
                                for (int kw = 0; kw < c.kw; kw++) {
                                    int iw = ow * c.strw - c.padw + kw;
                                    int ih = oh * c.strh - c.padh + kh;
                                    if (iw < 0 || iw >= c.iw) continue;
                                    if (ih < 0 || ih >= c.ih) continue;
                                    int iidx = n * c.ic * c.ih * c.iw
                                            + g * c.ic / c.ng * c.ih * c.iw
                                            + ic * c.ih * c.iw + ih * c.iw + iw;
                                    int widx = g * c.oc / c.ng * c.ic
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

template <typename data_t>
class convolution_forward_test
        : public ::testing::TestWithParam<test_convolution_params_t> {
protected:
    virtual void SetUp()
    {
        test_convolution_params_t p
            = ::testing::TestWithParam<test_convolution_params_t>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aalgorithm, algorithm::convolution_direct);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        test_convolution_sizes_t cd = p.sizes;

        auto aprop_kind = prop_kind::forward;
        bool with_bias = p.formats.bias_format != memory::format::format_undef;

        auto c_src_desc = create_md({ cd.mb, cd.ic, cd.ih, cd.iw }, data_type,
                p.formats.src_format);
        auto c_weights_desc = cd.ng > 1 ?
                create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                        data_type, p.formats.weights_format) :
                create_md({ cd.oc, cd.ic, cd.kh, cd.kw },
                        data_type,p.formats.weights_format);
        auto c_dst_desc = create_md({ cd.mb, cd.oc, cd.oh, cd.ow },
                data_type, p.formats.dst_format);
        auto c_bias_desc = with_bias ?
                create_md({ cd.oc }, data_type, p.formats.bias_format) :
                create_md({}, data_type, p.formats.bias_format);

        auto src_primitive_desc = memory::primitive_desc(c_src_desc, eng);
        auto weights_primitive_desc = memory::primitive_desc(
                c_weights_desc, eng);
        auto bias_primitive_desc = memory::primitive_desc(c_bias_desc, eng);
        auto dst_primitive_desc = memory::primitive_desc(c_dst_desc, eng);

        auto dst_size = dst_primitive_desc.get_size();

        // TODO: free
        auto ref_dst_data = new data_t[dst_size];

        auto c_src = memory(src_primitive_desc);
        auto c_weights = memory(weights_primitive_desc);
        auto c_bias = memory(bias_primitive_desc);
        auto c_dst = memory(dst_primitive_desc);

        // Only true for dense format
        fill_data<data_t>(
                c_src.get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)c_src.get_data_handle());
        fill_data<data_t>(
                c_weights.get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)c_weights.get_data_handle());
        if (with_bias) {
            fill_data<data_t>(
                    c_bias.get_primitive_desc().get_size() / sizeof(data_t),
                    (data_t *)c_bias.get_data_handle());
        }

        std::vector<int> padR = { cd.padh, cd.padw };
        for (int i = 0; i < 2; ++i) {
            if ((cd.ih + cd.padh + padR[0] - cd.kh) / cd.strh + 1 != cd.oh)
                ++padR[0];
            if ((cd.iw + cd.padw + padR[1] - cd.kw) / cd.strw + 1 != cd.ow)
                ++padR[1];
        }

        auto conv_desc = with_bias ?
            convolution_forward::desc(aprop_kind, p.aalgorithm,
                c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
                { cd.strh, cd.strw }, { cd.padh, cd.padw }, padR,
                padding_kind::zero) :
            convolution_forward::desc(aprop_kind, p.aalgorithm,
                c_src_desc, c_weights_desc, c_dst_desc,
                { cd.strh, cd.strw }, { cd.padh, cd.padw }, padR,
                padding_kind::zero);

        auto conv_primitive_desc = convolution_forward::primitive_desc(
                conv_desc, eng);

        auto conv = with_bias ?
            convolution_forward(conv_primitive_desc,
                    c_src, c_weights, c_bias, c_dst) :
            convolution_forward(conv_primitive_desc,
                    c_src, c_weights, c_dst);

        std::vector<primitive> pipeline;
        pipeline.push_back(conv);
        auto s = stream(stream::kind::lazy);
        s.submit(pipeline).wait();

        auto ref_memory = memory(memory::primitive_desc(c_dst_desc, eng),
                ref_dst_data);
        compute_ref_conv_fwd<data_t>(cd, c_src_desc, c_weights_desc,
                c_bias_desc, c_dst_desc, c_src, c_weights, c_bias, ref_memory);
        compare_data<data_t>(ref_memory, c_dst);
    }
};

using convolution_test = convolution_forward_test<float>;

TEST_P(convolution_test, TestConvolution)
{
}

#define DIRECTION_FORWARD
#include "convolution_common.h"

}
