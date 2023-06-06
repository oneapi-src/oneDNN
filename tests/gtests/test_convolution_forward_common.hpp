/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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

#ifndef TEST_CONVOLUTION_FORWARD_COMMON_H
#define TEST_CONVOLUTION_FORWARD_COMMON_H

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include <stdint.h>
#include "oneapi/dnnl/dnnl.hpp"

#include <math.h>

namespace dnnl {

template <typename data_t_src, typename data_t_wei, typename data_t_acc,
        typename data_t_dst>
void compute_ref_conv_fwd(const test_convolution_sizes_t &c,
        const test_convolution_attr_t &attr, const memory::desc &src_d,
        const memory::desc &weights_d, const memory::desc &bias_d,
        const memory::desc &dst_d, const memory &src, const memory &weights,
        const memory &bias, const memory &dst) {
    const bool w_bias = bias_d.get_ndims() != 0;
    auto src_data = map_memory<data_t_src>(src);
    auto weights_data = map_memory<data_t_wei>(weights);

    auto bias_data = w_bias ? map_memory<data_t_dst>(bias) : nullptr;
    auto dst_data = map_memory<data_t_dst>(dst);

    auto padded_ic = src_d.get_padded_dims()[1];
    auto padded_oc = dst_d.get_padded_dims()[1];

    const dnnl::impl::memory_desc_wrapper src_mdw(src_d.get());
    const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.get());
    const dnnl::impl::memory_desc_wrapper weights_mdw(weights_d.get());
    const dnnl::impl::memory_desc_wrapper bias_mdw(bias_d.get());

    dnnl::impl::parallel_nd(c.mb, c.ng, c.oc / c.ng, c.oh, c.ow,
            [&](memory::dim n, memory::dim g, memory::dim oc, memory::dim oh,
                    memory::dim ow) {
                data_t_acc a = 0;
                for (memory::dim ic = 0; ic < c.ic / c.ng; ic++) {
                    for (memory::dim kh = 0; kh < c.kh; kh++) {
                        for (memory::dim kw = 0; kw < c.kw; kw++) {
                            memory::dim iw
                                    = ow * c.strw - c.padw + kw * (1 + c.dilw);
                            memory::dim ih
                                    = oh * c.strh - c.padh + kh * (1 + c.dilh);
                            if (iw < 0 || iw >= c.iw) continue;
                            if (ih < 0 || ih >= c.ih) continue;
                            memory::dim iidx = n * padded_ic * c.ih * c.iw
                                    + g * padded_ic / c.ng * c.ih * c.iw
                                    + ic * c.ih * c.iw + ih * c.iw + iw;
                            memory::dim widx = g * padded_oc / c.ng * padded_ic
                                            / c.ng * c.kh * c.kw
                                    + oc * padded_ic / c.ng * c.kh * c.kw
                                    + ic * c.kh * c.kw + kh * c.kw + kw;
                            a += ((data_t_acc)src_data[src_mdw.off_l(
                                         iidx, true)])
                                    * weights_data[weights_mdw.off_l(
                                            widx, true)];
                        }
                    }
                }

                float a_fp = (float)a;

                if (attr.src_scale.is_def()) {
                    const auto &s = attr.src_scale;
                    using P = test_convolution_attr_t::scale_t;
                    if (s.policy == P::policy_t::COMMON) { a_fp *= s.scale; }
                }

                if (attr.wei_scale.is_def()) {
                    const auto &s = attr.wei_scale;
                    using P = test_convolution_attr_t::scale_t;
                    if (s.policy == P::policy_t::COMMON) { a_fp *= s.scale; }
                }

                a_fp += (float)(bias_data ? bias_data[bias_mdw.off_l(
                                        g * c.oc / c.ng + oc, true)]
                                          : 0);

                if (attr.dst_scale.is_def()) {
                    const auto &s = attr.dst_scale;
                    using P = test_convolution_attr_t::scale_t;
                    if (s.policy == P::policy_t::COMMON) { a_fp /= s.scale; }
                }

                a_fp = out_round<data_t_dst>(a_fp);

                memory::dim oidx = n * padded_oc * c.oh * c.ow
                        + g * padded_oc / c.ng * c.oh * c.ow + oc * c.oh * c.ow
                        + oh * c.ow + ow;
                dst_data[dst_mdw.off_l(oidx, true)] = (data_t_dst)a_fp;
            });
}

template <typename data_t_src, typename data_t_wei, typename data_t_acc,
        typename data_t_dst>
class convolution_forward_test
    : public ::testing::TestWithParam<test_convolution_params_t> {
protected:
    virtual void SetUp() {
        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;
        memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;

        SKIP_IF(unsupported_data_type(data_type_src),
                "Engine does not support this data type.");
        SKIP_IF(unsupported_data_type(data_type_dst),
                "Engine does not support this data type.");
        SKIP_IF(unsupported_data_type(data_type_wei),
                "Engine does not support this data type.");

        auto p = ::testing::TestWithParam<
                test_convolution_params_t>::GetParam();

        SKIP_IF_CUDA(
                !(cuda_check_format_tags(p.formats.src_format, data_type_src)
                        && cuda_check_format_tags(
                                p.formats.dst_format, data_type_dst)
                        && (cuda_check_format_tags(
                                    p.formats.weights_format, data_type_wei)
                                || impl::utils::one_of(p.formats.weights_format,
                                        /* weights formats */
                                        memory::format_tag::gowi,
                                        memory::format_tag::gohwi,
                                        memory::format_tag::godhwi,
                                        memory::format_tag::owi,
                                        memory::format_tag::ohwi,
                                        memory::format_tag::odhwi))),
                "Format is not supported.");

        SKIP_IF_HIP(
                !(hip_check_format_tags(p.formats.src_format, data_type_src)
                        && hip_check_format_tags(
                                p.formats.dst_format, data_type_dst)
                        && (hip_check_format_tags(
                                    p.formats.weights_format, data_type_wei)
                                || impl::utils::one_of(p.formats.weights_format,
                                        /* weights formats */
                                        memory::format_tag::gowi,
                                        memory::format_tag::gohwi,
                                        memory::format_tag::godhwi,
                                        memory::format_tag::owi,
                                        memory::format_tag::ohwi,
                                        memory::format_tag::odhwi))),
                "Format is not supported.");

        catch_expected_failures(
                [&]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    bool cuda_check_format_tags(memory::format_tag tag, memory::data_type dt) {
        return ((impl::utils::one_of(tag, memory::format_tag::ab,
                        memory::format_tag::abc, memory::format_tag::abcd,
                        memory::format_tag::abcde, memory::format_tag::abcdef,
                        memory::format_tag::acb, memory::format_tag::acdb,
                        memory::format_tag::acdeb))
                || (dt == memory::data_type::s8
                        && impl::utils::one_of(tag, memory::format_tag::aBcd4b,
                                memory::format_tag::aBcde4b)));
    }

    bool hip_check_format_tags(memory::format_tag tag, memory::data_type dt) {
        return ((impl::utils::one_of(tag, memory::format_tag::ab,
                        memory::format_tag::abc, memory::format_tag::abcd,
                        memory::format_tag::abcde, memory::format_tag::abcdef))
                || (dt == memory::data_type::s8
                        && impl::utils::one_of(tag, memory::format_tag::aBcd4b,
                                memory::format_tag::aBcde4b)));
    }

    void Test() {
        auto p = ::testing::TestWithParam<
                test_convolution_params_t>::GetParam();
        ASSERT_EQ(p.aalgorithm, algorithm::convolution_direct);
        auto eng = get_test_engine();
        auto strm = stream(eng);

        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;
        memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;

        test_convolution_sizes_t cd = p.sizes;

        test_convolution_attr_t attr = p.attr;
        attr.dnnl_attr_recreate();

        auto aprop_kind = prop_kind::forward;
        bool with_bias = p.formats.bias_format != memory::format_tag::undef;
        bool with_src_scales = attr.src_scale.is_def();
        bool with_wei_scales = attr.wei_scale.is_def();
        bool with_dst_scales = attr.dst_scale.is_def();

        auto c_src_desc = create_md({cd.mb, cd.ic, cd.ih, cd.iw}, data_type_src,
                p.formats.src_format);
        auto c_weights_desc = cd.ng > 1
                ? create_md({cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw},
                        data_type_wei, p.formats.weights_format)
                : create_md({cd.oc, cd.ic, cd.kh, cd.kw}, data_type_wei,
                        p.formats.weights_format);
        auto c_dst_desc = create_md({cd.mb, cd.oc, cd.oh, cd.ow}, data_type_dst,
                p.formats.dst_format);
        auto c_bias_desc = with_bias
                ? create_md({cd.oc}, data_type_dst, p.formats.bias_format)
                : create_md({}, data_type_dst, p.formats.bias_format);
        auto c_src_scales_desc = with_src_scales
                ? create_md({1}, memory::data_type::f32, memory::format_tag::x)
                : create_md({}, memory::data_type::f32, memory::format_tag::x);
        auto c_wei_scales_desc = with_wei_scales
                ? create_md({1}, memory::data_type::f32, memory::format_tag::x)
                : create_md({}, memory::data_type::f32, memory::format_tag::x);
        auto c_dst_scales_desc = with_dst_scales
                ? create_md({1}, memory::data_type::f32, memory::format_tag::x)
                : create_md({}, memory::data_type::f32, memory::format_tag::x);

        auto c_src = test_memory(c_src_desc, eng);
        auto c_weights = test_memory(c_weights_desc, eng);
        auto c_bias = test_memory(c_bias_desc, eng);
        auto c_dst = test_memory(c_dst_desc, eng);
        auto c_src_scales = test_memory(c_src_scales_desc, eng);
        auto c_wei_scales = test_memory(c_wei_scales_desc, eng);
        auto c_dst_scales = test_memory(c_dst_scales_desc, eng);

        // Only true for dense format
        fill_data<data_t_dst>(
                c_dst.get_size() / sizeof(data_t_dst), c_dst.get());
        fill_data<data_t_src>(
                c_src.get_size() / sizeof(data_t_src), c_src.get());
        fill_data<data_t_wei>(
                c_weights.get_size() / sizeof(data_t_wei), c_weights.get());
        if (with_bias) {
            fill_data<data_t_dst>(
                    c_bias.get_size() / sizeof(data_t_dst), c_bias.get());
        }
        if (with_src_scales) {
            fill_data<float>(c_src_scales.get_size() / sizeof(float),
                    c_src_scales.get(), attr.src_scale.scale, 0.0f);
        }
        if (with_wei_scales) {
            fill_data<float>(c_wei_scales.get_size() / sizeof(float),
                    c_wei_scales.get(), attr.wei_scale.scale, 0.0f);
        }
        if (with_dst_scales) {
            fill_data<float>(c_dst_scales.get_size() / sizeof(float),
                    c_dst_scales.get(), attr.dst_scale.scale, 0.0f);
        }

        check_zero_tail<data_t_src>(1, c_src.get());
        check_zero_tail<data_t_wei>(1, c_weights.get());
        check_zero_tail<data_t_dst>(1, c_dst.get());

        memory::dims strides = {cd.strh, cd.strw};
        memory::dims dilations = {cd.dilh, cd.dilw};
        memory::dims padL = {cd.padh, cd.padw};
        memory::dims padR = {
                right_padding(cd.ih, cd.oh, cd.kh, cd.padh, cd.strh, cd.dilh),
                right_padding(cd.iw, cd.ow, cd.kw, cd.padw, cd.strw, cd.dilw)};

        auto conv_primitive_desc = with_bias
                ? convolution_forward::primitive_desc(eng, aprop_kind,
                        p.aalgorithm, c_src_desc, c_weights_desc, c_bias_desc,
                        c_dst_desc, strides, dilations, padL, padR,
                        attr.dnnl_attr)
                : convolution_forward::primitive_desc(eng, aprop_kind,
                        p.aalgorithm, c_src_desc, c_weights_desc, c_dst_desc,
                        strides, dilations, padL, padR, attr.dnnl_attr);

        conv_primitive_desc = convolution_forward::primitive_desc(
                conv_primitive_desc.get()); // test construction from a C pd

        ASSERT_TRUE(
                conv_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == conv_primitive_desc.src_desc());
        ASSERT_TRUE(
                conv_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == conv_primitive_desc.dst_desc());
        ASSERT_TRUE(conv_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_WEIGHTS)
                == conv_primitive_desc.weights_desc());
        ASSERT_TRUE(
                conv_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_BIAS)
                == conv_primitive_desc.bias_desc());

        ASSERT_EQ(conv_primitive_desc.get_algorithm(), p.aalgorithm);
        ASSERT_EQ(conv_primitive_desc.get_prop_kind(), aprop_kind);
        ASSERT_EQ(conv_primitive_desc.get_strides(), strides);
        ASSERT_EQ(conv_primitive_desc.get_dilations(), dilations);
        ASSERT_EQ(conv_primitive_desc.get_padding_l(), padL);
        ASSERT_EQ(conv_primitive_desc.get_padding_r(), padR);

        EXPECT_ANY_THROW(convolution_forward(conv_primitive_desc, {}));
        convolution_forward(conv_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_SRC, c_src.get()},
                                {DNNL_ARG_WEIGHTS, c_weights.get()},
                                {DNNL_ARG_BIAS, c_bias.get()},
                                {DNNL_ARG_DST, c_dst.get()},
                                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                                        c_src_scales.get()},
                                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                                        c_wei_scales.get()},
                                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                                        c_dst_scales.get()}});
        strm.wait();

        auto ref_memory = test::make_memory(c_dst_desc, eng);
        compute_ref_conv_fwd<data_t_src, data_t_wei, data_t_acc, data_t_dst>(cd,
                attr, c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
                c_src.get(), c_weights.get(), c_bias.get(), ref_memory);
        check_zero_tail<data_t_dst>(1, ref_memory);

        compare_data<data_t_dst>(ref_memory, c_dst.get());
        check_zero_tail<data_t_dst>(0, c_dst.get());
    }
};

} // namespace dnnl
#endif
