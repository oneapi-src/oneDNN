/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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
#include "math_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::math;

namespace dnnl {

template <typename data_t_src, typename data_t_wei, typename data_t_acc,
        typename data_t_dst>
void compute_ref_conv_eltwise_fwd(const test_convolution_sizes_t &c,
        const memory &src, const memory &weights, const memory &bias,
        const memory &dst, bool w_bias, algorithm elt_alg, float elt_alpha,
        float elt_beta) {
    auto src_data = map_memory<data_t_src>(src);
    auto weights_data = map_memory<data_t_wei>(weights);
    auto bias_data = w_bias ? map_memory<data_t_dst>(bias) : nullptr;
    auto dst_data = map_memory<data_t_dst>(dst);

    const memory::desc src_d = src.get_desc();
    const memory::desc weights_d = weights.get_desc();
    const memory::desc dst_d = dst.get_desc();

    auto padded_ic = src_d.get_padded_dims()[1];
    auto padded_oc = dst_d.get_padded_dims()[1];

    const dnnl::impl::memory_desc_wrapper src_mdw(src_d.get());
    const dnnl::impl::memory_desc_wrapper weights_mdw(weights_d.get());
    const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.get());

    dnnl::impl::parallel_nd(c.mb, c.ng, c.oc / c.ng, c.oh, c.ow,
            [&](memory::dim n, memory::dim g, memory::dim oc, memory::dim oh,
                    memory::dim ow) {
                memory::dim oidx = n * padded_oc * c.oh * c.ow
                        + g * padded_oc / c.ng * c.oh * c.ow + oc * c.oh * c.ow
                        + oh * c.ow + ow;

                memory::dim didx = dst_mdw.off_l(oidx, true);
                dst_data[didx] = bias_data ? bias_data[g * c.oc / c.ng + oc]
                                           : data_t_dst {0};

                for_(memory::dim ic = 0; ic < c.ic / c.ng; ic++)
                for_(memory::dim kh = 0; kh < c.kh; kh++)
                for (memory::dim kw = 0; kw < c.kw; kw++) {
                    memory::dim ih = oh * c.strh - c.padh + kh * (1 + c.dilh);
                    if (ih < 0 || ih >= c.ih) continue;
                    memory::dim iw = ow * c.strw - c.padw + kw * (1 + c.dilw);
                    if (iw < 0 || iw >= c.iw) continue;

                    memory::dim iidx = n * padded_ic * c.ih * c.iw
                            + g * padded_ic / c.ng * c.ih * c.iw
                            + ic * c.ih * c.iw + ih * c.iw + iw;
                    memory::dim widx = 0
                            + g * padded_oc / c.ng * padded_ic / c.ng * c.kh
                                    * c.kw
                            + oc * padded_ic / c.ng * c.kh * c.kw
                            + ic * c.kh * c.kw + kh * c.kw + kw;

                    dst_data[didx] += src_data[src_mdw.off_l(iidx, true)]
                            * weights_data[weights_mdw.off_l(widx, true)];
                }

                auto &d = dst_data[didx];
                switch (elt_alg) {
                    case algorithm::eltwise_relu:
                        d = relu_fwd(d, elt_alpha);
                        break;
                    case algorithm::eltwise_tanh: d = tanh_fwd(d); break;
                    case algorithm::eltwise_elu:
                        d = elu_fwd(d, elt_alpha);
                        break;
                    case algorithm::eltwise_square: d = square_fwd(d); break;
                    case algorithm::eltwise_abs: d = abs_fwd(d); break;
                    case algorithm::eltwise_linear:
                        d = linear_fwd(d, elt_alpha, elt_beta);
                        break;
                    case algorithm::eltwise_clip:
                        d = clip_fwd(d, elt_alpha, elt_beta);
                        break;
                    case algorithm::eltwise_soft_relu:
                        d = soft_relu_fwd(d, elt_alpha);
                        break;
                    case algorithm::eltwise_logistic:
                        d = logistic_fwd(d);
                        break;
                    case algorithm::eltwise_exp: d = exp_fwd(d); break;
                    case algorithm::eltwise_swish:
                        d = swish_fwd(d, elt_alpha);
                        break;
                    default: assert(!"unknown alg_kind");
                }
            });
}

template <typename data_t_src, typename data_t_wei, typename data_t_acc,
        typename data_t_dst>
class convolution_eltwise_test
    : public ::testing::TestWithParam<test_convolution_eltwise_params_t> {
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

        test_convolution_eltwise_params_t p = ::testing::TestWithParam<
                test_convolution_eltwise_params_t>::GetParam();

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

        SKIP_IF_CUDA(p.alg != algorithm::eltwise_relu
                        && p.alg != algorithm::eltwise_tanh
                        && p.alg != algorithm::eltwise_elu
                        && p.alg != algorithm::eltwise_logistic,
                "Unsupported algorithm type for CUDA");
        SKIP_IF_CUDA(p.alg == algorithm::eltwise_relu && p.eltwise_alpha != 0.0,
                "DNNL only supports relu w/ slope=0 for integers");

        SKIP_IF_HIP(p.alg != algorithm::eltwise_relu
                        && p.alg != algorithm::eltwise_tanh
                        && p.alg != algorithm::eltwise_elu
                        && p.alg != algorithm::eltwise_logistic,
                "Unsupported algorithm type for HIP");
        SKIP_IF_HIP(p.alg == algorithm::eltwise_relu || p.eltwise_alpha != 0.0,
                "DNNL only supports relu w/ slope=0 for integers");

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

    virtual void Test() {
        test_convolution_eltwise_params_t p = ::testing::TestWithParam<
                test_convolution_eltwise_params_t>::GetParam();
        ASSERT_EQ(p.aalgorithm, algorithm::convolution_direct);
        auto eng = get_test_engine();
        auto strm = stream(eng);
        float eltwise_alpha = p.eltwise_alpha;
        float eltwise_beta = p.eltwise_beta;

        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;
        memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;

        test_convolution_sizes_t cd = p.sizes;

        auto c_src_desc = create_md({cd.mb, cd.ic, cd.ih, cd.iw}, data_type_src,
                p.formats.src_format);
        auto c_weights_desc = cd.ng > 1
                ? create_md({cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw},
                        data_type_wei, p.formats.weights_format)
                : create_md({cd.oc, cd.ic, cd.kh, cd.kw}, data_type_wei,
                        p.formats.weights_format);
        auto c_dst_desc = create_md({cd.mb, cd.oc, cd.oh, cd.ow}, data_type_dst,
                p.formats.dst_format);

        auto c_src = test::make_memory(c_src_desc, eng);
        auto c_weights = test::make_memory(c_weights_desc, eng);
        auto c_dst = test::make_memory(c_dst_desc, eng);

        auto dst_ref = test::make_memory(c_dst_desc, eng);

        fill_data<data_t_src>(c_src.get_desc().get_size() / sizeof(data_t_src),
                c_src, data_t_src(0), data_t_src(1));
        check_zero_tail<data_t_src>(1, c_src);

        fill_data<data_t_wei>(
                c_weights.get_desc().get_size() / sizeof(data_t_wei), c_weights,
                data_t_wei(0), data_t_wei(1));
        check_zero_tail<data_t_wei>(1, c_weights);

        bool with_bias = p.formats.bias_format != memory::format_tag::undef;
        auto c_bias_desc = with_bias
                ? create_md({cd.oc}, data_type_dst, p.formats.bias_format)
                : create_md({0}, data_type_dst, p.formats.bias_format);
        auto c_bias = test::make_memory(c_bias_desc, eng);
        if (with_bias) {
            fill_data<data_t_dst>(
                    c_bias.get_desc().get_size() / sizeof(data_t_dst), c_bias,
                    1., true);
        }

        memory::dims strides = {cd.strh, cd.strw};
        memory::dims dilations = {cd.dilh, cd.dilw};
        memory::dims padL = {cd.padh, cd.padw};
        memory::dims padR = {cd.padh, cd.padw};
        for (int i = 0; i < 2; ++i) {
            if ((cd.ih - ((cd.kh - 1) * (cd.dilh + 1) + 1) + cd.padh + padR[0])
                                    / cd.strh
                            + 1
                    != cd.oh)
                ++padR[0];
            if ((cd.iw - ((cd.kw - 1) * (cd.dilw + 1) + 1) + cd.padw + padR[1])
                                    / cd.strw
                            + 1
                    != cd.ow)
                ++padR[1];
        }

        SKIP_IF_CUDA(cd.padh < padR[0] || cd.padw < padR[1],
                "Unsupported padding for CUDA.");

        dnnl::post_ops ops;
        ops.append_eltwise(p.alg, p.eltwise_alpha, p.eltwise_beta);

        dnnl::primitive_attr attr;
        attr.set_post_ops(ops);

        auto conv_primitive_desc = with_bias
                ? convolution_forward::primitive_desc(eng,
                        prop_kind::forward_inference, p.aalgorithm, c_src_desc,
                        c_weights_desc, c_bias_desc, c_dst_desc, strides,
                        dilations, padL, padR, attr)
                : convolution_forward::primitive_desc(eng,
                        prop_kind::forward_inference, p.aalgorithm, c_src_desc,
                        c_weights_desc, c_dst_desc, strides, dilations, padL,
                        padR, attr);

        ASSERT_EQ(conv_primitive_desc.get_algorithm(), p.aalgorithm);
        ASSERT_EQ(conv_primitive_desc.get_prop_kind(),
                prop_kind::forward_inference);
        ASSERT_EQ(conv_primitive_desc.get_strides(), strides);
        ASSERT_EQ(conv_primitive_desc.get_dilations(), dilations);
        ASSERT_EQ(conv_primitive_desc.get_padding_l(), padL);
        ASSERT_EQ(conv_primitive_desc.get_padding_r(), padR);

        EXPECT_ANY_THROW(convolution_forward(conv_primitive_desc, {}));
        convolution_forward(conv_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_SRC, c_src}, {DNNL_ARG_WEIGHTS, c_weights},
                                {DNNL_ARG_BIAS, c_bias},
                                {DNNL_ARG_DST, c_dst}});
        strm.wait();

        compute_ref_conv_eltwise_fwd<data_t_src, data_t_wei, data_t_wei,
                data_t_dst>(cd, c_src, c_weights, c_bias, dst_ref, with_bias,
                p.alg, eltwise_alpha, eltwise_beta);
        check_zero_tail<data_t_dst>(1, dst_ref);

        static constexpr data_t_dst threshold = static_cast<data_t_dst>(1e-2);
        compare_data<data_t_dst>(dst_ref, c_dst, threshold);
        check_zero_tail<data_t_dst>(0, c_dst);
    }
};

} // namespace dnnl
