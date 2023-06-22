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

#include <memory>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
namespace dnnl {
using fmt = memory::format_tag;
struct deconvolution_test_params_t {
    dnnl::algorithm aalgorithm;
    test_convolution_formats_t formats;
    test_convolution_attr_t attr;
    test_convolution_sizes_t sizes;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};
template <typename data_t>
void compute_bias_fwd(const test_convolution_sizes_t &c,
        const dnnl::memory &dst, const dnnl::memory &bias) {
    auto bias_data = map_memory<data_t>(bias);
    auto dst_data = map_memory<data_t>(dst);

    const memory::desc bias_d = bias.get_desc();
    const memory::desc dst_d = dst.get_desc();
    const dnnl::impl::memory_desc_wrapper bias_mdw(bias_d.get());
    const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.get());

    dnnl::impl::parallel_nd(c.mb, c.ng, c.oc / c.ng, c.oh, c.ow,
            [&](memory::dim n, memory::dim g, memory::dim oc, memory::dim oh,
                    memory::dim ow) {
                data_t b
                        = bias_data[bias_mdw.off_l(g * c.oc / c.ng + oc, true)];
                memory::dim oidx = n * c.oc * c.oh * c.ow
                        + g * c.oc / c.ng * c.oh * c.ow + oc * c.oh * c.ow
                        + oh * c.ow + ow;
                dst_data[dst_mdw.off_l(oidx, true)] += b;
            });
}

template <typename data_t>
void compute_bias_bwd(const test_convolution_sizes_t &c,
        const dnnl::memory &dst, const dnnl::memory &bias) {
    auto bias_data = map_memory<data_t>(bias);
    auto dst_data = map_memory<data_t>(dst);

    const memory::desc bias_d = bias.get_desc();
    const memory::desc dst_d = dst.get_desc();
    const dnnl::impl::memory_desc_wrapper bias_mdw(bias_d.get());
    const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.get());

    dnnl::impl::parallel_nd(
            c.ng, c.oc / c.ng, [&](memory::dim g, memory::dim oc) {
                memory::dim bidx = g * c.oc / c.ng + oc;
                bias_data[bias_mdw.off_l(bidx, true)] = 0.0;
                for_(memory::dim mb = 0; mb < c.mb; ++mb)
                for_(memory::dim oh = 0; oh < c.oh; ++oh)
                for (memory::dim ow = 0; ow < c.ow; ++ow) {
                    memory::dim oidx = mb * c.oc * c.oh * c.ow
                            + g * c.oc / c.ng * c.oh * c.ow + oc * c.oh * c.ow
                            + oh * c.ow + ow;
                    bias_data[bias_mdw.off_l(bidx, true)]
                            += dst_data[dst_mdw.off_l(oidx, true)];
                }
            });
}

template <typename data_t>
void transpose_wei(const test_convolution_sizes_t &c,
        const dnnl::memory &weights, const dnnl::memory &weights_tr) {

    auto weights_data = map_memory<data_t>(weights);
    const memory::desc weights_d = weights.get_desc();
    const dnnl::impl::memory_desc_wrapper weights_mdw(weights_d.get());
    auto weights_tr_data = map_memory<data_t>(weights_tr);
    const memory::desc weights_tr_d = weights_tr.get_desc();
    const dnnl::impl::memory_desc_wrapper weights_tr_mdw(weights_tr_d.get());

    dnnl::impl::parallel_nd(c.ng, c.oc / c.ng, c.ic / c.ng, c.kh, c.kw,
            [&](memory::dim g, memory::dim oc, memory::dim ic, memory::dim kh,
                    memory::dim kw) {
                memory::dim widx = g * c.oc / c.ng * c.ic / c.ng * c.kh * c.kw
                        + oc * c.ic / c.ng * c.kh * c.kw + ic * c.kh * c.kw
                        + kh * c.kw + kw;
                memory::dim widx_tr
                        = g * c.oc / c.ng * c.ic / c.ng * c.kh * c.kw
                        + ic * c.oc / c.ng * c.kh * c.kw + oc * c.kh * c.kw
                        + kh * c.kw + kw;
                weights_tr_data[weights_tr_mdw.off_l(widx_tr, true)]
                        = weights_data[weights_mdw.off_l(widx, true)];
            });
}

template <typename data_t>
class deconvolution_test_t
    : public ::testing::TestWithParam<deconvolution_test_params_t> {
private:
    std::shared_ptr<test_memory> src;
    std::shared_ptr<test_memory> weights;
    std::shared_ptr<test_memory> dst;
    std::shared_ptr<test_memory> bias;

    std::shared_ptr<memory::desc> dec_src_desc;
    std::shared_ptr<memory::desc> dec_weights_desc;
    std::shared_ptr<memory::desc> dec_bias_desc;
    std::shared_ptr<memory::desc> dec_dst_desc;

    std::shared_ptr<memory::desc> con_src_desc;
    std::shared_ptr<memory::desc> con_bias_desc;
    std::shared_ptr<memory::desc> con_dst_desc;
    std::shared_ptr<memory::desc> con_weights_desc;

    engine eng;
    stream strm;
    bool with_bias;
    memory::dims padL;
    memory::dims padR;
    memory::dims strides;

protected:
    void SetUp() override {
        memory::data_type data_type = data_traits<data_t>::data_type;
        SKIP_IF(unsupported_data_type(data_type),
                "Engine does not support this data type.");

        auto p = ::testing::TestWithParam<
                deconvolution_test_params_t>::GetParam();

        SKIP_IF_CUDA(
                !(cuda_check_format_tags(p.formats.src_format, data_type)
                        && cuda_check_format_tags(
                                p.formats.dst_format, data_type)
                        && cuda_check_src_wei_format_tags(p.formats.src_format,
                                p.formats.weights_format, p.sizes.ng > 1)),
                "Format is not supported.");

        SKIP_IF_HIP(
                !(hip_check_format_tags(p.formats.src_format)
                        && hip_check_format_tags(p.formats.dst_format)
                        && hip_check_src_wei_dst_format_tags(
                                p.formats.src_format, p.formats.weights_format,
                                p.formats.dst_format, p.sizes.ng > 1)),
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

    bool hip_check_format_tags(memory::format_tag tag) {
        return impl::utils::one_of(tag, memory::format_tag::ab,
                memory::format_tag::abc, memory::format_tag::abcd,
                memory::format_tag::abcde, memory::format_tag::abcdef);
    }

    bool cuda_check_src_wei_format_tags(
            memory::format_tag src, memory::format_tag wei, bool is_grouped) {
        if (src == memory::format_tag::abcd) return true;
        if (src == memory::format_tag::acdb)
            return wei
                    != (is_grouped ? memory::format_tag::abcde
                                   : memory::format_tag::abcd);
        return false;
    }

    bool hip_check_src_wei_dst_format_tags(memory::format_tag src,
            memory::format_tag wei, memory::format_tag dst, bool is_grouped) {
        if (src == memory::format_tag::abcd) {
            return (src == dst)
                    && (is_grouped ? (wei == memory::format_tag::acbde)
                                   : (wei == memory::format_tag::bacd));
        }
        if (src == memory::format_tag::acdb) {
            return (src == dst)
                    && (is_grouped ? (wei == memory::format_tag::acdeb)
                                   : (wei == memory::format_tag::bcda));
        }
        return false;
    }

    void Test() {
        auto p = ::testing::TestWithParam<
                deconvolution_test_params_t>::GetParam();

        eng = get_test_engine();
        strm = make_stream(eng);

        ASSERT_EQ(p.aalgorithm, algorithm::deconvolution_direct);
        memory::data_type data_type = data_traits<data_t>::data_type;

        test_convolution_sizes_t dd = p.sizes;
        with_bias = p.formats.bias_format != memory::format_tag::undef;

        memory::dims src_dims = {dd.mb, dd.ic, dd.ih, dd.iw};
        memory::dims dst_dims = {dd.mb, dd.oc, dd.oh, dd.ow};
        memory::dims weights_dims, c_weights_dims;
        if (dd.ng > 1) {
            weights_dims = {dd.ng, dd.oc / dd.ng, dd.ic / dd.ng, dd.kh, dd.kw};
            c_weights_dims
                    = {dd.ng, dd.ic / dd.ng, dd.oc / dd.ng, dd.kh, dd.kw};
        } else {
            weights_dims = {dd.oc, dd.ic, dd.kh, dd.kw};
            c_weights_dims = {dd.ic, dd.oc, dd.kh, dd.kw};
        }
        memory::dims bias_dims;
        if (with_bias)
            bias_dims = {dd.oc};
        else
            bias_dims = {};

        dec_src_desc = std::make_shared<memory::desc>(
                src_dims, data_type, p.formats.src_format);
        dec_dst_desc = std::make_shared<memory::desc>(
                dst_dims, data_type, p.formats.src_format);
        dec_weights_desc = std::make_shared<memory::desc>(
                weights_dims, data_type, p.formats.weights_format);
        dec_bias_desc = std::make_shared<memory::desc>(
                bias_dims, data_type, p.formats.bias_format);

        con_src_desc = std::make_shared<memory::desc>(
                dst_dims, data_type, p.formats.src_format);
        con_dst_desc = std::make_shared<memory::desc>(
                src_dims, data_type, p.formats.src_format);
        con_weights_desc = std::make_shared<memory::desc>(
                c_weights_dims, data_type, p.formats.weights_format);

        src = std::make_shared<test_memory>(*dec_src_desc, eng);
        weights = std::make_shared<test_memory>(*dec_weights_desc, eng);
        bias = std::make_shared<test_memory>(*dec_bias_desc, eng);
        dst = std::make_shared<test_memory>(*dec_dst_desc, eng);

        strides = {dd.strh, dd.strw};
        padL = {dd.padh, dd.padw};
        padR = {right_padding(dd.oh, dd.ih, dd.kh, dd.padh, dd.strh, dd.dilh),
                right_padding(dd.ow, dd.iw, dd.kw, dd.padw, dd.strw, dd.dilw)};
        SKIP_IF_CUDA(p.sizes.padh < padR[0] || p.sizes.padw < padR[1],
                "Padding not supported");
        SKIP_IF_HIP(p.sizes.padh < padR[0] || p.sizes.padw < padR[1],
                "Padding not supported");
        Forward();
        BackwardData();
        BackwardWeights();
    }

    void Forward() {
        auto aprop_kind = prop_kind::forward;
        deconvolution_test_params_t p = ::testing::TestWithParam<
                deconvolution_test_params_t>::GetParam();

        // deconvolution specific types and values
        using pd_t = deconvolution_forward::primitive_desc;

        auto conv_src = test_memory(*con_src_desc, eng);
        auto conv_dst = src;
        test_convolution_sizes_t dd = p.sizes;

        fill_data<data_t>(src->get_size() / sizeof(data_t), src->get());

        fill_data<data_t>(weights->get_size() / sizeof(data_t), weights->get());
        if (with_bias) {
            fill_data<data_t>(bias->get_size() / sizeof(data_t), bias->get());
        }

        auto weights_tr = test::make_memory(*con_weights_desc, eng);
        transpose_wei<data_t>(dd, weights->get(), weights_tr);
        auto deconv_primitive_desc = with_bias
                ? pd_t(eng, aprop_kind, algorithm::deconvolution_direct,
                        *dec_src_desc, *dec_weights_desc, *dec_bias_desc,
                        *dec_dst_desc, strides, padL, padR)
                : pd_t(eng, aprop_kind, algorithm::deconvolution_direct,
                        *dec_src_desc, *dec_weights_desc, *dec_dst_desc,
                        strides, padL, padR);

        auto aa = allows_attr_t {false};
        aa.po_binary = !is_nvidia_gpu(eng) && !is_amd_gpu(eng);
        aa.po_eltwise = !is_nvidia_gpu(eng) && !is_amd_gpu(eng);
        aa.po_prelu = !is_nvidia_gpu(eng) && !is_amd_gpu(eng);
        aa.po_sum = !is_nvidia_gpu(eng) && !is_amd_gpu(eng);

        bool is_int8 = impl::utils::one_of(dec_src_desc->get_data_type(),
                memory::data_type::s8, memory::data_type::u8);
        if (is_int8) {
            aa.scales = true;
            aa.zp = true;
        }
        if (with_bias)
            test_fwd_pd_constructors<pd_t>(deconv_primitive_desc, aa,
                    aprop_kind, algorithm::deconvolution_direct, *dec_src_desc,
                    *dec_weights_desc, *dec_bias_desc, *dec_dst_desc, strides,
                    padL, padR);
        else
            test_fwd_pd_constructors<pd_t>(deconv_primitive_desc, aa,
                    aprop_kind, algorithm::deconvolution_direct, *dec_src_desc,
                    *dec_weights_desc, *dec_dst_desc, strides, padL, padR);

        ASSERT_TRUE(
                deconv_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == deconv_primitive_desc.src_desc());
        ASSERT_TRUE(
                deconv_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == deconv_primitive_desc.dst_desc());
        ASSERT_TRUE(deconv_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_WEIGHTS)
                == deconv_primitive_desc.weights_desc());
        ASSERT_TRUE(deconv_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_BIAS)
                == deconv_primitive_desc.bias_desc());

        ASSERT_EQ(deconv_primitive_desc.get_algorithm(),
                algorithm::deconvolution_direct);
        ASSERT_EQ(deconv_primitive_desc.get_prop_kind(), aprop_kind);
        ASSERT_EQ(deconv_primitive_desc.get_strides(), strides);
        ASSERT_EQ(deconv_primitive_desc.get_padding_l(), padL);
        ASSERT_EQ(deconv_primitive_desc.get_padding_r(), padR);

        EXPECT_ANY_THROW(deconvolution_forward(deconv_primitive_desc, {}));
        deconvolution_forward(deconv_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_SRC, src->get()},
                                {DNNL_ARG_WEIGHTS, weights->get()},
                                {DNNL_ARG_BIAS, bias->get()},
                                {DNNL_ARG_DST, dst->get()}});
        strm.wait();

        auto conv_primitive_desc = convolution_forward::primitive_desc(eng,
                prop_kind::forward_training, algorithm::convolution_direct,
                *con_src_desc, *con_weights_desc, *con_dst_desc, strides, padL,
                padR);

        auto conv_bwd_data_primitive_desc
                = convolution_backward_data::primitive_desc(eng,
                        algorithm::convolution_direct, *con_src_desc,
                        *con_weights_desc, *con_dst_desc, strides, padL, padR,
                        conv_primitive_desc);

        convolution_backward_data(conv_bwd_data_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, conv_dst->get()},
                                {DNNL_ARG_WEIGHTS, weights_tr},
                                {DNNL_ARG_DIFF_SRC, conv_src.get()}});
        strm.wait();

        if (with_bias)
            compute_bias_fwd<data_t>(dd, conv_src.get(), bias->get());
        compare_data<data_t>(conv_src.get(), dst->get());
    }

    void BackwardData() {
        auto p = ::testing::TestWithParam<
                deconvolution_test_params_t>::GetParam();
        // deconv specific types and values
        using pd_t = deconvolution_backward_data::primitive_desc;
        using hint_pd_t = deconvolution_forward::primitive_desc;

        auto conv_src = dst;
        auto conv_dst = test_memory(*con_dst_desc, eng);
        test_convolution_sizes_t dd = p.sizes;

        fill_data<data_t>(weights->get_size() / sizeof(data_t), weights->get());

        fill_data<data_t>(dst->get_size() / sizeof(data_t), dst->get());

        auto weights_tr = test::make_memory(*con_weights_desc, eng);
        transpose_wei<data_t>(dd, weights->get(), weights_tr);

        auto deconv_primitive_desc = hint_pd_t(eng, prop_kind::forward_training,
                algorithm::deconvolution_direct, *dec_src_desc,
                *dec_weights_desc, *dec_dst_desc, strides, padL, padR);

        auto deconv_bwd_data_primitive_desc
                = pd_t(eng, algorithm::deconvolution_direct, *dec_src_desc,
                        *dec_weights_desc, *dec_dst_desc, strides, padL, padR,
                        deconv_primitive_desc);

        auto aa = allows_attr_t {false};
        test_bwd_pd_constructors<pd_t, hint_pd_t>(
                deconv_bwd_data_primitive_desc, deconv_primitive_desc, aa,
                algorithm::deconvolution_direct, *dec_src_desc,
                *dec_weights_desc, *dec_dst_desc, strides, padL, padR);

        ASSERT_TRUE(deconv_bwd_data_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == deconv_bwd_data_primitive_desc.diff_src_desc());
        ASSERT_TRUE(deconv_bwd_data_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == deconv_bwd_data_primitive_desc.diff_dst_desc());
        ASSERT_TRUE(deconv_bwd_data_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_WEIGHTS)
                == deconv_bwd_data_primitive_desc.weights_desc());

        ASSERT_EQ(deconv_bwd_data_primitive_desc.get_algorithm(),
                algorithm::deconvolution_direct);
        ASSERT_EQ(deconv_bwd_data_primitive_desc.get_prop_kind(),
                prop_kind::backward_data);
        ASSERT_EQ(deconv_bwd_data_primitive_desc.get_strides(), strides);
        ASSERT_EQ(deconv_bwd_data_primitive_desc.get_padding_l(), padL);
        ASSERT_EQ(deconv_bwd_data_primitive_desc.get_padding_r(), padR);

        deconvolution_backward_data(deconv_bwd_data_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, dst->get()},
                                {DNNL_ARG_WEIGHTS, weights->get()},
                                {DNNL_ARG_DIFF_SRC, src->get()}});
        strm.wait();

        auto conv_primitive_desc = convolution_forward::primitive_desc(eng,
                prop_kind::forward_training, algorithm::convolution_direct,
                *con_src_desc, *con_weights_desc, *con_dst_desc, strides, padL,
                padR);

        convolution_forward(conv_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_SRC, conv_src->get()},
                                {DNNL_ARG_WEIGHTS, weights_tr},
                                {DNNL_ARG_DST, conv_dst.get()}});
        strm.wait();

        compare_data<data_t>(conv_dst.get(), src->get());
    }

    void BackwardWeights() {
        auto p = ::testing::TestWithParam<
                deconvolution_test_params_t>::GetParam();

        // deconv specific types and values
        using pd_t = deconvolution_backward_weights::primitive_desc;
        using hint_pd_t = deconvolution_forward::primitive_desc;

        auto conv_src = dst;
        auto conv_dst = src;
        auto conv_weights = test::make_memory(*con_weights_desc, eng);
        test_convolution_sizes_t dd = p.sizes;

        fill_data<data_t>(src->get_size() / sizeof(data_t), src->get());

        fill_data<data_t>(dst->get_size() / sizeof(data_t), dst->get());

        auto deconv_primitive_desc = hint_pd_t(eng, prop_kind::forward_training,
                algorithm::deconvolution_direct, *dec_src_desc,
                *dec_weights_desc, *dec_bias_desc, *dec_dst_desc,
                {dd.strh, dd.strw}, {dd.padh, dd.padw}, padR);

        auto deconv_bwd_weights_primitive_desc
                = pd_t(eng, algorithm::deconvolution_direct, *dec_src_desc,
                        *dec_weights_desc, *dec_bias_desc, *dec_dst_desc,
                        strides, padL, padR, deconv_primitive_desc);

        auto aa = allows_attr_t {false};
        test_bwd_pd_constructors<pd_t, hint_pd_t>(
                deconv_bwd_weights_primitive_desc, deconv_primitive_desc, aa,
                algorithm::deconvolution_direct, *dec_src_desc,
                *dec_weights_desc, *dec_bias_desc, *dec_dst_desc, strides, padL,
                padR);

        ASSERT_TRUE(deconv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_SRC)
                == deconv_bwd_weights_primitive_desc.src_desc());
        ASSERT_TRUE(deconv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == deconv_bwd_weights_primitive_desc.diff_dst_desc());
        ASSERT_TRUE(deconv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS)
                == deconv_bwd_weights_primitive_desc.diff_weights_desc());
        ASSERT_TRUE(deconv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_BIAS)
                == deconv_bwd_weights_primitive_desc.diff_bias_desc());

        ASSERT_EQ(deconv_bwd_weights_primitive_desc.get_algorithm(),
                algorithm::deconvolution_direct);
        ASSERT_EQ(deconv_bwd_weights_primitive_desc.get_prop_kind(),
                prop_kind::backward_weights);
        ASSERT_EQ(deconv_bwd_weights_primitive_desc.get_strides(), strides);
        ASSERT_EQ(deconv_bwd_weights_primitive_desc.get_padding_l(), padL);
        ASSERT_EQ(deconv_bwd_weights_primitive_desc.get_padding_r(), padR);

        deconvolution_backward_weights(deconv_bwd_weights_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, dst->get()},
                                {DNNL_ARG_SRC, src->get()},
                                {DNNL_ARG_DIFF_WEIGHTS, weights->get()},
                                {DNNL_ARG_DIFF_BIAS, bias->get()}});
        strm.wait();

        auto conv_primitive_desc = convolution_forward::primitive_desc(eng,
                prop_kind::forward_training, algorithm::convolution_direct,
                *con_src_desc, *con_weights_desc, *con_dst_desc, strides, padL,
                padR);

        deconv_bwd_weights_primitive_desc
                = pd_t(deconv_bwd_weights_primitive_desc
                                .get()); // test construction from a C pd

        auto conv_bwd_weights_primitive_desc
                = convolution_backward_weights::primitive_desc(eng,
                        algorithm::convolution_direct, *con_src_desc,
                        *con_weights_desc, *con_dst_desc, strides, padL, padR,
                        conv_primitive_desc);

        convolution_backward_weights(conv_bwd_weights_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, conv_dst->get()},
                                {DNNL_ARG_SRC, conv_src->get()},
                                {DNNL_ARG_DIFF_WEIGHTS, conv_weights}});
        strm.wait();

        auto weights_tr = test::make_memory(*con_weights_desc, eng);
        transpose_wei<data_t>(dd, weights->get(), weights_tr);

        compare_data<data_t>(weights_tr, conv_weights);

        if (with_bias) {
            auto ref_bias = test::make_memory(*dec_bias_desc, eng);
            compute_bias_bwd<data_t>(dd, dst->get(), ref_bias);
            compare_data<data_t>(ref_bias, bias->get());
        }
    }
};

using deconvolution_test_float = deconvolution_test_t<float>;

TEST_P(deconvolution_test_float, TestDeconvolution) {}

#define EXPAND_FORMATS(src, weights, bias, dst) \
    { \
        dnnl::memory::format_tag::src, dnnl::memory::format_tag::weights, \
                dnnl::memory::format_tag::bias, dnnl::memory::format_tag::dst \
    }

#define ALGORITHM dnnl::algorithm::deconvolution_direct

#define PARAMS(src, weights, bias, dst, ...) \
    deconvolution_test_params_t { \
        ALGORITHM, EXPAND_FORMATS(src, weights, bias, dst), {}, { \
            __VA_ARGS__ \
        } \
    }

#define CPU_INST_TEST_CASE(str, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P( \
            str, deconvolution_test_float, ::testing::Values(__VA_ARGS__))
#define GPU_INST_TEST_CASE(str, ...) \
    GPU_INSTANTIATE_TEST_SUITE_P( \
            str, deconvolution_test_float, ::testing::Values(__VA_ARGS__))

#define FMT_BIAS x
#define FMT_DATA_BLOCKED nChw8c
#define FMT_WEIGHTS_BLOCKED Ohwi8o
#define FMT_DATA_BLOCKED_GPU NChw16n16c
#define FMT_WEIGHTS_BLOCKED_GPU IOhw16i16o

CPU_INST_TEST_CASE(SimpleSmall_NCHW,
        PARAMS(nchw, oihw, x, nchw, 2, 1, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1),
        PARAMS(nchw, oihw, x, nchw, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, oihw, x, nhwc, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, hwio, x, nhwc, 2, 1, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1),
        PARAMS(nhwc, hwio, x, nhwc, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, goihw, x, nhwc, 2, 2, 6, 4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, hwigo, x, nhwc, 2, 2, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1)

);

CPU_INST_TEST_CASE(SimpleSmall_Blocked,
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 32, 12, 12, 32, 13, 13, 3, 3, 0, 0, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 32, 4, 4, 32, 3, 3, 3, 3, 1, 1, 1, 1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 32, 4, 4, 32, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 32, 2, 2, 32, 3, 3, 3, 3, 0, 0, 1, 1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 32, 2, 2, 32, 2, 2, 3, 3, 1, 1, 1, 1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 48, 13, 13, 32, 13, 13, 3, 3, 1, 1, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 48, 11, 11, 32, 13, 13, 3, 3, 0, 0, 1,
                1));

GPU_INST_TEST_CASE(SimpleSmall_Blocked,
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 32, 12, 12, 32, 10, 10, 3, 3, 0, 0,
                1, 1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 32, 4, 4, 32, 3, 3, 3, 3, 1, 1, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 32, 4, 4, 32, 4, 4, 3, 3, 0, 0, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 32, 2, 2, 32, 3, 3, 3, 3, 0, 0, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 32, 2, 2, 32, 2, 2, 3, 3, 1, 1, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 48, 13, 13, 32, 13, 13, 3, 3, 1, 1,
                1, 1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 48, 11, 11, 32, 13, 13, 3, 3, 0, 0,
                1, 1));

GPU_INST_TEST_CASE(SimpleSmall_NCHW,
        PARAMS(nchw, oihw, x, nchw, 2, 1, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1),
        PARAMS(nchw, oihw, x, nchw, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, oihw, x, nhwc, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, hwio, x, nhwc, 2, 1, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1),
        PARAMS(nhwc, hwio, x, nhwc, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, goihw, x, nhwc, 2, 2, 6, 4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, hwigo, x, nhwc, 2, 2, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1));

GPU_INST_TEST_CASE(SimpleSmall_NHWC,
        PARAMS(nchw, oihw, x, nhwc, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, ohwi, x, nhwc, 2, 1, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1),
        PARAMS(nhwc, ohwi, x, nhwc, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nchw, goihw, x, nchw, 2, 2, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1),
        PARAMS(nchw, goihw, x, nhwc, 2, 2, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1),
        PARAMS(nhwc, gohwi, x, nhwc, 2, 2, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1));
} // namespace dnnl
