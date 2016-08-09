#include "mkl_dnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkl_dnn.hpp"

namespace mkl_dnn {

struct test_convolution_descr_t {
    uint32_t mb;
    uint32_t ng;
    uint32_t ic, ih, iw;
    uint32_t oc, oh, ow;
    uint32_t kh, kw;
    int32_t padh, padw;
    uint32_t strh, strw;
};

template <typename data_t>
void compute_ref_conv_fwd_nchw(test_convolution_descr_t c, data_t *in,
        data_t *filt, data_t *bias, data_t *out)
{
#pragma omp parallel for collapse(5)
    for (uint32_t n = 0; n < c.mb; n++) {
        for (uint32_t g = 0; g < c.ng; g++) {
            for (uint32_t oc = 0; oc < c.oc / c.ng; oc++) {
                for (uint32_t oh = 0; oh < c.oh; oh++) {
                    for (uint32_t ow = 0; ow < c.ow; ow++) {
                        uint32_t oidx = n * c.oc * c.oh * c.ow
                                + g * c.oc / c.ng * c.oh * c.ow
                                + oc * c.oh * c.ow + oh * c.ow + ow;
                        out[oidx] = bias ? bias[g * c.oc / c.ng + oc] : 0.0;
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
                                            + ic * c.ih * c.ow + ih * c.iw + iw;
                                    uint32_t fidx = g * c.oc / c.ng * c.ic
                                                    / c.ng * c.kh * c.kw
                                            + oc * c.ic / c.ng * c.kh * c.kw
                                            + ic * c.kh * c.kw + kh * c.kw + kw;

                                    out[oidx] += in[iidx] * filt[fidx];
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

        ASSERT_EQ(p.src_format, memory::format::nchw);
        ASSERT_TRUE(
                (p.weights_format == memory::format::oihw && p.test_cd.ng == 1)
                || (p.weights_format == memory::format::goihw
                           && p.test_cd.ng > 1));
        ASSERT_EQ(p.bias_format, memory::format::x);
        ASSERT_EQ(p.dst_format, memory::format::nchw);
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu
                || p.engine_kind == engine::kind::cpu_lazy);
        ASSERT_EQ(p.aprop_kind, prop_kind::forward);
        ASSERT_EQ(p.aalgorithm, convolution::direct);
        auto eng = engine(p.engine_kind, 0);
        memory::precision prec = data_traits<data_t>::prec;
        ASSERT_EQ(prec, mkl_dnn::memory::precision::f32);

        test_convolution_descr_t cd = p.test_cd;
        size_t src_size = cd.mb * cd.ic * cd.ih * cd.iw;
        data_t *src_data = new data_t[src_size];
        fill_data(src_size, src_data);
        size_t weights_size
                = cd.ng * (cd.oc / cd.ng) * (cd.ic / cd.ng) * cd.kh * cd.kw;
        data_t *weights_data = new data_t[weights_size];
        fill_data(weights_size, weights_data);
        size_t bias_size = cd.oc;
        data_t *bias_data = new data_t[bias_size];
        fill_data(bias_size, bias_data);
        size_t dst_size = cd.mb * cd.oc * cd.oh * cd.ow;
        data_t *dst_data = new data_t[dst_size];
        // fillData(dst_size, output_data);
        data_t *dst_ref_data = new data_t[dst_size];
        // fillData(dst_size, output_ref_data);

        auto c_src_desc
                = create_md({ cd.mb, cd.ic, cd.ih, cd.iw }, prec, p.src_format);
        auto c_weights_desc = cd.ng > 1 ?
                create_md({ cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw },
                        prec, p.weights_format) :
                create_md(
                        { cd.oc, cd.ic, cd.kh, cd.kw }, prec, p.weights_format);
        auto c_bias_desc = create_md({ cd.oc }, prec, p.bias_format);
        auto c_dst_desc
                = create_md({ cd.mb, cd.oc, cd.oh, cd.ow }, prec, p.dst_format);

        auto c_src = memory(
                memory::primitive_desc(c_src_desc, eng), (void *)src_data);
        auto c_weights = memory(memory::primitive_desc(c_weights_desc, eng),
                (void *)weights_data);
        auto c_bias = memory(
                memory::primitive_desc(c_bias_desc, eng), (void *)bias_data);
        auto c_dst = memory(
                memory::primitive_desc(c_dst_desc, eng), (void *)dst_data);

        auto c = convolution(p.aprop_kind, p.aalgorithm, c_src, c_weights,
                c_bias, c_dst, { cd.strh, cd.strw }, { cd.padh, cd.padw },
                padding_kind::zero);

        stream().submit({ c }).wait();

        compute_ref_conv_fwd_nchw(
                cd, src_data, weights_data, bias_data, dst_ref_data);
        compare_data(dst_ref_data, dst_data, dst_size);
    }
};

using convolution_test_float = convolution_test<float>;
using conv_test_params_float = conv_test_params;

TEST_P(convolution_test_float, TestsConvolution)
{
}
INSTANTIATE_TEST_CASE_P(TestConvolutionForward, convolution_test_float,
        ::testing::Values(conv_test_params_float{ prop_kind::forward,
                engine::kind::cpu, convolution::direct, memory::format::nchw,
                memory::format::oihw, memory::format::x, memory::format::nchw,
                { 2, 1, 4, 4, 4, 6, 4, 4, 3, 3, 1, 1, 1, 1 } }));
INSTANTIATE_TEST_CASE_P(
        TestConvolutionAlexnetForward, convolution_test_float,
        ::testing::Values(
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
}
