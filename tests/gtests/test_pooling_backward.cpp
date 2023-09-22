/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
* Copyright 2022-2023 Arm Ltd. and affiliates
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
namespace dnnl {

struct test_pool_bwd_desc_t {
    memory::dim mb, c;
    memory::dim id, ih, iw;
    memory::dim od, oh, ow;
    memory::dim kd, kh, kw;
    memory::dim dd, dh, dw;
    memory::dim padf, padt, padl;
    memory::dim strd, strh, strw;
};

struct pool_bwd_test_params_t {
    algorithm aalgorithm;
    memory::format_tag diff_src_format;
    memory::format_tag diff_dst_format;
    int ndims;
    test_pool_bwd_desc_t test_pd;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

bool cuda_check_format_tags(memory::format_tag format) {
    bool format_ok = format == memory::format_tag::ncdhw
            || format == memory::format_tag::ndhwc
            || format == memory::format_tag::nchw
            || format == memory::format_tag::nhwc
            || format == memory::format_tag::ncw
            || format == memory::format_tag::nwc
            || format == memory::format_tag::any
            || format == memory::format_tag::nCdhw4c;

    return format_ok;
}

bool hip_check_format_tags(memory::format_tag format) {
    bool format_ok = format == memory::format_tag::nchw
            || format == memory::format_tag::ncdhw;
    return format_ok;
}

template <typename data_t>
class pooling_bwd_test_t
    : public ::testing::TestWithParam<pool_bwd_test_params_t> {
private:
    std::shared_ptr<memory::desc> src_desc;
    std::shared_ptr<memory::desc> dst_desc;
    memory workspace;
    pooling_forward::primitive_desc pool_prim_desc;
    pool_bwd_test_params_t p;
    memory::dims strides, ker, dilation, pad_l, pad_r;
    engine eng;
    stream strm;
    memory::data_type data_type;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        SKIP_IF_CUDA(!cuda_check_format_tags(p.diff_src_format),
                "Unsupported format tag");
        SKIP_IF_CUDA(!cuda_check_format_tags(p.diff_dst_format),
                "Unsupported format tag");
        // This test makes assumptions on workspace content for the max
        // algorithm therefore it cannot be used for non-intel implementations.
        SKIP_IF_CUDA(p.aalgorithm == algorithm::pooling_max,
                "Test is not designed to test non-intel implementations of max "
                "algorithm");

        SKIP_IF_HIP(!hip_check_format_tags(p.diff_src_format),
                "Unsupported format tag");
        SKIP_IF_HIP(!hip_check_format_tags(p.diff_dst_format),
                "Unsupported format tag");
        // This test makes assumptions on workspace content for the max
        // algorithm therefore it cannot be used for non-intel implementations.
        SKIP_IF_HIP(p.aalgorithm == algorithm::pooling_max,
                "Test is not designed to test non-intel implementations of max "
                "algorithm");

        catch_expected_failures(
                [&]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        test_pool_bwd_desc_t pd = p.test_pd;

        eng = get_test_engine();
        strm = make_stream(eng);
        data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, dnnl::memory::data_type::f32);

        if (p.ndims == 5) {
            auto src_dims = {pd.mb, pd.c, pd.id, pd.ih, pd.iw};
            auto dst_dims = {pd.mb, pd.c, pd.od, pd.oh, pd.ow};
            src_desc = std::make_shared<memory::desc>(
                    src_dims, data_type, p.diff_src_format);
            dst_desc = std::make_shared<memory::desc>(
                    dst_dims, data_type, p.diff_dst_format);
        } else {
            auto src_dims = {pd.mb, pd.c, pd.ih, pd.iw};
            auto dst_dims = {pd.mb, pd.c, pd.oh, pd.ow};
            src_desc = std::make_shared<memory::desc>(
                    src_dims, data_type, p.diff_src_format);
            dst_desc = std::make_shared<memory::desc>(
                    dst_dims, data_type, p.diff_dst_format);
        }

        if (p.ndims == 5) {
            strides = memory::dims({pd.strd, pd.strh, pd.strw});
            ker = memory::dims({pd.kd, pd.kh, pd.kw});
            dilation = memory::dims({pd.dd, pd.dh, pd.dw});
            pad_l = memory::dims({pd.padf, pd.padt, pd.padl});
            pad_r = memory::dims({right_padding(pd.id, pd.od, pd.kd, pd.padf,
                                          pd.strd, pd.dd),
                    right_padding(pd.ih, pd.oh, pd.kh, pd.padt, pd.strh, pd.dh),
                    right_padding(
                            pd.iw, pd.ow, pd.kw, pd.padl, pd.strw, pd.dw)});
        } else {
            strides = memory::dims({pd.strh, pd.strw});
            ker = memory::dims({pd.kh, pd.kw});
            dilation = memory::dims({pd.dh, pd.dw});
            pad_l = memory::dims({pd.padt, pd.padl});
            pad_r = memory::dims({right_padding(pd.ih, pd.oh, pd.kh, pd.padt,
                                          pd.strh, pd.dh),
                    right_padding(
                            pd.iw, pd.ow, pd.kw, pd.padl, pd.strw, pd.dw)});
        }

        Forward();
        Backward();
    }

    void check_prim_desc(
            const pooling_backward::primitive_desc &pool_bwd_prim_desc) {
        ASSERT_TRUE(pool_bwd_prim_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == pool_bwd_prim_desc.diff_src_desc());
        ASSERT_TRUE(pool_bwd_prim_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == pool_bwd_prim_desc.diff_dst_desc());
        ASSERT_TRUE(pool_bwd_prim_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_WORKSPACE)
                == pool_bwd_prim_desc.workspace_desc());

        ASSERT_EQ(pool_bwd_prim_desc.get_prop_kind(), prop_kind::backward_data);
        ASSERT_EQ(pool_bwd_prim_desc.get_algorithm(), p.aalgorithm);
        ASSERT_EQ(pool_bwd_prim_desc.get_kernel(), ker);
        ASSERT_EQ(pool_bwd_prim_desc.get_strides(), strides);
        ASSERT_EQ(pool_bwd_prim_desc.get_padding_l(), pad_l);
        ASSERT_EQ(pool_bwd_prim_desc.get_padding_r(), pad_r);

        if (p.test_pd.dd == 0 && p.test_pd.dh == 0 && p.test_pd.dw == 0)
            ASSERT_EQ(pool_prim_desc.get_dilations(),
                    memory::dims(pool_prim_desc.src_desc().get_ndims() - 2));
        else
            ASSERT_EQ(pool_prim_desc.get_dilations(), dilation);
    }

    void Forward() {
        auto src = test::make_memory(*src_desc, eng);
        auto dst = test::make_memory(*dst_desc, eng);

        fill_data<data_t>(src.get_desc().get_size() / sizeof(data_t), src);
        fill_data<data_t>(dst.get_desc().get_size() / sizeof(data_t), dst);
        check_zero_tail<data_t>(1, src);
        check_zero_tail<data_t>(1, dst);

        pool_prim_desc = pooling_forward::primitive_desc(eng,
                prop_kind::forward_training, p.aalgorithm, *src_desc, *dst_desc,
                strides, ker, dilation, pad_l, pad_r);

        auto p_workspace_desc = pool_prim_desc.workspace_desc();
        workspace = test::make_memory(p_workspace_desc, eng);

        EXPECT_ANY_THROW(pooling_forward(pool_prim_desc, {}));
        pooling_forward(pool_prim_desc)
                .execute(strm,
                        {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst},
                                {DNNL_ARG_WORKSPACE, workspace}});

        strm.wait();

        check_zero_tail<data_t>(0, dst);
    }

    void Backward() {
        // pooling specific types and values
        using pd_t = pooling_backward::primitive_desc;
        using hint_pd_t = pooling_forward::primitive_desc;

        auto diff_src = test::make_memory(*src_desc, eng);
        auto diff_dst = test::make_memory(*dst_desc, eng);

        fill_data<data_t>(
                diff_dst.get_desc().get_size() / sizeof(data_t), diff_dst);
        fill_data<data_t>(
                diff_src.get_desc().get_size() / sizeof(data_t), diff_src);
        check_zero_tail<data_t>(1, diff_dst);
        check_zero_tail<data_t>(1, diff_src);

        auto pool_bwd_prim_desc = pd_t(eng, p.aalgorithm, *src_desc, *dst_desc,
                strides, ker, dilation, pad_l, pad_r, pool_prim_desc);
        // test all pd ctors
        allows_attr_t aa {false}; // doesn't support anything
        test_bwd_pd_constructors<pd_t, hint_pd_t>(pool_bwd_prim_desc,
                pool_prim_desc, aa, p.aalgorithm, *src_desc, *dst_desc, strides,
                ker, dilation, pad_l, pad_r);
        check_prim_desc(pool_bwd_prim_desc);

        pooling_backward(pool_bwd_prim_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, diff_dst},
                                {DNNL_ARG_DIFF_SRC, diff_src},
                                {DNNL_ARG_WORKSPACE, workspace}});
        strm.wait();

        check_zero_tail<data_t>(0, diff_src);
    }
};

using pooling_bwd_test_float = pooling_bwd_test_t<float>;
using pool_bwd_test_params_float = pool_bwd_test_params_t;

#define EXPAND_SIZES_3D(...) \
    5, { __VA_ARGS__ }
#define EXPAND_SIZES_2D( \
        mb, ic, ih, iw, oh, ow, kh, kw, dh, dw, padt, padl, strh, strw) \
    4, { \
        mb, ic, 1, ih, iw, 1, oh, ow, 1, kh, kw, 0, dh, dw, 0, padt, padl, 1, \
                strh, strw \
    }

TEST_P(pooling_bwd_test_float, TestsPoolingBackward) {}

INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardZeroDim, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 0, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 0, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                0, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 0, 4, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1)}));

INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardEF, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, -4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1),
                        true, dnnl_invalid_arguments},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                -2, 4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1),
                        true, dnnl_invalid_arguments},
                pool_bwd_test_params_float {algorithm::eltwise_square,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1),
                        true, dnnl_invalid_arguments}));

INSTANTIATE_TEST_SUITE_P(TestPooling_nChw16c_padded, pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_float {algorithm::pooling_max,
                                  memory::format_tag::nChw16c,
                                  memory::format_tag::nChw16c,
                                  EXPAND_SIZES_2D(4, 17, 6, 6, 7, 7, 2, 2, 0, 0,
                                          1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 23, 60, 60, 31, 31, 3, 4, 1, 1, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 3, 2, 2, 2, 1, 1, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 17, 60, 60, 31, 31, 4, 3, 2, 2, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 2, 3, 1, 1, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(4, 28, 60, 60, 31, 31, 4, 2, 1, 1, 1, 1,
                                2, 2)}));

INSTANTIATE_TEST_SUITE_P(TestPooling_nChw8c_padded, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 5, 6, 6, 7, 7, 2, 2, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 23, 60, 60, 31, 31, 3, 4, 0, 0, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 3, 2, 0, 0, 1, 1, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 17, 60, 60, 31, 31, 4, 3, 1, 1, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 2, 3, 1, 1, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(4, 28, 60, 60, 31, 31, 4, 2, 1, 1, 1, 1,
                                2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxKernelSlipsToPadding,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10, 5, 5)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10, 5, 5)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10, 5, 5)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10,
                                5, 5)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_nCdhw16c, pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_float {algorithm::pooling_max,
                                  memory::format_tag::nCdhw16c,
                                  memory::format_tag::nCdhw16c,
                                  EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30,
                                          2, 3, 4, 0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 23, 23, 23, 11, 11, 11, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 2, 2, 2, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 2, 2, 2, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_ncdhw, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30, 2, 3, 4,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_ndhwc, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30, 2, 3, 4,
                                1, 1, 0, 0, 0, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                2, 2, 2, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                2, 2, 2, 1, 1, 1, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_nCdhw8c, pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_float {algorithm::pooling_max,
                                  memory::format_tag::nCdhw8c,
                                  memory::format_tag::nCdhw8c,
                                  EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30,
                                          2, 3, 4, 0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMax3DunetNCDHW,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(1, 64, 64, 64, 64, 64, 64, 64, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(1, 128, 28, 28, 28, 28, 28, 28, 2, 2, 2,
                                1, 1, 1, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(1, 256, 12, 12, 12, 12, 12, 12, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMax3DunetNDHWC,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(1, 64, 64, 64, 64, 64, 64, 64, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(1, 128, 28, 28, 28, 28, 28, 28, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(1, 256, 12, 12, 12, 12, 12, 12, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxAlexNetNCHW,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxCIFAR10NCHW,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMax, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 3, 3, 1, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxBlocked,
        pooling_bwd_test_float,
        ::testing::Values(

                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 2, 2, 3, 3, 2, 2, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(122, 32, 32, 2, 32, 2, 3, 3, 2, 2, 1, 1,
                                1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardAvgBlocked,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 5, 5, 2, 2, 3, 3, 2, 2, 0, 0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 5, 5, 2, 2, 3, 3, 3, 3, 0, 0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 8, 3, 2, 2, 2, 3, 3, 5, 5, 1, 1, 2, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 8, 3, 2, 2, 2, 3, 3, 1, 1, 1, 1, 2, 1)}

                ));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxBlocked16,
        pooling_bwd_test_float,
        ::testing::Values(

                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                1, 16, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 2, 2, 3, 3, 2, 2, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(122, 32, 32, 2, 32, 2, 3, 3, 2, 2, 1, 1,
                                1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardAvgBlocked16,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 5, 5, 2, 2, 3, 3, 2, 2, 0, 0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 5, 5, 2, 2, 3, 3, 3, 3, 0, 0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 3, 2, 2, 2, 3, 3, 5, 5, 1, 1, 2, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 3, 2, 2, 2, 3, 3, 1, 1, 1, 1, 2, 1)}

                ));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxBlockedPerf,
        pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_float {algorithm::pooling_max,
                memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                EXPAND_SIZES_2D(
                        16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardAvgBlockedPerf,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 1,
                                1, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxBlocked16Perf,
        pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_float {algorithm::pooling_max,
                memory::format_tag::nChw16c, memory::format_tag::nChw16c,
                EXPAND_SIZES_2D(
                        16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardAvgBlocked16Perf,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 1, 1, 0,
                                0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardAsymmPadding,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 1, 1, 0, 1, 1, 1)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 14, 1, 8, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 14, 1, 8, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 14, 1, 8, 3, 3, 1, 1, 0, 1, 1, 2)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 100, 1, 51, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 100, 1, 51, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 100, 1, 51, 3, 3, 1, 1, 0, 1, 1, 2)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 102, 1, 52, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 102, 1, 52, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 102, 1, 52, 3, 3, 1, 1, 0, 1, 1, 2)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 9, 103, 7, 52, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 9, 103, 7, 52, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 9, 103, 7, 52, 3, 3, 1, 1, 0, 1, 1, 2)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 0, 0,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 0, 0,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 1, 1,
                                1, 1, 2, 2)}

                ));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAsymmDilation, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 1, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 1, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 1, 0, 1, 1, 1, 1)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1, 1, 1)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 2, 4,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 2, 4,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 2, 4,
                                1, 1, 2, 2)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 4, 2,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 4, 2,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 4, 2,
                                1, 1, 2, 2)}));

GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingSlipsToPadding, pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_t {algorithm::pooling_max,
                                  memory::format_tag::NChw16n16c,
                                  memory::format_tag::NChw16n16c,
                                  EXPAND_SIZES_2D(64, 64, 56, 56, 56, 56, 3, 3,
                                          0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_t {algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(64, 64, 56, 56, 56, 56, 3, 3, 0, 0, 1,
                                1, 1, 1)},
                pool_bwd_test_params_t {algorithm::pooling_avg_include_padding,
                        memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(64, 64, 56, 56, 56, 56, 3, 3, 0, 0, 1,
                                1, 1, 1)}));

GPU_INSTANTIATE_TEST_SUITE_P(TestPooling_ncdhw, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(5, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(5, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(5, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::NCdhw16n16c,
                        memory::format_tag::NCdhw16n16c,
                        EXPAND_SIZES_3D(32, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::NCdhw16n16c,
                        memory::format_tag::NCdhw16n16c,
                        EXPAND_SIZES_3D(32, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::NCdhw16n16c,
                        memory::format_tag::NCdhw16n16c,
                        EXPAND_SIZES_3D(32, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(3, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                2, 2, 2, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(3, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                3, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(3, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                5, 5, 5, 1, 1, 1, 1, 1, 1)}));

} // namespace dnnl
