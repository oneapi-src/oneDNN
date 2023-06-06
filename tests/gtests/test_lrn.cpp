/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

using tag = memory::format_tag;
using dt = memory::data_type;

struct lrn_test_params_t {
    dt src_dt;
    dt dst_dt; // diff_dst_dt
    dt diff_src_dt;
    tag src_tag;
    tag dst_tag; // diff_dst_tag
    tag diff_src_tag;
    memory::dims dims;
    memory::dim local_size;
    float alpha, beta, k;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

bool cuda_check_format_tag(tag atag) {
    return impl::utils::one_of(atag, tag::ncdhw, tag::nchw, tag::nhwc, tag::ncw,
            tag::nwc, tag::any);
}

template <typename... Rest>
bool cuda_check_format_tag(tag first_tag, Rest... rest_tags) {
    const bool ok = cuda_check_format_tag(first_tag);
    if (!ok) return ok;
    return cuda_check_format_tag(rest_tags...);
}

bool hip_check_format_tag(tag atag) {
    return impl::utils::one_of(atag, tag::nchw, tag::any);
}

template <typename... Rest>
bool hip_check_format_tag(tag first_tag, Rest... rest_tags) {
    const bool ok = hip_check_format_tag(first_tag);
    if (!ok) return ok;
    return hip_check_format_tag(rest_tags...);
}

class lrn_test_t : public ::testing::TestWithParam<lrn_test_params_t> {
private:
    lrn_test_params_t p;
    memory src, workspace;
    std::shared_ptr<lrn_forward::primitive_desc> pd_fwd_hint;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<lrn_test_params_t>::GetParam();

        SKIP_IF(unsupported_data_type(p.src_dt, p.dst_dt),
                "Engine does not support this data type.");

        SKIP_IF_CUDA(
                p.dst_dt == dt::s8, "Unsupported int8 destination data type");
        SKIP_IF_HIP(
                p.dst_dt == dt::s8, "Unsupported int8 destination data type");

        SKIP_IF_CUDA(!cuda_check_format_tag(p.src_tag, p.dst_tag),
                "Unsupported format tag");
        SKIP_IF_HIP(!hip_check_format_tag(p.src_tag, p.dst_tag),
                "Unsupported format tag");

        SKIP_IF_CUDA(p.src_dt != p.dst_dt && p.src_dt != dt::undef
                        && p.dst_dt != dt::undef,
                "Unsupported different data types for source and "
                "destination");
        SKIP_IF_HIP(p.src_dt != p.dst_dt && p.src_dt != dt::undef
                        && p.dst_dt != dt::undef,
                "Unsupported different data types for source and "
                "destination");

        SKIP_IF_CUDA(p.src_tag != p.dst_tag && p.src_tag != tag::any
                        && p.dst_tag != tag::any,
                "Unsupported different memory formats for source and "
                "destination");
        SKIP_IF_HIP(p.src_tag != p.dst_tag && p.src_tag != tag::any
                        && p.dst_tag != tag::any,
                "Unsupported different memory formats for source and "
                "destination");

        catch_expected_failures(
                [&]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Forward(prop_kind pk, algorithm aalgorithm) {
        // lrn specific types and values
        using pd_t = lrn_forward::primitive_desc;

        auto eng = get_test_engine();
        auto strm = make_stream(eng);

        auto aa = allows_attr_t {false};

        auto src_md = memory::desc(p.dims, p.src_dt, p.src_tag);
        auto dst_md = memory::desc(p.dims, p.dst_dt, p.dst_tag);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        pd = pd_t(eng, pk, aalgorithm, src_md, dst_md, p.local_size, p.alpha,
                p.beta, p.k);
        // test all pd ctors
        test_fwd_pd_constructors<pd_t>(pd, aa, pk, aalgorithm, src_md, dst_md,
                p.local_size, p.alpha, p.beta, p.k);
        pd_fwd_hint = std::make_shared<pd_t>(pd);

        EXPECT_ANY_THROW(lrn_forward(pd, {}));
        // default primitive ctor
        auto lrn = lrn_forward();
        // regular primitive ctor
        lrn = lrn_forward(pd);

        // check primitive kind is lrn
        ASSERT_TRUE(lrn.get_kind() == primitive::kind::lrn);
        // query for descs from pd
        const auto src_desc = pd.src_desc();
        const auto dst_desc = pd.dst_desc();
        // query for src_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC) == src_desc);
        if (p.src_tag != tag::any) { ASSERT_TRUE(src_md == src_desc); }
        // query for dst_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DST) == dst_desc);
        if (p.dst_tag != tag::any) { ASSERT_TRUE(dst_md == dst_desc); }

        // query primitive parameters
        ASSERT_EQ(pd.get_prop_kind(), pk);
        ASSERT_EQ(pd.get_algorithm(), aalgorithm);
        ASSERT_EQ(pd.get_local_size(), p.local_size);
        ASSERT_EQ(pd.get_alpha(), p.alpha);
        ASSERT_EQ(pd.get_beta(), p.beta);
        ASSERT_EQ(pd.get_k(), p.k);

        // query for workspace
        const auto workspace_desc = pd.workspace_desc();

        // check primitive returns zero_md for all rest md
        ASSERT_TRUE(pd.weights_desc().is_zero());
        ASSERT_TRUE(pd.diff_src_desc().is_zero());
        ASSERT_TRUE(pd.diff_dst_desc().is_zero());
        ASSERT_TRUE(pd.diff_weights_desc().is_zero());

        src = test::make_memory(src_desc, eng);
        auto dst = test::make_memory(dst_desc, eng);
        workspace = test::make_memory(workspace_desc, eng);

        fill_data(p.src_dt, src, 1, 1);
        // test out-place mode
        lrn.execute(strm,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst},
                        {DNNL_ARG_WORKSPACE, workspace}});
        strm.wait();
    }

    void Backward(algorithm aalgorithm) {
        // lrn specific types and values
        using pd_t = lrn_backward::primitive_desc;
        using hint_pd_t = lrn_forward::primitive_desc;
        allows_attr_t aa {false}; // doesn't support anything

        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        auto diff_src_md = memory::desc(p.dims, p.diff_src_dt, p.diff_src_tag);
        auto diff_dst_md = memory::desc(p.dims, p.dst_dt, p.dst_tag);
        auto src_md = memory::desc(p.dims, p.src_dt, p.src_tag);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        pd = pd_t(eng, aalgorithm, diff_src_md, diff_dst_md, src_md,
                p.local_size, p.alpha, p.beta, p.k, *pd_fwd_hint);
        // test all pd ctors
        test_bwd_pd_constructors<pd_t, hint_pd_t>(pd, *pd_fwd_hint, aa,
                aalgorithm, diff_src_md, diff_dst_md, src_md, p.local_size,
                p.alpha, p.beta, p.k);

        EXPECT_ANY_THROW(lrn_backward(pd, {}));
        // default primitive ctor
        auto lrn = lrn_backward();
        // regular primitive ctor
        lrn = lrn_backward(pd);

        // check primitive kind is lrn
        ASSERT_TRUE(lrn.get_kind() == primitive::kind::lrn);

        // query for descs from pd
        const auto diff_src_desc = pd.diff_src_desc();
        const auto diff_dst_desc = pd.diff_dst_desc();
        const auto src_desc = pd.src_desc();
        // query for diff_src_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == diff_src_desc);
        if (p.diff_src_tag != tag::any) {
            ASSERT_TRUE(diff_src_md == diff_src_desc);
        }
        // query for diff_dst_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == diff_dst_desc);
        if (p.dst_tag != tag::any) {
            ASSERT_TRUE(diff_dst_md == diff_dst_desc);
        }
        // query for dst_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC) == src_desc);
        if (p.src_tag != tag::any) { ASSERT_TRUE(src_md == src_desc); }

        // query primitive parameters
        ASSERT_EQ(pd.get_prop_kind(), prop_kind::backward_data);
        ASSERT_EQ(pd.get_algorithm(), aalgorithm);
        ASSERT_EQ(pd.get_local_size(), p.local_size);
        ASSERT_EQ(pd.get_alpha(), p.alpha);
        ASSERT_EQ(pd.get_beta(), p.beta);
        ASSERT_EQ(pd.get_k(), p.k);

        // check primitive returns zero_md for all rest md
        ASSERT_TRUE(pd.weights_desc().is_zero());
        ASSERT_TRUE(pd.dst_desc().is_zero());
        ASSERT_TRUE(pd.diff_weights_desc().is_zero());

        auto diff_src = test::make_memory(diff_src_desc, eng);
        auto diff_dst = test::make_memory(diff_dst_desc, eng);

        fill_data(p.diff_src_dt, diff_dst, 2, 2);

        // test out-place mode
        lrn.execute(strm,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_DIFF_DST, diff_dst},
                        {DNNL_ARG_DIFF_SRC, diff_src},
                        {DNNL_ARG_WORKSPACE, workspace}});
        strm.wait();
    }

    void Test() {
        const std::vector<prop_kind> pks
                = {prop_kind::forward_training, prop_kind::forward_inference};
        const std::vector<algorithm> algs = {
                algorithm::lrn_across_channels, algorithm::lrn_within_channel};

        for_(auto pk : pks)
        for (auto alg : algs) {
            SKIP_FOR_LOOP_CUDA(alg != algorithm::lrn_across_channels,
                    "Unsupported algorithm");

            Forward(pk, alg);
            if (pk == prop_kind::forward_training
                    && p.diff_src_dt != memory::data_type::undef) {
                SKIP_IF(unsupported_data_type(p.diff_src_dt),
                        "Engine does not support this data type.");
                SKIP_IF_CUDA(!cuda_check_format_tag(p.diff_src_tag),
                        "Unsupported format tag");
                SKIP_IF_HIP(!hip_check_format_tag(p.diff_src_tag),
                        "Unsupported format tag");

                SKIP_IF_CUDA(p.src_dt != p.diff_src_dt && p.src_dt != dt::undef
                                && p.diff_src_dt != dt::undef,
                        "Unsupported different data types for diff_source and "
                        "diff_destination");
                SKIP_IF_HIP(p.src_dt != p.diff_src_dt && p.src_dt != dt::undef
                                && p.diff_src_dt != dt::undef,
                        "Unsupported different data types for diff_source and "
                        "diff_destination");

                SKIP_IF_CUDA(p.src_tag != p.diff_src_tag
                                && p.src_tag != tag::any
                                && p.diff_src_tag != tag::any,
                        "Unsupported different memory formats for diff_source "
                        "and diff_destination");
                SKIP_IF_HIP(p.src_tag != p.diff_src_tag && p.src_tag != tag::any
                                && p.diff_src_tag != tag::any,
                        "Unsupported different memory formats for diff_source "
                        "and diff_destination");

                Backward(alg);
            }
        }
    }

    bool is_fwd(prop_kind pk) const {
        return pk == prop_kind::forward_training
                || pk == prop_kind::forward_inference;
    }
};

using tp = lrn_test_params_t;

TEST_P(lrn_test_t, TestsLRN) {}

INSTANTIATE_TEST_SUITE_P(Test_LRN_EF, lrn_test_t,
        ::testing::Values(
                // Negative dims
                tp {dt::f32, dt::f32, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, -2, 128, 256}, 5, 1e-4f, 0.75f, 1.f,
                        true, dnnl_invalid_arguments},
                // Negative local size
                tp {dt::f32, dt::f32, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, -1, 1e-4f, 0.75f, 1.f,
                        true, dnnl_invalid_arguments},
                // Tag for src on forward is not specified
                tp {dt::f32, dt::f32, dt::undef, tag::any, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 5, 1e-4f, 0.75f, 1.f,
                        true, dnnl_invalid_arguments},
                // Tag for src on backward is not specified
                tp {dt::f32, dt::f32, dt::f32, tag::any, tag::nchw, tag::nchw,
                        {2, 2, 128, 256}, 5, 1e-4f, 0.75f, 1.f, true,
                        dnnl_invalid_arguments},
                // Data type for src is not specified
                tp {dt::undef, dt::f32, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 5, 1e-4f, 0.75f, 1.f,
                        true, dnnl_invalid_arguments},
                // Different data types are not supported
                tp {dt::f32, dt::bf16, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 5, 1e-4f, 0.75f, 1.f,
                        true, dnnl_unimplemented},
                // Different data types are not supported
                tp {dt::f32, dt::bf16, dt::f32, tag::nchw, tag::nchw, tag::nchw,
                        {2, 2, 128, 256}, 5, 1e-4f, 0.75f, 1.f, true,
                        dnnl_unimplemented},
                // Different memory formats are not supported
                tp {dt::f32, dt::f32, dt::undef, tag::nchw, tag::nhwc,
                        tag::undef, {2, 2, 128, 256}, 5, 1e-4f, 0.75f, 1.f,
                        true, dnnl_unimplemented},
                // Different memory formats are not supported
                tp {dt::f32, dt::f32, dt::f32, tag::nchw, tag::nhwc, tag::nchw,
                        {2, 2, 128, 256}, 5, 1e-4f, 0.75f, 1.f, true,
                        dnnl_unimplemented}));

static auto all_cases = [](memory::data_type src_dt, memory::data_type dst_dt,
                                memory::data_type diff_src_dt) {
    return ::testing::Values(
            // Forward_nChw16c_padded_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw16c, tag::nChw16c,
                    tag::nChw16c, {2, 17, 4, 4}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw16c, tag::nChw16c,
                    tag::nChw16c, {2, 26, 4, 4}, 5, 1.0e-4f, 0.75f, 5.7f},
            // Forward_nChw8c_padded_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 7, 4, 4}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 26, 4, 4}, 5, 1.0e-4f, 0.75f, 5.7f},
            // Forward_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {2, 10, 4, 4}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {2, 10, 4, 4}, 5, 1.0e-4f, 0.75f, 3.0f},
            // ForwardNHWC_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nhwc, tag::nhwc, tag::nhwc,
                    {2, 10, 4, 4}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nhwc, tag::nhwc, tag::nhwc,
                    {2, 10, 4, 4}, 5, 1.0e-4f, 0.75f, 4.85f},
            // Forward_nChw8c_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 16, 4, 4}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 16, 4, 4}, 5, 1.0e-4f, 0.75f, 5.7f},
            // Forward_nChw16c_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw16c, tag::nChw16c,
                    tag::nChw16c, {2, 16, 4, 4}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw16c, tag::nChw16c,
                    tag::nChw16c, {2, 16, 4, 4}, 5, 1.0e-4f, 0.75f, 5.7f},
            // AlexnetForwardNCHW_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {2, 96, 55, 55}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {2, 256, 27, 27}, 5, 1.0e-4f, 0.75f, 1.0f},
            // AlexnetForwardNHWC_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nhwc, tag::nhwc, tag::nhwc,
                    {2, 96, 55, 55}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nhwc, tag::nhwc, tag::nhwc,
                    {2, 256, 27, 27}, 5, 1.0e-4f, 0.75f, 1.0f},
            // AlexnetForward_nChw8c_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 96, 55, 55}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 256, 27, 27}, 5, 1.0e-4f, 0.75f, 1.0f},
            // AlexnetForward_nChw16c_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw16c, tag::nChw16c,
                    tag::nChw16c, {2, 96, 55, 55}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw16c, tag::nChw16c,
                    tag::nChw16c, {2, 256, 27, 27}, 5, 1.0e-4f, 0.75f, 1.0f},
            // GoogleNetV1ForwardNCHW_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {2, 64, 56, 56}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {2, 192, 56, 56}, 5, 1.0e-4f, 0.75f, 1.0f},
            // GoogleNetV1Forward_nChw8c_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 64, 56, 56}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 192, 56, 56}, 5, 1.0e-4f, 0.75f, 1.0f},
            // GoogleNetV1Forward_nChw16c_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw16c, tag::nChw16c,
                    tag::nChw16c, {2, 64, 56, 56}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw16c, tag::nChw16c,
                    tag::nChw16c, {2, 192, 56, 56}, 5, 1.0e-4f, 0.75f, 1.0f},
            // RCNNForwardBlocked_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 96, 55, 55}, 3, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 256, 27, 27}, 3, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 96, 55, 55}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 256, 27, 27}, 5, 1.0e-4f, 0.75f, 1.0f},
            // ForwardNCHWTail_cases
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {1, 64, 1, 9}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {1, 64, 2, 9}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {1, 64, 3, 9}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {1, 64, 4, 9}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {1, 64, 5, 9}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {1, 64, 9, 6}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {1, 64, 7, 9}, 5, 1.0e-4f, 0.75f, 1.0f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {1, 64, 8, 9}, 5, 1.0e-4f, 0.75f, 1.0f});
};

#define EXPAND_DTS(src, dst, diff_src) \
    memory::data_type::src, memory::data_type::dst, memory::data_type::diff_src

#define INST_TEST_CASE(name, suite, ...) \
    INSTANTIATE_TEST_SUITE_P(name, lrn_test_t, suite(__VA_ARGS__));

#define CPU_INST_TEST_CASE(name, suite, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P(name, lrn_test_t, suite(__VA_ARGS__));

#define GPU_INST_TEST_CASE(name, suite, ...) \
    GPU_INSTANTIATE_TEST_SUITE_P(name, lrn_test_t, suite(__VA_ARGS__));

INST_TEST_CASE(LRNSimpleF32, all_cases, EXPAND_DTS(f32, f32, f32));
INST_TEST_CASE(LRNSimpleBF16, all_cases, EXPAND_DTS(bf16, bf16, bf16));
INST_TEST_CASE(LRNSimpleF16, all_cases, EXPAND_DTS(f16, f16, undef));
} // namespace dnnl
