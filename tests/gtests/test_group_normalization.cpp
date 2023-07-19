/*******************************************************************************
* Copyright 2023 Intel Corporation
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

struct group_normalization_test_params_t {
    dt src_dt;
    dt dst_dt; // diff_dst_dt
    dt diff_src_dt;
    tag src_tag;
    tag dst_tag; // diff_dst_tag
    tag diff_src_tag;
    memory::dims dims;
    memory::dim groups;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

class group_normalization_test_t
    : public ::testing::TestWithParam<group_normalization_test_params_t> {
private:
    group_normalization_test_params_t p;
    memory src, workspace, mean, variance, scale, shift;
    std::shared_ptr<group_normalization_forward::primitive_desc> pd_fwd_hint;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<
                group_normalization_test_params_t>::GetParam();

        SKIP_IF(unsupported_data_type(p.src_dt, p.dst_dt),
                "Engine does not support this data type.");

        SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
                "GPU engine is not supported");

        catch_expected_failures(
                [&]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Forward(prop_kind pk, normalization_flags flags) {
        SKIP_IF(unsupported_prop_kind(pk, p.src_dt, p.dst_dt),
                "Engine does not support this prop kind with this data types");

        // group_normalization specific types and values
        using pd_t = group_normalization_forward::primitive_desc;

        auto eng = get_test_engine();
        auto strm = make_stream(eng);

        const bool is_src_int8 = p.src_dt == memory::data_type::s8
                || p.src_dt == memory::data_type::u8;
        auto aa = allows_attr_t {false};
        if (get_test_engine_kind() == engine::kind::cpu && is_src_int8) {
            aa.scales = true;
        }
        if (get_test_engine_kind() == engine::kind::cpu) {
            aa.po_eltwise = true;
            aa.po_binary = true;
        }

        auto src_md = memory::desc(p.dims, p.src_dt, p.src_tag);
        auto dst_md = memory::desc(p.dims, p.dst_dt, p.dst_tag);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        pd = pd_t(eng, pk, src_md, dst_md, p.groups, /* epsilon = */ 1e-4f,
                flags);
        // test all pd ctors
        test_fwd_pd_constructors<pd_t>(pd, aa, pk, src_md, dst_md, p.groups,
                /* epsilon = */ 1e-4f, flags);
        pd_fwd_hint = std::make_shared<pd_t>(pd);

        EXPECT_ANY_THROW(group_normalization_forward(pd, {}));
        // default primitive ctor
        auto group_normalization = group_normalization_forward();
        // regular primitive ctor
        group_normalization = group_normalization_forward(pd);

        // check primitive kind is group_normalization
        ASSERT_TRUE(group_normalization.get_kind()
                == primitive::kind::group_normalization);
        // query for descs from pd
        const auto src_desc = pd.src_desc();
        const auto dst_desc = pd.dst_desc();
        // query for src_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC) == src_desc);
        if (p.src_tag != tag::any) { ASSERT_TRUE(src_md == src_desc); }
        // query for dst_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DST) == dst_desc);
        if (p.dst_tag != tag::any) { ASSERT_TRUE(dst_md == dst_desc); }
        // query for stats and scales via exec arg
        const auto mean_desc = pd.mean_desc();
        const auto variance_desc = pd.variance_desc();
        const auto scale_desc = pd.weights_desc();
        const auto shift_desc = pd.weights_desc();
        ASSERT_TRUE(
                pd.query_md(query::exec_arg_md, DNNL_ARG_MEAN) == mean_desc);
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_VARIANCE)
                == variance_desc);

        if (has_scale(flags)) {
            ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SCALE)
                    == scale_desc);
        }
        if (has_shift(flags)) {
            ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SHIFT)
                    == shift_desc);
        }

        // query for workspace
        const auto workspace_desc = pd.workspace_desc();
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_WORKSPACE)
                == workspace_desc);

        // query primitive parameters
        ASSERT_EQ(pd.get_prop_kind(), pk);
        ASSERT_EQ(pd.get_flags(), flags);
        ASSERT_EQ(pd.get_epsilon(), 1e-4f);
        ASSERT_EQ(pd.get_group_size(), p.groups);

        // check primitive returns zero_md for all rest md
        if (!has_scale(flags) && !has_shift(flags)) {
            ASSERT_TRUE(pd.weights_desc().is_zero());
        }
        ASSERT_TRUE(pd.diff_src_desc().is_zero());
        ASSERT_TRUE(pd.diff_dst_desc().is_zero());
        ASSERT_TRUE(pd.diff_weights_desc().is_zero());

        src = test::make_memory(src_desc, eng);
        auto dst = test::make_memory(dst_desc, eng);
        workspace = test::make_memory(workspace_desc, eng);
        mean = test::make_memory(mean_desc, eng);
        variance = test::make_memory(variance_desc, eng);
        scale = test::make_memory(scale_desc, eng);
        shift = test::make_memory(shift_desc, eng);

        fill_data(p.src_dt, src, 1, 1);
        if (has_scale(flags)) fill_data(dt::f32, scale, 1, 1);
        if (has_shift(flags)) fill_data(dt::f32, shift, 1, 1);
        if (use_global_stats(flags)) {
            fill_data(dt::f32, mean, 1, 1);
            fill_data(dt::f32, variance, 1, 1);
        }
        // test out-place mode
        group_normalization.execute(strm,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst},
                        {DNNL_ARG_MEAN, mean}, {DNNL_ARG_VARIANCE, variance},
                        {DNNL_ARG_SCALE, scale}, {DNNL_ARG_SHIFT, shift},
                        {DNNL_ARG_WORKSPACE, workspace}});
        strm.wait();

        // test in-place mode on forward
        if (p.src_tag == p.dst_tag && p.src_dt == p.dst_dt) {
            // TODO: add a copy of memory and result comparison with previous
            // dst output.
            group_normalization.execute(strm,
                    {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, src},
                            {DNNL_ARG_MEAN, mean},
                            {DNNL_ARG_VARIANCE, variance},
                            {DNNL_ARG_SCALE, scale}, {DNNL_ARG_SHIFT, shift},
                            {DNNL_ARG_WORKSPACE, workspace}});
            strm.wait();
        }
    }

    void Backward(prop_kind pk, normalization_flags flags) {
        SKIP_IF(unsupported_prop_kind(pk, p.src_dt, p.diff_src_dt, p.dst_dt),
                "Engine does not support this prop kind with this data types");
        // group_normalization specific types and values
        using pd_t = group_normalization_backward::primitive_desc;
        using hint_pd_t = group_normalization_forward::primitive_desc;
        allows_attr_t aa {false}; // doesn't support anything

        auto eng = get_test_engine();
        auto strm = make_stream(eng);

        auto diff_src_md = memory::desc(p.dims, p.diff_src_dt, p.diff_src_tag);
        auto diff_dst_md = memory::desc(p.dims, p.dst_dt, p.dst_tag);
        auto src_md = memory::desc(p.dims, p.src_dt, p.src_tag);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        pd = pd_t(eng, pk, diff_src_md, diff_dst_md, src_md, p.groups,
                /* epsilon = */ 1e-4f, flags, *pd_fwd_hint);
        // test all pd ctors
        test_bwd_pd_constructors<pd_t, hint_pd_t>(pd, *pd_fwd_hint, aa, pk,
                diff_src_md, diff_dst_md, src_md, p.groups,
                /* epsilon = */ 1e-4f, flags);

        EXPECT_ANY_THROW(group_normalization_backward(pd, {}));
        // default primitive ctor
        auto group_normalization = group_normalization_backward();
        // regular primitive ctor
        group_normalization = group_normalization_backward(pd);

        // check primitive kind is group_normalization
        ASSERT_TRUE(group_normalization.get_kind()
                == primitive::kind::group_normalization);

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

        // query for stats and scales via exec arg
        const auto mean_desc = pd.mean_desc();
        const auto variance_desc = pd.variance_desc();
        const auto scale_desc = pd.weights_desc();
        const auto diff_scale_desc = pd.diff_weights_desc();
        const auto diff_shift_desc = pd.diff_weights_desc();
        ASSERT_TRUE(
                pd.query_md(query::exec_arg_md, DNNL_ARG_MEAN) == mean_desc);
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_VARIANCE)
                == variance_desc);

        if (has_scale(flags)) {
            ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SCALE)
                    == scale_desc);
            ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SCALE)
                    == diff_scale_desc);
        }
        if (has_shift(flags)) {
            ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SHIFT)
                    == diff_shift_desc);
        }

        // query for workspace
        const auto workspace_desc = pd.workspace_desc();
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_WORKSPACE)
                == workspace_desc);

        // query primitive parameters
        ASSERT_EQ(pd.get_prop_kind(), pk);
        ASSERT_EQ(pd.get_flags(), flags);
        ASSERT_EQ(pd.get_epsilon(), 1e-4f);
        ASSERT_EQ(pd.get_group_size(), p.groups);

        // check primitive returns zero_md for all rest md
        ASSERT_TRUE(pd.dst_desc().is_zero());
        if (!has_scale(flags) && !has_shift(flags)) {
            ASSERT_TRUE(pd.weights_desc().is_zero());
            ASSERT_TRUE(pd.diff_weights_desc().is_zero());
        }

        auto diff_src = test::make_memory(diff_src_desc, eng);
        auto diff_dst = test::make_memory(diff_dst_desc, eng);
        auto diff_scale = test::make_memory(diff_scale_desc, eng);
        auto diff_shift = test::make_memory(diff_shift_desc, eng);

        fill_data(p.dst_dt, diff_dst, 2, 2);

        // test out-place mode
        group_normalization.execute(strm,
                {{DNNL_ARG_DIFF_SRC, diff_src}, {DNNL_ARG_DIFF_DST, diff_dst},
                        {DNNL_ARG_SRC, src}, {DNNL_ARG_MEAN, mean},
                        {DNNL_ARG_VARIANCE, variance},
                        {DNNL_ARG_DIFF_SCALE, diff_scale},
                        {DNNL_ARG_DIFF_SHIFT, diff_shift},
                        {DNNL_ARG_SCALE, scale},
                        {DNNL_ARG_WORKSPACE, workspace}});
        strm.wait();

        // test in-place mode
        if (p.dst_tag == p.diff_src_tag && p.dst_dt == p.diff_src_dt) {
            group_normalization.execute(strm,
                    {{DNNL_ARG_DIFF_SRC, diff_dst},
                            {DNNL_ARG_DIFF_DST, diff_dst}, {DNNL_ARG_SRC, src},
                            {DNNL_ARG_MEAN, mean},
                            {DNNL_ARG_VARIANCE, variance},
                            {DNNL_ARG_DIFF_SCALE, diff_scale},
                            {DNNL_ARG_DIFF_SHIFT, diff_shift},
                            {DNNL_ARG_SCALE, scale},
                            {DNNL_ARG_WORKSPACE, workspace}});
            strm.wait();
        }
    }

    void Test() {
        using nf = normalization_flags;
        std::vector<normalization_flags> inference_flags {nf::none,
                nf::use_global_stats, nf::use_scale, nf::use_shift,
                nf::use_global_stats | nf::use_scale,
                nf::use_global_stats | nf::use_shift,
                nf::use_scale | nf::use_shift,
                nf::use_global_stats | nf::use_scale | nf::use_shift};

        for (auto flags : inference_flags) {
            Forward(prop_kind::forward_inference, flags);
        }

        // No training for int8.
        if (p.src_dt == dt::s8) return;

        std::vector<normalization_flags> training_flags {nf::none,
                nf::use_scale, nf::use_shift, nf::use_scale | nf::use_shift};

        for (auto flags : training_flags) {
            Forward(prop_kind::forward_training, flags);

            if (p.diff_src_dt != dt::undef) {
                SKIP_IF(unsupported_data_type(p.diff_src_dt),
                        "Engine does not support this data type.");

                const prop_kind bwd_pk = (has_scale(flags) || has_shift(flags))
                        ? prop_kind::backward
                        : prop_kind::backward_data;
                Backward(bwd_pk, flags);
            }
        }
    }

    bool use_global_stats(normalization_flags flags) const {
        return static_cast<bool>(flags & normalization_flags::use_global_stats);
    }

    bool has_scale(normalization_flags flags) const {
        return static_cast<bool>(flags & normalization_flags::use_scale);
    }

    bool has_shift(normalization_flags flags) const {
        return static_cast<bool>(flags & normalization_flags::use_shift);
    }

    bool is_training(prop_kind pk) const {
        return pk == prop_kind::forward_training;
    }
};

using tp = group_normalization_test_params_t;

TEST_P(group_normalization_test_t, TestsGroupNormalization) {}

INSTANTIATE_TEST_SUITE_P(Test_GroupNormalization_EF, group_normalization_test_t,
        ::testing::Values(
                // Negative dims
                tp {dt::f32, dt::f32, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, -2, 128, 256}, 1, true,
                        dnnl_invalid_arguments},
                // Invalid groups
                tp {dt::f32, dt::f32, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, -2, true,
                        dnnl_invalid_arguments},
                tp {dt::f32, dt::f32, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 3, true,
                        dnnl_invalid_arguments},
                tp {dt::f32, dt::f32, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, 5, 128, 256}, 2, true,
                        dnnl_invalid_arguments},
                // Tag for src on forward is not specified
                tp {dt::f32, dt::f32, dt::undef, tag::any, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 2, true,
                        dnnl_invalid_arguments},
                // Tag for src on backward is not specified
                tp {dt::f32, dt::f32, dt::f32, tag::any, tag::nchw, tag::nchw,
                        {2, 2, 128, 256}, 2, true, dnnl_invalid_arguments},
                // Data type for src is not specified
                tp {dt::undef, dt::f32, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 1, true,
                        dnnl_invalid_arguments}));

static auto all_cases = [](memory::data_type src_dt, memory::data_type dst_dt,
                                memory::data_type diff_src_dt) {
    return ::testing::Values(
            tp {src_dt, dst_dt, diff_src_dt, tag::nCdhw16c, tag::nCdhw16c,
                    tag::nCdhw16c, {2, 17, 5, 4, 4}, 17},
            tp {src_dt, dst_dt, diff_src_dt, tag::ncdhw, tag::ncdhw, tag::ncdhw,
                    {2, 7, 3, 4, 4}, 7},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw16c, tag::nChw16c,
                    tag::nChw16c, {2, 17, 4, 4}, 17},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 7, 4, 4}, 7},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {2, 10, 4, 4}, 5},
            tp {src_dt, dst_dt, diff_src_dt, tag::nhwc, tag::nhwc, tag::nhwc,
                    {2, 10, 4, 4}, 2},
            tp {src_dt, dst_dt, diff_src_dt, tag::nCw8c, tag::nCw8c, tag::nCw8c,
                    {2, 7, 4}, 7},
            tp {src_dt, dst_dt, diff_src_dt, tag::nwc, tag::nwc, tag::nwc,
                    {2, 10, 4}, 1});
}; // namespace dnnl

#define EXPAND_DTS(src, dst, diff_src) \
    memory::data_type::src, memory::data_type::dst, memory::data_type::diff_src

#define INST_TEST_CASE(name, suite, ...) \
    INSTANTIATE_TEST_SUITE_P( \
            name, group_normalization_test_t, suite(__VA_ARGS__));

#define CPU_INST_TEST_CASE(name, suite, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P( \
            name, group_normalization_test_t, suite(__VA_ARGS__));

#define GPU_INST_TEST_CASE(name, suite, ...) \
    GPU_INSTANTIATE_TEST_SUITE_P( \
            name, group_normalization_test_t, suite(__VA_ARGS__));

INST_TEST_CASE(
        GroupNormalizationSimpleF32, all_cases, EXPAND_DTS(f32, f32, f32));
INST_TEST_CASE(
        GroupNormalizationSimpleBF16, all_cases, EXPAND_DTS(bf16, bf16, bf16));
INST_TEST_CASE(GroupNormalizationSimpleF32BF16, all_cases,
        EXPAND_DTS(f32, bf16, bf16));
INST_TEST_CASE(
        GroupNormalizationSimpleF16, all_cases, EXPAND_DTS(f16, f16, undef));
INST_TEST_CASE(
        GroupNormalizationSimpleF16F32, all_cases, EXPAND_DTS(f16, f32, undef));
INST_TEST_CASE(
        GroupNormalizationSimpleS8, all_cases, EXPAND_DTS(s8, s8, undef));
INST_TEST_CASE(
        GroupNormalizationSimpleF32S8, all_cases, EXPAND_DTS(f32, s8, undef));
INST_TEST_CASE(
        GroupNormalizationSimpleS8F32, all_cases, EXPAND_DTS(s8, f32, undef));
} // namespace dnnl
