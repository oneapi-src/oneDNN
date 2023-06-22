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

struct eltwise_test_params_t {
    dt src_dt;
    dt dst_dt;
    dt diff_src_dt;
    tag src_tag;
    tag dst_tag;
    tag diff_src_tag;
    memory::dims dims;
    float alpha, beta;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

bool cuda_check_format_tag(tag atag) {
    // Blocking is not supported by cuDNN
    return !impl::utils::one_of(
            atag, tag::aBcd8b, tag::aBcd16b, tag::aBcde8b, tag::aBcde16b);
}

template <typename... Rest>
bool cuda_check_format_tag(tag first_tag, Rest... rest_tags) {
    const bool ok = cuda_check_format_tag(first_tag);
    if (!ok) return ok;
    return cuda_check_format_tag(rest_tags...);
}

bool hip_check_format_tag(tag atag) {
    // HIP has the same limitations for `tag` as CUDA.
    return cuda_check_format_tag(atag);
}

template <typename... Rest>
bool hip_check_format_tag(tag first_tag, Rest... rest_tags) {
    const bool ok = hip_check_format_tag(first_tag);
    if (!ok) return ok;
    return hip_check_format_tag(rest_tags...);
}

class eltwise_test_t : public ::testing::TestWithParam<eltwise_test_params_t> {
private:
    eltwise_test_params_t p;
    memory src, dst;
    std::shared_ptr<eltwise_forward::primitive_desc> pd_fwd_hint;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<eltwise_test_params_t>::GetParam();

        SKIP_IF(unsupported_data_type(p.src_dt, p.dst_dt),
                "Engine does not support this data type.");

        SKIP_IF_CUDA(
                p.dst_dt == dt::s8, "Unsupported int8 destination data type");
        SKIP_IF_HIP(p.src_dt == dt::s8, "Unsupported int8 source data type");

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
        // eltwise specific types and values
        using pd_t = eltwise_forward::primitive_desc;

        auto eng = get_test_engine();
        auto strm = make_stream(eng);

        auto aa = allows_attr_t {false};
        aa.po_binary = true;

        auto src_md = memory::desc(p.dims, p.src_dt, p.src_tag);
        auto dst_md = memory::desc(p.dims, p.dst_dt, p.dst_tag);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        pd = pd_t(eng, pk, aalgorithm, src_md, dst_md, p.alpha, p.beta);
        // test all pd ctors
        test_fwd_pd_constructors<pd_t>(
                pd, aa, pk, aalgorithm, src_md, dst_md, p.alpha, p.beta);
        pd_fwd_hint = std::make_shared<pd_t>(pd);

        EXPECT_ANY_THROW(eltwise_forward(pd, {}));
        // default primitive ctor
        auto eltwise = eltwise_forward();
        // regular primitive ctor
        eltwise = eltwise_forward(pd);

        // check primitive kind is eltwise
        ASSERT_TRUE(eltwise.get_kind() == primitive::kind::eltwise);
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
        ASSERT_EQ(pd.get_alpha(), p.alpha);
        ASSERT_EQ(pd.get_beta(), p.beta);

        // check primitive returns zero_md for all rest md
        ASSERT_TRUE(pd.weights_desc().is_zero());
        ASSERT_TRUE(pd.diff_src_desc().is_zero());
        ASSERT_TRUE(pd.diff_dst_desc().is_zero());
        ASSERT_TRUE(pd.diff_weights_desc().is_zero());

        src = test::make_memory(src_desc, eng);
        dst = test::make_memory(dst_desc, eng);

        fill_data(p.src_dt, src, 1, 1);
        // test out-place mode
        eltwise.execute(strm, {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
        strm.wait();

        // test in-place mode on forward
        if (p.src_tag == p.dst_tag && p.src_dt == p.dst_dt) {
            // TODO: add a copy of memory and result comparison with previous
            // dst output.
            eltwise.execute(strm, {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, src}});
            strm.wait();
        }
    }

    void Backward(algorithm aalgorithm) {
        // eltwise specific types and values
        using pd_t = eltwise_backward::primitive_desc;
        using hint_pd_t = eltwise_forward::primitive_desc;
        allows_attr_t aa {false}; // doesn't support anything

        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        auto diff_src_md = memory::desc(p.dims, p.diff_src_dt, p.diff_src_tag);
        auto diff_dst_md = memory::desc(p.dims, p.dst_dt, p.dst_tag);
        auto src_md = memory::desc(p.dims, p.src_dt, p.src_tag);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        pd = pd_t(eng, aalgorithm, diff_src_md, diff_dst_md, src_md, p.alpha,
                p.beta, *pd_fwd_hint);
        // test all pd ctors
        test_bwd_pd_constructors<pd_t, hint_pd_t>(pd, *pd_fwd_hint, aa,
                aalgorithm, diff_src_md, diff_dst_md, src_md, p.alpha, p.beta);

        EXPECT_ANY_THROW(eltwise_backward(pd, {}));
        // default primitive ctor
        auto eltwise = eltwise_backward();
        // regular primitive ctor
        eltwise = eltwise_backward(pd);

        // check primitive kind is eltwise
        ASSERT_TRUE(eltwise.get_kind() == primitive::kind::eltwise);

        // query for descs from pd
        const auto diff_src_desc = pd.diff_src_desc();
        const auto diff_dst_desc = pd.diff_dst_desc();
        const auto src_desc = pd.src_desc();
        const auto dst_desc = pd.dst_desc();
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
        // query for src_desc via exec arg
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC) == src_desc);
        ASSERT_TRUE((p.src_tag != tag::any && src_md == src_desc)
                || (p.dst_tag != tag::any
                        && pd_fwd_hint.get()->dst_desc() == dst_desc));

        // query primitive parameters
        ASSERT_EQ(pd.get_prop_kind(), prop_kind::backward_data);
        ASSERT_EQ(pd.get_algorithm(), aalgorithm);
        ASSERT_EQ(pd.get_alpha(), p.alpha);
        ASSERT_EQ(pd.get_beta(), p.beta);

        // check primitive returns zero_md for all rest md
        ASSERT_TRUE(pd.weights_desc().is_zero());
        ASSERT_TRUE(pd.dst_desc().is_zero() || pd.src_desc().is_zero());
        ASSERT_TRUE(pd.diff_weights_desc().is_zero());

        auto diff_src = test::make_memory(diff_src_desc, eng);
        auto diff_dst = test::make_memory(diff_dst_desc, eng);

        fill_data(p.diff_src_dt, diff_dst, 2, 2);

        // test out-place mode
        eltwise.execute(strm,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst},
                        {DNNL_ARG_DIFF_DST, diff_dst},
                        {DNNL_ARG_DIFF_SRC, diff_src}});
        strm.wait();

        // test in-place mode
        if (p.dst_tag == p.diff_src_tag && p.dst_dt == p.diff_src_dt) {
            eltwise.execute(strm,
                    {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst},
                            {DNNL_ARG_DIFF_DST, diff_dst},
                            {DNNL_ARG_DIFF_SRC, diff_dst}});
            strm.wait();
        }
    }

    void Test() {
        const bool is_int8 = p.src_dt == dt::s8 || p.src_dt == dt::u8;
        std::vector<prop_kind> pks = {is_int8 ? prop_kind::forward_inference
                                              : prop_kind::forward_training};

        std::vector<algorithm> algs_all = {algorithm::eltwise_relu,
                algorithm::eltwise_tanh, algorithm::eltwise_elu,
                algorithm::eltwise_square, algorithm::eltwise_abs,
                algorithm::eltwise_sqrt, algorithm::eltwise_swish,
                algorithm::eltwise_linear, algorithm::eltwise_soft_relu,
                algorithm::eltwise_mish, algorithm::eltwise_logistic,
                algorithm::eltwise_exp, algorithm::eltwise_gelu_tanh,
                algorithm::eltwise_gelu_erf, algorithm::eltwise_log,
                algorithm::eltwise_clip, algorithm::eltwise_clip_v2,
                algorithm::eltwise_pow, algorithm::eltwise_round,
                algorithm::eltwise_hardswish, algorithm::eltwise_hardsigmoid,
                algorithm::eltwise_relu_use_dst_for_bwd,
                algorithm::eltwise_tanh_use_dst_for_bwd,
                algorithm::eltwise_elu_use_dst_for_bwd,
                algorithm::eltwise_sqrt_use_dst_for_bwd,
                algorithm::eltwise_logistic_use_dst_for_bwd,
                algorithm::eltwise_exp_use_dst_for_bwd,
                algorithm::eltwise_clip_v2_use_dst_for_bwd};
        // TODO: generalize this function.
        if (p.src_dt != dt::f32) {
            auto it = algs_all.begin();
            while (true) {
                if (*it == algorithm::eltwise_round) {
                    algs_all.erase(it);
                    break;
                }
                it++;
                if (it == algs_all.end()) break;
            }
        }

        std::vector<algorithm> algs_int8
                = {algorithm::eltwise_relu, algorithm::eltwise_linear};
        const auto &algs = is_int8 ? algs_int8 : algs_all;

        for_(auto pk : pks)
        for (auto alg : algs) {
            SKIP_FOR_LOOP_CUDA(is_fwd(pk)
                            && !impl::utils::one_of(alg,
                                    algorithm::eltwise_relu,
                                    algorithm::eltwise_tanh,
                                    algorithm::eltwise_elu,
                                    algorithm::eltwise_logistic),
                    "Unsupported algorithm type for CUDA");
            SKIP_FOR_LOOP_CUDA(alg == algorithm::eltwise_relu && p.alpha != 0.f,
                    "Unsupported combination of algorithm type and alpha "
                    "parameter for CUDA");

            SKIP_FOR_LOOP_HIP(
                    !impl::utils::one_of(alg, algorithm::eltwise_relu,
                            algorithm::eltwise_tanh, algorithm::eltwise_elu,
                            algorithm::eltwise_logistic,
                            algorithm::eltwise_soft_relu,
                            algorithm::eltwise_abs),
                    "Unsupported algorithm type for HIP");

            Forward(pk, alg);

            bool to_continue = pk != prop_kind::forward_training
                    || p.diff_src_dt == dt::undef
                    || alg == algorithm::eltwise_round;
            if (to_continue) continue;

            SKIP_FOR_LOOP_CUDA(
                    !impl::utils::one_of(alg, algorithm::eltwise_relu),
                    "Unsupported algorithm type for CUDA");

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

            SKIP_IF_CUDA(p.src_tag != p.diff_src_tag && p.src_tag != tag::any
                            && p.diff_src_tag != tag::any,
                    "Unsupported different memory formats for diff_source "
                    "and "
                    "diff_destination");
            SKIP_IF_HIP(p.src_tag != p.diff_src_tag && p.src_tag != tag::any
                            && p.diff_src_tag != tag::any,
                    "Unsupported different memory formats for diff_source "
                    "and diff_destination");

            Backward(alg);
        }
    }

    bool is_fwd(prop_kind pk) const {
        return pk == prop_kind::forward_training
                || pk == prop_kind::forward_inference;
    }
};

using tp = eltwise_test_params_t;

TEST_P(eltwise_test_t, TestsEltwise) {}

INSTANTIATE_TEST_SUITE_P(Test_Eltwise_EF, eltwise_test_t,
        ::testing::Values(
                // Negative dims
                tp {dt::f32, dt::f32, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, -2, 128, 256}, 1.f, 2.f, true,
                        dnnl_invalid_arguments},
                // Tag for src on forward is not specified
                tp {dt::f32, dt::f32, dt::undef, tag::any, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 1.f, 2.f, true,
                        dnnl_invalid_arguments},
                // Tag for src on backward is not specified
                tp {dt::f32, dt::f32, dt::f32, tag::any, tag::nchw, tag::nchw,
                        {2, 2, 128, 256}, 1.f, 2.f, true,
                        dnnl_invalid_arguments},
                // Data type for src is not specified
                tp {dt::undef, dt::f32, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 1.f, 2.f, true,
                        dnnl_invalid_arguments},
                // Different data types are not supported
                tp {dt::f32, dt::bf16, dt::undef, tag::nchw, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 1.f, 2.f, true,
                        dnnl_unimplemented},
                // Different data types are not supported
                tp {dt::f32, dt::bf16, dt::f32, tag::nchw, tag::nchw, tag::nchw,
                        {2, 2, 128, 256}, 1.f, 2.f, true, dnnl_unimplemented},
                // Different memory formats are not supported
                tp {dt::f32, dt::f32, dt::undef, tag::nchw, tag::nhwc,
                        tag::undef, {2, 2, 128, 256}, 1.f, 2.f, true,
                        dnnl_unimplemented},
                // Different memory formats are not supported
                tp {dt::f32, dt::f32, dt::f32, tag::nchw, tag::nhwc, tag::nchw,
                        {2, 2, 128, 256}, 1.f, 2.f, true, dnnl_unimplemented}));

static auto all_cases = [](dt src_dt, dt dst_dt, dt diff_src_dt) {
    return ::testing::Values(tp {src_dt, dst_dt, diff_src_dt, tag::nwc,
                                     tag::nwc, tag::nwc, {2, 16, 10}, 0.f, 0.f},
            tp {src_dt, dst_dt, diff_src_dt, tag::ncw, tag::ncw, tag::ncw,
                    {2, 64, 27}, 1.f, 2.f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nhwc, tag::nhwc, tag::nhwc,
                    {2, 16, 10, 8}, 0.f, 0.9f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nchw, tag::nchw, tag::nchw,
                    {2, 64, 27, 27}, 1.f, 2.f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw8c, tag::nChw8c,
                    tag::nChw8c, {2, 16, 16, 8}, 0.1f, 0.9f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nChw16c, tag::nChw16c,
                    tag::nChw16c, {2, 16, 4, 4}, 0.f, 0.f},
            tp {src_dt, dst_dt, diff_src_dt, tag::ncdhw, tag::ncdhw, tag::ncdhw,
                    {2, 64, 7, 7, 7}, 1.f, 1.f},
            tp {src_dt, dst_dt, diff_src_dt, tag::ncdhw, tag::ncdhw, tag::ncdhw,
                    {10, 10, 10, 10, 10}, 0.f, 0.f},
            tp {src_dt, dst_dt, diff_src_dt, tag::nCdhw16c, tag::nCdhw16c,
                    tag::nCdhw16c, {4, 15, 2, 2, 2}, 0.1f, 0.2f});
};

#define EXPAND_DTS(src, dst, diff_src) \
    memory::data_type::src, memory::data_type::dst, memory::data_type::diff_src

#define INST_TEST_CASE(name, suite, ...) \
    INSTANTIATE_TEST_SUITE_P(name, eltwise_test_t, suite(__VA_ARGS__));

#define CPU_INST_TEST_CASE(name, suite, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P(name, eltwise_test_t, suite(__VA_ARGS__));

#define GPU_INST_TEST_CASE(name, suite, ...) \
    GPU_INSTANTIATE_TEST_SUITE_P(name, eltwise_test_t, suite(__VA_ARGS__));

INST_TEST_CASE(EltwiseSimpleF32, all_cases, EXPAND_DTS(f32, f32, f32));
INST_TEST_CASE(EltwiseSimpleBF16, all_cases, EXPAND_DTS(bf16, bf16, bf16));
INST_TEST_CASE(EltwiseSimpleF16, all_cases, EXPAND_DTS(f16, f16, undef));
INST_TEST_CASE(EltwiseSimpleU8, all_cases, EXPAND_DTS(u8, u8, undef));
} // namespace dnnl
