/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

struct reduction_test_params_t {
    memory::format_tag src_format;
    memory::format_tag dst_format;
    algorithm aalgorithm;
    float p;
    float eps;
    memory::dims src_dims;
    memory::dims dst_dims;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename src_data_t, typename dst_data_t = src_data_t>
class reduction_test_t
    : public ::testing::TestWithParam<reduction_test_params_t> {
private:
    reduction_test_params_t p;
    memory::data_type src_dt, dst_dt;

protected:
    void SetUp() override {
        src_dt = data_traits<src_data_t>::data_type;
        dst_dt = data_traits<dst_data_t>::data_type;

        p = ::testing::TestWithParam<reduction_test_params_t>::GetParam();

        SKIP_IF(unsupported_data_type(src_dt),
                "Engine does not support this data type.");
#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
        SKIP_IF(get_test_engine().get_kind() != engine::kind::cpu,
                "Engine does not support this primitive.");
#endif
        SKIP_IF_CUDA(p.aalgorithm != algorithm::reduction_max
                        && p.aalgorithm != algorithm::reduction_min
                        && p.aalgorithm != algorithm::reduction_sum
                        && p.aalgorithm != algorithm::reduction_mul
                        && p.aalgorithm != algorithm::reduction_mean
                        && p.aalgorithm != algorithm::reduction_norm_lp_max
                        && p.aalgorithm
                                != algorithm::reduction_norm_lp_power_p_max
                        && p.eps != 0.0f,
                "Unsupported algorithm type for CUDA");
        SKIP_IF_GENERIC(!(generic_supported_format_tag(p.src_format)
                                && generic_supported_format_tag(p.dst_format)),
                "Unsupported format tag");

        catch_expected_failures(
                [&]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    bool generic_supported_format_tag(memory::format_tag tag) {
        return impl::utils::one_of(tag, impl::format_tag::a,
                impl::format_tag::ab, impl::format_tag::abc,
                impl::format_tag::abcd, impl::format_tag::abcde,
                impl::format_tag::abcdef, impl::format_tag::abdec,
                impl::format_tag::acb, impl::format_tag::acbde,
                impl::format_tag::acbdef, impl::format_tag::acdb,
                impl::format_tag::acdeb, impl::format_tag::ba,
                impl::format_tag::bac, impl::format_tag::bacd,
                impl::format_tag::bca, impl::format_tag::bcda,
                impl::format_tag::bcdea, impl::format_tag::cba,
                impl::format_tag::cdba, impl::format_tag::cdeba,
                impl::format_tag::decab, impl::format_tag::defcab);
    }

    void Test() {
        // reduction specific types and values
        using pd_t = reduction::primitive_desc;
        allows_attr_t allowed_attributes {false}; // doesn't support anything
        allowed_attributes.po_sum = true;
        allowed_attributes.po_eltwise = true;
        allowed_attributes.po_binary = true;

        auto eng = get_test_engine();
        auto strm = make_stream(eng);

        auto desc_src = memory::desc(p.src_dims, src_dt, p.src_format);
        auto desc_dst = memory::desc(p.dst_dims, dst_dt, p.dst_format);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        pd = pd_t(eng, p.aalgorithm, desc_src, desc_dst, p.p, p.eps);
        // test all pd ctors
        test_fwd_pd_constructors<pd_t>(pd, allowed_attributes, p.aalgorithm,
                desc_src, desc_dst, p.p, p.eps);

        EXPECT_ANY_THROW(reduction(pd, {}));
        // default primitive ctor
        auto prim = reduction();
        // regular primitive ctor
        prim = reduction(pd);

        const auto src_desc = pd.src_desc();
        const auto dst_desc = pd.dst_desc();

        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC) == src_desc);
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DST) == dst_desc);

        ASSERT_EQ(pd.get_algorithm(), p.aalgorithm);
        ASSERT_EQ(pd.get_p(), p.p);
        ASSERT_EQ(pd.get_epsilon(), p.eps);

        const auto test_engine = pd.get_engine();

        auto mem_src = memory(src_desc, test_engine);
        auto mem_dst = memory(dst_desc, test_engine);

        fill_data<src_data_t>(
                src_desc.get_size() / sizeof(src_data_t), mem_src);

        prim.execute(strm, {{DNNL_ARG_SRC, mem_src}, {DNNL_ARG_DST, mem_dst}});
        strm.wait();
    }
};

using tag = memory::format_tag;

static auto expected_failures = []() {
    return ::testing::Values(
            // The same src and dst dims
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_sum, 0.0f, 0.0f, {1, 1, 1, 4},
                    {1, 1, 1, 4}, true, dnnl_invalid_arguments},
            // not supported alg_kind
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::eltwise_relu, 0.0f, 0.0f, {1, 1, 1, 4},
                    {1, 1, 1, 4}, true, dnnl_invalid_arguments},
            // negative dim
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_sum, 0.0f, 0.0f, {-1, 1, 1, 4},
                    {-1, 1, 1, 1}, true, dnnl_invalid_arguments},
            // not supported p
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_norm_lp_max, 0.5f, 0.0f, {1, 8, 4, 4},
                    {1, 8, 4, 4}, true, dnnl_invalid_arguments},
            // invalid tag
            reduction_test_params_t {tag::any, tag::nchw,
                    algorithm::reduction_sum, 0.0f, 0.0f, {1, 1, 1, 4},
                    {1, 1, 1, 1}, true, dnnl_invalid_arguments});
};

static auto zero_dim = []() {
    return ::testing::Values(reduction_test_params_t {tag::nchw, tag::nchw,
            algorithm::reduction_sum, 0.0f, 0.0f, {0, 1, 1, 4}, {0, 1, 1, 1}});
};

static auto simple_cases = []() {
    return ::testing::Values(reduction_test_params_t {tag::nchw, tag::nchw,
                                     algorithm::reduction_sum, 0.0f, 0.0f,
                                     {1, 1, 1, 4}, {1, 1, 1, 1}},
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_max, 0.0f, 0.0f, {1, 1, 4, 4},
                    {1, 1, 1, 4}},
            reduction_test_params_t {tag::nChw16c, tag::nChw16c,
                    algorithm::reduction_min, 0.0f, 0.0f, {4, 4, 4, 4},
                    {1, 4, 4, 4}},
            reduction_test_params_t {tag::nChw16c, tag::nchw,
                    algorithm::reduction_sum, 0.0f, 0.0f, {4, 4, 4, 4},
                    {1, 4, 4, 1}},
            reduction_test_params_t {tag::nChw16c, tag::any,
                    algorithm::reduction_min, 0.0f, 0.0f, {4, 4, 4, 4},
                    {1, 1, 1, 1}});
};

static auto f32_cases = []() {
    return ::testing::Values(reduction_test_params_t {tag::nchw, tag::nchw,
                                     algorithm::reduction_norm_lp_max, 1.0f,
                                     0.0f, {1, 1, 1, 4}, {1, 1, 1, 1}},
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_norm_lp_power_p_max, 2.0f, 0.0f,
                    {1, 1, 1, 4}, {1, 1, 1, 1}},
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_mean, 0.0f, 0.0f, {1, 4, 4, 4},
                    {1, 1, 4, 4}});
};

#define INST_TEST_CASE(test) \
    TEST_P(test, TestsReduction) {} \
    INSTANTIATE_TEST_SUITE_P(TestReductionEF, test, expected_failures()); \
    INSTANTIATE_TEST_SUITE_P(TestReductionZero, test, zero_dim()); \
    INSTANTIATE_TEST_SUITE_P(TestReductionSimple, test, simple_cases());

#define INST_TEST_CASE_F32(test) \
    TEST_P(test, TestsReduction) {} \
    INSTANTIATE_TEST_SUITE_P(TestReductionEF, test, expected_failures()); \
    INSTANTIATE_TEST_SUITE_P(TestReductionZero, test, zero_dim()); \
    INSTANTIATE_TEST_SUITE_P(TestReductionSimple, test, simple_cases()); \
    INSTANTIATE_TEST_SUITE_P(TestReductionNorm, test, f32_cases());

using reduction_test_f32 = reduction_test_t<float>;
using reduction_test_bf16 = reduction_test_t<bfloat16_t>;
using reduction_test_f16 = reduction_test_t<float16_t>;
using reduction_test_s8 = reduction_test_t<int8_t>;
using reduction_test_u8 = reduction_test_t<uint8_t>;

INST_TEST_CASE_F32(reduction_test_f32)
INST_TEST_CASE(reduction_test_bf16)
INST_TEST_CASE(reduction_test_f16)
INST_TEST_CASE(reduction_test_s8)
INST_TEST_CASE(reduction_test_u8)

} // namespace dnnl
