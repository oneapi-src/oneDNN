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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#include "tests/test_isa_common.hpp"

namespace dnnl {

using tag = memory::format_tag;

enum class data_fmt_t { flat, blocked_cX };

#define FLT data_fmt_t::flat
#define BLK data_fmt_t::blocked_cX

struct conv_any_fmt_test_params_t {
    prop_kind aprop_kind;
    algorithm aalgorithm;
    data_fmt_t expected_src_fmt;
    data_fmt_t expected_dst_fmt;
    test_convolution_sizes_t test_cd;
};

template <typename data_t>
class convolution_any_fmt_test_t
    : public ::testing::TestWithParam<conv_any_fmt_test_params_t> {
protected:
    void SetUp() override {
#if DNNL_X64
        // Skip this test if the library cannot select blocked format a priori.
        // Currently blocking is supported only for sse41 and later CPUs.
        bool implementation_supports_blocking = dnnl::mayiuse(cpu_isa::sse41);
        if (!implementation_supports_blocking) return;
#else
        return;
#endif

        auto p = ::testing::TestWithParam<
                conv_any_fmt_test_params_t>::GetParam();

        ASSERT_EQ(p.aprop_kind, prop_kind::forward);
        ASSERT_EQ(p.aalgorithm, algorithm::convolution_direct);
        auto eng = get_test_engine();
        memory::data_type data_type = data_traits<data_t>::data_type;
        SKIP_IF_CUDA((p.expected_src_fmt == BLK || p.expected_dst_fmt == BLK),
                "unsupported format");
        SKIP_IF_HIP((p.expected_src_fmt == BLK || p.expected_dst_fmt == BLK),
                "unsupported format");
        ASSERT_EQ(data_type, dnnl::memory::data_type::f32);

        test_convolution_sizes_t cd = p.test_cd;

        auto c_src_desc
                = create_md({cd.mb, cd.ic, cd.ih, cd.iw}, data_type, tag::any);
        auto c_weights_desc = cd.ng > 1
                ? create_md({cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw},
                        data_type, tag::any)
                : create_md({cd.oc, cd.ic, cd.kh, cd.kw}, data_type, tag::any);
        auto c_dst_desc
                = create_md({cd.mb, cd.oc, cd.oh, cd.ow}, data_type, tag::any);

        auto conv_prim_desc = convolution_forward::primitive_desc(eng,
                p.aprop_kind, p.aalgorithm, c_src_desc, c_weights_desc,
                c_dst_desc, {cd.strh, cd.strw}, {cd.padh, cd.padw},
                {cd.padh, cd.padw});

        auto check_fmt = [&](const memory::desc &md, data_fmt_t expected) {
            bool ok = false;
            if (expected == FLT) {
                ok = true
                        && md.get_format_kind() == memory::format_kind::blocked
                        && md.get_inner_nblks() == 0;
            } else if (expected == BLK) {
                ok = true
                        && md.get_format_kind() == memory::format_kind::blocked
                        && md.get_inner_nblks() == 1
                        && md.get_inner_idxs()[0] == 1
                        && (false || md.get_inner_blks()[0] == 8
                                || md.get_inner_blks()[0] == 16);
            }
            return ok;
        };

        ASSERT_TRUE(check_fmt(conv_prim_desc.src_desc(), p.expected_src_fmt));
        ASSERT_TRUE(check_fmt(conv_prim_desc.dst_desc(), p.expected_dst_fmt));
    }
};

using conv_any_fmt_test_float = convolution_any_fmt_test_t<float>;

#define CPARAMS prop_kind::forward, algorithm::convolution_direct

using tf32 = conv_any_fmt_test_params_t;

#define ALEXNET_SUITE(EFMT) \
    tf32 {CPARAMS, FLT, EFMT, \
            {2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4}}, \
            tf32 {CPARAMS, EFMT, EFMT, \
                    {2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1}}, \
            tf32 {CPARAMS, EFMT, EFMT, \
                    {2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1}}, \
            tf32 {CPARAMS, EFMT, EFMT, \
                    {2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1}}, \
            tf32 { \
        CPARAMS, EFMT, EFMT, { \
            2, 2, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1, 1 \
        } \
    }

#if DNNL_X64
TEST_P(conv_any_fmt_test_float, TestsConvolutionAnyFmt) {}

CPU_INSTANTIATE_TEST_SUITE_P(TestConvolutionAlexnetAnyFmtForward,
        conv_any_fmt_test_float, ::testing::Values(ALEXNET_SUITE(BLK)));
GPU_INSTANTIATE_TEST_SUITE_P(TestConvolutionAlexnetAnyFmtForward,
        conv_any_fmt_test_float, ::testing::Values(ALEXNET_SUITE(BLK)));
#endif
} // namespace dnnl
