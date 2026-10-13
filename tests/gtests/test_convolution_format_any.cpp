/*******************************************************************************
* Copyright 2016 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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
        // Blocked-vs-plain format check is not reliable on GPU, optimized
        // implementation can use plain format with "any".
        SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
                "blocked format check is not reliable on GPU");
#if DNNL_X64
        // Skip this test if the library cannot select blocked format a priori.
        // Currently blocking is supported only for avx2 and later CPUs.
        bool implementation_supports_blocking = dnnl::mayiuse(cpu_isa::avx2);
        if (!implementation_supports_blocking) return;
#else
        return;
#endif

        auto p = ::testing::TestWithParam<
                conv_any_fmt_test_params_t>::GetParam();

        ASSERT_EQ(p.aprop_kind, prop_kind::forward);
        ASSERT_EQ(p.aalgorithm, algorithm::convolution_direct);
        auto eng = get_test_engine();
        memory::data_type data_type = data_traits_t<data_t>::data_type;
        SKIP_IF_CUDA((p.expected_src_fmt == BLK || p.expected_dst_fmt == BLK),
                "unsupported format");
        SKIP_IF_HIP((p.expected_src_fmt == BLK || p.expected_dst_fmt == BLK),
                "unsupported format");
        SKIP_IF_GENERIC(
                (p.expected_src_fmt == BLK || p.expected_dst_fmt == BLK),
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
#endif

#if DNNL_AARCH64
namespace {

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
struct scoped_omp_threads_t {
    explicit scoped_omp_threads_t(int nthr)
        : saved_nthr(omp_get_max_threads()) {
        omp_set_num_threads(nthr);
    }
    ~scoped_omp_threads_t() { omp_set_num_threads(saved_nthr); }

    const int saved_nthr;
};
#endif

bool is_brgconv(const convolution_forward::primitive_desc &pd) {
    const auto impl = std::string(pd.impl_info_str());
    return impl.find("brgconv:sve") != std::string::npos
            || impl.find("brgconv_1x1:sve") != std::string::npos;
}

convolution_forward::primitive_desc make_bf16_mmla_conv_pd(const engine &eng,
        const memory::desc &weights_md, memory::dim ih = 16,
        memory::dim iw = 16, memory::dim stride = 1, memory::dim padding = 1) {
    const auto weights_dims = weights_md.get_dims();
    const bool with_groups = weights_dims.size() == 5;
    const auto groups = with_groups ? weights_dims[0] : 1;
    const auto oc = groups * weights_dims[with_groups];
    const auto ic = groups * weights_dims[with_groups + 1];
    const auto kh = weights_dims[with_groups + 2];
    const auto kw = weights_dims[with_groups + 3];
    const auto oh = (ih + 2 * padding - kh) / stride + 1;
    const auto ow = (iw + 2 * padding - kw) / stride + 1;
    const auto src_md
            = memory::desc({1, ic, ih, iw}, memory::data_type::bf16, tag::nhwc);
    const auto dst_md
            = memory::desc({1, oc, oh, ow}, memory::data_type::f32, tag::nhwc);
    auto pd = convolution_forward::primitive_desc(eng,
            prop_kind::forward_inference, algorithm::convolution_direct, src_md,
            weights_md, memory::desc(), dst_md, {stride, stride},
            {padding, padding}, {padding, padding});
    // Skip ACL; reference is not a valid DOT fallback.
    while (!is_brgconv(pd) && pd.next_impl()) {}
    EXPECT_TRUE(is_brgconv(pd)) << pd.impl_info_str();
    return pd;
}

bool is_bf16_mmla_weights(const memory::desc &desc) {
    return aarch64_mmla_test::matches_weights_desc(desc, 1, 0);
}

} // namespace

TEST(AArch64MmlaConvolution, Bf16SelectionAndFallbacks) {
    SKIP_IF(get_test_engine_kind() != engine::kind::cpu,
            "This test targets the CPU convolution implementation.");

    auto eng = get_test_engine();
    SKIP_IF(!aarch64_mmla_test::has_bf16_mmla_impl(),
            "This test requires SVE128 or SVE256 with BF16.");

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    const scoped_omp_threads_t omp_threads(1);
#endif

    const auto any_3x3
            = memory::desc({64, 64, 3, 3}, memory::data_type::bf16, tag::any);
    const auto selected_3x3 = make_bf16_mmla_conv_pd(eng, any_3x3);
    for (const auto &pd :
            {selected_3x3, make_bf16_mmla_conv_pd(eng, any_3x3, 16, 16, 1, 0),
                    make_bf16_mmla_conv_pd(eng, any_3x3, 17, 19, 2, 1),
                    make_bf16_mmla_conv_pd(eng, any_3x3, 1, 1, 1, 1)}) {
        EXPECT_TRUE(is_bf16_mmla_weights(pd.weights_desc()));
    }

    // Horizontal padding above one selects DOT directly.
    const auto padding_fallback
            = make_bf16_mmla_conv_pd(eng, any_3x3, 16, 16, 1, 2);
    EXPECT_FALSE(is_bf16_mmla_weights(padding_fallback.weights_desc()));
    EXPECT_EQ(padding_fallback.src_desc(),
            memory::desc({1, 64, 16, 16}, memory::data_type::bf16, tag::nhwc));
    EXPECT_EQ(padding_fallback.dst_desc(),
            memory::desc({1, 64, 18, 18}, memory::data_type::f32, tag::nhwc));

    // A group dimension selects DOT even for G = 1.
    const auto grouped_3x3 = memory::desc(
            {1, 64, 64, 3, 3}, memory::data_type::bf16, tag::any);
    const auto grouped_fallback = make_bf16_mmla_conv_pd(eng, grouped_3x3);
    EXPECT_FALSE(is_bf16_mmla_weights(grouped_fallback.weights_desc()));

    using namespace impl::cpu::aarch64;
    const auto packed_tag
            = mayiuse(sve_256) ? tag::OhwI2i32o4i : tag::OhwI2i16o4i;
    const auto packed = memory::desc(
            any_3x3.get_dims(), memory::data_type::bf16, packed_tag);
    EXPECT_EQ(selected_3x3.weights_desc(), packed);
    const auto packed_pd = make_bf16_mmla_conv_pd(eng, packed);
    EXPECT_EQ(packed_pd.weights_desc(), packed);

#if DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL
    const auto any_1x1
            = memory::desc({64, 64, 1, 1}, memory::data_type::bf16, tag::any);
    const auto selected_1x1
            = make_bf16_mmla_conv_pd(eng, any_1x1, 11, 13, 1, 0);
    EXPECT_TRUE(is_bf16_mmla_weights(selected_1x1.weights_desc()));
    const auto packed_1x1 = memory::desc(
            any_1x1.get_dims(), memory::data_type::bf16, packed_tag);
    EXPECT_EQ(selected_1x1.weights_desc(), packed_1x1);
    EXPECT_EQ(make_bf16_mmla_conv_pd(eng, packed_1x1, 11, 13, 1, 0)
                      .weights_desc(),
            packed_1x1);

    const auto grouped_1x1 = memory::desc(
            {1, 64, 64, 1, 1}, memory::data_type::bf16, tag::any);
    const auto grouped_1x1_fallback
            = make_bf16_mmla_conv_pd(eng, grouped_1x1, 11, 13, 1, 0);
    EXPECT_FALSE(is_bf16_mmla_weights(grouped_1x1_fallback.weights_desc()));

    const auto stride_fallback
            = make_bf16_mmla_conv_pd(eng, any_1x1, 16, 16, 2, 0);
    EXPECT_FALSE(is_bf16_mmla_weights(stride_fallback.weights_desc()));

    // Neither input nor output channels may have an MMLA tile tail.
    for (const auto &dims :
            {memory::dims {64, 66, 1, 1}, memory::dims {65, 64, 1, 1}}) {
        SCOPED_TRACE(::testing::PrintToString(dims));
        const auto weights
                = memory::desc(dims, memory::data_type::bf16, tag::any);
        EXPECT_FALSE(is_bf16_mmla_weights(
                make_bf16_mmla_conv_pd(eng, weights, 8, 8, 1, 0)
                        .weights_desc()));
    }

    const auto dot = stride_fallback.weights_desc();
    const auto dot_pd = make_bf16_mmla_conv_pd(eng, dot, 16, 16, 1, 0);
    EXPECT_EQ(dot_pd.weights_desc(), dot);
    EXPECT_FALSE(is_bf16_mmla_weights(dot_pd.weights_desc()));
#endif
}

#endif
} // namespace dnnl
