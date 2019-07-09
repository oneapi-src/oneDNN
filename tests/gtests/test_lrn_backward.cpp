/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include <cmath>

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "cpu_isa_traits.hpp"
#include "mkldnn.hpp"

namespace mkldnn {

using fmt = memory::format_tag;

struct test_lrn_desc_t {
    memory::dim mb, c;
    memory::dim h, w;
    memory::dim local_size;
    float alpha, beta, k;
};

struct lrn_test_params {
    prop_kind aprop_kind;
    algorithm aalgorithm;
    memory::format_tag data_format;
    memory::format_tag diff_data_format;
    test_lrn_desc_t test_ld;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

template <typename data_t, typename acc_data_t = data_t>
void check_lrn_fwd(
        const lrn_test_params &p, const memory &src, const memory &dst) {
    auto src_ptr = map_memory<data_t>(src);
    auto dst_ptr = map_memory<data_t>(dst);

    const memory::dim C = p.test_ld.c;
    const memory::dim H = p.test_ld.h;
    const memory::dim W = p.test_ld.w;
    const memory::dim size = p.test_ld.local_size;
    const memory::dim CSIZE
            = p.aalgorithm == algorithm::lrn_across_channels ? size : 1;
    const memory::dim HWSIZE = size + 1 - CSIZE;
    const memory::dim summands = p.aalgorithm == algorithm::lrn_across_channels
            ? size
            : size * size;
    auto padded_c = src.get_desc().data.padded_dims[1];

    const memory::desc src_d = src.get_desc();
    const memory::desc dst_d = dst.get_desc();
    const mkldnn::impl::memory_desc_wrapper src_mdw(src_d.data);
    const mkldnn::impl::memory_desc_wrapper dst_mdw(dst_d.data);

    auto off = [=](memory::dim n, memory::dim c, memory::dim h, memory::dim w) {
        return ((n * padded_c + c) * p.test_ld.h + h) * p.test_ld.w + w;
    };

    auto ker = [&](data_t *d, memory::dim n, memory::dim oc, memory::dim oh,
                       memory::dim ow) {
        acc_data_t sum = 0.0;
        for (memory::dim c = oc; c < oc + CSIZE; ++c) {
            if (c < (CSIZE - 1) / 2)
                continue;
            if (c >= C + (CSIZE - 1) / 2)
                continue;
            for (memory::dim h = oh; h < oh + HWSIZE; ++h) {
                if (h < (HWSIZE - 1) / 2)
                    continue;
                if (h >= H + (HWSIZE - 1) / 2)
                    continue;
                for (memory::dim w = ow; w < ow + HWSIZE; ++w) {
                    if (w < (HWSIZE - 1) / 2)
                        continue;
                    if (w >= W + (HWSIZE - 1) / 2)
                        continue;
                    acc_data_t s = src_ptr[src_mdw.off_l(
                            off(n, c - (CSIZE - 1) / 2, h - (HWSIZE - 1) / 2,
                                    w - (HWSIZE - 1) / 2),
                            true)];
                    sum += s * s;
                }
            }
        }

        auto const norm_coef = std::pow(
                p.test_ld.k + p.test_ld.alpha * sum / summands, p.test_ld.beta);
        acc_data_t ref_out = static_cast<acc_data_t>(
                src_ptr[src_mdw.off_l(off(n, oc, oh, ow), true)] / norm_coef);
        acc_data_t eps = static_cast<acc_data_t>(1.e-7f * (2 * summands + 5));
        memory::data_type data_type = data_traits<data_t>::data_type;
        if (data_type == mkldnn::memory::data_type::bf16)
            eps = static_cast<acc_data_t>(1.e-3f * (2 * summands + 5));
        acc_data_t out = d[0];
        acc_data_t norm_max = (std::max)(fabs(out), fabs(ref_out));
        if (norm_max < eps)
            norm_max = 1.;
        ASSERT_NEAR(out, ref_out, eps * norm_max);
    };

    const memory::dim N = p.test_ld.mb;
    mkldnn::impl::parallel_nd(N, padded_c, H, W,
            [&](memory::dim n, memory::dim c, memory::dim h, memory::dim w) {
                ker(&dst_ptr[dst_mdw.off_l(off(n, c, h, w), true)], n, c, h, w);
            });
}

template <typename data_t, typename acc_data_t = data_t>
void check_lrn_bwd(const lrn_test_params &p, const memory &src,
        const memory &diff_dst, const memory &diff_src) {
    ASSERT_TRUE(p.aalgorithm == algorithm::lrn_across_channels);

    auto src_ptr = map_memory<data_t>(src);
    auto diff_dst_ptr = map_memory<data_t>(diff_dst);
    auto diff_src_ptr = map_memory<data_t>(diff_src);

    const memory::dim MB = p.test_ld.mb;
    const memory::dim C = p.test_ld.c;
    const memory::dim H = p.test_ld.h;
    const memory::dim W = p.test_ld.w;
    const memory::dim local_size = p.test_ld.local_size;
    auto padded_c = src.get_desc().data.padded_dims[1];

    data_t *ref_diff_src_ptr = new data_t[MB * (padded_c)*H * W];

    const memory::desc src_d = src.get_desc();
    const memory::desc diff_dst_d = diff_dst.get_desc();
    const memory::desc diff_src_d = diff_src.get_desc();
    const mkldnn::impl::memory_desc_wrapper src_mdw(src_d.data);
    const mkldnn::impl::memory_desc_wrapper diff_dst_mdw(diff_dst_d.data);
    const mkldnn::impl::memory_desc_wrapper diff_src_mdw(diff_src_d.data);

    auto off = [=](memory::dim n, memory::dim c, memory::dim h, memory::dim w) {
        return ((n * padded_c + c) * H + h) * W + w;
    };

    auto get_omega = [=](acc_data_t c_k, memory::dim kernel_size, float alpha,
                             memory::dim C, const data_t *src, memory::dim n,
                             memory::dim c, memory::dim h, memory::dim w) {
        acc_data_t sum = 0.0;

        memory::dim half_kernel_size = (kernel_size - 1) / 2;
        memory::dim c_start = (c < half_kernel_size) ? 0 : c - half_kernel_size;
        memory::dim c_end = c + kernel_size - half_kernel_size;
        c_end = c_end < C ? c_end : C;
        for (memory::dim i = c_start; i < c_end; ++i) {
            acc_data_t value = src[src_mdw.off_l(off(n, i, h, w))];
            sum += value * value;
        }
        sum *= alpha / kernel_size;
        return c_k + sum;
    };

    auto ker = [&](data_t *d, memory::dim mb, memory::dim oc, memory::dim oh,
                       memory::dim ow) {
        const float alpha = p.test_ld.alpha;
        const float beta = p.test_ld.beta;
        const float k = p.test_ld.k;
        const memory::dim kernel_size = p.test_ld.local_size;
        memory::dim ks_start = kernel_size / 2 > oc ? kernel_size / 2 - oc : 0;
        memory::dim ks_stop = C - oc <= kernel_size / 2
                ? C - oc + kernel_size / 2
                : kernel_size;

        acc_data_t A = 0, B = 0, omega_mid = 0;

        for (memory::dim ks = ks_start; ks < ks_stop; ks++) {
            memory::dim _t = oc + ks - (kernel_size / 2);
            acc_data_t omega = get_omega(static_cast<acc_data_t>(k),
                    kernel_size, alpha, C, src_ptr, mb, _t, oh, ow);

            if (ks == kernel_size / 2)
                omega_mid = omega;

            acc_data_t t = src_ptr[src_mdw.off_l(off(mb, _t, oh, ow), true)]
                    / powf((float)omega, (float)beta);
            B += (1.0f / omega) * t
                    * diff_dst_ptr[diff_dst_mdw.off_l(
                              off(mb, _t, oh, ow), true)];
        }

        A = (1.0f / powf((float)omega_mid, (float)beta))
                * diff_dst_ptr[diff_dst_mdw.off_l(off(mb, oc, oh, ow), true)];
        B *= src_ptr[src_mdw.off_l(off(mb, oc, oh, ow), true)];
        B *= (2.0f * alpha * beta) / kernel_size;
        *d = A - B;
    };

    mkldnn::impl::parallel_nd(MB, C, H, W,
            [&](memory::dim mb, memory::dim c, memory::dim h, memory::dim w) {
                if (is_current_test_failed())
                    return;

                ker(&ref_diff_src_ptr[diff_src_mdw.off_l(
                            off(mb, c, h, w), true)],
                        mb, c, h, w);
                auto A = ref_diff_src_ptr[diff_src_mdw.off_l(
                        off(mb, c, h, w), true)];
                auto B = diff_src_ptr[diff_src_mdw.off_l(
                        off(mb, c, h, w), true)];
                acc_data_t eps = static_cast<acc_data_t>(1.e-6
                        * ((2 * (2 * local_size + 3) + 6) * local_size
                                  + (2 * local_size + 3) + 9));
                memory::data_type data_type = data_traits<data_t>::data_type;
                if (data_type == mkldnn::memory::data_type::bf16)
                    eps = static_cast<acc_data_t>(1.e-3f
                            * ((2 * (2 * local_size + 3) + 6) * local_size
                                      + (2 * local_size + 3) + 9));
                acc_data_t norm_max = (std::max)(fabs(A), fabs(B));
                if (norm_max < eps)
                    norm_max = 1.;
                ASSERT_NEAR(A, B, eps * norm_max);
            });

    delete[] ref_diff_src_ptr;
}

template <typename data_t>
class lrn_test : public ::testing::TestWithParam<lrn_test_params>
{
private:
    std::shared_ptr<test_memory> src;
    std::shared_ptr<test_memory> dst;
    std::shared_ptr<test_memory> diff_src;
    std::shared_ptr<test_memory> diff_dst;
    memory workspace;
    std::shared_ptr<memory::desc> src_desc;
    std::shared_ptr<memory::desc> dst_desc;
    std::shared_ptr<memory::desc> diff_src_desc;
    std::shared_ptr<memory::desc> diff_dst_desc;
    lrn_forward::primitive_desc lrn_fwd_prim_desc;
    lrn_test_params p;
    memory::dims padR;
    engine eng;
    stream strm;
    memory::data_type data_type;
    bool is_training;

protected:
    virtual void SetUp() {
        data_type = data_traits<data_t>::data_type;
        SKIP_IF(data_type == memory::data_type::bf16
                && get_test_engine_kind() == engine::kind::gpu,
                "GPU does not support bf16 data type.");
        SKIP_IF(data_type == memory::data_type::bf16
                && !impl::cpu::mayiuse(impl::cpu::avx512_core),
                "ISA does not support bf16 data type.");

        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();

        eng = engine(get_test_engine_kind(), 0);
        strm = stream(eng);
        ASSERT_EQ(true,
                mkldnn::impl::utils::one_of(data_type,
                        mkldnn::memory::data_type::f32,
                        mkldnn::memory::data_type::bf16));

        test_lrn_desc_t ld = p.test_ld;

        src_desc.reset(new memory::desc(
                { ld.mb, ld.c, ld.h, ld.w }, data_type, p.data_format));
        dst_desc.reset(new memory::desc(
                { ld.mb, ld.c, ld.h, ld.w }, data_type, p.data_format));
        diff_src_desc.reset(new memory::desc(
                { ld.mb, ld.c, ld.h, ld.w }, data_type, p.diff_data_format));
        diff_dst_desc.reset(new memory::desc(
                { ld.mb, ld.c, ld.h, ld.w }, data_type, p.diff_data_format));

        is_training = p.aprop_kind == prop_kind::forward_training;

        Forward();
        if (is_training)
            Backward();
    }

    void Forward() {
        auto lrn_desc = lrn_forward::desc(p.aprop_kind, p.aalgorithm, *src_desc,
                p.test_ld.local_size, p.test_ld.alpha, p.test_ld.beta,
                p.test_ld.k);
        lrn_fwd_prim_desc = lrn_forward::primitive_desc(lrn_desc, eng);

        src.reset(new test_memory(*src_desc, eng));
        dst.reset(new test_memory(*dst_desc, eng));

        fill_data<data_t>(src->get_size() / sizeof(data_t), src->get());
        fill_data<data_t>(dst->get_size() / sizeof(data_t), dst->get());
        check_zero_tail<data_t>(1, src->get());
        check_zero_tail<data_t>(1, dst->get());

        // Execute
        auto l = lrn_forward(lrn_fwd_prim_desc);
        std::unordered_map<int, memory> args = { { MKLDNN_ARG_SRC, src->get() },
            { MKLDNN_ARG_DST, dst->get() } };
        if (is_training) {
            auto workspace_md = lrn_fwd_prim_desc.workspace_desc();
            workspace = memory(workspace_md, eng);
            args.insert({ MKLDNN_ARG_WORKSPACE, workspace });
        }
        l.execute(strm, args);
        strm.wait();

        check_zero_tail<data_t>(0, dst->get());

        check_lrn_fwd<data_t, float>(p, src->get(), dst->get());
    }

    void Backward() {
        auto lrn_desc = lrn_backward::desc(p.aalgorithm, *src_desc,
                *diff_dst_desc, p.test_ld.local_size, p.test_ld.alpha,
                p.test_ld.beta, p.test_ld.k);

        src.reset(new test_memory(*src_desc, eng));
        diff_src.reset(new test_memory(*diff_src_desc, eng));
        diff_dst.reset(new test_memory(*diff_dst_desc, eng));

        auto lrn_prim_desc = lrn_backward::primitive_desc(
                lrn_desc, eng, lrn_fwd_prim_desc);

        fill_data<data_t>(src->get_size() / sizeof(data_t), src->get());

        fill_data<data_t>(
                diff_dst->get_size() / sizeof(data_t), diff_dst->get());

        fill_data<data_t>(
                diff_src->get_size() / sizeof(data_t), diff_src->get());
        check_zero_tail<data_t>(1, src->get());
        check_zero_tail<data_t>(1, diff_dst->get());
        check_zero_tail<data_t>(1, diff_src->get());

        // Execute
        lrn_backward(lrn_prim_desc)
                .execute(strm,
                        { { MKLDNN_ARG_SRC, src->get() },
                                { MKLDNN_ARG_DIFF_DST, diff_dst->get() },
                                { MKLDNN_ARG_WORKSPACE, workspace },
                                { MKLDNN_ARG_DIFF_SRC, diff_src->get() } });
        strm.wait();

        check_zero_tail<data_t>(0, diff_src->get());

        check_lrn_bwd<data_t, float>(
                p, src->get(), diff_dst->get(), diff_src->get());
    }
};

static auto padded_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, { 0, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 0, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nChw16c, { 2, 16, 0, 4, 5, 1.0e-4f, 0.75f, 3.0f } });
};

static auto EF_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, { -1, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f }, true,
                    mkldnn_invalid_arguments },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 2, -10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f }, true,
                    mkldnn_invalid_arguments },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, { 2, 10, -4, 4, 5, 1.0e-4f, 0.75f, 3.0f }, true,
                    mkldnn_invalid_arguments });
};

static auto nChw16c_padded_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 17, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 19, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 26, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 12, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f } });
};

static auto nChw8c_padded_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 7, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 9, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 26, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 12, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f } });
};

static auto cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 4.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 4.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 20, 12, 7, 7, 3, 1.0e-2f, 0.5f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 20, 12, 7, 7, 3, 1.0e-2f, 0.5f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 20, 12, 7, 7, 3, 1.0e-2f, 0.5f, 6.5f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 20, 12, 7, 7, 3, 1.0e-2f, 0.5f, 6.5f } });
};

static auto NHWC_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 3.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 3.0f } });
};

static auto nChw8c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 1, 8, 1, 1, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 1, 8, 1, 1, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 1, 8, 1, 1, 5, 1.0e-4f, 0.75f, 2.2f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 1, 8, 1, 1, 5, 1.0e-4f, 0.75f, 2.2f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 1, 32, 5, 5, 3, 1.0e-2f, 0.7f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 1, 32, 5, 5, 3, 1.0e-2f, 0.7f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 1, 32, 5, 5, 3, 1.0e-2f, 0.7f, 0.1f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 1, 32, 5, 5, 3, 1.0e-2f, 0.7f, 0.1f } });
};

static auto nChw16c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 1, 16, 1, 1, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 1, 16, 1, 1, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 1, 16, 1, 1, 5, 1.0e-4f, 0.75f, 2.2f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 1, 16, 1, 1, 5, 1.0e-4f, 0.75f, 2.2f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 1, 32, 5, 5, 3, 1.0e-2f, 0.7f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 1, 32, 5, 5, 3, 1.0e-2f, 0.7f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 1, 32, 5, 5, 3, 1.0e-2f, 0.7f, 0.1f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 1, 32, 5, 5, 3, 1.0e-2f, 0.7f, 0.1f } });
};

static auto CaffeNCHW_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, { 2, 4, 5, 5, 5, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 2, 4, 5, 5, 5, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, { 2, 4, 5, 5, 5, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 2, 4, 5, 5, 5, 1.0f, 0.75f, 1.0f } });
};

static auto CaffeNHWC_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 4, 5, 5, 5, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 4, 5, 5, 5, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 4, 5, 5, 5, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 4, 5, 5, 5, 1.0f, 0.75f, 1.0f } });
};

static auto Caffe_nChw8c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 96, 55, 55, 3, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 96, 55, 55, 3, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 96, 55, 55, 3, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 96, 55, 55, 3, 1.0f, 0.75f, 1.0f } });
};

static auto Caffe_nChw16c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 96, 55, 55, 3, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 96, 55, 55, 3, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 96, 55, 55, 3, 1.0f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 96, 55, 55, 3, 1.0f, 0.75f, 1.0f } });
};
static auto AlexnetNCHW_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f } });
};

static auto AlexnetNHWC_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nhwc,
                    fmt::nhwc, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f } });
};

static auto Alexnet_nChw8c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f } });
};

static auto Alexnet_nChw16c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c,
                    { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f } });
};

static auto GoogleNetV1NCHW_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nchw,
                    fmt::nchw, { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } });
};

static auto GoogleNetV1_nChw8c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw8c,
                    fmt::nChw8c, { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } });
};

static auto GoogleNetV1_nChw16c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } },
            lrn_test_params{ prop_kind::forward_scoring, lk, fmt::nChw16c,
                    fmt::nChw16c,
                    { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } });
};

static auto RegressionWeightFormat_cases = [](algorithm lk) {
    return ::testing::Values(lrn_test_params{ prop_kind::forward_training,
            lk, fmt::oihw, fmt::oihw,
            { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f } });
};

#define INST_TEST_CASE(test, lk)                                              \
    TEST_P(test, TestsLRN) {}                                                 \
    INSTANTIATE_TEST_SUITE_P(Backward_padded, test, padded_cases(lk));        \
    INSTANTIATE_TEST_SUITE_P(BackwardEF, test, EF_cases(lk));             \
    INSTANTIATE_TEST_SUITE_P(                                                 \
            Backward_nChw16c_padded, test, nChw16c_padded_cases(lk));         \
    INSTANTIATE_TEST_SUITE_P(                                                 \
            Backward_nChw8c_padded, test, nChw8c_padded_cases(lk));           \
    INSTANTIATE_TEST_SUITE_P(LRN, test, cases(lk));                           \
    INSTANTIATE_TEST_SUITE_P(NHWC, test, NHWC_cases(lk));                     \
    INSTANTIATE_TEST_SUITE_P(nChw8c, test, nChw8c_cases(lk));                 \
    INSTANTIATE_TEST_SUITE_P(nChw16c, test, nChw16c_cases(lk));               \
    INSTANTIATE_TEST_SUITE_P(CaffeNCHW, test, CaffeNCHW_cases(lk));           \
    INSTANTIATE_TEST_SUITE_P(CaffeNHWC, test, CaffeNHWC_cases(lk));           \
    INSTANTIATE_TEST_SUITE_P(Caffe_nChw8c, test, Caffe_nChw8c_cases(lk));     \
    INSTANTIATE_TEST_SUITE_P(Caffe_nChw16c, test, Caffe_nChw16c_cases(lk));   \
    INSTANTIATE_TEST_SUITE_P(AlexnetNCHW, test, AlexnetNCHW_cases(lk));       \
    INSTANTIATE_TEST_SUITE_P(AlexnetNHWC, test, AlexnetNHWC_cases(lk));       \
    INSTANTIATE_TEST_SUITE_P(Alexnet_nChw8c, test, Alexnet_nChw8c_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(                                                 \
            Alexnet_nChw16c, test, Alexnet_nChw16c_cases(lk));                \
    INSTANTIATE_TEST_SUITE_P(                                                 \
            GoogleNetV1NCHW, test, GoogleNetV1NCHW_cases(lk));                \
    INSTANTIATE_TEST_SUITE_P(                                                 \
            GoogleNetV1_nChw8c, test, GoogleNetV1_nChw8c_cases(lk));          \
    INSTANTIATE_TEST_SUITE_P(                                                 \
            GoogleNetV1_nChw16c, test, GoogleNetV1_nChw16c_cases(lk));        \
    INSTANTIATE_TEST_SUITE_P(RegressionWeightFormat, test,                    \
            RegressionWeightFormat_cases(                                     \
                    lk)); // This tests compatibility with MKL-DNN 0.14

using float_across = lrn_test<float>;
using float_within = lrn_test<float>;
using bfloat16_across = lrn_test<bfloat16_t>;
using bfloat16_within = lrn_test<bfloat16_t>;

INST_TEST_CASE(float_across, algorithm::lrn_across_channels)
INST_TEST_CASE(bfloat16_across, algorithm::lrn_across_channels)

//INST_TEST_CASE(float_within, algorithm::lrn_within_channel)
//INST_TEST_CASE(bfloat16_within, algorithm::lrn_within_channel)

} // namespace mkldnn
