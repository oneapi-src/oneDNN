/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "mkldnn.hpp"

namespace mkldnn {

enum {ACROSS=0,WITHIN=1};

struct test_lrn_desc_t {
    uint32_t mb, c;
    uint32_t h, w;
    double alpha, beta;
    uint32_t local_size;
    int32_t kind; // 0 ac, 1 wc
};

template <typename data_t>
void check_lrn_fwd(test_lrn_desc_t ld, memory &src, memory &dst)
{
    data_t *src_ptr = (data_t *)src.get_data_handle();
    data_t *dst_ptr = (data_t *)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

    const uint32_t C = ld.c;
    const uint32_t H = ld.h;
    const uint32_t W = ld.w;
    const uint32_t size = ld.local_size;
    const uint32_t CSIZE = ld.kind == ACROSS ? size : 1;
    const uint32_t HWSIZE = size + 1 - CSIZE;
    const uint32_t summands = ld.kind == ACROSS ? size : size*size;

    auto off = [=](uint32_t n, uint32_t c, uint32_t h, uint32_t w)
    {
        return ((n * ld.c + c) * ld.h + h) * ld.w + w;
    };

    auto ker = [=](data_t *d, uint32_t n, uint32_t oc, uint32_t oh, uint32_t ow)
    {
        data_t sum = 0.0;
        for (uint32_t c = oc; c < oc + CSIZE; ++c) {
            if (c < (CSIZE - 1) / 2) continue;
            if (c >= C + (CSIZE - 1) / 2) continue;
            for (uint32_t h = oh; h < oh + HWSIZE; ++h) {
                if (h < (HWSIZE - 1) / 2) continue;
                if (h >= H + (HWSIZE - 1) / 2) continue;
                for (uint32_t w = ow; w < ow + HWSIZE; ++w) {
                    if (w < (HWSIZE - 1) / 2) continue;
                    if (w >= W + (HWSIZE - 1) / 2) continue;
                    data_t s = src_ptr[map_index(src_d,off(n, c - (CSIZE - 1) / 2, h - (HWSIZE - 1) / 2, w - (HWSIZE - 1) / 2))];
                    sum += s * s;
                }
            }
        }
        data_t norm_coef = powf(1 + ld.alpha * sum / summands, ld.beta);
        data_t ref_out = src_ptr[map_index(src_d, off(n, oc, oh, ow))]/norm_coef;
        data_t eps = 1.e-7*(2*summands+5);
        data_t out = d[0];
        data_t norm_max = std::max(fabs(out), fabs(ref_out));
        if (norm_max < eps) norm_max = 1.;
        EXPECT_NEAR(out, ref_out, eps*norm_max);
    };

    const uint32_t N = ld.mb;
#   pragma omp parallel for collapse(4)
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t c = 0; c < C; ++c) {
            for (uint32_t h = 0; h < H; ++h) {
                for (uint32_t w = 0; w < W; ++w) {
                    ker(&dst_ptr[map_index(dst_d,off(n, c, h, w))], n, c, h, w);
                }
            }
        }
    }
}

struct lrn_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    lrn::algorithm aalgorithm;
    memory::format src_format;
    memory::format dst_format;
    test_lrn_desc_t test_ld;
};

template <typename data_t>
class lrn_test : public ::testing::TestWithParam<lrn_test_params> {
protected:
    virtual void SetUp()
    {
        lrn_test_params p
                = ::testing::TestWithParam<lrn_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu
                || p.engine_kind == engine::kind::cpu_lazy);
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        auto eng = engine(p.engine_kind, 0);
        memory::precision prec = data_traits<data_t>::prec;
        ASSERT_EQ(prec, mkldnn::memory::precision::f32);

        test_lrn_desc_t ld = p.test_ld;

        auto l_src_desc
                = create_md({ ld.mb, ld.c, ld.h, ld.w }, prec, p.src_format);
        auto l_dst_desc
                = create_md({ ld.mb, ld.c, ld.h, ld.w }, prec, p.dst_format);

        auto lrn_desc = lrn::desc(p.aprop_kind, p.aalgorithm, l_src_desc,
                l_dst_desc, ld.alpha, ld.beta, ld.local_size);

        auto lrn_prim_desc = lrn::primitive_desc(lrn_desc, eng);

        auto l_scr_desc = lrn_prim_desc.data.scratch_primitive_desc;
        auto l_src = memory(memory::primitive_desc(l_src_desc, eng));
        auto l_scr = memory(memory::primitive_desc(l_scr_desc));
        auto l_dst = memory(memory::primitive_desc(l_dst_desc, eng));

        fill_data<data_t>(l_src.get_primitive_desc().get_number_of_elements(),
                (data_t *)l_src.get_data_handle());

        auto l = lrn(lrn_prim_desc, l_src, l_scr, l_dst);

        std::vector<primitive> pipeline;
        pipeline.push_back(l);

        stream().submit(pipeline).wait();

        check_lrn_fwd<data_t>(ld, l_src, l_dst);
    }
};

using lrn_test_float = lrn_test<float>;
using lrn_test_params_float = lrn_test_params;

TEST_P(lrn_test_float, TestsLRN)
{
}

INSTANTIATE_TEST_CASE_P(TestLRNForward, lrn_test_float,
        ::testing::Values(
            lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, lrn::across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 10, 4, 4, 1.0e-4, 0.75, 5, ACROSS } },
            lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, lrn::across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 10, 4, 4, 1.0e-4, 0.75, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(TestLRNForwardNHWC, lrn_test_float,
        ::testing::Values(
            lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, lrn::across_channels, memory::format::nhwc,
            memory::format::nhwc, { 2, 10, 4, 4, 1.0e-4, 0.75, 5, ACROSS } },
            lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, lrn::across_channels, memory::format::nhwc,
            memory::format::nhwc, { 2, 10, 4, 4, 1.0e-4, 0.75, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(TestLRNForwardBlocked, lrn_test_float,
        ::testing::Values(
            lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, lrn::across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 4, 4, 1.0e-4, 0.75, 5, ACROSS } },
            lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, lrn::across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 4, 4, 1.0e-4, 0.75, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNAlexnetForwardNCHW, lrn_test_float,
        ::testing::Values(
            lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, lrn::across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 96, 55, 55, 1.0e-4, 0.75, 5, ACROSS } },
            lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, lrn::across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 96, 55, 55, 1.0e-4, 0.75, 5, ACROSS } },
            lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, lrn::across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 256, 27, 27, 1.0e-4, 0.75, 5, ACROSS } },
            lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, lrn::across_channels, memory::format::nchw,
            memory::format::nchw, { 2, 256, 27, 27, 1.0e-4, 0.75, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNAlexnetForwardNHWC, lrn_test_float,
        ::testing::Values(
                lrn_test_params_float{ prop_kind::forward_training,
                engine::kind::cpu, lrn::across_channels, memory::format::nhwc,
                memory::format::nhwc, { 2, 96, 55, 55, 1.0e-4, 0.75, 5, ACROSS } },
                lrn_test_params_float{ prop_kind::forward_scoring,
                engine::kind::cpu, lrn::across_channels, memory::format::nhwc,
                memory::format::nhwc, { 2, 96, 55, 55, 1.0e-4, 0.75, 5, ACROSS } },
                lrn_test_params_float{ prop_kind::forward_training,
                engine::kind::cpu, lrn::across_channels, memory::format::nhwc,
                memory::format::nhwc, { 2, 256, 27, 27, 1.0e-4, 0.75, 5, ACROSS } },
                lrn_test_params_float{ prop_kind::forward_scoring,
                engine::kind::cpu, lrn::across_channels, memory::format::nhwc,
                memory::format::nhwc, { 2, 256, 27, 27, 1.0e-4, 0.75, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNAlexnetForwardBlocked, lrn_test_float,
        ::testing::Values(
            lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, lrn::across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4, 0.75, 5, ACROSS } },
            lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, lrn::across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4, 0.75, 5, ACROSS } },
            lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, lrn::across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4, 0.75, 5, ACROSS } },
            lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, lrn::across_channels, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4, 0.75, 5, ACROSS } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestLRNRCNNForwardBlocked, lrn_test_float,
        ::testing::Values(
            lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, lrn::within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4, 0.75, 3, WITHIN } }
            , lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, lrn::within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 96, 55, 55, 1.0e-4, 0.75, 3, WITHIN } },
            lrn_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, lrn::within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4, 0.75, 3, WITHIN } },
            lrn_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, lrn::within_channel, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 256, 27, 27, 1.0e-4, 0.75, 3, WITHIN } }
            ));
}
