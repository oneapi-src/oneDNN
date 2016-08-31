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

struct test_bnorm_desc_t {
    uint32_t mb, c;
    uint32_t h, w;
    double eps;
};

template <typename data_t>
void check_bnorm_fwd(test_bnorm_desc_t bnd, memory &src, memory &dst, memory &scaleshift, memory &workspace)
{

    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();
    data_t *ws_data = (data_t *)workspace.get_data_handle();
    data_t *ss_data = (data_t *)scaleshift.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();
    const memory::desc ws_d = workspace.get_primitive_desc().desc();
    const memory::desc ss_d = scaleshift.get_primitive_desc().desc();

#pragma omp parallel for
    for (uint32_t c = 0; c < bnd.c; c++)
    {
        ws_data[map_index(ws_d, c)] = 0.0;
        for (uint32_t n = 0; n < bnd.mb; n++)
        for (uint32_t h = 0; h < bnd.h; h++)
        for (uint32_t w = 0; w < bnd.w; w++)
        {
             uint32_t sidx = n*bnd.c*bnd.h*bnd.w + c*bnd.h*bnd.w + h*bnd.w + w;
             ws_data[map_index(ws_d, c)] += src_data[map_index(src_d, sidx)];
        }
        ws_data[map_index(ws_d, c)] /= bnd.mb*bnd.h*bnd.w;

        ws_data[map_index(ws_d, bnd.c + c)] = 0.0;
        for (uint32_t n = 0; n < bnd.mb; n++)
        for (uint32_t h = 0; h < bnd.h; h++)
        for (uint32_t w = 0; w < bnd.w; w++)
        {
             uint32_t sidx = n*bnd.c*bnd.h*bnd.w + c*bnd.h*bnd.w + h*bnd.w + w;
             data_t tmp = src_data[map_index(src_d, sidx)] - ws_data[map_index(ws_d, c)];
             ws_data[map_index(ws_d, bnd.c + c)] += tmp*tmp;
        }
        ws_data[map_index(ws_d, bnd.c + c)] = ws_data[map_index(ws_d, bnd.c + c)]/(bnd.mb*bnd.h*bnd.w) + bnd.eps;
        ws_data[map_index(ws_d, bnd.c + c)] = (data_t)1.0/sqrt(ws_data[map_index(ws_d, bnd.c + c)]);

        for (uint32_t n = 0; n < bnd.mb; n++)
        for (uint32_t h = 0; h < bnd.h; h++)
        for (uint32_t w = 0; w < bnd.w; w++)
        {
             uint32_t sdidx = n*bnd.c*bnd.h*bnd.w + c*bnd.h*bnd.w + h*bnd.w + w;
             data_t ref_dst = ss_data[map_index(ss_d, c)] * (src_data[map_index(src_d, sdidx)] - ws_data[map_index(ws_d, c)]) * ws_data[map_index(ws_d, bnd.c + c)] + ss_data[map_index(ss_d, bnd.c + c)];
             data_t out = dst_data[map_index(dst_d, sdidx)];
             data_t eps = 1.e-6*bnd.mb*bnd.h*bnd.w;
             data_t norm_max = std::max(fabs(out), fabs(ref_dst));
             if (norm_max < eps)
                 norm_max = 1.;
             EXPECT_NEAR((out - ref_dst) / norm_max, 0., eps);
        }
    }
}

struct bnorm_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    memory::format src_format;
    memory::format dst_format;
    memory::format scaleshift_format;
    test_bnorm_desc_t test_bnd;
};

template <typename data_t>
class bnorm_test : public ::testing::TestWithParam<bnorm_test_params> {
protected:
    virtual void SetUp()
    {
        bnorm_test_params p
                = ::testing::TestWithParam<bnorm_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu
                || p.engine_kind == engine::kind::cpu_lazy);
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        auto eng = engine(p.engine_kind, 0);
        memory::precision prec = data_traits<data_t>::prec;
        ASSERT_EQ(prec, mkldnn::memory::precision::f32);

        test_bnorm_desc_t bnd = p.test_bnd;

        auto bn_src_desc = create_md({ bnd.mb, bnd.c, bnd.h, bnd.w }, prec, p.src_format);
        auto bn_scaleshift_desc = create_md({ 2, bnd.c }, prec, memory::format::nc);
        auto bn_dst_desc = create_md({ bnd.mb, bnd.c, bnd.h, bnd.w }, prec, p.dst_format);

        auto bnorm_desc = batch_normalization::desc(p.aprop_kind, bn_src_desc, bn_dst_desc, bn_scaleshift_desc, bnd.eps);

        auto bnorm_prim_desc = batch_normalization::primitive_desc(bnorm_desc, eng);

        auto bn_ws_desc = bnorm_prim_desc.data.workspace_primitive_desc;
        auto bn_src = memory(memory::primitive_desc(bn_src_desc, eng));
        auto bn_ws = memory(memory::primitive_desc(bn_ws_desc));
        auto bn_scaleshift = memory(memory::primitive_desc(bn_scaleshift_desc, eng));
        auto bn_dst = memory(memory::primitive_desc(bn_dst_desc, eng));

        fill_data<data_t>(bn_src.get_primitive_desc().get_number_of_elements(),(data_t *)bn_src.get_data_handle());

        fill_data<data_t>(bn_scaleshift.get_primitive_desc().get_number_of_elements(),(data_t *)bn_scaleshift.get_data_handle());

        auto bn = batch_normalization(bnorm_prim_desc, bn_src, bn_dst, bn_scaleshift, bn_ws);

        std::vector<primitive> pipeline;
        pipeline.push_back(bn);

        stream().submit(pipeline).wait();

        auto test_ws = memory(memory::primitive_desc(bn_ws_desc));
        check_bnorm_fwd<data_t>(bnd, bn_src, bn_dst, bn_scaleshift, test_ws);
    }
};

using bnorm_test_float = bnorm_test<float>;
using bnorm_test_params_float = bnorm_test_params;

TEST_P(bnorm_test_float, TestsBNorm)
{
}

INSTANTIATE_TEST_CASE_P(TestBNormForward, bnorm_test_float,
        ::testing::Values(
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw, memory::format::x,
            { 2, 10, 4, 4, 1.0e-4} }));
INSTANTIATE_TEST_CASE_P(TestBNormGoogleNetForwardNCHW, bnorm_test_float,
        ::testing::Values(
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  112,   64,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  56,    64,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  56,    192,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  28,    96,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  28,    16,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  28,    64,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  28,    128,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  28,    32,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  28,    96,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    96,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    16,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    192,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    208,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    48,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    64,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    112,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    24,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    160,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    224,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  4,     128,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    128,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    512,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    256,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    144,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    32,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    228,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    528,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  14,    320,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  7,     160,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  7,     32,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  7,     256,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  7,     320,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  7,     128,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  7,     192,  0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  7,     48,   0.1 } },
            bnorm_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::x, { 2,  7,     384,  0.1 } }));
}
