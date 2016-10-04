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
    int mb, c;
    int h, w;
    float eps;
};

template <typename data_t>
void check_bnorm_fwd(test_bnorm_desc_t &bnd,
        const memory &src, const memory &weights, memory &dst)
{
    const data_t *src_data = (const data_t *)src.get_data_handle();
    const data_t *weights_data = (const data_t *)weights.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();
    data_t *workspace_data = new data_t[2*bnd.c];

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

#pragma omp parallel for
    for (int c = 0; c < bnd.c; c++) {
        workspace_data[c] = data_t(0);
        for (int n = 0; n < bnd.mb; n++)
        for (int h = 0; h < bnd.h; h++)
            for (int w = 0; w < bnd.w; w++) {
                int sidx = n * bnd.c * bnd.h * bnd.w + c * bnd.h * bnd.w
                        + h * bnd.w + w;
                workspace_data[c] += src_data[map_index(src_d, sidx)];
            }
        workspace_data[c] /= bnd.mb * bnd.h * bnd.w;

        workspace_data[bnd.c + c] = data_t(0);
        for (int n = 0; n < bnd.mb; n++)
        for (int h = 0; h < bnd.h; h++)
            for (int w = 0; w < bnd.w; w++) {
                int sidx = n * bnd.c * bnd.h * bnd.w + c * bnd.h * bnd.w
                        + h * bnd.w + w;
                data_t tmp = src_data[map_index(src_d, sidx)]
                        - workspace_data[c];
                workspace_data[bnd.c + c] += tmp * tmp;
            }
        workspace_data[bnd.c + c] = workspace_data[bnd.c + c]
                / (bnd.mb * bnd.h * bnd.w) + bnd.eps;
        workspace_data[bnd.c + c] = data_t(1)
                / sqrt(workspace_data[bnd.c + c]);

        for (int n = 0; n < bnd.mb; n++)
        for (int h = 0; h < bnd.h; h++)
            for (int w = 0; w < bnd.w; w++) {
                int sdidx = n * bnd.c * bnd.h * bnd.w + c * bnd.h * bnd.w
                        + h * bnd.w + w;
                data_t ref_dst = weights_data[map_index(weights_d, c)]
                        * (src_data[map_index(src_d, sdidx)]
                        - workspace_data[c]) * workspace_data[bnd.c + c]
                        + weights_data[map_index(weights_d, bnd.c + c)];
                data_t out = dst_data[map_index(dst_d, sdidx)];
                data_t eps = 1.e-6 * bnd.mb * bnd.h * bnd.w;
                data_t norm_max = std::max(fabs(out), fabs(ref_dst));
                if (norm_max < eps) norm_max = data_t(1);
                EXPECT_NEAR((out - ref_dst) / norm_max, 0., eps);
            }
    }
    delete[] workspace_data;
}

struct bnorm_fwd_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    memory::format src_format;
    memory::format dst_format;
    memory::format weights_format;
    test_bnorm_desc_t test_bnd;
};

template <typename data_t>
class bnorm_forward_test : public ::testing::TestWithParam<bnorm_fwd_test_params> {
protected:
    virtual void SetUp()
    {
        bnorm_fwd_test_params p
                = ::testing::TestWithParam<bnorm_fwd_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        test_bnorm_desc_t bnd = p.test_bnd;
        bool with_workspace = p.aprop_kind == prop_kind::forward_training;

        auto src_desc = create_md(
                { bnd.mb, bnd.c, bnd.h, bnd.w }, data_type, p.src_format);
        auto dst_desc = create_md(
                { bnd.mb, bnd.c, bnd.h, bnd.w }, data_type, p.dst_format);

        auto src_primitive_desc = memory::primitive_desc(src_desc, eng);
        auto dst_primitive_desc = memory::primitive_desc(dst_desc, eng);

        auto src_size = src_primitive_desc.get_size();
        auto dst_size = dst_primitive_desc.get_size();

        // TODO: free
        data_t *src_data = new data_t[src_size];
        data_t *weights_data = nullptr;
        data_t *workspace_data = nullptr;
        data_t *dst_data = new data_t[dst_size];

        auto src = memory(src_primitive_desc, src_data);
        auto dst = memory(dst_primitive_desc, dst_data);

        auto bn_desc
            = batch_normalization_forward::desc(p.aprop_kind, src_desc, bnd.eps);
        auto bn_prim_desc = batch_normalization_forward::primitive_desc(bn_desc, eng);

        auto weights_primitive_desc = bn_prim_desc.weights_primitive_desc();
        auto weights_size = weights_primitive_desc.get_size();
        weights_data = new data_t[weights_size];
        auto weights = memory(weights_primitive_desc, weights_data);

        fill_data<data_t>(
                src.get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)src.get_data_handle());
        fill_data<data_t>(
                weights.get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)weights.get_data_handle());

        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);
        if (with_workspace) {
            auto workspace_primitive_desc =
                bn_prim_desc.workspace_primitive_desc();
            auto workspace_size = workspace_primitive_desc.get_size();
            workspace_data = new data_t[workspace_size];
            auto workspace = memory(workspace_primitive_desc, workspace_data);
            auto bn = batch_normalization_forward(bn_prim_desc,
                    src, weights, workspace, dst);
            pipeline.push_back(bn);
            s.submit(pipeline).wait();
        } else {
            auto bn = batch_normalization_forward(bn_prim_desc, src, weights, dst);
            pipeline.push_back(bn);
            s.submit(pipeline).wait();
        }

        check_bnorm_fwd<data_t>(bnd, src, weights, dst);
    }
};

using bnorm_forward_test_float = bnorm_forward_test<float>;
using bnorm_fwd_test_params_float = bnorm_fwd_test_params;

TEST_P(bnorm_forward_test_float, TestsBNorm)
{
}

INSTANTIATE_TEST_CASE_P(TestBNormForward, bnorm_forward_test_float,
        ::testing::Values(bnorm_fwd_test_params_float{ prop_kind::forward_training,
                engine::kind::cpu, memory::format::nchw, memory::format::nchw,
                memory::format::nc, { 2, 10, 4, 4, 0.1 } }));

INSTANTIATE_TEST_CASE_P(
        TestBNormForwardBlocked, bnorm_forward_test_float,
        ::testing::Values(
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 8, 4, 4, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 4, 4, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 8, 8, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 16, 8, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 16, 10, 0.1 } }));

INSTANTIATE_TEST_CASE_P(
        TestBNormGoogleNetForwardNCHW, bnorm_forward_test_float,
        ::testing::Values(
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 64, 112, 112, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 64, 56, 56, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 192, 56, 56, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 96, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 16, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 64, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 128, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 32, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 96, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 96, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 16, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 192, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 208, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 48, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 64, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 112, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 24, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 160, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 224, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 128, 4, 4, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 128, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 512, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 256, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 144, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 32, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 228, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 528, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 320, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 160, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 32, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 256, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 320, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 128, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 192, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 48, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 384, 7, 7, 0.1 } }));

INSTANTIATE_TEST_CASE_P(
        TestBNormGoogleNetForwardBlocked, bnorm_forward_test_float,
        ::testing::Values(
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 64, 112, 112, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 64, 56, 56, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 192, 56, 56, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 96, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 64, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 128, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 32, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 96, 28, 28, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 96, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 192, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 208, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 48, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 64, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 112, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 24, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 160, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 224, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 128, 4, 4, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 128, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 512, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 256, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 144, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 32, 14, 14, 0.1 } },
                /* size is not supported by nChw8c format yet
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 228, 14, 14, 0.1 } },
                */
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 528, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 320, 14, 14, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 160, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 32, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 256, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 320, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 128, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 192, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 48, 7, 7, 0.1 } },
                bnorm_fwd_test_params_float{ prop_kind::forward_training,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 384, 7, 7, 0.1 } }));
}
