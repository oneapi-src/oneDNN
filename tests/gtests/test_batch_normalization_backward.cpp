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

template <typename data_t>
void check_bnorm_bwd(test_bnorm_desc_t &bnd, prop_kind aprop_kind,
        const memory &src, const memory &diff_dst, const memory &weights,
        const memory &workspace, memory &diff_src, const memory &diff_weights)
{
    const data_t *src_data = (const data_t *)src.get_data_handle();
    const data_t *weights_data = (const data_t *)weights.get_data_handle();
    const data_t *diff_dst_data = (const data_t *)diff_dst.get_data_handle();
    const data_t *workspace_data = (const data_t *)workspace.get_data_handle();
    data_t *diff_src_data = (data_t *)diff_src.get_data_handle();
    const data_t *diff_weights_data = (const data_t *)diff_weights.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc workspace_d = workspace.get_primitive_desc().desc();
    const memory::desc diff_src_d = diff_src.get_primitive_desc().desc();
    const memory::desc diff_weights_d = diff_weights.get_primitive_desc().desc();

    auto ws_mean = &workspace_data[map_index(workspace_d, 0)];
    auto ws_variance = &workspace_data[map_index(workspace_d, bnd.c)];

    const data_t eps = 1.e-6 * bnd.mb * bnd.h * bnd.w;

#pragma omp parallel for
    for (int c = 0; c < bnd.c; c++) {
        data_t ref_diff_gamma = data_t(0);
        data_t ref_diff_beta = data_t(0);

        auto mean = ws_mean[c];
        auto variance = ws_variance[c];

        auto gamma = weights_data[map_index(weights_d, c)];

        for (int n = 0; n < bnd.mb; n++)
        for (int h = 0; h < bnd.h; h++)
            for (int w = 0; w < bnd.w; w++) {
                int sidx = n * bnd.c * bnd.h * bnd.w + c * bnd.h * bnd.w
                        + h * bnd.w + w;
                ref_diff_gamma += (src_data[map_index(src_d, sidx)] - mean)
                    * diff_dst_data[map_index(diff_dst_d, sidx)];
                ref_diff_beta += diff_dst_data[map_index(diff_dst_d, sidx)];
            }
        ref_diff_gamma *= variance;

        if (aprop_kind == backward) {
            auto diff_gamma = diff_weights_data[map_index(diff_weights_d, c)];
            auto diff_beta = diff_weights_data[map_index(diff_weights_d, bnd.c + c)];
            EXPECT_NEAR(diff_gamma, ref_diff_gamma, eps);
            EXPECT_NEAR(diff_beta, ref_diff_beta, eps);
        }

        for (int n = 0; n < bnd.mb; n++)
        for (int h = 0; h < bnd.h; h++)
            for (int w = 0; w < bnd.w; w++) {
                int sidx = n * bnd.c * bnd.h * bnd.w + c * bnd.h * bnd.w
                        + h * bnd.w + w;
                data_t ref_diff_src = diff_dst_data[map_index(diff_dst_d, sidx)]
                        - ref_diff_beta/(bnd.mb*bnd.h*bnd.w)
                        - (src_data[map_index(src_d, sidx)] - mean)
                        *ref_diff_gamma*variance/(bnd.mb*bnd.h*bnd.w);
                ref_diff_src *= gamma*variance;
                data_t out_diff_src = diff_src_data[map_index(diff_src_d, sidx)];
                data_t norm_max = std::max(fabs(out_diff_src), fabs(ref_diff_src));
                if (norm_max < eps) norm_max = data_t(1);
                EXPECT_NEAR((out_diff_src - ref_diff_src) / norm_max, 0., eps);
            }
    }
}

struct bnorm_bwd_test_params {
    prop_kind aprop_kind;
    engine::kind engine_kind;
    memory::format data_format;
    memory::format diff_format;
    memory::format weights_format;
    test_bnorm_desc_t test_bnd;
};

template <typename data_t>
class bnorm_backward_test : public ::testing::TestWithParam<bnorm_bwd_test_params> {
private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;
    std::shared_ptr<memory> diff_src;
    std::shared_ptr<memory> diff_dst;
    std::shared_ptr<memory> workspace;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> diff_weights;
    std::shared_ptr<memory::desc> data_desc;
    std::shared_ptr<memory::desc> diff_desc;
    std::shared_ptr<batch_normalization_forward::primitive_desc> bnrm_prim_desc;
    bnorm_bwd_test_params p;
    std::shared_ptr<engine> eng;
    memory::data_type data_type;

protected:
    virtual void SetUp()
    {
        p = ::testing::TestWithParam<bnorm_bwd_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_TRUE(p.aprop_kind == prop_kind::backward
                || p.aprop_kind == prop_kind::backward_data);
        eng.reset(new engine(p.engine_kind, 0));
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        test_bnorm_desc_t bnd = p.test_bnd;

        data_desc.reset(new memory::desc(
                { bnd.mb, bnd.c, bnd.h, bnd.w }, data_type, p.data_format));
        diff_desc.reset(new memory::desc(
                { bnd.mb, bnd.c, bnd.h, bnd.w }, data_type, p.data_format));

        Forward();
        Backward();
    }

    void Forward() {
        test_bnorm_desc_t bnd = p.test_bnd;

        src.reset(new memory({*data_desc, *eng}));
        dst.reset(new memory({*data_desc, *eng}));

        auto bnrm_desc =
                batch_normalization_forward::desc(prop_kind::forward_training,
                *data_desc, bnd.eps);
        bnrm_prim_desc.reset(
                new batch_normalization_forward::primitive_desc(bnrm_desc, *eng));

        weights.reset(new memory(bnrm_prim_desc->weights_primitive_desc()));

        fill_data<data_t>(
                src->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)src->get_data_handle());
        fill_data<data_t>(
                weights->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)weights->get_data_handle());

        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);

        workspace.reset(new memory(
                bnrm_prim_desc->workspace_primitive_desc()));
        auto bn = batch_normalization_forward(*bnrm_prim_desc,
                *src, *weights, *workspace, *dst);
        pipeline.push_back(bn);
        s.submit(pipeline).wait();

        check_bnorm_fwd<data_t>(bnd, *src, *weights, *dst);
    }

    void Backward()
    {
        test_bnorm_desc_t bnd = p.test_bnd;

        diff_src.reset(new memory({*diff_desc, *eng}));
        diff_dst.reset(new memory({*diff_desc, *eng}));

        auto bnrm_bwd_desc = batch_normalization_backward::desc(p.aprop_kind,
                    *diff_desc, *data_desc);
        auto bnrm_bwd_prim_desc = batch_normalization_backward::primitive_desc(
                    bnrm_bwd_desc, *eng, *bnrm_prim_desc);

        diff_weights.reset(
                new memory(bnrm_bwd_prim_desc.weights_primitive_desc()));

        fill_data<data_t>(
                diff_dst->get_primitive_desc().get_size() / sizeof(data_t),
                (data_t *)diff_dst->get_data_handle());

        std::vector<primitive> pipeline;
        auto s = stream(stream::kind::lazy);

        auto bn_bwd = p.aprop_kind == backward_data ?
                batch_normalization_backward(bnrm_bwd_prim_desc, *src,
                    *diff_dst, *weights, *workspace, *diff_src) :
                batch_normalization_backward(bnrm_bwd_prim_desc, *src,
                    *diff_dst, *weights, *workspace, *diff_src, *diff_weights);

        pipeline.push_back(bn_bwd);
        s.submit(pipeline).wait();

        check_bnorm_bwd<data_t>(bnd, p.aprop_kind, *src, *diff_dst, *weights,
            *workspace, *diff_src, *diff_weights);
    }
};

using bnorm_backward_test_float = bnorm_backward_test<float>;
using bnorm_bwd_test_params_float = bnorm_bwd_test_params;

TEST_P(bnorm_backward_test_float, TestsBNormBwd)
{
}

INSTANTIATE_TEST_CASE_P(TestBNormBackward, bnorm_backward_test_float,
        ::testing::Values(
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 10, 4, 4, 0.1 } }));

INSTANTIATE_TEST_CASE_P(
        TestBNormBackwardBlocked, bnorm_backward_test_float,
        ::testing::Values(
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 8, 4, 4, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward_data,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 8, 4, 4, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 4, 4, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward_data,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 4, 4, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 8, 8, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward_data,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 8, 8, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 16, 8, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward_data,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 16, 8, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 16, 10, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward_data,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 16, 10, 0.1 } }));

INSTANTIATE_TEST_CASE_P(
        TestBNormGoogleNetBackwardNCHW, bnorm_backward_test_float,
        ::testing::Values(
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 64, 112, 112, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 64, 56, 56, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 192, 56, 56, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 96, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 16, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 64, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 128, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 32, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 96, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 96, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 16, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 192, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 208, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 48, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 64, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 112, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 24, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 160, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 224, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 128, 4, 4, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 128, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 512, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 256, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 144, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 32, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 228, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 528, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 320, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 160, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 32, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 256, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 320, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 128, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 192, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 48, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nchw,
                        memory::format::nchw, memory::format::nc,
                        { 2, 384, 7, 7, 0.1 } }));

INSTANTIATE_TEST_CASE_P(
        TestBNormGoogleNetBackwardBlocked, bnorm_backward_test_float,
        ::testing::Values(
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 64, 112, 112, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 64, 56, 56, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 192, 56, 56, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 96, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 64, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 128, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 32, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 96, 28, 28, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 96, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 16, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 192, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 208, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 48, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 64, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 112, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 24, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 160, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 224, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 128, 4, 4, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 128, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 512, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 256, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 144, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 32, 14, 14, 0.1 } },
                /* size is not supported by nChw8c format yet
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 228, 14, 14, 0.1 } },
                */
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 528, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 320, 14, 14, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 160, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 32, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 256, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 320, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 128, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 192, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 48, 7, 7, 0.1 } },
                bnorm_bwd_test_params_float{ prop_kind::backward,
                        engine::kind::cpu, memory::format::nChw8c,
                        memory::format::nChw8c, memory::format::nc,
                        { 2, 384, 7, 7, 0.1 } }));

}
