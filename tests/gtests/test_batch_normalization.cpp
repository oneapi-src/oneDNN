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

struct test_bnrm_sizes_t {
    int mb, c, h, w;
};

struct test_bnrm_formats_t {
    mkldnn::memory::format data_format;
    mkldnn::memory::format diff_format;
};

struct test_bnrm_params_t {
    mkldnn::engine::kind engine_kind;
    test_bnrm_formats_t formats;
    test_bnrm_sizes_t sizes;
    double eps;
};

template <typename data_t>
void check_bnrm_fwd(const test_bnrm_params_t &bp,
        const memory &src, const memory &weights, const memory &dst)
{
    (void)bp;
    (void)src;
    (void)weights;
    (void)dst;
    /*
    const data_t *src_data = (const data_t *)src.get_data_handle();
    const data_t *weights_data = (const data_t *)weights.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

#pragma omp parallel for
    for (int c = 0; c < bp.c; c++) {
        workspace_data[c] = data_t(0);
        for (int n = 0; n < bp.mb; n++)
        for (int h = 0; h < bp.h; h++)
            for (int w = 0; w < bp.w; w++) {
                int sidx = n * bp.c * bp.h * bp.w + c * bp.h * bp.w
                        + h * bp.w + w;
                workspace_data[c] += src_data[map_index(src_d, sidx)];
            }
        workspace_data[c] /= bp.mb * bp.h * bp.w;

        workspace_data[bp.c + c] = data_t(0);
        for (int n = 0; n < bp.mb; n++)
        for (int h = 0; h < bp.h; h++)
            for (int w = 0; w < bp.w; w++) {
                int sidx = n * bp.c * bp.h * bp.w + c * bp.h * bp.w
                        + h * bp.w + w;
                data_t tmp = src_data[map_index(src_d, sidx)]
                        - workspace_data[c];
                workspace_data[bp.c + c] += tmp * tmp;
            }
        workspace_data[bp.c + c] = workspace_data[bp.c + c]
                / (bp.mb * bp.h * bp.w) + bp.eps;
        workspace_data[bp.c + c] = data_t(1)
                / sqrt(workspace_data[bp.c + c]);

        for (int n = 0; n < bp.mb; n++)
        for (int h = 0; h < bp.h; h++)
            for (int w = 0; w < bp.w; w++) {
                int sdidx = n * bp.c * bp.h * bp.w + c * bp.h * bp.w
                        + h * bp.w + w;
                data_t ref_dst = weights_data[map_index(weights_d, c)]
                        * (src_data[map_index(src_d, sdidx)]
                        - workspace_data[c]) * workspace_data[bp.c + c]
                        + weights_data[map_index(weights_d, bp.c + c)];
                data_t out = dst_data[map_index(dst_d, sdidx)];
                data_t eps = 1.e-6 * bp.mb * bp.h * bp.w;
                data_t norm_max = std::max(fabs(out), fabs(ref_dst));
                if (norm_max < eps) norm_max = data_t(1);
                EXPECT_NEAR((out - ref_dst) / norm_max, 0., eps);
            }
    }
    */
}

template <typename data_t>
void check_bnrm_bwd(const test_bnrm_params_t &bp,
        const memory &src, const memory &diff_dst, const memory &weights,
        const memory &diff_src, const memory &diff_weights)
{
    (void)bp;
    (void)src;
    (void)diff_dst;
    (void)weights;
    (void)diff_src;
    (void)diff_weights;
    /*
    const data_t *src_data = (const data_t *)src.get_data_handle();
    const data_t *weights_data = (const data_t *)weights.get_data_handle();
    const data_t *diff_dst_data = (const data_t *)diff_dst.get_data_handle();
    data_t *diff_src_data = (data_t *)diff_src.get_data_handle();
    const data_t *diff_weights_data = (const data_t *)diff_weights.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc diff_src_d = diff_src.get_primitive_desc().desc();
    const memory::desc diff_weights_d = diff_weights.get_primitive_desc().desc();

    const data_t eps = 1.e-6 * bp.mb * bp.h * bp.w;

#pragma omp parallel for
    for (int c = 0; c < bp.c; c++) {
        data_t ref_diff_gamma = data_t(0);
        data_t ref_diff_beta = data_t(0);

        auto mean = ws_mean[c];
        auto variance = ws_variance[c];

        auto gamma = weights_data[map_index(weights_d, c)];

        for (int n = 0; n < bp.mb; n++)
        for (int h = 0; h < bp.h; h++)
            for (int w = 0; w < bp.w; w++) {
                int sidx = n * bp.c * bp.h * bp.w + c * bp.h * bp.w
                        + h * bp.w + w;
                ref_diff_gamma += (src_data[map_index(src_d, sidx)] - mean)
                    * diff_dst_data[map_index(diff_dst_d, sidx)];
                ref_diff_beta += diff_dst_data[map_index(diff_dst_d, sidx)];
            }
        ref_diff_gamma *= variance;

        if (aprop_kind == backward) {
            auto diff_gamma = diff_weights_data[map_index(diff_weights_d, c)];
            auto diff_beta = diff_weights_data[map_index(diff_weights_d, bp.c + c)];
            EXPECT_NEAR(diff_gamma, ref_diff_gamma, eps);
            EXPECT_NEAR(diff_beta, ref_diff_beta, eps);
        }

        for (int n = 0; n < bp.mb; n++)
        for (int h = 0; h < bp.h; h++)
            for (int w = 0; w < bp.w; w++) {
                int sidx = n * bp.c * bp.h * bp.w + c * bp.h * bp.w
                        + h * bp.w + w;
                data_t ref_diff_src = diff_dst_data[map_index(diff_dst_d, sidx)]
                        - ref_diff_beta/(bp.mb*bp.h*bp.w)
                        - (src_data[map_index(src_d, sidx)] - mean)
                        *ref_diff_gamma*variance/(bp.mb*bp.h*bp.w);
                ref_diff_src *= gamma*variance;
                data_t out_diff_src = diff_src_data[map_index(diff_src_d, sidx)];
                data_t norm_max = std::max(fabs(out_diff_src), fabs(ref_diff_src));
                if (norm_max < eps) norm_max = data_t(1);
                EXPECT_NEAR((out_diff_src - ref_diff_src) / norm_max, 0., eps);
            }
    }
    */
}

template <typename data_t>
class bnrm_test : public ::testing::TestWithParam<test_bnrm_params_t> {
private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;
    std::shared_ptr<memory> diff_src;
    std::shared_ptr<memory> diff_dst;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> diff_weights;
    std::shared_ptr<memory> mean;
    std::shared_ptr<memory> variance;
    std::shared_ptr<memory::desc> data_desc;
    std::shared_ptr<memory::desc> diff_desc;
    std::shared_ptr<batch_normalization_forward::primitive_desc> bnrm_prim_desc;
    std::shared_ptr<batch_normalization_backward::primitive_desc>
        bnrm_bwd_prim_desc;
    test_bnrm_params_t p;
    std::shared_ptr<engine> eng;
    memory::data_type data_type;

protected:
    virtual void SetUp() {
        p = ::testing::TestWithParam<test_bnrm_params_t>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        eng.reset(new engine(p.engine_kind, 0));
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        test_bnrm_sizes_t bs = p.sizes;
        data_desc.reset(new memory::desc({ bs.mb, bs.c, bs.h, bs.w },
                    data_type, p.formats.data_format));
        diff_desc.reset(new memory::desc({ bs.mb, bs.c, bs.h, bs.w },
                    data_type, p.formats.diff_format));

        src.reset(new memory({*data_desc, *eng}));
        dst.reset(new memory({*data_desc, *eng}));
        diff_src.reset(new memory({*diff_desc, *eng}));
        diff_dst.reset(new memory({*diff_desc, *eng}));

        auto training = prop_kind::forward_training;
        auto scoring = prop_kind::forward_scoring;

        auto backward = prop_kind::backward;
        auto backward_data = prop_kind::backward_data;

        // TODO: check me
        Forward(use_global_stats, training);
        Forward(0u, prop_kind::forward_scoring);
        Forward(0u, training);
        Forward(use_scale_shift | use_global_stats, training);
        Forward(use_scale_shift, scoring);
        Forward(use_scale_shift, training);

        Backward(use_scale_shift | omit_stats, backward);
        Backward(use_scale_shift, backward);
        Backward(omit_stats, backward_data);
        Backward(0u, backward_data);
        Backward(use_scale_shift | omit_stats, backward_data);
        Backward(use_scale_shift, backward_data);
    }

    void Forward(unsigned flags, prop_kind pk) {
        bool useScaleShift = flags & use_scale_shift;
        bool useGlobalStats = flags & use_global_stats;
        bool isTraining = pk == prop_kind::forward_training;

        auto bnrm_desc = batch_normalization_forward::desc(pk,
                    *data_desc, p.eps, flags);

        bnrm_prim_desc.reset(new batch_normalization_forward::primitive_desc(
                    bnrm_desc, *eng));

        if (useScaleShift) weights.reset(new memory(
                    bnrm_prim_desc->weights_primitive_desc()));
        if (isTraining || useGlobalStats) {
            mean.reset(new memory(bnrm_prim_desc->mean_primitive_desc()));
            variance.reset(
                    new memory(bnrm_prim_desc->variance_primitive_desc()));
        }

        fill(*src);
        if (useScaleShift) fill(*weights);
        if (useGlobalStats) {
            fill(*mean);
            fill(*variance);
        }

        auto bn = createBnrmFwd(isTraining, useGlobalStats, useScaleShift);

        std::vector<primitive> pipeline;
        pipeline.push_back(bn);
        stream(stream::kind::lazy).submit(pipeline).wait();

        check_bnrm_fwd<data_t>(p, *src, *weights, *dst);
    }

    void Backward(unsigned flags, prop_kind pk) {
        bool useScaleShift = flags & use_scale_shift;

        auto bnrm_bwd_desc = batch_normalization_backward::desc(
                pk, *diff_desc, *data_desc, p.eps, flags);

        bnrm_bwd_prim_desc.reset(
                new batch_normalization_backward::primitive_desc(
                bnrm_bwd_desc, *eng, *bnrm_prim_desc));

        if (useScaleShift) weights.reset(new memory(
                    bnrm_bwd_prim_desc->weights_primitive_desc()));
        if (pk == prop_kind::backward) diff_weights.reset(new memory(
                    bnrm_bwd_prim_desc->diff_weights_primitive_desc()));
        mean.reset(new memory(bnrm_bwd_prim_desc->mean_primitive_desc()));
        variance.reset(new memory(
                    bnrm_bwd_prim_desc->variance_primitive_desc()));

        if (useScaleShift) fill(*weights);
        fill(*diff_dst);
        fill(*mean);
        fill(*variance);

        auto bnrm_bwd = createBnrmBwd(useScaleShift, pk);

        std::vector<primitive> pipeline;
        pipeline.push_back(bnrm_bwd);
        stream(stream::kind::lazy).submit(pipeline).wait();

        check_bnrm_bwd<data_t>(p,
                *src, *diff_dst, *weights, *diff_src, *diff_weights);
    }

    void fill(memory &m) {
        fill_data<data_t>(m.get_primitive_desc().get_size() / sizeof(data_t),
                reinterpret_cast<data_t *>(m.get_data_handle()));
    }

    primitive createBnrmFwd(bool isTraining, bool useGlobalStats,
            bool useScaleShift)
    {
        if (!isTraining && !useGlobalStats) {
            return useScaleShift
                ? batch_normalization_forward(*bnrm_prim_desc,
                    *src, *weights, *dst)
                : batch_normalization_forward(*bnrm_prim_desc, *src, *dst);
        } else {
            return useScaleShift
                ? batch_normalization_forward(*bnrm_prim_desc,
                    *src, *mean, *variance, *weights, *dst)
                : batch_normalization_forward(*bnrm_prim_desc,
                    *src, *mean, *variance, *dst);
        }
    }

    primitive createBnrmBwd(bool useScaleShift, prop_kind pk)
    {
        if (useScaleShift) {
            return pk == prop_kind::backward
                ? batch_normalization_backward(*bnrm_bwd_prim_desc,
                    *src, *mean, *variance, *diff_dst, *weights, *diff_src)
                : batch_normalization_backward(*bnrm_bwd_prim_desc,
                    *src, *mean, *variance, *diff_dst, *weights,
                    *diff_src, *diff_weights);
        } else {
            return batch_normalization_backward(*bnrm_bwd_prim_desc,
                    *src, *mean, *variance, *diff_dst, *diff_src);
        }
    }
};

using bnrm_test_float = bnrm_test<float>;

TEST_P(bnrm_test_float, TestsBnrmBwd)
{
}

#define EXPAND_SIZES(mb, c, h, w) { mb, c, h, w }
#define EXPAND_FORMATS(data, diff) \
    { memory::format::data, memory::format::diff }

#define ENGINE engine::kind::cpu
#define EPS 1e-5

#define PARAMS(data, diff, mb, c, h, w, eps) \
    test_bnrm_params_t { ENGINE, \
    EXPAND_FORMATS(data, diff), EXPAND_SIZES(mb, c, h, w), eps }

#define INST_TEST_CASE(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, bnrm_test_float, ::testing::Values(__VA_ARGS__))

INST_TEST_CASE(Simple_NCHW,
    PARAMS(nchw, nchw, 2, 10, 4, 4, EPS)
);

INST_TEST_CASE(Simple_Blocked,
    PARAMS(nChw8c, nChw8c, 2, 8, 4, 4, EPS),
    PARAMS(nChw8c, nChw8c, 2, 8, 4, 4, EPS),
    PARAMS(nChw8c, nChw8c, 2, 16, 4, 4, EPS),
    PARAMS(nChw8c, nChw8c, 2, 16, 4, 4, EPS),
    PARAMS(nChw8c, nChw8c, 2, 16, 8, 8, EPS),
    PARAMS(nChw8c, nChw8c, 2, 16, 8, 8, EPS),
    PARAMS(nChw8c, nChw8c, 2, 16, 16, 8, EPS),
    PARAMS(nChw8c, nChw8c, 2, 16, 16, 8, EPS),
    PARAMS(nChw8c, nChw8c, 2, 16, 10, 8, EPS),
    PARAMS(nChw8c, nChw8c, 2, 16, 10, 8, EPS)
);

INST_TEST_CASE(GoogleNet_NCHW,
    PARAMS(nchw, nchw, 2, 64, 112, 112, EPS),
    PARAMS(nchw, nchw, 2, 64, 56, 56, EPS),
    PARAMS(nchw, nchw, 2, 192, 56, 56, EPS),
    PARAMS(nchw, nchw, 2, 96, 28, 28, EPS),
    PARAMS(nchw, nchw, 2, 16, 28, 28, EPS),
    PARAMS(nchw, nchw, 2, 64, 28, 28, EPS),
    PARAMS(nchw, nchw, 2, 128, 28, 28, EPS),
    PARAMS(nchw, nchw, 2, 32, 28, 28, EPS),
    PARAMS(nchw, nchw, 2, 96, 28, 28, EPS),
    PARAMS(nchw, nchw, 2, 96, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 16, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 192, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 208, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 48, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 64, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 112, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 24, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 160, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 224, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 128, 4, 4, EPS),
    PARAMS(nchw, nchw, 2, 128, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 512, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 256, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 144, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 32, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 228, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 528, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 320, 14, 14, EPS),
    PARAMS(nchw, nchw, 2, 160, 7, 7, EPS),
    PARAMS(nchw, nchw, 2, 32, 7, 7, EPS),
    PARAMS(nchw, nchw, 2, 256, 7, 7, EPS),
    PARAMS(nchw, nchw, 2, 320, 7, 7, EPS),
    PARAMS(nchw, nchw, 2, 128, 7, 7, EPS),
    PARAMS(nchw, nchw, 2, 192, 7, 7, EPS),
    PARAMS(nchw, nchw, 2, 48, 7, 7, EPS),
    PARAMS(nchw, nchw, 2, 384, 7, 7, EPS)
);

INST_TEST_CASE(GoogleNet_Blocked,
    PARAMS(nChw8c, nChw8c, 2, 64, 112, 112, EPS),
    PARAMS(nChw8c, nChw8c, 2, 64, 56, 56, EPS),
    PARAMS(nChw8c, nChw8c, 2, 192, 56, 56, EPS),
    PARAMS(nChw8c, nChw8c, 2, 96, 28, 28, EPS),
    PARAMS(nChw8c, nChw8c, 2, 16, 28, 28, EPS),
    PARAMS(nChw8c, nChw8c, 2, 64, 28, 28, EPS),
    PARAMS(nChw8c, nChw8c, 2, 128, 28, 28, EPS),
    PARAMS(nChw8c, nChw8c, 2, 32, 28, 28, EPS),
    PARAMS(nChw8c, nChw8c, 2, 96, 28, 28, EPS),
    PARAMS(nChw8c, nChw8c, 2, 96, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 16, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 192, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 208, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 48, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 64, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 112, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 24, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 160, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 224, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 128, 4, 4, EPS),
    PARAMS(nChw8c, nChw8c, 2, 128, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 512, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 256, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 144, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 32, 14, 14, EPS),
    /* size is not supported by nChw8c format yet
    PARAMS(nChw8c, nChw8c, nc, 2, 228, 14, 14, EPS),
    */
    PARAMS(nChw8c, nChw8c, 2, 528, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 320, 14, 14, EPS),
    PARAMS(nChw8c, nChw8c, 2, 160, 7, 7, EPS),
    PARAMS(nChw8c, nChw8c, 2, 32, 7, 7, EPS),
    PARAMS(nChw8c, nChw8c, 2, 256, 7, 7, EPS),
    PARAMS(nChw8c, nChw8c, 2, 320, 7, 7, EPS),
    PARAMS(nChw8c, nChw8c, 2, 128, 7, 7, EPS),
    PARAMS(nChw8c, nChw8c, 2, 192, 7, 7, EPS),
    PARAMS(nChw8c, nChw8c, 2, 48, 7, 7, EPS),
    PARAMS(nChw8c, nChw8c, 2, 384, 7, 7, EPS)
);

}
