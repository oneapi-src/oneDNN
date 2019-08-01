/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <assert.h>
#include <math.h>

#include "cpu_engine.hpp"

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"

#include "cpu_batch_normalization_utils.hpp"
#include "simple_layer_normalization.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;

void simple_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const float *, MKLDNN_ARG_SRC);
    auto dst = CTX_OUT_MEM(float *, MKLDNN_ARG_DST);
    auto scaleshift = CTX_IN_MEM(const float *, MKLDNN_ARG_SCALE_SHIFT);

    float *mean, *variance;
    if (pd()->use_tmp_stats()) {
        auto scratchpad = this->scratchpad(ctx);
        mean = scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean = pd()->stats_are_src() ? const_cast<float *>(
                       CTX_IN_MEM(const float *, MKLDNN_ARG_MEAN))
                                     : CTX_OUT_MEM(float *, MKLDNN_ARG_MEAN);
        variance = pd()->stats_are_src()
                ? const_cast<float *>(
                        CTX_IN_MEM(const float *, MKLDNN_ARG_VARIANCE))
                : CTX_OUT_MEM(float *, MKLDNN_ARG_VARIANCE);
    }

    const memory_desc_wrapper src_d(pd()->src_md());

    const dim_t N = pd()->across_axis();
    const dim_t C = pd()->norm_axis();
    const dim_t C_padded = src_d.padded_dims()[pd()->ndims() - 1];

    const float eps = pd()->desc()->layer_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();
    const bool save_stats = pd()->is_training();
    const bool calculate_stats = !pd()->stats_are_src();

    parallel_nd(N, [&](dim_t n) {
        auto v_mean = calculate_stats ? 0 : mean[n];
        auto v_variance = calculate_stats ? 0 : variance[n];

        if (calculate_stats) {
            PRAGMA_OMP_SIMD(reduction(+ : v_mean))
            for (dim_t c = 0; c < C; ++c) {
                v_mean += src[n * C_padded + c];
            }
            v_mean /= C;

            PRAGMA_OMP_SIMD(reduction(+ : v_variance))
            for (dim_t c = 0; c < C; ++c) {
                auto m = src[n * C_padded + c] - v_mean;
                v_variance += m * m;
            }
            v_variance /= C;
        }

        float sqrt_variance = sqrtf(v_variance + eps);
        PRAGMA_OMP_SIMD()
        for (dim_t c = 0; c < C; ++c) {
            const float sm
                    = (use_scaleshift ? scaleshift[c] : 1.0f) / sqrt_variance;
            const float sv = use_scaleshift ? scaleshift[C + c] : 0;
            const size_t data_off = n * C_padded + c;

            dst[data_off] = sm * (src[data_off] - v_mean) + sv;
        }

        if (calculate_stats) {
            if (save_stats) {
                mean[n] = v_mean;
                variance[n] = v_variance;
            }
        }
    });
}

void simple_layer_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    auto scratchpad = this->scratchpad(ctx);
    auto src = CTX_IN_MEM(const float *, MKLDNN_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const float *, MKLDNN_ARG_DIFF_DST);
    auto scaleshift = CTX_IN_MEM(const float *, MKLDNN_ARG_SCALE_SHIFT);
    auto diff_src = CTX_OUT_MEM(float *, MKLDNN_ARG_DIFF_SRC);
    auto diff_scaleshift = CTX_OUT_MEM(float *, MKLDNN_ARG_DIFF_SCALE_SHIFT);

    const float *mean, *variance;
    if (pd()->use_tmp_stats()) {
        mean = scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean = CTX_IN_MEM(const float *, MKLDNN_ARG_MEAN);
        variance = CTX_IN_MEM(const float *, MKLDNN_ARG_VARIANCE);
    }

    const memory_desc_wrapper src_d(pd()->src_md());

    const dim_t N = pd()->across_axis();
    const dim_t C = pd()->norm_axis();
    const dim_t C_padded = src_d.padded_dims()[pd()->ndims() - 1];

    const float eps = pd()->desc()->layer_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();
    const bool calculate_diff_stats = !pd()->use_global_stats();

    float *reduce = scratchpad.template get<float>(key_lnorm_reduction);
    if (diff_scaleshift == nullptr)
        diff_scaleshift = scratchpad.template get<float>(key_lnorm_tmp_diff_ss);

    int nthr = mkldnn_get_max_threads();
    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t N_s = 0, N_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);

        float *my_diff_gamma = reduce + C * ithr;
        float *my_diff_beta = reduce + C * nthr + C * ithr;
        for (dim_t c = 0; c < C; c++) {
            my_diff_gamma[c] = 0.;
            my_diff_beta[c] = 0.;
        }
        for (dim_t n = N_s; n < N_e; n++) {
            float v_mean = mean[n];
            float inv_sqrt_variance
                    = static_cast<float>(1.0f / sqrtf(variance[n] + eps));
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C; c++) {
                const size_t data_off = n * C_padded + c;
                float dd = diff_dst[data_off];
                my_diff_gamma[c]
                        += (src[data_off] - v_mean) * dd * inv_sqrt_variance;
                my_diff_beta[c] += dd;
            }
        }
    });

    parallel_nd(C, [&](dim_t c) {
        float diff_gamma = 0, diff_beta = 0;
        for (dim_t n = 0; n < nthr; n++) {
            diff_gamma += reduce[C * n + c];
            diff_beta += reduce[C * nthr + C * n + c];
        }
        diff_scaleshift[c] = diff_gamma;
        diff_scaleshift[C + c] = diff_beta;
    });

    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t N_s = 0, N_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);

        float *my_diff_gamma = reduce + C * ithr;
        float *my_diff_beta = reduce + C * nthr + C * ithr;
        for (dim_t c = 0; c < C; c++) {
            my_diff_gamma[c] = diff_scaleshift[c];
            my_diff_beta[c] = diff_scaleshift[C + c];
        }

        for (dim_t n = N_s; n < N_e; n++) {
            float v_mean = mean[n];
            float inv_sqrt_variance
                    = static_cast<float>(1.0f / sqrtf(variance[n] + eps));
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C; c++) {
                float gamma = use_scaleshift ? scaleshift[c] : 1;
                const size_t data_off = n * C_padded + c;
                float v_diff_src = diff_dst[data_off];
                if (calculate_diff_stats)
                    v_diff_src -= my_diff_beta[c] / C
                            + (src[data_off] - v_mean) * my_diff_gamma[c]
                                    * inv_sqrt_variance / C;
                v_diff_src *= gamma * inv_sqrt_variance;
                diff_src[data_off] = v_diff_src;
            }
        }
    });
}

} // namespace cpu
} // namespace impl
} // namespace mkldnn
