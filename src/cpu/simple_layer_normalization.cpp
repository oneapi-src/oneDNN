/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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
#include "dnnl_thread.hpp"
#include "type_helpers.hpp"

#include "cpu_batch_normalization_utils.hpp"
#include "simple_layer_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;

void simple_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);
    auto scaleshift = CTX_IN_MEM(const float *, DNNL_ARG_SCALE_SHIFT);

    float *mean, *variance;
    if (pd()->use_tmp_stats()) {
        auto scratchpad = ctx.get_scratchpad_grantor();
        mean = scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean = pd()->stats_are_src()
                ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
                : CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
        variance = pd()->stats_are_src()
                ? const_cast<float *>(
                        CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
                : CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);
    }

    const memory_desc_wrapper src_d(pd()->src_md());

    const dim_t N = pd()->across_axis();
    const dim_t C_padded = src_d.padded_dims()[pd()->ndims() - 1];

    const bool save_stats = pd()->is_training();
    const bool calculate_stats = !pd()->stats_are_src();

    parallel_nd(N, [&](dim_t n) {
        auto v_mean = calculate_stats ? 0 : mean[n];
        auto v_variance = calculate_stats ? 0 : variance[n];

        if (calculate_stats)
            (*stat_kernel_)(&src[n * C_padded], &v_mean, &v_variance);

        (*data_kernel_)(&src[n * C_padded], &dst[n * C_padded], scaleshift,
                &v_mean, &v_variance);

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
    auto scratchpad = ctx.get_scratchpad_grantor();
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const float *, DNNL_ARG_DIFF_DST);
    auto scaleshift = CTX_IN_MEM(const float *, DNNL_ARG_SCALE_SHIFT);
    auto diff_src = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SRC);
    auto diff_scaleshift = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SCALE_SHIFT);

    const float *mean, *variance;
    if (pd()->use_tmp_stats()) {
        mean = scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
        variance = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);
    }

    const memory_desc_wrapper src_d(pd()->src_md());

    const dim_t N = pd()->across_axis();
    const dim_t C = pd()->norm_axis();
    const dim_t C_padded = src_d.padded_dims()[pd()->ndims() - 1];

    float *reduce = scratchpad.template get<float>(key_lnorm_reduction);
    if (diff_scaleshift == nullptr)
        diff_scaleshift = scratchpad.template get<float>(key_lnorm_tmp_diff_ss);

    int max_nthr = dnnl_get_max_threads();
    parallel(max_nthr, [&](int ithr, int nthr) {
        assert(nthr == max_nthr);
        dim_t N_s = 0, N_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);

        float *my_diff_gamma = reduce + C * ithr;
        float *my_diff_beta = reduce + C * nthr + C * ithr;
        for (dim_t c = 0; c < C; c++) {
            my_diff_gamma[c] = 0.;
            my_diff_beta[c] = 0.;
        }
        for (dim_t n = N_s; n < N_e; n++) {
            (*diff_ss_kernel_)(&src[n * C_padded], &diff_dst[n * C_padded],
                    my_diff_gamma, my_diff_beta, &mean[n], &variance[n]);
        }
    });

    parallel_nd(C, [&](dim_t c) {
        float diff_gamma = 0, diff_beta = 0;
        for (dim_t n = 0; n < max_nthr; n++) {
            diff_gamma += reduce[C * n + c];
            diff_beta += reduce[C * max_nthr + C * n + c];
        }
        diff_scaleshift[c] = diff_gamma;
        diff_scaleshift[C + c] = diff_beta;
    });

    parallel(max_nthr, [&](int ithr, int nthr) {
        assert(nthr == max_nthr);
        dim_t N_s = 0, N_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);

        for (dim_t n = N_s; n < N_e; n++) {
            (*diff_data_kernel_)(&src[n * C_padded], &diff_dst[n * C_padded],
                    &diff_src[n * C_padded], scaleshift, &mean[n],
                    &variance[n]);
        }
    });
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
