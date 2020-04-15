/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "cpu/platform.hpp"

#if DNNL_X64
#include "cpu/x64/jit_simple_layer_normalization_kernels.hpp"
#endif

#include "cpu/simple_layer_normalization_kernels.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace lnorm_utils {

void statistics_kernel_t::operator()(
        const float *src, float *mean, float *var) const {
    float v_mean = 0;
    PRAGMA_OMP_SIMD(reduction(+ : v_mean))
    for (dim_t c = 0; c < C_; ++c) {
        v_mean += src[c];
    }
    v_mean /= C_;

    float v_variance = 0;
    PRAGMA_OMP_SIMD(reduction(+ : v_variance))
    for (dim_t c = 0; c < C_; ++c) {
        auto m = src[c] - v_mean;
        v_variance += m * m;
    }
    v_variance /= C_;

    *mean = v_mean;
    *var = v_variance;
}

void data_kernel_t::operator()(const float *src, float *dst, const float *ss,
        const float *mean, const float *var) const {
    float inv_sqrtvar = 1. / sqrtf(*var + eps_);
    PRAGMA_OMP_SIMD()
    for (dim_t c = 0; c < C_; ++c) {
        const float sm = (use_scaleshift_ ? ss[c] : 1.0f) * inv_sqrtvar;
        const float sv = use_scaleshift_ ? ss[C_ + c] : 0;
        dst[c] = sm * (src[c] - *mean) + sv;
    }
}

void diff_ss_kernel_t::operator()(const float *src, const float *diff_dst,
        float *diff_gamma, float *diff_beta, const float *mean,
        const float *var) const {
    float inv_sqrtvar = 1. / sqrtf(*var + eps_);
    PRAGMA_OMP_SIMD()
    for (dim_t c = 0; c < C_; c++) {
        float dd = diff_dst[c];
        diff_gamma[c] += (src[c] - *mean) * dd * inv_sqrtvar;
        diff_beta[c] += dd;
    }
}

void diff_data_kernel_t::operator()(const float *src, const float *diff_dst,
        float *diff_src, const float *ss, const float *mean,
        const float *var) const {
    float inv_sqrtvar = 1.f / sqrtf(*var + eps_);
    float dd_gamma = 0, dd_gamma_x = 0;
    if (calculate_diff_stats_) {
        PRAGMA_OMP_SIMD(reduction(+ : dd_gamma, dd_gamma_x))
        for (dim_t c = 0; c < C_; c++) {
            float gamma = use_scaleshift_ ? ss[c] : 1;
            dd_gamma += diff_dst[c] * gamma;
            dd_gamma_x += diff_dst[c] * gamma * (src[c] - *mean);
        }
        dd_gamma_x *= inv_sqrtvar;
    }
    PRAGMA_OMP_SIMD()
    for (dim_t c = 0; c < C_; c++) {
        float gamma = use_scaleshift_ ? ss[c] : 1;
        float v_diff_src = diff_dst[c] * gamma;
        if (calculate_diff_stats_)
            v_diff_src -= dd_gamma / C_
                    + (src[c] - *mean) * dd_gamma_x * inv_sqrtvar / C_;
        v_diff_src *= inv_sqrtvar;
        diff_src[c] = v_diff_src;
    }
}

// Interface section

statistics_kernel_t *statistics_kernel_t::create(
        const layer_normalization_pd_t *pd) {
#if DNNL_X64
    auto *res = x64::lnorm_utils::jit_statistics_kernel_create(pd);
    if (res) return res;
#endif

    return new statistics_kernel_t(pd);
}

data_kernel_t *data_kernel_t::create(const layer_normalization_pd_t *pd) {
#if DNNL_X64
    auto *res = x64::lnorm_utils::jit_data_kernel_create(pd);
    if (res) return res;
#endif

    return new data_kernel_t(pd);
}

diff_ss_kernel_t *diff_ss_kernel_t::create(const layer_normalization_pd_t *pd) {
#if DNNL_X64
    auto *res = x64::lnorm_utils::jit_diff_ss_kernel_create(pd);
    if (res) return res;
#endif

    return new diff_ss_kernel_t(pd);
}

diff_data_kernel_t *diff_data_kernel_t::create(
        const layer_normalization_pd_t *pd) {
#if DNNL_X64
    auto *res = x64::lnorm_utils::jit_diff_data_kernel_create(pd);
    if (res) return res;
#endif

    return new diff_data_kernel_t(pd);
}

} // namespace lnorm_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
