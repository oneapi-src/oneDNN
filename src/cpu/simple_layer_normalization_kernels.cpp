/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"

#include "cpu/platform.hpp"
#include "cpu/ref_io_helper.hpp"

// Control to enable/disable kernels for debugging.
#define ENABLE_JIT_KERNELS 1

#if ENABLE_JIT_KERNELS && DNNL_X64
#include "cpu/x64/jit_uni_layer_normalization_kernels.hpp"
#endif

#include "cpu/simple_layer_normalization_kernels.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace lnorm_utils {

using namespace data_type;

void stat_and_data_kernel_t::operator()(const void *src, void *dst,
        const float *scale, const float *shift, float *mean, float *var,
        const size_t block_size) const {
    // XXX: manual unrolling for use_scaleshift_ due to clang issue.
    //      see: CLANG_WA_01_SAFE_TO_USE_OMP_SIMD

    for (size_t offset = 0; offset < block_size; offset++) {
        float v_mean = 0, v_variance = 0;
        if (calculate_stats_) {
            PRAGMA_OMP_SIMD(reduction(+ : v_mean))
            for (dim_t c = 0; c < C_; ++c) {
                float s = io::load_float_value(
                        data_type_, src, c + C_ * offset);
                v_mean += s;
            }
            v_mean /= C_;

            PRAGMA_OMP_SIMD(reduction(+ : v_variance))
            for (dim_t c = 0; c < C_; ++c) {
                float s = io::load_float_value(
                        data_type_, src, c + C_ * offset);
                float src_sub_mean = s - v_mean;
                v_variance += src_sub_mean * src_sub_mean;
            }
            v_variance /= C_;
        } else {
            v_mean = mean[offset];
            v_variance = var[offset];
        }

        const float inv_sqrtvar = 1.f / sqrtf(v_variance + eps_);
        if (use_scaleshift_ || (use_scale_ && use_shift_)) {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; ++c) {
                const float sm = scale[c] * inv_sqrtvar;
                const float sv = shift[c];
                const size_t off = c + C_ * offset;
                float s = io::load_float_value(data_type_, src, off);
                float d = sm * (s - v_mean) + sv;
                io::store_float_value(data_type_, d, dst, off);
            }
        } else if (use_scale_) {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; ++c) {
                const float sm = scale[c] * inv_sqrtvar;
                const size_t off = c + C_ * offset;
                float s = io::load_float_value(data_type_, src, off);
                float d = sm * (s - v_mean);
                io::store_float_value(data_type_, d, dst, off);
            }
        } else if (use_shift_) {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; ++c) {
                const float sm = 1.f * inv_sqrtvar;
                const float sv = shift[c];
                const size_t off = c + C_ * offset;
                float s = io::load_float_value(data_type_, src, off);
                float d = sm * (s - v_mean) + sv;
                io::store_float_value(data_type_, d, dst, off);
            }
        } else {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; ++c) {
                const float sm = 1.f * inv_sqrtvar;
                const size_t off = c + C_ * offset;
                float s = io::load_float_value(data_type_, src, off);
                float d = sm * (s - v_mean);
                io::store_float_value(data_type_, d, dst, off);
            }
        }
        if (calculate_stats_ && save_stats_) {
            mean[offset] = v_mean;
            var[offset] = v_variance;
        }
    }
}

void diff_ss_kernel_t::operator()(const void *src, const void *diff_dst,
        float *diff_gamma, float *diff_beta, const float *mean,
        const float *var, float *const inv_sqrtvar,
        const size_t block_size) const {
    for (size_t offset = 0; offset < block_size; offset++) {
        inv_sqrtvar[offset] = 1. / sqrtf(var[offset] + eps_);
        PRAGMA_OMP_SIMD()
        for (dim_t c = 0; c < C_; c++) {
            const size_t off = c + C_ * offset;
            float s = io::load_float_value(data_type_, src, off);
            float dd = io::load_float_value(data_type_, diff_dst, off);
            diff_gamma[c] += (s - mean[offset]) * dd * inv_sqrtvar[offset];
            diff_beta[c] += dd;
        }
    }
}

void diff_data_kernel_t::operator()(const void *src, const void *diff_dst,
        void *diff_src, const float *ss, const float *mean,
        float *const inv_sqrtvar, const size_t block_size) const {
    // XXX: manual unrolling for use_scaleshift_ due to clang issue.
    //      see: CLANG_WA_01_SAFE_TO_USE_OMP_SIMD
    float dd_gamma, dd_gamma_x;
    for (size_t offset = 0; offset < block_size; offset++) {
        // reduce gamma
        dd_gamma = dd_gamma_x = 0;
        if (calculate_diff_stats_) {
            if (use_scaleshift_ || use_scale_) {
                PRAGMA_OMP_SIMD(reduction(+ : dd_gamma, dd_gamma_x))
                for (dim_t c = 0; c < C_; c++) {
                    const size_t off = c + C_ * offset;
                    float s = io::load_float_value(data_type_, src, off);
                    float dd = io::load_float_value(data_type_, diff_dst, off);
                    dd_gamma += dd * ss[c];
                    dd_gamma_x += dd * ss[c] * (s - mean[offset]);
                }
            } else {
                PRAGMA_OMP_SIMD(reduction(+ : dd_gamma, dd_gamma_x))
                for (dim_t c = 0; c < C_; c++) {
                    const size_t off = c + C_ * offset;
                    float s = io::load_float_value(data_type_, src, off);
                    float dd = io::load_float_value(data_type_, diff_dst, off);
                    dd_gamma += dd;
                    dd_gamma_x += dd * (s - mean[offset]);
                }
            }
            dd_gamma_x *= inv_sqrtvar[offset];
        }

        // calculate diff_dst
        if (use_scaleshift_ || use_scale_) {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; c++) {
                const size_t off = c + C_ * offset;
                float dd = io::load_float_value(data_type_, diff_dst, off);
                float ds = dd * ss[c];
                if (calculate_diff_stats_) {
                    float s = io::load_float_value(data_type_, src, off);
                    ds -= dd_gamma / C_;
                    ds -= (s - mean[offset]) * dd_gamma_x * inv_sqrtvar[offset]
                            / C_;
                }
                ds *= inv_sqrtvar[offset];
                io::store_float_value(data_type_, ds, diff_src, off);
            }
        } else {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; c++) {
                const size_t off = c + C_ * offset;
                float dd = io::load_float_value(data_type_, diff_dst, off);
                float ds = dd;
                if (calculate_diff_stats_) {
                    float s = io::load_float_value(data_type_, src, off);
                    ds -= dd_gamma / C_;
                    ds -= (s - mean[offset]) * dd_gamma_x * inv_sqrtvar[offset]
                            / C_;
                }
                ds *= inv_sqrtvar[offset];
                io::store_float_value(data_type_, ds, diff_src, off);
            }
        }
    }
}

// Interface section

stat_and_data_kernel_t *stat_and_data_kernel_t::create(
        const layer_normalization_pd_t *pd) {
#if ENABLE_JIT_KERNELS && DNNL_X64
    if (auto *res = x64::lnorm_utils::stat_and_data_kernel_create(pd))
        return res;
#endif
    return new stat_and_data_kernel_t(pd);
}

diff_ss_kernel_t *diff_ss_kernel_t::create(const layer_normalization_pd_t *pd) {
#if ENABLE_JIT_KERNELS && DNNL_X64
    if (auto *res = x64::lnorm_utils::diff_ss_kernel_create(pd)) return res;
#endif
    return new diff_ss_kernel_t(pd);
}

diff_data_kernel_t *diff_data_kernel_t::create(
        const layer_normalization_pd_t *pd) {
#if ENABLE_JIT_KERNELS && DNNL_X64
    if (auto *res = x64::lnorm_utils::diff_data_kernel_create(pd)) return res;
#endif
    return new diff_data_kernel_t(pd);
}

} // namespace lnorm_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
