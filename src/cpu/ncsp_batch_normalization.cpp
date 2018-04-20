/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "cpu_batch_normalization_utils.hpp"
#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "ncsp_batch_normalization.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

typedef float data_t;
ncsp_batch_normalization_fwd_t::ncsp_batch_normalization_fwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), stats_reduction_(nullptr),
    tmp_mean_(nullptr), tmp_variance_(nullptr), conf_(*pd) {
    if (!conf_.stats_is_src()) {
        this->stats_reduction_ = (data_t *)malloc(
                conf_.C() * omp_get_max_threads() * sizeof(data_t), 64);
        if (!conf_.is_training()) {
            this->tmp_mean_ = (data_t *)malloc(conf_.C() * sizeof(data_t), 64);
            this->tmp_variance_
                    = (data_t *)malloc(conf_.C() * sizeof(data_t), 64);
        }
    }
}
ncsp_batch_normalization_fwd_t::~ncsp_batch_normalization_fwd_t() {
    if (!conf_.stats_is_src()) {
        free(this->stats_reduction_);
        if (!conf_.is_training()) {
            free(this->tmp_mean_);
            free(this->tmp_variance_);
        }
    }
}

void ncsp_batch_normalization_fwd_t::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));
    const bool calculate_stats = !conf_.stats_is_src();
    const bool save_stats = conf_.is_training();
    data_t *mean, *variance;
    if (!calculate_stats) {
        mean = reinterpret_cast<data_t *>(
                const_cast<char *>(this->input_memory(1)));
        variance = reinterpret_cast<data_t *>(
                const_cast<char *>(this->input_memory(2)));
    } else {
        if (save_stats) {
            mean = reinterpret_cast<data_t *>(this->memory(1));
            variance = reinterpret_cast<data_t *>(this->memory(2));
        } else {
            mean = this->tmp_mean_;
            variance = this->tmp_variance_;
        }
    }
    auto idx_scale_shift = 1 + 2 * conf_.stats_is_src();
    auto scaleshift = reinterpret_cast<const data_t *>(
            this->input_memory(idx_scale_shift));
    auto ws = reinterpret_cast<uint8_t *>(this->memory(conf_.ws_idx()));
    data_t *ws_reduce = this->stats_reduction_;

    const float eps = conf_.desc()->batch_norm_epsilon;
    const bool use_scaleshift = conf_.use_scaleshift();
    ;
    const bool with_relu = conf_.with_relu_post_op();
    auto maybe_post_op
            = [&](data_t res) { return (with_relu && res < 0) ? 0 : res; };
    const bool has_spatial = utils::one_of(conf_.ndims(), 4, 5);
    int SP = (has_spatial) ? conf_.H() * conf_.W() * conf_.D() : 1;
    int N = conf_.MB();
    int C = conf_.C();

    int nthr = omp_get_max_threads();
    int l3_size_ = get_cache_size(3, true) * nthr / 2;
    int data_size = N * C * SP * sizeof(data_t);
    bool do_blocking = (data_size >= l3_size_ / 2 && l3_size_ > 0);
#pragma omp parallel
    {
        int C_blks_per_iter = 1, iters = 1, ithr = omp_get_thread_num();
        int C_ithr = 0, C_nthr = 0, N_ithr = 0, N_nthr = 0, N_s = 0, N_e = 0;
        int C_blk_gl_s = 0, C_blk_gl_e = 0, C_blk_s = 0, C_blk_e = 0;
        if (do_blocking) {
            size_t working_set_size = N * SP * sizeof(data_t);
            bnorm_utils::cache_balance(
                    working_set_size, C, C_blks_per_iter, iters);
        } else
            C_blks_per_iter = C;
        int last_iter_blks = C - (iters - 1) * C_blks_per_iter;

        bnorm_utils::thread_balance(do_blocking, ithr, nthr, N, C_blks_per_iter,
                C_ithr, C_nthr, C_blk_s, C_blk_e, N_ithr, N_nthr, N_s, N_e);
        balance211(C_blks_per_iter, nthr, ithr, C_blk_gl_s, C_blk_gl_e);

        for (int it = 0; it < iters; ++it) {
            if (it == iters - 1 && iters > 1) {
                C_blk_s = C_blk_e = N_s = N_e = 0;
                bnorm_utils::thread_balance(do_blocking, ithr, nthr, N,
                        last_iter_blks, C_ithr, C_nthr, C_blk_s, C_blk_e,
                        N_ithr, N_nthr, N_s, N_e);
                balance211(last_iter_blks, nthr, ithr, C_blk_gl_s, C_blk_gl_e);
            }
            int C_off = it * C_blks_per_iter;
            if (calculate_stats) {
                data_t *mean_blk = mean + C_off;
                data_t *variance_blk = variance + C_off;
                for (int c = C_blk_s; c < C_blk_e; c++) {
                    auto off = (c + C_off) * SP;
                    data_t sum = 0;
                    for (int n = N_s; n < N_e; ++n)
#pragma omp simd reduction(+ : sum)
                        for (int sp = 0; sp < SP; ++sp) {
                            sum += src[off + n * C * SP + sp];
                        }
                    ws_reduce[N_ithr * C_blks_per_iter + c] = sum;
                }
#pragma omp barrier
                for (int c = C_blk_gl_s; c < C_blk_gl_e; c++) {
                    mean_blk[c] = 0.;
                    for (int n = 0; n < N_nthr; n++)
                        mean_blk[c] += ws_reduce[n * C_blks_per_iter + c];
                    mean_blk[c] /= (N * SP);
                }
#pragma omp barrier
                for (int c = C_blk_s; c < C_blk_e; c++) {
                    auto off = c + C_off;
                    data_t sum = 0.;
                    for (int n = N_s; n < N_e; ++n)
#pragma omp simd reduction(+ : sum)
                        for (int sp = 0; sp < SP; ++sp) {
                            data_t m = src[off * SP + n * C * SP + sp]
                                    - mean[off];
                            sum += m * m;
                        }
                    ws_reduce[N_ithr * C_blks_per_iter + c] = sum;
                }
#pragma omp barrier
                for (int c = C_blk_gl_s; c < C_blk_gl_e; c++) {
                    variance_blk[c] = 0.;
                    for (int n = 0; n < N_nthr; n++)
                        variance_blk[c] += ws_reduce[n * C_blks_per_iter + c];
                    variance_blk[c] /= (N * SP);
                }
#pragma omp barrier
            }
            for (int c = C_blk_s; c < C_blk_e; c++) {
                auto off = c + C_off;
                data_t sm = use_scaleshift ? scaleshift[off] : 1;
                data_t sv = use_scaleshift ? scaleshift[C + off] : 0;
                data_t sqrt_variance
                        = static_cast<data_t>(1. / sqrt(variance[off] + eps));
                for (int n = N_s; n < N_e; ++n)
#pragma omp simd
                    for (int sp = 0; sp < SP; ++sp) {
                        auto d_off = off * SP + n * C * SP + sp;
                        data_t bn_res
                                = sm * (src[d_off] - mean[off]) * sqrt_variance
                                + sv;
                        if (conf_.fuse_bn_relu()) {
                            if (bn_res <= 0) {
                                bn_res = 0;
                                if (ws)
                                    ws[d_off] = 0;
                            } else {
                                if (ws)
                                    ws[d_off] = 1;
                            }
                        }
                        dst[d_off] = maybe_post_op(bn_res);
                    }
            }
        }
    }
}

ncsp_batch_normalization_bwd_t::ncsp_batch_normalization_bwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    , stats_reduction_(nullptr), tmp_diff_scaleshift_(nullptr) {
    this->stats_reduction_ = (data_t *)malloc(
            conf_.C() * 2 * omp_get_max_threads() * sizeof(data_t), 64);
    if (!(conf_.use_scaleshift()
                && conf_.desc()->prop_kind == prop_kind::backward))
        this->tmp_diff_scaleshift_
                = (data_t *)malloc(conf_.C() * 2 * sizeof(data_t), 64);
}

ncsp_batch_normalization_bwd_t::~ncsp_batch_normalization_bwd_t() {
    free(this->stats_reduction_);
    free(this->tmp_diff_scaleshift_);
}

void ncsp_batch_normalization_bwd_t::execute_backward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto mean = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto variance = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(3));
    auto scaleshift = reinterpret_cast<const data_t *>(this->input_memory(4));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_scaleshift = (this->memory(1)) ?
            reinterpret_cast<data_t *>(this->memory(1)) :
            this->tmp_diff_scaleshift_;
    auto ws = reinterpret_cast<const uint8_t *>(
            this->input_memory(conf_.ws_idx()));
    data_t *ws_reduce = this->stats_reduction_;

    const bool has_spatial = utils::one_of(conf_.ndims(), 4, 5);
    int SP = (has_spatial) ? conf_.H() * conf_.W() * conf_.D() : 1;
    auto C = conf_.C(), N = conf_.MB();
    const bool use_scaleshift = conf_.use_scaleshift();
    const float eps = conf_.desc()->batch_norm_epsilon;
    const bool calculate_diff_stats = !conf_.omit_stats();

    int nthr = omp_get_max_threads();
    int l3_size_ = get_cache_size(3, true) * nthr / 2;
    int data_size = N * C * SP * sizeof(data_t);
    bool do_blocking = (data_size >= l3_size_ / 2 && l3_size_ > 0);
#pragma omp parallel
    {
        int C_blks_per_iter = 1, iters = 1, ithr = omp_get_thread_num();
        int C_ithr = 0, C_nthr = 0, N_ithr = 0, N_nthr = 0, N_s = 0, N_e = 0;
        int C_blk_gl_s = 0, C_blk_gl_e = 0, C_blk_s = 0, C_blk_e = 0;
        if (do_blocking) {
            size_t working_set_size = 2 * N * SP * sizeof(data_t);
            bnorm_utils::cache_balance(
                    working_set_size, C, C_blks_per_iter, iters);
        } else
            C_blks_per_iter = C;
        int last_iter_blks = C - (iters - 1) * C_blks_per_iter;

        bnorm_utils::thread_balance(do_blocking, ithr, nthr, N, C_blks_per_iter,
                C_ithr, C_nthr, C_blk_s, C_blk_e, N_ithr, N_nthr, N_s, N_e);
        balance211(C_blks_per_iter, nthr, ithr, C_blk_gl_s, C_blk_gl_e);

        for (int it = 0; it < iters; ++it) {
            if (it == iters - 1 && iters > 1) {
                C_blk_s = C_blk_e = N_s = N_e = 0;
                bnorm_utils::thread_balance(do_blocking, ithr, nthr, N,
                        last_iter_blks, C_ithr, C_nthr, C_blk_s, C_blk_e,
                        N_ithr, N_nthr, N_s, N_e);
                balance211(last_iter_blks, nthr, ithr, C_blk_gl_s, C_blk_gl_e);
            }
            int C_off = it * C_blks_per_iter;
            data_t *diff_gamma_blk = diff_scaleshift + C_off;
            data_t *diff_beta_blk = diff_scaleshift + C + C_off;
            for (int c = C_blk_s; c < C_blk_e; c++) {
                auto off = c + C_off;
                data_t diff_gamma = 0.0, diff_beta = 0.0;
                data_t v_mean = mean[off];
                for (int n = N_s; n < N_e; ++n)
#pragma omp simd reduction(+ : diff_gamma, diff_beta)
                    for (int sp = 0; sp < SP; ++sp) {
                        const auto d_off = off * SP + n * C * SP + sp;
                        data_t dd;
                        if (ws)
                            dd = (!ws[d_off]) ? 0 : diff_dst[d_off];
                        else
                            dd = diff_dst[d_off];
                        diff_gamma += (src[d_off] - v_mean) * dd;
                        diff_beta += dd;
                    }
                ws_reduce[N_ithr * C_blks_per_iter + c] = diff_gamma;
                ws_reduce[N_nthr * C_blks_per_iter + N_ithr * C_blks_per_iter
                        + c]
                        = diff_beta;
            }
#pragma omp barrier
            for (int c = C_blk_gl_s; c < C_blk_gl_e; c++) {
                data_t sqrt_variance = static_cast<data_t>(
                        1. / sqrt(variance[c + C_off] + eps));
                diff_gamma_blk[c] = 0.;
                diff_beta_blk[c] = 0.;
                for (int n = 0; n < N_nthr; n++) {
                    diff_gamma_blk[c] += ws_reduce[n * C_blks_per_iter + c];
                    diff_beta_blk[c] += ws_reduce[N_nthr * C_blks_per_iter
                            + n * C_blks_per_iter + c];
                }
                diff_gamma_blk[c] *= sqrt_variance;
            }
#pragma omp barrier
            for (int c = C_blk_s; c < C_blk_e; c++) {
                auto off = c + C_off;
                data_t gamma = use_scaleshift ? scaleshift[off] : 1;
                data_t sqrt_variance
                        = static_cast<data_t>(1. / sqrt(variance[off] + eps));
                data_t v_mean = mean[off];
                for (int n = N_s; n < N_e; ++n)
#pragma omp simd
                    for (int sp = 0; sp < SP; ++sp) {
                        const auto d_off = off * SP + n * C * SP + sp;
                        ;
                        data_t v_diff_src;
                        if (ws)
                            v_diff_src = (!ws[d_off]) ? 0 : diff_dst[d_off];
                        else
                            v_diff_src = diff_dst[d_off];
                        if (calculate_diff_stats) {
                            v_diff_src -= diff_beta_blk[c] / (SP * N)
                                    + (src[d_off] - v_mean) * diff_gamma_blk[c]
                                            * sqrt_variance / (SP * N);
                        }
                        v_diff_src *= gamma * sqrt_variance;
                        diff_src[d_off] = v_diff_src;
                    }
            }
        }
    }
}
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
