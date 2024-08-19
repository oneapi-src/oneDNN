/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
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

#include <algorithm>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_batch_normalization_utils.hpp"

#include "cpu/nspc_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;
using namespace data_type;

template <data_type_t d_type>
status_t nspc_batch_normalization_fwd_t<d_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    const bool save_stats = pd()->is_training();
    const bool is_training = pd()->is_training();
    const bool fuse_norm_relu = pd()->fuse_norm_relu();
    const bool calculate_stats = !pd()->stats_is_src();
    const bool with_relu = pd()->with_relu_post_op(is_training);

    const auto use_scale = pd()->use_scale();
    const auto use_shift = pd()->use_shift();

    auto scratchpad = ctx.get_scratchpad_grantor();
    auto tmp_mean = scratchpad.template get<acc_data_t>(key_bnorm_tmp_mean);
    auto tmp_var = scratchpad.template get<acc_data_t>(key_bnorm_tmp_var);
    auto *ws_reduce = scratchpad.template get<acc_data_t>(key_bnorm_reduction);

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
    auto shift = CTX_IN_MEM(const float *, DNNL_ARG_SHIFT);

    acc_data_t *mean, *variance;
    if (!calculate_stats) {
        mean = const_cast<acc_data_t *>(
                CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN));
        variance = const_cast<acc_data_t *>(
                CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE));
    } else {
        if (save_stats) {
            mean = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_MEAN);
            variance = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_VARIANCE);
        } else {
            mean = tmp_mean;
            variance = tmp_var;
        }
    }

    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(uint8_t *, DNNL_ARG_WORKSPACE);
    acc_data_t *tmp_data_ = scratchpad.template get<acc_data_t>(key_bnorm_cvt);

    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();
    const int simd_w = 16;
    const dim_t C_align = utils::rnd_up(C, simd_w);
    const dim_t SP = pd()->H() * pd()->W() * pd()->D();

    const float eps = pd()->desc()->batch_norm_epsilon;
    auto maybe_post_op = [&](acc_data_t res) {
        if (with_relu) return math::relu_fwd(res, pd()->alpha());
        return res;
    };
    const int nthr = pd()->nthr_;

    if (calculate_stats) {
        parallel(nthr, [&](const int ithr, const int nthr) {
            dim_t N_s = 0, N_e = 0;
            balance211(N, nthr, ithr, N_s, N_e);

            for (dim_t c = 0; c < C; c++)
                ws_reduce[C * ithr + c] = 0.;

            for (dim_t n = N_s; n < N_e; n++) {
                for (dim_t sp = 0; sp < SP; sp++) {
                    const acc_data_t *_src;
                    const size_t s_off = (size_t)n * SP * C + sp * C;
                    if (utils::one_of(d_type, bf16, f16)) {
                        // convert src from xf16 to f32
                        acc_data_t *tmp_src = tmp_data_ + ithr * C_align;
                        types::cvt_to_float(tmp_src, src + s_off, C);
                        _src = tmp_src;
                    } else {
                        _src = reinterpret_cast<const acc_data_t *>(
                                src + s_off);
                    }
                    PRAGMA_OMP_SIMD()
                    for (int c = 0; c < C; c++) {
                        ws_reduce[C * ithr + c] += _src[c];
                    }
                }
            }
        });
        parallel_nd(C, [&](dim_t c) {
            mean[c] = 0;
            for (dim_t n = 0; n < nthr; n++)
                mean[c] += ws_reduce[C * n + c];
            mean[c] /= SP * N;
        });
        parallel(nthr, [&](const int ithr, const int nthr) {
            dim_t N_s = 0, N_e = 0;
            balance211(N, nthr, ithr, N_s, N_e);

            acc_data_t *mean_loc = tmp_mean + nstl::max(C, (dim_t)16) * ithr;

            if (ithr > 0 || save_stats) {
                for (dim_t c = 0; c < C; c++)
                    mean_loc[c] = mean[c];
            }

            for (dim_t c = 0; c < C; c++)
                ws_reduce[C * ithr + c] = 0.;

            for (dim_t n = N_s; n < N_e; n++) {
                for (dim_t sp = 0; sp < SP; sp++) {
                    const acc_data_t *_src;
                    const size_t s_off = (size_t)n * SP * C + sp * C;
                    if (utils::one_of(d_type, bf16, f16)) {
                        // convert src from xf16 to f32
                        acc_data_t *tmp_src = tmp_data_ + ithr * C_align;
                        types::cvt_to_float(tmp_src, src + s_off, C);
                        _src = tmp_src;
                    } else {
                        _src = reinterpret_cast<const acc_data_t *>(
                                src + s_off);
                    }
                    PRAGMA_OMP_SIMD()
                    for (int c = 0; c < C; c++) {
                        acc_data_t m = _src[c] - mean_loc[c];
                        ws_reduce[C * ithr + c] += m * m;
                    }
                }
            }
        });
        parallel_nd(C, [&](dim_t c) {
            variance[c] = 0;
            for (dim_t n = 0; n < nthr; n++)
                variance[c] += ws_reduce[C * n + c];
            variance[c] /= SP * N;
        });
        parallel(nthr, [&](const int ithr, const int nthr) {
            acc_data_t *variance_loc = tmp_var + nstl::max(C, (dim_t)16) * ithr;
            if (ithr > 0 || save_stats) {
                for (dim_t c = 0; c < C; c++)
                    variance_loc[c] = variance[c];
            }
        });
    }

    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t N_s = 0, N_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);

        acc_data_t *mean_loc, *variance_loc;
        if (calculate_stats) {
            mean_loc = tmp_mean + nstl::max(C, (dim_t)16) * ithr;
            variance_loc = tmp_var + nstl::max(C, (dim_t)16) * ithr;
        } else {
            mean_loc = mean;
            variance_loc = variance;
        }

        for (dim_t n = N_s; n < N_e; n++) {
            for (dim_t sp = 0; sp < SP; sp++) {
                acc_data_t *_dst;
                const acc_data_t *_src;
                const size_t s_off = (size_t)n * SP * C + sp * C;
                if (utils::one_of(d_type, bf16, f16)) {
                    // store dst to f32 buffer
                    _dst = tmp_data_ + ithr * C_align;
                    // convert src from xf16 to f32
                    acc_data_t *tmp_src = tmp_data_ + (nthr + ithr) * C_align;
                    types::cvt_to_float(tmp_src, src + s_off, C);
                    _src = tmp_src;
                } else {
                    _dst = reinterpret_cast<acc_data_t *>(dst + s_off);
                    _src = reinterpret_cast<const acc_data_t *>(src + s_off);
                }
#if CLANG_WA_02_SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (int c = 0; c < C; c++) {
                    const size_t c_off = s_off + c;
                    acc_data_t sqrt_variance = static_cast<acc_data_t>(
                            sqrtf(variance_loc[c] + eps));
                    acc_data_t sm = (use_scale ? (acc_data_t)scale[c]
                                               : (acc_data_t)1.0f)
                            / sqrt_variance;
                    acc_data_t sv
                            = use_shift ? (acc_data_t)shift[c] : (acc_data_t)0;
                    acc_data_t bn_res = sm * (_src[c] - mean_loc[c]) + sv;
                    if (fuse_norm_relu) {
                        if (bn_res <= 0) {
                            bn_res = 0;
                            if (is_training) ws[c_off] = 0;
                        } else {
                            if (is_training) ws[c_off] = 1;
                        }
                    }
                    _dst[c] = maybe_post_op(bn_res);
                }
                if (utils::one_of(d_type, bf16, f16)) {
                    // convert dst from f32 to xf16
                    types::cvt_from_float(dst + s_off, _dst, C);
                }
            }
        }
    });
    return status::success;
}

template struct nspc_batch_normalization_fwd_t<f32>;
template struct nspc_batch_normalization_fwd_t<bf16>;
template struct nspc_batch_normalization_fwd_t<f16>;

template <data_type_t d_type>
status_t nspc_batch_normalization_bwd_t<d_type>::execute_backward(
        const exec_ctx_t &ctx) const {

    const auto use_scale = pd()->use_scale();

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto mean = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN);
    auto variance = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE);
    auto scale = CTX_IN_MEM(acc_data_t *, DNNL_ARG_SCALE);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const uint8_t *, DNNL_ARG_WORKSPACE);

    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);
    auto diff_scale = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_DIFF_SCALE);
    auto diff_shift = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_DIFF_SHIFT);

    auto scratchpad = ctx.get_scratchpad_grantor();
    auto tmp_diff_ss
            = scratchpad.template get<acc_data_t>(key_bnorm_tmp_diff_ss);

    const size_t scratch_diff_shift_off = diff_scale ? 0 : pd()->C();
    if (diff_scale == nullptr) diff_scale = tmp_diff_ss;
    if (diff_shift == nullptr)
        diff_shift = &tmp_diff_ss[scratch_diff_shift_off];

    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();
    const int simd_w = 16;
    const dim_t C_align = utils::rnd_up(C, simd_w);
    const dim_t SP = pd()->D() * pd()->H() * pd()->W();
    acc_data_t *diff_gamma = diff_scale, *diff_beta = diff_shift;
    acc_data_t *ws_reduce
            = scratchpad.template get<acc_data_t>(key_bnorm_reduction);
    acc_data_t *tmp_data_ = scratchpad.template get<acc_data_t>(key_bnorm_cvt);

    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool calculate_diff_stats = !pd()->use_global_stats();
    const bool fuse_norm_relu = pd()->fuse_norm_relu();

    /* Note: potential seg-fault from incorrectly compiled vectorized-loop.
     * Explicit tail-processing fixes this issue. */
    const dim_t c_blk = std::max(
            platform::get_vector_register_size() / (int)sizeof(float), 8);
    const dim_t tail = C % c_blk;
    const dim_t nb_c_blk = (size_t)C / c_blk;
    const int nthr = pd()->nthr_;

    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t N_s = 0, N_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);

        for (dim_t c = 0; c < C; c++) {
            ws_reduce[C * ithr + c] = 0.;
            ws_reduce[C * nthr + C * ithr + c] = 0.;
        }

        for (dim_t n = N_s; n < N_e; n++) {
            for (dim_t sp = 0; sp < SP; sp++) {
                const acc_data_t *_diff_dst;
                const acc_data_t *_src;
                const size_t s_off = (size_t)n * SP * C + sp * C;
                if (utils::one_of(d_type, bf16, f16)) {
                    // convert diff_dst to f32
                    acc_data_t *tmp_diff_dst = tmp_data_ + ithr * C_align;
                    types::cvt_to_float(tmp_diff_dst, diff_dst + s_off, C);
                    _diff_dst = tmp_diff_dst;
                    // convert src to f32
                    acc_data_t *tmp_src = tmp_data_ + (nthr + ithr) * C_align;
                    types::cvt_to_float(tmp_src, src + s_off, C);
                    _src = tmp_src;
                } else {
                    _diff_dst = reinterpret_cast<const acc_data_t *>(
                            diff_dst + s_off);
                    _src = reinterpret_cast<const acc_data_t *>(src + s_off);
                }
#if CLANG_WA_02_SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (dim_t c = 0; c < C; c++) {
                    const size_t c_off = s_off + c;
                    acc_data_t dd;
                    if (fuse_norm_relu && !ws[c_off])
                        dd = 0;
                    else
                        dd = _diff_dst[c];
                    ws_reduce[C * ithr + c] += (_src[c] - mean[c]) * dd;
                    ws_reduce[C * nthr + C * ithr + c] += dd;
                }
            }
        }
    });

    parallel_nd(C, [&](dim_t c) {
        acc_data_t sqrt_variance
                = static_cast<acc_data_t>(1.0f / sqrtf(variance[c] + eps));
        diff_gamma[c] = 0;
        diff_beta[c] = 0;
        for (dim_t n = 0; n < nthr; n++) {
            diff_gamma[c] += ws_reduce[C * n + c];
            diff_beta[c] += ws_reduce[C * nthr + C * n + c];
        }
        diff_gamma[c] *= sqrt_variance;
    });

    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t N_s = 0, N_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);

        acc_data_t *diff_gamma_loc = tmp_diff_ss + 2 * C + C * ithr;
        acc_data_t *diff_beta_loc = tmp_diff_ss + 2 * C + C * (nthr + ithr);

        for (dim_t c = 0; c < C; c++) {
            diff_gamma_loc[c] = diff_gamma[c];
            diff_beta_loc[c] = diff_beta[c];
        }

        for (dim_t n = N_s; n < N_e; n++) {
            for (dim_t sp = 0; sp < SP; sp++) {
                acc_data_t *_diff_src;
                const acc_data_t *_diff_dst;
                const acc_data_t *_src;
                const size_t s_off = (size_t)n * SP * C + sp * C;
                if (utils::one_of(d_type, bf16, f16)) {
                    // store diff_src to f32 buffer
                    _diff_src = tmp_data_ + ithr * C_align;
                    // convert diff_dst to f32
                    acc_data_t *tmp_diff_dst = tmp_data_ + ithr * C_align;
                    types::cvt_to_float(tmp_diff_dst, diff_dst + s_off, C);
                    _diff_dst = tmp_diff_dst;
                    if (calculate_diff_stats) {
                        // convert src to f32
                        acc_data_t *tmp_src
                                = tmp_data_ + (2 * nthr + ithr) * C_align;
                        types::cvt_to_float(tmp_src, src + s_off, C);
                        _src = tmp_src;
                    } else
                        _src = nullptr; // to avoid compiler warning w/ gcc483
                } else {
                    _diff_src
                            = reinterpret_cast<acc_data_t *>(diff_src + s_off);
                    _diff_dst = reinterpret_cast<const acc_data_t *>(
                            diff_dst + s_off);
                    _src = reinterpret_cast<const acc_data_t *>(src + s_off);
                }

#if CLANG_WA_02_SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD(simdlen(16))
#endif
                for (dim_t c = 0; c < nb_c_blk * c_blk; c++) {
                    const size_t c_off = s_off + c;
                    acc_data_t gamma = use_scale ? scale[c] : 1;
                    acc_data_t sqrt_variance = static_cast<acc_data_t>(
                            1.0f / sqrtf(variance[c] + eps));
                    acc_data_t v_diff_src;
                    if (fuse_norm_relu && !ws[c_off])
                        v_diff_src = 0;
                    else
                        v_diff_src = _diff_dst[c];
                    if (calculate_diff_stats) {
                        v_diff_src -= diff_beta_loc[c] / (SP * N)
                                + (_src[c] - mean[c]) * diff_gamma_loc[c]
                                        * sqrt_variance / (SP * N);
                    }
                    v_diff_src *= gamma * sqrt_variance;
                    _diff_src[c] = v_diff_src;
                }
                for (dim_t c = 0; c < tail; c++) {
                    const size_t c_off = s_off + nb_c_blk * c_blk + c;
                    acc_data_t gamma
                            = use_scale ? scale[nb_c_blk * c_blk + c] : 1;
                    acc_data_t sqrt_variance = static_cast<acc_data_t>(
                            1.0f / sqrtf(variance[nb_c_blk * c_blk + c] + eps));
                    acc_data_t v_diff_src;
                    if (fuse_norm_relu && !ws[c_off])
                        v_diff_src = 0;
                    else
                        v_diff_src = _diff_dst[nb_c_blk * c_blk + c];
                    if (calculate_diff_stats) {
                        v_diff_src -= diff_beta_loc[nb_c_blk * c_blk + c]
                                        / (SP * N)
                                + (_src[nb_c_blk * c_blk + c]
                                          - mean[nb_c_blk * c_blk + c])
                                        * diff_gamma_loc[nb_c_blk * c_blk + c]
                                        * sqrt_variance / (SP * N);
                    }
                    v_diff_src *= gamma * sqrt_variance;
                    _diff_src[nb_c_blk * c_blk + c] = v_diff_src;
                }
                if (utils::one_of(d_type, bf16, f16)) {
                    // convert diff_src from f32
                    types::cvt_from_float(diff_src + s_off, _diff_src, C);
                }
            }
        }
    });
    return status::success;
}

template struct nspc_batch_normalization_bwd_t<f32>;
template struct nspc_batch_normalization_bwd_t<bf16>;
template struct nspc_batch_normalization_bwd_t<f16>;
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
