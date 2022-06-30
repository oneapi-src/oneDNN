/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "bnorm/bnorm.hpp"
#include "lnorm/lnorm.hpp"

using namespace bnorm;

namespace lnorm {

int prepare_fwd(const prb_t *prb, dnn_mem_t &src, dnn_mem_t &mean,
        dnn_mem_t &var, dnn_mem_t &ss, dnn_mem_t &sh) {
    /** Idea: choose src[] values so that both mean and variance are computed
     * exactly (independently of the order of the computations).
     *
     * The `exactness` is achieved via [a1]: src[i] + src[i+1] = 2 * mean.
     *
     * The variation in src is allowed in the last flex_bits bits.
     * If the sequence (L) is too big (flex_bits <= min_flex_bits), the mean
     * value is set to 0 and src is partially filled with zeros (according to
     * density so that at least want_flex_bits is reserved for src variation.
     * Once src is set, variance is computed.
     *
     * ALG_0: mean is set to 0
     * ALG_1: mean is set to 2^prb, where prb \in {-2, -1, ..., 4}
     * ALG_AUTO: choose between ALG_0 and ALG_1 automatically */
    const int64_t exact_bits = digits_dt(prb->dt);
    const int64_t L = prb->c;
    const int64_t logL = (int64_t)ceilf(log2f(L));

    assert(logL <= 0 || (1LL << (logL - 1)) < L);
    assert(L <= (1LL << logL));

    const int64_t min_flex_bits = 3;
    const int64_t want_flex_bits = MIN2(6, exact_bits / 2);

    check_alg_t alg = prb->check_alg;
    if (alg == ALG_AUTO) /* choose appropriate checking algorithm */
        alg = (exact_bits - logL) / 2 - 1 >= min_flex_bits ? ALG_1 : ALG_0;

    const int64_t flex_bits = alg == ALG_0
            ? want_flex_bits /* BFloat16 has only 7 bits of mantissa */
            : MIN2(prb->dt == dnnl_bf16 ? 7 : exact_bits,
                    (exact_bits - logL) / 2 - 1);

    if (flex_bits < min_flex_bits) return FAIL;

    const int64_t flex_mask = (1 << flex_bits) - 1;

    /* density: (exact_bits - log_2(L * density)) / 2 >= flex_bits */
    const float density = alg == ALG_0
            ? 1.f * (1 << (exact_bits - 2 * flex_bits)) / L
            : 1.f;
    assert((exact_bits - ceilf(log2f(L * density))) / 2 >= flex_bits);

    BENCHDNN_PRINT(6, "check_alg: %s, density = %g, flex_bits = " IFMT "\n",
            check_alg2str(alg), density, flex_bits);

    benchdnn_parallel_nd(prb->n, [&](int64_t n) {
        const float m = alg == ALG_0 ? 0.f : 0.25f * (1 << (n % 7));
        float v = 0; /* current variance */

        float *s = (float *)src + n * prb->c;
        for (int64_t c = 0; c < prb->c; ++c) {
            const int64_t l = c + n * 239 * 2; // l[0] must be even

            if (alg == ALG_0 && !flip_coin(l / 2 * 257ULL, density)) {
                s[c] = 0;
                continue;
            }

            const int64_t gen = (l / 2 * 1637) & flex_mask;
            const int sgn = l % 2 == 0 ? 1 : -1; /* [a1] */
            const float f = 1.f * sgn * gen / (1 << flex_bits);

            src.set_elem(n * prb->c + c, alg == ALG_0 ? f : m * (1.f + f));
            if (L % 2 && (c == L - 1)) { s[c] = m; }
            v += (s[c] - m) * (s[c] - m);
        }
        mean.set_elem(n, m);
        var.set_elem(n, v / prb->c);
    });

    const bool use_ss = prb->use_ss();
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    benchdnn_parallel_nd(prb->c, [&](int64_t c) {
        float sc_value = 1.f / 8 * (1 << (c % 7));
        float sh_value = (c % 3 + 1) * sc_value / 64;
        if (use_sc || use_sh) {
            ((float *)ss)[c] = use_sc ? sc_value : 1.0f;
            ((float *)sh)[c] = use_sh ? sh_value : 0.0f;
        } else {
            ((float *)ss)[c] = use_ss ? sc_value : 1.0f;
            ((float *)ss)[prb->c + c] = use_ss ? sh_value : 0.0f;
        }
    });
    return OK;
}
/** @brief L = 2^k * P, P % 2 != 0 */
static void decompose2(int64_t L, int64_t &k, int64_t &P) {
    P = L;
    for (k = 0; P % 2 == 0; ++k)
        P /= 2;
}
int prepare_bwd(const prb_t *prb, dnn_mem_t &src, dnn_mem_t &d_dst,
        dnn_mem_t &mean, dnn_mem_t &var, dnn_mem_t &ss, dnn_mem_t &sh) {
    const int64_t exact_bits = 24;

    if (prb->c < 2) return FAIL;

    const int64_t L = prb->c;
    /** Stabilization idea...
     * Layer Normalization (unlike batch normalization) features two types of
     * accumulations in bwd step:
     * First, accumulation over n:
     *      d_gamma[c] = sum_over_n ddst[n, c] * (src[n, c] - mean[n]) * inv_sigma
     *      d_beta[c] = ...
     * Second, accumulation over c:
     *      dd_gamma[n] = sum_over_c ddst[n, c] * (src[n, c] - mean[n])
     *          * inv_sigma * gamma
     *      dd_gamma_x[n] = ...
     * that is used when computing d_src:
     *      d_src = func(dd_gamma / C, dd_gamma_x / C, ...)
     * To avoid accumulation error in the first case we will force sparsity
     * of ddst over n if d_gamma and d_beta need to be computed.
     * To get exact result of division in the second case we use the same
     * approach as in batch normalization:
     * Try to make dd_gamma = L / 2^t_dd_gamma and dd_gamma_x = L / 2^t_dd_gamma_x,
     * where both t_dd_gamma and t_dd_gamma_x are in {1, .., max_k}.
     * Currently, with no obvious reason, max_k is set to 4 for
     * reasonably small problems and to 8 for big problems.
     *
     * We might hope that division by L would be exact in that case,
     * but that might happen iff L is less than 2^exact_bits, hence
     * restriction [r1].
     * */

    int64_t k, P;
    decompose2(L, k, P);

    int64_t log2P = (int64_t)ceilf(log2f(P));
    if (log2P >= exact_bits) return FAIL; /* [r1] */

    const int64_t max_k = 4;
    if (k > max_k && exact_bits - log2P > max_k + 4) {
        log2P += (k - max_k);
        P <<= k - max_k;
        k = max_k;
    }

    const int64_t param_dd_p2 = 7; // factor_dd <- 2^{0, .., -param_dd_p2+1}
    const int64_t param_dd_gen = 32; // gen_dd <- {1, .., param_dd_gen}

    const int64_t param_f_p2 = 1; // factor_f <- 2^{-1, ..., -param_f_p2}
    const int64_t param_f_gen = 16; // gen_f <- {2, ..., param_s_gen}

    const bool use_ss = prb->use_ss();
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    const float density
            = (use_ss || use_sc || use_sh) ? MIN2(1.f, 10.f / prb->n) : 1.f;

    BENCHDNN_PRINT(5,
            "prep_bwd: k:" IFMT ", P:" IFMT " log2P:" IFMT ", density = %g\n",
            k, P, log2P, density);

    // fill gamma and beta
    for (int64_t c = 0; c < prb->c; ++c) {
        const float sc_value = 1.f / 8 * (1 << (c % 7));
        const float sh_value = sc_value / 64;
        if (use_sc || use_sh) {
            ((float *)ss)[c] = use_sc ? sc_value : 1.0f;
            ((float *)sh)[c] = use_sh ? sh_value : 0.0f;
        } else {
            ((float *)ss)[c] = use_ss ? sc_value : 1.0f;
            ((float *)ss)[prb->c + c] = use_ss ? sh_value : 0.0f;
        }
    }

    for (int64_t n = 0; n < prb->n; ++n) {
        const float m = ((float *)mean)[n] = n % 2;

        /* var + eps \in {1/4, 1, 4} */
        const float ve_denom = 4.f / (1 << 2 * (n % 3));
        ((float *)var)[n] = ve_denom - prb->eps;

        const int64_t dd_p2 = (n * 127 % param_dd_p2);
        const float factor_dd = 1.f / (1 << dd_p2);
        const int64_t f_p2 = 1 + (n % param_f_p2);
        const float factor_f = 1.f / (1 << f_p2);

        const float target_dd_g = factor_dd * P;
        const float target_dd_g_x = 2 * target_dd_g;

        if (!flip_coin(n, density) && n != 0 && n != prb->n - 1) {
            for (int64_t c = 0; c < prb->c; ++c) {
                ((float *)d_dst)[n * prb->c + c] = 0;
                ((float *)src)[n * prb->c + c] = m;
            }
            continue;
        }
        float dd_g = 0, dd_g_x = 0; /* current dd_gamma and dd_gamma_x */
        for (int64_t c = 0; c < prb->c - 2; ++c) {
            const float g = ((float *)ss)[c];
            float &s = ((float *)src)[n * prb->c + c];
            float &dd = ((float *)d_dst)[n * prb->c + c];

            const int sgn_dd = dd_g < target_dd_g ? 1 : -1;
            dd = sgn_dd * factor_dd * (1 + ((c + n) * 3 % param_dd_gen));
            dd_g += dd * g;

            const int sgn_f = dd_g_x < target_dd_g_x ? 1 : -1;
            const float f = sgn_f * factor_f
                    * (2 + ((c + n) * 7 % (param_f_gen - 1)));

            dd_g_x += f * dd * g;
            s = f + m;
        }

        /* the last 2 elements in src and d_dst are set, so that:
         *      dd_gamma == target_dd_gamma
         *      dd_gamma_x == target_dd_gamma_x
         * For this we need to solve the system:
         *      d_dst[l1] * g[c1]           + d_dst[l0] * g[c0]
         *          = target_dd_gamma - dd_gamma
         *      d_dst[l1] * src[l1] * g[c1] + d_dst[l0] * src[l0] * g[c0]
         *          = target_dd_gamam_x - dd_gamma_x
         *
         * Here l0 -- last index, l1 -- last but one.
         * More over, let's assume src[l1] = 1 and src[l0] = -1. */
        int64_t l0 = n * prb->c + prb->c - 1;
        int64_t l1 = n * prb->c + prb->c - 2;

        ((float *)src)[l1] = 1.f + m;
        ((float *)src)[l0] = -1.f + m;
        const float g1 = ((float *)ss)[prb->c - 2];
        const float g0 = ((float *)ss)[prb->c - 1];

        float f1 = ((target_dd_g - dd_g) + (target_dd_g_x - dd_g_x)) / 2;
        float f0 = ((target_dd_g - dd_g) - (target_dd_g_x - dd_g_x)) / 2;

        ((float *)d_dst)[l1] = f1 / g1;
        ((float *)d_dst)[l0] = f0 / g0;

        if (prb->dt == dnnl_bf16) { // truncate to bf16
            ((uint16_t *)(&((float *)d_dst)[l1]))[0] = 0;
            ((uint16_t *)(&((float *)d_dst)[l0]))[0] = 0;
        }
    }

    return OK;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    dnnl_layer_normalization_desc_t ld;

    const int64_t *data_dims = &prb->dims[0];

    auto data_d = dnn_mem_t::init_md(prb->ndims, data_dims, prb->dt, prb->tag);

    dnnl_memory_desc_t stat_d;
    const dnnl_memory_desc_t *stat_d_ptr = nullptr;
    if (prb->stat_tag != tag::undef) {
        stat_d = dnn_mem_t::init_md(
                prb->ndims - 1, data_dims, dnnl_f32, prb->stat_tag);
        stat_d_ptr = &stat_d;
    }

    auto flags = (dnnl_normalization_flags_t)prb->flags;
    if (prb->dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        DNN_SAFE_STATUS(dnnl_layer_normalization_forward_desc_init(
                &ld, prop, &data_d, stat_d_ptr, prb->eps, flags));
    } else {
        auto diff_data_d
                = dnn_mem_t::init_md(prb->ndims, data_dims, prb->dt, tag::any);
        auto prop = prb->dir & FLAG_WEI ? dnnl_backward : dnnl_backward_data;
        DNN_SAFE_STATUS(dnnl_layer_normalization_backward_desc_init(
                &ld, prop, &diff_data_d, &data_d, stat_d_ptr, prb->eps, flags));
    }

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    return dnnl_primitive_desc_create(&init_pd_args.pd, &ld, dnnl_attr,
            init_pd_args.engine, init_pd_args.hint);
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->dt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    if (prb->use_ss() && (prb->use_sc() || prb->use_sh()))
        res->state = SKIPPED, res->reason = INVALID_CASE;

    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        skip_invalid_inplace(res, prb->dt, prb->dt, prb->tag, prb->tag);
        if (res->state == SKIPPED) return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const int f32_mant_digits = 24;
    const float trh_coeff = (1 << (f32_mant_digits - digits_dt(prb->dt)));
    float trh = trh_coeff * ((kind == SRC || kind == DST) ? 5e-7 : 0);
    if ((kind == SS || kind == SC || kind == SH) && prb->dir & FLAG_BWD)
        trh = trh_coeff * 5e-6;
    cmp.set_threshold(trh);

    // TODO: improve bf16 filling
    if (prb->dt == dnnl_bf16) cmp.set_zero_trust_percent(99.f);

    // When the error is larger than `trh`, it could be due to a catastrophic
    // cancellation in final result which is computed as `Y = a * X + b`.
    // When `a * X` is close to `b` and their signs are opposite, then large
    // error in `a * X` could result in a final result (which has a cancellation
    // i.e. `|Y| = |a*X - (-b)|`), which has no meaningful digits left in
    // mantissa.
    //
    // Since lambda is called when stack is unavailable, need to capture `prb`
    // and `kind` by value to avoid using dangling references.
    const auto lnorm_add_check =
            [&, kind, prb](
                    const compare::compare_t::driver_check_func_args_t &args) {
                const bool has_shift = prb->use_sh() || prb->use_ss();
                if (!((prb->dir & FLAG_FWD) && kind == DST && has_shift))
                    return false;

                const auto &ss = ref_args.find(DNNL_ARG_SCALE_SHIFT);
                const auto &sh = ref_args.find(DNNL_ARG_SHIFT);
                const bool shift_only = prb->use_sh();
                const float *sh_ptr
                        = shift_only ? (const float *)sh : (const float *)ss;

                const auto &dst = ref_args.find(DNNL_ARG_DST);
                const int64_t c = dst.get_scale_idx(
                        args.idx, 1 << (prb->ndims - 1) /* last_dim_mask */);
                const int64_t c_idx = c + (shift_only ? 0 : prb->c);
                const float beta = sh_ptr[c_idx];
                // Using an empirically derived threshold, check if
                // cancellation error in `|Y| = |a*X - (-b)|` is huge.
                const float abs_exp = fabsf(args.exp);
                const float norm_denom = abs_exp > FLT_MIN ? abs_exp : 1.f;
                const float abs_exp_delta = fabsf(args.exp - beta);
                bool maybe_cancel_error = abs_exp_delta / norm_denom > 1.f;
                if (!maybe_cancel_error) return false;

                // Check for error in `a * X`
                float diff_aX = fabsf((args.exp - beta) - (args.got - beta));
                float rel_diff_aX = diff_aX
                        / (abs_exp_delta > FLT_MIN ? abs_exp_delta : 1.f);
                return rel_diff_aX <= args.trh;
            };
    cmp.set_driver_check_function(lnorm_add_check);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);

    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const bool use_ss = prb->use_ss();
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    const auto &data_md = query_md(const_pd, DNNL_ARG_SRC);
    const auto &mean_md = query_md(const_pd, DNNL_ARG_MEAN);
    const auto &var_md = query_md(const_pd, DNNL_ARG_VARIANCE);
    const auto &ss_md = query_md(const_pd, DNNL_ARG_SCALE_SHIFT);
    const auto &scratchpad_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    dnn_mem_t src_fp(data_md, fp, tag, ref_engine);
    dnn_mem_t src_dt(data_md, test_engine);

    dnn_mem_t &dst_fp = src_fp; // in-place reference
    dnn_mem_t placeholder_dst_dt;
    if (!prb->inplace) { placeholder_dst_dt = dnn_mem_t(data_md, test_engine); }
    dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    // On inference w/o global stats the layer norm doesn't require stat
    // memories. Hence, we need to prepare the mean_fp and var_fp ourselves.
    const auto stat_ndims = prb->ndims - 1;
    const auto stat_tag = tag::abx;
    dnn_mem_t mean_fp(stat_ndims, data_md.dims, fp, stat_tag, ref_engine);
    dnn_mem_t mean_dt(mean_md, test_engine);

    dnn_mem_t var_fp(stat_ndims, data_md.dims, fp, stat_tag, ref_engine);
    dnn_mem_t var_dt(var_md, test_engine);

    dnn_mem_t ss_fp(ss_md, fp, tag::abx, ref_engine);
    dnn_mem_t ss_dt(ss_md, test_engine);
    dnn_mem_t d_ss_fp(ss_md, fp, tag::abx, ref_engine);
    dnn_mem_t d_ss_dt(ss_md, test_engine);

    dnn_mem_t sh_fp(ss_md, fp, use_sh ? tag::x : tag::abx, ref_engine);
    dnn_mem_t sh_dt(ss_md, test_engine);
    dnn_mem_t d_sh_fp(ss_md, fp, use_sh ? tag::x : tag::abx, ref_engine);
    dnn_mem_t d_sh_dt(ss_md, test_engine);

    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    dnn_mem_t d_dst_dt, placeholder_d_src_dt;

    args_t args, ref_args;

    if (prb->dir & FLAG_FWD) {
        if (prepare_fwd(prb, src_fp, mean_fp, var_fp, ss_fp, sh_fp) != OK) {
            return res->state = MISTRUSTED, OK;
        }

        SAFE(src_dt.reorder(src_fp), WARN);
        if (prb->flags & GLOB_STATS) {
            /* prepare mean & var if they are inputs */
            SAFE(mean_dt.reorder(mean_fp), WARN);
            SAFE(var_dt.reorder(var_fp), WARN);
        }
        if (use_ss || use_sc) { SAFE(ss_dt.reorder(ss_fp), WARN); }
        if (use_sh) { SAFE(sh_dt.reorder(sh_fp), WARN); }

        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_MEAN, mean_dt);
        args.set(DNNL_ARG_VARIANCE, var_dt);
        args.set(use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT, ss_dt);
        args.set(DNNL_ARG_SHIFT, sh_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT, ss_fp);
            ref_args.set(DNNL_ARG_SHIFT, sh_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);

            std::vector<data_kind_t> kinds {DST};
            if (!(prb->flags & GLOB_STATS) && !(prb->dir & FLAG_INF)) {
                kinds.push_back(MEAN);
                kinds.push_back(VAR);
            }

            check_correctness(prb, kinds, args, ref_args, setup_cmp, res);
        }
    } else {
        const auto &d_data_md = query_md(const_pd, DNNL_ARG_DIFF_DST);

        dnn_mem_t d_dst_fp(d_data_md, fp, tag, ref_engine);
        d_dst_dt = dnn_mem_t(d_data_md, test_engine);

        dnn_mem_t &d_src_fp = d_dst_fp; // in-place in ref code
        if (!prb->inplace) {
            placeholder_d_src_dt = dnn_mem_t(d_data_md, test_engine);
        }
        dnn_mem_t &d_src_dt = prb->inplace ? d_dst_dt : placeholder_d_src_dt;

        if (prepare_bwd(prb, src_fp, d_dst_fp, mean_fp, var_fp, ss_fp, sh_fp)
                != OK) {
            return res->state = MISTRUSTED, OK;
        }

        SAFE(src_dt.reorder(src_fp), WARN);
        SAFE(d_dst_dt.reorder(d_dst_fp), WARN);
        SAFE(mean_dt.reorder(mean_fp), WARN);
        SAFE(var_dt.reorder(var_fp), WARN);
        if (use_ss || use_sc) { SAFE(ss_dt.reorder(ss_fp), WARN); }
        if (use_sh) { SAFE(sh_dt.reorder(sh_fp), WARN); }

        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_MEAN, mean_dt);
        args.set(DNNL_ARG_VARIANCE, var_dt);
        args.set(use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT, ss_dt);
        args.set(use_sc ? DNNL_ARG_DIFF_SCALE : DNNL_ARG_DIFF_SCALE_SHIFT,
                d_ss_dt);
        args.set(DNNL_ARG_SHIFT, sh_dt);
        args.set(DNNL_ARG_DIFF_SHIFT, d_sh_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT, ss_fp);
            ref_args.set(DNNL_ARG_SHIFT, sh_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);
            ref_args.set(
                    use_sc ? DNNL_ARG_DIFF_SCALE : DNNL_ARG_DIFF_SCALE_SHIFT,
                    d_ss_fp);
            ref_args.set(DNNL_ARG_DIFF_SHIFT, d_sh_fp);

            std::vector<data_kind_t> kinds {SRC};
            if ((use_ss || use_sc) && (prb->dir & FLAG_WEI))
                kinds.push_back(use_sc ? SC : SS);
            if (use_sh && (prb->dir & FLAG_WEI)) kinds.push_back(SH);

            check_correctness(prb, kinds, args, ref_args, setup_cmp, res);
        }
    }

    return measure_perf(res, prim, args);
}

} // namespace lnorm
