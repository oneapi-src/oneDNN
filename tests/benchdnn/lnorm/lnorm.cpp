/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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
#include <float.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "utils/fill.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "bnorm/bnorm.hpp"
#include "lnorm/lnorm.hpp"

using namespace bnorm;

namespace lnorm {

static int prepare_fwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &mean, const dnn_mem_t &var, const dnn_mem_t &sc,
        const dnn_mem_t &sh, res_t *res) {
    // Fast exit for zero channels as logic relies on it to be non-zero.
    if (prb->c == 0) {
        benchdnn_parallel_nd(prb->n, [&](int64_t n) {
            mean.set_elem(n, 0);
            var.set_elem(n, 0);
        });
        return OK;
    }

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
     * ALG_AUTO: choose between ALG_0 and ALG_1 automatically
     * ALG_2: if fall back to ALG_0 gives only one non-zero element, use the
     *        filling which doesn't use strict approach.
     */
    const int64_t exact_bits = digits_dt(prb->dt[0]);
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
            ? want_flex_bits
            : MIN2(exact_bits, (exact_bits - logL) / 2 - 1);
    if (flex_bits < min_flex_bits) {
        res->state = UNTESTED;
        return FAIL;
    }

    if (exact_bits / 2 == flex_bits) alg = ALG_2;

    if ((alg == ALG_0 || alg == ALG_1) && !is_integral_dt(prb->dt[0])) {
        const int64_t flex_mask = (static_cast<int64_t>(1) << flex_bits) - 1;

        /* density: (exact_bits - log_2(L * density)) / 2 >= flex_bits */
        const float density = alg == ALG_0
                ? 1.f * (1 << (exact_bits - 2 * flex_bits)) / L
                : 1.f;
        assert((exact_bits - ceilf(log2f(L * density))) / 2 >= flex_bits);

        BENCHDNN_PRINT(99, "check_alg: %s, density = %g, flex_bits = %ld\n",
                check_alg2str(alg), density, (long)flex_bits);

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
    } else {
        assert(alg == ALG_2);

        benchdnn_parallel_nd(prb->n, [&](int64_t n) {
            // Note: we use a different seed for each chunk to avoid
            // repeating patterns. We could use discard(idx_start) too but
            // it has a complexity in O(idx_start). We also add 1 to avoid
            // seeding with 0.
            std::minstd_rand int_seed(n + 1);
            int_seed.discard(1);
            std::minstd_rand b_seed(n + 1);
            b_seed.discard(2);

            const float val_coeff = is_integral_dt(prb->dt[0]) ? 4.f : 1.f;
            const int distr_shift = prb->dt[0] == dnnl_u8 ? 2 : 0;
            std::uniform_int_distribution<> int_dist(0 + distr_shift, 6);
            std::bernoulli_distribution b_dist(0.5f);
            const float m = val_coeff * 0.25f * (1 << int_dist(int_seed));
            float v = 0; /* current variance */

            const int64_t c_shift = n * prb->c;
            float *s = (float *)src + c_shift;

            bool bigger_val = false;
            float val = 0.f;

            for (int64_t c = 0; c < prb->c; ++c) {
                const int64_t idx = c_shift + c;

                if (c % 2 == 0) {
                    bigger_val = b_dist(b_seed);
                    val = bigger_val ? (m + val_coeff * 1.f)
                                     : (m + val_coeff * 0.25f);
                } else {
                    val = bigger_val ? (m - val_coeff * 1.f)
                                     : (m - val_coeff * 0.25f);
                }
                src.set_elem(idx, val);

                v += (s[c] - m) * (s[c] - m);
            }
            // Update last element with s[c] = m.
            if (prb->c % 2 == 1) {
                v -= (s[prb->c - 1] - m) * (s[prb->c - 1] - m);
                s[prb->c - 1] = m;
            }
            mean.set_elem(n, m);
            var.set_elem(n, v / prb->c);
        });
    }

    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    benchdnn_parallel_nd(prb->c, [&](int64_t c) {
        float sc_value = 1.f / 8 * (1 << (c % 7));
        float sh_value = (c % 3 + 1) * sc_value / 64;
        ((float *)sc)[c] = use_sc ? sc_value : 1.0f;
        ((float *)sh)[c] = use_sh ? sh_value : 0.0f;
    });

    return OK;
}

int prepare_fwd(const prb_t *prb, dnn_mem_map_t &mem_map,
        dnn_mem_map_t &ref_mem_map, res_t *res) {
    const auto &ref_src = ref_mem_map[DNNL_ARG_SRC];
    const auto &ref_mean = ref_mem_map[DNNL_ARG_MEAN];
    const auto &ref_var = ref_mem_map[DNNL_ARG_VARIANCE];
    const auto &ref_sc = ref_mem_map[DNNL_ARG_SCALE];
    const auto &ref_sh = ref_mem_map[DNNL_ARG_SHIFT];

    SAFE(prepare_fwd(prb, ref_src, ref_mean, ref_var, ref_sc, ref_sh, res),
            WARN);

    auto &src = mem_map[DNNL_ARG_SRC];
    SAFE(src.reorder(ref_src), WARN);

    auto &mean = mem_map[DNNL_ARG_MEAN];
    if (mean && prb->use_stats()) SAFE(mean.reorder(ref_mean), WARN);

    auto &var = mem_map[DNNL_ARG_VARIANCE];
    if (var && prb->use_stats()) SAFE(var.reorder(ref_var), WARN);

    auto &sc = mem_map[DNNL_ARG_SCALE];
    if (sc) SAFE(sc.reorder(ref_sc), WARN);

    auto &sh = mem_map[DNNL_ARG_SHIFT];
    if (sh) SAFE(sh.reorder(ref_sh), WARN);

    return OK;
}

static int prepare_bwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &d_dst, const dnn_mem_t &mean, const dnn_mem_t &var,
        const dnn_mem_t &sc, res_t *res) {
    const bool use_sc = prb->use_sc();

    // fill gamma
    for (int64_t c = 0; c < prb->c; ++c) {
        const float sc_value = 0.125f * (1 << (c % 7));
        ((float *)sc)[c] = use_sc ? sc_value : 1.0f;
    }

    benchdnn_parallel_nd(prb->n, [&](int64_t n) {
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(n + 1);
        int_seed.discard(1);
        std::minstd_rand b_seed(n + 1);
        b_seed.discard(2);

        // Idea behind the filling is to reduce a possibility of cancellation
        // when subtracting a part accumulated over N. For that, we simplify
        // src data to (m+1) and (m-1) points, d_dst data is more or less
        // random but we keep all values as pow2 values to have almost exact
        // summation result.
        std::uniform_int_distribution<> stat_dist(0, 2);
        std::uniform_int_distribution<> data_dist(0, 6);
        std::bernoulli_distribution half_dist(0.5f);

        // mean = {-0.5f, 0.f, 0.5f}
        const float m = 0.5f * (stat_dist(int_seed) - 1);
        mean.set_elem(n, m);

        // final variance = {0.25f, 1.f, 4.f}
        const float v = 0.25f * (1 << (stat_dist(int_seed) * 2));
        var.set_elem(n, v - prb->eps);

        const int64_t c_shift = n * prb->c;

        for (int64_t c = 0; c < prb->c; ++c) {
            int sign = half_dist(b_seed) ? 1.f : -1.f;
            // d_dst = powf(2, {-4, ... , 2})
            float dd = sign * 0.0625f * (1LL << data_dist(int_seed));
            d_dst.set_elem(c_shift + c,
                    round_to_nearest_representable(prb->dt[1], dd));

            float s = c % 2 == 0 ? (m - 1.f) : (m + 1.f);
            src.set_elem(
                    c_shift + c, round_to_nearest_representable(prb->dt[0], s));
        }
    });

    return OK;
}

int prepare_bwd(const prb_t *prb, dnn_mem_map_t &mem_map,
        dnn_mem_map_t &ref_mem_map, res_t *res) {
    const auto &ref_src = ref_mem_map[DNNL_ARG_SRC];
    const auto &ref_d_dst = ref_mem_map[DNNL_ARG_DIFF_DST];
    const auto &ref_mean = ref_mem_map[DNNL_ARG_MEAN];
    const auto &ref_var = ref_mem_map[DNNL_ARG_VARIANCE];
    const auto &ref_sc = ref_mem_map[DNNL_ARG_SCALE];
    const auto &ref_sh = ref_mem_map[DNNL_ARG_SHIFT];

    SAFE(prepare_bwd(prb, ref_src, ref_d_dst, ref_mean, ref_var, ref_sc, res),
            WARN);

    auto &src = mem_map[DNNL_ARG_SRC];
    SAFE(src.reorder(ref_src), WARN);

    auto &d_dst = mem_map[DNNL_ARG_DIFF_DST];
    SAFE(d_dst.reorder(ref_d_dst), WARN);

    auto &mean = mem_map[DNNL_ARG_MEAN];
    if (mean) SAFE(mean.reorder(ref_mean), WARN);

    auto &var = mem_map[DNNL_ARG_VARIANCE];
    if (var) SAFE(var.reorder(ref_var), WARN);

    auto &sc = mem_map[DNNL_ARG_SCALE];
    if (sc) SAFE(sc.reorder(ref_sc), WARN);

    auto &sh = mem_map[DNNL_ARG_SHIFT];
    if (sh) SAFE(sh.reorder(ref_sh), WARN);

    return OK;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;

    auto src_d = dnn_mem_t::init_md(
            prb->ndims, prb->dims.data(), prb->dt[0], prb->tag[0]);

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> stat_d {};
    if (prb->stat_tag != tag::undef) {
        stat_d = dnn_mem_t::init_md(
                prb->ndims - 1, prb->dims.data(), dnnl_f32, prb->stat_tag);
    }

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    auto flags = (dnnl_normalization_flags_t)prb->flags;
    if (prb->dir & FLAG_FWD) {
        auto dst_d = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->dt[1], prb->tag[1]);
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        TIME_C_PD(DNN_SAFE_STATUS(
                dnnl_layer_normalization_forward_primitive_desc_create(
                        &init_pd_args.pd, init_pd_args.engine, prop,
                        init_pd_args.src_md ? init_pd_args.src_md : src_d,
                        dst_d, stat_d, prb->eps, flags, dnnl_attr)));
    } else {
        auto diff_src_d = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->dt[0], prb->tag[0]);
        auto diff_dst_d = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->dt[1], prb->tag[1]);
        auto prop = prb->dir & FLAG_WEI ? dnnl_backward : dnnl_backward_data;
        TIME_C_PD(DNN_SAFE_STATUS(
                dnnl_layer_normalization_backward_primitive_desc_create(
                        &init_pd_args.pd, init_pd_args.engine, prop, diff_src_d,
                        diff_dst_d, src_d, stat_d, prb->eps, flags,
                        init_pd_args.hint, dnnl_attr)));
    }

    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->dt[0], prb->dt[1]}, prb->dir, res);
    skip_unimplemented_sum_po(
            prb->attr, res, dnnl_layer_normalization, prb->dt[0]);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_layer_normalization);

    if (is_gpu()) {
        const bool dt_ok = prb->dt[0] == prb->dt[1]
                && !is_integral_dt(prb->dt[0]) && !is_integral_dt(prb->dt[1]);
        if (!dt_ok) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        skip_invalid_inplace(
                res, prb->dt[0], prb->dt[1], prb->tag[0], prb->tag[1]);
        if (res->state == SKIPPED) return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const bool compare_with_norm = (prb->dir & FLAG_BWD);
    cmp.set_norm_validation_mode(compare_with_norm);

    const auto dt = prb->dir & FLAG_FWD ? prb->dt[1] : prb->dt[0];
    // Digits must be non-negative for safe left-shifting when `digits_dt`
    // exceeds `digits_f32`.
    const int safe_digits = MAX2(0, digits_dt(dnnl_f32) - digits_dt(dt));
    const float trh_coeff = (1 << safe_digits);
    float trh = trh_coeff * ((kind == SRC || kind == DST) ? 5e-7 : 0);
    if ((kind == SC || kind == SH) && prb->dir & FLAG_BWD)
        trh = trh_coeff * 5e-6;
    cmp.set_threshold(trh);

    // u8 turns half of output into zeros.
    if (prb->dt[1] == dnnl_u8) cmp.set_zero_trust_percent(60.f);

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
                if (!((prb->dir & FLAG_FWD) && kind == DST && prb->use_sh()))
                    return false;

                const auto &sh = ref_args.find(DNNL_ARG_SHIFT);
                const auto &dst = ref_args.find(DNNL_ARG_DST);
                const int64_t c = dst.get_scale_idx(
                        args.idx, 1 << (prb->ndims - 1) /* last_dim_mask */);
                const float beta = sh.get_elem(c);
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

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_MEAN,
            DNNL_ARG_VARIANCE,
            DNNL_ARG_SCALE,
            DNNL_ARG_SHIFT,
            DNNL_ARG_DST,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_MEAN,
            DNNL_ARG_VARIANCE,
            DNNL_ARG_SCALE,
            DNNL_ARG_SHIFT,
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DIFF_SCALE,
            DNNL_ARG_DIFF_SHIFT,
            DNNL_ARG_DIFF_SRC,
    };
    return (dir & FLAG_FWD) ? exec_fwd_args : exec_bwd_args;
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref) {
    update_inplace_memory_args(mem_map, prb, dir);
    if (has_bench_mode_modifier(mode_modifier_t::no_host_memory)) return OK;

    // TODO: this function still allocates the full memory print needed to fill
    // the data and each argument can't be destroyed right away since filling
    // requires all of them at a time.
    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second;

        ref_mem_map.emplace(
                exec_arg, dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_MEAN:
            case DNNL_ARG_VARIANCE:
                if (prb->dir & FLAG_INF) {
                    const auto &src_md = mem_map[DNNL_ARG_SRC].md_;
                    const auto stat_dims = query_md_dims(src_md);
                    ref_mem_map[exec_arg] = dnn_mem_t(prb->ndims - 1, stat_dims,
                            dnnl_f32, tag::abx, ref_engine);
                }
                break;
            default: {
                bool is_scales_arg = (exec_arg & DNNL_ARG_ATTR_SCALES);
                if (is_scales_arg) {
                    int exec_src_arg = exec_arg ^ DNNL_ARG_ATTR_SCALES;
                    SAFE(fill_scales(prb->attr, exec_src_arg, mem, ref_mem),
                            WARN);
                }
            } break;
        }
    }

    if (dir & FLAG_FWD) {
        SAFE(prepare_fwd(prb, mem_map, ref_mem_map, res), WARN);
    } else {
        SAFE(prepare_bwd(prb, mem_map, ref_mem_map, res), WARN);
    }

    // Don't keep reference memory if it is not used further.
    if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds;
    if (prb->dir & FLAG_FWD) {
        check_kinds = {DST};
        if (!(prb->flags & GLOB_STATS) && !(prb->dir & FLAG_INF)) {
            check_kinds.push_back(MEAN);
            check_kinds.push_back(VAR);
        }
    } else {
        check_kinds = {SRC};
        if (prb->use_sc() && (prb->dir & FLAG_WEI)) check_kinds.push_back(SC);
        if (prb->use_sh() && (prb->dir & FLAG_WEI)) check_kinds.push_back(SH);
    }
    assert(!check_kinds.empty());
    return check_kinds;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(1);
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    return OK;
}

int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    return check_caches(v_prim[0], prb, res);
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = v_prim[0];

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(mem_map, prb, prim, supported_exec_args(prb->dir));
    TIME_FILL(SAFE(init_ref_memory_args(
                           ref_mem_map, mem_map, prim, prb, res, prb->dir),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (has_bench_mode_bit(mode_bit_t::corr)) {
        check_correctness(
                prb, get_kinds_to_check(prb), args, ref_args, setup_cmp, res);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace lnorm
