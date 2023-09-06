/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#include <random>
#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "bnorm/bnorm.hpp"

namespace bnorm {

static int prepare_fwd_with_stats(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &src_add, const dnn_mem_t &mean, const dnn_mem_t &var,
        const dnn_mem_t &sc, const dnn_mem_t &sh, res_t *res) {
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();
    const bool fill_src_add = prb->fuse_add_relu();

    benchdnn_parallel_nd(prb->ic, [&](int64_t c) {
        mean.set_elem(c, 4 * ((c % 5) - 2));
        var.set_elem(c, ((c % 7) << 1));

        const float sc_value = 1 << (c % 7);
        const float sh_value = ((c % 3) - 1) * sc_value;
        if (use_sc) sc.set_elem(c, sc_value);
        if (use_sh) sh.set_elem(c, sh_value);
    });

    benchdnn_parallel_nd(prb->ic, prb->mb, prb->id, prb->ih, prb->iw,
            [&](int64_t c, int64_t mb, int64_t d, int64_t h, int64_t w) {
                int64_t l_base = mb * prb->id * prb->ih * prb->iw + c * 239 * 2;
                float *s = (float *)src + data_off(prb, mb, c, 0, 0, 0);

                const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                const int64_t l = l_base + sp;
                const int64_t value = (l % 65) - 32;
                s[sp] = round_to_nearest_representable(prb->dt, value);
                if (fill_src_add) {
                    float *s_add
                            = (float *)src_add + data_off(prb, mb, c, 0, 0, 0);
                    s_add[sp] = round_to_nearest_representable(
                            prb->dt, (l % 17) - 8);
                }
            });

    return OK;
}

static int prepare_fwd_no_stats(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &src_add, const dnn_mem_t &mean, const dnn_mem_t &var,
        const dnn_mem_t &sc, const dnn_mem_t &sh, res_t *res) {
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
    const int64_t L = prb->mb * prb->id * prb->ih * prb->iw;
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

    if (flex_bits < min_flex_bits) {
        res->state = UNTESTED;
        return FAIL;
    }

    const int64_t flex_mask = (static_cast<int64_t>(1) << flex_bits) - 1;

    /* density: (exact_bits - log_2(L * density)) / 2 >= flex_bits */
    const float density = alg == ALG_0
            ? 1.f * (1 << (exact_bits - 2 * flex_bits)) / L
            : 1.f;
    assert((exact_bits - ceilf(log2f(L * density))) / 2 >= flex_bits);

    BENCHDNN_PRINT(6, "check_alg: %s, density = %g, flex_bits = " IFMT "\n",
            check_alg2str(alg), density, flex_bits);

    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();
    const bool fill_src_add = prb->fuse_add_relu();

    benchdnn_parallel_nd(prb->ic, [&](int64_t c) {
        const float m = ((float *)mean)[c]
                = alg == ALG_0 ? 0.f : 0.25f * (1 << (c % 7));
        float v = 0; /* current variance */

        for (int64_t mb = 0; mb < prb->mb; ++mb) {
            int64_t l_base = mb * prb->id * prb->ih * prb->iw
                    + c * 239 * 2; // l[0] must be even
            int64_t off = data_off(prb, mb, c, 0, 0, 0);
            float *s = (float *)src + off;

            for_(int64_t d = 0; d < prb->id; ++d)
            for_(int64_t h = 0; h < prb->ih; ++h)
            for (int64_t w = 0; w < prb->iw; ++w) {

                const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                const int64_t l = l_base + sp;

                if (alg == ALG_0 && !flip_coin(l / 2 * 257ULL, density)) {
                    s[sp] = 0;
                    if (fill_src_add) src_add.set_elem(off + sp, 1.f);
                    continue;
                }

                const int64_t gen = (l / 2 * 1637) & flex_mask;
                const int sgn = l % 2 == 0 ? 1 : -1; /* [a1] */
                const float f = 1.f * sgn * gen / (1 << flex_bits);

                s[sp] = alg == ALG_0 ? f : m * (1.f + f);
                if (L % 2 && (mb * prb->id * prb->ih * prb->iw + sp == L - 1)) {
                    s[sp] = m;
                }
                v += (s[sp] - m) * (s[sp] - m);
                if (fill_src_add) {
                    // The main purpose of such filling is to avoid catastrophic
                    // cancellation. To do that, the sign of `Add` tensor final
                    // values is kept the same as it would be after applying
                    // bnorm: what's below mean, that has negative sign, what's
                    // equal or higher - positive.
                    const int64_t mod2_base = (mb + c + d + h + w) % 5;
                    const float mod2_val = 1.f / (2LL << mod2_base);
                    const int64_t sign_val = s[sp] < m ? -1 : 1;
                    float *s_add = (float *)src_add + off;
                    s_add[sp] = round_to_nearest_representable(
                            prb->dt, mod2_val * sign_val);
                }
            }
        }

        ((float *)var)[c] = v / (prb->mb * prb->id * prb->ih * prb->iw);

        const float sc_value = 1.f / 8 * (1 << (c % 7));
        const float sh_value = ((c % 3) - 1) * sc_value / 64;
        if (use_sc) sc.set_elem(c, sc_value);
        if (use_sh) sh.set_elem(c, sh_value);
    });

    return OK;
}

int prepare_fwd(const prb_t *prb, dnn_mem_map_t &mem_map,
        dnn_mem_map_t &ref_mem_map, res_t *res) {
    const auto &ref_src = ref_mem_map[DNNL_ARG_SRC];
    const auto &ref_src_add = ref_mem_map[DNNL_ARG_SRC_1];
    const auto &ref_mean = ref_mem_map[DNNL_ARG_MEAN];
    const auto &ref_var = ref_mem_map[DNNL_ARG_VARIANCE];
    const auto &ref_sc = ref_mem_map[DNNL_ARG_SCALE];
    const auto &ref_sh = ref_mem_map[DNNL_ARG_SHIFT];

    if (prb->flags & GLOB_STATS)
        SAFE(prepare_fwd_with_stats(prb, ref_src, ref_src_add, ref_mean,
                     ref_var, ref_sc, ref_sh, res),
                WARN);
    else
        SAFE(prepare_fwd_no_stats(prb, ref_src, ref_src_add, ref_mean, ref_var,
                     ref_sc, ref_sh, res),
                WARN);

    auto &src = mem_map[DNNL_ARG_SRC];
    SAFE(src.reorder(ref_src), WARN);

    auto &src_add = mem_map[DNNL_ARG_SRC_1];
    if (src_add) SAFE(src_add.reorder(ref_src_add), WARN);

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

int prepare_bwd(
        const prb_t *prb, dnn_mem_map_t &mem_map, dnn_mem_map_t &ref_mem_map) {
    const auto &mem_fp = ref_mem_map[DNNL_ARG_DIFF_DST];
    auto &mem_dt = mem_map[DNNL_ARG_DIFF_DST];

    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Idea behind filling: integer diff_dst values decrease norms unlike fp32
    // values in [-1.f, 1.f] range. To decrease norms more, make data pretty
    // sparse as answers sum all diff_dst values.

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);

        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand msr(idx_start + 1);
        msr.discard(1);

        std::uniform_int_distribution<> igen_val(-2, 2);
        std::uniform_int_distribution<> igen_coin(0, 256 * 1024);

        // at least 20 non-zero elems
        float sparsity = MAX2(0.05f, MIN2(1.f, 20.f / nelems));

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value = flip_coin(igen_coin(msr), sparsity)
                    ? round_to_nearest_representable(prb->dt, igen_val(msr))
                    : 0;
            mem_fp.set_elem(idx, value);
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int check_fwd_ws(dnn_mem_map_t &mem_map, res_t *res) {
    const auto &dst_dt = mem_map[DNNL_ARG_DST];
    const auto &ws_dt = mem_map[DNNL_ARG_WORKSPACE];

    if (ws_dt.ndims() == 0) return OK;

    /* so far we know ws is just bit-mask of whether value was negative or
     * positive */
    const auto nelems = dst_dt.nelems(true);
    const uint8_t *ws = (const uint8_t *)ws_dt;

    /* some internal knowledge: flags in ws are either stored as bytes (e.g.
     * for the ref implementation) or as bits (e.g. for the jitted one); in
     * the latter case the ws memory has fewer elements than the data memory */
    enum { ws_byte, ws_bit } ws_type;
    ws_type = ws_dt.nelems(true) < nelems ? ws_bit : ws_byte;

    /* more internal knowledge: dst_dt and ws_dt are expected to have exactly
     * the same data layout, and dst_dt padded regions are expected to be
     * zero, and the respective ws_dt elements should be set accordingly */
    for (int64_t i = 0; i < nelems; i += 8) {
        for (int64_t j = 0; j < MIN2(8, nelems - i); ++j) {
            const float data = dst_dt.get_elem(i + j);
            const bool want = data > 0.f;
            const bool bit_set = ws_type == ws_byte ? *ws : !!(*ws & (1 << j));

            const bool ok = bit_set == want;
            res->errors += !ok;

            bool dump = false || (!ok && (res->errors < 10 || verbose >= 10))
                    || (verbose >= 50 && i < 30);
            if (dump) {
                BENCHDNN_PRINT(0, "[%4ld] ws exp:%d got:%d (data:%g:%a)\n",
                        (long)(i + j), want, bit_set, data, data);
            }

            if (ws_type == ws_byte) ++ws;
        }
        if (ws_type == ws_bit) ++ws;
    }

    if (res->errors) res->state = FAILED;
    if (res->state == EXECUTED) res->state = PASSED;

    return res->state == FAILED ? FAIL : OK;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    const dir_t dir = init_pd_args.dir;
    res_t *res = init_pd_args.res;

    auto src_d = dnn_mem_t::init_md(
            prb->ndims, prb->data_dims().data(), prb->dt, prb->tag);

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    auto flags = (dnnl_normalization_flags_t)prb->flags;
    if (dir & FLAG_FWD) {
        auto dst_d = dnn_mem_t::init_md(
                prb->ndims, prb->data_dims().data(), prb->dt, tag::any);
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        TIME_C_PD(DNN_SAFE_STATUS(
                dnnl_batch_normalization_forward_primitive_desc_create(
                        &init_pd_args.pd, init_pd_args.engine, prop,
                        init_pd_args.src_md ? init_pd_args.src_md : src_d,
                        dst_d, prb->eps, flags, dnnl_attr)));
    } else {
        auto diff_src_d = dnn_mem_t::init_md(
                prb->ndims, prb->data_dims().data(), prb->dt, tag::any);
        auto diff_dst_d = dnn_mem_t::init_md(
                prb->ndims, prb->data_dims().data(), prb->dt, tag::any);
        auto prop = prb->dir & FLAG_WEI ? dnnl_backward : dnnl_backward_data;
        TIME_C_PD(DNN_SAFE_STATUS(
                dnnl_batch_normalization_backward_primitive_desc_create(
                        &init_pd_args.pd, init_pd_args.engine, prop, diff_src_d,
                        diff_dst_d, src_d, prb->eps, flags, init_pd_args.hint,
                        dnnl_attr)));
    }

    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->dt}, prb->dir, res);
    skip_unimplemented_sum_po(
            prb->attr, res, dnnl_batch_normalization, prb->dt);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_batch_normalization);

    // Non-zero alpha is not supported for training in general.
    const auto &po = prb->attr.post_ops;
    const auto relu_idx = po.find(attr_t::post_ops_t::kind_t::RELU);
    if (relu_idx >= 0) {
        const auto &e = po.entry[relu_idx];
        float alpha = e.eltwise.alpha;
        bool alpha_ok = IMPLICATION(alpha != 0.f, (prb->dir & FLAG_INF));
        if (!alpha_ok) {
            res->state = SKIPPED;
            res->reason = CASE_NOT_SUPPORTED;
        }
    }
    // BN+Add+ReLU fusion is not supported on CPU
    if (is_cpu() && prb->fuse_add_relu()) {
        res->state = SKIPPED;
        res->reason = CASE_NOT_SUPPORTED;
    }
    // int8 only supports forward s8 w/ global stats
    const bool u8_not_ok = prb->dt == dnnl_u8;
    const bool s8_not_ok = prb->dt == dnnl_s8
            && ((prb->dir & FLAG_BWD) || (prb->flags & GLOB_STATS) == 0);
    if (s8_not_ok || u8_not_ok) {
        res->state = SKIPPED;
        res->reason = CASE_NOT_SUPPORTED;
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // MIOpen cannot handle inplace cases correctly.
    if (is_amd_gpu() && prb->inplace) {
        res->state = SKIPPED;
        return;
    }

    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        skip_invalid_inplace(res, prb->dt, prb->dt, prb->tag, prb->tag);
        if (res->state == SKIPPED) return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    // Since bwd testing is done using results from forward which are random
    // fp32 values, diff_scale starts fluctuating, so we check norm for both
    // data, SC, and SH.
    const bool compare_with_norm = (prb->dir & FLAG_BWD);
    cmp.set_norm_validation_mode(compare_with_norm);

    // Digits must be non-negative for safe left-shifting when `digits_dt`
    // exceeds `digits_f32`.
    const int safe_digits = MAX2(0, digits_dt(dnnl_f32) - digits_dt(prb->dt));
    const float trh_coeff = (1 << safe_digits);
    float trh = trh_coeff
            * ((kind == SRC || kind == DST || kind == SRC_1) ? 6e-7f : 0);
    if ((kind == SC || kind == SH) && prb->dir & FLAG_BWD)
        trh = trh_coeff * 5e-6f;

#ifdef DNNL_EXPERIMENTAL
    const bool bnorm_single_pass
            = dnnl::impl::experimental::use_bnorm_stats_one_pass();
#else
    const bool bnorm_single_pass = false;
#endif

    const bool use_relaxed_validation
            = is_nvidia_gpu() || is_amd_gpu() || bnorm_single_pass;
    if (use_relaxed_validation) {
        // Nvidia (cuDNN) and AMD (MIOpen): store unbiased variance which
        // requires rescaling by `(N - 1) / N`, where `N = MB * Spatial`.
        // Hence, we cannot set the threshold to 0...
        // Also mean could be computed using a single pass formula.
        //
        // On Intel GPUs mean and variance could be rounded incorrectly because
        // they are calculated using fast but potentially unstable formula.
        if (kind == MEAN) trh = 1e-7f;
        if (kind == VAR) trh = 5e-7f;
    }
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
    const auto bnorm_add_check =
            [&, kind, prb](
                    const compare::compare_t::driver_check_func_args_t &args) {
                if (!((prb->dir & FLAG_FWD) && kind == DST && prb->use_sh()))
                    return false;

                const auto &sh = ref_args.find(DNNL_ARG_SHIFT);
                const auto &dst = ref_args.find(DNNL_ARG_DST);
                const int64_t c = dst.get_scale_idx(
                        args.idx, 1 << 1 /* channel_mask */);
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
    cmp.set_driver_check_function(bnorm_add_check);
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_SRC_1,
            DNNL_ARG_MEAN,
            DNNL_ARG_VARIANCE,
            DNNL_ARG_SCALE,
            DNNL_ARG_SHIFT,
            DNNL_ARG_DST,
            DNNL_ARG_WORKSPACE,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_SRC_1,
            DNNL_ARG_MEAN,
            DNNL_ARG_VARIANCE,
            DNNL_ARG_SCALE,
            DNNL_ARG_WORKSPACE,
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DIFF_SCALE,
            DNNL_ARG_DIFF_SHIFT,
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_DIFF_SRC_1,
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

        switch (exec_arg) {
            case DNNL_ARG_DST:
                if (prb->dir & FLAG_BWD) {
                    // Stash for backward which is used in reference code:
                    //     src_hat[i] = (src[i] - mean) / sqrt(var + prb->eps)
                    ref_mem_map.emplace(DNNL_ARG_DST_1,
                            dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
                }
                break;
            case DNNL_ARG_DIFF_SRC: break; // Skip on backward.
            case DNNL_ARG_DST_1: break; // Skip on backward.
            case DNNL_ARG_MEAN:
            case DNNL_ARG_VARIANCE:
                if (prb->dir & FLAG_INF) {
                    const dnnl_dims_t dims1d = {prb->ic};
                    ref_mem_map[exec_arg] = dnn_mem_t(
                            1, dims1d, dnnl_f32, tag::abx, ref_engine);
                }
                break;
            default: break;
        }
    }

    if (dir & FLAG_FWD) {
        SAFE(prepare_fwd(prb, mem_map, ref_mem_map, res), WARN);
    } else {
        SAFE(prepare_bwd(prb, mem_map, ref_mem_map), WARN);
    }

    // Don't keep reference memory if it is not used further.
    if (!has_bench_mode_bit(mode_bit_t::corr)) {
        ref_mem_map.clear();
        return OK;
    }

    // Reference code uses different kind of workspace. Adjust to ref needs.
    if (ref_mem_map.count(DNNL_ARG_WORKSPACE)) {
        const auto &src_md = ref_mem_map[DNNL_ARG_SRC].md_;
        ref_mem_map[DNNL_ARG_WORKSPACE]
                = dnn_mem_t(src_md, dnnl_u8, tag::abx, ref_engine);
    }

    return OK;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(2); // just fwd or fwd + bwd.
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res, FLAG_FWD,
                 nullptr, /* is_service_prim = */ prb->dir & FLAG_BWD),
            WARN);
    if (prb->dir & FLAG_BWD) {
        SAFE(init_prim(prb->ctx_init, v_prim[1], init_pd, prb, res, FLAG_BWD,
                     query_pd(v_prim[0])),
                WARN);
    }
    return OK;
}

int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    SAFE(check_caches(v_prim[0], prb, res), WARN);
    if (v_prim[1]) { SAFE(check_caches(v_prim[1], prb, res), WARN); }
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = prb->dir & FLAG_FWD ? v_prim[0] : v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(
            mem_map, prb, v_prim[0], supported_exec_args(FLAG_FWD));
    TIME_FILL(SAFE(init_ref_memory_args(
                           ref_mem_map, mem_map, v_prim[0], prb, res, FLAG_FWD),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(execute_and_wait(v_prim[0], args, res), WARN);

    // Running ref to collect src_hat (used instead of src + mean) and ws, if
    // fuse_relu flag is requested.
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        if (prb->dir & FLAG_FWD) {
            std::vector<data_kind_t> kinds {DST};
            if (!(prb->flags & GLOB_STATS) && !(prb->dir & FLAG_INF)) {
                kinds.push_back(MEAN);
                kinds.push_back(VAR);
            }

            // TODO: make kinds empty for BWD?
            check_correctness(prb, kinds, args, ref_args, setup_cmp, res);

            if (prb->debug_check_ws) check_fwd_ws(mem_map, res);
        }
    }

    if (prb->dir & FLAG_BWD) {
        // Pass same memory map as we need data from forward on backward.
        init_memory_args<prb_t>(
                mem_map, prb, v_prim[1], supported_exec_args(FLAG_BWD));
        TIME_FILL(SAFE(init_ref_memory_args(ref_mem_map, mem_map, v_prim[1],
                               prb, res, FLAG_BWD),
                WARN));

        args = args_t(mem_map);
        ref_args = args_t(ref_mem_map);

        SAFE(execute_and_wait(v_prim[1], args, res), WARN);

        if (has_bench_mode_bit(mode_bit_t::corr)) {
            std::vector<data_kind_t> kinds {SRC};
            if (prb->use_sc() && (prb->dir & FLAG_WEI)) kinds.push_back(SC);
            if (prb->use_sh() && (prb->dir & FLAG_WEI)) kinds.push_back(SH);

            check_correctness(prb, kinds, args, ref_args, setup_cmp, res);
        }
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace bnorm
