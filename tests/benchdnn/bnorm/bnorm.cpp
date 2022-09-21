/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

static int prepare_fwd_with_stats(const prb_t *prb, dnn_mem_t &src,
        dnn_mem_t &src_add, dnn_mem_t &mean, dnn_mem_t &var, dnn_mem_t &sc,
        dnn_mem_t &sh) {
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();
    const bool fill_src_add = prb->fuse_add_relu();

    benchdnn_parallel_nd(prb->ic, [&](int64_t c) {
        mean.set_elem(c, 4 * ((c % 5) - 2));
        var.set_elem(c, ((c % 7) << 1));

        const float sc_value = 1 << (c % 7);
        const float sh_value = ((c % 3) - 1) * sc_value;
        sc.set_elem(c, use_sc ? sc_value : 1.0f);
        sh.set_elem(c, use_sh ? sh_value : 0.0f);
    });

    benchdnn_parallel_nd(prb->ic, prb->mb, prb->id, prb->ih, prb->iw,
            [&](int64_t c, int64_t mb, int64_t d, int64_t h, int64_t w) {
                int64_t l_base = mb * prb->id * prb->ih * prb->iw + c * 239 * 2;
                float *s = (float *)src + data_off(prb, mb, c, 0, 0, 0);
                float *s_add = (float *)src_add + data_off(prb, mb, c, 0, 0, 0);

                const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                const int64_t l = l_base + sp;
                const int64_t value = (l % 65) - 32;
                s[sp] = round_to_nearest_representable(prb->dt, value);
                if (fill_src_add)
                    s_add[sp] = round_to_nearest_representable(
                            prb->dt, (l % 17) - 8);
            });

    return OK;
}

static int prepare_fwd_no_stats(const prb_t *prb, dnn_mem_t &src,
        dnn_mem_t &src_add, dnn_mem_t &mean, dnn_mem_t &var, dnn_mem_t &sc,
        dnn_mem_t &sh) {
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

    if (flex_bits < min_flex_bits) return FAIL;

    const int64_t flex_mask = (1 << flex_bits) - 1;

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
            float *s = (float *)src + data_off(prb, mb, c, 0, 0, 0);
            float *s_add = (float *)src_add + data_off(prb, mb, c, 0, 0, 0);

            for_(int64_t d = 0; d < prb->id; ++d)
            for_(int64_t h = 0; h < prb->ih; ++h)
            for (int64_t w = 0; w < prb->iw; ++w) {

                const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                const int64_t l = l_base + sp;

                if (alg == ALG_0 && !flip_coin(l / 2 * 257ULL, density)) {
                    s[sp] = 0;
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
                    s_add[sp] = round_to_nearest_representable(
                            prb->dt, mod2_val * sign_val);
                }
            }
        }

        ((float *)var)[c] = v / (prb->mb * prb->id * prb->ih * prb->iw);

        const float sc_value = 1.f / 8 * (1 << (c % 7));
        const float sh_value = ((c % 3) - 1) * sc_value / 64;
        ((float *)sc)[c] = use_sc ? sc_value : 1.0f;
        ((float *)sh)[c] = use_sh ? sh_value : 0.0f;
    });

    return OK;
}

static int prepare_fwd(const prb_t *prb, dnn_mem_t &src, dnn_mem_t &src_add,
        dnn_mem_t &mean, dnn_mem_t &var, dnn_mem_t &sc, dnn_mem_t &sh) {
    if (prb->flags & GLOB_STATS)
        return prepare_fwd_with_stats(prb, src, src_add, mean, var, sc, sh);
    else
        return prepare_fwd_no_stats(prb, src, src_add, mean, var, sc, sh);
}

static int prepare_bwd(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
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

int check_fwd_ws(const dnn_mem_t &dst_dt, const dnn_mem_t &ws_dt, res_t *res) {
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

    auto data_d = dnn_mem_t::init_md(
            prb->ndims, prb->data_dims().data(), prb->dt, prb->tag);

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    auto flags = (dnnl_normalization_flags_t)prb->flags;
    if (dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        DNN_SAFE_STATUS(dnnl_batch_normalization_forward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, prop, data_d, prb->eps,
                flags, dnnl_attr));
    } else {
        auto diff_data_d = dnn_mem_t::init_md(
                prb->ndims, prb->data_dims().data(), prb->dt, tag::any);
        auto prop = prb->dir & FLAG_WEI ? dnnl_backward : dnnl_backward_data;
        DNN_SAFE_STATUS(dnnl_batch_normalization_backward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, prop, diff_data_d,
                data_d, prb->eps, flags, init_pd_args.hint, dnnl_attr));
    }

    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->dt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);

    // Non-zero alpha is not supported on GPU and for training in general.
    const auto &po = prb->attr.post_ops;
    const auto relu_idx = po.find(attr_t::post_ops_t::kind_t::RELU);
    if (relu_idx >= 0) {
        const auto &e = po.entry[relu_idx];
        float alpha = e.eltwise.alpha;
        bool alpha_ok
                = IMPLICATION(alpha != 0.f, (prb->dir & FLAG_INF) && is_cpu());
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
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
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

    const int f32_mant_digits = 24;
    const float trh_coeff = (1 << (f32_mant_digits - digits_dt(prb->dt)));
    float trh = trh_coeff
            * ((kind == SRC || kind == DST || kind == SRC_1) ? 5e-7 : 0);
    if ((kind == SC || kind == SH) && prb->dir & FLAG_BWD)
        trh = trh_coeff * 5e-6;

#ifdef DNNL_EXPERIMENTAL
    const bool bnorm_single_pass
            = dnnl::impl::experimental::use_bnorm_stats_one_pass();
#else
    const bool bnorm_single_pass = false;
#endif

    const bool use_relaxed_validation = is_nvidia_gpu() || bnorm_single_pass;
    if (use_relaxed_validation) {
        // Nvidia: cuDNN stores unbiased variance which requires rescaling by
        // `(N - 1) / N`, where `N = MB * Spatial`. Hence, we cannot set the
        // threshold to 0...
        // Also mean could be computed using a single pass formula.
        //
        // On Intel GPUs mean and variance could be rounded incorrectly because
        // they are calculated using fast but potentially unstable formula.
        if (kind == MEAN) trh = 1e-7;
        if (kind == VAR) trh = 4e-7;
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

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    bool is_service_prim = prb->dir & FLAG_BWD;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res, FLAG_FWD, nullptr,
                 is_service_prim),
            WARN);

    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_fpd = query_pd(prim);

    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();
    const bool fuse_add_relu = prb->fuse_add_relu();

    const auto &data_md = query_md(const_fpd, DNNL_ARG_SRC);
    const auto &mean_md = query_md(const_fpd, DNNL_ARG_MEAN);
    const auto &var_md = query_md(const_fpd, DNNL_ARG_VARIANCE);
    const auto &sc_md = query_md(const_fpd, DNNL_ARG_SCALE);
    const auto &sh_md = query_md(const_fpd, DNNL_ARG_SHIFT);
    const auto &ws_md = query_md(const_fpd, DNNL_ARG_WORKSPACE);
    const auto &scratchpad_md = query_md(const_fpd, DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    dnn_mem_t src_fp(data_md, fp, tag, ref_engine);
    dnn_mem_t src_dt(data_md, test_engine);
    dnn_mem_t src_add_fp(data_md, fp, tag, ref_engine);
    dnn_mem_t src_add_dt(data_md, test_engine);
    // stash for bwd: src_hat[i] = (src[i] - mean) / sqrt(var + prb->eps)
    dnn_mem_t src_hat_fp(data_md, fp, tag, ref_engine);

    dnn_mem_t &dst_fp = src_fp; // in-place in ref code
    dnn_mem_t placeholder_dst_dt;
    const bool inplace_fwd = prb->inplace && (prb->dir & FLAG_FWD);
    if (!inplace_fwd) { placeholder_dst_dt = dnn_mem_t(data_md, test_engine); }
    dnn_mem_t &dst_dt = inplace_fwd ? src_dt : placeholder_dst_dt;

    // On inference w/o global stats the batch norm doesn't require stat
    // memories. Hence, we need to prepare the mean_fp and var_fp ourselves.
    const dnnl_dims_t dims1d = {prb->ic};
    dnn_mem_t mean_fp(1, dims1d, fp, tag::abx, ref_engine);
    dnn_mem_t mean_dt(mean_md, test_engine);
    dnn_mem_t var_fp(1, dims1d, fp, tag::abx, ref_engine);
    dnn_mem_t var_dt(var_md, test_engine);

    dnn_mem_t sc_fp(1, dims1d, fp, tag::abx, ref_engine);
    dnn_mem_t sc_dt(sc_md, test_engine);
    dnn_mem_t d_sc_fp(1, dims1d, fp, tag::abx, ref_engine);
    dnn_mem_t d_sc_dt(sc_md, test_engine);

    dnn_mem_t sh_fp(1, dims1d, fp, tag::abx, ref_engine);
    dnn_mem_t sh_dt(sh_md, test_engine);
    dnn_mem_t d_sh_fp(1, dims1d, fp, tag::abx, ref_engine);
    dnn_mem_t d_sh_dt(sh_md, test_engine);

    dnn_mem_t ws_fp(data_md, dnnl_u8, tag, ref_engine);
    dnn_mem_t ws_dt(ws_md, test_engine);
    if (prb->need_ws()) SAFE(ws_dt.ndims() != 0 ? OK : FAIL, WARN);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    dnn_mem_t d_dst_dt, placeholder_d_src_dt;

    if (prepare_fwd(prb, src_fp, src_add_fp, mean_fp, var_fp, sc_fp, sh_fp)
            != OK) {
        return res->state = MISTRUSTED, OK;
    }

    SAFE(src_dt.reorder(src_fp), WARN);
    SAFE(src_add_dt.reorder(src_add_fp), WARN);
    if (prb->flags & GLOB_STATS) {
        SAFE(mean_dt.reorder(mean_fp), WARN);
        SAFE(var_dt.reorder(var_fp), WARN);
    }
    if (use_sc) { SAFE(sc_dt.reorder(sc_fp), WARN); }
    if (use_sh) { SAFE(sh_dt.reorder(sh_fp), WARN); }

    args_t args, ref_args;

    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_SRC_1, src_add_dt);
    args.set(DNNL_ARG_MEAN, mean_dt);
    args.set(DNNL_ARG_VARIANCE, var_dt);
    args.set(DNNL_ARG_SCALE, sc_dt);
    args.set(DNNL_ARG_SHIFT, sh_dt);
    args.set(DNNL_ARG_WORKSPACE, ws_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(DNNL_ARG_DST, dst_dt);

    SAFE(execute_and_wait(prim, args, res), WARN);

    // Running ref to collect src_hat (used instead of src + mean) and ws, if
    // fuse_relu flag is requested.
    if (is_bench_mode(CORR)) {
        if (prb->dir & FLAG_FWD) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_SRC_1, src_add_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(DNNL_ARG_SCALE, sc_fp);
            ref_args.set(DNNL_ARG_SHIFT, sh_fp);
            ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_DST_1, src_hat_fp); // Reference aux arg.

            std::vector<data_kind_t> kinds {DST};
            if (!(prb->flags & GLOB_STATS) && !(prb->dir & FLAG_INF)) {
                kinds.push_back(MEAN);
                kinds.push_back(VAR);
            }

            check_correctness(prb, kinds, args, ref_args, setup_cmp, res);

            if (prb->debug_check_ws) check_fwd_ws(dst_dt, ws_dt, res);
        }
    }

    if (prb->dir & FLAG_BWD) {
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> tmp_prim;
        SAFE(init_prim(tmp_prim, init_pd, prb, res, FLAG_BWD, const_fpd), WARN);
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
        prim.reset(tmp_prim.release());

        auto const_bpd = query_pd(prim);

        const auto &d_data_md = query_md(const_bpd, DNNL_ARG_DIFF_DST);
        const auto &d_scratchpad_md = query_md(const_bpd, DNNL_ARG_SCRATCHPAD);

        dnn_mem_t d_dst_fp(d_data_md, fp, tag, ref_engine);
        d_dst_dt = dnn_mem_t(d_data_md, test_engine);

        dnn_mem_t &d_src_fp = d_dst_fp; // in-place in ref code
        dnn_mem_t d_src_add_fp(d_data_md, fp, tag, ref_engine);
        if (!prb->inplace) {
            placeholder_d_src_dt = dnn_mem_t(d_data_md, test_engine);
        }
        dnn_mem_t &d_src_dt = prb->inplace ? d_dst_dt : placeholder_d_src_dt;
        dnn_mem_t d_src_add_dt = dnn_mem_t(d_data_md, test_engine);

        scratchpad_dt = dnn_mem_t(d_scratchpad_md, test_engine);

        SAFE(prepare_bwd(prb, d_dst_dt, d_dst_fp), WARN);

        args.clear();
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_SRC_1, src_add_dt);
        args.set(DNNL_ARG_MEAN, mean_dt);
        args.set(DNNL_ARG_VARIANCE, var_dt);
        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_SCALE, sc_dt);
        args.set(DNNL_ARG_SHIFT, sh_dt);
        args.set(DNNL_ARG_WORKSPACE, ws_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_DIFF_SCALE, d_sc_dt);
        args.set(DNNL_ARG_DIFF_SHIFT, d_sh_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
        // Since DIFF_SRC_1 is the second output it can be in blocked format
        // and unconditional including leads zero-paddiing failures
        if (fuse_add_relu) args.set(DNNL_ARG_DIFF_SRC_1, d_src_add_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_SRC_1, src_add_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(DNNL_ARG_SCALE, sc_fp);
            ref_args.set(DNNL_ARG_SHIFT, sh_fp);
            ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_DST_1, src_hat_fp); // Reference aux arg.
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC_1, d_src_add_fp);
            ref_args.set(DNNL_ARG_DIFF_SCALE, d_sc_fp);
            ref_args.set(DNNL_ARG_DIFF_SHIFT, d_sh_fp);

            std::vector<data_kind_t> kinds {SRC};
            if (use_sc && (prb->dir & FLAG_WEI)) kinds.push_back(SC);
            if (use_sh && (prb->dir & FLAG_WEI)) kinds.push_back(SH);
            if (fuse_add_relu) kinds.push_back(SRC_1);

            check_correctness(prb, kinds, args, ref_args, setup_cmp, res);
        }
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace bnorm
