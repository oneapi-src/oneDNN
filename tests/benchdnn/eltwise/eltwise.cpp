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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dnnl.h"

#include "src/common/dnnl_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "eltwise/eltwise.hpp"

namespace eltwise {

static int init_pd(const prb_t *p, dnnl_eltwise_desc_t &ed,
        dnnl_primitive_desc_t &epd, res_t *r) {
    dnnl_memory_desc_t data_d;

    const int ndims = (int)p->dims.size();

    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &data_d, ndims, p->dims.data(), p->dt, p->tag),
            WARN);

    dnnl_alg_kind_t alg = attr_t::post_ops_t::kind2dnnl_kind(p->alg);

    if (p->dir & FLAG_FWD) {
        auto prop = p->dir & FLAG_INF ? dnnl_forward_inference
                                      : dnnl_forward_training;

        DNN_SAFE(dnnl_eltwise_forward_desc_init(
                         &ed, prop, alg, &data_d, p->alpha, p->beta),
                WARN);
    } else {
        dnnl_memory_desc_t diff_data_d;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_data_d, ndims,
                         p->dims.data(), p->dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_eltwise_backward_desc_init(
                         &ed, alg, &diff_data_d, &data_d, p->alpha, p->beta),
                WARN);
    }

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&epd, &ed, NULL, engine_tgt, NULL);

    if (init_status == dnnl_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(epd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: dnnl implementation: %s\n", impl_str);
        DNN_SAFE(dnnl_primitive_desc_destroy(epd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "dnnl implementation: %s\n", impl_str);
    }

    return OK;
}

// check that on a given input specific alg may return NaN or inf
static bool check_extreme_values(
        const prb_t *p, const float &src, const float &library_output) {
    switch (p->alg) {
        case alg_t::SQRT:
            if ((p->dir & FLAG_FWD) && src < 0) return true;
            if ((p->dir & FLAG_BWD) && src <= 0) return true;
        case alg_t::LOG:
            if ((p->dir & FLAG_FWD) && src < 0) return true;
            if ((p->dir & FLAG_FWD) && src == 0 && library_output == logf(src))
                return true;
            if ((p->dir & FLAG_BWD) && src <= 0) return true;
        default: return false;
    }
}

static bool check_abs_err(const prb_t *p, const float &s, const float &trh) {
    float approx_machine_eps = 2e-7;

    switch (p->alg) {
        case alg_t::GELU: {
            // catch catastrophic cancellation
            const float sqrt_2_over_pi = 0.797884;
            const float fitting_const = 0.044715;
            float v = tanhf(sqrt_2_over_pi * s * (1 + fitting_const * s * s));
            float dg = sqrt_2_over_pi * (1 + 3 * fitting_const * s * s);
            if (p->dir & FLAG_FWD)
                return fabsf(1.f + v) <= (approx_machine_eps / trh);
            else
                return fabsf(1.f + v) <= (approx_machine_eps / trh)
                        || (std::signbit(s)
                                && fabsf(1.f + s * (1.f - v) * dg)
                                        <= (approx_machine_eps / trh));
        }
        case alg_t::TANH:
            // catch catastrophic cancellation,
            // which occurs when err in tanh(s) is high
            // and tanh(s) is close to 1.
            return (p->dir & FLAG_BWD)
                    && fabsf(1.f - tanhf(fabsf(s)))
                    <= (approx_machine_eps / trh);
        case alg_t::SRELU:
            // when s is negative, expf(s) -> 0 rapidly
            // which leads to log1pf(expf(s)) -> 0
            // which leads to high relative error,
            // while abs error is still low.
            // (10.f is magic scale for bf16)
            return (p->dir & FLAG_FWD) && std::signbit(s)
                    && log1pf(expf(s)) <= 10.f * (approx_machine_eps / trh);
        default: return false;
    }
}

static int compare(const prb_t *p, const dnn_mem_t &mem_src_fp,
        const dnn_mem_t &mem_fp, const dnn_mem_t &mem_dt, res_t *r) {
    // Tolerate ~3 ulp of relative error for fp32.
    float trh = 2e-6;
    // Tolerate only rounding error(~1/2 ulp) for reduced precision.
    if (p->dt == dnnl_f16 || p->dt == dnnl_bf16) trh = epsilon_dt(p->dt);

    // Tolerate ~7ulp for complex primitives in fp32.
    if (p->dt == dnnl_f32
            && (p->alg == alg_t::GELU || p->alg == alg_t::ELU
                    || p->alg == alg_t::SWISH || p->alg == alg_t::TANH
                    || p->alg == alg_t::SRELU || p->alg == alg_t::LOG))
        trh = 3e-5;

    const auto nelems = mem_dt.nelems();
    r->errors = 0;
    r->total = nelems;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = mem_dt.get_elem(i);
        const float src = mem_src_fp.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = maybe_saturate(p->dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);

        bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= trh;

        if (!ok) ok = check_extreme_values(p, src, dt);

        if (!ok && check_abs_err(p, src, trh)) ok = diff <= trh;

        r->errors += !ok;

        const bool dump = false || (!ok && (r->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = off2dims_idx(p->dims, i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            print(0,
                    "[%4ld][%s] src:% 9.6g fp0:% 9.6g fp:% 9.6g dt:% 9.6g "
                    "diff:%8.3g rdiff:%8.3g\n",
                    (long)i, ind_str.c_str(), src, fp0, fp, dt, diff, rel_diff);
        }
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

static int compare_padded_area_for_zeros(
        const prb_t *p, const dnn_mem_t &mem_dt, res_t *r) {
    const auto nelems = mem_dt.nelems();
    const auto nelems_padded = mem_dt.nelems(true);
    if (nelems == nelems_padded) return OK; // no padding - no worries

    const auto md = mem_dt.md_;
    const dnnl_dim_t *padded_dims = md.padded_dims;

    // Create memory with dims = md.padded_dims with same format.
    // This way reorder to plain format keeps padding values.
    dnnl_memory_desc_t pad_data_d;
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &pad_data_d, md.ndims, padded_dims, md.data_type, p->tag),
            WARN);
    dnn_mem_t padded_mem_dt(pad_data_d, engine_tgt);
    for (int64_t i = 0; i < nelems_padded; i++)
        padded_mem_dt.set_elem(i, mem_dt.get_elem(i));

    const auto tag = get_default_tag(md.ndims);
    dnn_mem_t plain_padded_mem_dt(padded_mem_dt, md.data_type, tag);

    r->errors = 0;
    r->total = nelems_padded - nelems;

    const auto bd = md.format_desc.blocking;
    int in_blk = bd.inner_nblks;

    // TODO: temporary don't test layouts w/ double and more blocking
    if (in_blk > 1) return OK;

    int64_t idx = bd.inner_idxs[in_blk - 1];
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < idx; i++)
        outer *= md.dims[i];

    for (int64_t i = idx + 1; i < md.ndims; i++)
        inner *= md.dims[i];

    dnnl::impl::parallel_nd(outer, [&](int64_t ou) {
        int64_t offt = (ou * md.padded_dims[idx] + md.dims[idx]) * inner;
        for (int64_t ax = 0; ax < md.padded_dims[idx] - md.dims[idx]; ++ax) {
            for (int64_t in = offt; in < inner + offt; ++in) {
                auto i = ax * inner + in;
                auto dt = plain_padded_mem_dt.get_elem(i);

                bool ok = dt == 0;
                r->errors += !ok;

                const bool dump = false
                        || (!ok && (r->errors < 10 || verbose >= 10))
                        || (verbose >= 50 && i < 30) || (verbose >= 99);
                if (dump) {
                    print(0, "[%4ld] fp:  0.f dt:% 9.6g \n", (long)i, dt);
                }
            }
        }
    });

    if (r->errors) r->state = FAILED;

    return r->state == FAILED ? FAIL : OK;
}

int fill_data_fwd(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        bool is_fwd = true) {
    const auto nelems = mem_fp.nelems();

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const int gen
                = is_fwd ? ((103 * i) + 107) % 109 : ((101 * i) + 103) % 107;

        float value = FLT_MAX;
        switch (i % 4) {
            case 0: value = (gen % 11); break; // int positive
            case 1: value = -(gen % 11); break; // int negative
            case 2: value = gen / 128.; break; // fraction positive
            case 3: value = -gen / 128.; break; // fraction negative
        }

        mem_fp.set_elem(i, maybe_saturate(p->dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_data_bwd(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_data_fwd(p, mem_dt, mem_fp, false);
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    dnnl_eltwise_desc_t ed;
    dnnl_primitive_desc_t epd;
    dnnl_primitive_t e;

    SAFE(init_pd(p, ed, epd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    DNN_SAFE(dnnl_primitive_create(&e, epd), WARN);
    DNN_SAFE(dnnl_primitive_desc_destroy(epd), CRIT);

    const auto fp = dnnl_f32;
    const auto tag = get_default_tag((int)p->dims.size());
    auto &data_desc = ed.data_desc;

    dnn_mem_t src_fp(data_desc, fp, tag, engine_tgt);
    dnn_mem_t src_dt(data_desc, engine_tgt);

    dnn_mem_t dst_fp(data_desc, fp, tag, engine_tgt);
    dnn_mem_t placeholder_dst_dt;
    if (!p->inplace) {
        placeholder_dst_dt = dnn_mem_t(data_desc, engine_tgt);
        SAFE(placeholder_dst_dt.reorder(dst_fp), WARN);
    }
    dnn_mem_t &dst_dt = !p->inplace ? placeholder_dst_dt : src_dt;

    SAFE(fill_data_fwd(p, src_dt, src_fp), WARN);

    dnn_mem_t d_dst_fp, d_dst_dt;
    dnn_mem_t d_src_fp, d_src_dt;

    args_t args;
    args.set(DNNL_ARG_SRC, src_dt);

    if (p->dir & FLAG_FWD) {
        args.set(DNNL_ARG_DST, dst_dt);

        DNN_SAFE(execute_and_wait(e, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(p, src_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, tag, engine_tgt);
            SAFE(compare(p, src_fp, dst_fp, dst, r), WARN);
            SAFE(compare_padded_area_for_zeros(p, dst_dt, r), WARN);
        }
    } else {
        const_dnnl_primitive_desc_t const_epd;
        DNN_SAFE(dnnl_primitive_get_primitive_desc(e, &const_epd), CRIT);
        const auto &d_data_desc = *dnnl_primitive_desc_query_md(
                const_epd, dnnl_query_diff_src_md, 0);

        d_dst_fp = dnn_mem_t(d_data_desc, fp, tag, engine_tgt);
        d_dst_dt = dnn_mem_t(d_data_desc, engine_tgt);

        d_src_fp = dnn_mem_t(d_data_desc, fp, tag, engine_tgt);
        dnn_mem_t placeholder_d_src_dt;
        if (!p->inplace) {
            placeholder_d_src_dt = dnn_mem_t(d_data_desc, engine_tgt);
            SAFE(placeholder_d_src_dt.reorder(d_src_fp), WARN);
        }
        dnn_mem_t &d_src_dt = !p->inplace ? placeholder_d_src_dt : d_dst_dt;

        SAFE(fill_data_bwd(p, d_dst_dt, d_dst_fp), WARN);

        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);

        DNN_SAFE(execute_and_wait(e, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, src_fp, d_dst_fp, d_src_fp);
            dnn_mem_t d_src(d_src_dt, fp, tag, engine_tgt);
            SAFE(compare(p, src_fp, d_src_fp, d_src, r), WARN);
        }
    }

    measure_perf(r->timer, e, args);

    DNN_SAFE(dnnl_primitive_destroy(e), CRIT);

    return OK;
}

} // namespace eltwise
