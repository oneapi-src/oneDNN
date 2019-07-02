/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <sstream>

#include "mkldnn.h"

#include "src/common/mkldnn_thread.hpp"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"
#include "norm.hpp"

#include "lrn/lrn.hpp"

namespace lrn {

inline bool is_3d(const prb_t *p) {
    return p->id > 1;
}

inline bool is_1d(const prb_t *p) {
    return !is_3d(p) && p->ih == 1;
}

int compare(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    const auto nelems = mem_dt.nelems();

    r->errors = 0;
    r->total = nelems;
    float trh = 8e-7 * p->ls;
    if (p->dt == mkldnn_f16)
        trh = 1e-3 * p->ls;
    if (p->dt == mkldnn_bf16)
        trh = 1e-2 * p->ls;

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = maybe_saturate(p->dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= trh;

        r->errors += !ok;

        const bool dump = false
            || (!ok && (r->errors < 10 || verbose >= 10))
            || (verbose >= 50 && i < 30)
            || (verbose >= 99);
        if (dump) {
            int64_t mb = 0, ic = 0, d = 0, h = 0, w = 0;
            inv_data_off(p, i, mb, ic, d, h, w);

            print(0, "[%4ld][" IFMT "," IFMT "," IFMT "," IFMT "," IFMT "] "
                    "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, mb, ic, d, h, w, fp, fp0, dt, diff, rel_diff);
        }
    }

    if (r->errors)
        r->state = FAILED;

    if (r->state == UNTESTED)
        r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int fill_dat(const prb_t *p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    const int range = 16;
    const int f_min = p->dt == mkldnn_u8 ? 0 : -range / 2;

    mkldnn::impl::parallel_nd(nelems, [&](int64_t i) {
        const int64_t gen = kind == SRC ? 1091 * i + 1637 : 1279 * i + 1009;
        const float value = f_min + gen % range;
        mem_fp.set_elem(i, value);
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_dat(p, SRC, mem_dt, mem_fp);
}

int fill_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_dat(p, DST, mem_dt, mem_fp);
}

int init_pd(const prb_t *p, dir_t dir, mkldnn_lrn_desc_t &ld,
        mkldnn_primitive_desc_t &lpd, const_mkldnn_primitive_desc_t hint,
        res_t *r) {
    mkldnn_memory_desc_t data_d;

    const int ndims = is_3d(p) ? 5 : is_1d(p) ? 3 : 4;

    mkldnn_dims_t data_dims_1d = { p->mb, p->ic, p->iw };
    mkldnn_dims_t data_dims_2d = { p->mb, p->ic, p->ih, p->iw };
    mkldnn_dims_t data_dims_3d = { p->mb, p->ic, p->id, p->ih, p->iw };

    mkldnn_dim_t *data_dims
            = is_3d(p) ? data_dims_3d : is_1d(p) ? data_dims_1d : data_dims_2d;

    DNN_SAFE(mkldnn_memory_desc_init_by_tag(
                     &data_d, ndims, data_dims, p->dt, p->tag),
            WARN);

    mkldnn_alg_kind_t alg = alg2alg_kind(p->alg);
    if (dir & FLAG_FWD) {
        auto prop = p->dir & FLAG_INF
            ? mkldnn_forward_inference : mkldnn_forward_training;
        DNN_SAFE(mkldnn_lrn_forward_desc_init(&ld, prop, alg, &data_d, p->ls,
                         p->alpha, p->beta, p->k),
                WARN);
    } else {
        DNN_SAFE(mkldnn_lrn_backward_desc_init(&ld, alg, &data_d, &data_d,
                         p->ls, p->alpha, p->beta, p->k),
                WARN);
    }

    mkldnn_status_t init_status
            = mkldnn_primitive_desc_create(&lpd, &ld, NULL, engine_tgt, hint);

    if (init_status == mkldnn_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(lpd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: mkldnn implementation: %s\n", impl_str);
        DNN_SAFE(mkldnn_primitive_desc_destroy(lpd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "mkldnn implementation: %s\n", impl_str);
    }

    return OK;
}

int init_pd_fwd(const prb_t *p, mkldnn_lrn_desc_t &ld,
        mkldnn_primitive_desc_t &lpd, res_t *r) {
    return init_pd(p, FLAG_FWD, ld, lpd, nullptr, r);
}

int init_pd_bwd(const prb_t *p, mkldnn_lrn_desc_t &ld,
        mkldnn_primitive_desc_t &lpd, const_mkldnn_primitive_desc_t hint,
        res_t *r) {
    return init_pd(p, FLAG_BWD, ld, lpd, hint, r);
}

int doit(const prb_t *p, res_t *r) {
    mkldnn_lrn_desc_t lfd, lbd;
    mkldnn_primitive_desc_t lfpd, lbpd;
    mkldnn_primitive_t lf, lb;

    SAFE(init_pd_fwd(p, lfd, lfpd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED)
        return OK;

    dnn_mem_t ws_dt, ws_fp;
    if (!(p->dir & FLAG_INF)) {
        const auto &ws_d = *mkldnn_primitive_desc_query_md(
                lfpd, mkldnn_query_workspace_md, 0);
        if (ws_d.format_kind != mkldnn_format_kind_undef) {
            ws_dt = dnn_mem_t(ws_d, engine_tgt);
            ws_fp = dnn_mem_t(ws_d, engine_ref);
        }
    }

    DNN_SAFE(mkldnn_primitive_create(&lf, lfpd), WARN);

    auto &data_desc = lfd.data_desc;
    dnn_mem_t src_dt(data_desc, engine_tgt);
    dnn_mem_t dst_dt(data_desc, engine_tgt);
    dnn_mem_t d_dst_dt, d_src_dt;

    const auto fp = mkldnn_f32;
    const auto tag = get_default_tag(src_dt.md_.ndims);

    dnn_mem_t src_fp(data_desc, fp, tag, engine_ref);
    dnn_mem_t dst_fp(data_desc, fp, tag, engine_ref);
    dnn_mem_t d_dst_fp, d_src_fp;

    SAFE(fill_src(p, src_dt, src_fp), WARN);
    SAFE(dst_dt.reorder(dst_fp), WARN);

    args_t args_fwd, args_bwd;
    args_fwd.set(MKLDNN_ARG_SRC, src_dt.m_);
    args_fwd.set(MKLDNN_ARG_DST, dst_dt.m_);
    if (!(p->dir & FLAG_INF))
        args_fwd.set(MKLDNN_ARG_WORKSPACE, ws_dt.m_);

    args_t &args = args_fwd;
    mkldnn_primitive_t l = lf;

    DNN_SAFE(execute_and_wait(l, stream_tgt, args.size(), args), WARN);

    if (p->dir & FLAG_FWD) {
        if (bench_mode & CORR) {
            compute_ref_fwd(p, src_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, tag, engine_ref);
            SAFE(compare(p, dst, dst_fp, r), WARN);
        }
    }

    if (p->dir & FLAG_BWD) {
        SAFE(init_pd_bwd(p, lbd, lbpd, lfpd, r), WARN);
        if (r->state == SKIPPED || r->state == UNIMPLEMENTED)
            return OK;

        DNN_SAFE(mkldnn_primitive_create(&lb, lbpd), WARN);
        DNN_SAFE(mkldnn_primitive_desc_destroy(lbpd), CRIT);

        d_dst_dt = dnn_mem_t(data_desc, engine_tgt),
        d_src_dt = dnn_mem_t(data_desc, engine_tgt),
        d_dst_fp = dnn_mem_t(data_desc, fp, tag, engine_ref),
        d_src_fp = dnn_mem_t(data_desc, fp, tag, engine_ref);

        SAFE(fill_dst(p, d_dst_dt, d_dst_fp), WARN);
        SAFE(d_src_dt.reorder(d_src_fp), WARN);

        args_bwd.set(MKLDNN_ARG_SRC, src_dt.m_);
        args_bwd.set(MKLDNN_ARG_DIFF_DST, d_dst_dt.m_);
        args_bwd.set(MKLDNN_ARG_DIFF_SRC, d_src_dt.m_);
        args_bwd.set(MKLDNN_ARG_WORKSPACE, ws_dt.m_);

        args = args_bwd;
        l = lb;

        DNN_SAFE(execute_and_wait(l, stream_tgt, args.size(), args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, src_fp, d_dst_fp, d_src_fp);
            dnn_mem_t d_src(d_src_dt, fp, tag, engine_ref);
            SAFE(compare(p, d_src, d_src_fp, r), WARN);
        }
    }

    measure_perf(r->timer, l, args);

    DNN_SAFE(mkldnn_primitive_desc_destroy(lfpd), CRIT);
    DNN_SAFE(mkldnn_primitive_destroy(lf), CRIT);
    if (p->dir & FLAG_BWD)
        DNN_SAFE(mkldnn_primitive_destroy(lb), CRIT);

    return OK;
}

}
