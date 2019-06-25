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

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "mkldnn.h"

#include "src/common/mkldnn_thread.hpp"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"
#include "norm.hpp"

#include "pool/pool.hpp"

namespace pool {

inline bool is_3d(const prb_t *p) {
    return p->id > 1 || p->od > 1 || p->kd > 1;
}

inline bool is_1d(const prb_t *p) {
    return !is_3d(p) && p->ih == 1 && p->oh == 1 && p->kh == 1;
}

inline int compare_dat(const prb_t *p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r) {
    const auto nelems = mem_dt.nelems();

    r->errors = 0;
    r->total = nelems;

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = maybe_saturate(p->cfg[kind].dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        bool ok = false;
        if (fp < p->cfg[kind].min)
            ok = dt == p->cfg[kind].min;
        else
            ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= p->cfg[kind].eps;

        r->errors += !ok;

        if ((!ok && (r->errors < 10 || verbose >= 10))
            || (verbose >= 50 && i < 30)) {
                int64_t mb = 0, ic = 0, d = 0, h = 0, w = 0;
                switch (kind) {
                case SRC: inv_src_off_f(p, i, mb, ic, d, h, w); break;
                case DST: inv_dst_off_f(p, i, mb, ic, d, h, w); break;
                }
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

int compare_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r)
{
    return compare_dat(p, SRC, mem_dt, mem_fp, r);
}

int compare_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r)
{
    return compare_dat(p, DST, mem_dt, mem_fp, r);
}

int fill_dat(const prb_t *p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r) {
    const int64_t MB{p->mb};
    const int64_t IC{p->ic};
    const int64_t D{kind == SRC ? p->id : p->od};
    const int64_t H{kind == SRC ? p->ih : p->oh};
    const int64_t W{kind == SRC ? p->iw : p->ow};
    const int64_t ker_size{p->kd * p->kh * p->kw};
    const auto &c = p->cfg[kind];

    mkldnn::impl::parallel_nd(MB, IC, D, H, W,
            [&](int64_t mb, int64_t ic, int64_t d, int64_t h, int64_t w) {
        const int64_t factor = p->alg == MAX ? 1 : ker_size;
        // keep values for avg_exclude_pad positive to prevent cancellation err
        const int64_t f_min = p->alg == MAX ? c.f_min / factor : 0;
        // divide on factor to keep value in the range
        const int64_t range = c.f_max / factor - f_min + 1;
        const int64_t gen = 5 * d + 17 * h + 13 * w + 13 * mb + 19 * ic + 1637;
        const float value = (f_min + gen % range) * factor;

        const size_t off = kind == SRC
            ? src_off_f(p, mb, ic, d, h, w)
            : dst_off_f(p, mb, ic, d, h, w);
        ((float *)mem_fp)[off] = value;
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    return fill_dat(p, SRC, mem_dt, mem_fp, r);
}

int fill_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    return fill_dat(p, DST, mem_dt, mem_fp, r);
}

// fill ws with big numbers to reliably cause a correctness issue (and not
// anything else) in case of a bug in the library
int fill_ws(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    mkldnn::impl::parallel_nd(mem_fp.nelems(), [&](int64_t i) {
        mem_fp.set_elem(i, (1<<24) - 1);
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

inline int init_pd(const prb_t *p, mkldnn_pooling_desc_t &pd,
        mkldnn_primitive_desc_t &ppd, dir_t dir, res_t *r) {
    mkldnn_memory_desc_t src_d, dst_d;

    const int ndims = is_3d(p) ? 5 : is_1d(p) ? 3 : 4;

    mkldnn_dims_t src_1d_dims = { p->mb, p->ic, p->iw };
    mkldnn_dims_t src_2d_dims = { p->mb, p->ic, p->ih, p->iw };
    mkldnn_dims_t src_3d_dims = { p->mb, p->ic, p->id, p->ih, p->iw };
    mkldnn_dim_t *src_dims
            = is_3d(p) ? src_3d_dims : is_1d(p) ? src_1d_dims : src_2d_dims;

    mkldnn_dims_t dst_1d_dims = { p->mb, p->ic, p->ow };
    mkldnn_dims_t dst_2d_dims = { p->mb, p->ic, p->oh, p->ow };
    mkldnn_dims_t dst_3d_dims = { p->mb, p->ic, p->od, p->oh, p->ow };
    mkldnn_dim_t *dst_dims
            = is_3d(p) ? dst_3d_dims : is_1d(p) ? dst_1d_dims : dst_2d_dims;

    DNN_SAFE(mkldnn_memory_desc_init_by_tag(
                     &src_d, ndims, src_dims, p->cfg[SRC].dt, p->tag),
            WARN);

    DNN_SAFE(mkldnn_memory_desc_init_by_tag(
                     &dst_d, ndims, dst_dims, p->cfg[DST].dt, p->tag),
            WARN);

    mkldnn_dim_t strides_nd[] = {p->sd, p->sh, p->sw};
    mkldnn_dim_t kernel_nd[] = {p->kd, p->kh, p->kw};
    mkldnn_dim_t padding_l_nd[] = {p->pd, p->ph, p->pw};

    auto bph = [&](int64_t ih, int64_t oh, int64_t kh, int64_t sh, int64_t ph) {
        return (oh - 1) * sh - ih + kh - ph;
    };
    mkldnn_dim_t padding_r_nd[] = {
        bph(p->id, p->od, p->kd, p->sd, p->pd),
        bph(p->ih, p->oh, p->kh, p->sh, p->ph),
        bph(p->iw, p->ow, p->kw, p->sw, p->pw)
    };

    mkldnn_dim_t *strides = strides_nd + (5 - ndims);
    mkldnn_dim_t *kernel = kernel_nd + (5 - ndims);
    mkldnn_dim_t *padding_l = padding_l_nd + (5 - ndims);
    mkldnn_dim_t *padding_r = padding_r_nd + (5 - ndims);

    mkldnn_alg_kind_t alg = alg2alg_kind(p->alg);
    if (dir & FLAG_FWD) {
        auto prop_kind = p->dir & FLAG_INF
            ? mkldnn_forward_inference : mkldnn_forward_training;
        DNN_SAFE(mkldnn_pooling_forward_desc_init(&pd, prop_kind, alg, &src_d,
                    &dst_d, strides, kernel, padding_l, padding_r), WARN);
    } else {
        DNN_SAFE(mkldnn_pooling_backward_desc_init(&pd, alg, &src_d, &dst_d,
                    strides, kernel, padding_l, padding_r), WARN);
    }

    mkldnn_status_t init_status = mkldnn_success;
    mkldnn_primitive_desc_t hint_fwd_pd = NULL;
    if (dir & FLAG_BWD) {
        mkldnn_pooling_desc_t pd_fwd;
        DNN_SAFE(mkldnn_pooling_forward_desc_init(&pd_fwd,
                    mkldnn_forward_training, alg, &src_d, &dst_d, strides,
                    kernel, padding_l, padding_r), WARN);

        init_status = mkldnn_primitive_desc_create(&hint_fwd_pd, &pd_fwd, NULL,
                engine_tgt, NULL);

        if (init_status == mkldnn_unimplemented)
            return r->state = UNIMPLEMENTED, OK;
        else
            SAFE(init_status, WARN);
    }

    init_status = mkldnn_primitive_desc_create(&ppd, &pd, NULL, engine_tgt,
            hint_fwd_pd);

    if (init_status == mkldnn_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(ppd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: mkldnn implementation: %s\n", impl_str);
        DNN_SAFE(mkldnn_primitive_desc_destroy(ppd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "mkldnn implementation: %s\n", impl_str);
    }

    auto q = [=](mkldnn_query_t query, int index = 0) {
        return *mkldnn_primitive_desc_query_md(ppd, query, index);
    };

    if (dir & FLAG_FWD) {
        pd.src_desc = q(mkldnn_query_src_md);
        pd.dst_desc = q(mkldnn_query_dst_md);
    } else {
        pd.diff_src_desc = q(mkldnn_query_diff_src_md);
        pd.diff_dst_desc = q(mkldnn_query_diff_dst_md);
    }

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    mkldnn_pooling_desc_t pfd, pbd;
    mkldnn_primitive_desc_t pfpd, pbpd;
    mkldnn_primitive_t pf, pb;

    SAFE(init_pd(p, pfd, pfpd, FLAG_FWD, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) {
        return OK;
    }

    DNN_SAFE(mkldnn_primitive_create(&pf, pfpd), WARN);

    dnn_mem_t ws_dt, ws_fp;
    if (p->alg == MAX && !(p->dir & FLAG_INF)) {
        auto &ws_d = *mkldnn_primitive_desc_query_md(pfpd,
                mkldnn_query_workspace_md, 0);
        ws_dt = dnn_mem_t(ws_d, engine_tgt);
        ws_fp = dnn_mem_t(ws_d, engine_ref);
        // to catch usage of uninitialized values in the library
        SAFE(fill_ws(p, ws_dt, ws_fp, r), WARN);
    }

    DNN_SAFE(mkldnn_primitive_desc_destroy(pfpd), CRIT);

    const auto &src_desc = pfd.src_desc;
    const auto &dst_desc = pfd.dst_desc;
    dnn_mem_t src_dt(src_desc, p->cfg[SRC].dt, engine_tgt);
    dnn_mem_t dst_dt(dst_desc, p->cfg[DST].dt, engine_tgt);
    dnn_mem_t d_src_dt, d_dst_dt;

    const auto tag = get_default_tag(src_dt.md_.ndims);
    const auto fp = mkldnn_f32;

    dnn_mem_t src_fp(src_desc, fp, tag, engine_ref);
    dnn_mem_t dst_fp(dst_desc, fp, tag, engine_ref);
    dnn_mem_t d_dst_fp, d_src_fp;

    SAFE(fill_src(p, src_dt, src_fp, r), WARN);

    args_t args_fwd, args_bwd;
    args_fwd.set(MKLDNN_ARG_SRC, src_dt.m_);
    args_fwd.set(MKLDNN_ARG_DST, dst_dt.m_);
    if (p->alg == MAX && !(p->dir & FLAG_INF))
        args_fwd.set(MKLDNN_ARG_WORKSPACE, ws_dt.m_);

    args_t &args = args_fwd;
    mkldnn_primitive_t pl = pf;

    DNN_SAFE(execute_and_wait(pl, stream_tgt, args.size(), args), WARN);

    if (bench_mode & CORR) {
        compute_ref_fwd(p, src_fp, dst_fp, ws_fp);
        if (p->dir & FLAG_FWD) {
            dnn_mem_t dst(dst_dt, fp, tag, engine_ref);
            SAFE(compare_dst(p, dst, dst_fp, r), WARN);
        }
    }

    if (p->dir & FLAG_BWD) {
        SAFE(init_pd(p, pbd, pbpd, FLAG_BWD, r), WARN);
        if (r->state == SKIPPED || r->state == UNIMPLEMENTED)
            return OK;

        DNN_SAFE(mkldnn_primitive_create(&pb, pbpd), WARN);
        DNN_SAFE(mkldnn_primitive_desc_destroy(pbpd), CRIT);

        const auto &d_src_desc = pbd.diff_src_desc;
        const auto &d_dst_desc = pbd.diff_dst_desc;
        d_dst_dt = dnn_mem_t(d_dst_desc, p->cfg[DST].dt, engine_tgt);
        d_src_dt = dnn_mem_t(d_src_desc, p->cfg[SRC].dt, engine_tgt);

        d_dst_fp = dnn_mem_t(d_dst_desc, fp, tag, engine_ref);
        d_src_fp = dnn_mem_t(d_src_desc, fp, tag, engine_ref);

        SAFE(fill_dst(p, d_dst_dt, d_dst_fp, r), WARN);

        args_bwd.set(MKLDNN_ARG_DIFF_DST, d_dst_dt.m_);
        args_bwd.set(MKLDNN_ARG_DIFF_SRC, d_src_dt.m_);
        if(p->alg == MAX)
            args_bwd.set(MKLDNN_ARG_WORKSPACE, ws_dt.m_);

        args = args_bwd;
        pl = pb;

        DNN_SAFE(execute_and_wait(pl, stream_tgt, args.size(), args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, d_src_fp, d_dst_fp, ws_fp);
            dnn_mem_t diff_src(d_src_dt, fp, tag, engine_ref);
            SAFE(compare_src(p, diff_src, d_src_fp, r), WARN);
        }
    }

    measure_perf(r->timer, pl, args);

    DNN_SAFE(mkldnn_primitive_destroy(pf), CRIT);
    if (p->dir & FLAG_BWD)
        DNN_SAFE(mkldnn_primitive_destroy(pb), CRIT);

    return OK;
}

}
