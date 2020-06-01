/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

#include "pool/pool.hpp"

namespace pool {

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
            BENCHDNN_PRINT(0,
                    "[%4ld][" IFMT "," IFMT "," IFMT "," IFMT "," IFMT
                    "] "
                    "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, mb, ic, d, h, w, fp, fp0, dt, diff, rel_diff);
        }
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int compare_src(
        const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    return compare_dat(p, SRC, mem_dt, mem_fp, r);
}

int compare_dst(
        const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    return compare_dat(p, DST, mem_dt, mem_fp, r);
}

int fill_dat(const prb_t *p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r) {
    const int64_t MB {p->mb};
    const int64_t IC {p->ic};
    const int64_t D {kind == SRC ? p->id : p->od};
    const int64_t H {kind == SRC ? p->ih : p->oh};
    const int64_t W {kind == SRC ? p->iw : p->ow};
    const int64_t ker_size {p->kd * p->kh * p->kw};
    const auto &c = p->cfg[kind];

    dnnl::impl::parallel_nd(MB, IC, D, H, W,
            [&](int64_t mb, int64_t ic, int64_t d, int64_t h, int64_t w) {
                const int64_t factor = p->alg == MAX ? 1 : ker_size;
                // keep values for avg_exclude_pad positive to prevent cancellation err
                const int64_t f_min = p->alg == MAX ? c.f_min / factor : 0;
                // divide on factor to keep value in the range
                const int64_t range = c.f_max / factor - f_min + 1;
                const int64_t gen
                        = 5 * d + 17 * h + 13 * w + 13 * mb + 19 * ic + 1637;
                const float value = (f_min + gen % range) * factor;

                const size_t off = kind == SRC ? src_off_f(p, mb, ic, d, h, w)
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
    dnnl::impl::parallel_nd(mem_fp.nelems(),
            [&](int64_t i) { mem_fp.set_elem(i, (1 << 24) - 1); });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

static int init_pd(dnnl_engine_t engine, const prb_t *p,
        dnnl_primitive_desc_t &ppd, res_t *r, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_memory_desc_t src_d, dst_d;

    dnnl_dims_t src_1d_dims = {p->mb, p->ic, p->iw};
    dnnl_dims_t src_2d_dims = {p->mb, p->ic, p->ih, p->iw};
    dnnl_dims_t src_3d_dims = {p->mb, p->ic, p->id, p->ih, p->iw};
    dnnl_dim_t *src_dims = p->ndims == 5
            ? src_3d_dims
            : p->ndims == 4 ? src_2d_dims : src_1d_dims;

    dnnl_dims_t dst_1d_dims = {p->mb, p->ic, p->ow};
    dnnl_dims_t dst_2d_dims = {p->mb, p->ic, p->oh, p->ow};
    dnnl_dims_t dst_3d_dims = {p->mb, p->ic, p->od, p->oh, p->ow};
    dnnl_dim_t *dst_dims = p->ndims == 5
            ? dst_3d_dims
            : p->ndims == 4 ? dst_2d_dims : dst_1d_dims;

    const auto src_tag = (dir & FLAG_FWD) ? convert_tag(p->tag, p->ndims)
                                          : dnnl_format_tag_any;
    const auto dst_tag = dnnl_format_tag_any;

    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &src_d, p->ndims, src_dims, p->cfg[SRC].dt, src_tag),
            WARN);

    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &dst_d, p->ndims, dst_dims, p->cfg[DST].dt, dst_tag),
            WARN);

    dnnl_dim_t strides_nd[] = {p->sd, p->sh, p->sw};
    dnnl_dim_t kernel_nd[] = {p->kd, p->kh, p->kw};
    dnnl_dim_t padding_l_nd[] = {p->pd, p->ph, p->pw};

    auto bph = [](int64_t ih, int64_t oh, int64_t kh, int64_t sh, int64_t ph) {
        return (oh - 1) * sh - ih + kh - ph;
    };
    dnnl_dim_t padding_r_nd[] = {bph(p->id, p->od, p->kd, p->sd, p->pd),
            bph(p->ih, p->oh, p->kh, p->sh, p->ph),
            bph(p->iw, p->ow, p->kw, p->sw, p->pw)};

    dnnl_dim_t *strides = strides_nd + (5 - p->ndims);
    dnnl_dim_t *kernel = kernel_nd + (5 - p->ndims);
    dnnl_dim_t *padding_l = padding_l_nd + (5 - p->ndims);
    dnnl_dim_t *padding_r = padding_r_nd + (5 - p->ndims);

    dnnl_alg_kind_t alg = alg2alg_kind(p->alg);
    dnnl_pooling_desc_t pd;

    if (dir & FLAG_FWD) {
        auto prop_kind = p->dir & FLAG_INF ? dnnl_forward_inference
                                           : dnnl_forward_training;
        DNN_SAFE(dnnl_pooling_forward_desc_init(&pd, prop_kind, alg, &src_d,
                         &dst_d, strides, kernel, padding_l, padding_r),
                WARN);
    } else {
        DNN_SAFE(dnnl_pooling_backward_desc_init(&pd, alg, &src_d, &dst_d,
                         strides, kernel, padding_l, padding_r),
                WARN);
    }

    auto dnnl_attr = create_dnnl_attr(attr_t());

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&ppd, &pd, dnnl_attr, engine, hint);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    // Return if pd is not the one being tested
    if ((dir & FLAG_FWD) != (p->dir & FLAG_FWD)) return OK;

    r->impl_name = query_impl_info(ppd);
    if (maybe_skip(r->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                r->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(ppd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", r->impl_name.c_str());
    }

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    dnnl_primitive_t pp {};
    SAFE(init_prim(&pp, init_pd, p, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_fpd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(pp, &const_fpd), CRIT);

    if (dnn_mem_t::check_mem_size(const_fpd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(pp));
        return r->state = SKIPPED, OK;
    }

    const auto q = [](const_dnnl_primitive_desc_t pd,
                           int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(pd, dnnl_query_exec_arg_md, index);
    };

    const auto &src_md = q(const_fpd, DNNL_ARG_SRC);
    const auto &dst_md = q(const_fpd, DNNL_ARG_DST);
    const auto &ws_md = q(const_fpd, DNNL_ARG_WORKSPACE);
    const auto &scratchpad_md = q(const_fpd, DNNL_ARG_SCRATCHPAD);

    SAFE(!check_md_consistency_with_tag(dst_md, p->tag), WARN);

    const auto fp = dnnl_f32;
    const auto tag = get_abx_tag(p->ndims);

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_fp(src_md, fp, tag, test_engine);
    dnn_mem_t src_dt(src_md, test_engine);

    dnn_mem_t dst_fp(dst_md, fp, tag, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);

    if (p->dir & FLAG_INF) SAFE(ws_md.ndims == 0 ? OK : FAIL, WARN);
    dnn_mem_t ws_fp(ws_md, test_engine);
    dnn_mem_t ws_dt(ws_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    dnn_mem_t d_src_dt, d_dst_dt;

    SAFE(fill_src(p, src_dt, src_fp, r), WARN);

    args_t args;
    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_WORKSPACE, ws_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    DNN_SAFE(execute_and_wait(pp, args), WARN);

    // want this pass on backward to get ws_fp filled properly
    if (bench_mode & CORR) {
        compute_ref_fwd(p, src_fp, dst_fp, ws_fp);
        if (p->dir & FLAG_FWD) {
            dnn_mem_t dst(dst_dt, fp, tag, test_engine);
            SAFE(compare_dst(p, dst, dst_fp, r), WARN);
        }
    }

    if (p->dir & FLAG_BWD) {
        dnnl_primitive_t bwd_p {};
        int status = init_prim(&bwd_p, init_pd, p, r, FLAG_BWD, const_fpd);
        dnnl_primitive_destroy(pp);
        if (status != OK) return status;
        if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;
        pp = bwd_p;

        const_dnnl_primitive_desc_t const_bpd;
        DNN_SAFE(dnnl_primitive_get_primitive_desc(pp, &const_bpd), CRIT);

        if (dnn_mem_t::check_mem_size(const_bpd) != OK) {
            DNN_SAFE_V(dnnl_primitive_destroy(pp));
            return r->state = SKIPPED, OK;
        }

        const auto &d_dst_md = q(const_bpd, DNNL_ARG_DIFF_DST);
        const auto &d_src_md = q(const_bpd, DNNL_ARG_DIFF_SRC);
        const auto &d_scratchpad_md = q(const_bpd, DNNL_ARG_SCRATCHPAD);

        dnn_mem_t d_dst_fp = dnn_mem_t(d_dst_md, fp, tag, test_engine);
        d_dst_dt = dnn_mem_t(d_dst_md, p->cfg[DST].dt, test_engine);

        dnn_mem_t d_src_fp = dnn_mem_t(d_src_md, fp, tag, test_engine);
        d_src_dt = dnn_mem_t(d_src_md, p->cfg[SRC].dt, test_engine);

        scratchpad_dt = dnn_mem_t(d_scratchpad_md, test_engine);

        SAFE(fill_dst(p, d_dst_dt, d_dst_fp, r), WARN);

        args.clear();
        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_WORKSPACE, ws_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        DNN_SAFE(execute_and_wait(pp, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, d_src_fp, d_dst_fp, ws_fp);
            dnn_mem_t diff_src(d_src_dt, fp, tag, test_engine);
            SAFE(compare_src(p, diff_src, d_src_fp, r), WARN);
        }
    }

    measure_perf(r->timer, pp, args);

    DNN_SAFE_V(dnnl_primitive_destroy(pp));

    return OK;
}

} // namespace pool
