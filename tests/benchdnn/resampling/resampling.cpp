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

#include "src/common/dnnl_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

#include "resampling/resampling.hpp"

namespace resampling {

inline int compare_dat(const prb_t *p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r) {
    const auto nelems = mem_dt.nelems();
    r->errors = 0;
    r->total = nelems;

    float trh = 0;
    if (p->alg == nearest) {
        // On forward, `dst` consists of exact `src` elements, hence the result
        // shall be exact (no matter what data type is). On backward, the
        // diff_src might be a result of accumulation of multiple diff_dst.
        // However, we rely on the fact that benchdnn reference implementation
        // does absolutely the same as the library implementations. We only need
        // to take into account the conversion from accumulation data type
        // (which is float) to the resulting data type.
        if (p->dir & FLAG_FWD)
            trh = 0;
        else
            trh = p->dt != dnnl_f32 ? epsilon_dt(p->dt) : 0;
    } else {
        assert(p->alg == linear);
        trh = p->dt == dnnl_bf16 ? 1e-2 : 1e-6;
    }

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = maybe_saturate(p->dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= trh;

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
    const auto nelems = mem_fp.nelems();
    const auto dt = p->dt;
    const int range = 16;
    const int f_min = 0;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) - 17 * kind + 101) % (range + 1);
        const float value = (dt == dnnl_bf16 || dt == dnnl_f16)
                ? (f_min + gen) / range
                : (f_min + gen) * (1.0f + 4.0f / range);
        mem_fp.set_elem(i, maybe_saturate(dt, value));
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

int init_pd(const engine_t &engine_tgt, const prb_t *p,
        dnnl_primitive_desc_t &rpd, res_t *r) {
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

    std::string src_tag = (p->dir & FLAG_FWD) ? p->tag : tag::any;
    std::string dst_tag = (p->dir & FLAG_BWD) ? p->tag : tag::any;

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_d, p->ndims, src_dims, p->dt,
                     convert_tag(src_tag, p->ndims)),
            WARN);

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_d, p->ndims, dst_dims, p->dt,
                     convert_tag(dst_tag, p->ndims)),
            WARN);

    dnnl_alg_kind_t alg = alg2alg_kind(p->alg);
    dnnl_resampling_desc_t pd;

    if (p->dir & FLAG_FWD) {
        auto prop_kind = p->dir & FLAG_INF ? dnnl_forward_inference
                                           : dnnl_forward_training;
        DNN_SAFE(dnnl_resampling_forward_desc_init(
                         &pd, prop_kind, alg, nullptr, &src_d, &dst_d),
                WARN);
    } else {
        DNN_SAFE(dnnl_resampling_backward_desc_init(
                         &pd, alg, nullptr, &src_d, &dst_d),
                WARN);
    }

    dnnl_primitive_desc_t hint = NULL;
    if (p->dir & FLAG_BWD) {
        dnnl_resampling_desc_t rd_fwd;
        DNN_SAFE(dnnl_resampling_forward_desc_init(&rd_fwd,
                         dnnl_forward_training, alg, nullptr, &src_d, &dst_d),
                WARN);
        dnnl_status_t init_fwd_status = dnnl_primitive_desc_create(
                &hint, &rd_fwd, NULL, engine_tgt, NULL);
        if (init_fwd_status == dnnl_unimplemented)
            return r->state = UNIMPLEMENTED, OK;
        SAFE(init_fwd_status, WARN);
    }

    auto dnnl_attr = create_dnnl_attr(attr_t());

    dnnl_status_t init_status = dnnl_primitive_desc_create(
            &rpd, &pd, dnnl_attr, engine_tgt, hint);

    dnnl_primitive_desc_destroy(hint);
    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented) return r->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(rpd);
    if (maybe_skip(impl_str)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n", impl_str);
        DNN_SAFE(dnnl_primitive_desc_destroy(rpd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", impl_str);
    }

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;
    engine_t engine_tgt(engine_tgt_kind);

    dnnl_primitive_desc_t rpd;
    SAFE(init_pd(engine_tgt, p, rpd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) { return OK; }

    dnnl_primitive_t rp;
    DNN_SAFE(dnnl_primitive_create(&rp, rpd), WARN);
    DNN_SAFE(dnnl_primitive_desc_destroy(rpd), CRIT);

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(rp, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(rp));
        return r->state = SKIPPED, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto &src_md
            = p->dir == BWD_D ? q(DNNL_ARG_DIFF_SRC) : q(DNNL_ARG_SRC);
    const auto &dst_md
            = p->dir == BWD_D ? q(DNNL_ARG_DIFF_DST) : q(DNNL_ARG_DST);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = get_abx_tag(p->ndims);

    dnn_mem_t src_fp(src_md, fp, tag, engine_tgt);
    dnn_mem_t src_dt(src_md, engine_tgt);

    dnn_mem_t dst_fp(dst_md, fp, tag, engine_tgt);
    dnn_mem_t dst_dt(dst_md, engine_tgt);

    dnn_mem_t scratchpad_dt(scratchpad_md, engine_tgt);

    args_t args;

    if (p->dir & FLAG_FWD) {
        SAFE(fill_src(p, src_dt, src_fp, r), WARN);
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        DNN_SAFE(execute_and_wait(rp, engine_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(p, src_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, tag, engine_tgt);
            SAFE(compare_dst(p, dst, dst_fp, r), WARN);
        }
    } else {
        SAFE(fill_dst(p, dst_dt, dst_fp, r), WARN);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        DNN_SAFE(execute_and_wait(rp, engine_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, src_fp, dst_fp);
            dnn_mem_t diff_src(src_dt, fp, tag, engine_tgt);
            SAFE(compare_src(p, diff_src, src_fp, r), WARN);
        }
    }

    measure_perf(r->timer, engine_tgt, rp, args);

    DNN_SAFE_V(dnnl_primitive_destroy(rp));

    return OK;
}

} // namespace resampling
