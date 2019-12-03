/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "dnnl.h"

#include "src/common/dnnl_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "ip/ip.hpp"

namespace ip {
/* extra control parameter which shouldn't be placed in prb_t */
extern const char *skip_impl; /* NULL or "" means do not skip anything */

inline int init_pd(const prb_t *p, dnnl_inner_product_desc_t &ipd,
        dnnl_primitive_desc_t &ippd, res_t *r) {
    dnnl_memory_desc_t src_d, wei_d, bia_d, dst_d;

    dnnl_dims_t src_dims_0d = {p->mb, p->ic};
    dnnl_dims_t src_dims_1d = {p->mb, p->ic, p->iw};
    dnnl_dims_t src_dims_2d = {p->mb, p->ic, p->ih, p->iw};
    dnnl_dims_t src_dims_3d = {p->mb, p->ic, p->id, p->ih, p->iw};
    dnnl_dims_t wei_dims_0d = {p->oc, p->ic};
    dnnl_dims_t wei_dims_1d = {p->oc, p->ic, p->iw};
    dnnl_dims_t wei_dims_2d = {p->oc, p->ic, p->ih, p->iw};
    dnnl_dims_t wei_dims_3d = {p->oc, p->ic, p->id, p->ih, p->iw};
    dnnl_dims_t bia_dims = {p->oc};
    dnnl_dims_t dst_dims = {p->mb, p->oc};

    dnnl_dim_t *src_dims = p->ndims == 5
            ? src_dims_3d
            : p->ndims == 4 ? src_dims_2d
                            : p->ndims == 3 ? src_dims_1d : src_dims_0d;

    dnnl_dim_t *wei_dims = p->ndims == 5
            ? wei_dims_3d
            : p->ndims == 4 ? wei_dims_2d
                            : p->ndims == 3 ? wei_dims_1d : wei_dims_0d;

    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &src_d, p->ndims, src_dims, p->cfg[SRC].dt, p->stag),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &wei_d, p->ndims, wei_dims, p->cfg[WEI].dt, p->wtag),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &bia_d, 1, bia_dims, p->cfg[BIA].dt, dnnl_format_tag_any),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &dst_d, 2, dst_dims, p->cfg[DST].dt, p->dtag),
            WARN);

    switch (p->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            DNN_SAFE(dnnl_inner_product_forward_desc_init(&ipd,
                             p->dir == FWD_I ? dnnl_forward_inference
                                             : dnnl_forward_training,
                             &src_d, &wei_d, p->dir == FWD_B ? &bia_d : NULL,
                             &dst_d),
                    WARN);
            break;
        case BWD_D:
            DNN_SAFE(dnnl_inner_product_backward_data_desc_init(
                             &ipd, &src_d, &wei_d, &dst_d),
                    WARN);
            break;
        case BWD_W:
        case BWD_WB:
            DNN_SAFE(dnnl_inner_product_backward_weights_desc_init(&ipd, &src_d,
                             &wei_d, p->dir == BWD_W ? NULL : &bia_d, &dst_d),
                    WARN);
            break;
        default: DNN_SAFE(dnnl_invalid_arguments, CRIT);
    }

    DNN_SAFE(ipd.accum_data_type == p->cfg[ACC].dt ? dnnl_success
                                                   : dnnl_unimplemented,
            CRIT);

    auto dnnl_attr = create_dnnl_attr(p->attr, p->oc, p->scales);

    dnnl_status_t init_status = dnnl_success;
    init_status = dnnl_primitive_desc_create(
            &ippd, &ipd, dnnl_attr, engine_tgt, NULL);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(ippd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: dnnl implementation: %s\n", impl_str);
        DNN_SAFE(dnnl_primitive_desc_destroy(ippd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "dnnl implementation: %s\n", impl_str);
    }

    auto q = [=](dnnl_query_t query, int index = 0) {
        return *dnnl_primitive_desc_query_md(ippd, query, index);
    };

    if (p->dir == BWD_D)
        ipd.diff_src_desc = q(dnnl_query_diff_src_md);
    else
        ipd.src_desc = q(dnnl_query_src_md);

    if (p->dir & FLAG_WEI)
        ipd.diff_weights_desc = q(dnnl_query_diff_weights_md);
    else
        ipd.weights_desc = q(dnnl_query_weights_md);

    if (p->dir & FLAG_BIA) {
        if (p->dir & FLAG_BWD)
            ipd.diff_bias_desc = q(dnnl_query_diff_weights_md, 1);
        else
            ipd.bias_desc = q(dnnl_query_weights_md, 1);
    }

    if (p->dir & FLAG_BWD)
        ipd.diff_dst_desc = q(dnnl_query_diff_dst_md);
    else
        ipd.dst_desc = q(dnnl_query_dst_md);

    return OK;
}

inline int compare_dat(const prb_t *p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r) {
    const auto nelems = mem_dt.nelems();
    int64_t non_zero = 0;
    const char *skind = data_kind2str(kind);

    r->errors = 0;
    r->total = nelems;

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = maybe_saturate(p->cfg[kind].dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= p->cfg[kind].eps;

        r->errors += !ok;

        const bool dump = false || (!ok && (r->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            print(0,
                    "[%4ld][%s]"
                    "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, skind, fp, fp0, dt, diff, rel_diff);
        }
        non_zero += fp != 0;
    }

    const double trust_nz = (double)non_zero / r->total;
    bool no_trust = trust_nz < 0.1;
    if (no_trust) {
        r->state = MISTRUSTED;
        const char *skind = data_kind2str(kind);
        print(0,
                "@@@ [%s] test-bug: trust is too low."
                " Nonzeros in output: %.2f\n",
                skind, trust_nz);
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int fill_data(data_kind_t kind, const prb_t *p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r) {
    dnn_mem_t mem_00(mem_dt.md_, dnnl_f32, get_default_tag(mem_dt.md_.ndims),
            engine_tgt);

    const auto nelems = mem_dt.nelems();
    assert(mem_dt.nelems() == mem_fp.nelems());

    const auto &c = p->cfg[kind];

    dnnl::impl::parallel(0, [&](int ithr, int nthr) {
        int64_t chunk_size = (nelems + nthr - 1) / nthr;
        int64_t idx_start = ithr * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        std::minstd_rand msr;
        std::uniform_int_distribution<> gen(c.f_min, c.f_max);
        msr.discard(kind + idx_start);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            auto val = (float)gen(msr) * c.f_scale;
            mem_00.set_elem(idx, val);
        }
    });

    SAFE(mem_dt.reorder(mem_00), WARN);
    SAFE(mem_fp.reorder(mem_dt), WARN);
    return OK;
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    dnnl_inner_product_desc_t ipd;
    dnnl_primitive_desc_t ippd;
    dnnl_primitive_t ip;

    SAFE(init_pd(p, ipd, ippd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    DNN_SAFE(dnnl_primitive_create(&ip, ippd), WARN);
    DNN_SAFE(dnnl_primitive_desc_destroy(ippd), CRIT);

    auto &src_dt_d = p->dir == BWD_D ? ipd.diff_src_desc : ipd.src_desc;
    auto &wei_dt_d
            = p->dir & FLAG_WEI ? ipd.diff_weights_desc : ipd.weights_desc;
    auto &bia_dt_d = p->dir & FLAG_BWD ? ipd.diff_bias_desc : ipd.bias_desc;
    auto &dst_dt_d = p->dir & FLAG_BWD ? ipd.diff_dst_desc : ipd.dst_desc;

    const auto fp = dnnl_f32;
    dnn_mem_t src_dt(
            src_dt_d, p->cfg[SRC].dt, dnnl_format_tag_undef, engine_tgt);
    dnn_mem_t wei_dt(
            wei_dt_d, p->cfg[WEI].dt, dnnl_format_tag_undef, engine_tgt);
    dnn_mem_t dst_dt(
            dst_dt_d, p->cfg[DST].dt, dnnl_format_tag_undef, engine_tgt);
    dnn_mem_t bia_dt = p->dir & FLAG_BIA
            ? dnn_mem_t(bia_dt_d, p->cfg[BIA].dt, engine_tgt)
            : dnn_mem_t();

    const auto src_tag = get_default_tag(src_dt.md_.ndims);
    const auto wei_tag = get_default_tag(wei_dt.md_.ndims);
    dnn_mem_t src_fp(src_dt_d, fp, src_tag, engine_tgt);
    dnn_mem_t wei_fp(wei_dt_d, fp, wei_tag, engine_tgt);
    dnn_mem_t dst_fp(dst_dt_d, fp, dnnl_nc, engine_tgt);
    dnn_mem_t bia_fp = p->dir & FLAG_BIA
            ? dnn_mem_t(bia_dt_d, fp, dnnl_x, engine_tgt)
            : dnn_mem_t();

    SAFE(fill_data(SRC, p, src_dt, src_fp, r), WARN);
    SAFE(fill_data(WEI, p, wei_dt, wei_fp, r), WARN);
    SAFE(fill_data(DST, p, dst_dt, dst_fp, r), WARN);
    if (p->dir & FLAG_BIA) SAFE(fill_data(BIA, p, bia_dt, bia_fp, r), WARN);

    args_t args;

    if (p->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        if (p->dir & FLAG_BIA) args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        DNN_SAFE(execute_and_wait(ip, stream_tgt, args), WARN);
        if (bench_mode & CORR) {
            compute_ref_fwd(p, src_fp, wei_fp, bia_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, dnnl_nc, engine_tgt);
            SAFE(compare_dat(p, DST, dst, dst_fp, r), WARN);
        }
    } else if (p->dir == BWD_D) {
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);

        DNN_SAFE(execute_and_wait(ip, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_d(p, src_fp, wei_fp, dst_fp);
            dnn_mem_t src(src_dt, fp, src_tag, engine_tgt);
            SAFE(compare_dat(p, SRC, src, src_fp, r), WARN);
        }
    } else if (p->dir & FLAG_BWD && p->dir & FLAG_WEI) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
        if (p->dir & FLAG_BIA) args.set(DNNL_ARG_DIFF_BIAS, bia_dt);

        DNN_SAFE(execute_and_wait(ip, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_w(p, src_fp, wei_fp, bia_fp, dst_fp);
            dnn_mem_t wei(wei_dt, fp, wei_tag, engine_tgt);
            if (compare_dat(p, WEI, wei, wei_fp, r) != OK) return FAIL;
            if (p->dir & FLAG_BIA) {
                dnn_mem_t bia(bia_dt, fp, dnnl_x, engine_tgt);
                SAFE(compare_dat(p, BIA, bia, bia_fp, r), WARN);
            }
        }
    }

    measure_perf(r->timer, ip, args);

    DNN_SAFE(dnnl_primitive_destroy(ip), CRIT);

    return OK;
}

} // namespace ip
