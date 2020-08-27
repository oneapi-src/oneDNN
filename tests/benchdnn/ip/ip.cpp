/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "ip/ip.hpp"

namespace ip {
/* extra control parameter which shouldn't be placed in prb_t */

static int init_pd(dnnl_engine_t engine, const prb_t *p,
        dnnl_primitive_desc_t &ippd, res_t *r, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_inner_product_desc_t ipd;
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

    SAFE(init_md(&src_d, p->ndims, src_dims, p->cfg[SRC].dt, p->stag), CRIT);
    SAFE(init_md(&wei_d, p->ndims, wei_dims, p->cfg[WEI].dt, p->wtag), CRIT);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &bia_d, 1, bia_dims, p->cfg[BIA].dt, dnnl_format_tag_any),
            WARN);
    SAFE(init_md(&dst_d, 2, dst_dims, p->cfg[DST].dt, p->dtag), CRIT);

    switch (p->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            DNN_SAFE(dnnl_inner_product_forward_desc_init(&ipd,
                             p->dir == FWD_I ? dnnl_forward_inference
                                             : dnnl_forward_training,
                             &src_d, &wei_d, p->dir == FWD_B ? &bia_d : nullptr,
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
            DNN_SAFE(
                    dnnl_inner_product_backward_weights_desc_init(&ipd, &src_d,
                            &wei_d, p->dir == BWD_W ? nullptr : &bia_d, &dst_d),
                    WARN);
            break;
        default: DNN_SAFE(dnnl_invalid_arguments, CRIT);
    }

    DNN_SAFE(ipd.accum_data_type == p->cfg[ACC].dt ? dnnl_success
                                                   : dnnl_unimplemented,
            CRIT);

    attr_args_t attr_args;
    attr_args.prepare_output_scales(p->attr, p->scales, p->oc);
    auto dnnl_attr = create_dnnl_attr(p->attr, attr_args);

    dnnl_status_t init_status = dnnl_success;
    init_status = dnnl_primitive_desc_create(
            &ippd, &ipd, dnnl_attr, engine, nullptr);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    r->impl_name = query_impl_info(ippd);
    if (maybe_skip(r->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                r->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(ippd), WARN);
        return r->state = SKIPPED, r->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", r->impl_name.c_str());
    }

    return OK;
}

inline int compare_dat(const prb_t *p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return r->state = PASSED, OK;

    r->total = nelems;

    int64_t non_zero = 0;
    const char *skind = data_kind2str(kind);

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = round_to_nearest_representable(p->cfg[kind].dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= p->cfg[kind].eps;

        r->errors += !ok;

        const bool dump = false || (!ok && (r->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            BENCHDNN_PRINT(0,
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
        BENCHDNN_PRINT(0,
                "@@@ [%s] test-bug: trust is too low."
                " Nonzeros in output: %.2f\n",
                skind, trust_nz);
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int fill_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    const bool need_extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t extra_mem;
    if (need_extra_mem) {
        const auto tag = tag::abx;
        extra_mem = dnn_mem_t(mem_dt.md_, dnnl_f32, tag, get_test_engine());
    }
    dnn_mem_t &mem_00 = need_extra_mem ? extra_mem : mem_fp;

    const auto &c = p->cfg[SRC];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(p->mb, p->ic, p->id, p->ih, p->iw,
            [&](int mb, int ic, int id, int ih, int iw) {
                const int gen
                        = 101 * id + 103 * ih + 107 * iw + 109 * mb + 113 * ic;
                const bool non_base = flip_coin(gen, c.f_sparsity);
                const float value
                        = non_base ? c.f_min + gen * 1 % range : c.f_base;

                ((float *)mem_00)[src_off_f(p, mb, ic, id, ih, iw)] = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) { SAFE(mem_fp.reorder(mem_dt), WARN); }

    return OK;
}

int fill_wei(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    const bool s8_s8 = p->cfg[WEI].dt == dnnl_s8 && p->cfg[SRC].dt == dnnl_s8;
    const bool diff_data_type = mem_dt.dt() != mem_fp.dt();
    const bool check_reorder = diff_data_type && !s8_s8;

    dnn_mem_t extra_mem;
    if (check_reorder) {
        const auto tag = tag::abx;
        extra_mem = dnn_mem_t(mem_dt.md_, dnnl_f32, tag, get_test_engine());
    }
    dnn_mem_t &mem_00 = check_reorder ? extra_mem : mem_fp;

    const auto &c = p->cfg[WEI];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(p->oc, p->ic, p->id, p->ih, p->iw,
            [&](int oc, int ic, int kd, int kh, int kw) {
                const int gen
                        = 127 * kd + 131 * kh + 137 * kw + 139 * oc + 149 * ic;
                const bool non_base = flip_coin(gen, c.f_sparsity);
                const float value
                        = non_base ? c.f_min + gen * 1 % range : c.f_base;
                ((float *)mem_00)[wei_off_f(p, oc, ic, kd, kh, kw)] = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (check_reorder) { SAFE(mem_fp.reorder(mem_dt), WARN); }

    return OK;
}

int fill_bia(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    const bool need_extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t extra_mem;
    if (need_extra_mem)
        extra_mem = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::x, get_test_engine());
    dnn_mem_t &mem_00 = need_extra_mem ? extra_mem : mem_fp;

    const size_t nelems = mem_00.nelems();
    if (nelems == 0) return OK;

    const auto &c = p->cfg[BIA];
    const int range = c.f_max - c.f_min + 1;

    for (size_t i = 0; i < nelems; ++i) {
        const int gen = (int)(151 * i);
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value = non_base ? c.f_min + gen * 1 % range : c.f_base;

        ((float *)mem_00)[i] = value;
    }

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) { SAFE(mem_fp.reorder(mem_dt), WARN); }
    return OK;
}

int fill_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    const bool need_extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t extra_mem;
    if (need_extra_mem) {
        const auto tag = tag::abx;
        extra_mem = dnn_mem_t(mem_dt.md_, dnnl_f32, tag, get_test_engine());
    }
    dnn_mem_t &mem_00 = need_extra_mem ? extra_mem : mem_fp;

    const auto &c = p->cfg[DST];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(p->mb, p->oc, [&](int mb, int oc) {
        const int gen = 173 * mb + 179 * oc;
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value = non_base ? c.f_min + gen * 1 % range : c.f_base;

        ((float *)mem_00)[dst_off_f(p, mb, oc)] = value;
    });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) { SAFE(mem_fp.reorder(mem_dt), WARN); }

    return OK;
}

void check_known_skipped_case(const prb_t *p, res_t *r) {
    check_known_skipped_case_common(
            {p->cfg[SRC].dt, p->cfg[WEI].dt, p->cfg[DST].dt}, p->dir, r);
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    check_known_skipped_case(p, r);
    if (r->state == SKIPPED) return OK;

    dnnl_primitive_t ip {};
    SAFE(init_prim(&ip, init_pd, p, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(ip, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(ip));
        return r->state = SKIPPED, r->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto &src_md
            = p->dir == BWD_D ? q(DNNL_ARG_DIFF_SRC) : q(DNNL_ARG_SRC);
    const auto &wei_md = p->dir & FLAG_WEI ? q(DNNL_ARG_DIFF_WEIGHTS)
                                           : q(DNNL_ARG_WEIGHTS);
    const auto &bia_md
            = p->dir & FLAG_WEI ? q(DNNL_ARG_DIFF_BIAS) : q(DNNL_ARG_BIAS);
    const auto &dst_md
            = p->dir & FLAG_BWD ? q(DNNL_ARG_DIFF_DST) : q(DNNL_ARG_DST);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto src_tag = tag::abx;
    const auto wei_tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t bia_dt(bia_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    dnn_mem_t src_fp(src_md, fp, src_tag, test_engine);
    dnn_mem_t wei_fp(wei_md, fp, wei_tag, test_engine);
    dnn_mem_t bia_fp(bia_md, fp, tag::x, test_engine);
    dnn_mem_t dst_fp(dst_md, fp, tag::abx, test_engine);

    SAFE(fill_src(p, src_dt, src_fp, r), WARN);
    SAFE(fill_wei(p, wei_dt, wei_fp, r), WARN);
    SAFE(fill_bia(p, bia_dt, bia_fp, r), WARN);
    SAFE(fill_dst(p, dst_dt, dst_fp, r), WARN);

    args_t args;

    if (p->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(ip, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(test_engine, p, src_fp, wei_fp, bia_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, tag::abx, test_engine);
            SAFE(compare_dat(p, DST, dst, dst_fp, r), WARN);
        }
    } else if (p->dir == BWD_D) {
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(ip, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_d(p, src_fp, wei_fp, dst_fp);
            dnn_mem_t src(src_dt, fp, src_tag, test_engine);
            SAFE(compare_dat(p, SRC, src, src_fp, r), WARN);
        }
    } else if (p->dir & FLAG_BWD && p->dir & FLAG_WEI) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_BIAS, bia_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(ip, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_w(p, src_fp, wei_fp, bia_fp, dst_fp);
            dnn_mem_t wei(wei_dt, fp, wei_tag, test_engine);
            if (compare_dat(p, WEI, wei, wei_fp, r) != OK) return FAIL;
            if (p->dir & FLAG_BIA) {
                dnn_mem_t bia(bia_dt, fp, tag::x, test_engine);
                SAFE(compare_dat(p, BIA, bia, bia_fp, r), WARN);
            }
        }
    }

    measure_perf(r->timer, ip, args);

    DNN_SAFE_V(dnnl_primitive_destroy(ip));

    return OK;
}

} // namespace ip
