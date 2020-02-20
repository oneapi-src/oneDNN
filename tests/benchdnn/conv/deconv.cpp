/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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
#include <stdio.h>
#include <stdlib.h>

#include "dnnl.h"

#include "src/common/dnnl_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "norm.hpp"

#include "conv/deconv.hpp"
using namespace conv;

namespace deconv {

inline static void swap(int64_t &a, int64_t &b) {
    int64_t temp = a;
    a = b;
    b = temp;
}

inline int transpose_data_wei(
        const prb_t *p, dnn_mem_t &wei, dnn_mem_t &wei_tr) {
    dnnl::impl::parallel_nd(p->g, p->oc / p->g, p->ic / p->g, p->kd, p->kh,
            p->kw,
            [&](int64_t g, int64_t oc, int64_t ic, int64_t kd, int64_t kh,
                    int64_t kw) {
                size_t idx = (((((size_t)g * p->ic / p->g + ic) * p->oc / p->g
                                       + oc) * p->kd
                                      + kd) * p->kh
                                     + kh)
                                * p->kw
                        + kw;
                ((float *)wei_tr)[idx]
                        = ((float *)wei)[wei_off_f(p, g, oc, ic, kd, kh, kw)];
            });

    return OK;
}

inline int init_pd(const prb_t *p, dnnl_deconvolution_desc_t &cd,
        dnnl_primitive_desc_t &dpd, res_t *r) {
    dnnl_memory_desc_t src_d, wei_d, bia_d, dst_d;
    dnnl_dims_t src_1d_dims = {p->mb, p->ic, p->iw};
    dnnl_dims_t src_2d_dims = {p->mb, p->ic, p->ih, p->iw};
    dnnl_dims_t src_3d_dims = {p->mb, p->ic, p->id, p->ih, p->iw};
    dnnl_dims_t wei_1d_dims = {p->g, p->oc / p->g, p->ic / p->g, p->kw};
    dnnl_dims_t wei_2d_dims = {p->g, p->oc / p->g, p->ic / p->g, p->kh, p->kw};
    dnnl_dims_t wei_3d_dims
            = {p->g, p->oc / p->g, p->ic / p->g, p->kd, p->kh, p->kw};
    dnnl_dims_t bia_dims = {p->oc};
    dnnl_dims_t dst_1d_dims = {p->mb, p->oc, p->ow};
    dnnl_dims_t dst_2d_dims = {p->mb, p->oc, p->oh, p->ow};
    dnnl_dims_t dst_3d_dims = {p->mb, p->oc, p->od, p->oh, p->ow};

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_d, p->ndims,
                     p->ndims == 5 ? src_3d_dims
                                   : p->ndims == 3 ? src_1d_dims : src_2d_dims,
                     p->cfg[SRC].dt, p->stag),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&wei_d, p->ndims + p->has_groups,
                     p->ndims == 5
                             ? &wei_3d_dims[!p->has_groups]
                             : p->ndims == 3 ? &wei_1d_dims[!p->has_groups]
                                             : &wei_2d_dims[!p->has_groups],
                     p->cfg[WEI].dt, p->wtag),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &bia_d, 1, bia_dims, p->cfg[BIA].dt, dnnl_format_tag_any),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_d, p->ndims,
                     p->ndims == 5 ? dst_3d_dims
                                   : p->ndims == 3 ? dst_1d_dims : dst_2d_dims,
                     p->cfg[DST].dt, p->dtag),
            WARN);

    dnnl_dim_t strides_nd[] = {p->sd, p->sh, p->sw};
    dnnl_dim_t dilates_nd[] = {p->dd, p->dh, p->dw};
    dnnl_dim_t padding_nd[] = {p->pd, p->ph, p->pw};

    auto bph = [&](int64_t ih, int64_t oh, int64_t kh, int64_t sh, int64_t ph,
                       int64_t dh) {
        return (oh - 1) * sh - ih + ((kh - 1) * (dh + 1) + 1) - ph;
    };

    dnnl_dim_t padding_r_nd[] = {bph(p->od, p->id, p->kd, p->sd, p->pd, p->dd),
            bph(p->oh, p->ih, p->kh, p->sh, p->ph, p->dh),
            bph(p->ow, p->iw, p->kw, p->sw, p->pw, p->dw)};

    dnnl_dim_t *strides = strides_nd + (5 - p->ndims);
    dnnl_dim_t *dilates = dilates_nd + (5 - p->ndims);
    dnnl_dim_t *padding = padding_nd + (5 - p->ndims);
    dnnl_dim_t *padding_r = padding_r_nd + (5 - p->ndims);

    dnnl_alg_kind_t alg = dnnl_deconvolution_direct;
    if (p->alg == WINO) alg = dnnl_deconvolution_winograd;

    switch (p->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            DNN_SAFE(dnnl_dilated_deconvolution_forward_desc_init(&cd,
                             p->dir == FWD_I ? dnnl_forward_inference
                                             : dnnl_forward_training,
                             alg, &src_d, &wei_d,
                             p->dir == FWD_B ? &bia_d : NULL, &dst_d, strides,
                             dilates, padding, padding_r),
                    WARN);
            break;
        case BWD_D:
            DNN_SAFE(dnnl_dilated_deconvolution_backward_data_desc_init(&cd,
                             alg, &src_d, &wei_d, &dst_d, strides, dilates,
                             padding, padding_r),
                    WARN);
            break;
        case BWD_W:
        case BWD_WB:
            DNN_SAFE(dnnl_dilated_deconvolution_backward_weights_desc_init(&cd,
                             alg, &src_d, &wei_d,
                             p->dir == BWD_W ? NULL : &bia_d, &dst_d, strides,
                             dilates, padding, padding_r),
                    WARN);
            break;
        default: DNN_SAFE(dnnl_invalid_arguments, CRIT);
    }

    DNN_SAFE(cd.accum_data_type == p->cfg[ACC].dt ? dnnl_success
                                                  : dnnl_unimplemented,
            CRIT);

    auto dnnl_attr = create_dnnl_attr(p->attr, p->oc, p->scales);

    dnnl_status_t init_status = dnnl_success;
    init_status = dnnl_primitive_desc_create(
            &dpd, &cd, dnnl_attr, engine_tgt, NULL);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented) {
        return r->state = UNIMPLEMENTED, OK;
    } else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(dpd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: dnnl implementation: %s\n", impl_str);
        DNN_SAFE(dnnl_primitive_desc_destroy(dpd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "dnnl implementation: %s\n", impl_str);
    }

    auto q = [=](dnnl_query_t query, int index = 0) {
        return *dnnl_primitive_desc_query_md(dpd, query, index);
    };

    if (p->dir == BWD_D)
        cd.diff_src_desc = q(dnnl_query_diff_src_md);
    else
        cd.src_desc = q(dnnl_query_src_md);

    if (p->dir & FLAG_WEI)
        cd.diff_weights_desc = q(dnnl_query_diff_weights_md);
    else
        cd.weights_desc = q(dnnl_query_weights_md);

    if (p->dir & FLAG_BIA) {
        if (p->dir & FLAG_BWD)
            cd.diff_bias_desc = q(dnnl_query_diff_weights_md, 1);
        else
            cd.bias_desc = q(dnnl_query_weights_md, 1);
    }

    if (p->dir & FLAG_BWD)
        cd.diff_dst_desc = q(dnnl_query_diff_dst_md);
    else
        cd.dst_desc = q(dnnl_query_dst_md);

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    res_t res_zero {};
    *r = res_zero;
    bool with_groups = 1;

    prb_t p_tr((desc_t)*p, p->dir, p->cfg, p->stag, p->wtag, p->dtag, p->alg,
            p->attr, p->mb, true);
    swap(p_tr.ic, p_tr.oc);
    swap(p_tr.ih, p_tr.oh);
    swap(p_tr.id, p_tr.od);
    swap(p_tr.iw, p_tr.ow);

    dnnl_deconvolution_desc_t cd;
    dnnl_primitive_desc_t dpd;
    dnnl_primitive_t c {};

    SAFE(init_pd(p, cd, dpd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    DNN_SAFE(dnnl_primitive_create(&c, dpd), WARN);
    DNN_SAFE_V(dnnl_primitive_desc_destroy(dpd));

    auto &src_dt_d = p->dir == BWD_D ? cd.diff_src_desc : cd.src_desc;
    auto &wei_dt_d = p->dir & FLAG_WEI ? cd.diff_weights_desc : cd.weights_desc;
    auto &bia_dt_d = p->dir & FLAG_BWD ? cd.diff_bias_desc : cd.bias_desc;
    auto &dst_dt_d = p->dir & FLAG_BWD ? cd.diff_dst_desc : cd.dst_desc;
    auto wei_tr_dt_d = wei_dt_d;
    swap(wei_tr_dt_d.dims[with_groups + 0], wei_tr_dt_d.dims[with_groups + 1]);

    dnn_mem_t src_dt(src_dt_d, p->cfg[SRC].dt, engine_tgt);
    dnn_mem_t wei_dt(wei_dt_d, p->cfg[WEI].dt, engine_tgt);
    dnn_mem_t dst_dt(dst_dt_d, p->cfg[DST].dt, engine_tgt);
    dnn_mem_t bia_dt;
    if (p->dir & FLAG_BIA)
        bia_dt = dnn_mem_t(bia_dt_d, p->cfg[BIA].dt, engine_tgt);

    auto src_tag = get_default_tag(src_dt.md_.ndims);
    auto wei_tag = get_default_tag(wei_dt.md_.ndims);

    const auto fp = dnnl_f32;

    /* memory for ref */
    dnn_mem_t src_fp(src_dt_d, fp, src_tag, engine_tgt);
    dnn_mem_t wei_fp(wei_dt_d, fp, wei_tag, engine_tgt);
    dnn_mem_t dst_fp(dst_dt_d, fp, src_tag, engine_tgt);
    dnn_mem_t wei_tr_fp(wei_tr_dt_d, fp, wei_tag, engine_tgt);
    dnn_mem_t bia_fp;
    if (p->dir & FLAG_BIA) bia_fp = dnn_mem_t(bia_dt_d, fp, dnnl_x, engine_tgt);
    dnn_mem_t zero_fp;

    /* fill memory + reorders <-> */
    SAFE(fill_dst(p, dst_dt, dst_fp, r), WARN);
    SAFE(fill_wei(p, wei_dt, wei_fp, r), WARN);
    SAFE(fill_src(p, src_dt, src_fp, r), WARN);

    SAFE(transpose_data_wei(p, wei_fp, wei_tr_fp), WARN);
    if (p->dir & FLAG_BIA) SAFE(fill_bia(p, bia_dt, bia_fp, r), WARN);

    args_t args;

    if (p->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        if (p->dir & FLAG_BIA) args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);

        DNN_SAFE(execute_and_wait(c, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_d(
                    &p_tr, nullptr, dst_fp, wei_tr_fp, bia_fp, src_fp);
            dnn_mem_t dst(dst_dt, fp, src_tag, engine_tgt);
            SAFE(compare_dst(p, dst, dst_fp, r, true), WARN);
        }
    } else if (p->dir == BWD_D) {
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);

        DNN_SAFE(execute_and_wait(c, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(&p_tr, nullptr, dst_fp, wei_tr_fp, zero_fp, src_fp);
            dnn_mem_t src(src_dt, fp, src_tag, engine_tgt);
            SAFE(compare_src(p, src, src_fp, r, true), WARN);
        }
    } else if (p->dir & FLAG_BWD && p->dir & FLAG_WEI) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
        if (p->dir & FLAG_BIA) args.set(DNNL_ARG_DIFF_BIAS, bia_dt);

        DNN_SAFE(execute_and_wait(c, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_weights(&p_tr, dst_fp, wei_tr_fp, src_fp);
            transpose_data_wei(&p_tr, wei_tr_fp, wei_fp);
            dnn_mem_t wei(wei_dt, fp, wei_tag, engine_tgt);
            SAFE(compare_wei(&p_tr, wei, wei_fp, r, true), WARN);
            if (p->dir & FLAG_BIA) {
                compute_ref_bwd_bias(p, bia_fp, dst_fp);
                dnn_mem_t bia(bia_dt, fp, dnnl_x, engine_tgt);
                SAFE(compare_bia(p, bia, bia_fp, r, true), WARN);
            }
        }
    } else {
        SAFE(FAIL, CRIT);
    }

    measure_perf(r->timer, c, args);

    DNN_SAFE_V(dnnl_primitive_destroy(c));

    return OK;
}

} // namespace deconv
