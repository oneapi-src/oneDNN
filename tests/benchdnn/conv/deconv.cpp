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

#include "tests/test_thread.hpp"

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

static int init_pd(dnnl_engine_t engine, const prb_t *p,
        dnnl_primitive_desc_t &dpd, res_t *r, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_deconvolution_desc_t cd;
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
                     p->cfg[SRC].dt, convert_tag(p->stag, p->ndims)),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&wei_d, p->ndims + p->has_groups,
                     p->ndims == 5
                             ? &wei_3d_dims[!p->has_groups]
                             : p->ndims == 3 ? &wei_1d_dims[!p->has_groups]
                                             : &wei_2d_dims[!p->has_groups],
                     p->cfg[WEI].dt, convert_tag(p->wtag, p->ndims)),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &bia_d, 1, bia_dims, p->cfg[BIA].dt, dnnl_format_tag_any),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_d, p->ndims,
                     p->ndims == 5 ? dst_3d_dims
                                   : p->ndims == 3 ? dst_1d_dims : dst_2d_dims,
                     p->cfg[DST].dt, convert_tag(p->dtag, p->ndims)),
            WARN);

    dnnl_dim_t strides_nd[] = {p->sd, p->sh, p->sw};
    dnnl_dim_t dilates_nd[] = {p->dd, p->dh, p->dw};
    dnnl_dim_t padding_nd[] = {p->pd, p->ph, p->pw};
    dnnl_dim_t padding_r_nd[] = {p->pd_r, p->ph_r, p->pw_r};

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

    attr_args_t attr_args;
    attr_args.prepare_output_scales(p->attr, p->scales, p->oc);
    auto dnnl_attr = create_dnnl_attr(p->attr, attr_args);

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&dpd, &cd, dnnl_attr, engine, NULL);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented) {
        return r->state = UNIMPLEMENTED, OK;
    }
    SAFE(init_status, WARN);

    r->impl_name = query_impl_info(dpd);
    if (maybe_skip(r->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                r->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(dpd), WARN);
        return r->state = SKIPPED, r->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", r->impl_name.c_str());
    }

    return OK;
}

void check_known_skipped_case(const prb_t *p, res_t *r) {
    check_known_skipped_case_common(
            {p->cfg[SRC].dt, p->cfg[WEI].dt, p->cfg[DST].dt}, p->dir, r);
    if (r->state == SKIPPED) return;

    // TODO: shapes with dilation and non-unit stride go to reference which
    // does not support attributes yet. Remove the whole condition once
    // attributes support is added to reference.
    // TODO: uncomment deconv shape mb96ic64ih6oc32oh14kh5ph2dh2n"4d/5x5"
    static auto isa = dnnl_get_effective_cpu_isa();
    static bool has_avx512_bw = isa >= dnnl_cpu_isa_avx512_core;
    bool is_int8 = p->cfg[WEI].dt == dnnl_s8;
    bool has_attr_support = IMPLICATION(is_int8, has_avx512_bw)
            && IMPLICATION(p->dd != 0, p->sd == 1)
            && IMPLICATION(p->dh != 0, p->sh == 1)
            && IMPLICATION(p->dw != 0, p->sw == 1);
    if (!p->attr.is_def() && !has_attr_support) {
        r->state = SKIPPED, r->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    check_known_skipped_case(p, r);
    if (r->state == SKIPPED) return OK;

    prb_t p_tr((desc_t)*p, p->dir, p->cfg, p->stag, p->wtag, p->dtag, p->alg,
            p->attr, p->mb, true);
    swap(p_tr.ic, p_tr.oc);
    swap(p_tr.ih, p_tr.oh);
    swap(p_tr.id, p_tr.od);
    swap(p_tr.iw, p_tr.ow);

    dnnl_primitive_t d {};
    SAFE(init_prim(&d, init_pd, p, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(d, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(d));
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
    auto wei_tr_md = wei_md;

    const bool with_groups = 1;
    swap(wei_tr_md.dims[with_groups + 0], wei_tr_md.dims[with_groups + 1]);

    const auto fp = dnnl_f32;
    const auto src_tag = get_abx_tag(src_md.ndims);
    const auto wei_tag = get_abx_tag(wei_md.ndims);

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t bia_dt(bia_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    dnn_mem_t src_fp(src_md, fp, src_tag, test_engine);
    dnn_mem_t wei_fp(wei_md, fp, wei_tag, test_engine);
    dnn_mem_t dst_fp(dst_md, fp, src_tag, test_engine);
    dnn_mem_t wei_tr_fp(wei_tr_md, fp, wei_tag, test_engine);
    dnn_mem_t bia_fp(bia_md, fp, dnnl_x, test_engine);

    /* fill memory + reorders <-> */
    SAFE(fill_dst(p, dst_dt, dst_fp, r), WARN);
    SAFE(fill_wei(p, wei_dt, wei_fp, r), WARN);
    SAFE(fill_src(p, src_dt, src_fp, r), WARN);
    SAFE(transpose_data_wei(p, wei_fp, wei_tr_fp), WARN);
    SAFE(fill_bia(p, bia_dt, bia_fp, r), WARN);

    args_t args;

    if (p->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(d, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_d(
                    &p_tr, nullptr, dst_fp, wei_tr_fp, bia_fp, src_fp);
            dnn_mem_t dst(dst_dt, fp, src_tag, test_engine);
            SAFE(compare_dst(p, dst, dst_fp, r, true), WARN);
        }
    } else if (p->dir == BWD_D) {
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(d, args), WARN);

        if (bench_mode & CORR) {
            dnn_mem_t zero_fp;
            compute_ref_fwd(&p_tr, nullptr, dst_fp, wei_tr_fp, zero_fp, src_fp);
            dnn_mem_t src(src_dt, fp, src_tag, test_engine);
            SAFE(compare_src(p, src, src_fp, r, true), WARN);
        }
    } else if (p->dir & FLAG_BWD && p->dir & FLAG_WEI) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_BIAS, bia_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(d, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_weights(&p_tr, dst_fp, wei_tr_fp, src_fp);
            transpose_data_wei(&p_tr, wei_tr_fp, wei_fp);
            dnn_mem_t wei(wei_dt, fp, wei_tag, test_engine);
            SAFE(compare_wei(&p_tr, wei, wei_fp, r, true), WARN);
            if (p->dir & FLAG_BIA) {
                compute_ref_bwd_bias(p, bia_fp, dst_fp);
                dnn_mem_t bia(bia_dt, fp, dnnl_x, test_engine);
                SAFE(compare_bia(p, bia, bia_fp, r, true), WARN);
            }
        }
    } else {
        SAFE(FAIL, CRIT);
    }

    measure_perf(r->timer, d, args);

    DNN_SAFE_V(dnnl_primitive_destroy(d));

    return OK;
}

} // namespace deconv
