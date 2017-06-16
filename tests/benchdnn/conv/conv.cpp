/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "conv/conv.hpp"

namespace conv {

double get_trust_nz_level(const prb_t *p, int what, bool final_compare) {
    if (!final_compare)
        return p->cfg[what].f_sparsity;

    double trust = 0.3; /* why? */
    switch (what) {
        case SRC:
            trust /= p->sh * p->sw;
            break;
        case WEI:
            trust /= 1. * p->kh * p->kw
                / MIN3(p->kh * p->kw, p->ih * p->iw, p->oh * p->ow);
            break;
        case BIA:
            trust = 0.8 * p->cfg[DST].f_sparsity; /* why? */
            break;
        case DST:
            trust /= p->merge == RELU ? 2. : 1.;
            break;
    }

    return trust;
}

inline int compare_dat(const prb_t *p, int what, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare = false) {
    int nelems = mem_dt.nelems();

    const char *swhat = inp_type2str(what);

    int in = 0, below = 0, above = 0;
    int in_ok = 0, below_ok = 0, above_ok = 0;
    int non_zero = 0;

    float max_rel_diff = 0;

    r->errors = 0;
    r->total = nelems;

    for (int i = 0; i < nelems; ++i) {
        const float dt = ((float*)mem_dt)[i];
        const float fp = ((float*)mem_fp)[i];
        const float diff = fabs(fp - dt);
        const float rel_diff = diff / (fabs(fp) > FLT_MIN ? fabs(fp) : 1);
        if (max_rel_diff < rel_diff) max_rel_diff = rel_diff;

        bool ok = true;
        if (fp < p->cfg[what].min) {
            ok = dt == p->cfg[what].min;
            below += 1;
            below_ok += ok;
        } else if (fp > p->cfg[what].max) {
            ok = dt == p->cfg[what].max;
            above += 1;
            above_ok += ok;
        } else {
            ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= p->cfg[what].eps;
            in += 1;
            in_ok += ok;
        }
        if (!ok) {
            r->errors++;
            if (r->errors < 10 || verbose >= 10) {
                int mb_or_g = 0, g_or_oc = 0, c = 0, h = 0, w = 0;
                switch (what) {
                case SRC: inv_src_off_f(p, i, mb_or_g, g_or_oc, c, h, w); break;
                case WEI: inv_wei_off_f(p, i, mb_or_g, g_or_oc, c, h, w); break;
                case BIA: inv_bia_off_f(p, i, mb_or_g, g_or_oc); break;
                case DST: inv_dst_off_f(p, i, mb_or_g, g_or_oc, c, h, w); break;
                }
                print(0, "[%4d][%s][%d,%d,%d,%d,%d] "
                        "fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                        i, swhat, mb_or_g, g_or_oc, c, h, w,
                        fp, dt, diff, rel_diff);
            }
        }

        /* for debug purposes only: dump the output */
        if (final_compare && verbose >= 50 && i < 30) {
            int mb_or_g = 0, g_or_oc = 0, c = 0, h = 0, w = 0;
            switch (what) {
            case SRC: inv_src_off_f(p, i, mb_or_g, g_or_oc, c, h, w); break;
            case WEI: inv_wei_off_f(p, i, mb_or_g, g_or_oc, c, h, w); break;
            case BIA: inv_bia_off_f(p, i, mb_or_g, g_or_oc); break;
            case DST: inv_dst_off_f(p, i, mb_or_g, g_or_oc, c, h, w); break;
            }

            print(0, "[%4d][%s][%d,%d,%d,%d,%d] fp:%8g dt:%8g\n",
                    i, swhat, mb_or_g, g_or_oc, c, h, w, fp, dt);
        }

        non_zero += fp != 0;
    }

    if (final_compare || r->errors)
        print(2, "[%s] max_rel_diff:%g\n", swhat, max_rel_diff);

    const double trust_rg_level = 0.3;
    const double trust_nz_level = get_trust_nz_level(p, what, final_compare);

    const double trust_rg = (double)in / r->total;
    const double trust_nz = (double)non_zero / r->total;

    const bool no_trust = true /* ...in the test ...at all */
        && final_compare
        && (trust_rg < trust_rg_level || trust_nz < trust_nz_level);

    const bool dump = verbose >= 20
        || (verbose >= 10 && (trust_rg < 1. || trust_nz < 1.));
    if (dump)
        print(0, "@@@ [%s] %strust range:%.2f nz:%.2f "
                "(level range:%.2f nz:%.2f). "
                "in:%d (ok:%d) below:%d (ok:%d) above:%d (ok:%d) nz:%d "
                "total:%d\n", swhat, final_compare ? "final: " : "",
                trust_rg, trust_nz, trust_rg_level, trust_nz_level, in, in_ok,
                below, below_ok, above, above_ok, non_zero, r->total);

    if (no_trust) {
        r->state = MISTRUSTED;
        print(0, "@@@ [%s] test-bug: trust is too low. "
                "range:%.2f (?<%.2f) nz:%.2f (?<%.2f) (nz: %d total: %d)\n",
                swhat, trust_rg, trust_rg_level, trust_nz, trust_nz_level,
                non_zero, r->total);
    }

    if (r->errors)
        r->state = FAILED;

    if (final_compare && r->state == UNTESTED)
        r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

inline int compare_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false)
{ return compare_dat(p, SRC, mem_dt, mem_fp, r, final_compare); }
inline int compare_wei(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false)
{ return compare_dat(p, WEI, mem_dt, mem_fp, r, final_compare); }
inline int compare_bia(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false)
{ return compare_dat(p, BIA, mem_dt, mem_fp, r, final_compare); }
inline int compare_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false)
{ return compare_dat(p, DST, mem_dt, mem_fp, r, final_compare); }

inline int fill_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r) {
    const bool extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t *p_mem_00 = extra_mem
        ? new dnn_mem_t(mem_dt.md_, mkldnn_f32, mkldnn_nchw)
        : &mem_fp;
    dnn_mem_t &mem_00 = *p_mem_00;

    const auto &c = p->cfg[SRC];
    const int range = c.f_max - c.f_min + 1;

#   pragma omp parallel for collapse(4)
    for (int mb = 0; mb < p->mb; ++mb)
    for (int ic = 0; ic < p->ic; ++ic)
    for (int ih = 0; ih < p->ih; ++ih)
    for (int iw = 0; iw < p->iw; ++iw)
    {
        const int gen = 17 * ih + 13 * iw + 13 * mb + 19 * ic + 1637;
        const bool non_base = true
            && gen % (p->kh * p->kw) <= c.f_sparsity * (p->kh * p->kw);
//            && (17 * ih + 13 * mb) % p->kh == 0
//            && (13 * iw + 19 * ic) % p->kw == 0;
        const float value =
            non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float*)mem_00)[src_off_f(p, mb, 0, ic, ih, iw)] = value;
    }

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_src(p, mem_fp, mem_00, r), WARN);
        delete &mem_00;
    }

    return OK;
}

inline int fill_wei(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
    res_t *r) {
    const bool extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t *p_mem_00 = extra_mem
        ? new dnn_mem_t(mem_dt.md_, mkldnn_f32, mkldnn_goihw)
        : &mem_fp;
    dnn_mem_t &mem_00 = *p_mem_00;

    const auto &c = p->cfg[WEI];
    const int range = c.f_max - c.f_min + 1;

#   pragma omp parallel for collapse(5)
    for (int g = 0; g < p->g; ++g)
    for (int oc = 0; oc < p->oc / p->g; ++oc)
    for (int ic = 0; ic < p->ic / p->g; ++ic)
    for (int kh = 0; kh < p->kh; ++kh)
    for (int kw = 0; kw < p->kw; ++kw)
    {
        const int gen = 17 * kh + 13 * kw + 13 * oc + 19 * ic + 38;
        const bool non_base = true
            && gen % (p->kh * p->kw) <= c.f_sparsity * (p->kh * p->kw);
//            && (17 * kh + 13 * oc) % p->kh == 0
//            && (13 * kw + 19 * ic) % p->kw == 0;
        const float value =
            non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float*)mem_00)[wei_off_f(p, g, oc, ic, kh, kw)] = value;
    }

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_wei(p, mem_fp, mem_00, r), WARN);
        delete &mem_00;
    }

    return OK;
}

inline int fill_bia(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r) {
    const bool extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t *p_mem_00 = extra_mem
        ? new dnn_mem_t(mem_dt.md_, mkldnn_f32, mkldnn_x)
        : &mem_fp;
    dnn_mem_t &mem_00 = *p_mem_00;

    const auto &c = p->cfg[BIA];
    const int range = c.f_max - c.f_min + 1;

    const int sz = mem_00.nelems();
    for (int i = 0; i < sz; ++i) {
        const int gen = 19 * i;
        const bool non_base = true
            && gen % (p->kh * p->kw) <= c.f_sparsity * (p->kh * p->kw);
        const float value =
            non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float*)mem_00)[i] = value;
    }

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_bia(p, mem_fp, mem_00, r), WARN);
        delete &mem_00;
    }

    return OK;
}

inline int fill_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r) {
    const bool extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t *p_mem_00 = extra_mem
        ? new dnn_mem_t(mem_dt.md_, mkldnn_f32, mkldnn_nchw)
        : &mem_fp;
    dnn_mem_t &mem_00 = *p_mem_00;

    const auto &c = p->cfg[DST];
    const int range = c.f_max - c.f_min + 1;

#   pragma omp parallel for collapse(4)
    for (int mb = 0; mb < p->mb; ++mb)
    for (int oc = 0; oc < p->oc; ++oc)
    for (int oh = 0; oh < p->oh; ++oh)
    for (int ow = 0; ow < p->ow; ++ow)
    {
        const int gen = 19 * oh + 17 * ow + 13 * mb + 13 * oc + 223;
        const bool non_base = true
            && gen % (p->kh * p->kw) <= c.f_sparsity * (p->kh * p->kw);
//            && (19 * oh + 13 * mb) % p->kh == 0
//            && (17 * ow + 13 * oc) % p->kw == 0;
        const float value =
            non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float*)mem_00)[dst_off_f(p, mb, 0, oc, oh, ow)] = value;
    }

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_dst(p, mem_fp, mem_00, r), WARN);
        delete &mem_00;
    }

    return OK;
}

inline int init_pd(const prb_t *p, mkldnn_convolution_desc_t &cd,
        mkldnn_primitive_desc_t &cpd, res_t *r) {
    mkldnn_memory_desc_t src_d, wei_d, bia_d, dst_d;

    mkldnn_dims_t src_dims = {p->mb, p->ic, p->ih, p->iw};
    mkldnn_dims_t wei_dims = {p->g, p->oc / p->g, p->ic / p->g, p->kh, p->kw};
    mkldnn_dims_t bia_dims = {p->oc};
    mkldnn_dims_t dst_dims = {p->mb, p->oc, p->oh, p->ow};

    DNN_SAFE(mkldnn_memory_desc_init(&src_d, 4, src_dims, p->cfg[SRC].dt, mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&wei_d, 5, wei_dims, p->cfg[WEI].dt, mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&bia_d, 1, bia_dims, p->cfg[BIA].dt, mkldnn_any), WARN);
    DNN_SAFE(mkldnn_memory_desc_init(&dst_d, 4, dst_dims, p->cfg[DST].dt, mkldnn_any), WARN);

    int strides[] = {p->sh, p->sw};
    int dilates[] = {p->dh, p->dw};
    int padding[] = {p->ph, p->pw};
    auto bph = [&](int ih, int oh, int kh, int sh, int ph, int dh) {
        return (oh - 1) * sh - ih + ((kh - 1) * (dh + 1) + 1) - ph;
    };
    int padding_r[] = {
        bph(p->ih, p->oh, p->kh, p->sh, p->ph, p->dh),
        bph(p->iw, p->ow, p->kw, p->sw, p->pw, p->dw)};

    mkldnn_alg_kind_t alg = mkldnn_convolution_direct;
    if (p->alg == WINO) alg = mkldnn_convolution_winograd;

    switch (p->dir) {
    case FWD_D: case FWD_B:
        DNN_SAFE(mkldnn_dilated_convolution_forward_desc_init(&cd,
                    mkldnn_forward_inference, alg, &src_d, &wei_d,
                    p->dir == FWD_D ? NULL : &bia_d, &dst_d, strides, dilates,
                    padding, padding_r, mkldnn_padding_zero), WARN);
        break;
    case BWD_D:
        DNN_SAFE(mkldnn_convolution_backward_data_desc_init(&cd, alg, &src_d,
                    &wei_d, &dst_d, strides, padding, padding_r,
                    mkldnn_padding_zero), WARN);
        break;
    case BWD_W: case BWD_WB:
        DNN_SAFE(mkldnn_convolution_backward_weights_desc_init(&cd, alg,
                    &src_d, &wei_d, p->dir == BWD_W ? NULL : &bia_d, &dst_d,
                    strides, padding, padding_r, mkldnn_padding_zero), WARN);
        break;
    default: DNN_SAFE(mkldnn_invalid_arguments, CRIT);
    }

    DNN_SAFE(cd.accum_data_type == p->cfg[ACC].dt
            ? mkldnn_success : mkldnn_unimplemented, CRIT);

    mkldnn_status_t init_status = mkldnn_success;
    if (p->merge == RELU) {
        mkldnn_convolution_relu_desc_t crd;
        DNN_SAFE(mkldnn_convolution_relu_desc_init(&crd, &cd, 0), WARN);
        init_status = mkldnn_primitive_desc_create(&cpd, &crd, engine, NULL);
    } else {
        init_status = mkldnn_primitive_desc_create(&cpd, &cd, engine, NULL);
    }

    if (init_status == mkldnn_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(cpd);
    if (maybe_skip(impl_str)) {
        print(2, "SKIPPED: mkldnn implementation: %s\n", impl_str);
        DNN_SAFE(mkldnn_primitive_desc_destroy(cpd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(50, "mkldnn implementation: %s\n", impl_str);
    }

    auto q = [=](mkldnn_query_t query, int index = 0) {
        return *mkldnn_primitive_desc_query_memory_d(
                mkldnn_primitive_desc_query_pd(cpd, query, index));
    };

    if (p->dir == BWD_D)
        cd.diff_src_desc = q(mkldnn_query_diff_src_pd);
    else
        cd.src_desc = q(mkldnn_query_src_pd);

    if (p->dir & FLAG_WEI)
        cd.diff_weights_desc = q(mkldnn_query_diff_weights_pd);
    else
        cd.weights_desc = q(mkldnn_query_weights_pd);

    if (p->dir & FLAG_BIA) {
        if (p->dir & FLAG_BWD)
            cd.diff_bias_desc = q(mkldnn_query_diff_weights_pd, 1);
        else
            cd.bias_desc = q(mkldnn_query_weights_pd, 1);
    }

    if (p->dir & FLAG_BWD)
        cd.diff_dst_desc = q(mkldnn_query_diff_dst_pd);
    else
        cd.dst_desc = q(mkldnn_query_dst_pd);

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    res_t res_zero{};
    *r = res_zero;

    mkldnn_convolution_desc_t cd;
    mkldnn_primitive_desc_t cpd;
    mkldnn_primitive_t c;

    SAFE(init_pd(p, cd, cpd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED)
        return OK;

    auto &src_dt_d = p->dir == BWD_D ? cd.diff_src_desc : cd.src_desc;
    auto &wei_dt_d = p->dir & FLAG_WEI ? cd.diff_weights_desc : cd.weights_desc;
    auto &bia_dt_d = p->dir & FLAG_BWD ? cd.diff_bias_desc : cd.bias_desc;
    auto &dst_dt_d = p->dir & FLAG_BWD ? cd.diff_dst_desc: cd.dst_desc;

    dnn_mem_t src_dt(src_dt_d, p->cfg[SRC].dt);
    dnn_mem_t wei_dt(wei_dt_d, p->cfg[WEI].dt);
    dnn_mem_t dst_dt(dst_dt_d, p->cfg[DST].dt);
    dnn_mem_t bia_dt = p->dir & FLAG_BIA
        ? dnn_mem_t(bia_dt_d, p->cfg[BIA].dt) : dnn_mem_t();

    const auto fp = mkldnn_f32;
    dnn_mem_t src_fp(src_dt_d, fp, mkldnn_nchw);
    dnn_mem_t wei_fp(wei_dt_d, fp, mkldnn_goihw);
    dnn_mem_t dst_fp(dst_dt_d, fp, mkldnn_nchw);
    dnn_mem_t bia_fp = p->dir & FLAG_BIA
        ? dnn_mem_t(bia_dt_d, fp, mkldnn_x) : dnn_mem_t();

    SAFE(fill_src(p, src_dt, src_fp, r), WARN);
    SAFE(fill_wei(p, wei_dt, wei_fp, r), WARN);
    SAFE(fill_dst(p, dst_dt, dst_fp, r), WARN);
    if (p->dir & FLAG_BIA)
        SAFE(fill_bia(p, bia_dt, bia_fp, r), WARN);

    if (p->dir & FLAG_FWD) {
        compute_ref_fwd(p, src_fp, wei_fp, bia_fp, dst_fp);
        mkldnn_primitive_at_t inputs[3] = { {src_dt.p_, 0}, {wei_dt.p_, 0},
            {p->dir & FLAG_BIA ? bia_dt.p_ : NULL, 0}
        };
        const_mkldnn_primitive_t outputs[] = { dst_dt.p_ };
        DNN_SAFE(mkldnn_primitive_create(&c, cpd, inputs, outputs), WARN);
        SAFE(execute(c), WARN);
        dnn_mem_t dst(dst_dt, fp, mkldnn_nchw);
        SAFE(dst.reorder(dst_dt), WARN);
        SAFE(compare_dst(p, dst, dst_fp, r, true), WARN);
    } else if (p->dir == BWD_D) {
        compute_ref_bwd_d(p, src_fp, wei_fp, dst_fp);
        mkldnn_primitive_at_t inputs[3] = { {dst_dt.p_, 0}, {wei_dt.p_, 0}, };
        const_mkldnn_primitive_t outputs[] = { src_dt.p_ };
        DNN_SAFE(mkldnn_primitive_create(&c, cpd, inputs, outputs), WARN);
        SAFE(execute(c), WARN);
        dnn_mem_t src(src_dt, fp, mkldnn_nchw);
        SAFE(src.reorder(src_dt), WARN);
        SAFE(compare_src(p, src, src_fp, r, true), WARN);
    } else if (p->dir & FLAG_BWD && p->dir & FLAG_WEI) {
        compute_ref_bwd_w(p, src_fp, wei_fp, bia_fp, dst_fp);
        mkldnn_primitive_at_t inputs[3] = { {src_dt.p_, 0}, {dst_dt.p_, 0}, };
        const_mkldnn_primitive_t outputs[] = { wei_dt.p_,
            p->dir & FLAG_BIA ? bia_dt.p_ : NULL,
        };
        DNN_SAFE(mkldnn_primitive_create(&c, cpd, inputs, outputs), WARN);
        SAFE(execute(c), WARN);
        dnn_mem_t wei(wei_dt, fp, mkldnn_goihw);
        SAFE(wei.reorder(wei_dt), WARN);
        SAFE(compare_wei(p, wei, wei_fp, r, true), WARN);
        if (p->dir & FLAG_BIA) {
            dnn_mem_t bia(bia_dt, fp, mkldnn_x);
            SAFE(bia.reorder(bia_dt), WARN);
            SAFE(compare_bia(p, bia, bia_fp, r, true), WARN);
        }
    }

    return OK;
}

}
