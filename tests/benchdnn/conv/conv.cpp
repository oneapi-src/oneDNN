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
#include <stdio.h>
#include <stdlib.h>

#include "dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

#include "binary/binary.hpp"
#include "conv/conv_common.hpp"
#include "eltwise/eltwise.hpp"

namespace conv {

double get_trust_nz_level(
        const prb_t *p, data_kind_t kind, bool final_compare) {
    if (!final_compare) return p->cfg[kind].f_sparsity;

    auto negative_to_zero = [&]() {
        using pk = attr_t::post_ops_t::kind_t;
        const auto &po = p->attr.post_ops;
        int count = 0;
        for (int i = 0; i < po.len(); ++i) {
            auto k = po.entry[i].kind;
            count += k == pk::RELU || k == pk::ELU || k == pk::SQRT
                    || k == pk::BRELU;
        }
        return !!count;
    };

    double trust = 0.3; /* why? */
    switch (kind) {
        case SRC: trust /= p->sd * p->sh * p->sw; break;
        case WEI:
            trust /= 1. * p->kd * p->kh * p->kw
                    / MIN3(p->kd * p->kh * p->kw, p->id * p->ih * p->iw,
                            p->od * p->oh * p->ow);
            break;
        case BIA:
            trust = 0.8 * p->cfg[DST].f_sparsity; /* why? */
            break;
        case DST: trust /= negative_to_zero() == 0 ? 1 : 2; break;
    }

    return trust;
}

inline bool post_ops_require_integral_check(const prb_t *p) {
    const auto &po = p->attr.post_ops;
    if (po.len() == 0) return false;

    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];
        using pk_t = attr_t::post_ops_t::kind_t;

        if (e.kind == pk_t::SUM || e.kind == pk_t::ABS) continue;
        if (e.kind == pk_t::RELU && e.eltwise.alpha == 0.f) continue;
        return true;
    }

    return false;
}

inline double get_eps(const prb_t *p, const data_kind_t kind) {
    // Winograd specifics
    if (p->alg & WINO && p->dir & FLAG_WEI) {
        /*This is an empirical equation derived by observing growth error
          with increasing 'k' dimension in gemm of winograd*/
        return p->cfg[kind].eps
                * (MAX2(1,
                        pow(10, 0.4 * log10(0.125 * p->mb * p->oh * p->ow))));
    }

    // post-ops specifics
    if (post_ops_require_integral_check(p)) return MAX2(1e-5, p->cfg[kind].eps);

    return p->cfg[kind].eps;
}

inline void get_result(const prb_t *p, const data_kind_t kind, res_t *r,
        const diff_norm_t diff_norm) {
    const float eps = get_eps(p, kind);

    /* Ignoring element-wise errors for Winograd and in some cases of post-ops,
     * since large relative error in few elements (which are anyways close
     * to zero) results in false positive failures */

    bool wino_test = (p->alg & WINO) && diff_norm.rel_diff(norm_t::L2) <= eps;
    if (wino_test) r->errors = 0;

    bool post_ops_test = post_ops_require_integral_check(p)
            && diff_norm.rel_diff(norm_t::L2) <= eps;
    if (post_ops_test) r->errors = 0;

    if (r->errors) r->state = FAILED;
}

inline int compare_dat(const prb_t *p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare = false) {

    const bool dont_complain
            = false || (p->alg & WINO) || post_ops_require_integral_check(p);

    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return r->state = PASSED, OK;
    r->total = nelems;

    const char *skind = data_kind2str(kind);

    int in = 0, below = 0, above = 0;
    int in_ok = 0, below_ok = 0, above_ok = 0;
    int non_zero = 0;

    const int eltwise_idx = p->attr.post_ops.eltwise_index();
    const bool has_eltwise = eltwise_idx >= 0;

    int sum_ind = p->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM);
    auto sum_dt = (sum_ind != -1) ? p->attr.post_ops.entry[sum_ind].sum.dt
                                  : dnnl_data_type_undef;

    bool diff_sum_dt = kind == DST && !final_compare
            && sum_dt != dnnl_data_type_undef && sum_dt != p->cfg[kind].dt;
    dnnl_data_type_t f_dt = diff_sum_dt ? sum_dt : p->cfg[kind].dt;
    float f_min = diff_sum_dt ? lowest_dt(f_dt) : p->cfg[kind].min;
    float f_max = diff_sum_dt ? max_dt(f_dt) : p->cfg[kind].max;

    diff_norm_t diff_norm;

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = round_to_nearest_representable(f_dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);

        bool ok = true;
        if (fp < f_min) {
            diff_norm.update(f_min, dt);
            ok = dt == f_min;
            if (!ok && has_eltwise)
                ok = eltwise::check_extreme_values(
                        fp, dt, p->attr.post_ops.entry[eltwise_idx].kind);
            below += 1;
            below_ok += ok;
        } else if (fp > f_max) {
            diff_norm.update(f_max, dt);
            ok = dt == f_max;
            if (!ok && has_eltwise)
                ok = eltwise::check_extreme_values(
                        fp, dt, p->attr.post_ops.entry[eltwise_idx].kind);
            above += 1;
            above_ok += ok;
        } else {
            diff_norm.update(fp, dt);
            ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= get_eps(p, kind);
            if (!ok && has_eltwise)
                ok = eltwise::check_extreme_values(
                        fp, dt, p->attr.post_ops.entry[eltwise_idx].kind);
            in += 1;
            in_ok += ok;
        }

        r->errors += !ok;

        bool dump
                = (!ok && ((!dont_complain && r->errors < 10) || verbose >= 10))
                || (final_compare
                        && ((verbose >= 50 && i < 30) || (verbose >= 99)));

        if (dump) {
            int64_t mb_or_g = 0, g_or_oc = 0, c = 0, d = 0, h = 0, w = 0;
            switch (kind) {
                case SRC:
                    inv_src_off_f(p, i, mb_or_g, g_or_oc, c, d, h, w);
                    break;
                case WEI:
                    inv_wei_off_f(p, i, mb_or_g, g_or_oc, c, d, h, w);
                    break;
                case BIA: inv_bia_off_f(p, i, mb_or_g, g_or_oc); break;
                case DST:
                    inv_dst_off_f(p, i, mb_or_g, g_or_oc, c, d, h, w);
                    break;
            }
            BENCHDNN_PRINT(0,
                    "[%4ld][%s%s]"
                    "[" IFMT "," IFMT "," IFMT "," IFMT "," IFMT "," IFMT
                    "] "
                    "fp:% 12.6g fp0:% 12.6g dt:% 12.6g diff:%8g rdiff:%8g\n",
                    (long)i, final_compare ? "" : "REORDER ", skind, mb_or_g,
                    g_or_oc, c, d, h, w, fp, fp0, dt, diff, rel_diff);
        }

        non_zero += fp != 0;
    }

    diff_norm.done();
    get_result(p, kind, r, diff_norm);

    if (final_compare || r->errors) {
        const int vl = r->errors ? 0 : 2;
        BENCHDNN_PRINT(vl,
                "@@@ [%s] %sdiff: err:%d, l0(``%g``) "
                "l1:(%g,%g,%g,``%g``) "
                "l2:(%g,%g,%g,``%g``) "
                "l8:(%g,%g,%g,``%g``)\n",
                skind, final_compare ? "final: " : "", (int)r->errors,
                diff_norm.rel_diff(norm_t::L0), diff_norm.a_[norm_t::L1],
                diff_norm.b_[norm_t::L1], diff_norm.diff_[norm_t::L1],
                diff_norm.rel_diff(norm_t::L1), diff_norm.a_[norm_t::L2],
                diff_norm.b_[norm_t::L2], diff_norm.diff_[norm_t::L2],
                diff_norm.rel_diff(norm_t::L2), diff_norm.a_[norm_t::L8],
                diff_norm.b_[norm_t::L8], diff_norm.diff_[norm_t::L8],
                diff_norm.rel_diff(norm_t::L8));
    }

    const double trust_rg_level = 0.3;
    const double trust_nz_level = get_trust_nz_level(p, kind, final_compare);

    const double trust_rg = (double)in / r->total;
    const double trust_nz = (double)non_zero / r->total;

    const bool no_trust = true /* ...in the test ...at all */
            && final_compare
            && (trust_rg < trust_rg_level || trust_nz < trust_nz_level);

    const bool dump = verbose >= 20
            || (verbose >= 10 && (trust_rg < 1. || trust_nz < 1.));
    if (dump) {
        BENCHDNN_PRINT(0,
                "@@@ [%s] %strust range:%.2f nz:%.2f "
                "(level range:%.2f nz:%.2f). "
                "in:%d (ok:%d) below:%d (ok:%d) above:%d (ok:%d) nz:%d "
                "total:%lu\n",
                skind, final_compare ? "final: " : "", trust_rg, trust_nz,
                trust_rg_level, trust_nz_level, in, in_ok, below, below_ok,
                above, above_ok, non_zero, (unsigned long)r->total);
    }

    if (no_trust) {
        if (r->state != FAILED) r->state = MISTRUSTED;
        BENCHDNN_PRINT(0,
                "@@@ [%s] test-bug: trust is too low. "
                "range:%.2f (?<%.2f) nz:%.2f (?<%.2f) (nz: %d total: %lu)\n",
                skind, trust_rg, trust_rg_level, trust_nz, trust_nz_level,
                non_zero, (unsigned long)r->total);
    }

    if (final_compare && r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int compare_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r,
        bool final_compare) {
    return compare_dat(p, SRC, mem_dt, mem_fp, r, final_compare);
}
int compare_wei(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r,
        bool final_compare) {
    return compare_dat(p, WEI, mem_dt, mem_fp, r, final_compare);
}
int compare_bia(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r,
        bool final_compare) {
    return compare_dat(p, BIA, mem_dt, mem_fp, r, final_compare);
}
int compare_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r,
        bool final_compare) {
    return compare_dat(p, DST, mem_dt, mem_fp, r, final_compare);
}

int fill_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    const bool need_extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t extra_mem;
    if (need_extra_mem) {
        extra_mem
                = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::abx, get_test_engine());
    }
    dnn_mem_t &mem_00 = need_extra_mem ? extra_mem : mem_fp;

    const auto &c = p->cfg[SRC];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(p->mb, p->ic, p->id, p->ih, p->iw,
            [&](int mb, int ic, int id, int ih, int iw) {
                const int gen
                        = 101 * id + 103 * ih + 107 * iw + 109 * mb + 113 * ic;
                const bool non_base = flip_coin(gen, c.f_sparsity);
                float value = non_base ? c.f_min + gen * c.f_step % range
                                       : c.f_base;

                maybe_zero_point(
                        p->attr, value, p->src_zp, ic, DNNL_ARG_SRC, true);

                ((float *)mem_00)[src_off_f(p, mb, 0, ic, id, ih, iw)] = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_src(p, mem_fp, mem_00, r), WARN);
    }

    return OK;
}

int fill_wei(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    const bool wino_s8 = p->alg == WINO && p->cfg[WEI].dt == dnnl_s8;
    const bool s8_s8 = p->cfg[WEI].dt == dnnl_s8 && p->cfg[SRC].dt == dnnl_s8;
    const bool diff_data_type = mem_dt.dt() != mem_fp.dt();
    const bool check_reorder = diff_data_type && !wino_s8 && !s8_s8;

    dnn_mem_t extra_mem;
    if (check_reorder) {
        extra_mem
                = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::abx, get_test_engine());
    }
    dnn_mem_t &mem_00 = check_reorder ? extra_mem : mem_fp;

    const auto &c = p->cfg[WEI];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(p->g, p->oc / p->g, p->ic / p->g, p->kd, p->kh,
            p->kw, [&](int g, int oc, int ic, int kd, int kh, int kw) {
                const int gen
                        = 127 * kd + 131 * kh + 137 * kw + 139 * oc + 149 * ic;
                const bool non_base = flip_coin(gen, c.f_sparsity);
                const float value = non_base ? c.f_min + gen * c.f_step % range
                                             : c.f_base;

                ((float *)mem_00)[wei_off_f(p, g, oc, ic, kd, kh, kw)] = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (check_reorder) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_wei(p, mem_fp, mem_00, r), WARN);
    }

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
        const float value
                = non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float *)mem_00)[i] = value;
    }

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_bia(p, mem_fp, mem_00, r), WARN);
    }
    return OK;
}

int fill_dst_with_params(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        dnnl_data_type_t dt, double sparsity, int min, int max, int base,
        int step, res_t *r) {
    const bool need_extra_mem = mem_dt.dt() != mem_fp.dt();
    dnn_mem_t extra_mem;
    if (need_extra_mem) {
        extra_mem
                = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::abx, get_test_engine());
    }

    dnn_mem_t &mem_00 = need_extra_mem ? extra_mem : mem_fp;
    const int range = max - min + 1;

    dnnl::impl::parallel_nd(p->mb, p->oc, p->od, p->oh, p->ow,
            [&](int mb, int oc, int od, int oh, int ow) {
                const int gen
                        = 157 * od + 163 * oh + 167 * ow + 173 * mb + 179 * oc;
                const bool non_base = flip_coin(gen, sparsity);
                const float value = non_base ? min + gen * step % range : base;

                ((float *)mem_00)[dst_off_f(p, mb, 0, oc, od, oh, ow)] = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (need_extra_mem) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        SAFE(compare_dst(p, mem_fp, mem_00, r), WARN);
    }

    return OK;
}

int fill_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    auto dst_dt = mem_dt.dt();
    int sum_ind = p->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM);
    auto sum_dt = (sum_ind != -1) ? p->attr.post_ops.entry[sum_ind].sum.dt
                                  : dnnl_data_type_undef;
    bool diff_sum_dst_types
            = sum_dt != dnnl_data_type_undef && sum_dt != dst_dt;

    const auto &c = p->cfg[DST];
    float f_min = (diff_sum_dst_types) ? lowest_dt(sum_dt) : c.f_min;
    float f_max = (diff_sum_dst_types) ? max_dt(sum_dt) : c.f_max;

    // Change mem dt to sum dt, so we can save sum data properly.
    if (diff_sum_dst_types) { mem_dt.set_dt(sum_dt); }

    fill_dst_with_params(p, mem_dt, mem_fp, sum_dt, c.f_sparsity, f_min, f_max,
            c.f_base, c.f_step, r);

    // Return dst data type back.
    if (diff_sum_dst_types) { mem_dt.set_dt(dst_dt); }
    return OK;
}

inline int init_pd_custom(dnnl_engine_t engine, const prb_t *p,
        dnnl_primitive_desc_t &cpd, res_t *r,
        dnnl_data_type_t src_dt = dnnl_data_type_undef,
        dnnl_data_type_t wei_dt = dnnl_data_type_undef,
        dnnl_data_type_t bia_dt = dnnl_data_type_undef,
        dnnl_data_type_t dst_dt = dnnl_data_type_undef,
        dnnl_data_type_t acc_dt = dnnl_data_type_undef,
        std::string src_tag = tag::undef, std::string wei_tag = tag::undef,
        std::string bia_tag = tag::undef, std::string dst_tag = tag::undef) {
    dnnl_convolution_desc_t cd;
    dnnl_memory_desc_t src_d, wei_d, bia_d, dst_d;

    dnnl_dims_t src_1d_dims = {p->mb, p->ic, p->iw};
    dnnl_dims_t src_2d_dims = {p->mb, p->ic, p->ih, p->iw};
    dnnl_dims_t src_3d_dims = {p->mb, p->ic, p->id, p->ih, p->iw};
    dnnl_dim_t *src_dims = p->ndims == 5
            ? src_3d_dims
            : p->ndims == 4 ? src_2d_dims : src_1d_dims;

    dnnl_dims_t wei_1d_dims = {p->g, p->oc / p->g, p->ic / p->g, p->kw};
    dnnl_dims_t wei_2d_dims = {p->g, p->oc / p->g, p->ic / p->g, p->kh, p->kw};
    dnnl_dims_t wei_3d_dims
            = {p->g, p->oc / p->g, p->ic / p->g, p->kd, p->kh, p->kw};
    dnnl_dim_t *wei_dims = p->ndims == 5
            ? &wei_3d_dims[!p->has_groups]
            : p->ndims == 4 ? &wei_2d_dims[!p->has_groups]
                            : &wei_1d_dims[!p->has_groups];

    dnnl_dims_t bia_dims = {p->oc};

    dnnl_dims_t dst_1d_dims = {p->mb, p->oc, p->ow};
    dnnl_dims_t dst_2d_dims = {p->mb, p->oc, p->oh, p->ow};
    dnnl_dims_t dst_3d_dims = {p->mb, p->oc, p->od, p->oh, p->ow};
    dnnl_dim_t *dst_dims = p->ndims == 5
            ? dst_3d_dims
            : p->ndims == 4 ? dst_2d_dims : dst_1d_dims;

    if (src_dt == dnnl_data_type_undef) src_dt = p->cfg[SRC].dt;
    if (wei_dt == dnnl_data_type_undef) wei_dt = p->cfg[WEI].dt;
    if (bia_dt == dnnl_data_type_undef) bia_dt = p->cfg[BIA].dt;
    if (dst_dt == dnnl_data_type_undef) dst_dt = p->cfg[DST].dt;
    if (acc_dt == dnnl_data_type_undef) acc_dt = p->cfg[ACC].dt;
    if (src_tag == tag::undef) src_tag = normalize_tag(p->stag, p->ndims);
    if (wei_tag == tag::undef) wei_tag = normalize_tag(p->wtag, p->ndims);
    if (bia_tag == tag::undef) bia_tag = tag::any;
    if (dst_tag == tag::undef) dst_tag = normalize_tag(p->dtag, p->ndims);

    SAFE(init_md(&src_d, p->ndims, src_dims, src_dt, src_tag), WARN);

    SAFE(init_md(&wei_d, p->ndims + p->has_groups, wei_dims, wei_dt, wei_tag),
            WARN);

    SAFE(init_md(&bia_d, 1, bia_dims, bia_dt, bia_tag), WARN);

    SAFE(init_md(&dst_d, p->ndims, dst_dims, dst_dt, dst_tag), WARN);

    dnnl_dim_t strides_nd[] = {p->sd, p->sh, p->sw};
    dnnl_dim_t dilates_nd[] = {p->dd, p->dh, p->dw};
    dnnl_dim_t padding_nd[] = {p->pd, p->ph, p->pw};
    dnnl_dim_t padding_r_nd[] = {p->pd_r, p->ph_r, p->pw_r};

    dnnl_dim_t *strides = strides_nd + (5 - p->ndims);
    dnnl_dim_t *dilates = dilates_nd + (5 - p->ndims);
    dnnl_dim_t *padding = padding_nd + (5 - p->ndims);
    dnnl_dim_t *padding_r = padding_r_nd + (5 - p->ndims);

    dnnl_alg_kind_t alg = dnnl_convolution_direct;
    if (p->alg == WINO) alg = dnnl_convolution_winograd;
    if (p->alg == AUTO) alg = dnnl_convolution_auto;

    switch (p->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            DNN_SAFE(dnnl_dilated_convolution_forward_desc_init(&cd,
                             p->dir == FWD_I ? dnnl_forward_inference
                                             : dnnl_forward_training,
                             alg, &src_d, &wei_d,
                             p->dir == FWD_B ? &bia_d : nullptr, &dst_d,
                             strides, dilates, padding, padding_r),
                    WARN);
            break;
        case BWD_D:
            DNN_SAFE(dnnl_dilated_convolution_backward_data_desc_init(&cd, alg,
                             &src_d, &wei_d, &dst_d, strides, dilates, padding,
                             padding_r),
                    WARN);
            break;
        case BWD_W:
        case BWD_WB:
            DNN_SAFE(dnnl_dilated_convolution_backward_weights_desc_init(&cd,
                             alg, &src_d, &wei_d,
                             p->dir == BWD_W ? nullptr : &bia_d, &dst_d,
                             strides, dilates, padding, padding_r),
                    WARN);
            break;
        default: DNN_SAFE(dnnl_invalid_arguments, CRIT);
    }

    DNN_SAFE(cd.accum_data_type == acc_dt ? dnnl_success : dnnl_unimplemented,
            CRIT);

    attr_args_t attr_args;
    attr_args.prepare_output_scales(p->attr, p->scales, p->oc);
    attr_args.prepare_binary_post_op_mds(p->attr, p->ndims, dst_dims);
    auto dnnl_attr = create_dnnl_attr(p->attr, attr_args);

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&cpd, &cd, dnnl_attr, engine, nullptr);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (!r) return OK;

    if (init_status == dnnl_unimplemented) return r->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    r->impl_name = query_impl_info(cpd);
    if (maybe_skip(r->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                r->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(cpd), WARN);
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

    // TODO: temporary disable binary post-op on GPU
    if (engine_tgt_kind == dnnl_gpu && p->attr.post_ops.binary_index() != -1) {
        r->state = SKIPPED, r->reason = CASE_NOT_SUPPORTED;
        return;
    }

    // Winograd implementation limitations.
    if (p->alg == WINO) {
        static auto isa = dnnl_get_effective_cpu_isa();
        static bool has_avx512_common = isa >= dnnl_cpu_isa_avx512_mic;
        static bool has_avx512_bw = isa >= dnnl_cpu_isa_avx512_core;
        bool is_int8 = p->cfg[WEI].dt == dnnl_s8;

        bool pad_ok_f32
                = p->pw <= 1 && p->ph <= 1 && p->pw_r <= 1 && p->ph_r <= 1;
        bool pad_ok_int8 = p->pw <= 1 && p->ph <= 1 && p->pw == p->pw_r
                && p->ph == p->ph_r;

        bool shape_ok = p->ndims == 4 && p->g == 1 && p->kh == 3 && p->kw == 3
                && p->sh == 1 && p->sw == 1 && p->dh == 0 && p->dw == 0
                && IMPLICATION(!is_int8, pad_ok_f32)
                && IMPLICATION(is_int8,
                        (p->ic % 16 == 0) && (p->oc % 16 == 0) && pad_ok_int8);
        bool bwd_is_syncable = IMPLICATION(
                (p->dir & FLAG_BWD), dnnl::impl::dnnl_thr_syncable());

        const auto stag = normalize_tag(p->stag, p->ndims);
        const bool stag_is_abx = stag == normalize_tag(tag::abx, p->ndims);
        const bool stag_is_axb = stag == normalize_tag(tag::axb, p->ndims);
        const auto dtag = normalize_tag(p->dtag, p->ndims);
        const bool dtag_is_abx = dtag == normalize_tag(tag::abx, p->ndims);
        const bool dtag_is_axb = dtag == normalize_tag(tag::axb, p->ndims);
        const bool is_plain
                = stag_is_abx || stag_is_axb || dtag_is_abx || dtag_is_axb;
        const bool plain_ok = is_int8 && !stag_is_abx && !dtag_is_abx
                && (stag_is_axb || dtag_is_axb);

        const auto &po = p->attr.post_ops;
        const auto sum_idx = po.find(attr_t::post_ops_t::kind_t::SUM);
        const bool sum_post_op_ok
                = sum_idx == -1 || po.entry[sum_idx].sum.scale == 1.f;

        if (!has_avx512_common || !shape_ok || (!has_avx512_bw && is_int8)
                || !bwd_is_syncable || (is_plain && !plain_ok)
                || !sum_post_op_ok) {
            r->state = SKIPPED, r->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    check_known_skipped_case(p, r);
    if (r->state == SKIPPED) return OK;

    dnnl_primitive_t c {};
    // TODO: align init_pd interface with a common one which is used
    // in the rest of the benchdnn drivers
    auto init_pd = [&](dnnl_engine_t engine, const prb_t *p,
                           dnnl_primitive_desc_t &cpd, res_t *r, dir_t dir,
                           const_dnnl_primitive_desc_t hint) {
        SAFE(init_pd_custom(engine, p, cpd, r), WARN);
        return OK;
    };

    SAFE(init_prim(&c, init_pd, p, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(c, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(c));
        return r->state = SKIPPED, r->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    alg_t alg = p->alg;
    if (alg == AUTO) {
        dnnl_convolution_desc_t *temp_conv_desc = {nullptr};
        DNN_SAFE(dnnl_primitive_desc_query(const_pd, dnnl_query_convolution_d,
                         0, &temp_conv_desc),
                CRIT);
        alg = alg_kind2alg(temp_conv_desc->alg_kind);
    }
    const auto cfg = auto_cfg(alg, p->cfg);
    prb_t p_new((desc_t)*p, p->dir, cfg, p->stag, p->wtag, p->dtag, alg,
            p->attr, p->mb);
    p = &p_new;

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

    // Try to use CPU primitive as the reference in GPU testing to reduce
    // testing time
    dnnl_primitive_t c_ref {};

    if (bench_mode & CORR && engine_tgt_kind == dnnl_gpu && fast_ref_gpu) {
        dnnl_primitive_desc_t cpd_ref = nullptr;
        SAFE(init_pd_custom(get_cpu_engine(), p, cpd_ref, nullptr, fp, fp, fp,
                     fp, fp, src_tag, wei_tag, tag::x, src_tag),
                WARN);
        if (cpd_ref) {
            DNN_SAFE(dnnl_primitive_create(&c_ref, cpd_ref), WARN);
            BENCHDNN_PRINT(
                    5, "%s\n", "benchdnn: use CPU primitive as the reference");
            DNN_SAFE(dnnl_primitive_desc_destroy(cpd_ref), CRIT);
        }
    }

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t bia_dt(bia_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    dnn_mem_t scales;
    dnn_mem_t src_zero_points_m;
    dnn_mem_t dst_zero_points_m;
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(
                 const_pd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    dnn_mem_t src_fp(src_md, fp, src_tag, test_engine);
    dnn_mem_t wei_fp(wei_md, fp, wei_tag, test_engine);
    dnn_mem_t dst_fp(dst_md, fp, src_tag, test_engine);
    dnn_mem_t bia_fp(bia_md, fp, tag::x, test_engine);

    SAFE(fill_src(p, src_dt, src_fp, r), WARN);
    SAFE(fill_wei(p, wei_dt, wei_fp, r), WARN);
    SAFE(fill_dst(p, dst_dt, dst_fp, r), WARN);
    SAFE(fill_bia(p, bia_dt, bia_fp, r), WARN);
    maybe_prepare_runtime_scales(scales, p->attr, p->oc, p->scales);
    maybe_prepare_runtime_zero_points(
            src_zero_points_m, p->attr, DNNL_ARG_SRC, p->ic, p->src_zp);
    maybe_prepare_runtime_zero_points(
            dst_zero_points_m, p->attr, DNNL_ARG_DST, p->oc, p->dst_zp);

    args_t args;

    if (p->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
        args.set(DNNL_ARG_ATTR_OUTPUT_SCALES, scales);
        args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_points_m);
        args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_points_m);
        args.set(binary_po_args, binary_po_dt);

        SAFE(execute_and_wait(c, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(
                    p, c_ref, src_fp, wei_fp, bia_fp, binary_po_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, src_tag, test_engine);
            SAFE(compare_dst(p, dst, dst_fp, r, true), WARN);
        }
    } else if (p->dir == BWD_D) {
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(c, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_d(p, c_ref, src_fp, wei_fp, bia_fp, dst_fp);
            dnn_mem_t src(src_dt, fp, src_tag, test_engine);
            SAFE(compare_src(p, src, src_fp, r, true), WARN);
        }
    } else if (p->dir & FLAG_BWD && p->dir & FLAG_WEI) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_BIAS, bia_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(c, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd_w(p, c_ref, src_fp, wei_fp, bia_fp, dst_fp);
            dnn_mem_t wei(wei_dt, fp, wei_tag, test_engine);
            SAFE(compare_wei(p, wei, wei_fp, r, true), WARN);
            if (p->dir & FLAG_BIA) {
                dnn_mem_t bia(bia_dt, fp, tag::x, test_engine);
                SAFE(compare_bia(p, bia, bia_fp, r, true), WARN);
            }
        }
    } else {
        SAFE(FAIL, CRIT);
    }

    measure_perf(r->timer, c, args);

    DNN_SAFE_V(dnnl_primitive_destroy(c));
    DNN_SAFE_V(dnnl_primitive_destroy(c_ref));

    return OK;
}

} // namespace conv
