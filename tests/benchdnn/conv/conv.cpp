/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
* Copyright 2021 Arm Ltd. and affiliates
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

#include <cstring>
#include <vector>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "tests/test_isa_common.hpp"
#endif

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"
#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "conv/conv.hpp"
#include "eltwise/eltwise.hpp"
#include "prelu/prelu.hpp"

#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE
extern "C" bool dnnl_impl_gpu_conv_wino_should_silence_unimplemented(
        dnnl_engine_t engine);
#endif

namespace conv {

double get_non_zero_trust_percent(const prb_t *prb, data_kind_t kind) {
    auto negative_to_zero = [&]() {
        using pk = attr_t::post_ops_t::kind_t;
        const auto &po = prb->attr.post_ops;
        int count = 0;

        // Check for all post-ops that convert negative to zero
        std::vector<pk> non_neg_po {pk::ABS, pk::BRELU};
        std::vector<pk> non_neg_alpha_0_po {
                pk::CLIP, pk::CLIP_V2, pk::ELU, pk::RELU};
        for (int i = 0; i < po.len(); ++i) {
            const auto &e = po.entry[i];
            if (!e.is_eltwise_kind()) continue;

            auto k = e.kind;
            auto alpha = e.eltwise.alpha;

            count += std::any_of(non_neg_po.cbegin(), non_neg_po.cend(),
                    [k](const pk alg) { return alg == k; });
            count += std::any_of(non_neg_alpha_0_po.cbegin(),
                    non_neg_alpha_0_po.cend(), [k, alpha](const pk alg) {
                        return alg == k && alpha == 0;
                    });
        }
        // Check for u8 dst
        count += prb->cfg[DST].dt == dnnl_u8;
        // Check for physically padded area in the output
        count += prb->od > prb->id || prb->oh > prb->ih || prb->ow > prb->iw;

        return !!count;
    };

    double trust = 0.3; /* why? */
    switch (kind) {
        case SRC: trust /= prb->sd * prb->sh * prb->sw; break;
        case WEI:
            trust /= 1. * prb->kd * prb->kh * prb->kw
                    / MIN3(prb->kd * prb->kh * prb->kw,
                            prb->id * prb->ih * prb->iw,
                            prb->od * prb->oh * prb->ow);
            break;
        case BIA:
            trust = 0.8 * prb->cfg[DST].f_sparsity; /* why? */
            break;
        case DST: trust /= (1.f + negative_to_zero()); break;
    }

    return trust;
}

inline double get_eps(const prb_t *prb, const data_kind_t kind) {
    // Winograd specifics
    if (prb->alg & WINO && prb->dir & FLAG_WEI) {
        // This is an empirical equation derived by observing growth error with
        // increasing 'k' dimension in gemm of winograd
        const float log_const = log10(0.125 * prb->mb * prb->oh * prb->ow);
        return prb->cfg[kind].eps * (MAX2(1, pow(10, 0.4 * log_const)));
    }
    return prb->cfg[kind].eps;
}

inline int compare_data_p2p(const prb_t *prb, data_kind_t kind,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    compare::compare_t cmp;
    cmp.set_threshold(prb->cfg[kind].eps);
    cmp.set_data_kind(kind);
    const float zpp = (1.f - get_non_zero_trust_percent(prb, kind)) * 100.f;
    cmp.set_zero_trust_percent(zpp);
    SAFE(cmp.compare(mem_fp, mem_dt, prb->attr, res), WARN);
    return res->state == FAILED ? FAIL : OK;
}

inline int compare_data_norm(const prb_t *prb, data_kind_t kind,
        dnn_mem_t &mem_dt0, dnn_mem_t &mem_fp, res_t *res,
        bool final_compare = true) {
    const auto nelems = mem_dt0.nelems();
    if (nelems == 0) return res->state = PASSED, OK;
    res->total = nelems;
    res->errors = 1;

    dnn_mem_t mem_dt(mem_dt0, dnnl_f32, tag::abx, get_test_engine());

    const char *skind = data_kind2str(kind);

    int in = 0, not_a_num = 0, below = 0, above = 0, non_zero = 0;

    int sum_ind = prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM);
    auto sum_dt = (sum_ind != -1) ? prb->attr.post_ops.entry[sum_ind].sum.dt
                                  : dnnl_data_type_undef;

    bool diff_sum_dt = kind == DST && sum_dt != dnnl_data_type_undef
            && sum_dt != prb->cfg[kind].dt;
    dnnl_data_type_t f_dt = diff_sum_dt ? sum_dt : prb->cfg[kind].dt;
    float f_min = diff_sum_dt ? lowest_dt(f_dt) : prb->cfg[kind].min;
    float f_max = diff_sum_dt ? max_dt(f_dt) : prb->cfg[kind].max;

    diff_norm_t diff_norm;

    for (int64_t i = 0; i < mem_fp.nelems(); ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = round_to_nearest_representable(f_dt, fp0);

        if (std::isnan(fp0) && is_integral_dt(f_dt)) {
            // XXX: if reference fp0 value is nan, allow to return anything from
            // the library for integral target data types.
            not_a_num += 1;
        } else if (is_cpu() && f_dt == dnnl_s32 && fp == max_dt(dnnl_s32)
                && dt >= BENCHDNN_S32_TO_F32_SAT_CONST
                && dt < max_dt(dnnl_s32)) {
            // Don't include f32->s32 saturation values into final norm
            above += 1;
        } else if (fp0 < f_min) {
            diff_norm.update(f_min, dt);
            below += 1;
        } else if (fp0 > f_max) {
            diff_norm.update(f_max, dt);
            above += 1;
        } else {
            diff_norm.update(fp, dt);
            in += 1;
        }

        non_zero += fp != 0;
    }

    diff_norm.done();

    const float eps = get_eps(prb, kind);
    bool ok = diff_norm.rel_diff(norm_t::L2) <= eps;
    if (ok) res->errors = 0;

    if (res->errors) {
        res->state = FAILED;
        BENCHDNN_PRINT(0,
                "@@@ [%s] %sdiff: err:%d, l0(``%g``) "
                "l1:(%g,%g,%g,``%g``) "
                "l2:(%g,%g,%g,``%g``) "
                "l8:(%g,%g,%g,``%g``)\n",
                skind, "final: ", (int)res->errors,
                diff_norm.rel_diff(norm_t::L0), diff_norm.a_[norm_t::L1],
                diff_norm.b_[norm_t::L1], diff_norm.diff_[norm_t::L1],
                diff_norm.rel_diff(norm_t::L1), diff_norm.a_[norm_t::L2],
                diff_norm.b_[norm_t::L2], diff_norm.diff_[norm_t::L2],
                diff_norm.rel_diff(norm_t::L2), diff_norm.a_[norm_t::L8],
                diff_norm.b_[norm_t::L8], diff_norm.diff_[norm_t::L8],
                diff_norm.rel_diff(norm_t::L8));
    }

    const double trust_rg_level = 0.3;
    const double trust_nz_level = get_non_zero_trust_percent(prb, kind);

    const double trust_rg = (double)in / res->total;
    const double trust_nz = (double)non_zero / res->total;

    const bool no_trust
            = (trust_rg < trust_rg_level || trust_nz < trust_nz_level);

    const bool dump = verbose >= 20
            || (verbose >= 10 && (trust_rg < 1. || trust_nz < 1.));
    if (dump) {
        BENCHDNN_PRINT(0,
                "@@@ [%s] %strust range:%.2f nz:%.2f "
                "(level range:%.2f nz:%.2f). "
                "in:%d nan:%d below:%d above:%d nz:%d total:%lu\n",
                skind, "final: ", trust_rg, trust_nz, trust_rg_level,
                trust_nz_level, in, not_a_num, below, above, non_zero,
                (unsigned long)res->total);
    }

    if (no_trust) {
        if (res->state != FAILED) res->state = MISTRUSTED;
        BENCHDNN_PRINT(0,
                "@@@ [%s] test-bug: trust is too low. "
                "range:%.2f (?<%.2f) nz:%.2f (?<%.2f) (nz: %d total: %lu)\n",
                skind, trust_rg, trust_rg_level, trust_nz, trust_nz_level,
                non_zero, (unsigned long)res->total);
    }

    if (res->state == EXECUTED) res->state = PASSED;

    return res->state == FAILED ? FAIL : OK;
}
int compare_data(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    const bool compare_with_norm = (prb->alg & WINO);
    if (compare_with_norm)
        return compare_data_norm(prb, kind, mem_dt, mem_fp, res);
    return compare_data_p2p(prb, kind, mem_dt, mem_fp, res);
}

bool need_src_init(const prb_t *prb) {
    return !(prb->dir == BWD_D);
}

bool need_wei_init(const prb_t *prb) {
    return !(prb->dir & FLAG_BWD && prb->dir & FLAG_WEI);
}

bool need_bia_init(const prb_t *prb) {
    return need_wei_init(prb);
}

bool need_dst_init(const prb_t *prb) {
    return !(prb->dir & FLAG_FWD)
            || (prb->attr.post_ops.find(attr_t::post_ops_t::SUM) >= 0);
}

int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const bool check_reorder
            = (is_bench_mode(CORR)) && (mem_dt.dt() != mem_fp.dt());
    dnn_mem_t extra_mem;
    if (check_reorder) {
        extra_mem
                = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::abx, get_test_engine());
    }
    dnn_mem_t &mem_00 = check_reorder ? extra_mem : mem_fp;

    // Use dense filling for small problems.
    int src_nelems_mask = powf(2.f, prb->ndims) - 1;
    src_nelems_mask -= 1; // remove minibatch as independent dimension
    auto src_nelems = prb->desc_nelems(DNNL_ARG_SRC, src_nelems_mask);
    if (prb->has_groups) src_nelems /= prb->g; // groups are also independent

    const auto &c = prb->cfg[SRC];
    const int range = c.f_max - c.f_min + 1;
    const float sparsity = src_nelems < 100 ? 1.f : c.f_sparsity;

    dnnl::impl::parallel_nd(prb->mb, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int64_t mb, int64_t ic, int64_t id, int64_t ih, int64_t iw) {
                const int64_t gen
                        = 101 * id + 103 * ih + 107 * iw + 109 * mb + 113 * ic;
                const bool non_base = flip_coin(gen, sparsity);
                float value = non_base ? c.f_min + gen * c.f_step % range
                                       : c.f_base;

                maybe_zero_point(
                        prb->attr, value, prb->src_zp, ic, DNNL_ARG_SRC, true);

                ((float *)mem_00)[src_off_f(prb, mb, 0, ic, id, ih, iw)]
                        = round_to_nearest_representable(mem_dt.dt(), value);
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (check_reorder) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        int rc = std::memcmp((void *)mem_fp, (void *)mem_00, mem_00.size());
        if (rc != 0) {
            res->state = FAILED;
            SAFE(FAIL, WARN);
        }
    }

    return OK;
}

int fill_wei(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const bool wino_s8 = prb->alg == WINO && prb->cfg[WEI].dt == dnnl_s8;
    const bool is_def_zp = prb->attr.zero_points.is_def(DNNL_ARG_SRC);
    const bool diff_data_type = mem_dt.dt() != mem_fp.dt();

    dnnl_data_type_t dt_check = dnnl_s8;
#if DNNL_AARCH64
    /* Note for x64:
    Both data types of src and weight are s8, oneDNN addds 128 to one of the s8
    input to make it of type u8 instead, as explained in
    https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html or
    doc/advanced/int8_computations.md
    It is because `VPDPBUSD` instruction uses the combination of s8 and u8 as
    input.

    Note for AArch64:
    Because dot product instructions of AArch64 "SDOT" receives s8 input
    for both src and weight, the addition (and its counterpart of subtraction)
    is not required for AArch64.
    */
    if (res->impl_name.find("jit", 0) == 0) dt_check = dnnl_u8;
#endif

    const bool wei_x8x8
            = prb->cfg[WEI].dt == dnnl_s8 && prb->cfg[SRC].dt == dt_check;
    const bool check_reorder = (is_bench_mode(CORR)) && diff_data_type
            && !wino_s8 && !wei_x8x8 && is_def_zp;

    dnn_mem_t extra_mem;
    if (check_reorder) {
        extra_mem
                = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::abx, get_test_engine());
    }
    dnn_mem_t &mem_00 = check_reorder ? extra_mem : mem_fp;

    const auto &c = prb->cfg[WEI];
    const int range = c.f_max - c.f_min + 1;

    dnnl::impl::parallel_nd(prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kd,
            prb->kh, prb->kw,
            [&](int64_t g, int64_t oc, int64_t ic, int64_t kd, int64_t kh,
                    int64_t kw) {
                const int64_t gen = 113 * g + 127 * kd + 131 * kh + 137 * kw
                        + 139 * oc + 149 * ic + 151;
                const bool non_base = flip_coin(gen, c.f_sparsity);
                const float value = non_base ? c.f_min + gen * c.f_step % range
                                             : c.f_base;
                ((float *)mem_00)[wei_off_f(prb, g, oc, ic, kd, kh, kw)]
                        = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (check_reorder) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        int rc = std::memcmp((void *)mem_fp, (void *)mem_00, mem_00.size());
        if (rc != 0) {
            res->state = FAILED;
            SAFE(FAIL, WARN);
        }
    }
    if ((wei_x8x8 || !is_def_zp) && is_cpu()) {
        // Check that s8 -> s8_comp exists in the library since users may have
        // already quantized data.
        dnn_mem_t mem_fp_s8(mem_fp.md_, dnnl_s8, get_test_engine());
        dnn_mem_t mem_dt_s8(mem_dt.md_, dnnl_s8, get_test_engine());
        SAFE(mem_fp_s8.reorder(mem_fp), WARN);
        SAFE(mem_dt_s8.reorder(mem_fp_s8), WARN);
        SAFE(mem_dt.size() == mem_dt_s8.size() ? OK : FAIL, WARN);
        int rc = std::memcmp((void *)mem_dt, (void *)mem_dt_s8, mem_dt.size());
        SAFE(rc == 0 ? OK : FAIL, WARN);
    }
    return OK;
}

int fill_bia(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const bool check_reorder
            = (is_bench_mode(CORR)) && (mem_dt.dt() != mem_fp.dt());
    dnn_mem_t extra_mem;
    if (check_reorder)
        extra_mem = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::x, get_test_engine());
    dnn_mem_t &mem_00 = check_reorder ? extra_mem : mem_fp;

    const size_t nelems = mem_00.nelems();
    if (nelems == 0) return OK;

    const auto &c = prb->cfg[BIA];
    const int range = c.f_max - c.f_min + 1;

    for (size_t i = 0; i < nelems; ++i) {
        const int gen = (int)(151 * i);
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value
                = non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float *)mem_00)[i] = value;
    }

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (check_reorder) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        int rc = std::memcmp((void *)mem_fp, (void *)mem_00, mem_00.size());
        if (rc != 0) {
            res->state = FAILED;
            SAFE(FAIL, WARN);
        }
    }
    return OK;
}

int fill_dst_with_params(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        dnnl_data_type_t dt, double sparsity, int min, int max, int base,
        int step, res_t *res) {
    const bool check_reorder
            = (is_bench_mode(CORR)) && (mem_dt.dt() != mem_fp.dt());
    dnn_mem_t extra_mem;
    if (check_reorder) {
        extra_mem
                = dnn_mem_t(mem_dt.md_, dnnl_f32, tag::abx, get_test_engine());
    }

    dnn_mem_t &mem_00 = check_reorder ? extra_mem : mem_fp;
    const int range = max - min + 1;

    dnnl::impl::parallel_nd(prb->mb, prb->oc, prb->od, prb->oh, prb->ow,
            [&](int64_t mb, int64_t oc, int64_t od, int64_t oh, int64_t ow) {
                const int64_t gen
                        = 157 * od + 163 * oh + 167 * ow + 173 * mb + 179 * oc;
                const bool non_base = flip_coin(gen, sparsity);
                const float value = non_base ? min + gen * step % range : base;

                ((float *)mem_00)[dst_off_f(prb, mb, 0, oc, od, oh, ow)]
                        = value;
            });

    SAFE(mem_dt.reorder(mem_00), WARN);
    if (check_reorder) {
        SAFE(mem_fp.reorder(mem_dt), WARN);
        int rc = std::memcmp((void *)mem_fp, (void *)mem_00, mem_00.size());
        if (rc != 0) {
            res->state = FAILED;
            SAFE(FAIL, WARN);
        }
    }

    return OK;
}

int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    auto dst_dt = mem_dt.dt();
    int sum_ind = prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM);
    auto sum_dt = (sum_ind != -1) ? prb->attr.post_ops.entry[sum_ind].sum.dt
                                  : dnnl_data_type_undef;
    bool diff_sum_dst_types
            = sum_dt != dnnl_data_type_undef && sum_dt != dst_dt;
    bool sum_dt_is_int8 = sum_dt == dnnl_s8 || sum_dt == dnnl_u8;

    const auto &c = prb->cfg[DST];
    float f_min = c.f_min;
    float f_max = c.f_max;
    if (diff_sum_dst_types && sum_dt_is_int8) {
        f_min = lowest_dt(sum_dt);
        f_max = max_dt(sum_dt);
    }

    // Change mem dt to sum dt, so we can save sum data properly.
    if (diff_sum_dst_types) { mem_dt.set_dt(sum_dt); }

    fill_dst_with_params(prb, mem_dt, mem_fp, sum_dt, c.f_sparsity, f_min,
            f_max, c.f_base, c.f_step, res);

    // Return dst data type back.
    if (diff_sum_dst_types) { mem_dt.set_dt(dst_dt); }
    return OK;
}

int init_pd(dnnl_engine_t engine, const prb_t *prb, dnnl_primitive_desc_t &cpd,
        res_t *res, dir_t dir, const_dnnl_primitive_desc_t hint) {
    dnnl_convolution_desc_t cd;
    dnnl_memory_desc_t src_d, wei_d, bia_d, dst_d;

    dnnl_dims_t src_1d_dims = {prb->mb, prb->ic, prb->iw};
    dnnl_dims_t src_2d_dims = {prb->mb, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t src_3d_dims = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};
    dnnl_dim_t *src_dims = prb->ndims == 5
            ? src_3d_dims
            : prb->ndims == 4 ? src_2d_dims : src_1d_dims;

    dnnl_dims_t wei_1d_dims
            = {prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kw};
    dnnl_dims_t wei_2d_dims
            = {prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kh, prb->kw};
    dnnl_dims_t wei_3d_dims = {prb->g, prb->oc / prb->g, prb->ic / prb->g,
            prb->kd, prb->kh, prb->kw};
    dnnl_dim_t *wei_dims = prb->ndims == 5
            ? &wei_3d_dims[!prb->has_groups]
            : prb->ndims == 4 ? &wei_2d_dims[!prb->has_groups]
                              : &wei_1d_dims[!prb->has_groups];

    dnnl_dims_t bia_dims = {prb->oc};

    dnnl_dims_t dst_1d_dims = {prb->mb, prb->oc, prb->ow};
    dnnl_dims_t dst_2d_dims = {prb->mb, prb->oc, prb->oh, prb->ow};
    dnnl_dims_t dst_3d_dims = {prb->mb, prb->oc, prb->od, prb->oh, prb->ow};
    dnnl_dim_t *dst_dims = prb->ndims == 5
            ? dst_3d_dims
            : prb->ndims == 4 ? dst_2d_dims : dst_1d_dims;

    SAFE(init_md(&src_d, prb->ndims, src_dims, prb->cfg[SRC].dt,
                 normalize_tag(prb->stag, prb->ndims)),
            WARN);
    SAFE(init_md(&wei_d, prb->ndims + prb->has_groups, wei_dims,
                 prb->cfg[WEI].dt,
                 normalize_tag(prb->wtag, prb->ndims + prb->has_groups)),
            WARN);
    SAFE(init_md(&bia_d, 1, bia_dims, prb->cfg[BIA].dt, tag::any), WARN);
    SAFE(init_md(&dst_d, prb->ndims, dst_dims, prb->cfg[DST].dt,
                 normalize_tag(prb->dtag, prb->ndims)),
            WARN);

    dnnl_dim_t strides_nd[] = {prb->sd, prb->sh, prb->sw};
    dnnl_dim_t dilates_nd[] = {prb->dd, prb->dh, prb->dw};
    dnnl_dim_t padding_nd[] = {prb->pd, prb->ph, prb->pw};
    dnnl_dim_t padding_r_nd[] = {prb->pd_r, prb->ph_r, prb->pw_r};

    dnnl_dim_t *strides = strides_nd + (5 - prb->ndims);
    dnnl_dim_t *dilates = dilates_nd + (5 - prb->ndims);
    dnnl_dim_t *padding = padding_nd + (5 - prb->ndims);
    dnnl_dim_t *padding_r = padding_r_nd + (5 - prb->ndims);

    dnnl_alg_kind_t alg = dnnl_convolution_direct;
    if (prb->alg == WINO) alg = dnnl_convolution_winograd;
    if (prb->alg == AUTO) alg = dnnl_convolution_auto;

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            DNN_SAFE(dnnl_dilated_convolution_forward_desc_init(&cd,
                             prb->dir == FWD_I ? dnnl_forward_inference
                                               : dnnl_forward_training,
                             alg, &src_d, &wei_d,
                             prb->dir == FWD_B ? &bia_d : nullptr, &dst_d,
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
                             prb->dir == BWD_W ? nullptr : &bia_d, &dst_d,
                             strides, dilates, padding, padding_r),
                    WARN);
            break;
        default: DNN_SAFE(dnnl_invalid_arguments, CRIT);
    }

    DNN_SAFE(cd.accum_data_type == prb->cfg[ACC].dt ? dnnl_success
                                                    : dnnl_unimplemented,
            CRIT);

    attr_args_t attr_args;
    attr_args.prepare_output_scales(prb->attr, prb->scales, prb->oc);
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, dst_dims);
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&cpd, &cd, dnnl_attr, engine, nullptr);

    if (!res) return OK;

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    res->impl_name = query_impl_info(cpd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    SAFE(check_pd_w_and_wo_attr(res, prb->attr, cd), WARN);

    return OK;
}

int init_prim_ref(
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim_ref, const prb_t *prb) {
    if (!(is_bench_mode(CORR) && is_gpu() && fast_ref_gpu)) return OK;

    // Create a new copy of prb to avoid potentially corrupting the test by
    // modifying prb in place.
    // DIRECT algorithm is used to prevent fallback  to the slow benchdnn
    // reference implementation.
    auto cpu_attr = prb->attr;
    update_cpu_ref_attrs(cpu_attr);
    prb_t prb_cpu {*prb, prb->dir, conf_f32, tag::abx, tag::abx, tag::abx,
            DIRECT, cpu_attr, prb->mb, prb->is_deconv};
    dnnl_primitive_desc_t pd_ref_ {};
    SAFE(init_pd(get_cpu_engine(), &prb_cpu, pd_ref_, nullptr, prb->dir,
                 nullptr),
            WARN);
    auto pd_ref = make_benchdnn_dnnl_wrapper(pd_ref_);

    dnnl_primitive_t prim_ref_ {};
    if (pd_ref) {
        DNN_SAFE(dnnl_primitive_create(&prim_ref_, pd_ref), WARN);
        BENCHDNN_PRINT(
                5, "%s\n", "benchdnn: use CPU primitive as the reference");
    }
    prim_ref.reset(prim_ref_);
    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common(
            {prb->cfg[SRC].dt, prb->cfg[WEI].dt, prb->cfg[DST].dt}, prb->dir,
            res);
    if (res->state == SKIPPED) return;
    // GPU only case
    bool is_f32_wei = prb->cfg[WEI].dt == dnnl_f32;
    bool is_f32_src = prb->cfg[SRC].dt == dnnl_f32;
    bool is_int8_dst = prb->cfg[DST].dt == dnnl_s8;
    const bool f32_s8_conv = is_f32_src && is_f32_wei && is_int8_dst;

    if (is_cpu() && f32_s8_conv) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    if (is_nvidia_gpu()) {
        const int64_t ID = prb->id, IH = prb->ih, IW = prb->iw;
        const int64_t OD = prb->od, OH = prb->oh, OW = prb->ow;
        const int64_t KD = prb->kd, KH = prb->kh, KW = prb->kw;
        const int64_t SD = prb->sd, SH = prb->sh, SW = prb->sw;
        const int64_t PD = prb->pd, PH = prb->ph, PW = prb->pw;
        const int64_t PD_R = prb->pd_r, PH_R = prb->ph_r, PW_R = prb->pw_r;
        const bool pad_ok = PD >= PD_R && PH >= PH_R && PW >= PW_R;
        // copy-pasted from str2desc, dilation is not supported for Nvidia
        const auto compute_out
                = [](int64_t i, int64_t k, int64_t s, int64_t p) {
                      return (i - k + 2 * p) / s + 1;
                  };
        const bool out_ok = OD == compute_out(ID, KD, SD, PD)
                && OH == compute_out(IH, KH, SH, PH)
                && OW == compute_out(IW, KW, SW, PW);

        const auto &po = prb->attr.post_ops;
        bool post_ops_ok = true;
        for (int i = 0; i < po.len(); ++i) {
            const auto &e = po.entry[i];
            if (e.is_sum_kind())
                continue;
            else if (e.is_eltwise_kind())
                post_ops_ok = post_ops_ok && is_nvidia_eltwise_ok(prb->dir, e);
            else if (e.is_binary_kind() || e.is_convolution_kind()
                    || e.is_prelu_kind())
                post_ops_ok = false;
            else
                assert(!"unknown post-op type");
        }

        const auto dtag = normalize_tag(prb->dtag, prb->ndims);
        const bool dtag_is_axb = dtag == normalize_tag(tag::axb, prb->ndims);
        const bool tag_ok = !((prb->dir & FLAG_BWD) && dtag_is_axb);

        const bool is_f16_src = prb->cfg[SRC].dt == dnnl_f16;
        const bool is_f16_wei = prb->cfg[WEI].dt == dnnl_f16;
        const bool f16_s8_conv = is_f16_src && is_f16_wei && is_int8_dst;

        // TODO: specified wtag (even for supported formats) is not working?
        if (!pad_ok || !out_ok || !post_ops_ok || !tag_ok || f32_s8_conv
                || f16_s8_conv) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }

    // Winograd implementation limitations.
    if (prb->alg == WINO) {
        if (is_cpu()) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#if DNNL_AARCH64
            bool pad_ok_f32 = prb->pw <= 1 && prb->ph <= 1 && prb->pw_r <= 1
                    && prb->ph_r <= 1;
            bool shape_ok = prb->ndims == 4 && prb->g == 1 && prb->kh == 3
                    && prb->kw == 3 && prb->sh == 1 && prb->sw == 1
                    && prb->dh == 0 && prb->dw == 0 && !prb->has_groups
                    && pad_ok_f32;

            const bool dir_ok = (prb->dir & FLAG_FWD);

            const auto &po = prb->attr.post_ops;

            // "true" here stands for eltwise.scale == 1.f check
            const auto is_eltwise
                    = [&](int idx) { return po.entry[idx].is_eltwise_kind(); };
            auto is_sum = [&](int idx) { return po.entry[idx].is_sum_kind(); };

            const bool sum_with_eltwise
                    = (po.len() == 2) && is_sum(0) && is_eltwise(1);
            const bool eltwise_only = (po.len() == 1) ? is_eltwise(0) : false;
            bool eltwise_ok = false;

            // Compute Library supports only one eltwise post-op or
            // sum+eltwise post-ops
            if (eltwise_only || sum_with_eltwise) {
                using alg_t = attr_t::post_ops_t::kind_t;
                const std::vector<alg_t> supported_algs
                        = {alg_t::RELU, alg_t::TANH, alg_t::ELU, alg_t::SQUARE,
                                alg_t::ABS, alg_t::SQRT, alg_t::LINEAR,
                                alg_t::BRELU, alg_t::SRELU, alg_t::LOGISTIC};
                const auto act_type = po.entry[sum_with_eltwise].kind;
                eltwise_ok = std::any_of(supported_algs.cbegin(),
                        supported_algs.cend(),
                        [&](const alg_t alg) { return act_type == alg; });
            }

            const bool post_ops_ok = eltwise_ok || (po.len() == 0);

            if (!shape_ok || !dir_ok || !post_ops_ok) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }
#else
            static auto isa = dnnl_get_effective_cpu_isa();
            static bool has_avx512_bw
                    = dnnl::is_superset(isa, dnnl_cpu_isa_avx512_core);

            bool is_int8 = prb->cfg[WEI].dt == dnnl_s8;

            bool pad_ok_f32 = prb->pw <= 1 && prb->ph <= 1 && prb->pw_r <= 1
                    && prb->ph_r <= 1;
            bool pad_ok_int8 = prb->pw <= 1 && prb->ph <= 1
                    && prb->pw == prb->pw_r && prb->ph == prb->ph_r;

            bool shape_ok = prb->ndims == 4 && prb->g == 1 && prb->kh == 3
                    && prb->kw == 3 && prb->sh == 1 && prb->sw == 1
                    && prb->dh == 0 && prb->dw == 0
                    && IMPLICATION(!is_int8, pad_ok_f32)
                    && IMPLICATION(is_int8,
                            (prb->ic % 16 == 0) && (prb->oc % 16 == 0)
                                    && pad_ok_int8);
            bool bwd_is_syncable = IMPLICATION(
                    (prb->dir & FLAG_BWD), dnnl::impl::dnnl_thr_syncable());

            const auto stag = normalize_tag(prb->stag, prb->ndims);
            const bool stag_is_abx
                    = stag == normalize_tag(tag::abx, prb->ndims);
            const bool stag_is_axb
                    = stag == normalize_tag(tag::axb, prb->ndims);
            const auto dtag = normalize_tag(prb->dtag, prb->ndims);
            const bool dtag_is_abx
                    = dtag == normalize_tag(tag::abx, prb->ndims);
            const bool dtag_is_axb
                    = dtag == normalize_tag(tag::axb, prb->ndims);
            const bool is_plain
                    = stag_is_abx || stag_is_axb || dtag_is_abx || dtag_is_axb;
            const bool plain_ok = is_int8 && !stag_is_abx && !dtag_is_abx
                    && (stag_is_axb || dtag_is_axb);

            const auto &po = prb->attr.post_ops;
            const auto sum_idx = po.find(attr_t::post_ops_t::kind_t::SUM);
            const bool sum_post_op_ok
                    = sum_idx == -1 || po.entry[sum_idx].sum.scale == 1.f;

            if (!has_avx512_bw || !shape_ok || !bwd_is_syncable
                    || (is_plain && !plain_ok) || !sum_post_op_ok) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }
#endif
#endif
        } else if (is_gpu()) {
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE
            if (dnnl_impl_gpu_conv_wino_should_silence_unimplemented(
                        get_test_engine())) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }
#endif

            bool shape_ok = prb->ndims == 4 && prb->g == 1 && prb->kh == 3
                    && prb->kw == 3 && prb->sh == 1 && prb->sw == 1
                    && prb->dh == 0 && prb->dw == 0 && prb->pw < prb->kw
                    && prb->pw_r < prb->kw && prb->ph < prb->kh
                    && prb->ph_r < prb->kh;
            if (!shape_ok) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }

            const auto stag = normalize_tag(prb->stag, prb->ndims);
            const bool stag_is_axb
                    = stag == normalize_tag(tag::axb, prb->ndims);
            bool is_axb_ok = (prb->ic % 16 == 0) && (prb->oc % 16 == 0);
            if (stag_is_axb && !is_axb_ok) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }

        } else {
            assert(!"Unknown Engine");
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    check_sum_post_ops(prb->attr, res, prb->cfg[DST].dt);
    if (res->state == SKIPPED) return OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(prim, &const_pd), CRIT);

    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    alg_t alg = prb->alg;
    if (alg == AUTO) {
        dnnl_convolution_desc_t *temp_conv_desc = {nullptr};
        DNN_SAFE(dnnl_primitive_desc_query(const_pd, dnnl_query_convolution_d,
                         0, &temp_conv_desc),
                CRIT);
        alg = alg_kind2alg(temp_conv_desc->alg_kind);
    }
    const auto cfg = auto_cfg(alg, prb->cfg);
    prb_t p_new((desc_t)*prb, prb->dir, cfg, prb->stag, prb->wtag, prb->dtag,
            alg, prb->attr, prb->mb);
    prb = &p_new;

    const auto &src_md
            = prb->dir == BWD_D ? q(DNNL_ARG_DIFF_SRC) : q(DNNL_ARG_SRC);
    const auto &wei_md = prb->dir & FLAG_WEI ? q(DNNL_ARG_DIFF_WEIGHTS)
                                             : q(DNNL_ARG_WEIGHTS);
    const auto &bia_md
            = prb->dir & FLAG_WEI ? q(DNNL_ARG_DIFF_BIAS) : q(DNNL_ARG_BIAS);
    const auto &dst_md
            = prb->dir & FLAG_BWD ? q(DNNL_ARG_DIFF_DST) : q(DNNL_ARG_DST);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto src_tag = tag::abx;
    const auto wei_tag = tag::abx;

    // Use CPU prim as the reference in GPU testing to reduce testing time.
    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim_ref;
    SAFE(init_prim_ref(prim_ref, prb), WARN);

    const_dnnl_primitive_desc_t const_pd_ref;
    if (prim_ref)
        DNN_SAFE(dnnl_primitive_get_primitive_desc(prim_ref, &const_pd_ref),
                CRIT);
    const auto q_ref = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd_ref, dnnl_query_exec_arg_md, index);
    };

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = prim_ref ? get_cpu_engine() : get_test_engine();

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
    SAFE(binary::setup_binary_po(const_pd, binary_po_args, binary_po_dt,
                 binary_po_fp, ref_engine),
            WARN);
    std::vector<dnn_mem_t> prelu_po_fp, prelu_po_dt;
    std::vector<int> prelu_po_args;
    SAFE(prelu::setup_prelu_po(
                 const_pd, prelu_po_args, prelu_po_fp, prelu_po_dt, ref_engine),
            WARN);

    dnn_mem_t src_fp(src_md, fp, src_tag, ref_engine);
    dnn_mem_t wei_fp(wei_md, fp, wei_tag, ref_engine);
    dnn_mem_t dst_fp(dst_md, fp, src_tag, ref_engine);
    dnn_mem_t bia_fp(bia_md, fp, tag::x, ref_engine);
    dnn_mem_t scratchpad_fp;
    if (prim_ref)
        scratchpad_fp = dnn_mem_t(q_ref(DNNL_ARG_SCRATCHPAD), ref_engine);

    if (need_src_init(prb)) SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    if (need_dst_init(prb)) SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);
    if (need_wei_init(prb)) SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
    if (need_bia_init(prb)) SAFE(fill_bia(prb, bia_dt, bia_fp, res), WARN);

    maybe_prepare_runtime_scales(
            scales, prb->attr.oscale, prb->oc, prb->scales);
    maybe_prepare_runtime_zero_points(
            src_zero_points_m, prb->attr, DNNL_ARG_SRC, prb->ic, prb->src_zp);
    maybe_prepare_runtime_zero_points(
            dst_zero_points_m, prb->attr, DNNL_ARG_DST, prb->oc, prb->dst_zp);

    args_t args, ref_args;

    if (prb->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
        args.set(DNNL_ARG_ATTR_OUTPUT_SCALES, scales);
        args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_points_m);
        args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_points_m);
        args.set(binary_po_args, binary_po_dt);
        args.set(prelu_po_args, prelu_po_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_BIAS, bia_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_SCRATCHPAD, scratchpad_fp);
            ref_args.set(binary_po_args, binary_po_fp);
            ref_args.set(prelu_po_args, prelu_po_fp);

            TIME_REF(compute_ref(prb, ref_args, prim_ref));
            SAFE(compare_data(prb, DST, dst_dt, dst_fp, res), WARN);
        }
    } else if (prb->dir == BWD_D) {
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_DIFF_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, dst_fp);
            ref_args.set(DNNL_ARG_SCRATCHPAD, scratchpad_fp);

            TIME_REF(compute_ref(prb, ref_args, prim_ref));
            SAFE(compare_data(prb, SRC, src_dt, src_fp, res), WARN);
        }
    } else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_BIAS, bia_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, dst_fp);
            ref_args.set(DNNL_ARG_DIFF_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_DIFF_BIAS, bia_fp);
            ref_args.set(DNNL_ARG_SCRATCHPAD, scratchpad_fp);

            TIME_REF(compute_ref(prb, ref_args, prim_ref));
            SAFE(compare_data(prb, WEI, wei_dt, wei_fp, res), WARN);
            if (prb->dir & FLAG_BIA)
                SAFE(compare_data(prb, BIA, bia_dt, bia_fp, res), WARN);
        }
    } else {
        SAFE(FAIL, CRIT);
    }

    return measure_perf(res, prim, args);
}

} // namespace conv
