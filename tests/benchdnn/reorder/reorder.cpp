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

#include <cmath>
#include <stdlib.h>

#include "dnnl.h"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "reorder.hpp"

namespace reorder {

// prepare the output scales and mask
int prepare_attr_bundle(const prb_t *p, attr_bundle_t &attr_bundle) {
    auto get_scale_mask = [](const attr_t &attr) {
        using P = attr_t::scale_t::policy_t;
        switch (attr.oscale.policy) {
            case P::PER_DIM_0: return (1 << 0);
            case P::PER_DIM_1: return (1 << 1);
            case P::PER_DIM_01: return (1 << 0) + (1 << 1);
            case P::COMMON: return 0;
            default: SAFE_V(FAIL); return 0;
        }
    };

    const int mask = get_scale_mask(p->attr);

    int64_t uniq_scales = 1;
    for (int d = 0; d < p->ndims; ++d)
        if (mask & (1 << d)) uniq_scales *= p->reorder.dims[d];

    attr_bundle.oscale.resize(uniq_scales, p->attr.oscale.scale);
    if (uniq_scales > 1) attr_bundle.oscale[uniq_scales - 1] /= 2.f;

    return attr_bundle.generate(mask);
}

int fill_memory(const prb_t *p, data_kind_t kind, dnn_mem_t &mem,
        const attr_bundle_t &attr_bundle) {
    const dt_conf_t c_src = p->conf_in;
    const auto dt = c_src->dt;
    const int range = c_src->range;
    const int max = c_src->min + range - 1;
    const int scale_mask = attr_bundle.scale_mask();

    const auto nelems = mem.nelems();

    for (int64_t idx = 0; idx < nelems; ++idx) {
        const int64_t mask_idx = mem.get_scale_idx(idx, scale_mask);
        const float scale = attr_bundle.oscale[mask_idx];

        const float gen[7] = {
                (float)max, /* saturate to max of output data type */
                (float)c_src->min, /* saturate to min of output data type */
                (float)1.6 / scale, /* rounding check */
                (float)0.2 / scale, /* saturate to 0 */
                (float)1.0,
                (float)2.0,
                (float)scale,
        };

        const int rng = kind == SRC ? (idx % 7) : ((idx * 8 / 7) % 7);
        mem.set_elem(idx, maybe_saturate(dt, gen[rng]));
    }

    return OK;
}

int fill_memory_extra(const prb_t *p, dnnl_memory_extra_desc_t &extra) {
    extra.flags = dnnl_memory_extra_flag_none;

    if (p->alg == ALG_BOOT
            && (p->oflag == FLAG_CONV_S8S8 || p->oflag == FLAG_GCONV_S8S8)) {
        int with_groups = p->oflag == FLAG_GCONV_S8S8 ? 1 : 0;
        extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
        extra.compensation_mask = (1 << 0) + with_groups * (1 << 1);
    }

    return OK;
}

int ref_reorder(const prb_t *p, dnn_mem_t &dst, const dnn_mem_t &src,
        const attr_bundle_t &attr_bundle) {
    auto dst_dt = dst.dt();

    const auto nelems = src.nelems();

    const int scale_mask = attr_bundle.scale_mask();

    const int src_zero_point = attr_bundle.attr.zero_points[DNNL_ARG_SRC];
    const int dst_zero_point = attr_bundle.attr.zero_points[DNNL_ARG_DST];

    float beta = 0;
    const auto &po = attr_bundle.attr.post_ops;
    const int beta_idx = po.find(attr_t::post_ops_t::kind_t::SUM);
    if (beta_idx >= 0) beta = po.entry[beta_idx].sum.scale;

    for (int64_t idx = 0; idx < nelems; ++idx) {
        float s = src.get_elem(idx) - src_zero_point;
        float d = 0;
        if (beta_idx >= 0) d = dst.get_elem(idx) - dst_zero_point;

        const int64_t scale_idx = dst.get_scale_idx(idx, scale_mask);
        const float alpha = attr_bundle.oscale[scale_idx];

        dst.set_elem(idx,
                maybe_saturate(dst_dt, alpha * s + beta * d + dst_zero_point));
    }

    return OK;
}

int compare_bootstrap(dnn_mem_t &mem_ref, dnn_mem_t &mem_got, res_t *r) {
    bool ok = false;
    // demand bit-wise identical results
    const auto size_ref = mem_ref.size();
    if (size_ref == mem_got.size())
        ok = !memcmp((void *)mem_ref, (void *)mem_got, size_ref);

    r->errors = !ok;
    r->state = ok ? PASSED : FAILED;
    r->total = 1;

    return r->state == FAILED ? FAIL : OK;
}

static int compare(const prb_t *p, const dnn_mem_t &mem_ref,
        const dnn_mem_t &mem_got, const attr_bundle_t &attr_bundle, res_t *r) {
    const auto nelems = mem_got.nelems();
    r->errors = 0;
    r->total = nelems;
    int64_t inf_p = 0, inf_n = 0, zeros = 0, reg = 0;

    const auto dt_out = mem_ref.dt();
    const size_t width = mem_ref.sizeof_dt() * 8;
    const float dt_out_min
            = dt_out == dnnl_u8 ? 0.f : -(float)(1l << (width - 1));
    const float dt_out_max
            = dt_out == dnnl_u8 ? 255.f : (float)((1l << (width - 1)) - 1);
    const float tolerance = (dt_out == dnnl_bf16)
            ? 4e-3 // due to bf16 truncation (7th mantissa bit -> 1/129)
            : 0.;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = mem_got.get_elem(i);
        const float fp = mem_ref.get_elem(i);

        if (fp == dt_out_max)
            inf_p++;
        else if (fp == dt_out_min)
            inf_n++;
        else if (fp == 0.0)
            zeros++;
        else
            reg++;

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        bool ok = rel_diff <= tolerance;

        // f32->f16 results in inf for FLT_MAX input
        if (!ok) ok = std::isinf(fp) && std::isinf(dt);

        r->errors += !ok;

        const bool dump = false || (!ok && (r->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = off2dims_idx(p->reorder.dims, i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            BENCHDNN_PRINT(0,
                    "[%4ld][%s] fp:% 12.6g dt:% 12.6g diff:%8.3g rdiff:%8.3g\n",
                    (long)i, ind_str.c_str(), fp, dt, diff, rel_diff);
        }
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    if (r->state != FAILED) {
        float max_scale = attr_bundle.oscale[0];
        for (size_t i = 1; i < attr_bundle.oscale.size(); ++i)
            max_scale = MAX2(max_scale, attr_bundle.oscale[i]);

        dt_conf_t c_src = p->conf_in;
        dt_conf_t c_dst = p->conf_out;
        const int c_src_max = c_src->min + c_src->range - 1;
        const int c_dst_max = c_dst->min + c_dst->range - 1;

        bool check_int_overflow = (dt_out != dnnl_f32 && dt_out != dnnl_f16
                && dt_out != dnnl_bf16);
        bool check_inf_p = (check_int_overflow && dt_out != dnnl_s32)
                && (c_src_max * max_scale > c_dst_max);
        bool check_inf_n = (check_int_overflow && dt_out != dnnl_s32)
                && (c_src->min * max_scale < c_dst->min);
        bool check_zeros = (check_int_overflow)
                && (dt_out_min != 0 && dt_out_max != 0)
                && attr_bundle.attr.zero_points[DNNL_ARG_SRC] == 0
                && attr_bundle.attr.zero_points[DNNL_ARG_DST] == 0
                && attr_bundle.attr.post_ops.find(
                           attr_t::post_ops_t::kind_t::SUM)
                        == -1;

        bool mistrusted = (check_inf_p && inf_p == 0)
                || (check_inf_n && inf_n == 0) || (check_zeros && zeros == 0);

        bool expect_regular = max_scale < 2e9 || dt_out == dnnl_f32;
        if (expect_regular) mistrusted = mistrusted || reg == 0;

        if (mistrusted) r->state = MISTRUSTED;
    }

    return r->state == FAILED ? FAIL : OK;
}

static int init_pd_custom(dnnl_engine_t engine, const prb_t *p,
        dnnl_primitive_desc_t &rpd, const attr_bundle_t &attr_bundle,
        res_t *r) {
    const auto &rc = p->reorder;
    auto dims = rc.dims;
    for (int d = 0; d < p->ndims; ++d)
        if (p->runtime_dim_mask & (1 << d)) dims[d] = DNNL_RUNTIME_DIM_VAL;

    dnnl_memory_desc_t src_d, dst_d;
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_d, p->ndims, dims.data(),
                     p->conf_in->dt, convert_tag(rc.tag_in, p->ndims)),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_d, p->ndims, dims.data(),
                     p->conf_out->dt, convert_tag(rc.tag_out, p->ndims)),
            WARN);

    // assign extra for dst_md
    dnnl_memory_extra_desc_t dst_md_extra {};
    fill_memory_extra(p, dst_md_extra);
    dst_d.extra = dst_md_extra;

    dnnl_status_t init_status = dnnl_reorder_primitive_desc_create(
            &rpd, &src_d, engine, &dst_d, engine, attr_bundle.dnnl_attr());
    if (init_status == dnnl_unimplemented) return r->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    r->impl_name = query_impl_info(rpd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", r->impl_name.c_str());

    return OK;
}

void check_known_skipped_case(const prb_t *p, res_t *r) {
    check_known_skipped_case_common({p->conf_in->dt, p->conf_out->dt}, r);
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    check_known_skipped_case(p, r);
    if (r->state == SKIPPED) return OK;

    //                                       ___________________
    //                                      |                   |
    //                                      | performance timer |
    //                                      |___________________|
    //                                                |
    //   _______________           ______________     V     ________________
    //  |               | oneDNN  |              | oneDNN  |                |
    //  | dt_in fmt_ref |-------->| dt_in fmt_in |-------->| dt_out fmt_out |
    //  |_______________|         |______________|    ^    |________________|
    //           |                                    |            |
    //  benchdnn |<-------------------------------- scales         | oneDNN
    //   ________V_______                                   _______V________
    //  |                |                                 |                |
    //  | dt_out fmt_ref |         <= compare =>           | dt_out fmt_ref |
    //  |________________|                                 |________________|
    //
    // Steps:
    // 1. fill scales
    // 2. create target reorder primitive
    // 3. create memories
    // 4. fill input memory
    // 5. execute oneDNN and benchdnn reorders / q10n
    // 6. compare results
    // 7. performance measurement

    /* Step 1: fill scales */
    attr_bundle_t attr_bundle(p->attr);
    SAFE(prepare_attr_bundle(p, attr_bundle), WARN);

    /* Step 2: create target reorder primitive */
    dnnl_primitive_t rp {};
    // TODO: align init_pd interface with a common one which is used
    // in the rest of the benchdnn drivers
    auto init_pd = [&](dnnl_engine_t engine, const prb_t *p,
                           dnnl_primitive_desc_t &rpd, res_t *r, dir_t dir,
                           const_dnnl_primitive_desc_t hint) {
        SAFE(init_pd_custom(engine, p, rpd, attr_bundle, r), WARN);
        return OK;
    };

    SAFE(init_prim(&rp, init_pd, p, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(rp, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(rp));
        return r->state = SKIPPED, r->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    /* Step 3: create memories */
    dnnl_memory_desc_t src_md, dst_md;
    if (p->runtime_dim_mask != 0) {
        // re-create memory descriptors with defined dims
        const auto &rc = p->reorder;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_md, p->ndims, rc.dims.data(),
                         p->conf_in->dt, convert_tag(rc.tag_in, p->ndims)),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_md, p->ndims, rc.dims.data(),
                         p->conf_out->dt, convert_tag(rc.tag_out, p->ndims)),
                WARN);
    } else {
        src_md = q(DNNL_ARG_SRC);
        dst_md = q(DNNL_ARG_DST);
    }
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto tag = get_abx_tag(p->ndims);
    const auto src_dt = src_md.data_type;
    const auto dst_dt = dst_md.data_type;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_dt_in_fmt_ref(src_md, src_dt, tag, test_engine);
    dnn_mem_t src_dt_in_fmt_in(src_md, test_engine);

    dnn_mem_t dst_dt_out_fmt_ref(dst_md, dst_dt, tag, test_engine);
    dnn_mem_t dst_dt_out_fmt_out(dst_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    /* Step 4: fill input memory */
    SAFE(fill_memory(p, SRC, src_dt_in_fmt_ref, attr_bundle), WARN);

    /* Step 5: execute necessary reorders */
    SAFE(src_dt_in_fmt_in.reorder(src_dt_in_fmt_ref), WARN);

    if (attr_bundle.attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0) {
        SAFE(fill_memory(p, DST, dst_dt_out_fmt_ref, attr_bundle), WARN);
        SAFE(dst_dt_out_fmt_out.reorder(dst_dt_out_fmt_ref), WARN);
    }

    dnn_mem_t scales, src_zero_points_m, dst_zero_points_m;
    maybe_prepare_runtime_scales(scales, attr_bundle);
    maybe_prepare_runtime_zero_points(
            src_zero_points_m, attr_bundle.attr, DNNL_ARG_SRC);
    maybe_prepare_runtime_zero_points(
            dst_zero_points_m, attr_bundle.attr, DNNL_ARG_DST);

    args_t args;
    args.set(DNNL_ARG_FROM, src_dt_in_fmt_in);
    args.set(DNNL_ARG_TO, dst_dt_out_fmt_out);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(DNNL_ARG_ATTR_OUTPUT_SCALES, scales);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_points_m);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_points_m);

    SAFE(execute_and_wait(rp, args), WARN);

    /* Step 6: check correctness */
    if (bench_mode & CORR) {
        if (p->alg == ALG_BOOT) {
            /* "bootstrap" algorithm: compare to another oneDNN reorder. use
             * this when benchdnn does not know about all details of the data
             * layout, as is the case for compensated weights formats. */

            /* Step 5a: oneDNN reorder from ref format to output format */
            dnnl_memory_extra_desc_t dst_extra {};
            fill_memory_extra(p, dst_extra);
            dnn_mem_t ref_dst_dt_out_fmt_out(dst_md, test_engine);
            ref_dst_dt_out_fmt_out.md_.extra = dst_extra;

            SAFE(ref_dst_dt_out_fmt_out.reorder(src_dt_in_fmt_ref, attr_bundle),
                    WARN);

            /* Step 5b: compare results (expect bit-wise exactness) */
            SAFE(compare_bootstrap(
                         ref_dst_dt_out_fmt_out, dst_dt_out_fmt_out, r),
                    WARN);
        } else {
            /* (default) "reference" algorithm: compare to benchdnn reorder */

            /* Step 5b: execute benchdnn reorder */
            SAFE(ref_reorder(
                         p, dst_dt_out_fmt_ref, src_dt_in_fmt_ref, attr_bundle),
                    WARN);

            /* Step 5c: compare benchdnn and oneDNN output */
            dnn_mem_t dst_dt_out(dst_md, dst_dt, tag, test_engine);
            SAFE(dst_dt_out.reorder(dst_dt_out_fmt_out), WARN);
            SAFE(compare(p, dst_dt_out_fmt_ref, dst_dt_out, attr_bundle, r),
                    WARN);
        }
    }

    /* Step 7: performance measurement */
    measure_perf(r->timer, rp, args);

    DNN_SAFE_V(dnnl_primitive_destroy(rp));

    return OK;
}

} // namespace reorder
