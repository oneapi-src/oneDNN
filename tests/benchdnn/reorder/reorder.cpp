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

#include <stdlib.h>
#include <cmath>

#include "dnnl.h"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "reorder.hpp"

namespace reorder {

dnnl_status_t maybe_runtime_md(const prb_t *p, data_kind_t kind,
        const dnnl_memory_desc_t &ref_md, dnnl_memory_desc_t &runtime_md) {
    const reorder_conf_t &rc = p->reorder;

    dims_t dims = rc.dims;
    const int ndims = (int)dims.size();
    for (int d = 0; d < ndims; ++d)
        if (p->runtime_dim_mask & (1 << d)) dims[d] = DNNL_RUNTIME_DIM_VAL;

    dnnl_data_type_t dt = kind == SRC ? p->conf_in->dt : p->conf_out->dt;
    dnnl_format_tag_t tag = kind == SRC ? rc.tag_in : rc.tag_out;

    dnnl_status_t status = dnnl_memory_desc_init_by_tag(
            &runtime_md, ndims, dims.data(), dt, tag);
    if (status != dnnl_success) return status;

    runtime_md.extra = ref_md.extra;

    return dnnl_success;
}

// prepare the output scales and mask
int prepare_attr_bundle(const prb_t *p, attr_bundle_t &attr_bundle) {
    auto get_scale_mask = [](const attr_t &attr) {
        using P = attr_t::scale_t::policy_t;
        switch (attr.oscale.policy) {
            case P::PER_DIM_0: return (1 << 0);
            case P::PER_DIM_1: return (1 << 1);
            case P::PER_DIM_01: return (1 << 0) + (1 << 1);
            case P::COMMON:
            case P::NONE: return 0;
            default: SAFE_V(FAIL); return 0;
        }
    };

    const int mask = get_scale_mask(p->attr);

    int64_t uniq_scales = 1;
    for (size_t d = 0; d < p->reorder.dims.size(); ++d)
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

            print(0,
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

        bool mistrusted = reg == 0 || (check_inf_p && inf_p == 0)
                || (check_inf_n && inf_n == 0) || (check_zeros && zeros == 0);
        if (mistrusted) r->state = MISTRUSTED;
    }

    return r->state == FAILED ? FAIL : OK;
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;
    //                                       ___________________
    //                                      |                   |
    //                                      | performance timer |
    //                                      |___________________|
    //                                                |
    //   _______________           ______________     V     ________________
    //  |               |  DNNL   |              |  DNNL   |                |
    //  | dt_in fmt_ref |-------->| dt_in fmt_in |-------->| dt_out fmt_out |
    //  |_______________|         |______________|    ^    |________________|
    //           |                                    |            |
    //  benchdnn |<-------------------------------- scales         | DNNL
    //   ________V_______                                   _______V________
    //  |                |                                 |                |
    //  | dt_out fmt_ref |         <= compare =>           | dt_out fmt_ref |
    //  |________________|                                 |________________|
    //
    // Steps:
    // 1. create memory
    // 2. fill scales
    // 3. create target reorder primitive
    // 4. fill input memory
    // 5. execute DNNL and benchdnn reorders / q10n
    // 6. compare results
    // 7. performance measurement

    const reorder_conf_t &rc = p->reorder;
    const int ndims = (int)rc.dims.size();
    const int64_t *dims = &rc.dims[0];

    /* Step 1: create memory */

    /* check for extra flags on output format, and create extra information
     * descriptor for output memory. */
    dnnl_memory_extra_desc_t dst_md_extra = {};
    fill_memory_extra(p, dst_md_extra);

    dnn_mem_t src_dt_in_fmt_ref(
            ndims, dims, p->conf_in->dt, nullptr, engine_tgt);
    dnn_mem_t src_dt_in_fmt_in(
            ndims, dims, p->conf_in->dt, rc.tag_in, engine_tgt);
    dnn_mem_t dst_dt_out_fmt_out(
            ndims, dims, p->conf_out->dt, rc.tag_out, dst_md_extra, engine_tgt);
    dnn_mem_t dst_dt_out_fmt_ref(
            ndims, dims, p->conf_out->dt, nullptr, engine_tgt);

    /* Step 2: fill scales */
    attr_bundle_t attr_bundle(p->attr);
    SAFE(prepare_attr_bundle(p, attr_bundle), WARN);

    /* Step 3: create target reorder primitive */
    dnnl_primitive_desc_t rpd = NULL;
    dnnl_memory_desc_t src_md, dst_md;
    DNN_SAFE(maybe_runtime_md(p, SRC, src_dt_in_fmt_in.md_, src_md), WARN);
    DNN_SAFE(maybe_runtime_md(p, DST, dst_dt_out_fmt_out.md_, dst_md), WARN);
    dnnl_status_t init_status = dnnl_reorder_primitive_desc_create(&rpd,
            &src_md, engine_tgt, &dst_md, engine_tgt, attr_bundle.dnnl_attr());
    if (init_status == dnnl_unimplemented) {
        r->state = UNIMPLEMENTED;
    } else {
        const char *impl_str = query_impl_info(rpd);
        print(5, "dnnl implementation: %s\n", impl_str);
    }

    SAFE(init_status, WARN);

    dnnl_primitive_t rp = NULL;
    DNN_SAFE(dnnl_primitive_create(&rp, rpd), WARN);
    dnnl_primitive_desc_destroy(rpd);

    /* Step 4: fill input memory */
    SAFE(fill_memory(p, SRC, src_dt_in_fmt_ref, attr_bundle), WARN);

    /* Step 5: execute necessary reorders */
    SAFE(src_dt_in_fmt_in.reorder(src_dt_in_fmt_ref), WARN);

    if (attr_bundle.attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0) {
        SAFE(fill_memory(p, DST, dst_dt_out_fmt_ref, attr_bundle), WARN);
        SAFE(dst_dt_out_fmt_out.reorder(dst_dt_out_fmt_ref), WARN);
    }

    dnn_mem_t scales, src_zero_points_m, dst_zero_points_m;
    maybe_prepare_runtime_scales(scales, attr_bundle, engine_tgt);
    maybe_prepare_runtime_zero_points(
            src_zero_points_m, attr_bundle.attr, DNNL_ARG_SRC, engine_tgt);
    maybe_prepare_runtime_zero_points(
            dst_zero_points_m, attr_bundle.attr, DNNL_ARG_DST, engine_tgt);

    args_t args;
    args.set(DNNL_ARG_FROM, src_dt_in_fmt_in);
    args.set(DNNL_ARG_TO, dst_dt_out_fmt_out);
    args.set(DNNL_ARG_ATTR_OUTPUT_SCALES, scales);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_points_m);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_points_m);

    DNN_SAFE(execute_and_wait(rp, stream_tgt, args), WARN);

    /* Step 6: check correctness */
    if (bench_mode & CORR) {
        if (p->alg == ALG_BOOT) {
            /* "bootstrap" algorithm: compare to another dnnl reorder. use
             * this when benchdnn does not know about all details of the data
             * layout, as is the case for compensated weights formats. */

            /* Step 5a: dnnl reorder from ref format to output format */
            dnnl_memory_extra_desc_t dst_extra = {};
            fill_memory_extra(p, dst_extra);
            dnn_mem_t ref_dst_dt_out_fmt_out(ndims, dims, p->conf_out->dt,
                    rc.tag_out, dst_extra, engine_tgt);
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

            /* Step 5c: compare benchdnn and dnnl output */
            dnn_mem_t dst_dt_out(
                    ndims, dims, p->conf_out->dt, nullptr, engine_tgt);
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
