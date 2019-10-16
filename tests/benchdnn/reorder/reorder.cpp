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

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "reorder.hpp"

namespace reorder {

int get_scale_mask(const attr_t &attr) {
    using P = attr_t::scale_t::policy_t;
    const auto policy = attr.oscale.policy;

    int scale_mask = 0;

    switch (policy) {
        case P::PER_DIM_0: scale_mask = (1 << 0); break;
        case P::PER_DIM_1: scale_mask = (1 << 1); break;
        case P::PER_DIM_01: scale_mask = (1 << 0) + (1 << 1); break;
        case P::COMMON:
        case P::NONE: scale_mask = 0; break;
        default: SAFE_V(FAIL);
    }

    return scale_mask;
}

int scales_count(const prb_t *p, int64_t *count, int *mask) {
    const int scale_mask = get_scale_mask(p->attr);
    if (mask) *mask = scale_mask;

    int64_t uniq_scales = 1;
    for (size_t d = 0; d < p->reorder.dims.size(); ++d) {
        if (scale_mask & (1 << d)) uniq_scales *= p->reorder.dims[d];
    }
    *count = uniq_scales;
    return OK;
}

int fill_scales(const prb_t *p, float *scales, int64_t count) {
    const float scale_value = p->attr.oscale.scale;

    for (int64_t i = 0; i < count; ++i)
        scales[i] = scale_value;

    if (count != 1) scales[count - 1] = scale_value + 1.1;

    return OK;
}

int fill_memory(const prb_t *p, dnn_mem_t &mem, const float *scales,
        const attr_t &attr) {
    const dt_conf_t c_src = p->conf_in;
    const auto dt = c_src->dt;
    const int range = c_src->range;
    const int max = c_src->min + range - 1;
    int scale_mask = get_scale_mask(attr);

    const auto nelems = mem.nelems();

    for (int64_t idx = 0; idx < nelems; ++idx) {
        const int64_t mask_idx = mem.get_scale_idx(idx, scale_mask);
        const float scale = scales[mask_idx];

        const float gen[7] = {
                (float)max, /* saturate to max of output data type */
                (float)c_src->min, /* saturate to min of output data type */
                (float)1.6 / scale, /* rounding check */
                (float)0.2 / scale, /* saturate to 0 */
                (float)1.0,
                (float)2.0,
                (float)scale,
        };

        mem.set_elem(idx, maybe_saturate(dt, gen[idx % 7]));
    }

    return OK;
}

/* TODO: Complete */
int reorder(const prb_t *p, dnn_mem_t &dst, const dnn_mem_t &src,
        const float *scales) {
    auto dst_dt = dst.dt();

    const auto nelems = src.nelems();

    /* TODO: add dst range support */
    //    const auto c_dst = p->conf_out;
    //    const float dst_conf_min = c_dst.min;
    //    const float dst_conf_max = dst_conf_min + c_dst.range - 1;
    //    const float dst_max = MIN2(dst_conf_max, dst_dt_max);
    //    const float dst_min = MAX2(dst_conf_min, dst_dt_min);

    const int scale_mask = get_scale_mask(p->attr);

    for (int64_t idx = 0; idx < nelems; ++idx) {
        float src_ = src.get_elem(idx);
        const int64_t scale_idx = dst.get_scale_idx(idx, scale_mask);
        const float scale = scales[scale_idx];

        dst.set_elem(idx, maybe_saturate(dst_dt, src_ * scale));
    }

    return OK;
}

int compare_bootstrap(
        dnn_mem_t &mem_expected, dnn_mem_t &mem_computed, res_t *r) {
    int diff = 0;
    // demand bit-wise identical results
    size_t expected_size = mem_expected.size();
    size_t computed_size = mem_computed.size();
    if (expected_size == computed_size)
        diff = memcmp(
                (void *)mem_expected, (void *)mem_computed, expected_size);
    else
        diff = 1;
    // set results and check state for failure
    r->errors = diff == 0 ? 0 : 1;
    r->state = diff == 0 ? PASSED : FAILED;
    r->total = 1;
    return r->state == FAILED ? FAIL : OK;
}

int compare(const prb_t *p, dnn_mem_t &mem_expected, dnn_mem_t &mem_computed,
        const float *scales, int64_t count, res_t *r) {
    const auto nelems = mem_expected.nelems();
    assert(nelems == mem_computed.nelems());

    r->errors = 0;
    r->total = nelems;

    /* TODO: range support */
    const auto dt = mem_expected.dt();
    const size_t width = mem_expected.sizeof_dt() * 8;

    const float dt_min = dt == dnnl_u8 ? 0.f : -(float)(1l << (width - 1));
    const float dt_max
            = dt == dnnl_u8 ? 255.f : (float)((1l << (width - 1)) - 1);

    int64_t inf_p = 0, inf_n = 0, zeros = 0, reg = 0;

    const float tolerance = mem_computed.dt() == dnnl_bf16
            ? 8.e-3f // due to bf16 truncation (7th mantissa bit -> 1/129)
            : 0.0f;
    for (int64_t i = 0; i < nelems; ++i) {
        const float expected = mem_expected.get_elem(i);
        const float computed = mem_computed.get_elem(i);
        const float diff
                = fabsf(computed - expected) / MAX2(FLT_MIN, fabsf(expected));

        if (expected == dt_max)
            inf_p++;
        else if (expected == dt_min)
            inf_n++;
        else if (expected == 0.0)
            zeros++;
        else
            reg++;

        if (r->errors < 10 && diff > tolerance) {
            printf("idx: " IFMT " exp: %f com:%f\n", i, expected, computed);
            r->errors++;
        }
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    float max_scale = scales[0];
    for (int64_t i = 1; i < count; ++i) {
        if (scales[i] > max_scale) max_scale = scales[i];
    }

    dt_conf_t c_src = p->conf_in;
    dt_conf_t c_dst = p->conf_out;
    const int c_src_max = c_src->min + c_src->range - 1;
    const int c_dst_max = c_dst->min + c_dst->range - 1;

    bool check_int_overflow
            = (dt != dnnl_f32 && dt != dnnl_f16 && dt != dnnl_bf16);
    bool check_inf_p = (check_int_overflow && dt != dnnl_s32)
            && (c_src_max * max_scale > c_dst_max);
    bool check_inf_n = (check_int_overflow && dt != dnnl_s32)
            && (c_src->min * max_scale < c_dst->min);
    bool check_zeros = (check_int_overflow) && (dt_min != 0 && dt_max != 0);

    bool mistrusted = reg == 0 || (check_inf_p && inf_p == 0)
            || (check_inf_n && inf_n == 0) || (check_zeros && zeros == 0);
    if (mistrusted) r->state = MISTRUSTED;

    return r->state == FAILED ? FAIL : OK;
}

int check_reorder(const prb_t *p, res_t *res) {
    /*                                       ___________________
 *                                      |                   |
 *                                      | performance timer |
 *                                      |___________________|
 *                                                |
 *   _______________           ______________     V     ________________
 *  |               | DNNL |              | DNNL |                |
 *  | dt_in fmt_ref |-------->| dt_in fmt_in |-------->| dt_out fmt_out |
 *  |_______________|         |______________|    ^    |________________|
 *           |                                    |            |
 *  benchdnn |<-------------------------------- scales         | DNNL
 *   ________V_______                                   _______V________
 *  |                |                                 |                |
 *  | dt_out fmt_ref |         <= compare =>           | dt_out fmt_ref |
 *  |________________|                                 |________________|
 *
 * Steps:
 * 1. create memory
 * 2. fill scales
 * 3. fill input memory
 * 4. execute DNNL: reorder->q10n->reorder
 * 5. execute benchdnn: q10n
 * 6. compare results
 * 7. performance measurment
 * 8. clean up
 */

    const reorder_conf_t &r = p->reorder;
    const int ndims = (int)r.dims.size();
    const int64_t *dims = &r.dims[0];

    /* Step 1: create memory */

    /* check for extra flags on output format, and create extra information
     * descriptor for output memory. */
    dnnl_memory_extra_desc_t mem_extra_dt_out_fmt_out = {};
    dnnl_memory_extra_desc_t mem_extra_test_dt_out_fmt_out = {};

    mem_extra_dt_out_fmt_out.flags = dnnl_memory_extra_flag_none;
    mem_extra_test_dt_out_fmt_out.flags = dnnl_memory_extra_flag_none;

    if (p->alg == ALG_BOOT
            && (p->oflag == FLAG_CONV_S8S8 || p->oflag == FLAG_GCONV_S8S8)) {
        int with_groups = p->oflag == FLAG_GCONV_S8S8 ? 1 : 0;
        auto set_oflags = [=](dnnl_memory_extra_desc_t &extra) {
            extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            extra.compensation_mask = (1 << 0) + with_groups * (1 << 1);
        };
        set_oflags(mem_extra_dt_out_fmt_out);
        set_oflags(mem_extra_test_dt_out_fmt_out);
    }

    dnn_mem_t mem_dt_in_fmt_ref(
            ndims, dims, p->conf_in->dt, nullptr, engine_tgt);
    dnn_mem_t mem_dt_in_fmt_in(
            ndims, dims, p->conf_in->dt, r.tag_in, engine_tgt);
    dnn_mem_t mem_dt_out_fmt_out(ndims, dims, p->conf_out->dt, r.tag_out,
            mem_extra_dt_out_fmt_out, engine_tgt);
    dnn_mem_t mem_dt_out_fmt_ref(
            ndims, dims, p->conf_out->dt, nullptr, engine_tgt);
    dnn_mem_t mem_test_dt_out_fmt_ref(
            ndims, dims, p->conf_out->dt, nullptr, engine_tgt);
    dnn_mem_t mem_test_dt_out_fmt_out(ndims, dims, p->conf_out->dt, r.tag_out,
            mem_extra_test_dt_out_fmt_out, engine_tgt);

    /* Step 2: fill scales */
    int64_t count = 0;
    int mask = 0;
    SAFE(scales_count(p, &count, &mask), WARN);
    float *scales = (float *)zmalloc(sizeof(float) * count, 64);
    SAFE(scales != NULL ? OK : FAIL, CRIT);
    SAFE(fill_scales(p, scales, count), WARN);

    auto dnnl_attr = create_dnnl_attr(p->attr, count, mask, scales);

    /* check for extra flags on output format, and set them */
    if (p->alg == ALG_BOOT
            && (p->oflag == FLAG_CONV_S8S8 || p->oflag == FLAG_GCONV_S8S8)) {
        int with_groups = p->oflag == FLAG_GCONV_S8S8 ? 1 : 0;
        auto set_oflags = [=](dnn_mem_t &m) {
            m.md_.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            m.md_.extra.compensation_mask = (1 << 0) + with_groups * (1 << 1);
        };
        set_oflags(mem_dt_out_fmt_out);
        set_oflags(mem_test_dt_out_fmt_out);
    }

    /* Step 3: check creation of reorder primitive */
    dnnl_primitive_desc_t rpd;
    dnnl_status_t init_status
            = dnnl_reorder_primitive_desc_create(&rpd, &mem_dt_in_fmt_in.md_,
                    engine_tgt, &mem_dt_out_fmt_out.md_, engine_tgt, dnnl_attr);
    if (init_status == dnnl_unimplemented) {
        res->state = UNIMPLEMENTED;
        goto cleanup;
    } else {
        const char *impl_str = query_impl_info(rpd);
        print(5, "dnnl implementation: %s\n", impl_str);
    }

    dnnl_primitive_desc_destroy(rpd);
    SAFE(init_status, WARN);

    /* Step 4: fill input memory */
    SAFE(fill_memory(p, mem_dt_in_fmt_ref, scales, p->attr), WARN);

    /* Step 5: execute DNNL */
    SAFE(mem_dt_in_fmt_in.reorder(mem_dt_in_fmt_ref), WARN);

    SAFE(mem_dt_out_fmt_out.reorder(mem_dt_in_fmt_in, dnnl_attr), WARN);

    /* Step 6: check correctness */
    if (bench_mode & CORR) {
        if (p->alg == ALG_BOOT) {
            /* "bootstrap" algorithm: compare to another dnnl reorder. use
             * this when benchdnn does not know about all details of the data
             * layout, as is the case for compensated weights formats. */

            /* Step 5a: dnnl reorder from ref format to output format */
            SAFE(mem_test_dt_out_fmt_out.reorder(mem_dt_in_fmt_ref, dnnl_attr),
                    WARN);

            /* Step 5b: compare results (expect bit-wise exactness) */
            SAFE(compare_bootstrap(
                         mem_test_dt_out_fmt_out, mem_dt_out_fmt_out, res),
                    WARN);
        } else {
            /* (default) "reference" algorithm: compare to benchdnn reorder */

            /* Step 5a: reorder output from dnnl to ref format using dnnl */
            SAFE(mem_dt_out_fmt_ref.reorder(mem_dt_out_fmt_out), WARN);

            /* Step 5b: execute benchdnn reorder */
            SAFE(reorder(p, mem_test_dt_out_fmt_ref, mem_dt_in_fmt_ref, scales),
                    WARN);

            /* Step 5c: compare benchdnn and dnnl output */
            SAFE(compare(p, mem_test_dt_out_fmt_ref, mem_dt_out_fmt_ref, scales,
                         count, res),
                    WARN);
        }
    }

    /* Step 7: performance measurement */
    if (bench_mode & PERF) {
        dnnl_primitive_desc_t perf_r_pd;
        DNN_SAFE(dnnl_reorder_primitive_desc_create(&perf_r_pd,
                         &mem_dt_in_fmt_in.md_, engine_tgt,
                         &mem_dt_out_fmt_out.md_, engine_tgt, dnnl_attr),
                WARN);

        dnnl_primitive_t perf_r;
        DNN_SAFE(dnnl_primitive_create(&perf_r, perf_r_pd), WARN);
        DNN_SAFE_V(dnnl_primitive_desc_destroy(perf_r_pd));

        args_t args;
        args.set(DNNL_ARG_FROM, mem_dt_in_fmt_in);
        args.set(DNNL_ARG_TO, mem_dt_out_fmt_out);

        measure_perf(res->timer, perf_r, args);

        DNN_SAFE_V(dnnl_primitive_destroy(perf_r));
    }

    /* Step 8: clean up */
cleanup:
    dnnl_primitive_attr_destroy(dnnl_attr);
    zfree(scales);

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    return check_reorder(p, r);
}

} // namespace reorder
