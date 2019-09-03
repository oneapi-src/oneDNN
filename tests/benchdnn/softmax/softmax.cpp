/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "softmax/softmax.hpp"

namespace softmax {

const int64_t global_fill_range = 200;

static int init_pd(const prb_t *p, dnnl_softmax_desc_t &sd,
        dnnl_primitive_desc_t &spd, res_t *r) {
    dnnl_memory_desc_t data_d;

    const int ndims = (int)p->dims.size();
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &data_d, ndims, p->dims.data(), p->dt, p->tag),
            WARN);

    if (p->dir & FLAG_FWD) {
        auto prop = p->dir & FLAG_INF ? dnnl_forward_inference
                                      : dnnl_forward_training;

        DNN_SAFE(dnnl_softmax_forward_desc_init(&sd, prop, &data_d, p->axis),
                WARN);
    } else {
        dnnl_memory_desc_t diff_data_d;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_data_d, ndims,
                         p->dims.data(), p->dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_softmax_backward_desc_init(
                         &sd, &diff_data_d, &data_d, p->axis),
                WARN);
    }

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&spd, &sd, NULL, engine_tgt, NULL);

    if (init_status == dnnl_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(spd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: dnnl implementation: %s\n", impl_str);
        DNN_SAFE(dnnl_primitive_desc_destroy(spd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "dnnl implementation: %s\n", impl_str);
    }

    return OK;
}

static int compare(const prb_t *p, const dnn_mem_t &fp_mem,
        const dnn_mem_t &dt_mem, res_t *r) {
    // FWD
    // When axis_size is big, significant values will be only in points with the
    // biggest values are. So we adjust machine epsilon to the amount of such
    // inputs, which is equally distibuted by fill_data(), thus, dividing
    // axis_size by global_fill_range gives that number.
    // When axis_size is small, significant values are coming not only from the
    // biggest input, but also from some smaller. For this case we estimate the
    // amount of such number by log2f(axis_size).
    // exp accuracy is expected to be no lower than 1e-6, so we bump the
    // coefficient to 10.
    // The final criterion picks the max of these numbers.
    // BWD
    // We have sum over axis dim, the worst case for error is amount of elements
    // times machine eps and additional subtract.
    const float num_significant_values
            = MAX2(div_up(p->dims[p->axis], global_fill_range),
                    MAX2(log2f(p->dims[p->axis]), 10));
    const int f32_mant_digits = 24;
    const float trh_coeff = (1 << (f32_mant_digits - digits_dt(p->dt)));
    const float trh = trh_coeff * 1e-7
            * (p->dir & FLAG_FWD ? num_significant_values
                                 : (p->dims[p->axis] + 1));

    const auto nelems = dt_mem.nelems();
    r->errors = 0;
    r->total = nelems;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = dt_mem.get_elem(i);
        const float fp = fp_mem.get_elem(i);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= trh;

        r->errors += !ok;

        const bool dump = false || (!ok && (r->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = off2dims_idx(p->dims, i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            print(0, "[%4ld][%s] fp:%8g dt:%8g diff:%8g rdiff:%8g\n", (long)i,
                    ind_str.c_str(), fp, dt, diff, rel_diff);
        }
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int fill_data_fwd(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        int64_t mb {0}, c {0};
        map_off_to_mb_ic(p, i, mb, c);

        const float f_min = ((p->axis > 0) ? mb : c) % 2 == 0
                ? -global_fill_range
                : -global_fill_range / 2;
        const float gen = ((11 * i) + 37) % global_fill_range;
        const float value = f_min + gen;
        mem_fp.set_elem(i, value);
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_data_bwd(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();

    // keep all values negative to have sum and sub of same sign, avoiding
    // cancellation error.
    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((11 * i) + 37) % global_fill_range;
        const float value = -gen / global_fill_range;
        mem_fp.set_elem(i, value);
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    dnnl_softmax_desc_t sd;
    dnnl_primitive_desc_t spd;
    dnnl_primitive_t s;

    SAFE(init_pd(p, sd, spd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    DNN_SAFE(dnnl_primitive_create(&s, spd), WARN);
    DNN_SAFE(dnnl_primitive_desc_destroy(spd), CRIT);

    const auto fp = dnnl_f32;
    const auto tag = get_default_tag((int)p->dims.size());
    auto &data_desc = sd.data_desc;
    dnn_mem_t src_fp(data_desc, fp, tag, engine_tgt),
            src_dt(data_desc, engine_tgt);

    dnn_mem_t dst_fp(data_desc, fp, tag, engine_tgt);
    dnn_mem_t dst_dt;
    if (!p->inplace) {
        dst_dt = dnn_mem_t(data_desc, engine_tgt);
        SAFE(dst_dt.reorder(dst_fp), WARN);
    }

    dnn_mem_t d_dst_dt, d_src_dt;
    dnn_mem_t d_dst_fp, d_src_fp;

    args_t args;

    if (p->dir & FLAG_FWD) {
        SAFE(fill_data_fwd(p, src_dt, src_fp), WARN);

        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DST, p->inplace ? src_dt : dst_dt);

        DNN_SAFE(execute_and_wait(s, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(p, src_fp, dst_fp);
            dnn_mem_t dst(p->inplace ? src_dt : dst_dt, fp, tag, engine_tgt);
            SAFE(compare(p, dst_fp, dst, r), WARN);
        }
    } else {
        const_dnnl_primitive_desc_t const_spd;
        DNN_SAFE(dnnl_primitive_get_primitive_desc(s, &const_spd), CRIT);
        const auto &d_data_desc = *dnnl_primitive_desc_query_md(
                const_spd, dnnl_query_diff_src_md, 0);

        d_dst_fp = dnn_mem_t(d_data_desc, fp, tag, engine_tgt),
        d_dst_dt = dnn_mem_t(d_data_desc, engine_tgt);

        d_src_fp = dnn_mem_t(d_data_desc, fp, tag, engine_tgt);
        if (!p->inplace) {
            d_src_dt = dnn_mem_t(d_data_desc, engine_tgt);
            SAFE(d_src_dt.reorder(d_src_fp), WARN);
        }

        SAFE(fill_data_bwd(p, src_dt, src_fp), WARN);
        SAFE(fill_data_bwd(p, d_dst_dt, d_dst_fp), WARN);

        args.set(DNNL_ARG_DST, src_dt);
        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, p->inplace ? d_dst_dt : d_src_dt);

        DNN_SAFE(execute_and_wait(s, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, src_fp, d_dst_fp, d_src_fp);
            dnn_mem_t d_src(
                    p->inplace ? d_dst_dt : d_src_dt, fp, tag, engine_tgt);
            SAFE(compare(p, d_src_fp, d_src, r), WARN);
        }
    }

    measure_perf(r->timer, s, args);

    DNN_SAFE(dnnl_primitive_destroy(s), CRIT);

    return OK;
}

} // namespace softmax
