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

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "mkldnn.h"

#include "src/common/mkldnn_thread.hpp"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "softmax/softmax.hpp"

namespace softmax {

const int64_t global_fill_range = 200;

static int init_pd(const prb_t *p, mkldnn_softmax_desc_t &sd,
        mkldnn_primitive_desc_t &spd, res_t *r) {
    mkldnn_memory_desc_t data_d;
    mkldnn_dims_t data_dims;
    const int ndims = (int)p->dims.size();
    for (int i = 0; i < ndims; ++i)
        data_dims[i] = p->dims[i];

    DNN_SAFE(mkldnn_memory_desc_init_by_tag(&data_d, ndims, data_dims,
                p->dt, p->tag), WARN);

    if (p->dir & FLAG_FWD) {
        auto prop = p->dir & FLAG_INF
            ? mkldnn_forward_inference : mkldnn_forward_training;

        DNN_SAFE(mkldnn_softmax_forward_desc_init(&sd, prop, &data_d, p->axis),
                WARN);
    } else {
        DNN_SAFE(mkldnn_softmax_backward_desc_init(&sd, &data_d, &data_d,
                    p->axis), WARN);
    }

    mkldnn_status_t init_status = mkldnn_primitive_desc_create(&spd, &sd,
            NULL, engine_tgt, NULL);

    const char *impl_str = query_impl_info(spd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: mkldnn implementation: %s\n", impl_str);
        DNN_SAFE(mkldnn_primitive_desc_destroy(spd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "mkldnn implementation: %s\n", impl_str);
    }

    if (init_status == mkldnn_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

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
    // TODO: sse41 exp has lower accuracy, so for now we take max(log2(x), 10).
    // 10 is taken empirically considering it's close to log2 value.
    // The final criterion picks the max of these numbers.
    // BWD
    // We have sum over axis dim, the worst case for error is amount of elements
    // times machine eps and additional subtract.
    const float num_significant_values =
        MAX2(div_up(p->dims[p->axis], global_fill_range),
                MAX2(log2f(p->dims[p->axis]), 10));
    const float trh = 1e-7 * (p->dir & FLAG_FWD
        ? num_significant_values : (p->dims[p->axis] + 1));

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

        const bool dump = false
            || (!ok && (r->errors < 10 || verbose >= 10))
            || (verbose >= 50 && i < 30)
            || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = off2dims_idx(p->dims, i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            print(0, "[%4ld][%s] fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, ind_str.c_str(), fp, dt, diff, rel_diff);
        }
    }

    if (r->errors)
        r->state = FAILED;

    if (r->state == UNTESTED)
        r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int fill_data_fwd(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();

    mkldnn::impl::parallel_nd(nelems, [&](int64_t i) {
            int64_t mb{0}, c{0};
            map_off_to_mb_ic(p, i, mb, c);

            const float f_min = ((p->axis > 0) ? mb : c) % 2 == 0
                ? -global_fill_range
                : -global_fill_range / 2;
            const float gen = ((11 * i) + 37) % global_fill_range;
            const float value = f_min + gen;
            mem_fp.set_elem(i, value);
        }
    );

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_data_bwd(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();

    // keep all values negative to have sum and sub of same sign, avoiding
    // cancellation error.
    mkldnn::impl::parallel_nd(nelems, [&](int64_t i) {
            const float gen = ((11 * i) + 37) % global_fill_range;
            const float value = -gen / global_fill_range;
            mem_fp.set_elem(i, value);
        }
    );

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    mkldnn_softmax_desc_t sd;
    mkldnn_primitive_desc_t spd;
    mkldnn_primitive_t s;

    SAFE(init_pd(p, sd, spd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED)
        return OK;

    DNN_SAFE(mkldnn_primitive_create(&s, spd), WARN);
    DNN_SAFE(mkldnn_primitive_desc_destroy(spd), CRIT);

    const auto fp = mkldnn_f32;
    const auto tag = get_default_tag((int)p->dims.size());
    auto &data_desc = sd.data_desc;
    dnn_mem_t data_fp(data_desc, fp, tag, engine_ref),
              data_dt(data_desc, engine_tgt);

    auto &diff_desc = sd.diff_desc;
    dnn_mem_t d_data_fp(diff_desc, fp, tag, engine_ref),
              d_data_dt(diff_desc, engine_tgt);

    args_t args;

    if (p->dir & FLAG_FWD) {
        SAFE(fill_data_fwd(p, data_dt, data_fp), WARN);

        args.set(MKLDNN_ARG_SRC, data_dt.m_);
        args.set(MKLDNN_ARG_DST, data_dt.m_);

        DNN_SAFE(execute_and_wait(s, stream_tgt, args.size(), args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(p, data_fp, data_fp);
            dnn_mem_t data(data_dt, fp, tag, engine_ref);
            SAFE(compare(p, data_fp, data, r), WARN);
        }
    } else {
        SAFE(fill_data_bwd(p, data_dt, data_fp), WARN);
        SAFE(fill_data_bwd(p, d_data_dt, d_data_fp), WARN);

        args.set(MKLDNN_ARG_DST, data_dt.m_);
        args.set(MKLDNN_ARG_DIFF_DST, d_data_dt.m_);
        args.set(MKLDNN_ARG_DIFF_SRC, d_data_dt.m_);

        DNN_SAFE(execute_and_wait(s, stream_tgt, args.size(), args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, data_fp, d_data_fp, d_data_fp);
            dnn_mem_t data(d_data_dt, fp, tag, engine_ref);
            SAFE(compare(p, data, d_data_fp, r), WARN);
        }
    }

    if (bench_mode & PERF) {
        auto &t = r->timer;
        t.reset();
        while (true) {
            DNN_SAFE(execute_and_wait(s, stream_tgt, args.size(), args),
                    WARN);
            t.stamp();
            const bool stop = false
                || (fix_times_per_prb && t.times() >= fix_times_per_prb)
                || (!fix_times_per_prb
                        && t.total_ms() >= max_ms_per_prb
                        && t.times() >= min_times_per_prb);
            if (stop) break;
        }
    }

    DNN_SAFE(mkldnn_primitive_destroy(s), CRIT);

    return OK;
}

}
