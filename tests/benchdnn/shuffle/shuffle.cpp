/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include <stddef.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "mkldnn.h"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"
#include "norm.hpp"

#include "shuffle/shuffle.hpp"

namespace shuffle {

inline float saturate(float value, float min, float max) {
    return MAX2(min, MIN2(max, value));
}

int fill_memory(const prb_t *p, dnn_mem_t &mem_fp, dnn_mem_t &mem_dt) {
    dnn_mem_t mem_00(mem_dt.md_, mkldnn_f32, get_default_tag(mem_dt.md_.ndims),
        engine_ref);
    dt_conf_t c_src;
    switch (p->dt) {
        case mkldnn_u8: c_src = conf_u8; break;
        case mkldnn_s8: c_src = conf_s8; break;
        case mkldnn_bf16: c_src = conf_bf16; break;
        case mkldnn_s32: c_src = conf_s32; break;
        default: c_src = conf_f32; break;
    }
    const int range = c_src.range;
    const int max = c_src.min + range - 1;

    const size_t nelems = mem_dt.nelems();
    assert(mem_dt.nelems() == mem_fp.nelems());

    for (size_t idx = 0; idx < nelems; ++idx) {
        float value = saturate((float)(idx % c_src.range), c_src.min, max);
        mem_00.set_elem(idx, value);
    }
    SAFE(mem_dt.reorder(mem_00), WARN);
    SAFE(mem_fp.reorder(mem_dt), WARN);

    return OK;
}

static int compare(const prb_t *p, const dnn_mem_t &fp_mem,
        const dnn_mem_t &dt_mem, res_t *r) {
    int64_t nelems = fp_mem.nelems();
    assert(nelems == dt_mem.nelems());
    r->errors = 0;

    for (int64_t i = 0; i < nelems; ++i) {
        const float fp = fp_mem.get_elem(i);
        const float dt = dt_mem.get_elem(i);
        const float diff = fabsf(fp - dt);
        if (r->errors < 10 && diff != 0.0) {
            printf("idx: " IFMT " fp:%f dt:%f\n", i, fp, dt);
            r->errors++;
        }
    }

    if (r->errors)
        r->state = FAILED;

    if (r->state == UNTESTED)
        r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

static int init_pd(const prb_t *p, mkldnn_shuffle_desc_t &sd,
        mkldnn_primitive_desc_t &spd, res_t *r) {

    mkldnn_memory_desc_t data_d;
    mkldnn_dims_t data_dims;
    const int ndims = (int)p->dims.size();

    for (int i = 0; i < ndims; ++i) data_dims[i] = p->dims[i];
    DNN_SAFE(mkldnn_memory_desc_init_by_tag(&data_d, ndims, data_dims, p->dt, p->tag),
           WARN);

    mkldnn_status_t init_status = mkldnn_success;
    mkldnn_primitive_desc_t hint_fwd_pd = NULL;
    if (p->dir == FWD_D) {
        auto prop = mkldnn_forward_training;
        DNN_SAFE(mkldnn_shuffle_forward_desc_init(&sd, prop,
                    &data_d, p->axis, p->group), WARN);
    } else if (p->dir == BWD_D) {
        DNN_SAFE(mkldnn_shuffle_backward_desc_init(&sd, &data_d, p->axis,
                    p->group), WARN);
    }
    init_status
            = mkldnn_primitive_desc_create(&spd, &sd, NULL, engine_tgt, NULL);
    mkldnn_primitive_desc_destroy(hint_fwd_pd);

    if (init_status == mkldnn_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(spd);
    print(5, "mkldnn implementation: %s\n", impl_str);

    return OK;
}

int doit(const prb_t *p, res_t *r) {

    res_t res_zero{};
    *r = res_zero;

    mkldnn_shuffle_desc_t sd;
    mkldnn_primitive_desc_t spd;
    mkldnn_primitive_t s{};

    SAFE(init_pd(p, sd, spd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED)
        return OK;

    DNN_SAFE(mkldnn_primitive_create(&s, spd), WARN);
    DNN_SAFE(mkldnn_primitive_desc_destroy(spd), CRIT);

    const auto fp = mkldnn_f32;
    auto &src_dt_d = sd.data_desc;

    const int ndims = (int)p->dims.size();
    const auto src_tag = get_default_tag(ndims);

    dnn_mem_t src_fp(src_dt_d, fp, src_tag, engine_ref),
            src_dt(src_dt_d, engine_tgt);
    dnn_mem_t dst_fp(src_dt_d, fp, src_tag, engine_ref),
            dst_dt(src_dt_d, engine_tgt);

    SAFE(fill_memory(p, src_dt, src_fp), WARN);

    const int i_arg = p->dir == FWD_D ? MKLDNN_ARG_SRC : MKLDNN_ARG_DIFF_DST;
    const int o_arg = p->dir == FWD_D ? MKLDNN_ARG_DST : MKLDNN_ARG_DIFF_SRC;
    args_t args;
    args.set(i_arg, src_dt.m_);
    args.set(o_arg, dst_dt.m_);

    DNN_SAFE(execute_and_wait(s, stream_tgt, args.size(), args), WARN);

    if (bench_mode & CORR) {
        compute_shuffle(p, src_fp, dst_fp);
        dnn_mem_t data(dst_dt, fp, src_tag, engine_ref);
        SAFE(compare(p, dst_fp, data, r), WARN);
    }

    if (bench_mode & PERF) {
        auto &t = r->timer;
        t.reset();
        while (true) {
            DNN_SAFE(execute_and_wait(s, stream_tgt, args.size(), args), WARN);
            t.stamp();
            const bool stop = false
                || (fix_times_per_prb && t.times() >= fix_times_per_prb)
                || (!fix_times_per_prb
                        && t.total_ms() >= max_ms_per_prb
                        && t.times() >= min_times_per_prb);
            if (stop) break;
        }
    }

    DNN_SAFE_V(mkldnn_primitive_destroy(s));

    return OK;
}

}
