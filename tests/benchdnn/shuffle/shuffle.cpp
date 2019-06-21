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

#include "mkldnn.h"

#include "src/common/mkldnn_thread.hpp"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "shuffle/shuffle.hpp"

namespace shuffle {

int fill_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    auto get_range = [](const mkldnn_data_type_t dt) {
        if (dt == mkldnn_s8 || dt == mkldnn_u8)
            return 256;
        else if (dt == mkldnn_bf16 || dt == mkldnn_f16)
            return 128;
        return 1024;
    };

    const auto nelems = mem_fp.nelems();
    const int range = get_range(p->dt);
    const int f_min = p->dt == mkldnn_u8 ? 0 : -range / 2;

    mkldnn::impl::parallel_nd(nelems, [&](int64_t i) {
            const float gen = ((97 * i) + 101) % range;
            const float value = (p->dt == mkldnn_bf16 || p->dt == mkldnn_f16)
                ? (f_min + gen) / range
                : (f_min + gen) * (1.0f + 4.0f / range);
            mem_fp.set_elem(i, maybe_saturate(p->dt, value));
        }
    );

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

static int compare(const prb_t *p, const dnn_mem_t &fp_mem,
                const dnn_mem_t &dt_mem, res_t *r) {
    const float trh = 0;
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

static int init_pd(const prb_t *p, mkldnn_shuffle_desc_t &sd,
        mkldnn_primitive_desc_t &spd, res_t *r) {
    mkldnn_memory_desc_t data_d;

    const int ndims = (int)p->dims.size();
    DNN_SAFE(mkldnn_memory_desc_init_by_tag(
                     &data_d, ndims, p->dims.data(), p->dt, p->tag),
            WARN);

    if (p->dir == FWD_D) {
        auto prop = mkldnn_forward_training;
        DNN_SAFE(mkldnn_shuffle_forward_desc_init(&sd, prop, &data_d, p->axis,
                    p->group), WARN);
    } else if (p->dir == BWD_D) {
        DNN_SAFE(mkldnn_shuffle_backward_desc_init(&sd, &data_d, p->axis,
                    p->group), WARN);
    }

    mkldnn_status_t init_status = mkldnn_success;
    init_status = mkldnn_primitive_desc_create(&spd, &sd, NULL, engine_tgt,
            NULL);

    if (init_status == mkldnn_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(spd);
    print(5, "mkldnn implementation: %s\n", impl_str);

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    mkldnn_shuffle_desc_t sd;
    mkldnn_primitive_desc_t spd;
    mkldnn_primitive_t s{};

    SAFE(init_pd(p, sd, spd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED)
        return OK;

    DNN_SAFE(mkldnn_primitive_create(&s, spd), WARN);
    DNN_SAFE(mkldnn_primitive_desc_destroy(spd), CRIT);

    auto &src_dt_d = sd.data_desc;

    const auto fp = mkldnn_f32;
    const int ndims = (int)p->dims.size();
    const auto src_tag = get_default_tag(ndims);

    dnn_mem_t src_fp(src_dt_d, fp, src_tag, engine_ref),
              src_dt(src_dt_d, engine_tgt);
    dnn_mem_t dst_fp(src_dt_d, fp, src_tag, engine_ref),
              dst_dt(src_dt_d, engine_tgt);

    SAFE(fill_src(p, src_dt, src_fp), WARN);

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

    measure_perf(r->timer, s, args);

    DNN_SAFE_V(mkldnn_primitive_destroy(s));

    return OK;
}

}
