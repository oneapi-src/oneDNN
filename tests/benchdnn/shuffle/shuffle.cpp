/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "dnnl.h"

#include "src/common/dnnl_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "shuffle/shuffle.hpp"

namespace shuffle {

int fill_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    auto get_range = [](const dnnl_data_type_t dt) {
        if (dt == dnnl_s8 || dt == dnnl_u8)
            return 256;
        else if (dt == dnnl_bf16 || dt == dnnl_f16)
            return 128;
        return 1024;
    };

    const auto nelems = mem_fp.nelems();
    const int range = get_range(p->dt);
    const int f_min = p->dt == dnnl_u8 ? 0 : -range / 2;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) + 101) % range;
        const float value = (p->dt == dnnl_bf16 || p->dt == dnnl_f16)
                ? (f_min + gen) / range
                : (f_min + gen) * (1.0f + 4.0f / range);
        mem_fp.set_elem(i, maybe_saturate(p->dt, value));
    });

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

        const bool dump = false || (!ok && (r->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = off2dims_idx(p->dims, i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            BENCHDNN_PRINT(0, "[%4ld][%s] fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, ind_str.c_str(), fp, dt, diff, rel_diff);
        }
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

static int init_pd(const engine_t &engine_tgt, const prb_t *p,
        dnnl_primitive_desc_t &spd, res_t *r, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_memory_desc_t data_d;
    dnnl_shuffle_desc_t sd;

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&data_d, p->ndims, p->dims.data(),
                     p->dt, convert_tag(p->tag, p->ndims)),
            WARN);

    auto prop_kind = p->dir & FLAG_INF ? dnnl_forward_inference
                                       : dnnl_forward_training;
    DNN_SAFE(dnnl_shuffle_forward_desc_init(
                     &sd, prop_kind, &data_d, p->axis, p->group),
            WARN);

    dnnl_primitive_desc_t _hint = NULL;
    if (p->dir & FLAG_BWD) {
        dnnl_status_t init_fwd_status = dnnl_primitive_desc_create(
                &_hint, &sd, NULL, engine_tgt, NULL);
        if (init_fwd_status == dnnl_unimplemented)
            return r->state = UNIMPLEMENTED, OK;
        SAFE(init_fwd_status, WARN);

        DNN_SAFE(dnnl_memory_desc_init_by_tag(&data_d, p->ndims, p->dims.data(),
                         p->dt, dnnl_format_tag_any),
                WARN);

        DNN_SAFE(dnnl_shuffle_backward_desc_init(
                         &sd, &data_d, p->axis, p->group),
                WARN);
    }

    auto dnnl_attr = create_dnnl_attr(attr_t());

    dnnl_status_t init_status = dnnl_primitive_desc_create(
            &spd, &sd, dnnl_attr, engine_tgt, _hint);

    dnnl_primitive_desc_destroy(_hint);
    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented) return r->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(spd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", impl_str);

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;
    engine_t engine_tgt;

    dnnl_primitive_t s;
    SAFE(init_prim(&s, init_pd, engine_tgt, p, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(s, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(s));
        return r->state = SKIPPED, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto &data_md
            = p->dir & FLAG_FWD ? q(DNNL_ARG_SRC) : q(DNNL_ARG_DIFF_SRC);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = get_abx_tag(p->ndims);

    dnn_mem_t src_fp(data_md, fp, tag, engine_tgt);
    dnn_mem_t src_dt(data_md, engine_tgt);

    dnn_mem_t dst_fp(data_md, fp, tag, engine_tgt);
    dnn_mem_t dst_dt(data_md, engine_tgt);

    dnn_mem_t scratchpad_dt(scratchpad_md, engine_tgt);

    SAFE(fill_src(p, src_dt, src_fp), WARN);

    const int i_arg = p->dir == FWD_D ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST;
    const int o_arg = p->dir == FWD_D ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC;

    args_t args;

    args.set(i_arg, src_dt);
    args.set(o_arg, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    DNN_SAFE(execute_and_wait(s, engine_tgt, args), WARN);

    if (bench_mode & CORR) {
        compute_shuffle(p, src_fp, dst_fp);
        dnn_mem_t data(dst_dt, fp, tag, engine_tgt);
        SAFE(compare(p, dst_fp, data, r), WARN);
    }

    measure_perf(r->timer, engine_tgt, s, args);

    DNN_SAFE_V(dnnl_primitive_destroy(s));

    return OK;
}

} // namespace shuffle
