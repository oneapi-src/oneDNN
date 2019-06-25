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

#include "mkldnn.h"

#include "src/common/mkldnn_thread.hpp"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "eltwise/eltwise.hpp"

namespace eltwise {

static int init_pd(const prb_t *p, mkldnn_eltwise_desc_t &ed,
        mkldnn_primitive_desc_t &epd, res_t *r) {
    mkldnn_memory_desc_t data_d;

    const int ndims = (int)p->dims.size();

    DNN_SAFE(mkldnn_memory_desc_init_by_tag(
                     &data_d, ndims, p->dims.data(), p->dt, p->tag),
            WARN);

    mkldnn_alg_kind_t alg = attr_t::post_ops_t::kind2mkldnn_kind(p->alg);

    if (p->dir & FLAG_FWD) {
        auto prop = p->dir & FLAG_INF
            ? mkldnn_forward_inference : mkldnn_forward_training;

        DNN_SAFE(mkldnn_eltwise_forward_desc_init(&ed, prop, alg, &data_d,
                    p->alpha, p->beta), WARN);
    } else {
        DNN_SAFE(mkldnn_eltwise_backward_desc_init(&ed, alg, &data_d, &data_d,
                    p->alpha, p->beta), WARN);
    }

    mkldnn_status_t init_status = mkldnn_primitive_desc_create(&epd, &ed,
            NULL, engine_tgt, NULL);

    if (init_status == mkldnn_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(epd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: mkldnn implementation: %s\n", impl_str);
        DNN_SAFE(mkldnn_primitive_desc_destroy(epd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "mkldnn implementation: %s\n", impl_str);
    }

    return OK;
}

static int compare(const prb_t *p, const dnn_mem_t &mem_fp,
        const dnn_mem_t &mem_dt, res_t *r) {
    float trh = 1e-6;
    if (p->alg == alg_t::GELU) // when x < -3 (tanh(g(x)) + 1) has cancellation
        trh = 4e-6;            // subtract, which leads to low accuracy.
    if (p->alg == alg_t::ELU) // when x -> -0, a(exp(-x) - 1) has cancellation
        trh = 4e-5;           // subtract, which leads to low accuracy.
    if (p->dt == mkldnn_f16)
        trh = 1e-3;
    if (p->dt == mkldnn_bf16)
        trh = 1e-2;

    const auto nelems = mem_dt.nelems();
    r->errors = 0;
    r->total = nelems;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = maybe_saturate(p->dt, fp0);

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

            print(0, "[%4ld][%s] fp0:%8g fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, ind_str.c_str(), fp0, fp, dt, diff, rel_diff);
        }
    }

    if (r->errors)
        r->state = FAILED;

    if (r->state == UNTESTED)
        r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int fill_data_fwd(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        bool is_fwd = true) {
    const auto nelems = mem_fp.nelems();

    mkldnn::impl::parallel_nd(nelems, [&](int64_t i) {
            const int gen = is_fwd
                ? ((103 * i) + 107) % 109
                : ((101 * i) + 103) % 107;

            float value = FLT_MAX;
            switch (i % 4) {
            case 0: value = (gen % 11); break;  // int positive
            case 1: value = -(gen % 11); break; // int negative
            case 2: value = gen / 128.; break;  // fraction positive
            case 3: value = -gen / 128.; break; // fraction negative
            }

            mem_fp.set_elem(i, maybe_saturate(p->dt, value));
        }
    );

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_data_bwd(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_data_fwd(p, mem_dt, mem_fp, false);
}

int doit(const prb_t *p, res_t *r) {
    mkldnn_eltwise_desc_t ed;
    mkldnn_primitive_desc_t epd;
    mkldnn_primitive_t e;

    SAFE(init_pd(p, ed, epd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED)
        return OK;

    DNN_SAFE(mkldnn_primitive_create(&e, epd), WARN);
    DNN_SAFE(mkldnn_primitive_desc_destroy(epd), CRIT);

    const auto fp = mkldnn_f32;
    const auto tag = get_default_tag((int)p->dims.size());
    auto &data_desc = ed.data_desc;
    dnn_mem_t src_fp(data_desc, fp, tag, engine_ref),
              src_dt(data_desc, engine_tgt);

    dnn_mem_t dst_fp(data_desc, fp, tag, engine_ref);
    dnn_mem_t dst_dt;
    if (!p->inplace) {
        dst_dt = dnn_mem_t(data_desc, engine_tgt);
        SAFE(dst_dt.reorder(dst_fp), WARN);
    }

    SAFE(fill_data_fwd(p, src_dt, src_fp), WARN);

    dnn_mem_t d_dst_fp(data_desc, fp, tag, engine_ref),
              d_dst_dt(data_desc, engine_tgt);

    dnn_mem_t d_src_fp(data_desc, fp, tag, engine_ref);
    dnn_mem_t d_src_dt;
    if (!p->inplace) {
        d_src_dt = dnn_mem_t(data_desc, engine_tgt);
        SAFE(d_src_dt.reorder(d_src_fp), WARN);
    }

    args_t args;
    args.set(MKLDNN_ARG_SRC, src_dt.m_);

    if (p->dir & FLAG_FWD) {
        args.set(MKLDNN_ARG_DST, p->inplace ? src_dt.m_ : dst_dt.m_);

        DNN_SAFE(execute_and_wait(e, stream_tgt, args.size(), args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(p, src_fp, dst_fp);
            dnn_mem_t dst(p->inplace ? src_dt : dst_dt, fp, tag, engine_ref);
            SAFE(compare(p, dst_fp, dst, r), WARN);
        }
    } else {
        SAFE(fill_data_bwd(p, d_dst_dt, d_dst_fp), WARN);

        args.set(MKLDNN_ARG_DIFF_DST, d_dst_dt.m_);
        args.set(MKLDNN_ARG_DIFF_SRC, p->inplace ? d_dst_dt.m_ : d_src_dt.m_);

        DNN_SAFE(execute_and_wait(e, stream_tgt, args.size(), args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, src_fp, d_dst_fp, d_src_fp);
            dnn_mem_t d_src(p->inplace ? d_dst_dt : d_src_dt, fp, tag,
                    engine_ref);
            SAFE(compare(p, d_src_fp, d_src, r), WARN);
        }
    }

    measure_perf(r->timer, e, args);

    DNN_SAFE(mkldnn_primitive_destroy(e), CRIT);

    return OK;
}

}
