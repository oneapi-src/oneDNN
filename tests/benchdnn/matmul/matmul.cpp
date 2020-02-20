/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "dnnl.h"

#include "src/common/dnnl_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

void prep_bia_dims(
        const prb_t *p, dnnl_dims_t &bia_dims, const dnnl_dims_t &dst_dims) {
    for (int d = 0; d < p->ndims; ++d)
        bia_dims[d] = (p->bia_mask & (1 << d)) ? dst_dims[d] : 1;
}

int init_pd(const prb_t *p, dnnl_primitive_desc_t &matmul_pd, res_t *r) {
    const int64_t MB = p->runtime_mb ? DNNL_RUNTIME_DIM_VAL : p->mb;
    const int64_t M = p->runtime_m ? DNNL_RUNTIME_DIM_VAL : p->m;
    const int64_t N = p->runtime_n ? DNNL_RUNTIME_DIM_VAL : p->n;
    const int64_t K = p->runtime_k ? DNNL_RUNTIME_DIM_VAL : p->k;

    dnnl_dims_t src_dims {}, wei_dims {}, dst_dims {}, bia_dims {};
    src_dims[0 + (p->ndims == 3)] = dst_dims[0 + (p->ndims == 3)] = M;
    src_dims[1 + (p->ndims == 3)] = wei_dims[0 + (p->ndims == 3)] = K;
    wei_dims[1 + (p->ndims == 3)] = dst_dims[1 + (p->ndims == 3)] = N;
    if (p->ndims == 3) src_dims[0] = wei_dims[0] = dst_dims[0] = MB;

    prep_bia_dims(p, bia_dims, dst_dims);

    dnnl_memory_desc_t src_d, wei_d, dst_d, bia_d {};
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &src_d, p->ndims, src_dims, p->cfg[SRC].dt, p->stag),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &wei_d, p->ndims, wei_dims, p->cfg[WEI].dt, p->wtag),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &dst_d, p->ndims, dst_dims, p->cfg[DST].dt, p->dtag),
            WARN);
    if (p->bia_dt != dnnl_data_type_undef)
        DNN_SAFE(dnnl_memory_desc_init_by_strides(
                         &bia_d, p->ndims, bia_dims, p->bia_dt, NULL),
                WARN);

    dnnl_matmul_desc_t op_d;
    DNN_SAFE(
            dnnl_matmul_desc_init(&op_d, &src_d, &wei_d, &bia_d, &dst_d), WARN);
    DNN_SAFE(op_d.accum_data_type == p->cfg[ACC].dt ? dnnl_success
                                                    : dnnl_unimplemented,
            CRIT);

    auto dnnl_attr = create_dnnl_attr(p->attr, p->n, p->scales);

    dnnl_status_t init_status = dnnl_success;
    init_status = dnnl_primitive_desc_create(
            &matmul_pd, &op_d, dnnl_attr, engine_tgt, NULL);
    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(matmul_pd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: dnnl implementation: %s\n", impl_str);
        DNN_SAFE(dnnl_primitive_desc_destroy(matmul_pd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "dnnl implementation: %s\n", impl_str);
    }

    return OK;
}

int compare_dat(const prb_t *p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r) {
    const auto nelems = mem_dt.nelems();
    int64_t non_zero = 0;
    const char *skind = data_kind2str(kind);

    r->errors = 0;
    r->total = nelems;

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = maybe_saturate(p->cfg[kind].dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= p->cfg[kind].eps;

        r->errors += !ok;

        const bool dump = false || (!ok && (r->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            print(0,
                    "[%4ld][%s]"
                    "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, skind, fp, fp0, dt, diff, rel_diff);
        }
        non_zero += fp != 0;
    }

    const double trust_nz = (double)non_zero / r->total;
    bool no_trust = trust_nz < 0.1;
    if (no_trust) {
        r->state = MISTRUSTED;
        const char *skind = data_kind2str(kind);
        print(0,
                "@@@ [%s] test-bug: trust is too low."
                " Nonzeros in output: %.2f\n",
                skind, trust_nz);
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int fill_data(data_kind_t kind, const prb_t *p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r) {
    const auto nelems = mem_dt.nelems();
    assert(mem_dt.nelems() == mem_fp.nelems());

    const auto &c = p->cfg[kind];
    float c_f_min = c.f_min, c_f_max = c.f_max;

    if (kind == BIA && mem_dt.dt() == dnnl_u8) c_f_min = 0;

    dnnl::impl::parallel(0, [&](int ithr, int nthr) {
        int64_t chunk_size = (nelems + nthr - 1) / nthr;
        int64_t idx_start = ithr * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        std::minstd_rand msr;
        std::uniform_int_distribution<> gen(c_f_min, c_f_max);
        msr.discard(kind + idx_start);

        // make sure the first element is not zero
        if (idx_start == 0) {
            float val = 0;
            while (val == 0)
                val = (float)gen(msr);
            mem_fp.set_elem(0, val * c.f_scale);
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            auto val = (float)gen(msr) * c.f_scale;
            mem_fp.set_elem(idx, val);
        }
    });

    // work-around mistrusted when A > 0 && B < 0  && C.dt = u8 (or relu)
    if (kind == WEI && nelems == 1 && p->cfg[SRC].dt == dnnl_u8) {
        if (c.f_max >= 1) mem_fp.set_elem(0, c.f_scale);
    }

    SAFE(mem_dt.reorder(mem_fp), WARN);
    return OK;
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    const auto def_tag = get_default_tag(p->ndims);

    dnnl_primitive_t matmul = NULL;

    // init matmul primitive
    {
        dnnl_primitive_desc_t matmul_pd;

        SAFE(init_pd(p, matmul_pd, r), WARN);
        if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

        DNN_SAFE(dnnl_primitive_create(&matmul, matmul_pd), WARN);
        DNN_SAFE(dnnl_primitive_desc_destroy(matmul_pd), CRIT);
    }

    const_dnnl_primitive_desc_t matmul_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(matmul, &matmul_pd), CRIT);

    auto q_md = [&](dnnl_query_t what, int idx) {
        const auto *md = dnnl_primitive_desc_query_md(matmul_pd, what, idx);
        SAFE_V(md != nullptr ? OK : FAIL);
        return md;
    };

    dnnl_memory_desc_t src_dt_d = *q_md(dnnl_query_src_md, 0);
    dnnl_memory_desc_t wei_dt_d = *q_md(dnnl_query_weights_md, 0);
    dnnl_memory_desc_t dst_dt_d = *q_md(dnnl_query_dst_md, 0);
    dnnl_memory_desc_t bia_dt_d = *q_md(dnnl_query_weights_md, 1);

    dnn_mem_t src_dt, wei_dt, dst_dt, bia_dt;

    if (p->runtime_mb || p->runtime_m || p->runtime_k) {
        src_dt_d.dims[0 + (p->ndims == 3)] = p->m;
        src_dt_d.dims[1 + (p->ndims == 3)] = p->k;
        if (p->ndims == 3) src_dt_d.dims[0] = p->mb;
        src_dt = dnn_mem_t(
                p->ndims, src_dt_d.dims, p->cfg[SRC].dt, p->stag, engine_tgt);
    } else {
        src_dt = dnn_mem_t(src_dt_d, engine_tgt);
    }
    if (p->runtime_mb || p->runtime_k || p->runtime_n) {
        wei_dt_d.dims[0 + (p->ndims == 3)] = p->k;
        wei_dt_d.dims[1 + (p->ndims == 3)] = p->n;
        if (p->ndims == 3) wei_dt_d.dims[0] = p->mb;
        wei_dt = dnn_mem_t(
                p->ndims, wei_dt_d.dims, p->cfg[WEI].dt, p->wtag, engine_tgt);
    } else {
        wei_dt = dnn_mem_t(wei_dt_d, engine_tgt);
    }
    if (p->runtime_mb || p->runtime_m || p->runtime_n) {
        dst_dt_d.dims[0 + (p->ndims == 3)] = p->m;
        dst_dt_d.dims[1 + (p->ndims == 3)] = p->n;
        if (p->ndims == 3) dst_dt_d.dims[0] = p->mb;
        dst_dt = dnn_mem_t(
                p->ndims, dst_dt_d.dims, p->cfg[DST].dt, p->dtag, engine_tgt);

        if (p->bia_dt != dnnl_data_type_undef) {
            prep_bia_dims(p, bia_dt_d.dims, dst_dt_d.dims);
            bia_dt = dnn_mem_t(
                    p->ndims, bia_dt_d.dims, p->bia_dt, def_tag, engine_tgt);
        }
    } else {
        dst_dt = dnn_mem_t(dst_dt_d, engine_tgt);
        if (p->bia_dt != dnnl_data_type_undef)
            bia_dt = dnn_mem_t(bia_dt_d, engine_tgt);
    }

    const auto fp = dnnl_f32;
    dnn_mem_t src_fp(p->ndims, src_dt_d.dims, fp, NULL, engine_tgt);
    dnn_mem_t wei_fp(p->ndims, wei_dt_d.dims, fp, NULL, engine_tgt);
    dnn_mem_t dst_fp(p->ndims, dst_dt_d.dims, fp, NULL, engine_tgt);
    dnn_mem_t bia_fp;
    if (p->bia_dt != dnnl_data_type_undef)
        bia_fp = dnn_mem_t(p->ndims, bia_dt_d.dims, fp, NULL, engine_tgt);

    SAFE(fill_data(SRC, p, src_dt, src_fp, r), WARN);
    SAFE(fill_data(WEI, p, wei_dt, wei_fp, r), WARN);
    SAFE(fill_data(DST, p, dst_dt, dst_fp, r), WARN);
    if (p->bia_dt != dnnl_data_type_undef)
        SAFE(fill_data(BIA, p, bia_dt, bia_fp, r), WARN);

    dnn_mem_t scales;
    dnn_mem_t src_zero_points_m, wei_zero_points_m, dst_zero_points_m;
    maybe_prepare_runtime_scales(scales, p->attr, p->n, p->scales, engine_tgt);
    maybe_prepare_runtime_zero_points(
            src_zero_points_m, p->attr, DNNL_ARG_SRC, engine_tgt);
    maybe_prepare_runtime_zero_points(
            wei_zero_points_m, p->attr, DNNL_ARG_WEIGHTS, engine_tgt);
    maybe_prepare_runtime_zero_points(
            dst_zero_points_m, p->attr, DNNL_ARG_DST, engine_tgt);

    args_t args;

    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_WEIGHTS, wei_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    if (p->bia_dt != dnnl_data_type_undef) args.set(DNNL_ARG_BIAS, bia_dt);
    args.set(DNNL_ARG_ATTR_OUTPUT_SCALES, scales);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_points_m);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zero_points_m);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_points_m);
    DNN_SAFE(execute_and_wait(matmul, stream_tgt, args), WARN);

    if (bench_mode & CORR) {
        compute_ref(p, src_fp, wei_fp, bia_fp, dst_fp);
        dnn_mem_t c(dst_dt, fp, def_tag, engine_tgt);
        SAFE(compare_dat(p, DST, c, dst_fp, r), WARN);
    }

    measure_perf(r->timer, matmul, args);

    DNN_SAFE(dnnl_primitive_destroy(matmul), CRIT);

    return OK;
}

} // namespace matmul
