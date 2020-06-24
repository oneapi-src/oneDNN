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

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

void prep_bia_dims(
        const prb_t *p, dnnl_dims_t &bia_dims, const dnnl_dims_t &dst_dims) {
    for (int d = 0; d < p->ndims; ++d)
        bia_dims[d] = (p->bia_mask & (1 << d)) ? dst_dims[d] : 1;
}

static int init_pd(dnnl_engine_t engine, const prb_t *p,
        dnnl_primitive_desc_t &mpd, res_t *r, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
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
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_d, p->ndims, src_dims,
                     p->cfg[SRC].dt, convert_tag(p->stag, p->ndims)),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&wei_d, p->ndims, wei_dims,
                     p->cfg[WEI].dt, convert_tag(p->wtag, p->ndims)),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_d, p->ndims, dst_dims,
                     p->cfg[DST].dt, convert_tag(p->dtag, p->ndims)),
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

    // Overload PER_OC mask definition for batched case
    int mask = 0;
    if (p->attr.oscale.policy == policy_t::PER_OC)
        mask = p->ndims == 3 ? 1 << 2 : 1 << 1;

    auto dnnl_attr = create_dnnl_attr(p->attr, p->n, mask, p->scales);

    dnnl_status_t init_status = dnnl_success;
    init_status
            = dnnl_primitive_desc_create(&mpd, &op_d, dnnl_attr, engine, NULL);
    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    r->impl_name = query_impl_info(mpd);
    if (maybe_skip(r->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                r->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(mpd), WARN);
        return r->state = SKIPPED, r->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", r->impl_name.c_str());
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
            BENCHDNN_PRINT(0,
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
        BENCHDNN_PRINT(0,
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
    if (nelems == 0) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    const auto &c = p->cfg[kind];
    float c_f_min = c.f_min, c_f_max = c.f_max;

    if (kind == BIA && mem_dt.dt() == dnnl_u8) c_f_min = 0;

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    dnnl::impl::parallel_nd(n_chunks, [&](int idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand msr(kind * nelems + idx_start + 1);
        msr.discard(1);

        std::uniform_int_distribution<> gen(c_f_min, c_f_max);

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

void check_known_skipped_case(const prb_t *p, res_t *r) {
    check_known_skipped_case_common(
            {p->cfg[SRC].dt, p->cfg[WEI].dt, p->cfg[DST].dt}, FWD_D, r);
    if (r->state == SKIPPED) return;

    // zero points for non-integral data type does not make sense
    if (!p->attr.zero_points.is_def() && p->cfg[WEI].dt != dnnl_s8) {
        r->state = SKIPPED, r->reason = INVALID_CASE;
        return;
    }

    // memory layout should be defined when some dimension is unknown in pd
    // creation time
    if (p->runtime_mb
            && (p->stag == "any" || p->wtag == "any" || p->dtag == "any")) {
        r->state = SKIPPED, r->reason = INVALID_CASE;
        return;
    }
    if (p->runtime_m && (p->stag == "any" || p->dtag == "any")) {
        r->state = SKIPPED, r->reason = INVALID_CASE;
        return;
    }
    if (p->runtime_n && (p->wtag == "any" || p->dtag == "any")) {
        r->state = SKIPPED, r->reason = INVALID_CASE;
        return;
    }
    if (p->runtime_k && (p->stag == "any" || p->wtag == "any")) {
        r->state = SKIPPED, r->reason = INVALID_CASE;
        return;
    }
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    check_known_skipped_case(p, r);
    if (r->state == SKIPPED) return OK;

    dnnl_primitive_t m {};
    SAFE(init_prim(&m, init_pd, p, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(m, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE(dnnl_primitive_destroy(m), CRIT);
        return r->state = SKIPPED, r->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    dnnl_memory_desc_t src_md {}, wei_md {}, dst_md {}, bia_md {}, def_md {};
    // query md if it was defined at pd creation time
    if (!p->runtime_mb && !p->runtime_m && !p->runtime_k)
        src_md = q(DNNL_ARG_SRC);
    if (!p->runtime_mb && !p->runtime_n && !p->runtime_k)
        wei_md = q(DNNL_ARG_WEIGHTS);
    if (!p->runtime_mb && !p->runtime_m && !p->runtime_n) {
        dst_md = q(DNNL_ARG_DST);
        if (p->bia_dt != dnnl_data_type_undef) bia_md = q(DNNL_ARG_BIAS);
    }

    // if md is same as default, it means we need to re-create it
    if (dnnl_memory_desc_equal(&src_md, &def_md)) {
        assert(p->stag != "any");
        src_md.dims[0 + (p->ndims == 3)] = p->m;
        src_md.dims[1 + (p->ndims == 3)] = p->k;
        if (p->ndims == 3) src_md.dims[0] = p->mb;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_md, p->ndims, src_md.dims,
                         p->cfg[SRC].dt, convert_tag(p->stag, p->ndims)),
                WARN);
    }

    if (dnnl_memory_desc_equal(&wei_md, &def_md)) {
        assert(p->wtag != "any");
        wei_md.dims[0 + (p->ndims == 3)] = p->k;
        wei_md.dims[1 + (p->ndims == 3)] = p->n;
        if (p->ndims == 3) wei_md.dims[0] = p->mb;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&wei_md, p->ndims, wei_md.dims,
                         p->cfg[WEI].dt, convert_tag(p->wtag, p->ndims)),
                WARN);
    }

    if (dnnl_memory_desc_equal(&dst_md, &def_md)) {
        assert(p->dtag != "any");
        dst_md.dims[0 + (p->ndims == 3)] = p->m;
        dst_md.dims[1 + (p->ndims == 3)] = p->n;
        if (p->ndims == 3) dst_md.dims[0] = p->mb;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_md, p->ndims, dst_md.dims,
                         p->cfg[DST].dt, convert_tag(p->dtag, p->ndims)),
                WARN);
        if (p->bia_dt != dnnl_data_type_undef) {
            prep_bia_dims(p, bia_md.dims, dst_md.dims);
            DNN_SAFE(dnnl_memory_desc_init_by_strides(
                             &bia_md, p->ndims, bia_md.dims, p->bia_dt, NULL),
                    WARN);
        }
    }

    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t bia_dt;
    if (p->bia_dt != dnnl_data_type_undef)
        bia_dt = dnn_mem_t(bia_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    const auto fp = dnnl_f32;
    dnn_mem_t src_fp(p->ndims, src_md.dims, fp, NULL, test_engine);
    dnn_mem_t wei_fp(p->ndims, wei_md.dims, fp, NULL, test_engine);
    dnn_mem_t dst_fp(p->ndims, dst_md.dims, fp, NULL, test_engine);
    dnn_mem_t bia_fp;
    if (p->bia_dt != dnnl_data_type_undef)
        bia_fp = dnn_mem_t(p->ndims, bia_md.dims, fp, NULL, test_engine);

    SAFE(fill_data(SRC, p, src_dt, src_fp, r), WARN);
    SAFE(fill_data(WEI, p, wei_dt, wei_fp, r), WARN);
    SAFE(fill_data(DST, p, dst_dt, dst_fp, r), WARN);
    if (p->bia_dt != dnnl_data_type_undef)
        SAFE(fill_data(BIA, p, bia_dt, bia_fp, r), WARN);

    dnn_mem_t scales;
    dnn_mem_t src_zero_points_m, wei_zero_points_m, dst_zero_points_m;
    maybe_prepare_runtime_scales(scales, p->attr, p->n, p->scales);
    maybe_prepare_runtime_zero_points(src_zero_points_m, p->attr, DNNL_ARG_SRC);
    maybe_prepare_runtime_zero_points(
            wei_zero_points_m, p->attr, DNNL_ARG_WEIGHTS);
    maybe_prepare_runtime_zero_points(dst_zero_points_m, p->attr, DNNL_ARG_DST);

    args_t args;

    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_WEIGHTS, wei_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    if (p->bia_dt != dnnl_data_type_undef) args.set(DNNL_ARG_BIAS, bia_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(DNNL_ARG_ATTR_OUTPUT_SCALES, scales);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_points_m);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zero_points_m);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_points_m);
    SAFE(execute_and_wait(m, args), WARN);

    if (bench_mode & CORR) {
        compute_ref(test_engine, p, src_fp, wei_fp, bia_fp, dst_fp);
        dnn_mem_t c(dst_dt, fp, get_abx_tag(p->ndims), test_engine);
        SAFE(compare_dat(p, DST, c, dst_fp, r), WARN);
    }

    measure_perf(r->timer, m, args);

    DNN_SAFE_V(dnnl_primitive_destroy(m));

    return OK;
}

} // namespace matmul
