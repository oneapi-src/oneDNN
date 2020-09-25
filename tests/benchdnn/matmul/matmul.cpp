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

void prep_bia_dims(const prb_t *p, dims_t &bia_dims, const dims_t &dst_dims) {
    bia_dims.resize(dst_dims.size());
    for (int d = 0; d < p->ndims; ++d)
        bia_dims[d] = (p->bia_mask & (1 << d)) ? dst_dims[d] : 1;
}

dims_t get_runtime_dims(const dims_t &dims, const dims_mask_t &mask) {
    if (mask.none() || dims.empty()) return dims;
    dims_t runtime_dims;
    runtime_dims.resize(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        runtime_dims[i] = mask[i] ? DNNL_RUNTIME_DIM_VAL : dims[i];
    }
    return runtime_dims;
}

static int init_pd(dnnl_engine_t engine, const prb_t *p,
        dnnl_primitive_desc_t &mpd, res_t *r, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    const auto &src_rt_dims
            = get_runtime_dims(p->src_dims(), p->src_runtime_dim_mask());
    const auto &weights_rt_dims = get_runtime_dims(
            p->weights_dims(), p->weights_runtime_dim_mask());
    const auto &dst_rt_dims
            = get_runtime_dims(p->dst_dims(), p->dst_runtime_dim_mask());

    dnnl_memory_desc_t src_d, wei_d, dst_d, bia_d {};
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_d, p->ndims, src_rt_dims.data(),
                     p->cfg[SRC].dt, convert_tag(p->stag, p->ndims)),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&wei_d, p->ndims,
                     weights_rt_dims.data(), p->cfg[WEI].dt,
                     convert_tag(p->wtag, p->ndims)),
            WARN);
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_d, p->ndims, dst_rt_dims.data(),
                     p->cfg[DST].dt, convert_tag(p->dtag, p->ndims)),
            WARN);
    if (p->bia_dt != dnnl_data_type_undef) {
        dims_t bia_dims;
        prep_bia_dims(p, bia_dims, p->dst_dims());
        bia_dims = get_runtime_dims(bia_dims, p->dst_runtime_dim_mask());
        DNN_SAFE(dnnl_memory_desc_init_by_strides(
                         &bia_d, p->ndims, bia_dims.data(), p->bia_dt, NULL),
                WARN);
    }

    dnnl_matmul_desc_t op_d;
    DNN_SAFE(
            dnnl_matmul_desc_init(&op_d, &src_d, &wei_d, &bia_d, &dst_d), WARN);
    DNN_SAFE(op_d.accum_data_type == p->cfg[ACC].dt ? dnnl_success
                                                    : dnnl_unimplemented,
            CRIT);

    auto dnnl_attr = create_dnnl_attr(p->attr, p->n, p->scales);

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
            {p->cfg[SRC].dt, p->cfg[WEI].dt, p->cfg[DST].dt}, r);
    if (r->state == SKIPPED) return;

    // zero points for non-integral data type does not make sense
    if (!p->attr.zero_points.is_def() && p->cfg[WEI].dt != dnnl_s8) {
        r->state = SKIPPED, r->reason = INVALID_CASE;
        return;
    }

    auto src_rt_mask = p->src_runtime_dim_mask();
    auto wei_rt_mask = p->weights_runtime_dim_mask();
    auto dst_rt_mask = p->dst_runtime_dim_mask();

    // memory layout should be defined when some dimension is unknown in pd
    // creation time
    if ((src_rt_mask.any() && p->stag == "any")
            || (wei_rt_mask.any() && p->wtag == "any")
            || (dst_rt_mask.any() && p->dtag == "any")) {
        r->state = SKIPPED, r->reason = INVALID_CASE;
        return;
    }

    // inconsistent runtime mask for m, k, n are not supported
    const int m_idx = p->ndims - 2;
    const int k_idx_src = p->ndims - 1;
    const int k_idx_wei = p->ndims - 2;
    const int n_idx = p->ndims - 1;
    if (src_rt_mask[m_idx] != dst_rt_mask[m_idx]
            || src_rt_mask[k_idx_src] != wei_rt_mask[k_idx_wei]
            || wei_rt_mask[n_idx] != dst_rt_mask[n_idx]) {
        r->state = SKIPPED, r->reason = INVALID_CASE;
        return;
    }

    // inconsistent runtime masks for batch dims are not supported
    if (p->ndims > 2) {
        dims_mask_t batch_rt_mask;
        for (int i = 0; i < p->ndims - 2; ++i)
            batch_rt_mask[i] = true;
        src_rt_mask &= batch_rt_mask;
        wei_rt_mask &= batch_rt_mask;
        dst_rt_mask &= batch_rt_mask;
        if (src_rt_mask != wei_rt_mask || src_rt_mask != dst_rt_mask) {
            r->state = SKIPPED, r->reason = INVALID_CASE;
            return;
        }
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
    if (p->src_runtime_dim_mask().none()) src_md = q(DNNL_ARG_SRC);
    if (p->weights_runtime_dim_mask().none()) wei_md = q(DNNL_ARG_WEIGHTS);
    if (p->dst_runtime_dim_mask().none()) {
        dst_md = q(DNNL_ARG_DST);
        if (p->bia_dt != dnnl_data_type_undef) bia_md = q(DNNL_ARG_BIAS);
    }

    // if md is same as default, it means we need to re-create it
    const auto &src_dims = p->src_dims();
    if (dnnl_memory_desc_equal(&src_md, &def_md)) {
        assert(p->stag != tag::any);
        DNN_SAFE(
                dnnl_memory_desc_init_by_tag(&src_md, p->ndims, src_dims.data(),
                        p->cfg[SRC].dt, convert_tag(p->stag, p->ndims)),
                WARN);
    }

    const auto &weights_dims = p->weights_dims();
    if (dnnl_memory_desc_equal(&wei_md, &def_md)) {
        assert(p->wtag != "any");
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&wei_md, p->ndims,
                         weights_dims.data(), p->cfg[WEI].dt,
                         convert_tag(p->wtag, p->ndims)),
                WARN);
    }

    const auto &dst_dims = p->dst_dims();
    if (dnnl_memory_desc_equal(&dst_md, &def_md)) {
        assert(p->dtag != tag::any);
        DNN_SAFE(
                dnnl_memory_desc_init_by_tag(&dst_md, p->ndims, dst_dims.data(),
                        p->cfg[DST].dt, convert_tag(p->dtag, p->ndims)),
                WARN);
    }

    if (p->bia_dt != dnnl_data_type_undef) {
        dims_t bia_dims;
        prep_bia_dims(p, bia_dims, dst_dims);
        DNN_SAFE(dnnl_memory_desc_init_by_strides(
                         &bia_md, p->ndims, bia_dims.data(), p->bia_dt, NULL),
                WARN);
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
