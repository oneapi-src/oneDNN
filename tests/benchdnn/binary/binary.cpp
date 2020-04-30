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

#include <stdio.h>
#include <stdlib.h>

#include "dnnl.h"

#include "tests/test_thread.hpp"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "eltwise/eltwise.hpp"

namespace binary {

static int init_pd(dnnl_engine_t engine, const prb_t *p,
        dnnl_primitive_desc_t &bpd, res_t *r, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_binary_desc_t bd;
    std::vector<dnnl_memory_desc_t> src_d;
    src_d.resize(p->n_inputs());

    for (int i_input = 0; i_input < p->n_inputs(); ++i_input) {
        const dims_t &i_sdims = p->sdims[i_input];
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_d[i_input],
                         p->ndims[i_input], i_sdims.data(), p->sdt[i_input],
                         convert_tag(p->stag[i_input], p->ndims[i_input])),
                WARN);
    }

    if (p->ndims[1] < p->ndims[0]) { // need to reshape B
        dnnl_dims_t dims;
        for (int d = 0; d < p->ndims[1]; ++d)
            dims[d] = p->sdims[1][d];
        for (int d = p->ndims[1]; d < p->ndims[0]; ++d)
            dims[d] = 1;
        DNN_SAFE(dnnl_memory_desc_reshape(
                         &src_d[1], &src_d[1], p->ndims[0], dims),
                WARN);
    }

    dnnl_memory_desc_t dst_d;
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_d, p->ndims[0],
                     p->sdims[0].data(), p->ddt, dnnl_format_tag_any),
            WARN);

    dnnl_alg_kind_t alg = alg2alg_kind(p->alg);

    DNN_SAFE(dnnl_binary_desc_init(&bd, alg, &src_d[0], &src_d[1], &dst_d),
            WARN);

    auto dnnl_attr = create_dnnl_attr(p->attr);

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&bpd, &bd, dnnl_attr, engine, NULL);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    r->impl_name = query_impl_info(bpd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", r->impl_name.c_str());

    return OK;
}

static int compare(const prb_t *p, const dnn_mem_t &fp_mem,
        const dnn_mem_t &dt_mem, res_t *r) {
    const auto nelems = dt_mem.nelems();
    r->errors = 0;
    r->total = nelems;
    const float trh = epsilon_dt(p->ddt == dnnl_f16 ? dnnl_f16 : dnnl_f32)
            * p->n_inputs();
    const int eltwise_idx = p->attr.post_ops.eltwise_index();

    const bool has_eltwise = eltwise_idx >= 0;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = dt_mem.get_elem(i);
        const float fp0 = fp_mem.get_elem(i);
        const float fp = maybe_saturate(p->ddt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= trh;
        if (!ok && has_eltwise)
            ok = eltwise::check_extreme_values(
                    fp, dt, p->attr.post_ops.entry[eltwise_idx].kind);

        r->errors += !ok;

        const bool dump = false || (!ok && (r->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = off2dims_idx(p->sdims[0], i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            BENCHDNN_PRINT(0,
                    "[%4ld][%s] fp0:%8g fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, ind_str.c_str(), fp0, fp, dt, diff, rel_diff);
        }
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int fill_src(
        const prb_t *p, int input_idx, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    const auto dt = p->sdt[input_idx];
    const int range = 16;
    const int f_min = dt == dnnl_u8 ? 0 : -range / 2;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) - 5 * input_idx + 101) % (range + 1);
        const float value = (dt == dnnl_bf16 || dt == dnnl_f16)
                ? (f_min + gen) / range
                : (f_min + gen) * (1.0f + 4.0f / range);
        mem_fp.set_elem(i, maybe_saturate(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void check_known_skipped_case(const prb_t *p, res_t *r) {
    check_known_skipped_case_common(p->sdt, r);
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    check_known_skipped_case(p, r);
    if (r->state == SKIPPED) return OK;

    dnnl_primitive_t b {};
    SAFE(init_prim(&b, init_pd, p, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(b, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(b));
        return r->state = SKIPPED, r->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto &src0_md = q(DNNL_ARG_SRC_0);
    const auto &src1_md = q(DNNL_ARG_SRC_1);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = get_abx_tag(p->ndims[0]);

    const auto &test_engine = get_test_engine();

    dnn_mem_t src0_fp(src0_md, fp, tag, test_engine);
    dnn_mem_t src0_dt(src0_md, test_engine);
    SAFE(fill_src(p, 0, src0_dt, src0_fp), WARN);

    dnn_mem_t src1_fp(src1_md, fp, tag, test_engine);
    dnn_mem_t src1_dt(src1_md, test_engine);
    SAFE(fill_src(p, 1, src1_dt, src1_fp), WARN);

    dnn_mem_t &dst_fp = src0_fp; // in-place in ref code
    dnn_mem_t placeholder_dst_dt;
    if (!p->inplace) {
        const auto &dst_md = q(DNNL_ARG_DST);
        placeholder_dst_dt = dnn_mem_t(dst_md, test_engine);

        if (p->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0)
            SAFE(placeholder_dst_dt.reorder(dst_fp), WARN);
    }
    dnn_mem_t &dst_dt = p->inplace ? src0_dt : placeholder_dst_dt;

    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    args_t args;
    args.set(DNNL_ARG_SRC_0, src0_dt);
    args.set(DNNL_ARG_SRC_1, src1_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    DNN_SAFE(execute_and_wait(b, args), WARN);

    if (bench_mode & CORR) {
        compute_ref(p, src0_fp, src1_fp, dst_fp);
        dnn_mem_t dst(dst_dt, fp, tag, test_engine);
        SAFE(compare(p, dst_fp, dst, r), WARN);
    }

    measure_perf(r->timer, b, args);

    DNN_SAFE_V(dnnl_primitive_destroy(b));

    return OK;
}

} // namespace binary
