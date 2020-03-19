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

#include "src/common/dnnl_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "concat/concat.hpp"

namespace concat {

static int init_pd(const engine_t &engine_tgt, const prb_t *p,
        dnnl_primitive_desc_t &cpd, res_t *r) {
    std::vector<dnnl_memory_desc_t> src_d;
    src_d.resize(p->n_inputs());

    dnnl_memory_desc_t dst_d;

    for (int i_input = 0; i_input < p->n_inputs(); ++i_input) {
        const dims_t &i_sdims = p->sdims[i_input];
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_d[i_input], p->ndims,
                         i_sdims.data(), p->sdt,
                         convert_tag(p->stag[i_input], p->ndims)),
                WARN);
    }

    if (p->dtag != tag::undef) {
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_d, p->ndims, p->ddims.data(),
                         p->ddt, convert_tag(p->dtag, p->ndims)),
                WARN);
    }

    auto dnnl_attr = create_dnnl_attr(attr_t());

    dnnl_status_t init_status = dnnl_concat_primitive_desc_create(&cpd,
            p->dtag != tag::undef ? &dst_d : NULL, p->n_inputs(), p->axis,
            src_d.data(), dnnl_attr, engine_tgt);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(cpd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", impl_str);

    return OK;
}

static int compare(const prb_t *p, const dnnl_data_type_t dst_data_type,
        const dnn_mem_t &fp_mem, const dnn_mem_t &dt_mem, res_t *r) {
    const auto nelems = dt_mem.nelems();
    r->errors = 0;
    r->total = nelems;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = dt_mem.get_elem(i);
        const float fp0 = fp_mem.get_elem(i);
        const float fp = maybe_saturate(dst_data_type, fp0);

        const bool ok = dt == fp; // expect exact answer due to int values

        r->errors += !ok;

        const bool dump = false || (!ok && (r->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t ddims_idx = off2dims_idx(p->ddims, i);
            ss << ddims_idx;
            std::string ind_str = ss.str();

            BENCHDNN_PRINT(0, "[%4ld][%s] fp0:%8g fp:%8g dt:%8g\n", (long)i,
                    ind_str.c_str(), fp0, fp, dt);
        }
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int fill_src(
        const prb_t *p, int input_idx, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    auto get_range = [](const dnnl_data_type_t dt) {
        if (dt == dnnl_s8 || dt == dnnl_u8)
            return 256;
        else if (dt == dnnl_bf16 || dt == dnnl_f16)
            return 128;
        return 1024;
    };

    const auto nelems = mem_fp.nelems();
    const int range = get_range(p->sdt);
    const int f_min = p->sdt == dnnl_u8 ? 0 : -range / 2;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) - 17 * input_idx + 101) % range;
        const float value = f_min + gen;
        mem_fp.set_elem(i, maybe_saturate(p->sdt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;
    engine_t engine_tgt(engine_tgt_kind);

    dnnl_primitive_desc_t cpd;
    SAFE(init_pd(engine_tgt, p, cpd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    dnnl_primitive_t c;
    DNN_SAFE(dnnl_primitive_create(&c, cpd), WARN);
    DNN_SAFE(dnnl_primitive_desc_destroy(cpd), CRIT);

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(c, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(c));
        return r->state = SKIPPED, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto fp = dnnl_f32;
    const auto tag = get_abx_tag(p->ndims);

    const auto &dst_md = q(DNNL_ARG_DST);
    const auto dst_data_type = dst_md.data_type; // needed for deduced dst
    dnn_mem_t dst_fp(dst_md, fp, tag, engine_tgt);
    dnn_mem_t dst_dt(dst_md, engine_tgt);

    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);
    dnn_mem_t scratchpad_dt(scratchpad_md, engine_tgt);

    args_t args;
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    std::vector<dnn_mem_t> src_fp, src_dt;
    src_fp.reserve(p->n_inputs());
    src_dt.reserve(p->n_inputs());

    for (int i_input = 0; i_input < p->n_inputs(); ++i_input) {
        const auto &src_md = q(DNNL_ARG_MULTIPLE_SRC + i_input);
        src_fp.emplace_back(src_md, fp, tag, engine_tgt);
        src_dt.emplace_back(src_md, engine_tgt);
        SAFE(fill_src(p, i_input, src_dt[i_input], src_fp[i_input]), WARN);
        args.set(DNNL_ARG_MULTIPLE_SRC + i_input, src_dt[i_input]);
    }

    DNN_SAFE(execute_and_wait(c, engine_tgt, args), WARN);

    if (bench_mode & CORR) {
        compute_ref(p, src_fp, dst_fp);

        // convert dst_fp into target precision back and forth for proper
        // answer comparison
        dnn_mem_t dst_fp_dt(dst_fp, dst_data_type, tag, engine_tgt);
        SAFE(dst_fp_dt.reorder(dst_fp), WARN);
        SAFE(dst_fp.reorder(dst_fp_dt), WARN);

        dnn_mem_t dst(dst_dt, fp, tag, engine_tgt);
        SAFE(compare(p, dst_data_type, dst_fp, dst, r), WARN);
    }

    measure_perf(r->timer, engine_tgt, c, args);

    DNN_SAFE_V(dnnl_primitive_destroy(c));

    return OK;
}

} // namespace concat
