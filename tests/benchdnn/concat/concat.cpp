/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include <random>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "compare.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "concat/concat.hpp"

namespace concat {

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &cpd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    std::vector<dnnl_memory_desc_t> src_d;
    src_d.resize(prb->n_inputs());

    dnnl_memory_desc_t dst_d;

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const dims_t &i_sdims = prb->sdims[i_input];
        SAFE(init_md(&src_d[i_input], prb->ndims, i_sdims.data(), prb->sdt,
                     prb->stag[i_input]),
                CRIT);
    }

    if (prb->dtag != tag::undef) {
        SAFE(init_md(&dst_d, prb->ndims, prb->ddims.data(), prb->ddt,
                     prb->dtag),
                CRIT);
    }

    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args_t());

    dnnl_status_t init_status = dnnl_concat_primitive_desc_create(&cpd,
            prb->dtag != tag::undef ? &dst_d : nullptr, prb->n_inputs(),
            prb->axis, src_d.data(), dnnl_attr, engine);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(cpd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", res->impl_name.c_str());

    return OK;
}

int fill_src(int input_idx, dnnl_data_type_t dt, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    // Do fixed partitioning to have same filling for any number of threads.
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);
    // Set proper range of valid values to avoid any reorders back and forth.
    const bool s8u8_or_u8s8 = (dt == dnnl_s8 && mem_dt.dt() == dnnl_u8)
            || (dt == dnnl_u8 && mem_dt.dt() == dnnl_s8);
    float min_val = lowest_dt(dnnl_s8);
    float max_val = max_dt(dnnl_u8);
    if (s8u8_or_u8s8) {
        min_val = lowest_dt(dnnl_u8);
        max_val = max_dt(dnnl_s8);
    } else if (dt == dnnl_s8 || mem_dt.dt() == dnnl_s8) {
        max_val = max_dt(dnnl_s8);
    } else if (dt == dnnl_u8 || mem_dt.dt() == dnnl_u8) {
        min_val = lowest_dt(dnnl_u8);
    }

    dnnl::impl::parallel_nd(n_chunks, [&](int idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // See eltwise.cpp for implementation details.
        std::minstd_rand msr(input_idx * n_chunks + idx_start + 1);
        msr.discard(1);
        std::uniform_int_distribution<> igen(min_val, max_val);
        // No need to round final value as it's already in needed dt.
        for (int64_t idx = idx_start; idx < idx_end; ++idx)
            mem_fp.set_elem(idx, (float)igen(msr));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common({prb->sdt, prb->ddt}, FWD_D, res);
    if (res->state == SKIPPED) return;

    // ref concat is reorder-based, hence, inherits some reorder limitations.
    // bf16 reorder on cpu supports only bf16/f32 src_dt/dst_dt
    bool valid_bf16_input = IMPLICATION(prb->sdt == dnnl_bf16,
            prb->dtag == tag::undef || prb->ddt == dnnl_f32
                    || prb->ddt == dnnl_bf16);
    bool valid_bf16_output
            = IMPLICATION(prb->ddt == dnnl_bf16 && prb->dtag != tag::undef,
                    (prb->sdt == dnnl_f32 || prb->sdt == dnnl_bf16));

    if (is_cpu() && (!valid_bf16_input || !valid_bf16_output)) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    dnnl_primitive_t c {};
    SAFE(init_prim(&c, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(c, &const_pd), CRIT);

    if (check_mem_size(const_pd) != OK) {
        DNN_SAFE(dnnl_primitive_destroy(c), CRIT);
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto &test_engine = get_test_engine();
    const auto &dst_md = q(DNNL_ARG_DST);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    dnn_mem_t dst_fp(dst_md, dnnl_f32, tag::abx, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    args_t args;
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    std::vector<dnn_mem_t> src_fp, src_dt;
    src_fp.reserve(prb->n_inputs());
    src_dt.reserve(prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const auto &src_md = q(DNNL_ARG_MULTIPLE_SRC + i_input);
        src_fp.emplace_back(src_md, dnnl_f32, tag::abx, test_engine);
        src_dt.emplace_back(src_md, test_engine);
        SAFE(fill_src(i_input, dst_md.data_type, src_dt[i_input],
                     src_fp[i_input]),
                WARN);
        args.set(DNNL_ARG_MULTIPLE_SRC + i_input, src_dt[i_input]);
    }

    SAFE(execute_and_wait(c, args), WARN);

    if (bench_mode & CORR) {
        compute_ref(prb, src_fp, dst_fp);
        compare::compare_t cmp;
        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }

    measure_perf(res->timer, c, args);

    DNN_SAFE(dnnl_primitive_destroy(c), CRIT);

    return OK;
}

} // namespace concat
