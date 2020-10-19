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
#include <stdio.h>
#include <stdlib.h>

#include <random>

#include "dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "sum/sum.hpp"

namespace sum {

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &spd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    std::vector<dnnl_memory_desc_t> src_d;
    src_d.resize(prb->n_inputs());

    dnnl_memory_desc_t dst_d;

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input)
        SAFE(init_md(&src_d[i_input], prb->ndims, prb->dims.data(),
                     prb->sdt[i_input], prb->stag[i_input]),
                CRIT);

    if (prb->dtag != tag::undef) {
        SAFE(init_md(&dst_d, prb->ndims, prb->dims.data(), prb->ddt, prb->dtag),
                CRIT);
    }

    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args_t());

    dnnl_status_t init_status = dnnl_sum_primitive_desc_create(&spd,
            prb->dtag != tag::undef ? &dst_d : nullptr, prb->n_inputs(),
            prb->scales.data(), src_d.data(), dnnl_attr, engine);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(spd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", res->impl_name.c_str());

    return OK;
}

static int compare(const prb_t *prb, const dnnl_data_type_t dst_data_type,
        const dnn_mem_t &fp_mem, const dnn_mem_t &dt_mem, res_t *res) {
    const auto nelems = dt_mem.nelems();
    if (nelems == 0) return res->state = PASSED, OK;

    res->total = nelems;

    float trh = epsilon_dt(dst_data_type) * prb->n_inputs();

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = dt_mem.get_elem(i);
        const float fp0 = fp_mem.get_elem(i);
        const float fp = round_to_nearest_representable(dst_data_type, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= trh;

        res->errors += !ok;

        const bool dump = false || (!ok && (res->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = off2dims_idx(prb->dims, i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            BENCHDNN_PRINT(0,
                    "[%4ld][%s] fp0:%8g fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, ind_str.c_str(), fp0, fp, dt, diff, rel_diff);
        }
    }

    if (res->errors) res->state = FAILED;

    if (res->state == UNTESTED) res->state = PASSED; /* optimism */

    return res->state == FAILED ? FAIL : OK;
}

int fill_src(int input_idx, int n_inputs, dnnl_data_type_t dt,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    // Do fixed partitioning to have same filling for any number of threads.
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);
    // Set proper range of valid values to avoid any reorders back and forth.
    const bool s8u8_or_u8s8 = (dt == dnnl_s8 && mem_dt.dt() == dnnl_u8)
            || (dt == dnnl_u8 && mem_dt.dt() == dnnl_s8);
    float min_val = lowest_dt(dnnl_s8) / n_inputs;
    float max_val = max_dt(dnnl_u8) / n_inputs;
    if (s8u8_or_u8s8) {
        min_val = lowest_dt(dnnl_u8) / n_inputs;
        max_val = max_dt(dnnl_s8) / n_inputs;
    } else if (dt == dnnl_s8 || mem_dt.dt() == dnnl_s8) {
        max_val = max_dt(dnnl_s8) / n_inputs;
    } else if (dt == dnnl_u8 || mem_dt.dt() == dnnl_u8) {
        min_val = lowest_dt(dnnl_u8) / n_inputs;
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
    std::vector<dnnl_data_type_t> dts = prb->sdt;
    dts.push_back(prb->ddt);
    check_known_skipped_case_common(dts, FWD_D, res);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    dnnl_primitive_t s {};
    SAFE(init_prim(&s, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(s, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(s));
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
        SAFE(fill_src(i_input, prb->n_inputs(), dst_md.data_type,
                     src_dt[i_input], src_fp[i_input]),
                WARN);
        args.set(DNNL_ARG_MULTIPLE_SRC + i_input, src_dt[i_input]);
    }

    SAFE(execute_and_wait(s, args), WARN);

    if (bench_mode & CORR) {
        compute_ref(prb, src_fp, dst_fp);
        dnn_mem_t dst(dst_dt, dnnl_f32, tag::abx, test_engine);
        SAFE(compare(prb, dst_md.data_type, dst_fp, dst, res), WARN);
    }

    measure_perf(res->timer, s, args);

    DNN_SAFE_V(dnnl_primitive_destroy(s));

    return OK;
}

} // namespace sum
