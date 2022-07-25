/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "sum/sum.hpp"

namespace sum {

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    std::vector<dnnl_memory_desc_t> src_d(prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input)
        src_d[i_input] = dnn_mem_t::init_md(prb->ndims, prb->dims.data(),
                prb->sdt[i_input], prb->stag[i_input]);

    dnnl_memory_desc_t dst_d {};
    if (prb->dtag != tag::undef) {
        dst_d = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->ddt, prb->dtag);
    }

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    init_pd_args.is_iterator_supported = false;
    return dnnl_sum_primitive_desc_create(&init_pd_args.pd,
            prb->dtag != tag::undef ? &dst_d : nullptr, prb->n_inputs(),
            prb->input_scales.data(), src_d.data(), dnnl_attr,
            init_pd_args.engine);
}

int fill_src(
        const prb_t *prb, int input_idx, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {

    const auto nelems = mem_fp.nelems();
    const auto dt = prb->sdt[input_idx];
    const int range = 16;
    const int f_min = dt == dnnl_u8 ? 0 : -range / 2;

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) - 17 * input_idx + 101) % range;
        const float value = (dt == dnnl_bf16 || dt == dnnl_f16)
                ? (f_min + gen) / range
                : (f_min + gen) * (1.0f + 4.0f / range);
        mem_fp.set_elem(i, round_to_nearest_representable(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    std::vector<dnnl_data_type_t> dts = prb->sdt;
    dts.push_back(prb->ddt);
    skip_unimplemented_data_type(dts, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        skip_invalid_inplace(
                res, prb->sdt[0], prb->ddt, prb->stag[0], prb->dtag);
        if (res->state == SKIPPED) return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    cmp.set_threshold(epsilon_dt(prb->ddt) * prb->n_inputs());
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();
    const auto &dst_md = query_md(const_pd, DNNL_ARG_DST);
    const auto &scratchpad_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);

    std::vector<dnn_mem_t> src_fp, src_dt;
    src_fp.reserve(prb->n_inputs());
    src_dt.reserve(prb->n_inputs());

    args_t args, ref_args;
    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const auto &src_md
                = query_md(const_pd, DNNL_ARG_MULTIPLE_SRC + i_input);
        src_fp.emplace_back(src_md, dnnl_f32, tag::abx, ref_engine);
        src_dt.emplace_back(src_md, test_engine);
        SAFE(fill_src(prb, i_input, src_dt[i_input], src_fp[i_input]), WARN);
        args.set(DNNL_ARG_MULTIPLE_SRC + i_input, src_dt[i_input]);
        if (is_bench_mode(CORR))
            ref_args.set(DNNL_ARG_MULTIPLE_SRC + i_input, src_fp[i_input]);
    }
    dnn_mem_t dst_fp(dst_md, dnnl_f32, tag::abx, ref_engine);
    dnn_mem_t placeholder_dst_dt;

    if (!prb->inplace) { placeholder_dst_dt = dnn_mem_t(dst_md, test_engine); }
    dnn_mem_t &dst_dt = prb->inplace ? src_dt[0] : placeholder_dst_dt;
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (is_bench_mode(CORR)) {
        ref_args.set(DNNL_ARG_DST, dst_fp);

        check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace sum
