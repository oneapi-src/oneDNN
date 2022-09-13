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

#include <stdio.h>
#include <stdlib.h>

#include <random>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "concat/concat.hpp"

namespace concat {

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    std::vector<dnnl_memory_desc_t> src_d(prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const dims_t &i_vdims = prb->vdims[i_input];
        src_d[i_input] = dnn_mem_t::init_md(
                prb->ndims, i_vdims.data(), prb->sdt, prb->stag[i_input]);
    }

    dnnl_memory_desc_t dst_d {};
    if (prb->dtag != tag::undef) {
        dst_d = dnn_mem_t::init_md(
                prb->ndims, prb->dst_dims.data(), prb->ddt, prb->dtag);
    }

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    init_pd_args.is_iterator_supported = false;
    return dnnl_concat_primitive_desc_create(&init_pd_args.pd,
            init_pd_args.engine, prb->dtag != tag::undef ? &dst_d : nullptr,
            prb->n_inputs(), prb->axis, src_d.data(), dnnl_attr);
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

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
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

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->sdt, prb->ddt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);
    skip_unimplemented_arg_scale(prb->attr, res);

    // ref concat is reorder-based, hence, inherits some reorder limitations.
    // bf16, f16 reorders on cpu supports only [bf16, f16]<->f32
    bool valid_xf16_input
            = IMPLICATION(prb->sdt == dnnl_bf16 || prb->sdt == dnnl_f16,
                    prb->dtag == tag::undef || prb->ddt == dnnl_f32
                            || prb->ddt == prb->sdt);
    bool valid_xf16_output
            = IMPLICATION((prb->ddt == dnnl_bf16 || prb->ddt == dnnl_f16)
                            && prb->dtag != tag::undef,
                    (prb->sdt == dnnl_f32 || prb->sdt == prb->ddt));

    if (is_cpu() && (!valid_xf16_input || !valid_xf16_output)) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {}

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

    dnn_mem_t dst_fp(dst_md, dnnl_f32, tag::abx, ref_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    args_t args, ref_args;

    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    std::vector<dnn_mem_t> src_fp, src_dt, scales;
    src_fp.reserve(prb->n_inputs());
    src_dt.reserve(prb->n_inputs());
    scales.resize(prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const auto &src_md
                = query_md(const_pd, DNNL_ARG_MULTIPLE_SRC + i_input);
        src_fp.emplace_back(src_md, dnnl_f32, tag::abx, ref_engine);
        src_dt.emplace_back(src_md, test_engine);
        SAFE(fill_src(i_input, dst_dt.dt(), src_dt[i_input], src_fp[i_input]),
                WARN);
        args.set(DNNL_ARG_MULTIPLE_SRC + i_input, src_dt[i_input]);
        if (is_bench_mode(CORR))
            ref_args.set(DNNL_ARG_MULTIPLE_SRC + i_input, src_fp[i_input]);

        // scales
        const auto &sc = prb->attr.scales.get(DNNL_ARG_MULTIPLE_SRC + i_input);
        float scale_val = sc.scale;
        maybe_prepare_runtime_scales(scales[i_input], sc, 1, &scale_val);
        args.set((DNNL_ARG_MULTIPLE_SRC + i_input) | DNNL_ARG_ATTR_INPUT_SCALES,
                scales[i_input]);
    }

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (is_bench_mode(CORR)) {
        ref_args.set(DNNL_ARG_DST, dst_fp);

        check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace concat
