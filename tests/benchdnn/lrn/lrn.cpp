/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "lrn/lrn.hpp"

namespace lrn {

int fill_dat(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    const int range = 16;
    const int f_min = prb->dt == dnnl_u8 ? 0 : -range / 2;

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const int64_t gen = kind == SRC ? 1091 * i + 1637 : 1279 * i + 1009;
        const float value = f_min + gen % range;
        mem_fp.set_elem(i, value);
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_src(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_dat(prb, SRC, mem_dt, mem_fp);
}

int fill_dst(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_dat(prb, DST, mem_dt, mem_fp);
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    const dir_t dir = init_pd_args.dir;

    dnnl_lrn_desc_t ld;

    dnnl_dims_t data_dims_0d = {prb->mb, prb->ic};
    dnnl_dims_t data_dims_1d = {prb->mb, prb->ic, prb->iw};
    dnnl_dims_t data_dims_2d = {prb->mb, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t data_dims_3d = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};

    dnnl_dim_t *data_dims = prb->ndims == 5
            ? data_dims_3d
            : prb->ndims == 4 ? data_dims_2d
                              : prb->ndims == 3 ? data_dims_1d : data_dims_0d;

    auto data_d = dnn_mem_t::init_md(prb->ndims, data_dims, prb->dt, prb->tag);

    dnnl_alg_kind_t alg = alg2alg_kind(prb->alg);
    if (dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        DNN_SAFE_STATUS(dnnl_lrn_forward_desc_init(&ld, prop, alg, &data_d,
                prb->ls, prb->alpha, prb->beta, prb->k));
    } else {
        auto diff_data_d
                = dnn_mem_t::init_md(prb->ndims, data_dims, prb->dt, tag::any);
        DNN_SAFE_STATUS(dnnl_lrn_backward_desc_init(&ld, alg, &diff_data_d,
                &data_d, prb->ls, prb->alpha, prb->beta, prb->k));
    }

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    return dnnl_primitive_desc_iterator_create(&init_pd_args.pd_it, &ld,
            dnnl_attr, init_pd_args.engine, init_pd_args.hint);
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->dt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    // `3` is a const needed to adjust division error
    cmp.set_threshold(compute_n_summands(prb) * 3 * epsilon_dt(prb->dt));
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_fpd = query_pd(prim);

    if (check_mem_size(const_fpd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto &data_md = query_md(const_fpd, DNNL_ARG_SRC);
    const auto &ws_md = query_md(const_fpd, DNNL_ARG_WORKSPACE);
    const auto &scratchpad_md = query_md(const_fpd, DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    dnn_mem_t src_fp(data_md, fp, tag, ref_engine);
    dnn_mem_t src_dt(data_md, test_engine);

    dnn_mem_t dst_fp(data_md, fp, tag, ref_engine);
    dnn_mem_t dst_dt(data_md, test_engine);

    if (prb->dir & FLAG_INF) SAFE(ws_md.ndims == 0 ? OK : FAIL, WARN);
    dnn_mem_t ws_fp(ws_md, ref_engine);
    dnn_mem_t ws_dt(ws_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    dnn_mem_t d_dst_dt, d_src_dt;

    SAFE(fill_src(prb, src_dt, src_fp), WARN);

    args_t args, ref_args;

    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_WORKSPACE, ws_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (prb->dir & FLAG_FWD) {
        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);

            check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
        }
    }

    if (prb->dir & FLAG_BWD) {
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> tmp_prim;
        SAFE(init_prim(tmp_prim, init_pd, prb, res, FLAG_BWD, const_fpd), WARN);
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
        prim.reset(tmp_prim.release());

        auto const_bpd = query_pd(prim);

        if (check_mem_size(const_bpd) != OK) {
            return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
        }

        const auto &d_data_md = query_md(const_bpd, DNNL_ARG_DIFF_DST);
        const auto &d_scratchpad_md = query_md(const_bpd, DNNL_ARG_SCRATCHPAD);

        dnn_mem_t d_dst_fp(d_data_md, fp, tag, ref_engine);
        d_dst_dt = dnn_mem_t(d_data_md, test_engine);

        dnn_mem_t d_src_fp(d_data_md, fp, tag, ref_engine);
        d_src_dt = dnn_mem_t(d_data_md, test_engine);

        scratchpad_dt = dnn_mem_t(d_scratchpad_md, test_engine);

        SAFE(fill_dst(prb, d_dst_dt, d_dst_fp), WARN);

        args.clear();
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_WORKSPACE, ws_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);

            check_correctness(prb, {SRC}, args, ref_args, setup_cmp, res);
        }
    }

    return measure_perf(res, prim, args);
}

} // namespace lrn
