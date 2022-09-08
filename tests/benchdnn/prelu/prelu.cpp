/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "prelu/prelu.hpp"

namespace prelu {

int fill_data(data_kind_t kind, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Do fixed partitioning to have same filling for any number of threads.
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note 1: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // we avoid it for two reasons:
        //   a. it has a complexity in O(idx_start).
        //   b. igen below might require more than 1 sample
        //   per idx, so the we cannot deterministically compute the
        //   number of states we need to discard
        // Note 2: We also advance the state to avoid having only
        // small values as first chunk input.  The +1 is necessary to
        // avoid generating zeros in first chunk.
        // Note 3: we multiply by kind + 1 to have different values in
        // src/dst and diff_dst. The +1 is to avoid 0 again.
        std::minstd_rand msr((idx_start + 1) * (kind + 1));
        msr.discard(1);
        std::uniform_int_distribution<> igen_02(0, 2), igen_05(0, 5),
                igen_06(0, 6);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value = 0;
            if (is_integral_dt(mem_dt.dt())) {
                value = igen_05(msr);
            } else {
                // TODO: amount of negative values should depend on number of points
                // to reduce as summation becomes inaccurate.
                switch (kind) {
                    case SRC: value = igen_02(msr); break;
                    case WEI:
                        value = (64 >> igen_06(msr)) / 8.f; // pow2 [0.125f, 8f]
                        break;
                    case DST: value = igen_02(msr) / 16.f; break;
                    default: assert(!"unexpected"); break;
                }
            }
            float sign = mem_dt.dt() == dnnl_u8
                    ? 1.f
                    : flip_coin(idx, 0.1f) ? -1.f : 1.f;
            value = round_to_nearest_representable(mem_dt.dt(), sign * value);
            mem_fp.set_elem(idx, value);
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int setup_prelu_po(const_dnnl_primitive_desc_t pd, std::vector<int> &args,
        std::vector<dnn_mem_t> &ref_mem, std::vector<dnn_mem_t> &prim_mem) {
    const auto &dst_md = query_md(pd, DNNL_ARG_DST);
    auto const_attr_po = query_post_ops(pd);
    const auto po_len = dnnl_post_ops_len(const_attr_po);
    for (int idx = 0; idx < po_len; ++idx) {
        const auto kind = dnnl_post_ops_get_kind(const_attr_po, idx);
        if (kind != dnnl_prelu) continue;

        const auto ndims = dst_md.ndims;
        int mask = 0;
        dnnl_dims_t dims = {0};
        dnnl_post_ops_get_params_prelu(const_attr_po, idx, &mask);

        // Deduce prelu weights dims based on input policy.
        for (int d = 0; d < ndims; ++d) {
            dims[d] = (mask & (1 << d)) ? dst_md.dims[d] : 1;
        }

        // Following call can not be executed if po_md has runtime dimension due
        // to undefined size.
        ref_mem.emplace_back(ndims, dims, dnnl_f32, tag::abx, get_cpu_engine());
        prim_mem.emplace_back(
                ndims, dims, dnnl_f32, tag::axb, get_test_engine());
        args.push_back(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_WEIGHTS);
        fill_data(WEI, prim_mem.back(), ref_mem.back());
    }
    return OK;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    const auto &src_dims = prb->vdims[0];
    const auto &weight_dims = prb->vdims[1];

    auto data_d = dnn_mem_t::init_md(
            prb->ndims, src_dims.data(), prb->sdt[0], prb->stag[0]);
    auto weights_d = dnn_mem_t::init_md(
            prb->ndims, weight_dims.data(), prb->sdt[1], prb->stag[1]);

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    if (prb->dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        DNN_SAFE_STATUS(dnnl_prelu_forward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, prop, &data_d,
                &weights_d, dnnl_attr));
    } else {
        auto diff_data_d = dnn_mem_t::init_md(
                prb->ndims, src_dims.data(), prb->sdt[0], prb->stag[0]);
        auto diff_weights_d = dnn_mem_t::init_md(
                prb->ndims, weight_dims.data(), prb->sdt[1], prb->stag[1]);

        DNN_SAFE_STATUS(dnnl_prelu_backward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, &data_d, &weights_d,
                &diff_data_d, &diff_weights_d, init_pd_args.hint, dnnl_attr));
    }

    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type(prb->sdt, FWD_D, res);
    skip_unimplemented_sum_po(prb->attr, res);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const auto trh_dt = kind == WEI ? prb->sdt[1] : prb->sdt[0];
    cmp.set_threshold(2 * epsilon_dt(trh_dt));

    // Weights are very sparse, no sense to test for trust, otherwise filling
    // is specific to cover half non-zeros only.
    const float zero_trust_percent = kind == WEI ? 99.f : 50.f;
    cmp.set_zero_trust_percent(zero_trust_percent);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);

    const auto &data_md = query_md(const_pd, DNNL_ARG_SRC);
    const auto &weight_md = query_md(const_pd, DNNL_ARG_WEIGHTS);
    const auto &scratchpad_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);
    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    dnn_mem_t src_fp(data_md, dnnl_f32, tag::abx, ref_engine);
    dnn_mem_t weights_fp(weight_md, dnnl_f32, tag::abx, ref_engine);

    dnn_mem_t src_dt(data_md, test_engine);
    dnn_mem_t weights_dt(weight_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    SAFE(fill_data(SRC, src_dt, src_fp), WARN);
    SAFE(fill_data(WEI, weights_dt, weights_fp), WARN);

    args_t args, ref_args;

    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_WEIGHTS, weights_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    dnn_mem_t dst_dt, d_src_fp, d_src_dt, d_dst_fp, d_dst_dt, d_weights_fp,
            d_weights_dt;

    if (prb->dir & FLAG_FWD) {
        dnn_mem_t dst_fp(data_md, dnnl_f32, tag::abx, ref_engine);
        dst_dt = dnn_mem_t(data_md, test_engine);

        args.set(DNNL_ARG_DST, dst_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, weights_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);

            check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
        }
    } else {
        const auto &d_data_md = query_md(const_pd, DNNL_ARG_DIFF_DST);
        const auto &d_weights_md = query_md(const_pd, DNNL_ARG_DIFF_WEIGHTS);

        dnn_mem_t d_src_fp(d_data_md, dnnl_f32, tag::abx, ref_engine);
        dnn_mem_t d_weights_fp(d_weights_md, dnnl_f32, tag::abx, ref_engine);
        dnn_mem_t d_dst_fp(d_data_md, dnnl_f32, tag::abx, ref_engine);

        d_src_dt = dnn_mem_t(d_data_md, test_engine);
        d_weights_dt = dnn_mem_t(d_weights_md, test_engine);
        d_dst_dt = dnn_mem_t(d_data_md, test_engine);

        SAFE(fill_data(DST, d_dst_dt, d_dst_fp), WARN);

        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, d_weights_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, weights_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);
            ref_args.set(DNNL_ARG_DIFF_WEIGHTS, d_weights_fp);

            check_correctness(prb, {SRC, WEI}, args, ref_args, setup_cmp, res);
        }
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace prelu
