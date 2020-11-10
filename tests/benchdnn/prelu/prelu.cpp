/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "prelu/prelu.hpp"

namespace prelu {

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &epd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_prelu_desc_t ed;
    dnnl_memory_desc_t data_d;
    dnnl_memory_desc_t weights_d;

    const auto &src_dims = prb->dims[0];
    const auto &weight_dims = prb->dims[1];

    SAFE(init_md(&data_d, prb->ndims, src_dims.data(), prb->dt, prb->tag),
            CRIT);

    auto weight_dims_temp = weight_dims;
    if (weight_dims.size() < src_dims.size()) { // need to reshape weights
        for (size_t d = weight_dims_temp.size(); d < src_dims.size(); ++d)
            weight_dims_temp.push_back(1);
    }

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&weights_d, (int)(src_dims.size()),
                     weight_dims_temp.data(), prb->dt, dnnl_format_tag_any),
            WARN);

    if (prb->dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        DNN_SAFE(dnnl_prelu_forward_desc_init(&ed, prop, &data_d, &weights_d),
                WARN);
    } else {
        dnnl_memory_desc_t diff_data_d;
        dnnl_memory_desc_t diff_weights_d;
        SAFE(init_md(&diff_data_d, prb->ndims, src_dims.data(), prb->dt,
                     prb->tag),
                CRIT);

        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_weights_d, prb->ndims,
                         weight_dims_temp.data(), prb->dt, dnnl_format_tag_any),
                WARN);

        DNN_SAFE(dnnl_prelu_backward_desc_init(&ed, &data_d, &weights_d,
                         &diff_data_d, &diff_weights_d),
                WARN);
    }

    auto dnnl_attr = create_dnnl_attr(prb->attr, attr_args_t());

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&epd, &ed, dnnl_attr, engine, hint);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(epd);

    // Return if pd is not the one being tested
    if ((dir & FLAG_FWD) != (prb->dir & FLAG_FWD)) return OK;

    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        DNN_SAFE(dnnl_primitive_desc_destroy(epd), WARN);
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    return OK;
}

static int compare(const prb_t *prb, data_kind_t kind,
        const dnn_mem_t &mem_arg_fp, const dnn_mem_t &mem_fp,
        const dnn_mem_t &mem_dt, res_t *res) {
    const auto nelems = mem_dt.nelems();
    const auto trh = 2 * epsilon_dt(prb->dt); //TODO: check if narrow down

    if (nelems == 0) return res->state = PASSED, OK;

    res->total = nelems;

    for (int64_t i = 0; i < nelems; i++) {
        const float dt = mem_dt.get_elem(i);
        const float src = mem_arg_fp.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = round_to_nearest_representable(prb->dt, fp0);
        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);

        bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= trh;

        res->errors += !ok;

        const bool dump = (!ok && (res->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            std::stringstream ss;
            dims_t dims_idx = (kind == WEI) ? off2dims_idx(prb->dims[1], i)
                                            : off2dims_idx(prb->dims[0], i);
            ss << dims_idx;
            std::string ind_str = ss.str();

            BENCHDNN_PRINT(0,
                    "[%4ld][%s][%s] src:% 9.6g fp0:% 9.6g fp:% 9.6g dt:% 9.6g "
                    "diff:%8.3g rdiff:%8.3g\n",
                    (long)i, data_kind2str(kind), ind_str.c_str(), src, fp0, fp,
                    dt, diff, rel_diff);
        }
    }

    if (res->errors) res->state = FAILED;

    if (res->state == UNTESTED) res->state = PASSED; /* optimism */

    return res->state == FAILED ? FAIL : OK;
}

int fill_data(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    dnnl::impl::parallel_nd(n_chunks, [&](int idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note 1: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // we avoid it for two reasons:
        //   a. it has a complexity in O(idx_start).
        //   b. igen and fgen below might require more than 1 sample
        //   per idx, so the we cannot deterministically compute the
        //   number of states we need to discard
        // Note 2: We also advance the state to avoid having only
        // small values as first chunk input.  The +1 is necessary to
        // avoid generating zeros in first chunk.
        // Note 3: we multiply by kind + 1 to have different values in
        // src/dst and diff_dst. The +1 is to avoid 0 again.
        std::minstd_rand msr((idx_start + 1) * (kind + 1));
        msr.discard(1);
        std::uniform_int_distribution<> igen(0, 10);
        // TODO: 0.09 due to log impl doesn't give good accuracy in 0.99 points
        std::uniform_real_distribution<> fgen(0.f, 0.09f);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value = FLT_MAX;
            if (kind == WEI) {
                value = 10.f * fgen(msr); // [0.-1.) pos
            } else {
                switch (idx % 8) {
                    case 0: value = (float)igen(msr); break; // [0-10] pos
                    case 1: value = -(float)igen(msr); break; // [0-10] neg
                    case 2: value = fgen(msr); break; // [0.-0.1) pos
                    case 3: value = -fgen(msr); break; // [0.-0.1) neg
                    case 4: value = 10 * (float)igen(msr); break; // [0-100] pos
                    case 5:
                        value = -10 * (float)igen(msr);
                        break; // [0-100] neg
                    case 6: value = 10.f * fgen(msr); break; // [0.-1.) pos
                    case 7: value = -10.f * fgen(msr); break; // [0.-1.) neg
                }
            }
            value = round_to_nearest_representable(prb->dt, value);

            // Hack: -0 may lead to different sign in the answer since input
            // passes through simple reorder which converts -0 into +0.
            if (value == -0.f) value = 0.f;

            mem_fp.set_elem(idx, value);
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    dnnl_primitive_t e {};
    SAFE(init_prim(&e, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(e, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(e));
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto &data_md = q(DNNL_ARG_SRC);
    const auto &weight_md = q(DNNL_ARG_WEIGHTS);
    const auto &d_data_md = q(DNNL_ARG_DIFF_DST);
    const auto &d_weights_md = q(DNNL_ARG_DIFF_WEIGHTS);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_fp(data_md, fp, tag, test_engine);
    dnn_mem_t src_dt(data_md, test_engine);

    dnn_mem_t weights_fp(weight_md, fp, tag, test_engine);
    dnn_mem_t weights_dt(weight_md, test_engine);

    SAFE(fill_data(prb, SRC, src_dt, src_fp), WARN);
    SAFE(fill_data(prb, WEI, weights_dt, weights_fp), WARN);

    args_t args;
    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_WEIGHTS, weights_dt);

    dnn_mem_t dst_fp, dst_dt, d_src_fp, d_src_dt, d_dst_fp, d_dst_dt,
            d_weights_fp, d_weights_dt, scratchpad_dt;

    if (prb->dir & FLAG_FWD) {
        dst_fp = dnn_mem_t(data_md, fp, tag, test_engine);
        dst_dt = dnn_mem_t(data_md, test_engine);

        args.set(DNNL_ARG_DST, dst_dt);
        SAFE(execute_and_wait(e, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(prb, src_fp, weights_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, tag, test_engine);
            SAFE(compare(prb, SRC, src_fp, dst_fp, dst, res), WARN);
        }
    } else {
        d_src_fp = dnn_mem_t(d_data_md, fp, tag, test_engine);
        d_src_dt = dnn_mem_t(d_data_md, test_engine);

        d_dst_fp = dnn_mem_t(d_data_md, fp, tag, test_engine);
        d_dst_dt = dnn_mem_t(d_data_md, test_engine);

        d_weights_fp = dnn_mem_t(d_weights_md, fp, tag, test_engine);
        d_weights_dt = dnn_mem_t(d_weights_md, test_engine);

        scratchpad_dt = dnn_mem_t(scratchpad_md, test_engine);

        SAFE(fill_data(prb, DST, d_dst_dt, d_dst_fp), WARN);

        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, d_weights_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
        SAFE(execute_and_wait(e, args), WARN);

        if (bench_mode & CORR) {
            dnn_mem_t &arg_fp = src_fp;
            compute_ref_bwd(
                    prb, src_fp, weights_fp, d_src_fp, d_dst_fp, d_weights_fp);
            dnn_mem_t d_src(d_src_dt, fp, tag, test_engine);
            dnn_mem_t d_weights(d_weights_dt, fp, tag, test_engine);

            SAFE(compare(prb, SRC, arg_fp, d_src_fp, d_src, res), WARN);
            SAFE(compare(prb, WEI, weights_fp, d_weights_fp, d_weights, res),
                    WARN);
        }
    }

    measure_perf(res->timer, e, args);

    DNN_SAFE_V(dnnl_primitive_destroy(e));

    return OK;
}

} // namespace prelu
