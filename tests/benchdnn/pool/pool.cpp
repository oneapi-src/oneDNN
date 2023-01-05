/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
* Copyright 2022 Arm Ltd. and affiliates
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

#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "pool/pool.hpp"

namespace pool {

int fill_dat(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const int64_t MB {prb->mb};
    const int64_t IC {prb->ic};
    const int64_t D {kind == SRC ? prb->id : prb->od};
    const int64_t H {kind == SRC ? prb->ih : prb->oh};
    const int64_t W {kind == SRC ? prb->iw : prb->ow};
    const int64_t ker_size {prb->kd * prb->kh * prb->kw};
    const auto &c = prb->cfg[kind];
    // For huge kernels to get different output values filling should be very
    // variative, thus, use a factor of 1.
    const bool has_huge_kernel = ker_size >= c.f_max;

    benchdnn_parallel_nd(MB, IC, D, H, W,
            [&](int64_t mb, int64_t ic, int64_t d, int64_t h, int64_t w) {
                const int64_t factor
                        = prb->alg == max || has_huge_kernel ? 1 : ker_size;
                // keep values for avg_exclude_pad positive to prevent cancellation err
                const int64_t f_min = prb->alg == max ? c.f_min / factor : 0;
                // divide on factor to keep value in the range
                const int64_t range = c.f_max / factor - f_min + 1;
                const int64_t gen
                        = 5 * d + 17 * h + 13 * w + 13 * mb + 19 * ic + 1637;
                const float value = (f_min + gen % range) * factor;

                const size_t off = kind == SRC
                        ? src_off_f(prb, mb, ic, d, h, w)
                        : dst_off_f(prb, mb, ic, d, h, w);
                ((float *)mem_fp)[off] = value;
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

// fill ws with big numbers to reliably cause a correctness issue (and not
// anything else) in case of a bug in the library
int fill_ws(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const size_t nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    benchdnn_parallel_nd(
            nelems, [&](int64_t i) { mem_fp.set_elem(i, (1 << 24) - 1); });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    const dir_t dir = init_pd_args.dir;

    const auto src_tag = (dir & FLAG_FWD) ? prb->tag : tag::any;

    auto src_d = dnn_mem_t::init_md(
            prb->ndims, prb->src_dims().data(), prb->cfg[SRC].dt, src_tag);
    auto dst_d = dnn_mem_t::init_md(
            prb->ndims, prb->dst_dims().data(), prb->cfg[DST].dt, tag::any);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(
            prb->attr, prb->ndims, prb->dst_dims().data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    dnnl_alg_kind_t alg = alg2alg_kind(prb->alg);

    if (dir & FLAG_FWD) {
        auto prop_kind = prb->dir & FLAG_INF ? dnnl_forward_inference
                                             : dnnl_forward_training;
        DNN_SAFE_STATUS(dnnl_pooling_forward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, prop_kind, alg, src_d,
                dst_d, prb->strides().data(), prb->kernel().data(),
                prb->dilations().data(), prb->padding().data(),
                prb->padding_r().data(), dnnl_attr));
    } else {
        DNN_SAFE_STATUS(dnnl_pooling_backward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, alg, src_d, dst_d,
                prb->strides().data(), prb->kernel().data(),
                prb->dilations().data(), prb->padding().data(),
                prb->padding_r().data(), init_pd_args.hint, dnnl_attr));
    }
    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type(
            {prb->cfg[SRC].dt, prb->cfg[DST].dt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);

    if (is_cpu() && prb->cfg[SRC].dt != prb->cfg[DST].dt) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

#if DNNL_AARCH64_USE_ACL
    // Since ACL supports only forward pass.
    // Ref: https://github.com/oneapi-src/oneDNN/issues/1205
    if (prb->dir & FLAG_BWD) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
#endif
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // Average pooling without padding can't handle cases when kernel window is
    // applied to padded area only.
    if (prb->alg == avg_np) {
        bool ker_in_pad_d = prb->pd >= prb->kd || prb->pd_r >= prb->kd;
        bool ker_in_pad_h = prb->ph >= prb->kh || prb->ph_r >= prb->kh;
        bool ker_in_pad_w = prb->pw >= prb->kw || prb->pw_r >= prb->kw;
        bool ker_in_pad = ker_in_pad_d || ker_in_pad_h || ker_in_pad_w;

        if (ker_in_pad) {
            res->state = SKIPPED, res->reason = INVALID_CASE;
            return;
        }
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    cmp.set_threshold(prb->cfg[kind].eps);
    // Backward may have most zeroes for ker_in_pad with huge kernels problems.
    const float zero_percent = (prb->dir & FLAG_FWD) ? 99.f : 100.f;
    cmp.set_zero_trust_percent(zero_percent); // TODO: consider enabling

    const auto pooling_add_check
            = [&](const compare::compare_t::driver_check_func_args_t &args) {
                  // cuDNN bug: it spits fp16 min value as -inf,
                  // not -65504.
                  if (is_nvidia_gpu() && args.dt == dnnl_f16) {
                      return args.exp == lowest_dt(args.dt)
                              && std::isinf(args.got) && std::signbit(args.got);
                  }
                  return false;
              };
    cmp.set_driver_check_function(pooling_add_check);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    bool is_service_prim = prb->dir & FLAG_BWD;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res, FLAG_FWD, nullptr,
                 is_service_prim),
            WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
    if (!is_service_prim && is_bench_mode(INIT)) return OK;

    auto const_fpd = query_pd(prim);

    const auto &src_md = query_md(const_fpd, DNNL_ARG_SRC);
    const auto &dst_md = query_md(const_fpd, DNNL_ARG_DST);
    const auto &ws_md = query_md(const_fpd, DNNL_ARG_WORKSPACE);
    const auto &scratchpad_md = query_md(const_fpd, DNNL_ARG_SCRATCHPAD);

    SAFE(!check_md_consistency_with_tag(dst_md, prb->tag), WARN);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    dnn_mem_t src_fp(src_md, fp, tag, ref_engine);
    dnn_mem_t src_dt(src_md, test_engine);

    dnn_mem_t dst_fp(dst_md, fp, tag, ref_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);

    dnn_mem_t ws_fp(ws_md, dnnl_s32, tag::abx, ref_engine);
    dnn_mem_t ws_dt(ws_md, test_engine);
    if (prb->dir & FLAG_INF) SAFE(ws_dt.ndims() == 0 ? OK : FAIL, WARN);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(
                 const_fpd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    dnn_mem_t d_src_dt, d_dst_dt;

    SAFE(fill_dat(prb, SRC, src_dt, src_fp), WARN);

    args_t args, ref_args;

    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_WORKSPACE, ws_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(binary_po_args, binary_po_dt);

    if (!is_bench_mode(INIT)) SAFE(execute_and_wait(prim, args, res), WARN);

    // want this pass on backward to get ws_fp filled properly
    if (is_bench_mode(CORR)) {
        if (prb->dir & FLAG_FWD) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
            ref_args.set(binary_po_args, binary_po_fp);

            check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
        }
    }

    if (prb->dir & FLAG_BWD) {
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> tmp_prim;
        SAFE(init_prim(tmp_prim, init_pd, prb, res, FLAG_BWD, const_fpd), WARN);
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
        if (is_bench_mode(INIT)) return OK;
        prim.reset(tmp_prim.release());

        auto const_bpd = query_pd(prim);

        const auto &d_dst_md = query_md(const_bpd, DNNL_ARG_DIFF_DST);
        const auto &d_src_md = query_md(const_bpd, DNNL_ARG_DIFF_SRC);
        const auto &d_scratchpad_md = query_md(const_bpd, DNNL_ARG_SCRATCHPAD);

        dnn_mem_t d_dst_fp = dnn_mem_t(d_dst_md, fp, tag, ref_engine);
        d_dst_dt = dnn_mem_t(d_dst_md, test_engine);

        dnn_mem_t d_src_fp = dnn_mem_t(d_src_md, fp, tag, ref_engine);
        d_src_dt = dnn_mem_t(d_src_md, test_engine);

        scratchpad_dt = dnn_mem_t(d_scratchpad_md, test_engine);

        SAFE(fill_dat(prb, DST, d_dst_dt, d_dst_fp), WARN);

        args.clear();
        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_WORKSPACE, ws_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
            ref_args.set(binary_po_args, binary_po_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);

            check_correctness(prb, {SRC}, args, ref_args, setup_cmp, res);
        }
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace pool