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

#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "resampling/resampling.hpp"

namespace resampling {

int fill_dat(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    const auto dt = mem_dt.dt();
    const int range = 16;
    const int f_min = 0;

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) - 19 * kind + 101) % (range + 1);
        const float value = dt == dnnl_f32 || is_integral_dt(dt)
                ? (f_min + gen) * (1.0f + 4.0f / range)
                : (f_min + gen) / range;

        mem_fp.set_elem(i, round_to_nearest_representable(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    return fill_dat(prb, SRC, mem_dt, mem_fp, res);
}

int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    return fill_dat(prb, DST, mem_dt, mem_fp, res);
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    std::string src_tag = (prb->dir & FLAG_FWD) ? prb->tag : tag::any;
    std::string dst_tag = (prb->dir & FLAG_BWD) ? prb->tag : tag::any;

    auto src_d = dnn_mem_t::init_md(
            prb->ndims, prb->src_dims().data(), prb->sdt, src_tag);
    auto dst_d = dnn_mem_t::init_md(
            prb->ndims, prb->dst_dims().data(), prb->ddt, dst_tag);

    dnnl_alg_kind_t alg = alg2alg_kind(prb->alg);
    dnnl_resampling_desc_t rd;

    if (prb->dir & FLAG_FWD) {
        auto prop_kind = prb->dir & FLAG_INF ? dnnl_forward_inference
                                             : dnnl_forward_training;
        DNN_SAFE_STATUS(dnnl_resampling_forward_desc_init(
                &rd, prop_kind, alg, nullptr, &src_d, &dst_d));
    } else {
        DNN_SAFE_STATUS(dnnl_resampling_backward_desc_init(
                &rd, alg, nullptr, &src_d, &dst_d));
    }

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(
            prb->attr, prb->ndims, prb->dst_dims().data());
    const auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    return dnnl_primitive_desc_iterator_create(&init_pd_args.pd_it, &rd,
            dnnl_attr, init_pd_args.engine, init_pd_args.hint);
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->sdt, prb->ddt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const auto dt_from = (prb->dir & FLAG_FWD) ? prb->sdt : prb->ddt;
    const auto dt_to = (prb->dir & FLAG_FWD) ? prb->ddt : prb->sdt;
    const float linear_trh = epsilon_dt(dt_from) > epsilon_dt(dt_to)
            ? epsilon_dt(dt_from) // conversion error for dt_to
            : 7 * epsilon_dt(dt_to); // algorithm calculation error
    float trh = prb->alg == nearest ? 0.f : linear_trh;
    if (is_nvidia_gpu()) {
        // cuDNN precision is different from ref one due to different
        // computation algorithm used for resampling.
        trh = prb->ddt == dnnl_f16 ? 4e-2 : 2e-5;
    }
    cmp.set_threshold(trh);

    // No sense to test zero trust for upsampling since it produces valid zeros.
    // TODO: validate this once again.
    cmp.set_zero_trust_percent(99.f);

    // Resampling on integer data types may result in sporadic order of
    // operations. This may cause a difference around `x.5` floating-point, and
    // can be rounded either way to `x` or to `x+1` which can't be fixed by
    // filling.
    const auto resampling_add_check
            = [&](const compare::compare_t::driver_check_func_args_t &args) {
                  if (!is_integral_dt(args.dt)) return false;
                  // Check that original value is close to x.5f
                  static constexpr float small_eps = 9e-6;
                  if (fabsf((floorf(args.exp_f32) + 0.5f) - args.exp_f32)
                          >= small_eps)
                      return false;
                  // If it was, check that exp and got values reside on opposite
                  // sides of it.
                  if (args.exp == floorf(args.exp_f32))
                      return args.got == ceilf(args.exp_f32);
                  else if (args.exp == ceilf(args.exp_f32))
                      return args.got == floorf(args.exp_f32);
                  else {
                      assert(!"unexpected scenario");
                      return false;
                  }
              };
    if (prb->alg == linear) cmp.set_driver_check_function(resampling_add_check);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);

    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto &src_md = prb->dir == BWD_D
            ? query_md(const_pd, DNNL_ARG_DIFF_SRC)
            : query_md(const_pd, DNNL_ARG_SRC);
    const auto &dst_md = prb->dir == BWD_D
            ? query_md(const_pd, DNNL_ARG_DIFF_DST)
            : query_md(const_pd, DNNL_ARG_DST);
    const auto &scratchpad_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    dnn_mem_t src_fp(src_md, fp, tag, ref_engine);
    dnn_mem_t src_dt(src_md, test_engine);

    dnn_mem_t dst_fp(dst_md, fp, tag, ref_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0)
        SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    // When post-ops occur, the relative difference can change
    // between the output from reference and the kernel. The compare
    // function usually uses to compare a relative difference.
    // Therefore, we should not lead to a situation where the
    // relative difference is very small after executing a
    // post-ops operation. Therefore, all values for binary post_ops
    // are positive when the linear algorithm is present. This is
    // important because there may be small differences in the result
    // between the expected value and the gotten value with this algorithm.
    const bool only_positive_values = prb->alg == linear;
    SAFE(binary::setup_binary_po(const_pd, binary_po_args, binary_po_dt,
                 binary_po_fp, only_positive_values),
            WARN);

    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    args_t args, ref_args;

    if (prb->dir & FLAG_FWD) {
        SAFE(fill_src(prb, src_dt, src_fp, res), WARN);

        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(binary_po_args, binary_po_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(binary_po_args, binary_po_fp);

            check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
        }
    } else {
        SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_DIFF_DST, dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, src_fp);

            check_correctness(prb, {SRC}, args, ref_args, setup_cmp, res);
        }
    }

    return measure_perf(res, prim, args);
}

} // namespace resampling
