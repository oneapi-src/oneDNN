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

#include <algorithm>

#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "eltwise/eltwise.hpp"

namespace binary {

//TODO: Consider filling with powers of 2 for division to avoid rounding errors
int fill_mem(int input_idx, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        bool only_positive_values = false, bool only_integer_values = false) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const auto dt = mem_dt.dt();
    const int range = 16;
    const int f_min = dt == dnnl_u8 ? 0 : -range / 2;

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const int64_t gen = (12 * i + 5 * input_idx + 16) % (range + 1);
        const float scale = only_integer_values ? 1.f : 1.25f;
        float value = (f_min + gen) * scale;
        if (only_positive_values) value = fabs(value);
        // Remove zeroes in src1 to avoid division by zero
        if (input_idx == 1 && value == 0.0f) value = 1.0f;
        mem_fp.set_elem(i, round_to_nearest_representable(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int setup_binary_po(const_dnnl_primitive_desc_t pd, std::vector<int> &args,
        std::vector<dnn_mem_t> &mem_dt, std::vector<dnn_mem_t> &mem_fp,
        bool only_positive_values, bool only_integer_values) {
    // TODO: currently run-time dimensions are not supported in binary post-op.
    // To add a support two ways are possible: 1) add query support to the
    // library and extract expected md from pd; 2) pass a vector of pre-defined
    // (no run-time values) of `po_md`s and create memories from them in case
    // the library will lack of query mechanism.
    auto const_attr_po = query_post_ops(pd);
    auto po_len = dnnl_post_ops_len(const_attr_po);
    for (int idx = 0; idx < po_len; ++idx) {
        auto kind = dnnl_post_ops_get_kind(const_attr_po, idx);
        if (kind != dnnl_binary) continue;

        int po_idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1;
        const auto &po_md = query_md(pd, po_idx);

        // Following call can not be executed if po_md has runtime dimension due
        // to undefined size.
        mem_fp.emplace_back(po_md, dnnl_f32, tag::abx, get_cpu_engine());
        mem_dt.emplace_back(po_md, get_test_engine());
        args.push_back(po_idx);
        fill_mem(po_idx, mem_dt.back(), mem_fp.back(), only_positive_values,
                only_integer_values);
    }
    return OK;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    dnnl_binary_desc_t bd;
    std::vector<dnnl_memory_desc_t> src_d(prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const dims_t &i_vdims = prb->vdims[i_input];
        src_d[i_input] = dnn_mem_t::init_md(prb->ndims, i_vdims.data(),
                prb->sdt[i_input], prb->stag[i_input]);
    }

    auto dst_d = dnn_mem_t::init_md(
            prb->ndims, prb->dst_dims.data(), prb->ddt, prb->dtag);

    dnnl_alg_kind_t alg = attr_t::post_ops_t::kind2dnnl_kind(prb->alg);

    DNN_SAFE_STATUS(
            dnnl_binary_desc_init(&bd, alg, &src_d[0], &src_d[1], &dst_d));

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, prb->dst_dims.data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    return dnnl_primitive_desc_iterator_create(&init_pd_args.pd_it, &bd,
            dnnl_attr, init_pd_args.engine, init_pd_args.hint);
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    std::vector<dnnl_data_type_t> dts = {prb->sdt[0], prb->sdt[1], prb->ddt};
    skip_unimplemented_data_type(dts, prb->dir, res);
    skip_unimplemented_arg_scale(prb->attr, res);

    // N.B: Adding this for gpu as cfg is not supported in POST-OPS
    if (is_gpu()) {
        bool have_post_ops = !prb->attr.post_ops.is_def();
        bool is_bf16u8 = (dts[0] == dnnl_bf16 && dts[1] == dnnl_bf16
                && dts[2] == dnnl_u8);
        if (is_bf16u8 && have_post_ops) {
            res->state = SKIPPED, res->reason = DATA_TYPE_NOT_SUPPORTED;
            return;
        }
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    const bool is_sum = prb->attr.post_ops.find(alg_t::SUM) >= 0;
    bool bcast_src0 = false;
    for (int d = 0; d < prb->ndims; ++d)
        if (prb->vdims[0][d] != prb->vdims[1][d] && prb->vdims[0][d] == 1) {
            bcast_src0 = true;
            break;
        }

    // In case src0 is broadcasted into src1, it means that src0 has smaller
    // memory footprint and doing sum post-op or in-place will cause a crash.
    if (bcast_src0 && (prb->inplace || is_sum)) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }

    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        if (is_sum) {
            res->state = SKIPPED, res->reason = INVALID_CASE;
            return;
        }

        skip_invalid_inplace(
                res, prb->sdt[0], prb->ddt, prb->stag[0], prb->dtag);
        if (res->state == SKIPPED) return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    cmp.set_threshold(epsilon_dt(prb->ddt));
    // Since lambda is called when stack is unavailable, need to capture `prb`
    // by value to avoid using dangling references.
    const auto binary_add_check
            = [prb](const compare::compare_t::driver_check_func_args_t &args) {
                  // fp16 result can slightly mismatch for division due to
                  // difference in backends implementations.
                  return prb->alg == alg_t::DIV
                          ? args.diff < epsilon_dt(args.dt)
                          : false;
              };
    cmp.set_driver_check_function(binary_add_check);

    const std::vector<alg_t> cmp_alg = {
            alg_t::GE, alg_t::GT, alg_t::LE, alg_t::LT, alg_t::EQ, alg_t::NE};
    const bool is_cmp = std::any_of(
            cmp_alg.cbegin(), cmp_alg.cend(), [&](const alg_t alg) {
                return (prb->alg == alg) || prb->attr.post_ops.find(alg) >= 0;
            });

    if (is_cmp) cmp.set_zero_trust_percent(99.f);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);

    const auto &src0_md = query_md(const_pd, DNNL_ARG_SRC_0);
    const auto &src1_md = query_md(const_pd, DNNL_ARG_SRC_1);
    const auto &dst_md = query_md(const_pd, DNNL_ARG_DST);
    const auto &scratchpad_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    dnn_mem_t src0_fp(src0_md, fp, tag, ref_engine);
    dnn_mem_t src0_dt(src0_md, test_engine);
    SAFE(fill_mem(0, src0_dt, src0_fp), WARN);

    dnn_mem_t src1_fp(src1_md, fp, tag, ref_engine);
    dnn_mem_t src1_dt(src1_md, test_engine);
    SAFE(fill_mem(1, src1_dt, src1_fp), WARN);

    dnn_mem_t dst_fp(dst_md, fp, tag, ref_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    if (prb->attr.post_ops.find(alg_t::SUM) >= 0 || is_amd_gpu())
        SAFE(fill_mem(2, dst_dt, dst_fp), WARN);

    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(setup_binary_po(const_pd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    args_t args, ref_args;

    args.set(DNNL_ARG_SRC_0, src0_dt);
    args.set(DNNL_ARG_SRC_1, src1_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(binary_po_args, binary_po_dt);

    dnn_mem_t input_scales_m0;
    float scale0 = prb->attr.scales.get(DNNL_ARG_SRC_0).scale;
    maybe_prepare_runtime_scales(
            input_scales_m0, prb->attr.scales.get(DNNL_ARG_SRC_0), 1, &scale0);
    args.set(DNNL_ARG_ATTR_INPUT_SCALES | DNNL_ARG_SRC_0, input_scales_m0);
    dnn_mem_t input_scales_m1;
    float scale1 = prb->attr.scales.get(DNNL_ARG_SRC_1).scale;
    maybe_prepare_runtime_scales(
            input_scales_m1, prb->attr.scales.get(DNNL_ARG_SRC_1), 1, &scale1);
    args.set(DNNL_ARG_ATTR_INPUT_SCALES | DNNL_ARG_SRC_1, input_scales_m1);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (is_bench_mode(CORR)) {
        ref_args.set(DNNL_ARG_SRC_0, src0_fp);
        ref_args.set(DNNL_ARG_SRC_1, src1_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);
        ref_args.set(binary_po_args, binary_po_fp);

        check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace binary
