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

#include <cstring>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "ip/ip.hpp"

namespace ip {

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    dnnl_inner_product_desc_t ipd;

    auto src_d = dnn_mem_t::init_md(
            prb->ndims, prb->src_dims().data(), prb->cfg[SRC].dt, prb->stag);
    auto wei_d = dnn_mem_t::init_md(
            prb->ndims, prb->wei_dims().data(), prb->cfg[WEI].dt, prb->wtag);
    auto bia_d = dnn_mem_t::init_md(
            1, prb->bia_dims().data(), prb->cfg[BIA].dt, tag::any);
    auto dst_d = dnn_mem_t::init_md(
            2, prb->dst_dims().data(), prb->cfg[DST].dt, prb->dtag);

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            DNN_SAFE_STATUS(dnnl_inner_product_forward_desc_init(&ipd,
                    prb->dir == FWD_I ? dnnl_forward_inference
                                      : dnnl_forward_training,
                    &src_d, &wei_d, prb->dir == FWD_B ? &bia_d : nullptr,
                    &dst_d));
            break;
        case BWD_D:
            DNN_SAFE_STATUS(dnnl_inner_product_backward_data_desc_init(
                    &ipd, &src_d, &wei_d, &dst_d));
            break;
        case BWD_W:
        case BWD_WB:
            DNN_SAFE_STATUS(dnnl_inner_product_backward_weights_desc_init(&ipd,
                    &src_d, &wei_d, prb->dir == BWD_W ? nullptr : &bia_d,
                    &dst_d));
            break;
        default: DNN_SAFE_STATUS(dnnl_invalid_arguments);
    }

    DNN_SAFE_STATUS(ipd.accum_data_type == prb->cfg[ACC].dt
                    ? dnnl_success
                    : dnnl_unimplemented);

    attr_args_t attr_args;
    attr_args.prepare_output_scales(prb->attr, prb->scales, prb->oc);
    attr_args.prepare_post_ops_mds(prb->attr, 2, prb->dst_dims().data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    return dnnl_primitive_desc_iterator_create(&init_pd_args.pd_it, &ipd,
            dnnl_attr, init_pd_args.engine, init_pd_args.hint);
}

int init_prim_ref(
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim_ref, const prb_t *prb) {
    if (!(is_bench_mode(CORR) && is_gpu() && fast_ref_gpu)) return OK;

    // Create a new copy of prb to avoid potentially corrupting the test by
    // modifying prb in place.
    auto cpu_attr = prb->attr;
    update_cpu_ref_attrs(cpu_attr);
    prb_t prb_cpu {*prb, prb->mb, prb->dir, conf_f32, tag::abx, tag::abx,
            tag::abx, cpu_attr, prb->ctx_init, prb->ctx_exe};

    init_pd_args_t<prb_t> init_pd_args(
            /* res = */ nullptr, get_cpu_engine(), &prb_cpu, prb->dir,
            /* hint = */ nullptr);
    init_pd(init_pd_args);

    benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> pdw;
    benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_iterator_t> pd_itw;
    fetch_impl(pdw, pd_itw, init_pd_args, /* res = */ nullptr,
            /* is_service_prim = */ true);

    dnnl_primitive_t prim_ref_ {};
    if (pdw) {
        if (query_impl_info(pdw) == "ref:any") return OK;
        DNN_SAFE(dnnl_primitive_create(&prim_ref_, pdw), WARN);
        BENCHDNN_PRINT(5, "CPU reference oneDNN implementation: %s\n",
                query_impl_info(pdw).c_str());
    }
    prim_ref.reset(prim_ref_);
    return OK;
}

bool need_src_init(const prb_t *prb) {
    return !(prb->dir == BWD_D);
}

bool need_wei_init(const prb_t *prb) {
    return !(prb->dir & FLAG_BWD && prb->dir & FLAG_WEI);
}

bool need_bia_init(const prb_t *prb) {
    return need_wei_init(prb);
}

bool need_dst_init(const prb_t *prb) {
    return !(prb->dir & FLAG_FWD)
            || (prb->attr.post_ops.find(attr_t::post_ops_t::SUM) >= 0);
}

int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto &c = prb->get_dt_conf(SRC);
    const int range = c.f_max - c.f_min + 1;

    benchdnn_parallel_nd(prb->mb, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int64_t mb, int64_t ic, int64_t id, int64_t ih, int64_t iw) {
                const int gen
                        = 101 * id + 103 * ih + 107 * iw + 109 * mb + 113 * ic;
                const float sparsity = prb->ic < 5 ? 1.f : c.f_sparsity;
                const bool non_base = flip_coin(gen, sparsity);
                const float value
                        = non_base ? c.f_min + gen * 1 % range : c.f_base;
                ((float *)mem_fp)[src_off_f(prb, mb, ic, id, ih, iw)] = value;
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_wei(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const bool s8_s8
            = prb->cfg[WEI].dt == dnnl_s8 && prb->cfg[SRC].dt == dnnl_s8;

    const auto &c = prb->get_dt_conf(WEI);
    const int range = c.f_max - c.f_min + 1;

    benchdnn_parallel_nd(prb->oc, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int64_t oc, int64_t ic, int64_t kd, int64_t kh, int64_t kw) {
                const int gen = 127 * kd + 131 * kh + 137 * kw + 139 * oc
                        + 149 * ic + 7;
                const float sparsity = prb->ic < 5 ? 1.f : c.f_sparsity;
                const bool non_base = flip_coin(gen, sparsity);
                const float value
                        = non_base ? c.f_min + gen * 1 % range : c.f_base;
                ((float *)mem_fp)[wei_off_f(prb, oc, ic, kd, kh, kw)] = value;
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);
    if (s8_s8 && is_cpu()) {
        // Check that s8 -> s8_comp exists in the library since users may have
        // already quantized data.
        dnn_mem_t mem_fp_s8(mem_fp.md_, dnnl_s8, tag::abx, get_cpu_engine());
        dnn_mem_t mem_dt_s8(mem_dt.md_, get_test_engine());
        SAFE(mem_fp_s8.reorder(mem_fp), WARN);
        SAFE(mem_dt_s8.reorder(mem_fp_s8), WARN);
        SAFE(mem_dt.size() == mem_dt_s8.size() ? OK : FAIL, WARN);
        int rc = std::memcmp((void *)mem_dt, (void *)mem_dt_s8, mem_dt.size());
        SAFE(rc == 0 ? OK : FAIL, WARN);
    }

    return OK;
}

int fill_bia(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const size_t nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const auto &c = prb->get_dt_conf(BIA);
    const int range = c.f_max - c.f_min + 1;

    for (size_t i = 0; i < nelems; ++i) {
        const int gen = (int)(151 * i + 11);
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value = non_base ? c.f_min + gen * 1 % range : c.f_base;
        ((float *)mem_fp)[i] = value;
    }

    SAFE(mem_dt.reorder(mem_fp), WARN);
    return OK;
}

int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto &c = prb->get_dt_conf(DST);
    const int range = c.f_max - c.f_min + 1;

    benchdnn_parallel_nd(prb->mb, prb->oc, [&](int64_t mb, int64_t oc) {
        const int gen = 173 * mb + 179 * oc;
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value = non_base ? c.f_min + gen * 1 % range : c.f_base;

        ((float *)mem_fp)[dst_off_f(prb, mb, oc)] = value;
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type(
            {prb->cfg[SRC].dt, prb->cfg[WEI].dt, prb->cfg[DST].dt}, prb->dir,
            res);

    if (is_cpu()) {

        auto is_dt_f16_or_f32 = [&](dnnl_data_type_t dt) {
            return dt == dnnl_f16 || dt == dnnl_f32;
        };

        if (!IMPLICATION(prb->cfg[SRC].dt == dnnl_f16
                            || prb->cfg[WEI].dt == dnnl_f16
                            || prb->cfg[DST].dt == dnnl_f16,
                    is_dt_f16_or_f32(prb->cfg[SRC].dt)
                            && is_dt_f16_or_f32(prb->cfg[WEI].dt)
                            && is_dt_f16_or_f32(prb->cfg[DST].dt))) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        }
    }

    skip_unimplemented_sum_po(prb->attr, res, prb->get_dt_conf(DST).dt);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    cmp.set_threshold(prb->cfg[DST].eps);

    // TODO: why so bad filling?
    const float zero_trust_percent = kind == WEI || kind == BIA ? 90.f : 80.f;
    cmp.set_zero_trust_percent(zero_trust_percent);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);

    const auto &src_md = prb->dir == BWD_D
            ? query_md(const_pd, DNNL_ARG_DIFF_SRC)
            : query_md(const_pd, DNNL_ARG_SRC);
    const auto &wei_md = prb->dir & FLAG_WEI
            ? query_md(const_pd, DNNL_ARG_DIFF_WEIGHTS)
            : query_md(const_pd, DNNL_ARG_WEIGHTS);
    const auto &bia_md = prb->dir & FLAG_WEI
            ? query_md(const_pd, DNNL_ARG_DIFF_BIAS)
            : query_md(const_pd, DNNL_ARG_BIAS);
    const auto &dst_md = prb->dir & FLAG_BWD
            ? query_md(const_pd, DNNL_ARG_DIFF_DST)
            : query_md(const_pd, DNNL_ARG_DST);
    const auto &scratchpad_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto src_tag = tag::abx;
    const auto wei_tag = tag::abx;

    // Use CPU prim as the reference in GPU testing to reduce testing time.
    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim_ref;
    SAFE(init_prim_ref(prim_ref, prb), WARN);

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t bia_dt(bia_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    dnn_mem_t scales;
    maybe_prepare_runtime_scales(
            scales, prb->attr.oscale, prb->oc, prb->scales);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(
                 const_pd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    dnn_mem_t src_fp(src_md, fp, src_tag, ref_engine);
    dnn_mem_t wei_fp(wei_md, fp, wei_tag, ref_engine);
    dnn_mem_t bia_fp(bia_md, fp, tag::x, ref_engine);
    dnn_mem_t dst_fp(dst_md, fp, tag::abx, ref_engine);

    if (need_src_init(prb)) SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    if (need_wei_init(prb)) SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
    if (need_bia_init(prb)) SAFE(fill_bia(prb, bia_dt, bia_fp, res), WARN);
    if (need_dst_init(prb)) SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

    dnn_mem_t scratchpad_fp;
    if (prim_ref)
        scratchpad_fp = dnn_mem_t(
                query_md(query_pd(prim_ref), DNNL_ARG_SCRATCHPAD), ref_engine);

    args_t args, ref_args;

    if (prb->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_BIAS, bia_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
        args.set(DNNL_ARG_ATTR_OUTPUT_SCALES, scales);
        args.set(binary_po_args, binary_po_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_BIAS, bia_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(binary_po_args, binary_po_fp);
            ref_args.set(DNNL_ARG_SCRATCHPAD, scratchpad_fp);

            check_correctness(
                    prb, {DST}, args, ref_args, setup_cmp, res, prim_ref);
        }
    } else if (prb->dir == BWD_D) {
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_DIFF_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, dst_fp);
            ref_args.set(DNNL_ARG_SCRATCHPAD, scratchpad_fp);

            check_correctness(
                    prb, {SRC}, args, ref_args, setup_cmp, res, prim_ref);
        }
    } else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
        args.set(DNNL_ARG_DIFF_BIAS, bia_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DIFF_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, dst_fp);
            ref_args.set(DNNL_ARG_DIFF_BIAS, bia_fp);
            ref_args.set(DNNL_ARG_SCRATCHPAD, scratchpad_fp);

            check_correctness(
                    prb, {WEI, BIA}, args, ref_args, setup_cmp, res, prim_ref);
        }
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace ip
