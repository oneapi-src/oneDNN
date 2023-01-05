/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#include "utils/fill.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "ip/ip.hpp"

namespace ip {

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    auto src_d = dnn_mem_t::init_md(
            prb->ndims, prb->src_dims().data(), prb->cfg[SRC].dt, prb->stag);
    auto wei_d = dnn_mem_t::init_md(
            prb->ndims, prb->wei_dims().data(), prb->cfg[WEI].dt, prb->wtag);
    auto bia_d = dnn_mem_t::init_md(
            1, prb->bia_dims().data(), prb->cfg[BIA].dt, tag::any);
    auto dst_d = dnn_mem_t::init_md(
            2, prb->dst_dims().data(), prb->cfg[DST].dt, prb->dtag);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, 2, prb->dst_dims().data());
    auto wei_scale = prb->attr.scales.get(DNNL_ARG_WEIGHTS);
    if (wei_scale.policy == policy_t::PER_OC) {
        attr_args.prepare_scales(prb->attr, DNNL_ARG_WEIGHTS, 1);
    }
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            if (prb->dir != FWD_B) bia_d.reset(nullptr);
            DNN_SAFE_STATUS(dnnl_inner_product_forward_primitive_desc_create(
                    &init_pd_args.pd, init_pd_args.engine,
                    prb->dir == FWD_I ? dnnl_forward_inference
                                      : dnnl_forward_training,
                    init_pd_args.src_md ? init_pd_args.src_md : src_d, wei_d,
                    bia_d, dst_d, dnnl_attr));
            break;
        case BWD_D:
            DNN_SAFE_STATUS(
                    dnnl_inner_product_backward_data_primitive_desc_create(
                            &init_pd_args.pd, init_pd_args.engine, src_d, wei_d,
                            dst_d, init_pd_args.hint, dnnl_attr));
            break;
        case BWD_W:
        case BWD_WB:
            if (prb->dir == BWD_W) bia_d.reset(nullptr);
            DNN_SAFE_STATUS(
                    dnnl_inner_product_backward_weights_primitive_desc_create(
                            &init_pd_args.pd, init_pd_args.engine, src_d, wei_d,
                            bia_d, dst_d, init_pd_args.hint, dnnl_attr));
            break;
        default: DNN_SAFE_STATUS(dnnl_invalid_arguments);
    }

    return dnnl_success;
}

int init_prim_ref(
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim_ref, const prb_t *prb) {
    if (!(has_bench_mode_bit(mode_bit_t::corr) && is_gpu() && fast_ref_gpu))
        return OK;

    // Create a new copy of prb to avoid potentially corrupting the test by
    // modifying prb in place.
    auto cpu_attr = prb->attr;
    update_cpu_ref_attrs(cpu_attr);
    prb_t prb_cpu {*prb, prb->mb, prb->dir, conf_f32, tag::abx, tag::abx,
            tag::abx, cpu_attr, prb->ctx_init, prb->ctx_exe};

    init_pd_args_t<prb_t> init_pd_args(
            /* res = */ nullptr, get_cpu_engine(), &prb_cpu, prb->dir,
            /* hint = */ nullptr, /* src_md = */ nullptr);
    init_pd(init_pd_args);

    benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> pdw;
    fetch_impl(pdw, init_pd_args, /* res = */ nullptr,
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

int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const size_t nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const auto &c = prb->get_dt_conf(SRC);
    const int range = c.f_max - c.f_min + 1;
    const float sparsity
            = (!has_bench_mode_bit(mode_bit_t::corr) || prb->ic < 5)
            ? 1.f
            : c.f_sparsity;

    benchdnn_parallel_nd(prb->mb, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int64_t mb, int64_t ic, int64_t id, int64_t ih, int64_t iw) {
                const int gen
                        = 101 * id + 103 * ih + 107 * iw + 109 * mb + 113 * ic;
                const bool non_base = flip_coin(gen, sparsity);
                const float value
                        = non_base ? c.f_min + gen * 1 % range : c.f_base;
                ((float *)mem_fp)[src_off_f(prb, mb, ic, id, ih, iw)]
                        = round_to_nearest_representable(mem_dt.dt(), value);
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_wei(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const size_t nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const auto &c = prb->get_dt_conf(WEI);
    const int range = c.f_max - c.f_min + 1;
    const float sparsity
            = (!has_bench_mode_bit(mode_bit_t::corr) || prb->ic < 5)
            ? 1.f
            : c.f_sparsity;

    benchdnn_parallel_nd(prb->oc, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int64_t oc, int64_t ic, int64_t kd, int64_t kh, int64_t kw) {
                const int gen = 127 * kd + 131 * kh + 137 * kw + 139 * oc
                        + 149 * ic + 7;
                const bool non_base = flip_coin(gen, sparsity);
                const float value
                        = non_base ? c.f_min + gen * 1 % range : c.f_base;
                ((float *)mem_fp)[wei_off_f(prb, oc, ic, kd, kh, kw)]
                        = round_to_nearest_representable(mem_dt.dt(), value);
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    const bool s8_s8
            = prb->cfg[WEI].dt == dnnl_s8 && prb->cfg[SRC].dt == dnnl_s8;
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
        ((float *)mem_fp)[i]
                = round_to_nearest_representable(mem_dt.dt(), value);
    }

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const size_t nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const auto &c = prb->get_dt_conf(DST);
    const int range = c.f_max - c.f_min + 1;

    benchdnn_parallel_nd(prb->mb, prb->oc, [&](int64_t mb, int64_t oc) {
        const int gen = 173 * mb + 179 * oc;
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value = non_base ? c.f_min + gen * 1 % range : c.f_base;

        ((float *)mem_fp)[dst_off_f(prb, mb, oc)]
                = round_to_nearest_representable(mem_dt.dt(), value);
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

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_BIAS,
            DNNL_ARG_DST,
    };
    static const std::vector<int> exec_bwd_d_args = {
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_BIAS,
            DNNL_ARG_DIFF_DST,
    };
    static const std::vector<int> exec_bwd_w_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DIFF_WEIGHTS,
            DNNL_ARG_DIFF_BIAS,
            DNNL_ARG_DIFF_DST,
    };
    return (dir & FLAG_FWD)    ? exec_fwd_args
            : (dir & FLAG_WEI) ? exec_bwd_w_args
                               : exec_bwd_d_args;
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref) {
    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        ref_mem_map.emplace(
                exec_arg, dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_src(prb, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_WEIGHTS:
                SAFE(fill_wei(prb, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_BIAS:
                SAFE(fill_bia(prb, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_DST:
                if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM)
                        >= 0)
                    SAFE(fill_dst(prb, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_dst(prb, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_SCRATCHPAD:
                // Reference CPU impl may need a different size for scratchpad.
                // Need to query it instead of replicating one from GPU.
                if (prim_ref) {
                    ref_mem_map[exec_arg] = dnn_mem_t(
                            query_md(query_pd(prim_ref), DNNL_ARG_SCRATCHPAD),
                            ref_engine);
                }
                break;
            default: { // Process all attributes here
                int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
                        - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
                bool is_post_ops_arg = (exec_arg & post_ops_range);
                bool is_scales_arg = (exec_arg & DNNL_ARG_ATTR_SCALES);

                if (is_post_ops_arg) {
                    if (exec_arg & DNNL_ARG_SRC_1)
                        SAFE(binary::fill_mem(exec_arg, mem, ref_mem), WARN);
                } else if (is_scales_arg) {
                    int local_exec_arg = exec_arg ^ DNNL_ARG_ATTR_SCALES;
                    SAFE(fill_scales(prb->attr, local_exec_arg, mem, ref_mem),
                            WARN);
                }
            } break;
        }
        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds;
    if (prb->dir & FLAG_FWD) {
        check_kinds = {DST};
    } else if (prb->dir == BWD_D) {
        check_kinds = {SRC};
    } else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI) {
        check_kinds = {WEI, BIA};
    } else {
        assert(!"unexpected!");
        SAFE_V(FAIL);
    }
    assert(!check_kinds.empty());
    return check_kinds;
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == bench_mode_t::list) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
    if (bench_mode == bench_mode_t::init) return OK;

    // Use CPU prim as the reference in GPU testing to reduce testing time.
    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim_ref;
    SAFE(init_prim_ref(prim_ref, prb), WARN);

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(mem_map, prb, prim, supported_exec_args(prb->dir));
    SAFE(init_ref_memory_args(
                 ref_mem_map, mem_map, prim, prb, res, prb->dir, prim_ref),
            WARN);

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (has_bench_mode_bit(mode_bit_t::corr)) {
        check_correctness(prb, get_kinds_to_check(prb), args, ref_args,
                setup_cmp, res, prim_ref);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace ip
