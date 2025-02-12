/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "utils/fill.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "resampling/resampling.hpp"

namespace resampling {

int fill_dat(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

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

int fill_src(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_dat(prb, SRC, mem_dt, mem_fp);
}

int fill_dst(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_dat(prb, DST, mem_dt, mem_fp);
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    std::string src_tag = (prb->dir & FLAG_FWD) ? prb->tag : tag::any;
    std::string dst_tag = (prb->dir & FLAG_BWD) ? prb->tag : tag::any;

    auto src_d = dnn_mem_t::init_md(prb->ndims, prb->src_dims().data(),
            force_f32_dt ? dnnl_f32 : prb->sdt, src_tag);
    auto dst_d = dnn_mem_t::init_md(prb->ndims, prb->dst_dims().data(),
            force_f32_dt ? dnnl_f32 : prb->ddt, dst_tag);

    dnnl_alg_kind_t alg = alg2alg_kind(prb->alg);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(
            prb->attr, prb->ndims, prb->dst_dims().data());
    const auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    if (prb->dir & FLAG_FWD) {
        auto prop_kind = prb->dir & FLAG_INF ? dnnl_forward_inference
                                             : dnnl_forward_training;
        TIME_C_PD(DNN_SAFE_STATUS(dnnl_resampling_forward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, prop_kind, alg, nullptr,
                init_pd_args.src_md ? init_pd_args.src_md : src_d, dst_d,
                dnnl_attr)));
    } else {
        TIME_C_PD(
                DNN_SAFE_STATUS(dnnl_resampling_backward_primitive_desc_create(
                        &init_pd_args.pd, init_pd_args.engine, alg, nullptr,
                        src_d, dst_d, init_pd_args.hint, dnnl_attr)));
    }
    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->sdt, prb->ddt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_resampling, prb->sdt);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_resampling);
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
        trh = (prb->ddt == dnnl_f16 || prb->sdt == dnnl_bf16
                      || prb->sdt == dnnl_f16 || prb->ddt == dnnl_bf16)
                ? 4e-2
                : 2e-5;
    }
    cmp.set_threshold(trh);

    // No sense to test zero trust for upsampling since it produces valid zeros.
    // TODO: validate this once again.
    cmp.set_zero_trust_percent(99.f);
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DST,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_DIFF_DST,
    };
    return (dir & FLAG_FWD) ? exec_fwd_args : exec_bwd_args;
};

fill_cfg_t binary_po_fill_cfg(
        int exec_arg, const dnn_mem_t &mem, const attr_t &attr) {
    fill_cfg_t cfg;
    const int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
            - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
    const bool is_post_ops_arg = (exec_arg & post_ops_range);
    if (is_post_ops_arg) {
        // Config secures only positive values since resampling inputs are only
        // positive, and using negative values leads to the cancellation effect.
        const int bin_po_idx
                = exec_arg / DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE - 1;
        assert(bin_po_idx < attr.post_ops.len());
        const auto alg = attr.post_ops.entry[bin_po_idx].kind;
        cfg = fill_cfg_t(mem.dt(), 0.f, 16.f, /* int = */ true, alg,
                "resampling_binary_post_op");
    }
    return cfg;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // Scratchpad memory relates to a primitive. If reference needs it,
        // use switch below to define a memory desc for it.
        if (exec_arg != DNNL_ARG_SCRATCHPAD) {
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
        }
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC: SAFE(fill_src(prb, mem, ref_mem), WARN); break;
            case DNNL_ARG_DST:
                if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM)
                        >= 0) {
                    SAFE(fill_dst(prb, mem, ref_mem), WARN);

                    // Bitwise mode for sum requires a copy due to data for
                    // post-op will be overwritten and it must be refreshed.
                    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
                        SAFE(mem_map.at(-exec_arg).reorder(ref_mem), WARN);
                    }
                }
                break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_dst(prb, mem, ref_mem), WARN);
                break;
            default: {
                const auto &binary_fill_cfg
                        = binary_po_fill_cfg(exec_arg, mem, prb->attr);
                std::unordered_map<int, fill_cfg_t> fill_cfg_map {
                        {DNNL_ARG_SRC_1, binary_fill_cfg}};
                SAFE(init_ref_memory_args_default_case(exec_arg, mem, ref_mem,
                             prb->attr, res, fill_cfg_map),
                        WARN);
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
    } else if (prb->dir & FLAG_BWD) {
        check_kinds = {SRC};
    } else {
        assert(!"unexpected!");
        SAFE_V(FAIL);
    }
    assert(!check_kinds.empty());
    return check_kinds;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(1);
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    if (has_bench_mode_bit(mode_bit_t::exec)) {
        SAFE(check_total_size(res), WARN);
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb, res), WARN);
    }
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = v_prim[0];

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(mem_map, prb, prim, supported_exec_args(prb->dir));
    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, prim, prb, res), WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(execute_and_wait(prim, args, res), WARN);

    check_correctness(
            prb, get_kinds_to_check(prb), args, ref_args, setup_cmp, res);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb), args, prb->attr,
                 prb->inplace, res),
            WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace resampling
