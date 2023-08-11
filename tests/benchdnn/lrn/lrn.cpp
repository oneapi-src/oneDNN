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
    // LRN in MIOpen 2.17 and older doesn't support negative input. The support
    // was added in https://github.com/ROCmSoftwarePlatform/MIOpen/pull/1562.
    // The plan is to use only positive input at this point but bump the
    // minimum required MIOpen version to 2.18 once it's released and enable
    // negative input back.
    const int f_min
            = prb->dt == dnnl_u8 ? 0 : (is_amd_gpu() ? range : -range) / 2;

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
    res_t *res = init_pd_args.res;

    dnnl_dims_t data_dims_0d = {prb->mb, prb->ic};
    dnnl_dims_t data_dims_1d = {prb->mb, prb->ic, prb->iw};
    dnnl_dims_t data_dims_2d = {prb->mb, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t data_dims_3d = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};

    dnnl_dim_t *data_dims = prb->ndims == 5 ? data_dims_3d
            : prb->ndims == 4               ? data_dims_2d
            : prb->ndims == 3               ? data_dims_1d
                                            : data_dims_0d;

    auto src_d = dnn_mem_t::init_md(prb->ndims, data_dims, prb->dt, prb->tag);
    auto dst_d = dnn_mem_t::init_md(prb->ndims, data_dims, prb->dt, tag::any);

    dnnl_alg_kind_t alg = alg2alg_kind(prb->alg);

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    if (dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        TIME_C_PD(DNN_SAFE_STATUS(dnnl_lrn_forward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, prop, alg,
                init_pd_args.src_md ? init_pd_args.src_md : src_d, dst_d,
                prb->ls, prb->alpha, prb->beta, prb->k, dnnl_attr)));
    } else {
        auto diff_src_d
                = dnn_mem_t::init_md(prb->ndims, data_dims, prb->dt, tag::any);
        auto diff_dst_d
                = dnn_mem_t::init_md(prb->ndims, data_dims, prb->dt, tag::any);
        TIME_C_PD(DNN_SAFE_STATUS(dnnl_lrn_backward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, alg, diff_src_d,
                diff_dst_d, src_d, prb->ls, prb->alpha, prb->beta, prb->k,
                init_pd_args.hint, dnnl_attr)));
    }

    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->dt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_lrn, prb->dt);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_lrn);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    // `3` is a const needed to adjust division error
    cmp.set_threshold(compute_n_summands(prb) * 3 * epsilon_dt(prb->dt));
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DST,
            DNNL_ARG_WORKSPACE,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_WORKSPACE,
    };
    return (dir & FLAG_FWD) ? exec_fwd_args : exec_bwd_args;
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref) {
    if (has_bench_mode_modifier(mode_modifier_t::no_host_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        ref_mem_map.emplace(
                exec_arg, dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC: SAFE(fill_src(prb, mem, ref_mem), WARN); break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_dst(prb, mem, ref_mem), WARN);
                break;
            default: break;
        }
        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(2); // just fwd or fwd + bwd.
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res, FLAG_FWD,
                 nullptr, /* is_service_prim = */ prb->dir & FLAG_BWD),
            WARN);
    if (prb->dir & FLAG_BWD) {
        SAFE(init_prim(prb->ctx_init, v_prim[1], init_pd, prb, res, FLAG_BWD,
                     query_pd(v_prim[0])),
                WARN);
    }
    return OK;
}

int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    SAFE(check_caches(v_prim[0], prb, res), WARN);
    if (v_prim[1]) { SAFE(check_caches(v_prim[1], prb, res), WARN); }
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = prb->dir & FLAG_FWD ? v_prim[0] : v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(
            mem_map, prb, v_prim[0], supported_exec_args(FLAG_FWD));
    TIME_FILL(SAFE(init_ref_memory_args(
                           ref_mem_map, mem_map, v_prim[0], prb, res, FLAG_FWD),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(execute_and_wait(v_prim[0], args, res), WARN);

    if (prb->dir & FLAG_FWD) {
        if (has_bench_mode_bit(mode_bit_t::corr)) {
            check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
        }
    }

    if (prb->dir & FLAG_BWD) {
        // Pass same memory map as we need data from forward on backward.
        init_memory_args<prb_t>(
                mem_map, prb, v_prim[1], supported_exec_args(FLAG_BWD));
        TIME_FILL(SAFE(init_ref_memory_args(ref_mem_map, mem_map, v_prim[1],
                               prb, res, FLAG_BWD),
                WARN));

        args = args_t(mem_map);
        ref_args = args_t(ref_mem_map);

        SAFE(execute_and_wait(v_prim[1], args, res), WARN);

        if (has_bench_mode_bit(mode_bit_t::corr)) {
            check_correctness(prb, {SRC}, args, ref_args, setup_cmp, res);
        }
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace lrn
