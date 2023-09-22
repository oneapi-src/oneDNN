/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <random>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "sum/sum.hpp"

namespace sum {

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;

    std::vector<benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t>> src_d_wrappers(
            prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input)
        src_d_wrappers[i_input] = dnn_mem_t::init_md(prb->ndims,
                prb->dims.data(), prb->sdt[i_input], prb->stag[i_input]);

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_d {};
    if (prb->dtag != tag::undef) {
        dst_d = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->ddt, prb->dtag);
    }

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    std::vector<dnnl_memory_desc_t> src_d(
            src_d_wrappers.begin(), src_d_wrappers.end());
    init_pd_args.is_iterator_supported = false;
    TIME_C_PD(DNN_SAFE_STATUS(dnnl_sum_primitive_desc_create(&init_pd_args.pd,
            init_pd_args.engine, dst_d, prb->n_inputs(),
            prb->input_scales.data(), src_d.data(), dnnl_attr)));

    return dnnl_success;
}

int fill_src(int input_idx, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {

    const auto nelems = mem_dt.nelems();
    const auto dt = mem_dt.dt();
    const int range = 16;
    const int f_min = dt == dnnl_u8 ? 0 : -range / 2;

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) - 17 * input_idx + 101) % range;
        const float value = (dt == dnnl_bf16 || dt == dnnl_f16)
                ? (f_min + gen) / range
                : (f_min + gen) * (1.0f + 4.0f / range);
        mem_fp.set_elem(i, round_to_nearest_representable(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    std::vector<dnnl_data_type_t> dts = prb->sdt;
    dts.push_back(prb->ddt);
    skip_unimplemented_data_type(dts, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_sum, prb->sdt[0]);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_sum);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        skip_invalid_inplace(
                res, prb->sdt[0], prb->ddt, prb->stag[0], prb->dtag);
        if (res->state == SKIPPED) return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    cmp.set_threshold(epsilon_dt(prb->ddt) * prb->n_inputs());
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_args = {
            DNNL_ARG_MULTIPLE_SRC,
            DNNL_ARG_DST,
    };
    return exec_args;
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref) {
    update_inplace_memory_args(mem_map, prb, dir);
    if (has_bench_mode_modifier(mode_modifier_t::no_host_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        ref_mem_map.emplace(
                exec_arg, dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SCRATCHPAD: break;
            default:
                bool is_src_arg = (exec_arg & DNNL_ARG_MULTIPLE_SRC);
                if (is_src_arg) {
                    SAFE(fill_src(exec_arg, mem, ref_mem), WARN);
                }
                break;
        }
        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(1);
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    return OK;
}

int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    return check_caches(v_prim[0], prb, res);
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = v_prim[0];

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(mem_map, prb, prim, supported_exec_args(prb->dir));
    TIME_FILL(SAFE(init_ref_memory_args(
                           ref_mem_map, mem_map, prim, prb, res, prb->dir),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (has_bench_mode_bit(mode_bit_t::corr)) {
        check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace sum
