/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include <random>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "concat/concat.hpp"

namespace concat {

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    std::vector<benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t>> src_d_wrappers(
            prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const dims_t &i_vdims = prb->vdims[i_input];
        src_d_wrappers[i_input] = dnn_mem_t::init_md(prb->ndims, i_vdims.data(),
                force_f32_dt ? dnnl_f32 : prb->sdt, prb->stag[i_input]);
    }

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_d {};
    if (prb->dtag != tag::undef) {
        dst_d = dnn_mem_t::init_md(prb->ndims, prb->dst_dims.data(),
                force_f32_dt ? dnnl_f32 : prb->ddt, prb->dtag);
    }

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    std::vector<dnnl_memory_desc_t> src_d(
            src_d_wrappers.begin(), src_d_wrappers.end());
    init_pd_args.is_iterator_supported = false;
    TIME_C_PD(DNN_SAFE_STATUS(dnnl_concat_primitive_desc_create(
            &init_pd_args.pd, init_pd_args.engine, dst_d, prb->n_inputs(),
            prb->axis, src_d.data(), dnnl_attr)));

    return dnnl_success;
}

int fill_src(int input_idx, dnnl_data_type_t dt, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }

    // Do fixed partitioning to have same filling for any number of threads.
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);
    // Set proper range of valid values to avoid any reorders back and forth.
    const bool s8u8_or_u8s8 = (dt == dnnl_s8 && mem_dt.dt() == dnnl_u8)
            || (dt == dnnl_u8 && mem_dt.dt() == dnnl_s8);
    float min_val = lowest_dt(dnnl_s8);
    float max_val = max_dt(dnnl_u8);
    if (s8u8_or_u8s8) {
        min_val = lowest_dt(dnnl_u8);
        max_val = max_dt(dnnl_s8);
    } else if (dt == dnnl_s8 || mem_dt.dt() == dnnl_s8) {
        max_val = max_dt(dnnl_s8);
    } else if (dt == dnnl_u8 || mem_dt.dt() == dnnl_u8) {
        min_val = lowest_dt(dnnl_u8);
    }

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // See eltwise.cpp for implementation details.
        std::minstd_rand msr(input_idx * n_chunks + idx_start + 1);
        msr.discard(1);
        std::uniform_int_distribution<> igen(min_val, max_val);
        // No need to round final value as it's already in needed dt.
        for (int64_t idx = idx_start; idx < idx_end; ++idx)
            mem_fp.set_elem(idx, (float)igen(msr));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->sdt, prb->ddt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_concat, prb->sdt);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_concat);
    skip_unimplemented_arg_scale(prb->attr, res);

    // ref concat is reorder-based, hence, inherits some reorder limitations.
    // bf16, f16 reorders on cpu supports only [bf16, f16]<->f32
    bool valid_xf16_input
            = IMPLICATION(prb->sdt == dnnl_bf16 || prb->sdt == dnnl_f16,
                    prb->dtag == tag::undef || prb->ddt == dnnl_f32
                            || prb->ddt == prb->sdt);
    bool valid_xf16_output
            = IMPLICATION((prb->ddt == dnnl_bf16 || prb->ddt == dnnl_f16)
                            && prb->dtag != tag::undef,
                    (prb->sdt == dnnl_f32 || prb->sdt == prb->ddt));

    if (is_cpu() && (!valid_xf16_input || !valid_xf16_output)) {
        res->state = SKIPPED;
        res->reason = skip_reason::case_not_supported;
        return;
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_args = {
            DNNL_ARG_MULTIPLE_SRC,
            DNNL_ARG_DST,
    };
    return exec_args;
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    if (has_bench_mode_modifier(mode_modifier_t::no_host_memory)) return OK;

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

        // Scales attach to same (multiple_src + i) arguments. They should be
        // filtered out and re-directed to a common call.
        if ((exec_arg & DNNL_ARG_MULTIPLE_SRC)
                && !(exec_arg & DNNL_ARG_ATTR_SCALES)) {
            SAFE(fill_src(exec_arg, prb->ddt, mem, ref_mem), WARN);
        } else {
            SAFE(init_ref_memory_args_default_case(
                         exec_arg, mem, ref_mem, prb->attr, res),
                    WARN);
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
    // Assume it doesn't change through the execution.
    static int capacity = 0;
    static auto st = dnnl_get_primitive_cache_capacity(&capacity);
    if (st != dnnl_success) return FAIL;
    if (capacity > 0 && prb->n_inputs() + 1 > capacity) {
        BENCHDNN_PRINT(2, "%s\n",
                "[INFO] The number of potential internal reorder pds plus "
                "concat itself exceeds the cache capacity which will lead to a "
                "test case false-positive failure.");
        return OK;
    }
    return check_caches(v_prim[0], prb, res);
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

    check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
    SAFE(check_bitwise(prim, {DST}, args, prb->attr, prb->inplace, res), WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace concat
