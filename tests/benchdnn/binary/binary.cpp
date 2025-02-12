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

#include <algorithm>
#include <random>

#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/fill.hpp"
#include "utils/parallel.hpp"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "eltwise/eltwise.hpp"

namespace binary {

int fill_mem(
        const prb_t *prb, int input_idx, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        // Some algorithms mandate positive filling.
        const std::vector<alg_t> alg_list {alg_t::DIV};
        const bool use_one_min_val = std::any_of(alg_list.begin(),
                alg_list.end(), [&](alg_t alg) { return alg == prb->alg; });
        const float range_min_val = use_one_min_val ? 1.f : -16.f;
        fill_cfg_t fill_cfg(mem_dt.dt(), range_min_val, 16.f, /* int = */ false,
                prb->alg, "binary");
        return fill_random_real(mem_dt, mem_fp, nullptr, fill_cfg);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    int min_val = MAX2(-8, static_cast<int>(lowest_dt(mem_dt.dt())));
    // Tenrary op supports a third input which can't be negative so far.
    if (input_idx == 2) min_val = 0;

    /* Do fixed partitioning to have same filling for any number of threads */
    static constexpr int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);
    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(idx_start + nelems * input_idx + 1);
        int_seed.discard(1);

        std::uniform_int_distribution<> gen(min_val, 8);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float val = gen(int_seed);
            // Make floating-point values only for src0 as src1 filling can be
            // used in other drivers and preferred to be integer.
            if (input_idx == 0) val *= 0.5f;
            // Remove zeroes in src1 to avoid division by zero.
            if (input_idx == 1 && val == 0.0f) val = 1.0f;
            val = round_to_nearest_representable(mem_dt.dt(), val);
            mem_fp.set_elem(idx, val);
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    auto src0_d = dnn_mem_t::init_md(prb->ndims, prb->vdims[0].data(),
            force_f32_dt ? dnnl_f32 : prb->sdt[0], prb->stag[0]);

    auto src1_d = dnn_mem_t::init_md(prb->ndims, prb->vdims[1].data(),
            force_f32_dt ? dnnl_f32 : prb->sdt[1], prb->stag[1]);

    auto dst_d = dnn_mem_t::init_md(prb->ndims, prb->dst_dims.data(),
            force_f32_dt ? dnnl_f32 : prb->ddt, prb->dtag);

    dnnl_alg_kind_t alg = attr_t::post_ops_t::kind2dnnl_kind(prb->alg);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, prb->dst_dims.data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    auto src2_d = prb->is_ternary_op() ? dnn_mem_t::init_md(prb->ndims,
                          prb->vdims[0].data(), dnnl_s8, prb->stag[0])
                                       : nullptr;

    TIME_C_PD(DNN_SAFE_STATUS(dnnl_binary_primitive_desc_create_v2(
            &init_pd_args.pd, init_pd_args.engine, alg,
            init_pd_args.src_md ? init_pd_args.src_md : src0_d, src1_d, src2_d,
            dst_d, dnnl_attr)));

    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    std::vector<dnnl_data_type_t> dts = {prb->sdt[0], prb->sdt[1], prb->ddt};
    skip_unimplemented_data_type(dts, prb->dir, res);
    skip_unimplemented_arg_scale(prb->attr, res);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_binary);

    if (is_gpu()) {
        // N.B: Adding this for gpu as cfg is not supported in POST-OPS
        bool have_post_ops = !prb->attr.post_ops.is_def();
        bool is_bf16u8 = (dts[0] == dnnl_bf16 && dts[1] == dnnl_bf16
                && dts[2] == dnnl_u8);
        if (is_bf16u8 && have_post_ops) {
            res->state = SKIPPED;
            res->reason = skip_reason::data_type_not_supported;
            return;
        }

        if (prb->is_ternary_op()) {
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }

        // gpu does not support s32
        for (const auto &dt : dts)
            if (dt == dnnl_s32) {
                res->state = SKIPPED;
                res->reason = skip_reason::data_type_not_supported;
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
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }

    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        if (is_sum) {
            res->state = SKIPPED;
            res->reason = skip_reason::invalid_case;
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

    static const std::vector<alg_t> cmp_alg = {
            alg_t::GE, alg_t::GT, alg_t::LE, alg_t::LT, alg_t::EQ, alg_t::NE};
    const bool is_cmp = std::any_of(
            cmp_alg.cbegin(), cmp_alg.cend(), [&](const alg_t alg) {
                return (prb->alg == alg) || prb->attr.post_ops.find(alg) >= 0;
            });

    if (is_cmp) cmp.set_zero_trust_percent(99.f);
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_args = {
            DNNL_ARG_SRC_0,
            DNNL_ARG_SRC_1,
            DNNL_ARG_SRC_2,
            DNNL_ARG_DST,
    };
    return exec_args;
};

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
            case DNNL_ARG_SRC_0:
                SAFE(fill_mem(prb, 0, mem, ref_mem), WARN);
                // Need a copy of source data for inplace mode for bitwise
                // testing.
                if (has_bench_mode_bit(mode_bit_t::bitwise) && prb->inplace) {
                    auto &src_copy = mem_map.at(-exec_arg);
                    SAFE(bool(src_copy) ? OK : FAIL, WARN);
                    SAFE(src_copy.reorder(mem), WARN);
                }
                break;
            case DNNL_ARG_SRC_1:
                SAFE(fill_mem(prb, 1, mem, ref_mem), WARN);
                break;
            case DNNL_ARG_SRC_2:
                SAFE(fill_mem(prb, 2, mem, ref_mem), WARN);
                break;
            case DNNL_ARG_DST:
                if (prb->attr.post_ops.find(alg_t::SUM) >= 0) {
                    SAFE(fill_mem(prb, 3, mem, ref_mem), WARN);

                    // Bitwise mode for sum requires a copy due to data for
                    // post-op will be overwritten and it must be refreshed.
                    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
                        SAFE(mem_map.at(-exec_arg).reorder(ref_mem), WARN);
                    }
                }
                break;
            default:
                SAFE(init_ref_memory_args_default_case(
                             exec_arg, mem, ref_mem, prb->attr, res),
                        WARN);
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

    check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
    SAFE(check_bitwise(prim, {DST}, args, prb->attr, prb->inplace, res), WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace binary
