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
        bool only_positive_values, bool only_integer_values) {
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

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;

    auto src0_d = dnn_mem_t::init_md(
            prb->ndims, prb->vdims[0].data(), prb->sdt[0], prb->stag[0]);

    auto src1_d = dnn_mem_t::init_md(
            prb->ndims, prb->vdims[1].data(), prb->sdt[1], prb->stag[1]);

    auto dst_d = dnn_mem_t::init_md(
            prb->ndims, prb->dst_dims.data(), prb->ddt, prb->dtag);

    dnnl_alg_kind_t alg = attr_t::post_ops_t::kind2dnnl_kind(prb->alg);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, prb->dst_dims.data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    TIME_C_PD(DNN_SAFE_STATUS(dnnl_binary_primitive_desc_create(
            &init_pd_args.pd, init_pd_args.engine, alg,
            init_pd_args.src_md ? init_pd_args.src_md : src0_d, src1_d, dst_d,
            dnnl_attr)));

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
            res->state = SKIPPED, res->reason = DATA_TYPE_NOT_SUPPORTED;
            return;
        }

        // gpu does not support s32
        for (const auto &dt : dts)
            if (dt == dnnl_s32) {
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
            case DNNL_ARG_SRC_0: SAFE(fill_mem(0, mem, ref_mem), WARN); break;
            case DNNL_ARG_SRC_1: SAFE(fill_mem(1, mem, ref_mem), WARN); break;
            case DNNL_ARG_DST:
                if (prb->attr.post_ops.find(alg_t::SUM) >= 0) {
                    SAFE(fill_mem(2, mem, ref_mem), WARN);
                }
                break;
            case DNNL_ARG_SCRATCHPAD: break;
            default: { // Process all attributes here
                int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
                        - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
                bool is_post_ops_arg = (exec_arg & post_ops_range);
                bool is_scales_arg = (exec_arg & DNNL_ARG_ATTR_SCALES);
                if (is_post_ops_arg) {
                    SAFE(binary::fill_mem(exec_arg, mem, ref_mem), WARN);
                } else if (is_scales_arg) {
                    int exec_src_arg = exec_arg ^ DNNL_ARG_ATTR_SCALES;
                    // Leave hard coded until supported mask is 0 only.
                    ref_mem.set_elem(
                            0, prb->attr.scales.get(exec_src_arg).scale);
                    SAFE(mem.reorder(ref_mem), WARN);
                }
            } break;
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

} // namespace binary
