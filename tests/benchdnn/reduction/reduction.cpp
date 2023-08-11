/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include <math.h>

#include <random>
#include <sstream>

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "reduction/reduction.hpp"

namespace reduction {

// first: nonneutral elements
// second: maximum range
using problem_bounds = std::pair<const int, const int>;

// acc | acc | elems | value_range | worst case
// s32 | mul |  10   |       3     |     3^10=2^16, out of 2^30 (max integer)
// f16 | mul |  10   |       1     | (2^1)^10=2^10, out of 2^16 (max exponent)
// f32 | mul |  30   |       3     | (2^3)^30=2^90, out of 2^128 (max exponent)
// s32 | sum | 10000 |      50     | 10000*50=2^19, out of 2^30 (max integer)
// f16 | sum | 1000  |       8     | 1000*8=2^13, out of 2^10 (max mantissa/integer)
// f32 | sum | 10000 |      16     | 10000*16=2^18, out of 2^23 (max mantissa/integer)
//  min/max  |  all  |    1000     | no limits on accumulation chain

// In f16 cases, the worst case exceeds the data type bounds, however it's rare
// to reach these extreme cases as long as they're close (can't just use f32 bounds)
const problem_bounds MUL_INT = problem_bounds(10, 3);
const problem_bounds MUL_F16 = problem_bounds(10, 1);
const problem_bounds MUL_F32 = problem_bounds(30, 3);
const problem_bounds SUM_INT = problem_bounds(10000, 50);
const problem_bounds SUM_F16 = problem_bounds(1000, 8);
const problem_bounds SUM_F32 = problem_bounds(10000, 16);
const problem_bounds MINMAX_INT = problem_bounds(-1, 1000);
const problem_bounds MINMAX_FP = problem_bounds(-1, 1000);

problem_bounds get_problem_bounds(alg_t alg, dnnl_data_type_t dt) {
    const bool is_int = is_integral_dt(dt);

    // Integer cases
    if (is_int) {
        switch (alg) {
            case alg_t::max:
            case alg_t::min: return MINMAX_INT;
            case alg_t::mul: return MUL_INT;
            // All remaining cases accumulate via sum
            default: return SUM_INT;
        }
    }

    // Floating-point cases
    const bool is_f16 = (dt == dnnl_f16);
    switch (alg) {
        case alg_t::max:
        case alg_t::min: return MINMAX_FP;
        case alg_t::mul: return is_f16 ? MUL_F16 : MUL_F32;
        // All remaining cases accumulate via sum
        default: return is_f16 ? SUM_F16 : SUM_F32;
    }
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;

    auto src_d = dnn_mem_t::init_md(
            prb->ndims, prb->vdims[0].data(), prb->sdt, prb->stag);
    auto dst_d = dnn_mem_t::init_md(
            prb->ndims, prb->vdims[1].data(), prb->ddt, prb->dtag);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, prb->vdims[1].data());
    const auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    TIME_C_PD(DNN_SAFE_STATUS(dnnl_reduction_primitive_desc_create(
            &init_pd_args.pd, init_pd_args.engine, alg2alg_kind(prb->alg),
            init_pd_args.src_md ? init_pd_args.src_md : src_d, dst_d, prb->p,
            prb->eps, dnnl_attr)));

    return dnnl_success;
}

bool is_norm_alg(const alg_t alg) {
    return alg == alg_t::norm_lp_max || alg == alg_t::norm_lp_sum
            || alg == alg_t::norm_lp_power_p_max
            || alg == alg_t::norm_lp_power_p_sum;
}

int fill_mem(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        float non_neutral_prob, bool expanded_range,
        bool only_positive_values) {
    const auto sdt = mem_dt.dt();
    const auto ddt = prb->ddt;
    const auto nelems = mem_fp.nelems();
    const float neutral_value = prb->alg == alg_t::mul ? 1.0f : 0.0f;
    // include ddt in is_signed to avoid mistrusted rounding negative -> 0
    const bool is_signed = sdt != dnnl_u8 && ddt != dnnl_u8;
    float shift = 0.0f;
    if (prb->alg == alg_t::mean || (prb->alg == alg_t::min && !is_signed))
        shift = 1.0f;
    const bool is_int = is_integral_dt(sdt);

    // Follow table in comments of fill_src
    int value_range = get_problem_bounds(prb->alg, sdt).second;

    const bool is_mul_fp = prb->alg == alg_t::mul && !is_int;
    const int min_range = is_mul_fp ? -value_range : 1;

    bool fill_with_powers_of_two = is_mul_fp;
    if (expanded_range) {
        // when using the expanded range, never fill with powers of 2
        fill_with_powers_of_two = false;
        value_range = 1000;
    }

    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        const int64_t idx_start = idx_chunk * chunk_size;
        const int64_t idx_end = MIN2(idx_start + chunk_size, nelems);

        std::minstd_rand msr(idx_start + 1);
        msr.discard(1);
        std::uniform_int_distribution<> igen(min_range, value_range);
        std::uniform_int_distribution<> fifty_fifty(0, 1);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value = neutral_value;
            if (flip_coin(idx, non_neutral_prob)) {
                const int gen = igen(msr);
                value = fill_with_powers_of_two ? std::pow(2, gen) : gen;

                if (!only_positive_values && is_signed && fifty_fifty(msr) == 1)
                    value = -value;
            }
            value += shift;
            mem_fp.set_elem(idx, round_to_nearest_representable(sdt, value));
        }
    });
    SAFE(mem_dt.reorder(mem_fp), WARN);
    return OK;
}

int fill_src(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    const auto sdt = prb->sdt;
    if (!nelems) return OK;

    int nelems_to_reduce = 1;
    for (int dim = 0; dim < prb->ndims; dim++) {
        if (prb->vdims[0][dim] != prb->vdims[1][dim]) {
            nelems_to_reduce *= prb->vdims[0][dim];
        }
    }

    // Determine number of non-neutral elements to have in the reduction chain
    int safe_to_reduce_elems = get_problem_bounds(prb->alg, sdt).first;
    if (safe_to_reduce_elems == -1) safe_to_reduce_elems = nelems_to_reduce;

    const float non_neutral_prob
            = 1.f * safe_to_reduce_elems / nelems_to_reduce;

    return fill_mem(prb, mem_dt, mem_fp, non_neutral_prob, false, false);
}

int fill_dst(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const bool only_positive_values = is_norm_alg(prb->alg);
    return fill_mem(prb, mem_dt, mem_fp, 1.0f, true, only_positive_values);
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->sdt, prb->ddt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_reduction, prb->sdt);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_reduction);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // Normalization algorithms don't make sense for integer data type.
    // They also can't have `p` parameter less than one.
    const bool is_invalid = is_norm_alg(prb->alg)
            && (is_integral_dt(prb->sdt) || prb->p < 1.f);

    if (is_invalid) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    // accounts for inaccurate rootn/pow functions in norm algs.
    float scale = is_norm_alg(prb->alg) ? 5.0f : 1.0f;
    cmp.set_threshold(scale * epsilon_dt(prb->ddt));

    if (is_amd_gpu()) {
        // MIOpen implementation is less accurate for f16 data type therefore
        // adjust the threshold.
        if (prb->sdt == dnnl_f16 || prb->ddt == dnnl_f16)
            cmp.set_threshold(1.5e-4f * 4.f);
    }
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DST,
    };
    return exec_args;
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
            case DNNL_ARG_DST:
                if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM)
                        >= 0)
                    SAFE(fill_dst(prb, mem, ref_mem), WARN);
                break;
            case DNNL_ARG_SCRATCHPAD: break;
            default: { // Process all attributes here
                int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
                        - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
                bool is_post_ops_arg = (exec_arg & post_ops_range);
                if (is_post_ops_arg) {
                    if (exec_arg & DNNL_ARG_SRC_1) {
                        const bool is_signed
                                = prb->sdt != dnnl_u8 && prb->ddt != dnnl_u8;
                        const bool use_positive
                                = is_norm_alg(prb->alg) || !is_signed;
                        SAFE(binary::fill_mem(
                                     exec_arg, mem, ref_mem, use_positive),
                                WARN);
                    }
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

} // namespace reduction
