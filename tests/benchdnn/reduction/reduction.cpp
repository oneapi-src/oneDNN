/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "tests/test_thread.hpp"

#include "compare.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "reduction/reduction.hpp"

namespace reduction {

int init_pd(dnnl_engine_t engine, const prb_t *prb, dnnl_primitive_desc_t &rpd,
        res_t *res, dir_t dir, const_dnnl_primitive_desc_t hint) {
    dnnl_reduction_desc_t rd;
    dnnl_memory_desc_t src_desc, dst_desc;

    SAFE(init_md(&src_desc, prb->ndims, prb->src_dims.data(), prb->sdt,
                 prb->stag),
            WARN);

    SAFE(init_md(&dst_desc, prb->ndims, prb->dst_dims.data(), prb->ddt,
                 prb->dtag),
            WARN);

    DNN_SAFE(dnnl_reduction_desc_init(&rd, alg2alg_kind(prb->alg), &src_desc,
                     &dst_desc, prb->p, prb->eps),
            WARN);

    attr_args_t attr_args;
    attr_args.prepare_binary_post_op_mds(
            prb->attr, prb->ndims, prb->dst_dims.data());
    const auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&rpd, &rd, dnnl_attr, engine, nullptr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(rpd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    return OK;
}

int fill_mem(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    const auto sdt = mem_dt.dt();
    const auto ddt = prb->ddt;
    if (!nelems) return OK;

    const int range = prb->alg == alg_t::MUL
            ? (sdt == dnnl_u8 || sdt == dnnl_s8) ? 1024 : 4
            : 16;
    const int f_min = sdt == dnnl_u8 ? 1 : -range / 2;

    int nelems_to_reduce = 1;
    for (int dim = 0; dim < prb->ndims; dim++) {
        if (prb->src_dims.at(dim) != prb->dst_dims.at(dim)) {
            nelems_to_reduce *= prb->src_dims.at(dim);
        }
    }
    // Fill only some number of elements with values other than neutral value
    // to avoid precision problems caused by different orders of accumulation
    // between some kernels and benchdnn reference. Number of elements was
    // selected experimentally.
    int safe_to_reduce_elems = nelems_to_reduce;
    if (prb->alg == alg_t::NORM_LP_MAX || prb->alg == alg_t::NORM_LP_POWER_P_MAX
            || prb->alg == alg_t::NORM_LP_POWER_P_SUM
            || prb->alg == alg_t::NORM_LP_SUM) {
        safe_to_reduce_elems = 5e3;
    } else if (ddt == dnnl_f32 || ddt == dnnl_f16 || ddt == dnnl_bf16) {
        // It should work if float is used as an accumulator for f16 and bf16
        safe_to_reduce_elems = 1e5;
    }
    const int64_t non_neutral_elems
            = nelems / nelems_to_reduce * safe_to_reduce_elems;
    // No special meaning; just to increase the precision of neutral_gen
    const int prob_range = 1000;
    const float non_neutral_prob = non_neutral_elems / nelems;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        std::minstd_rand msr(i);
        msr.discard(1);
        std::uniform_int_distribution<> igen(0, range);
        const float gen = static_cast<float>(igen(msr));
        const float neutral_value = prb->alg == alg_t::MUL ? 1.0f : 0.0f;
        float value = neutral_value;

        const float neutral_gen = ((89 * i) + 73) % prob_range;
        if (neutral_gen <= non_neutral_prob * prob_range) {
            if (prb->alg == alg_t::MUL) {
                if (sdt == dnnl_s8 || sdt == dnnl_u8) {
                    // generate {1, 2}, but probability of 2 is 1/range
                    value = gen == range ? 2.0f : 1.0f;
                } else {
                    // generate {-2, -0.5, 1, 0.5, 2} to avoid underflow/overflow
                    value = powf(f_min + gen, 2.0f) / 2;
                    if (f_min + gen != 0.0f) {
                        const float sign = fabs(f_min + gen) / (f_min + gen);
                        value *= sign;
                    } else {
                        value = 1.0f;
                    }
                }
            } else if (prb->alg == alg_t::MEAN && ddt == dnnl_f16) {
                // Shift the mean to value different than 0 as results equal
                // to 0 may be treated as mistrusted.
                float mean_shift = 0.5;
                value = (f_min + gen) / range + mean_shift;
            } else {
                value = (sdt == dnnl_bf16 || sdt == dnnl_f16)
                        ? (f_min + gen) / range
                        : (f_min + gen) * (1.0f + 4.0f / range);
            }
        }
        mem_fp.set_elem(i, round_to_nearest_representable(sdt, value));
    });
    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common({prb->sdt, prb->ddt}, FWD_D, res);
    if (res->state == SKIPPED) return;

    bool is_invalid = false;
    switch (prb->alg) {
        case alg_t::NORM_LP_MAX:
        case alg_t::NORM_LP_SUM:
        case alg_t::NORM_LP_POWER_P_MAX:
        case alg_t::NORM_LP_POWER_P_SUM:
            is_invalid = is_integral_dt(prb->sdt) || prb->p < 1.f;
            break;
        default: break;
    }
    if (is_invalid) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }

    if (is_nvidia_gpu()) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(prim, &const_pd), CRIT);

    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto fp_dt = dnnl_f32;
    const auto abx_tag = tag::abx;

    const auto &test_engine = get_test_engine();

    const auto &src_md = q(DNNL_ARG_SRC);
    dnn_mem_t src_fp(src_md, fp_dt, abx_tag, test_engine);
    dnn_mem_t src_dt(src_md, test_engine);
    SAFE(fill_mem(prb, src_dt, src_fp), WARN);

    const auto &dst_md = q(DNNL_ARG_DST);
    dnn_mem_t dst_fp(dst_md, fp_dt, abx_tag, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0)
        SAFE(fill_mem(prb, dst_dt, dst_fp), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(
                 const_pd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    args_t args;
    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(binary_po_args, binary_po_dt);

    SAFE(execute_and_wait(prim, args), WARN);

    if (bench_mode & CORR) {
        compute_ref(prb, src_fp, binary_po_fp, dst_fp);
        compare::compare_t cmp;
        // `5` is a temporary magic const for GPU to pass norm algs.
        // TODO: consider change the filling with power-of-two values for better
        // answer precision.
        cmp.set_threshold(5 * epsilon_dt(prb->ddt));
        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }

    return measure_perf(res->timer, prim, args);
}

} // namespace reduction
