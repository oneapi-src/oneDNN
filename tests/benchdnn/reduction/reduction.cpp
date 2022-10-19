/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    auto src_desc = dnn_mem_t::init_md(
            prb->ndims, prb->vdims[0].data(), prb->sdt, prb->stag);
    auto dst_desc = dnn_mem_t::init_md(
            prb->ndims, prb->vdims[1].data(), prb->ddt, prb->dtag);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, prb->vdims[1].data());
    const auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    DNN_SAFE_STATUS(dnnl_reduction_primitive_desc_create(&init_pd_args.pd,
            init_pd_args.engine, alg2alg_kind(prb->alg), src_desc, dst_desc,
            prb->p, prb->eps, dnnl_attr));

    return dnnl_success;
}

bool is_norm_alg(const alg_t alg) {
    return alg == alg_t::norm_lp_max || alg == alg_t::norm_lp_sum
            || alg == alg_t::norm_lp_power_p_max
            || alg == alg_t::norm_lp_power_p_sum;
}

int fill_mem(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        float non_neutral_prob, bool use_reduced_range,
        bool only_positive_values) {
    const auto sdt = mem_dt.dt();
    const auto nelems = mem_fp.nelems();
    const float neutral_value = prb->alg == alg_t::mul ? 1.0f : 0.0f;
    const float mean_shift = prb->alg == alg_t::mean ? 1.0f : 0.0f;
    const bool is_signed = sdt != dnnl_u8;
    const bool is_int = is_integral_dt(sdt);

    int value_range = use_reduced_range ? 16 : 1000;
    if (is_int) value_range = use_reduced_range ? 3 : max_dt(dnnl_s8);

    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        const int64_t idx_start = idx_chunk * chunk_size;
        const int64_t idx_end = MIN2(idx_start + chunk_size, nelems);

        std::minstd_rand msr(idx_start + 1);
        msr.discard(1);
        std::uniform_int_distribution<> igen(1, value_range);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value = neutral_value;
            if (flip_coin(idx, non_neutral_prob)) {
                const int gen = igen(msr);
                value = is_int ? gen : gen / 8.f;
                if (!only_positive_values && is_signed && flip_coin(gen, 0.5f))
                    value = -value;
            }
            value += mean_shift;
            mem_fp.set_elem(idx, round_to_nearest_representable(sdt, value));
        }
    });
    SAFE(mem_dt.reorder(mem_fp), WARN);
    return OK;
}

int fill_src(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    const auto ddt = prb->ddt;
    if (!nelems) return OK;

    int nelems_to_reduce = 1;
    for (int dim = 0; dim < prb->ndims; dim++) {
        if (prb->vdims[0][dim] != prb->vdims[1][dim]) {
            nelems_to_reduce *= prb->vdims[0][dim];
        }
    }
    // There is no accumulation error in case of min or max algorithm
    const bool is_min_or_max = prb->alg == alg_t::min || prb->alg == alg_t::max;
    // Number of elements that should not exceed datatype limit after reduction
    int safe_to_reduce_elems = nelems_to_reduce;
    if (!is_min_or_max) { // Other algs do computations, reduce final values
        safe_to_reduce_elems = prb->alg == alg_t::mul ? 10 : 1000;
        // Integral values easily reach border values,
        // shrink their final values more
        if (is_integral_dt(ddt))
            safe_to_reduce_elems = prb->alg == alg_t::mul ? 3 : 10;
    }
    const float non_neutral_prob
            = 1.f * safe_to_reduce_elems / nelems_to_reduce;

    return fill_mem(
            prb, mem_dt, mem_fp, non_neutral_prob, !is_min_or_max, false);
}

int fill_dst(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const bool only_positive_values = is_norm_alg(prb->alg);
    return fill_mem(prb, mem_dt, mem_fp, 1.0f, false, only_positive_values);
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->sdt, prb->ddt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);
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
    // `5` is a temporary magic const for GPU to pass norm algs.
    // TODO: consider change the filling with power-of-two values for better
    // answer precision.
    cmp.set_threshold(5 * epsilon_dt(prb->ddt));
    if (is_amd_gpu()) {
        // MIOpen implementation is less accurate for f16 data type therefore
        // adjust the threshold.
        if (prb->sdt == dnnl_f16 || prb->ddt == dnnl_f16)
            cmp.set_threshold(1.5e-4 * 4);
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);

    const auto fp_dt = dnnl_f32;
    const auto abx_tag = tag::abx;

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    const auto &src_md = query_md(const_pd, DNNL_ARG_SRC);
    dnn_mem_t src_fp(src_md, fp_dt, abx_tag, ref_engine);
    dnn_mem_t src_dt(src_md, test_engine);
    SAFE(fill_src(prb, src_dt, src_fp), WARN);

    const auto &dst_md = query_md(const_pd, DNNL_ARG_DST);
    dnn_mem_t dst_fp(dst_md, fp_dt, abx_tag, ref_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0)
        SAFE(fill_dst(prb, dst_dt, dst_fp), WARN);

    const bool binary_po_only_positive_vals = is_norm_alg(prb->alg);
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(const_pd, binary_po_args, binary_po_dt,
                 binary_po_fp, binary_po_only_positive_vals),
            WARN);

    args_t args, ref_args;

    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(binary_po_args, binary_po_dt);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (is_bench_mode(CORR)) {
        ref_args.set(DNNL_ARG_SRC, src_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);
        ref_args.set(binary_po_args, binary_po_fp);

        check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace reduction