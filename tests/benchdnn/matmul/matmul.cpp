/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "matmul/matmul.hpp"

namespace matmul {

void prep_bia_dims(const prb_t *prb, dims_t &bia_dims) {
    bia_dims.resize(prb->ndims);
    for (int d = 0; d < prb->ndims; ++d)
        bia_dims[d] = (prb->bia_mask & (1 << d)) ? prb->dst_dims[d] : 1;
}

dims_t get_runtime_dims(const dims_t &dims, const dims_mask_t &mask) {
    if (mask.none() || dims.empty()) return dims;
    dims_t runtime_dims;
    runtime_dims.resize(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        runtime_dims[i] = mask[i] ? DNNL_RUNTIME_DIM_VAL : dims[i];
    }
    return runtime_dims;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    const auto &src_rt_dims
            = get_runtime_dims(prb->src_dims(), prb->src_runtime_dim_mask());
    const auto &weights_rt_dims = get_runtime_dims(
            prb->weights_dims(), prb->weights_runtime_dim_mask());
    const auto &dst_rt_dims
            = get_runtime_dims(prb->dst_dims, prb->dst_runtime_dim_mask());

    auto src_d = dnn_mem_t::init_md(prb->ndims, src_rt_dims.data(),
            prb->src_dt(), prb->stag, prb->strides[STRIDES_SRC]);
    auto wei_d = dnn_mem_t::init_md(prb->ndims, weights_rt_dims.data(),
            prb->wei_dt(), prb->wtag, prb->strides[STRIDES_WEI]);
    auto dst_d = dnn_mem_t::init_md(prb->ndims, dst_rt_dims.data(),
            prb->dst_dt(), prb->dtag, prb->strides[STRIDES_DST]);

    dnnl_memory_desc_t bia_d {};
    if (prb->bia_dt != dnnl_data_type_undef) {
        dims_t bia_dims;
        prep_bia_dims(prb, bia_dims);
        bia_dims = get_runtime_dims(bia_dims, prb->dst_runtime_dim_mask());
        bia_d = dnn_mem_t::init_md(prb->ndims, bia_dims.data(), prb->bia_dt,
                prb->dst_runtime_dim_mask() != 0 ? tag::abx : tag::any);
    }

    dnnl_matmul_desc_t op_d;
    DNN_SAFE_STATUS(
            dnnl_matmul_desc_init(&op_d, &src_d, &wei_d, &bia_d, &dst_d));

    // Overload PER_OC mask definition for batched case
    int mask = 0;
    if (prb->attr.oscale.policy == policy_t::PER_OC)
        mask = (1 << (dst_rt_dims.size() - 1));

    attr_args_t attr_args;
    attr_args.prepare_output_scales(prb->attr, prb->scales, prb->n, mask);
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, prb->dst_dims.data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    return dnnl_primitive_desc_iterator_create(&init_pd_args.pd_it, &op_d,
            dnnl_attr, init_pd_args.engine, init_pd_args.hint);
}

int init_prim_ref(
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim_ref, const prb_t *prb) {
    if (!(is_bench_mode(CORR) && is_gpu() && fast_ref_gpu)) return OK;

    // Create a new copy of prb to avoid potentially corrupting the test by
    // modifying prb in place.
    const auto cpu_bia_dt = prb->bia_dt == dnnl_data_type_undef
            ? dnnl_data_type_undef
            : dnnl_f32;
    const auto cpu_bia_mask
            = prb->bia_dt == dnnl_data_type_undef ? 0 : prb->bia_mask;
    auto cpu_attr = prb->attr;
    update_cpu_ref_attrs(cpu_attr);
    prb_t prb_cpu {*prb, {dnnl_f32}, tag::abx, tag::abx, tag::abx,
            {vdims_t(STRIDES_SIZE)}, cpu_bia_dt, cpu_bia_mask, {0, 0, 0},
            cpu_attr};

    init_pd_args_t<prb_t> init_pd_args(
            /* res = */ nullptr, get_cpu_engine(), &prb_cpu, prb->dir,
            /* hint = */ nullptr);
    init_pd(init_pd_args);
    auto pd_ref = make_benchdnn_dnnl_wrapper(init_pd_args.pd);

    dnnl_primitive_t prim_ref_ {};
    if (pd_ref) {
        if (query_impl_info(pd_ref) == "ref:any") return OK;
        DNN_SAFE(dnnl_primitive_create(&prim_ref_, pd_ref), WARN);
        BENCHDNN_PRINT(5, "CPU reference oneDNN implementation: %s\n",
                query_impl_info(pd_ref).c_str());
    }
    prim_ref.reset(prim_ref_);
    return OK;
}

int fill_data(data_kind_t kind, const prb_t *prb, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {

    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    cfg_t cfg(prb, {SRC, WEI, BIA, DST});
    cfg_t::density_args_t density_args;
    density_args.data_kind = kind;
    density_args.n_acc = prb->k;
    const auto density = cfg.get_density(density_args);

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(kind * nelems + idx_start + 1);
        int_seed.discard(1);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::uniform_int_distribution<> gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));
        std::bernoulli_distribution b_dist(density);

        // make sure the first element is positive
        if (idx_start == 0) {
            float val = 0;
            while (val <= 0)
                val = gen(int_seed);
            mem_fp.set_elem(
                    0, round_to_nearest_representable(cfg.get_dt(kind), val));
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            bool is_one = density == 1.f ? true : b_dist(b_seed);
            float val = is_one * gen(int_seed);
            mem_fp.set_elem(
                    idx, round_to_nearest_representable(cfg.get_dt(kind), val));
        }
    });

    const bool swap_dt
            = kind == DST && cfg.get_orig_dt(kind) != cfg.get_dt(kind);
    if (swap_dt) mem_dt.set_dt(cfg.get_dt(kind));
    SAFE(mem_dt.reorder(mem_fp), WARN);
    if (swap_dt) mem_dt.set_dt(cfg.get_orig_dt(kind));

    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type(
            {prb->src_dt(), prb->wei_dt(), prb->bia_dt, prb->dst_dt()},
            prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, prb->dst_dt());

    if (is_gpu()) {
        // GPU supports only single zero-point per tensor.
        if (prb->attr.zero_points.get(DNNL_ARG_SRC).policy != policy_t::COMMON
                || prb->attr.zero_points.get(DNNL_ARG_DST).policy
                        != policy_t::COMMON) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }

        // GPU supports only default sum_dt argument.
        const auto &po = prb->attr.post_ops;
        const int sum_idx = po.find(attr_t::post_ops_t::kind_t::SUM);
        if (sum_idx != -1 && po.entry[sum_idx].sum.dt != dnnl_data_type_undef) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }

        // GPU for x8s8bf16 doesn't support:
        // * Destination zero-point.
        // * Any run-time dimensions.
        // * Any batch dimensions.
        const bool is_x8s8bf16
                = prb->wei_dt() == dnnl_s8 && prb->dst_dt() == dnnl_bf16;
        const bool rt_dims_are_none = prb->src_runtime_dim_mask().none()
                && prb->weights_runtime_dim_mask().none()
                && prb->dst_runtime_dim_mask().none();
        const bool x8s8bf16_ok = IMPLICATION(is_x8s8bf16,
                prb->attr.zero_points.get(DNNL_ARG_DST).is_def()
                        && rt_dims_are_none && prb->ndims <= 2);
        if (!x8s8bf16_ok) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }

        // GPU supports bf16 bias only for bf16 config, with a single batch dim.
        const bool is_bf16 = prb->src_dt() == dnnl_bf16
                && prb->wei_dt() == dnnl_bf16
                && (prb->dst_dt() == dnnl_bf16 || prb->dst_dt() == dnnl_f32);
        const bool bf16_bias_ok = IMPLICATION(
                prb->bia_dt == dnnl_bf16, prb->ndims <= 2 + is_bf16);
        if (!bf16_bias_ok) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // Zero-points for non-integral data type does not make sense
    if (!prb->attr.zero_points.is_def() && prb->wei_dt() != dnnl_s8) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }

    auto src_rt_mask = prb->src_runtime_dim_mask();
    auto wei_rt_mask = prb->weights_runtime_dim_mask();
    auto dst_rt_mask = prb->dst_runtime_dim_mask();

    // Memory layouts must be defined when some dimensions are unknown at pd
    // creation time.
    if ((src_rt_mask.any() && prb->stag == "any")
            || (wei_rt_mask.any() && prb->wtag == "any")
            || (dst_rt_mask.any() && prb->dtag == "any")) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }

    // Runtime masks for `m`, `k`, and `n` dimensions must be consistent.
    const int m_idx = prb->ndims - 2;
    const int k_idx_src = prb->ndims - 1;
    const int k_idx_wei = prb->ndims - 2;
    const int n_idx = prb->ndims - 1;
    if (src_rt_mask[m_idx] != dst_rt_mask[m_idx]
            || src_rt_mask[k_idx_src] != wei_rt_mask[k_idx_wei]
            || wei_rt_mask[n_idx] != dst_rt_mask[n_idx]) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }

    // Runtime masks for batch dimensions must be consistent.
    if (prb->ndims > 2) {
        dims_mask_t batch_rt_mask;
        for (int i = 0; i < prb->ndims - 2; ++i)
            batch_rt_mask[i] = true;
        src_rt_mask &= batch_rt_mask;
        wei_rt_mask &= batch_rt_mask;
        dst_rt_mask &= batch_rt_mask;
        if (src_rt_mask != wei_rt_mask || src_rt_mask != dst_rt_mask) {
            res->state = SKIPPED, res->reason = INVALID_CASE;
            return;
        }
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const auto dt = prb->get_dt(kind);
    const float trh = dt == dnnl_f32 ? 1e-6f : epsilon_dt(dt);
    cmp.set_threshold(trh);
    cmp.set_zero_trust_percent(90.f); // TODO: why so bad filling?
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);

    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    dnnl_memory_desc_t src_md {}, wei_md {}, dst_md {}, bia_md {}, def_md {};
    // query md if it was defined at pd creation time
    if (prb->src_runtime_dim_mask().none())
        src_md = query_md(const_pd, DNNL_ARG_SRC);
    if (prb->weights_runtime_dim_mask().none())
        wei_md = query_md(const_pd, DNNL_ARG_WEIGHTS);
    if (prb->dst_runtime_dim_mask().none()) {
        dst_md = query_md(const_pd, DNNL_ARG_DST);
        if (prb->bia_dt != dnnl_data_type_undef)
            bia_md = query_md(const_pd, DNNL_ARG_BIAS);
    }

    // if md is same as default, it means we need to re-create it
    const auto &src_dims = prb->src_dims();
    if (dnnl_memory_desc_equal(&src_md, &def_md)) {
        assert(prb->stag != tag::any);
        src_md = dnn_mem_t::init_md(prb->ndims, src_dims.data(), prb->src_dt(),
                prb->stag, prb->strides[STRIDES_SRC]);
    }

    const auto &weights_dims = prb->weights_dims();
    if (dnnl_memory_desc_equal(&wei_md, &def_md)) {
        assert(prb->wtag != tag::any);
        wei_md = dnn_mem_t::init_md(prb->ndims, weights_dims.data(),
                prb->wei_dt(), prb->wtag, prb->strides[STRIDES_WEI]);
    }

    if (dnnl_memory_desc_equal(&dst_md, &def_md)) {
        assert(prb->dtag != tag::any);
        dst_md = dnn_mem_t::init_md(prb->ndims, prb->dst_dims.data(),
                prb->dst_dt(), prb->dtag, prb->strides[STRIDES_DST]);
    }
    if (prb->bia_dt != dnnl_data_type_undef
            && dnnl_memory_desc_equal(&bia_md, &def_md)) {
        dims_t bia_dims;
        prep_bia_dims(prb, bia_dims);
        bia_md = dnn_mem_t::init_md(
                prb->ndims, bia_dims.data(), prb->bia_dt, tag::abx);
    }

    const auto &scratchpad_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);

    // Use CPU prim as the reference in GPU testing to reduce testing time.
    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim_ref;
    SAFE(init_prim_ref(prim_ref, prb), WARN);

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    dnn_mem_t src_dt(src_md, test_engine);
    dnn_mem_t wei_dt(wei_md, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t bia_dt;
    if (prb->bia_dt != dnnl_data_type_undef)
        bia_dt = dnn_mem_t(bia_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    const auto fp = dnnl_f32;
    dnn_mem_t src_fp(src_md, fp, tag::abx, ref_engine);
    dnn_mem_t wei_fp(wei_md, fp, tag::abx, ref_engine);
    dnn_mem_t dst_fp(dst_md, fp, tag::abx, ref_engine);
    dnn_mem_t bia_fp;
    if (prb->bia_dt != dnnl_data_type_undef)
        bia_fp = dnn_mem_t(bia_md, fp, tag::abx, ref_engine);
    dnn_mem_t scratchpad_fp;
    if (prim_ref)
        scratchpad_fp = dnn_mem_t(
                query_md(query_pd(prim_ref), DNNL_ARG_SCRATCHPAD), ref_engine);

    SAFE(fill_data(SRC, prb, src_dt, src_fp, res), WARN);
    SAFE(fill_data(WEI, prb, wei_dt, wei_fp, res), WARN);
    const int sum_idx = prb->attr.post_ops.find(attr_t::post_ops_t::SUM);
    if (sum_idx >= 0) SAFE(fill_data(DST, prb, dst_dt, dst_fp, res), WARN);
    if (prb->bia_dt != dnnl_data_type_undef)
        SAFE(fill_data(BIA, prb, bia_dt, bia_fp, res), WARN);

    dnn_mem_t scales;
    dnn_mem_t src_zero_points_m, wei_zero_points_m, dst_zero_points_m;
    const auto &wei_zero_point_val
            = prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).value;
    maybe_prepare_runtime_scales(scales, prb->attr.oscale, prb->n, prb->scales);
    maybe_prepare_runtime_zero_points(
            src_zero_points_m, prb->attr, DNNL_ARG_SRC, prb->k, prb->src_zp);
    maybe_prepare_runtime_zero_points(wei_zero_points_m, prb->attr,
            DNNL_ARG_WEIGHTS, 1, &(wei_zero_point_val));
    maybe_prepare_runtime_zero_points(
            dst_zero_points_m, prb->attr, DNNL_ARG_DST, prb->n, prb->dst_zp);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(const_pd, binary_po_args, binary_po_dt,
                 binary_po_fp, /*only_positive=*/false, /*only_integer=*/true),
            WARN);

    args_t args, ref_args;

    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_WEIGHTS, wei_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    if (prb->bia_dt != dnnl_data_type_undef) args.set(DNNL_ARG_BIAS, bia_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(DNNL_ARG_ATTR_OUTPUT_SCALES, scales);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_points_m);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zero_points_m);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_points_m);
    args.set(binary_po_args, binary_po_dt);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (is_bench_mode(CORR)) {
        ref_args.set(DNNL_ARG_SRC, src_fp);
        ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
        if (prb->bia_dt != dnnl_data_type_undef)
            ref_args.set(DNNL_ARG_BIAS, bia_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);
        ref_args.set(DNNL_ARG_SCRATCHPAD, scratchpad_fp);
        ref_args.set(binary_po_args, binary_po_fp);

        check_correctness(prb, {DST}, args, ref_args, setup_cmp, res, prim_ref);
    }

    return measure_perf(res, prim, args);
}

} // namespace matmul
