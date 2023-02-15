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

#include <algorithm>
#include <cstring>

#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

// TODO: refactor the driver to avoid using extra flags of a memory descriptor.
#include "common/memory_desc.hpp"

#include "utils/parallel.hpp"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "reorder.hpp"

namespace reorder {

// Filling for integers is different due to problematic int -> float conversion.
// And it doesn't require many different points to be tested.
int fill_memory_int(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto conf = prb->get_conf(kind);

    for (int64_t idx = 0; idx < mem_fp.nelems(); ++idx) {
        const float gen[4] = {
                conf->max, // saturate to max of output data type
                conf->min, // saturate to min of output data type
                0,
                16,
        };

        const int64_t rng = kind == SRC ? (idx % 4) : ((idx * 5 / 4) % 4);
        mem_fp.set_elem(
                idx, round_to_nearest_representable(conf->dt, gen[rng]));
    }

    SAFE(mem_dt.reorder(mem_fp), WARN);
    return OK;
}

int fill_memory_fp(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto conf = prb->get_conf(kind);
    const auto &src_scales = prb->attr.scales.get(DNNL_ARG_FROM);
    const auto &dst_scales = prb->attr.scales.get(DNNL_ARG_TO);
    const int src_scale_mask = attr_t::get_default_mask(src_scales.policy);
    const int dst_scale_mask = attr_t::get_default_mask(dst_scales.policy);

    for (int64_t idx = 0; idx < mem_fp.nelems(); ++idx) {
        float src_scale = 1.f, dst_scale = 1.f;
        if (!src_scales.is_def()) {
            int64_t src_mask_idx = mem_fp.get_scale_idx(idx, src_scale_mask);
            src_scale = prb->src_scales[src_mask_idx];
        }
        if (!dst_scales.is_def()) {
            int64_t dst_mask_idx = mem_fp.get_scale_idx(idx, dst_scale_mask);
            dst_scale = prb->dst_scales[dst_mask_idx];
        }
        const float scale = src_scale / dst_scale;

        const float gen[7] = {
                conf->max, // saturate to max of output data type
                conf->min, // saturate to min of output data type
                1.6f, // rounding check
                0.2f, // saturate to 0
                1.f / scale, // exact multiplication check
                2.f,
                scale,
        };

        const int64_t rng = kind == SRC ? (idx % 7) : ((idx * 8 / 7) % 7);
        mem_fp.set_elem(
                idx, round_to_nearest_representable(conf->dt, gen[rng]));
    }

    SAFE(mem_dt.reorder(mem_fp), WARN);
    return OK;
}

int fill_memory(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto dt = kind == SRC ? prb->sdt : prb->ddt;
    if (is_integral_dt(dt)) return fill_memory_int(prb, kind, mem_dt, mem_fp);
    return fill_memory_fp(prb, kind, mem_dt, mem_fp);
}

int compare_compensation(const prb_t *prb, dnn_mem_t &mem_s8_comp_ref,
        dnn_mem_t &mem_zp_comp_ref, dnn_mem_t &mem_got, res_t *res) {
    // Note: following check relies on certain assumptions on CPU. These
    // assumptions may not hold for GPU. In addition, it's prohibit to work
    // with raw pointers directly for buffer type of memory.
    if (!is_cpu(get_test_engine())) return FAIL;

    const auto padded_nelems = mem_got.nelems(true);
    // Note: internally offset is aligned on 4, otherwise it's UB.
    size_t first_comp_offset = div_up(padded_nelems, 4) * 4;
    int *comp_handle
            = reinterpret_cast<int *>((char *)mem_got + first_comp_offset);

    const auto cmp_compensation = [&](const dnn_mem_t &mem_ref, int comp_mask) {
        // Idea behind this check:
        // Using knowledge from the library where `comp_handle` starts, and that
        // memory utilizes blocking over OC and G, if present, we wrap that
        // piece of memory which is described by shortened tag coming from prb
        // into a separate memory and reorder it to plain so that it is a
        // straight comparison of values in native plain layout.
        auto comp_md = dnn_mem_t::init_md(mem_ref.ndims(), mem_ref.dims(),
                mem_ref.dt(), trim_tag_by_mask(prb->dtag, comp_mask));
        dnn_mem_t comp_m(comp_md, mem_ref.engine(), {false, comp_handle});

        compare::compare_t cmp;
        cmp.set_zero_trust_percent(100.f); // No sense in zero trust test.
        int status = cmp.compare(mem_ref, comp_m, attr_t(), res);

        // Shift original compensation pointer for next compensation
        comp_handle += comp_m.nelems(true);
        return status;
    };

    if (mem_s8_comp_ref.ndims())
        SAFE(cmp_compensation(mem_s8_comp_ref,
                     prb->get_compensation_mask(FLAG_S8S8_COMP)),
                WARN);
    if (mem_zp_comp_ref.ndims())
        SAFE(cmp_compensation(
                     mem_zp_comp_ref, prb->get_compensation_mask(FLAG_ZP_COMP)),
                WARN);

    return res->state == FAILED ? FAIL : OK;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    auto dims = prb->dims;
    for (int d = 0; d < prb->ndims; ++d)
        if (prb->runtime_dim_mask & (1 << d)) dims[d] = DNNL_RUNTIME_DIM_VAL;

    auto src_d
            = dnn_mem_t::init_md(prb->ndims, dims.data(), prb->sdt, prb->stag);
    auto dst_d
            = dnn_mem_t::init_md(prb->ndims, dims.data(), prb->ddt, prb->dtag);

    // Prepare and assign extra for dst_md.
    auto &extra = static_cast<dnnl_memory_desc_t>(dst_d)->extra;
    extra.flags = dnnl::impl::memory_extra_flags::none;
    if (prb->is_reorder_with_compensation(FLAG_ANY)) {
        for (const auto &i_oflag : prb->oflag) {
            if (i_oflag.first & FLAG_S8S8_COMP) {
                extra.flags |= dnnl::impl::memory_extra_flags::
                        compensation_conv_s8s8;
                extra.compensation_mask = i_oflag.second;

                const float s8_scale_factor = reorder_rescale_factor();
                const bool need_rescale = s8_scale_factor != 1.f;
                if (need_rescale) {
                    extra.flags |= dnnl::impl::memory_extra_flags::scale_adjust;
                    extra.scale_adjust = s8_scale_factor;
                }
            }
            if (i_oflag.first & FLAG_ZP_COMP) {
                extra.flags |= dnnl::impl::memory_extra_flags::
                        compensation_conv_asymmetric_src;
                extra.asymm_compensation_mask = i_oflag.second;
            }
        }
    }

    auto src_engine = init_pd_args.engine;
    auto dst_engine = init_pd_args.engine;
    if (is_gpu()) {
        switch (prb->cross_engine) {
            case CPU2GPU: src_engine = get_cpu_engine(); break;
            case GPU2CPU: dst_engine = get_cpu_engine(); break;
            default: break;
        }
    }

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    init_pd_args.is_iterator_supported = false;
    return dnnl_reorder_primitive_desc_create(
            &init_pd_args.pd, src_d, src_engine, dst_d, dst_engine, dnnl_attr);
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    const auto sdt = prb->sdt;
    const auto ddt = prb->ddt;
    skip_unimplemented_data_type({sdt, ddt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);

    bool scales_ok = true;
#if !defined(DNNL_X64) || DNNL_X64 == 0
    {
        // reference reorder supports only a subset of scale policies
        const std::vector<policy_t> supported_policy = {policy_t::COMMON,
                policy_t::PER_DIM_0, policy_t::PER_DIM_1, policy_t::PER_DIM_01};

        for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
            scales_ok = std::any_of(supported_policy.cbegin(),
                    supported_policy.cend(), [&](const policy_t policy) {
                        return prb->attr.scales.get(arg).policy == policy;
                    });
        }
    }
#endif
    if (!scales_ok) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    if (prb->is_reorder_with_compensation(FLAG_ANY)) {
        // Compensation is supported for s8 dst data type.
        const bool dt_ok = ddt == dnnl_s8;
        // Compensation can be paired with dst scale only.
        const bool attr_ok
                = prb->attr.zero_points.is_def() && prb->attr.post_ops.is_def();
        // Compensation does not support runtime dims.
        const bool rt_ok = prb->runtime_dim_mask == 0;

        // Compensation and scales mask should coincide
        const auto comp_mask = prb->get_compensation_mask(FLAG_ANY);
        bool masks_ok = true;
        for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
            const auto &e = prb->attr.scales.get(arg);
            if (!e.is_def()) {
                int e_mask = attr_t::get_default_mask(e.policy);
                masks_ok = masks_ok && e_mask == comp_mask;
            }
        }

        if (!dt_ok || !attr_ok || !rt_ok || !masks_ok) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }

#if !defined(DNNL_X64) || DNNL_X64 == 0
        // Simple reorder doesn't provide decent coverage for compensated cases.
        // Shut them down unconditionally by default.
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
#endif
    }

    // Destination scale is not supported for runtime dimensions since the
    // implementation logic inverts dst scales and requires scratchpad for
    // `mask > 0` cases which is impossible to estimate with rt dims.
    const auto &dst_scales = prb->attr.scales.get(DNNL_ARG_DST);
    if (!dst_scales.is_def() && attr_t::get_default_mask(dst_scales.policy) > 0
            && prb->runtime_dim_mask != 0) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    // Compensation is supported through jit reorder only, but jit reorder
    // doesn't support different masks for source and destination scales.
    const auto &src_scales = prb->attr.scales.get(DNNL_ARG_SRC);
    if (!src_scales.is_def() && !dst_scales.is_def()) {
        if (attr_t::get_default_mask(src_scales.policy)
                        != attr_t::get_default_mask(dst_scales.policy)
                && prb->is_reorder_with_compensation(FLAG_ANY)) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }

    if (is_cpu()) {
        // CPU reorder doesn't support bf16<-->s32 combinations.
        const bool s32_src_ok = IMPLICATION(sdt == dnnl_s32, ddt != dnnl_bf16);
        const bool s32_dst_ok = IMPLICATION(ddt == dnnl_s32, sdt != dnnl_bf16);
        if (!s32_src_ok || !s32_dst_ok) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }

        // CPU f16 reorders only support f16<->f32 combinations
        const bool f16_src_ok = IMPLICATION(
                sdt == dnnl_f16, ddt == dnnl_f16 || ddt == dnnl_f32);
        const bool f16_dst_ok = IMPLICATION(
                ddt == dnnl_f16, sdt == dnnl_f16 || sdt == dnnl_f32);
        if (!f16_src_ok || !f16_dst_ok) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }

    if (is_gpu()) {
        // GPU does not support run-time dims.
        // Reorders w/ compensation are not supported by design: zp_comp is done
        // in kernels directly, but s8s8 instructions are available in HW.
        if (prb->runtime_dim_mask != 0
                || prb->is_reorder_with_compensation(FLAG_ANY)) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // No sense in cross engine reorders when one of devices is switched off.
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_NONE
    auto cross_engine = prb->cross_engine;
    if (cross_engine == CPU2GPU || cross_engine == GPU2CPU)
        res->state = SKIPPED, res->reason = INVALID_CASE;
#endif

    // Zero-points can't be used with sum post-op.
    if (!prb->attr.zero_points.is_def(DNNL_ARG_DST)
            && prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) != -1) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }

    // only integral data types can have zero points
    const bool is_src_zp_ok = is_integral_dt(prb->sdt)
            || prb->attr.zero_points.is_def(DNNL_ARG_SRC);
    const bool is_dst_zp_ok = is_integral_dt(prb->ddt)
            || prb->attr.zero_points.is_def(DNNL_ARG_DST);
    if (!(is_src_zp_ok && is_dst_zp_ok)) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const bool has_s32 = prb->sdt == dnnl_s32 || prb->ddt == dnnl_s32;
    const bool has_s8 = prb->sdt == dnnl_s8 || prb->ddt == dnnl_s8;
    const bool has_u8 = prb->sdt == dnnl_u8 || prb->ddt == dnnl_u8;
    // For u8 4/7 inputs becomes 0, for s32/s8 3/7 inputs becomes 0;
    const float zero_trust_percent = has_u8 ? 58.f
            : (has_s32 || has_s8)           ? 43.f
                                            : 30.f;
    cmp.set_zero_trust_percent(zero_trust_percent);

    // Additional check to avoid false-positive result from f32->s32 conversion
    // in case of sum post-op on GPU happening when two max_dt values
    // are summed together.
    const auto reorder_add_check
            = [&](const compare::compare_t::driver_check_func_args_t &args) {
                  if (args.dt == dnnl_s32 && args.got == max_dt(args.dt)
                          && is_gpu()) {
                      // 128.f = float(INT_MAX) - BENCHDNN_S32_TO_F32_SAT_CONST;
                      return args.diff == 128.f;
                  }
                  return false;
              };
    cmp.set_driver_check_function(reorder_add_check);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> src_md {}, dst_md {};
    if (prb->runtime_dim_mask != 0) {
        // re-create memory descriptors with defined dims
        src_md = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->sdt, prb->stag);
        dst_md = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->ddt, prb->dtag);
    } else {
        src_md = clone_md(query_md(const_pd, DNNL_ARG_SRC));
        dst_md = clone_md(query_md(const_pd, DNNL_ARG_DST));
    }
    const auto &scratchpad_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);

    dnnl_engine_t src_engine
            = query_engine(const_pd, dnnl_query_reorder_src_engine);
    dnnl_engine_t dst_engine
            = query_engine(const_pd, dnnl_query_reorder_dst_engine);
    const auto &ref_engine = get_cpu_engine();

    dnn_mem_t src_fp(src_md, dnnl_f32, tag::abx, ref_engine);
    dnn_mem_t src_dt(src_md, src_engine);

    dnn_mem_t scratchpad_dt(scratchpad_md, src_engine);

    dnn_mem_t dst_fp(dst_md, dnnl_f32, tag::abx, ref_engine);
    dnn_mem_t dst_dt(dst_md, dst_engine);

    SAFE(fill_memory(prb, SRC, src_dt, src_fp), WARN);

    const bool has_sum
            = prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0;
    if (has_sum) { SAFE(fill_memory(prb, DST, dst_dt, dst_fp), WARN); }

    const int src_mask = attr_t::get_default_mask(
            prb->attr.scales.get(DNNL_ARG_SRC).policy);
    const int dst_mask = attr_t::get_default_mask(
            prb->attr.scales.get(DNNL_ARG_DST).policy);
    dnn_mem_t src_scales, dst_scales;
    dnn_mem_t src_zero_points_m, dst_zero_points_m;

    maybe_prepare_runtime_scales(src_scales, prb->attr.scales.get(DNNL_ARG_SRC),
            prb->nelems(src_mask), prb->src_scales);
    maybe_prepare_runtime_scales(dst_scales, prb->attr.scales.get(DNNL_ARG_DST),
            prb->nelems(dst_mask), prb->dst_scales);
    maybe_prepare_runtime_zero_points(
            src_zero_points_m, prb->attr, DNNL_ARG_SRC, 1, prb->src_zp);
    maybe_prepare_runtime_zero_points(
            dst_zero_points_m, prb->attr, DNNL_ARG_DST, 1, prb->dst_zp);

    args_t args, ref_args;

    args.set(DNNL_ARG_FROM, src_dt);
    args.set(DNNL_ARG_TO, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales);
    args.set(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_points_m);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_points_m);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (is_bench_mode(CORR)) {
        const auto assign_comp_mem = [&](dnn_mem_t &m, flag_bit_t flag) {
            if (prb->is_reorder_with_compensation(flag)) {
                dims_t dims = prb->get_compensation_dims(flag);
                int ndims = static_cast<int>(dims.size());
                auto md = dnn_mem_t::init_md(
                        ndims, dims.data(), dnnl_s32, tag::abx);
                m = dnn_mem_t(md, ref_engine);
            }
            return OK;
        };

        dnn_mem_t dst_s8_comp_ref, dst_zp_comp_ref;
        assign_comp_mem(dst_s8_comp_ref, FLAG_S8S8_COMP);
        assign_comp_mem(dst_zp_comp_ref, FLAG_ZP_COMP);

        ref_args.set(DNNL_ARG_FROM, src_fp);
        ref_args.set(DNNL_ARG_TO, dst_fp);
        ref_args.set(DNNL_ARG_SRC_1, dst_s8_comp_ref); // Additional input
        ref_args.set(DNNL_ARG_SRC_2, dst_zp_comp_ref); // Additional input
        ref_args.set(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales);
        ref_args.set(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales);

        // Remove extra desc so that reorders with compensation could have
        // proper reorder from blocked layout to plain for comparison.
        dnnl::impl::memory_extra_desc_t empty_extra {};
        const auto orig_dst_extra = dst_dt.md_->extra;
        dst_dt.md_->extra = empty_extra;

        // Validate main reorder part.
        check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);

        // Restore extra for compensation comparison and performance mode.
        dst_dt.md_->extra = orig_dst_extra;

        // Validate compensated reorder part.
        if (prb->is_reorder_with_compensation(FLAG_ANY)) {
            compare_compensation(
                    prb, dst_s8_comp_ref, dst_zp_comp_ref, dst_dt, res);
        }
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace reorder
