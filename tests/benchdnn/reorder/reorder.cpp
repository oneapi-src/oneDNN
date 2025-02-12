/*******************************************************************************
* Copyright 2017-2025 Intel Corporation
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

#include "utils/fill.hpp"
#include "utils/parallel.hpp"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "reorder.hpp"

namespace reorder {

int fill_mem(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    const auto conf = prb->get_conf(kind);

    const float gen[] = {
            conf->max, // saturate to max
            conf->min, // saturate to min
            0.f,
            0.25f,
            0.5f,
            1.f,
            1.5f,
            2.f,
            16.f,
            64.f,
    };
    const auto table_size = sizeof(gen) / sizeof(gen[0]);
    // MIOpen doesn't work properly when tensors are filled with 0xFF.
    const bool zero_out_wa = is_amd_gpu();

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const int64_t table_idx = kind == SRC
                ? (i % table_size)
                : ((i * (table_size + 1) / table_size) % table_size);
        mem_fp.set_elem(
                i, round_to_nearest_representable(conf->dt, gen[table_idx]));
        if (zero_out_wa) mem_dt.set_elem(i, 0);
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

dnn_mem_t setup_compensation_memory(const prb_t *prb, flag_bit_t flag) {
    dnn_mem_t m;
    if (prb->is_reorder_with_compensation(flag)) {
        dims_t dims = prb->get_compensation_dims(flag);
        int ndims = static_cast<int>(dims.size());
        auto md = dnn_mem_t::init_md(ndims, dims.data(), dnnl_s32, tag::abx);
        m = dnn_mem_t(md, get_cpu_engine());
    }
    return m;
};

int compare_compensation(const prb_t *prb, dnn_mem_map_t &mem_map,
        dnn_mem_map_t &ref_mem_map, res_t *res) {
    // Note: following check relies on certain assumptions on CPU. These
    // assumptions may not hold for GPU. In addition, it's prohibit to work
    // with raw pointers directly for buffer type of memory.
    if (!is_cpu(get_test_engine())) return FAIL;

    const auto &mem_got = mem_map[DNNL_ARG_DST];
    const auto &mem_s8_comp_ref = ref_mem_map[DNNL_ARG_SRC_1];
    const auto &mem_zp_comp_ref = ref_mem_map[DNNL_ARG_SRC_2];

    const auto padded_nelems = mem_got.nelems(true);
    // Note: internally offset is aligned on 4, otherwise it's UB.
    size_t first_comp_offset = rnd_up(padded_nelems, 4);
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
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    auto dims = prb->dims;
    for (int d = 0; d < prb->ndims; ++d)
        if (prb->runtime_dim_mask & (1 << d)) dims[d] = DNNL_RUNTIME_DIM_VAL;

    auto src_d = dnn_mem_t::init_md(prb->ndims, dims.data(),
            force_f32_dt ? dnnl_f32 : prb->sdt, prb->stag, prb->strides[0]);
    auto dst_d = dnn_mem_t::init_md(prb->ndims, dims.data(),
            force_f32_dt ? dnnl_f32 : prb->ddt, prb->dtag, prb->strides[1]);

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
    TIME_C_PD(DNN_SAFE_STATUS(dnnl_reorder_primitive_desc_create(
            &init_pd_args.pd, init_pd_args.src_md ? init_pd_args.src_md : src_d,
            src_engine, dst_d, dst_engine, dnnl_attr)));

    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    const auto sdt = prb->sdt;
    const auto ddt = prb->ddt;
    skip_unimplemented_data_type({sdt, ddt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_reorder, sdt);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_reorder);

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
        res->state = SKIPPED;
        res->reason = skip_reason::case_not_supported;
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
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }

#if !defined(DNNL_X64) || DNNL_X64 == 0
        // Simple reorder doesn't provide decent coverage for compensated cases.
        // Shut them down unconditionally by default.
        res->state = SKIPPED;
        res->reason = skip_reason::case_not_supported;
        return;
#endif
    }

    // Destination scale is not supported for runtime dimensions since the
    // implementation logic inverts dst scales and requires scratchpad for
    // `mask > 0` cases which is impossible to estimate with rt dims.
    const auto &dst_scales = prb->attr.scales.get(DNNL_ARG_DST);
    if (!dst_scales.is_def() && attr_t::get_default_mask(dst_scales.policy) > 0
            && prb->runtime_dim_mask != 0) {
        res->state = SKIPPED;
        res->reason = skip_reason::case_not_supported;
        return;
    }

    // Compensation is supported through jit reorder only, but jit reorder
    // doesn't support different masks for source and destination scales.
    const auto &src_scales = prb->attr.scales.get(DNNL_ARG_SRC);
    if (!src_scales.is_def() && !dst_scales.is_def()) {
        if (attr_t::get_default_mask(src_scales.policy)
                        != attr_t::get_default_mask(dst_scales.policy)
                && prb->is_reorder_with_compensation(FLAG_ANY)) {
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }
    }

    if (is_cpu()) {
        // Int4 reorder support is limited on CPU.
        if (sdt == dnnl_s4 || ddt == dnnl_s4 || sdt == dnnl_u4
                || ddt == dnnl_u4) {
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }

        // CPU reorder doesn't support (xf8,xf16)<-->s32 combinations.
        const bool s32_src_ok = IMPLICATION(sdt == dnnl_s32,
                ddt != dnnl_f8_e5m2 && ddt != dnnl_f8_e4m3 && ddt != dnnl_bf16
                        && ddt != dnnl_f16);
        const bool s32_dst_ok = IMPLICATION(ddt == dnnl_s32,
                sdt != dnnl_f8_e5m2 && sdt != dnnl_f8_e4m3 && sdt != dnnl_bf16
                        && sdt != dnnl_f16);
        if (!s32_src_ok || !s32_dst_ok) {
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }

        // CPU f16 reorders only support f16<->f32 combinations
        const bool f16_src_ok = IMPLICATION(
                sdt == dnnl_f16, ddt == dnnl_f16 || ddt == dnnl_f32);
        const bool f16_dst_ok = IMPLICATION(
                ddt == dnnl_f16, sdt == dnnl_f16 || sdt == dnnl_f32);
        if (!f16_src_ok || !f16_dst_ok) {
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }

        // CPU xf8 reorders only support xf8<->(f16,f32) combinations
        const bool xf8_src_ok
                = IMPLICATION(ddt == dnnl_f8_e5m2 || ddt == dnnl_f8_e4m3,
                        sdt == dnnl_f16 || sdt == dnnl_f32);
        const bool xf8_dst_ok
                = IMPLICATION(sdt == dnnl_f8_e5m2 || sdt == dnnl_f8_e4m3,
                        ddt == dnnl_f16 || ddt == dnnl_f32);
        if (!xf8_src_ok || !xf8_dst_ok) {
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }
    }

    if (is_gpu()) {
        // GPU does not support run-time dims.
        // Reorders w/ compensation are not supported by design: zp_comp is done
        // in kernels directly, but s8s8 instructions are available in HW.
        if (prb->runtime_dim_mask != 0
                || prb->is_reorder_with_compensation(FLAG_ANY)) {
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }

        // GPU doesn't support f8_e5m2/f8_e4m3.
        const bool is_xf8 = prb->sdt == dnnl_f8_e5m2 || prb->sdt == dnnl_f8_e4m3
                || prb->ddt == dnnl_f8_e5m2 || prb->ddt == dnnl_f8_e4m3;
        if (is_xf8) {
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // No sense in cross engine reorders when one of devices is switched off.
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_NONE
    auto cross_engine = prb->cross_engine;
    if (cross_engine == CPU2GPU || cross_engine == GPU2CPU) {
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }
#endif

    // Zero-points can't be used with sum post-op.
    if (!prb->attr.zero_points.is_def(DNNL_ARG_DST)
            && prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) != -1) {
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }

    // only integral data types can have zero points
    const bool is_src_zp_ok = is_integral_dt(prb->sdt)
            || prb->attr.zero_points.is_def(DNNL_ARG_SRC);
    const bool is_dst_zp_ok = is_integral_dt(prb->ddt)
            || prb->attr.zero_points.is_def(DNNL_ARG_DST);
    if (!(is_src_zp_ok && is_dst_zp_ok)) {
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    // This value can be exact without scales. Scales may affect the value.
    // Avoid any scales logic involved until needed.
    cmp.set_zero_trust_percent(80.f);

    // `f8_e4m3` range is very short which makes inputs convert into NaNs.
    cmp.set_op_output_has_nans(prb->sdt == dnnl_f8_e4m3);

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

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_args = {
            DNNL_ARG_FROM,
            DNNL_ARG_TO,
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
            case DNNL_ARG_FROM: {
                SAFE(fill_mem(prb, SRC, mem, ref_mem), WARN);
                // Additional inputs to compare compensation buffers.
                ref_mem_map.emplace(DNNL_ARG_SRC_1,
                        setup_compensation_memory(prb, FLAG_S8S8_COMP));
                ref_mem_map.emplace(DNNL_ARG_SRC_2,
                        setup_compensation_memory(prb, FLAG_ZP_COMP));
            } break;
            case DNNL_ARG_TO: {
                const auto &po = prb->attr.post_ops;
                const int sum_idx = po.find(attr_t::post_ops_t::SUM);
                // MIOpen doesn't work properly when tensors are filled with 0xFF.
                if (sum_idx >= 0 || is_amd_gpu()) {
                    SAFE(fill_mem(prb, DST, mem, ref_mem), WARN);

                    // Bitwise mode for sum requires a copy due to data for
                    // post-op will be overwritten and it must be refreshed.
                    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
                        SAFE(mem_map.at(-exec_arg).reorder(ref_mem), WARN);
                    }
                }
            } break;
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

    if (has_bench_mode_bit(mode_bit_t::corr)) {
        // Remove extra desc so that reorders with compensation could have
        // proper reorder from blocked layout to plain for comparison.
        auto &dst_dt = mem_map[DNNL_ARG_DST];
        dnnl::impl::memory_extra_desc_t empty_extra {};
        const auto orig_dst_extra = dst_dt.md_->extra;
        dst_dt.md_->extra = empty_extra;

        // Validate main reorder part.
        check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);

        // Restore extra for compensation comparison and performance mode.
        dst_dt.md_->extra = orig_dst_extra;

        // Validate compensated reorder part.
        if (prb->is_reorder_with_compensation(FLAG_ANY)) {
            compare_compensation(prb, mem_map, ref_mem_map, res);
        }
    }
    SAFE(check_bitwise(prim, {DST}, args, prb->attr, prb->inplace, res), WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace reorder
