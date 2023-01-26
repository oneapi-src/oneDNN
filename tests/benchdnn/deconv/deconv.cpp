/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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
#include <utility>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils/fill.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "deconv/deconv.hpp"
#include "prelu/prelu.hpp"

namespace deconv {

int transpose_data_wei(
        const prb_t *prb, const dnn_mem_t &wei, const dnn_mem_t &wei_tr) {
    benchdnn_parallel_nd(prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kd,
            prb->kh, prb->kw,
            [&](int64_t g, int64_t oc, int64_t ic, int64_t kd, int64_t kh,
                    int64_t kw) {
                int64_t ch_idx
                        = (g * prb->ic / prb->g + ic) * prb->oc / prb->g + oc;
                int64_t idx = ((ch_idx * prb->kd + kd) * prb->kh + kh) * prb->kw
                        + kw;
                ((float *)wei_tr)[idx]
                        = ((float *)wei)[wei_off_f(prb, g, oc, ic, kd, kh, kw)];
            });

    return OK;
}

double get_non_zero_trust_percent(const prb_t *prb, data_kind_t kind) {
    auto negative_to_zero = [&]() {
        using pk = attr_t::post_ops_t::kind_t;
        const auto &po = prb->attr.post_ops;
        int count = 0;

        // Check for all post-ops that convert negative to zero
        std::vector<pk> non_neg_po {pk::ABS};
        std::vector<pk> non_neg_alpha_0_po {
                pk::CLIP, pk::CLIP_V2, pk::ELU, pk::RELU};
        for (int i = 0; i < po.len(); ++i) {
            const auto &e = po.entry[i];
            if (!e.is_eltwise_kind()) continue;

            auto k = e.kind;
            auto alpha = e.eltwise.alpha;

            count += std::any_of(non_neg_po.cbegin(), non_neg_po.cend(),
                    [k](const pk alg) { return alg == k; });
            count += std::any_of(non_neg_alpha_0_po.cbegin(),
                    non_neg_alpha_0_po.cend(), [k, alpha](const pk alg) {
                        return alg == k && alpha == 0;
                    });
        }
        // Check for u8 dst
        count += prb->cfg[DST].dt == dnnl_u8;
        // Check for physically padded area in the output
        count += prb->od > prb->id || prb->oh > prb->ih || prb->ow > prb->iw;

        return !!count;
    };

    double trust = 0.3; /* why? */
    switch (kind) {
        case SRC: trust /= prb->sd * prb->sh * prb->sw; break;
        case WEI:
            trust /= 1. * prb->kd * prb->kh * prb->kw
                    / MIN3(prb->kd * prb->kh * prb->kw,
                            prb->id * prb->ih * prb->iw,
                            prb->od * prb->oh * prb->ow);
            break;
        case BIA:
            trust = 0.8 * prb->cfg[DST].f_sparsity; /* why? */
            break;
        case DST: trust /= (1.f + negative_to_zero()); break;
        default: assert(!"unsupported data kind");
    }

    return trust;
}

int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const size_t nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Use dense filling for small problems.
    int src_nelems_mask = powf(2.f, prb->ndims) - 1;
    src_nelems_mask -= 1; // remove minibatch as independent dimension
    auto src_nelems = prb->desc_nelems(DNNL_ARG_SRC, src_nelems_mask);
    if (prb->has_groups) src_nelems /= prb->g; // groups are also independent

    const auto &c = prb->cfg[SRC];
    const int range = c.f_max - c.f_min + 1;
    const float sparsity
            = (!has_bench_mode_bit(mode_bit_t::corr) || src_nelems < 100)
            ? 1.f
            : c.f_sparsity;

    const auto &e_zp_src = prb->attr.zero_points.get(DNNL_ARG_SRC);
    const bool has_src_zp = !e_zp_src.is_def();
    const int src_zp_mask = attr_t::get_default_mask(e_zp_src.policy);
    int src_zp = has_src_zp && src_zp_mask == 0 ? e_zp_src.value : 0;

    benchdnn_parallel_nd(prb->mb, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int64_t mb, int64_t ic, int64_t id, int64_t ih, int64_t iw) {
                const int64_t gen
                        = 101 * id + 103 * ih + 107 * iw + 109 * mb + 113 * ic;
                const bool non_base = flip_coin(gen, sparsity);
                float value = non_base ? c.f_min + gen * c.f_step % range
                                       : c.f_base;
                value += src_zp; // Add zp so that it will be subtracted.

                ((float *)mem_fp)[src_off_f(prb, mb, 0, ic, id, ih, iw)]
                        = round_to_nearest_representable(mem_dt.dt(), value);
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_wei(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const size_t nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    dnnl_data_type_t dt_check = dnnl_s8;
#if DNNL_AARCH64
    /* Note for x64:
    Both data types of src and weight are s8, oneDNN addds 128 to one of the s8
    input to make it of type u8 instead, as explained in
    https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html or
    doc/advanced/int8_computations.md
    It is because `VPDPBUSD` instruction uses the combination of s8 and u8 as
    input.

    Note for AArch64:
    Because dot product instructions of AArch64 "SDOT" receives s8 input
    for both src and weight, the addition (and its counterpart of subtraction)
    is not required for AArch64.
    */
    if (res->impl_name.find("jit", 0) == 0) dt_check = dnnl_u8;
#endif

    const auto &c = prb->cfg[WEI];
    const int range = c.f_max - c.f_min + 1;
    const float sparsity
            = !has_bench_mode_bit(mode_bit_t::corr) ? 1.f : c.f_sparsity;

    benchdnn_parallel_nd(prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kd,
            prb->kh, prb->kw,
            [&](int64_t g, int64_t oc, int64_t ic, int64_t kd, int64_t kh,
                    int64_t kw) {
                const int64_t gen = 113 * g + 127 * kd + 131 * kh + 137 * kw
                        + 139 * oc + 149 * ic + 151;
                const bool non_base = flip_coin(gen, sparsity);
                const float value = non_base ? c.f_min + gen * c.f_step % range
                                             : c.f_base;
                ((float *)mem_fp)[wei_off_f(prb, g, oc, ic, kd, kh, kw)]
                        = round_to_nearest_representable(mem_dt.dt(), value);
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    const bool wei_x8x8
            = prb->cfg[WEI].dt == dnnl_s8 && prb->cfg[SRC].dt == dt_check;
    const bool is_def_zp = prb->attr.zero_points.is_def(DNNL_ARG_SRC);
    if ((wei_x8x8 || !is_def_zp) && is_cpu()) {
        // Check that s8 -> s8_comp exists in the library since users may have
        // already quantized data.
        dnn_mem_t mem_fp_s8(mem_fp.md_, dnnl_s8, tag::abx, get_cpu_engine());
        dnn_mem_t mem_dt_s8(mem_dt.md_, get_test_engine());
        SAFE(mem_fp_s8.reorder(mem_fp), WARN);
        SAFE(mem_dt_s8.reorder(mem_fp_s8), WARN);
        SAFE(mem_dt.size() == mem_dt_s8.size() ? OK : FAIL, WARN);
        int rc = std::memcmp((void *)mem_dt, (void *)mem_dt_s8, mem_dt.size());
        SAFE(rc == 0 ? OK : FAIL, WARN);
    }

    return OK;
}

int fill_bia(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const size_t nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const auto &c = prb->cfg[BIA];
    const int range = c.f_max - c.f_min + 1;

    for (size_t i = 0; i < nelems; ++i) {
        const int gen = (int)(151 * i);
        const bool non_base = flip_coin(gen, c.f_sparsity);
        const float value
                = non_base ? c.f_min + gen * c.f_step % range : c.f_base;

        ((float *)mem_fp)[i]
                = round_to_nearest_representable(mem_dt.dt(), value);
    }

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_dst_with_params(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        dnnl_data_type_t dt, double sparsity, int min, int max, int base,
        int step, res_t *res) {
    const size_t nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const int range = max - min + 1;

    benchdnn_parallel_nd(prb->mb, prb->oc, prb->od, prb->oh, prb->ow,
            [&](int64_t mb, int64_t oc, int64_t od, int64_t oh, int64_t ow) {
                const int64_t gen
                        = 157 * od + 163 * oh + 167 * ow + 173 * mb + 179 * oc;
                const bool non_base = flip_coin(gen, sparsity);
                const float value = non_base ? min + gen * step % range : base;

                ((float *)mem_fp)[dst_off_f(prb, mb, 0, oc, od, oh, ow)]
                        = round_to_nearest_representable(mem_dt.dt(), value);
            });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    auto dst_dt = mem_dt.dt();
    int sum_ind = prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM);
    auto sum_dt = (sum_ind != -1) ? prb->attr.post_ops.entry[sum_ind].sum.dt
                                  : dnnl_data_type_undef;
    bool diff_sum_dst_types
            = sum_dt != dnnl_data_type_undef && sum_dt != dst_dt;
    bool sum_dt_is_int8 = sum_dt == dnnl_s8 || sum_dt == dnnl_u8;

    const auto &c = prb->cfg[DST];
    float f_min = c.f_min;
    float f_max = c.f_max;
    if (diff_sum_dst_types && sum_dt_is_int8) {
        f_min = lowest_dt(sum_dt);
        f_max = max_dt(sum_dt);
    }

    // Change mem dt to sum dt, so we can save sum data properly.
    if (diff_sum_dst_types) { mem_dt.set_dt(sum_dt); }

    fill_dst_with_params(prb, mem_dt, mem_fp, sum_dt, c.f_sparsity, f_min,
            f_max, c.f_base, c.f_step, res);

    // Return dst data type back.
    if (diff_sum_dst_types) { mem_dt.set_dt(dst_dt); }
    return OK;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    auto src_d = dnn_mem_t::init_md(
            prb->ndims, prb->src_dims().data(), prb->cfg[SRC].dt, prb->stag);
    auto wei_d = dnn_mem_t::init_md(prb->ndims + prb->has_groups,
            prb->wei_dims().data(), prb->cfg[WEI].dt, prb->wtag);
    auto bia_d = dnn_mem_t::init_md(
            1, prb->bia_dims().data(), prb->cfg[BIA].dt, tag::any);
    auto dst_d = dnn_mem_t::init_md(
            prb->ndims, prb->dst_dims().data(), prb->cfg[DST].dt, prb->dtag);

    dnnl_alg_kind_t alg = dnnl_deconvolution_direct;
    if (prb->alg == WINO) alg = dnnl_deconvolution_winograd;

    attr_args_t attr_args;
    auto wei_scale = prb->attr.scales.get(DNNL_ARG_WEIGHTS);
    if (wei_scale.policy == policy_t::PER_OC) {
        // oihw: per_oc: 1 << 0 -> 1
        // goihw: per_oc: 1 << 1 + 1 << 0 -> 3
        auto wei_mask = prb->has_groups ? 3 : 1;
        attr_args.prepare_scales(prb->attr, DNNL_ARG_WEIGHTS, wei_mask);
    }
    attr_args.prepare_post_ops_mds(
            prb->attr, prb->ndims, prb->dst_dims().data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            if (prb->dir != FWD_B) bia_d.reset(nullptr);
            DNN_SAFE_STATUS(dnnl_deconvolution_forward_primitive_desc_create(
                    &init_pd_args.pd, init_pd_args.engine,
                    prb->dir == FWD_I ? dnnl_forward_inference
                                      : dnnl_forward_training,
                    alg, init_pd_args.src_md ? init_pd_args.src_md : src_d,
                    wei_d, bia_d, dst_d, prb->strides().data(),
                    prb->dilations().data(), prb->padding().data(),
                    prb->padding_r().data(), dnnl_attr));
            break;
        case BWD_D:
            DNN_SAFE_STATUS(
                    dnnl_deconvolution_backward_data_primitive_desc_create(
                            &init_pd_args.pd, init_pd_args.engine, alg, src_d,
                            wei_d, dst_d, prb->strides().data(),
                            prb->dilations().data(), prb->padding().data(),
                            prb->padding_r().data(), init_pd_args.hint,
                            dnnl_attr));
            break;
        case BWD_W:
        case BWD_WB:
            if (prb->dir == BWD_W) bia_d.reset(nullptr);
            DNN_SAFE_STATUS(
                    dnnl_deconvolution_backward_weights_primitive_desc_create(
                            &init_pd_args.pd, init_pd_args.engine, alg, src_d,
                            wei_d, bia_d, dst_d, prb->strides().data(),
                            prb->dilations().data(), prb->padding().data(),
                            prb->padding_r().data(), init_pd_args.hint,
                            dnnl_attr));
            break;
        default: DNN_SAFE_STATUS(dnnl_invalid_arguments);
    }

    // TODO: enabled back when query is added to pd??
    //DNN_SAFE_STATUS(cd.accum_data_type == prb->cfg[ACC].dt
    //                ? dnnl_success
    //                : dnnl_unimplemented);

    return dnnl_success;
}

int init_prim_ref(
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim_ref, const prb_t *prb) {
    if (!(has_bench_mode_bit(mode_bit_t::corr) && is_gpu() && fast_ref_gpu))
        return OK;

    // Create a new copy of prb to avoid potentially corrupting the test by
    // modifying prb in place.
    // DIRECT algorithm is used to prevent fallback  to the slow benchdnn
    // reference implementation.
    auto cpu_attr = prb->attr;
    update_cpu_ref_attrs(cpu_attr);
    prb_t prb_cpu {*prb, prb->dir, conf_f32, tag::abx, tag::abx, tag::abx,
            DIRECT, cpu_attr, prb->ctx_init, prb->ctx_exe, prb->mb};

    init_pd_args_t<prb_t> init_pd_args(
            /* res = */ nullptr, get_cpu_engine(), &prb_cpu, prb->dir,
            /* hint = */ nullptr, /* src_md = */ nullptr);
    init_pd(init_pd_args);

    benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> pdw;
    fetch_impl(pdw, init_pd_args, /* res = */ nullptr,
            /* is_service_prim = */ true);

    dnnl_primitive_t prim_ref_ {};
    if (pdw) {
        if (query_impl_info(pdw) == "ref:any") return OK;
        DNN_SAFE(dnnl_primitive_create(&prim_ref_, pdw), WARN);
        BENCHDNN_PRINT(5, "CPU reference oneDNN implementation: %s\n",
                query_impl_info(pdw).c_str());
    }
    prim_ref.reset(prim_ref_);
    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type(
            {prb->cfg[SRC].dt, prb->cfg[WEI].dt, prb->cfg[DST].dt}, prb->dir,
            res);
    skip_unimplemented_sum_po(prb->attr, res);

    // GPU supports only post ops and all but x8s8bf16 cfg
    if (is_gpu()) {
        const bool is_x8s8bf16_cfg
                = prb->cfg[WEI].dt == dnnl_s8 && prb->cfg[DST].dt == dnnl_bf16;
        const bool fwd_ok = !is_x8s8bf16_cfg;
        if (!fwd_ok) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const bool compare_with_norm = (prb->alg & WINO);
    cmp.set_norm_validation_mode(compare_with_norm);

    float trh = prb->cfg[kind].eps;
    if ((prb->alg & WINO) && (prb->dir & FLAG_WEI)) {
        // This is an empirical equation derived by observing growth error with
        // increasing 'k' dimension in gemm of winograd
        const float log_const = log10(0.125 * prb->mb * prb->oh * prb->ow);
        trh = prb->cfg[kind].eps * (MAX2(1, pow(10, 0.4 * log_const)));
    }
    cmp.set_threshold(trh);

    const float zpp = (1.f - get_non_zero_trust_percent(prb, kind)) * 100.f;
    cmp.set_zero_trust_percent(zpp);
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_BIAS,
            DNNL_ARG_DST,
    };
    static const std::vector<int> exec_bwd_d_args = {
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_BIAS,
            DNNL_ARG_DIFF_DST,
    };
    static const std::vector<int> exec_bwd_w_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DIFF_WEIGHTS,
            DNNL_ARG_DIFF_BIAS,
            DNNL_ARG_DIFF_DST,
    };
    return (dir & FLAG_FWD)    ? exec_fwd_args
            : (dir & FLAG_WEI) ? exec_bwd_w_args
                               : exec_bwd_d_args;
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
            case DNNL_ARG_SRC:
                SAFE(fill_src(prb, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_WEIGHTS: {
                SAFE(fill_wei(prb, mem, ref_mem, res), WARN);
                // To re-use conv implementation, we need additional weights
                // with transposed input/output channels.
                dnnl_dims_t wei_tr_dims {};
                auto wei_ndims = query_md_ndims(mem.md_);
                std::copy(query_md_dims(mem.md_),
                        query_md_dims(mem.md_) + wei_ndims, wei_tr_dims);
                // Queried weights are always with groups.
                std::swap(wei_tr_dims[1], wei_tr_dims[2]);
                auto wei_tr_md = dnn_mem_t::init_md(
                        wei_ndims, wei_tr_dims, dnnl_f32, tag::abx);

                ref_mem_map.emplace(
                        DNNL_ARG_WEIGHTS_1, dnn_mem_t(wei_tr_md, ref_engine));
                SAFE(transpose_data_wei(
                             prb, ref_mem, ref_mem_map[DNNL_ARG_WEIGHTS_1]),
                        WARN);
            } break;
            case DNNL_ARG_BIAS:
                SAFE(fill_bia(prb, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_DST:
                if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM)
                        >= 0)
                    SAFE(fill_dst(prb, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_dst(prb, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_DIFF_WEIGHTS: {
                // To re-use conv implementation, we need additional weights
                // with transposed input/output channels.
                dnnl_dims_t wei_tr_dims {};
                auto wei_ndims = query_md_ndims(mem.md_);
                std::copy(query_md_dims(mem.md_),
                        query_md_dims(mem.md_) + wei_ndims, wei_tr_dims);
                // Queried weights are always with groups.
                std::swap(wei_tr_dims[1], wei_tr_dims[2]);
                auto wei_tr_md = dnn_mem_t::init_md(
                        wei_ndims, wei_tr_dims, dnnl_f32, tag::abx);
                ref_mem_map.emplace(DNNL_ARG_DIFF_WEIGHTS_1,
                        dnn_mem_t(wei_tr_md, ref_engine));
            } break;
            case DNNL_ARG_SCRATCHPAD:
                // Reference CPU impl may need a different size for scratchpad.
                // Need to query it instead of replicating one from GPU.
                if (prim_ref) {
                    ref_mem_map[exec_arg] = dnn_mem_t(
                            query_md(query_pd(prim_ref), DNNL_ARG_SCRATCHPAD),
                            ref_engine);
                }
                break;
            default: { // Process all attributes here
                int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
                        - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
                bool is_post_ops_arg = (exec_arg & post_ops_range);
                bool is_scales_arg = (exec_arg & DNNL_ARG_ATTR_SCALES);
                bool is_zero_point_arg = (exec_arg & DNNL_ARG_ATTR_ZERO_POINTS);

                if (is_post_ops_arg) {
                    if (exec_arg & DNNL_ARG_SRC_1)
                        SAFE(binary::fill_mem(exec_arg, mem, ref_mem), WARN);
                    else if (exec_arg & DNNL_ARG_WEIGHTS)
                        SAFE(prelu::fill_data(WEI, mem, ref_mem), WARN);
                } else if (is_scales_arg) {
                    int local_exec_arg = exec_arg ^ DNNL_ARG_ATTR_SCALES;
                    SAFE(fill_scales(prb->attr, local_exec_arg, mem, ref_mem),
                            WARN);
                } else if (is_zero_point_arg) {
                    int local_exec_arg = exec_arg ^ DNNL_ARG_ATTR_ZERO_POINTS;
                    SAFE(fill_zero_points(
                                 prb->attr, local_exec_arg, mem, ref_mem),
                            WARN);
                }
            } break;
        }
        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds;
    if (prb->dir & FLAG_FWD) {
        check_kinds = {DST};
    } else if (prb->dir == BWD_D) {
        check_kinds = {SRC};
    } else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI) {
        check_kinds = {WEI, BIA};
    } else {
        assert(!"unexpected!");
        SAFE_V(FAIL);
    }
    assert(!check_kinds.empty());
    return check_kinds;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(2); // regular + cpu_ref
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    // Use CPU prim as the reference in GPU testing to reduce testing time.
    SAFE(init_prim_ref(v_prim[1], prb), WARN);
    return OK;
}

int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        res_t *res) {
    SAFE(check_caches(v_prim[0], res), WARN);
    // Don't check caches for CPU prim as the reference.
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = v_prim[0];
    const auto &prim_ref = v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(mem_map, prb, prim, supported_exec_args(prb->dir));
    SAFE(init_ref_memory_args(
                 ref_mem_map, mem_map, prim, prb, res, prb->dir, prim_ref),
            WARN);

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (has_bench_mode_bit(mode_bit_t::corr)) {
        check_correctness(prb, get_kinds_to_check(prb), args, ref_args,
                setup_cmp, res, prim_ref);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace deconv
