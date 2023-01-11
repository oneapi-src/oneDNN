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

#include <iterator>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "conv/conv_dw_fusion.hpp"

namespace conv_dw_fusion {

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

    dnnl_alg_kind_t alg = dnnl_convolution_direct;
    if (prb->alg == alg_t::WINO) alg = dnnl_convolution_winograd;
    if (prb->alg == alg_t::AUTO) alg = dnnl_convolution_auto;

    attr_args_t attr_args;

    auto wei_scale = prb->attr.scales.get(DNNL_ARG_WEIGHTS);
    if (wei_scale.policy == policy_t::PER_OC) {
        auto wei_mask = prb->has_groups ? 3 : 1;
        attr_args.prepare_scales(prb->attr, DNNL_ARG_WEIGHTS, wei_mask);
    }
    const auto dw_bia_dt = prb->dir == FWD_B ? dnnl_f32 : dnnl_data_type_undef;
    attr_args.prepare_dw_post_op(prb->attr, prb->cfg[WEI].dt, dw_bia_dt);
    attr_args.prepare_post_ops_mds(
            prb->attr, prb->ndims, prb->dst_dims().data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            if (prb->dir != FWD_B) bia_d.reset(nullptr);
            DNN_SAFE_STATUS(dnnl_convolution_forward_primitive_desc_create(
                    &init_pd_args.pd, init_pd_args.engine,
                    prb->dir == FWD_I ? dnnl_forward_inference
                                      : dnnl_forward_training,
                    alg, src_d, wei_d, bia_d, dst_d, prb->strides().data(),
                    prb->dilations().data(), prb->padding().data(),
                    prb->padding_r().data(), dnnl_attr));
            break;
        case BWD_D:
            DNN_SAFE_STATUS(
                    dnnl_convolution_backward_data_primitive_desc_create(
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
                    dnnl_convolution_backward_weights_primitive_desc_create(
                            &init_pd_args.pd, init_pd_args.engine, alg, src_d,
                            wei_d, bia_d, dst_d, prb->strides().data(),
                            prb->dilations().data(), prb->padding().data(),
                            prb->padding_r().data(), init_pd_args.hint,
                            dnnl_attr));
            break;
        default: DNN_SAFE_STATUS(dnnl_invalid_arguments);
    }

    // TODO: add query in od fir accum type.
    //DNN_SAFE_STATUS(cd.accum_data_type == prb->cfg[ACC].dt
    //                ? dnnl_success
    //                : dnnl_unimplemented);
    return dnnl_success;
}

std::unique_ptr<prb_t> get_first_conv_prb(const prb_t *prb) {
    const auto &po = prb->attr.post_ops;
    int fusion_index = po.convolution_index();

    attr_t attr;
    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
        auto sc = prb->attr.scales.get(arg);
        if (!sc.is_def()) attr.scales.set(arg, sc);
    }

    for (int i = 0; i < fusion_index; ++i) {
        attr.post_ops.entry.push_back(prb->attr.post_ops.entry[i]);
    }

    return std::unique_ptr<prb_t>(new prb_t((desc_t)*prb, prb->dir, prb->cfg,
            prb->stag, prb->wtag, tag::any, prb->alg, attr, prb->ctx_init,
            prb->ctx_exe, prb->mb));
}

std::unique_ptr<prb_t> get_fused_conv_prb(const prb_t *prb) {
    const auto &po = prb->attr.post_ops;
    int fusion_index = po.convolution_index();
    if (fusion_index == -1) return nullptr;
    const auto &fused_conv_po = po.entry[fusion_index].convolution;

    attr_t fusion_attr;
    // dw_conv src_scale = 1x1_conv dst_scale
    if (!prb->attr.scales.get(DNNL_ARG_DST).is_def())
        fusion_attr.scales.set(
                DNNL_ARG_SRC, prb->attr.scales.get(DNNL_ARG_DST));
    if (!fused_conv_po.wei_scale.is_def())
        fusion_attr.scales.set(DNNL_ARG_WEIGHTS, fused_conv_po.wei_scale);
    if (!fused_conv_po.dst_scale.is_def())
        fusion_attr.scales.set(DNNL_ARG_DST, fused_conv_po.dst_scale);

    for (int i = fusion_index + 1; i < po.len(); ++i) {
        fusion_attr.post_ops.entry.push_back(prb->attr.post_ops.entry[i]);
    }

    const auto f32 = dnnl_f32;
    std::stringstream dw_cfg_ss;
    if (prb->cfg[DST].dt == f32 && prb->cfg[WEI].dt == f32
            && fused_conv_po.dst_dt == f32)
        dw_cfg_ss << prb->cfg[DST].dt; // f32 is a single name
    else // else have all three dt in cfg name
        dw_cfg_ss << prb->cfg[DST].dt << prb->cfg[WEI].dt
                  << fused_conv_po.dst_dt;
    auto p_dw_cfg = conv::str2cfg(dw_cfg_ss.str().c_str());

    const auto kernel = fused_conv_po.kernel;
    const auto stride = fused_conv_po.stride;
    const auto padding = fused_conv_po.padding;
    bool is_3d = prb->ndims >= 5;
    bool is_2d = prb->ndims >= 4;

    desc_t cd {0};
    cd.g = prb->oc;
    cd.mb = prb->mb;
    cd.ic = prb->oc;
    cd.id = is_3d ? prb->od : 1;
    cd.ih = is_2d ? prb->oh : 1;
    cd.iw = prb->ow;
    cd.oc = prb->oc;
    cd.kd = is_3d ? kernel : 1;
    cd.kh = is_2d ? kernel : 1;
    cd.kw = kernel;
    cd.sd = is_3d ? stride : 1;
    cd.sh = is_2d ? stride : 1;
    cd.sw = stride;
    cd.pd = is_3d ? padding : 0;
    cd.ph = is_2d ? padding : 0;
    cd.pw = padding;
    // Not following standard convolution formula for output shapes since
    // right/top padding might be greated than left/top one.
    cd.od = is_3d ? div_up(cd.id, stride) : 1;
    cd.oh = is_2d ? div_up(cd.ih, stride) : 1;
    cd.ow = div_up(cd.iw, stride);

    cd.has_groups = true;
    cd.ndims = prb->ndims;
    cd.init_pad_r();

    return std::unique_ptr<prb_t>(new prb_t(cd, prb->dir, p_dw_cfg, tag::any,
            tag::any, prb->dtag, alg_t::DIRECT, fusion_attr, prb->ctx_init,
            prb->ctx_exe, prb->mb));
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type(
            {prb->cfg[SRC].dt, prb->cfg[WEI].dt, prb->cfg[DST].dt}, prb->dir,
            res);
    skip_unimplemented_sum_po(prb->attr, res);

    // GPU does not support depthwise fusion
    if (is_gpu() && prb->attr.post_ops.convolution_index() != -1) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

std::vector<int> supported_fused_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_BIAS,
            DNNL_ARG_DST,
            DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS,
            DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS,
    };
    return exec_fwd_args;
};

int init_ref_memory_args(dnn_mem_map_t &mem_map0, dnn_mem_map_t &mem_map1,
        dnn_mem_map_t &mem_map, dnnl_primitive_t prim0, const prb_t *prb0,
        const prb_t *prb1, const prb_t *prb, res_t *res, dir_t dir) {
    const auto &ref_engine = get_cpu_engine();

    const int dw_idx = prb->attr.post_ops.convolution_index();
    // Memory filling is the first one who uses updated problem alg and cfg.
    if (prb0->alg == conv::AUTO) prb0->alg = conv::DIRECT;
    prb0->cfg = auto_cfg(prb0->alg, prb0->cfg);

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        dnn_mem_t ref_mem(mem.md_, dnnl_f32, tag::abx, ref_engine);

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_src(prb0, mem, ref_mem, res), WARN);
                SAFE(mem_map0.at(exec_arg).reorder(ref_mem), WARN);
                break;
            case DNNL_ARG_WEIGHTS:
                SAFE(fill_wei(prb0, mem, ref_mem, res), WARN);
                SAFE(mem_map0.at(exec_arg).reorder(ref_mem), WARN);
                break;
            case DNNL_ARG_BIAS:
                SAFE(fill_bia(prb0, mem, ref_mem, res), WARN);
                if (ref_mem.ndims() > 0)
                    SAFE(mem_map0.at(exec_arg).reorder(ref_mem), WARN);
                break;
            case (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS):
                SAFE(fill_wei(prb1, mem, ref_mem, res), WARN);
                SAFE(mem_map1.at(DNNL_ARG_WEIGHTS).reorder(ref_mem), WARN);
                break;
            case (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS):
                SAFE(fill_bia(prb1, mem, ref_mem, res), WARN);
                if (ref_mem.ndims() > 0)
                    SAFE(mem_map1.at(DNNL_ARG_BIAS).reorder(ref_mem), WARN);
                break;
            default: { // Process all attributes here
                int pre_dw_post_ops_range
                        = DNNL_ARG_ATTR_MULTIPLE_POST_OP(dw_idx)
                        - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
                int post_dw_post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
                        - DNNL_ARG_ATTR_MULTIPLE_POST_OP(dw_idx);
                bool is_pre_dw_post_ops_arg
                        = (exec_arg & pre_dw_post_ops_range);
                bool is_post_dw_post_ops_arg
                        = (exec_arg & post_dw_post_ops_range);
                bool is_pre_dw_scales_arg = (exec_arg & DNNL_ARG_ATTR_SCALES);
                bool is_post_dw_scales_arg = is_pre_dw_scales_arg
                        && (exec_arg & DNNL_ARG_ATTR_POST_OP_DW);

                if (is_pre_dw_post_ops_arg && !is_post_dw_post_ops_arg) {
                    if (exec_arg & DNNL_ARG_SRC_1) {
                        SAFE(binary::fill_mem(exec_arg, mem, ref_mem), WARN);
                        SAFE(mem_map0.at(exec_arg).reorder(ref_mem), WARN);
                    }
                } else if (is_pre_dw_scales_arg && !is_post_dw_scales_arg) {
                    int local_exec_arg = exec_arg ^ DNNL_ARG_ATTR_SCALES;
                    float *prb_ptr = nullptr;
                    switch (local_exec_arg) {
                        case DNNL_ARG_SRC: prb_ptr = prb0->src_scales; break;
                        case DNNL_ARG_WEIGHTS:
                            prb_ptr = prb0->wei_scales;
                            break;
                        case DNNL_ARG_DST: prb_ptr = prb0->dst_scales; break;
                        default: break;
                    }
                    // Fill library scales directly.
                    for (int64_t idx = 0; idx < mem.nelems(); ++idx) {
                        mem.set_elem(idx, prb_ptr[idx]);
                        mem_map0.at(exec_arg).set_elem(idx, prb_ptr[idx]);
                    }
                }
            } break;
        }
        // Don't keep reference memory if it is not used further.
        if (!is_bench_mode(CORR)) {
            mem_map0.clear();
            mem_map1.clear();
        }
    }

    // Copy binary post_ops from second conv and reverse engineer an index in
    // the original map for those.
    for (const auto &e : mem_map1) {
        int arg = e.first;
        int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
                - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
        bool is_post_ops_arg = (arg & post_ops_range);
        if (is_post_ops_arg && (arg & DNNL_ARG_SRC_1)) {
            int second_conv_idx = arg / DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE - 1;
            int orig_idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                                   second_conv_idx + dw_idx + 1)
                    | DNNL_ARG_SRC_1;
            SAFE(binary::fill_mem(
                         orig_idx, mem_map.at(orig_idx), mem_map1.at(arg)),
                    WARN);
        }
    }

    // Post DW scales handling.
    if (!prb0->attr.scales.get(DNNL_ARG_DST).is_def()) {
        for (int64_t idx = 0;
                idx < mem_map0.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST).nelems();
                ++idx) {
            float val = mem_map0.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)
                                .get_elem(idx);
            mem_map1.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC).set_elem(idx, val);
        }
    }
    if (!prb1->attr.scales.get(DNNL_ARG_WEIGHTS).is_def()) {
        // Scales after dw can't be queried, create them from scratch.
        int wei_scale_arg = DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS;
        int dw_wei_scale_arg = DNNL_ARG_ATTR_POST_OP_DW | wei_scale_arg;
        const auto &wei_scale_md = mem_map1.at(wei_scale_arg).md_;
        mem_map[dw_wei_scale_arg] = dnn_mem_t(wei_scale_md, get_test_engine());
        for (int64_t idx = 0; idx < mem_map.at(dw_wei_scale_arg).nelems();
                ++idx) {
            mem_map.at(dw_wei_scale_arg).set_elem(idx, prb1->wei_scales[idx]);
            mem_map1.at(wei_scale_arg).set_elem(idx, prb1->wei_scales[idx]);
        }
    }
    if (!prb1->attr.scales.get(DNNL_ARG_DST).is_def()) {
        // Scales after dw can't be queried, create them from scratch.
        int dst_scale_arg = DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST;
        int dw_dst_scale_arg = DNNL_ARG_ATTR_POST_OP_DW | dst_scale_arg;
        const auto &dst_scale_md = mem_map1.at(dst_scale_arg).md_;
        mem_map[dw_dst_scale_arg] = dnn_mem_t(dst_scale_md, get_test_engine());
        for (int64_t idx = 0; idx < mem_map.at(dw_dst_scale_arg).nelems();
                ++idx) {
            mem_map.at(dw_dst_scale_arg).set_elem(idx, prb1->dst_scales[idx]);
            mem_map1.at(dst_scale_arg).set_elem(idx, prb1->dst_scales[idx]);
        }
    }

    return OK;
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    conv_dw_fusion::skip_unimplemented_prb(prb, res);
    if (res->state == SKIPPED) return OK;

    // Original problem with fusion attributes
    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    dnn_mem_map_t mem_map;
    init_memory_args<prb_t>(
            mem_map, prb, prim, supported_fused_exec_args(prb->dir));

    // Fill first convolution
    std::unique_ptr<prb_t> prb0 = get_first_conv_prb(prb);

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim0;
    SAFE(init_prim(prb->ctx_init, prim0, init_pd, prb0.get(), res, FLAG_FWD,
                 nullptr,
                 /* is_service_prim = */ true),
            WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    dnn_mem_map_t mem_map0;
    init_memory_args<prb_t>(
            mem_map0, prb0.get(), prim0, conv::supported_exec_args(prb->dir));

    // Fill next convolution
    std::unique_ptr<prb_t> prb1 = get_fused_conv_prb(prb);
    if (!prb1) SAFE(FAIL, WARN);

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim1;
    SAFE(init_prim(prb->ctx_init, prim1, init_pd, prb1.get(), res, FLAG_FWD,
                 nullptr,
                 /* is_service_prim = */ true),
            WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    dnn_mem_map_t mem_map1;
    init_memory_args<prb_t>(
            mem_map1, prb1.get(), prim1, conv::supported_exec_args(prb->dir));

    SAFE(init_ref_memory_args(mem_map0, mem_map1, mem_map, prim0, prb0.get(),
                 prb1.get(), prb, res, prb->dir),
            WARN);

    args_t args(mem_map), args0(mem_map0), args1(mem_map1);

    if (prb->dir & FLAG_FWD) {
        SAFE(execute_and_wait(prim, args, res), WARN);

        if (is_bench_mode(CORR)) {
            SAFE(execute_and_wait(prim0, args0), WARN);
            SAFE(mem_map1.at(DNNL_ARG_SRC).reorder(mem_map0.at(DNNL_ARG_DST)),
                    WARN);
            SAFE(execute_and_wait(prim1, args1), WARN);

            compare::compare_t cmp;
            cmp.set_data_kind(DST);
            // Used prb1 to avoid writing separate compare function. Compare
            // uses prb->cfg which can be u8s8u8 while after fusion it may be
            // u8s8s8, thus, compare() will saturate values which is not correct
            conv::setup_cmp(cmp, prb1.get(), DST, args1);

            dnn_mem_t dst_fused(mem_map.at(DNNL_ARG_DST), dnnl_f32, tag::abx,
                    get_test_engine());
            dnn_mem_t dst_unfused(mem_map1.at(DNNL_ARG_DST), dnnl_f32, tag::abx,
                    get_test_engine());

            cmp.compare(dst_unfused, dst_fused, prb->attr, res);
        }
    } else {
        assert(!"Backward is not supported");
        SAFE(FAIL, CRIT);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace conv_dw_fusion
