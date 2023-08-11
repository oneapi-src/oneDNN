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

#include "utils/fill.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "conv/conv_dw_fusion.hpp"

namespace conv_dw_fusion {

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

    return std::unique_ptr<prb_t>(new prb_t((desc_t)*prb, prb->dir, prb->dt,
            prb->stag, prb->wtag, tag::any, prb->alg, attr, prb->ctx_init,
            prb->ctx_exe, prb->mb));
}

void get_fused_conv_dst_dims(const int ndims,
        const attr_t::post_ops_t::entry_t &po_entry, const dnnl_dims_t dims,
        dnnl_dims_t fused_conv_dst_dims) {
    const auto &conv_po = po_entry.convolution;
    const auto stride = conv_po.stride;

    for (int d = 0; d < ndims; ++d) {
        if (d < 2)
            fused_conv_dst_dims[d] = dims[d];
        else
            // Not following standard convolution formula for output shapes since
            // right/top padding might be greater than left/top one.
            fused_conv_dst_dims[d] = div_up(dims[d], stride);
    }
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

    std::vector<dnnl_data_type_t> dw_dt {
            prb->get_dt(DST), prb->get_dt(WEI), fused_conv_po.dst_dt};

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
    cd.kd = is_3d ? kernel : 1;
    cd.kh = is_2d ? kernel : 1;
    cd.kw = kernel;
    cd.sd = is_3d ? stride : 1;
    cd.sh = is_2d ? stride : 1;
    cd.sw = stride;
    cd.pd = is_3d ? padding : 0;
    cd.ph = is_2d ? padding : 0;
    cd.pw = padding;

    dnnl_dims_t fused_conv_dst_dims;
    get_fused_conv_dst_dims(prb->ndims, po.entry[fusion_index],
            prb->dst_dims().data(), fused_conv_dst_dims);
    cd.oc = prb->oc;
    cd.od = is_3d ? fused_conv_dst_dims[prb->ndims - 3] : 1;
    cd.oh = is_2d ? fused_conv_dst_dims[prb->ndims - 2] : 1;
    cd.ow = fused_conv_dst_dims[prb->ndims - 1];

    cd.has_groups = true;
    cd.ndims = prb->ndims;
    cd.init_pad_r();

    return std::unique_ptr<prb_t>(new prb_t(cd, prb->dir, dw_dt, tag::any,
            tag::any, prb->dtag, alg_t::DIRECT, fusion_attr, prb->ctx_init,
            prb->ctx_exe, prb->mb));
}

int init_ref_memory_args(dnn_mem_map_t &mem_map0, dnn_mem_map_t &mem_map1,
        dnn_mem_map_t &mem_map, dnnl_primitive_t prim0, const prb_t *prb0,
        const prb_t *prb1, const prb_t *prb, res_t *res, dir_t dir) {
    const auto &ref_engine = get_cpu_engine();

    const int dw_idx = prb->attr.post_ops.convolution_index();
    // Memory filling is the first one who uses updated problem alg.
    if (prb0->alg == conv::AUTO) prb0->alg = conv::DIRECT;

    // Move cfg out of filling since its creation is not free.
    cfg_t cfg(prb, {SRC, WEI, BIA, DST});

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        dnn_mem_t ref_mem(mem.md_, dnnl_f32, tag::abx, ref_engine);

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_data(SRC, prb0, cfg, mem, ref_mem, res), WARN);
                if (has_bench_mode_bit(mode_bit_t::corr))
                    SAFE(mem_map0.at(exec_arg).reorder(ref_mem), WARN);
                break;
            case DNNL_ARG_WEIGHTS:
                SAFE(fill_data(WEI, prb0, cfg, mem, ref_mem, res), WARN);
                if (has_bench_mode_bit(mode_bit_t::corr))
                    SAFE(mem_map0.at(exec_arg).reorder(ref_mem), WARN);
                break;
            case DNNL_ARG_BIAS:
                SAFE(fill_data(BIA, prb0, cfg, mem, ref_mem, res), WARN);
                if (ref_mem.ndims() > 0 && has_bench_mode_bit(mode_bit_t::corr))
                    SAFE(mem_map0.at(exec_arg).reorder(ref_mem), WARN);
                break;
            case (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS):
                SAFE(fill_data(WEI, prb1, cfg, mem, ref_mem, res), WARN);
                if (has_bench_mode_bit(mode_bit_t::corr))
                    SAFE(mem_map1.at(DNNL_ARG_WEIGHTS).reorder(ref_mem), WARN);
                break;
            case (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS):
                SAFE(fill_data(BIA, prb1, cfg, mem, ref_mem, res), WARN);
                if (ref_mem.ndims() > 0 && has_bench_mode_bit(mode_bit_t::corr))
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
                    SAFE(fill_scales(prb0->attr, local_exec_arg, mem, ref_mem),
                            WARN);
                    SAFE(mem_map0.at(exec_arg).reorder(mem), WARN);
                }
            } break;
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
        SAFE(fill_scales(prb1->attr, DNNL_ARG_WEIGHTS,
                     mem_map.at(dw_wei_scale_arg), mem_map1.at(wei_scale_arg)),
                WARN);
    }
    if (!prb1->attr.scales.get(DNNL_ARG_DST).is_def()) {
        // Scales after dw can't be queried, create them from scratch.
        int dst_scale_arg = DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST;
        int dw_dst_scale_arg = DNNL_ARG_ATTR_POST_OP_DW | dst_scale_arg;
        const auto &dst_scale_md = mem_map1.at(dst_scale_arg).md_;
        mem_map[dw_dst_scale_arg] = dnn_mem_t(dst_scale_md, get_test_engine());
        SAFE(fill_scales(prb1->attr, DNNL_ARG_DST, mem_map.at(dw_dst_scale_arg),
                     mem_map1.at(dst_scale_arg)),
                WARN);
    }

    // Don't keep reference memory if it is not used further.
    if (!has_bench_mode_bit(mode_bit_t::corr)) {
        mem_map0.clear();
        mem_map1.clear();
    }

    return OK;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(3); // fused + 2 unfused

    SAFE(init_prim(prb->ctx_init, v_prim[0], conv::init_pd, prb, res), WARN);

    // Fill first convolution
    std::unique_ptr<prb_t> prb0 = get_first_conv_prb(prb);
    if (!prb0) SAFE(FAIL, WARN);

    SAFE(init_prim(prb->ctx_init, v_prim[1], conv::init_pd, prb0.get(), res,
                 FLAG_FWD, nullptr,
                 /* is_service_prim = */ true),
            WARN);

    // Fill next convolution
    std::unique_ptr<prb_t> prb1 = get_fused_conv_prb(prb);
    if (!prb1) SAFE(FAIL, WARN);

    SAFE(init_prim(prb->ctx_init, v_prim[2], conv::init_pd, prb1.get(), res,
                 FLAG_FWD, nullptr,
                 /* is_service_prim = */ true),
            WARN);

    return OK;
}

int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    SAFE(check_caches(v_prim[0], prb, res), WARN);

    SAFE(check_caches(v_prim[1], prb, res), WARN);

    SAFE(check_caches(v_prim[2], prb, res), WARN);

    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = v_prim[0];
    const auto &prim0 = v_prim[1];
    const auto &prim1 = v_prim[2];

    dnn_mem_map_t mem_map;
    init_memory_args<prb_t>(
            mem_map, prb, prim, conv::supported_exec_args(prb->dir));

    // Fill first convolution
    std::unique_ptr<prb_t> prb0 = get_first_conv_prb(prb);
    if (!prb0) SAFE(FAIL, WARN);

    dnn_mem_map_t mem_map0;
    init_memory_args<prb_t>(
            mem_map0, prb0.get(), prim0, conv::supported_exec_args(prb->dir));

    // Fill next convolution
    std::unique_ptr<prb_t> prb1 = get_fused_conv_prb(prb);
    if (!prb1) SAFE(FAIL, WARN);

    dnn_mem_map_t mem_map1;
    init_memory_args<prb_t>(
            mem_map1, prb1.get(), prim1, conv::supported_exec_args(prb->dir));

    TIME_FILL(SAFE(init_ref_memory_args(mem_map0, mem_map1, mem_map, prim0,
                           prb0.get(), prb1.get(), prb, res, prb->dir),
            WARN));

    args_t args(mem_map), args0(mem_map0), args1(mem_map1);

    if (prb->dir & FLAG_FWD) {
        SAFE(execute_and_wait(prim, args, res), WARN);

        if (has_bench_mode_bit(mode_bit_t::corr)) {
            SAFE(execute_and_wait(prim0, args0), WARN);
            SAFE(mem_map1.at(DNNL_ARG_SRC).reorder(mem_map0.at(DNNL_ARG_DST)),
                    WARN);
            SAFE(execute_and_wait(prim1, args1), WARN);

            compare::compare_t cmp;
            cmp.set_data_kind(DST);
            // Used prb1 to avoid writing separate compare function. Compare
            // uses prb->dt which can be u8s8u8 while after fusion it may be
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
