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

#include "utils/parallel.hpp"

#include "ip/ip.hpp"

namespace ip {

void compute_ref_fwd_ip(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &bia_m = args.find(DNNL_ARG_BIAS);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);
    const dnn_mem_t &src_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const dnn_mem_t &wei_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_scale = !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def();
    const bool has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    assert(IMPLICATION(has_src_scale, src_scales.nelems() == 1));
    assert(IMPLICATION(has_dst_scale, dst_scales.nelems() == 1));
    float src_scale = has_src_scale ? src_scales.get_elem(0) : 1.f;
    float dst_scale = has_dst_scale ? 1.f / dst_scales.get_elem(0) : 1.f;
    const int wei_scale_mask
            = prb->attr.scales.get_mask(DNNL_ARG_WEIGHTS, dnnl_inner_product);

    int64_t M = prb->mb;
    int64_t N = prb->oc;
    int64_t K = prb->ic * prb->id * prb->ih * prb->iw;

    dnn_mem_t dst_tmp(dst_m.md_, dnnl_f32, tag::abx, dst_m.engine());

    gemm("C", "N", "T", M, N, K, 1.f, (float *)src_m, K, (float *)wei_m, K, 0.f,
            (float *)dst_tmp, N);

    auto v_po_masks = prb->attr.post_ops.get_po_masks();
    benchdnn_parallel_nd(prb->mb, prb->oc, [&](int64_t mb, int64_t oc) {
        size_t dst_off = dst_off_f(prb, mb, oc);
        float &dst = ((float *)dst_m)[dst_off];

        float d = ((float *)dst_tmp)[dst_off];

        float wei_scale = 1.f;
        if (has_wei_scale)
            wei_scale = wei_scales.get_elem(wei_scale_mask > 0 ? oc : 0);

        d *= src_scale * wei_scale;

        if (prb->dir & FLAG_BIA) {
            size_t bia_off = bia_off_f(prb, oc);
            d += ((float *)bia_m)[bia_off];
        }

        const auto v_po_vals
                = prepare_po_vals(dst_m, args, v_po_masks, dst_off);

        maybe_post_ops(prb->attr, d, dst, v_po_vals);

        dst = d * dst_scale;
    });
}

void compute_ref_bwd_d_ip(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &diff_src_m = args.find(DNNL_ARG_DIFF_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &diff_dst_m = args.find(DNNL_ARG_DIFF_DST);

    int64_t M = prb->mb;
    int64_t N = prb->ic * prb->id * prb->ih * prb->iw;
    int64_t K = prb->oc;

    gemm("C", "N", "N", M, N, K, 1.f, (float *)diff_dst_m, K, (float *)wei_m, N,
            0.f, (float *)diff_src_m, N);
}

void compute_ref_bwd_w_ip(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &diff_wei_m = args.find(DNNL_ARG_DIFF_WEIGHTS);
    const dnn_mem_t &diff_dst_m = args.find(DNNL_ARG_DIFF_DST);
    const dnn_mem_t &diff_bia_m = args.find(DNNL_ARG_DIFF_BIAS);

    int64_t M = prb->oc;
    int64_t N = prb->ic * prb->id * prb->ih * prb->iw;
    int64_t K = prb->mb;

    gemm("C", "T", "N", M, N, K, 1.f, (float *)diff_dst_m, M, (float *)src_m, N,
            0.f, (float *)diff_wei_m, N);

    if (!(prb->dir & FLAG_BIA)) return;

    benchdnn_parallel_nd(prb->oc, [&](int64_t oc) {
        size_t bia_off = bia_off_f(prb, oc);
        float &db = ((float *)diff_bia_m)[bia_off];
        db = 0;
        for (int64_t mb = 0; mb < prb->mb; ++mb) {
            size_t dst_off = dst_off_f(prb, mb, oc);
            db += ((float *)diff_dst_m)[dst_off];
        }
    });
}

void compute_ref_fwd(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    compute_ref_fwd_ip(prb, args);
}

void compute_ref_bwd_d(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    compute_ref_bwd_d_ip(prb, args);
}

void compute_ref_bwd_w(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    compute_ref_bwd_w_ip(prb, args);
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prb->dir & FLAG_FWD)
        compute_ref_fwd(prb, args, prim_ref);
    else if (prb->dir == BWD_D)
        compute_ref_bwd_d(prb, args, prim_ref);
    else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI)
        compute_ref_bwd_w(prb, args, prim_ref);
}

} // namespace ip
