/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include <utility>

#include "utils/parallel.hpp"

#include "deconv/ref_deconv.hpp"

namespace deconv {

void compute_ref_direct_fwd(const prb_t *prb, const args_t &args) {
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
    const dnn_mem_t &src_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const dnn_mem_t &dst_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_scale = !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def();
    const bool has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    assert(IMPLICATION(has_src_scale, src_scales.nelems() == 1));
    assert(IMPLICATION(has_dst_scale, dst_scales.nelems() == 1));
    float src_scale = has_src_scale ? src_scales.get_elem(0) : 1.f;
    float dst_scale = has_dst_scale ? 1.f / dst_scales.get_elem(0) : 1.f;
    const int wei_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_WEIGHTS, dnnl_convolution, prb->has_groups);

    const bool has_src_zp = !prb->attr.zero_points.get(DNNL_ARG_SRC).is_def();
    const bool has_dst_zp = !prb->attr.zero_points.get(DNNL_ARG_DST).is_def();
    const int src_zp_mask = attr_t::get_default_mask(
            prb->attr.zero_points.get(DNNL_ARG_SRC).policy);
    const int dst_zp_mask = attr_t::get_default_mask(
            prb->attr.zero_points.get(DNNL_ARG_DST).policy);

    /* help compiler optimize the code */
    const int64_t MB = prb->mb, G = prb->g, OC = prb->oc, IC = prb->ic;
    const int64_t OCG = OC / G, ICG = IC / G;
    const int64_t OD = prb->od, OH = prb->oh, OW = prb->ow;
    const int64_t ID = prb->id, IH = prb->ih, IW = prb->iw;
    const int64_t SD = prb->sd, SH = prb->sh, SW = prb->sw;
    const int64_t PD = prb->pd, PH = prb->ph, PW = prb->pw;
    const int64_t KD = prb->kd, KH = prb->kh, KW = prb->kw;
    const int64_t DD = prb->dd + 1;
    const int64_t DH = prb->dh + 1;
    const int64_t DW = prb->dw + 1;

    auto ker = [&](float &d, int64_t g, int64_t mb, int64_t oc, int64_t od,
                       int64_t oh, int64_t ow) {
        const float *__restrict src_loc
                = (const float *)src_m + (mb * IC + g * ICG) * ID * IH * IW;
        const float *__restrict wei_loc
                = (const float *)wei_m + (g * OCG + oc) * ICG * KD * KH * KW;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t id = od * SD - PD + kd * DD;
            if (id < 0 || id >= ID) continue;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH - PH + kh * DH;
                if (ih < 0 || ih >= IH) continue;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW - PW + kw * DW;
                    if (iw < 0 || iw >= IW) continue;

                    for (int64_t ic = 0; ic < ICG; ++ic) {
                        int64_t src_off = ((ic * ID + id) * IH + ih) * IW + iw;
                        int64_t wei_off = ((ic * KD + kd) * KH + kh) * KW + kw;
                        int src_zp = has_src_zp ? src_zps.get_elem(
                                             src_zp_mask > 0 ? g * ICG + ic : 0)
                                                : 0;
                        float s = (src_loc[src_off] - src_zp) * src_scale;
                        float wei_scale = 1.f;
                        if (has_wei_scale)
                            wei_scale = wei_scales.get_elem(
                                    wei_scale_mask > 0 ? g * OCG + oc : 0);
                        float w = wei_loc[wei_off] * wei_scale;
                        d += s * w;
                    }
                }
            }
        }
    };

    auto v_po_masks = prb->attr.post_ops.get_po_masks();
    benchdnn_parallel_nd(G, MB, OCG, OD, OH, OW,
            [&](int64_t g, int64_t mb, int64_t oc, int64_t od, int64_t oh,
                    int64_t ow) {
                const size_t dst_off = dst_off_f(prb, mb, g, oc, od, oh, ow);
                float &dst = ((float *)dst_m)[dst_off];

                float conv_res = 0;
                ker(conv_res, g, mb, oc, od, oh, ow);

                if (prb->dir & FLAG_BIA) {
                    const size_t bia_off = bia_off_f(prb, g, oc);
                    conv_res += ((float *)bia_m)[bia_off];
                }

                const auto v_po_vals
                        = prepare_po_vals(dst_m, args, v_po_masks, dst_off);

                maybe_post_ops(prb->attr, conv_res, dst, v_po_vals);

                int dst_zp = has_dst_zp
                        ? dst_zps.get_elem(dst_zp_mask > 0 ? g * OCG + oc : 0)
                        : 0;
                dst = conv_res * dst_scale + dst_zp;
            });
}

void compute_ref_direct_bwd_d(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &diff_src_m = args.find(DNNL_ARG_DIFF_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &bia_m = args.find(DNNL_ARG_BIAS);
    const dnn_mem_t &diff_dst_m = args.find(DNNL_ARG_DIFF_DST);
    const dnn_mem_t &src_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const dnn_mem_t &wei_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    const dnn_mem_t &src_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const dnn_mem_t &dst_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_scale = !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def();
    const bool has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    assert(IMPLICATION(has_src_scale, src_scales.nelems() == 1));
    assert(IMPLICATION(has_dst_scale, dst_scales.nelems() == 1));
    float src_scale = has_src_scale ? src_scales.get_elem(0) : 1.f;
    float dst_scale = has_dst_scale ? 1.f / dst_scales.get_elem(0) : 1.f;
    const int wei_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_WEIGHTS, dnnl_convolution, prb->has_groups);

    const bool has_src_zp = !prb->attr.zero_points.get(DNNL_ARG_SRC).is_def();
    const bool has_dst_zp = !prb->attr.zero_points.get(DNNL_ARG_DST).is_def();
    const int src_zp_mask = attr_t::get_default_mask(
            prb->attr.zero_points.get(DNNL_ARG_SRC).policy);
    const int dst_zp_mask = attr_t::get_default_mask(
            prb->attr.zero_points.get(DNNL_ARG_DST).policy);

    /* help compiler optimize the code */
    const int64_t MB = prb->mb, G = prb->g, OC = prb->oc, IC = prb->ic;
    const int64_t OCG = OC / G, ICG = IC / G;
    const int64_t OD = prb->od, OH = prb->oh, OW = prb->ow;
    const int64_t ID = prb->id, IH = prb->ih, IW = prb->iw;
    const int64_t SD = prb->sd, SH = prb->sh, SW = prb->sw;
    const int64_t PD = prb->pd, PH = prb->ph, PW = prb->pw;
    const int64_t KD = prb->kd, KH = prb->kh, KW = prb->kw;
    const int64_t DD = prb->dd + 1;
    const int64_t DH = prb->dh + 1;
    const int64_t DW = prb->dw + 1;

    enum { precompute_size = 16 };
    const bool fast = MAX3(KD, KH, KW) <= precompute_size;

    /* pre-computes arrays of oh(ow) and kh(kw) for traversing in kernel */
    auto precompute_ok
            = [](int64_t i, int64_t O, int64_t K, int64_t S, int64_t P,
                      int64_t D, int64_t &num, int64_t *_o, int64_t *_k) {
                  assert(K <= precompute_size);
                  num = 0;
                  for (int64_t k = 0; k < K; ++k) {
                      int64_t o = i - k * D + P;
                      if (o < 0 || o % S) continue;
                      o /= S;
                      if (o >= O) continue;
                      _k[num] = k;
                      _o[num] = o;
                      ++num;
                  }
              };

    auto ker_fast = [&](float &ds, int64_t g, int64_t mb, int64_t ic,
                            int64_t id, int64_t ih, int64_t iw) {
        int64_t kd[precompute_size], od[precompute_size], num_d;
        int64_t kh[precompute_size], oh[precompute_size], num_h;
        int64_t kw[precompute_size], ow[precompute_size], num_w;
        precompute_ok(id, OD, KD, SD, PD, DD, num_d, od, kd);
        precompute_ok(ih, OH, KH, SH, PH, DH, num_h, oh, kh);
        precompute_ok(iw, OW, KW, SW, PW, DW, num_w, ow, kw);

        const float *__restrict diff_dst_loc = (const float *)diff_dst_m
                + (mb * OC + g * OCG) * OD * OH * OW;
        const float *__restrict wei_loc
                = (const float *)wei_m + ((g * OCG) * ICG + ic) * KD * KH * KW;

        for_(int64_t d = 0; d < num_d; ++d)
        for_(int64_t h = 0; h < num_h; ++h)
        for_(int64_t w = 0; w < num_w; ++w)
        for (int64_t oc = 0; oc < OCG; ++oc) {
            const int64_t diff_dst_off
                    = ((oc * OD + od[d]) * OH + oh[h]) * OW + ow[w];
            const int64_t wei_off
                    = ((oc * ICG * KD + kd[d]) * KH + kh[h]) * KW + kw[w];
            int src_zp = has_src_zp
                    ? src_zps.get_elem(src_zp_mask > 0 ? g * OCG + oc : 0)
                    : 0;
            float diff_dst_val
                    = (diff_dst_loc[diff_dst_off] - src_zp) * src_scale;

            float wei_scale = 1.f;
            if (has_wei_scale)
                wei_scale = wei_scales.get_elem(
                        wei_scale_mask > 0 ? g * ICG + ic : 0);
            float wei_val = wei_loc[wei_off] * wei_scale;
            ds += diff_dst_val * wei_val;
        }
    };

    auto ker = [&](float &ds, int64_t g, int64_t mb, int64_t ic, int64_t id,
                       int64_t ih, int64_t iw) {
        const float *__restrict diff_dst_loc = (const float *)diff_dst_m
                + (mb * OC + g * OCG) * OD * OH * OW;
        const float *__restrict wei_loc
                = (const float *)wei_m + ((g * OCG) * ICG + ic) * KD * KH * KW;

        for (int64_t kd = 0; kd < KD; ++kd) {
            int64_t od = id - kd * DD + PD;
            if (od < 0 || od % SD || od >= OD * SD) continue;
            od /= SD;
            for (int64_t kh = 0; kh < KH; ++kh) {
                int64_t oh = ih - kh * DH + PH;
                if (oh < 0 || oh % SH || oh >= OH * SH) continue;
                oh /= SH;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    int64_t ow = iw - kw * DW + PW;
                    if (ow < 0 || ow % SW || ow >= OW * SW) continue;
                    ow /= SW;
                    for (int64_t oc = 0; oc < OCG; ++oc) {
                        const int64_t diff_dst_off
                                = ((oc * OD + od) * OH + oh) * OW + ow;
                        const int64_t wei_off
                                = ((oc * ICG * KD + kd) * KH + kh) * KW + kw;
                        int src_zp = has_src_zp ? src_zps.get_elem(
                                             src_zp_mask > 0 ? g * OCG + oc : 0)
                                                : 0;
                        float diff_dst_val
                                = (diff_dst_loc[diff_dst_off] - src_zp)
                                * src_scale;

                        float wei_scale = 1.f;
                        if (has_wei_scale)
                            wei_scale = wei_scales.get_elem(
                                    wei_scale_mask > 0 ? g * ICG + ic : 0);
                        float wei_val = wei_loc[wei_off] * wei_scale;
                        ds += diff_dst_val * wei_val;
                    }
                }
            }
        }
    };

    auto v_po_masks = prb->attr.post_ops.get_po_masks();
    benchdnn_parallel_nd(G, MB, ICG, ID, IH, IW,
            [&](int64_t g, int64_t mb, int64_t ic, int64_t id, int64_t ih,
                    int64_t iw) {
                size_t src_off = src_off_f(prb, mb, g, ic, id, ih, iw);
                float &ds = ((float *)diff_src_m)[src_off];
                float conv_res = 0;
                if (fast)
                    ker_fast(conv_res, g, mb, ic, id, ih, iw);
                else
                    ker(conv_res, g, mb, ic, id, ih, iw);

                if (prb->dir & FLAG_BIA) {
                    const size_t bia_off = (size_t)g * ICG + ic;
                    conv_res += ((float *)bia_m)[bia_off];
                }

                const auto v_po_vals = prepare_po_vals(
                        diff_src_m, args, v_po_masks, src_off);

                maybe_post_ops(prb->attr, conv_res, ds, v_po_vals);

                int dst_zp = has_dst_zp
                        ? dst_zps.get_elem(dst_zp_mask > 0 ? g * ICG + ic : 0)
                        : 0;
                ds = conv_res * dst_scale + dst_zp;
            });
}

void compute_ref_bwd_weights(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &diff_wei_m = args.find(DNNL_ARG_DIFF_WEIGHTS);
    const dnn_mem_t &diff_dst_m = args.find(DNNL_ARG_DIFF_DST);
    /* help compiler optimize the code */
    const int64_t MB = prb->mb, G = prb->g, OC = prb->oc, IC = prb->ic;
    const int64_t OCG = OC / G, ICG = IC / G;
    const int64_t OD = prb->od, OH = prb->oh, OW = prb->ow;
    const int64_t ID = prb->id, IH = prb->ih, IW = prb->iw;
    const int64_t SD = prb->sd, SH = prb->sh, SW = prb->sw;
    const int64_t PD = prb->pd, PH = prb->ph, PW = prb->pw;
    const int64_t KD = prb->kd, KH = prb->kh, KW = prb->kw;
    const int64_t DD = prb->dd + 1;
    const int64_t DH = prb->dh + 1;
    const int64_t DW = prb->dw + 1;

    auto compute_bounds
            = [](int64_t I, int64_t O, int64_t k, int64_t S, int64_t P,
                      int64_t D, int64_t &o_s, int64_t &o_e) {
                  const float tmp = P - k * D;
                  o_s = MAX2(0, ceilf(tmp / S));
                  o_e = MIN2(O, ceilf((I + tmp) / S));
              };

    auto ker = [&](float &dw, int64_t g, int64_t oc, int64_t ic, int64_t kd,
                       int64_t kh, int64_t kw) {
        int64_t od_s, od_e, oh_s, oh_e, ow_s, ow_e;
        compute_bounds(ID, OD, kd, SD, PD, DD, od_s, od_e);
        compute_bounds(IH, OH, kh, SH, PH, DH, oh_s, oh_e);
        compute_bounds(IW, OW, kw, SW, PW, DW, ow_s, ow_e);
        const int64_t id_s = kd * DD - PD;
        const int64_t ih_s = kh * DH - PH;
        const int64_t iw_s = kw * DW - PW;

        for (int64_t mb = 0; mb < MB; ++mb) {
            const float *__restrict diff_dst_loc = (const float *)diff_dst_m
                    + (mb * OC + g * OCG + oc) * OD * OH * OW;
            const float *__restrict src_loc = (const float *)src_m
                    + (mb * IC + g * ICG + ic) * ID * IH * IW;

            for_(int64_t od = od_s; od < od_e; ++od)
            for_(int64_t oh = oh_s; oh < oh_e; ++oh)
            for (int64_t ow = ow_s; ow < ow_e; ++ow) {
                const int64_t id = od * SD + id_s;
                const int64_t ih = oh * SH + ih_s;
                const int64_t iw = ow * SW + iw_s;

                size_t diff_dst_off = (od * OH + oh) * OW + ow;
                size_t src_off = (id * IH + ih) * IW + iw;
                dw += diff_dst_loc[diff_dst_off] * src_loc[src_off];
            }
        }
    };

    benchdnn_parallel_nd(G, OCG, ICG, KD, KH, KW,
            [&](int64_t g, int64_t oc, int64_t ic, int64_t kd, int64_t kh,
                    int64_t kw) {
                size_t wei_off = wei_off_f(prb, g, oc, ic, kd, kh, kw);
                float &dw = ((float *)diff_wei_m)[wei_off];
                dw = 0;
                ker(dw, g, oc, ic, kd, kh, kw);
            });
}

void compute_ref_bwd_bias(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &diff_bia_m = args.find(DNNL_ARG_DIFF_BIAS);
    const dnn_mem_t &diff_dst_m = args.find(DNNL_ARG_DIFF_DST);
    /* help compiler optimize the code */
    const int64_t MB = prb->mb, G = prb->g, OC = prb->oc;
    const int64_t OCG = OC / G;
    const int64_t OD = prb->od, OH = prb->oh, OW = prb->ow;

    benchdnn_parallel_nd(G, OCG, [&](int64_t g, int64_t oc) {
        size_t bia_off = bia_off_f(prb, g, oc);
        double sum = 0;

        for_(int64_t mb = 0; mb < MB; ++mb)
        for_(int64_t od = 0; od < OD; ++od)
        for_(int64_t oh = 0; oh < OH; ++oh)
        for (int64_t ow = 0; ow < OW; ++ow) {
            size_t dst_off = dst_off_f(prb, mb, g, oc, od, oh, ow);
            sum += ((float *)diff_dst_m)[dst_off];
        }
        ((float *)diff_bia_m)[bia_off] = (float)sum;
    });
}

void compute_ref_fwd(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    // Swap arguments to re-use existing conv ref implementation.
    args_t ref_conv_args;
    for (int i = 0; i < args.size(); i++) {
        if (args.arg(i) == DNNL_ARG_SRC)
            ref_conv_args.set(DNNL_ARG_DIFF_DST, args.dnn_mem(i));
        else if (args.arg(i) == DNNL_ARG_WEIGHTS)
            ref_conv_args.set(DNNL_ARG_WEIGHTS, args.find(DNNL_ARG_WEIGHTS_1));
        else if (args.arg(i) == DNNL_ARG_DST)
            ref_conv_args.set(DNNL_ARG_DIFF_SRC, args.dnn_mem(i));
        else
            ref_conv_args.set(args.arg(i), args.dnn_mem(i));
    }

    if (prb->alg == WINO && prb->get_dt(SRC) == dnnl_f32) {
        compute_wino_ref_bwd_d(prb, ref_conv_args);
    } else {
        compute_ref_direct_bwd_d(prb, ref_conv_args);
    }
}

void compute_ref_bwd_d(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    // Swap arguments to re-use existing conv ref implementation.
    args_t ref_conv_args;
    for (int i = 0; i < args.size(); i++) {
        if (args.arg(i) == DNNL_ARG_DIFF_SRC)
            ref_conv_args.set(DNNL_ARG_DST, args.dnn_mem(i));
        else if (args.arg(i) == DNNL_ARG_WEIGHTS)
            ref_conv_args.set(DNNL_ARG_WEIGHTS, args.find(DNNL_ARG_WEIGHTS_1));
        else if (args.arg(i) == DNNL_ARG_DIFF_DST)
            ref_conv_args.set(DNNL_ARG_SRC, args.dnn_mem(i));
        else
            ref_conv_args.set(args.arg(i), args.dnn_mem(i));
    }

    if (prb->alg == WINO && prb->get_dt(SRC) == dnnl_f32) {
        compute_wino_ref_fwd(prb, ref_conv_args);
    } else {
        compute_ref_direct_fwd(prb, ref_conv_args);
    }
}

void compute_ref_bwd_w(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    // Swap arguments to re-use existing conv ref implementation.
    args_t ref_conv_args;
    for (int i = 0; i < args.size(); i++) {
        if (args.arg(i) == DNNL_ARG_SRC)
            ref_conv_args.set(DNNL_ARG_DIFF_DST, args.dnn_mem(i));
        else if (args.arg(i) == DNNL_ARG_DIFF_WEIGHTS)
            ref_conv_args.set(
                    DNNL_ARG_DIFF_WEIGHTS, args.find(DNNL_ARG_DIFF_WEIGHTS_1));
        else if (args.arg(i) == DNNL_ARG_DIFF_DST)
            ref_conv_args.set(DNNL_ARG_SRC, args.dnn_mem(i));
        else
            ref_conv_args.set(args.arg(i), args.dnn_mem(i));
    }

    if (prb->alg == WINO && prb->get_dt(SRC) == dnnl_f32) {
        compute_wino_ref_bwd_w(prb, ref_conv_args);
    } else {
        compute_ref_bwd_weights(prb, ref_conv_args);
    }

    // Need to transpose data in weights back for proper comparison. This step
    // is done here as it's not needed for fast-ref-gpu.
    transpose_data_wei(prb, args.find(DNNL_ARG_DIFF_WEIGHTS_1),
            args.find(DNNL_ARG_DIFF_WEIGHTS));

    // We don't reuse `compute_ref_bwd_bias` as it doesn't match arguments and
    // entry problem which is transposed - `p_tr`. Simpler to use the kernel
    // directly.
    // Take original memories, not `ref_conv_args`.
    if (prb->dir & FLAG_BIA) {
        const dnn_mem_t &diff_bia_m = args.find(DNNL_ARG_DIFF_BIAS);
        const dnn_mem_t &diff_dst_m = args.find(DNNL_ARG_DIFF_DST);
        /* help compiler optimize the code */
        const int64_t MB = prb->mb, G = prb->g;
        const int64_t OC = prb->ic; // prb.oc = p_tr.ic
        const int64_t OCG = OC / G;
        const int64_t OD = prb->id; // prb.od = p_tr.id
        const int64_t OH = prb->ih; // prb.oh = p_tr.ih
        const int64_t OW = prb->iw; // prb.ow = p_tr.iw

        benchdnn_parallel_nd(G, OCG, [&](int64_t g, int64_t oc) {
            size_t bia_off = g * OCG + oc;
            double sum = 0;

            for_(int64_t mb = 0; mb < MB; ++mb)
            for_(int64_t od = 0; od < OD; ++od)
            for_(int64_t oh = 0; oh < OH; ++oh)
            for (int64_t ow = 0; ow < OW; ++ow) {
                // src_off_f instead of dst_off_f due to inverse descriptor.
                size_t dst_off = src_off_f(prb, mb, g, oc, od, oh, ow);
                sum += ((float *)diff_dst_m)[dst_off];
            }
            ((float *)diff_bia_m)[bia_off] = (float)sum;
        });
    }
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    // Update prb descriptor to re-use convolution reference.
    prb_t prb_tr((desc_t)*prb, prb->dir, prb->dt, prb->stag, prb->wtag,
            prb->dtag, prb->alg, prb->attr, prb->ctx_init, prb->ctx_exe,
            prb->mb);
    std::swap(prb_tr.ic, prb_tr.oc);
    std::swap(prb_tr.ih, prb_tr.oh);
    std::swap(prb_tr.id, prb_tr.od);
    std::swap(prb_tr.iw, prb_tr.ow);

    if (prb->dir & FLAG_FWD)
        compute_ref_fwd(&prb_tr, args, prim_ref);
    else if (prb->dir == BWD_D)
        compute_ref_bwd_d(&prb_tr, args, prim_ref);
    else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI)
        compute_ref_bwd_w(&prb_tr, args, prim_ref);
}

} // namespace deconv
