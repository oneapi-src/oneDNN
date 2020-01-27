/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "dnnl_traits.hpp"
#include "math_utils.hpp"
#include "simple_q10n.hpp"
#include "type_helpers.hpp"

#include "ref_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using math::get_bias;
using math::saturate;

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type,
        data_type_t acc_type>
void ref_convolution_fwd_t<src_type, wei_type, dst_type,
        acc_type>::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const bool with_groups = pd()->with_groups();

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    const int OC = pd()->OC() / G;
    const int IC = pd()->IC() / G;
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();

    const int KSD = pd()->KSD();
    const int KSH = pd()->KSH();
    const int KSW = pd()->KSW();

    const int KDD = pd()->KDD() + 1;
    const int KDH = pd()->KDH() + 1;
    const int KDW = pd()->KDW() + 1;

    const int padFront = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const int ndims = pd()->desc()->src_desc.ndims;

    using namespace data_type;
    bool is_int_conv = utils::one_of(src_type, s32, s8, u8);

    auto maybe_oscale = [=](float &d, int g, int oc) {
        // scale_idx_mult = 1 for per_oc scales and 0, otherwise
        const int scale_idx_mult
                = pd()->attr()->output_scales_.mask_ == (1 << 1);
        const float *scales = pd()->attr()->output_scales_.scales_;
        d *= scales[(g * OC + oc) * scale_idx_mult];
    };

    auto maybe_postops = [=](float &d, dst_data_t dst) {
        // Sum and post ops:
        const post_ops_t &ops = pd()->attr()->post_ops_;
        for (int idx = 0; idx < ops.len_; ++idx) {
            const auto &e = ops.entry_[idx];
            if (e.kind == dnnl_sum)
                d += e.sum.scale * dst;
            else
                d = eltwises_[idx]->compute_scalar(d);
        }
    };

    auto ker = [=](int g, int mb, int oc, int od, int oh, int ow) {
        acc_data_t d = 0;
        for_(int ic = 0; ic < IC; ++ic)
        for_(int kd = 0; kd < KD; ++kd)
        for_(int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            const int id = od * KSD - padFront + kd * KDD;
            const int ih = oh * KSH - padT + kh * KDH;
            const int iw = ow * KSW - padL + kw * KDW;

            if (id < 0 || id >= ID) continue;
            if (ih < 0 || ih >= IH) continue;
            if (iw < 0 || iw >= IW) continue;

            if (ndims == 5)
                d += (acc_data_t)src[src_d.off(mb, g * IC + ic, id, ih, iw)]
                        * (with_groups ? weights[weights_d.off(
                                   g, oc, ic, kd, kh, kw)]
                                       : weights[weights_d.off(
                                               oc, ic, kd, kh, kw)]);
            else if (ndims == 4)
                d += (acc_data_t)src[src_d.off(mb, g * IC + ic, ih, iw)]
                        * (with_groups ? weights[weights_d.off(
                                   g, oc, ic, kh, kw)]
                                       : weights[weights_d.off(
                                               oc, ic, kh, kw)]);
            else if (ndims == 3)
                d += (acc_data_t)src[src_d.off(mb, g * IC + ic, iw)]
                        * (with_groups ? weights[weights_d.off(g, oc, ic, kw)]
                                       : weights[weights_d.off(oc, ic, kw)]);
            else
                assert(false);
        }
        return d;
    };

    // help compiler optimize the code
    // constants for plain layouts kernel
    const dnnl_dims_t &src_str = src_d.blocking_desc().strides;
    const dim_t src_ic_stride = src_str[1];
    const dim_t src_id_stride = (ndims == 5) ? src_str[2] : 0;
    const dim_t src_ih_stride = (ndims >= 4) ? src_str[ndims - 2] : 0;
    const dim_t src_iw_stride = (ndims >= 3) ? src_str[ndims - 1] : 0;
    const dnnl_dims_t &weights_str = weights_d.blocking_desc().strides;
    const int gr_shift = with_groups ? 1 : 0;
    const dim_t weights_ic_stride = weights_str[1 + gr_shift];
    const dim_t weights_kd_stride
            = (ndims == 5) ? weights_str[2 + gr_shift] : 0;
    const dim_t weights_kh_stride
            = (ndims >= 4) ? weights_str[ndims - 2 + gr_shift] : 0;
    const dim_t weights_kw_stride
            = (ndims >= 3) ? weights_str[ndims - 1 + gr_shift] : 0;

    auto ker_plain = [=](int g, int mb, int oc, int od, int oh, int ow) {
        assert(3 <= ndims && ndims <= 5);
        acc_data_t d = 0;
        const dim_t src_loc_off = (ndims == 5)
                ? src_d.off(mb, g * IC, 0, 0, 0)
                : (ndims == 4) ? src_d.off(mb, g * IC, 0, 0)
                               : (ndims == 3) ? src_d.off(mb, g * IC, 0) : 0;

        const dim_t weights_loc_off = (ndims == 5)
                ? (with_groups ? weights_d.off(g, oc, 0, 0, 0, 0)
                               : weights_d.off(oc, 0, 0, 0, 0))
                : (ndims == 4) ? (with_groups ? weights_d.off(g, oc, 0, 0, 0)
                                              : weights_d.off(oc, 0, 0, 0))
                               : (ndims == 3)
                                ? (with_groups ? weights_d.off(g, oc, 0, 0)
                                               : weights_d.off(oc, 0, 0))
                                : 0;

        const src_data_t *__restrict src_loc = src + src_loc_off;
        const wei_data_t *__restrict weights_loc = weights + weights_loc_off;

        if (IC > KW) {
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                const dim_t id = od * KSD - padFront + kd * KDD;
                const dim_t ih = oh * KSH - padT + kh * KDH;
                const dim_t iw = ow * KSW - padL + kw * KDW;
                if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0
                        || iw >= IW)
                    continue;
                for (int ic = 0; ic < IC; ++ic) {
                    const dim_t src_off = ic + id * src_id_stride
                            + ih * src_ih_stride + iw * src_iw_stride;
                    const dim_t weights_off = ic * weights_ic_stride
                            + kd * weights_kd_stride + kh * weights_kh_stride
                            + kw;
                    d += (acc_data_t)src_loc[src_off]
                            * weights_loc[weights_off];
                }
            }
        } else {
            for_(dim_t ic = 0; ic < IC; ++ic)
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                const dim_t id = od * KSD - padFront + kd * KDD;
                const dim_t ih = oh * KSH - padT + kh * KDH;
                const dim_t iw = ow * KSW - padL + kw * KDW;
                if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0
                        || iw >= IW)
                    continue;
                const dim_t src_off = ic + id * src_id_stride
                        + ih * src_ih_stride + iw * src_iw_stride;
                const dim_t weights_off = ic * weights_ic_stride
                        + kd * weights_kd_stride + kh * weights_kh_stride + kw;
                d += (acc_data_t)src_loc[src_off] * weights_loc[weights_off];
            }
        }
        return d;
    };

    parallel_nd(G, MB, OC, OD, OH, OW,
            [&](int g, int mb, int oc, int od, int oh, int ow) {
                float a = bias ? get_bias(bias, bias_d.off(g * OC + oc),
                                  pd()->desc()->bias_desc.data_type)
                               : 0;

                if (src_d.is_plain() && weights_d.is_plain()
                        && src_ic_stride == 1 && weights_kw_stride == 1)
                    a += ker_plain(g, mb, oc, od, oh, ow);
                else
                    a += ker(g, mb, oc, od, oh, ow);

                dim_t dst_off {0};
                if (ndims == 5)
                    dst_off = dst_d.off(mb, g * OC + oc, od, oh, ow);
                else if (ndims == 4)
                    dst_off = dst_d.off(mb, g * OC + oc, oh, ow);
                else if (ndims == 3)
                    dst_off = dst_d.off(mb, g * OC + oc, ow);
                else
                    assert(false);

                maybe_oscale(a, g, oc);
                maybe_postops(a, dst[dst_off]);

                if (is_int_conv)
                    dst[dst_off] = qz_a1b0<float, dst_data_t>()(a);
                else
                    dst[dst_off] = saturate<dst_data_t>(a);
            });
}

template <data_type_t diff_src_type, data_type_t wei_type,
        data_type_t diff_dst_type, data_type_t acc_type>
void ref_convolution_bwd_data_t<diff_src_type, wei_type, diff_dst_type,
        acc_type>::execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const bool with_groups = pd()->with_groups();

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    const int OC = pd()->OC() / G;
    const int IC = pd()->IC() / G;
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();

    const int KSD = pd()->KSD();
    const int KSH = pd()->KSH();
    const int KSW = pd()->KSW();

    const int KDD = pd()->KDD() + 1;
    const int KDH = pd()->KDH() + 1;
    const int KDW = pd()->KDW() + 1;

    const int padFront = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const int ndims = pd()->desc()->diff_src_desc.ndims;

    using namespace data_type;
    bool is_int_conv = utils::one_of(diff_dst_type, s32, s8, u8);

    auto maybe_oscale = [=](float &d, int g, int ic) {
        /* scale_idx_mult = 1 for per_oc scales and 0, otherwise */
        const int scale_idx_mult
                = pd()->attr()->output_scales_.mask_ == (1 << 1);
        const float *scales = pd()->attr()->output_scales_.scales_;
        d *= scales[(g * OC + ic) * scale_idx_mult];
    };

    auto ker = [=](int g, int mb, int ic, int id, int ih, int iw) {
        acc_data_t d = 0;
        for_(int oc = 0; oc < OC; ++oc)
        for_(int kd = 0; kd < KD; ++kd)
        for_(int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            if (iw + padL < kw * KDW || ih + padT < kh * KDH
                    || id + padFront < kd * KDD)
                continue;
            int ow = iw - kw * KDW + padL;
            int oh = ih - kh * KDH + padT;
            int od = id - kd * KDD + padFront;
            if (ow % KSW != 0 || oh % KSH != 0 || od % KSD != 0) continue;

            ow /= KSW;
            oh /= KSH;
            od /= KSD;

            if (od < OD && oh < OH && ow < OW) {
                if (ndims == 5)
                    d += (acc_data_t)diff_dst[diff_dst_d.off(
                                 mb, g * OC + oc, od, oh, ow)]
                            * (with_groups ? weights[weights_d.off(
                                       g, oc, ic, kd, kh, kw)]
                                           : weights[weights_d.off(
                                                   oc, ic, kd, kh, kw)]);
                else if (ndims == 4)
                    d += (acc_data_t)diff_dst[diff_dst_d.off(
                                 mb, g * OC + oc, oh, ow)]
                            * (with_groups ? weights[weights_d.off(
                                       g, oc, ic, kh, kw)]
                                           : weights[weights_d.off(
                                                   oc, ic, kh, kw)]);
                else if (ndims == 3)
                    d += (acc_data_t)diff_dst[diff_dst_d.off(
                                 mb, g * OC + oc, ow)]
                            * (with_groups ? weights[weights_d.off(
                                       g, oc, ic, kw)]
                                           : weights[weights_d.off(
                                                   oc, ic, kw)]);
                else
                    assert(false);
            }
        }
        return d;
    };

    // help compiler optimize the code
    // constants for plain layouts kernel
    const dnnl_dims_t &diff_dst_str = diff_dst_d.blocking_desc().strides;
    const dim_t diff_dst_oc_stride = diff_dst_str[1];
    const dim_t diff_dst_ow_stride = diff_dst_str[ndims - 1];
    const dim_t diff_dst_oh_stride = (ndims >= 4) ? diff_dst_str[ndims - 2] : 0;
    const dim_t diff_dst_od_stride = (ndims >= 5) ? diff_dst_str[ndims - 3] : 0;

    const dnnl_dims_t &weights_str = weights_d.blocking_desc().strides;
    const int gr_shift = with_groups ? 1 : 0;
    const dim_t weights_oc_stride = weights_str[0 + gr_shift];
    const dim_t weights_kw_stride = weights_str[ndims - 1 + gr_shift];
    const dim_t weights_kh_stride
            = (ndims >= 4) ? weights_str[ndims - 2 + gr_shift] : 0;
    const dim_t weights_kd_stride
            = (ndims >= 4) ? weights_str[ndims - 3 + gr_shift] : 0;

    auto ker_plain = [=](int g, int mb, int ic, int id, int ih, int iw) {
        assert(3 <= ndims && ndims <= 5);
        acc_data_t d = 0;
        const dim_t diff_dst_loc_off = (ndims == 5)
                ? diff_dst_d.off(mb, g * OC, 0, 0, 0)
                : (ndims == 4)
                        ? diff_dst_d.off(mb, g * OC, 0, 0)
                        : (ndims == 3) ? diff_dst_d.off(mb, g * OC, 0) : 0;
        const dim_t weights_loc_off = (ndims == 5)
                ? with_groups ? weights_d.off(g, 0, ic, 0, 0, 0)
                              : weights_d.off(0, ic, 0, 0, 0)
                : (ndims == 4) ? with_groups ? weights_d.off(g, 0, ic, 0, 0)
                                             : weights_d.off(0, ic, 0, 0)
                               : (ndims == 3) ? with_groups
                                        ? weights_d.off(g, 0, ic, 0)
                                        : weights_d.off(0, ic, 0)
                                              : 0;

        const diff_dst_data_t *__restrict diff_dst_loc
                = diff_dst + diff_dst_loc_off;
        const wei_data_t *__restrict weights_loc = weights + weights_loc_off;

        if (OC > KW) {
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                dim_t ow = iw - kw * KDW + padL;
                dim_t oh = ih - kh * KDH + padT;
                dim_t od = id - kd * KDD + padFront;
                if (ow < 0 || oh < 0 || od < 0 || ow % KSW != 0 || oh % KSH != 0
                        || od % KSD != 0)
                    continue;
                ow /= KSW;
                oh /= KSH;
                od /= KSD;
                if (od >= OD || oh >= OH || ow >= OW) continue;
                for (dim_t oc = 0; oc < OC; ++oc) {
                    const dim_t diff_dst_off = oc + od * diff_dst_od_stride
                            + oh * diff_dst_oh_stride + ow * diff_dst_ow_stride;
                    const dim_t weights_off = oc * weights_oc_stride
                            + kd * weights_kd_stride + kh * weights_kh_stride
                            + kw;
                    d += (acc_data_t)diff_dst_loc[diff_dst_off]
                            * weights_loc[weights_off];
                }
            }
        } else {
            for_(dim_t oc = 0; oc < OC; ++oc)
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                dim_t ow = iw - kw * KDW + padL;
                dim_t oh = ih - kh * KDH + padT;
                dim_t od = id - kd * KDD + padFront;
                if (ow < 0 || oh < 0 || od < 0 || ow % KSW != 0 || oh % KSH != 0
                        || od % KSD != 0)
                    continue;
                ow /= KSW;
                oh /= KSH;
                od /= KSD;
                if (od >= OD || oh >= OH || ow >= OW) continue;
                const dim_t diff_dst_off = oc + od * diff_dst_od_stride
                        + oh * diff_dst_oh_stride + ow * diff_dst_ow_stride;
                const dim_t weights_off = oc * weights_oc_stride
                        + kd * weights_kd_stride + kh * weights_kh_stride + kw;
                d += (acc_data_t)diff_dst_loc[diff_dst_off]
                        * weights_loc[weights_off];
            }
        }
        return d;
    };

    parallel_nd(G, MB, IC, ID, IH, IW,
            [&](int g, int mb, int ic, int id, int ih, int iw) {
                auto ds_idx = (ndims == 5)
                        ? diff_src_d.off(mb, g * IC + ic, id, ih, iw)
                        : (ndims == 4) ? diff_src_d.off(mb, g * IC + ic, ih, iw)
                                       : diff_src_d.off(mb, g * IC + ic, iw);
                float a = bias ? get_bias(bias, bias_d.off(g * IC + ic),
                                  pd()->desc()->bias_desc.data_type)
                               : 0;

                if (diff_dst_d.is_plain() && weights_d.is_plain()
                        && diff_dst_oc_stride == 1 && weights_kw_stride == 1)
                    a += ker_plain(g, mb, ic, id, ih, iw);
                else
                    a += ker(g, mb, ic, id, ih, iw);
                maybe_oscale(a, g, ic);
                if (is_int_conv)
                    diff_src[ds_idx] = round_and_saturate<diff_src_data_t>(a);
                else
                    diff_src[ds_idx] = saturate<diff_src_data_t>(a);
            });
}

template <data_type_t src_type, data_type_t diff_wei_type,
        data_type_t diff_dst_type, data_type_t acc_type>
void ref_convolution_bwd_weights_t<src_type, diff_wei_type, diff_dst_type,
        acc_type>::execute_backward_weights(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(diff_wei_data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(diff_wei_data_t *, DNNL_ARG_DIFF_BIAS);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_md(1));

    const bool with_groups = pd()->with_groups();

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    const int OC = pd()->OC() / G;
    const int IC = pd()->IC() / G;
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();

    const int KSD = pd()->KSD();
    const int KSH = pd()->KSH();
    const int KSW = pd()->KSW();

    const int KDD = pd()->KDD() + 1;
    const int KDH = pd()->KDH() + 1;
    const int KDW = pd()->KDW() + 1;

    const int padFront = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const int ndims = pd()->desc()->src_desc.ndims;

    using namespace data_type;
    bool is_int_conv = utils::one_of(src_type, s32, s8, u8);

    auto ker = [=](acc_data_t &d, int g, int oc, int ic, int kd, int kh,
                       int kw) {
        for_(int mb = 0; mb < MB; ++mb)
        for_(int od = 0; od < OD; ++od)
        for_(int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
            if (ow * KSW + kw * KDW < padL || oh * KSH + kh * KDH < padT
                    || od * KSD + kd * KDD < padFront
                    || ow * KSW + kw * KDW >= IW + padL
                    || oh * KSH + kh * KDH >= IH + padT
                    || od * KSD + kd * KDD >= ID + padFront)
                continue;

            int id = od * KSD - padFront + kd * KDD;
            int ih = oh * KSH - padT + kh * KDH;
            int iw = ow * KSW - padL + kw * KDW;
            if (ndims == 5)
                d += (acc_data_t)diff_dst[diff_dst_d.off(
                             mb, g * OC + oc, od, oh, ow)]
                        * src[src_d.off(mb, g * IC + ic, id, ih, iw)];
            else if (ndims == 4)
                d += (acc_data_t)diff_dst[diff_dst_d.off(
                             mb, g * OC + oc, oh, ow)]
                        * src[src_d.off(mb, g * IC + ic, ih, iw)];
            else if (ndims == 3)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g * OC + oc, ow)]
                        * src[src_d.off(mb, g * IC + ic, iw)];
            else
                assert(false);
        }
    };

    auto ker_plain = [=](acc_data_t &d, int g, int oc, int ic, int kd, int kh,
                             int kw) {
        assert(3 <= ndims && ndims <= 5);
        // help compiler optimize the code
        // constants for plain layouts kernel
        const dnnl_dims_t &diff_dst_str = diff_dst_d.blocking_desc().strides;
        const dim_t diff_dst_mb_stride = diff_dst_str[0];
        const dim_t diff_dst_ow_stride = diff_dst_str[ndims - 1];
        const dim_t diff_dst_oh_stride
                = (ndims >= 4) ? diff_dst_str[ndims - 2] : 0;
        const dim_t diff_dst_od_stride
                = (ndims >= 5) ? diff_dst_str[ndims - 3] : 0;
        const dnnl_dims_t &src_str = src_d.blocking_desc().strides;
        const dim_t src_mb_stride = src_str[0];
        const dim_t src_iw_stride = src_str[ndims - 1];
        const dim_t src_ih_stride = (ndims >= 4) ? src_str[ndims - 2] : 0;
        const dim_t src_id_stride = (ndims >= 5) ? src_str[ndims - 3] : 0;

        const dim_t diff_dst_loc_off = (ndims == 5)
                ? diff_dst_d.off(0, g * OC + oc, 0, 0, 0)
                : (ndims == 4)
                        ? diff_dst_d.off(0, g * OC + oc, 0, 0)
                        : (ndims == 3) ? diff_dst_d.off(0, g * OC + oc, 0) : 0;

        const dim_t src_loc_off = (ndims == 5)
                ? src_d.off(0, g * IC + ic, 0, 0, 0)
                : (ndims == 4)
                        ? src_d.off(0, g * IC + ic, 0, 0)
                        : (ndims == 3) ? src_d.off(0, g * IC + ic, 0) : 0;

        const diff_dst_data_t *__restrict diff_dst_loc
                = diff_dst + diff_dst_loc_off;
        const src_data_t *__restrict src_loc = src + src_loc_off;

        for_(dim_t mb = 0; mb < MB; ++mb)
        for_(dim_t od = 0; od < OD; ++od)
        for_(dim_t oh = 0; oh < OH; ++oh)
        for (dim_t ow = 0; ow < OW; ++ow) {
            const dim_t id = od * KSD - padFront + kd * KDD;
            const dim_t ih = oh * KSH - padT + kh * KDH;
            const dim_t iw = ow * KSW - padL + kw * KDW;
            if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                continue;
            const dim_t diff_dst_off = mb * diff_dst_mb_stride
                    + od * diff_dst_od_stride + oh * diff_dst_oh_stride
                    + ow * diff_dst_ow_stride;
            const dim_t src_off = mb * src_mb_stride + id * src_id_stride
                    + ih * src_ih_stride + iw * src_iw_stride;
            d += (acc_data_t)diff_dst_loc[diff_dst_off] * src_loc[src_off];
        }
    };

    auto ker_bias = [=](acc_data_t &d, int g, int oc) {
        for_(int mb = 0; mb < MB; ++mb)
        for_(int od = 0; od < OD; ++od)
        for_(int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
            if (ndims == 5)
                d += (acc_data_t)
                        diff_dst[diff_dst_d.off(mb, g * OC + oc, od, oh, ow)];
            else if (ndims == 4)
                d += (acc_data_t)
                        diff_dst[diff_dst_d.off(mb, g * OC + oc, oh, ow)];
            else if (ndims == 3)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g * OC + oc, ow)];
            else
                assert(false);
        }
    };

    parallel_nd(G, OC, [&](int g, int oc) {
        if (diff_bias) {
            // XXX: loss of precision when bias is a float...
            acc_data_t db = 0;
            ker_bias(db, g, oc);
            if (is_int_conv)
                diff_bias[diff_bias_d.off(g * OC + oc)]
                        = round_and_saturate<diff_wei_data_t>(db);
            else
                diff_bias[diff_bias_d.off(g * OC + oc)]
                        = saturate<diff_wei_data_t>(db);
        }

        for_(int ic = 0; ic < IC; ++ic)
        for_(int kd = 0; kd < KD; ++kd)
        for_(int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            acc_data_t dw = 0;
            if (diff_dst_d.is_plain() && src_d.is_plain())
                ker_plain(dw, g, oc, ic, kd, kh, kw);
            else
                ker(dw, g, oc, ic, kd, kh, kw);

            dim_t idx {0};
            if (ndims == 5)
                idx = with_groups ? diff_weights_d.off(g, oc, ic, kd, kh, kw)
                                  : diff_weights_d.off(oc, ic, kd, kh, kw);
            else if (ndims == 4)
                idx = with_groups ? diff_weights_d.off(g, oc, ic, kh, kw)
                                  : diff_weights_d.off(oc, ic, kh, kw);
            else if (ndims == 3)
                idx = with_groups ? diff_weights_d.off(g, oc, ic, kw)
                                  : diff_weights_d.off(oc, ic, kw);
            else
                assert(false);
            if (is_int_conv)
                diff_weights[idx] = round_and_saturate<diff_wei_data_t>(dw);
            else
                diff_weights[idx] = saturate<diff_wei_data_t>(dw);
        }
    });
}

using namespace data_type;

template struct ref_convolution_fwd_t<f32>;

template struct ref_convolution_fwd_t<u8, s8, f32, s32>;
template struct ref_convolution_fwd_t<u8, s8, s32, s32>;
template struct ref_convolution_fwd_t<u8, s8, s8, s32>;
template struct ref_convolution_fwd_t<u8, s8, u8, s32>;

template struct ref_convolution_bwd_data_t<f32, f32, f32, f32>;

template struct ref_convolution_bwd_data_t<f32, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<s32, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<s8, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<u8, s8, u8, s32>;

template struct ref_convolution_bwd_weights_t<f32, f32, f32, f32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
