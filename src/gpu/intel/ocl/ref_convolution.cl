/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "gpu/intel/ocl/dispatch.h"
#include "gpu/intel/ocl/ocl_post_ops.h"
#include "gpu/intel/ocl/ocl_types.h"

#undef SRC_OFF
#undef WEI_OFF
#undef DST_OFF

#define SRC_OFF CONV_SRC_OFF
#define WEI_OFF CONV_WEI_OFF
#define DST_OFF CONV_DST_OFF

#if IS_FWD
KERNEL_ATTR
__kernel void ref_convolution_fwd(const __global SRC_DATA_T *src,
        const __global WEI_DATA_T *wei, const __global BIA_DATA_T *bias,
        __global DST_DATA_T *dst POST_OP_ARGS, const __global float *src_scales,
        const __global float *wei_scales, const __global float *dst_scales,
        const __global int *src_zpoints, const __global int *dst_zpoints) {

    src += SRC_OFFSET0;
    dst += DST_OFFSET0;

    const int n = GWS_GET_MB();
    const int oc = GWS_GET_OC();
    const int g = GWS_GET_G();
    const int od = GWS_GET_OD();
    const int oh = GWS_GET_OH();
    const int ow = GWS_GET_OW();

    ACC_DATA_T d = 0;
    for (int ic = 0; ic < IC; ++ic)
        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {
                    const int id = od * SD - PD + kd * (1 + DD);
                    const int ih = oh * SH - PH + kh * (1 + DH);
                    const int iw = ow * SW - PW + kw * (1 + DW);

                    if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0
                            || iw >= IW)
                        continue;

                    const uint src_off = SRC_OFF(n, g * IC + ic, id, ih, iw);
                    const uint wei_off = WEI_OFF(g, oc, ic, kd, kh, kw);
                    d += SRC_TO_REF(src[src_off]) * WEI_TO_REF(wei[wei_off]);
#if WITH_SRC_ZPOINTS
                    const int src_zp
                            = src_zpoints[WITH_SRC_ZPOINTS_PER_IC ? g * IC + ic
                                                                  : 0];
                    d -= src_zp * WEI_TO_REF(wei[wei_off]);
#endif
                }
    POST_OP_DATA_T tmp = d;

#if WITH_SRC_SCALES
    tmp *= src_scales[0];
#endif
#if WITH_WEI_SCALES
#if WEI_SCALES_MASK == 0
    tmp *= wei_scales[0];
#else
    tmp *= wei_scales[g * OC + oc];
#endif
#endif

#if WITH_BIAS
    tmp += (POST_OP_DATA_T)BIA_TO_REF(bias[g * OC + oc]);
#endif

    POST_OP_DATA_T sum_src;
#if WITH_SUM
    sum_src = (POST_OP_DATA_T)SUM_TO_REF(
            AS_SUM_DATA_T(dst[DST_OFF(n, g * OC + oc, od, oh, ow)]));
#endif

#if NDIMS == 3
    const unsigned po_d2 = ow;
    const unsigned po_d3 = 0;
    const unsigned po_d4 = 0;
#elif NDIMS == 4
    const unsigned po_d2 = oh;
    const unsigned po_d3 = ow;
    const unsigned po_d4 = 0;
#elif NDIMS == 5
    const unsigned po_d2 = od;
    const unsigned po_d3 = oh;
    const unsigned po_d4 = ow;
#else
    const unsigned po_d2 = 0;
    const unsigned po_d3 = 0;
    const unsigned po_d4 = 0;
#endif
    APPLY_POST_OPS_SERIAL(tmp, POST_OP_DATA_T, sum_src, POST_OP_DATA_T, n, 1,
            g * OC + oc, 1, po_d2, 1, po_d3, 1, po_d4, 1, 0, 1);

#if WITH_DST_SCALES
    tmp /= dst_scales[0];
#endif

#if WITH_DST_ZPOINTS
    const int dst_zp = dst_zpoints[WITH_DST_ZPOINTS_PER_OC ? g * OC + oc : 0];
    tmp += dst_zp;
#endif // WITH_DST_ZPOINTS
    dst[DST_OFF(n, g * OC + oc, od, oh, ow)] = TO_DST(tmp);
}
#endif

#if IS_BWD_D
KERNEL_ATTR
__kernel void ref_convolution_bwd_data(__global SRC_DATA_T *diff_src,
        const __global WEI_DATA_T *wei, const __global DST_DATA_T *diff_dst,
        const __global BIA_DATA_T *bias POST_OP_ARGS,
        const __global float *src_scales, const __global float *wei_scales,
        const __global float *dst_scales, const __global int *src_zpoints,
        const __global int *dst_zpoints) {
    const int n = GWS_GET_MB();
    const int ic = GWS_GET_IC();
    const int g = GWS_GET_G();
    const int id = GWS_GET_ID();
    const int ih = GWS_GET_IH();
    const int iw = GWS_GET_IW();
    ACC_DATA_T d = 0.0;
    for_(int oc = 0; oc < OC; ++oc)
    for_(int kd = 0; kd < KD; ++kd)
    for_(int kh = 0; kh < KH; ++kh)
    for (int kw = 0; kw < KW; ++kw) {
        if (iw + PW < kw * (1 + DW) || ih + PH < kh * (1 + DH)
                || id + PD < kd * (1 + DD))
            continue;
        int ow = iw - kw * (1 + DW) + PW;
        int oh = ih - kh * (1 + DH) + PH;
        int od = id - kd * (1 + DD) + PD;
        if (ow % SW != 0 || oh % SH != 0 || od % SD != 0) continue;

        ow /= SW;
        oh /= SH;
        od /= SD;
        if (oh < OH && ow < OW && od < OD) {
            const uint dst_off = DST_OFF(n, g * OC + oc, od, oh, ow);
            const uint wei_off = WEI_OFF(g, oc, ic, kd, kh, kw);
            d += DST_TO_REF(diff_dst[dst_off]) * WEI_TO_REF(wei[wei_off]);
#if WITH_SRC_ZPOINTS
            const int src_zp
                    = src_zpoints[WITH_SRC_ZPOINTS_PER_IC ? g * OC + oc : 0];
            d -= src_zp * WEI_TO_REF(wei[wei_off]);
#endif
        }
    }

    float sum_src;
#if WITH_SUM
    sum_src = convert_float(
            SRC_TO_REF(diff_src[SRC_OFF(n, g * IC + ic, id, ih, iw)]));
#endif

    float accumulator = convert_float(d);

#if WITH_SRC_SCALES
    accumulator *= src_scales[0];
#endif
#if WITH_WEI_SCALES
#if WEI_SCALES_MASK == 0
    accumulator *= wei_scales[0];
#else
    accumulator *= wei_scales[g * IC + ic];
#endif
#endif

#if WITH_BIAS
    accumulator += BIA_TO_REF(bias[g * IC + ic]);
#endif

#if NDIMS == 3
    const unsigned po_d2 = iw;
    const unsigned po_d3 = 0;
    const unsigned po_d4 = 0;
#elif NDIMS == 4
    const unsigned po_d2 = ih;
    const unsigned po_d3 = iw;
    const unsigned po_d4 = 0;
#elif NDIMS == 5
    const unsigned po_d2 = id;
    const unsigned po_d3 = ih;
    const unsigned po_d4 = iw;
#else
    const unsigned po_d2 = 0;
    const unsigned po_d3 = 0;
    const unsigned po_d4 = 0;
#endif
    APPLY_POST_OPS_SERIAL(accumulator, float, sum_src, float, n, 1, g *IC + ic,
            1, po_d2, 1, po_d3, 1, po_d4, 1, 0, 1);

#if WITH_DST_SCALES
    accumulator /= dst_scales[0];
#endif

#if WITH_DST_ZPOINTS
    const int dst_zp = dst_zpoints[WITH_DST_ZPOINTS_PER_OC ? g * IC + ic : 0];
    accumulator += dst_zp;
#endif // WITH_DST_ZPOINTS

    diff_src[SRC_OFF(n, g * IC + ic, id, ih, iw)] = TO_SRC(accumulator);
}
#endif

#if IS_BWD_W
KERNEL_ATTR
__kernel void ref_convolution_bwd_weights(const __global SRC_DATA_T *src,
        __global WEI_DATA_T *diff_wei, __global BIA_DATA_T *diff_bias,
        const __global DST_DATA_T *diff_dst) {
    const int g = GWS_GET_G();
    const int ic = GWS_GET_IC();
    const int oc = GWS_GET_OC();
    const int kd = GWS_GET_KD();
    const int kh = GWS_GET_KH();
    const int kw = GWS_GET_KW();

#if WITH_BIAS
    if (ic == 0 && kh == 0 && kw == 0 & kd == 0) {
        ACC_DATA_T d = 0.0;
        for (int n = 0; n < MB; ++n)
            for (int od = 0; od < OD; ++od)
                for (int oh = 0; oh < OH; ++oh)
                    for (int ow = 0; ow < OW; ++ow) {
                        d += DST_TO_REF(
                                diff_dst[DST_OFF(n, g * OC + oc, od, oh, ow)]);
                    }
        diff_bias[g * OC + oc] = TO_BIA(d);
    }
#endif

    ACC_DATA_T dw = 0.0;
    for (int n = 0; n < MB; ++n)
        for (int od = 0; od < OD; ++od)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    if (ow * SW + kw * (1 + DW) < PW
                            || oh * SH + kh * (1 + DH) < PH
                            || od * SD + kd * (1 + DD) < PD
                            || ow * SW + kw * (1 + DW) >= IW + PW
                            || oh * SH + kh * (1 + DH) >= IH + PH
                            || od * SD + kd * (1 + DD) >= ID + PD)
                        continue;

                    int id = od * SD - PD + kd * (1 + DD);
                    int ih = oh * SH - PH + kh * (1 + DH);
                    int iw = ow * SW - PW + kw * (1 + DW);

                    dw += DST_TO_REF(
                                  diff_dst[DST_OFF(n, g * OC + oc, od, oh, ow)])
                            * SRC_TO_REF(
                                    src[SRC_OFF(n, g * IC + ic, id, ih, iw)]);
                }
    diff_wei[WEI_OFF(g, oc, ic, kd, kh, kw)] = TO_WEI(dw);
}
#endif
