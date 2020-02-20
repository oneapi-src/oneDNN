/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "ocl/ocl_types.h"
#if WITH_ELTWISE == 1 || WITH_POST_SUM_ELTWISE == 1
#include "ocl/ocl_post_ops.h"
#endif

#if IS_FWD
KERNEL_ATTR
__kernel void ref_convolution_fwd(
        const __global SRC_DATA_T *src, const __global WEI_DATA_T *wei,
        const __global BIA_DATA_T *bias, __global DST_DATA_T *dst,
        float eltwise_alpha, float eltwise_beta, float sum_scale
#if SRC_DT_S8 == 1 || SRC_DT_U8 == 1
        ,
        float scales
#endif
) {
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
                    const uint wht_off = WHT_OFF(g, oc, ic, kd, kh, kw);
                    d += SRC_TO_REF(src[src_off]) * WEI_TO_REF(wei[wht_off]);
                }
    POST_OP_DATA_T tmp = d;

#if WITH_BIAS
    tmp += (POST_OP_DATA_T)BIA_TO_REF(bias[g * OC + oc]);
#endif

#if SRC_DT_S8 == 1 || SRC_DT_U8 == 1
    tmp *= scales;
#endif

#if WITH_ELTWISE == 1
    tmp = fwd_eltwise(tmp, eltwise_alpha, eltwise_beta);
#endif

#if WITH_SUM == 1
#if SUM_SCALE == 1
    tmp += (POST_OP_DATA_T)dst[DST_OFF(n, g * OC + oc, od, oh, ow)];
#else
    tmp += sum_scale * (POST_OP_DATA_T)dst[DST_OFF(n, g * OC + oc, od, oh, ow)];
#endif
#endif

#if WITH_POST_SUM_ELTWISE == 1
    tmp = fwd_eltwise(tmp, eltwise_alpha, eltwise_beta);
#endif

    dst[DST_OFF(n, g * OC + oc, od, oh, ow)] = TO_DST(tmp);
}
#endif

#if IS_BWD_D
KERNEL_ATTR
__kernel void ref_convolution_bwd_data(__global SRC_DATA_T *diff_src,
        const __global WEI_DATA_T *wei, const __global DST_DATA_T *diff_dst,
        const __global BIA_DATA_T *bias) {
    const int n = GWS_GET_MB();
    const int ic = GWS_GET_IC();
    const int g = GWS_GET_G();
    const int id = GWS_GET_ID();
    const int ih = GWS_GET_IH();
    const int iw = GWS_GET_IW();

    ACC_DATA_T d = WITH_BIAS ? BIA_TO_REF(bias[g * IC + ic]) : 0.0;

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
            const uint wht_off = WHT_OFF(g, oc, ic, kd, kh, kw);
            d += DST_TO_REF(diff_dst[dst_off]) * WEI_TO_REF(wei[wht_off]);
        }
    }
    diff_src[SRC_OFF(n, g * IC + ic, id, ih, iw)] = TO_SRC(d);
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
    diff_wei[WHT_OFF(g, oc, ic, kd, kh, kw)] = TO_WEI(dw);
}
#endif
