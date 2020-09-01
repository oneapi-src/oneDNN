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

#include "gpu/ocl/ocl_types.h"

#if DT_S8 || DT_U8
#define RINT rint
#else
#define RINT
#endif

#if IS_FWD
KERNEL_ATTR
__kernel void ref_pooling_fwd(
        __global DATA_T *src, __global int *ws, __global DST_DATA_T *dst) {
    const int mb = GWS_GET_MB();
    const int oc = GWS_GET_OC();

    for (int od = 0; od < OD; ++od)
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                const uint dst_off = DST_OFF(mb, oc, od, oh, ow);
#if ALG_MAX && IS_TRAINING
                ws[dst_off] = -1;
#endif
#if ALG_AVG_P || ALG_AVG_NP
                float d = 0;
#endif
#if ALG_MAX && DT_BF16
                DEF_ACC_DATA_T d = DATA_MIN;
#elif ALG_MAX // DT_BF16
                DATA_T d = DATA_MIN;
#endif
                for (int kd = 0; kd < KD; ++kd)
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            const int id = od * SD - PD + kd * (DD + 1);
                            const int ih = oh * SH - PH + kh * (DH + 1);
                            const int iw = ow * SW - PW + kw * (DW + 1);

                            if (id < 0 || id >= ID) continue;
                            if (ih < 0 || ih >= IH) continue;
                            if (iw < 0 || iw >= IW) continue;

                            int src_off = SRC_OFF(mb, oc, id, ih, iw);
#if ALG_MAX
#if IS_TRAINING
                            if (ws[dst_off] < 0)
                                ws[dst_off] = kd * KH * KW + kh * KW + kw;
#endif
#if DT_BF16
                            DEF_ACC_DATA_T s = DATA_TO_REF(src[src_off]);
#else // DT_BF16 == 0
                            DATA_T s = src[src_off];
#endif
                            if (s > d) {
                                d = s;
#if IS_TRAINING
                                ws[dst_off] = kd * KH * KW + kh * KW + kw;
#endif
                            }
#else
                            d += DATA_TO_REF(src[src_off]);
#endif
                        }
                    }
#if ALG_MAX
                dst[dst_off] = TO_DST(d);
#if IS_TRAINING
                if (ws[dst_off] < 0) ws[dst_off] = 0;
#endif
#else
#if ALG_AVG_P
                const int num_summands = KD * KW * KH;
#else
                const int id_start = od * SD - PD;
                const int ih_start = oh * SH - PH;
                const int iw_start = ow * SW - PW;
                const int id_end = od * SD - PD + (KD - 1) * DD + KD;
                const int ih_end = oh * SH - PH + (KH - 1) * DH + KH;
                const int iw_end = ow * SW - PW + (KW - 1) * DW + KW;

                const int id_start_excluded
                        = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
                const int ih_start_excluded
                        = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
                const int iw_start_excluded
                        = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
                const int id_end_excluded
                        = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
                const int ih_end_excluded
                        = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
                const int iw_end_excluded
                        = iw_end > IW ? (iw_end - IW - 1) / (DW + 1) + 1 : 0;

                const int num_summands
                        = (KD - id_start_excluded - id_end_excluded)
                        * (KH - ih_start_excluded - ih_end_excluded)
                        * (KW - iw_start_excluded - iw_end_excluded);
#endif
                dst[dst_off] = TO_DST(d / num_summands);
#endif
            }
}
#endif

#if IS_BWD
KERNEL_ATTR
__kernel void ref_pooling_bwd(__global DATA_T *diff_src, __global int *ws,
        __global DST_DATA_T *diff_dst) {
    const int mb = GWS_GET_MB();
    const int oc = GWS_GET_OC();

    for (int id = 0; id < ID; ++id)
        for (int ih = 0; ih < IH; ++ih)
            for (int iw = 0; iw < IW; ++iw) {
                uint diff_src_offset = SRC_OFF(mb, oc, id, ih, iw);
                diff_src[diff_src_offset] = 0;
            }

#if DT_BF16
    // For bfloat16, iterate over input to use a temporary float accumulator
    for (int id = 0; id < ID; ++id)
        for (int ih = 0; ih < IH; ++ih)
            for (int iw = 0; iw < IW; ++iw) {
                float s = 0;
                for (int kd = 0; kd < KD; ++kd)
                    for (int kh = 0; kh < KH; ++kh)
                        for (int kw = 0; kw < KW; ++kw) {
                            int _od = id + PD - kd * (DD + 1);
                            int _oh = ih + PH - kh * (DH + 1);
                            int _ow = iw + PW - kw * (DW + 1);
                            if (_od % SD != 0 || _oh % SH != 0 || _ow % SW != 0)
                                continue;

                            int od = _od / SD;
                            int oh = _oh / SH;
                            int ow = _ow / SW;

                            if (od < 0 || od >= OD) continue;
                            if (oh < 0 || oh >= OH) continue;
                            if (ow < 0 || ow >= OW) continue;

                            const uint dst_off = DST_OFF(mb, oc, od, oh, ow);

#if ALG_MAX
                            const int index = ws[dst_off];

                            const int hw = index % (KW * KH);
                            const int w_kd = index / (KW * KH);
                            const int w_kw = hw % KW;
                            const int w_kh = hw / KW;
                            if (w_kd != kd || w_kh != kh || w_kw != kw)
                                continue;
#endif

#if ALG_MAX
                            const int denom = 1;
#elif ALG_AVG_P
                            const int denom = KD * KH * KW;
#elif ALG_AVG_NP
                            const int id_start = od * SD - PD;
                            const int ih_start = oh * SH - PH;
                            const int iw_start = ow * SW - PW;
                            const int id_end
                                    = od * SD - PD + (KD - 1) * DD + KD;
                            const int ih_end
                                    = oh * SH - PH + (KH - 1) * DH + KH;
                            const int iw_end
                                    = ow * SW - PW + (KW - 1) * DW + KW;

                            const int id_start_excluded = id_start < 0
                                    ? (0 - id_start - 1) / (DD + 1) + 1
                                    : 0;
                            const int ih_start_excluded = ih_start < 0
                                    ? (0 - ih_start - 1) / (DH + 1) + 1
                                    : 0;
                            const int iw_start_excluded = iw_start < 0
                                    ? (0 - iw_start - 1) / (DW + 1) + 1
                                    : 0;
                            const int id_end_excluded = id_end > ID
                                    ? (id_end - ID - 1) / (DD + 1) + 1
                                    : 0;
                            const int ih_end_excluded = ih_end > IH
                                    ? (ih_end - IH - 1) / (DH + 1) + 1
                                    : 0;
                            const int iw_end_excluded = iw_end > IW
                                    ? (iw_end - IW - 1) / (DW + 1) + 1
                                    : 0;

                            const int denom
                                    = (KD - id_start_excluded - id_end_excluded)
                                    * (KH - ih_start_excluded - ih_end_excluded)
                                    * (KW - iw_start_excluded
                                            - iw_end_excluded);
#endif
                            s += DATA_TO_REF(diff_dst[dst_off]) / denom;
                        }
                uint diff_src_offset = SRC_OFF(mb, oc, id, ih, iw);
                diff_src[diff_src_offset] = CONVERT_DATA_T(s);
            }
#else // DT_BF16
    for (int od = 0; od < OD; ++od)
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                const uint dst_off = DST_OFF(mb, oc, od, oh, ow);
                const float d = diff_dst[dst_off];

#if ALG_MAX
                const int index = ws[dst_off];
                const int kd = index / (KW * KH);
                const int hw = index % (KW * KH);
                const int kw = hw % KW;
                const int kh = hw / KW;

                const int id = od * SD - PD + kd * (DD + 1);
                const int ih = oh * SH - PH + kh * (DH + 1);
                const int iw = ow * SW - PW + kw * (DW + 1);

                if (id < 0 || id >= ID) continue;
                if (ih < 0 || ih >= IH) continue;
                if (iw < 0 || iw >= IW) continue;

                uint diff_src_offset = SRC_OFF(mb, oc, id, ih, iw);
                diff_src[diff_src_offset] += d;
#else
                const int id_start = od * SD - PD;
                const int ih_start = oh * SH - PH;
                const int iw_start = ow * SW - PW;
                const int id_end = od * SD - PD + (KD - 1) * DD + KD;
                const int ih_end = oh * SH - PH + (KH - 1) * DH + KH;
                const int iw_end = ow * SW - PW + (KW - 1) * DW + KW;

                const int id_start_excluded
                        = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
                const int ih_start_excluded
                        = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
                const int iw_start_excluded
                        = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
                const int id_end_excluded
                        = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
                const int ih_end_excluded
                        = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
                const int iw_end_excluded
                        = iw_end > IW ? (iw_end - IW - 1) / (DW + 1) + 1 : 0;

#if ALG_AVG_P
                const int num_summands = KD * KW * KH;
#else
                const int num_summands
                        = (KD - id_start_excluded - id_end_excluded)
                        * (KH - ih_start_excluded - ih_end_excluded)
                        * (KW - iw_start_excluded - iw_end_excluded);
#endif
                for (int id = id_start; id < min(id_end, ID); id += DD + 1)
                    for (int ih = ih_start; ih < min(ih_end, IH); ih += DH + 1)
                        for (int iw = iw_start; iw < min(iw_end, IW);
                                iw += DW + 1) {
                            if (id < 0 || ih < 0 || iw < 0) continue;
                            const uint diff_src_offset
                                    = SRC_OFF(mb, oc, id, ih, iw);
                            diff_src[diff_src_offset] += d / num_summands;
                        }
#endif
            }
#endif
}
#endif
