/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

// these are all supported combinations of different src/dst datatypes
#if DST_DT_F32
#define DST_DATA_MIN -FLT_MAX
#elif DST_DT_F64
// limitation caused by Benchdnn reference written for float, for correct
// -DBL_MAX value displays -inf value instead. To be corrected after Benchdnn
// starts supporting f64 in core.
#define DST_DATA_MIN -FLT_MAX
#elif DST_DT_S8
#define DST_DATA_MIN CHAR_MIN
#elif DST_DT_U8
#define DST_DATA_MIN 0
#elif DST_DT_F16
#define DST_DATA_MIN -HALF_MAX
#else
#define DST_DATA_MIN DATA_MIN
#endif

#if DT_F64 || DST_DT_F64
#define ACC_DATA_T double
#else
#define ACC_DATA_T float
#endif

#if IS_FWD
KERNEL_ATTR
__kernel void ref_pooling_fwd(__global DATA_T *src, __global int *ws,
        __global DST_DATA_T *dst POST_OP_ARGS) {
    const int mb = GWS_GET_MB();
    const int oc = GWS_GET_OC();
    const int od = GWS_GET_OD();
    const int oh = GWS_GET_OH();
    const int ow = GWS_GET_OW();

    if (mb >= SRC_D0 || oc >= SRC_D1) {
        const uint dst_off = DST_OFF(mb, oc, od, oh, ow);
        dst[dst_off] = TO_DST(0.f);
#if ALG_MAX && IS_TRAINING
        ws[dst_off] = 0;
#endif
        return;
    }

    const uint dst_off = DST_OFF(mb, oc, od, oh, ow);
#if ALG_MAX && IS_TRAINING
    ws[dst_off] = -1;
#endif
#if ALG_AVG_P || ALG_AVG_NP
    ACC_DATA_T d = 0;
#endif
#if ALG_MAX
#if DT_BF16
    DEF_ACC_DATA_T d = TO_DEF_ACC_DATA_T(DST_DATA_MIN);
#else
    ACC_DATA_T d = DST_DATA_MIN;
#endif // DT_BF16
#endif // ALG_MAX

    for (int kd = 0; kd < KD; ++kd) {
        const int id = od * SD - PD + kd * (DD + 1);
        if (id < 0 || id >= ID) continue;
        for (int kh = 0; kh < KH; ++kh) {
            const int ih = oh * SH - PH + kh * (DH + 1);
            if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; ++kw) {
                const int iw = ow * SW - PW + kw * (DW + 1);
                if (iw < 0 || iw >= IW) continue;

                int src_off = SRC_OFF(mb, oc, id, ih, iw);
#if ALG_MAX
#if IS_TRAINING
                if (ws[dst_off] < 0) ws[dst_off] = kd * KH * KW + kh * KW + kw;
#endif
#if DT_BF16
                DEF_ACC_DATA_T s = DATA_TO_REF(src[src_off]);
#elif DT_F64
                ACC_DATA_T s = src[src_off];
#else
                ACC_DATA_T s = DATA_TO_REF(src[src_off]);
#endif // DT_BF16
                if (s > d) {
                    d = s;
#if IS_TRAINING
                    ws[dst_off] = kd * KH * KW + kh * KW + kw;
#endif
                }
#else // ALG_MAX == 0
#if DT_F64
                d += src[src_off];
#else
                d += DATA_TO_REF(src[src_off]);
#endif // DT_F64
#endif // ALG_MAX
            }
        }
    }
#if ALG_MAX
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

    const int num_summands = (KD - id_start_excluded - id_end_excluded)
            * (KH - ih_start_excluded - ih_end_excluded)
            * (KW - iw_start_excluded - iw_end_excluded);
#endif
    d /= num_summands;
#endif

        // post op service
#if DT_BF16 || DT_F64
    POST_OP_DATA_T tmp = d;
#else
    POST_OP_DATA_T tmp = (POST_OP_DATA_T)DATA_TO_REF(d);
#endif // DT_BF16
    POST_OP_DATA_T sum_src;
#if WITH_SUM
    sum_src = (POST_OP_DATA_T)DATA_TO_REF(dst[dst_off]);
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
    APPLY_POST_OPS_SERIAL(tmp, POST_OP_DATA_T, sum_src, POST_OP_DATA_T, mb, 1,
            oc, 1, po_d2, 1, po_d3, 1, po_d4, 1, 0, 1);

    // store result
#if DT_F64
    dst[dst_off] = tmp;
#else
    dst[dst_off] = TO_DST(tmp);
#endif
}
#endif

#if IS_BWD
KERNEL_ATTR
__kernel void ref_pooling_bwd(__global DATA_T *diff_src, __global int *ws,
        __global DST_DATA_T *diff_dst) {
    const int mb = GWS_GET_MB();
    const int oc = GWS_GET_OC();
    const int id = GWS_GET_ID();
    const int ih = GWS_GET_IH();
    const int iw = GWS_GET_IW();

    DEF_ACC_DATA_T s = 0;
    int denom = 1;
#if ALG_AVG_P
    denom = KD * KH * KW;
#endif

    for (int kd = 0; kd < KD; ++kd) {
        int _od = id + PD - kd * (DD + 1);
        if (_od % SD != 0) continue;
        int od = _od / SD;
        if (od < 0 || od >= OD) continue;

        for (int kh = 0; kh < KH; ++kh) {
            int _oh = ih + PH - kh * (DH + 1);
            if (_oh % SH != 0) continue;
            int oh = _oh / SH;
            if (oh < 0 || oh >= OH) continue;

            for (int kw = 0; kw < KW; ++kw) {
                int _ow = iw + PW - kw * (DW + 1);
                if (_ow % SW != 0) continue;
                int ow = _ow / SW;
                if (ow < 0 || ow >= OW) continue;

                const uint dst_off = DST_OFF(mb, oc, od, oh, ow);

#if ALG_MAX
                const int index = ws[dst_off];

                const int hw = index % (KW * KH);
                const int w_kd = index / (KW * KH);
                const int w_kw = hw % KW;
                const int w_kh = hw / KW;
                if (w_kd != kd || w_kh != kh || w_kw != kw) continue;
#endif

#if ALG_AVG_NP
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

                denom = (KD - id_start_excluded - id_end_excluded)
                        * (KH - ih_start_excluded - ih_end_excluded)
                        * (KW - iw_start_excluded - iw_end_excluded);
#endif
#if ALG_MAX || ALG_AVG_P
                s += TO_DEF_ACC_DATA_T(diff_dst[dst_off]);
#elif ALG_AVG_NP
                s += TO_DEF_ACC_DATA_T(diff_dst[dst_off]) / denom;
#endif
            }
        }
    }

// Divide before storing for better accuracy.
#if ALG_MAX || ALG_AVG_P
    s /= denom;
#endif

    uint diff_src_offset = SRC_OFF(mb, oc, id, ih, iw);
    diff_src[diff_src_offset] = CONVERT_DATA_T(s);
}
#endif
