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
#elif DST_DT_HF8
#define DST_DATA_MIN (uchar)0xFE
#elif DST_DT_BF8
#define DST_DATA_MIN (uchar)0xFB
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

    if (GWS_OVERFLOW) return;

    const off_t mb = GWS_GET_MB();
    const off_t oc = GWS_GET_OC();
    const off_t od = GWS_GET_OD();
    const off_t oh = GWS_GET_OH();
    const off_t ow = GWS_GET_OW();

    if (mb >= SRC_D0 || oc >= SRC_D1) {
        const off_t dst_off = DST_OFF(mb, oc, od, oh, ow);
        dst[dst_off] = TO_DST(0.0f);
#if ALG_MAX && IS_TRAINING
        ws[dst_off] = 0;
#endif
        return;
    }

    const off_t dst_off = DST_OFF(mb, oc, od, oh, ow);
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

    for (off_t kd = 0; kd < KD; ++kd) {
        const off_t id = od * SD - PD + kd * (DD + 1);
        if (id < 0 || id >= ID) continue;
        for (off_t kh = 0; kh < KH; ++kh) {
            const off_t ih = oh * SH - PH + kh * (DH + 1);
            if (ih < 0 || ih >= IH) continue;
            for (off_t kw = 0; kw < KW; ++kw) {
                const off_t iw = ow * SW - PW + kw * (DW + 1);
                if (iw < 0 || iw >= IW) continue;

                off_t src_off = SRC_OFF(mb, oc, id, ih, iw);
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
    const off_t num_summands = KD * KW * KH;
#else
    const off_t id_start = od * SD - PD;
    const off_t ih_start = oh * SH - PH;
    const off_t iw_start = ow * SW - PW;
    const off_t id_end = od * SD - PD + (KD - 1) * DD + KD;
    const off_t ih_end = oh * SH - PH + (KH - 1) * DH + KH;
    const off_t iw_end = ow * SW - PW + (KW - 1) * DW + KW;

    const off_t id_start_excluded
            = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
    const off_t ih_start_excluded
            = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
    const off_t iw_start_excluded
            = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
    const off_t id_end_excluded
            = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
    const off_t ih_end_excluded
            = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
    const off_t iw_end_excluded
            = iw_end > IW ? (iw_end - IW - 1) / (DW + 1) + 1 : 0;

    const off_t num_summands = (KD - id_start_excluded - id_end_excluded)
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

    if (GWS_OVERFLOW) return;

    const off_t mb = GWS_GET_MB();
    const off_t oc = GWS_GET_OC();
    const off_t id = GWS_GET_ID();
    const off_t ih = GWS_GET_IH();
    const off_t iw = GWS_GET_IW();

    DEF_ACC_DATA_T s = 0;
    off_t denom = 1;
#if ALG_AVG_P
    denom = KD * KH * KW;
#endif

    for (off_t kd = 0; kd < KD; ++kd) {
        off_t _od = id + PD - kd * (DD + 1);
        if (_od % SD != 0) continue;
        off_t od = _od / SD;
        if (od < 0 || od >= OD) continue;

        for (off_t kh = 0; kh < KH; ++kh) {
            off_t _oh = ih + PH - kh * (DH + 1);
            if (_oh % SH != 0) continue;
            off_t oh = _oh / SH;
            if (oh < 0 || oh >= OH) continue;

            for (off_t kw = 0; kw < KW; ++kw) {
                off_t _ow = iw + PW - kw * (DW + 1);
                if (_ow % SW != 0) continue;
                off_t ow = _ow / SW;
                if (ow < 0 || ow >= OW) continue;

                const off_t dst_off = DST_OFF(mb, oc, od, oh, ow);

#if ALG_MAX
                const off_t index = ws[dst_off];

                const off_t hw = index % (KW * KH);
                const off_t w_kd = index / (KW * KH);
                const off_t w_kw = hw % KW;
                const off_t w_kh = hw / KW;
                if (w_kd != kd || w_kh != kh || w_kw != kw) continue;
#endif

#if ALG_AVG_NP
                const off_t id_start = od * SD - PD;
                const off_t ih_start = oh * SH - PH;
                const off_t iw_start = ow * SW - PW;
                const off_t id_end = od * SD - PD + (KD - 1) * DD + KD;
                const off_t ih_end = oh * SH - PH + (KH - 1) * DH + KH;
                const off_t iw_end = ow * SW - PW + (KW - 1) * DW + KW;

                const off_t id_start_excluded
                        = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
                const off_t ih_start_excluded
                        = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
                const off_t iw_start_excluded
                        = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
                const off_t id_end_excluded
                        = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
                const off_t ih_end_excluded
                        = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
                const off_t iw_end_excluded
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

    off_t diff_src_offset = SRC_OFF(mb, oc, id, ih, iw);
    diff_src[diff_src_offset] = CONVERT_DATA_T(s);
}
#endif
