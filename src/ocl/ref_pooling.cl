/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#if USE_16MB_UNROLL == 1
#define MB_BLOCK 16
#define MB16
#define VECT_DT_N 8
#else
#define MB_BLOCK 1
#define VECT_DT_N 1
#endif

#if VECT_DT_N == 1
#if DT_F16 == 1
#define VECT_ACC_FLOAT_T half
#else
#define VECT_ACC_FLOAT_T float
#endif
#elif VECT_DT_N == 8
#if DT_F16 == 1
#define VECT_ACC_FLOAT_T half8
#else
#define VECT_ACC_FLOAT_T float8
#endif
#endif

#include "ocl/ocl_types.h"

#if IS_FWD == 1
KERNEL_ATTR
__kernel void ref_pooling_fwd(
        __global DATA_T *src, __global int *ws, __global DATA_T *dst) {

#if USE_16C_UNROLL == 1
    const int mb = GWS_GET_MB();
    const int oc = GWS_GET_OC();
    const int od = GWS_GET_OD();
    const int oh = GWS_GET_OH();

    for (int ow = 0; ow < OW; ++ow) {
        const int id = od * SD - PD;
        const int ih = oh * SH - PH;
        const int iw = ow * SW - PW;

        __global DATA_T *src_ = src + SRC_OFF(mb, oc, id, ih, iw);
        __global DATA_T *dst_ = dst + DST_OFF(mb, oc, od, oh, ow);

#if POOLING_MAX == 1
#if IS_TRAINING == 1
        __global DATA_T *ws_ = ws + DST_OFF(mb, oc, od, oh, ow);
        VECT_INT_T blockWS0 = 0;
#ifdef MB16
        VECT_INT_T blockWS1 = 0;
#endif
#endif // IS_TRAINING
        VECT_DATA_T blockD0 = DATA_MIN;
#ifdef MB16
        VECT_DATA_T blockD1 = DATA_MIN;
#endif
#else // POOLING_MAX
        VECT_DEF_ACC_DATA_T blockD0 = DATA_ZERO;
#ifdef MB16
        VECT_DEF_ACC_DATA_T blockD1 = DATA_ZERO;
#endif
#endif // POOLING_MAX
        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {

                    if (id + kd < 0 || id + kd >= ID) continue;
                    if (ih + kh < 0 || ih + kh >= IH) continue;
                    if (iw + kw < 0 || iw + kw >= IW) continue;

                    const int src_off = kd * IH * IW * MB_BLOCK * 16
                            + kh * IW * MB_BLOCK * 16 + kw * MB_BLOCK * 16;

                    VECT_DATA_T blockS0 = AS_VECT_DATA_T(VECT_BLOCK_READ((
                            const __global VECT_BLOCK_DATA_T *)&src_[src_off]));
#ifdef MB16
                    VECT_DATA_T blockS1 = AS_VECT_DATA_T(
                            VECT_BLOCK_READ((const __global VECT_BLOCK_DATA_T
                                            *)&src_[src_off + 8 * 16]));
#endif
#if POOLING_MAX == 1
#if IS_TRAINING == 1
                    VECT_INT_T blockCMP0 = isless(blockD0, blockS0);
                    blockWS0 = select(blockWS0,
                            (VECT_INT_T)(kd * KH * KW + kh * KW + kw),
                            blockCMP0);
                    blockD0 = select(blockD0, blockS0, blockCMP0);
#ifdef MB16
                    VECT_INT_T blockCMP1 = isless(blockD1, blockS1);
                    blockWS1 = select(blockWS1,
                            (VECT_INT_T)(kd * KH * KW + kh * KW + kw),
                            blockCMP1);
                    blockD1 = select(blockD1, blockS1, blockCMP1);
#endif
#else // TRAINING
                    blockD0 = max(blockD0, blockS0);
#ifdef MB16
                    blockD1 = max(blockD1, blockS1);
#endif
#endif // TRAINING
#else // POOLING_MAX
                    blockD0 += blockS0;
#ifdef MB16
                    blockD1 += blockS1;
#endif
#endif // POOLING_MAX
                }
            }

#ifdef POOLING_AVG_INCLUDE_PADDING
        blockD0 = ROUND((VECT_ACC_FLOAT_T)blockD0 / (KD * KH * KW));
#ifdef MB16
        blockD1 = ROUND((VECT_ACC_FLOAT_T)blockD1 / (KD * KH * KW));
#endif
#endif // POOLING_AVG_INCLUDE_PADDING

#ifdef POOLING_AVG_EXCLUDE_PADDING
        const int id_start = max(od * SD - PD, 0);
        const int ih_start = max(oh * SH - PH, 0);
        const int iw_start = max(ow * SW - PW, 0);
        const int id_end = min(od * SD - PD + KD, ID);
        const int ih_end = min(oh * SH - PH + KH, IH);
        const int iw_end = min(ow * SW - PW + KW, IW);
        const DEF_ACC_DATA_T num_summands = (ih_end - ih_start)
                * (iw_end - iw_start) * (id_end - id_start);
        blockD0 = ROUND((VECT_ACC_FLOAT_T)blockD0 / num_summands);
#ifdef MB16
        blockD1 = ROUND((VECT_ACC_FLOAT_T)blockD1 / num_summands);
#endif
#endif // POOLING_AVG_EXCLUDE_PADDING

        VECT_BLOCK_WRITE((__global VECT_BLOCK_DATA_T *)&dst_[0],
                AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(blockD0)));
#ifdef MB16
        VECT_BLOCK_WRITE((__global VECT_BLOCK_DATA_T *)&dst_[8 * 16],
                AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(blockD1)));
#endif
#if POOLING_MAX == 1 && IS_TRAINING == 1
        VECT_BLOCK_WRITE((__global VECT_BLOCK_DATA_T *)&ws_[0],
                AS_VECT_BLOCK_DATA_T(blockWS0));
#ifdef MB16
        VECT_BLOCK_WRITE((__global VECT_BLOCK_DATA_T *)&ws_[8 * 16],
                AS_VECT_BLOCK_DATA_T(blockWS1));
#endif
#endif // POOLING_MAX && IS_TRAINING
    }

#else // USE_16C_UNROLL == 0
    const int mb = GWS_GET_MB();
    const int oc = GWS_GET_OC();

    for (int od = 0; od < OD; ++od)
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                const uint dst_off = DST_OFF(mb, oc, od, oh, ow);
#if POOLING_MAX == 1 && IS_TRAINING == 1
                ws[dst_off] = -1;
#endif
#if POOLING_MAX == 1
#if DT_BF16 == 1
                DEF_ACC_DATA_T d = DATA_MIN;
#else // DT_BF16 == 0
                DATA_T d = DATA_MIN;
#endif
                for (int kd = 0; kd < KD; ++kd)
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            const int id = od * SD - PD + kd;
                            const int ih = oh * SH - PH + kh;
                            const int iw = ow * SW - PW + kw;

                            if (id < 0 || id >= ID) continue;
                            if (ih < 0 || ih >= IH) continue;
                            if (iw < 0 || iw >= IW) continue;

#if POOLING_MAX == 1 && IS_TRAINING == 1
                            if (ws[dst_off] < 0)
                                ws[dst_off] = kd * KH * KW + kh * KW + kw;
#endif
                            int src_off = SRC_OFF(mb, oc, id, ih, iw);
#if DT_BF16 == 1
                            DEF_ACC_DATA_T s = DATA_TO_REF(src[src_off]);
#else // DT_BF16 == 0
                            DATA_T s = src[src_off];
#endif
                            if (s > d) {
                                d = s;
#if POOLING_MAX == 1 && IS_TRAINING == 1
                                ws[dst_off] = kd * KH * KW + kh * KW + kw;
#endif
                            }
                        }
                    }
                dst[dst_off] = CONVERT_DATA_T(d);
#if POOLING_MAX == 1 && IS_TRAINING == 1
                if (ws[dst_off] < 0) ws[dst_off] = 0;
#endif
#else
                const int id_start = max(od * SD - PD, 0);
                const int ih_start = max(oh * SH - PH, 0);
                const int iw_start = max(ow * SW - PW, 0);
                const int id_end = min(od * SD - PD + KD, ID);
                const int ih_end = min(oh * SH - PH + KH, IH);
                const int iw_end = min(ow * SW - PW + KW, IW);

#ifdef POOLING_AVG_INCLUDE_PADDING
                const int num_summands = KD * KW * KH;
#else
                const int num_summands = (ih_end - ih_start)
                        * (iw_end - iw_start) * (id_end - id_start);
#endif
                float d = 0;
                for (int id = id_start; id < id_end; ++id)
                    for (int ih = ih_start; ih < ih_end; ++ih) {
                        for (int iw = iw_start; iw < iw_end; ++iw) {
                            int src_off = SRC_OFF(mb, oc, id, ih, iw);
                            d += DATA_TO_REF(src[src_off]);
                        }
                    }
                dst[dst_off] = CONVERT_DATA_T(ROUND(d / num_summands));
#endif
            }
#endif
}
#endif
#if IS_BWD == 1
KERNEL_ATTR
__kernel void ref_pooling_bwd(__global DATA_T *diff_src, __global int *ws,
        __global DATA_T *diff_dst) {

#if USE_16C_UNROLL == 1
    const int mb = GWS_GET_MB();
    const int oc = GWS_GET_OC();
    const int id = GWS_GET_ID();
    const int ih = GWS_GET_IH();
    const int iw = GWS_GET_IW();

    diff_src += SRC_OFF(mb, oc, id, ih, iw);

    VECT_DATA_T blockS0 = 0.0f;
#ifdef MB16
    VECT_DATA_T blockS1 = 0.0f;
#endif

    for (int kd = 0; kd < KD; kd++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                int od = (id + PD - kd);
                int oh = (ih + PH - kh);
                int ow = (iw + PW - kw);
                if (oh % SH != 0 || ow % SW != 0 || od % SD != 0) continue;
                oh /= SH;
                ow /= SW;
                od /= SD;
                if (od < 0 || od >= OD) continue;
                if (oh < 0 || oh >= OH) continue;
                if (ow < 0 || ow >= OW) continue;

                const int dst_off = DST_OFF(mb, oc, od, oh, ow);
                VECT_DATA_T blockD0 = AS_VECT_DATA_T(VECT_BLOCK_READ((
                        const __global VECT_BLOCK_DATA_T *)&diff_dst[dst_off]));
#ifdef MB16
                VECT_DATA_T blockD1 = AS_VECT_DATA_T(
                        VECT_BLOCK_READ((const __global VECT_BLOCK_DATA_T
                                        *)&diff_dst[dst_off + 8 * 16]));
#endif

#if POOLING_MAX == 1
                VECT_INT_T blockWS0 = AS_VECT_INT_T(VECT_BLOCK_READ(
                        (const __global VECT_BLOCK_DATA_T *)&ws[dst_off]));
                VECT_INT_T blockCMP0 = isnotequal(
                        AS_VECT_DATA_T(blockWS0 - kd * KH * KW - kh * KW - kw),
                        (VECT_DATA_T)0.0f);
                blockD0 = select(blockD0, (VECT_DATA_T)0.0f, blockCMP0);

#ifdef MB16
                VECT_INT_T blockWS1 = AS_VECT_INT_T(
                        VECT_BLOCK_READ((const __global VECT_BLOCK_DATA_T
                                        *)&ws[dst_off + 8 * 16]));
                VECT_INT_T blockCMP1 = isnotequal(
                        AS_VECT_DATA_T(blockWS1 - kd * KH * KW - kh * KW - kw),
                        (VECT_DATA_T)0.0f);
                blockD1 = select(blockD1, (VECT_DATA_T)0.0f, blockCMP1);
#endif
#endif

#ifdef POOLING_AVG_EXCLUDE_PADDING
                const int id_start = max(id - kd, 0);
                const int ih_start = max(ih - kh, 0);
                const int iw_start = max(iw - kw, 0);
                const int id_end = min(id - kd + KD, ID);
                const int ih_end = min(ih - kh + KH, IH);
                const int iw_end = min(iw - kw + KW, IW);
                const DATA_T num_summands = (ih_end - ih_start)
                        * (iw_end - iw_start) * (id_end - id_start);
                blockD0 /= (VECT_DATA_T)num_summands;
#ifdef MB16
                blockD1 /= (VECT_DATA_T)num_summands;
#endif
#endif

                blockS0 += blockD0;
#ifdef MB16
                blockS1 += blockD1;
#endif
            }
#ifdef POOLING_AVG_INCLUDE_PADDING
    blockS0 /= KD * KH * KW;
#ifdef MB16
    blockS1 /= KD * KH * KW;
#endif
#endif
    VECT_BLOCK_WRITE((__global VECT_BLOCK_DATA_T *)&diff_src[0],
            AS_VECT_BLOCK_DATA_T(blockS0));
#ifdef MB16
    VECT_BLOCK_WRITE((__global VECT_BLOCK_DATA_T *)&diff_src[8 * 16],
            AS_VECT_BLOCK_DATA_T(blockS1));
#endif
#else
    const int mb = GWS_GET_MB();
    const int oc = GWS_GET_OC();

    for (int id = 0; id < ID; ++id)
        for (int ih = 0; ih < IH; ++ih)
            for (int iw = 0; iw < IW; ++iw) {
                uint diff_src_offset = SRC_OFF(mb, oc, id, ih, iw);
                diff_src[diff_src_offset] = 0;
            }

#if DT_BF16 == 1
    // For bfloat16, iterate over input to use a temporary float accumulator
    for (int id = 0; id < ID; ++id)
        for (int ih = 0; ih < IH; ++ih)
            for (int iw = 0; iw < IW; ++iw) {
                float s = 0;
                for (int kd = 0; kd < KD; ++kd)
                    for (int kh = 0; kh < KH; ++kh)
                        for (int kw = 0; kw < KW; ++kw) {
                            int _od = id + PD - kd;
                            int _oh = ih + PH - kh;
                            int _ow = iw + PW - kw;
                            if (_od % SD != 0 || _oh % SH != 0 || _ow % SW != 0)
                                continue;

                            int od = _od / SD;
                            int oh = _oh / SH;
                            int ow = _ow / SW;

                            if (od < 0 || od >= OD) continue;
                            if (oh < 0 || oh >= OH) continue;
                            if (ow < 0 || ow >= OW) continue;

                            const uint dst_off = DST_OFF(mb, oc, od, oh, ow);

#ifdef POOLING_MAX
                            const int index = ws[dst_off];

                            const int hw = index % (KW * KH);
                            const int w_kd = index / (KW * KH);
                            const int w_kw = hw % KW;
                            const int w_kh = hw / KW;
                            if (w_kd != kd || w_kh != kh || w_kw != kw)
                                continue;
#endif

#ifdef POOLING_MAX
                            const int denom = 1;
#elif defined(POOLING_AVG_INCLUDE_PADDING)
                            const int denom = KD * KH * KW;
#elif defined(POOLING_AVG_EXCLUDE_PADDING)
                            const int id_start = max(od * SD - PD, 0);
                            const int ih_start = max(oh * SH - PH, 0);
                            const int iw_start = max(ow * SW - PW, 0);
                            const int id_end = min(od * SD - PD + KD, ID);
                            const int ih_end = min(oh * SH - PH + KH, IH);
                            const int iw_end = min(ow * SW - PW + KW, IW);
                            const int denom = (ih_end - ih_start)
                                    * (iw_end - iw_start) * (id_end - id_start);
#endif
                            s += DATA_TO_REF(diff_dst[dst_off]) / denom;
                        }
                uint diff_src_offset = SRC_OFF(mb, oc, id, ih, iw);
                diff_src[diff_src_offset] = CONVERT_DATA_T(s);
            }
#else // DT_BF16 == 1
    for (int od = 0; od < OD; ++od)
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                const uint dst_off = DST_OFF(mb, oc, od, oh, ow);
                const float d = diff_dst[dst_off];

#if POOLING_MAX
                const int index = ws[dst_off];
                const int kd = index / (KW * KH);
                const int hw = index % (KW * KH);
                const int kw = hw % KW;
                const int kh = hw / KW;

                const int id = od * SD - PD + kd;
                const int ih = oh * SH - PH + kh;
                const int iw = ow * SW - PW + kw;

                if (id < 0 || id >= ID) continue;
                if (ih < 0 || ih >= IH) continue;
                if (iw < 0 || iw >= IW) continue;

                uint diff_src_offset = SRC_OFF(mb, oc, id, ih, iw);
                diff_src[diff_src_offset] += d;
#else
                const int id_start = max(od * SD - PD, 0);
                const int ih_start = max(oh * SH - PH, 0);
                const int iw_start = max(ow * SW - PW, 0);
                const int id_end = min(od * SD - PD + KD, ID);
                const int ih_end = min(oh * SH - PH + KH, IH);
                const int iw_end = min(ow * SW - PW + KW, IW);

#ifdef POOLING_AVG_INCLUDE_PADDING
                const int num_summands = KD * KW * KH;
#else
                const int num_summands = (ih_end - ih_start)
                        * (iw_end - iw_start) * (id_end - id_start);
#endif
                for (int id = id_start; id < id_end; ++id)
                    for (int ih = ih_start; ih < ih_end; ++ih)
                        for (int iw = iw_start; iw < iw_end; ++iw) {
                            const uint diff_src_offset
                                    = SRC_OFF(mb, oc, id, ih, iw);
                            diff_src[diff_src_offset] += d / num_summands;
                        }
#endif
            }
#endif
#endif
}
#endif
