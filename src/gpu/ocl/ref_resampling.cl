/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#if IS_FWD == 1
KERNEL_ATTR
__kernel void ref_resampling_fwd(
        __global const DATA_T *src, __global DST_DATA_T *dst POST_OP_ARGS) {
    const uint mb = GWS_GET_MB();
    const uint c = GWS_GET_C();
    const uint od = GWS_GET_OD();
    const uint oh = GWS_GET_OH();
    const uint ow = GWS_GET_OW();
    const float id = (od + .5f) * ID / OD;
    const float ih = (oh + .5f) * IH / OH;
    const float iw = (ow + .5f) * IW / OW;

    float result;
    const uint dst_index = DST_OFF(mb, c, od, oh, ow);

    if (mb >= DST_D0 || c >= DST_D1) {
        dst[dst_index] = TO_DST(0.0f);
        return;
    }

#if RESAMPLING_ALG_NEAREST
    const uint src_index = SRC_OFF(mb, c, (uint)id, (uint)ih, (uint)iw);
    result = CONVERT_FLOAT_T(src[src_index]);
#else
    const int id0 = max((int)floor(id - .5f), 0);
    const int id1 = min((int)ceil(id - .5f), ID - 1);
    const int ih0 = max((int)floor(ih - .5f), 0);
    const int ih1 = min((int)ceil(ih - .5f), IH - 1);
    const int iw0 = max((int)floor(iw - .5f), 0);
    const int iw1 = min((int)ceil(iw - .5f), IW - 1);

    const float wd[2] = {1.0f - fabs(id - .5f - id0), fabs(id - .5f - id0)};
    const float wh[2] = {1.0f - fabs(ih - .5f - ih0), fabs(ih - .5f - ih0)};
    const float ww[2] = {1.0f - fabs(iw - .5f - iw0), fabs(iw - .5f - iw0)};

    const int ih_arr[2] = {ih0, ih1};
    const int iw_arr[2] = {iw0, iw1};

    float cd[2][2];
    for_(int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
        cd[i][j] = CONVERT_FLOAT_T(
                           src[SRC_OFF(mb, c, id0, ih_arr[i], iw_arr[j])])
                        * wd[0]
                + CONVERT_FLOAT_T(
                          src[SRC_OFF(mb, c, id1, ih_arr[i], iw_arr[j])])
                        * wd[1];
    float ch[2];
    for (int i = 0; i < 2; i++)
        ch[i] = cd[0][i] * wh[0] + cd[1][i] * wh[1];

    result = ch[0] * ww[0] + ch[1] * ww[1];

#endif

    float sum_src;
#if WITH_SUM
    sum_src = DST_TO_REF(dst[dst_index]);
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
    APPLY_POST_OPS_SERIAL(result, float, sum_src, float, mb, 1, c, 1, po_d2, 1,
            po_d3, 1, po_d4, 1, 0, 1);
    dst[dst_index] = TO_DST(result);
}
#endif
#if IS_BWD == 1
float linear(int x, int fo, int fi) {
    return ((x + .5f) * fo / fi) - .5f;
}
KERNEL_ATTR
__kernel void ref_resampling_bwd(
        __global DATA_T *diff_src, __global const DST_DATA_T *diff_dst) {
#define CEIL(x) max((int)ceil(x), (int)0)
#define L(x, fo, fi) linear(x, fo, fi)
#define LS(x, fo, fi) CEIL(L(x, fo, fi))
#define RS(x, fo, fi) \
    L((int)x - 1, fo, fi) < 0 ? 0 : (int)(L((int)x - 1, fo, fi)) + 1
#define LE(x, fo, fi, lim) min(CEIL(L(x + 1, fo, fi)), (int)lim)
#define RE(x, fo, fi, lim) \
    min((L(x, fo, fi) < 0 ? 0 : (int)(L(x, fo, fi)) + 1), (int)lim)
    const uint mb = GWS_GET_MB();
    const uint c = GWS_GET_C();
    const uint id = GWS_GET_ID();
    const uint ih = GWS_GET_IH();
    const uint iw = GWS_GET_IW();
    const uint src_index = SRC_OFF(mb, c, id, ih, iw);

    if (mb >= DST_D0 || c >= DST_D1) {
        diff_src[src_index] = TO_DST(0.f);
        return;
    }
#if RESAMPLING_ALG_NEAREST
    int od_start = CEIL(id * FD - .5f);
    int oh_start = CEIL(ih * FH - .5f);
    int ow_start = CEIL(iw * FW - .5f);
    int od_end = CEIL((id + 1.f) * FD - .5f);
    int oh_end = CEIL((ih + 1.f) * FH - .5f);
    int ow_end = CEIL((iw + 1.f) * FW - .5f);
    float src_val = 0;
    for (int i = od_start; i < od_end; i++) {
        for (int j = oh_start; j < oh_end; j++) {
            for (int k = ow_start; k < ow_end; k++) {
                const int dst_index = DST_OFF(mb, c, i, j, k);
                src_val += DST_TO_REF(diff_dst[dst_index]);
            }
        }
    }
#else
    int left_sd = id == 0 ? 0 : LS(id, OD, ID);
    int left_sh = ih == 0 ? 0 : LS(ih, OH, IH);
    int left_sw = iw == 0 ? 0 : LS(iw, OW, IW);
    int right_sd = RS(id, OD, ID);
    int right_sh = RS(ih, OH, IH);
    int right_sw = RS(iw, OW, IW);
    int left_ed = LE(id, OD, ID, OD);
    int left_eh = LE(ih, OH, IH, OH);
    int left_ew = LE(iw, OW, IW, OW);
    int right_ed = id == (ID - 1) ? OD : RE(id, OD, ID, OD);
    int right_eh = ih == (IH - 1) ? OH : RE(ih, OH, IH, OH);
    int right_ew = iw == (IW - 1) ? OW : RE(iw, OW, IW, OW);
    int od_start[2] = {left_sd, right_sd};
    int oh_start[2] = {left_sh, right_sh};
    int ow_start[2] = {left_sw, right_sw};
    int od_end[2] = {left_ed, right_ed};
    int oh_end[2] = {left_eh, right_eh};
    int ow_end[2] = {left_ew, right_ew};
    float src_val = 0.0f;
    for (int c1 = 0; c1 < 2; c1++) {
        for (int c2 = 0; c2 < 2; c2++) {
            for (int c3 = 0; c3 < 2; c3++) {
                for (int i = od_start[c1]; i < od_end[c1]; i++) {
                    for (int j = oh_start[c2]; j < oh_end[c2]; j++) {
                        for (int k = ow_start[c3]; k < ow_end[c3]; k++) {
                            float dst_val = DST_TO_REF(
                                    diff_dst[DST_OFF(mb, c, i, j, k)]);
                            float d = L(i, ID, OD);
                            float h = L(j, IH, OH);
                            float w = L(k, IW, OW);
                            float Wid = c1 == 0 ? 1.f - fabs(d - (int)d)
                                                : fabs(d - (int)d);
                            float Wih = c2 == 0 ? 1.f - fabs(h - (int)h)
                                                : fabs(h - (int)h);
                            float Wiw = c3 == 0 ? 1.f - fabs(w - (int)w)
                                                : fabs(w - (int)w);
                            src_val += dst_val * Wid * Wih * Wiw;
                        }
                    }
                }
            }
        }
    }
#endif
#if DT_S32 == 1
    diff_src[src_index] = CONVERT_DATA_T(src_val);
#else // #if DT_S32 == 1
    diff_src[src_index] = TO_DATA_T(src_val);
#endif // #else // #if DT_S32 == 1
}
#endif
