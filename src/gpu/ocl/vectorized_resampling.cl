/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#if IS_BWD == 1
inline float linear(int x, int fo, int fi) {
    return ((x + .5f) * fo / fi) - .5f;
}

inline int ceil_pos(float x) {
    return max((int)ceil(x), (int)0);
}

#define DST_MB_STRIDE(x) (x % DST_B0) * DST_SB0 + (x / DST_B0) * DST_S0
#define DST_C_STRIDE(x) (x % DST_B1) * DST_SB1 + (x / DST_B1) * DST_S1
#if NDIMS == 3
#define OW_STRIDE(x) (x % DST_B2) * DST_SB2 + (x / DST_B2) * DST_S2
#define OH_STRIDE(x) 0
#define OD_STRIDE(x) 0
#elif NDIMS == 4
#define OW_STRIDE(x) (x % DST_B3) * DST_SB3 + (x / DST_B3) * DST_S3
#define OH_STRIDE(x) (x % DST_B2) * DST_SB2 + (x / DST_B2) * DST_S2
#define OD_STRIDE(x) 0
#elif NDIMS == 5
#define OW_STRIDE(x) (x % DST_B4) * DST_SB4 + (x / DST_B4) * DST_S4
#define OH_STRIDE(x) (x % DST_B3) * DST_SB3 + (x / DST_B3) * DST_S3
#define OD_STRIDE(x) (x % DST_B2) * DST_SB2 + (x / DST_B2) * DST_S2
#endif

KERNEL_ATTR
__kernel void vectorized_resampling_bwd(
        __global DST_DATA_T *diff_src, __global const DATA_T *diff_dst) {
    const uint sglid = get_sub_group_local_id();

    const uint mb = (get_global_id(0) / MB_STRIDE);
    const uint c_start = (get_global_id(0) / GWS_SGS_DEFAULT) * GWS_SGS_DEFAULT
            * VECT_DT_N % PADDED_C;
    const uint c = c_start + sglid;
    const uint id = (get_global_id(0) / ID_STRIDE) % ID;
    const uint ih = (get_global_id(0) / IH_STRIDE) % IH;
    const uint iw = (get_global_id(0) / IW_STRIDE) % IW;
    const uint src_index = SRC_OFF(mb, c_start, id, ih, iw);

    VECT_DEF_ACC_DATA_T src_val = 0.0f;

#if RESAMPLING_ALG_NEAREST
    if (mb >= MB || c >= C) return;

    int od_start = ceil_pos(id * FD - .5f);
    int oh_start = ceil_pos(ih * FH - .5f);
    int ow_start = ceil_pos(iw * FW - .5f);
    int od_end = ceil_pos((id + 1.f) * FD - .5f);
    int oh_end = ceil_pos((ih + 1.f) * FH - .5f);
    int ow_end = ceil_pos((iw + 1.f) * FW - .5f);
    for_(int i = od_start; i < od_end; i++)
    for_(int j = oh_start; j < oh_end; j++)
    for (int k = ow_start; k < ow_end; k++) {
        const int dst_index = DST_OFF(mb, c_start, i, j, k);
#if VECT_DT_N == 1
        src_val += AS_VECT_DEF_ACC_DATA_T(diff_dst[dst_index + sglid]);
#else
        for (int i = 0; i < VECT_DT_N && c + GWS_SGS_DEFAULT * i < C; i++) {
            src_val[i] += TO_DEF_ACC_DATA_T(
                    diff_dst[dst_index + sglid + GWS_SGS_DEFAULT * i]);
        }
#endif
    }
#else
    int my_idx;
    {
        int ix, OX, IX;
        if (sglid < 4) {
            ix = id;
            OX = OD;
            IX = ID;
        } else if (sglid < 8) {
            ix = ih;
            OX = OH;
            IX = IH;
        } else {
            ix = iw;
            OX = OW;
            IX = IW;
        }

        // Right start operates on x-1
        if (sglid % 4 == 1) { ix -= 1; }

        // Left end operates on x+1
        if (sglid % 4 == 2) { ix += 1; }

        // This is the computationally expensive step
        float idx_intermediate = linear(ix, OX, IX);

        // Finalize the calculations by clamping
        if (sglid % 2 == 0) {
            my_idx = ceil_pos(idx_intermediate);
        } else {
            my_idx = (idx_intermediate < 0) ? 0 : (int)idx_intermediate + 1;
        }

        if (sglid % 4 >= 2) { my_idx = min(my_idx, OX); }

        // If left start ix==0 or right end ix==IX-1, include edges
        if (sglid % 4 == 0 && ix == 0) { my_idx = 0; }
        if (sglid % 4 == 3 && ix == IX - 1) { my_idx = OX; }
    }

    // Package into useful arrays
    // TODO: error when SGS < 16
    int od_start[2]
            = {sub_group_shuffle(my_idx, 0), sub_group_shuffle(my_idx, 1)};
    int od_end[2]
            = {sub_group_shuffle(my_idx, 2), sub_group_shuffle(my_idx, 3)};
    int oh_start[2]
            = {sub_group_shuffle(my_idx, 4), sub_group_shuffle(my_idx, 5)};
    int oh_end[2]
            = {sub_group_shuffle(my_idx, 6), sub_group_shuffle(my_idx, 7)};
    int ow_start[2]
            = {sub_group_shuffle(my_idx, 8), sub_group_shuffle(my_idx, 9)};
    int ow_end[2]
            = {sub_group_shuffle(my_idx, 10), sub_group_shuffle(my_idx, 11)};

    const int num_od_left = od_end[0] - od_start[0];
    const int num_od_right = od_end[1] - od_start[1];
    const int num_oh_left = oh_end[0] - oh_start[0];
    const int num_oh_right = oh_end[1] - oh_start[1];
    const int num_ow_left = ow_end[0] - ow_start[0];
    const int num_ow_right = ow_end[1] - ow_start[1];

    // Distribute linear calculations across SIMD channels
    float myres;
    {
        int ox, IX, OX;
        int offset = sglid;
        if (0 <= offset && offset < num_od_left) {
            OX = OD;
            IX = ID;
            ox = od_start[0] + offset;
        }
        offset -= num_od_left;
        if (0 <= offset && offset < num_od_right) {
            OX = OD;
            IX = ID;
            ox = od_start[1] + offset;
        }
        offset -= num_od_right;
        if (0 <= offset && offset < num_oh_left) {
            OX = OH;
            IX = IH;
            ox = oh_start[0] + offset;
        }
        offset -= num_oh_left;
        if (0 <= offset && offset < num_oh_right) {
            OX = OH;
            IX = IH;
            ox = oh_start[1] + offset;
        }
        offset -= num_oh_right;
        if (0 <= offset && offset < num_ow_left) {
            OX = OW;
            IX = IW;
            ox = ow_start[0] + offset;
        }
        offset -= num_ow_left;
        if (0 <= offset && offset < num_ow_right) {
            OX = OW;
            IX = IW;
            ox = ow_start[1] + offset;
        }

        // Do the (costly) linear calculations
        const float x = linear(ox, IX, OX);
        myres = fabs(x - trunc(x));
    }

    // Package to useful arrays
    float d_list[2][MAX_NUM_D];
    float h_list[2][MAX_NUM_H];
    float w_list[2][MAX_NUM_W];
    for (int d = 0; d < num_od_left; d++) {
        d_list[0][d] = 1.0f - sub_group_shuffle(myres, d);
    }
    int offset = num_od_left;
    for (int d = 0; d < num_od_right; d++) {
        d_list[1][d] = sub_group_shuffle(myres, d + offset);
    }
    offset += num_od_right;
    for (int h = 0; h < num_oh_left; h++) {
        h_list[0][h] = 1.0f - sub_group_shuffle(myres, h + offset);
    }
    offset += num_oh_left;
    for (int h = 0; h < num_oh_right; h++) {
        h_list[1][h] = sub_group_shuffle(myres, h + offset);
    }
    offset += num_oh_right;
    for (int w = 0; w < num_ow_left; w++) {
        w_list[0][w] = 1.0f - sub_group_shuffle(myres, w + offset);
    }
    offset += num_ow_left;
    for (int w = 0; w < num_ow_right; w++) {
        w_list[1][w] = sub_group_shuffle(myres, w + offset);
    }
    // Have to wait to drop out until shuffles are done
    if (mb >= MB || c >= C) return;

    const uint mb_c_off = DST_MB_STRIDE(mb) + DST_C_STRIDE(c_start);
    for_(int c1 = 0; c1 < 2; c1++)
    for (int od = od_start[c1], i = 0; i < MAX_NUM_D && od < od_end[c1];
            od++, i++) {
        const uint d_off = mb_c_off + OD_STRIDE(od);
        float Wid = d_list[c1][i];
        for_(int c2 = 0; c2 < 2; c2++)
        for (int oh = oh_start[c2], j = 0; j < MAX_NUM_H && oh < oh_end[c2];
                oh++, j++) {
            const uint h_off = d_off + OH_STRIDE(oh);
            float Wih = h_list[c2][j];
            unroll_for(int c3 = 0; c3 < 2; c3++)
                    unroll_for(int k = 0, ow = ow_start[c3];
                               k < MAX_NUM_W && ow < ow_end[c3]; k++, ow++) {
                const uint dst_off = h_off + OW_STRIDE(ow);
#if VECT_DT_N == 1
                VECT_DEF_ACC_DATA_T dst_val
                        = AS_VECT_DEF_ACC_DATA_T(diff_dst[dst_off + sglid]);
#else
                VECT_DEF_ACC_DATA_T dst_val = 0;
                for (int idx = 0;
                        idx < VECT_DT_N && c + GWS_SGS_DEFAULT * idx < C;
                        idx++) {
                    dst_val[idx] = TO_DEF_ACC_DATA_T(
                            diff_dst[dst_off + sglid + GWS_SGS_DEFAULT * idx]);
                }
#endif
                float Wiw = w_list[c3][k];
                src_val += dst_val * Wid * Wih * Wiw;
            }
        }
    }
#endif

#if VECT_DT_N == 1
    diff_src[src_index + sglid] = TO_DST(src_val);
#else
    for (int i = 0; i < VECT_DT_N && c + GWS_SGS_DEFAULT * i < C; i++) {
        diff_src[src_index + sglid + GWS_SGS_DEFAULT * i] = TO_DST(src_val[i]);
    }
#endif
}
#endif
