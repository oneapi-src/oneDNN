/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "gpu/ocl/reorder_common.h"

KERNEL_ATTR
__kernel void generic_reorder(__global SRC_DATA_T *restrict src,
        __global DST_DATA_T *restrict dst, __global float *restrict src_scales,
        __global int *restrict src_zps, __global float *restrict dst_scales,
        __global int *restrict dst_zps, float sum_scale, int sum_zp) {

    const int src_zp = GET_SRC_ZP(src_zps);
    const int dst_zp = GET_DST_ZP(dst_zps);
    float src_scale = 1.0f;
    float dst_scale = 1.0f;

    src += SRC_OFFSET0;
    dst += DST_OFFSET0;

#define LOOP_NEST_LEVEL 4
    const uint sgId = get_sub_group_local_id();
    uint d[6]; // tensor coordinates from workitem ID
    uint b[6] = {0, 0, 0, 0, 0, 0}; // ajustment to coordinates per block (loop)

    d[0] = GWS_GET_D0();
    d[1] = GWS_GET_D1();
    d[2] = GWS_GET_D2();
    d[3] = GWS_GET_D3();
    d[4] = GWS_GET_D4();
    d[5] = GWS_GET_D5();

    // Dispatcher code does not allow vectorization of dimensions that are not
    // divisible by 16. As workaround, we cheat dispatcher by pretending
    // vectorized dim is 16x larger than it really is. Here we adjust its size
    // back to original.
    d[VECT_DIM] /= RESCALE_COEFF;
    // sg_off = offset into 'cache' local mem for given subgroup.
    // Local memory will be split by subgroups so that given address in local
    // can only be accessed by single subgroup. This lets us avoid barriers.
    const uint cache_size_per_sg = D_BLK_SIZE_0 * D_BLK_SIZE_1 * D_BLK_SIZE_2
            * D_BLK_SIZE_3 * VECT_SIZE;
    const uint sg_off = get_sub_group_id() * cache_size_per_sg;

    // TODO: decide whether to store cache as SRC_DATA_T or DST_DATA_T
    __local SRC_DATA_T cache[SG_PER_WG * cache_size_per_sg];
    uint iter[LOOP_NEST_LEVEL] = {0, 0, 0, 0};

// Loop across dimensions described in src_block.
// Example: block 2a4c2b would mean:
// for(a = 0..1) { for (c = 0..3) { for (b = 0..2) {}}}
#if S_BLK_SIZE_3 > 1
    for_(iter[3] = 0; iter[3] < S_BLK_SIZE_3; iter[3]++)
#endif
#if S_BLK_SIZE_2 > 1
    for_(iter[2] = 0; iter[2] < S_BLK_SIZE_2; iter[2]++)
#endif
#if S_BLK_SIZE_1 > 1
    for_(iter[1] = 0; iter[1] < S_BLK_SIZE_1; iter[1]++)
#endif
#if S_BLK_SIZE_0 > 1
    for_(iter[0] = 0; iter[0] < S_BLK_SIZE_0; iter[0]++)
#endif
    {
        // the same IDX could be in more than one loop, this makes offset calculation tricky
        b[0] = 0;
        b[1] = 0;
        b[2] = 0;
        b[3] = 0;
        b[4] = 0;
        b[5] = 0;
        b[S_BLK_IDX_0] += iter[0] * S_BLK_STEP_0;
        b[S_BLK_IDX_1] += iter[1] * S_BLK_STEP_1;
        b[S_BLK_IDX_2] += iter[2] * S_BLK_STEP_2;
        b[S_BLK_IDX_3] += iter[3] * S_BLK_STEP_3;

#if S_MOD_3 > 1
        b[S_IDX_3] += S_MUL_3 * ((sgId / S_DIV_3) % S_MOD_3);
#endif
#if S_MOD_2 > 1
        b[S_IDX_2] += S_MUL_2 * ((sgId / S_DIV_2) % S_MOD_2);
#endif
#if S_MOD_1 > 1
        b[S_IDX_1] += S_MUL_1 * ((sgId / S_DIV_1) % S_MOD_1);
#endif
#if S_MOD_0 > 1
        b[S_IDX_0] += S_MUL_0 * ((sgId / S_DIV_0) % S_MOD_0);
#endif

        // In a majority of cases, src_off contains neighboring values for
        // neighboring workitems. Exceptions occur when the src tensor has
        // additional striding in some dimension(s).
        const uint src_off = SRC_OFF(d[0] + b[0], d[1] + b[1], d[2] + b[2],
                d[3] + b[3], d[4] + b[4], d[5] + b[5]);

        // Data in cache (local mem) is organized as if it had 'fedcba' format
        // tag. This is neither src's nor dst's format so some performance is
        // wasted here, but otherwise the logic to calculate offsets would be
        // too complicated.
        uint cache_idx = sg_off + b[5] * CACHE_STRIDE_5 + b[4] * CACHE_STRIDE_4
                + b[3] * CACHE_STRIDE_3 + b[2] * CACHE_STRIDE_2
                + b[1] * CACHE_STRIDE_1 + b[0] * CACHE_STRIDE_0;
        const int pad_d0 = d[0] + b[0] >= SRC_D0;
        const int pad_d1 = NDIMS > 1 && d[1] + b[1] >= SRC_D1;
        const int pad_d2 = NDIMS > 2 && d[2] + b[2] >= SRC_D2;
        const int pad_d3 = NDIMS > 3 && d[3] + b[3] >= SRC_D3;
        const int pad_d4 = NDIMS > 4 && d[4] + b[4] >= SRC_D4;
        const int pad_d5 = NDIMS > 5 && d[5] + b[5] >= SRC_D5;
        const bool pad_sgid = sgId >= LIMIT_SSGID;
        const int pad
                = pad_d0 || pad_d1 || pad_d2 || pad_d3 || pad_d4 || pad_d5;
        if (!pad_sgid) {
            SRC_DATA_T src_tmp = pad ? 0 : src[src_off];
            cache[cache_idx] = src_tmp;
        }
    }
    for (uint i = 0; i < LOOP_NEST_LEVEL; i++) {
        iter[i] = 0;
    }
#if D_BLK_SIZE_3 > 1
    for_(iter[3] = 0; iter[3] < D_BLK_SIZE_3; iter[3]++)
#endif
#if D_BLK_SIZE_2 > 1
    for_(iter[2] = 0; iter[2] < D_BLK_SIZE_2; iter[2]++)
#endif
#if D_BLK_SIZE_1 > 1
    for_(iter[1] = 0; iter[1] < D_BLK_SIZE_1; iter[1]++)
#endif
#if D_BLK_SIZE_0 > 1
    for_(iter[0] = 0; iter[0] < D_BLK_SIZE_0; iter[0]++)
#endif
    {
        // the same IDX could be in more than one loop, this makes offset calculation tricky
        b[0] = 0;
        b[1] = 0;
        b[2] = 0;
        b[3] = 0;
        b[4] = 0;
        b[5] = 0;
        b[D_BLK_IDX_0] += iter[0] * D_BLK_STEP_0;
        b[D_BLK_IDX_1] += iter[1] * D_BLK_STEP_1;
        b[D_BLK_IDX_2] += iter[2] * D_BLK_STEP_2;
        b[D_BLK_IDX_3] += iter[3] * D_BLK_STEP_3;

#if D_MOD_3 > 1
        b[D_IDX_3] += D_MUL_3 * ((sgId / D_DIV_3) % D_MOD_3);
#endif
#if D_MOD_2 > 1
        b[D_IDX_2] += D_MUL_2 * ((sgId / D_DIV_2) % D_MOD_2);
#endif
#if D_MOD_1 > 1
        b[D_IDX_1] += D_MUL_1 * ((sgId / D_DIV_1) % D_MOD_1);
#endif
#if D_MOD_0 > 1
        b[D_IDX_0] += D_MUL_0 * ((sgId / D_DIV_0) % D_MOD_0);
#endif

        // In a majority of cases, dst_off contains neighboring values for
        // neighboring workitems. Exceptions occur when the dst tensor has
        // additional striding in some dimension(s), e.g., when reorder is used
        // in reference concat.
        const uint dst_off = DST_OFF(d[0] + b[0], d[1] + b[1], d[2] + b[2],
                d[3] + b[3], d[4] + b[4], d[5] + b[5]);

        DST_DATA_T dst_tmp;
        uint cache_idx = sg_off + b[5] * CACHE_STRIDE_5 + b[4] * CACHE_STRIDE_4
                + b[3] * CACHE_STRIDE_3 + b[2] * CACHE_STRIDE_2
                + b[1] * CACHE_STRIDE_1 + b[0] * CACHE_STRIDE_0;

        const int pad_d0 = d[0] + b[0] >= DST_PD0;
        const int pad_d1 = NDIMS > 1 && d[1] + b[1] >= DST_PD1;
        const int pad_d2 = NDIMS > 2 && d[2] + b[2] >= DST_PD2;
        const int pad_d3 = NDIMS > 3 && d[3] + b[3] >= DST_PD3;
        const int pad_d4 = NDIMS > 4 && d[4] + b[4] >= DST_PD4;
        const int pad_d5 = NDIMS > 5 && d[5] + b[5] >= DST_PD5;
        const bool pad_sgid = sgId >= LIMIT_DSGID;
        const int pad = pad_d0 || pad_d1 || pad_d2 || pad_d3 || pad_d4 || pad_d5
                || pad_sgid;

        if (!pad) {
            SRC_DATA_T from_cache = cache[cache_idx];
#if WITH_SUM_SCALE || WITH_SUM_ZPOINT
            // TODO: move to separate loop to enable burst reads from dst?
            dst_tmp = dst[dst_off];
#endif
#if WITH_SRC_SCALE
            uint src_scale_idx = SCALE_OFF(SRC, d[0] + b[0], d[1] + b[1],
                    d[2] + b[2], d[3] + b[3], d[4] + b[4], d[5] + b[5]);
            src_scale = src_scale_idx < SRC_NUM_SCALES
                    ? src_scales[src_scale_idx]
                    : 0.0;
#endif
#if WITH_DST_SCALE
            uint dst_scale_idx = SCALE_OFF(DST, d[0] + b[0], d[1] + b[1],
                    d[2] + b[2], d[3] + b[3], d[4] + b[4], d[5] + b[5]);
            dst_scale = dst_scale_idx < DST_NUM_SCALES
                    ? dst_scales[dst_scale_idx]
                    : 0.0;
#endif
            REORDER(dst_tmp, from_cache, src_scale, dst_scale, sum_scale,
                    src_zp, dst_zp, sum_zp);
            dst[dst_off] = dst_tmp;
        }
    }
}
