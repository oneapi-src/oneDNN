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
#include "gpu/ocl/reorder_common.h"

KERNEL_ATTR
__kernel void ref_reorder(__global SRC_DATA_T *restrict src,
        __global DST_DATA_T *restrict dst, __global float *restrict src_scales,
        __global int *restrict src_zps, __global float *restrict dst_scales,
        __global int *dst_zps, float sum_scale, int sum_zp) {

    const int src_zp = GET_SRC_ZP(src_zps);
    const int dst_zp = GET_DST_ZP(dst_zps);
    float src_scale = 1.0f;
    float dst_scale = 1.0f;

    src += SRC_OFFSET0;
    dst += DST_OFFSET0;

    const int d0_blk_start = GWS_GET_D0();
    const int d1_blk_start = GWS_GET_D1();
    const int d2_blk_start = GWS_GET_D2();
    const int d3_blk_start = GWS_GET_D3();
    const int d4_blk_start = GWS_GET_D4();
    const int d5_blk_start = GWS_GET_D5();

    const int d0_blk_end = d0_blk_start + GWS_GET_D0_BLOCK();
    const int d1_blk_end = d1_blk_start + GWS_GET_D1_BLOCK();
    const int d2_blk_end = d2_blk_start + GWS_GET_D2_BLOCK();
    const int d3_blk_end = d3_blk_start + GWS_GET_D3_BLOCK();
    const int d4_blk_end = d4_blk_start + GWS_GET_D4_BLOCK();
    const int d5_blk_end = d5_blk_start + GWS_GET_D5_BLOCK();

    for_(int d0 = d0_blk_start; d0 < d0_blk_end; ++d0)
    for_(int d1 = d1_blk_start; d1 < d1_blk_end; ++d1)
    for_(int d2 = d2_blk_start; d2 < d2_blk_end; ++d2)
    for_(int d3 = d3_blk_start; d3 < d3_blk_end; ++d3)
    for_(int d4 = d4_blk_start; d4 < d4_blk_end; ++d4)
    for (int d5 = d5_blk_start; d5 < d5_blk_end; ++d5) {
        const int src_off = SRC_OFF(d0, d1, d2, d3, d4, d5);
        const int dst_off = DST_OFF(d0, d1, d2, d3, d4, d5);
#if PAD_FILL_ZERO == 1
        int pad_d0 = d0 >= SRC_D0;
        int pad_d1 = NDIMS > 1 && d1 >= SRC_D1;
        int pad_d2 = NDIMS > 2 && d2 >= SRC_D2;
        int pad_d3 = NDIMS > 3 && d3 >= SRC_D3;
        int pad_d4 = NDIMS > 4 && d4 >= SRC_D4;
        int pad_d5 = NDIMS > 5 && d5 >= SRC_D5;
        if (pad_d0 || pad_d1 || pad_d2 || pad_d3 || pad_d4 || pad_d5) {
            dst[dst_off] = 0;
            continue;
        }
#endif
#if WITH_SRC_SCALE
        src_scale = src_scales[SCALE_OFF(SRC, d0, d1, d2, d3, d4, d5)];
#endif
#if WITH_DST_SCALE
        dst_scale = dst_scales[SCALE_OFF(DST, d0, d1, d2, d3, d4, d5)];
#endif
        REORDER(dst[dst_off], src[src_off], src_scale, dst_scale, sum_scale,
                src_zp, dst_zp, sum_zp);
    }
}
