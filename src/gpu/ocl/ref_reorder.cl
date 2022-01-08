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
        __global DST_DATA_T *restrict dst, float alpha, float beta,
        __global float *restrict scales) {

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

    for (int d0 = d0_blk_start; d0 < d0_blk_end; ++d0) {
        for (int d1 = d1_blk_start; d1 < d1_blk_end; ++d1) {
            for (int d2 = d2_blk_start; d2 < d2_blk_end; ++d2) {
                for (int d3 = d3_blk_start; d3 < d3_blk_end; ++d3) {
                    for (int d4 = d4_blk_start; d4 < d4_blk_end; ++d4) {
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
                            if (pad_d0 || pad_d1 || pad_d2 || pad_d3 || pad_d4
                                    || pad_d5) {
                                dst[dst_off] = 0;
                                continue;
                            }
#endif
#if SCALE_QUANT
                            alpha = scales[SCALE_OFF(d0, d1, d2, d3, d4, d5)];
#endif
                            REORDER(dst[dst_off], src[src_off], alpha, beta);
                        }
                    }
                }
            }
        }
    }
}
