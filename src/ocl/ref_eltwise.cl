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

#include "ocl/ocl_post_ops.h"
#include "ocl/ocl_types.h"

#define DATA_OFF(x0, x1, x2, x3, x4, x5) \
    (((x0) % DATA_B0) * DATA_SB0 + ((x0) / DATA_B0) * DATA_S0 \
            + ((x1) % DATA_B1) * DATA_SB1 + ((x1) / DATA_B1) * DATA_S1 \
            + ((x2) % DATA_B2) * DATA_SB2 + ((x2) / DATA_B2) * DATA_S2 \
            + ((x3) % DATA_B3) * DATA_SB3 + ((x3) / DATA_B3) * DATA_S3 \
            + ((x4) % DATA_B4) * DATA_SB4 + ((x4) / DATA_B4) * DATA_S4 \
            + ((x5) % DATA_B5) * DATA_SB5 + ((x5) / DATA_B5) * DATA_S5)

#define DIFF_DATA_OFF(x0, x1, x2, x3, x4, x5) \
    (((x0) % DIFF_DATA_B0) * DIFF_DATA_SB0 \
            + ((x0) / DIFF_DATA_B0) * DIFF_DATA_S0 \
            + ((x1) % DIFF_DATA_B1) * DIFF_DATA_SB1 \
            + ((x1) / DIFF_DATA_B1) * DIFF_DATA_S1 \
            + ((x2) % DIFF_DATA_B2) * DIFF_DATA_SB2 \
            + ((x2) / DIFF_DATA_B2) * DIFF_DATA_S2 \
            + ((x3) % DIFF_DATA_B3) * DIFF_DATA_SB3 \
            + ((x3) / DIFF_DATA_B3) * DIFF_DATA_S3 \
            + ((x4) % DIFF_DATA_B4) * DIFF_DATA_SB4 \
            + ((x4) / DIFF_DATA_B4) * DIFF_DATA_S4 \
            + ((x5) % DIFF_DATA_B5) * DIFF_DATA_SB5 \
            + ((x5) / DIFF_DATA_B5) * DIFF_DATA_S5)

__kernel void ref_eltwise_fwd(
        __global DATA_T *src, __global DATA_T *dst, float alpha, float beta) {
    const int i = get_global_id(0);

    int d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0;
    d0 = i % DATA_D0;
#if NDIMS > 1
    d1 = (i / DATA_D0) % DATA_D1;
#endif
#if NDIMS > 2
    d2 = (i / (DATA_D0 * DATA_D1)) % DATA_D2;
#endif
#if NDIMS > 3
    d3 = (i / (DATA_D0 * DATA_D1 * DATA_D2)) % DATA_D3;
#endif
#if NDIMS > 4
    d4 = (i / (DATA_D0 * DATA_D1 * DATA_D2 * DATA_D3)) % DATA_D4;
#endif
#if NDIMS > 5
    d5 = (i / (DATA_D0 * DATA_D1 * DATA_D2 * DATA_D3 * DATA_D4)) % DATA_D5;
#endif

    const size_t data_off = DATA_OFF(d0, d1, d2, d3, d4, d5);

    POST_OP_DATA_T tmp_s = DATA_TO_REF(src[data_off]);

    dst[data_off] = CONVERT_DATA_T(fwd_eltwise(tmp_s, alpha, beta));
}

__kernel void ref_eltwise_bwd(__global DATA_T *src, __global DATA_T *diff_src,
        __global DATA_T *diff_dst, float alpha, float beta) {
    const int i = get_global_id(0);

    int d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0;
    d0 = i % DATA_D0;
#if NDIMS > 1
    d1 = (i / DATA_D0) % DATA_D1;
#endif
#if NDIMS > 2
    d2 = (i / (DATA_D0 * DATA_D1)) % DATA_D2;
#endif
#if NDIMS > 3
    d3 = (i / (DATA_D0 * DATA_D1 * DATA_D2)) % DATA_D3;
#endif
#if NDIMS > 4
    d4 = (i / (DATA_D0 * DATA_D1 * DATA_D2 * DATA_D3)) % DATA_D4;
#endif
#if NDIMS > 5
    d5 = (i / (DATA_D0 * DATA_D1 * DATA_D2 * DATA_D3 * DATA_D4)) % DATA_D5;
#endif

    const size_t data_off = DATA_OFF(d0, d1, d2, d3, d4, d5);
    const size_t diff_data_off = DIFF_DATA_OFF(d0, d1, d2, d3, d4, d5);

    POST_OP_DATA_T tmp_dd = DATA_TO_REF(diff_dst[diff_data_off]);
    POST_OP_DATA_T tmp_s = DATA_TO_REF(src[data_off]);

    diff_src[diff_data_off] = CONVERT_DATA_T(bwd_eltwise(tmp_dd, tmp_s, alpha, beta));
}
