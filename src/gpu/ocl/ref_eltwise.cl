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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

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

KERNEL_ATTR
__kernel void ref_eltwise_fwd(
        __global DATA_T *src, __global DATA_T *dst, float alpha, float beta) {
#if ZERO_PADDING
    int d0 = GWS_GET_D0();
    int d1 = GWS_GET_D1();
    int d2 = GWS_GET_D2();
    int d3 = GWS_GET_D3();
    int d4 = GWS_GET_D4();
    int d5 = GWS_GET_D5();

    const size_t data_off = DATA_OFF(d0, d1, d2, d3, d4, d5);
#else
    const size_t data_off = get_global_id(0)
#if GWS1 > 1
            + get_global_id(1) * GWS0
#endif
#if GWS2 > 1
            + get_global_id(2) * GWS0 * GWS1
#endif
            ;
#endif
    POST_OP_DATA_T tmp_s = DATA_TO_REF(src[data_off]);

    dst[data_off] = CONVERT_DATA_T(fwd_eltwise(tmp_s, alpha, beta, 1.0f));
}

#if DT_F32 == 1 || DT_BF16 == 1

KERNEL_ATTR
__kernel void ref_eltwise_bwd(__global DATA_T *src, __global DATA_T *diff_src,
        __global DATA_T *diff_dst, float alpha, float beta) {

    int d0 = GWS_GET_D0();
    int d1 = GWS_GET_D1();
    int d2 = GWS_GET_D2();
    int d3 = GWS_GET_D3();
    int d4 = GWS_GET_D4();
    int d5 = GWS_GET_D5();

    const size_t data_off = DATA_OFF(d0, d1, d2, d3, d4, d5);
    const size_t diff_data_off = DIFF_DATA_OFF(d0, d1, d2, d3, d4, d5);

    POST_OP_DATA_T tmp_dd = DATA_TO_REF(diff_dst[diff_data_off]);
    POST_OP_DATA_T tmp_s = DATA_TO_REF(src[data_off]);

    diff_src[diff_data_off]
            = CONVERT_DATA_T(bwd_eltwise(tmp_dd, tmp_s, alpha, beta));
}
#endif
