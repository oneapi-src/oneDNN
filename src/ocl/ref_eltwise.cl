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

#include "ocl/ocl_types.h"

#include "ocl/ocl_post_ops.h"

#if DT_BF16 == 1
#   define WITH_BF16 1
#else
#   define WITH_BF16 0
#endif

#define IN_OFF(x0, x1, x2, x3, x4, x5) ( \
    ((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 + \
    ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 + \
    ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 + \
    ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 + \
    ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4 + \
    ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)

__kernel void ref_eltwise_fwd(
        __global DATA_T *src, __global DATA_T *dst, float alpha, float beta) {
    const int i = get_global_id(0);

    int d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0;
    d0 = i % SRC_D0;
    #if NDIMS > 1
    d1 = (i / SRC_D0) % SRC_D1;
    #endif
    #if NDIMS > 2
    d2 = (i / (SRC_D0 * SRC_D1)) % SRC_D2;
    #endif
    #if NDIMS > 3
    d3 = (i / (SRC_D0 * SRC_D1 * SRC_D2)) % SRC_D3;
    #endif
    #if NDIMS > 4
    d4 = (i / (SRC_D0 * SRC_D1 * SRC_D2 * SRC_D3)) % SRC_D4;
    #endif
    #if NDIMS > 5
    d5 = (i / (SRC_D0 * SRC_D1 * SRC_D2 * SRC_D3 * SRC_D4)) % SRC_D5;
    #endif

    const size_t off = IN_OFF(d0,d1,d2,d3,d4,d5);

    POST_OP_DATA_T tmp_s = (WITH_BF16 == 1) ? convert_bf16_to_f32(src[off]) : src[off];

    dst[off]  = (WITH_BF16 == 1) ? convert_f32_to_bf16(fwd_eltwise(tmp_s, alpha, beta)) : fwd_eltwise(tmp_s, alpha, beta);
}

__kernel void ref_eltwise_bwd(__global DATA_T *src, __global DATA_T *diff_src,
        __global DATA_T *diff_dst, float alpha) {
    const int i = get_global_id(0);

    int d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0;
    d0 = i % SRC_D0;
    #if NDIMS > 1
    d1 = (i / SRC_D0) % SRC_D1;
    #endif
    #if NDIMS > 2
    d2 = (i / (SRC_D0 * SRC_D1)) % SRC_D2;
    #endif
    #if NDIMS > 3
    d3 = (i / (SRC_D0 * SRC_D1 * SRC_D2)) % SRC_D3;
    #endif
    #if NDIMS > 4
    d4 = (i / (SRC_D0 * SRC_D1 * SRC_D2 * SRC_D3)) % SRC_D4;
    #endif
    #if NDIMS > 5
    d5 = (i / (SRC_D0 * SRC_D1 * SRC_D2 * SRC_D3 * SRC_D4)) % SRC_D5;
    #endif

    const size_t off = IN_OFF(d0,d1,d2,d3,d4,d5);

    POST_OP_DATA_T tmp_dd = (WITH_BF16 == 1) ? convert_bf16_to_f32(diff_dst[off]) : diff_dst[off];
    POST_OP_DATA_T tmp_s = (WITH_BF16 == 1) ? convert_bf16_to_f32(src[off]): src[off];

    diff_src[off] = (WITH_BF16 == 1) ? convert_f32_to_bf16(bwd_eltwise(tmp_dd, tmp_s, alpha)) : bwd_eltwise(tmp_dd, tmp_s, alpha);
}
