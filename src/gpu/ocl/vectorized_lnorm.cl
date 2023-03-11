/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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
#include "gpu/ocl/ocl_types.h"

#undef SRC_OFF
#undef DST_OFF
#define SRC_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC, x0, x1, x2, x3, x4, x5)
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)
#define STAT_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(STAT, x0, x1, x2, x3, x4, x5)

#if IS_FWD
#if VECT_DT_N == 1
#define CALC_V_STAT(v, acc) v = acc;
#else
#define CALC_V_STAT(v, acc) \
    v = 0; \
    for (int i = 0; i < VECT_DT_N; ++i) { \
        v += acc[i]; \
    }
#endif
#define LOAD_VECT_FLOAT(ptr) \
    AS_VECT_FLOAT_T(VECT_UINT_READ((const __global uint *)(ptr)))

#define STORE_VECT_DATA(ptr, val) \
    VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)(ptr), \
            AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(val)))
#define VLEN_C (C / (SUB_GROUP_SIZE * VECT_DT_N))

KERNEL_ATTR
__kernel void vectorized_lnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst, __global float *scale,
        __global float *shift, float eps) {

    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    int simd_id = get_sub_group_local_id();
    int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

    float v_mean = CALCULATE_STATS ? 0 : mean[s_off];
    float v_variance = CALCULATE_STATS ? 0 : variance[s_off];

    // Key feature of this version is single reading src data, keeping it in
    // v_src buffer and reusing this buffer for stats calculation.
    VECT_FLOAT_T v_src[VLEN_C];
    for (int c = 0; c < VLEN_C; c++) {
        x[NDIMS - 1] = c * SUB_GROUP_SIZE * VECT_DT_N;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        v_src[c] = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off])));
    }
    if (CALCULATE_STATS) {
        VECT_FLOAT_T v_acc = 0;
        for (int c = 0; c < VLEN_C; c++) {
            v_acc += v_src[c];
        }
        CALC_V_STAT(v_mean, v_acc);
        v_mean = sub_group_reduce_add(v_mean) / C;

        v_acc = 0;
        VECT_FLOAT_T m = 0;
        for (int c = 0; c < VLEN_C; c++) {
            m = v_src[c] - v_mean;
            v_acc += m * m;
        }
        CALC_V_STAT(v_variance, v_acc);
        v_variance = sub_group_reduce_add(v_variance) / C;
    }
    float sqrt_variance = sqrt(v_variance + eps);

    for (int c = 0; c < VLEN_C; c++) {
        VECT_FLOAT_T sm
#if USE_SCALE
                = LOAD_VECT_FLOAT(&scale[c * SUB_GROUP_SIZE * VECT_DT_N])
                / sqrt_variance;
#else
                = 1.0f / sqrt_variance;
#endif
        VECT_FLOAT_T sv
#if USE_SHIFT
                = LOAD_VECT_FLOAT(&shift[c * SUB_GROUP_SIZE * VECT_DT_N]);
#else
                = 0.0f;
#endif
        x[NDIMS - 1] = c * SUB_GROUP_SIZE * VECT_DT_N;
        int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        VECT_FLOAT_T v_dst = sm * (v_src[c] - v_mean) + sv;
        STORE_VECT_DATA(&dst[dst_off], v_dst);
    }
    if (CALCULATE_STATS && SAVE_STATS) {
        mean[s_off] = v_mean;
        variance[s_off] = v_variance;
    }
}
#endif
