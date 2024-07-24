
/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/intel/ocl/layer_norm_common.h"

#if IS_FWD
KERNEL_ATTR
__kernel void simple_lnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DST_DATA_T *dst,
        __global WEI_DATA_T *scale, __global WEI_DATA_T *shift, float eps,
        __global float *src_scale, __global float *dst_scale) {

    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    if (x[0] >= DST_D0 || x[1] >= DST_D1 || x[2] >= DST_D2 || x[3] >= DST_D3) {
        int local_id = get_sub_group_local_id();
        for (int c = 0; c < C; c += SUB_GROUP_SIZE) {
            x[NDIMS - 1] = c + local_id;
            int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            dst[dst_off] = TO_DST(CONVERT_DATA_T(0.f));
        }
        return;
    }

    int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

    float v_mean = CALCULATE_STATS ? 0 : mean[s_off];
    float v_variance = CALCULATE_STATS ? 0 : variance[s_off];

    if (CALCULATE_STATS) {
        VECT_FLOAT_T v_acc = 0;
        for (int c = 0; c < C; c += SUB_GROUP_SIZE * VECT_DT_N) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            v_acc += CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                    (const __global BLOCK_DATA_T *)&src[src_off])));
        }
#if VECT_DT_N == 1
        v_mean = v_acc;
#else // VECT_DT_N == 1
        v_mean = 0;
        for (int i = 0; i < VECT_DT_N; ++i) {
            v_mean += v_acc[i];
        }
#endif // VECT_DT_N == 1

        float total_sum = sub_group_reduce_add(v_mean);
        v_mean = total_sum / C;

        v_acc = 0;
        VECT_FLOAT_T m = 0;

        for (int c = 0; c < C; c += SUB_GROUP_SIZE * VECT_DT_N) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            m = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                    (const __global BLOCK_DATA_T *)&src[src_off])));
            m -= v_mean;
            v_acc += m * m;
        }
#if VECT_DT_N == 1
        v_variance = v_acc;
#else // VECT_DT_N == 1
        v_variance = 0;
        for (int i = 0; i < VECT_DT_N; ++i) {
            v_variance += v_acc[i];
        }
#endif // VECT_DT_N == 1

        total_sum = sub_group_reduce_add(v_variance);
        v_variance = total_sum / C;
    }
    const float rsqrt_variance = rsqrt(v_variance + eps);

    int local_id = get_sub_group_local_id();
    for (int c = 0; c < C; c += SUB_GROUP_SIZE) {
        float sm = (scale ? CONVERT_WEI_FLOAT_T(scale[c + local_id]) : 1.0f)
                * rsqrt_variance;
        float sv = shift ? CONVERT_WEI_FLOAT_T(shift[c + local_id]) : 0.0f;

        x[NDIMS - 1] = c + local_id;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        float d = (sm * (SRC_TO_REF(src[src_off]) - v_mean) + sv);
#if WITH_SRC_SCALES
        d *= src_scale[0];
#endif
#if WITH_DST_SCALES
        d /= dst_scale[0];
#endif
        dst[dst_off] = TO_DST(d);
    }

    if (CALCULATE_STATS) {
        if (SAVE_STATS) {
            mean[s_off] = v_mean;
            variance[s_off] = v_variance;
        }
    }
}
#endif
