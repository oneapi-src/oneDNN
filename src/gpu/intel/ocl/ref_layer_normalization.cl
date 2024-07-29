/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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
__kernel void ref_lnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DST_DATA_T *dst,
        __global WEI_DATA_T *scale, __global WEI_DATA_T *shift, float eps,
        __global float *src_scale, __global float *dst_scale) {

    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    if (x[0] >= DST_D0 || x[1] >= DST_D1 || x[2] >= DST_D2 || x[3] >= DST_D3) {
        for (int c = 0; c < C; ++c) {
            x[NDIMS - 1] = c;
            int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            dst[dst_off] = TO_DST(CONVERT_DATA_T(0.f));
        }
        return;
    }

    int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

    ACC_DATA_T v_mean = CALCULATE_STATS ? 0 : mean[s_off];
    ACC_DATA_T v_variance = CALCULATE_STATS ? 0 : variance[s_off];

    if (CALCULATE_STATS) {
        for (int c = 0; c < C; ++c) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            v_mean += SRC_TO_REF(src[src_off]);
        }
        v_mean /= C;

        for (int c = 0; c < C; ++c) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            ACC_DATA_T m = SRC_TO_REF(src[src_off]) - v_mean;
            v_variance += m * m;
        }
        v_variance /= C;
    }
    ACC_DATA_T rsqrt_variance = rsqrt(v_variance + eps);
    for (int c = 0; c < C; ++c) {
        ACC_DATA_T sm = (scale ? CONVERT_WEI_FLOAT_T(scale[c]) : 1.0f)
                * rsqrt_variance;
        ACC_DATA_T sv = shift ? CONVERT_WEI_FLOAT_T(shift[c]) : 0.0f;

        x[NDIMS - 1] = c;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        ACC_DATA_T d = (sm * (SRC_TO_REF(src[src_off]) - v_mean) + sv);

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
            mean[s_off] = convert_float(v_mean);
            variance[s_off] = convert_float(v_variance);
        }
    }
}

#else
#if USE_SCALE || USE_SHIFT
NAMED_KERNEL_ATTR(SCALESHIFT)
__kernel void ref_lnorm_bwd_scaleshift(__global SRC_DATA_T *src,
        __global float *mean, __global float *variance,
        __global DATA_T *diff_dst, __global WEI_DATA_T *diff_scale,
        __global WEI_DATA_T *diff_shift, float eps) {

    const int c = GWS_GET_C();
    int x[6] = {0};

    float diff_gamma = 0;
    float diff_beta = 0;

    for (x[0] = 0; x[0] < max(1, STAT_D0); ++x[0]) {
        for (x[1] = 0; x[1] < max(1, STAT_D1); ++x[1]) {
            for (x[2] = 0; x[2] < max(1, STAT_D2); ++x[2]) {
                for (x[3] = 0; x[3] < max(1, STAT_D3); ++x[3]) {
                    x[NDIMS - 1] = 0;
                    const int s_off
                            = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

                    x[NDIMS - 1] = c;
                    const int src_off
                            = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
                    const int dst_off
                            = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

                    const float inv_sqrt_variance
                            = rsqrt(variance[s_off] + eps);
                    const float dd = DST_TO_REF(diff_dst[dst_off]);

                    diff_gamma += (SRC_TO_REF(src[src_off]) - mean[s_off]) * dd
                            * inv_sqrt_variance;
                    diff_beta += dd;
                }
            }
        }
    }
    if (diff_scale) diff_scale[c] = CONVERT_WEI_DATA_T(diff_gamma);
    if (diff_shift) diff_shift[c] = CONVERT_WEI_DATA_T(diff_beta);
}

#endif

KERNEL_ATTR
__kernel void ref_lnorm_bwd(__global SRC_DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global WEI_DATA_T *scale, __global SRC_DATA_T *diff_src, float eps) {
    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    const int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
    const ACC_DATA_T mean_val = mean[s_off];

    const ACC_DATA_T inv_sqrt_variance = rsqrt(variance[s_off] + eps);
    ACC_DATA_T dd_gamma = 0;
    ACC_DATA_T dd_gamma_x = 0;

    if (CALCULATE_STATS) {
        for (int c = 0; c < C; ++c) {
            const ACC_DATA_T gamma
                    = scale ? CONVERT_WEI_FLOAT_T(scale[c]) : 1.0f;

            x[NDIMS - 1] = c;
            const int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            const int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            const ACC_DATA_T dd = DST_TO_REF(diff_dst[dst_off]);
            dd_gamma += dd * gamma;
            dd_gamma_x += dd * gamma * (SRC_TO_REF(src[src_off]) - mean_val);
        }
        dd_gamma_x *= inv_sqrt_variance;
    }

    for (int c = 0; c < C; ++c) {
        const ACC_DATA_T gamma = scale ? CONVERT_WEI_FLOAT_T(scale[c]) : 1.0f;

        x[NDIMS - 1] = c;
        const int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        const int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        ACC_DATA_T v_diff_src = DST_TO_REF(diff_dst[dst_off]) * gamma;
        if (CALCULATE_STATS) {
            v_diff_src -= dd_gamma / C
                    + (SRC_TO_REF(src[src_off]) - mean_val) * dd_gamma_x
                            * inv_sqrt_variance / C;
        }
        v_diff_src *= inv_sqrt_variance;
        diff_src[src_off] = TO_SRC(v_diff_src);
    }
}

#endif // IS_FWD
