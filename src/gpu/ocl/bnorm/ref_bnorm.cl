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

#include "gpu/ocl/dispatch.h"
#include "gpu/ocl/ocl_types.h"

int reduce_index(int x[5]) {
    int dim[5] = {MB, IC, ID, IH, IW};
    dim[REDUCE_DIM_IDX] = 1;
    return x[0] * (dim[2] * dim[3] * dim[4]) + x[2] * (dim[3] * dim[4])
            + x[3] * dim[4] + x[4];
}

#if IS_FWD == 1
#if CALCULATE_STATS == 1

NAMED_KERNEL_ATTR(CALC)
__kernel void calculate_mean(__global DATA_T *src, __global float *mean) {
    int x[5];
    x[0] = GWS_GET_STAT_MB();
    x[1] = GWS_GET_STAT_IC();
    x[2] = GWS_GET_STAT_ID();
    x[3] = GWS_GET_STAT_IH();
    x[4] = GWS_GET_STAT_IW();
    float sum = 0;
    for (int i = 0; i < REDUCE_DIM; i++) {
        x[REDUCE_DIM_IDX] = i;
        sum += TO_DEF_ACC_DATA_T(src[SRC_OFF(x[0], x[1], x[2], x[3], x[4])]);
    }
    x[REDUCE_DIM_IDX] = 0;
    int reduce_idx = reduce_index(x);
    mean[reduce_idx * IC + x[1]] = sum;
}

NAMED_KERNEL_ATTR(CALC)
__kernel void calculate_variance(
        __global DATA_T *src, __global float *mean, __global float *variance) {
    int x[5];
    x[0] = GWS_GET_STAT_MB();
    x[1] = GWS_GET_STAT_IC();
    x[2] = GWS_GET_STAT_ID();
    x[3] = GWS_GET_STAT_IH();
    x[4] = GWS_GET_STAT_IW();
    float sum = 0;
    for (int i = 0; i < REDUCE_DIM; i++) {
        x[REDUCE_DIM_IDX] = i;
        DEF_ACC_DATA_T v0
                = TO_DEF_ACC_DATA_T(src[SRC_OFF(x[0], x[1], x[2], x[3], x[4])])
                - mean[x[1]];
        sum += v0 * v0;
    }
    variance += MB * ID * IH * IW * IC / REDUCE_DIM;
    x[REDUCE_DIM_IDX] = 0;
    int reduce_idx = reduce_index(x);

    variance[reduce_idx * IC + x[1]] = sum;
}

NAMED_KERNEL_ATTR(REDUCE)
__kernel void reduce_mean(__global float *reduce_temp, __global float *mean) {
    const int c = GWS_GET_REDUCE_STAT_IC();
    reduce_temp += c;
    float sum = 0.0f;
    int reduce_size = MB * ID * IH * IW / REDUCE_DIM;
    for (int i = 0; i < reduce_size; i++) {
        sum += reduce_temp[i * IC];
    }
    mean[c] = sum / (MB * ID * IH * IW);
}

NAMED_KERNEL_ATTR(REDUCE)
__kernel void reduce_variance(
        __global float *reduce_temp, __global float *variance) {
    const int c = GWS_GET_REDUCE_STAT_IC();
#if SAVE_STATS == 0
    variance += IC;
#endif
    float sum = 0.0f;
    int reduce_size = MB * ID * IH * IW / REDUCE_DIM;
    reduce_temp += reduce_size * IC + c;
    for (int i = 0; i < reduce_size; i++)
        sum += reduce_temp[i * IC];

    variance[c] = sum / (MB * ID * IH * IW);
}

#endif

KERNEL_ATTR
__kernel void ref_bnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst, __global float *scale,
        __global float *shift, __global char *ws, float eps,
        __global DATA_T *src_add, float relu_alpha) {
    const int n = GWS_GET_MB();
    const int c = GWS_GET_IC();
    const int d = GWS_GET_ID();
    const int h = GWS_GET_IH();
    const int w = GWS_GET_IW();
#if USE_SCALE == 1
    float sm = scale[c];
#else
    float sm = 1;
#endif
#if USE_SHIFT == 1
    float sv = shift[c];
#else
    float sv = 0;
#endif

#if SAVE_STATS == 0 && CALCULATE_STATS == 1
    variance += IC;
#endif
    float v_mean = mean[c];
    float v_variance = variance[c];
    const int off = SRC_OFF(n, c, d, h, w);
    float v0 = TO_DEF_ACC_DATA_T(src[off]);
    float sqrt_variance = 1.0f / sqrt(v_variance + eps);
    float bn_res = sm * (v0 - v_mean) * sqrt_variance + sv;
#if FUSE_BN_ADD_RELU == 1
    bn_res += TO_DEF_ACC_DATA_T(src_add[off]);
#endif

#if FUSE_BN_RELU == 1
    if (bn_res <= 0) {
        bn_res = 0;
#if IS_TRAINING == 1
        ws[off] = 0;
    } else {
        ws[off] = -1;
#endif
    }
#endif

#if WITH_RELU
#if WITH_LEAKY_RELU
    if (bn_res < 0) { bn_res *= relu_alpha; }
#else
    bn_res = max(bn_res, 0.0f);
#endif //WITH_LEAKY_RELU
#endif //WITH_RELU

    dst[off] = TO_DATA_T(bn_res);
}
#endif

#if IS_BWD == 1

NAMED_KERNEL_ATTR(CALC)
__kernel void calculate_stats(__global DATA_T *src, __global float *mean,
        __global DATA_T *diff_dst, __global char *ws,
        __global float *reduce_temp) {
    float diff_gamma = 0;
    float diff_beta = 0;
    int x[5];
    x[0] = GWS_GET_STAT_MB();
    x[1] = GWS_GET_STAT_IC();
    x[2] = GWS_GET_STAT_ID();
    x[3] = GWS_GET_STAT_IH();
    x[4] = GWS_GET_STAT_IW();
    for (int i = 0; i < REDUCE_DIM; i++) {
        x[REDUCE_DIM_IDX] = i;
        int off = SRC_OFF(x[0], x[1], x[2], x[3], x[4]);
        float dd = CONVERT_FLOAT_T(diff_dst[off]);
#if FUSE_BN_RELU == 1
        if (!ws[off]) dd = 0;
#endif
        diff_gamma += (CONVERT_FLOAT_T(src[off]) - mean[x[1]]) * dd;
        diff_beta += dd;
    }

    int ss_off = MB * ID * IH * IW * IC / REDUCE_DIM;
    x[REDUCE_DIM_IDX] = 0;
    int reduce_idx = reduce_index(x);

    reduce_temp[reduce_idx * IC + x[1]] = diff_gamma;
    reduce_temp[ss_off + reduce_idx * IC + x[1]] = diff_beta;
}

NAMED_KERNEL_ATTR(REDUCE)
__kernel void reduce_stats(__global float *reduce_temp,
        __global float *diff_scale, __global float *diff_shift,
        __global float *variance, float eps) {
    const int c = GWS_GET_REDUCE_STAT_IC();
    float diff_gamma = 0.0f;
    float diff_beta = 0.0f;
    int reduce_size = MB * ID * IH * IW / REDUCE_DIM;

    for (int i = 0; i < reduce_size; i++) {
        diff_gamma += reduce_temp[c + i * IC];
        diff_beta += reduce_temp[IC * reduce_size + c + i * IC];
    }
    float sqrt_variance = 1.0f / sqrt(variance[c] + eps);

    diff_scale[c] = diff_gamma * sqrt_variance;
#if USE_SHIFT == 1
    diff_shift[c] = diff_beta;
#else
    // When USE_SHIFT == 0, `diff_shift` is a second part of reduce_temp
    diff_shift[IC * reduce_size + c] = diff_beta;
#endif
}

KERNEL_ATTR
__kernel void ref_bnorm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scale, __global char *ws, __global DATA_T *diff_src,
        __global float *diff_scale, __global float *diff_shift, float eps,
        __global DATA_T *diff_src_add) {

    const int n = GWS_GET_MB();
    const int c = GWS_GET_IC();
    const int d = GWS_GET_ID();
    const int h = GWS_GET_IH();
    const int w = GWS_GET_IW();
    float v_variance = variance[c];
    float sqrt_variance = 1.0f / sqrt(v_variance + eps);
#if USE_SCALE == 1
    float gamma = scale[c];
#else
    float gamma = 1;
#endif
#if CALCULATE_STATS == 1
    float v_mean = mean[c];
    float diff_gamma = diff_scale[c];
#if USE_SHIFT == 1
    float diff_beta = diff_shift[c];
#else
    int reduce_size = MB * ID * IH * IW / REDUCE_DIM;
    float diff_beta = diff_shift[reduce_size * IC + c];
#endif // #if USE_SHIFT == 1
#endif

    const int off = SRC_OFF(n, c, d, h, w);
    float dd = TO_DEF_ACC_DATA_T(diff_dst[off]);
#if FUSE_BN_RELU == 1
    if (!ws[off]) dd = 0;
#if FUSE_BN_ADD_RELU == 1
    diff_src_add[off] = TO_DATA_T(dd);
#endif
#endif

    float v_diff_src = dd;
#if CALCULATE_STATS == 1
    v_diff_src -= diff_beta / (MB * ID * IH * IW)
            + (CONVERT_FLOAT_T(src[off]) - v_mean) * diff_gamma * sqrt_variance
                    / (MB * ID * IH * IW);
#endif
    v_diff_src *= gamma * sqrt_variance;

    diff_src[off] = TO_DATA_T(v_diff_src);
}
#endif
