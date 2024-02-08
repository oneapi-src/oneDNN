/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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
#include "gpu/ocl/types_interop.h"

NAMED_KERNEL_ATTR(STAT)
__kernel void lnorm_reusable_calc_mean(__global DATA_T *src,
        __global float *mean, dim_t reduce_size, dim_t reduce_stride,
        dispatch_gws_rt_params_t gws_params) {
    src = GWS_GET_BUFFER_POS_NAMED(SRC, STAT, gws_params, src);
    mean = GWS_GET_BUFFER_POS_NAMED(STAT, STAT, gws_params, mean);

    float sum = 0;
    for (off_t i = 0; i < reduce_size; i++) {
        sum += TO_DEF_ACC_DATA_T(src[i * (off_t)reduce_stride]);
    }

    *mean = sum / reduce_size;
}

NAMED_KERNEL_ATTR(STAT)
__kernel void lnorm_reusable_calc_var(__global DATA_T *src,
        __global float *mean, __global float *variance, dim_t reduce_size,
        dim_t reduce_stride, dispatch_gws_rt_params_t gws_params) {
    src = GWS_GET_BUFFER_POS_NAMED(SRC, STAT, gws_params, src);
    mean = GWS_GET_BUFFER_POS_NAMED(STAT, STAT, gws_params, mean);
    variance = GWS_GET_BUFFER_POS_NAMED(STAT, STAT, gws_params, variance);

    float mean_val = *mean;
    float sum = 0;
    for (off_t i = 0; i < reduce_size; i++) {
        DEF_ACC_DATA_T v0
                = TO_DEF_ACC_DATA_T(src[i * (off_t)reduce_stride]) - mean_val;
        sum += v0 * v0;
    }

    *variance = sum / reduce_size;
}

KERNEL_ATTR
__kernel void lnorm_reusable_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DST_DATA_T *dst,
        __global WEI_DATA_T *scale, __global WEI_DATA_T *shift, float eps,
        __global float *src_scale, __global float *dst_scale,
        dispatch_gws_rt_params_t gws_params) {
    mean = GWS_GET_BUFFER_POS(STAT, gws_params, mean);
    variance = GWS_GET_BUFFER_POS(STAT, gws_params, variance);

    scale = GWS_GET_BUFFER_POS(SS, gws_params, scale);
    shift = GWS_GET_BUFFER_POS(SS, gws_params, shift);

    src = GWS_GET_BUFFER_POS(SRC, gws_params, src);
    dst = GWS_GET_BUFFER_POS(DST, gws_params, dst);

    float sm = USE_SCALE ? WEI_TO_REF(*scale) : 1.0f;
    float sv = USE_SHIFT ? WEI_TO_REF(*shift) : 0.0f;
    DEF_ACC_DATA_T src_val = TO_DEF_ACC_DATA_T(*src);
    float sqrt_variance = 1.0f / sqrt(*variance + eps);
    float res = sm * (src_val - *mean) * sqrt_variance + sv;

    if (WITH_SRC_SCALES) res *= *src_scale;
    if (WITH_DST_SCALES) res /= *dst_scale;

    *dst = TO_DST(res);
}

//************ BWD kernels *************//

NAMED_KERNEL_ATTR(SS)
__kernel void lnorm_reusable_bwd_scaleshift(__global DST_DATA_T *src,
        __global float *mean, __global float *variance,
        __global DATA_T *diff_dst, __global WEI_DATA_T *diff_scale,
        __global WEI_DATA_T *diff_shift, dim_t stat_size, dim_t stat_stride,
        float eps, dispatch_gws_rt_params_t gws_params) {

    src = GWS_GET_BUFFER_POS_NAMED(SRC, SS, gws_params, src);
    diff_dst = GWS_GET_BUFFER_POS_NAMED(DST, SS, gws_params, diff_dst);

    diff_scale = GWS_GET_BUFFER_POS_NAMED(SS, SS, gws_params, diff_scale);
    diff_shift = GWS_GET_BUFFER_POS_NAMED(SS, SS, gws_params, diff_shift);

    float gamma = 0;
    float beta = 0;
    for (int i = 0; i < stat_size; i++) {
        off_t off = i * stat_stride;
        DEF_ACC_DATA_T dst_val = TO_DEF_ACC_DATA_T(diff_dst[off]);
        float src_val = DST_TO_REF(src[off]);

        beta += dst_val;
        gamma += dst_val * (src_val - mean[i]) / sqrt(variance[i] + eps);
    }

    if (diff_shift) *diff_shift = TO_WEI(beta);
    if (diff_scale) *diff_scale = TO_WEI(gamma);
}

NAMED_KERNEL_ATTR(STAT)
__kernel void lnorm_reusable_bwd(__global DST_DATA_T *src,
        __global DATA_T *diff_dst, __global DST_DATA_T *diff_src,
        __global WEI_DATA_T *scale, __global float *mean,
        __global float *variance, float eps, dim_t norm_stride, dim_t norm_size,
        char include_stats, dispatch_gws_rt_params_t gws_params) {

    // Cannot pass bools into kernels - we will only pass a 0 or 1
    ASSUME((include_stats == 0) || (include_stats == 1));

    src = GWS_GET_BUFFER_POS_NAMED(SRC, STAT, gws_params, src);
    diff_dst = GWS_GET_BUFFER_POS_NAMED(DST, STAT, gws_params, diff_dst);
    diff_src = GWS_GET_BUFFER_POS_NAMED(SRC, STAT, gws_params, diff_src);

    mean = GWS_GET_BUFFER_POS_NAMED(STAT, STAT, gws_params, mean);
    variance = GWS_GET_BUFFER_POS_NAMED(STAT, STAT, gws_params, variance);

    float mean_val = *mean;
    float var_val = *variance;

    float inv_var = rsqrt(var_val + eps);

    // If we calculate mean/variance from src, include this contribution in diff_src
    DEF_ACC_DATA_T mu = 0;
    DEF_ACC_DATA_T sigma = 0;
    if (include_stats) {
        for (int i = 0; i < norm_size; i++) {
            off_t off = i * norm_stride;
            DEF_ACC_DATA_T dst_val = TO_DEF_ACC_DATA_T(diff_dst[off]);
            float src_val = DST_TO_REF(src[off]);
            float scale_val = (scale ? CONVERT_WEI_FLOAT_T(scale[i]) : 1.0f);

            mu += dst_val * scale_val;
            sigma += dst_val * scale_val * src_val;
        }
        sigma -= mu * mean_val;
        sigma *= inv_var * inv_var;
    }

    // Apply these stats to the entirety of the norm dim
    for (int i = 0; i < norm_size; i++) {
        off_t off = i * norm_stride;
        DEF_ACC_DATA_T dst_val = TO_DEF_ACC_DATA_T(diff_dst[off]);
        float src_val = DST_TO_REF(src[off]);
        float scale_val = (scale ? CONVERT_WEI_FLOAT_T(scale[i]) : 1.0f);

        DEF_ACC_DATA_T res = dst_val * scale_val;

        if (include_stats) {
            res -= (mu + sigma * (src_val - mean_val)) / norm_size;
        }

        diff_src[off] = TO_DST(inv_var * res);
    }
}
