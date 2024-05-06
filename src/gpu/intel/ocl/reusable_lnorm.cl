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

#include "gpu/intel/ocl/dispatch.h"
#include "gpu/intel/ocl/ocl_io.h"
#include "gpu/intel/ocl/types_interop.h"

NAMED_KERNEL_ATTR(STAT)
__kernel void lnorm_reusable_calc_mean(__global SRC_STAT_DT *src,
        __global float *mean, dim_t reduce_size, dim_t reduce_stride,
        dispatch_gws_rt_params_t gws_params) {
    src = GWS_GET_BUFFER_POS_NAMED(SRC, STAT, gws_params, src);
    mean = GWS_GET_BUFFER_POS_NAMED(STAT, STAT, gws_params, mean);

    ACC_DT sum = SPECIAL(ACC_DT, zero);
    for (off_t i = 0; i < reduce_size; i++) {
        ACC_DT src_val;
        load(&src_val, src + i * (off_t)reduce_stride);
        sum += src_val;
    }

    float res = into_float(sum) / reduce_size;
    write(mean, &res);
}

NAMED_KERNEL_ATTR(STAT)
__kernel void lnorm_reusable_calc_var(__global SRC_STAT_DT *src,
        __global float *mean, __global float *variance, dim_t reduce_size,
        dim_t reduce_stride, dispatch_gws_rt_params_t gws_params) {
    src = GWS_GET_BUFFER_POS_NAMED(SRC, STAT, gws_params, src);
    mean = GWS_GET_BUFFER_POS_NAMED(STAT, STAT, gws_params, mean);
    variance = GWS_GET_BUFFER_POS_NAMED(STAT, STAT, gws_params, variance);

    float mean_val;
    load(&mean_val, mean);
    float sum = 0.0f;
    for (off_t i = 0; i < reduce_size; i++) {
        ACC_DT src_val;
        load(&src_val, src + i * (off_t)reduce_stride);
        float v0 = src_val - mean_val;
        sum += v0 * v0;
    }

    float res = sum / reduce_size;
    write(variance, &res);
}

KERNEL_ATTR
__kernel void lnorm_reusable_fwd(__global SRC_DT *src, __global float *mean,
        __global float *variance, __global DST_DT *dst, __global SS_DT *scale,
        __global SS_DT *shift, float eps, __global float *src_scale,
        __global float *dst_scale, dispatch_gws_rt_params_t gws_params) {
    mean = GWS_GET_BUFFER_POS(STAT, gws_params, mean);
    variance = GWS_GET_BUFFER_POS(STAT, gws_params, variance);

    scale = GWS_GET_BUFFER_POS(SS, gws_params, scale);
    shift = GWS_GET_BUFFER_POS(SS, gws_params, shift);

    src = GWS_GET_BUFFER_POS(SRC, gws_params, src);
    dst = GWS_GET_BUFFER_POS(DST, gws_params, dst);

    float sm = 1.0f;
    float sv = 0.0f;
    if (USE_SCALE) load(&sm, scale);
    if (USE_SHIFT) load(&sv, shift);
    float src_val, var_val, mean_val;
    load(&src_val, src);
    load(&var_val, variance);
    load(&mean_val, mean);
    float sqrt_variance = 1.0f / sqrt(var_val + eps);
    float res = sm * (src_val - mean_val) * sqrt_variance + sv;

    if (WITH_SRC_SCALES) {
        float src_scale_val;
        load(&src_scale_val, src_scale);
        res *= src_scale_val;
    }
    if (WITH_DST_SCALES) {
        float dst_scale_val;
        load(&dst_scale_val, dst_scale);
        res /= dst_scale_val;
    }

    write(dst, &res);
}

//************ BWD kernels *************//

NAMED_KERNEL_ATTR(SS)
__kernel void lnorm_reusable_bwd_scaleshift(__global SRC_SS_DT *src,
        __global float *mean, __global float *variance,
        __global DST_SS_DT *diff_dst, __global SS_SS_DT *diff_scale,
        __global SS_SS_DT *diff_shift, dim_t stat_size, dim_t stat_stride,
        float eps, dispatch_gws_rt_params_t gws_params) {

    src = GWS_GET_BUFFER_POS_NAMED(SRC, SS, gws_params, src);
    diff_dst = GWS_GET_BUFFER_POS_NAMED(DST, SS, gws_params, diff_dst);

    diff_scale = GWS_GET_BUFFER_POS_NAMED(SS, SS, gws_params, diff_scale);
    diff_shift = GWS_GET_BUFFER_POS_NAMED(SS, SS, gws_params, diff_shift);

    float gamma = 0.0f;
    float beta = 0.0f;
    for (int i = 0; i < stat_size; i++) {
        off_t off = i * stat_stride;
        ACC_BWD_DT dst_val;
        load(&dst_val, diff_dst + off);
        ACC_DT src_val;
        load(&src_val, src + off);

        float mean_val, var_val;
        load(&mean_val, mean + i);
        load(&var_val, variance + i);

        beta += dst_val;
        gamma += dst_val * (src_val - mean_val) / sqrt(var_val + eps);
    }

    if (diff_shift) write(diff_shift, &beta);
    if (diff_scale) write(diff_scale, &gamma);
}

NAMED_KERNEL_ATTR(STAT)
__kernel void lnorm_reusable_bwd(__global SRC_STAT_DT *src,
        __global DST_STAT_DT *diff_dst, __global SRC_STAT_DT *diff_src,
        __global SS_DT *scale, __global float *mean, __global float *variance,
        float eps, dim_t norm_stride, dim_t norm_size, char include_stats,
        dispatch_gws_rt_params_t gws_params) {

    // Cannot pass bools into kernels - we will only pass a 0 or 1
    ASSUME((include_stats == 0) || (include_stats == 1));

    src = GWS_GET_BUFFER_POS_NAMED(SRC, STAT, gws_params, src);
    diff_dst = GWS_GET_BUFFER_POS_NAMED(DST, STAT, gws_params, diff_dst);
    diff_src = GWS_GET_BUFFER_POS_NAMED(SRC, STAT, gws_params, diff_src);

    mean = GWS_GET_BUFFER_POS_NAMED(STAT, STAT, gws_params, mean);
    variance = GWS_GET_BUFFER_POS_NAMED(STAT, STAT, gws_params, variance);

    float mean_val, var_val;
    load(&mean_val, mean);
    load(&var_val, variance);

    float inv_var = rsqrt(var_val + eps);

    // If we calculate mean/variance from src, include this contribution in diff_src
    float mu = 0.0f;
    float sigma = 0.0f;
    if (include_stats) {
        for (int i = 0; i < norm_size; i++) {
            off_t off = i * norm_stride;
            ACC_BWD_DT dst_val;
            load(&dst_val, diff_dst + off);
            ACC_DT src_val;
            load(&src_val, src + off);
            float scale_val = 1.0f;
            if (scale) load(&scale_val, scale + i);

            mu += dst_val * scale_val;
            sigma += dst_val * scale_val * src_val;
        }
        sigma -= mu * mean_val;
        sigma *= inv_var * inv_var;
    }

    // Apply these stats to the entirety of the norm dim
    for (int i = 0; i < norm_size; i++) {
        off_t off = i * norm_stride;
        ACC_BWD_DT dst_val;
        load(&dst_val, diff_dst + off);
        ACC_DT src_val;
        load(&src_val, src + off);
        float scale_val = 1.0f;
        if (scale) load(&scale_val, scale + i);

        float res = dst_val * scale_val;

        if (include_stats) {
            res -= (mu + sigma * (src_val - mean_val)) / norm_size;
        }

        float out = inv_var * res;
        write(diff_src + off, &out);
    }
}
