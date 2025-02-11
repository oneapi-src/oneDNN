/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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
#include "gpu/intel/ocl/ocl_types.h"
#include "gpu/intel/ocl/types_interop.h"

NAMED_KERNEL_ATTR(CALC)
__kernel void reusable_calculate_mean(__global DATA_T *src,
        __global float *mean, off_t reduce_dim_stride, off_t reduce_dim,
        dispatch_gws_rt_params_t gws_params) {
    src = GWS_GET_BUFFER_POS_NAMED(SRC, CALC, gws_params, src);
    mean = GWS_GET_BUFFER_POS_NAMED(DST, CALC, gws_params, mean);
    float sum = 0;
    unroll_16_for(off_t i = 0; i < reduce_dim; i++) {
        sum += load(sum, src + i * (off_t)reduce_dim_stride);
    }

    *mean = sum;
}

NAMED_KERNEL_ATTR(CALC)
__kernel void reusable_calculate_variance(__global DATA_T *src,
        __global float *mean, __global float *variance, off_t reduce_dim_stride,
        off_t reduce_dim, dispatch_gws_rt_params_t gws_params) {
    const off_t c = GWS_GET_OFF_NAMED(IC_DIM, CALC, gws_params);
    src = GWS_GET_BUFFER_POS_NAMED(SRC, CALC, gws_params, src);
    variance = GWS_GET_BUFFER_POS_NAMED(DST, CALC, gws_params, variance);
    float sum = 0;
    unroll_16_for(off_t i = 0; i < reduce_dim; i++) {
        DEF_ACC_DATA_T v0
                = load(v0, src + i * (off_t)reduce_dim_stride) - mean[c];
        sum += v0 * v0;
    }

    *variance = sum;
}

NAMED_KERNEL_ATTR(REDUCE)
__kernel void reusable_reduce_mean(__global float *reduce_temp,
        __global float *mean, off_t ic, off_t reduce_dim, off_t div,
        dispatch_gws_rt_params_t gws_params) {
    reduce_temp
            = GWS_GET_BUFFER_POS_NAMED(BUFFER, REDUCE, gws_params, reduce_temp);
    mean = GWS_GET_BUFFER_POS_NAMED(BUFFER, REDUCE, gws_params, mean);

    float sum = 0.0f;
    unroll_16_for(off_t i = 0; i < reduce_dim; i++) {
        sum += reduce_temp[i * (off_t)ic];
    }

    *mean = sum / div;
}

NAMED_KERNEL_ATTR(REDUCE)
__kernel void reusable_reduce_variance(__global float *reduce_temp,
        __global float *variance, off_t ic, off_t reduce_dim, off_t div,
        dispatch_gws_rt_params_t gws_params) {
    reduce_temp
            = GWS_GET_BUFFER_POS_NAMED(BUFFER, REDUCE, gws_params, reduce_temp);
    variance = GWS_GET_BUFFER_POS_NAMED(BUFFER, REDUCE, gws_params, variance);
    float sum = 0.0f;
    unroll_16_for(off_t i = 0; i < reduce_dim; i++) {
        sum += reduce_temp[i * (off_t)ic];
    }

    *variance = sum / div;
}

KERNEL_ATTR
__kernel void reusable_bnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst, __global float *scale,
        __global float *shift, __global char *ws, float eps,
        __global DATA_T *src_add, float relu_alpha,
        dispatch_gws_rt_params_t gws_params) {
    const off_t c = GWS_GET_OFF(IC_DIM, gws_params);
    src = GWS_GET_BUFFER_POS(BUFFER, gws_params, src);
    src_add = GWS_GET_BUFFER_POS(BUFFER, gws_params, src_add);
    ws = GWS_GET_BUFFER_POS(BUFFER, gws_params, ws);
    dst = GWS_GET_BUFFER_POS(BUFFER, gws_params, dst);

    float sm = USE_SCALE ? scale[c] : 1;
    float sv = USE_SHIFT ? shift[c] : 0;
    float v_mean = mean[c];
    float v_variance = variance[c];
    float v0 = load(v0, src);
    float sqrt_variance = 1.0f / sqrt(v_variance + eps);
    float bn_res = sm * (v0 - v_mean) * sqrt_variance + sv;
    if (FUSE_BN_ADD_RELU) bn_res += load(bn_res, src_add);

#if FUSE_BN_RELU == 1
    if (bn_res <= 0) {
        bn_res = 0;
#if IS_TRAINING == 1
        *ws = 0;
    } else {
        *ws = -1;
#endif
    }
#endif

#if WITH_RELU
#if WITH_LEAKY_RELU
    if (bn_res < 0) bn_res *= relu_alpha;
#else
    bn_res = max(bn_res, 0.0f);
#endif //WITH_LEAKY_RELU
#endif //WITH_RELU

    write(dst, bn_res);
}

NAMED_KERNEL_ATTR(CALC)
__kernel void reusable_calculate_stats(__global DATA_T *src,
        __global float *mean, __global DATA_T *diff_dst, __global char *ws,
        __global float *reduce_temp, __global float *reduce_temp_shift,
        off_t reduce_dim_stride, off_t reduce_dim,
        dispatch_gws_rt_params_t gws_params) {
    float diff_gamma = 0;
    float diff_beta = 0;

    const off_t c = GWS_GET_OFF_NAMED(IC_DIM, CALC, gws_params);
    reduce_temp = GWS_GET_BUFFER_POS_NAMED(DST, CALC, gws_params, reduce_temp);
    reduce_temp_shift = GWS_GET_BUFFER_POS_NAMED(
            DST, CALC, gws_params, reduce_temp_shift);
    diff_dst = GWS_GET_BUFFER_POS_NAMED(SRC, CALC, gws_params, diff_dst);
    ws = GWS_GET_BUFFER_POS_NAMED(SRC, CALC, gws_params, ws);
    src = GWS_GET_BUFFER_POS_NAMED(SRC, CALC, gws_params, src);

    unroll_16_for(off_t i = 0; i < reduce_dim; i++) {
        const off_t offi = i * (off_t)reduce_dim_stride;
        float dd = load(dd, diff_dst + offi);
        if (FUSE_BN_RELU && !ws[offi]) dd = 0;
        diff_gamma += (load(diff_gamma, src + offi) - mean[c]) * dd;
        diff_beta += dd;
    }

    *reduce_temp = diff_gamma;
    *reduce_temp_shift = diff_beta;
}

NAMED_KERNEL_ATTR(REDUCE)
__kernel void reusable_reduce_stats(__global float *reduce_temp,
        __global float *reduce_temp_shift, __global float *diff_scale,
        __global float *diff_shift, __global float *variance, float eps,
        off_t ic, off_t reduce_dim, dispatch_gws_rt_params_t gws_params) {
    reduce_temp
            = GWS_GET_BUFFER_POS_NAMED(BUFFER, REDUCE, gws_params, reduce_temp);
    reduce_temp_shift = GWS_GET_BUFFER_POS_NAMED(
            BUFFER, REDUCE, gws_params, reduce_temp_shift);
    variance = GWS_GET_BUFFER_POS_NAMED(BUFFER, REDUCE, gws_params, variance);
    diff_scale
            = GWS_GET_BUFFER_POS_NAMED(BUFFER, REDUCE, gws_params, diff_scale);
    diff_shift
            = GWS_GET_BUFFER_POS_NAMED(BUFFER, REDUCE, gws_params, diff_shift);
    float diff_gamma = 0.0f;
    float diff_beta = 0.0f;

    unroll_16_for(off_t i = 0; i < reduce_dim; i++) {
        diff_gamma += reduce_temp[i * (off_t)ic];
        diff_beta += reduce_temp_shift[i * (off_t)ic];
    }
    float sqrt_variance = 1.0f / sqrt(*variance + eps);

    *diff_scale = diff_gamma * sqrt_variance;
    *diff_shift = diff_beta;
}

KERNEL_ATTR
__kernel void reusable_bnorm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scale, __global char *ws, __global DATA_T *diff_src,
        __global float *diff_scale, __global float *diff_shift, float eps,
        __global DATA_T *diff_src_add, off_t div,
        dispatch_gws_rt_params_t gws_params) {
    const off_t c = GWS_GET_OFF(IC_DIM, gws_params);
    diff_dst = GWS_GET_BUFFER_POS(BUFFER, gws_params, diff_dst);
    ws = GWS_GET_BUFFER_POS(BUFFER, gws_params, ws);
    diff_src_add = GWS_GET_BUFFER_POS(BUFFER, gws_params, diff_src_add);
    src = GWS_GET_BUFFER_POS(BUFFER, gws_params, src);
    diff_src = GWS_GET_BUFFER_POS(BUFFER, gws_params, diff_src);
    float v_variance = variance[c];
    float sqrt_variance = 1.0f / sqrt(v_variance + eps);
    float gamma = USE_SCALE ? scale[c] : 1;

    float dd = load(dd, diff_dst);
#if FUSE_BN_RELU == 1
    if (!*ws) dd = 0;
    if (FUSE_BN_ADD_RELU) write(diff_src_add, dd);
#endif

    float v_diff_src = dd;
#if CALCULATE_STATS == 1
    float v_mean = mean[c];
    float diff_gamma = diff_scale[c];
    float diff_beta = diff_shift[c];
    v_diff_src -= diff_beta / div
            + (load(v_diff_src, src) - v_mean) * diff_gamma * sqrt_variance
                    / div;
#endif
    v_diff_src *= gamma * sqrt_variance;

    write(diff_src, v_diff_src);
}
