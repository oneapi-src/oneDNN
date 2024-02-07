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
