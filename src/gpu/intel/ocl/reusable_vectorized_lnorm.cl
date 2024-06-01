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
#include "gpu/intel/ocl/ocl_math_utils.h"
#include "gpu/intel/ocl/ocl_types.h"
#include "gpu/intel/ocl/types_interop.h"

#define AT(x, y) if (get_global_id(0) == x && get_global_id(1) == y)

#define VECT_SUM_DEFINE(TYPE) \
    __attribute__((overloadable)) float vec_sum(TYPE val) { \
        return vec_sum(val.even + val.odd); \
    }

__attribute__((overloadable)) float vec_sum(float val) {
    return val;
}

VECT_SUM_DEFINE(float2)
VECT_SUM_DEFINE(float4)
VECT_SUM_DEFINE(float8)

__attribute__((vec_type_hint(VECT_DATA_T)))
__attribute__((intel_reqd_sub_group_size(SG_SIZE))) __kernel void
lnorm_reusable_vectorized(__global SRC_DATA_T *src,
#if STATS_ARE_TMP == 0
        __global float *mean, __global float *variance,
#endif
        dim_t reduce_size, __global DST_DATA_T *dst, __global WEI_DATA_T *scale,
        __global WEI_DATA_T *shift, float eps, __global float *src_scale,
        __global float *dst_scale, int greads, float rrs,
        dispatch_gws_rt_params_t gws_params) {
    src = (GWS_GET_BUFFER_POS(SRC, gws_params, src)) - get_local_id(0);

    DEF_ACC_DATA_T running_variance = 0.f;
    DEF_ACC_DATA_T running_mean = 0.f;
#if CALCULATE_STATS
    float M2 = 0.f;
    int wt = 0;

    /// Read global memory and mean and variance
    float sum = 0;
#pragma unroll N_UNROLL
    for (int sg_idx = 0; sg_idx < reduce_size; sg_idx += SG_STRIDE) {
        VECT_FLOAT_T val = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                (const __global BLOCK_DATA_T *)(&src[sg_idx]))));
        sum += vec_sum(val);
    }

    running_mean = sub_group_reduce_add(sum) * rrs;
    float sumsq = 0;
#pragma unroll N_UNROLL
    for (int i = 0; i < greads; i++) {
        VECT_FLOAT_T val;
        val = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                      (const __global BLOCK_DATA_T *)(&src[i * SG_STRIDE]))))
                - running_mean;
        val *= val;
        sumsq += vec_sum(val);
    }
    running_variance = sub_group_reduce_add(sumsq) * rrs;

#else // CALCULATE_STATS
    mean = GWS_GET_BUFFER_POS(STAT, gws_params, mean);
    variance = GWS_GET_BUFFER_POS(STAT, gws_params, variance);
    running_mean = *mean;
    running_variance = *variance;
#endif // CALCULATE_STATS

    scale = GWS_GET_BUFFER_POS(SS, gws_params, scale)
            + ((greads - 1) * SG_STRIDE);
    shift = GWS_GET_BUFFER_POS(SS, gws_params, shift)
            + ((greads - 1) * SG_STRIDE);

    /// Normalize layer
    float sqrt_variance = rsqrt(running_variance + eps);
    __global DST_DATA_T *dst_vect = (GWS_GET_BUFFER_POS(DST, gws_params, dst))
            - get_local_id(0) + ((greads - 1) * SG_STRIDE);

#if WITH_SRC_SCALES
    float src_scale_val = *src_scale;
#endif
#if WITH_DST_SCALES
    float dst_scale_val = *dst_scale;
#endif
#pragma unroll N_UNROLL
    for (int i = greads - 1; i >= 0; i--) {
        VECT_FLOAT_T sm = USE_SCALE ? LOAD_VECT_WEI(scale) : 1.0f;
        VECT_FLOAT_T sv = USE_SHIFT ? LOAD_VECT_WEI(shift) : 0.0f;

        VECT_FLOAT_T src_val;
        src_val = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                (const __global BLOCK_DATA_T *)(&src[i * SG_STRIDE]))));
        VECT_FLOAT_T res = sm * (src_val - running_mean) * sqrt_variance + sv;

#if WITH_SRC_SCALES
        res *= src_scale_val;
#endif
#if WITH_DST_SCALES
        res /= dst_scale_val;
#endif

        VECT_DST_BLOCK_WRITE(dst_vect, CONVERT_VECTOR_DST_DATA_T(res));
        dst_vect -= SG_STRIDE;
        scale -= SG_STRIDE;
        shift -= SG_STRIDE;
    }
#if SAVE_STATS
    if (get_local_id(0) == 0) {
        mean = GWS_GET_BUFFER_POS(STAT, gws_params, mean);
        variance = GWS_GET_BUFFER_POS(STAT, gws_params, variance);
        *mean = running_mean;
        *variance = running_variance;
    }
#endif
}
