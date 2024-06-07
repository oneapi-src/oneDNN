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

#define VEC_SUM_DEFINE(TYPE) \
    __attribute__((overloadable)) float vec_sum(TYPE val) { \
        return vec_sum(val.even + val.odd); \
    }

__attribute__((overloadable)) float vec_sum(float val) {
    return val;
}

VEC_SUM_DEFINE(float2)
VEC_SUM_DEFINE(float4)
VEC_SUM_DEFINE(float8)

__attribute__((intel_reqd_sub_group_size(SG_SIZE))) __kernel void
lnorm_reusable_vectorized(__global SRC_DATA_T *src, __global float *mean,
        __global float *variance, dim_t reduce_size, __global DST_DATA_T *dst,
        __global WEI_DATA_T *scale, __global WEI_DATA_T *shift, float eps,
        __global float *src_scale, __global float *dst_scale, int greads,
        float rrs, dispatch_gws_rt_params_t gws_params) {
    src = (GWS_GET_BUFFER_POS(SRC, gws_params, src)) - get_sub_group_local_id();

    FLT_ACC_DATA_T local_variance = 0.f;
    FLT_ACC_DATA_T local_mean = 0.f;
    if (CALCULATE_STATS) {
        /// Read global memory and mean and variance
        FLT_ACC_DATA_T sum = 0;
        unroll_for_by(N_UNROLL)(int sg_idx = 0; sg_idx < reduce_size;
                                sg_idx += SG_STRIDE) {
            VECT_FLOAT_T val
                    = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                            (const __global BLOCK_DATA_T *)(&src[sg_idx]))));
            sum += vec_sum(val);
        }

        local_mean = sub_group_reduce_add(sum) * rrs;
        FLT_ACC_DATA_T sumsq = 0;
        unroll_for_by(N_UNROLL)(int i = 0; i < greads; i++) {
            VECT_FLOAT_T val;
            val = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ((
                          const __global BLOCK_DATA_T *)(&src[i * SG_STRIDE]))))
                    - local_mean;
            val *= val;
            sumsq += vec_sum(val);
        }
        local_variance = sub_group_reduce_add(sumsq) * rrs;
    } else {
        mean = GWS_GET_BUFFER_POS(STAT, gws_params, mean);
        variance = GWS_GET_BUFFER_POS(STAT, gws_params, variance);
        local_mean = *mean;
        local_variance = *variance;
    }

    if (USE_SCALE)
        scale = GWS_GET_BUFFER_POS(SS, gws_params, scale)
                + ((greads - 1) * SG_STRIDE);
    if (USE_SHIFT)
        shift = GWS_GET_BUFFER_POS(SS, gws_params, shift)
                + ((greads - 1) * SG_STRIDE);

    /// Normalize layer
    FLT_ACC_DATA_T sqrt_variance = rsqrt(local_variance + eps);
    __global DST_DATA_T *dst_vect = (GWS_GET_BUFFER_POS(DST, gws_params, dst))
            - get_sub_group_local_id() + ((greads - 1) * SG_STRIDE);

    float src_scale_val = src_scale ? *src_scale : 1.f;
    float dst_scale_val = dst_scale ? native_recip(*dst_scale) : 1.f;

    unroll_for_by(N_UNROLL)(int i = greads - 1; i >= 0; i--) {
        VECT_FLOAT_T res
                = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ((
                          const __global BLOCK_DATA_T *)(&src[i * SG_STRIDE]))))
                - local_mean;
        res *= sqrt_variance;
        if (USE_SCALE) res *= LOAD_VECT_WEI(scale);
        if (USE_SHIFT) res += LOAD_VECT_WEI(shift);

        res *= src_scale_val;
        res *= dst_scale_val;

        VECT_DST_BLOCK_WRITE(dst_vect, CONVERT_VECTOR_DST_DATA_T(res));
        dst_vect -= SG_STRIDE;
        if (USE_SCALE) scale -= SG_STRIDE;
        if (USE_SHIFT) shift -= SG_STRIDE;
    }
    if (SAVE_STATS && get_sub_group_local_id() == 0) {
        mean = GWS_GET_BUFFER_POS(STAT, gws_params, mean);
        variance = GWS_GET_BUFFER_POS(STAT, gws_params, variance);
        *mean = local_mean;
        *variance = local_variance;
    }
}
