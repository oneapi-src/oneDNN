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

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG_SIZE))) __kernel void
lnorm_reusable_vectorized(__global SRC_DATA_T *src, __global float *mean,
        __global float *variance, dim_t reduce_size, __global DST_DATA_T *dst,
        __global WEI_DATA_T *scale, __global WEI_DATA_T *shift, float eps,
        __global float *src_scale, __global float *dst_scale, float rrs,
        dispatch_gws_rt_params_t gws_params) {
    int sg_offset
            = -(get_local_id(0)) + get_sub_group_id() * SG_SIZE * VECT_DT_N;
    src = GWS_GET_BUFFER_POS(SRC, gws_params, src) + sg_offset;

    FLT_ACC_DATA_T local_variance = 0.f;
    FLT_ACC_DATA_T local_mean = 0.f;
#if PVT_MEM_SIZE > 1
    VECT_FLOAT_T val[PVT_MEM_SIZE];
#else
    VECT_FLOAT_T val;
#endif
    if (CALCULATE_STATS) {
        /// Read global memory and mean and variance
        VECT_FLOAT_T sum = 0;
#if PVT_MEM_SIZE > 1
        unroll_for_by(N_UNROLL)(int sg_idx = 0, i = 0; i < PVT_MEM_SIZE;
                                sg_idx += GROUP_STRIDE, i++) {
            val[i] = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                    (const __global BLOCK_DATA_T *)(&src[sg_idx]))));
        }
        unroll_for_by(N_UNROLL)(int i = 0; i < PVT_MEM_SIZE; i++) {
            sum += val[i];
        }
#else
        val = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)(src))));
        sum = val;
#endif

        local_mean = GROUP_ADD(vec_sum(sum)) * rrs;

#if PVT_MEM_SIZE > 1
        sum = 0;
        unroll_for_by(N_UNROLL)(int i = 0; i < PVT_MEM_SIZE; i++) {
            VECT_FLOAT_T var_val;
            var_val = val[i] - local_mean;
            var_val *= var_val;
            sum += var_val;
        }
#else
        sum = val - local_mean;
        sum *= sum;
        sum = sum;
#endif
        local_variance = GROUP_ADD(vec_sum(sum)) * rrs;
    } else {
        mean = GWS_GET_BUFFER_POS(STAT, gws_params, mean);
        variance = GWS_GET_BUFFER_POS(STAT, gws_params, variance);
        local_mean = *mean;
        local_variance = *variance;
    }

    if (USE_SCALE)
        scale = GWS_GET_BUFFER_POS(SS, gws_params, scale) + sg_offset;
    if (USE_SHIFT)
        shift = GWS_GET_BUFFER_POS(SS, gws_params, shift) + sg_offset;

    FLT_ACC_DATA_T sqrt_variance = rsqrt(local_variance + eps);
    __global DST_DATA_T *dst_vect
            = (GWS_GET_BUFFER_POS(DST, gws_params, dst)) + sg_offset;

    float src_scale_val = src_scale ? *src_scale : 1.f;
    float dst_scale_val = dst_scale ? native_recip(*dst_scale) : 1.f;

#if PVT_MEM_SIZE > 1
    unroll_for_by(N_UNROLL)(int i = 0; i < PVT_MEM_SIZE; i++) {
#endif
        VECT_FLOAT_T res;
#if PVT_MEM_SIZE > 1
        res = val[i] - local_mean;
#else
    res = val - local_mean;
#endif
        VECT_FLOAT_T sc = (USE_SCALE) ? LOAD_VECT_WEI(scale) : 1.f;
        VECT_FLOAT_T sh = (USE_SHIFT) ? LOAD_VECT_WEI(shift) : 0.f;
        VECT_FLOAT_T out = (sc * res * sqrt_variance + sh) * src_scale_val
                * dst_scale_val;

        VECT_DST_BLOCK_WRITE(dst_vect, CONVERT_VECTOR_DST_DATA_T(out));
#if PVT_MEM_SIZE > 1
        dst_vect += GROUP_STRIDE;
        if (USE_SCALE) scale += GROUP_STRIDE;
        if (USE_SHIFT) shift += GROUP_STRIDE;
    }
#endif
    if (SAVE_STATS && get_local_id(0) == 0) {
        mean = GWS_GET_BUFFER_POS(STAT, gws_params, mean);
        variance = GWS_GET_BUFFER_POS(STAT, gws_params, variance);
        *mean = local_mean;
        *variance = local_variance;
    }
}
